"""Quant News Tracker - Main Application Entry Point.

A quantitative stock tracking dashboard with technical analysis,
news aggregation, and AI-powered insights.
"""

import asyncio
import atexit
import json
import logging
import os
import re
import threading
import time
from datetime import datetime
from io import StringIO
from pathlib import Path

import dash
import diskcache
import pandas as pd
import dash_bootstrap_components as dbc
from dash import Input, Output, State, callback, ctx, html, ALL, MATCH, dcc
from dash import DiskcacheManager
from dash.exceptions import PreventUpdate

# Force 'spawn' start method for background subprocesses.
# macOS MPS (Metal Performance Shaders) segfaults in forked processes
# when torch has been imported in the parent. 'spawn' starts fresh.
import multiprocess as _mp
try:
    _mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass  # already set

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Background callback manager using diskcache
_CACHE_DIR = Path("cache/dash_bg_callbacks")
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_bg_cache = diskcache.Cache(str(_CACHE_DIR))
_bg_manager = DiskcacheManager(_bg_cache)

from callbacks.chart_callbacks import (
    create_comparison_chart,
    create_empty_chart,
    create_macd_chart,
    create_price_chart,
    create_rsi_chart,
    create_volume_chart,
)
from config import APP, COLORS
from layouts.components import (
    calculate_period_label,
    create_metric_card,
    create_overview_empty_state,
    create_symbol_tag,
)
from layouts.main_layout import create_layout
from layouts.formatters import MODEL_DISPLAY, json_report_to_markdown
from layouts.history_sections import (
    applied_filter_chips,
    build_activity_section,
    build_predictions_section,
    filter_history_data,
)
from layouts.nav import SECTION_TITLES
from layouts.pages import activity as activity_page
from layouts.pages import analyze as analyze_page
from layouts.pages import home as home_page
from layouts.pages import performance as performance_page
from layouts.pages import reports as reports_page
from layouts.pages import schedule as schedule_page
from layouts.pages.analyze import (
    build_overall_tab_content,
    build_tab_content,
    create_loading_state,
    restore_tab,
)
from services.analytics import (
    add_indicators_to_df,
    calculate_performance_metrics,
    get_latest_signals,
)
from services.cache_service import get_cache
from services.llm_service import get_llm
from services.news_service import (
    fetch_news,
    fetch_news_cached,
)

# Only the app (and the prediction subprocess it spawns, which inherits the
# env) publishes to the progress feed -- headless tools share the same model
# code and would otherwise interleave into the browser's panel.
from services import progress_service as _progress_service
_progress_service.enable()
# The panel is an audit log, so a new session starts from the stored history
# rather than an empty feed. (Hydration itself happens in _startup(), via the
# ASGI lifespan, so spawn children and reloader re-imports skip the DB read.)

# Initialize Dash app on a FastAPI backend (Dash 4.2+). We construct the
# FastAPI instance ourselves to own the lifespan: startup/shutdown run exactly
# once per server process, which module-level code cannot guarantee (the
# background-callback subprocess re-imports this module via spawn).
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi import Response as HTTPResponse


@asynccontextmanager
async def _lifespan(_fastapi_server):
    _startup()
    yield
    _shutdown()


app = dash.Dash(
    __name__,
    server=FastAPI(lifespan=_lifespan),
    backend="fastapi",
    external_stylesheets=[
        dbc.themes.DARKLY,
        dbc.icons.BOOTSTRAP,
        "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap",
    ],
    suppress_callback_exceptions=True,
    title="QuantNews - Stock Analysis",
    background_callback_manager=_bg_manager,
)

# Set layout
app.layout = create_layout()

server = app.server


# =============================================================================
# CYGNET SSO AUTH (middleware + login/logout/SSO-handoff routes)
# =============================================================================

from fastapi import Form, Request
from fastapi.responses import RedirectResponse

from services import auth_service as _auth


@server.middleware("http")
async def _auth_middleware(request: Request, call_next):
    """Resolve the signed session cookie to a CurrentUser for this request.

    Identity rides a ContextVar, so sync callbacks (threadpool) and
    asyncio.to_thread work inherit it. Anonymous requests proceed normally —
    the app is public by default; auth only attaches ownership.
    """
    user = None
    if request.url.path not in ("/assets", "/_dash-component-suites"):
        try:
            user = await asyncio.to_thread(
                _auth.resolve_cookie, request.cookies.get(_auth.COOKIE_NAME))
        except Exception as e:
            logger.debug(f"auth resolve failed, continuing anonymous: {e}")
    token = _auth.set_current_user(user)
    try:
        return await call_next(request)
    finally:
        _auth._current_user.reset(token)


def _set_session_cookie(resp, raw_token: str, remember: bool) -> None:
    resp.set_cookie(
        _auth.COOKIE_NAME,
        _auth.sign_token(raw_token),
        max_age=_auth.PERSISTENT_COOKIE_MAX_AGE if remember else None,
        httponly=True,
        secure=os.environ.get("COOKIE_SECURE", "true").strip().lower()
        not in ("false", "0", "no"),
        # Lax (not Strict): the SSO handoff arrives as a cross-site top-level
        # navigation from the portal; Strict would drop the fresh cookie on
        # the post-login redirect. Once all apps live under cygnetsystems.us,
        # subdomain hops are same-site anyway.
        samesite="lax",
        # ".cygnetsystems.us" in production — the shared-SSO cookie visible to
        # portal/terminal/quantnews alike. None locally (host-only cookie).
        domain=_auth.COOKIE_DOMAIN,
    )


@server.post("/auth/login")
def _auth_login(request: Request, userid: str = Form(""),
                password: str = Form(""), remember: str = Form("")):
    """Credential login against the shared Cygnet user store."""
    raw_token, err = _auth.attempt_login(
        userid, password,
        ip=request.client.host if request.client else None,
        user_agent=request.headers.get("User-Agent"),
    )
    if err:
        return RedirectResponse(f"/?login_error={err}", status_code=303)
    resp = RedirectResponse("/", status_code=303)
    _set_session_cookie(resp, raw_token, remember=bool(remember))
    return resp


@server.get("/auth/logout")
def _auth_logout():
    user = _auth.current_user()
    if user:
        _auth.revoke_session(user.token_hash)
        _auth._audit("logout_user", user.uid, None)
    _auth.invalidate_caches()
    resp = RedirectResponse("/", status_code=303)
    resp.delete_cookie(_auth.COOKIE_NAME, domain=_auth.COOKIE_DOMAIN)
    return resp


@server.get("/sso/login")
def _sso_login(token: str = "", next: str = "/"):
    """Portal handoff: the SSO portal signs the RAW session token with the
    shared secret (salt 'cygnet-sso-handoff') and redirects here. We validate
    the session in the shared store and set our local cookie."""
    raw = _auth.unsign_token(token, salt=_auth.SSO_HANDOFF_SALT,
                             max_age=_auth.SSO_HANDOFF_MAX_AGE)
    if raw is None or _auth._read_session(raw) is None:
        raise HTTPException(status_code=401, detail="Invalid or expired SSO token")
    dest = next if next.startswith("/") and not next.startswith("//") else "/"
    resp = RedirectResponse(dest, status_code=303)
    _set_session_cookie(resp, raw, remember=True)
    return resp


@server.get("/healthz")
def _healthz():
    """Liveness for the process AND for the schedule it is supposed to keep.

    Returns 503 when the scheduler thread is dead or a job has missed its
    window, so an uptime monitor watching this URL reports the thing that
    actually matters — "did today's analysis happen" — rather than "is the web
    server answering", which stays true while nothing runs.

    Unauthenticated on purpose: a monitor cannot log in, and this exposes
    schedule state only, no market data or user content.
    """
    from fastapi.responses import JSONResponse

    from services import scheduler_service

    try:
        state = scheduler_service.health()
    except Exception as e:
        return JSONResponse({"healthy": False, "error": str(e)[:200]},
                            status_code=503)
    return JSONResponse(state, status_code=200 if state["healthy"] else 503)


@server.get("/auth/whoami")
def _auth_whoami():
    """Tiny identity probe for the portal catalog and for debugging."""
    u = _auth.current_user()
    return {"authenticated": u is not None,
            "uid": u.uid if u else None,
            "name": u.display_name if u else None,
            "role": u.role if u else None}


@callback(
    Output("auth-chip", "children"),
    Output("login-modal", "is_open"),
    Output("login-error-msg", "children"),
    Input("url", "pathname"),
    State("url", "search"),
)
def render_auth_chip(_pathname, search):
    """Per page load: show the signed-in identity or a sign-in button.

    The middleware resolved this request's cookie before the callback ran,
    so auth state is just the ContextVar. A ?login_error= query (set by the
    /auth/login redirect) reopens the modal with the message.
    """
    from urllib.parse import parse_qs, unquote

    err = ""
    if search:
        err = unquote(parse_qs(search.lstrip("?")).get("login_error", [""])[0])

    user = _auth.current_user()
    if user:
        chip = html.Div([
            html.I(className="bi bi-person-check me-1"),
            html.Span(user.display_name, className="me-2"),
            html.A("Sign out", href="/auth/logout",
                   className="auth-signout-link"),
        ], className="auth-chip-signed-in")
        return chip, False, ""

    chip = html.Button(
        [html.I(className="bi bi-person me-1"), "Sign in"],
        id="login-open-btn", n_clicks=0,
        className="btn btn-sm btn-outline-secondary",
    )
    return chip, bool(err), err


@callback(
    Output("login-modal", "is_open", allow_duplicate=True),
    # Rendered dynamically inside auth-chip — allow_optional so Dash 4's
    # renderer tolerates its absence while signed in.
    Input("login-open-btn", "n_clicks", allow_optional=True),
    prevent_initial_call=True,
)
def open_login_modal(n_clicks):
    if not n_clicks:
        raise PreventUpdate
    return True


# =============================================================================
# SERVER-SIDE DOWNLOAD ROUTES (proper Content-Disposition headers)
# =============================================================================

# Plain `def` endpoints: Starlette runs sync routes in its threadpool, so the
# blocking PDF/S3/XLSX work below never occupies the event loop.


@server.get("/api/download/ta-report/{report_id}")
def serve_ta_report(report_id: str):
    """Serve a TradingAgents report as PDF via proper HTTP download."""
    try:
        cache = get_cache()
        ta_reports = cache.get_all_trading_agent_reports(limit=500)
        report = next((r for r in ta_reports if str(r.get("id")) == report_id), None)
    except Exception as e:
        logger.warning(f"TA report download failed for id={report_id}: {e}")
        raise HTTPException(status_code=500)
    if report is None:
        raise HTTPException(status_code=404)

    symbol = report.get("symbol", "UNKNOWN")
    trade_date = report.get("trade_date", "")

    # Model predictions for this symbol — prefer the report's trade date,
    # fall back to the most recent prediction date available.
    predictions = []
    try:
        all_preds = cache.list_all_predictions(symbol=symbol, limit=50)
        predictions = [p for p in all_preds if p.get("prediction_date") == trade_date]
        if not predictions and all_preds:
            latest_date = all_preds[0].get("prediction_date")
            predictions = [p for p in all_preds if p.get("prediction_date") == latest_date]
    except Exception as e:
        logger.debug(f"Could not load predictions for TA PDF ({symbol}): {e}")

    # Latest recommendation run (Luna reasoning) that covers this symbol
    recommendation = None
    try:
        for run in cache.list_recommendation_runs(limit=50):
            result = run.get("result_json") or {}
            sym_rec = (result.get("by_symbol") or {}).get(symbol)
            if sym_rec:
                recommendation = {
                    "model_used": run.get("model_used", ""),
                    "created_at": run.get("created_at", ""),
                    "symbol_rec": sym_rec,
                    "overall": result.get("overall") or {},
                }
                break
    except Exception as e:
        logger.debug(f"Could not load recommendation for TA PDF ({symbol}): {e}")

    try:
        from services.report_service import generate_ta_report_pdf
        pdf_bytes = generate_ta_report_pdf(report, predictions=predictions, recommendation=recommendation)
    except Exception as e:
        logger.warning(f"TA report PDF generation failed for id={report_id}: {e}")
        pdf_bytes = None

    if pdf_bytes:
        return HTTPResponse(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{symbol}_trading_agents_{trade_date}.pdf"'},
        )

    # Fallback: Markdown if PDF rendering fails
    decision = report.get("decision", "HOLD")
    conf_pct = int((report.get("confidence") or 0) * 100)
    content = f"# {symbol} — {decision} ({conf_pct}%) — {trade_date}\n\n{report.get('report_text', '')}"
    return HTTPResponse(
        content=content,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{symbol}_trading_agents_{trade_date}.md"'},
    )


@server.get("/api/download/saved-report")
def serve_saved_report(key: str = ""):
    """Serve a saved report (AI report JSON → Markdown, or PDF) via HTTP download."""
    storage_key = key
    if not storage_key:
        raise HTTPException(status_code=400)
    try:
        from services.storage_service import download_report
        content = download_report(storage_key)
    except Exception as e:
        logger.warning(f"Saved report download failed for {storage_key}: {e}")
        raise HTTPException(status_code=500)
    if not content:
        raise HTTPException(status_code=404)

    parts = storage_key.split("/")
    if len(parts) >= 4:
        sym, dt, base = parts[1], parts[2], parts[3]
        rtype, ext = base.rsplit(".", 1) if "." in base else (base, "")
        fname = f"{sym}_{rtype}_{dt}.{ext}"
    else:
        fname = storage_key.rsplit("/", 1)[-1] if "/" in storage_key else storage_key
        ext = fname.rsplit(".", 1)[-1].lower() if "." in fname else ""

    if ext == "json":
        try:
            parsed = json.loads(content)
            content = json_report_to_markdown(parsed).encode("utf-8")
            fname = fname.replace(".json", ".md")
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass

    return HTTPResponse(
        content=content,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{fname}"'},
    )


@server.get("/api/download/report-inputs")
def serve_report_inputs(symbols: str = "", date: str = ""):
    """Serve the point-in-time model-input workbook for a report/prediction.

    Reconstructs inputs with the same lookahead-safe builders the models use.
    No LLM is involved — downloads must never incur model cost.
    """
    as_of = date[:10]
    syms = [s.strip().upper() for s in symbols.split(",") if s.strip()][:12]
    if not syms or not as_of:
        raise HTTPException(status_code=400)
    try:
        from services.export_service import build_model_inputs_xlsx
        payload = build_model_inputs_xlsx(syms, as_of)
    except Exception as e:
        logger.warning(f"Input-data export failed for {syms} @ {as_of}: {e}")
        raise HTTPException(status_code=500)
    try:
        from services import progress_service as prog
        prog.emit("action", f"Input-data export: {', '.join(syms)} @ {as_of}")
    except Exception:
        pass
    fname = (f"inputs_{'_'.join(syms[:4])}"
             f"{'_etc' if len(syms) > 4 else ''}_{as_of}.xlsx")
    return HTTPResponse(
        content=payload,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="{fname}"'},
    )


# =============================================================================
# CLEANUP ON EXIT
# =============================================================================


def cleanup_on_exit():
    """Cleanup resources on application exit."""
    try:
        cache = get_cache()
        cache.close()
    except Exception:
        pass


atexit.register(cleanup_on_exit)


# =============================================================================
# OBJECT STORAGE (S3) — gracefully optional
# =============================================================================

_s3_available = False

def _init_s3():
    """Try to connect to S3 for report storage. If unavailable, skip report caching."""
    global _s3_available
    try:
        from services.storage_service import ensure_bucket
        ensure_bucket()
        _s3_available = True
        logger.info("S3 object storage ready")
    except Exception as e:
        _s3_available = False
        logger.info(f"S3 unavailable, report caching disabled: {e}")

# _init_s3() runs in _startup() via the ASGI lifespan — not at import, so the
# background-callback spawn child doesn't repeat the bucket check.


# =============================================================================
# ROUTING
# =============================================================================

# Section content is built on navigation rather than all at once. The win is
# not just a smaller DOM: a callback whose Output is unmounted does not fire,
# so leaving /analyze silently stops its chart, news and AI callbacks. Toggling
# hidden divs instead would keep every callback live on every page.
def _build_route(path, history_data, filter_symbols, filter_range,
                 filter_specific, activity_scope, activity_since,
                 outcome="all"):
    """Build one section. Each page reads only the state it renders."""
    if path == "/analyze":
        return analyze_page.layout()
    if path in ("/performance", "/reports"):
        # Live read, not the store: see fetch_report_history. Each page asks
        # only for the buckets it renders.
        if path == "/performance":
            return performance_page.layout(
                history_data or fetch_report_history(only={"predictions"}),
                filter_symbols, filter_range, filter_specific, outcome)
        return reports_page.layout(
            history_data or fetch_report_history(
                only={"reports", "recommendations", "trading_agent_reports"}),
            filter_symbols, filter_range, filter_specific)
    if path == "/schedule":
        return schedule_page.layout()
    if path == "/activity":
        from services import watchlist_service
        return activity_page.layout(
            activity_scope=activity_scope,
            searches=watchlist_service.search_history(limit=100),
            since_days=activity_since,
        )

    from services import dashboard_service as ds
    jobs = []
    try:
        from services import scheduler_service
        jobs = scheduler_service.list_jobs()
    except Exception as e:
        logger.warning("Could not read scheduled jobs for Home: %s", e)
    return home_page.layout(
        cohort=ds.get_latest_cohort(),
        open_preds=ds.get_open_predictions(),
        rolling=ds.get_rolling_performance(days=HOME_ROLLING_DAYS),
        last_run=ds.get_last_run(),
        jobs=jobs,
        rolling_days=HOME_ROLLING_DAYS,
    )


HOME_ROLLING_DAYS = 30

_ROUTES = ["/", "/analyze", "/performance", "/reports", "/schedule", "/activity"]


@callback(
    Output("page-content", "children"),
    Output("topbar-title", "children"),
    Input("url", "pathname"),
    State("report-history-store", "data"),
    State("history-filter-symbols", "data"),
    State("history-filter-date-range", "data"),
    State("history-filter-date-specific", "data"),
    State("history-activity-scope", "data"),
    State("activity-since-days", "data"),
    State("history-filter-outcome", "data"),
)
def render_page(pathname, history_data, filter_symbols, filter_range,
                filter_specific, activity_scope, activity_since, outcome):
    """Mount the section for this URL.

    The URL is the only Input on purpose. When the archive stores were Inputs
    too, a store-triggered invocation could carry a stale pathname and land
    after the navigation that superseded it, leaving one section's content
    under another section's title. Filter changes now rebuild #archive-body
    instead (render_archive_body), which cannot outrank a route change.

    Unknown paths fall back to Home rather than erroring, so a stale bookmark
    from the pre-routing single page still lands somewhere useful.
    """
    path = (pathname or "/").rstrip("/") or "/"
    if path not in _ROUTES:
        logger.info("Unknown route %s, falling back to Home", path)
        path = "/"
    try:
        page = _build_route(path, history_data, filter_symbols, filter_range,
                            filter_specific, activity_scope, activity_since,
                            outcome)
        return page, SECTION_TITLES.get(path, "QuantNews")
    except Exception as e:
        logger.exception("Failed to render %s", path)
        return (
            html.Div(
                [
                    html.Div("This section failed to load.",
                             className="empty-state-title"),
                    html.Div(str(e), className="empty-state-note"),
                ],
                className="empty-state",
            ),
            SECTION_TITLES.get(path, "QuantNews"),
        )


@callback(
    Output("predictions-log-body", "children"),
    Input({"type": "history-section-toggle", "section": "predictions"}, "n_clicks"),
    State("history-filter-symbols", "data"),
    State("history-filter-date-range", "data"),
    State("history-filter-date-specific", "data"),
    prevent_initial_call=True,
)
def render_prediction_log(n_clicks, filter_symbols, filter_range, filter_specific):
    """Build the prediction log the first time its section is opened."""
    if not n_clicks:
        raise PreventUpdate
    buckets = filter_history_data(fetch_report_history(), filter_symbols,
                                  filter_range, filter_specific)
    section = build_predictions_section(buckets["predictions"])
    if section is None:
        return html.Div("No predictions match this filter.",
                        className="history-empty-msg")
    # Unwrap: the outer collapsible already exists on the page.
    return section.children[1].children


@callback(
    Output("history-applied-filters", "children"),
    Input("history-filter-symbols", "data"),
    Input("history-filter-date-range", "data"),
    Input("history-filter-date-specific", "data"),
    prevent_initial_call=True,
)
def render_filter_chips(filter_symbols, filter_range, filter_specific):
    """Refresh the applied-filter chips.

    Separate from the filter bar so the bar is never rebuilt: re-creating the
    dropdown re-fires its value Input, which writes this store, which would
    rebuild the bar again. Chips contain only buttons, which are guarded
    against firing on insertion, so this is safe to re-render.
    """
    return applied_filter_chips(filter_symbols, filter_range, filter_specific)


@callback(
    Output("archive-body", "children"),
    Input("report-history-store", "data"),
    Input("history-filter-symbols", "data"),
    Input("history-filter-date-range", "data"),
    Input("history-filter-date-specific", "data"),
    Input("history-filter-outcome", "data"),
    State("url", "pathname"),
    prevent_initial_call=True,
)
def render_archive_body(history_data, filter_symbols, filter_range,
                        filter_specific, outcome, pathname):
    """Rebuild the filterable part of Performance or Reports.

    #archive-body exists only on those two routes, and only their own controls
    write these stores, so this never fires with the Output unmounted.
    """
    path = (pathname or "/").rstrip("/") or "/"
    if path not in ("/performance", "/reports"):
        raise PreventUpdate
    data = history_data or fetch_report_history()
    if path == "/performance":
        return performance_page.body(data, filter_symbols, filter_range,
                                     filter_specific, outcome)
    return reports_page.body(data, filter_symbols, filter_range, filter_specific)


@callback(
    Output("history-filter-outcome", "data"),
    Input({"type": "perf-outcome-chip", "outcome": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def set_outcome_filter(clicks):
    """Slice the prediction log by how the calls turned out.

    The counts in the eval bar are the buttons: reading "31 wrong" and then
    hunting for a control that reproduces that set is the long way round.
    """
    if not clicks or not any(c for c in clicks if c):
        raise PreventUpdate
    return ctx.triggered_id["outcome"]


@callback(
    Output("watchlist-panel", "is_open"),
    Input("watchlist-toggle-btn", "n_clicks"),
    State("watchlist-panel", "is_open"),
    prevent_initial_call=True,
)
def toggle_watchlist_panel(n_clicks, is_open):
    """Reveal the symbol editor under the toolbar."""
    return not is_open if n_clicks else is_open


# =============================================================================
# ACTIVITY PAGE
# =============================================================================


@callback(
    Output("activity-since-days", "data"),
    Input({"type": "activity-since-btn", "days": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def set_activity_since(clicks):
    """Time window for the run log. 0 means no lower bound."""
    if not any(c for c in (clicks or []) if c):
        raise PreventUpdate
    return ctx.triggered_id.get("days", 0)


@callback(
    Output("activity-runs", "children"),
    Input("activity-symbol-filter", "value", allow_optional=True),
    Input("activity-stage-filter", "value", allow_optional=True),
    Input("activity-since-days", "data"),
    Input("history-activity-scope", "data"),
)
def render_activity_runs(symbol, stages, since_days, scope):
    """Re-query the audit trail whenever a filter moves.

    Filtering happens in SQL rather than over an already-rendered list, so a
    narrow filter reads a few rows instead of fifty runs' worth of events.
    """
    return build_activity_section(
        activity_scope=scope or "all",
        stages=stages or None,
        symbol=(symbol or "").strip() or None,
        since_days=since_days or None,
    )


@callback(
    Output("selected-symbols", "data", allow_duplicate=True),
    Input({"type": "search-restore", "csv": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def restore_past_search(clicks):
    """Reload a watchlist from the durable search history."""
    if not any(c for c in (clicks or []) if c):
        raise PreventUpdate
    csv = (ctx.triggered_id or {}).get("csv", "")
    symbols = [s for s in csv.split(",") if s]
    if not symbols:
        raise PreventUpdate
    from services import progress_service as prog
    prog.emit("action", f"Restored past search: {', '.join(symbols)}")
    return symbols


# =============================================================================
# SYMBOL MANAGEMENT CALLBACKS
# =============================================================================


@callback(
    Output("selected-symbols", "data"),
    Output("symbol-input", "value"),
    Input("add-symbol-btn", "n_clicks"),
    Input("symbol-input", "n_submit"),
    Input("clear-symbols-btn", "n_clicks"),
    Input({"type": "recent-group", "idx": ALL}, "n_clicks"),
    Input({"type": "remove-symbol", "symbol": ALL}, "n_clicks"),
    State("symbol-input", "value"),
    State("selected-symbols", "data"),
    State("recent-symbol-groups", "data"),
    prevent_initial_call=True,
)
def manage_symbols(add_click, input_submit, clear_click, group_clicks,
                   remove_clicks, input_value, current_symbols, recent_groups):
    """Handle adding, removing, clearing, and restoring symbol selections."""
    current_symbols = current_symbols or []

    # Get the triggered context safely
    if not ctx.triggered:
        raise PreventUpdate

    triggered = ctx.triggered_id

    # Guard against None triggered_id
    if triggered is None:
        raise PreventUpdate

    # Handle add button — or Enter in the input, which previously did nothing
    if triggered in ("add-symbol-btn", "symbol-input"):
        if not (add_click or input_submit) or not input_value:
            raise PreventUpdate
        # Accept comma- AND space-separated input ("AAPL, MSFT" / "AAPL MSFT")
        # — the industry-standard multi-ticker search convention.
        new_symbols = [s.strip().upper()
                       for s in re.split(r"[,\s]+", input_value) if s.strip()]
        added = [sym for sym in new_symbols
                 if sym and sym not in current_symbols]
        current_symbols.extend(added)
        if added:
            from services import progress_service as prog
            prog.emit("action", f"Symbol search: added {', '.join(added)}"
                                f" (selection: {', '.join(current_symbols)})")
        return current_symbols, ""

    # Clear the whole selection in one click
    if triggered == "clear-symbols-btn":
        if not clear_click:
            raise PreventUpdate
        from services import progress_service as prog
        prog.emit("action", f"Symbols cleared ({', '.join(current_symbols) or 'none'})")
        return [], dash.no_update

    # Restore a recent group — REPLACES the selection (that's the point:
    # one click back to a previous watchlist, not a merge)
    if isinstance(triggered, dict) and triggered.get("type") == "recent-group":
        if not any(c and c > 0 for c in group_clicks):
            raise PreventUpdate
        idx = triggered.get("idx")
        groups = recent_groups or []
        if not isinstance(idx, int) or idx >= len(groups):
            raise PreventUpdate
        from services import progress_service as prog
        prog.emit("action", f"Recent group restored: {', '.join(groups[idx])}")
        return list(groups[idx]), dash.no_update

    # Handle remove buttons
    if isinstance(triggered, dict) and triggered.get("type") == "remove-symbol":
        # Find which button was clicked by checking n_clicks values
        clicked_any = any(c and c > 0 for c in remove_clicks)
        if not clicked_any:
            raise PreventUpdate
        symbol = triggered["symbol"]
        if symbol in current_symbols:
            current_symbols = [s for s in current_symbols if s != symbol]
            from services import progress_service as prog
            prog.emit("action", f"Symbol removed: {symbol}")
        return current_symbols, dash.no_update

    raise PreventUpdate


@callback(
    Output("recent-symbol-groups", "data"),
    Input("stock-data-store", "data"),
    State("selected-symbols", "data"),
    State("recent-symbol-groups", "data"),
    prevent_initial_call=True,
)
def record_recent_group(stock_data, symbols, groups):
    """Record the symbol group in durable history and refresh the chips.

    Triggered by the data store (not the raw selection) so only symbols that
    actually returned data are recorded: a typo'd ticker with zero results
    never pollutes Recent.

    The write goes to watchlist_history, which keeps every distinct group.
    The store this returns is now a display cache of the top few rather than
    the record itself, so clearing browser storage no longer destroys history.
    Collapsing (REX, then REX+WGO, then REX+WGO+IOVA leaving one chip) happens
    at read time in watchlist_service.recent_groups.
    """
    symbols = [s for s in (symbols or [])
               if s and ((stock_data or {}).get(s) or {}).get("prices")]
    if not symbols:
        raise PreventUpdate

    from services import watchlist_service
    new = sorted(symbols)
    watchlist_service.record_group(new)

    fresh = watchlist_service.recent_groups(limit=5)
    if fresh:
        if fresh == [list(g) for g in (groups or [])]:
            raise PreventUpdate
        return fresh

    # History unavailable (DB down): fall back to the previous in-store merge
    # so the chips keep working for the rest of the session.
    groups = [list(g) for g in (groups or [])]
    if any(set(new) <= set(g) for g in groups):
        raise PreventUpdate
    groups = [g for g in groups if not set(g) < set(new)]
    return [new] + groups[:4]


@callback(
    Output("recent-symbol-groups", "data", allow_duplicate=True),
    Input("url", "pathname"),
    prevent_initial_call="initial_duplicate",
)
def hydrate_recent_groups(_pathname):
    """Seed the chips from durable history on every page load.

    The store is browser-local, so without this a fresh browser (or a cleared
    cache) starts with no recent groups even though the account has years of
    them. It also has to write the store, not just the chips: the chip click
    handler restores by index into this same list.
    """
    from services import watchlist_service
    groups = watchlist_service.recent_groups(limit=5)
    if not groups:
        raise PreventUpdate
    return groups


@callback(
    Output("recent-groups", "children"),
    Input("recent-symbol-groups", "data"),
)
def render_recent_groups(groups):
    """Render recent symbol groups as one-click restore chips."""
    groups = groups or []
    if not groups:
        return html.Span("none yet — add symbols to build history",
                         className="recent-groups-empty")
    chips = []
    for i, group in enumerate(groups[:5]):
        label = ", ".join(group[:3]) + (f" +{len(group) - 3}" if len(group) > 3 else "")
        chips.append(dbc.Button(
            label,
            id={"type": "recent-group", "idx": i},
            size="sm",
            outline=True,
            color="secondary",
            className="quick-add-btn me-1 mb-1",
            title=", ".join(group),
        ))
    return chips


@callback(
    Output("symbol-tags", "children"),
    Input("selected-symbols", "data"),
)
def update_symbol_tags(symbols):
    """Update the symbol tags display."""
    if not symbols:
        return html.Div(
            [
                html.I(className="bi bi-info-circle me-2"),
                html.Span("Add stocks using the input above or a recent group"),
            ],
            className="text-muted",
            style={"fontSize": "12px", "padding": "8px 0"},
        )

    return [create_symbol_tag(sym) for sym in symbols]


@callback(
    Output("clear-symbols-btn", "style"),
    Input("selected-symbols", "data"),
)
def toggle_clear_button(symbols):
    """Clear-all only makes sense when there is something to clear."""
    return {} if symbols else {"display": "none"}


# =============================================================================
# DATA FETCHING CALLBACKS
# =============================================================================


# UI period -> {daily fetch period, display bars (incl. one prior close so
# the window return has a baseline), intraday spec for the price chart}.
# Sub-6mo windows fetch a 6-month daily floor; 1D/3D/1W additionally render
# intraday bars on the price chart only — intraday NEVER enters the shared
# store, so models and metric blocks always consume daily data.
_PERIOD_CONFIG = {
    "1d":  {"fetch": "6mo", "bars": 2,    "intraday": ("1d", "5m", 1)},
    "3d":  {"fetch": "6mo", "bars": 4,    "intraday": ("5d", "30m", 3)},
    "1wk": {"fetch": "6mo", "bars": 6,    "intraday": ("5d", "30m", 5)},
    "2wk": {"fetch": "6mo", "bars": 11,   "intraday": None},
    "3wk": {"fetch": "6mo", "bars": 16,   "intraday": None},
    "1mo": {"fetch": "6mo", "bars": 22,   "intraday": None},
    "3mo": {"fetch": "6mo", "bars": 64,   "intraday": None},
    "6mo": {"fetch": "6mo", "bars": None, "intraday": None},
    "ytd": {"fetch": "ytd", "bars": None, "intraday": None},
    "1y":  {"fetch": "1y",  "bars": None, "intraday": None},
    "2y":  {"fetch": "2y",  "bars": None, "intraday": None},
    "5y":  {"fetch": "5y",  "bars": None, "intraday": None},
}


@callback(
    Output("stock-data-store", "data"),
    Output("cache-status", "children"),
    Output("data-source-indicator", "children"),
    Input("selected-symbols", "data"),
    Input("current-period", "data"),
    Input("refresh-data-btn", "n_clicks"),
    Input("cache-enabled", "data"),
    # Fires on load so a refresh rehydrates charts from the restored symbol
    # list. Returns the empty state when there are no symbols, so a fresh
    # session behaves exactly as before.
    prevent_initial_call=False,
)
async def fetch_stock_data_callback(symbols, period, refresh_click, cache_enabled):
    """Fetch stock data for selected symbols.

    UI period is a DISPLAY window: sub-6-month windows keep a 6-month daily
    floor in the store (models, signals, and SMA-200 always see enough
    history — a bare 1-month fetch used to starve them), while metrics are
    computed on the sliced window and charts slice/render it themselves.

    Async: per-symbol fetches were sequential (~Nx wall time for N symbols);
    they now run concurrently in threads, capped so yfinance isn't hammered.
    """
    if not symbols:
        return {}, "No data", ""

    cache = get_cache()
    triggered = ctx.triggered_id
    # Force refresh if button clicked OR if cache is disabled
    force_refresh = triggered == "refresh-data-btn" or not cache_enabled
    if triggered == "refresh-data-btn":
        from services import progress_service as prog
        prog.emit("action", f"Data refresh forced ({len(symbols)} symbols, "
                            f"period {period})")

    cfg = _PERIOD_CONFIG.get(period) or {
        "fetch": period or "1y", "bars": None, "intraday": None,
    }

    def _fetch_one(symbol):
        """Blocking per-symbol fetch — runs in a worker thread."""
        df, metadata = cache.get_stock_prices(
            symbol, cfg["fetch"], force_refresh=force_refresh)
        # Add technical indicators
        df = add_indicators_to_df(df)
        # Metrics reflect the SELECTED window (bars includes one prior
        # close so the window return has a baseline); prices/signals keep
        # the full daily history.
        display_df = df.tail(cfg["bars"]) if cfg["bars"] else df
        entry = {
            "prices": df.to_json(date_format="iso"),
            "metrics": calculate_performance_metrics(display_df),
            "signals": get_latest_signals(df),
            "period": period,
            "display_bars": cfg["bars"],
            "from_cache": metadata.get("from_cache", False),
            "api_error": metadata.get("api_error"),
        }
        return entry, metadata

    sem = asyncio.Semaphore(APP.STOCK_FETCH_CONCURRENCY)

    async def _fetch_guarded(symbol):
        async with sem:
            return await asyncio.to_thread(_fetch_one, symbol)

    results = await asyncio.gather(
        *(_fetch_guarded(s) for s in symbols), return_exceptions=True)

    data = {}
    all_from_cache = True
    any_api_error = None
    for symbol, result in zip(symbols, results):
        if isinstance(result, BaseException):
            logger.warning(f"Error fetching {symbol}: {result}")
            continue
        entry, metadata = result
        if not metadata.get("from_cache"):
            all_from_cache = False
        if metadata.get("api_error"):
            any_api_error = metadata["api_error"]
        data[symbol] = entry

    # Update cache status (quick Postgres read — still off the event loop)
    cache_status = await asyncio.to_thread(cache.get_cache_status)
    if cache_status:
        last = cache_status[-1]
        status_text = f"Last update: {last['last_updated'].strftime('%H:%M')}"
    else:
        status_text = "No cached data"

    # Create data source indicator
    if any_api_error:
        source_indicator = html.Span(
            [
                html.I(className="bi bi-exclamation-triangle-fill me-1"),
                "API Error - Using cached data",
            ],
            className="data-source-badge data-source-error",
            title=f"API Error: {any_api_error}",
        )
    elif all_from_cache and data:
        source_indicator = html.Span(
            [
                html.I(className="bi bi-database me-1"),
                "Cached",
            ],
            className="data-source-badge data-source-cache",
        )
    elif data:
        source_indicator = html.Span(
            [
                html.I(className="bi bi-cloud-download me-1"),
                "Live",
            ],
            className="data-source-badge data-source-live",
        )
    else:
        source_indicator = ""

    return data, status_text, source_indicator


# =============================================================================
# CHART UPDATE CALLBACKS
# =============================================================================


@callback(
    Output("price-chart", "figure"),
    Output("chart-title", "children"),
    Output("chart-subtitle", "children"),
    Input("stock-data-store", "data"),
    # Analyze page only.
    Input("indicator-toggles", "value", allow_optional=True),
    State("selected-symbols", "data"),
)
def update_price_chart(stock_data, indicators, symbols):
    """Update the main price chart."""
    if ctx.triggered_id == "indicator-toggles":
        from services import progress_service as prog
        prog.emit("action", f"Indicators -> {', '.join(indicators or []) or 'none'}")
    if not stock_data or not symbols:
        return create_empty_chart("Select stocks to view price chart"), "Price Chart", "Add stocks to get started"

    # Use first symbol for main chart, or comparison for multiple
    if len(symbols) == 1:
        symbol = symbols[0]
        if symbol not in stock_data:
            return create_empty_chart(f"No data for {symbol}"), symbol, ""

        entry = stock_data[symbol]

        # Intraday windows (1D/3D/1W): render intraday bars on the price
        # chart only. Daily-period SMA overlays are omitted — drawing a
        # daily SMA-50 across 5-minute candles is a different quantity than
        # the label claims. Falls back to sliced daily bars on any failure.
        cfg = _PERIOD_CONFIG.get(entry.get("period")) or {}
        if cfg.get("intraday"):
            yf_period, interval, sessions = cfg["intraday"]
            try:
                from services.stock_data import fetch_intraday
                idf = fetch_intraday(symbol, yf_period, interval, sessions=sessions)
                if idf is not None and not idf.empty:
                    fig = create_price_chart(idf, symbol, [])
                    subtitle = (f"intraday {interval} bars · "
                                f"{sessions} session{'s' if sessions > 1 else ''} — "
                                "indicator overlays are daily-only")
                    return fig, symbol, subtitle
            except Exception as e:
                logger.warning(f"Intraday fetch failed for {symbol}: {e}")

        df = pd.read_json(StringIO(entry["prices"]))
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        bars = entry.get("display_bars")
        if bars:
            df = df.tail(bars)

        metrics = entry.get("metrics", {})
        subtitle = f"{metrics.get('start_date', '')} to {metrics.get('end_date', '')}"

        fig = create_price_chart(df, symbol, indicators)
        return fig, symbol, subtitle
    else:
        # Multiple stocks - create comparison chart (daily bars; sliced to
        # the selected display window)
        data_dict = {}
        for symbol in symbols:
            if symbol in stock_data:
                entry = stock_data[symbol]
                df = pd.read_json(StringIO(entry["prices"]))
                df.index = pd.to_datetime(df.index)
                df = df.sort_index()
                bars = entry.get("display_bars")
                if bars:
                    df = df.tail(bars)
                data_dict[symbol] = df

        fig = create_comparison_chart(data_dict)
        return fig, "Comparison", f"{len(symbols)} stocks"


@callback(
    Output("macd-chart", "figure"),
    Input("stock-data-store", "data"),
    State("selected-symbols", "data"),
)
def update_macd_chart(stock_data, symbols):
    """Update MACD chart."""
    if not stock_data or not symbols:
        return create_empty_chart("Select stocks to view MACD")

    symbol = symbols[0]
    if symbol not in stock_data:
        return create_empty_chart("No MACD data available")

    try:
        entry = stock_data[symbol]
        df = pd.read_json(StringIO(entry["prices"]))
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        # Momentum needs context: floor short display windows at ~1 month of
        # daily bars — a 2-point MACD line reads as noise, not momentum.
        bars = entry.get("display_bars")
        if bars:
            df = df.tail(max(bars, 22))
        return create_macd_chart(df)
    except Exception as e:
        logger.warning(f"Error creating MACD chart: {e}")
        return create_empty_chart("Error loading MACD data")


@callback(
    Output("rsi-chart", "figure"),
    Input("stock-data-store", "data"),
    State("selected-symbols", "data"),
)
def update_rsi_chart(stock_data, symbols):
    """Update RSI chart."""
    if not stock_data or not symbols:
        return create_empty_chart("Select stocks to view RSI")

    symbol = symbols[0]
    if symbol not in stock_data:
        return create_empty_chart("No RSI data available")

    try:
        entry = stock_data[symbol]
        df = pd.read_json(StringIO(entry["prices"]))
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        bars = entry.get("display_bars")
        if bars:
            df = df.tail(max(bars, 22))
        return create_rsi_chart(df)
    except Exception as e:
        logger.warning(f"Error creating RSI chart: {e}")
        return create_empty_chart("Error loading RSI data")


@callback(
    Output("volume-chart", "figure"),
    Input("stock-data-store", "data"),
    State("selected-symbols", "data"),
)
def update_volume_chart(stock_data, symbols):
    """Update volume chart."""
    if not stock_data or not symbols:
        return create_empty_chart("Select stocks to view volume")

    symbol = symbols[0]
    if symbol not in stock_data:
        return create_empty_chart("No volume data available")

    try:
        entry = stock_data[symbol]
        df = pd.read_json(StringIO(entry["prices"]))
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        bars = entry.get("display_bars")
        if bars:
            df = df.tail(bars)
        return create_volume_chart(df)
    except Exception as e:
        logger.warning(f"Error creating volume chart: {e}")
        return create_empty_chart("Error loading volume data")


# =============================================================================
# SUMMARY CARDS CALLBACK
# =============================================================================


@callback(
    Output("summary-cards", "children"),
    Input("stock-data-store", "data"),
    State("selected-symbols", "data"),
)
def update_summary_cards(stock_data, symbols):
    """Update summary metric cards."""
    if not stock_data or not symbols:
        return html.Div(
            [
                html.Div(
                    [
                        html.I(className="bi bi-graph-up", style={"fontSize": "32px", "opacity": "0.5"}),
                        html.P("Select stocks to view metrics", className="mt-2 mb-0"),
                    ],
                    className="empty-state",
                )
            ],
            style={"gridColumn": "1 / -1"},
        )

    cards = []
    for symbol in symbols:
        if symbol not in stock_data:
            continue

        metrics = stock_data[symbol].get("metrics", {})
        signals = stock_data[symbol].get("signals", {})

        # Current price card with dynamic period label
        price = metrics.get("end_price", 0)
        total_return = metrics.get("total_return", 0)
        start_date = metrics.get("start_date", "")
        end_date = metrics.get("end_date", "")

        # Calculate dynamic period label
        period_label = calculate_period_label(start_date, end_date)

        cards.append(
            create_metric_card(
                title=symbol,
                value=f"${price:,.2f}",
                change=f"{total_return:+.2f}%",
                change_positive=total_return > 0,
                subtitle=period_label,
            )
        )

    # Add aggregate cards if multiple stocks
    if len(symbols) > 1 and cards:
        avg_return = sum(
            stock_data[s].get("metrics", {}).get("total_return", 0)
            for s in symbols if s in stock_data
        ) / len([s for s in symbols if s in stock_data])

        cards.append(
            create_metric_card(
                title="AVG RETURN",
                value=f"{avg_return:+.1f}%",
                change_positive=avg_return > 0,
            )
        )

    if not cards:
        return html.Div(
            [
                html.Div(
                    [
                        html.I(className="bi bi-exclamation-circle", style={"fontSize": "32px", "opacity": "0.5"}),
                        html.P("Unable to load stock data", className="mt-2 mb-0"),
                    ],
                    className="empty-state",
                )
            ],
            style={"gridColumn": "1 / -1"},
        )
    return cards


# =============================================================================
# NEWS & AI CALLBACKS
# =============================================================================


@callback(
    Output("cache-enabled", "data"),
    Input("cache-toggle", "value"),
)
def update_cache_enabled(toggle_value):
    """Sync cache toggle with store."""
    return toggle_value


@callback(
    Output("news-data-store", "data"),
    Input("selected-symbols", "data"),
    Input("refresh-data-btn", "n_clicks"),
    Input("cache-enabled", "data"),
    # See fetch_stock_data_callback: fires on load to rehydrate after a refresh.
    prevent_initial_call=False,
)
async def fetch_news_data(symbols, refresh_click, cache_enabled):
    """Fetch news for ALL selected symbols.

    Returns a dict with articles organized by symbol for per-symbol tabs.

    Async: per-symbol fetches run concurrently in threads, capped LOW —
    Alpha Vantage's free tier rate-limits hard, and a parallel blast would
    trade a slow callback for a 429'd one.
    """
    if not symbols:
        return {}

    def _fetch_one(symbol):
        """Blocking per-symbol news fetch — runs in a worker thread."""
        articles = fetch_news_cached(symbol) if cache_enabled else fetch_news(symbol)
        return [
            {
                "id": a.id,
                "symbol": a.symbol,
                "title": a.title,
                "source": a.source,
                "url": a.url,
                "published_at": a.published_at.isoformat(),
                "summary": a.summary,
                "sentiment": a.sentiment,
                "sentiment_score": a.sentiment_score,
                "impact": a.impact,
                "price_change_percent": a.price_change_percent,
                "ticker_relevance_score": a.ticker_relevance_score,
                "topics": a.topics,
                "overall_sentiment_score": a.overall_sentiment_score,
                "overall_sentiment_label": a.overall_sentiment_label,
            }
            for a in (articles or [])
        ]

    sem = asyncio.Semaphore(APP.NEWS_FETCH_CONCURRENCY)

    async def _fetch_guarded(symbol):
        async with sem:
            return await asyncio.to_thread(_fetch_one, symbol)

    results = await asyncio.gather(
        *(_fetch_guarded(s) for s in symbols), return_exceptions=True)

    articles_by_symbol = {}
    for symbol, result in zip(symbols, results):
        if isinstance(result, BaseException):
            logger.warning(f"News fetch failed for {symbol}: {result}")
            articles_by_symbol[symbol] = []
        else:
            articles_by_symbol[symbol] = result

    return {
        "symbols": symbols,
        "articles_by_symbol": articles_by_symbol,
        "fetched_at": datetime.now().isoformat(),
    }


@callback(
    Output("ai-analysis-store", "data"),
    Input("run-confirm-btn", "n_clicks"),
    Input("ai-retry-btn", "n_clicks"),
    State("run-scope", "value"),
    State("news-data-store", "data"),
    State("selected-symbols", "data"),
    State("stock-data-store", "data"),
    State("run-date-picker", "date"),
    State("run-lookback", "value"),
    State("run-model", "value"),
    State("run-type", "value"),
    State("run-include-research", "value"),
    State("run-recs", "value"),
    State("run-recs-model", "value"),
    prevent_initial_call=True,
)
async def generate_ai_analysis(n_clicks, retry_clicks, scope, news_data, symbols,
                               stock_data, run_date, lookback, model, depth,
                               research, recs, recs_model):
    """Generate structured AI analysis grounded in financial data and news.

    Triggered by user clicking "AI Report" or "Full Analysis" button.
    Async callback: LLM/research calls fan out through asyncio.to_thread, so
    the minutes-long run holds no callback-threadpool slot while waiting.
    Checks Postgres/S3 cache first if persistence layer is available.

    The Full Analysis modal supplies an as-of date: articles are filtered to
    that date, price-derived metrics are computed from truncated OHLCV, and
    live quote data is suppressed for past dates — no lookahead in backtests.
    """
    from datetime import date as date_cls

    # A models-only run has no report to build.
    scope = scope or "full"
    if ctx.triggered_id == "run-confirm-btn" and scope == "models":
        raise PreventUpdate
    if not (n_clicks or retry_clicks) or not news_data or not news_data.get("articles_by_symbol"):
        raise PreventUpdate

    # Target date: the session whose close is being predicted, taken from the
    # Full Analysis picker when that flow triggered this, else the AI Report
    # modal's own. Data is cut off at the PREVIOUS trading day, so a Monday
    # target sees nothing after the preceding Friday's close.
    today = date_cls.today()
    is_full = scope == "full"
    picked = str(run_date)[:10] if run_date else None
    from utils.trading_calendar import resolve_target_and_cutoff
    target_d, as_of_d = resolve_target_and_cutoff(picked)
    is_backtest = target_d < today
    as_of_str = as_of_d.isoformat()
    target_str = target_d.isoformat()

    # One set of controls, read once. There is no longer a second, hidden
    # parameter panel for another flow to read by mistake.
    lookback_days = int(lookback or 7)
    report_model = model or "gpt-5.6-luna"
    include_thesis_flag = (depth or "thesis") != "standard"
    recs_mode = recs or "auto"
    recs_model_val = recs_model or "claude-sonnet-5"
    report_provider = "openai" if report_model.startswith("gpt-") else "anthropic"
    # Full pipeline already produces research through the prediction stage;
    # running it here too would double the LLM spend for identical output.
    include_research = bool(research) and not is_full

    from services import progress_service as prog
    if is_full:
        prog.start_run(f"Full Analysis — {len(symbols or [])} symbols, "
                       f"target {target_str} (data through {as_of_str})"
                       + (" (backtest)" if is_backtest else ""))
    prog.emit("ai", f"AI Report starting for {', '.join(symbols or [])}")
    prog.emit("action", f"Options: model={report_model}, window={lookback_days}d, "
                        f"type={'thesis' if include_thesis_flag else 'standard'}, "
                        f"research={'on' if include_research else 'pipeline' if is_full else 'off'}, "
                        f"recs={recs_mode} ({recs_model_val}), "
                        f"target {target_str}, data through {as_of_str}")

    articles_by_symbol = news_data.get("articles_by_symbol", {})

    # News window: the SAME point-in-time fetch the modal preview showed —
    # windowed [as_of - lookback, as_of], DB-cached, lookahead-safe. The
    # dcc.Store's articles are only a fallback: they hold whatever the last
    # background refresh grabbed, which may not cover the chosen window.
    from services.news_window import fetch_point_in_time_news, filter_articles_as_of

    def _art_dict(a):
        return {
            "id": getattr(a, "id", None), "symbol": getattr(a, "symbol", None),
            "title": getattr(a, "title", ""), "source": getattr(a, "source", None),
            "url": getattr(a, "url", None),
            "published_at": (a.published_at.isoformat()
                             if getattr(a, "published_at", None) else None),
            "summary": getattr(a, "summary", None),
            "sentiment": getattr(a, "sentiment", None),
            "sentiment_score": getattr(a, "sentiment_score", None),
            "impact": getattr(a, "impact", None),
            "price_change_percent": getattr(a, "price_change_percent", None),
            "ticker_relevance_score": getattr(a, "ticker_relevance_score", None),
            "topics": getattr(a, "topics", None),
            "overall_sentiment_score": getattr(a, "overall_sentiment_score", None),
            "overall_sentiment_label": getattr(a, "overall_sentiment_label", None),
        }

    windowed: dict[str, list] = {}
    for sym in (symbols or []):
        try:
            fetched = fetch_point_in_time_news(sym, as_of_str, lookback_days=lookback_days)
            windowed[sym] = [_art_dict(a) for a in fetched]
        except Exception as e:
            logger.warning(f"PIT news fetch failed for {sym}, falling back to store: {e}")
            windowed[sym] = filter_articles_as_of(
                articles_by_symbol.get(sym, []), as_of_str, lookback_days=lookback_days)
    articles_by_symbol = windowed
    total_articles = sum(len(a) for a in articles_by_symbol.values())
    prog.emit("news", f"News window {lookback_days}d ending {as_of_str} "
                      f"(target {target_str} minus 1 trading day): "
                      f"{total_articles} articles fetched (point-in-time)")

    # Check persistent cache (Postgres + S3) before running LLM
    if _s3_available and ctx.triggered_id != "ai-retry-btn":
        try:
            from services import persistence_service as ps
            data_hash = ps.compute_data_hash({
                "news": articles_by_symbol,
                "symbols": sorted(symbols or []),
                "as_of": as_of_str,
                # v4-merged: research epilogue feeds banner/watch/thesis;
                # Full Analysis no longer runs the shallow per-symbol pass.
                # Model, window and analysis type key the cache — switching
                # model must not serve the other model's cached report.
                "schema": "v4-merged",
                "model": report_model,
                "lookback": lookback_days,
                "thesis": include_thesis_flag,
                "research": include_research,
                "recs": recs_mode,
            })
            cached = ps.get_cached_report(None, as_of_str, "ai_report", data_hash)
            if cached:
                logger.info("AI report cache hit (Postgres/S3)")
                cached_result = json.loads(cached)
                cached_result["from_cache"] = True
                return cached_result
        except Exception as e:
            logger.debug(f"AI report cache check failed: {e}")

    llm = get_llm()
    stock_data = stock_data or {}

    result = {
        "overall": None,
        "by_symbol": {},
        "as_of": as_of_str,
        "generated_at": datetime.now().isoformat(),
    }

    # Per-symbol context: fundamentals, validated metric blocks, events, peers
    enriched_stock_data = {}
    extra_blocks_by_symbol = {}
    for symbol in (symbols or []):
        sym_data = stock_data.get(symbol, {})
        enriched = {
            "metrics": sym_data.get("metrics", {}),
            "signals": sym_data.get("signals", {}),
            "info": {},
        }
        try:
            from services.stock_data import get_stock_info
            # Only static identity, live run or not. get_stock_info() returns a
            # CURRENT quote, so on a run whose target is today's close it would
            # hand the LLM intraday prices from the very session being
            # predicted. Keeping live and backtest on one path is what makes
            # the window leak-proof rather than leak-proof-only-in-backtest.
            # Prices, technicals and fundamentals all arrive instead through
            # the validated blocks below, which are computed from OHLCV
            # truncated to the as-of date.
            info = get_stock_info(symbol)
            enriched["info"] = {
                "name": info.name, "sector": info.sector, "industry": info.industry,
            }
            # Store metrics/signals are computed on current data — always drop.
            enriched["metrics"] = {}
            enriched["signals"] = {}
        except Exception as e:
            logger.warning(f"Could not fetch stock info for {symbol}: {e}")
        enriched_stock_data[symbol] = enriched

        # Validated blocks: computed metrics (from OHLCV truncated to as-of),
        # event calendar, peer relative strength, company profile
        blocks = {}
        try:
            from utils.metrics import (
                compute_trading_metrics, format_metrics_block,
                compute_peer_relative_strength,
            )
            from utils.events import get_upcoming_events, format_events_block
            from models.sector_map import get_peers
            from services.stock_data import get_company_profile

            prices_json = sym_data.get("prices")
            if prices_json:
                df = pd.read_json(StringIO(prices_json))
                if "Date" in df.columns:
                    df = df.set_index("Date")
                df.index = pd.to_datetime(df.index)
                df = df[df.index <= as_of_str]
                m = compute_trading_metrics(df)
                block = format_metrics_block(symbol, m)
                if block:
                    blocks["metrics"] = block

            ev = get_upcoming_events(symbol, as_of_str)
            block = format_events_block(symbol, ev)
            if block:
                blocks["events"] = block

            peers = get_peers(symbol)
            if peers:
                # Cutoff applies live too — peer relative strength must be
                # measured through the previous trading day, not "latest".
                block = compute_peer_relative_strength(
                    symbol, peers, as_of=as_of_str)
                if block:
                    blocks["peers"] = block

            profile = get_company_profile(symbol)
            if profile:
                blocks["profile"] = profile
        except Exception as e:
            logger.warning(f"Validated block build failed for {symbol}: {e}")
        extra_blocks_by_symbol[symbol] = blocks

    # Collect all articles for overall analysis
    all_articles = []
    symbol_tasks = {}
    for symbol, articles in articles_by_symbol.items():
        if not articles:
            continue
        all_articles.extend(articles)
        symbol_tasks[symbol] = articles

    overall_articles = all_articles if all_articles else None

    # Execute LLM calls in parallel
    # Research tasks run beside the summaries (they dominate wall-clock at
    # ~35s each, so give them their own worker slots).
    def _run_research(sym: str):
        from models.trading_agents_model import TradingAgentsModel
        entry = (stock_data or {}).get(sym) or {}
        df = None
        if entry.get("prices"):
            from io import StringIO
            df = pd.read_json(StringIO(entry["prices"]))
        # The store's frame can be stale (cached from an old fetch, or a
        # silent vendor failure fell back to cache). Passing it through
        # trips the model's staleness guard and fails the report; passing
        # None instead makes the agent fetch fresh 1y data itself — the
        # guard then only fires if even the FRESH fetch is stale, which is
        # the case it exists for.
        if df is not None and len(df):
            last_bar = pd.to_datetime(df.index.max()).date()
            age = (date_cls.fromisoformat(as_of_str) - last_bar).days
            if age > 5:
                prog.emit("ta", f"{sym}: cached prices end {last_bar} "
                                f"({age}d before {as_of_str}) — refetching fresh")
                df = None
        model_obj = TradingAgentsModel()
        res = model_obj.predict(sym, df, as_of=as_of_str, model=report_model,
                                include_thesis=include_thesis_flag)
        return res

    # Async fan-out: each task holds a worker thread only while its blocking
    # LLM/research call runs; result processing happens on the event loop as
    # each task completes (single-threaded — `result` mutations are safe).
    n_workers = min(len(symbol_tasks) * (2 if include_research else 1) + 1, 8)
    sem = asyncio.Semaphore(n_workers)

    async def _run_task(task_type, symbol, fn, *args, **kw):
        try:
            async with sem:
                analysis = await asyncio.to_thread(fn, *args, **kw)
        except Exception as e:
            logger.warning(f"Error in LLM analysis for {task_type} {symbol}: {e}")
            prog.emit("error", f"AI analysis failed for {symbol or 'overall'}: {str(e)[:80]}")
            return

        if task_type == "research":
            r = analysis  # PredictionResult
            if r is None or r.error:
                prog.emit("error", f"{symbol}: research report failed: "
                                   f"{(r.error if r else 'no result')[:80]}")
                return
            details = r.details or {}
            raw = details.get("raw_response", "")
            result.setdefault("research_by_symbol", {})[symbol] = {
                "decision": r.decision,
                "confidence": r.confidence,
                "raw_response": raw,
                "triggers": details.get("triggers") or {},
                "structured": details.get("structured") or {},
                "provenance": details.get("provenance") or {},
                "model": report_model,
            }
            # Persist so History + PDF pick it up like Predict-flow reports
            try:
                await asyncio.to_thread(
                    get_cache().save_trading_agent_report,
                    symbol=symbol, trade_date=as_of_str,
                    decision=r.decision, confidence=r.confidence,
                    report_text=raw, model_name=report_model,
                    # This flow was omitting token counts entirely, so every
                    # report it wrote recorded 0 and cost was unmeasurable.
                    input_tokens=details.get("input_tokens", 0),
                    output_tokens=details.get("output_tokens", 0),
                )
            except Exception as e:
                logger.warning(f"research report persist failed for {symbol}: {e}")
            prog.emit("ta", f"{symbol}: research → {r.decision} "
                            f"({r.confidence:.0%})")
            return
        if analysis:
            if task_type == "symbol":
                result["by_symbol"][symbol] = analysis
                prog.emit("ai", f"{symbol}: {analysis.get('recommendation', '?')} "
                                f"({int((analysis.get('confidence') or 0) * 100)}%)")
            else:
                result["overall"] = analysis
                prog.emit("ai", f"Overall: {analysis.get('recommendation', '?')}")

    aio_tasks = []

    if include_research:
        for symbol in symbol_tasks:
            aio_tasks.append(_run_task("research", symbol, _run_research, symbol))
            prog.emit("ta", f"{symbol}: research report starting ({report_model})…")

    # When the research report runs, it IS the per-symbol text analysis —
    # running the shallow summary beside it paid twice for one answer.
    # Full Analysis is the same situation through a different door: its
    # prediction pipeline always runs the trading_agents model, so the
    # research text arrives via the signals store — running the shallow
    # pass beside it produced a second, thinner opinion on the same data
    # that could contradict the research verdict in the same tab.
    # The portfolio-level overall call below still runs either way (the
    # research reports are single-symbol and cannot replace it).
    if not include_research and not is_full:
        for symbol, articles in symbol_tasks.items():
            sym_stock = {symbol: enriched_stock_data.get(symbol, {})}
            aio_tasks.append(_run_task(
                "symbol", symbol, llm.summarize_news_structured, articles, [symbol],
                stock_data=sym_stock,
                as_of_date=as_of_str,
                extra_blocks=extra_blocks_by_symbol.get(symbol, {}),
                include_thesis=include_thesis_flag,
                model=report_model,
                provider=report_provider,
            ))
            prog.emit("ai", f"{symbol}: analyzing {len(articles)} articles "
                            f"({report_model})…")

    # The overall shallow call is Luna's understudy: when research runs
    # per symbol AND Luna will synthesize the portfolio view, it adds a
    # third opinion nobody consumes. Run it only when one of those is off.
    run_overall = overall_articles and not (
        include_research and recs_mode != "off")
    if run_overall:
        # The per-symbol metric blocks are already computed above from OHLCV
        # truncated to the cutoff. The overall call used to get none of them,
        # and since this flow also withholds live quotes, its financial block
        # was empty — the model reported a broken data feed instead of an
        # analysis. Hand it the same validated numbers the symbol calls get.
        overall_metrics = "\n\n".join(
            b["metrics"] for b in extra_blocks_by_symbol.values() if b.get("metrics")
        )
        aio_tasks.append(_run_task(
            "overall", None, llm.summarize_news_structured, overall_articles, symbols or [],
            stock_data=enriched_stock_data,
            as_of_date=as_of_str,
            extra_blocks={"metrics": overall_metrics} if overall_metrics else None,
            model=report_model,
            provider=report_provider,
        ))
        prog.emit("ai", f"Overall: synthesizing {len(overall_articles)} articles "
                        f"across {len(symbols or [])} symbols…")

    await asyncio.gather(*aio_tasks)

    # Attach research to each symbol's analysis dict so the tab renderer
    # gets everything through one object (additive key; old payloads and
    # consumers that don't know about it are unaffected). When the shallow
    # per-symbol pass was skipped, derive the banner stance from the
    # research verdict so the UI (and XLSX summary) still shows one.
    _stance = {"BUY": "BULLISH", "SELL": "BEARISH", "HOLD": "NEUTRAL"}
    for sym, res in (result.get("research_by_symbol") or {}).items():
        entry = result["by_symbol"].setdefault(sym, {})
        entry["research"] = res
        if "recommendation" not in entry:
            st = res.get("structured") or {}
            entry["recommendation"] = (st.get("stance")
                                       or _stance.get(res.get("decision"), "NEUTRAL"))
            entry["confidence"] = res.get("confidence")
            entry["stance_source"] = "research_verdict"
            # The epilogue supplies the panel fields the shallow pass used to —
            # one analyst call now feeds banner, watch items, and thesis.
            if st.get("sentiment_alignment"):
                entry["sentiment_explanation"] = st["sentiment_alignment"]
            if st.get("watch_items"):
                entry["watch_items"] = st["watch_items"]
            if st.get("company_thesis"):
                entry["company_thesis"] = st["company_thesis"]
            if res.get("provenance"):
                entry["provenance"] = res["provenance"]

    # Recommendation (Luna) request: recorded on the store so the Luna
    # callback knows to fire and WHAT its evidence basis is.
    if recs_mode != "off":
        if recs_mode == "signals":
            result["recs_request"] = "signals"
        else:
            result["recs_request"] = ("research+signals" if include_research
                                      else "news+signals")
        result["recs_model"] = recs_model_val
    else:
        # Explicit marker: the Full Analysis flow sets the requested flag
        # regardless, so the Luna callback needs to know recs were opted out.
        result["recs_off"] = True

    if not result["by_symbol"] and not result["overall"]:
        result["failed"] = True
        prog.emit("error", "AI Report produced no analysis")
    else:
        prog.emit("ai", f"AI Report complete ({len(result['by_symbol'])} symbols) — "
                        "model predictions next")

    # Store to Postgres/S3 for future cache hits
    if _s3_available and not result.get("failed"):
        try:
            from services import persistence_service as ps
            data_hash = ps.compute_data_hash({
                "news": articles_by_symbol,
                "symbols": sorted(symbols or []),
                "as_of": as_of_str,
                # v4-merged: research epilogue feeds banner/watch/thesis;
                # Full Analysis no longer runs the shallow per-symbol pass.
                # Model, window and analysis type key the cache — switching
                # model must not serve the other model's cached report.
                "schema": "v4-merged",
                "model": report_model,
                "lookback": lookback_days,
                "thesis": include_thesis_flag,
                "research": include_research,
                "recs": recs_mode,
            })
            ps.store_report(
                symbol=None,
                trade_date=as_of_str,
                report_type="ai_report",
                input_data_hash=data_hash,
                content=json.dumps(result, default=str, indent=2),
                file_format="json",
            )
        except Exception as e:
            logger.warning(f"Failed to persist AI report to S3: {e}")

    return result


@callback(
    Output("active-tab-store", "data"),
    # symbol-tabs is built by a callback and only exists on /analyze.
    Input("symbol-tabs", "active_tab", allow_optional=True),
    prevent_initial_call=True,
)
def track_active_tab(active_tab):
    """Remember the user's active tab so re-renders don't reset it."""
    if not active_tab:
        raise PreventUpdate
    return active_tab


def _tab_ids(symbols, news_data):
    """Which tabs exist right now, and which is the default.

    No History tab: the archive it held is now the Performance, Reports and
    Activity sections. Analyze is about the symbols in front of you.
    """
    if not symbols:
        return ["tab-overview-empty"], "tab-overview-empty"
    if not news_data or not news_data.get("articles_by_symbol"):
        return ["tab-loading"], "tab-loading"
    return ["tab-overall"] + [f"tab-{s}" for s in symbols], "tab-overall"


@callback(
    Output("symbol-tabs-container", "children"),
    Input("selected-symbols", "data"),
    Input("news-data-store", "data"),
    State("active-tab-store", "data"),
)
def update_symbol_tabs(symbols, news_data, stored_tab):
    """Build the tab BAR only: labels, ids, and which one is selected.

    Content is rendered separately by render_active_tab. Previously this one
    callback built the body of every tab on every store change, so a 20-symbol
    watchlist serialized 20 full analysis trees to the browser in order to
    display one. It also took eight Inputs, so any store update rebuilt all of
    them; the bar itself only depends on the symbol list and whether news has
    arrived.
    """
    tab_ids, default = _tab_ids(symbols, news_data)
    labels = {
        "tab-overview-empty": "Overview",
        "tab-loading": "Loading...",
        "tab-overall": "Overall",
    }
    return html.Div(
        [
            dbc.Tabs(
                [
                    dbc.Tab(
                        label=labels.get(tid, tid.removeprefix("tab-")),
                        tab_id=tid,
                        className="context-tab",
                    )
                    for tid in tab_ids
                ],
                id="symbol-tabs",
                active_tab=restore_tab(stored_tab, tab_ids, default),
                className="symbol-tabs",
            ),
            dcc.Loading(
                html.Div(id="tab-content", className="tab-content-host"),
                type="circle",
                color="#00D4AA",
                target_components={"tab-content": "children"},
                overlay_style={"visibility": "visible", "opacity": 0.45},
                delay_show=350,
            ),
        ],
    )


@callback(
    Output("tab-content", "children"),
    Input("symbol-tabs", "active_tab", allow_optional=True),
    Input("news-data-store", "data"),
    Input("ai-analysis-store", "data"),
    Input("selected-symbols", "data"),
    Input("model-signals-store", "data"),
    Input("strategy-metrics-store", "data"),
    Input("strategy-evaluations-store", "data"),
    Input("report-history-store", "data"),
    Input("recommendations-store", "data"),
)
def render_active_tab(
    active_tab, news_data, ai_analysis, symbols, model_signals,
    strategy_metrics, strategy_evaluations, report_history, recommendations,
):
    """Render the body of the selected tab, and only that one."""
    tab_ids, default = _tab_ids(symbols, news_data)
    active = active_tab if active_tab in tab_ids else default

    if active == "tab-overview-empty":
        return create_overview_empty_state()
    if active == "tab-loading":
        return create_loading_state(symbols or [])

    articles_by_symbol = (news_data or {}).get("articles_by_symbol", {})
    analysis_by_symbol = (ai_analysis or {}).get("by_symbol", {})
    ai_failed = bool(ai_analysis and ai_analysis.get("failed"))

    if active == "tab-overall":
        return build_overall_tab_content(
            articles_by_symbol=articles_by_symbol,
            analysis_by_symbol=analysis_by_symbol,
            overall_analysis=(ai_analysis or {}).get("overall", {}),
            symbols=symbols,
            ai_failed=ai_failed,
            recommendations=recommendations,
        )

    symbol = active.removeprefix("tab-")
    return build_tab_content(
        articles=articles_by_symbol.get(symbol, []),
        analysis=analysis_by_symbol.get(symbol, {}),
        symbols=[symbol],
        is_overall=False,
        model_signals=(model_signals or {}).get(symbol, {}),
        strategy_metrics=[m for m in (strategy_metrics or [])
                          if m.get("symbol") == symbol],
        strategy_evaluations=[e for e in (strategy_evaluations or [])
                              if e.get("symbol") == symbol],
        ai_failed=ai_failed,
        recommendations=(recommendations or {}).get("by_symbol", {}).get(symbol, {}),
    )


@callback(
    Output("llm-status", "children"),
    Input("news-data-store", "data"),
)
def update_llm_status(news_data):
    """Update LLM status indicator in panel header."""
    llm = get_llm()

    if not llm.is_available():
        return "LLM: Offline"

    return f"LLM: {llm.provider}"


@callback(
    Output("signals-display", "children"),
    Input("stock-data-store", "data"),
    State("selected-symbols", "data"),
)
def update_signals_display(stock_data, symbols):
    """Update technical signals display."""
    if not stock_data or not symbols:
        return html.Div(
            [
                html.I(className="bi bi-activity", style={"fontSize": "20px", "opacity": "0.5", "display": "block", "marginBottom": "8px"}),
                html.Span("Select stocks to view signals"),
            ],
            className="empty-state",
            style={"padding": "16px", "fontSize": "13px"},
        )

    symbol = symbols[0]
    if symbol not in stock_data:
        return html.Div(
            [
                html.I(className="bi bi-exclamation-triangle", style={"fontSize": "20px", "opacity": "0.5", "display": "block", "marginBottom": "8px"}),
                html.Span("No signal data available"),
            ],
            className="empty-state",
            style={"padding": "16px", "fontSize": "13px"},
        )

    signals = stock_data[symbol].get("signals", {})

    if not signals:
        return html.Div(
            [
                html.I(className="bi bi-dash-circle", style={"fontSize": "20px", "opacity": "0.5", "display": "block", "marginBottom": "8px"}),
                html.Span("No signals available for this stock"),
            ],
            className="empty-state",
            style={"padding": "16px", "fontSize": "13px"},
        )

    # Tooltips explaining each signal type
    signal_tooltips = {
        "rsi": "RSI (14-day): Overbought >70, Oversold <30",
        "macd": "MACD (12/26/9): Bullish when MACD crosses above Signal",
        "trend_50": "Price position relative to 50-day SMA",
        "trend_200": "Price position relative to 200-day SMA (long-term)",
        "cross": "Golden Cross = SMA 50 > SMA 200 (Bullish), Death Cross = opposite",
        "bollinger": "Price position relative to Bollinger Bands (20-day, ±2σ)",
        "stochastic": "Stochastic (14/3): Overbought >80, Oversold <20",
        "momentum": "Price momentum based on rate of change",
    }

    signal_items = []

    for key, val in signals.items():
        if isinstance(val, dict):
            signal_text = val.get("signal", str(val))
            is_bullish = val.get("bullish", None)

            if is_bullish is True:
                color_class = "signal-bullish"
            elif is_bullish is False:
                color_class = "signal-bearish"
            else:
                # Check signal text for sentiment
                if "bullish" in signal_text.lower() or "above" in signal_text.lower():
                    color_class = "signal-bullish"
                elif "bearish" in signal_text.lower() or "below" in signal_text.lower():
                    color_class = "signal-bearish"
                else:
                    color_class = "signal-neutral"

            # Create unique ID for tooltip target
            signal_id = f"signal-{key}"

            # Get tooltip text for this signal type
            tooltip_text = signal_tooltips.get(key, f"{key.replace('_', ' ').title()} indicator")

            signal_items.append(
                html.Div(
                    [
                        html.Span(
                            key.replace("_", " ").title(),
                            className="signal-name",
                            id=signal_id,
                        ),
                        dbc.Tooltip(tooltip_text, target=signal_id, placement="left"),
                        html.Span(signal_text.replace("_", " ").title(), className=f"signal-value {color_class}"),
                    ],
                    className="signal-item",
                )
            )

    if not signal_items:
        return html.Div(
            [
                html.I(className="bi bi-dash-circle", style={"fontSize": "20px", "opacity": "0.5", "display": "block", "marginBottom": "8px"}),
                html.Span("No signals detected"),
            ],
            className="empty-state",
            style={"padding": "16px", "fontSize": "13px"},
        )
    return signal_items


# =============================================================================
# HISTORICAL REPORT DOWNLOAD
# =============================================================================


# =============================================================================
# DATA MODAL CALLBACKS
# =============================================================================


@callback(
    Output("data-modal", "is_open"),
    Output("data-table-container", "children"),
    Input("view-data-btn", "n_clicks"),
    Input("modal-close-btn", "n_clicks"),
    State("selected-symbols", "data"),
    State("data-modal", "is_open"),
    prevent_initial_call=True,
)
def toggle_data_modal(view_click, close_click, symbols, is_open):
    """Toggle data modal and populate table."""
    triggered = ctx.triggered_id

    if triggered == "modal-close-btn":
        return False, dash.no_update

    if triggered == "view-data-btn":
        if not symbols:
            return False, html.Div("No stocks selected", className="text-muted")

        cache = get_cache()
        symbol = symbols[0]

        try:
            df = cache.get_raw_data(symbol)
            if df.empty:
                return True, html.Div("No cached data", className="text-muted")

            # Create simple table
            table = dbc.Table.from_dataframe(
                df.head(50),
                striped=True,
                bordered=True,
                hover=True,
                responsive=True,
                className="table-dark",
            )
            return True, table
        except Exception as e:
            return True, html.Div(f"Error loading data: {e}", className="text-danger")

    return is_open, dash.no_update


@callback(
    Output("download-data", "data"),
    Input("export-data-btn", "n_clicks"),
    Input("modal-export-btn", "n_clicks"),
    State("selected-symbols", "data"),
    State("stock-data-store", "data"),
    State("indicator-toggles", "value"),
    State("model-signals-store", "data"),
    State("ai-analysis-store", "data"),
    State("recommendations-store", "data"),
    prevent_initial_call=True,
)
def export_data(export_click, modal_export_click, symbols, stock_data,
                indicators, model_signals, ai_analysis, recommendations):
    """Export everything on screen as a multi-sheet .xlsx.

    Replaces the old single-symbol Parquet dump. Sheets are dynamic: prices
    (+ only the toggled indicators) per symbol, then Predictions / AI
    Analysis / Recommendations whenever those stores hold data.
    """
    if not symbols:
        raise PreventUpdate

    try:
        from services.export_service import build_xlsx
        ai_ok = ai_analysis if ai_analysis and not ai_analysis.get("failed") else {}
        # Full Analysis keeps its research on the signals store — merge it in
        # so the AI Analysis sheet reflects what was on screen.
        ai_ok, _ = _merge_research_into_analysis(ai_ok, model_signals, symbols)
        payload = build_xlsx(
            symbols,
            stock_data or {},
            selected_indicators=indicators or [],
            model_signals=model_signals or {},
            ai_analysis=ai_ok,
            recommendations=recommendations or {},
        )
        filename = (f"quantnews_{'_'.join(symbols[:4])}"
                    f"{'_etc' if len(symbols) > 4 else ''}"
                    f"_{datetime.now():%Y-%m-%d_%H%M}.xlsx")
        from services import progress_service as prog
        prog.emit("action", f"XLSX export: {filename}")
        return dcc.send_bytes(payload, filename)
    except Exception as e:
        logger.error(f"Export error: {e}")
        raise PreventUpdate


@callback(
    Output("download-report", "data"),
    # Hidden button inside the Analyze context panel.
    Input("download-report-btn", "n_clicks", allow_optional=True),
    State("selected-symbols", "data"),
    State("ai-analysis-store", "data"),
    State("model-signals-store", "data"),
    State("recommendations-store", "data"),
    State("news-data-store", "data"),
    prevent_initial_call=True,
)
def download_report_pdf(n_clicks, symbols, ai_analysis, model_signals, recommendations, news_data):
    """Generate and download a PDF analysis report, also persisting to S3.

    Checks S3 cache first — if an identical report already exists (same input hash),
    returns the cached PDF immediately without regenerating.
    """
    if not n_clicks or not symbols:
        raise PreventUpdate

    symbols_str = "_".join(symbols)
    date_str = datetime.now().strftime("%Y-%m-%d")
    filename = f"{symbols_str}_Analysis_{date_str}.pdf"

    # Build a content hash from all the inputs
    data_hash = None
    if _s3_available:
        try:
            from services import persistence_service as ps
            data_hash = ps.compute_data_hash({
                "symbols": sorted(symbols),
                "ai": ai_analysis if ai_analysis and not ai_analysis.get("failed") else None,
                "signals": model_signals or {},
                "recs": recommendations.get("overall") if recommendations else None,
            })

            # Check for cached PDF with same inputs
            cached = ps.get_cached_report(None, date_str, "pdf_report", data_hash, raw=True)
            if cached:
                logger.info("PDF report cache hit — returning existing report")
                return dcc.send_bytes(cached, filename, mime_type="application/pdf")
        except Exception as e:
            logger.debug(f"PDF cache check failed: {e}")

    # Always include predictions and recommendations if available in DB
    if not model_signals and _s3_available:
        try:
            cache = get_cache()
            preds = cache.list_all_predictions(limit=200)
            if preds:
                model_signals = {}
                for p in preds:
                    sym = p["symbol"]
                    model = p["model_name"]
                    if sym not in model_signals:
                        model_signals[sym] = {}
                    model_signals[sym][model] = {
                        "decision": p.get("decision", "HOLD"),
                        "confidence": p.get("confidence"),
                        "up_probability": p.get("up_probability"),
                    }
                # These are whatever is newest in the DB across ALL symbols,
                # not the dates of any particular run, so they are labelled
                # as such instead of being passed off as this report's target.
                meta = preds[0] if preds else {}
                model_signals["_meta"] = {
                    "predict_date": meta.get("prediction_date", ""),
                    "target_date": meta.get("target_date", ""),
                    "dates_are_latest_available": True,
                }
        except Exception as e:
            logger.debug(f"Could not load predictions for report: {e}")

    if not recommendations and _s3_available:
        try:
            cache = get_cache()
            recs = cache.list_recommendation_runs(limit=1)
            if recs and recs[0].get("result_json"):
                recommendations = recs[0]["result_json"]
                recommendations["model_used"] = recs[0].get("model_used", "")
        except Exception as e:
            logger.debug(f"Could not load recommendations for report: {e}")

    from services.report_service import generate_report_pdf

    # Full Analysis keeps its research on the signals store — merge it in so
    # the PDF's per-symbol chapters carry the full report, whichever flow ran.
    ai_analysis, _ = _merge_research_into_analysis(ai_analysis, model_signals, symbols)

    # The run's own dates, so the report states them rather than inferring
    # them from whichever predictions happen to be in the payload. The DB
    # fallback above is NOT the run's, so it is left for the report to label
    # as "latest available" rather than being passed off as authoritative.
    _sig_meta = (model_signals or {}).get("_meta", {}) if isinstance(model_signals, dict) else {}
    if _sig_meta.get("dates_are_latest_available"):
        _sig_meta = {}

    pdf_bytes = generate_report_pdf(
        symbols=symbols,
        ai_analysis=ai_analysis,
        model_signals=model_signals,
        recommendations=recommendations,
        news_data=news_data,
        target_date=_sig_meta.get("target_date") or None,
        data_through=_sig_meta.get("predict_date") or None,
    )

    if not pdf_bytes:
        raise PreventUpdate

    if _s3_available and data_hash:
        try:
            from services import persistence_service as ps
            ps.store_report(
                symbol=None,
                trade_date=date_str,
                report_type="pdf_report",
                input_data_hash=data_hash,
                content=pdf_bytes,
                file_format="pdf",
                metadata={"filename": filename, "symbols": symbols},
            )
        except Exception as e:
            logger.warning(f"Failed to persist PDF to S3: {e}")

    from services import progress_service as prog
    prog.emit("action", f"PDF report generated: {filename}")
    return dcc.send_bytes(pdf_bytes, filename, mime_type="application/pdf")


# =============================================================================
# PERIOD SELECTOR CALLBACK
# =============================================================================


@callback(
    Output("current-period", "data"),
    Output({"type": "period-btn", "period": ALL}, "color"),
    Output({"type": "period-btn", "period": ALL}, "outline"),
    Input({"type": "period-btn", "period": ALL}, "n_clicks"),
    State("current-period", "data"),
    prevent_initial_call=True,
)
def update_period(clicks, current_period):
    """Update selected time period."""
    triggered = ctx.triggered_id

    if isinstance(triggered, dict) and triggered.get("type") == "period-btn":
        new_period = triggered["period"]
    else:
        new_period = current_period

    # Button styles: derive the period order from the pattern-matched inputs
    # themselves — a hardcoded list silently desyncs when the selector grows.
    periods = [inp["id"]["period"] for inp in ctx.inputs_list[0]]
    colors = ["primary" if p == new_period else "secondary" for p in periods]
    outlines = [p != new_period for p in periods]

    if new_period != current_period:
        from services import progress_service as prog
        prog.emit("action", f"Time period -> {new_period}")

    return new_period, colors, outlines


# =============================================================================
# MODEL PREDICTION CALLBACKS
# =============================================================================


@callback(
    Output("model-signals-store", "data"),
    Input("run-confirm-btn", "n_clicks"),
    State("run-scope", "value"),
    State("stock-data-store", "data"),
    State("selected-symbols", "data"),
    State("ensemble-config-store", "data"),
    State("news-data-store", "data"),
    State({"type": "run-model-check", "model": ALL}, "value"),
    State("run-ensemble-check", "value"),
    State("run-date-picker", "date"),
    State("run-model", "value"),
    State("run-type", "value"),
    background=True,
    running=[
        (Output("prediction-running-indicator", "style"), {"display": "block"}, {"display": "none"}),
    ],
    prevent_initial_call=True,
)
def generate_model_signals(n_clicks, scope, stock_data, symbols, ensemble_config,
                           news_data, model_checks, run_ensemble,
                           predict_date_str, research_model, research_depth):
    """Generate model predictions in a background subprocess.

    Supports backtesting: when the selected as-of date is in the past, OHLCV
    data is truncated to that date so models only see data available
    as of that day. The selected date is stored in metadata for
    correct persistence.
    """
    # A report-only run has no models to execute.
    scope = scope or "full"
    if scope == "report":
        raise PreventUpdate
    if not n_clicks or not stock_data or not symbols:
        raise PreventUpdate

    # The model checkboxes are now honoured on every scope. Full pipeline used
    # to overwrite them with [True] * 5, so the boxes you unticked ran anyway.
    is_full_analysis = scope == "full"
    research_kwargs = {}
    if is_full_analysis:
        # The research report (trading_agents) honours the report model/depth
        # choices; `research_model` is a distinct kwarg so no other model can
        # mistake it for its own.
        research_kwargs = {
            "research_model": research_model or None,
            "include_thesis": (research_depth or "thesis") != "standard",
        }
    else:
        # Models-only owns the feed; the full pipeline started it already.
        from services import progress_service as _prog
        _prog.start_run(f"Predictions — {len(symbols or [])} symbols")

    import os
    os.environ["_DASH_BG_SUBPROCESS"] = "1"

    from datetime import date as date_cls

    # The picker holds the TARGET session — the close being predicted. Models
    # are truncated to `predict_date`, the previous trading day, so a Monday
    # target trains and scores on nothing after the preceding Friday.
    from utils.trading_calendar import resolve_target_and_cutoff
    target_date, predict_date = resolve_target_and_cutoff(
        predict_date_str[:10] if predict_date_str else None
    )

    is_backtest = target_date < date_cls.today()

    _ALL_MODEL_IDS = [
        "kronos_mini", "xgboost_shap", "lightgbm",
        "deberta_sentiment", "trading_agents",
    ]
    selected_models = set()
    for model_id, checked in zip(_ALL_MODEL_IDS, model_checks or []):
        if checked:
            selected_models.add(model_id)

    try:
        from services.prediction_service import get_prediction_service
        from services.stock_data import fetch_stock_data as fetch_ohlcv

        service = get_prediction_service()
        results = {}

        spy_df = None
        try:
            spy_df = fetch_ohlcv("SPY", period="2y")
            # Always truncate to the cutoff, not only when backtesting: a live
            # run whose target is TODAY's close would otherwise be handed
            # today's partial intraday bar, which the target has not produced yet.
            spy_df = spy_df[spy_df.index <= str(predict_date)]
        except Exception as e:
            logger.warning(f"SPY fetch failed: {e}")

        needs_historical = selected_models & {"xgboost_shap", "lightgbm"}
        historical_global_news = {}
        if needs_historical:
            try:
                from services.news_service import fetch_historical_av_news
                from config import MODEL
                logger.info("Fetching historical global news for training")
                historical_global_news = fetch_historical_av_news(
                    "", months=MODEL.NEWS_LOOKBACK_MONTHS,
                )
                logger.info(
                    f"Global news: "
                    f"{sum(len(v) for v in historical_global_news.values())} "
                    f"articles across {len(historical_global_news)} days"
                )
            except Exception as e:
                logger.warning(f"Historical global news fetch failed: {e}")

        for symbol in symbols:
            sym_data = stock_data.get(symbol, {})
            prices_json = sym_data.get("prices")
            if not prices_json:
                continue

            try:
                df = pd.read_json(StringIO(prices_json))
                if df.empty:
                    continue
            except Exception:
                continue

            # Truncate OHLCV to the cutoff on EVERY run, not just backtests.
            # The target session's own bar must never be visible — live, that
            # bar is today's partial intraday print.
            if "Date" in df.columns:
                df = df[df["Date"] <= str(predict_date)]
            else:
                df = df[df.index <= str(predict_date)]
            if df.empty:
                logger.warning(
                    f"{symbol}: no data available as of {predict_date}"
                )
                continue

            sym_news = []
            if news_data and news_data.get("articles_by_symbol"):
                sym_news = news_data["articles_by_symbol"].get(symbol) or []

            # No lookahead: the news store holds articles fetched today, so it
            # is windowed to the cutoff on EVERY run — live included, where it
            # would otherwise leak articles published during the target session.
            # Robust half-open UTC window (see services.news_window).
            if sym_news:
                from services.news_window import filter_articles_as_of
                sym_news = filter_articles_as_of(sym_news, predict_date, lookback_days=3650)

            historical_av_news = {}
            if needs_historical:
                try:
                    from services.news_service import fetch_historical_av_news
                    from config import MODEL
                    logger.info(f"{symbol}: fetching historical AV news")
                    historical_av_news = fetch_historical_av_news(
                        symbol, months=MODEL.NEWS_LOOKBACK_MONTHS,
                    )
                    logger.info(
                        f"{symbol}: "
                        f"{sum(len(v) for v in historical_av_news.values())} "
                        f"articles across {len(historical_av_news)} days"
                    )
                except Exception as e:
                    logger.warning(
                        f"{symbol}: historical AV news fetch failed: {e}"
                    )

            mode = "backtest" if is_backtest else "live"
            logger.info(
                f"{symbol}: running {selected_models} ({mode}, "
                f"date={predict_date})"
            )
            from services import progress_service as _prog
            _prog.emit("models", f"{symbol}: running {len(selected_models)} models "
                                 f"({mode}, as-of {predict_date})")

            new_results = service.predict_symbol_no_store(
                symbol, df, spy_df=spy_df,
                news=sym_news,
                ensemble_config=ensemble_config,
                models_to_run=selected_models,
                run_ensemble=bool(run_ensemble),
                historical_av_news=historical_av_news,
                historical_global_news=historical_global_news,
                # Always the cutoff, live or backtest — downstream slicing has
                # to stop at the previous trading day either way.
                as_of=str(predict_date),
                **research_kwargs,
            )

            results[symbol] = new_results
            decisions = {m: r.get("decision") for m, r in new_results.items()
                         if isinstance(r, dict) and not r.get("error")}
            _prog.emit("models", f"{symbol}: " + ", ".join(
                f"{m}={d}" for m, d in decisions.items()))

        # Attach metadata so persist_predictions knows the date, and so the
        # report can state the target session rather than infer one.
        results["_meta"] = {
            "predict_date": predict_date.isoformat(),   # data cutoff
            "target_date": target_date.isoformat(),     # session being predicted
            "is_backtest": is_backtest,
        }

        return results

    except Exception as e:
        logger.error(f"Model signal generation error: {e}")
        return {}


@callback(
    Output("prediction-store-status", "data"),
    Input("model-signals-store", "data"),
    State("full-analysis-requested", "data"),
    prevent_initial_call=True,
)
def persist_predictions(signals, fa_requested):
    """Persist model predictions to Postgres in the server process.

    This callback receives the serializable dict from model-signals-store
    (produced by the background-callback subprocess) and writes it out. For
    backtest predictions, uses the selected date instead of today and
    auto-evaluates against actual prices.
    """
    if not signals:
        raise PreventUpdate

    meta = signals.pop("_meta", {})
    predict_date_str = meta.get("predict_date")
    is_backtest = meta.get("is_backtest", False)

    try:
        cache = get_cache()
        stored = 0
        for symbol, model_results in signals.items():
            if not isinstance(model_results, dict):
                continue
            for model_name, result_dict in model_results.items():
                if not isinstance(result_dict, dict):
                    continue
                if result_dict.get("error"):
                    continue
                cache.store_prediction(
                    symbol, model_name, result_dict,
                    prediction_date_str=predict_date_str,
                )
                stored += 1

                if model_name == "trading_agents":
                    details = result_dict.get("details", {})
                    raw_response = details.get("raw_response", "")
                    if raw_response:
                        cache.save_trading_agent_report(
                            symbol=symbol,
                            trade_date=details.get("trade_date", ""),
                            decision=result_dict.get("decision", "HOLD"),
                            confidence=result_dict.get("confidence", 0.0),
                            report_text=raw_response,
                            model_name=details.get("model", ""),
                            input_tokens=details.get("input_tokens", 0),
                            output_tokens=details.get("output_tokens", 0),
                        )

        # Auto-evaluate backtest predictions (target date is in the past)
        evaluated = 0
        if is_backtest:
            evaluated = cache.evaluate_predictions()

        from services import progress_service as prog
        prog.emit("store", f"Stored {stored} predictions"
                           + (f", evaluated {evaluated}" if evaluated else ""))
        # New rows: the launch screen must show this run, not the last one.
        from services.dashboard_service import invalidate_memo
        invalidate_memo()
        if not fa_requested:
            prog.finish_run(f"Predictions complete — {stored} stored")
        else:
            prog.emit("luna", "Handing off to recommendation synthesis (Luna)…")

        return {
            "stored_at": str(datetime.now()),
            "count": stored,
            "evaluated": evaluated,
            "is_backtest": is_backtest,
            "predict_date": predict_date_str,
        }

    except Exception as e:
        logger.error(f"Prediction persistence error: {e}")
        return {"error": str(e)}


# =============================================================================
# TRADINGAGENTS REPORT HISTORY
# =============================================================================


@callback(
    Output("history-eval-status", "data"),
    Output("history-eval-toast", "is_open"),
    Output("history-eval-toast", "children"),
    Output("history-eval-toast", "icon"),
    # Dash 4's renderer hard-errors on dispatch when an Input id is missing
    # from the layout (Dash 3 tolerated it); allow_optional restores that.
    # This button lives on the Performance page, so it is absent elsewhere.
    Input("perf-evaluate-btn", "n_clicks", allow_optional=True),
    prevent_initial_call=True,
)
def evaluate_predictions_now(n_clicks):
    """Score pending predictions against actual closes, on user demand.

    Triggered from the Performance page. The n_clicks guard also covers the
    re-render insertion trigger: n_clicks resets to None whenever the page
    rebuilds its button.
    """
    if not n_clicks:
        raise PreventUpdate
    try:
        cache = get_cache()
        count = cache.evaluate_predictions()
        from services import progress_service as prog
        prog.emit("action", f"Evaluation run: {count} predictions scored")
        # Scoring rewrites was_correct/pnl on existing rows, so the cached
        # cohort and rolling stats are now wrong.
        from services.dashboard_service import invalidate_memo
        invalidate_memo()
        if count > 0:
            msg = f"Evaluated {count} prediction{'s' if count != 1 else ''} against actual closing prices."
            icon = "success"
        else:
            msg = ("No predictions could be evaluated yet — either their target date "
                   "hasn't closed, or the closing prices haven't been fetched "
                   "(load the symbol to refresh price data).")
            icon = "warning"
        return {"evaluated": count, "at": datetime.now().isoformat()}, True, msg, icon
    except Exception as e:
        logger.warning(f"Manual evaluation failed: {e}")
        return dash.no_update, True, f"Evaluation failed: {str(e)[:120]}", "danger"


@callback(
    Output("report-history-store", "data"),
    Input("selected-symbols", "data"),
    Input("prediction-store-status", "data"),
    Input("download-report", "data"),
    Input("history-eval-status", "data"),
)
def load_report_history(symbols, _prediction_status, _download_done, _eval_status):
    """Load all historical data: reports, predictions, recommendations, TradingAgents."""
    return fetch_report_history(symbols)


_HISTORY_TTL_S = 3.0
_history_memo: dict = {}


def fetch_report_history(symbols=None, only=None) -> dict:
    """Read the archive straight from the cache layer.

    Called both by the store-populating callback and directly by the router.
    The router cannot wait for the store: on a deep link the store update can
    land before #archive-body is mounted, and that update is then dropped,
    leaving Performance and Reports permanently empty.

    A very short TTL collapses the cold-load burst, where the router and the
    store callback both ask for the same rows within milliseconds. It is
    deliberately shorter than any human round trip, so it cannot show stale
    data after a run.
    """
    key = (tuple(sorted(symbols)) if symbols else (),
           tuple(sorted(only)) if only else ())
    hit = _history_memo.get(key)
    if hit and (time.monotonic() - hit[0]) < _HISTORY_TTL_S:
        return hit[1]

    def wanted(bucket):
        return only is None or bucket in only

    result = {
        "reports": [],
        "predictions": [],
        "recommendations": [],
        "trading_agent_reports": [],
    }
    try:
        cache = get_cache()

        # TradingAgents reports — load for selected symbols or all if none selected
        ta_reports = []
        if not wanted("trading_agent_reports"):
            pass
        elif symbols:
            for symbol in symbols:
                ta_reports.extend(cache.get_trading_agent_reports(symbol, limit=10))
        else:
            ta_reports = cache.get_all_trading_agent_reports(limit=30)
        ta_reports.sort(key=lambda r: r.get("created_at", ""), reverse=True)
        result["trading_agent_reports"] = ta_reports

        if wanted("reports"):
            result["reports"] = cache.list_report_catalog(limit=50)
        if wanted("predictions"):
            result["predictions"] = cache.list_all_predictions(limit=1000)
        if wanted("recommendations"):
            result["recommendations"] = cache.list_recommendation_runs(limit=50)

        _history_memo[key] = (time.monotonic(), result)
        return result
    except Exception as e:
        logger.debug(f"History load error: {e}")
        return result


# =============================================================================
# PIPELINE ACTIVITY FEED
# =============================================================================


_STAGE_ICONS = {
    "run": "bi-play-circle",
    "news": "bi-newspaper",
    "ai": "bi-file-text",
    "models": "bi-cpu",
    "ta": "bi-robot",
    "luna": "bi-stars",
    "store": "bi-database",
    "done": "bi-check-circle-fill",
    "error": "bi-x-circle-fill",
}


# Poll fast enough to feel live during a run, slowly enough the rest of the
# time that an idle tab is not making 40 requests a minute forever. A
# scheduled 08:00 job is still picked up within the idle interval.
_PROGRESS_POLL_ACTIVE_MS = 1500
_PROGRESS_POLL_IDLE_MS = 10_000


@callback(
    Output("progress-feed-scroll", "children"),
    Output("progress-count", "children"),
    Output("progress-header-icon", "children"),
    Output("progress-interval", "interval"),
    Input("progress-interval", "n_intervals"),
    State("progress-interval", "interval"),
)
def render_progress_panel(_n, _current_interval):
    """Live activity feed for pipeline runs.

    Streams events emitted by every stage (including the background model
    subprocess). Visible on every route: this is the live view, and the
    Activity section is the filtered archive on top of it.
    """
    from services import progress_service as prog

    feed = prog.get_feed()
    events = feed.get("events") or []
    poll = (_PROGRESS_POLL_ACTIVE_MS if feed.get("active")
            else _PROGRESS_POLL_IDLE_MS)
    # Writing interval on every tick restarts the timer and costs a DOM update
    # for no change; only send it when the rate actually changes.
    poll_out = poll if poll != _current_interval else dash.no_update
    if not events:
        return dash.no_update, dash.no_update, dash.no_update, poll_out

    # No auto-hide: the panel is an audit log now, so it stays up until the
    # user closes it. (It previously vanished 5 minutes after a run finished.)
    rows = []
    for e in events[-45:]:
        stage = e.get("stage", "")
        icon = _STAGE_ICONS.get(stage, "bi-dot")
        cls = "progress-line-error" if stage == "error" else (
            "progress-line-done" if stage == "done" else "")
        rows.append(html.Div(
            [
                html.Span(e.get("ts", ""), className="progress-ts"),
                html.I(className=f"bi {icon} progress-icon progress-icon-{stage}"),
                html.Span(e.get("message", ""), className="progress-msg"),
            ],
            className=f"progress-line {cls}",
        ))

    header_icon = (html.Div(className="progress-spinner")
                   if feed.get("active")
                   else html.I(className="bi bi-check-circle-fill progress-header-done"))

    return rows, f"{len(events)} events", header_icon, poll_out


@callback(
    Output("progress-panel-state", "data"),
    Input("progress-expand-btn", "n_clicks"),
    Input("progress-min-btn", "n_clicks"),
    Input("progress-close-btn", "n_clicks"),
    Input("progress-reopen-btn", "n_clicks"),
    State("progress-panel-state", "data"),
    prevent_initial_call=True,
)
def update_progress_panel_state(_e, _m, _c, _r, state):
    """Expand / minimise / close / reopen the activity panel."""
    state = dict(state or {"mode": "normal", "closed": False})
    trigger = ctx.triggered_id

    if trigger == "progress-close-btn":
        state["closed"] = True
    elif trigger == "progress-reopen-btn":
        state["closed"] = False
    elif trigger == "progress-expand-btn":
        state["mode"] = "normal" if state.get("mode") == "expanded" else "expanded"
    elif trigger == "progress-min-btn":
        state["mode"] = "normal" if state.get("mode") == "minimised" else "minimised"
    return state


@callback(
    Output("analysis-progress-panel", "style"),
    Output("progress-panel", "className"),
    Output("progress-reopen-btn", "style"),
    Output("progress-expand-icon", "className"),
    Output("progress-min-icon", "className"),
    Output("progress-interval", "disabled"),
    Input("progress-panel-state", "data"),
)
def apply_progress_panel_state(state):
    """Translate panel state into layout. Sizing lives in CSS classes.

    Closing the panel also stops the poll. The feed is written by a background
    subprocess (and possibly by another instance running the scheduler), so
    the browser has no way to be notified and has to ask; but there is nothing
    to ask for while the panel is shut. Reopening rehydrates from the stored
    feed, so nothing is lost by not having polled meanwhile.
    """
    state = state or {}
    closed = bool(state.get("closed"))
    mode = state.get("mode", "normal")

    panel_cls = "progress-panel"
    if mode == "expanded":
        panel_cls += " progress-panel-expanded"
    elif mode == "minimised":
        panel_cls += " progress-panel-minimised"

    return (
        {"display": "none"} if closed else {},
        panel_cls,
        {} if closed else {"display": "none"},
        "bi bi-arrows-angle-contract" if mode == "expanded"
        else "bi bi-arrows-angle-expand",
        "bi bi-plus-lg" if mode == "minimised" else "bi bi-dash-lg",
        closed,
    )


# =============================================================================
# MODEL SCOREBOARD MODAL
# =============================================================================


# =============================================================================
# VIEW FULL REPORT (switch to History tab + expand accordion)
# =============================================================================


# ── History filter callbacks ────────────────────────────────────────────

@callback(
    Output("history-filter-symbols", "data"),
    # Filter bar: Performance and Reports only.
    Input("history-symbol-dropdown", "value", allow_optional=True),
    Input({"type": "history-recent-chip", "symbol": ALL}, "n_clicks"),
    State("history-filter-symbols", "data"),
    prevent_initial_call=True,
)
def update_history_symbol_filter(dropdown_val, chip_clicks, current_filter):
    """Write the symbol filter to its store.

    Writes the store only. The dropdown itself is seeded from the store when
    the filter bar is built, because it exists on two of six routes and an
    Output on a missing component is a hard error in Dash 4.
    """
    triggered = ctx.triggered_id
    if isinstance(triggered, dict) and triggered.get("type") == "history-recent-chip":
        if not any(chip_clicks or []):
            raise PreventUpdate
        sym = triggered["symbol"]
        current = list(current_filter or [])
        if sym in current:
            raise PreventUpdate
        current.append(sym)
        return current
    # The dropdown unmounting also fires this. Do not let that clear the store.
    if triggered != "history-symbol-dropdown":
        raise PreventUpdate
    new_val = list(dropdown_val or [])
    # Writing back a value the store already holds would re-render whatever
    # depends on it, for no change.
    if new_val == list(current_filter or []):
        raise PreventUpdate
    return new_val


@callback(
    Output("history-filter-date-range", "data"),
    Output("history-filter-date-specific", "data"),
    Input({"type": "history-date-btn", "range": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def update_history_date_filter(btn_clicks):
    """Set the date range filter from the button group; clear any specific date.

    Clearing the picker is implicit: it is seeded from history-filter-date-
    specific when the bar is rebuilt, which the router does on this write.
    """
    if not any(btn_clicks or []):
        raise PreventUpdate
    triggered = ctx.triggered_id
    rng = "all"
    if isinstance(triggered, dict):
        rng = triggered.get("range", "all")
    return rng, None


@callback(
    Output("history-filter-date-specific", "data", allow_duplicate=True),
    Output("history-filter-date-range", "data", allow_duplicate=True),
    # Filter bar: Performance and Reports only.
    Input("history-date-picker", "date", allow_optional=True),
    State("history-filter-date-specific", "data"),
    prevent_initial_call=True,
)
def update_history_specific_date(picked_date, current_specific):
    """Set specific date filter from date picker; reset range on a real pick.

    When the picker is cleared programmatically (range button / Clear all),
    leave the range store alone — otherwise clicking "7d" would immediately
    be undone by this callback firing with date=None.

    The equality guard matters because this Input also fires when the picker
    is re-created by a re-render; writing back the value it was just seeded
    with would loop.
    """
    if picked_date == current_specific:
        raise PreventUpdate
    if not picked_date:
        return None, dash.no_update
    return picked_date, "all"


@callback(
    Output({"type": "history-date-btn", "range": ALL}, "color"),
    Output({"type": "history-date-btn", "range": ALL}, "outline"),
    Input("history-filter-date-range", "data"),
    Input("history-filter-date-specific", "data"),
)
def style_history_date_buttons(rng, specific):
    """Highlight the active date-range button; none active when a specific date is picked."""
    outputs = ctx.outputs_list[0] if ctx.outputs_list else []
    colors, outlines = [], []
    for o in outputs:
        r = o["id"].get("range", "")
        active = (not specific) and (r == (rng or "all"))
        colors.append("primary" if active else "secondary")
        outlines.append(not active)
    return colors, outlines


@callback(
    Output("history-activity-scope", "data"),
    Input({"type": "activity-scope-btn", "scope": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def set_activity_scope(_clicks):
    """Admin-only Activity Log scope switch.

    The buttons render only for Administrators and the server re-checks the
    role on every read, so a forged store value cannot widen visibility.
    """
    if not ctx.triggered_id:
        raise PreventUpdate
    return ctx.triggered_id.get("scope", "all")


@callback(
    Output("history-filter-symbols", "data", allow_duplicate=True),
    Output("history-filter-date-range", "data", allow_duplicate=True),
    Output("history-filter-date-specific", "data", allow_duplicate=True),
    Input({"type": "history-remove-filter", "symbol": ALL}, "n_clicks"),
    Input({"type": "history-remove-date-filter", "range": ALL}, "n_clicks"),
    # Rendered only when at least one filter chip is showing.
    Input("history-clear-all-filters", "n_clicks", allow_optional=True),
    State("history-filter-symbols", "data"),
    State("history-filter-date-range", "data"),
    State("history-filter-date-specific", "data"),
    prevent_initial_call=True,
)
def remove_history_filter(sym_clicks, date_clicks, clear_click, current_symbols, current_range, current_specific):
    """Handle filter chip removal and Clear All.

    Guard against spurious firings: Dash re-triggers this callback whenever
    chips/Clear-all are (re)inserted into the layout, with all n_clicks None —
    prevent_initial_call does not cover that case.
    """
    real_clicks = list(sym_clicks or []) + list(date_clicks or []) + [clear_click]
    if not any(c for c in real_clicks if c):
        raise PreventUpdate

    triggered = ctx.triggered_id

    if triggered == "history-clear-all-filters":
        return [], "all", None

    if isinstance(triggered, dict):
        if triggered.get("type") == "history-remove-filter":
            sym = triggered["symbol"]
            new_syms = [s for s in (current_symbols or []) if s != sym]
            return new_syms, current_range or "all", current_specific
        if triggered.get("type") == "history-remove-date-filter":
            return current_symbols or [], "all", None

    raise PreventUpdate


@callback(
    Output({"type": "history-section-body", "section": MATCH}, "style"),
    Output({"type": "history-section-chevron", "section": MATCH}, "className"),
    Input({"type": "history-section-toggle", "section": MATCH}, "n_clicks"),
    State({"type": "history-section-body", "section": MATCH}, "style"),
    prevent_initial_call=True,
)
def toggle_history_section(n_clicks, current_style):
    """Toggle collapsible section visibility."""
    if not n_clicks:
        raise PreventUpdate
    is_visible = (current_style or {}).get("display", "none") != "none"
    if is_visible:
        return {"display": "none"}, "bi bi-chevron-down history-chevron ms-auto"
    return {"display": "block"}, "bi bi-chevron-up history-chevron ms-auto"


@callback(
    Output("ta-report-modal", "is_open"),
    Output("ta-report-modal-title", "children"),
    Output("ta-report-modal-body", "children"),
    Input({"type": "view-full-report-btn", "symbol": ALL}, "n_clicks"),
    Input({"type": "ta-view-btn", "report": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def jump_to_full_report(view_clicks, ta_view_clicks):
    """Open the matching research report in the reader modal.

    Resolves against a live read rather than report-history-store. The store
    is loaded for the *selected symbols*, while the Reports page renders every
    recent report, so the two lists differ (often the store holds none at all)
    and a position-based lookup found nothing.
    """
    all_clicks = (view_clicks or []) + (ta_view_clicks or [])
    if not any(all_clicks):
        raise PreventUpdate

    triggered = ctx.triggered_id
    if not triggered or not isinstance(triggered, dict):
        raise PreventUpdate

    try:
        ta_reports = get_cache().get_all_trading_agent_reports(limit=500)
    except Exception as e:
        logger.warning("Could not load research reports: %s", e)
        raise PreventUpdate

    if triggered.get("type") == "ta-view-btn":
        wanted = str(triggered.get("report", ""))
        report = next((r for r in ta_reports if str(r.get("id")) == wanted), None)
    else:
        symbol = triggered.get("symbol", "")
        report = next((r for r in ta_reports if r.get("symbol") == symbol), None)

    if not report or not report.get("report_text"):
        logger.info("No research report body for %s", triggered)
        raise PreventUpdate

    from models.single_agent import strip_epilogue

    conf_pct = int((report.get("confidence") or 0) * 100)
    title = (f"{report.get('symbol', '?')} — {report.get('decision', '?')} "
             f"({conf_pct}%) — {report.get('trade_date', '')}")
    from services import progress_service as prog
    prog.emit("action", f"Report viewed: {report.get('symbol', '?')} "
                        f"{report.get('trade_date', '')}")
    body = [
        html.Div(
            [
                html.Span(f"Generated: {report.get('created_at', '')}",
                          className="ta-accordion-meta"),
                html.Span(f" | model: {report.get('model_name', '?')}",
                          className="ta-accordion-meta"),
            ],
            style={"marginBottom": "8px"},
        ),
        dcc.Markdown(
            # Machine-read epilogue is stripped for reading; the report keeps
            # its own "Compiled by … Sources: …" footer for transparency.
            strip_epilogue(report.get("report_text", "")),
            className="ta-report-body",
            style={"fontSize": "0.85rem", "lineHeight": "1.6"},
        ),
    ]
    return dash.no_update, True, title, body


# =============================================================================
# STRATEGY DATA LOADING
# =============================================================================


@callback(
    Output("strategy-metrics-store", "data"),
    Output("strategy-evaluations-store", "data"),
    Input("selected-symbols", "data"),
    Input("prediction-store-status", "data"),
)
def load_strategy_data(symbols, _prediction_status):
    """Load strategy metrics and evaluations for selected symbols."""
    if not symbols:
        return [], []

    try:
        cache = get_cache()
        all_metrics = []
        all_evals = []

        for symbol in symbols:
            metrics = cache.get_strategy_metrics(symbol=symbol)
            all_metrics.extend(metrics)

            for strategy_name in ["directional", "confidence_threshold", "ensemble_vote"]:
                evals = cache.get_strategy_evaluations(strategy_name, symbol=symbol, limit=20)
                all_evals.extend(evals)

        # Deduplicate metrics by id
        seen_ids = set()
        unique_metrics = []
        for m in all_metrics:
            mid = m.get("id")
            if mid not in seen_ids:
                seen_ids.add(mid)
                unique_metrics.append(m)

        # Convert datetime objects to strings for JSON serialization
        for m in unique_metrics:
            if m.get("computed_at"):
                m["computed_at"] = str(m["computed_at"])
        for e in all_evals:
            if e.get("evaluated_at"):
                e["evaluated_at"] = str(e["evaluated_at"])
            if e.get("target_date"):
                e["target_date"] = str(e["target_date"])

        return unique_metrics, all_evals

    except Exception as e:
        logger.error(f"Strategy data load error: {e}")
        return [], []


# =============================================================================
# ENSEMBLE CONFIGURATION CALLBACKS
# =============================================================================


@callback(
    Output("ensemble-config-drawer", "is_open"),
    # Gear on the Ensemble signal card, itself rendered by a callback.
    Input("ensemble-config-btn", "n_clicks", allow_optional=True),
    State("ensemble-config-drawer", "is_open"),
    prevent_initial_call=True,
)
def toggle_ensemble_drawer(n_clicks, is_open):
    """Toggle the ensemble configuration drawer."""
    if n_clicks:
        return not is_open
    return is_open


@callback(
    Output("ensemble-config-store", "data"),
    Output({"type": "ensemble-weight-slider", "model": ALL}, "disabled"),
    Output({"type": "ensemble-weight-input", "model": ALL}, "disabled"),
    Output({"type": "ensemble-weight-slider", "model": ALL}, "value"),
    Output({"type": "ensemble-weight-input", "model": ALL}, "value"),
    Output({"type": "ensemble-model-switch", "model": ALL}, "value"),
    Input({"type": "ensemble-model-switch", "model": ALL}, "value"),
    Input({"type": "ensemble-weight-slider", "model": ALL}, "value"),
    Input({"type": "ensemble-weight-input", "model": ALL}, "value"),
    Input("ensemble-reset-btn", "n_clicks"),
    prevent_initial_call=True,
)
def sync_ensemble_config(switches, slider_values, input_values, reset_clicks):
    """Single callback to sync ensemble config from UI controls to store.

    Handles switches, sliders, number inputs, and reset button.
    Returns store data + updated UI state (disabled flags, synced values).
    """
    from config import MODEL

    model_order = ["kronos_mini", "xgboost_shap", "lightgbm", "deberta_sentiment", "trading_agents"]
    triggered = ctx.triggered_id

    # Reset to defaults
    if triggered == "ensemble-reset-btn":
        default_enabled = set(MODEL.ENSEMBLE_DEFAULT_ENABLED)
        default_weights = dict(MODEL.ENSEMBLE_DEFAULT_WEIGHTS)
        store = {
            "enabled_models": list(default_enabled),
            "weights": default_weights,
        }
        slider_disabled = [m not in default_enabled for m in model_order]
        input_disabled = slider_disabled[:]
        slider_out = [default_weights.get(m, 1.0) for m in model_order]
        input_out = slider_out[:]
        switch_out = [m in default_enabled for m in model_order]
        return store, slider_disabled, input_disabled, slider_out, input_out, switch_out

    # Build config from current UI state
    enabled = []
    weights = {}
    slider_out = []
    input_out = []
    slider_disabled = []
    input_disabled = []

    for i, model_id in enumerate(model_order):
        is_on = i < len(switches) and switches[i]
        if is_on:
            enabled.append(model_id)

        # Determine weight: use input if that's what triggered, else slider
        weight = 1.0
        if (isinstance(triggered, dict)
                and triggered.get("type") == "ensemble-weight-input"
                and triggered.get("model") == model_id):
            if i < len(input_values) and input_values[i] is not None:
                weight = float(input_values[i])
        elif (isinstance(triggered, dict)
                and triggered.get("type") == "ensemble-weight-slider"
                and triggered.get("model") == model_id):
            if i < len(slider_values) and slider_values[i] is not None:
                weight = float(slider_values[i])
        else:
            # Not the trigger — keep slider value as-is
            if i < len(slider_values) and slider_values[i] is not None:
                weight = float(slider_values[i])

        weight = round(weight, 1)
        weights[model_id] = weight
        slider_out.append(weight)
        input_out.append(weight)
        slider_disabled.append(not is_on)
        input_disabled.append(not is_on)

    store = {"enabled_models": enabled, "weights": weights}
    return store, slider_disabled, input_disabled, slider_out, input_out, switches


@callback(
    Output("model-signals-store", "data", allow_duplicate=True),
    Input("ensemble-config-store", "data"),
    State("model-signals-store", "data"),
    prevent_initial_call=True,
)
def recompute_ensemble(ensemble_config, signals):
    """Recompute ensemble from existing model results when config changes.

    Runs the EnsembleModel.predict() with updated config against the
    individual model results already in the store. No re-running of
    individual models needed.
    """
    if not signals or not ensemble_config:
        raise PreventUpdate

    from models.ensemble_model import EnsembleModel
    import pandas as pd

    ensemble = EnsembleModel()
    updated = dict(signals)

    for symbol, model_results in updated.items():
        # Collect individual (non-ensemble) results
        other_results = {
            k: v for k, v in model_results.items() if k != "ensemble"
        }
        if not other_results:
            continue

        # Recompute ensemble with new config
        result = ensemble.predict(
            symbol,
            pd.DataFrame(),  # not used by ensemble
            other_results=other_results,
            ensemble_config=ensemble_config,
        )
        updated[symbol] = {**model_results, "ensemble": result.to_dict()}

    return updated


# =============================================================================
# SCHEDULER PANEL (History tab)
# =============================================================================


@callback(
    Output("scheduler-panel-container", "children"),
    # Both live on the Schedule page, so neither exists on other routes.
    Input("scheduler-refresh", "n_intervals", allow_optional=True),
    Input("scheduler-action-status", "data", allow_optional=True),
)
def render_scheduler_panel(_n, _action):
    """Redraw the schedule panel from the database.

    Reads on a timer rather than caching in a Store: the job state is written
    by the scheduler thread (and possibly by another instance), so the browser
    is never the source of truth for it.
    """
    from layouts.scheduler_components import build_scheduler_panel
    from services import scheduler_service

    try:
        return build_scheduler_panel(
            scheduler_service.list_jobs(),
            scheduler_service.recent_runs(limit=8),
            scheduler_service.list_job_types(),
        )
    except Exception as e:
        logger.warning(f"Scheduler panel render failed: {e}")
        return html.Div(f"Scheduler unavailable: {str(e)[:160]}",
                        className="scheduler-empty")


@callback(
    Output({"type": "sched-feedback", "job": MATCH}, "children"),
    Input({"type": "sched-save", "job": MATCH}, "n_clicks"),
    State({"type": "sched-enabled", "job": MATCH}, "value"),
    State({"type": "sched-hour", "job": MATCH}, "value"),
    State({"type": "sched-minute", "job": MATCH}, "value"),
    State({"type": "sched-days", "job": MATCH}, "value"),
    State({"type": "sched-tz", "job": MATCH}, "value"),
    # Only the analysis job renders a symbol box.
    State({"type": "sched-symbols", "job": MATCH}, "value", allow_optional=True),
    prevent_initial_call=True,
)
def save_schedule(n_clicks, enabled, hour, minute, days, tz, symbols):
    """Persist one job's schedule and reschedule it immediately."""
    if not n_clicks:
        raise PreventUpdate

    job_id = ctx.triggered_id["job"]
    try:
        hour = max(0, min(23, int(hour)))
        minute = max(0, min(59, int(minute)))
    except (TypeError, ValueError):
        return html.Span("Time must be a number", className="scheduler-feedback-error")

    fields = {
        "enabled": bool(enabled),
        "hour": hour,
        "minute": minute,
        "days_of_week": days,
        "timezone": tz,
    }
    if symbols is not None:
        cleaned = ",".join(
            s.strip().upper() for s in str(symbols).replace("\n", ",").split(",")
            if s.strip()
        )
        if not cleaned:
            return html.Span("Add at least one symbol",
                             className="scheduler-feedback-error")
        fields["symbols_csv"] = cleaned

    from services import scheduler_service
    if not scheduler_service.update_job(job_id, **fields):
        return html.Span("Save failed", className="scheduler-feedback-error")

    state = "enabled" if enabled else "disabled"
    return html.Span(f"Saved — {state}, {hour:02d}:{minute:02d} {tz}",
                     className="scheduler-feedback-ok")


@callback(
    Output("scheduler-action-status", "data", allow_duplicate=True),
    Output("sched-create-feedback", "children"),
    Input("sched-create", "n_clicks"),
    State("sched-new-kind", "value"),
    State("sched-new-name", "value"),
    State("sched-new-hour", "value"),
    State("sched-new-minute", "value"),
    State("sched-new-days", "value"),
    State("sched-new-tz", "value"),
    State("sched-new-symbols", "value"),
    prevent_initial_call=True,
)
def create_scheduled_job(n_clicks, kind, name, hour, minute, days, tz, symbols):
    """Add a job of any registered operation type."""
    if not n_clicks:
        raise PreventUpdate

    from services import scheduler_service

    if not kind:
        return dash.no_update, html.Span("Pick an operation",
                                         className="scheduler-feedback-error")
    try:
        hour, minute = int(hour), int(minute)
    except (TypeError, ValueError):
        return dash.no_update, html.Span("Time must be a number",
                                         className="scheduler-feedback-error")

    cleaned = ",".join(s.strip().upper()
                       for s in str(symbols or "").replace("\n", ",").split(",")
                       if s.strip())
    needs = {t["kind"]: t["needs_symbols"] for t in scheduler_service.list_job_types()}
    if needs.get(kind) and not cleaned:
        return dash.no_update, html.Span("This operation needs at least one symbol",
                                         className="scheduler-feedback-error")

    job_id = scheduler_service.create_job(
        kind=kind, description=(name or "").strip(), hour=hour, minute=minute,
        days_of_week=days, timezone=tz, symbols_csv=cleaned or None,
        params={"only_trading_days": True},
    )
    if not job_id:
        return dash.no_update, html.Span("Could not create the job",
                                         className="scheduler-feedback-error")
    return ({"created": job_id, "at": datetime.now().isoformat()},
            html.Span(f"Created {job_id}", className="scheduler-feedback-ok"))


@callback(
    Output("scheduler-action-status", "data", allow_duplicate=True),
    Input({"type": "sched-delete", "job": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def delete_scheduled_job(clicks):
    """Remove a job. Its run history stays — that is the record of what ran."""
    if not clicks or not any(c for c in clicks if c):
        raise PreventUpdate

    from services import scheduler_service
    job_id = ctx.triggered_id["job"]
    scheduler_service.delete_job(job_id)
    return {"deleted": job_id, "at": datetime.now().isoformat()}


@callback(
    Output("scheduler-action-status", "data"),
    Input({"type": "sched-run", "job": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def run_job_now(clicks):
    """Trigger a job by hand, off the callback thread.

    An analysis run is ~10 minutes; waiting on it here would pin a callback
    worker for the duration and time the browser out. The thread writes its
    progress to job_runs, which the panel is already polling.
    """
    if not clicks or not any(clicks):
        raise PreventUpdate

    job_id = ctx.triggered_id["job"]
    from services import scheduler_service

    threading.Thread(
        target=scheduler_service.run_job,
        args=(job_id,),
        kwargs={"trigger": "manual"},
        daemon=True,
        name=f"manual-{job_id}",
    ).start()
    return {"started": job_id, "at": datetime.now().isoformat()}


# =============================================================================
# SCHEDULED JOBS (see services/scheduler_service.py)
# =============================================================================


# Startup/shutdown run through the ASGI lifespan (see _lifespan near the Dash
# constructor): exactly once per server process. The old WERKZEUG_RUN_MAIN
# guard was Werkzeug-only — under uvicorn it silently skipped the scheduler
# when DEBUG=true, and the background-callback spawn child started a duplicate
# scheduler when DEBUG=false.
def _startup():
    """One-time server-process startup: progress hydrate, S3, auth, scheduler."""
    _progress_service.hydrate_from_db()
    _init_s3()
    # Auth tables (no-op when AUTH_DATABASE_URL points at the shared Cygnet
    # auth DB, which already has them from CygnetResearchTerminal).
    from services.auth_service import ensure_auth_tables
    ensure_auth_tables()

    # Scheduling now lives in scheduler_service: the schedule is DB-backed and
    # editable from the dashboard, runs are advisory-locked so a deploy overlap
    # cannot double-fire an expensive job, and both the analysis and the
    # evaluation are jobs rather than one hardcoded evaluation trigger.
    from services import scheduler_service
    scheduler_service.start()


def _shutdown():
    """Lifespan shutdown: stop the scheduler without waiting on running jobs."""
    from services import scheduler_service
    scheduler_service.shutdown()


# =============================================================================
# CONFIRMATION MODAL CALLBACKS
# =============================================================================




@callback(
    Output("ai-json-modal", "is_open"),
    Output("ai-json-body", "children"),
    # Rendered inside the Overall tab body.
    Input("ai-json-view-btn", "n_clicks", allow_optional=True),
    State("ai-analysis-store", "data"),
    prevent_initial_call=True,
)
def view_ai_json(n_clicks, ai_analysis):
    """Open the raw AI Report payload in a readable modal."""
    if not n_clicks or not ai_analysis:
        raise PreventUpdate
    return True, json.dumps(ai_analysis, indent=2, default=str)


@callback(
    Output("download-ai-json", "data"),
    Input("ai-json-download-btn", "n_clicks"),
    State("ai-analysis-store", "data"),
    prevent_initial_call=True,
)
def download_ai_json(n_clicks, ai_analysis):
    """Download the AI Report payload as a .json file."""
    if not n_clicks or not ai_analysis:
        raise PreventUpdate
    as_of = (ai_analysis.get("as_of") or datetime.now().strftime("%Y-%m-%d"))[:10]
    return dcc.send_string(
        json.dumps(ai_analysis, indent=2, default=str),
        f"ai_report_{as_of}.json",
    )


@callback(
    Output("run-article-preview", "children"),
    Input("run-modal", "is_open"),
    Input("run-lookback", "value"),
    Input("run-date-picker", "date"),
    State("selected-symbols", "data"),
    prevent_initial_call=True,
)
async def preview_ai_report_articles(is_open, lookback, ai_date, symbols):
    """Fetch and show article availability for the chosen window BEFORE
    generation — the same point-in-time fetch generation uses, so the
    preview counts are the counts, not an estimate from the news store.

    Async: fires on modal open, so the live vendor fetch runs in a worker
    thread instead of freezing the UI thread pool."""
    if not is_open or not symbols:
        raise PreventUpdate

    from utils.trading_calendar import get_next_trading_day
    from services.news_window import fetch_point_in_time_news
    from datetime import date as date_cls

    as_of = str(ai_date)[:10] if ai_date else get_next_trading_day(date_cls.today()).isoformat()
    lookback_days = int(lookback or 7)

    def _count(sym):
        try:
            return len(fetch_point_in_time_news(sym, as_of, lookback_days=lookback_days))
        except Exception as e:
            logger.debug(f"article preview fetch failed for {sym}: {e}")
            return 0

    sem = asyncio.Semaphore(APP.NEWS_FETCH_CONCURRENCY)

    async def _count_guarded(sym):
        async with sem:
            return await asyncio.to_thread(_count, sym)

    counts = await asyncio.gather(*(_count_guarded(s) for s in symbols))

    rows, total = [], 0
    for sym, n in zip(symbols, counts):
        total += n
        rows.append(html.Tr([
            html.Td(sym),
            html.Td(str(n), style={"textAlign": "right"}),
        ]))

    return html.Div([
        html.Div([
            html.Strong(f"{total} articles"),
            html.Span(f" in the {lookback_days}-day window ending {as_of}",
                      style={"color": "var(--text-secondary)"}),
        ], className="mb-1"),
        dbc.Table(
            [html.Thead(html.Tr([html.Th("Symbol"), html.Th("Articles", style={"textAlign": "right"})])),
             html.Tbody(rows)],
            bordered=False, color="dark", size="sm", className="mb-0",
        ),
    ])


@callback(
    Output("run-date-mode-label", "children"),
    Input("run-date-picker", "date"),
    prevent_initial_call=True,
)
def update_predict_date_label(date_str):
    """Show Live or Backtest mode, and the data cutoff the target implies."""
    from datetime import date as date_cls

    from utils.trading_calendar import resolve_target_and_cutoff

    if not date_str:
        return "", {"display": "none"}
    target, cutoff = resolve_target_and_cutoff(date_str[:10])
    selected = target
    today = date_cls.today()
    if selected >= today:
        return [
            html.I(className="bi bi-broadcast me-1"),
            f"Live — targeting {target} close, data through {cutoff}",
        ], {"fontSize": "0.85rem", "color": "var(--positive)"}
    else:
        days_back = (today - selected).days
        return [
            html.I(className="bi bi-clock-history me-1"),
            f"Backtest ({days_back}d ago) — targeting {target} close, "
            f"data truncated to {cutoff}",
        ], {"fontSize": "0.85rem", "color": "var(--warning, #ffc107)"}


@callback(
    Output("run-modal", "is_open"),
    Output("run-data-summary", "children"),
    Output("run-ensemble-summary", "children"),
    Input("run-analysis-btn", "n_clicks"),
    Input("home-run-btn", "n_clicks", allow_optional=True),
    Input("run-cancel-btn", "n_clicks"),
    Input("run-confirm-btn", "n_clicks"),
    State("stock-data-store", "data"),
    State("selected-symbols", "data"),
    State("ensemble-config-store", "data"),
    State("news-data-store", "data"),
    State("run-modal", "is_open"),
    prevent_initial_call=True,
)
def toggle_run_modal(open_clicks, home_clicks, cancel_clicks, confirm_clicks,
                     stock_data, symbols, ensemble_config, news_data, is_open):
    """Open or close the single run dialog, with a summary of the input data.

    Two openers: the toolbar button and the Home call to action. Both land on
    the same dialog, so there is one place where a run is configured.
    """
    triggered = ctx.triggered_id
    no_update_body = (dash.no_update, dash.no_update)
    openers = ("run-analysis-btn", "home-run-btn")

    if triggered not in openers + ("run-cancel-btn", "run-confirm-btn"):
        raise PreventUpdate

    if triggered in ("run-cancel-btn", "run-confirm-btn"):
        return False, *no_update_body

    if triggered in openers:
        # Both openers fire this callback when they mount, with their own
        # n_clicks still None. Checking "either has clicks" is not enough:
        # the toolbar button never unmounts, so its count survives, and
        # arriving on Home would then open the dialog by itself. Only the
        # button that actually triggered may open it.
        if not {"run-analysis-btn": open_clicks,
                "home-run-btn": home_clicks}.get(triggered):
            raise PreventUpdate
        if not symbols:
            # Opening onto an explanation beats the old behaviour, which was
            # for the button to do nothing at all with an empty watchlist.
            return True, html.Div(
                [
                    html.Div("No symbols selected", className="empty-state-title"),
                    html.Div("Add symbols from the Watchlist button in the "
                             "toolbar, then run.", className="empty-state-note"),
                ],
                className="empty-state",
            ), dash.no_update

        stock_data = stock_data or {}
        ensemble_config = ensemble_config or {}
        enabled = ensemble_config.get("enabled_models", [])
        weights = ensemble_config.get("weights", {})

        # Symbol data summary table
        articles_by_symbol = (news_data or {}).get("articles_by_symbol", {})
        sym_rows = []
        for sym in symbols:
            sym_data = stock_data.get(sym, {})
            data_points = "N/A"
            date_range = "N/A"
            if sym_data.get("prices"):
                try:
                    df = pd.read_json(StringIO(sym_data["prices"]))
                    data_points = str(len(df))
                    if not df.empty and "Date" in df.columns:
                        date_range = (
                            f"{str(df['Date'].min())[:10]} to "
                            f"{str(df['Date'].max())[:10]}"
                        )
                    elif not df.empty:
                        date_range = (
                            f"{str(df.index.min())[:10]} to "
                            f"{str(df.index.max())[:10]}"
                        )
                except Exception:
                    pass
            from_cache = sym_data.get("from_cache", False)
            source = "Cached" if from_cache else "Live"
            news_count = len(articles_by_symbol.get(sym, []))
            sym_rows.append(html.Tr([
                html.Td(sym), html.Td(data_points),
                html.Td(date_range), html.Td(str(news_count)),
                html.Td(source),
            ]))

        data_summary = html.Div([
            html.H6("Stock Data", className="mb-3"),
            dbc.Table(
                [
                    html.Thead(html.Tr([
                        html.Th("Symbol"), html.Th("Bars"),
                        html.Th("Date Range"), html.Th("Articles"),
                        html.Th("Source"),
                    ])),
                    html.Tbody(sym_rows),
                ],
                bordered=True, color="dark", size="sm",
            ),
        ])

        # Ensemble config summary (compact inline)
        all_models = ["kronos_mini", "xgboost_shap", "lightgbm",
                      "deberta_sentiment", "trading_agents"]
        ens_items = []
        for m in all_models:
            display = MODEL_DISPLAY.get(m, m)
            in_ens = m in enabled
            w = weights.get(m, 1.0)
            style = (
                {"color": "var(--positive)"}
                if in_ens
                else {"color": "var(--text-muted)", "textDecoration": "line-through"}
            )
            ens_items.append(
                html.Span(f"{display} (w={w:.1f})", style=style, className="me-3")
            )

        ens_summary = html.Div([
            html.Span("Enabled: ", style={"color": "var(--text-secondary)"}),
            *ens_items,
        ], style={"fontSize": "0.85rem"})

        return True, data_summary, ens_summary

    return is_open, *no_update_body


# =============================================================================
# FULL ANALYSIS MODAL + RECOMMENDATIONS
# =============================================================================




@callback(
    Output("run-models-section", "style"),
    Output("run-report-section", "style"),
    Output("run-recs-section", "style"),
    Output("run-scope-hint", "children"),
    Input("run-scope", "value"),
)
def apply_run_scope(scope):
    """Show only the controls the chosen scope actually uses.

    Hidden rather than unmounted: the callbacks that read these values take
    them as State, and an unmounted State would break the run.
    """
    from layouts.modals import RUN_SCOPES

    scope = scope or "full"
    show, hide = {}, {"display": "none"}
    hint = next((d for v, _, d in RUN_SCOPES if v == scope), "")
    return (
        show if scope in ("models", "full") else hide,
        show if scope in ("report", "full") else hide,
        show if scope in ("report", "full") else hide,
        hint,
    )


@callback(
    Output("full-analysis-requested", "data"),
    Input("run-confirm-btn", "n_clicks"),
    State("run-scope", "value"),
    prevent_initial_call=True,
)
def set_full_analysis_flag(n_clicks, scope):
    """Arm the synthesis stage, but only for a full-pipeline run."""
    if n_clicks and (scope or "full") == "full":
        return True
    raise PreventUpdate


# Shared with the scheduled runner (scripts/daily_analysis.py) so the UI and
# cron paths merge research identically.
from services.analysis_runner import (  # noqa: E402
    merge_research_into_analysis as _merge_research_into_analysis,
)


@callback(
    Output("recommendations-store", "data"),
    Input("ai-analysis-store", "data"),
    Input("model-signals-store", "data"),
    State("full-analysis-requested", "data"),
    State("selected-symbols", "data"),
    prevent_initial_call=True,
)
async def generate_recommendations_callback(ai_analysis, model_signals, requested, symbols):
    """Generate recommendations when their inputs are ready.

    Fires for the Full Analysis flow (requested flag) AND for AI Report runs
    that asked for recommendations (recs_request on the store, carrying the
    evidence basis: research+signals / news+signals / signals).

    Async: the ~1-minute Luna synthesis and the Postgres persist loop run in
    worker threads instead of pinning a callback slot.
    """
    basis = (ai_analysis or {}).get("recs_request")
    if not requested and not basis:
        raise PreventUpdate

    if not ai_analysis or ai_analysis.get("failed"):
        raise PreventUpdate
    basis = basis or "news+signals"  # Full Analysis default

    valid_signals = {
        k: v for k, v in (model_signals or {}).items()
        if k != "_meta" and isinstance(v, dict)
    }
    if not valid_signals:
        if basis == "signals":
            # Predictions-only synthesis with no predictions is a no-op —
            # say so instead of silently doing nothing.
            from services import progress_service as _prog
            _prog.emit("error", "Recommendations (predictions only) skipped — "
                                "no model predictions in this session. Run Predict first.")
            raise PreventUpdate
        # Text-based bases can proceed without signals; Luna sees the gap.
        valid_signals = {}

    # Recommendations explicitly turned off for this run: the Full Analysis
    # flow still sets the requested flag, so honor the store's opt-out marker
    # (and close the progress run once predictions have landed).
    if ai_analysis.get("recs_off"):
        if requested and valid_signals:
            from services import progress_service as _prog
            _prog.finish_run("Full Analysis complete (recommendations off)")
        raise PreventUpdate

    # As-of date travels with the AI report (set by the Full Analysis picker)
    trade_date = (ai_analysis.get("as_of") or datetime.now().strftime("%Y-%m-%d"))[:10]

    # Check persistent cache
    rec_data_hash = None
    if _s3_available:
        try:
            from services import persistence_service as ps
            rec_data_hash = ps.compute_data_hash({
                "ai_analysis": ai_analysis,
                "model_signals": valid_signals,
                "symbols": sorted(symbols or []),
            })
            cached = await asyncio.to_thread(
                ps.get_cached_recommendation, trade_date, rec_data_hash)
            if cached:
                logger.info("Recommendation cache hit (Postgres)")
                cached["from_cache"] = True
                return cached
        except Exception as e:
            logger.debug(f"Recommendation cache check failed: {e}")

    # Full Analysis no longer runs the shallow per-symbol pass — its research
    # text lives on the trading_agents signal. Backfill it into the analysis
    # dict so Luna synthesizes from the SAME report the user reads, and stamp
    # the basis honestly (History records what evidence Luna actually saw).
    if basis != "signals":
        ai_analysis, backfilled = _merge_research_into_analysis(
            ai_analysis, valid_signals, symbols)
        if backfilled and basis == "news+signals":
            basis = "research+signals"

    logger.info("Both stores ready — generating recommendations")
    from services import progress_service as prog
    from config import MODEL as _MODEL_CFG
    prog.emit("luna", f"Luna ({_MODEL_CFG.RECOMMENDATIONS_MODEL}) synthesizing "
                      f"{len(symbols or [])} symbols — reasoning may take a minute…")
    llm = get_llm()
    _luna_t0 = time.time()
    result = await asyncio.to_thread(
        llm.generate_recommendations, ai_analysis, model_signals, symbols or [],
        basis=basis,
        model_override=ai_analysis.get("recs_model"))
    _luna_elapsed = time.time() - _luna_t0

    if result:
        result["generated_at"] = datetime.now().isoformat()
        result["as_of"] = trade_date
        actions = {s: v.get("action") for s, v in (result.get("by_symbol") or {}).items()}
        prog.emit("luna", f"Luna finished in {_luna_elapsed:.0f}s — "
                          + (", ".join(f"{s}={a}" for s, a in actions.items()) or "done"))
        prog.finish_run("Full Analysis complete")

        # Score the synthesis like any model: persist each per-symbol action
        # as a prediction under "recommendation_synthesis" so the existing
        # evaluation loop, History tab, and Scoreboard pick it up unchanged.
        # Conviction labels map to nominal confidences purely for display —
        # our backtests showed they carry no calibration signal, so the raw
        # label is kept in details as the honest record.
        _conv2conf = {"HIGH": 0.8, "MEDIUM": 0.6, "LOW": 0.4}

        def _persist_all():
            """Blocking Postgres writes — one worker thread for the batch."""
            for sym, rec in (result.get("by_symbol") or {}).items():
                action = (rec.get("action") or "").upper()
                if action not in ("BUY", "SELL", "HOLD"):
                    continue
                try:
                    get_cache().store_prediction(
                        sym, "recommendation_synthesis",
                        {
                            "decision": action,
                            "confidence": _conv2conf.get(
                                str(rec.get("conviction", "")).upper()),
                            "up_probability": None,
                            "details": {
                                "synthesis_model": result.get("model_used"),
                                "basis": result.get("basis"),
                                "conviction": rec.get("conviction"),
                                "key_level": rec.get("key_level"),
                                "change_trigger": rec.get("change_trigger"),
                                "reasoning": (rec.get("reasoning") or "")[:400],
                            },
                        },
                        prediction_date_str=trade_date,
                    )
                except Exception as e:
                    logger.warning(
                        f"recommendation prediction persist failed for {sym}: {e}")

            # Store to Postgres
            if _s3_available and rec_data_hash:
                try:
                    from services import persistence_service as ps
                    from config import MODEL
                    ps.store_recommendation(
                        trade_date=trade_date,
                        symbols=symbols or [],
                        input_data_hash=rec_data_hash,
                        result=result,
                        model_used=MODEL.RECOMMENDATIONS_MODEL,
                        provider_used=MODEL.RECOMMENDATIONS_PROVIDER,
                    )
                except Exception as e:
                    logger.warning(f"Failed to persist recommendation: {e}")

        await asyncio.to_thread(_persist_all)

        return result

    prog.emit("error", "Luna returned empty response — synthesis failed")
    prog.finish_run("Full Analysis finished with errors")
    return {"error": "Recommendations model unavailable or returned empty response",
            "generated_at": datetime.now().isoformat()}


@callback(
    Output("full-analysis-requested", "data", allow_duplicate=True),
    Input("recommendations-store", "data"),
    prevent_initial_call=True,
)
def reset_full_analysis_flag(rec_data):
    """Reset the flag after recommendations complete."""
    if rec_data:
        return False
    raise PreventUpdate


# =============================================================================
# RUN SERVER
# =============================================================================

if __name__ == "__main__":
    # Single worker is mandatory: torch/MPS models, per-process caches, and
    # the APScheduler all live in this process. Concurrency comes from the
    # event loop + threadpools, never from more workers.
    import uvicorn

    uvicorn.run(
        "app:server",
        host=APP.HOST,
        port=APP.PORT,
        workers=1,
        reload=APP.DEBUG,
        # plotly_cloud's devtool hook applies nest_asyncio at import, which
        # cannot patch uvloop -- pin the stock asyncio loop.
        loop="asyncio",
    )

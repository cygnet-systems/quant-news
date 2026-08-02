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
    create_recommendation_banner,
    create_sentiment_breakdown,
    create_symbol_tag,
    create_top_headlines,
    create_news_quick_stats,
)
from layouts.signal_components import create_signal_cards
from layouts.strategy_components import create_strategy_section
from layouts.main_layout import create_layout
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
            content = _json_report_to_markdown(parsed).encode("utf-8")
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
    """Keep the last 5 distinct symbol GROUPS for one-click restore.

    Triggered by the data store (not the raw selection) so only symbols that
    actually returned data are recorded — a typo'd ticker with zero results
    never pollutes Recent. Subset-merge so incremental adds grow one entry
    instead of littering the list with partial prefixes (REX / REX+WGO /
    REX+WGO+IOVA...). Clearing or trimming a selection never destroys a
    recorded group.
    """
    symbols = [s for s in (symbols or [])
               if s and ((stock_data or {}).get(s) or {}).get("prices")]
    if not symbols:
        raise PreventUpdate
    new = sorted(symbols)
    groups = [list(g) for g in (groups or [])]
    # Already represented by an identical or superset group → nothing to do
    if any(set(new) <= set(g) for g in groups):
        raise PreventUpdate
    # Drop groups the new one supersedes, prepend, cap at 5
    groups = [g for g in groups if not set(g) < set(new)]
    return [new] + groups[:4]


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
    Input("indicator-toggles", "value"),
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
    Input("report-confirm-btn", "n_clicks"),
    Input("ai-retry-btn", "n_clicks"),
    Input("full-analysis-confirm-btn", "n_clicks"),
    State("news-data-store", "data"),
    State("selected-symbols", "data"),
    State("stock-data-store", "data"),
    State("fa-date-picker", "date"),
    State("ai-report-date", "date"),
    State("ai-report-lookback", "value"),
    State("ai-report-model", "value"),
    State("ai-report-type", "value"),
    State("ai-report-include-research", "value"),
    State("ai-report-recs", "value"),
    State("ai-report-recs-model", "value"),
    State("fa-lookback", "value"),
    State("fa-model", "value"),
    State("fa-type", "value"),
    State("fa-recs", "value"),
    State("fa-recs-model", "value"),
    prevent_initial_call=True,
)
async def generate_ai_analysis(n_clicks, retry_clicks, full_clicks, news_data, symbols,
                               stock_data, fa_date, ai_date, ai_lookback, ai_model, ai_type,
                               ai_research, ai_recs, ai_recs_model,
                               fa_lookback, fa_model, fa_type, fa_recs, fa_recs_model):
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

    if not (n_clicks or retry_clicks or full_clicks) or not news_data or not news_data.get("articles_by_symbol"):
        raise PreventUpdate

    # As-of date: the Full Analysis picker when that flow triggered this,
    # else the AI Report modal's own date (empty = today).
    today = date_cls.today()
    as_of = None
    is_full = ctx.triggered_id == "full-analysis-confirm-btn"
    if is_full and fa_date:
        as_of = str(fa_date)[:10]
    elif not is_full and ai_date:
        as_of = str(ai_date)[:10]
    if as_of:
        as_of_d = date_cls.fromisoformat(as_of)
    else:
        # No date chosen: the session the report is FOR — next trading day
        # (matches the picker default; plain "today" could be a weekend).
        from utils.trading_calendar import get_next_trading_day
        as_of_d = get_next_trading_day(today)
    is_backtest = as_of_d < today
    as_of_str = as_of_d.isoformat()

    # Parameters come from the modal that actually triggered this run — each
    # flow owns its own visible controls (the Full Analysis flow used to
    # silently read the AI Report modal's state).
    if is_full:
        lookback_days = int(fa_lookback or 7)
        report_model = fa_model or "gpt-5.6-luna"
        include_thesis_flag = (fa_type or "thesis") != "standard"
        recs_mode = fa_recs or "auto"
        recs_model_val = fa_recs_model or "claude-sonnet-5"
    else:
        lookback_days = int(ai_lookback or 7)
        report_model = ai_model or "gpt-5.6-luna"
        include_thesis_flag = (ai_type or "thesis") != "standard"
        recs_mode = ai_recs or "auto"
        recs_model_val = ai_recs_model or "claude-sonnet-5"
    report_provider = "openai" if report_model.startswith("gpt-") else "anthropic"
    # Research reports only for the AI Report flow proper — Full Analysis
    # already generates them through the prediction pipeline, and doubling
    # the run would double the LLM spend for identical output.
    include_research = bool(ai_research) and not is_full

    from services import progress_service as prog
    if is_full:
        prog.start_run(f"Full Analysis — {len(symbols or [])} symbols, as-of {as_of_str}"
                       + (" (backtest)" if is_backtest else ""))
    prog.emit("ai", f"AI Report starting for {', '.join(symbols or [])}")
    prog.emit("action", f"Options: model={report_model}, window={lookback_days}d, "
                        f"type={'thesis' if include_thesis_flag else 'standard'}, "
                        f"research={'on' if include_research else 'pipeline' if is_full else 'off'}, "
                        f"recs={recs_mode} ({recs_model_val}), as-of {as_of_str}")

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
    prog.emit("news", f"News window {lookback_days}d ending {as_of_str}: "
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
            # Live quote/info only when analyzing as of today — a backtest
            # must not see current prices. Static identity is kept either way.
            info = get_stock_info(symbol)
            if is_backtest:
                enriched["info"] = {
                    "name": info.name, "sector": info.sector, "industry": info.industry,
                }
                # Period metrics/signals in the store are computed on current
                # data — drop them for backtests; validated blocks replace them.
                enriched["metrics"] = {}
                enriched["signals"] = {}
            else:
                enriched["info"] = {
                    "name": info.name,
                    "sector": info.sector,
                    "industry": info.industry,
                    "market_cap": info.market_cap,
                    "current_price": info.current_price,
                    "previous_close": info.previous_close,
                    "day_change_percent": info.day_change_percent,
                    "volume": info.volume,
                    "avg_volume": info.avg_volume,
                    "fifty_two_week_high": info.fifty_two_week_high,
                    "fifty_two_week_low": info.fifty_two_week_low,
                    "pe_ratio": info.pe_ratio,
                    "dividend_yield": info.dividend_yield,
                }
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
                block = compute_peer_relative_strength(
                    symbol, peers, as_of=as_of_str if is_backtest else None)
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
        aio_tasks.append(_run_task(
            "overall", None, llm.summarize_news_structured, overall_articles, symbols or [],
            stock_data=enriched_stock_data,
            as_of_date=as_of_str,
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
    Input("symbol-tabs", "active_tab"),
    prevent_initial_call=True,
)
def track_active_tab(active_tab):
    """Remember the user's active tab so re-renders don't reset it."""
    if not active_tab:
        raise PreventUpdate
    return active_tab


def _restore_tab(stored: str | None, available: list[str], default: str) -> str:
    """Return the stored tab if it still exists, else the default."""
    return stored if stored in available else default


@callback(
    Output("symbol-tabs-container", "children"),
    Input("news-data-store", "data"),
    Input("ai-analysis-store", "data"),
    Input("selected-symbols", "data"),
    Input("model-signals-store", "data"),
    Input("strategy-metrics-store", "data"),
    Input("strategy-evaluations-store", "data"),
    Input("report-history-store", "data"),
    Input("recommendations-store", "data"),
    State("active-tab-store", "data"),
)
def update_symbol_tabs(
    news_data, ai_analysis, symbols, model_signals,
    strategy_metrics, strategy_evaluations, report_history,
    recommendations, stored_tab,
):
    """Build dynamic tabs based on selected symbols.

    Creates an "Overall" tab (always first) plus one tab per symbol. The
    user's active tab is preserved across re-renders (background store
    updates used to reset the view to Overall).
    """
    # Handle empty state — still show History tab
    if not symbols:
        history_content = _build_history_tab_layout(report_history or {})
        return dbc.Tabs(
            [
                dbc.Tab(
                    create_overview_empty_state(),
                    label="Overview",
                    tab_id="tab-overview-empty",
                    className="context-tab",
                ),
                dbc.Tab(
                    history_content,
                    label="History",
                    tab_id="tab-history",
                    className="context-tab",
                ),
            ],
            id="symbol-tabs",
            active_tab=_restore_tab(stored_tab, ["tab-overview-empty", "tab-history"],
                                    "tab-overview-empty"),
            className="symbol-tabs",
        )

    # Show loading state if symbols selected but no news data yet
    if not news_data or not news_data.get("articles_by_symbol"):
        history_content = _build_history_tab_layout(report_history or {})
        loading = _create_loading_state(symbols)
        return dbc.Tabs(
            [
                dbc.Tab(loading, label="Loading...", tab_id="tab-loading", className="context-tab"),
                dbc.Tab(history_content, label="History", tab_id="tab-history", className="context-tab"),
            ],
            id="symbol-tabs",
            active_tab=_restore_tab(stored_tab, ["tab-loading", "tab-history"], "tab-loading"),
            className="symbol-tabs",
        )

    articles_by_symbol = news_data.get("articles_by_symbol", {}) if news_data else {}
    analysis_by_symbol = ai_analysis.get("by_symbol", {}) if ai_analysis else {}
    overall_analysis = ai_analysis.get("overall", {}) if ai_analysis else {}

    # Build tabs list
    tabs = []

    # --- Overall Tab (always first) ---
    all_articles = []
    for sym_articles in articles_by_symbol.values():
        all_articles.extend(sym_articles or [])

    overall_content = _build_overall_tab_content(
        articles_by_symbol=articles_by_symbol,
        analysis_by_symbol=analysis_by_symbol,
        overall_analysis=overall_analysis,
        symbols=symbols,
        ai_failed=bool(ai_analysis and ai_analysis.get("failed")),
        recommendations=recommendations,
    )

    tabs.append(
        dbc.Tab(
            overall_content,
            label="Overall",
            tab_id="tab-overall",
            className="context-tab",
        )
    )

    # --- Per-Symbol Tabs ---
    for symbol in symbols:
        sym_articles = articles_by_symbol.get(symbol, [])
        sym_analysis = analysis_by_symbol.get(symbol, {})
        sym_signals = (model_signals or {}).get(symbol, {})

        # Filter strategy data for this symbol
        sym_metrics = [
            m for m in (strategy_metrics or [])
            if m.get("symbol") == symbol
        ]
        sym_evals = [
            e for e in (strategy_evaluations or [])
            if e.get("symbol") == symbol
        ]

        sym_recs = (recommendations or {}).get("by_symbol", {}).get(symbol, {})

        tab_content = _build_tab_content(
            articles=sym_articles,
            analysis=sym_analysis,
            symbols=[symbol],
            is_overall=False,
            model_signals=sym_signals,
            strategy_metrics=sym_metrics,
            strategy_evaluations=sym_evals,
            ai_failed=bool(ai_analysis and ai_analysis.get("failed")),
            recommendations=sym_recs,
        )

        tabs.append(
            dbc.Tab(
                tab_content,
                label=symbol,
                tab_id=f"tab-{symbol}",
                className="context-tab",
            )
        )

    # --- History Tab (TradingAgents research reports) ---
    history_content = _build_history_tab_layout(report_history or {})
    tabs.append(
        dbc.Tab(
            history_content,
            label="History",
            tab_id="tab-history",
            className="context-tab",
        )
    )

    available_ids = ["tab-overall"] + [f"tab-{s}" for s in symbols] + ["tab-history"]
    return dbc.Tabs(
        tabs,
        id="symbol-tabs",
        active_tab=_restore_tab(stored_tab, available_ids, "tab-overall"),
        className="symbol-tabs",
    )


def _create_recommendations_section(rec_data: dict, symbol: str | None = None) -> html.Div | None:
    """Render recommendations from the synthesis model.

    When symbol is None, renders the overall portfolio view.
    When symbol is provided, renders per-symbol view (rec_data is already the symbol's dict).
    """
    if not rec_data:
        return None

    ACCENT = "#7B61FF"

    if symbol is None:
        # Overall view
        overall = rec_data.get("overall", {})
        if not overall:
            return None

        model_used = rec_data.get("model_used", "")

        children = [
            html.Div(
                [
                    html.I(className="bi bi-lightning-fill", style={"color": ACCENT, "marginRight": "6px"}),
                    html.Span("Recommendations", style={"color": ACCENT}),
                    html.Span(
                        f" — {model_used}" if model_used else "",
                        style={"color": "var(--text-muted)", "fontSize": "0.75rem", "marginLeft": "8px"},
                    ),
                    html.Span(
                        {"research+signals": "based on research + predictions",
                         "news+signals": "based on news analysis + predictions",
                         "signals": "based on predictions only",
                         }.get(rec_data.get("basis"), ""),
                        style={"color": "var(--text-muted)", "fontSize": "0.72rem",
                               "marginLeft": "auto", "fontStyle": "italic"},
                    ),
                ],
                className="section-title",
                style={"display": "flex", "alignItems": "center"},
            ),
        ]

        if overall.get("portfolio_action"):
            children.append(
                html.Div(
                    overall["portfolio_action"],
                    className="recommendations-action-banner",
                    style={
                        "borderLeft": f"3px solid {ACCENT}",
                        "padding": "8px 12px",
                        "marginBottom": "10px",
                        "fontWeight": "600",
                        "fontSize": "0.9rem",
                        "backgroundColor": "rgba(123, 97, 255, 0.08)",
                        "borderRadius": "4px",
                    },
                )
            )

        if overall.get("summary"):
            children.append(
                html.Div(
                    overall["summary"],
                    style={"fontSize": "0.85rem", "lineHeight": "1.5", "marginBottom": "10px",
                           "color": "var(--text-secondary)"},
                )
            )

        conflicts = overall.get("key_conflicts", [])
        if conflicts:
            conflict_items = []
            for conflict in conflicts:
                conflict_items.append(
                    html.Div(
                        [
                            html.I(className="bi bi-exclamation-triangle me-2",
                                   style={"color": "#FFB020"}),
                            html.Span(conflict, style={"fontSize": "0.82rem"}),
                        ],
                        className="conflict-card",
                    )
                )
            children.append(html.Div(conflict_items, className="mb-2"))

        if overall.get("risk_assessment"):
            children.append(
                html.Div(
                    [
                        html.Span("Risk: ", style={"fontWeight": "600", "color": "var(--text-muted)", "fontSize": "0.8rem"}),
                        html.Span(overall["risk_assessment"], style={"fontSize": "0.82rem", "color": "var(--text-secondary)"}),
                    ],
                    style={"marginBottom": "8px"},
                )
            )

        luna_watch = overall.get("watch_items") or []
        if isinstance(luna_watch, list) and luna_watch:
            children.append(html.Div(
                [
                    html.Span("Watch: ", style={"fontWeight": "600",
                                                "color": "var(--text-muted)",
                                                "fontSize": "0.8rem"}),
                    html.Ul(
                        [html.Li(str(w)) for w in luna_watch[:4]],
                        className="watch-items-list",
                        style={"display": "inline-block", "margin": 0},
                    ),
                ],
                style={"marginBottom": "8px"},
            ))

        # Per-symbol recommendations table
        by_symbol = rec_data.get("by_symbol", {})
        if by_symbol:
            rows = []
            for sym, sym_rec in by_symbol.items():
                action = sym_rec.get("action", "HOLD")
                conviction = sym_rec.get("conviction", "")
                action_color = (
                    "var(--positive)" if action == "BUY" else
                    "var(--negative)" if action == "SELL" else
                    "var(--text-secondary)"
                )
                # Full reasoning (the old 100-char cut hid the actual case
                # for the call) + the level/trigger that make it actionable.
                reason_cell = [html.Div(sym_rec.get("reasoning", ""))]
                if sym_rec.get("key_level"):
                    reason_cell.append(html.Div(
                        f"Key level: {sym_rec['key_level']}",
                        className="rec-key-level",
                    ))
                if sym_rec.get("change_trigger"):
                    reason_cell.append(html.Div(
                        f"Flips on: {sym_rec['change_trigger']}",
                        className="rec-change-trigger",
                    ))
                rows.append(html.Tr([
                    html.Td(sym, style={"fontWeight": "600"}),
                    html.Td(action, style={"color": action_color, "fontWeight": "600"}),
                    html.Td(conviction, style={"fontSize": "0.8rem"}),
                    html.Td(
                        reason_cell,
                        style={"fontSize": "0.8rem", "color": "var(--text-secondary)"},
                    ),
                ]))
            children.append(
                dbc.Table(
                    [
                        html.Thead(html.Tr([
                            html.Th("Symbol"), html.Th("Action"),
                            html.Th("Conviction"), html.Th("Reasoning"),
                        ])),
                        html.Tbody(rows),
                    ],
                    bordered=True, color="dark", size="sm",
                    style={"fontSize": "0.82rem"},
                )
            )

        return html.Div(children, className="recommendations-section")

    else:
        # Per-symbol view — rec_data is already the symbol's dict
        action = rec_data.get("action", "")
        if not action:
            return None

        conviction = rec_data.get("conviction", "")
        reasoning = rec_data.get("reasoning", "")
        conflicts = rec_data.get("conflicts", [])
        model_notes = rec_data.get("model_notes", "")

        action_color = (
            "var(--positive)" if action == "BUY" else
            "var(--negative)" if action == "SELL" else
            "var(--text-secondary)"
        )

        children = [
            html.Div(
                [
                    html.I(className="bi bi-lightning-fill", style={"color": ACCENT, "marginRight": "6px"}),
                    html.Span("Recommendation", style={"color": ACCENT}),
                ],
                className="section-title",
                style={"display": "flex", "alignItems": "center"},
            ),
            html.Div(
                [
                    html.Span(
                        action,
                        className="action-badge",
                        style={
                            "color": action_color,
                            "backgroundColor": f"{action_color}22" if action != "HOLD" else "var(--bg-tertiary)",
                            "padding": "2px 10px",
                            "borderRadius": "4px",
                            "fontWeight": "700",
                            "fontSize": "0.85rem",
                            "marginRight": "8px",
                        },
                    ),
                    html.Span(
                        conviction,
                        className="conviction-badge",
                        style={
                            "color": "var(--text-muted)",
                            "fontSize": "0.8rem",
                            "fontWeight": "600",
                        },
                    ) if conviction else html.Span(),
                ],
                style={"marginBottom": "8px"},
            ),
        ]

        if reasoning:
            children.append(
                html.Div(
                    reasoning,
                    style={"fontSize": "0.82rem", "lineHeight": "1.5", "marginBottom": "8px",
                           "color": "var(--text-secondary)"},
                )
            )

        if rec_data.get("key_level"):
            children.append(html.Div(
                f"Key level: {rec_data['key_level']}",
                className="rec-key-level",
            ))
        if rec_data.get("change_trigger"):
            children.append(html.Div(
                f"Flips on: {rec_data['change_trigger']}",
                className="rec-change-trigger",
            ))

        if conflicts:
            for conflict in conflicts:
                children.append(
                    html.Div(
                        [
                            html.I(className="bi bi-exclamation-triangle me-2",
                                   style={"color": "#FFB020"}),
                            html.Span(conflict, style={"fontSize": "0.8rem"}),
                        ],
                        className="conflict-card",
                    )
                )

        if model_notes:
            children.append(
                html.Div(
                    [
                        html.I(className="bi bi-info-circle me-2",
                               style={"color": "var(--text-muted)"}),
                        html.Span(model_notes, style={"fontSize": "0.8rem", "color": "var(--text-secondary)"}),
                    ],
                    style={"marginTop": "6px"},
                )
            )

        return html.Div(children, className="recommendations-section")


def _build_history_filter_bar(history_data: dict) -> html.Div:
    """Build the filter bar for the History tab: symbol dropdown, recent chips, date buttons."""
    if isinstance(history_data, list):
        history_data = {"trading_agent_reports": history_data}

    all_symbols = set()
    for ta in history_data.get("trading_agent_reports", []):
        if ta.get("symbol"):
            all_symbols.add(ta["symbol"])
    for r in history_data.get("reports", []):
        if r.get("symbol"):
            all_symbols.add(r["symbol"])
    for p in history_data.get("predictions", []):
        if p.get("symbol"):
            all_symbols.add(p["symbol"])
    for rec in history_data.get("recommendations", []):
        for sym in (rec.get("symbols_csv", "") or "").split(","):
            sym = sym.strip()
            if sym:
                all_symbols.add(sym)

    sorted_symbols = sorted(all_symbols)

    recent_symbols = []
    seen = set()
    for ta in history_data.get("trading_agent_reports", []):
        sym = ta.get("symbol", "")
        if sym and sym not in seen:
            recent_symbols.append(sym)
            seen.add(sym)
        if len(recent_symbols) >= 5:
            break
    for p in history_data.get("predictions", []):
        sym = p.get("symbol", "")
        if sym and sym not in seen:
            recent_symbols.append(sym)
            seen.add(sym)
        if len(recent_symbols) >= 5:
            break

    rows = []

    rows.append(
        html.Div(
            dcc.Dropdown(
                id="history-symbol-dropdown",
                options=[{"label": s, "value": s} for s in sorted_symbols],
                value=[],
                multi=True,
                searchable=True,
                placeholder="Filter by symbol...",
                className="history-symbol-dropdown",
                persistence=True,
                persistence_type="memory",
            ),
            className="history-dropdown-row",
        )
    )

    if recent_symbols:
        rows.append(
            html.Div(
                [
                    html.Span("Recent:", className="history-recent-label"),
                    html.Div(
                        [
                            html.Button(
                                sym,
                                id={"type": "history-recent-chip", "symbol": sym},
                                className="history-recent-chip",
                            )
                            for sym in recent_symbols
                        ],
                        className="history-recent-chips",
                    ),
                ],
                className="history-filter-row",
            )
        )

    from utils.trading_calendar import is_trading_day, get_previous_trading_day
    from datetime import date as _date
    today = _date.today()
    active_day = today if is_trading_day(today) else get_previous_trading_day(today)

    rows.append(
        html.Div(
            [
                html.Div(
                    [
                        html.Span("Date:", className="history-date-label"),
                        dcc.DatePickerSingle(
                            id="history-date-picker",
                            date=None,
                            initial_visible_month=active_day.isoformat(),
                            display_format="YYYY-MM-DD",
                            placeholder=active_day.isoformat(),
                            className="history-date-picker",
                            persistence=True,
                            persistence_type="memory",
                        ),
                    ],
                    className="history-date-picker-row",
                ),
                html.Div(
                    [
                        dbc.ButtonGroup(
                            [
                                dbc.Button("7d", id={"type": "history-date-btn", "range": "7d"},
                                           size="sm", outline=True, color="secondary",
                                           className="history-date-btn"),
                                dbc.Button("30d", id={"type": "history-date-btn", "range": "30d"},
                                           size="sm", outline=True, color="secondary",
                                           className="history-date-btn"),
                                dbc.Button("90d", id={"type": "history-date-btn", "range": "90d"},
                                           size="sm", outline=True, color="secondary",
                                           className="history-date-btn"),
                                dbc.Button("All", id={"type": "history-date-btn", "range": "all"},
                                           size="sm", outline=True, color="secondary",
                                           className="history-date-btn"),
                            ],
                            size="sm",
                            className="history-date-group",
                        ),
                    ],
                    className="history-range-btns",
                ),
            ],
            className="history-filter-row history-date-row",
        )
    )

    rows.append(html.Div(id="history-applied-filters", className="history-applied-filters"))

    return html.Div(rows, className="history-filter-bar")


def _build_history_tab_layout(history_data: dict) -> html.Div:
    """Build the full History tab: filter bar (static) + content placeholder (dynamic)."""
    if isinstance(history_data, list):
        history_data = {"trading_agent_reports": history_data}

    return html.Div(
        [
            _build_history_filter_bar(history_data),
            html.Div(id="history-tab-content"),
        ],
        className="history-tab",
    )


def _filter_items(items, filter_symbols, filter_date_range, specific_date=None, sym_key="symbol", date_key="trade_date"):
    """Filter a list of dicts by symbol set and date range or specific date."""
    from datetime import timedelta
    if filter_symbols:
        items = [i for i in items if i.get(sym_key, "") in filter_symbols]
    if specific_date:
        items = [i for i in items if (i.get(date_key, "") or "")[:10] == specific_date[:10]]
    elif filter_date_range and filter_date_range != "all":
        days = {"7d": 7, "30d": 30, "90d": 90}.get(filter_date_range, 0)
        if days:
            cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
            items = [i for i in items if (i.get(date_key, "") or "") >= cutoff]
    return items


def _collapsible_section(title, section_id, children, icon_class, default_open=False, count=0):
    """Wrap content in a collapsible section with a clickable header."""
    badge = dbc.Badge(str(count), color="secondary", pill=True,
                      className="ms-2 history-count-badge") if count > 0 else None
    chevron_cls = "bi bi-chevron-up history-chevron" if default_open else "bi bi-chevron-down history-chevron"

    return html.Div(
        [
            html.Div(
                [
                    html.I(className=f"bi {icon_class} me-1"),
                    html.Span(title),
                    badge,
                    html.I(className=f"{chevron_cls} ms-auto",
                           id={"type": "history-section-chevron", "section": section_id}),
                ],
                className="history-section-header",
                id={"type": "history-section-toggle", "section": section_id},
            ),
            html.Div(
                children,
                id={"type": "history-section-body", "section": section_id},
                className="history-section-body",
                style={"display": "block" if default_open else "none"},
            ),
        ],
        className="history-collapsible-section",
    )


def _build_filtered_history_sections(history_data, filter_symbols, filter_date_range,
                                     specific_date=None, activity_scope="all"):
    """Build all history data sections with filtering and collapsible wrappers."""
    if isinstance(history_data, list):
        history_data = {"trading_agent_reports": history_data}

    reports = _filter_items(history_data.get("reports", []), filter_symbols, filter_date_range, specific_date)
    predictions = _filter_items(history_data.get("predictions", []), filter_symbols, filter_date_range, specific_date, date_key="prediction_date")
    ta_reports = _filter_items(history_data.get("trading_agent_reports", []), filter_symbols, filter_date_range, specific_date)

    # Recommendations have symbols_csv, need special filtering
    raw_recs = history_data.get("recommendations", [])
    if filter_symbols:
        recommendations = []
        filter_set = set(filter_symbols)
        for rec in raw_recs:
            rec_syms = {s.strip() for s in (rec.get("symbols_csv", "") or "").split(",") if s.strip()}
            if rec_syms & filter_set:
                recommendations.append(rec)
    else:
        recommendations = raw_recs
    if specific_date:
        recommendations = _filter_items(recommendations, [], "all", specific_date, date_key="created_at")
    elif filter_date_range and filter_date_range != "all":
        recommendations = _filter_items(recommendations, [], filter_date_range, date_key="created_at")

    has_any = reports or predictions or recommendations or ta_reports
    has_filters = filter_symbols or (filter_date_range and filter_date_range != "all") or specific_date

    if not has_any:
        return [
            html.Div(
                [
                    html.I(className="bi bi-clock-history",
                           style={"fontSize": "1.6rem", "opacity": "0.3", "display": "block", "marginBottom": "10px"}),
                    html.Div("No matching data" if has_filters else "No historical data yet",
                             style={"fontWeight": "600", "marginBottom": "4px"}),
                    html.Div(
                        "Adjust filters or run Full Analysis to generate data.",
                        style={"color": "var(--text-secondary)", "fontSize": "0.85rem"},
                    ),
                ],
                className="history-empty-msg",
            ),
        ]

    sections = []

    # === HERO: TradingAgents Reports ===
    if ta_reports:
        ta_cards = []
        for i, report in enumerate(ta_reports):
            symbol = report.get("symbol", "")
            decision = report.get("decision", "HOLD")
            confidence = report.get("confidence", 0)
            trade_date = report.get("trade_date", "")
            created_at = report.get("created_at", "")
            report_text = report.get("report_text", "")
            input_tokens = report.get("input_tokens", 0)
            output_tokens = report.get("output_tokens", 0)

            conf_pct = int((confidence or 0) * 100)
            dec_cls = "positive" if decision == "BUY" else "negative" if decision == "SELL" else "neutral"

            ta_cards.append(
                html.Div(
                    [
                        html.Div(
                            [
                                html.Span(symbol, className="ta-card-symbol"),
                                html.Span(decision, className=f"history-decision {dec_cls}"),
                                html.Span(f"{conf_pct}%", className="ta-card-conf"),
                            ],
                            className="ta-card-header",
                        ),
                        html.Div(
                            [
                                html.Span(trade_date, className="ta-card-date"),
                                html.Span(f"{input_tokens + output_tokens:,} tokens", className="ta-card-meta"),
                            ],
                            className="ta-card-meta-row",
                        ),
                        html.Div(
                            [
                                html.Button(
                                    [html.I(className="bi bi-eye me-1"), "View"],
                                    id={"type": "ta-view-btn", "idx": i},
                                    className="ta-view-btn",
                                ),
                                html.A(
                                    [html.I(className="bi bi-file-earmark-pdf me-1"), "PDF"],
                                    href=f"/api/download/ta-report/{report.get('id', 0)}",
                                    className="ta-pdf-btn",
                                ),
                                html.A(
                                    [html.I(className="bi bi-table me-1"), "Data"],
                                    href=(f"/api/download/report-inputs?symbols={symbol}"
                                          f"&date={trade_date}"),
                                    className="ta-pdf-btn",
                                    title="Download the point-in-time inputs this report used (.xlsx)",
                                ),
                            ],
                            className="ta-card-actions",
                        ),
                    ],
                    className="ta-report-card",
                )
            )

        # Cards only — View opens the report in a modal. The old duplicate
        # accordion stack below the cards (two ways to open the same report)
        # is gone.
        sections.append(_collapsible_section(
            "TradingAgents Reports", "ta",
            html.Div(ta_cards, className="ta-cards-grid"),
            icon_class="bi-robot", default_open=True, count=len(ta_reports),
        ))

    # === Saved Reports (PDF/JSON/MD) ===
    if reports:
        report_rows = []
        for r in reports:
            sym = r.get("symbol") or "Portfolio"
            rtype = r.get("report_type", "").replace("_", " ").title()
            fmt = (r.get("file_format") or "").upper()
            # JSON-stored AI reports are converted to Markdown on download
            # (serve_saved_report) — label the button by what the user GETS.
            if fmt == "JSON":
                fmt = "MD"
            storage_key = r.get("storage_key", "")

            if storage_key:
                from urllib.parse import quote
                dl_btn = html.A(
                    [html.I(className="bi bi-download me-1"), fmt or "Download"],
                    href=f"/api/download/saved-report?key={quote(storage_key, safe='')}",
                    className="history-dl-btn",
                )
            else:
                dl_btn = html.Span(fmt or "—", className="history-fmt-badge")

            report_rows.append(html.Tr([
                html.Td(r.get("trade_date", "")), html.Td(sym), html.Td(rtype), html.Td(dl_btn),
            ]))

        sections.append(_collapsible_section(
            "Saved Reports", "reports",
            html.Div(
                html.Table([
                    html.Thead(html.Tr([html.Th("Date"), html.Th("Symbol"), html.Th("Type"), html.Th("")])),
                    html.Tbody(report_rows),
                ], className="history-data-table"),
                className="history-table-wrap",
            ),
            icon_class="bi-file-earmark-text", default_open=False, count=len(reports),
        ))

    # (The inline Model Scoreboard section was removed 2026-07-26: the
    # Scoreboard modal renders the same _aggregate_scoreboard table as a
    # strict superset — by-symbol view, filters, pending count, Evaluate.)

    # === Model Predictions — grouped by symbol, then date ===
    if predictions:
        pending_count = sum(1 for p in predictions
                            if p.get("was_correct") is None and p.get("pnl_dollars") is None)

        def _pred_row(p):
            decision = p.get("decision", "HOLD")
            dec_cls = "positive" if decision == "BUY" else "negative" if decision == "SELL" else "neutral"
            conf = p.get("confidence")
            conf_str = f"{int(conf * 100)}%" if conf else "—"
            pnl = p.get("pnl_dollars")
            correct = p.get("was_correct")
            if pnl is not None:
                result_str = f"${pnl:+.2f}"
                result_cls = "positive" if pnl > 0 else "negative" if pnl < 0 else ""
                if correct is not None:
                    result_str += " ✓" if correct else " ✗"
            elif correct is not None:
                result_str = "Correct" if correct else "Wrong"
                result_cls = "positive" if correct else "negative"
            else:
                result_str = f"Pending → {p.get('target_date', '?')}"
                result_cls = ""
            return html.Tr([
                html.Td(p.get("model_name", "")),
                html.Td(html.Span(decision, className=f"history-decision {dec_cls}")),
                html.Td(conf_str),
                html.Td(html.Span(result_str, className=result_cls)),
            ])

        by_symbol: dict = {}
        for p in predictions:
            by_symbol.setdefault(p.get("symbol", "?"), []).append(p)

        symbol_groups = []
        for sym in sorted(by_symbol):
            sym_preds = by_symbol[sym]
            scored = [p for p in sym_preds if p.get("was_correct") is not None]
            hits = sum(1 for p in scored if p["was_correct"])
            pnl_total = sum(p.get("pnl_dollars") or 0 for p in sym_preds)
            pnl_cls = "positive" if pnl_total > 0 else "negative" if pnl_total < 0 else ""

            by_date: dict = {}
            for p in sym_preds:
                by_date.setdefault(p.get("prediction_date", "?"), []).append(p)

            date_blocks = []
            for d in sorted(by_date, reverse=True):
                d_preds = by_date[d]
                target = d_preds[0].get("target_date", "?")
                date_blocks.append(html.Div([
                    html.Div(
                        [html.Span(f"As-of {d}", className="history-pred-date"),
                         html.Span(f"→ predicts {target} close", className="history-pred-target"),
                         html.A(
                             [html.I(className="bi bi-table me-1"), "Data"],
                             href=f"/api/download/report-inputs?symbols={sym}&date={d}",
                             className="history-dl-btn ms-auto",
                             title="Download the inputs these models saw (.xlsx)",
                         )],
                        className="history-pred-date-row",
                    ),
                    html.Table([
                        html.Thead(html.Tr([html.Th("Model"), html.Th("Signal"),
                                            html.Th("Conf"), html.Th("Result")])),
                        html.Tbody([_pred_row(p) for p in
                                    sorted(d_preds, key=lambda x: x.get("model_name", ""))]),
                    ], className="history-data-table history-pred-table"),
                ], className="history-pred-date-block"))

            summary_bits = [html.Span(sym, className="history-pred-symbol")]
            if scored:
                summary_bits.append(html.Span(
                    f"{hits}/{len(scored)} correct", className="history-pred-hits"))
            summary_bits.append(html.Span(
                f"${pnl_total:+.2f}", className=f"history-pred-pnl {pnl_cls}"))
            summary_bits.append(html.Span(
                f"{len(by_date)} days", className="history-pred-days"))
            summary_bits.append(html.I(
                className="bi bi-chevron-down history-chevron ms-auto",
                id={"type": "history-section-chevron", "section": f"pred-{sym}"}))

            symbol_groups.append(html.Div([
                html.Div(summary_bits, className="history-section-header history-pred-header",
                         id={"type": "history-section-toggle", "section": f"pred-{sym}"}),
                html.Div(date_blocks,
                         id={"type": "history-section-body", "section": f"pred-{sym}"},
                         className="history-section-body",
                         style={"display": "none"}),
            ], className="history-collapsible-section history-pred-symbol-group"))

        eval_bar = html.Div(
            [
                html.Span(
                    f"{pending_count} prediction{'s' if pending_count != 1 else ''} awaiting evaluation"
                    if pending_count else "All predictions evaluated",
                    className="history-eval-hint",
                ),
                dbc.Button(
                    [html.I(className="bi bi-check2-circle me-1"), "Evaluate now"],
                    id="history-evaluate-btn",
                    size="sm", color="success", outline=True,
                    disabled=pending_count == 0,
                    className="history-evaluate-btn",
                ),
            ],
            className="history-eval-bar",
        )

        sections.append(_collapsible_section(
            "Model Predictions", "predictions",
            html.Div([eval_bar] + symbol_groups),
            icon_class="bi-cpu", default_open=False, count=len(predictions),
        ))

    # === Recommendations ===
    if recommendations:
        rec_rows = []
        for rec in recommendations:
            syms = rec.get("symbols_csv", "")
            model = rec.get("model_used", "")
            created = (rec.get("created_at") or "")[:10]
            result = rec.get("result_json", {}) or {}
            overall = result.get("overall", {})
            action = overall.get("portfolio_action", "") if overall else ""
            # Evidence basis: what this recommendation was synthesized FROM.
            # Old rows predate the field — label them as the legacy default.
            basis = result.get("basis") or "news+signals"
            basis_label = {
                "research+signals": "Research + predictions",
                "news+signals": "News analysis + predictions",
                "signals": "Predictions only",
            }.get(basis, basis)

            rec_as_of = (result.get("as_of") or created or "")[:10]
            sym_qs = ",".join(sorted({x.strip() for x in syms.split(",") if x.strip()}))
            dl = html.A(
                [html.I(className="bi bi-table me-1"), "Data"],
                href=f"/api/download/report-inputs?symbols={sym_qs}&date={rec_as_of}",
                className="history-dl-btn",
                title="Download all model inputs + news behind this recommendation (.xlsx)",
            ) if sym_qs and rec_as_of else html.Span("—")
            rec_rows.append(html.Tr([
                html.Td(created), html.Td(syms), html.Td(model),
                html.Td(basis_label, className="history-rec-basis"),
                html.Td(action, className="history-rec-action"),
                html.Td(dl),
            ]))

        sections.append(_collapsible_section(
            "Recommendations", "recs",
            html.Div(
                html.Table([
                    html.Thead(html.Tr([html.Th("Date"), html.Th("Symbols"), html.Th("Model"),
                                        html.Th("Based on"), html.Th("Portfolio Action"),
                                        html.Th("")])),
                    html.Tbody(rec_rows),
                ], className="history-data-table"),
                className="history-table-wrap",
            ),
            icon_class="bi-lightning", default_open=False, count=len(recommendations),
        ))

    # === Activity Log — every past run from the audit trail ===
    # Not filtered by symbol/date: this is the record of what the system did,
    # which is exactly what you want intact when something looks wrong.
    from services import progress_service as _prog
    is_admin = _prog.viewer_is_admin()
    scope = activity_scope if is_admin else "self"
    activity_runs = _prog.get_activity_runs(limit_runs=50, scope=scope)
    if activity_runs:
        run_blocks = []
        for run_idx, run in enumerate(activity_runs):
            lines = [
                html.Div([
                    html.Span(e["ts"], className="progress-ts"),
                    html.Span(e["message"], className="progress-msg"),
                ], className="progress-line"
                   + (" progress-line-error" if e["stage"] == "error" else ""))
                for e in run["events"]
            ]
            started = run["started"].strftime("%Y-%m-%d %H:%M")
            err = (f" · {run['errors']} error{'s' if run['errors'] != 1 else ''}"
                   if run["errors"] else "")
            # Whose run it was only matters when more than one person's rows
            # can appear, i.e. the admin "All users" view.
            owner = f"[{run['user_id']}] " if scope == "all" else ""
            # Index-keyed: runs are grouped by (user_id, run_id) and the
            # ad-hoc ids ("adhoc", "auth") repeat across users, so run_id
            # alone would emit duplicate Dash component ids.
            run_blocks.append(_collapsible_section(
                f"{started} — {owner}{run['title']}",
                f"activity-{run_idx}",
                html.Div(lines, className="activity-run-feed"),
                icon_class="bi-clock-history",
                default_open=False,
                count=len(run["events"]),
            ))
            run_blocks.append(html.Div(
                f"{run['duration_s']:.0f}s{err}",
                className="activity-run-meta",
            ))

        if is_admin:
            hint = ("Every pipeline run, newest first — you are an "
                    "Administrator, so this spans all users. Click a run to "
                    "expand its events.") if scope == "all" else (
                   "Your own pipeline runs, newest first. Click a run to "
                   "expand its events.")
            scope_control = dbc.ButtonGroup(
                [
                    dbc.Button(
                        "All users", id={"type": "activity-scope-btn", "scope": "all"},
                        size="sm", color="primary" if scope == "all" else "secondary",
                        outline=scope != "all",
                    ),
                    dbc.Button(
                        "Just me", id={"type": "activity-scope-btn", "scope": "self"},
                        size="sm", color="primary" if scope == "self" else "secondary",
                        outline=scope != "self",
                    ),
                ],
                className="activity-scope-group mb-2",
            )
        else:
            hint = ("Every pipeline run this account has executed, newest "
                    "first. Click a run to expand its events.")
            scope_control = None

        sections.append(_collapsible_section(
            "Activity Log", "activity",
            html.Div([
                html.Div(hint, className="history-scoreboard-hint"),
                scope_control,
                html.Div(run_blocks),
            ]),
            icon_class="bi-journal-text", default_open=False,
            count=len(activity_runs),
        ))

    return sections


def _create_loading_state(symbols: list, stage: str = "news") -> html.Div:
    """Create loading state while fetching news data or generating AI analysis.

    Args:
        symbols: List of symbols being loaded
        stage: Loading stage - "news" for fetching news, "analysis" for AI analysis

    Returns:
        Loading state component with spinner and status message
    """
    symbols_text = ", ".join(symbols) if len(symbols) <= 3 else f"{len(symbols)} stocks"

    if stage == "news":
        status_text = f"Fetching news for {symbols_text}..."
        subtext = "Retrieving latest articles from sources"
    else:
        status_text = "Generating AI analysis..."
        subtext = f"Analyzing sentiment for {symbols_text}"

    return html.Div(
        [
            html.Div(
                [
                    # Spinner
                    html.Div(
                        [
                            html.Div(className="loading-spinner"),
                        ],
                        className="loading-spinner-container",
                    ),
                    # Status text
                    html.Div(
                        status_text,
                        className="loading-status-text",
                    ),
                    # Sub-text
                    html.Div(
                        subtext,
                        className="loading-subtext",
                    ),
                ],
                className="news-loading-state",
            ),
        ],
        className="news-loading-container",
    )


def _create_ai_failure_indicator() -> html.Div:
    """Show an error message with a retry button when AI analysis fails."""
    return html.Div(
        [
            html.Div(
                [
                    html.I(
                        className="bi bi-exclamation-triangle",
                        style={"fontSize": "1.2rem", "color": "#FFD700"},
                    ),
                    html.Span(
                        "AI analysis unavailable — the LLM provider didn't respond.",
                        className="loading-inline-text",
                        style={"color": "var(--text-secondary)", "marginLeft": "8px"},
                    ),
                    dbc.Button(
                        [html.I(className="bi bi-arrow-clockwise"), " Retry"],
                        id="ai-retry-btn",
                        size="sm",
                        color="secondary",
                        outline=True,
                        style={"marginLeft": "12px"},
                    ),
                ],
                className="ai-loading-inline",
            ),
        ],
        className="ai-loading-section",
    )


def _create_ai_loading_indicator() -> html.Div:
    """Create prompt to generate AI analysis via the sidebar button."""
    return html.Div(
        [
            html.Div(
                [
                    html.I(className="bi bi-file-text", style={"fontSize": "1.2rem", "color": "#17a2b8"}),
                    html.Span(
                        'Click "AI Report" in sidebar to generate insights',
                        className="loading-inline-text",
                        style={"color": "var(--text-secondary)"},
                    ),
                ],
                className="ai-loading-inline",
            ),
        ],
        className="ai-loading-section",
    )


def _build_overall_tab_content(
    articles_by_symbol: dict,
    analysis_by_symbol: dict,
    overall_analysis: dict,
    symbols: list,
    ai_failed: bool = False,
    recommendations: dict | None = None,
) -> html.Div:
    """Build content for the Overall tab with summary table and combined analysis.

    Args:
        articles_by_symbol: Dict mapping symbol -> list of articles
        analysis_by_symbol: Dict mapping symbol -> analysis dict
        overall_analysis: Combined analysis across all symbols
        symbols: List of all selected symbols

    Returns:
        Overall tab content component
    """
    children = []

    # Collect all articles for stats
    all_articles = []
    for sym_articles in articles_by_symbol.values():
        all_articles.extend(sym_articles or [])

    # -- AI Summary (digest of all symbols) --
    # Show loading, failure, or pending prompt depending on state
    has_ai_analysis = bool(analysis_by_symbol) or bool(overall_analysis)
    if not has_ai_analysis and all_articles:
        if ai_failed:
            children.append(_create_ai_failure_indicator())
        else:
            children.append(_create_ai_loading_indicator())

    # Build a comprehensive summary from per-symbol analyses
    summary_parts = []

    if analysis_by_symbol:
        # Count recommendations
        rec_counts = {"bullish": 0, "bearish": 0, "neutral": 0}
        for symbol in symbols:
            sym_analysis = analysis_by_symbol.get(symbol, {})
            rec = sym_analysis.get("recommendation", "").lower()
            if "bullish" in rec:
                rec_counts["bullish"] += 1
            elif "bearish" in rec:
                rec_counts["bearish"] += 1
            else:
                rec_counts["neutral"] += 1

        # Build summary text
        total_symbols = len(symbols)
        if rec_counts["bullish"] > 0:
            summary_parts.append(f"{rec_counts['bullish']} of {total_symbols} stocks show bullish signals")
        if rec_counts["bearish"] > 0:
            summary_parts.append(f"{rec_counts['bearish']} of {total_symbols} stocks show bearish signals")
        if rec_counts["neutral"] > 0 and rec_counts["bullish"] == 0 and rec_counts["bearish"] == 0:
            summary_parts.append(f"All {total_symbols} stocks show neutral sentiment")

    # Add overall key developments if available
    if overall_analysis and overall_analysis.get("key_developments"):
        summary_parts.append(overall_analysis.get("key_developments", ""))

    if summary_parts:
        ai_summary = html.Div(
            [
                html.Div(
                    [
                        html.Span("AI Summary", className="section-title mb-0"),
                        html.Div(
                            [
                                html.Button(
                                    [html.I(className="bi bi-braces me-1"), "View data"],
                                    id="ai-json-view-btn", className="ai-json-btn",
                                    title="Inspect the raw report payload",
                                ),
                            ],
                            className="ms-auto",
                        ),
                    ],
                    style={"display": "flex", "alignItems": "center"},
                ),
                html.Div(
                    ". ".join(summary_parts) if len(summary_parts) > 1 else summary_parts[0],
                    className="key-developments-content",
                ),
            ],
            className="key-developments",
        )
        children.append(ai_summary)

        # Portfolio-level provenance: which model compiled this and from what.
        if overall_analysis and overall_analysis.get("model_used"):
            src = overall_analysis.get("sources") or {}
            bits = [f"Compiled by {overall_analysis['model_used']}"]
            if src.get("articles") is not None:
                bits.append(f"{src['articles']} articles across "
                            f"{len(symbols or [])} symbols")
            if src.get("as_of"):
                bits.append(f"as-of {src['as_of']}")
            if src.get("analysis_tier") == "sentiment_fallback":
                bits.append("sentiment-count fallback (LLM parse failed)")
            children.append(html.Div(
                " · ".join(bits),
                className="research-report-footnote",
                style={"marginTop": "-4px", "marginBottom": "8px"},
            ))

        if overall_analysis and overall_analysis.get("risk_factors"):
            risk_children = [
                html.Div(
                    [html.I(className="bi bi-exclamation-triangle me-2"), "Risk Factors"],
                    className="section-title",
                    style={"color": "var(--negative)"},
                ),
                html.Div(
                    overall_analysis["risk_factors"],
                    className="key-developments-content",
                    style={"borderLeft": "3px solid var(--negative)", "paddingLeft": "10px"},
                ),
            ]
            if overall_analysis.get("risks_read"):
                risk_children.append(html.Div(
                    [html.I(className="bi bi-arrow-return-right me-1"),
                     overall_analysis["risks_read"]],
                    className="analysis-read-line",
                ))
            children.append(html.Div(risk_children, className="key-developments"))

        overall_watch = (overall_analysis or {}).get("watch_items") or []
        if isinstance(overall_watch, list) and overall_watch:
            children.append(html.Div(
                [
                    html.Div(
                        [html.I(className="bi bi-binoculars me-2"), "Watch Items"],
                        className="section-title",
                    ),
                    html.Ul(
                        [html.Li(str(w)) for w in overall_watch[:4]],
                        className="watch-items-list",
                    ),
                ],
                className="key-developments",
            ))

    # -- Per-Symbol Recommendations Table --
    if symbols and analysis_by_symbol:
        table_rows = []
        for symbol in symbols:
            sym_analysis = analysis_by_symbol.get(symbol, {})
            sym_articles = articles_by_symbol.get(symbol, [])

            rec = sym_analysis.get("recommendation", "—")
            confidence = sym_analysis.get("confidence")
            article_count = len(sym_articles)

            # Determine color class for recommendation
            rec_lower = rec.lower() if rec != "—" else ""
            if "bullish" in rec_lower:
                rec_class = "rec-bullish"
            elif "bearish" in rec_lower:
                rec_class = "rec-bearish"
            else:
                rec_class = "rec-neutral"

            # Format recommendation display
            rec_display = rec.replace("_", " ") if rec != "—" else "—"

            # Format confidence
            conf_display = f"{int(confidence * 100)}%" if confidence else "—"

            table_rows.append(
                html.Tr(
                    [
                        html.Td(symbol, className="symbol-cell"),
                        html.Td(
                            rec_display,
                            className=f"recommendation-cell {rec_class}",
                        ),
                        html.Td(conf_display, className="confidence-cell"),
                        html.Td(str(article_count), className="articles-cell"),
                    ]
                )
            )

        recommendations_table = html.Div(
            [
                html.Div("Recommendations by Symbol", className="section-title"),
                html.Table(
                    [
                        html.Thead(
                            html.Tr(
                                [
                                    html.Th("Symbol"),
                                    html.Th("Recommendation"),
                                    html.Th("Confidence"),
                                    html.Th("Articles"),
                                ]
                            )
                        ),
                        html.Tbody(table_rows),
                    ],
                    className="recommendations-table",
                ),
            ],
            className="recommendations-section",
        )
        children.append(recommendations_table)

    # -- Aggregated Sentiment Breakdown --
    sentiment_counts = {"bullish": 0, "neutral": 0, "bearish": 0}
    for a in all_articles:
        s = (a.get("sentiment") or "neutral").lower()
        if "bullish" in s:
            sentiment_counts["bullish"] += 1
        elif "bearish" in s:
            sentiment_counts["bearish"] += 1
        else:
            sentiment_counts["neutral"] += 1

    if any(sentiment_counts.values()):
        sentiment_breakdown = create_sentiment_breakdown(
            bullish=sentiment_counts["bullish"],
            neutral=sentiment_counts["neutral"],
            bearish=sentiment_counts["bearish"],
        )
        children.append(sentiment_breakdown)

    # -- Quick Stats --
    if all_articles:
        sources = list(set(a.get("source", "") for a in all_articles if a.get("source")))
        quick_stats = create_news_quick_stats(
            article_count=len(all_articles),
            source_count=len(sources),
            date_range=_get_date_range(all_articles),
            symbols=symbols,
        )
        children.append(quick_stats)

    # -- Recommendations Section (Full Analysis) --
    if recommendations and not recommendations.get("error"):
        rec_section = _create_recommendations_section(recommendations)
        if rec_section:
            children.insert(0, rec_section)

    # Handle empty state
    if not children:
        children.append(
            html.Div(
                [
                    html.I(className="bi bi-newspaper", style={"fontSize": "24px", "opacity": "0.5", "marginBottom": "8px"}),
                    html.P("No news available for selected stocks", style={"color": "#6B7280", "margin": "0"}),
                ],
                className="tab-empty-state",
                style={"textAlign": "center", "padding": "32px 16px"},
            )
        )

    return html.Div(children, className="tab-content-inner")


def _build_tab_content(
    articles: list,
    analysis: dict,
    symbols: list,
    is_overall: bool = False,
    model_signals: dict | None = None,
    strategy_metrics: list | None = None,
    strategy_evaluations: list | None = None,
    ai_failed: bool = False,
    recommendations: dict | None = None,
) -> html.Div:
    """Build content for a single tab (overall or per-symbol).

    Args:
        articles: List of article dictionaries for this tab
        analysis: AI analysis dictionary for this tab
        symbols: List of symbols (single symbol for per-symbol tab)
        is_overall: True if this is the Overall tab
        model_signals: Model prediction results for this symbol

    Returns:
        Tab content component
    """
    children = []

    # -- Unify the two research-text sources --
    # The research report reaches this tab through EITHER the AI Report flow
    # (analysis["research"]) OR the prediction pipeline (trading_agents entry
    # in the signals store — the only path in Full Analysis, which no longer
    # runs the shallow per-symbol pass). One resolution, one rendering.
    research = (analysis or {}).get("research") or {}
    if not research.get("raw_response") and model_signals and not is_overall:
        ta_sig = model_signals.get("trading_agents")
        if isinstance(ta_sig, dict) and not ta_sig.get("error"):
            det = ta_sig.get("details") or {}
            if det.get("raw_response"):
                research = {
                    "decision": ta_sig.get("decision", "HOLD"),
                    "confidence": ta_sig.get("confidence"),
                    "raw_response": det["raw_response"],
                    "triggers": det.get("triggers") or {},
                    "structured": det.get("structured") or {},
                    "provenance": det.get("provenance") or {},
                    "model": det.get("model", ""),
                }

    # When the shallow pass didn't run, the research epilogue supplies the
    # banner stance and the watch/thesis panels — same fields, same renderer,
    # one analyst voice. Existing shallow keys always win (dict-merge order).
    if research.get("raw_response"):
        st = research.get("structured") or {}
        derived = {"research": research}
        if st.get("stance"):
            derived["recommendation"] = st["stance"]
            derived["stance_source"] = "research_verdict"
        if research.get("confidence") is not None:
            derived["confidence"] = research["confidence"]
        if st.get("sentiment_alignment"):
            derived["sentiment_explanation"] = st["sentiment_alignment"]
        if st.get("watch_items"):
            derived["watch_items"] = st["watch_items"]
        if st.get("company_thesis"):
            derived["company_thesis"] = st["company_thesis"]
        if research.get("provenance"):
            derived["provenance"] = research["provenance"]
        analysis = {**derived, **(analysis or {})}

    # -- Model Signal Cards (per-symbol tabs only) --
    if model_signals and not is_overall:
        symbol = symbols[0] if symbols else ""
        signal_cards = create_signal_cards(model_signals, symbol=symbol)
        children.append(signal_cards)

    # -- Per-symbol Recommendations (Full Analysis) --
    if recommendations and not is_overall:
        rec_section = _create_recommendations_section(recommendations, symbol=symbols[0] if symbols else None)
        if rec_section:
            children.append(rec_section)

    # -- Strategy Performance (per-symbol tabs only) --
    if not is_overall and (strategy_metrics or strategy_evaluations):
        strategy_section = create_strategy_section(
            strategy_metrics or [], strategy_evaluations or []
        )
        children.append(strategy_section)

    # -- Recommendation Banner --
    if analysis and analysis.get("recommendation"):
        rec_banner = create_recommendation_banner(
            recommendation=analysis.get("recommendation", "NEUTRAL"),
            confidence=analysis.get("confidence"),
            article_count=len(articles),
            date_range=_get_date_range(articles),
        )
    elif articles:
        # Show loading state while waiting for AI analysis
        rec_banner = create_recommendation_banner(recommendation="LOADING")
    else:
        rec_banner = None

    if rec_banner:
        children.append(rec_banner)

    # -- Research Report — the verdict-first deep dive (either source) --
    if research.get("raw_response"):
        from models.single_agent import strip_epilogue
        r_dec = research.get("decision", "HOLD")
        r_cls = ("positive" if r_dec == "BUY"
                 else "negative" if r_dec == "SELL" else "neutral")
        body = strip_epilogue(research["raw_response"])
        # New reports end with their own "Compiled by …" sources footer; only
        # older persisted reports need the model named in the footnote.
        footnote = "Saved to History → TradingAgents Reports (PDF available there)."
        if "Compiled by" not in body and research.get("model"):
            footnote = (f"Model: {research['model']}. " + footnote)
        children.append(html.Div(
            [
                html.Div(
                    [
                        html.I(className="bi bi-journal-richtext me-2"),
                        html.Span("Research Report", className="section-title mb-0"),
                        html.Span(
                            f"{r_dec} · {research.get('confidence', 0):.0%}",
                            className=f"research-verdict-badge {r_cls}",
                        ),
                    ],
                    className="research-report-header",
                ),
                dcc.Markdown(
                    body,
                    className="ta-report-body",
                    style={"maxHeight": "420px", "overflowY": "auto",
                           "fontSize": "0.82rem", "lineHeight": "1.55"},
                ),
                html.Div(footnote, className="research-report-footnote"),
            ],
            className="key-developments research-report-section",
        ))

    # -- Key Developments --
    if analysis and analysis.get("key_developments"):
        kd_children = [
            html.Div("Key Developments", className="section-title"),
            html.Div(
                analysis.get("key_developments", ""),
                className="key-developments-content",
            ),
        ]
        # v3 interpretation line — old cached payloads simply lack the key
        if analysis.get("developments_read"):
            kd_children.append(html.Div(
                [html.I(className="bi bi-arrow-return-right me-1"),
                 analysis["developments_read"]],
                className="analysis-read-line",
            ))
        children.append(html.Div(kd_children, className="key-developments"))

        if analysis.get("risk_factors"):
            risk_children = [
                html.Div(
                    [html.I(className="bi bi-exclamation-triangle me-2"), "Risk Factors"],
                    className="section-title",
                    style={"color": "var(--negative)"},
                ),
                html.Div(
                    analysis["risk_factors"],
                    className="key-developments-content",
                    style={"borderLeft": "3px solid var(--negative)", "paddingLeft": "10px"},
                ),
            ]
            if analysis.get("risks_read"):
                risk_children.append(html.Div(
                    [html.I(className="bi bi-arrow-return-right me-1"),
                     analysis["risks_read"]],
                    className="analysis-read-line",
                ))
            children.append(html.Div(risk_children, className="key-developments"))

        # Shallow-tier provenance: name the compiling model and its inputs so
        # the reader knows this analysis is news+metrics, not deep research.
        if analysis.get("model_used"):
            src = analysis.get("sources") or {}
            bits = [f"Compiled by {analysis['model_used']}"]
            if src.get("articles") is not None:
                bits.append(f"{src['articles']} articles")
            if src.get("validated_blocks"):
                bits.append("validated " + "/".join(src["validated_blocks"]))
            if src.get("analysis_tier") == "sentiment_fallback":
                bits.append("sentiment-count fallback (LLM parse failed)")
            children.append(html.Div(
                " · ".join(bits),
                className="research-report-footnote",
                style={"marginTop": "-4px", "marginBottom": "8px"},
            ))
    elif articles and not analysis:
        if ai_failed:
            children.append(_create_ai_failure_indicator())
        else:
            children.append(_create_ai_loading_indicator())

    # -- Watch Items / Company Thesis — from either tier (shallow JSON or the
    # research epilogue); rendered identically so there is ONE report style.
    if analysis:
        watch = analysis.get("watch_items") or []
        if isinstance(watch, list) and watch:
            children.append(html.Div(
                [
                    html.Div(
                        [html.I(className="bi bi-binoculars me-2"), "Watch Items"],
                        className="section-title",
                    ),
                    html.Ul(
                        [html.Li(str(w)) for w in watch[:4]],
                        className="watch-items-list",
                    ),
                ],
                className="key-developments",
            ))

        thesis = analysis.get("company_thesis") or {}
        if thesis:
            thesis_children = [
                html.Div(
                    [html.I(className="bi bi-building me-2"), "Company Thesis"],
                    className="section-title",
                ),
            ]
            if thesis.get("perception"):
                thesis_children.append(html.Div([
                    html.Strong("Market Perception: "), thesis["perception"],
                ], className="key-developments-content", style={"marginBottom": "8px"}))
            if thesis.get("goal_alignment"):
                thesis_children.append(html.Div([
                    html.Strong("Goal Alignment: "), thesis["goal_alignment"],
                ], className="key-developments-content", style={"marginBottom": "8px"}))
            catalysts = []
            for c in (thesis.get("positive_catalysts") or []):
                catalysts.append(html.Li(f"▲ {c}", style={"color": "var(--positive)"}))
            for c in (thesis.get("negative_catalysts") or []):
                catalysts.append(html.Li(f"▼ {c}", style={"color": "var(--negative)"}))
            if catalysts:
                thesis_children.append(html.Div([
                    html.Strong("Catalysts:"),
                    html.Ul(catalysts, style={"marginBottom": "8px", "paddingLeft": "18px",
                                              "fontSize": "0.85rem", "lineHeight": "1.5"}),
                ], className="key-developments-content"))
            if thesis.get("regime_risks"):
                thesis_children.append(html.Div([
                    html.Strong("Regime / Systematic Risks: "), thesis["regime_risks"],
                ], className="key-developments-content"))
            children.append(html.Div(thesis_children, className="key-developments"))

    # -- Top Headlines --
    if articles:
        top_headlines = create_top_headlines(articles, max_count=5)
        children.append(top_headlines)

    # -- Sentiment Breakdown --
    sentiment_counts = {"bullish": 0, "neutral": 0, "bearish": 0}
    for a in articles:
        s = (a.get("sentiment") or "neutral").lower()
        if "bullish" in s:
            sentiment_counts["bullish"] += 1
        elif "bearish" in s:
            sentiment_counts["bearish"] += 1
        else:
            sentiment_counts["neutral"] += 1

    if any(sentiment_counts.values()):
        sentiment_breakdown = create_sentiment_breakdown(
            bullish=sentiment_counts["bullish"],
            neutral=sentiment_counts["neutral"],
            bearish=sentiment_counts["bearish"],
        )
        children.append(sentiment_breakdown)

    # -- Quick Stats --
    if articles:
        sources = list(set(a.get("source", "") for a in articles if a.get("source")))
        quick_stats = create_news_quick_stats(
            article_count=len(articles),
            source_count=len(sources),
            date_range=_get_date_range(articles),
            symbols=symbols if is_overall else None,
        )
        children.append(quick_stats)

    # Handle empty state for this tab
    if not children:
        label = "all stocks" if is_overall else symbols[0] if symbols else "this stock"
        children.append(
            html.Div(
                [
                    html.I(className="bi bi-newspaper", style={"fontSize": "24px", "opacity": "0.5", "marginBottom": "8px"}),
                    html.P(f"No news available for {label}", style={"color": "#6B7280", "margin": "0"}),
                ],
                className="tab-empty-state",
                style={"textAlign": "center", "padding": "32px 16px"},
            )
        )

    return html.Div(children, className="tab-content-inner")


def _get_date_range(articles: list) -> str:
    """Get formatted date range from articles list."""
    if not articles:
        return ""

    dates = []
    for a in articles:
        pub = a.get("published_at")
        if pub:
            try:
                if isinstance(pub, str):
                    dt = datetime.fromisoformat(pub)
                else:
                    dt = pub
                # Articles mix tz-aware and naive timestamps depending on
                # source/cache — strip tzinfo so min/max can compare them.
                if dt.tzinfo is not None:
                    dt = dt.replace(tzinfo=None)
                dates.append(dt)
            except (ValueError, TypeError):
                continue

    if not dates:
        return ""

    oldest = min(dates)
    newest = max(dates)

    if oldest.date() == newest.date():
        return newest.strftime("%b %d")
    else:
        return f"{oldest.strftime('%b %d')} - {newest.strftime('%b %d')}"


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


def _json_report_to_markdown(data: dict) -> str:
    """Convert an AI report JSON dict into a readable Markdown document."""
    lines = []
    generated = data.get("generated_at", "")
    if generated:
        lines.append(f"*Generated: {generated}*\n")

    overall = data.get("overall", {})
    if overall:
        lines.append("# Portfolio Summary\n")
        rec = overall.get("recommendation", "—")
        conf = overall.get("confidence", 0)
        lines.append(f"**Recommendation:** {rec}  ")
        lines.append(f"**Confidence:** {int(conf * 100) if isinstance(conf, float) and conf <= 1 else conf}%  ")
        lines.append(f"**Sentiment:** {overall.get('market_sentiment', '—')}\n")
        if overall.get("sentiment_explanation"):
            lines.append(f"{overall['sentiment_explanation']}\n")
        if overall.get("key_developments"):
            lines.append(f"**Key Developments:** {overall['key_developments']}\n")

    by_symbol = data.get("by_symbol", {})
    if by_symbol:
        lines.append("---\n")
        for sym, info in by_symbol.items():
            lines.append(f"# {sym}\n")
            rec = info.get("recommendation", "—")
            conf = info.get("confidence", 0)
            lines.append(f"**Recommendation:** {rec}  ")
            lines.append(f"**Confidence:** {int(conf * 100) if isinstance(conf, float) and conf <= 1 else conf}%  ")
            lines.append(f"**Sentiment:** {info.get('market_sentiment', '—')}\n")
            if info.get("key_developments"):
                lines.append(f"### Key Developments\n{info['key_developments']}\n")
            if info.get("sentiment_explanation"):
                lines.append(f"### Sentiment Analysis\n{info['sentiment_explanation']}\n")
            if info.get("risk_factors"):
                lines.append(f"### Risk Factors\n{info['risk_factors']}\n")

            thesis = info.get("company_thesis") or {}
            if thesis:
                lines.append("### Company Thesis")
                if thesis.get("perception"):
                    lines.append(f"**Market Perception:** {thesis['perception']}\n")
                if thesis.get("goal_alignment"):
                    lines.append(f"**Goal Alignment:** {thesis['goal_alignment']}\n")
                if thesis.get("positive_catalysts"):
                    lines.append("**Positive Catalysts:**")
                    for c in thesis["positive_catalysts"]:
                        lines.append(f"- {c}")
                    lines.append("")
                if thesis.get("negative_catalysts"):
                    lines.append("**Negative Catalysts:**")
                    for c in thesis["negative_catalysts"]:
                        lines.append(f"- {c}")
                    lines.append("")
                if thesis.get("regime_risks"):
                    lines.append(f"**Regime / Systematic Risks:** {thesis['regime_risks']}\n")
            lines.append("")

    return "\n".join(lines)






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
    Input("download-report-btn", "n_clicks"),
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
                meta = preds[0] if preds else {}
                model_signals["_meta"] = {"predict_date": meta.get("prediction_date", "")}
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

    pdf_bytes = generate_report_pdf(
        symbols=symbols,
        ai_analysis=ai_analysis,
        model_signals=model_signals,
        recommendations=recommendations,
        news_data=news_data,
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
    Input("predict-confirm-btn", "n_clicks"),
    Input("full-analysis-confirm-btn", "n_clicks"),
    State("stock-data-store", "data"),
    State("selected-symbols", "data"),
    State("ensemble-config-store", "data"),
    State("news-data-store", "data"),
    State({"type": "predict-model-check", "model": ALL}, "value"),
    State("predict-ensemble-check", "value"),
    State("predict-date-picker", "date"),
    State("fa-date-picker", "date"),
    State("fa-model", "value"),
    State("fa-type", "value"),
    background=True,
    running=[
        (Output("prediction-running-indicator", "style"), {"display": "block"}, {"display": "none"}),
    ],
    prevent_initial_call=True,
)
def generate_model_signals(n_clicks, full_clicks, stock_data, symbols, ensemble_config,
                           news_data, model_checks, run_ensemble,
                           predict_date_str, fa_date_str, fa_model, fa_type):
    """Generate model predictions in a background subprocess.

    Supports backtesting: when the selected as-of date is in the past, OHLCV
    data is truncated to that date so models only see data available
    as of that day. The selected date is stored in metadata for
    correct persistence.
    """
    if not (n_clicks or full_clicks) or not stock_data or not symbols:
        raise PreventUpdate

    # Full Analysis: run all models with ensemble enabled (override checkboxes)
    # and take the as-of date from the Full Analysis modal's own picker. The
    # research report (trading_agents) honors the modal's model/type choices —
    # `research_model` is a distinct kwarg so no other model can mistake it.
    is_full_analysis = ctx.triggered_id == "full-analysis-confirm-btn"
    research_kwargs = {}
    if is_full_analysis:
        model_checks = [True] * 5
        run_ensemble = True
        if fa_date_str:
            predict_date_str = fa_date_str
        research_kwargs = {
            "research_model": fa_model or None,
            "include_thesis": (fa_type or "thesis") != "standard",
        }
    else:
        # Predict-only flow owns the feed; Full Analysis started it already
        from services import progress_service as _prog
        _prog.start_run(f"Predictions — {len(symbols or [])} symbols")

    import os
    os.environ["_DASH_BG_SUBPROCESS"] = "1"

    from datetime import date as date_cls

    # Parse the selected prediction date (default: today)
    if predict_date_str:
        predict_date = date_cls.fromisoformat(
            predict_date_str[:10]
        )
    else:
        predict_date = date_cls.today()

    is_backtest = predict_date < date_cls.today()

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
            if is_backtest:
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

            # Backtest: truncate OHLCV to only data available as of predict_date
            if is_backtest:
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

            # No lookahead: the news store holds articles fetched today —
            # a backtest must not see anything published after the as-of date.
            # Robust half-open UTC window (see services.news_window).
            if is_backtest and sym_news:
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
                as_of=str(predict_date) if is_backtest else None,
                **research_kwargs,
            )

            results[symbol] = new_results
            decisions = {m: r.get("decision") for m, r in new_results.items()
                         if isinstance(r, dict) and not r.get("error")}
            _prog.emit("models", f"{symbol}: " + ", ".join(
                f"{m}={d}" for m, d in decisions.items()))

        # Attach metadata so persist_predictions knows the date
        results["_meta"] = {
            "predict_date": predict_date.isoformat(),
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
    # Both buttons are rendered dynamically (History tab / Scoreboard modal).
    # Dash 4's renderer hard-errors on dispatch when an Input id is missing
    # from the layout (Dash 3 tolerated it); allow_optional restores that.
    Input("history-evaluate-btn", "n_clicks", allow_optional=True),
    Input("scoreboard-evaluate-btn", "n_clicks", allow_optional=True),
    prevent_initial_call=True,
)
def evaluate_predictions_now(n_clicks, sb_clicks):
    """Score pending predictions against actual closes, on user demand.

    Triggered from the History tab or the Scoreboard modal. The n_clicks
    guard also covers the re-render insertion trigger (n_clicks resets to
    None whenever the History tab re-renders its button).
    """
    if not n_clicks and not sb_clicks:
        raise PreventUpdate
    try:
        cache = get_cache()
        count = cache.evaluate_predictions()
        from services import progress_service as prog
        prog.emit("action", f"Evaluation run: {count} predictions scored")
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
        if symbols:
            for symbol in symbols:
                ta_reports.extend(cache.get_trading_agent_reports(symbol, limit=10))
        else:
            ta_reports = cache.get_all_trading_agent_reports(limit=30)
        ta_reports.sort(key=lambda r: r.get("created_at", ""), reverse=True)
        result["trading_agent_reports"] = ta_reports

        # PDF / AI reports from report_catalog
        result["reports"] = cache.list_report_catalog(limit=50)

        # Model predictions
        result["predictions"] = cache.list_all_predictions(limit=1000)

        # Recommendation runs
        result["recommendations"] = cache.list_recommendation_runs(limit=50)

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


@callback(
    Output("progress-feed-scroll", "children"),
    Output("progress-count", "children"),
    Output("progress-header-icon", "children"),
    Input("progress-interval", "n_intervals"),
)
def render_progress_panel(_n):
    """Live activity feed for Full Analysis / Predict runs.

    Streams events emitted by every stage (including the background model
    subprocess). Shows while a run is active and for a grace period after,
    so late viewers can catch up on what happened.
    """
    import time as _time
    from services import progress_service as prog

    feed = prog.get_feed()
    events = feed.get("events") or []
    if not events:
        raise PreventUpdate

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

    return rows, f"{len(events)} events", header_icon


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
    Input("progress-panel-state", "data"),
)
def apply_progress_panel_state(state):
    """Translate panel state into layout. Sizing lives in CSS classes."""
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
    )


# =============================================================================
# MODEL SCOREBOARD MODAL
# =============================================================================


# Column definitions for the scoreboard, surfaced as header tooltips --
# several of these (Trades vs held, hit rate excluding HOLDs, the $1,000
# notional behind P&L) are not guessable from the label alone.
_SCOREBOARD_TIPS = {
    "Model": "The prediction model that produced these calls.",
    "Symbol": "The ticker these calls were made on.",
    "Trades": "BUY/SELL calls that took a position. HOLD days take no "
              "position and are counted separately as 'held'.",
    "Hit Rate": "Share of BUY/SELL calls where the next close moved the way "
                "the model predicted. HOLD days are excluded — they cannot "
                "be right or wrong.",
    "Avg Conf": "The model's own average stated confidence. Compare it with "
                "Hit Rate: a model claiming 70% should be right ~70% of the "
                "time, and a large gap means it is miscalibrated.",
    "P&L": "Total dollars across all trades. Each takes a fixed $1,000 "
           "notional position, entered at the close and exited at the next "
           "close. Gross — before commission, spread and slippage.",
    "$/Trade": "P&L divided by Trades: the average edge per position. This "
               "is the number to judge, since a large P&L over many trades "
               "can still be a per-trade edge of roughly zero.",
}


def _sb_th(label: str) -> html.Th:
    """Scoreboard header cell carrying its column definition as a tooltip."""
    return html.Th(label, title=_SCOREBOARD_TIPS.get(label, ""),
                   className="scoreboard-th")


def _aggregate_scoreboard(preds: list[dict], group_key: str) -> list[html.Tr]:
    """Aggregate evaluated predictions by model or symbol into table rows."""
    groups = {}
    for p in preds:
        g = groups.setdefault(p.get(group_key, "?"),
                              {"scored": 0, "hits": 0, "trades": 0, "trade_hits": 0,
                               "holds": 0, "pnl": 0.0, "conf": []})
        is_trade = (p.get("decision") or "HOLD").upper() != "HOLD"
        if p.get("was_correct") is not None:
            g["scored"] += 1
            g["hits"] += 1 if p["was_correct"] else 0
            # Only BUY/SELL take a position, so only they can be right or wrong
            # in a way that moves money -- HOLD scores "correct" by default and
            # would otherwise inflate the hit rate.
            if is_trade:
                g["trades"] += 1
                g["trade_hits"] += 1 if p["was_correct"] else 0
        elif not is_trade and p.get("pnl_dollars") is not None:
            # Evaluated HOLDs: the evaluator leaves was_correct as None (a
            # HOLD can't be right or wrong) but sets pnl_dollars to 0.0 --
            # keying "held" on was_correct kept this counter at zero forever.
            g["holds"] += 1
        if p.get("pnl_dollars") is not None:
            g["pnl"] += p["pnl_dollars"]
        if p.get("confidence") is not None:
            g["conf"].append(p["confidence"])

    rows = []
    for name in sorted(groups):
        g = groups[name]
        trades, pnl = g["trades"], g["pnl"]
        hit_rate = f"{g['trade_hits'] / trades:.0%}" if trades else "—"
        avg_conf = f"{sum(g['conf']) / len(g['conf']):.0%}" if g["conf"] else "—"
        pnl_cls = "positive" if pnl > 0 else "negative" if pnl < 0 else ""
        per_trade = f"${pnl / trades:+.2f}" if trades else "—"
        display = _MODEL_DISPLAY.get(name, name) if group_key == "model_name" else name
        rows.append(html.Tr([
            html.Td(display),
            # "35 · 4 held" rather than "35 of 39": the bare "of" left the
            # denominator ambiguous (of what? trades? days?). Naming the
            # second number makes it self-describing.
            html.Td(html.Span([
                html.Strong(str(trades)),
                html.Span(f" · {g['holds']} held", className="scoreboard-muted")
                if g["holds"] else "",
            ]), title=f"{trades} BUY/SELL positions taken, "
                      f"{g['holds']} HOLD days took no position "
                      f"({g['scored']} predictions scored in total)"),
            html.Td(hit_rate, title=f"{g['trade_hits']} of {trades} BUY/SELL calls "
                                    f"went the predicted way (HOLDs excluded)"),
            html.Td(avg_conf),
            html.Td(html.Span(f"${pnl:+.2f}", className=pnl_cls),
                    title=f"{trades} trades x $1,000 notional"),
            html.Td(html.Span(per_trade, className=pnl_cls)),
        ]))
    return rows


@callback(
    Output("scoreboard-modal", "is_open"),
    Output("scoreboard-content", "children"),
    Output("scoreboard-symbols", "options"),
    Output("scoreboard-pending-label", "children"),
    Input("scoreboard-btn", "n_clicks"),
    Input("scoreboard-symbols", "value"),
    Input("scoreboard-date-range", "start_date"),
    Input("scoreboard-date-range", "end_date"),
    Input("history-eval-status", "data"),
    State("scoreboard-modal", "is_open"),
    prevent_initial_call=True,
)
def render_scoreboard(open_clicks, sym_filter, start_date, end_date, _eval_status, is_open):
    """Populate the Model Scoreboard modal.

    Defaults to ALL symbols and ALL dates; the dropdown and date range
    narrow the view. Re-renders after an evaluation run.
    """
    triggered = ctx.triggered_id
    if triggered == "scoreboard-btn":
        if not open_clicks:
            raise PreventUpdate
        is_open = True
    if not is_open:
        raise PreventUpdate

    try:
        cache = get_cache()
        all_preds = cache.list_all_predictions(limit=2000)
    except Exception as e:
        logger.warning(f"Scoreboard load failed: {e}")
        return True, html.Div("Could not load predictions.", className="text-danger"), [], ""

    symbols_available = sorted({p.get("symbol", "") for p in all_preds if p.get("symbol")})
    options = [{"label": s, "value": s} for s in symbols_available]

    preds = all_preds
    if sym_filter:
        wanted = set(sym_filter)
        preds = [p for p in preds if p.get("symbol") in wanted]
    if start_date:
        preds = [p for p in preds if (p.get("prediction_date") or "") >= str(start_date)[:10]]
    if end_date:
        preds = [p for p in preds if (p.get("prediction_date") or "") <= str(end_date)[:10]]

    evaluated = [p for p in preds
                 if p.get("was_correct") is not None or p.get("pnl_dollars") is not None]
    pending = len(preds) - len(evaluated)
    pending_label = (f"{pending} pending prediction{'s' if pending != 1 else ''} in current filter"
                     if pending else "All predictions in filter are evaluated")

    if not evaluated:
        content = html.Div(
            [
                html.I(className="bi bi-trophy",
                       style={"fontSize": "1.6rem", "opacity": "0.3", "display": "block",
                              "marginBottom": "10px"}),
                html.Div("No evaluated predictions match this filter.",
                         style={"fontWeight": "600"}),
                html.Div("Run predictions, wait for the target date's close, then Evaluate pending.",
                         style={"color": "var(--text-secondary)", "fontSize": "0.85rem"}),
            ],
            className="history-empty-msg",
        )
        return True, content, options, pending_label

    dates = sorted(p.get("prediction_date", "") for p in evaluated)
    header = html.Div(
        f"{len(evaluated)} evaluated predictions · "
        f"{len({p.get('symbol') for p in evaluated})} symbols · "
        f"{dates[0]} → {dates[-1]}",
        className="scoreboard-summary",
    )
    hint = html.Div([
        html.Div(
            "Trades counts BUY/SELL only — each takes a fixed $1,000 notional "
            "position, held one session and closed at the next close. HOLD days "
            "take no position, score $0, and are excluded from hit rate.",
            className="history-scoreboard-hint",
        ),
        html.Div(
            "Hit rate vs avg confidence shows calibration: a model claiming 70% "
            "should be right ~70% of the time.",
            className="history-scoreboard-hint",
        ),
    ])

    thead = html.Thead(html.Tr([_sb_th("Model"), _sb_th("Trades"), _sb_th("Hit Rate"),
                                _sb_th("Avg Conf"), _sb_th("P&L"), _sb_th("$/Trade")]))
    model_table = html.Div(
        html.Table([thead, html.Tbody(_aggregate_scoreboard(evaluated, "model_name"))],
                   className="history-data-table"),
        className="history-table-wrap",
    )

    sym_thead = html.Thead(html.Tr([_sb_th("Symbol"), _sb_th("Trades"), _sb_th("Hit Rate"),
                                    _sb_th("Avg Conf"), _sb_th("P&L"), _sb_th("$/Trade")]))
    symbol_table = html.Div(
        html.Table([sym_thead, html.Tbody(_aggregate_scoreboard(evaluated, "symbol"))],
                   className="history-data-table"),
        className="history-table-wrap",
    )

    content = html.Div([
        header,
        hint,
        html.Div("By model", className="scoreboard-subtitle"),
        model_table,
        html.Div("By symbol", className="scoreboard-subtitle"),
        symbol_table,
    ])
    return True, content, options, pending_label


# =============================================================================
# VIEW FULL REPORT (switch to History tab + expand accordion)
# =============================================================================


# ── History filter callbacks ────────────────────────────────────────────

@callback(
    Output("history-filter-symbols", "data"),
    Output("history-symbol-dropdown", "value"),
    Input("history-symbol-dropdown", "value"),
    Input({"type": "history-recent-chip", "symbol": ALL}, "n_clicks"),
    State("history-filter-symbols", "data"),
    prevent_initial_call=True,
)
def update_history_symbol_filter(dropdown_val, chip_clicks, current_filter):
    """Sync symbol dropdown with filter store; handle recent-chip clicks."""
    triggered = ctx.triggered_id
    if isinstance(triggered, dict) and triggered.get("type") == "history-recent-chip":
        if not any(chip_clicks or []):
            raise PreventUpdate
        sym = triggered["symbol"]
        current = list(current_filter or [])
        if sym not in current:
            current.append(sym)
        return current, current
    new_val = list(dropdown_val or [])
    return new_val, new_val


@callback(
    Output("history-filter-date-range", "data"),
    Output("history-filter-date-specific", "data"),
    Output("history-date-picker", "date"),
    Input({"type": "history-date-btn", "range": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def update_history_date_filter(btn_clicks):
    """Set the date range filter from button group clicks; clear specific date."""
    if not any(btn_clicks or []):
        raise PreventUpdate
    triggered = ctx.triggered_id
    rng = "all"
    if isinstance(triggered, dict):
        rng = triggered.get("range", "all")
    return rng, None, None


@callback(
    Output("history-filter-date-specific", "data", allow_duplicate=True),
    Output("history-filter-date-range", "data", allow_duplicate=True),
    Input("history-date-picker", "date"),
    prevent_initial_call=True,
)
def update_history_specific_date(picked_date):
    """Set specific date filter from date picker; reset range on a real pick.

    When the picker is cleared programmatically (range button / Clear all),
    leave the range store alone — otherwise clicking "7d" would immediately
    be undone by this callback firing with date=None.
    """
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
    Output("history-tab-content", "children"),
    Output("history-applied-filters", "children"),
    Input("report-history-store", "data"),
    Input("history-filter-symbols", "data"),
    Input("history-filter-date-range", "data"),
    Input("history-filter-date-specific", "data"),
    Input("history-activity-scope", "data"),
)
def render_filtered_history(history_data, filter_symbols, filter_date_range, specific_date,
                            activity_scope):
    """Re-render history content when data or filters change."""
    data = history_data or {}
    syms = filter_symbols or []
    dr = filter_date_range or "all"
    sd = specific_date or None

    sections = _build_filtered_history_sections(data, syms, dr, sd,
                                                activity_scope or "all")

    # Build applied-filter chips
    chips = []
    if syms:
        for s in syms:
            chips.append(
                html.Span(
                    [
                        s,
                        html.Button(
                            "×",
                            id={"type": "history-remove-filter", "symbol": s},
                            className="history-filter-chip-remove",
                        ),
                    ],
                    className="history-filter-chip",
                )
            )
    if sd:
        chips.append(
            html.Span(
                [
                    sd[:10],
                    html.Button(
                        "×",
                        id={"type": "history-remove-date-filter", "range": "specific"},
                        className="history-filter-chip-remove",
                    ),
                ],
                className="history-filter-chip",
            )
        )
    elif dr and dr != "all":
        chips.append(
            html.Span(
                [
                    f"Last {dr}",
                    html.Button(
                        "×",
                        id={"type": "history-remove-date-filter", "range": dr},
                        className="history-filter-chip-remove",
                    ),
                ],
                className="history-filter-chip",
            )
        )
    if chips:
        chips.append(
            html.Button("Clear all", id="history-clear-all-filters", className="history-clear-all-link")
        )

    return sections, chips


@callback(
    Output("history-filter-symbols", "data", allow_duplicate=True),
    Output("history-filter-date-range", "data", allow_duplicate=True),
    Output("history-filter-date-specific", "data", allow_duplicate=True),
    Output("history-symbol-dropdown", "value", allow_duplicate=True),
    Output("history-date-picker", "date", allow_duplicate=True),
    Input({"type": "history-remove-filter", "symbol": ALL}, "n_clicks"),
    Input({"type": "history-remove-date-filter", "range": ALL}, "n_clicks"),
    Input("history-clear-all-filters", "n_clicks"),
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
        return [], "all", None, [], None

    if isinstance(triggered, dict):
        if triggered.get("type") == "history-remove-filter":
            sym = triggered["symbol"]
            new_syms = [s for s in (current_symbols or []) if s != sym]
            return new_syms, current_range or "all", current_specific, new_syms, dash.no_update
        if triggered.get("type") == "history-remove-date-filter":
            return current_symbols or [], "all", None, current_symbols or [], None

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
    Output("symbol-tabs", "active_tab"),
    Output("ta-report-modal", "is_open"),
    Output("ta-report-modal-title", "children"),
    Output("ta-report-modal-body", "children"),
    Input({"type": "view-full-report-btn", "symbol": ALL}, "n_clicks"),
    Input({"type": "ta-view-btn", "idx": ALL}, "n_clicks"),
    State("report-history-store", "data"),
    prevent_initial_call=True,
)
def jump_to_full_report(view_clicks, ta_view_clicks, history_data):
    """Open the matching research report in the reader modal.

    Replaces the old jump-to-History-accordion behavior: the accordion
    duplicated the cards' View/PDF affordances, so reports now open in one
    place, in a modal, from wherever the View button was clicked.
    """
    all_clicks = (view_clicks or []) + (ta_view_clicks or [])
    if not any(all_clicks):
        raise PreventUpdate

    triggered = ctx.triggered_id
    if not triggered or not isinstance(triggered, dict):
        raise PreventUpdate

    ta_reports = (history_data or {}).get("trading_agent_reports", []) \
        if isinstance(history_data, dict) else history_data or []

    report = None
    if triggered.get("type") == "ta-view-btn":
        idx = triggered.get("idx")
        if isinstance(idx, int) and 0 <= idx < len(ta_reports):
            report = ta_reports[idx]
    else:
        symbol = triggered.get("symbol", "")
        report = next((r for r in ta_reports if r.get("symbol") == symbol), None)

    if not report or not report.get("report_text"):
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
    Input("ensemble-config-btn", "n_clicks"),
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
# SCHEDULED EVALUATION (APScheduler)
# =============================================================================


def _scheduled_evaluation():
    """Daily evaluation job: evaluate predictions, run strategies, refresh metrics."""
    try:
        from services.evaluation_service import get_evaluation_service

        service = get_evaluation_service()
        results = service.run_evaluation()
        if any(v > 0 for v in results.values()):
            logger.info(f"Scheduled evaluation: {results}")
    except Exception as e:
        logger.error(f"Scheduled evaluation error: {e}")


# Startup/shutdown run through the ASGI lifespan (see _lifespan near the Dash
# constructor): exactly once per server process. The old WERKZEUG_RUN_MAIN
# guard was Werkzeug-only — under uvicorn it silently skipped the scheduler
# when DEBUG=true, and the background-callback spawn child started a duplicate
# scheduler when DEBUG=false.
_scheduler = None


def _startup():
    """One-time server-process startup: progress hydrate, S3, auth, scheduler."""
    global _scheduler

    _progress_service.hydrate_from_db()
    _init_s3()
    # Auth tables (no-op when AUTH_DATABASE_URL points at the shared Cygnet
    # auth DB, which already has them from CygnetResearchTerminal).
    from services.auth_service import ensure_auth_tables
    ensure_auth_tables()

    try:
        from apscheduler.schedulers.background import BackgroundScheduler
        from apscheduler.triggers.cron import CronTrigger
        from apscheduler.triggers.date import DateTrigger
        import pytz

        from config import STRATEGY

        _scheduler = BackgroundScheduler(timezone=pytz.timezone("US/Eastern"))
        _scheduler.add_job(
            _scheduled_evaluation,
            CronTrigger(
                day_of_week="mon-fri",
                hour=STRATEGY.EVAL_SCHEDULE_HOUR,
                minute=STRATEGY.EVAL_SCHEDULE_MINUTE,
            ),
            id="daily_evaluation",
            replace_existing=True,
        )
        # Catch-up pass for missed evaluations runs on the scheduler thread —
        # inline it used to block server startup for the whole evaluation.
        _scheduler.add_job(_scheduled_evaluation, DateTrigger(), id="startup_evaluation")
        _scheduler.start()
        logger.info(
            f"APScheduler: daily evaluation at "
            f"{STRATEGY.EVAL_SCHEDULE_HOUR}:{STRATEGY.EVAL_SCHEDULE_MINUTE:02d} ET"
        )
    except Exception as e:
        logger.error(f"APScheduler setup failed: {e}")


def _shutdown():
    """Lifespan shutdown: stop the scheduler without waiting on running jobs."""
    if _scheduler is not None:
        _scheduler.shutdown(wait=False)


# =============================================================================
# CONFIRMATION MODAL CALLBACKS
# =============================================================================

_MODEL_DISPLAY = {
    "kronos_mini": "Kronos",
    "xgboost_shap": "XGBoost SHAP",
    "lightgbm": "LightGBM",
    "deberta_sentiment": "DeBERTa Sentiment",
    "trading_agents": "TradingAgents",
    "ensemble": "Ensemble",
    "recommendation_synthesis": "Recommendations (LLM)",
}


@callback(
    Output("report-confirm-modal", "is_open"),
    Output("report-confirm-body", "children"),
    Input("generate-report-btn", "n_clicks"),
    Input("report-cancel-btn", "n_clicks"),
    Input("report-confirm-btn", "n_clicks"),
    State("news-data-store", "data"),
    State("selected-symbols", "data"),
    State("report-confirm-modal", "is_open"),
    prevent_initial_call=True,
)
def toggle_report_modal(open_clicks, cancel_clicks, confirm_clicks,
                        news_data, symbols, is_open):
    """Open/close the AI Report confirmation modal with data summary."""
    triggered = ctx.triggered_id

    if triggered in ("report-cancel-btn", "report-confirm-btn"):
        return False, dash.no_update

    if triggered == "generate-report-btn":
        if not symbols:
            return False, dash.no_update

        # Build data summary for the modal
        articles_by_symbol = (news_data or {}).get("articles_by_symbol", {})
        fetched_at = (news_data or {}).get("fetched_at", "N/A")

        from services.llm_service import get_llm
        llm = get_llm()
        provider = getattr(llm, "provider", "unknown") or "none"
        model_name = llm._get_model() if hasattr(llm, "_get_model") else "unknown"

        # No per-symbol article table here — the live windowed preview below
        # the Parameters section is the authoritative count (it reflects the
        # chosen date/window; the raw store count does not and the two tables
        # side by side read as a contradiction).
        body = html.Div([
            html.Div([
                html.Span("Symbols: ", style={"color": "var(--text-secondary)"}),
                html.Strong(", ".join(symbols)),
            ], className="mb-2"),
            html.Div([
                html.Span("News store refreshed: ", style={"color": "var(--text-secondary)"}),
                html.Strong(fetched_at[:19] if fetched_at != "N/A" else "Not fetched yet"),
            ], className="mb-3"),
            html.Hr(),
            # Model and analysis type are user-adjustable in the Parameters
            # section below — repeating the provider default here as "Model:"
            # contradicted the picker.
            html.Div([
                html.Span("Output: ", style={"color": "var(--text-secondary)"}),
                html.Strong("Per-symbol analysis + overall summary"),
                html.Span(f"  ·  provider fallback: {provider}",
                          style={"color": "var(--text-muted)", "fontSize": "0.8rem"}),
            ]),
        ])

        return True, body

    return is_open, dash.no_update


@callback(
    Output("ai-json-modal", "is_open"),
    Output("ai-json-body", "children"),
    Input("ai-json-view-btn", "n_clicks"),
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
    Output("ai-report-article-preview", "children"),
    Input("report-confirm-modal", "is_open"),
    Input("ai-report-lookback", "value"),
    Input("ai-report-date", "date"),
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
    Output("predict-date-mode-label", "children"),
    Output("predict-date-mode-label", "style"),
    Input("predict-date-picker", "date"),
    prevent_initial_call=True,
)
def update_predict_date_label(date_str):
    """Show Live or Backtest mode label based on selected date."""
    from datetime import date as date_cls

    if not date_str:
        return "", {"display": "none"}
    selected = date_cls.fromisoformat(date_str[:10])
    today = date_cls.today()
    if selected >= today:
        return [
            html.I(className="bi bi-broadcast me-1"),
            "Live prediction (next trading day)",
        ], {"fontSize": "0.85rem", "color": "var(--positive)"}
    else:
        days_back = (today - selected).days
        return [
            html.I(className="bi bi-clock-history me-1"),
            f"Backtest mode ({days_back}d ago — data truncated to {selected})",
        ], {"fontSize": "0.85rem", "color": "var(--warning, #ffc107)"}


@callback(
    Output("predict-confirm-modal", "is_open"),
    Output("predict-data-summary", "children"),
    Output("predict-ensemble-summary", "children"),
    Input("run-predictions-btn", "n_clicks"),
    Input("predict-cancel-btn", "n_clicks"),
    Input("predict-confirm-btn", "n_clicks"),
    State("stock-data-store", "data"),
    State("selected-symbols", "data"),
    State("ensemble-config-store", "data"),
    State("news-data-store", "data"),
    State("predict-confirm-modal", "is_open"),
    prevent_initial_call=True,
)
def toggle_predict_modal(open_clicks, cancel_clicks, confirm_clicks,
                         stock_data, symbols, ensemble_config, news_data, is_open):
    """Open/close the Predict confirmation modal with data summary."""
    triggered = ctx.triggered_id
    no_update_body = (dash.no_update, dash.no_update)

    # Guard: if no trigger matched, don't change modal state
    if triggered not in ("run-predictions-btn", "predict-cancel-btn", "predict-confirm-btn"):
        raise PreventUpdate

    if triggered in ("predict-cancel-btn", "predict-confirm-btn"):
        return False, *no_update_body

    if triggered == "run-predictions-btn":
        if not symbols:
            return False, *no_update_body

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
            display = _MODEL_DISPLAY.get(m, m)
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
    Output("full-analysis-modal", "is_open"),
    Output("full-analysis-body", "children"),
    Input("full-analysis-btn", "n_clicks"),
    Input("full-analysis-cancel-btn", "n_clicks"),
    Input("full-analysis-confirm-btn", "n_clicks"),
    State("stock-data-store", "data"),
    State("selected-symbols", "data"),
    State("news-data-store", "data"),
    State("full-analysis-modal", "is_open"),
    prevent_initial_call=True,
)
def toggle_full_analysis_modal(open_clicks, cancel_clicks, confirm_clicks,
                               stock_data, symbols, news_data, is_open):
    """Open/close the Full Analysis confirmation modal."""
    triggered = ctx.triggered_id
    if triggered not in ("full-analysis-btn", "full-analysis-cancel-btn",
                         "full-analysis-confirm-btn"):
        raise PreventUpdate

    if triggered in ("full-analysis-cancel-btn", "full-analysis-confirm-btn"):
        return False, dash.no_update

    if triggered == "full-analysis-btn":
        if not symbols:
            return False, dash.no_update

        articles_by_symbol = (news_data or {}).get("articles_by_symbol", {})
        sym_rows = []
        for sym in symbols:
            sym_data = (stock_data or {}).get(sym, {})
            bars = "N/A"
            if sym_data.get("prices"):
                try:
                    df = pd.read_json(StringIO(sym_data["prices"]))
                    bars = str(len(df))
                except Exception:
                    pass
            news_count = len(articles_by_symbol.get(sym, []))
            sym_rows.append(html.Tr([
                html.Td(sym), html.Td(bars), html.Td(str(news_count)),
            ]))

        body = html.Div([
            html.H6("Symbols", className="mb-2"),
            dbc.Table(
                [
                    html.Thead(html.Tr([
                        html.Th("Symbol"), html.Th("Bars"), html.Th("Articles"),
                    ])),
                    html.Tbody(sym_rows),
                ],
                bordered=True, color="dark", size="sm",
            ),
        ])
        return True, body

    return is_open, dash.no_update


@callback(
    Output("full-analysis-requested", "data"),
    Input("full-analysis-confirm-btn", "n_clicks"),
    prevent_initial_call=True,
)
def set_full_analysis_flag(n_clicks):
    """Set the full-analysis flag when the user confirms."""
    if n_clicks:
        return True
    raise PreventUpdate


def _merge_research_into_analysis(
    ai_analysis: dict | None,
    model_signals: dict | None,
    symbols: list | None,
) -> tuple[dict, bool]:
    """Backfill per-symbol research (and its derived stance fields) from the
    trading_agents signal into an AI-analysis dict.

    Full Analysis produces its research text through the prediction pipeline
    only; consumers that read ai_analysis["by_symbol"] (Luna, XLSX export,
    PDF report) get the same report the user sees on screen. Entries that
    already carry text analysis are untouched. Returns (dict, changed).
    """
    _stance = {"BUY": "BULLISH", "SELL": "BEARISH", "HOLD": "NEUTRAL"}
    ai_analysis = ai_analysis or {}
    by_sym = dict(ai_analysis.get("by_symbol") or {})
    changed = False
    for sym in (symbols or []):
        entry = dict(by_sym.get(sym) or {})
        if entry.get("research"):
            continue
        sig = ((model_signals or {}).get(sym) or {}).get("trading_agents") or {}
        det = (sig.get("details") or {}) if isinstance(sig, dict) else {}
        if not det.get("raw_response"):
            continue
        st = det.get("structured") or {}
        entry["research"] = {
            "decision": sig.get("decision"),
            "confidence": sig.get("confidence"),
            "raw_response": det["raw_response"],
            "triggers": det.get("triggers") or {},
            "structured": st,
            "provenance": det.get("provenance") or {},
            "model": det.get("model", ""),
        }
        if "recommendation" not in entry:
            entry["recommendation"] = (st.get("stance")
                                       or _stance.get(sig.get("decision"), "NEUTRAL"))
            entry["confidence"] = sig.get("confidence")
            entry["stance_source"] = "research_verdict"
            if st.get("sentiment_alignment"):
                entry["sentiment_explanation"] = st["sentiment_alignment"]
            if st.get("watch_items"):
                entry["watch_items"] = st["watch_items"]
            if st.get("company_thesis"):
                entry["company_thesis"] = st["company_thesis"]
            if det.get("provenance"):
                entry["provenance"] = det["provenance"]
        by_sym[sym] = entry
        changed = True
    if not changed:
        return ai_analysis, False
    return {**ai_analysis, "by_symbol": by_sym}, True


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

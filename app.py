"""Quant News Tracker - Main Application Entry Point.

A quantitative stock tracking dashboard with technical analysis,
news aggregation, and AI-powered insights.
"""

import asyncio
import atexit
import hashlib
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
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from dash import (
    Input, Output, State, callback, clientside_callback, ctx, html, ALL,
    MATCH, dcc,
)
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
from config import APP, COLORS, MODEL
from layouts.components import (
    calculate_period_label,
    create_metric_card,
    create_overview_empty_state,
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
from layouts.pages import trace as trace_page
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
def serve_report_inputs(symbols: str = "", date: str = "",
                        lookback: int | None = None, max_articles: int = 0):
    """Serve the point-in-time model-input workbook for a report/prediction.

    Reconstructs inputs with the same lookahead-safe builders the models use,
    for the news window the RECORD was made with (``lookback`` rides on the
    link; a link without it is refused rather than rebuilt at some default).
    No LLM is involved — downloads must never incur model cost.
    """
    as_of = date[:10]
    syms = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    if not syms or not as_of:
        raise HTTPException(status_code=400)
    if lookback is None:
        raise HTTPException(
            status_code=400,
            detail="lookback (the run's news window in days) is required — "
                   "this record predates the stamp, so its inputs cannot be "
                   "rebuilt faithfully")
    if len(syms) > 12:
        raise HTTPException(status_code=400,
                            detail=f"at most 12 symbols per workbook ({len(syms)} given)")
    try:
        from services.export_service import build_model_inputs_xlsx
        payload = build_model_inputs_xlsx(syms, as_of, news_lookback_days=lookback,
                                          max_articles=max_articles)
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
                 outcome="all", model="all", home_symbol=None,
                 watchlist=None, period=None, home_cutoff=None):
    """Build one section. Each page reads only the state it renders."""
    if path == "/analyze":
        return analyze_page.layout(period or "1y")
    if path in ("/performance", "/reports"):
        # Live read, not the store: see fetch_report_history. Each page asks
        # only for the buckets it renders.
        if path == "/performance":
            return performance_page.layout(
                fetch_report_history(
                    only={"predictions"},
                    filters=history_filters(filter_symbols, filter_range,
                                            filter_specific, model)),
                filter_symbols, filter_range, filter_specific, outcome, model)
        return reports_page.layout(
            fetch_report_history(
                only={"reports", "recommendations", "trading_agent_reports"}),
            filter_symbols, filter_range, filter_specific)
    if path == "/schedule":
        return schedule_page.layout()
    if path == "/trace":
        return trace_page.layout()
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
    recent = []
    try:
        from services import watchlist_service
        recent = watchlist_service.recent_groups(limit=5)
    except Exception as e:
        logger.warning("Could not read recent groups for Home: %s", e)
    cutoffs = ds.get_available_cutoffs()
    return home_page.layout(
        cohort=ds.get_cohort(home_cutoff),
        open_preds=ds.get_open_predictions(),
        rolling=ds.get_rolling_performance(days=HOME_ROLLING_DAYS),
        last_run=ds.get_last_run() if not home_cutoff else None,
        jobs=jobs,
        rolling_days=HOME_ROLLING_DAYS,
        reports_by_symbol=_home_reports_by_symbol(),
        active_symbol=home_symbol,
        symbol_reports=_home_symbol_reports(home_symbol),
        watchlist=watchlist or [],
        recent_groups=recent,
        symbol_detail=_home_symbol_detail(home_symbol),
        cutoffs=cutoffs,
        active_cutoff=home_cutoff or (cutoffs[0] if cutoffs else None),
    )


def _serialize_articles(articles) -> list[dict]:
    """NewsArticle objects -> the store/UI dict shape."""
    from services.news_window import article_to_dict
    return [article_to_dict(a) for a in articles]


def _fill_run_inputs(symbols, stock_data, news_data=None):
    """Fetch OHLCV/news for run symbols the browser stores don't cover.

    The stores only ever hold watchlist symbols; a run scoped to the last
    cohort or a single non-watchlist name fills its own gaps here, through
    the same cache layer the store fetch uses. Blocking — call it from the
    background run subprocess or via asyncio.to_thread.
    """
    stock_data = dict(stock_data or {})
    news = dict(news_data or {})
    articles = dict(news.get("articles_by_symbol") or {})
    for sym in symbols:
        if not (stock_data.get(sym) or {}).get("prices"):
            try:
                df, meta = get_cache().get_stock_prices(sym, "1y")
                df = add_indicators_to_df(df)
                stock_data[sym] = {
                    "prices": df.to_json(date_format="iso"),
                    "metrics": calculate_performance_metrics(df),
                    "signals": get_latest_signals(df),
                    "period": "1y",
                    "from_cache": meta.get("from_cache", False),
                    # Origin stamp for the run-level data_load event: entries
                    # without one came from the browser session store.
                    "source": ("cache" if meta.get("from_cache")
                               else "fetch"),
                }
            except Exception as e:
                logger.warning("Run input: price fetch failed for %s: %s",
                               sym, e)
        if news_data is not None and not articles.get(sym):
            try:
                articles[sym] = _serialize_articles(fetch_news_cached(sym))
            except Exception as e:
                logger.warning("Run input: news fetch failed for %s: %s",
                               sym, e)
                articles[sym] = []
    if news_data is not None:
        news["articles_by_symbol"] = articles
        news.setdefault("symbols", list(symbols))
        return stock_data, news
    return stock_data


def _home_symbol_detail(symbol: str | None) -> dict | None:
    """Chart + news for the Home research pane, fetched server-side.

    Works for ANY symbol — watchlist or last-run cohort — which is the
    point: a cohort-only name used to have no path to its chart or news
    without re-typing it into the watchlist. Prices and news both come
    through the normal cache layer, so a cached symbol costs two quick
    Postgres reads and an uncached one a single provider fetch.
    """
    if not symbol:
        return None
    detail: dict = {"figure": None, "signals": {}, "articles": []}
    try:
        df, _meta = get_cache().get_stock_prices(symbol, "6mo")
        df = add_indicators_to_df(df)
        if not df.empty:
            detail["signals"] = get_latest_signals(df)
            closes = df["Close"]
            dates = df["Date"] if "Date" in df.columns else df.index
            fig = go.Figure(go.Scatter(
                x=list(dates), y=list(closes), mode="lines",
                line={"color": COLORS.ACCENT_PRIMARY, "width": 1.6},
                hovertemplate="%{x|%Y-%m-%d} · %{y:.2f}<extra></extra>",
            ))
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin={"l": 40, "r": 12, "t": 8, "b": 24},
                height=240,
                showlegend=False,
                xaxis={"showgrid": False},
                yaxis={"gridcolor": "#242424"},
            )
            detail["figure"] = fig
    except Exception as e:
        logger.warning("Home detail: price fetch failed for %s: %s", symbol, e)
    try:
        detail["articles"] = _serialize_articles(fetch_news_cached(symbol))
    except Exception as e:
        logger.warning("Home detail: news fetch failed for %s: %s", symbol, e)
    return detail


def _home_symbol_reports(symbol: str | None) -> list[dict]:
    """Recent reports for one symbol, newest first, epilogue pre-stripped.

    Feeds the Home reading pane. Stripping happens here so the layout module
    never imports the models package."""
    if not symbol:
        return []
    try:
        from models.single_agent import render_report_markdown
        reports = get_cache().get_trading_agent_reports(symbol, limit=5)
        for r in reports:
            # Epilogue stripped AND the verdict fields turned into list items,
            # so the Home reading pane gets the same discrete labelled lines
            # the modal and the PDF do.
            r["report_text"] = render_report_markdown(r.get("report_text") or "")
        return reports
    except Exception as e:
        logger.warning("Could not read %s reports for Home: %s", symbol, e)
        return []


_home_reports_memo: dict = {}


def _home_reports_by_symbol() -> dict:
    """Newest research report per symbol for the Home index. Best-effort:
    Home must render even if the reports table is unreachable. Short-TTL
    memo because the search box re-renders the index per keystroke.
    Keyed by viewer uid: visibility is per-user, so one user's memo must
    never serve another's private rows."""
    try:
        from services.auth_service import current_uid
        key = current_uid() or "anon"
    except Exception:
        key = "anon"
    hit = _home_reports_memo.get(key)
    if hit and (time.monotonic() - hit[0]) < 3.0:
        return hit[1]
    try:
        result = get_cache().latest_reports_by_symbol()
    except Exception as e:
        logger.warning("Could not read latest reports for Home: %s", e)
        return {}
    _home_reports_memo[key] = (time.monotonic(), result)
    return result


HOME_ROLLING_DAYS = 30

_ROUTES = ["/", "/analyze", "/performance", "/reports", "/schedule",
           "/activity", "/trace"]


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
    State("history-filter-model", "data"),
    State("home-symbol-filter", "data"),
    State("selected-symbols", "data"),
    State("current-period", "data"),
    State("home-cutoff-date", "data"),
)
def render_page(pathname, history_data, filter_symbols, filter_range,
                filter_specific, activity_scope, activity_since, outcome,
                model, home_symbol, watchlist, period, home_cutoff):
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
                            outcome, model, home_symbol=home_symbol,
                            watchlist=watchlist, period=period,
                            home_cutoff=home_cutoff)
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
    State("history-filter-model", "data"),
    State("history-filter-outcome", "data"),
    State("history-page", "data"),
    prevent_initial_call=True,
)
def render_prediction_log(n_clicks, filter_symbols, filter_range, filter_specific,
                          model, outcome, page):
    """Build the prediction log the first time its section is opened.

    Applies the SAME outcome slice the section header announces — the body
    used to ignore it, so "incorrect only (31)" expanded to every row."""
    if not n_clicks:
        raise PreventUpdate
    data = fetch_report_history(
        only={"predictions"},
        filters=history_filters(filter_symbols, filter_range, filter_specific,
                                model, outcome))
    section = build_predictions_section(data["predictions"], page=page)
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
    Input("history-filter-model", "data"),
    Input("history-page", "data"),
    State("url", "pathname"),
    prevent_initial_call=True,
)
def render_archive_body(history_data, filter_symbols, filter_range,
                        filter_specific, outcome, model, page, pathname):
    """Rebuild the filterable part of Performance or Reports.

    #archive-body exists only on those two routes, and only their own controls
    write these stores, so this never fires with the Output unmounted.
    """
    path = (pathname or "/").rstrip("/") or "/"
    if path not in ("/performance", "/reports"):
        raise PreventUpdate
    # A filter change starts from page one; only a pager click keeps offsets.
    page = page if ctx.triggered_id == "history-page" else {}
    if path == "/performance":
        data = fetch_report_history(
            only={"predictions"},
            filters=history_filters(filter_symbols, filter_range,
                                    filter_specific, model))
        return performance_page.body(data, filter_symbols, filter_range,
                                     filter_specific, outcome, model, page=page)
    data = fetch_report_history(
        only={"reports", "recommendations", "trading_agent_reports"})
    return reports_page.body(data, filter_symbols, filter_range, filter_specific,
                             page=page)


@callback(
    Output("history-page", "data"),
    Input({"type": "history-pager", "bucket": ALL, "dir": ALL}, "n_clicks"),
    State("history-page", "data"),
    prevent_initial_call=True,
)
def turn_history_page(clicks, page):
    """Prev/Next on one archive bucket. Offsets are clamped at render."""
    if not clicks or not any(c for c in clicks if c):
        raise PreventUpdate
    from layouts.history_sections import PAGE_SIZE
    trig = ctx.triggered_id
    page = dict(page or {})
    bucket, direction = trig["bucket"], trig["dir"]
    step = PAGE_SIZE[bucket] if direction == "next" else -PAGE_SIZE[bucket]
    page[bucket] = max(0, int(page.get(bucket) or 0) + step)
    return page


def _performance_rows(f_symbols, f_range, f_specific, f_outcome,
                      f_model="all") -> list[dict]:
    """The prediction rows the Performance page is currently rendering.

    Re-derived from the same helpers the page uses rather than scraped from
    the DOM, so View Data and Export can never disagree with the table above
    them.
    """
    preds = fetch_report_history(
        only={"predictions"},
        filters=history_filters(f_symbols, f_range or "all", f_specific,
                                f_model or "all", f_outcome or "all"),
    )["predictions"]

    keep = ("symbol", "model_name", "prediction_date", "target_date", "decision",
            "confidence", "up_probability", "previous_close", "predicted_close",
            "actual_close", "was_correct", "pnl_dollars")
    return [{k: p.get(k) for k in keep} for p in preds]


# Export and View Data render only on the pages whose data they act on
# (Analyze, Performance) — see create_data_actions. Their callbacks take them
# as allow_optional Inputs; there is no disabled state to manage anymore.


@callback(
    Output("history-filter-symbols", "data", allow_duplicate=True),
    Output("history-filter-outcome", "data", allow_duplicate=True),
    Input({"type": "perf-symbol-drill", "symbol": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def drill_into_symbol(clicks):
    """Narrow the page to one symbol: its calls, dates and outcomes.

    Clears any outcome slice on the way in — arriving at a symbol filtered to
    "wrong only" because of an earlier click would misrepresent its record.
    """
    if not clicks or not any(c for c in clicks if c):
        raise PreventUpdate
    return [ctx.triggered_id["symbol"]], "all"


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
    Output("history-filter-model", "data"),
    # Model dropdown: Performance only.
    Input("history-model-dropdown", "value", allow_optional=True),
    State("history-filter-model", "data"),
    prevent_initial_call=True,
)
def set_model_filter(value, current):
    """Write the scoreboard's model filter to its store.

    Writes the store only — the dropdown is seeded from it when the page is
    built, because it exists on one of six routes and an Output on a missing
    component is a hard error in Dash 4. The equality guard matters: this
    Input also fires when the dropdown is (re)mounted with the value it was
    just seeded with, and writing that back would re-render the body for no
    change.
    """
    if value is None or value == (current or "all"):
        raise PreventUpdate
    return value


@callback(
    Output("home-symbol-filter", "data"),
    Input({"type": "home-sym-btn", "symbol": ALL}, "n_clicks"),
    State("home-symbol-filter", "data"),
    prevent_initial_call=True,
)
def toggle_home_symbol(clicks, current):
    """Narrow the Home prediction board to one symbol, or widen back out.

    The symbol rows on the left ARE the filter control — clicking the active
    one (or the board's own "Show all" chip, which reuses the same pattern
    id) clears the narrow.
    """
    if not clicks or not any(c for c in clicks if c):
        raise PreventUpdate
    sym = ctx.triggered_id["symbol"]
    return None if sym == current else sym


@callback(
    Output("home-cutoff-date", "data"),
    Input("home-cutoff-dropdown", "value", allow_optional=True),
    State("home-cutoff-date", "data"),
    prevent_initial_call=True,
)
def set_home_cutoff(value, current):
    """Point the Home board at a past prediction cutoff.

    Selecting the newest cutoff stores None, so the board tracks "latest"
    again as new runs land instead of pinning to what happened to be newest
    at click time. The equality guard also swallows the firing that happens
    when navigation mounts the dropdown with its seeded value.
    """
    if not value:
        raise PreventUpdate
    from services import dashboard_service as ds
    cutoffs = ds.get_available_cutoffs()
    normalized = None if (cutoffs and value == cutoffs[0]) else value
    if normalized == (current or None):
        raise PreventUpdate
    return normalized


@callback(
    Output("home-symbol-list", "children"),
    Output("home-cohort-table", "children"),
    Output("home-board-title", "children"),
    Output("home-meta-wrap", "children"),
    Input("home-symbol-filter", "data"),
    Input("home-symbol-search", "value", allow_optional=True),
    # The rail shows watchlist membership, so it re-renders when the
    # watchlist changes (add/remove/clear/restore); the board follows the
    # cutoff override. Guarded to Home below — the stores fire everywhere
    # but the Outputs exist on one route.
    Input("selected-symbols", "data"),
    Input("home-cutoff-date", "data"),
    # A completed run must land here without a manual reload: predictions
    # arrive via the store status, report-only runs via the analysis store
    # (their research reports feed the right-hand reader).
    Input("prediction-store-status", "data"),
    Input("ai-analysis-store", "data"),
    State("url", "pathname"),
    prevent_initial_call=True,
)
def render_home_panes(active_symbol, search, watchlist, cutoff,
                      _pred_status, _ai_analysis, pathname):
    """Re-render the symbol rail, the board, and its date header together.

    One callback for all four so the row highlight, the board, and the
    "predicting X, data through Y" line can never disagree about which
    symbol or cutoff is active. The pathname guard PreventUpdates off Home,
    where these Outputs are not mounted (see render_archive_body).
    """
    path = (pathname or "/").rstrip("/") or "/"
    if path != "/":
        raise PreventUpdate
    from services import dashboard_service as ds
    cohort = ds.get_cohort(cutoff)
    if (not cohort or not cohort.get("prediction_date")) and not watchlist:
        raise PreventUpdate
    rail = home_page.symbol_list(cohort, _home_reports_by_symbol(),
                                 active_symbol, search or "",
                                 watchlist=watchlist or [])
    # Typing in the filter box only narrows the rail. The board (and the
    # per-symbol chart/news fetch behind it) rebuilds only when the
    # selection, the watchlist or the cutoff actually changed.
    if ctx.triggered_id == "home-symbol-search":
        return rail, dash.no_update, dash.no_update, dash.no_update
    if not cohort or not cohort.get("prediction_date"):
        return rail, dash.no_update, dash.no_update, dash.no_update
    cutoffs = ds.get_available_cutoffs()
    return (
        rail,
        home_page.cohort_table(cohort, active_symbol,
                               symbol_reports=_home_symbol_reports(active_symbol),
                               symbol_detail=_home_symbol_detail(active_symbol)),
        home_page.board_title(cutoffs, cutoff or (cutoffs[0] if cutoffs else None)),
        # Run-time/error metadata belongs to the latest run only; a
        # historical cutoff shows its cohort's own dates and outcomes.
        home_page.last_run_header(cohort, ds.get_last_run() if not cutoff
                                  else None),
    )


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
    # symbol-input lives in the always-mounted watchlist strip under the
    # toolbar, so the watchlist is editable from every page. clear-symbols
    # sits in the Home rail's ⋯ menu (one route → allow_optional). Two
    # remove patterns exist because the strip chips and the Home rail rows
    # are mounted at the same time on Home, and duplicate component ids are
    # a hard error.
    Input("symbol-input", "n_submit"),
    Input("clear-symbols-btn", "n_clicks", allow_optional=True),
    Input({"type": "add-symbol", "symbol": ALL}, "n_clicks"),
    Input({"type": "remove-symbol", "symbol": ALL}, "n_clicks"),
    Input({"type": "wl-remove", "symbol": ALL}, "n_clicks"),
    State("symbol-input", "value"),
    State("selected-symbols", "data"),
    prevent_initial_call=True,
)
def manage_symbols(input_submit, clear_click, add_clicks,
                   remove_clicks, wl_remove_clicks, input_value,
                   current_symbols):
    """Handle adding, removing and clearing watchlist symbols."""
    current_symbols = current_symbols or []

    # Get the triggered context safely
    if not ctx.triggered:
        raise PreventUpdate

    triggered = ctx.triggered_id

    # Guard against None triggered_id
    if triggered is None:
        raise PreventUpdate

    # Enter in the rail combobox adds whatever was typed
    if triggered == "symbol-input":
        if not input_submit or not input_value:
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

    # Clear the whole selection in one click (rail ⋯ menu)
    if triggered == "clear-symbols-btn":
        if not clear_click:
            raise PreventUpdate
        from services import progress_service as prog
        prog.emit("action", f"Symbols cleared ({', '.join(current_symbols) or 'none'})")
        return [], dash.no_update

    # One-click ＋ on a from-last-run rail row
    if isinstance(triggered, dict) and triggered.get("type") == "add-symbol":
        if not any(c and c > 0 for c in add_clicks):
            raise PreventUpdate
        symbol = triggered["symbol"]
        if symbol not in current_symbols:
            current_symbols = current_symbols + [symbol]
            from services import progress_service as prog
            prog.emit("action", f"Symbol added to watchlist: {symbol}")
        return current_symbols, dash.no_update

    # Handle remove buttons — the Home rail rows and the global strip chips
    if isinstance(triggered, dict) \
            and triggered.get("type") in ("remove-symbol", "wl-remove"):
        clicks = (remove_clicks if triggered["type"] == "remove-symbol"
                  else wl_remove_clicks)
        if not any(c and c > 0 for c in clicks):
            raise PreventUpdate
        symbol = triggered["symbol"]
        if symbol in current_symbols:
            current_symbols = [s for s in current_symbols if s != symbol]
            from services import progress_service as prog
            prog.emit("action", f"Symbol removed: {symbol}")
        return current_symbols, dash.no_update

    raise PreventUpdate


@callback(
    Output("watchlist-strip-chips", "children"),
    Input("selected-symbols", "data"),
)
def render_watchlist_strip(symbols):
    """The always-visible watchlist chips under the toolbar."""
    if not symbols:
        return html.Span("empty — type a symbol and press Enter",
                         className="wl-strip-empty")
    return [
        html.Span(
            [
                html.Span(sym, className="wl-chip-text"),
                html.Button(
                    "✕",
                    id={"type": "wl-remove", "symbol": sym},
                    className="wl-chip-remove",
                    title=f"Remove {sym} from the watchlist",
                ),
            ],
            className="wl-chip",
        )
        for sym in symbols
    ]


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
    # Refresh lives on Analyze's action bar now — absent on other routes.
    Input("refresh-data-btn", "n_clicks", allow_optional=True),
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
    # refresh_click guard: the button is Analyze-local, so this callback also
    # fires when navigation mounts it (n_clicks None) — that must not force
    # a cache-bypassing refetch.
    force_refresh = ((triggered == "refresh-data-btn" and bool(refresh_click))
                     or not cache_enabled)
    if triggered == "refresh-data-btn" and refresh_click:
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
    Input("refresh-data-btn", "n_clicks", allow_optional=True),
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
        return _serialize_articles(articles)

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
    State("run-symbols-store", "data"),
    State("stock-data-store", "data"),
    State("run-date-picker", "date"),
    State("run-lookback", "value"),
    State("run-max-articles", "value"),
    State("run-model", "value"),
    State("run-type", "value"),
    State("run-recs", "value"),
    State("run-recs-model", "value"),
    State("run-evidence", "value"),
    State("run-tools", "value"),
    prevent_initial_call=True,
)
async def generate_ai_analysis(n_clicks, retry_clicks, scope,
                               run_symbols, stock_data, run_date, lookback,
                               max_articles_val, model, depth, recs,
                               recs_model, run_evidence, run_tools):
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
    # The run's symbol set comes from the dialog (watchlist / last cohort /
    # custom), not from the watchlist directly. The report does its own
    # point-in-time fetch per symbol; the browser's news store plays no part.
    symbols = (run_symbols or {}).get("symbols") or []
    if not (n_clicks or retry_clicks) or not symbols:
        raise PreventUpdate

    # Metric/signal context for symbols outside the browser stores (e.g. a
    # cohort-scoped or single-symbol run) is fetched server-side.
    missing = [s for s in symbols
               if not ((stock_data or {}).get(s) or {}).get("prices")]
    if missing:
        stock_data = await asyncio.to_thread(
            _fill_run_inputs, symbols, stock_data)

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
    from services.news_window import (
        RunParameterMissing, normalize_article_cap, normalize_lookback)
    try:
        overnight_news, lookback_days = normalize_lookback(lookback)
        max_articles = normalize_article_cap(max_articles_val)
    except RunParameterMissing as e:
        # Refuse, visibly. No default: a report on a window the user did not
        # pick would be indistinguishable from the one they asked for.
        from services import progress_service as prog
        prog.emit("error", f"Report rejected — {e}")
        prog.finish_run(f"Report failed — {e}")
        return {"failed": True, "error": str(e), "scope": scope,
                "run_seq": n_clicks, "generated_at": datetime.now().isoformat()}
    window_desc = ("overnight close→open" if overnight_news
                   else f"{lookback_days}d")
    report_model = model or "gpt-5.6-luna"
    include_thesis_flag = (depth or "thesis") != "standard"
    recs_mode = recs or "auto"
    recs_model_val = recs_model or "claude-sonnet-5"
    # Terminal-derived evidence blocks (modal checklist); None only before
    # the modal has rendered once — treat as the default, both on.
    evidence_sel = (sorted(run_evidence) if run_evidence is not None
                    else sorted(MODEL.DEFAULT_EVIDENCE))
    # Tools are opt-in: absent means none (the backend never switches one on).
    tools_sel = sorted(set(run_tools or []))
    report_provider = "openai" if report_model.startswith("gpt-") else "anthropic"
    # Per-symbol analysis is always the full research agent now — the shallow
    # summarize_news_structured pass produced a thinner second opinion on the
    # same data and was removed (2026-08-08). Full pipeline still gets its
    # research through the prediction stage's trading_agents model; running
    # it here too would double the LLM spend for identical output.
    include_research = not is_full

    from services import progress_service as prog
    if is_full:
        prog.start_run(f"Full Analysis — {len(symbols or [])} symbols, "
                       f"target {target_str} (data through {as_of_str})"
                       + (" (backtest)" if is_backtest else ""))
    else:
        # A report-only run is a run too: without its own boundary the panel
        # keeps the previous run's green check through the whole generation
        # and polls at the idle rate.
        prog.start_run(f"Report — {len(symbols or [])} symbols, "
                       f"target {target_str} (data through {as_of_str})"
                       + (" (backtest)" if is_backtest else ""))
    prog.emit("ai", f"AI Report starting for {', '.join(symbols or [])}")
    prog.emit("action", f"Options: model={report_model}, window={window_desc}, "
                        f"type={'thesis' if include_thesis_flag else 'standard'}, "
                        f"research={'pipeline' if is_full else 'on'}, "
                        f"recs={recs_mode} ({recs_model_val}), "
                        f"target {target_str}, data through {as_of_str}")

    # One data_load event covering EVERY symbol's price input for this run,
    # whatever its origin — the browser session store (the normal case, which
    # previously produced no data_load at all) or the server gap-fill above.
    # In-memory parse of frames the report reads anyway; nothing is refetched.
    price_inputs: dict[str, dict] = {}
    for sym in (symbols or []):
        entry = (stock_data or {}).get(sym) or {}
        if not entry.get("prices"):
            price_inputs[sym] = {"bars": 0, "error": "no price data"}
            continue
        try:
            _pf = pd.read_json(StringIO(entry["prices"]))
            if "Date" in _pf.columns:
                _pf = _pf.set_index("Date")
            _pf.index = pd.to_datetime(_pf.index)
            _pf = _pf[_pf.index <= as_of_str]
            price_inputs[sym] = {
                "bars": len(_pf),
                "start": str(_pf.index.min())[:10] if len(_pf) else None,
                "end": str(_pf.index.max())[:10] if len(_pf) else None,
                "source": entry.get("source") or "session-store",
            }
        except Exception as e:
            price_inputs[sym] = {"bars": 0, "error": str(e)[:120]}
    # No `period`: the store's label is the UI display window, not the data
    # window — per-symbol start/end record what was actually consumed.
    prog.emit("action",
              f"Price inputs resolved: "
              f"{sum(1 for v in price_inputs.values() if v.get('bars'))}"
              f"/{len(symbols or [])} symbols (data through {as_of_str})",
              payload={"event": "data_load", "by_symbol": price_inputs})

    # News window: the SAME point-in-time fetch the modal preview showed and
    # the models subprocess uses — windowed [as_of - lookback, as_of] (or the
    # overnight close→open gap), capped at the newest max_articles, lookahead-
    # safe. The dcc.Store's articles are never used here: they hold whatever
    # the last background refresh grabbed from "now", which is the wrong
    # window for any backtest and an unlabelled one live.
    from services.news_window import (
        describe_news_window, fetch_run_news, news_window_payload)
    articles_by_symbol, news_stats = await asyncio.to_thread(
        fetch_run_news, symbols or [], as_of_str, target_str,
        overnight=overnight_news, lookback_days=lookback_days,
        max_articles=max_articles)
    total_articles = sum(len(a) for a in articles_by_symbol.values())
    news_payload = news_window_payload(
        overnight=overnight_news, lookback_days=lookback_days,
        max_articles=max_articles, as_of=as_of_str, target=target_str,
        stats_by_symbol=news_stats)
    prog.emit("news", describe_news_window(news_payload), payload=news_payload)
    news_down = [s for s, v in news_stats.items() if v["status"] == "unavailable"]
    if news_down:
        prog.emit("error",
                  f"News source unavailable for {len(news_down)} of "
                  f"{len(symbols or [])} symbols: {', '.join(news_down)} — "
                  f"the report has NO news for them; treat their sections "
                  f"as unsupported, not as quiet weeks.")

    # One key dict for both the cache check here and the store at the end —
    # and the trace payload records its hash plus a compact summary of what
    # fed it, so a cross-restart cache miss stops being undiagnosable.
    from services.analysis_runner import report_cache_key as _report_key
    report_cache_key = _report_key(
        articles_by_symbol, symbols or [], as_of_str, report_model,
        lookback_days, include_thesis_flag, evidence=evidence_sel,
        max_articles=max_articles, overnight=overnight_news,
        include_research=include_research, recs_mode=recs_mode,
        tools=tools_sel)

    # Check persistent cache (Postgres + S3) before running LLM
    if _s3_available and ctx.triggered_id != "ai-retry-btn":
        try:
            from services import persistence_service as ps
            from services.analysis_runner import (
                _report_key_summary as report_key_summary,
            )
            data_hash = ps.compute_data_hash(report_cache_key)
            report_cache_payload = {
                "event": "cache",
                "kind": "ai_report",
                "input_data_hash": data_hash,
                "summary": report_key_summary(articles_by_symbol,
                                              sorted(symbols or []),
                                              report_cache_key),
            }
            cached = ps.get_cached_report(None, as_of_str, "ai_report", data_hash)
            if not cached:
                prog.emit("ai", "AI report cache miss — generating fresh",
                          payload={**report_cache_payload, "outcome": "miss"})
            if cached:
                logger.info("AI report cache hit (Postgres/S3)")
                prog.emit("ai", "AI report cache hit — regeneration skipped",
                          payload={**report_cache_payload, "outcome": "hit"})
                cached_result = json.loads(cached)
                cached_result["from_cache"] = True
                # Re-stamp for THIS click — the stored copy carries the
                # correlation keys of the run that originally produced it.
                cached_result["run_seq"] = n_clicks
                cached_result["scope"] = scope
                if not is_full and recs_mode == "off":
                    # Nothing left to run for this scope: close the run.
                    prog.finish_run(f"Report complete ({len(symbols or [])} "
                                    "symbols, from cache)")
                return cached_result
        except Exception as e:
            logger.debug(f"AI report cache check failed: {e}")

    llm = get_llm()
    stock_data = stock_data or {}

    result = {
        "overall": None,
        "news_window": {"lookback_days": lookback_days, "overnight": overnight_news,
                        "max_articles": max_articles},
        "by_symbol": {},
        "as_of": as_of_str,
        "generated_at": datetime.now().isoformat(),
        # Correlation keys: run_seq ties this payload to the confirm click
        # that produced it (the model-signals _meta carries the same value),
        # and scope lets the synthesis callback tell a report-only payload
        # from a full-pipeline one even if a stale requested flag lingers.
        "run_seq": n_clicks,
        "scope": scope,
    }

    # Per-symbol context: fundamentals, validated metric blocks, events, peers
    enriched_stock_data = {}
    extra_blocks_by_symbol = {}
    for symbol in (symbols or []):
        # One line per symbol: this loop is network-bound (profile, events,
        # peers, options, quality) and used to run in total silence.
        prog.emit("ai", f"{symbol}: gathering fundamentals, options, "
                        f"quality screens…")
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

            # Options positioning + Bad Apples quality screen — both are
            # as-of-safe and cached per (symbol, as_of) inside their services.
            # Gated by the modal's Evidence-blocks checklist.
            if "options" in evidence_sel:
                from services.options_service import (
                    get_put_call_metrics, format_options_block,
                )
                block = format_options_block(
                    symbol, get_put_call_metrics(symbol, as_of_str))
                if block:
                    blocks["options"] = block
            if "quality" in evidence_sel:
                from services.bad_apples_service import (
                    analyze_symbol as _ba_analyze, format_bad_apples_block,
                )
                block = format_bad_apples_block(
                    symbol, _ba_analyze(symbol, as_of_str))
                if block:
                    blocks["quality"] = block
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
        from services import usage_service as _usage
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
        # Runs in a worker thread via asyncio.to_thread (context copies in),
        # but the label is set here so the trace/spend attribution never
        # depends on what the event-loop context happened to hold.
        with _usage.track("research", symbol=sym, trade_date=as_of_str,
                          section=f"research:{sym}"):
            # The run's own windowed news — without it the agent fetched
            # its own default window and the report footer printed that
            # default while the dialog had said something else.
            res = model_obj.predict(sym, df, as_of=as_of_str,
                                    model=report_model,
                                    include_thesis=include_thesis_flag,
                                    news=articles_by_symbol.get(sym) or [],
                                    news_lookback_days=lookback_days,
                                    evidence=evidence_sel,
                                    # Source failure vs quiet week: the
                                    # research arm refuses to write blind.
                                    news_status=(news_stats.get(sym) or {}).get("status"),
                                    target_date=target_str,
                                    tools=tools_sel)
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
                # Both numbers, kept apart: `confidence` is the measured
                # track-record weight, `stated_conviction` is what the report
                # itself claimed. The UI labels each; it cannot if only one
                # travels.
                "confidence": r.confidence,
                "stated_conviction": details.get("stated_conviction"),
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

    # The research report IS the per-symbol text analysis. The shallow
    # summarize_news_structured per-symbol pass that used to run instead of
    # it was removed 2026-08-08: it produced a thinner opinion on the same
    # data that could contradict the research verdict in the same tab, and
    # cost an extra LLM call per symbol. Full Analysis still gets its
    # research from the prediction stage's trading_agents model, so nothing
    # runs here in that scope. The portfolio-level overall call below is
    # separate either way (the research reports are single-symbol and
    # cannot replace it).

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

        def _run_overall(*args, **kw):
            from services import usage_service as _usage
            with _usage.track("ai_report", trade_date=as_of_str,
                              section="ai_report:overall"):
                return llm.summarize_news_structured(*args, **kw)

        aio_tasks.append(_run_task(
            "overall", None, _run_overall, overall_articles, symbols or [],
            stock_data=enriched_stock_data,
            as_of_date=as_of_str,
            extra_blocks={"metrics": overall_metrics} if overall_metrics else None,
            include_thesis=include_thesis_flag,
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
        # Count what the report covered. Full scope defers the per-symbol
        # research to the model stage, so by_symbol is legitimately empty
        # there — fall back to the run's symbol list rather than saying "0".
        n_covered = len(result["by_symbol"]) or len(symbols or [])
        if is_full:
            prog.emit("ai", f"AI Report complete ({n_covered} symbols) — "
                            "model predictions next")
        elif recs_mode != "off":
            prog.emit("ai", f"AI Report complete ({n_covered} symbols) — "
                            "synthesizing recommendations next")
        else:
            prog.emit("ai", f"AI Report complete ({n_covered} symbols)")

    # Per-symbol options positioning + quality screen ride the payload so the
    # synthesis prompt and every renderer read one source. After the failed
    # check (an entry per symbol must not mask an empty report), off the event
    # loop (both services hit the network on a cache miss).
    if not result.get("failed"):
        from services.analysis_runner import attach_positioning_quality
        result = await asyncio.to_thread(
            attach_positioning_quality, result, symbols or [], as_of_str,
            evidence_sel)

    # Store to Postgres/S3 for future cache hits
    if _s3_available and not result.get("failed"):
        try:
            from services import persistence_service as ps
            data_hash = ps.compute_data_hash(report_cache_key)
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

    # Report-only runs end here unless recommendations still have to run
    # (the synthesis callback closes those). Full runs are closed by the
    # persist/synthesis stages once the models land.
    if not is_full:
        if result.get("failed"):
            prog.finish_run("Report failed — no analysis produced")
        elif recs_mode == "off":
            prog.finish_run(f"Report complete "
                            f"({len(result['by_symbol']) or len(symbols or [])}"
                            " symbols)")

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


# =============================================================================
# HISTORICAL REPORT DOWNLOAD
# =============================================================================


# =============================================================================
# DATA MODAL CALLBACKS
# =============================================================================


@callback(
    Output("data-modal", "is_open"),
    Output("data-table-container", "children"),
    Input("view-data-btn", "n_clicks", allow_optional=True),
    Input("modal-close-btn", "n_clicks"),
    State("selected-symbols", "data"),
    State("data-modal", "is_open"),
    State("url", "pathname"),
    State("history-filter-symbols", "data"),
    State("history-filter-date-range", "data"),
    State("history-filter-date-specific", "data"),
    State("history-filter-outcome", "data"),
    State("history-filter-model", "data"),
    prevent_initial_call=True,
)
def toggle_data_modal(view_click, close_click, symbols, is_open, pathname,
                      f_symbols, f_range, f_specific, f_outcome, f_model):
    """Show the rows behind whatever the current page is rendering.

    The button used to read the Analyze page's stores wherever you clicked
    it, so on Performance it reported "No stocks selected" and never opened —
    a control that looked broken rather than inapplicable.
    """
    triggered = ctx.triggered_id

    if triggered == "modal-close-btn":
        return False, dash.no_update

    # The button is page-local now (allow_optional Input), so this callback
    # also fires when navigation MOUNTS it, with n_clicks still None — that
    # firing must not open the modal.
    if triggered == "view-data-btn" and not view_click:
        raise PreventUpdate

    path = (pathname or "/").rstrip("/") or "/"

    if triggered == "view-data-btn" and path == "/performance":
        rows = _performance_rows(f_symbols, f_range, f_specific, f_outcome,
                                 f_model)
        if not rows:
            return True, html.Div("No predictions in the current filter",
                                  className="text-muted")
        import pandas as _pd
        df = _pd.DataFrame(rows)
        return True, html.Div([
            html.Div(f"{len(df)} rows in scope · showing the first 200",
                     className="scoreboard-summary"),
            dbc.Table.from_dataframe(df.head(200), striped=True, bordered=True,
                                     hover=True, responsive=True,
                                     className="table-dark"),
        ])

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
    Input("export-data-btn", "n_clicks", allow_optional=True),
    Input("modal-export-btn", "n_clicks"),
    State("selected-symbols", "data"),
    State("stock-data-store", "data"),
    State("indicator-toggles", "value", allow_optional=True),
    State("model-signals-store", "data"),
    State("ai-analysis-store", "data"),
    State("recommendations-store", "data"),
    State("url", "pathname"),
    State("history-filter-symbols", "data"),
    State("history-filter-date-range", "data"),
    State("history-filter-date-specific", "data"),
    State("history-filter-outcome", "data"),
    State("history-filter-model", "data"),
    prevent_initial_call=True,
)
def export_data(export_click, modal_export_click, symbols, stock_data,
                indicators, model_signals, ai_analysis, recommendations,
                pathname, f_symbols, f_range, f_specific, f_outcome, f_model):
    """Export everything on screen as a multi-sheet .xlsx.

    Replaces the old single-symbol Parquet dump. Sheets are dynamic: prices
    (+ only the toggled indicators) per symbol, then Predictions / AI
    Analysis / Recommendations whenever those stores hold data.
    """
    # Mount-firing guard (see toggle_data_modal): navigating to a page that
    # renders the button must not trigger a download.
    if ctx.triggered_id == "export-data-btn" and not export_click:
        raise PreventUpdate
    path = (pathname or "/").rstrip("/") or "/"
    if path == "/performance":
        # Export what the page is showing, filters and all — the scoreboard is
        # the thing worth taking away from this page, not the Analyze stores.
        rows = _performance_rows(f_symbols, f_range, f_specific, f_outcome,
                                 f_model)
        if not rows:
            raise PreventUpdate
        import pandas as _pd
        fname = f"quantnews_predictions_{datetime.now():%Y-%m-%d_%H%M}.csv"
        from services import progress_service as prog
        prog.emit("action", f"CSV export: {fname} ({len(rows)} rows)")
        return dcc.send_data_frame(_pd.DataFrame(rows).to_csv, fname, index=False)

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
    State("run-symbols-store", "data"),
    State("ensemble-config-store", "data"),
    State({"type": "run-model-check", "model": ALL}, "value"),
    State("run-ensemble-check", "value"),
    State({"type": "run-ens-member", "model": ALL}, "value"),
    State({"type": "run-ens-weight", "model": ALL}, "value"),
    State("run-ensemble-method", "value"),
    State("run-ensemble-min-agree", "value"),
    State("run-date-picker", "date"),
    State("run-lookback", "value"),
    State("run-max-articles", "value"),
    State("run-model", "value"),
    State("run-type", "value"),
    State("run-evidence", "value"),
    State("run-tools", "value"),
    background=True,
    running=[
        (Output("prediction-running-indicator", "style"), {"display": "block"}, {"display": "none"}),
    ],
    prevent_initial_call=True,
)
def generate_model_signals(n_clicks, scope, stock_data, run_symbols,
                           ensemble_config, model_checks,
                           run_ensemble, ens_members, ens_weights, ens_method,
                           ens_min_agree, predict_date_str, lookback,
                           max_articles_val, research_model,
                           research_depth, run_evidence, run_tools):
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
    symbols = (run_symbols or {}).get("symbols") or []
    if not n_clicks or not symbols:
        raise PreventUpdate

    # The model checkboxes are now honoured on every scope. Full pipeline used
    # to overwrite them with [True] * 5, so the boxes you unticked ran anyway.
    is_full_analysis = scope == "full"
    from services import progress_service as _prog

    def _abort_run(exc: Exception) -> dict:
        """Close the run as a failure and hand downstream a marked payload.

        logger.exception, not error: this body runs in the background
        subprocess, where an escaped exception is swallowed into Dash's job
        error — the browser sees a bare HTTP 500 and the server log NOTHING.
        The traceback must be written here or it exists nowhere.

        A marked payload, not {}: downstream must be able to tell "the run
        failed" from "no predictions". The empty dict cleared the running
        badge as if successful and left the panel spinning forever.
        """
        logger.exception("Model signal generation error")
        _prog.emit("error", f"Model run failed: {str(exc)[:200]}")
        _prog.finish_run(("Full Analysis" if is_full_analysis
                          else "Predictions")
                         + f" failed — {str(exc)[:120]}")
        return {"_run_failed": str(exc), "_meta": {"run_seq": n_clicks}}

    # Everything after the guards runs under a catch-all. A one-line bug
    # before the old try boundary shipped as "every run dies at +1s with a
    # bare HTTP 500 and a forever-running badge" — nothing here may execute
    # unprotected.
    try:
        if not is_full_analysis:
            # Models-only owns the feed; the full pipeline started it
            # already. Started BEFORE the input fill so the first emits below
            # group under this run, not the previous one.
            _prog.start_run(f"Predictions — {len(symbols or [])} symbols")

        # First line before any heavy work: the subprocess spends its first
        # seconds on imports and input fetches with nothing on the panel.
        # (emit itself touches only diskcache + Postgres — no torch.)
        # The pid rides along and is recorded beside the run key, so the
        # server's watchdog can tell "worker died" from "worker still
        # grinding" — this process has been observed dying without a
        # traceback (torch/MPS native teardown), which previously left the
        # run spinning forever.
        _prog.emit("models", "Preparing model engine (first run loads model "
                             "weights)…",
                   payload={"event": "run_process", "pid": os.getpid()})
        _prog.record_run_pid()

        # The dialog's news window and cap govern the models too — the
        # sentiment and research models used to read the browser's news
        # store (a rolling week from "now", 50 articles), whatever the
        # dialog said. The window's days ride along so the research agent
        # does not re-filter to its own default.
        from services.news_window import (
            normalize_article_cap, normalize_lookback)
        overnight_news, lookback_days = normalize_lookback(lookback)
        max_articles = normalize_article_cap(max_articles_val)
        research_kwargs = {"news_lookback_days": lookback_days}
        if is_full_analysis:
            # The research report (trading_agents) honours the report
            # model/depth choices; `research_model` is a distinct kwarg so no
            # other model can mistake it for its own.
            research_kwargs.update({
                "research_model": research_model or None,
                "include_thesis": (research_depth or "thesis") != "standard",
                # Modal checklist; None only before the modal rendered once.
                "evidence": sorted(run_evidence) if run_evidence is not None
                            else sorted(MODEL.DEFAULT_EVIDENCE),
                "tools": sorted(set(run_tools or [])),
            })

        # Module-level os — a function-local `import os` here once shadowed
        # the name for the WHOLE body, so the pid emit above raised
        # UnboundLocalError on every run. Never re-import module-level names
        # inside this function.
        os.environ["_DASH_BG_SUBPROCESS"] = "1"

        from datetime import date as date_cls

        # The picker holds the TARGET session — the close being predicted.
        # Models are truncated to `predict_date`, the previous trading day,
        # so a Monday target trains and scores on nothing after the
        # preceding Friday.
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

        # Build the ensemble config from the dialog's own controls: they are
        # what the user sees at click time, whereas the store State could
        # still be one sync callback behind the last edit. The store keeps
        # the same values for the drawer and live recompute.
        from config import MODEL as _MODEL
        if ens_members is not None:
            weights = {}
            for model_id, w in zip(_ALL_MODEL_IDS, ens_weights or []):
                try:
                    weights[model_id] = round(float(w), 1)
                except (TypeError, ValueError):
                    weights[model_id] = 1.0
            try:
                min_agree_val = int(ens_min_agree)
            except (TypeError, ValueError):
                min_agree_val = _MODEL.ENSEMBLE_MIN_AGREE
            ensemble_config = {
                "enabled_models": [m for m, on
                                   in zip(_ALL_MODEL_IDS, ens_members) if on],
                "weights": weights,
                "method": ens_method or _MODEL.ENSEMBLE_DEFAULT_METHOD,
                "min_agree": min_agree_val,
            }
    except Exception as e:
        return _abort_run(e)

    try:
        from services.analysis_runner import load_market_data, run_predictions
        from services.news_window import (
            describe_news_window, fetch_run_news, news_window_payload)

        # Model inputs come from the server-side price cache at a fixed
        # depth, never from the browser store: the store holds whatever the
        # chart's display period fetched (6 months for any window under a
        # year), so the tree models trained on ~120 bars with SMA-200 blank
        # whenever the chart happened to be on "1 month". This is the same
        # loader the scheduled run uses.
        stock_data = load_market_data(symbols, period="2y")
        priced = [s for s in symbols if s in stock_data]
        for sym in symbols:
            if sym not in stock_data:
                _prog.emit("error", f"{sym}: skipped — no price data available")

        # Point-in-time news for the run: the same fetch, window and cap the
        # report and the dialog preview use.
        run_news, news_stats = fetch_run_news(
            priced, str(predict_date), str(target_date),
            overnight=overnight_news, lookback_days=lookback_days,
            max_articles=max_articles)
        news_payload = news_window_payload(
            overnight=overnight_news, lookback_days=lookback_days,
            max_articles=max_articles, as_of=str(predict_date),
            target=str(target_date), stats_by_symbol=news_stats)
        _prog.emit("news", describe_news_window(news_payload),
                   payload=news_payload)
        news_down = [s for s, v in news_stats.items()
                     if v["status"] == "unavailable"]
        if news_down:
            _prog.emit("error",
                       f"News source unavailable for {len(news_down)} of "
                       f"{len(priced)} symbols: {', '.join(news_down)}. "
                       f"Their sentiment and research models run WITHOUT "
                       f"news; treat those calls as unsupported.")

        # ONE implementation of the model stage (services.analysis_runner):
        # the interactive copy that lived here had drifted from the
        # scheduled one — no pipeline-epoch/news-status/regime stamps (so
        # the scheduler could never reuse a UI run and the scoreboard could
        # not tell supported calls from blind ones), and no abstention when
        # the news source was down. force=True: a click is an explicit
        # request to run, not a cache lookup.
        results = run_predictions(
            priced, stock_data, run_news,
            target_date=target_date, cutoff_date=predict_date,
            models=selected_models,
            research_model=research_kwargs.get("research_model"),
            news_status_by_symbol={s: v["status"] for s, v in news_stats.items()},
            include_thesis=research_kwargs.get("include_thesis", True),
            force=True,
            evidence=research_kwargs.get("evidence"),
            news_lookback_days=lookback_days,
            ensemble_config=ensemble_config,
            run_ensemble=bool(run_ensemble),
        )
        # Ties this store to the confirm click that produced it, so the
        # synthesis stage can pair it with the same click's AI report.
        results["_meta"]["run_seq"] = n_clicks
        return results

    except Exception as e:
        return _abort_run(e)


@callback(
    Output("prediction-store-status", "data"),
    Input("model-signals-store", "data"),
    State("full-analysis-requested", "data"),
    State("ai-analysis-store", "data"),
    prevent_initial_call=True,
)
def persist_predictions(signals, fa_requested, ai_analysis):
    """Persist model predictions to Postgres in the server process.

    This callback receives the serializable dict from model-signals-store
    (produced by the background-callback subprocess) and writes it out. For
    backtest predictions, uses the selected date instead of today and
    auto-evaluates against actual prices.
    """
    if not signals:
        raise PreventUpdate
    # The worker's output is in hand — its pid no longer stands for the
    # run's health, alive or not (the watchdog must not flag a subprocess
    # that exited after delivering).
    from services import progress_service as _prog_pid
    _prog_pid.clear_run_pid()
    if signals.get("_run_failed"):
        # The background handler already emitted the failure and finished
        # the run — pass a failed status through so the pages listening on
        # this store still refresh, instead of pretending "Stored 0".
        return {"failed": signals["_run_failed"], "count": 0,
                "stored_at": str(datetime.now())}

    meta = signals.get("_meta") or {}
    predict_date_str = meta.get("predict_date")
    is_backtest = meta.get("is_backtest", False)

    try:
        # ONE writer (services.analysis_runner.persist_predictions): the copy
        # that lived here recorded the model that was ASKED for on research
        # reports instead of the one that answered after a provider fallback.
        from services.analysis_runner import persist_predictions as _persist
        stored, evaluated = _persist(signals)
        from services import progress_service as prog
        # New rows: the launch screen must show this run, not the last one.
        from services.dashboard_service import invalidate_memo
        invalidate_memo()
        if not fa_requested:
            prog.finish_run(f"Predictions complete — {stored} stored")
        elif ((ai_analysis or {}).get("failed")
              and (ai_analysis or {}).get("run_seq") == meta.get("run_seq")):
            # This run's report already failed: no synthesis is coming, so
            # don't announce a handoff that never happens. The synthesis
            # callback sees the same store pair and closes the run.
            pass
        else:
            prog.emit("luna", "Handing off to recommendation synthesis…")

        return {
            "stored_at": str(datetime.now()),
            "count": stored,
            "evaluated": evaluated,
            "is_backtest": is_backtest,
            "predict_date": predict_date_str,
        }

    except Exception as e:
        logger.error(f"Prediction persistence error: {e}")
        from services import progress_service as prog
        prog.emit("error", f"Prediction storage failed: {str(e)[:200]}")
        if not fa_requested:
            prog.finish_run("Predictions finished with errors — storage "
                            "failed")
        else:
            # The signals are still in the store, so synthesis can proceed
            # and owns the finish; only the persisted rows are missing.
            prog.emit("luna", "Handing off to recommendation synthesis…")
        return {"error": str(e), "stored_at": str(datetime.now())}


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
            # Zero is the usual outcome — the 6pm scheduler and post-backtest
            # auto-eval normally score everything first. Only warn when mature
            # rows are genuinely stuck without a close to score against.
            backlog = cache.evaluation_backlog()
            pending = backlog.get("pending_mature", 0)
            if pending:
                dates = ", ".join(sorted(backlog.get("by_target_date", {}))[:4])
                msg = (f"{pending} prediction{'s' if pending != 1 else ''} "
                       f"due for scoring could not be evaluated — no closing "
                       f"price stored for target date{'s' if pending != 1 else ''} "
                       f"{dates}. Load those symbols to refresh price data.")
                icon = "warning"
            else:
                msg = ("All caught up — every matured prediction is already "
                       "scored. The rest will be evaluable once their target "
                       "date closes.")
                icon = "success"
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
    # A report-only run never touches prediction-store-status, so reload
    # when the report itself lands and when a recommendation run persists —
    # otherwise a fresh report only appears after re-navigating.
    Input("ai-analysis-store", "data"),
    Input("recommendations-store", "data"),
)
def load_report_history(symbols, _prediction_status, _download_done,
                        _eval_status, _ai_analysis, _recommendations):
    """Signal that the archive changed. The pages read the archive live
    (fetch_report_history, filtered in SQL); shipping every prediction row
    to the browser here was the reason the loaders had row caps at all."""
    _history_memo.clear()
    return {"refreshed_at": datetime.now().isoformat()}


_HISTORY_TTL_S = 3.0
_history_memo: dict = {}


def history_filters(filter_symbols=None, filter_range="all", specific_date=None,
                    model="all", outcome="all") -> dict:
    """The History filter stores as a SQL-ready dict (dates on target_date).
    Same semantics as layouts.history_sections.filter_items."""
    from datetime import date as _date, timedelta as _td
    from layouts.history_sections import current_session
    start = end = None
    if specific_date:
        start = end = _date.fromisoformat(str(specific_date)[:10])
    elif filter_range == "today":
        start = end = current_session()
    elif filter_range and filter_range != "all":
        days = {"7d": 7, "30d": 30, "90d": 90}.get(filter_range, 0)
        if days:
            start = _date.today() - _td(days=days)
    return {"symbols": sorted(filter_symbols) if filter_symbols else None,
            "start": start, "end": end,
            "model": model if model and model != "all" else None,
            "outcome": outcome if outcome and outcome != "all" else None}


def fetch_report_history(symbols=None, only=None, filters=None) -> dict:
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
    filters = filters or {}
    key = (tuple(sorted(symbols)) if symbols else (),
           tuple(sorted(only)) if only else (),
           tuple(sorted((k, str(v)) for k, v in filters.items())))
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

        # Research reports — the recent archive for everyone, plus deeper
        # per-symbol history for the current watchlist. Loading ONLY the
        # watchlist made every other symbol's past reports unreachable: the
        # filter bar narrows what was loaded, it cannot load more.
        ta_reports = []
        if wanted("trading_agent_reports"):
            ta_reports = cache.get_all_trading_agent_reports(limit=None)
            if symbols:
                seen = {r.get("id") for r in ta_reports}
                for symbol in symbols:
                    ta_reports.extend(
                        r for r in cache.get_trading_agent_reports(symbol, limit=10)
                        if r.get("id") not in seen
                    )
        ta_reports.sort(key=lambda r: r.get("created_at", ""), reverse=True)
        result["trading_agent_reports"] = ta_reports

        # No row caps: the pages paginate what they render, and a capped
        # load made any date past the cap look like an empty day.
        if wanted("reports"):
            result["reports"] = cache.list_report_catalog(limit=None)
        if wanted("predictions"):
            result["predictions"] = cache.query_predictions(
                symbols=filters.get("symbols"), start=filters.get("start"),
                end=filters.get("end"), model=filters.get("model"),
                outcome=filters.get("outcome"))
        if wanted("recommendations"):
            result["recommendations"] = cache.list_recommendation_runs(limit=None)

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
    Output("progress-snap-store", "data"),
    Output("progress-fp-store", "data"),
    Input("progress-interval", "n_intervals"),
    State("progress-interval", "interval"),
    State("progress-snap-store", "data"),
    State("progress-fp-store", "data"),
)
def render_progress_panel(_n, _current_interval, last_snap, last_fp):
    """Live activity feed for pipeline runs.

    Streams events emitted by every stage (including the background model
    subprocess). Visible on every route: this is the live view, and the
    Activity section is the filtered archive on top of it.
    """
    from services import progress_service as prog

    # Watchdog first: a run whose worker process died (observed: torch/MPS
    # native crash with no traceback) or that stopped emitting must become a
    # visible failure on this very tick, not an eternal spinner. Cheap when
    # idle (one diskcache read); emits + closes the run at most once.
    prog.watchdog_check()

    feed = prog.get_feed()
    events = feed.get("events") or []
    active = bool(feed.get("active"))
    poll = (_PROGRESS_POLL_ACTIVE_MS if active
            else _PROGRESS_POLL_IDLE_MS)
    # Writing interval on every tick restarts the timer and costs a DOM update
    # for no change; only send it when the rate actually changes.
    poll_out = poll if poll != _current_interval else dash.no_update
    # Newest run boundary. Written only when a NEW run appears, so the
    # clientside snap-to-newest fires once per run — a scroll offset parked
    # on the previous run's lines otherwise hides the new run entirely.
    snap = next((e.get("t") for e in reversed(events)
                 if e.get("stage") == "run"), None)
    snap_out = snap if (snap is not None and snap != last_snap) \
        else dash.no_update
    if not events:
        return (dash.no_update, dash.no_update, dash.no_update, poll_out,
                dash.no_update, dash.no_update)

    # What this tick would render: the visible window's newest edge and
    # size, the run boundary, and the active flag (drives the header icon).
    # If none of it moved, rewrite nothing — rebuilding the ~45 row nodes
    # every tick destroyed and recreated them all, which reset the feed's
    # scroll position on every poll, idle included. Count + newest
    # timestamp + boundary also cover the feed being swapped underneath us
    # (cold-start rehydrate, another publisher process).
    newest = events[-1]
    fp = [len(events), newest.get("t"), newest.get("stage"),
          newest.get("message"), snap, active]
    if fp == last_fp:
        return (dash.no_update, dash.no_update, dash.no_update, poll_out,
                snap_out, dash.no_update)

    # No auto-hide: the panel is an audit log now, so it stays up until the
    # user closes it. (It previously vanished 5 minutes after a run finished.)
    rows = []
    for e in events[-45:]:
        stage = e.get("stage", "")
        icon = _STAGE_ICONS.get(stage, "bi-dot")
        cls = "progress-line-error" if stage == "error" else (
            "progress-line-done" if stage == "done" else "")
        # Rendered from the event's epoch, not its pre-formatted string: rows
        # written by an older process carried a naive local clock while rows
        # restored from Postgres carried UTC, so the same panel showed both.
        rows.append(html.Div(
            [
                html.Span(prog.event_clock(e), className="progress-ts",
                          title=f"{prog.DISPLAY_TZ_LABEL} "
                                f"({prog.DISPLAY_TZ.key})"),
                html.I(className=f"bi {icon} progress-icon progress-icon-{stage}"),
                html.Span(e.get("message", ""), className="progress-msg"),
            ],
            className=f"progress-line {cls}",
        ))
    # No `key` props: dash-renderer keys child wrappers by tree path, not by
    # the component's key, so re-rendered rows are recreated wholesale
    # whatever we stamp on them. assets/feed_scroll_anchor.js preserves the
    # reading position across those rewrites instead.

    header_icon = (html.Div(className="progress-spinner")
                   if active
                   else html.I(className="bi bi-check-circle-fill progress-header-done"))

    return (rows, f"{len(events)} events", header_icon, poll_out, snap_out,
            fp)


# Scroll the feed to the newest lines when a new run starts. Clientside:
# scroll offsets are a DOM concern no server response can touch. Two gates:
# the server writes the token only when a new run boundary appears, and the
# browser remembers the last token it acted on — so even a rewrite of the
# same token can never re-snap, and the inert sink output means nothing can
# clear the store and re-arm the loop (outputting the store's own
# clear_data did exactly that: token wiped → every tick saw "new" → the
# user's scroll-away was reverted on each poll).
clientside_callback(
    """
    function(token) {
        var ns = window._quantnewsSnap = window._quantnewsSnap || {};
        if (token == null || token === ns.last) {
            return "";
        }
        ns.last = token;
        var feed = document.getElementById("progress-feed-scroll");
        if (feed && feed.lastElementChild) {
            feed.lastElementChild.scrollIntoView({block: "nearest"});
        }
        return "";
    }
    """,
    Output("progress-snap-sink", "children"),
    Input("progress-snap-store", "data"),
    prevent_initial_call=True,
)


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
# TRACE PAGE
# =============================================================================


@callback(
    Output("trace-data-wrap", "children"),
    Output("trace-models-wrap", "children"),
    Output("trace-llm-wrap", "children"),
    Output("trace-interval", "interval"),
    Output("trace-run-select", "options"),
    Output("trace-watermark-store", "data"),
    # Page-local components: the callback only dispatches on /trace.
    Input("trace-interval", "n_intervals", allow_optional=True),
    Input("trace-run-select", "value", allow_optional=True),
    State("trace-watermark-store", "data", allow_optional=True),
    prevent_initial_call=True,
)
def render_trace_view(_n, run_id, wm):
    """Keep the Trace page current for the selected run.

    Incremental on purpose: each tick asks only for the run's newest row ids
    (indexed max() per table) and rewrites a section ONLY when its watermark
    moved — the Data/Models groups follow activity_log, the LLM group follows
    llm_traces, so streaming events cannot wipe an expanded prompt body.
    Poll rate mirrors the progress panel: fast while the selected run is the
    live one, idle for historical runs.
    """
    from services import progress_service as prog
    from services import trace_service

    wm = dict(wm or {})
    marks = (trace_service.run_watermarks(run_id) if run_id
             else {"events": 0, "traces": 0})

    selection_changed = (ctx.triggered_id == "trace-run-select"
                         or wm.get("run_id") != run_id)
    events_moved = selection_changed or marks["events"] != wm.get("events")
    traces_moved = selection_changed or marks["traces"] != wm.get("traces")

    feed = prog.get_feed()
    active = bool(feed.get("active"))
    live_run = prog.current_run_id()
    poll = (trace_page.POLL_ACTIVE_MS if active and run_id == live_run
            else trace_page.POLL_IDLE_MS)
    poll_out = poll if poll != wm.get("poll") else dash.no_update

    # Refresh the selector's OPTIONS (never its value — an explicitly picked
    # historical run stays picked) when the run list actually changed: a new
    # run boundary landed, or the live flag flipped (a listed run's status
    # label just changed). Both probes are cheap; the 50-run listing query
    # runs only on those transitions.
    known = wm.get("runs") or []
    run_marker = trace_service.latest_run_marker()
    options_out = dash.no_update
    if (not known or run_marker != wm.get("run_marker")
            or active != wm.get("active")):
        runs = trace_page.list_trace_runs()
        options_out = trace_page.run_options(runs)
        known = [r["run_id"] for r in runs]

    if not (events_moved or traces_moved) and poll_out is dash.no_update \
            and options_out is dash.no_update:
        raise PreventUpdate

    return (
        trace_page.build_data_section(run_id) if events_moved else dash.no_update,
        trace_page.build_models_section(run_id) if events_moved else dash.no_update,
        trace_page.build_llm_section(run_id) if traces_moved else dash.no_update,
        poll_out,
        options_out,
        {"run_id": run_id, "events": marks["events"], "traces": marks["traces"],
         "poll": poll, "active": active, "run_marker": run_marker,
         "runs": known},
    )


@callback(
    Output({"type": "trace-llm-body", "id": MATCH}, "children"),
    Output({"type": "trace-llm-body", "id": MATCH}, "style"),
    Input({"type": "trace-llm-expand", "id": MATCH}, "n_clicks"),
    prevent_initial_call=True,
)
def toggle_trace_llm_body(n_clicks):
    """Fetch one call's prompt/response bodies on expand (and only then) —
    the list query never selects the Text columns."""
    if not n_clicks:
        raise PreventUpdate
    if n_clicks % 2 == 0:
        return dash.no_update, {"display": "none"}
    from services import trace_service
    bodies = trace_service.get_llm_call_bodies(ctx.triggered_id["id"])
    return trace_page.render_llm_bodies(bodies), {"display": "block"}


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

    from models.single_agent import extract_confidence, render_report_markdown
    from layouts.formatters import conviction_label, weight_label

    # The bare "(50%)" this used to print was the reliability weight, sitting
    # unlabeled beside the report body's own conviction line. Both numbers are
    # named now so a reader can tell which is which.
    stated = extract_confidence(report.get("report_text") or "")
    title = (f"{report.get('symbol', '?')} — {report.get('decision', '?')} · "
             f"{conviction_label(stated)} · {weight_label(report.get('confidence'))}"
             f" — {report.get('trade_date', '')}")
    from services import progress_service as prog
    prog.emit("action", f"Report viewed: {report.get('symbol', '?')} "
                        f"{report.get('trade_date', '')}")
    body = [
        html.Div(
            [
                # Was raw UTC with microseconds ("…14:37:42.473370+00:00"),
                # which also showed tomorrow's date all evening.
                html.Span(f"Generated: {prog.format_stamp(report.get('created_at'))}",
                          className="ta-accordion-meta"),
                html.Span(f" | model: {report.get('model_name', '?')}",
                          className="ta-accordion-meta"),
            ],
            style={"marginBottom": "8px"},
        ),
        dcc.Markdown(
            # Machine-read epilogue is stripped for reading, and the verdict's
            # field lines become list items so they render as discrete labelled
            # lines instead of one folded paragraph. The report keeps its own
            # "Compiled by … Sources: …" footer for transparency.
            render_report_markdown(report.get("report_text", "")),
            className="ta-report-body",
            style={"fontSize": "0.85rem", "lineHeight": "1.6"},
        ),
    ]
    return True, title, body


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
    Input("ensemble-config-drawer", "is_open"),
    State("ensemble-config-store", "data"),
    prevent_initial_call=True,
)
def sync_ensemble_config(switches, slider_values, input_values, reset_clicks,
                         drawer_open, current_store):
    """Single callback to sync ensemble config from UI controls to store.

    Handles switches, sliders, number inputs, and reset button. Returns
    store data + updated UI state (disabled flags, synced values). Method
    and min_agree are edited in the Run dialog, not here — carry them
    through unchanged so a drawer edit doesn't silently reset them.
    """
    from config import MODEL

    model_order = ["kronos_mini", "xgboost_shap", "lightgbm", "deberta_sentiment", "trading_agents"]
    triggered = ctx.triggered_id
    current_store = current_store or {}

    # Drawer opened: the store may have been edited from the Run dialog while
    # the drawer's controls sat stale. Repaint them from the store.
    if triggered == "ensemble-config-drawer":
        if not drawer_open:
            raise PreventUpdate
        enabled = set(current_store.get("enabled_models")
                      or MODEL.ENSEMBLE_DEFAULT_ENABLED)
        weights = current_store.get("weights") or dict(MODEL.ENSEMBLE_DEFAULT_WEIGHTS)
        disabled = [m not in enabled for m in model_order]
        vals = [float(weights.get(m, 1.0)) for m in model_order]
        return (dash.no_update, disabled, disabled[:], vals, vals[:],
                [m in enabled for m in model_order])

    # Reset to defaults
    if triggered == "ensemble-reset-btn":
        default_enabled = set(MODEL.ENSEMBLE_DEFAULT_ENABLED)
        default_weights = dict(MODEL.ENSEMBLE_DEFAULT_WEIGHTS)
        store = {
            "enabled_models": list(default_enabled),
            "weights": default_weights,
            "method": MODEL.ENSEMBLE_DEFAULT_METHOD,
            "min_agree": MODEL.ENSEMBLE_MIN_AGREE,
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

    store = {
        "enabled_models": enabled,
        "weights": weights,
        "method": current_store.get("method", MODEL.ENSEMBLE_DEFAULT_METHOD),
        "min_agree": current_store.get("min_agree", MODEL.ENSEMBLE_MIN_AGREE),
    }
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
        # _meta / a failure marker are not per-symbol result dicts
        if not isinstance(model_results, dict):
            continue
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


@callback(
    Output({"type": "run-ens-member", "model": ALL}, "value"),
    Output({"type": "run-ens-weight", "model": ALL}, "value"),
    Output("run-ensemble-method", "value"),
    Output("run-ensemble-min-agree", "value"),
    Input("run-modal", "is_open"),
    State("ensemble-config-store", "data"),
    prevent_initial_call=True,
)
def populate_run_ensemble(is_open, store):
    """Paint the Run dialog's ensemble controls from the shared store.

    The store is the single source of truth (also edited from the signal-card
    drawer), so the dialog must reflect it on open rather than whatever its
    controls held last time.
    """
    if not is_open:
        raise PreventUpdate
    from config import MODEL

    store = store or {}
    enabled = set(store.get("enabled_models") or MODEL.ENSEMBLE_DEFAULT_ENABLED)
    weights = store.get("weights") or dict(MODEL.ENSEMBLE_DEFAULT_WEIGHTS)
    member_ids = [o["id"]["model"] for o in ctx.outputs_list[0]]
    weight_ids = [o["id"]["model"] for o in ctx.outputs_list[1]]
    return (
        [m in enabled for m in member_ids],
        [float(weights.get(m, 1.0)) for m in weight_ids],
        store.get("method") or MODEL.ENSEMBLE_DEFAULT_METHOD,
        int(store.get("min_agree") or MODEL.ENSEMBLE_MIN_AGREE),
    )


@callback(
    Output("ensemble-config-store", "data", allow_duplicate=True),
    Input({"type": "run-ens-member", "model": ALL}, "value"),
    Input({"type": "run-ens-weight", "model": ALL}, "value"),
    Input("run-ensemble-method", "value"),
    Input("run-ensemble-min-agree", "value"),
    State("ensemble-config-store", "data"),
    prevent_initial_call=True,
)
def sync_run_ensemble(members, weight_vals, method, min_agree, current):
    """Run-dialog ensemble controls → shared store.

    Writing the store (rather than reading these controls at run time) keeps
    one config for the run itself, the drawer, and the live recompute of
    already-displayed signals.
    """
    from config import MODEL

    member_ids = [i["id"]["model"] for i in ctx.inputs_list[0]]
    weight_ids = [i["id"]["model"] for i in ctx.inputs_list[1]]
    enabled = [m for m, on in zip(member_ids, members or []) if on]
    weights = {}
    for m, v in zip(weight_ids, weight_vals or []):
        try:
            weights[m] = round(float(v), 1)
        except (TypeError, ValueError):
            weights[m] = 1.0
    try:
        min_agree = int(min_agree)
    except (TypeError, ValueError):
        min_agree = MODEL.ENSEMBLE_MIN_AGREE

    store = {
        "enabled_models": enabled,
        "weights": weights,
        "method": method or MODEL.ENSEMBLE_DEFAULT_METHOD,
        "min_agree": min_agree,
    }
    # The open-dialog repaint triggers this callback with the store's own
    # values; writing them back would recompute the ensemble for nothing.
    if store == (current or {}):
        raise PreventUpdate
    return store


@callback(
    Output("run-ensemble-method-hint", "children"),
    Output("run-ensemble-min-agree-wrap", "style"),
    Output({"type": "run-ens-weight", "model": ALL}, "disabled"),
    Input("run-ensemble-method", "value"),
    Input("run-ensemble-min-agree", "value"),
)
def run_ensemble_method_ui(method, min_agree):
    """Explain the chosen method and hide the controls it ignores.

    The explanation is a formula plus one worked example rather than a
    sentence: the methods differ only in how DIRECTION is decided, and the
    same four members resolve to BUY under the vote methods and HOLD under
    the other two. That divergence is the thing worth choosing between, and
    prose does not show it.
    """
    from config import MODEL
    from layouts.modals import ENSEMBLE_METHODS, ensemble_method_detail

    method = method or MODEL.ENSEMBLE_DEFAULT_METHOD
    hints = {val: hint for val, _, hint in ENSEMBLE_METHODS}
    n_weights = len(ctx.outputs_list[2])
    is_agreement = method == "agreement"
    try:
        gate = int(min_agree)
    except (TypeError, ValueError):
        gate = MODEL.ENSEMBLE_MIN_AGREE
    body = html.Div([
        html.Div(hints.get(method, ""), className="run-field-hint"),
        ensemble_method_detail(method, gate),
    ])
    return (
        body,
        {} if is_agreement else {"display": "none"},
        [is_agreement] * n_weights,
    )


@callback(
    Output("run-ensemble-body", "style"),
    Output("run-ensemble-summary", "children"),
    Input("run-ensemble-check", "value"),
    Input({"type": "run-model-check", "model": ALL}, "value"),
    Input({"type": "run-ens-member", "model": ALL}, "value"),
    Input("run-ensemble-method", "value"),
    Input("run-ensemble-min-agree", "value"),
)
def run_ensemble_effective(ens_on, run_checks, members, method, min_agree):
    """Say what the ensemble will actually combine — before the run.

    Membership only counts for models that are also checked to RUN, and the
    ensemble abstains below 2 valid members. Both used to be discovered
    after a run as an 'insufficient enabled models' row; surface them here.
    """
    from config import MODEL

    if not ens_on:
        return {"display": "none"}, None

    run_ids = [i["id"]["model"] for i in ctx.inputs_list[1]]
    member_ids = [i["id"]["model"] for i in ctx.inputs_list[2]]
    running = {m for m, on in zip(run_ids, run_checks or []) if on}
    chosen = {m for m, on in zip(member_ids, members or []) if on}
    effective = [m for m in member_ids if m in chosen and m in running]
    skipped = sorted(chosen - running)

    def warn(text):
        return html.Div([html.I(className="bi bi-exclamation-triangle me-1"),
                         text],
                        style={"fontSize": "0.85rem",
                               "color": "var(--warning, #ffc107)"})

    if len(effective) < 2:
        return {}, warn(
            f"Ensemble will abstain: it needs at least 2 member models that "
            f"are also checked to run (currently {len(effective)})."
        )

    try:
        min_agree = int(min_agree)
    except (TypeError, ValueError):
        min_agree = MODEL.ENSEMBLE_MIN_AGREE
    if method == "agreement" and min_agree > len(effective):
        return {}, warn(
            f"Consensus gate can never open: it requires {min_agree} "
            f"agreeing models but only {len(effective)} members are running."
        )

    names = ", ".join(MODEL_DISPLAY.get(m, m) for m in effective)
    note = (f" — {', '.join(MODEL_DISPLAY.get(m, m) for m in skipped)} "
            f"not running, so not counted" if skipped else "")
    return {}, html.Div(
        f"Combines {len(effective)} models: {names}{note}",
        style={"fontSize": "0.85rem", "color": "var(--text-muted)"},
    )


# =============================================================================
# SCHEDULER PANEL (History tab)
# =============================================================================


@callback(
    Output("scheduler-panel-container", "children"),
    # Both live on the Schedule page, so neither exists on other routes.
    Output("scheduler-fp", "data"),
    Input("scheduler-refresh", "n_intervals", allow_optional=True),
    Input("scheduler-action-status", "data", allow_optional=True),
    State("sched-new-name", "value", allow_optional=True),
    State("sched-new-symbols", "value", allow_optional=True),
    State("scheduler-fp", "data", allow_optional=True),
)
def render_scheduler_panel(_n, _action, draft_name, draft_symbols, last_fp):
    """Redraw the schedule panel from the database.

    Reads on a timer rather than caching in a Store: the job state is written
    by the scheduler thread (and possibly by another instance), so the browser
    is never the source of truth for it.

    The timer must not clobber a form someone is filling in: a re-render
    resets every input, so half-typed jobs used to vanish on the next tick.
    Draft content in the create form defers the redraw; an explicit action
    (create/save/delete, via scheduler-action-status) always redraws.
    """
    from layouts.scheduler_components import build_scheduler_panel
    from services import scheduler_service

    timer_fired = (ctx.triggered_id == "scheduler-refresh")
    if timer_fired and ((draft_name or "").strip()
                        or (draft_symbols or "").strip()):
        raise PreventUpdate

    try:
        jobs = scheduler_service.list_jobs()
        runs = scheduler_service.recent_runs(limit=8)
        # A timer tick redraws only when the database changed. Redrawing
        # every 15s reset every job card's inputs, so an edit typed into an
        # existing card was silently replaced by the stored values before
        # Save read them — and Save then confirmed a change that never
        # happened.
        fp = hashlib.md5(json.dumps([jobs, runs], sort_keys=True,
                                    default=str).encode()).hexdigest()
        if timer_fired and fp == last_fp:
            raise PreventUpdate
        return build_scheduler_panel(
            jobs, runs, scheduler_service.list_job_types()), fp
    except PreventUpdate:
        raise
    except Exception as e:
        logger.warning(f"Scheduler panel render failed: {e}")
        return html.Div(f"Scheduler unavailable: {str(e)[:160]}",
                        className="scheduler-empty"), last_fp


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
    State({"type": "sched-visibility", "job": MATCH}, "value"),
    State({"type": "sched-param", "job": MATCH, "key": ALL}, "value"),
    State({"type": "sched-param", "job": MATCH, "key": ALL}, "id"),
    prevent_initial_call=True,
)
def save_schedule(n_clicks, enabled, hour, minute, days, tz, symbols,
                  visibility, param_values, param_ids):
    """Persist one job's schedule (and its tuning params) and reschedule it."""
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
        "is_public": visibility != "private",
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
    # Tuning params were create-only: the card never showed them and this
    # save never wrote them, so a window chosen at creation was frozen.
    if param_ids:
        current = next((j.get("params") or {} for j in scheduler_service.list_jobs()
                        if j["id"] == job_id), {})
        params = dict(current)
        for pid, val in zip(param_ids, param_values or []):
            if val is None or val == "":
                continue
            try:
                params[pid["key"]] = (int(val) if float(val).is_integer()
                                      else float(val))
            except (TypeError, ValueError):
                return html.Span(f"{pid['key']} must be a number",
                                 className="scheduler-feedback-error")
        fields["params_json"] = params
    if not scheduler_service.update_job(job_id, **fields):
        return html.Span("Save failed", className="scheduler-feedback-error")

    state = "enabled" if enabled else "disabled"
    return html.Span(f"Saved — {state}, {hour:02d}:{minute:02d} {tz}",
                     className="scheduler-feedback-ok")


@callback(
    Output({"type": "sched-new-params-group", "kind": ALL}, "style"),
    Output("sched-new-hour", "value"),
    Output("sched-new-minute", "value"),
    Input("sched-new-kind", "value", allow_optional=True),
    State({"type": "sched-new-params-group", "kind": ALL}, "id"),
    prevent_initial_call=True,
)
def show_kind_params(kind, group_ids):
    """Reveal only the selected operation's tuning knobs, and start the time
    at the type's own default (07:00 predict, 18:00 evaluate, …) — the
    declared defaults were exported and then ignored by a hardcoded 08:30."""
    from services import scheduler_service
    jt = next((t for t in scheduler_service.list_job_types() if t["kind"] == kind), None)
    hour = jt["default_hour"] if jt else dash.no_update
    minute = jt["default_minute"] if jt else dash.no_update
    return ([{"display": "flex" if g.get("kind") == kind else "none"}
             for g in (group_ids or [])], hour, minute)


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
    State("sched-new-visibility", "value"),
    State({"type": "sched-new-param", "kind": ALL, "key": ALL}, "value"),
    State({"type": "sched-new-param", "kind": ALL, "key": ALL}, "id"),
    prevent_initial_call=True,
)
def create_scheduled_job(n_clicks, kind, name, hour, minute, days, tz, symbols,
                         visibility, param_values, param_ids):
    """Add a job of any registered operation type."""
    if not n_clicks:
        raise PreventUpdate
    # (param_values/param_ids are the per-kind tuning inputs; filtered to the
    # selected kind below, so knobs for unselected operations are ignored.)

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

    # Per-kind tuning knobs: only the selected kind's inputs apply.
    # only_trading_days is read by the symbol-taking (analyze) jobs only.
    params = {"only_trading_days": True} if needs.get(kind) else {}
    for pid, val in zip(param_ids or [], param_values or []):
        if pid.get("kind") == kind and val is not None:
            params[pid["key"]] = val

    job_id = scheduler_service.create_job(
        kind=kind, description=(name or "").strip(), hour=hour, minute=minute,
        days_of_week=days, timezone=tz, symbols_csv=cleaned or None,
        params=params,
        is_public=visibility != "private",
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
    Output("run-tools", "value"),
    Input("run-date-picker", "date"),
    Input("run-modal", "is_open"),
    prevent_initial_call=True,
)
def default_run_tools(ai_date, is_open):
    """Web research is on for a next-day (forward-testing) run and off for a
    backtest date, where the open web would leak the future. The user can
    still flip the box after the date is chosen; this only sets the default
    each time the date changes or the dialog opens."""
    if not is_open:
        raise PreventUpdate
    from services.investigation_service import default_tools
    from utils.trading_calendar import resolve_target_and_cutoff
    picked = str(ai_date)[:10] if ai_date else None
    target_d, _ = resolve_target_and_cutoff(picked)
    return default_tools(target_d)


@callback(
    Output("run-article-preview", "children"),
    Input("run-modal", "is_open"),
    Input("run-lookback", "value"),
    Input("run-max-articles", "value"),
    Input("run-date-picker", "date"),
    # Input, not State: editing the run's symbol set refreshes the counts.
    Input("run-symbols-store", "data"),
    prevent_initial_call=True,
)
async def preview_ai_report_articles(is_open, lookback, max_articles_val,
                                     ai_date, run_symbols):
    """Fetch and show article availability for the chosen window BEFORE
    generation — the same point-in-time fetch generation uses, so the
    preview counts are the counts, not an estimate from the news store.

    Async: fires on modal open, so the live vendor fetch runs in a worker
    thread instead of freezing the UI thread pool."""
    symbols = (run_symbols or {}).get("symbols") or []
    if not is_open:
        raise PreventUpdate
    if not symbols:
        # Removing the last chip must clear the counts too — PreventUpdate
        # here left the previous symbol's table on screen.
        return html.Div("No symbols in this run — add one to preview "
                        "article availability.",
                        className="run-symbols-empty")

    from config import MODEL as _M
    from utils.trading_calendar import resolve_target_and_cutoff
    from services.news_window import (
        RunParameterMissing, fetch_run_news, normalize_article_cap,
        normalize_lookback)

    # Same target/cutoff resolution as generation: the window ends at the
    # cutoff (previous trading day), not the target — previewing a window
    # ending at the target overstated what the report would actually see.
    picked = str(ai_date)[:10] if ai_date else None
    target_d, as_of_d = resolve_target_and_cutoff(picked)
    as_of, target = as_of_d.isoformat(), target_d.isoformat()
    try:
        overnight, lookback_days = normalize_lookback(lookback)
        max_articles = normalize_article_cap(max_articles_val)
    except RunParameterMissing as e:
        return html.Div(f"Cannot preview — {e}", className="text-danger")

    sem = asyncio.Semaphore(APP.NEWS_FETCH_CONCURRENCY)

    async def _one(sym):
        # No retries in a preview: a throttled vendor shows as "unavailable"
        # now rather than freezing the dialog for the backoff.
        async with sem:
            _, stats = await asyncio.to_thread(
                fetch_run_news, [sym], as_of, target, overnight=overnight,
                lookback_days=lookback_days, max_articles=max_articles,
                retries=1)
            return stats[sym]

    stats = await asyncio.gather(*(_one(s) for s in symbols))

    rows, total, capped = [], 0, []
    for sym, st in zip(symbols, stats):
        n = int(st.get("kept") or 0)
        total += n
        if st.get("capped"):
            capped.append(sym)
        span = (f"{st['oldest']} → {st['newest']}"
                if st.get("oldest") and st.get("newest") else "—")
        note = ""
        if st.get("status") == "unavailable":
            note = "source unavailable"
        elif st.get("capped"):
            note = (f"capped {st['fetched']}→{n}, effective "
                    f"{st.get('effective_days')}d")
        rows.append(html.Tr([
            html.Td(sym),
            html.Td(str(n), style={"textAlign": "right"}),
            html.Td(span, style={"color": "var(--text-secondary)"}),
            html.Td(note, style={"color": ("var(--warning, #ffc107)"
                                           if note else "inherit")}),
        ]))

    window_label = (
        f" in the overnight window {as_of} {_M.NEWS_OVERNIGHT_START_ET} ET → "
        f"{target} {_M.NEWS_OVERNIGHT_END_ET} ET (relevance ≥ {_M.NEWS_OVERNIGHT_RELEVANCE})"
        if overnight else
        f" in the {lookback_days}-day window ending {as_of} (data cutoff for target {target})"
    ) + (f", newest {max_articles} per symbol" if max_articles else ", no cap")
    warn = None
    if capped:
        warn = html.Div(
            f"Cap drops the oldest articles for {', '.join(capped)} — raise "
            f"the cap (0 = all) to keep the whole window.",
            style={"color": "var(--warning, #ffc107)"}, className="mb-1")
    return html.Div([
        html.Div([
            html.Strong(f"{total} articles"),
            html.Span(window_label,
                      style={"color": "var(--text-secondary)"}),
        ], className="mb-1"),
        warn,
        dbc.Table(
            [html.Thead(html.Tr([html.Th("Symbol"),
                                 html.Th("Articles", style={"textAlign": "right"}),
                                 html.Th("Span"), html.Th("")])),
             html.Tbody(rows)],
            bordered=False, color="dark", size="sm", className="mb-0",
        ),
    ])


@callback(
    Output("run-date-mode-label", "children"),
    # Every return path below is (children, style); with only the children
    # Output declared, Dash rendered the style dict as part of the label.
    Output("run-date-mode-label", "style"),
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


def _run_data_summary(symbols, stock_data, news_data):
    """The dialog's per-symbol input summary for the resolved run set.

    A symbol outside the browser stores shows "at run time" — its data is
    fetched server-side when the run starts (see _fill_run_inputs), so an
    empty row here is a statement of when, not a problem.
    """
    if not symbols:
        return html.Div(
            [
                html.Div("No symbols selected", className="empty-state-title"),
                html.Div("Pick a source above or add a symbol to this run.",
                         className="empty-state-note"),
            ],
            className="empty-state",
        )
    stock_data = stock_data or {}
    articles_by_symbol = (news_data or {}).get("articles_by_symbol", {})
    sym_rows = []
    for sym in symbols:
        sym_data = stock_data.get(sym, {})
        data_points = "—"
        date_range = "—"
        source = "at run time"
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
                source = ("Cached" if sym_data.get("from_cache")
                          else "Live")
            except Exception:
                pass
        news_count = len(articles_by_symbol.get(sym, []))
        sym_rows.append(html.Tr([
            html.Td(sym), html.Td(data_points),
            html.Td(date_range),
            html.Td(str(news_count) if sym in articles_by_symbol else "—"),
            html.Td(source),
        ]))

    return html.Div([
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


def _run_source_options(watchlist, cohort_syms):
    return [
        {"label": f"Watchlist ({len(watchlist)})", "value": "watchlist",
         "disabled": not watchlist},
        {"label": f"Last run ({len(cohort_syms)})", "value": "cohort",
         "disabled": not cohort_syms},
        {"label": "Custom", "value": "custom"},
    ]


@callback(
    Output("run-modal", "is_open"),
    Output("run-data-summary", "children"),
    Output("run-scope", "value"),
    Output("run-symbols-store", "data"),
    Output("run-symbols-source", "options"),
    Output("run-symbols-source", "value"),
    Output("run-validation-msg", "children"),
    Output("progress-panel-state", "data", allow_duplicate=True),
    Output("run-started-toast", "is_open"),
    Output("run-started-toast", "children"),
    # Snap the panel's poll to the fast rate on confirm — waiting for a slow
    # idle tick to notice the active flag cost ~an idle interval before the
    # first run events rendered.
    Output("progress-interval", "interval", allow_duplicate=True),
    # The picker's default/max/holidays are baked into the layout at process
    # start; a long-lived server would otherwise pin the dialog to launch day.
    Output("run-date-picker", "date"),
    Output("run-date-picker", "max_date_allowed"),
    Output("run-date-picker", "disabled_days"),
    Input("run-analysis-btn", "n_clicks"),
    Input("reports-new-btn", "n_clicks", allow_optional=True),
    Input({"type": "new-report-btn", "symbol": ALL}, "n_clicks"),
    Input("run-cancel-btn", "n_clicks"),
    Input("run-confirm-btn", "n_clicks"),
    State("stock-data-store", "data"),
    State("selected-symbols", "data"),
    State("news-data-store", "data"),
    State("run-modal", "is_open"),
    State("run-symbols-store", "data"),
    State("run-scope", "value"),
    State("progress-panel-state", "data"),
    prevent_initial_call=True,
)
def toggle_run_modal(open_clicks, reports_clicks, ctx_clicks, cancel_clicks,
                     confirm_clicks, stock_data, watchlist,
                     news_data, is_open, run_store, run_scope, panel_state):
    """Open or close the single run dialog, preset by where it was opened.

    Two kinds of opener, deliberately: the toolbar button (the ONE global
    entry point — it starts from the watchlist and whatever scope was last
    chosen) and contextual shortcuts that carry their context in — the
    Reports page's New Report (scope=report, watchlist) and any per-symbol
    New-report button (scope=report, just that symbol). The old Home header
    duplicate, which carried no context at all, is gone.

    Confirm also owns the immediate acknowledgement: force-open the activity
    panel (a user who once closed it otherwise gets zero feedback) and toast
    where the results will land. An empty symbol set keeps the dialog open
    with an inline message instead of silently no-opping — every downstream
    stage guards on the list being non-empty.
    """
    triggered = ctx.triggered_id
    no_sym_update = (dash.no_update,) * 3
    # panel state + toast open/children + poll rate
    no_feedback = (dash.no_update,) * 4
    # date-picker date + max_date_allowed + disabled_days
    no_picker = (dash.no_update,) * 3

    if triggered == "run-cancel-btn":
        return (False, dash.no_update, dash.no_update) + no_sym_update \
            + ("",) + no_feedback + no_picker

    if triggered == "run-confirm-btn":
        symbols = (run_store or {}).get("symbols") or []
        if not symbols:
            return (dash.no_update,) * 6 \
                + ("Add at least one symbol to run.",) + no_feedback \
                + no_picker
        panel = dict(panel_state or {})
        panel.setdefault("mode", "normal")
        panel["closed"] = False
        # Mark the feed active NOW: the stage that calls start_run is a
        # separate callback, and the fast tick below must not see an idle
        # feed and drop straight back to the slow rate.
        from services import progress_service as prog
        prog.mark_run_pending()
        scope = run_scope or "full"
        toast_msg = {
            "models": "Run started — predictions will appear on Home "
                      "and Analyze.",
            "report": "Report started — it will appear under Reports.",
        }.get(scope, "Run started — predictions will appear on Home and "
                     "Analyze, the report under Reports.")
        return (False, dash.no_update, dash.no_update) + no_sym_update \
            + ("", panel, True, toast_msg, _PROGRESS_POLL_ACTIVE_MS) \
            + no_picker

    is_context_btn = (isinstance(triggered, dict)
                      and triggered.get("type") == "new-report-btn")
    if triggered not in ("run-analysis-btn", "reports-new-btn") \
            and not is_context_btn:
        raise PreventUpdate

    # All openers fire this callback when they mount with n_clicks still
    # None. Only the control that actually triggered may open the dialog.
    if is_context_btn:
        if not (ctx.triggered and ctx.triggered[0].get("value")):
            raise PreventUpdate
    elif not {"run-analysis-btn": open_clicks,
              "reports-new-btn": reports_clicks}.get(triggered):
        raise PreventUpdate

    watchlist = [s for s in (watchlist or []) if s]
    cohort_syms = []
    try:
        from services import dashboard_service as ds
        cohort = ds.get_latest_cohort() or {}
        cohort_syms = [r["symbol"] for r in cohort.get("symbols") or []]
    except Exception as e:
        logger.warning("Run dialog: could not read latest cohort: %s", e)

    scope = dash.no_update
    if is_context_btn:
        scope = "report"
        source = "custom"
        run_symbols = [triggered.get("symbol")]
    elif triggered == "reports-new-btn":
        scope = "report"
        source = "watchlist" if watchlist else "cohort"
        run_symbols = watchlist or cohort_syms
    else:
        source = "watchlist" if watchlist else "cohort"
        run_symbols = watchlist or cohort_syms

    store = {
        "source": source,
        "symbols": run_symbols,
        "watchlist": watchlist,
        "cohort": cohort_syms,
    }
    picker = no_picker
    try:
        from utils.trading_calendar import (get_default_target_day,
                                            non_trading_days)
        target = get_default_target_day()
        picker = (target.isoformat(), target.isoformat(),
                  [d.isoformat()
                   for d in non_trading_days("2020-01-01", target)])
    except Exception as e:
        logger.warning("Run dialog: could not refresh target session: %s", e)
    return (
        True,
        _run_data_summary(run_symbols, stock_data, news_data),
        scope,
        store,
        _run_source_options(watchlist, cohort_syms),
        source,
        "",
    ) + no_feedback + picker


@callback(
    Output("model-info-modal", "is_open"),
    Output("model-info-title", "children"),
    Output("model-info-body", "children"),
    Input({"type": "run-model-info", "model": ALL}, "n_clicks"),
    Input("model-info-close-btn", "n_clicks"),
    State("run-evidence", "value"),
    State("run-type", "value"),
    State("run-tools", "value"),
    prevent_initial_call=True,
)
def toggle_model_info(info_clicks, close_click, run_evidence, depth, run_tools):
    """Model explainer for researchers, opened from the ⓘ icons in the Run
    dialog. TradingAgents' body embeds the verbatim research prompt with the
    context-block list reflecting the Evidence checkboxes AS SELECTED NOW —
    reopen after changing them to see the updated preview."""
    trig = ctx.triggered_id
    if trig == "model-info-close-btn":
        return False, dash.no_update, dash.no_update
    # Pattern inputs fire once with n_clicks=None when the modal mounts.
    if not isinstance(trig, dict) or not any(c for c in (info_clicks or []) if c):
        raise PreventUpdate
    from layouts.model_info import MODEL_EXPLAINERS, build_model_info_body
    model_id = trig.get("model", "")
    title = MODEL_EXPLAINERS.get(model_id, {}).get("title", model_id)
    body = build_model_info_body(
        model_id,
        evidence=run_evidence if run_evidence is not None
                 else list(MODEL.DEFAULT_EVIDENCE),
        include_thesis=(depth or "thesis") != "standard",
        tools=run_tools or [],
    )
    return True, title, body


@callback(
    Output("run-symbols-store", "data", allow_duplicate=True),
    Output("run-symbol-add", "value"),
    Input("run-symbols-source", "value"),
    Input({"type": "run-sym-remove", "symbol": ALL}, "n_clicks"),
    Input("run-symbol-add", "n_submit"),
    State("run-symbol-add", "value"),
    State("run-symbols-store", "data"),
    prevent_initial_call=True,
)
def set_run_symbols(source, remove_clicks, add_submit, add_value, store):
    """Edit the run's symbol set without touching the watchlist.

    Switching source swaps in that set wholesale; removing a chip or adding
    a symbol turns the set custom (the radio follows via its options, the
    label stays honest). The store's watchlist/cohort snapshots were taken
    when the dialog opened, which is the set the user was looking at.
    """
    store = dict(store or {})
    symbols = list(store.get("symbols") or [])
    triggered = ctx.triggered_id

    if triggered == "run-symbols-source":
        if not source or source == store.get("source"):
            raise PreventUpdate  # echo of the open-time preset
        if source in ("watchlist", "cohort"):
            store["symbols"] = list(store.get(source) or [])
        store["source"] = source
        return store, dash.no_update

    if isinstance(triggered, dict) and triggered.get("type") == "run-sym-remove":
        if not any(c and c > 0 for c in remove_clicks):
            raise PreventUpdate
        sym = triggered.get("symbol")
        store["symbols"] = [s for s in symbols if s != sym]
        store["source"] = "custom"
        return store, dash.no_update

    if triggered == "run-symbol-add":
        if not add_submit or not add_value:
            raise PreventUpdate
        new_symbols = [s.strip().upper()
                       for s in re.split(r"[,\s]+", add_value) if s.strip()]
        store["symbols"] = symbols + [s for s in new_symbols
                                      if s and s not in symbols]
        store["source"] = "custom"
        return store, ""

    raise PreventUpdate


@callback(
    Output("run-symbols-chips", "children"),
    Output("run-symbols-source", "value", allow_duplicate=True),
    Input("run-symbols-store", "data"),
    prevent_initial_call=True,
)
def render_run_symbol_chips(store):
    """Chips for the effective run set, each removable for this run only."""
    store = store or {}
    symbols = store.get("symbols") or []
    if not symbols:
        chips = [html.Span("No symbols — add one below or pick a source.",
                           className="run-symbols-empty")]
    else:
        chips = [
            html.Span(
                [
                    html.Span(sym, className="run-symchip-text"),
                    html.Button(
                        "✕",
                        id={"type": "run-sym-remove", "symbol": sym},
                        className="run-symchip-remove",
                        title=f"Drop {sym} from this run",
                    ),
                ],
                className="run-symchip",
            )
            for sym in symbols
        ]
    return chips, store.get("source") or "custom"


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
    Output("run-preflight", "children"),
    Input("run-modal", "is_open"),
    Input("run-scope", "value"),
    Input("run-symbols-store", "data"),
    Input({"type": "run-model-check", "model": ALL}, "value"),
    Input("run-model", "value"),
    Input("run-recs", "value"),
    Input("run-recs-model", "value"),
)
def run_preflight(is_open, scope, run_symbols, _model_checks, report_model,
                  recs_basis, recs_model):
    """Name what this run needs and cannot reach, before it is started.

    A missing API key used to surface only as a stage failure minutes in, and
    the dialog gave no idea how long a run would take.
    """
    from layouts.modals import run_preflight_children

    if not is_open:
        return dash.no_update
    # Read the ids alongside the values instead of zipping against a
    # hand-maintained order — a pattern-matching Input's ordering is Dash's to
    # decide, and a silent mis-zip here would name the wrong models.
    checks = next((entry for entry in ctx.inputs_list
                   if isinstance(entry, list) and entry
                   and entry[0].get("id", {}).get("type") == "run-model-check"),
                  [])
    selected = [c["id"]["model"] for c in checks if c.get("value")]
    return run_preflight_children(
        scope or "full",
        (run_symbols or {}).get("symbols") or [],
        selected,
        report_model or "",
        recs_basis or "auto",
        recs_model or "",
    )


@callback(
    Output("full-analysis-requested", "data"),
    Input("run-confirm-btn", "n_clicks"),
    State("run-scope", "value"),
    State("run-symbols-store", "data"),
    prevent_initial_call=True,
)
def set_full_analysis_flag(n_clicks, scope, run_symbols):
    """Arm the synthesis stage, but only for a full-pipeline run.

    An empty symbol set never arms: the run dialog rejects it inline and no
    stage will fire, so an armed flag would just lie in wait for the next
    unrelated store update.
    """
    if n_clicks and (scope or "full") == "full" \
            and ((run_symbols or {}).get("symbols") or []):
        return True
    raise PreventUpdate


# Shared with the scheduled runner (scripts/daily_analysis.py) so the UI and
# cron paths merge research identically.
from services.analysis_runner import (  # noqa: E402
    merge_research_into_analysis as _merge_research_into_analysis,
)


@callback(
    Output("recommendations-store", "data"),
    # Every terminal branch also disarms the flag — a stale armed flag made
    # the NEXT run's stores look like a full run still in flight.
    Output("full-analysis-requested", "data", allow_duplicate=True),
    Input("ai-analysis-store", "data"),
    Input("model-signals-store", "data"),
    State("full-analysis-requested", "data"),
    State("run-symbols-store", "data"),
    prevent_initial_call=True,
)
async def generate_recommendations_callback(ai_analysis, model_signals,
                                            requested, run_symbols):
    """Generate recommendations when their inputs are ready.

    Fires for the Full Analysis flow (requested flag) AND for AI Report runs
    that asked for recommendations (recs_request on the store, carrying the
    evidence basis: research+signals / news+signals / signals).

    A full run synthesizes exactly once, from BOTH of that click's stores:
    the report lands first, and synthesizing from it alone ran Luna without
    any model signals — and "finished" the run while the models were still
    working. run_seq (stamped on the report and the signals _meta by the
    same confirm click) pairs the two. Report-only payloads proceed without
    waiting, but only on the report's own landing — a later models-only
    run's signals must never resurrect a stale report's recs_request.

    Async: the ~1-minute Luna synthesis and the Postgres persist loop run in
    worker threads instead of pinning a callback slot.
    """
    basis = (ai_analysis or {}).get("recs_request")
    if not requested and not basis:
        raise PreventUpdate

    from services import progress_service as prog

    # A report-scope payload never gets model signals, so a lingering
    # requested flag must not make it wait for them.
    is_full_run = bool(requested) and (ai_analysis or {}).get("scope") != "report"
    if is_full_run:
        sig_meta = (model_signals or {}).get("_meta") or {}
        run_seq = (ai_analysis or {}).get("run_seq")
        if run_seq is None or run_seq != sig_meta.get("run_seq"):
            raise PreventUpdate
        if (model_signals or {}).get("_run_failed"):
            # The model stage crashed; its handler already closed the run.
            return {"error": "Model run failed — recommendations skipped",
                    "generated_at": datetime.now().isoformat()}, False
    elif ctx.triggered_id != "ai-analysis-store":
        # No full run armed for this click, so only the report's OWN landing
        # may synthesize. Fresh signals from a models-only run must not be
        # glued to whatever stale report (and its recs_request) still sits
        # in the session store — that ran Luna right after a models-only
        # "Predictions complete". No finish needed: persist_predictions
        # already closed that run.
        raise PreventUpdate

    if not ai_analysis:
        raise PreventUpdate
    if ai_analysis.get("failed"):
        if is_full_run:
            # The report is a synthesis input; without it this run cannot
            # produce recommendations. Close the run honestly.
            prog.finish_run("Full Analysis finished with errors — report "
                            "failed")
            return {"error": "AI report failed — recommendations skipped",
                    "generated_at": datetime.now().isoformat()}, False
        # Report-only failures are closed by the report callback itself.
        raise PreventUpdate

    # Recommendations explicitly turned off for this run: the Full Analysis
    # flow still sets the requested flag, so honor the store's opt-out marker
    # (and close the progress run once predictions have landed).
    if ai_analysis.get("recs_off"):
        if is_full_run:
            prog.finish_run("Full Analysis complete (recommendations off)")
            return dash.no_update, False
        raise PreventUpdate

    basis = basis or "news+signals"  # Full Analysis default
    symbols = (run_symbols or {}).get("symbols") or []

    valid_signals = {
        k: v for k, v in (model_signals or {}).items()
        if k not in ("_meta", "_run_failed") and isinstance(v, dict)
    }
    if not valid_signals:
        if basis == "signals":
            # Predictions-only synthesis with no predictions is a no-op —
            # say so instead of silently doing nothing.
            prog.emit("error", "Recommendations (predictions only) skipped — "
                               "no model predictions in this session. Run Predict first.")
            if is_full_run:
                prog.finish_run("Full Analysis finished with errors — "
                                "no model predictions")
                return dash.no_update, False
            prog.finish_run("Report complete — recommendations skipped "
                            "(no predictions this session)")
            raise PreventUpdate
        # Text-based bases can proceed without signals; Luna sees the gap.
        valid_signals = {}

    # As-of date travels with the AI report (set by the Full Analysis picker)
    trade_date = (ai_analysis.get("as_of") or datetime.now().strftime("%Y-%m-%d"))[:10]

    # ONE synthesis stage (services.analysis_runner.run_recommendations):
    # cache key, research merge, p_correct parsing, persistence and the
    # model-that-answered stamp all live there. The copy that lived here
    # stored NULL confidence whenever the model answered with p_correct
    # instead of a conviction label, and recorded the configured model as
    # the one used even after a dialog override or provider fallback.
    from services.analysis_runner import run_recommendations
    result = await asyncio.to_thread(
        run_recommendations, ai_analysis, model_signals, symbols or [], trade_date)

    if result and result.get("from_cache"):
        # A cache-served run is still a completed run — without the finish
        # the panel spinner spun forever on a success.
        prog.finish_run(
            "Full Analysis complete (recommendations from cache)"
            if is_full_run else "Report complete (recommendations from cache)")
        return result, False
    if result:
        prog.finish_run("Full Analysis complete" if is_full_run
                        else "Report complete")
        return result, False

    prog.finish_run(("Full Analysis" if is_full_run else "Report")
                    + " finished with errors")
    return {"error": "Recommendations model unavailable or returned empty response",
            "generated_at": datetime.now().isoformat()}, False


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

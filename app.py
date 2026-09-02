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
from datetime import datetime, timezone
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
    asyncio.to_thread work inherit it. Anonymous requests proceed normally,
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
        # ".cygnetsystems.us" in production, the shared-SSO cookie visible to
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
    actually matters, "did today's analysis happen". Rather than "is the web
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
    # Rendered dynamically inside auth-chip. Allow_optional so Dash 4's
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

    # Model predictions for this symbol. Prefer the report's trade date,
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
    content = f"# {symbol}: {decision} ({conf_pct}%): {trade_date}\n\n{report.get('report_text', '')}"
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
    No LLM is involved, downloads must never incur model cost.
    """
    as_of = date[:10]
    syms = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    if not syms or not as_of:
        raise HTTPException(status_code=400)
    if lookback is None:
        raise HTTPException(
            status_code=400,
            detail="lookback (the run's news window in days) is required. "
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
# OBJECT STORAGE (S3), gracefully optional
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

# _init_s3() runs in _startup() via the ASGI lifespan, not at import, so the
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
    the same cache layer the store fetch uses. Blocking: call it from the
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

    Works for ANY symbol, watchlist or last-run cohort, which is the
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

    Applies the SAME outcome slice the section header announces, the body
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
# (Analyze, Performance): see create_data_actions. Their callbacks take them
# as allow_optional Inputs; there is no disabled state to manage anymore.


@callback(
    Output("history-filter-symbols", "data", allow_duplicate=True),
    Output("history-filter-outcome", "data", allow_duplicate=True),
    Input({"type": "perf-symbol-drill", "symbol": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def drill_into_symbol(clicks):
    """Narrow the page to one symbol: its calls, dates and outcomes.

    Clears any outcome slice on the way in. Arriving at a symbol filtered to
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

    Writes the store only, the dropdown is seeded from it when the page is
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

    The symbol rows on the left ARE the filter control, clicking the active
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
    # cutoff override. Guarded to Home below, the stores fire everywhere
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
        #: the industry-standard multi-ticker search convention.
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

    # Handle remove buttons, the Home rail rows and the global strip chips
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
        return html.Span("empty: type a symbol and press Enter",
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
# intraday bars on the price chart only. Intraday NEVER enters the shared
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
    # Refresh lives on Analyze's action bar now. Absent on other routes.
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
    history: a bare 1-month fetch used to starve them), while metrics are
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
    # fires when navigation mounts it (n_clicks None). That must not force
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
        """Blocking per-symbol fetch, runs in a worker thread."""
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

    # Update cache status (quick Postgres read. Still off the event loop)
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
        # chart only. Daily-period SMA overlays are omitted, drawing a
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
                                f"{sessions} session{'s' if sessions > 1 else ''}: "
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
        # daily bars: a 2-point MACD line reads as noise, not momentum.
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

    Async: per-symbol fetches run concurrently in threads, capped LOW.
    Alpha Vantage's free tier rate-limits hard, and a parallel blast would
    trade a slow callback for a 429'd one.
    """
    if not symbols:
        return {}

    def _fetch_one(symbol):
        """Blocking per-symbol news fetch. Runs in a worker thread."""
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


# =============================================================================
# RUN RECORD: the analysis_runs row behind a confirm, and its dispatch
# =============================================================================

# The run dialog's model checklist, in the order the modal renders it.
_RUN_MODEL_IDS = [
    "kronos_mini", "xgboost_shap", "lightgbm",
    "deberta_sentiment", "trading_agents",
]


def _ensemble_config_of(config: dict) -> dict | None:
    """The ensemble the run's config recorded, in the shape the runner and
    the drawer share; None for a row from before members were recorded
    (the runner then applies its configured defaults)."""
    members = config.get("ensemble_members")
    if members is None:
        return None
    weights = {}
    for model_id in _RUN_MODEL_IDS:
        try:
            weights[model_id] = round(
                float((config.get("ensemble_weights") or {}).get(model_id, 1.0)),
                1)
        except (TypeError, ValueError):
            weights[model_id] = 1.0
    try:
        min_agree = int(config.get("ensemble_min_agree"))
    except (TypeError, ValueError):
        min_agree = MODEL.ENSEMBLE_MIN_AGREE
    return {
        "enabled_models": [m for m in members if m in _RUN_MODEL_IDS],
        "weights": weights,
        "method": config.get("ensemble_method") or MODEL.ENSEMBLE_DEFAULT_METHOD,
        "min_agree": min_agree,
    }


def _run_scope_of(run, run_store=None):
    """The scope a recorded run ran with. config_json carries it; rows
    from before presets had a name kept the scope in the preset column."""
    from layouts.modals import RUN_SCOPES
    scopes = {v for v, _, _ in RUN_SCOPES}
    config_scope = ((run or {}).get("config") or {}).get("scope")
    if config_scope in scopes:
        return config_scope
    if (run or {}).get("preset") in scopes:
        return run["preset"]
    store_scope = (run_store or {}).get("scope")
    return store_scope if store_scope in scopes else "full"


def _run_owner_uid():
    """Owner of a run confirmed in this request: the signed-in user, None
    when anonymous. The same attribution predictions and reports get, so a
    run row and its artifacts always agree on whose they are."""
    try:
        from services.auth_service import effective_uid
        return effective_uid()
    except Exception:
        return None


def _dispatch_run(run_store, dispatched) -> tuple:
    """The run a stage callback fires for, as ``(run_id, row)``; PreventUpdate
    when there is none.

    The stages trigger on run-dispatch (a memory store the dispatcher
    writes on confirm) rather than on the confirm click: the click's own
    callbacks all fire at once, before the dispatcher could have written
    the id. A value the stages already handled (run-dispatched carries the
    last one) or a run no longer in flight is not a new confirm. A dict
    with no run_id is a dispatcher bug, not an idle store, and is said so
    on the feed; only None is silent.

    The row rides along because it is the stage's settings (see
    _run_settings); ``row`` is None only when the record could not be read.
    """
    if run_store is None:
        raise PreventUpdate
    run_id = (run_store or {}).get("run_id")
    if not run_id:
        logger.warning("run dispatch ignored: store carries no run_id (%r)",
                       run_store)
        try:
            from services import progress_service as prog
            # Ad-hoc bucket, not the newest run in flight: that could be
            # another user's feed.
            prog.emit("error", "Run not started: the dispatcher wrote a run "
                      "with no id; confirm again", run_id=prog.ADHOC_RUN_ID)
        except Exception as e:
            logger.debug("run dispatch feed line failed: %s", e)
        raise PreventUpdate
    if run_id == (dispatched or {}).get("run_id"):
        raise PreventUpdate
    try:
        from services import run_service
        row = run_service.get_run(run_id)
    except Exception as e:
        logger.warning("run %s: run row unreadable: %s", run_id[:8], e)
        row = None
    if row is not None and not row["active"]:
        raise PreventUpdate
    return run_id, row


def _dispatch_run_id(run_store, dispatched) -> str:
    """_dispatch_run without the row, for callers that only need the id."""
    return _dispatch_run(run_store, dispatched)[0]


def _run_settings(row) -> dict | None:
    """What a stage runs with: the run row's config_json, written once by
    the confirm (or copied by a retry) and never by the dialog afterwards.

    The dialog's controls are not consulted: the user may have changed a
    preset or a model tick since confirming, a retry must re-run what the
    previous run recorded, and the run page must be able to show exactly
    what ran. None when the row is unreadable, which a stage treats as a
    failure of that run rather than a reason to guess.
    """
    if row is None:
        return None
    return dict(row.get("config") or {})


def _run_symbols_of(row, run_store) -> list:
    """The run's symbol list as the row normalised it (upper-cased, deduped),
    falling back to the dispatcher's snapshot when the row is unreadable."""
    return list((row or {}).get("symbols")
                or (run_store or {}).get("symbols") or [])


def _run_acknowledgement(panel_state, symbols, estimate_s, scope) -> tuple:
    """What every accepted run answers with, confirm and retry alike:
    the activity panel forced open (a user who once closed it otherwise
    gets zero feedback), the started toast with symbols, estimate and
    where the output lands, and the panel's poll snapped to the active
    rate (waiting for a slow idle tick to notice the active flag cost an
    idle interval before the first run events rendered). Returns the
    values for progress-panel-state, run-started-toast is_open/children and
    progress-interval, in that order."""
    from layouts.modals import _fmt_duration

    panel = dict(panel_state or {})
    panel.setdefault("mode", "normal")
    panel["closed"] = False
    where = {
        "models": "predictions will appear on Home and Analyze",
        "report": "the report will appear under Reports",
    }.get(scope, "predictions on Home and Analyze, the report under Reports")
    # A row without an estimate (a retry of one recorded before estimates,
    # or a scope the estimator prices at zero) still gets its toast.
    duration = _fmt_duration(estimate_s) if estimate_s else "duration unknown"
    toast_msg = f"{', '.join(symbols)} · {duration} · {where}."
    return panel, True, toast_msg, _PROGRESS_POLL_ACTIVE_MS


def _active_run_refusal():
    """The validation line for a run this owner may not start because a
    manual run of theirs is in flight, or None when the lock is free.

    A row whose process is provably gone (nothing has reported on it past
    the run ceiling) is failed here rather than refusing this owner
    forever. Scheduled runs never lock: they belong to the scheduler, not
    to whoever is signed in (an anonymous session would otherwise be
    refused for the whole daily job), and the topbar pill is where they
    show. Raises when the row cannot be read; the caller words that.
    """
    from services import progress_service as prog
    from services import run_service

    active = run_service.active_run_for(_run_owner_uid())
    if active is not None and prog.fail_orphan(active):
        active = None
    if active is None:
        return None
    return [
        "You have a run in progress. Cancel it to start another. ",
        dbc.Button("Cancel run", id="run-cancel-active-btn",
                   size="sm", outline=True, color="danger",
                   className="ms-2"),
    ]


def _close_run(run_id, message: str, status: str = "done", error=None) -> None:
    """Close a UI run: the feed's closing line and the analysis_runs status,
    together, from whichever process finishes last. Terminal statuses are
    sticky on the row, so a stage closing after a cancel changes nothing."""
    from services import progress_service as prog
    prog.finish_run(message, run_id=run_id)
    if not run_id:
        return
    try:
        from services import run_service
        first_line = (str(error).strip().splitlines() or [""])[0][:500] \
            if error is not None else None
        run_service.set_status(run_id, status, error=first_line or None)
    except Exception as e:
        logger.warning("run %s: status %s not recorded: %s",
                       run_id[:8], status, e)


@callback(
    Output("run-dispatched", "data"),
    Input("run-dispatch", "data"),
    State("run-dispatched", "data"),
    prevent_initial_call=True,
)
def mark_run_dispatched(run_store, dispatched):
    """Remember which run the stages were handed. Fires in the same batch as
    the stages, so they still see the previous value; a second emission of
    the same id sees this one."""
    run_id = (run_store or {}).get("run_id")
    if not run_id or run_id == (dispatched or {}).get("run_id"):
        raise PreventUpdate
    return {"run_id": run_id, "at": datetime.now().isoformat()}


@callback(
    Output("ai-analysis-store", "data"),
    Input("run-dispatch", "data"),
    # Not a dialog control in sight: every setting comes from the run row's
    # config (see _run_settings). The price store is the only browser-side
    # input, and only as a warm start for the server-side fill.
    State("stock-data-store", "data"),
    State("run-dispatched", "data"),
    prevent_initial_call=True,
)
async def generate_ai_analysis(run_store, stock_data, dispatched):
    """Generate structured AI analysis grounded in financial data and news.

    Triggered by user clicking "AI Report" or "Full Analysis" button.
    Async callback: LLM/research calls fan out through asyncio.to_thread, so
    the minutes-long run holds no callback-threadpool slot while waiting.
    Checks Postgres/S3 cache first if persistence layer is available.

    The run's config supplies an as-of date: articles are filtered to
    that date, price-derived metrics are computed from truncated OHLCV, and
    live quote data is suppressed for past dates, no lookahead in backtests.
    """
    from datetime import date as date_cls
    from functools import partial

    run_id, row = _dispatch_run(run_store, dispatched)
    # The run row's config is the ONE record of what was confirmed; the
    # dialog controls may have moved since (and a retry re-runs what the
    # previous row recorded, not what the dialog shows now). The
    # dispatcher's snapshot is only the fallback for an unreadable row.
    config = _run_settings(row)
    scope = ((config or {}).get("scope") or (run_store or {}).get("scope")
             or "full")
    # A models-only run has no report to build.
    if scope == "models":
        raise PreventUpdate
    # The run's symbol set is the row's (the dialog's chips, normalised),
    # not the watchlist. The report does its own point-in-time fetch per
    # symbol; the browser's news store plays no part.
    symbols = _run_symbols_of(row, run_store)
    if not symbols:
        raise PreventUpdate

    # Every feed line of this stage names its run: the server process hosts
    # every user's report stage at once, so an unnamed emit could land in
    # another user's feed.
    from services import progress_service as prog
    emit = partial(prog.emit, run_id=run_id)
    # The structured counterpart (stage state, done/total, counters): what
    # the run row, the pill and the run page read. Same rule, same run.
    progress = partial(prog.emit_progress, run_id=run_id)
    n_symbols = len(symbols)

    def _refuse(reason: str) -> dict:
        emit("error", f"Report rejected: {reason}")
        progress("report", state="failed", done=0, total=n_symbols)
        _close_run(run_id, f"Report failed: {reason}", "failed", error=reason)
        return {"failed": True, "error": reason, "scope": scope,
                "run_seq": run_id, "run_id": run_id,
                "generated_at": datetime.now().isoformat()}

    if config is None:
        return _refuse("the run's record could not be read, so its "
                       "settings are unknown")

    # Target date: the session whose close is being predicted, as the
    # confirm recorded it. Data is cut off at the PREVIOUS trading day, so
    # a Monday target sees nothing after the preceding Friday's close.
    today = date_cls.today()
    is_full = scope == "full"
    picked = (str(config.get("target_date"))[:10]
              if config.get("target_date") else None)
    from utils.trading_calendar import resolve_target_and_cutoff
    target_d, as_of_d = resolve_target_and_cutoff(picked)
    is_backtest = target_d < today
    as_of_str = as_of_d.isoformat()
    target_str = target_d.isoformat()

    owner_uid = (run_store or {}).get("owner_uid") or _run_owner_uid()
    if is_full:
        prog.start_run(f"Full Analysis: {len(symbols or [])} symbols, "
                       f"target {target_str} (data through {as_of_str})"
                       + (" (backtest)" if is_backtest else ""),
                       run_id=run_id, owner_uid=owner_uid, kind="manual")
    else:
        # A report-only run is a run too: without its own boundary the panel
        # keeps the previous run's green check through the whole generation
        # and polls at the idle rate.
        prog.start_run(f"Report: {len(symbols or [])} symbols, "
                       f"target {target_str} (data through {as_of_str})"
                       + (" (backtest)" if is_backtest else ""),
                       run_id=run_id, owner_uid=owner_uid, kind="manual")
    # First structured event of the run: this is what flips the row from
    # queued to running. Before the price fill below, not after it: a row
    # still queued past the stall window is reaped as a run whose stage
    # never started, and a cold price load can take that long.
    progress("report", state="running", done=0, total=n_symbols)

    # One set of settings, read once, from the row.
    from services.news_window import (
        RunParameterMissing, normalize_article_cap, normalize_lookback)
    try:
        overnight_news, lookback_days = normalize_lookback(config.get("lookback"))
        max_articles = normalize_article_cap(config.get("max_articles"))
    except RunParameterMissing as e:
        # Refuse, visibly. No default: a report on a window the user did not
        # pick would be indistinguishable from the one they asked for.
        return _refuse(str(e))
    window_desc = ("overnight close→open" if overnight_news
                   else f"{lookback_days}d")
    report_model = config.get("report_model") or "gpt-5.6-luna"
    include_thesis_flag = (config.get("depth") or "thesis") != "standard"
    recs_mode = config.get("recs") or "auto"
    recs_model_val = config.get("recs_model") or "claude-sonnet-5"
    # Terminal-derived evidence blocks (modal checklist); None only when the
    # modal had not rendered once at confirm. Treat as the default, both on.
    evidence_sel = (sorted(config["evidence"])
                    if config.get("evidence") is not None
                    else sorted(MODEL.DEFAULT_EVIDENCE))
    # Tools are opt-in: absent means none (the backend never switches one on).
    tools_sel = sorted(set(config.get("tools") or []))
    report_provider = "openai" if report_model.startswith("gpt-") else "anthropic"
    # Per-symbol analysis is always the full research agent now, the shallow
    # summarize_news_structured pass produced a thinner second opinion on the
    # same data and was removed (2026-08-08). Full pipeline still gets its
    # research through the prediction stage's trading_agents model; running
    # it here too would double the LLM spend for identical output.
    include_research = not is_full
    if recs_mode == "off":
        progress("synthesis", state="skipped")

    # Metric/signal context for symbols outside the browser stores (e.g. a
    # cohort-scoped or single-symbol run) is fetched server-side.
    missing = [s for s in symbols
               if not ((stock_data or {}).get(s) or {}).get("prices")]
    if missing:
        stock_data = await asyncio.to_thread(
            _fill_run_inputs, symbols, stock_data)

    emit("ai", f"AI Report starting for {', '.join(symbols or [])}")
    emit("action", f"Options: model={report_model}, window={window_desc}, "
                        f"type={'thesis' if include_thesis_flag else 'standard'}, "
                        f"research={'pipeline' if is_full else 'on'}, "
                        f"recs={recs_mode} ({recs_model_val}), "
                        f"target {target_str}, data through {as_of_str}")

    # One data_load event covering EVERY symbol's price input for this run,
    # whatever its origin, the browser session store (the normal case, which
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
    # window: per-symbol start/end record what was actually consumed.
    emit("action",
              f"Price inputs resolved: "
              f"{sum(1 for v in price_inputs.values() if v.get('bars'))}"
              f"/{len(symbols or [])} symbols (data through {as_of_str})",
              payload={"event": "data_load", "by_symbol": price_inputs})

    # News window: the SAME point-in-time fetch the modal preview showed and
    # the models subprocess uses, windowed [as_of - lookback, as_of] (or the
    # overnight close→open gap), capped at the newest max_articles, lookahead-
    # safe. The dcc.Store's articles are never used here: they hold whatever
    # the last background refresh grabbed from "now", which is the wrong
    # window for any backtest and an unlabelled one live.
    from services.news_window import (
        describe_news_window, fetch_run_news, news_window_payload)
    news_seen = {"symbols": 0, "articles": 0}

    def _news_progress(sym: str, articles: list, stats: dict) -> None:
        news_seen["symbols"] += 1
        news_seen["articles"] += len(articles)
        # Same verdict the scheduled runner records: the source failing
        # for this symbol is a per-symbol failure, a quiet window is not.
        unavailable = stats.get("status") == "unavailable"
        progress("news", symbol=sym, state="failed" if unavailable else "done",
                 error=(stats.get("error") or "news source unavailable")
                 if unavailable else None)
        progress("news", done=news_seen["symbols"], total=n_symbols,
                 state="done" if news_seen["symbols"] == n_symbols else "running",
                 articles=news_seen["articles"])

    progress("news", done=0, total=n_symbols, state="running")
    articles_by_symbol, news_stats = await asyncio.to_thread(
        fetch_run_news, symbols or [], as_of_str, target_str,
        overnight=overnight_news, lookback_days=lookback_days,
        max_articles=max_articles, on_symbol=_news_progress)
    total_articles = sum(len(a) for a in articles_by_symbol.values())
    news_payload = news_window_payload(
        overnight=overnight_news, lookback_days=lookback_days,
        max_articles=max_articles, as_of=as_of_str, target=target_str,
        stats_by_symbol=news_stats)
    emit("news", describe_news_window(news_payload), payload=news_payload)
    news_down = [s for s, v in news_stats.items() if v["status"] == "unavailable"]
    if news_down:
        emit("error",
                  f"News source unavailable for {len(news_down)} of "
                  f"{len(symbols or [])} symbols: {', '.join(news_down)}: "
                  f"the report has NO news for them; treat their sections "
                  f"as unsupported, not as quiet weeks.")

    # One key dict for both the cache check here and the store at the end. 
    # and the trace payload records its hash plus a compact summary of what
    # fed it, so a cross-restart cache miss stops being undiagnosable.
    from services.analysis_runner import report_cache_key as _report_key
    report_cache_key = _report_key(
        articles_by_symbol, symbols or [], as_of_str, report_model,
        lookback_days, include_thesis_flag, evidence=evidence_sel,
        max_articles=max_articles, overnight=overnight_news,
        include_research=include_research, recs_mode=recs_mode,
        tools=tools_sel)

    # Check persistent cache (Postgres + S3) before running LLM. A retry
    # exists because the cached (or failed) answer was not good enough.
    if _s3_available and not (run_store or {}).get("retry_of"):
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
                emit("ai", "AI report cache miss, generating fresh",
                          payload={**report_cache_payload, "outcome": "miss"})
            if cached:
                logger.info("AI report cache hit (Postgres/S3)")
                emit("ai", "AI report cache hit, regeneration skipped",
                          payload={**report_cache_payload, "outcome": "hit"})
                cached_result = json.loads(cached)
                cached_result["from_cache"] = True
                # Re-stamp for THIS click. The stored copy carries the
                # correlation keys of the run that originally produced it.
                cached_result["run_seq"] = run_id
                cached_result["run_id"] = run_id
                cached_result["scope"] = scope
                progress("report", state="done", done=n_symbols,
                         total=n_symbols)
                if not is_full and recs_mode == "off":
                    # Nothing left to run for this scope: close the run.
                    _close_run(run_id, f"Report complete ({len(symbols or [])} "
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
        # Correlation keys: run_seq ties this payload to the run that
        # produced it (the model-signals _meta carries the same value),
        # and scope lets the synthesis callback tell a report-only payload
        # from a full-pipeline one even if a stale requested flag lingers.
        "run_seq": run_id,
        "run_id": run_id,
        "scope": scope,
    }

    # Per-symbol context: fundamentals, validated metric blocks, events, peers
    enriched_stock_data = {}
    extra_blocks_by_symbol = {}
    for symbol in (symbols or []):
        # One line per symbol: this loop is network-bound (profile, events,
        # peers, options, quality) and used to run in total silence.
        emit("ai", f"{symbol}: gathering fundamentals, options, "
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
            # Store metrics/signals are computed on current data, always drop.
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
                # Cutoff applies live too, peer relative strength must be
                # measured through the previous trading day, not "latest".
                block = compute_peer_relative_strength(
                    symbol, peers, as_of=as_of_str)
                if block:
                    blocks["peers"] = block

            profile = get_company_profile(symbol)
            if profile:
                blocks["profile"] = profile

            # Options positioning + Bad Apples quality screen, both are
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
        # None instead makes the agent fetch fresh 1y data itself, the
        # guard then only fires if even the FRESH fetch is stale, which is
        # the case it exists for.
        if df is not None and len(df):
            last_bar = pd.to_datetime(df.index.max()).date()
            age = (date_cls.fromisoformat(as_of_str) - last_bar).days
            if age > 5:
                emit("ta", f"{sym}: cached prices end {last_bar} "
                                f"({age}d before {as_of_str}): refetching fresh")
                df = None
        model_obj = TradingAgentsModel()
        # Runs in a worker thread via asyncio.to_thread (context copies in),
        # but the label is set here so the trace/spend attribution never
        # depends on what the event-loop context happened to hold.
        with _usage.track("research", symbol=sym, trade_date=as_of_str,
                          section=f"research:{sym}"):
            # The run's own windowed news. Without it the agent fetched
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
    # each task completes (single-threaded. `result` mutations are safe).
    n_workers = min(len(symbol_tasks) * (2 if include_research else 1) + 1, 8)
    sem = asyncio.Semaphore(n_workers)

    # Research progress: one step per symbol whose task ended, written or
    # not; the reports counter is what actually landed.
    research_seen = {"done": 0, "written": 0}

    def _research_step(symbol: str, written: bool, error=None) -> None:
        research_seen["done"] += 1
        research_seen["written"] += 1 if written else 0
        # The symbol's own glyph first, then the stage's running count.
        progress("research", symbol=symbol,
                 state="done" if written else "failed",
                 error=None if written else (error or "no report written"))
        progress("research", done=research_seen["done"],
                 total=len(symbol_tasks),
                 research_reports=research_seen["written"])

    async def _run_task(task_type, symbol, fn, *args, **kw):
        try:
            async with sem:
                if task_type == "research":
                    progress("research", symbol=symbol, state="running")
                analysis = await asyncio.to_thread(fn, *args, **kw)
        except Exception as e:
            logger.warning(f"Error in LLM analysis for {task_type} {symbol}: {e}")
            emit("error", f"AI analysis failed for {symbol or 'overall'}: {str(e)[:80]}")
            if task_type == "research":
                _research_step(symbol, False, str(e))
            return

        if task_type == "research":
            r = analysis  # PredictionResult
            if r is None or r.error:
                emit("error", f"{symbol}: research report failed: "
                                   f"{(r.error if r else 'no result')[:80]}")
                _research_step(symbol, False, r.error if r else "no result")
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
                    run_id=run_id,
                )
            except Exception as e:
                logger.warning(f"research report persist failed for {symbol}: {e}")
            emit("ta", f"{symbol}: research → {r.decision} "
                            f"({r.confidence:.0%})")
            _research_step(symbol, True)
            return
        if analysis:
            if task_type == "symbol":
                result["by_symbol"][symbol] = analysis
                emit("ai", f"{symbol}: {analysis.get('recommendation', '?')} "
                                f"({int((analysis.get('confidence') or 0) * 100)}%)")
            else:
                result["overall"] = analysis
                emit("ai", f"Overall: {analysis.get('recommendation', '?')}")

    aio_tasks = []

    if include_research:
        progress("research", done=0, total=len(symbol_tasks),
                 state="running", research_reports=0)
        for symbol in symbol_tasks:
            aio_tasks.append(_run_task("research", symbol, _run_research, symbol))
            emit("ta", f"{symbol}: research report starting ({report_model})…")

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
        # was empty: the model reported a broken data feed instead of an
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
        emit("ai", f"Overall: synthesizing {len(overall_articles)} articles "
                        f"across {len(symbols or [])} symbols…")

    await asyncio.gather(*aio_tasks)

    if include_research:
        # Same verdicts the scheduled runner records: done when reports
        # landed, failed when research was asked for and none did.
        progress("research", done=research_seen["done"],
                 total=len(symbol_tasks),
                 state="done" if research_seen["written"] else "failed",
                 research_reports=research_seen["written"])

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
            # The epilogue supplies the panel fields the shallow pass used to. 
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
        emit("error", "AI Report produced no analysis")
        progress("report", state="failed", done=0, total=n_symbols)
    else:
        # Count what the report covered. Full scope defers the per-symbol
        # research to the model stage, so by_symbol is legitimately empty
        # there: fall back to the run's symbol list rather than saying "0".
        n_covered = len(result["by_symbol"]) or len(symbols or [])
        if is_full:
            emit("ai", f"AI Report complete ({n_covered} symbols): "
                            "model predictions next")
        elif recs_mode != "off":
            emit("ai", f"AI Report complete ({n_covered} symbols): "
                            "synthesizing recommendations next")
        else:
            emit("ai", f"AI Report complete ({n_covered} symbols)")
        progress("report", state="done", done=n_covered, total=n_symbols)

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
            _close_run(run_id, "Report failed: no analysis produced", "failed",
                       error="AI report produced no analysis")
        elif recs_mode == "off":
            _close_run(run_id, f"Report complete "
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
    it, so on Performance it reported "No stocks selected" and never opened.
    a control that looked broken rather than inapplicable.
    """
    triggered = ctx.triggered_id

    if triggered == "modal-close-btn":
        return False, dash.no_update

    # The button is page-local now (allow_optional Input), so this callback
    # also fires when navigation MOUNTS it, with n_clicks still None, that
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
        # Export what the page is showing, filters and all, the scoreboard is
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
        # Full Analysis keeps its research on the signals store, merge it in
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

    Checks S3 cache first, if an identical report already exists (same input hash),
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
                logger.info("PDF report cache hit, returning existing report")
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

    # Full Analysis keeps its research on the signals store. Merge it in so
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
    # themselves: a hardcoded list silently desyncs when the selector grows.
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
    Input("run-dispatch", "data"),
    # No dialog control and no browser store: the run row's config is the
    # whole setting set (see _run_settings), and the price input comes from
    # the server-side cache at a fixed depth. Fewer States also means less
    # serialised into the subprocess (the price store was never read).
    State("run-dispatched", "data"),
    background=True,
    # No running= spec: the topbar pill reads the run row on the poll and
    # covers every stage, not just this subprocess.
    prevent_initial_call=True,
)
def generate_model_signals(run_store, dispatched):
    """Generate model predictions in a background subprocess.

    Supports backtesting: when the selected as-of date is in the past, OHLCV
    data is truncated to that date so models only see data available
    as of that day. The selected date is stored in metadata for
    correct persistence.

    The run id arrives as an argument (run-dispatch is an Input, so its
    value is serialised into this subprocess) and is asserted before any work:
    the row it names is what the persisted predictions, the feed and the
    cancel path key on, so a stage without one must not run at all.
    """
    from functools import partial

    run_id, row = _dispatch_run(run_store, dispatched)
    # The row's config is what this run was confirmed with; the dialog's
    # controls may have moved since, and a retry re-runs the previous row.
    # The dispatcher's snapshot only stands in for an unreadable row.
    config = _run_settings(row)
    scope = ((config or {}).get("scope") or (run_store or {}).get("scope")
             or "full")
    # A report-only run has no models to execute.
    if scope == "report":
        raise PreventUpdate
    symbols = _run_symbols_of(row, run_store)
    if not symbols:
        raise PreventUpdate

    # The model checkboxes are now honoured on every scope. Full pipeline used
    # to overwrite them with [True] * 5, so the boxes you unticked ran anyway.
    is_full_analysis = scope == "full"
    from services import progress_service as _prog
    _emit = partial(_prog.emit, run_id=run_id)

    def _abort_run(exc: Exception) -> dict:
        """Close the run as a failure and hand downstream a marked payload.

        logger.exception, not error: this body runs in the background
        subprocess, where an escaped exception is swallowed into Dash's job
        error: the browser sees a bare HTTP 500 and the server log NOTHING.
        The traceback must be written here or it exists nowhere.

        A marked payload, not {}: downstream must be able to tell "the run
        failed" from "no predictions". The empty dict cleared the running
        badge as if successful and left the panel spinning forever.
        """
        logger.exception("Model signal generation error")
        _emit("error", f"Model run failed: {str(exc)[:200]}")
        _close_run(run_id, ("Full Analysis" if is_full_analysis
                            else "Predictions")
                   + f" failed: {str(exc)[:120]}", "failed", error=exc)
        return {"_run_failed": str(exc),
                "_meta": {"run_seq": run_id, "run_id": run_id}}

    # Everything after the guards runs under a catch-all. A one-line bug
    # before the old try boundary shipped as "every run dies at +1s with a
    # bare HTTP 500 and a forever-running badge". Nothing here may execute
    # unprotected.
    try:
        if config is None:
            raise RuntimeError("the run's record could not be read, so its "
                               "settings are unknown")
        # For anything in this process that reads the run from the
        # environment; the id itself travels as an argument.
        os.environ["QUANTNEWS_RUN_ID"] = run_id
        if not is_full_analysis:
            # Models-only owns the feed; the full pipeline started it
            # already. Started BEFORE the input fill so the first emits below
            # group under this run, not the previous one.
            _prog.start_run(f"Predictions: {len(symbols or [])} symbols",
                            run_id=run_id,
                            owner_uid=(run_store or {}).get("owner_uid"),
                            kind="manual")
        else:
            # The server opened this run; this process only executes one of
            # its stages. Unnamed emits and the LLM usage rows written here
            # must still group under it.
            _prog.adopt_run(run_id)

        # First line before any heavy work: the subprocess spends its first
        # seconds on imports and input fetches with nothing on the panel.
        # (emit itself touches only diskcache + Postgres, no torch.)
        # The pid rides along and is recorded beside the run key, so the
        # server's watchdog can tell "worker died" from "worker still
        # grinding": this process has been observed dying without a
        # traceback (torch/MPS native teardown), which previously left the
        # run spinning forever.
        _emit("models", "Preparing model engine (first run loads model "
                        "weights)…",
              payload={"event": "run_process", "pid": os.getpid()})
        _prog.record_run_pid(run_id)
        # Flip the row from queued to running now, not after the input
        # fill and the model loads: a row still queued past the stall
        # window is reaped as a run whose stage never started.
        _prog.emit_progress("models", state="running", done=0,
                            total=len(symbols), run_id=run_id)

        # The dialog's news window and cap govern the models too, the
        # sentiment and research models used to read the browser's news
        # store (a rolling week from "now", 50 articles), whatever the
        # dialog said. The window's days ride along so the research agent
        # does not re-filter to its own default.
        from services.news_window import (
            normalize_article_cap, normalize_lookback)
        overnight_news, lookback_days = normalize_lookback(config.get("lookback"))
        max_articles = normalize_article_cap(config.get("max_articles"))
        research_kwargs = {"news_lookback_days": lookback_days}
        if is_full_analysis:
            # The research report (trading_agents) honours the report
            # model/depth choices; `research_model` is a distinct kwarg so no
            # other model can mistake it for its own.
            research_kwargs.update({
                "research_model": config.get("report_model") or None,
                "include_thesis": (config.get("depth") or "thesis") != "standard",
                # Modal checklist; None only when the modal had not
                # rendered once at confirm.
                "evidence": (sorted(config["evidence"])
                             if config.get("evidence") is not None
                             else sorted(MODEL.DEFAULT_EVIDENCE)),
                "tools": sorted(set(config.get("tools") or [])),
            })

        # Module-level os, a function-local `import os` here once shadowed
        # the name for the WHOLE body, so the pid emit above raised
        # UnboundLocalError on every run. Never re-import module-level names
        # inside this function.
        os.environ["_DASH_BG_SUBPROCESS"] = "1"

        from datetime import date as date_cls

        # The row's target_date is the TARGET session, the close being
        # predicted. Models are truncated to `predict_date`, the previous
        # trading day, so a Monday target trains and scores on nothing
        # after the preceding Friday.
        from utils.trading_calendar import resolve_target_and_cutoff
        target_date, predict_date = resolve_target_and_cutoff(
            str(config["target_date"])[:10] if config.get("target_date")
            else None
        )

        is_backtest = target_date < date_cls.today()

        # An empty list runs the runner's full set, as an unticked dialog
        # always did.
        selected_models = set(config.get("models") or [])
        run_ensemble = bool(config.get("ensemble"))
        ensemble_config = _ensemble_config_of(config)
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
                _emit("error", f"{sym}: skipped: no price data available")

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
        _emit("news", describe_news_window(news_payload),
              payload=news_payload)
        news_down = [s for s, v in news_stats.items()
                     if v["status"] == "unavailable"]
        if news_down:
            _emit("error",
                  f"News source unavailable for {len(news_down)} of "
                  f"{len(priced)} symbols: {', '.join(news_down)}. "
                  f"Their sentiment and research models run WITHOUT "
                  f"news; treat those calls as unsupported.")

        # ONE implementation of the model stage (services.analysis_runner):
        # the interactive copy that lived here had drifted from the
        # scheduled one: no pipeline-epoch/news-status/regime stamps (so
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
        # Ties this store to the run that produced it, so the synthesis
        # stage can pair it with the same run's AI report and the server
        # can stamp the rows it persists.
        results["_meta"]["run_seq"] = run_id
        results["_meta"]["run_id"] = run_id
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
    meta = signals.get("_meta") or {}
    run_id = meta.get("run_id")
    fa_requested = _synthesis_armed(fa_requested, run_id)
    # The worker's output is in hand. Its pid no longer stands for the
    # run's health, alive or not (the watchdog must not flag a subprocess
    # that exited after delivering).
    from services import progress_service as _prog_pid
    _prog_pid.clear_run_pid(run_id)
    if signals.get("_run_failed"):
        # The background handler already emitted the failure and finished
        # the run: pass a failed status through so the pages listening on
        # this store still refresh, instead of pretending "Stored 0".
        return {"failed": signals["_run_failed"], "count": 0,
                "stored_at": str(datetime.now())}

    predict_date_str = meta.get("predict_date")
    is_backtest = meta.get("is_backtest", False)

    try:
        # ONE writer (services.analysis_runner.persist_predictions): the copy
        # that lived here recorded the model that was ASKED for on research
        # reports instead of the one that answered after a provider fallback.
        from services.analysis_runner import persist_predictions as _persist
        stored, evaluated = _persist(signals, run_id=run_id)
        from services import progress_service as prog
        # New rows: the launch screen must show this run, not the last one.
        from services.dashboard_service import invalidate_memo
        invalidate_memo()
        if not fa_requested:
            _close_run(run_id, f"Predictions complete: {stored} stored")
        elif ((ai_analysis or {}).get("failed")
              and (ai_analysis or {}).get("run_seq") == meta.get("run_seq")):
            # This run's report already failed: no synthesis is coming, so
            # don't announce a handoff that never happens. The synthesis
            # callback sees the same store pair and closes the run.
            pass
        else:
            prog.emit("luna", "Handing off to recommendation synthesis…",
                      run_id=run_id)

        return {
            "stored_at": str(datetime.now()),
            "count": stored,
            "evaluated": evaluated,
            "is_backtest": is_backtest,
            "predict_date": predict_date_str,
            "run_id": run_id,
        }

    except Exception as e:
        logger.error(f"Prediction persistence error: {e}")
        from services import progress_service as prog
        prog.emit("error", f"Prediction storage failed: {str(e)[:200]}",
                  run_id=run_id)
        if not fa_requested:
            _close_run(run_id, "Predictions finished with errors, storage "
                               "failed", "failed", error=e)
        else:
            # The signals are still in the store, so synthesis can proceed
            # and owns the finish; only the persisted rows are missing. The
            # row keeps the error text while it stays running, so the run
            # page can say what was lost once synthesis closes it as done.
            if run_id:
                try:
                    from services import run_service
                    run_service.set_status(
                        run_id, "running",
                        error=f"prediction storage failed: {str(e)[:400]}")
                except Exception as e2:
                    logger.warning("run %s: storage error not recorded: %s",
                                   run_id[:8], e2)
            prog.emit("luna", "Handing off to recommendation synthesis…",
                      run_id=run_id)
        return {"error": str(e), "stored_at": str(datetime.now()),
                "run_id": run_id}


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
            # Zero is the usual outcome. The 6pm scheduler and post-backtest
            # auto-eval normally score everything first. Only warn when mature
            # rows are genuinely stuck without a close to score against.
            backlog = cache.evaluation_backlog()
            pending = backlog.get("pending_mature", 0)
            if pending:
                dates = ", ".join(sorted(backlog.get("by_target_date", {}))[:4])
                msg = (f"{pending} prediction{'s' if pending != 1 else ''} "
                       f"due for scoring could not be evaluated, no closing "
                       f"price stored for target date{'s' if pending != 1 else ''} "
                       f"{dates}. Load those symbols to refresh price data.")
                icon = "warning"
            else:
                msg = ("All caught up, every matured prediction is already "
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
    # when the report itself lands and when a recommendation run persists. 
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

        # Research reports: the recent archive for everyone, plus deeper
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


# Poll fast enough to feel live during a run, slowly enough the rest of the
# time that an idle tab is not making 40 requests a minute forever. A
# scheduled 08:00 job is still picked up within the idle interval. The rate
# is owned by render_run_pill (it already reads the active runs); the
# confirm and retry acknowledgements snap it with allow_duplicate.
_PROGRESS_POLL_ACTIVE_MS = 1500
_PROGRESS_POLL_IDLE_MS = 10_000


@callback(
    Output("progress-feed-scroll", "children"),
    Output("progress-count", "children"),
    Output("progress-header-icon", "children"),
    Output("progress-snap-store", "data"),
    Output("progress-fp-store", "data"),
    Input("progress-interval", "n_intervals"),
    State("progress-snap-store", "data"),
    State("progress-fp-store", "data"),
    State("run-store", "data"),
)
def render_progress_panel(_n, last_snap, last_fp, run_store=None):
    """Live activity panel: a stepper over the run it follows, the log
    beneath it (layouts/progress_panel.py draws both).

    Streams events emitted by every stage (including the background model
    subprocess). Visible on every route: this is the live view, and the
    Activity section is the filtered archive on top of it. Runs on every
    poll tick, so it is one row read (get_run) and one feed read, nothing
    else, and it rewrites nothing while its fingerprint stands still.
    """
    from layouts import progress_panel as pp
    from services import progress_service as prog
    from services import run_service

    # Watchdog first: a run whose worker process died (observed: torch/MPS
    # native crash with no traceback) or that stopped emitting must become a
    # visible failure on this very tick, not an eternal spinner. Cheap when
    # idle (one diskcache read); emits + closes the run at most once.
    prog.watchdog_check()

    # Which run to follow is decided from the feed's active list and the
    # session's pin BEFORE the row read, so there is exactly one: this
    # viewer's own run while it is in flight or for an hour after it ended
    # (its result stays in front of them), else the newest run in flight
    # (a scheduled job, another session's run), else the rolling log.
    # Without the pin, confirming a run in another session (or a scheduled
    # job starting) switched every open panel to that run's feed.
    now = pp.now_utc()
    active_ids = prog.active_run_ids()
    target = pp.pin_target(run_store, active_ids, now)
    run = None
    if target:
        try:
            run = run_service.get_run(target)
        except Exception as e:
            logger.warning("activity panel: run %s unreadable: %s", target, e)
        if run is not None and not pp.pin_holds(run, active_ids, now):
            run = None
    feed = prog.get_feed(run["run_id"] if run else None)
    events = feed.get("events") or []
    active = bool(feed.get("active"))
    # Newest run boundary. Written only when a NEW run appears, so the
    # clientside snap-to-newest fires once per run. A scroll offset parked
    # on the previous run's lines otherwise hides the new run entirely.
    snap = next((e.get("t") for e in reversed(events)
                 if e.get("stage") == "run"), None)
    snap_out = snap if (snap is not None and snap != last_snap) \
        else dash.no_update
    if not events and run is None:
        return (dash.no_update, dash.no_update, dash.no_update,
                dash.no_update, dash.no_update)

    # What this tick would render: the visible window's newest edge and
    # size, the run boundary, the active flag (drives the header icon) and
    # the run row's stepper facts. If none of it moved, rewrite nothing.
    # Rebuilding the ~45 row nodes every tick destroyed and recreated them
    # all, which reset the feed's scroll position on every poll, idle
    # included. Count + newest timestamp + boundary also cover the feed
    # being swapped underneath us (cold-start rehydrate, another publisher
    # process). The feed's run id stays last: readers index it from the end.
    newest = events[-1] if events else {}
    fp = [len(events), newest.get("t"), newest.get("stage"),
          newest.get("message"), snap, active, pp.run_fingerprint(run),
          feed.get("run_id")]
    if fp == last_fp:
        return (dash.no_update, dash.no_update, dash.no_update, snap_out,
                dash.no_update)

    # No auto-hide: the panel is an audit log now, so it stays up until the
    # user closes it. (It previously vanished 5 minutes after a run finished.)
    return (pp.body(run, events), f"{len(events)} events",
            pp.header_icon(active, run), snap_out, fp)


# Scroll the feed to the newest lines when a new run starts. Clientside:
# scroll offsets are a DOM concern no server response can touch. Two gates:
# the server writes the token only when a new run boundary appears, and the
# browser remembers the last token it acted on, so even a rewrite of the
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
    Input("progress-panel-state", "data"),
)
def apply_progress_panel_state(state):
    """Translate panel state into layout. Sizing lives in CSS classes.

    The poll keeps running with the panel shut: the topbar pill reads the
    same interval, and a closed panel used to silence the app for the rest
    of the run (no pill update, no completion, no watchdog tick).
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
    )


# =============================================================================
# TOPBAR RUN PILL
# =============================================================================


def _read_pill_runs(owner_uid, run_store) -> tuple:
    """The two reads behind the pill: every run in flight, and, only when
    none of them is this viewer's, the viewer's newest manual run (its
    Ready/Failed state). One indexed select each.

    A scheduled run in flight does not skip the second read: the daily job
    runs for tens of minutes, and a manual run that finishes inside that
    window must still be announced (and could be replaced by a newer one
    before the job ends, losing its completion for good).
    """
    from layouts.run_pill import is_own
    from services import run_service

    active = run_service.active_runs()
    latest = None
    if not any(is_own(r, owner_uid, run_store) for r in active):
        rows = run_service.list_runs(limit=1, kind="manual",
                                     owner_uid=owner_uid or "")
        latest = rows[0] if rows else None
    return active, latest


@callback(
    Output("run-pill", "hidden"),
    Output("run-pill", "className"),
    Output("run-pill", "children"),
    Output("run-pill-fp", "data"),
    Output("progress-interval", "interval"),
    Output("run-done-toast", "is_open"),
    Output("run-done-toast", "header"),
    Output("run-done-toast", "children"),
    Output("run-done-toast", "icon"),
    Output("run-notified-store", "data"),
    Input("progress-interval", "n_intervals"),
    State("run-store", "data"),
    State("run-seen-store", "data"),
    State("run-pill-fp", "data"),
    State("progress-interval", "interval"),
    State("run-notified-store", "data"),
)
async def render_run_pill(_n, run_store, seen, last_fp, current_interval,
                          notified=None):
    """The topbar pill, the poll rate and the completion toast, from the
    run rows.

    Runs every tick, so it is one or two indexed selects in a worker thread
    and nothing else; the pill is rewritten only when its fingerprint
    moves (layouts/run_pill.py). The poll rate follows the rows, not the
    feed or the panel: fast while any run is in flight, idle otherwise.

    The toast is the viewer's own run announced once: the tick that first
    sees it finished opens it and records the run in run-notified-store,
    and no later tick (fingerprint moved or not) opens it again for that
    run. It stays until dismissed. It is decided from the finished run,
    not from the pill's choice, so a scheduled run in flight never
    swallows it.
    """
    from layouts.run_pill import (done_toast, finished_view, pill_children,
                                  pill_view)

    owner_uid = _run_owner_uid()
    try:
        active, latest = await asyncio.to_thread(
            _read_pill_runs, owner_uid, run_store)
    except Exception as e:
        # Leave the pill as it was rather than blanking it on a DB hiccup.
        logger.warning("run pill: runs unreadable: %s", e)
        raise PreventUpdate
    poll = _PROGRESS_POLL_ACTIVE_MS if active else _PROGRESS_POLL_IDLE_MS
    # Writing interval on every tick restarts the timer and costs a DOM
    # update for no change; only send it when the rate actually changes.
    poll_out = poll if poll != current_interval else dash.no_update

    view = pill_view(active, latest, owner_uid, run_store, seen)
    toast = done_toast(finished_view(latest, owner_uid, run_store, seen),
                       notified)
    toast_out = ((True, toast["header"], toast["body"], toast["icon"],
                  {"run_id": toast["run_id"]}) if toast
                 else (dash.no_update,) * 5)
    fp = view["fp"] if view else None
    last = last_fp.get("fp") if isinstance(last_fp, dict) else None
    if fp == last:
        return (dash.no_update, dash.no_update, dash.no_update,
                dash.no_update, poll_out) + toast_out
    if view is None:
        return (True, "run-pill", [], None, poll_out) + toast_out
    return (False, view["className"], pill_children(view),
            {"fp": fp, "pin": view["pin"]}, poll_out) + toast_out


@callback(
    Output("progress-panel-state", "data", allow_duplicate=True),
    Output("run-store", "data", allow_duplicate=True),
    Input("run-pill", "n_clicks"),
    State("run-pill-fp", "data"),
    State("run-store", "data"),
    State("progress-panel-state", "data"),
    prevent_initial_call=True,
)
def open_run_from_pill(n_clicks, fp, run_store, panel_state):
    """A click on the pill opens the activity panel on that run.

    The panel pins its feed to run-store, so a pill showing a run this
    session did not confirm (a scheduled job, or the viewer's run from
    another tab) hands the panel that run's dict. The pin was written by
    the poll with the pill, so no query here.
    """
    if not n_clicks:
        raise PreventUpdate
    panel = dict(panel_state or {})
    panel.setdefault("mode", "normal")
    panel["closed"] = False
    pin = (fp or {}).get("pin") or {}
    if not pin.get("run_id") or pin["run_id"] == (run_store or {}).get("run_id"):
        return panel, dash.no_update
    return panel, pin


@callback(
    Output("run-pill", "hidden", allow_duplicate=True),
    Output("run-pill-fp", "data", allow_duplicate=True),
    # A cancelled full run leaves the synthesis flag armed (see
    # cancel_active_run); the pill's cancel must disarm it the same way.
    Output("full-analysis-requested", "data", allow_duplicate=True),
    # The button is only rendered while the pill shows the viewer's own run.
    Input("run-pill-cancel", "n_clicks", allow_optional=True),
    prevent_initial_call=True,
)
def cancel_run_from_pill(n_clicks):
    """The pill's cancel: the same cancel the dialog offers, then the pill
    goes away (a cancelled run is never shown; the next tick brings back
    whatever else is in flight)."""
    if not n_clicks:
        raise PreventUpdate
    _message, run_id = _cancel_own_run()
    if run_id is None:
        # Already gone; the poll hides the pill on its own.
        raise PreventUpdate
    return True, None, False


# Visiting a run's page marks it seen, so the Ready/Failed pill clears.
# Clientside on the URL rather than on the View link: dcc.Link has no
# n_clicks, and a direct visit should count the same as a click.
clientside_callback(
    """
    function(pathname, seen) {
        var m = /^[/]runs[/]([0-9a-fA-F-]{36})(?:[/?#]|$)/.exec(pathname || "");
        if (!m) {
            return window.dash_clientside.no_update;
        }
        if (seen && seen.run_id === m[1]) {
            return window.dash_clientside.no_update;
        }
        return {run_id: m[1]};
    }
    """,
    Output("run-seen-store", "data"),
    Input("url", "pathname"),
    State("run-seen-store", "data"),
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
    moved: the Data/Models groups follow activity_log, the LLM group follows
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
    # The feed names the run in flight whichever process opened it; this
    # process's own run id is only set for runs it started itself.
    live_run = feed.get("run_id")
    poll = (trace_page.POLL_ACTIVE_MS if active and run_id == live_run
            else trace_page.POLL_IDLE_MS)
    poll_out = poll if poll != wm.get("poll") else dash.no_update

    # Refresh the selector's OPTIONS (never its value, an explicitly picked
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
    """Fetch one call's prompt/response bodies on expand (and only then).
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
    leave the range store alone. Otherwise clicking "7d" would immediately
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
    chips/Clear-all are (re)inserted into the layout, with all n_clicks None.
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
    Output("ta-report-modal-footer", "children"),
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

    from models.single_agent import extract_confidence
    from layouts.formatters import conviction_label, weight_label
    from layouts.report_view import build_report_view

    # The bare "(50%)" this used to print was the reliability weight, sitting
    # unlabeled beside the report body's own conviction line. Both numbers are
    # named now so a reader can tell which is which.
    stated = extract_confidence(report.get("report_text") or "")
    title = (f"{report.get('symbol', '?')}: {report.get('decision', '?')} · "
             f"{conviction_label(stated)} · {weight_label(report.get('confidence'))}"
             f": {report.get('trade_date', '')}")
    from services import progress_service as prog
    prog.emit("action", f"Report viewed: {report.get('symbol', '?')} "
                        f"{report.get('trade_date', '')}")
    # The report as a page: verdict term sheet, numbered sections with their
    # takeaways, Read lines pulled out, provenance footer. The PDF below uses
    # the same structure.
    body = build_report_view(
        report.get("report_text", ""),
        symbol=report.get("symbol", ""), decision=report.get("decision", ""),
        weight=report.get("confidence"), trade_date=report.get("trade_date", ""),
        model_name=report.get("model_name", ""),
        generated=prog.format_stamp(report.get("created_at")),
    )
    footer = html.A(
        [html.I(className="bi bi-file-earmark-pdf me-1"), "Download PDF"],
        href=f"/api/download/ta-report/{report.get('id', '')}",
        className="btn btn-sm btn-outline-info",
        title="The same report as a PDF, formatted like this page",
    )
    return True, title, body, footer


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
    and min_agree are edited in the Run dialog, not here, carry them
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
            # Not the trigger, keep slider value as-is
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
    """Say what the ensemble will actually combine, before the run.

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
    note = (f": {', '.join(MODEL_DISPLAY.get(m, m) for m in skipped)} "
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
    State("sj-modal", "is_open", allow_optional=True),
    State("scheduler-fp", "data", allow_optional=True),
)
def render_scheduler_panel(_n, _action, modal_open, last_fp):
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
    if timer_fired and modal_open:
        # A redraw while the Schedule modal is open would be harmless to the
        # modal itself (it lives outside the panel) but is pointless.
        raise PreventUpdate

    try:
        jobs = scheduler_service.list_jobs()
        runs = scheduler_service.recent_runs(limit=8)
        # A timer tick redraws only when the database changed. Redrawing
        # every 15s reset every job card's inputs, so an edit typed into an
        # existing card was silently replaced by the stored values before
        # Save read them, and Save then confirmed a change that never
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


# --- Schedule modal: one dialog to create, edit or delete a job -------------
# The value outputs below are, in order, every control the modal renders;
# _SJ_FIELDS names them once so open/save cannot drift apart.
_SJ_SCALARS = [
    ("sj-kind", "value"), ("sj-name", "value"), ("sj-hour", "value"),
    ("sj-minute", "value"), ("sj-days", "value"), ("sj-tz", "value"),
    ("sj-visibility", "value"), ("sj-enabled", "value"), ("sj-symbols", "value"),
    ("sj-lookback", "value"), ("sj-max-articles", "value"),
    ("sj-ensemble-check", "value"), ("sj-ensemble-method", "value"),
    ("sj-ensemble-min-agree", "value"), ("sj-model", "value"), ("sj-type", "value"),
    ("sj-evidence", "value"), ("sj-tools", "value"), ("sj-recs", "value"),
    ("sj-recs-model", "value"),
]


def _sj_values_from_params(kind: str, params: dict, model_ids, member_ids,
                           param_ids) -> dict:
    """Control values for a job's params (defaults filled for missing keys)."""
    from services.scheduler_service import default_run_params
    p = {**default_run_params(), **(params or {})} if kind == "analysis" else (params or {})
    ens = p.get("ensemble") or {}
    weights = dict(ens.get("weights") or {})
    return {
        "sj-lookback": str(p.get("lookback", "")),
        "sj-max-articles": p.get("max_articles"),
        "models": [m in set(p.get("models") or model_ids) for m in model_ids],
        "sj-ensemble-check": bool(p.get("run_ensemble", True)),
        "sj-ensemble-method": ens.get("method"),
        "sj-ensemble-min-agree": ens.get("min_agree"),
        "members": [m in set(ens.get("enabled_models") or member_ids) for m in member_ids],
        "weights": [weights.get(m, 1.0) for m in member_ids],
        "sj-model": p.get("report_model"),
        "sj-type": p.get("depth"),
        "sj-evidence": list(p.get("evidence") or []),
        "sj-tools": list(p.get("tools") or []),
        "sj-recs": p.get("recs"),
        "sj-recs-model": p.get("recs_model"),
        "params": [(params or {}).get(pid["key"]) if pid["kind"] == kind else dash.no_update
                   for pid in param_ids],
    }


@callback(
    Output("sj-modal", "is_open"),
    Output("sj-job-id", "data"),
    Output("sj-title", "children"),
    Output("sj-delete", "style"),
    *[Output(cid, prop) for cid, prop in _SJ_SCALARS],
    Output({"type": "sj-model-check", "model": ALL}, "value"),
    Output({"type": "sj-ens-member", "model": ALL}, "value"),
    Output({"type": "sj-ens-weight", "model": ALL}, "value"),
    Output({"type": "sj-param", "kind": ALL, "key": ALL}, "value"),
    Input("sched-new-open", "n_clicks"),
    Input({"type": "sched-edit", "job": ALL}, "n_clicks"),
    Input("sj-cancel", "n_clicks"),
    State({"type": "sj-model-check", "model": ALL}, "id"),
    State({"type": "sj-ens-member", "model": ALL}, "id"),
    State({"type": "sj-param", "kind": ALL, "key": ALL}, "id"),
    prevent_initial_call=True,
)
def open_schedule_modal(new_click, edit_clicks, cancel_click, model_ids, member_ids,
                        param_ids):
    """Open the modal seeded from a job (edit) or from the defaults (create)."""
    from services import scheduler_service

    trig = ctx.triggered_id
    n_scalars = len(_SJ_SCALARS)
    n_models, n_members, n_params = len(model_ids), len(member_ids), len(param_ids)
    if trig == "sj-cancel" or (trig == "sched-new-open" and not new_click) or (
            isinstance(trig, dict) and not any(c for c in edit_clicks if c)):
        if trig == "sj-cancel":
            return (False, dash.no_update, dash.no_update, dash.no_update,
                    *([dash.no_update] * n_scalars),
                    [dash.no_update] * n_models, [dash.no_update] * n_members,
                    [dash.no_update] * n_members, [dash.no_update] * n_params)
        raise PreventUpdate

    model_names = [i["model"] for i in model_ids]
    member_names = [i["model"] for i in member_ids]
    if isinstance(trig, dict):
        job = next((j for j in scheduler_service.list_jobs() if j["id"] == trig["job"]), None)
        if job is None:
            raise PreventUpdate
        kind, job_id = job["kind"], job["id"]
        title = f"Edit schedule — {job.get('description') or job_id}"
        delete_style = {}
        scalars = {
            "sj-kind": kind, "sj-name": job.get("description") or "",
            "sj-hour": job["hour"], "sj-minute": job["minute"],
            "sj-days": job["days_of_week"], "sj-tz": job["timezone"],
            "sj-visibility": "public" if job.get("is_public", True) else "private",
            "sj-enabled": bool(job["enabled"]), "sj-symbols": job.get("symbols_csv") or "",
        }
        vals = _sj_values_from_params(kind, job.get("params") or {}, model_names,
                                      member_names, param_ids)
    else:
        types = scheduler_service.list_job_types()
        kind = types[0]["kind"] if types else "analysis"
        jt = next((t for t in types if t["kind"] == kind), None)
        job_id, title, delete_style = None, "New schedule", {"display": "none"}
        scalars = {
            "sj-kind": kind, "sj-name": "",
            "sj-hour": jt["default_hour"] if jt else 7,
            "sj-minute": jt["default_minute"] if jt else 0,
            "sj-days": "mon-fri", "sj-tz": "US/Eastern", "sj-visibility": "private",
            "sj-enabled": True, "sj-symbols": "",
        }
        vals = _sj_values_from_params(kind, {}, model_names, member_names, param_ids)
    scalars.update({k: v for k, v in vals.items() if k.startswith("sj-")})
    return (True, job_id, title, delete_style,
            *[scalars.get(cid) for cid, _ in _SJ_SCALARS],
            vals["models"], vals["members"], vals["weights"], vals["params"])


@callback(
    Output("sj-analysis-section", "style"),
    Output("sj-symbols-wrap", "style"),
    Output({"type": "sj-params-group", "kind": ALL}, "style"),
    Output("sj-hour", "value", allow_duplicate=True),
    Output("sj-minute", "value", allow_duplicate=True),
    Input("sj-kind", "value"),
    State({"type": "sj-params-group", "kind": ALL}, "id"),
    State("sj-job-id", "data"),
    prevent_initial_call=True,
)
def schedule_modal_kind(kind, group_ids, job_id):
    """Show the controls the chosen operation uses; seed the time from the
    type's default on a NEW job only (never clobber an edit)."""
    from services import scheduler_service

    types = {t["kind"]: t for t in scheduler_service.list_job_types()}
    jt = types.get(kind)
    is_analysis = kind == "analysis"
    takes_symbols = bool(jt and jt["needs_symbols"])
    hour = minute = dash.no_update
    if job_id is None and jt:
        hour, minute = jt["default_hour"], jt["default_minute"]
    return (
        {} if is_analysis else {"display": "none"},
        {} if takes_symbols else {"display": "none"},
        [{"display": "flex" if g.get("kind") == kind else "none"} for g in (group_ids or [])],
        hour, minute,
    )


@callback(
    Output("sj-ensemble-method-hint", "children"),
    Output("sj-ensemble-min-agree-wrap", "style"),
    Output({"type": "sj-ens-weight", "model": ALL}, "disabled"),
    Output("sj-ensemble-body", "style"),
    Input("sj-ensemble-method", "value"),
    Input("sj-ensemble-min-agree", "value"),
    Input("sj-ensemble-check", "value"),
)
def schedule_modal_ensemble_ui(method, min_agree, ens_on):
    """Same explanation and gating the Run dialog shows for its ensemble."""
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
    body = html.Div([html.Div(hints.get(method, ""), className="run-field-hint"),
                     ensemble_method_detail(method, gate)])
    return (body, {} if is_agreement else {"display": "none"},
            [is_agreement] * n_weights, {} if ens_on else {"display": "none"})


@callback(
    Output("scheduler-action-status", "data", allow_duplicate=True),
    Output("sj-modal", "is_open", allow_duplicate=True),
    Output("sj-feedback", "children"),
    Input("sj-save", "n_clicks"),
    State("sj-job-id", "data"),
    *[State(cid, prop) for cid, prop in _SJ_SCALARS],
    State({"type": "sj-model-check", "model": ALL}, "value"),
    State({"type": "sj-model-check", "model": ALL}, "id"),
    State({"type": "sj-ens-member", "model": ALL}, "value"),
    State({"type": "sj-ens-weight", "model": ALL}, "value"),
    State({"type": "sj-ens-member", "model": ALL}, "id"),
    State({"type": "sj-param", "kind": ALL, "key": ALL}, "value"),
    State({"type": "sj-param", "kind": ALL, "key": ALL}, "id"),
    prevent_initial_call=True,
)
def save_schedule_job(n_clicks, job_id, *rest):
    """Create or update a job from the modal — schedule AND run settings."""
    if not n_clicks:
        raise PreventUpdate
    from services import scheduler_service
    from services.news_window import (
        RunParameterMissing, normalize_article_cap, normalize_lookback)

    n = len(_SJ_SCALARS)
    scalars = dict(zip([cid for cid, _ in _SJ_SCALARS], rest[:n]))
    (model_vals, model_ids, member_vals, weight_vals, member_ids,
     param_vals, param_ids) = rest[n:]

    def err(msg):
        return dash.no_update, dash.no_update, html.Span(msg, className="text-danger")

    kind = scalars["sj-kind"]
    if not kind:
        return err("Pick an operation")
    try:
        hour = max(0, min(23, int(scalars["sj-hour"])))
        minute = max(0, min(59, int(scalars["sj-minute"])))
    except (TypeError, ValueError):
        return err("Time must be a number")
    needs = {t["kind"]: t["needs_symbols"] for t in scheduler_service.list_job_types()}
    symbols_csv = ",".join(
        x.strip().upper() for x in str(scalars["sj-symbols"] or "").replace("\n", ",").split(",")
        if x.strip())
    if needs.get(kind) and not symbols_csv:
        return err("This operation needs at least one symbol")

    if kind == "analysis":
        try:
            overnight, days = normalize_lookback(scalars["sj-lookback"])
            cap = normalize_article_cap(scalars["sj-max-articles"])
        except RunParameterMissing as e:
            return err(str(e))
        models = [i["model"] for i, on in zip(model_ids, model_vals or []) if on]
        if not models:
            return err("Pick at least one model")
        weights = {}
        for i, w in zip(member_ids, weight_vals or []):
            try:
                weights[i["model"]] = round(float(w), 1)
            except (TypeError, ValueError):
                weights[i["model"]] = 1.0
        try:
            min_agree = int(scalars["sj-ensemble-min-agree"])
        except (TypeError, ValueError):
            return err("Min. agreeing models must be a number")
        params = {
            "only_trading_days": True,
            "lookback": "overnight" if overnight else days,
            "max_articles": cap,
            "models": models,
            "run_ensemble": bool(scalars["sj-ensemble-check"]),
            "ensemble": {
                "method": scalars["sj-ensemble-method"],
                "min_agree": min_agree,
                "enabled_models": [i["model"] for i, on in zip(member_ids, member_vals or []) if on],
                "weights": weights,
            },
            "report_model": scalars["sj-model"],
            "depth": scalars["sj-type"],
            "recs": scalars["sj-recs"],
            "recs_model": scalars["sj-recs-model"],
            "evidence": sorted(scalars["sj-evidence"] or []),
            "tools": sorted(scalars["sj-tools"] or []),
        }
    else:
        params = {"only_trading_days": True} if needs.get(kind) else {}
        for pid, val in zip(param_ids or [], param_vals or []):
            if pid.get("kind") == kind and val is not None:
                params[pid["key"]] = val

    fields = {
        "description": (scalars["sj-name"] or "").strip(),
        "hour": hour, "minute": minute,
        "days_of_week": scalars["sj-days"], "timezone": scalars["sj-tz"],
        "is_public": scalars["sj-visibility"] != "private",
        "enabled": bool(scalars["sj-enabled"]),
        "symbols_csv": symbols_csv or None,
        "params_json": params,
    }
    if job_id:
        if not scheduler_service.update_job(job_id, **fields):
            return err("Save failed")
        return ({"updated": job_id, "at": datetime.now().isoformat()}, False, None)
    new_id = scheduler_service.create_job(
        kind=kind, description=fields["description"], hour=hour, minute=minute,
        days_of_week=fields["days_of_week"], timezone=fields["timezone"],
        symbols_csv=fields["symbols_csv"], params=params,
        is_public=fields["is_public"])
    if not new_id:
        return err("Could not create the job")
    if not fields["enabled"]:
        scheduler_service.update_job(new_id, enabled=False)
    return ({"created": new_id, "at": datetime.now().isoformat()}, False, None)


@callback(
    Output("scheduler-action-status", "data", allow_duplicate=True),
    Output("sj-modal", "is_open", allow_duplicate=True),
    Input("sj-delete", "n_clicks"),
    State("sj-job-id", "data"),
    prevent_initial_call=True,
)
def delete_schedule_job_from_modal(n_clicks, job_id):
    if not n_clicks or not job_id:
        raise PreventUpdate
    from services import scheduler_service
    scheduler_service.delete_job(job_id)
    return {"deleted": job_id, "at": datetime.now().isoformat()}, False


@callback(
    Output("scheduler-action-status", "data", allow_duplicate=True),
    Input({"type": "sched-delete", "job": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def delete_scheduled_job(clicks):
    """Remove a job. Its run history stays, that is the record of what ran."""
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
# guard was Werkzeug-only, under uvicorn it silently skipped the scheduler
# when DEBUG=true, and the background-callback spawn child started a duplicate
# scheduler when DEBUG=false.
def _startup():
    """One-time server-process startup: progress hydrate, S3, auth, scheduler."""
    _progress_service.hydrate_from_db()
    # Every run this server (or the one it replaced) had in flight died with
    # it, and a queued/running row is its owner's lock on the Run dialog.
    from services import run_service
    reaped = _progress_service.reap_orphans(
        max_age_s=0, error=run_service.ORPHAN_RESTART_ERROR)
    if reaped:
        logger.warning("Failed %d run(s) left in flight by the previous "
                       "server process", len(reaped))
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
    Output("run-scope", "value", allow_duplicate=True),
    Output({"type": "run-model-check", "model": ALL}, "value"),
    Output("run-recs", "value"),
    Output("run-evidence", "value"),
    Output("run-tools", "value"),
    Input("run-preset", "value"),
    Input("run-date-picker", "date"),
    State({"type": "run-model-check", "model": ALL}, "id"),
    prevent_initial_call=True,
)
def apply_run_preset(preset, run_date, check_ids):
    """Write the chosen preset into the controls under Customize.

    Tools follow the target session as well as the preset: web research is
    never on for a backtest date (the open web would leak the future), and
    a preset that fixes no tools takes the date default. A date change
    alone touches only the tools, so a customised dialog keeps its values
    when the user moves the target; a preset change rewrites every field
    the preset names and leaves the rest alone.
    """
    from layouts.modals import preset_fields, preset_run_tools
    from utils.trading_calendar import resolve_target_and_cutoff

    picked = str(run_date)[:10] if run_date else None
    try:
        target_d, _ = resolve_target_and_cutoff(picked)
    except Exception as e:
        logger.warning("Run dialog: target date unresolved (%s): %s", picked, e)
        target_d = None
    tools = preset_run_tools(preset, target_d)
    # A pattern-matching (ALL) output must always get a list, one entry per
    # matched checkbox; a bare no_update there is a 500.
    keep_checks = [dash.no_update] * len(check_ids or [])
    fired = {t["prop_id"].split(".")[0] for t in (ctx.triggered or [])}
    if "run-preset" not in fired:
        return dash.no_update, keep_checks, dash.no_update, dash.no_update, tools

    fields = preset_fields(preset)
    models = fields.get("models")
    checks = ([(c or {}).get("model") in models for c in (check_ids or [])]
              if models is not None else keep_checks)
    return (
        fields.get("scope", dash.no_update),
        checks,
        fields.get("recs", dash.no_update),
        fields.get("evidence", dash.no_update),
        tools,
    )


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
    generation: the same point-in-time fetch generation uses, so the
    preview counts are the counts, not an estimate from the news store.

    Async: fires on modal open, so the live vendor fetch runs in a worker
    thread instead of freezing the UI thread pool."""
    symbols = (run_symbols or {}).get("symbols") or []
    if not is_open:
        raise PreventUpdate
    if not symbols:
        # Removing the last chip must clear the counts too. PreventUpdate
        # here left the previous symbol's table on screen.
        return html.Div("No symbols in this run. Add one to preview "
                        "article availability.",
                        className="run-symbols-empty")

    from config import MODEL as _M
    from utils.trading_calendar import resolve_target_and_cutoff
    from services.news_window import (
        RunParameterMissing, fetch_run_news, normalize_article_cap,
        normalize_lookback)

    # Same target/cutoff resolution as generation: the window ends at the
    # cutoff (previous trading day), not the target, previewing a window
    # ending at the target overstated what the report would actually see.
    picked = str(ai_date)[:10] if ai_date else None
    target_d, as_of_d = resolve_target_and_cutoff(picked)
    as_of, target = as_of_d.isoformat(), target_d.isoformat()
    try:
        overnight, lookback_days = normalize_lookback(lookback)
        max_articles = normalize_article_cap(max_articles_val)
    except RunParameterMissing as e:
        return html.Div(f"Cannot preview: {e}", className="text-danger")

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
                if st.get("oldest") and st.get("newest") else "n/a")
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
            f"Cap drops the oldest articles for {', '.join(capped)}: raise "
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
            f"Live: targeting {target} close, data through {cutoff}",
        ], {"fontSize": "0.85rem", "color": "var(--positive)"}
    else:
        days_back = (today - selected).days
        return [
            html.I(className="bi bi-clock-history me-1"),
            f"Backtest ({days_back}d ago): targeting {target} close, "
            f"data truncated to {cutoff}",
        ], {"fontSize": "0.85rem", "color": "var(--warning, #ffc107)"}


def _run_data_summary(symbols, stock_data, news_data):
    """The dialog's per-symbol input summary for the resolved run set.

    A symbol outside the browser stores shows "at run time", its data is
    fetched server-side when the run starts (see _fill_run_inputs), so an
    empty row here is a statement of when, not a problem.
    """
    if not symbols:
        return html.Div(
            [
                html.Div("No symbols selected", className="empty-state-title"),
                html.Div("Search for a symbol above, or add the watchlist "
                         "or your last run.",
                         className="empty-state-note"),
            ],
            className="empty-state",
        )
    stock_data = stock_data or {}
    articles_by_symbol = (news_data or {}).get("articles_by_symbol", {})
    sym_rows = []
    for sym in symbols:
        sym_data = stock_data.get(sym, {})
        data_points = ", "
        date_range = ", "
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
            html.Td(str(news_count) if sym in articles_by_symbol else "n/a"),
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


def _last_run_symbols(owner_uid, prefs=None):
    """The symbols of this owner's newest manual run, for "+ Last run".

    Read from analysis_runs, not the Home cohort: the cohort is whatever
    the latest cutoff holds (any owner, the scheduled job included), which
    is not what "run the same names again" means. With no run on record
    the browser's remembered ad-hoc list stands in.
    """
    try:
        from services import run_service
        runs = run_service.list_runs(limit=1, kind="manual",
                                     owner_uid=owner_uid or "")
    except Exception as e:
        logger.warning("Run dialog: could not read the last run: %s", e)
        runs = []
    if runs and runs[0].get("symbols"):
        return list(runs[0]["symbols"])
    return [s for s in ((prefs or {}).get("symbols") or []) if s]


def _resolve_run_symbols(tokens, existing):
    """Split typed/picked tokens into (accepted, rejected) symbols.

    A name the cache knows is accepted on a primary-key read; an unknown
    one costs one price-info lookup (validate_symbol), after which it is
    cached as validated. Duplicates of the chips already in the run are
    dropped silently, not reported.
    """
    from services import ticker_service as ts

    accepted, rejected = [], []
    for raw in tokens:
        sym = ts.normalize_symbol(raw)
        if not sym:
            rejected.append((raw or "").strip().upper() or raw)
            continue
        if sym in existing or sym in accepted:
            continue
        try:
            verdict = ts.validate_symbol(sym)
        except Exception as e:
            logger.warning("Run dialog: could not validate %s: %s", sym, e)
            verdict = {"ok": False}
        (accepted if verdict.get("ok") else rejected).append(sym)
    return accepted, rejected


@callback(
    Output("run-modal", "is_open"),
    Output("run-scope", "value"),
    Output("run-symbols-store", "data"),
    Output("run-validation-msg", "children"),
    Output("progress-panel-state", "data", allow_duplicate=True),
    Output("run-started-toast", "is_open"),
    Output("run-started-toast", "children"),
    # Snap the panel's poll to the fast rate on confirm. Waiting for a slow
    # idle tick to notice the active flag cost ~an idle interval before the
    # first run events rendered.
    Output("progress-interval", "interval", allow_duplicate=True),
    # The picker's default/max/holidays are baked into the layout at process
    # start; a long-lived server would otherwise pin the dialog to launch day.
    Output("run-date-picker", "date"),
    Output("run-date-picker", "max_date_allowed"),
    Output("run-date-picker", "disabled_days"),
    # The run record, written once per accepted confirm: run-store keeps
    # it for the session (the panel pins to it), run-dispatch is the memory
    # copy the stage callbacks trigger on (see _dispatch_run_id).
    Output("run-store", "data"),
    Output("run-dispatch", "data"),
    # Preset and what the browser remembers of it; Customize folds on open.
    Output("run-preset", "value"),
    Output("run-prefs-store", "data"),
    Output("run-customize-collapse", "is_open"),
    # The clientside click handler swaps the confirm button for a spinner
    # before the round trip; every server return puts it back.
    Output("run-confirm-btn", "disabled", allow_duplicate=True),
    Output("run-confirm-btn", "children", allow_duplicate=True),
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
    # The dialog's values, recorded on the run row as its config.
    State("run-date-picker", "date"),
    State("run-lookback", "value"),
    State("run-max-articles", "value"),
    State("run-model", "value"),
    State("run-type", "value"),
    State("run-recs", "value"),
    State("run-recs-model", "value"),
    State("run-evidence", "value"),
    State("run-tools", "value"),
    State({"type": "run-model-check", "model": ALL}, "value"),
    State("run-ensemble-check", "value"),
    State("run-ensemble-method", "value"),
    State("run-ensemble-min-agree", "value"),
    State("run-preset", "value"),
    State("run-prefs-store", "data"),
    # Ensemble members and weights, so the row's config is the WHOLE run:
    # the stages read nothing else. The shared store stands in when the
    # controls have not rendered (it is what they would show).
    State({"type": "run-ens-member", "model": ALL}, "value"),
    State({"type": "run-ens-weight", "model": ALL}, "value"),
    State("ensemble-config-store", "data"),
    prevent_initial_call=True,
)
def toggle_run_modal(open_clicks, reports_clicks, ctx_clicks, cancel_clicks,
                     confirm_clicks, stock_data, watchlist,
                     news_data, is_open, run_store, run_scope, panel_state,
                     run_date=None, lookback=None, max_articles=None,
                     report_model=None, depth=None, recs=None,
                     recs_model=None, run_evidence=None, run_tools=None,
                     model_checks=None, run_ensemble=None, ens_method=None,
                     ens_min_agree=None, preset=None, prefs=None,
                     ens_members=None, ens_weights=None, ensemble_store=None):
    """Open or close the single run dialog, preset by where it was opened.

    Two kinds of opener, deliberately: the toolbar button (the ONE global
    entry point: it opens EMPTY, on the preset this browser last confirmed
    with, and "+ Watchlist" / "+ Last run" add names on top) and contextual
    shortcuts that carry their context in: the Reports page's New Report
    (scope=report, watchlist) and any per-symbol New-report button
    (scope=report, just that symbol). Those two force the scope and leave
    the preset alone, so the dialog opens diverging from it with Customize
    unfolded, which is an honest picture of what will run.

    Confirm is the run dispatcher: one run per owner at a time (a second
    confirm is refused with a Cancel button while one is in flight), then
    the analysis_runs row is created, with the dialog's every value as its
    config (the stages read the row, never the dialog, so what ran is what
    the row says), and its id written to run-store and run-dispatch, which
    is what starts the stages. The immediate acknowledgement (panel open,
    toast, fast poll) is _run_acknowledgement, shared with retry. An empty
    symbol set keeps the dialog open with an inline message instead of
    silently no-opping, every downstream stage guards on the list being
    non-empty.
    """
    from layouts.modals import (DEFAULT_RUN_PRESET, RUN_PRESETS,
                                run_confirm_label)

    triggered = ctx.triggered_id
    # panel state + toast open/children + poll rate
    no_feedback = (dash.no_update,) * 4
    # date-picker date + max_date_allowed + disabled_days
    no_picker = (dash.no_update,) * 3
    # run-store + run-dispatch
    no_run = (dash.no_update,) * 2
    # preset value + prefs store + customize collapse
    no_preset = (dash.no_update,) * 3
    button_ready = (False, run_confirm_label())

    if triggered == "run-cancel-btn":
        return (False, dash.no_update, dash.no_update, "") + no_feedback \
            + no_picker + no_run + no_preset + button_ready

    if triggered == "run-confirm-btn":
        symbols = (run_store or {}).get("symbols") or []
        keep_open = (dash.no_update,) * 3

        def refuse(message):
            return keep_open + (message,) + no_feedback + no_picker + no_run \
                + no_preset + button_ready

        if not symbols:
            return refuse("Add at least one symbol to run.")
        scope = run_scope or "full"
        from services import progress_service as prog
        from services import run_service

        owner_uid = _run_owner_uid()
        try:
            refusal = _active_run_refusal()
        except Exception as e:
            logger.warning("Run dialog: could not check for an active run: %s", e)
            return refuse(f"Could not check for a run in progress: {str(e)[:120]}")
        if refusal is not None:
            return refuse(refusal)

        # The dialog's values as the row's config, and the estimate the
        # toast quotes; the same inputs the preflight line reads.
        from layouts.modals import estimate_run_seconds, preset_divergence
        from utils.trading_calendar import resolve_target_and_cutoff
        selected_models = [m for m, on in zip(_RUN_MODEL_IDS, model_checks or [])
                           if on]
        recs_on = scope == "full" and (recs or "auto") != "off"
        estimate_s = estimate_run_seconds(scope, len(symbols), selected_models,
                                          recs_on)
        picked = str(run_date)[:10] if run_date else None
        try:
            target_d, cutoff_d = resolve_target_and_cutoff(picked)
        except Exception as e:
            logger.warning("Run dialog: target date unresolved (%s): %s",
                           picked, e)
            target_d = cutoff_d = None
        # The row's preset is the name only while the controls still match
        # it; anything the user touched makes it "custom", so a later
        # median-duration estimate keyed on preset never mixes the two.
        preset_name = preset if preset in RUN_PRESETS else DEFAULT_RUN_PRESET
        # The raw values, as run_preflight passes them: a None (unmounted)
        # control is not a divergence there, and a recs defaulted to
        # "auto" here would call an untouched Standard run "custom".
        customized = preset_divergence(preset_name, {
            "scope": scope,
            "models": selected_models if model_checks else None,
            "recs": recs,
            "evidence": run_evidence,
            "tools": run_tools,
        })
        row_preset = preset_name if not customized else "custom"
        if ens_members is not None:
            ensemble_members = [m for m, on in zip(_RUN_MODEL_IDS, ens_members)
                                if on]
            ensemble_weights = {}
            for model_id, w in zip(_RUN_MODEL_IDS, ens_weights or []):
                try:
                    ensemble_weights[model_id] = round(float(w), 1)
                except (TypeError, ValueError):
                    ensemble_weights[model_id] = 1.0
        else:
            ensemble_members = list((ensemble_store or {}).get("enabled_models")
                                    or MODEL.ENSEMBLE_DEFAULT_ENABLED)
            ensemble_weights = dict((ensemble_store or {}).get("weights")
                                    or MODEL.ENSEMBLE_DEFAULT_WEIGHTS)
        try:
            min_agree_val = int(ens_min_agree)
        except (TypeError, ValueError):
            min_agree_val = MODEL.ENSEMBLE_MIN_AGREE
        # The run, complete: every stage reads its settings from here and
        # nowhere else, and a retry copies it verbatim.
        config = {
            "scope": scope,
            "preset": preset_name,
            "customized": customized,
            # The resolved sessions (the picker's day snaps forward to a
            # trading day); the raw pick is kept for the record.
            "target_date": target_d.isoformat() if target_d else picked,
            "prediction_date": cutoff_d.isoformat() if cutoff_d else None,
            "picked_date": picked,
            "lookback": lookback,
            "max_articles": max_articles,
            "report_model": report_model,
            "depth": depth,
            "recs": recs,
            "recs_model": recs_model,
            "evidence": sorted(run_evidence) if run_evidence is not None else None,
            "tools": sorted(set(run_tools or [])),
            "models": selected_models,
            "ensemble": bool(run_ensemble),
            "ensemble_members": ensemble_members,
            "ensemble_weights": ensemble_weights,
            "ensemble_method": ens_method or MODEL.ENSEMBLE_DEFAULT_METHOD,
            "ensemble_min_agree": min_agree_val,
        }
        try:
            run_id = run_service.create_run(
                kind="manual", symbols=symbols, owner_uid=owner_uid,
                preset=row_preset, config=config,
                prediction_date=cutoff_d, target_date=target_d,
                estimate_s=int(round(estimate_s)) if estimate_s else None)
        except Exception as e:
            # No row, no run: the stages key everything on the id.
            logger.error("Run dialog: run row not created: %s", e)
            return refuse(f"Could not record the run: {str(e)[:120]}")
        # Mark the feed active NOW: the stage that calls start_run is a
        # separate callback, and the fast tick below must not see an idle
        # feed and drop straight back to the slow rate.
        prog.mark_run_pending(run_id)
        # Every symbol that runs joins the lookup cache, so the typeahead
        # knows it next time. Never worth failing a run over.
        try:
            from services import ticker_service
            ticker_service.ensure_symbols(symbols)
        except Exception as e:
            logger.warning("Run dialog: symbol cache not updated: %s", e)

        run_data = {
            "run_id": run_id,
            "started": datetime.now(timezone.utc).isoformat(),
            "scope": scope,
            "preset": row_preset,
            "symbols": list(symbols),
            "owner_uid": owner_uid,
            "kind": "manual",
        }
        prefs_out = {"preset": preset_name, "symbols": list(symbols)}
        return (False, dash.no_update, dash.no_update, "") \
            + _run_acknowledgement(panel_state, symbols, estimate_s, scope) \
            + no_picker + (run_data, run_data) \
            + (dash.no_update, prefs_out, dash.no_update) + button_ready

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

    prefs = prefs if isinstance(prefs, dict) else {}
    watchlist = [s for s in (watchlist or []) if s]
    lastrun = _last_run_symbols(_run_owner_uid(), prefs)

    scope = dash.no_update
    preset_out = dash.no_update
    if is_context_btn:
        scope = "report"
        run_symbols = [triggered.get("symbol")]
    elif triggered == "reports-new-btn":
        scope = "report"
        run_symbols = watchlist or lastrun
    else:
        run_symbols = []
        remembered = prefs.get("preset")
        preset_out = (remembered if remembered in RUN_PRESETS
                      else DEFAULT_RUN_PRESET)

    store = {
        "symbols": run_symbols,
        "watchlist": watchlist,
        "lastrun": lastrun,
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
    return (True, scope, store, "") + no_feedback + picker + no_run \
        + (preset_out, dash.no_update, False) + button_ready


@callback(
    Output("run-validation-msg", "children", allow_duplicate=True),
    # A cancelled full run leaves the synthesis flag armed: its killed
    # model stage returns nothing and its report lands on a run_seq the
    # synthesis callback no longer pairs, so neither of the two writers
    # that disarm it ever fires. The next models-only run would then hand
    # off to a synthesis that never comes and never close its row.
    Output("full-analysis-requested", "data", allow_duplicate=True),
    # The refused confirm lives in the still-open dialog, so the button is
    # only mounted while a run is in flight for this owner.
    Input("run-cancel-active-btn", "n_clicks", allow_optional=True),
    prevent_initial_call=True,
)
def cancel_active_run(n_clicks):
    """Cancel this owner's run in flight so a new confirm can go through
    (the dialog's button; the topbar pill's cancel shares the body)."""
    if not n_clicks:
        raise PreventUpdate
    message, run_id = _cancel_own_run()
    return message, (False if run_id else dash.no_update)


def _cancel_own_run() -> tuple:
    """Cancel this owner's manual run in flight; ``(message, run_id)``,
    run_id None when there was nothing to cancel.

    The worker pid the model stage recorded is terminated; the row is
    marked cancelled (sticky, so a stage finishing late cannot flip it to
    done) and its feed closed. A stage still running in the server process
    (the report) finishes on its own; its late close is ignored by the row.
    Manual runs only: a scheduled run belongs to the scheduler, never locks
    this owner, and cannot be cancelled from the UI.
    """
    from services import progress_service as prog
    from services import run_service

    active = run_service.active_run_for(_run_owner_uid())
    if active is None:
        return "No run in progress. You can start one.", None
    run_id = active["run_id"]
    pid = prog.terminate_run_worker(run_id)
    run_service.cancel_run(run_id)
    prog.emit("action", "Run cancelled by user"
              + (f" (worker pid {pid} terminated)" if pid else ""),
              run_id=run_id)
    prog.finish_run("Cancelled", run_id=run_id)
    return "Run cancelled. You can start another.", run_id


@callback(
    Output("run-store", "data", allow_duplicate=True),
    Output("run-dispatch", "data", allow_duplicate=True),
    Output("run-modal", "is_open", allow_duplicate=True),
    Output("run-validation-msg", "children", allow_duplicate=True),
    # The same acknowledgement a confirm gives (_run_acknowledgement).
    Output("progress-panel-state", "data", allow_duplicate=True),
    Output("run-started-toast", "is_open", allow_duplicate=True),
    Output("run-started-toast", "children", allow_duplicate=True),
    Output("progress-interval", "interval", allow_duplicate=True),
    # The button lives in the Analyze page's failure block, so it exists
    # only there and only after a report failed.
    Input("ai-retry-btn", "n_clicks", allow_optional=True),
    State("run-store", "data"),
    State("progress-panel-state", "data"),
    prevent_initial_call=True,
)
def retry_run(n_clicks, run_store, panel_state=None):
    """Retry the run this session last started, as a new run.

    The old retry re-ran the report stage under the previous run's id,
    which walked past the per-owner lock and reopened a row that had
    already closed. A retry is a confirm with the previous run's config:
    it is refused while this owner has a run in flight (the dialog opens
    to show why, with the Cancel button), otherwise a fresh manual row is
    created from the previous row's config and handed to the same stores
    the confirm writes, so the normal dispatch path runs, and the same
    acknowledgement the confirm gives (panel open, started toast, fast
    poll) follows. The dict is marked retry_of so the report stage skips
    its persistent cache.
    """
    if not n_clicks:
        raise PreventUpdate
    prev_id = (run_store or {}).get("run_id")
    if not prev_id:
        raise PreventUpdate
    from services import progress_service as prog
    from services import run_service

    no_feedback = (dash.no_update,) * 4

    def refuse(message):
        return (dash.no_update, dash.no_update, True, message) + no_feedback

    try:
        refusal = _active_run_refusal()
    except Exception as e:
        logger.warning("Retry: could not check for an active run: %s", e)
        return refuse(f"Could not check for a run in progress: {str(e)[:120]}")
    if refusal is not None:
        return refuse(refusal)
    prev = run_service.get_run(prev_id)
    if prev is None:
        logger.warning("Retry: run %s has no row to retry from", prev_id[:8])
        return refuse("The last run's record is gone. Start it again from here.")
    if prev.get("kind") != "manual":
        # run-store can point at a scheduled run after a click on its pill;
        # its config is the scheduler's vocabulary, not the dialog's.
        return refuse("That was a scheduled run. Start yours from Run analysis.")
    owner_uid = _run_owner_uid()
    try:
        run_id = run_service.create_run(
            kind="manual", symbols=prev["symbols"], owner_uid=owner_uid,
            preset=prev["preset"], config=prev["config"],
            prediction_date=prev["prediction_date"],
            target_date=prev["target_date"], estimate_s=prev["estimate_s"])
    except Exception as e:
        logger.error("Retry: run row not created: %s", e)
        return refuse(f"Could not record the run: {str(e)[:120]}")
    prog.mark_run_pending(run_id)
    scope = _run_scope_of(prev, run_store)
    run_data = {
        "run_id": run_id,
        "started": datetime.now(timezone.utc).isoformat(),
        "scope": scope,
        "preset": prev["preset"],
        "symbols": list(prev["symbols"]),
        "owner_uid": owner_uid,
        "kind": "manual",
        "retry_of": prev_id,
    }
    return (run_data, run_data, dash.no_update, "") \
        + _run_acknowledgement(panel_state, prev["symbols"],
                               prev["estimate_s"], scope)


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
    context-block list reflecting the Evidence checkboxes AS SELECTED NOW.
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
    Output("run-symbol-search", "options"),
    Input("run-symbol-search", "search_value"),
    prevent_initial_call=True,
)
async def search_run_symbols(search_value):
    """Typeahead options from the local ticker cache, one indexed query
    per keystroke and no vendor call. The Dropdown clears search_value on
    select, which empties the list again."""
    from layouts.modals import (is_symbol_list, run_symbol_options,
                                split_symbol_input)

    query = (search_value or "").strip()
    if not split_symbol_input(query):
        # The Dropdown clears search_value the moment an option is picked;
        # emptying the list in that same round made the picked value
        # invalid before its own callback ran. The last list stays.
        raise PreventUpdate
    if is_symbol_list(query):
        # A paste: one option that adds them all; nothing to look up.
        return run_symbol_options(query, [])
    from services import ticker_service
    try:
        hits = await asyncio.to_thread(ticker_service.search, query, 12)
    except Exception as e:
        logger.warning("Run dialog: symbol search failed: %s", e)
        hits = []
    return run_symbol_options(query, hits)


@callback(
    Output("run-symbols-store", "data", allow_duplicate=True),
    Output("run-symbol-search", "value"),
    Output("run-validation-msg", "children", allow_duplicate=True),
    Input("run-add-watchlist", "n_clicks"),
    Input("run-add-lastrun", "n_clicks"),
    Input({"type": "run-sym-remove", "symbol": ALL}, "n_clicks"),
    Input("run-symbol-search", "value"),
    State("run-symbols-store", "data"),
    prevent_initial_call=True,
)
async def set_run_symbols(add_watchlist, add_lastrun, remove_clicks, picked,
                          store):
    """Edit the run's symbol set without touching the watchlist.

    Every source is additive: "+ Watchlist" and "+ Last run" append what is
    not there yet, a typeahead pick (or a pasted list) appends after the
    cache or one price lookup vouches for it, and a chip's cross removes
    it. The store's watchlist/lastrun snapshots were taken when the dialog
    opened, which is the set the user was looking at.
    """
    from layouts.modals import split_symbol_input

    store = dict(store or {})
    symbols = list(store.get("symbols") or [])
    triggered = ctx.triggered_id

    if triggered in ("run-add-watchlist", "run-add-lastrun"):
        clicks = add_watchlist if triggered == "run-add-watchlist" else add_lastrun
        if not clicks:
            raise PreventUpdate
        key = "watchlist" if triggered == "run-add-watchlist" else "lastrun"
        extra = [s for s in (store.get(key) or []) if s and s not in symbols]
        if not extra:
            raise PreventUpdate
        store["symbols"] = symbols + extra
        return store, dash.no_update, ""

    if isinstance(triggered, dict) and triggered.get("type") == "run-sym-remove":
        if not any(c and c > 0 for c in remove_clicks):
            raise PreventUpdate
        sym = triggered.get("symbol")
        store["symbols"] = [s for s in symbols if s != sym]
        return store, dash.no_update, ""

    if triggered == "run-symbol-search":
        tokens = split_symbol_input(picked)
        if not tokens:
            raise PreventUpdate  # the echo of the clear below
        accepted, rejected = await asyncio.to_thread(
            _resolve_run_symbols, tokens, symbols)
        msg = ""
        if rejected:
            msg = f"No price data for {', '.join(rejected)}"
        if not accepted:
            return dash.no_update, None, msg
        store["symbols"] = symbols + accepted
        return store, None, msg

    raise PreventUpdate


@callback(
    Output("run-symbols-chips", "children"),
    Output("run-add-watchlist", "children"),
    Output("run-add-watchlist", "disabled"),
    Output("run-add-lastrun", "children"),
    Output("run-add-lastrun", "disabled"),
    Input("run-symbols-store", "data"),
    prevent_initial_call=True,
)
def render_run_symbol_chips(store):
    """Chips for the effective run set, each removable for this run only,
    and the two add buttons with their live counts."""
    store = store or {}
    symbols = store.get("symbols") or []
    if not symbols:
        chips = [html.Span("No symbols yet: search below, or add the "
                           "watchlist or your last run.",
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
    watchlist = [s for s in (store.get("watchlist") or []) if s]
    lastrun = [s for s in (store.get("lastrun") or []) if s]
    return (
        chips,
        f"+ Watchlist ({len(watchlist)})", not watchlist,
        f"+ Last run ({len(lastrun)})", not lastrun,
    )


@callback(
    Output("run-data-summary", "children"),
    # Input, not State: the summary follows the chip set, the same way the
    # article preview does; it used to render once on open and go stale.
    Input("run-symbols-store", "data"),
    State("stock-data-store", "data"),
    State("news-data-store", "data"),
    prevent_initial_call=True,
)
def render_run_data_summary(store, stock_data, news_data):
    return _run_data_summary((store or {}).get("symbols") or [],
                             stock_data, news_data)


@callback(
    Output("run-customize-collapse", "is_open", allow_duplicate=True),
    Input("run-customize-btn", "n_clicks"),
    State("run-customize-collapse", "is_open"),
    prevent_initial_call=True,
)
def toggle_run_customize(n_clicks, is_open):
    if not n_clicks:
        raise PreventUpdate
    return not is_open


# The confirm button's own acknowledgement, before the round trip: the
# server's confirm branch takes a lock check, a row insert and a calendar
# resolve, long enough for a second click. Both server returns (started or
# refused) restore the label through toggle_run_modal's duplicate outputs.
clientside_callback(
    """
    function(n) {
        if (!n) {
            return [window.dash_clientside.no_update,
                    window.dash_clientside.no_update];
        }
        return [true, [
            {namespace: "dash_html_components", type: "Span",
             props: {className: "spinner-border spinner-border-sm me-1",
                     role: "status"}},
            "Starting\u2026"
        ]];
    }
    """,
    Output("run-confirm-btn", "disabled"),
    Output("run-confirm-btn", "children"),
    Input("run-confirm-btn", "n_clicks"),
    prevent_initial_call=True,
)

# Cursor in the symbol search when the dialog opens: the modal fades in,
# so the focus waits for the transition. Best effort: a missing element
# costs nothing.
clientside_callback(
    """
    function(open) {
        if (!open) { return ""; }
        setTimeout(function() {
            var root = document.getElementById("run-symbol-search");
            if (!root) { return; }
            var el = root.querySelector('[role="combobox"], input, button, [tabindex]') || root;
            try { el.focus(); } catch (e) {}
        }, 300);
        return "";
    }
    """,
    Output("run-focus-sink", "children"),
    Input("run-modal", "is_open"),
    prevent_initial_call=True,
)


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
    Output("run-preset-hint", "children"),
    Output("run-customize-collapse", "is_open", allow_duplicate=True),
    Output("run-customize-auto", "data"),
    Input("run-modal", "is_open"),
    Input("run-preset", "value"),
    Input("run-scope", "value"),
    Input("run-symbols-store", "data"),
    Input({"type": "run-model-check", "model": ALL}, "value"),
    Input("run-model", "value"),
    Input("run-recs", "value"),
    Input("run-recs-model", "value"),
    Input("run-evidence", "value"),
    Input("run-tools", "value"),
    State("run-customize-collapse", "is_open"),
    State("run-customize-auto", "data"),
    prevent_initial_call=True,
)
def run_preflight(is_open, preset, scope, run_symbols, _model_checks,
                  report_model, recs_basis, recs_model, run_evidence,
                  run_tools, customize_open=False, auto=None):
    """Name what this run needs and cannot reach, before it is started,
    and say where the controls have left the preset.

    A missing API key used to surface only as a stage failure minutes in,
    and the dialog gave no idea how long a run would take. The divergence
    check lives here because this callback already sees every field a
    preset fixes: when a value no longer matches, Customize unfolds so the
    difference is on screen; it never folds a section the user opened.

    It unfolds once per divergence set, not on every input change while a
    field differs: on the report shortcuts the scope always diverges, and
    a collapse that reopened on each keystroke could never be folded. The
    set it last opened for is kept in run-customize-auto; the modal
    opening starts over, so the next open shows the difference again.
    """
    from layouts.modals import (DEFAULT_RUN_PRESET, RUN_PRESETS,
                                preset_divergence, run_preflight_children)

    if not is_open:
        return (dash.no_update,) * 4
    # Read the ids alongside the values instead of zipping against a
    # hand-maintained order, a pattern-matching Input's ordering is Dash's to
    # decide, and a silent mis-zip here would name the wrong models.
    checks = next((entry for entry in ctx.inputs_list
                   if isinstance(entry, list) and entry
                   and entry[0].get("id", {}).get("type") == "run-model-check"),
                  [])
    selected = [c["id"]["model"] for c in checks if c.get("value")]
    preflight = run_preflight_children(
        scope or "full",
        (run_symbols or {}).get("symbols") or [],
        selected,
        report_model or "",
        recs_basis or "auto",
        recs_model or "",
    )
    preset_name = preset if preset in RUN_PRESETS else DEFAULT_RUN_PRESET
    customized = preset_divergence(preset_name, {
        "scope": scope,
        "models": selected if checks else None,
        "recs": recs_basis,
        "evidence": run_evidence,
        "tools": run_tools,
    })
    labels = {"scope": "what to run", "models": "models",
              "recs": "recommendations", "evidence": "evidence blocks",
              "tools": "tools"}
    hint = [RUN_PRESETS[preset_name]["hint"]]
    if customized:
        hint.append(html.Span(
            " Customized: " + ", ".join(labels[f] for f in customized) + ".",
            className="run-preset-customized"))
    # The modal's is_open among the triggers is the dialog opening (a
    # close returned above): what it unfolded for last time is forgotten.
    # Otherwise only a field that was not diverging before is news worth
    # unfolding a folded section for; a difference the user removed is
    # not, and a section already open needs nothing.
    just_opened = any(t.get("prop_id") == "run-modal.is_open"
                      for t in (ctx.triggered or []))
    last = [] if just_opened else list((auto or {}).get("diverged") or [])
    news = [f for f in customized if f not in last]
    unfold = bool(news) and (just_opened or not customize_open)
    auto_out = ({"diverged": customized}
                if just_opened or customized != last else dash.no_update)
    return preflight, hint, (True if unfold else dash.no_update), auto_out


@callback(
    Output("full-analysis-requested", "data"),
    Input("run-dispatch", "data"),
    State("run-dispatched", "data"),
    prevent_initial_call=True,
)
def set_full_analysis_flag(run_store, dispatched):
    """Arm the synthesis stage, but only for a full-pipeline run.

    The armed value is the run's id, not a bare True: the stages that read
    it (persist_predictions, the synthesis callback) treat it as armed only
    for the run it names, so a flag left behind by a run that ended without
    disarming it cannot make a later run wait for a synthesis that never
    comes. An empty symbol set never arms: the run dialog rejects it inline
    and no stage will fire. The scope is the row's (its config), the same
    one the other stages read; the dispatcher's snapshot stands in only
    for an unreadable row.
    """
    run_id, row = _dispatch_run(run_store, dispatched)
    config = _run_settings(row)
    scope = ((config or {}).get("scope") or (run_store or {}).get("scope")
             or "full")
    if scope == "full" and _run_symbols_of(row, run_store):
        return run_id
    raise PreventUpdate


def _synthesis_armed(flag, run_id) -> bool:
    """Whether full-analysis-requested is armed FOR this run. A bare True
    is a session store written before the flag carried the id."""
    return flag is True or (bool(flag) and flag == run_id)


# Shared with the scheduled runner (scripts/daily_analysis.py) so the UI and
# cron paths merge research identically.
from services.analysis_runner import (  # noqa: E402
    merge_research_into_analysis as _merge_research_into_analysis,
)


@callback(
    Output("recommendations-store", "data"),
    # Every terminal branch also disarms the flag. A stale armed flag made
    # the NEXT run's stores look like a full run still in flight.
    Output("full-analysis-requested", "data", allow_duplicate=True),
    Input("ai-analysis-store", "data"),
    Input("model-signals-store", "data"),
    State("full-analysis-requested", "data"),
    prevent_initial_call=True,
)
async def generate_recommendations_callback(ai_analysis, model_signals,
                                            requested):
    """Generate recommendations when their inputs are ready.

    Fires for the Full Analysis flow (requested flag) AND for AI Report runs
    that asked for recommendations (recs_request on the store, carrying the
    evidence basis: research+signals / news+signals / signals).

    A full run synthesizes exactly once, from BOTH of that click's stores:
    the report lands first, and synthesizing from it alone ran Luna without
    any model signals, and "finished" the run while the models were still
    working. run_seq (stamped on the report and the signals _meta by the
    same confirm click) pairs the two. Report-only payloads proceed without
    waiting, but only on the report's own landing. A later models-only
    run's signals must never resurrect a stale report's recs_request.

    Async: the ~1-minute Luna synthesis and the Postgres persist loop run in
    worker threads instead of pinning a callback slot.
    """
    basis = (ai_analysis or {}).get("recs_request")
    if not requested and not basis:
        raise PreventUpdate

    from services import progress_service as prog

    # The run this synthesis belongs to: the report's own stamp, else the
    # signals' (a models-only payload never reaches synthesis, so the report
    # is the authority when both are present).
    sig_meta = (model_signals or {}).get("_meta") or {}
    run_id = (ai_analysis or {}).get("run_id") or sig_meta.get("run_id")

    # A cancelled run must not synthesize: its report can still land after
    # the cancel (the report stage cannot be killed), and the cancel has
    # already disarmed the flag, so this landing would otherwise read as a
    # report-only run with recommendations still to produce.
    row = None
    if run_id:
        try:
            from services import run_service
            row = run_service.get_run(run_id)
        except Exception as e:
            logger.warning("run %s: run row unreadable: %s", run_id[:8], e)
        if row is not None and row["status"] == "cancelled":
            raise PreventUpdate

    # A report-scope payload never gets model signals, so a lingering
    # requested flag must not make it wait for them. The flag carries the
    # run it was armed for; one left by another run is not armed here.
    is_full_run = (_synthesis_armed(requested, run_id)
                   and (ai_analysis or {}).get("scope") != "report")
    if is_full_run:
        run_seq = (ai_analysis or {}).get("run_seq")
        if run_seq is None or run_seq != sig_meta.get("run_seq"):
            raise PreventUpdate
        if (model_signals or {}).get("_run_failed"):
            # The model stage crashed; its handler already closed the run.
            return {"error": "Model run failed, recommendations skipped",
                    "generated_at": datetime.now().isoformat()}, False
    elif ctx.triggered_id != "ai-analysis-store":
        # No full run armed for this click, so only the report's OWN landing
        # may synthesize. Fresh signals from a models-only run must not be
        # glued to whatever stale report (and its recs_request) still sits
        # in the session store, that ran Luna right after a models-only
        # "Predictions complete". No finish needed: persist_predictions
        # already closed that run.
        raise PreventUpdate

    if not ai_analysis:
        raise PreventUpdate
    if ai_analysis.get("failed"):
        if is_full_run:
            # The report is a synthesis input; without it this run cannot
            # produce recommendations. Close the run honestly.
            _close_run(run_id, "Full Analysis finished with errors, report "
                               "failed", "failed",
                       error=ai_analysis.get("error") or "AI report failed")
            return {"error": "AI report failed, recommendations skipped",
                    "generated_at": datetime.now().isoformat()}, False
        # Report-only failures are closed by the report callback itself.
        raise PreventUpdate

    # Recommendations explicitly turned off for this run: the Full Analysis
    # flow still sets the requested flag, so honor the store's opt-out marker
    # (and close the progress run once predictions have landed).
    if ai_analysis.get("recs_off"):
        if is_full_run:
            prog.emit_progress("synthesis", state="skipped", run_id=run_id)
            _close_run(run_id, "Full Analysis complete (recommendations off)")
            return dash.no_update, False
        raise PreventUpdate

    basis = basis or "news+signals"  # Full Analysis default
    # The run's symbols are the row's, not the dialog's current chips (the
    # dialog may have been reopened and edited while this run was in
    # flight). Without a readable row, the report's and signals' own
    # symbol keys are what there is to synthesize over.
    symbols = list((row or {}).get("symbols") or [])
    if not symbols:
        seen = list((ai_analysis.get("by_symbol") or {}).keys())
        seen += [k for k in (model_signals or {})
                 if k not in seen and k not in ("_meta", "_run_failed")]
        symbols = seen

    valid_signals = {
        k: v for k, v in (model_signals or {}).items()
        if k not in ("_meta", "_run_failed") and isinstance(v, dict)
    }
    if not valid_signals:
        if basis == "signals":
            # Predictions-only synthesis with no predictions is a no-op. 
            # say so instead of silently doing nothing.
            prog.emit("error", "Recommendations (predictions only) skipped, "
                               "no model predictions in this session. Run Predict first.",
                      run_id=run_id)
            if is_full_run:
                _close_run(run_id, "Full Analysis finished with errors, "
                                   "no model predictions", "failed",
                           error="no model predictions to synthesize")
                return dash.no_update, False
            _close_run(run_id, "Report complete: recommendations skipped "
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
        run_recommendations, ai_analysis, model_signals, symbols or [], trade_date,
        run_id=run_id)

    if result and result.get("from_cache"):
        # A cache-served run is still a completed run, without the finish
        # the panel spinner spun forever on a success.
        _close_run(run_id,
                   "Full Analysis complete (recommendations from cache)"
                   if is_full_run else "Report complete (recommendations from cache)")
        return result, False
    if result:
        _close_run(run_id, "Full Analysis complete" if is_full_run
                           else "Report complete")
        return result, False

    _close_run(run_id, ("Full Analysis" if is_full_run else "Report")
                       + " finished with errors", "failed",
               error="recommendations model unavailable or returned empty response")
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

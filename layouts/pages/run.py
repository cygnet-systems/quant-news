"""Run page: one URL per run.

The completion toast and the topbar pill both land here. The header says
what ran (kind, preset, owner, sessions, timing, status), a failed or
cancelled run states why at the top, then one board row per symbol using
the same cells as the Home board plus the research report and a way to
add the name to the watchlist, and the portfolio synthesis under the rows.

Pure rendering over the dict services.dashboard_service.get_run_view
returns; the reader modal is opened by app.py on ``?open=first``.
"""

from datetime import datetime, timezone

from dash import dcc, html

from layouts.pages.home import DECISION_CLASS, board_headers, symbol_row
from services.progress_service import format_stamp
from services.run_service import (
    ACTIVE_STATUSES,
    STAGES,
    TERMINAL_STATUSES,
    first_line,
)

STATUS_LABEL = {
    "queued": "Starting",
    "running": "Running",
    "done": "Complete",
    "failed": "Failed",
    "cancelled": "Cancelled",
}

_PRESET_LABEL = {
    "quick": "Quick", "standard": "Standard", "deep": "Deep",
    "custom": "Custom",
}


def _parse_ts(value) -> datetime | None:
    if not value:
        return None
    try:
        ts = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    # SQLite hands back naive stamps; the row is written in UTC.
    return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)


def _fmt_secs(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    if seconds < 90:
        return f"{seconds}s"
    minutes, rest = divmod(seconds, 60)
    return f"{minutes} min {rest}s" if rest else f"{minutes} min"


def duration_label(run: dict, now: datetime | None = None) -> str | None:
    """Wall clock of a finished run, or how long a live one has been going.

    None when the row has no start stamp (never happens for rows the
    service wrote, but a hand-edited row must not crash the page).
    """
    started = _parse_ts(run.get("started_at"))
    if started is None:
        return None
    finished = _parse_ts(run.get("finished_at"))
    if finished is not None:
        return _fmt_secs((finished - started).total_seconds())
    now = now or datetime.now(timezone.utc)
    return f"{_fmt_secs((now - started).total_seconds())} so far"


def cancelled_before(run: dict) -> str | None:
    """"cancelled before <stage>" for the first stage the cancel cut off.

    A stage that finished or was deliberately skipped counts as done; the
    first one still pending or running is where the run stopped. When
    every stage had closed the cancel changed nothing visible.
    """
    if run.get("status") != "cancelled":
        return None
    stages = run.get("stages") or {}
    for stage in STAGES:
        state = (stages.get(stage) or {}).get("state") or "pending"
        if state not in ("done", "skipped", "failed"):
            return f"cancelled before {stage}"
    return "cancelled after every stage had finished"


def banner(run: dict) -> html.Div | None:
    """The error line for a failed run, the cut-off stage for a cancelled
    one, nothing otherwise. Sits above the header so it is the first
    thing read."""
    status = run.get("status")
    if status == "failed":
        reason = first_line(run.get("error")) or "unknown error"
        return html.Div(
            [html.I(className="bi bi-x-octagon me-2"),
             html.Span("Run failed: ", className="run-banner-label"),
             html.Span(reason)],
            className="run-banner run-banner-failed",
        )
    if status == "cancelled":
        return html.Div(
            [html.I(className="bi bi-slash-circle me-2"),
             html.Span("Run ", className="run-banner-label"),
             html.Span(cancelled_before(run))],
            className="run-banner run-banner-cancelled",
        )
    return None


def _meta(label: str, value, cls: str = "") -> html.Span:
    return html.Span(
        [html.Span(f"{label} ", className="home-meta-label"),
         html.Span(value if value not in (None, "") else "n/a",
                   className=("num " + cls).strip())],
        className="run-meta",
    )


def status_pill(run: dict) -> html.Span:
    """The header's status word. Carries an id because the poll rewrites
    it in place while the run is live (app.refresh_live_run)."""
    status = run.get("status") or "queued"
    return html.Span(STATUS_LABEL.get(status, status), id="run-status-pill",
                     className=f"run-status run-status-{status}")


def live_fingerprint(run: dict) -> dict:
    """Everything a poll tick must notice on a live run page.

    The status drives the pill and the empty-state wording; the stage
    entries (their state, their done/total and their per-symbol glyphs)
    and the counters move whenever a stage lands or another symbol is
    stored, which is exactly when there are new rows to draw. Nothing on
    this page changes without one of the three moving, so an unchanged
    fingerprint means an unchanged board and the tick can write nothing.
    The status is kept at the top level because the poll reads it back to
    decide whether the run is worth another query at all.
    """
    return {
        "status": run.get("status"),
        "stages": run.get("stages") or {},
        "counters": run.get("counters") or {},
    }


def header(view: dict) -> html.Div:
    run = view["run"]
    kind = run.get("kind") or "manual"
    preset = run.get("preset")
    symbols = run.get("symbols") or []
    n_reports = sum(1 for r in view.get("symbols") or [] if r.get("report"))

    title_bits = [
        html.Span(f"{kind} run", className=f"run-kind-badge run-kind-{kind}",
                  title="Scheduled runs come from the daily job; manual "
                        "runs from the Run analysis dialog"),
        html.Span(_PRESET_LABEL.get(preset, preset) if preset else "",
                  className="run-preset") if preset else "",
        html.Span(", ".join(symbols), className="run-symbols"),
        status_pill(run),
    ]
    facts = [
        _meta("Owner", run.get("owner_uid") or "anonymous"),
        _meta("Predicting", run.get("target_date")),
        _meta("Data through", run.get("prediction_date")),
        _meta("Started", format_stamp(run.get("started_at")) or None),
        _meta("Duration", duration_label(run)),
        _meta("Symbols", str(len(symbols))),
        _meta("Models", str(len(view.get("model_names") or []))),
        _meta("Reports", str(n_reports)),
    ]
    return html.Div(
        [
            html.Div(title_bits, className="run-titlerow"),
            html.Div(facts, className="home-meta-row run-facts"),
        ],
        className="run-header",
    )


def _report_cell(row: dict) -> html.Td:
    report = row.get("report")
    if not report:
        return html.Td(html.Span("no report", className="run-report-none"),
                       className="run-report-cell")
    conf = report.get("confidence")
    decision = report.get("decision") or "?"
    return html.Td(
        [
            html.Span(
                decision + (f" {conf:.0%}" if conf is not None else ""),
                className="home-sym-report-verdict "
                          + DECISION_CLASS.get(decision, "neutral"),
            ),
            html.Span(report.get("trade_date") or "",
                      className="num home-sym-report-date"),
            html.Button(
                "Open",
                id={"type": "ta-view-btn", "report": str(report.get("id", ""))},
                className="ta-view-btn run-report-open",
                title="Open the research report",
            ),
        ],
        className="run-report-cell",
    )


def _watchlist_cell(symbol: str, in_watchlist: bool) -> html.Td:
    if in_watchlist:
        return html.Td(html.Span("on watchlist", className="run-on-watchlist"),
                       className="run-watch-cell")
    return html.Td(
        html.Button(
            "+ watchlist",
            id={"type": "add-symbol", "symbol": symbol},
            className="run-add-btn",
            title=f"Add {symbol} to the watchlist",
        ),
        className="run-watch-cell",
    )


def first_report(view: dict) -> dict | None:
    """The report the reader opens on arrival: the first row that has one,
    in the run's own symbol order."""
    for row in view.get("symbols") or []:
        if row.get("report"):
            return row["report"]
    return None


def symbol_table(view: dict, watchlist=None, open_first: bool = False) -> html.Div:
    """The board rows with the report and watchlist cells appended.

    Rendered into #run-symbol-table so the watchlist callback can swap it
    when a name is added without rebuilding the page.
    """
    run = view["run"]
    models = view.get("model_names") or []
    rows = view.get("symbols") or []
    wl = set(watchlist or [])
    opened = first_report(view) if open_first else None

    trs = []
    for row in rows:
        props = {}
        if opened and row.get("report") is opened:
            props["className"] = "run-row-opened"
        trs.append(symbol_row(
            row, models,
            extra_cells=[_report_cell(row),
                         _watchlist_cell(row["symbol"], row["symbol"] in wl)],
            **props,
        ))

    children = []
    has_calls = any(r.get("models") or r.get("synthesis") for r in rows)
    if not has_calls:
        status = run.get("status")
        if status in ACTIVE_STATUSES:
            note = "No predictions yet. Rows fill in as the models finish."
        elif status in TERMINAL_STATUSES:
            note = "No predictions were stored for this run."
        else:
            note = "No predictions."
        children.append(html.Div(note, className="home-empty-note run-empty"))

    header_cells = board_headers(models, extra=[html.Th("Report"), html.Th("")])
    children.append(html.Div(
        html.Table([html.Thead(html.Tr(header_cells)), html.Tbody(trs)],
                   className="history-data-table"),
        className="history-table-wrap",
    ))
    return html.Div(children, className="run-board")


def _position_row(symbol: str, rec: dict) -> html.Div:
    action = (rec.get("action") or "HOLD").upper()
    p = rec.get("p_correct")
    try:
        p_label = f"{float(p):.0%}" if p is not None else ""
    except (TypeError, ValueError):
        p_label = ""
    facts = []
    if rec.get("key_level"):
        facts.append(html.Span([html.Span("level ", className="home-meta-label"),
                                html.Span(str(rec["key_level"]))],
                               className="run-pos-fact"))
    if rec.get("change_trigger"):
        facts.append(html.Span([html.Span("flips if ", className="home-meta-label"),
                                html.Span(str(rec["change_trigger"]))],
                               className="run-pos-fact"))
    return html.Div(
        [
            html.Div(
                [
                    html.Span(symbol, className="run-pos-symbol"),
                    html.Span(
                        [html.Span(action, className="home-chip-decision"),
                         html.Span(p_label, className="home-chip-conf num")
                         if p_label else ""],
                        className="home-chip home-chip-"
                                  + DECISION_CLASS.get(action, "neutral"),
                    ),
                ]
                + facts,
                className="run-pos-head",
            ),
            html.Div(rec.get("reasoning") or "", className="run-pos-reasoning")
            if rec.get("reasoning") else "",
        ],
        className="run-pos",
    )


def synthesis_card(recommendation: dict | None, run: dict) -> html.Div:
    """The portfolio synthesis from recommendation_runs.result_json.

    A compact card rather than the archive's table row: the archive lists
    many runs in one line each, this page has one run and room for its
    positions. Without a stored synthesis the card says why there is none.
    """
    if not recommendation:
        state = ((run.get("stages") or {}).get("synthesis") or {}).get("state")
        if state == "skipped":
            note = "Recommendations were off for this run."
        elif state == "failed":
            reason = ((run.get("stages") or {}).get("synthesis") or {}).get("error")
            note = "Synthesis failed" + (f": {reason}" if reason else ".")
        elif run.get("status") in ACTIVE_STATUSES:
            note = "Synthesis pending."
        else:
            note = "No synthesis was stored for this run."
        return html.Div(
            [html.Div("Synthesis", className="run-card-title"),
             html.Div(note, className="home-empty-note")],
            className="home-card run-synthesis",
        )

    result = recommendation.get("result_json") or {}
    overall = result.get("overall") or {}
    by_symbol = result.get("by_symbol") or {}
    duration_ms = recommendation.get("duration_ms")

    meta = [
        html.Span(recommendation.get("model_used") or "", className="run-syn-model"),
        html.Span(f"{duration_ms / 1000:.0f}s", className="num run-syn-duration",
                  title="Synthesis wall clock") if duration_ms else "",
        html.Span(format_stamp(recommendation.get("created_at")),
                  className="num run-syn-when"),
    ]
    body = []
    if overall.get("portfolio_action"):
        body.append(html.Div(overall["portfolio_action"], className="run-syn-action"))
    if overall.get("summary"):
        body.append(html.Div(overall["summary"], className="run-syn-summary"))
    if by_symbol:
        body.append(html.Div(
            [_position_row(sym, rec or {}) for sym, rec in by_symbol.items()],
            className="run-positions",
        ))
    if overall.get("risk_assessment"):
        body.append(html.Div(
            [html.Span("Risk ", className="home-meta-label"),
             html.Span(overall["risk_assessment"])],
            className="run-syn-risk",
        ))
    items = overall.get("watch_items") or []
    if items:
        body.append(html.Div(
            [html.Span("Watch ", className="home-meta-label"),
             html.Ul([html.Li(str(i)) for i in items], className="run-syn-watch")],
            className="run-syn-watchrow",
        ))
    if not body:
        body.append(html.Div("The synthesis carried no readable content.",
                             className="home-empty-note"))

    return html.Div(
        [
            html.Div(
                [html.Div("Synthesis", className="run-card-title"),
                 html.Div(meta, className="run-syn-meta")],
                className="run-card-head",
            ),
            html.Div(body, className="run-syn-body"),
        ],
        className="home-card run-synthesis",
    )


def layout(view: dict, watchlist=None, open_first: bool = False) -> html.Div:
    run = view["run"]
    top = banner(run)
    return html.Div(
        [
            top if top is not None else "",
            header(view),
            html.Div(symbol_table(view, watchlist, open_first),
                     id="run-symbol-table"),
            synthesis_card(view.get("recommendation"), run),
            # Seeded from the render so the first poll tick after a load
            # only redraws when something actually moved since.
            dcc.Store(id="run-live-fp", data=live_fingerprint(run)),
        ],
        className="page page-run",
        id="run-page",
    )


def not_found(run_id: str) -> html.Div:
    return html.Div(
        [
            html.Div("Run not found", className="empty-state-title"),
            html.Div(f"No run with id {run_id}. It may have been made on "
                     f"another deployment, or the link is stale.",
                     className="empty-state-note"),
        ],
        className="empty-state page page-run",
        id="run-page",
    )

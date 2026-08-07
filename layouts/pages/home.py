"""Home: what was predicted last, and how it turned out.

Opening the app used to show empty charts. The question it should answer on
arrival is the one you actually have: what is the current call on each name,
what did it cost or make, and what is still in flight.

Two panes, one viewport. The left pane is the symbol index — every name in
the latest cohort with its synthesis call and newest research report, one
click from the report itself. The right pane is the prediction board for the
same cohort. Clicking a symbol on the left narrows the board to that name;
clicking it again widens back out. Neither pane scrolls the page — each
scrolls internally, so the layout holds at any watchlist size.

Built around the latest prediction cutoff rather than "the last run" because
predictions carry no run id. The cutoff is also the more useful unit: it
survives a run being re-executed or topped up one symbol at a time.
"""

import dash_bootstrap_components as dbc
from dash import dcc, html

from layouts.formatters import MODEL_DISPLAY

DECISION_CLASS = {"BUY": "positive", "SELL": "negative", "HOLD": "neutral"}


def _decision_chip(pred: dict, compact: bool = True) -> html.Span:
    if not pred:
        return html.Span("no call", className="home-chip home-chip-none")
    decision = (pred.get("decision") or "HOLD").upper()
    conf = pred.get("confidence")
    label = decision if compact else f"{decision}"
    return html.Span(
        [
            html.Span(label, className="home-chip-decision"),
            html.Span(f"{conf:.0%}", className="home-chip-conf num")
            if conf is not None else "",
        ],
        className=f"home-chip home-chip-{DECISION_CLASS.get(decision, 'neutral')}",
        title=f"{decision}"
              + (f" at {conf:.0%} confidence" if conf is not None else ""),
    )


def _resolution_cell(row: dict) -> html.Td:
    """Resolved, held, or awaiting the target close.

    The pending case is not an error state and must not read like one: a
    next-session call cannot be scored until that session closes and the
    evaluator runs.
    """
    preds = list(row["models"].values())
    if row.get("synthesis"):
        preds.append(row["synthesis"])
    states = {p["state"] for p in preds}

    if states == {"pending"}:
        return html.Td(
            html.Span(f"awaiting {row.get('target_date', '')} close",
                      className="home-pending-pill"),
            className="home-resolution",
        )

    scored = [p for p in preds if p["state"] != "pending"]
    hits = sum(1 for p in scored if p.get("was_correct") is True)
    directional = [p for p in scored if p.get("was_correct") is not None]
    pnl = sum(p.get("pnl_dollars") or 0.0 for p in scored)
    actual = next((p.get("actual_close") for p in scored
                   if p.get("actual_close") is not None), None)

    pnl_cls = "positive" if pnl > 0 else "negative" if pnl < 0 else ""
    return html.Td(
        [
            html.Span(f"{actual:.2f}" if actual is not None else "n/a",
                      className="num home-actual"),
            html.Span(f"{hits}/{len(directional)} right" if directional
                      else "held", className="home-hits"),
            html.Span(f"${pnl:+.2f}", className=f"num {pnl_cls}"),
        ],
        className="home-resolution",
    )


def _report_reader(symbol: str, reports: list[dict]) -> html.Div:
    """The inline reading pane: the selected symbol's latest report, whole.

    Reading the latest report is the primary flow, so selecting a symbol
    puts the report itself on the page — not a button that leads to it.
    Older reports open in the reader modal via their date chips.
    """
    if not reports:
        return html.Div(
            [
                html.I(className="bi bi-journal home-reader-empty-icon"),
                html.Div(f"No research report for {symbol} yet",
                         className="home-reader-empty-title"),
                html.Div('Use "New report" on the left to generate one.',
                         className="home-reader-empty-note"),
            ],
            className="home-report-reader home-reader-empty",
        )

    latest = reports[0]
    conf = latest.get("confidence")
    decision = latest.get("decision") or "?"
    history = [
        html.Button(
            r.get("trade_date", ""),
            id={"type": "ta-view-btn", "report": str(r.get("id", ""))},
            className="home-reader-hist-chip",
            title=f"Open the {r.get('trade_date', '')} report "
                  f"({r.get('decision', '?')})",
        )
        for r in reports[1:]
    ]

    header = html.Div(
        [
            html.I(className="bi bi-journal-text home-reader-icon"),
            html.Span("Research report", className="home-reader-title"),
            html.Span(
                decision + (f" {conf:.0%}" if conf is not None else ""),
                className="home-sym-report-verdict "
                          + DECISION_CLASS.get(decision, "neutral"),
            ),
            html.Span(latest.get("trade_date", ""),
                      className="num home-reader-date"),
            html.Span(latest.get("model_name", ""),
                      className="home-reader-model"),
            html.Span(
                ([html.Span("earlier:", className="home-reader-hist-label")]
                 + history) if history else "",
                className="home-reader-history",
            ),
            html.A(
                [html.I(className="bi bi-file-earmark-pdf me-1"), "PDF"],
                href=f"/api/download/ta-report/{latest.get('id', '')}",
                className="home-reader-action",
            ),
            html.Button(
                [html.I(className="bi bi-arrows-fullscreen me-1"), "Expand"],
                id={"type": "ta-view-btn", "report": str(latest.get("id", ""))},
                className="home-reader-action",
                title="Open in the full-screen reader",
            ),
        ],
        className="home-reader-head",
    )

    return html.Div(
        [
            header,
            html.Div(
                dcc.Markdown(latest.get("report_text", ""),
                             className="ta-report-body"),
                className="home-report-body",
            ),
        ],
        className="home-report-reader",
    )


def cohort_table(cohort: dict, active_symbol: str | None = None,
                 symbol_reports: list[dict] | None = None) -> html.Div:
    """The prediction board, optionally narrowed to one symbol.

    Narrowed, the board compacts to that symbol's row and the remaining
    space becomes the reading pane with its latest research report.
    Exposed (not underscored) because the symbol-filter callback re-renders
    it without rebuilding the whole page.
    """
    models = cohort["model_names"]
    header = html.Thead(html.Tr(
        [html.Th("Symbol"), html.Th("Prev close")]
        + [html.Th(MODEL_DISPLAY.get(m, m), title=m) for m in models]
        + [html.Th("Synthesis",
                   title="Luna's verdict over the models, stored as a "
                         "prediction so it is scored the same way. It is a "
                         "synthesis of the others, not a peer model."),
           html.Th("Outcome")]
    ))

    symbols = cohort["symbols"]
    if active_symbol:
        symbols = [r for r in symbols if r["symbol"] == active_symbol]

    rows = []
    for row in symbols:
        prev = row.get("previous_close")
        rows.append(html.Tr(
            [
                html.Td(row["symbol"], className="home-symbol"),
                html.Td(f"{prev:.2f}" if prev is not None else "n/a",
                        className="num"),
            ]
            + [html.Td(_decision_chip(row["models"].get(m))) for m in models]
            + [html.Td(_decision_chip(row.get("synthesis"))),
               _resolution_cell(row)]
        ))

    children = []
    if active_symbol:
        children.append(html.Div(
            [
                html.Span(f"Showing {active_symbol} only",
                          className="home-filter-note-text"),
                html.Button(
                    [html.I(className="bi bi-x-lg me-1"), "Show all"],
                    id={"type": "home-sym-btn", "symbol": active_symbol},
                    className="home-filter-clear",
                ),
            ],
            className="home-filter-note",
        ))
    board_cls = "history-table-wrap home-board-scroll"
    if active_symbol:
        # One row: let the reading pane have the flex space instead.
        board_cls += " home-board-compact"
    children.append(html.Div(
        html.Table([header, html.Tbody(rows)], className="history-data-table"),
        className=board_cls,
    ))
    if active_symbol:
        children.append(_report_reader(active_symbol, symbol_reports or []))
    return html.Div(children)


def _last_run_header(cohort: dict, last_run: dict | None) -> html.Div:
    counts = cohort["counts"]
    bits = [
        html.Span([html.Span("Predicting ", className="home-meta-label"),
                   html.Span(cohort.get("target_date") or "n/a", className="num")]),
        html.Span([html.Span("Data through ", className="home-meta-label"),
                   html.Span(cohort["prediction_date"], className="num")]),
        html.Span([html.Span(str(len(cohort["symbols"])), className="num"),
                   html.Span(" symbols", className="home-meta-label")]),
        html.Span([html.Span(str(len(cohort["model_names"])), className="num"),
                   html.Span(" models", className="home-meta-label")]),
    ]
    if last_run:
        bits.append(html.Span(
            [html.Span(f"{last_run['duration_s']:.0f}s", className="num"),
             html.Span(" run time", className="home-meta-label")],
            title=last_run.get("title", ""),
        ))
        if last_run.get("errors"):
            bits.append(html.Span(
                f"{last_run['errors']} error"
                f"{'s' if last_run['errors'] != 1 else ''}",
                className="negative",
            ))

    # Both counts link to Performance, where the same numbers are filterable
    # per call. A count you cannot act on is a dead end.
    status = []
    if counts["pending"]:
        status.append(dcc.Link(
            f"{counts['pending']} awaiting close",
            href="/performance",
            className="home-pending-pill home-status-link",
            title="See these predictions on the Performance page",
        ))
    if counts["resolved"] or counts["held"]:
        pnl_cls = ("positive" if cohort["pnl"] > 0
                   else "negative" if cohort["pnl"] < 0 else "")
        status.append(dcc.Link(
            f"{counts['resolved'] + counts['held']} scored · "
            f"${cohort['pnl']:+.2f}",
            href="/performance",
            className=f"num home-status-link {pnl_cls}",
            title="See which calls were right or wrong",
        ))

    return html.Div(
        [
            html.Div(bits, className="home-meta-row"),
            html.Div(status, className="home-status-row"),
        ],
        className="home-card-header",
    )


def _inflight_strip(open_preds: dict) -> html.Div:
    """One line per unresolved target session — not a card of its own."""
    if not open_preds["total"]:
        return html.Div("Nothing in flight — every call has been scored.",
                        className="home-inflight home-empty-note")
    lines = []
    for d in open_preds["dates"]:
        syms = d["symbols"]
        shown = ", ".join(syms[:10]) + ("…" if len(syms) > 10 else "")
        lines.append(html.Div(
            [
                html.Span("In flight", className="home-meta-label"),
                html.Span(d["target_date"], className="num home-open-date"),
                html.Span(f"{len(d['predictions'])} calls · "
                          f"{len(syms)} symbols", className="home-meta-label"),
                html.Span(shown, className="home-open-symbols", title=", ".join(syms)),
            ],
            className="home-inflight-line",
        ))
    return html.Div(lines, className="home-inflight")


def _rolling(rolling: list[dict], days: int) -> html.Div:
    if not rolling:
        return html.Div(f"No scored calls in the last {days} days.",
                        className="home-empty-note")
    cells = []
    for g in sorted(rolling, key=lambda g: -(g["trades"] or 0)):
        if not g["trades"]:
            continue
        pnl_cls = ("positive" if g["pnl"] > 0
                   else "negative" if g["pnl"] < 0 else "")
        cells.append(html.Div(
            [
                html.Div(MODEL_DISPLAY.get(g["name"], g["name"]),
                         className="home-stat-name"),
                html.Div(f"{g['hit_rate']:.0%}" if g["hit_rate"] is not None
                         else "n/a", className="num home-stat-big"),
                html.Div(
                    [
                        html.Span(f"{g['trades']} trades",
                                  className="home-meta-label"),
                        html.Span(f"${g['pnl']:+.2f}", className=f"num {pnl_cls}"),
                    ],
                    className="home-stat-sub",
                ),
            ],
            className="home-stat",
        ))
    if not cells:
        return html.Div(f"No positions taken in the last {days} days.",
                        className="home-empty-note")
    return html.Div(cells, className="home-stat-row")


def _symbol_row(row: dict, report: dict | None, active: bool) -> html.Div:
    """One name on the index: the call, the report, and the way in.

    The whole header line is a button — the row IS the filter control for
    the board on the right. The report line keeps its own View button, which
    opens the shared reader modal (the ta-view-btn pattern is global).
    """
    sym = row["symbol"]
    prev = row.get("previous_close")

    header = html.Button(
        [
            html.Span(sym, className="home-sym-name"),
            html.Span(f"{prev:.2f}" if prev is not None else "",
                      className="num home-sym-close"),
            _decision_chip(row.get("synthesis")),
        ],
        id={"type": "home-sym-btn", "symbol": sym},
        className="home-sym-head",
        title=f"Read {sym}'s latest report and narrow the board to it"
              if not active else "Back to all symbols",
    )

    if report:
        conf = report.get("confidence")
        report_line = html.Div(
            [
                html.I(className="bi bi-journal-text home-sym-report-icon"),
                html.Span(
                    (report.get("decision") or "?")
                    + (f" {conf:.0%}" if conf is not None else ""),
                    className="home-sym-report-verdict "
                              + DECISION_CLASS.get(report.get("decision"), "neutral"),
                ),
                html.Span(report.get("trade_date", ""),
                          className="num home-sym-report-date"),
                html.Button(
                    "View",
                    id={"type": "ta-view-btn", "report": str(report.get("id", ""))},
                    className="home-sym-report-view",
                    title="Open the research report",
                ),
            ],
            className="home-sym-report",
        )
    else:
        report_line = html.Div(
            [
                html.I(className="bi bi-journal home-sym-report-icon"),
                html.Span("no report yet", className="home-sym-report-none"),
            ],
            className="home-sym-report",
        )

    return html.Div(
        [header, report_line],
        className="home-sym-row" + (" active" if active else ""),
    )


def symbol_list(cohort: dict, reports_by_symbol: dict | None,
                active_symbol: str | None = None,
                search: str = "") -> list:
    """The left-pane index rows, filtered by the search box if non-empty."""
    reports_by_symbol = reports_by_symbol or {}
    needle = (search or "").strip().upper()
    rows = [r for r in cohort["symbols"]
            if not needle or needle in r["symbol"]]
    if not rows:
        return [html.Div(f'No symbol matching "{needle}"',
                         className="home-empty-note")]
    return [
        _symbol_row(r, reports_by_symbol.get(r["symbol"]),
                    active=(r["symbol"] == active_symbol))
        for r in rows
    ]


def _jobs_line(jobs: list[dict]) -> html.Div:
    """The scheduler, reduced to a heartbeat: name, time, last outcome."""
    if not jobs:
        return html.Div(
            dcc.Link("No scheduled jobs — set one up", href="/schedule",
                     className="home-card-link"),
            className="home-jobs-line",
        )
    parts = []
    for j in jobs:
        status = j.get("last_status") or "never run"
        cls = ("negative" if status == "error"
               else "positive" if status == "success" else "")
        parts.append(html.Span(
            [
                html.I(className=f"bi bi-circle-fill home-job-dot {cls}"),
                html.Span(j.get("id", ""), className="home-job-name"),
                html.Span(f"{j.get('hour', 0):02d}:{j.get('minute', 0):02d}",
                          className="num home-meta-label"),
            ],
            className="home-job-pill",
            title=f"{j.get('id', '')}: {status}, runs "
                  f"{j.get('hour', 0):02d}:{j.get('minute', 0):02d} "
                  f"{j.get('days_of_week', '')}",
        ))
    parts.append(dcc.Link("Schedule", href="/schedule",
                          className="home-card-link"))
    return html.Div(parts, className="home-jobs-line")


def layout(cohort, open_preds, rolling, last_run, jobs, rolling_days=30,
           reports_by_symbol=None, active_symbol=None,
           symbol_reports=None) -> html.Div:
    """Assemble the launch screen from already-shaped data."""
    if not cohort or not cohort.get("prediction_date"):
        return html.Div(
            html.Div(
                [
                    html.Div("No predictions yet", className="empty-state-title"),
                    html.Div(
                        "Add symbols in the toolbar, then run Full Analysis. "
                        "Once a run completes, this page shows what it "
                        "predicted and how those calls resolved.",
                        className="empty-state-note",
                    ),
                    dcc.Link("Go to Analyze", href="/analyze",
                             className="btn btn-primary btn-sm mt-2"),
                ],
                className="empty-state",
            ),
            className="page page-home",
        )

    left = html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.H2("Symbols", className="home-card-title"),
                            html.Span(str(len(cohort["symbols"])),
                                      className="home-count-badge num"),
                        ],
                        className="home-left-titlerow",
                    ),
                    dcc.Input(
                        id="home-symbol-search",
                        type="text",
                        placeholder="Filter symbols…",
                        debounce=False,
                        className="home-symbol-search",
                        autoComplete="off",
                    ),
                ],
                className="home-left-head",
            ),
            html.Div(
                symbol_list(cohort, reports_by_symbol, active_symbol),
                id="home-symbol-list",
                className="home-symbol-list",
            ),
            html.Div(
                [
                    dbc.Button(
                        [html.I(className="bi bi-plus-lg me-1"), "New report"],
                        id="reports-new-btn", size="sm", color="success",
                        className="home-new-report-btn",
                    ),
                    _jobs_line(jobs),
                ],
                className="home-left-foot",
            ),
        ],
        className="home-left",
    )

    right = html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.H2("Latest calls", className="home-card-title"),
                            html.Span("the most recent prediction cutoff",
                                      className="home-card-subtitle"),
                        ],
                        className="home-right-titlerow",
                    ),
                    dbc.Button(
                        [html.I(className="bi bi-lightning-fill me-1"),
                         "Run analysis"],
                        id="home-run-btn", size="sm", color="success",
                        outline=True,
                    ),
                ],
                className="home-right-head",
            ),
            _last_run_header(cohort, last_run),
            html.Div(
                [
                    html.Span(f"Last {rolling_days}d",
                              className="home-card-title-sm",
                              title=f"Hit rate on BUY/SELL calls over the "
                                    f"last {rolling_days} days"),
                    _rolling(rolling, rolling_days),
                    dcc.Link("Performance", href="/performance",
                             className="home-card-link"),
                ],
                className="home-rolling",
            ),
            html.Div(cohort_table(cohort, active_symbol,
                                  symbol_reports=symbol_reports),
                     id="home-cohort-table"),
            _inflight_strip(open_preds),
        ],
        className="home-right",
    )

    return html.Div([left, right], className="page page-home home-split")

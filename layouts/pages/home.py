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
    raw_conf = pred.get("confidence")
    label = decision if compact else f"{decision}"

    # Raw model confidence is anti-calibrated on this platform and is never
    # shown. The badge is the CALIBRATED value — what this raw confidence has
    # historically meant for this model — or nothing when the model lacks
    # enough evaluated history to earn a number.
    cal = None
    try:
        from services.calibration_service import calibrate
        cal = calibrate(pred.get("model_name", ""), raw_conf)
    except Exception:
        cal = None

    made = pred.get("prediction_date")
    target = pred.get("target_date")
    title = decision
    if cal is not None:
        title += (f" — calls like this one have been right {cal:.0%} of the "
                  f"time (calibrated from evaluated history; model claimed "
                  f"{raw_conf:.0%})")
    elif raw_conf is not None:
        title += (f" — model claims {raw_conf:.0%}, but has too little "
                  f"evaluated history to calibrate; treat as unsized")
    if made:
        title += f" — made with data through {made}"
    if target:
        title += f", predicting the {target} close"
    return html.Span(
        [
            html.Span(label, className="home-chip-decision"),
            html.Span(f"{cal:.0%}", className="home-chip-conf num")
            if cal is not None else "",
        ],
        className=f"home-chip home-chip-{DECISION_CLASS.get(decision, 'neutral')}",
        title=title,
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


def _report_tab(symbol: str, reports: list[dict]) -> html.Div:
    """The Report tab: the selected symbol's latest report, whole.

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
                dbc.Button(
                    [html.I(className="bi bi-plus-lg me-1"),
                     f"Generate report for {symbol}"],
                    id={"type": "new-report-btn", "symbol": symbol},
                    size="sm", color="success", outline=True,
                    className="mt-2",
                ),
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
            html.Span(f"by {latest['owner_uid']}",
                      className="home-reader-owner",
                      title="Who ran this report")
            if latest.get("owner_uid") else "",
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


_SIGNAL_CHIP_LABELS = {
    "rsi": lambda s: f"RSI {s.get('value', 0):.0f} · {s.get('signal', '')}",
    "macd": lambda s: f"MACD {s.get('signal', '').replace('_', ' ')}",
    "trend_50": lambda s: "above SMA50" if s.get("bullish") else "below SMA50",
    "trend_200": lambda s: "above SMA200" if s.get("bullish") else "below SMA200",
}


def _chart_tab(symbol: str, detail: dict | None) -> html.Div:
    """Price chart + latest indicator signals for the selected symbol."""
    if not detail or not detail.get("figure"):
        return html.Div(
            "Price data unavailable for this symbol right now.",
            className="home-empty-note home-detail-empty",
        )
    chips = []
    for key, fmt in _SIGNAL_CHIP_LABELS.items():
        sig = (detail.get("signals") or {}).get(key)
        if sig:
            try:
                chips.append(html.Span(fmt(sig), className="home-detail-chip"))
            except (TypeError, ValueError):
                continue
    return html.Div(
        [
            dcc.Graph(
                figure=detail["figure"],
                config={"displayModeBar": False},
                className="home-detail-graph",
            ),
            html.Div(chips, className="home-detail-chips") if chips else "",
            html.Div(
                "Daily bars, 6-month window. Open Analyze for indicators, "
                "longer windows and intraday.",
                className="home-detail-note",
            ),
        ],
        className="home-detail-chart",
    )


def _news_tab(symbol: str, detail: dict | None) -> html.Div:
    """Latest headlines for the selected symbol, sentiment-chipped."""
    articles = (detail or {}).get("articles") or []
    if not articles:
        return html.Div(
            f"No cached news for {symbol}. News is fetched when a run "
            "includes it, or from Analyze.",
            className="home-empty-note home-detail-empty",
        )
    rows = []
    for a in articles[:15]:
        score = a.get("sentiment_score")
        cls = "neutral"
        if isinstance(score, (int, float)):
            cls = "positive" if score > 0.15 else "negative" if score < -0.15 else "neutral"
        rows.append(html.Div(
            [
                html.Span(
                    f"{score:+.2f}" if isinstance(score, (int, float)) else "·",
                    className=f"num home-news-score {cls}",
                ),
                html.A(a.get("title", ""), href=a.get("url") or None,
                       target="_blank", className="home-news-title"),
                html.Span(
                    f"{a.get('source', '')} · "
                    f"{str(a.get('published_at', ''))[:16].replace('T', ' ')}",
                    className="home-news-meta",
                ),
            ],
            className="home-news-row",
        ))
    return html.Div(rows, className="home-detail-news")


def _detail_pane(symbol: str, reports: list[dict],
                 detail: dict | None) -> html.Div:
    """Selected-symbol research pane: Report | Chart | News.

    Restores the pre-run loop (look at the market data and the news before
    deciding to run) exactly where the symbol was selected, instead of
    requiring the name to be re-typed into the watchlist and opened on
    Analyze. All three tabs are prerendered server-side, so switching is
    instant and needs no callback.
    """
    n_articles = len((detail or {}).get("articles") or [])
    actions = html.Div(
        dbc.Button(
            [html.I(className="bi bi-plus-lg me-1"), f"New report · {symbol}"],
            id={"type": "new-report-btn", "symbol": symbol},
            size="sm", color="success", outline=True,
            title=f"Open the run dialog scoped to a report for {symbol}",
        ),
        className="home-detail-actions",
    )
    return html.Div(
        [
            actions,
            dbc.Tabs(
                [
                    dbc.Tab(_report_tab(symbol, reports), label="Report",
                            tab_class_name="home-detail-tab"),
                    dbc.Tab(_chart_tab(symbol, detail), label="Chart",
                            tab_class_name="home-detail-tab"),
                    dbc.Tab(_news_tab(symbol, detail),
                            label=f"News ({n_articles})" if n_articles else "News",
                            tab_class_name="home-detail-tab"),
                ],
                className="home-detail-tabs",
            ),
        ],
        className="home-detail-pane",
    )


def _news_flag(row: dict) -> html.Span | None:
    """Mark a symbol whose models ran without the news they expected.

    Only "unavailable" is flagged. A genuinely quiet window is a real
    observation and the models are entitled to act on it; a fetch that failed
    means the call was made blind, and that is not the same evidence.
    """
    statuses = {m.get("news_status")
                for m in list(row.get("models", {}).values())
                + ([row["synthesis"]] if row.get("synthesis") else [])
                if isinstance(m, dict)}
    if "unavailable" not in statuses:
        return None
    return html.Span(
        "no news",
        className="home-news-flag",
        title="The news source failed for this symbol, so its models ran "
              "without news. Treat this call as unsupported rather than as "
              "a neutral read.",
    )


def cohort_table(cohort: dict, active_symbol: str | None = None,
                 symbol_reports: list[dict] | None = None,
                 symbol_detail: dict | None = None) -> html.Div:
    """The prediction board, optionally narrowed to one symbol.

    Narrowed, the board compacts to that symbol's row and the remaining
    space becomes the research pane (report / chart / news tabs).
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
    in_cohort = True
    if active_symbol:
        symbols = [r for r in symbols if r["symbol"] == active_symbol]
        in_cohort = bool(symbols)

    rows = []
    for row in symbols:
        prev = row.get("previous_close")
        rows.append(html.Tr(
            [
                html.Td([row["symbol"], _news_flag(row)],
                        className="home-symbol"),
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
        # One row: let the research pane have the flex space instead.
        board_cls += " home-board-compact"
    if active_symbol and not in_cohort:
        children.append(html.Div(
            f"No calls for {active_symbol} in the latest run.",
            className="home-empty-note",
        ))
    else:
        children.append(html.Div(
            html.Table([header, html.Tbody(rows)],
                       className="history-data-table"),
            className=board_cls,
        ))
    if active_symbol:
        children.append(_detail_pane(active_symbol, symbol_reports or [],
                                     symbol_detail))
    return html.Div(children)


def last_run_header(cohort: dict, last_run: dict | None) -> html.Div:
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


def _symbol_row(row: dict, report: dict | None, active: bool,
                in_watchlist: bool = True) -> html.Div:
    """One name on the index: the call, the report, and the way in.

    The whole header line is a button — the row IS the filter control for
    the board on the right. Membership is a control too: watchlist rows get
    a remove ✕, cohort-only rows a one-click ＋ into the watchlist (both
    reuse the global manage_symbols patterns). The report line keeps its own
    View button, which opens the shared reader modal.
    """
    sym = row["symbol"]
    prev = row.get("previous_close")

    if in_watchlist:
        membership = html.Button(
            "✕",
            id={"type": "remove-symbol", "symbol": sym},
            className="home-sym-remove",
            title=f"Remove {sym} from the watchlist",
        )
    else:
        membership = html.Button(
            "＋",
            id={"type": "add-symbol", "symbol": sym},
            className="home-sym-add",
            title=f"Add {sym} to the watchlist",
        )

    header = html.Div(
        [
            html.Button(
                [
                    html.Span(sym, className="home-sym-name"),
                    html.Span(f"{prev:.2f}" if prev is not None else "",
                              className="num home-sym-close"),
                    _decision_chip(row.get("synthesis")),
                ],
                id={"type": "home-sym-btn", "symbol": sym},
                className="home-sym-head",
                title=f"Research {sym}: report, chart and news"
                      if not active else "Back to all symbols",
            ),
            membership,
        ],
        className="home-sym-headrow",
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


def symbol_list(cohort: dict | None, reports_by_symbol: dict | None,
                active_symbol: str | None = None,
                search: str = "",
                watchlist: list[str] | None = None) -> list:
    """The rail rows: watchlist first, then cohort names outside it.

    One list, membership visible — the rail is the union of the watchlist
    (the input to the next run) and the latest cohort (the output of the
    last one), grouped so the 5-vs-20 mismatch reads as fact, not mystery.
    Watchlist names with no calls yet still get a row, so a freshly added
    symbol is immediately researchable.
    """
    reports_by_symbol = reports_by_symbol or {}
    watchlist = [s.upper() for s in (watchlist or [])]
    needle = (search or "").strip().upper()

    cohort_rows = {r["symbol"]: r
                   for r in (cohort or {}).get("symbols") or []}

    def _match(sym):
        return not needle or needle in sym

    wl_rows = [cohort_rows.get(s) or {"symbol": s, "models": {}}
               for s in watchlist if _match(s)]
    extra_rows = [r for s, r in cohort_rows.items()
                  if s not in watchlist and _match(s)]

    if not wl_rows and not extra_rows:
        return [html.Div(f'No symbol matching "{needle}"'
                         if needle else "No symbols yet — add one above.",
                         className="home-empty-note")]

    def _group(label, count, hint=None):
        return html.Div(
            [html.Span(label),
             html.Span(str(count), className="home-group-count num")],
            className="home-group-label",
            title=hint or "",
        )

    out = []
    if wl_rows:
        out.append(_group("Watchlist", len(wl_rows),
                          "Your symbols — the default set for new runs"))
        out.extend(
            _symbol_row(r, reports_by_symbol.get(r["symbol"]),
                        active=(r["symbol"] == active_symbol),
                        in_watchlist=True)
            for r in wl_rows
        )
    if extra_rows:
        out.append(_group("Not in watchlist", len(extra_rows),
                          "Covered by the most recent run but not on your "
                          "watchlist — click ＋ on a row to add it"))
        out.extend(
            _symbol_row(r, reports_by_symbol.get(r["symbol"]),
                        active=(r["symbol"] == active_symbol),
                        in_watchlist=False)
            for r in extra_rows
        )
    return out


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
                html.I(className="bi bi-lock-fill home-job-lock")
                if not j.get("is_public", True) else "",
                html.Span(j.get("id", ""), className="home-job-name"),
                html.Span(f"{j.get('hour', 0):02d}:{j.get('minute', 0):02d}",
                          className="num home-meta-label"),
            ],
            className="home-job-pill",
            title=f"{j.get('id', '')}: {status}, runs "
                  f"{j.get('hour', 0):02d}:{j.get('minute', 0):02d} "
                  f"{j.get('days_of_week', '')}"
                  + ("" if j.get("is_public", True) else " · private"),
        ))
    parts.append(dcc.Link("Schedule", href="/schedule",
                          className="home-card-link"))
    return html.Div(parts, className="home-jobs-line")


def _rail_menu(recent_groups: list[list[str]] | None) -> dbc.DropdownMenu:
    """The rail's overflow menu: past watchlists and Clear all.

    Recent groups are rendered server-side at layout time (the durable
    history is a DB read anyway) and restore through the existing
    search-restore pattern, so no store round-trip is needed. Clear-all
    keeps its fixed id — manage_symbols takes it as an allow_optional
    Input now that it exists only on Home.
    """
    items = [dbc.DropdownMenuItem("Recent watchlists", header=True)]
    groups = recent_groups or []
    if not groups:
        items.append(dbc.DropdownMenuItem("none yet", disabled=True))
    for group in groups[:5]:
        label = ", ".join(group[:3]) + (f" +{len(group) - 3}"
                                        if len(group) > 3 else "")
        items.append(dbc.DropdownMenuItem(
            label,
            id={"type": "search-restore", "csv": ",".join(group)},
        ))
    items.append(dbc.DropdownMenuItem(divider=True))
    items.append(dbc.DropdownMenuItem(
        "Clear watchlist", id="clear-symbols-btn",
        class_name="text-danger",
    ))
    return dbc.DropdownMenu(
        items,
        label=html.I(className="bi bi-three-dots"),
        size="sm", color="link", caret=False,
        className="home-rail-menu",
        align_end=True,
    )


def board_title(cutoffs: list[str] | None, active_cutoff: str | None) -> html.Div:
    """The board's title row — re-rendered when the cutoff changes.

    Exposed (not underscored) because render_home_panes rebuilds it without
    rebuilding the whole page, same as cohort_table and symbol_list.
    """
    cutoffs = cutoffs or []
    is_latest = (not cutoffs or not active_cutoff
                 or active_cutoff == cutoffs[0])
    return html.Div(
        [
            html.H2("Latest calls" if is_latest else "Calls",
                    className="home-card-title"),
            html.Span(
                "the most recent prediction cutoff" if is_latest
                else f"as of the {active_cutoff} cutoff",
                className="home-card-subtitle",
            ),
        ],
        className="home-right-titlerow",
    )


def _cutoff_selector(cutoffs: list[str], active: str | None) -> html.Div:
    """Point the board at any past prediction cutoff, not only the newest.

    The options are the dates predictions actually exist for, so there is no
    empty-result dead end a free calendar picker would allow.
    """
    cutoffs = cutoffs or []
    latest = cutoffs[0] if cutoffs else None
    options = [
        {"label": f"{d} (latest)" if d == latest else d, "value": d}
        for d in cutoffs
    ]
    return html.Div(
        [
            html.Span("Cutoff", className="home-meta-label"),
            dcc.Dropdown(
                id="home-cutoff-dropdown",
                options=options,
                value=active or latest,
                clearable=False,
                searchable=True,
                className="home-cutoff-dropdown",
            ),
        ],
        className="home-cutoff-wrap",
        title="Show the prediction board as of an earlier run date",
    )


def layout(cohort, open_preds, rolling, last_run, jobs, rolling_days=30,
           reports_by_symbol=None, active_symbol=None,
           symbol_reports=None, watchlist=None, recent_groups=None,
           symbol_detail=None, cutoffs=None, active_cutoff=None) -> html.Div:
    """Assemble the launch screen from already-shaped data.

    The left rail always renders — it is the watchlist editor now, so a
    brand-new user lands on the add box, not on an empty page telling them
    to find the editor somewhere else.
    """
    watchlist = watchlist or []
    has_cohort = bool(cohort and cohort.get("prediction_date"))
    n_rail = len(set(watchlist)
                 | {r["symbol"] for r in (cohort or {}).get("symbols") or []})

    left = html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.H2("Symbols", className="home-card-title"),
                            html.Span(str(n_rail),
                                      className="home-count-badge num"),
                            _rail_menu(recent_groups),
                        ],
                        className="home-left-titlerow",
                    ),
                    # Filter only — adding happens in the global watchlist
                    # strip (always mounted) or via a row's ＋ button.
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
                symbol_list(cohort, reports_by_symbol, active_symbol,
                            watchlist=watchlist),
                id="home-symbol-list",
                className="home-symbol-list",
            ),
            html.Div(
                _jobs_line(jobs),
                className="home-left-foot",
            ),
        ],
        className="home-left",
    )

    if not has_cohort:
        right = html.Div(
            html.Div(
                [
                    html.Div("No predictions yet", className="empty-state-title"),
                    html.Div(
                        "Add symbols on the left, then use Run analysis in "
                        "the toolbar. Once a run completes, this page shows "
                        "what it predicted and how those calls resolved.",
                        className="empty-state-note",
                    ),
                ],
                className="empty-state",
            ),
            className="home-right",
        )
        return html.Div([left, right], className="page page-home home-split")

    cutoffs = cutoffs or []
    right = html.Div(
        [
            html.Div(
                [
                    html.Div(board_title(cutoffs, active_cutoff),
                             id="home-board-title"),
                    _cutoff_selector(cutoffs, active_cutoff),
                ],
                className="home-right-head",
            ),
            html.Div(last_run_header(cohort, last_run),
                     id="home-meta-wrap"),
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
                                  symbol_reports=symbol_reports,
                                  symbol_detail=symbol_detail),
                     id="home-cohort-table"),
            _inflight_strip(open_preds),
        ],
        className="home-right",
    )

    return html.Div([left, right], className="page page-home home-split")

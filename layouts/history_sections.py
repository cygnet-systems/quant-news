"""Archive sections, one builder per bucket.

These used to be one 400-line function rendering every bucket into a single
History tab. Splitting them lets each page build only what it shows, which is
the difference between the 14,000-node History tab and a page that renders its
own content.

Filtering is shared (filter_history_data) so the Performance, Reports and
Activity pages agree on what "this symbol, these dates" means.
"""

from datetime import datetime, timedelta

import dash_bootstrap_components as dbc
from dash import dcc, html

from layouts.formatters import confidence_tooltip, conviction_label, weight_label


def filter_items(items, filter_symbols, filter_date_range, specific_date=None, sym_key="symbol", date_key="trade_date"):
    """Filter a list of dicts by symbol set and date range or specific date."""
    if filter_symbols:
        items = [i for i in items if i.get(sym_key, "") in filter_symbols]
    if specific_date:
        items = [i for i in items if (i.get(date_key, "") or "")[:10] == specific_date[:10]]
    elif filter_date_range == "today":
        # The current SESSION, not the calendar day: on a weekend or holiday
        # "today" holds nothing, and showing an empty table would read as data
        # loss rather than as the market being shut.
        session = current_session().isoformat()
        items = [i for i in items if (i.get(date_key, "") or "")[:10] == session]
    elif filter_date_range and filter_date_range != "all":
        days = {"7d": 7, "30d": 30, "90d": 90}.get(filter_date_range, 0)
        if days:
            cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
            items = [i for i in items if (i.get(date_key, "") or "") >= cutoff]
    return items


def current_session():
    """Today if the market trades today, else the last session that ran."""
    from datetime import date as _date

    from utils.trading_calendar import get_previous_trading_day, is_trading_day
    today = _date.today()
    return today if is_trading_day(today) else get_previous_trading_day(today)


def collapsible_section(title, section_id, children, icon_class, default_open=False, count=0):
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


def build_history_filter_bar(history_data: dict, filter_symbols=None,
                             filter_date_range="all",
                             specific_date=None) -> html.Div:
    """Symbol dropdown, recent chips, date range, and the applied-filter chips.

    Shared by Performance and Reports so "this symbol, these dates" means the
    same thing on both.
    """
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
                # Seeded from the store rather than written back by a callback:
                # this component exists only on Performance and Reports, and a
                # callback writing to a missing Output is a hard error in Dash 4.
                value=list(filter_symbols or []),
                multi=True,
                searchable=True,
                placeholder="Filter by symbol...",
                className="history-symbol-dropdown",
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

    active_day = current_session()

    # Predictions are only ever made for a market session, so a weekend or
    # holiday can only ever filter to nothing. Grey them out instead of
    # letting the user pick a date that returns an empty list.
    try:
        from utils.trading_calendar import non_trading_days
        closed_days = [d.isoformat()
                       for d in non_trading_days("2020-01-01", active_day)]
    except Exception:
        closed_days = []

    rows.append(
        html.Div(
            [
                html.Div(
                    [
                        html.Span("Date:", className="history-date-label"),
                        dcc.DatePickerSingle(
                            id="history-date-picker",
                            date=specific_date,
                            initial_visible_month=active_day.isoformat(),
                            max_date_allowed=active_day.isoformat(),
                            disabled_days=closed_days,
                            display_format="YYYY-MM-DD",
                            placeholder=active_day.isoformat(),
                            className="history-date-picker",
                        ),
                    ],
                    className="history-date-picker-row",
                ),
                html.Div(
                    [
                        dbc.ButtonGroup(
                            [
                                # The common case — "what did we just do?" —
                                # needed a calendar click before this.
                                dbc.Button("Today", id={"type": "history-date-btn", "range": "today"},
                                           size="sm", outline=True, color="secondary",
                                           className="history-date-btn"),
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

    # Chips are filled by a callback, not built here: this bar must NOT be
    # rebuilt when a filter changes. Re-creating the dropdown re-fires its
    # value Input, which writes the filter store, which would rebuild the bar
    # again -- an infinite render loop.
    rows.append(html.Div(
        applied_filter_chips(filter_symbols, filter_date_range, specific_date),
        id="history-applied-filters",
        className="history-applied-filters",
    ))

    return html.Div(rows, className="history-filter-bar")


def applied_filter_chips(filter_symbols, filter_date_range, specific_date):
    """Removable chips for whatever narrowing is currently applied.

    Rendered inline rather than by a callback: the router already rebuilds the
    page when a filter store changes, so a separate callback would have to
    target an Output that does not exist on the routes without a filter bar.
    """
    chips = []
    for s in (filter_symbols or []):
        chips.append(html.Span(
            [
                s,
                html.Button("x", id={"type": "history-remove-filter", "symbol": s},
                            className="history-filter-chip-remove"),
            ],
            className="history-filter-chip",
        ))

    if specific_date:
        chips.append(html.Span(
            [
                specific_date[:10],
                html.Button("x",
                            id={"type": "history-remove-date-filter",
                                "range": "specific"},
                            className="history-filter-chip-remove"),
            ],
            className="history-filter-chip",
        ))
    elif filter_date_range and filter_date_range != "all":
        chips.append(html.Span(
            [
                f"Last {filter_date_range}",
                html.Button("x",
                            id={"type": "history-remove-date-filter",
                                "range": filter_date_range},
                            className="history-filter-chip-remove"),
            ],
            className="history-filter-chip",
        ))

    if chips:
        chips.append(html.Button("Clear all", id="history-clear-all-filters",
                                 className="history-clear-all-link"))
    return chips


def filter_history_data(history_data, filter_symbols, filter_date_range,
                        specific_date=None):
    """Apply the shared symbol and date filters to every bucket.

    Recommendations are filtered separately: they carry symbols_csv (a run
    covers a set of tickers) rather than one symbol, so a run stays visible if
    any of its symbols matches.
    """
    if isinstance(history_data, list):
        history_data = {"trading_agent_reports": history_data}

    reports = filter_items(history_data.get("reports", []), filter_symbols,
                           filter_date_range, specific_date)
    # Filtered by the session predicted, not the data cutoff. Every prediction
    # carries prediction_date = the previous trading day, so filtering on it
    # put this morning's run under yesterday's date: asking for today returned
    # nothing while all of today's predictions sat one day back.
    predictions = filter_items(history_data.get("predictions", []), filter_symbols,
                               filter_date_range, specific_date,
                               date_key="target_date")
    ta_reports = filter_items(history_data.get("trading_agent_reports", []),
                              filter_symbols, filter_date_range, specific_date)

    raw_recs = history_data.get("recommendations", [])
    if filter_symbols:
        filter_set = set(filter_symbols)
        recommendations = [
            rec for rec in raw_recs
            if {s.strip() for s in (rec.get("symbols_csv", "") or "").split(",")
                if s.strip()} & filter_set
        ]
    else:
        recommendations = raw_recs
    if specific_date:
        recommendations = filter_items(recommendations, [], "all", specific_date,
                                       date_key="created_at")
    elif filter_date_range and filter_date_range != "all":
        recommendations = filter_items(recommendations, [], filter_date_range,
                                       date_key="created_at")

    return {
        "reports": reports,
        "predictions": predictions,
        "trading_agent_reports": ta_reports,
        "recommendations": recommendations,
    }


def empty_history_message(has_filters: bool, noun: str = "data") -> html.Div:
    # The Reports page has its own generate affordance; everywhere else the
    # run lives behind the toolbar's Run analysis button.
    action = ('use "New report" above to generate one'
              if noun == "reports"
              else 'use "Run analysis" in the toolbar to generate data')
    return html.Div(
        [
            html.I(className="bi bi-clock-history",
                   style={"fontSize": "1.6rem", "opacity": "0.3",
                          "display": "block", "marginBottom": "10px"}),
            html.Div(f"No matching {noun}" if has_filters
                     else f"No {noun} yet",
                     style={"fontWeight": "600", "marginBottom": "4px"}),
            html.Div(
                f"Adjust the filters above, or {action}."
                if has_filters else
                f"{action[0].upper()}{action[1:]}.",
                style={"color": "var(--text-secondary)", "fontSize": "0.85rem"},
            ),
        ],
        className="history-empty-msg",
    )


def _ta_report_card(report):
    symbol = report.get("symbol", "")
    decision = report.get("decision", "HOLD")
    confidence = report.get("confidence", 0)
    trade_date = report.get("trade_date", "")
    input_tokens = report.get("input_tokens", 0)
    output_tokens = report.get("output_tokens", 0)

    dec_cls = "positive" if decision == "BUY" else "negative" if decision == "SELL" else "neutral"

    # The bare percentage here was the reliability weight, which readers took
    # for the report's own confidence. Name it, and show the report's stated
    # conviction beside it so the two are never confused again.
    from models.single_agent import extract_confidence
    stated = (extract_confidence(report["report_text"])
              if report.get("report_text") else None)

    return html.Div(
        [
            html.Div(
                [
                    html.Span(symbol, className="ta-card-symbol"),
                    html.Span(decision, className=f"history-decision {dec_cls}"),
                    html.Span(weight_label(confidence), className="ta-card-conf",
                              title=confidence_tooltip()),
                ],
                className="ta-card-header",
            ),
            html.Div(conviction_label(stated), className="ta-card-meta",
                     title=confidence_tooltip()),
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
                        # Keyed by report id, not list position: the
                        # page renders a filtered list while the
                        # callback resolved against a differently
                        # filtered store, so indices did not line up
                        # and View silently did nothing.
                        id={"type": "ta-view-btn",
                            "report": str(report.get("id", ""))},
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


def build_ta_reports_section(ta_reports):
    """Research reports, the richest artifact a run produces.

    Grouped by symbol so a ticker's report history reads as a timeline
    rather than being interleaved with every other name in the archive.
    Symbols are ordered by their newest report; within a symbol the rows
    keep the store's newest-first order.
    """
    if not ta_reports:
        return None

    by_symbol: dict[str, list] = {}
    for report in ta_reports:
        by_symbol.setdefault(report.get("symbol", "?"), []).append(report)

    groups = []
    for symbol, reports in by_symbol.items():
        latest = reports[0]
        groups.append(html.Div(
            [
                html.Div(
                    [
                        html.Span(symbol, className="ta-group-symbol"),
                        html.Span(f"{len(reports)} report"
                                  f"{'s' if len(reports) != 1 else ''}",
                                  className="ta-group-count"),
                        html.Span(f"latest {latest.get('trade_date', '')}",
                                  className="ta-group-latest"),
                    ],
                    className="ta-group-header",
                ),
                html.Div([_ta_report_card(r) for r in reports],
                         className="ta-cards-grid"),
            ],
            className="ta-symbol-group",
        ))

    # Cards only — View opens the report in a modal. The old duplicate
    # accordion stack below the cards (two ways to open the same report)
    # is gone.
    return collapsible_section(
        "Research Reports", "ta",
        html.Div(groups),
        icon_class="bi-robot", default_open=True, count=len(ta_reports),
    )


_REPORT_TYPE_LABELS = {
    "ai_report": "AI report",
    "trading_agents": "Research report",
    "recommendations": "Recommendations",
}


def build_saved_reports_section(reports):
    """Stored PDF/JSON/MD report artifacts."""
    if not reports:
        return None
    report_rows = []
    for r in reports:
        sym = r.get("symbol") or "Portfolio"
        # .title() alone turned "ai_report" into "Ai Report". Known types get
        # a written label; anything else still falls back to title-casing.
        raw_type = r.get("report_type", "") or ""
        rtype = _REPORT_TYPE_LABELS.get(
            raw_type, raw_type.replace("_", " ").title())
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

    return collapsible_section(
        "Saved Reports", "reports",
        html.Div(
            html.Table([
                html.Thead(html.Tr([html.Th("Date"), html.Th("Symbol"), html.Th("Type"), html.Th("")])),
                html.Tbody(report_rows),
            ], className="history-data-table"),
            className="history-table-wrap",
        ),
        # Open on arrival: a collapsed archive reads as an empty one.
        icon_class="bi-file-earmark-text", default_open=True, count=len(reports),
    )


# Rows rendered in the prediction log before it truncates. Deferring the
# build stopped it blocking page load, but opening the section still painted
# every row: 886 predictions is 1044 table rows, enough to lock the browser
# for seconds and read as a crash. The cap is on the SYMBOL groups actually
# built, and what was dropped is stated rather than silently missing.
PREDICTION_LOG_MAX_SYMBOLS = 12


def build_predictions_section(predictions, deferred: bool = False):
    """The per-call prediction log, grouped by symbol then date.

    With deferred=True the body is left empty and filled when the section is
    first opened. The log is a row per prediction per model, so a few hundred
    predictions is several thousand nodes: building it eagerly for a section
    that starts collapsed was most of the weight of the Performance page, and
    heavy enough to make Dash's async component chunks time out.

    Even deferred, the full set is too much to paint at once, so only the most
    recently active symbols are rendered. Narrow with the symbol filter or a
    symbol row in the scorecard to see the rest.
    """
    if not predictions:
        return None
    if deferred:
        return collapsible_section(
            "Model Predictions", "predictions",
            html.Div(id="predictions-log-body"),
            icon_class="bi-cpu", default_open=False, count=len(predictions),
        )
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

    # Most recently active first, then cap. Alphabetical order would truncate
    # to whatever happens to start with A, which is never what someone opening
    # this log is looking for.
    def _latest(sym):
        return max((p.get("target_date") or p.get("prediction_date") or "")
                   for p in by_symbol[sym])

    ordered = sorted(by_symbol, key=_latest, reverse=True)
    shown = ordered[:PREDICTION_LOG_MAX_SYMBOLS]
    hidden = ordered[PREDICTION_LOG_MAX_SYMBOLS:]

    symbol_groups = []
    for sym in shown:
        sym_preds = by_symbol[sym]
        # Directional hit rate only. HOLDs are now scored too (against the
        # no-trade band), but folding them in here would silently change
        # what this percentage has always meant.
        scored = [p for p in sym_preds
                  if p.get("was_correct") is not None
                  and p.get("decision") != "HOLD"]
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

    trailer = []
    if hidden:
        trailer.append(html.Div(
            f"Showing the {len(shown)} most recently active symbols. "
            f"{len(hidden)} more are in scope ({', '.join(hidden[:8])}"
            f"{'…' if len(hidden) > 8 else ''}) — filter by symbol, or click a "
            f"symbol in the scorecard above, to see them.",
            className="history-log-truncated",
        ))

    return collapsible_section(
        "Model Predictions", "predictions",
        html.Div([eval_bar] + symbol_groups + trailer),
        icon_class="bi-cpu", default_open=False, count=len(predictions),
    )


def build_recommendations_section(recommendations):
    """Portfolio-level synthesis runs."""
    if not recommendations:
        return None
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

    return collapsible_section(
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
        # Open on arrival: a collapsed archive reads as an empty one.
        icon_class="bi-lightning", default_open=True, count=len(recommendations),
    )


def build_activity_section(activity_scope="all", stages=None, symbol=None,
                           since_days=None):
    """The audit trail, optionally narrowed.

    Filters are opt-in and default to nothing: unfiltered, this is the whole
    record of what the system did, which is exactly what you want intact when
    something looks wrong.
    """
    from services import progress_service as _prog
    is_admin = _prog.viewer_is_admin()
    scope = activity_scope if is_admin else "self"
    activity_runs = _prog.get_activity_runs(
        limit_runs=50, scope=scope, stages=stages, symbol=symbol,
        since_days=since_days or None)
    if activity_runs:
        run_blocks = []
        for run_idx, run in enumerate(activity_runs):
            lines = [
                html.Div([
                    html.Span(_prog.event_clock(e), className="progress-ts",
                              title=_prog.DISPLAY_TZ_LABEL),
                    html.Span(e["message"], className="progress-msg"),
                ], className="progress-line"
                   + (" progress-line-error" if e["stage"] == "error" else ""))
                for e in run["events"]
            ]
            # get_activity_runs already hands these over in the display zone.
            started = (f"{run['started'].strftime('%Y-%m-%d %H:%M')} "
                       f"{_prog.DISPLAY_TZ_LABEL}")
            err = (f" · {run['errors']} error{'s' if run['errors'] != 1 else ''}"
                   if run["errors"] else "")
            # Whose run it was only matters when more than one person's rows
            # can appear, i.e. the admin "All users" view.
            owner = f"[{run['user_id']}] " if scope == "all" else ""
            # Index-keyed: runs are grouped by (user_id, run_id) and the
            # ad-hoc ids ("adhoc", "auth") repeat across users, so run_id
            # alone would emit duplicate Dash component ids.
            run_blocks.append(collapsible_section(
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

        return collapsible_section(
            "Activity Log", "activity",
            html.Div([
                html.Div(hint, className="history-scoreboard-hint"),
                scope_control,
                html.Div(run_blocks),
            ]),
            icon_class="bi-journal-text", default_open=True,
            count=len(activity_runs),
        )
    # Distinguish "filtered everything out" from "nothing ever ran": with
    # filters applied an empty result is a normal outcome, not a broken page.
    filtered = bool(stages or symbol or since_days)
    return html.Div(
        [
            html.Div("No matching activity" if filtered else "No activity recorded yet",
                     style={"fontWeight": "600", "marginBottom": "4px"}),
            html.Div(
                "Widen the filters above." if filtered
                else "Run an analysis and its progress will be logged here.",
                style={"color": "var(--text-secondary)", "fontSize": "0.85rem"},
            ),
        ],
        className="history-empty-msg",
    )

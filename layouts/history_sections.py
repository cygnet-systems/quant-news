"""Archive sections shared by the Performance, Reports and Activity pages.

Filtering, the collapsible wrapper and the per-bucket renderers all used to
live in app.py alongside the callbacks that drive them.
"""

from datetime import datetime, timedelta

import dash_bootstrap_components as dbc
from dash import dcc, html


def filter_items(items, filter_symbols, filter_date_range, specific_date=None, sym_key="symbol", date_key="trade_date"):
    """Filter a list of dicts by symbol set and date range or specific date."""
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


def build_history_filter_bar(history_data: dict) -> html.Div:
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


def build_history_tab_layout(history_data: dict) -> html.Div:
    """Build the full History tab: filter bar (static) + content placeholder (dynamic)."""
    if isinstance(history_data, list):
        history_data = {"trading_agent_reports": history_data}

    return html.Div(
        [
            # Above the filter bar and outside the filtered content: the
            # schedule is not history, and it must stay reachable when the
            # filters match nothing (that view early-returns an empty state).
            collapsible_section(
                "Scheduled Jobs", "scheduler",
                html.Div(id="scheduler-panel-container"),
                icon_class="bi-alarm", default_open=False,
            ),
            dcc.Interval(id="scheduler-refresh", interval=15_000),
            dcc.Store(id="scheduler-action-status"),
            build_history_filter_bar(history_data),
            html.Div(id="history-tab-content"),
        ],
        className="history-tab",
    )


def build_filtered_history_sections(history_data, filter_symbols, filter_date_range,
                                     specific_date=None, activity_scope="all"):
    """Build all history data sections with filtering and collapsible wrappers."""
    if isinstance(history_data, list):
        history_data = {"trading_agent_reports": history_data}

    reports = filter_items(history_data.get("reports", []), filter_symbols, filter_date_range, specific_date)
    predictions = filter_items(history_data.get("predictions", []), filter_symbols, filter_date_range, specific_date, date_key="prediction_date")
    ta_reports = filter_items(history_data.get("trading_agent_reports", []), filter_symbols, filter_date_range, specific_date)

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
        recommendations = filter_items(recommendations, [], "all", specific_date, date_key="created_at")
    elif filter_date_range and filter_date_range != "all":
        recommendations = filter_items(recommendations, [], filter_date_range, date_key="created_at")

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
        sections.append(collapsible_section(
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

        sections.append(collapsible_section(
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

        sections.append(collapsible_section(
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

        sections.append(collapsible_section(
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

        sections.append(collapsible_section(
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

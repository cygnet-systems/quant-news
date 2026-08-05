"""Home: what was predicted last, and how it turned out.

Opening the app used to show empty charts. The question it should answer on
arrival is the one you actually have: what is the current call on each name,
what did it cost or make, and what is still in flight.

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


def _cohort_table(cohort: dict) -> html.Div:
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

    rows = []
    for row in cohort["symbols"]:
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

    return html.Div(
        html.Table([header, html.Tbody(rows)], className="history-data-table"),
        className="history-table-wrap",
    )


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

    status = []
    if counts["pending"]:
        status.append(html.Span(f"{counts['pending']} awaiting close",
                                className="home-pending-pill"))
    if counts["resolved"] or counts["held"]:
        pnl_cls = ("positive" if cohort["pnl"] > 0
                   else "negative" if cohort["pnl"] < 0 else "")
        status.append(html.Span(
            f"{counts['resolved'] + counts['held']} scored · "
            f"${cohort['pnl']:+.2f}",
            className=f"num {pnl_cls}"))

    return html.Div(
        [
            html.Div(bits, className="home-meta-row"),
            html.Div(status, className="home-status-row"),
        ],
        className="home-card-header",
    )


def _open_predictions(open_preds: dict) -> html.Div:
    if not open_preds["total"]:
        return html.Div("Nothing in flight. Every call has been scored.",
                        className="home-empty-note")
    blocks = []
    for d in open_preds["dates"]:
        blocks.append(html.Div(
            [
                html.Div(
                    [
                        html.Span(d["target_date"], className="num home-open-date"),
                        html.Span(f"{len(d['predictions'])} calls · "
                                  f"{len(d['symbols'])} symbols",
                                  className="home-meta-label"),
                    ],
                    className="home-open-head",
                ),
                html.Div(", ".join(d["symbols"]), className="home-open-symbols"),
            ],
            className="home-open-block",
        ))
    return html.Div(blocks)


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


def _job_status(jobs: list[dict]) -> html.Div:
    if not jobs:
        return html.Div("No scheduled jobs configured.",
                        className="home-empty-note")
    rows = []
    for j in jobs:
        # scheduler_service returns a datetime here, not an ISO string.
        raw = j.get("last_run_at")
        last = raw.strftime("%Y-%m-%d %H:%M") if hasattr(raw, "strftime") else (
            str(raw)[:16].replace("T", " ") if raw else "never")
        status = j.get("last_status") or "never run"
        cls = ("negative" if status == "error"
               else "positive" if status == "success" else "")
        rows.append(html.Div(
            [
                html.Span(j.get("id", ""), className="home-job-name"),
                html.Span(f"{j.get('hour', 0):02d}:{j.get('minute', 0):02d} "
                          f"{j.get('days_of_week', '')}",
                          className="num home-meta-label"),
                html.Span(status, className=cls),
                html.Span(last, className="num home-meta-label"),
            ],
            className="home-job-row",
        ))
    return html.Div(rows)


def layout(cohort, open_preds, rolling, last_run, jobs, rolling_days=30) -> html.Div:
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

    def card(title, body, subtitle=None, action=None):
        return html.Div(
            [
                html.Div(
                    [
                        html.H2(title, className="home-card-title"),
                        html.Span(subtitle, className="home-card-subtitle")
                        if subtitle else "",
                        action if action is not None else "",
                    ],
                    className="home-card-titlerow",
                ),
                body,
            ],
            className="home-card",
        )

    return html.Div(
        [
            card(
                "Latest calls",
                html.Div([_last_run_header(cohort, last_run),
                          _cohort_table(cohort)]),
                subtitle="the most recent prediction cutoff",
                action=dbc.Button(
                    [html.I(className="bi bi-lightning-fill me-1"), "Run analysis"],
                    id="home-run-btn", size="sm", color="success", outline=True,
                ),
            ),
            html.Div(
                [
                    card("In flight", _open_predictions(open_preds),
                         subtitle="made, not yet scoreable"),
                    card(f"Last {rolling_days} days", _rolling(rolling, rolling_days),
                         subtitle="hit rate on BUY/SELL only",
                         action=dcc.Link("Performance", href="/performance",
                                         className="home-card-link")),
                ],
                className="home-two-col",
            ),
            card("Scheduled jobs", _job_status(jobs),
                 subtitle="the app's own clock",
                 action=dcc.Link("Schedule", href="/schedule",
                                 className="home-card-link")),
        ],
        className="page page-home",
    )

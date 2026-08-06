"""Performance page: the scorecard, and the per-call log beneath it.

Aggregates and the raw log live on one surface on purpose. Splitting them is
what produced the old Scoreboard-modal / History-tab duplication, where the
same numbers were reachable two ways and drifted.
"""

import dash_bootstrap_components as dbc
from dash import html

from layouts.formatters import MODEL_DISPLAY
from layouts.history_sections import (
    build_history_filter_bar,
    build_predictions_section,
    empty_history_message,
    filter_history_data,
)
from services.dashboard_service import aggregate_predictions

# Column definitions, surfaced as header tooltips. Several of these (Trades vs
# held, hit rate excluding HOLDs, the $1,000 notional behind P&L) are not
# guessable from the label alone.
SCOREBOARD_TIPS = {
    "Model": "The prediction model that produced these calls.",
    "Symbol": "The ticker these calls were made on.",
    "Trades": "BUY/SELL calls that took a position. HOLD days take no "
              "position and are counted separately as 'held'.",
    "Hit Rate": "Share of BUY/SELL calls where the next close moved the way "
                "the model predicted. HOLD days are excluded, since they "
                "cannot be right or wrong.",
    "Avg Conf": "The model's own average stated confidence. Compare it with "
                "Hit Rate: a model claiming 70% should be right ~70% of the "
                "time, and a large gap means it is miscalibrated.",
    "P&L": "Total dollars across all trades. Each takes a fixed $1,000 "
           "notional position, entered at the close and exited at the next "
           "close. Gross, before commission, spread and slippage.",
    "$/Trade": "P&L divided by Trades: the average edge per position. This "
               "is the number to judge, since a large P&L over many trades "
               "can still be a per-trade edge of roughly zero.",
}

SCOREBOARD_COLUMNS = ["Trades", "Hit Rate", "Avg Conf", "P&L", "$/Trade"]


def sb_th(label: str) -> html.Th:
    """Scoreboard header cell carrying its column definition as a tooltip."""
    return html.Th(label, title=SCOREBOARD_TIPS.get(label, ""),
                   className="scoreboard-th")


def scoreboard_header(first_label: str) -> html.Thead:
    return html.Thead(html.Tr(
        [sb_th(first_label)] + [sb_th(c) for c in SCOREBOARD_COLUMNS]))


def scoreboard_rows(groups: list[dict], group_key: str) -> list[html.Tr]:
    """Render aggregate rows from services.dashboard_service.aggregate_predictions."""
    rows = []
    for g in groups:
        trades, pnl, holds = g["trades"], g["pnl"], g["holds"]
        hit_rate = f"{g['hit_rate']:.0%}" if g["hit_rate"] is not None else "n/a"
        avg_conf = (f"{g['avg_confidence']:.0%}"
                    if g["avg_confidence"] is not None else "n/a")
        per_trade = (f"${g['pnl_per_trade']:+.2f}"
                     if g["pnl_per_trade"] is not None else "n/a")
        pnl_cls = "positive" if pnl > 0 else "negative" if pnl < 0 else ""
        display = (MODEL_DISPLAY.get(g["name"], g["name"])
                   if group_key == "model_name" else g["name"])
        # A symbol row is a question — "what did we actually call on this
        # name, and when was it right?" — so make it the way to ask it.
        first_cell = (
            html.Button(
                display,
                id={"type": "perf-symbol-drill", "symbol": g["name"]},
                className="perf-symbol-drill",
                title=f"Show only {g['name']}: every call, its date and outcome",
            )
            if group_key == "symbol" else display
        )
        rows.append(html.Tr([
            html.Td(first_cell),
            # "35 · 4 held" rather than "35 of 39": the bare "of" left the
            # denominator ambiguous (of what? trades? days?). Naming the
            # second number makes it self-describing.
            html.Td(html.Span([
                html.Strong(str(trades), className="num"),
                html.Span(f" · {holds} held", className="scoreboard-muted")
                if holds else "",
            ]), title=f"{trades} BUY/SELL positions taken, "
                      f"{holds} HOLD days took no position "
                      f"({g['scored']} predictions scored in total)"),
            html.Td(hit_rate, className="num",
                    title=f"{g['trade_hits']} of {trades} BUY/SELL calls "
                          f"went the predicted way (HOLDs excluded)"),
            html.Td(avg_conf, className="num"),
            html.Td(html.Span(f"${pnl:+.2f}", className=f"num {pnl_cls}"),
                    title=f"{trades} trades x $1,000 notional"),
            html.Td(html.Span(per_trade, className=f"num {pnl_cls}")),
        ]))
    return rows


def scoreboard_table(groups: list[dict], group_key: str,
                     first_label: str) -> html.Div:
    """A complete scorecard table, header included."""
    return html.Div(
        html.Table(
            [scoreboard_header(first_label),
             html.Tbody(scoreboard_rows(groups, group_key))],
            className="history-data-table",
        ),
        className="history-table-wrap",
    )


def scorecard(preds: list[dict]) -> html.Div:
    """Aggregates by model and by symbol, over whatever is in scope.

    "Evaluated" is not `was_correct is not None`: the evaluator leaves that
    None for a HOLD while still writing pnl_dollars, so keying on it alone
    would report scored HOLDs as pending forever.
    """
    evaluated = [p for p in preds
                 if p.get("was_correct") is not None or p.get("pnl_dollars") is not None]
    pending = len(preds) - len(evaluated)

    right = sum(1 for p in evaluated if p.get("was_correct") is True)
    wrong = sum(1 for p in evaluated if p.get("was_correct") is False)

    # Counts are the natural handles for "show me those" — reading a number
    # and then hunting for a filter that reproduces it is the long way round.
    def _chip(label, outcome, count, extra=""):
        return html.Button(
            f"{count} {label}",
            id={"type": "perf-outcome-chip", "outcome": outcome},
            className=f"perf-outcome-chip {extra}",
            disabled=not count,
            title=f"Show only {label}",
        )

    eval_bar = html.Div(
        [
            html.Div(
                [
                    _chip("awaiting their target session close", "pending",
                          pending, "perf-chip-pending"),
                    _chip("right", "right", right, "perf-chip-right"),
                    _chip("wrong", "wrong", wrong, "perf-chip-wrong"),
                    html.Button("Show all", id={"type": "perf-outcome-chip",
                                                "outcome": "all"},
                                className="perf-outcome-chip perf-chip-all"),
                ],
                className="perf-outcome-chips",
            ),
            dbc.Button(
                [html.I(className="bi bi-check2-square me-1"), "Evaluate pending"],
                id="perf-evaluate-btn", size="sm", outline=True,
                color="secondary", disabled=not pending,
            ),
        ],
        className="history-eval-bar",
    )

    if not evaluated:
        return html.Div([eval_bar, empty_history_message(False, "scored predictions")])

    # Range by TARGET date — the session whose close these were scored
    # against. Labelling it by prediction_date showed the data cutoff instead,
    # so a run made this morning for today's close read as "to 2026-08-04" and
    # looked like today was missing when all of it was present.
    dates = sorted(p.get("target_date") or p.get("prediction_date", "")
                   for p in evaluated)
    summary = html.Div(
        f"{len(evaluated)} scored predictions · "
        f"{len({p.get('symbol') for p in evaluated})} symbols · "
        f"sessions {dates[0]} to {dates[-1]}",
        className="scoreboard-summary",
    )
    hint = html.Div(
        [
            html.Div(
                "Trades counts BUY/SELL only. Each takes a fixed $1,000 notional "
                "position, held one session and closed at the next close. HOLD "
                "days take no position, score $0, and are excluded from hit rate.",
                className="history-scoreboard-hint",
            ),
            html.Div(
                "Hit rate against average confidence shows calibration: a model "
                "claiming 70% should be right about 70% of the time.",
                className="history-scoreboard-hint",
            ),
        ]
    )

    return html.Div([
        eval_bar,
        summary,
        hint,
        html.Div("By model", className="scoreboard-subtitle"),
        scoreboard_table(aggregate_predictions(evaluated, "model_name"),
                         "model_name", "Model"),
        html.Div("By symbol", className="scoreboard-subtitle"),
        scoreboard_table(aggregate_predictions(evaluated, "symbol"),
                         "symbol", "Symbol"),
    ])


def layout(history_data=None, filter_symbols=None, filter_date_range="all",
           specific_date=None, outcome="all") -> html.Div:
    """Scorecard on top, the per-call log beneath it.

    One surface on purpose. The aggregate used to be a modal and the log a
    History section, so the same numbers were reachable two ways, each with
    its own Evaluate button.

    Everything filter-dependent lives inside #archive-body so a filter change
    rebuilds only that, never the router. Routing on its own Input is what
    keeps a slow filter response from overwriting a newer page.
    """
    return html.Div(
        [
            build_history_filter_bar(history_data, filter_symbols,
                                     filter_date_range, specific_date),
            html.Div(
                body(history_data, filter_symbols, filter_date_range,
                     specific_date, outcome),
                id="archive-body",
            ),
        ],
        className="page page-performance",
    )


def body(history_data=None, filter_symbols=None, filter_date_range="all",
         specific_date=None, outcome="all") -> list:
    history_data = history_data or {}
    buckets = filter_history_data(history_data, filter_symbols,
                                  filter_date_range, specific_date)
    preds = buckets["predictions"]

    # The scorecard always describes the full scope; only the log narrows.
    # Filtering the aggregates too would make "wrong" show a 0% hit rate,
    # which is arithmetic rather than information.
    log_preds = _by_outcome(preds, outcome)
    log = build_predictions_section(log_preds, deferred=True)

    subtitle = "Prediction log"
    if outcome != "all":
        label = {"pending": "awaiting close", "right": "correct",
                 "wrong": "incorrect"}.get(outcome, outcome)
        subtitle = f"Prediction log · {label} only ({len(log_preds)})"

    return [
        scorecard(preds),
        html.Div(subtitle, className="scoreboard-subtitle") if log else None,
        log,
    ]


def _by_outcome(preds: list[dict], outcome: str) -> list[dict]:
    """Slice the log by how a prediction turned out.

    Pending is "no verdict AND no P&L" for the same reason the scorecard uses
    it: a scored HOLD has pnl_dollars 0.0 and was_correct None, so keying on
    the verdict alone would file every scored HOLD as still waiting.
    """
    if outcome == "pending":
        return [p for p in preds if p.get("was_correct") is None
                and p.get("pnl_dollars") is None]
    if outcome == "right":
        return [p for p in preds if p.get("was_correct") is True]
    if outcome == "wrong":
        return [p for p in preds if p.get("was_correct") is False]
    return preds

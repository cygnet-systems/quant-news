"""Performance page: the scorecard, and the per-call log beneath it.

Aggregates and the raw log live on one surface on purpose. Splitting them is
what produced the old Scoreboard-modal / History-tab duplication, where the
same numbers were reachable two ways and drifted.
"""

from dash import html

from layouts.formatters import MODEL_DISPLAY

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
        rows.append(html.Tr([
            html.Td(display),
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

"""Performance page: the scorecard, and the per-call log beneath it.

Aggregates and the raw log live on one surface on purpose. Splitting them is
what produced the old Scoreboard-modal / History-tab duplication, where the
same numbers were reachable two ways and drifted.
"""

import dash_bootstrap_components as dbc
from dash import dcc, html

from layouts.components import create_data_actions
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
                "the model predicted, with its ± standard error. Bold means "
                "statistically distinguishable from a coin flip (2 SE); "
                "anything else is noise at this sample size, however good "
                "it looks.",
    "Calibration": "Claimed vs delivered: the model's average stated "
                   "confidence against its actual hit rate. Red means it "
                   "claims at least 15 points more than it delivers — its "
                   "confidence cannot be used for sizing.",
    "P&L": "Total dollars across all trades at $1,000 notional each, gross "
           "(before friction).",
    "Net P&L": "P&L minus an estimated round-trip friction haircut per trade "
               "(8-40 bps by price bucket: cheaper stocks trade wider). The "
               "number a real account would keep.",
    "$/Trade": "Gross P&L divided by Trades: the average edge per position. "
               "Judge this together with Net P&L — many small wins can "
               "vanish into friction.",
}

SCOREBOARD_COLUMNS = ["Trades", "Hit Rate", "Calibration", "P&L", "Net P&L", "$/Trade"]


def sb_th(label: str) -> html.Th:
    """Scoreboard header cell carrying its column definition as a tooltip."""
    return html.Th(label, title=SCOREBOARD_TIPS.get(label, ""),
                   className="scoreboard-th")


def scoreboard_header(first_label: str) -> html.Thead:
    return html.Thead(html.Tr(
        [sb_th(first_label)] + [sb_th(c) for c in SCOREBOARD_COLUMNS]))


def trade_detail_table(trades: list[dict]) -> html.Table:
    """Every scored call behind one symbol row, newest session first."""

    def _row(p):
        decision = p.get("decision", "HOLD")
        dec_cls = ("positive" if decision == "BUY"
                   else "negative" if decision == "SELL" else "neutral")
        conf = p.get("confidence")
        correct = p.get("was_correct")
        if correct is True:
            result, result_cls = "✓ right", "positive"
        elif correct is False:
            result, result_cls = "✗ wrong", "negative"
        else:
            # A scored HOLD: it has P&L (0) but no directional verdict.
            result, result_cls = "—", ""
        pnl = p.get("pnl_dollars")
        pnl_cls = ("positive" if pnl and pnl > 0
                   else "negative" if pnl and pnl < 0 else "")
        return html.Tr([
            html.Td(p.get("target_date") or p.get("prediction_date", "?"),
                    title=f"As-of {p.get('prediction_date', '?')}, scored "
                          f"against the {p.get('target_date', '?')} close"),
            html.Td(MODEL_DISPLAY.get(p.get("model_name", ""),
                                      p.get("model_name", ""))),
            html.Td(html.Span(decision, className=f"history-decision {dec_cls}")),
            html.Td(f"{int(conf * 100)}%" if conf else "—", className="num"),
            html.Td(html.Span(result, className=result_cls)),
            html.Td(html.Span(f"${pnl:+.2f}" if pnl is not None else "—",
                              className=f"num {pnl_cls}")),
        ])

    # Date descending, model ascending: sort by model, then stably by date.
    ordered = sorted(trades, key=lambda p: p.get("model_name") or "")
    ordered.sort(key=lambda p: p.get("target_date")
                 or p.get("prediction_date") or "", reverse=True)
    return html.Table([
        html.Thead(html.Tr([html.Th(h) for h in
                            ["Date", "Model", "Signal", "Conf", "Result", "P&L"]])),
        html.Tbody([_row(p) for p in ordered]),
    ], className="history-data-table history-pred-table")


def scoreboard_rows(groups: list[dict], group_key: str,
                    trades_by_symbol: dict | None = None) -> list[html.Tr]:
    """Render aggregate rows from services.dashboard_service.aggregate_predictions.

    With trades_by_symbol (symbol rows only), each row also gets a hidden
    sibling <tr> holding its individual trades, toggled by the chevron via the
    generic history-section-toggle callback (section "sbrow-{sym}").
    """
    rows = []
    for g in groups:
        trades, pnl, holds = g["trades"], g["pnl"], g["holds"]
        per_trade = (f"${g['pnl_per_trade']:+.2f}"
                     if g["pnl_per_trade"] is not None else "n/a")
        pnl_cls = "positive" if pnl > 0 else "negative" if pnl < 0 else ""
        display = (MODEL_DISPLAY.get(g["name"], g["name"])
                   if group_key == "model_name" else g["name"])

        # Hit rate with its standard error; bold only when the rate is
        # 2 SE away from a coin flip. Everything else is presented as the
        # noise it statistically is.
        if g["hit_rate"] is not None:
            se = g.get("hit_se")
            hit_txt = (f"{g['hit_rate']:.0%} ±{se:.0%}" if se is not None
                       else f"{g['hit_rate']:.0%}")
            hit_cell = (html.Strong(hit_txt, title="Distinguishable from a "
                                    "coin flip at 2 standard errors")
                        if g.get("significant") else
                        html.Span(hit_txt, className="scoreboard-muted",
                                  title="Within 2 SE of 50% — statistically "
                                        "indistinguishable from chance at "
                                        "this sample size"))
        else:
            hit_cell = html.Span("n/a")

        # Calibration: claimed → delivered, red when the gap is a lie.
        if g["avg_confidence"] is not None and g["hit_rate"] is not None:
            gap = g["avg_confidence"] - g["hit_rate"]
            cal_cls = "negative" if gap > 0.15 else ""
            cal_cell = html.Span(
                f"{g['avg_confidence']:.0%}→{g['hit_rate']:.0%}",
                className=f"num {cal_cls}",
                title=f"Claims {g['avg_confidence']:.0%} on average, delivers "
                      f"{g['hit_rate']:.0%}"
                      + (" — overconfident by more than 15 points; its stated"
                         " confidence cannot size positions" if gap > 0.15
                         else ""),
            )
        else:
            cal_cell = html.Span("n/a")

        net = g.get("net_pnl")
        net_cls = ("positive" if net and net > 0
                   else "negative" if net and net < 0 else "")
        net_cell = (html.Span(f"${net:+.2f}", className=f"num {net_cls}",
                              title=f"Gross ${pnl:+.2f} minus "
                                    f"~${g.get('est_costs', 0):.2f} estimated "
                                    f"friction")
                    if net is not None else html.Span("n/a"))

        conc = g.get("concentration")
        conc_flag = (html.Span(
            f" ⚠ {conc:.0%} one name",
            className="scoreboard-muted",
            title="This share of gross P&L comes from a single symbol — "
                  "an edge that is one ticker is a position, not a strategy",
        ) if conc is not None and conc > 0.4 and trades >= 10 else "")
        # A symbol row is a question — "what did we actually call on this
        # name, and when was it right?" — so make it the way to ask it.
        sym_trades = (trades_by_symbol or {}).get(g["name"]) \
            if group_key == "symbol" else None
        first_cell = (
            html.Button(
                display,
                id={"type": "perf-symbol-drill", "symbol": g["name"]},
                className="perf-symbol-drill",
                title=f"Show only {g['name']}: every call, its date and outcome",
            )
            if group_key == "symbol" else display
        )
        if sym_trades:
            sect = f"sbrow-{g['name']}"
            first_cell = html.Div([
                first_cell,
                html.Span(
                    html.I(className="bi bi-chevron-down history-chevron ms-auto",
                           id={"type": "history-section-chevron", "section": sect}),
                    id={"type": "history-section-toggle", "section": sect},
                    className="perf-row-expand",
                    title=f"Show each {g['name']} trade in place",
                ),
            ], className="perf-symbol-cell")
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
            html.Td(hit_cell, className="num"),
            html.Td(cal_cell, className="num"),
            html.Td(html.Span([
                html.Span(f"${pnl:+.2f}", className=f"num {pnl_cls}"),
                conc_flag,
            ]), title=f"{trades} trades x $1,000 notional, gross"),
            html.Td(net_cell),
            html.Td(html.Span(per_trade, className=f"num {pnl_cls}")),
        ]))
        if sym_trades:
            rows.append(html.Tr(
                html.Td(
                    html.Div(trade_detail_table(sym_trades),
                             id={"type": "history-section-body", "section": sect},
                             className="history-section-body perf-trades-body",
                             style={"display": "none"}),
                    colSpan=len(SCOREBOARD_COLUMNS) + 1,
                ),
                className="perf-trades-row",
            ))
    return rows


def scoreboard_table(groups: list[dict], group_key: str, first_label: str,
                     trades_by_symbol: dict | None = None) -> html.Div:
    """A complete scorecard table, header included."""
    return html.Div(
        html.Table(
            [scoreboard_header(first_label),
             html.Tbody(scoreboard_rows(groups, group_key, trades_by_symbol))],
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
    trades_by_symbol: dict = {}
    for p in evaluated:
        trades_by_symbol.setdefault(p.get("symbol", "?"), []).append(p)
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
                         "symbol", "Symbol",
                         trades_by_symbol=trades_by_symbol),
    ])


def model_filter_row(history_data, model="all") -> html.Div:
    """The model dropdown, seeded from its store rather than written back.

    Lives outside #archive-body for the same reason the filter bar does: a
    rebuilt dropdown re-fires its value Input, which writes the store, which
    would rebuild whatever contains it — a render loop.
    """
    preds = (history_data or {}).get("predictions", [])
    names = sorted({p.get("model_name") for p in preds if p.get("model_name")})
    if model not in ("all", None) and model not in names:
        # Keep a selection with no rows visible, so it can be changed away.
        names.append(model)
    return html.Div(
        [
            html.Span("Model:", className="history-date-label"),
            dcc.Dropdown(
                id="history-model-dropdown",
                options=[{"label": "All models", "value": "all"}]
                + [{"label": MODEL_DISPLAY.get(n, n), "value": n} for n in names],
                value=model or "all",
                clearable=False,
                searchable=False,
                className="history-model-dropdown",
            ),
        ],
        className="history-filter-row perf-model-row",
    )


def layout(history_data=None, filter_symbols=None, filter_date_range="all",
           specific_date=None, outcome="all", model="all") -> html.Div:
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
            html.Div(
                create_data_actions(include_refresh=False),
                className="perf-action-bar",
            ),
            build_history_filter_bar(history_data, filter_symbols,
                                     filter_date_range, specific_date),
            model_filter_row(history_data, model),
            html.Div(
                body(history_data, filter_symbols, filter_date_range,
                     specific_date, outcome, model),
                id="archive-body",
            ),
        ],
        className="page page-performance",
    )


def body(history_data=None, filter_symbols=None, filter_date_range="all",
         specific_date=None, outcome="all", model="all") -> list:
    history_data = history_data or {}
    buckets = filter_history_data(history_data, filter_symbols,
                                  filter_date_range, specific_date)
    preds = _by_model(buckets["predictions"], model)

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


def _by_model(preds: list[dict], model: str) -> list[dict]:
    """Narrow to one model. Unlike the outcome slice, this filters the
    scorecard too: "how is Kronos doing?" is a question about the aggregates,
    not just the log."""
    if not model or model == "all":
        return preds
    return [p for p in preds if p.get("model_name") == model]


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

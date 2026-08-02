"""Strategy performance UI components.

Renders strategy evaluation results: performance cards, comparison table,
and equity curves. Follows the signal_components.py pattern.
"""

import dash_bootstrap_components as dbc
from dash import html

from config import COLORS


def create_strategy_section(metrics: list[dict], evaluations: list[dict]) -> html.Div:
    """Create the complete strategy performance section.

    Args:
        metrics: List of strategy_metrics dicts from cache.
        evaluations: List of recent strategy evaluations from cache.

    Returns:
        Strategy performance section, or empty div if no data.
    """
    if not metrics and not evaluations:
        return html.Div()

    children = [
        html.Div(
            [
                html.I(className="bi bi-bar-chart-line", style={"marginRight": "6px"}),
                "Strategy Performance",
            ],
            className="section-title",
        ),
    ]

    if metrics:
        children.append(_create_metrics_cards(metrics))

    if evaluations:
        children.append(_create_evaluations_table(evaluations))

    return html.Div(children, className="strategy-section")


def _create_metrics_cards(metrics: list[dict]) -> html.Div:
    """Create performance metric cards for each strategy."""
    cards = []
    for m in metrics:
        strategy = m.get("strategy_name", "")
        symbol = m.get("symbol", "all")
        total_trades = m.get("total_trades", 0) or 0

        if total_trades == 0:
            continue

        sharpe = m.get("sharpe_ratio")
        win_rate = m.get("win_rate")
        total_pnl = m.get("total_pnl")
        max_dd = m.get("max_drawdown")

        # Card color based on Sharpe
        if sharpe is not None and sharpe > 0:
            border_color = COLORS.POSITIVE
        elif sharpe is not None and sharpe < 0:
            border_color = COLORS.NEGATIVE
        else:
            border_color = COLORS.NEUTRAL

        display_name = _strategy_display_name(strategy)

        stat_rows = []

        if sharpe is not None:
            stat_rows.append(
                _stat_row("Sharpe", f"{sharpe:.2f}", _metric_color(sharpe, 0))
            )
        if win_rate is not None:
            stat_rows.append(
                _stat_row("Win Rate", f"{win_rate:.1f}%", _metric_color(win_rate, 50))
            )
        if total_pnl is not None:
            stat_rows.append(
                _stat_row("Return", f"{total_pnl:+.1f}%", _metric_color(total_pnl, 0))
            )
        if max_dd is not None:
            stat_rows.append(
                _stat_row("Max DD", f"{abs(max_dd):.1f}%", COLORS.NEGATIVE)
            )

        stat_rows.append(
            _stat_row("Trades", str(total_trades), COLORS.TEXT_SECONDARY)
        )

        cards.append(
            html.Div(
                [
                    html.Div(
                        [
                            html.Span(display_name, className="strategy-card-name"),
                            html.Span(
                                symbol,
                                className="strategy-card-symbol",
                            ),
                        ],
                        className="strategy-card-header",
                    ),
                    html.Div(stat_rows, className="strategy-card-stats"),
                ],
                className="strategy-card",
                style={"borderLeft": f"3px solid {border_color}"},
            )
        )

    if not cards:
        return html.Div()

    return html.Div(cards, className="strategy-cards-grid")


def _create_evaluations_table(evaluations: list[dict]) -> html.Div:
    """Create recent strategy evaluations table."""
    rows = []
    for ev in evaluations[:15]:
        action = ev.get("action", "SKIP")
        was_correct = ev.get("was_correct")
        pnl = ev.get("pnl_dollars")
        strategy = ev.get("strategy_name", "")
        model = ev.get("model_name", "")
        date_str = str(ev.get("target_date", ""))[:10]

        if action == "SKIP":
            continue

        # Result icon
        if was_correct is True:
            result_icon = "bi-check-circle-fill"
            result_color = COLORS.POSITIVE
        elif was_correct is False:
            result_icon = "bi-x-circle-fill"
            result_color = COLORS.NEGATIVE
        else:
            result_icon = "bi-clock"
            result_color = COLORS.TEXT_MUTED

        pnl_text = f"${pnl:+.0f}" if pnl is not None else "—"
        pnl_color = (
            COLORS.POSITIVE if pnl and pnl > 0
            else COLORS.NEGATIVE if pnl and pnl < 0
            else COLORS.TEXT_MUTED
        )

        action_color = (
            COLORS.POSITIVE if action == "BUY"
            else COLORS.NEGATIVE if action == "SELL"
            else COLORS.TEXT_MUTED
        )

        rows.append(
            html.Tr([
                html.Td(date_str, style={"fontSize": "11px"}),
                html.Td(
                    _strategy_display_name(strategy),
                    style={"fontSize": "11px"},
                ),
                html.Td(model, style={"fontSize": "11px"}),
                html.Td(
                    action,
                    style={"fontSize": "11px", "fontWeight": "600", "color": action_color},
                ),
                html.Td(
                    html.I(className=f"bi {result_icon}", style={"color": result_color}),
                ),
                html.Td(pnl_text, style={"color": pnl_color, "fontSize": "11px"}),
            ])
        )

    if not rows:
        return html.Div()

    return html.Div(
        [
            html.Div("Recent Evaluations", className="section-title",
                      style={"marginTop": "12px"}),
            html.Table(
                [
                    html.Thead(
                        html.Tr([
                            html.Th("Date"),
                            html.Th("Strategy"),
                            html.Th("Model"),
                            html.Th("Action"),
                            html.Th(""),
                            html.Th("P&L"),
                        ])
                    ),
                    html.Tbody(rows),
                ],
                className="prediction-history-table",
            ),
        ],
    )


def _stat_row(label: str, value: str, color: str) -> html.Div:
    """Create a single stat row for a strategy card."""
    return html.Div(
        [
            html.Span(label, className="strategy-stat-label"),
            html.Span(value, className="strategy-stat-value", style={"color": color}),
        ],
        className="strategy-stat-row",
    )


def _metric_color(value: float, threshold: float) -> str:
    """Return green/red based on value vs threshold."""
    if value > threshold:
        return COLORS.POSITIVE
    elif value < threshold:
        return COLORS.NEGATIVE
    return COLORS.TEXT_SECONDARY


def _strategy_display_name(name: str) -> str:
    """Convert strategy name to display name."""
    return {
        "directional": "Directional",
        "confidence_threshold": "High Confidence",
        "ensemble_vote": "Ensemble Vote",
    }.get(name, name.replace("_", " ").title())

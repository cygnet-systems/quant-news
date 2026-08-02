"""Model signal UI components for per-symbol tabs.

Renders prediction cards for each model (Kronos, XGBoost, LightGBM,
DeBERTa, TradingAgents) with decision, confidence, and model-specific details.
The Ensemble card is full-width with a weight composition bar and vote summary.
"""

import dash_bootstrap_components as dbc
from dash import dcc, html

from config import COLORS

# Display name mapping
_DISPLAY_NAMES = {
    "recommendation_synthesis": "Recommendations (LLM)",
    "kronos_mini": "Kronos",
    "xgboost_shap": "XGBoost",
    "lightgbm": "LightGBM",
    "deberta_sentiment": "DeBERTa",
    "trading_agents": "TradingAgents",
    "ensemble": "Ensemble",
}

# Colors for the ensemble weight bar segments
_MODEL_COLORS = {
    "kronos_mini": "#00D4AA",
    "xgboost_shap": "#7B61FF",
    "lightgbm": "#4ECDC4",
    "deberta_sentiment": "#FFE66D",
    "trading_agents": "#FF6B6B",
}

# Short labels for vote summary
_MODEL_SHORT = {
    "kronos_mini": "K",
    "xgboost_shap": "XG",
    "lightgbm": "LG",
    "deberta_sentiment": "D",
    "trading_agents": "TA",
}


def create_signal_cards(model_signals: dict, symbol: str = "") -> html.Div:
    """Create signal cards for all models in a symbol's prediction.

    Individual model cards go in a 2-column grid.
    The ensemble card spans full width below.
    """
    if not model_signals:
        return html.Div()

    individual_cards = []
    ensemble_card = None

    for model_name, result in model_signals.items():
        if model_name == "ensemble":
            ensemble_card = _create_ensemble_card(result)
        elif model_name == "trading_agents":
            individual_cards.append(_create_trading_agents_card(result, symbol))
        else:
            individual_cards.append(_create_model_card(model_name, result))

    children = [
        html.Div("Model Predictions", className="section-title"),
        html.Div(individual_cards, className="signal-cards-grid"),
    ]

    if ensemble_card is not None:
        children.append(ensemble_card)

    return html.Div(children, className="signal-cards-section")


def _decision_colors(decision: str, error: str | None):
    """Return (color, bg_color) for a decision."""
    if error:
        return COLORS.TEXT_MUTED, COLORS.BG_TERTIARY
    if decision == "BUY":
        return COLORS.POSITIVE, COLORS.POSITIVE_MUTED
    if decision == "SELL":
        return COLORS.NEGATIVE, COLORS.NEGATIVE_MUTED
    return COLORS.NEUTRAL, COLORS.BG_TERTIARY


def _create_model_card(model_name: str, result: dict) -> html.Div:
    """Create a single model signal card."""
    decision = result.get("decision", "HOLD")
    confidence = result.get("confidence", 0)
    up_prob = result.get("up_probability", 0.5)
    error = result.get("error")
    details = result.get("details", {})

    display_name = _DISPLAY_NAMES.get(model_name, model_name)
    decision_color, decision_bg = _decision_colors(decision, error)

    # Confidence label varies by model
    confidence_type = details.get("confidence_type", "")
    if confidence_type.startswith("empirical_reliability"):
        confidence_label = "Realized reliability"
    elif confidence_type == "undeclared_neutral":
        confidence_label = "Directional (no track record)"
    elif confidence_type == "self_reported":
        confidence_label = "Model Certainty"
    else:
        confidence_label = "Confidence"

    # Build card content
    card_children = [
        html.Div(
            [
                html.Span(display_name, className="signal-card-model-name"),
                html.Span(
                    decision if not error else "ERROR",
                    className="signal-card-decision",
                    style={"color": decision_color, "backgroundColor": decision_bg},
                ),
            ],
            className="signal-card-header",
        ),
    ]

    if error:
        card_children.append(
            html.Div(
                error[:80],
                className="signal-card-error",
                style={"color": COLORS.TEXT_MUTED, "fontSize": "11px"},
            )
        )
    else:
        # Confidence bar
        conf_pct = int(confidence * 100)
        card_children.append(
            html.Div(
                [
                    html.Div(
                        [
                            html.Span(confidence_label, className="signal-card-conf-label"),
                            html.Span(f"{conf_pct}%", className="signal-card-conf-value"),
                        ],
                        className="signal-card-conf-row",
                    ),
                    html.Div(
                        html.Div(
                            style={
                                "width": f"{conf_pct}%",
                                "backgroundColor": decision_color,
                                "height": "100%",
                                "borderRadius": "2px",
                                "transition": "width 0.3s ease",
                            },
                        ),
                        className="signal-card-conf-bar-bg",
                    ),
                ],
                className="signal-card-confidence",
            )
        )

        # Up probability
        up_pct = int(up_prob * 100)
        card_children.append(
            html.Div(
                [
                    html.Span("Up Prob:", className="signal-card-detail-label"),
                    html.Span(f"{up_pct}%", className="signal-card-detail-value"),
                ],
                className="signal-card-detail-row",
            )
        )

        # Model-specific extras
        if details.get("reasoning"):
            reasoning = details["reasoning"]
            if len(reasoning) > 120:
                reasoning = reasoning[:117] + "..."
            card_children.append(
                html.Div(reasoning, className="signal-card-reasoning")
            )

        if details.get("predicted_close"):
            card_children.append(
                html.Div(
                    [
                        html.Span("Pred Close:", className="signal-card-detail-label"),
                        html.Span(
                            f"${details['predicted_close']:.2f}",
                            className="signal-card-detail-value",
                        ),
                    ],
                    className="signal-card-detail-row",
                )
            )

        if details.get("training_samples"):
            card_children.append(
                html.Div(
                    [
                        html.Span("Samples:", className="signal-card-detail-label"),
                        html.Span(
                            str(details["training_samples"]),
                            className="signal-card-detail-value",
                        ),
                    ],
                    className="signal-card-detail-row",
                )
            )

        # DeBERTa-specific: articles analyzed
        if details.get("articles_relevant") is not None and model_name == "deberta_sentiment":
            card_children.append(
                html.Div(
                    [
                        html.Span("Articles:", className="signal-card-detail-label"),
                        html.Span(
                            f"{details['articles_relevant']}/{details.get('articles_total', 0)}",
                            className="signal-card-detail-value",
                        ),
                    ],
                    className="signal-card-detail-row",
                )
            )
            if details.get("avg_positive_ratio") is not None:
                card_children.append(
                    html.Div(
                        [
                            html.Span("Sentiment:", className="signal-card-detail-label"),
                            html.Span(
                                f"{details['avg_positive_ratio']:.0%} positive",
                                className="signal-card-detail-value",
                            ),
                        ],
                        className="signal-card-detail-row",
                    )
                )

    return html.Div(
        card_children,
        className="signal-card",
        style={"borderLeft": f"3px solid {decision_color}"},
    )


def _create_trading_agents_card(result: dict, symbol: str = "") -> html.Div:
    """Create a TradingAgents signal card with report preview."""
    decision = result.get("decision", "HOLD")
    confidence = result.get("confidence", 0)
    error = result.get("error")
    details = result.get("details", {})

    decision_color, decision_bg = _decision_colors(decision, error)

    card_children = [
        html.Div(
            [
                html.Span("TradingAgents", className="signal-card-model-name"),
                html.Span(
                    decision if not error else "ERROR",
                    className="signal-card-decision",
                    style={"color": decision_color, "backgroundColor": decision_bg},
                ),
            ],
            className="signal-card-header",
        ),
    ]

    if error:
        card_children.append(
            html.Div(
                error[:80],
                className="signal-card-error",
                style={"color": COLORS.TEXT_MUTED, "fontSize": "11px"},
            )
        )
    else:
        # Confidence bar
        conf_pct = int(confidence * 100)
        card_children.append(
            html.Div(
                [
                    html.Div(
                        [
                            html.Span(
                                "Research Confidence",
                                className="signal-card-conf-label",
                            ),
                            html.Span(f"{conf_pct}%", className="signal-card-conf-value"),
                        ],
                        className="signal-card-conf-row",
                    ),
                    html.Div(
                        html.Div(
                            style={
                                "width": f"{conf_pct}%",
                                "backgroundColor": decision_color,
                                "height": "100%",
                                "borderRadius": "2px",
                                "transition": "width 0.3s ease",
                            },
                        ),
                        className="signal-card-conf-bar-bg",
                    ),
                ],
                className="signal-card-confidence",
            )
        )

        # Token usage
        input_tokens = details.get("input_tokens", 0)
        output_tokens = details.get("output_tokens", 0)
        if input_tokens or output_tokens:
            card_children.append(
                html.Div(
                    [
                        html.Span("Tokens:", className="signal-card-detail-label"),
                        html.Span(
                            f"{input_tokens:,} in / {output_tokens:,} out",
                            className="signal-card-detail-value",
                        ),
                    ],
                    className="signal-card-detail-row",
                )
            )

        # Report preview: the Verdict block (new reports lead with it), i.e.
        # everything before the first section heading. Legacy free-form
        # reports fall back to the old fixed slice.
        raw_response = details.get("raw_response", "")
        if raw_response:
            cut = raw_response.find("\n### ")
            if 0 < cut <= 900:
                preview = raw_response[:cut].strip()
            else:
                preview = raw_response[:300]
                if len(raw_response) > 300:
                    preview += "..."
            card_children.append(
                html.Div(
                    dcc.Markdown(
                        preview,
                        style={"fontSize": "0.75rem", "lineHeight": "1.3"},
                    ),
                    className="ta-report-preview",
                )
            )
            card_children.append(
                html.Button(
                    [
                        html.I(className="bi bi-file-text me-1"),
                        "View Full Report",
                    ],
                    id={"type": "view-full-report-btn", "symbol": symbol},
                    className="view-full-report-btn",
                    n_clicks=0,
                )
            )

    return html.Div(
        card_children,
        className="signal-card",
        style={"borderLeft": f"3px solid {decision_color}"},
    )


def _create_ensemble_card(result: dict) -> html.Div:
    """Create the full-width ensemble card with weight bar and vote summary."""
    decision = result.get("decision", "HOLD")
    confidence = result.get("confidence", 0)
    error = result.get("error")
    details = result.get("details", {})

    decision_color, decision_bg = _decision_colors(decision, error)

    # Header with gear icon
    header = html.Div(
        [
            html.Div(
                [
                    html.Span("Ensemble", className="signal-card-model-name"),
                    html.Span(
                        decision if not error else "ERROR",
                        className="signal-card-decision",
                        style={"color": decision_color, "backgroundColor": decision_bg},
                    ),
                ],
                style={"display": "flex", "alignItems": "center", "gap": "8px"},
            ),
            html.Button(
                html.I(className="bi bi-gear"),
                id="ensemble-config-btn",
                className="ensemble-gear-btn",
                n_clicks=0,
            ),
        ],
        className="signal-card-header ensemble-card-header",
    )

    card_children = [header]

    if error:
        card_children.append(
            html.Div(
                error[:120],
                className="signal-card-error",
                style={"color": COLORS.TEXT_MUTED, "fontSize": "11px"},
            )
        )
    else:
        # Confidence bar
        conf_pct = int(confidence * 100)
        card_children.append(
            html.Div(
                [
                    html.Div(
                        [
                            html.Span("Confidence", className="signal-card-conf-label"),
                            html.Span(f"{conf_pct}%", className="signal-card-conf-value"),
                        ],
                        className="signal-card-conf-row",
                    ),
                    html.Div(
                        html.Div(
                            style={
                                "width": f"{conf_pct}%",
                                "backgroundColor": decision_color,
                                "height": "100%",
                                "borderRadius": "2px",
                                "transition": "width 0.3s ease",
                            },
                        ),
                        className="signal-card-conf-bar-bg",
                    ),
                ],
                className="signal-card-confidence",
            )
        )

        # Weight composition bar
        weights_used = details.get("weights_used", {})
        if weights_used:
            card_children.append(_create_weight_bar(weights_used))

        # Vote summary row
        votes = details.get("votes", {})
        models_excluded = details.get("models_excluded", [])
        if votes or models_excluded:
            card_children.append(_create_vote_summary(votes, models_excluded))

        # Score
        normalized = details.get("normalized_score")
        if normalized is not None:
            card_children.append(
                html.Div(
                    [
                        html.Span("Score:", className="signal-card-detail-label"),
                        html.Span(
                            f"{normalized:+.3f}",
                            className="signal-card-detail-value",
                            style={"color": decision_color},
                        ),
                    ],
                    className="signal-card-detail-row",
                )
            )

    return html.Div(
        card_children,
        className="signal-card ensemble-card",
        style={"borderLeft": f"3px solid {decision_color}"},
    )


def _create_weight_bar(weights_used: dict) -> html.Div:
    """Create a horizontal weight composition bar.

    Pure CSS flex bar — each segment's flex-grow equals the model's weight.
    Color-coded per model with initials inside.
    """
    total = sum(weights_used.values())
    if total == 0:
        return html.Div()

    segments = []
    for model_name, weight in weights_used.items():
        color = _MODEL_COLORS.get(model_name, "#888")
        short = _MODEL_SHORT.get(model_name, model_name[:2])
        pct = (weight / total) * 100

        segments.append(
            html.Div(
                short,
                className="ensemble-weight-segment",
                style={
                    "flexGrow": str(weight),
                    "backgroundColor": color,
                },
                title=f"{_DISPLAY_NAMES.get(model_name, model_name)}: {weight:.1f}",
            )
        )

    return html.Div(
        [
            html.Div("Weight Mix", className="signal-card-detail-label",
                      style={"marginBottom": "4px"}),
            html.Div(segments, className="ensemble-weight-bar"),
        ],
        style={"marginTop": "6px"},
    )


def _create_vote_summary(votes: dict, excluded: list) -> html.Div:
    """Create compact vote summary: K: BUY | XG: SELL | ..."""
    spans = []
    # Active votes
    for model_name, vote in votes.items():
        short = _MODEL_SHORT.get(model_name, model_name[:2])
        vote_color, _ = _decision_colors(vote, None)
        if spans:
            spans.append(html.Span(" | ", style={"color": COLORS.TEXT_MUTED}))
        spans.append(
            html.Span(
                f"{short}: {vote}",
                style={"color": vote_color, "fontWeight": "600"},
            )
        )

    # Excluded (dimmed)
    for model_name in excluded:
        short = _MODEL_SHORT.get(model_name, model_name[:2])
        if spans:
            spans.append(html.Span(" | ", style={"color": COLORS.TEXT_MUTED}))
        spans.append(
            html.Span(
                f"{short}: OFF",
                style={
                    "color": COLORS.TEXT_MUTED,
                    "textDecoration": "line-through",
                    "fontSize": "10px",
                },
            )
        )

    return html.Div(
        spans,
        className="ensemble-vote-summary",
    )


def create_prediction_history_table(history: list[dict]) -> html.Div:
    """Create a compact prediction history table."""
    if not history:
        return html.Div()

    rows = []
    for pred in history[:10]:
        decision = pred.get("decision", "")
        was_correct = pred.get("was_correct")
        pnl = pred.get("pnl_dollars")

        if was_correct is True:
            result_icon = "bi-check-circle-fill"
            result_color = COLORS.POSITIVE
        elif was_correct is False:
            result_icon = "bi-x-circle-fill"
            result_color = COLORS.NEGATIVE
        else:
            result_icon = "bi-clock"
            result_color = COLORS.TEXT_MUTED

        pnl_text = f"${pnl:+.0f}" if pnl is not None else "\u2014"
        pnl_color = (
            COLORS.POSITIVE if pnl and pnl > 0 else
            COLORS.NEGATIVE if pnl and pnl < 0 else
            COLORS.TEXT_MUTED
        )

        rows.append(
            html.Tr([
                html.Td(str(pred.get("prediction_date", ""))[:10], style={"fontSize": "11px"}),
                html.Td(pred.get("model_name", ""), style={"fontSize": "11px"}),
                html.Td(decision, style={"fontSize": "11px", "fontWeight": "600"}),
                html.Td(
                    html.I(className=f"bi {result_icon}", style={"color": result_color}),
                ),
                html.Td(pnl_text, style={"color": pnl_color, "fontSize": "11px"}),
            ])
        )

    return html.Div(
        [
            html.Div("Prediction History", className="section-title"),
            html.Table(
                [
                    html.Thead(
                        html.Tr([
                            html.Th("Date"),
                            html.Th("Model"),
                            html.Th("Call"),
                            html.Th(""),
                            html.Th("P&L"),
                        ])
                    ),
                    html.Tbody(rows),
                ],
                className="prediction-history-table",
            ),
        ],
        className="prediction-history-section",
    )

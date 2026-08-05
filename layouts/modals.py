"""Modal dialogs mounted at the app root.

These sit outside the routed page content so a modal opened from one section
survives navigation, and so their callbacks always have their Inputs mounted.
"""

import dash_bootstrap_components as dbc
from dash import dcc, html



def create_data_modal() -> dbc.Modal:
    """Create the raw data view modal.

    Returns:
        Modal component for viewing data.
    """
    return dbc.Modal(
        [
            dbc.ModalHeader(
                dbc.ModalTitle("Raw Data"),
                close_button=True,
            ),
            dbc.ModalBody(
                [
                    html.Div(id="data-table-container"),
                ],
            ),
            dbc.ModalFooter(
                [
                    dbc.Button(
                        "Export Parquet",
                        id="modal-export-btn",
                        color="primary",
                        className="me-2",
                    ),
                    dbc.Button(
                        "Close",
                        id="modal-close-btn",
                        color="secondary",
                    ),
                ],
            ),
        ],
        id="data-modal",
        size="xl",
        is_open=False,
    )


def _report_param_selects(prefix: str) -> dict:
    """Shared parameter selects for report generation.

    Rendered in BOTH the AI Report modal (prefix "ai-report" — preserves its
    existing component ids) and the Full Analysis modal (prefix "fa"), so each
    flow owns its own visible, configurable state instead of one modal
    silently reading the other's.
    """
    return {
        "lookback": dbc.Select(
            id=f"{prefix}-lookback",
            options=[
                {"label": "3 days", "value": "3"},
                {"label": "7 days", "value": "7"},
                {"label": "14 days", "value": "14"},
                {"label": "30 days", "value": "30"},
            ],
            value="7",
            size="sm",
        ),
        "model": dbc.Select(
            id=f"{prefix}-model",
            options=[
                {"label": "GPT-5.6 Luna (default)", "value": "gpt-5.6-luna"},
                {"label": "Claude Sonnet 5", "value": "claude-sonnet-5"},
                {"label": "Claude Sonnet 4.6", "value": "claude-sonnet-4-6"},
                {"label": "Claude Haiku 4.5 (fast)", "value": "claude-haiku-4-5"},
            ],
            value="gpt-5.6-luna",
            size="sm",
        ),
        "type": dbc.Select(
            id=f"{prefix}-type",
            options=[
                {"label": "Deep (company thesis)", "value": "thesis"},
                {"label": "Standard (faster)", "value": "standard"},
            ],
            value="thesis",
            size="sm",
        ),
        "recs": dbc.Select(
            id=f"{prefix}-recs",
            options=[
                {"label": "From text analysis + predictions", "value": "auto"},
                {"label": "From predictions only (no text)", "value": "signals"},
                {"label": "Off", "value": "off"},
            ],
            value="auto",
            size="sm",
        ),
        "recs-model": dbc.Select(
            id=f"{prefix}-recs-model",
            options=[
                {"label": "Claude Sonnet 5 (default)", "value": "claude-sonnet-5"},
                {"label": "GPT-5.6 Luna (reasoning)", "value": "gpt-5.6-luna"},
                {"label": "Claude Sonnet 4.6", "value": "claude-sonnet-4-6"},
            ],
            value="claude-sonnet-5",
            size="sm",
        ),
    }


# =============================================================================
# UNIFIED RUN MODAL
# =============================================================================

RUN_MODELS = [
    ("kronos_mini", "Kronos", "90 bars OHLCV (min 30)"),
    ("xgboost_shap", "XGBoost SHAP", "1Y OHLCV + SPY + news (SMA-200)"),
    ("lightgbm", "LightGBM", "1Y OHLCV + SPY + news (SMA-200)"),
    ("deberta_sentiment", "DeBERTa Sentiment", "News articles (relevance >= 0.7)"),
    ("trading_agents", "TradingAgents",
     "Full research: technicals + fundamentals + news (~60s)"),
]

RUN_SCOPES = [
    ("models", "Models only",
     "Numerical predictions. No LLM report, no recommendations."),
    ("report", "Report only",
     "News and research analysis. No model predictions."),
    ("full", "Full pipeline",
     "Report, then all models, then the synthesis recommendation."),
]


def create_run_modal() -> dbc.Modal:
    """One dialog for every way of starting a run.

    Replaces three near-identical modals (Predict, AI Report, Full Analysis)
    whose parameter panels were duplicated: the report selects existed twice,
    once as ai-report-* and once as fa-*, and Full Analysis silently
    overrode the Predict modal's model checkboxes. Scope is now an explicit
    choice rather than something inferred from which button was pressed, and
    there is exactly one set of controls behind it.
    """
    from datetime import date

    try:
        from utils.trading_calendar import get_default_target_day
        default_target = get_default_target_day()
    except Exception:
        default_target = date.today()

    sel = _report_param_selects("run")

    model_checks = [
        html.Div(
            [
                dbc.Checkbox(id={"type": "run-model-check", "model": mid},
                             value=True, className="me-2"),
                html.Span(display, className="run-model-name"),
                html.Span(requirement, className="run-model-req"),
            ],
            className="d-flex align-items-center mb-2",
        )
        for mid, display, requirement in RUN_MODELS
    ]

    def field(label, control, hint=None):
        return html.Div(
            [
                html.Label(label, className="input-label"),
                control,
                html.Div(hint, className="run-field-hint") if hint else None,
            ],
            className="run-field",
        )

    return dbc.Modal(
        [
            dbc.ModalHeader(
                dbc.ModalTitle([html.I(className="bi bi-play-circle me-2"),
                                "Run analysis"]),
                close_button=True,
            ),
            dbc.ModalBody([
                html.Div(
                    [
                        html.Label("What to run", className="input-label"),
                        dbc.RadioItems(
                            id="run-scope",
                            options=[{"label": lbl, "value": val}
                                     for val, lbl, _ in RUN_SCOPES],
                            value="full",
                            inline=True,
                            className="run-scope-radio",
                        ),
                        html.Div(id="run-scope-hint", className="run-field-hint"),
                    ],
                    className="run-field",
                ),
                html.Hr(),

                field(
                    "Target session",
                    html.Div(
                        [
                            dcc.DatePickerSingle(
                                id="run-date-picker",
                                date=default_target.isoformat(),
                                max_date_allowed=default_target.isoformat(),
                                min_date_allowed="2020-01-01",
                                display_format="YYYY-MM-DD",
                                className="predict-date-picker",
                            ),
                            html.Span(id="run-date-mode-label", className="ms-2"),
                        ],
                        className="d-flex align-items-center",
                    ),
                    "The close being predicted. Data is cut off at the previous "
                    "trading day, so a Monday target sees nothing after Friday.",
                ),
                html.Div(id="run-data-summary"),

                # --- Models ---
                html.Div(
                    [
                        html.Hr(),
                        html.H6("Models", className="mb-2"),
                        html.Div(model_checks),
                        html.Div(
                            [
                                dbc.Checkbox(id="run-ensemble-check", value=True,
                                             className="me-2"),
                                html.Span("Ensemble", className="run-model-name"),
                                html.Span("combines only the models enabled in "
                                          "Ensemble Config",
                                          className="run-model-req"),
                            ],
                            className="d-flex align-items-center mb-2",
                        ),
                        html.Div(id="run-ensemble-summary"),
                    ],
                    id="run-models-section",
                ),

                # --- Report ---
                html.Div(
                    [
                        html.Hr(),
                        html.H6("Report", className="mb-2"),
                        html.Div(
                            [
                                field("News window", sel["lookback"]),
                                field("Report model", sel["model"]),
                                field("Depth", sel["type"]),
                            ],
                            className="run-field-grid",
                        ),
                        html.Div(
                            [
                                dbc.Checkbox(id="run-include-research", value=False,
                                             className="me-2"),
                                html.Span("Include per-symbol research reports",
                                          className="run-model-name"),
                                html.Span("ignored on Full pipeline, which already "
                                          "produces them", className="run-model-req"),
                            ],
                            className="d-flex align-items-center mb-2",
                        ),
                        html.Div(id="run-article-preview",
                                 className="run-article-preview"),
                    ],
                    id="run-report-section",
                ),

                # --- Recommendations ---
                html.Div(
                    [
                        html.Hr(),
                        html.H6("Recommendations", className="mb-2"),
                        html.Div(
                            [
                                field("Basis", sel["recs"]),
                                field("Synthesis model", sel["recs-model"]),
                            ],
                            className="run-field-grid",
                        ),
                    ],
                    id="run-recs-section",
                ),
            ]),
            dbc.ModalFooter([
                dbc.Button("Cancel", id="run-cancel-btn", color="secondary",
                           className="me-2"),
                dbc.Button([html.I(className="bi bi-play-fill me-1"), "Run"],
                           id="run-confirm-btn", color="success"),
            ]),
        ],
        id="run-modal",
        is_open=False,
        size="lg",
        scrollable=True,
    )

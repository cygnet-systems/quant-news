"""Main dashboard layout for Quant News Tracker.

This module defines the overall page structure following the
3-column layout specified in PROJECT.md.
"""

import dash_bootstrap_components as dbc
from dash import dcc, html

from config import MODEL, API
from layouts.components import (
    create_data_actions,
    create_ensemble_config_drawer,
    create_indicator_toggles,
    create_period_selector,
    create_stock_input,
)


def create_sidebar() -> html.Div:
    """Create the left sidebar with stock input and controls.

    Returns:
        Sidebar div component.
    """
    return html.Div(
        [
            # Logo/Title
            html.Div(
                [
                    html.H1("QuantNews", className="app-title"),
                    html.P("Stock Analysis Dashboard", className="app-subtitle"),
                    # Cygnet SSO identity chip — filled per page load by the
                    # render_auth_chip callback (sign-in button or uid+logout).
                    html.Div(id="auth-chip", className="auth-chip mt-1"),
                ],
                className="sidebar-header",
            ),
            html.Hr(className="sidebar-divider"),

            # Stock Input
            create_stock_input(),

            html.Hr(className="sidebar-divider"),

            # Period Selector
            html.Div(
                [
                    html.Label("Time Period", className="input-label"),
                    create_period_selector("1y"),
                ],
                className="period-section",
            ),

            html.Hr(className="sidebar-divider"),

            # Indicator Toggles
            html.Div(
                [
                    html.Label("Indicators", className="input-label"),
                    create_indicator_toggles(),
                ],
                className="indicator-section",
            ),

            html.Hr(className="sidebar-divider"),

            # Data Actions
            html.Div(
                [
                    html.Label("Data", className="input-label"),
                    create_data_actions(),
                ],
                className="data-section",
            ),

            # Spacer
            html.Div(className="sidebar-spacer"),

            # Footer
            html.Div(
                [
                    html.Div(
                        [
                            html.Span("Cache: ", className="cache-label"),
                            html.Span(id="cache-status", className="cache-status-text"),
                            dbc.Switch(
                                id="cache-toggle",
                                value=True,
                                className="cache-toggle-switch",
                                style={"marginLeft": "auto"},
                            ),
                        ],
                        className="cache-status-row",
                        style={"display": "flex", "alignItems": "center"},
                    ),
                    html.Div(id="data-source-indicator", className="data-source-row"),
                ],
                className="sidebar-footer",
            ),
        ],
        className="sidebar",
        id="sidebar",
    )


def create_main_content() -> html.Div:
    """Create the main content area with charts.

    Returns:
        Main content div component.
    """
    return html.Div(
        [
            # Summary Cards Row
            html.Div(
                id="summary-cards",
                className="summary-cards-row",
            ),

            # Main Price Chart
            html.Div(
                [
                    html.Div(
                        [
                            html.H3(id="chart-title", className="chart-title"),
                            html.Div(id="chart-subtitle", className="chart-subtitle"),
                        ],
                        className="chart-header",
                    ),
                    dcc.Loading(
                        dcc.Graph(
                            id="price-chart",
                            className="main-chart",
                            style={"height": "380px"},
                            config={
                                "displayModeBar": True,
                                "modeBarButtonsToRemove": [
                                    "lasso2d",
                                    "select2d",
                                ],
                                "displaylogo": False,
                                "responsive": False,
                            },
                        ),
                        type="circle",
                        color="#00D4AA",
                    ),
                ],
                className="chart-container price-chart-container",
            ),

            # Technical Indicators Row
            html.Div(
                [
                    # MACD Chart
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H4("MACD", className="subplot-title"),
                                    html.I(
                                        className="bi bi-info-circle ms-2 info-icon",
                                        id="macd-info-icon",
                                    ),
                                    dbc.Tooltip(
                                        "Moving Average Convergence Divergence: "
                                        "MACD Line = EMA(12) - EMA(26), "
                                        "Signal Line = 9-day EMA of MACD. "
                                        "Bullish when MACD crosses above Signal.",
                                        target="macd-info-icon",
                                        placement="top",
                                    ),
                                ],
                                className="subplot-title-row",
                            ),
                            dcc.Loading(
                                dcc.Graph(
                                    id="macd-chart",
                                    className="subplot-chart",
                                    style={"height": "140px"},
                                    config={"displayModeBar": False, "responsive": False},
                                ),
                                type="circle",
                                color="#00D4AA",
                            ),
                        ],
                        className="subplot-container",
                    ),
                    # RSI Chart
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H4("RSI", className="subplot-title"),
                                    html.I(
                                        className="bi bi-info-circle ms-2 info-icon",
                                        id="rsi-info-icon",
                                    ),
                                    dbc.Tooltip(
                                        "Relative Strength Index (14-day): "
                                        "Measures momentum on a 0-100 scale. "
                                        "Overbought: >70, Oversold: <30.",
                                        target="rsi-info-icon",
                                        placement="top",
                                    ),
                                ],
                                className="subplot-title-row",
                            ),
                            dcc.Loading(
                                dcc.Graph(
                                    id="rsi-chart",
                                    className="subplot-chart",
                                    style={"height": "140px"},
                                    config={"displayModeBar": False, "responsive": False},
                                ),
                                type="circle",
                                color="#00D4AA",
                            ),
                        ],
                        className="subplot-container",
                    ),
                ],
                className="indicators-row",
            ),

            # Volume Chart
            html.Div(
                [
                    html.Div(
                        [
                            html.H4("Volume", className="subplot-title"),
                            html.I(
                                className="bi bi-info-circle ms-2 info-icon",
                                id="volume-info-icon",
                            ),
                            dbc.Tooltip(
                                "Trading Volume: Number of shares traded. "
                                "Green = up day, Red = down day. "
                                "Line shows 20-day moving average.",
                                target="volume-info-icon",
                                placement="top",
                            ),
                        ],
                        className="subplot-title-row",
                    ),
                    dcc.Loading(
                        dcc.Graph(
                            id="volume-chart",
                            className="volume-chart",
                            style={"height": "110px"},
                            config={"displayModeBar": False, "responsive": False},
                        ),
                        type="circle",
                        color="#00D4AA",
                    ),
                ],
                className="chart-container volume-container",
            ),
        ],
        className="main-content",
        id="main-content",
    )


def create_context_panel() -> html.Div:
    """Create the right context panel with dynamic per-symbol tabs.

    Returns:
        Context panel div component with:
        - Panel header with LLM status
        - Dynamic tabs container (populated by callback based on selected symbols)

    Tab structure: [ Overall ] [ AAPL ] [ GOOGL ] [ MSFT ] ...
    Each symbol gets its own tab with recommendation + news + sentiment.
    """
    return html.Div(
        [
            # Panel header (stays outside tabs)
            html.Div(
                [
                    html.H2("Analysis", className="panel-main-title"),
                    html.Span(id="llm-status", className="llm-status-badge"),
                    # Background prediction running indicator
                    html.Span(
                        [
                            html.I(className="bi bi-gear-fill spinning-icon"),
                            " Predicting...",
                        ],
                        id="prediction-running-indicator",
                        className="prediction-running-badge",
                        style={"display": "none"},
                    ),
                    # Hidden — keeps download_report_pdf callback wired
                    html.Button(id="download-report-btn", style={"display": "none"}),
                ],
                className="panel-header-row",
            ),

            # Dynamic tabs container - populated by update_symbol_tabs callback
            dcc.Loading(
                html.Div(id="symbol-tabs-container"),
                type="circle",
                color="#00D4AA",
            ),
        ],
        className="context-panel",
        id="context-panel",
    )


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


def _ai_report_date_picker() -> dcc.DatePickerSingle:
    """Analysis-date picker: NYSE trading days only.

    Defaults to the NEXT trading day — the session the report is for.
    Weekends/holidays are unpickable; range is ~13 months back (backtests)
    through the next session (nothing further ahead is analyzable).
    """
    from datetime import date, timedelta

    from utils.trading_calendar import get_next_trading_day, non_trading_days

    today = date.today()
    default = get_next_trading_day(today)
    min_d = today - timedelta(days=400)
    try:
        disabled = non_trading_days(min_d, default)
    except Exception:
        disabled = []  # calendar hiccup: picker still works, just unrestricted

    return dcc.DatePickerSingle(
        id="ai-report-date",
        date=default,
        min_date_allowed=min_d,
        max_date_allowed=default,
        disabled_days=disabled,
        initial_visible_month=default,
        display_format="YYYY-MM-DD",
        first_day_of_week=1,
        className="ai-report-datepicker",
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


def create_report_confirm_modal() -> dbc.Modal:
    """Create confirmation modal for AI Report generation."""
    params = _report_param_selects("ai-report")
    return dbc.Modal(
        [
            dbc.ModalHeader(
                dbc.ModalTitle([
                    html.I(className="bi bi-file-text me-2"),
                    "Generate AI Report",
                ]),
                close_button=True,
            ),
            dbc.ModalBody(id="report-confirm-body"),
            # Adjustable parameters: static (not rebuilt per open) so the
            # user's choices persist across modal opens within a session.
            dbc.ModalBody(
                [
                    html.Hr(className="mt-0 mb-3"),
                    html.H6("Parameters", className="mb-2"),
                    dbc.Row(
                        [
                            dbc.Col([
                                dbc.Label("Analysis date", size="sm"),
                                # Market days only; default = the session the
                                # report is FOR (next trading day). Bounds and
                                # holiday list computed like the sibling
                                # predict/full-analysis pickers.
                                _ai_report_date_picker(),
                            ], width=6),
                            dbc.Col([
                                dbc.Label("News window", size="sm"),
                                params["lookback"],
                            ], width=6),
                        ],
                        className="mb-2",
                    ),
                    dbc.Row(
                        [
                            dbc.Col([
                                dbc.Label("Model", size="sm"),
                                params["model"],
                            ], width=6),
                            dbc.Col([
                                dbc.Label("Analysis type", size="sm"),
                                params["type"],
                            ], width=6),
                        ],
                    ),
                    dbc.Row(
                        [
                            dbc.Col([
                                dbc.Label("Recommendations", size="sm"),
                                params["recs"],
                            ], width=6),
                            dbc.Col([
                                dbc.Label("Recommendations model", size="sm"),
                                params["recs-model"],
                            ], width=6),
                        ],
                        className="mt-2",
                    ),
                    dbc.Checkbox(
                        id="ai-report-include-research",
                        value=True,
                        label=html.Span([
                            "Deep research report per symbol (recommended) ",
                            html.Span("— one analyst call writes the full report "
                                      "+ banner/watch/thesis fields (~30s/symbol). "
                                      "Uncheck for the fast news-only tier (~15s).",
                                      className="text-muted",
                                      style={"fontSize": "0.75rem"}),
                        ]),
                        className="mt-2",
                    ),
                    # Live article availability for the chosen window/date —
                    # fetched (point-in-time, DB-cached) so the numbers are
                    # what generation will actually use, not the stale store.
                    dcc.Loading(
                        html.Div(id="ai-report-article-preview",
                                 className="ai-report-article-preview mt-3"),
                        type="dot", color="#00D4AA",
                    ),
                ],
                className="pt-0",
            ),
            dbc.ModalFooter(
                [
                    dbc.Button(
                        "Cancel",
                        id="report-cancel-btn",
                        color="secondary",
                        className="me-2",
                    ),
                    dbc.Button(
                        [html.I(className="bi bi-play-fill me-1"), "Generate Report"],
                        id="report-confirm-btn",
                        color="info",
                    ),
                ],
            ),
        ],
        id="report-confirm-modal",
        is_open=False,
    )


def create_predict_confirm_modal() -> dbc.Modal:
    """Create confirmation modal for model predictions.

    Model and ensemble checkboxes are fixed layout elements (not dynamic)
    so they can be wired to callbacks. Only the data summary is dynamic.
    Includes a date picker for backtesting (default: today).
    """
    from datetime import date

    _ALL_MODELS = [
        ("kronos_mini", "Kronos", "90 bars OHLCV (min 30)"),
        ("xgboost_shap", "XGBoost SHAP", "1Y OHLCV + SPY + news (SMA-200)"),
        ("lightgbm", "LightGBM", "1Y OHLCV + SPY + news (SMA-200)"),
        ("deberta_sentiment", "DeBERTa Sentiment", "News articles (relevance >= 0.7)"),
        ("trading_agents", "TradingAgents", "Full research: technicals + fundamentals + news (~60s)"),
    ]

    model_checks = []
    for model_id, display, requirement in _ALL_MODELS:
        model_checks.append(
            html.Div([
                dbc.Checkbox(
                    id={"type": "predict-model-check", "model": model_id},
                    value=True,
                    className="me-2",
                ),
                html.Span(display, style={"fontWeight": "bold", "marginRight": "8px"}),
                html.Span(requirement, style={
                    "color": "var(--text-secondary)", "fontSize": "0.85rem",
                }),
            ], className="d-flex align-items-center mb-2")
        )

    ensemble_check = html.Div([
        dbc.Checkbox(
            id="predict-ensemble-check",
            value=True,
            className="me-2",
        ),
        html.Span("Ensemble", style={"fontWeight": "bold", "marginRight": "8px"}),
        html.Span(
            "(combines enabled models from Ensemble Config)",
            style={"color": "var(--text-secondary)", "fontSize": "0.85rem"},
        ),
    ], className="d-flex align-items-center mb-2")

    today = date.today()

    # The picker holds the TARGET session — the close being predicted — and
    # defaults to the next one whose close has not happened yet. Data is cut
    # off at the previous trading day, so a Monday target sees through Friday.
    try:
        from utils.trading_calendar import get_default_target_day
        default_target = get_default_target_day()
    except Exception:
        default_target = today

    return dbc.Modal(
        [
            dbc.ModalHeader(
                dbc.ModalTitle([
                    html.I(className="bi bi-cpu me-2"),
                    "Run Predictions",
                ]),
                close_button=True,
            ),
            dbc.ModalBody([
                # Dynamic data summary (populated by callback)
                html.Div(id="predict-data-summary"),
                html.Hr(),
                # Target date picker — the session whose close is predicted
                html.H6("Target Date", className="mb-2"),
                html.Div([
                    dcc.DatePickerSingle(
                        id="predict-date-picker",
                        date=default_target.isoformat(),
                        max_date_allowed=default_target.isoformat(),
                        min_date_allowed="2020-01-01",
                        display_format="YYYY-MM-DD",
                        className="predict-date-picker",
                    ),
                    html.Span(
                        id="predict-date-mode-label",
                        className="ms-2",
                        style={"fontSize": "0.85rem"},
                    ),
                ], className="d-flex align-items-center mb-3"),
                html.Hr(),
                # Fixed model selection checkboxes
                html.H6("Select Models to Run", className="mb-3"),
                html.Div(model_checks, className="mb-3"),
                html.Hr(),
                html.H6("Ensemble", className="mb-3"),
                ensemble_check,
                html.Div(id="predict-ensemble-summary", className="mt-2"),
                html.Div(
                    [
                        html.I(className="bi bi-info-circle me-2"),
                        "Each selected model runs independently. Ensemble "
                        "combines only the models enabled in Ensemble Config.",
                    ],
                    className="text-muted small mt-3",
                ),
            ]),
            dbc.ModalFooter(
                [
                    dbc.Button(
                        "Cancel",
                        id="predict-cancel-btn",
                        color="secondary",
                        className="me-2",
                    ),
                    dbc.Button(
                        [html.I(className="bi bi-play-fill me-1"), "Run Predictions"],
                        id="predict-confirm-btn",
                        color="warning",
                    ),
                ],
            ),
        ],
        id="predict-confirm-modal",
        is_open=False,
        size="lg",
    )


def create_full_analysis_modal() -> dbc.Modal:
    """Create confirmation modal for Full Analysis (Report + Predictions + Recommendations)."""
    from datetime import date as _date

    rec_model = MODEL.RECOMMENDATIONS_MODEL
    rec_provider = MODEL.RECOMMENDATIONS_PROVIDER
    has_key = bool(API.OPENAI_API_KEY) if rec_provider == "openai" else True
    params = _report_param_selects("fa")

    _today = _date.today()
    try:
        from utils.trading_calendar import get_default_target_day
        _fa_default = get_default_target_day()
    except Exception:
        _fa_default = _today

    return dbc.Modal(
        [
            dbc.ModalHeader(
                dbc.ModalTitle([
                    html.I(className="bi bi-lightning-fill me-2"),
                    "Full Analysis",
                ]),
                close_button=True,
            ),
            dbc.ModalBody([
                html.Div(id="full-analysis-body"),
                html.Hr(),
                html.Div([
                    html.H6("Target Date", className="mb-1"),
                    html.Div([
                        dcc.DatePickerSingle(
                            id="fa-date-picker",
                            date=_fa_default.isoformat(),
                            max_date_allowed=_fa_default.isoformat(),
                            min_date_allowed="2020-01-01",
                            display_format="YYYY-MM-DD",
                            className="predict-date-picker",
                        ),
                        html.Span(
                            "The session whose close is being predicted. All three "
                            "stages only see data through the PREVIOUS trading day "
                            "(prices, news, metrics) — a Monday target sees nothing "
                            "after the preceding Friday's close. Pick a past date to "
                            "backtest without lookahead bias.",
                            className="ms-2 text-muted small",
                            style={"maxWidth": "380px", "display": "inline-block",
                                   "verticalAlign": "middle"},
                        ),
                    ], className="d-flex align-items-center mb-3"),
                ]),
                html.Hr(),
                # Same individual options as the AI Report modal (fa-* ids) —
                # this flow used to silently read the other modal's state.
                html.H6("Parameters", className="mb-2"),
                dbc.Row(
                    [
                        dbc.Col([
                            dbc.Label("News window", size="sm"),
                            params["lookback"],
                        ], width=6),
                        dbc.Col([
                            dbc.Label("Research model", size="sm"),
                            params["model"],
                        ], width=6),
                    ],
                    className="mb-2",
                ),
                dbc.Row(
                    [
                        dbc.Col([
                            dbc.Label("Analysis type", size="sm"),
                            params["type"],
                        ], width=6),
                    ],
                    className="mb-2",
                ),
                dbc.Row(
                    [
                        dbc.Col([
                            dbc.Label("Recommendations", size="sm"),
                            params["recs"],
                        ], width=6),
                        dbc.Col([
                            dbc.Label("Recommendations model", size="sm"),
                            params["recs-model"],
                        ], width=6),
                    ],
                    className="mb-3",
                ),
                html.Hr(),
                html.H6("Pipeline", className="mb-2"),
                html.Div([
                    html.Div([
                        html.Span("1. ", style={"fontWeight": "bold"}),
                        html.Span("AI Report "),
                        html.Span("— portfolio news overview",
                                  style={"color": "var(--text-secondary)", "fontSize": "0.85rem"}),
                    ], className="mb-1"),
                    html.Div([
                        html.Span("2. ", style={"fontWeight": "bold"}),
                        html.Span("Model Predictions "),
                        html.Span("— all ML models; per-symbol research report "
                                  "uses the model selected above",
                                  style={"color": "var(--text-secondary)", "fontSize": "0.85rem"}),
                    ], className="mb-1"),
                    html.Div([
                        html.Span("3. ", style={"fontWeight": "bold"}),
                        html.Span("Recommendations "),
                        html.Span("— synthesis, configured above",
                                  style={"color": "var(--text-secondary)", "fontSize": "0.85rem"}),
                    ], className="mb-1"),
                ], className="mb-3"),
                html.Div(
                    [
                        html.I(className="bi bi-info-circle me-2"),
                        "Runs all three stages sequentially. "
                        "Recommendations synthesize the research reports + "
                        "predictions into actionable advice.",
                    ],
                    className="text-muted small",
                ),
            ] + ([] if has_key else [
                html.Div(
                    [
                        html.I(className="bi bi-exclamation-triangle me-2"),
                        f"OPENAI_API_KEY not set — {rec_model} recommendations will be unavailable.",
                    ],
                    className="text-warning small mt-2",
                ),
            ])),
            dbc.ModalFooter([
                dbc.Button(
                    "Cancel",
                    id="full-analysis-cancel-btn",
                    color="secondary",
                    className="me-2",
                ),
                dbc.Button(
                    [html.I(className="bi bi-lightning-fill me-1"), "Run Full Analysis"],
                    id="full-analysis-confirm-btn",
                    color="success",
                ),
            ]),
        ],
        id="full-analysis-modal",
        is_open=False,
        size="lg",
    )


def create_scoreboard_modal() -> dbc.Modal:
    """Model Scoreboard: calibration and P&L across ALL evaluated predictions.

    Defaults to every symbol and date; filters narrow when needed.
    """
    return dbc.Modal(
        [
            dbc.ModalHeader(
                dbc.ModalTitle([
                    html.I(className="bi bi-trophy me-2"),
                    "Model Scoreboard",
                ]),
                close_button=True,
            ),
            dbc.ModalBody([
                html.Div(
                    [
                        html.Div(
                            dcc.Dropdown(
                                id="scoreboard-symbols",
                                options=[],
                                value=[],
                                multi=True,
                                searchable=True,
                                placeholder="All symbols",
                                className="history-symbol-dropdown",
                            ),
                            className="scoreboard-filter-symbols",
                        ),
                        html.Div(
                            dcc.DatePickerRange(
                                id="scoreboard-date-range",
                                start_date=None,
                                end_date=None,
                                display_format="YYYY-MM-DD",
                                start_date_placeholder_text="All dates",
                                end_date_placeholder_text="to date",
                                clearable=True,
                                className="predict-date-picker",
                            ),
                            className="scoreboard-filter-dates",
                        ),
                    ],
                    className="scoreboard-filter-row",
                ),
                html.Div(id="scoreboard-content"),
            ]),
            dbc.ModalFooter([
                html.Span(id="scoreboard-pending-label", className="text-muted small me-auto"),
                dbc.Button(
                    [html.I(className="bi bi-check2-circle me-1"), "Evaluate pending"],
                    id="scoreboard-evaluate-btn",
                    color="success",
                    size="sm",
                    outline=True,
                ),
            ]),
        ],
        id="scoreboard-modal",
        is_open=False,
        size="lg",
        scrollable=True,
    )


def create_layout() -> html.Div:
    """Create the complete dashboard layout.

    Returns:
        Root layout component.
    """
    return html.Div(
        [
            # Selection/config state persists across refreshes (storage_type
            # "local"); the heavy payload stores below stay in memory because
            # localStorage caps at ~5MB and full OHLCV + articles blow past it.
            # They re-derive automatically: their callbacks fire on
            # "selected-symbols", which is restored on load.
            dcc.Store(id="selected-symbols", data=[], storage_type="local"),
            # Recent symbol GROUPS (list of lists, newest first, capped) —
            # local so past sessions' watchlists survive restarts.
            dcc.Store(id="recent-symbol-groups", data=[], storage_type="local"),
            dcc.Store(id="current-period", data="1y", storage_type="local"),
            dcc.Store(id="stock-data-store", data={}),
            dcc.Store(id="news-data-store", data={}),
            dcc.Store(id="ai-analysis-store", data={}),
            dcc.Store(id="cache-enabled", data=True, storage_type="local"),
            # Model prediction stores
            dcc.Store(id="model-signals-store", data={}),
            dcc.Store(id="prediction-store-status", data={}),
            # Strategy evaluation stores
            dcc.Store(id="strategy-metrics-store", data=[]),
            dcc.Store(id="strategy-evaluations-store", data=[]),
            # Ensemble configuration store (user-adjustable)
            dcc.Store(id="ensemble-config-store", data={
                "enabled_models": list(MODEL.ENSEMBLE_DEFAULT_ENABLED),
                "weights": dict(MODEL.ENSEMBLE_DEFAULT_WEIGHTS),
            }, storage_type="local"),

            # Historical data store (reports, predictions, recommendations)
            dcc.Store(id="report-history-store", data={}),

            # History tab filter state (client-side only)
            dcc.Store(id="history-filter-symbols", data=[], storage_type="local"),
            dcc.Store(id="history-filter-date-range", data="all", storage_type="local"),
            dcc.Store(id="history-filter-date-specific", data=None, storage_type="local"),
            # Activity Log scope. Honoured only for Administrators — the
            # server pins everyone else to their own rows regardless.
            dcc.Store(id="history-activity-scope", data="all", storage_type="local"),
            dcc.Store(id="history-eval-status", data=None),
            dcc.Store(id="active-tab-store", data=None, storage_type="local"),

            # URL — drives the once-per-page-load auth chip render and carries
            # ?login_error= back from the /auth/login redirect.
            dcc.Location(id="url", refresh=False),

            # Cygnet SSO login modal. Native html.Form POST to /auth/login —
            # the server sets the signed session cookie on the redirect, which
            # a Dash callback cannot do.
            dbc.Modal(
                [
                    dbc.ModalHeader(dbc.ModalTitle("Sign in to Cygnet")),
                    dbc.ModalBody(
                        html.Form(
                            [
                                html.Div(id="login-error-msg",
                                         className="text-danger mb-2"),
                                html.Label("User ID", className="input-label"),
                                # dcc.Input renders a native <input> with the
                                # name attr, so the plain-HTML form POST works
                                # (Dash 4 removed the html.Input wrapper).
                                dcc.Input(name="userid", type="text",
                                          className="form-control mb-2",
                                          persistence=False),
                                html.Label("Password", className="input-label"),
                                dcc.Input(name="password", type="password",
                                          className="form-control mb-3",
                                          persistence=False),
                                # Sessions are persistent (7-day) like the SSO
                                # handoff — no "keep me signed in" toggle.
                                dcc.Input(name="remember", type="hidden", value="1"),
                                html.Button("Sign in", type="submit",
                                            className="btn btn-primary w-100"),
                            ],
                            action="/auth/login", method="post",
                        )
                    ),
                ],
                id="login-modal", is_open=False, size="sm",
            ),

            # Live pipeline activity feed (Full Analysis / Predict).
            # The shell is static and only the rows/count/icon are refreshed --
            # re-rendering the whole panel every tick would reset n_clicks on
            # its own controls, so the buttons could never be wired up.
            dcc.Interval(id="progress-interval", interval=1500, disabled=False),
            dcc.Store(id="progress-panel-state", storage_type="local",
                      data={"mode": "normal", "closed": False}),
            html.Div(
                html.Div(
                    [
                        html.Div(
                            [
                                html.Div(id="progress-header-icon"),
                                html.Span("Pipeline Activity", className="progress-title"),
                                html.Span(id="progress-count", className="progress-count"),
                                html.Div(
                                    [
                                        html.Button(
                                            html.I(id="progress-expand-icon",
                                                   className="bi bi-arrows-angle-expand"),
                                            id="progress-expand-btn",
                                            className="progress-ctl",
                                            title="Expand / restore",
                                        ),
                                        html.Button(
                                            html.I(id="progress-min-icon",
                                                   className="bi bi-dash-lg"),
                                            id="progress-min-btn",
                                            className="progress-ctl",
                                            title="Minimise / restore",
                                        ),
                                        html.Button(
                                            html.I(className="bi bi-x-lg"),
                                            id="progress-close-btn",
                                            className="progress-ctl",
                                            title="Close",
                                        ),
                                    ],
                                    className="progress-controls",
                                ),
                            ],
                            className="progress-header",
                        ),
                        html.Div(id="progress-feed-scroll", className="progress-feed"),
                    ],
                    id="progress-panel",
                    className="progress-panel",
                ),
                id="analysis-progress-panel",
                className="progress-panel-container",
            ),
            html.Button(
                [html.I(className="bi bi-activity me-1"), "Activity"],
                id="progress-reopen-btn",
                className="progress-reopen",
                style={"display": "none"},
            ),

            # Full Analysis / Recommendations stores
            dcc.Store(id="recommendations-store", data={}),
            dcc.Store(id="full-analysis-requested", data=False),

            # AI Report raw-data modal + download (the JSON is machine food —
            # Luna and the renderers eat it — but it should still be
            # inspectable and exportable on demand)
            dcc.Download(id="download-ai-json"),
            dbc.Modal(
                [
                    dbc.ModalHeader(dbc.ModalTitle("AI Report — raw data"), close_button=True),
                    dbc.ModalBody(html.Pre(id="ai-json-body", className="ai-json-body")),
                    dbc.ModalFooter(
                        dbc.Button([html.I(className="bi bi-download me-1"), "Download .json"],
                                   id="ai-json-download-btn", color="info", size="sm"),
                    ),
                ],
                id="ai-json-modal", is_open=False, size="lg", scrollable=True,
            ),
            # Research report reader — replaces the History accordion stack
            dbc.Modal(
                [
                    dbc.ModalHeader(dbc.ModalTitle(id="ta-report-modal-title"), close_button=True),
                    dbc.ModalBody(id="ta-report-modal-body"),
                ],
                id="ta-report-modal", is_open=False, size="xl", scrollable=True,
            ),
            # Download components for exports
            dcc.Download(id="download-data"),
            dcc.Download(id="download-report"),
            dcc.Download(id="download-hist-report"),
            dcc.Download(id="download-ta-report"),

            # Main Layout Grid
            html.Div(
                [
                    create_sidebar(),
                    create_main_content(),
                    create_context_panel(),
                ],
                className="dashboard-grid",
            ),

            # Modals
            create_data_modal(),
            create_report_confirm_modal(),
            create_predict_confirm_modal(),
            create_full_analysis_modal(),
            create_scoreboard_modal(),

            # Hidden retry button for AI analysis failover
            html.Button(id="ai-retry-btn", style={"display": "none"}),

            # Ensemble config drawer (offcanvas)
            create_ensemble_config_drawer(),

            # Toast notifications
            dbc.Toast(
                "",
                id="download-error-toast",
                header="Download Failed",
                icon="danger",
                is_open=False,
                dismissable=True,
                duration=5000,
                style={
                    "position": "fixed",
                    "top": 16,
                    "right": 16,
                    "zIndex": 9999,
                    "width": 320,
                },
            ),
            dbc.Toast(
                "",
                id="history-eval-toast",
                header="Prediction Evaluation",
                icon="success",
                is_open=False,
                dismissable=True,
                duration=6000,
                style={
                    "position": "fixed",
                    "top": 16,
                    "right": 16,
                    "zIndex": 9999,
                    "width": 340,
                },
            ),
            html.Div(id="toast-container"),
        ],
        className="app-container",
    )

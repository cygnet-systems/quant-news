"""Main dashboard layout for Quant News Tracker.

This module defines the overall page structure following the
3-column layout specified in PROJECT.md.
"""

import dash_bootstrap_components as dbc
from dash import dcc, html

from config import MODEL
from layouts.components import (
    create_data_actions,
    create_ensemble_config_drawer,
    create_indicator_toggles,
    create_period_selector,
    create_stock_input,
)
from layouts.modals import (
    create_data_modal,
    create_full_analysis_modal,
    create_predict_confirm_modal,
    create_report_confirm_modal,
    create_scoreboard_modal,
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

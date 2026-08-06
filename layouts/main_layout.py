"""Root layout for Quant News Tracker.

This module owns the persistent frame only: global stores, the shell (rail,
toolbar, routed #page-content), and every overlay that must outlive a route
change. Section content lives in layouts/pages/.
"""

import dash_bootstrap_components as dbc
from dash import dcc, html

from config import MODEL
from layouts.components import create_ensemble_config_drawer
from layouts.modals import create_data_modal, create_run_modal
from layouts.nav import create_nav_rail, create_topbar, create_watchlist_panel


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
            # Outcome slice of the prediction log: all | pending | right | wrong.
            # Session-scoped, not local: "show me the wrong ones" answers a
            # question you are asking now, and should not still be applied
            # tomorrow when you wonder where your predictions went.
            dcc.Store(id="history-filter-outcome", data="all"),
            # Activity Log scope. Honoured only for Administrators — the
            # server pins everyone else to their own rows regardless.
            dcc.Store(id="history-activity-scope", data="all", storage_type="local"),
            # Activity page filters. At root so a filter survives navigating
            # away and back, like every other filter in the app.
            dcc.Store(id="activity-since-days", data=0, storage_type="local"),
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

            # The shell. Only #page-content is routed: the stores above and the
            # modals, downloads, activity panel and toasts below stay mounted
            # on every route, so navigating never drops state and never leaves
            # a callback pointing at an Input that no longer exists.
            html.Div(
                [
                    create_nav_rail(),
                    html.Div(
                        [
                            create_topbar(),
                            create_watchlist_panel(),
                            # target_components pins this to route changes.
                            # By default a Loading reacts to every callback
                            # writing anywhere inside it, so a filter change
                            # unmounted the whole section for as long as the
                            # request took. overlay_style keeps the children
                            # on screen and dims them instead of blanking,
                            # and delay_show stops fast updates flashing.
                            dcc.Loading(
                                html.Div(id="page-content", className="page-content"),
                                type="circle",
                                color="#00D4AA",
                                target_components={"page-content": "children"},
                                overlay_style={"visibility": "visible",
                                               "opacity": 0.45},
                                delay_show=350,
                            ),
                        ],
                        className="shell-main",
                    ),
                ],
                className="app-shell",
            ),

            # Modals
            create_data_modal(),
            create_run_modal(),

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

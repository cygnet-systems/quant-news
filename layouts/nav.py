"""The application shell: a section rail on the left, a global toolbar on top.

The rail carries navigation only. The toolbar carries the page title and the
one genuinely global action, Run analysis. Everything else that used to live
here (period selector, refresh, export, view-data, the watchlist editor) was
page-local pretending to be global: two of the buttons sat permanently
disabled on four of the six pages. Those controls now render on the pages
they operate on, and the watchlist editor is the Home symbol rail.

Predictions and Performance are deliberately one section. Splitting the
aggregate from the per-call log is what produced the old Scoreboard-modal and
History-tab duplication, where the same numbers were reachable two ways.
"""

import dash_bootstrap_components as dbc
from dash import dcc, html

# (path, label, bootstrap icon). Order is the rail order.
NAV_SECTIONS = [
    ("/", "Home", "bi-house-door"),
    ("/analyze", "Analyze", "bi-graph-up"),
    ("/performance", "Performance", "bi-trophy"),
    ("/reports", "Reports", "bi-journal-text"),
    ("/schedule", "Schedule", "bi-alarm"),
    ("/activity", "Activity", "bi-activity"),
    ("/trace", "Trace", "bi-diagram-3"),
]

SECTION_TITLES = {path: label for path, label, _ in NAV_SECTIONS}


def create_nav_rail() -> html.Div:
    """Left rail: brand, section links, then account and data status."""
    return html.Div(
        [
            html.Div(
                [
                    html.Span("QuantNews", className="rail-brand-name"),
                    html.Span("Stock analysis", className="rail-brand-sub"),
                ],
                className="rail-brand",
            ),

            # active="exact" lets dcc.Location drive the highlight, so there is
            # no callback to keep in step with the URL.
            dbc.Nav(
                [
                    dbc.NavLink(
                        [html.I(className=f"bi {icon} rail-icon"),
                         html.Span(label, className="rail-label")],
                        href=path,
                        active="exact",
                        className="rail-link",
                    )
                    for path, label, icon in NAV_SECTIONS
                ],
                vertical=True,
                pills=True,
                className="rail-nav",
            ),

            html.Div(className="rail-spacer"),

            html.Div(
                [
                    html.Div(
                        [
                            html.Span("Cache", className="cache-label"),
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
                    # Cygnet SSO identity, filled per page load by render_auth_chip.
                    html.Div(id="auth-chip", className="auth-chip"),
                ],
                className="rail-footer",
            ),
        ],
        className="nav-rail",
        id="nav-rail",
    )


def create_topbar() -> html.Div:
    """Global toolbar: where you are, plus the one global action.

    run-analysis-btn must stay mounted on every route — toggle_run_modal takes
    it as a fixed-id Input, and it is the single entry point to the run dialog
    (page-local shortcuts use the {"type": "new-report-btn"} pattern instead).
    """
    return html.Div(
        [
            html.Div(
                [
                    html.H1(id="topbar-title", className="topbar-title"),
                ],
                className="topbar-heading",
            ),

            html.Div(
                [
                    # Background prediction running indicator. It lives in the
                    # always-mounted topbar (not a page) because the running=
                    # spec on generate_model_signals targets it, and the badge
                    # must be visible whatever route the run was started from.
                    html.Span(
                        [
                            html.I(className="bi bi-gear-fill spinning-icon"),
                            " Predicting...",
                        ],
                        id="prediction-running-indicator",
                        className="prediction-running-badge",
                        style={"display": "none"},
                    ),
                    dbc.Button(
                        [html.I(className="bi bi-play-fill me-1"), "Run analysis"],
                        id="run-analysis-btn",
                        color="success",
                        size="sm",
                    ),
                ],
                className="topbar-actions",
            ),
        ]
        , className="topbar", id="topbar",
    )


def create_watchlist_strip() -> html.Div:
    """The watchlist, always in view and editable from every page.

    One slim row under the toolbar: chips with a remove ✕ and an inline add
    box. This is THE editor — the Home rail shows the same set with more
    context (calls, reports, membership groups) but its input only filters.
    Chips use the wl-remove pattern, distinct from the rail rows'
    remove-symbol pattern, because both surfaces are mounted at once on Home
    and duplicate component ids are a hard error.
    """
    return html.Div(
        [
            html.Span("Watchlist", className="wl-strip-label"),
            html.Div(id="watchlist-strip-chips", className="wl-strip-chips"),
            dcc.Input(
                id="symbol-input",
                type="text",
                placeholder="Add symbol… (Enter)",
                debounce=False,
                autoComplete="off",
                className="wl-strip-add",
            ),
        ],
        className="watchlist-strip",
        id="watchlist-strip",
    )

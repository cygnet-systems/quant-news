"""The application shell: a section rail on the left, a global toolbar on top.

The rail carries navigation only. Controls that scope every section (the
watchlist, the period, the run action) sit in the toolbar, because putting them
in the rail is what made the old sidebar a mixture of "where am I going" and
"what am I looking at".

Predictions and Performance are deliberately one section. Splitting the
aggregate from the per-call log is what produced the old Scoreboard-modal and
History-tab duplication, where the same numbers were reachable two ways.
"""

import dash_bootstrap_components as dbc
from dash import html

from layouts.components import (
    create_data_actions,
    create_period_selector,
    create_stock_input,
)

# (path, label, bootstrap icon). Order is the rail order.
NAV_SECTIONS = [
    ("/", "Home", "bi-house-door"),
    ("/analyze", "Analyze", "bi-graph-up"),
    ("/performance", "Performance", "bi-trophy"),
    ("/reports", "Reports", "bi-journal-text"),
    ("/schedule", "Schedule", "bi-alarm"),
    ("/activity", "Activity", "bi-activity"),
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
    """Global toolbar: where you are, what you are looking at, what to run.

    The watchlist editor is a Collapse rather than a Popover so its inputs stay
    mounted while hidden. Several callbacks take symbol-input and the recent
    chips as fixed-id Inputs, and Dash 4 errors on an Input that is not in the
    layout.
    """
    return html.Div(
        [
            html.Div(
                [
                    html.H1(id="topbar-title", className="topbar-title"),
                    html.Div(id="topbar-subtitle", className="topbar-subtitle"),
                ],
                className="topbar-heading",
            ),

            html.Div(
                [
                    dbc.Button(
                        [html.I(className="bi bi-sliders me-1"), "Watchlist"],
                        id="watchlist-toggle-btn",
                        size="sm",
                        outline=True,
                        color="secondary",
                        className="watchlist-toggle-btn",
                    ),
                    create_period_selector("1y"),
                    create_data_actions(),
                ],
                className="topbar-actions",
            ),
        ]
        , className="topbar", id="topbar",
    )


def create_watchlist_panel() -> dbc.Collapse:
    """The symbol editor, revealed under the toolbar."""
    return dbc.Collapse(
        html.Div(create_stock_input(), className="watchlist-panel-inner"),
        id="watchlist-panel",
        # Open by default; the watchlist-panel-open store (localStorage)
        # overrides this on load so a user's close/open choice persists.
        is_open=True,
    )

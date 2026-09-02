"""Reports page: the artifact archive.

Research reports, saved report files and portfolio synthesis runs. These three
are one section because they are all "something a run wrote down", as opposed
to Performance, which is "how the calls turned out".
"""

import dash_bootstrap_components as dbc
from dash import html

from layouts.history_sections import (
    build_history_filter_bar,
    build_recommendations_section,
    build_saved_reports_section,
    build_ta_reports_section,
    empty_history_message,
    filter_history_data,
)


def _action_bar() -> html.Div:
    """Generate a report from here, not from a toolbar two pages away.

    The button opens the shared run modal preset to report scope, the page
    stays the single archive, the modal stays the single place a run is
    configured.
    """
    return html.Div(
        [
            html.Div(
                [
                    html.Div("Research reports", className="reports-bar-title"),
                    html.Div("Generate a new report for the watchlist, or "
                             "reopen any past one below.",
                             className="reports-bar-sub"),
                ],
            ),
            dbc.Button(
                [html.I(className="bi bi-plus-lg me-1"), "New report"],
                id="reports-new-btn",
                color="success",
                className="reports-new-btn",
            ),
        ],
        className="reports-action-bar d-flex align-items-center "
                  "justify-content-between mb-3",
    )


def layout(history_data=None, filter_symbols=None, filter_date_range="all",
           specific_date=None) -> html.Div:
    """Shell only. See performance.layout for why the body is separate."""
    return html.Div(
        [
            _action_bar(),
            build_history_filter_bar(history_data, filter_symbols,
                                     filter_date_range, specific_date),
            html.Div(
                body(history_data, filter_symbols, filter_date_range,
                     specific_date),
                id="archive-body",
            ),
        ],
        className="page page-reports",
    )


def body(history_data=None, filter_symbols=None, filter_date_range="all",
         specific_date=None, page=None) -> list:
    history_data = history_data or {}
    buckets = filter_history_data(history_data, filter_symbols,
                                  filter_date_range, specific_date)

    sections = [
        build_ta_reports_section(buckets["trading_agent_reports"], page=page),
        build_saved_reports_section(buckets["reports"], page=page),
        build_recommendations_section(buckets["recommendations"], page=page),
    ]
    sections = [s for s in sections if s is not None]

    has_filters = bool(filter_symbols
                       or (filter_date_range and filter_date_range != "all")
                       or specific_date)
    sections_or_empty = sections or [empty_history_message(has_filters, "reports")]

    return [html.Div(sections_or_empty, id="reports-sections")]

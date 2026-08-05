"""Reports page: the artifact archive.

Research reports, saved report files and portfolio synthesis runs. These three
are one section because they are all "something a run wrote down", as opposed
to Performance, which is "how the calls turned out".
"""

from dash import html

from layouts.history_sections import (
    build_history_filter_bar,
    build_recommendations_section,
    build_saved_reports_section,
    build_ta_reports_section,
    empty_history_message,
    filter_history_data,
)


def layout(history_data=None, filter_symbols=None, filter_date_range="all",
           specific_date=None) -> html.Div:
    """Shell only. See performance.layout for why the body is separate."""
    return html.Div(
        [
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
         specific_date=None) -> list:
    history_data = history_data or {}
    buckets = filter_history_data(history_data, filter_symbols,
                                  filter_date_range, specific_date)

    sections = [
        build_ta_reports_section(buckets["trading_agent_reports"]),
        build_saved_reports_section(buckets["reports"]),
        build_recommendations_section(buckets["recommendations"]),
    ]
    sections = [s for s in sections if s is not None]

    has_filters = bool(filter_symbols
                       or (filter_date_range and filter_date_range != "all")
                       or specific_date)
    sections_or_empty = sections or [empty_history_message(has_filters, "reports")]

    return [html.Div(sections_or_empty, id="reports-sections")]

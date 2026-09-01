"""Activity page: the run log, and the symbol searches that led to it.

This is the deeper view. The live Pipeline Activity panel stays mounted on
every route and remains the place to watch a run happen; this page is what the
panel cannot be, namely the whole history with filters over it.
"""

import dash_bootstrap_components as dbc
from dash import dcc, html

from layouts.history_sections import build_activity_section
from services import progress_service as _prog

# The vocabulary emit() actually uses. Filtering by stage is the difference
# between "what did the system do" and "show me only the failures".
STAGE_OPTIONS = [
    ("run", "Run"), ("news", "News"), ("ai", "AI report"),
    ("models", "Models"), ("ta", "Research"), ("luna", "Synthesis"),
    ("store", "Storage"), ("done", "Completed"), ("error", "Errors"),
    ("action", "User actions"), ("auth", "Auth"),
]


def _search_history_rows(searches: list[dict]) -> html.Div:
    if not searches:
        return html.Div(
            "No past searches recorded yet. Adding symbols that return data "
            "records the group here.",
            className="history-empty-msg",
        )
    rows = []
    for s in searches:
        # last_used_at is a UTC isoformat string. Slicing it to 16 chars
        # printed raw UTC with no zone marker, so a late-evening ET search
        # showed under TOMORROW's date — a wrong day, not just a wrong hour.
        last = _prog.format_stamp(s.get("last_used_at"))
        rows.append(html.Tr([
            html.Td(html.Button(
                ", ".join(s["symbols"]),
                id={"type": "search-restore", "csv": s["symbols_csv"]},
                className="search-restore-btn",
                title="Load this watchlist",
            )),
            html.Td(str(len(s["symbols"])), className="num"),
            html.Td(str(s.get("use_count", 1)), className="num"),
            html.Td(last, className="num"),
        ]))
    return html.Div(
        html.Table(
            [
                html.Thead(html.Tr([
                    html.Th("Symbols"), html.Th("Count"),
                    html.Th("Times used"),
                    html.Th("Last used", title=f"Times are "
                                               f"{_prog.DISPLAY_TZ.key}"),
                ])),
                html.Tbody(rows),
            ],
            className="history-data-table",
        ),
        className="history-table-wrap",
    )


def layout(activity_scope="all", searches=None, stage_filter=None,
           symbol_filter=None, since_days=None) -> html.Div:
    """Run log with filters, plus durable search history."""
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("Filter by symbol", className="input-label"),
                            dbc.Input(
                                id="activity-symbol-filter",
                                placeholder="AAPL",
                                value=symbol_filter or "",
                                size="sm",
                                debounce=True,
                            ),
                        ],
                        className="activity-filter-field",
                    ),
                    html.Div(
                        [
                            html.Label("Activity type", className="input-label"),
                            dcc.Dropdown(
                                id="activity-stage-filter",
                                options=[{"label": lbl, "value": val}
                                         for val, lbl in STAGE_OPTIONS],
                                value=stage_filter or [],
                                multi=True,
                                placeholder="All types",
                                className="activity-stage-dropdown",
                            ),
                        ],
                        className="activity-filter-field activity-filter-wide",
                    ),
                    html.Div(
                        [
                            html.Label("Since", className="input-label"),
                            dbc.ButtonGroup(
                                [
                                    dbc.Button(
                                        lbl,
                                        id={"type": "activity-since-btn", "days": days},
                                        size="sm",
                                        color="primary" if since_days == days else "secondary",
                                        outline=since_days != days,
                                    )
                                    for days, lbl in [(1, "24h"), (7, "7d"),
                                                      (30, "30d"), (0, "All")]
                                ],
                            ),
                        ],
                        className="activity-filter-field",
                    ),
                ],
                className="activity-filter-bar",
            ),

            dcc.Loading(
                html.Div(id="activity-runs", children=build_activity_section(activity_scope)),
                type="circle",
                color="#00D4AA",
                target_components={"activity-runs": "children"},
                overlay_style={"visibility": "visible", "opacity": 0.45},
                delay_show=350,
            ),

            html.Div("Past searches", className="scoreboard-subtitle"),
            html.Div(
                "Every watchlist you have pulled data for, kept server-side so "
                "it survives a cleared browser or a different machine.",
                className="history-scoreboard-hint",
            ),
            html.Div(_search_history_rows(searches or []), id="search-history"),
        ],
        className="page page-activity",
    )

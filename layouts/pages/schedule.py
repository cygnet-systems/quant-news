"""Schedule page: the jobs the app runs on its own clock.

This lived inside the History tab, which was wrong twice over: a schedule is
configuration rather than history, and burying it there meant the only way to
see when the 8am run last fired was to open an archive.
"""

from dash import dcc, html


def layout() -> html.Div:
    return html.Div(
        [
            html.Div(
                "Jobs run on the application's own clock, not an external cron. "
                "A job whose window has passed with no success recorded for "
                "today runs once on startup, so a restart cannot silently skip "
                "a day.",
                className="page-lede",
            ),
            # Both live on this page rather than at the app root: job state is
            # written by the scheduler thread (possibly in another instance),
            # so it has to be polled, and polling it from every other section
            # would be 4 pointless queries a minute.
            dcc.Interval(id="scheduler-refresh", interval=15_000),
            dcc.Store(id="scheduler-action-status"),
            # Populated by render_scheduler_panel.
            dcc.Loading(
                html.Div(id="scheduler-panel-container"),
                type="circle",
                color="#00D4AA",
                target_components={"scheduler-panel-container": "children"},
                overlay_style={"visibility": "visible", "opacity": 0.45},
                delay_show=350,
            ),
        ],
        className="page page-schedule",
    )

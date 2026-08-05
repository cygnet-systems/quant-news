"""Scheduler panel — read and edit the app's own job schedule.

Rendered inside the History tab beside the evaluation controls, because the
two answer the same question: what has the platform done for me lately, and
when will it do it again.

Every control is a pattern-matching id keyed by job, so the panel works for
however many jobs exist without the callbacks knowing their names.
"""

from __future__ import annotations

from datetime import datetime, timezone

import dash_bootstrap_components as dbc
from dash import html

_DAY_CHOICES = [
    {"label": "Weekdays", "value": "mon-fri"},
    {"label": "Every day", "value": "mon-fri,sat,sun"},
    {"label": "Mon/Wed/Fri", "value": "mon,wed,fri"},
]

_STATUS_STYLE = {
    "success": ("bi-check-circle-fill", "scheduler-status-ok"),
    "error": ("bi-x-circle-fill", "scheduler-status-error"),
    "skipped": ("bi-slash-circle", "scheduler-status-muted"),
    "running": ("bi-arrow-repeat", "scheduler-status-running"),
}


def _relative(when: datetime | None) -> str:
    """'12m ago' / 'in 3h' — a schedule is read in relative terms."""
    if not when:
        return "never"
    now = datetime.now(timezone.utc)
    stamp = when if when.tzinfo else when.replace(tzinfo=timezone.utc)
    delta = (stamp - now).total_seconds()
    ahead = delta > 0
    secs = abs(delta)
    if secs < 90:
        text = "just now" if not ahead else "in under a minute"
        return text
    if secs < 3600:
        value, unit = round(secs / 60), "m"
    elif secs < 86400:
        value, unit = round(secs / 3600), "h"
    else:
        value, unit = round(secs / 86400), "d"
    return f"in {value}{unit}" if ahead else f"{value}{unit} ago"


def _status_line(job: dict) -> html.Div:
    if job.get("running"):
        status = "running"
    else:
        status = (job.get("last_status") or "").lower()
    icon, cls = _STATUS_STYLE.get(status, ("bi-dash-circle", "scheduler-status-muted"))

    bits = [
        html.I(className=f"bi {icon} me-1"),
        html.Span(status or "not run yet", className=cls),
    ]
    if job.get("last_run_at"):
        bits.append(html.Span(f" · last {_relative(job['last_run_at'])}",
                              className="scheduler-meta"))
    if job.get("last_duration_ms"):
        bits.append(html.Span(f" · took {job['last_duration_ms'] // 1000}s",
                              className="scheduler-meta"))
    if job.get("next_run_at") and job.get("enabled"):
        bits.append(html.Span(f" · next {_relative(job['next_run_at'])}",
                              className="scheduler-meta"))
    elif not job.get("enabled"):
        bits.append(html.Span(" · disabled", className="scheduler-meta"))
    return html.Div(bits, className="scheduler-status-line")


def _job_card(job: dict) -> dbc.Card:
    job_id = job["id"]
    is_analysis = job["kind"] == "analysis"

    header = html.Div(
        [
            dbc.Switch(
                id={"type": "sched-enabled", "job": job_id},
                value=bool(job["enabled"]),
                className="scheduler-switch",
            ),
            html.Div(
                [
                    html.Span(job.get("description") or job_id,
                              className="scheduler-job-title"),
                    html.Code(job_id, className="scheduler-job-id"),
                ],
                className="scheduler-job-heading",
            ),
            dbc.Button(
                [html.I(className="bi bi-play-fill me-1"), "Run now"],
                id={"type": "sched-run", "job": job_id},
                size="sm", color="primary", outline=True,
                disabled=bool(job.get("running")),
                className="ms-auto",
            ),
        ],
        className="scheduler-job-header",
    )

    schedule_row = html.Div(
        [
            html.Div(
                [
                    dbc.Label("Time", className="scheduler-label"),
                    html.Div(
                        [
                            dbc.Input(
                                id={"type": "sched-hour", "job": job_id},
                                type="number", min=0, max=23, step=1,
                                value=job["hour"], size="sm",
                                className="scheduler-time-input",
                            ),
                            html.Span(":", className="scheduler-time-sep"),
                            dbc.Input(
                                id={"type": "sched-minute", "job": job_id},
                                type="number", min=0, max=59, step=1,
                                value=job["minute"], size="sm",
                                className="scheduler-time-input",
                            ),
                        ],
                        className="scheduler-time-group",
                    ),
                ],
                className="scheduler-field",
            ),
            html.Div(
                [
                    dbc.Label("Days", className="scheduler-label"),
                    dbc.Select(
                        id={"type": "sched-days", "job": job_id},
                        options=_DAY_CHOICES,
                        value=job["days_of_week"],
                        size="sm",
                    ),
                ],
                className="scheduler-field",
            ),
            html.Div(
                [
                    dbc.Label("Timezone", className="scheduler-label"),
                    dbc.Select(
                        id={"type": "sched-tz", "job": job_id},
                        options=[{"label": tz, "value": tz} for tz in
                                 ("US/Eastern", "US/Central", "US/Pacific", "UTC")],
                        value=job["timezone"],
                        size="sm",
                    ),
                ],
                className="scheduler-field",
            ),
        ],
        className="scheduler-row",
    )

    children = [header, _status_line(job), schedule_row]

    if is_analysis:
        children.append(
            html.Div(
                [
                    dbc.Label("Symbols", className="scheduler-label"),
                    dbc.Textarea(
                        id={"type": "sched-symbols", "job": job_id},
                        value=job.get("symbols_csv") or "",
                        placeholder="PANW,BAC,VZ,…",
                        rows=2, size="sm",
                        className="scheduler-symbols",
                    ),
                    html.Small(
                        "Comma-separated. The run targets the session opening "
                        "that morning, with data cut off at the previous close.",
                        className="scheduler-hint",
                    ),
                ],
                className="scheduler-field scheduler-field-wide",
            )
        )

    children.append(
        html.Div(
            [
                dbc.Button(
                    [html.I(className="bi bi-save me-1"), "Save schedule"],
                    id={"type": "sched-save", "job": job_id},
                    size="sm", color="secondary", outline=True,
                ),
                html.Span(id={"type": "sched-feedback", "job": job_id},
                          className="scheduler-feedback"),
            ],
            className="scheduler-actions",
        )
    )

    if job.get("last_detail"):
        children.append(
            html.Details(
                [
                    html.Summary("Last run output", className="scheduler-detail-summary"),
                    html.Pre(job["last_detail"], className="scheduler-detail"),
                ],
                className="scheduler-details",
            )
        )

    return dbc.Card(dbc.CardBody(children), className="scheduler-card")


def build_scheduler_panel(jobs: list[dict], runs: list[dict] | None = None) -> html.Div:
    """The whole panel: one card per job, then recent run history."""
    if not jobs:
        return html.Div(
            "No scheduled jobs. The scheduler seeds its defaults on startup — "
            "if this stays empty, the app could not reach the database.",
            className="scheduler-empty",
        )

    children = [_job_card(job) for job in jobs]

    if runs:
        rows = [
            html.Tr([
                html.Td(r["job_id"], className="scheduler-run-job"),
                html.Td(r["trigger"]),
                html.Td(r["status"],
                        className=_STATUS_STYLE.get(r["status"], ("", ""))[1]),
                html.Td(f"{(r['duration_ms'] or 0) // 1000}s"),
                html.Td(_relative(r["started_at"]), className="scheduler-meta"),
            ])
            for r in runs
        ]
        children.append(
            html.Div(
                [
                    html.H6("Recent runs", className="scheduler-subhead"),
                    dbc.Table(
                        [html.Thead(html.Tr([html.Th(h) for h in
                                             ("Job", "Trigger", "Status", "Took", "When")])),
                         html.Tbody(rows)],
                        size="sm", borderless=True, hover=True,
                        className="scheduler-run-table",
                    ),
                ],
                className="scheduler-history",
            )
        )

    return html.Div(children, className="scheduler-panel")

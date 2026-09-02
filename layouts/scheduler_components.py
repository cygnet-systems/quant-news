"""Scheduler panel: read and edit the app's own job schedule.

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
    """'12m ago' / 'in 3h', a schedule is read in relative terms."""
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


def _run_log_block(detail: str, label: str = "Run output") -> html.Details:
    """A run's captured output, collapsed until asked for.

    Scrolls inside its own box: a 400-line log that pushes the schedule off
    the screen is a log nobody expands twice.
    """
    return html.Details(
        [
            html.Summary(
                [label,
                 html.Span(f" · {len(detail.splitlines())} lines",
                           className="scheduler-meta")],
                className="scheduler-detail-summary",
            ),
            html.Pre(detail, className="scheduler-detail"),
        ],
        className="scheduler-details",
    )


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
    # Whether this operation takes a symbol list is the type's business, not
    # something this renderer should know per kind.
    is_analysis = job.get("needs_symbols", job["kind"] == "analysis")
    # can_manage is decided server-side (owner, admin, or legacy unowned). 
    # the service re-checks on every write, so this only shapes the UI.
    can_manage = job.get("can_manage", True)

    if job.get("is_mine"):
        owner_badge = dbc.Badge("mine", color="success",
                                className="scheduler-owner-badge",
                                title="You own this schedule")
    elif job.get("owner_uid"):
        owner_badge = dbc.Badge(f"by {job['owner_uid']}", color="dark",
                                className="scheduler-owner-badge",
                                title="Owned by another user")
    else:
        owner_badge = None
    visibility_badge = None
    if not job.get("is_public", True):
        visibility_badge = dbc.Badge(
            [html.I(className="bi bi-lock-fill me-1"), "private"],
            color="warning", className="scheduler-owner-badge",
            title="Only you (and Administrators) see this schedule; its "
                  "runs write private predictions and reports",
        )

    header = html.Div(
        [
            dbc.Switch(
                id={"type": "sched-enabled", "job": job_id},
                value=bool(job["enabled"]),
                disabled=not can_manage,
                className="scheduler-switch",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Span(job.get("description") or job_id,
                                      className="scheduler-job-title"),
                            dbc.Badge(job.get("type_label") or job["kind"],
                                      color="secondary", className="scheduler-type-badge"),
                            owner_badge,
                            visibility_badge,
                        ],
                        className="scheduler-title-row",
                    ),
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
            dbc.Button(
                html.I(className="bi bi-trash"),
                id={"type": "sched-delete", "job": job_id},
                size="sm", color="danger", outline=True,
                disabled=not can_manage,
                title="Delete this job (its run history is kept)"
                      if can_manage else
                      f"Only {job.get('owner_uid') or 'the owner'} or an "
                      f"Administrator can delete this job",
                className="scheduler-delete-btn",
            ),
        ],
        className="scheduler-job-header",
    )

    from layouts.schedule_modal import settings_summary

    when = (f"{job['hour']:02d}:{job['minute']:02d} {job['timezone']} · "
            f"{job['days_of_week']}")
    syms = [x for x in (job.get("symbols_csv") or "").split(",") if x.strip()]
    summary = html.Div(
        [
            html.Div([html.I(className="bi bi-clock me-1"), when],
                     className="scheduler-meta"),
            html.Div([html.I(className="bi bi-list-ul me-1"),
                      f"{len(syms)} symbols: " + ", ".join(syms[:8])
                      + ("…" if len(syms) > 8 else "")],
                     className="scheduler-meta") if is_analysis else None,
            html.Div([html.I(className="bi bi-sliders me-1"), settings_summary(job)],
                     className="scheduler-meta"),
        ],
        className="scheduler-summary",
    )

    actions = html.Div(
        [
            dbc.Button(
                [html.I(className="bi bi-pencil me-1"), "Edit schedule & settings"],
                id={"type": "sched-edit", "job": job_id},
                size="sm", color="secondary", outline=True,
                disabled=not can_manage,
            ),
        ],
        className="scheduler-actions",
    )

    children = [header, _status_line(job), summary, actions]

    if job.get("last_detail"):
        children.append(_run_log_block(job["last_detail"], "Last run output"))

    return dbc.Card(dbc.CardBody(children), className="scheduler-card")


def build_scheduler_panel(jobs: list[dict], runs: list[dict] | None = None,
                          job_types: list[dict] | None = None) -> html.Div:
    """The whole panel: one card per job, then recent run history."""
    job_types = job_types or []

    new_btn = html.Div(
        dbc.Button([html.I(className="bi bi-plus-lg me-1"), "New schedule"],
                   id="sched-new-open", size="sm", color="primary"),
        className="scheduler-actions mb-3",
    )
    if not jobs:
        children = [new_btn, html.Div(
            "No scheduled jobs. Create one with the button above; the "
            "defaults are only seeded into an empty schedule at startup, so "
            "a job you delete stays deleted.",
            className="scheduler-empty",
        )]
    else:
        children = [new_btn] + [_job_card(job) for job in jobs]

    if runs:
        rows = []
        for r in runs:
            rows.append(html.Tr([
                html.Td(r["job_id"], className="scheduler-run-job"),
                html.Td(r["trigger"]),
                html.Td(r["status"],
                        className=_STATUS_STYLE.get(r["status"], ("", ""))[1]),
                html.Td(f"{(r['duration_ms'] or 0) // 1000}s"),
                html.Td(_relative(r["started_at"]), className="scheduler-meta"),
            ]))
            # Every run carries its own log, expandable in place. A status
            # column alone cannot answer "what happened", which is the only
            # question anyone opens this table to ask.
            if r.get("detail"):
                rows.append(html.Tr(
                    html.Td(_run_log_block(r["detail"]), colSpan=5,
                            className="scheduler-run-logcell"),
                    className="scheduler-run-logrow",
                ))
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

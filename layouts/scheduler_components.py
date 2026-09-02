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
    # can_manage is decided server-side (owner, admin, or legacy unowned) —
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
                                disabled=not can_manage,
                                className="scheduler-time-input",
                            ),
                            html.Span(":", className="scheduler-time-sep"),
                            dbc.Input(
                                id={"type": "sched-minute", "job": job_id},
                                type="number", min=0, max=59, step=1,
                                value=job["minute"], size="sm",
                                disabled=not can_manage,
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
                        disabled=not can_manage,
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
                        disabled=not can_manage,
                    ),
                ],
                className="scheduler-field",
            ),
            html.Div(
                [
                    dbc.Label("Visibility", className="scheduler-label"),
                    dbc.Select(
                        id={"type": "sched-visibility", "job": job_id},
                        options=[
                            {"label": "Public", "value": "public"},
                            {"label": "Private (only me)", "value": "private"},
                        ],
                        value="public" if job.get("is_public", True) else "private",
                        size="sm",
                        # Private needs an owner to be visible to anyone at
                        # all; the service enforces the same rule on write.
                        disabled=not can_manage or not job.get("owner_uid"),
                    ),
                ],
                className="scheduler-field",
            ),
        ],
        className="scheduler-row",
    )

    children = [header, _status_line(job), schedule_row]

    # The type's tuning knobs, editable in place. They used to be settable
    # only on the create form and invisible afterwards — a 30-day news
    # window chosen at creation could never be seen or changed again.
    spec = job.get("params_spec") or []
    if spec:
        params = job.get("params") or {}
        children.append(html.Div(
            [
                html.Div(
                    [
                        dbc.Label(item["label"], className="scheduler-label"),
                        dbc.Input(
                            id={"type": "sched-param", "job": job_id,
                                "key": item["key"]},
                            type="number",
                            value=params.get(item["key"], item["default"]),
                            size="sm",
                            disabled=not can_manage,
                            className="scheduler-time-input",
                        ),
                        html.Small(item["help"], className="scheduler-hint"),
                    ],
                    className="scheduler-field",
                )
                for item in spec
            ],
            className="scheduler-row scheduler-params-row",
        ))

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
                        disabled=not can_manage,
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
                    disabled=not can_manage,
                ),
                html.Span(id={"type": "sched-feedback", "job": job_id},
                          className="scheduler-feedback"),
            ],
            className="scheduler-actions",
        )
    )

    if job.get("last_detail"):
        children.append(_run_log_block(job["last_detail"], "Last run output"))

    return dbc.Card(dbc.CardBody(children), className="scheduler-card")


def _create_form(job_types: list[dict]) -> html.Div:
    """Add a job of any registered operation type.

    The symbol box is always rendered but only meaningful for types that take
    one; the service ignores it otherwise, so the form does not have to
    re-render on every type change to stay honest.
    """
    takes_symbols = [t["kind"] for t in job_types if t["needs_symbols"]]
    return html.Details(
        [
            html.Summary(
                [html.I(className="bi bi-plus-circle me-1"), "New scheduled job"],
                className="scheduler-new-summary",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                [
                                    dbc.Label("Operation", className="scheduler-label"),
                                    dbc.Select(
                                        id="sched-new-kind",
                                        options=[{"label": t["label"], "value": t["kind"]}
                                                 for t in job_types],
                                        value=job_types[0]["kind"] if job_types else None,
                                        size="sm",
                                    ),
                                ],
                                className="scheduler-field",
                            ),
                            html.Div(
                                [
                                    dbc.Label("Name", className="scheduler-label"),
                                    dbc.Input(id="sched-new-name", type="text", size="sm",
                                              placeholder="Morning predict"),
                                ],
                                className="scheduler-field scheduler-field-grow",
                            ),
                            html.Div(
                                [
                                    dbc.Label("Time", className="scheduler-label"),
                                    html.Div(
                                        [
                                            dbc.Input(id="sched-new-hour", type="number",
                                                      min=0, max=23, value=8, size="sm",
                                                      className="scheduler-time-input"),
                                            html.Span(":", className="scheduler-time-sep"),
                                            dbc.Input(id="sched-new-minute", type="number",
                                                      min=0, max=59, value=30, size="sm",
                                                      className="scheduler-time-input"),
                                        ],
                                        className="scheduler-time-group",
                                    ),
                                ],
                                className="scheduler-field",
                            ),
                            html.Div(
                                [
                                    dbc.Label("Days", className="scheduler-label"),
                                    dbc.Select(id="sched-new-days", options=_DAY_CHOICES,
                                               value="mon-fri", size="sm"),
                                ],
                                className="scheduler-field",
                            ),
                            html.Div(
                                [
                                    dbc.Label("Timezone", className="scheduler-label"),
                                    dbc.Select(
                                        id="sched-new-tz",
                                        options=[{"label": tz, "value": tz} for tz in
                                                 ("US/Eastern", "US/Central",
                                                  "US/Pacific", "UTC")],
                                        value="US/Eastern", size="sm",
                                    ),
                                ],
                                className="scheduler-field",
                            ),
                            html.Div(
                                [
                                    dbc.Label("Visibility", className="scheduler-label"),
                                    dbc.Select(
                                        id="sched-new-visibility",
                                        options=[
                                            {"label": "Private (only me)",
                                             "value": "private"},
                                            {"label": "Public", "value": "public"},
                                        ],
                                        # Your schedule is yours by default;
                                        # anonymous sessions are forced public
                                        # by the service (nobody could see an
                                        # ownerless private job again).
                                        value="private", size="sm",
                                    ),
                                ],
                                className="scheduler-field",
                            ),
                        ],
                        className="scheduler-row",
                    ),
                    # Tunable knobs per operation type, from its declarative
                    # params_spec. Every knob for every type is rendered once
                    # (hidden ids can't be read by the create callback in Dash
                    # 4); the container for a type shows only while that type
                    # is selected, so the visible form stays minimal.
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            dbc.Label(spec["label"],
                                                      className="scheduler-label"),
                                            dbc.Input(
                                                id={"type": "sched-new-param",
                                                    "kind": t["kind"],
                                                    "key": spec["key"]},
                                                type="number",
                                                value=spec["default"],
                                                size="sm",
                                                className="scheduler-time-input",
                                            ),
                                            html.Small(spec["help"],
                                                       className="scheduler-hint"),
                                        ],
                                        className="scheduler-field",
                                    )
                                    for spec in t["params_spec"]
                                ],
                                id={"type": "sched-new-params-group",
                                    "kind": t["kind"]},
                                className="scheduler-row scheduler-params-row",
                                style={"display": "none"},
                            )
                            for t in job_types if t.get("params_spec")
                        ],
                        id="sched-new-params",
                    ),
                    html.Div(
                        [
                            dbc.Label("Symbols", className="scheduler-label"),
                            dbc.Textarea(id="sched-new-symbols", rows=2, size="sm",
                                         placeholder="PANW,BAC,VZ,…",
                                         className="scheduler-symbols"),
                            html.Small(
                                "Used by: " + (", ".join(takes_symbols) or "no operation"),
                                className="scheduler-hint",
                            ),
                        ],
                        className="scheduler-field scheduler-field-wide",
                    ),
                    html.Div(
                        [
                            dbc.Button(
                                [html.I(className="bi bi-plus-lg me-1"), "Create job"],
                                id="sched-create", size="sm", color="primary",
                            ),
                            html.Span(id="sched-create-feedback",
                                      className="scheduler-feedback"),
                        ],
                        className="scheduler-actions",
                    ),
                ],
                className="scheduler-new-body",
            ),
        ],
        className="scheduler-new",
    )


def build_scheduler_panel(jobs: list[dict], runs: list[dict] | None = None,
                          job_types: list[dict] | None = None) -> html.Div:
    """The whole panel: one card per job, then recent run history."""
    job_types = job_types or []

    if not jobs:
        children = [html.Div(
            "No scheduled jobs. Create one below — the defaults are only "
            "seeded into an empty schedule at startup, so a job you delete "
            "stays deleted.",
            className="scheduler-empty",
        )]
    else:
        children = [_job_card(job) for job in jobs]

    if job_types:
        children.append(_create_form(job_types))

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

"""The Schedule modal: create, edit or delete a scheduled job.

One dialog for a job's whole configuration. The schedule fields (when, days,
timezone, visibility, symbols) sit above the SAME settings sections the Run
dialog renders (services -> layouts.modals.run_settings_sections with the
"sj" prefix), so a scheduled analysis job is a saved Run dialog rather than
the two-knob subset the old inline card form exposed.
"""

import dash_bootstrap_components as dbc
from dash import dcc, html

from layouts.modals import run_settings_sections
from layouts.scheduler_components import _DAY_CHOICES

TIMEZONES = ("US/Eastern", "US/Central", "US/Pacific", "UTC")


def _field(label, control, hint=None, grow=False):
    return html.Div(
        [dbc.Label(label, className="scheduler-label"), control,
         html.Small(hint, className="scheduler-hint") if hint else None],
        className="scheduler-field" + (" scheduler-field-grow" if grow else ""),
    )


def create_schedule_modal(job_types: list[dict]) -> dbc.Modal:
    """Built once on the Schedule page; the open callback fills its values."""
    kinds = [{"label": t["label"], "value": t["kind"]} for t in job_types]

    schedule_rows = html.Div(
        [
            html.Div(
                [
                    _field("Operation", dbc.Select(id="sj-kind", options=kinds,
                                                   value=kinds[0]["value"] if kinds else None,
                                                   size="sm")),
                    _field("Name", dbc.Input(id="sj-name", type="text", size="sm",
                                             placeholder="Morning predict"), grow=True),
                ],
                className="scheduler-row",
            ),
            html.Div(
                [
                    _field("Time", html.Div(
                        [
                            dbc.Input(id="sj-hour", type="number", min=0, max=23,
                                      value=7, size="sm", className="scheduler-time-input"),
                            html.Span(":", className="scheduler-time-sep"),
                            dbc.Input(id="sj-minute", type="number", min=0, max=59,
                                      value=0, size="sm", className="scheduler-time-input"),
                        ],
                        className="scheduler-time-group",
                    )),
                    _field("Days", dbc.Select(id="sj-days", options=_DAY_CHOICES,
                                              value="mon-fri", size="sm")),
                    _field("Timezone", dbc.Select(
                        id="sj-tz", options=[{"label": tz, "value": tz} for tz in TIMEZONES],
                        value="US/Eastern", size="sm")),
                    _field("Visibility", dbc.Select(
                        id="sj-visibility",
                        options=[{"label": "Private (only me)", "value": "private"},
                                 {"label": "Public", "value": "public"}],
                        value="private", size="sm")),
                    _field("Enabled", dbc.Switch(id="sj-enabled", value=True,
                                                 className="scheduler-switch")),
                ],
                className="scheduler-row",
            ),
            html.Div(
                _field("Symbols", dbc.Textarea(id="sj-symbols", rows=2, size="sm",
                                               placeholder="PANW,BAC,VZ,…",
                                               className="scheduler-symbols"),
                       "Comma-separated. The run targets the session opening that "
                       "morning, with data cut off at the previous close."),
                id="sj-symbols-wrap",
                className="scheduler-field scheduler-field-wide",
            ),
        ]
    )

    # Tuning knobs for the non-analysis types, from their params_spec.
    param_groups = [
        html.Div(
            [
                _field(spec["label"],
                       dbc.Input(id={"type": "sj-param", "kind": t["kind"],
                                     "key": spec["key"]},
                                 type="number", value=spec["default"], size="sm",
                                 className="scheduler-time-input"),
                       spec["help"])
                for spec in t["params_spec"]
            ],
            id={"type": "sj-params-group", "kind": t["kind"]},
            className="scheduler-row scheduler-params-row",
            style={"display": "none"},
        )
        for t in job_types if t.get("params_spec")
    ]

    return dbc.Modal(
        [
            dcc.Store(id="sj-job-id"),
            dbc.ModalHeader(dbc.ModalTitle(id="sj-title"), close_button=True),
            dbc.ModalBody([
                schedule_rows,
                html.Div(param_groups, id="sj-params"),
                # The Run dialog's settings, verbatim, namespaced "sj".
                html.Div(run_settings_sections("sj"), id="sj-analysis-section"),
            ]),
            dbc.ModalFooter([
                html.Div(id="sj-feedback", className="run-validation-msg me-auto"),
                dbc.Button([html.I(className="bi bi-trash me-1"), "Delete"],
                           id="sj-delete", color="danger", outline=True,
                           className="me-2"),
                dbc.Button("Cancel", id="sj-cancel", color="secondary", className="me-2"),
                dbc.Button([html.I(className="bi bi-save me-1"), "Save"],
                           id="sj-save", color="success"),
            ]),
        ],
        id="sj-modal", is_open=False, size="lg", scrollable=True,
    )


def settings_summary(job: dict) -> str:
    """One line describing an analysis job's saved run settings."""
    p = job.get("params") or {}
    if job.get("kind") != "analysis":
        knobs = {k: v for k, v in p.items() if k != "only_trading_days"}
        return ", ".join(f"{k}={v}" for k, v in knobs.items()) or "defaults"
    window = p.get("lookback")
    window = "overnight" if str(window).lower() == "overnight" else f"{window}d news"
    models = p.get("models")
    n_models = len(models) if isinstance(models, list) else "all"
    bits = [
        window,
        f"cap {p.get('max_articles', '?')}",
        f"{n_models} models" + ("" if p.get("run_ensemble", True) else ", no ensemble"),
        f"report {p.get('report_model', '?')} ({p.get('depth', 'thesis')})",
        f"recs {p.get('recs', 'auto')}",
        f"{len(p.get('evidence') or [])} evidence blocks",
        ("web research on" if "web_research" in (p.get("tools") or [])
         else "web research off"),
    ]
    return " · ".join(bits)

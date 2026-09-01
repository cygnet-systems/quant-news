"""Trace page: one run, fully joined back together.

The Activity page answers "what did the system do"; this page answers "why
did it conclude this" for a single run — the data that was fetched (counts,
windows, cache hits with their hash), each model's inputs and outcome, and
every LLM call down to the exact prompts and raw response.

Everything renders from Postgres (activity_log payloads, model_predictions
by run_id, llm_traces), so a run is inspectable while it executes and
forever after. The LLM list deliberately renders envelopes only; prompt and
response bodies are fetched one call at a time when expanded — a run can
hold dozens of multi-kilobyte prompts and the list must not drag them all
through every poll.
"""

import json

import dash_bootstrap_components as dbc
from dash import dcc, html

from layouts.history_sections import collapsible_section
from services import progress_service as _prog

POLL_ACTIVE_MS = 1500
POLL_IDLE_MS = 10_000
# Selector depth. Anything older is reachable through the Activity page;
# the dropdown says so when it fills up rather than growing pagination.
RUN_LIMIT = 50

# Payload events that belong to the Data group (model_run rows render in the
# Models group instead).
_DATA_EVENTS = ("news_window", "data_load", "enrichment", "cache")


def run_options(runs: list[dict]) -> list[dict]:
    """Dropdown options for the selector — shared by the initial render and
    the poll's refresh so both agree on the cap hint."""
    options = [{"label": r["label"], "value": r["run_id"]} for r in runs]
    if len(runs) >= RUN_LIMIT:
        options.append({"label": "Older runs: see the Activity page",
                        "value": "_older_runs_hint", "disabled": True})
    return options


def list_trace_runs(limit: int = RUN_LIMIT) -> list[dict]:
    """Recent runs for the selector, newest first, deduped by run_id.

    Reuses the activity log's run grouping (and therefore its admin/user
    scoping): title + start time + a coarse status derived from the events.
    """
    from datetime import datetime, timedelta, timezone

    from services import progress_service as prog

    out, seen = [], set()
    for run in prog.get_activity_runs(limit_runs=limit):
        if run["run_id"] in seen:
            continue
        seen.add(run["run_id"])
        # A run with no completion boundary is only "running" while its feed
        # is fresh — an abandoned run (crashed process, killed scheduler)
        # never writes one, and must not claim to be running hours later.
        stale = False
        try:
            ended = run["ended"]
            if ended.tzinfo is None:
                ended = ended.replace(tzinfo=timezone.utc)
            stale = (datetime.now(timezone.utc) - ended
                     > timedelta(minutes=30))
        except Exception:
            pass
        if any(e["stage"] == "done" for e in run["events"]):
            status = "finished with errors" if run["errors"] else "done"
        elif stale:
            status = "incomplete"
        elif run["errors"]:
            status = "errors"
        else:
            status = "running"
        out.append({
            "run_id": run["run_id"],
            "label": (f"{run['started']:%m-%d %H:%M} — "
                      f"{(run['title'] or 'Activity')[:70]} · {status}"),
            "status": status,
        })
    return out


def layout(runs: list[dict] | None = None,
           selected_run: str | None = None) -> html.Div:
    """Run selector + the three trace groups, polled while the run is live."""
    runs = runs if runs is not None else list_trace_runs()
    if selected_run is None and runs:
        selected_run = runs[0]["run_id"]
        # A run in flight is what the viewer came to watch — select it over
        # the merely-most-recent one.
        try:
            from services import progress_service as prog
            if prog.get_feed().get("active"):
                live = prog.current_run_id()
                if any(r["run_id"] == live for r in runs):
                    selected_run = live
        except Exception:
            pass

    # Seed the watermark store with the state this render used, so the poll's
    # first tick has nothing to rebuild.
    from services import trace_service
    marks = (trace_service.run_watermarks(selected_run) if selected_run
             else {"events": 0, "traces": 0})
    seed = {"run_id": selected_run, "events": marks["events"],
            "traces": marks["traces"], "poll": None, "active": None,
            "run_marker": trace_service.latest_run_marker(),
            "runs": [r["run_id"] for r in runs]}
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("Run", className="input-label"),
                            dcc.Dropdown(
                                id="trace-run-select",
                                options=run_options(runs),
                                value=selected_run,
                                clearable=False,
                                placeholder="No runs recorded yet",
                                className="activity-stage-dropdown",
                            ),
                        ],
                        className="activity-filter-field activity-filter-wide",
                    ),
                ],
                className="activity-filter-bar",
            ),
            html.Div(id="trace-body", children=build_trace_body(selected_run)),
            dcc.Store(id="trace-watermark-store", data=seed),
            dcc.Interval(id="trace-interval", interval=POLL_ACTIVE_MS),
        ],
        className="page page-trace",
    )


def build_trace_body(run_id: str | None) -> html.Div:
    """The three trace groups, each in its own wrapper.

    The wrappers are what the poll callback rewrites — separately, per
    watermark — so fresh Data events cannot destroy an LLM body the user has
    expanded, and vice versa. They must exist even with no run selected:
    the callback's Outputs target them unconditionally.
    """
    return html.Div([
        html.Div(build_data_section(run_id), id="trace-data-wrap"),
        html.Div(build_models_section(run_id), id="trace-models-wrap"),
        html.Div(build_llm_section(run_id), id="trace-llm-wrap"),
    ])


def build_data_section(run_id: str | None) -> html.Div:
    if not run_id:
        return _no_run()
    from services import trace_service
    events = [e for e in trace_service.list_run_events(run_id)
              if (e.get("payload") or {}).get("event") in _DATA_EVENTS]
    return collapsible_section(
        "Data", "trace-data", _data_group(events),
        icon_class="bi-database", default_open=True, count=len(events))


def build_models_section(run_id: str | None) -> html.Div:
    if not run_id:
        return html.Div()
    from services import trace_service
    model_events = [e for e in trace_service.list_run_events(run_id)
                    if (e.get("payload") or {}).get("event") == "model_run"]
    preds = trace_service.list_run_predictions(run_id)
    return collapsible_section(
        "Models", "trace-models", _model_group(model_events, preds),
        icon_class="bi-cpu", default_open=True,
        count=max(len(model_events), len(preds)))


def build_llm_section(run_id: str | None) -> html.Div:
    if not run_id:
        return html.Div()
    from services import trace_service
    calls = trace_service.list_llm_calls(run_id)
    return collapsible_section(
        "LLM calls", "trace-llm", _llm_group(calls),
        icon_class="bi-chat-square-text", default_open=True, count=len(calls))


def _no_run() -> html.Div:
    return html.Div(
        [
            html.Div("No run selected", style={"fontWeight": "600",
                                               "marginBottom": "4px"}),
            html.Div("Start an analysis, or pick a past run above.",
                     style={"color": "var(--text-secondary)",
                            "fontSize": "0.85rem"}),
        ],
        className="history-empty-msg",
    )


# ---------------------------------------------------------------------------
# Data group
# ---------------------------------------------------------------------------


def _empty(msg: str) -> html.Div:
    return html.Div(msg, className="history-empty-msg")


def _kv(label: str, value) -> html.Span:
    return html.Span(
        [html.Span(f"{label}: ", style={"color": "var(--text-muted)"}),
         html.Span(str(value))],
        style={"marginRight": "14px", "whiteSpace": "nowrap"},
    )


def _payload_detail(payload: dict) -> html.Div:
    """One readable line (or block) per structured event type."""
    kind = payload.get("event")
    parts: list = []
    if kind == "news_window":
        parts.append(_kv("window",
                         f"{payload.get('filter')} "
                         f"{payload.get('lookback_days')}d → "
                         f"{payload.get('as_of')}"))
        parts.append(_kv("articles", payload.get("articles")))
        per_sym = payload.get("articles_by_symbol") or {}
        status = payload.get("source_status") or {}
        if per_sym:
            parts.append(_kv("per symbol", ", ".join(
                f"{s}:{n}" + (f" ({status[s]})" if status.get(s) not in
                              (None, "ok") else "")
                for s, n in sorted(per_sym.items()))))
    elif kind == "data_load":
        by_sym = payload.get("by_symbol") or {}
        parts.append(_kv("symbols", ", ".join(
            f"{s}: {v.get('bars', 0)} bars"
            + (f" {v.get('start')}→{v.get('end')}" if v.get("end") else "")
            + (f" ({v.get('source')})" if v.get("source") else "")
            + (" — FAILED" if v.get("error") else "")
            for s, v in sorted(by_sym.items()))))
    elif kind == "enrichment":
        by_sym = payload.get("by_symbol") or {}
        parts.append(_kv("evidence", ", ".join(payload.get("evidence") or [])))
        parts.append(_kv("symbols", ", ".join(
            f"{s}: options {v.get('options')}, quality {v.get('quality')}"
            + (f" ({v['quality_fails']} fails)"
               if v.get("quality_fails") is not None else "")
            for s, v in sorted(by_sym.items()))))
    elif kind == "cache":
        outcome = str(payload.get("outcome") or "?").upper()
        parts.append(html.Span(
            f"{payload.get('kind')} · {outcome}",
            style={"fontWeight": "600", "marginRight": "14px",
                   "color": ("var(--positive)" if outcome == "HIT"
                             else "var(--neutral)")}))
        if payload.get("input_data_hash"):
            parts.append(_kv("hash", str(payload["input_data_hash"])[:16]))
        for k, v in (payload.get("summary") or {}).items():
            if isinstance(v, dict):
                v = ", ".join(f"{a}:{b}" for a, b in sorted(v.items()))
            elif isinstance(v, list):
                v = ", ".join(str(x) for x in v)
            parts.append(_kv(k, v))
    else:
        # Unknown payloads still render — raw, but never lost.
        parts.append(html.Pre(json.dumps(payload, indent=2, default=str),
                              className="trace-pre"))
    return html.Div(parts, className="trace-payload-detail",
                    style={"fontSize": "0.8rem", "paddingLeft": "20px",
                           "display": "flex", "flexWrap": "wrap"})


def _data_group(events: list[dict]) -> html.Div:
    if not events:
        return _empty("No structured data events recorded for this run.")
    rows = []
    for e in events:
        rows.append(html.Div(
            [
                html.Div(
                    [
                        html.Span(_prog.format_clock(e["ts"])
                                  if hasattr(e["ts"], "strftime") else str(e["ts"]),
                                  className="progress-ts",
                                  title=_prog.DISPLAY_TZ_LABEL),
                        html.Span(e["message"], className="progress-msg"),
                    ],
                    className="progress-line",
                ),
                _payload_detail(e.get("payload") or {}),
            ],
            style={"marginBottom": "6px"},
        ))
    return html.Div(rows)


# ---------------------------------------------------------------------------
# Models group
# ---------------------------------------------------------------------------


def _model_group(model_events: list[dict], preds: list[dict]) -> html.Div:
    """One row per model × symbol: the live payload record joined with the
    persisted prediction (persisted rows exist only after the store stage)."""
    merged: dict[tuple, dict] = {}
    for p in preds:
        merged[(p["symbol"], p["model_name"])] = {**p, "persisted": True}
    for e in model_events:
        pl = e.get("payload") or {}
        key = (pl.get("symbol"), pl.get("model"))
        row = merged.setdefault(key, {"symbol": pl.get("symbol"),
                                      "model_name": pl.get("model"),
                                      "persisted": False})
        # The payload is the richer record (input summary, error, abstain);
        # decision/confidence from the persisted row win when present.
        row.setdefault("decision", pl.get("decision"))
        row.setdefault("confidence", pl.get("confidence"))
        row.setdefault("duration_ms", pl.get("duration_ms"))
        row["error"] = pl.get("error")
        row["abstained"] = pl.get("abstained")
        row["input"] = pl.get("input") or {}
    if not merged:
        return _empty("No model executions recorded for this run.")

    def _conf(v, model=""):
        # "Conf" meant two different things in this one column: the research
        # arm's number is a measured track-record weight (0.5 = none earned
        # yet), every other model's is its own raw score. Say which.
        if not isinstance(v, (int, float)):
            return "—"
        if model == "trading_agents":
            return "unrated" if float(v) == 0.5 else f"{v:.0%} measured"
        return f"{v:.0%} raw"

    def _dur(v):
        return f"{v / 1000:.1f}s" if isinstance(v, (int, float)) else "—"

    header = html.Thead(html.Tr([
        html.Th("Symbol"), html.Th("Model"), html.Th("Decision"),
        html.Th("Score",
                title="TradingAgents reports a measured track-record weight "
                      "(unrated until it has enough resolved calls); the other "
                      "models report their own raw, uncalibrated score. Neither "
                      "is the report's stated conviction."),
        html.Th("Time"), html.Th("Bars"),
        html.Th("Last bar"), html.Th("News"), html.Th("Status"),
    ]))
    body = []
    for (sym, model), row in sorted(merged.items(),
                                    key=lambda kv: (kv[0][0] or "",
                                                    kv[0][1] or "")):
        inp = row.get("input") or {}
        if row.get("abstained"):
            status = html.Span("abstained", style={"color": "var(--neutral)"})
        elif row.get("error"):
            status = html.Span(f"failed: {str(row['error'])[:60]}",
                               style={"color": "var(--negative)"})
        elif row.get("persisted"):
            status = "stored"
        else:
            status = "ran"
        # A failed or abstaining model made no call — its error-carrying
        # HOLD (0%) placeholder must never render as a decision.
        no_call = bool(row.get("error"))
        body.append(html.Tr([
            html.Td(sym or "—"),
            html.Td(model or "—"),
            html.Td("—" if no_call else (row.get("decision") or "—")),
            html.Td("—" if no_call else _conf(row.get("confidence"), model or ""),
                    className="num"),
            html.Td(_dur(row.get("duration_ms")), className="num"),
            html.Td(inp.get("bars", "—"), className="num"),
            html.Td(inp.get("last_bar") or "—", className="num"),
            html.Td(inp.get("news", "—"), className="num"),
            html.Td(status),
        ]))
    return html.Div(
        html.Table([header, html.Tbody(body)], className="history-data-table"),
        className="history-table-wrap",
    )


# ---------------------------------------------------------------------------
# LLM calls group
# ---------------------------------------------------------------------------


def _llm_group(calls: list[dict]) -> html.Div:
    if not calls:
        return _empty("No LLM calls recorded for this run.")

    groups: dict[str, list[dict]] = {}
    for c in calls:
        label = c.get("section") or c.get("stage") or "unknown"
        groups.setdefault(label, []).append(c)

    blocks = []
    for label, group in groups.items():
        blocks.append(html.Div(label, className="scoreboard-subtitle"))
        for c in group:
            blocks.append(_llm_call_row(c))
    return html.Div(blocks)


def _llm_call_row(c: dict) -> html.Div:
    ok = bool(c.get("ok"))
    tokens = "—"
    if c.get("input_tokens") is not None or c.get("output_tokens") is not None:
        tokens = f"{c.get('input_tokens') or 0}→{c.get('output_tokens') or 0}"
    cost = (f"${c['cost_usd']:.4f}" if isinstance(c.get("cost_usd"), float)
            else "—")
    dur = (f"{c['duration_ms'] / 1000:.1f}s"
           if isinstance(c.get("duration_ms"), (int, float)) else "—")
    ts = c.get("created_at")
    ts_str = (_prog.format_clock(ts) if hasattr(ts, "strftime") else str(ts))

    summary = [
        html.Span(ts_str, className="progress-ts"),
        html.I(className="bi "
               + ("bi-check-circle" if ok else "bi-x-circle")
               + " progress-icon",
               style={"color": "var(--positive)" if ok
                      else "var(--negative)"}),
        html.Span(f"{c.get('provider') or '?'} / {c.get('model') or '?'}",
                  style={"marginRight": "12px"}),
        _kv("attempt", c.get("attempt")),
        _kv("time", dur),
        _kv("tokens", tokens),
        _kv("cost", cost),
    ]
    if c.get("symbol"):
        summary.append(_kv("symbol", c["symbol"]))
    if not ok and c.get("error"):
        summary.append(html.Span(str(c["error"])[:80],
                                 style={"color": "var(--negative)"}))
    summary.append(dbc.Button(
        "Prompts & response",
        id={"type": "trace-llm-expand", "id": c["id"]},
        size="sm", color="secondary", outline=True,
        className="ms-auto trace-expand-btn",
    ))
    return html.Div(
        [
            html.Div(summary, className="progress-line",
                     style={"display": "flex", "alignItems": "center",
                            "flexWrap": "wrap", "gap": "4px"}),
            html.Div(id={"type": "trace-llm-body", "id": c["id"]},
                     style={"display": "none"}),
        ],
        className="trace-llm-call",
        style={"marginBottom": "6px"},
    )


def render_llm_bodies(bodies: dict | None) -> html.Div:
    """The expanded view of one call — fetched only when the row is opened."""
    if not bodies:
        return _empty("Trace bodies not found (or not visible to you).")

    pre_style = {
        "whiteSpace": "pre-wrap",
        "fontSize": "0.75rem",
        "background": "var(--bg-tertiary)",
        "border": "1px solid var(--border-subtle)",
        "borderRadius": "4px",
        "padding": "8px",
        "maxHeight": "420px",
        "overflowY": "auto",
        "color": "var(--text-primary)",
    }

    def _section(title: str, content: str | None):
        if not content:
            return None
        return html.Div([
            html.Div(title, style={"fontWeight": "600", "fontSize": "0.8rem",
                                   "margin": "8px 0 4px"}),
            html.Pre(content, style=pre_style),
        ])

    params = bodies.get("params")
    sections = [
        _section("Request params",
                 json.dumps(params, indent=2, default=str) if params else None),
        _section("System prompt", bodies.get("system_prompt")),
        _section("Prompt", bodies.get("prompt")),
        _section("Response", bodies.get("response")),
        _section("Error", bodies.get("error")),
    ]
    return html.Div([s for s in sections if s is not None],
                    style={"paddingLeft": "20px"})

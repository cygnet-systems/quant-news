"""The activity panel's body: a stepper over one run, the log beneath it.

The panel used to be the log alone. The log answers "what just happened";
it does not answer "how far along is NVDA" without reading forty lines.
The stepper reads the run row (analysis_runs.stages_json, written by
emit_progress) and draws one cell per stage and one line per symbol, so
the answer is a glance. The log keeps everything it had, folded under
"Details".

Which run the panel follows is decided here too, from the session's pin
and the feed's active list, before the callback spends its single row
read: the viewer's own run while it is in flight, for an hour after it
ended, or for as long as they pinned it by hand (a click on the pill),
else the newest run in flight, else nothing (the rolling log). One rule,
``pinnable``, answers on both sides of the row read. Everything in this
module is plain data in, Dash components out, so the tests need no
browser and no database.
"""

import hashlib
import json
from datetime import datetime, timezone

from dash import html

from services.run_service import ACTIVE_STATUSES, STAGES

# How long a finished run stays pinned in the session that started it:
# long enough to come back to the result, short enough that tomorrow's
# panel is not stuck on yesterday's run.
PIN_WINDOW_S = 3600

# Icons for the free-text log lines, by the stage the emitter named.
STAGE_ICONS = {
    "run": "bi-play-circle",
    "news": "bi-newspaper",
    "ai": "bi-file-text",
    "models": "bi-cpu",
    "ta": "bi-robot",
    "luna": "bi-stars",
    "store": "bi-database",
    "done": "bi-check-circle-fill",
    "error": "bi-x-circle-fill",
}

_GLYPHS = {
    "pending": "bi bi-circle",
    "running": "bi bi-arrow-repeat progress-glyph-spin",
    "done": "bi bi-check-circle-fill",
    "failed": "bi bi-x-circle-fill",
    "skipped": "bi bi-dash-circle",
}


def _parse_ts(value) -> datetime | None:
    if not value:
        return None
    try:
        ts = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    # Naive stamps come from SQLite rows, written in UTC.
    return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)


def within_window(value, now: datetime, window_s: int = PIN_WINDOW_S) -> bool:
    """Whether an ISO stamp lies in the last ``window_s`` seconds. A stamp
    in the future (clock skew between the row writer and this process)
    is not "recent": it never counts."""
    ts = _parse_ts(value)
    return ts is not None and 0 <= (now - ts).total_seconds() <= window_s


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def pinnable(run_id, active_ids, now, *, active=False, pinned=False,
             finished_at=None, started_at=None, ceiling_s=0) -> bool:
    """The one rule for whether a run is worth the panel's attention:
    in flight (the feed lists it, or its row says so), pinned by hand (a
    pill click; the flag lives in run-store until the next confirm
    replaces the store, so age never unpins it), or ended within
    PIN_WINDOW_S.

    The end is ``finished_at``; ``started_at`` stands in when the end is
    not known. run-store only ever knows when its run started, so for that
    side ``ceiling_s`` (the longest a run may live before the watchdog
    kills it) widens the start-based window: a run that started 70 minutes
    ago and finished 10 minutes ago is still recent, and only the row read
    that follows can tell. With the end known the window is exact.
    """
    if pinned or active:
        return True
    if run_id and run_id in (active_ids or []):
        return True
    if finished_at is not None:
        return within_window(finished_at, now)
    return within_window(started_at, now, PIN_WINDOW_S + max(0, ceiling_s or 0))


def pin_target(run_store, active_ids, now=None, ceiling_s=0) -> str | None:
    """The run id the panel should read this tick, or None for the log.

    The session's own run (run-store) wins while ``pinnable`` says so from
    what the store knows: the feed's active list, the pinned flag, the
    stamps it carries. Otherwise the newest run in flight. Decided from
    the feed and the store alone so the callback reads one row, never two.
    """
    now = now or now_utc()
    store = run_store or {}
    own = store.get("run_id")
    if own and pinnable(own, active_ids, now, pinned=bool(store.get("pinned")),
                        finished_at=store.get("finished"),
                        started_at=store.get("started"), ceiling_s=ceiling_s):
        return own
    return active_ids[-1] if active_ids else None


def pin_holds(run, active_ids, now=None, run_store=None) -> bool:
    """Whether the row read for the pin is still worth showing: the same
    rule as pin_target, now with the row's own status and end stamp (and
    the store's pinned flag, when the store names this run)."""
    if not run:
        return False
    now = now or now_utc()
    store = run_store or {}
    pinned = bool(store.get("pinned")) and store.get("run_id") == run.get("run_id")
    return pinnable(run.get("run_id"), active_ids, now,
                    active=bool(run.get("active"))
                    or run.get("status") in ACTIVE_STATUSES,
                    pinned=pinned, finished_at=run.get("finished_at"),
                    started_at=run.get("started_at"))


def symbol_state(stages, stage, symbol) -> str:
    """One symbol's state in one stage.

    Recorded states win. Without one, a stage that tracks symbols and has
    finished never reached this symbol (skipped); a stage that does not
    track symbols (the portfolio report) lends its own terminal state;
    anything still open is pending.
    """
    entry = (stages or {}).get(stage) or {}
    tracked = entry.get("symbols") or {}
    if symbol in tracked:
        return tracked[symbol]
    state = entry.get("state") or "pending"
    if state in ("done", "failed", "skipped"):
        return "skipped" if tracked else state
    return "pending"


def symbol_errors(stages, symbol) -> list[tuple[str, str]]:
    """(stage, reason) for every stage that recorded an error for the
    symbol, in pipeline order."""
    out = []
    for stage in STAGES:
        errors = ((stages or {}).get(stage) or {}).get("errors") or {}
        reason = errors.get(symbol)
        if reason:
            out.append((stage, reason))
    return out


def failure_text(errors: list[tuple[str, str]]) -> str | None:
    if not errors:
        return None
    if len(errors) == 1:
        return f"failed: {errors[0][1]}"
    return "failed: " + "; ".join(f"{stage}: {reason}" for stage, reason in errors)


def run_fingerprint(run) -> str | None:
    """What the stepper is drawn from, hashed: the panel is rewritten only
    when this or the log moves."""
    if not run:
        return None
    facts = {
        "status": run.get("status"),
        "error": run.get("error"),
        "stages": run.get("stages") or {},
        "symbols": run.get("symbols") or [],
    }
    raw = json.dumps(facts, sort_keys=True, default=str).encode()
    return hashlib.sha1(raw).hexdigest()[:16]


def _glyph(state: str, title: str) -> html.I:
    cls = _GLYPHS.get(state, _GLYPHS["pending"])
    return html.I(className=f"{cls} progress-glyph progress-glyph-{state}",
                  title=title)


def _stage_cell(stages, stage) -> html.Div:
    entry = (stages or {}).get(stage) or {}
    state = entry.get("state") or "pending"
    done, total = entry.get("done"), entry.get("total")
    title = f"{stage}: {state}"
    if entry.get("error"):
        title += f": {entry['error']}"
    children = [_glyph(state, title),
                html.Span(stage, className="progress-stage-name")]
    if total is not None:
        children.append(html.Span(f"{0 if done is None else done}/{total}",
                                  className="progress-stage-count"))
    return html.Div(children, className=f"progress-stage progress-stage-{state}",
                    title=title)


def _symbol_row(stages, symbol) -> html.Div:
    glyphs = [_glyph(symbol_state(stages, stage, symbol),
                     f"{symbol} {stage}: {symbol_state(stages, stage, symbol)}")
              for stage in STAGES]
    children = [
        html.Span(symbol, className="progress-symbol-name"),
        html.Span(glyphs, className="progress-symbol-glyphs"),
    ]
    failure = failure_text(symbol_errors(stages, symbol))
    if failure:
        children.append(html.Span(failure, className="progress-symbol-error",
                                  title=failure))
    return html.Div(children, className="progress-symbol-row")


def _caption(run) -> html.Div:
    status = run.get("status") or "queued"
    parts = [f"{run.get('kind') or 'manual'} run", status]
    error = run.get("error")
    if status in ("failed", "cancelled") and error:
        parts.append(str(error).strip().splitlines()[0][:160])
    return html.Div(" · ".join(parts),
                    className=f"progress-run-caption progress-run-{status}")


def stepper(run) -> html.Div:
    """Stage cells, then a line per symbol, from one run dict."""
    stages = run.get("stages") or {}
    symbols = list(run.get("symbols") or [])
    return html.Div(
        [
            _caption(run),
            html.Div([_stage_cell(stages, stage) for stage in STAGES],
                     className="progress-stages"),
            html.Div([_symbol_row(stages, sym) for sym in symbols],
                     className="progress-symbols"),
        ],
        className="progress-stepper",
    )


def feed_lines(events, window: int = 45) -> html.Div:
    """The free-text log, newest ``window`` lines, as it has always been
    drawn. The wrapper is the scroll container assets/feed_scroll_anchor.js
    anchors on."""
    from services import progress_service as prog

    rows = []
    for e in events[-window:]:
        stage = e.get("stage", "")
        icon = STAGE_ICONS.get(stage, "bi-dot")
        cls = "progress-line-error" if stage == "error" else (
            "progress-line-done" if stage == "done" else "")
        # Rendered from the event's epoch, not its pre-formatted string: rows
        # written by an older process carried a naive local clock while rows
        # restored from Postgres carried UTC, so the same panel showed both.
        rows.append(html.Div(
            [
                html.Span(prog.event_clock(e), className="progress-ts",
                          title=f"{prog.DISPLAY_TZ_LABEL} "
                                f"({prog.DISPLAY_TZ.key})"),
                html.I(className=f"bi {icon} progress-icon progress-icon-{stage}"),
                html.Span(e.get("message", ""), className="progress-msg"),
            ],
            className=f"progress-line {cls}",
        ))
    # No `key` props: dash-renderer keys child wrappers by tree path, not by
    # the component's key, so re-rendered rows are recreated wholesale
    # whatever we stamp on them. assets/feed_scroll_anchor.js preserves the
    # reading position across those rewrites instead.
    return html.Div(rows, className="progress-feed-lines")


def body(run, events) -> list:
    """The panel's children: stepper over a folded log for a run, the bare
    log otherwise."""
    lines = feed_lines(events)
    if run is None:
        return [lines]
    return [
        stepper(run),
        html.Details(
            [html.Summary("Details", className="progress-details-summary"), lines],
            className="progress-details",
        ),
    ]


def header_icon(active: bool, run=None):
    """Spinner while the feed is live; a cross for a run that ended badly;
    the check otherwise."""
    if active:
        return html.Div(className="progress-spinner")
    if (run or {}).get("status") in ("failed", "cancelled"):
        return html.I(className="bi bi-x-circle-fill progress-header-failed")
    return html.I(className="bi bi-check-circle-fill progress-header-done")

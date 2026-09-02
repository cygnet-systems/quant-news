"""The topbar run pill: one run, one line, on every route.

The pill answers "where is my run" without the activity panel being open.
What it shows is decided here from plain run dicts (run_service's shape)
so the poll callback in app.py stays a thin read-and-compare, and the
precedence and text patterns can be pinned by tests without a browser.

Precedence, first match wins:
  1. the viewer's own manual run in flight  -> Running (cancellable)
  2. the viewer's newest finished manual run -> Ready / Failed, with a View
     link to its run page, until that page is visited (run-seen-store)
  3. any scheduled run in flight             -> Scheduled (calendar glyph)
  4. nothing                                 -> hidden

The viewer's own result outranks the daily job's progress: the scheduled
run takes tens of minutes and a Ready that waited for it would arrive
long after the report did. Once the run page has been visited the
scheduled run shows again. The completion toast is decided from the
finished run alone (finished_view), never from what the pill displays.

A cancelled run is never shown: the user asked for it to go away.
"""

from datetime import datetime, timezone

from dash import dcc, html

from services.run_service import STAGES

# Symbols named on the pill before the rest collapse into "+n".
MAX_SYMBOLS = 3
# Below this many seconds left the ETA reads "finishing up": a minute
# count that small is noise, and a run past its estimate is still ending.
FINISHING_S = 30

_STATE_CLASS = {
    "running": "run-pill run-pill-running",
    "scheduled": "run-pill run-pill-scheduled",
    "ready": "run-pill run-pill-ready",
    "failed": "run-pill run-pill-failed",
}


def symbols_label(symbols) -> str:
    """'NVDA, AMD, TSLA +2': the first few names, then a count."""
    names = [s for s in (symbols or []) if s]
    head = ", ".join(names[:MAX_SYMBOLS])
    rest = len(names) - MAX_SYMBOLS
    return f"{head} +{rest}" if rest > 0 else head


def _parse_ts(value) -> datetime | None:
    if not value:
        return None
    try:
        ts = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    # SQLite hands back naive stamps; the row is written in UTC.
    return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)


def eta_label(estimate_s, started_at, now=None) -> str | None:
    """'~3 min', 'finishing up' under FINISHING_S, None without an estimate.

    Remaining = estimate minus elapsed, floored at zero: a run past its
    estimate says "finishing up" rather than counting into the negative.
    """
    if estimate_s is None:
        return None
    started = _parse_ts(started_at)
    now = now or datetime.now(timezone.utc)
    elapsed = (now - started).total_seconds() if started else 0.0
    remaining = max(0.0, float(estimate_s) - elapsed)
    if remaining < FINISHING_S:
        return "finishing up"
    return f"~{max(1, int(round(remaining / 60.0)))} min"


def current_stage(stages) -> tuple | None:
    """(name, done, total) of the stage in flight, or None.

    "Newest" is by pipeline order, not dict order: the row's JSONB does not
    keep insertion order, and in a full run the report and the models run
    side by side, so the later stage is the one worth naming.
    """
    stages = stages or {}
    running = [name for name in STAGES
               if (stages.get(name) or {}).get("state") == "running"]
    if not running:
        return None
    name = running[-1]
    entry = stages.get(name) or {}
    return name, entry.get("done"), entry.get("total")


def _stage_part(stage) -> str | None:
    if stage is None:
        return None
    name, done, total = stage
    if done is None or total is None:
        return name
    return f"{name} {done}/{total}"


def is_own(run: dict, owner_uid, run_store) -> bool:
    """A manual run of this viewer's: same owner (anonymous compares as
    ''), or the run this browser session confirmed (run-store)."""
    if (run or {}).get("kind") != "manual":
        return False
    if (run.get("owner_uid") or "") == (owner_uid or ""):
        return True
    return bool(run.get("run_id")) and \
        run.get("run_id") == (run_store or {}).get("run_id")


def _pin(run: dict) -> dict:
    """The run-store dict the panel pins to when the pill is clicked."""
    config = run.get("config") or {}
    return {
        "run_id": run["run_id"],
        "started": run.get("started_at"),
        "scope": config.get("scope") or "full",
        "preset": run.get("preset"),
        "symbols": list(run.get("symbols") or []),
        "owner_uid": run.get("owner_uid"),
        "kind": run.get("kind"),
    }


def _view(run: dict, state: str, parts: list, *, href=None, cancel=False,
          stage=None, eta=None, title=None) -> dict:
    text = " · ".join(p for p in parts if p)
    name, done, total = stage if stage else (None, None, None)
    return {
        "run_id": run["run_id"],
        "kind": run.get("kind"),
        "status": run.get("status"),
        "state": state,
        "className": _STATE_CLASS[state],
        # The span's text; ``label`` adds the link's word for the full line.
        "text": text,
        "label": f"{text} · View" if href else text,
        "href": href,
        "cancel": cancel,
        "title": title,
        # What must change for the pill to be rewritten. The ETA string
        # stands in for a minute bucket: it moves exactly when the shown
        # minute does (and when it drops to "finishing up").
        "fp": [run["run_id"], run.get("status"), name, done, total, eta],
        "pin": _pin(run),
    }


def run_href(run_id: str) -> str:
    return f"/runs/{run_id}?open=first"


def _active_view(run: dict, owner_uid, run_store, now) -> dict:
    stage = current_stage(run.get("stages"))
    eta = eta_label(run.get("estimate_s"), run.get("started_at"), now)
    if run.get("kind") == "scheduled":
        # The scheduled run is the daily job's; the label names the job,
        # not the viewer, and it cannot be cancelled from here.
        return _view(run, "scheduled",
                     ["Scheduled", "daily analysis", _stage_part(stage)],
                     stage=stage, eta=eta)
    word = "Starting" if run.get("status") == "queued" else "Running"
    return _view(run, "running",
                 [word, symbols_label(run.get("symbols")), _stage_part(stage),
                  eta],
                 cancel=is_own(run, owner_uid, run_store), stage=stage, eta=eta)


def _terminal_view(run: dict) -> dict | None:
    status = run.get("status")
    href = run_href(run["run_id"])
    if status == "done":
        return _view(run, "ready", ["Ready", symbols_label(run.get("symbols"))],
                     href=href)
    if status == "failed":
        first_line = (str(run.get("error") or "").strip().splitlines()
                      or ["unknown error"])[0]
        return _view(run, "failed", ["Failed", first_line], href=href,
                     title=first_line)
    return None


def finished_view(latest, owner_uid, run_store=None, seen=None) -> dict | None:
    """The Ready/Failed view of the viewer's newest manual run, or None
    when it is not theirs, still open, already seen or cancelled.

    Kept apart from pill_view so the completion toast is decided by the
    finished run alone: whatever the pill chooses to display (a scheduled
    run, say), a run that just ended is still announced.
    """
    if not latest or latest.get("active"):
        return None
    if not is_own(latest, owner_uid, run_store):
        return None
    if latest.get("run_id") == (seen or {}).get("run_id"):
        return None
    return _terminal_view(latest)


def pill_view(active, latest, owner_uid, run_store=None, seen=None,
              now=None) -> dict | None:
    """What the pill shows, or None for hidden.

    ``active`` is run_service.active_runs() (oldest first), ``latest`` the
    viewer's newest manual run of any status (or None), ``seen`` the
    run-seen-store dict naming the last run page visited.
    """
    now = now or datetime.now(timezone.utc)
    own = [r for r in (active or []) if is_own(r, owner_uid, run_store)]
    if own:
        return _active_view(own[-1], owner_uid, run_store, now)
    if latest and latest.get("active") and is_own(latest, owner_uid, run_store):
        # Not in the active list this tick but still open on the row: a
        # row created between the two reads. Treat it as in flight.
        return _active_view(latest, owner_uid, run_store, now)
    finished = finished_view(latest, owner_uid, run_store, seen)
    if finished:
        return finished
    scheduled = [r for r in (active or []) if r.get("kind") == "scheduled"]
    if scheduled:
        return _active_view(scheduled[0], owner_uid, run_store, now)
    return None


def done_toast(view, notified) -> dict | None:
    """The completion toast for a Ready/Failed view (finished_view), or None.

    ``notified`` is the run-notified-store dict naming the run last
    announced; the same run is never announced twice, whatever the
    fingerprint does. Done: "Report ready" over the symbols; failed: "Run
    failed" over the error's first line. Both link to the run page.
    """
    if not view or view.get("state") not in ("ready", "failed"):
        return None
    if view["run_id"] == (notified or {}).get("run_id"):
        return None
    failed = view["state"] == "failed"
    text = (view.get("title") or "unknown error") if failed \
        else symbols_label(view["pin"].get("symbols"))
    return {
        "run_id": view["run_id"],
        "header": "Run failed" if failed else "Report ready",
        "icon": "danger" if failed else "success",
        "body": [
            html.Span(text, className="run-done-text"),
            dcc.Link("View", href=view["href"], className="run-done-link"),
        ],
    }


def pill_children(view: dict) -> list:
    """The pill's inner components for one view: dot or calendar glyph,
    the text span, the View link when the run has a page, the cancel
    button only for the viewer's own run in flight."""
    if view["state"] == "scheduled":
        lead = html.I(className="bi bi-calendar3 run-pill-glyph")
    else:
        lead = html.Span(className="run-pill-dot")
    children = [
        lead,
        html.Span(view["text"], id="run-pill-text", className="run-pill-text",
                  title=view.get("title")),
    ]
    if view.get("href"):
        children.append(dcc.Link("View", id="run-pill-link", href=view["href"],
                                 className="run-pill-link"))
    if view.get("cancel"):
        children.append(html.Button(
            html.I(className="bi bi-x-lg"), id="run-pill-cancel",
            className="run-pill-cancel", title="Cancel run",
            **{"aria-label": "Cancel run"}))
    return children

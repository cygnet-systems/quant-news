"""Cross-process progress feed for long-running pipelines.

Full Analysis spans a browser callback, a background subprocess (models),
and a synthesis callback, the user previously saw nothing until the end.
Stages emit events here; a dcc.Interval-driven panel streams them live and
late viewers catch up on the full feed.

diskcache is file-based, so the main process and the prediction subprocess
share the same feed without any wiring.
"""

import logging
import os
import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import diskcache

logger = logging.getLogger(__name__)

# One display zone for every activity timestamp. The feed used to mix two:
# emit() formatted naive local wall-clock in whichever process was running,
# while hydrate_from_db()/get_activity_runs() formatted the tz-aware UTC value
# straight out of Postgres, so restored rows rendered ~4 hours off next to
# live ones. Everything is stored UTC-aware and rendered here, in the market
# zone this app already uses everywhere else (news_window, cache_service,
# scheduler job defaults all spell it "US/Eastern").
DISPLAY_TZ = ZoneInfo("US/Eastern")
DISPLAY_TZ_LABEL = "ET"

_CACHE_DIR = "cache/progress"
_EVENTS_KEY = "events"
_ACTIVE_KEY = "active"
_RUN_ID_KEY = "run_id"
_RUN_TITLE_KEY = "run_title"
# Watchdog state: the OS pid executing the run's heavy stage (recorded by the
# prediction subprocess, cleared when its results reach the server), when that
# pid was first seen dead, and which run was already aborted (once-only guard).
_RUN_PID_KEY = "run_pid"
_PID_DEAD_SINCE_KEY = "run_pid_dead_since"
_ABORTED_RUN_KEY = "run_aborted"

# A recorded worker pid must stay dead this long before the run is declared
# aborted: on a healthy run there is a window of a few seconds between the
# subprocess exiting and persist_predictions (server-side) clearing the pid.
WATCHDOG_PID_GRACE_S = 30
# Fallback for runs with no recorded worker (report-only runs execute in the
# server process): an active feed with no event this long is a stall.
WATCHDOG_STALL_S = 15 * 60
# In-memory window the panel renders. The full history lives in the
# activity_log table; this is just how much we keep hot for polling.
_MAX_EVENTS = 500

# The feed is a single shared channel, so any process that writes to it lands
# in the UI panel. Headless tools (benchmark.py, fetch scripts) share the same
# model code, and their events were interleaving with the user's run -- worse,
# their start_run() cleared the browser's feed mid-pipeline. app.py sets this
# flag for itself and its prediction subprocess; nothing else emits.
_ENV_FLAG = "QUANTNEWS_UI_PROGRESS"

_cache = None


def enable() -> None:
    """Mark this process (and any it spawns) as a progress publisher."""
    os.environ[_ENV_FLAG] = "1"


def _enabled() -> bool:
    return os.environ.get(_ENV_FLAG) == "1"


def _get_cache() -> diskcache.Cache:
    global _cache
    if _cache is None:
        _cache = diskcache.Cache(_CACHE_DIR)
    return _cache


def _user_id() -> str:
    """Identity for audit rows: the signed-in Cygnet uid when this code runs
    inside a request (middleware sets the ContextVar), else the local/env
    fallback (background subprocess, benchmark, scripts)."""
    try:
        from services.auth_service import current_uid
        uid = current_uid()
        if uid:
            return uid
    except Exception:
        pass
    return os.environ.get("QUANTNEWS_USER") or os.environ.get("USER") or "local"


def viewer_is_admin() -> bool:
    """Whether the requesting user may read other people's activity.

    Role lives in the shared Cygnet users table; anonymous requests are never
    admin, so an unauthenticated deployment sees only its own fallback bucket.
    """
    try:
        from services.auth_service import current_user
        u = current_user()
        return bool(u and u.is_admin)
    except Exception:
        return False


# Run identity for processes that are NOT progress publishers (seeds,
# benchmarks, scripts). They must not read the diskcache run keys: that file
# is shared with a running UI on the same host, so they would stamp their
# rows with the browser's run_id and title.
_local_run: dict[str, str | None] = {"id": None, "title": None}


def _run_identity() -> tuple[str, str | None]:
    if _enabled():
        c = _get_cache()
        return c.get(_RUN_ID_KEY) or "adhoc", c.get(_RUN_TITLE_KEY)
    return _local_run["id"] or "adhoc", _local_run["title"]


def current_run_id() -> str:
    """The run id events and spend are grouped under ("adhoc" outside a run)."""
    return _run_identity()[0]


def _write_audit(stage: str, message: str, payload: dict | None = None,
                 run_id: str | None = None, run_title: str | None = None) -> None:
    """Append to the durable audit trail. Best-effort: the DB being down must
    never break a pipeline, and the diskcache feed still drives the panel.

    ``run_id`` overrides the ambient run for events that belong to something
    other than the run this process is publishing (a scheduled job's
    start/finish lines written from the server process)."""
    try:
        import json

        from db.models import ActivityLog
        from db.session import get_session

        # JSON-clean the payload up front (dates, numpy scalars): losing a
        # payload to a stray type must not lose the whole audit row.
        if payload is not None:
            try:
                payload = json.loads(json.dumps(payload, default=str))
            except (TypeError, ValueError):
                payload = {"unserializable": str(payload)[:500]}

        if run_id is None:
            run_id, run_title = _run_identity()
        with get_session() as session:
            session.add(ActivityLog(
                user_id=_user_id(),
                run_id=run_id,
                run_title=run_title,
                stage=stage,
                message=message[:4000],
                payload=payload,
            ))
    except Exception as e:
        logger.debug(f"activity_log write skipped: {e}")


def mark_run_pending() -> None:
    """Flip the feed active the moment a run is confirmed.

    Confirming a run and the stage that calls start_run() are separate
    callbacks (the model stage even lives in a spawned subprocess), so on a
    cold idle panel the poller's next tick would still see active=False and
    stay at the idle rate. The first events then took an idle interval to
    appear. Publisher-only, like the rest of the live feed.
    """
    try:
        if _enabled():
            _get_cache().set(_ACTIVE_KEY, True)
    except Exception as e:
        logger.debug(f"progress mark_run_pending failed: {e}")


def start_run(title: str) -> None:
    """Begin a new run.

    Does NOT clear prior events -- the panel is an audit log, so runs
    accumulate behind a boundary marker and the rolling cap trims the oldest.

    Non-publisher processes still open a run: they keep its identity in
    process memory so their events group into a proper run block in the audit
    trail without touching the UI's live feed.
    """
    try:
        import uuid

        run_id = str(uuid.uuid4())
        if _enabled():
            c = _get_cache()
            c.set(_RUN_ID_KEY, run_id)
            c.set(_RUN_TITLE_KEY, title)
            c.set(_ACTIVE_KEY, True)
            # A previous run's worker pid must never be read as this run's.
            c.set(_RUN_PID_KEY, None)
            c.set(_PID_DEAD_SINCE_KEY, None)
        else:
            _local_run["id"], _local_run["title"] = run_id, title
        emit("run", title)
    except Exception as e:
        logger.debug(f"progress start_run failed: {e}")


def record_run_pid() -> None:
    """Publisher-only: remember which OS process executes the run's heavy
    stage. The prediction subprocess calls this as its first act; the server
    clears it (clear_run_pid) once that stage's results have arrived, so a
    recorded-but-dead pid means the run lost its worker."""
    try:
        if _enabled():
            c = _get_cache()
            c.set(_RUN_PID_KEY, os.getpid())
            c.set(_PID_DEAD_SINCE_KEY, None)
    except Exception as e:
        logger.debug(f"progress record_run_pid failed: {e}")


def clear_run_pid() -> None:
    """The recorded worker's output has been received (or the run is over).
    its pid no longer stands for the run's health."""
    try:
        if _enabled():
            c = _get_cache()
            c.set(_RUN_PID_KEY, None)
            c.set(_PID_DEAD_SINCE_KEY, None)
    except Exception as e:
        logger.debug(f"progress clear_run_pid failed: {e}")


def _pid_alive(pid: int) -> bool:
    """True while the process exists and is not a zombie (a died spawn child
    can linger as a zombie until its parent reaps it, that is dead)."""
    try:
        import psutil
        p = psutil.Process(int(pid))
        return p.is_running() and p.status() != psutil.STATUS_ZOMBIE
    except Exception:
        return False


def rss_mb() -> float | None:
    """This process's resident set in MB, None when psutil cannot say."""
    try:
        import psutil
        return round(psutil.Process().memory_info().rss / 1e6, 1)
    except Exception:
        return None


def emit_memory(stage: str, **extra) -> None:
    """One ``memory`` event: RSS at a stage boundary.

    Three container deaths on 2026-09-02 left nothing behind but the phase
    they died in; the footprint had to be inferred from imports and
    weights. This is the instrument the next incident reads instead.
    """
    mb = rss_mb()
    if mb is None:
        return
    emit("action", f"Memory: {mb:,.0f} MB RSS after {stage}",
         payload={"event": "memory", "stage": stage, "rss_mb": mb, **extra})


def watchdog_check() -> bool:
    """Close a run whose worker died (or that stopped emitting) as a failure.

    Called from the server's progress poll. Two detections, exactly once per
    run (the aborted-run mark guards re-emission):

    * A recorded worker pid that has stayed dead past WATCHDOG_PID_GRACE_S.
      the grace covers the healthy seconds between the subprocess exiting
      and persist_predictions clearing the pid.
    * No recorded pid (report-only runs execute in the server process): no
      feed event for WATCHDOG_STALL_S while the run claims to be active.

    Returns True when it aborted a run on this call.
    """
    try:
        if not _enabled():
            return False
        c = _get_cache()
        if not c.get(_ACTIVE_KEY):
            return False
        run_id = c.get(_RUN_ID_KEY)
        if not run_id or c.get(_ABORTED_RUN_KEY) == run_id:
            return False

        now = time.time()
        pid = c.get(_RUN_PID_KEY)
        if pid:
            if _pid_alive(pid):
                if c.get(_PID_DEAD_SINCE_KEY):
                    c.set(_PID_DEAD_SINCE_KEY, None)
                return False
            dead_since = c.get(_PID_DEAD_SINCE_KEY)
            if not dead_since:
                c.set(_PID_DEAD_SINCE_KEY, now)
                return False
            if now - dead_since < WATCHDOG_PID_GRACE_S:
                return False
            message = ("Run aborted: the prediction process died "
                       "unexpectedly (no results were stored)")
            finish = "Run failed: prediction process died"
        else:
            events = c.get(_EVENTS_KEY) or []
            if not events:
                return False
            # The run must actually be OPEN in the feed. active=True over a
            # feed whose newest boundary is a completion is mark_run_pending
            # arming the panel for a run whose start_run hasn't landed yet. 
            # aborting there would stamp a spurious failure on the PREVIOUS
            # run during the subprocess spawn window.
            for e in reversed(events):
                if e.get("stage") == "done":
                    return False
                if e.get("stage") == "run":
                    break
            last_t = events[-1].get("t")
            if not last_t or now - last_t < WATCHDOG_STALL_S:
                return False
            message = (f"Run aborted: no activity for "
                       f"{int(WATCHDOG_STALL_S // 60)} minutes (stalled)")
            finish = "Run failed: stalled with no activity"

        # Mark BEFORE emitting: the abort's own events must not retrigger it.
        c.set(_ABORTED_RUN_KEY, run_id)
        # The abort otherwise lives only in the feed and activity_log; a
        # post-mortem from the server log alone would show nothing at all.
        logger.warning(f"Watchdog aborted run {run_id}: {message}")
        emit("error", message)
        finish_run(finish)
        return True
    except Exception as e:
        logger.debug(f"progress watchdog check failed: {e}")
        return False


def get_activity_runs(limit_runs: int = 50, scope: str = "all",
                      stages: list[str] | None = None,
                      symbol: str | None = None,
                      since_days: int | None = None) -> list[dict]:
    """Past runs from the audit trail, newest first, each with its events.

    Visibility follows the shared Cygnet role: an Administrator sees every
    user's runs and may narrow to their own with ``scope="self"``; everyone
    else is pinned to their own rows regardless of what scope is requested.

    Two queries rather than one grouped fetch: pick the most recent runs by
    their latest event, then pull only those runs' events. A single ORDER BY
    over the whole table would need an arbitrary row cap, which truncates runs
    mid-way and makes counts wrong.

    Runs are keyed by (user_id, run_id), not run_id alone: ad-hoc events share
    the literal ids "adhoc" and "auth", so grouping on run_id would fuse
    different people's unrelated events into one block in the admin view.

    The optional filters narrow both queries: a run qualifies if it has any
    matching event, and only its matching events are returned. ``symbol``
    matches on a word boundary so filtering for SPY does not also surface
    every SPYG line.
    """
    try:
        from datetime import timedelta

        from sqlalchemy import func, or_, select, tuple_

        from db.models import ActivityLog
        from db.session import get_session

        show_all = scope == "all" and viewer_is_admin()

        conds = []
        if not show_all:
            conds.append(ActivityLog.user_id == _user_id())
        if stages:
            conds.append(ActivityLog.stage.in_(list(stages)))
        if since_days:
            conds.append(ActivityLog.created_at
                         >= datetime.now(timezone.utc) - timedelta(days=since_days))
        if symbol and symbol.strip():
            sym = symbol.strip().upper()
            conds.append(or_(
                ActivityLog.message.op("~*")(rf"\y{sym}\y"),
                ActivityLog.run_title.op("~*")(rf"\y{sym}\y"),
            ))

        with get_session() as session:
            recent_q = select(
                ActivityLog.user_id,
                ActivityLog.run_id,
                func.max(ActivityLog.created_at).label("last_at"),
            )
            for c in conds:
                recent_q = recent_q.where(c)
            recent = session.execute(
                recent_q
                .group_by(ActivityLog.user_id, ActivityLog.run_id)
                .order_by(func.max(ActivityLog.created_at).desc())
                .limit(limit_runs)
            ).all()
            keys = [(r.user_id, r.run_id) for r in recent]
            if not keys:
                return []

            events_q = select(ActivityLog).where(
                tuple_(ActivityLog.user_id, ActivityLog.run_id).in_(keys))
            # Reapply the event-level filters so an expanded run shows the
            # lines that matched, not the whole run.
            for c in conds:
                events_q = events_q.where(c)
            rows = session.execute(
                events_q.order_by(ActivityLog.created_at.asc(),
                                  ActivityLog.id.asc())
            ).scalars().all()

        by_run: dict[tuple[str, str], dict] = {}
        for r in rows:
            # started/ended are handed to the page already in DISPLAY_TZ, so a
            # renderer calling .strftime() on them cannot reintroduce the UTC
            # skew this feed used to show.
            stamp = to_display_tz(r.created_at)
            run = by_run.setdefault((r.user_id, r.run_id), {
                "run_id": r.run_id,
                "user_id": r.user_id,
                "title": r.run_title or "Activity",
                "started": stamp,
                "ended": stamp,
                "events": [],
                "errors": 0,
                "gaps": 0,
            })
            run["ended"] = stamp
            run["errors"] += 1 if r.stage == "error" else 0
            # A report written without expected evidence: not a crash, but
            # a run that carries one is not whole either.
            run["gaps"] += 1 if r.stage == "gap" else 0
            run["events"].append({
                "ts": format_clock(r.created_at),
                "t": stamp.timestamp(),
                "stage": r.stage,
                "message": r.message,
            })

        ordered = [by_run[k] for k in keys if k in by_run]
        for run in ordered:
            run["duration_s"] = max(
                0.0, (run["ended"] - run["started"]).total_seconds()
            )
        return ordered
    except Exception as e:
        logger.debug(f"activity_log run query skipped: {e}")
        return []


def to_display_tz(value: "datetime | float | str | None") -> "datetime | None":
    """Normalize an epoch, ISO string or datetime to a DISPLAY_TZ datetime.

    A naive value is *assumed UTC*. The same rule services/news_window.py
    documents, and correct for every naive value this app persists (Postgres
    timestamptz round-trips aware; the few naive ones came from
    ``datetime.utcnow()``-shaped code). Strings are accepted because several
    cache accessors hand timestamps to the UI already str()-ed.
    """
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=timezone.utc).astimezone(DISPLAY_TZ)
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError:
            return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(DISPLAY_TZ)


def format_stamp(value: "datetime | float | str | None",
                 with_seconds: bool = False) -> str:
    """"YYYY-MM-DD HH:MM ET": for absolute timestamps shown to a reader.

    Raw UTC was leaking into two surfaces with no marker at all, and late
    evening ET rendered under tomorrow's DATE. Microseconds are always dropped:
    nothing a reader does with a stamp needs them.
    """
    dt = to_display_tz(value)
    if dt is None:
        return ""
    fmt = "%Y-%m-%d %H:%M:%S" if with_seconds else "%Y-%m-%d %H:%M"
    return f"{dt.strftime(fmt)} {DISPLAY_TZ_LABEL}"


def format_clock(value: "datetime | float | None") -> str:
    """HH:MM:SS in DISPLAY_TZ, or "". The feed's one timestamp formatter."""
    dt = to_display_tz(value)
    return dt.strftime("%H:%M:%S") if dt else ""


def event_clock(event: dict) -> str:
    """Render one feed event's time.

    Prefers the epoch ``t`` every writer stores, so events already sitting in
    the diskcache feed from an older process. Whose pre-formatted ``ts``
    string may be in either zone. Still render in DISPLAY_TZ. Falls back to
    the stored string only when there is no epoch to work from.
    """
    t = event.get("t")
    if isinstance(t, (int, float)):
        return format_clock(t)
    return str(event.get("ts") or "")


def hydrate_from_db(limit: int = _MAX_EVENTS) -> int:
    """Refill the in-memory feed from the audit trail on a cold start.

    Without this a fresh session shows an empty panel even though the history
    exists in Postgres. Returns the number of events restored.

    Always self-scoped, even for admins: this refills the live diskcache feed,
    which is ONE shared cross-process channel that every viewer polls. Pouring
    other users' rows into it would put them on everyone's live panel, which
    is not what the Activity Log's admin view means.
    """
    try:
        from sqlalchemy import select

        from db.models import ActivityLog
        from db.session import get_session

        with get_session() as session:
            rows = session.execute(
                select(ActivityLog)
                .where(ActivityLog.user_id == _user_id())
                .order_by(ActivityLog.created_at.desc())
                .limit(limit)
            ).scalars().all()

        events = [{
            "ts": format_clock(r.created_at),
            "t": to_display_tz(r.created_at).timestamp(),
            "stage": r.stage,
            "message": r.message,
        } for r in reversed(rows)]

        c = _get_cache()
        c.set(_EVENTS_KEY, events)
        c.set(_ACTIVE_KEY, False)
        return len(events)
    except Exception as e:
        logger.debug(f"activity_log hydrate skipped: {e}")
        return 0


def emit(stage: str, message: str, payload: dict | None = None, *,
         feed: bool = True, run_id: str | None = None,
         run_title: str | None = None) -> None:
    """Append an event. Never raises: progress must not break the pipeline.

    ``feed=False`` writes the audit trail only. The scheduler's own status
    lines are emitted from the server process, where the live feed belongs
    to whatever run the browser has open. They used to land inside it and
    a "done" event from a scheduled job closed the user's in-flight run on
    the Trace page. Pass ``run_id`` to file them under the job run instead.

    The two sinks are gated separately. The diskcache feed is the live UI
    panel and stays publisher-only, so headless tools never interleave with a
    browser run; the durable audit trail always records, so a seed or backfill
    run outside the web process still shows up in the Activity Log.

    ``payload`` is optional structured data behind the message (counts,
    windows, hashes) for the Trace page. It goes to Postgres ONLY, the
    diskcache feed is read whole on every panel tick under a transact lock,
    so feed events must stay small; they carry at most a has_payload flag.
    """
    if feed and _enabled():
        try:
            c = _get_cache()
            # transact() serializes the read-modify-write across threads AND
            # processes (SQLite-backed), the bare get/append/set lost events
            # whenever model threads and the prediction subprocess emitted
            # concurrently.
            with c.transact():
                events = c.get(_EVENTS_KEY) or []
                now = datetime.now(timezone.utc)
                event = {
                    # Both fields are UTC-derived: "t" is the epoch every
                    # renderer prefers, "ts" is the same instant already
                    # rendered in DISPLAY_TZ for anything reading it raw.
                    "ts": format_clock(now),
                    "t": now.timestamp(),
                    "stage": stage,
                    "message": message,
                }
                if payload is not None:
                    event["has_payload"] = True
                events.append(event)
                c.set(_EVENTS_KEY, events[-_MAX_EVENTS:])
        except Exception as e:
            logger.debug(f"progress emit failed: {e}")

    _write_audit(stage, message, payload, run_id=run_id, run_title=run_title)


def finish_run(message: str = "Pipeline complete") -> None:
    """Mark the current run finished."""
    try:
        emit("done", message)
        if _enabled():
            c = _get_cache()
            c.set(_ACTIVE_KEY, False)
            # Terminal state: the worker pid (if any) no longer matters.
            c.set(_RUN_PID_KEY, None)
            c.set(_PID_DEAD_SINCE_KEY, None)
    except Exception as e:
        logger.debug(f"progress finish_run failed: {e}")


def get_feed() -> dict:
    """Return {active: bool, events: [...]} for the current run."""
    try:
        c = _get_cache()
        return {
            "active": bool(c.get(_ACTIVE_KEY)),
            "events": c.get(_EVENTS_KEY) or [],
        }
    except Exception:
        return {"active": False, "events": []}

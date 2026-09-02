"""Cross-process progress feed for long-running pipelines.

Full Analysis spans a browser callback, a background subprocess (models),
and a synthesis callback, the user previously saw nothing until the end.
Stages emit events here; a dcc.Interval-driven panel streams them live and
late viewers catch up on the full feed.

diskcache is file-based, so the main process and the prediction subprocess
share the same feed without any wiring.

The feed is kept per run. Every run has its own event list under
``events:<run_id>`` and the set of runs in flight is a list under
``active_runs``; the legacy ``events`` key stays as a rolling log across
runs (the panel's idle view, and what a cold start rehydrates). Two people
running at once no longer interleave or close each other's feed, as long
as writers name the run: ``emit(..., run_id=...)``. Writers that do not
name it fall back to this process's own run, then to the newest run in
flight (the prediction subprocess reports on the run the browser opened).

Run identity for persistence (``current_run_id``) is strictly this
process's own run, set by ``start_run``: it is never read back from the
shared cache, because that cache is shared by every process on the host
and would stamp one user's rows with another user's run. Anything that
stores predictions, reports or spend for a specific run must be handed
the run_id explicitly.

"Own run" is task-scoped where it can be: the server process hosts every
user's report stage at once, each in its own asyncio task, so the run
start_run() opened travels in a ContextVar (inherited by asyncio.to_thread
and any executor that submits through copy_context) before the
process-wide fallback. Research code several layers down emits unnamed
and stamps its LLM spend through current_run_id(); with two report runs
in flight those used to land in whichever run the server started last.
"""

import logging
import os
import time
from contextvars import ContextVar
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
# Rolling log across runs (and the events written outside any run).
_EVENTS_KEY = "events"
# Legacy boolean, kept in sync with the active-run list so readers that
# only know "is anything running" keep working. mark_run_pending() without
# a run id may also arm it ahead of the run's start_run.
_ACTIVE_KEY = "active"
_ACTIVE_RUNS_KEY = "active_runs"
# The most recently STARTED run, whichever process started it. Read only by
# get_feed() to pick what the idle panel shows; never used as identity.
_RUN_ID_KEY = "run_id"
# Ad-hoc bucket: events emitted outside any run (symbol search, exports).
ADHOC_RUN_ID = "adhoc"

# Per-run feeds live in the cache for a week and then expire; the durable
# copy is the activity_log table.
_PER_RUN_TTL_S = 7 * 86400

# A recorded worker pid must stay dead this long before the run is declared
# aborted: on a healthy run there is a window of a few seconds between the
# subprocess exiting and persist_predictions (server-side) clearing the pid.
WATCHDOG_PID_GRACE_S = 30
# Fallback for runs with no recorded worker (report-only runs execute in the
# server process): an active feed with no event this long is a stall.
WATCHDOG_STALL_S = 15 * 60
# The run rows are checked against the feed at most this often from the
# panel's poll: an analysis_runs row left queued/running by a process that
# died without closing it (container OOM mid-run, the scheduler's ceiling
# kill, a restart) is the per-owner lock, so it has to be failed by someone
# who is still alive. One small indexed query per interval, not per tick.
_DB_REAP_INTERVAL_S = 30
_last_db_reap = 0.0
# In-memory window the panel renders. The full history lives in the
# activity_log table; this is just how much we keep hot for polling.
_MAX_EVENTS = 500

# The feed is a shared channel, so any process that writes to it lands in
# the UI panel. Headless tools (benchmark.py, fetch scripts) share the same
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


def _events_key(run_id: str) -> str:
    return f"events:{run_id}"


def _meta_key(run_id: str) -> str:
    return f"run_meta:{run_id}"


def _pid_key(run_id: str) -> str:
    return f"run_pid:{run_id}"


def _dead_since_key(run_id: str) -> str:
    return f"run_pid_dead_since:{run_id}"


def _aborted_key(run_id: str) -> str:
    return f"run_aborted:{run_id}"


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


# This process's own run. Non-publisher processes (seeds, benchmarks,
# scripts) must not read the diskcache run keys: that file is shared with a
# running UI on the same host, so they would stamp their rows with the
# browser's run_id and title. Publishers keep it too: it is the only run
# identity current_run_id() will ever report.
_local_run: dict[str, str | None] = {"id": None, "title": None}
# The task-scoped copy of the same identity: set alongside _local_run by
# start_run/adopt_run, read first. Bare worker threads (a ThreadPoolExecutor
# that did not copy the context) see the default and fall back to the
# process-wide value, which is what they had before.
_run_ctx: ContextVar[dict | None] = ContextVar("qn_progress_run", default=None)


def _own_run() -> dict:
    """This task's run if one was opened in it, else the process's."""
    return _run_ctx.get() or _local_run


def _set_own_run(run_id: str | None, title: str | None) -> None:
    _local_run["id"], _local_run["title"] = run_id, title
    _run_ctx.set({"id": run_id, "title": title})


def _active_runs() -> list[str]:
    try:
        runs = _get_cache().get(_ACTIVE_RUNS_KEY)
    except Exception:
        return []
    return [r for r in (runs or []) if r]


def is_active(run_id: str | None) -> bool:
    """Whether the feed lists this run as in flight (a live process is
    reporting on it, or the confirm dispatcher just armed it)."""
    return bool(run_id) and run_id in _active_runs()


def _resolve_run_id(run_id: str | None = None) -> str:
    """Which run an unnamed feed write belongs to.

    Explicit wins. Then this process's own run while it is still in
    flight; then the newest run in flight (the model subprocess of a
    browser run has no run of its own yet, and a server callback closing a
    models-only run started by that subprocess has a stale local one);
    then the ad-hoc bucket. A finished run never receives unnamed events.
    Non-publishers keep their local run: they never read the shared cache.
    """
    if run_id:
        return run_id
    local = _own_run()["id"]
    if not _enabled():
        return local or ADHOC_RUN_ID
    active = _active_runs()
    if local and local in active:
        return local
    if active:
        return active[-1]
    return ADHOC_RUN_ID


def current_run_id() -> str:
    """This process's own run id, for grouping persisted rows and spend.

    Never a cross-process guess: only the id start_run() set in this
    process, and (for publishers) only while that run is still in flight;
    otherwise "adhoc". Anything writing rows for a run that lives in
    another process (the server persisting a models-only run's output)
    must be handed the run_id explicitly.
    """
    local = _own_run()["id"]
    if not local:
        return ADHOC_RUN_ID
    if _enabled() and local not in _active_runs():
        return ADHOC_RUN_ID
    return local


def _run_title(run_id: str) -> str | None:
    own = _own_run()
    if run_id == own["id"]:
        return own["title"]
    if run_id == ADHOC_RUN_ID or not _enabled():
        return None
    try:
        meta = _get_cache().get(_meta_key(run_id)) or {}
        return meta.get("title")
    except Exception:
        return None


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
            run_id = _resolve_run_id()
        if run_title is None:
            run_title = _run_title(run_id)
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


def _set_active(c: diskcache.Cache, runs: list[str]) -> None:
    """Write the active list and keep the legacy boolean in step with it."""
    c.set(_ACTIVE_RUNS_KEY, runs)
    c.set(_ACTIVE_KEY, bool(runs))


def _add_active(c: diskcache.Cache, run_id: str) -> None:
    runs = [r for r in (c.get(_ACTIVE_RUNS_KEY) or []) if r and r != run_id]
    runs.append(run_id)
    _set_active(c, runs)


def _remove_active(c: diskcache.Cache, run_id: str) -> None:
    runs = [r for r in (c.get(_ACTIVE_RUNS_KEY) or []) if r and r != run_id]
    _set_active(c, runs)


def mark_run_pending(run_id: str | None = None) -> None:
    """Flip the feed active the moment a run is confirmed.

    Confirming a run and the stage that calls start_run() are separate
    callbacks (the model stage even lives in a spawned subprocess), so on a
    cold idle panel the poller's next tick would still see active=False and
    stay at the idle rate. The first events then took an idle interval to
    appear. Publisher-only, like the rest of the live feed.

    With a ``run_id`` (the row the confirm dispatcher created) the run
    joins the active list right away; without one only the legacy boolean
    is armed, and start_run() brings the list in line.
    """
    try:
        if not _enabled():
            return
        c = _get_cache()
        if run_id:
            with c.transact():
                _add_active(c, run_id)
                c.set(_aborted_key(run_id), None)
        else:
            c.set(_ACTIVE_KEY, True)
    except Exception as e:
        logger.debug(f"progress mark_run_pending failed: {e}")


def start_run(title: str, run_id: str | None = None,
              owner_uid: str | None = None, kind: str = "manual") -> str | None:
    """Begin a run and return its id.

    ``run_id`` is the analysis_runs row the caller already created; without
    one an id is minted here so headless callers (benchmark, scripts) still
    get a proper run block in the audit trail. The run becomes this
    process's own run (see current_run_id) and, for publishers, joins the
    active list and gets its own feed.

    Does NOT clear prior events -- the rolling log is an audit log, so runs
    accumulate behind a boundary marker and the cap trims the oldest.
    """
    try:
        import uuid

        run_id = run_id or str(uuid.uuid4())
        _set_own_run(run_id, title)
        if _enabled():
            c = _get_cache()
            with c.transact():
                _add_active(c, run_id)
                c.set(_RUN_ID_KEY, run_id)
                c.set(_meta_key(run_id), {
                    "title": title, "kind": kind, "owner_uid": owner_uid,
                    "started": time.time(),
                }, expire=_PER_RUN_TTL_S)
                # A previous run's worker pid must never be read as this
                # run's, and an id reused after an abort starts clean.
                c.set(_pid_key(run_id), None)
                c.set(_dead_since_key(run_id), None)
                c.set(_aborted_key(run_id), None)
        emit("run", title, run_id=run_id,
             payload={"event": "run_start", "kind": kind, "owner_uid": owner_uid})
        return run_id
    except Exception as e:
        logger.debug(f"progress start_run failed: {e}")
        return run_id


def adopt_run(run_id: str, title: str | None = None) -> None:
    """Make an already-open run this process's own without re-opening it.

    The model subprocess of a browser run executes a stage of a run the
    server opened; its unnamed emits and its LLM usage/trace rows must
    group under that run, not "adhoc", and a second run_start line from
    start_run() would put a spurious boundary in the feed. Sets the local
    identity only; the active list and the run's meta are left as they are.
    """
    if not run_id:
        return
    _set_own_run(run_id, title or _run_title(run_id))


def terminate_run_worker(run_id: str, grace_s: float = 5.0) -> int | None:
    """Kill the recorded worker process of a run (a cancel), returning the
    pid it signalled or None when no worker was recorded or it is gone.

    SIGTERM first, SIGKILL if it is still there after ``grace_s``; the pid
    record is cleared either way so the watchdog does not later report the
    kill as a worker death. The caller records the cancellation on the run
    row and closes the feed; this only stops the process.
    """
    pid = run_pid(run_id)
    if not pid or pid == os.getpid():
        return None
    try:
        import psutil
        proc = psutil.Process(pid)
        if not _pid_alive(pid):
            return None
        proc.terminate()
        try:
            proc.wait(timeout=grace_s)
        except psutil.TimeoutExpired:
            proc.kill()
    except Exception as e:
        logger.warning(f"could not terminate worker {pid} of run {run_id}: {e}")
        return None
    finally:
        clear_run_pid(run_id)
    logger.info(f"Run {run_id[:8]} cancelled: worker pid {pid} terminated")
    return pid


def record_run_pid(run_id: str | None = None) -> None:
    """Publisher-only: remember which OS process executes the run's heavy
    stage. The prediction subprocess calls this as its first act; the server
    clears it (clear_run_pid) once that stage's results have arrived, so a
    recorded-but-dead pid means the run lost its worker."""
    try:
        if not _enabled():
            return
        run_id = _resolve_run_id(run_id)
        if run_id == ADHOC_RUN_ID:
            return
        c = _get_cache()
        c.set(_pid_key(run_id), os.getpid(), expire=_PER_RUN_TTL_S)
        c.set(_dead_since_key(run_id), None)
    except Exception as e:
        logger.debug(f"progress record_run_pid failed: {e}")


def clear_run_pid(run_id: str | None = None) -> None:
    """The recorded worker's output has been received (or the run is over).
    its pid no longer stands for the run's health."""
    try:
        if not _enabled():
            return
        run_id = _resolve_run_id(run_id)
        if run_id == ADHOC_RUN_ID:
            return
        c = _get_cache()
        c.set(_pid_key(run_id), None)
        c.set(_dead_since_key(run_id), None)
    except Exception as e:
        logger.debug(f"progress clear_run_pid failed: {e}")


def run_pid(run_id: str | None = None) -> int | None:
    """The recorded worker pid for a run, if any (what a cancel must kill)."""
    try:
        if not _enabled():
            return None
        run_id = _resolve_run_id(run_id)
        pid = _get_cache().get(_pid_key(run_id))
        return int(pid) if pid else None
    except Exception:
        return None


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


def _watchdog_one(c: diskcache.Cache, run_id: str, now: float) -> bool:
    """The two detections for one run; True when it aborted the run."""
    if c.get(_aborted_key(run_id)):
        return False
    pid = c.get(_pid_key(run_id))
    if pid:
        if _pid_alive(pid):
            if c.get(_dead_since_key(run_id)):
                c.set(_dead_since_key(run_id), None)
            return False
        dead_since = c.get(_dead_since_key(run_id))
        if not dead_since:
            c.set(_dead_since_key(run_id), now, expire=_PER_RUN_TTL_S)
            return False
        if now - dead_since < WATCHDOG_PID_GRACE_S:
            return False
        message = ("Run aborted: the prediction process died "
                   "unexpectedly (no results were stored)")
        finish = "Run failed: prediction process died"
    else:
        events = c.get(_events_key(run_id)) or []
        if not events:
            return False
        # A feed whose newest line is a completion is a run that already
        # closed; only its active-list entry is stale, and stamping a
        # failure on it would be wrong.
        if events[-1].get("stage") == "done":
            return False
        last_t = events[-1].get("t")
        if not last_t or now - last_t < WATCHDOG_STALL_S:
            return False
        message = (f"Run aborted: no activity for "
                   f"{int(WATCHDOG_STALL_S // 60)} minutes (stalled)")
        finish = "Run failed: stalled with no activity"

    # Mark BEFORE emitting: the abort's own events must not retrigger it.
    c.set(_aborted_key(run_id), True, expire=_PER_RUN_TTL_S)
    # The abort otherwise lives only in the feed and activity_log; a
    # post-mortem from the server log alone would show nothing at all.
    logger.warning(f"Watchdog aborted run {run_id}: {message}")
    emit("error", message, run_id=run_id)
    finish_run(finish, run_id=run_id)
    try:
        from services import run_service
        run_service.set_status(run_id, "failed", error=message)
    except Exception as e:
        logger.debug(f"watchdog could not fail run row {run_id}: {e}")
    return True


def _close_reaped(c: diskcache.Cache, run: dict) -> None:
    """The feed's side of a reaped run row: the failure line and the close,
    once, for a run the feed still listed (a scheduled run whose process the
    scheduler killed never reached finish_run)."""
    run_id = run["run_id"]
    if c.get(_aborted_key(run_id)):
        return
    c.set(_aborted_key(run_id), True, expire=_PER_RUN_TTL_S)
    message = f"Run aborted: {run.get('error') or 'no process reporting on it'}"
    logger.warning(f"Reaped run {run_id}: {message}")
    if run_id in _active_runs():
        emit("error", message, run_id=run_id)
        finish_run("Run failed: " + (run.get("error") or "abandoned"),
                   run_id=run_id)
    else:
        # Not in the feed: the audit trail alone records why it closed.
        emit("error", message, feed=False, run_id=run_id)


def close_reaped(run: dict) -> None:
    """Feed side of a run row somebody else already failed (the scheduler
    closing the rows linked to a finalized job run). Never raises."""
    try:
        _close_reaped(_get_cache(), run)
    except Exception as e:
        logger.debug(f"reaped run feed close failed: {e}")


def reap_orphans(max_age_s: float | None = -1, error: str | None = None) -> list[dict]:
    """Fail the analysis_runs rows nothing is working on any more and close
    their feeds. The feed's active list is the witness handed to
    run_service.reap_orphans; ``max_age_s`` and ``error`` pass through
    (0 at server start: nothing survives a restart, so every active row is
    an orphan). Never raises."""
    try:
        from services import run_service
        reaped = run_service.reap_orphans(live=_active_runs(),
                                          max_age_s=max_age_s, error=error)
    except Exception as e:
        logger.debug(f"run row reap skipped: {e}")
        return []
    try:
        c = _get_cache()
        for run in reaped:
            _close_reaped(c, run)
    except Exception as e:
        logger.debug(f"reaped run feed close failed: {e}")
    return reaped


def fail_orphan(run: dict) -> str | None:
    """Fail one active run if it is provably dead (run_service.orphan_reason
    with this feed's active list as the witness) and close its feed.
    Returns the reason, or None when the run may still be working."""
    try:
        from services import run_service
        reason = run_service.orphan_reason(run, live=_active_runs())
        if reason is None:
            return None
        closed = run_service.set_status(run["run_id"], "failed", error=reason)
        if closed is None or closed["status"] != "failed":
            return None
        _close_reaped(_get_cache(), closed)
        return reason
    except Exception as e:
        logger.debug(f"orphan check for {run.get('run_id')} failed: {e}")
        return None


def watchdog_check(run_id: str | None = None) -> bool:
    """Close a run whose worker died (or that stopped emitting) as a failure.

    Called from the server's progress poll. Two detections, exactly once per
    run (the aborted-run mark guards re-emission):

    * A recorded worker pid that has stayed dead past WATCHDOG_PID_GRACE_S.
      the grace covers the healthy seconds between the subprocess exiting
      and persist_predictions clearing the pid.
    * No recorded pid (report-only runs execute in the server process): no
      feed event for WATCHDOG_STALL_S while the run claims to be active.

    Checks every run in flight, or just ``run_id``. Every
    _DB_REAP_INTERVAL_S it also fails the run rows the feed no longer
    vouches for (reap_orphans): a row the feed never lists (its process
    died before closing it, or the server restarted and emptied the list)
    would otherwise stay queued/running and lock its owner out. Returns
    True when it aborted a run on this call.
    """
    global _last_db_reap
    try:
        if not _enabled():
            return False
        c = _get_cache()
        runs = [run_id] if run_id else _active_runs()
        now = time.time()
        aborted = False
        for rid in runs:
            aborted = _watchdog_one(c, rid, now) or aborted
        if run_id is None and now - _last_db_reap >= _DB_REAP_INTERVAL_S:
            _last_db_reap = now
            aborted = bool(reap_orphans()) or aborted
        return aborted
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
    """Refill the rolling log from the audit trail on a cold start.

    Without this a fresh session shows an empty panel even though the history
    exists in Postgres. Returns the number of events restored. Nothing can
    be in flight across a server restart, so the active list is cleared;
    the run rows that were in flight are failed by the caller
    (reap_orphans(max_age_s=0)), not here, this is the feed's side only.

    Always self-scoped, even for admins: this refills the live diskcache feed,
    which is ONE shared cross-process channel that every viewer polls. Pouring
    other users' rows into it would put them on everyone's live panel, which
    is not what the Activity Log's admin view means.
    """
    # The clear does not wait on the audit query: a boot whose first DB
    # read fails must still not present last life's runs as in flight (the
    # startup reap trusts this list as its witness).
    try:
        with _get_cache().transact():
            _set_active(_get_cache(), [])
    except Exception as e:
        logger.debug(f"active-run list not cleared at boot: {e}")
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

        _get_cache().set(_EVENTS_KEY, events)
        return len(events)
    except Exception as e:
        logger.debug(f"activity_log hydrate skipped: {e}")
        return 0


def _new_event(stage: str, message: str) -> dict:
    now = datetime.now(timezone.utc)
    return {
        # Both fields are UTC-derived: "t" is the epoch every renderer
        # prefers, "ts" is the same instant already rendered in DISPLAY_TZ
        # for anything reading it raw.
        "ts": format_clock(now),
        "t": now.timestamp(),
        "stage": stage,
        "message": message,
    }


def _append_feed(run_id: str, event: dict) -> None:
    """Publisher-only. One write under the cross-process lock: the run's own
    feed plus the rolling log the idle panel shows."""
    if not _enabled():
        return
    try:
        c = _get_cache()
        # transact() serializes the read-modify-write across threads AND
        # processes (SQLite-backed), the bare get/append/set lost events
        # whenever model threads and the prediction subprocess emitted
        # concurrently.
        with c.transact():
            log = c.get(_EVENTS_KEY) or []
            log.append(event)
            c.set(_EVENTS_KEY, log[-_MAX_EVENTS:])
            if run_id != ADHOC_RUN_ID:
                key = _events_key(run_id)
                events = c.get(key) or []
                events.append(event)
                c.set(key, events[-_MAX_EVENTS:], expire=_PER_RUN_TTL_S)
    except Exception as e:
        logger.debug(f"progress feed append failed: {e}")


def emit(stage: str, message: str, payload: dict | None = None, *,
         feed: bool = True, run_id: str | None = None,
         run_title: str | None = None) -> None:
    """Append an event. Never raises: progress must not break the pipeline.

    ``run_id`` names the run the event belongs to, in the feed and in the
    audit trail; unnamed events resolve as _resolve_run_id describes.
    ``feed=False`` writes the audit trail only. The scheduler's own status
    lines are emitted from the server process under the job's own id, so
    a "done" event from a scheduled job cannot close the user's in-flight
    run on the Trace page.

    The two sinks are gated separately. The diskcache feed is the live UI
    panel and stays publisher-only, so headless tools never interleave with a
    browser run; the durable audit trail always records, so a seed or backfill
    run outside the web process still shows up in the Activity Log.

    ``payload`` is optional structured data behind the message (counts,
    windows, hashes) for the Trace page. It goes to Postgres ONLY, the
    diskcache feed is read whole on every panel tick under a transact lock,
    so feed events must stay small; they carry at most a has_payload flag.
    """
    target = _resolve_run_id(run_id)
    if feed and _enabled():
        event = _new_event(stage, message)
        if payload is not None:
            event["has_payload"] = True
        _append_feed(target, event)

    _write_audit(stage, message, payload, run_id=target, run_title=run_title)


def _progress_message(stage: str, done, total, state, counters: dict) -> str:
    parts = [stage]
    if done is not None or total is not None:
        parts.append(f"{'?' if done is None else done}/"
                     f"{'?' if total is None else total}")
    if state:
        parts.append(state)
    if counters:
        parts.append(", ".join(f"{k}={v}" for k, v in counters.items()))
    return " ".join(parts)


def emit_progress(stage: str, done: int | None = None, total: int | None = None,
                  state: str | None = None, run_id: str | None = None,
                  **counters) -> None:
    """One structured progress event: where a stage stands, in numbers.

    Appends ``{stage, done, total, state, counters, ts}`` to the run's feed
    (the free-text emit() lines are the details behind it) and merges the
    same figures into the run's analysis_runs row through
    run_service.update_progress, which is what the pill, the ETA and the
    run page read. Counters are running TOTALS keyed by name, not
    increments. Never raises; a run with no row (headless, or a UI run
    before the dispatcher created one) still gets the feed event.
    """
    target = _resolve_run_id(run_id)
    counters = {k: v for k, v in counters.items() if v is not None}
    if _enabled():
        event = _new_event(stage, _progress_message(stage, done, total, state,
                                                    counters))
        event.update({"event": "progress", "done": done, "total": total,
                      "state": state, "counters": counters})
        _append_feed(target, event)
    if target == ADHOC_RUN_ID:
        return
    try:
        from services import run_service
        run_service.update_progress(target, stage, state=state, done=done,
                                    total=total, **counters)
    except Exception as e:
        logger.debug(f"run progress update skipped for {target}: {e}")


def finish_run(message: str = "Pipeline complete", run_id: str | None = None) -> None:
    """Mark a run finished: a closing line in its feed, out of the active
    list, its worker pid forgotten. The run row's status is the runner's
    job (run_service.set_status), not this feed's."""
    try:
        target = _resolve_run_id(run_id)
        emit("done", message, run_id=target)
        if _enabled() and target != ADHOC_RUN_ID:
            c = _get_cache()
            with c.transact():
                _remove_active(c, target)
                # Terminal state: the worker pid (if any) no longer matters.
                c.set(_pid_key(target), None)
                c.set(_dead_since_key(target), None)
    except Exception as e:
        logger.debug(f"progress finish_run failed: {e}")


def get_feed(run_id: str | None = None) -> dict:
    """``{active, run_id, events}`` for one run.

    With ``run_id`` the run's own feed. Without one, the newest run in
    flight; when nothing is in flight, the rolling log across runs (with the
    legacy active flag, which mark_run_pending may have armed ahead of a
    start_run), so the idle panel reads as it always has.
    """
    try:
        c = _get_cache()
        active = _active_runs()
        if run_id is None and active:
            run_id = active[-1]
        if run_id is None:
            return {
                "active": bool(c.get(_ACTIVE_KEY)),
                "run_id": c.get(_RUN_ID_KEY),
                "events": c.get(_EVENTS_KEY) or [],
            }
        return {
            "active": run_id in active,
            "run_id": run_id,
            "events": c.get(_events_key(run_id)) or [],
        }
    except Exception:
        return {"active": False, "run_id": run_id, "events": []}

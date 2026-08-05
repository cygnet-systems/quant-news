"""Cross-process progress feed for long-running pipelines.

Full Analysis spans a browser callback, a background subprocess (models),
and a synthesis callback — the user previously saw nothing until the end.
Stages emit events here; a dcc.Interval-driven panel streams them live and
late viewers catch up on the full feed.

diskcache is file-based, so the main process and the prediction subprocess
share the same feed without any wiring.
"""

import logging
import os
import time
from datetime import datetime

import diskcache

logger = logging.getLogger(__name__)

_CACHE_DIR = "cache/progress"
_EVENTS_KEY = "events"
_ACTIVE_KEY = "active"
_RUN_ID_KEY = "run_id"
_RUN_TITLE_KEY = "run_title"
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


def _write_audit(stage: str, message: str) -> None:
    """Append to the durable audit trail. Best-effort: the DB being down must
    never break a pipeline, and the diskcache feed still drives the panel."""
    try:
        from db.models import ActivityLog
        from db.session import get_session

        run_id, run_title = _run_identity()
        with get_session() as session:
            session.add(ActivityLog(
                user_id=_user_id(),
                run_id=run_id,
                run_title=run_title,
                stage=stage,
                message=message[:4000],
            ))
    except Exception as e:
        logger.debug(f"activity_log write skipped: {e}")


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
        else:
            _local_run["id"], _local_run["title"] = run_id, title
        emit("run", title)
    except Exception as e:
        logger.debug(f"progress start_run failed: {e}")


def get_activity_runs(limit_runs: int = 50, scope: str = "all") -> list[dict]:
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
    """
    try:
        from sqlalchemy import func, select, tuple_

        from db.models import ActivityLog
        from db.session import get_session

        show_all = scope == "all" and viewer_is_admin()

        with get_session() as session:
            recent_q = select(
                ActivityLog.user_id,
                ActivityLog.run_id,
                func.max(ActivityLog.created_at).label("last_at"),
            )
            if not show_all:
                recent_q = recent_q.where(ActivityLog.user_id == _user_id())
            recent = session.execute(
                recent_q
                .group_by(ActivityLog.user_id, ActivityLog.run_id)
                .order_by(func.max(ActivityLog.created_at).desc())
                .limit(limit_runs)
            ).all()
            keys = [(r.user_id, r.run_id) for r in recent]
            if not keys:
                return []

            rows = session.execute(
                select(ActivityLog)
                .where(tuple_(ActivityLog.user_id,
                              ActivityLog.run_id).in_(keys))
                .order_by(ActivityLog.created_at.asc(), ActivityLog.id.asc())
            ).scalars().all()

        by_run: dict[tuple[str, str], dict] = {}
        for r in rows:
            run = by_run.setdefault((r.user_id, r.run_id), {
                "run_id": r.run_id,
                "user_id": r.user_id,
                "title": r.run_title or "Activity",
                "started": r.created_at,
                "ended": r.created_at,
                "events": [],
                "errors": 0,
            })
            run["ended"] = r.created_at
            run["errors"] += 1 if r.stage == "error" else 0
            run["events"].append({
                "ts": r.created_at.strftime("%H:%M:%S"),
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
            "ts": r.created_at.strftime("%H:%M:%S"),
            "t": r.created_at.timestamp(),
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


def emit(stage: str, message: str) -> None:
    """Append an event. Never raises — progress must not break the pipeline.

    The two sinks are gated separately. The diskcache feed is the live UI
    panel and stays publisher-only, so headless tools never interleave with a
    browser run; the durable audit trail always records, so a seed or backfill
    run outside the web process still shows up in the Activity Log.
    """
    if _enabled():
        try:
            c = _get_cache()
            # transact() serializes the read-modify-write across threads AND
            # processes (SQLite-backed) — the bare get/append/set lost events
            # whenever model threads and the prediction subprocess emitted
            # concurrently.
            with c.transact():
                events = c.get(_EVENTS_KEY) or []
                events.append({
                    "ts": datetime.now().strftime("%H:%M:%S"),
                    "t": time.time(),
                    "stage": stage,
                    "message": message,
                })
                c.set(_EVENTS_KEY, events[-_MAX_EVENTS:])
        except Exception as e:
            logger.debug(f"progress emit failed: {e}")

    _write_audit(stage, message)


def finish_run(message: str = "Pipeline complete") -> None:
    """Mark the current run finished."""
    try:
        emit("done", message)
        if _enabled():
            _get_cache().set(_ACTIVE_KEY, False)
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

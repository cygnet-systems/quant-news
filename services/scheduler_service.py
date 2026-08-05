"""In-app job scheduling — the app owns its own clock.

Replaces the external cron/launchd arrangement: an APScheduler
``BackgroundScheduler`` lives in the server process (started from the ASGI
lifespan) and runs the daily analysis and evaluation itself. That works
because the app is deployed as a single always-on service with exactly one
uvicorn worker — the constraint the Dockerfile already states for models and
caches applies to the scheduler for the same reason.

Three things make this safe to run against a shared database:

**Advisory lock.** A deploy can briefly leave the old and new instances both
alive, and both would fire the same job. Every run takes a Postgres
session-level advisory lock keyed on the job id and gives up immediately if
another process holds it. An analysis firing twice is not just wasted CPU —
it is a duplicate LLM bill.

**Catch-up by date, not by boot.** A restart across the scheduled window would
otherwise silently skip the day. On startup each job whose window has already
passed today, and which has no success recorded *for today*, runs once. Keying
on the date rather than on "did we just boot" is what stops a crash-loop from
re-running an expensive job repeatedly.

**Subprocess execution.** Jobs shell out to ``scripts/daily_analysis.py``
rather than importing the pipeline into the web process. Torch/MPS state,
model memory and a 10-minute CPU burn stay outside the process serving the
UI, and a job that dies cannot take the server with it.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
from datetime import date, datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLI = str(PROJECT_ROOT / "scripts" / "daily_analysis.py")

ANALYSIS_JOB = "daily_analysis"
EVALUATION_JOB = "daily_evaluation"

# Wall-clock ceiling per run. A 20-symbol analysis is ~11 minutes locally and
# slower on a CPU-only container; 45 minutes is generous enough not to kill a
# healthy run and short enough that a hung provider call cannot hold the lock
# all day.
JOB_TIMEOUT_SECONDS = 45 * 60

_scheduler = None
_lock = threading.Lock()
# Schedule spec last applied to the live scheduler, per job — the comparison
# point for the database sync below.
_applied: dict[str, tuple] = {}
# Last overdue alert sent, so the half-hourly watchdog does not mail the
# same miss all day.
_last_overdue_alert: dict[str, object] = {}


# ---------------------------------------------------------------------------
# Job definitions
# ---------------------------------------------------------------------------

DEFAULT_JOBS = (
    {
        "id": ANALYSIS_JOB,
        "kind": "analysis",
        "description": "Full Analysis on the watchlist, before the open",
        # 08:30 ET — an hour before the open, and late enough that the
        # previous session's bar has settled with the vendor.
        "hour": 8,
        "minute": 30,
        "days_of_week": "mon-fri",
        "symbols_csv": ("PANW,BAC,VZ,HWM,DOC,HPQ,LUV,TPL,MPWR,MCD,"
                        "ROP,ETR,CMS,XYZ,HIG,IP,FLEX,MET,FIS,TYL"),
        "params_json": {"only_trading_days": True, "lookback": 7},
    },
    {
        "id": EVALUATION_JOB,
        "kind": "evaluation",
        "description": "Score predictions whose target session has closed",
        "hour": 18,
        "minute": 0,
        "days_of_week": "mon-fri",
        "symbols_csv": None,
        "params_json": {},
    },
)


def seed_default_jobs() -> None:
    """Create the two standard jobs if they don't exist. Never overwrites."""
    from db.models import ScheduledJob
    from db.session import get_session

    with get_session() as session:
        for spec in DEFAULT_JOBS:
            if session.get(ScheduledJob, spec["id"]) is None:
                session.add(ScheduledJob(**spec))
                logger.info(f"Seeded scheduled job: {spec['id']}")


def list_jobs() -> list[dict]:
    """Every job with its schedule, last outcome and next fire time."""
    from db.models import ScheduledJob
    from db.session import get_session
    from sqlalchemy import select

    with get_session() as session:
        rows = session.execute(
            select(ScheduledJob).order_by(ScheduledJob.hour, ScheduledJob.minute)
        ).scalars().all()
        jobs = [{
            "id": r.id,
            "kind": r.kind,
            "description": r.description,
            "enabled": r.enabled,
            "hour": r.hour,
            "minute": r.minute,
            "days_of_week": r.days_of_week,
            "timezone": r.timezone,
            "symbols_csv": r.symbols_csv,
            "params": r.params_json or {},
            "last_run_at": r.last_run_at,
            "last_status": r.last_status,
            "last_detail": r.last_detail,
            "last_duration_ms": r.last_duration_ms,
            "last_success_date": r.last_success_date,
        } for r in rows]

    for job in jobs:
        job["next_run_at"] = _next_run_at(job["id"])
        job["running"] = job["id"] in _running_jobs()
    return jobs


def update_job(job_id: str, **fields) -> bool:
    """Apply UI edits and reschedule immediately.

    Only the fields the dashboard exposes are writable; anything else is
    ignored rather than trusted, so a stray key cannot rewrite run history.
    """
    from db.models import ScheduledJob
    from db.session import get_session

    writable = {"enabled", "hour", "minute", "days_of_week", "timezone",
                "symbols_csv", "params_json", "description"}
    changes = {k: v for k, v in fields.items() if k in writable}
    if not changes:
        return False

    with get_session() as session:
        row = session.get(ScheduledJob, job_id)
        if row is None:
            return False
        for key, value in changes.items():
            setattr(row, key, value)
        row.updated_at = datetime.now()
        try:
            from services.auth_service import current_uid
            row.owner_uid = current_uid()
        except Exception:
            pass

    _reschedule(job_id)
    logger.info(f"Scheduled job {job_id} updated: {sorted(changes)}")
    return True


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

def _running_jobs() -> set[str]:
    from db.models import JobRun
    from db.session import get_session
    from sqlalchemy import select

    with get_session() as session:
        rows = session.execute(
            select(JobRun.job_id).where(JobRun.status == "running")
        ).scalars().all()
    return set(rows)


def _build_command(job: dict) -> list[str]:
    params = job.get("params") or {}
    if job["kind"] == "evaluation":
        return [sys.executable, CLI, "evaluate", "--json"]

    cmd = [sys.executable, CLI, "analyze", "--json"]
    if job.get("symbols_csv"):
        cmd += ["--symbols", job["symbols_csv"]]
    if params.get("only_trading_days", True):
        cmd.append("--only-trading-days")
    if params.get("lookback"):
        cmd += ["--lookback", str(params["lookback"])]
    if params.get("report_model"):
        cmd += ["--report-model", params["report_model"]]
    if params.get("recs_model"):
        cmd += ["--recs-model", params["recs_model"]]
    if params.get("force"):
        cmd.append("--force")
    return cmd


def run_job(job_id: str, trigger: str = "schedule") -> dict:
    """Execute one job under the advisory lock. Returns a result summary."""
    from db.models import JobRun, ScheduledJob
    from db.session import get_session
    from services import progress_service as prog

    with get_session() as session:
        row = session.get(ScheduledJob, job_id)
        if row is None:
            return {"status": "error", "detail": f"unknown job {job_id}"}
        job = {"id": row.id, "kind": row.kind, "symbols_csv": row.symbols_csv,
               "params": row.params_json or {}, "enabled": row.enabled}

    if not job["enabled"] and trigger == "schedule":
        return {"status": "skipped", "detail": "job disabled"}

    lock = _AdvisoryLock(job_id)
    if not lock.acquire():
        logger.info(f"{job_id}: another instance holds the lock — skipping")
        return {"status": "skipped", "detail": "held by another instance"}

    started = datetime.now()
    with get_session() as session:
        run = JobRun(job_id=job_id, trigger=trigger, status="running")
        session.add(run)
        session.flush()
        run_pk = run.id

    prog.emit("run", f"Scheduled job started: {job_id} ({trigger})")
    cmd = _build_command(job)
    status, detail = "success", ""
    try:
        proc = subprocess.run(
            cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True,
            timeout=JOB_TIMEOUT_SECONDS,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        tail = (proc.stdout or "").strip().splitlines()
        detail = "\n".join(tail[-25:]) if tail else ""
        if proc.returncode != 0:
            status = "error"
            err = (proc.stderr or "").strip().splitlines()
            detail = (detail + "\n" + "\n".join(err[-15:])).strip()
    except subprocess.TimeoutExpired:
        status, detail = "error", f"exceeded {JOB_TIMEOUT_SECONDS}s wall clock"
    except Exception as e:
        status, detail = "error", str(e)
    finally:
        duration_ms = int((datetime.now() - started).total_seconds() * 1000)
        try:
            with get_session() as session:
                run = session.get(JobRun, run_pk)
                if run is not None:
                    run.status = status
                    run.finished_at = datetime.now()
                    run.duration_ms = duration_ms
                    run.detail = detail[:4000] or None
                job_row = session.get(ScheduledJob, job_id)
                if job_row is not None:
                    job_row.last_run_at = started
                    job_row.last_status = status
                    job_row.last_detail = detail[:4000] or None
                    job_row.last_duration_ms = duration_ms
                    if status == "success":
                        job_row.last_success_date = started.date().isoformat()
        finally:
            lock.release()

    prog.emit("done" if status == "success" else "error",
              f"Scheduled job {job_id}: {status} in {duration_ms // 1000}s")
    logger.info(f"Scheduled job {job_id} finished: {status} ({duration_ms}ms)")

    _notify(job, status, detail, started, duration_ms)
    return {"status": status, "detail": detail, "duration_ms": duration_ms}


def _notify(job: dict, status: str, detail: str, started: datetime,
            duration_ms: int) -> None:
    """Mail the outcome. Never raises — the run already happened."""
    from services import notify_service

    try:
        if not notify_service.enabled():
            return
        if status != "success":
            notify_service.notify_job_failure(job["id"], detail)
        elif job["kind"] == "analysis":
            notify_service.notify_analysis(
                _parse_summary(detail),
                cost=notify_service.run_cost(started),
                duration_ms=duration_ms,
            )
        elif job["kind"] == "evaluation":
            notify_service.notify_evaluation()
    except Exception as e:
        logger.warning(f"notification for {job['id']} failed: {e}")


def _parse_summary(stdout_tail: str) -> dict:
    """Pull the CLI's JSON summary out of its captured output.

    The tail may carry log lines ahead of the JSON, so scan for the first line
    that opens an object and parse from there rather than assuming position.
    """
    import json

    lines = (stdout_tail or "").splitlines()
    for i, line in enumerate(lines):
        if line.strip().startswith("{"):
            try:
                return json.loads("\n".join(lines[i:]))
            except json.JSONDecodeError:
                continue
    return {}


class _AdvisoryLock:
    """Postgres session-level advisory lock on a dedicated connection.

    Session-scoped rather than transaction-scoped: the lock has to outlive the
    ORM sessions the job opens while it runs. Held on its own connection so a
    pooled connection being recycled mid-job cannot drop it.
    """

    def __init__(self, key: str) -> None:
        self._key = key
        self._conn = None

    def acquire(self) -> bool:
        from sqlalchemy import text

        from db.session import get_engine
        try:
            self._conn = get_engine().connect()
            got = self._conn.execute(
                text("SELECT pg_try_advisory_lock(hashtext(:k))"), {"k": self._key}
            ).scalar()
            if not got:
                self._conn.close()
                self._conn = None
                return False
            return True
        except Exception as e:
            logger.warning(f"advisory lock failed for {self._key}: {e}")
            if self._conn is not None:
                self._conn.close()
                self._conn = None
            # No lock means no coordination; running is still better than a
            # silently skipped day on a single-instance deployment.
            return True

    def release(self) -> None:
        if self._conn is None:
            return
        from sqlalchemy import text
        try:
            self._conn.execute(
                text("SELECT pg_advisory_unlock(hashtext(:k))"), {"k": self._key})
        except Exception as e:
            logger.debug(f"advisory unlock failed for {self._key}: {e}")
        finally:
            self._conn.close()
            self._conn = None


# ---------------------------------------------------------------------------
# Scheduler wiring
# ---------------------------------------------------------------------------

def _next_run_at(job_id: str) -> Optional[datetime]:
    if _scheduler is None:
        return None
    job = _scheduler.get_job(job_id)
    return getattr(job, "next_run_time", None) if job else None


def _reschedule(job_id: str) -> None:
    """Rebuild one job's trigger from its current DB row."""
    from db.models import ScheduledJob
    from db.session import get_session

    if _scheduler is None:
        return

    import pytz
    from apscheduler.triggers.cron import CronTrigger

    with get_session() as session:
        row = session.get(ScheduledJob, job_id)
        if row is None:
            return
        spec = (row.enabled, row.hour, row.minute, row.days_of_week, row.timezone)

    enabled, hour, minute, dow, tz = spec
    existing = _scheduler.get_job(job_id)
    if not enabled:
        if existing:
            _scheduler.remove_job(job_id)
            logger.info(f"Scheduled job {job_id} disabled")
        return

    _scheduler.add_job(
        run_job, CronTrigger(day_of_week=dow, hour=hour, minute=minute,
                             timezone=pytz.timezone(tz)),
        args=[job_id], id=job_id, replace_existing=True,
        max_instances=1, coalesce=True, misfire_grace_time=3600,
    )


def _sync_from_db() -> None:
    """Reconcile the live triggers with the stored schedule.

    An edit only reaches the APScheduler instance in the process that handled
    it. A second instance (a Railway deploy mid-rollout, or a schedule changed
    straight in the database) would otherwise keep firing the old times until
    it restarted. Polling the stored ``updated_at`` once a minute is cheap and
    makes the database the single source of truth it is supposed to be.
    """
    from db.models import ScheduledJob
    from db.session import get_session
    from sqlalchemy import select

    if _scheduler is None:
        return
    try:
        with get_session() as session:
            rows = session.execute(select(ScheduledJob)).scalars().all()
            current = {r.id: (r.enabled, r.hour, r.minute, r.days_of_week,
                              r.timezone, r.updated_at) for r in rows}
    except Exception as e:
        logger.debug(f"scheduler sync skipped: {e}")
        return

    for job_id, spec in current.items():
        if _applied.get(job_id) != spec:
            _reschedule(job_id)
            _applied[job_id] = spec
            logger.info(f"Scheduler synced {job_id} from database")

    for gone in set(_applied) - set(current):
        if _scheduler.get_job(gone):
            _scheduler.remove_job(gone)
        _applied.pop(gone, None)


def _catch_up() -> None:
    """Run jobs whose window passed today with no success recorded today.

    Keyed on ``last_success_date`` rather than on process start: a restart
    loop must not re-run a 10-minute analysis on every boot.
    """
    import pytz

    for job in list_jobs():
        if not job["enabled"]:
            continue
        now = datetime.now(pytz.timezone(job["timezone"]))
        today = now.date()
        if today.strftime("%a").lower() not in _weekday_set(job["days_of_week"]):
            continue
        window_passed = (now.hour, now.minute) >= (job["hour"], job["minute"])
        if not window_passed:
            continue
        if job["last_success_date"] == today.isoformat():
            continue
        logger.info(f"Catch-up: {job['id']} missed its {job['hour']:02d}:"
                    f"{job['minute']:02d} window today — running now")
        run_job(job["id"], trigger="catchup")


def health() -> dict:
    """Liveness of the scheduler and whether any job has missed its window.

    Catch-up handles the app being DOWN across a window. Nothing handled the
    app being UP and the job failing, being skipped by a stuck lock, or the
    scheduler thread having died — all of which look identical from outside:
    no analysis appears and no one is told. This is the signal for that, and
    it is what an uptime monitor should watch rather than "does / return 200".
    """
    import pytz

    jobs, overdue = [], []
    for job in list_jobs():
        entry = {
            "id": job["id"],
            "enabled": job["enabled"],
            "schedule": f"{job['hour']:02d}:{job['minute']:02d} "
                        f"{job['days_of_week']} {job['timezone']}",
            "next_run_at": job["next_run_at"].isoformat() if job["next_run_at"] else None,
            "last_status": job["last_status"],
            "last_success_date": job["last_success_date"],
            "running": job["running"],
        }
        entry["overdue"] = _is_overdue(job, pytz.timezone(job["timezone"]))
        if entry["overdue"]:
            overdue.append(job["id"])
        jobs.append(entry)

    alive = _scheduler is not None and getattr(_scheduler, "running", False)
    return {
        "scheduler_running": alive,
        "jobs": jobs,
        "overdue": overdue,
        "healthy": alive and not overdue,
    }


def _is_overdue(job: dict, tz) -> bool:
    """True when today's window has passed by a margin with no success today.

    The margin is the job's own timeout plus five minutes: a run that is
    simply still going must not read as a miss.
    """
    if not job["enabled"] or job["running"]:
        return False
    now = datetime.now(tz)
    if now.strftime("%a").lower() not in _weekday_set(job["days_of_week"]):
        return False
    if job["last_success_date"] == now.date().isoformat():
        return False
    scheduled_minutes = job["hour"] * 60 + job["minute"]
    grace = (JOB_TIMEOUT_SECONDS // 60) + 5
    return (now.hour * 60 + now.minute) > scheduled_minutes + grace


def _watchdog() -> None:
    """Log and record a job that should have run today and has not."""
    from services import progress_service as prog

    try:
        state = health()
    except Exception as e:
        logger.debug(f"watchdog skipped: {e}")
        return
    for job_id in state["overdue"]:
        msg = (f"Scheduled job {job_id} has no successful run today and its "
               f"window has passed — check /healthz")
        logger.error(msg)
        prog.emit("error", msg)

    # Mail once per overdue set per day: the watchdog runs every 30 minutes,
    # and an inbox full of the same alert is an alert nobody reads.
    if state["overdue"]:
        today = date.today().isoformat()
        signature = (today, tuple(sorted(state["overdue"])))
        if _last_overdue_alert.get("signature") != signature:
            from services import notify_service
            try:
                if notify_service.enabled():
                    notify_service.notify_overdue(state["overdue"])
                _last_overdue_alert["signature"] = signature
            except Exception as e:
                logger.warning(f"overdue notification failed: {e}")


def _weekday_set(days_of_week: str) -> set[str]:
    """Expand an APScheduler day spec ('mon-fri', 'mon,wed') to day names."""
    names = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
    spec = (days_of_week or "mon-fri").strip().lower()
    if spec in ("*", "all"):
        return set(names)
    out: set[str] = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            if start in names and end in names:
                i, j = names.index(start), names.index(end)
                out.update(names[i:j + 1] if i <= j else names[i:] + names[:j + 1])
        elif part in names:
            out.add(part)
    return out or set(names[:5])


def start() -> None:
    """Start the scheduler and register every enabled job. Idempotent."""
    global _scheduler

    with _lock:
        if _scheduler is not None:
            return
        if os.environ.get("SCHEDULER_ENABLED", "1").lower() in ("0", "false", "no"):
            logger.info("Scheduler disabled by SCHEDULER_ENABLED")
            return
        try:
            from apscheduler.schedulers.background import BackgroundScheduler
            from apscheduler.triggers.date import DateTrigger
            from apscheduler.triggers.interval import IntervalTrigger

            seed_default_jobs()
            _scheduler = BackgroundScheduler()
            _scheduler.start()
            for job in list_jobs():
                _reschedule(job["id"])
            _sync_from_db()
            # Catch-up runs on the scheduler thread — inline it would block
            # server startup for the length of the job it decides to run.
            _scheduler.add_job(_catch_up, DateTrigger(), id="_catch_up")
            # Pick up schedule edits made by another instance (or straight in
            # the database) without waiting for a restart.
            _scheduler.add_job(
                _sync_from_db, IntervalTrigger(seconds=60), id="_sync_from_db",
                max_instances=1, coalesce=True,
            )
            # Notice a job that silently didn't happen. Half-hourly is often
            # enough to catch a miss the same morning without adding noise.
            _scheduler.add_job(
                _watchdog, IntervalTrigger(minutes=30), id="_watchdog",
                max_instances=1, coalesce=True,
            )
            summary = ", ".join(
                f"{j['id']} {j['hour']:02d}:{j['minute']:02d} {j['timezone']}"
                for j in list_jobs() if j["enabled"]
            )
            logger.info(f"Scheduler started: {summary or 'no enabled jobs'}")
        except Exception as e:
            logger.error(f"Scheduler startup failed: {e}")
            _scheduler = None


def shutdown() -> None:
    global _scheduler
    with _lock:
        if _scheduler is not None:
            _scheduler.shutdown(wait=False)
            _scheduler = None


def recent_runs(job_id: str | None = None, limit: int = 20) -> list[dict]:
    from db.models import JobRun
    from db.session import get_session
    from sqlalchemy import select

    with get_session() as session:
        stmt = select(JobRun).order_by(JobRun.started_at.desc()).limit(limit)
        if job_id:
            stmt = stmt.where(JobRun.job_id == job_id)
        rows = session.execute(stmt).scalars().all()
        return [{
            "id": r.id, "job_id": r.job_id, "trigger": r.trigger,
            "status": r.status, "started_at": r.started_at,
            "duration_ms": r.duration_ms, "detail": r.detail,
        } for r in rows]

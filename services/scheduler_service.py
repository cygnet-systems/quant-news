"""In-app job scheduling, the app owns its own clock.

Replaces the external cron/launchd arrangement: an APScheduler
``BackgroundScheduler`` lives in the server process (started from the ASGI
lifespan) and runs the daily analysis and evaluation itself. That works
because the app is deployed as a single always-on service with exactly one
uvicorn worker: the constraint the Dockerfile already states for models and
caches applies to the scheduler for the same reason.

Three things make this safe to run against a shared database:

**Advisory lock.** A deploy can briefly leave the old and new instances both
alive, and both would fire the same job. Every run takes a Postgres
session-level advisory lock keyed on the job id and gives up immediately if
another process holds it. An analysis firing twice is not just wasted CPU.
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

import json
import logging
import os
import re
import subprocess

from config import MODEL
import sys
import threading
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLI = str(PROJECT_ROOT / "scripts" / "daily_analysis.py")

ANALYSIS_JOB = "daily_analysis"
EVALUATION_JOB = "daily_evaluation"

# Wall-clock ceiling per run, env-tunable. Sized from the measured scheduled
# runs of 2026-08-14..09-01: 41-55 min typical, 68.6 min worst (a day with
# sequential situation-classification calls). 95 min = worst observed x1.35 —
# enough that no healthy run has ever come within 25 minutes of it, small
# enough that a hung provider call cannot hold the advisory lock all morning.
# Resize with the measured distribution, not by feel: the near-ceiling
# warning below fires at 80% and is the signal that this needs raising.
JOB_TIMEOUT_SECONDS = int(os.getenv("JOB_TIMEOUT_MINUTES", "95")) * 60

# What a person reads afterwards to find out what the run did. The subprocess
# writes its summary to stdout and its log to stderr; both are kept, because
# "it said success and I have no idea what happened" is the state this panel
# exists to prevent.
RUN_LOG_MAX_CHARS = 16_000
RUN_LOG_MAX_LINES = 400

_scheduler = None
_lock = threading.Lock()
# Schedule spec last applied to the live scheduler, per job, the comparison
# point for the database sync below.
_applied: dict[str, tuple] = {}
# Last overdue alert sent, so the half-hourly watchdog does not mail the
# same miss all day.
_last_overdue_alert: dict[str, object] = {}


# ---------------------------------------------------------------------------
# Operation types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class JobType:
    """One kind of scheduled operation.

    Declarative so a new operation. Options flow, a rebalance, a different
    research pass: is an entry in this table plus a CLI verb, rather than
    another branch in the command builder and another special case in the UI.
    ``needs_symbols`` drives whether the form shows a symbol list at all.
    """

    kind: str
    label: str
    description: str
    verb: str                      # the daily_analysis.py subcommand
    needs_symbols: bool = False
    default_hour: int = 7
    default_minute: int = 0
    # Tunable knobs, rendered by the schedule UI as labeled inputs and stored
    # in params_json: (param_key, label, default, help). Declarative so a new
    # knob is one tuple here, not a form change, a callback change and a
    # command-builder change.
    params_spec: tuple = ()


JOB_TYPES: dict[str, JobType] = {
    "analysis": JobType(
        kind="analysis",
        label="Predict",
        description="Run every model over a watchlist and synthesize the calls",
        verb="analyze",
        needs_symbols=True,
        default_hour=7,
        default_minute=0,
        # No params_spec: an analysis job is configured with the Run
        # dialog's full settings (default_run_params) in the Schedule modal.
    ),
    "evaluation": JobType(
        kind="evaluation",
        label="Evaluate",
        description="Score predictions whose target session has closed",
        verb="evaluate",
        needs_symbols=False,
        default_hour=18,
        default_minute=0,
    ),
    "replay": JobType(
        kind="replay",
        label="Ensemble replay",
        description="Re-score the four ensemble methods over every stored "
                    "member vote and report which would have done best",
        verb="replay",
        needs_symbols=False,
        default_hour=19,
        default_minute=0,
    ),
    "alpha_lab": JobType(
        kind="alpha_lab",
        label="Alpha Lab",
        description="Re-run the standing edge hypotheses (rank spread, event "
                    "drift, calibration gate) against all accumulated "
                    "outcomes and flag any that cross the pre-registered "
                    "significance bar",
        verb="alpha-lab",
        needs_symbols=False,
        default_hour=20,
        default_minute=0,
        params_spec=(
            ("event_move", "Event move %", 5.0,
             "One-day move that counts as an event for the drift test"),
            ("gate_p", "Gate probability", 0.55,
             "Calibrated-probability threshold for the trade-gate test"),
            ("top_frac", "Basket divisor", 6.0,
             "Rank basket size = symbols / this (6 = top/bottom sixth)"),
        ),
    ),
}


def list_job_types() -> list[dict]:
    """Operation types the UI can offer, for a create form."""
    return [
        {"kind": t.kind, "label": t.label, "description": t.description,
         "needs_symbols": t.needs_symbols,
         "default_hour": t.default_hour, "default_minute": t.default_minute,
         "params_spec": [
             {"key": k, "label": lbl, "default": dflt, "help": hlp}
             for k, lbl, dflt, hlp in t.params_spec
         ]}
        for t in JOB_TYPES.values()
    ]


# ---------------------------------------------------------------------------
# Job definitions
# ---------------------------------------------------------------------------

def default_run_params() -> dict:
    """Every setting the Run dialog offers, at its defaults — the shape an
    analysis job's params_json carries. One vocabulary for both entry points:
    a scheduled job is a saved Run dialog."""
    from layouts.modals import RUN_MODELS
    return {
        "only_trading_days": True,
        "lookback": MODEL.NEWS_LOOKBACK_DAYS,          # days, or "overnight"
        "max_articles": MODEL.NEWS_MAX_ARTICLES,        # newest N, 0 = all
        "models": [mid for mid, _, _ in RUN_MODELS],
        "run_ensemble": True,
        "ensemble": {
            "method": MODEL.ENSEMBLE_DEFAULT_METHOD,
            "min_agree": MODEL.ENSEMBLE_MIN_AGREE,
            "enabled_models": list(MODEL.ENSEMBLE_DEFAULT_ENABLED),
            "weights": dict(MODEL.ENSEMBLE_DEFAULT_WEIGHTS),
        },
        "report_model": MODEL.REPORT_MODEL,
        "depth": "thesis",                              # thesis | standard
        "recs": "auto",                                 # auto | signals | off
        "recs_model": MODEL.RECOMMENDATIONS_MODEL,
        "evidence": list(MODEL.DEFAULT_EVIDENCE),
        "tools": ["web_research"],
    }


DEFAULT_JOBS = (
    {
        "id": ANALYSIS_JOB,
        "kind": "analysis",
        "description": "Full Analysis on the watchlist, before the open",
        # 07:00 ET: two and a half hours before the open (was 08:30; a
        # 40-minute container run finishing near the bell left no reading
        # time). The previous session's bar settled overnight either way.
        "hour": 7,
        "minute": 0,
        "days_of_week": "mon-fri",
        "symbols_csv": ("PANW,BAC,VZ,HWM,DOC,HPQ,LUV,TPL,MPWR,MCD,"
                        "ROP,ETR,CMS,XYZ,HIG,IP,FLEX,MET,FIS,TYL"),
        "params_json": None,  # filled from default_run_params() at seed time
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
    for job in DEFAULT_JOBS:
        if job["params_json"] is None:
            job["params_json"] = default_run_params()
    from db.models import ScheduledJob
    from db.session import get_session
    from sqlalchemy import func, select

    with get_session() as session:
        # Only into an EMPTY table. Seeding per-id would resurrect a job the
        # user deliberately deleted on the next restart, which is a confusing
        # thing for a schedule to do.
        if session.execute(select(func.count()).select_from(ScheduledJob)).scalar():
            return
        for spec in DEFAULT_JOBS:
            session.add(ScheduledJob(**spec))
            logger.info(f"Seeded scheduled job: {spec['id']}")


def _viewer() -> tuple[Optional[str], bool]:
    """(uid, is_admin) for the current request; anonymous -> (None, False)."""
    try:
        from services.auth_service import current_user
        u = current_user()
        return (u.uid, u.is_admin) if u else (None, False)
    except Exception:
        return (None, False)


def can_manage_job(owner_uid: Optional[str],
                   viewer: tuple[Optional[str], bool] | None = None) -> bool:
    """Whether the current viewer may edit/delete a job.

    Unowned (legacy) jobs are manageable by anyone, owned jobs by their
    owner and Administrators. This is the write-side rule; visibility is
    list_jobs' read-side rule.
    """
    uid, is_admin = viewer if viewer is not None else _viewer()
    return is_admin or owner_uid is None or owner_uid == uid


def list_jobs() -> list[dict]:
    """The jobs this viewer may see, own jobs first.

    Visibility: public jobs and your own; Administrators see everything.
    Own-first ordering because everyone cares about their own schedules
    more than the shared ones, even when both matter.
    """
    from db.models import ScheduledJob
    from db.session import get_session
    from sqlalchemy import select

    uid, is_admin = _viewer()
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
            "params_spec": [
                {"key": k, "label": lbl, "default": dflt, "help": hlp}
                for k, lbl, dflt, hlp in
                (JOB_TYPES[r.kind].params_spec if r.kind in JOB_TYPES else ())
            ],
            "last_run_at": r.last_run_at,
            "last_status": r.last_status,
            "last_detail": r.last_detail,
            "last_duration_ms": r.last_duration_ms,
            "last_success_date": r.last_success_date,
            "owner_uid": r.owner_uid,
            "is_public": bool(r.is_public),
            "is_mine": bool(uid and r.owner_uid == uid),
            "can_manage": can_manage_job(r.owner_uid, (uid, is_admin)),
        } for r in rows
            if is_admin or bool(r.is_public) or r.owner_uid is None
            or (uid and r.owner_uid == uid)]

    jobs.sort(key=lambda j: (not j["is_mine"], j["hour"], j["minute"]))
    for job in jobs:
        job["next_run_at"] = _next_run_at(job["id"])
        job["running"] = job["id"] in _running_jobs()
        # The type drives what the form shows, so the UI never has to know
        # which kinds take a symbol list.
        job_type = JOB_TYPES.get(job["kind"])
        job["type_label"] = job_type.label if job_type else job["kind"]
        job["needs_symbols"] = bool(job_type and job_type.needs_symbols)
    return jobs


def create_job(kind: str, description: str, hour: int, minute: int,
               days_of_week: str = "mon-fri", timezone: str = "US/Eastern",
               symbols_csv: str | None = None,
               params: dict | None = None,
               is_public: bool = True) -> Optional[str]:
    """Add a scheduled job. Returns its id, or None if the type is unknown.

    The id is derived from the description so the run history and the
    advisory lock read as something a person chose, with a numeric suffix
    only when a name is reused.
    """
    from db.models import ScheduledJob
    from db.session import get_session

    job_type = JOB_TYPES.get(kind)
    if job_type is None:
        logger.warning(f"create_job: unknown operation type {kind!r}")
        return None

    base = re.sub(r"[^a-z0-9]+", "_", (description or job_type.label).lower()).strip("_")
    base = (base or job_type.kind)[:48]

    with get_session() as session:
        job_id, n = base, 2
        while session.get(ScheduledJob, job_id) is not None:
            job_id = f"{base}_{n}"
            n += 1
        session.add(ScheduledJob(
            id=job_id,
            kind=kind,
            description=description or job_type.label,
            enabled=True,
            hour=max(0, min(23, int(hour))),
            minute=max(0, min(59, int(minute))),
            days_of_week=days_of_week or "mon-fri",
            timezone=timezone or "US/Eastern",
            symbols_csv=symbols_csv if job_type.needs_symbols else None,
            params_json=params or {},
            owner_uid=_owner_uid(),
            # An anonymous session cannot own a private job, nobody could
            # ever see it again, so ownerless jobs are forced public.
            is_public=bool(is_public) or _owner_uid() is None,
        ))

    _reschedule(job_id)
    logger.info(f"Created scheduled job {job_id} ({kind})")
    return job_id


def delete_job(job_id: str) -> bool:
    """Remove a job from the schedule, keeping its run history.

    The runs stay: they are the record of what the platform did, and losing
    them because someone retired a job would make the scoreboard's provenance
    unexplainable.
    """
    from db.models import ScheduledJob
    from db.session import get_session

    with get_session() as session:
        row = session.get(ScheduledJob, job_id)
        if row is None:
            return False
        if not can_manage_job(row.owner_uid):
            logger.warning(f"delete_job: viewer may not delete {job_id} "
                           f"(owned by {row.owner_uid})")
            return False
        session.delete(row)

    if _scheduler is not None and _scheduler.get_job(job_id):
        _scheduler.remove_job(job_id)
    _applied.pop(job_id, None)
    logger.info(f"Deleted scheduled job {job_id}")
    return True


def _owner_uid():
    """Same attribution rule as every other writer (auth_service.effective_uid):
    the signed-in user, else the owner a scheduled subprocess runs as."""
    try:
        from services.auth_service import effective_uid
        return effective_uid()
    except Exception:
        return None


def update_job(job_id: str, **fields) -> bool:
    """Apply UI edits and reschedule immediately.

    Only the fields the dashboard exposes are writable; anything else is
    ignored rather than trusted, so a stray key cannot rewrite run history.
    """
    from db.models import ScheduledJob
    from db.session import get_session

    writable = {"enabled", "hour", "minute", "days_of_week", "timezone",
                "symbols_csv", "params_json", "description", "is_public"}
    changes = {k: v for k, v in fields.items() if k in writable}
    if not changes:
        return False

    with get_session() as session:
        row = session.get(ScheduledJob, job_id)
        if row is None:
            return False
        if not can_manage_job(row.owner_uid):
            logger.warning(f"update_job: viewer may not edit {job_id} "
                           f"(owned by {row.owner_uid})")
            return False
        for key, value in changes.items():
            setattr(row, key, value)
        row.updated_at = datetime.now(timezone.utc)
        # Editing used to REASSIGN ownership to whoever saved (and an
        # anonymous edit nulled it). Ownership now only flows one way:
        # a signed-in editor claims a legacy unowned job, nothing else.
        if row.owner_uid is None:
            row.owner_uid = _owner_uid()
        # A job that goes private without an owner would vanish for everyone.
        if not row.is_public and row.owner_uid is None:
            row.is_public = True

    _reschedule(job_id)
    logger.info(f"Scheduled job {job_id} updated: {sorted(changes)}")
    return True


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

def _running_jobs() -> set[str]:
    """Jobs genuinely in flight right now.

    Bounded by the job timeout, because a "running" row is only evidence that
    a process STARTED one, a container killed mid-run (every deploy does
    this) never gets to write its ending. Treating those as running forever
    would pin the job: the UI disables its Run-now button, and the overdue
    check skips anything running, so /healthz would report healthy while
    nothing had run for days.
    """
    from datetime import timedelta

    from db.models import JobRun
    from db.session import get_session
    from sqlalchemy import select

    cutoff = datetime.now().astimezone() - timedelta(seconds=JOB_TIMEOUT_SECONDS)
    with get_session() as session:
        rows = session.execute(
            select(JobRun.job_id).where(
                JobRun.status == "running",
                JobRun.started_at >= cutoff,
            )
        ).scalars().all()
    return set(rows)


def reap_abandoned_runs() -> int:
    """Close out runs whose process died, so history says so.

    Without this the row stays "running" and the failure is invisible in both
    the panel and the run table. Indistinguishable from a job still working.
    """
    from datetime import timedelta

    from db.models import JobRun, ScheduledJob
    from db.session import get_session
    from sqlalchemy import select

    cutoff = datetime.now().astimezone() - timedelta(seconds=JOB_TIMEOUT_SECONDS)
    reaped = 0
    with get_session() as session:
        stale = session.execute(
            select(JobRun).where(JobRun.status == "running",
                                 JobRun.started_at < cutoff)
        ).scalars().all()
        for run in stale:
            run.status = "interrupted"
            job = session.get(ScheduledJob, run.job_id)
            if job is not None and job.last_run_at is not None and (
                    job.last_run_at <= run.started_at):
                # The card showed the PREVIOUS run's green "success" next
                # to an "interrupted" row in the table.
                job.last_status = "interrupted"
                job.last_detail = (f"run started {run.started_at:%Y-%m-%d %H:%M} "
                                   f"never reported back (process died or "
                                   f"deploy restarted it)")
            run.finished_at = datetime.now(timezone.utc)
            run.detail = ((run.detail or "") +
                          "\nProcess ended before the run finished "
                          "(deploy, restart or crash).").strip()
            reaped += 1
    if reaped:
        logger.warning(f"Reaped {reaped} abandoned job run(s)")
    return reaped


def _build_command(job: dict, overrides: Optional[dict] = None) -> list[str]:
    params = {**(job.get("params") or {}), **(overrides or {})}
    job_type = JOB_TYPES.get(job["kind"])
    if job_type is None:
        raise ValueError(f"unknown operation type: {job['kind']}")

    cmd = [sys.executable, CLI, job_type.verb, "--json"]
    if not job_type.needs_symbols:
        # Symbol-less operations still take tuning params.
        if job["kind"] == "alpha_lab":
            if params.get("event_move"):
                cmd += ["--event-move", str(params["event_move"])]
            if params.get("gate_p"):
                cmd += ["--gate-p", str(params["gate_p"])]
            if params.get("top_frac"):
                cmd += ["--top-frac", str(params["top_frac"])]
        return cmd

    if job.get("symbols_csv"):
        cmd += ["--symbols", job["symbols_csv"]]
    if params.get("target"):
        cmd += ["--target", str(params["target"])]
    if params.get("only_trading_days", True):
        cmd.append("--only-trading-days")
    # Both REQUIRED for an analysis job — the CLI has no default for them,
    # and a job that lacks them fails here with a message that names the fix
    # rather than running on a window nobody chose. (Jobs created before the
    # Schedule modal existed: open the job there and save it once.)
    missing = [k for k in ("lookback", "max_articles") if params.get(k) is None]
    if missing:
        raise ValueError(
            f"job params missing {', '.join(missing)} — open this job on the "
            f"Schedule page and save it once to set them")
    overnight = str(params["lookback"]).strip().lower() == "overnight"
    cmd += ["--news-filter", "overnight" if overnight else "lookback"]
    cmd += ["--lookback", "1" if overnight else str(int(params["lookback"]))]
    cmd += ["--max-articles", str(int(params["max_articles"]))]
    models = params.get("models")
    if models:
        cmd += ["--models", models if isinstance(models, str) else ",".join(models)]
    if params.get("report_model"):
        cmd += ["--report-model", params["report_model"]]
    if params.get("recs_model"):
        cmd += ["--recs-model", params["recs_model"]]
    if params.get("depth"):
        cmd += ["--depth", str(params["depth"])]
    if params.get("recs"):
        cmd += ["--recs", str(params["recs"])]
    if params.get("evidence") is not None:
        ev = params["evidence"]
        cmd += ["--evidence", (ev if isinstance(ev, str) else ",".join(ev)) or "none"]
    if params.get("ensemble"):
        cmd += ["--ensemble-json", json.dumps(params["ensemble"])]
    if params.get("run_ensemble") is False:
        cmd.append("--no-ensemble")
    # tools: the list from the modal; a pre-modal job stored web_research=1.
    tools = params.get("tools")
    if tools is None and int(params.get("web_research") or 0):
        tools = ["web_research"]
    if tools:
        cmd += ["--tools", tools if isinstance(tools, str) else ",".join(tools)]
    if params.get("force"):
        cmd.append("--force")
    return cmd


def run_job(job_id: str, trigger: str = "schedule",
            overrides: Optional[dict] = None) -> dict:
    """Execute one job under the advisory lock. Returns a result summary.

    ``overrides`` are merged over the job's stored params for this run only.
    the backfill path uses this to pass an explicit ``target`` session.
    """
    from db.models import JobRun, ScheduledJob
    from db.session import get_session
    from services import progress_service as prog

    with get_session() as session:
        row = session.get(ScheduledJob, job_id)
        if row is None:
            return {"status": "error", "detail": f"unknown job {job_id}"}
        job = {"id": row.id, "kind": row.kind, "symbols_csv": row.symbols_csv,
               "params": row.params_json or {}, "enabled": row.enabled,
               "timezone": row.timezone,
               "owner_uid": row.owner_uid, "is_public": bool(row.is_public)}

    if not job["enabled"] and trigger == "schedule":
        return {"status": "skipped", "detail": "job disabled"}

    lock = _AdvisoryLock(job_id)
    if not lock.acquire():
        logger.info(f"{job_id}: another instance holds the lock, skipping")
        return {"status": "skipped", "detail": "held by another instance"}

    started = datetime.now(timezone.utc)
    with get_session() as session:
        run = JobRun(job_id=job_id, trigger=trigger, status="running",
                     owner_uid=job["owner_uid"])
        session.add(run)
        session.flush()
        run_pk = run.id

    job_run_id = f"job:{job_id}:{run_pk}"
    prog.emit("run", f"Scheduled job started: {job_id} ({trigger})",
              feed=False, run_id=job_run_id, run_title=f"Scheduled job {job_id}")
    status, detail, summary = "success", "", {}
    # Inside the guarded region: a bad params_json value (e.g. a non-numeric
    # lookback) used to raise here, BEFORE the try/finally, leaving the run
    # row "running" forever and the advisory lock held.
    try:
        cmd = _build_command(job, overrides)
    except Exception as e:
        cmd = None
        status, detail = "failed", f"could not build command: {e}"
    # Run under the job owner's identity: the subprocess has no request
    # context, so the uid travels by env. Everything the run writes
    # (predictions, reports, activity rows) is attributed to the owner, and
    # a private job's output is private to them (auth_service.effective_uid
    # / run_is_public read these; progress_service reads QUANTNEWS_USER).
    run_env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    if job["owner_uid"]:
        run_env["QUANTNEWS_RUN_OWNER"] = job["owner_uid"]
        run_env["QUANTNEWS_USER"] = job["owner_uid"]
    run_env["QUANTNEWS_RUN_PUBLIC"] = "1" if job["is_public"] else "0"
    try:
        if cmd is None:
            raise RuntimeError(detail)
        # Popen + its own session, NOT subprocess.run: on timeout, run() kills
        # only the direct child and then drains its pipes with no timeout. 
        # but the CLI spawns model workers that inherit those pipes, so a
        # surviving grandchild wedges the drain (and this thread, and the
        # bookkeeping below) forever. Seen live 2026-09-01: a manual run
        # stuck "running" for 6h40m past a 75-minute ceiling. Killing the
        # whole process group closes every pipe holder, so the second
        # communicate() below always returns.
        proc = subprocess.Popen(
            cmd, cwd=str(PROJECT_ROOT), stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, env=run_env,
            start_new_session=True,
        )
        try:
            out, err = proc.communicate(timeout=JOB_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            import signal as _signal
            try:
                # start_new_session makes the child its own group leader, so
                # its pid is the pgid.
                os.killpg(proc.pid, _signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                proc.kill()
            out, err = proc.communicate()
            status = "error"
            # The output it managed before the kill is the only evidence of
            # where it hung, so keep it rather than reporting the ceiling
            # alone.
            detail = _run_log(out, err,
                              header=f"TIMED OUT: exceeded "
                                     f"{JOB_TIMEOUT_SECONDS}s wall clock; "
                                     f"process group killed.")
        else:
            # Parse the summary from the WHOLE of stdout, never from the
            # stored tail. A 20-symbol summary is 53 lines of indented JSON,
            # so a 25-line tail cannot contain a parseable object, which is
            # why every scheduled run mailed "produced no recommendations"
            # while having stored 20 calls and exited zero (2026-08-06).
            summary = _parse_summary(out)
            detail = _run_log(out, err)
            if proc.returncode == 2:
                # Ran and stored something, but not everything asked for.
                status = "partial"
            elif proc.returncode != 0:
                status = "error"
    except Exception as e:
        status, detail = "error", str(e)
    finally:
        duration_ms = int((datetime.now(timezone.utc) - started).total_seconds() * 1000)
        try:
            with get_session() as session:
                run = session.get(JobRun, run_pk)
                if run is not None:
                    run.status = status
                    run.finished_at = datetime.now(timezone.utc)
                    run.duration_ms = duration_ms
                    run.detail = detail or None
                job_row = session.get(ScheduledJob, job_id)
                if job_row is not None:
                    job_row.last_run_at = started
                    job_row.last_status = status
                    job_row.last_detail = detail or None
                    job_row.last_duration_ms = duration_ms
                    # A back-dated run must not stamp today as done, that
                    # would make catch-up skip the day's own window.
                    if status == "success" and not (overrides or {}).get("target"):
                        # The date must be taken in the JOB's timezone, not
                        # the container's. On a UTC host, a 20:00-ET run is
                        # already "tomorrow" in UTC, stamping that date made
                        # the watchdog (which asks about today in job tz)
                        # mail a false "overdue" minutes after a success.
                        # This stamp shares a transaction with the run row's
                        # own finalize: an exception here rolls back BOTH,
                        # leaving the run "running" forever and the day
                        # unstamped: which is how a missing job key turned
                        # one good 07:00 run into 28 phantom catch-up reruns
                        # (2026-08-11). The date is best-effort; the row
                        # bookkeeping is not allowed to die for it.
                        import pytz
                        try:
                            local = started.astimezone(
                                pytz.timezone(job.get("timezone") or "UTC"))
                        except Exception:
                            local = started
                        job_row.last_success_date = local.date().isoformat()
        finally:
            lock.release()

    prog.emit("done" if status == "success" else "error",
              f"Scheduled job {job_id}: {status} in {duration_ms // 1000}s",
              feed=False, run_id=job_run_id, run_title=f"Scheduled job {job_id}")
    logger.info(f"Scheduled job {job_id} finished: {status} ({duration_ms}ms)")

    # A run that finishes just inside the ceiling is a failure waiting for a
    # slow provider day, and nothing else would say so until the day it is
    # killed mid-flight.
    if duration_ms > 0.8 * JOB_TIMEOUT_SECONDS * 1000:
        near = (f"{job_id} took {duration_ms // 1000}s of a "
                f"{JOB_TIMEOUT_SECONDS}s ceiling: raise the timeout or "
                f"shorten the watchlist before it gets killed mid-run")
        logger.warning(near)
        prog.emit("error", near, feed=False, run_id=job_run_id,
                  run_title=f"Scheduled job {job_id}")

    _notify(job, status, summary, detail, started, duration_ms)
    return {"status": status, "detail": detail, "duration_ms": duration_ms}


def _notify(job: dict, status: str, summary: dict, log: str, started: datetime,
            duration_ms: int) -> None:
    """Mail the outcome. Never raises: the run already happened.

    Takes the parsed summary and the run log separately: the summary decides
    which mail to send, the log is what the mail shows a person.
    """
    from services import notify_service

    try:
        if not notify_service.enabled():
            return
        # A market holiday is not an outcome worth mailing about, the CLI
        # no-ops by design when --only-trading-days is set.
        if summary.get("no_session"):
            return
        if status == "partial":
            notify_service.notify_partial(
                job["id"], summary.get("degraded") or [], summary)
        elif status != "success":
            notify_service.notify_job_failure(job["id"], log)
        elif job["kind"] == "analysis":
            notify_service.notify_analysis(
                summary,
                cost=notify_service.run_cost(started),
                duration_ms=duration_ms,
                log=log,
            )
        elif job["kind"] == "evaluation":
            notify_service.notify_evaluation()
        elif job["kind"] == "alpha_lab":
            notify_service.notify_alpha_lab(summary)
    except Exception as e:
        logger.warning(f"notification for {job['id']} failed: {e}")


def _parse_summary(stdout: str) -> dict:
    """Pull the CLI's JSON summary out of its captured stdout.

    Must be given the whole of stdout: the object spans as many lines as the
    watchlist is long. Log lines may precede it, so scan for a line that opens
    an object and decode from there, tolerating anything printed after it.
    """
    import json

    decoder = json.JSONDecoder()
    text = stdout or ""
    for i, line in enumerate(text.splitlines()):
        if not line.startswith("{"):
            continue
        rest = "\n".join(text.splitlines()[i:])
        try:
            obj, _ = decoder.raw_decode(rest)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return {}


def _run_log(stdout: str | None, stderr: str | None, header: str = "") -> str:
    """The run's own account of itself, for the panel and the failure mail.

    Both streams are kept. Only stdout used to be, and only its last 25 lines,
    so a run that succeeded left no trace of what it actually did and a run
    that failed inside the pipeline showed the summary rather than the error.
    The subprocess logs to stderr (``basicConfig`` writes there), which is the
    half worth reading.
    """
    summary_block = (stdout or "").strip()
    if header:
        summary_block = f"{header}\n\n{summary_block}".strip()

    log_lines = (stderr or "").strip().splitlines()
    if not log_lines:
        return summary_block[:RUN_LOG_MAX_CHARS]

    dropped = max(0, len(log_lines) - RUN_LOG_MAX_LINES)
    tail = log_lines[-RUN_LOG_MAX_LINES:]
    marker = (f"--- run log (last {len(tail)} of {len(log_lines)} lines) ---"
              if dropped else "--- run log ---")
    # Trim the log rather than the summary: the summary is what the panel is
    # read for, and it is bounded while the log is not.
    budget = max(0, RUN_LOG_MAX_CHARS - len(summary_block) - len(marker) - 4)
    log_block = "\n".join(tail)[-budget:]
    return f"{summary_block}\n\n{marker}\n{log_block}".strip()


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
        # Already attempted today, do not retry automatically. A partial run
        # deliberately withholds last_success_date so it shows as overdue, and
        # without this a restart loop would re-run an expensive analysis on
        # every boot chasing a completeness it cannot reach on its own.
        # Re-running a partial is a decision, so it stays manual.
        last_run = job.get("last_run_at")
        if last_run and last_run.astimezone(now.tzinfo).date() == today:
            logger.info(f"Catch-up: {job['id']} already attempted today "
                        f"({job.get('last_status')}): not retrying automatically")
            continue
        logger.info(f"Catch-up: {job['id']} missed its {job['hour']:02d}:"
                    f"{job['minute']:02d} window today: running now")
        run_job(job["id"], trigger="catchup")

    _backfill_missed_sessions()


# The furthest back a backfill will reach. Predictions older than this have
# lost their pre-open news context anyway; a wider hole means the deployment
# was down long enough that a person should decide what to rerun.
BACKFILL_MAX_SESSIONS = 5


def _backfill_missed_sessions() -> None:
    """Re-run analysis for recent trading sessions that have no predictions.

    Catch-up above only covers *today*; a container that slept through a
    window (Railway app sleep freezes the scheduler thread, and APScheduler
    drops fires older than its misfire grace) used to lose the day for good.
    there is no per-date ledger, so a five-day gap looked identical to no gap.
    The ledger here is the predictions table itself: a past session with zero
    rows targeting it was never analysed, so run it with an explicit
    ``--target``. Idempotent by construction.
    """
    from sqlalchemy import func, select

    from db.models import ModelPrediction
    from db.session import get_session
    from utils.trading_calendar import is_trading_day

    analysis_jobs = [
        j for j in list_jobs()
        if j["enabled"]
        and (jt := JOB_TYPES.get(j["kind"])) is not None
        and jt.verb == "analyze"
    ]
    if not analysis_jobs:
        return

    today = date.today()
    sessions = [
        d for d in (today - timedelta(days=n) for n in range(1, 15))
        if is_trading_day(d)
    ][:BACKFILL_MAX_SESSIONS]

    # Per job, not per date: one ad-hoc single-symbol backtest used to mark
    # the whole session as analysed and suppress the watchlist backfill.
    missing_by_job: dict[str, list[date]] = {}
    with get_session() as session:
        for job in analysis_jobs:
            syms = [x.strip().upper() for x in (job.get("symbols_csv") or "").split(",")
                    if x.strip()]
            for d in sessions:
                q = (select(func.count()).select_from(ModelPrediction)
                     .where(func.date(ModelPrediction.target_date) == d))
                if syms:
                    q = q.where(ModelPrediction.symbol.in_(syms))
                if job.get("owner_uid"):
                    q = q.where(ModelPrediction.owner_uid == job["owner_uid"])
                if not session.execute(q).scalar():
                    missing_by_job.setdefault(job["id"], []).append(d)

    for job in analysis_jobs:
        missing = missing_by_job.get(job["id"], [])
        for d in sorted(missing):
            logger.warning(f"Backfill: no predictions target {d}: running "
                           f"{job['id']} with --target {d}")
            result = run_job(
                job["id"], trigger="backfill",
                overrides={"target": d.isoformat(), "only_trading_days": False},
            )
            if result.get("status") not in ("success", "partial"):
                # One failure means the rest would likely fail the same way
                # (quota, vendor outage); stop rather than burn the budget.
                logger.warning(f"Backfill for {d} did not complete "
                               f"({result.get('status')}): stopping sweep")
                return


def scheduling_enabled() -> bool:
    """Whether THIS process is allowed to execute jobs.

    Deliberately per-process, and deliberately not the same lever as a job's
    ``enabled`` column: that one means "nobody should run this job" and takes
    every instance with it. This one means "not on this machine", the case
    where a laptop is pointed at the production database for UI work and
    should not win the lock and spend money on an analysis.
    """
    return os.environ.get("SCHEDULER_ENABLED", "1").strip().lower() not in (
        "0", "false", "no", "off"
    )


def health() -> dict:
    """Liveness of the scheduler and whether any job has missed its window.

    Catch-up handles the app being DOWN across a window. Nothing handled the
    app being UP and the job failing, being skipped by a stuck lock, or the
    scheduler thread having died, all of which look identical from outside:
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
    # A process told not to schedule is not a broken one. Without this
    # distinction an intentional stand-down is indistinguishable from a
    # crashed scheduler, and a monitor cannot tell which it is looking at.
    stood_down = not scheduling_enabled()
    return {
        "scheduler_running": alive,
        "scheduling_disabled": stood_down,
        "jobs": jobs,
        "overdue": overdue,
        "healthy": stood_down or (alive and not overdue),
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

    # Reaping used to happen only at boot, so a run whose thread wedged
    # mid-life stayed "running" until the next deploy (2026-09-01: 6h40m).
    # A row past the job ceiling is dead by definition. Close it here too.
    try:
        reaped = reap_abandoned_runs()
        if reaped:
            logger.warning(f"watchdog reaped {reaped} stale run(s)")
            prog.emit("error", f"Reaped {reaped} run(s) stuck past the "
                               f"{JOB_TIMEOUT_SECONDS // 60}-minute ceiling")
    except Exception as e:
        logger.debug(f"watchdog reap skipped: {e}")

    try:
        state = health()
    except Exception as e:
        logger.debug(f"watchdog skipped: {e}")
        return
    for job_id in state["overdue"]:
        msg = (f"Scheduled job {job_id} has no successful run today and its "
               f"window has passed, check /healthz")
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
        if not scheduling_enabled():
            logger.info(
                "SCHEDULER_ENABLED is off, this process will not run jobs. "
                "Another instance sharing this database still will."
            )
            return
        try:
            from apscheduler.schedulers.background import BackgroundScheduler
            from apscheduler.triggers.date import DateTrigger
            from apscheduler.triggers.interval import IntervalTrigger

            seed_default_jobs()
            # A restart is the usual reason a run never finished, so this is
            # exactly the moment to close those rows out.
            reap_abandoned_runs()
            _scheduler = BackgroundScheduler()
            _scheduler.start()
            for job in list_jobs():
                _reschedule(job["id"])
            _sync_from_db()
            # Catch-up runs on the scheduler thread. Inline it would block
            # server startup for the length of the job it decides to run.
            _scheduler.add_job(_catch_up, DateTrigger(), id="_catch_up")
            # And again half-hourly: a container thawed from platform sleep
            # never re-enters start(), and cron fires older than the misfire
            # grace are dropped, so the boot-time pass alone cannot recover a
            # slept-through window. misfire_grace_time=None means "run no
            # matter how late", being late is this job's entire purpose.
            _scheduler.add_job(
                _catch_up, IntervalTrigger(minutes=30), id="_catch_up_interval",
                max_instances=1, coalesce=True, misfire_grace_time=None,
            )
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


def recent_runs(job_id: str | None = None, limit: int = 20,
                detail_chars: int = 4000) -> list[dict]:
    """Run history, with each run's log abridged for the panel.

    The panel that shows these re-polls every 15 seconds, so shipping every
    run's full log would send a quarter of a megabyte a minute to each open
    tab to redraw a five-row table. The abridgement keeps both ends, the
    summary the run printed and the tail where a failure shows up, and the
    job card still carries the newest run's log in full.
    """
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
            "duration_ms": r.duration_ms,
            "detail": _abridge(r.detail, detail_chars),
        } for r in rows]


def _abridge(text: str | None, budget: int) -> str | None:
    """Keep the head and the tail of a log, drop the middle.

    Both ends carry meaning: the head is the summary the run printed, the tail
    is where it went wrong. Cutting from either end alone loses one of them.
    """
    if not text or budget <= 0 or len(text) <= budget:
        return text
    head = budget // 3
    tail = budget - head
    return (f"{text[:head]}\n\n… {len(text) - budget} characters omitted …\n\n"
            f"{text[-tail:]}")

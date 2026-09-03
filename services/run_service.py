"""The run record: one analysis_runs row per run, manual or scheduled.

Every function opens its own session so it is safe from any process: the
request thread that confirms a run, the background prediction subprocess,
and the scheduler's job subprocess all write to the same row by run_id.

Progress writes come from more than one process at once (the in-process
research stage and the model subprocess both report on the same run), so
update_progress locks the row, merges in Python and writes stages_json and
counters_json in one UPDATE; two concurrent reports for different stages
both survive. Terminal statuses are sticky: a subprocess that finishes after
the user cancelled must not flip the row back to done.

A row nobody closes is worse than a failed one: queued/running is the
per-owner lock, so an orphan (container died mid-run, scheduler killed the
runner at the ceiling, server restarted) refuses every confirm from its owner
until someone edits the row. orphan_reason / reap_orphans decide when an
active row is provably dead and fail it, from the runner's side of the
world, never from inside scheduler_service.run_job's finalize block.
"""

import logging
import uuid
from datetime import date, datetime, timezone

from sqlalchemy import func, select, update

from db.session import get_session

logger = logging.getLogger(__name__)

KINDS = ("scheduled", "manual")
STATUSES = ("queued", "running", "done", "failed", "cancelled")
ACTIVE_STATUSES = ("queued", "running")
TERMINAL_STATUSES = ("done", "failed", "cancelled")
STAGES = ("news", "models", "research", "synthesis", "report")
STAGE_STATES = ("pending", "running", "done", "failed", "skipped")
# A job_runs row in any other status has been finalized by run_job: the
# runner subprocess it tracked has exited, whatever the analysis_runs
# row still claims.
JOB_RUN_LIVE_STATUS = "running"
ORPHAN_RESTART_ERROR = "server restarted before the run finished"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _symbols_csv(symbols) -> str:
    """Upper-cased, de-duplicated, order preserved: the dialog's chip order
    is what the toast and the run page read back."""
    if isinstance(symbols, str):
        symbols = symbols.split(",")
    seen: list[str] = []
    for s in symbols or []:
        s = (s or "").strip().upper()
        if s and s not in seen:
            seen.append(s)
    return ",".join(seen)


def _coerce_date(value) -> date | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def _iso(value) -> str | None:
    return value.isoformat() if value is not None else None


def _to_dict(row) -> dict:
    csv = row.symbols_csv or ""
    return {
        "run_id": row.run_id,
        "owner_uid": row.owner_uid,
        "kind": row.kind,
        "status": row.status,
        "preset": row.preset,
        "config": row.config_json or {},
        "symbols_csv": csv,
        "symbols": csv.split(",") if csv else [],
        "prediction_date": _iso(row.prediction_date),
        "target_date": _iso(row.target_date),
        "stages": row.stages_json or {},
        "counters": row.counters_json or {},
        "estimate_s": row.estimate_s,
        "started_at": _iso(row.started_at),
        "finished_at": _iso(row.finished_at),
        "error": row.error,
        "job_run_id": row.job_run_id,
        "is_public": bool(row.is_public),
        "active": row.status in ACTIVE_STATUSES,
    }


def create_run(kind: str, symbols, owner_uid: str | None, preset: str | None = None,
               config: dict | None = None, prediction_date=None, target_date=None,
               estimate_s: int | None = None, job_run_id: int | None = None,
               is_public: bool = True) -> str:
    """Insert a queued run and return its run_id (uuid4 string).

    ``is_public`` should match what the run's predictions and reports are
    stamped with (a private scheduled job writes private rows), so a run
    page never lists a run whose artifacts its viewer cannot see.
    """
    from db.models import AnalysisRun

    if kind not in KINDS:
        raise ValueError(f"unknown run kind {kind!r}")
    csv = _symbols_csv(symbols)
    if not csv:
        raise ValueError("a run needs at least one symbol")

    run_id = str(uuid.uuid4())
    with get_session() as session:
        session.add(AnalysisRun(
            run_id=run_id,
            owner_uid=owner_uid or None,
            kind=kind,
            status="queued",
            preset=preset,
            config_json=dict(config) if config else None,
            symbols_csv=csv,
            prediction_date=_coerce_date(prediction_date),
            target_date=_coerce_date(target_date),
            estimate_s=int(estimate_s) if estimate_s is not None else None,
            started_at=_now(),
            job_run_id=job_run_id,
            is_public=bool(is_public),
        ))
    logger.info("Run %s created: kind=%s owner=%s symbols=%s",
                run_id[:8], kind, owner_uid, csv)
    return run_id


def set_status(run_id: str, status: str, error: str | None = None) -> dict | None:
    """Move a run to ``status``; terminal statuses stamp finished_at.

    A row already in a terminal state is left alone (the run dict is still
    returned): a cancelled run stays cancelled when its subprocess finishes
    late, and a failed run keeps its error text.
    """
    from db.models import AnalysisRun

    if status not in STATUSES:
        raise ValueError(f"unknown run status {status!r}")
    with get_session() as session:
        row = session.execute(
            select(AnalysisRun).where(AnalysisRun.run_id == run_id).with_for_update()
        ).scalar_one_or_none()
        if row is None:
            return None
        if row.status in TERMINAL_STATUSES:
            return _to_dict(row)
        values: dict = {"status": status}
        if status in TERMINAL_STATUSES:
            values["finished_at"] = _now()
        if error is not None:
            values["error"] = str(error)
        session.execute(
            update(AnalysisRun).where(AnalysisRun.run_id == run_id).values(**values))
        session.flush()
        session.refresh(row)
        return _to_dict(row)


def first_line(text, limit: int = 200) -> str:
    """The first non-empty line of an error, cut to ``limit``: what a row,
    a pill or a stepper cell has room for."""
    lines = [ln.strip() for ln in str(text or "").splitlines() if ln.strip()]
    return (lines[0] if lines else "")[:limit]


def update_progress(run_id: str, stage: str, state: str | None = None,
                    done: int | None = None, total: int | None = None,
                    symbol: str | None = None, error=None,
                    **counters) -> dict | None:
    """Merge one stage's progress and any counters into the run row.

    Only the fields passed are touched: reporting ``done=3`` keeps the
    stage's earlier ``total`` and ``state``. Counters are totals, not
    increments; a caller reporting per-symbol work keeps its own running
    sum. The first call flips a queued run to running.

    With ``symbol`` the ``state`` is that symbol's, recorded under
    ``stages_json[stage]["symbols"][symbol]``; the stage's own state is
    left alone (one symbol failing does not fail the stage). ``error`` is
    kept as its first line under ``["errors"][symbol]`` (or the stage's
    ``"error"`` without a symbol), which is what the stepper prints.
    """
    from db.models import AnalysisRun

    if state is not None and state not in STAGE_STATES:
        raise ValueError(f"unknown stage state {state!r}")
    with get_session() as session:
        row = session.execute(
            select(AnalysisRun).where(AnalysisRun.run_id == run_id).with_for_update()
        ).scalar_one_or_none()
        if row is None:
            return None

        stages = dict(row.stages_json or {})
        entry = dict(stages.get(stage) or {})
        if symbol is not None:
            errors = dict(entry.get("errors") or {})
            if state is not None:
                symbols = dict(entry.get("symbols") or {})
                symbols[symbol] = state
                entry["symbols"] = symbols
                # A symbol that comes back from failed (a retry within the
                # stage) drops the reason it carried.
                if state != "failed":
                    errors.pop(symbol, None)
            if error is not None:
                errors[symbol] = first_line(error)
            if errors or "errors" in entry:
                entry["errors"] = errors
        else:
            if state is not None:
                entry["state"] = state
            if error is not None:
                entry["error"] = first_line(error)
        if done is not None:
            entry["done"] = int(done)
        if total is not None:
            entry["total"] = int(total)
        if "state" not in entry:
            entry["state"] = "running"
        stages[stage] = entry

        merged = dict(row.counters_json or {})
        merged.update({k: v for k, v in counters.items() if v is not None})

        values: dict = {"stages_json": stages, "counters_json": merged}
        if row.status == "queued":
            values["status"] = "running"
        session.execute(
            update(AnalysisRun).where(AnalysisRun.run_id == run_id).values(**values))
        session.flush()
        session.refresh(row)
        return _to_dict(row)


def get_run(run_id: str) -> dict | None:
    from db.models import AnalysisRun

    if not run_id:
        return None
    with get_session() as session:
        row = session.get(AnalysisRun, run_id)
        return _to_dict(row) if row is not None else None


def active_run_for(owner_uid: str | None) -> dict | None:
    """The newest queued/running MANUAL run for this owner; the per-owner
    lock behind the Run dialog's confirm.

    NULL owners compare as '' so anonymous sessions still lock against
    each other, the same rule watchlist_service uses. Scheduled runs are
    left out: their owner is the job's (None for the public watchlist
    job), so counting them would refuse every anonymous session for the
    whole daily run. They stay in active_runs(), which is what the pill
    reads.
    """
    from db.models import AnalysisRun

    with get_session() as session:
        row = session.execute(
            select(AnalysisRun)
            .where(func.coalesce(AnalysisRun.owner_uid, "") == (owner_uid or ""),
                   AnalysisRun.kind == "manual",
                   AnalysisRun.status.in_(ACTIVE_STATUSES))
            .order_by(AnalysisRun.started_at.desc())
            .limit(1)
        ).scalar_one_or_none()
        return _to_dict(row) if row is not None else None


def active_runs() -> list[dict]:
    """Every queued/running run, any owner, oldest first (the topbar pill)."""
    from db.models import AnalysisRun

    with get_session() as session:
        rows = session.execute(
            select(AnalysisRun)
            .where(AnalysisRun.status.in_(ACTIVE_STATUSES))
            .order_by(AnalysisRun.started_at.asc())
        ).scalars().all()
        return [_to_dict(r) for r in rows]


def list_runs(limit: int = 50, kind: str | None = None,
              owner_uid: str | None = None) -> list[dict]:
    """Newest first. ``owner_uid`` filters when given; None means any owner."""
    from db.models import AnalysisRun

    with get_session() as session:
        q = select(AnalysisRun).order_by(AnalysisRun.started_at.desc())
        if kind:
            q = q.where(AnalysisRun.kind == kind)
        if owner_uid is not None:
            q = q.where(func.coalesce(AnalysisRun.owner_uid, "") == owner_uid)
        rows = session.execute(q.limit(limit)).scalars().all()
        return [_to_dict(r) for r in rows]


def cancel_run(run_id: str) -> dict | None:
    """Mark the run cancelled and return it. Killing the worker is the
    caller's job (progress_service tracks the pid); this only records the
    decision, so a run that already finished keeps its real outcome."""
    return set_status(run_id, "cancelled")


def run_ceiling_s() -> int:
    """The scheduler's wall-clock ceiling, the longest any run is allowed to
    take before its process is killed; read lazily so this module stays
    importable without the scheduler's dependencies."""
    from services.scheduler_service import JOB_TIMEOUT_SECONDS
    return JOB_TIMEOUT_SECONDS


def _queued_stall_s() -> int:
    """How long a queued run may sit before any stage reports on it: the
    feed's own stall window, so both witnesses agree."""
    from services.progress_service import WATCHDOG_STALL_S
    return WATCHDOG_STALL_S


def _age_s(run: dict, now: datetime) -> float:
    started = run.get("started_at")
    started_dt = datetime.fromisoformat(started) if started else None
    if started_dt is not None and started_dt.tzinfo is None:
        started_dt = started_dt.replace(tzinfo=timezone.utc)
    return (now - started_dt).total_seconds() if started_dt else float("inf")


def _job_run_statuses(session, job_run_ids) -> dict[int, str]:
    from db.models import JobRun

    ids = sorted({int(i) for i in job_run_ids if i is not None})
    if not ids:
        return {}
    rows = session.execute(
        select(JobRun.id, JobRun.status).where(JobRun.id.in_(ids))).all()
    return {int(i): s for i, s in rows}


def _orphan_reason(run: dict, live, max_age_s, job_statuses: dict,
                   now: datetime, queued_stall_s=-1) -> str | None:
    """Why an active run is dead, or None while it may still be working.

    Three witnesses, in order of certainty. A run linked to a job_runs row
    is dead the moment run_job has finalized that row (clean exit that
    never closed, ceiling kill, crash): the runner process is gone whatever
    the feed says. A run the progress feed lists as in flight is otherwise
    trusted, the feed's own watchdog handles stalls and dead worker pids
    there, with one exception: a run still QUEUED past ``queued_stall_s``
    (the feed's stall window; None: never) was armed by the confirm and
    then no stage ever reported, the feed has nothing to watch, so the
    row is failed from here. A run absent from the feed with no live
    process to vouch for it is presumed dead once older than ``max_age_s``
    (0: at once, the server just restarted and nothing survives that;
    None: never by age).
    """
    if run.get("status") not in ACTIVE_STATUSES:
        return None
    job_run_id = run.get("job_run_id")
    if job_run_id is not None:
        job_status = job_statuses.get(int(job_run_id))
        if job_status is not None and job_status != JOB_RUN_LIVE_STATUS:
            return (f"scheduler job run {job_run_id} finished ({job_status}) "
                    f"before the run closed")
    if run.get("run_id") in set(live or ()):
        if run.get("status") != "queued" or queued_stall_s is None:
            return None
        if queued_stall_s == -1:
            queued_stall_s = _queued_stall_s()
        age_s = _age_s(run, now)
        if age_s >= queued_stall_s:
            return (f"no stage started on this run in "
                    f"{int(age_s // 60)} minutes, past the "
                    f"{int(queued_stall_s // 60)}-minute stall window")
        return None
    if max_age_s is None:
        return None
    age_s = _age_s(run, now)
    if age_s >= max_age_s:
        if max_age_s <= 0:
            return "no process is reporting on this run"
        return (f"no process reported on this run for "
                f"{int(age_s // 60)} minutes, past the "
                f"{int(max_age_s // 60)}-minute run ceiling")
    return None


def orphan_reason(run: dict, live=(), max_age_s: float | None = -1,
                  queued_stall_s: float | None = -1) -> str | None:
    """One run's verdict (see _orphan_reason); ``max_age_s`` defaults to the
    scheduler's ceiling and ``queued_stall_s`` to the feed's stall window.
    ``live`` is the progress feed's active list."""
    if max_age_s == -1:
        max_age_s = run_ceiling_s()
    with get_session() as session:
        statuses = _job_run_statuses(session, [run.get("job_run_id")])
    return _orphan_reason(run, live, max_age_s, statuses, _now(),
                          queued_stall_s=queued_stall_s)


def reap_orphans(live=(), max_age_s: float | None = -1,
                 error: str | None = None,
                 queued_stall_s: float | None = -1) -> list[dict]:
    """Fail every active run that is provably dead and return those runs.

    ``live`` is the progress feed's active list; ``max_age_s`` and
    ``queued_stall_s`` as in orphan_reason (defaults: the scheduler's
    ceiling and the feed's stall window; max_age_s=0 on a server restart
    fails everything not live). ``error`` overrides the per-run reason.
    Sticky statuses make this safe to repeat: a run that closed itself in
    the meantime keeps its own outcome.
    """
    from db.models import AnalysisRun

    if max_age_s == -1:
        max_age_s = run_ceiling_s()
    now = _now()
    with get_session() as session:
        rows = session.execute(
            select(AnalysisRun).where(AnalysisRun.status.in_(ACTIVE_STATUSES))
        ).scalars().all()
        runs = [_to_dict(r) for r in rows]
        statuses = _job_run_statuses(session, [r["job_run_id"] for r in runs])

    reaped: list[dict] = []
    for run in runs:
        reason = _orphan_reason(run, live, max_age_s, statuses, now,
                                queued_stall_s=queued_stall_s)
        if reason is None:
            continue
        closed = set_status(run["run_id"], "failed", error=error or reason)
        if closed is not None and closed["status"] == "failed":
            logger.warning("Run %s reaped: %s", run["run_id"][:8], reason)
            reaped.append(closed)
    return reaped


def fail_linked(job_run_id: int, error: str) -> list[dict]:
    """Fail the active runs linked to one job_runs row, once run_job has
    finalized it. Called from the scheduler AFTER (never inside) that
    finalize: a runner that closed its own row keeps that outcome."""
    from db.models import AnalysisRun

    with get_session() as session:
        ids = session.execute(
            select(AnalysisRun.run_id).where(
                AnalysisRun.job_run_id == int(job_run_id),
                AnalysisRun.status.in_(ACTIVE_STATUSES))
        ).scalars().all()
    closed = []
    for run_id in ids:
        row = set_status(run_id, "failed", error=error)
        if row is not None and row["status"] == "failed":
            closed.append(row)
    return closed

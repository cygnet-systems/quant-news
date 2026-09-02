"""The run row must be finalized no matter what the success stamp does.

The 2026-08-11 outage lived in one transaction: run_job finalizes the JobRun
row and stamps last_success_date in the same session, and the stamp read
job["timezone"] from a dict that never carried that key. The KeyError rolled
back the WHOLE finalize, the row stayed "running" forever, the day stayed
unstamped, the watchdog mailed "overdue" about a job that had succeeded, and
catch-up re-ran the full analysis every half hour all day (28 phantom rows,
each dying the same way).

These tests pin the two guarantees that prevent a repeat:

* the job dict run_job builds carries the timezone, and the stamp lands in
  the JOB's timezone (the original point of the change that broke this)
* a garbage timezone value degrades to a fallback date. It must never take
  the row bookkeeping down with it
"""

from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import sessionmaker

from db.models import Base, JobRun, ScheduledJob
from services import scheduler_service as ss


@compiles(JSONB, "sqlite")
def _jsonb_as_json(type_, compiler, **kw):
    # Test-only: lets the Postgres models build on in-memory SQLite.
    return "JSON"


# 00:30 UTC is the case the timezone stamp exists for: still 20:30 the
# previous evening in New York, so a correct stamp and a UTC stamp disagree.
FROZEN_NOW = datetime(2026, 8, 12, 0, 30, tzinfo=timezone.utc)


class _FrozenDatetime(datetime):
    @classmethod
    def now(cls, tz=None):
        return FROZEN_NOW.astimezone(tz) if tz else FROZEN_NOW


class _NoLock:
    def __init__(self, key):
        pass

    def acquire(self):
        return True

    def release(self):
        pass


@pytest.fixture
def db(monkeypatch):
    import db.session as dbs

    eng = create_engine("sqlite://")
    Base.metadata.create_all(
        eng, tables=[ScheduledJob.__table__, JobRun.__table__])
    monkeypatch.setattr(dbs, "_engine", eng)
    monkeypatch.setattr(
        dbs, "_SessionLocal", sessionmaker(bind=eng, expire_on_commit=False))
    return dbs


@pytest.fixture
def quiet_run(monkeypatch):
    """run_job with the world outside the database stubbed out."""
    import services.progress_service as prog

    class _Proc:
        returncode = 0
        stdout = "{}\n"
        stderr = ""

    monkeypatch.setattr(ss, "_AdvisoryLock", _NoLock)
    monkeypatch.setattr(ss.subprocess, "run", lambda *a, **kw: _Proc())
    monkeypatch.setattr(ss, "_notify", lambda *a, **kw: None)
    monkeypatch.setattr(prog, "emit", lambda *a, **kw: None)
    monkeypatch.setattr(ss, "datetime", _FrozenDatetime)


def _seed_job(db, timezone_name):
    with db.get_session() as session:
        session.add(ScheduledJob(
            id="daily_analysis", kind="analysis", enabled=True,
            hour=7, minute=0, days_of_week="mon-fri",
            timezone=timezone_name, symbols_csv="AAPL",
            is_public=True,
        ))


def _run_rows(db):
    with db.get_session() as session:
        return session.execute(select(JobRun)).scalars().all()


def test_success_stamps_the_date_in_the_jobs_timezone(db, quiet_run):
    _seed_job(db, "US/Eastern")

    result = ss.run_job("daily_analysis", trigger="schedule")

    assert result["status"] == "success"
    (run,) = _run_rows(db)
    assert run.status == "success"
    assert run.finished_at is not None
    with db.get_session() as session:
        job = session.get(ScheduledJob, "daily_analysis")
        # 00:30 UTC on the 12th is still the 11th in New York.
        assert job.last_success_date == "2026-08-11"
        assert job.last_status == "success"


def test_bad_timezone_cannot_orphan_the_run_row(db, quiet_run):
    _seed_job(db, "Not/AZone")

    result = ss.run_job("daily_analysis", trigger="schedule")

    assert result["status"] == "success"
    (run,) = _run_rows(db)
    assert run.status == "success", (
        "a timezone problem rolled back the run finalize. This is the "
        "2026-08-11 phantom-'running' bug again")
    assert run.finished_at is not None
    with db.get_session() as session:
        job = session.get(ScheduledJob, "daily_analysis")
        # Fallback stamp (UTC date) is acceptable; a missing stamp is not,
        # because catch-up would re-run the job all day.
        assert job.last_success_date == "2026-08-12"

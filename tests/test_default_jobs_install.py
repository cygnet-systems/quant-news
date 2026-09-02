"""How a default job reaches a database that already has a schedule.

seed_default_jobs only writes into an empty scheduled_jobs table, so a kind
added later (ticker_refresh, 015/016) must be installed once by a data
migration. The rules pinned here: the migration inserts on a populated
table without the kind, matches the DEFAULT_JOBS spec exactly, leaves an
empty table to the seed (else the daily jobs would never be seeded), and
never resurrects a job that ran and was deleted.
"""

import importlib.util
from pathlib import Path

import pytest
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, select
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import sessionmaker

from db.models import Base, JobRun, ScheduledJob
from services import scheduler_service as ss

MIGRATION = (Path(__file__).resolve().parents[1]
             / "db" / "migrations" / "versions" / "016_ticker_refresh_job.py")


@compiles(JSONB, "sqlite")
def _jsonb_as_json(type_, compiler, **kw):
    # Test-only: lets the Postgres models build on in-memory SQLite.
    return "JSON"


def _load_migration():
    spec = importlib.util.spec_from_file_location("migration_016", MIGRATION)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def db(monkeypatch):
    import db.session as dbs

    eng = create_engine("sqlite://")
    Base.metadata.create_all(eng, tables=[ScheduledJob.__table__, JobRun.__table__])
    monkeypatch.setattr(dbs, "_engine", eng)
    monkeypatch.setattr(
        dbs, "_SessionLocal", sessionmaker(bind=eng, expire_on_commit=False))
    return dbs


def _run(db, direction="upgrade"):
    module = _load_migration()
    with db._engine.begin() as conn:
        ctx = MigrationContext.configure(conn)
        with Operations.context(ctx):
            getattr(module, direction)()


def _jobs(db):
    with db.get_session() as session:
        return {row.id: row for row in session.execute(select(ScheduledJob)).scalars()}


def _add_daily(db):
    with db.get_session() as session:
        session.add(ScheduledJob(id="daily_analysis", kind="analysis",
                                 hour=7, minute=0, symbols_csv="AAPL",
                                 params_json={}))


WEEKLY = next(j for j in ss.DEFAULT_JOBS if j["id"] == ss.TICKER_REFRESH_JOB)


class TestMigration016:
    def test_installs_the_weekly_job_on_a_populated_schedule(self, db):
        _add_daily(db)
        _run(db)
        jobs = _jobs(db)
        assert set(jobs) == {"daily_analysis", ss.TICKER_REFRESH_JOB}
        row = jobs[ss.TICKER_REFRESH_JOB]
        # The migration carries literals; they must not drift from the seed.
        for key in ("kind", "description", "hour", "minute", "days_of_week",
                    "symbols_csv", "params_json"):
            assert getattr(row, key) == WEEKLY[key], key
        assert row.enabled is True and row.is_public is True
        assert row.owner_uid is None
        assert ss._weekday_set(row.days_of_week) == {"sun"}

    def test_leaves_an_empty_schedule_to_the_seed(self, db):
        _run(db)
        assert _jobs(db) == {}
        ss.seed_default_jobs()
        assert set(_jobs(db)) == {j["id"] for j in ss.DEFAULT_JOBS}

    def test_is_idempotent(self, db):
        _add_daily(db)
        _run(db)
        _run(db)
        assert len(_jobs(db)) == 2

    def test_respects_a_job_of_the_kind_under_another_id(self, db):
        _add_daily(db)
        with db.get_session() as session:
            session.add(ScheduledJob(id="my_refresh", kind=ss.TICKER_REFRESH_JOB,
                                     hour=5, minute=30, days_of_week="sat",
                                     params_json={}))
        _run(db)
        assert set(_jobs(db)) == {"daily_analysis", "my_refresh"}

    def test_never_resurrects_a_deleted_job_that_ran(self, db):
        _add_daily(db)
        with db.get_session() as session:
            session.add(JobRun(job_id=ss.TICKER_REFRESH_JOB, trigger="schedule",
                               status="success"))
        _run(db)
        assert set(_jobs(db)) == {"daily_analysis"}

    def test_downgrade_removes_only_the_installed_row(self, db):
        _add_daily(db)
        _run(db)
        _run(db, "downgrade")
        assert set(_jobs(db)) == {"daily_analysis"}


class TestSeedRuleUnchanged:
    def test_seed_still_never_touches_a_populated_schedule(self, db):
        _add_daily(db)
        ss.seed_default_jobs()
        assert set(_jobs(db)) == {"daily_analysis"}

    def test_weekly_spec_is_in_the_defaults_for_a_fresh_database(self):
        assert WEEKLY["kind"] == ss.TICKER_REFRESH_JOB
        assert ss.JOB_TYPES[WEEKLY["kind"]].needs_symbols is False

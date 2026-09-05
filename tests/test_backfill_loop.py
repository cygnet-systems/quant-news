"""The backfill sweep must not be able to re-run a session for ever.

2026-09-02..04: `daily_analysis` ran 109 times on `backfill`, every 30
minutes, ~190s and a full LLM bill each, all on the same target session that
was already stored. Two correct rules made a loop when they met:

* `store_prediction` preserves an existing row's owner across a rerun -- a
  rerun must not change who owns a call;
* the sweep's "was this session analysed?" ledger counts only rows owned by
  the job.

A session whose rows were written before the job had an owner keeps
`owner_uid IS NULL` for ever, so the ledger read zero, ran the job, the
rerun preserved the NULL, and the ledger read zero again.

Two guarantees are pinned here: the ledger counts a legacy NULL-owner row as
the job's (the same rule cache_service._visible and _by_run_kind use), and --
whatever any future ledger says -- one process never spends more than one
backfill run on the same date.
"""

from datetime import date, timedelta
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import sessionmaker

from db.models import Base, ModelPrediction
from services import scheduler_service as ss


@compiles(JSONB, "sqlite")
def _jsonb_as_json(type_, compiler, **kw):
    return "JSON"


JOB = {
    "id": "daily_analysis", "kind": "analysis", "enabled": True,
    "symbols_csv": "AAPL,MSFT", "owner_uid": "UJ74593",
    "hour": 7, "minute": 0,
}


@pytest.fixture
def db(monkeypatch):
    import db.session as dbs

    eng = create_engine("sqlite://")
    Base.metadata.create_all(eng, tables=[ModelPrediction.__table__])
    monkeypatch.setattr(dbs, "_engine", eng)
    monkeypatch.setattr(
        dbs, "_SessionLocal", sessionmaker(bind=eng, expire_on_commit=False))
    return dbs


@pytest.fixture(autouse=True)
def _clear_attempts():
    ss._BACKFILL_ATTEMPTED.clear()
    yield
    ss._BACKFILL_ATTEMPTED.clear()


def _last_session() -> date:
    from utils.trading_calendar import is_trading_day

    d = date.today() - timedelta(days=1)
    while not is_trading_day(d):
        d -= timedelta(days=1)
    return d


def _store(db, target: date, owner):
    with db.get_session() as s:
        for i, sym in enumerate(("AAPL", "MSFT")):
            s.add(ModelPrediction(
                id=f"{sym}_m_{target:%Y%m%d}_{i}", symbol=sym,
                model_name="kronos_mini", prediction_date=target,
                target_date=target, decision="BUY", owner_uid=owner,
                is_public=True))


def _sweep(runs):
    """Run the sweep with run_job recorded rather than executed."""
    def _run_job(job_id, trigger=None, overrides=None):
        runs.append((job_id, (overrides or {}).get("target")))
        return {"status": "partial"}

    with patch.object(ss, "list_jobs", return_value=[dict(JOB)]), \
         patch.object(ss, "run_job", side_effect=_run_job):
        ss._backfill_missed_sessions()


class TestLegacyOwnerRowsCount:
    def test_a_null_owner_session_is_not_re_analysed(self, db):
        """The exact 2026-09-02 shape: rows on disk, owner NULL, job owned."""
        target = _last_session()
        _store(db, target, None)
        runs = []
        _sweep(runs)
        assert not [r for r in runs if r[1] == target.isoformat()], (
            "a session already on disk under a legacy NULL owner was queued "
            "for backfill: this is the 109-run loop")

    def test_a_session_owned_by_the_job_is_not_re_analysed(self, db):
        target = _last_session()
        _store(db, target, JOB["owner_uid"])
        runs = []
        _sweep(runs)
        assert not [r for r in runs if r[1] == target.isoformat()]

    def test_a_genuinely_missing_session_is_still_run(self, db):
        target = _last_session()
        runs = []
        _sweep(runs)
        assert (JOB["id"], target.isoformat()) in runs, (
            "the sweep stopped filling real gaps")


class TestOneAttemptPerDate:
    def test_a_date_is_never_run_twice_by_one_process(self, db):
        """Defence in depth: even with a ledger that never registers the
        write, the second sweep must not spend another run."""
        first, second = [], []
        _sweep(first)
        _sweep(second)
        target = _last_session().isoformat()
        assert (JOB["id"], target) in first
        assert (JOB["id"], target) not in second, (
            "the sweep re-ran a date it had already spent a run on; the "
            "circuit breaker is gone and the loop can come back")

"""Guards for the run record behind the Run dialog, the pill and the run page.

The rules worth pinning: progress reports merge field by field (a stage that
reports done=3 keeps the total it reported earlier), the first report flips
queued to running, terminal statuses are sticky (a subprocess finishing after
a cancel must not resurrect the run), and the per-owner lock sees only that
owner's queued/running rows, with anonymous sessions locking against each
other.
"""

from datetime import timedelta

import pytest
from sqlalchemy import create_engine, update
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import sessionmaker

from db.models import AnalysisRun, Base, JobRun
from services import run_service as rs


@compiles(JSONB, "sqlite")
def _jsonb_as_json(type_, compiler, **kw):
    # Test-only: lets the Postgres models build on in-memory SQLite.
    return "JSON"


@pytest.fixture
def db(monkeypatch):
    import db.session as dbs

    eng = create_engine("sqlite://")
    Base.metadata.create_all(eng, tables=[AnalysisRun.__table__, JobRun.__table__])
    monkeypatch.setattr(dbs, "_engine", eng)
    monkeypatch.setattr(
        dbs, "_SessionLocal", sessionmaker(bind=eng, expire_on_commit=False))
    return dbs


class TestCreate:
    def test_new_run_is_queued_with_normalised_symbols(self, db):
        run_id = rs.create_run("manual", [" nvda", "AMD", "nvda", ""], "u1",
                               preset="standard", config={"deep": False},
                               prediction_date="2026-09-02", estimate_s=240)
        run = rs.get_run(run_id)
        assert len(run_id) == 36
        assert run["status"] == "queued"
        assert run["symbols"] == ["NVDA", "AMD"]
        assert run["symbols_csv"] == "NVDA,AMD"
        assert run["prediction_date"] == "2026-09-02"
        assert run["config"] == {"deep": False}
        assert run["estimate_s"] == 240
        assert run["stages"] == {} and run["counters"] == {}
        assert run["finished_at"] is None
        assert run["active"] is True

    def test_scheduled_run_keeps_its_job_run_link(self, db):
        run_id = rs.create_run("scheduled", "AAPL,MSFT", None, job_run_id=7)
        run = rs.get_run(run_id)
        assert run["kind"] == "scheduled"
        assert run["job_run_id"] == 7
        assert run["owner_uid"] is None

    def test_rejects_unknown_kind_and_empty_symbols(self, db):
        with pytest.raises(ValueError):
            rs.create_run("adhoc", ["AAPL"], "u1")
        with pytest.raises(ValueError):
            rs.create_run("manual", ["", " "], "u1")

    def test_missing_run_reads_as_none(self, db):
        assert rs.get_run("nope") is None
        assert rs.get_run("") is None
        assert rs.set_status("nope", "done") is None
        assert rs.update_progress("nope", "models", done=1) is None


class TestProgress:
    def test_first_report_flips_queued_to_running(self, db):
        run_id = rs.create_run("manual", ["AAPL"], "u1")
        run = rs.update_progress(run_id, "news", state="running", total=1)
        assert run["status"] == "running"
        assert run["stages"] == {"news": {"state": "running", "total": 1}}

    def test_fields_merge_rather_than_replace(self, db):
        """done=1 must not erase the total reported when the stage opened."""
        run_id = rs.create_run("manual", ["AAPL", "MSFT"], "u1")
        rs.update_progress(run_id, "models", state="running", total=2)
        rs.update_progress(run_id, "models", done=1)
        run = rs.update_progress(run_id, "models", done=2, state="done")
        assert run["stages"]["models"] == {"state": "done", "done": 2, "total": 2}

    def test_stages_accumulate_independently(self, db):
        run_id = rs.create_run("manual", ["AAPL"], "u1")
        rs.update_progress(run_id, "news", state="done", done=1, total=1)
        run = rs.update_progress(run_id, "models", state="running", total=1)
        assert set(run["stages"]) == {"news", "models"}
        assert run["stages"]["news"]["state"] == "done"

    def test_stage_without_state_defaults_to_running(self, db):
        run_id = rs.create_run("manual", ["AAPL"], "u1")
        run = rs.update_progress(run_id, "research", done=0, total=3)
        assert run["stages"]["research"]["state"] == "running"

    def test_counters_are_totals_merged_by_key(self, db):
        run_id = rs.create_run("manual", ["AAPL"], "u1")
        rs.update_progress(run_id, "news", articles=12)
        rs.update_progress(run_id, "models", predictions=4, articles=None)
        run = rs.update_progress(run_id, "models", predictions=8)
        assert run["counters"] == {"articles": 12, "predictions": 8}

    def test_progress_never_reopens_a_terminal_run(self, db):
        run_id = rs.create_run("manual", ["AAPL"], "u1")
        rs.cancel_run(run_id)
        run = rs.update_progress(run_id, "models", done=1, total=1)
        assert run["status"] == "cancelled"
        assert run["stages"]["models"]["done"] == 1

    def test_rejects_unknown_stage_state(self, db):
        run_id = rs.create_run("manual", ["AAPL"], "u1")
        with pytest.raises(ValueError):
            rs.update_progress(run_id, "models", state="exploded")


class TestStatus:
    def test_done_stamps_finished_at(self, db):
        run_id = rs.create_run("manual", ["AAPL"], "u1")
        rs.set_status(run_id, "running")
        assert rs.get_run(run_id)["finished_at"] is None
        run = rs.set_status(run_id, "done")
        assert run["status"] == "done"
        assert run["finished_at"] is not None
        assert run["active"] is False

    def test_failed_keeps_the_error(self, db):
        run_id = rs.create_run("manual", ["AAPL"], "u1")
        run = rs.set_status(run_id, "failed", error="watchdog: no progress for 20 min")
        assert run["status"] == "failed"
        assert run["error"].startswith("watchdog")
        assert run["finished_at"] is not None

    def test_terminal_status_is_sticky(self, db):
        """The cancel race: the worker's late 'done' must not win."""
        run_id = rs.create_run("manual", ["AAPL"], "u1")
        rs.update_progress(run_id, "models", total=1)
        cancelled = rs.cancel_run(run_id)
        assert cancelled["status"] == "cancelled"
        finished_at = cancelled["finished_at"]
        run = rs.set_status(run_id, "done")
        assert run["status"] == "cancelled"
        assert run["finished_at"] == finished_at
        assert rs.set_status(run_id, "failed", error="late")["error"] is None

    def test_rejects_unknown_status(self, db):
        run_id = rs.create_run("manual", ["AAPL"], "u1")
        with pytest.raises(ValueError):
            rs.set_status(run_id, "paused")


class TestActiveRuns:
    def test_lock_sees_only_this_owners_live_run(self, db):
        mine = rs.create_run("manual", ["AAPL"], "u1")
        rs.create_run("manual", ["MSFT"], "u2")
        finished = rs.create_run("manual", ["TSLA"], "u1")
        rs.set_status(finished, "done")
        assert rs.active_run_for("u1")["run_id"] == mine
        assert rs.active_run_for("u3") is None
        rs.set_status(mine, "failed", error="boom")
        assert rs.active_run_for("u1") is None

    def test_newest_live_run_wins(self, db):
        older = rs.create_run("manual", ["AAPL"], "u1")
        newer = rs.create_run("manual", ["MSFT"], "u1")
        assert rs.active_run_for("u1")["run_id"] == newer
        rs.cancel_run(newer)
        assert rs.active_run_for("u1")["run_id"] == older

    def test_anonymous_sessions_lock_against_each_other(self, db):
        run_id = rs.create_run("manual", ["AAPL"], None)
        assert rs.active_run_for(None)["run_id"] == run_id
        assert rs.active_run_for("")["run_id"] == run_id
        assert rs.active_run_for("u1") is None

    def test_pill_sees_every_owner_oldest_first(self, db):
        a = rs.create_run("scheduled", ["AAPL"], None, job_run_id=1)
        b = rs.create_run("manual", ["MSFT"], "u2")
        done = rs.create_run("manual", ["TSLA"], "u1")
        rs.set_status(done, "done")
        assert [r["run_id"] for r in rs.active_runs()] == [a, b]

    def test_scheduled_runs_never_lock_but_stay_on_the_pill(self, db):
        # The daily job's row is owner None; if it counted as the lock,
        # every anonymous session would be refused for the whole run.
        sched = rs.create_run("scheduled", ["AAPL"], None, job_run_id=1)
        mine = rs.create_run("scheduled", ["AAPL"], "u1", job_run_id=2)
        assert rs.active_run_for(None) is None
        assert rs.active_run_for("u1") is None
        assert [r["run_id"] for r in rs.active_runs()] == [sched, mine]
        manual = rs.create_run("manual", ["MSFT"], None)
        assert rs.active_run_for(None)["run_id"] == manual

    def test_visibility_is_recorded(self, db):
        private = rs.create_run("scheduled", ["AAPL"], "u1", is_public=False)
        default = rs.create_run("manual", ["AAPL"], "u1")
        assert rs.get_run(private)["is_public"] is False
        assert rs.get_run(default)["is_public"] is True


class TestListRuns:
    def test_newest_first_with_filters(self, db):
        s = rs.create_run("scheduled", ["AAPL"], None, job_run_id=1)
        m1 = rs.create_run("manual", ["MSFT"], "u1")
        m2 = rs.create_run("manual", ["TSLA"], "u2")
        assert [r["run_id"] for r in rs.list_runs()] == [m2, m1, s]
        assert [r["run_id"] for r in rs.list_runs(kind="scheduled")] == [s]
        assert [r["run_id"] for r in rs.list_runs(owner_uid="u1")] == [m1]
        assert [r["run_id"] for r in rs.list_runs(limit=1)] == [m2]


def _age(run_id, seconds):
    """Back-date a run so the age rule can be exercised."""
    with rs.get_session() as session:
        session.execute(update(AnalysisRun).where(AnalysisRun.run_id == run_id)
                        .values(started_at=rs._now() - timedelta(seconds=seconds)))


def _job_run(status="running"):
    with rs.get_session() as session:
        job = JobRun(job_id="daily_analysis", trigger="schedule", status=status)
        session.add(job)
        session.flush()
        return job.id


class TestOrphans:
    """An active row nobody is working on is the owner's lock; the reaper
    must fail exactly those and leave live runs alone."""

    def test_live_unlinked_running_run_is_never_an_orphan(self, db):
        run_id = rs.create_run("manual", ["NVDA"], "u1")
        rs.update_progress(run_id, "models", done=1, total=3)
        _age(run_id, 10 * 3600)
        run = rs.get_run(run_id)
        assert rs.orphan_reason(run, live=[run_id], max_age_s=60) is None
        assert rs.reap_orphans(live=[run_id], max_age_s=60) == []
        assert rs.get_run(run_id)["status"] == "running"

    def test_live_queued_run_ages_out_on_the_stall_window(self, db, monkeypatch):
        # Armed by the confirm (so the feed lists it) but no stage ever
        # reported: nothing on the feed side can age it, so the row's own
        # clock decides, on the feed's stall window, live list or not.
        from datetime import timedelta
        run_id = rs.create_run("manual", ["NVDA"], "u1")
        run = rs.get_run(run_id)
        t0 = rs._now()
        stall = 15 * 60
        assert rs._orphan_reason(run, [run_id], None, {},
                                 t0 + timedelta(seconds=stall - 1),
                                 queued_stall_s=stall) is None
        reason = rs._orphan_reason(run, [run_id], None, {},
                                   t0 + timedelta(seconds=stall + 1),
                                   queued_stall_s=stall)
        assert reason and "no stage started" in reason and "15-minute" in reason
        # None disables it; -1 reads the feed's constant.
        assert rs._orphan_reason(run, [run_id], None, {},
                                 t0 + timedelta(days=1), queued_stall_s=None) is None
        monkeypatch.setattr("services.progress_service.WATCHDOG_STALL_S", 30)
        assert rs._orphan_reason(run, [run_id], None, {},
                                 t0 + timedelta(seconds=31)) is not None

        _age(run_id, 3600)
        assert rs.reap_orphans(live=[run_id], max_age_s=None,
                               queued_stall_s=None) == []
        reaped = rs.reap_orphans(live=[run_id], max_age_s=None)
        assert [r["run_id"] for r in reaped] == [run_id]
        assert rs.get_run(run_id)["status"] == "failed"
        assert rs.active_run_for("u1") is None

    def test_absent_run_is_an_orphan_only_past_the_age(self, db):
        young = rs.create_run("manual", ["NVDA"], "u1")
        old = rs.create_run("scheduled", ["AMD"], None)
        _age(old, 2 * 3600)

        reaped = rs.reap_orphans(live=[], max_age_s=3600)

        assert [r["run_id"] for r in reaped] == [old]
        assert rs.get_run(old)["status"] == "failed"
        assert "ceiling" in rs.get_run(old)["error"]
        assert rs.get_run(old)["finished_at"] is not None
        assert rs.get_run(young)["status"] == "queued"
        assert rs.active_run_for(None) is None
        assert rs.active_run_for("u1")["run_id"] == young

    def test_age_zero_reaps_everything_not_live(self, db):
        a = rs.create_run("manual", ["NVDA"], "u1")
        b = rs.create_run("scheduled", ["AMD"], None)
        reaped = rs.reap_orphans(live=[b], max_age_s=0,
                                 error=rs.ORPHAN_RESTART_ERROR)
        assert [r["run_id"] for r in reaped] == [a]
        assert rs.get_run(a)["error"] == rs.ORPHAN_RESTART_ERROR
        assert rs.get_run(b)["status"] == "queued"

    def test_age_none_never_reaps_by_age(self, db):
        run_id = rs.create_run("manual", ["NVDA"], "u1")
        _age(run_id, 10 * 3600)
        assert rs.reap_orphans(live=[], max_age_s=None) == []

    def test_linked_run_dies_with_its_job_run_even_while_live(self, db):
        finished = _job_run("error")
        running = _job_run("running")
        dead = rs.create_run("scheduled", ["NVDA"], None, job_run_id=finished)
        alive = rs.create_run("scheduled", ["AMD"], None, job_run_id=running)
        rs.update_progress(dead, "models", done=1, total=3)

        assert "finished (error)" in rs.orphan_reason(
            rs.get_run(dead), live=[dead, alive], max_age_s=None)
        assert rs.orphan_reason(rs.get_run(alive), live=[], max_age_s=None) is None

        reaped = rs.reap_orphans(live=[dead, alive], max_age_s=None)
        assert [r["run_id"] for r in reaped] == [dead]
        row = rs.get_run(dead)
        assert row["status"] == "failed"
        assert row["stages"]["models"] == {"state": "running", "done": 1, "total": 3}
        assert rs.get_run(alive)["status"] == "queued"

    def test_reap_respects_sticky_terminal_rows(self, db):
        run_id = rs.create_run("manual", ["NVDA"], "u1")
        rs.set_status(run_id, "done")
        _age(run_id, 10 * 3600)
        assert rs.reap_orphans(live=[], max_age_s=0) == []
        assert rs.get_run(run_id)["status"] == "done"

    def test_fail_linked_closes_only_that_job_runs_active_rows(self, db):
        job = _job_run("error")
        other = _job_run("running")
        closed_itself = rs.create_run("scheduled", ["NVDA"], None, job_run_id=job)
        rs.set_status(closed_itself, "done")
        killed = rs.create_run("scheduled", ["AMD"], None, job_run_id=job)
        unrelated = rs.create_run("scheduled", ["TSLA"], None, job_run_id=other)

        closed = rs.fail_linked(job, "TIMED OUT")

        assert [r["run_id"] for r in closed] == [killed]
        assert rs.get_run(killed)["error"] == "TIMED OUT"
        assert rs.get_run(closed_itself)["status"] == "done"
        assert rs.get_run(unrelated)["status"] == "queued"

    def test_default_age_is_the_scheduler_ceiling(self, db, monkeypatch):
        from services import scheduler_service as ss
        monkeypatch.setattr(ss, "JOB_TIMEOUT_SECONDS", 600)
        run_id = rs.create_run("manual", ["NVDA"], "u1")
        _age(run_id, 601)
        assert rs.reap_orphans(live=[]) and rs.get_run(run_id)["status"] == "failed"

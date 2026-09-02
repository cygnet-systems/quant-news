"""The live feed is per run.

Two runs in flight at once must not share a feed, close each other, or
stamp each other's rows. These tests pin: events land under their own run
and nowhere else, get_feed() without a run id follows the newest run in
flight, finish_run() retires only its own run (and keeps the legacy active
flag in step), current_run_id() never reads another process's run out of
the shared cache, and emit_progress() merges counters into the run row.
"""

import asyncio
import os
import threading

import diskcache
import pytest
from sqlalchemy import create_engine
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import sessionmaker

from db.models import AnalysisRun, Base, JobRun, TradingAgentReport
from services import progress_service as prog
from services import run_service as rs


@compiles(JSONB, "sqlite")
def _jsonb_as_json(type_, compiler, **kw):
    # Test-only: lets the Postgres models build on in-memory SQLite.
    return "JSON"


@pytest.fixture
def db(monkeypatch):
    import db.session as dbs

    eng = create_engine("sqlite://")
    Base.metadata.create_all(eng, tables=[AnalysisRun.__table__, JobRun.__table__,
                                          TradingAgentReport.__table__])
    monkeypatch.setattr(dbs, "_engine", eng)
    monkeypatch.setattr(
        dbs, "_SessionLocal", sessionmaker(bind=eng, expire_on_commit=False))
    return dbs


@pytest.fixture
def feed(monkeypatch, tmp_path):
    """A publisher process with its own empty cache and no audit trail."""
    monkeypatch.setenv(prog._ENV_FLAG, "1")
    monkeypatch.setattr(prog, "_cache", diskcache.Cache(str(tmp_path)))
    monkeypatch.setattr(prog, "_local_run", {"id": None, "title": None})
    monkeypatch.setattr(prog, "_write_audit", lambda *a, **kw: None)
    # The watchdog's row reap is throttled on a module clock; every test
    # starts with it due.
    monkeypatch.setattr(prog, "_last_db_reap", 0.0)
    # The task-scoped identity is a ContextVar monkeypatch cannot undo.
    token = prog._run_ctx.set(None)
    yield prog
    prog._run_ctx.reset(token)


def _messages(events):
    return [e["message"] for e in events]


class TestIsolation:
    def test_events_stay_with_their_run(self, feed):
        a = feed.start_run("run A", run_id="run-a")
        b = feed.start_run("run B", run_id="run-b")
        feed.emit("news", "a1", run_id=a)
        feed.emit("news", "b1", run_id=b)
        feed.emit("news", "a2", run_id=a)

        assert _messages(feed.get_feed(a)["events"]) == ["run A", "a1", "a2"]
        assert _messages(feed.get_feed(b)["events"]) == ["run B", "b1"]
        assert feed.get_feed(a)["active"] and feed.get_feed(b)["active"]

    def test_unnamed_feed_follows_the_newest_active_run(self, feed):
        feed.start_run("run A", run_id="run-a")
        feed.start_run("run B", run_id="run-b")
        assert feed.get_feed()["run_id"] == "run-b"

        feed.finish_run("B done", run_id="run-b")
        picked = feed.get_feed()
        assert picked["run_id"] == "run-a"
        assert picked["active"] is True
        assert _messages(feed.get_feed("run-b")["events"])[-1] == "B done"

    def test_finish_retires_only_its_own_run(self, feed):
        feed.start_run("run A", run_id="run-a")
        feed.start_run("run B", run_id="run-b")
        feed.finish_run("A done", run_id="run-a")

        assert feed._active_runs() == ["run-b"]
        assert feed._get_cache().get("active") is True
        feed.finish_run("B done", run_id="run-b")
        assert feed._active_runs() == []
        assert feed._get_cache().get("active") is False

    def test_idle_feed_is_the_rolling_log_across_runs(self, feed):
        feed.start_run("run A", run_id="run-a")
        feed.finish_run("A done", run_id="run-a")
        feed.emit("action", "symbol search")
        idle = feed.get_feed()
        assert idle["active"] is False
        assert _messages(idle["events"]) == ["run A", "A done", "symbol search"]
        # The ad-hoc line never lands in a run's own feed.
        assert _messages(feed.get_feed("run-a")["events"]) == ["run A", "A done"]

    def test_mark_run_pending_opens_the_run_ahead_of_start(self, feed):
        feed.mark_run_pending("run-a")
        assert feed._active_runs() == ["run-a"]
        assert feed.get_feed()["run_id"] == "run-a"
        feed.start_run("run A", run_id="run-a")
        assert feed._active_runs() == ["run-a"]

    def test_mark_run_pending_without_id_arms_the_legacy_flag(self, feed):
        feed.mark_run_pending()
        assert feed.get_feed()["active"] is True
        assert feed._active_runs() == []

    def test_start_run_mints_an_id_for_headless_callers(self, feed):
        run_id = feed.start_run("bench")
        assert run_id and len(run_id) == 36
        assert feed._active_runs() == [run_id]


class TestRunIdentity:
    def test_current_run_id_is_this_process_only(self, feed):
        # Another process opened a run; this one never started anything.
        feed._get_cache().set("active_runs", ["someone-elses"])
        feed._get_cache().set("run_id", "someone-elses")
        assert feed.current_run_id() == "adhoc"

    def test_current_run_id_follows_start_and_finish(self, feed):
        feed.start_run("mine", run_id="mine")
        assert feed.current_run_id() == "mine"
        feed.finish_run("done", run_id="mine")
        assert feed.current_run_id() == "adhoc"

    def test_unnamed_emit_prefers_own_run_then_newest_active(self, feed):
        feed._get_cache().set("active_runs", ["other"])
        feed.emit("models", "from the subprocess")
        assert _messages(feed.get_feed("other")["events"]) == ["from the subprocess"]

        feed.start_run("mine", run_id="mine")
        feed.emit("models", "mine too")
        assert "mine too" in _messages(feed.get_feed("mine")["events"])
        assert "mine too" not in _messages(feed.get_feed("other")["events"])

    def test_server_closing_a_subprocess_run_finds_it(self, feed):
        # The models-only flow: this process's own run finished earlier, the
        # subprocess opened the current one, and this process closes it.
        feed.start_run("earlier", run_id="earlier")
        feed.finish_run("done", run_id="earlier")
        feed._get_cache().set("active_runs", ["sub"])
        feed._get_cache().set("active", True)
        feed.finish_run("Predictions complete")
        assert feed._active_runs() == []
        assert _messages(feed.get_feed("sub")["events"]) == ["Predictions complete"]


class TestRunContext:
    """The server runs every user's report stage in one process, each in
    its own asyncio task; research code under it emits unnamed and stamps
    spend through current_run_id(). Both must follow the task's run."""

    def test_concurrent_report_stages_keep_their_own_run(self, feed, db):
        from sqlalchemy import select
        from services.cache_service import get_cache

        async def stage(run_id):
            feed.start_run(f"Report {run_id}", run_id=run_id)
            # Let the other stage open its run before this one emits, so a
            # process-wide "last started" would point at the wrong run.
            await asyncio.sleep(0.01)
            feed.emit("ta", f"research step from {run_id}")
            # The research model runs in a worker via to_thread.
            await asyncio.to_thread(feed.emit, "ta", f"thread step from {run_id}")
            # The report it writes is stamped with the id it was handed
            # (the stage passes run_id explicitly), and with the run's
            # trade date as a string, the way the stage has it. (On the
            # task's own thread: the in-memory SQLite engine is per-thread.)
            get_cache().save_trading_agent_report(
                f"SYM{run_id}", "2026-09-03", "BUY", 0.6, "text",
                model_name="m", run_id=run_id)
            return feed.current_run_id()

        async def main():
            return await asyncio.gather(stage("A"), stage("B"))

        assert asyncio.run(main()) == ["A", "B"]
        assert _messages(feed.get_feed("A")["events"]) == [
            "Report A", "research step from A", "thread step from A"]
        assert _messages(feed.get_feed("B")["events"]) == [
            "Report B", "research step from B", "thread step from B"]
        with rs.get_session() as session:
            rows = session.execute(select(TradingAgentReport)).scalars().all()
        assert {r.symbol: r.run_id for r in rows} == {"SYMA": "A", "SYMB": "B"}
        assert {str(r.trade_date) for r in rows} == {"2026-09-03"}

    def test_bare_thread_falls_back_to_the_process_run(self, feed):
        # An executor that did not copy the context (the runner's
        # investigation prefetch pool) still sees the process's run.
        feed.start_run("mine", run_id="mine")
        seen = {}
        t = threading.Thread(target=lambda: seen.update(rid=feed.current_run_id()))
        t.start()
        t.join()
        assert seen["rid"] == "mine"

    def test_adopt_run_is_task_scoped_too(self, feed):
        feed._get_cache().set("active_runs", ["sub"])
        feed.adopt_run("sub", title="Models")
        assert feed.current_run_id() == "sub"
        assert feed._run_title("sub") == "Models"


class TestWatchdog:
    def test_pid_is_tracked_per_run(self, feed, monkeypatch):
        feed.start_run("run A", run_id="run-a")
        feed.start_run("run B", run_id="run-b")
        feed.record_run_pid(run_id="run-a")
        assert feed.run_pid("run-a") == os.getpid()
        assert feed.run_pid("run-b") is None
        feed.clear_run_pid(run_id="run-a")
        assert feed.run_pid("run-a") is None

    def test_dead_worker_fails_only_its_run(self, feed, db, monkeypatch):
        feed.start_run("run A", run_id="run-a")
        feed.start_run("run B", run_id="run-b")
        feed._get_cache().set(feed._pid_key("run-a"), 999_999_999)
        monkeypatch.setattr(feed, "_pid_alive", lambda pid: False)
        failed = {}
        monkeypatch.setattr(
            "services.run_service.set_status",
            lambda run_id, status, error=None: failed.__setitem__(run_id, status))
        clock = {"now": 1_000.0}
        monkeypatch.setattr(feed.time, "time", lambda: clock["now"])

        assert feed.watchdog_check() is False        # first sighting, grace starts
        clock["now"] += feed.WATCHDOG_PID_GRACE_S + 1
        assert feed.watchdog_check() is True
        assert feed._active_runs() == ["run-b"]
        assert failed == {"run-a": "failed"}
        assert feed.get_feed("run-a")["events"][-1]["stage"] == "done"
        # Once only: the abort's own events must not retrigger it.
        assert feed.watchdog_check() is False

    def test_armed_run_whose_stages_never_start_is_reaped(self, feed, db, monkeypatch):
        # mark_run_pending puts the run in the active list with no events
        # and no pid; before it recorded when, nothing could age it and
        # the row held its owner's lock forever.
        clock = {"now": 1_000.0}
        monkeypatch.setattr(feed.time, "time", lambda: clock["now"])
        run_id = rs.create_run("manual", ["NVDA"], "u1")
        feed.mark_run_pending(run_id)
        assert feed._get_cache().get(feed._meta_key(run_id))["pending"] == 1_000.0

        assert feed.watchdog_check() is False
        clock["now"] += feed.WATCHDOG_STALL_S - 1
        assert feed.watchdog_check() is False
        assert rs.get_run(run_id)["status"] == "queued"

        clock["now"] += 2
        assert feed.watchdog_check() is True
        row = rs.get_run(run_id)
        assert row["status"] == "failed"
        assert "no stage started" in row["error"]
        assert run_id not in feed._active_runs()
        events = feed.get_feed(run_id)["events"]
        assert events[-1]["stage"] == "done"
        assert any("no stage started" in e["message"] for e in events)
        assert rs.active_run_for("u1") is None
        assert feed.watchdog_check() is False

    def test_started_run_keeps_its_pending_mark_and_ages_on_events(self, feed, db,
                                                                   monkeypatch):
        clock = {"now": 1_000.0}
        monkeypatch.setattr(feed.time, "time", lambda: clock["now"])
        run_id = rs.create_run("manual", ["NVDA"], "u1")
        feed.mark_run_pending(run_id)
        clock["now"] += feed.WATCHDOG_STALL_S - 5
        feed.start_run("late start", run_id=run_id)
        meta = feed._get_cache().get(feed._meta_key(run_id))
        assert meta["pending"] == 1_000.0 and meta["title"] == "late start"
        # The stall window now runs from the newest event, not the confirm.
        clock["now"] += 10
        assert feed.watchdog_check() is False
        assert rs.get_run(run_id)["status"] == "queued"


def _job_run(status):
    with rs.get_session() as session:
        job = JobRun(job_id="daily_analysis", trigger="schedule", status=status)
        session.add(job)
        session.flush()
        return job.id


class TestOrphanReap:
    """Rows the feed no longer vouches for are failed from the poll tick,
    and a row whose scheduler job has finalized is failed even while the
    feed still lists it (the runner was killed before finish_run)."""

    def test_watchdog_fails_a_row_whose_job_run_finished(self, feed, db, monkeypatch):
        clock = {"now": 1_000.0}
        monkeypatch.setattr(feed.time, "time", lambda: clock["now"])
        job = _job_run("error")
        killed = rs.create_run("scheduled", ["NVDA"], None, job_run_id=job)
        alive = rs.create_run("manual", ["AMD"], "u1")
        feed.start_run("scheduled", run_id=killed, kind="scheduled")
        feed.start_run("manual", run_id=alive)

        assert feed.watchdog_check() is True

        assert rs.get_run(killed)["status"] == "failed"
        assert "finished (error)" in rs.get_run(killed)["error"]
        assert feed._active_runs() == [alive]
        events = feed.get_feed(killed)["events"]
        assert events[-1]["stage"] == "done"
        assert any(e["stage"] == "error" and "Run aborted" in e["message"]
                   for e in events)
        assert rs.get_run(alive)["status"] == "queued"
        assert rs.active_run_for(None) is None

        # Throttled: the next tick inside the interval does not query again.
        seen = []
        monkeypatch.setattr("services.run_service.reap_orphans",
                            lambda **kw: seen.append(kw) or [])
        clock["now"] += 1
        feed.watchdog_check()
        assert seen == []
        clock["now"] += feed._DB_REAP_INTERVAL_S
        feed.watchdog_check()
        assert seen and seen[0]["live"] == [alive]

    def test_row_absent_from_feed_is_left_alone_until_the_ceiling(self, feed, db, monkeypatch):
        monkeypatch.setattr("services.scheduler_service.JOB_TIMEOUT_SECONDS", 3600)
        run_id = rs.create_run("manual", ["NVDA"], "u1")   # never armed in the feed
        assert feed.watchdog_check() is False
        assert rs.get_run(run_id)["status"] == "queued"

    def test_restart_reap_fails_everything_not_live(self, feed, db):
        run_id = rs.create_run("manual", ["NVDA"], "u1")
        feed.hydrate_from_db()   # empties the active list, as a boot does
        reaped = feed.reap_orphans(max_age_s=0, error=rs.ORPHAN_RESTART_ERROR)
        assert [r["run_id"] for r in reaped] == [run_id]
        assert rs.get_run(run_id)["error"] == rs.ORPHAN_RESTART_ERROR
        assert feed.get_feed(run_id)["events"] == []   # nothing to close in the feed
        assert feed.reap_orphans(max_age_s=0) == []      # sticky, once

    def test_fail_orphan_verdict(self, feed, db):
        live = rs.create_run("manual", ["NVDA"], "u1")
        feed.mark_run_pending(live)
        assert feed.fail_orphan(rs.get_run(live)) is None
        assert rs.get_run(live)["status"] == "queued"

        dead = rs.create_run("scheduled", ["AMD"], None, job_run_id=_job_run("interrupted"))
        feed.start_run("sched", run_id=dead, kind="scheduled")
        reason = feed.fail_orphan(rs.get_run(dead))
        assert reason and "interrupted" in reason
        assert rs.get_run(dead)["status"] == "failed"
        assert dead not in feed._active_runs()


class TestEmitProgress:
    def test_counters_merge_into_the_run_row(self, feed, db):
        run_id = rs.create_run("manual", ["NVDA", "AMD"], "u1")
        feed.start_run("run", run_id=run_id)

        feed.emit_progress("news", done=1, total=2, state="running",
                           articles=7, run_id=run_id)
        feed.emit_progress("news", done=2, total=2, state="done",
                           articles=12, run_id=run_id)
        feed.emit_progress("models", predictions=5, run_id=run_id)

        run = rs.get_run(run_id)
        assert run["status"] == "running"
        assert run["stages"]["news"] == {"state": "done", "done": 2, "total": 2}
        assert run["stages"]["models"]["state"] == "running"
        assert run["counters"] == {"articles": 12, "predictions": 5}

        events = [e for e in feed.get_feed(run_id)["events"]
                  if e.get("event") == "progress"]
        assert [e["done"] for e in events] == [1, 2, None]
        assert events[-1]["counters"] == {"predictions": 5}
        assert events[0]["stage"] == "news" and events[0]["state"] == "running"
        assert events[0]["message"] == "news 1/2 running articles=7"

    def test_unnamed_progress_resolves_to_the_open_run(self, feed, db):
        run_id = rs.create_run("scheduled", ["AAPL"], None)
        feed.start_run("scheduled", run_id=run_id, kind="scheduled")
        feed.emit_progress("models", done=1, total=1, state="done")
        assert rs.get_run(run_id)["stages"]["models"]["done"] == 1

    def test_progress_without_a_row_still_feeds(self, feed, db):
        feed.start_run("headless", run_id="no-row")
        feed.emit_progress("models", done=1, total=3)
        assert feed.get_feed("no-row")["events"][-1]["done"] == 1
        assert rs.get_run("no-row") is None

    def test_non_publisher_still_updates_the_row(self, feed, db, monkeypatch):
        monkeypatch.delenv(prog._ENV_FLAG)
        run_id = rs.create_run("scheduled", ["AAPL"], None)
        feed.start_run("cli", run_id=run_id, kind="scheduled")
        feed.emit_progress("news", done=1, total=1, state="done", articles=3)
        assert rs.get_run(run_id)["counters"] == {"articles": 3}
        assert feed.get_feed(run_id)["events"] == []

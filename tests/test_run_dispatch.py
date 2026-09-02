"""The confirm dispatcher and the run-dispatch handoff to the stage callbacks.

One run per owner: a confirm while this owner has a manual run in flight
is refused with a Cancel button and creates nothing (a scheduled run never
locks); otherwise the analysis_runs row is created, the feed armed, and
the run dict written to run-store (session, the panel pins to it) and
run-dispatch (memory), the latter being the only thing that starts the
stages: a session store re-emits on every mount, and stages hanging off
it re-fired on every page load. The stages read the dict through
_dispatch_run_id, which must ignore a value they already handled and a
run that is no longer in flight. Retry is a confirm with the previous
run's config. Cancel terminates the recorded worker and leaves a sticky
cancelled row behind that a late stage close cannot overturn.

Importing app registers every callback; the decorated functions are still
plain callables, so the confirm branch runs here with a stubbed ctx and
an in-memory SQLite behind run_service. No stage does any work: nothing
here fetches, models or calls an LLM.
"""

import asyncio
import os
import subprocess
import sys
from types import SimpleNamespace

import dash
import diskcache
import pytest
from dash.exceptions import PreventUpdate
from sqlalchemy import create_engine
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import sessionmaker

from db.models import AnalysisRun, Base, JobRun, Ticker
from services import progress_service as prog
from services import run_service as rs

# app.enable()s the feed for its own process at import; the rest of the
# suite must not inherit that and start writing to the real cache dir.
_flag_before = os.environ.get(prog._ENV_FLAG)
import app as app_module  # noqa: E402

if _flag_before is None:
    os.environ.pop(prog._ENV_FLAG, None)


@compiles(JSONB, "sqlite")
def _jsonb_as_json(type_, compiler, **kw):
    return "JSON"


@pytest.fixture
def db(monkeypatch):
    import db.session as dbs

    eng = create_engine("sqlite://")
    Base.metadata.create_all(eng, tables=[AnalysisRun.__table__, JobRun.__table__,
                                          Ticker.__table__])
    monkeypatch.setattr(dbs, "_engine", eng)
    monkeypatch.setattr(
        dbs, "_SessionLocal", sessionmaker(bind=eng, expire_on_commit=False))
    return dbs


@pytest.fixture
def feed(monkeypatch, tmp_path):
    monkeypatch.setenv(prog._ENV_FLAG, "1")
    monkeypatch.setattr(prog, "_cache", diskcache.Cache(str(tmp_path)))
    monkeypatch.setattr(prog, "_local_run", {"id": None, "title": None})
    monkeypatch.setattr(prog, "_write_audit", lambda *a, **kw: None)
    monkeypatch.setattr(prog, "_last_db_reap", 0.0)
    token = prog._run_ctx.set(None)
    yield prog
    prog._run_ctx.reset(token)


def _finished_job_run(status="error"):
    with rs.get_session() as session:
        job = JobRun(job_id="daily_analysis", trigger="schedule", status=status)
        session.add(job)
        session.flush()
        return job.id


@pytest.fixture
def confirm(db, feed, monkeypatch):
    """toggle_run_modal's confirm branch as the browser would call it."""
    monkeypatch.setattr(app_module, "ctx", SimpleNamespace(
        triggered_id="run-confirm-btn",
        triggered=[{"prop_id": "run-confirm-btn.n_clicks", "value": 1}]))
    monkeypatch.setattr(app_module, "_run_owner_uid", lambda: "u1")

    def _confirm(symbols, scope="full", **dialog):
        dialog.setdefault("run_date", "2026-09-03")
        dialog.setdefault("model_checks", [True, True, False, False, False])
        dialog.setdefault("recs", "auto")
        return app_module.toggle_run_modal(
            None, None, [], None, 1,
            {}, [], {}, True,
            {"symbols": symbols}, scope,
            {"mode": "normal", "closed": True},
            **dialog)

    return _confirm


# Output positions of toggle_run_modal.
IS_OPEN, VALIDATION, PANEL, TOAST_OPEN, TOAST_MSG, INTERVAL = 0, 3, 4, 5, 6, 7
RUN_STORE, RUN_DISPATCH, PRESET, PREFS, CUSTOMIZE = 11, 12, 13, 14, 15
BTN_DISABLED, BTN_LABEL = 16, 17


class TestConfirmDispatch:
    def test_creates_row_arms_feed_and_writes_run_store(self, confirm, feed):
        out = confirm(["nvda", "AMD"], scope="full")

        assert out[IS_OPEN] is False
        assert out[VALIDATION] == ""
        run_data = out[RUN_STORE]
        run_id = run_data["run_id"]
        assert run_data["scope"] == "full"
        assert run_data["symbols"] == ["nvda", "AMD"]
        assert run_data["owner_uid"] == "u1"
        assert run_data["kind"] == "manual"
        # The same dict lands in both stores in the one callback.
        assert out[RUN_DISPATCH] == run_data

        row = rs.get_run(run_id)
        assert row["status"] == "queued"
        assert row["kind"] == "manual"
        assert row["owner_uid"] == "u1"
        # Two models and recommendations on is not the Standard preset the
        # dialog defaulted to: the row says so, and names the fields.
        assert row["preset"] == "custom"
        assert run_data["preset"] == "custom"
        assert row["config"]["preset"] == "standard"
        assert set(row["config"]["customized"]) == {"models", "recs"}
        assert row["symbols"] == ["NVDA", "AMD"]
        assert row["config"]["models"] == ["kronos_mini", "xgboost_shap"]
        assert row["config"]["scope"] == "full"
        assert row["estimate_s"] and row["estimate_s"] > 0
        assert row["target_date"] == "2026-09-03"
        assert row["prediction_date"] < row["target_date"]

        # The feed knows the run before any stage has called start_run.
        assert run_id in feed._active_runs()
        assert out[PANEL]["closed"] is False
        assert out[TOAST_OPEN] is True
        assert "nvda, AMD" in out[TOAST_MSG]
        assert out[INTERVAL] == app_module._PROGRESS_POLL_ACTIVE_MS
        # What the browser remembers, and the confirm button back at rest.
        assert out[PREFS] == {"preset": "standard", "symbols": ["nvda", "AMD"]}
        assert out[BTN_DISABLED] is False
        assert out[BTN_LABEL][-1] == "Run"
        # The run's symbols joined the lookup cache.
        from services import ticker_service as ts
        assert ts.get("NVDA")["source"] == "run" and ts.get("AMD") is not None

    def test_refused_while_this_owner_has_a_run(self, confirm, feed):
        first = rs.create_run("manual", ["AAPL"], "u1")

        out = confirm(["NVDA"])

        assert out[IS_OPEN] is dash.no_update
        assert out[RUN_STORE] is dash.no_update
        assert out[RUN_DISPATCH] is dash.no_update
        assert out[TOAST_OPEN] is dash.no_update
        msg = out[VALIDATION]
        assert isinstance(msg, list)
        assert "run in progress" in msg[0]
        button = msg[1]
        assert button.id == "run-cancel-active-btn"
        assert button.size == "sm" and button.outline is True
        assert [r["run_id"] for r in rs.list_runs()] == [first]
        assert first not in feed._active_runs()
        # A refusal restores the confirm button the clientside spinner took.
        assert out[BTN_DISABLED] is False
        assert out[BTN_LABEL][-1] == "Run"
        assert out[PREFS] is dash.no_update

    def test_other_owners_run_does_not_block(self, confirm):
        rs.create_run("manual", ["AAPL"], "u2")

        out = confirm(["NVDA"])

        assert out[IS_OPEN] is False
        assert len(rs.list_runs()) == 2

    def test_scheduled_run_never_refuses(self, confirm, monkeypatch):
        # The daily job's row has owner None: it must not lock every
        # anonymous session for the length of the run. The pill shows it.
        monkeypatch.setattr(app_module, "_run_owner_uid", lambda: None)
        sched = rs.create_run("scheduled", ["AAPL"], None)

        out = confirm(["NVDA"])

        assert out[IS_OPEN] is False
        new_id = out[RUN_STORE]["run_id"]
        assert new_id != sched
        assert rs.get_run(sched)["status"] == "queued"
        assert {r["run_id"] for r in rs.active_runs()} == {sched, new_id}

    def test_empty_symbols_refused_before_any_lookup(self, confirm):
        out = confirm([])

        assert out[VALIDATION] == "Add at least one symbol to run."
        assert out[RUN_STORE] is dash.no_update
        assert rs.list_runs() == []

    def test_row_failure_keeps_dialog_open(self, confirm, monkeypatch):
        def _boom(**kw):
            raise RuntimeError("db down")
        monkeypatch.setattr(rs, "create_run", _boom)

        out = confirm(["NVDA"])

        assert out[IS_OPEN] is dash.no_update
        assert "Could not record the run" in out[VALIDATION]
        assert out[RUN_STORE] is dash.no_update

    def test_models_scope_estimate_and_preset(self, confirm):
        out = confirm(["NVDA"], scope="models",
                      model_checks=[True, False, False, False, False])
        row = rs.get_run(out[RUN_STORE]["run_id"])
        assert row["preset"] == "custom"
        assert row["config"]["scope"] == "models"
        assert row["config"]["models"] == ["kronos_mini"]
        assert row["config"]["ensemble"] is False

    def test_untouched_preset_is_recorded_by_name(self, confirm):
        # Quick is the four numerical models; TradingAgents (last box) off.
        out = confirm(["NVDA"], scope="models", preset="quick",
                      model_checks=[True] * 4 + [False], recs="off")
        row = rs.get_run(out[RUN_STORE]["run_id"])
        assert row["preset"] == "quick"
        assert row["config"]["customized"] == []
        assert out[RUN_STORE]["preset"] == "quick"
        assert out[PREFS]["preset"] == "quick"

    def test_unmounted_recs_keeps_the_preset_name(self, confirm):
        # The confirm branch reads the same raw values run_preflight does:
        # a run-recs that never rendered (None) is not a divergence, so
        # the row is "standard", the way the dialog's hint called it.
        out = confirm(["NVDA"], scope="full", preset="standard",
                      model_checks=[True] * 5, recs=None, run_tools=[])
        row = rs.get_run(out[RUN_STORE]["run_id"])
        assert row["preset"] == "standard"
        assert row["config"]["customized"] == []
        assert row["config"]["recs"] is None

    def test_symbol_cache_failure_never_blocks_a_run(self, confirm, monkeypatch):
        from services import ticker_service as ts

        def _boom(*a, **kw):
            raise RuntimeError("tickers table missing")
        monkeypatch.setattr(ts, "ensure_symbols", _boom)
        out = confirm(["NVDA"])
        assert out[IS_OPEN] is False
        assert rs.get_run(out[RUN_STORE]["run_id"])["status"] == "queued"


class TestStageDispatchGuard:
    def test_no_store_and_handled_store_are_ignored(self, db):
        run_id = rs.create_run("manual", ["NVDA"], "u1")
        with pytest.raises(PreventUpdate):
            app_module._dispatch_run_id(None, None)
        with pytest.raises(PreventUpdate):
            app_module._dispatch_run_id({}, None)
        with pytest.raises(PreventUpdate):
            app_module._dispatch_run_id({"run_id": run_id}, {"run_id": run_id})

    def test_dict_without_an_id_is_reported_not_swallowed(self, db, feed, caplog):
        # None is the idle store; a dict with no run_id is a dispatcher
        # bug and must leave a trace on the feed and in the log.
        with pytest.raises(PreventUpdate):
            app_module._dispatch_run_id(None, None)
        # Nothing in flight: the idle feed is the rolling log, where an
        # unnamed line lands.
        assert feed.get_feed()["events"] == []
        with caplog.at_level("WARNING", logger=app_module.logger.name):
            with pytest.raises(PreventUpdate):
                app_module._dispatch_run_id({"scope": "full"}, None)
        assert any("no run_id" in r.message for r in caplog.records)
        events = feed.get_feed()["events"]
        assert events and events[-1]["stage"] == "error"
        assert "no id" in events[-1]["message"]

    def test_fresh_store_dispatches_until_the_run_closes(self, db):
        run_id = rs.create_run("manual", ["NVDA"], "u1")
        assert app_module._dispatch_run_id({"run_id": run_id}, None) == run_id
        assert app_module._dispatch_run_id(
            {"run_id": run_id}, {"run_id": "older"}) == run_id
        rs.set_status(run_id, "done")
        with pytest.raises(PreventUpdate):
            app_module._dispatch_run_id({"run_id": run_id}, None)

    def test_mark_run_dispatched(self, db):
        out = app_module.mark_run_dispatched({"run_id": "r1"}, None)
        assert out["run_id"] == "r1"
        with pytest.raises(PreventUpdate):
            app_module.mark_run_dispatched({"run_id": "r1"}, out)
        with pytest.raises(PreventUpdate):
            app_module.mark_run_dispatched(None, None)

    def test_full_analysis_flag_follows_the_row_scope(self, db):
        run_id = rs.create_run("manual", ["NVDA"], "u1", config={"scope": "full"})
        store = {"run_id": run_id, "scope": "full", "symbols": ["NVDA"]}
        # The armed value names the run, so a stale flag cannot arm another.
        assert app_module.set_full_analysis_flag(store, None) == run_id
        models = rs.create_run("manual", ["NVDA"], "u1", config={"scope": "models"})
        with pytest.raises(PreventUpdate):
            app_module.set_full_analysis_flag(
                {**store, "run_id": models, "scope": "models"}, None)
        with pytest.raises(PreventUpdate):
            app_module.set_full_analysis_flag(store, {"run_id": run_id})


class TestCloseAndCancel:
    def test_close_run_writes_feed_and_first_error_line(self, db, feed):
        run_id = rs.create_run("manual", ["NVDA"], "u1")
        feed.start_run("t", run_id=run_id)

        app_module._close_run(run_id, "Run failed: boom", "failed",
                              error=RuntimeError("boom\nTraceback line"))

        row = rs.get_run(run_id)
        assert row["status"] == "failed"
        assert row["error"] == "boom"
        assert row["finished_at"] is not None
        assert run_id not in feed._active_runs()
        assert feed.get_feed(run_id)["events"][-1]["stage"] == "done"

    def test_close_run_done_without_error(self, db, feed):
        run_id = rs.create_run("manual", ["NVDA"], "u1")
        app_module._close_run(run_id, "Predictions complete: 2 stored")
        assert rs.get_run(run_id)["status"] == "done"
        assert rs.get_run(run_id)["error"] is None

    def test_cancel_terminates_worker_and_sticks(self, db, feed, monkeypatch):
        monkeypatch.setattr(app_module, "_run_owner_uid", lambda: "u1")
        run_id = rs.create_run("manual", ["NVDA"], "u1")
        feed.mark_run_pending(run_id)
        worker = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"])
        try:
            feed._get_cache().set(feed._pid_key(run_id), worker.pid)

            msg, flag = app_module.cancel_active_run(1)

            assert "cancelled" in msg
            # A cancelled full run's stages never disarm the synthesis flag
            # themselves; the cancel must, or the next models-only run
            # hands off to a synthesis that never comes.
            assert flag is False
            worker.wait(timeout=10)
            assert worker.returncode is not None
        finally:
            if worker.poll() is None:
                worker.kill()

        row = rs.get_run(run_id)
        assert row["status"] == "cancelled"
        assert run_id not in feed._active_runs()
        assert feed.run_pid(run_id) is None
        # The stage that was killed (or one still running in the server)
        # closing late must not overturn the cancel.
        app_module._close_run(run_id, "Predictions complete: 2 stored")
        assert rs.get_run(run_id)["status"] == "cancelled"
        assert rs.active_run_for("u1") is None

    def test_cancel_with_nothing_active(self, db, feed, monkeypatch):
        monkeypatch.setattr(app_module, "_run_owner_uid", lambda: "u1")
        msg, flag = app_module.cancel_active_run(1)
        assert "No run in progress" in msg
        assert flag is dash.no_update
        with pytest.raises(PreventUpdate):
            app_module.cancel_active_run(None)

    def test_cancel_never_sees_a_scheduled_run(self, db, feed, monkeypatch):
        # Scheduled runs are the scheduler's: not this owner's lock, not
        # cancellable from the dialog, live or orphaned alike (the reaper
        # closes the orphan; the dialog has nothing to clear).
        monkeypatch.setattr(app_module, "_run_owner_uid", lambda: None)
        live = rs.create_run("scheduled", ["NVDA"], None)
        feed.start_run("sched", run_id=live, kind="scheduled")
        dead = rs.create_run("scheduled", ["AMD"], None,
                             job_run_id=_finished_job_run("error"))
        msg, flag = app_module.cancel_active_run(1)
        assert "No run in progress" in msg
        assert flag is dash.no_update
        assert rs.get_run(live)["status"] == "queued"
        assert rs.get_run(dead)["status"] == "queued"
        assert live in feed._active_runs()


class TestSynthesisFlag:
    """full-analysis-requested names the run it was armed for. A flag left
    behind by a run that ended without disarming it (a cancelled full run)
    must not make the next models-only run wait for a synthesis that never
    comes, and a cancelled run's late report must not synthesize."""

    @pytest.fixture
    def persist(self, monkeypatch):
        import services.analysis_runner as runner
        import services.dashboard_service as dash_svc
        monkeypatch.setattr(runner, "persist_predictions",
                            lambda signals, run_id=None: (2, 0))
        monkeypatch.setattr(dash_svc, "invalidate_memo", lambda: None)

    def _signals(self, run_id):
        return {"NVDA": {"kronos": {"decision": "BUY"}},
                "_meta": {"run_id": run_id, "run_seq": run_id,
                          "predict_date": "2026-09-03"}}

    def test_armed_states(self):
        assert app_module._synthesis_armed("r1", "r1")
        assert app_module._synthesis_armed(True, "r1")
        assert not app_module._synthesis_armed("r0", "r1")
        assert not app_module._synthesis_armed(False, "r1")
        assert not app_module._synthesis_armed(None, "r1")

    def test_flag_left_by_another_run_does_not_hold_the_row(self, db, feed, persist):
        cancelled = rs.create_run("manual", ["NVDA"], "u1")
        rs.cancel_run(cancelled)
        run_id = rs.create_run("manual", ["NVDA"], "u1", preset="models")
        feed.mark_run_pending(run_id)

        out = app_module.persist_predictions(self._signals(run_id), cancelled, None)

        assert out["count"] == 2
        assert rs.get_run(run_id)["status"] == "done"
        assert run_id not in feed._active_runs()
        assert rs.active_run_for("u1") is None

    def test_flag_for_this_run_hands_off(self, db, feed, persist):
        run_id = rs.create_run("manual", ["NVDA"], "u1")
        feed.mark_run_pending(run_id)

        app_module.persist_predictions(self._signals(run_id), run_id, None)

        assert rs.get_run(run_id)["status"] == "queued"
        assert run_id in feed._active_runs()
        assert any("Handing off" in e["message"]
                   for e in feed.get_feed(run_id)["events"])

    def test_cancelled_runs_late_report_never_synthesizes(self, db, feed, monkeypatch):
        import services.analysis_runner as runner
        run_id = rs.create_run("manual", ["NVDA"], "u1")
        rs.cancel_run(run_id)
        monkeypatch.setattr(app_module, "ctx",
                            SimpleNamespace(triggered_id="ai-analysis-store"))

        def _never(*a, **kw):
            raise AssertionError("synthesis ran for a cancelled run")
        monkeypatch.setattr(runner, "run_recommendations", _never)

        report = {"run_id": run_id, "run_seq": run_id, "scope": "full",
                  "recs_request": "news+signals", "by_symbol": {"NVDA": {}}}
        with pytest.raises(PreventUpdate):
            asyncio.run(app_module.generate_recommendations_callback(
                report, None, False))
        assert rs.get_run(run_id)["status"] == "cancelled"


class TestReportStageProgress:
    def test_rejected_report_records_the_stage_and_fails_the_row(self, db, feed,
                                                                 monkeypatch):
        run_id = rs.create_run(
            "manual", ["NVDA"], "u1", preset="report",
            config={"scope": "report", "target_date": "2026-09-03",
                    "lookback": None, "max_articles": 25})
        feed.mark_run_pending(run_id)
        monkeypatch.setattr(app_module, "ctx",
                            SimpleNamespace(triggered_id="run-dispatch"))
        store = {"run_id": run_id, "scope": "report", "symbols": ["NVDA"],
                 "owner_uid": "u1", "kind": "manual"}

        # No news window on the row: the stage refuses before any fetch,
        # and the refusal must reach the row as a failed report.
        out = asyncio.run(app_module.generate_ai_analysis(
            store, {"NVDA": {"prices": "{}"}}, None))

        assert out["failed"] is True and out["run_id"] == run_id
        row = rs.get_run(run_id)
        assert row["status"] == "failed"
        assert row["stages"]["report"] == {"state": "failed", "done": 0, "total": 1}
        assert run_id not in feed._active_runs()

    def test_row_is_running_before_the_price_fill(self, db, feed, monkeypatch):
        # The server-side price fill is the first slow thing the report
        # stage does; a row still queued through it is what the watchdog
        # reaps as a run that never started. So the flip to running must
        # come first, and the fill observes it.
        from sqlalchemy.pool import StaticPool

        # The fill runs in a worker thread (asyncio.to_thread), where a
        # per-thread in-memory SQLite is empty: one shared connection.
        eng = create_engine("sqlite://", poolclass=StaticPool,
                            connect_args={"check_same_thread": False})
        Base.metadata.create_all(eng, tables=[AnalysisRun.__table__])
        monkeypatch.setattr(db, "_engine", eng)
        monkeypatch.setattr(
            db, "_SessionLocal", sessionmaker(bind=eng, expire_on_commit=False))
        run_id = rs.create_run(
            "manual", ["NVDA"], "u1", preset="report",
            config={"scope": "report", "target_date": "2026-09-03",
                    "lookback": 7, "max_articles": 10})
        feed.mark_run_pending(run_id)
        monkeypatch.setattr(app_module, "ctx",
                            SimpleNamespace(triggered_id="run-dispatch"))
        seen = {}

        class _Stop(Exception):
            """Ends the stage once the fill has looked at the row."""

        def _fill(symbols, stock_data, news_data=None):
            seen["status"] = rs.get_run(run_id)["status"]
            seen["symbols"] = list(symbols)
            raise _Stop()
        monkeypatch.setattr(app_module, "_fill_run_inputs", _fill)
        store = {"run_id": run_id, "scope": "report", "symbols": ["NVDA"],
                 "owner_uid": "u1", "kind": "manual"}

        with pytest.raises(_Stop):
            asyncio.run(app_module.generate_ai_analysis(store, {}, None))
        assert seen == {"status": "running", "symbols": ["NVDA"]}
        assert rs.get_run(run_id)["stages"]["report"]["state"] == "running"


class TestOrphanEscape:
    """A row whose process is provably gone must not lock its owner out."""

    def test_confirm_proceeds_past_an_orphaned_manual_row(self, confirm, db, feed):
        # Past the ceiling with nothing reporting on it: failed on the
        # spot rather than refusing this owner until someone edits the row.
        from datetime import timedelta
        from sqlalchemy import update
        orphan = rs.create_run("manual", ["NVDA"], "u1")
        with rs.get_session() as session:
            session.execute(update(AnalysisRun).where(AnalysisRun.run_id == orphan)
                            .values(started_at=rs._now() - timedelta(hours=3)))

        out = confirm(["AMD"], scope="models")

        assert out[IS_OPEN] is False
        assert rs.get_run(orphan)["status"] == "failed"
        new_id = out[RUN_STORE]["run_id"]
        assert new_id != orphan
        assert rs.active_run_for("u1")["run_id"] == new_id

    def test_confirm_ignores_scheduled_rows_live_or_dead(self, confirm, db, feed,
                                                        monkeypatch):
        monkeypatch.setattr(app_module, "_run_owner_uid", lambda: None)
        live = rs.create_run("scheduled", ["NVDA"], None,
                             job_run_id=_finished_job_run("running"))
        feed.start_run("sched", run_id=live, kind="scheduled")
        dead = rs.create_run("scheduled", ["TSLA"], None,
                             job_run_id=_finished_job_run("interrupted"))

        out = confirm(["AMD"], scope="models")

        assert out[IS_OPEN] is False
        assert rs.get_run(live)["status"] == "queued"
        # Not the dialog's job to reap: the watchdog and the scheduler do.
        assert rs.get_run(dead)["status"] == "queued"
        assert rs.active_run_for(None)["run_id"] == out[RUN_STORE]["run_id"]

    def test_startup_reap_unlocks_a_restarted_server(self, db, feed):
        run_id = rs.create_run("manual", ["NVDA"], "u1")
        feed.mark_run_pending(run_id)
        feed.hydrate_from_db()
        feed.reap_orphans(max_age_s=0, error=rs.ORPHAN_RESTART_ERROR)
        assert rs.active_run_for("u1") is None
        assert rs.get_run(run_id)["error"] == rs.ORPHAN_RESTART_ERROR


class TestPanelFollowsOwnRun:
    def _fp(self, out):
        return out[4]

    def test_panel_pins_the_viewers_run(self, db, feed):
        mine = rs.create_run("manual", ["NVDA"], "u1")
        theirs = rs.create_run("manual", ["AMD"], "u2")
        feed.start_run("mine", run_id=mine)
        feed.emit("news", "mine: fetching", run_id=mine)
        feed.start_run("theirs", run_id=theirs)     # newest in flight
        feed.emit("news", "theirs: fetching", run_id=theirs)

        pinned = app_module.render_progress_panel(
            1, None, None, {"run_id": mine})
        assert self._fp(pinned)[-1] == mine
        assert self._fp(pinned)[3] == "mine: fetching"

        # No run of my own (or one that already finished): the newest run
        # in flight, as before.
        unpinned = app_module.render_progress_panel(1, None, None, None)
        assert self._fp(unpinned)[-1] == theirs
        feed.finish_run("done", run_id=mine)
        after = app_module.render_progress_panel(
            1, None, None, {"run_id": mine})
        assert self._fp(after)[-1] == theirs

    def test_panel_state_is_wired(self):
        from dash._callback import GLOBAL_CALLBACK_MAP
        cb = next(v for k, v in GLOBAL_CALLBACK_MAP.items()
                  if k.startswith("..progress-feed-scroll.children"))
        assert any(st["id"] == "run-store" for st in cb.get("state", []))


class TestRetry:
    """Retry is a confirm with the previous run's config: it goes through
    the lock and creates a fresh row, never reopening the old one."""

    @pytest.fixture
    def prev(self, db, feed, monkeypatch):
        monkeypatch.setattr(app_module, "_run_owner_uid", lambda: "u1")
        run_id = rs.create_run(
            "manual", ["nvda", "AMD"], "u1", preset="report",
            config={"scope": "report", "lookback": 14, "models": []},
            prediction_date="2026-09-02", target_date="2026-09-03",
            estimate_s=90)
        rs.set_status(run_id, "failed", error="provider timeout")
        return {"run_id": run_id, "scope": "report", "symbols": ["nvda", "AMD"],
                "owner_uid": "u1", "kind": "manual"}

    def test_nothing_to_retry(self, db, feed):
        with pytest.raises(PreventUpdate):
            app_module.retry_run(None, {"run_id": "r1"})
        with pytest.raises(PreventUpdate):
            app_module.retry_run(1, None)
        with pytest.raises(PreventUpdate):
            app_module.retry_run(1, {})
        assert rs.list_runs() == []

    def test_retry_is_a_fresh_run_through_both_stores(self, prev, feed):
        out = app_module.retry_run(1, prev, {"mode": "normal", "closed": True})
        store, dispatch, is_open, msg = out[:4]

        assert msg == "" and is_open is dash.no_update
        assert store == dispatch
        new_id = store["run_id"]
        assert new_id != prev["run_id"]
        assert store["retry_of"] == prev["run_id"]
        assert store["scope"] == "report"
        assert store["symbols"] == ["NVDA", "AMD"]
        assert store["owner_uid"] == "u1" and store["kind"] == "manual"

        row = rs.get_run(new_id)
        old = rs.get_run(prev["run_id"])
        assert row["status"] == "queued" and row["kind"] == "manual"
        assert row["preset"] == old["preset"]
        assert row["config"] == old["config"]
        assert row["prediction_date"] == old["prediction_date"]
        assert row["target_date"] == old["target_date"]
        assert row["estimate_s"] == old["estimate_s"]
        # The old row is untouched, and the new one is armed for dispatch.
        assert old["status"] == "failed" and old["error"] == "provider timeout"
        assert new_id in feed._active_runs()
        assert app_module._dispatch_run_id(dispatch, None) == new_id
        # Acknowledged like a confirm: panel forced open, toast with the
        # previous run's symbols and estimate, poll at the active rate.
        panel, toast_open, toast_msg, interval = out[4:]
        assert panel["closed"] is False and panel["mode"] == "normal"
        assert toast_open is True
        assert toast_msg.startswith("NVDA, AMD · ")
        assert "Reports" in toast_msg
        assert interval == app_module._PROGRESS_POLL_ACTIVE_MS

    def test_retry_refused_while_a_run_is_in_flight(self, prev, feed):
        active = rs.create_run("manual", ["TSLA"], "u1")

        out = app_module.retry_run(1, prev)
        store, dispatch, is_open, msg = out[:4]

        assert store is dash.no_update and dispatch is dash.no_update
        assert is_open is True
        assert "run in progress" in msg[0]
        assert msg[1].id == "run-cancel-active-btn"
        assert [r["run_id"] for r in rs.list_runs()] == [active, prev["run_id"]]
        # No acknowledgement for a run that did not start.
        assert all(v is dash.no_update for v in out[4:])

    def test_retry_of_a_row_without_an_estimate_still_acknowledges(self, db, feed,
                                                                  monkeypatch):
        monkeypatch.setattr(app_module, "_run_owner_uid", lambda: "u1")
        prev_id = rs.create_run("manual", ["NVDA"], "u1", preset="report",
                                config={"scope": "report"})
        rs.set_status(prev_id, "failed", error="x")
        out = app_module.retry_run(1, {"run_id": prev_id, "scope": "report"})
        assert out[0]["retry_of"] == prev_id
        assert out[5] is True and "duration unknown" in out[6]

    def test_retry_without_a_row_says_so(self, db, feed, monkeypatch):
        monkeypatch.setattr(app_module, "_run_owner_uid", lambda: "u1")
        out = app_module.retry_run(1, {"run_id": "gone"})
        store, dispatch, is_open, msg = out[:4]
        assert store is dash.no_update and is_open is True
        assert "record is gone" in msg
        assert rs.list_runs() == []
        assert all(v is dash.no_update for v in out[4:])


class TestRunConfigAuthority:
    """A stage runs with the settings on its run row, never with what the
    dialog's controls show at the time it fires: the dialog is not among
    its States at all, and where the dispatcher's snapshot disagrees with
    the row, the row wins. This is what makes a retry (a row copied from
    the previous one) run the previous run."""

    CONFIG = {
        "scope": "full", "target_date": "2026-09-03",
        "prediction_date": "2026-09-02", "lookback": 7, "max_articles": 10,
        "report_model": "gpt-5.6-luna", "depth": "standard", "recs": "off",
        "recs_model": "claude-sonnet-5", "evidence": [], "tools": [],
        "models": ["kronos_mini", "lightgbm"], "ensemble": True,
        "ensemble_members": ["kronos_mini", "lightgbm", "xgboost_shap"],
        "ensemble_weights": {"kronos_mini": 1.5, "lightgbm": 0.5},
        "ensemble_method": "agreement", "ensemble_min_agree": 2,
    }

    @pytest.fixture
    def dispatch(self, db, feed, monkeypatch):
        monkeypatch.setattr(app_module, "ctx",
                            SimpleNamespace(triggered_id="run-dispatch"))
        monkeypatch.setattr(app_module, "_run_owner_uid", lambda: "u1")
        return db

    def test_stage_callbacks_read_no_dialog_control(self):
        from dash._callback import GLOBAL_CALLBACK_MAP
        dialog = {"run-scope", "run-symbols-store", "run-date-picker",
                  "run-lookback", "run-max-articles", "run-model", "run-type",
                  "run-recs", "run-recs-model", "run-evidence", "run-tools",
                  "run-ensemble-check", "run-ensemble-method",
                  "run-ensemble-min-agree", "ensemble-config-store"}
        for wanted in ("ai-analysis-store.data", "model-signals-store.data",
                       "full-analysis-requested.data", "recommendations-store.data"):
            cb = next(v for k, v in GLOBAL_CALLBACK_MAP.items()
                      if wanted in k and "run-dispatch" in {
                          i["id"] for i in v.get("inputs", [])
                          if isinstance(i["id"], str)}
                      or (wanted == "recommendations-store.data"
                          and k.startswith("..recommendations-store.data")))
            state_ids = [st["id"] for st in cb.get("state", [])]
            assert not any(isinstance(i, dict) for i in state_ids), wanted
            assert not (set(state_ids) & dialog), (wanted, state_ids)

    def test_flag_follows_the_row_when_the_snapshot_disagrees(self, dispatch):
        full = rs.create_run("manual", ["NVDA"], "u1", config={"scope": "full"})
        report = rs.create_run("manual", ["NVDA"], "u1", config={"scope": "report"})
        # The row says full: armed, whatever the snapshot claims.
        assert app_module.set_full_analysis_flag(
            {"run_id": full, "scope": "report", "symbols": ["NVDA"]}, None) == full
        # The row says report: never armed, even when the snapshot says full.
        with pytest.raises(PreventUpdate):
            app_module.set_full_analysis_flag(
                {"run_id": report, "scope": "full", "symbols": ["NVDA"]}, None)

    def test_model_stage_runs_the_rows_config(self, dispatch, feed, monkeypatch):
        import services.analysis_runner as runner
        import services.news_window as nw
        run_id = rs.create_run("manual", ["nvda"], "u1", config=self.CONFIG)
        feed.mark_run_pending(run_id)
        # The subprocess stamps its environment; keep that out of the suite.
        monkeypatch.delenv("QUANTNEWS_RUN_ID", raising=False)
        monkeypatch.delenv("_DASH_BG_SUBPROCESS", raising=False)
        seen = {}
        monkeypatch.setattr(runner, "load_market_data",
                            lambda symbols, period="2y": {s: {} for s in symbols})
        monkeypatch.setattr(nw, "fetch_run_news",
                            lambda symbols, *a, **kw: (
                                {s: [] for s in symbols},
                                {s: {"status": "empty", "kept": 0} for s in symbols}))

        def _predict(symbols, stock_data, news, **kw):
            seen.update(kw, symbols=symbols)
            return {"_meta": {}}
        monkeypatch.setattr(runner, "run_predictions", _predict)

        # The snapshot claims a report-only run with another symbol; the
        # row says full-scope models for NVDA, and the row is what runs.
        store = {"run_id": run_id, "scope": "models", "symbols": ["amd"],
                 "owner_uid": "u1", "kind": "manual"}
        out = app_module.generate_model_signals(store, None)

        assert out["_meta"]["run_id"] == run_id
        assert seen["symbols"] == ["NVDA"]
        assert seen["models"] == {"kronos_mini", "lightgbm"}
        assert str(seen["target_date"]) == "2026-09-03"
        assert str(seen["cutoff_date"]) == "2026-09-02"
        assert seen["news_lookback_days"] == 7
        assert seen["run_ensemble"] is True
        assert seen["ensemble_config"] == {
            "enabled_models": ["kronos_mini", "lightgbm", "xgboost_shap"],
            "weights": {"kronos_mini": 1.5, "xgboost_shap": 1.0, "lightgbm": 0.5,
                        "deberta_sentiment": 1.0, "trading_agents": 1.0},
            "method": "agreement", "min_agree": 2}
        # Full scope: the research kwargs come from the row's report fields.
        assert seen["research_model"] == "gpt-5.6-luna"
        assert seen["include_thesis"] is False
        assert seen["evidence"] == []
        row = rs.get_run(run_id)
        assert row["status"] == "running"
        assert row["stages"]["models"]["state"] == "running"

    def test_report_stage_runs_the_rows_config(self, dispatch, feed, monkeypatch):
        import services.news_window as nw
        import services.stock_data as sd
        import utils.metrics as metrics
        run_id = rs.create_run("manual", ["NVDA"], "u1", config=self.CONFIG)
        feed.mark_run_pending(run_id)
        seen = {}

        def _news(symbols, as_of, target, **kw):
            seen.update(kw, as_of=as_of, target=target)
            return ({s: [] for s in symbols},
                    {s: {"status": "empty", "kept": 0} for s in symbols})
        monkeypatch.setattr(nw, "fetch_run_news", _news)
        monkeypatch.setattr(sd, "get_stock_info", lambda sym: SimpleNamespace(
            name="Nvidia", sector="Tech", industry="Semis"))

        def _no_network(df):
            raise RuntimeError("metrics need the network")
        monkeypatch.setattr(metrics, "compute_trading_metrics", _no_network)
        monkeypatch.setattr(app_module, "_s3_available", False)
        monkeypatch.setattr(app_module, "get_llm", lambda: None)

        store = {"run_id": run_id, "scope": "report", "symbols": ["AMD"],
                 "owner_uid": "u1", "kind": "manual"}
        out = asyncio.run(app_module.generate_ai_analysis(
            store, {"NVDA": {"prices": "{}"}}, None))

        # The window, cap and dates are the row's; the snapshot's scope and
        # symbol were ignored (full scope: no research here, no overall
        # call without articles, so the report is honestly empty).
        assert seen["lookback_days"] == 7 and seen["max_articles"] == 10
        assert seen["overnight"] is False
        assert seen["as_of"] == "2026-09-02" and seen["target"] == "2026-09-03"
        assert out["scope"] == "full" and out["run_id"] == run_id
        assert out["news_window"] == {"lookback_days": 7, "overnight": False,
                                      "max_articles": 10}
        assert out["as_of"] == "2026-09-02"
        assert out.get("recs_off") is True and "recs_request" not in out
        options = next(e["message"] for e in feed.get_feed(run_id)["events"]
                       if e["message"].startswith("Options:"))
        assert "model=gpt-5.6-luna" in options and "window=7d" in options
        assert "type=standard" in options and "recs=off" in options
        row = rs.get_run(run_id)
        assert row["stages"]["synthesis"] == {"state": "skipped"}
        assert row["stages"]["news"]["total"] == 1

    def test_unreadable_row_fails_the_run_instead_of_guessing(self, dispatch, feed,
                                                              monkeypatch):
        run_id = rs.create_run("manual", ["NVDA"], "u1", config=self.CONFIG)
        feed.mark_run_pending(run_id)

        def _boom(rid):
            raise RuntimeError("db away")
        monkeypatch.setattr(rs, "get_run", _boom)
        store = {"run_id": run_id, "scope": "report", "symbols": ["NVDA"],
                 "owner_uid": "u1", "kind": "manual"}
        out = asyncio.run(app_module.generate_ai_analysis(store, {}, None))
        assert out["failed"] is True and "could not be read" in out["error"]
        assert run_id not in feed._active_runs()

        out = app_module.generate_model_signals(
            {**store, "scope": "models"}, None)
        assert "could not be read" in out["_run_failed"]

    def test_confirm_records_the_whole_dialog(self, confirm):
        out = confirm(["NVDA"], scope="full", lookback=14, max_articles=25,
                      report_model="gpt-5.6-luna", depth="thesis", recs="auto",
                      recs_model="claude-sonnet-5", run_evidence=["quality"],
                      run_tools=["web_research"], run_ensemble=True,
                      ens_method="agreement", ens_min_agree="2",
                      ens_members=[True, False, True, False, False],
                      ens_weights=[1.5, 1.0, "x", 1.0, 1.0])
        cfg = rs.get_run(out[RUN_STORE]["run_id"])["config"]
        assert cfg["target_date"] == "2026-09-03"
        assert cfg["prediction_date"] == "2026-09-02"
        assert cfg["picked_date"] == "2026-09-03"
        assert cfg["lookback"] == 14 and cfg["max_articles"] == 25
        assert cfg["report_model"] == "gpt-5.6-luna" and cfg["depth"] == "thesis"
        assert cfg["recs"] == "auto" and cfg["recs_model"] == "claude-sonnet-5"
        assert cfg["evidence"] == ["quality"] and cfg["tools"] == ["web_research"]
        assert cfg["models"] == ["kronos_mini", "xgboost_shap"]
        assert cfg["ensemble"] is True
        assert cfg["ensemble_members"] == ["kronos_mini", "lightgbm"]
        assert cfg["ensemble_weights"] == {
            "kronos_mini": 1.5, "xgboost_shap": 1.0, "lightgbm": 1.0,
            "deberta_sentiment": 1.0, "trading_agents": 1.0}
        assert cfg["ensemble_method"] == "agreement"
        assert cfg["ensemble_min_agree"] == 2
        # A retry copies it verbatim, so it runs the same run.
        assert app_module._ensemble_config_of(cfg)["enabled_models"] == [
            "kronos_mini", "lightgbm"]

    def test_confirm_falls_back_to_the_shared_ensemble_store(self, confirm):
        out = confirm(["NVDA"], ens_members=None,
                      ensemble_store={"enabled_models": ["lightgbm"],
                                      "weights": {"lightgbm": 0.7}})
        cfg = rs.get_run(out[RUN_STORE]["run_id"])["config"]
        assert cfg["ensemble_members"] == ["lightgbm"]
        assert cfg["ensemble_weights"] == {"lightgbm": 0.7}


class TestWiring:
    """The confirm click has ONE listener; the stages hang off run-dispatch
    (memory), never run-store (session: it re-emits on every mount)."""

    def _inputs(self, cb):
        return {i["id"] for i in cb.get("inputs", []) if isinstance(i["id"], str)}

    def test_confirm_button_feeds_only_the_dispatcher(self):
        # Plus the clientside spinner, which touches nothing but the
        # button's own props and runs in the browser.
        from dash._callback import GLOBAL_CALLBACK_LIST, GLOBAL_CALLBACK_MAP
        listeners = [k for k, v in GLOBAL_CALLBACK_MAP.items()
                     if "run-confirm-btn" in self._inputs(v)]
        assert len(listeners) == 2
        server = [k for k in listeners if "run-modal.is_open" in k]
        assert len(server) == 1
        spinner = next(c for c in GLOBAL_CALLBACK_LIST
                       if str(c["output"]).startswith("..run-confirm-btn.disabled"))
        assert spinner["clientside_function"] is not None
        assert {i["id"] for i in spinner["inputs"]} == {"run-confirm-btn"}

    def test_stages_listen_on_run_dispatch(self):
        from dash._callback import GLOBAL_CALLBACK_MAP
        listeners = {k for k, v in GLOBAL_CALLBACK_MAP.items()
                     if "run-dispatch" in self._inputs(v)}
        for wanted in ("ai-analysis-store.data", "model-signals-store.data",
                       "full-analysis-requested.data", "run-dispatched.data"):
            assert any(wanted in k for k in listeners), wanted

    def test_run_store_alone_dispatches_nothing(self):
        # A session store restoring on mount re-emits its value; if any
        # callback took run-store as an Input, every page load in a tab
        # that once confirmed a run would fire it (a worker fork for the
        # model stage). It may only ever be read as State.
        from dash._callback import GLOBAL_CALLBACK_MAP
        assert [k for k, v in GLOBAL_CALLBACK_MAP.items()
                if "run-store" in self._inputs(v)] == []
        readers = {k for k, v in GLOBAL_CALLBACK_MAP.items()
                   if any(st["id"] == "run-store" for st in v.get("state", []))}
        assert any("progress-feed-scroll.children" in k for k in readers)

    def test_retry_button_feeds_only_the_retry_dispatcher(self):
        from dash._callback import GLOBAL_CALLBACK_MAP
        listeners = [k for k, v in GLOBAL_CALLBACK_MAP.items()
                     if "ai-retry-btn" in self._inputs(v)]
        assert len(listeners) == 1
        assert "run-store.data" in listeners[0]
        assert "run-dispatch.data" in listeners[0]
        # It acknowledges like the confirm does.
        for wanted in ("progress-panel-state.data", "run-started-toast.is_open",
                       "run-started-toast.children", "progress-interval.interval"):
            assert wanted in listeners[0], wanted
        # The only button is the Analyze page's failure block, absent on
        # every other route; the renderer must tolerate that.
        retry = GLOBAL_CALLBACK_MAP[listeners[0]]
        btn = next(i for i in retry["inputs"] if i["id"] == "ai-retry-btn")
        assert btn.get("allow_optional") is True

    def test_retry_button_is_not_a_hidden_placeholder(self):
        from layouts.main_layout import create_layout
        from layouts.pages.analyze import create_ai_failure_indicator

        def ids(node, acc):
            i = getattr(node, "id", None)
            if isinstance(i, str):
                acc.add(i)
            ch = getattr(node, "children", None)
            for c in (ch if isinstance(ch, (list, tuple)) else [ch]):
                if c is not None and (hasattr(c, "children") or hasattr(c, "id")):
                    ids(c, acc)
            return acc

        assert "ai-retry-btn" not in ids(create_layout(), set())
        assert "ai-retry-btn" in ids(create_ai_failure_indicator(), set())

    def test_toasts_and_run_stores_are_mounted(self):
        from layouts.main_layout import create_layout

        def ids(node, acc):
            i = getattr(node, "id", None)
            if isinstance(i, str):
                acc.add(i)
            ch = getattr(node, "children", None)
            for c in (ch if isinstance(ch, (list, tuple)) else [ch]):
                # Leaf components (dcc.Store, dbc.Toast) carry no children.
                if c is not None and (hasattr(c, "children") or hasattr(c, "id")):
                    ids(c, acc)
            return acc

        mounted = ids(create_layout(), set())
        assert {"run-store", "run-dispatch", "run-dispatched", "run-prefs-store",
                "run-started-toast", "history-eval-toast"} <= mounted

    def test_run_dispatch_is_a_memory_store(self):
        from layouts.main_layout import create_layout

        def find(node, wanted):
            if getattr(node, "id", None) == wanted:
                return node
            ch = getattr(node, "children", None)
            for c in (ch if isinstance(ch, (list, tuple)) else [ch]):
                if c is not None and hasattr(c, "id"):
                    hit = find(c, wanted)
                    if hit is not None:
                        return hit
            return None

        layout = create_layout()
        assert getattr(find(layout, "run-store"), "storage_type", None) == "session"
        dispatch = find(layout, "run-dispatch")
        assert dispatch is not None
        assert getattr(dispatch, "storage_type", "memory") == "memory"

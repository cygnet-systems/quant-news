"""The confirm dispatcher and the run-store handoff to the stage callbacks.

One run per owner: a confirm while this owner has a run in flight is
refused with a Cancel button and creates nothing; otherwise the
analysis_runs row is created, the feed armed, and the id written to
run-store, which is the only thing that starts the stages. The stages
read that store through _dispatch_run_id, which must ignore a store value
they already handled (a session store restoring on reload) and a run that
is no longer in flight. Cancel terminates the recorded worker and leaves a
sticky cancelled row behind that a late stage close cannot overturn.

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

from db.models import AnalysisRun, Base, JobRun
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
    Base.metadata.create_all(eng, tables=[AnalysisRun.__table__, JobRun.__table__])
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
            {"source": "custom", "symbols": symbols}, scope,
            {"mode": "normal", "closed": True},
            **dialog)

    return _confirm


# Output positions of toggle_run_modal.
IS_OPEN, VALIDATION, PANEL, TOAST_OPEN, TOAST_MSG, INTERVAL, RUN_STORE = \
    0, 6, 7, 8, 9, 10, 14


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

        row = rs.get_run(run_id)
        assert row["status"] == "queued"
        assert row["kind"] == "manual"
        assert row["owner_uid"] == "u1"
        assert row["preset"] == "full"
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

    def test_refused_while_this_owner_has_a_run(self, confirm, feed):
        first = rs.create_run("manual", ["AAPL"], "u1")

        out = confirm(["NVDA"])

        assert out[IS_OPEN] is dash.no_update
        assert out[RUN_STORE] is dash.no_update
        assert out[TOAST_OPEN] is dash.no_update
        msg = out[VALIDATION]
        assert isinstance(msg, list)
        assert "run in progress" in msg[0]
        button = msg[1]
        assert button.id == "run-cancel-active-btn"
        assert button.size == "sm" and button.outline is True
        assert [r["run_id"] for r in rs.list_runs()] == [first]
        assert first not in feed._active_runs()

    def test_other_owners_run_does_not_block(self, confirm):
        rs.create_run("manual", ["AAPL"], "u2")

        out = confirm(["NVDA"])

        assert out[IS_OPEN] is False
        assert len(rs.list_runs()) == 2

    def test_scheduled_run_refuses_without_cancel_button(self, confirm):
        rs.create_run("scheduled", ["AAPL"], "u1")

        out = confirm(["NVDA"])

        assert out[IS_OPEN] is dash.no_update
        assert isinstance(out[VALIDATION], str)
        assert "scheduled run" in out[VALIDATION]
        assert len(rs.list_runs()) == 1

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
        assert row["preset"] == "models"
        assert row["config"]["models"] == ["kronos_mini"]
        assert row["config"]["ensemble"] is False


class TestStageDispatchGuard:
    def test_no_store_and_handled_store_are_ignored(self, db):
        run_id = rs.create_run("manual", ["NVDA"], "u1")
        with pytest.raises(PreventUpdate):
            app_module._dispatch_run_id(None, None)
        with pytest.raises(PreventUpdate):
            app_module._dispatch_run_id({}, None)
        with pytest.raises(PreventUpdate):
            app_module._dispatch_run_id({"run_id": run_id}, {"run_id": run_id})

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

    def test_full_analysis_flag_follows_the_store_scope(self, db):
        run_id = rs.create_run("manual", ["NVDA"], "u1")
        store = {"run_id": run_id, "scope": "full", "symbols": ["NVDA"]}
        # The armed value names the run, so a stale flag cannot arm another.
        assert app_module.set_full_analysis_flag(store, "models", {}, None) == run_id
        with pytest.raises(PreventUpdate):
            app_module.set_full_analysis_flag(
                {**store, "scope": "models"}, "full", {}, None)
        with pytest.raises(PreventUpdate):
            app_module.set_full_analysis_flag(store, "full", {}, {"run_id": run_id})


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

    def test_cancel_refuses_scheduled(self, db, feed, monkeypatch):
        monkeypatch.setattr(app_module, "_run_owner_uid", lambda: None)
        run_id = rs.create_run("scheduled", ["NVDA"], None)
        msg, flag = app_module.cancel_active_run(1)
        assert "scheduled" in msg
        assert flag is dash.no_update
        assert rs.get_run(run_id)["status"] == "queued"

    def test_cancel_clears_a_scheduled_run_whose_job_finalized(self, db, feed, monkeypatch):
        monkeypatch.setattr(app_module, "_run_owner_uid", lambda: None)
        run_id = rs.create_run("scheduled", ["NVDA"], None,
                               job_run_id=_finished_job_run("error"))
        feed.start_run("sched", run_id=run_id, kind="scheduled")

        msg, _flag = app_module.cancel_active_run(1)

        assert "Stale scheduled run cleared" in msg
        assert rs.get_run(run_id)["status"] == "failed"
        assert run_id not in feed._active_runs()
        assert rs.active_run_for(None) is None


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
                report, None, False, {"symbols": ["NVDA"]}))
        assert rs.get_run(run_id)["status"] == "cancelled"


class TestReportStageProgress:
    def test_rejected_report_records_the_stage_and_fails_the_row(self, db, feed,
                                                                 monkeypatch):
        run_id = rs.create_run("manual", ["NVDA"], "u1", preset="report")
        feed.mark_run_pending(run_id)
        monkeypatch.setattr(app_module, "ctx",
                            SimpleNamespace(triggered_id="run-store"))
        store = {"run_id": run_id, "scope": "report", "symbols": ["NVDA"],
                 "owner_uid": "u1", "kind": "manual"}

        # No news window from the dialog: the stage refuses before any
        # fetch, and the refusal must reach the row as a failed report.
        out = asyncio.run(app_module.generate_ai_analysis(
            store, None, "report", None, {"NVDA": {"prices": "{}"}},
            "2026-09-03", None, 25, None, None, None, None, None, None, None))

        assert out["failed"] is True and out["run_id"] == run_id
        row = rs.get_run(run_id)
        assert row["status"] == "failed"
        assert row["stages"]["report"] == {"state": "failed", "done": 0, "total": 1}
        assert run_id not in feed._active_runs()


class TestOrphanEscape:
    """A row whose process is provably gone must not lock its owner out."""

    def test_confirm_proceeds_past_an_orphaned_scheduled_row(self, confirm, db, feed,
                                                            monkeypatch):
        monkeypatch.setattr(app_module, "_run_owner_uid", lambda: None)
        orphan = rs.create_run("scheduled", ["NVDA"], None,
                               job_run_id=_finished_job_run("interrupted"))

        out = confirm(["AMD"], scope="models")

        assert out[IS_OPEN] is False
        assert rs.get_run(orphan)["status"] == "failed"
        new_id = out[RUN_STORE]["run_id"]
        assert new_id != orphan
        assert rs.active_run_for(None)["run_id"] == new_id

    def test_confirm_still_refused_by_a_live_scheduled_row(self, confirm, db, feed,
                                                          monkeypatch):
        monkeypatch.setattr(app_module, "_run_owner_uid", lambda: None)
        live = rs.create_run("scheduled", ["NVDA"], None,
                             job_run_id=_finished_job_run("running"))
        feed.start_run("sched", run_id=live, kind="scheduled")
        out = confirm(["AMD"], scope="models")
        assert out[IS_OPEN] is dash.no_update
        assert "scheduled run is in progress" in out[VALIDATION]
        assert rs.get_run(live)["status"] == "queued"

    def test_startup_reap_unlocks_a_restarted_server(self, db, feed):
        run_id = rs.create_run("manual", ["NVDA"], "u1")
        feed.mark_run_pending(run_id)
        feed.hydrate_from_db()
        feed.reap_orphans(max_age_s=0, error=rs.ORPHAN_RESTART_ERROR)
        assert rs.active_run_for("u1") is None
        assert rs.get_run(run_id)["error"] == rs.ORPHAN_RESTART_ERROR


class TestPanelFollowsOwnRun:
    def _fp(self, out):
        return out[5]

    def test_panel_pins_the_viewers_run(self, db, feed):
        mine = rs.create_run("manual", ["NVDA"], "u1")
        theirs = rs.create_run("manual", ["AMD"], "u2")
        feed.start_run("mine", run_id=mine)
        feed.emit("news", "mine: fetching", run_id=mine)
        feed.start_run("theirs", run_id=theirs)     # newest in flight
        feed.emit("news", "theirs: fetching", run_id=theirs)

        pinned = app_module.render_progress_panel(
            1, 1500, None, None, {"run_id": mine})
        assert self._fp(pinned)[-1] == mine
        assert self._fp(pinned)[3] == "mine: fetching"

        # No run of my own (or one that already finished): the newest run
        # in flight, as before.
        unpinned = app_module.render_progress_panel(1, 1500, None, None, None)
        assert self._fp(unpinned)[-1] == theirs
        feed.finish_run("done", run_id=mine)
        after = app_module.render_progress_panel(
            1, 1500, None, None, {"run_id": mine})
        assert self._fp(after)[-1] == theirs

    def test_panel_state_is_wired(self):
        from dash._callback import GLOBAL_CALLBACK_MAP
        cb = next(v for k, v in GLOBAL_CALLBACK_MAP.items()
                  if k.startswith("..progress-feed-scroll.children"))
        assert any(st["id"] == "run-store" for st in cb.get("state", []))


class TestWiring:
    """The confirm click has ONE listener; the stages hang off run-store."""

    def _inputs(self, cb):
        return {i["id"] for i in cb.get("inputs", []) if isinstance(i["id"], str)}

    def test_confirm_button_feeds_only_the_dispatcher(self):
        from dash._callback import GLOBAL_CALLBACK_MAP
        listeners = [k for k, v in GLOBAL_CALLBACK_MAP.items()
                     if "run-confirm-btn" in self._inputs(v)]
        assert len(listeners) == 1
        assert "run-modal.is_open" in listeners[0]

    def test_stages_listen_on_run_store(self):
        from dash._callback import GLOBAL_CALLBACK_MAP
        listeners = {k for k, v in GLOBAL_CALLBACK_MAP.items()
                     if "run-store" in self._inputs(v)}
        for wanted in ("ai-analysis-store.data", "model-signals-store.data",
                       "full-analysis-requested.data", "run-dispatched.data"):
            assert any(wanted in k for k in listeners), wanted

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
        assert {"run-store", "run-dispatched", "run-started-toast",
                "history-eval-toast"} <= mounted

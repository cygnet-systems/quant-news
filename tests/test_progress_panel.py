"""The activity panel is a stepper over the run it follows.

layouts/progress_panel.py decides which run the panel pins (from the
session's run-store and the feed's active list, before the one row read)
and draws stage cells, one line per symbol and the folded log from plain
dicts. app.render_progress_panel wires that to one get_run and one
get_feed per tick and rewrites nothing while its fingerprint stands
still. run_service.update_progress and progress_service.emit_progress
carry the per-symbol states the stepper reads.
"""

import os
from datetime import datetime, timedelta, timezone

import dash
import diskcache
import pytest
from dash import html
from sqlalchemy import create_engine
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import sessionmaker

from db.models import AnalysisRun, Base, JobRun
from layouts import progress_panel as pp
from services import progress_service as prog
from services import run_service as rs

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


# A fixed clock for the pure pin tests; the callback tests run against
# the real clock because the row's finished_at is stamped by the service.
NOW = datetime(2026, 9, 2, 14, 0, tzinfo=timezone.utc)


def _ago(seconds, now=None):
    now = now or datetime.now(timezone.utc)
    return (now - timedelta(seconds=seconds)).isoformat()


def _text(node) -> str:
    """Every string in a component tree, space-joined."""
    if node is None:
        return ""
    if isinstance(node, (str, int, float)):
        return str(node)
    if isinstance(node, (list, tuple)):
        return " ".join(t for t in (_text(c) for c in node) if t)
    return _text(getattr(node, "children", None))


def _find(node, **attrs):
    """First component whose attributes all match; class names match on
    any word."""
    def _matches(n):
        for k, v in attrs.items():
            got = getattr(n, k, None)
            if k == "className":
                if not got or v not in str(got).split():
                    return False
            elif got != v:
                return False
        return True

    if hasattr(node, "children") or hasattr(node, "id"):
        if _matches(node):
            return node
        node = getattr(node, "children", None)
    if isinstance(node, (list, tuple)):
        for c in node:
            hit = _find(c, **attrs)
            if hit is not None:
                return hit
    return None


def _find_all(node, **attrs) -> list:
    out = []

    def walk(n):
        if n is None or isinstance(n, (str, int, float)):
            return
        if isinstance(n, (list, tuple)):
            for c in n:
                walk(c)
            return
        if _find(n, **attrs) is n:
            out.append(n)
        walk(getattr(n, "children", None))

    walk(node)
    return out


STAGES_FIXTURE = {
    "news": {"state": "done", "done": 2, "total": 2,
             "symbols": {"NVDA": "done", "AMD": "done"}},
    "models": {"state": "running", "done": 1, "total": 2,
               "symbols": {"NVDA": "done", "AMD": "failed"},
               "errors": {"AMD": "kronos_mini: weights missing"}},
    "research": {"state": "running", "done": 0, "total": 2,
                 "symbols": {"NVDA": "running"}},
    "synthesis": {"state": "skipped"},
}


def _run(run_id="r1", status="running", stages=None, symbols=("NVDA", "AMD"),
         finished_at=None, error=None, kind="manual"):
    return {
        "run_id": run_id, "kind": kind, "status": status, "owner_uid": "u1",
        "symbols": list(symbols), "stages": STAGES_FIXTURE if stages is None else stages,
        "counters": {}, "started_at": _ago(120), "finished_at": finished_at,
        "error": error, "active": status in rs.ACTIVE_STATUSES,
    }


class TestPinTarget:
    def test_own_run_in_flight_wins_over_a_newer_active_run(self):
        store = {"run_id": "mine", "started": _ago(30, NOW)}
        assert pp.pin_target(store, ["mine", "theirs"], NOW) == "mine"
        assert pp.pin_target(store, ["theirs", "mine"], NOW) == "mine"

    def test_own_run_recently_started_is_still_the_pin_after_it_ended(self):
        store = {"run_id": "mine", "started": _ago(600, NOW)}
        assert pp.pin_target(store, ["theirs"], NOW) == "mine"
        assert pp.pin_target(store, [], NOW) == "mine"

    def test_stale_pin_falls_to_the_newest_run_in_flight_then_the_log(self):
        store = {"run_id": "mine", "started": _ago(pp.PIN_WINDOW_S + 1, NOW)}
        assert pp.pin_target(store, ["old", "new"], NOW) == "new"
        assert pp.pin_target(store, [], NOW) is None
        assert pp.pin_target(None, ["new"], NOW) == "new"
        assert pp.pin_target({}, [], NOW) is None

    def test_a_naive_started_stamp_is_read_as_utc(self):
        naive = (NOW - timedelta(seconds=5)).replace(tzinfo=None).isoformat()
        assert pp.pin_target({"run_id": "mine", "started": naive}, [], NOW) == "mine"

    def test_pin_holds_while_open_or_for_an_hour_after(self):
        assert pp.pin_holds(_run(status="running"), [], NOW)
        assert pp.pin_holds(_run(status="done", finished_at=_ago(1800, NOW)), [], NOW)
        assert pp.pin_holds(_run(status="done", finished_at=_ago(7200, NOW)), ["r1"], NOW)
        assert not pp.pin_holds(_run(status="done", finished_at=_ago(7200, NOW)), [], NOW)
        assert not pp.pin_holds(None, [], NOW)


class TestSymbolState:
    def test_recorded_state_wins(self):
        assert pp.symbol_state(STAGES_FIXTURE, "models", "AMD") == "failed"
        assert pp.symbol_state(STAGES_FIXTURE, "research", "NVDA") == "running"

    def test_open_stage_without_a_record_is_pending(self):
        assert pp.symbol_state(STAGES_FIXTURE, "research", "AMD") == "pending"
        assert pp.symbol_state(STAGES_FIXTURE, "report", "NVDA") == "pending"

    def test_finished_stage_that_tracks_symbols_skipped_this_one(self):
        stages = {"news": {"state": "done", "symbols": {"NVDA": "done"}}}
        assert pp.symbol_state(stages, "news", "AMD") == "skipped"

    def test_untracked_stage_lends_its_terminal_state(self):
        stages = {"report": {"state": "done"}, "synthesis": {"state": "skipped"}}
        assert pp.symbol_state(stages, "report", "NVDA") == "done"
        assert pp.symbol_state(stages, "synthesis", "NVDA") == "skipped"

    def test_errors_in_pipeline_order_and_their_text(self):
        stages = {
            "synthesis": {"errors": {"AMD": "no action"}},
            "models": {"errors": {"AMD": "weights missing"}},
        }
        errors = pp.symbol_errors(stages, "AMD")
        assert errors == [("models", "weights missing"), ("synthesis", "no action")]
        assert pp.failure_text(errors) == \
            "failed: models: weights missing; synthesis: no action"
        assert pp.failure_text(errors[:1]) == "failed: weights missing"
        assert pp.failure_text([]) is None

    def test_fingerprint_moves_with_the_stages(self):
        a = pp.run_fingerprint(_run())
        moved = _run(stages={**STAGES_FIXTURE,
                             "models": {**STAGES_FIXTURE["models"], "done": 2}})
        assert a != pp.run_fingerprint(moved)
        assert a == pp.run_fingerprint(_run())
        assert pp.run_fingerprint(None) is None


class TestStepper:
    def test_stage_cells_carry_state_and_counts(self):
        node = pp.stepper(_run())
        cells = _find_all(node, className="progress-stage")
        assert [c.className.split()[-1] for c in cells] == [
            "progress-stage-done", "progress-stage-running",
            "progress-stage-running", "progress-stage-skipped",
            "progress-stage-pending"]
        assert _text(cells[1]) == "models 1/2"
        assert _text(cells[3]) == "synthesis"
        assert _text(cells[0]) == "news 2/2"

    def test_symbol_rows_have_a_glyph_per_stage(self):
        node = pp.stepper(_run())
        rows = _find_all(node, className="progress-symbol-row")
        assert [_find(r, className="progress-symbol-name").children for r in rows] \
            == ["NVDA", "AMD"]
        nvda = [g.className.split()[-1]
                for g in _find_all(rows[0], className="progress-glyph")]
        assert nvda == ["progress-glyph-done", "progress-glyph-done",
                        "progress-glyph-running", "progress-glyph-skipped",
                        "progress-glyph-pending"]
        amd = [g.className.split()[-1]
               for g in _find_all(rows[1], className="progress-glyph")]
        assert amd[1] == "progress-glyph-failed"

    def test_failed_symbol_shows_its_reason_never_a_verdict(self):
        node = pp.stepper(_run())
        rows = _find_all(node, className="progress-symbol-row")
        assert _find(rows[0], className="progress-symbol-error") is None
        err = _find(rows[1], className="progress-symbol-error")
        assert err.children == "failed: kronos_mini: weights missing"
        assert "HOLD" not in _text(node)

    def test_caption_names_the_run_and_a_failure(self):
        ok = pp.stepper(_run())
        assert _find(ok, className="progress-run-caption").children == \
            "manual run · running"
        bad = pp.stepper(_run(status="failed", error="worker pid 4 died\nmore"))
        assert _find(bad, className="progress-run-caption").children == \
            "manual run · failed · worker pid 4 died"

    def test_body_folds_the_log_under_details(self):
        events = [{"t": 1.0, "stage": "news", "message": "NVDA: fetched 3"}]
        body = pp.body(_run(), events)
        assert body[0].className == "progress-stepper"
        details = body[1]
        assert isinstance(details, html.Details)
        assert isinstance(details.children[0], html.Summary)
        assert details.children[0].children == "Details"
        lines = details.children[1]
        assert lines.className == "progress-feed-lines"
        assert _text(lines).endswith("NVDA: fetched 3")
        assert _find(lines, className="progress-line") is not None

    def test_idle_body_is_the_bare_log(self):
        events = [{"t": 1.0, "stage": "done", "message": "Pipeline complete"}]
        (lines,) = pp.body(None, events)
        assert lines.className == "progress-feed-lines"
        assert _find(lines, className="progress-line-done") is not None

    def test_header_icon(self):
        assert pp.header_icon(True).className == "progress-spinner"
        assert "progress-header-done" in pp.header_icon(False, _run(status="done")).className
        assert "progress-header-failed" in pp.header_icon(
            False, _run(status="failed")).className


class TestPerSymbolProgress:
    def test_update_progress_merges_symbol_states_beside_the_counts(self, db):
        run_id = rs.create_run("manual", ["NVDA", "AMD"], "u1")
        rs.update_progress(run_id, "models", state="running", done=0, total=2)
        rs.update_progress(run_id, "models", symbol="NVDA", state="running")
        rs.update_progress(run_id, "models", symbol="NVDA", state="done")
        rs.update_progress(run_id, "models", symbol="AMD", state="failed",
                           error="kronos_mini: weights missing\n  Traceback")
        rs.update_progress(run_id, "models", done=1, total=2)
        models = rs.get_run(run_id)["stages"]["models"]
        # The stage's own state is not the symbol's: one failure does not
        # fail the stage.
        assert models["state"] == "running"
        assert (models["done"], models["total"]) == (1, 2)
        assert models["symbols"] == {"NVDA": "done", "AMD": "failed"}
        assert models["errors"] == {"AMD": "kronos_mini: weights missing"}

    def test_stage_level_error_is_kept_as_its_first_line(self, db):
        run_id = rs.create_run("manual", ["NVDA"], "u1")
        rs.update_progress(run_id, "report", state="failed",
                           error="\nprovider timeout\nafter 30s")
        report = rs.get_run(run_id)["stages"]["report"]
        assert report == {"state": "failed", "error": "provider timeout"}

    def test_emit_progress_carries_symbol_and_reason_to_feed_and_row(self, db, feed):
        run_id = rs.create_run("manual", ["NVDA"], "u1")
        feed.start_run("run", run_id=run_id)
        feed.emit_progress("models", symbol="NVDA", state="failed",
                           error="no price data\nsecond line", run_id=run_id)
        event = feed.get_feed(run_id)["events"][-1]
        assert event["message"] == "models NVDA failed: no price data"
        assert (event["symbol"], event["error"]) == ("NVDA", "no price data")
        stage = rs.get_run(run_id)["stages"]["models"]
        assert stage["symbols"] == {"NVDA": "failed"}
        assert stage["errors"] == {"NVDA": "no price data"}
        # Plain progress events keep their old shape.
        feed.emit_progress("models", done=1, total=1, run_id=run_id)
        plain = feed.get_feed(run_id)["events"][-1]
        assert "symbol" not in plain and "error" not in plain


BODY, COUNT, ICON, SNAP, FP = range(5)


def _panel(run_store=None, last_snap=None, last_fp=None):
    return app_module.render_progress_panel(1, last_snap, last_fp, run_store)


@pytest.fixture
def pinned(db, feed):
    """A run of the viewer's in flight, with per-symbol progress."""
    run_id = rs.create_run("manual", ["NVDA", "AMD"], "u1")
    feed.start_run("Run analysis", run_id=run_id)
    feed.emit("news", "NVDA: fetched 3 articles", run_id=run_id)
    feed.emit_progress("news", done=2, total=2, state="done", run_id=run_id)
    feed.emit_progress("models", done=1, total=2, state="running", run_id=run_id)
    feed.emit_progress("models", symbol="NVDA", state="done", run_id=run_id)
    feed.emit_progress("models", symbol="AMD", state="failed",
                       error="kronos_mini: weights missing\nTraceback",
                       run_id=run_id)
    return run_id


class TestPanelCallback:
    def test_stepper_from_the_row_and_the_log_under_details(self, pinned):
        out = _panel({"run_id": pinned, "started": _ago(60)})
        stepper, details = out[BODY]
        cells = _find_all(stepper, className="progress-stage")
        assert _text(cells[0]) == "news 2/2"
        assert _text(cells[1]) == "models 1/2"
        rows = _find_all(stepper, className="progress-symbol-row")
        amd = _find(rows[1], className="progress-symbol-error")
        assert amd.children == "failed: kronos_mini: weights missing"
        assert isinstance(details, html.Details)
        assert "NVDA: fetched 3 articles" in _text(details)
        assert "Run analysis" in _text(details)
        assert out[ICON].className == "progress-spinner"
        assert out[FP][-1] == pinned
        assert out[COUNT].endswith("events")

    def test_unchanged_tick_is_no_update(self, pinned):
        store = {"run_id": pinned, "started": _ago(60)}
        first = _panel(store)
        snap = None if first[SNAP] is dash.no_update else first[SNAP]
        again = _panel(store, last_snap=snap, last_fp=first[FP])
        assert all(v is dash.no_update for v in again)

    def test_symbol_progress_alone_moves_the_fingerprint(self, pinned, feed, db,
                                                          monkeypatch):
        store = {"run_id": pinned, "started": _ago(60)}
        first = _panel(store)
        # A row write with no feed line (another process updating the row).
        rs.update_progress(pinned, "models", symbol="AMD", state="done")
        moved = _panel(store, last_snap=first[SNAP], last_fp=first[FP])
        assert moved[FP] != first[FP]
        rows = _find_all(moved[BODY][0], className="progress-symbol-row")
        assert _find(rows[1], className="progress-symbol-error") is None

    def test_one_row_read_and_one_feed_read_per_tick(self, pinned, monkeypatch):
        reads = {"run": 0, "feed": 0}
        real_get, real_feed = rs.get_run, prog.get_feed
        monkeypatch.setattr(rs, "get_run", lambda rid: (
            reads.__setitem__("run", reads["run"] + 1) or real_get(rid)))
        monkeypatch.setattr(prog, "get_feed", lambda rid=None: (
            reads.__setitem__("feed", reads["feed"] + 1) or real_feed(rid)))
        _panel({"run_id": pinned, "started": _ago(60)})
        assert reads == {"run": 1, "feed": 1}
        # No pin and nothing in flight: the log alone, no row read at all.
        prog.finish_run("done", run_id=pinned)
        reads.update(run=0, feed=0)
        _panel(None)
        assert reads == {"run": 0, "feed": 1}

    def test_finished_run_stays_pinned_for_an_hour(self, pinned, feed):
        rs.set_status(pinned, "done")
        feed.finish_run("Pipeline complete", run_id=pinned)
        out = _panel({"run_id": pinned, "started": _ago(600)})
        stepper = out[BODY][0]
        assert stepper.className == "progress-stepper"
        assert _find(stepper, className="progress-run-caption").children == \
            "manual run · done"
        assert "progress-header-done" in out[ICON].className

    def test_failed_run_shows_the_cross_and_the_reason(self, pinned, feed):
        rs.set_status(pinned, "failed", error="worker pid 4242 died\nTraceback")
        feed.finish_run("Run failed", run_id=pinned)
        out = _panel({"run_id": pinned, "started": _ago(600)})
        assert "progress-header-failed" in out[ICON].className
        assert _find(out[BODY][0], className="progress-run-caption").children == \
            "manual run · failed · worker pid 4242 died"

    def test_stale_pin_falls_back_to_the_rolling_log(self, pinned, feed):
        rs.set_status(pinned, "done")
        feed.finish_run("Pipeline complete", run_id=pinned)
        out = _panel({"run_id": pinned, "started": _ago(pp.PIN_WINDOW_S + 60)})
        (lines,) = out[BODY]
        assert lines.className == "progress-feed-lines"
        assert "Pipeline complete" in _text(lines)
        assert "progress-header-done" in out[ICON].className

    def test_newest_run_in_flight_without_a_pin(self, pinned, db, feed):
        theirs = rs.create_run("scheduled", ["TSLA"], None)
        feed.start_run("daily", run_id=theirs)
        feed.emit_progress("news", done=0, total=1, state="running", run_id=theirs)
        out = _panel(None)
        assert out[FP][-1] == theirs
        assert _find(out[BODY][0], className="progress-run-caption").children == \
            "scheduled run · running"
        assert [r.children[0].children for r in
                _find_all(out[BODY][0], className="progress-symbol-row")] == ["TSLA"]

    def test_queued_run_with_no_lines_yet_still_draws_the_stepper(self, db, feed):
        run_id = rs.create_run("manual", ["NVDA"], "u1")
        feed.mark_run_pending(run_id)
        out = _panel({"run_id": run_id, "started": _ago(5)})
        stepper, details = out[BODY]
        assert _find(stepper, className="progress-run-caption").children == \
            "manual run · queued"
        glyphs = _find_all(stepper, className="progress-glyph")
        assert all(g.className.endswith("progress-glyph-pending") for g in glyphs)
        assert out[COUNT] == "0 events"

    def test_nothing_anywhere_is_no_update(self, db, feed):
        assert all(v is dash.no_update for v in _panel(None))

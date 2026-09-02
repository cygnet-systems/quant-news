"""The topbar run pill and the poll it rides on.

The pill is decided from run dicts by layouts/run_pill.py (precedence and
text patterns pinned here without a browser), rendered by app.render_run_pill
on every progress-interval tick with run_service stubbed (no DB, no model,
no LLM), rewritten only when its fingerprint moves, and it owns the poll
rate: fast while any run is in flight, idle otherwise, whatever the panel
state. A click opens the panel on that run; its cancel is the dialog's
cancel.
"""

import asyncio
import os
from datetime import datetime, timedelta, timezone
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
from layouts import run_pill as rp
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


NOW = datetime(2026, 9, 2, 14, 0, tzinfo=timezone.utc)


def _run(run_id="r1", kind="manual", status="running", owner="u1",
         symbols=("NVDA", "AMD"), stages=None, estimate_s=None,
         started_ago_s=0, error=None):
    return {
        "run_id": run_id, "kind": kind, "status": status, "owner_uid": owner,
        "preset": "standard", "config": {"scope": "full"},
        "symbols": list(symbols), "stages": stages or {}, "counters": {},
        "estimate_s": estimate_s,
        "started_at": (NOW - timedelta(seconds=started_ago_s)).isoformat(),
        "finished_at": None, "error": error, "job_run_id": None,
        "is_public": True, "active": status in rs.ACTIVE_STATUSES,
    }


def _view(active=(), latest=None, owner="u1", run_store=None, seen=None):
    return rp.pill_view(list(active), latest, owner, run_store, seen, now=NOW)


class TestTextPatterns:
    def test_running_own_run(self):
        run = _run(stages={"news": {"state": "done", "done": 2, "total": 2},
                           "models": {"state": "running", "done": 1, "total": 2}},
                   estimate_s=240, started_ago_s=60)
        view = _view([run])
        assert view["label"] == "Running · NVDA, AMD · models 1/2 · ~3 min"
        assert view["state"] == "running"
        assert view["className"] == "run-pill run-pill-running"
        assert view["cancel"] is True
        assert view["href"] is None
        assert view["fp"] == ["r1", "running", "models", 1, 2, "~3 min"]

    def test_symbols_collapse_after_three(self):
        assert rp.symbols_label(["NVDA", "AMD", "TSLA", "AAPL", "MSFT"]) \
            == "NVDA, AMD, TSLA +2"
        assert rp.symbols_label(["NVDA", "AMD", "TSLA"]) == "NVDA, AMD, TSLA"
        run = _run(symbols=["NVDA", "AMD", "TSLA", "AAPL", "MSFT"])
        assert _view([run])["label"] == "Running · NVDA, AMD, TSLA +2"

    def test_queued_reads_starting(self):
        view = _view([_run(status="queued", estimate_s=100)])
        assert view["label"] == "Starting · NVDA, AMD · ~2 min"

    def test_eta_finishing_up_under_thirty_seconds_and_past_estimate(self):
        assert rp.eta_label(60, (NOW - timedelta(seconds=45)).isoformat(), NOW) \
            == "finishing up"
        assert rp.eta_label(60, (NOW - timedelta(seconds=600)).isoformat(), NOW) \
            == "finishing up"
        assert rp.eta_label(240, (NOW - timedelta(seconds=60)).isoformat(), NOW) \
            == "~3 min"
        # Never "~0 min": a minute is the floor above the finishing band.
        assert rp.eta_label(40, NOW.isoformat(), NOW) == "~1 min"

    def test_eta_omitted_without_an_estimate(self):
        run = _run(stages={"models": {"state": "running", "done": 0, "total": 1}})
        assert _view([run])["label"] == "Running · NVDA, AMD · models 0/1"
        assert rp.eta_label(None, NOW.isoformat(), NOW) is None

    def test_naive_started_at_is_read_as_utc(self):
        # SQLite hands back naive stamps for the timestamptz column.
        naive = (NOW - timedelta(seconds=60)).replace(tzinfo=None).isoformat()
        assert rp.eta_label(240, naive, NOW) == "~3 min"

    def test_stage_is_the_newest_running_in_pipeline_order(self):
        # A full run's report and models run side by side; the later stage
        # is named. Dict order is irrelevant (JSONB does not keep it).
        stages = {"report": {"state": "running", "done": 0, "total": 2},
                  "models": {"state": "running", "done": 1, "total": 2},
                  "news": {"state": "done", "done": 2, "total": 2}}
        assert rp.current_stage(stages) == ("report", 0, 2)
        assert rp.current_stage({"news": {"state": "done"}}) is None
        # A running stage without counts is named without them.
        run = _run(stages={"synthesis": {"state": "running"}})
        assert _view([run])["label"] == "Running · NVDA, AMD · synthesis"

    def test_scheduled_run(self):
        run = _run(kind="scheduled", owner=None, symbols=["A"] * 20,
                   stages={"news": {"state": "running", "done": 12, "total": 20}})
        view = _view([run])
        assert view["label"] == "Scheduled · daily analysis · news 12/20"
        assert view["state"] == "scheduled"
        assert view["cancel"] is False
        assert view["href"] is None

    def test_ready_links_to_the_run_page(self):
        view = _view([], latest=_run(status="done"))
        assert view["label"] == "Ready · NVDA, AMD · View"
        assert view["text"] == "Ready · NVDA, AMD"
        assert view["href"] == "/runs/r1?open=first"
        assert view["state"] == "ready"
        assert view["cancel"] is False

    def test_failed_shows_the_first_error_line(self):
        run = _run(status="failed", error="worker pid 4242 died\nTraceback ...")
        view = _view([], latest=run)
        assert view["label"] == "Failed · worker pid 4242 died · View"
        assert view["title"] == "worker pid 4242 died"
        assert view["href"] == "/runs/r1?open=first"
        assert view["className"] == "run-pill run-pill-failed"
        assert _view([], latest=_run(status="failed", error=None))["text"] \
            == "Failed · unknown error"

    def test_cancelled_is_hidden(self):
        assert _view([], latest=_run(status="cancelled")) is None


class TestPrecedence:
    def test_own_active_run_beats_a_scheduled_one(self):
        sched = _run("s1", kind="scheduled", owner=None)
        mine = _run("m1")
        assert _view([sched, mine])["run_id"] == "m1"
        assert _view([mine, sched])["run_id"] == "m1"

    def test_newest_own_active_run_wins(self):
        assert _view([_run("m1"), _run("m2")])["run_id"] == "m2"

    def test_scheduled_beats_another_owners_run(self):
        theirs = _run("t1", owner="u2")
        sched = _run("s1", kind="scheduled", owner=None)
        assert _view([theirs, sched])["run_id"] == "s1"
        assert _view([theirs, sched], latest=_run("d1", status="done",
                                                  owner="u2"))["run_id"] == "s1"

    def test_own_finished_run_beats_a_scheduled_one_until_seen(self):
        # The daily job runs for tens of minutes; the viewer's own result
        # must not wait behind it. Once the run page was visited the
        # scheduled run shows again.
        sched = _run("s1", kind="scheduled", owner=None)
        done = _run("d1", status="done")
        assert _view([sched], latest=done)["state"] == "ready"
        failed = _run("f1", status="failed", error="boom")
        assert _view([sched], latest=failed)["state"] == "failed"
        assert _view([sched], latest=done, seen={"run_id": "d1"})["run_id"] == "s1"
        # An own row in flight that the active read missed still wins.
        assert _view([sched], latest=_run("q1", status="queued"))["run_id"] == "q1"

    def test_finished_view_ignores_what_the_pill_shows(self):
        done = _run("d1", status="done")
        assert rp.finished_view(done, "u1")["state"] == "ready"
        assert rp.finished_view(done, "u1", seen={"run_id": "d1"}) is None
        assert rp.finished_view(done, "u2") is None
        assert rp.finished_view(_run("q1", status="queued"), "u1") is None
        assert rp.finished_view(_run("c1", status="cancelled"), "u1") is None
        assert rp.finished_view(None, "u1") is None
        assert rp.done_toast(rp.finished_view(done, "u1"), None)["run_id"] == "d1"

    def test_another_owners_run_alone_shows_nothing(self):
        assert _view([_run("t1", owner="u2")]) is None
        # Anonymous compares as '', so an anonymous viewer sees anonymous runs.
        assert _view([_run("a1", owner=None)], owner=None)["run_id"] == "a1"
        assert _view([_run("a1", owner=None)], owner="u1") is None

    def test_the_run_this_session_confirmed_counts_as_own(self):
        # The row's owner may differ from the request's (attribution edge),
        # but the session that confirmed it is following it.
        run = _run("t1", owner="u2")
        view = _view([run], run_store={"run_id": "t1"})
        assert view is not None and view["cancel"] is True

    def test_finished_run_shows_until_seen(self):
        latest = _run("d1", status="done")
        assert _view([], latest=latest)["run_id"] == "d1"
        assert _view([], latest=latest, seen={"run_id": "d1"}) is None
        assert _view([], latest=latest, seen={"run_id": "other"}) is not None

    def test_finished_run_of_another_owner_is_not_shown(self):
        assert _view([], latest=_run("d1", status="done", owner="u2")) is None

    def test_row_open_but_missing_from_active_is_still_running(self):
        # Created between the two reads: not in active_runs yet, active on
        # the row. It is not a finished run.
        view = _view([], latest=_run("q1", status="queued"))
        assert view["state"] == "running"

    def test_nothing_is_hidden(self):
        assert _view([], latest=None) is None


class TestChildren:
    def _ids(self, children):
        return [getattr(c, "id", None) for c in children]

    def test_own_running_has_dot_text_and_cancel(self):
        run = _run(estimate_s=120)
        children = rp.pill_children(_view([run]))
        assert self._ids(children) == [None, "run-pill-text", "run-pill-cancel"]
        assert children[0].className == "run-pill-dot"
        assert children[1].children.startswith("Running · NVDA, AMD")

    def test_scheduled_has_calendar_glyph_and_no_cancel_or_link(self):
        run = _run(kind="scheduled", owner=None)
        children = rp.pill_children(_view([run]))
        assert self._ids(children) == [None, "run-pill-text"]
        assert "bi-calendar3" in children[0].className

    def test_ready_has_a_view_link_and_no_cancel(self):
        children = rp.pill_children(_view([], latest=_run(status="done")))
        assert self._ids(children) == [None, "run-pill-text", "run-pill-link"]
        assert children[2].href == "/runs/r1?open=first"
        assert children[2].children == "View"


@pytest.fixture
def runs(monkeypatch):
    """run_service stubbed: what active_runs / list_runs answer, and the
    calls list_runs saw."""
    state = SimpleNamespace(active=[], latest=None, list_calls=[], fail=False)

    def _active():
        if state.fail:
            raise RuntimeError("db down")
        return list(state.active)

    def _list(limit=50, kind=None, owner_uid=None):
        state.list_calls.append({"limit": limit, "kind": kind,
                                 "owner_uid": owner_uid})
        return [state.latest] if state.latest else []

    monkeypatch.setattr(rs, "active_runs", _active)
    monkeypatch.setattr(rs, "list_runs", _list)
    monkeypatch.setattr(app_module, "_run_owner_uid", lambda: "u1")
    return state


def _tick(run_store=None, seen=None, last_fp=None, interval=None, notified=None):
    interval = app_module._PROGRESS_POLL_IDLE_MS if interval is None else interval
    return asyncio.run(app_module.render_run_pill(
        1, run_store, seen, last_fp, interval, notified))


HIDDEN, CLASS, CHILDREN, FP, INTERVAL = range(5)
TOAST_OPEN, TOAST_HEADER, TOAST_BODY, TOAST_ICON, NOTIFIED = range(5, 10)


class TestRenderRunPill:
    def test_own_active_run_renders_with_cancel_and_snaps_the_poll(self, runs):
        runs.active = [_run(estimate_s=200)]
        out = _tick()
        assert out[HIDDEN] is False
        assert out[CLASS] == "run-pill run-pill-running"
        ids = [getattr(c, "id", None) for c in out[CHILDREN]]
        assert "run-pill-cancel" in ids
        assert out[FP]["fp"][0] == "r1"
        assert out[FP]["pin"]["run_id"] == "r1"
        assert out[FP]["pin"]["kind"] == "manual"
        assert out[INTERVAL] == app_module._PROGRESS_POLL_ACTIVE_MS
        # The finished-run read is skipped while an own run is in flight.
        assert runs.list_calls == []

    def test_scheduled_run_still_reads_the_finished_run(self, runs):
        runs.active = [_run("s1", kind="scheduled", owner=None,
                            stages={"news": {"state": "running",
                                             "done": 3, "total": 20}})]
        out = _tick()
        assert out[CHILDREN][1].children == "Scheduled · daily analysis · news 3/20"
        assert runs.list_calls == [{"limit": 1, "kind": "manual", "owner_uid": "u1"}]
        # The viewer's own result outranks the job; seen, the job is back.
        runs.latest = _run("d1", status="done")
        out = _tick()
        assert out[CLASS] == "run-pill run-pill-ready"
        out = _tick(seen={"run_id": "d1"})
        assert out[CLASS] == "run-pill run-pill-scheduled"

    def test_finished_run_reads_this_owners_newest_manual_run(self, runs):
        runs.latest = _run("d1", status="done")
        out = _tick(interval=app_module._PROGRESS_POLL_ACTIVE_MS)
        assert out[HIDDEN] is False
        assert out[CLASS] == "run-pill run-pill-ready"
        link = next(c for c in out[CHILDREN] if getattr(c, "id", None) == "run-pill-link")
        assert link.href == "/runs/d1?open=first"
        assert runs.list_calls == [{"limit": 1, "kind": "manual", "owner_uid": "u1"}]
        # Nothing in flight: back to the idle rate.
        assert out[INTERVAL] == app_module._PROGRESS_POLL_IDLE_MS

    def test_anonymous_owner_reads_the_anonymous_bucket(self, runs, monkeypatch):
        monkeypatch.setattr(app_module, "_run_owner_uid", lambda: None)
        _tick()
        assert runs.list_calls[0]["owner_uid"] == ""

    def test_unchanged_fingerprint_is_no_update(self, runs):
        runs.active = [_run(estimate_s=200)]
        first = _tick()
        again = _tick(last_fp=first[FP], interval=first[INTERVAL])
        assert all(v is dash.no_update for v in again)
        # The pill is unchanged but the rate is off: only the rate is sent.
        rate_only = _tick(last_fp=first[FP], interval=app_module._PROGRESS_POLL_IDLE_MS)
        assert rate_only[:4] == (dash.no_update,) * 4
        assert rate_only[INTERVAL] == app_module._PROGRESS_POLL_ACTIVE_MS

    def test_progress_moves_the_fingerprint(self, runs):
        runs.active = [_run(stages={"models": {"state": "running",
                                               "done": 0, "total": 2}})]
        first = _tick()
        runs.active[0]["stages"]["models"]["done"] = 1
        moved = _tick(last_fp=first[FP], interval=first[INTERVAL])
        assert moved[CHILDREN][1].children == "Running · NVDA, AMD · models 1/2"

    def test_hidden_once_then_no_update(self, runs):
        runs.active = [_run(estimate_s=200)]
        shown = _tick()
        runs.active = []
        hidden = _tick(last_fp=shown[FP], interval=shown[INTERVAL])
        assert hidden[HIDDEN] is True
        assert hidden[CHILDREN] == []
        assert hidden[FP] is None
        assert hidden[INTERVAL] == app_module._PROGRESS_POLL_IDLE_MS
        idle = _tick(last_fp=hidden[FP], interval=hidden[INTERVAL])
        assert all(v is dash.no_update for v in idle)

    def test_seen_store_clears_the_ready_pill(self, runs):
        runs.latest = _run("d1", status="done")
        shown = _tick()
        assert shown[HIDDEN] is False
        assert _tick(seen={"run_id": "d1"}, last_fp=shown[FP])[HIDDEN] is True

    def test_db_failure_leaves_the_pill_alone(self, runs):
        runs.fail = True
        with pytest.raises(PreventUpdate):
            _tick()


class TestDoneToast:
    """The pill's Ready/Failed state announced once, on the tick that first
    sees it, then never again for that run: the notified store is the
    guard, not the fingerprint."""

    def test_ready_opens_report_ready_with_a_view_link_once(self, runs):
        runs.latest = _run("d1", status="done", symbols=("NVDA", "AMD"))
        out = _tick()
        assert out[TOAST_OPEN] is True
        assert out[TOAST_HEADER] == "Report ready"
        assert out[TOAST_ICON] == "success"
        text, link = out[TOAST_BODY]
        assert text.children == "NVDA, AMD"
        assert link.children == "View" and link.href == "/runs/d1?open=first"
        assert out[NOTIFIED] == {"run_id": "d1"}
        # Next tick: same pill (no_update) and, with the run recorded as
        # announced, no toast either.
        again = _tick(last_fp=out[FP], interval=app_module._PROGRESS_POLL_IDLE_MS,
                      notified=out[NOTIFIED])
        assert all(v is dash.no_update for v in again)

    def test_announces_even_when_the_pill_did_not_move(self, runs):
        # A reload: the pill fingerprint is restored but the browser was
        # never told (notified store empty). Once, then guarded.
        runs.latest = _run("d1", status="done")
        first = _tick()
        idle = app_module._PROGRESS_POLL_IDLE_MS
        repeat = _tick(last_fp=first[FP], interval=idle, notified=None)
        assert repeat[:4] == (dash.no_update,) * 4
        assert repeat[TOAST_OPEN] is True and repeat[NOTIFIED] == {"run_id": "d1"}
        guarded = _tick(last_fp=first[FP], interval=idle, notified={"run_id": "d1"})
        assert all(v is dash.no_update for v in guarded)

    def test_failed_names_the_first_error_line(self, runs):
        runs.latest = _run("f1", status="failed",
                           error="provider timeout\nTraceback (most recent)")
        out = _tick()
        assert out[TOAST_HEADER] == "Run failed"
        assert out[TOAST_ICON] == "danger"
        text, link = out[TOAST_BODY]
        assert text.children == "provider timeout"
        assert link.href == "/runs/f1?open=first"
        assert out[NOTIFIED] == {"run_id": "f1"}

    def test_a_new_run_is_announced_after_an_old_one(self, runs):
        runs.latest = _run("d1", status="done")
        first = _tick()
        runs.latest = _run("d2", status="done")
        out = _tick(last_fp=first[FP], interval=app_module._PROGRESS_POLL_IDLE_MS,
                    notified=first[NOTIFIED])
        assert out[TOAST_OPEN] is True and out[NOTIFIED] == {"run_id": "d2"}

    def test_running_seen_and_hidden_never_toast(self, runs):
        runs.active = [_run(estimate_s=100)]
        assert _tick()[TOAST_OPEN] is dash.no_update
        runs.active = []
        runs.latest = _run("d1", status="done")
        assert _tick(seen={"run_id": "d1"})[TOAST_OPEN] is dash.no_update
        runs.latest = None
        assert _tick()[TOAST_OPEN] is dash.no_update
        assert rp.done_toast(None, None) is None

    def test_announced_while_the_daily_job_is_in_flight(self, runs):
        # The job takes tens of minutes; a manual run finishing inside that
        # window is announced on the first tick, and a newer own run
        # confirmed before the job ends does not lose the announcement.
        runs.active = [_run("s1", kind="scheduled", owner=None)]
        runs.latest = _run("d1", status="done")
        out = _tick()
        assert out[TOAST_OPEN] is True and out[NOTIFIED] == {"run_id": "d1"}
        assert out[CLASS] == "run-pill run-pill-ready"
        again = _tick(last_fp=out[FP], interval=app_module._PROGRESS_POLL_ACTIVE_MS,
                      notified=out[NOTIFIED])
        assert all(v is dash.no_update for v in again)
        # Seen: the pill goes back to the job, the toast stays quiet.
        seen = _tick(last_fp=out[FP], interval=app_module._PROGRESS_POLL_ACTIVE_MS,
                     seen={"run_id": "d1"}, notified=out[NOTIFIED])
        assert seen[CLASS] == "run-pill run-pill-scheduled"
        assert seen[TOAST_OPEN] is dash.no_update
        # A failed one under the same job.
        runs.latest = _run("f1", status="failed", error="provider timeout")
        out = _tick(notified={"run_id": "d1"})
        assert out[TOAST_HEADER] == "Run failed" and out[NOTIFIED] == {"run_id": "f1"}

    def test_another_owners_finished_run_is_not_announced(self, runs):
        runs.latest = _run("d1", status="done", owner="u2")
        out = _tick()
        # Hidden is the pill's state already (None fingerprint both sides).
        assert out[HIDDEN] in (True, dash.no_update)
        assert out[CHILDREN] in ([], dash.no_update)
        assert out[TOAST_OPEN] is dash.no_update and out[NOTIFIED] is dash.no_update


class TestPollLifetime:
    def test_rate_follows_the_rows_with_the_panel_closed(self, runs):
        # A closed panel used to disable the interval; now the rate is the
        # pill's and the panel state is not even read.
        runs.active = [_run()]
        out = _tick(interval=app_module._PROGRESS_POLL_IDLE_MS)
        assert out[INTERVAL] == app_module._PROGRESS_POLL_ACTIVE_MS
        layout = app_module.apply_progress_panel_state({"closed": True, "mode": "normal"})
        assert len(layout) == 5
        assert layout[0] == {"display": "none"}

    def test_nobody_disables_the_interval(self):
        from dash._callback import GLOBAL_CALLBACK_MAP
        assert [k for k in GLOBAL_CALLBACK_MAP
                if "progress-interval.disabled" in k] == []

    def test_one_owner_of_the_interval_rate(self):
        # Everyone else (confirm and retry acknowledgements) writes it with
        # allow_duplicate, which Dash marks with an @hash suffix.
        from dash._callback import GLOBAL_CALLBACK_MAP
        plain = [k for k in GLOBAL_CALLBACK_MAP
                 if "progress-interval.interval.." in k
                 or k.endswith("progress-interval.interval")]
        assert len(plain) == 1
        assert "run-pill.hidden" in plain[0]
        cb = GLOBAL_CALLBACK_MAP[plain[0]]
        assert [i["id"] for i in cb["inputs"]] == ["progress-interval"]
        assert asyncio.iscoroutinefunction(app_module.render_run_pill)

    def test_panel_no_longer_writes_the_rate(self):
        from dash._callback import GLOBAL_CALLBACK_MAP
        panel = next(k for k in GLOBAL_CALLBACK_MAP
                     if k.startswith("..progress-feed-scroll.children"))
        assert "progress-interval.interval" not in panel


class TestPillClick:
    def test_click_opens_the_panel_on_the_pills_run(self):
        fp = {"fp": ["s1"], "pin": {"run_id": "s1", "kind": "scheduled",
                                    "symbols": ["A"], "scope": "full"}}
        panel, store = app_module.open_run_from_pill(
            1, fp, {"run_id": "mine"}, {"mode": "normal", "closed": True})
        assert panel == {"mode": "normal", "closed": False}
        assert store == fp["pin"]

    def test_click_on_own_run_keeps_run_store(self):
        fp = {"fp": ["m1"], "pin": {"run_id": "m1"}}
        panel, store = app_module.open_run_from_pill(1, fp, {"run_id": "m1"}, None)
        assert panel["closed"] is False
        assert store is dash.no_update
        # No pin at all (pill hidden, stale click): still just opens.
        panel, store = app_module.open_run_from_pill(1, None, None, None)
        assert panel["closed"] is False and store is dash.no_update

    def test_no_click_is_no_call(self):
        with pytest.raises(PreventUpdate):
            app_module.open_run_from_pill(None, None, None, None)
        with pytest.raises(PreventUpdate):
            app_module.open_run_from_pill(0, None, None, None)

    def test_retry_refuses_a_pinned_scheduled_run(self, db, feed, monkeypatch):
        monkeypatch.setattr(app_module, "_run_owner_uid", lambda: "u1")
        sched = rs.create_run("scheduled", ["NVDA"], None)
        rs.set_status(sched, "done")
        out = app_module.retry_run(1, {"run_id": sched}, None)
        assert out[2] is True
        assert "scheduled" in out[3]
        assert rs.list_runs(kind="manual") == []


class TestPillCancel:
    def test_cancel_from_the_pill_is_the_dialogs_cancel(self, db, feed, monkeypatch):
        monkeypatch.setattr(app_module, "_run_owner_uid", lambda: "u1")
        run_id = rs.create_run("manual", ["NVDA"], "u1")
        feed.mark_run_pending(run_id)

        hidden, fp, flag = app_module.cancel_run_from_pill(1)

        assert (hidden, fp, flag) == (True, None, False)
        assert rs.get_run(run_id)["status"] == "cancelled"
        assert rs.active_run_for("u1") is None
        assert run_id not in feed._active_runs()
        # The next tick shows nothing for it: a cancelled run is hidden.
        assert rp.pill_view([], rs.get_run(run_id), "u1") is None

    def test_cancel_with_nothing_in_flight(self, db, feed, monkeypatch):
        monkeypatch.setattr(app_module, "_run_owner_uid", lambda: "u1")
        with pytest.raises(PreventUpdate):
            app_module.cancel_run_from_pill(1)
        with pytest.raises(PreventUpdate):
            app_module.cancel_run_from_pill(None)

    def test_both_buttons_share_the_helper(self, db, feed, monkeypatch):
        monkeypatch.setattr(app_module, "_run_owner_uid", lambda: "u1")
        run_id = rs.create_run("manual", ["NVDA"], "u1")
        feed.mark_run_pending(run_id)
        msg, flag = app_module.cancel_active_run(1)
        assert "cancelled" in msg and flag is False
        assert rs.get_run(run_id)["status"] == "cancelled"
        msg, flag = app_module.cancel_active_run(1)
        assert "No run in progress" in msg and flag is dash.no_update


class TestWiring:
    def _ids(self, node, acc):
        i = getattr(node, "id", None)
        if isinstance(i, str):
            acc.add(i)
        ch = getattr(node, "children", None)
        for c in (ch if isinstance(ch, (list, tuple)) else [ch]):
            if c is not None and (hasattr(c, "children") or hasattr(c, "id")):
                self._ids(c, acc)
        return acc

    def test_pill_and_stores_are_mounted_and_the_badge_is_gone(self):
        from layouts.main_layout import create_layout
        from layouts.nav import create_topbar

        mounted = self._ids(create_layout(), set())
        assert {"run-pill", "run-pill-fp", "run-seen-store", "run-done-toast",
                "run-notified-store"} <= mounted
        assert "prediction-running-indicator" not in mounted
        topbar = self._ids(create_topbar(), set())
        assert {"run-pill", "run-analysis-btn"} <= topbar

    def test_pill_starts_hidden_and_clickable(self):
        from layouts.nav import create_topbar

        def find(node):
            if getattr(node, "id", None) == "run-pill":
                return node
            ch = getattr(node, "children", None)
            for c in (ch if isinstance(ch, (list, tuple)) else [ch]):
                if c is not None and (hasattr(c, "children") or hasattr(c, "id")):
                    hit = find(c)
                    if hit is not None:
                        return hit
            return None

        pill = find(create_topbar())
        assert pill.hidden is True
        assert pill.className == "run-pill"

    def test_no_callback_names_the_old_badge_or_a_running_spec(self):
        from dash._callback import GLOBAL_CALLBACK_LIST, GLOBAL_CALLBACK_MAP
        assert [k for k in GLOBAL_CALLBACK_MAP if "prediction-running" in k] == []
        models = next(c for c in GLOBAL_CALLBACK_LIST
                      if str(c["output"]) == "model-signals-store.data")
        assert not models.get("running")

    def test_done_toast_is_the_pill_callbacks_and_persists(self):
        from dash._callback import GLOBAL_CALLBACK_MAP
        from layouts.main_layout import create_layout
        writers = [k for k in GLOBAL_CALLBACK_MAP if "run-done-toast.is_open" in k]
        assert len(writers) == 1 and "run-pill.hidden" in writers[0]
        cb = GLOBAL_CALLBACK_MAP[writers[0]]
        assert any(st["id"] == "run-notified-store" for st in cb["state"])
        assert "run-notified-store.data" in writers[0]
        assert asyncio.iscoroutinefunction(app_module.render_run_pill)

        def find(node):
            if getattr(node, "id", None) == "run-done-toast":
                return node
            ch = getattr(node, "children", None)
            for c in (ch if isinstance(ch, (list, tuple)) else [ch]):
                if c is not None and (hasattr(c, "children") or hasattr(c, "id")):
                    hit = find(c)
                    if hit is not None:
                        return hit
            return None

        toast = find(create_layout())
        assert toast.dismissable is True
        assert getattr(toast, "duration", None) is None
        assert toast.is_open is False

    def test_cancel_and_link_listeners(self):
        from dash._callback import GLOBAL_CALLBACK_LIST, GLOBAL_CALLBACK_MAP
        cancel = [v for k, v in GLOBAL_CALLBACK_MAP.items()
                  if any(i["id"] == "run-pill-cancel" for i in v["inputs"])]
        assert len(cancel) == 1
        btn = next(i for i in cancel[0]["inputs"] if i["id"] == "run-pill-cancel")
        assert btn.get("allow_optional") is True
        seen = next(c for c in GLOBAL_CALLBACK_LIST
                    if str(c["output"]) == "run-seen-store.data")
        assert seen["clientside_function"] is not None
        assert [i["id"] for i in seen["inputs"]] == ["url"]

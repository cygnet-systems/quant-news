"""The run page: /runs/<id>, its loader, its rows and the arrival reader.

get_run_view reads one indexed query per table by run_id and shapes rows
with the same helper the Home board uses; a symbol the run was asked for
but never wrote anything about still gets a row. The layout renders the
header, the failure/cancel banner, the board rows with the report and
watchlist cells, and the synthesis card. The router serves the page for a
run path and "Run not found" for a stale id, and ?open=first opens the
reader on the first symbol's report through the same builder the click
path uses. Everything runs on in-memory SQLite; no model, no LLM.
"""

import os
import re
from datetime import date, datetime, timezone
from types import SimpleNamespace

import dash
import diskcache
import pytest
from dash.exceptions import PreventUpdate
from sqlalchemy import create_engine
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from db.models import (
    AnalysisRun,
    Base,
    JobRun,
    ModelPrediction,
    RecommendationRun,
    TradingAgentReport,
)
from layouts.pages import run as run_page
from services import dashboard_service as ds
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
def db(monkeypatch, tmp_path):
    import db.session as dbs

    eng = create_engine("sqlite://", poolclass=StaticPool,
                        connect_args={"check_same_thread": False})
    Base.metadata.create_all(eng, tables=[
        AnalysisRun.__table__, JobRun.__table__, ModelPrediction.__table__,
        TradingAgentReport.__table__, RecommendationRun.__table__])
    monkeypatch.setattr(dbs, "_engine", eng)
    monkeypatch.setattr(
        dbs, "_SessionLocal", sessionmaker(bind=eng, expire_on_commit=False))
    # The dashboard memo must not touch the repo's cache/dashboard dir.
    monkeypatch.setattr(ds, "_cache_handle",
                        lambda: diskcache.Cache(str(tmp_path / "memo")))
    monkeypatch.setattr(prog, "_write_audit", lambda *a, **kw: None)
    return dbs


def _pred(run_id, symbol, model, decision="BUY", correct=None, pnl=None,
          actual=None, prev=100.0, prediction_date=date(2026, 9, 2),
          target_date=date(2026, 9, 3), confidence=0.6):
    return ModelPrediction(
        id=f"{symbol}_{model}_{prediction_date:%Y%m%d}_{run_id[:8]}",
        symbol=symbol, model_name=model, prediction_date=prediction_date,
        target_date=target_date, decision=decision, confidence=confidence,
        previous_close=prev, actual_close=actual, was_correct=correct,
        pnl_dollars=pnl, run_id=run_id, is_public=True,
    )


def _report(run_id, symbol, decision="BUY", confidence=0.62,
            created_at=None, text="# Report\n\nBuy it."):
    return TradingAgentReport(
        id=f"{symbol}_{run_id[:8]}_{(created_at or datetime.now(timezone.utc)):%H%M%S%f}",
        symbol=symbol, trade_date=date(2026, 9, 2), decision=decision,
        confidence=confidence, report_text=text, model_name="luna",
        run_id=run_id, is_public=True,
        created_at=created_at or datetime.now(timezone.utc),
    )


RESULT = {
    "overall": {"summary": "Two longs, one pass.",
                "portfolio_action": "Lean long into the open.",
                "risk_assessment": "Momentum crowded.",
                "watch_items": ["NVDA 120 support"]},
    "by_symbol": {
        "NVDA": {"action": "BUY", "p_correct": 0.58, "key_level": "support 120",
                 "change_trigger": "close under 120", "reasoning": "Trend intact."},
        "AMD": {"action": "HOLD", "p_correct": 0.5, "reasoning": "No edge."},
    },
    "model_used": "luna",
}


def seed(db, symbols=("NVDA", "AMD", "TSLA"), status="done", owner="u1",
         kind="manual", with_report=True, with_rec=True, error=None,
         stages=None):
    run_id = rs.create_run(kind, list(symbols), owner, preset="standard",
                           config={"scope": "full", "preset": "standard"},
                           prediction_date="2026-09-02",
                           target_date="2026-09-03", estimate_s=120)
    for stage, spec in (stages or {}).items():
        rs.update_progress(run_id, stage, **spec)
    if status != "queued":
        rs.set_status(run_id, status, error=error)
    with rs.get_session() as session:
        session.add(_pred(run_id, "NVDA", "kronos_mini", "BUY", correct=True,
                          pnl=12.0, actual=104.0))
        session.add(_pred(run_id, "NVDA", "xgboost_shap", "SELL", correct=False,
                          pnl=-8.0, actual=104.0))
        session.add(_pred(run_id, "NVDA", ds.SYNTHESIS_MODEL, "BUY",
                          correct=True, pnl=12.0, actual=104.0))
        session.add(_pred(run_id, "AMD", "kronos_mini", "HOLD", prev=150.0))
        if with_report:
            session.add(_report(run_id, "NVDA",
                                created_at=datetime(2026, 9, 2, 12, 0,
                                                    tzinfo=timezone.utc)))
            session.add(_report(run_id, "AMD", decision="SELL", confidence=0.55,
                                created_at=datetime(2026, 9, 2, 12, 1,
                                                    tzinfo=timezone.utc)))
        if with_rec:
            session.add(RecommendationRun(
                trade_date="2026-09-02", symbols_csv="NVDA,AMD,TSLA",
                input_data_hash=run_id[:16], model_used="luna",
                provider_used="lmstudio", result_json=RESULT, run_id=run_id,
                duration_ms=42000, is_public=True))
    return run_id


# --- tree helpers ---------------------------------------------------------

def _text(node) -> str:
    if node is None:
        return ""
    if isinstance(node, (str, int, float)):
        return str(node)
    if isinstance(node, (list, tuple)):
        return " ".join(t for t in (_text(c) for c in node) if t)
    return _text(getattr(node, "children", None))


def _find_all(node, **attrs) -> list:
    def matches(n):
        for k, v in attrs.items():
            got = getattr(n, k, None)
            if k == "className":
                if not got or v not in str(got).split():
                    return False
            elif got != v:
                return False
        return True

    out = []

    def walk(n):
        if n is None or isinstance(n, (str, int, float)):
            return
        if isinstance(n, (list, tuple)):
            for c in n:
                walk(c)
            return
        if matches(n):
            out.append(n)
        walk(getattr(n, "children", None))

    walk(node)
    return out


def _find(node, **attrs):
    hits = _find_all(node, **attrs)
    return hits[0] if hits else None


def _buttons(node, kind: str) -> list:
    return [n for n in _find_all(node)
            if isinstance(getattr(n, "id", None), dict)
            and n.id.get("type") == kind]


# --- loader ---------------------------------------------------------------

class TestRunView:
    def test_rows_follow_the_runs_symbol_order_with_artifacts_attached(self, db):
        run_id = seed(db)
        view = ds.get_run_view(run_id)

        assert view["run"]["run_id"] == run_id
        assert [r["symbol"] for r in view["symbols"]] == ["NVDA", "AMD", "TSLA"]
        assert view["model_names"] == ["kronos_mini", "xgboost_shap"]

        nvda, amd, tsla = view["symbols"]
        assert set(nvda["models"]) == {"kronos_mini", "xgboost_shap"}
        assert nvda["models"]["kronos_mini"]["state"] == "resolved"
        assert nvda["synthesis"]["decision"] == "BUY"
        assert nvda["previous_close"] == 100.0
        assert nvda["report"]["decision"] == "BUY"
        assert nvda["report"]["confidence"] == 0.62
        assert set(nvda["report"]) == {"id", "decision", "confidence",
                                       "trade_date", "model_name"}

        assert amd["models"]["kronos_mini"]["state"] == "pending"
        assert amd["synthesis"] is None
        assert amd["report"]["decision"] == "SELL"

        # Asked for, never written: a row with nothing in it, not a gap.
        assert tsla["models"] == {} and tsla["synthesis"] is None
        assert tsla["report"] is None
        assert tsla["target_date"] == "2026-09-03"

        assert view["counts"] == {"resolved": 3, "held": 0, "pending": 1}
        assert view["pnl"] == 16.0

    def test_recommendation_is_the_runs_synthesis(self, db):
        run_id = seed(db)
        rec = ds.get_run_view(run_id)["recommendation"]
        assert rec["model_used"] == "luna"
        assert rec["duration_ms"] == 42000
        assert rec["result_json"]["overall"]["portfolio_action"] == \
            "Lean long into the open."
        assert rec["created_at"]

    def test_only_this_runs_artifacts(self, db):
        run_id = seed(db)
        other = seed(db, symbols=("NVDA",))
        view = ds.get_run_view(other)
        # AMD was not asked for but the run wrote it: appended after the
        # configured names, never dropped.
        assert [r["symbol"] for r in view["symbols"]] == ["NVDA", "AMD"]
        assert ds.get_run_view(run_id)["symbols"][1]["symbol"] == "AMD"
        # Each run's NVDA report is its own.
        mine = ds.get_run_view(run_id)["symbols"][0]["report"]["id"]
        assert view["symbols"][0]["report"]["id"] != mine

    def test_newest_report_per_symbol_wins(self, db):
        run_id = seed(db, with_report=False)
        with rs.get_session() as session:
            session.add(_report(run_id, "NVDA", decision="HOLD",
                                created_at=datetime(2026, 9, 2, 9, 0,
                                                    tzinfo=timezone.utc)))
            session.add(_report(run_id, "NVDA", decision="SELL",
                                created_at=datetime(2026, 9, 2, 11, 0,
                                                    tzinfo=timezone.utc)))
        assert ds.get_run_view(run_id)["symbols"][0]["report"]["decision"] == "SELL"

    def test_nothing_written_yet_still_lists_every_symbol(self, db):
        run_id = rs.create_run("manual", ["NVDA", "AMD"], "u1")
        view = ds.get_run_view(run_id)
        assert [r["symbol"] for r in view["symbols"]] == ["NVDA", "AMD"]
        assert all(r["models"] == {} and r["report"] is None
                   for r in view["symbols"])
        assert view["recommendation"] is None
        assert view["model_names"] == []

    def test_unknown_run_is_none(self, db):
        assert ds.get_run_view("nope") is None
        assert ds.get_run_view("") is None

    def test_memoizes_terminal_runs_only(self, db, monkeypatch):
        calls = []

        def spy(name, build):
            calls.append(name)
            return build()

        monkeypatch.setattr(ds, "_memoized", spy)
        live = seed(db, status="running")
        done = seed(db, status="done")
        failed = seed(db, status="failed", error="boom")

        cancelled = seed(db, status="cancelled")

        ds.get_run_view(live)
        assert calls == []
        # Cancelling closes the row, but the research stage cannot be
        # killed mid-call: a report written after the cancel would be
        # invisible behind a memo taken at cancel time.
        ds.get_run_view(cancelled)
        assert calls == []
        ds.get_run_view(done)
        ds.get_run_view(failed)
        assert calls == [f"run:{done}", f"run:{failed}"]
        assert ds.MEMOIZED_RUN_STATUSES == ("done", "failed")

    def test_a_cancelled_runs_late_report_still_shows(self, db):
        """The un-killable research stage can land after the cancel."""
        run_id = seed(db, status="cancelled", with_report=False)
        assert ds.get_run_view(run_id)["symbols"][0]["report"] is None
        with rs.get_session() as session:
            session.add(_report(run_id, "NVDA"))
        view = ds.get_run_view(run_id)
        assert view["symbols"][0]["report"]["decision"] == "BUY"

    def test_terminal_read_is_served_from_the_memo(self, db, monkeypatch):
        run_id = seed(db, status="done")
        first = ds.get_run_view(run_id)
        reads = []
        monkeypatch.setattr(ds, "_run_artifacts_uncached",
                            lambda run: reads.append(run["run_id"]) or {})
        again = ds.get_run_view(run_id)
        assert reads == []
        assert again["symbols"] == first["symbols"]

    def test_home_board_shaping_is_the_same_helper(self, db):
        """The board and the run page must never drift: one shaper."""
        seed(db)
        cohort = ds._cohort_uncached("2026-09-02")
        assert [r["symbol"] for r in cohort["symbols"]] == ["AMD", "NVDA"]
        nvda = cohort["symbols"][1]
        assert nvda["synthesis"]["state"] == "resolved"
        assert cohort["model_names"] == ["kronos_mini", "xgboost_shap"]
        assert "report" not in nvda


# --- layout ---------------------------------------------------------------

class TestLayout:
    def test_header_states_what_ran(self, db):
        run_id = seed(db)
        page = run_page.layout(ds.get_run_view(run_id), ["NVDA"])
        head = _find(page, className="run-header")
        text = _text(head)
        assert "manual run" in text
        assert "Standard" in text
        assert "u1" in text
        assert "2026-09-03" in text and "2026-09-02" in text
        assert "Complete" in text
        assert _find(head, className="run-status-done") is not None
        assert "Duration" in text and "so far" not in text
        assert re.search(r"Reports\s+2", text)
        assert _find(page, className="run-banner") is None

    def test_failed_run_leads_with_its_error(self, db):
        run_id = seed(db, status="failed",
                      error="kronos_mini: weights missing\nTraceback...")
        page = run_page.layout(ds.get_run_view(run_id), [])
        banner = _find(page, className="run-banner-failed")
        assert "kronos_mini: weights missing" in _text(banner)
        assert "Traceback" not in _text(banner)
        assert page.children[0] is banner
        # Whatever was written before the stop still renders.
        assert len(_find_all(page, className="home-symbol")) == 3

    def test_cancelled_run_names_the_stage_it_stopped_before(self, db):
        run_id = seed(db, status="cancelled", stages={
            "news": {"state": "done", "done": 3, "total": 3},
            "models": {"state": "done", "done": 3, "total": 3},
        })
        run = ds.get_run_view(run_id)["run"]
        assert run_page.cancelled_before(run) == "cancelled before research"
        page = run_page.layout(ds.get_run_view(run_id), [])
        assert "cancelled before research" in _text(
            _find(page, className="run-banner-cancelled"))

    def test_cancelled_before_anything_started(self, db):
        run_id = seed(db, status="cancelled")
        assert run_page.cancelled_before(ds.get_run_view(run_id)["run"]) == \
            "cancelled before news"

    def test_rows_carry_home_cells_plus_report_and_watchlist(self, db):
        run_id = seed(db)
        view = ds.get_run_view(run_id)
        page = run_page.layout(view, ["NVDA"])

        table = _find(page, className="history-data-table")
        heads = [_text(th) for th in _find_all(table.children[0])
                 if type(th).__name__ == "Th"]
        assert heads == ["Symbol", "Prev close", "Kronos", "XGBoost SHAP",
                         "Synthesis", "Outcome", "Report", ""]
        rows = table.children[1].children
        assert [_text(r.children[0]) for r in rows] == ["NVDA", "AMD", "TSLA"]
        assert _text(rows[0].children[1]) == "100.00"
        assert _text(rows[2].children[1]) == "n/a"
        # Chips and outcome are Home's cells.
        assert _find(rows[0], className="home-chip-positive") is not None
        # Two models and the synthesis scored: Home's outcome arithmetic.
        assert "2/3 right" in _text(_find(rows[0], className="home-resolution"))
        assert "awaiting 2026-09-03 close" in _text(rows[1])

        opens = _buttons(page, "ta-view-btn")
        assert [b.id["report"] for b in opens] == [
            view["symbols"][0]["report"]["id"], view["symbols"][1]["report"]["id"]]
        assert all(_text(b) == "Open" for b in opens)
        assert "SELL 55%" in _text(rows[1])
        assert "no report" in _text(rows[2])

        adds = _buttons(page, "add-symbol")
        assert [b.id["symbol"] for b in adds] == ["AMD", "TSLA"]
        assert "on watchlist" in _text(rows[0])

    def test_a_symbol_with_no_predictions_reads_as_no_call(self, db):
        """TSLA was configured but nothing was ever stored for it. The
        outcome cell must say so rather than score an empty row as a
        flat, held $0.00, which is what a run cancelled before its models
        would have shown on every name."""
        run_id = seed(db, status="cancelled")
        page = run_page.layout(ds.get_run_view(run_id), [])
        tsla = _find(page, className="history-data-table").children[1].children[2]
        cell = _find(tsla, className="home-resolution")
        assert _text(cell) == "no call"
        assert "held" not in _text(cell)
        assert "$" not in _text(cell)
        # A symbol that does have calls is untouched.
        nvda = _find(page, className="history-data-table").children[1].children[0]
        assert "2/3 right" in _text(_find(nvda, className="home-resolution"))

    def test_open_first_marks_the_row_the_reader_opened(self, db):
        run_id = seed(db)
        view = ds.get_run_view(run_id)
        page = run_page.layout(view, [], open_first=True)
        opened = _find_all(page, className="run-row-opened")
        assert len(opened) == 1 and _text(opened[0].children[0]) == "NVDA"
        assert _find_all(run_page.layout(view, []), className="run-row-opened") == []

    def test_synthesis_card_shows_positions_model_and_duration(self, db):
        run_id = seed(db)
        page = run_page.layout(ds.get_run_view(run_id), [])
        card = _find(page, className="run-synthesis")
        text = _text(card)
        assert "Lean long into the open." in text
        assert "Two longs, one pass." in text
        assert "luna" in text and "42s" in text
        positions = _find_all(card, className="run-pos")
        assert [_text(p.children[0].children[0]) for p in positions] == ["NVDA", "AMD"]
        assert "BUY 58%" in _text(positions[0])
        assert "support 120" in _text(positions[0])
        assert "NVDA 120 support" in text

    def test_synthesis_card_explains_its_absence(self, db):
        off = seed(db, with_rec=False,
                   stages={"synthesis": {"state": "skipped"}})
        assert "Recommendations were off" in _text(
            _find(run_page.layout(ds.get_run_view(off), []),
                  className="run-synthesis"))
        live = seed(db, with_rec=False, status="running")
        assert "Synthesis pending" in _text(
            _find(run_page.layout(ds.get_run_view(live), []),
                  className="run-synthesis"))
        done = seed(db, with_rec=False)
        assert "No synthesis was stored" in _text(
            _find(run_page.layout(ds.get_run_view(done), []),
                  className="run-synthesis"))

    def test_running_run_with_nothing_yet_says_so(self, db):
        run_id = rs.create_run("manual", ["NVDA", "AMD"], "u1")
        rs.update_progress(run_id, "news", state="running", done=0, total=2)
        page = run_page.layout(ds.get_run_view(run_id), ["NVDA", "AMD"])
        assert "No predictions yet" in _text(_find(page, className="run-empty"))
        assert "so far" in _text(_find(page, className="run-header"))
        assert _find(page, className="run-status-running") is not None
        assert len(_find_all(page, className="home-symbol")) == 2
        assert _buttons(page, "add-symbol") == []

    def test_scheduled_run_is_badged(self, db):
        run_id = seed(db, kind="scheduled", owner=None)
        head = _find(run_page.layout(ds.get_run_view(run_id), []),
                     className="run-header")
        assert _find(head, className="run-kind-scheduled") is not None
        assert "anonymous" in _text(head)

    def test_table_is_mounted_under_its_refresh_id(self, db):
        run_id = seed(db)
        page = run_page.layout(ds.get_run_view(run_id), [])
        assert _find(page, id="run-symbol-table") is not None

    def test_not_found_page(self):
        page = run_page.not_found("abc")
        assert "Run not found" in _text(page)
        assert "abc" in _text(page)

    def test_duration_label(self):
        run = {"started_at": "2026-09-02T14:00:00+00:00",
               "finished_at": "2026-09-02T14:04:12+00:00"}
        assert run_page.duration_label(run) == "4 min 12s"
        run["finished_at"] = "2026-09-02T14:00:40+00:00"
        assert run_page.duration_label(run) == "40s"
        now = datetime(2026, 9, 2, 14, 2, 0, tzinfo=timezone.utc)
        assert run_page.duration_label({"started_at": "2026-09-02T14:00:00"},
                                       now=now) == "2 min so far"
        assert run_page.duration_label({}) is None


# --- router ---------------------------------------------------------------

def _render(pathname, watchlist=None, search=None):
    return app_module.render_page(pathname, None, None, "all", None, "all",
                                  7, "all", "all", None, watchlist, "1y",
                                  None, search)


class TestRouter:
    def test_run_path_serves_the_run_page(self, db):
        run_id = seed(db)
        page, title = _render(f"/runs/{run_id}", ["NVDA"])
        assert title == "Run"
        assert page.id == "run-page"
        assert "Complete" in _text(_find(page, className="run-header"))
        assert [b.id["symbol"] for b in _buttons(page, "add-symbol")] == ["AMD", "TSLA"]

    def test_unknown_id_is_not_found_not_home(self, db):
        page, title = _render("/runs/" + "0" * 36)
        assert title == "Run"
        assert "Run not found" in _text(page)

    def test_open_first_marks_the_first_report_row(self, db):
        run_id = seed(db)
        page, _ = _render(f"/runs/{run_id}", [], search="?open=first")
        assert len(_find_all(page, className="run-row-opened")) == 1
        page, _ = _render(f"/runs/{run_id}", [])
        assert _find_all(page, className="run-row-opened") == []

    def test_run_route_takes_any_single_segment(self):
        """A malformed id is still a run link: it must reach the run page
        and be told the run is unknown, not silently land on Home."""
        rid = "12345678-1234-1234-1234-123456789abc"
        assert app_module._run_route(f"/runs/{rid}") == rid
        assert app_module._run_route(f"/runs/{rid}/") == rid
        assert app_module._run_route("/runs/nope") == "nope"
        assert app_module._run_route("/runs/nope/") == "nope"
        assert app_module._run_route("/runs") is None
        assert app_module._run_route("/runs/") is None
        assert app_module._run_route("/runs/a/b") is None
        assert app_module._run_route("/schedule") is None
        assert app_module._run_route(None) is None
        assert "/runs" not in app_module._ROUTES

    def test_malformed_id_renders_not_found(self, db):
        page, title = _render("/runs/nope")
        assert title == "Run"
        assert "Run not found" in _text(page)

    def test_malformed_id_is_a_no_op_for_the_page_callbacks(self, db):
        """Every other reader of the route must survive a non-uuid id."""
        with pytest.raises(PreventUpdate):
            app_module.refresh_run_watchlist_cells(["NVDA"], "/runs/nope")
        with pytest.raises(PreventUpdate):
            app_module.open_first_report("?open=first", "/runs/nope")
        with pytest.raises(PreventUpdate):
            app_module.refresh_live_run(1, "/runs/nope", ["NVDA"], None)

    def test_wants_first_report(self):
        assert app_module._wants_first_report("?open=first") is True
        assert app_module._wants_first_report("?x=1&open=first") is True
        assert app_module._wants_first_report("?open=none") is False
        assert app_module._wants_first_report("") is False
        assert app_module._wants_first_report(None) is False

    def test_watchlist_change_swaps_the_cells_on_the_run_page(self, db):
        run_id = seed(db)
        table = app_module.refresh_run_watchlist_cells(["NVDA", "AMD"],
                                                       f"/runs/{run_id}")
        assert [b.id["symbol"] for b in _buttons(table, "add-symbol")] == ["TSLA"]
        with pytest.raises(PreventUpdate):
            app_module.refresh_run_watchlist_cells(["NVDA"], "/")
        with pytest.raises(PreventUpdate):
            app_module.refresh_run_watchlist_cells(["NVDA"], "/runs/" + "0" * 36)


# --- live refresh ---------------------------------------------------------

class TestLivePoll:
    """The run page is built once per visit; the progress poll is what
    fills a live run's rows in without a reload."""

    def test_rows_land_between_ticks(self, db):
        run_id = rs.create_run("manual", ["NVDA", "AMD"], "u1",
                               preset="standard", prediction_date="2026-09-02",
                               target_date="2026-09-03")
        rs.update_progress(run_id, "models", state="running", done=0, total=2)
        first = app_module.refresh_live_run(1, f"/runs/{run_id}", ["NVDA"], None)
        table, pill_text, pill_cls, fp = first
        assert "No predictions yet" in _text(table)
        assert pill_text == "Running"
        assert "run-status-running" in pill_cls
        assert fp["status"] == "running"

        with rs.get_session() as session:
            session.add(_pred(run_id, "NVDA", "kronos_mini", "BUY"))
        rs.update_progress(run_id, "models", done=1, total=2)
        table, _, _, fp2 = app_module.refresh_live_run(
            2, f"/runs/{run_id}", ["NVDA"], fp)
        assert "No predictions yet" not in _text(table)
        assert _find(table, className="home-chip-positive") is not None
        assert fp2 != fp

    def test_an_unchanged_tick_writes_nothing(self, db):
        run_id = seed(db, status="running")
        _, _, _, fp = app_module.refresh_live_run(1, f"/runs/{run_id}", [], None)
        again = app_module.refresh_live_run(2, f"/runs/{run_id}", [], fp)
        assert all(part is dash.no_update for part in again)

    def test_the_finishing_tick_is_the_last_one(self, db):
        run_id = seed(db, status="running")
        _, _, _, fp = app_module.refresh_live_run(1, f"/runs/{run_id}", [], None)
        rs.set_status(run_id, "done")
        table, pill_text, pill_cls, fp2 = app_module.refresh_live_run(
            2, f"/runs/{run_id}", [], fp)
        assert pill_text == "Complete"
        assert "run-status-done" in pill_cls
        assert table is not dash.no_update
        assert fp2["status"] == "done"
        # And from there the poll costs nothing at all.
        with pytest.raises(PreventUpdate):
            app_module.refresh_live_run(3, f"/runs/{run_id}", [], fp2)

    def test_a_terminal_run_is_not_queried_again(self, db, monkeypatch):
        monkeypatch.setattr(ds, "get_run_view",
                            lambda run_id: pytest.fail("read a settled run"))
        for status in ("done", "failed", "cancelled"):
            with pytest.raises(PreventUpdate):
                app_module.refresh_live_run(1, "/runs/x", [], {"status": status})

    def test_off_a_run_route_it_does_nothing(self, db):
        for path in ("/", "/reports", "/schedule", None):
            with pytest.raises(PreventUpdate):
                app_module.refresh_live_run(1, path, ["NVDA"], None)

    def test_a_db_failure_leaves_the_page_alone(self, db, monkeypatch):
        run_id = seed(db, status="running")

        def boom(_run_id):
            raise RuntimeError("connection reset")

        monkeypatch.setattr(ds, "get_run_view", boom)
        with pytest.raises(PreventUpdate):
            app_module.refresh_live_run(1, f"/runs/{run_id}", [], None)

    def test_the_page_seeds_the_fingerprint_it_polls_against(self, db):
        run_id = seed(db, status="running")
        view = ds.get_run_view(run_id)
        page = run_page.layout(view, [])
        store = _find(page, id="run-live-fp")
        assert store.data == run_page.live_fingerprint(view["run"])
        assert _find(page, id="run-status-pill") is not None
        # Seeded from the render, so the first tick after a load that
        # missed nothing writes nothing.
        assert all(part is dash.no_update for part in
                   app_module.refresh_live_run(1, f"/runs/{run_id}", [],
                                               store.data))

    def test_the_watchlist_cells_survive_a_refresh(self, db):
        run_id = seed(db, status="running")
        table, _, _, _ = app_module.refresh_live_run(
            1, f"/runs/{run_id}", ["NVDA"], None)
        assert [b.id["symbol"] for b in _buttons(table, "add-symbol")] == \
            ["AMD", "TSLA"]


# --- ?open=first ----------------------------------------------------------

class TestOpenFirst:
    def test_arrival_opens_the_first_symbols_report(self, db):
        run_id = seed(db)
        is_open, title, body, footer = app_module.open_first_report(
            "?open=first", f"/runs/{run_id}")
        assert is_open is True
        assert title.startswith("NVDA: BUY")
        assert body is not None
        first_id = ds.get_run_view(run_id)["symbols"][0]["report"]["id"]
        assert footer.href == f"/api/download/ta-report/{first_id}"

    def test_first_report_skips_symbols_without_one(self, db):
        run_id = seed(db, symbols=("TSLA", "NVDA", "AMD"))
        _, title, _, _ = app_module.open_first_report(
            "?open=first", f"/runs/{run_id}")
        assert title.startswith("NVDA")

    def test_same_builder_as_the_click_path(self, db, monkeypatch):
        run_id = seed(db)
        report = {"id": "x", "symbol": "NVDA", "decision": "BUY",
                  "confidence": 0.6, "trade_date": "2026-09-02",
                  "report_text": "# R\n\nText.", "model_name": "luna",
                  "created_at": "2026-09-02T12:00:00+00:00"}
        seen = []
        monkeypatch.setattr(app_module, "_report_modal_parts",
                            lambda r: seen.append(r["id"]) or ("t", "b", "f"))
        assert app_module.open_first_report("?open=first", f"/runs/{run_id}") \
            == (True, "t", "b", "f")
        monkeypatch.setattr(app_module, "ctx", SimpleNamespace(
            triggered_id={"type": "ta-view-btn", "report": "x"}))
        monkeypatch.setattr(app_module.get_cache(),
                            "get_all_trading_agent_reports",
                            lambda limit=500: [report])
        assert app_module.jump_to_full_report([], [1]) == (True, "t", "b", "f")
        assert seen[1] == "x"

    def test_no_update_without_the_flag_or_a_report(self, db):
        run_id = seed(db)
        with pytest.raises(PreventUpdate):
            app_module.open_first_report("", f"/runs/{run_id}")
        with pytest.raises(PreventUpdate):
            app_module.open_first_report("?open=first", "/reports")
        bare = seed(db, with_report=False)
        with pytest.raises(PreventUpdate):
            app_module.open_first_report("?open=first", f"/runs/{bare}")
        with pytest.raises(PreventUpdate):
            app_module.open_first_report("?open=first", "/runs/" + "0" * 36)

    def test_db_failure_leaves_the_modal_alone(self, db, monkeypatch):
        run_id = seed(db)

        def boom(_):
            raise RuntimeError("db down")

        monkeypatch.setattr(ds, "get_run_view", boom)
        with pytest.raises(PreventUpdate):
            app_module.open_first_report("?open=first", f"/runs/{run_id}")


# --- wiring ---------------------------------------------------------------

def _callbacks_writing(prop: str) -> list:
    from dash._callback import GLOBAL_CALLBACK_LIST
    return [c for c in GLOBAL_CALLBACK_LIST if prop in c["output"]]


def _inline_scripts() -> list:
    from dash._callback import GLOBAL_INLINE_SCRIPTS
    return list(GLOBAL_INLINE_SCRIPTS)


class TestWiring:
    def test_arrival_writes_the_modal_as_a_duplicate(self):
        owners = _callbacks_writing("ta-report-modal.is_open")
        plain = [c for c in owners if "@" not in c["output"]]
        dup = [c for c in owners if "@" in c["output"]]
        assert len(plain) == 1 and len(dup) == 1
        cb = dup[0]
        assert [(i["id"], i["property"]) for i in cb["inputs"]] == \
            [("url", "search"), ("url", "pathname")]
        # Dash records initial_duplicate as "not prevented" on a duplicate
        # writer: the arrival must fire on the first page load.
        assert cb["prevent_initial_call"] is False
        assert cb["output"].count("@") == 4

    def test_router_reads_the_query_string(self):
        from dash._callback import GLOBAL_CALLBACK_MAP
        router = next(k for k in GLOBAL_CALLBACK_MAP
                      if k.startswith("..page-content.children"))
        cb = GLOBAL_CALLBACK_MAP[router]
        assert [i["id"] for i in cb["inputs"]] == ["url"]
        states = [(s["id"], s["property"]) for s in cb["state"]]
        assert ("url", "search") in states
        assert ("selected-symbols", "data") in states

    def test_run_page_visit_marks_the_run_seen(self):
        """The clientside seen-store writer keys on url.pathname, which the
        run page route shares; its pattern must accept the paths the router
        serves (?open=first lives in url.search, not the pathname)."""
        cb = _callbacks_writing("run-seen-store.data")
        assert len(cb) == 1
        assert [(i["id"], i["property"]) for i in cb[0]["inputs"]] == \
            [("url", "pathname")]
        assert cb[0]["clientside_function"] is not None
        script = next(s for s in _inline_scripts() if "run-seen-store" in s
                      or "seen.run_id" in s)
        m = re.search(r"var m = /(\^\[/\]runs\[/\].*?)/\.exec", script)
        assert m is not None
        pattern = m.group(1)
        rid = "12345678-1234-1234-1234-123456789abc"
        assert re.match(pattern, f"/runs/{rid}")
        assert re.match(pattern, f"/runs/{rid}/")
        assert not re.match(pattern, "/runs/nope")
        assert not re.match(pattern, f"/reports/{rid}")
        assert app_module._run_route(f"/runs/{rid}") == rid

    def test_closing_the_reader_clears_the_query_on_a_run_page(self):
        writers = _callbacks_writing("url.search")
        assert len(writers) == 1
        cb = writers[0]
        assert [(i["id"], i["property"]) for i in cb["inputs"]] == \
            [("ta-report-modal", "is_open")]
        assert cb["clientside_function"] is not None
        assert cb["prevent_initial_call"] is True
        assert any("open=first" in s for s in _inline_scripts())

    def test_watchlist_refresh_targets_the_run_table(self):
        cbs = _callbacks_writing("run-symbol-table.children")
        plain = [c for c in cbs if "@" not in c["output"]]
        dup = [c for c in cbs if "@" in c["output"]]
        assert len(plain) == 1 and len(dup) == 1
        assert [i["id"] for i in plain[0]["inputs"]] == ["selected-symbols"]
        assert plain[0]["prevent_initial_call"] is True

    def test_the_poll_is_the_second_writer_of_the_run_table(self):
        """The live refresh shares the Output, so it must be a duplicate
        writer, and it must ride the existing progress poll."""
        cb = next(c for c in _callbacks_writing("run-symbol-table.children")
                  if "@" in c["output"])
        assert [(i["id"], i["property"]) for i in cb["inputs"]] == \
            [("progress-interval", "n_intervals")]
        assert [(st["id"], st["property"]) for st in cb["state"]] == \
            [("url", "pathname"), ("selected-symbols", "data"),
             ("run-live-fp", "data")]
        assert cb["prevent_initial_call"] is True
        assert "run-status-pill.children" in cb["output"]
        assert "run-live-fp.data" in cb["output"]

"""Home board: the Scheduled tab's rows and the This session tab.

Pure rendering over dicts for the cells (no DB), then a SQLite-backed
check that a manual run at the scheduled cutoff stays off the Scheduled
tab and shows under This session, and that the panes callback feeds both
tabs from one archive bump.
"""

import os
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

import diskcache
import pytest
from dash.exceptions import PreventUpdate
from sqlalchemy import create_engine
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from db.models import AnalysisRun, Base, ModelPrediction
from layouts.pages import home
from services import dashboard_service as ds
from services import progress_service as prog
from services import run_service as rs

_flag_before = os.environ.get(prog._ENV_FLAG)
import app as app_module  # noqa: E402

if _flag_before is None:
    os.environ.pop(prog._ENV_FLAG, None)


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


def _cells(tr) -> list[str]:
    return [_text(td) for td in tr.children]


# --- row fixtures ---------------------------------------------------------

TARGET = "2026-09-03"


def pred(model="kronos_mini", decision="BUY", correct=None, pnl=None,
         actual=None, state=None):
    if state is None:
        state = ("resolved" if correct is not None
                 else "held" if actual is not None or pnl is not None
                 else "pending")
    return {"model_name": model, "symbol": "AAPL", "decision": decision,
            "confidence": 0.6, "prediction_date": "2026-09-02",
            "target_date": TARGET, "was_correct": correct,
            "pnl_dollars": pnl, "actual_close": actual, "state": state}


def row(symbol="AAPL", models=None, synthesis=None, prev=100.0):
    return {"symbol": symbol, "previous_close": prev, "target_date": TARGET,
            "models": models or {}, "synthesis": synthesis}


HIT = row(
    models={"kronos_mini": pred(correct=True, pnl=10.0, actual=104.0),
            "xgboost_shap": pred("xgboost_shap", "SELL", correct=False,
                                 pnl=-4.0, actual=104.0)},
    synthesis=pred(ds.SYNTHESIS_MODEL, "BUY", correct=True, pnl=12.2,
                   actual=104.0),
)
MISS = row(
    "BE",
    models={"kronos_mini": pred(decision="SELL", correct=False, pnl=-22.05,
                                actual=27.9)},
    synthesis=pred(ds.SYNTHESIS_MODEL, "SELL", correct=False, pnl=-22.05,
                   actual=27.9),
    prev=27.33,
)
PENDING = row("NVDA", models={"kronos_mini": pred(decision="HOLD")},
              synthesis=pred(ds.SYNTHESIS_MODEL, "HOLD"))
NOT_RUN = row("ZZZQ", prev=None)
MODELS = ["kronos_mini", "xgboost_shap"]


# --- the outcome arithmetic -----------------------------------------------

def _legacy_resolution(row):
    """The math _resolution_cell used before it was extracted."""
    preds = list(row["models"].values())
    if row.get("synthesis"):
        preds.append(row["synthesis"])
    states = {p["state"] for p in preds}
    if states == {"pending"}:
        return {"pending": True}
    scored = [p for p in preds if p["state"] != "pending"]
    hits = sum(1 for p in scored if p.get("was_correct") is True)
    directional = [p for p in scored if p.get("was_correct") is not None]
    pnl = sum(p.get("pnl_dollars") or 0.0 for p in scored)
    actual = next((p.get("actual_close") for p in scored
                   if p.get("actual_close") is not None), None)
    return {"pending": False, "hits": hits, "directional": len(directional),
            "pnl": pnl, "actual": actual, "held": not directional}


class TestResolutionSummary:
    @pytest.mark.parametrize("r", [
        HIT, MISS,
        # Legacy held row: priced, never scored directionally.
        row(models={"kronos_mini": pred(decision="HOLD", pnl=0.0, actual=101.0)}),
        # Mixed: one scored, one still pending.
        row(models={"kronos_mini": pred(correct=True, pnl=3.0, actual=101.0),
                    "xgboost_shap": pred("xgboost_shap")}),
        # Scored with a missing pnl on one leg.
        row(models={"kronos_mini": pred(correct=False, actual=99.0)},
            synthesis=pred(ds.SYNTHESIS_MODEL, correct=True, pnl=5.0,
                           actual=99.0)),
    ])
    def test_matches_the_old_cell_math(self, r):
        legacy = _legacy_resolution(r)
        got = home.resolution_summary(r)
        assert got["hits"] == legacy["hits"]
        assert got["directional"] == legacy["directional"]
        assert got["pnl"] == pytest.approx(legacy["pnl"])
        assert got["actual"] == legacy["actual"]
        assert got["state"] == ("held" if legacy["held"] else "resolved")

    def test_pending_and_none_states(self):
        assert home.resolution_summary(PENDING)["state"] == "pending"
        assert _legacy_resolution(PENDING)["pending"] is True
        assert home.resolution_summary(NOT_RUN)["state"] == "none"

    def test_run_page_cell_reads_no_call_for_an_empty_row(self):
        # Carried minor from Phase 3: an unpredicted symbol used to read
        # "n/a · held · $+0.00".
        cell = home._resolution_cell(NOT_RUN)
        assert _text(cell) == "no call"
        assert "held" not in _text(cell)
        # The scored path is untouched.
        assert "2/3 right" in _text(home._resolution_cell(HIT))
        assert "awaiting 2026-09-03 close" in _text(home._resolution_cell(PENDING))


# --- scheduled rows ---------------------------------------------------------

class TestScheduledRow:
    def test_headers(self):
        assert [_text(th) for th in home.scheduled_headers()] == \
            ["Symbol", "Call", "Actual", "Result", "P&L", ""]

    def test_hit(self):
        tr = home.scheduled_row(HIT, MODELS)
        symbol, call, actual, result, pnl, models = _cells(tr)
        assert symbol == "AAPL"
        assert call.startswith("BUY")
        assert _find(tr, className="home-call") is not None
        assert "104.00" in actual and "+4.0%" in actual
        assert result == "hit"
        assert _find(tr, className="home-result-hit") is not None
        # Sum over every scored call on the row, as before.
        assert pnl == "$+18.20"
        assert "positive" in _find(tr, className="home-pnl").className
        # The per-model chips live behind the expander, closed by default.
        details = _find(tr, className="home-models-details")
        assert details.open is False
        assert _text(details.children[0]) == "models"
        chips = _find_all(details, className="home-model-chip")
        assert [_text(c).split()[0] for c in chips] == ["Kronos", "XGBoost"]
        assert _find(chips[0], className="home-chip-positive") is not None
        assert _find(chips[1], className="home-chip-negative") is not None

    def test_expanded_opens_the_details(self):
        tr = home.scheduled_row(HIT, MODELS, expanded=True)
        assert _find(tr, className="home-models-details").open is True

    def test_miss(self):
        tr = home.scheduled_row(MISS, ["kronos_mini"])
        _, call, actual, result, pnl, _ = _cells(tr)
        assert call.startswith("SELL")
        assert "27.90" in actual and "+2.1%" in actual
        assert result == "miss"
        assert _find(tr, className="home-result-miss") is not None
        assert pnl == "$-44.10"
        assert "negative" in _find(tr, className="home-pnl").className

    def test_pending(self):
        tr = home.scheduled_row(PENDING, ["kronos_mini"])
        _, call, actual, result, pnl, _ = _cells(tr)
        assert call.startswith("HOLD")
        assert actual == "awaiting 2026-09-03 close"
        assert _find(tr, className="home-pending-pill") is not None
        assert result == "pending"
        assert pnl == ""

    def test_not_run(self):
        tr = home.scheduled_row(NOT_RUN, MODELS)
        assert _cells(tr) == ["ZZZQ", "not run", "", "", "", ""]
        assert "home-row-not-run" in tr.className
        assert _find(tr, className="home-models-details") is None

    def test_call_falls_back_to_the_model_majority(self):
        r = row(models={
            "kronos_mini": pred(decision="SELL", correct=True, pnl=5.0, actual=98.0),
            "xgboost_shap": pred("xgboost_shap", "SELL", correct=True, pnl=5.0,
                                 actual=98.0),
            "lightgbm": pred("lightgbm", "BUY", correct=False, pnl=-5.0,
                             actual=98.0),
        })
        tr = home.scheduled_row(r, ["kronos_mini", "xgboost_shap", "lightgbm"])
        chip = _find(tr, className="home-call").children
        assert _text(chip) == "SELL"
        assert "majority" in chip.title and "2 SELL" in chip.title
        # Result judged on the majority: two of three right.
        assert _cells(tr)[3] == "hit"
        assert "2/3" in _find(tr, className="home-result-pill").title

    def test_majority_tie_is_a_hold_and_even_split_is_not_a_hit(self):
        r = row(models={
            "kronos_mini": pred(decision="SELL", correct=True, pnl=5.0, actual=98.0),
            "xgboost_shap": pred("xgboost_shap", "BUY", correct=False, pnl=-5.0,
                                 actual=98.0),
        })
        tr = home.scheduled_row(r, MODELS)
        chip = _find(tr, className="home-call").children
        assert _text(chip) == "HOLD"
        assert _cells(tr)[3] == "miss"

    def test_no_call_when_the_models_carry_no_decision(self):
        r = row(models={"kronos_mini": {**pred(), "decision": None}})
        assert _cells(home.scheduled_row(r, ["kronos_mini"]))[1] == "no call"

    def test_synthesis_result_wins_over_the_model_tally(self):
        r = row(models={"kronos_mini": pred(correct=False, pnl=-5.0, actual=98.0)},
                synthesis=pred(ds.SYNTHESIS_MODEL, "SELL", correct=True,
                               pnl=5.0, actual=98.0))
        assert _cells(home.scheduled_row(r, ["kronos_mini"]))[3] == "hit"

    def test_news_flag_stays_on_the_symbol_cell(self):
        r = row(models={"kronos_mini": {**pred(), "news_status": "unavailable"}})
        tr = home.scheduled_row(r, ["kronos_mini"])
        assert _find(tr.children[0], className="home-news-flag") is not None


class TestScheduledRows:
    COHORT = {"prediction_date": "2026-09-02", "target_date": TARGET,
              "symbols": [HIT, MISS, PENDING], "model_names": MODELS,
              "counts": {"resolved": 2, "held": 0, "pending": 1}, "pnl": 0.0}

    def test_watchlist_order_then_the_rest_with_not_run_rows(self):
        rows = home.scheduled_rows(self.COHORT, ["nvda", "ZZZQ", "AAPL"])
        assert [r["symbol"] for r in rows] == ["NVDA", "ZZZQ", "AAPL", "BE"]
        assert rows[1]["models"] == {} and rows[1]["target_date"] == TARGET

    def test_cohort_table_uses_the_scheduled_cells(self):
        table = home.cohort_table(self.COHORT, watchlist=["AAPL", "ZZZQ"])
        heads = [_text(th) for th in _find_all(table)
                 if type(th).__name__ == "Th"]
        assert heads == ["Symbol", "Call", "Actual", "Result", "P&L", ""]
        trs = [n for n in _find_all(table) if type(n).__name__ == "Tr"][1:]
        assert [_cells(t)[0] for t in trs] == ["AAPL", "ZZZQ", "BE", "NVDA"]
        assert _cells(trs[1])[1] == "not run"
        assert _cells(trs[0])[3] == "hit"

    def test_narrowed_to_a_watchlist_name_that_was_not_run(self):
        table = home.cohort_table(self.COHORT, active_symbol="ZZZQ",
                                  watchlist=["ZZZQ"])
        trs = [n for n in _find_all(table) if type(n).__name__ == "Tr"][1:]
        assert len(trs) == 1 and _cells(trs[0])[1] == "not run"

    def test_narrowed_row_opens_its_models(self):
        table = home.cohort_table(self.COHORT, active_symbol="AAPL")
        assert _find(table, className="home-models-details").open is True
        assert "No scheduled calls" not in _text(table)

    def test_narrowed_to_an_unknown_name(self):
        table = home.cohort_table(self.COHORT, active_symbol="QQQQ")
        assert "No scheduled calls for QQQQ" in _text(table)


# --- the tabs and the session tab -------------------------------------------

def _run_dict(run_id="r1", status="done", preset="standard", owner="u1",
              symbols=("TSLA",), started="2026-09-02T14:05:00+00:00",
              error=None):
    return {"run_id": run_id, "owner_uid": owner, "kind": "manual",
            "status": status, "preset": preset, "symbols": list(symbols),
            "started_at": started, "finished_at": None, "error": error,
            "active": status in rs.ACTIVE_STATUSES,
            "prediction_date": "2026-09-02", "target_date": TARGET}


def _entry(run, rows, models=("kronos_mini",)):
    return {"run": run, "symbols": rows, "model_names": list(models),
            "counts": {"resolved": 0, "held": 0, "pending": len(rows)},
            "pnl": 0.0, "target_date": TARGET}


EMPTY_COHORT = {"prediction_date": None, "symbols": [], "model_names": [],
                "counts": {"resolved": 0, "held": 0, "pending": 0},
                "pnl": 0.0, "target_date": None}
NO_OPEN = {"total": 0, "dates": []}


def _layout(cohort=EMPTY_COHORT, **kw):
    kw.setdefault("watchlist", ["AAPL"])
    return home.layout(cohort, NO_OPEN, [], None, [], **kw)


class TestSessionTab:
    def test_empty_state(self):
        tab = home.session_tab([])
        assert _text(tab) == "No ad-hoc runs today. Run Analysis to add one."
        assert home.session_tab(None) is not None

    def test_one_block_per_run_with_the_run_pages_rows(self):
        running = _run_dict("r2", status="running", preset="quick", owner=None,
                            symbols=("AMD",), started=None)
        done = _run_dict()
        tsla = row("TSLA", models={"kronos_mini": pred(decision="SELL")})
        amd = row("AMD", prev=None)
        tab = home.session_tab([_entry(running, [amd], models=()),
                                _entry(done, [tsla])])
        blocks = _find_all(tab, className="home-session-run")
        assert [b.id["run"] for b in blocks] == ["r2", "r1"]

        head = _find(blocks[1], className="home-session-head")
        text = _text(head)
        assert "2026-09-02 10:05 ET" in text
        assert "Standard" in text and "u1" in text and "Complete" in text
        link = _find(head, href="/runs/r1")
        assert link is not None and _text(link) == "Open"
        assert "run-status-done" in _find(head, className="run-status").className
        heads = [_text(th) for th in _find_all(blocks[1])
                 if type(th).__name__ == "Th"]
        assert heads == ["Symbol", "Prev close", "Kronos", "Synthesis", "Outcome"]
        assert _find(blocks[1], className="home-chip-negative") is not None
        assert "awaiting 2026-09-03 close" in _text(blocks[1])

        text = _text(_find(blocks[0], className="home-session-head"))
        assert "not started" in text and "Quick" in text
        assert "anonymous" in text and "Running" in text
        assert "Rows fill in as the models finish" in _text(blocks[0])
        assert _find(blocks[0], href="/runs/r2") is not None

    def test_failed_run_carries_its_error_on_the_pill(self):
        run = _run_dict(status="failed", error="kronos: weights missing\nmore")
        tab = home.session_tab([_entry(run, [row("TSLA", prev=None)])])
        pill = _find(tab, className="run-status-failed")
        assert pill.title == "kronos: weights missing"
        assert "No predictions were stored" in _text(tab)


class TestTabs:
    def test_both_tabs_render_with_the_empty_states(self):
        page = _layout(session_runs=[])
        tabs = _find(page, id="home-board-tabs")
        assert tabs is not None
        assert [t.tab_id for t in tabs.children] == ["scheduled", "session"]
        assert [t.label for t in tabs.children] == ["Scheduled", "This session"]
        assert tabs.active_tab == "scheduled"
        # The Scheduled empty state keeps the callback targets mounted.
        for cid in ("home-board-title", "home-meta-wrap", "home-cohort-table",
                    "home-session-runs", "home-symbol-list"):
            assert _find(page, id=cid) is not None, cid
        assert "No scheduled calls yet" in _text(_find(page, id="home-cohort-table"))
        assert _text(_find(page, id="home-session-runs")) == home.SESSION_EMPTY_TEXT
        # No cutoff selector without a cutoff to pick.
        assert _find(page, id="home-cutoff-dropdown") is None

    def test_the_tab_strip_sits_in_its_own_flex_wrapper(self):
        """The structural contract the tab CSS depends on.

        dbc.Tabs renders no element of its own: the className it is given
        lands on the nav <ul> and the pane container is that ul's sibling.
        A flex rule on .home-board-tabs therefore sizes the strip alone,
        collapsing it to a line the board paints over, so the layout rules
        key off this wrapper instead.
        """
        page = _layout(session_runs=[])
        wrap = _find(page, className="home-board-tabs-wrap")
        assert wrap is not None
        assert getattr(wrap.children, "id", None) == "home-board-tabs"
        assert _find(page, className="home-right").children is wrap

        css = (Path(__file__).resolve().parents[1] / "assets/styles.css").read_text()
        assert ".home-board-tabs-wrap {" in css
        for selector in (".home-board-tabs {", ".home-board-tabs >"):
            assert selector not in css, selector

    def test_active_tab_comes_from_the_store(self):
        assert _find(_layout(active_tab="session"),
                     id="home-board-tabs").active_tab == "session"
        assert _find(_layout(active_tab="bogus"),
                     id="home-board-tabs").active_tab == "scheduled"

    def test_scheduled_tab_holds_the_meta_cutoff_rolling_and_inflight(self):
        cohort = dict(TestScheduledRows.COHORT)
        page = _layout(cohort, cutoffs=["2026-09-02", "2026-09-01"],
                       active_cutoff="2026-09-02", session_runs=[])
        tabs = _find(page, id="home-board-tabs")
        scheduled, session = tabs.children
        assert _find(scheduled, id="home-cutoff-dropdown") is not None
        assert _find(scheduled, id="home-meta-wrap") is not None
        assert _find(scheduled, className="home-rolling") is not None
        assert _find(scheduled, className="home-inflight") is not None
        assert _find(scheduled, className="home-models-details") is not None
        assert _find(session, id="home-cutoff-dropdown") is None
        assert _find(session, id="home-session-runs") is not None


class TestSymbolList:
    def test_watchlist_first_then_session_only_names_with_add_buttons(self):
        cohort = dict(TestScheduledRows.COHORT)
        runs = [
            _entry(_run_dict("r2", symbols=("AMD", "AAPL")),
                   [row("AMD", prev=None), HIT]),
            _entry(_run_dict("r1", symbols=("TSLA", "AMD")),
                   [row("TSLA", models={"kronos_mini": pred(decision="SELL")}),
                    row("AMD", prev=None)]),
        ]
        rail = home.symbol_list(cohort, {}, watchlist=["NVDA", "AAPL"],
                                session_runs=runs)
        labels = [_text(n) for n in _find_all(rail, className="home-group-label")]
        assert [lbl.split()[0] for lbl in labels] == ["Watchlist", "This"]
        names = [_text(n) for n in _find_all(rail, className="home-sym-name")]
        # Watchlist order, then newest run's names first, no duplicates,
        # and BE (scheduled cohort only, off the watchlist) is not listed.
        assert names == ["NVDA", "AAPL", "AMD", "TSLA"]
        adds = [n for n in _find_all(rail)
                if isinstance(getattr(n, "id", None), dict)
                and n.id.get("type") == "add-symbol"]
        assert [b.id["symbol"] for b in adds] == ["AMD", "TSLA"]
        removes = [n for n in _find_all(rail)
                   if isinstance(getattr(n, "id", None), dict)
                   and n.id.get("type") == "remove-symbol"]
        assert [b.id["symbol"] for b in removes] == ["NVDA", "AAPL"]

    def test_search_narrows_both_groups(self):
        runs = [_entry(_run_dict(symbols=("TSLA",)),
                       [row("TSLA", prev=None)])]
        rail = home.symbol_list(None, {}, search="ts", watchlist=["AAPL"],
                                session_runs=runs)
        names = [_text(n) for n in _find_all(rail, className="home-sym-name")]
        assert names == ["TSLA"]

    def test_rail_count_is_watchlist_plus_session_only_names(self):
        runs = [_entry(_run_dict(symbols=("TSLA", "AAPL")),
                       [row("TSLA", prev=None), row("AAPL", prev=None)])]
        page = _layout(watchlist=["AAPL", "NVDA"], session_runs=runs)
        assert _text(_find(page, className="home-count-badge")) == "3"

    def test_an_empty_watchlist_lists_the_names_the_board_is_showing(self):
        """With no watchlist the Scheduled board falls back to every name
        the daily job wrote (app._home_cohort). The rail has to list those
        names, or the board's rows cannot be opened from it."""
        cohort = dict(TestScheduledRows.COHORT)
        rail = home.symbol_list(cohort, {}, watchlist=[], session_runs=[])
        labels = [_text(n) for n in _find_all(rail, className="home-group-label")]
        assert [lbl.rsplit(" ", 1)[0] for lbl in labels] == ["Scheduled run"]
        names = [_text(n) for n in _find_all(rail, className="home-sym-name")]
        assert names == [r["symbol"] for r in cohort["symbols"]]
        adds = [n for n in _find_all(rail)
                if isinstance(getattr(n, "id", None), dict)
                and n.id.get("type") == "add-symbol"]
        assert [b.id["symbol"] for b in adds] == names
        assert "No symbols yet" not in _text(rail)
        page = _layout(cohort, watchlist=[], session_runs=[])
        assert _text(_find(page, className="home-count-badge")) == str(len(names))

    def test_the_fallback_names_are_not_repeated_by_a_session_run(self):
        cohort = dict(TestScheduledRows.COHORT)
        runs = [_entry(_run_dict(symbols=("AAPL", "TSLA")),
                       [row("AAPL", prev=None), row("TSLA", prev=None)])]
        rail = home.symbol_list(cohort, {}, watchlist=[], session_runs=runs)
        names = [_text(n) for n in _find_all(rail, className="home-sym-name")]
        assert names == ["AAPL", "BE", "NVDA", "TSLA"]
        assert names.count("AAPL") == 1

    def test_a_watchlist_replaces_the_fallback(self):
        cohort = dict(TestScheduledRows.COHORT)
        rail = home.symbol_list(cohort, {}, watchlist=["AAPL"], session_runs=[])
        labels = [_text(n) for n in _find_all(rail, className="home-group-label")]
        assert [lbl.rsplit(" ", 1)[0] for lbl in labels] == ["Watchlist"]
        names = [_text(n) for n in _find_all(rail, className="home-sym-name")]
        assert names == ["AAPL"]

    def test_search_narrows_the_fallback_too(self):
        cohort = dict(TestScheduledRows.COHORT)
        rail = home.symbol_list(cohort, {}, search="nv", watchlist=[],
                                session_runs=[])
        names = [_text(n) for n in _find_all(rail, className="home-sym-name")]
        assert names == ["NVDA"]
        empty = home.symbol_list(cohort, {}, search="zz", watchlist=[],
                                 session_runs=[])
        assert 'No symbol matching "ZZ"' in _text(empty)


# --- scheduled vs session at one cutoff (SQLite) ----------------------------

@compiles(JSONB, "sqlite")
def _jsonb_as_json(type_, compiler, **kw):
    return "JSON"


@pytest.fixture
def db(monkeypatch, tmp_path):
    import db.session as dbs

    eng = create_engine("sqlite://", poolclass=StaticPool,
                        connect_args={"check_same_thread": False})
    Base.metadata.create_all(eng, tables=[AnalysisRun.__table__,
                                          ModelPrediction.__table__])
    monkeypatch.setattr(dbs, "_engine", eng)
    monkeypatch.setattr(
        dbs, "_SessionLocal", sessionmaker(bind=eng, expire_on_commit=False))
    monkeypatch.setattr(ds, "_cache_handle",
                        lambda: diskcache.Cache(str(tmp_path / "memo")))
    # This session is now the CURRENT cutoff, not the newest manual run's
    # (get_session_runs), so the seeded date has to be the current one or
    # every run below reads as history.
    monkeypatch.setattr(ds, "current_session_date", lambda: TODAY.isoformat())
    return dbs


TODAY = date(2026, 9, 2)


def _pred_row(symbol, model, run_id, decision="BUY", correct=None, pnl=None,
              actual=None):
    return ModelPrediction(
        id=f"{symbol}_{model}_{run_id[:8]}", symbol=symbol, model_name=model,
        prediction_date=TODAY, target_date=TODAY + timedelta(days=1),
        decision=decision, confidence=0.6, previous_close=100.0,
        actual_close=actual, was_correct=correct, pnl_dollars=pnl,
        run_id=run_id, is_public=True,
    )


@pytest.fixture
def same_cutoff(db):
    """The daily job called NVDA and AMD; a manual run then called SELL on
    NVDA and TSLA at the same cutoff."""
    sched = rs.create_run("scheduled", ["NVDA", "AMD"], None,
                          prediction_date=TODAY,
                          target_date=TODAY + timedelta(days=1))
    rs.set_status(sched, "done")
    manual = rs.create_run("manual", ["NVDA", "TSLA"], "u1", preset="quick",
                           prediction_date=TODAY,
                           target_date=TODAY + timedelta(days=1))
    rs.set_status(manual, "running")
    rs.set_status(manual, "done")
    with db.get_session() as session:
        session.add_all([
            _pred_row("NVDA", "kronos_mini", sched, correct=True, pnl=10.0,
                      actual=104.0),
            _pred_row("NVDA", ds.SYNTHESIS_MODEL, sched, correct=True, pnl=10.0,
                      actual=104.0),
            _pred_row("AMD", "kronos_mini", sched, correct=False, pnl=-4.0,
                      actual=99.0),
            _pred_row("NVDA", "xgboost_shap", manual, decision="SELL"),
            _pred_row("TSLA", "xgboost_shap", manual, decision="SELL"),
        ])
    return {"scheduled": sched, "manual": manual}


class TestScheduledVersusSession:
    def test_manual_run_is_off_the_scheduled_tab_and_on_this_session(
            self, db, same_cutoff):
        watchlist = ["NVDA", "TSLA"]
        cohort = app_module._home_cohort(None, watchlist)
        table = home.cohort_table(cohort, watchlist=watchlist)
        trs = [n for n in _find_all(table) if type(n).__name__ == "Tr"][1:]
        cells = {_cells(t)[0]: _cells(t) for t in trs}
        assert list(cells) == ["NVDA", "TSLA"]
        # The scheduled BUY, scored a hit; the manual SELL never enters.
        assert cells["NVDA"][1].startswith("BUY") and cells["NVDA"][3] == "hit"
        assert "SELL" not in _text(table)
        # TSLA was only run ad hoc: on the watchlist, so a row, but not run.
        assert cells["TSLA"][1] == "not run"

        runs = ds.get_session_runs()
        tab = home.session_tab(runs)
        blocks = _find_all(tab, className="home-session-run")
        assert [b.id["run"] for b in blocks] == [same_cutoff["manual"]]
        assert _find(tab, href=f"/runs/{same_cutoff['manual']}") is not None
        names = [_text(td) for td in _find_all(blocks[0], className="home-symbol")]
        assert names == ["NVDA", "TSLA"]
        assert len(_find_all(blocks[0], className="home-chip-negative")) == 2
        assert "Quick" in _text(_find(tab, className="home-session-head"))

    def test_panes_callback_feeds_both_tabs(self, db, same_cutoff, monkeypatch):
        monkeypatch.setattr(app_module, "ctx",
                            SimpleNamespace(triggered_id="report-history-store"))
        monkeypatch.setattr(app_module, "_home_reports_by_symbol", lambda: {})
        rail, table, title, meta, session = app_module.render_home_panes(
            None, "", None, {"refreshed_at": "x"}, ["NVDA", "TSLA"], "/")
        names = [_text(n) for n in _find_all(rail, className="home-sym-name")]
        assert names == ["NVDA", "TSLA"]
        trs = [n for n in _find_all(table) if type(n).__name__ == "Tr"][1:]
        assert [_cells(t)[0] for t in trs] == ["NVDA", "TSLA"]
        assert "Latest calls" in _text(title)
        assert "2026-09-02" in _text(meta)
        assert _find(session, href=f"/runs/{same_cutoff['manual']}") is not None

    def test_panes_callback_off_home_and_search_only(self, db, same_cutoff,
                                                     monkeypatch):
        monkeypatch.setattr(app_module, "_home_reports_by_symbol", lambda: {})
        monkeypatch.setattr(app_module, "ctx",
                            SimpleNamespace(triggered_id="home-symbol-search"))
        with pytest.raises(PreventUpdate):
            app_module.render_home_panes(None, "", None, {}, ["NVDA"], "/reports")
        out = app_module.render_home_panes(None, "ts", None, {},
                                           ["NVDA", "TSLA"], "/")
        names = [_text(n) for n in _find_all(out[0], className="home-sym-name")]
        assert names == ["TSLA"]
        assert all(o is app_module.dash.no_update for o in out[1:])

    def test_session_tab_still_updates_without_a_scheduled_cutoff(
            self, db, monkeypatch):
        manual = rs.create_run("manual", ["TSLA"], None, prediction_date=TODAY,
                               target_date=TODAY + timedelta(days=1))
        with db.get_session() as session:
            session.add(_pred_row("TSLA", "xgboost_shap", manual, decision="SELL"))
        monkeypatch.setattr(app_module, "_home_reports_by_symbol", lambda: {})
        monkeypatch.setattr(app_module, "ctx",
                            SimpleNamespace(triggered_id="report-history-store"))
        out = app_module.render_home_panes(None, "", None, {}, [], "/")
        assert out[1] is app_module.dash.no_update
        assert _find(out[4], href=f"/runs/{manual}") is not None
        # The rail lists the session-only name with a way in.
        names = [_text(n) for n in _find_all(out[0], className="home-sym-name")]
        assert names == ["TSLA"]


class TestSessionCutoff:
    """This session is the CURRENT cutoff's ad-hoc runs, not the newest
    ones whatever their age: the tab is labelled today and its empty state
    says "No ad-hoc runs today"."""

    def test_current_session_date_is_what_a_new_run_would_carry(self):
        from utils.trading_calendar import resolve_target_and_cutoff
        assert ds.current_session_date() == \
            resolve_target_and_cutoff(None)[1].isoformat()

    def test_a_run_from_an_older_cutoff_is_not_this_session(self, db):
        old_day = TODAY - timedelta(days=30)
        run_id = rs.create_run("manual", ["TSLA"], "u1", preset="quick",
                               prediction_date=old_day,
                               target_date=old_day + timedelta(days=1))
        with db.get_session() as session:
            session.add(_pred_row("TSLA", "xgboost_shap", run_id))
        assert ds.get_session_runs() == []
        assert _text(home.session_tab(ds.get_session_runs())) == \
            home.SESSION_EMPTY_TEXT
        # It is still readable as that day's session, on demand.
        listed = ds.get_session_runs(prediction_date=old_day.isoformat())
        assert [e["run"]["run_id"] for e in listed] == [run_id]

    def test_todays_runs_are_this_session(self, db, same_cutoff):
        assert [e["run"]["run_id"] for e in ds.get_session_runs()] == \
            [same_cutoff["manual"]]

    def test_each_block_names_the_cutoff_it_ran_from(self, db, same_cutoff):
        tab = home.session_tab(ds.get_session_runs())
        head = _find(tab, className="home-session-head")
        assert _text(_find(head, className="home-session-cutoff")) == \
            f"{TODAY.isoformat()} cutoff"


class TestNullWatchlist:
    """selected-symbols is None until the browser has ever written it."""

    def test_the_cohort_reads_with_no_watchlist_at_all(self, db, same_cutoff):
        cohort = app_module._home_cohort(None, None)
        assert [r["symbol"] for r in cohort["symbols"]] == ["AMD", "NVDA"]

    def test_the_panes_callback_renders_with_no_watchlist_at_all(
            self, db, same_cutoff, monkeypatch):
        monkeypatch.setattr(app_module, "_home_reports_by_symbol", lambda: {})
        monkeypatch.setattr(app_module, "ctx",
                            SimpleNamespace(triggered_id="report-history-store"))
        rail, table, _, _, _ = app_module.render_home_panes(
            None, "", None, {"refreshed_at": "x"}, None, "/")
        names = [_text(n) for n in _find_all(rail, className="home-sym-name")]
        # The board's own names, since there is no watchlist to draw.
        assert names == ["AMD", "NVDA", "TSLA"]
        trs = [n for n in _find_all(table) if type(n).__name__ == "Tr"][1:]
        assert [_cells(t)[0] for t in trs] == ["AMD", "NVDA"]


# --- wiring -----------------------------------------------------------------

def _callbacks_writing(prop: str) -> list:
    from dash._callback import GLOBAL_CALLBACK_LIST
    return [c for c in GLOBAL_CALLBACK_LIST if prop in c["output"]]


class TestWiring:
    def test_panes_callback_writes_the_session_tab_off_the_archive_bump(self):
        cbs = _callbacks_writing("home-session-runs.children")
        assert len(cbs) == 1
        cb = cbs[0]
        assert "home-cohort-table.children" in cb["output"]
        inputs = [(i["id"], i["property"]) for i in cb["inputs"]]
        states = [(st["id"], st["property"]) for st in cb["state"]]
        assert ("report-history-store", "data") in inputs
        assert ("prediction-store-status", "data") not in inputs
        # The watchlist is read, never watched: load_report_history already
        # fires on it, so an Input here would render Home twice per edit.
        assert ("selected-symbols", "data") not in inputs
        assert ("selected-symbols", "data") in states

    def test_one_watchlist_edit_renders_home_once(self):
        """Walk the callback graph from a watchlist write and count the
        edges that reach the panes callback. Two of them means Home (a
        cohort query, list_runs and a predictions query per manual run)
        is built twice for one click."""
        from dash._callback import GLOBAL_CALLBACK_LIST

        changed = {("selected-symbols", "data")}
        # Everything a watchlist write eventually changes, callbacks fired
        # by that set included (load_report_history is one hop away).
        for _ in range(4):
            for cb in GLOBAL_CALLBACK_LIST:
                if any((i["id"], i["property"]) in changed
                       for i in cb["inputs"]):
                    changed |= {tuple(o.split("."))
                                for o in cb["output"].strip(".").split("...")
                                if "." in o and "{" not in o}
        panes = _callbacks_writing("home-session-runs.children")[0]
        edges = [(i["id"], i["property"]) for i in panes["inputs"]
                 if (i["id"], i["property"]) in changed]
        assert edges == [("report-history-store", "data")]

    def test_tab_store_is_written_from_the_tabs_and_read_by_the_router(self):
        cbs = _callbacks_writing("home-tab-store.data")
        assert len(cbs) == 1
        assert [(i["id"], i["property"]) for i in cbs[0]["inputs"]] == \
            [("home-board-tabs", "active_tab")]
        from dash._callback import GLOBAL_CALLBACK_MAP
        router = next(k for k in GLOBAL_CALLBACK_MAP
                      if k.startswith("..page-content.children"))
        states = [(s["id"], s["property"])
                  for s in GLOBAL_CALLBACK_MAP[router]["state"]]
        # Membership, not position: the router keeps gaining States (the
        # Performance run-scope store was appended after this one), and
        # what matters here is that it reads the tab.
        assert ("home-tab-store", "data") in states

    def test_tab_store_is_mounted(self):
        from layouts.main_layout import create_layout
        store = _find(create_layout(), id="home-tab-store")
        assert store is not None and store.storage_type == "local"

    def test_the_rail_click_pulls_the_board_back_to_scheduled(self):
        """Opening a symbol must reveal the pane it renders into.

        The research pane is home-cohort-table's children, which lives in
        the Scheduled tab; a click made while This session was open used to
        write it into a hidden pane and look like a dead control.
        """
        monkey = SimpleNamespace(triggered_id={"type": "home-sym-btn",
                                               "symbol": "NVDA"})
        import dash
        old_ctx = app_module.ctx
        app_module.ctx = monkey
        try:
            assert app_module.toggle_home_symbol([1], None) == \
                ("NVDA", "scheduled")
            # Clicking the open symbol (or "Show all") clears the narrow and
            # has no reason to move the tab.
            cleared, tab = app_module.toggle_home_symbol([1], "NVDA")
            assert cleared is None and tab is dash.no_update
            with pytest.raises(PreventUpdate):
                app_module.toggle_home_symbol([None], None)
        finally:
            app_module.ctx = old_ctx

    def test_one_writer_of_the_active_tab(self):
        cbs = _callbacks_writing("home-board-tabs.active_tab")
        assert len(cbs) == 1
        assert "home-symbol-filter.data" in cbs[0]["output"]

    def test_track_home_tab(self):
        assert app_module.track_home_tab("session", None) == "session"
        with pytest.raises(PreventUpdate):
            app_module.track_home_tab("session", "session")
        with pytest.raises(PreventUpdate):
            app_module.track_home_tab(None, "scheduled")

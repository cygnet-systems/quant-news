"""The Performance scoreboard counts each call once, from the runs it says.

An ad-hoc run of a name the daily job already called stores its OWN
prediction row: manual predictions live in a separate id space so the rerun
cannot overwrite the scheduled row or throw away its evaluation
(cache_service._prediction_id). The evaluator then scores every one of those
rows, so a page that aggregates them all reported three reruns of one TSLA
call as three trades and three times its P&L -- on the surface this project
uses to decide whether the models have any alpha at all.

Two defences, pinned here:
  * the track record is the scheduled job, and that is the default scope
    (query_predictions(kind=...), and the same rule in calibration_service,
    whose sentence is quoted verbatim INTO reports);
  * even when a user deliberately widens the page to every run, one
    (symbol, model, cutoff) is still one call.
"""

import os
from datetime import date, datetime, timedelta

import diskcache
import pytest
from dash.exceptions import PreventUpdate
from sqlalchemy import create_engine
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from db.models import (AnalysisRun, Base, ModelPrediction,
                       StrategyEvaluation)
from layouts.pages import performance as perf
from services import calibration_service as cal
from services import dashboard_service as ds
from services import progress_service as prog
from services import run_service as rs
from services.cache_service import get_cache
from services.dashboard_service import (aggregate_predictions,
                                        collapse_rerun_duplicates)

_flag_before = os.environ.get(prog._ENV_FLAG)
import app as app_module  # noqa: E402

if _flag_before is None:
    os.environ.pop(prog._ENV_FLAG, None)


@compiles(JSONB, "sqlite")
def _jsonb_as_json(type_, compiler, **kw):
    return "JSON"


TODAY = date(2026, 9, 2)


@pytest.fixture
def db(monkeypatch, tmp_path):
    import db.session as dbs

    eng = create_engine("sqlite://", poolclass=StaticPool,
                        connect_args={"check_same_thread": False})
    Base.metadata.create_all(eng, tables=[AnalysisRun.__table__,
                                          ModelPrediction.__table__,
                                          StrategyEvaluation.__table__])
    monkeypatch.setattr(dbs, "_engine", eng)
    monkeypatch.setattr(
        dbs, "_SessionLocal", sessionmaker(bind=eng, expire_on_commit=False))
    monkeypatch.setattr(ds, "_cache_handle",
                        lambda: diskcache.Cache(str(tmp_path / "memo")))
    # calibration caches its fits for 15 minutes in module globals.
    cal.invalidate()
    cal._fits_as_of.clear()
    yield dbs
    cal.invalidate()
    cal._fits_as_of.clear()


def _run(kind, symbols, cutoff=TODAY):
    run_id = rs.create_run(kind, list(symbols), None, preset="standard",
                           prediction_date=cutoff,
                           target_date=cutoff + timedelta(days=1))
    rs.set_status(run_id, "done")
    return run_id


def _call(symbol, model, run_id=None, cutoff=TODAY, decision="BUY",
          correct=True, pnl=10.0, conf=0.6, created=datetime(2026, 9, 2, 8)):
    """One stored, scored prediction, with the id shape cache_service gives
    it: the scheduled row owns the bare id, a manual rerun appends its run
    (see cache_service._prediction_id)."""
    pred_id = f"{symbol}_{model}_{cutoff:%Y%m%d}"
    if run_id:
        pred_id = f"{pred_id}_{run_id}"
    return ModelPrediction(
        id=pred_id, symbol=symbol, model_name=model, prediction_date=cutoff,
        target_date=cutoff + timedelta(days=1), decision=decision,
        confidence=conf, previous_close=100.0, actual_close=104.0,
        was_correct=correct, pnl_dollars=pnl, run_id=run_id, is_public=True,
        created_at=created,
    )


def _add(db, *rows):
    with db.get_session() as session:
        for r in rows:
            session.add(r)


@pytest.fixture
def one_call_four_rows(db, monkeypatch):
    """The measured defect: one scheduled TSLA/kronos_mini call, then three
    ad-hoc reruns of it. Four rows, four evaluations, one actual call."""
    sched = _run("scheduled", ["TSLA"])
    manual_ids = [_run("manual", ["TSLA"]) for _ in range(3)]
    sched_row = _call("TSLA", "kronos_mini", sched)
    sched_row.id = "TSLA_kronos_mini_20260902"  # the historic, run-less shape
    _add(db, sched_row,
         *[_call("TSLA", "kronos_mini", rid,
                 created=datetime(2026, 9, 2, 10 + i))
           for i, rid in enumerate(manual_ids)])
    return {"scheduled": sched, "manual": manual_ids}


# =============================================================================
# (a) the default scope
# =============================================================================

class TestScheduledIsTheDefault:
    def test_three_reruns_of_one_call_are_one_trade(self, one_call_four_rows):
        rows = get_cache().query_predictions(kind="scheduled")
        assert len(rows) == 1
        g = aggregate_predictions(rows, "model_name")[0]
        assert (g["trades"], g["pnl"]) == (1, 10.0)

    def test_without_the_filter_the_page_would_report_four(self,
                                                           one_call_four_rows):
        """The defect itself, so a regression cannot pass silently."""
        rows = get_cache().query_predictions()
        g = aggregate_predictions(rows, "model_name")[0]
        assert (len(rows), g["trades"], g["pnl"]) == (4, 4, 40.0)

    def test_the_manual_half_is_still_reachable(self, one_call_four_rows):
        rows = get_cache().query_predictions(kind="manual")
        assert len(rows) == 3

    def test_a_row_with_no_run_counts_as_scheduled(self, db):
        """Every prediction written before analysis_runs existed was the
        daily job's; _by_run_kind and _prediction_id agree on that."""
        _add(db, _call("AAPL", "kronos_mini"))
        assert len(get_cache().query_predictions(kind="scheduled")) == 1

    def test_the_filter_stacks_with_the_other_filters(self, one_call_four_rows,
                                                      db):
        _add(db, _call("NVDA", "lightgbm", one_call_four_rows["scheduled"]))
        assert [r["symbol"] for r in get_cache().query_predictions(
            kind="scheduled", model="lightgbm")] == ["NVDA"]
        assert get_cache().query_predictions(kind="scheduled",
                                             start=TODAY + timedelta(days=5)) == []

    def test_strategy_evaluation_skips_the_reruns(self, one_call_four_rows):
        """Each extra row would otherwise become another strategy trade,
        inflating the win rate and Sharpe the Analyze page prints."""
        pending = get_cache().get_unevaluated_predictions_for_strategy(
            "directional")
        assert [p["id"] for p in pending] == ["TSLA_kronos_mini_20260902"]

    def test_history_filters_default_to_scheduled(self):
        assert app_module.history_filters()["kind"] == "scheduled"
        assert app_module.history_filters(run_kind="all")["kind"] is None


# =============================================================================
# (b) + (c) the explicit widening, and what it does with duplicates
# =============================================================================

class TestCollapseRerunDuplicates:
    def test_the_newest_row_wins(self):
        preds = [
            {"symbol": "TSLA", "model_name": "k", "prediction_date": "2026-09-02",
             "id": "a", "created_at": "2026-09-02T08:00:00", "decision": "BUY"},
            {"symbol": "TSLA", "model_name": "k", "prediction_date": "2026-09-02",
             "id": "b", "created_at": "2026-09-02T11:00:00", "decision": "SELL"},
        ]
        kept, dropped = collapse_rerun_duplicates(preds)
        assert dropped == 1
        assert [p["decision"] for p in kept] == ["SELL"]

    def test_a_different_model_symbol_or_cutoff_is_a_different_call(self):
        base = {"symbol": "TSLA", "model_name": "k",
                "prediction_date": "2026-09-02", "created_at": "", "id": "x"}
        preds = [base,
                 {**base, "model_name": "lgbm"},
                 {**base, "symbol": "NVDA"},
                 {**base, "prediction_date": "2026-09-01"}]
        kept, dropped = collapse_rerun_duplicates(preds)
        assert (len(kept), dropped) == (4, 0)

    def test_order_is_preserved(self):
        preds = [
            {"symbol": s, "model_name": "k", "prediction_date": "d",
             "id": s, "created_at": "2026-09-02T02:00:00"}
            for s in ("C", "A", "B")]
        older = {**preds[0], "id": "C-old",
                 "created_at": "2026-09-02T01:00:00"}
        kept, dropped = collapse_rerun_duplicates(preds + [older])
        assert (dropped, [p["symbol"] for p in kept]) == (1, ["C", "A", "B"])

    def test_a_pending_rerun_never_hides_a_scored_call(self):
        """The newest row is not automatically the informative one: a rerun
        started this morning has no outcome yet, and letting it win would
        drop yesterday's resolved call out of the scorecard entirely."""
        scored = {"symbol": "TSLA", "model_name": "k",
                  "prediction_date": "2026-09-02", "id": "sched",
                  "created_at": "2026-09-02T08:00:00", "was_correct": True,
                  "pnl_dollars": 10.0}
        pending = {"symbol": "TSLA", "model_name": "k",
                   "prediction_date": "2026-09-02", "id": "adhoc",
                   "created_at": "2026-09-02T11:00:00", "was_correct": None,
                   "pnl_dollars": None}
        kept, dropped = collapse_rerun_duplicates([scored, pending])
        assert (dropped, [p["id"] for p in kept]) == (1, ["sched"])

    def test_rows_without_a_timestamp_still_collapse(self):
        preds = [{"symbol": "TSLA", "model_name": "k", "prediction_date": "d",
                  "id": i, "created_at": None} for i in ("a", "b")]
        kept, dropped = collapse_rerun_duplicates(preds)
        assert (len(kept), dropped) == (1, 1)

    def test_all_runs_mode_still_counts_the_call_once(self, one_call_four_rows):
        rows = get_cache().query_predictions()
        kept, dropped = collapse_rerun_duplicates(rows)
        assert dropped == 3
        g = aggregate_predictions(kept, "model_name")[0]
        assert (g["trades"], g["pnl"]) == (1, 10.0)


class TestThePageSaysWhatItIsCounting:
    def _body(self, preds, run_kind="scheduled"):
        return perf.body({"predictions": preds}, run_kind=run_kind)

    def _text(self, node):
        if node is None:
            return ""
        if isinstance(node, (str, int, float)):
            return str(node)
        if isinstance(node, (list, tuple)):
            return " ".join(self._text(c) for c in node)
        return self._text(getattr(node, "children", None))

    def test_the_scheduled_default_says_so(self):
        bar = perf.run_scope_bar("scheduled", 0)
        text = self._text(bar)
        assert "daily scheduled job only" in text
        assert bar.children[0].children[1].value == "scheduled"

    def test_all_runs_names_the_collapse(self):
        assert "3 reruns" in self._text(perf.run_scope_bar("all", 3))
        assert "1 rerun " in self._text(perf.run_scope_bar("all", 1))
        assert "counted once" in self._text(perf.run_scope_bar("all", 0))

    def test_the_body_collapses_before_it_aggregates(self):
        rows = [{"symbol": "TSLA", "model_name": "kronos_mini",
                 "prediction_date": "2026-09-02", "target_date": "2026-09-03",
                 "decision": "BUY", "confidence": 0.6, "was_correct": True,
                 "pnl_dollars": 10.0, "previous_close": 100.0,
                 "id": f"r{i}", "created_at": f"2026-09-02T0{i}:00:00"}
                for i in range(4)]
        text = self._text(self._body(rows, "all"))
        assert "1 scored predictions" in text
        assert "3 reruns" in text
        assert "all runs" in text

    def test_the_summary_line_names_the_scope(self):
        rows = [{"symbol": "TSLA", "model_name": "kronos_mini",
                 "prediction_date": "2026-09-02", "target_date": "2026-09-03",
                 "decision": "BUY", "confidence": 0.6, "was_correct": True,
                 "pnl_dollars": 10.0, "previous_close": 100.0,
                 "id": "r", "created_at": ""}]
        assert "scheduled runs only" in self._text(self._body(rows))


class TestWiring:
    def test_the_scope_store_is_mounted_and_defaults_to_scheduled(self):
        from layouts.main_layout import create_layout

        def find(node, target):
            if getattr(node, "id", None) == target:
                return node
            kids = getattr(node, "children", None)
            kids = kids if isinstance(kids, (list, tuple)) else [kids]
            for k in kids:
                if k is not None and not isinstance(k, (str, int, float)):
                    hit = find(k, target)
                    if hit is not None:
                        return hit
            return None

        store = find(create_layout(), "history-run-kind")
        assert store is not None and store.data == "scheduled"

    def test_one_writer_of_the_scope_store(self):
        from dash._callback import GLOBAL_CALLBACK_LIST
        cbs = [c for c in GLOBAL_CALLBACK_LIST
               if "history-run-kind.data" in c["output"]]
        assert len(cbs) == 1
        assert [(i["id"], i["property"]) for i in cbs[0]["inputs"]] == \
            [("perf-run-scope", "value")]

    def test_the_body_rebuilds_when_the_scope_changes(self):
        from dash._callback import GLOBAL_CALLBACK_LIST
        cb = next(c for c in GLOBAL_CALLBACK_LIST
                  if c["output"] == "archive-body.children")
        assert ("history-run-kind", "data") in \
            [(i["id"], i["property"]) for i in cb["inputs"]]

    def test_the_remount_of_the_control_does_not_rewrite_the_store(self):
        """The control lives inside #archive-body, so every rebuild re-fires
        its value Input with the value the body was just built from."""
        with pytest.raises(PreventUpdate):
            app_module.set_run_scope("scheduled", "scheduled")
        with pytest.raises(PreventUpdate):
            app_module.set_run_scope(None, "scheduled")
        assert app_module.set_run_scope("all", "scheduled") == "all"


# =============================================================================
# (d) the number that is quoted into a report
# =============================================================================

class TestCalibrationIgnoresAdHocReruns:
    def test_hit_rate_is_unchanged_by_manual_reruns(self, db):
        sched = _run("scheduled", ["TSLA", "NVDA"])
        _add(db,
             _call("TSLA", "kronos_mini", sched, correct=True),
             _call("NVDA", "kronos_mini", sched, correct=False, pnl=-4.0))
        before = cal.evaluated_hit_rate("kronos_mini", days=90,
                                        as_of=str(TODAY + timedelta(days=1)))
        assert (before["n"], before["hit_rate"]) == (2, None)  # below the floor

        for _ in range(5):
            _add(db, _call("TSLA", "kronos_mini", _run("manual", ["TSLA"]),
                           correct=True))
        after = cal.evaluated_hit_rate("kronos_mini", days=90,
                                       as_of=str(TODAY + timedelta(days=1)))
        assert after == before

    def test_the_sentence_a_report_quotes_counts_scheduled_calls_only(self, db):
        """MIN_STATEABLE_SAMPLES is a floor on the JOB's record; ad-hoc
        reruns must not be able to lift a model over it."""
        sched = _run("scheduled", ["TSLA"])
        rows = []
        for i in range(cal.MIN_STATEABLE_SAMPLES):
            cutoff = TODAY - timedelta(days=i + 1)
            rows.append(_call("TSLA", "kronos_mini", sched, cutoff=cutoff,
                              correct=(i % 2 == 0)))
        _add(db, *rows)
        as_of = str(TODAY + timedelta(days=1))
        stated = cal.track_record_sentence("kronos_mini", days=90, as_of=as_of)
        assert f"n={cal.MIN_STATEABLE_SAMPLES}" in stated
        assert "50%" in stated

        for _ in range(10):
            _add(db, _call("TSLA", "kronos_mini", _run("manual", ["TSLA"]),
                           cutoff=TODAY - timedelta(days=1), correct=True))
        assert cal.track_record_sentence("kronos_mini", days=90,
                                         as_of=as_of) == stated

    def test_the_isotonic_fit_sees_scheduled_pairs_only(self, db):
        sched = _run("scheduled", ["TSLA"])
        rows = [_call("TSLA", "kronos_mini", sched,
                      cutoff=TODAY - timedelta(days=i + 1), conf=0.6,
                      correct=(i % 2 == 0))
                for i in range(4)]
        _add(db, *rows)
        for _ in range(20):
            _add(db, _call("TSLA", "kronos_mini", _run("manual", ["TSLA"]),
                           cutoff=TODAY - timedelta(days=1), correct=True))
        fits = cal._fit_all(str(TODAY))
        assert fits["kronos_mini"].n == 4

    def test_rolling_hit_rate_ignores_them_too(self, db):
        sched = _run("scheduled", ["TSLA"])
        _add(db, *[_call("TSLA", "kronos_mini", sched,
                         cutoff=TODAY - timedelta(days=i + 1),
                         correct=(i % 2 == 0))
                   for i in range(20)])
        as_of = str(TODAY)
        cal._hit_cache.clear()
        before = cal.rolling_hit_rate("kronos_mini", days=30, as_of=as_of)
        assert before == 0.5
        for _ in range(10):
            _add(db, _call("TSLA", "kronos_mini", _run("manual", ["TSLA"]),
                           cutoff=TODAY - timedelta(days=1), correct=True))
        cal._hit_cache.clear()
        assert cal.rolling_hit_rate("kronos_mini", days=30, as_of=as_of) == before

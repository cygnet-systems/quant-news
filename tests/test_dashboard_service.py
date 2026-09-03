"""Guards for the read-side aggregation behind Home and Performance.

The subtlety worth pinning is that a prediction has three outcomes, not two.
`was_correct` distinguishes hit from miss, but it is left None for a HOLD,
which the evaluator still scores (against the symbol's no-trade band) and still
stamps with pnl_dollars. Reading was_correct alone therefore reports an
evaluated HOLD as if it were still awaiting its close, and the launch screen
would show resolved work as in-flight forever.

The other invariant here is that hit rate counts positions, not predictions.
HOLD takes no position; folding HOLDs into the denominator inflates the rate
toward whatever share of days the models sat out.
"""

from services.dashboard_service import (
    SYNTHESIS_MODEL,
    aggregate_predictions,
    resolution_state,
)


def pred(model="kronos_mini", symbol="AAPL", decision="BUY", correct=None,
         pnl=None, actual=None, confidence=None):
    return {
        "model_name": model,
        "symbol": symbol,
        "decision": decision,
        "was_correct": correct,
        "pnl_dollars": pnl,
        "actual_close": actual,
        "confidence": confidence,
    }


class TestResolutionState:
    def test_scored_direction_is_resolved(self):
        assert resolution_state(pred(correct=True, pnl=12.0, actual=101.0)) == "resolved"
        assert resolution_state(pred(correct=False, pnl=-8.0, actual=99.0)) == "resolved"

    def test_evaluated_hold_is_held_not_pending(self):
        """The regression this module exists to prevent."""
        assert resolution_state(
            pred(decision="HOLD", correct=None, pnl=0.0, actual=100.2)) == "held"

    def test_unevaluated_is_pending(self):
        assert resolution_state(pred(decision="BUY")) == "pending"
        assert resolution_state(pred(decision="HOLD")) == "pending"

    def test_priced_but_unscored_still_counts_as_resolved_work(self):
        """actual_close set means the session closed and the row was touched."""
        assert resolution_state(pred(actual=101.0)) == "held"


class TestAggregate:
    def test_holds_are_excluded_from_hit_rate(self):
        preds = [
            pred(decision="BUY", correct=True, pnl=10.0),
            pred(decision="SELL", correct=False, pnl=-5.0),
            pred(decision="HOLD", correct=None, pnl=0.0, actual=100.0),
            pred(decision="HOLD", correct=None, pnl=0.0, actual=100.0),
        ]
        g = aggregate_predictions(preds, "model_name")[0]
        assert g["trades"] == 2
        assert g["holds"] == 2
        assert g["hit_rate"] == 0.5
        assert g["pnl"] == 5.0
        assert g["pnl_per_trade"] == 2.5

    def test_all_holds_yields_no_hit_rate_rather_than_zero(self):
        """A model that never took a position has no rate, not a 0% one."""
        preds = [pred(decision="HOLD", correct=None, pnl=0.0, actual=100.0)]
        g = aggregate_predictions(preds, "model_name")[0]
        assert g["trades"] == 0
        assert g["hit_rate"] is None
        assert g["pnl_per_trade"] is None

    def test_pending_rows_contribute_nothing(self):
        preds = [pred(decision="BUY", correct=True, pnl=10.0), pred(decision="BUY")]
        g = aggregate_predictions(preds, "model_name")[0]
        assert g["trades"] == 1
        assert g["holds"] == 0
        assert g["pnl"] == 10.0

    def test_groups_split_by_key(self):
        preds = [
            pred(model="kronos_mini", symbol="AAPL", correct=True, pnl=10.0),
            pred(model="lightgbm", symbol="MSFT", correct=False, pnl=-4.0),
        ]
        assert [g["name"] for g in aggregate_predictions(preds, "model_name")] == \
            ["kronos_mini", "lightgbm"]
        assert [g["name"] for g in aggregate_predictions(preds, "symbol")] == \
            ["AAPL", "MSFT"]

    def test_average_confidence_ignores_missing(self):
        preds = [
            pred(correct=True, pnl=1.0, confidence=0.8),
            pred(correct=True, pnl=1.0, confidence=0.6),
            pred(correct=True, pnl=1.0),
        ]
        assert aggregate_predictions(preds, "model_name")[0]["avg_confidence"] == 0.7

    def test_synthesis_is_a_distinct_group(self):
        """Luna's verdict is scored as its own row in the scorecards.

        It makes a fresh call over reports + signals, so it competes as a
        peer (2026-08-08 decision); the aggregator must keep it as its own
        group rather than folding it into any member model's stats.
        """
        preds = [
            pred(model="kronos_mini", correct=True, pnl=10.0),
            pred(model=SYNTHESIS_MODEL, correct=False, pnl=-3.0),
        ]
        names = [g["name"] for g in aggregate_predictions(preds, "model_name")]
        assert SYNTHESIS_MODEL in names and "kronos_mini" in names


# =============================================================================
# Cohort by run kind (Home's Scheduled / This session split)
# =============================================================================
#
# The Scheduled tab reads the newest cutoff a scheduled run wrote, watchlist
# names only, and must not move when an ad-hoc run lands on the same day.
# The join is a LEFT JOIN on analysis_runs, and a prediction with no run
# row counts as scheduled: every row written before that table existed was
# the daily job's.

import diskcache
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from datetime import date, datetime, timedelta

from db.models import AnalysisRun, Base, ModelPrediction, StockPrice
from services import dashboard_service as ds
from services import run_service as rs
from services.cache_service import get_cache


@compiles(JSONB, "sqlite")
def _jsonb_as_json(type_, compiler, **kw):
    return "JSON"


@pytest.fixture
def db(monkeypatch, tmp_path):
    import db.session as dbs

    eng = create_engine("sqlite://", poolclass=StaticPool,
                        connect_args={"check_same_thread": False})
    Base.metadata.create_all(eng, tables=[AnalysisRun.__table__,
                                          ModelPrediction.__table__,
                                          StockPrice.__table__])
    monkeypatch.setattr(dbs, "_engine", eng)
    monkeypatch.setattr(
        dbs, "_SessionLocal", sessionmaker(bind=eng, expire_on_commit=False))
    # Never the repo's cache/dashboard dir.
    monkeypatch.setattr(ds, "_cache_handle",
                        lambda: diskcache.Cache(str(tmp_path / "memo")))
    return dbs


TODAY = date(2026, 9, 2)


def _row(symbol, model, run_id=None, cutoff=TODAY, decision="BUY",
         correct=None, pnl=None, actual=None, prev=100.0):
    tag = (run_id or "legacy")[:8]
    return ModelPrediction(
        id=f"{symbol}_{model}_{cutoff:%Y%m%d}_{tag}",
        symbol=symbol, model_name=model, prediction_date=cutoff,
        target_date=cutoff + timedelta(days=1), decision=decision,
        confidence=0.6, previous_close=prev, actual_close=actual,
        was_correct=correct, pnl_dollars=pnl, run_id=run_id, is_public=True,
    )


def _run(kind, symbols, cutoff=TODAY, status="done", owner=None):
    run_id = rs.create_run(kind, list(symbols), owner, preset="standard",
                           prediction_date=cutoff,
                           target_date=cutoff + timedelta(days=1))
    if status != "queued":
        rs.set_status(run_id, status)
    return run_id


def _add(db, *rows):
    with db.get_session() as session:
        for r in rows:
            session.add(r)


def _symbols(cohort):
    return [r["symbol"] for r in cohort["symbols"]]


@pytest.fixture
def same_cutoff(db):
    """A scheduled run and a manual run at the same cutoff, both on NVDA.

    The scheduled run called BUY on NVDA and AMD; the manual one called
    SELL on NVDA and TSLA. A board that mixed them would show NVDA twice
    over (or the manual SELL winning by model name).
    """
    sched = _run("scheduled", ["NVDA", "AMD"])
    manual = _run("manual", ["NVDA", "TSLA"])
    _add(db,
         _row("NVDA", "kronos_mini", sched, correct=True, pnl=10.0, actual=104.0),
         _row("AMD", "kronos_mini", sched, correct=False, pnl=-4.0, actual=99.0),
         _row("NVDA", "xgboost_shap", manual, decision="SELL"),
         _row("TSLA", "xgboost_shap", manual, decision="SELL"))
    return {"scheduled": sched, "manual": manual}


class TestCohortByKind:
    def test_scheduled_cohort_excludes_the_manual_run(self, db, same_cutoff):
        cohort = ds.get_cohort(kind="scheduled")
        assert cohort["prediction_date"] == "2026-09-02"
        assert _symbols(cohort) == ["AMD", "NVDA"]
        nvda = cohort["symbols"][1]
        assert set(nvda["models"]) == {"kronos_mini"}
        assert cohort["model_names"] == ["kronos_mini"]
        assert cohort["counts"] == {"resolved": 2, "held": 0, "pending": 0}

    def test_manual_cohort_is_the_other_half(self, db, same_cutoff):
        cohort = ds.get_cohort(kind="manual")
        assert _symbols(cohort) == ["NVDA", "TSLA"]
        assert cohort["symbols"][0]["models"]["xgboost_shap"]["decision"] == "SELL"

    def test_no_kind_is_everything_at_the_cutoff(self, db, same_cutoff):
        cohort = ds.get_cohort()
        assert _symbols(cohort) == ["AMD", "NVDA", "TSLA"]
        assert set(cohort["symbols"][1]["models"]) == {"kronos_mini", "xgboost_shap"}

    def test_legacy_rows_without_a_run_count_as_scheduled(self, db):
        """Pre-analysis_runs predictions were all the daily job's."""
        manual = _run("manual", ["NVDA"])
        _add(db,
             _row("AAPL", "kronos_mini"),
             _row("MSFT", "kronos_mini", run_id="adhoc"),
             _row("NVDA", "kronos_mini", manual))
        assert _symbols(ds.get_cohort(kind="scheduled")) == ["AAPL", "MSFT"]
        assert _symbols(ds.get_cohort(kind="manual")) == ["NVDA"]

    def test_latest_scheduled_cutoff_ignores_a_newer_manual_run(self, db):
        """The board never jumps to the cutoff an ad-hoc run used."""
        sched = _run("scheduled", ["NVDA"], cutoff=TODAY - timedelta(days=1))
        manual = _run("manual", ["AMD"], cutoff=TODAY)
        _add(db,
             _row("NVDA", "kronos_mini", sched, cutoff=TODAY - timedelta(days=1)),
             _row("AMD", "kronos_mini", manual, cutoff=TODAY))
        assert ds.get_cohort(kind="scheduled")["prediction_date"] == "2026-09-01"
        assert ds.get_latest_cohort()["prediction_date"] == "2026-09-01"
        assert ds.get_cohort()["prediction_date"] == "2026-09-02"

    def test_symbols_filter(self, db, same_cutoff):
        cohort = ds.get_cohort(kind="scheduled", symbols=["nvda", "TSLA"])
        assert _symbols(cohort) == ["NVDA"]
        assert ds.get_latest_cohort(symbols=["AMD"])["counts"]["resolved"] == 1

    def test_empty_watchlist_is_an_empty_board_not_every_row(self, db, same_cutoff):
        cohort = ds.get_cohort(kind="scheduled", symbols=[])
        assert cohort["prediction_date"] == "2026-09-02"
        assert cohort["symbols"] == []

    def test_kind_and_symbols_are_part_of_the_memo_key(self, db, same_cutoff):
        assert _symbols(ds.get_cohort()) == ["AMD", "NVDA", "TSLA"]
        assert _symbols(ds.get_cohort(kind="scheduled")) == ["AMD", "NVDA"]
        assert _symbols(ds.get_cohort(kind="scheduled", symbols=["AMD"])) == ["AMD"]
        # Second reads come from the memo and must still differ.
        assert _symbols(ds.get_cohort()) == ["AMD", "NVDA", "TSLA"]
        assert _symbols(ds.get_cohort(kind="scheduled", symbols=["AMD"])) == ["AMD"]

    def test_cutoffs_per_kind(self, db):
        d1, d2, d3 = (TODAY - timedelta(days=2), TODAY - timedelta(days=1), TODAY)
        sched = _run("scheduled", ["NVDA"], cutoff=d2)
        manual = _run("manual", ["NVDA"], cutoff=d3)
        _add(db,
             _row("NVDA", "kronos_mini", cutoff=d1),
             _row("NVDA", "kronos_mini", sched, cutoff=d2),
             _row("NVDA", "kronos_mini", manual, cutoff=d3))
        assert ds.get_available_cutoffs() == ["2026-09-01", "2026-08-31"]
        assert ds.get_available_cutoffs(kind="manual") == ["2026-09-02"]
        assert ds.get_available_cutoffs(kind=None) == [
            "2026-09-02", "2026-09-01", "2026-08-31"]
        assert ds.get_available_cutoffs(limit=1) == ["2026-09-01"]

    def test_reader_kind_filter_at_the_cache_layer(self, db, same_cutoff):
        cache = get_cache()
        rows = cache.get_predictions_between(TODAY, TODAY, kind="scheduled")
        assert {(r["symbol"], r["model_name"]) for r in rows} == {
            ("NVDA", "kronos_mini"), ("AMD", "kronos_mini")}
        assert cache.get_latest_prediction_date(kind="manual") == TODAY
        assert cache.get_predictions_between(TODAY, TODAY, symbols=[]) == []


class TestRollingByKind:
    def test_rolling_strip_counts_scheduled_runs_only(self, db, monkeypatch):
        monkeypatch.setattr(ds, "datetime", _FrozenNow)
        sched = _run("scheduled", ["NVDA"])
        manual = _run("manual", ["NVDA"])
        _add(db,
             _row("NVDA", "kronos_mini", sched, correct=True, pnl=10.0, actual=104.0),
             _row("AMD", "kronos_mini", correct=True, pnl=5.0, actual=101.0),
             _row("NVDA", "kronos_mini", manual, correct=False, pnl=-30.0,
                  actual=104.0))
        sched_only = ds.get_rolling_performance(days=30, kind="scheduled")
        assert [(g["trades"], g["trade_hits"], g["pnl"]) for g in sched_only] == \
            [(2, 2, 15.0)]
        everything = ds.get_rolling_performance(days=30)
        assert [(g["trades"], g["trade_hits"], g["pnl"]) for g in everything] == \
            [(3, 2, -15.0)]
        # Per-symbol reads bypass the memo but keep the filter.
        by_symbol = ds.get_rolling_performance(days=30, group_key="symbol",
                                               symbols=["NVDA"], kind="scheduled")
        assert [(g["name"], g["trades"], g["pnl"]) for g in by_symbol] == \
            [("NVDA", 1, 10.0)]


class TestAdHocRerunOfAScheduledName:
    """The real writer, twice, for one (symbol, model, cutoff).

    Every other fixture here hands ModelPrediction rows their ids, which
    hides the collision production actually hits: `store_prediction` derives
    the id, and a shared one made an ad-hoc run REPLACE the scheduled row.
    """

    def _store(self, run_id, decision):
        get_cache().store_prediction(
            "NVDA", "kronos_mini",
            {"decision": decision, "confidence": 0.6},
            prediction_date_str=str(TODAY), run_id=run_id)

    def _scored(self, db, was_correct=True):
        with db.get_session() as session:
            row = session.execute(
                select(ModelPrediction).where(
                    ModelPrediction.symbol == "NVDA")
            ).scalars().one()
            row.actual_close = 104.0
            row.was_correct = was_correct
            row.pnl_dollars = 10.0
            return row.id

    def test_the_scheduled_row_and_its_score_survive_the_ad_hoc_run(self, db):
        sched = _run("scheduled", ["NVDA"])
        self._store(sched, "BUY")
        sched_id = self._scored(db)

        manual = _run("manual", ["NVDA"])
        self._store(manual, "SELL")

        board = ds.get_cohort(kind="scheduled")
        assert _symbols(board) == ["NVDA"]
        call = board["symbols"][0]["models"]["kronos_mini"]
        assert (call["decision"], call["was_correct"]) == ("BUY", True)
        assert board["counts"]["resolved"] == 1

        session_board = ds.get_cohort(kind="manual")
        assert session_board["symbols"][0]["models"]["kronos_mini"]["decision"] \
            == "SELL"

        with db.get_session() as session:
            ids = sorted(session.execute(
                select(ModelPrediction.id)).scalars().all())
        assert ids == sorted([sched_id, f"{sched_id}_{manual}"])

    def test_a_repeat_of_the_same_run_still_upserts(self, db):
        manual = _run("manual", ["NVDA"])
        self._store(manual, "BUY")
        self._scored(db)
        self._store(manual, "SELL")
        with db.get_session() as session:
            rows = session.execute(select(ModelPrediction)).scalars().all()
        assert len(rows) == 1
        # Re-storing an id still invalidates the evaluation it carried.
        assert rows[0].decision == "SELL" and rows[0].was_correct is None

    def test_a_run_less_write_keeps_the_historic_id(self, db):
        self._store(None, "BUY")
        with db.get_session() as session:
            ids = session.execute(select(ModelPrediction.id)).scalars().all()
        assert ids == [f"NVDA_kronos_mini_{TODAY:%Y%m%d}"]
        assert _symbols(ds.get_cohort(kind="scheduled")) == ["NVDA"]


class _FrozenNow(datetime):
    @classmethod
    def now(cls, tz=None):
        return cls(2026, 9, 2, 15, 0, tzinfo=tz)


class TestSessionRuns:
    def test_groups_by_run_newest_first(self, db):
        first = _run("manual", ["NVDA", "AMD"])
        second = _run("manual", ["TSLA"], status="running")
        sched = _run("scheduled", ["NVDA"])
        _add(db,
             _row("NVDA", "kronos_mini", first, correct=True, pnl=10.0, actual=104.0),
             _row("NVDA", ds.SYNTHESIS_MODEL, first, correct=True, pnl=10.0,
                  actual=104.0),
             _row("TSLA", "kronos_mini", second, decision="SELL"),
             _row("NVDA", "kronos_mini", sched))
        runs = ds.get_session_runs()
        assert [r["run"]["run_id"] for r in runs] == [second, first]
        assert runs[0]["run"]["active"] is True
        assert [row["symbol"] for row in runs[0]["symbols"]] == ["TSLA"]
        assert runs[0]["counts"] == {"resolved": 0, "held": 0, "pending": 1}
        # Run order for the rows, a configured symbol with nothing written
        # still gets its row, and the synthesis verdict takes its own slot.
        assert [row["symbol"] for row in runs[1]["symbols"]] == ["NVDA", "AMD"]
        assert runs[1]["symbols"][0]["synthesis"]["decision"] == "BUY"
        assert runs[1]["symbols"][1]["models"] == {}
        assert runs[1]["model_names"] == ["kronos_mini"]
        assert runs[1]["pnl"] == 20.0
        assert runs[1]["target_date"] == "2026-09-03"

    def test_session_is_the_newest_manual_cutoff(self, db):
        old = _run("manual", ["NVDA"], cutoff=TODAY - timedelta(days=1))
        new = _run("manual", ["AMD"], cutoff=TODAY)
        _run("scheduled", ["NVDA"], cutoff=TODAY + timedelta(days=1))
        runs = ds.get_session_runs()
        assert [r["run"]["run_id"] for r in runs] == [new]
        assert [r["run"]["run_id"] for r in ds.get_session_runs("2026-09-01")] == [old]
        assert ds.get_session_runs("2026-08-01") == []

    def test_no_manual_runs_is_an_empty_session(self, db):
        _run("scheduled", ["NVDA"])
        assert ds.get_session_runs() == []

    def test_limit_keeps_the_newest(self, db):
        ids = [_run("manual", ["NVDA"]) for _ in range(3)]
        runs = ds.get_session_runs(limit=2)
        assert [r["run"]["run_id"] for r in runs] == [ids[2], ids[1]]

    def test_active_runs_are_read_live_finished_ones_from_the_memo(self, db, monkeypatch):
        running = _run("manual", ["NVDA"], status="running")
        # The memo needs a latest prediction date to key on.
        _add(db, _row("NVDA", "kronos_mini", running))
        calls = []
        real = ds._session_runs_uncached
        monkeypatch.setattr(ds, "_session_runs_uncached",
                            lambda *a: calls.append(a) or real(*a))
        ds.get_session_runs()
        ds.get_session_runs()
        assert len(calls) == 2
        rs.set_status(running, "done")
        ds.get_session_runs()
        ds.get_session_runs()
        assert len(calls) == 3

    def test_a_run_started_after_the_last_read_is_not_hidden_by_the_memo(self, db):
        """The memo's shared key moves with the prediction data, which a new
        run row does not touch. Without the runs generation in the key, a run
        started after Home last rendered would be missing from this tab for
        the whole time it was in flight."""
        sched = _run("scheduled", ["NVDA"])
        _add(db, _row("NVDA", "kronos_mini", sched))
        assert ds.get_session_runs() == []
        started = _run("manual", ["AMD"], status="running")
        assert [r["run"]["run_id"] for r in ds.get_session_runs()] == [started]

    def test_a_run_that_wrote_nothing_appears_once_it_finishes(self, db):
        """A report-only manual run stores no prediction, so nothing else
        invalidates the memo for it."""
        sched = _run("scheduled", ["NVDA"])
        _add(db, _row("NVDA", "kronos_mini", sched))
        assert ds.get_session_runs() == []
        run_id = rs.create_run("manual", ["AMD"], None, prediction_date=TODAY,
                               target_date=TODAY + timedelta(days=1))
        rs.set_status(run_id, "running")
        assert [r["run"]["status"] for r in ds.get_session_runs()] == ["running"]
        rs.set_status(run_id, "done")
        assert [r["run"]["status"] for r in ds.get_session_runs()] == ["done"]

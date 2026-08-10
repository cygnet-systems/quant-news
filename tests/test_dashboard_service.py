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

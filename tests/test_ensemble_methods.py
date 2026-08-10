"""The run dialog's worked example must match what the combiner really does.

The method selector shows a formula and one scenario resolved four ways, so a
user can pick by seeing the answers diverge. Documentation that drifts from the
code is worse than none here: it would talk someone into a method on the
strength of arithmetic the ensemble no longer performs.

These tests replay the documented scenario through the real EnsembleModel and
assert every published figure.
"""

import pandas as pd
import pytest

from layouts.modals import (
    ENSEMBLE_EXAMPLE_MEMBERS,
    ENSEMBLE_EXAMPLE_PROB,
    ENSEMBLE_METHOD_DETAIL,
    ENSEMBLE_METHODS,
    ensemble_method_detail,
)
from models.ensemble_model import METHODS, EnsembleModel

# The display names in the example map onto these registry ids.
MODEL_IDS = ["kronos_mini", "xgboost_shap", "lightgbm", "deberta_sentiment"]


@pytest.fixture
def scenario():
    members, weights = {}, {}
    for (_, dec, conf, p_up, w), mid in zip(ENSEMBLE_EXAMPLE_MEMBERS, MODEL_IDS):
        members[mid] = {"decision": dec, "confidence": conf, "up_probability": p_up}
        weights[mid] = w
    return members, weights


def run(method, scenario, min_agree=3):
    members, weights = scenario
    return EnsembleModel().predict(
        "TEST", pd.DataFrame({"Close": [100.0] * 60}),
        other_results=members,
        ensemble_config={"enabled_models": list(members), "weights": weights,
                         "method": method, "min_agree": min_agree},
    )


class TestDocumentedOutcomes:
    @pytest.mark.parametrize("method", list(ENSEMBLE_METHOD_DETAIL))
    def test_verdict_matches_the_real_combiner(self, method, scenario):
        assert run(method, scenario).decision == ENSEMBLE_METHOD_DETAIL[method]["verdict"]

    def test_the_methods_actually_diverge(self, scenario):
        """If they all agreed, the explainer would not be worth showing."""
        verdicts = {m: run(m, scenario).decision for m in ENSEMBLE_METHOD_DETAIL}
        assert set(verdicts.values()) == {"BUY", "HOLD"}, verdicts
        assert verdicts["majority"] == verdicts["confidence_weighted"] == "BUY"
        assert verdicts["prob_mean"] == verdicts["agreement"] == "HOLD"

    @pytest.mark.parametrize("method", list(ENSEMBLE_METHOD_DETAIL))
    def test_confidence_is_method_independent(self, method, scenario):
        """The panel claims switching method never changes the confidence."""
        r = run(method, scenario)
        assert round(r.up_probability, 2) == ENSEMBLE_EXAMPLE_PROB


class TestPublishedArithmetic:
    """Spot-check the numbers printed in the worked-example steps."""

    def test_confidence_weighted_score(self, scenario):
        d = run("confidence_weighted", scenario).details
        assert d["weighted_score"] == pytest.approx(1.15, abs=0.005)
        assert d["direction_agreement"] == pytest.approx(0.46, abs=0.005)

    def test_majority_score(self, scenario):
        d = run("majority", scenario).details
        assert d["weighted_score"] == pytest.approx(1.5, abs=0.005)
        assert d["direction_agreement"] == pytest.approx(0.375, abs=0.005)

    def test_prob_mean_falls_just_short_of_the_threshold(self, scenario):
        """The example's whole point: +0.12 does not clear +0.15."""
        from config import MODEL
        d = run("prob_mean", scenario).details
        assert d["direction_agreement"] == pytest.approx(0.12, abs=0.005)
        assert 0.12 < MODEL.ENSEMBLE_BUY_THRESHOLD


class TestGateIsHonoured:
    def test_lowering_min_agree_still_cannot_open_a_vetoed_gate(self, scenario):
        """Two BUYs and one SELL: the dissenter vetoes at any threshold."""
        assert run("agreement", scenario, min_agree=2).decision == "HOLD"

    def test_gate_opens_when_nobody_dissents(self):
        members = {"kronos_mini": {"decision": "BUY", "confidence": 0.6,
                                   "up_probability": 0.8},
                   "xgboost_shap": {"decision": "BUY", "confidence": 0.6,
                                    "up_probability": 0.8},
                   "lightgbm": {"decision": "HOLD", "confidence": 0.5,
                                "up_probability": 0.5}}
        weights = {k: 1.0 for k in members}
        assert run("agreement", (members, weights), min_agree=2).decision == "BUY"


class TestPanelIntegrity:
    def test_every_selectable_method_has_a_detail_panel(self):
        for value, _, _ in ENSEMBLE_METHODS:
            assert value in ENSEMBLE_METHOD_DETAIL, f"{value} has no explainer"

    def test_every_documented_method_is_a_real_combiner_method(self):
        assert set(ENSEMBLE_METHOD_DETAIL) == set(METHODS)

    def test_the_gate_threshold_echoes_the_live_value(self):
        """The consensus text must not hardcode the default when it is changed."""
        rendered = str(ensemble_method_detail("agreement", min_agree=4))
        assert "≥ 4" in rendered and "≥ 3" not in rendered

    def test_an_unknown_method_renders_empty_rather_than_raising(self):
        assert ensemble_method_detail("no_such_method") is not None

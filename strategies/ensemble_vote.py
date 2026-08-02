"""Ensemble vote strategy — majority-weighted vote across models.

Requires context: all model predictions for the same (symbol, target_date).
Uses configurable weights from config.MODEL.ENSEMBLE_*_WEIGHT.
Needs at least 2 non-error predictions to act.
"""

from config import MODEL
from strategies.base import BaseStrategy, StrategySignal

# Model name → weight mapping (used for strategy evaluations)
_MODEL_WEIGHTS = {
    "kronos_mini": MODEL.ENSEMBLE_KRONOS_WEIGHT,
    "xgboost_shap": MODEL.ENSEMBLE_XGBOOST_WEIGHT,
    "lightgbm": 1.0,
    "deberta_sentiment": 0.6,
    "trading_agents": MODEL.ENSEMBLE_TRADING_AGENTS_WEIGHT,
}

_DIRECTION_MAP = {"BUY": 1.0, "SELL": -1.0, "HOLD": 0.0}


class EnsembleVoteStrategy(BaseStrategy):
    """Weighted majority vote across all models for the same symbol+date."""

    requires_context = True

    @property
    def name(self) -> str:
        return "ensemble_vote"

    def evaluate(self, prediction, context=None):
        if not context:
            return StrategySignal(action="SKIP")

        # Filter to non-error predictions
        valid = [p for p in context if not p.get("error")]
        if len(valid) < 2:
            return StrategySignal(
                action="SKIP",
                metadata={"reason": f"insufficient_models ({len(valid)})"},
            )

        # Compute weighted score
        weighted_score = 0.0
        total_weight = 0.0
        votes = {}

        for p in valid:
            model = p.get("model_name", "")
            decision = p.get("decision", "HOLD")
            weight = _MODEL_WEIGHTS.get(model, 1.0)
            direction = _DIRECTION_MAP.get(decision, 0.0)

            weighted_score += weight * direction
            total_weight += weight
            votes[model] = decision

        if total_weight == 0:
            return StrategySignal(action="SKIP")

        normalized = weighted_score / total_weight

        if normalized > MODEL.ENSEMBLE_BUY_THRESHOLD:
            action = "BUY"
        elif normalized < MODEL.ENSEMBLE_SELL_THRESHOLD:
            action = "SELL"
        else:
            action = "SKIP"

        return StrategySignal(
            action=action,
            metadata={
                "weighted_score": round(weighted_score, 3),
                "normalized_score": round(normalized, 3),
                "votes": votes,
                "models_used": len(valid),
            },
        )

"""Live ensemble model — weighted majority vote across individual models.

Runs after all individual models complete (Phase 3). Receives their
results via other_results kwarg. Uses ensemble_config from the UI
to determine which models participate and their weights.

All configuration is transparent — metadata shows exactly which
models voted, their weights, and the computed score.
"""

import logging
from typing import Optional

import pandas as pd

from config import MODEL
from models.base import BaseModel, PredictionResult

logger = logging.getLogger(__name__)

_DIRECTION_MAP = {"BUY": 1.0, "SELL": -1.0, "HOLD": 0.0}


class EnsembleModel(BaseModel):
    """Weighted majority vote ensemble across individual models."""

    @property
    def name(self) -> str:
        return "ensemble"

    def is_ready(self) -> bool:
        return True

    def predict(
        self,
        symbol: str,
        ohlcv_df: pd.DataFrame,
        **kwargs,
    ) -> PredictionResult:
        """Combine individual model predictions via weighted vote.

        Required kwargs:
            other_results: dict[str, dict] — model_name → result dict
                from Phase 1+2 predictions.

        Optional kwargs:
            ensemble_config: dict — from UI store:
                {"enabled_models": [...], "weights": {model: weight, ...}}
                If not provided, uses config defaults.
        """
        other_results = kwargs.get("other_results", {})
        ensemble_config = kwargs.get("ensemble_config")

        # Resolve config: UI store takes precedence, then config defaults
        if ensemble_config:
            enabled_models = set(ensemble_config.get("enabled_models", []))
            weights = ensemble_config.get("weights", {})
        else:
            enabled_models = set(MODEL.ENSEMBLE_DEFAULT_ENABLED)
            weights = dict(MODEL.ENSEMBLE_DEFAULT_WEIGHTS)

        if not other_results:
            # No inputs means no vote — an error result is never persisted,
            # where a 0-confidence HOLD row would be scored as if decided.
            return PredictionResult(
                model_name=self.name,
                decision="HOLD",
                confidence=0.0,
                up_probability=0.5,
                error="no individual model results available",
                details={
                    "models_enabled": sorted(enabled_models),
                },
            )

        # Filter to enabled, non-error results
        valid = {}
        excluded = []
        for model_name, result in other_results.items():
            if model_name not in enabled_models:
                excluded.append(model_name)
                continue
            if result.get("error"):
                excluded.append(model_name)
                continue
            valid[model_name] = result

        if len(valid) < 2:
            return PredictionResult(
                model_name=self.name,
                decision="HOLD",
                confidence=0.0,
                up_probability=0.5,
                error=f"insufficient enabled models ({len(valid)})",
                details={
                    "models_enabled": sorted(enabled_models),
                    "models_excluded": sorted(excluded),
                    "models_valid": sorted(valid.keys()),
                },
            )

        # Confidence-weighted vote: each model's vote counts as
        # (config weight x its own confidence). This decides the DIRECTION.
        # It deliberately does not decide the confidence — see below.
        weighted_score = 0.0
        total_weight = 0.0
        prob_sum = 0.0
        prob_weight = 0.0
        votes = {}
        weights_used = {}

        for model_name, result in valid.items():
            decision = result.get("decision", "HOLD")
            weight = float(weights.get(model_name, 1.0))
            model_conf = result.get("confidence")
            model_conf = float(model_conf) if model_conf is not None else 0.5
            direction = _DIRECTION_MAP.get(decision, 0.0)

            effective = weight * model_conf
            weighted_score += effective * direction
            total_weight += effective
            votes[model_name] = f"{decision} ({model_conf:.0%})"
            weights_used[model_name] = round(effective, 3)

            # Members' own probabilities, weighted by config weight only.
            member_p = result.get("up_probability")
            if member_p is None:
                # No probability published: fall back to the member's stated
                # confidence pushed to the side it actually voted.
                member_p = 0.5 + direction * (model_conf / 2.0)
            prob_sum += weight * float(member_p)
            prob_weight += weight

        if total_weight == 0:
            return PredictionResult(
                model_name=self.name,
                decision="HOLD",
                confidence=0.0,
                up_probability=0.5,
                details={"reason": "zero total weight"},
            )

        normalized = weighted_score / total_weight

        if normalized > MODEL.ENSEMBLE_BUY_THRESHOLD:
            action = "BUY"
        elif normalized < MODEL.ENSEMBLE_SELL_THRESHOLD:
            action = "SELL"
        else:
            action = "HOLD"

        # Confidence and up_probability come from the MEAN MEMBER PROBABILITY,
        # not from `normalized`. When every member agrees on direction the
        # per-model confidences cancel in weighted_score/total_weight, so
        # `normalized` is exactly +/-1 no matter how unsure the members were,
        # pinning confidence at 1.0. That fired on ~45% of predictions, whose
        # realised up-rate was 0.499 — the ensemble claimed certainty on a coin
        # flip. Measured over a 2024-2026 walk-forward plus a 2026-04..07
        # holdout, this change improves Brier from 0.408 to 0.263 (design) and
        # 0.381 to 0.255 (holdout); a constant 0.5 forecast scores 0.25, so the
        # old formula was worse than declining to answer. It adds no edge — it
        # stops the ensemble overstating what it knows.
        up_probability = prob_sum / prob_weight if prob_weight else 0.5
        confidence = min(abs(up_probability - 0.5) * 2, 1.0)

        # `normalized` is retained as a directional-agreement diagnostic. It is
        # a measure of consensus, not of confidence, and is named accordingly.
        return PredictionResult(
            model_name=self.name,
            decision=action,
            confidence=round(confidence, 2),
            up_probability=round(up_probability, 4),
            details={
                "votes": votes,
                "weights_used": weights_used,
                "weighted_score": round(weighted_score, 3),
                "normalized_score": round(normalized, 3),
                "direction_agreement": round(abs(normalized), 3),
                "models_enabled": sorted(enabled_models),
                "models_excluded": sorted(excluded),
                "models_used": len(valid),
            },
        )

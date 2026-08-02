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
            return PredictionResult(
                model_name=self.name,
                decision="HOLD",
                confidence=0.0,
                up_probability=0.5,
                details={
                    "reason": "no individual model results available",
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
                details={
                    "reason": f"insufficient enabled models ({len(valid)})",
                    "models_enabled": sorted(enabled_models),
                    "models_excluded": sorted(excluded),
                    "models_valid": sorted(valid.keys()),
                },
            )

        # Compute confidence-weighted vote: each model's vote counts as
        # (config weight x its own confidence). A unanimous BUY from three
        # low-confidence models no longer produces 100% ensemble confidence,
        # and HOLD votes dilute the lean instead of being ignored.
        weighted_score = 0.0
        total_weight = 0.0
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

        # Confidence: how strongly the ensemble leans one way
        confidence = min(abs(normalized), 1.0)

        # Up probability: map normalized score [-1, 1] → [0, 1]
        up_probability = (normalized + 1) / 2

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
                "models_enabled": sorted(enabled_models),
                "models_excluded": sorted(excluded),
                "models_used": len(valid),
            },
        )

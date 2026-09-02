"""Live ensemble model, combines individual models under a selectable method.

Runs after all individual models complete (Phase 3). Receives their
results via other_results kwarg. Uses ensemble_config from the UI
to determine which models participate, their weights, and how the
votes are combined (see METHODS).

All configuration is transparent, metadata shows exactly which
models voted, their weights, the method, and the computed score.
"""

import logging
from typing import Optional

import pandas as pd

from config import MODEL
from models.base import BaseModel, PredictionResult

logger = logging.getLogger(__name__)

_DIRECTION_MAP = {"BUY": 1.0, "SELL": -1.0, "HOLD": 0.0}

# Combination methods. Every method takes its confidence/up_probability from
# the weighted mean member probability (the calibrated formula, see the Brier
# note in predict()); they differ only in how the DIRECTION is decided.
#   confidence_weighted: each vote counts weight x the member's own
#       confidence. Platform evals found member confidence carries little
#       calibration signal, so this mostly behaves like `majority` with noise.
#   majority: each vote counts its config weight only. One model, one
#       (weighted) vote; transparent baseline.
#   prob_mean: direction from the weighted mean up-probability itself,
#       thresholded. The best-calibrated signal the members publish.
#   agreement: trade only on consensus: at least `min_agree` members back one
#       direction and none back the other, else HOLD. Weights are ignored.
#       Cuts trade count, which matters because whipsaw, not direction, is
#       the documented failure mode (R2000 diversity test, 2026-07).
METHODS = ("confidence_weighted", "majority", "prob_mean", "agreement")


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
            other_results: dict[str, dict], model_name → result dict
                from Phase 1+2 predictions.

        Optional kwargs:
            ensemble_config: dict: from UI store:
                {"enabled_models": [...], "weights": {model: weight, ...},
                 "method": one of METHODS, "min_agree": int}
                If not provided, uses config defaults.
        """
        other_results = kwargs.get("other_results", {})
        ensemble_config = kwargs.get("ensemble_config")

        # Resolve config: UI store takes precedence, then config defaults
        if ensemble_config:
            enabled_models = set(ensemble_config.get("enabled_models", []))
            weights = ensemble_config.get("weights", {})
        else:
            ensemble_config = {}
            enabled_models = set(MODEL.ENSEMBLE_DEFAULT_ENABLED)
            weights = dict(MODEL.ENSEMBLE_DEFAULT_WEIGHTS)

        method = ensemble_config.get("method") or MODEL.ENSEMBLE_DEFAULT_METHOD
        if method not in METHODS:
            logger.warning(f"unknown ensemble method {method!r}, using default")
            method = MODEL.ENSEMBLE_DEFAULT_METHOD
        try:
            min_agree = int(ensemble_config.get("min_agree")
                            or MODEL.ENSEMBLE_MIN_AGREE)
        except (TypeError, ValueError):
            min_agree = MODEL.ENSEMBLE_MIN_AGREE

        if not other_results:
            # No inputs means no vote. An error result is never persisted,
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

        # The vote weight decides the DIRECTION only, per `method`. It
        # deliberately does not decide the confidence, see below.
        #
        # Direction weights run through two evaluated-history corrections:
        #   * calibrate(): what this member's raw confidence has historically
        #     meant (isotonic fit). Raw confidences are anti-calibrated here,
        #     and `confidence_weighted` on raw numbers rewards overconfidence.
        #     Members without enough history get shrunk halfway toward 0.5.
        #   * rolling_hit_rate(), decays chronically-wrong members instead of
        #     letting config weights carry them at full strength; clamped to
        #     [0.5x, 1.5x] so one hot or cold month cannot zero a member out.
        # The probability aggregation below stays config-weight-only: that
        # formula is Brier-validated as-is and is not re-weighted lightly.
        try:
            from functools import partial

            from services.calibration_service import (
                calibrate as _calibrate, rolling_hit_rate as _rolling_hit_rate)
            # Bounded by the run's cutoff: on a backtest the fits and the
            # decay factor used to read outcomes from after the as-of date.
            calibrate = partial(_calibrate, as_of=kwargs.get("as_of"))
            rolling_hit_rate = partial(_rolling_hit_rate, as_of=kwargs.get("as_of"))
        except Exception:
            calibrate = rolling_hit_rate = None  # type: ignore

        weighted_score = 0.0
        total_weight = 0.0
        prob_sum = 0.0
        prob_weight = 0.0
        buy_votes = 0
        sell_votes = 0
        votes = {}
        weights_used = {}
        perf_factors = {}

        for model_name, result in valid.items():
            decision = result.get("decision", "HOLD")
            weight = float(weights.get(model_name, 1.0))
            model_conf = result.get("confidence")
            model_conf = float(model_conf) if model_conf is not None else 0.5
            direction = _DIRECTION_MAP.get(decision, 0.0)

            cal_conf = calibrate(model_name, model_conf) if calibrate else None
            if cal_conf is None:
                cal_conf = 0.5 + (model_conf - 0.5) * 0.5

            perf = rolling_hit_rate(model_name) if rolling_hit_rate else None
            perf_factor = (min(1.5, max(0.5, perf / 0.5))
                           if perf is not None else 1.0)
            perf_factors[model_name] = round(perf_factor, 2)

            if method == "confidence_weighted":
                effective = weight * cal_conf * perf_factor
            elif method == "agreement":
                effective = 1.0
            else:  # majority, prob_mean, config weight x performance decay
                effective = weight * perf_factor
            weighted_score += effective * direction
            total_weight += effective
            if direction > 0:
                buy_votes += 1
            elif direction < 0:
                sell_votes += 1
            votes[model_name] = (f"{decision} (cal {cal_conf:.0%}"
                                 + (f", 30d {perf:.0%}" if perf is not None else "")
                                 + ")")
            weights_used[model_name] = round(effective, 3)

            # Members' own probabilities, weighted by config weight only.
            member_p = result.get("up_probability")
            if member_p is None:
                # No probability published: fall back to the member's
                # calibrated confidence pushed to the side it actually voted.
                member_p = 0.5 + direction * (cal_conf / 2.0)
            prob_sum += weight * float(member_p)
            prob_weight += weight

        if total_weight == 0:
            return PredictionResult(
                model_name=self.name,
                decision="HOLD",
                confidence=0.0,
                up_probability=0.5,
                details={"reason": "zero total weight", "method": method},
            )

        mean_member_p = prob_sum / prob_weight if prob_weight else 0.5

        # `normalized` is the directional score in [-1, 1] that the standard
        # thresholds are applied to; what it measures depends on the method.
        if method == "prob_mean":
            normalized = (mean_member_p - 0.5) * 2.0
        else:
            normalized = weighted_score / total_weight

        if method == "agreement":
            # Consensus gate: trade only when enough members back one side
            # and nobody backs the other. min_agree above the number of
            # running members means the gate can never open, the UI warns,
            # and the details below make the abstention auditable.
            if buy_votes >= min_agree and sell_votes == 0:
                action = "BUY"
            elif sell_votes >= min_agree and buy_votes == 0:
                action = "SELL"
            else:
                action = "HOLD"
        elif normalized > MODEL.ENSEMBLE_BUY_THRESHOLD:
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
        # realised up-rate was 0.499: the ensemble claimed certainty on a coin
        # flip. Measured over a 2024-2026 walk-forward plus a 2026-04..07
        # holdout, this change improves Brier from 0.408 to 0.263 (design) and
        # 0.381 to 0.255 (holdout); a constant 0.5 forecast scores 0.25, so the
        # old formula was worse than declining to answer. It adds no edge, it
        # stops the ensemble overstating what it knows.
        up_probability = mean_member_p
        confidence = min(abs(up_probability - 0.5) * 2, 1.0)

        details = {
            "method": method,
            "votes": votes,
            "weights_used": weights_used,
            "perf_factors": perf_factors,
            "weighted_score": round(weighted_score, 3),
            "normalized_score": round(normalized, 3),
            "direction_agreement": round(abs(normalized), 3),
            "models_enabled": sorted(enabled_models),
            "models_excluded": sorted(excluded),
            "models_used": len(valid),
        }
        if method == "agreement":
            details.update({
                "buy_votes": buy_votes,
                "sell_votes": sell_votes,
                "min_agree": min_agree,
            })

        # `normalized` is retained as a directional-agreement diagnostic. It is
        # a measure of consensus, not of confidence, and is named accordingly.
        return PredictionResult(
            model_name=self.name,
            decision=action,
            confidence=round(confidence, 2),
            up_probability=round(up_probability, 4),
            details=details,
        )

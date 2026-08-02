"""Confidence threshold strategy — only act on high-conviction signals.

Filters out low-confidence predictions. Only executes when model
confidence exceeds a configurable threshold (default from config).
"""

from config import STRATEGY
from strategies.base import BaseStrategy, StrategySignal


class ConfidenceThresholdStrategy(BaseStrategy):
    """Only trade when model confidence exceeds threshold."""

    @property
    def name(self) -> str:
        return "confidence_threshold"

    @property
    def version(self) -> str:
        return f"1.0_t{STRATEGY.CONFIDENCE_THRESHOLD}"

    def evaluate(self, prediction, context=None):
        decision = prediction.get("decision", "HOLD")
        confidence = prediction.get("confidence", 0.0) or 0.0

        if decision in ("BUY", "SELL") and confidence >= STRATEGY.CONFIDENCE_THRESHOLD:
            return StrategySignal(
                action=decision,
                metadata={"threshold": STRATEGY.CONFIDENCE_THRESHOLD},
            )
        return StrategySignal(action="SKIP")

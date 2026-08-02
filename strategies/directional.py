"""Directional strategy — follow the model's signal directly.

If model says BUY, go long. If SELL, go short. HOLD maps to SKIP.
Simplest possible baseline strategy.
"""

from strategies.base import BaseStrategy, StrategySignal


class DirectionalStrategy(BaseStrategy):
    """Follow the model's BUY/SELL decision without filtering."""

    @property
    def name(self) -> str:
        return "directional"

    def evaluate(self, prediction, context=None):
        decision = prediction.get("decision", "HOLD")
        if decision in ("BUY", "SELL"):
            return StrategySignal(action=decision)
        return StrategySignal(action="SKIP")

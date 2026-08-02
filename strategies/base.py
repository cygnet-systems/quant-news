"""Strategy base classes for prediction evaluation.

Defines the plugin contract: subclass BaseStrategy, implement evaluate(),
drop the file in strategies/ — auto-discovered by StrategyRegistry.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class StrategySignal:
    """Output from a strategy for a single prediction evaluation.

    Attributes:
        action: Trading action — "BUY", "SELL", "HOLD", or "SKIP".
        position_size: Notional position size in dollars.
        metadata: Strategy-specific context (thresholds, votes, etc.).
    """

    action: str
    position_size: float = 1000.0
    metadata: dict = field(default_factory=dict)


class BaseStrategy(ABC):
    """Abstract base class for all trading strategies.

    To create a new strategy:
        1. Create a new file in strategies/ (e.g., strategies/my_strategy.py)
        2. Subclass BaseStrategy
        3. Implement name property and evaluate() method
        4. The strategy is auto-discovered on next app/CLI start

    Set requires_context = True if the strategy needs all model predictions
    for the same (symbol, target_date) — used by ensemble strategies.
    """

    requires_context: bool = False

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique strategy identifier (e.g., 'directional')."""

    @property
    def version(self) -> str:
        """Strategy version for tracking parameter changes."""
        return "1.0"

    @abstractmethod
    def evaluate(
        self,
        prediction: dict,
        context: Optional[list[dict]] = None,
    ) -> StrategySignal:
        """Generate a trading signal from an evaluated prediction.

        Args:
            prediction: Row from model_predictions (actual_close filled).
            context: For ensemble strategies — all predictions for the
                     same (symbol, target_date). None for single-model strategies.

        Returns:
            StrategySignal with action and optional metadata.
        """

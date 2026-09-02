"""Base model classes and shared prediction infrastructure.

All prediction models inherit from BaseModel and return PredictionResult.
All P&L computation uses compute_pnl(). The single canonical function.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd


# Fixed notional per trade (dollars)
POSITION_SIZE_USD: float = 1000.0

# Direction multiplier for P&L calculation
DIRECTION_SIGN: dict[str, int] = {
    "BUY": +1,
    "SELL": -1,
    "HOLD": 0,
}


@dataclass
class PredictionResult:
    """Standardized prediction output from any model.

    Attributes:
        model_name: Identifier for the model that produced this prediction.
        decision: Trading decision, "BUY", "SELL", or "HOLD".
        confidence: Model confidence in [0, 1]. Interpretation varies by model:
            - XGBoost: class probability (trained probabilistic output)
            - Kronos: Monte Carlo sample proportion
            - LLM: self-reported certainty (not calibrated)
        up_probability: Probability of price going up in [0, 1].
        details: Model-specific metadata (reasoning, feature values, etc.).
        error: Error message if prediction failed, None otherwise.
        predicted_close: Predicted closing price (Kronos only, optional).
        model_version: Version string for debugging and reproducibility.
    """

    model_name: str
    decision: str
    confidence: float
    up_probability: float
    details: dict = field(default_factory=dict)
    error: Optional[str] = None
    predicted_close: Optional[float] = None
    model_version: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate and normalize prediction fields."""
        self.decision = self.decision.upper()
        if self.decision not in ("BUY", "SELL", "HOLD"):
            self.decision = "HOLD"
        self.confidence = max(0.0, min(1.0, self.confidence))
        self.up_probability = max(0.0, min(1.0, self.up_probability))

    def to_dict(self) -> dict:
        """Serialize to dictionary for dcc.Store transport."""
        return {
            "model_name": self.model_name,
            "decision": self.decision,
            "confidence": self.confidence,
            "up_probability": self.up_probability,
            "details": self.details,
            "error": self.error,
            "predicted_close": self.predicted_close,
            "model_version": self.model_version,
        }

def compute_pnl(
    decision: str,
    previous_close: float,
    actual_close: float,
    position_size: float = POSITION_SIZE_USD,
) -> float:
    """Compute P&L for a fixed-notional directional trade.

    BUY:  long  $1,000 -- profit if actual > previous
    SELL: short $1,000 -- profit if actual < previous
    HOLD: no position  -- P&L = 0

    Args:
        decision: Trading decision ("BUY", "SELL", or "HOLD").
        previous_close: Closing price on prediction date.
        actual_close: Actual closing price on target date.
        position_size: Dollar amount of the position.

    Returns:
        Dollars gained/lost (negative = loss).
    """
    sign = DIRECTION_SIGN.get(decision.upper(), 0)
    if sign == 0 or previous_close <= 0:
        return 0.0
    shares = position_size / previous_close
    return shares * (actual_close - previous_close) * sign


def apply_torch_thread_cap() -> None:
    """Honour ``TORCH_NUM_THREADS`` before the first torch model loads.

    In a container torch sizes its intra-op pool to the HOST's cores, and
    it shares the box with xgboost's and lightgbm's OpenMP pools; the env
    var lets ops pin it next to OMP_NUM_THREADS without a code change.
    Unset means torch's own default. Never raises.
    """
    import logging
    import os
    raw = os.getenv("TORCH_NUM_THREADS")
    if not raw:
        return
    try:
        import torch
        n = max(1, int(raw))
        if torch.get_num_threads() != n:
            torch.set_num_threads(n)
            logging.getLogger(__name__).info(f"torch intra-op threads capped at {n}")
    except Exception as e:  # noqa: BLE001
        logging.getLogger(__name__).debug(f"torch thread cap skipped: {e}")


class BaseModel(ABC):
    """Abstract base class for all prediction models."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique model identifier (e.g., 'kronos_mini', 'xgboost_shap')."""

    @abstractmethod
    def is_ready(self) -> bool:
        """Check if the model is ready to make predictions.

        Returns:
            True if all dependencies are available and model can predict.
        """

    @abstractmethod
    def predict(
        self,
        symbol: str,
        ohlcv_df: pd.DataFrame,
        **kwargs,
    ) -> PredictionResult:
        """Generate a next-day BUY/SELL/HOLD prediction.

        Args:
            symbol: Stock ticker symbol.
            ohlcv_df: DataFrame with OHLCV data (title-case columns).
            **kwargs: Model-specific arguments (spy_df, sector_df, news, etc.).

        Returns:
            PredictionResult with the prediction and metadata.
        """

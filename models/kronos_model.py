"""Kronos time-series foundation model wrapper.

Uses vendored Kronos-mini from TradingAgents. Generates next-day
directional predictions from OHLCV patterns using Monte Carlo sampling.
"""

import logging
from datetime import date

import numpy as np
import pandas as pd

from config import MODEL
from models.base import BaseModel, PredictionResult

logger = logging.getLogger(__name__)

try:
    import torch
    from models.kronos import KronosTokenizer, Kronos, KronosPredictor

    KRONOS_AVAILABLE = True
except ImportError as e:
    KRONOS_AVAILABLE = False
    _KRONOS_IMPORT_ERROR = str(e)

# HuggingFace model IDs
_TOKENIZER_REPO = "NeoQuasar/Kronos-Tokenizer-base"
_MODEL_REPO = f"NeoQuasar/Kronos-{MODEL.KRONOS_MODEL_SIZE}"

# Singleton predictor (lazy loaded)
_predictor = None


def _get_predictor() -> "KronosPredictor":
    """Lazy-load Kronos predictor (downloads weights on first call)."""
    global _predictor
    if _predictor is not None:
        return _predictor

    logger.info(f"Loading Kronos-{MODEL.KRONOS_MODEL_SIZE} (first call downloads weights)...")
    from models.base import apply_torch_thread_cap
    apply_torch_thread_cap()

    tokenizer = KronosTokenizer.from_pretrained(_TOKENIZER_REPO)
    model = Kronos.from_pretrained(_MODEL_REPO)

    tokenizer.eval()
    model.eval()

    # Force CPU in background subprocesses. MPS deadlocks after fork() on macOS.
    # The env var is set by the background callback in app.py.
    import os
    device = "cpu" if os.environ.get("_DASH_BG_SUBPROCESS") else None
    _predictor = KronosPredictor(model, tokenizer, device=device)
    logger.info(f"Kronos loaded on device: {_predictor.device}")
    return _predictor


class KronosModel(BaseModel):
    """Kronos-mini directional prediction model."""

    @property
    def name(self) -> str:
        return "kronos_mini"

    def is_ready(self) -> bool:
        return KRONOS_AVAILABLE

    def predict(
        self,
        symbol: str,
        ohlcv_df: pd.DataFrame,
        **kwargs,
    ) -> PredictionResult:
        """Generate next-day prediction using Kronos Monte Carlo sampling.

        Takes the last KRONOS_CONTEXT_BARS bars, generates KRONOS_PRED_DAYS
        future bars with KRONOS_SAMPLE_COUNT Monte Carlo samples, and
        determines direction from the median predicted close vs current close.

        Args:
            symbol: Stock ticker symbol.
            ohlcv_df: DataFrame with OHLCV data (title-case columns).

        Returns:
            PredictionResult with BUY/SELL/HOLD decision.
        """
        if not KRONOS_AVAILABLE:
            return PredictionResult(
                model_name=self.name,
                decision="HOLD",
                confidence=0.0,
                up_probability=0.5,
                details={},
                error=f"Kronos not available: {_KRONOS_IMPORT_ERROR}",
            )

        try:
            predictor = _get_predictor()
            return self._run_prediction(predictor, symbol, ohlcv_df)
        except Exception as e:
            logger.error(f"Kronos prediction failed for {symbol}: {e}")
            return PredictionResult(
                model_name=self.name,
                decision="HOLD",
                confidence=0.0,
                up_probability=0.5,
                details={},
                error=str(e),
            )

    def _run_prediction(
        self,
        predictor: "KronosPredictor",
        symbol: str,
        ohlcv_df: pd.DataFrame,
    ) -> PredictionResult:
        """Core prediction logic."""
        context_bars = MODEL.KRONOS_CONTEXT_BARS

        # Prepare input DataFrame with lowercase columns for Kronos
        df = ohlcv_df.tail(context_bars).copy()
        df = df.rename(columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        })

        # Ensure required columns exist
        for col in ["open", "high", "low", "close"]:
            if col not in df.columns:
                raise ValueError(f"Missing column: {col}")

        if "volume" not in df.columns:
            df["volume"] = 0.0

        # Drop NaN rows
        df = df.dropna(subset=["open", "high", "low", "close"])
        if len(df) < 30:
            raise ValueError(f"Insufficient data: {len(df)} bars (need >= 30)")

        # Build timestamps
        x_timestamp = pd.DatetimeIndex(df.index)

        # Generate future timestamps for prediction
        from utils.trading_calendar import get_next_trading_day

        pred_days = MODEL.KRONOS_PRED_DAYS
        last_date = x_timestamp[-1].date()
        future_dates = []
        current = last_date
        for _ in range(pred_days):
            current = get_next_trading_day(current)
            future_dates.append(pd.Timestamp(current))
        y_timestamp = pd.DatetimeIndex(future_dates)

        # Run Monte Carlo predictions
        sample_count = MODEL.KRONOS_SAMPLE_COUNT
        pred_closes = []

        for _ in range(sample_count):
            pred_df = predictor.predict(
                df[["open", "high", "low", "close", "volume"]],
                x_timestamp,
                y_timestamp,
                pred_len=pred_days,
                T=MODEL.KRONOS_TEMPERATURE,
                top_k=0,
                top_p=0.9,
                sample_count=1,
                verbose=False,
            )
            pred_closes.append(float(pred_df["close"].iloc[0]))

        # Analyze results
        current_close = float(df["close"].iloc[-1])
        pred_close_median = float(np.median(pred_closes))
        up_count = sum(1 for c in pred_closes if c > current_close)
        up_probability = up_count / sample_count

        if up_probability > MODEL.BUY_THRESHOLD:
            decision = "BUY"
        elif up_probability < MODEL.SELL_THRESHOLD:
            decision = "SELL"
        else:
            decision = "HOLD"

        confidence = max(up_probability, 1 - up_probability)

        return PredictionResult(
            model_name=self.name,
            decision=decision,
            confidence=round(confidence, 2),
            up_probability=round(up_probability, 4),
            predicted_close=round(pred_close_median, 2),
            details={
                "model_version": "kronos_mini_v1",
                "context_bars": len(df),
                "sample_count": sample_count,
                "pred_close_median": round(pred_close_median, 2),
                "pred_close_std": round(float(np.std(pred_closes)), 2),
                "current_close": round(current_close, 2),
            },
        )

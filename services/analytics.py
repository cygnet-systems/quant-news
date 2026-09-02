"""Technical analysis service.

This module provides functions to calculate technical indicators
including trend, momentum, volatility, and volume indicators.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import ta
from ta.momentum import RSIIndicator, StochasticOscillator, ROCIndicator
from ta.trend import MACD, SMAIndicator, EMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator

from config import INDICATORS


@dataclass
class TechnicalIndicators:
    """Container for all calculated technical indicators."""

    # Trend
    sma_20: pd.Series
    sma_50: pd.Series
    sma_200: pd.Series
    ema_12: pd.Series
    ema_26: pd.Series
    macd_line: pd.Series
    macd_signal: pd.Series
    macd_histogram: pd.Series

    # Momentum
    rsi: pd.Series
    stochastic_k: pd.Series
    stochastic_d: pd.Series
    roc: pd.Series

    # Volatility
    bollinger_upper: pd.Series
    bollinger_mid: pd.Series
    bollinger_lower: pd.Series
    atr: pd.Series
    rolling_std: pd.Series

    # Volume
    obv: pd.Series
    volume_ma: pd.Series


def calculate_sma(close: pd.Series, period: int) -> pd.Series:
    """Calculate Simple Moving Average.

    Args:
        close: Close price series.
        period: Number of periods.

    Returns:
        SMA series.
    """
    indicator = SMAIndicator(close=close, window=period)
    return indicator.sma_indicator()


def calculate_ema(close: pd.Series, period: int) -> pd.Series:
    """Calculate Exponential Moving Average.

    Args:
        close: Close price series.
        period: Number of periods.

    Returns:
        EMA series.
    """
    indicator = EMAIndicator(close=close, window=period)
    return indicator.ema_indicator()


def calculate_macd(
    close: pd.Series,
    fast: int = INDICATORS.MACD_FAST,
    slow: int = INDICATORS.MACD_SLOW,
    signal: int = INDICATORS.MACD_SIGNAL,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate MACD indicator.

    Args:
        close: Close price series.
        fast: Fast EMA period.
        slow: Slow EMA period.
        signal: Signal line period.

    Returns:
        Tuple of (MACD line, signal line, histogram).
    """
    macd = MACD(close=close, window_fast=fast, window_slow=slow, window_sign=signal)
    return macd.macd(), macd.macd_signal(), macd.macd_diff()


def calculate_rsi(
    close: pd.Series,
    period: int = INDICATORS.RSI_PERIOD,
) -> pd.Series:
    """Calculate Relative Strength Index.

    Args:
        close: Close price series.
        period: RSI period.

    Returns:
        RSI series (0-100).
    """
    indicator = RSIIndicator(close=close, window=period)
    return indicator.rsi()


def calculate_stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = INDICATORS.STOCHASTIC_K,
    d_period: int = INDICATORS.STOCHASTIC_D,
) -> tuple[pd.Series, pd.Series]:
    """Calculate Stochastic Oscillator.

    Args:
        high: High price series.
        low: Low price series.
        close: Close price series.
        k_period: %K period.
        d_period: %D smoothing period.

    Returns:
        Tuple of (%K, %D) series.
    """
    stoch = StochasticOscillator(
        high=high, low=low, close=close, window=k_period, smooth_window=d_period
    )
    return stoch.stoch(), stoch.stoch_signal()


def calculate_roc(close: pd.Series, period: int = 12) -> pd.Series:
    """Calculate Rate of Change.

    Args:
        close: Close price series.
        period: ROC period.

    Returns:
        ROC series (percentage).
    """
    indicator = ROCIndicator(close=close, window=period)
    return indicator.roc()


def calculate_bollinger_bands(
    close: pd.Series,
    period: int = INDICATORS.BOLLINGER_PERIOD,
    std_dev: float = INDICATORS.BOLLINGER_STD,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate Bollinger Bands.

    Args:
        close: Close price series.
        period: Moving average period.
        std_dev: Number of standard deviations.

    Returns:
        Tuple of (upper band, middle band, lower band).
    """
    bb = BollingerBands(close=close, window=period, window_dev=std_dev)
    return bb.bollinger_hband(), bb.bollinger_mavg(), bb.bollinger_lband()


def calculate_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = INDICATORS.ATR_PERIOD,
) -> pd.Series:
    """Calculate Average True Range.

    Args:
        high: High price series.
        low: Low price series.
        close: Close price series.
        period: ATR period.

    Returns:
        ATR series.
    """
    atr = AverageTrueRange(high=high, low=low, close=close, window=period)
    return atr.average_true_range()


def calculate_rolling_std(
    close: pd.Series,
    period: int = 20,
) -> pd.Series:
    """Calculate rolling standard deviation.

    Args:
        close: Close price series.
        period: Rolling window period.

    Returns:
        Rolling standard deviation series.
    """
    # ddof=0 to agree with the ta BollingerBands columns this lands next to
    # in the same DataFrame, with the pandas default (ddof=1),
    # (BB_Upper - BB_Mid) / 2 != Rolling_Std and the chart hover's
    # "= SMA(20) + 2 x Std Dev" claim was false.
    return close.rolling(window=period).std(ddof=0)


def calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Calculate On-Balance Volume.

    Args:
        close: Close price series.
        volume: Volume series.

    Returns:
        OBV series.
    """
    obv = OnBalanceVolumeIndicator(close=close, volume=volume)
    return obv.on_balance_volume()


def calculate_volume_ma(
    volume: pd.Series,
    period: int = INDICATORS.VOLUME_MA_PERIOD,
) -> pd.Series:
    """Calculate Volume Moving Average.

    Args:
        volume: Volume series.
        period: Moving average period.

    Returns:
        Volume MA series.
    """
    return volume.rolling(window=period).mean()


def calculate_all_indicators(df: pd.DataFrame) -> TechnicalIndicators:
    """Calculate all technical indicators from OHLCV data.

    Args:
        df: DataFrame with Open, High, Low, Close, Volume columns.

    Returns:
        TechnicalIndicators dataclass with all calculated indicators.
    """
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    # Trend indicators
    sma_20 = calculate_sma(close, 20)
    sma_50 = calculate_sma(close, 50)
    sma_200 = calculate_sma(close, 200)
    ema_12 = calculate_ema(close, 12)
    ema_26 = calculate_ema(close, 26)
    macd_line, macd_signal, macd_histogram = calculate_macd(close)

    # Momentum indicators
    rsi = calculate_rsi(close)
    stoch_k, stoch_d = calculate_stochastic(high, low, close)
    roc = calculate_roc(close)

    # Volatility indicators
    bb_upper, bb_mid, bb_lower = calculate_bollinger_bands(close)
    atr = calculate_atr(high, low, close)
    rolling_std = calculate_rolling_std(close)

    # Volume indicators
    obv = calculate_obv(close, volume)
    volume_ma = calculate_volume_ma(volume)

    return TechnicalIndicators(
        sma_20=sma_20,
        sma_50=sma_50,
        sma_200=sma_200,
        ema_12=ema_12,
        ema_26=ema_26,
        macd_line=macd_line,
        macd_signal=macd_signal,
        macd_histogram=macd_histogram,
        rsi=rsi,
        stochastic_k=stoch_k,
        stochastic_d=stoch_d,
        roc=roc,
        bollinger_upper=bb_upper,
        bollinger_mid=bb_mid,
        bollinger_lower=bb_lower,
        atr=atr,
        rolling_std=rolling_std,
        obv=obv,
        volume_ma=volume_ma,
    )


def add_indicators_to_df(df: pd.DataFrame) -> pd.DataFrame:
    """Add all technical indicators as columns to DataFrame.

    Args:
        df: DataFrame with OHLCV data.

    Returns:
        DataFrame with additional indicator columns.
    """
    result = df.copy()
    indicators = calculate_all_indicators(df)

    # Add all indicators as columns
    result["SMA_20"] = indicators.sma_20
    result["SMA_50"] = indicators.sma_50
    result["SMA_200"] = indicators.sma_200
    result["EMA_12"] = indicators.ema_12
    result["EMA_26"] = indicators.ema_26
    result["MACD"] = indicators.macd_line
    result["MACD_Signal"] = indicators.macd_signal
    result["MACD_Histogram"] = indicators.macd_histogram
    result["RSI"] = indicators.rsi
    result["Stoch_K"] = indicators.stochastic_k
    result["Stoch_D"] = indicators.stochastic_d
    result["ROC"] = indicators.roc
    result["BB_Upper"] = indicators.bollinger_upper
    result["BB_Mid"] = indicators.bollinger_mid
    result["BB_Lower"] = indicators.bollinger_lower
    result["ATR"] = indicators.atr
    result["Rolling_Std"] = indicators.rolling_std
    result["OBV"] = indicators.obv
    result["Volume_MA"] = indicators.volume_ma

    return result


def get_latest_signals(df: pd.DataFrame) -> dict:
    """Get latest technical signals from indicator data.

    Args:
        df: DataFrame with indicator columns (from add_indicators_to_df).

    Returns:
        Dictionary with signal assessments.
    """
    if df.empty:
        return {}

    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest

    signals = {}

    # RSI signal
    rsi_val = latest.get("RSI")
    if rsi_val is not None:
        if rsi_val > INDICATORS.RSI_OVERBOUGHT:
            signals["rsi"] = {"value": rsi_val, "signal": "overbought"}
        elif rsi_val < INDICATORS.RSI_OVERSOLD:
            signals["rsi"] = {"value": rsi_val, "signal": "oversold"}
        else:
            signals["rsi"] = {"value": rsi_val, "signal": "neutral"}

    # MACD signal (crossover)
    macd = latest.get("MACD")
    macd_signal = latest.get("MACD_Signal")
    prev_macd = prev.get("MACD")
    prev_signal = prev.get("MACD_Signal")

    if all(v is not None for v in [macd, macd_signal, prev_macd, prev_signal]):
        if prev_macd < prev_signal and macd > macd_signal:
            signals["macd"] = {"value": macd, "signal": "bullish_crossover"}
        elif prev_macd > prev_signal and macd < macd_signal:
            signals["macd"] = {"value": macd, "signal": "bearish_crossover"}
        elif macd > macd_signal:
            signals["macd"] = {"value": macd, "signal": "bullish"}
        else:
            signals["macd"] = {"value": macd, "signal": "bearish"}

    # Price vs SMA (trend)
    close = latest.get("Close")
    sma_50 = latest.get("SMA_50")
    sma_200 = latest.get("SMA_200")

    if close is not None and sma_50 is not None:
        if close > sma_50:
            signals["trend_50"] = {"signal": "above_sma50", "bullish": True}
        else:
            signals["trend_50"] = {"signal": "below_sma50", "bullish": False}

    if close is not None and sma_200 is not None:
        if close > sma_200:
            signals["trend_200"] = {"signal": "above_sma200", "bullish": True}
        else:
            signals["trend_200"] = {"signal": "below_sma200", "bullish": False}

    # Golden/Death Cross
    if sma_50 is not None and sma_200 is not None:
        prev_sma_50 = prev.get("SMA_50")
        prev_sma_200 = prev.get("SMA_200")
        if prev_sma_50 is not None and prev_sma_200 is not None:
            if prev_sma_50 < prev_sma_200 and sma_50 > sma_200:
                signals["cross"] = {"signal": "golden_cross", "bullish": True}
            elif prev_sma_50 > prev_sma_200 and sma_50 < sma_200:
                signals["cross"] = {"signal": "death_cross", "bullish": False}

    # Bollinger Band position
    bb_upper = latest.get("BB_Upper")
    bb_lower = latest.get("BB_Lower")
    if close is not None and bb_upper is not None and bb_lower is not None:
        if close > bb_upper:
            signals["bollinger"] = {"signal": "above_upper", "extreme": True}
        elif close < bb_lower:
            signals["bollinger"] = {"signal": "below_lower", "extreme": True}
        else:
            signals["bollinger"] = {"signal": "within_bands", "extreme": False}

    return signals


def calculate_performance_metrics(df: pd.DataFrame) -> dict:
    """Calculate comprehensive performance metrics.

    Args:
        df: DataFrame with OHLCV data.

    Returns:
        Dictionary with performance metrics.
    """
    if df.empty:
        return {}

    close = df["Close"]

    # Basic returns
    start_price = close.iloc[0]
    end_price = close.iloc[-1]
    total_return = ((end_price - start_price) / start_price) * 100

    # Daily returns
    daily_returns = close.pct_change().dropna()

    # Annualized metrics (assuming 252 trading days)
    trading_days = 252

    # Volatility (annualized)
    volatility = daily_returns.std() * np.sqrt(trading_days) * 100

    # Sharpe Ratio (assuming 0% risk-free rate for simplicity)
    mean_return = daily_returns.mean() * trading_days
    sharpe = mean_return / (daily_returns.std() * np.sqrt(trading_days)) if daily_returns.std() > 0 else 0

    # Max Drawdown
    cumulative = (1 + daily_returns).cumprod()
    rolling_max = cumulative.expanding().max()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_drawdown = drawdown.min() * 100

    # Win rate
    positive_days = (daily_returns > 0).sum()
    total_days = len(daily_returns)
    win_rate = (positive_days / total_days) * 100 if total_days > 0 else 0

    # Best/Worst days
    best_day = daily_returns.max() * 100
    worst_day = daily_returns.min() * 100

    return {
        "total_return": round(total_return, 2),
        "volatility": round(volatility, 2),
        "sharpe_ratio": round(sharpe, 2),
        "max_drawdown": round(max_drawdown, 2),
        "win_rate": round(win_rate, 1),
        "best_day": round(best_day, 2),
        "worst_day": round(worst_day, 2),
        "start_price": round(start_price, 2),
        "end_price": round(end_price, 2),
        "start_date": df.index[0].strftime("%Y-%m-%d"),
        "end_date": df.index[-1].strftime("%Y-%m-%d"),
        "trading_days": len(df),
    }

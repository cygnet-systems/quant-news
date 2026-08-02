"""Stock data service using yfinance.

This module provides functions to fetch stock data, company info,
and basic metrics from Yahoo Finance.
"""

import atexit
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import yfinance as yf

from config import APP

# Ticker cache with TTL to avoid creating multiple Ticker objects
_ticker_cache: dict[str, tuple[yf.Ticker, datetime]] = {}
_TICKER_CACHE_TTL = timedelta(minutes=5)


def get_ticker(symbol: str) -> yf.Ticker:
    """Get or create cached yfinance Ticker object.

    Caches Ticker objects for 5 minutes to reduce redundant
    network requests when multiple functions need the same ticker.

    Args:
        symbol: Stock ticker symbol.

    Returns:
        Cached or new yf.Ticker instance.
    """
    symbol = symbol.upper().strip()
    now = datetime.now()

    if symbol in _ticker_cache:
        ticker, cached_at = _ticker_cache[symbol]
        if now - cached_at < _TICKER_CACHE_TTL:
            return ticker

    ticker = yf.Ticker(symbol)
    _ticker_cache[symbol] = (ticker, now)
    return ticker


def clear_ticker_cache(symbol: Optional[str] = None) -> None:
    """Clear ticker cache.

    Args:
        symbol: Specific symbol to clear, or None for all.
    """
    global _ticker_cache
    if symbol:
        _ticker_cache.pop(symbol.upper().strip(), None)
    else:
        _ticker_cache.clear()


# Register cleanup on shutdown to prevent semaphore leaks from yfinance Ticker objects
atexit.register(clear_ticker_cache)


@dataclass
class StockInfo:
    """Company information and current metrics."""

    symbol: str
    name: str
    sector: str
    industry: str
    market_cap: float
    current_price: float
    previous_close: float
    day_change: float
    day_change_percent: float
    volume: int
    avg_volume: int
    fifty_two_week_high: float
    fifty_two_week_low: float
    pe_ratio: Optional[float]
    dividend_yield: Optional[float]


def fetch_stock_data(
    symbol: str,
    period: str = APP.DEFAULT_PERIOD,
) -> pd.DataFrame:
    """Fetch historical stock data from yfinance.

    Args:
        symbol: Stock ticker symbol (e.g., "MSFT")
        period: Time period for historical data. Valid values:
            1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max

    Returns:
        DataFrame with OHLCV data indexed by date. Columns:
            Open, High, Low, Close, Volume, Dividends, Stock Splits

    Raises:
        ValueError: If symbol is invalid or data unavailable
    """
    try:
        ticker = get_ticker(symbol)
        df = ticker.history(period=period)

        if df.empty:
            raise ValueError(f"No data available for symbol: {symbol}")

        # Ensure index is datetime
        df.index = pd.to_datetime(df.index)
        df.index = df.index.tz_localize(None)  # Remove timezone for simplicity

        return df

    except Exception as e:
        raise ValueError(f"Failed to fetch data for {symbol}: {str(e)}") from e


# Intraday bars are chart-only: never written to the daily StockPrice cache
# and never placed in the shared data store, so models/metrics always consume
# daily data. Short in-process TTL absorbs chart re-renders.
_INTRADAY_CACHE: dict[tuple, tuple[float, pd.DataFrame]] = {}
_INTRADAY_TTL_SECONDS = 60


def fetch_intraday(
    symbol: str,
    yf_period: str = "1d",
    interval: str = "5m",
    sessions: Optional[int] = None,
) -> pd.DataFrame:
    """Fetch intraday OHLCV for the price chart.

    Args:
        symbol: Ticker.
        yf_period: yfinance period ("1d" / "5d").
        interval: bar interval ("5m" / "30m").
        sessions: keep only the last N trading sessions (e.g. 3D slices a
                  5d fetch down to 3 sessions).
    """
    import time as _time

    key = (symbol.upper(), yf_period, interval)
    hit = _INTRADAY_CACHE.get(key)
    if hit and _time.time() - hit[0] < _INTRADAY_TTL_SECONDS:
        df = hit[1]
    else:
        df = get_ticker(symbol).history(period=yf_period, interval=interval)
        if not df.empty and df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        _INTRADAY_CACHE[key] = (_time.time(), df)

    if sessions and not df.empty:
        keep = set(sorted({ts.date() for ts in df.index})[-sessions:])
        df = df[[ts.date() in keep for ts in df.index]]
    return df


def get_company_profile(symbol: str, max_chars: int = 900) -> str:
    """Return a short company background block: name, sector, industry,
    and the business summary (truncated). Empty string on failure.
    """
    try:
        info = get_ticker(symbol).info or {}
        summary = (info.get("longBusinessSummary") or "").strip()
        if len(summary) > max_chars:
            cut = summary[:max_chars].rfind(". ")
            summary = summary[:cut + 1] if cut > max_chars * 0.5 else summary[:max_chars]
        parts = [
            f"{info.get('shortName', symbol)} — {info.get('sector', 'N/A')} / {info.get('industry', 'N/A')}",
        ]
        if summary:
            parts.append(summary)
        return "\n".join(parts)
    except Exception:
        return ""


def get_stock_info(symbol: str) -> StockInfo:
    """Get company information and current metrics.

    Args:
        symbol: Stock ticker symbol (e.g., "MSFT")

    Returns:
        StockInfo dataclass with company details and metrics

    Raises:
        ValueError: If symbol is invalid or info unavailable
    """
    try:
        ticker = get_ticker(symbol)
        info = ticker.info

        if not info or "shortName" not in info:
            raise ValueError(f"No info available for symbol: {symbol}")

        current_price = info.get("currentPrice") or info.get("regularMarketPrice", 0)
        previous_close = info.get("previousClose", 0)
        day_change = current_price - previous_close
        day_change_percent = (day_change / previous_close * 100) if previous_close else 0

        return StockInfo(
            symbol=symbol.upper(),
            name=info.get("shortName", symbol),
            sector=info.get("sector", "N/A"),
            industry=info.get("industry", "N/A"),
            market_cap=info.get("marketCap", 0),
            current_price=current_price,
            previous_close=previous_close,
            day_change=day_change,
            day_change_percent=day_change_percent,
            volume=info.get("volume", 0),
            avg_volume=info.get("averageVolume", 0),
            fifty_two_week_high=info.get("fiftyTwoWeekHigh", 0),
            fifty_two_week_low=info.get("fiftyTwoWeekLow", 0),
            pe_ratio=info.get("trailingPE"),
            dividend_yield=info.get("dividendYield"),
        )

    except Exception as e:
        raise ValueError(f"Failed to get info for {symbol}: {str(e)}") from e


def validate_symbol(symbol: str) -> bool:
    """Check if a stock symbol is valid.

    Args:
        symbol: Stock ticker symbol to validate

    Returns:
        True if symbol exists and has data, False otherwise
    """
    try:
        ticker = get_ticker(symbol)
        # Try to get basic info - if it fails, symbol is invalid
        info = ticker.info
        return bool(info and info.get("shortName"))
    except Exception:
        return False


def get_multiple_stocks(
    symbols: list[str],
    period: str = APP.DEFAULT_PERIOD,
) -> dict[str, pd.DataFrame]:
    """Fetch data for multiple stocks.

    Args:
        symbols: List of stock ticker symbols
        period: Time period for historical data

    Returns:
        Dictionary mapping symbol to DataFrame
    """
    result: dict[str, pd.DataFrame] = {}

    for symbol in symbols:
        try:
            df = fetch_stock_data(symbol, period)
            result[symbol.upper()] = df
        except ValueError:
            # Skip invalid symbols
            continue

    return result

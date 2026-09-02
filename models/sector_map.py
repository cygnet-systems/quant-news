"""Sector-to-ETF mapping for sector context (ML features + research reports).

Resolution is metadata-driven: a symbol's OWN industry/sector names (from its
info metadata) are mapped name -> ETF. An industry-level ETF (e.g.
Semiconductors -> SMH) is preferred when one is mapped. It is a much tighter
comparator than a broad sector fund; otherwise the sector-level SPDR is used.
There is deliberately no ticker -> ETF table. A hardcoded symbol map silently
mislabels anything it doesn't know (e.g. it sent Consumer-Defensive UNFI to
XLK) and goes stale as listings change. Falls back to SPY when metadata is
unavailable, which callers treat as "no distinct sector context". The resolved
level ("industry" / "sector" / "unknown") is reported so downstream text can
say which proxy it is using.
"""

import logging
import threading
from typing import Optional

logger = logging.getLogger(__name__)

# Sector name -> SPDR sector ETF
SECTOR_TO_ETF: dict[str, str] = {
    "Technology": "XLK",
    "Healthcare": "XLV",
    "Financials": "XLF",
    "Consumer Defensive": "XLP",
    "Consumer Cyclical": "XLY",
    "Energy": "XLE",
    "Utilities": "XLU",
    "Industrials": "XLI",
    "Real Estate": "XLRE",
    "Basic Materials": "XLB",
    "Communication Services": "XLC",
    # yfinance sector name aliases
    "Consumer Staples": "XLP",
    "Consumer Discretionary": "XLY",
    "Materials": "XLB",
    "Information Technology": "XLK",
    "Health Care": "XLV",
    "Financial Services": "XLF",
}

# Industry name -> industry ETF, preferred over the broad sector fund when
# mapped (XLK is dominated by mega-caps; SMH/XBI/IGV track the actual cohort).
# Deliberately sparse: only liquid, well-known industry ETFs. Unmapped
# industries fall through to the sector-level SPDR.
INDUSTRY_TO_ETF: dict[str, str] = {
    "Semiconductors": "SMH",
    "Semiconductor Equipment & Materials": "SMH",
    "Biotechnology": "XBI",
    "Software - Application": "IGV",
    "Software - Infrastructure": "IGV",
    "Information Technology Services": "XSW",
    "Banks - Regional": "KRE",
    "Aerospace & Defense": "ITA",
    "Oil & Gas E&P": "XOP",
    "Residential Construction": "XHB",
    "Gold": "GDX",
}

# Ticker -> direct business peers. The sector ETF (XLK etc.) is a weak
# benchmark for mid-caps: XLK is dominated by AAPL/MSFT/NVDA. Relative
# strength vs actual peers distinguishes company-specific repricing from
# sector-wide repricing.
PEER_MAP: dict[str, list[str]] = {
    # IT services / consulting
    "EPAM": ["ACN", "GLOB", "CTSH", "DAVA"],
    "ACN": ["EPAM", "CTSH", "IBM", "INFY"],
    "CTSH": ["ACN", "EPAM", "INFY", "WIT"],
    "GLOB": ["EPAM", "DAVA", "ACN", "CTSH"],
    # Mega-cap tech
    "AAPL": ["MSFT", "GOOGL", "AMZN", "META"],
    "MSFT": ["AAPL", "GOOGL", "AMZN", "ORCL"],
    "GOOGL": ["META", "MSFT", "AMZN", "AAPL"],
    "META": ["GOOGL", "SNAP", "PINS", "MSFT"],
    "AMZN": ["MSFT", "GOOGL", "WMT", "BABA"],
    # Semis
    "NVDA": ["AMD", "AVGO", "TSM", "QCOM"],
    "AMD": ["NVDA", "INTC", "QCOM", "AVGO"],
    "INTC": ["AMD", "TSM", "MU", "TXN"],
    "AMBA": ["NVDA", "AMD", "LSCC", "SLAB"],
    # Utilities (CA-exposed)
    "EIX": ["PCG", "SRE", "PNW", "POR"],
    "PCG": ["EIX", "SRE", "PNW", "POR"],
    # EV / auto
    "TSLA": ["RIVN", "LCID", "GM", "F"],
    # Progressive-style insurers
    "PGR": ["ALL", "TRV", "CB", "HIG"],
    # Life / annuity insurers (variable-annuity and spread businesses)
    "BHF": ["JXN", "CRBG", "LNC", "PRU"],
    "JXN": ["BHF", "CRBG", "LNC", "PRU"],
    "CRBG": ["BHF", "JXN", "LNC", "EQH"],
    "LNC": ["BHF", "JXN", "PRU", "MET"],
    "MET": ["PRU", "LNC", "AFL", "PFG"],
    "HIG": ["TRV", "ALL", "CB", "PGR"],
}

# Runtime cache: symbol -> resolved info dict
_sector_cache: dict[str, dict] = {}
_sector_lock = threading.Lock()


def get_peers(symbol: str, limit: int = 4) -> list[str]:
    """Return the direct-peer set for a ticker (empty list if unmapped)."""
    return PEER_MAP.get(symbol.upper(), [])[:limit]


def get_sector_info(symbol: str) -> dict:
    """Resolve the context ETF from the symbol's own industry/sector metadata.

    Industry-level ETF preferred when mapped; sector-level SPDR otherwise.
    Sector identity is static company metadata, so a live lookup is
    lookahead-safe in backtests (same rationale as the business profile).

    Returns:
        {"etf", "sector", "industry", "level"} where level is "industry"
        (industry ETF matched), "sector" (sector SPDR), or "unknown"
        (metadata unavailable: etf falls back to SPY, which callers should
        treat as "no distinct sector context", not as a sector).
    """
    symbol = symbol.upper()
    cached = _sector_cache.get(symbol)
    if cached is not None:
        return cached

    # One fetch per symbol, serialized. The prediction pipeline resolves the
    # sector from several models at once (Phase 2 runs XGBoost, LightGBM and
    # the research agent concurrently); without this lock they all missed the
    # cache together and yfinance returned an EMPTY info dict to some of them.
    # Those callers silently got etf="SPY", so the sector features were a
    # duplicate of the SPY features. Measured at 11 of 40 GBM predictions in
    # one 20-symbol run, and nondeterministic between models on one symbol.
    with _sector_lock:
        cached = _sector_cache.get(symbol)
        if cached is not None:
            return cached

        sector = industry = ""
        try:
            from services.stock_data import get_ticker
            info = get_ticker(symbol).info or {}
            sector = (info.get("sector") or "").strip()
            industry = (info.get("industry") or "").strip()
        except Exception as e:
            logger.warning(f"Sector lookup failed for {symbol}: {e}")

        if industry and industry in INDUSTRY_TO_ETF:
            etf, level = INDUSTRY_TO_ETF[industry], "industry"
        elif sector in SECTOR_TO_ETF:
            etf, level = SECTOR_TO_ETF[sector], "sector"
        else:
            etf, level = "SPY", "unknown"
            if sector:
                logger.warning(f"Unmapped sector name for {symbol}: {sector!r}")

        result = {"etf": etf, "sector": sector or "unknown",
                  "industry": industry or "unknown", "level": level}

        if level == "unknown" and not sector:
            # An empty info dict is a transient vendor failure, not a fact about
            # the company. Caching it would pin every later consumer in this
            # process to the SPY fallback; leaving it uncached costs one retry.
            logger.warning(
                f"Sector metadata unavailable for {symbol}: falling back to "
                f"SPY for this call only (not cached; sector features will be "
                f"a duplicate of SPY wherever this result is used)"
            )
            return result

        _sector_cache[symbol] = result
        logger.info(f"Sector lookup: {symbol} -> {sector or 'unknown'} / "
                    f"{industry or 'unknown'} -> {etf} ({level})")
        return result


def get_sector_etf(symbol: str) -> str:
    """Context ETF for a ticker, resolved from its industry/sector metadata."""
    return get_sector_info(symbol)["etf"]

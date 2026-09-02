"""Options positioning from point-in-time chains.

Put/call ratios per symbol per date via Alpha Vantage HISTORICAL_OPTIONS,
which serves the full chain (volume, open interest) as it stood on a given
trading day — so the same call is lookahead-safe live and in backtests.

The 2026-08-07 one-day experiment (20 symbols) showed the volume P/C carries
only a weak directional lean (low-P/C names outperformed high-P/C by ~1.2pp,
not significant at n=20) and the OI ratio none at all. These numbers are
therefore surfaced to the research/synthesis models as POSITIONING CONTEXT,
never as a standalone timing rule.
"""

import logging
import threading
from datetime import date, timedelta

import requests

from config import API
from services.rate_limiter import alpha_vantage_bucket

logger = logging.getLogger(__name__)

# Below this many contracts traded the ratio is noise (a 2-lot flips it).
MIN_LIQUID_VOLUME = 500

_CACHE: dict[tuple[str, str], dict | None] = {}
_CACHE_LOCK = threading.Lock()


def _chain_date(as_of: str) -> str:
    """AV serves chains for trading days only; walk back over weekends."""
    d = date.fromisoformat(str(as_of)[:10])
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d.isoformat()


def get_put_call_metrics(symbol: str, as_of: str) -> dict | None:
    """Aggregate put/call volume and open interest as of a trading day.

    Tiered: the Terminal's warmed L2 cache first (nightly post-close warmer,
    no network cost beyond a local cache read), then Alpha Vantage
    HISTORICAL_OPTIONS. Returns None when neither has the chain (unlisted
    options, throttled, no key) — callers must treat that as "no data",
    not neutral positioning.
    """
    chain_date = _chain_date(as_of)
    key = (symbol.upper(), chain_date)
    with _CACHE_LOCK:
        if key in _CACHE:
            return _CACHE[key]

    try:
        from services import terminal_cache
        warm = terminal_cache.get_putcall_summary(symbol, chain_date)
    except Exception:
        warm = None
    if warm:
        result = _aggregate_warm(warm, chain_date)
        with _CACHE_LOCK:
            _CACHE[key] = result
        return result

    if not API.ALPHA_VANTAGE_API_KEY:
        return None

    result = None
    try:
        # Shares the quota with the news fetches; pace against the same bucket.
        alpha_vantage_bucket().acquire(timeout=API.DEFAULT_TIMEOUT * 4)
        response = requests.get(
            API.ALPHA_VANTAGE_BASE_URL,
            params={
                "function": "HISTORICAL_OPTIONS",
                "symbol": symbol.upper(),
                "date": chain_date,
                "apikey": API.ALPHA_VANTAGE_API_KEY,
            },
            timeout=API.DEFAULT_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()
        # AV signals throttling/bad requests with HTTP 200 and a message body.
        for k in ("Note", "Information", "Error Message"):
            if k in data:
                logger.warning(f"{symbol}: HISTORICAL_OPTIONS {k}: "
                               f"{str(data[k])[:150]}")
                return None  # transient — do not cache
        contracts = data.get("data") or []
        if contracts:
            result = _aggregate(contracts, chain_date)
    except Exception as e:
        logger.warning(f"{symbol}: options chain fetch failed: {e}")
        return None

    with _CACHE_LOCK:
        _CACHE[key] = result
    return result


def _aggregate_warm(summary: dict, chain_date: str) -> dict:
    """Metrics from the Terminal warmer's per-expiry summary. Covers
    expirations within its 90-day cutoff — near-dated positioning, vs the
    all-expiry AV aggregate."""
    put_vol = call_vol = put_oi = call_oi = 0
    for e in summary.get("expiries") or []:
        put_vol += int(e.get("put_vol") or 0)
        call_vol += int(e.get("call_vol") or 0)
        put_oi += int(e.get("put_oi") or 0)
        call_oi += int(e.get("call_oi") or 0)
    m = _metrics(put_vol, call_vol, put_oi, call_oi, chain_date)
    m["source"] = "terminal_warm_cache"
    m["expiry_window"] = "≤90d"
    return m


def _aggregate(contracts: list[dict], chain_date: str) -> dict:
    put_vol = call_vol = put_oi = call_oi = 0
    for c in contracts:
        vol = int(float(c.get("volume") or 0))
        oi = int(float(c.get("open_interest") or 0))
        if c.get("type") == "put":
            put_vol += vol
            put_oi += oi
        elif c.get("type") == "call":
            call_vol += vol
            call_oi += oi
    m = _metrics(put_vol, call_vol, put_oi, call_oi, chain_date)
    m["source"] = "alpha_vantage"
    m["expiry_window"] = "all"
    return m


def _metrics(put_vol: int, call_vol: int, put_oi: int, call_oi: int,
             chain_date: str) -> dict:
    total_vol = put_vol + call_vol
    pc_volume = round(put_vol / call_vol, 3) if call_vol else None
    pc_oi = round(put_oi / call_oi, 3) if call_oi else None

    if total_vol < MIN_LIQUID_VOLUME:
        read = "low-liquidity"
    elif pc_volume is None:
        read = "no-call-volume"
    elif pc_volume >= 1.0:
        read = "put-tilted"
    elif pc_volume <= 0.5:
        read = "call-tilted"
    else:
        read = "balanced"

    return {
        "as_of": chain_date,
        "put_volume": put_vol,
        "call_volume": call_vol,
        "total_volume": total_vol,
        "put_oi": put_oi,
        "call_oi": call_oi,
        "pc_volume": pc_volume,
        "pc_oi": pc_oi,
        "read": read,
    }


def get_put_call_by_expiry(symbol: str, as_of: str) -> dict | None:
    """Vendor-computed put/call volume ratio per expiration as of a session
    (Alpha Vantage HISTORICAL_PUT_CALL_RATIO, dated → point-in-time).

    Complements the whole-chain aggregate above: a 6x full-chain ratio can
    hide that the front month is 13x and the next is 0.1x. Returns
    {"as_of", "full_chain", "by_expiry": [(date, ratio), ...]} or None.
    """
    chain_date = _chain_date(as_of)
    key = (symbol.upper() + ":expiry", chain_date)
    with _CACHE_LOCK:
        if key in _CACHE:
            return _CACHE[key]
    if not API.ALPHA_VANTAGE_API_KEY:
        return None
    result = None
    try:
        alpha_vantage_bucket().acquire(timeout=API.DEFAULT_TIMEOUT * 4)
        response = requests.get(
            API.ALPHA_VANTAGE_BASE_URL,
            params={"function": "HISTORICAL_PUT_CALL_RATIO",
                    "symbol": symbol.upper(), "date": chain_date,
                    "apikey": API.ALPHA_VANTAGE_API_KEY},
            timeout=API.DEFAULT_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()
        for k in ("Note", "Information", "Error Message"):
            if k in data:
                logger.warning(f"{symbol}: HISTORICAL_PUT_CALL_RATIO {k}: "
                               f"{str(data[k])[:150]}")
                return None
        rows = []
        for r in data.get("put_call_ratio_by_expiration") or []:
            try:
                if r.get("value") is not None:
                    rows.append((str(r.get("date"))[:10], float(r["value"])))
            except (TypeError, ValueError):
                continue
        full = data.get("put_call_ratio_full_chain")
        try:
            full = float(full) if full is not None else None
        except (TypeError, ValueError):
            full = None
        if rows or full is not None:
            result = {"as_of": str(data.get("date") or chain_date)[:10],
                      "full_chain": full, "by_expiry": rows[:6]}
    except Exception as e:
        logger.warning(f"{symbol}: put/call by expiry fetch failed: {e}")
        return None
    with _CACHE_LOCK:
        _CACHE[key] = result
    return result


def format_options_block(symbol: str, m: dict | None,
                         by_expiry: dict | None = None) -> str:
    """Prompt block. Empty string when no data — never a fabricated neutral."""
    if not m:
        return ""
    window = (" (expirations within 90d)"
              if m.get("expiry_window") == "≤90d" else "")
    lines = [f"[{symbol} — options positioning from the chain as of "
             f"{m['as_of']}{window}]"]
    pcv = f"{m['pc_volume']:.2f}" if m.get("pc_volume") is not None else "n/a"
    pcoi = f"{m['pc_oi']:.2f}" if m.get("pc_oi") is not None else "n/a"
    lines.append(
        f"Put/Call volume ratio: {pcv} "
        f"({m['put_volume']:,} puts vs {m['call_volume']:,} calls traded)"
    )
    lines.append(
        f"Put/Call open-interest ratio: {pcoi} "
        f"({m['put_oi']:,} put OI vs {m['call_oi']:,} call OI)"
    )
    lines.append(f"Session read: {m['read']}")
    if by_expiry and by_expiry.get("by_expiry"):
        full = by_expiry.get("full_chain")
        lines.append(
            "Vendor put/call volume by expiration"
            + (f" (full chain {full:.2f})" if full is not None else "")
            + ": " + " | ".join(f"{d} {v:.2f}" for d, v in by_expiry["by_expiry"])
            + " — a front-month skew that the whole-chain figure averages away "
              "is event positioning; a skew spread evenly is a stance."
        )
    if m["read"] == "low-liquidity":
        lines.append(
            f"Only {m['total_volume']:,} contracts traded — too thin to read; "
            f"ignore this block for direction."
        )
    else:
        lines.append(
            "Treat as positioning context only (hedging demand vs speculation), "
            "not a standalone timing signal; heavy puts can be protective "
            "hedging on a name holders refuse to sell."
        )
    return "\n".join(lines)

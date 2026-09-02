"""Political and institutional flow blocks from Alpha Vantage.

Two point-in-time-aware evidence blocks for the research prompt:

* Congressional trades (``CONGRESS_TRADES``). STOCK Act disclosures by
  House and Senate members, with party, chamber, owner (self/spouse/child)
  and the amount band. Point-in-time on ``filed_date``: a trade is visible
  only from the day it was filed, which is what a reader on ``as_of`` could
  have known. Disclosure lags the trade by up to 45 days, so this is
  positioning context, never a timing signal.
* Institutional holdings (``INSTITUTIONAL_HOLDINGS``). 13F holders adding
  vs cutting, with the largest movers. Holder rows carry the quarter-end
  they were reported for and are filtered to ``last_reported <= as_of``;
  the vendor's aggregate counts are a current snapshot and are labelled as
  such rather than presented as point-in-time.

Both endpoints were verified against the project key on 2026-09-01. Each
is one Alpha Vantage call per symbol per run and shares the news bucket.
"""

import logging
import threading
from datetime import date, timedelta

import requests

from config import API
from services.rate_limiter import alpha_vantage_bucket

logger = logging.getLogger(__name__)

_CACHE: dict[tuple, dict | None] = {}
_CACHE_LOCK = threading.Lock()

CONGRESS_WINDOW_DAYS = 180
INSTITUTIONAL_TOP_N = 4


class AlphaVantageUnavailable(RuntimeError):
    """The vendor answered with a throttle/limit message, not data."""


def _av_get(params: dict) -> dict:
    if not API.ALPHA_VANTAGE_API_KEY:
        raise AlphaVantageUnavailable("no ALPHA_VANTAGE_API_KEY")
    alpha_vantage_bucket().acquire(timeout=API.DEFAULT_TIMEOUT * 4)
    response = requests.get(
        API.ALPHA_VANTAGE_BASE_URL,
        params={**params, "apikey": API.ALPHA_VANTAGE_API_KEY},
        timeout=API.DEFAULT_TIMEOUT,
    )
    response.raise_for_status()
    data = response.json()
    for k in ("Note", "Information", "Error Message"):
        if k in data:
            raise AlphaVantageUnavailable(f"{params.get('function')} {k}: "
                                          f"{str(data[k])[:150]}")
    return data


def _iso(d: str | None) -> date | None:
    try:
        return date.fromisoformat(str(d)[:10]) if d else None
    except ValueError:
        return None


def _band(lo, hi) -> str:
    try:
        lo_f, hi_f = float(lo), float(hi)
    except (TypeError, ValueError):
        return "amount n/a"

    def _k(v: float) -> str:
        return f"${v / 1e6:.1f}M" if v >= 1e6 else f"${v / 1e3:.0f}K"
    return f"{_k(lo_f)}–{_k(hi_f)}"


# ---------------------------------------------------------------------------
# Congressional trades
# ---------------------------------------------------------------------------

def get_congress_trades(symbol: str, as_of: str,
                        days: int = CONGRESS_WINDOW_DAYS) -> dict:
    """Trades in ``symbol`` filed on or before ``as_of`` whose transaction
    date falls in the last ``days``. Raises AlphaVantageUnavailable on a
    vendor throttle so the caller can record a gap instead of "no trades".
    """
    as_of_d = date.fromisoformat(str(as_of)[:10])
    key = ("congress", symbol.upper(), as_of_d.isoformat(), days)
    with _CACHE_LOCK:
        if key in _CACHE:
            return _CACHE[key]

    data = _av_get({"function": "CONGRESS_TRADES", "symbol": symbol.upper()})
    floor = as_of_d - timedelta(days=days)
    trades = []
    for t in data.get("trades") or []:
        # Visibility is the filing date; fall back to the notification date
        # (Senate rows carry no filed_date on some records). A row with
        # neither cannot be placed in time and is dropped.
        visible = _iso(t.get("filed_date")) or _iso(t.get("notification_date"))
        tx = _iso(t.get("transaction_date"))
        if visible is None or tx is None:
            continue
        if visible > as_of_d or tx < floor:
            continue
        trades.append({
            "politician": t.get("politician_canonical") or t.get("politician"),
            "party": t.get("party"),
            "chamber": t.get("chamber"),
            "state": t.get("state_district") or t.get("state"),
            "type": (t.get("transaction_type") or "").upper(),
            "owner": t.get("owner_code"),
            "transaction_date": tx.isoformat(),
            "filed_date": visible.isoformat(),
            "amount_min": t.get("amount_min"),
            "amount_max": t.get("amount_max"),
        })
    trades.sort(key=lambda r: r["transaction_date"], reverse=True)
    buys = [t for t in trades if t["type"].startswith("BUY") or t["type"] == "PURCHASE"]
    sells = [t for t in trades if t["type"].startswith("SELL") or t["type"] == "SALE"]
    result = {
        "symbol": symbol.upper(), "as_of": as_of_d.isoformat(), "window_days": days,
        "n": len(trades), "buys": len(buys), "sells": len(sells),
        "by_party": {
            p: {"buys": sum(1 for t in buys if t["party"] == p),
                "sells": sum(1 for t in sells if t["party"] == p)}
            for p in sorted({t["party"] for t in trades if t.get("party")})
        },
        "trades": trades[:12],
        "total_disclosed_all_time": len(data.get("trades") or []),
    }
    with _CACHE_LOCK:
        _CACHE[key] = result
    return result


def format_congress_block(symbol: str, c: dict | None) -> str:
    if not c:
        return ""
    head = (f"[{symbol}: congressional trades (STOCK Act filings visible "
            f"through {c['as_of']}, transactions in last {c['window_days']}d)]")
    if not c["n"]:
        return (head + f"\nNone disclosed in the window "
                f"({c['total_disclosed_all_time']} on record all-time). "
                f"Absence is uninformative for a small/mid cap.")
    party = ", ".join(f"{p}: {v['buys']} buy / {v['sells']} sell"
                      for p, v in c["by_party"].items())
    # The skew is computed here so the model reads a number, not a story:
    # 6 sells against 4 buys inside one party is balance, not a stance.
    total = c["buys"] + c["sells"]
    sell_share = c["sells"] / total if total else 0.5
    skew = ("sell-skewed" if sell_share >= 0.7 else
            "buy-skewed" if sell_share <= 0.3 else "balanced")
    party_skews = []
    for p, v in c["by_party"].items():
        n = v["buys"] + v["sells"]
        if n >= 4:
            share = v["sells"] / n
            party_skews.append(f"{p} {'sell-skewed' if share >= 0.7 else 'buy-skewed' if share <= 0.3 else 'balanced'}")
    lines = [head,
             f"{c['n']} trades: {c['buys']} buys, {c['sells']} sells"
             + (f" ({party})" if party else ""),
             f"Skew: {skew} overall"
             + (f"; by party: {', '.join(party_skews)}" if party_skews else
                "; no party has enough trades to read a stance")
             + ". Sells are the common side for members trimming winners; only a "
               "lopsided count is a signal, and never on its own."]
    for t in c["trades"][:6]:
        who = f"{t['politician']} ({t['party'] or '?'}-{t['state'] or '?'}, {t['chamber']})"
        owner = f", {t['owner'].lower()}" if t.get("owner") and t["owner"] != "SELF" else ""
        lines.append(f"- {t['transaction_date']} {t['type']} "
                     f"{_band(t['amount_min'], t['amount_max'])}: {who}{owner}; "
                     f"filed {t['filed_date']}")
    lines.append("Disclosures lag the trade by up to 45 days and amounts are "
                 "bands; read as positioning by informed-but-slow actors, "
                 "never as a timing signal.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Institutional holdings (13F)
# ---------------------------------------------------------------------------

def get_institutional_holdings(symbol: str, as_of: str) -> dict:
    as_of_d = date.fromisoformat(str(as_of)[:10])
    key = ("inst", symbol.upper(), as_of_d.isoformat())
    with _CACHE_LOCK:
        if key in _CACHE:
            return _CACHE[key]

    data = _av_get({"function": "INSTITUTIONAL_HOLDINGS",
                    "symbol": symbol.upper()})
    rows = []
    for h in data.get("holdings") or []:
        reported = _iso(h.get("last_reported"))
        if reported is None or reported > as_of_d:
            continue
        try:
            shares = int(float(h.get("shares_held") or 0))
            changed = int(float(h.get("shares_changed") or 0))
        except (TypeError, ValueError):
            continue
        rows.append({
            "holder": h.get("holder_name"), "shares": shares, "changed": changed,
            "pct": h.get("shares_changed_percentage"),
            "reported": reported.isoformat(),
        })
    rows.sort(key=lambda r: r["shares"], reverse=True)
    increased = sorted((r for r in rows if r["changed"] > 0),
                       key=lambda r: r["changed"], reverse=True)
    decreased = sorted((r for r in rows if r["changed"] < 0),
                       key=lambda r: r["changed"])

    def _num(k):
        try:
            return int(float(data.get(k) or 0))
        except (TypeError, ValueError):
            return None

    result = {
        "symbol": symbol.upper(), "as_of": as_of_d.isoformat(),
        "holders_visible": len(rows),
        "latest_report": max((r["reported"] for r in rows), default=None),
        "top_holders": rows[:INSTITUTIONAL_TOP_N],
        "largest_increases": increased[:INSTITUTIONAL_TOP_N],
        "largest_decreases": decreased[:INSTITUTIONAL_TOP_N],
        "visible_added": len(increased), "visible_cut": len(decreased),
        "snapshot": {
            "total_holders": _num("total_institutional_holders"),
            "increased": _num("holders_with_increased_holdings"),
            "decreased": _num("holders_with_decreased_holdings"),
            "ownership_pct": data.get("total_institutional_ownership_percentage"),
        },
    }
    with _CACHE_LOCK:
        _CACHE[key] = result
    return result


def _shares(n: int) -> str:
    n = abs(n)
    return f"{n / 1e6:.2f}M" if n >= 1e6 else f"{n / 1e3:.0f}K"


def format_institutional_block(symbol: str, h: dict | None) -> str:
    if not h or not h["holders_visible"]:
        return ""
    lines = [f"[{symbol}: institutional (13F) holder flows, filings reported "
             f"through {h['latest_report']} (visible as of {h['as_of']})]",
             f"{h['visible_added']} holders added vs {h['visible_cut']} cut "
             f"among {h['holders_visible']} visible filers."]
    if h["top_holders"]:
        lines.append("Largest holders: " + "; ".join(
            f"{r['holder']} {_shares(r['shares'])} ({r['pct']}, {r['reported']})"
            for r in h["top_holders"]))
    if h["largest_increases"]:
        lines.append("Biggest adds: " + "; ".join(
            f"{r['holder']} +{_shares(r['changed'])} ({r['pct']})"
            for r in h["largest_increases"]))
    if h["largest_decreases"]:
        lines.append("Biggest cuts: " + "; ".join(
            f"{r['holder']} -{_shares(r['changed'])} ({r['pct']})"
            for r in h["largest_decreases"]))
    lines.append("Quarterly filings, up to 45 days stale at report time: "
                 "ownership context, not a signal about the next session.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Composite
# ---------------------------------------------------------------------------

def political_blocks(symbol: str, as_of: str) -> tuple[list[str], list[str]]:
    """(blocks, problems). Each sub-block fails independently; a problem
    string names what could not be fetched so the caller can log a gap."""
    blocks: list[str] = []
    problems: list[str] = []
    try:
        blocks.append(format_congress_block(
            symbol, get_congress_trades(symbol, as_of)))
    except Exception as e:
        problems.append(f"congressional trades: {str(e)[:120]}")
    try:
        block = format_institutional_block(
            symbol, get_institutional_holdings(symbol, as_of))
        if block:
            blocks.append(block)
    except Exception as e:
        problems.append(f"institutional holdings: {str(e)[:120]}")
    return [b for b in blocks if b], problems

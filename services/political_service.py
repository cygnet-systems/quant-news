"""Political and institutional flow blocks from Alpha Vantage.

Two point-in-time-aware evidence blocks for the research prompt:

* Congressional trades. STOCK Act disclosures by House and Senate members,
  with party, chamber, owner (self/spouse/child) and the amount band, read
  out of ``services.av_store``: the endpoint returns the symbol's entire
  history on every call, so it is synced weekly and filtered locally rather
  than fetched per run. Point-in-time on ``filed_date``: a trade is visible
  only from the day it was filed, which is what a reader on ``as_of`` could
  have known. Disclosure lags the trade by up to 45 days, so this is
  positioning context, never a timing signal. When a run also carries the
  congressional dossier (``services.politician_dossier``) this half is left
  out: same rows, same window, and two renderings of one dataset in one
  prompt is how a report ends up quoting two different counts.
* Institutional holdings (``INSTITUTIONAL_HOLDINGS``). 13F holders adding
  vs cutting, with the largest movers. Holder rows carry the quarter-end
  they were reported for and are filtered to ``last_reported <= as_of``;
  the vendor's aggregate counts are a current snapshot and are labelled as
  such rather than presented as point-in-time.

Both endpoints were verified against the project key on 2026-09-01. The 13F
half is one Alpha Vantage call per symbol per run and shares the news
bucket; the congressional half spends one at most weekly, through the store.
"""

import logging
import threading
from datetime import date

from services import av_store
# AlphaVantageUnavailable is re-exported: the two blocks here raise it and
# political_blocks' callers catch it by this module's name.
from services.alpha_vantage import AlphaVantageUnavailable, fetch  # noqa: F401

logger = logging.getLogger(__name__)

_CACHE: dict[tuple, dict | None] = {}
_CACHE_LOCK = threading.Lock()

CONGRESS_WINDOW_DAYS = 180
INSTITUTIONAL_TOP_N = 4


def _av_get(params: dict) -> dict:
    """The shared client, behind the name the callers below already use.

    Both blocks let AlphaVantageUnavailable out: political evidence is
    OPTIONAL, and ``political_blocks`` turns the failure into a named gap
    rather than into a silent "no trades disclosed".
    """
    params = dict(params)
    return fetch(params.pop("function"), **params)


def _iso(d: str | None) -> date | None:
    try:
        return date.fromisoformat(str(d)[:10]) if d else None
    except ValueError:
        return None


def amount_band(lo, hi) -> str:
    """The STOCK Act amount band as text. Public because the congressional
    dossier renders the same bands and the two must read identically."""
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
    date falls in the last ``days``.

    Read from the store, which the top-up refreshes at most weekly: the
    endpoint hands back the symbol's whole history whatever dates are asked
    for, so a per-run fetch bought nothing and spent a call. Raises
    AlphaVantageUnavailable only when the top-up failed AND nothing is
    stored, so the caller records a gap rather than reporting "none
    disclosed" about a symbol nobody has ever fetched.
    """
    symbol = symbol.upper()
    as_of_d = date.fromisoformat(str(as_of)[:10])
    key = ("congress", symbol, as_of_d.isoformat(), days)
    with _CACHE_LOCK:
        if key in _CACHE:
            return _CACHE[key]

    freshness = av_store.ensure_fresh(av_store.CONGRESS_FUNCTION, symbol)
    trades = av_store.congress_trades_for(symbol, as_of_d, days=days)
    on_record = av_store.congress_trades_count(symbol, as_of_d)
    if not on_record and freshness["reason"].startswith("unavailable"):
        raise AlphaVantageUnavailable(freshness["reason"])
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
        "total_disclosed_visible": on_record,
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
                f"({c['total_disclosed_visible']} on record and public by "
                f"{c['as_of']}). "
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
                     f"{amount_band(t['amount_min'], t['amount_max'])}: {who}{owner}; "
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

def political_blocks(symbol: str, as_of: str,
                     include_congress: bool = True
                     ) -> tuple[list[str], list[str]]:
    """(blocks, problems). Each sub-block fails independently; a problem
    string names what could not be fetched so the caller can log a gap.

    ``include_congress`` is False when the run also carries the congressional
    dossier: it reads the same stored rows over the same window and names the
    members, so this block would only repeat it.

    A non-empty ``problems`` alongside a non-empty ``blocks`` means one
    sub-block failed and the other was written, not that the evidence is
    missing: the caveat is appended to the rendered text, where the model
    reading the rows sees it, and the caller records the block as present.
    Recording it as a gap instead put a rendered block in the prompt and a
    "this was not available" line about the same block beside it.
    """
    blocks: list[str] = []
    problems: list[str] = []
    if include_congress:
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
    blocks = [b for b in blocks if b]
    if blocks and problems:
        blocks[-1] += ("\nNot every source behind this section refreshed on "
                       f"this run ({'; '.join(problems)}); what is above is "
                       f"what could be read.")
    return blocks, problems

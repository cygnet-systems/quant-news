"""Who inside the company has been buying and selling, as a prompt block.

Form 4 lines come from the local store (``services.av_store``), never from a
per-run fetch: the vendor returns the symbol's entire history on every call
(NVDA: 6,920 rows to 2003), so a run tops the symbol up at most once a week
and filters locally.

Two things this block refuses to do, because both were easy to get wrong:

* It never turns a zero share price into a dollar figure. Grants, awards,
  gifts and option exercises come through the vendor priced at 0.0, and a
  reader (or a model) shown "acquired 50,000 shares for $0" concludes the
  executive bought at no cost. Those rows are counted, named, and excluded
  from every dollar total, and the block says so.
* It never calls an acquisition a purchase. ``acquisition_or_disposal`` is
  A or D; an A row can be a vesting award as easily as an open-market buy,
  and only the priced rows can be read as the latter.

The visibility rule is a PROXY and is stated in the block itself: a row is
served from the second trading day after the transaction, the SEC filing
deadline. Real filings often land sooner, so the dates here are the latest a
reader could have seen the trade, not when the tape learned about it.
"""

import logging

from services import av_store

logger = logging.getLogger(__name__)

WINDOW_DAYS = av_store.INSIDER_WINDOW_DAYS
MAX_EXECUTIVES = 6
MAX_LINES = 6

ACQUIRED = "A"
DISPOSED = "D"


def _shares(n: float | None) -> str:
    if not n:
        return "0 sh"
    n = abs(float(n))
    if n >= 1e6:
        return f"{n / 1e6:.2f}M sh"
    if n >= 1e3:
        return f"{n:,.0f} sh"
    return f"{n:,.2f} sh".replace(".00 sh", " sh")


def _usd(v: float | None) -> str:
    if not v:
        return "$0"
    v = abs(float(v))
    if v >= 1e9:
        return f"${v / 1e9:.2f}B"
    if v >= 1e6:
        return f"${v / 1e6:.2f}M"
    if v >= 1e3:
        return f"${v / 1e3:.0f}K"
    return f"${v:,.0f}"


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def _person(name: str, title: str | None) -> str:
    return f"{name} ({title})" if title else name


def summarize_insiders(symbol: str, as_of, days: int = WINDOW_DAYS) -> dict:
    """Aggregate the visible Form 4 lines for ``symbol`` as of ``as_of``.

    Reads through ``av_store.insider_transactions_for``, which is the only
    point-in-time door; ``as_of`` is required there and stays required here.
    """
    rows = av_store.insider_transactions_for(symbol, as_of, days=days)
    people: dict[str, dict] = {}
    # Bounded by as_of by construction: the reader serves nothing whose
    # visibility date is later. It is what the header dates the block with
    # when the wall-clock sync stamp would sit after the run's own cutoff.
    latest_visible = max((r["visible_from"] for r in rows), default=None)
    totals = {"acquired": 0, "disposed": 0, "other": 0, "priced": 0,
              "unpriced": 0, "value_acquired": 0.0, "value_disposed": 0.0,
              "shares_acquired": 0.0, "shares_disposed": 0.0}

    for r in rows:
        name = r["executive"] or "undisclosed filer"
        p = people.setdefault(name, {
            "executive": name, "title": r["title"], "acquired": 0,
            "disposed": 0, "shares_acquired": 0.0, "shares_disposed": 0.0,
            "value_acquired": 0.0, "value_disposed": 0.0, "unpriced": 0,
            "latest": r["transaction_date"],
        })
        if p["title"] is None:
            p["title"] = r["title"]
        side = (r["side"] or "").upper()
        shares = float(r["shares"] or 0.0)
        value = r["value_usd"]
        # value_usd is NULL exactly when the vendor priced the row at 0 (or
        # gave no price at all). Those rows move shares but no money we can
        # name, so they are counted separately rather than summed as zero.
        if value is None:
            p["unpriced"] += 1
            totals["unpriced"] += 1
        else:
            totals["priced"] += 1
        if side == ACQUIRED:
            p["acquired"] += 1
            p["shares_acquired"] += shares
            totals["acquired"] += 1
            totals["shares_acquired"] += shares
            if value is not None:
                p["value_acquired"] += float(value)
                totals["value_acquired"] += float(value)
        elif side == DISPOSED:
            p["disposed"] += 1
            p["shares_disposed"] += shares
            totals["disposed"] += 1
            totals["shares_disposed"] += shares
            if value is not None:
                p["value_disposed"] += float(value)
                totals["value_disposed"] += float(value)
        else:
            totals["other"] += 1

    ranked = sorted(
        people.values(),
        key=lambda p: (p["value_acquired"] + p["value_disposed"],
                       p["shares_acquired"] + p["shares_disposed"]),
        reverse=True)
    return {
        "symbol": (symbol or "").upper(),
        "as_of": str(as_of)[:10],
        "window_days": days,
        "n": len(rows),
        "executives": len(people),
        "latest_visible": latest_visible,
        "by_executive": ranked,
        "recent": rows[:MAX_LINES],
        **totals,
    }


def format_insider_block(symbol: str, s: dict | None,
                         synced: str | None = None) -> str:
    """The prompt block. Empty only when ``s`` is missing entirely: a window
    with no filings is stated, because "no insider sold" is itself evidence
    and a silently absent block is not."""
    if not s:
        return ""
    symbol = (symbol or s.get("symbol") or "").upper()
    # The sync stamp is wall clock, and every other date in this block is
    # bounded by as_of. On a backtest it would put a date AFTER the run's
    # cutoff into the research prompt, on the path this platform measures
    # alpha with, so past that point the block dates itself by the newest
    # row it is actually serving.
    stamp = str(synced)[:10] if synced else ""
    if stamp and stamp <= s["as_of"]:
        dated = f", store synced {stamp}"
    elif s.get("latest_visible"):
        dated = f", newest row visible from {s['latest_visible']}"
    else:
        dated = ""
    head = (f"[{symbol}: insider transactions (SEC Form 4), transactions in "
            f"the {s['window_days']}d to {s['as_of']}{dated}]")
    lag = ("Visibility is a PROXY: a row enters this block on the second "
           "trading day after the transaction, the SEC Form 4 filing "
           "deadline. Most filings land sooner, so these dates are the "
           "latest a reader could have seen the trade, not the moment the "
           "market learned it. Do not read the spacing between transaction "
           "and visibility as a timing signal.")
    if not s["n"]:
        return "\n".join([
            head,
            "No Form 4 lines were visible in this window. For most boards "
            "that is normal: insiders transact around vesting and window "
            "openings, so an empty stretch is not a stance.",
            lag])

    lines = [head,
             f"{_plural(s['n'], 'filing')} by "
             f"{_plural(s['executives'], 'named executive')}: "
             f"{_plural(s['acquired'], 'acquisition')}, "
             f"{_plural(s['disposed'], 'disposal')}"
             + (f", {s['other']} rows with no direction code" if s["other"]
                else "") + "."]
    # Shares and dollars are stated as two separate counts on purpose: the
    # share totals cover every row, the dollar totals only the priced ones,
    # and folding them into one sentence implies the money moved with all
    # of the shares.
    lines.append(f"Shares moved, all {s['n']} rows: "
                 f"{_shares(s['shares_acquired'])} acquired vs "
                 f"{_shares(s['shares_disposed'])} disposed.")
    if s["priced"]:
        lines.append(
            f"Dollar value, over the {s['priced']} row(s) carrying a real "
            f"share price: {_usd(s['value_acquired'])} acquired, "
            f"{_usd(s['value_disposed'])} disposed.")
    else:
        lines.append("No row in this window carried a share price above "
                     "zero, so no dollar value can be stated for any of them.")
    if s["unpriced"]:
        lines.append(
            f"{s['unpriced']} of {s['n']} rows report a share price of 0.00. "
            f"Those are grants, awards, gifts or option exercises, not "
            f"open-market purchases at no cost; they are excluded from the "
            f"dollar totals above and no price is invented for them.")

    lines.append("By executive:")
    for p in s["by_executive"][:MAX_EXECUTIVES]:
        parts = []
        if p["acquired"]:
            parts.append(f"{_plural(p['acquired'], 'acquisition')} "
                         f"{_shares(p['shares_acquired'])}"
                         + (f" / {_usd(p['value_acquired'])}"
                            if p["value_acquired"] else ""))
        if p["disposed"]:
            parts.append(f"{_plural(p['disposed'], 'disposal')} "
                         f"{_shares(p['shares_disposed'])}"
                         + (f" / {_usd(p['value_disposed'])}"
                            if p["value_disposed"] else ""))
        tail = (f"; {p['unpriced']} at price 0.00 (no cash figure)"
                if p["unpriced"] else "")
        lines.append(f"- {_person(p['executive'], p['title'])}: "
                     + ("; ".join(parts) or "no directional rows")
                     + f"{tail}; latest {p['latest']}")
    hidden = max(0, s["executives"] - MAX_EXECUTIVES)
    if hidden:
        lines.append(f"({hidden} further executive(s) with smaller activity "
                     f"are not listed.)")

    lines.append("Most recent lines:")
    for r in s["recent"]:
        side = {ACQUIRED: "acquired", DISPOSED: "disposed"}.get(
            (r["side"] or "").upper(), "undirected")
        if r["value_usd"] is not None:
            money = (f" at ${float(r['share_price']):,.2f} = "
                     f"{_usd(r['value_usd'])}")
        else:
            money = (" with the share price reported as 0.00, so no dollar "
                     "value applies (grant, gift or option exercise)")
        lines.append(f"- {r['transaction_date']} {side} "
                     f"{_shares(r['shares'])}{money}: "
                     f"{_person(r['executive'] or 'undisclosed filer', r['title'])}"
                     f"; visible from {r['visible_from']}")
    lines.append(lag)
    return "\n".join(lines)


def insider_block(symbol: str, as_of, days: int = WINDOW_DAYS
                  ) -> tuple[str, list[str]]:
    """(block, problems) for the research prompt.

    ``ensure_fresh`` decides whether this run spends a call at all; a vendor
    failure is only a gap when it leaves nothing to read. With stored rows
    from a previous week the block is still written and the run is not
    marked degraded for evidence it does have.

    The top-up is inside the try with everything else. It reports vendor
    failures rather than raising them, but a database error underneath it
    would otherwise take out a block the stored rows could still write.
    """
    symbol = (symbol or "").strip().upper()
    problems: list[str] = []
    try:
        freshness = av_store.ensure_fresh(av_store.INSIDER_FUNCTION, symbol)
        summary = summarize_insiders(symbol, as_of, days=days)
        # The last SUCCESSFUL sync, not the last attempt: the top-up that
        # just failed writes a fetch stamp of its own, and a header reading
        # that would say "synced today" above rows from last week. The
        # formatter drops it when it falls after as_of.
        stamp = av_store.last_synced_date(av_store.INSIDER_FUNCTION, symbol)
    except Exception as e:
        logger.warning("%s: insider block failed: %s", symbol, e)
        return "", [f"insider transactions: {str(e)[:120]}"]

    stale = freshness["reason"].startswith("unavailable")
    if stale and not summary["n"]:
        return "", [f"insider transactions: {freshness['reason']}"]

    block = format_insider_block(symbol, summary, synced=stamp)
    if stale:
        block += ("\nThe weekly top-up for this symbol failed on this run "
                  f"({freshness['reason']}); the rows above are as stored.")
    return block, problems

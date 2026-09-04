"""Who in office is trading this name, and who in this run's news is one.

The congressional block that shipped before this counted buys and sells and
named nobody in particular. A count is not evidence: "9 buys, 13 sells" tells
a reader nothing they can check, follow, or weigh. This block answers the
questions a count leaves open. Who is the member. Are they sitting, and on
what. What did they trade in THIS symbol. What else have they been buying and
selling lately, which is the start of telling a sector view from a one-off --
bounded by the symbols this system syncs, so it is a floor on their activity
and the block says so rather than implying a member traded nothing else.

It also reads the run's own news. A member of Congress who appears in an
article about this company is context the price chart cannot give, whether or
not they have ever traded it: the roster is matched against article titles and
summaries, longest alias first so "nancy pelosi" wins over any shorter variant
overlapping the same span, and an alias owned by two members is never matched
at all.

Rules the formatter holds to:

* A name is not an identification. Steve Cohen sits for TN-9 and also runs
  Point72, so a match is kept only when the article puts a title next to the
  name or the member disclosed a trade in this symbol. Everything else is
  dropped rather than asserted.
* Three members, no more. The cap is stated in the block and what it dropped
  is named and logged, because a truncated list that looks complete is the
  same failure as a missing evidence block. The over-cap list is bounded and
  written AFTER the caveats: the caller slices this block to a fixed width,
  and the caveats are the lines that have to survive that.
* Tenure is the CURRENT unbroken stint, in the current chamber. A first-ever
  term start is not a length of service: a member with a gap would otherwise
  be credited with the years they were out of office, in a chamber they had
  not yet been elected to.
* A member whose term had not begun on ``as_of`` is not described as sitting.
  The roster carries future terms, and "Senator X bought this" about somebody
  who was not yet sworn in is fabrication.
* Reading only through ``av_store``'s point-in-time helpers, so nothing here
  can surface a filing before it was public.
"""

import bisect
import logging
import re
from datetime import date

from services import av_store
from services.political_service import amount_band

logger = logging.getLogger(__name__)

WINDOW_DAYS = av_store.CONGRESS_WINDOW_DAYS
MAX_POLITICIANS = 3
MAX_OTHER_TRADES = 5
MAX_OWN_TRADES = 4
# The over-cap list grows with the number of members trading the name
# (NVDA: nine over the cap in a normal 180-day window) and the caller
# slices the whole block to a fixed size, so an unbounded list here is
# paid for by the caveats below it.
MAX_DROPPED_NAMED = 6
# One separator per article so a name cannot be matched across the join.
_JOIN = "\n||\n"
_PUNCT = re.compile(r"[^a-z0-9 '\-]")
_SPACE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    """Article text in the same shape ``av_store.alias_key`` puts a name.

    Punctuation becomes a space rather than being deleted, because in prose
    it separates words ("..., Pelosi said") where in a name field it merely
    decorates one.
    """
    return _SPACE.sub(" ", _PUNCT.sub(" ", (text or "").lower())).strip()


def _article_text(article) -> tuple[str, str, str]:
    """(title, summary, published date) from either shape the news layer
    hands around: the dataclass from news_service or a plain dict."""
    if hasattr(article, "title"):
        return (article.title or "",
                getattr(article, "summary", "") or "",
                str(getattr(article, "published_at", "") or "")[:10])
    return (article.get("title") or "",
            article.get("summary") or article.get("description") or "",
            str(article.get("published_at")
                or article.get("published_date") or "")[:10])


# Words that make a name in prose a member of Congress rather than a
# namesake. Kept deliberately short: every entry here is a word a newsroom
# uses as a title, not one that merely appears near politics.
_TITLE_WORDS = frozenset((
    "rep", "reps", "representative", "representatives", "congressman",
    "congresswoman", "congressmen", "congresswomen", "sen", "sens",
    "senator", "senators", "lawmaker", "lawmakers", "hon",
))
# "Rep. Dan Newhouse" and "Dan Newhouse, the Washington representative"
# both identify him; 48 characters covers either side without reaching the
# next sentence.
_TITLE_WINDOW = 48


def _titled(blob: str, article: tuple[int, int], span: tuple[int, int]
            ) -> str | None:
    """The title word next to this name, or None.

    Clipped to the article the name was found in, so a title in the previous
    headline cannot vouch for a name in the next one.
    """
    lo = max(article[0], span[0] - _TITLE_WINDOW)
    hi = min(article[1], span[1] + _TITLE_WINDOW)
    for token in f"{blob[lo:span[0]]} {blob[span[1]:hi]}".split():
        if token in _TITLE_WORDS:
            return token
    return None


def politicians_in_news(news: list | None,
                        index: dict[str, str] | None = None,
                        known: set[str] | None = None) -> list[dict]:
    """Members named in the run's own articles.

    Longest alias first: a shorter alias belonging to a different member can
    otherwise claim part of a longer name. Once a span of text is claimed it
    is not matched again, so "nancy pelosi" cannot also be read as some other
    member's "pelosi".

    A name is not an identification. Steve Cohen is a sitting House member
    and also the manager of Point72, so "Smart money is buying: Steve Cohen
    initiated a position" matches the roster and means somebody else
    entirely. A match is therefore kept only when the article corroborates
    it: a title word next to the name, or the member being in ``known`` --
    the set of members who disclosed a trade in this symbol, whose identity
    the filing already established.
    """
    articles = list(news or [])
    if not articles:
        return []
    index = av_store.alias_index() if index is None else index
    if not index:
        return []
    known = known or set()

    blob_parts, bounds, meta = [], [], []
    cursor = 0
    for a in articles:
        title, summary, published = _article_text(a)
        text = _normalize(f"{title} {summary}")
        bounds.append((cursor, cursor + len(text)))
        meta.append((title.strip(), published))
        blob_parts.append(text)
        cursor += len(text) + len(_JOIN)
    blob = _JOIN.join(blob_parts)
    starts = [b[0] for b in bounds]

    claimed: list[tuple[int, int]] = []
    hits: dict[str, dict] = {}
    for alias in sorted(index, key=len, reverse=True):
        # Cheap containment test first: the roster is ~1,100 members and a
        # regex per alias per article is the difference between a block that
        # costs nothing and one that costs a second per symbol.
        if alias not in blob:
            continue
        pattern = re.compile(rf"\b{re.escape(alias)}\b")
        for match in pattern.finditer(blob):
            span = match.span()
            if any(s < span[1] and span[0] < e for s, e in claimed):
                continue
            claimed.append(span)
            which = max(0, bisect.bisect_right(starts, span[0]) - 1)
            titled = _titled(blob, bounds[which], span)
            if not titled and index[alias] not in known:
                logger.debug("dossier: %r in an article with no title word "
                             "and no disclosed trade; not attributed", alias)
                continue
            headline, published = meta[which]
            hit = hits.setdefault(index[alias], {
                "bioguide_id": index[alias], "alias": alias,
                "headline": headline, "published": published,
                "mentions": 0, "titled": titled,
            })
            hit["mentions"] += 1
            # A titled mention is the stronger identification, so it takes
            # over the headline the block quotes.
            if titled and not hit["titled"]:
                hit.update(titled=titled, alias=alias, headline=headline,
                           published=published)
    return list(hits.values())


def _iso(value) -> date | None:
    try:
        return date.fromisoformat(str(value)[:10]) if value else None
    except (TypeError, ValueError):
        return None


# Consecutive terms in the roster do not quite meet: a term ends 2013-01-03
# and the next starts 2013-01-03, 2013-01-04 or 2013-01-06. Measured over
# the live roster every such seam is 18 days or less and every real break in
# service is 406 days or more, so a month of slack separates the two without
# ambiguity.
_SEAM_SLACK_DAYS = 31


def _stint_start(parsed: list, index: int, chamber: str | None = None) -> date:
    """The start of the unbroken run of service ending at ``parsed[index]``.

    Walk back while each earlier term ends within the seam slack of the next
    one's start; stop at the first real gap. With ``chamber`` given, also
    stop when the earlier term was served in a different one, so a member who
    moved from the House to the Senate is not described as having been a
    senator since they were a representative.
    """
    i = index
    while i > 0:
        previous = parsed[i - 1]
        if previous[1] is None:
            break
        if (parsed[i][0] - previous[1]).days > _SEAM_SLACK_DAYS:
            break
        if chamber is not None and previous[2] != chamber:
            break
        i -= 1
    return parsed[i][0]


def describe_tenure(profile: dict | None, as_of) -> str:
    """Chamber and service, as of the report date and no later.

    The roster carries terms that had not started yet. Describing a
    member-elect as sitting, or a member who left in January as current, is
    the kind of confident wrongness this whole phase exists to remove. So is
    reading a first-ever term start as continuous service: Maria Cantwell's
    earliest roster term is a 1993 House seat she left in 1995, and "in the
    Senate since 1993" is wrong twice over.
    """
    as_of_d = _iso(as_of)
    terms = (profile or {}).get("terms") or []
    parsed = []
    for t in terms:
        if not isinstance(t, dict):
            continue
        start = _iso(t.get("start_date"))
        if start is None:
            continue
        parsed.append((start, _iso(t.get("end_date")),
                       (t.get("chamber") or "").strip()))
    if not parsed or as_of_d is None:
        chamber = (profile or {}).get("chamber")
        return (f"{chamber} member; term dates unavailable" if chamber
                else "roster metadata unavailable")

    parsed.sort()
    begun = [i for i, t in enumerate(parsed) if t[0] <= as_of_d]
    if not begun:
        first = parsed[0]
        return (f"not yet in office as of {as_of_d.isoformat()}: earliest "
                f"term begins {first[0].isoformat()}"
                + (f" ({first[2]})" if first[2] else ""))
    current = [i for i in begun
               if parsed[i][1] is None or parsed[i][1] >= as_of_d]
    if current:
        latest = current[-1]
        chamber = parsed[latest][2] or (profile or {}).get("chamber") \
            or "Congress"
        seated = _stint_start(parsed, latest, chamber=parsed[latest][2] or None)
        in_congress = _stint_start(parsed, latest)
        years = (as_of_d - seated).days // 365
        sentence = (f"sitting {chamber} member, in office since "
                    f"{seated.isoformat()} ({years} yr)")
        if in_congress != seated:
            # Continuous service that began in the other chamber: the seat
            # is the claim being made, so the longer run is the footnote.
            sentence += (f", in Congress without a break since "
                         f"{in_congress.isoformat()}")
        return sentence
    last = begun[-1]
    left = parsed[last][1]
    served = _stint_start(parsed, last)
    sentence = (f"former member: served {served.isoformat()} to "
                f"{left.isoformat()}, out of office on {as_of_d.isoformat()}")
    if parsed[0][0] < served:
        # Bass sat 1995-2007 and again 2011-2013. One span across both is a
        # stint he never served.
        sentence += (f"; earlier, non-consecutive service from "
                     f"{parsed[0][0].isoformat()} is not in that span")
    return sentence


def _identity(profile: dict | None, trades: list[dict]) -> str:
    """Name, party, chamber and seat. The roster is authoritative; a filing
    row is the fallback, so a member who traded the name is still identified
    when the roster sync is behind."""
    if profile:
        seat = "-".join(p for p in (profile.get("state"),
                                    profile.get("district")) if p)
        bits = [b for b in (profile.get("party"), profile.get("chamber"),
                            seat) if b]
        return f"{profile.get('display_name') or profile.get('bioguide_id')}" \
               + (f" ({', '.join(bits)})" if bits else "")
    if trades:
        t = trades[0]
        bits = [b for b in (t.get("party"), t.get("chamber"),
                            t.get("state")) if b]
        return f"{t.get('politician') or 'unnamed filer'}" \
               + (f" ({', '.join(bits)})" if bits else "")
    return "unnamed member"


def _trade_line(t: dict, with_symbol: bool = False) -> str:
    owner = (t.get("owner") or "").upper()
    tail = f", {owner.lower()} account" if owner and owner != "SELF" else ""
    return (f"{t['transaction_date']} {t['type']}"
            + (f" {t['symbol']}" if with_symbol else "")
            + f" {amount_band(t.get('amount_min'), t.get('amount_max'))}"
            f"{tail}, public from {t.get('filed_date')}")


def build_dossier(symbol: str, as_of, news: list | None = None,
                  days: int = WINDOW_DAYS,
                  max_politicians: int = MAX_POLITICIANS,
                  max_other: int = MAX_OTHER_TRADES) -> dict:
    """The three most relevant members, with what they traded and what else.

    Relevance order: named in this run's news AND disclosed a trade in the
    symbol, then members who traded the symbol (most recent first), then
    members named only in the news. That order is deliberate: a member who
    both appears in the story and holds the stock is the only combination
    that is more than a coincidence.
    """
    symbol = (symbol or "").strip().upper()
    trades = av_store.congress_trades_for(symbol, as_of, days=days)

    by_member: dict[str, list[dict]] = {}
    for t in trades:
        if t.get("bioguide_id"):
            by_member.setdefault(t["bioguide_id"], []).append(t)
    # Members who filed a trade in this name are matchable on the name
    # alone; for everybody else the article has to say they hold an office.
    mentions = {m["bioguide_id"]: m
                for m in politicians_in_news(news, known=set(by_member))}

    candidates = []
    for bioguide in set(by_member) | set(mentions):
        own = by_member.get(bioguide, [])
        latest = max((t["transaction_date"] for t in own if
                      t.get("transaction_date")), default="")
        named = bioguide in mentions
        rank = 0 if (named and own) else (1 if own else 2)
        candidates.append({
            "bioguide_id": bioguide, "rank": rank, "own": own,
            "latest": latest, "named": named,
            "mention": mentions.get(bioguide),
        })
    # Three stable passes rather than one key with negated dates: rank
    # decides, recency breaks it, and the id makes the order the same on
    # every run instead of whatever the set iterated.
    candidates.sort(key=lambda c: c["bioguide_id"])
    candidates.sort(key=lambda c: (c["latest"], len(c["own"])), reverse=True)
    candidates.sort(key=lambda c: c["rank"])

    kept, dropped = candidates[:max_politicians], candidates[max_politicians:]
    entries = []
    for c in kept:
        profile = av_store.politician(c["bioguide_id"])
        # The member's whole recent tape, then this symbol filtered out of
        # it: the "what else" is the part a symbol-scoped block cannot show.
        recent = av_store.congress_trades_by_politician(
            c["bioguide_id"], as_of, limit=(max_other + len(c["own"])) * 3)
        elsewhere = [t for t in recent if (t.get("symbol") or "") != symbol]
        entries.append({
            "bioguide_id": c["bioguide_id"],
            "identity": _identity(profile, c["own"]),
            "tenure": describe_tenure(profile, as_of),
            "own": c["own"][:MAX_OWN_TRADES],
            "own_total": len(c["own"]),
            "buys": sum(1 for t in c["own"] if t["type"].startswith("BUY")
                        or t["type"] == "PURCHASE"),
            "sells": sum(1 for t in c["own"] if t["type"].startswith("SELL")
                         or t["type"] == "SALE"),
            "elsewhere": elsewhere[:max_other],
            "elsewhere_hidden": max(0, len(elsewhere) - max_other),
            "mention": c["mention"],
        })

    dropped_named = []
    for c in dropped:
        profile = av_store.politician(c["bioguide_id"])
        who = _identity(profile, c["own"]).split(" (")[0]
        why = (f"{len(c['own'])} trade(s) in {symbol}" if c["own"]
               else "named in the news, no disclosed trade here")
        dropped_named.append(f"{who} ({why})")
    if dropped_named:
        logger.info("dossier %s: %d member(s) over the cap of %d not shown: %s",
                    symbol, len(dropped_named), max_politicians,
                    "; ".join(dropped_named))

    return {
        "symbol": symbol, "as_of": str(as_of)[:10], "window_days": days,
        "entries": entries, "dropped": dropped_named,
        "trades_in_window": len(trades),
        "members_named_in_news": len(mentions),
        "news_scanned": len(news or []),
    }


def format_dossier_block(d: dict | None) -> str:
    if not d:
        return ""
    symbol = d["symbol"]
    head = (f"[{symbol}: congressional dossier. Members with disclosed trades "
            f"in this name (transactions in the last {d['window_days']}d, "
            f"disclosures public on or before {d['as_of']}) and members "
            f"named in this run's news]")
    footer = ("Disclosures lag the trade by up to 45 days and amounts are "
              "bands, never exact. The cross-symbol view is bounded by the "
              "symbols this system syncs, so treat a member's other trades "
              "as a floor on their activity and never a full account of it. "
              "A member appearing in this run's news is not evidence they "
              "traded on it, and a disclosed trade is not evidence of "
              "foreknowledge; both are context about who is paying attention "
              "to this name.")
    if not d["entries"]:
        return "\n".join([
            head,
            f"No member of Congress disclosed a trade in {symbol} in the "
            f"window, and none of the {d['news_scanned']} article(s) in this "
            f"run named a member of the roster.",
            footer])

    lines = [head,
             f"{d['trades_in_window']} disclosed trade(s) in {symbol} in the "
             f"window; {d['members_named_in_news']} roster member(s) named "
             f"across {d['news_scanned']} article(s) in this run. Showing "
             f"{len(d['entries'])}, ranked by news mention first, then by "
             f"most recent trade in {symbol}."]
    for e in d["entries"]:
        lines.append(f"* {e['identity']}. {e['tenure']}.")
        if e["own_total"]:
            lines.append(f"  In {symbol}: {e['own_total']} disclosed trade(s) "
                         f"in the window, {e['buys']} buy / {e['sells']} sell."
                         + "".join(f"\n    - {_trade_line(t)}"
                                   for t in e["own"]))
            if e["own_total"] > len(e["own"]):
                lines.append(f"    ({e['own_total'] - len(e['own'])} further "
                             f"trade(s) in {symbol} not listed.)")
        else:
            lines.append(f"  In {symbol}: no disclosed trade in the window. "
                         f"They appear here because this run's news names "
                         f"them, not because they hold it.")
        if e["mention"]:
            headline = e["mention"].get("headline") or "(untitled article)"
            # Without a title word next to the name, the only reason this is
            # the member and not a namesake is their own filing. Say which.
            how = (f"called \"{e['mention']['titled']}\" there"
                   if e["mention"].get("titled")
                   else "with no title next to the name, matched to them "
                        "because they filed a trade in it")
            lines.append(f"  Named in this run's news as \"{e['mention']['alias']}\""
                         f" ({how}, {e['mention']['mentions']} mention(s)), "
                         f"first in: "
                         f"{e['mention'].get('published') or 'undated'} "
                         f"{headline}")
        if e["elsewhere"]:
            lines.append("  Also disclosed lately, among the other symbols "
                         "this system syncs: "
                         + "; ".join(_trade_line(t, with_symbol=True)
                                     for t in e["elsewhere"])
                         + (f". {e['elsewhere_hidden']} further trade(s) in "
                            f"the fetched range not shown."
                            if e["elsewhere_hidden"] else "."))
        else:
            # Never "they traded nothing else": the store only holds the
            # symbols this system syncs, so an empty cross-symbol view is a
            # statement about coverage and reads as one.
            lines.append("  No other trade by this member is in the store. "
                         "The store holds only the symbols this system syncs, "
                         "so that is the limit of what was looked at, not "
                         "evidence they traded nothing else.")
    # Footer first, over-cap names last: the caller truncates this block to a
    # fixed width, and the caveat is the line that must survive it.
    lines.append(footer)
    if d["dropped"]:
        shown = d["dropped"][:MAX_DROPPED_NAMED]
        rest = len(d["dropped"]) - len(shown)
        lines.append(f"{len(d['dropped'])} further member(s) over the cap of "
                     f"{MAX_POLITICIANS} are not shown: "
                     + "; ".join(shown)
                     + (f"; and {rest} more" if rest else "") + ".")
    return "\n".join(lines)


def politician_block(symbol: str, as_of, news: list | None = None,
                     days: int = WINDOW_DAYS) -> tuple[str, list[str]]:
    """(block, problems) for the research prompt.

    Two stored sources back this block and each is topped up on its own
    schedule: the symbol's congressional trades weekly, the roster monthly.
    A vendor failure is a gap only where it leaves nothing to read; with
    stored rows the block is still written rather than the run marked
    degraded for evidence it has.
    """
    symbol = (symbol or "").strip().upper()
    problems: list[str] = []
    try:
        # Both top-ups are inside the try. They report vendor failures rather
        # than raising them, but a rate-limiter timeout or a database error
        # underneath would otherwise lose a block the stored rows can write.
        trades_fresh = av_store.ensure_fresh(av_store.CONGRESS_FUNCTION,
                                             symbol)
        roster_fresh = av_store.ensure_fresh(av_store.ROSTER_FUNCTION,
                                             av_store.ALL_SUBJECTS)
        dossier = build_dossier(symbol, as_of, news=news, days=days)
    except Exception as e:
        logger.warning("%s: congressional dossier failed: %s", symbol, e)
        return "", [f"congressional dossier: {str(e)[:120]}"]

    if (trades_fresh["reason"].startswith("unavailable")
            and not dossier["trades_in_window"]):
        problems.append(f"congressional trades: {trades_fresh['reason']}")
    if (roster_fresh["reason"].startswith("unavailable")
            and not dossier["members_named_in_news"]):
        problems.append(f"roster: {roster_fresh['reason']}")
    if problems and not dossier["entries"]:
        return "", problems
    return format_dossier_block(dossier), problems

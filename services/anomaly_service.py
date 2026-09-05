"""What is unusual about this symbol today, so the report can be about that.

Every report the platform writes renders the same sections in the same order
whether or not there is anything in them, which is why they all read alike: a
recital of price action and indicator levels describes what already happened
and commits to nothing about what happens next. This module is the other half
of that fix. It scores the evidence a run has already gathered and returns the
handful of things that actually stand out for THIS symbol — a cluster of
insiders selling, a chain positioned three to one against the tape, a member
of Congress buying into a decline — so the stage after it can research those
specific questions and give each one its own section. A quiet symbol produces
an empty list and a short report; that is the intended outcome, not a failure.

Two rules hold everywhere below.

* Nothing here fetches. ``detect`` takes the blocks the run computed and does
  arithmetic on them. No vendor call, no LLM call, no database read, so it is
  safe to call on the poll path and costs nothing to call twice.
* A detector with missing inputs returns nothing. It never emits a
  neutral-looking anomaly to fill a slot, because a section headed "options
  positioning is balanced" written from a chain that was never fetched is the
  BHF failure again in a new place.

Severity is a 0-1 ordering key, not a probability. It exists to decide which
questions are worth a web search when more of them clear their thresholds than
the cap allows.
"""

import logging
import math
from datetime import date, timedelta

from services.options_service import MIN_LIQUID_VOLUME

logger = logging.getLogger(__name__)

# Each surviving anomaly becomes its own researched section and its own web
# search, so the cap is a spend limit as much as a reading limit. Four is what
# fits a report a person will read to the end.
MAX_ANOMALIES = 4

# Why a question came back unresearched, in the words the prompt block uses.
# The caller stamps the key on the anomaly; the block never infers it. There
# are three live reasons and they contradict each other, so a block that
# guessed would have the report state something false about its own run.
UNRESEARCHED_REASONS = {
    "no_web": "web research was off for this run",
    "capped": "this run's research budget went to higher-ranked questions",
    "failed": "the search failed on this run",
    "stage_failed": "the research stage failed for this symbol",
}
# When nothing stamped a reason. Says only what is certainly true.
UNRESEARCHED_DEFAULT = "this question was not researched on this run"

# The categories detect() can rule out, and the inputs each one needs. A run
# that did not fetch an input has NOT ruled its category out, and a report
# that says "no insider cluster" over a block that was never fetched is the
# absence-asserted-from-nothing failure this module exists to stop. screened()
# below is the honest list of what a run actually looked at, and it is told
# what was fetched rather than guessing it from whether rows came back.
# (label, "any"/"all", the detect() keywords it needs). One row per detector
# family, so the list a run reports is the list of detectors that could fire.
SCREENS = (
    ("the options chain", "any", ("options", "by_expiry")),
    ("insider filings", "any", ("insiders",)),
    ("congressional filings", "any", ("congress",)),
    ("the accounting-quality screen", "any", ("quality",)),
    ("news volume", "any", ("news",)),
    ("insider and options positioning against the price trend", "all",
     ("options", "insiders", "trend")),
    ("the price and volume tape", "any", ("bars",)),
)

# --- options -------------------------------------------------------------
# options_service reads pc_volume >= 1.0 as put-tilted and <= 0.5 as
# call-tilted, so the open interval between them is what "neutral" already
# means on this platform; severity measures how far past the edge of that band
# the ratio sits. One full point out (2.0 puts per call, or a chain with no
# puts at all) saturates.
NEUTRAL_PC_LOW = 0.5
NEUTRAL_PC_HIGH = 1.0
SKEW_SATURATION = 1.0
SKEW_BASE = 0.35
SKEW_SPAN = 0.45

# A near-dated expiry whose ratio sits at least half away from the whole-chain
# ratio is the chain saying something is expected before that date; the
# whole-chain number alone hides it (a 6x full chain can be a 13x front month
# against a 0.1x back month). Divergence is measured against the full chain, so
# 0.5 is "half the chain's own level" rather than half a ratio point.
TERM_DIVERGENCE_FRACTION = 0.5
TERM_SATURATION = 2.0
TERM_BASE = 0.4
TERM_SPAN = 0.45

# --- insiders ------------------------------------------------------------
# One executive selling is a calendar entry: vesting, a window opening, a
# 10b5-1 tranche. Several people moving the same way TOGETHER is the board
# acting on the same information, and "together" is days or weeks, not the six
# months the store serves. Four gates say so, and each earns its place against
# the local insider store read the way the pipeline reads it, through
# summarize_insiders (CRWD, MU, NVDA, HPE, each at 26 weekly as-of dates = 104
# symbol-days): counting three one-sided executives anywhere in the 180-day
# window at any price fired on 87 of those 104, so nearly every symbol claimed
# a research question and no report was ever short. The four together fire on
# 16 of 104, and what survives are events (four MU officers selling inside one
# month in July) rather than calendars.
#
#   window   only filings inside CLUSTER_WINDOW_DAYS count toward the cluster
#            (180d -> 30d alone: 87 -> 30 of 104)
#   priced   an unpriced row is a grant, a gift or an option exercise, which
#            the vendor sends at share price 0 and av_store stores with a NULL
#            value. Three directors taking the same annual award on the same
#            day is the single most common shape in the store (NVDA 2026-06-25,
#            CRWD 2026-06-18) and it is a calendar entry, not a decision
#            (30 -> 18 of 104)
#   base     the count has to beat what THIS symbol does anyway, measured over
#            the rest of the caller's window. Three MU officers selling in a
#            month is MU's normal month (18 -> 16 of 104)
#   same row the window is read off the PRICED span, so the filing that dates
#            the cluster is the filing that priced it. Reading recency off all
#            rows and money off the priced ones let a $9M sale from May be
#            reported as a disposal inside the last 30 days because an unpriced
#            grant landed in August. No symbol-day in this store has that shape
#            (16 -> 16 of 104), which is why it survived the retune above and
#            why the guard is a test rather than a number
#
# With the same window applied to the single-large-disposal branch below, the
# whole detector fires on 23 of those 104 symbol-days (16 cluster, 7 single
# disposal) instead of 87, and on none of the 4 symbols in the store as of
# 2026-09-02.
MIN_CLUSTER_EXECUTIVES = 3
CLUSTER_WINDOW_DAYS = 30
# How many times the symbol's own trailing rate the cluster has to be. Two,
# because the trailing rate is a small number over a short window and anything
# tighter reads noise as a signal.
CLUSTER_BASE_RATE_MULTIPLE = 2.0
# av_store.INSIDER_WINDOW_DAYS, restated rather than imported: this module
# touches no database and importing the store for one integer would pull the
# ORM in behind it. Only used when the caller passes rows with no window.
DEFAULT_INSIDER_WINDOW_DAYS = 180
CLUSTER_BASE = 0.45
CLUSTER_SPAN = 0.4
CLUSTER_SATURATION = 3

# A single Form 4 disposal this large is not tax withholding at any market cap
# this platform runs. Zero-price rows never reach the test: av_store leaves
# value_usd NULL for grants, gifts and option exercises, so a fabricated
# "$0 sale" cannot clear a dollar floor. CLUSTER_WINDOW_DAYS bounds this
# branch too: reading the largest row of the whole 180 days meant a symbol
# kept a section about a sale five months old for as long as the row stayed
# in the window (21 of the same 104 symbol-days, down to 7 with the gate).
LARGE_DISPOSAL_USD = 5_000_000
DISPOSAL_BASE = 0.4
DISPOSAL_SPAN = 0.4
DISPOSAL_SATURATION = 4

# How many distinct executives one side needs over the other before the
# insider tape counts as leaning at all.
INSIDER_LEAN_MARGIN = 2

# --- congress ------------------------------------------------------------
# Congressional disclosure is sparse — most symbols on the watchlist go a
# whole window with none — so a single filing is already the exception and the
# floor is one. What lifts it is a second member, or a purchase into a decline.
MIN_CONGRESS_TRADES = 1
CONGRESS_BASE = 0.3
CONGRESS_SECOND_MEMBER = 0.2
CONGRESS_AGAINST_TREND = 0.25

# --- quality -------------------------------------------------------------
# bad_apples_service calls 5 failed checks "caution" and 8 "bad_apple". Reusing
# its own boundary keeps one definition of a failing screen in the codebase.
QUALITY_FAIL_FLOOR = 5
QUALITY_BAD_APPLE = 8
QUALITY_BASE = 0.35
QUALITY_SPAN = 0.4

# --- news ----------------------------------------------------------------
# Daily article counts are small integers, so a multiple alone fires on 1 -> 3.
# Both gates must clear: enough articles on the day to be a story, and enough
# multiple of this symbol's OWN average over the rest of the window — the norm
# is the symbol's, never a cross-symbol constant, because coverage density
# differs by an order of magnitude between a mega cap and a small cap.
MIN_SPIKE_ARTICLES = 4
SPIKE_MULTIPLE = 2.5
MIN_BASELINE_DAYS = 3
SPIKE_BASE = 0.3
SPIKE_SPAN = 0.45
SPIKE_SATURATION = 3.0

# --- price trend ---------------------------------------------------------
# 20 sessions is a month of tape: long enough that one gap does not set the
# direction, short enough to still be the current move. Below 3% "trending"
# overstates a drift for anything liquid.
TREND_DAYS = 20
TREND_PCT = 3.0
MIN_TREND_BARS = 10

# --- price shock ------------------------------------------------------------
# One session's close-to-close move against the symbol's OWN recent daily
# moves, so a 4% day on a utility fires and the same day on a biotech does
# not. The z-score is over the prior SHOCK_LOOKBACK sessions excluding the
# day itself; the absolute floor stops a dead-calm name flagging a 1% day.
SHOCK_LOOKBACK = 60
MIN_SHOCK_BARS = 21
SHOCK_Z = 2.5
SHOCK_MIN_PCT = 3.0
SHOCK_BASE = 0.4
SHOCK_SPAN = 0.45
SHOCK_SATURATION = 3.0     # z-score points past SHOCK_Z that saturate

# --- volume shock -----------------------------------------------------------
# The last session's volume against the median of the prior VOLUME_LOOKBACK
# sessions. Median, not mean: one earlier blow-off day would otherwise hide
# this one. The share floor keeps a name that trades a few thousand shares
# from firing on a single block.
VOLUME_LOOKBACK = 20
VOLUME_MULTIPLE = 2.5
MIN_SHOCK_VOLUME = 100_000
VOLUME_BASE = 0.35
VOLUME_SPAN = 0.45
VOLUME_SATURATION = 3.0    # multiples past VOLUME_MULTIPLE that saturate

# --- options order flow -----------------------------------------------------
# Contracts traded in the session against contracts standing open: turnover
# near or above 1.0 means the session opened (or closed) as many positions as
# existed, which is orders being placed, not a skew that has stood for weeks.
# Reads off the same metrics block as the skew detector, so it fires on a
# balanced chain too: heavy two-sided flow is its own question.
FLOW_TURNOVER = 0.75
FLOW_BASE = 0.4
FLOW_SPAN = 0.45
FLOW_SATURATION = 1.5      # turnover past FLOW_TURNOVER that saturates

# --- positioning vs price ------------------------------------------------
# Two independent parties positioned against the tape. Weighted above every
# single-source detector on purpose: it is the only one that speaks to how the
# symbol may behave rather than how it has behaved, so when it fires it takes a
# section even against a louder-looking number. The floor is written as the
# ceiling of the loudest single-source detector rather than as a literal, so
# retuning one of those cannot quietly demote this one below it.
POSITIONING_BASE = CLUSTER_BASE + CLUSTER_SPAN
POSITIONING_SPAN = 1.0 - POSITIONING_BASE
POSITIONING_SATURATION = 3.0

ACQUIRED = "A"
DISPOSED = "D"


def _num(value):
    """A finite float, or None. Yahoo null-bucket bars and vendor blanks both
    arrive as NaN, and NaN compares false against every threshold below except
    the ones written as ``not >``, so it is stripped once here."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _severity(base: float, span: float, ratio: float) -> float:
    """Base severity plus a share of ``span``, both ends clamped."""
    ratio = min(1.0, max(0.0, ratio if ratio is not None else 0.0))
    return round(min(1.0, max(0.0, base + span * ratio)), 3)


def _usd(v: float) -> str:
    v = abs(float(v))
    if v >= 1e9:
        return f"${v / 1e9:.2f}B"
    if v >= 1e6:
        return f"${v / 1e6:.1f}M"
    if v >= 1e3:
        return f"${v / 1e3:.0f}K"
    return f"${v:,.0f}"


def _as_of_date(as_of):
    try:
        return date.fromisoformat(str(as_of)[:10])
    except (TypeError, ValueError):
        return None


def _anomaly(key, title, severity, facts, question, evidence_key) -> dict:
    return {"key": key, "title": title, "severity": severity,
            "facts": [f for f in facts if f], "question": question,
            "evidence_key": evidence_key}


def _closes(ohlcv) -> list[float]:
    """Closing prices out of whatever the caller has: the point-in-time
    DataFrame the models already carry, a Close series, a list of bar dicts,
    or a plain list of prices."""
    if ohlcv is None:
        return []
    series = ohlcv
    try:
        series = ohlcv["Close"]
    except Exception:
        pass
    try:
        values = list(series)
    except TypeError:
        return []
    out = []
    for v in values:
        if isinstance(v, dict):
            v = v.get("Close", v.get("close"))
        f = _num(v)
        if f is not None:
            out.append(f)
    return out


def _column(ohlcv, name: str) -> list:
    """One OHLCV column as floats (None for a blank), or [] when the caller's
    frame does not carry it. A Close-only frame is a valid input for every
    detector that only needs closes."""
    if ohlcv is None:
        return []
    try:
        series = ohlcv[name]
    except Exception:
        return []
    try:
        return [_num(v) for v in list(series)]
    except TypeError:
        return []


def _bar_dates(ohlcv) -> list[str]:
    try:
        return [str(d)[:10] for d in list(ohlcv.index)]
    except Exception:
        return []


def price_trend(ohlcv, days: int = TREND_DAYS) -> dict | None:
    """Direction and size of the recent move, or None when there are too few
    bars to call one. Public because the positioning detector's facts quote it
    and a caller writing the section header wants the same numbers."""
    closes = _closes(ohlcv)
    if len(closes) < MIN_TREND_BARS:
        return None
    window = closes[-(days + 1):]
    first, last = window[0], window[-1]
    if not first:
        return None
    pct = (last - first) / abs(first) * 100.0
    direction = ("up" if pct >= TREND_PCT
                 else "down" if pct <= -TREND_PCT else "flat")
    return {"direction": direction, "pct": round(pct, 2),
            "sessions": len(window) - 1, "first": first, "last": last}


def _detect_price_shock(symbol, ohlcv) -> dict | None:
    closes = _closes(ohlcv)
    if len(closes) < MIN_SHOCK_BARS:
        return None
    rets = [(b - a) / abs(a) * 100.0 for a, b in zip(closes[:-1], closes[1:])
            if a]
    if len(rets) < MIN_SHOCK_BARS - 1:
        return None
    today, prior = rets[-1], rets[-(SHOCK_LOOKBACK + 1):-1]
    if abs(today) < SHOCK_MIN_PCT or len(prior) < MIN_SHOCK_BARS - 1:
        return None
    mean = sum(prior) / len(prior)
    var = sum((r - mean) ** 2 for r in prior) / max(1, len(prior) - 1)
    sigma = math.sqrt(var)
    if not sigma:
        return None
    z = (today - mean) / sigma
    if abs(z) < SHOCK_Z:
        return None

    dates = _bar_dates(ohlcv)
    day = dates[-1] if dates else "the last session"
    direction = "up" if today > 0 else "down"
    facts = [
        f"{symbol} closed {direction} {abs(today):.1f}% on {day} "
        f"({closes[-2]:.2f} to {closes[-1]:.2f})",
        f"that is {abs(z):.1f} standard deviations from its own mean daily "
        f"move over the prior {len(prior)} sessions (sigma {sigma:.2f}%), "
        f"measured against {symbol} alone",
    ]
    opens = _column(ohlcv, "Open")
    if opens and opens[-1] is not None and closes[-2]:
        gap = (opens[-1] - closes[-2]) / abs(closes[-2]) * 100.0
        if abs(gap) >= 1.0:
            facts.append(f"it opened {'up' if gap > 0 else 'down'} "
                         f"{abs(gap):.1f}% from the prior close, so most of "
                         f"the move was overnight, not intraday")
        else:
            facts.append("it opened within 1% of the prior close, so the "
                         "move was built during the session")
    return _anomaly(
        "price_shock",
        f"Price moved {abs(today):.1f}% in one session, {abs(z):.1f} sigma",
        _severity(SHOCK_BASE, SHOCK_SPAN, (abs(z) - SHOCK_Z) / SHOCK_SATURATION),
        facts,
        f"What moved {symbol} {abs(today):.1f}% on {day}, and is the cause "
        f"a one-off print or a change in the business?",
        "ohlcv")


def _detect_volume_shock(symbol, ohlcv) -> dict | None:
    volumes = _column(ohlcv, "Volume")
    vols = [v for v in volumes if v is not None and v >= 0]
    if len(vols) < len(volumes) or len(vols) < VOLUME_LOOKBACK + 1:
        return None
    today, prior = vols[-1], sorted(vols[-(VOLUME_LOOKBACK + 1):-1])
    median = prior[len(prior) // 2]
    if today < MIN_SHOCK_VOLUME or not median or today < median * VOLUME_MULTIPLE:
        return None
    multiple = today / median

    closes = _closes(ohlcv)
    dates = _bar_dates(ohlcv)
    day = dates[-1] if dates else "the last session"
    facts = [
        f"{int(today):,} shares traded on {day} against a median of "
        f"{int(median):,} over the prior {len(prior)} sessions",
        f"that is {multiple:.1f}x the symbol's own recent norm, measured "
        f"against {symbol} alone and not against any cross-symbol constant",
    ]
    if len(closes) >= 2 and closes[-2]:
        move = (closes[-1] - closes[-2]) / abs(closes[-2]) * 100.0
        facts.append(f"the price {'rose' if move >= 0 else 'fell'} "
                     f"{abs(move):.1f}% on that volume"
                     + ("" if abs(move) >= 1.0 else
                        ", so the tape absorbed the flow without moving: "
                        "a hand-off, not a break"))
    return _anomaly(
        "volume_shock",
        f"Volume {multiple:.1f}x its own norm",
        _severity(VOLUME_BASE, VOLUME_SPAN,
                  (multiple - VOLUME_MULTIPLE) / VOLUME_SATURATION),
        facts,
        f"Who was trading {symbol} on {day} at {multiple:.1f}x its normal "
        f"volume, and was it a print, an index or fund event, or news?",
        "ohlcv")


def _detect_options_flow(symbol, options) -> dict | None:
    if not isinstance(options, dict):
        return None
    total = _num(options.get("total_volume"))
    put_vol = _num(options.get("put_volume")) or 0.0
    call_vol = _num(options.get("call_volume")) or 0.0
    put_oi = _num(options.get("put_oi")) or 0.0
    call_oi = _num(options.get("call_oi")) or 0.0
    open_interest = put_oi + call_oi
    if total is None or total < MIN_LIQUID_VOLUME or open_interest <= 0:
        return None
    turnover = total / open_interest
    if turnover < FLOW_TURNOVER:
        return None

    put_turn = (put_vol / put_oi) if put_oi else None
    call_turn = (call_vol / call_oi) if call_oi else None
    if put_turn is not None and call_turn is not None:
        if put_turn >= call_turn * 1.5:
            side = "puts"
        elif call_turn >= put_turn * 1.5:
            side = "calls"
        else:
            side = "both sides"
        split = (f"the session's orders were concentrated in {side}: put "
                 f"volume was {put_turn:.2f}x put open interest, call volume "
                 f"{call_turn:.2f}x call open interest")
    else:
        side = "puts" if put_turn is not None else "calls"
        split = (f"the session's orders were all in {side}: the other side "
                 f"has no open interest to measure against")
    facts = [
        f"{int(total):,} contracts traded against {int(open_interest):,} "
        f"standing open, a turnover of {turnover:.2f} as of "
        f"{options.get('as_of') or 'the chain date'}",
        split,
        f"the chain covers {options.get('expiry_window') or 'all'} "
        f"expiries, sourced from {options.get('source') or 'the vendor'}",
    ]
    return _anomaly(
        "options_flow",
        f"Options orders at {turnover:.2f}x open interest, concentrated in {side}",
        _severity(FLOW_BASE, FLOW_SPAN, (turnover - FLOW_TURNOVER) / FLOW_SATURATION),
        facts,
        f"Who placed a session's worth of {symbol} options orders equal to "
        f"{turnover:.2f}x the standing open interest, weighted to {side}, "
        f"and against what event?",
        "options")


def _insider_people(insiders):
    """(people, rows) from either shape the pipeline already computed.

    A dict is ``insider_service.summarize_insiders`` output (``by_executive``
    is already aggregated, ``recent`` is the head of the raw rows); a list is
    ``av_store.insider_transactions_for`` output and is aggregated here.
    """
    if isinstance(insiders, dict):
        people = [p for p in (insiders.get("by_executive") or [])
                  if isinstance(p, dict)]
        rows = [r for r in (insiders.get("recent") or []) if isinstance(r, dict)]
        return people, rows
    if isinstance(insiders, (list, tuple)):
        rows = [r for r in insiders if isinstance(r, dict)]
        people: dict[str, dict] = {}
        for r in rows:
            name = r.get("executive") or "undisclosed filer"
            p = people.setdefault(name, {
                "executive": name, "title": r.get("title"),
                "acquired": 0, "disposed": 0,
                "value_acquired": 0.0, "value_disposed": 0.0,
                "priced_acquired": 0, "priced_disposed": 0,
                "first_priced_acquired": None, "latest_priced_acquired": None,
                "first_priced_disposed": None, "latest_priced_disposed": None})
            side = (r.get("side") or "").upper()
            value = _num(r.get("value_usd")) or 0.0
            key = ("acquired" if side == ACQUIRED
                   else "disposed" if side == DISPOSED else None)
            if not key:
                continue
            p[key] += 1
            p[f"value_{key}"] += value
            if value > 0:
                # Same per-side priced span and count summarize_insiders
                # keeps, built here so the two input shapes read identically.
                # PRICED rows only: the window test and the value are then
                # asked of the same filings, and a person whose only recent
                # row is an unpriced grant cannot be counted as having
                # sold or bought inside the cluster window.
                p[f"priced_{key}"] += 1
                _span(p, f"priced_{key}", r.get("transaction_date"))
        return list(people.values()), rows
    return [], []


def _span(person: dict, side: str, when) -> None:
    """Widen ``person``'s first/latest dates on ``side`` to include ``when``.
    ISO strings compare as dates, so no parsing is needed to order them."""
    when = str(when)[:10] if when else None
    if not when:
        return
    first, latest = person.get(f"first_{side}"), person.get(f"latest_{side}")
    if first is None or when < first:
        person[f"first_{side}"] = when
    if latest is None or when > latest:
        person[f"latest_{side}"] = when


def _one_sided(people, side: str) -> list[dict]:
    """Executives who transacted only in ``side`` inside the window. A person
    who both acquired and disposed is not evidence of a direction: an award
    followed by a same-day sale to cover the tax is one event, not two."""
    other = "acquired" if side == "disposed" else "disposed"
    return [p for p in people
            if int(p.get(side) or 0) > 0 and not int(p.get(other) or 0)]


def _insider_lean(people) -> str | None:
    """Which way the insider tape leans, counted in people rather than rows so
    one executive filing six tranches does not outvote three colleagues."""
    sellers = len(_one_sided(people, "disposed"))
    buyers = len(_one_sided(people, "acquired"))
    if sellers - buyers >= INSIDER_LEAN_MARGIN:
        return "bearish"
    if buyers - sellers >= INSIDER_LEAN_MARGIN:
        return "bullish"
    return None


def _options_lean(options) -> str | None:
    """Which way the chain leans, or None when it is balanced or too thin to
    read. Put-tilted is bearish positioning, call-tilted bullish."""
    if not isinstance(options, dict):
        return None
    total = _num(options.get("total_volume")) or 0.0
    if total < MIN_LIQUID_VOLUME:
        return None
    read = options.get("read")
    if read == "put-tilted":
        return "bearish"
    if read == "call-tilted":
        return "bullish"
    return None


def _quality_counts(quality):
    """(total_fails, total_checks, failed check labels) from either the raw
    ``bad_apples_service.analyze_symbol`` result or its ``summarize`` form."""
    if not isinstance(quality, dict):
        return None, None, []
    fails = _num(quality.get("total_fails"))
    checks = _num(quality.get("total_checks"))
    if fails is None:
        return None, None, []
    named = []
    for c in (quality.get("failed_checks") or []):
        if isinstance(c, dict) and c.get("check"):
            named.append(f"{c.get('category') or 'check'}: {c['check']}")
    if not named:
        for c in (quality.get("checks") or []):
            if isinstance(c, dict) and c.get("status") == "fail" and c.get("check"):
                named.append(f"{c.get('category') or 'check'}: {c['check']}")
    return int(fails), (int(checks) if checks is not None else None), named


def _article_field(a, name: str):
    """Either shape the pipeline carries an article in: the NewsArticle
    dataclass or its dict form."""
    if isinstance(a, dict):
        return a.get(name)
    return getattr(a, name, None)


def _detect_options_skew(symbol, options) -> dict | None:
    if not isinstance(options, dict):
        return None
    read = options.get("read")
    if read not in ("put-tilted", "call-tilted"):
        return None
    total = _num(options.get("total_volume"))
    pc = _num(options.get("pc_volume"))
    if total is None or total < MIN_LIQUID_VOLUME or pc is None:
        return None

    if read == "put-tilted":
        distance = (pc - NEUTRAL_PC_HIGH) / SKEW_SATURATION
        side = (f"{pc:.2f} puts traded per call against "
                f"{int(total):,} contracts on the chain")
        question = (f"What are options traders on {symbol} hedging or betting "
                    f"against, to leave the chain at {pc:.2f} puts per call?")
    else:
        distance = (NEUTRAL_PC_LOW - pc) / NEUTRAL_PC_LOW
        side = (f"{pc:.2f} puts traded per call ({1 / pc:.1f} calls per put) "
                f"against {int(total):,} contracts on the chain"
                if pc else
                f"no put volume at all against {int(total):,} contracts")
        question = (f"What are options traders on {symbol} positioning for, to "
                    f"leave the chain at {pc:.2f} puts per call?")

    facts = [
        side,
        f"put volume {int(options.get('put_volume') or 0):,} vs call volume "
        f"{int(options.get('call_volume') or 0):,} as of "
        f"{options.get('as_of') or 'the chain date'}",
    ]
    pc_oi = _num(options.get("pc_oi"))
    if pc_oi is not None:
        facts.append(
            f"open interest is {pc_oi:.2f} puts per call, so today's flow "
            f"{'matches' if (pc_oi >= 1) == (pc >= 1) else 'runs against'} "
            f"the standing position")
    facts.append(f"the chain covers {options.get('expiry_window') or 'all'} "
                 f"expiries, sourced from {options.get('source') or 'the vendor'}")
    return _anomaly("options_skew",
                    f"Options chain is {read}",
                    _severity(SKEW_BASE, SKEW_SPAN, distance),
                    facts, question, "options")


def _detect_options_term_divergence(symbol, by_expiry, as_of) -> dict | None:
    if not isinstance(by_expiry, dict):
        return None
    full = _num(by_expiry.get("full_chain"))
    rows = []
    for r in (by_expiry.get("by_expiry") or []):
        try:
            when, ratio = r[0], _num(r[1])
        except (TypeError, IndexError, KeyError):
            continue
        if ratio is not None and when:
            rows.append((str(when)[:10], ratio))
    if not full or full <= 0 or not rows:
        return None

    rows.sort(key=lambda r: r[0])
    when, near = rows[0]
    divergence = abs(near - full) / full
    if divergence < TERM_DIVERGENCE_FRACTION:
        return None

    ref = _as_of_date(by_expiry.get("as_of") or as_of)
    expiry = _as_of_date(when)
    days_out = (expiry - ref).days if (ref and expiry) else None
    horizon = f" ({days_out}d out)" if days_out is not None and days_out >= 0 else ""
    direction = "above" if near > full else "below"
    facts = [
        f"the {when} expiry{horizon} prices {near:.2f} puts per call while the "
        f"whole chain sits at {full:.2f}",
        f"that is {divergence * 100:.0f}% {direction} the chain, so the tilt is "
        f"concentrated in a dated window rather than spread across the curve",
    ]
    if len(rows) > 1:
        later = ", ".join(f"{d} {v:.2f}" for d, v in rows[1:4])
        facts.append(f"the expiries behind it read {later}")
    return _anomaly(
        "options_term_divergence",
        "Near-dated expiry diverges from the chain",
        _severity(TERM_BASE, TERM_SPAN, divergence / TERM_SATURATION),
        facts,
        f"What is the options market bracing for in {symbol} before {when}, "
        f"where positioning is {near:.2f} against {full:.2f} for the whole chain?",
        "options")


def _cluster_group(people, side, cutoff):
    """(acting together now, active earlier) on ``side``.

    Both lists are drawn from the executives who moved ONE way over the
    caller's whole window, and both the recency test and the base rate read
    the PRICED span: a person's priced rows and their recent rows are not
    necessarily the same rows, and asking the two questions of different
    filings let a $9M sale from May be reported as a disposal inside the last
    30 days because an unpriced grant landed in August. A person priced on
    both sides of ``cutoff`` is in both lists, which is what a rate wants.
    """
    group = [p for p in _one_sided(people, side)
             if int(p.get(f"priced_{side}") or 0) > 0]
    recent = [p for p in group
              if (p.get(f"latest_priced_{side}") or "") >= cutoff]
    earlier = [p for p in group
               if (p.get(f"first_priced_{side}") or "9999") < cutoff]
    return recent, earlier


def _detect_insider_cluster(symbol, insiders, window_days, as_of) -> dict | None:
    people, rows = _insider_people(insiders)
    if not people:
        return None

    ref = _as_of_date(as_of)
    if ref is None:
        # Both branches below are bounded by the cluster window, and an
        # unreadable as-of gives no window to bound them with. Reading the
        # whole six months instead is the flat count this was retuned away
        # from, so the honest answer is that nothing was detected.
        return None
    window_days = int(window_days or DEFAULT_INSIDER_WINDOW_DAYS)
    window = f"the last {window_days} days"
    cutoff = (ref - timedelta(days=CLUSTER_WINDOW_DAYS)).isoformat()
    # The rest of the caller's window is the base rate; a caller whose window
    # is no longer than the cluster window has no earlier period to compare to.
    base_days = max(window_days - CLUSTER_WINDOW_DAYS, 0)
    for side, noun in (("disposed", "disposals"), ("acquired", "acquisitions")):
        cluster, earlier = _cluster_group(people, side, cutoff)
        if len(cluster) < MIN_CLUSTER_EXECUTIVES:
            continue
        rate = len(earlier) * CLUSTER_WINDOW_DAYS / base_days if base_days else 0.0
        if len(cluster) < CLUSTER_BASE_RATE_MULTIPLE * rate:
            continue
        shown = cluster[:MIN_CLUSTER_EXECUTIVES + 1]
        named = ", ".join(
            f"{p.get('executive')}"
            + (f" ({p['title']})" if p.get("title") else "")
            for p in shown)
        if len(cluster) > len(shown):
            named += f", and {len(cluster) - len(shown)} more"
        value = sum(_num(p.get(f"value_{side}")) or 0.0 for p in cluster)
        filings = sum(int(p.get(f"priced_{side}") or 0) for p in cluster)
        facts = [
            f"{len(cluster)} distinct executives filed priced {noun} inside the "
            f"last {CLUSTER_WINDOW_DAYS} days, and none of them transacted the "
            f"other way anywhere in {window}",
            f"they are {named}",
            (f"over the {base_days} days before that, this symbol averaged "
             f"{rate:.1f} such executives per {CLUSTER_WINDOW_DAYS} days, so "
             f"this is {len(cluster) / rate:.1f}x its own recent rate"
             if rate else
             f"no executive filed one-sided priced {noun} at all over the "
             f"{base_days} days before that"),
            f"{filings} priced filings from them across {window}, worth "
            f"{_usd(value)}; grants, gifts and option exercises are priced at "
            f"zero by the vendor and are excluded from this count, from the "
            f"total beside it and from the executives named above",
        ]
        if side == "acquired":
            facts.append("a priced acquisition is an open-market purchase or "
                         "an exercise settled in cash, not a vesting award: "
                         "awards come through unpriced and are not counted")
        ratio = (len(cluster) - MIN_CLUSTER_EXECUTIVES) / CLUSTER_SATURATION
        return _anomaly(
            "insider_cluster",
            f"{len(cluster)} insiders {side} within {CLUSTER_WINDOW_DAYS} days",
            _severity(CLUSTER_BASE, CLUSTER_SPAN, ratio),
            facts,
            f"What are the {len(cluster)} {symbol} executives who all {side} "
            f"within {CLUSTER_WINDOW_DAYS} days of {str(as_of)[:10]} acting "
            f"on, and do their filings give a reason?",
            "insiders")

    # No cluster: a single disposal can still be the whole story if it is big
    # enough and RECENT. Candidates come from the rows when the caller passed
    # them, and from any executive with exactly one disposal (whose window
    # total IS that one row) when the caller passed a summary. The same
    # cluster window bounds both: an $8M sale five months ago is a fact about
    # the window, not something that stands out today, and before this gate it
    # was the largest such row in the whole 180 days that decided the section.
    candidates = []
    for r in rows:
        if (r.get("side") or "").upper() != DISPOSED:
            continue
        when = str(r.get("transaction_date") or "")[:10]
        value = _num(r.get("value_usd"))
        if value and when >= cutoff:
            candidates.append((value, r.get("executive") or "an executive",
                               r.get("title"), when))
    for p in people:
        # The priced span, for the same reason the cluster reads it: the one
        # disposal this branch prices has to BE the row inside the window.
        when = str(p.get("latest_priced_disposed") or "")[:10]
        if int(p.get("disposed") or 0) == 1 and when >= cutoff:
            value = _num(p.get("value_disposed"))
            if value:
                candidates.append((value, p.get("executive") or "an executive",
                                   p.get("title"), when or None))
    if not candidates:
        return None
    value, who, title, when = max(candidates, key=lambda c: c[0])
    if value < LARGE_DISPOSAL_USD:
        return None
    person = f"{who} ({title})" if title else who
    dated = f" on {when}" if when else f" in {window}"
    facts = [
        f"{person} disposed of {_usd(value)} of {symbol}{dated} in a single "
        f"filing",
        f"the floor for this to register is {_usd(LARGE_DISPOSAL_USD)}, and "
        f"only rows carrying a real share price can reach it",
        f"{len(people)} executive(s) filed in {window} in total",
    ]
    ratio = (value / LARGE_DISPOSAL_USD - 1) / DISPOSAL_SATURATION
    return _anomaly(
        "insider_cluster",
        f"Single insider disposal of {_usd(value)}",
        _severity(DISPOSAL_BASE, DISPOSAL_SPAN, ratio),
        facts,
        f"Why did {who} dispose of {_usd(value)} of {symbol}, and was it a "
        f"scheduled 10b5-1 sale or a discretionary one?",
        "insiders")


def _detect_congress_activity(symbol, congress, dossier, trend) -> dict | None:
    rows = [t for t in (congress or []) if isinstance(t, dict)]
    if len(rows) < MIN_CONGRESS_TRADES:
        return None

    members = {}
    for t in rows:
        key = t.get("bioguide_id") or t.get("politician") or "unnamed filer"
        members.setdefault(key, t)
    names = []
    identities = {}
    for e in (dossier or {}).get("entries", []) if isinstance(dossier, dict) else []:
        if e.get("bioguide_id"):
            identities[e["bioguide_id"]] = e.get("identity")
    for key, t in members.items():
        names.append(identities.get(key) or t.get("politician") or "an unnamed filer")

    buys = [t for t in rows if (t.get("type") or "").startswith(("BUY", "PURCHASE"))]
    sells = [t for t in rows if (t.get("type") or "").startswith(("SELL", "SALE"))]
    against_trend = bool(buys) and bool(trend) and trend["direction"] == "down"

    severity = CONGRESS_BASE
    if len(members) >= 2:
        severity += CONGRESS_SECOND_MEMBER
    if against_trend:
        severity += CONGRESS_AGAINST_TREND
    severity = round(min(1.0, severity), 3)

    latest = max((t.get("transaction_date") or "" for t in rows), default="")
    facts = [
        f"{len(rows)} disclosed congressional trade(s) in {symbol} by "
        f"{len(members)} member(s): {', '.join(names[:4])}",
        f"{len(buys)} purchase(s) and {len(sells)} sale(s)"
        + (f", most recent transaction {latest}" if latest else ""),
    ]
    sizes = [(_num(t.get("amount_min")), _num(t.get("amount_max"))) for t in rows]
    lo = sum(s[0] for s in sizes if s[0])
    hi = sum(s[1] for s in sizes if s[1])
    if lo or hi:
        facts.append(f"disclosed size totals {_usd(lo)} to {_usd(hi)}; the "
                     f"bands are the disclosure's own, not exact amounts")
    if against_trend:
        facts.append(f"the purchases land while the price is {trend['pct']:+.1f}% "
                     f"over {trend['sessions']} sessions, so the member is "
                     f"buying into a decline rather than following it")
    facts.append("a disclosure is visible only from its filing date, which can "
                 "trail the transaction by weeks")
    who = names[0] if names else "the filer"
    return _anomaly(
        "congress_activity",
        f"{len(members)} member(s) of Congress traded {symbol}",
        severity, facts,
        f"Why is {who} trading {symbol}, what does their committee work touch, "
        f"and how does this fit their disclosed record?",
        "politicians")


def _detect_quality_failures(symbol, quality) -> dict | None:
    fails, checks, named = _quality_counts(quality)
    if fails is None or fails < QUALITY_FAIL_FLOOR:
        return None
    denominator = f" of {checks}" if checks else ""
    verdict = "BAD APPLE" if fails >= QUALITY_BAD_APPLE else "CAUTION"
    facts = [
        f"{fails}{denominator} quality checks failed on the Bad Apples "
        f"screen, which the screen itself calls {verdict}",
    ]
    if named:
        facts.append("the failures are " + "; ".join(named[:5]))
    by_cat = quality.get("by_category") if isinstance(quality, dict) else None
    if isinstance(by_cat, dict):
        worst = sorted(((c, int(n)) for c, n in by_cat.items() if n),
                       key=lambda x: -x[1])[:3]
        if worst:
            facts.append("concentrated in " +
                         ", ".join(f"{c} ({n})" for c, n in worst))
    facts.append("this is a quality and risk screen, not a timing signal: it "
                 "argues for size and skepticism, not for direction")
    ratio = (fails - QUALITY_FAIL_FLOOR) / max(
        1, QUALITY_BAD_APPLE - QUALITY_FAIL_FLOOR)
    return _anomaly(
        "quality_failures",
        f"{fails} quality checks failing",
        _severity(QUALITY_BASE, QUALITY_SPAN, ratio),
        facts,
        f"Do {symbol}'s {fails} failed quality checks reflect a deteriorating "
        f"business, or are they artefacts of the screen's inputs?",
        "quality")


def _detect_news_spike(symbol, news) -> dict | None:
    articles = list(news or [])
    if not articles:
        return None
    by_day: dict[str, int] = {}
    for a in articles:
        published = _article_field(a, "published_at")
        day = str(published)[:10] if published else ""
        if len(day) == 10:
            by_day[day] = by_day.get(day, 0) + 1
    if len(by_day) < MIN_BASELINE_DAYS + 1:
        return None

    days = sorted(by_day)
    peak_day = max(days, key=lambda d: (by_day[d], d))
    peak = by_day[peak_day]
    baseline_days = [d for d in days if d != peak_day]
    baseline = sum(by_day[d] for d in baseline_days) / len(baseline_days)
    if peak < MIN_SPIKE_ARTICLES or not baseline or peak < baseline * SPIKE_MULTIPLE:
        return None

    multiple = peak / baseline
    headlines = []
    for a in articles:
        published = _article_field(a, "published_at")
        if str(published)[:10] == peak_day:
            title = _article_field(a, "title")
            if title:
                headlines.append(str(title)[:120])
    facts = [
        f"{peak} articles on {peak_day} against {baseline:.1f}/day over the "
        f"other {len(baseline_days)} day(s) of this run's news window",
        f"that is {multiple:.1f}x the symbol's own recent norm, measured "
        f"against {symbol} alone and not against any cross-symbol constant",
    ]
    if headlines:
        facts.append("headlines that day include: " + "; ".join(headlines[:3]))
    return _anomaly(
        "news_spike",
        f"News volume {multiple:.1f}x its own norm",
        _severity(SPIKE_BASE, SPIKE_SPAN, (multiple - SPIKE_MULTIPLE)
                  / SPIKE_SATURATION),
        facts,
        f"What drove {peak} {symbol} stories on {peak_day}, and is it one "
        f"event being syndicated or several?",
        "news_source")


def _detect_positioning_vs_price(symbol, trend, options, insiders) -> dict | None:
    if not trend or trend["direction"] == "flat":
        return None
    people, _ = _insider_people(insiders)
    insider_lean = _insider_lean(people)
    options_lean = _options_lean(options)
    if not insider_lean or not options_lean or insider_lean != options_lean:
        return None
    opposed = ("bearish" if trend["direction"] == "up" else "bullish")
    if insider_lean != opposed:
        return None

    sellers = len(_one_sided(people, "disposed"))
    buyers = len(_one_sided(people, "acquired"))
    pc = _num((options or {}).get("pc_volume"))
    leaning = (f"{sellers} distinct executives selling against {buyers} buying"
               if opposed == "bearish" else
               f"{buyers} distinct executives buying against {sellers} selling")
    facts = [
        f"price is {trend['pct']:+.1f}% over {trend['sessions']} sessions "
        f"({trend['first']:.2f} to {trend['last']:.2f}), and both parties who "
        f"had to commit money are positioned the other way",
        f"insiders: {leaning} inside the window",
        f"options: the chain reads {(options or {}).get('read')}"
        + (f" at {pc:.2f} puts per call" if pc is not None else ""),
        "this is the one reading here that speaks to how the symbol may "
        "behave rather than how it has behaved, and it is weighted above the "
        "single-source anomalies for that reason",
        "two independent parties agreeing is not a forecast: insiders sell "
        "for liquidity and chains carry hedges, so the question is what would "
        "make both explanations innocent at once",
    ]
    edge = max(abs(sellers - buyers) - INSIDER_LEAN_MARGIN, 0)
    return _anomaly(
        "positioning_vs_price",
        f"Insiders and options lean {opposed} while price is "
        f"{'rising' if trend['direction'] == 'up' else 'falling'}",
        _severity(POSITIONING_BASE, POSITIONING_SPAN, edge / POSITIONING_SATURATION),
        facts,
        f"Why are {symbol} insiders and the options chain positioned "
        f"{opposed} while the stock is {trend['pct']:+.1f}% over "
        f"{trend['sessions']} sessions?",
        "ohlcv")


def detect(symbol, as_of, *, options=None, by_expiry=None, insiders=None,
           congress=None, dossier=None, quality=None, news=None,
           ohlcv=None) -> list[dict]:
    """What stands out for ``symbol`` as of ``as_of``, most severe first.

    Every argument is a block the run has already computed:
    ``options`` from ``options_service.get_put_call_metrics``, ``by_expiry``
    from ``get_put_call_by_expiry``, ``insiders`` from either
    ``insider_service.summarize_insiders`` or ``av_store.insider_transactions_for``,
    ``congress`` from ``av_store.congress_trades_for``, ``dossier`` from
    ``politician_dossier.build_dossier`` (names only, it adds no detector of
    its own), ``quality`` from ``bad_apples_service.analyze_symbol`` or its
    ``summarize`` form, ``news`` the run's point-in-time articles and
    ``ohlcv`` the truncated price frame (Close for the trend and the price
    shock, Volume for the volume shock, Open for the gap fact when present).

    Nothing is fetched and nothing is inferred from an absent block: passing
    no arguments returns an empty list, which is the correct description of a
    run that gathered nothing.
    """
    symbol = (symbol or "").strip().upper()
    trend = price_trend(ohlcv)
    window_days = (insiders.get("window_days")
                   if isinstance(insiders, dict) else None)

    found = [a for a in (
        _detect_price_shock(symbol, ohlcv),
        _detect_volume_shock(symbol, ohlcv),
        _detect_options_skew(symbol, options),
        _detect_options_flow(symbol, options),
        _detect_options_term_divergence(symbol, by_expiry, as_of),
        _detect_insider_cluster(symbol, insiders, window_days, as_of),
        _detect_congress_activity(symbol, congress, dossier, trend),
        _detect_quality_failures(symbol, quality),
        _detect_news_spike(symbol, news),
        _detect_positioning_vs_price(symbol, trend, options, insiders),
    ) if a]

    # Severity decides; the key breaks ties so two runs over the same evidence
    # research the same questions in the same order.
    found.sort(key=lambda a: (-a["severity"], a["key"]))
    kept, dropped = found[:MAX_ANOMALIES], found[MAX_ANOMALIES:]
    if dropped:
        logger.info(
            "%s: %d anomalies over the cap of %d not researched: %s",
            symbol, len(dropped), MAX_ANOMALIES,
            "; ".join(f"{a['key']} ({a['severity']})" for a in dropped))
    logger.debug("%s: %d anomaly/anomalies as of %s: %s", symbol, len(kept),
                 str(as_of)[:10],
                 ", ".join(f"{a['key']}={a['severity']}" for a in kept) or "none")
    return kept


def screened(gathered: "dict | None" = None, *, ohlcv=None) -> list[str]:
    """The categories ``detect`` could actually rule out on this run.

    ``gathered`` maps ``detect``'s input keywords ("options", "by_expiry",
    "insiders", "congress", "quality", "news") to whether the run FETCHED
    that input. Fetched, not non-empty: the two are different questions and
    reading the second as the first is how this went wrong. Congressional
    disclosure is sparse by design — most symbols go a whole window with
    none — so the common case is a dossier block that rendered, said no
    member of Congress traded the name, and handed back zero rows. Inferring
    gathered-ness from the truthiness of those rows put "this run did NOT
    screen congressional filings, they were not looked at" in the same
    prompt as the block stating they were checked and none were found, and
    made the report unable to ever say that nobody in Congress traded it.

    ``detect`` returning [] means "nothing stood out in THESE categories",
    and only this list says which they were. Without it the empty list reads
    as "nothing stands out anywhere", which is a claim about evidence the run
    may never have fetched.

    "trend" and "bars" are not fetches and are not the caller's to declare:
    a frame with three bars in it was gathered but still cannot carry a
    trend or a shock baseline, so both are derived from ``ohlcv`` here and
    the screens that need them count it absent.
    """
    have = {k: bool(v) for k, v in (gathered or {}).items()}
    have["trend"] = bool(price_trend(ohlcv))
    have["bars"] = len(_closes(ohlcv)) >= MIN_SHOCK_BARS
    out = []
    for label, mode, needs in SCREENS:
        got = [have.get(k, False) for k in needs]
        if (all(got) if mode == "all" else any(got)):
            out.append(label)
    return out


def format_anomaly_block(symbol: str, anomaly: dict,
                         answer: dict | None = None) -> str:
    """The prompt block for one anomaly: its figures, and what the research
    aimed at it found.

    ``answer`` is one entry from ``investigation_service.research_questions``
    ({question, finding, citations}). None, or an entry carrying an error,
    renders as NOT RESEARCHED and says so in the block, because a section
    written as though a cause were established is the whole failure this
    module exists to stop. Severity is left out on purpose: it orders the
    questions and is not a fact about the company, so a report has no
    business quoting it.

    WHY it went unresearched is read off ``anomaly["unresearched"]``, never
    inferred here. The block is the report model's only account of the run's
    own provenance, and a block that guessed "web research was off" on a run
    where it was on would have the model state that in writing.
    """
    key, title = anomaly.get("key") or "anomaly", anomaly.get("title") or ""
    # Hoisted out of the f-string below: a backslash inside an f-string
    # expression is a SyntaxError before Python 3.12 and the container runs
    # 3.11, where the whole module would fail to import.
    source = anomaly.get("evidence_key") or "this run's evidence"
    lines = [f'[{symbol} ANOMALY "{key}": {title}. Source block: {source}]',
             f"The question this raises: {anomaly.get('question') or ''}",
             "What the evidence shows (quote these figures exactly, "
             "compute nothing from them):"]
    lines += [f"- {f}" for f in (anomaly.get("facts") or [])]

    finding = (answer or {}).get("finding") or ""
    if answer and finding and not answer.get("error"):
        searches = answer.get("searches") or 0
        lines.append(f"Web research on that question ({searches} "
                     f"{'search' if searches == 1 else 'searches'}):")
        lines.append(finding)
        cites = [c for c in (answer.get("citations") or []) if isinstance(c, dict)]
        if cites:
            lines.append("Sources for that finding:")
            for c in cites[:6]:
                stamp = f", {c['date']}" if c.get("date") else ""
                url = f", {c['url']}" if c.get("url") else ""
                lines.append(f"- {c.get('source') or 'unattributed'}{stamp}{url}")
        else:
            lines.append("The research returned no citation for that finding; "
                         "treat it as unsourced and say so if you use it.")
        if answer.get("unresolved"):
            lines.append(f"Left open by the research: {answer['unresolved']}")
    else:
        error = (answer or {}).get("error")
        if error:
            why = f"{UNRESEARCHED_REASONS['failed']} ({error})"
        else:
            why = UNRESEARCHED_REASONS.get(anomaly.get("unresearched"),
                                           UNRESEARCHED_DEFAULT)
        lines.append(f"NOT RESEARCHED: {why}, so nothing beyond the figures "
                     f"above is known about why this is happening. Write the "
                     f"section from those figures, state that the cause was "
                     f"not researched, and do not supply one of your own.")
    return "\n".join(lines)

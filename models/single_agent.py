"""Self-contained single-agent research model, decoupled from TradingAgents.

This is a native quant-news reimplementation of the single-LLM-call research
agent we previously imported from `tradingagents.baselines.single_agent`. That
import broke when the upstream fork's `main` advanced to v0.3.1 (the `baselines`
submodule lives only on a feature branch, and its pre-v0.3.0 internal API,
`tradingagents.llm`, `agent_utils.normalize_content`, the flat `get_*` tools,
`signal_processing.extract_*`: was all moved/renamed/deleted upstream).

Design goals (per the incorporation analysis, §6 Tier 0-C):
  * Robust / swappable: implements the `BaseModel` contract, so the research
    strategy is one interchangeable implementation behind a stable seam. No
    dependency on any TradingAgents internal. An upstream release can never
    silently kill this model again.
  * Lookahead-safe: every data block is bounded to `as_of`. News uses the
    half-open UTC window (services.news_window); OHLCV is staleness-guarded;
    fundamentals are filtered to `fiscalDateEnding <= as_of`.

The prompt is ported from our own `feat/ace_alpaca` `SINGLE_AGENT_PROMPT`
(already strong for a 1-5 day horizon: in-prompt data discipline, a
regime -> sector -> idiosyncratic framework, and falsification triggers), with
one correctness fix: the decision is framed as "at the close of {date}, predict
the next session", matching how the backtest scores it (close_T -> close_T+1)
and removing the old intraday-leak ambiguity.
"""

import logging
import re
from datetime import datetime, timezone
from typing import Any, Optional

import pandas as pd

from config import MODEL
from services.evidence_contract import EvidenceLedger
from services.news_window import (
    RunParameterMissing,
    article_span,
    select_spread,
    fetch_point_in_time_news,
    filter_articles_as_of,
)

logger = logging.getLogger(__name__)

# Reject an OHLCV frame whose latest row is more than this many calendar days
# before the requested date, catches the "present but wrong" year-old frame a
# simple empty-check misses (ported from TradingAgents MAX_OHLCV_STALE_DAYS).
MAX_OHLCV_STALE_DAYS = 10


SINGLE_AGENT_PROMPT = """You are a trading analyst deciding, at the close of {date}, whether {ticker} will move UP or DOWN in the NEXT trading session.

IMPORTANT: Use ONLY information available through the close of {date}. Every data
block below is bounded to that date; do not reason about anything that happened
afterward. Your thesis window is 1-5 trading days. Focus on catalysts and
momentum that play out within it. Ignore long-term (months/years) arguments.

Every number you cite must be traceable to the data below:
- Every price, indicator value, or statistic you cite MUST come verbatim from the
  data blocks below. Never estimate, extrapolate, or invent a number. If a value
  you need is not in the data, write "not in data" instead of guessing.
- Treat the PRECOMPUTED METRICS / verified blocks as the source of truth. Where two
  blocks give different values for the same thing, name both numbers and say which
  block each came from, then carry that uncertainty into the call. Never average
  them into a third number that appears in neither block. Do not claim historical
  support/resistance bounces or exact percentage moves unless a data block states
  them with concrete dates and prices.
- Do not prefix price LEVELS with +/- signs; signs belong on returns/changes only.
- A claim about a group (peers, sectors, models) must hold for EVERY member you
  name. If it doesn't, name only the members it holds for, or quantify exactly
  ("3 of 4 peers"), never stretch "all" or "much" over a member the numbers
  don't support.
- Use only the evidence in this prompt. You cannot browse. The SITUATION &
  INVESTIGATION block, when present, was gathered by a research stage WITH web
  access before this prompt was built and carries its own sources. Cite those
  exactly as you cite news. If something is missing, say so explicitly rather
  than filling it in. A block headed "Evidence NOT available" lists what this
  run could not gather; never reason as if those inputs were neutral.
- NEWS ATTRIBUTION: every claim you take from a news article must name the outlet
  and the publication date inline, e.g. "shares fell 18.9% (Reuters, 2026-08-14)".
  Each article below carries "src:" and a date. Use those exact values. If you
  cannot attribute a news claim to a listed article, leave the claim out entirely
  rather than stating it unsourced.
- WRITE FOR A READER WHO CANNOT SEE THESE INSTRUCTIONS. Never name, quote, or
  allude to any instruction, section label, or rule from this prompt in your
  output: no "framework", "step", "gate", "cap", "penalty", "discipline",
  "as instructed", or "per the rules". State conclusions about the market, never
  conclusions about your own instructions.
- The ALL-CAPS names in the verdict format below (REASSESS_TO_BUY, MOVE_TO_SELL,
  SINCE LAST REPORT, and the rest) are field labels for that block ONLY. They are
  machine keys, not words. Never write one inside a sentence anywhere else in the
  report: in prose, say "turn bullish"/"turn bearish" or name the price level.

Analyze ALL of the following data carefully before deciding.

== HOW OFTEN CALLS LIKE THIS RESOLVE CORRECTLY ==
{track_record_block}

== PRIOR STANCE ON {ticker} ==
{continuity_block}

== {ticker} BUSINESS PROFILE ==
{business_block}

== {ticker} PRICE ACTION ==
{price_block}

== {ticker} TECHNICAL INDICATORS ==
{tech_block}

== {ticker} FUNDAMENTALS (as of {date}) ==
{fundamentals_block}

== {ticker} NEWS (through close of {date}) ==
{news_block}

== BROAD MARKET CONTEXT (S&P 500 / SPY) ==
{spy_block}

== SECTOR CONTEXT ({sector_etf}) ==
{sector_block}
{extra_context}
== HOW TO REASON (this is the order to think in, not the order to write in) ==

**Step 0: Situation (read this first)**: {situation_line}
The situation decides how every other block is read:
- PENDING_ACQUISITION: the next 1-5 sessions are about deal-completion odds,
  not trend. Anchor to the offer price and the computed spread; weigh the
  regulatory milestones, the break risk and the decisive actors named in the
  block. Moving averages are secondary here. Never call a direction "because
  price is below its SMAs" when the spread to a cash offer is the real question.
  Triggers may be events (a regulator's order, a hearing, a vote) rather than
  price levels; a price trigger must be stated relative to the offer price.
- LEGAL_REGULATORY_OVERHANG, STRATEGIC_REVIEW, LEADERSHIP_CHANGE, DISTRESS,
  PRODUCT_OR_CONTRACT: lead with the situation, its dated milestones and who
  decides it; technicals describe how the tape is pricing it, not the thesis.
- EARNINGS_EVENT: the event gate in Step 4 applies.
- MOMENTUM_ONLY or no situation block: Steps 1-3 carry the call.

**Step 1: Market Regime (Systematic Risk)**: from the SPY block:
- SPY above 50 SMA AND 200 SMA with positive MACD = BULL (prior lean BUY)
- SPY below 50 SMA AND 200 SMA with negative MACD = BEAR (prior lean SELL)
- Mixed = NEUTRAL (no prior, decide on ticker-specific evidence)

**Step 1b: Sector Regime ({sector_etf})**: leading, lagging, or in line vs SPY.
Outperforming = tailwind (strengthens BUY); underperforming = headwind.

**Step 2: Idiosyncratic Analysis ({ticker})**
Bullish: price above 50 SMA; RSI 40-60 rising; MACD positive/bullish cross;
positive earnings surprise or guidance raise; strong FCF growth.
Bearish: price below 50 AND 200 SMA; RSI > 75 with declining volume; MACD bearish
divergence; negative revision/guidance cut; deteriorating margins or rising debt.
Neutral: RSI 45-55 with flat MACD; no material catalyst; tight range (<1% moves).

**Step 3: Combine Systematic + Idiosyncratic**
- BULL + bullish ticker = strong BUY; BULL + bearish = HOLD (support limits downside)
- BEAR + bullish = HOLD (headwinds cap upside); BEAR + bearish = strong SELL
- NEUTRAL = follow ticker-specific signals

**Step 4: Apply before finalizing**
- If a precomputed events block shows earnings or an ex-dividend date inside the
  hold window, keep CONVICTION at or below 0.6 and name that event in the Risk
  section as the reason the position is smaller than the evidence would allow.
- If {ticker} is down >20% from its period high, state a causal hypothesis backed
  by the news/fundamentals blocks; if none is identifiable, write exactly
  "cause unknown: elevated risk" and treat it as bearish.
- Any risk/reward claim must use the support/resistance and ATR arithmetic from
  the PRECOMPUTED METRICS block (when present) rather than your own arithmetic.

DECISIVENESS: commit to a clear BUY or SELL whenever the strongest evidence warrants
one. Reserve HOLD for when the evidence is genuinely balanced. Do not hedge by default.

WHAT CONVICTION MEANS: CONVICTION is YOUR estimated probability (0.0-1.0) that the
direction of your call is correct over the next 1-5 sessions. It is a judgement, not
a measurement: it is not the same thing as the measured track record above, and it
has not itself been scored against outcomes. 0.5 means no edge. Use it only when
evidence is genuinely balanced, and prefer HOLD in that case.

PRICE LEVELS ARE APPROXIMATE: the moving averages and support/resistance levels in
these blocks are recomputed each run from vendor bars that get revised, so a long
average can shift by a dollar or more between one session's report and the next.
Never present a level as an exact tripwire. Round every trigger, invalidation and
target level to the nearest $0.05 under $50, the nearest $0.25 from $50 to $500,
and the nearest $1 above $500, or give a narrow band ("$146.50-$147.00"). Say
"as of {date}" next to the levels in the Trade Plan so a reader knows when they
were measured.


== VOICE AND PUNCTUATION ==
Write like one analyst who means it, for a desk that will trade on it.
- Never use an em dash or an en dash as punctuation. Use a comma, a period, a colon
  after a label, or parentheses. Hyphens inside compound words are fine.
- No "it's not just X, it's Y" constructions. State Y.
- No "delve", "dive into", "deep dive", "landscape", "unlock", "navigate",
  "leverage" (except as the financial term), "robust", "seamless".
- No "in conclusion", "in summary", "to summarize", and no closing recap.
- No filler openers ("it is worth noting", "importantly"). Say the thing.
- Vary sentence length. One long sentence, then a short one. Do not write every
  sentence to the same length or the same shape.
- Do not pad. A section with three real points beats one with six restated ones.

== REQUIRED OUTPUT FORMAT (markdown) ==

You MUST BEGIN your response with exactly this block:

## Verdict
FINAL TRANSACTION PROPOSAL: **BUY** (or **SELL** or **HOLD**)
CONVICTION: **0.X** (this report's own probability that the direction is right, not a measured hit rate)
MEASURED ACCURACY: <copy the single line under "HOW OFTEN CALLS LIKE THIS RESOLVE CORRECTLY" above, word for word and digit for digit; do not paraphrase it, do not round it, and do not substitute a number of your own>
SINCE LAST REPORT: <one line. If there is a prior stance above: name its date and call, say whether either trigger it stated was actually met by the price action shown, and if your call differs from it while no trigger was met, say so plainly in this same line. If that prior report used the same data cutoff as this run, no new price exists, so a different call is a change of interpretation on identical evidence and must be labelled that way. If there is no prior stance above, write exactly "no prior report on record">
REASSESS_TO_BUY: <one concrete single-line trigger with a rounded level from the data, e.g. "close above 50-day SMA (~$X.XX) on >1.2x avg volume">
MOVE_TO_SELL: <one concrete single-line trigger with a rounded level from the data, e.g. "close below 20-day support (~$X.XX)">
- <the single strongest reason for this call>
- <the strongest opposing evidence, and why it does not flip the call>
- <what the next session must show for the thesis to stay alive>

Then the analysis, as sections in this order. Formatting rules:
- Section headings: "### <n>. <Name>: <one-line takeaway>" (the takeaway IS the
  interpretation, e.g. "### 1. Technicals: bearish trend, but stretched").
- END each section with one line: "**Read:** <what this means for the trade>".
  The Read line must be usable ONLY for {ticker}: anchor it to at least one of
  this symbol's own values from the blocks (a level, an indicator reading, a
  named article, a peer gap). If the same sentence would still be true with a
  different ticker pasted in, it is wrong. Rewrite it. In particular, never
  write a takeaway of the shape "the market provides a supportive/unsupportive
  backdrop, but <indicator> limits confidence"; that sentence describes nothing
  about this company.
- Interpret, don't compute: cite numbers verbatim from the blocks; use the
  precomputed distances/R:R rather than doing arithmetic yourself.
- Plain markdown only (###, **bold**, - bullets). No HTML. Keep each section tight.
  The value is the takeaway and the Read line, not exhaustive narration.
- Format large figures human-readably, "$4.69B market cap", "$1.40B revenue",
  "1.52M shares": never paste raw unformatted values like "4694163968".
  Rounding for readability is not estimating; the underlying digits must come
  from the data.

1. Situation & Key Figures: what kind of situation {ticker} is in and the one
   question that decides the next 1-5 sessions. From the SITUATION &
   INVESTIGATION block: the deal or proceeding and its terms (offer price,
   spread, consideration, approvals and their status), the dated milestones,
   and the decisive actors with what the sourced record says about them,
   every fact with its src and date. If the block says a finding is an
   inference, say so. If no situation block was gathered, say exactly that in
   one sentence and classify the situation yourself from the news block. For
   MOMENTUM_ONLY, one short paragraph saying nothing situational is in play.
2. Technicals ({ticker}): cover price vs ALL THREE SMAs (20/50/200, the
   200SMA anchors the long-term trend and must not be skipped), RSI, MACD,
   volume, and volatility. In a PENDING_ACQUISITION, state every level relative
   to the offer price as well.
3. News & Catalysts: company-specific first, then sector, then macro. Every
   claim carries its outlet and date inline; is anything actually new?
4. Fundamentals: valuation and quality, only as they bear on the 1-5 day window
   (in a PENDING_ACQUISITION, only as they bear on completion or break value)
5. Peer Comparison: {ticker} vs the peer set in the data; company-specific move
   or sector-wide repricing? (omit this section only if no peer block was provided)
6. Business Context: what the company actually does, and which of tomorrow's
   drivers (sector beta, own catalysts, liquidity) dominate for a name this size
7. Market & Sector Backdrop: SPY regime and {sector_etf} versus SPY, in AT MOST
   three sentences. This context is identical for every symbol analysed today, so
   it earns no more space than that; spend the words on what it changes for
   {ticker} specifically
8. Bull vs Bear: the debate, not a summary. First "**Bull:**" with the 2-3
   strongest arguments FOR upside, each anchored to a specific number or article
   in the blocks; then "**Bear:**" with the 2-3 strongest arguments for downside,
   same standard. Argue each side at full strength. Do not soften the side you
   disagree with. The Read line states which side wins over 1-5 sessions and on
   what evidence the loser's case would take over.
9. Risk: systematic / sector / idiosyncratic; name the single biggest risk to
   THIS call and the falsification conditions that would flip it. Name any
   evidence this run could not gather and what it would have changed.
10. Trade Plan: stance; the key levels (support, resistance, SMAs) rounded as
   described above and stamped "as of {date}"; invalidation (which close, level
   or event kills the thesis); what to watch next session. If a prior stance is
   shown above, this section must also say in one sentence what changed since
   it, or that nothing did
"""


# Appended to the prompt (not part of the format template, the JSON braces
# would need escaping there). One extra fenced block makes the research call
# also the machine-readable analysis: stance for the UI banner, watch items,
# and news-vs-technicals alignment, in the same voice as the report itself.
EPILOGUE_INSTRUCTIONS = """
== STRUCTURED EPILOGUE (mandatory) ==

After the last numbered section, END your response with exactly one fenced JSON block, valid
JSON, double quotes, no comments, and NOTHING after the closing fence:

```json
{"stance": "BULLISH|CAUTIOUS_BULLISH|NEUTRAL|CAUTIOUS_BEARISH|BEARISH",
 "sentiment_alignment": "<one sentence: does the news sentiment confirm or conflict with the technical picture, and which should the reader weight here>",
 "watch_items": ["<2-3 short, concrete, checkable items. A level, a date, a metric>"]%(thesis)s}
```

The stance MUST be consistent with your Verdict: BUY maps to BULLISH
(CAUTIOUS_BULLISH if CONVICTION < 0.6), SELL to BEARISH (CAUTIOUS_BEARISH if
CONVICTION < 0.6), HOLD to NEUTRAL. watch_items must reuse levels/dates already
cited in your report. Do not introduce new numbers here.
Text inside the JSON follows the same voice rules as the report: no em dashes,
no "not just X, it's Y", no filler.
"""

THESIS_EPILOGUE_SCHEMA = (
    ',\n "company_thesis": {"perception": "<2-3 sentences: how the market currently'
    ' perceives this company (from the news) vs what it claims to be (from the'
    ' profile)>", "goal_alignment": "<1-2 sentences: do its recent activities align'
    ' with its stated goals?>", "positive_catalysts": ["<2-4 concrete events that'
    ' would move the stock UP>"], "negative_catalysts": ["<2-4 concrete events that'
    ' would move the stock DOWN>"], "regime_risks": "<1-2 sentences on'
    ' systematic/regime risks (rates, rotation, regulation)>"}'
)

VALID_STANCES = {"BULLISH", "CAUTIOUS_BULLISH", "NEUTRAL",
                 "CAUTIOUS_BEARISH", "BEARISH"}


# Sector ETF resolution lives in models.sector_map: the symbol's OWN sector
# metadata mapped sector-name -> SPDR ETF (no hardcoded ticker table, that
# silently mislabeled anything it didn't know, e.g. UNFI -> XLK).
from models.sector_map import get_sector_info


# In-process caches so a walk-forward backtest doesn't re-fetch the same
# full-history SPY/sector frames (and slow yfinance .info) on every date. The
# cached frame is full-history; the per-date lookahead slice is still applied
# by the caller, so caching does not leak future data.
_FRAME_CACHE: dict[str, "pd.DataFrame"] = {}
_FUND_CACHE: dict[str, tuple] = {}


def _cached_frame(symbol: str, period: str = "2y") -> "pd.DataFrame":
    key = f"{symbol}:{period}"
    if key not in _FRAME_CACHE:
        from services.stock_data import fetch_stock_data
        _FRAME_CACHE[key] = fetch_stock_data(symbol, period=period)
    return _FRAME_CACHE[key]


def _smart_truncate(data: str, max_chars: int) -> str:
    """Truncate at a line boundary instead of mid-line/mid-number."""
    if len(data) <= max_chars:
        return data
    cut = data[:max_chars].rfind("\n")
    if cut > max_chars * 0.5:
        return data[:cut] + "\n... [truncated]"
    return data[:max_chars] + "\n... [truncated]"


def _assert_ohlcv_not_stale(df: pd.DataFrame, as_of: str, symbol: str) -> None:
    """Raise if the latest row is > MAX_OHLCV_STALE_DAYS before as_of.

    Guards against a vendor returning a present-but-year-old partial frame that
    passes an empty-check and silently feeds a wrong price to the model.
    """
    if df is None or df.empty:
        return
    requested = pd.to_datetime(as_of, errors="coerce")
    if pd.isna(requested):
        return
    latest = pd.to_datetime(df.index.max())
    stale_days = (requested.normalize() - latest.normalize()).days
    if stale_days > MAX_OHLCV_STALE_DAYS:
        raise ValueError(
            f"{symbol}: latest OHLCV row is {latest.date()}, {stale_days} days "
            f"before requested {requested.date()} (stale): refusing to use it"
        )


def _as_of_slice(df: pd.DataFrame, as_of: str) -> pd.DataFrame:
    """Keep only rows on or before as_of (lookahead guard, defensive)."""
    if df is None or df.empty:
        return df
    cutoff = pd.to_datetime(as_of)
    return df[pd.to_datetime(df.index) <= cutoff]


def _fmt(x: Optional[float], nd: int = 2) -> str:
    return f"{x:.{nd}f}" if x is not None and pd.notna(x) else "n/a"


def _sf(x) -> Optional[float]:
    """Coerce a possibly-NA/NaN pandas value to a plain float or None."""
    try:
        return float(x) if pd.notna(x) else None
    except (TypeError, ValueError):
        return None


def _technicals(df: pd.DataFrame) -> dict:
    """Compute a compact, self-contained technicals summary from a price frame."""
    if df is None or len(df) < 2:
        return {}
    close = df["Close"].astype(float)
    high = df["High"].astype(float) if "High" in df else close
    low = df["Low"].astype(float) if "Low" in df else close
    vol = df["Volume"].astype(float) if "Volume" in df else None
    last = float(close.iloc[-1])

    def sma(n):
        return _sf(close.rolling(n).mean().iloc[-1]) if len(close) >= n else None

    # RSI(14)/ATR(14): shared true-Wilder helpers, the same math as the ta
    # library the ML features use, so the LLM and the models see one number.
    # (The previous hand-rolled RSI was a simple rolling mean claiming to be
    # Wilder's; it sat ~9 points below the standard value.)
    from utils.metrics import wilder_atr_series, wilder_rsi_series
    rsi = _sf(wilder_rsi_series(close).iloc[-1]) if len(close) >= 15 else None
    atr = _sf(wilder_atr_series(high, low, close).iloc[-1]) if len(df) >= 14 else None

    # MACD (12/26) + signal(9). Needs enough bars to mean anything; None
    # (rendered "n/a"), never a fabricated 0.0 that reads as a real value.
    macd_last = macd_sig_last = None
    if len(close) >= 26:
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        macd_sig = macd.ewm(span=9, adjust=False).mean()
        macd_last = _sf(macd.iloc[-1])
        macd_sig_last = _sf(macd_sig.iloc[-1])

    # 20d realized vol (annualized %)
    rets = close.pct_change().dropna()
    rvol = _sf(rets.tail(20).std() * (252 ** 0.5) * 100) if len(rets) >= 20 else None

    # Trailing high from intraday highs (a "high" from closes understates it);
    # only a full 252-bar window may be called 52-week.
    hi_basis = high.tail(252)
    hi_period = _sf(hi_basis.max())
    from_high = (last - hi_period) / hi_period * 100 if hi_period else None

    avg_vol = _sf(vol.tail(20).mean()) if vol is not None and len(vol) >= 20 else None
    last_vol = _sf(vol.iloc[-1]) if vol is not None else None

    return {
        "last": last, "sma20": sma(20), "sma50": sma(50), "sma200": sma(200),
        "rsi": rsi, "macd": macd_last, "macd_sig": macd_sig_last,
        "atr": atr, "rvol": rvol, "from_high_pct": from_high,
        "high_window_bars": int(len(hi_basis)),
        "vol_ratio": (last_vol / avg_vol) if avg_vol else None,
    }


def _technicals_block(name: str, df: pd.DataFrame) -> str:
    t = _technicals(df)
    if not t:
        return f"No data for {name}."
    trend = []
    # Percent distances precomputed, the LLM interprets, it must not do
    # this arithmetic itself.
    for key, label in (("sma20", "20SMA"), ("sma50", "50SMA"), ("sma200", "200SMA")):
        ref = t[key]
        if ref is not None and ref > 0:
            pct = (t["last"] / ref - 1) * 100
            trend.append(f"{pct:+.1f}% vs {label}")
    if t["macd"] is None or t["macd_sig"] is None:
        macd_state = "n/a"
    else:
        macd_state = "bullish" if t["macd"] > t["macd_sig"] else "bearish"
    bars = t.get("high_window_bars") or 0
    high_label = "52w high" if bars >= 252 else f"{bars}-bar high"
    return (
        f"Close: {_fmt(t['last'])} | 20SMA: {_fmt(t['sma20'])} | "
        f"50SMA: {_fmt(t['sma50'])} | 200SMA: {_fmt(t['sma200'])} ({', '.join(trend) or 'n/a'})\n"
        f"RSI(14): {_fmt(t['rsi'],1)} | MACD: {_fmt(t['macd'],3)} vs signal "
        f"{_fmt(t['macd_sig'],3)} ({macd_state}) | ATR(14): {_fmt(t['atr'])}\n"
        f"20d realized vol: {_fmt(t['rvol'],1)}% | from {high_label}: {_fmt(t['from_high_pct'],1)}% | "
        f"volume vs 20d avg: {_fmt(t['vol_ratio'],2)}x"
    )


def _price_action_block(df: pd.DataFrame, n: int = 15) -> str:
    if df is None or df.empty:
        return "No price data."
    tail = df.tail(n)
    lines = ["date        close     ret%    volume"]
    prev = None
    for idx, row in tail.iterrows():
        c = float(row["Close"])
        ret = f"{(c/prev-1)*100:+.2f}" if prev else "   -"
        v = int(row["Volume"]) if "Volume" in row and pd.notna(row["Volume"]) else 0
        lines.append(f"{str(idx)[:10]}  {c:8.2f}  {ret:>6}  {v:>12,}")
        prev = c
    return "\n".join(lines)


def _news_block(articles: list, n: Optional[int] = None) -> tuple[str, int, tuple]:
    """Point-in-time headlines, each carrying the outlet and date to cite.

    The prompt requires inline attribution on every news-derived claim, so the
    outlet has to be in the block the model reads, without ``src:`` here the
    only honest option left to the model is to drop the claim.

    Shows up to ``n`` (config NEWS_PROMPT_ARTICLES) articles SPREAD across
    the window, not the newest ``n``: newest-first truncation is how a
    30-day window used to reach the model as its last three days.
    """
    if not articles:
        return "No news in the point-in-time window.", 0, (None, None)
    if n is None:
        n = MODEL.NEWS_PROMPT_ARTICLES
    shown = select_spread(articles, n)
    oldest, newest = article_span(shown)
    lines = [f"({len(shown)} of {len(articles)} articles in the window, "
             f"sampled across {oldest} → {newest}, newest first)"]
    for a in shown:
        if hasattr(a, "title"):
            title = a.title or ""
            summary = (a.summary or "")[:200]
            sent = a.sentiment or "neutral"
            rel = a.ticker_relevance_score
            date = str(a.published_at)[:10] if a.published_at else "?"
            source = getattr(a, "source", None)
        else:
            title = a.get("title", "")
            summary = (a.get("summary") or "")[:200]
            sent = a.get("sentiment", "neutral")
            rel = a.get("ticker_relevance_score")
            date = str(a.get("published_at") or a.get("published_date") or "?")[:10]
            source = a.get("source")
        relf = f"{rel:.2f}" if isinstance(rel, (int, float)) else "?"
        # "unattributed" is deliberately not a plausible outlet name, the
        # prompt tells the model to drop claims it cannot attribute, and a
        # blank would have read as "no source needed".
        src = str(source).strip() if source else "unattributed"
        lines.append(f"- [{sent}] (src: {src} | {date} | rel:{relf}) {title}: {summary}")
    return "\n".join(lines), len(shown), (oldest, newest)


def _dollars(v) -> str:
    """1234567890 -> "$1.23B". The prompt forbids raw unformatted values, so
    the block must not contain them either."""
    if not isinstance(v, (int, float)) or pd.isna(v):
        return "n/a"
    sign = "-" if v < 0 else ""
    v = abs(v)
    for cut, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if v >= cut:
            return f"{sign}${v / cut:.2f}{suffix}"
    return f"{sign}${v:.0f}"


def _fundamentals_block(symbol: str, as_of: str) -> str:
    """Compact fundamentals via yfinance, filtered to fiscalDateEnding <= as_of.

    Best-effort: yfinance exposes fiscal-period-end columns, so we drop any
    statement column dated after as_of (the fundamentals look-ahead filter).
    Note: yfinance keys on period-end, not filing date, so a report can surface
    a few days early vs. when it was actually public. Acceptable for a 1-5 day
    horizon where fundamentals are a minor input.
    """
    try:
        import yfinance as yf
        if symbol not in _FUND_CACHE:
            tk = yf.Ticker(symbol)
            try:
                cashflow = tk.quarterly_cashflow
            except Exception:
                cashflow = None
            _FUND_CACHE[symbol] = (tk.info or {}, tk.quarterly_financials,
                                   cashflow)
        info, fin, cashflow = _FUND_CACHE[symbol]
        cutoff = pd.Timestamp(as_of)
        parts = []
        pe = info.get("trailingPE"); fpe = info.get("forwardPE")
        mcap = info.get("marketCap"); margin = info.get("profitMargins")
        if any(v is not None for v in (pe, fpe, mcap, margin)):
            parts.append(
                f"Trailing P/E: {_fmt(pe)} | Forward P/E: {_fmt(fpe)} | "
                f"Market cap: {_dollars(mcap)} | Profit margin: "
                f"{_fmt(margin*100,1) if isinstance(margin,(int,float)) else 'n/a'}%"
            )
        if fin is not None and not fin.empty:
            cols = sorted(c for c in fin.columns if pd.to_datetime(c) <= cutoff)
            if cols:
                latest = cols[-1]
                col = fin[latest]
                rev = col.get("Total Revenue"); ni = col.get("Net Income")
                parts.append(
                    f"Latest quarter (period ending {str(latest)[:10]}): "
                    f"Revenue {_dollars(rev)}, Net income {_dollars(ni)}"
                )
                # Year-over-year growth: the same fiscal quarter a year back,
                # matched within ~45 days so an off-cycle fiscal calendar
                # never pairs Q2 against Q3.
                target = pd.to_datetime(latest) - pd.DateOffset(years=1)
                yoy_col = min(cols[:-1], default=None,
                              key=lambda c: abs((pd.to_datetime(c) - target).days))
                if (yoy_col is not None
                        and abs((pd.to_datetime(yoy_col) - target).days) <= 45):
                    prev_rev = fin[yoy_col].get("Total Revenue")
                    prev_ni = fin[yoy_col].get("Net Income")
                    growth = []
                    if pd.notna(rev) and pd.notna(prev_rev) and prev_rev:
                        growth.append(
                            f"Revenue YoY: {(rev / prev_rev - 1) * 100:+.1f}%")
                    if pd.notna(ni) and pd.notna(prev_ni):
                        # A negative base makes the ratio meaningless, state
                        # the swing instead of a nonsense percentage.
                        if prev_ni > 0:
                            growth.append(
                                f"Net income YoY: {(ni / prev_ni - 1) * 100:+.1f}%")
                        elif ni > 0 >= prev_ni:
                            growth.append("Net income YoY: swung to profit "
                                          f"from {_dollars(prev_ni)} loss")
                    if growth:
                        growth.append(f"(vs quarter ending {str(yoy_col)[:10]})")
                        parts.append(" | ".join(growth))
        if cashflow is not None and not cashflow.empty:
            ccols = sorted(c for c in cashflow.columns
                           if pd.to_datetime(c) <= cutoff)
            if ccols:
                fcf = cashflow[ccols[-1]].get("Free Cash Flow")
                if pd.notna(fcf):
                    parts.append(f"Free cash flow (quarter ending "
                                 f"{str(ccols[-1])[:10]}): {_dollars(fcf)}")
        return "\n".join(parts) if parts else "Fundamentals not available."
    except Exception as e:
        logger.debug(f"fundamentals fetch failed for {symbol}: {e}")
        return "Fundamentals not available."


NO_TRACK_RECORD_LINE = (
    "No evaluated history was supplied for this run, so no hit rate can be "
    "stated. Treat the conviction below as an unproven estimate."
)

NO_PRIOR_REPORT_BLOCK = (
    "No earlier report for this symbol is on record. This is the first stance "
    "in the archive, so there is nothing to be consistent with yet."
)


def _price_since(df: pd.DataFrame, since: str) -> str:
    """What the price actually did between ``since`` and the last bar.

    The continuity block is only accountable if the model can check the prior
    report's triggers against real subsequent prices rather than being asked to
    remember them.
    """
    if df is None or df.empty:
        return "Price action since that report: not in data."
    idx = pd.to_datetime(df.index)
    span = df[idx >= pd.to_datetime(since)]
    if span.empty or len(span) < 2:
        # The prior report's own session is the only bar we have, no move to
        # report, and saying "flat" would be a fabrication.
        return (f"Price action since that report: no completed session between "
                f"{since} and the latest bar in this dataset.")
    close = span["Close"].astype(float)
    first, last = float(close.iloc[0]), float(close.iloc[-1])
    lo = float(span["Low"].astype(float).min()) if "Low" in span else float(close.min())
    hi = float(span["High"].astype(float).max()) if "High" in span else float(close.max())
    chg = (last / first - 1) * 100 if first else 0.0
    return (f"Price action since that report: close {first:.2f} on "
            f"{str(span.index[0])[:10]} to {last:.2f} on {str(span.index[-1])[:10]} "
            f"({chg:+.2f}% over {len(span) - 1} session"
            f"{'' if len(span) - 1 == 1 else 's'}); intraday low {lo:.2f}, "
            f"high {hi:.2f} across that span.")


def _continuity_block(prior: Optional[dict], df: pd.DataFrame,
                      as_of: str = "") -> str:
    """The previous stance on this symbol, its triggers, and what price did.

    Without this the model has no memory across days and a stance can flip
    overnight while every invalidation level it published went untouched, the
    defect this block exists to make visible.

    A predecessor written against the SAME data cutoff is called out as such:
    no new price exists between the two, so a different call there is a change
    of interpretation on identical evidence, which is the more damning case and
    the one the reader most needs named.
    """
    if not prior:
        return NO_PRIOR_REPORT_BLOCK

    text_ = prior.get("report_text") or ""
    stated = extract_confidence(text_) if text_ else None
    triggers = _extract_triggers(text_)
    weight = prior.get("confidence")
    prior_date = str(prior.get("trade_date") or "")[:10]
    same_cutoff = bool(as_of) and prior_date == str(as_of)[:10]

    lines = [
        (f"Previous report: trade date {prior_date or '?'}"
         + (" (SAME data cutoff as this run. An earlier run today)"
            if same_cutoff else "")
         + f", written by {prior.get('model_name') or 'an earlier run'}."),
        f"  Call: {(prior.get('decision') or '?').upper()}"
        + (f" | conviction stated then: {stated:.2f}" if stated is not None else "")
        + (f" | track-record weight carried then: {weight:.0%}"
           if isinstance(weight, (int, float)) else ""),
    ]
    if triggers.get("reassess_to_buy"):
        lines.append(f"  Trigger it published to turn bullish: "
                     f"{triggers['reassess_to_buy'][:220]}")
    if triggers.get("move_to_sell"):
        lines.append(f"  Trigger it published to turn bearish: "
                     f"{triggers['move_to_sell'][:220]}")
    if not triggers:
        lines.append("  It published no machine-readable trigger levels.")
    if same_cutoff:
        lines.append("  No new price data exists since that report, it saw "
                     "exactly the bars shown below. Any difference in your call "
                     "is a change of interpretation, not a response to news or "
                     "price.")
    else:
        lines.append("  " + _price_since(df, prior_date))
    return "\n".join(lines)


# ---- decision extraction (ported from the old signal_processing helpers) ----

_DECISION_RE = re.compile(
    r"FINAL TRANSACTION PROPOSAL:\s*\**\s*(BUY|SELL|HOLD)", re.IGNORECASE
)
# "CONVICTION" is the reader-facing label the prompt now asks for (the report
# body used to say CONFIDENCE inches from an unrelated reliability weight also
# rendered as a percentage). Both spellings are accepted so reports already in
# the archive keep parsing, and an optional parenthetical between the label and
# the colon is tolerated the same way the trigger regexes do it.
_CONF_RE = re.compile(
    r"\b(?:CONVICTION|CONFIDENCE)\s*(?:\([^)\n]*\))?\s*:\s*\**\s*([0-9]*\.?[0-9]+)",
    re.IGNORECASE,
)


def extract_decision(text: str) -> str:
    m = _DECISION_RE.search(text or "")
    if m:
        return m.group(1).upper()
    # Fallback: last standalone BUY/SELL/HOLD keyword. Loud on purpose --
    # prose like "...would move me to SELL" late in a report can silently
    # flip the recorded decision when the anchor line is missing.
    found = re.findall(r"\b(BUY|SELL|HOLD)\b", (text or "").upper())
    if found:
        logger.warning(
            "extract_decision: Verdict anchor missing, falling back to last "
            f"keyword ({found[-1]}); decision may be unreliable"
        )
        return found[-1]
    raise ValueError("Could not extract decision from response")


def extract_confidence(text: str) -> float:
    """The report's stated CONFIDENCE, or 0.5 when it cannot be read.

    The fallback is indistinguishable from a report that genuinely said 0.50,
    so it is logged, silently defaulting made a parse failure look like a
    real neutral call.
    """
    m = _CONF_RE.search(text or "")
    if m:
        try:
            return max(0.0, min(1.0, float(m.group(1))))
        except ValueError:
            logger.warning(
                "extract_confidence: CONFIDENCE anchor found but unparseable "
                f"({m.group(1)!r}): defaulting to 0.5"
            )
            return 0.5
    logger.warning(
        "extract_confidence: no CONFIDENCE anchor in report, defaulting to "
        "0.5; this is NOT a stated neutral call"
    )
    return 0.5


_EPILOGUE_RE = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)


def parse_epilogue(text: str) -> Optional[dict]:
    """Parse the structured JSON epilogue (last fenced json block), or None."""
    import json
    matches = _EPILOGUE_RE.findall(text or "")
    if not matches:
        return None
    try:
        parsed = json.loads(matches[-1])
        return parsed if isinstance(parsed, dict) else None
    except (json.JSONDecodeError, ValueError):
        return None


def strip_epilogue(text: str) -> str:
    """Remove the structured JSON epilogue fence for human-facing rendering.

    The epilogue is machine-read (banner stance, watch items); showing raw JSON
    in the report body is noise. Removes only the LAST fenced json block so any
    json the analysis itself quotes is untouched.
    """
    matches = list(_EPILOGUE_RE.finditer(text or ""))
    if not matches:
        return text or ""
    m = matches[-1]
    return (text[:m.start()].rstrip("\n") + "\n" + text[m.end():].lstrip("\n")).strip()


# Reader-facing names for the verdict fields, longest label first so
# "MEASURED ACCURACY" is matched before a bare "ACCURACY" ever could be.
_VERDICT_LABELS = [
    ("FINAL TRANSACTION PROPOSAL", "Final call"),
    ("MEASURED ACCURACY", "Measured accuracy"),
    ("ACCURACY TO DATE", "Measured accuracy"),
    ("SINCE LAST REPORT", "Since last report"),
    ("REASSESS_TO_BUY", "Reassess to BUY"),
    ("MOVE_TO_SELL", "Move to SELL"),
    ("CONVICTION", "Conviction (this report's own)"),
    ("CONFIDENCE", "Conviction (this report's own)"),
]

_LEADING_BOLD_RE = re.compile(r"^\*\*(.+?)\*\*")

_VERDICT_FIELD_RE = re.compile(
    r"^\s*(?:[-*+]\s*)?(?:\*\*)?\s*("
    + "|".join(re.escape(k) for k, _ in _VERDICT_LABELS)
    + r")\s*(?:\([^)\n]*\))?\s*(?:\*\*)?\s*:\s*(.*)$",
    re.IGNORECASE,
)


def render_report_markdown(text: str) -> str:
    """Report body prepared for a markdown renderer.

    Strips the machine-read epilogue, then rewrites the verdict block's field
    lines into list items. The model writes those fields one per line with a
    single newline between them, which every CommonMark renderer folds into one
    paragraph: so both the in-app modal and the PDF were emitting the whole
    verdict as a run-on sentence with "REASSESS_TO_BUY:" stranded mid-prose.
    List items survive that folding, and unlike trailing-whitespace hard breaks
    they cannot be silently eaten by a strip() somewhere in the pipeline. The
    summary bullets already below the fields keep their own markers, so the
    verdict reads as one bulleted card in both renderers.

    Applies only above the first "### " section heading: these labels are the
    verdict's, and a later mention in prose must not be rewritten. Runs on
    stored text too, so reports already in the archive render correctly.
    """
    body = strip_epilogue(text or "")
    head, sep, tail = body.partition("\n### ")

    out = []
    for line in head.split("\n"):
        m = _VERDICT_FIELD_RE.match(line)
        if not m:
            out.append(line)
            continue
        label = next(pretty for key, pretty in _VERDICT_LABELS
                     if key.upper() == m.group(1).upper())
        # The model bolds its own values ("**BUY**"); a bold label plus a bold
        # value reads as one undifferentiated run, so the leading bold span is
        # unwrapped. A plain strip("*") cannot do this. The closing "**" sits
        # mid-string on lines like "**0.56**: this report's own probability".
        value = _LEADING_BOLD_RE.sub(r"\1", m.group(2).strip()).strip()
        out.append(f"- **{label}:** {value}" if value else f"- **{label}**")
    return "\n".join(out) + (sep + tail if sep else "")


def derive_stance(decision: str, confidence: float) -> str:
    """Deterministic decision -> 5-level stance mapping (epilogue fallback)."""
    d = (decision or "HOLD").upper()
    if d == "BUY":
        return "BULLISH" if (confidence or 0) >= 0.6 else "CAUTIOUS_BULLISH"
    if d == "SELL":
        return "BEARISH" if (confidence or 0) >= 0.6 else "CAUTIOUS_BEARISH"
    return "NEUTRAL"


_STANCE_DIR = {"BULLISH": "BUY", "CAUTIOUS_BULLISH": "BUY", "NEUTRAL": "HOLD",
               "CAUTIOUS_BEARISH": "SELL", "BEARISH": "SELL"}


def _extract_triggers(text: str) -> dict:
    # Tolerate a parenthetical between the label and the colon, models
    # write e.g. "MOVE_TO_SELL (add to short / confirm): ..." and the strict
    # form silently dropped the trigger.
    triggers = {}
    m = re.search(r"REASSESS_TO_BUY[^:\n]*:\s*(.+)", text or "")
    if m:
        triggers["reassess_to_buy"] = m.group(1).strip().strip("*<>")
    m = re.search(r"MOVE_TO_SELL[^:\n]*:\s*(.+)", text or "")
    if m:
        triggers["move_to_sell"] = m.group(1).strip().strip("*<>")
    return triggers


class SingleAgentResearch:
    """Single-LLM-call research agent. Fully self-contained (no TradingAgents)."""

    # 6000 (was 4000): the structured report adds per-section Read lines and
    # two sections; truncation used to cut the decision footer silently.
    # The Verdict block now leads, but headroom keeps sections whole.
    def __init__(self, model: Optional[str] = None, provider: Optional[str] = None,
                 max_tokens: int = 6000):
        self.model = model or MODEL.TRADING_AGENTS_MODEL
        # Provider follows the model unless explicitly overridden, a gpt-*
        # research model must route to OpenAI (the old hardcoded "anthropic"
        # default made every gpt-* selection fail with a provider mismatch).
        self.provider = provider or (
            "openai" if self.model.startswith("gpt-") else "anthropic")
        self.max_tokens = max_tokens

    def analyze(
        self,
        symbol: str,
        as_of: str,
        ohlcv_df: Optional[pd.DataFrame] = None,
        news: Optional[list] = None,
        extra_context: str = "",
        news_lookback_days: Optional[int] = None,
        use_news: bool = True,
        include_thesis: bool = False,
        track_record: Optional[str] = None,
        use_continuity: bool = True,
        ledger: "Optional[EvidenceLedger]" = None,
        situation: Optional[str] = None,
    ) -> dict[str, Any]:
        """Run the analysis for `symbol` as of `as_of` (YYYY-MM-DD).

        `ohlcv_df` (already truncated by the caller in backtests) is used if
        given, else fetched. `news` may be pre-fetched (it is re-filtered to the
        window); if None and `use_news`, point-in-time news is fetched. Set
        `use_news=False` for the news-ablation backtest arm.

        `track_record` is the one literal sentence about measured accuracy that
        the report must quote verbatim (see
        :func:`services.calibration_service.track_record_sentence`). It is
        passed in rather than looked up here so a walk-forward backtest cannot
        quote a hit rate measured on sessions it has not reached; when omitted
        the report states that no rate can be given. `use_continuity=False`
        suppresses the prior-stance lookup for the same reason.
        """
        from services.stock_data import fetch_stock_data

        # Every block records itself as present or as a gap; a required block
        # that cannot be built raises here and the report is not written.
        ledger = ledger or EvidenceLedger(symbol)

        # Stamped before anything is fetched: the continuity lookup uses it to
        # exclude reports written by this very run (a retry, or a concurrent
        # worker on the same symbol), which is the only thing trade_date can
        # no longer do now that the bound is inclusive.
        started_at = datetime.now(timezone.utc)

        # --- ticker OHLCV (caller-truncated in backtests) ---
        if ohlcv_df is None:
            ohlcv_df = _as_of_slice(fetch_stock_data(symbol, period="1y"), as_of)
        else:
            ohlcv_df = _as_of_slice(ohlcv_df, as_of)
        _assert_ohlcv_not_stale(ohlcv_df, as_of, symbol)

        # --- broad market + sector context (lookahead-sliced) ---
        # Context ETF comes from the symbol's own industry/sector metadata:
        # industry ETF when mapped (tighter cohort), sector SPDR otherwise.
        # The proxy level is stated in the block so the report is explicit
        # about what it is comparing against. SPY as the resolved ETF means
        # the metadata is unknown, say so instead of presenting a duplicate
        # SPY block as sector evidence.
        sinfo = get_sector_info(symbol)
        sector_etf = sinfo["etf"]
        sector_name = sinfo["sector"]
        industry_name = sinfo["industry"]
        proxy_level = sinfo["level"]
        try:
            spy_block = _technicals_block("SPY", _as_of_slice(_cached_frame("SPY"), as_of))
            ledger.have("spy")
        except Exception as e:
            spy_block = "SPY data unavailable."
            ledger.missing("spy", f"SPY technicals failed: {str(e)[:80]}")
        if proxy_level == "unknown":
            sector_block = (f"No distinct sector ETF resolved for {symbol} "
                            f"(sector metadata unavailable), rely on the SPY "
                            f"market context; treat sector evidence as missing.")
            ledger.missing("sector", "sector metadata unavailable")
        else:
            if proxy_level == "industry":
                proxy_line = (f"Sector: {sector_name} | ETF proxy: {sector_etf} "
                              f"(industry-level: {industry_name})")
            else:
                proxy_line = (f"Sector: {sector_name} | ETF proxy: {sector_etf} "
                              f"(sector-level; no industry ETF mapped for "
                              f"'{industry_name}')")
            try:
                sector_block = proxy_line + "\n" + _technicals_block(
                    sector_etf, _as_of_slice(_cached_frame(sector_etf), as_of))
                ledger.have("sector")
            except Exception as e:
                sector_block = f"{sector_etf} data unavailable."
                ledger.missing("sector", f"{sector_etf} bars unavailable: {str(e)[:60]}")

        # --- news (point-in-time) ---
        # The window is the run's, never this module's: with no value there
        # is nothing honest to filter by, so refuse rather than guess.
        if use_news and news_lookback_days is None:
            raise RunParameterMissing(
                f"{symbol}: research agent called without news_lookback_days")
        if not use_news:
            news_articles = []
        elif news is not None:
            news_articles = filter_articles_as_of(news, as_of, news_lookback_days)
        else:
            try:
                news_articles = fetch_point_in_time_news(
                    symbol, as_of, lookback_days=news_lookback_days
                )
            except Exception as e:
                # The source failed, not a quiet week. Required evidence:
                # the report is not written blind (raises).
                logger.warning(f"PIT news fetch failed for {symbol}: {e}")
                ledger.missing("news_source", f"point-in-time fetch failed: {str(e)[:80]}")
                news_articles = []

        # Static business identity (sector/industry/summary), same rationale
        # as _fundamentals_block's info usage: identity is not price data, so
        # it is acceptable in a historical run.
        from services.stock_data import get_company_profile
        business = get_company_profile(symbol)
        if business:
            ledger.have("business")
        else:
            business = "Business profile not available."
            ledger.missing("business", "company profile lookup failed")

        # --- prior stance (cross-day continuity) ---
        # trade_date <= as_of is the lookahead guard (a later session's report
        # is never visible); `started_at` excludes anything written after this
        # call began, which is what lets a same-cutoff re-run see this
        # morning's stance instead of reporting "no prior report on record".
        # A missing/failed lookup degrades to "first stance on record" rather
        # than taking the report down.
        prior_report = None
        if use_continuity:
            try:
                from services.cache_service import get_cache
                prior_report = get_cache().get_prior_trading_agent_report(
                    symbol, as_of, generated_before=started_at)
            except Exception as e:
                logger.debug(f"prior report lookup failed for {symbol}: {e}")
        continuity = _continuity_block(prior_report, ohlcv_df, as_of)
        if prior_report:
            ledger.have("continuity")

        fundamentals_block = _fundamentals_block(symbol, as_of)
        fund_ok = not fundamentals_block.startswith("Fundamentals not available")
        if fund_ok:
            ledger.have("fundamentals")
        else:
            ledger.missing("fundamentals", "no filings on or before the as-of date")

        extra_block = ""
        # 12000: the assembled blocks now open with the situation &
        # investigation block and can carry political flows too; callers
        # truncate per block, this is the whole-string backstop. The gaps
        # block is appended LAST and never truncated away.
        gaps_block = ledger.prompt_block()
        if extra_context or gaps_block:
            extra_block = (
                "\n== PRECOMPUTED METRICS & EVENTS (validated. Prefer these numbers) ==\n"
                + _smart_truncate(extra_context, 12000)
                + (("\n\n" + gaps_block) if gaps_block else "")
                + "\n"
            )
        if situation:
            situation_line = (f"the investigation stage classified {symbol} as "
                              f"{situation} (see the SITUATION & INVESTIGATION block).")
        else:
            situation_line = (f"no situation block was gathered for {symbol} in this "
                              f"run: classify the situation yourself from the news "
                              f"and filings blocks before Step 1, and say that you did.")
        # Rendered once, measured once: the footer reports THIS count and
        # span. The char budget scales with the article budget so the tail
        # truncation (which would eat the oldest strata. The whole point of
        # spreading) cannot fire on a normal block; ~320 chars per line.
        news_block_text, news_shown, news_shown_span = _news_block(news_articles)
        news_block_text = _smart_truncate(
            news_block_text, max(6000, 320 * (MODEL.NEWS_PROMPT_ARTICLES or 50) + 400))
        prompt = SINGLE_AGENT_PROMPT.format(
            ticker=symbol,
            date=as_of,
            sector_etf=sector_etf,
            situation_line=situation_line,
            track_record_block=track_record or NO_TRACK_RECORD_LINE,
            continuity_block=_smart_truncate(continuity, 1500),
            business_block=_smart_truncate(business, 1200),
            spy_block=_smart_truncate(spy_block, 2000),
            sector_block=_smart_truncate(sector_block, 2000),
            price_block=_smart_truncate(_price_action_block(ohlcv_df), 3000),
            tech_block=_smart_truncate(_technicals_block(symbol, ohlcv_df), 2000),
            fundamentals_block=_smart_truncate(fundamentals_block, 2000),
            news_block=news_block_text,
            extra_context=extra_block,
        ) + EPILOGUE_INSTRUCTIONS % {
            "thesis": THESIS_EPILOGUE_SCHEMA if include_thesis else "",
        }

        from services.llm_service import get_llm
        llm = get_llm()

        # Deterministic post-generation validation: the Verdict block and the
        # structured epilogue are the machine-readable parts of the report. If
        # either is missing/unparseable the decision would fall through to
        # heuristics that can silently mislabel. One retry is cheap insurance.
        # Reasoning models bill reasoning as output tokens; reasoning_effort
        # routes generate() through its max_completion_tokens headroom logic
        # so the report (and its epilogue) actually completes.
        gen_kwargs: dict = {}
        if self.provider == "openai" and self.model.startswith("gpt-"):
            gen_kwargs["reasoning_effort"] = "medium"

        # Token cost ACCUMULATES across attempts: a retried report really did
        # cost both calls, and billing the caller only for the surviving one
        # understates it by up to half.
        usage_total = {"input_tokens": 0, "output_tokens": 0,
                       "model": self.model or "", "provider": self.provider or ""}
        raw_text = None
        for attempt in (1, 2):
            attempt_usage: dict = {}
            raw_text = llm.generate(
                prompt,
                max_tokens=self.max_tokens,
                temperature=0.3,
                model=self.model,
                provider=self.provider,
                usage_out=attempt_usage,
                **gen_kwargs,
            )
            if attempt_usage:
                usage_total["input_tokens"] += attempt_usage.get("input_tokens", 0)
                usage_total["output_tokens"] += attempt_usage.get("output_tokens", 0)
                # Record what actually served the call. A failover may have
                # moved it off the requested model.
                usage_total["model"] = attempt_usage.get("model") or usage_total["model"]
                usage_total["provider"] = (attempt_usage.get("provider")
                                           or usage_total["provider"])
            if not raw_text:
                raise ValueError("LLM returned empty response")
            if (_DECISION_RE.search(raw_text) and _CONF_RE.search(raw_text)
                    and parse_epilogue(raw_text)):
                break
            logger.warning(
                f"{symbol}: report missing Verdict anchors or epilogue "
                f"(attempt {attempt}): {'retrying' if attempt == 1 else 'using fallbacks'}"
            )

        decision = extract_decision(raw_text)
        confidence = extract_confidence(raw_text)
        triggers = _extract_triggers(raw_text)

        # Structured epilogue: the model's own stance/watch fields, with a
        # deterministic fallback so downstream consumers ALWAYS get them. A
        # stance that contradicts the Verdict is overridden (the Verdict is the
        # scored call) and the adjustment is flagged rather than hidden.
        structured = parse_epilogue(raw_text)
        epilogue_source = "model"
        if structured:
            stance = str(structured.get("stance", "")).upper().replace(" ", "_")
            if stance not in VALID_STANCES:
                structured["stance"] = derive_stance(decision, confidence)
                structured["stance_adjusted"] = True
            elif _STANCE_DIR[stance] != decision:
                structured["stance"] = derive_stance(decision, confidence)
                structured["stance_adjusted"] = True
            else:
                structured["stance"] = stance
            if not isinstance(structured.get("watch_items"), list):
                structured["watch_items"] = [v for v in triggers.values() if v]
        else:
            epilogue_source = "derived"
            structured = {
                "stance": derive_stance(decision, confidence),
                "sentiment_alignment": "",
                "watch_items": [v for v in triggers.values() if v],
            }
        structured["source"] = epilogue_source

        # Provenance: computed from what was ACTUALLY assembled, never
        # asserted by the LLM. The footer travels inside report_text so every
        # surface (UI, History, PDF, downloads) carries its own audit trail.
        provenance = {
            "model": self.model,
            "provider": self.provider,
            "as_of": as_of,
            "news_count": len(news_articles),
            "news_window_days": news_lookback_days,
            "news_span": list(article_span(news_articles)),
            "news_prompt_articles": news_shown,
            "news_prompt_span": list(news_shown_span),
            "news_enabled": use_news,
            "ohlcv_bars": int(len(ohlcv_df)) if ohlcv_df is not None else 0,
            "ohlcv_through": (str(ohlcv_df.index.max())[:10]
                              if ohlcv_df is not None and len(ohlcv_df) else None),
            "spy_context": not spy_block.startswith("SPY data unavailable"),
            "sector_etf": sector_etf,
            "sector": sector_name,
            "industry": industry_name,
            "sector_proxy_level": proxy_level,
            "fundamentals": fund_ok,
            "precomputed_blocks": bool(extra_block),
            "epilogue_source": epilogue_source,
            "track_record_stated": bool(track_record),
            "prior_report_date": (prior_report or {}).get("trade_date"),
            "prior_decision": (prior_report or {}).get("decision"),
            "situation": situation,
            "evidence": ledger.to_dict(),
        }
        news_desc = (
            f"{news_shown} of {len(news_articles)} news articles read "
            f"({news_lookback_days}d point-in-time window; read span "
            f"{news_shown_span[0]} → {news_shown_span[1]})"
            if use_news and news_articles else
            "no news in the window" if use_news else
            "news disabled for this run"
        )
        if proxy_level == "industry":
            sector_src = f"SPY & {sector_etf} ({industry_name} industry proxy) context"
        elif proxy_level == "sector":
            sector_src = f"SPY & {sector_etf} ({sector_name} sector proxy) context"
        else:
            sector_src = "SPY context (sector unresolved)"
        footer = (
            f"\n\n---\n*Compiled by {self.model} ({self.provider}) · as-of {as_of}"
            f" · Sources: {provenance['ohlcv_bars']}-bar OHLCV through "
            f"{provenance['ohlcv_through']}; {sector_src}; "
            f"fundamentals {'(as-of filtered)' if fund_ok else 'unavailable'}; "
            f"{news_desc}"
            f"{'; precomputed metrics/events/peers' if extra_block else ''}"
            + (f"; prior stance from the {prior_report['trade_date']} report"
               if prior_report else "; no prior report on record")
            + (f"; situation {situation}" if situation else "")
            + "."
            + (f" **Written without expected evidence: {ledger.summary()}.**"
               if ledger.degraded else "")
            + " Every figure above comes from these blocks, flagged 'not in"
              " data' where missing.*"
        )

        # Deterministic figure audit: every number the report cites should
        # exist in the prompt it was generated from. The prompt IS the source
        # of record here, auditing against it needs no extra fetch and can't
        # drift from what the model actually saw. High unmatched ratios mean
        # invention; a few stragglers are usually the model's own arithmetic.
        figure_check = None
        try:
            from utils.figure_check import check_figures
            # The stated conviction is the model's own judgement, so it exists
            # in no data block by construction. Auditing it guaranteed one
            # false positive on every report.
            fc = check_figures(raw_text, prompt, ignore_values=(confidence,))
            figure_check = {
                "checked": fc.checked,
                "unmatched": fc.unmatched[:12],
                "grounded_ratio": round(fc.grounded_ratio, 3),
            }
            if fc.checked >= 10 and fc.grounded_ratio < 0.8:
                logger.warning(
                    f"{symbol}: report cites {len(fc.unmatched)} of "
                    f"{fc.checked} figures with no source in its prompt: "
                    f"{fc.unmatched[:6]}")
                from services import progress_service as _prog
                _prog.emit("error",
                           f"{symbol}: {len(fc.unmatched)}/{fc.checked} report "
                           f"figures lack a source in the data shown to the "
                           f"model: treat unverified numbers as suspect")
        except Exception as e:
            logger.debug(f"figure check skipped: {e}")

        return {
            "decision": decision,
            "confidence": confidence,
            "raw_response": raw_text + footer,
            "model": self.model,
            "triggers": triggers,
            "structured": structured,
            "provenance": provenance,
            "figure_check": figure_check,
            "news_count": len(news_articles),
            "sector_etf": sector_etf,
            "situation": situation,
            "evidence": ledger.to_dict(),
            "input_tokens": usage_total["input_tokens"],
            "output_tokens": usage_total["output_tokens"],
            "served_by_model": usage_total["model"],
        }

"""Self-contained single-agent research model — decoupled from TradingAgents.

This is a native quant-news reimplementation of the single-LLM-call research
agent we previously imported from `tradingagents.baselines.single_agent`. That
import broke when the upstream fork's `main` advanced to v0.3.1 (the `baselines`
submodule lives only on a feature branch, and its pre-v0.3.0 internal API —
`tradingagents.llm`, `agent_utils.normalize_content`, the flat `get_*` tools,
`signal_processing.extract_*` — was all moved/renamed/deleted upstream).

Design goals (per the incorporation analysis, §6 Tier 0-C):
  * Robust / swappable: implements the `BaseModel` contract, so the research
    strategy is one interchangeable implementation behind a stable seam. No
    dependency on any TradingAgents internal — an upstream release can never
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
from datetime import datetime, timedelta
from typing import Any, Optional

import pandas as pd

from config import MODEL
from services.news_window import (
    DEFAULT_NEWS_LOOKBACK_DAYS,
    fetch_point_in_time_news,
    filter_articles_as_of,
)

logger = logging.getLogger(__name__)

# Reject an OHLCV frame whose latest row is more than this many calendar days
# before the requested date — catches the "present but wrong" year-old frame a
# simple empty-check misses (ported from TradingAgents MAX_OHLCV_STALE_DAYS).
MAX_OHLCV_STALE_DAYS = 10


SINGLE_AGENT_PROMPT = """You are a trading analyst deciding, at the close of {date}, whether {ticker} will move UP or DOWN in the NEXT trading session.

IMPORTANT: Use ONLY information available through the close of {date}. Every data
block below is bounded to that date; do not reason about anything that happened
afterward. Your thesis window is 1-5 trading days — focus on catalysts and
momentum that play out within it. Ignore long-term (months/years) arguments.

DATA DISCIPLINE (mandatory):
- Every price, indicator value, or statistic you cite MUST come verbatim from the
  data blocks below. Never estimate, extrapolate, or invent a number. If a value
  you need is not in the data, write "not in data" instead of guessing.
- Treat the PRECOMPUTED METRICS / verified blocks as the source of truth. If two
  blocks conflict, FLAG the discrepancy rather than inventing a reconciled number.
  Do not claim historical support/resistance bounces or exact percentage moves
  unless a data block states them with concrete dates and prices.
- Do not prefix price LEVELS with +/- signs; signs belong on returns/changes only.
- A claim about a group (peers, sectors, models) must hold for EVERY member you
  name. If it doesn't, name only the members it holds for, or quantify exactly
  ("3 of 4 peers") — never stretch "all" or "much" over a member the numbers
  don't support.
- Use only the evidence in this prompt. You have no tools and cannot browse; if
  something is missing, say so explicitly rather than filling it in.

Analyze ALL of the following data carefully before deciding.

== {ticker} BUSINESS PROFILE ==
{business_block}

== BROAD MARKET CONTEXT (S&P 500 / SPY) ==
{spy_block}

== SECTOR CONTEXT ({sector_etf}) ==
{sector_block}

== {ticker} PRICE ACTION ==
{price_block}

== {ticker} TECHNICAL INDICATORS ==
{tech_block}

== {ticker} FUNDAMENTALS (as of {date}) ==
{fundamentals_block}

== {ticker} NEWS (through close of {date}) ==
{news_block}
{extra_context}
== DECISION FRAMEWORK ==

**Step 1: Market Regime (Systematic Risk)** — from the SPY block:
- SPY above 50 SMA AND 200 SMA with positive MACD = BULL (prior lean BUY)
- SPY below 50 SMA AND 200 SMA with negative MACD = BEAR (prior lean SELL)
- Mixed = NEUTRAL (no prior — decide on ticker-specific evidence)

**Step 1b: Sector Regime ({sector_etf})** — leading, lagging, or in line vs SPY.
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

**Step 4: Hard Rules (apply before finalizing)**
- EVENT GATE: if a PRECOMPUTED EVENTS block shows earnings or ex-dividend inside
  the hold window, cap CONFIDENCE at 0.6 and name the event in your risk section.
- DRAWDOWN CAUSALITY: if {ticker} is down >20% from its period high, state a causal
  hypothesis backed by the news/fundamentals blocks; if none is identifiable, write
  exactly "cause unknown — elevated risk" and treat it as bearish.
- COMPUTE, DON'T ASSERT: any risk/reward claim must use the support/resistance and
  ATR arithmetic from the PRECOMPUTED METRICS block (when present).

DECISIVENESS: commit to a clear BUY or SELL whenever the strongest evidence warrants
one. Reserve HOLD for when the evidence is genuinely balanced — do not hedge by default.

CONFIDENCE SEMANTICS: CONFIDENCE is your estimated probability (0.0-1.0) that the
direction of your call is correct over the next 1-5 sessions. 0.5 means no edge —
use it only when evidence is genuinely balanced, and prefer HOLD in that case.

== REQUIRED OUTPUT FORMAT (markdown) ==

You MUST BEGIN your response with exactly this block:

## Verdict
FINAL TRANSACTION PROPOSAL: **BUY** (or **SELL** or **HOLD**)
CONFIDENCE: **0.X**
REASSESS_TO_BUY: <one concrete single-line trigger with a level from the data, e.g. "close above 50-day SMA ($X) on >1.2x avg volume">
MOVE_TO_SELL: <one concrete single-line trigger with a level from the data, e.g. "close below 20-day support ($X)">
- <the single strongest reason for this call>
- <the strongest opposing evidence, and why it does not flip the call>
- <what the next session must show for the thesis to stay alive>

Then the analysis, as sections in this order. Formatting rules:
- Section headings: "### <n>. <Name> — <one-line takeaway>" (the takeaway IS the
  interpretation, e.g. "### 4. Technicals — bearish trend, but stretched").
- END each section with one line: "**Read:** <what this means for the trade>".
  Data above the Read line, judgment in it. Never restate numbers in the Read line.
- Interpret, don't compute: cite numbers verbatim from the blocks; use the
  precomputed distances/R:R rather than doing arithmetic yourself.
- Plain markdown only (###, **bold**, - bullets). No HTML. Keep each section tight —
  the value is the takeaway and the Read line, not exhaustive narration.
- Format large figures human-readably — "$4.69B market cap", "$1.40B revenue",
  "1.52M shares" — never paste raw unformatted values like "4694163968".
  Rounding for readability is not estimating; the underlying digits must come
  from the data.

1. Business Context — what the company actually does, and which of tomorrow's
   drivers (sector beta, own catalysts, liquidity) dominate for a name this size
2. Market Regime (SPY)
3. Sector ({sector_etf}) — leading or lagging SPY, and what that means for {ticker}
4. Technicals ({ticker}) — cover price vs ALL THREE SMAs (20/50/200 — the
   200SMA anchors the long-term trend and must not be skipped), RSI, MACD,
   volume, and volatility
5. Peer Comparison — {ticker} vs the peer set in the data; company-specific move
   or sector-wide repricing? (omit this section only if no peer block was provided)
6. Fundamentals — valuation and quality, only as they bear on the 1-5 day window
7. News & Catalysts — macro vs sector vs company-specific; is anything new?
8. Risk — systematic / sector / idiosyncratic; name the single biggest risk to
   THIS call and the falsification conditions that would flip it
9. Trade Plan — stance; the key levels verbatim (support, resistance, SMAs);
   invalidation (which close or level kills the thesis); what to watch next session
"""


# Appended to the prompt (not part of the format template — the JSON braces
# would need escaping there). One extra fenced block makes the research call
# also the machine-readable analysis: stance for the UI banner, watch items,
# and news-vs-technicals alignment, in the same voice as the report itself.
EPILOGUE_INSTRUCTIONS = """
== STRUCTURED EPILOGUE (mandatory) ==

After section 9, END your response with exactly one fenced JSON block — valid
JSON, double quotes, no comments, and NOTHING after the closing fence:

```json
{"stance": "BULLISH|CAUTIOUS_BULLISH|NEUTRAL|CAUTIOUS_BEARISH|BEARISH",
 "sentiment_alignment": "<one sentence: does the news sentiment confirm or conflict with the technical picture, and which should the reader weight here>",
 "watch_items": ["<2-3 short, concrete, checkable items — a level, a date, a metric>"]%(thesis)s}
```

The stance MUST be consistent with your Verdict: BUY maps to BULLISH
(CAUTIOUS_BULLISH if CONFIDENCE < 0.6), SELL to BEARISH (CAUTIOUS_BEARISH if
CONFIDENCE < 0.6), HOLD to NEUTRAL. watch_items must reuse levels/dates already
cited in your report — do not introduce new numbers here.
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
# metadata mapped sector-name -> SPDR ETF (no hardcoded ticker table — that
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


def clear_caches() -> None:
    """Drop the in-process frame/fundamentals caches (use between live runs)."""
    _FRAME_CACHE.clear()
    _FUND_CACHE.clear()


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
            f"before requested {requested.date()} (stale) — refusing to use it"
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

    # RSI(14)/ATR(14): shared true-Wilder helpers — the same math as the ta
    # library the ML features use, so the LLM and the models see one number.
    # (The previous hand-rolled RSI was a simple rolling mean claiming to be
    # Wilder's; it sat ~9 points below the standard value.)
    from utils.metrics import wilder_atr_series, wilder_rsi_series
    rsi = _sf(wilder_rsi_series(close).iloc[-1]) if len(close) >= 15 else None
    atr = _sf(wilder_atr_series(high, low, close).iloc[-1]) if len(df) >= 14 else None

    # MACD (12/26) + signal(9) — needs enough bars to mean anything; None
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
    # Percent distances precomputed — the LLM interprets, it must not do
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


def _news_block(articles: list, n: int = 20) -> str:
    if not articles:
        return "No news in the point-in-time window."
    lines = []
    for a in articles[:n]:
        if hasattr(a, "title"):
            title = a.title or ""
            summary = (a.summary or "")[:200]
            sent = a.sentiment or "neutral"
            rel = a.ticker_relevance_score
            date = str(a.published_at)[:10] if a.published_at else "?"
        else:
            title = a.get("title", "")
            summary = (a.get("summary") or "")[:200]
            sent = a.get("sentiment", "neutral")
            rel = a.get("ticker_relevance_score")
            date = str(a.get("published_at") or a.get("published_date") or "?")[:10]
        relf = f"{rel:.2f}" if isinstance(rel, (int, float)) else "?"
        lines.append(f"- [{sent}] ({date}, rel:{relf}) {title}: {summary}")
    return "\n".join(lines)


def _fundamentals_block(symbol: str, as_of: str) -> str:
    """Compact fundamentals via yfinance, filtered to fiscalDateEnding <= as_of.

    Best-effort: yfinance exposes fiscal-period-end columns, so we drop any
    statement column dated after as_of (the fundamentals look-ahead filter).
    Note: yfinance keys on period-end, not filing date, so a report can surface
    a few days early vs. when it was actually public — acceptable for a 1-5 day
    horizon where fundamentals are a minor input.
    """
    try:
        import yfinance as yf
        if symbol not in _FUND_CACHE:
            tk = yf.Ticker(symbol)
            _FUND_CACHE[symbol] = (tk.info or {}, tk.quarterly_financials)
        info, fin = _FUND_CACHE[symbol]
        cutoff = pd.Timestamp(as_of)
        parts = []
        pe = info.get("trailingPE"); fpe = info.get("forwardPE")
        mcap = info.get("marketCap"); margin = info.get("profitMargins")
        if any(v is not None for v in (pe, fpe, mcap, margin)):
            parts.append(
                f"Trailing P/E: {_fmt(pe)} | Forward P/E: {_fmt(fpe)} | "
                f"Market cap: {mcap or 'n/a'} | Profit margin: "
                f"{_fmt(margin*100,1) if isinstance(margin,(int,float)) else 'n/a'}%"
            )
        if fin is not None and not fin.empty:
            cols = [c for c in fin.columns if pd.to_datetime(c) <= cutoff]
            if cols:
                latest = sorted(cols)[-1]
                col = fin[latest]
                rev = col.get("Total Revenue"); ni = col.get("Net Income")
                parts.append(
                    f"Latest quarter (period ending {str(latest)[:10]}): "
                    f"Revenue {rev if pd.notna(rev) else 'n/a'}, "
                    f"Net income {ni if pd.notna(ni) else 'n/a'}"
                )
        return "\n".join(parts) if parts else "Fundamentals not available."
    except Exception as e:
        logger.debug(f"fundamentals fetch failed for {symbol}: {e}")
        return "Fundamentals not available."


# ---- decision extraction (ported from the old signal_processing helpers) ----

_DECISION_RE = re.compile(
    r"FINAL TRANSACTION PROPOSAL:\s*\**\s*(BUY|SELL|HOLD)", re.IGNORECASE
)
_CONF_RE = re.compile(r"CONFIDENCE:\s*\**\s*([0-9]*\.?[0-9]+)", re.IGNORECASE)


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
            "extract_decision: Verdict anchor missing — falling back to last "
            f"keyword ({found[-1]}); decision may be unreliable"
        )
        return found[-1]
    raise ValueError("Could not extract decision from response")


def extract_confidence(text: str) -> float:
    m = _CONF_RE.search(text or "")
    if m:
        try:
            return max(0.0, min(1.0, float(m.group(1))))
        except ValueError:
            pass
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
    # Tolerate a parenthetical between the label and the colon — models
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
        # Provider follows the model unless explicitly overridden — a gpt-*
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
        news_lookback_days: int = DEFAULT_NEWS_LOOKBACK_DAYS,
        use_news: bool = True,
        include_thesis: bool = False,
    ) -> dict[str, Any]:
        """Run the analysis for `symbol` as of `as_of` (YYYY-MM-DD).

        `ohlcv_df` (already truncated by the caller in backtests) is used if
        given, else fetched. `news` may be pre-fetched (it is re-filtered to the
        window); if None and `use_news`, point-in-time news is fetched. Set
        `use_news=False` for the news-ablation backtest arm.
        """
        from services.stock_data import fetch_stock_data

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
        # the metadata is unknown — say so instead of presenting a duplicate
        # SPY block as sector evidence.
        sinfo = get_sector_info(symbol)
        sector_etf = sinfo["etf"]
        sector_name = sinfo["sector"]
        industry_name = sinfo["industry"]
        proxy_level = sinfo["level"]
        try:
            spy_block = _technicals_block("SPY", _as_of_slice(_cached_frame("SPY"), as_of))
        except Exception:
            spy_block = "SPY data unavailable."
        if proxy_level == "unknown":
            sector_block = (f"No distinct sector ETF resolved for {symbol} "
                            f"(sector metadata unavailable) — rely on the SPY "
                            f"market context; treat sector evidence as missing.")
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
            except Exception:
                sector_block = f"{sector_etf} data unavailable."

        # --- news (point-in-time) ---
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
                logger.warning(f"PIT news fetch failed for {symbol}: {e}")
                news_articles = []

        extra_block = ""
        if extra_context:
            # 8000 (was 4000): the assembled blocks (metrics + events +
            # peers + SPY regime) exceeded the old budget, silently cutting
            # the later blocks. Callers truncate per block; this is a backstop.
            extra_block = (
                "\n== PRECOMPUTED METRICS & EVENTS (validated — prefer these numbers) ==\n"
                + _smart_truncate(extra_context, 8000) + "\n"
            )

        # Static business identity (sector/industry/summary) — same rationale
        # as _fundamentals_block's info usage: identity is not price data, so
        # it is acceptable in a historical run.
        from services.stock_data import get_company_profile
        business = get_company_profile(symbol) or "Business profile not available."

        fundamentals_block = _fundamentals_block(symbol, as_of)
        prompt = SINGLE_AGENT_PROMPT.format(
            ticker=symbol,
            date=as_of,
            sector_etf=sector_etf,
            business_block=_smart_truncate(business, 1200),
            spy_block=_smart_truncate(spy_block, 2000),
            sector_block=_smart_truncate(sector_block, 2000),
            price_block=_smart_truncate(_price_action_block(ohlcv_df), 3000),
            tech_block=_smart_truncate(_technicals_block(symbol, ohlcv_df), 2000),
            fundamentals_block=_smart_truncate(fundamentals_block, 2000),
            news_block=_smart_truncate(_news_block(news_articles), 6000),
            extra_context=extra_block,
        ) + EPILOGUE_INSTRUCTIONS % {
            "thesis": THESIS_EPILOGUE_SCHEMA if include_thesis else "",
        }

        from services.llm_service import get_llm
        llm = get_llm()

        # Deterministic post-generation validation: the Verdict block and the
        # structured epilogue are the machine-readable parts of the report. If
        # either is missing/unparseable the decision would fall through to
        # heuristics that can silently mislabel — one retry is cheap insurance.
        # Reasoning models bill reasoning as output tokens; reasoning_effort
        # routes generate() through its max_completion_tokens headroom logic
        # so the report (and its epilogue) actually completes.
        gen_kwargs: dict = {}
        if self.provider == "openai" and self.model.startswith("gpt-"):
            gen_kwargs["reasoning_effort"] = "medium"

        raw_text = None
        for attempt in (1, 2):
            raw_text = llm.generate(
                prompt,
                max_tokens=self.max_tokens,
                temperature=0.3,
                model=self.model,
                provider=self.provider,
                **gen_kwargs,
            )
            if not raw_text:
                raise ValueError("LLM returned empty response")
            if (_DECISION_RE.search(raw_text) and _CONF_RE.search(raw_text)
                    and parse_epilogue(raw_text)):
                break
            logger.warning(
                f"{symbol}: report missing Verdict anchors or epilogue "
                f"(attempt {attempt}) — {'retrying' if attempt == 1 else 'using fallbacks'}"
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

        # Provenance — computed from what was ACTUALLY assembled, never
        # asserted by the LLM. The footer travels inside report_text so every
        # surface (UI, History, PDF, downloads) carries its own audit trail.
        fund_ok = not fundamentals_block.startswith("Fundamentals not available")
        provenance = {
            "model": self.model,
            "provider": self.provider,
            "as_of": as_of,
            "news_count": len(news_articles),
            "news_window_days": news_lookback_days,
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
        }
        news_desc = (
            f"{len(news_articles)} news articles "
            f"({news_lookback_days}d point-in-time window)"
            if use_news else "news disabled for this run"
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
            f"{'; precomputed metrics/events/peers' if extra_block else ''}."
            f" Every figure above comes from these blocks — flagged 'not in data'"
            f" where missing.*"
        )

        return {
            "decision": decision,
            "confidence": confidence,
            "raw_response": raw_text + footer,
            "model": self.model,
            "triggers": triggers,
            "structured": structured,
            "provenance": provenance,
            "news_count": len(news_articles),
            "sector_etf": sector_etf,
        }

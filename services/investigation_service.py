"""Situation classification and open-web investigation for one symbol.

The research report used to be written from whatever blocks the pipeline
happened to hold, with the prompt telling the model it "has no tools and
cannot browse". For a stock under a cash takeover that meant a momentum
SELL written off moving averages while the only question that mattered —
will the deal close — never entered the prompt (BHF, 2026-09-01).

This stage runs BEFORE the report is written. One tool-using call:

  1. classifies the situation the symbol is in (pending acquisition,
     legal/regulatory overhang, earnings event, leadership change,
     distress, momentum only, …) from the supplied evidence;
  2. investigates it on the open web — deal terms, regulators and dated
     milestones, the key figures involved and what their record and
     affiliations imply for the outcome — citing every finding;
  3. returns structured JSON the report prompt and the synthesis read.

Point-in-time: web search cannot be bounded to a past date, so on a
backtest (``live=False``) the classifier runs WITHOUT tools and says so;
only live runs research the web. The investigator is told the decision
moment (before the target session's open) and to date every source.
"""

import json
import logging
import re
import threading
from dataclasses import dataclass, field
from typing import Optional

from config import MODEL

logger = logging.getLogger(__name__)

SITUATIONS = (
    "PENDING_ACQUISITION",       # definitive agreement to be acquired / merged
    "STRATEGIC_REVIEW",          # activist, sale process, spin, bid rumours
    "LEGAL_REGULATORY_OVERHANG",  # investigation, litigation, regulator action
    "EARNINGS_EVENT",            # results/guidance inside or just outside window
    "LEADERSHIP_CHANGE",         # CEO/CFO/CAO turnover, board changes
    "DISTRESS",                  # liquidity, covenant, going-concern, dilution
    "PRODUCT_OR_CONTRACT",       # approval, launch, award, loss of a customer
    "MOMENTUM_ONLY",             # nothing situational — flow and trend
    "OTHER",
)

_CACHE: dict[tuple, "Investigation"] = {}
_CACHE_LOCK = threading.Lock()
# One lock per cache key: a prefetch thread and the in-loop call for the
# same symbol must never both pay for the search — the second waits.
_KEY_LOCKS: dict[tuple, threading.Lock] = {}


@dataclass
class Investigation:
    symbol: str
    as_of: str
    web: bool
    situation: str = "OTHER"
    situation_confidence: str = "low"
    one_line: str = ""
    deal: dict = field(default_factory=dict)
    key_figures: list = field(default_factory=list)
    findings: list = field(default_factory=list)
    dated_events: list = field(default_factory=list)
    open_questions: list = field(default_factory=list)
    sources: list = field(default_factory=list)
    searches: int = 0
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    spread_pct: Optional[float] = None
    parse_ok: bool = True
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


def _system_prompt(web: bool) -> str:
    base = """You are an investigative equity research analyst preparing the situational brief that a trading desk reads BEFORE it looks at any chart. Your job is to establish what kind of situation the company is in, who the decisive actors are, and what is scheduled — with sources.

Discipline:
- Every finding carries a source: outlet or document, publication date, URL. A claim you cannot source does not go in "findings"; it may go in "open_questions".
- Separate fact from inference. A relationship between two people or firms is a finding only when a source states it; what it implies for the outcome is an inference and must be labelled "inference:".
- Key figures matter only insofar as they bear on the outcome: an acquirer's principals, the regulator who decides, a controlling shareholder, a prosecutor. For each, establish role, relevant track record, and affiliations a source documents (prior firms, political ties, regulatory relationships, litigation). Do not speculate about affiliations no source states.
- Prefer primary documents (SEC filings, regulator releases, court dockets, company statements) over commentary. Note when reporting is contested.
- Never invent prices, dates or percentages. If a deal has an offer price, quote it exactly as the source states it.
- Output ONLY the JSON object described in the user message, inside one ```json fence, nothing after the fence."""
    if web:
        return base + """
- Use web search deliberately: start from the supplied headlines and filings, then search for (a) the transaction or proceeding itself, (b) each decisive actor, (c) the next dated milestone. Stop searching when new results repeat what you have."""
    return base + """
- You have NO web access in this run (historical as-of date). Classify from the supplied evidence only, list what you would have investigated under "open_questions", and leave "findings" to what the supplied evidence itself supports."""


def _user_prompt(symbol: str, as_of: str, target: Optional[str], *,
                 profile: str, headlines: str, filings: str, quality: str,
                 last_close: Optional[float], web: bool) -> str:
    when = (f"The decision is made before the open of {target}; information "
            f"published up to that moment is usable — state each source's date."
            if target else
            f"The decision is made at the close of {as_of}; use information "
            f"published on or before that date.")
    close_line = (f"Last close: ${last_close:.2f}" if last_close else
                  "Last close: not supplied")
    return f"""Company: {symbol}
As-of (data cutoff): {as_of}
{when}
{close_line}

== SUPPLIED EVIDENCE (point-in-time) ==
Business profile:
{profile or 'not available'}

Recent headlines (newest first):
{headlines or 'none in window'}

Filings:
{filings or 'none supplied'}

Quality/red-flag screen:
{quality or 'none supplied'}

== TASK ==
{'Investigate on the web, then r' if web else 'R'}eturn ONLY this JSON:

```json
{{
  "situation": "one of {', '.join(SITUATIONS)}",
  "situation_confidence": "high|medium|low",
  "one_line": "<one sentence: what kind of trade this is and the single question that decides the next 1-5 sessions>",
  "deal": {{
    "present": true|false,
    "acquirer": "<name or null>",
    "offer_price": <number or null>,
    "consideration": "cash|stock|mixed|null",
    "announced": "YYYY-MM-DD or null",
    "expected_close": "<text or null>",
    "approvals": [{{"body": "<regulator/shareholders>", "status": "<pending|approved|review opened|…>", "date": "YYYY-MM-DD or null", "source": "<outlet, date, url>"}}],
    "break_risk": "<one sentence on what could stop it, or null>"
  }},
  "key_figures": [
    {{"name": "<person or entity>", "role": "<why they decide the outcome>", "relevance": "<sourced facts about their record/affiliations that bear on the outcome; prefix inferences with 'inference:'>", "source": "<outlet, date, url>"}}
  ],
  "findings": [
    {{"claim": "<one factual sentence>", "source": "<outlet or document>", "date": "YYYY-MM-DD", "url": "<url>"}}
  ],
  "dated_events": [{{"date": "YYYY-MM-DD or 'TBD'", "what": "<hearing, deadline, earnings, vote>", "source": "<outlet, date, url>"}}],
  "open_questions": ["<what remains unknown and would change the call>"]
}}
```
Keep findings to the 6-10 that matter for the next week; key_figures to those who decide the outcome."""


def _extract_json(text: str) -> Optional[dict]:
    if not text:
        return None
    fences = re.findall(r"```json\s*(\{.*?\})\s*```", text, re.S)
    candidates = fences[::-1] if fences else []
    if not candidates:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            candidates = [text[start:end + 1]]
    for c in candidates:
        try:
            return json.loads(c)
        except json.JSONDecodeError:
            continue
    return None


def investigate(
    symbol: str,
    as_of: str,
    *,
    live: bool,
    target: Optional[str] = None,
    profile: str = "",
    headlines: str = "",
    filings: str = "",
    quality: str = "",
    last_close: Optional[float] = None,
    mode: Optional[str] = None,
    model: Optional[str] = None,
) -> Investigation:
    """Run the stage. Raises on provider failure — the caller classes the
    block (expected evidence on a live run) and records the gap."""
    mode = (mode or MODEL.INVESTIGATION_MODE).lower()
    web = mode == "always" or (mode == "auto" and live)
    key = (symbol.upper(), str(as_of)[:10], web, model or MODEL.INVESTIGATION_MODEL)
    with _CACHE_LOCK:
        if key in _CACHE:
            if _CACHE[key].error:
                raise RuntimeError(_CACHE[key].error)
            return _CACHE[key]
        key_lock = _KEY_LOCKS.setdefault(key, threading.Lock())
    with key_lock:
        with _CACHE_LOCK:
            if key in _CACHE:
                inv = _CACHE[key]
                if inv.error:
                    raise RuntimeError(inv.error)
                return inv
        try:
            inv = _investigate_uncached(
                key, symbol, as_of, web=web, target=target, profile=profile,
                headlines=headlines, filings=filings, quality=quality,
                last_close=last_close, model=model)
        except Exception as e:
            # A failure is cached too: the prefetch thread and the in-loop
            # call would otherwise both pay for the same failing search.
            inv = Investigation(symbol=symbol.upper(), as_of=str(as_of)[:10],
                                web=web, parse_ok=False, error=str(e)[:300])
        with _CACHE_LOCK:
            _CACHE[key] = inv
    if inv.error:
        raise RuntimeError(inv.error)
    return inv


def _investigate_uncached(key, symbol, as_of, *, web, target, profile, headlines,
                          filings, quality, last_close, model) -> "Investigation":
    from services.llm_service import get_llm
    llm = get_llm()
    system = _system_prompt(web)
    prompt = _user_prompt(symbol, as_of, target if web else None,
                          profile=profile, headlines=headlines,
                          filings=filings, quality=quality,
                          last_close=last_close, web=web)
    inv = Investigation(symbol=symbol.upper(), as_of=str(as_of)[:10], web=web)
    usage: dict = {}
    if web:
        out = llm.generate_with_web_search(
            prompt, system, model=model or MODEL.INVESTIGATION_MODEL,
            max_tokens=MODEL.INVESTIGATION_MAX_TOKENS,
            max_searches=MODEL.INVESTIGATION_MAX_SEARCHES, usage_out=usage)
        text, inv.searches, inv.sources = out["text"], out["searches"], out["sources"]
        inv.model = out["model"]
    else:
        use_model = model or MODEL.INVESTIGATION_MODEL
        text = llm.generate(prompt, system, max_tokens=MODEL.INVESTIGATION_MAX_TOKENS,
                            temperature=0.2, model=use_model,
                            provider="anthropic", usage_out=usage)
        inv.model = usage.get("model") or use_model
        if not text:
            raise RuntimeError("investigation model returned no text")
    inv.input_tokens = int(usage.get("input_tokens") or 0)
    inv.output_tokens = int(usage.get("output_tokens") or 0)

    data = _extract_json(text)
    if not data:
        inv.parse_ok = False
        inv.error = "investigation response was not parseable JSON"
        inv.one_line = (text or "")[:300]
        raise RuntimeError(inv.error)

    sit = str(data.get("situation") or "OTHER").upper().strip()
    inv.situation = sit if sit in SITUATIONS else "OTHER"
    inv.situation_confidence = str(data.get("situation_confidence") or "low").lower()
    inv.one_line = str(data.get("one_line") or "").strip()
    deal = data.get("deal") or {}
    inv.deal = deal if isinstance(deal, dict) else {}
    inv.key_figures = [f for f in (data.get("key_figures") or []) if isinstance(f, dict)][:8]
    inv.findings = [f for f in (data.get("findings") or []) if isinstance(f, dict)][:12]
    inv.dated_events = [e for e in (data.get("dated_events") or []) if isinstance(e, dict)][:8]
    inv.open_questions = [str(q) for q in (data.get("open_questions") or [])][:8]

    offer = inv.deal.get("offer_price")
    try:
        if inv.deal.get("present") and offer is not None and last_close:
            inv.spread_pct = round((float(offer) / float(last_close) - 1) * 100, 1)
    except (TypeError, ValueError):
        inv.spread_pct = None

    return inv


def prefetch_many(symbols: list[str], as_of: str, *, live: bool,
                  target: Optional[str], news_by_symbol: dict,
                  workers: int = 4) -> "concurrent.futures.ThreadPoolExecutor":
    """Start the investigations for a run's symbols in the background.

    The research loop runs one symbol at a time and each web investigation
    takes minutes; run sequentially on a 20-symbol watchlist it would push
    the scheduled job past its 75-minute ceiling. Started here, the
    investigations overlap the price models' work, and the in-loop call
    finds its symbol in the cache (or waits on that symbol's lock while it
    finishes). Failures are swallowed here — the in-loop call re-raises
    them into the ledger where they belong. The executor is returned so the
    caller can release it (``shutdown(wait=False)``) when the run ends.
    """
    import concurrent.futures
    from services import usage_service as _usage

    def _one(symbol: str) -> None:
        try:
            from services.stock_data import get_company_profile
            quality = ""
            filings = ""
            try:
                from services.bad_apples_service import analyze_symbol, format_bad_apples_block
                quality = format_bad_apples_block(symbol, analyze_symbol(symbol, as_of))
            except Exception:
                pass
            try:
                from services.terminal_data import filings_block
                filings = filings_block(symbol, as_of)
            except Exception:
                pass
            with _usage.track("investigation", symbol=symbol, trade_date=as_of,
                              section=f"investigation:{symbol}"):
                investigate(symbol, as_of, live=live, target=target,
                            profile=get_company_profile(symbol) or "",
                            headlines=headlines_for_prompt(news_by_symbol.get(symbol) or []),
                            filings=filings, quality=quality)
        except Exception as e:
            logger.warning(f"{symbol}: investigation prefetch failed: {e}")

    pool = concurrent.futures.ThreadPoolExecutor(
        max_workers=max(1, workers), thread_name_prefix="investigate")
    for symbol in symbols:
        pool.submit(_one, symbol)
    return pool


def format_investigation_block(inv: Investigation, last_close: Optional[float] = None) -> str:
    """The prompt block. States its own provenance (web vs supplied-only)."""
    src = ("web research + supplied evidence" if inv.web
           else "supplied evidence only — no web access on a historical as-of")
    lines = [f"[{inv.symbol} — situation & investigation as of {inv.as_of} ({src})]",
             f"Situation: {inv.situation} (confidence {inv.situation_confidence})"
             + (f" — {inv.one_line}" if inv.one_line else "")]
    d = inv.deal
    if d.get("present"):
        offer = d.get("offer_price")
        terms = (f"{d.get('acquirer') or 'acquirer n/a'}"
                 + (f", ${float(offer):.2f} per share" if isinstance(offer, (int, float)) else "")
                 + (f" {d['consideration']}" if d.get("consideration") else "")
                 + (f", announced {d['announced']}" if d.get("announced") else "")
                 + (f", expected close {d['expected_close']}" if d.get("expected_close") else ""))
        lines.append(f"Deal: {terms}")
        if inv.spread_pct is not None and last_close:
            lines.append(f"Gross spread to offer (computed): last close ${last_close:.2f} "
                         f"vs ${float(offer):.2f} = {inv.spread_pct:+.1f}% — the market is "
                         f"pricing meaningful completion risk when this is wide.")
        for a in (d.get("approvals") or [])[:6]:
            if isinstance(a, dict):
                lines.append(f"- Approval: {a.get('body')} — {a.get('status')}"
                             + (f" ({a['date']})" if a.get("date") else "")
                             + (f" (src: {a['source']})" if a.get("source") else ""))
        if d.get("break_risk"):
            lines.append(f"Break risk: {d['break_risk']}")
    if inv.key_figures:
        lines.append("Key figures:")
        for f in inv.key_figures[:6]:
            lines.append(f"- {f.get('name')} — {f.get('role')}. {f.get('relevance')}"
                         + (f" (src: {f['source']})" if f.get("source") else ""))
    if inv.findings:
        lines.append("Findings:")
        for f in inv.findings[:10]:
            lines.append(f"- {f.get('claim')} (src: {f.get('source')} | "
                         f"{f.get('date')} | {f.get('url')})")
    if inv.dated_events:
        lines.append("Dated events: " + "; ".join(
            f"{e.get('date')}: {e.get('what')}" for e in inv.dated_events[:6]))
    if inv.open_questions:
        lines.append("Open questions: " + " | ".join(inv.open_questions[:5]))
    lines.append("Cite these findings with their src/date exactly as news is "
                 "cited. Findings marked 'inference:' are the investigator's "
                 "reading, not sourced fact — say so if you lean on one.")
    if inv.web:
        lines.append(f"({inv.searches} web searches, {len(inv.sources)} sources consulted)")
    return "\n".join(lines)


def headlines_for_prompt(articles: list, n: int = 15) -> str:
    """Compact newest-first headline list for the investigator."""
    out = []
    for a in (articles or [])[:n]:
        if hasattr(a, "title"):
            title, src = a.title, getattr(a, "source", None)
            date = str(getattr(a, "published_at", "") or "")[:10]
        else:
            title, src = a.get("title", ""), a.get("source")
            date = str(a.get("published_at") or a.get("published_date") or "")[:10]
        out.append(f"- {date} [{src or 'unattributed'}] {title}")
    return "\n".join(out)

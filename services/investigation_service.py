"""Situation classification and open-web investigation for one symbol.

The research report used to be written from whatever blocks the pipeline
happened to hold, with the prompt telling the model it "has no tools and
cannot browse". For a stock under a cash takeover that meant a momentum
SELL written off moving averages while the only question that mattered.
will the deal close, never entered the prompt (BHF, 2026-09-01).

This stage runs BEFORE the report is written. One tool-using call:

  1. classifies the situation the symbol is in (pending acquisition,
     legal/regulatory overhang, earnings event, leadership change,
     distress, momentum only, …) from the supplied evidence;
  2. investigates it on the open web. Deal terms, regulators and dated
     milestones, the key figures involved and what their record and
     affiliations imply for the outcome, citing every finding;
  3. returns structured JSON the report prompt and the synthesis read.

Web access is a run TOOL the frontend switches (``web=True``), on by
default for next-day runs and off for backtest dates, because web search
cannot be bounded to a past date. Without it the classifier runs from the
supplied evidence only and says so. The backend default is off: a caller
that does not ask for the tool never pays for a search. The investigator is
told the decision moment (before the target session's open) and to date
every source.
"""

import hashlib
import json
import logging
import re
import threading
from contextvars import ContextVar
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
    "MOMENTUM_ONLY",             # nothing situational: flow and trend
    "OTHER",
)

# Each researched question is its own web search, so this is a spend cap as
# much as a reading cap: three sourced answers is what a report can carry
# without the sections turning into a list.
MAX_RESEARCH_QUESTIONS = 3

_CACHE: dict[tuple, "Investigation"] = {}
_CACHE_LOCK = threading.Lock()
# One lock per cache key: a prefetch thread and the in-loop call for the
# same symbol must never both pay for the search, the second waits.
_KEY_LOCKS: dict[tuple, threading.Lock] = {}

# Answers to specific questions live in their own cache: a different value
# shape, and a key that carries the question itself.
_ANSWER_CACHE: dict[tuple, dict] = {}
_ANSWER_LOCKS: dict[tuple, threading.Lock] = {}

# A run-level ceiling on question research, claimed here and opened by
# whoever starts a run. The per-symbol cap alone does not bound a run:
# three questions on each of a 20-symbol watchlist is 60 more web-search
# turns on the serial model loop of a job that already takes ~40 minutes and
# is killed at 95. Questions past the ceiling come back unanswered and their
# sections say so, which is the same contract as a failed search.
#
# The ceiling is held in a ContextVar rather than a module global because
# the scheduled job and an interactive report run in the SAME process: one
# global counter meant the 07:00 watchlist job drained the ceiling to zero
# and every report a user asked for afterwards was told its questions had
# been ranked out, while a run nobody had opened a ceiling for searched
# without any bound at all. A ContextVar gives each run its own ledger for
# free: the runner and the callback each open one, and the copy_context()
# hand-offs the model loop and the research fan-out already use carry the
# same ledger OBJECT into their workers, so a run's symbols still share one
# counter. Unset (a caller outside any run, e.g. an export) means unlimited.
_BUDGET: "ContextVar[dict | None]" = ContextVar("research_budget", default=None)
_BUDGET_LOCK = threading.Lock()


def begin_research_budget(questions: "int | None") -> None:
    """Open a ceiling of ``questions`` for the run starting in this context.

    Called once per run, before its symbols are dispatched. ``None`` lifts
    the ceiling. Context state rather than a threaded parameter because the
    questions are raised deep inside one symbol's model call and the ceiling
    is a property of the run around it.
    """
    _BUDGET.set(None if questions is None
                else {"left": max(0, int(questions))})
    logger.debug("research budget for this run: %s question(s)",
                 "unlimited" if questions is None else questions)


def research_budget_left() -> "int | None":
    """What is left of this run's ceiling. For tests and progress lines."""
    ledger = _BUDGET.get()
    if ledger is None:
        return None
    with _BUDGET_LOCK:
        return ledger["left"]


def _claim_budget(wanted: int) -> int:
    """Claim up to ``wanted`` questions from this run's ceiling."""
    ledger = _BUDGET.get()
    if ledger is None:
        return wanted
    with _BUDGET_LOCK:
        take = min(wanted, ledger["left"])
        ledger["left"] -= take
        return take


def _evidence_digest(profile: str, headlines: str, filings: str,
                     quality: str) -> str:
    """Fingerprint of the evidence one call was handed.

    The cache keyed on (symbol, date, web, model) alone, so a run that fed
    the investigator different headlines, new filings or a changed prompt
    read back the previous run's answer. ``last_close`` is deliberately
    outside the digest: it moves the spread arithmetic, not the question
    being researched, and the background prefetch does not have it, so
    folding it in would make every in-loop call miss the warm entry and pay
    for the same search a second time.
    """
    h = hashlib.sha256()
    for part in (profile, headlines, filings, quality):
        h.update((part or "").encode("utf-8", "replace"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


def _context_digest(context: str) -> str:
    """Fingerprint of the figures one researched question was raised by."""
    return hashlib.sha256(
        (context or "").encode("utf-8", "replace")).hexdigest()[:16]


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
    web_skipped: str = ""      # why a web-enabled run stayed web-free

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


def _system_prompt(web: bool) -> str:
    base = """You are an investigative equity research analyst preparing the situational brief that a trading desk reads BEFORE it looks at any chart. Your job is to establish what kind of situation the company is in, who the decisive actors are, and what is scheduled, with sources.

Discipline:
- Every finding carries a source: outlet or document, publication date, URL. A claim you cannot source does not go in "findings"; it may go in "open_questions".
- Separate fact from inference. A relationship between two people or firms is a finding only when a source states it; what it implies for the outcome is an inference and must be labelled "inference:".
- Key figures matter only insofar as they bear on the outcome: an acquirer's principals, the regulator who decides, a controlling shareholder, a prosecutor. For each, establish role, relevant track record, and affiliations a source documents (prior firms, political ties, regulatory relationships, litigation). Do not speculate about affiliations no source states.
- Prefer primary documents (SEC filings, regulator releases, court dockets, company statements) over commentary. Note when reporting is contested.
- Never invent prices, dates or percentages. If a deal has an offer price, quote it exactly as the source states it.
- Write every text field in plain sentences: no em dashes or en dashes as punctuation (commas, periods, parentheses instead), no "not just X, it's Y", no "delve", "dive", "landscape", "unlock", no filler.
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
            f"published up to that moment is usable. State each source's date."
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
    web: bool = False,
    target: Optional[str] = None,
    profile: str = "",
    headlines: str = "",
    filings: str = "",
    quality: str = "",
    last_close: Optional[float] = None,
    model: Optional[str] = None,
) -> Investigation:
    """Run the stage. ``web`` is the frontend's tool switch (default off).
    Raises on provider failure, the caller classes the block (expected
    evidence) and records the gap."""
    key = (symbol.upper(), str(as_of)[:10], web,
           model or MODEL.INVESTIGATION_MODEL,
           _evidence_digest(profile, headlines, filings, quality))
    with _CACHE_LOCK:
        if key in _CACHE:
            if _CACHE[key].error:
                raise RuntimeError(_CACHE[key].error)
            return _CACHE[key]
        key_lock = _KEY_LOCKS.setdefault(key, threading.Lock())

    # Triage before paying for a search: the web-free classification is a
    # fraction of a cent, and a name that comes back MOMENTUM_ONLY has
    # nothing for the web to add. The web result is cached under its own
    # key; the triage result under web=False, so nothing runs twice.
    if web and MODEL.INVESTIGATION_WEB_SKIP:
        try:
            triage = investigate(symbol, as_of, web=False, target=target,
                                 profile=profile, headlines=headlines,
                                 filings=filings, quality=quality,
                                 last_close=last_close, model=model)
        except Exception as e:
            # A failed triage must not block the real research.
            logger.warning(f"{symbol}: web-free triage failed ({str(e)[:80]}); "
                           f"searching anyway")
            triage = None
        if triage is not None and triage.situation in MODEL.INVESTIGATION_WEB_SKIP:
            skipped = Investigation(**{**triage.__dict__})
            skipped.web_skipped = (f"classified {triage.situation} from the "
                                   f"supplied evidence; web research not spent")
            with _CACHE_LOCK:
                _CACHE[key] = skipped
            return skipped
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
        provider = "openai" if use_model.startswith("gpt-") else "anthropic"
        gen_kwargs = ({"reasoning_effort": MODEL.INVESTIGATION_OPENAI_EFFORT}
                      if provider == "openai" else {})
        text = llm.generate(prompt, system, max_tokens=MODEL.INVESTIGATION_MAX_TOKENS,
                            temperature=0.2, model=use_model,
                            provider=provider, usage_out=usage, **gen_kwargs)
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


RESEARCH_SYSTEM_PROMPT = """You are an investigative equity research analyst. You are given ONE specific question about a company, raised by something measurable that stands out in today's data, and you answer that question and nothing else.

Discipline:
- Answer the question that was asked. Do not write a general note on the company, do not summarise its chart, do not offer a trading view.
- Every claim carries a source: outlet or document, publication date, URL. What you cannot source does not go in "finding"; it goes in "unresolved".
- Separate fact from inference. Label a reading of the facts "inference:" inside the sentence that makes it.
- Never invent a number, a date or a name. Quote figures exactly as the source states them.
- If the web says nothing about this question, say so plainly in "finding" and leave "citations" empty. An honest "no reporting found" is a useful answer; a plausible-sounding cause you cannot source is not.
- Write plain sentences: no em dashes or en dashes as punctuation, no "not just X, it's Y", no "delve", "dive", "landscape", "unlock", no filler.
- Output ONLY the JSON object described in the user message, inside one ```json fence, nothing after the fence."""


def _research_prompt(symbol: str, as_of: str, target: Optional[str],
                     question: str, context: str) -> str:
    when = (f"The decision this feeds is made before the open of {target}; "
            f"information published up to that moment is usable."
            if target else
            f"The decision this feeds is made at the close of {as_of}; use "
            f"information published on or before that date.")
    return f"""Company: {symbol}
As-of (data cutoff): {as_of}
{when}

== THE QUESTION ==
{question}

== WHAT RAISED IT (measured from this run's own data) ==
{context or 'no supporting figures supplied'}

== TASK ==
Search the open web for what explains or bears on that question, then return ONLY this JSON:

```json
{{
  "finding": "<2-5 sentences answering the question, every claim attributable to one of the citations below; say 'no reporting found' if that is the truth>",
  "citations": [
    {{"source": "<outlet or document>", "date": "YYYY-MM-DD", "url": "<url>"}}
  ],
  "unresolved": "<what the search could not establish and would change the answer, or an empty string>"
}}
```"""


def _answer(question: str, **fields) -> dict:
    """The shape every caller reads: the question it was asked, the finding,
    and the citations behind it. Everything else is bookkeeping."""
    out = {"question": question, "finding": "", "citations": [],
           "unresolved": "", "searches": 0, "model": "", "web": True,
           "input_tokens": 0, "output_tokens": 0, "parse_ok": True,
           "error": None}
    out.update(fields)
    return out


def _research_one(symbol: str, as_of: str, question: str, *,
                  target: Optional[str], context: str, model: Optional[str],
                  max_searches: int) -> dict:
    from services.llm_service import get_llm

    llm = get_llm()
    usage: dict = {}
    out = llm.generate_with_web_search(
        _research_prompt(symbol, as_of, target, question, context),
        RESEARCH_SYSTEM_PROMPT,
        model=model or MODEL.INVESTIGATION_MODEL,
        max_tokens=MODEL.INVESTIGATION_MAX_TOKENS,
        max_searches=max_searches, usage_out=usage)

    ans = _answer(question, searches=int(out.get("searches") or 0),
                  model=out.get("model") or "",
                  input_tokens=int(usage.get("input_tokens") or 0),
                  output_tokens=int(usage.get("output_tokens") or 0))
    data = _extract_json(out.get("text") or "")
    if data:
        ans["finding"] = str(data.get("finding") or "").strip()
        ans["unresolved"] = str(data.get("unresolved") or "").strip()
        ans["citations"] = [c for c in (data.get("citations") or [])
                            if isinstance(c, dict)][:6]
    else:
        # A prose answer with the provider's own sources still carries the
        # research; only the structure was lost. Recording it as an error
        # would throw away a search that was already paid for.
        ans["parse_ok"] = False
        ans["finding"] = (out.get("text") or "").strip()[:1200]
    if not ans["citations"]:
        ans["citations"] = [{"source": s.get("title") or "web", "date": "",
                             "url": s.get("url") or ""}
                            for s in (out.get("sources") or [])[:6]
                            if isinstance(s, dict)]
    if not ans["finding"]:
        raise RuntimeError("question research returned no text")
    return ans


def research_questions(symbol: str, as_of: str, questions: list[str], *,
                       web: bool = False, target: Optional[str] = None,
                       model: Optional[str] = None,
                       context_by_question: Optional[dict] = None,
                       max_questions: Optional[int] = None
                       ) -> list[dict]:
    """Research specific questions about ``symbol``, one web search each.

    ``investigate`` above asks what KIND of situation a symbol is in, and it
    triages first so a plain momentum name never pays for a search. This is
    the other half: the questions handed here were raised by something
    measurable that stands out (a chain positioned against the tape, three
    executives selling in a fortnight), which is exactly the case that
    should search, so nothing here triages them away.

    ``context_by_question`` carries the measured figures each question was
    raised by; the prompt puts them under "what raised it" so the search is
    aimed at the anomaly rather than the ticker.

    Each answer is {question, finding, citations} plus its own bookkeeping,
    cached under a key that carries the question AND a digest of its
    context, so a changed question or changed figures is a new search and a
    repeated one is free. A question whose search fails
    comes back with an empty finding and ``error`` set rather than raising:
    the caller still has the figures that raised it and must render them as
    unresearched.

    Returns [] when ``web`` is off. There is no web-free answer to "why is
    this happening" that is worth putting in a report, and a caller that got
    one back would print it as research.

    Fewer answers than questions is normal and is the caller's signal that a
    question went unasked: the per-symbol cap and the run-level ceiling
    (``begin_research_budget``) both trim the list, and an anomaly with no
    answer must be rendered as unresearched rather than dropped. The asked
    questions run CONCURRENTLY. They sit on the serial per-symbol model loop,
    so three web searches one after another would add their full latency to
    every symbol of a scheduled run; each one is an independent call and the
    caches below are already shared safely with the prefetch pool.
    """
    import concurrent.futures
    from contextvars import copy_context

    asked: list[str] = []
    for q in (questions or []):
        q = str(q or "").strip()
        if q and q not in asked:
            asked.append(q)
    # Read at call time, not bound as a default: the cap is a spend knob and
    # a test or a caller that lowers it must actually lower it.
    cap = MAX_RESEARCH_QUESTIONS if max_questions is None else max_questions
    asked = asked[:max(0, int(cap))]
    if not asked:
        return []
    if not web:
        logger.debug("%s: %d question(s) left unresearched, web research is "
                     "off for this run", symbol, len(asked))
        return []

    allowed = _claim_budget(len(asked))
    if allowed < len(asked):
        logger.info("%s: %d of %d question(s) not researched, this run's "
                    "research budget is spent", symbol,
                    len(asked) - allowed, len(asked))
        asked = asked[:allowed]
    if not asked:
        return []

    # One shared search budget: the run's ceiling divided over the questions
    # it is spending it on, never per question on top of it.
    per_question = max(1, MODEL.INVESTIGATION_MAX_SEARCHES // len(asked))
    use_model = model or MODEL.INVESTIGATION_MODEL
    contexts = context_by_question or {}

    def _one(question: str) -> dict:
        # The context is in the key, not just the question. Two runs can
        # raise the same sentence off different figures (the quality
        # question names the failure COUNT, the facts name which checks
        # failed), and serving the first run's answer for the second would
        # cite reporting about numbers this run never measured.
        key = (symbol.upper(), str(as_of)[:10], question, use_model,
               _context_digest(contexts.get(question, "")))
        with _CACHE_LOCK:
            cached = _ANSWER_CACHE.get(key)
            key_lock = _ANSWER_LOCKS.setdefault(key, threading.Lock())
        if cached is not None:
            return cached
        with key_lock:
            with _CACHE_LOCK:
                cached = _ANSWER_CACHE.get(key)
            if cached is not None:
                return cached
            try:
                cached = _research_one(
                    symbol, str(as_of)[:10], question, target=target,
                    context=contexts.get(question, ""), model=use_model,
                    max_searches=per_question)
            except Exception as e:
                logger.warning("%s: research failed for %r: %s",
                               symbol, question[:60], e)
                # Cached like a successful answer: two workers on the same
                # symbol must not both pay for a failing search.
                cached = _answer(question, error=str(e)[:200], parse_ok=False)
            with _CACHE_LOCK:
                _ANSWER_CACHE[key] = cached
            return cached

    if len(asked) == 1:
        return [_one(asked[0])]
    with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(asked), thread_name_prefix="research") as pool:
        # A fresh context copy per worker: usage attribution is a ContextVar
        # and a plain thread would start outside the caller's track() scope,
        # billing these searches to no stage at all.
        futures = [pool.submit(copy_context().run, _one, q) for q in asked]
        return [f.result() for f in futures]


def prefetch_many(symbols: list[str], as_of: str, *, web: bool,
                  target: Optional[str], news_by_symbol: dict,
                  evidence: "set | list | tuple | None" = None,
                  workers: int = 4) -> "concurrent.futures.ThreadPoolExecutor":
    """Start the investigations for a run's symbols in the background.

    The research loop runs one symbol at a time and each web investigation
    takes minutes; run sequentially on a 20-symbol watchlist it would push
    the scheduled job past its 75-minute ceiling. Started here, the
    investigations overlap the price models' work, and the in-loop call
    finds its symbol in the cache (or waits on that symbol's lock while it
    finishes). Failures are swallowed here, the in-loop call re-raises
    them into the ledger where they belong. The executor is returned so the
    caller can release it (``shutdown(wait=False)``) when the run ends.

    ``evidence`` is the run's selected evidence set. It exists so this
    thread assembles the SAME evidence the in-loop call will: the cache key
    carries a digest of it, so a prefetch that computed a quality screen the
    run did not select would miss the warm entry and pay for the search
    twice, which is the cost this function exists to avoid.
    """
    import concurrent.futures
    from services import usage_service as _usage

    evidence = set(evidence if evidence is not None else MODEL.DEFAULT_EVIDENCE)

    def _one(symbol: str) -> None:
        try:
            from services.stock_data import get_company_profile
            quality = ""
            filings = ""
            try:
                from services.bad_apples_service import analyze_symbol, format_bad_apples_block
                # The run's own capped window; an empty list means "quiet or
                # source down", which the scan must fetch to tell apart.
                if "quality" in evidence:
                    quality = format_bad_apples_block(
                        symbol, analyze_symbol(
                            symbol, as_of,
                            articles=(news_by_symbol.get(symbol) or None)))
            except Exception:
                pass
            try:
                from services.terminal_data import filings_block
                filings = filings_block(symbol, as_of)
            except Exception:
                pass
            with _usage.track("investigation", symbol=symbol, trade_date=as_of,
                              section=f"investigation:{symbol}"):
                investigate(symbol, as_of, web=web, target=target,
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


WEB_RESEARCH_TOOL = "web_research"


def default_tools(target_date) -> list[str]:
    """The frontend's default tool set for a run whose target session is
    ``target_date``: web research on for a next-day (forward-testing) run,
    off for a backtest, where the open web would leak the future."""
    from datetime import date
    try:
        t = date.fromisoformat(str(target_date)[:10])
    except (TypeError, ValueError):
        return [WEB_RESEARCH_TOOL]
    return [WEB_RESEARCH_TOOL] if t >= date.today() else []


def format_investigation_block(inv: Investigation, last_close: Optional[float] = None) -> str:
    """The prompt block. States its own provenance (web vs supplied-only)."""
    if inv.web:
        src = "web research + supplied evidence"
    elif inv.web_skipped:
        src = "supplied evidence only; " + inv.web_skipped
    else:
        src = "supplied evidence only, no web access on a historical as-of"
    lines = [f"[{inv.symbol}: situation & investigation as of {inv.as_of} ({src})]",
             f"Situation: {inv.situation} (confidence {inv.situation_confidence})"
             + (f": {inv.one_line}" if inv.one_line else "")]
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
                         f"vs ${float(offer):.2f} = {inv.spread_pct:+.1f}%: the market is "
                         f"pricing meaningful completion risk when this is wide.")
        for a in (d.get("approvals") or [])[:6]:
            if isinstance(a, dict):
                lines.append(f"- Approval: {a.get('body')}: {a.get('status')}"
                             + (f" ({a['date']})" if a.get("date") else "")
                             + (f" (src: {a['source']})" if a.get("source") else ""))
        if d.get("break_risk"):
            lines.append(f"Break risk: {d['break_risk']}")
    if inv.key_figures:
        lines.append("Key figures:")
        for f in inv.key_figures[:6]:
            lines.append(f"- {f.get('name')}: {f.get('role')}. {f.get('relevance')}"
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
                 "reading, not sourced fact, say so if you lean on one.")
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

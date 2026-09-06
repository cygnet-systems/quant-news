"""The portfolio view, rolled up from the per-symbol research epilogues.

There used to be a second model call here (``summarize_news_structured``):
it re-read a 40-headline sample of the run's news across every symbol and
wrote one portfolio JSON. Measured on prod it was a thinner opinion on data
the per-symbol research had already read in full, and it was the only place
news was read twice. Everything the portfolio view needs is already in the
research epilogues (stance, conviction, sentiment alignment, watch items,
scenarios, evidence gaps, the situation), so the roll-up is arithmetic and
text assembly, no model, no news.

``decide_action`` is the other half of the same decision. Measured over 480
deduplicated symbol-days on prod, the synthesis model's action agreed with
the research verdict 68% of the time and its active calls hit 49.8% against
the report's 46.1% on the same days: a re-vote with no edge. The action is
now the report's verdict, with one explicit HOLD rule, and the synthesis
model writes the memo around it (conflicts, level, trigger, model notes).
"""
from __future__ import annotations

import logging
from collections import Counter
from typing import Optional

logger = logging.getLogger(__name__)

STANCE_OF = {"BUY": "BULLISH", "SELL": "BEARISH", "HOLD": "NEUTRAL"}
BULL = ("BULLISH", "CAUTIOUS_BULLISH")
BEAR = ("BEARISH", "CAUTIOUS_BEARISH")
# Models whose decision is, or derives from, the research verdict itself, so
# they cannot count as a second opinion on it.
NOT_INDEPENDENT = ("trading_agents", "ensemble", "recommendation_synthesis")
RULE_VERDICT = "research_verdict"
RULE_HOLD = "models_disagree_and_missing_evidence"
RULE_NO_RESEARCH = "no_research_verdict"
MAX_WATCH_ITEMS = 8


def _verdict(entry: dict) -> tuple[str, Optional[float], bool]:
    """(BUY/SELL/HOLD, stated conviction, whether a research verdict exists)."""
    research = entry.get("research") or {}
    decision = str(research.get("decision") or "").upper()
    conviction = research.get("stated_conviction")
    if conviction is None:
        conviction = entry.get("stated_conviction")
    conviction = float(conviction) if isinstance(conviction, (int, float)) else None
    if decision in ("BUY", "SELL", "HOLD"):
        return decision, conviction, True
    stance = str(entry.get("recommendation") or "").upper()
    if stance in BULL:
        return "BUY", conviction, bool(entry.get("stance_source") == "research_verdict")
    if stance in BEAR:
        return "SELL", conviction, bool(entry.get("stance_source") == "research_verdict")
    return "HOLD", conviction, False


def _expected_gaps(entry: dict) -> list[str]:
    ev = (entry.get("research") or {}).get("evidence")
    gaps = (ev.get("gaps") if isinstance(ev, dict) else None) or []
    return [str(g.get("label") or g.get("block") or "evidence")
            for g in gaps if isinstance(g, dict) and g.get("severity") == "expected"]


def _situation(entry: dict) -> str:
    inv = (entry.get("research") or {}).get("investigation") or {}
    return str(inv.get("situation") or "").strip()


def decide_action(symbol: str, entry: dict, sym_signals: dict) -> dict:
    """The action for one symbol: the research verdict, or HOLD when the
    independent models mostly disagree with it AND the report says it was
    written without expected evidence. Both conditions, not either: the
    models are momentum readers that disagree with a research call for a
    living, and a report with all its evidence stands on its own."""
    verdict, conviction, has_research = _verdict(entry or {})
    agreeing = opposed = 0
    for model, sig in (sym_signals or {}).items():
        if model in NOT_INDEPENDENT or not isinstance(sig, dict):
            continue
        decision = str(sig.get("decision") or "").upper()
        if decision not in ("BUY", "SELL"):
            continue
        if verdict in ("BUY", "SELL"):
            if decision == verdict:
                agreeing += 1
            else:
                opposed += 1
    gaps = _expected_gaps(entry or {})
    if not has_research:
        action, rule = "HOLD", RULE_NO_RESEARCH
    elif verdict in ("BUY", "SELL") and opposed > agreeing and gaps:
        action, rule = "HOLD", RULE_HOLD
    else:
        action, rule = verdict, RULE_VERDICT
    return {
        "action": action,
        "verdict": verdict,
        # The report's own conviction is the only probability attached to
        # this call; a HOLD carries no directional claim to score.
        "p_correct": (round(conviction, 2) if conviction is not None
                      and action != "HOLD" else 0.5),
        "rule": rule,
        "models_agreeing": agreeing,
        "models_opposed": opposed,
        "missing_evidence": gaps,
    }


def fixed_actions(by_symbol: dict, model_signals: Optional[dict],
                  symbols: list) -> dict:
    return {s: decide_action(s, (by_symbol or {}).get(s) or {},
                             (model_signals or {}).get(s) or {})
            for s in symbols}


def build_overall_rollup(by_symbol: dict, model_signals: Optional[dict] = None
                         ) -> Optional[dict]:
    """The portfolio-level view in the shape the renderers and the markdown
    exporter already read (recommendation, confidence, market_sentiment,
    key_developments, developments_read, risk_factors, watch_items), built
    from the per-symbol entries. None when no symbol carries a verdict."""
    entries = {s: e for s, e in (by_symbol or {}).items() if isinstance(e, dict)}
    if not entries:
        return None
    verdicts = {}
    for sym, entry in entries.items():
        verdict, conviction, has_research = _verdict(entry)
        if has_research or verdict != "HOLD":
            verdicts[sym] = (verdict, conviction)
    if not verdicts:
        return None
    counts = Counter(v for v, _ in verdicts.values())
    n = len(verdicts)
    bull, bear = counts.get("BUY", 0), counts.get("SELL", 0)
    if bull > bear and bull * 3 >= n * 2:
        stance = "BULLISH"
    elif bull > bear:
        stance = "CAUTIOUS_BULLISH"
    elif bear > bull and bear * 3 >= n * 2:
        stance = "BEARISH"
    elif bear > bull:
        stance = "CAUTIOUS_BEARISH"
    else:
        stance = "NEUTRAL"
    convictions = [c for _, c in verdicts.values() if c is not None]
    confidence = (round(sum(convictions) / len(convictions), 2)
                  if convictions else None)

    situations = {s: _situation(e) for s, e in entries.items() if _situation(e)}
    gaps = {s: _expected_gaps(e) for s, e in entries.items() if _expected_gaps(e)}
    held = {}
    if model_signals is not None:
        for sym in verdicts:
            d = decide_action(sym, entries[sym], (model_signals or {}).get(sym) or {})
            if d["rule"] == RULE_HOLD:
                held[sym] = d

    per_symbol = []
    for sym in sorted(verdicts):
        verdict, conviction = verdicts[sym]
        piece = f"{sym} {verdict}"
        if conviction is not None:
            piece += f" ({conviction:.2f})"
        if sym in situations:
            piece += f", {situations[sym].lower().replace('_', ' ')}"
        if sym in held:
            piece += ", held back"
        per_symbol.append(piece)
    key_developments = (
        f"{n} symbol{'s' if n != 1 else ''} researched: {bull} bullish, "
        f"{bear} bearish, {counts.get('HOLD', 0)} neutral. "
        + "; ".join(per_symbol) + "."
    )
    read_bits = []
    if stance == "NEUTRAL":
        read_bits.append("No side dominates the research verdicts today.")
    else:
        side = "long" if stance in BULL else "short"
        read_bits.append(f"The research verdicts lean {side} "
                         f"({max(bull, bear)} of {n}).")
    if held:
        read_bits.append(
            f"{len(held)} held back to HOLD because the independent models "
            f"mostly disagree and the report was written without expected "
            f"evidence: {', '.join(sorted(held))}.")
    if situations:
        read_bits.append("Situations in play: " + "; ".join(
            f"{s} {v.lower().replace('_', ' ')}" for s, v in sorted(situations.items())) + ".")
    developments_read = " ".join(read_bits)

    risk_bits = []
    if gaps:
        risk_bits.append("Written without expected evidence: " + "; ".join(
            f"{s} ({', '.join(g)})" for s, g in sorted(gaps.items())) + ".")
    thin = [s for s, (v, c) in verdicts.items() if c is not None and c < 0.55]
    if thin:
        risk_bits.append(f"Conviction below 0.55 on {', '.join(sorted(thin))}: "
                         f"coin-flip calls by the report's own account.")
    risk_factors = " ".join(risk_bits) or "No evidence gaps declared by the reports."

    watch: list[str] = []
    for sym in sorted(verdicts):
        items = entries[sym].get("watch_items")
        if not items:
            items = ((entries[sym].get("research") or {}).get("structured") or {}
                     ).get("watch_items") or []
        for item in items[:2]:
            text = str(item).strip()
            if text and len(watch) < MAX_WATCH_ITEMS:
                watch.append(f"{sym}: {text}")

    alignment = [str(e.get("sentiment_explanation") or "").strip()
                 for e in entries.values()]
    alignment = [a for a in alignment if a]
    return {
        "recommendation": stance,
        "confidence": confidence,
        "market_sentiment": ("BULLISH" if stance in BULL else
                             "BEARISH" if stance in BEAR else "NEUTRAL"),
        "sentiment_explanation": (
            f"Rolled up from {n} research verdict{'s' if n != 1 else ''}; "
            f"confidence is the mean of the reports' own stated convictions"
            + (", none stated" if confidence is None else "") + "."
            + (f" {alignment[0]}" if len(alignment) == 1 else "")),
        "key_developments": key_developments,
        "developments_read": developments_read,
        "risk_factors": risk_factors,
        "risks_read": (
            "The live risk is the evidence the reports did not have, not a "
            "portfolio-wide call." if gaps else
            "No portfolio-wide risk beyond each symbol's own scenarios."),
        "watch_items": watch,
        "source": "research_epilogues",
        "rollup": True,
    }

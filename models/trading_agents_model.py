"""Research-driven single-agent prediction model.

Wraps `models.single_agent.SingleAgentResearch` — a self-contained,
lookahead-safe research agent that gathers SPY / sector / ticker technicals,
point-in-time news, and fundamentals, then produces a full-text analysis with a
BUY/SELL/HOLD decision in a single LLM call.

The model keeps the name "trading_agents" for backward compatibility with the
predictions/reports already persisted under that key, but it no longer depends
on the external TradingAgents package (that dependency broke when the upstream
fork advanced to v0.3.1; see TradingAgents_v0.3_Incorporation_Analysis.md). The
research strategy now lives in-tree behind the BaseModel seam, so it is
swappable and an upstream release can never silently disable it again.

Evidence discipline (services.evidence_contract): every context block this
wrapper assembles is classed required / expected / optional. A required block
that cannot be built raises — the symbol fails visibly instead of producing a
report that reads as complete. An expected block that cannot be built is
recorded as a gap that travels into the prompt, the footer, the prediction
details and the run's completeness check.

The raw_response (full report text) is stored in details for persistence in
Postgres and viewing in the History tab.
"""

import logging
import os

import pandas as pd

from config import MODEL
from models.base import BaseModel, PredictionResult
from services.evidence_contract import (
    OPTIONAL, EvidenceLedger, MissingRequiredEvidence,
)
from services.investigation_service import WEB_RESEARCH_TOOL

logger = logging.getLogger(__name__)


def _emit(stage: str, message: str, payload: dict | None = None) -> None:
    try:
        from services import progress_service as _prog
        _prog.emit(stage, message, payload=payload)
    except Exception:
        pass


class TradingAgentsModel(BaseModel):
    """Single-agent research-driven prediction model (decoupled, in-tree)."""

    @property
    def name(self) -> str:
        return "trading_agents"

    def is_ready(self) -> bool:
        # Key check follows the DEFAULT model's provider (a per-call
        # research_model override still degrades gracefully — predict()
        # catches provider errors and returns HOLD/error).
        if MODEL.TRADING_AGENTS_MODEL.startswith("gpt-"):
            return bool(os.environ.get("OPENAI_API_KEY"))
        return bool(os.environ.get("ANTHROPIC_API_KEY"))

    def predict(
        self,
        symbol: str,
        ohlcv_df: pd.DataFrame,
        **kwargs,
    ) -> PredictionResult:
        if not self.is_ready():
            return PredictionResult(
                model_name=self.name,
                decision="HOLD",
                confidence=0.0,
                up_probability=0.5,
                details={},
                error="ANTHROPIC_API_KEY not set",
            )

        try:
            return self._run_analysis(symbol, ohlcv_df, **kwargs)
        except MissingRequiredEvidence as e:
            # Deliberate refusal: the report is not written without this.
            logger.error(str(e))
            _emit("error", f"{symbol}: research report NOT written — {e.reason} "
                           f"({e.block}). A report without it would read as "
                           f"complete and be wrong.")
            return PredictionResult(
                model_name=self.name,
                decision="HOLD",
                confidence=0.0,
                up_probability=0.5,
                details={"missing_required": e.block,
                         "missing_reason": e.reason},
                error=str(e),
            )
        except Exception as e:
            logger.error(f"Single-agent research failed for {symbol}: {e}")
            return PredictionResult(
                model_name=self.name,
                decision="HOLD",
                confidence=0.0,
                up_probability=0.5,
                details={},
                error=str(e),
            )

    def _run_analysis(
        self,
        symbol: str,
        ohlcv_df: pd.DataFrame,
        **kwargs,
    ) -> PredictionResult:
        from models.research_backend import get_research_agent

        # as_of: explicit kwarg wins (live path), else the last bar of the
        # already-truncated frame (backtest path). This is the point-in-time
        # anchor for every data block the agent assembles.
        as_of = kwargs.get("as_of") or str(ohlcv_df.index[-1].date())
        news = kwargs.get("news")                 # optional pre-fetched PIT news
        use_news = kwargs.get("use_news", True)   # False for the news-ablation arm
        use_reflection = kwargs.get("use_reflection", False)  # opt-in memory loop
        # research_model: the Full Analysis pipeline's explicit choice (a
        # distinct key — "model" is too generic to share across all models
        # in the prediction service's common kwargs).
        model_name = (kwargs.get("research_model") or kwargs.get("model")
                      or MODEL.TRADING_AGENTS_MODEL)
        evidence = (set(kwargs["evidence"]) if kwargs.get("evidence") is not None
                    else set(MODEL.DEFAULT_EVIDENCE))
        ledger = EvidenceLedger(symbol)

        # The run's own verdict on the news SOURCE. "unavailable" means the
        # vendor failed, not that the week was quiet — and a research report
        # written blind on a news-driven name is the failure this model now
        # refuses (the sentiment model already abstains on the same signal).
        news_status = kwargs.get("news_status")
        if use_news and news_status == "unavailable":
            ledger.missing("news_source",
                           "news vendor failed for this symbol in this run")
        if use_news:
            ledger.have("news_source")

        logger.info(f"Running single-agent research for {symbol} @ {as_of}")
        _emit("ta", f"Research {symbol}: gathering point-in-time dataset "
                    f"(market, sector, news, fundamentals) @ {as_of}")

        # Precomputed, validated context: metrics from the (already truncated)
        # OHLCV, the event-calendar gate, peer relative strength, a computed
        # SPY regime, and the selected evidence blocks. Computed here — not
        # asserted by the LLM — and all lookahead-safe.
        tools = sorted(set(kwargs.get("tools") or []))
        extra_blocks, investigation = self._build_extra_context(
            symbol, ohlcv_df, as_of, evidence=evidence, ledger=ledger,
            news=news, target=kwargs.get("target_date"), tools=tools)

        # Deferred reflection: resolve any past decisions whose outcome is now
        # known (lookahead-safe — only outcomes on/before as_of) and inject the
        # recent lessons ahead of the validated blocks.
        if use_reflection:
            try:
                from services import reflection_service as _refl
                _refl.resolve_pending(symbol, as_of)
                past = _refl.get_past_context(symbol)
                if past:
                    extra_blocks.insert(0, past)
                    ledger.have("reflection")
            except Exception as e:
                logger.debug("reflection context failed: %s", e)
                ledger.missing("reflection", str(e)[:100])

        # The one accuracy sentence the report is required to quote verbatim.
        # Measured here (not phrased by the LLM) so a reader sees the platform's
        # own evaluated hit rate next to the model's self-assessed conviction,
        # and so a thin sample produces "not enough history" instead of a number.
        track_record = self._track_record_line(as_of)
        if track_record:
            ledger.have("track_record")
        else:
            ledger.missing("track_record", "calibration lookup failed")

        agent = get_research_agent(model=model_name, backend=kwargs.get("backend"))
        result = agent.analyze(
            symbol,
            as_of,
            ohlcv_df=ohlcv_df,
            news=news,
            track_record=track_record,
            # Per-block cap: one oversized block (usually news-heavy metrics)
            # must not push the later blocks (peers, SPY regime) past the
            # whole-string budget — that's how peers silently vanished.
            extra_context="\n\n".join(b[:2600] for b in extra_blocks),
            use_news=use_news,
            include_thesis=kwargs.get("include_thesis", False),
            # The run's own window, frontend-owned: no fallback here — the
            # agent raises when it is asked to read news without one.
            news_lookback_days=kwargs.get("news_lookback_days"),
            ledger=ledger,
            situation=(investigation.situation if investigation else None),
        )

        decision = (result.get("decision") or "HOLD").upper()
        # The LLM's own CONFIDENCE line is the model's self-report. Our own
        # results show it carries no calibration signal, so it is NEVER
        # propagated into scoring — only kept as labeled transparency metadata.
        stated_conviction = float(result.get("confidence") or 0.5)
        raw_response = result.get("raw_response", "")

        # Honest confidence + up_probability. Prefer a GROUNDED signal — the
        # model's realized directional hit-rate from resolved outcomes — over a
        # self-declared number. Until a track record exists we stay neutral
        # ("no declaration"), so a confident-sounding LLM cannot over-weight
        # itself in the ensemble or clear the confidence gate on words alone.
        confidence, up_probability, conf_source, emp_n = self._grounded_confidence(
            symbol, decision, kwargs.get("empirical_accuracy"),
            kwargs.get("empirical_n", 0),
        )

        gaps = ledger.expected_gaps()
        _emit("ta", f"Research {symbol}: {decision} "
                    f"[reliability {confidence:.0%} · {conf_source}] — "
                    f"{result.get('news_count', 0)} articles in window"
                    + (f"; situation {investigation.situation}" if investigation else "")
                    + (f"; WRITTEN WITHOUT {len(gaps)} expected block(s)" if gaps else ""))
        if gaps:
            # Stage "gap": the report exists, but a reader must know what it
            # was written without. The run's completeness check turns these
            # into PARTIAL; the UI counts them separately from crashes.
            _emit("gap", f"{symbol}: report written without expected evidence — "
                         f"{ledger.summary()}",
                  payload={"event": "evidence_gap", "symbol": symbol,
                           "gaps": [g.to_dict() for g in gaps]})

        # Record this decision as pending for a future reflection pass. Store the
        # LLM's stated conviction on the entry purely as a note (not for scoring).
        if use_reflection:
            try:
                from services import reflection_service as _refl
                _refl.record_pending(symbol, as_of, decision, stated_conviction,
                                     thesis=raw_response[:800])
            except Exception as e:
                logger.debug("reflection record failed: %s", e)

        details = {
            "confidence_type": conf_source,
            "stated_conviction": round(stated_conviction, 2),  # LLM self-report; uncalibrated, unused for scoring
            "empirical_n": emp_n,
            "raw_response": raw_response,
            "trade_date": as_of,
            "model": result.get("model", ""),
            "triggers": result.get("triggers", {}),
            "structured": result.get("structured", {}),
            "provenance": result.get("provenance", {}),
            "figure_check": result.get("figure_check"),
            "news_count": result.get("news_count", 0),
            "sector_etf": result.get("sector_etf", ""),
            "used_news": use_news,
            "evidence": ledger.to_dict(),
            "tools": tools,
            # Token cost of the report, summed over retries. Persisted so
            # cost per report is measurable instead of being stored as 0.
            "input_tokens": result.get("input_tokens", 0),
            "output_tokens": result.get("output_tokens", 0),
            "served_by_model": result.get("served_by_model", ""),
        }
        if investigation is not None:
            details["investigation"] = investigation.to_dict()
            details["input_tokens"] += investigation.input_tokens
            details["output_tokens"] += investigation.output_tokens

        return PredictionResult(
            model_name=self.name,
            decision=decision,
            confidence=round(confidence, 2),
            up_probability=round(up_probability, 4),
            details=details,
        )

    @staticmethod
    def _track_record_line(as_of: str) -> str:
        """The accuracy sentence for this run, or an explicit "can't say".

        Prefers this model's own evaluated history and falls back to the
        platform-wide number when the research arm alone is too thin — a
        reader is better served by "all models, n=140" than by silence. Both
        are bounded to `as_of` so a historical run cannot quote a rate
        measured on sessions it has not reached.
        """
        try:
            from services import calibration_service as cal
            own = cal.evaluated_hit_rate("trading_agents", days=90, as_of=as_of)
            if own["hit_rate"] is not None:
                return cal.track_record_sentence("trading_agents", days=90,
                                                 as_of=as_of)
            everything = cal.evaluated_hit_rate(None, days=90, as_of=as_of)
            if everything["hit_rate"] is not None:
                return (f"This research model has only {own['n']} "
                        f"scored non-HOLD calls in the last 90 days — too few to "
                        f"quote. Across all models on this platform the figure is "
                        f"{everything['hit_rate']:.0%} directionally correct "
                        f"(n={everything['n']}, through {everything['through']}). "
                        f"Coin-flip is 50%.")
            # Neither is quotable — the "not enough history" wording lives in
            # calibration_service so every surface says it the same way.
            return cal.track_record_sentence("trading_agents", days=90, as_of=as_of)
        except Exception as e:
            logger.debug(f"track record lookup failed: {e}")
            return ""

    @staticmethod
    def _grounded_confidence(
        symbol: str, decision: str,
        emp_acc: "float | None", emp_n: int,
    ) -> "tuple[float, float, str, int]":
        """Return (confidence, up_probability, source_label, n) with no fabrication.

        confidence = realized directional hit-rate when a track record exists,
        else 0.5 (neutral). up_probability leans in the decision's direction only
        by the *earned* edge (hit_rate - 0.5); with no history it stays 0.5 — the
        model declares a direction, not an invented probability.
        """
        if emp_acc is None:
            try:
                from services import reflection_service as _refl
                emp_acc, emp_n = _refl.empirical_edge(symbol)
            except Exception:
                emp_acc, emp_n = None, 0

        if emp_acc is not None:
            confidence = float(emp_acc)
            edge = max(0.0, emp_acc - 0.5)
            source = f"empirical_reliability(n={emp_n})"
        else:
            confidence = 0.5
            edge = 0.0
            source = "undeclared_neutral"

        if decision == "BUY":
            up_probability = 0.5 + edge
        elif decision == "SELL":
            up_probability = 0.5 - edge
        else:
            up_probability = 0.5
        return confidence, up_probability, source, emp_n

    def _build_extra_context(
        self, symbol: str, ohlcv_df: pd.DataFrame, as_of: str,
        evidence: "set[str] | list[str] | None" = None,
        ledger: EvidenceLedger | None = None,
        news: list | None = None,
        target: str | None = None,
        tools: "set[str] | list[str] | None" = None,
    ) -> "tuple[list[str], object | None]":
        """Assemble validated, lookahead-safe context blocks for the prompt.

        ``evidence`` gates the optional blocks per run (Run-Analysis modal
        checklist): "options", "quality", "investigation", "political".
        None means the configured default set. ``tools`` is the run's tool
        switches from the same dialog — "web_research" lets the
        investigation search the open web; absent means off.

        Returns (blocks, investigation). Every block records itself on the
        ledger as present or as a gap with the reason; required blocks
        raise through ``ledger.missing``.
        """
        evidence = set(evidence) if evidence is not None else set(MODEL.DEFAULT_EVIDENCE)
        ledger = ledger or EvidenceLedger(symbol)
        extra_blocks: list[str] = []
        investigation = None

        # Callers may pass None to request a fresh fetch (stale-cache
        # healing). The metrics/peer blocks still need a frame — fetch and
        # slice it here rather than silently dropping the "validated"
        # PRECOMPUTED METRICS block from the prompt.
        if ohlcv_df is None or not len(ohlcv_df):
            from services.stock_data import fetch_stock_data
            fresh = fetch_stock_data(symbol, period="1y")
            if fresh is not None and len(fresh):
                ohlcv_df = fresh[fresh.index <= str(as_of)]
        if ohlcv_df is None or not len(ohlcv_df):
            ledger.missing("ohlcv", f"no price bars through {as_of}")
        ledger.have("ohlcv")

        from utils.metrics import (
            compute_trading_metrics, format_metrics_block,
            compute_peer_relative_strength,
        )
        from utils.events import get_upcoming_events, format_events_block
        from models.sector_map import get_peers

        # --- validated metrics (required) ---
        try:
            metrics = compute_trading_metrics(ohlcv_df)
            block = format_metrics_block(symbol, metrics)
        except Exception as e:
            metrics, block = {}, ""
            logger.warning(f"{symbol}: metrics block failed: {e}")
        if block:
            extra_blocks.append(block)
            ledger.have("metrics")
        else:
            ledger.missing("metrics", "validated metrics could not be computed")
        last_close = None
        try:
            last_close = float(ohlcv_df["Close"].iloc[-1])
        except Exception:
            pass

        # --- event calendar (expected) ---
        try:
            events = get_upcoming_events(symbol, as_of)
            block = format_events_block(symbol, events)
        except Exception as e:
            events, block = {}, ""
            logger.warning(f"{symbol}: events block failed: {e}")
        if block:
            extra_blocks.append(block)
            ledger.have("events")
        else:
            ledger.missing("events", "earnings/ex-dividend calendar unavailable")

        # --- peers (expected; "no peers mapped" is a coverage gap worth
        # seeing, not silence) ---
        peers = get_peers(symbol)
        if peers:
            from models.sector_map import get_sector_etf
            block = None
            try:
                block = compute_peer_relative_strength(
                    symbol, peers, as_of=as_of, sector_etf=get_sector_etf(symbol))
            except Exception as e:
                logger.warning(f"{symbol}: peer RS failed: {e}")
            if block:
                extra_blocks.append(block)
                ledger.have("peers")
            else:
                logger.warning(f"{symbol}: peer RS block empty "
                               f"(peers: {', '.join(peers[:6])})")
                extra_blocks.append(
                    f"[Peer relative strength — UNAVAILABLE]\n"
                    f"Known peers: {', '.join(peers[:8])}. Their price "
                    f"data could not be fetched for this run; do not "
                    f"infer relative performance.")
                ledger.missing("peers", "peer price data could not be fetched")
        else:
            # A structural absence (no mapping), not a fetch failure: logged
            # and shown in the details, but it does not degrade the run.
            ledger.missing("peers", "no peer set mapped for this symbol",
                           severity=OPTIONAL)

        # --- SPY regime (required) ---
        try:
            from services.stock_data import fetch_stock_data
            spy = fetch_stock_data("SPY", period="2y")
            spy = spy[spy.index <= as_of]
        except Exception as e:
            spy = None
            logger.warning(f"{symbol}: SPY fetch failed: {e}")
        if spy is not None and len(spy) >= 200:
            close = float(spy["Close"].iloc[-1])
            sma50 = float(spy["Close"].rolling(50).mean().iloc[-1])
            sma200 = float(spy["Close"].rolling(200).mean().iloc[-1])
            regime = ("BULL" if close > sma50 and close > sma200
                      else "BEAR" if close < sma50 and close < sma200
                      else "MIXED")
            extra_blocks.append(
                f"[SPY regime — computed through {as_of}]\n"
                f"SPY close: ${close:.2f} | 50-day SMA: ${sma50:.2f} | "
                f"200-day SMA: ${sma200:.2f}\n"
                f"Regime by SMA rule: {regime} "
                f"(close {'above' if close > sma50 else 'below'} SMA50, "
                f"{'above' if close > sma200 else 'below'} SMA200)")
            ledger.have("spy")
        else:
            ledger.missing("spy", "fewer than 200 SPY bars available")

        # --- SEC filings + finviz market context (optional: Terminal DB) ---
        filings_text = ""
        try:
            from services.terminal_data import filings_block, market_context_block
            filings_text = filings_block(symbol, as_of)
            if filings_text:
                extra_blocks.append(filings_text)
                ledger.have("filings")
            else:
                ledger.missing("filings", "no Terminal filings rows for this window")
            block = market_context_block(symbol, as_of)
            if block:
                extra_blocks.append(block)
                ledger.have("market_context")
            else:
                ledger.missing("market_context", "no finviz snapshot")
        except Exception as e:
            logger.debug(f"terminal data blocks failed: {e}")
            ledger.missing("filings", f"Terminal data unavailable: {str(e)[:80]}")

        # --- options positioning (expected when selected) ---
        if "options" in evidence:
            try:
                from services.options_service import (
                    get_put_call_metrics, get_put_call_by_expiry,
                    format_options_block,
                )
                metrics_pc = get_put_call_metrics(symbol, as_of)
                by_expiry = get_put_call_by_expiry(symbol, as_of) if metrics_pc else None
                block = format_options_block(symbol, metrics_pc, by_expiry)
            except Exception as e:
                block = ""
                logger.warning(f"{symbol}: options block failed: {e}")
            if block:
                extra_blocks.append(block)
                ledger.have("options")
            else:
                ledger.missing("options", "no chain returned (vendor throttled, "
                                          "no listed options, or no key)")

        # --- quality screen (expected when selected) ---
        quality_text = ""
        if "quality" in evidence:
            try:
                from services.bad_apples_service import (
                    analyze_symbol as _ba_analyze, format_bad_apples_block,
                )
                quality_text = format_bad_apples_block(symbol, _ba_analyze(symbol, as_of))
            except Exception as e:
                logger.warning(f"{symbol}: quality screen failed: {e}")
            if quality_text:
                extra_blocks.append(quality_text)
                ledger.have("quality")
            else:
                ledger.missing("quality", "quality screen could not be computed")

        # --- political & institutional flows (optional: sparse by nature) ---
        if "political" in evidence:
            try:
                from services.political_service import political_blocks
                blocks, problems = political_blocks(symbol, as_of)
            except Exception as e:
                blocks, problems = [], [str(e)[:120]]
            extra_blocks.extend(blocks)
            if blocks and not problems:
                ledger.have("political")
            elif problems:
                ledger.missing("political", "; ".join(problems))

        # --- situation & investigation (expected when selected) ---
        # Placed FIRST in the prompt's precomputed section: the situation
        # decides how every other block is read.
        if "investigation" in evidence:
            web = WEB_RESEARCH_TOOL in set(tools or [])
            try:
                from services.investigation_service import (
                    investigate, format_investigation_block, headlines_for_prompt,
                )
                from services.stock_data import get_company_profile
                from services import usage_service as _usage
                _emit("ta", f"Research {symbol}: investigating the situation "
                            f"({'web research' if web else 'classification from supplied evidence only'})")
                ctx = _usage.current()
                with _usage.track("investigation", symbol=symbol,
                                  trade_date=ctx.trade_date or as_of,
                                  section=f"investigation:{symbol}"):
                    investigation = investigate(
                        symbol, as_of, web=web, target=target,
                        profile=get_company_profile(symbol) or "",
                        headlines=headlines_for_prompt(news or []),
                        filings=filings_text, quality=quality_text,
                        last_close=last_close)
                block = format_investigation_block(investigation, last_close)
                extra_blocks.insert(0, block)
                ledger.have("investigation")
                _emit("ta", f"Research {symbol}: situation {investigation.situation} "
                            f"({investigation.situation_confidence}) — "
                            f"{investigation.one_line[:140]}",
                      payload={"event": "investigation", "symbol": symbol,
                               "situation": investigation.situation,
                               "web": investigation.web,
                               "searches": investigation.searches,
                               "sources": len(investigation.sources),
                               "findings": len(investigation.findings),
                               "spread_pct": investigation.spread_pct})
            except Exception as e:
                logger.warning(f"{symbol}: investigation failed: {e}")
                investigation = None
                ledger.missing("investigation", f"investigation stage failed: {str(e)[:100]}")

        return extra_blocks, investigation

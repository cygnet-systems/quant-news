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

The raw_response (full report text) is stored in details for persistence in
DuckDB/Postgres and viewing in the History tab.
"""

import logging
import os

import pandas as pd

from config import MODEL
from models.base import BaseModel, PredictionResult

logger = logging.getLogger(__name__)


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

        logger.info(f"Running single-agent research for {symbol} @ {as_of}")
        try:
            from services import progress_service as _prog
            _prog.emit("ta", f"Research {symbol}: gathering point-in-time dataset "
                             f"(market, sector, news, fundamentals) @ {as_of}")
        except Exception:
            pass

        # Precomputed, validated context: metrics from the (already truncated)
        # OHLCV, the event-calendar gate, peer relative strength, and a computed
        # SPY regime. These are computed here — not asserted by the LLM — and
        # are all lookahead-safe.
        extra_blocks = self._build_extra_context(
            symbol, ohlcv_df, as_of, evidence=kwargs.get("evidence"))

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
            except Exception as e:
                logger.debug("reflection context failed: %s", e)

        # The one accuracy sentence the report is required to quote verbatim.
        # Measured here (not phrased by the LLM) so a reader sees the platform's
        # own evaluated hit rate next to the model's self-assessed conviction,
        # and so a thin sample produces "not enough history" instead of a number.
        track_record = self._track_record_line(as_of)

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
            extra_context="\n\n".join(b[:2000] for b in extra_blocks),
            use_news=use_news,
            include_thesis=kwargs.get("include_thesis", False),
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

        try:
            from services import progress_service as _prog
            _prog.emit("ta", f"Research {symbol}: {decision} "
                             f"[reliability {confidence:.0%} · {conf_source}] — "
                             f"{result.get('news_count', 0)} articles in window")
        except Exception:
            pass

        # Record this decision as pending for a future reflection pass. Store the
        # LLM's stated conviction on the entry purely as a note (not for scoring).
        if use_reflection:
            try:
                from services import reflection_service as _refl
                _refl.record_pending(symbol, as_of, decision, stated_conviction,
                                     thesis=raw_response[:800])
            except Exception as e:
                logger.debug("reflection record failed: %s", e)

        return PredictionResult(
            model_name=self.name,
            decision=decision,
            confidence=round(confidence, 2),
            up_probability=round(up_probability, 4),
            details={
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
                # Token cost of the report, summed over retries. Persisted so
                # cost per report is measurable instead of being stored as 0.
                "input_tokens": result.get("input_tokens", 0),
                "output_tokens": result.get("output_tokens", 0),
                "served_by_model": result.get("served_by_model", ""),
            },
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
        evidence: "list[str] | set[str] | None" = None,
    ) -> list[str]:
        """Assemble validated, lookahead-safe context blocks for the prompt.

        ``evidence`` gates the optional Terminal-derived blocks per run
        (Run-Analysis modal checklist): "options" and "quality". None means
        both — the scheduled/default path.
        """
        evidence = set(evidence) if evidence is not None else {"options", "quality"}
        extra_blocks: list[str] = []
        try:
            # Callers may pass None to request a fresh fetch (stale-cache
            # healing). The metrics/peer blocks still need a frame — fetch
            # and slice it here rather than silently dropping the
            # "validated" PRECOMPUTED METRICS block from the prompt.
            if ohlcv_df is None or not len(ohlcv_df):
                from services.stock_data import fetch_stock_data
                fresh = fetch_stock_data(symbol, period="1y")
                if fresh is not None and len(fresh):
                    ohlcv_df = fresh[fresh.index <= str(as_of)]
            from utils.metrics import (
                compute_trading_metrics, format_metrics_block,
                compute_peer_relative_strength,
            )
            from utils.events import get_upcoming_events, format_events_block
            from models.sector_map import get_peers

            metrics = compute_trading_metrics(ohlcv_df)
            block = format_metrics_block(symbol, metrics)
            if block:
                extra_blocks.append(block)

            events = get_upcoming_events(symbol, as_of)
            block = format_events_block(symbol, events)
            if block:
                extra_blocks.append(block)

            peers = get_peers(symbol)
            if peers:
                from models.sector_map import get_sector_etf
                block = compute_peer_relative_strength(
                    symbol, peers, as_of=as_of,
                    sector_etf=get_sector_etf(symbol),
                )
                if block:
                    extra_blocks.append(block)
                else:
                    # Name the gap instead of dropping the block silently —
                    # the report used to read "peers not in data" with no way
                    # to tell a fetch failure from a symbol with no peers.
                    logger.warning(f"{symbol}: peer RS block empty "
                                   f"(peers: {', '.join(peers[:6])})")
                    extra_blocks.append(
                        f"[Peer relative strength — UNAVAILABLE]\n"
                        f"Known peers: {', '.join(peers[:8])}. Their price "
                        f"data could not be fetched for this run; do not "
                        f"infer relative performance."
                    )

            # SPY regime with actual SMAs — the regime gate should not rest on
            # eyeballed price ranges.
            try:
                from services.stock_data import fetch_stock_data
                spy = fetch_stock_data("SPY", period="2y")
                spy = spy[spy.index <= as_of]
                if len(spy) >= 200:
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
                        f"{'above' if close > sma200 else 'below'} SMA200)"
                    )
            except Exception as e:
                logger.debug(f"SPY regime block failed: {e}")

            # SEC filings (8-K catalyst flags + Form 4 insider transactions)
            # and the whole-market finviz snapshot context. Both come from
            # the Terminal's nightly collectors and are point-in-time by
            # their own stamps; both degrade to absence, never to staleness.
            try:
                from services.terminal_data import (
                    filings_block, market_context_block,
                )
                block = filings_block(symbol, as_of)
                if block:
                    extra_blocks.append(block)
                block = market_context_block(symbol, as_of)
                if block:
                    extra_blocks.append(block)
            except Exception as e:
                logger.debug(f"terminal data blocks failed: {e}")

            # Options positioning (point-in-time chain) and the Bad Apples
            # quality screen. Both blocks state their own interpretation
            # limits — positioning context and risk framing, not timing.
            if "options" in evidence:
                try:
                    from services.options_service import (
                        get_put_call_metrics, format_options_block,
                    )
                    block = format_options_block(
                        symbol, get_put_call_metrics(symbol, as_of))
                    if block:
                        extra_blocks.append(block)
                except Exception as e:
                    logger.debug(f"Options block failed: {e}")

            if "quality" in evidence:
                try:
                    from services.bad_apples_service import (
                        analyze_symbol as _ba_analyze, format_bad_apples_block,
                    )
                    block = format_bad_apples_block(
                        symbol, _ba_analyze(symbol, as_of))
                    if block:
                        extra_blocks.append(block)
                except Exception as e:
                    logger.debug(f"Bad apples block failed: {e}")
        except Exception as e:
            logger.warning(f"Extra context build failed for {symbol}: {e}")
        return extra_blocks

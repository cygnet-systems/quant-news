"""Research-driven single-agent prediction model.

Wraps `models.single_agent.SingleAgentResearch`: a self-contained,
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
that cannot be built raises. The symbol fails visibly instead of producing a
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
    EXPECTED, OPTIONAL, EvidenceLedger, MissingRequiredEvidence,
)
from services.usage_service import SpendCeilingReached
from models.single_agent import MAX_EXTRA_CONTEXT_CHARS, _smart_truncate
from services.investigation_service import WEB_RESEARCH_TOOL

logger = logging.getLogger(__name__)


# Per-block ceiling. A single oversized block (usually news-heavy metrics)
# must not crowd the later ones out, but the ceiling has to clear the
# largest block written on purpose: the insider block runs to ~2.7 KB on a
# board that files often, the congressional dossier to ~2.8 KB.
PER_BLOCK_CHARS = 3400
# No block is cut below this, however tight the whole-section budget gets:
# a header, its caveat and a couple of lines still carry evidence.
MIN_BLOCK_CHARS = 600


def _block_ceiling(lengths: list[int], budget: int) -> int:
    """The largest per-block ceiling whose total fits ``budget``.

    Water-filling: raise one ceiling across every block until the sum hits
    the budget, so the excess comes off the longest blocks and the short
    ones are left whole.
    """
    ordered = sorted(lengths)
    for i, length in enumerate(ordered):
        remaining = len(ordered) - i
        kept = sum(ordered[:i])
        if kept + length * remaining > budget:
            return max(0, (budget - kept) // remaining)
    return ordered[-1]


def _fit_blocks(blocks: list[str], per_block: int = PER_BLOCK_CHARS,
                budget: int = MAX_EXTRA_CONTEXT_CHARS,
                protected: int = 0) -> str:
    """Join the context blocks so that every one of them survives.

    The old assembly sliced each block and let the whole string be cut from
    the end, which silently dropped the LAST blocks: the insider and
    congressional disclosures, appended after the technical ones and already
    recorded on the ledger as present. Here the budget is shared instead --
    each block is capped, then the longest are trimmed at a line boundary
    until the total fits -- so a block can be shortened but never disappear.

    ``protected`` counts leading blocks that give way LAST. Those are the
    anomaly sections: they are the reason the report is being written at all,
    and a water-fill that treats them like the SPY regime block would trim
    the researched finding out of the one section that says something the
    reader could not get from a chart. Everything else reaches its floor
    before they are touched.
    """
    head = [_smart_truncate(b, per_block) for b in blocks[:protected] if b]
    tail = [_smart_truncate(b, per_block) for b in blocks[protected:] if b]
    fitted = head + tail
    if not fitted:
        return ""
    # "\n\n" between blocks, plus _smart_truncate's marker on each one it
    # cuts, both come out of the same budget.
    room = budget - 2 * (len(fitted) - 1) - 16 * len(fitted)
    if sum(len(b) for b in fitted) > room:
        head_len = sum(len(b) for b in head)
        if tail:
            ceiling = max(MIN_BLOCK_CHARS,
                          _block_ceiling([len(b) for b in tail],
                                         room - head_len))
            trimmed = [b for b in tail if len(b) > ceiling]
            tail = [_smart_truncate(b, ceiling) for b in tail]
            if trimmed:
                logger.info("context budget: %d of %d blocks trimmed to "
                            "%d chars", len(trimmed), len(fitted), ceiling)
        tail_len = sum(len(b) for b in tail)
        if head and head_len + tail_len > room:
            # Everything unprotected is already at the floor; the anomaly
            # sections now share what is left rather than overflow the prompt.
            ceiling = max(MIN_BLOCK_CHARS,
                          _block_ceiling([len(b) for b in head],
                                         room - tail_len))
            logger.info("context budget: %d protected block(s) trimmed to "
                        "%d chars", len(head), ceiling)
            head = [_smart_truncate(b, ceiling) for b in head]
        fitted = head + tail
    return "\n\n".join(fitted)


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
        # research_model override still degrades gracefully, predict()
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
            # Two kinds arrive here and the activity trail names which: a
            # required block the source answered empty for, and a feed the
            # run selected that did not answer at all (FeedUnavailable).
            # Both stop this symbol's report; the payload carries the block
            # and the reason so the Activity page can group them by feed.
            logger.error(str(e))
            what = ("feed unavailable" if e.kind == "feed_unavailable"
                    else "required evidence missing")
            _emit("error",
                  f"{symbol}: research report NOT written, {what}: "
                  f"{e.reason} ({e.block}). A report without it would read "
                  f"as complete and be wrong.",
                  payload={"symbol": symbol, "block": e.block,
                           "reason": e.reason, "kind": e.kind})
            return PredictionResult(
                model_name=self.name,
                decision="HOLD",
                confidence=0.0,
                up_probability=0.5,
                details={"missing_required": e.block,
                         "missing_reason": e.reason,
                         "missing_kind": e.kind},
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
        # The SINCE LAST REPORT post-pass is one more model call per symbol.
        # It serves a reader following a live name from one morning to the
        # next; a backtest has no such reader, so it keeps the deterministic
        # record and the call is not bought.
        use_continuity = kwargs.get("use_continuity",
                                    not kwargs.get("is_backtest", False))
        # research_model: the Full Analysis pipeline's explicit choice (a
        # distinct key, "model" is too generic to share across all models
        # in the prediction service's common kwargs).
        model_name = (kwargs.get("research_model") or kwargs.get("model")
                      or MODEL.TRADING_AGENTS_MODEL)
        evidence = (set(kwargs["evidence"]) if kwargs.get("evidence") is not None
                    else set(MODEL.DEFAULT_EVIDENCE))
        ledger = EvidenceLedger(symbol)

        # The run's own verdict on the news SOURCE. "unavailable" means the
        # vendor failed, not that the week was quiet, and a research report
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
        # SPY regime, and the selected evidence blocks. Computed here: not
        # asserted by the LLM, and all lookahead-safe.
        tools = sorted(set(kwargs.get("tools") or []))
        (extra_blocks, investigation, anomalies, screened,
         scan_failed) = self._build_extra_context(
            symbol, ohlcv_df, as_of, evidence=evidence, ledger=ledger,
            news=news, target=kwargs.get("target_date"), tools=tools)

        # Deferred reflection: resolve any past decisions whose outcome is now
        # known (lookahead-safe, only outcomes on/before as_of) and inject the
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
            # Shared budget rather than a slice per block and a cut on the
            # tail: peers vanished that way once, and the disclosure blocks
            # (appended last, already on the ledger) would vanish next. The
            # anomaly blocks lead the list and are the last to give way.
            extra_context=_fit_blocks(extra_blocks, protected=len(anomalies)),
            use_news=use_news,
            include_thesis=kwargs.get("include_thesis", False),
            # The run's own window, frontend-owned: no fallback here, the
            # agent raises when it is asked to read news without one.
            news_lookback_days=kwargs.get("news_lookback_days"),
            use_continuity=use_continuity,
            ledger=ledger,
            situation=(investigation.situation if investigation else None),
            anomalies=anomalies,
            screened=screened,
            scan_failed=scan_failed,
        )

        decision = (result.get("decision") or "HOLD").upper()
        # The LLM's own CONFIDENCE line is the model's self-report. Our own
        # results show it carries no calibration signal, so it is NEVER
        # propagated into scoring, only kept as labeled transparency metadata.
        stated_conviction = float(result.get("confidence") or 0.5)
        raw_response = result.get("raw_response", "")

        # Honest confidence + up_probability. Prefer a GROUNDED signal, the
        # model's realized directional hit-rate from resolved outcomes, over a
        # self-declared number. Until a track record exists we stay neutral
        # ("no declaration"), so a confident-sounding LLM cannot over-weight
        # itself in the ensemble or clear the confidence gate on words alone.
        confidence, up_probability, conf_source, emp_n = self._grounded_confidence(
            symbol, decision, kwargs.get("empirical_accuracy"),
            kwargs.get("empirical_n", 0),
        )

        gaps = ledger.expected_gaps()
        _emit("ta", f"Research {symbol}: {decision} "
                    f"[reliability {confidence:.0%} · {conf_source}]: "
                    f"{result.get('news_count', 0)} articles in window"
                    + (f"; situation {investigation.situation}" if investigation else "")
                    + (f"; WRITTEN WITHOUT {len(gaps)} expected block(s)" if gaps else ""))
        if gaps:
            # Stage "gap": the report exists, but a reader must know what it
            # was written without. The run's completeness check turns these
            # into PARTIAL; the UI counts them separately from crashes.
            _emit("gap", f"{symbol}: report written without expected evidence, "
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
        details["anomalies"] = [
            {k: v for k, v in a.items() if k != "answer"} | {
                "citations": [c.get("url") for c in
                              (a.get("answer") or {}).get("citations", [])
                              if c.get("url")],
                "finding": (a.get("answer") or {}).get("finding", ""),
            }
            for a in anomalies
        ]
        for a in anomalies:
            answer = a.get("answer") or {}
            details["input_tokens"] += answer.get("input_tokens", 0)
            details["output_tokens"] += answer.get("output_tokens", 0)
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
        platform-wide number when the research arm alone is too thin, a
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
                        f"scored non-HOLD calls in the last 90 days, too few to "
                        f"quote. Across all models on this platform the figure is "
                        f"{everything['hit_rate']:.0%} directionally correct "
                        f"(n={everything['n']}, through {everything['through']}). "
                        f"Coin-flip is 50%.")
            # Neither is quotable, the "not enough history" wording lives in
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
        by the *earned* edge (hit_rate - 0.5); with no history it stays 0.5: the
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
    ) -> "tuple[list[str], object | None, list[dict], list[str], bool]":
        """Assemble validated, lookahead-safe context blocks for the prompt.

        ``evidence`` gates the optional blocks per run (Run-Analysis modal
        checklist): "options", "quality", "investigation", "political",
        "insiders", "politicians".
        None means the configured default set. ``tools`` is the run's tool
        switches from the same dialog, "web_research" lets the
        investigation search the open web; absent means off.

        Returns (blocks, investigation, anomalies, screened, scan_failed),
        where ``screened`` names the anomaly categories this run could
        actually rule out: an empty anomaly list says nothing about a
        category whose block was never fetched, and the report has to be
        able to tell the two apart. ``scan_failed`` separates a scan that
        raised from a scan that found nothing. Every block records itself on
        the ledger as present or as a gap with the reason; required blocks
        raise through ``ledger.missing``.
        """
        evidence = set(evidence) if evidence is not None else set(MODEL.DEFAULT_EVIDENCE)
        ledger = ledger or EvidenceLedger(symbol)
        extra_blocks: list[str] = []
        investigation = None

        # Callers may pass None to request a fresh fetch (stale-cache
        # healing). The metrics/peer blocks still need a frame, fetch and
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

        # --- event calendar (expected when it answers; a dead lookup stops
        # the report) ---
        try:
            events = get_upcoming_events(symbol, as_of)
        except Exception as e:
            logger.warning(f"{symbol}: events lookup failed: {e}")
            ledger.unavailable("events", f"calendar lookup failed: {str(e)[:80]}")
        if isinstance(events, dict) and events.get("unavailable"):
            # get_upcoming_events swallows the vendor error and says so;
            # that is the feed not answering, not a symbol with no events.
            ledger.unavailable("events", "calendar lookup failed at the vendor")
        block = format_events_block(symbol, events)
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
                # The mapping exists and the price feed did not answer for
                # it: a dead feed, not a coverage gap.
                logger.warning(f"{symbol}: peer RS block empty "
                               f"(peers: {', '.join(peers[:6])})")
                ledger.unavailable(
                    "peers", f"peer price data could not be fetched "
                             f"({', '.join(peers[:4])})")
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
                f"[SPY regime: computed through {as_of}]\n"
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
        # metrics_pc / by_expiry outlive the block: the anomaly scan below
        # reads the same numbers rather than fetching the chain a second time.
        metrics_pc = by_expiry = None
        if "options" in evidence:
            from services.options_service import (
                OptionsUnavailable, get_put_call_metrics,
                get_put_call_by_expiry, format_options_block,
            )
            try:
                metrics_pc = get_put_call_metrics(symbol, as_of)
                by_expiry = get_put_call_by_expiry(symbol, as_of) if metrics_pc else None
                block = format_options_block(symbol, metrics_pc, by_expiry)
            except OptionsUnavailable as e:
                # The chain could not be asked for. Before this, a throttle
                # and "no listed options" were one None and one gap, and a
                # report written through a throttle read as a symbol with
                # no chain.
                logger.warning(f"{symbol}: options feed unavailable: {e}")
                ledger.unavailable("options", str(e)[:120])
            except Exception as e:
                logger.warning(f"{symbol}: options block failed: {e}")
                ledger.unavailable("options", f"options block failed: {str(e)[:100]}")
            if block:
                extra_blocks.append(block)
                ledger.have("options")
            else:
                ledger.missing("options", "vendor answered with no chain: "
                                          "no listed options for this symbol")

        # --- quality screen (expected when selected) ---
        quality_text = ""
        quality_result = None
        if "quality" in evidence:
            try:
                from services.bad_apples_service import (
                    analyze_symbol as _ba_analyze, format_bad_apples_block,
                )
                quality_result = _ba_analyze(symbol, as_of,
                                             articles=(news or None))
                quality_text = format_bad_apples_block(symbol, quality_result)
            except Exception as e:
                logger.warning(f"{symbol}: quality screen failed: {e}")
                ledger.unavailable("quality", f"quality screen failed: {str(e)[:100]}")
            if isinstance(quality_result, dict) and quality_result.get("unavailable"):
                # analyze_symbol never raises; this is its word that the
                # fundamentals source did not answer and every check is n/a
                # for that reason, not because the filer is thin.
                ledger.unavailable("quality", quality_result["unavailable"])
            if quality_text:
                extra_blocks.append(quality_text)
                ledger.have("quality")
            else:
                ledger.missing("quality", "every check came back n/a for "
                                          "this filer")

        # --- political & institutional flows (optional: sparse by nature) ---
        if "political" in evidence:
            # The dossier reads the same stored congressional rows over the
            # same window and names the members who filed them, so with both
            # selected this block contributes the 13F half only. Two
            # renderings of one dataset is how a report quotes two counts.
            congress = "politicians" not in evidence
            try:
                from services.political_service import political_blocks
                blocks, problems = political_blocks(symbol, as_of,
                                                    include_congress=congress)
            except Exception as e:
                blocks, problems = [], [str(e)[:120]]
            extra_blocks.extend(blocks)
            # Same rule as the insider and dossier blocks below: a block that
            # was written is evidence the model can read, so it is recorded as
            # present even when the other half of the fetch failed. The caveat
            # rides in the block text (political_blocks appends it), not as an
            # absence of a block that is sitting in the prompt.
            if blocks:
                ledger.have("political")
            elif problems:
                # Nothing rendered and the sources said why: a dead feed,
                # whatever this block's severity for an empty window.
                ledger.unavailable("political", "; ".join(problems)[:160])
            else:
                # Nothing fetched and nothing broken: 13F rows can simply be
                # absent, and with the congressional half in the dossier
                # there is no other line for this key to write.
                ledger.missing("political",
                               "no 13F holder rows visible on or before "
                               "the as-of date"
                               + ("" if congress else
                                  "; congressional trades are in the "
                                  "dossier block"))

        # --- insider Form 4 flow (optional: many windows hold none) ---
        # Read from the local store, so this costs a vendor call at most
        # once a week per symbol however many runs ask for it.
        if "insiders" in evidence:
            try:
                from services.insider_service import insider_block
                block, problems = insider_block(symbol, as_of)
            except Exception as e:
                logger.warning(f"{symbol}: insider block failed: {e}")
                block, problems = "", [str(e)[:120]]
            # A block that was written is evidence the model can read, so
            # it is recorded as present even when a problem came back with
            # it. A partial failure (a top-up that did not run) is stated
            # inside the block text, not as an absence of the whole block.
            if block:
                extra_blocks.append(block)
                ledger.have("insiders")
            elif problems:
                # insider_block returns no text only when the top-up failed
                # AND nothing was stored to write from: the feed is down.
                ledger.unavailable("insiders", "; ".join(problems)[:160])
            else:
                ledger.missing("insiders", "no Form 4 rows stored for this symbol")

        # --- congressional dossier (optional) ---
        # Passed the run's OWN news: the dossier names members the articles
        # mention, and re-fetching that news here would be a second window
        # with a different cutoff.
        if "politicians" in evidence:
            try:
                from services.politician_dossier import politician_block
                block, problems = politician_block(symbol, as_of, news=news)
            except Exception as e:
                logger.warning(f"{symbol}: congressional dossier failed: {e}")
                block, problems = "", [str(e)[:120]]
            # Same rule as the insider block: politician_block returns a
            # rendered dossier together with problems when one of its two
            # sources went stale, and that dossier is in the prompt.
            if block:
                extra_blocks.append(block)
                ledger.have("politicians")
            elif problems:
                ledger.unavailable("politicians", "; ".join(problems)[:160])
            else:
                ledger.missing("politicians",
                               "no congressional rows stored for this symbol")

        web = WEB_RESEARCH_TOOL in set(tools or [])

        # --- situation & investigation (expected when selected) ---
        # Placed FIRST in the prompt's precomputed section: the situation
        # decides how every other block is read.
        # Detect FIRST. Detection is free arithmetic over the blocks built
        # above, and it is the only thing that can tell a symbol worth paying
        # to investigate from a quiet one before the money is spent. On a
        # 20-symbol watchlist, 14 to 16 names were buying a web search to
        # establish that nothing was happening.
        detected = self._detect_anomalies(
            symbol, as_of, evidence=evidence, ledger=ledger, news=news,
            ohlcv_df=ohlcv_df, options=metrics_pc, by_expiry=by_expiry,
            quality=quality_result)
        _anomalies_found = bool(detected[0])
        # A scan that FAILED knows nothing about whether this symbol is quiet,
        # so it must not be read as "nothing to investigate": the gate opens.
        _gate_open = (_anomalies_found or detected[2]
                      or not MODEL.INVESTIGATE_ONLY_ANOMALIES)

        if "investigation" in evidence and not web:
            # The web tool is the cost switch. Off means no investigation
            # call of any kind: the web-free classification is a model call
            # too, and it ran for every flagged symbol (and every symbol of
            # every day in a backtest, where the path strips the tool) while
            # the dialog said no research was being bought. The situation
            # line already tells the report to classify from the news and
            # filings blocks when no block was gathered.
            ledger.skip(
                "investigation",
                "web research is off for this run, so no situation research "
                "was bought; the situation is classified from the news and "
                "filings blocks instead")
            _emit("ta", f"Research {symbol}: web research off, skipping the "
                        f"investigation")
        elif "investigation" in evidence and not _gate_open:
            ledger.skip(
                "investigation",
                "nothing stood out for this symbol in "
                + (", ".join(detected[1]) or "the evidence this run gathered")
                + ", so no situation research was bought")
            _emit("ta", f"Research {symbol}: quiet symbol, skipping the paid "
                        f"investigation")
        elif "investigation" in evidence:
            try:
                from services.investigation_service import (
                    investigate, format_investigation_block, headlines_for_prompt,
                )
                from services.stock_data import get_company_profile
                from services import usage_service as _usage
                _emit("ta", f"Research {symbol}: investigating the situation "
                            f"(web research)")
                ctx = _usage.current()
                with _usage.track("investigation", symbol=symbol,
                                  trade_date=ctx.trade_date or as_of,
                                  section=f"investigation:{symbol}"):
                    investigation = investigate(
                        symbol, as_of, web=web, target=target,
                        # The scan already established something is going
                        # on with this name; a web-free triage asking
                        # whether it is plain momentum would be a second
                        # model call to answer a settled question. Triage
                        # stays for the paths that reach here without a
                        # finding (gate off, or the scan raised).
                        triage=not _anomalies_found,
                        profile=get_company_profile(symbol) or "",
                        headlines=headlines_for_prompt(news or []),
                        filings=filings_text, quality=quality_text,
                        last_close=last_close)
                block = format_investigation_block(investigation, last_close)
                extra_blocks.insert(0, block)
                ledger.have("investigation")
                _emit("ta", f"Research {symbol}: situation {investigation.situation} "
                            f"({investigation.situation_confidence}): "
                            f"{investigation.one_line[:140]}",
                      payload={"event": "investigation", "symbol": symbol,
                               "situation": investigation.situation,
                               "web": investigation.web,
                               "searches": investigation.searches,
                               "sources": len(investigation.sources),
                               "findings": len(investigation.findings),
                               "spread_pct": investigation.spread_pct})
            except SpendCeilingReached as e:
                # Not a failure of the investigator: the run refused to buy
                # the call. Saying "stage failed" here would send a vendor
                # alert for a budget decision, and the report would tell a
                # reader the evidence was unavailable when it was unbought.
                logger.warning(f"{symbol}: investigation not bought: {e}")
                investigation = None
                ledger.missing("investigation",
                               f"not researched: {e}", severity=EXPECTED)
                _emit("ta", f"Research {symbol}: spend ceiling reached, "
                            f"no investigation bought",
                      payload={"event": "spend_ceiling", "symbol": symbol,
                               "stage": "investigation"})
            except Exception as e:
                logger.warning(f"{symbol}: investigation failed: {e}")
                investigation = None
                ledger.missing("investigation", f"investigation stage failed: {str(e)[:100]}")

        anomalies, screened, scan_failed = self._scan_and_research(
            symbol, as_of, detected=detected, ledger=ledger,
            target=target, web=web)
        # Ahead of the situation block, which is itself ahead of everything
        # else: these are the only blocks in the prompt that say something a
        # reader could not read off the chart.
        if anomalies:
            from services.anomaly_service import format_anomaly_block
            for anomaly in reversed(anomalies):
                extra_blocks.insert(0, format_anomaly_block(
                    symbol, anomaly, anomaly.get("answer")))

        return extra_blocks, investigation, anomalies, screened, scan_failed

    def _detect_anomalies(
        self, symbol: str, as_of: str, *, evidence: set, ledger,
        news: list | None, ohlcv_df, options, by_expiry, quality,
    ) -> tuple[list[dict], list[str], bool]:
        """(anomalies, screened, scan_failed) with nothing researched yet.

        Split from _scan_and_research so it can run BEFORE the
        investigation stage decides whether this symbol is worth paying
        to investigate. Detection is arithmetic over blocks the run has
        already built: no fetch, no model call, no database write, so
        asking it first costs nothing and is the only thing that can
        tell a quiet symbol from a live one before the money is spent.
        """
        from services import anomaly_service
        # Gathered, not non-empty. A block is on the ledger when it RENDERED,
        # which is the only honest answer to "did this run look": the
        # congressional dossier renders a block saying nobody filed, and the
        # options and quality blocks are on the ledger only when their fetch
        # produced one. Computed before the scan so the failure path below
        # can still say what the run had in hand when it broke.
        present = set(ledger.present)
        looked = {
            "options": "options" in evidence and "options" in present,
            "by_expiry": "options" in evidence and "options" in present,
            "insiders": "insiders" in evidence and "insiders" in present,
            # Either block puts the congressional rows in the prompt: the
            # dossier when it is selected, and the political block's own
            # congressional half when it is not (political_blocks is called
            # with include_congress=False whenever the dossier is on). The
            # scan reads them from the same store either way, so the report
            # never says the filings were not looked at over a block that
            # lists them.
            "congress": (("politicians" in evidence and "politicians" in present)
                         or ("political" in evidence
                             and "politicians" not in evidence
                             and "political" in present)),
            "quality": "quality" in evidence and "quality" in present,
            # news_source is REQUIRED evidence, so reaching here with it on
            # the ledger means the window was read; an empty list from a
            # responding source is a quiet week, which IS a screen.
            "news": news is not None and "news_source" in present,
        }
        screened = anomaly_service.screened(looked, ohlcv=ohlcv_df)
        try:
            insider_summary = congress_rows = None
            if looked["insiders"]:
                from services.insider_service import summarize_insiders
                insider_summary = summarize_insiders(symbol, as_of)
            if looked["congress"]:
                from services import av_store
                congress_rows = av_store.congress_trades_for(symbol, as_of)
            anomalies = anomaly_service.detect(
                symbol, as_of, options=options, by_expiry=by_expiry,
                insiders=insider_summary, congress=congress_rows,
                quality=quality, news=news, ohlcv=ohlcv_df)
        except Exception as e:
            logger.warning(f"{symbol}: anomaly scan failed: {e}")
            return [], screened, True
        return anomalies, screened, False

    def _scan_and_research(
        self, symbol: str, as_of: str, *, detected: tuple, ledger,
        target: str | None, web: bool,
    ) -> tuple[list[dict], list[str], bool]:
        """What stands out for this symbol, each item researched on the web.

        Returns (anomalies, screened, scan_failed). ``screened`` names the
        categories the scan could actually rule out. An empty anomaly list
        means "nothing stood out in THOSE categories" and means nothing at
        all about the ones this run never fetched, so the caller needs both
        to write the quiet-symbol note without asserting an absence it did
        not check.

        ``scan_failed`` is the third state and exists because the first two
        do not cover it: a scan that raised AFTER the run gathered its
        evidence knows neither that the symbol is quiet nor that there was
        nothing to screen, and reusing the nothing-was-gathered wording for
        it would tell the report a falsehood about its own run.

        What counts as screened is read off the run's evidence set and the
        ledger, never off whether rows came back. A dossier that rendered
        "no member of Congress disclosed a trade" screened congressional
        filings; an empty list of rows is its finding, not its absence.

        ``detected`` is _detect_anomalies' result, computed by the caller
        BEFORE the investigation stage so the gate there can read it. This
        function only researches what detection already found.

        Each surviving anomaly is enriched in place with ``answer`` (its
        researched finding, or None), ``researched``, and — when it was not —
        ``unresearched``, one of anomaly_service.UNRESEARCHED_REASONS. That
        reason is stamped here because only this function knows which of the
        four happened; a block that inferred it from a missing answer would
        put a false statement about the run's own provenance in the report.
        """
        anomalies, screened, scan_failed = detected
        if scan_failed:
            return anomalies, screened, scan_failed
        if not anomalies:
            # Hoisted: nesting the same quote inside an f-string expression
            # is 3.12+ grammar and the container runs 3.11.
            looked_at = ", ".join(screened) or "any evidence this run gathered"
            _emit("ta", f"Research {symbol}: nothing stands out in "
                        f"{looked_at}, the report stays short")
            return anomalies, screened, False

        for anomaly in anomalies:
            anomaly["answer"] = None
            anomaly["researched"] = False
            anomaly["unresearched"] = "no_web" if not web else "capped"

        _emit("ta", f"Research {symbol}: {len(anomalies)} anomaly/anomalies to "
                    f"write about ("
                    + ", ".join(a["title"] for a in anomalies) + ")",
              payload={"event": "anomalies", "symbol": symbol,
                       "keys": [a["key"] for a in anomalies]})
        if not web:
            return anomalies, screened, False

        try:
            from services.investigation_service import research_questions
            from services import usage_service as _usage
            ctx = _usage.current()
            with _usage.track("investigation", symbol=symbol,
                              trade_date=ctx.trade_date or as_of,
                              section=f"anomaly_research:{symbol}"):
                # The figures that raised each question travel WITH it.
                # Without them the researcher was handed a one-line question
                # and told in writing that no supporting figures were
                # supplied, so it searched the ticker instead of the
                # anomaly: the insider names, put/call volumes, expiry dates
                # and disclosure sizes that make the question answerable are
                # exactly what these facts carry.
                answers = research_questions(
                    symbol, as_of, [a["question"] for a in anomalies],
                    web=True, target=target,
                    context_by_question={a["question"]: "\n".join(a["facts"])
                                         for a in anomalies})
        except Exception as e:
            logger.warning(f"{symbol}: anomaly research failed: {e}")
            for anomaly in anomalies:
                anomaly["unresearched"] = "stage_failed"
            return anomalies, screened, False

        by_question = {a["question"]: a for a in answers}
        for anomaly in anomalies:
            answer = by_question.get(anomaly["question"])
            if answer and answer.get("finding") and not answer.get("error"):
                anomaly["answer"] = answer
                anomaly["researched"] = True
                anomaly.pop("unresearched", None)
            elif answer:
                # A failed search still travels: the block prints the reason
                # instead of reading as though the question was never asked.
                anomaly["answer"] = answer
                anomaly["unresearched"] = "failed"
            # No entry at all means research_questions never asked it: the
            # per-symbol cap or the run's ceiling took it, which is what
            # "capped" (stamped above) says.
        researched = sum(1 for a in anomalies if a["researched"])
        _emit("ta", f"Research {symbol}: researched {researched} of "
                    f"{len(anomalies)} anomaly question(s) on the web")
        return anomalies, screened, False

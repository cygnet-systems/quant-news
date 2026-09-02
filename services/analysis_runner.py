"""Headless Full Analysis, the dashboard's pipeline without a browser.

The Full Analysis button fans out across four Dash callbacks
(``generate_model_signals`` → ``persist_predictions`` → ``generate_ai_analysis``
→ ``generate_recommendations_callback``). A scheduled run needs the same
sequence with no renderer attached, so the stages live here and app.py keeps
only the wiring. Anything that reads a UI control takes a keyword argument
whose default is what the Full Analysis modal ships with.

Ownership: nothing signs in, so every row lands anonymous, ``owner_uid`` NULL
and ``is_public`` true, i.e. the public user, visible to everyone.

Lookahead rules are the ones the callbacks enforce and are not relaxed here:
the picked date is the TARGET session, all data is cut off at the previous
trading day, and news is a point-in-time window ending at that cutoff.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import date as date_cls
from datetime import datetime
from io import StringIO
from typing import Iterable, Optional

import pandas as pd

from config import MODEL
from services.news_window import RunParameterMissing

logger = logging.getLogger(__name__)

ALL_MODELS: tuple[str, ...] = (
    "kronos_mini",
    "xgboost_shap",
    "lightgbm",
    "deberta_sentiment",
    "trading_agents",
)

# Stamped onto every prediction and required to match before one is reused.
# Bump this whenever a change alters what the models are FED, a data-quality
# fix, a new feature block, a different news window. Version numbers already
# cover code changes inside a model; this covers everything upstream of them,
# which is exactly what a stored prediction cannot tell you about itself.
#
# 2026-08-05.1: sector-lookup race fixed. Predictions written before this
# may hold sector features that are a duplicate of SPY.
# 2026-08-08.1: research prompts now carry options positioning (put/call)
# and the Bad Apples quality screen; earlier research predictions were made
# without them.
# 2026-08-08.2: quality screen adds the PIT news red-flag scan (leadership/
# layoffs/legal/guidance/short-seller/dilution) as an 18th check.
# 2026-08-08.3: quality screen grows to 20 checks (short interest, analyst
# revision momentum); evidence-set selection stamped per prediction.
# 2026-08-11.1: feature set v5 (news_present, atr_percentile,
# dist_52wk_high_pct; ATR-scaled training labels); ensemble votes run through
# calibrated confidence + rolling-performance decay; synthesis emits scored
# p_correct instead of conviction labels; peer blocks add relative valuation;
# options positioning carries day-over-day flow; SPY regime stamped on every
# prediction; report figures audited against their prompt.
# 2026-08-11.2: research prompts gain SEC filings (8-K events + Form 4
# insider transactions) and finviz market-context blocks, from the
# Terminal's nightly collectors (point-in-time by filing/snapshot stamps).
# 2026-08-23.1: research prompts gain the measured-accuracy block and the
# prior-stance continuity block, news items carry their source, and the report
# sections are reordered to lead with the symbol's own evidence. A prediction
# written before this was made from materially different inputs.
# 2026-09-02.1: research prompts open with a situation & investigation block
# (web-researched on live runs: deal terms, regulators, key figures), gain
# political/institutional flows and by-expiry put/call, lead with a Situation
# section, and carry an evidence ledger; the default news window is 14 days.
# Reports written before this were blind to the situation a symbol was in.
PIPELINE_EPOCH = "2026-09-02.1"

# Conviction labels map to nominal confidences for display only, backtests
# showed they carry no calibration signal, so the label stays in details.
_CONVICTION_CONF = {"HIGH": 0.8, "MEDIUM": 0.6, "LOW": 0.4}
_STANCE = {"BUY": "BULLISH", "SELL": "BEARISH", "HOLD": "NEUTRAL"}


def merge_research_into_analysis(
    ai_analysis: dict | None,
    model_signals: dict | None,
    symbols: list | None,
) -> tuple[dict, bool]:
    """Backfill per-symbol research (and its derived stance fields) from the
    trading_agents signal into an AI-analysis dict.

    Full Analysis produces its research text through the prediction pipeline
    only; consumers that read ai_analysis["by_symbol"] (the synthesis step,
    PDF report) get the same report the user sees on screen. Entries that
    already carry text analysis are untouched. Returns (dict, changed).
    """
    ai_analysis = ai_analysis or {}
    by_sym = dict(ai_analysis.get("by_symbol") or {})
    changed = False
    for sym in (symbols or []):
        entry = dict(by_sym.get(sym) or {})
        if entry.get("research"):
            continue
        sig = ((model_signals or {}).get(sym) or {}).get("trading_agents") or {}
        det = (sig.get("details") or {}) if isinstance(sig, dict) else {}
        if not det.get("raw_response"):
            continue
        st = det.get("structured") or {}
        entry["research"] = {
            "decision": sig.get("decision"),
            # Two different numbers travel together from here on: the
            # track-record weight the model layer computed (0.5 = no record
            # yet) and the report's own stated conviction. Downstream surfaces
            # kept conflating them because only one was ever carried.
            "confidence": sig.get("confidence"),
            "stated_conviction": det.get("stated_conviction"),
            "confidence_type": det.get("confidence_type"),
            "raw_response": det["raw_response"],
            "triggers": det.get("triggers") or {},
            "structured": st,
            "provenance": det.get("provenance") or {},
            "model": det.get("model", ""),
            # The situation the report was written under and what it was
            # written without: the synthesis reads both.
            "investigation": det.get("investigation") or {},
            "evidence": det.get("evidence") or {},
        }
        if "recommendation" not in entry:
            entry["recommendation"] = (st.get("stance")
                                       or _STANCE.get(sig.get("decision"), "NEUTRAL"))
            entry["confidence"] = sig.get("confidence")
            entry["stated_conviction"] = det.get("stated_conviction")
            entry["stance_source"] = "research_verdict"
            if st.get("sentiment_alignment"):
                entry["sentiment_explanation"] = st["sentiment_alignment"]
            if st.get("watch_items"):
                entry["watch_items"] = st["watch_items"]
            if st.get("company_thesis"):
                entry["company_thesis"] = st["company_thesis"]
            if det.get("provenance"):
                entry["provenance"] = det["provenance"]
        by_sym[sym] = entry
        changed = True
    if not changed:
        return ai_analysis, False
    return {**ai_analysis, "by_symbol": by_sym}, True


def _positioning_delta(sym: str, as_of: str, today: dict) -> dict | None:
    """Day-over-day put/call OI shift from the warmer's previous-session
    snapshot. None when yesterday wasn't warmed. The delta earns its place
    once the system runs daily, and is honestly absent until then."""
    try:
        from services import terminal_cache
        from utils.trading_calendar import get_previous_trading_day

        prev_session = get_previous_trading_day(as_of).isoformat()
        prev = terminal_cache.get_putcall_summary(sym, prev_session)
        if not prev:
            return None

        def _ratio(snapshot: dict) -> float | None:
            call_oi = sum(e.get("call_oi") or 0 for e in snapshot.get("expiries", []))
            put_oi = sum(e.get("put_oi") or 0 for e in snapshot.get("expiries", []))
            return (put_oi / call_oi) if call_oi else None

        prev_ratio = _ratio(prev)
        today_ratio = (today.get("put_call_oi_ratio")
                       or _ratio(today) if isinstance(today, dict) else None)
        if prev_ratio is None or today_ratio is None:
            return None
        return {
            "prev_session": prev_session,
            "put_call_oi_prev": round(prev_ratio, 3),
            "put_call_oi_now": round(float(today_ratio), 3),
            "shift": round(float(today_ratio) - prev_ratio, 3),
        }
    except Exception as e:
        logger.debug(f"{sym}: positioning delta skipped: {e}")
        return None


def attach_positioning_quality(
    ai_analysis: dict | None,
    symbols: list | None,
    as_of: str,
    evidence: Optional[Iterable[str]] = None,
    news_by_symbol: Optional[dict] = None,
) -> dict:
    """Stash options positioning and the Bad Apples quality screen into
    ``ai_analysis["by_symbol"]`` so the synthesis prompt, the on-screen
    report, the PDF and the markdown export all read one payload.

    Idempotent: symbols that already carry both keys (a cached report from a
    run that had them) are skipped; both services cache per (symbol, as_of),
    so a re-attach after a cache restore costs one fetch per missing symbol.
    """
    ai_analysis = ai_analysis or {}
    evidence = set(evidence) if evidence is not None else set(MODEL.DEFAULT_EVIDENCE)
    by_sym = dict(ai_analysis.get("by_symbol") or {})
    for sym in (symbols or []):
        entry = dict(by_sym.get(sym) or {})
        if "options" in evidence and "positioning" not in entry:
            try:
                from services.options_service import get_put_call_metrics
                entry["positioning"] = get_put_call_metrics(sym, as_of)
            except Exception as e:
                logger.warning(f"{sym}: options positioning failed: {e}")
                entry["positioning"] = None
        if "options" in evidence and entry.get("positioning") \
                and "positioning_delta" not in entry:
            # Flow over time, not just a snapshot: compare against the
            # previous session's WARMED summary only (cache read, no
            # network): a live re-fetch labeled "yesterday" would be
            # current-chain data wearing a costume.
            entry["positioning_delta"] = _positioning_delta(
                sym, as_of, entry["positioning"])
        # Recompute when absent OR when the stored dict predates the news
        # red-flag scan, a cached report must not freeze an older screen.
        stale_quality = (isinstance(entry.get("quality"), dict)
                         and "red_flags" not in entry["quality"])
        if "quality" in evidence and ("quality" not in entry or stale_quality):
            try:
                from services.bad_apples_service import analyze_symbol, summarize
                entry["quality"] = summarize(analyze_symbol(
                    sym, as_of,
                    articles=((news_by_symbol or {}).get(sym) or None)))
            except Exception as e:
                logger.warning(f"{sym}: quality screen failed: {e}")
                entry["quality"] = entry.get("quality") or None
        by_sym[sym] = entry

    if symbols:
        try:
            from services import progress_service as prog

            def _sym_summary(entry: dict) -> dict:
                pos = entry.get("positioning") or {}
                quality = entry.get("quality") or {}
                return {
                    "options": ("ok" if pos.get("pc_volume") is not None
                                else "missing"),
                    "pc_volume": pos.get("pc_volume"),
                    "pc_oi": pos.get("pc_oi"),
                    "quality": ((str(quality.get("flag") or "")).upper()
                                or "missing"),
                    "quality_fails": quality.get("total_fails"),
                }

            prog.emit("action",
                      f"Evidence attached ({', '.join(sorted(evidence)) or 'none'}) "
                      f"for {len(symbols)} symbols",
                      payload={
                          "event": "enrichment",
                          "as_of": as_of,
                          "evidence": sorted(evidence),
                          "by_symbol": {s: _sym_summary(by_sym.get(s) or {})
                                        for s in symbols},
                      })
        except Exception as e:
            logger.debug(f"enrichment emit skipped: {e}")

    return {**ai_analysis, "by_symbol": by_sym}


# ---------------------------------------------------------------------------
# Stage 1: market data
# ---------------------------------------------------------------------------

def load_market_data(
    symbols: Iterable[str],
    period: str = "2y",
    force_refresh: bool = False,
) -> dict:
    """Build the stock-data-store payload the prediction/report stages expect.

    Same shape as ``fetch_stock_data_callback``: full daily history as JSON
    plus window metrics and signals. ``get_stock_prices`` writes the bars to
    Postgres on the way through, which is also what lets the 6pm evaluation
    find closes to score against.
    """
    from services import progress_service as prog
    from services.analytics import (
        add_indicators_to_df, calculate_performance_metrics, get_latest_signals,
    )
    from services.cache_service import get_cache

    cache = get_cache()
    data: dict[str, dict] = {}
    bars_by_symbol: dict[str, dict] = {}
    for symbol in symbols:
        try:
            df, metadata = cache.get_stock_prices(
                symbol, period, force_refresh=force_refresh)
            df = add_indicators_to_df(df)
            data[symbol] = {
                "prices": df.to_json(date_format="iso"),
                "metrics": calculate_performance_metrics(df),
                "signals": get_latest_signals(df),
                "period": period,
                "from_cache": metadata.get("from_cache", False),
                "api_error": metadata.get("api_error"),
            }
            bars_by_symbol[symbol] = {
                "bars": len(df),
                "start": str(df.index.min())[:10] if len(df) else None,
                "end": str(df.index.max())[:10] if len(df) else None,
                "source": ("cache" if metadata.get("from_cache") else "fetch"),
            }
        except Exception as e:
            logger.warning(f"{symbol}: price fetch failed: {e}")
            bars_by_symbol[symbol] = {"bars": 0, "error": str(e)[:120]}
    prog.emit("action", f"Market data loaded: {len(data)}/{len(bars_by_symbol)} "
                        f"symbols ({period})",
              payload={
                  "event": "data_load",
                  "period": period,
                  "force_refresh": force_refresh,
                  "by_symbol": bars_by_symbol,
              })
    return data


# ---------------------------------------------------------------------------
# Stage 2: model predictions
# ---------------------------------------------------------------------------

def run_predictions(
    symbols: list[str],
    stock_data: dict,
    news_by_symbol: dict,
    target_date: date_cls,
    cutoff_date: date_cls,
    models: Optional[set[str]] = None,
    research_model: Optional[str] = None,
    news_status_by_symbol: Optional[dict[str, str]] = None,
    include_thesis: bool = True,
    force: bool = False,
    evidence: Optional[Iterable[str]] = None,
    news_lookback_days: Optional[int] = None,
    ensemble_config: Optional[dict] = None,
    run_ensemble: bool = True,
    tools: Optional[Iterable[str]] = None,
) -> dict:
    """Run every model for each symbol as of ``cutoff_date``.

    The ONE model stage: the Run dialog's background callback and the
    scheduled run both call this (the dialog passes its ensemble controls
    through ``ensemble_config``/``run_ensemble``).

    A symbol already analysed for this cutoff, against the same closing bar,
    is reused rather than re-run. The research call is an LLM round trip per
    symbol, so repeating a run that nothing has changed under is pure spend.
    ``force`` re-runs regardless.

    ``evidence`` selects the optional Terminal-derived context blocks
    ("options", "quality") from the Run-Analysis modal; None means both.
    ``tools`` is the dialog's Tools section ("web_research" lets the
    investigation search the open web); None means none, the backend
    never switches a tool on by itself.
    """
    from services import progress_service as prog
    from services import usage_service as usage
    from services.prediction_service import get_prediction_service
    from services.stock_data import fetch_stock_data as fetch_ohlcv

    if news_lookback_days is None:
        raise RunParameterMissing("run_predictions called without news_lookback_days")
    selected = set(models or ALL_MODELS)
    evidence_set = (sorted(evidence) if evidence is not None
                    else sorted(MODEL.DEFAULT_EVIDENCE))
    tools_set = sorted(set(tools or []))
    service = get_prediction_service()
    is_backtest = target_date < date_cls.today()

    spy_df = None
    spy_regime = None
    try:
        spy_df = fetch_ohlcv("SPY", period="2y")
        spy_df = spy_df[spy_df.index <= str(cutoff_date)]
    except Exception as e:
        logger.warning(f"SPY fetch failed: {e}")

    # Stamp the market regime each prediction was made under, so performance
    # can later be sliced by regime instead of reconstructed archaeologically.
    if spy_df is not None and len(spy_df) >= 200:
        try:
            close = float(spy_df["Close"].iloc[-1])
            sma50 = float(spy_df["Close"].rolling(50).mean().iloc[-1])
            sma200 = float(spy_df["Close"].rolling(200).mean().iloc[-1])
            ret20 = spy_df["Close"].pct_change().tail(20)
            vol20 = float(ret20.std() * (252 ** 0.5) * 100)
            spy_regime = {
                "state": ("BULL" if close > sma50 and close > sma200
                          else "BEAR" if close < sma50 and close < sma200
                          else "MIXED"),
                "spy_vol20_ann_pct": round(vol20, 1),
            }
        except Exception as e:
            logger.debug(f"regime stamp skipped: {e}")

    needs_historical = bool(selected & {"xgboost_shap", "lightgbm"})
    training_news_failures: list[str] = []
    historical_global_news: dict = {}
    if needs_historical:
        try:
            from services.news_service import fetch_historical_av_news
            # This is the run's longest silent phase on a cold cache, say so
            # before it starts instead of letting the feed go dark.
            prog.emit("news", f"Fetching historical news for training (up to "
                              f"{len(symbols)} symbols × "
                              f"{MODEL.NEWS_LOOKBACK_MONTHS} months: can take "
                              f"minutes on first run)…")
            historical_global_news = fetch_historical_av_news(
                "", months=MODEL.NEWS_LOOKBACK_MONTHS, as_of=str(cutoff_date))
        except Exception as e:
            logger.warning(f"Historical global news fetch failed: {e}")
            training_news_failures.append("_GLOBAL")

    # Web investigations overlap the model loop instead of extending it
    # (see investigation_service.prefetch_many), but NOT its first symbol.
    # The first symbol that runs is where every lazy model loads (torch,
    # transformers, the DeBERTa weights) while XGBoost and LightGBM train
    # beside it; starting N investigation workers, each pulling yfinance
    # histories and statements, on top of that spike is what took the
    # container down three times on 2026-09-02. The pool starts once that
    # symbol is through and covers the rest; symbol one investigates
    # in-loop, one thread wide, as it did before the prefetch existed.
    from services.investigation_service import WEB_RESEARCH_TOOL
    investigation_pool = None
    prefetch_pending = ("investigation" in evidence_set
                        and "trading_agents" in selected)
    web = WEB_RESEARCH_TOOL in tools_set

    def _start_prefetch(remaining: list[str]) -> None:
        nonlocal investigation_pool, prefetch_pending
        prefetch_pending = False
        if not remaining:
            return
        try:
            from services.investigation_service import prefetch_many
            investigation_pool = prefetch_many(
                remaining, str(cutoff_date), web=web,
                target=str(target_date), news_by_symbol=news_by_symbol,
                workers=MODEL.INVESTIGATION_WORKERS)
            prog.emit("ta", f"Investigating the remaining {len(remaining)} "
                            f"symbols in the background "
                            f"({MODEL.INVESTIGATION_WORKERS} at a time; "
                            f"web research {'ON' if web else 'off'})")
        except Exception as e:
            logger.warning(f"investigation prefetch not started: {e}")

    prog.emit_memory("model stage setup", symbols=len(symbols))
    sym_list = list(symbols)
    results: dict = {}
    skipped: dict[str, str] = {}
    for i, symbol in enumerate(sym_list):
        prices_json = (stock_data.get(symbol) or {}).get("prices")
        if not prices_json:
            logger.warning(f"{symbol}: no price data, skipping")
            skipped[symbol] = "no price data"
            continue
        try:
            df = pd.read_json(StringIO(prices_json))
        except Exception as e:
            logger.warning(f"{symbol}: price frame unreadable: {e}")
            skipped[symbol] = "price frame unreadable"
            continue
        if df.empty:
            logger.warning(f"{symbol}: empty price frame, skipping")
            skipped[symbol] = "empty price frame"
            continue

        # The target session's own bar must never be visible, live, that bar
        # is today's partial intraday print.
        if "Date" in df.columns:
            df = df[df["Date"] <= str(cutoff_date)]
        else:
            df = df[df.index <= str(cutoff_date)]
        if df.empty:
            logger.warning(f"{symbol}: no data available as of {cutoff_date}")
            skipped[symbol] = f"no data as of {cutoff_date}"
            continue

        sym_news = news_by_symbol.get(symbol) or []

        if not force:
            reused = _reusable_predictions(
                symbol, cutoff_date, selected, df,
                research_model=research_model, news_count=len(sym_news),
                evidence=evidence_set, news_lookback_days=news_lookback_days,
                include_thesis=include_thesis, tools=tools_set)
            if reused:
                results[symbol] = reused
                prog.emit("models", f"{symbol}: reusing {len(reused)} stored "
                                    f"predictions for {cutoff_date} (unchanged data)",
                          payload={
                              "event": "cache",
                              "kind": "predictions",
                              "outcome": "hit",
                              "symbol": symbol,
                              "models": sorted(reused),
                              "summary": {
                                  "cutoff": str(cutoff_date),
                                  "news_count": len(sym_news),
                                  "pipeline_epoch": PIPELINE_EPOCH,
                              },
                          })
                continue

        historical_av_news: dict = {}
        if needs_historical:
            try:
                from services.news_service import fetch_historical_av_news
                historical_av_news = fetch_historical_av_news(
                    symbol, months=MODEL.NEWS_LOOKBACK_MONTHS,
                    as_of=str(cutoff_date))
            except Exception as e:
                logger.warning(f"{symbol}: historical AV news fetch failed: {e}")
                training_news_failures.append(symbol)

        # "The source failed" and "the window was quiet" both yield zero
        # articles. The difference rides into the models (the research arm
        # refuses to write blind) and onto the stored prediction.
        sym_status = (news_status_by_symbol or {}).get(
            symbol, "ok" if sym_news else "empty")

        mode = "backtest" if is_backtest else "live"
        prog.emit("models", f"{symbol}: running {len(selected)} models "
                            f"({mode}, as-of {cutoff_date})")
        # The only LLM call under here is the research report, so its spend
        # lands against this symbol.
        with usage.track("research", symbol=symbol, trade_date=str(cutoff_date),
                         section=f"research:{symbol}"):
            results[symbol] = service.predict_symbol_no_store(
                symbol, df, spy_df=spy_df,
                news=sym_news,
                ensemble_config=ensemble_config,
                models_to_run=selected,
                run_ensemble=run_ensemble,
                historical_av_news=historical_av_news,
                historical_global_news=historical_global_news,
                as_of=str(cutoff_date),
                research_model=research_model,
                include_thesis=include_thesis,
                evidence=evidence_set,
                news_lookback_days=news_lookback_days,
                news_status=sym_status,
                target_date=str(target_date),
                tools=tools_set,
            )
        if prefetch_pending:
            prog.emit_memory("first symbol's models (weights loaded)",
                             symbol=symbol)
            _start_prefetch(sym_list[i + 1:])
        # Stamp what these were produced under, so a later run can tell
        # whether they are still valid rather than assuming they are.

        # A news-dependent model that could not read the news must not call a
        # direction: the call would be scored as if informed, poisoning both
        # the scoreboard and calibration. Quiet weeks ("empty") do not
        # abstain; only source failure does.
        from config import MODEL as _MODEL
        if (_MODEL.ABSTAIN_ON_NEWS_UNAVAILABLE
                and sym_status == "unavailable"):
            for news_model in ("deberta_sentiment",):
                entry = results[symbol].get(news_model)
                if (isinstance(entry, dict) and not entry.get("error")
                        and entry.get("decision") != "HOLD"):
                    prog.emit("models",
                              f"{symbol}: {news_model} abstained: news "
                              f"source unavailable, blind {entry['decision']} "
                              f"withheld")
                    entry.update(decision="HOLD", confidence=0.0,
                                 up_probability=0.5)
                    entry.setdefault("details", {})
                    if isinstance(entry["details"], dict):
                        entry["details"]["abstained"] = "news_unavailable"

        for entry in results[symbol].values():
            if isinstance(entry, dict) and not entry.get("error"):
                entry["news_status"] = sym_status
                entry.setdefault("details", {})
                if isinstance(entry["details"], dict):
                    entry["details"]["pipeline_epoch"] = PIPELINE_EPOCH
                    entry["details"]["evidence"] = evidence_set
                    entry["details"]["tools"] = tools_set
                    entry["details"].setdefault("news_count", len(sym_news))
                    entry["details"].setdefault("news_status", sym_status)
                    # What the research model was FED, so reuse can tell a
                    # 30-day/deep run from a 3-day/standard one that happened
                    # to see the same article count.
                    entry["details"]["news_window_days"] = news_lookback_days
                    entry["details"]["include_thesis"] = bool(include_thesis)
                    if spy_regime:
                        entry["details"]["regime"] = spy_regime

        decisions = {m: r.get("decision") for m, r in results[symbol].items()
                     if isinstance(r, dict) and not r.get("error")}
        prog.emit("models", f"{symbol}: " + ", ".join(
            f"{m}={d}" for m, d in decisions.items()))
        # One summary line per symbol when anything failed, the decisions
        # line above only lists the survivors. Abstentions (a deliberate
        # no-call) are not failures.
        failed = {
            m: (r.get("error") or "") for m, r in results[symbol].items()
            if isinstance(r, dict) and r.get("error")
            and not (r.get("details") or {}).get("abstained")
        }
        if failed:
            ran = len(results[symbol]) - len(failed)
            prog.emit("error",
                      f"{symbol}: {ran} of {len(results[symbol])} models ran: "
                      + ", ".join(f"{m} failed ({err[:60]})"
                                  for m, err in failed.items()))

    if investigation_pool is not None:
        investigation_pool.shutdown(wait=False)
    prog.emit_memory("model loop", symbols=len(results))
    results["_meta"] = {
        "predict_date": cutoff_date.isoformat(),
        "target_date": target_date.isoformat(),
        "is_backtest": is_backtest,
        "skipped_symbols": skipped,
        "training_news_failures": training_news_failures,
    }
    return results


def _reusable_predictions(
    symbol: str,
    cutoff_date: date_cls,
    selected: set[str],
    df: pd.DataFrame,
    research_model: Optional[str],
    news_count: int,
    evidence: Optional[list[str]] = None,
    news_lookback_days: Optional[int] = None,
    include_thesis: Optional[bool] = None,
    tools: Optional[list[str]] = None,
) -> Optional[dict]:
    """Stored predictions for this exact analysis, or None to re-run.

    A cache that returns a *nearly* right answer is worse than no cache: the
    predictions are the product, and a stale one is indistinguishable from a
    fresh one once stored. So reuse is deliberately conservative, every one
    of these must hold, and anything unrecognised re-runs:

    * every selected model (plus the ensemble) already has a row for this cutoff
    * the close those rows were made against still equals the close we hold.
      a vendor revision to the cutoff bar invalidates them
    * the research row came from the SAME model that was asked for, so
      switching ``--report-model`` cannot serve the other model's verdict
    * the news window has not grown since. The vendor indexes late articles
      for a session, and a verdict written on 30 articles is not the verdict
      the same prompt gives on 50
    * the pipeline epoch matches; bumping ``PIPELINE_EPOCH`` after a change to
      what the models are FED (not just how they are called) invalidates
      every stored prediction, because nothing else in this key would move
    """
    from services.cache_service import get_cache

    try:
        stored = get_cache().get_predictions_for_today(symbol, prediction_date=cutoff_date)
    except Exception as e:
        logger.debug(f"{symbol}: reuse lookup failed: {e}")
        return None

    if not stored or not (selected | {"ensemble"}) <= set(stored):
        return None

    try:
        current_close = float(df["Close"].iloc[-1])
    except Exception:
        return None

    for name, entry in stored.items():
        prev = entry.get("previous_close")
        if prev is None or abs(float(prev) - current_close) > 0.005:
            logger.info(
                f"{symbol}: stored predictions were made against close "
                f"{prev} but the cutoff bar now reads {current_close:.2f}: re-running"
            )
            return None
        if (entry.get("details") or {}).get("pipeline_epoch") != PIPELINE_EPOCH:
            logger.info(f"{symbol}: {name} predates pipeline epoch "
                        f"{PIPELINE_EPOCH}: re-running")
            return None

    research = (stored.get("trading_agents") or {}).get("details") or {}
    if research_model and research.get("model") not in (None, "", research_model):
        logger.info(f"{symbol}: stored research came from "
                    f"{research.get('model')}, asked for {research_model}: re-running")
        return None

    # The evidence set is part of what the research model was FED, a report
    # written with the quality screen must not be served to a run that
    # excluded it (and vice versa).
    if evidence is not None and "trading_agents" in stored:
        # Rows from this epoch but before the stamp existed were always fed
        # both blocks: that's what the default was.
        stored_evidence = research.get("evidence") or ["options", "quality"]
        if sorted(stored_evidence) != sorted(evidence):
            logger.info(f"{symbol}: stored research evidence "
                        f"{stored_evidence} != requested {sorted(evidence)} "
                        f": re-running")
            return None
    # Same for the tools: a report investigated with web access is not the
    # report a run without it would have produced (and vice versa).
    if tools is not None and "trading_agents" in stored:
        stored_tools = research.get("tools") or []
        if sorted(stored_tools) != sorted(tools):
            logger.info(f"{symbol}: stored research tools {stored_tools} != "
                        f"requested {sorted(tools)}: re-running")
            return None

    # A blind prediction must not be cached indefinitely: if the source was
    # down when it was made, re-running is exactly what we want next time.
    if research.get("news_status") == "unavailable":
        logger.info(f"{symbol}: stored prediction was made without news "
                    f"(source unavailable): re-running")
        return None

    stored_news = research.get("news_count")
    if stored_news is not None and int(stored_news) != int(news_count):
        logger.info(f"{symbol}: news window changed ({stored_news} → "
                    f"{news_count} articles): re-running")
        return None
    # Same count is not the same window: 7 days at 08:00 and overnight at
    # 09:00 can both hold six articles. Rows from before these stamps
    # existed carry None and are re-run once.
    for key, wanted in (("news_window_days", news_lookback_days),
                        ("include_thesis", include_thesis)):
        if "trading_agents" in stored and research.get(key) != wanted:
            logger.info(f"{symbol}: stored research {key}="
                        f"{research.get(key)!r}, asked for {wanted!r}: re-running")
            return None

    # Only the models this run asked for. The stored set for a date also holds
    # `recommendation_synthesis`: the synthesis step's OWN output, persisted as a
    # prediction so it gets scored. Handing that back as a model signal would
    # feed the synthesis step its own previous verdict as independent evidence.
    wanted = selected | {"ensemble"}
    return {name: {k: v for k, v in entry.items() if k != "previous_close"}
            for name, entry in stored.items() if name in wanted}


def persist_predictions(signals: dict) -> tuple[int, int]:
    """Write predictions (and research reports) to Postgres.

    Returns ``(stored, evaluated)``, backtest runs are scored immediately,
    exactly as ``persist_predictions`` does in the app.
    """
    from services import progress_service as prog
    from services.cache_service import get_cache

    signals = dict(signals)
    meta = signals.pop("_meta", {})
    predict_date_str = meta.get("predict_date")
    cache = get_cache()

    stored = 0
    for symbol, model_results in signals.items():
        if not isinstance(model_results, dict):
            continue
        for model_name, result_dict in model_results.items():
            if not isinstance(result_dict, dict) or result_dict.get("error"):
                continue
            cache.store_prediction(
                symbol, model_name, result_dict,
                prediction_date_str=predict_date_str,
            )
            stored += 1

            if model_name == "trading_agents":
                details = result_dict.get("details", {})
                raw_response = details.get("raw_response", "")
                if raw_response:
                    cache.save_trading_agent_report(
                        symbol=symbol,
                        trade_date=details.get("trade_date", ""),
                        decision=result_dict.get("decision", "HOLD"),
                        confidence=result_dict.get("confidence", 0.0),
                        report_text=raw_response,
                        # The model that answered, which after a provider
                        # fallback is not the one that was asked.
                        model_name=(details.get("served_by_model")
                                    or details.get("model", "")),
                        input_tokens=details.get("input_tokens", 0),
                        output_tokens=details.get("output_tokens", 0),
                    )

    evaluated = cache.evaluate_predictions() if meta.get("is_backtest") else 0
    prog.emit("store", f"Stored {stored} predictions"
                       + (f", evaluated {evaluated}" if evaluated else ""))
    return stored, evaluated


# ---------------------------------------------------------------------------
# Stage 3: AI report
# ---------------------------------------------------------------------------

def portfolio_metrics_block(symbols: list[str], stock_data: dict, as_of: str) -> str:
    """One VALIDATED METRICS block covering every symbol in the run.

    Live quote fields are withheld from this flow (they would expose the
    session being predicted), which left the portfolio call with no numbers at
    all: it saw an empty financial block and reported a broken data feed.
    These metrics come from OHLCV truncated to the cutoff, so they are both
    real and lookahead-safe.
    """
    from utils.metrics import compute_trading_metrics, format_metrics_block

    blocks = []
    for symbol in symbols:
        prices_json = (stock_data.get(symbol) or {}).get("prices")
        if not prices_json:
            continue
        try:
            df = pd.read_json(StringIO(prices_json))
            if "Date" in df.columns:
                df = df.set_index("Date")
            df.index = pd.to_datetime(df.index)
            df = df[df.index <= as_of]
            block = format_metrics_block(symbol, compute_trading_metrics(df))
            if block:
                blocks.append(block)
        except Exception as e:
            logger.warning(f"{symbol}: metrics block failed: {e}")
    return "\n\n".join(blocks)


def build_ai_report(
    symbols: list[str],
    news_by_symbol: dict,
    cutoff_date: date_cls,
    report_model: str,
    recs_model: str,
    stock_data: Optional[dict] = None,
    lookback_days: Optional[int] = None,
    include_thesis: bool = True,
    evidence: Optional[Iterable[str]] = None,
    max_articles: Optional[int] = None,
    overnight: bool = False,
) -> dict:
    """Portfolio-level AI analysis for the run.

    The Full Analysis flow skips the shallow per-symbol pass, each symbol's
    text analysis IS its research report, which arrives on the trading_agents
    signal and is merged in later. Only the portfolio-wide call runs here.
    Identity (name/sector/industry) comes from the live profile; every NUMBER
    comes from the truncated-OHLCV metrics block, never from a live quote.
    """
    from services import progress_service as prog
    from services import usage_service as usage
    from services.llm_service import get_llm
    from services.stock_data import get_stock_info

    if lookback_days is None or max_articles is None:
        raise RunParameterMissing("build_ai_report called without the news "
                                  "window / article cap")
    as_of = cutoff_date.isoformat()
    result: dict = {
        "overall": None,
        "by_symbol": {},
        "as_of": as_of,
        "generated_at": datetime.now().isoformat(),
        # What this report was built from. The input-data export and the
        # renderers read it back instead of assuming a window.
        "news_window": {"lookback_days": lookback_days, "overnight": overnight,
                        "max_articles": int(max_articles)},
    }

    enriched: dict = {}
    for symbol in symbols:
        entry = {"metrics": {}, "signals": {}, "info": {}}
        try:
            info = get_stock_info(symbol)
            entry["info"] = {
                "name": info.name, "sector": info.sector, "industry": info.industry,
            }
        except Exception as e:
            logger.warning(f"Could not fetch stock info for {symbol}: {e}")
        enriched[symbol] = entry

    all_articles: list = []
    for articles in news_by_symbol.values():
        all_articles.extend(articles or [])

    metrics_block = portfolio_metrics_block(symbols, stock_data or {}, as_of)

    # Same cache key the AI Report callback uses, so a scheduled run and a UI
    # run over identical scope share one entry instead of each paying.
    cache_key = report_cache_key(news_by_symbol, symbols, as_of, report_model,
                                 lookback_days, include_thesis,
                                 evidence=evidence, max_articles=max_articles,
                                 overnight=overnight)
    try:
        from services import persistence_service as ps
        cache_hash = ps.compute_data_hash(cache_key)
        cache_payload = {
            "event": "cache",
            "kind": "ai_report",
            "input_data_hash": cache_hash,
            "summary": _report_key_summary(news_by_symbol, symbols, cache_key),
        }
        cached = ps.get_cached_report(None, as_of, "ai_report", cache_hash)
        if cached:
            prog.emit("ai", "AI report cache hit, regeneration skipped",
                      payload={**cache_payload, "outcome": "hit"})
            restored = json.loads(cached)
            restored["from_cache"] = True
            restored["recs_request"] = "news+signals"
            restored["recs_model"] = recs_model
            return restored
        # A miss on inputs that look unchanged is a diagnosable event now:
        # the payload records the hash and what fed it, so a cross-restart
        # miss can be compared against the hit that should have happened.
        prog.emit("ai", "AI report cache miss, generating fresh",
                  payload={**cache_payload, "outcome": "miss"})
    except Exception as e:
        logger.debug(f"AI report cache check failed: {e}")

    if all_articles:
        prog.emit("ai", f"Overall: synthesizing {len(all_articles)} articles "
                        f"across {len(symbols)} symbols ({report_model})…")
        provider = "openai" if report_model.startswith("gpt-") else "anthropic"
        try:
            with usage.track("ai_report", trade_date=as_of,
                             section="ai_report:overall"):
                result["overall"] = get_llm().summarize_news_structured(
                    all_articles, symbols,
                    stock_data=enriched,
                    as_of_date=as_of,
                    extra_blocks={"metrics": metrics_block} if metrics_block else None,
                    # Depth used to key the cache here and never reach the
                    # call, "Deep" was a cache miss with a Standard report.
                    include_thesis=include_thesis,
                    model=report_model,
                    provider=provider,
                )
        except Exception as e:
            logger.warning(f"Overall AI analysis failed: {e}")
            prog.emit("error", f"Overall AI analysis failed: {str(e)[:80]}")

    result["recs_request"] = "news+signals"
    result["recs_model"] = recs_model
    if not result["overall"]:
        result["failed"] = True
    return result


def report_cache_key(news_by_symbol: dict, symbols: list[str], as_of: str,
                     report_model: str, lookback_days: int,
                     include_thesis: bool,
                     evidence: Optional[Iterable[str]] = None,
                     max_articles: Optional[int] = None,
                     overnight: bool = False,
                     include_research: bool = False,
                     recs_mode: str = "auto",
                     tools: Optional[Iterable[str]] = None) -> dict:
    """The ONE set of AI-report cache inputs. The Run dialog and the
    scheduled run both build their key here; when each had its own copy an
    overnight run or a non-default recommendations basis could never share
    the other's entry (and one of them silently mis-keyed the window)."""
    from services.news_window import normalize_article_cap
    return {
        "news": news_by_symbol,
        "symbols": sorted(symbols),
        "as_of": as_of,
        "schema": "v5-flowq",
        "model": report_model,
        "lookback": "overnight" if overnight else lookback_days,
        "max_articles": normalize_article_cap(max_articles),
        "thesis": include_thesis,
        "research": include_research,
        "recs": recs_mode,
        "evidence": (sorted(evidence) if evidence is not None
                     else sorted(MODEL.DEFAULT_EVIDENCE)),
        "tools": sorted(set(tools or [])),
    }


def _report_key_summary(news_by_symbol: dict, symbols: list[str],
                        cache_key: dict) -> dict:
    """A compact fingerprint of what fed the report cache hash.

    Instrumentation only: the hash itself is untouched. The news window is
    the volatile input (a late-indexed article shifts the hash), so the
    per-symbol counts and the newest article timestamp are what make a
    cross-restart miss diagnosable.
    """
    newest = None
    for arts in (news_by_symbol or {}).values():
        for a in arts or []:
            ts = a.get("published_at")
            if ts and (newest is None or ts > newest):
                newest = ts
    return {
        "symbols": sorted(symbols),
        "articles": sum(len(v or []) for v in (news_by_symbol or {}).values()),
        "articles_by_symbol": {s: len(v or [])
                               for s, v in (news_by_symbol or {}).items()},
        "newest_article": newest,
        **{k: cache_key.get(k) for k in
           ("as_of", "model", "lookback", "thesis", "research", "recs",
            "evidence", "schema")},
    }


# ---------------------------------------------------------------------------
# Stage 4: recommendation synthesis
# ---------------------------------------------------------------------------

RECOMMENDATION_KEY_VOLATILE = frozenset(
    {"generated_at", "from_cache", "run_seq", "scope"})


def run_recommendations(
    ai_analysis: dict,
    signals: dict,
    symbols: list[str],
    trade_date: str,
) -> Optional[dict]:
    """Synthesize predictions + research into per-symbol actions and persist.

    Each action is also stored as a ``recommendation_synthesis`` prediction so
    the evaluation loop, History tab and Scoreboard score the synthesis like
    any other model.
    """
    from services import persistence_service as ps
    from services import progress_service as prog
    from services import usage_service as usage
    from services.cache_service import get_cache
    from services.llm_service import get_llm

    valid_signals = {k: v for k, v in (signals or {}).items()
                     if k != "_meta" and isinstance(v, dict)}

    basis = ai_analysis.get("recs_request") or "news+signals"
    if basis != "signals":
        ai_analysis, backfilled = merge_research_into_analysis(
            ai_analysis, valid_signals, symbols)
        if backfilled and basis == "news+signals":
            basis = "research+signals"

    rec_data_hash = None
    try:
        # Hash the EVIDENCE, not the envelope. `generated_at` (and the
        # `from_cache` marker a cached report carries) change on every run,
        # so including them made the key unique every time and the cache
        # could never hit, the most expensive call in the run repeated on
        # identical inputs.
        # Envelope only. recs_model / recs_request are INPUTS: excluding
        # them served one synthesis model's answer to a run that asked for
        # another. run_seq/scope are the dialog's per-click correlation keys.
        volatile = RECOMMENDATION_KEY_VOLATILE
        rec_data_hash = ps.compute_data_hash({
            "ai_analysis": {k: v for k, v in ai_analysis.items()
                            if k not in volatile},
            "model_signals": valid_signals,
            "symbols": sorted(symbols),
        })
        rec_cache_payload = {
            "event": "cache",
            "kind": "recommendation",
            "input_data_hash": rec_data_hash,
            "summary": {
                "trade_date": trade_date,
                "symbols": sorted(symbols),
                "signal_models": sorted({m for v in valid_signals.values()
                                         for m in v if not m.startswith("_")}),
                "report_generated_at": ai_analysis.get("generated_at"),
            },
        }
        cached = ps.get_cached_recommendation(trade_date, rec_data_hash)
        if cached:
            # Identical evidence in, identical synthesis out, re-asking is
            # the single most expensive call in the run.
            prog.emit("luna", "Recommendation cache hit, synthesis skipped",
                      payload={**rec_cache_payload, "outcome": "hit"})
            cached["from_cache"] = True
            return cached
        prog.emit("luna", "Recommendation cache miss, synthesizing fresh",
                  payload={**rec_cache_payload, "outcome": "miss"})
    except Exception as e:
        logger.debug(f"Recommendation cache check failed: {e}")

    prog.emit("luna", f"Synthesis ({ai_analysis.get('recs_model')}) over "
                      f"{len(symbols)} symbols…")
    synthesis_started = time.time()
    with usage.track("recommendations", trade_date=trade_date,
                     section="recommendations"):
        result = get_llm().generate_recommendations(
            ai_analysis, valid_signals, symbols,
            basis=basis,
            model_override=ai_analysis.get("recs_model"),
        )
    synthesis_ms = int((time.time() - synthesis_started) * 1000)
    if not result:
        prog.emit("error", "Synthesis model returned empty response, "
                           "synthesis failed")
        return None

    result["generated_at"] = datetime.now().isoformat()
    result["as_of"] = trade_date
    # The window the report was built with rides along so the archive's
    # Data link can rebuild exactly those inputs.
    result["news_window"] = ai_analysis.get("news_window")

    cache = get_cache()
    for sym, rec in (result.get("by_symbol") or {}).items():
        action = (rec.get("action") or "").upper()
        if action not in ("BUY", "SELL", "HOLD"):
            logger.warning(
                f"{sym}: synthesis action {rec.get('action')!r} is not "
                f"BUY/SELL/HOLD, dropping this symbol's recommendation")
            continue
        # p_correct (scored probability) replaced the HIGH/MEDIUM/LOW
        # conviction labels: measured over 4 weeks, the labels carried no
        # calibration signal (HIGH hit 60%, MEDIUM 64%). A probability can be
        # scored against outcomes and calibrated; a mood cannot. The label
        # path stays as fallback for a model that ignores the new schema.
        p_correct = rec.get("p_correct")
        try:
            p_correct = float(p_correct) if p_correct is not None else None
            if p_correct is not None and not 0.4 <= p_correct <= 0.95:
                logger.warning(f"{sym}: synthesis p_correct {p_correct} out of "
                               f"range: clamping")
                p_correct = min(0.95, max(0.4, p_correct))
        except (TypeError, ValueError):
            p_correct = None
        if p_correct is None:
            conviction = str(rec.get("conviction", "")).upper()
            if conviction not in _CONVICTION_CONF:
                logger.warning(f"{sym}: synthesis gave neither p_correct nor a "
                               f"known conviction ({conviction!r}): treating "
                               f"as LOW")
            p_correct = _CONVICTION_CONF.get(conviction, _CONVICTION_CONF["LOW"])
        try:
            cache.store_prediction(
                sym, "recommendation_synthesis",
                {
                    "decision": action,
                    "confidence": p_correct,
                    "up_probability": None,
                    "details": {
                        "synthesis_model": result.get("model_used"),
                        "basis": result.get("basis"),
                        "p_correct": rec.get("p_correct"),
                        "conviction": rec.get("conviction"),
                        "key_level": rec.get("key_level"),
                        "change_trigger": rec.get("change_trigger"),
                        "reasoning": (rec.get("reasoning") or "")[:400],
                    },
                },
                prediction_date_str=trade_date,
            )
        except Exception as e:
            logger.warning(f"recommendation persist failed for {sym}: {e}")

    if rec_data_hash:
        try:
            # The model that actually answered, not the one configured, a
            # fallback run used to be indistinguishable from a primary one.
            ps.store_recommendation(
                trade_date=trade_date,
                symbols=symbols,
                input_data_hash=rec_data_hash,
                result=result,
                model_used=result.get("model_used") or MODEL.RECOMMENDATIONS_MODEL,
                provider_used=(result.get("provider_used")
                               or MODEL.RECOMMENDATIONS_PROVIDER),
                duration_ms=synthesis_ms,
            )
        except Exception as e:
            logger.warning(f"Failed to persist recommendation: {e}")

    actions = {s: v.get("action") for s, v in (result.get("by_symbol") or {}).items()}
    prog.emit("luna", "Synthesis finished, " + (", ".join(
        f"{s}={a}" for s, a in actions.items()) or "done"))
    return result


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_full_analysis(
    symbols: list[str],
    target: Optional[str] = None,
    lookback_days: Optional[int] = None,
    report_model: Optional[str] = None,
    recs_model: Optional[str] = None,
    include_thesis: bool = True,
    models: Optional[set[str]] = None,
    period: str = "2y",
    force_refresh: bool = True,
    force: bool = False,
    news_filter: Optional[str] = None,
    evidence: Optional[Iterable[str]] = None,
    max_articles: Optional[int] = None,
    tools: Optional[Iterable[str]] = None,
    recs_mode: str = "auto",
    ensemble_config: Optional[dict] = None,
    run_ensemble: bool = True,
) -> dict:
    """Run the whole Full Analysis pipeline for ``symbols``.

    ``recs_mode`` (auto | signals | off), ``ensemble_config`` and
    ``run_ensemble`` are the Run dialog's remaining settings, carried by a
    scheduled job's params so a job is a saved dialog, not a subset of it.

    ``target`` is the session whose close is being predicted (defaults to the
    next unresolved session); data is cut off at the previous trading day.
    ``news_filter`` selects the news window formula ("lookback" | "overnight",
    default from config). ``lookback_days`` and ``max_articles`` (newest N per
    symbol, 0 = all) are REQUIRED. They come from the job's params or the
    CLI, never from a default here. Returns a summary dict, the durable
    output is in Postgres.
    """
    if lookback_days is None or max_articles is None:
        raise RunParameterMissing(
            "run_full_analysis needs lookback_days and max_articles from the "
            "job params / CLI (got "
            f"lookback_days={lookback_days!r}, max_articles={max_articles!r})")
    from services import progress_service as prog
    from services.news_window import (
        describe_news_window, fetch_run_news, news_window_payload,
        normalize_article_cap, normalize_lookback)
    from utils.trading_calendar import resolve_target_and_cutoff

    report_model = report_model or MODEL.REPORT_MODEL
    recs_model = recs_model or MODEL.RECOMMENDATIONS_MODEL
    news_filter = (news_filter or MODEL.NEWS_FILTER_MODE).lower()
    target_date, cutoff_date = resolve_target_and_cutoff(target)
    as_of = cutoff_date.isoformat()
    started = datetime.now()

    prog.start_run(f"Full Analysis (scheduled), {len(symbols)} symbols, "
                   f"target {target_date} (data through {as_of})")

    prog.emit_memory("run start", symbols=len(symbols))
    stock_data = load_market_data(symbols, period=period, force_refresh=force_refresh)
    prog.emit_memory("market data")
    priced = [s for s in symbols if s in stock_data]
    if not priced:
        prog.finish_run("Full Analysis aborted, no price data")
        return {"error": "no price data", "symbols": symbols}

    max_articles = normalize_article_cap(max_articles)
    _, lookback_days = normalize_lookback(lookback_days)
    news_by_symbol, news_stats = fetch_run_news(
        priced, as_of, target_date.isoformat(),
        overnight=(news_filter == "overnight"),
        lookback_days=lookback_days, max_articles=max_articles)
    news_unavailable = [s for s in priced
                        if news_stats[s]["status"] == "unavailable"]
    news_empty = [s for s in priced if news_stats[s]["status"] == "empty"]
    for sym in news_unavailable:
        logger.warning("%s: news unavailable: %s", sym, news_stats[sym].get("error"))

    payload = news_window_payload(
        overnight=(news_filter == "overnight"), lookback_days=lookback_days,
        max_articles=max_articles, as_of=as_of,
        target=target_date.isoformat(), stats_by_symbol=news_stats)
    prog.emit("news", describe_news_window(payload), payload=payload)

    # "The source was down" and "the week was quiet" produce the same empty
    # list but mean opposite things, so they are reported separately. Without
    # this the models run on nothing and the report presents the hole as a
    # finding ("no company-specific news to confirm the technicals").
    if news_unavailable:
        prog.emit("error",
                  f"News source unavailable for {len(news_unavailable)} of "
                  f"{len(priced)} symbols: {', '.join(news_unavailable)}. "
                  f"Their sentiment and research models ran WITHOUT news; "
                  f"treat those calls as unsupported, not as bearish-neutral.")
    if news_empty:
        prog.emit("news", f"No articles in window for: {', '.join(news_empty)} "
                          f"(source responded; the window is genuinely empty)")

    # Archive exactly what the models are about to see. Predictions used to
    # keep only a news COUNT, so no past call could ever be audited against
    # its inputs ("what did the model read that morning?" had no answer).
    try:
        from services import persistence_service as ps
        snapshot = {
            sym: [{k: a.get(k) for k in
                   ("url", "title", "published_at", "source",
                    "ticker_relevance_score", "overall_sentiment_score")}
                  for a in arts]
            for sym, arts in news_by_symbol.items()
        }
        ps.store_report(
            symbol=None, trade_date=as_of, report_type="news_snapshot",
            input_data_hash=ps.compute_data_hash(snapshot),
            content=json.dumps({
                "as_of": as_of, "lookback_days": lookback_days,
                "max_articles": max_articles,
                "news_filter": news_filter,
                "news_unavailable": news_unavailable,
                "news_empty": news_empty,
                "window_stats": payload["by_symbol"],
                "articles": snapshot,
            }, default=str),
            file_format="json",
        )
    except Exception as e:
        logger.warning(f"news snapshot not archived: {e}")

    news_status_by_symbol = {sym: news_stats[sym]["status"] for sym in priced}
    prog.emit_memory("news fetch", articles=sum(
        len(v) for v in news_by_symbol.values()))

    signals = run_predictions(
        priced, stock_data, news_by_symbol,
        target_date=target_date, cutoff_date=cutoff_date,
        models=models, research_model=report_model,
        include_thesis=include_thesis, force=force,
        news_status_by_symbol=news_status_by_symbol,
        evidence=evidence,
        news_lookback_days=lookback_days,
        ensemble_config=ensemble_config,
        run_ensemble=run_ensemble,
        tools=tools,
    )
    stored, evaluated = persist_predictions(signals)
    prog.emit_memory("predictions persisted")

    ai_analysis = build_ai_report(
        priced, news_by_symbol, cutoff_date,
        report_model=report_model, recs_model=recs_model,
        stock_data=stock_data, lookback_days=lookback_days,
        include_thesis=include_thesis, evidence=evidence,
        max_articles=max_articles, overnight=(news_filter == "overnight"),
    )
    # After the cache check on purpose: a restored report predating these
    # sections gets them attached instead of silently lacking them.
    prog.emit_memory("report")
    ai_analysis = attach_positioning_quality(ai_analysis, priced, as_of,
                                             evidence=evidence,
                                             news_by_symbol=news_by_symbol)

    # The basis the dialog offers: text analysis + predictions, predictions
    # only, or no synthesis at all.
    ai_analysis["recs_request"] = ("signals" if recs_mode == "signals"
                                   else "news+signals")
    if recs_mode == "off":
        prog.emit("luna", "Recommendations off for this job — synthesis skipped")
        recommendations = None
    else:
        recommendations = run_recommendations(ai_analysis, signals, priced, as_of)
    prog.emit_memory("synthesis")

    # Synthesis rows are written AFTER persist_predictions' auto-evaluate, so
    # on a backtest they used to stay "pending" until the next scheduled
    # evaluation: every other model's row scored, the synthesis row didn't
    # (bitten twice: the stuck 08-06 PANW row, then all 5 in the random-5
    # QA run). Idempotent, so re-evaluating here is cheap.
    if recommendations:
        try:
            from services.cache_service import get_cache
            late = get_cache().evaluate_predictions()
            if late:
                logger.info(f"Post-synthesis evaluation: {late} rows scored")
        except Exception as e:
            logger.warning(f"post-synthesis evaluation failed: {e}")

    # The archived JSON is the only durable copy of the portfolio-level report
    #: predictions and recommendations have their own tables, this does not.
    # It used to fail silently at INFO, so an object-storage outage cost every
    # report with nothing to show it had happened.
    archived = False
    archive_error = None
    try:
        from services import persistence_service as ps
        merged, _ = merge_research_into_analysis(ai_analysis, signals, priced)
        ps.store_report(
            symbol=None, trade_date=as_of, report_type="ai_report",
            input_data_hash=ps.compute_data_hash(report_cache_key(
                news_by_symbol, priced, as_of, report_model,
                lookback_days, include_thesis, evidence=evidence,
                max_articles=max_articles,
                overnight=(news_filter == "overnight"), tools=tools)),
            content=json.dumps(merged, default=str, indent=2),
            file_format="json",
        )
        archived = True
    except Exception as e:
        archive_error = str(e)
        logger.error(f"AI report NOT archived, it is unrecoverable once this "
                     f"process exits: {e}")
        prog.emit("error", f"AI report not archived: {str(e)[:120]}")

    coverage, degraded = _assess_completeness(
        signals, priced, models, recommendations)
    if not archived:
        degraded.append(f"report not archived ({(archive_error or '')[:80]})")
    if news_unavailable:
        degraded.append(
            f"news source down for {len(news_unavailable)}/{len(priced)}: "
            f"{', '.join(news_unavailable)}")
    meta = signals.get("_meta") or {}
    if meta.get("training_news_failures"):
        degraded.append("training news failed for: "
                        + ", ".join(meta["training_news_failures"]))
    if meta.get("skipped_symbols"):
        degraded.append("symbols skipped: " + "; ".join(
            f"{s} ({why})" for s, why in meta["skipped_symbols"].items()))

    if degraded:
        prog.emit("error", "Full Analysis incomplete, " + "; ".join(degraded))
        prog.finish_run("Full Analysis PARTIAL, " + "; ".join(degraded))
    else:
        prog.finish_run("Full Analysis complete (scheduled)")

    return {
        "symbols": priced,
        "skipped": [s for s in symbols if s not in priced],
        "news_unavailable": news_unavailable,
        "news_empty": news_empty,
        "target_date": target_date.isoformat(),
        "as_of": as_of,
        "is_backtest": signals.get("_meta", {}).get("is_backtest", False),
        "predictions_stored": stored,
        "evaluated": evaluated,
        "actions": {s: v.get("action")
                    for s, v in ((recommendations or {}).get("by_symbol") or {}).items()},
        "model_coverage": coverage,
        "report_archived": archived,
        "degraded": degraded,
        "duration_s": round((datetime.now() - started).total_seconds(), 1),
    }


def _assess_completeness(
    signals: dict,
    symbols: list[str],
    models: Optional[set[str]],
    recommendations: Optional[dict],
) -> tuple[dict, list[str]]:
    """How much of the run actually landed, and what is missing.

    A run that ships four of six models and no synthesis is not a success.
    it is a partial result that happens to have exited zero. Every model
    failure here is caught per model by design (one broken model must not
    lose the other five), so without an explicit completeness check the only
    signal left is the exit code, which says nothing about coverage. That is
    how two dead models and a truncated synthesis ran unnoticed for a full
    cycle in production.

    Returns ``(coverage, degraded)``, symbols scored per model, and a list
    of human-readable reasons, empty when the run is whole.
    """
    expected = set(models or ALL_MODELS) | {"ensemble"}
    coverage = {name: 0 for name in expected}
    abstained: list[str] = []
    for symbol in symbols:
        for name, result in (signals.get(symbol) or {}).items():
            if name not in coverage or not isinstance(result, dict):
                continue
            if not result.get("error"):
                coverage[name] += 1
            elif (result.get("details") or {}).get("abstained"):
                # A model declining for lack of input (DeBERTa on a
                # quiet-news symbol) is a covered symbol with no opinion,
                # not a hole in the run. Counting it as incomplete made
                # every thin-news day mail a partial.
                coverage[name] += 1
                abstained.append(f"{name}:{symbol}")

    n = len(symbols)
    degraded: list[str] = []
    dead = sorted(m for m, scored in coverage.items() if scored == 0)
    partial = sorted(m for m, scored in coverage.items() if 0 < scored < n)
    if dead:
        degraded.append(f"scored nothing: {', '.join(dead)}")
    if partial:
        degraded.append("incomplete: " + ", ".join(
            f"{m} {coverage[m]}/{n}" for m in partial))
    if abstained:
        logger.info("abstentions (no-input, not failures): "
                    + ", ".join(sorted(abstained)))
    # Reports that exist but were written without evidence the run was
    # configured to gather. Not a crash, but not a whole run either.
    from services.evidence_contract import gaps_from_details
    gap_notes = []
    for symbol in symbols:
        research = (signals.get(symbol) or {}).get("trading_agents") or {}
        gaps = gaps_from_details(research.get("details")) if isinstance(research, dict) else []
        if gaps:
            gap_notes.append(f"{symbol} ({', '.join(g['label'] for g in gaps)})")
    if gap_notes:
        degraded.append("research written without expected evidence: "
                        + "; ".join(gap_notes))

    by_symbol = (recommendations or {}).get("by_symbol") or {}
    if not by_symbol:
        degraded.append("no recommendations produced")
    else:
        # The synthesis LLM can silently omit symbols from by_symbol; an
        # emptiness check alone never notices.
        missing = sorted(set(symbols) - set(by_symbol))
        if missing:
            degraded.append(
                f"synthesis missing {len(missing)}/{len(symbols)} symbols: "
                f"{', '.join(missing)}")
    return coverage, degraded


def evaluate_pending() -> int:
    """Score every prediction whose target session has closed.

    Two stages, both of which the old in-app evaluation trigger covered
    between them: score the raw predictions against actual closes, then run
    the registered strategies and refresh their metrics. Scoring alone would
    leave the strategy scoreboard frozen.
    """
    from services import progress_service as prog
    from services.cache_service import get_cache

    count = get_cache().evaluate_predictions()
    prog.emit("action", f"Scheduled evaluation: {count} predictions scored")

    if count:
        # New outcomes change what every confidence has historically meant.
        try:
            from services.calibration_service import invalidate
            invalidate()
        except Exception:
            pass

    try:
        from services.evaluation_service import get_evaluation_service
        results = get_evaluation_service().run_evaluation()
        scored = sum(results.values()) if results else 0
        if scored:
            prog.emit("action", f"Strategy evaluation: {scored} new evaluations")
            logger.info(f"Strategy evaluation: {results}")
    except Exception as e:
        logger.error(f"Strategy evaluation failed: {e}")

    return count



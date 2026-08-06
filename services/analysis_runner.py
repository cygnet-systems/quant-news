"""Headless Full Analysis — the dashboard's pipeline without a browser.

The Full Analysis button fans out across four Dash callbacks
(``generate_model_signals`` → ``persist_predictions`` → ``generate_ai_analysis``
→ ``generate_recommendations_callback``). A scheduled run needs the same
sequence with no renderer attached, so the stages live here and app.py keeps
only the wiring. Anything that reads a UI control takes a keyword argument
whose default is what the Full Analysis modal ships with.

Ownership: nothing signs in, so every row lands anonymous — ``owner_uid`` NULL
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

logger = logging.getLogger(__name__)

ALL_MODELS: tuple[str, ...] = (
    "kronos_mini",
    "xgboost_shap",
    "lightgbm",
    "deberta_sentiment",
    "trading_agents",
)

# Stamped onto every prediction and required to match before one is reused.
# Bump this whenever a change alters what the models are FED — a data-quality
# fix, a new feature block, a different news window. Version numbers already
# cover code changes inside a model; this covers everything upstream of them,
# which is exactly what a stored prediction cannot tell you about itself.
#
# 2026-08-05.1 — sector-lookup race fixed. Predictions written before this
# may hold sector features that are a duplicate of SPY.
PIPELINE_EPOCH = "2026-08-05.1"

# Conviction labels map to nominal confidences for display only — backtests
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
    only; consumers that read ai_analysis["by_symbol"] (Luna, XLSX export,
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
            "confidence": sig.get("confidence"),
            "raw_response": det["raw_response"],
            "triggers": det.get("triggers") or {},
            "structured": st,
            "provenance": det.get("provenance") or {},
            "model": det.get("model", ""),
        }
        if "recommendation" not in entry:
            entry["recommendation"] = (st.get("stance")
                                       or _STANCE.get(sig.get("decision"), "NEUTRAL"))
            entry["confidence"] = sig.get("confidence")
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


# ---------------------------------------------------------------------------
# Stage 1 — market data
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
    from services.analytics import (
        add_indicators_to_df, calculate_performance_metrics, get_latest_signals,
    )
    from services.cache_service import get_cache

    cache = get_cache()
    data: dict[str, dict] = {}
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
        except Exception as e:
            logger.warning(f"{symbol}: price fetch failed: {e}")
    return data


# ---------------------------------------------------------------------------
# Stage 2 — model predictions
# ---------------------------------------------------------------------------

def run_predictions(
    symbols: list[str],
    stock_data: dict,
    news_by_symbol: dict,
    target_date: date_cls,
    cutoff_date: date_cls,
    models: Optional[set[str]] = None,
    research_model: Optional[str] = None,
    include_thesis: bool = True,
    force: bool = False,
) -> dict:
    """Run every model for each symbol as of ``cutoff_date``.

    Mirrors ``generate_model_signals`` on its Full Analysis path: all models,
    ensemble on, research report produced by trading_agents.

    A symbol already analysed for this cutoff, against the same closing bar,
    is reused rather than re-run — the research call is an LLM round trip per
    symbol, so repeating a run that nothing has changed under is pure spend.
    ``force`` re-runs regardless.
    """
    from services import progress_service as prog
    from services import usage_service as usage
    from services.prediction_service import get_prediction_service
    from services.stock_data import fetch_stock_data as fetch_ohlcv

    selected = set(models or ALL_MODELS)
    service = get_prediction_service()
    is_backtest = target_date < date_cls.today()

    spy_df = None
    try:
        spy_df = fetch_ohlcv("SPY", period="2y")
        spy_df = spy_df[spy_df.index <= str(cutoff_date)]
    except Exception as e:
        logger.warning(f"SPY fetch failed: {e}")

    needs_historical = bool(selected & {"xgboost_shap", "lightgbm"})
    historical_global_news: dict = {}
    if needs_historical:
        try:
            from config import MODEL
            from services.news_service import fetch_historical_av_news
            historical_global_news = fetch_historical_av_news(
                "", months=MODEL.NEWS_LOOKBACK_MONTHS)
        except Exception as e:
            logger.warning(f"Historical global news fetch failed: {e}")

    results: dict = {}
    for symbol in symbols:
        prices_json = (stock_data.get(symbol) or {}).get("prices")
        if not prices_json:
            logger.warning(f"{symbol}: no price data, skipping")
            continue
        try:
            df = pd.read_json(StringIO(prices_json))
        except Exception as e:
            logger.warning(f"{symbol}: price frame unreadable: {e}")
            continue
        if df.empty:
            continue

        # The target session's own bar must never be visible — live, that bar
        # is today's partial intraday print.
        if "Date" in df.columns:
            df = df[df["Date"] <= str(cutoff_date)]
        else:
            df = df[df.index <= str(cutoff_date)]
        if df.empty:
            logger.warning(f"{symbol}: no data available as of {cutoff_date}")
            continue

        sym_news = news_by_symbol.get(symbol) or []

        if not force:
            reused = _reusable_predictions(
                symbol, cutoff_date, selected, df,
                research_model=research_model, news_count=len(sym_news))
            if reused:
                results[symbol] = reused
                prog.emit("models", f"{symbol}: reusing {len(reused)} stored "
                                    f"predictions for {cutoff_date} (unchanged data)")
                continue

        historical_av_news: dict = {}
        if needs_historical:
            try:
                from config import MODEL
                from services.news_service import fetch_historical_av_news
                historical_av_news = fetch_historical_av_news(
                    symbol, months=MODEL.NEWS_LOOKBACK_MONTHS)
            except Exception as e:
                logger.warning(f"{symbol}: historical AV news fetch failed: {e}")

        mode = "backtest" if is_backtest else "live"
        prog.emit("models", f"{symbol}: running {len(selected)} models "
                            f"({mode}, as-of {cutoff_date})")
        # The only LLM call under here is the research report, so its spend
        # lands against this symbol.
        with usage.track("research", symbol=symbol, trade_date=str(cutoff_date)):
            results[symbol] = service.predict_symbol_no_store(
                symbol, df, spy_df=spy_df,
                news=sym_news,
                models_to_run=selected,
                run_ensemble=True,
                historical_av_news=historical_av_news,
                historical_global_news=historical_global_news,
                as_of=str(cutoff_date),
                research_model=research_model,
                include_thesis=include_thesis,
            )
        # Stamp what these were produced under, so a later run can tell
        # whether they are still valid rather than assuming they are.
        for entry in results[symbol].values():
            if isinstance(entry, dict) and not entry.get("error"):
                entry.setdefault("details", {})
                if isinstance(entry["details"], dict):
                    entry["details"]["pipeline_epoch"] = PIPELINE_EPOCH
                    entry["details"].setdefault("news_count", len(sym_news))

        decisions = {m: r.get("decision") for m, r in results[symbol].items()
                     if isinstance(r, dict) and not r.get("error")}
        prog.emit("models", f"{symbol}: " + ", ".join(
            f"{m}={d}" for m, d in decisions.items()))

    results["_meta"] = {
        "predict_date": cutoff_date.isoformat(),
        "target_date": target_date.isoformat(),
        "is_backtest": is_backtest,
    }
    return results


def _reusable_predictions(
    symbol: str,
    cutoff_date: date_cls,
    selected: set[str],
    df: pd.DataFrame,
    research_model: Optional[str],
    news_count: int,
) -> Optional[dict]:
    """Stored predictions for this exact analysis, or None to re-run.

    A cache that returns a *nearly* right answer is worse than no cache: the
    predictions are the product, and a stale one is indistinguishable from a
    fresh one once stored. So reuse is deliberately conservative — every one
    of these must hold, and anything unrecognised re-runs:

    * every selected model (plus the ensemble) already has a row for this cutoff
    * the close those rows were made against still equals the close we hold —
      a vendor revision to the cutoff bar invalidates them
    * the research row came from the SAME model that was asked for, so
      switching ``--report-model`` cannot serve the other model's verdict
    * the news window has not grown since — the vendor indexes late articles
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
                f"{prev} but the cutoff bar now reads {current_close:.2f} — re-running"
            )
            return None
        if (entry.get("details") or {}).get("pipeline_epoch") != PIPELINE_EPOCH:
            logger.info(f"{symbol}: {name} predates pipeline epoch "
                        f"{PIPELINE_EPOCH} — re-running")
            return None

    research = (stored.get("trading_agents") or {}).get("details") or {}
    if research_model and research.get("model") not in (None, "", research_model):
        logger.info(f"{symbol}: stored research came from "
                    f"{research.get('model')}, asked for {research_model} — re-running")
        return None

    stored_news = research.get("news_count")
    if stored_news is not None and int(stored_news) != int(news_count):
        logger.info(f"{symbol}: news window changed ({stored_news} → "
                    f"{news_count} articles) — re-running")
        return None

    # Only the models this run asked for. The stored set for a date also holds
    # `recommendation_synthesis` — Luna's OWN output, persisted as a
    # prediction so it gets scored. Handing that back as a model signal would
    # feed Luna its own previous verdict as independent evidence.
    wanted = selected | {"ensemble"}
    return {name: {k: v for k, v in entry.items() if k != "previous_close"}
            for name, entry in stored.items() if name in wanted}


def persist_predictions(signals: dict) -> tuple[int, int]:
    """Write predictions (and research reports) to Postgres.

    Returns ``(stored, evaluated)`` — backtest runs are scored immediately,
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
                        model_name=details.get("model", ""),
                        input_tokens=details.get("input_tokens", 0),
                        output_tokens=details.get("output_tokens", 0),
                    )

    evaluated = cache.evaluate_predictions() if meta.get("is_backtest") else 0
    prog.emit("store", f"Stored {stored} predictions"
                       + (f", evaluated {evaluated}" if evaluated else ""))
    return stored, evaluated


# ---------------------------------------------------------------------------
# Stage 3 — AI report
# ---------------------------------------------------------------------------

def portfolio_metrics_block(symbols: list[str], stock_data: dict, as_of: str) -> str:
    """One VALIDATED METRICS block covering every symbol in the run.

    Live quote fields are withheld from this flow (they would expose the
    session being predicted), which left the portfolio call with no numbers at
    all — it saw an empty financial block and reported a broken data feed.
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
    lookback_days: int = 7,
    include_thesis: bool = True,
) -> dict:
    """Portfolio-level AI analysis for the run.

    The Full Analysis flow skips the shallow per-symbol pass — each symbol's
    text analysis IS its research report, which arrives on the trading_agents
    signal and is merged in later. Only the portfolio-wide call runs here.
    Identity (name/sector/industry) comes from the live profile; every NUMBER
    comes from the truncated-OHLCV metrics block, never from a live quote.
    """
    from services import progress_service as prog
    from services import usage_service as usage
    from services.llm_service import get_llm
    from services.stock_data import get_stock_info

    as_of = cutoff_date.isoformat()
    result: dict = {
        "overall": None,
        "by_symbol": {},
        "as_of": as_of,
        "generated_at": datetime.now().isoformat(),
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
    cache_key = _report_cache_key(news_by_symbol, symbols, as_of, report_model,
                                  lookback_days, include_thesis)
    try:
        from services import persistence_service as ps
        cached = ps.get_cached_report(None, as_of, "ai_report",
                                      ps.compute_data_hash(cache_key))
        if cached:
            prog.emit("ai", "AI report cache hit — regeneration skipped")
            restored = json.loads(cached)
            restored["from_cache"] = True
            restored["recs_request"] = "news+signals"
            restored["recs_model"] = recs_model
            return restored
    except Exception as e:
        logger.debug(f"AI report cache check failed: {e}")

    if all_articles:
        prog.emit("ai", f"Overall: synthesizing {len(all_articles)} articles "
                        f"across {len(symbols)} symbols ({report_model})…")
        provider = "openai" if report_model.startswith("gpt-") else "anthropic"
        try:
            with usage.track("ai_report", trade_date=as_of):
                result["overall"] = get_llm().summarize_news_structured(
                    all_articles, symbols,
                    stock_data=enriched,
                    as_of_date=as_of,
                    extra_blocks={"metrics": metrics_block} if metrics_block else None,
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


def _report_cache_key(news_by_symbol: dict, symbols: list[str], as_of: str,
                      report_model: str, lookback_days: int,
                      include_thesis: bool) -> dict:
    """Same hash inputs the AI Report callback uses, so a scheduled run and a
    UI run of identical scope share one cache entry."""
    return {
        "news": news_by_symbol,
        "symbols": sorted(symbols),
        "as_of": as_of,
        "schema": "v4-merged",
        "model": report_model,
        "lookback": lookback_days,
        "thesis": include_thesis,
        "research": False,
        "recs": "auto",
    }


# ---------------------------------------------------------------------------
# Stage 4 — recommendation synthesis (Luna)
# ---------------------------------------------------------------------------

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
    from config import MODEL
    from services import persistence_service as ps
    from services import progress_service as prog
    from services import usage_service as usage
    from services.cache_service import get_cache
    from services.llm_service import get_llm

    valid_signals = {k: v for k, v in (signals or {}).items()
                     if k != "_meta" and isinstance(v, dict)}

    basis = ai_analysis.get("recs_request") or "news+signals"
    ai_analysis, backfilled = merge_research_into_analysis(
        ai_analysis, valid_signals, symbols)
    if backfilled and basis == "news+signals":
        basis = "research+signals"

    rec_data_hash = None
    try:
        # Hash the EVIDENCE, not the envelope. `generated_at` (and the
        # `from_cache` marker a cached report carries) change on every run,
        # so including them made the key unique every time and the cache
        # could never hit — the most expensive call in the run repeated on
        # identical inputs.
        volatile = {"generated_at", "from_cache", "recs_model", "recs_request"}
        rec_data_hash = ps.compute_data_hash({
            "ai_analysis": {k: v for k, v in ai_analysis.items()
                            if k not in volatile},
            "model_signals": valid_signals,
            "symbols": sorted(symbols),
        })
        cached = ps.get_cached_recommendation(trade_date, rec_data_hash)
        if cached:
            # Identical evidence in, identical synthesis out — re-asking is
            # the single most expensive call in the run.
            prog.emit("luna", "Recommendation cache hit — synthesis skipped")
            cached["from_cache"] = True
            return cached
    except Exception as e:
        logger.debug(f"Recommendation cache check failed: {e}")

    prog.emit("luna", f"Luna ({ai_analysis.get('recs_model')}) synthesizing "
                      f"{len(symbols)} symbols…")
    with usage.track("recommendations", trade_date=trade_date):
        result = get_llm().generate_recommendations(
            ai_analysis, valid_signals, symbols,
            basis=basis,
            model_override=ai_analysis.get("recs_model"),
        )
    if not result:
        prog.emit("error", "Luna returned empty response — synthesis failed")
        return None

    result["generated_at"] = datetime.now().isoformat()
    result["as_of"] = trade_date

    cache = get_cache()
    for sym, rec in (result.get("by_symbol") or {}).items():
        action = (rec.get("action") or "").upper()
        if action not in ("BUY", "SELL", "HOLD"):
            continue
        try:
            cache.store_prediction(
                sym, "recommendation_synthesis",
                {
                    "decision": action,
                    "confidence": _CONVICTION_CONF.get(
                        str(rec.get("conviction", "")).upper()),
                    "up_probability": None,
                    "details": {
                        "synthesis_model": result.get("model_used"),
                        "basis": result.get("basis"),
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
            ps.store_recommendation(
                trade_date=trade_date,
                symbols=symbols,
                input_data_hash=rec_data_hash,
                result=result,
                model_used=MODEL.RECOMMENDATIONS_MODEL,
                provider_used=MODEL.RECOMMENDATIONS_PROVIDER,
            )
        except Exception as e:
            logger.warning(f"Failed to persist recommendation: {e}")

    actions = {s: v.get("action") for s, v in (result.get("by_symbol") or {}).items()}
    prog.emit("luna", "Luna finished — " + (", ".join(
        f"{s}={a}" for s, a in actions.items()) or "done"))
    return result


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_full_analysis(
    symbols: list[str],
    target: Optional[str] = None,
    lookback_days: int = 7,
    report_model: str = "gpt-5.6-luna",
    recs_model: Optional[str] = None,
    include_thesis: bool = True,
    models: Optional[set[str]] = None,
    period: str = "2y",
    force_refresh: bool = True,
    force: bool = False,
) -> dict:
    """Run the whole Full Analysis pipeline for ``symbols``.

    ``target`` is the session whose close is being predicted (defaults to the
    next unresolved session); data is cut off at the previous trading day.
    Returns a summary dict — the durable output is in Postgres.
    """
    from config import MODEL
    from services import progress_service as prog
    from services.news_window import fetch_point_in_time_news
    from utils.trading_calendar import resolve_target_and_cutoff

    recs_model = recs_model or MODEL.RECOMMENDATIONS_MODEL
    target_date, cutoff_date = resolve_target_and_cutoff(target)
    as_of = cutoff_date.isoformat()
    started = datetime.now()

    prog.start_run(f"Full Analysis (scheduled) — {len(symbols)} symbols, "
                   f"target {target_date} (data through {as_of})")

    stock_data = load_market_data(symbols, period=period, force_refresh=force_refresh)
    priced = [s for s in symbols if s in stock_data]
    if not priced:
        prog.finish_run("Full Analysis aborted — no price data")
        return {"error": "no price data", "symbols": symbols}

    from services.news_service import NewsUnavailable

    news_by_symbol: dict[str, list] = {}
    news_unavailable: list[str] = []   # the source failed
    news_empty: list[str] = []         # the source answered, with nothing

    for sym in priced:
        articles = []
        # Retry with backoff: the vendor throttles, and a loop over a whole
        # watchlist is exactly the shape that trips it. Only NewsUnavailable
        # is retried; a genuine empty window is not a failure to retry.
        for attempt in (1, 2, 3):
            try:
                articles = fetch_point_in_time_news(
                    sym, as_of, lookback_days=lookback_days)
                break
            except NewsUnavailable as e:
                logger.warning("%s: news unavailable (attempt %d/3): %s",
                               sym, attempt, e)
                if attempt == 3:
                    news_unavailable.append(sym)
                else:
                    time.sleep(2.0 * attempt)
            except Exception as e:
                logger.warning("%s: point-in-time news fetch failed: %s", sym, e)
                news_unavailable.append(sym)
                break

        news_by_symbol[sym] = [_article_dict(a) for a in articles]
        if not articles and sym not in news_unavailable:
            news_empty.append(sym)

    prog.emit("news", f"News window {lookback_days}d ending {as_of}: "
                      f"{sum(len(v) for v in news_by_symbol.values())} articles")

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

    signals = run_predictions(
        priced, stock_data, news_by_symbol,
        target_date=target_date, cutoff_date=cutoff_date,
        models=models, research_model=report_model,
        include_thesis=include_thesis, force=force,
    )
    stored, evaluated = persist_predictions(signals)

    ai_analysis = build_ai_report(
        priced, news_by_symbol, cutoff_date,
        report_model=report_model, recs_model=recs_model,
        stock_data=stock_data, lookback_days=lookback_days,
        include_thesis=include_thesis,
    )

    recommendations = run_recommendations(ai_analysis, signals, priced, as_of)

    # The archived JSON is the only durable copy of the portfolio-level report
    # — predictions and recommendations have their own tables, this does not.
    # It used to fail silently at INFO, so an object-storage outage cost every
    # report with nothing to show it had happened.
    archived = False
    archive_error = None
    try:
        from services import persistence_service as ps
        merged, _ = merge_research_into_analysis(ai_analysis, signals, priced)
        ps.store_report(
            symbol=None, trade_date=as_of, report_type="ai_report",
            input_data_hash=ps.compute_data_hash(_report_cache_key(
                news_by_symbol, priced, as_of, report_model,
                lookback_days, include_thesis)),
            content=json.dumps(merged, default=str, indent=2),
            file_format="json",
        )
        archived = True
    except Exception as e:
        archive_error = str(e)
        logger.error(f"AI report NOT archived — it is unrecoverable once this "
                     f"process exits: {e}")
        prog.emit("error", f"AI report not archived: {str(e)[:120]}")

    coverage, degraded = _assess_completeness(
        signals, priced, models, recommendations)
    if not archived:
        degraded.append(f"report not archived ({(archive_error or '')[:80]})")

    if degraded:
        prog.emit("error", "Full Analysis incomplete — " + "; ".join(degraded))
        prog.finish_run("Full Analysis PARTIAL — " + "; ".join(degraded))
    else:
        prog.finish_run("Full Analysis complete (scheduled)")

    return {
        "symbols": priced,
        "skipped": [s for s in symbols if s not in priced],
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

    A run that ships four of six models and no synthesis is not a success —
    it is a partial result that happens to have exited zero. Every model
    failure here is caught per model by design (one broken model must not
    lose the other five), so without an explicit completeness check the only
    signal left is the exit code, which says nothing about coverage. That is
    how two dead models and a truncated synthesis ran unnoticed for a full
    cycle in production.

    Returns ``(coverage, degraded)`` — symbols scored per model, and a list
    of human-readable reasons, empty when the run is whole.
    """
    expected = set(models or ALL_MODELS) | {"ensemble"}
    coverage = {name: 0 for name in expected}
    for symbol in symbols:
        for name, result in (signals.get(symbol) or {}).items():
            if name in coverage and isinstance(result, dict) and not result.get("error"):
                coverage[name] += 1

    n = len(symbols)
    degraded: list[str] = []
    dead = sorted(m for m, scored in coverage.items() if scored == 0)
    partial = sorted(m for m, scored in coverage.items() if 0 < scored < n)
    if dead:
        degraded.append(f"scored nothing: {', '.join(dead)}")
    if partial:
        degraded.append("incomplete: " + ", ".join(
            f"{m} {coverage[m]}/{n}" for m in partial))
    if not (recommendations or {}).get("by_symbol"):
        degraded.append("no recommendations produced")
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


def _article_dict(a) -> dict:
    """Flatten a NewsArticle to the dict shape the models and LLM expect."""
    published = getattr(a, "published_at", None)
    return {
        "id": getattr(a, "id", None),
        "symbol": getattr(a, "symbol", None),
        "title": getattr(a, "title", ""),
        "source": getattr(a, "source", None),
        "url": getattr(a, "url", None),
        "published_at": published.isoformat() if published else None,
        "summary": getattr(a, "summary", None),
        "sentiment": getattr(a, "sentiment", None),
        "sentiment_score": getattr(a, "sentiment_score", None),
        "impact": getattr(a, "impact", None),
        "price_change_percent": getattr(a, "price_change_percent", None),
        "ticker_relevance_score": getattr(a, "ticker_relevance_score", None),
        "topics": getattr(a, "topics", None),
        "overall_sentiment_score": getattr(a, "overall_sentiment_score", None),
        "overall_sentiment_label": getattr(a, "overall_sentiment_label", None),
    }

"""Prediction service orchestrator.

Wires all models (Kronos, XGBoost, LightGBM, DeBERTa, TradingAgents, Ensemble)
with per-model isolation. Returns serializable dicts (no Postgres writes —
those happen in the server process via the persist_predictions callback).

Parallelism strategy (3-phase):
  Phase 1: Kronos runs FIRST (torch model loading deadlocks if XGBoost
           pickle loads first on MPS).
  Phase 2: XGBoost, LightGBM (CPU), DeBERTa (CPU), TradingAgents (network I/O)
           run concurrently via ThreadPoolExecutor.
  Phase 3: Ensemble runs LAST — depends on all Phase 1+2 results.
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import copy_context
from functools import partial
from typing import Optional

import pandas as pd

from models.base import PredictionResult
from models.registry import ModelRegistry

logger = logging.getLogger(__name__)

# Singleton
_service_instance = None

# Models that must run before others (torch/MPS deadlock prevention)
_PRIORITY_MODELS = {"kronos_mini"}

# Ensemble runs after all others (needs their results)
_ENSEMBLE_MODELS = {"ensemble"}


class PredictionService:
    """Orchestrates model predictions with per-model isolation."""

    def __init__(self) -> None:
        self._registry = ModelRegistry()
        self._register_models()

    def _register_models(self) -> None:
        """Register all available models.

        Kronos is registered first so its lazy torch model loading runs
        before XGBoost's pickle deserialization, which can deadlock
        torch/MPS state if it runs first.
        """
        # Phase 1: Kronos first (requires torch — must load before XGBoost pickle)
        try:
            from models.kronos_model import KronosModel
            self._registry.register(KronosModel())
        except Exception as e:
            logger.warning(f"Kronos registration failed: {e}")

        # Phase 2: Parallel models
        try:
            from models.xgboost_model import XGBoostModel
            self._registry.register(XGBoostModel())
        except Exception as e:
            logger.warning(f"XGBoost registration failed: {e}")

        try:
            from models.lightgbm_model import LightGBMModel
            self._registry.register(LightGBMModel())
        except Exception as e:
            logger.warning(f"LightGBM registration failed: {e}")

        try:
            from models.deberta_model import DeBERTaModel
            self._registry.register(DeBERTaModel())
        except Exception as e:
            logger.warning(f"DeBERTa registration failed: {e}")

        try:
            from models.trading_agents_model import TradingAgentsModel
            self._registry.register(TradingAgentsModel())
        except Exception as e:
            logger.warning(f"TradingAgents registration failed: {e}")

        # Phase 3: Ensemble (runs last, depends on all above)
        try:
            from models.ensemble_model import EnsembleModel
            self._registry.register(EnsembleModel())
        except Exception as e:
            logger.warning(f"Ensemble registration failed: {e}")

        logger.info(
            f"Registered models: {self._registry.list_models()}, "
            f"ready: {self._registry.list_ready_models()}"
        )

    def _run_single_model(
        self,
        model_name: str,
        symbol: str,
        ohlcv_df: pd.DataFrame,
        **kwargs,
    ) -> tuple[str, dict]:
        """Run a single model with isolation. Returns (model_name, result_dict).

        Every model in every phase funnels through here, so this is also where
        per-model progress is emitted -- the panel would otherwise show one
        line per symbol covering ~60s of work with no indication of which
        model is running or what it decided.
        """
        from services import progress_service as prog

        model = self._registry.get(model_name)
        if model is None or not model.is_ready():
            prog.emit("model", f"{symbol} · {model_name} skipped (not ready)")
            return model_name, PredictionResult(
                model_name=model_name,
                decision="HOLD",
                confidence=0.0,
                up_probability=0.5,
                details={},
                error=f"{model_name} not ready",
            ).to_dict()

        prog.emit("model", f"{symbol} · {model_name} running…")
        t0 = time.time()
        try:
            result = model.predict(symbol, ohlcv_df, **kwargs)
            prog.emit(
                "model",
                f"{symbol} · {model_name} → {result.decision} "
                f"({result.confidence:.0%}) in {time.time() - t0:.1f}s",
            )
            return model_name, result.to_dict()
        except Exception as e:
            logger.error(f"{model_name} failed for {symbol}: {e}")
            prog.emit("error", f"{symbol} · {model_name} failed: {str(e)[:120]}")
            return model_name, PredictionResult(
                model_name=model_name,
                decision="HOLD",
                confidence=0.0,
                up_probability=0.5,
                details={},
                error=str(e),
            ).to_dict()

    def predict_symbol_no_store(
        self,
        symbol: str,
        ohlcv_df: pd.DataFrame,
        spy_df: Optional[pd.DataFrame] = None,
        sector_df: Optional[pd.DataFrame] = None,
        news: Optional[list[dict]] = None,
        global_news: Optional[list[dict]] = None,
        global_context: Optional[str] = None,
        ensemble_config: Optional[dict] = None,
        models_to_run: Optional[set[str]] = None,
        run_ensemble: bool = True,
        historical_av_news: Optional[dict] = None,
        historical_global_news: Optional[dict] = None,
        as_of: Optional[str] = None,
        **model_kwargs,
    ) -> dict[str, dict]:
        """Run selected models on a symbol with 3-phase parallelism.

        Does NOT write to DuckDB — returns serializable dict for dcc.Store.

        Execution order:
          Phase 1: Priority models (Kronos) run sequentially — torch/MPS deadlock.
          Phase 2: XGBoost, LightGBM, DeBERTa, TradingAgents run concurrently.
          Phase 3: Ensemble runs last with Phase 1+2 results.

        Args:
            ensemble_config: From UI store — {"enabled_models": [...], "weights": {...}}.
                If None, ensemble model uses config defaults.
            models_to_run: Set of model names to run. If None, runs all.
            run_ensemble: Whether to run the ensemble model.
            historical_av_news: Dict[date_str → list[dict]] for XGBoost/LightGBM training.
            historical_global_news: Dict[date_str → list[dict]] for global news features.
            as_of: Backtest cut-off date (ISO). Models must not use any data
                after this date — enforced for internally-fetched frames too.
            model_kwargs: Extra per-model options forwarded verbatim (e.g.
                the Full Analysis flow's ``research_model``/``include_thesis``,
                which only trading_agents reads). Models ignore what they
                don't recognize.
        """
        kwargs = dict(
            **model_kwargs,
            spy_df=spy_df,
            sector_df=sector_df,
            av_news=news or [],
            global_news=global_news or [],
            news=news or [],
            global_context=global_context,
            historical_av_news=historical_av_news or {},
            historical_global_news=historical_global_news or {},
            as_of=as_of,
        )

        all_models = self._registry.list_models()

        # Filter to user-selected models only
        if models_to_run is not None:
            individual_models = [m for m in all_models if m in models_to_run]
        else:
            individual_models = [
                m for m in all_models if m not in _ENSEMBLE_MODELS
            ]

        priority = [m for m in individual_models if m in _PRIORITY_MODELS]
        parallel = [
            m for m in individual_models
            if m not in _PRIORITY_MODELS and m not in _ENSEMBLE_MODELS
        ]
        ensemble = (
            [m for m in all_models if m in _ENSEMBLE_MODELS]
            if run_ensemble
            else []
        )

        results: dict[str, dict] = {}

        # Phase 1: Priority models (sequential — torch must load first)
        for model_name in priority:
            name, result = self._run_single_model(
                model_name, symbol, ohlcv_df, **kwargs,
            )
            results[name] = result

        # Phase 2: Remaining individual models (parallel — CPU + I/O don't contend)
        if parallel:
            # Worker threads start with an EMPTY context, unlike
            # asyncio.to_thread which copies the caller's. Anything the caller
            # set in a ContextVar — the signed-in identity, the usage/cost
            # stage label — would be invisible to the model running here, so
            # the research model's spend landed under stage "unknown". Copy
            # the context in explicitly at submit time.
            with ThreadPoolExecutor(max_workers=len(parallel)) as executor:
                futures = {
                    executor.submit(
                        copy_context().run,
                        partial(self._run_single_model,
                                model_name, symbol, ohlcv_df, **kwargs),
                    ): model_name
                    for model_name in parallel
                }
                for future in as_completed(futures):
                    name, result = future.result()
                    results[name] = result

        # Phase 3: Ensemble (sequential — depends on all Phase 1+2 results)
        for model_name in ensemble:
            ensemble_kwargs = {
                **kwargs,
                "other_results": results,
                "ensemble_config": ensemble_config,
            }
            name, result = self._run_single_model(
                model_name, symbol, ohlcv_df, **ensemble_kwargs,
            )
            results[name] = result

        return results

    def predict_all_symbols(
        self,
        symbols: list[str],
        stock_data_dict: dict,
        news_by_symbol: Optional[dict] = None,
        global_news: Optional[list[dict]] = None,
        global_context: Optional[str] = None,
        ensemble_config: Optional[dict] = None,
    ) -> dict[str, dict[str, dict]]:
        """Run predictions for all symbols.

        Fetches SPY once and reuses across symbols.

        Args:
            symbols: List of ticker symbols.
            stock_data_dict: Dict mapping symbol to OHLCV DataFrame.
            news_by_symbol: Dict mapping symbol to news articles.
            global_news: Global/macro news articles.
            global_context: Market context string.
            ensemble_config: From UI store for ensemble model.

        Returns:
            Dict mapping symbol to model results dict.
        """
        # Fetch SPY once for all symbols
        spy_df = None
        try:
            from services.stock_data import fetch_stock_data
            spy_df = fetch_stock_data("SPY", period="1y")
        except Exception as e:
            logger.warning(f"SPY fetch failed: {e}. XGBoost/LightGBM will be unavailable.")

        results: dict[str, dict[str, dict]] = {}
        for symbol in symbols:
            ohlcv_df = stock_data_dict.get(symbol)
            if ohlcv_df is None:
                continue

            news = (news_by_symbol or {}).get(symbol, [])

            results[symbol] = self.predict_symbol_no_store(
                symbol,
                ohlcv_df,
                spy_df=spy_df,
                news=news,
                global_news=global_news,
                global_context=global_context,
                ensemble_config=ensemble_config,
            )

        return results


def get_prediction_service() -> PredictionService:
    """Get the singleton prediction service instance."""
    global _service_instance
    if _service_instance is None:
        _service_instance = PredictionService()
    return _service_instance

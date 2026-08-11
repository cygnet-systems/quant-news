"""LightGBM model for next-day directional prediction.

Mirrors xgboost_model.py exactly — same 18 SHAP-selected features,
same walk-forward training, same disk cache. Swaps XGBClassifier
for LGBMClassifier.
"""

import json
import logging
import pickle
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from config import MODEL, API
from models.base import BaseModel, PredictionResult
from models.feature_builder import LiveFeatureBuilder, SELECTED_FEATURES
from models.sector_map import get_sector_etf
from services.analytics import add_indicators_to_df

logger = logging.getLogger(__name__)

MODELS_CACHE_DIR = Path("cache/trained_models")

LIGHTGBM_PARAMS = {
    "num_leaves": MODEL.LIGHTGBM_NUM_LEAVES,
    "learning_rate": MODEL.LIGHTGBM_LEARNING_RATE,
    "n_estimators": MODEL.LIGHTGBM_N_ESTIMATORS,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "verbose": -1,
}


class LightGBMModel(BaseModel):
    """LightGBM directional prediction model."""

    def __init__(self) -> None:
        self._feature_builder = LiveFeatureBuilder()
        self._last_training_samples: int = 0

    @property
    def name(self) -> str:
        return "lightgbm"

    def is_ready(self) -> bool:
        return True

    def predict(
        self,
        symbol: str,
        ohlcv_df: pd.DataFrame,
        **kwargs,
    ) -> PredictionResult:
        """Generate next-day prediction using LightGBM.

        Required kwargs: same as XGBoostModel (spy_df, sector_df, av_news, global_news).
        """
        if not API.ALPHA_VANTAGE_API_KEY:
            return PredictionResult(
                model_name=self.name,
                decision="HOLD",
                confidence=0.0,
                up_probability=0.5,
                details={},
                error="ALPHA_VANTAGE_API_KEY required for LightGBM features",
            )

        spy_df = kwargs.get("spy_df")
        if spy_df is None:
            return PredictionResult(
                model_name=self.name,
                decision="HOLD",
                confidence=0.0,
                up_probability=0.5,
                details={},
                error="SPY data required for LightGBM features",
            )

        as_of = kwargs.get("as_of")

        sector_df = kwargs.get("sector_df")
        # Recorded on the trained-model cache: a model trained when the sector
        # lookup fell back to SPY holds features that duplicate the SPY block,
        # and nothing else in the cache key changes when the lookup later
        # resolves correctly. Without this the corrupted model is served all
        # day from disk.
        sector_etf = "caller-provided"
        if sector_df is None:
            sector_etf = get_sector_etf(symbol)
            try:
                from services.stock_data import fetch_stock_data
                sector_df = fetch_stock_data(sector_etf, period="1y")
            except Exception as e:
                logger.warning(f"Sector ETF {sector_etf} fetch failed: {e}")
                sector_df = spy_df

        # No lookahead: internally-fetched frames include data past a backtest
        # cut-off — truncate everything to the as-of date.
        if as_of and sector_df is not None:
            sector_df = sector_df[sector_df.index <= str(as_of)]

        av_news = kwargs.get("av_news", [])
        global_news = kwargs.get("global_news", [])
        historical_av_news = kwargs.get("historical_av_news", {})
        historical_global_news = kwargs.get("historical_global_news", {})

        try:
            clf = self._get_or_train(
                symbol, ohlcv_df, spy_df, sector_df,
                historical_av_news, historical_global_news,
                sector_etf=sector_etf,
            )

            features = self._feature_builder.build_features(
                ohlcv_df, spy_df, sector_df,
                av_news=av_news,
                global_news=global_news,
            )

            return self._predict_from_model(clf, features, symbol)

        except Exception as e:
            logger.error(f"LightGBM prediction failed for {symbol}: {e}", exc_info=True)
            return PredictionResult(
                model_name=self.name,
                decision="HOLD",
                confidence=0.0,
                up_probability=0.5,
                details={},
                error=str(e),
            )

    def _get_or_train(
        self,
        symbol: str,
        ticker_df: pd.DataFrame,
        spy_df: pd.DataFrame,
        sector_df: pd.DataFrame,
        historical_av_news: dict[str, list[dict]],
        historical_global_news: dict[str, list[dict]],
        sector_etf: str = "",
    ) -> LGBMClassifier:
        """Get cached model or train a new one."""
        MODELS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path = MODELS_CACHE_DIR / f"{symbol}_lightgbm.pkl"
        meta_path = MODELS_CACHE_DIR / f"{symbol}_lightgbm_meta.json"

        # Cache validity is tied to the training-data window — a model trained
        # on full history must not serve an earlier-as-of backtest (leak).
        data_end = str(ticker_df.index[-1].date()) if len(ticker_df) else ""
        # The training NEWS window is an input too. Price data ending on the
        # same day says nothing about how much news backed each training row,
        # so a model trained on a frozen/partial news window would be served
        # unchanged after the window was repaired.
        news_end = max(historical_av_news) if historical_av_news else ""

        if cache_path.exists() and meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
                # feature_version: a model trained under old feature
                # definitions must not score new-definition inputs.
                from models.feature_builder import FEATURE_VERSION
                if (meta.get("training_date") == str(date.today())
                        and meta.get("data_end") == data_end
                        and meta.get("feature_version") == FEATURE_VERSION
                        and meta.get("sector_etf") == sector_etf
                        and meta.get("news_end") == news_end):
                    clf = pickle.loads(cache_path.read_bytes())
                    self._last_training_samples = meta.get("n_samples", 0)
                    logger.info(f"LightGBM cache hit: {symbol} (data through {data_end})")
                    return clf
            except Exception as e:
                logger.warning(f"Cache load failed for {symbol}: {e}")

        clf, n_samples = self._train_from_history(
            symbol, ticker_df, spy_df, sector_df,
            historical_av_news, historical_global_news,
        )

        cache_path.write_bytes(pickle.dumps(clf))
        from models.feature_builder import FEATURE_VERSION
        meta_path.write_text(json.dumps({
            "training_date": str(date.today()),
            "symbol": symbol,
            "n_samples": n_samples,
            "data_end": data_end,
            "feature_version": FEATURE_VERSION,
            "sector_etf": sector_etf,
            "news_end": news_end,
        }))

        self._last_training_samples = n_samples
        return clf

    def _train_from_history(
        self,
        symbol: str,
        ticker_df: pd.DataFrame,
        spy_df: pd.DataFrame,
        sector_df: pd.DataFrame,
        historical_av_news: dict[str, list[dict]],
        historical_global_news: dict[str, list[dict]],
    ) -> tuple[LGBMClassifier, int]:
        """Train LightGBM on walk-forward windows."""
        ticker_full = add_indicators_to_df(ticker_df.copy())
        spy_full = add_indicators_to_df(spy_df.copy())
        sector_full = add_indicators_to_df(sector_df.copy())

        threshold = MODEL.LABEL_AMBIGUITY_THRESHOLD
        rows: list[dict] = []
        labels: list[int] = []
        skipped = 0

        # ATR-scaled ambiguity band — see xgboost_model._train_from_history
        # for the rationale; the two trainers must label identically.
        atr_col = ticker_full["ATR"] if "ATR" in ticker_full.columns else None

        min_window = 60
        for t in range(min_window, len(ticker_full) - 1):
            close_today = float(ticker_full["Close"].iloc[t])
            close_tomorrow = float(ticker_full["Close"].iloc[t + 1])

            if close_today <= 0:
                continue

            pct = (close_tomorrow - close_today) / close_today

            day_threshold = threshold
            if atr_col is not None:
                atr_val = atr_col.iloc[t]
                if pd.notna(atr_val) and close_today > 0:
                    day_threshold = max(
                        threshold, 0.15 * float(atr_val) / close_today)
            if day_threshold > 0.0 and abs(pct) < day_threshold:
                skipped += 1
                continue

            label = 1 if pct > 0 else 0

            try:
                date_str = str(ticker_full.index[t])[:10]
                day_news = historical_av_news.get(date_str, [])
                day_global = historical_global_news.get(date_str, [])

                features = self._feature_builder.build_features(
                    ticker_full.iloc[: t + 1],
                    spy_full.iloc[: t + 1],
                    sector_full.iloc[: t + 1],
                    av_news=day_news,
                    global_news=day_global,
                )
                rows.append(features)
                labels.append(label)
            except Exception:
                continue

        logger.info(
            f"Training LightGBM {symbol}: {len(rows)} samples "
            f"({skipped} skipped ambiguous, threshold={threshold:.4f})"
        )

        if len(rows) < MODEL.MIN_TRAINING_SAMPLES:
            raise ValueError(
                f"Insufficient training samples: {len(rows)} < "
                f"{MODEL.MIN_TRAINING_SAMPLES}. "
                f"Try LABEL_AMBIGUITY_THRESHOLD=0.0"
            )

        X = pd.DataFrame(rows)[SELECTED_FEATURES]
        y = pd.Series(labels)

        clf = LGBMClassifier(**LIGHTGBM_PARAMS)
        clf.fit(X, y)

        return clf, len(rows)

    def _predict_from_model(
        self,
        clf: LGBMClassifier,
        features: dict,
        symbol: str,
    ) -> PredictionResult:
        """Predict using trained classifier."""
        X = pd.DataFrame([features])[SELECTED_FEATURES]
        prob = clf.predict_proba(X)[0]
        up_prob = float(prob[1]) if len(prob) > 1 else 0.5

        if up_prob > MODEL.BUY_THRESHOLD:
            decision = "BUY"
        elif up_prob < MODEL.SELL_THRESHOLD:
            decision = "SELL"
        else:
            decision = "HOLD"

        return PredictionResult(
            model_name=self.name,
            decision=decision,
            confidence=round(max(up_prob, 1 - up_prob), 2),
            up_probability=round(up_prob, 4),
            details={
                "feature_values": features,
                "model_version": f"lightgbm_v1_{date.today():%Y%m%d}",
                "training_samples": self._last_training_samples,
            },
        )

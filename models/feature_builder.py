"""Feature builder for XGBoost model.

Single-path design: one build_features() function handles both raw OHLCV
(live prediction) and pre-computed indicator DataFrames (training loop).
Uses _ensure_indicators() with sentinel column detection.

Column naming: maps quant-news DataFrame columns (from add_indicators_to_df)
to TradingAgents internal names used in the SHAP feature set.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Sentinel column to detect if add_indicators_to_df() has been applied.
# SMA_50 requires 50 rows of data and is always present after enrichment.
# Bump when feature SEMANTICS change (formulas, spans, missing-value policy).
# Trained-model caches key on this so a .pkl trained under old definitions
# retrains instead of silently scoring new-definition inputs.
# v2 (2026-07-25): missing price features are NaN (was fabricated 0.0 --
# an absent RSI scored as "maximally oversold"); *_5d spans are a true
# 5 sessions (was 4); ta's ATR warm-up zeros masked to NaN.
# v3 (2026-07-26): context ETF resolved from the symbol's industry/sector
# metadata (hardcoded ticker map removed; industry ETF preferred when mapped,
# e.g. Semiconductors -> SMH, else sector SPDR) -- sector-context features
# change for any ticker the old map labeled differently.
# v4 (2026-08-06): DB-cached articles expose topics under "topics_json", not
# "topics" -- every av_*/global_* feature trained as constant 0.0 on cache
# hits. Both spellings now read; all news-trained models must be refit.
# v5 (2026-08-11): +news_present (blind-vs-quiet disambiguation),
# +atr_percentile, +dist_52wk_high_pct (regime features. R2000 test showed
# failures cluster in high-vol names and the model could not see volatility
# regime at all). 18 -> 21 features; cached models retrain on version bump.
FEATURE_VERSION: int = 5

INDICATOR_SENTINEL_COLUMN: str = "SMA_50"

# Maps quant-news DataFrame column names (from add_indicators_to_df())
# to TradingAgents internal indicator names (used in feature naming).
INDICATOR_COLUMN_MAP: dict[str, str] = {
    "SMA_50": "close_50_sma",
    "SMA_200": "close_200_sma",
    "RSI": "rsi",
    "ATR": "atr",
    "MACD": "macd",
    "BB_Mid": "boll",
}

# OHLCV column names in quant-news DataFrames (from yfinance).
OHLCV_COLUMNS: dict[str, str] = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Volume": "volume",
}

# AV topic names used in features
_AV_TOPICS = [
    "mergers_and_acquisitions",
    "retail_wholesale",
    "real_estate",
    "ipo",
]

_GLOBAL_TOPICS = [
    "real_estate",
    "retail_wholesale",
    "energy_transportation",
]

# Load selected features list
_FEATURES_PATH = Path(__file__).parent / "selected_features.json"
with open(_FEATURES_PATH) as f:
    SELECTED_FEATURES: list[str] = json.load(f)["features"]


class LiveFeatureBuilder:
    """Builds the 18 SHAP-selected features for XGBoost prediction.

    Single entry point: build_features() handles both raw and enriched
    DataFrames via _ensure_indicators() sentinel detection.
    """

    def __init__(self) -> None:
        self._selected_features = SELECTED_FEATURES

    def _ensure_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add indicator columns if not already present.

        Detection: checks for INDICATOR_SENTINEL_COLUMN ('SMA_50').
        """
        if INDICATOR_SENTINEL_COLUMN not in df.columns:
            from services.analytics import add_indicators_to_df
            df = add_indicators_to_df(df.copy())
            # ta's AverageTrueRange writes literal 0.0 (not NaN) for its
            # first window-1 warm-up rows, so fake zero-volatility values
            # survive .dropna() downstream. Mask them.
            if "ATR" in df.columns and len(df) > 0:
                warm = min(13, len(df))
                df.iloc[:warm, df.columns.get_loc("ATR")] = np.nan
            return df
        return df

    def build_features(
        self,
        ticker_df: pd.DataFrame,
        spy_df: pd.DataFrame,
        sector_df: pd.DataFrame,
        av_news: list[dict] | None = None,
        global_news: list[dict] | None = None,
    ) -> dict[str, float]:
        """Build all 18 features from DataFrames and news.

        Args:
            ticker_df: Ticker OHLCV DataFrame (raw or enriched).
            spy_df: SPY OHLCV DataFrame.
            sector_df: Sector ETF OHLCV DataFrame.
            av_news: Ticker-specific AV news articles for the current day.
            global_news: Global/macro AV news articles for the current day.

        Returns:
            Dict mapping feature name to float value. All 18 features present.
        """
        ticker_df = self._ensure_indicators(ticker_df)
        spy_df = self._ensure_indicators(spy_df)
        sector_df = self._ensure_indicators(sector_df)

        features: dict[str, float] = {}

        # Ticker features
        features.update(self._gap_features(ticker_df, prefix=""))
        features.update(self._indicator_features(ticker_df, prefix=""))

        # SPY features
        features.update(self._gap_features(spy_df, prefix="spy_"))
        features.update(self._return_features(spy_df, prefix="spy_"))
        features.update(self._indicator_features(spy_df, prefix="spy_"))

        # Relative strength: ticker vs SPY
        features.update(self._relative_features(ticker_df, spy_df))

        # Sector features
        features.update(self._gap_features(sector_df, prefix="sector_"))
        features.update(self._range_features(sector_df, prefix="sector_"))

        # AV ticker news topic features
        features.update(self._av_topic_features(av_news or [], prefix="av_"))

        # Global news topic features
        features.update(
            self._global_topic_features(global_news or [], prefix="global_")
        )

        # Regime / evidence features (v3 additions).
        # news_present separates "quiet week" (topic features 0.0 WITH
        # articles seen) from "blind" (topic features 0.0 with NO articles), 
        # without it those opposite situations are the same input.
        features["news_present"] = 1.0 if av_news else 0.0
        features.update(self._regime_features(ticker_df))

        # Ensure all 18 features are present. Missing NEWS/topic features
        # fill with 0.0 -- "no articles" is a real zero. Missing PRICE
        # features fill with NaN: XGBoost/LightGBM handle NaN natively,
        # whereas the old blanket 0.0 fabricated signals (an absent RSI
        # became "maximally oversold").
        result: dict[str, float] = {}
        for feat in self._selected_features:
            if feat in features:
                result[feat] = features[feat]
            elif feat.startswith(("av_", "global_")):
                result[feat] = 0.0
            else:
                result[feat] = np.nan

        return result

    def _regime_features(self, df: pd.DataFrame) -> dict[str, float]:
        """Volatility-regime and trend-position features (v3).

        atr_percentile: today's ATR within its own trailing year, whipsaw
        risk is the documented failure mode, and absolute ATR is not
        comparable across names the way its own percentile is.
        dist_52wk_high_pct: percent below the trailing-252-session high.
        separates "extended near highs" from "washed out", which momentum
        features alone conflate.
        """
        features: dict[str, float] = {
            "atr_percentile": np.nan,
            "dist_52wk_high_pct": np.nan,
        }
        if "ATR" in df.columns:
            atr = df["ATR"].dropna()
            if len(atr) >= 60:
                window = atr.tail(252)
                current = float(window.iloc[-1])
                features["atr_percentile"] = float(
                    (window <= current).mean() * 100)
        if "Close" in df.columns and len(df) >= 60:
            closes = df["Close"].tail(252)
            high = float(closes.max())
            if high > 0:
                features["dist_52wk_high_pct"] = (
                    (float(closes.iloc[-1]) / high - 1) * 100)
        return features

    def _gap_features(
        self, df: pd.DataFrame, prefix: str
    ) -> dict[str, float]:
        """Overnight gap: (open[-1] / close[-2] - 1) * 100."""
        features: dict[str, float] = {}
        key = f"{prefix}gap"

        # Missing data is NaN, not a fabricated "no gap" of 0.0.
        if len(df) >= 2 and "Open" in df.columns and "Close" in df.columns:
            open_last = float(df["Open"].iloc[-1])
            close_prev = float(df["Close"].iloc[-2])
            features[key] = (
                (open_last / close_prev - 1) * 100 if close_prev > 0 else np.nan
            )
        else:
            features[key] = np.nan

        return features

    def _return_features(
        self, df: pd.DataFrame, prefix: str
    ) -> dict[str, float]:
        """5-session return: (close[-1] / close[-6] - 1) * 100.

        iloc[-6] on purpose: [-5] spans only 4 intervals, which disagreed
        with range_5d's true 5 sessions while both claimed "5-day".
        """
        features: dict[str, float] = {}
        key = f"{prefix}return_5d"

        if len(df) >= 6 and "Close" in df.columns:
            close_now = float(df["Close"].iloc[-1])
            close_5d = float(df["Close"].iloc[-6])
            features[key] = (
                (close_now / close_5d - 1) * 100 if close_5d > 0 else np.nan
            )
        else:
            features[key] = np.nan

        return features

    def _relative_features(
        self, ticker_df: pd.DataFrame, spy_df: pd.DataFrame
    ) -> dict[str, float]:
        """Ticker excess return vs SPY over 5 sessions.

        NaN when either leg is missing -- the old 0.0 default turned "no
        SPY data" into "ticker outperformed by its own absolute return".
        """
        features: dict[str, float] = {}

        def _ret5(df: pd.DataFrame) -> float:
            if len(df) >= 6 and "Close" in df.columns:
                now = float(df["Close"].iloc[-1])
                then = float(df["Close"].iloc[-6])
                if then > 0:
                    return (now / then - 1) * 100
            return np.nan

        ticker_ret = _ret5(ticker_df)
        spy_ret = _ret5(spy_df)
        features["vs_spy_excess_5d"] = ticker_ret - spy_ret  # NaN propagates
        return features

    def _range_features(
        self, df: pd.DataFrame, prefix: str
    ) -> dict[str, float]:
        """5-day range: (high_5d_max - low_5d_min) / close[-1] * 100."""
        features: dict[str, float] = {}
        key = f"{prefix}range_5d"

        if len(df) >= 5 and all(c in df.columns for c in ["High", "Low", "Close"]):
            high_max = float(df["High"].iloc[-5:].max())
            low_min = float(df["Low"].iloc[-5:].min())
            close_now = float(df["Close"].iloc[-1])
            features[key] = (
                (high_max - low_min) / close_now * 100 if close_now > 0 else np.nan
            )
        else:
            features[key] = np.nan

        return features

    def _indicator_features(
        self, df: pd.DataFrame, prefix: str = ""
    ) -> dict[str, float]:
        """Extract latest value and 5-day delta for mapped indicator columns."""
        features: dict[str, float] = {}
        for col_name, internal_name in INDICATOR_COLUMN_MAP.items():
            if col_name not in df.columns:
                continue
            series = df[col_name].dropna()
            if len(series) > 0:
                features[f"{prefix}ind_{internal_name}"] = float(series.iloc[-1])
                # iloc[-6]: a true 5-session delta (iloc[-5] was 4 intervals
                # under a "_delta5" name).
                if len(series) >= 6:
                    features[f"{prefix}ind_{internal_name}_delta5"] = float(
                        series.iloc[-1] - series.iloc[-6]
                    )
        return features

    def _av_topic_features(
        self, articles: list[dict], prefix: str = "av_"
    ) -> dict[str, float]:
        """Extract AV topic sentiment averages and tech article count.

        av_topic_*_avg features are in [-1, 1] range.
        av_n_tech_articles is an unbounded integer count.
        """
        features: dict[str, float] = {}

        # Initialize all topic features to 0
        for topic in _AV_TOPICS:
            features[f"{prefix}topic_{topic}_avg"] = 0.0
        features[f"{prefix}n_tech_articles"] = 0.0

        if not articles:
            return features

        # Collect topic scores
        topic_scores: dict[str, list[float]] = {t: [] for t in _AV_TOPICS}
        n_tech = 0

        for article in articles:
            # topics may be None, a JSON string, or a list depending on source;
            # DB-cached articles carry the raw column name "topics_json"
            topics = article.get("topics") or article.get("topics_json") or []
            if isinstance(topics, str):
                try:
                    import json as _json
                    topics = _json.loads(topics)
                except (ValueError, TypeError):
                    topics = []
            if not isinstance(topics, list):
                topics = []

            for topic_info in topics:
                if isinstance(topic_info, dict):
                    topic_name = topic_info.get("topic", "").lower().replace(" ", "_")
                    relevance = float(topic_info.get("relevance_score", 0))

                    # Map topic names to our feature names
                    for feat_topic in _AV_TOPICS:
                        if feat_topic in topic_name:
                            # Use overall_sentiment_score weighted by relevance
                            sentiment = article.get("overall_sentiment_score", 0)
                            if sentiment is not None:
                                topic_scores[feat_topic].append(
                                    float(sentiment) * relevance
                                )

                    if "technology" in topic_name:
                        n_tech += 1

        # Compute averages
        for topic in _AV_TOPICS:
            scores = topic_scores[topic]
            if scores:
                features[f"{prefix}topic_{topic}_avg"] = float(np.mean(scores))

        features[f"{prefix}n_tech_articles"] = float(n_tech)

        return features

    def _global_topic_features(
        self, articles: list[dict], prefix: str = "global_"
    ) -> dict[str, float]:
        """Extract global news topic sentiment averages."""
        features: dict[str, float] = {}

        for topic in _GLOBAL_TOPICS:
            features[f"{prefix}topic_{topic}_avg"] = 0.0

        if not articles:
            return features

        topic_scores: dict[str, list[float]] = {t: [] for t in _GLOBAL_TOPICS}

        for article in articles:
            # topics may be None, a JSON string, or a list depending on source;
            # DB-cached articles carry the raw column name "topics_json"
            topics = article.get("topics") or article.get("topics_json") or []
            if isinstance(topics, str):
                try:
                    import json as _json
                    topics = _json.loads(topics)
                except (ValueError, TypeError):
                    topics = []
            if not isinstance(topics, list):
                topics = []

            for topic_info in topics:
                if isinstance(topic_info, dict):
                    topic_name = topic_info.get("topic", "").lower().replace(" ", "_")
                    relevance = float(topic_info.get("relevance_score", 0))

                    for feat_topic in _GLOBAL_TOPICS:
                        if feat_topic in topic_name:
                            sentiment = article.get("overall_sentiment_score", 0)
                            if sentiment is not None:
                                topic_scores[feat_topic].append(
                                    float(sentiment) * relevance
                                )

        for topic in _GLOBAL_TOPICS:
            scores = topic_scores[topic]
            if scores:
                features[f"{prefix}topic_{topic}_avg"] = float(np.mean(scores))

        return features

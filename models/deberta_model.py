"""DeBERTa sentiment model for next-day directional prediction.

Uses HuggingFace mrm8488/deberta-v3-ft-financial-news-sentiment-analysis
to classify AV news headlines. Filters by relevance_score >= 0.7 to
avoid the 73% BUY bias from irrelevant articles.

Confidence is self-reported (not calibrated), same as LLM agent.
Pipeline is lazy-loaded on first call to avoid startup delay.
"""

import logging
from typing import Optional

import pandas as pd

from config import MODEL
from models.base import BaseModel, PredictionResult

logger = logging.getLogger(__name__)

# Lazy-loaded pipeline (avoid slow import at startup)
_pipeline = None


def _get_pipeline():
    """Lazy-load the HuggingFace sentiment pipeline."""
    global _pipeline
    if _pipeline is None:
        from transformers import pipeline
        _pipeline = pipeline(
            "sentiment-analysis",
            model=MODEL.DEBERTA_MODEL_NAME,
            truncation=True,
            max_length=512,
        )
    return _pipeline


class DeBERTaModel(BaseModel):
    """DeBERTa financial news sentiment model."""

    @property
    def name(self) -> str:
        return "deberta_sentiment"

    def is_ready(self) -> bool:
        try:
            import transformers  # noqa: F401
            return True
        except ImportError:
            return False

    def predict(
        self,
        symbol: str,
        ohlcv_df: pd.DataFrame,
        **kwargs,
    ) -> PredictionResult:
        """Generate prediction from news sentiment.

        Uses av_news from kwargs, filters by ticker_relevance_score.
        """
        av_news = kwargs.get("av_news", [])

        # Filter to relevant articles only
        relevant = [
            a for a in av_news
            if float(a.get("ticker_relevance_score") or 0)
            >= MODEL.DEBERTA_RELEVANCE_THRESHOLD
        ]

        if not relevant:
            return PredictionResult(
                model_name=self.name,
                decision="HOLD",
                confidence=0.3,
                up_probability=0.5,
                details={
                    "confidence_type": "self_reported",
                    "articles_total": len(av_news),
                    "articles_relevant": 0,
                    "reason": "no relevant articles",
                },
            )

        try:
            pipe = _get_pipeline()

            # Extract titles (primary signal) and summaries (fallback)
            texts = []
            for a in relevant:
                title = a.get("title", "")
                if title:
                    texts.append(title)

            if not texts:
                return PredictionResult(
                    model_name=self.name,
                    decision="HOLD",
                    confidence=0.3,
                    up_probability=0.5,
                    details={
                        "confidence_type": "self_reported",
                        "articles_total": len(av_news),
                        "articles_relevant": len(relevant),
                        "reason": "no article titles to analyze",
                    },
                )

            # Run sentiment pipeline
            results = pipe(texts)

            # Aggregate: map each article to a [0,1] sentiment value
            # positive → score (closer to 1 = more bullish)
            # negative → 1 - score (closer to 0 = more bearish)
            # neutral  → 0.5
            sentiment_values = []
            positive_count = 0
            negative_count = 0
            neutral_count = 0
            for r in results:
                label = r["label"].lower()
                score = r["score"]
                if label in ("positive", "pos"):
                    sentiment_values.append(score)
                    positive_count += 1
                elif label in ("negative", "neg"):
                    sentiment_values.append(1.0 - score)
                    negative_count += 1
                else:
                    sentiment_values.append(0.5)
                    neutral_count += 1

            if not sentiment_values:
                avg_positive = 0.5
            else:
                avg_positive = sum(sentiment_values) / len(sentiment_values)

            # Determine decision
            if avg_positive > MODEL.DEBERTA_BUY_THRESHOLD:
                decision = "BUY"
            elif avg_positive < MODEL.DEBERTA_SELL_THRESHOLD:
                decision = "SELL"
            else:
                decision = "HOLD"

            confidence = abs(avg_positive - 0.5) * 2  # 0-1 scale

            return PredictionResult(
                model_name=self.name,
                decision=decision,
                confidence=round(min(confidence, 1.0), 2),
                up_probability=round(avg_positive, 4),
                details={
                    "confidence_type": "self_reported",
                    "articles_total": len(av_news),
                    "articles_relevant": len(relevant),
                    "positive_count": positive_count,
                    "negative_count": negative_count,
                    "neutral_count": neutral_count,
                    "avg_positive_ratio": round(avg_positive, 4),
                },
            )

        except Exception as e:
            logger.error(f"DeBERTa prediction failed for {symbol}: {e}")
            return PredictionResult(
                model_name=self.name,
                decision="HOLD",
                confidence=0.0,
                up_probability=0.5,
                details={"confidence_type": "self_reported"},
                error=str(e),
            )

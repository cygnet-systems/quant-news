"""Configuration and constants for Quant News Tracker.

This module serves as the SINGLE SOURCE OF TRUTH for all constants,
configuration values, and default settings used throughout the application.
"""

import os
from dataclasses import dataclass
from typing import Final

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


# =============================================================================
# TECHNICAL INDICATOR DEFAULTS
# =============================================================================


@dataclass(frozen=True)
class IndicatorDefaults:
    """Default parameters for technical indicators."""

    # Moving Averages
    SMA_PERIODS: tuple[int, ...] = (20, 50, 200)
    EMA_PERIODS: tuple[int, ...] = (12, 26)

    # MACD
    MACD_FAST: int = 12
    MACD_SLOW: int = 26
    MACD_SIGNAL: int = 9

    # RSI
    RSI_PERIOD: int = 14
    RSI_OVERBOUGHT: int = 70
    RSI_OVERSOLD: int = 30

    # Bollinger Bands
    BOLLINGER_PERIOD: int = 20
    BOLLINGER_STD: float = 2.0

    # ATR
    ATR_PERIOD: int = 14

    # Stochastic
    STOCHASTIC_K: int = 14
    STOCHASTIC_D: int = 3

    # Volume
    VOLUME_MA_PERIOD: int = 20


INDICATORS: Final = IndicatorDefaults()


# =============================================================================
# API CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class APIConfig:
    """API endpoints and settings."""

    # LM Studio (local LLM)
    LM_STUDIO_URL: str = os.getenv("LM_STUDIO_URL", "http://localhost:1234/v1")

    # Anthropic (Claude)
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")

    # OpenAI (fallback)
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_BASE_URL: str = "https://api.openai.com/v1"

    # Alpha Vantage (news)
    ALPHA_VANTAGE_API_KEY: str = os.getenv("ALPHA_VANTAGE_API_KEY", "")
    ALPHA_VANTAGE_BASE_URL: str = "https://www.alphavantage.co/query"

    # Request settings
    DEFAULT_TIMEOUT: int = 30
    MAX_RETRIES: int = 3

    # LLM client timeout (seconds). Without it a hung socket pins a worker
    # thread for the SDK default of 600s. Synthesis calls run ~1 min.
    LLM_TIMEOUT: int = int(os.getenv("LLM_TIMEOUT", "180"))


API: Final = APIConfig()


# =============================================================================
# APP CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class AppConfig:
    """Application settings."""

    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    PORT: int = int(os.getenv("PORT", "8050"))

    # Concurrent per-symbol fetches in async callbacks. Stock fetches tolerate
    # a few parallel yfinance calls; Alpha Vantage free tier (~5 req/min)
    # punishes parallel news fetches, so that cap stays low.
    STOCK_FETCH_CONCURRENCY: int = int(os.getenv("STOCK_FETCH_CONCURRENCY", "4"))
    NEWS_FETCH_CONCURRENCY: int = int(os.getenv("NEWS_FETCH_CONCURRENCY", "2"))
    HOST: str = os.getenv("HOST", "127.0.0.1")

    # Default data period
    DEFAULT_PERIOD: str = "1y"

    # Popular stocks for quick-add
    POPULAR_STOCKS: tuple[str, ...] = (
        "AAPL",
        "MSFT",
        "GOOGL",
        "AMZN",
        "NVDA",
        "META",
        "TSLA",
        "JPM",
    )


APP: Final = AppConfig()


# =============================================================================
# UI THEME - DESIGN SYSTEM
# =============================================================================


@dataclass(frozen=True)
class Colors:
    """Color palette for the dark theme UI."""

    # Background
    BG_PRIMARY: str = "#0D0D0D"
    BG_SECONDARY: str = "#1A1A1A"
    BG_TERTIARY: str = "#242424"

    # Borders
    BORDER_SUBTLE: str = "#2A2A2A"
    BORDER_FOCUS: str = "#3D3D3D"

    # Text
    TEXT_PRIMARY: str = "#FFFFFF"
    TEXT_SECONDARY: str = "#A0A0A0"
    TEXT_MUTED: str = "#666666"

    # Semantic (Robinhood-inspired)
    POSITIVE: str = "#00C805"
    POSITIVE_MUTED: str = "#0A3D0A"
    NEGATIVE: str = "#FF5000"
    NEGATIVE_MUTED: str = "#3D1A0A"
    NEUTRAL: str = "#FFD700"

    # Accent
    ACCENT_PRIMARY: str = "#00D4AA"
    ACCENT_HOVER: str = "#00E5BB"

    # Chart colors (sequential)
    CHART_1: str = "#00D4AA"  # Primary (price)
    CHART_2: str = "#7B61FF"  # MA-20
    CHART_3: str = "#FF6B6B"  # MA-50
    CHART_4: str = "#4ECDC4"  # MA-200
    CHART_5: str = "#FFE66D"  # Bollinger
    VOLUME_UP: str = "#00C805"
    VOLUME_DOWN: str = "#FF5000"


COLORS: Final = Colors()


@dataclass(frozen=True)
class Typography:
    """Typography settings."""

    FONT_PRIMARY: str = "'Inter', -apple-system, BlinkMacSystemFont, sans-serif"
    FONT_MONO: str = "'JetBrains Mono', 'SF Mono', monospace"

    # Font sizes (px)
    TEXT_XS: int = 11
    TEXT_SM: int = 13
    TEXT_BASE: int = 15
    TEXT_LG: int = 18
    TEXT_XL: int = 24
    TEXT_2XL: int = 32
    TEXT_3XL: int = 48


TYPOGRAPHY: Final = Typography()


@dataclass(frozen=True)
class Spacing:
    """Spacing scale (px)."""

    SPACE_1: int = 4
    SPACE_2: int = 8
    SPACE_3: int = 12
    SPACE_4: int = 16
    SPACE_5: int = 24
    SPACE_6: int = 32
    SPACE_8: int = 48


SPACING: Final = Spacing()


@dataclass(frozen=True)
class Layout:
    """Layout dimensions."""

    SIDEBAR_WIDTH: int = 240
    CONTEXT_PANEL_WIDTH: int = 320
    CARD_BORDER_RADIUS: int = 12

    # Breakpoints
    BREAKPOINT_MOBILE: int = 768
    BREAKPOINT_TABLET: int = 1024
    BREAKPOINT_DESKTOP: int = 1440


LAYOUT: Final = Layout()


# =============================================================================
# PLOTLY CHART THEME
# =============================================================================


CHART_THEME: Final[dict] = {
    "paper_bgcolor": COLORS.BG_PRIMARY,
    "plot_bgcolor": COLORS.BG_PRIMARY,
    "font": {
        "color": COLORS.TEXT_SECONDARY,
        "family": TYPOGRAPHY.FONT_PRIMARY,
    },
    "xaxis": {
        "gridcolor": COLORS.BORDER_SUBTLE,
        "linecolor": COLORS.BORDER_SUBTLE,
        "tickfont": {"size": TYPOGRAPHY.TEXT_XS},
        "showgrid": True,
        "gridwidth": 1,
        "griddash": "dot",
    },
    "yaxis": {
        "gridcolor": COLORS.BORDER_SUBTLE,
        "linecolor": COLORS.BORDER_SUBTLE,
        "tickfont": {"size": TYPOGRAPHY.TEXT_XS},
        "side": "right",  # Bloomberg-style
        "showgrid": True,
        "gridwidth": 1,
        "griddash": "dot",
    },
    "hovermode": "x unified",
    "hoverlabel": {
        "bgcolor": COLORS.BG_SECONDARY,
        "bordercolor": COLORS.BORDER_FOCUS,
        "font": {
            "family": TYPOGRAPHY.FONT_MONO,
            "size": TYPOGRAPHY.TEXT_SM,
        },
    },
    "margin": {"l": 10, "r": 60, "t": 40, "b": 40},
}


# =============================================================================
# INDICATOR COLOR MAP
# =============================================================================


# =============================================================================
# MODEL CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class ModelConfig:
    """Model hyperparameters and training settings."""

    # Kronos
    KRONOS_MODEL_SIZE: str = "mini"
    KRONOS_PRED_DAYS: int = 3
    KRONOS_SAMPLE_COUNT: int = 20
    KRONOS_TEMPERATURE: float = 0.8
    KRONOS_CONTEXT_BARS: int = 90

    # XGBoost
    XGBOOST_N_ESTIMATORS: int = 150
    XGBOOST_MAX_DEPTH: int = 3
    XGBOOST_LEARNING_RATE: float = 0.05

    # Training
    MIN_TRAINING_SAMPLES: int = 30
    TRAINING_HISTORY_PERIOD: str = "1y"
    LABEL_AMBIGUITY_THRESHOLD: float = 0.0015  # 0.15%, set to 0.0 to disable
    NEWS_LOOKBACK_MONTHS: int = 3

    # News window formula for live/scheduled predictions. "lookback" is the
    # historical default: everything relevant from the past NEWS lookback
    # days. "overnight" keeps only articles published between the anchor
    # session's close and the target session's open — the premise being that
    # anchor-day intraday news is already priced into the anchor close, so
    # only the overnight tape is NEW information for the move being predicted.
    # Every parameter is an owner-tunable formula input (env-overridable), and
    # jobs can override the mode per run via params_json {"news_filter": ...}.
    NEWS_FILTER_MODE: str = os.getenv("NEWS_FILTER_MODE", "lookback")
    NEWS_OVERNIGHT_START_ET: str = os.getenv("NEWS_OVERNIGHT_START_ET", "16:00")
    NEWS_OVERNIGHT_END_ET: str = os.getenv("NEWS_OVERNIGHT_END_ET", "09:30")
    # The overnight window is short, so it uses the stricter FEATURE-grade
    # relevance bar (matches DEBERTA_RELEVANCE_THRESHOLD) vs the lookback
    # path's looser 0.5.
    NEWS_OVERNIGHT_RELEVANCE: float = float(os.getenv("NEWS_OVERNIGHT_RELEVANCE", "0.7"))
    # Post-filter ceiling per symbol per window, now that fetching paginates
    # past AV's page size instead of truncating at 50.
    NEWS_MAX_ARTICLES: int = int(os.getenv("NEWS_MAX_ARTICLES", "500"))

    # Decision thresholds (shared across Kronos and XGBoost)
    BUY_THRESHOLD: float = 0.55
    SELL_THRESHOLD: float = 0.45

    # No-trade band used to SCORE a HOLD: correct when the target session
    # moved less than the band, i.e. standing aside was right. Without it a
    # HOLD is never right or wrong, so a model can dodge accountability by
    # holding — 221 stored predictions were unscored for exactly this reason.
    #
    # The band is RELATIVE to each symbol's own typical daily move (its median
    # absolute daily return), times the multiplier below. A fixed band cannot
    # work across this universe: the median absolute move here is 1.78%, so a
    # fixed 0.15% band marked 96% of HOLDs wrong and was really measuring
    # volatility rather than the model. At a multiplier of 1.0 a quieter-than-
    # typical session counts as a good HOLD, which makes a coin-flip holder
    # score ~50% and lets a real skill difference show up as deviation from it.
    HOLD_BAND_VOL_MULTIPLE: float = 1.0
    # Fallback when a symbol has too little price history to derive a band.
    HOLD_BAND_PCT: float = 0.0015  # 0.15%
    HOLD_BAND_MIN_HISTORY: int = 20

    # Research-driven "trading_agents" model.
    # sonnet-5: ~2x faster and ~45% cheaper than sonnet-4-6 on report prompts
    # (measured 27s vs 51s); llm_service handles its no-sampling-params and
    # thinking-budget semantics.
    # 2026-07-26 A/B: Luna matched Sonnet on grounding (0 fabrications),
    # flagged data conflicts explicitly, 2x faster, ~2.8x cheaper at
    # $1/$6 per M vs $3/$15 — and research is the N-calls-per-run role.
    TRADING_AGENTS_MODEL: str = "gpt-5.6-luna"
    # Swappable research backend: "single_agent" (in-tree default) or
    # "tradingagents" (adapter over a pinned external release; see
    # models/research_backend.py). Lets us ingest a future TradingAgents version
    # without touching callers.
    RESEARCH_BACKEND: str = os.getenv("RESEARCH_BACKEND", "single_agent")

    # LightGBM
    LIGHTGBM_NUM_LEAVES: int = 31
    LIGHTGBM_LEARNING_RATE: float = 0.05
    LIGHTGBM_N_ESTIMATORS: int = 200

    # DeBERTa Sentiment
    DEBERTA_MODEL_NAME: str = "mrm8488/deberta-v3-ft-financial-news-sentiment-analysis"
    DEBERTA_RELEVANCE_THRESHOLD: float = 0.7
    DEBERTA_BUY_THRESHOLD: float = 0.6
    DEBERTA_SELL_THRESHOLD: float = 0.4

    # Ensemble thresholds
    ENSEMBLE_BUY_THRESHOLD: float = 0.15
    ENSEMBLE_SELL_THRESHOLD: float = -0.15

    # Ensemble defaults (initial UI values — user adjusts from dashboard).
    # These are also what the scheduled pipeline votes with (it passes no
    # ensemble_config), so they must list the full roster: the old 2-model
    # default meant every production ensemble row was a kronos+xgboost vote
    # that saturated at confidence 1.0 whenever the pair agreed.
    ENSEMBLE_DEFAULT_ENABLED: tuple[str, ...] = (
        "kronos_mini",
        "xgboost_shap",
        "lightgbm",
        "deberta_sentiment",
        "trading_agents",
    )
    ENSEMBLE_DEFAULT_WEIGHTS: tuple[tuple[str, float], ...] = (
        ("kronos_mini", 1.0),
        ("xgboost_shap", 1.0),
        ("lightgbm", 1.0),
        ("deberta_sentiment", 1.0),
        ("trading_agents", 1.0),
    )

    # Legacy strategy weights (used by ensemble_vote strategy)
    ENSEMBLE_KRONOS_WEIGHT: float = 1.1
    ENSEMBLE_XGBOOST_WEIGHT: float = 1.3
    ENSEMBLE_TRADING_AGENTS_WEIGHT: float = 0.8

    # Recommendations engine (second-pass synthesis model)
    # 2026-07-26 A/B: Sonnet's synthesis was more diagnostic (says which
    # side to trust and why, two-sided levels); this is ONE call per run
    # so the Luna price advantage is ~3 cents — quality wins here.
    RECOMMENDATIONS_MODEL: str = os.getenv("RECOMMENDATIONS_MODEL", "claude-sonnet-5")
    RECOMMENDATIONS_PROVIDER: str = os.getenv("RECOMMENDATIONS_PROVIDER", "anthropic")
    # 5000: key_level/change_trigger/watch_items grew the JSON; a truncated
    # payload fails the parser and blanks the whole Luna panel. The synthesis
    # is one call for the WHOLE run, so the ceiling has to cover the widest
    # watchlist someone runs — ~200 output tokens per symbol, before a
    # reasoning model spends any of the budget thinking. Env-tunable so a
    # 20-symbol scheduled run doesn't need a code change.
    RECOMMENDATIONS_MAX_TOKENS: int = int(
        os.getenv("RECOMMENDATIONS_MAX_TOKENS", "5000")
    )
    RECOMMENDATIONS_TEMPERATURE: float = 0.3
    RECOMMENDATIONS_REASONING_EFFORT: str = os.getenv(
        "RECOMMENDATIONS_REASONING_EFFORT", "high"
    )


MODEL: Final = ModelConfig()


# Dollars per MILLION tokens, keyed by model id. Used only to price telemetry
# rows — token counts themselves come from the provider response and are exact,
# so a wrong rate here misprices a report but never corrupts usage data. The
# rate applied is copied onto each llm_usage row, so editing this table does
# not rewrite the cost of calls already made.
#
# Seeded from the rates recorded in this repo's own A/B notes (2026-07-26:
# Luna $1/$6 per M, Sonnet $3/$15 per M — see ModelConfig above). VERIFY
# against current provider pricing before treating spend reports as exact.
LLM_PRICING: Final[dict[str, dict[str, float]]] = {
    "gpt-5.6-luna":     {"input": 1.00, "output": 6.00},
    "claude-sonnet-5":  {"input": 3.00, "output": 15.00},
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
}
LLM_PRICING_VERIFIED_ON: Final[str] = "2026-07-26"


def get_llm_rates(model: str | None) -> tuple[float | None, float | None]:
    """(input, output) $/Mtok for a model, or (None, None) when unpriced.

    An unknown model records tokens with a NULL cost rather than a guessed
    one — an unpriced call must be visibly unpriced, not quietly free.
    """
    entry = LLM_PRICING.get((model or "").strip())
    if not entry:
        return None, None
    return entry.get("input"), entry.get("output")


# =============================================================================
# DATABASE CONFIG (Postgres via SQLAlchemy)
# =============================================================================


@dataclass(frozen=True)
class DatabaseConfig:
    """PostgreSQL connection settings."""

    URL: str = os.getenv(
        "DATABASE_URL",
        "postgresql://quantnews:quantnews@localhost:5432/quantnews",
    )
    POOL_SIZE: int = int(os.getenv("DB_POOL_SIZE", "5"))
    MAX_OVERFLOW: int = int(os.getenv("DB_MAX_OVERFLOW", "10"))
    ECHO_SQL: bool = os.getenv("DB_ECHO_SQL", "").lower() == "true"


DB: Final = DatabaseConfig()


# =============================================================================
# OBJECT STORAGE CONFIG (S3-compatible — Railway / MinIO / R2)
# =============================================================================


@dataclass(frozen=True)
class StorageConfig:
    """S3-compatible object storage settings.

    Reads the project's own S3_* names first, then falls back to the standard
    AWS_* ones. Attaching a bucket on Railway injects the AWS_* set — the
    names boto3 and every other S3 client already expect — so without this
    fallback a correctly provisioned bucket looks entirely absent to the app,
    and report archiving fails silently because uploads are best-effort.
    Falling back beats copying credentials into a second set of variables.
    """

    ENDPOINT_URL: str = (
        os.getenv("S3_ENDPOINT_URL")
        or os.getenv("AWS_ENDPOINT_URL")
        or "http://localhost:9000"
    )
    ACCESS_KEY: str = (
        os.getenv("S3_ACCESS_KEY")
        or os.getenv("AWS_ACCESS_KEY_ID")
        or "minioadmin"
    )
    SECRET_KEY: str = (
        os.getenv("S3_SECRET_KEY")
        or os.getenv("AWS_SECRET_ACCESS_KEY")
        or "minioadmin"
    )
    BUCKET_NAME: str = (
        os.getenv("S3_BUCKET_NAME")
        or os.getenv("AWS_S3_BUCKET_NAME")
        or "quantnews-reports"
    )
    REGION: str = (
        os.getenv("S3_REGION")
        or os.getenv("AWS_DEFAULT_REGION")
        or "us-east-1"
    )


STORAGE: Final = StorageConfig()


# =============================================================================
# STRATEGY EVALUATION CONFIG
# =============================================================================


@dataclass(frozen=True)
class StrategyConfig:
    """Strategy evaluation and scheduling settings."""

    # Scheduler (ET timezone, weekdays only)
    EVAL_SCHEDULE_HOUR: int = 16
    EVAL_SCHEDULE_MINUTE: int = 35

    # Position sizing
    DEFAULT_POSITION_SIZE: float = 1000.0

    # Confidence threshold strategy
    CONFIDENCE_THRESHOLD: float = 0.65

    # vectorbt portfolio
    VECTORBT_INIT_CASH: float = 10000.0

    # Minimum trades before showing metrics
    MIN_TRADES_FOR_METRICS: int = 5

    # Minimum trades before Sharpe/Sortino are reported at all. A ratio built
    # from a handful of one-day calls is noise dressed as a number — the old
    # n=5 floor produced Sharpe -11.18 on 2 trades. Below this the counting
    # metrics (win rate, return, trade count) still show; the ratios do not.
    MIN_TRADES_FOR_RATIOS: int = 30


STRATEGY: Final = StrategyConfig()


# =============================================================================
# INDICATOR COLOR MAP
# =============================================================================


INDICATOR_COLORS: Final[dict[str, str]] = {
    "price": COLORS.CHART_1,
    "sma_20": COLORS.CHART_2,
    "sma_50": COLORS.CHART_3,
    "sma_200": COLORS.CHART_4,
    "ema_12": COLORS.CHART_2,
    "ema_26": COLORS.CHART_3,
    "bollinger_upper": COLORS.CHART_5,
    "bollinger_lower": COLORS.CHART_5,
    "bollinger_mid": COLORS.CHART_5,
    "macd_line": COLORS.CHART_1,
    "macd_signal": COLORS.CHART_3,
    "rsi": COLORS.CHART_2,
    "obv": COLORS.CHART_4,
}

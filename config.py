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
    # Calls per minute permitted across every process on this host. A run over
    # a 20-symbol watchlist issues one paginated fetch per symbol back to back,
    # which is what trips the quota; the limiter paces those instead of
    # letting the vendor answer 200-with-an-apology. Free tier is 25 per DAY,
    # so set this to 0 there and rely on caching. 0 disables limiting.
    ALPHA_VANTAGE_CALLS_PER_MIN: int = int(
        os.getenv("ALPHA_VANTAGE_CALLS_PER_MIN", "70"))

    # Request settings
    DEFAULT_TIMEOUT: int = 30

    # LLM client timeout (seconds). Without it a hung socket pins a worker
    # thread for the SDK default of 600s. Synthesis calls run ~1 min.
    LLM_TIMEOUT: int = int(os.getenv("LLM_TIMEOUT", "180"))
    # The investigation stage's tool loop (several web searches + a long
    # JSON answer) runs minutes; it gets its own ceiling.
    INVESTIGATION_TIMEOUT: int = int(os.getenv("INVESTIGATION_TIMEOUT", "600"))


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

    # Chart colors (sequential)
    CHART_1: str = "#00D4AA"  # Primary (price)
    CHART_2: str = "#7B61FF"  # MA-20
    CHART_3: str = "#FF6B6B"  # MA-50
    CHART_4: str = "#4ECDC4"  # MA-200
    CHART_5: str = "#FFE66D"  # Bollinger
    VOLUME_UP: str = "#00C805"
    VOLUME_DOWN: str = "#FF5000"


COLORS: Final = Colors()


# =============================================================================
# PLOTLY CHART THEME
# =============================================================================


CHART_THEME: Final[dict] = {
    "paper_bgcolor": COLORS.BG_PRIMARY,
    "plot_bgcolor": COLORS.BG_PRIMARY,
    "font": {
        "color": COLORS.TEXT_SECONDARY,
        "family": "'Inter', -apple-system, BlinkMacSystemFont, sans-serif",
    },
    "xaxis": {
        "gridcolor": COLORS.BORDER_SUBTLE,
        "linecolor": COLORS.BORDER_SUBTLE,
        "tickfont": {"size": 11},
        "showgrid": True,
        "gridwidth": 1,
        "griddash": "dot",
    },
    "yaxis": {
        "gridcolor": COLORS.BORDER_SUBTLE,
        "linecolor": COLORS.BORDER_SUBTLE,
        "tickfont": {"size": 11},
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
            "family": "'JetBrains Mono', 'SF Mono', monospace",
            "size": 13,
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
    LABEL_AMBIGUITY_THRESHOLD: float = 0.0015  # 0.15%, set to 0.0 to disable
    NEWS_LOOKBACK_MONTHS: int = 3

    # News window formula for live/scheduled predictions. "lookback" is the
    # historical default: everything relevant from the past NEWS lookback
    # days. "overnight" keeps only articles published between the anchor
    # session's close and the target session's open. The premise being that
    # anchor-day intraday news is already priced into the anchor close, so
    # only the overnight tape is NEW information for the move being predicted.
    # Every parameter is an owner-tunable formula input (env-overridable), and
    # jobs can override the mode per run via params_json {"news_filter": ...}.
    NEWS_FILTER_MODE: str = os.getenv("NEWS_FILTER_MODE", "lookback")
    # The one news-window default every entry point (Run dialog, scheduler
    # job form, CLI, library fallback) reads. 14, not 7: the 2026-09-01 BHF
    # report saw 1 of the 5 relevant articles Alpha Vantage held because the
    # dialog's 7-day default cut off the CAO resignation and the Delaware
    # review coverage a week earlier. 14 days is the drawdown-forensics
    # horizon the research prompt was written for.
    # FRONTEND default only: it seeds the Run dialog select, the job form and
    # the seeded daily job. No code path falls back to it. A run whose window
    # did not arrive from one of those raises RunParameterMissing.
    NEWS_LOOKBACK_DAYS: int = int(os.getenv("NEWS_LOOKBACK_DAYS", "14"))
    NEWS_OVERNIGHT_START_ET: str = os.getenv("NEWS_OVERNIGHT_START_ET", "16:00")
    NEWS_OVERNIGHT_END_ET: str = os.getenv("NEWS_OVERNIGHT_END_ET", "09:30")
    # The overnight window is short, so it uses the stricter FEATURE-grade
    # relevance bar (matches DEBERTA_RELEVANCE_THRESHOLD) vs the lookback
    # path's looser 0.5.
    NEWS_OVERNIGHT_RELEVANCE: float = float(os.getenv("NEWS_OVERNIGHT_RELEVANCE", "0.7"))
    # Default per-symbol cap on a run's news window: keep the NEWEST N of
    # the window, 0 = everything the window holds. The Run dialog, the
    # scheduler job form and the CLI all expose this; the trace records
    # when it bites (fetched vs kept, effective span).
    NEWS_MAX_ARTICLES: int = int(os.getenv("NEWS_MAX_ARTICLES", "500"))
    # The durable news store keeps every fetched article this long, then
    # prunes. A daily 7/14/30-day run reads its window from the store and
    # fetches only days it has not seen. The same articles are not paid for
    # again every morning.
    NEWS_RETENTION_DAYS: int = int(os.getenv("NEWS_RETENTION_DAYS", "90"))
    # How many of the window's articles a PROMPT actually reads. Sampled
    # spread across the window (services.news_window.select_spread), never
    # "the newest N", that is what quietly turned a 30-day window into a
    # 3-day one. 0 = every kept article (watch the token bill).
    #   research: per symbol, the trading_agents research report
    #   synthesis: in total across all symbols, the portfolio-level report
    NEWS_PROMPT_ARTICLES: int = int(os.getenv("NEWS_PROMPT_ARTICLES", "25"))
    NEWS_SYNTHESIS_ARTICLES: int = int(os.getenv("NEWS_SYNTHESIS_ARTICLES", "40"))

    # Decision thresholds (shared across Kronos and XGBoost)
    BUY_THRESHOLD: float = 0.55
    SELL_THRESHOLD: float = 0.45

    # No-trade band used to SCORE a HOLD: correct when the target session
    # moved less than the band, i.e. standing aside was right. Without it a
    # HOLD is never right or wrong, so a model can dodge accountability by
    # holding: 221 stored predictions were unscored for exactly this reason.
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
    # $1/$6 per M vs $3/$15, and research is the N-calls-per-run role.
    TRADING_AGENTS_MODEL: str = "gpt-5.6-luna"
    # Evidence blocks the research prompt and synthesis carry by default
    # (the Run dialog checklist, scheduled runs and the CLI all start from
    # this list). Adding a key here changes what every default run reads. 
    # bump PIPELINE_EPOCH alongside.
    #   options: point-in-time put/call positioning
    #   quality: Bad Apples screen + news red flags
    #   investigation: situation classifier + web-researched context
    #                   (deal terms, regulators, key figures), live runs only
    #   political: congressional trades + 13F holder flows (Alpha Vantage);
    #              with `politicians` also on, the congressional half is
    #              left to the dossier and this contributes 13F only
    #   insiders: named executives' Form 4 buys and sells, from the store
    #   politicians: the congressional dossier — who traded this name, who
    #                this run's news names, and what else they are trading
    DEFAULT_EVIDENCE: tuple[str, ...] = ("options", "quality",
                                         "investigation", "political",
                                         "insiders", "politicians")

    # Investigation stage (services/investigation_service.py): one
    # tool-using LLM call per symbol that classifies the situation
    # (pending acquisition, legal overhang, earnings event, momentum only…)
    # and researches it on the open web with citations. gpt-* models run on
    # OpenAI's Responses API + hosted web_search; anything else on
    # Anthropic's web_search server tool (opus-5 measured at ~$1.1/symbol,
    # too dear for a 20-name daily run).
    # Web access is a run TOOL, not an env switch: the Run dialog's Tools
    # section (and the scheduled job's web_research param) turn it on; the
    # backend default is off, so nothing searches unless a run asked.
    INVESTIGATION_MODEL: str = os.getenv("INVESTIGATION_MODEL", "gpt-5.6-luna")
    # Reasoning effort for gpt-* investigators (OpenAI Responses API).
    INVESTIGATION_OPENAI_EFFORT: str = os.getenv("INVESTIGATION_OPENAI_EFFORT", "medium")
    INVESTIGATION_MAX_SEARCHES: int = int(os.getenv("INVESTIGATION_MAX_SEARCHES", "6"))
    # Two-stage triage: every name is classified web-free first (a
    # fraction of a cent); the web search runs only for situations that
    # can be researched. OpenAI bills the hosted search per call on top of
    # tokens, so searching all 20 names a day blew the $0.80/day budget.
    INVESTIGATION_WEB_SKIP: tuple[str, ...] = tuple(
        s.strip().upper() for s in
        os.getenv("INVESTIGATION_WEB_SKIP", "MOMENTUM_ONLY").split(",") if s.strip())
    # Investigations run concurrently ahead of the model loop; each takes
    # minutes, and the scheduled job has a 75-minute ceiling.
    INVESTIGATION_WORKERS: int = int(os.getenv("INVESTIGATION_WORKERS", "4"))
    # Box-wide ceiling on open-web calls in flight, across ALL runs AND all
    # processes: a manual run's model stage is a forked background-callback
    # subprocess, so a per-process counter bounds nothing between users.
    # INVESTIGATION_WORKERS bounds one run; ten people running at once (the
    # concurrency the run flow allows) would open ten times that, plus three
    # question searches per symbol on top, against one API key. Alpha
    # Vantage is covered by the cross-process token bucket; the search
    # providers are not, so this is where that stage is bounded
    # (rate_limiter.FileSemaphore, one flock per slot under
    # RATE_LIMIT_STATE_DIR). Below 1 disables the bound.
    WEB_RESEARCH_CONCURRENCY: int = int(
        os.getenv("WEB_RESEARCH_CONCURRENCY", "6"))
    # Output budget covers thinking + interim text between searches + the
    # JSON; 6000 truncated the first live run before the JSON was written.
    INVESTIGATION_MAX_TOKENS: int = int(os.getenv("INVESTIGATION_MAX_TOKENS", "20000"))
    # Ceiling on anomaly-research questions for a WHOLE run, not per symbol.
    # Anomaly questions are researched inside the serial per-symbol model
    # loop, so the per-symbol cap of three would let a 20-symbol watchlist
    # add 60 web-search turns to a job that already runs ~40 minutes against
    # a 95-minute kill. 18 is six symbols' worth: enough that the names a run
    # actually flags get researched, bounded enough that the job still lands.
    # 0 turns anomaly research off; a run may lift it by passing None.
    ANOMALY_RESEARCH_BUDGET: int = int(
        os.getenv("ANOMALY_RESEARCH_BUDGET", "18"))

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

    # Ensemble defaults (initial UI values. User adjusts from dashboard).
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

    # How member votes are combined. The scheduled pipeline (which passes no
    # ensemble_config) uses this default, so changing it changes production
    # ensemble rows from that day forward.
    #   confidence_weighted: weight x member confidence votes on direction
    #   majority: weight-only votes; member confidence ignored
    #   prob_mean: weighted mean of member up-probabilities
    #   agreement: trade only when >= ENSEMBLE_MIN_AGREE members
    #                         back one direction and none back the other
    ENSEMBLE_DEFAULT_METHOD: str = "confidence_weighted"
    ENSEMBLE_MIN_AGREE: int = 3

    # Legacy strategy weights (used by ensemble_vote strategy)
    ENSEMBLE_KRONOS_WEIGHT: float = 1.1
    ENSEMBLE_XGBOOST_WEIGHT: float = 1.3
    ENSEMBLE_TRADING_AGENTS_WEIGHT: float = 0.8

    # Research/report model (first-pass per-symbol research). Single source
    # of truth: the CLI and the UI both resolve None to this, so changing
    # it here changes scheduled runs without touching the job rows.
    REPORT_MODEL: str = os.getenv("REPORT_MODEL", "gpt-5.6-luna")

    # When the news SOURCE fails (not a quiet week. An outage/rate-limit),
    # news-dependent models abstain (HOLD, zero confidence) instead of
    # calling direction blind. A blind call scored as a real one poisons
    # calibration and the scoreboard alike.
    ABSTAIN_ON_NEWS_UNAVAILABLE: bool = os.getenv(
        "ABSTAIN_ON_NEWS_UNAVAILABLE", "1").strip().lower() not in (
        "0", "false", "no", "off")

    # Recommendations engine (second-pass synthesis model)
    # 2026-07-26 A/B: Sonnet's synthesis was more diagnostic; Luna was equally
    # fabrication-free and 2x faster. Measured live (2026-08-25..09-01) the
    # sonnet-5 synthesis call costs ~$0.186 vs ~$0.03 on Luna — $0.16/day,
    # not the ~3 cents this comment once claimed. Budget directive 2026-09-02:
    # total spend under $0.80/day with headroom for manual runs, so Luna it
    # is; the A/B showed the quality gap is taste, not correctness.
    RECOMMENDATIONS_MODEL: str = os.getenv("RECOMMENDATIONS_MODEL", "gpt-5.6-luna")
    RECOMMENDATIONS_PROVIDER: str = os.getenv("RECOMMENDATIONS_PROVIDER", "openai")
    # Re-asked once on this model when the primary fails after the report
    # and predictions have already been paid for. The row records the model
    # that actually answered (model_used), so a fallback run is visible.
    RECOMMENDATIONS_FALLBACK_MODEL: str = os.getenv(
        "RECOMMENDATIONS_FALLBACK_MODEL", "claude-sonnet-4-6")
    # 5000: key_level/change_trigger/watch_items grew the JSON; a truncated
    # payload fails the parser and blanks the whole Luna panel. The synthesis
    # is one call for the WHOLE run, so the ceiling has to cover the widest
    # watchlist someone runs, ~200 output tokens per symbol, before a
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
# rows: token counts themselves come from the provider response and are exact,
# so a wrong rate here misprices a report but never corrupts usage data. The
# rate applied is copied onto each llm_usage row, so editing this table does
# not rewrite the cost of calls already made.
#
# Seeded from the rates recorded in this repo's own A/B notes (2026-07-26:
# Luna $1/$6 per M, Sonnet $3/$15 per M, see ModelConfig above). VERIFY
# against current provider pricing before treating spend reports as exact.
LLM_PRICING: Final[dict[str, dict[str, float]]] = {
    # GPT-5.6 tiers after OpenAI's 2026-07-30 cut (Luna -80%): Sol $5/$30,
    # Terra $2/$12, Luna $0.20/$1.20. Hosted web_search calls are billed
    # separately per call by OpenAI and are NOT in these token rates.
    "gpt-5.6-luna":     {"input": 0.20, "output": 1.20},
    "gpt-5.6-terra":    {"input": 2.00, "output": 12.00},
    "gpt-5.6-sol":      {"input": 5.00, "output": 30.00},
    # Anthropic list rates (2026-09-02): Opus 5 $5/$25, Sonnet 5 $2/$10,
    # Sonnet 4.6 $3/$15. The investigation stage runs on one of the first
    # two; unpriced rows were hiding its spend.
    "claude-opus-5":    {"input": 5.00, "output": 25.00},
    "claude-sonnet-5":  {"input": 2.00, "output": 10.00},
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
}
LLM_PRICING_VERIFIED_ON: Final[str] = "2026-09-02"

# Server-side web search, in dollars per THOUSAND searches. Billed per call,
# entirely separately from tokens, which is why llm_usage priced tokens only
# and the investigation stage reported about half what it really cost (the
# 2026-09-04 day read $9.67 against a bill near $20; 881 searches were the
# difference). The COUNT that gets priced here is observed from the
# provider's own response; only the rate is a constant, so a correction is
# a one-line change and every stored count re-prices with it.
# Anthropic publishes $10/1000. The OpenAI figure is the same order and is
# the one to check first if the ledger and the invoice disagree.
WEB_SEARCH_PRICING: Final[dict[str, float]] = {
    "anthropic": 10.00,
    "openai": 10.00,
}
WEB_SEARCH_PRICING_VERIFIED_ON: Final[str] = "2026-09-05 (openai rate UNVERIFIED)"


def get_web_search_rate(provider: str | None) -> float | None:
    """$/1000 server-side searches, or None when the provider is unpriced."""
    return WEB_SEARCH_PRICING.get((provider or "").strip().lower())


def get_llm_rates(model: str | None) -> tuple[float | None, float | None]:
    """(input, output) $/Mtok for a model, or (None, None) when unpriced.

    An unknown model records tokens with a NULL cost rather than a guessed
    one: an unpriced call must be visibly unpriced, not quietly free.
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
# OBJECT STORAGE CONFIG (S3-compatible. Railway / MinIO / R2)
# =============================================================================


@dataclass(frozen=True)
class StorageConfig:
    """S3-compatible object storage settings.

    Reads the project's own S3_* names first, then falls back to the standard
    AWS_* ones. Attaching a bucket on Railway injects the AWS_* set, the
    names boto3 and every other S3 client already expect, so without this
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

    # Confidence threshold strategy
    CONFIDENCE_THRESHOLD: float = 0.65

    # Minimum trades before showing metrics
    MIN_TRADES_FOR_METRICS: int = 5

    # Minimum trades before Sharpe/Sortino are reported at all. A ratio built
    # from a handful of one-day calls is noise dressed as a number, the old
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

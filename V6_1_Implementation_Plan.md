# V6.1 — Live Dashboard Integration Plan (Final)

**Date:** 2026-03-09
**Status:** Pre-implementation — all gaps resolved, ready for Phase 1
**Source Research:** V6 Complete Research Summary (TradingAgents_Expansion_V6.md)
**Target:** `quant-news` Dash dashboard

---

## Table of Contents

1. [Research Context](#1-research-context)
2. [Goal & Scope](#2-goal--scope)
3. [Architecture Overview](#3-architecture-overview)
4. [Critical Design Decisions](#4-critical-design-decisions)
5. [Indicator Column Mapping](#5-indicator-column-mapping)
6. [Selected Features (Final 18)](#6-selected-features-final-18)
7. [DuckDB Schema](#7-duckdb-schema)
8. [File Inventory](#8-file-inventory)
9. [Data Flow](#9-data-flow)
10. [Modular Implementation Sequence](#10-modular-implementation-sequence)
11. [Benchmark Tickers & Validation Protocol](#11-benchmark-tickers--validation-protocol)
12. [Ensemble Activation Criteria](#12-ensemble-activation-criteria)
13. [Dependency Management](#13-dependency-management)
14. [Deployment: HuggingFace Spaces](#14-deployment-huggingface-spaces)
15. [Phase Gate Summary](#15-phase-gate-summary)

---

## 1. Research Context

V6 research (TradingAgents project) tested 16 strategies on AMBA over n=30 tri-monthly snapshots (2025-03-10 to 2025-12-31).

### Key Results

| Strategy | 1d Accuracy | P&L | OOS Delta |
|----------|-------------|-----|-----------|
| **E: Kronos-mini + XGBoost SHAP** | **63.3%** | +$482 | **+5.6%** ← only strategy that improves OOS |
| E: Kronos-mini + LightGBM | 60.0% | +$1,787 | -2.8% |
| Kronos-mini (solo) | 56.7% | -$412 | -11.1% |
| XGBoost SHAP-Pruned (solo) | 55.2% | -$461 | -8.8% |
| Always-BUY | 50.0% | benchmark | — |
| LLM (V5, Sonnet 4.5) | 30.0% | -$875 | — |

### Critical Caveat

**63.3% is a hypothesis under test, not a baseline to preserve.** n=30 on a single ticker (AMBA +40% bull run) gives 95% CI of [46.7%, 80.0%]. The SHAP top features (`av_topic_retail_wholesale_avg`, `av_topic_real_estate_avg`) are oddly ticker-agnostic and may represent spurious correlation. The XGBoost features are macro-regime signals, not AMBA fundamentals — which is reason for cautious optimism about generalization, but requires multi-ticker validation to confirm.

### Models Chosen for V6.1

| Model | Type | V6 1d Accuracy | Role |
|-------|------|----------------|------|
| **Kronos-mini** | Time-series foundation model (AAAI 2026) | 56.7% solo | Directional prior from OHLCV patterns |
| **XGBoost SHAP-Pruned** | Gradient-boosted tree, 18 features | 55.2% solo, 63.3% in ensemble | Market regime signal from SPY/sector/AV |
| **LLM Single Agent** | LLM with macro/geopolitical context | 30% (V5, unstructured) | Geopolitical/macro awareness; informational |
| **Ensemble** | Confidence-weighted vote | DEFERRED | Only after per-model validation on 10+ tickers |

### What Was Ruled Out

- **DeBERTa in ensembles:** 73% BUY bias drops 1d accuracy by 16–20pp.
- **Kronos-small:** Worse than mini solo (50% vs 56.7%) and catastrophic in ensemble (36.7%).
- **LLMs for directional accuracy:** Non-deterministic, $0.08–0.10/call, and below Always-BUY.
- **LightGBM (deferred):** Better P&L (+$1,787) but degrades OOS (-2.8%). XGBoost chosen because it is the only strategy proven to improve out-of-sample. LightGBM deferred — see [Section 4.8](#48-xgboost-vs-lightgbm--explicit-deferral).

---

## 2. Goal & Scope

Bring the V6 model stack into the existing `quant-news` Dash dashboard so that:

1. Any ticker the user searches gets a next-day BUY/SELL/HOLD prediction from 3 independent models
2. Predictions are stored in DuckDB and evaluated against actuals on subsequent app launches
3. Model signals appear in per-symbol tabs alongside existing news and AI analysis
4. A market context section in the Overall tab tracks geopolitical, trade, and macro factors
5. The ensemble is deferred until per-model accuracy is validated on 10+ tickers over 30+ trading days

---

## 3. Architecture Overview

```
User adds symbol (e.g. AAPL)
        │
        ▼
stock-data-store updates
(existing: yfinance 1y OHLCV + indicators)
        │
        ▼
generate_model_signals  ← BACKGROUND callback (diskcache subprocess)
        │                  UI stays responsive; progress text updates
        │
        ├─── KronosModel.predict()          [try/except isolated]
        │    Last 90 OHLCV bars → Kronos-mini → PredictionResult
        │
        ├─── XGBoostModel.predict()         [try/except isolated]
        │    ├── Check disk cache (cache/trained_models/{symbol}_xgboost.pkl)
        │    ├── If stale: pre-compute indicators ONCE on full 1y DataFrames
        │    ├── Slide window → ~200 training samples (AV topic features)
        │    ├── Train XGBClassifier (3-fold TimeSeriesSplit CV), save to disk
        │    ├── Build today's 18 features
        │    └── Predict → PredictionResult (with feature_values for DuckDB)
        │
        └─── LLMAgentModel.predict()        [try/except isolated]
             ├── Ticker news (1 week) + global macro news
             ├── Technical signals + OHLCV summary
             ├── LLM generate (LM Studio / OpenAI fallback)
             ├── 3-layer JSON parse (code fences, field validation, fallback)
             └── PredictionResult (raw confidence, confidence_type="self_reported")
        │
        ▼
returns dict (NO DuckDB writes in subprocess)
        │
        ▼
model-signals-store updates → triggers persist_predictions callback (MAIN PROCESS)
        │
        ▼
CacheService.store_prediction()  ← DuckDB write (main process only, write lock)
        │
        ▼
UI renders per-model cards in per-symbol tab
        │
        ▼
Next app launch:
    1. evaluate_predictions(): target_date <= last completed trading day?
    2. JOIN model_predictions ON stock_prices → fill actual_close, was_correct, pnl_dollars
    3. Check ensemble activation criteria
```

---

## 4. Critical Design Decisions

### 4.1 Async Callback Architecture

**Problem:** The prediction callback is synchronous by default. With 4 symbols:
- Kronos: ~1s/prediction after model load (~150MB HuggingFace download on first call)
- XGBoost: Cold start = AV news fetch + ~191 training iterations + train = 2–10s/ticker
- LLM: Network round-trip, 2–5s

**Solution:** `background_callback` with `diskcache` backend (zero-config, no Celery, no Redis).

```python
import diskcache
from dash import DiskcacheManager

bg_cache = diskcache.Cache("cache/dash_bg_callbacks")
background_callback_manager = DiskcacheManager(bg_cache)

@callback(
    Output("model-signals-store", "data"),
    Input("stock-data-store", "data"),
    State("selected-symbols", "data"),
    background=True,
    manager=background_callback_manager,
    running=[
        (Output("model-signals-loading", "style"),
         {"display": "block"}, {"display": "none"}),
    ],
    progress=[Output("model-signals-progress", "children")],
    prevent_initial_call=True,
)
def generate_model_signals(set_progress, stock_data, symbols):
    """
    Runs in background subprocess.
    NEVER writes to DuckDB — returns serializable dict only.
    DuckDB writes happen in persist_predictions() in the main process.
    """
    service = get_prediction_service()
    results = {}
    for i, symbol in enumerate(symbols or []):
        set_progress(f"Predicting {symbol}... ({i+1}/{len(symbols)})")
        df = pd.read_json(StringIO(stock_data[symbol]["prices"]))
        news = fetch_news_cached(symbol)
        results[symbol] = service.predict_symbol_no_store(symbol, df, news=news)
    return results
```

**Per-model isolation:** Each model call is wrapped in `try/except`. If Kronos fails (HuggingFace unreachable, torch missing), XGBoost and LLM still produce results. Failed models return `PredictionResult(error="...", decision="HOLD", confidence=0.0)` rendered as a grey card in the UI.

### 4.2 DuckDB Write Serialization

**Problem:** DuckDB allows only one writer at a time. The background callback runs in a subprocess — concurrent writes produce `IOException`.

**Solution:** All DuckDB writes go through the main process only. The background callback returns a serializable dict to `dcc.Store`. A second callback in the main process receives the store update and writes to DuckDB.

```python
# Callback 1: background subprocess — computes, does NOT write DuckDB
def generate_model_signals(...) -> dict:
    return service.predict_symbol_no_store(...)

# Callback 2: main process — receives store data, writes DuckDB
@callback(
    Output("prediction-store-status", "data"),
    Input("model-signals-store", "data"),
    prevent_initial_call=True,
)
def persist_predictions(signals: dict):
    cache = get_cache()
    for symbol, model_results in (signals or {}).items():
        for model_name, result_dict in model_results.items():
            cache.store_prediction(symbol, model_name, result_dict)
    return {"stored_at": str(datetime.now())}
```

`CacheService` uses a `threading.Lock` to guard writes, and the connection is owned by the main process.

### 4.3 Single-Path Feature Builder

**Problem:** A dual-API surface (`build_features()` vs `build_features_from_precomputed()`) means every future feature addition must be made in two places simultaneously, and divergence is silent.

**Solution:** One function that detects whether indicators are already present and skips recomputation. Uses column mapping constants (see [Section 5](#5-indicator-column-mapping)) for detection.

```python
def _ensure_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
    """Add indicator columns if not already present.

    Detection: checks for INDICATOR_SENTINEL_COLUMN ('SMA_50') which is
    the first non-trivial column added by add_indicators_to_df().
    """
    if INDICATOR_SENTINEL_COLUMN not in df.columns:
        return add_indicators_to_df(df.copy())
    return df  # already enriched — skip recomputation

def build_features(
    self,
    ticker_df: pd.DataFrame,
    spy_df: pd.DataFrame,
    sector_df: pd.DataFrame,
    av_news: list[dict] | None = None,
    global_news: list[dict] | None = None,
) -> dict[str, float]:
    ticker_df = self._ensure_indicators(ticker_df)
    spy_df    = self._ensure_indicators(spy_df)
    sector_df = self._ensure_indicators(sector_df)
    # ... extract features from enriched DataFrames using INDICATOR_COLUMN_MAP ...
```

**Training loop pattern:**
```python
def _train_from_history(self, ticker_df, spy_df, sector_df, av_news_by_date):
    # Pre-compute ONCE — not inside the loop
    ticker_full = add_indicators_to_df(ticker_df.copy())
    spy_full    = add_indicators_to_df(spy_df.copy())
    sector_full = add_indicators_to_df(sector_df.copy())

    for t in range(60, len(ticker_full) - 1):
        # _ensure_indicators() detects 'SMA_50' column and skips recomputation
        features = self._feature_builder.build_features(
            ticker_full.iloc[:t+1],
            spy_full.iloc[:t+1],
            sector_full.iloc[:t+1],
            av_news=av_news_by_date.get(date_str, []),
        )
```

### 4.4 LLM Confidence — Display, Not Calibration

**Problem:** A hardcoded linear transform claims comparability with XGBoost/Kronos outputs. This is false equivalence:
- XGBoost `confidence` = class probability (trained probabilistic output)
- Kronos `confidence` = Monte Carlo sample proportion (20 draws)
- LLM `confidence` = self-reported certainty (clustered at 0.7–0.9 regardless of reliability)

**Solution:** Display LLM confidence in a visually distinct register with honest labeling. Do not transform the value.

```python
# models/llm_agent_model.py
return PredictionResult(
    model_name="llm_agent",
    decision=decision,
    confidence=round(confidence, 2),     # raw, unchanged
    up_probability=round(up_prob, 2),
    details={
        "reasoning": data.get("reasoning", ""),
        "confidence_type": "self_reported",   # consumed by UI for labeling
    },
)

# layouts/signal_components.py — different label for self-reported confidence
confidence_label = (
    "Model Certainty"
    if result["details"].get("confidence_type") == "self_reported"
    else "Confidence"
)
```

**Future calibration:** After accumulating 30+ LLM predictions with actuals in DuckDB, fit empirical Platt scaling using `was_correct` data. Only then apply calibration — with empirical backing, not a hardcoded formula.

### 4.5 Evaluation Gate Logic

**Problem:** Checking `is_market_open_today()` is the wrong condition. A prediction should be evaluated when its target date has passed and the closing price is available.

**Correct logic:** Evaluate when `target_date <= last_completed_trading_day`.

```python
def evaluate_predictions(self) -> int:
    from utils.trading_calendar import get_previous_trading_day, is_market_open_today

    # Last completed trading day: if market open now, yesterday's close is final.
    # If market closed (weekend/holiday/after hours), today's data is final.
    cutoff = (
        get_previous_trading_day(date.today())
        if is_market_open_today()
        else date.today()
    )

    pending = self._conn.execute("""
        SELECT id, symbol, target_date, previous_close, decision
        FROM model_predictions
        WHERE actual_close IS NULL
          AND target_date <= ?
    """, [cutoff]).fetchall()
    # ...
```

### 4.6 Ambiguity Filter — Configurable and Logged

**Problem:** Filtering days where `|pct_change| < 0.15%` is a new assumption not in V6 research. For Consumer Staples tickers (PG), daily moves under 0.5% are common — could remove 15–25% of training samples.

**Solution:** Named config constant, default on, logged impact.

```python
# config.py
@dataclass(frozen=True)
class ModelConfig:
    LABEL_AMBIGUITY_THRESHOLD: float = 0.0015  # 0.15%, set to 0.0 to disable

# models/xgboost_model.py
threshold = MODEL.LABEL_AMBIGUITY_THRESHOLD
skipped = 0
for t in range(60, len(ticker_full) - 1):
    pct = (ticker_full['Close'].iloc[t+1] - ticker_full['Close'].iloc[t]) \
          / ticker_full['Close'].iloc[t]
    if threshold > 0.0 and abs(pct) < threshold:
        skipped += 1
        continue
    # ...

logger.info(
    f"Training {symbol}: {len(rows)} samples "
    f"({skipped} skipped ambiguous, threshold={threshold:.4f})"
)

if len(rows) < MODEL.MIN_TRAINING_SAMPLES:
    raise ValueError(
        f"Insufficient training samples: {len(rows)} < "
        f"{MODEL.MIN_TRAINING_SAMPLES}. Try LABEL_AMBIGUITY_THRESHOLD=0.0"
    )
```

### 4.7 P&L Definition — Explicit and Canonical

**Problem:** Inline P&L formulas in multiple places diverge silently.

**Solution:** Single canonical function in `models/base.py`, used everywhere.

```python
# models/base.py

POSITION_SIZE_USD: float = 1000.0   # fixed notional per trade

DIRECTION_SIGN: dict[str, int] = {
    "BUY":  +1,
    "SELL": -1,
    "HOLD":  0,
}

def compute_pnl(
    decision: str,
    previous_close: float,
    actual_close: float,
    position_size: float = POSITION_SIZE_USD,
) -> float:
    """
    P&L for a fixed-notional directional trade.

    BUY:  long  $1,000 — profit if actual > previous
    SELL: short $1,000 — profit if actual < previous
    HOLD: no position   — P&L = 0

    Returns dollars gained/lost (negative = loss).
    """
    sign = DIRECTION_SIGN.get(decision.upper(), 0)
    if sign == 0 or previous_close <= 0:
        return 0.0
    shares = position_size / previous_close
    return shares * (actual_close - previous_close) * sign
```

All P&L references — `evaluate_predictions()`, validation protocol, unit tests — import and call this function.

### 4.8 XGBoost vs LightGBM — Explicit Deferral

V6 research finding:
- **Kronos-mini + XGBoost:** 63.3% 1d accuracy, +$482 P&L, **+5.6% OOS improvement**
- **Kronos-mini + LightGBM:** 60.0% 1d accuracy, +$1,787 P&L, **-2.8% OOS (degrades)**

XGBoost is chosen because it is the **only strategy proven to improve out-of-sample**. LightGBM's P&L advantage is traced to fewer catastrophic trades, not better directional accuracy. The correct fix for catastrophic trades is position sizing, not switching models.

**DEFERRED:** LightGBM as a model variant after 30-day live validation.

### 4.9 Duplicate Execution Guard

**Problem:** If a user navigates away and back before the background callback completes, Dash fires the callback again.

**Solution: DuckDB check only.** The session cache approach from the prior plan draft is ineffective because `DiskcacheManager` runs callbacks in a **forked subprocess** — the in-memory `_session_cache` dict is not shared with the main process or other subprocess invocations.

```python
# In generate_model_signals (background callback):
# NOTE: This runs in a forked subprocess (DiskcacheManager). DuckDB reads are
# safe (concurrent readers allowed), but writes are NOT — predictions are
# returned as a dict and persisted by the persist_predictions callback in the
# main process (see Section 4.9).
def generate_model_signals(set_progress, stock_data, symbols):
    cache = get_cache()
    results = {}
    for i, symbol in enumerate(symbols or []):
        set_progress(f"Predicting {symbol}... ({i+1}/{len(symbols)})")

        # DuckDB guard: skip if today's predictions already stored
        existing = cache.get_predictions_for_today(symbol)
        if existing:
            results[symbol] = existing
            continue

        df = pd.read_json(StringIO(stock_data[symbol]["prices"]))
        news = fetch_news_cached(symbol)
        results[symbol] = service.predict_symbol_no_store(symbol, df, news=news)
    return results
```

**Note:** The background subprocess CAN read DuckDB (concurrent readers allowed). Only writes are serialized to the main process.

### 4.10 Kronos Vendoring Strategy

Copy 3 files from `../TradingAgents/tradingagents/models/kronos/` with:

**Fix 1 — Version header in `models/kronos/__init__.py`:**
```python
"""Vendored Kronos time-series foundation model.

VENDORED_FROM: ../TradingAgents/tradingagents/models/kronos/
VENDORED_DATE: 2026-03-09
SOURCE: NeoQuasar/Kronos-mini (HuggingFace)
PAPER: AAAI 2026

Do not modify vendored code. To update: re-vendor from TradingAgents,
update this header with new date.
"""
from .kronos import KronosTokenizer, Kronos, KronosPredictor
```

**Fix 2 — Replace `import *` with explicit imports in `kronos.py`:**
```python
# BEFORE (TradingAgents original)
from tradingagents.models.kronos.module import *

# AFTER (quant-news — explicit)
from models.kronos.module import (
    BSQuantizer,
    DependencyAwareLayer,
    FeedForward,
    HierarchicalEmbedding,
    MultiHeadAttentionWithRoPE,
    RMSNorm,
    RotaryPositionalEmbedding,
    TemporalEmbedding,
    TransformerBlock,
)
```

**Fix 3 — Availability flag in `kronos_model.py`:**
```python
try:
    from models.kronos import KronosTokenizer, Kronos, KronosPredictor
    KRONOS_AVAILABLE = True
except ImportError as e:
    KRONOS_AVAILABLE = False
    _KRONOS_IMPORT_ERROR = str(e)
```

### 4.11 torch CPU-Only Build

`pip install torch` pulls the CUDA-enabled build (~2.4GB) on Linux.

**Solution:** Platform-specific instructions in `requirements.txt`, CPU-only explicit in `Dockerfile`.

```
# requirements.txt — torch
# Install torch separately for your platform BEFORE pip install -r requirements.txt:
#   macOS (MPS):       pip install torch
#   Linux (CPU only):  pip install torch --index-url https://download.pytorch.org/whl/cpu
#   Docker:            Handled in Dockerfile (torch+cpu)
torch>=2.0.0
einops>=0.7.0
huggingface_hub>=0.20.0
```

### 4.12 Ensemble Activation Criteria

**Activation requirements — ALL must be met:**

| Criterion | Threshold | Rationale |
|-----------|-----------|-----------|
| Tickers evaluated | ≥ 10 distinct | Cross-sector generalization |
| Trading days per ticker | ≥ 30 each | Statistical meaning (p < 0.1) |
| Per-model 1d accuracy (median) | > 52% for ≥ 2 of 3 models | Better than coin flip + margin |
| No catastrophic model | Each model > 45% on each ticker | No individual failures |
| P&L positive | Median ticker P&L > $0 | Accuracy alone insufficient |
| Prediction diversity | ≥ 20% SELL calls | Not "always BUY" in bull market |

### 4.13 Global Topic Features & Market Context Data Flow

The 18 selected features include `global_topic_*` features. The `build_features()` signature accepts both `av_news` (ticker-specific) and `global_news` (market-wide).

**Data source for global news:** `fetch_alpha_vantage_news(symbol="", topics="economy_macro,economy_fiscal,economy_monetary,financial_markets")`. Cached in `historical_news` table with `symbol = "_GLOBAL"`.

**Market context flow:**
```
App startup / symbol change
    ↓
fetch_global_market_news() → cached in DuckDB (symbol="_GLOBAL")
    ↓
PredictionService.generate_market_context() called ONCE (not per-ticker)
    ↓
LLMAgentModel.predict() receives global_context kwarg
    ↓
Per-symbol tab: LLM card shows reasoning (includes macro context)
Overall tab: Dedicated "Market Context" section
```

### 4.14 Hot-Reload Singleton Survival

Dash debug mode hot-reloads restart the Python process. Pickle-cache trained XGBoost models to disk.

```python
MODELS_CACHE_DIR = Path("cache/trained_models")

def _get_or_train(self, symbol, ...):
    cache_path = MODELS_CACHE_DIR / f"{symbol}_xgboost.pkl"
    meta_path  = MODELS_CACHE_DIR / f"{symbol}_xgboost_meta.json"

    if cache_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("training_date") == str(date.today()):
            model = pickle.loads(cache_path.read_bytes())
            return model

    model, n_samples = self._train_from_history(...)
    MODELS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(pickle.dumps(model))
    meta_path.write_text(json.dumps({
        "training_date": str(date.today()),
        "symbol": symbol,
        "n_samples": n_samples,
    }))
    return model
```

### 4.15 Alpha Vantage — Required, Not Optional

**Decision:** News data for model features comes exclusively from Alpha Vantage. If `ALPHA_VANTAGE_API_KEY` is not set, the XGBoost model and LLM agent raise explicit errors rather than silently producing degraded results with zero-filled features.

**Rationale:** Consistency in training and inference data is paramount. Zero-filled AV topic features during inference (because AV is unavailable) when training used real AV data produces a distribution mismatch that silently degrades predictions. Failing loudly is better than failing silently.

```python
# services/news_service.py
def fetch_historical_av_news(symbol: str, months: int = 3) -> dict[str, list[dict]]:
    """Fetch AV news for past N months, grouped by date.

    Raises:
        ValueError: If ALPHA_VANTAGE_API_KEY is not configured.
    """
    if not API.ALPHA_VANTAGE_API_KEY:
        raise ValueError(
            "ALPHA_VANTAGE_API_KEY required for model training features. "
            "Set it in .env or environment variables."
        )
    # ... pagination logic ...
```

```python
# models/xgboost_model.py
def predict(self, symbol, ohlcv_df, **kwargs):
    if not API.ALPHA_VANTAGE_API_KEY:
        return PredictionResult(
            model_name=self.name, decision="HOLD",
            confidence=0.0, up_probability=0.5, details={},
            error="ALPHA_VANTAGE_API_KEY required for XGBoost features",
        )
```

**AV pagination for 3-month fetch:** AV NEWS_SENTIMENT returns max 1000 articles per call. For 3 months, paginate with monthly `time_from`/`time_to` windows:
```python
def fetch_historical_av_news(symbol: str, months: int = 3) -> dict[str, list[dict]]:
    all_articles = {}
    end = datetime.now()
    for m in range(months):
        # 30-day approximation for monthly windows; overlapping days are
        # deduplicated by article URL when inserting into all_articles dict.
        month_end = end - timedelta(days=30 * m)
        month_start = end - timedelta(days=30 * (m + 1))
        time_from = month_start.strftime("%Y%m%dT0000")
        time_to = month_end.strftime("%Y%m%dT2359")
        articles = fetch_alpha_vantage_news(
            symbol=symbol, max_articles=200,
            time_from=time_from, time_to=time_to, sort="LATEST",
        )
        for article in articles:
            date_key = article.published_at.strftime("%Y-%m-%d")
            all_articles.setdefault(date_key, []).append(article)
    return all_articles
```

Fetched articles are cached in DuckDB `historical_news` table. Subsequent calls for the same symbol+date range hit the cache (< 0.5s).

### 4.16 XGBoost — Raw XGBClassifier, No MLTrainer Vendor

**Decision:** Use raw `xgboost.XGBClassifier` directly instead of vendoring the `MLTrainer` class from TradingAgents.

**Rationale:**
- `MLTrainer` has dependencies on `FinancialSentimentModel`, `data_parsers`, and the TradingAgents cached dataset format — none of which exist in quant-news.
- The actual training logic is ~30 lines: `XGBClassifier(**params).fit(X, y)` + `predict_proba()`.
- `pickle.dump/load` replaces `MLTrainer.save/load` (identical underlying mechanism).
- Fewer imports, no dependency chain, easier to debug.

```python
# models/xgboost_model.py

from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit

XGBOOST_PARAMS = {
    "n_estimators": MODEL.XGBOOST_N_ESTIMATORS,    # 150
    "max_depth": MODEL.XGBOOST_MAX_DEPTH,           # 3
    "learning_rate": MODEL.XGBOOST_LEARNING_RATE,   # 0.05
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "logloss",
    "random_state": 42,
    # Note: use_label_encoder removed — deprecated/removed in XGBoost 2.x
}

def _train(self, X: pd.DataFrame, y: pd.Series) -> XGBClassifier:
    """Train XGBClassifier with TimeSeriesSplit CV."""
    clf = XGBClassifier(**XGBOOST_PARAMS)
    clf.fit(X, y)
    return clf

def _predict(self, clf: XGBClassifier, features: dict) -> PredictionResult:
    """Predict using trained classifier."""
    X = pd.DataFrame([features])[self._selected_features]
    prob = clf.predict_proba(X)[0]
    up_prob = float(prob[1]) if len(prob) > 1 else 0.5

    if up_prob > 0.55:
        decision = "BUY"
    elif up_prob < 0.45:
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
            "model_version": f"xgboost_v1_{date.today():%Y%m%d}",
            "training_samples": self._last_training_samples,
        },
    )
```

**`model_version` format per model:**

| Model | `model_version` value | Example |
|-------|----------------------|---------|
| XGBoost | `f"xgboost_v1_{training_date:%Y%m%d}"` | `xgboost_v1_20260309` |
| Kronos | `"kronos_mini_v1"` (static — vendored, no retraining) | `kronos_mini_v1` |
| LLM Agent | `f"{provider}_{model_id}"` from `llm_service.get_provider_info()` | `lmstudio_local-model` or `openai_gpt-3.5-turbo` |

Each model's `predict()` must populate `details["model_version"]`. The `persist_predictions` callback extracts it and writes to the `model_version` column in DuckDB. This enables debugging which model version produced a given prediction, and detecting when retraining changed behavior.

### 4.17 Sector Map — yfinance Fallback for Unknown Tickers

**Problem:** The hardcoded `SECTOR_ETF_MAP` from TradingAgents only covers ~120 tickers. Benchmark ticker LNT (Alliant Energy) is absent, as are most tickers a user might search.

**Solution:** Hardcoded map for fast lookup + yfinance `info["sector"]` fallback with sector-to-ETF mapping.

```python
# models/sector_map.py

# Sector name → SPDR sector ETF
SECTOR_TO_ETF: dict[str, str] = {
    "Technology": "XLK",
    "Healthcare": "XLV",
    "Financials": "XLF",
    "Consumer Defensive": "XLP",
    "Consumer Cyclical": "XLY",
    "Energy": "XLE",
    "Utilities": "XLU",
    "Industrials": "XLI",
    "Real Estate": "XLRE",
    "Basic Materials": "XLB",
    "Communication Services": "XLC",
    # yfinance sector names can vary — add aliases
    "Consumer Staples": "XLP",
    "Consumer Discretionary": "XLY",
    "Materials": "XLB",
    "Information Technology": "XLK",
    "Health Care": "XLV",
    "Financial Services": "XLF",
}

# Direct ticker → ETF for known tickers (fast path, no API call)
TICKER_TO_ETF: dict[str, str] = {
    "AAPL": "XLK", "MSFT": "XLK", "GOOGL": "XLK", "META": "XLK",
    "NVDA": "SMH", "AMD": "SMH", "AMBA": "SMH", "AVGO": "SMH",
    "JNJ": "XLV", "PFE": "XLV", "UNH": "XLV", "LLY": "XLV",
    "JPM": "XLF", "BAC": "XLF", "GS": "XLF",
    "PG": "XLP", "KO": "XLP", "WMT": "XLP", "COST": "XLP",
    "TSLA": "XLY", "AMZN": "XLY", "HD": "XLY",
    "XOM": "XLE", "CVX": "XLE",
    "NEE": "XLU", "DUK": "XLU",
    "VZ": "XLC", "DIS": "XLC", "NFLX": "XLC",
    # ... expanded from TradingAgents SECTOR_ETF_MAP
}

# NOTE: _sector_cache is an in-process optimization only, not persistent.
# In the background subprocess (DiskcacheManager), entries added here are
# not visible to the main process or future subprocess invocations. This
# means yfinance info() is called once per unknown ticker per background
# callback invocation — negligible for benchmark tickers.
_sector_cache: dict[str, str] = {}

def get_sector_etf(symbol: str) -> str:
    """Get sector ETF for a ticker. Uses hardcoded map first, yfinance fallback.

    Returns:
        Sector ETF symbol (e.g., 'XLK'). Falls back to 'SPY' if sector unknown.
    """
    # Fast path: hardcoded
    if symbol in TICKER_TO_ETF:
        return TICKER_TO_ETF[symbol]

    # Cache check
    if symbol in _sector_cache:
        return _sector_cache[symbol]

    # yfinance fallback
    try:
        import yfinance as yf
        info = yf.Ticker(symbol).info
        sector = info.get("sector", "")
        etf = SECTOR_TO_ETF.get(sector, "SPY")
        _sector_cache[symbol] = etf
        return etf
    except Exception:
        _sector_cache[symbol] = "SPY"
        return "SPY"
```

### 4.18 update_symbol_tabs — Graceful Empty Signals

**Problem:** Adding `Input("model-signals-store", "data")` to `update_symbol_tabs` causes the callback to fire before the background prediction completes (when store is `{}`).

**Solution:** The tab builder renders without model cards when signals are empty, then re-renders when signals arrive. This is the natural Dash reactive pattern.

```python
@callback(
    Output("symbol-tabs-container", "children"),
    Input("news-data-store", "data"),
    Input("ai-analysis-store", "data"),
    Input("selected-symbols", "data"),
    Input("model-signals-store", "data"),   # new input
)
def update_symbol_tabs(news_data, ai_analysis, symbols, model_signals):
    # model_signals may be {} or None — handled gracefully
    sym_signals = (model_signals or {}).get(symbol, {})
    # If sym_signals is empty, the signal cards section is simply not rendered
    # When model-signals-store updates, callback re-fires and cards appear
```

### 4.19 SPY Fetch Failure Handling

**Problem:** XGBoost requires SPY and sector ETF data. If yfinance is rate-limited, the entire XGBoost prediction fails for all symbols in that batch.

**Solution:** SPY and sector data are fetched once and reused across all symbols. If SPY fetch fails, XGBoost returns a `PredictionResult(error=...)` and the other models (Kronos, LLM) continue.

```python
# services/prediction_service.py
def predict_all_symbols(self, symbols, stock_data_dict, ...):
    # Fetch SPY once for all symbols
    try:
        spy_df = fetch_stock_data("SPY", period="1y")
    except Exception as e:
        spy_df = None
        logger.warning(f"SPY fetch failed: {e}. XGBoost will be unavailable.")

    for symbol in symbols:
        # Pass spy_df to XGBoost — if None, model returns error result
        results[symbol] = self.predict_symbol_no_store(
            symbol, ohlcv_df, spy_df=spy_df, ...
        )
```

### 4.20 LM Studio Concurrency

**Problem:** The existing `generate_ai_analysis` callback uses `ThreadPoolExecutor` for parallel LLM calls. The new `generate_model_signals` background callback also calls LLM. If both fire simultaneously, LM Studio (single-threaded inference) may deadlock or produce timeouts.

**Solution:** The LLM agent wraps its `generate()` call with `MODEL.LLM_TIMEOUT_SECONDS` (default 30s, defined in `config.py:ModelConfig`). LM Studio queues requests internally. If LM Studio is saturated, the timeout fires and `LLMAgentModel.predict()` returns a fallback `PredictionResult(error="LLM timeout")`. This is acceptable — the LLM agent is informational, not critical for predictions.

```python
# models/llm_agent_model.py — in predict()
response = llm.generate(
    prompt, system_prompt,
    max_tokens=500,
    temperature=0.3,
)
# The OpenAI client's timeout is set at LLMService init.
# For LM Studio, requests.get() in _is_lm_studio_available() already
# uses timeout=2. The generate() call uses the OpenAI client which
# respects the timeout parameter. Add explicit timeout:

# In LLMService.__init__() or LLMAgentModel.predict():
from config import MODEL
self.client = OpenAI(
    base_url=...,
    api_key=...,
    timeout=MODEL.LLM_TIMEOUT_SECONDS,
)
```

---

## 5. Indicator Column Mapping

The existing `services/analytics.py:add_indicators_to_df()` produces columns with specific names (e.g., `SMA_50`, `RSI`, `ATR`). The TradingAgents feature builder used different names from parsed text sections (e.g., `close_50_sma`, `rsi`, `atr`). The feature names in `selected_features.json` reference the TradingAgents naming convention.

**Solution:** Define an explicit mapping as constants in the feature builder module, imported by all consumers.

```python
# models/feature_builder.py — top-level constants

# Sentinel column to detect if add_indicators_to_df() has been applied.
# SMA_50 is chosen because it requires meaningful data (50 rows) and is
# always present after enrichment.
INDICATOR_SENTINEL_COLUMN: str = "SMA_50"

# Maps quant-news DataFrame column names (from add_indicators_to_df())
# to TradingAgents internal indicator names (used in feature naming).
#
# quant-news column → internal name → produces features:
#   "SMA_50"  → "close_50_sma"  → ind_close_50_sma, ind_close_50_sma_delta5
#   "RSI"     → "rsi"           → ind_rsi, ind_rsi_delta5
#   "ATR"     → "atr"           → ind_atr, ind_atr_delta5
#   "MACD"    → "macd"          → ind_macd, ind_macd_delta5
#   "SMA_200" → "close_200_sma" → ind_close_200_sma, ind_close_200_sma_delta5
#   "BB_Mid"  → "boll"          → ind_boll, ind_boll_delta5
#
# The feature builder reads df[COLUMN_NAME] and emits features as
# f"{prefix}ind_{INTERNAL_NAME}" and f"{prefix}ind_{INTERNAL_NAME}_delta5".

INDICATOR_COLUMN_MAP: dict[str, str] = {
    # DataFrame column name  →  internal feature base name
    "SMA_50":  "close_50_sma",
    "SMA_200": "close_200_sma",
    "RSI":     "rsi",
    "ATR":     "atr",
    "MACD":    "macd",
    "BB_Mid":  "boll",
}

# OHLCV column names in quant-news DataFrames (from yfinance).
# TradingAgents uses lowercase; quant-news uses title-case.
OHLCV_COLUMNS: dict[str, str] = {
    "Open":   "open",
    "High":   "high",
    "Low":    "low",
    "Close":  "close",
    "Volume": "volume",
}
```

**Usage in `_indicator_features()`:**
```python
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
            if len(series) >= 5:
                features[f"{prefix}ind_{internal_name}_delta5"] = float(
                    series.iloc[-1] - series.iloc[-5]
                )
    return features
```

This mapping is the **single source of truth** for column name translation. If `add_indicators_to_df()` column names change, only this mapping needs updating.

---

## 6. Selected Features (Final 18)

The original TradingAgents SHAP analysis selected 20 features from 149. Two features — `sent_neg_pct` and `sent_neutral_pct` — are derived from DeBERTa sentiment scoring, which is not available in the quant-news pipeline (DeBERTa was ruled out due to 73% BUY bias and 99.3% irrelevant article scoring).

**Removed features:**
- `sent_neg_pct` (rank 8, SHAP 0.0976) — DeBERTa negative sentiment percentage
- `sent_neutral_pct` (rank 14, SHAP 0.0922) — DeBERTa neutral sentiment percentage

**Final 18 features** (ordered by SHAP importance):

| # | Feature | SHAP | Category | Source |
|---|---------|------|----------|--------|
| 1 | `spy_gap` | 0.4234 | SPY context | SPY OHLCV: `(open[-1] / close[-2] - 1) * 100` |
| 2 | `vs_spy_excess_5d` | 0.2814 | Relative strength | Ticker 5d return minus SPY 5d return |
| 3 | `av_topic_mergers_and_acquisitions_avg` | 0.2796 | AV topic sentiment | AV news topic score |
| 4 | `av_topic_retail_wholesale_avg` | 0.2146 | AV topic sentiment | AV news topic score |
| 5 | `sector_gap` | 0.2028 | Sector context | Sector ETF OHLCV: `(open[-1] / close[-2] - 1) * 100` |
| 6 | `spy_return_5d` | 0.1540 | SPY context | SPY: `(close[-1] / close[-5] - 1) * 100` |
| 7 | `spy_ind_atr_delta5` | 0.1377 | SPY indicators | SPY ATR 5-day change |
| 8 | `gap` | 0.1237 | Ticker OHLCV | Ticker: `(open[-1] / close[-2] - 1) * 100` |
| 9 | `av_topic_real_estate_avg` | 0.1117 | AV topic sentiment | AV news topic score |
| 10 | `sector_range_5d` | 0.1079 | Sector context | Sector ETF: `(high_5d_max - low_5d_min) / close[-1] * 100` |
| 11 | `global_topic_real_estate_avg` | 0.1028 | Global news | Global AV news topic score |
| 12 | `ind_close_50_sma_delta5` | 0.0933 | Ticker indicators | Ticker SMA_50 5-day change |
| 13 | `global_topic_retail_wholesale_avg` | 0.0888 | Global news | Global AV news topic score |
| 14 | `spy_ind_rsi` | 0.0859 | SPY indicators | SPY RSI latest value |
| 15 | `av_n_tech_articles` | 0.0781 | AV metadata | Count of technology-tagged AV articles. **Unbounded integer** (not [-1,1] like topic scores). Scale-invariant for XGBoost tree splits but high variance across tickers. |
| 16 | `av_topic_ipo_avg` | 0.0726 | AV topic sentiment | AV news topic score |
| 17 | `spy_ind_close_50_sma_delta5` | 0.0690 | SPY indicators | SPY SMA_50 5-day change |
| 18 | `global_topic_energy_transportation_avg` | 0.0665 | Global news | Global AV news topic score |

**Feature category breakdown:**
- **SPY context (6):** spy_gap, vs_spy_excess_5d, spy_return_5d, spy_ind_atr_delta5, spy_ind_rsi, spy_ind_close_50_sma_delta5
- **Sector context (2):** sector_gap, sector_range_5d
- **Ticker OHLCV (1):** gap
- **Ticker indicators (1):** ind_close_50_sma_delta5
- **AV ticker news (4):** av_topic_mergers_and_acquisitions_avg, av_topic_retail_wholesale_avg, av_topic_real_estate_avg, av_topic_ipo_avg
- **AV metadata (1):** av_n_tech_articles — **NOTE: this is an unbounded count, not a [-1, 1] score.** Scale differs from `av_topic_*_avg` features. XGBoost handles mixed scales natively (tree splits are scale-invariant), but the variance can be large for high-coverage tickers (e.g., JNJ may have 20+ tech articles some days, 0 others). Document in `_av_topic_features()` docstring.
- **Global news (3):** global_topic_real_estate_avg, global_topic_retail_wholesale_avg, global_topic_energy_transportation_avg

**`models/selected_features.json`:**
```json
{
  "version": 2,
  "derived_from": "TradingAgents V6 SHAP analysis (n=28, AMBA)",
  "n_original": 20,
  "n_selected": 18,
  "removed": ["sent_neg_pct", "sent_neutral_pct"],
  "removal_reason": "DeBERTa-derived; DeBERTa excluded from V6.1 pipeline",
  "features": [
    "spy_gap",
    "vs_spy_excess_5d",
    "av_topic_mergers_and_acquisitions_avg",
    "av_topic_retail_wholesale_avg",
    "sector_gap",
    "spy_return_5d",
    "spy_ind_atr_delta5",
    "gap",
    "av_topic_real_estate_avg",
    "sector_range_5d",
    "global_topic_real_estate_avg",
    "ind_close_50_sma_delta5",
    "global_topic_retail_wholesale_avg",
    "spy_ind_rsi",
    "av_n_tech_articles",
    "av_topic_ipo_avg",
    "spy_ind_close_50_sma_delta5",
    "global_topic_energy_transportation_avg"
  ]
}
```

---

## 7. DuckDB Schema

### `model_predictions`

```sql
CREATE TABLE IF NOT EXISTS model_predictions (
    -- Primary key: "{symbol}_{model_name}_{prediction_date_YYYYMMDD}"
    -- e.g. "AAPL_kronos_mini_20260307"
    id VARCHAR PRIMARY KEY,

    symbol VARCHAR NOT NULL,
    model_name VARCHAR NOT NULL,
    prediction_date DATE NOT NULL,   -- when prediction was generated
    target_date DATE NOT NULL,       -- next trading day being predicted

    -- Prediction output
    decision VARCHAR NOT NULL,       -- BUY / SELL / HOLD
    confidence DOUBLE,
    up_probability DOUBLE,
    predicted_close DOUBLE,
    previous_close DOUBLE,           -- close on prediction_date

    -- Evaluation (filled at next launch after target_date passes)
    actual_close DOUBLE,
    was_correct BOOLEAN,
    pnl_dollars DOUBLE,              -- compute_pnl(decision, previous_close, actual_close)

    -- Debugging & reproducibility
    model_version VARCHAR,           -- "xgboost_v1_20260307", "kronos_mini_v1",
                                     -- "openai_gpt-4o-mini", "lmstudio_local"
    training_samples INTEGER,        -- XGBoost only; NULL for Kronos/LLM
    feature_values_json VARCHAR,     -- full feature vector (XGBoost only)
    details_json VARCHAR,            -- model-specific extras

    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    evaluated_at TIMESTAMP
)
```

### `historical_news`

```sql
CREATE TABLE IF NOT EXISTS historical_news (
    -- Dedup key: md5(url + published_date)
    id VARCHAR PRIMARY KEY,

    symbol VARCHAR NOT NULL,         -- ticker or "_GLOBAL" for global news
    published_date DATE,
    title VARCHAR,
    summary VARCHAR,
    url VARCHAR,
    source VARCHAR,
    topics_json VARCHAR,             -- [{"topic": "Technology", "relevance_score": "0.85"}]
    overall_sentiment_score DOUBLE,
    overall_sentiment_label VARCHAR,
    ticker_sentiment_score DOUBLE,   -- AV ticker-specific (if relevance >= 0.7)
    ticker_relevance_score DOUBLE,
    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
```

---

## 8. File Inventory

### New Files

| File | Purpose | Phase |
|------|---------|-------|
| `models/__init__.py` | Package init | 1 |
| `models/base.py` | `BaseModel` ABC, `PredictionResult`, `compute_pnl`, `DIRECTION_SIGN` | 1 |
| `models/registry.py` | `ModelRegistry` | 1 |
| `utils/trading_calendar.py` | `get_next_trading_day`, `get_previous_trading_day`, `is_market_open_today` | 1 |
| `models/sector_map.py` | `SECTOR_TO_ETF`, `TICKER_TO_ETF`, `get_sector_etf()` with yfinance fallback | 3 |
| `models/feature_builder.py` | `LiveFeatureBuilder`, `INDICATOR_COLUMN_MAP`, `OHLCV_COLUMNS`, `INDICATOR_SENTINEL_COLUMN` | 4 |
| `models/selected_features.json` | 18 SHAP features (no DeBERTa, version 2) | 4 |
| `models/kronos/__init__.py` | Vendored Kronos — version header, explicit exports | 6 |
| `models/kronos/kronos.py` | Vendored Kronos model — explicit imports (no `import *`) | 6 |
| `models/kronos/module.py` | Vendored Kronos modules | 6 |
| `models/kronos_model.py` | `KronosModel(BaseModel)`, `KRONOS_AVAILABLE` flag, lazy loading | 6 |
| `models/xgboost_model.py` | `XGBoostModel(BaseModel)`, raw `XGBClassifier`, disk-cached training | 5 |
| `models/llm_agent_model.py` | `LLMAgentModel(BaseModel)`, 3-layer parse, display-honest confidence | 7 |
| `models/ensemble_model.py` | `EnsembleModel(BaseModel)` — **DEFERRED** | — |
| `services/prediction_service.py` | Orchestrator, singleton, per-model isolation, SPY prefetch | 8 |
| `layouts/signal_components.py` | Signal cards, history table, market context section | 9 |

### Modified Files

| File | Change | Phase |
|------|--------|-------|
| `config.py` | Add `ModelConfig` dataclass | 1 |
| `requirements.txt` / `pyproject.toml` | torch, einops, huggingface_hub, xgboost, scikit-learn, diskcache, pandas_market_calendars | 1 |
| `services/cache_service.py` | `model_predictions` + `historical_news` tables, `store_prediction`, `evaluate_predictions`, `get_model_accuracy`, `get_predictions_for_today`, `has_predictions_for_today` | 2 |
| `services/news_service.py` | `fetch_historical_av_news()`, `fetch_global_market_news()` — AV-only, raises if no key | 3 |
| `layouts/main_layout.py` | `model-signals-store`, `prediction-store-status`, loading div, progress div | 9 |
| `app.py` | Background callback, `persist_predictions` callback, `update_symbol_tabs` new Input, startup `evaluate_predictions()` | 9 |
| `assets/styles.css` | Signal card, history table, loading state, market context classes | 9 |

---

## 9. Data Flow

### Key Contracts

| Contract | Detail |
|----------|--------|
| Background subprocess → main process | Returns `dict[str, dict]` only. Never writes DuckDB. |
| Main process `persist_predictions` | Only callback that writes `model_predictions`. Holds `threading.Lock`. |
| `compute_pnl` | Single function in `models/base.py`. All P&L computation uses this. |
| `evaluate_predictions` gate | `target_date <= last_completed_trading_day` — not `is_market_open_today`. |
| Duplicate guard | DuckDB check in background callback (concurrent reads OK). No session cache. |
| Feature consistency | `_ensure_indicators` with `INDICATOR_SENTINEL_COLUMN` — same code path for training and inference. |
| Column name mapping | `INDICATOR_COLUMN_MAP` in `feature_builder.py` — single source of truth. |
| News data source | Alpha Vantage exclusively. `ValueError` if API key missing. |
| XGBoost implementation | Raw `XGBClassifier` + `pickle`. No `MLTrainer` vendor. |
| Sector resolution | `TICKER_TO_ETF` hardcoded → yfinance `info["sector"]` fallback → `SPY` default. |

---

## 10. Modular Implementation Sequence

Each phase is independently testable. Each gate test must pass before proceeding.

---

### Phase 1: Foundation — Base + Config + Registry

**Scope:** Pure Python. Zero model dependencies.

**Files:**
- `models/__init__.py`
- `models/base.py` — `PredictionResult`, `BaseModel`, `compute_pnl`, `DIRECTION_SIGN`, `POSITION_SIZE_USD`
- `models/registry.py` — `ModelRegistry`
- `utils/trading_calendar.py` — `get_next_trading_day`, `get_previous_trading_day`, `is_market_open_today`
- `config.py` — add `ModelConfig` dataclass
- `requirements.txt` — add all new dependencies with torch platform note

**`config.py` — ModelConfig:**
```python
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
    LABEL_AMBIGUITY_THRESHOLD: float = 0.0015  # set to 0.0 to disable
    NEWS_LOOKBACK_MONTHS: int = 3

    # Decision thresholds (shared across Kronos and XGBoost)
    BUY_THRESHOLD: float = 0.55
    SELL_THRESHOLD: float = 0.45

    # LLM Agent
    LLM_TIMEOUT_SECONDS: int = 30  # max_tokens generation timeout

    # Ensemble (DEFERRED)
    ENSEMBLE_BUY_THRESHOLD: float = 0.15
    ENSEMBLE_SELL_THRESHOLD: float = -0.15
    ENSEMBLE_KRONOS_WEIGHT: float = 1.1
    ENSEMBLE_XGBOOST_WEIGHT: float = 1.3
    ENSEMBLE_LLM_WEIGHT: float = 0.8

MODEL: Final = ModelConfig()
```

**Gate test:**
```python
from models.base import PredictionResult, compute_pnl
from models.registry import ModelRegistry
from utils.trading_calendar import get_next_trading_day

# Trading calendar
assert get_next_trading_day("2026-03-06").strftime("%Y-%m-%d") == "2026-03-09"  # Fri→Mon
assert get_next_trading_day("2026-03-07").strftime("%Y-%m-%d") == "2026-03-09"  # Sat→Mon

# P&L math
assert compute_pnl("BUY",  100.0, 105.0) == 50.0     # 10 shares * $5
assert compute_pnl("SELL", 100.0, 95.0)  == 50.0     # 10 shares * $5 (short)
assert compute_pnl("HOLD", 100.0, 105.0) == 0.0
assert compute_pnl("BUY",  100.0, 95.0)  == -50.0

# Registry
class MockModel(BaseModel):
    @property
    def name(self): return "mock"
    def is_ready(self): return True
    def predict(self, symbol, ohlcv_df, **kwargs):
        return PredictionResult("mock", "HOLD", 0.5, 0.5, {})

registry = ModelRegistry()
registry.register(MockModel())
assert registry.get("mock") is not None
assert len(registry.list_models()) == 1
```

---

### Phase 2: DuckDB Schema

**Scope:** Schema + round-trip tests. No model code.

**Files:** `services/cache_service.py` — add tables, `store_prediction`, `has_predictions_for_today`, `get_predictions_for_today`, `evaluate_predictions`, `get_prediction_history`, `get_model_accuracy`

**Gate test:**
```python
from services.cache_service import get_cache
from models.base import compute_pnl
from datetime import date, timedelta

cache = get_cache()

# --- Verify stock_prices table exists (needed for evaluation JOIN) ---
tables = cache._conn.execute("SHOW TABLES").fetchall()
table_names = [t[0] for t in tables]
assert "model_predictions" in table_names
assert "stock_prices" in table_names  # must exist for evaluate_predictions JOIN

# --- Store + Retrieve ---
mock_result = {
    "model_name": "test_model", "decision": "BUY",
    "confidence": 0.70, "up_probability": 0.65,
    "details": {"reasoning": "test"}, "error": None,
    "model_version": "test_v1",
}
cache.store_prediction("AAPL", "test_model", mock_result)
assert cache.has_predictions_for_today("AAPL")
rows = cache.get_prediction_history("AAPL", limit=1)
assert rows[0]["decision"] == "BUY"

# --- Dedup: upsert, not duplicate ---
cache.store_prediction("AAPL", "test_model", mock_result)
rows2 = cache.get_prediction_history("AAPL", limit=10)
assert len(rows2) == 1

# --- Evaluation path (most complex) ---
# Backdate target_date to yesterday so it qualifies for evaluation
yesterday = (date.today() - timedelta(days=1)).isoformat()
cache._conn.execute("""
    UPDATE model_predictions
    SET previous_close = 150.0, target_date = ?
    WHERE symbol = 'AAPL' AND model_name = 'test_model'
""", [yesterday])

# Simulate actual close being available in stock_prices
# (In production, evaluate_predictions JOINs on stock_prices table)
cache._conn.execute("""
    INSERT OR REPLACE INTO stock_prices (symbol, date, close)
    VALUES ('AAPL', ?, 155.0)
""", [yesterday])

count = cache.evaluate_predictions()
assert count == 1, f"Expected 1 evaluation, got {count}"

rows3 = cache.get_prediction_history("AAPL", limit=1)
assert rows3[0]["was_correct"] == True      # BUY and price went up
assert rows3[0]["actual_close"] == 155.0
assert rows3[0]["evaluated_at"] is not None

# P&L uses canonical compute_pnl — verify match
expected_pnl = compute_pnl("BUY", 150.0, 155.0)
assert abs(rows3[0]["pnl_dollars"] - expected_pnl) < 0.01, \
    f"P&L mismatch: {rows3[0]['pnl_dollars']} vs {expected_pnl}"

# --- Accuracy aggregation ---
accuracy = cache.get_model_accuracy("test_model", symbol="AAPL")
assert accuracy["total"] == 1
assert accuracy["correct"] == 1
assert accuracy["accuracy"] == 1.0
```

---

### Phase 3: Sector Map + News Infrastructure

**Scope:** News fetching and DuckDB caching. AV-only — no yfinance news fallback.

**Files:**
- `models/sector_map.py`
- `services/news_service.py` — add `fetch_historical_av_news`, `fetch_global_market_news`

**Gate test:**
```python
from models.sector_map import get_sector_etf
from config import API
import time

# Sector resolution — hardcoded
assert get_sector_etf("AAPL") == "XLK"
assert get_sector_etf("JNJ") == "XLV"
assert get_sector_etf("VZ") == "XLC"

# Sector resolution — yfinance fallback (LNT not in hardcoded map)
etf = get_sector_etf("LNT")
assert etf == "XLU", f"Expected XLU for LNT (Utilities), got {etf}"

# News fetch — requires AV premium key
assert API.ALPHA_VANTAGE_API_KEY, "AV key required"
from services.news_service import fetch_historical_av_news
news = fetch_historical_av_news("AAPL", months=1)
assert isinstance(news, dict)
assert len(news) > 0

# ValueError when no key
# IMPLEMENTATION NOTE: APIConfig is a frozen dataclass — os.environ override
# won't propagate to API.ALPHA_VANTAGE_API_KEY. fetch_historical_av_news must
# read os.environ.get("ALPHA_VANTAGE_API_KEY") directly at call time, NOT
# through the frozen config object, for this test to work correctly.
import os
original_key = os.environ.get("ALPHA_VANTAGE_API_KEY")
try:
    os.environ["ALPHA_VANTAGE_API_KEY"] = ""
    # Must raise — not silently return empty
    raised = False
    try:
        fetch_historical_av_news("AAPL", months=1)
    except ValueError:
        raised = True
    assert raised, "Should raise ValueError when AV key missing"
finally:
    if original_key:
        os.environ["ALPHA_VANTAGE_API_KEY"] = original_key

# Cache hit — second call is fast
t0 = time.time()
news2 = fetch_historical_av_news("AAPL", months=1)
assert time.time() - t0 < 0.5, "Cache miss"
```

---

### Phase 4: Feature Builder

**Scope:** Feature extraction with single-path design, column mapping constants.

**Files:**
- `models/feature_builder.py` — includes `INDICATOR_COLUMN_MAP`, `OHLCV_COLUMNS`, `INDICATOR_SENTINEL_COLUMN`
- `models/selected_features.json`

**Gate test:**
```python
from models.feature_builder import (
    LiveFeatureBuilder,
    INDICATOR_COLUMN_MAP,
    INDICATOR_SENTINEL_COLUMN,
)
from services.stock_data import fetch_stock_data
from services.analytics import add_indicators_to_df
import json

aapl = fetch_stock_data("AAPL", period="1y")
spy  = fetch_stock_data("SPY",  period="1y")
xlk  = fetch_stock_data("XLK",  period="1y")

builder = LiveFeatureBuilder()

# Path A: raw OHLCV (live prediction path)
features_raw = builder.build_features(aapl, spy, xlk)

# Path B: pre-computed indicators (training path)
aapl_enriched = add_indicators_to_df(aapl.copy())
spy_enriched  = add_indicators_to_df(spy.copy())
xlk_enriched  = add_indicators_to_df(xlk.copy())
features_pre  = builder.build_features(aapl_enriched, spy_enriched, xlk_enriched)

# Verify sentinel detection works
assert INDICATOR_SENTINEL_COLUMN in aapl_enriched.columns
assert INDICATOR_SENTINEL_COLUMN not in aapl.columns

# Parity: both paths produce identical features
with open("models/selected_features.json") as f:
    selected = json.load(f)["features"]

for feat in selected:
    assert feat in features_raw, f"Missing: {feat}"
    assert feat in features_pre, f"Missing precomputed: {feat}"
    delta = abs(features_raw[feat] - features_pre[feat])
    assert delta < 1e-6, f"Mismatch {feat}: {features_raw[feat]} vs {features_pre[feat]}"

# Zero-fill for missing AV news
features_no_news = builder.build_features(aapl, spy, xlk, av_news=[])
assert features_no_news["av_topic_mergers_and_acquisitions_avg"] == 0.0

# Column mapping consistency
for col_name in INDICATOR_COLUMN_MAP:
    assert col_name in aapl_enriched.columns, \
        f"INDICATOR_COLUMN_MAP key '{col_name}' not in enriched DataFrame"
```

---

### Phase 5: XGBoost Model

**Scope:** Walk-forward training with raw `XGBClassifier`, disk cache, sparse feature warnings.

**Files:** `models/xgboost_model.py`

**Gate test:**
```python
from models.xgboost_model import XGBoostModel
from services.stock_data import fetch_stock_data
import time

model = XGBoostModel()
aapl = fetch_stock_data("AAPL", period="1y")

# Cold start — must train in < 15s
t0 = time.time()
result = model.predict("AAPL", aapl)
cold_time = time.time() - t0
print(f"Cold start: {cold_time:.1f}s")
assert cold_time < 15.0, f"Too slow: {cold_time:.1f}s"

assert result.decision in ("BUY", "SELL", "HOLD")
assert 0.0 <= result.confidence <= 1.0
assert result.error is None

# Warm start — disk cache hit
t0 = time.time()
result2 = model.predict("AAPL", aapl)
warm_time = time.time() - t0
assert warm_time < 2.0, f"Warm too slow: {warm_time:.1f}s"
assert result.decision == result2.decision   # deterministic
```

---

### Phase 6: Kronos Model

**Scope:** Vendored Kronos with explicit imports, availability flag, lazy loading.

**Files:**
- `models/kronos/__init__.py`, `models/kronos/kronos.py`, `models/kronos/module.py`
- `models/kronos_model.py`

**Gate test:**
```python
from models.kronos_model import KronosModel, KRONOS_AVAILABLE
from services.stock_data import fetch_stock_data

if not KRONOS_AVAILABLE:
    print("Kronos not available — install: pip install torch einops")
else:
    model = KronosModel()
    assert model.is_ready()

    aapl = fetch_stock_data("AAPL", period="1y")
    result = model.predict("AAPL", aapl)

    assert result.decision in ("BUY", "SELL", "HOLD")
    assert result.model_name == "kronos_mini"
    assert result.error is None
    assert 0.0 <= result.confidence <= 1.0
```

---

### Phase 7: LLM Agent Model

**Scope:** LLM wrapper with 3-layer JSON parse, display-honest confidence. Parse testable without network.

**Files:** `models/llm_agent_model.py`

**Gate test (parse logic — no LLM needed):**
```python
from models.llm_agent_model import LLMAgentModel

model = LLMAgentModel()

cases = [
    ('{"decision":"BUY","confidence":0.8,"up_probability":0.7,"reasoning":"bullish"}',
     "BUY"),
    ('```json\n{"decision":"SELL","confidence":0.6,"up_probability":0.3}\n```',
     "SELL"),
    ('{"decision":"STRONG_BUY","confidence":0.9,"up_probability":0.8}',
     "HOLD"),   # invalid decision → HOLD
    ('{"decision":"HOLD"}',
     "HOLD"),   # missing fields → defaults
    ("not json at all",
     "HOLD"),   # garbage → fallback
    (None,
     "HOLD"),   # no response → fallback
]

for raw, expected in cases:
    result = model._parse_llm_prediction(raw)
    assert result.decision == expected, f"Input: {raw!r} → {result.decision}, expected {expected}"
    assert 0.0 <= result.confidence <= 1.0
    assert result.details.get("confidence_type") == "self_reported"
    if raw is None or "not json" in str(raw):
        assert result.error is not None
```

---

### Phase 8: Prediction Service

**Scope:** Orchestrator wiring all models with isolation and no-store contract.

**Files:** `services/prediction_service.py`

**Gate test:**
```python
from services.prediction_service import get_prediction_service
from services.stock_data import fetch_stock_data
from unittest.mock import patch

service = get_prediction_service()
aapl = fetch_stock_data("AAPL", period="1y")

# Simulate Kronos failure — other models must still return
kronos = service._registry.get("kronos_mini")
if kronos:
    with patch.object(kronos, "predict", side_effect=RuntimeError("Simulated")):
        results = service.predict_symbol_no_store("AAPL", aapl)

    assert results["kronos_mini"]["error"] is not None
    assert results["kronos_mini"]["decision"] == "HOLD"
    assert results["xgboost_shap"]["error"] is None
```

---

### Phase 9: UI + Callbacks

**Scope:** Background callback, `persist_predictions`, signal components, startup evaluation.

**Files:**
- `layouts/signal_components.py`
- `layouts/main_layout.py` — add stores and loading elements
- `app.py` — background callback, `persist_predictions`, modify `update_symbol_tabs`
- `assets/styles.css` — signal card classes

**Integration test:**
```python
from services.prediction_service import get_prediction_service
from services.cache_service import get_cache
from services.stock_data import fetch_stock_data

service = get_prediction_service()
cache   = get_cache()
aapl    = fetch_stock_data("AAPL", period="1y")

# Step 1: Predict (no DuckDB write)
results = service.predict_symbol_no_store("AAPL", aapl)
assert all(m in results for m in ["kronos_mini", "xgboost_shap", "llm_agent"]
           if service._registry.get(m) and service._registry.get(m).is_ready())

# Step 2: Persist (main process contract)
for model_name, result_dict in results.items():
    cache.store_prediction("AAPL", model_name, result_dict)

history = cache.get_prediction_history("AAPL")
assert len(history) >= 1
```

---

### Phase 10: Docker / HF Spaces

**Scope:** Containerization. Only after all Phase 1–9 gate tests pass locally.

**Files:** `Dockerfile`, `.dockerignore`

```dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    torch==2.2.0+cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-bake Kronos weights — zero runtime downloads
RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('NeoQuasar/Kronos-mini', cache_dir='/app/models_cache'); \
snapshot_download('NeoQuasar/Kronos-Tokenizer-base', cache_dir='/app/models_cache')"

ENV HF_HOME=/app/models_cache
ENV TRANSFORMERS_OFFLINE=1
ENV HF_HUB_OFFLINE=1

COPY . .
RUN mkdir -p /app/cache/trained_models /app/cache/dash_bg_callbacks

EXPOSE 8050
CMD ["python", "app.py"]
```

**`.dockerignore`:**
```
cache/
.env
__pycache__/
*.pyc
.git/
*.egg-info/
.mypy_cache/
.pytest_cache/
```

---

## 11. Benchmark Tickers & Validation Protocol

### Tickers

| Ticker | Sector | Rationale |
|--------|--------|-----------|
| **PG** | Consumer Staples | Defensive, low-vol — tests sparse AV features, low-signal regime |
| **LNT** | Utilities | Rate-sensitive — tests yfinance sector fallback, macro sensitivity |
| **JNJ** | Healthcare | High news coverage — tests relevance filtering |
| **VZ** | Communication Services | Dividend play — tests M&A feature near-zero rate |

### Per-Ticker Pass Criteria (30 trading days)

| Metric | Threshold |
|--------|-----------|
| 1d accuracy | > 52% |
| P&L (equal-weight $1000) | > $0 |
| SELL accuracy | > 40% |
| Prediction coverage | > 80% (not always HOLD) |

---

## 12. Ensemble Activation Criteria

See [Section 4.12](#412-ensemble-activation-criteria). All 6 criteria must be met. Checked at startup via `CacheService.get_model_accuracy()`.

---

## 13. Dependency Management

### New Dependencies

| Package | Purpose | Size |
|---------|---------|------|
| `torch>=2.0.0` | Kronos model inference | ~200MB (CPU) / ~2.4GB (CUDA) |
| `einops>=0.7.0` | Kronos tensor operations | ~100KB |
| `huggingface_hub>=0.20.0` | Kronos model download | ~1MB |
| `xgboost>=2.0.0` | XGBoost classifier | ~80MB |
| `scikit-learn>=1.3.0` | TimeSeriesSplit CV | ~30MB |
| `diskcache>=5.6.0` | Background callback manager | ~200KB |
| `pandas_market_calendars>=4.0.0` | NYSE trading calendar | ~5MB |

### Installation Note

```bash
# macOS (Apple Silicon — MPS acceleration)
pip install torch
pip install -r requirements.txt

# Linux (CPU only — smaller)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

---

## 14. Deployment: HuggingFace Spaces

### Key Design Decisions

- **Pre-baked Kronos weights** in Docker image layer. `HF_HUB_OFFLINE=1` prevents all runtime downloads.
- **torch CPU-only** via `+cpu` index URL (~200MB vs ~2.4GB).
- **Persistent storage** for DuckDB, trained XGBoost models, and diskcache. Survives container restarts.
- **AV + OpenAI keys** as HF Spaces secrets (environment variables).

### Persistent Storage Config

```python
# config.py
import os
PERSISTENT_DIR      = os.getenv("PERSISTENT_DIR", "cache")
DUCKDB_PATH         = f"{PERSISTENT_DIR}/quant_news.duckdb"
TRAINED_MODELS_DIR  = f"{PERSISTENT_DIR}/trained_models"
DASH_BG_CACHE_DIR   = f"{PERSISTENT_DIR}/dash_bg_callbacks"
```

### HF Spaces README Header

```yaml
---
title: QuantNews Trading Signals
emoji: 📈
colorFrom: green
colorTo: blue
sdk: docker
app_port: 8050
pinned: false
---
```

### Estimated Image Size

| Layer | Size |
|-------|------|
| python:3.11-slim | ~150MB |
| torch CPU-only | ~200MB |
| xgboost + scikit-learn | ~80MB |
| dash + plotly + pandas + duckdb | ~120MB |
| Kronos weights (mini + tokenizer) | ~150MB |
| App code | ~2MB |
| **Total** | **~700MB** |

---

## 15. Phase Gate Summary

| Phase | Gate Test | Blocks |
|-------|-----------|--------|
| 1. Foundation | `compute_pnl` math, `ModelRegistry`, `get_next_trading_day` | Everything |
| 2. DuckDB Schema | Round-trip store → retrieve → evaluate → accuracy | Phases 8-9 |
| 3. Sector + News | `get_sector_etf("LNT")=="XLU"`, AV historical fetch, cache hit | Phases 4-5 |
| 4. Feature Builder | 18 features present, raw/precomputed parity, zero-fill | Phase 5 |
| 5. XGBoost | Cold < 15s, warm < 2s, valid PredictionResult, deterministic | Phase 8 |
| 6. Kronos | Valid PredictionResult or graceful KRONOS_AVAILABLE=False | Phase 8 |
| 7. LLM Agent | Parse resilience: 6 test cases all produce valid results | Phase 8 |
| 8. Prediction Service | Per-model isolation (Kronos failure doesn't kill XGBoost/LLM) | Phase 9 |
| 9. UI + Callbacks | End-to-end: add AAPL → signals appear → persist to DuckDB | Phase 10 |
| 10. Docker | Build succeeds, `HF_HUB_OFFLINE=1`, app starts in container | Deploy |

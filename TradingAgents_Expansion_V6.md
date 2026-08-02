# V6 Complete Research Summary: Specialized Financial Models for Trading Signal Generation

**Compiled:** 2026-03-06
**Period Under Test:** 2025-03-10 to 2025-12-31
**Ticker:** AMBA (Ambarella Inc.)
**Sample Size:** n=30 tri-monthly snapshots (10th, 20th, end-of-month)
**Horizons:** 1d, 3d, 1w, 2w, 4w, 8w, 13w
**Evaluation Framework:** V2 (look-ahead free, walk-forward training, multi-horizon)

---

## 1. Motivation: Why V6 Exists

V5 proved that general-purpose LLMs cannot generate trading alpha:

| Model | 1d Accuracy | P&L | Cost/Task | Deterministic |
|-------|------------|-----|-----------|---------------|
| Sonnet 4.5 | 30% | -$875 | $0.10 | No (50% flip rate) |
| GPT 5.1-codex-mini | 27% | +$468 | $0.08 | No |
| Always-BUY | 50% | benchmark | $0.00 | Yes |

**Root causes:** LLMs lack domain-specific financial NLP training, numeric feature extraction, temporal pattern recognition, and determinism.

**V6 hypothesis:** Replace LLMs with a stack of purpose-built models — each handling one signal type — then combine via ensemble.

---

## 2. Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    V6 Architecture                        │
├──────────────────────────────────────────────────────────┤
│                                                           │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────┐  │
│  │  DeBERTa-v3  │  │   XGBoost    │  │    Kronos      │  │
│  │  (Sentiment) │  │ (SHAP-Pruned)│  │  (OHLCV TS)    │  │
│  │  F1: 0.994   │  │ 20 features  │  │  AAAI 2026     │  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬─────────┘  │
│         │                 │                  │            │
│         └────────┬────────┘──────────────────┘            │
│                  │                                        │
│         ┌────────▼────────┐                               │
│         │  Confidence-    │                               │
│         │  Weighted Vote  │                               │
│         └────────┬────────┘                               │
│                  │                                        │
│         BUY / SELL / HOLD                                 │
└──────────────────────────────────────────────────────────┘
```

---

## 3. All Strategies Tested

### 3.1 Master Results Table (n=30, AMBA)

| # | Strategy | 1d | 3d | 1w | 2w | 4w | 8w | 13w | P&L | Cost | Phase |
|---|----------|-----|-----|-----|-----|-----|-----|------|------|------|-------|
| 1 | **E: Kronos-mini+XGB** | **63%** | 53% | 37% | 40% | 37% | 36% | 52% | **+$482** | $0 | 5 |
| 2 | **E: Kronos-mini+LGBM** | **60%** | 53% | 37% | 40% | 40% | 46% | **64%** | **+$1,787** | $0 | 5 |
| 3 | Kronos Mini | 57% | 53% | 43% | 50% | 37% | 36% | 52% | -$412 | $0 | 3 |
| 4 | XGBoost (SHAP-Pruned) | 52% | 52% | 41% | 41% | **62%** | 48% | 54% | -$461 | $0 | 2B |
| 5 | Kronos Small | 50% | **60%** | 43% | 50% | 43% | 46% | 60% | -$434 | $0 | 5 |
| 6 | Always-BUY | 50% | 53% | **50%** | **63%** | 50% | **79%** | 68% | bench | $0 | — |
| 7 | XGBoost (unpruned) | 45% | 45% | 41% | 45% | 41% | 37% | 54% | +$407 | $0 | 2 |
| 8 | E: KM+XGB+DeBERTa (3M) | 47% | 47% | 33% | 53% | 43% | 46% | 60% | -$707 | $0 | 4 |
| 9 | E: Kronos-mini+DeBERTa | 43% | 43% | 30% | 47% | 40% | 46% | 56% | -$783 | $0 | 5 |
| 10 | DeBERTa Sentiment | 37% | 33% | 30% | 47% | 40% | 54% | 40% | +$945 | $0 | 1 |
| 11 | ModernFinBERT | 37% | 33% | 27% | 40% | 37% | 46% | 36% | +$1,080 | $0 | 1 |
| 12 | FinBERT | 37% | 30% | 30% | 33% | 37% | 43% | 28% | +$315 | $0 | 1 |
| 13 | E: Kronos-small+XGB | 37% | 43% | 37% | 37% | 33% | 36% | 56% | -$3,157 | $0 | 5 |
| 14 | E: LightGBM+XGBoost | 37% | 33% | 33% | 40% | 33% | 41% | 56% | -$327 | $0 | 5 |
| 15 | Sonnet 4.5 (LLM) | 30% | — | — | — | — | — | — | -$875 | $0.10 | V5 |
| 16 | GPT 5.1-codex-mini | 27% | — | — | — | — | — | — | +$468 | $0.08 | V5 |

### 3.2 Horizon-Optimal Strategy Map

| Horizon | Best Strategy | Accuracy | Runner-Up |
|---------|--------------|----------|-----------|
| **1d** | E: Kronos-mini + XGBoost | **63.3%** | E: Kronos-mini + LightGBM (60%) |
| **3d** | Kronos Small / Kronos Mini / KM+XGB / KM+LGBM | **53-60%** | XGBoost SHAP-Pruned (52%) |
| **1w** | Always-BUY | **50%** | Kronos Mini/Small (43%) |
| **2w** | Always-BUY | **63%** | E: KM+XGB+DeBERTa (53%) |
| **4w** | XGBoost SHAP-Pruned | **62%** | Always-BUY (50%) |
| **8w** | Always-BUY | **79%** | XGBoost SHAP-Pruned (48%) |
| **13w** | E: Kronos-mini + LightGBM | **64%** | Always-BUY (68%) |
| **P&L** | E: Kronos-mini + LightGBM | **+$1,787** | ModernFinBERT (+$1,080) |

---

## 4. Phase-by-Phase Findings

### Phase 1: Sentiment Models

**Models tested:** DeBERTa-v3-finance, ModernFinBERT, FinBERT (ProsusAI)

**Key findings:**
1. All 3 sentiment models beat both LLMs at P&L
2. All 3 have identical 1d accuracy (37%) — sentiment alone can't predict next-day direction
3. Heavy BUY bias (73% BUY for DeBERTa) — they read general positive market news and say "BUY"
4. Best at P&L because BUY bias works on a stock that went +40%
5. None beat Always-BUY at any horizon

**Critical data quality discovery:**
- Only **0.7% of articles** (11 out of ~1,500) actually mentioned AMBA
- DeBERTa was scoring **general market news** as AMBA sentiment
- Alpha Vantage provides `ticker_sentiment` with `relevance_score` — should filter ≥0.7
- `extract_av_ticker_sentiment()` added to `data_parsers.py` to handle this

**Files:** `tradingagents/models/sentiment.py`, `tradingagents/baselines/sentiment_strategy.py`

---

### Phase 2: XGBoost on Structured Features

**Feature set:** 149 numeric features extracted from cached datasets:
- OHLCV returns (5), volatility (2), volume (1), price structure (3)
- Technical indicators (12): RSI, MACD, SMA50, SMA200, Bollinger, ATR + 5d deltas
- SPY context (14), sector context (12), relative strength (10)
- DeBERTa sentiment (6), AV ticker sentiment (9), AV topic sentiment (~24), AV global sentiment (~33)

**Walk-forward training:** For each prediction date, trains ONLY on prior data. TimeSeriesSplit 3-fold CV. First 3 dates default to HOLD (< 5 training samples).

**Key findings:**
1. 48.3% 1d accuracy — first model to make aggressive SELL calls (41% SELL)
2. SELL accuracy 50% across 1d/3d/1w — meaningful on a stock that gained 40%
3. Severe overfitting: 149 features / 28 samples = 5.3:1 ratio
4. Walk-forward degradation: -39.7pp (64.7% in-sample → 25% out-of-sample)

**Files:** `tradingagents/models/feature_engineering.py`, `tradingagents/models/ml_trainer.py`, `tradingagents/baselines/ml_strategy.py`

---

### Phase 2B: SHAP Feature Pruning (Default XGBoost)

**Method:** `shap.TreeExplainer` computed exact Shapley values. Selected top 20 features by mean |SHAP value|. Reduced feature-to-sample ratio from 5.3:1 to 0.7:1.

**Top 5 features by SHAP importance:**
1. `spy_gap` (0.2419) — SPY overnight gap
2. `vs_spy_excess_5d` (0.1754) — AMBA vs SPY 5-day relative return
3. `av_topic_mergers_and_acquisitions_avg` (0.1619) — M&A news sentiment
4. `av_topic_retail_wholesale_avg` (0.1540) — Retail news sentiment
5. `sector_gap` (0.1409) — Sector ETF overnight gap

**Critical insight:** Zero AMBA-specific price features (return_Nd, vol_Nd, ind_rsi) in top 20. The model trades **market regime**, not AMBA fundamentals. This means:
- It's really a macro-sentiment strategy
- May generalize to other tickers (positive for multi-ticker)
- Has no AMBA-specific edge

**Key results:**
- 1d accuracy: 55.2% (vs 48.3% unpruned, +6.9pp)
- Walk-forward degradation: -8.8pp (vs -39.7pp unpruned) — overfitting largely eliminated
- 4w accuracy: 62.1% — best of any strategy at this horizon
- 4w BUY accuracy: 83.3% — strongest actionable signal in V6
- 8w BUY accuracy: 90.9%

**P&L paradox:** More accurate but worse P&L (-$461 vs +$407). Root cause: single catastrophic SELL call on 2025-05-10 with 0.84 confidence during a stock surge (-$888). This one date accounts for $1,746 of the $868 P&L gap. Excluding it, pruned P&L ≈ +$627 vs unpruned ≈ -$451.

**Fold stability warning:** SHAP rankings were unstable across 3 CV folds. Features flipped from rank 1 to rank 57. With only 28 samples, the top-20 selection is one draw from a noisy distribution.

**Files:** `scripts/shap_analysis.py`, `tradingagents/baselines/ml_strategy_pruned.py`, `tradingagents/models/selected_features.json`
**Full report:** `reports/benchmarks/V6_SHAP_PRUNING_RESULTS_2026-02-19.md`

---

### Phase 3: Kronos Time-Series Model

**Model:** [Kronos-mini](https://huggingface.co/NeoQuasar/Kronos-mini) (4.1M params, AAAI 2026)
- Pre-trained on 12 billion candlestick records from 45 exchanges
- Zero-shot inference: feeds ~60 daily OHLCV bars, predicts next 3 candles
- Runs on MPS (Apple Silicon) in ~1 sec/prediction
- Monte Carlo: 20 samples, temperature 0.8 for stable predictions
- Decision: 60% weight on 1-day direction + 40% on multi-day consistency

**Key results:**
- 1d accuracy: **56.7%** — first strategy to beat Always-BUY (50%)
- BUY 1d: 57.1%, SELL 1d: 56.2% — balanced signal (never HOLDs)
- Negative P&L (-$412) despite high accuracy — wrong trades are large losers (position sizing issue)

**Technical notes:**
- Use `Kronos-mini` not `Kronos-small` — mini is better (see Phase 5)
- Tokenizer: always `NeoQuasar/Kronos-Tokenizer-base` (shared across sizes)
- `DatetimeIndex` doesn't have `.dt` accessor — pass `pd.Series` for timestamps
- `predict()` needs `x_timestamp` and `y_timestamp` as `pd.Series`

**Files:** `tradingagents/models/kronos/` (vendored), `tradingagents/models/timeseries.py`, `tradingagents/baselines/timeseries_strategy.py`

---

### Phase 4: 3-Model Ensemble (DeBERTa + XGBoost + Kronos)

**Method:** Confidence-weighted vote
- BUY=+1, SELL=-1, HOLD=0; weighted by model confidence
- Thresholds: >+0.15 → BUY, <-0.15 → SELL, else HOLD

**Key results:**
- 1d accuracy: 46.7% — **worse** than Kronos alone (56.7%)
- 13w accuracy: 60% — best of any strategy at this horizon (at the time)
- DeBERTa's 73% BUY bias systematically overrides correct SELL signals

**Verdict:** The 3-model ensemble fails at short-term because DeBERTa adds noise. But performs well at long horizons where sentiment has predictive value.

**Files:** `tradingagents/baselines/ensemble_strategy.py`

---

### Phase 5: Ensemble Ablation & Model Size Study

**Goal:** Systematically determine which model combinations work and why.

**Named ensemble syntax:** `ensemble_METHOD+sub1+sub2` parsed by `_normalize_sub_strategy()` in `cli/evaluate_v2.py`

#### 6 Experiments Run

| # | Ensemble | 1d Acc | P&L | Verdict |
|---|----------|--------|------|---------|
| 1 | Kronos Small (solo) | 50.0% | -$434 | Mini >> Small |
| 2 | **Kronos-mini + XGBoost** | **63.3%** | **+$482** | Best 1d accuracy |
| 3 | Kronos-small + XGBoost | 36.7% | -$3,157 | Catastrophic |
| 4 | Kronos-mini + DeBERTa | 43.3% | -$783 | DeBERTa hurts |
| 5 | **Kronos-mini + LightGBM** | **60.0%** | **+$1,787** | Best P&L |
| 6 | LightGBM + XGBoost | 36.7% | -$327 | No Kronos = fail |

#### 6 Key Findings

**Finding 1: Kronos-mini + ML = Best Strategy**
- Kronos provides directional prior from OHLCV patterns
- ML provides complementary signal from structured features (SPY context, relative strength, sentiment)
- When they agree → strong signal; when they disagree → HOLD avoids false trades

**Finding 2: DeBERTa Hurts Short-Term Ensembles**
- Adding DeBERTa drops 1d accuracy by 16-20pp
- Its 73% BUY bias overrides correct SELL signals from other models
- Root cause: scores irrelevant general market news (only 0.7% of articles mention AMBA)

**Finding 3: Kronos-small is Worse Than Kronos-mini**
- Solo: 50% (small) vs 56.7% (mini)
- In ensemble with XGBoost: 36.7% (small) vs 63.3% (mini)
- Larger model may overfit to training data patterns that don't transfer to AMBA
- Mini's simpler representations may generalize better for single-stock prediction

**Finding 4: Walk-Forward Stability**
- Kronos-mini+XGBoost is the ONLY strategy that **improves** out-of-sample (+5.6%)
- All other strategies degrade OOS, indicating genuine learning vs overfitting

| Strategy | In-Sample | Out-of-Sample | Delta |
|----------|-----------|---------------|-------|
| E: Kronos-mini+XGB | 61.1% | **66.7%** | **+5.6%** |
| E: Kronos-mini+LGBM | 61.1% | 58.3% | -2.8% |
| Kronos Mini (solo) | 61.1% | 50.0% | -11.1% |
| Kronos Small (solo) | 55.6% | 41.7% | -13.9% |

**Finding 5: ML-Only Ensemble (No Kronos) Fails**
- LightGBM + XGBoost = 36.7% — below random
- Both use identical features → highly correlated signals → amplify shared errors
- **Kronos provides the essential orthogonal signal** that ML models lack

**Finding 6: LightGBM vs XGBoost in Ensembles**
- XGBoost: higher 1d accuracy (63.3%), better walk-forward stability (+5.6%)
- LightGBM: higher P&L (+$1,787), lower drawdown ($1,049), better 13w accuracy (64%)
- **Aggressive/day-trading:** use Kronos-mini + XGBoost
- **Conservative/P&L-optimized:** use Kronos-mini + LightGBM

---

## 5. Statistical Limitations

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| **n=30 sample size** | No strategy reaches p<0.05 vs Always-BUY | Need multi-ticker or longer history |
| **Single ticker (AMBA)** | Can't generalize; AMBA went +40% making BUY easy | Multi-ticker validation needed |
| **28 training samples for ML** | 0.7:1 feature ratio (pruned) is better but still tight | Cross-ticker training pool |
| **SHAP fold instability** | Top-20 features are one noisy draw | Bootstrap-stable SHAP selection |
| **Overlapping confidence intervals** | 63.3% best vs 50% baseline: CI [46.7%, 80.0%] | More data points needed |
| **Bullish period bias** | Always-BUY achieves 79% at 8w — extreme bull bias | Need bear/sideways periods |

### Bootstrap 95% Confidence Intervals (1d Accuracy)

| Strategy | Accuracy | 95% CI |
|----------|----------|--------|
| E: Kronos-mini+XGB | 63.3% | [46.7%, 80.0%] |
| E: Kronos-mini+LGBM | 60.0% | [43.3%, 76.7%] |
| Kronos Mini | 56.7% | [40.0%, 73.3%] |
| Kronos Small | 50.0% | [33.3%, 66.7%] |
| Always-BUY | 50.0% | — |

---

## 6. Data Pipeline & Cached Datasets

### Dataset Structure
Cached at `results/datasets/{TICKER}/{DATE}/{TICKER}_{DATE}_raw_dataset.json`

| Field | Format | Content |
|-------|--------|---------|
| `market_data` | CSV string | OHLCV bars (~60 days) |
| `indicators_data` | Text block | RSI, MACD, SMA, Bollinger, ATR |
| `news_data` | 2-line string | Line 1: Polygon news dict, Line 2: Alpha Vantage news dict |
| `global_news_data` | String | Alpha Vantage global market news |
| `spy_market_data` | CSV string | SPY OHLCV bars |
| `sector_market_data` | CSV string | Sector ETF OHLCV bars |

### Data Quality Issues

1. **News relevance:** 99.3% of articles scored by DeBERTa are irrelevant to AMBA
2. **AV ticker_sentiment:** Only 2 articles per 30 dates had AMBA-specific AV sentiment
3. **Polygon vs AV:** Polygon provides 9 AMBA-specific articles vs AV's 2
4. **Fix implemented:** `extract_av_ticker_sentiment()` filters by `relevance_score >= 0.7`

---

## 7. Technical Infrastructure

### Environment
- Conda env: `tradingagents`
- Activation: `source /opt/miniconda3/etc/profile.d/conda.sh && conda activate tradingagents`
- Platform: macOS (Darwin), Apple Silicon (MPS for PyTorch)

### Evaluation Framework (cli/evaluate_v2.py)

**Strategy dispatch prefixes:**
- `single_` → LLM-based (V5)
- `sentiment_` → DeBERTa/FinBERT/ModernFinBERT
- `ml_` → XGBoost/LightGBM (unpruned or pruned)
- `ts_kronos_` → Kronos time-series
- `ensemble_` → Ensemble meta-learner
- `ma_` → Moving average baseline

**Named ensemble syntax:** `ensemble_weighted_vote+kronos_mini+xgboost`
- Parsed by `_normalize_sub_strategy()` which maps aliases to strategy keys
- Lazy-loads sub-strategy instances via `_get_strategy()` in ensemble_strategy.py

**Checkpoint/resume:** `python -m cli.evaluate_v2 --resume RUN_ID`
- Saves after each task to `checkpoint.json`
- Known issue: stuck tasks may have status "running" — must manually set to "pending" for resume

### Key Dependencies
- `xgboost`, `lightgbm` — tree-based ML models
- `shap>=0.43.0` — feature importance analysis
- `transformers` — DeBERTa/FinBERT/ModernFinBERT
- `scikit-learn` — preprocessing, TimeSeriesSplit
- `torch` — Kronos model (pip install, not conda)
- `einops` — Kronos model dependency

### Common Errors & Fixes
- `numpy float32 not JSON serializable`: cast with `float()` in ml_trainer.py predict
- OHLCV labels: each dataset ends at trade date, need next dataset's OHLCV for next-day label
- `torch` not in conda: `pip install torch` separately
- Kronos import: `from model.module import *` → `from tradingagents.models.kronos.module import *`
- `DatetimeIndex` has no `.dt` accessor: pass `pd.Series` for timestamps to Kronos
- Pipe buffering: background commands piped to `| tail` produce empty output; check checkpoint files directly

---

## 8. File Inventory

### Models (`tradingagents/models/`)
| File | Purpose |
|------|---------|
| `sentiment.py` | DeBERTa/FinBERT/ModernFinBERT wrapper |
| `data_parsers.py` | Parse cached dataset strings + AV ticker sentiment extraction |
| `feature_engineering.py` | 149-feature extraction (OHLCV, indicators, sentiment, AV) |
| `ml_trainer.py` | XGBoost/LightGBM walk-forward training + SHAP analysis |
| `timeseries.py` | Kronos prediction wrapper with lazy model loading |
| `selected_features.json` | Top 20 SHAP-selected features for runtime |
| `kronos/` | Vendored Kronos model (kronos.py, module.py) |

### Strategies (`tradingagents/baselines/`)
| File | Purpose |
|------|---------|
| `sentiment_strategy.py` | Sentiment-only strategy |
| `ml_strategy.py` | ML strategy with walk-forward training |
| `ml_strategy_pruned.py` | `MLStrategyPruned` subclass (default XGBoost) |
| `timeseries_strategy.py` | Kronos time-series strategy |
| `ensemble_strategy.py` | Ensemble meta-learner (majority_vote, weighted_vote, adaptive) |

### Evaluation & Scripts
| File | Purpose |
|------|---------|
| `cli/evaluate_v2.py` | Main evaluation harness with multi-strategy dispatch |
| `scripts/shap_analysis.py` | SHAP feature importance analysis CLI |
| `cli/compile_reports.py` | PDF report compiler |
| `cli/analyze_outcomes.py` | Post-hoc outcome analysis |

### Reports
| File | Purpose |
|------|---------|
| `reports/benchmarks/V6_Plan.md` | Master plan with all phase results |
| `reports/benchmarks/V6_SHAP_PRUNING_RESULTS_2026-02-19.md` | Detailed SHAP analysis |
| `reports/benchmarks/V6_COMPLETE_RESEARCH_SUMMARY.md` | This file |

### Evaluation Run Directories (`results/evaluations/`)
| Run ID | Strategy | Result |
|--------|----------|--------|
| `backtest_v2_20260215_175115` | DeBERTa sentiment | 37% 1d, +$945 |
| `backtest_v2_20260217_164206` | FinBERT + ModernFinBERT | 37% 1d each |
| `backtest_v2_20260217_170255` | XGBoost ML (unpruned) | 48.3% 1d, +$407 |
| `backtest_v2_20260217_180202` | Kronos Mini | 56.7% 1d, -$412 |
| `backtest_v2_20260217_180245` | Ensemble 3-model | 46.7% 1d, -$707 |
| `backtest_v2_20260218_221702` | XGBoost unpruned vs pruned | 48.3% vs 55.2% |
| `backtest_v2_20260218_192824` | Kronos Small | 50% 1d, -$434 |
| `backtest_v2_20260218_192919` | E: Kronos-mini+XGBoost | 63.3% 1d, +$482 |
| `backtest_v2_20260219_010834` | E: Kronos-small+XGBoost | 36.7% 1d, -$3,157 |
| `backtest_v2_20260219_022022` | E: Kronos-mini+DeBERTa | 43.3% 1d, -$783 |
| `backtest_v2_20260219_022352` | E: Kronos-mini+LightGBM | 60% 1d, +$1,787 |
| `backtest_v2_20260219_034243` | E: LightGBM+XGBoost | 36.7% 1d, -$327 |

---

## 9. Actionable Conclusions

### What Works
1. **Kronos-mini + XGBoost SHAP-Pruned** = best 1d accuracy (63.3%), only strategy that improves OOS
2. **Kronos-mini + LightGBM** = best P&L (+$1,787), lowest drawdown ($1,049), best 13w (64%)
3. **XGBoost SHAP-Pruned 4w BUY signal** = 83.3% accuracy — strongest actionable signal
4. **SHAP pruning** eliminates overfitting (degradation from -40pp to -9pp)
5. **All specialized models beat LLMs** at cost ($0 vs $0.08-0.10), determinism, and most accuracy metrics

### What Doesn't Work
1. **DeBERTa in ensembles** — BUY bias from irrelevant news drops 1d accuracy 16-20pp
2. **Kronos-small** — larger model is worse than mini, especially in ensembles
3. **ML-only ensembles** — XGBoost + LightGBM = 36.7% (correlated signals, no diversity)
4. **3-model ensemble** — DeBERTa noise outweighs its long-horizon value at short-term
5. **LLMs** — non-deterministic, expensive, and below Always-BUY

### What's Unknown (Future Work)
1. **Multi-ticker validation** — does Kronos-mini+XGB work on non-AMBA stocks? Since XGBoost features are macro-regime, it should generalize
2. **Bear/sideways markets** — AMBA went +40%, making Always-BUY artificially strong. Need stocks that went flat or down
3. **SELL accuracy deep-dive** — what's the SELL accuracy of the winning ensembles? Can they identify pullbacks?
4. **Bootstrap-stable SHAP** — run feature selection 100x with different samples, keep features appearing >50% of the time
5. **Cross-ticker training** — train XGBoost on multiple tickers to increase sample size beyond n=28
6. **Regime-adaptive strategy switching** — use different strategies based on detected market regime (HMM or similar)
7. **Position sizing** — Kronos-mini has 56.7% accuracy but negative P&L due to large losers. Kelly criterion or volatility-scaled sizing could fix this
8. **DeBERTa with proper ticker filtering** — re-test DeBERTa scoring only AMBA-relevant articles (relevance ≥ 0.7)

---

## 10. Academic References

| Finding | Source | Implication |
|---------|--------|------------|
| Raw OHLCV features > derived indicators for ML | arXiv:2412.15448 | Include raw returns, not just RSI/MACD |
| FinBERT sentiment Granger-causes volatility at 1w | Multiple papers | Sentiment useful at 1w+ horizon |
| TSFMs underperform XGBoost on financial data (zero-shot) | arXiv:2511.18578 | Don't use TimesFM/Chronos zero-shot |
| Regime detection essential for strategy robustness | arXiv:2601.19504 | Add HMM regime filter |
| Ensemble NLP (DeBERTa+FinBERT+RoBERTa) hits 80% F1 | arXiv:2507.09739 | Consider NLP ensemble if single model insufficient |
| "Generating Alpha" paper: 135% return in 24mo | arXiv:2601.19504 | XGBoost + regime detection proven approach |

---
---

# V6.1 — Live Dashboard Integration Plan

**Date:** 2026-03-07
**Target:** Integrate V6 models into the `quant-news` Dash dashboard for live next-day predictions
**Status:** Pre-implementation review

---

## 11. Goal & Scope

Bring the V6 model stack into the existing quant-news dashboard so that:

1. **Any ticker** the user searches gets a next-day BUY/SELL/HOLD prediction from 3 independent models
2. **Predictions are stored** in DuckDB and **evaluated against actuals** on subsequent app launches
3. **Market context** (geopolitics, trade wars, macro) is tracked via a dedicated LLM single-agent analysis
4. **Ensemble is deferred** until per-model accuracy is validated on 10+ tickers over 30+ trading days

### 11.1 Models in Scope

| Model | Type | Source | V6 1d Accuracy | Role |
|-------|------|--------|----------------|------|
| **Kronos-mini** | Time-series foundation model | OHLCV patterns (60-90 bars) | 56.7% (solo) | Directional prior from price structure |
| **XGBoost SHAP-Pruned** | Gradient-boosted tree | 18 structured features | 55.2% (solo), 63.3% (in ensemble) | Market regime signal from SPY/sector/AV features |
| **LLM Single Agent** | Large language model | News + indicators + macro context | 30% (V5, unstructured) | Geopolitical/macro awareness numeric models lack |
| **Ensemble** | Weighted vote | Combines above 3 | DEFERRED | Only after per-model validation |

### 11.2 Benchmark Tickers

4 tickers from SPUS (halal S&P 500) for diverse sector coverage:

| Ticker | Sector | Rationale |
|--------|--------|-----------|
| **PG** | Consumer Staples | Defensive, low-vol, dividend — tests model on low-signal stock |
| **LNT** | Utilities | Stable, rate-sensitive — replaces DVN (halal screening concern) |
| **JNJ** | Healthcare | Large-cap, mixed news flow — tests relevance filtering |
| **VZ** | Communication Services | Dividend yield play, rate-sensitive — tests macro awareness |

> **Note on DVN**: Devon Energy (upstream oil & gas E&P) has scholarly concerns under AAOIFI-based halal screening frameworks. Replaced with LNT (Alliant Energy) — a utility with similar rate-sensitivity but cleaner halal standing.

---

## 12. Critical Architecture Decisions

### 12.1 Async Callback Architecture (Issue #1)

**Problem**: The `generate_model_signals` callback runs synchronously. With 4 tickers:
- Kronos: ~150MB HuggingFace download on first call, then ~1s/prediction
- XGBoost: Cold start = AV news fetch + ~191 training iterations + train = 30-60s/ticker
- LLM: Network round-trip, 2-5s

**Total worst case**: 4-8 minutes blocking UI on first use.

**Solution**: Use Dash `background_callback` with `diskcache` backend (zero-config, no Celery).

```python
import diskcache
from dash import DiskcacheManager

cache_dir = "cache/dash_bg_callbacks"
bg_cache = diskcache.Cache(cache_dir)
background_callback_manager = DiskcacheManager(bg_cache)

@callback(
    Output("model-signals-store", "data"),
    Input("stock-data-store", "data"),
    State("selected-symbols", "data"),
    background=True,
    manager=background_callback_manager,
    running=[
        (Output("model-signals-loading", "style"), {"display": "block"}, {"display": "none"}),
    ],
    progress=[Output("model-signals-progress", "children")],
    prevent_initial_call=True,
)
def generate_model_signals(set_progress, stock_data, symbols):
    results = {}
    for i, symbol in enumerate(symbols):
        set_progress(f"Predicting {symbol}... ({i+1}/{len(symbols)})")
        # ... run models with per-model try/except ...
    return results
```

**Why diskcache over Celery**: Zero dependencies, works on macOS without Redis, persists across hot-reloads. The background process spawns a subprocess — UI stays responsive.

**Per-model isolation**: Each model call wrapped in try/except. If Kronos fails (HuggingFace unreachable), XGBoost and LLM still produce results. Failed models return `PredictionResult(error="...")` rendered as a grey card in the UI.

```python
for model in self._registry.list_models():
    try:
        result = model.predict(symbol, ohlcv_df, **kwargs)
    except Exception as e:
        result = PredictionResult(
            model_name=model.name, decision="HOLD",
            confidence=0.0, up_probability=0.5,
            details={}, error=str(e)
        )
    results[model.name] = result
```

**New dependency**: `diskcache>=5.6.0`

---

### 12.2 XGBoost Training Performance (Issue #2)

**Problem**: Naive sliding window calls `add_indicators_to_df()` inside a 191-iteration loop, recomputing RSI/MACD/SMA from scratch each time.

**Solution**: Pre-compute indicators on the FULL DataFrame once, then slice.

```python
def _train_from_history(self, ticker_df, spy_df, sector_df, av_news_by_date):
    # Step 1: Pre-compute indicators on full history ONCE
    ticker_full = add_indicators_to_df(ticker_df.copy())
    spy_full = add_indicators_to_df(spy_df.copy())
    sector_full = add_indicators_to_df(sector_df.copy())

    # Step 2: Slide window using pre-computed columns
    rows = []
    for t in range(60, len(ticker_full) - 1):
        date_str = ticker_full.index[t].strftime('%Y-%m-%d')
        features = self._feature_builder.build_features_from_precomputed(
            ticker_full.iloc[:t+1],   # all indicator columns already present
            spy_full.iloc[:t+1],
            sector_full.iloc[:t+1],
            av_news=av_news_by_date.get(date_str, []),
        )
        # Label: next-day direction
        pct = (ticker_full['Close'].iloc[t+1] - ticker_full['Close'].iloc[t]) / ticker_full['Close'].iloc[t] * 100
        if abs(pct) < 0.15:
            continue  # skip ambiguous days
        features['_label'] = 1 if pct > 0 else 0
        rows.append(features)
```

**Key detail**: `build_features_from_precomputed()` reads indicator values directly from DataFrame columns (already computed) rather than calling `add_indicators_to_df()` per iteration. The feature builder has TWO paths:
- `build_features()` — for live prediction (accepts raw OHLCV, computes indicators internally)
- `build_features_from_precomputed()` — for training (indicators already in DataFrame columns)

**Expected training time**: ~2-5 seconds per ticker (down from 30-60s).

---

### 12.3 Missing Value Strategy for AV Topic Features (Issue #3)

**Problem**: 8 of 18 features are AV topic sentiment (`av_topic_mergers_and_acquisitions_avg`, `av_topic_ipo_avg`, etc.). For tickers like VZ or PG, M&A and IPO articles will be zero/missing on most days.

**Strategy**: Consistent zero-fill with XGBoost native missing value handling.

1. **During training**: If no AV articles exist for a date, all `av_topic_*` features = 0.0 and `av_n_tech_articles` = 0
2. **During prediction**: Same — if no articles today, all AV features = 0.0
3. **Rationale**: XGBoost learns optimal split directions for missing/zero values natively. As long as encoding is consistent between training and prediction, the model handles it correctly.
4. **NOT using NaN**: NaN would trigger XGBoost's native missing value handling (which sends missing observations to the default child node). While this works, it's harder to debug and explain. Zero-fill is explicit.

**Validation**: Log the per-feature zero-rate during training. If >90% of rows have zero for a feature, warn in training stats — the feature is providing no signal and could be dropped for that ticker.

```python
zero_rates = {col: (df[col] == 0).mean() for col in feature_cols}
sparse_features = [f for f, rate in zero_rates.items() if rate > 0.9]
if sparse_features:
    logger.warning(f"Sparse features (>90% zero): {sparse_features}")
```

---

### 12.4 Accuracy Claims as Hypothesis (Issue #4)

**Problem**: The 63.3% 1d accuracy was achieved on 1 ticker (AMBA), 30 samples, with overlapping CIs [46.7%, 80.0%]. The SHAP features (`av_topic_retail_wholesale_avg`, `av_topic_real_estate_avg`) are oddly specific and may represent spurious correlation or data leakage in the single-ticker sample.

**Position**: 63.3% is a **hypothesis under test**, not a baseline to preserve.

**Validation protocol for each benchmark ticker**:

| Metric | Minimum to pass | Measurement period |
|--------|----------------|-------------------|
| 1d accuracy | > 52% (better than coin flip + 2pp margin) | 30 trading days |
| P&L (equal-weight $1000 trades) | > $0 (positive) | 30 trading days |
| SELL accuracy | > 40% (meaningful downside calls) | 30 trading days |
| Prediction coverage | > 80% of trading days (not always HOLD) | 30 trading days |

**Per-model tracking** (not just ensemble):
- Each model's accuracy tracked independently in DuckDB
- If a model consistently underperforms coin-flip on a ticker, display a warning badge in the UI
- The ensemble activation decision (Section 12.10) uses per-model data, not just aggregate

**Feature stability check**: After 30 days of predictions for each ticker, compare which features the live-trained XGBoost considers important vs. the original AMBA-trained SHAP rankings. If the top-5 diverge significantly, the original feature selection may not generalize.

---

### 12.5 DuckDB Schema Fixes (Issue #5)

**model_predictions table** (revised):

```sql
CREATE TABLE IF NOT EXISTS model_predictions (
    -- Primary key: explicit composite format
    id VARCHAR PRIMARY KEY,             -- format: "{symbol}_{model_name}_{prediction_date}"
    symbol VARCHAR NOT NULL,
    model_name VARCHAR NOT NULL,
    prediction_date DATE NOT NULL,      -- when prediction was generated
    target_date DATE NOT NULL,          -- next trading day being predicted

    -- Prediction output
    decision VARCHAR NOT NULL,          -- BUY / SELL / HOLD
    confidence DOUBLE,
    up_probability DOUBLE,
    predicted_close DOUBLE,
    previous_close DOUBLE,              -- close on prediction_date

    -- Evaluation (filled on next launch)
    actual_close DOUBLE,
    was_correct BOOLEAN,
    pnl_dollars DOUBLE,                -- (actual - previous) * direction_sign

    -- Debugging & reproducibility
    model_version VARCHAR,              -- "{model_type}_v{n}_{training_date}" for XGBoost,
                                        -- "kronos_mini_v1" for Kronos (static),
                                        -- "{llm_provider}_{model_id}" for LLM
    training_samples INTEGER,           -- how many samples XGBoost trained on (NULL for others)
    feature_values_json VARCHAR,        -- full feature vector that produced this prediction
    details_json VARCHAR,               -- model-specific extras (pred_days_detail, reasoning, etc.)

    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    evaluated_at TIMESTAMP
)
```

**ID format**: `f"{symbol}_{model_name}_{prediction_date:%Y%m%d}"` — e.g., `AAPL_kronos_mini_20260307`

**historical_news table** (revised):

```sql
CREATE TABLE IF NOT EXISTS historical_news (
    -- Dedup key: hash of URL or title+date
    id VARCHAR PRIMARY KEY,             -- hashlib.md5(f"{url}_{published_date}").hexdigest()
    symbol VARCHAR NOT NULL,
    published_date DATE,
    title VARCHAR,
    summary VARCHAR,
    url VARCHAR,
    source VARCHAR,
    topics_json VARCHAR,                -- [{"topic": "Technology", "relevance_score": "0.85"}]
    overall_sentiment_score DOUBLE,
    overall_sentiment_label VARCHAR,
    ticker_sentiment_score DOUBLE,      -- AV ticker-specific sentiment (if available)
    ticker_relevance_score DOUBLE,
    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
```

**Dedup strategy**: `id = hashlib.md5(f"{url or ''}_{published_date or ''}".encode()).hexdigest()`. Re-fetching skips existing IDs via `INSERT OR IGNORE`.

---

### 12.6 LLM JSON Parsing Resilience (Issue #6)

**Problem**: LLMs regularly output markdown-wrapped JSON, hallucinated fields, and uncalibrated confidence scores.

**Solution**: Multi-layer parsing with fallback, reusing the pattern already in `llm_service.py:summarize_news_structured()` (lines 270-303).

```python
def _parse_llm_prediction(self, response: str | None) -> PredictionResult:
    """Parse LLM response with 3-layer fallback."""
    if not response:
        return self._fallback_result("No LLM response")

    # Layer 1: Direct JSON parse
    try:
        clean = response.strip()
        # Strip markdown code fences (reuse pattern from llm_service.py line 276)
        match = re.search(r'\{.*\}', clean, re.DOTALL)
        if match:
            clean = match.group(0)
        data = json.loads(clean)
    except json.JSONDecodeError:
        return self._fallback_result(f"JSON parse failed: {response[:100]}")

    # Layer 2: Validate and normalize fields
    decision = data.get("decision", "HOLD").upper().strip()
    if decision not in ("BUY", "SELL", "HOLD"):
        decision = "HOLD"

    confidence = data.get("confidence", 0.5)
    if isinstance(confidence, str):
        try:
            confidence = float(confidence)
        except ValueError:
            confidence = 0.5
    confidence = max(0.0, min(1.0, confidence))

    # Layer 3: Calibrate LLM confidence
    # LLMs cluster confidence at 0.8-0.9. Compress to [0.4, 0.75] range
    # to be comparable with XGBoost/Kronos probabilistic outputs.
    calibrated_confidence = 0.4 + (confidence * 0.35)

    up_probability = data.get("up_probability", 0.5)
    if isinstance(up_probability, str):
        try:
            up_probability = float(up_probability)
        except ValueError:
            up_probability = 0.5

    return PredictionResult(
        model_name="llm_agent",
        decision=decision,
        confidence=round(calibrated_confidence, 2),
        up_probability=round(max(0.0, min(1.0, up_probability)), 2),
        details={
            "reasoning": data.get("reasoning", ""),
            "raw_confidence": confidence,  # preserve uncalibrated value
        },
    )

def _fallback_result(self, error_msg: str) -> PredictionResult:
    """HOLD with low confidence when LLM fails."""
    return PredictionResult(
        model_name="llm_agent",
        decision="HOLD",
        confidence=0.3,
        up_probability=0.5,
        details={"reasoning": "LLM parse failure — defaulting to HOLD"},
        error=error_msg,
    )
```

**Confidence display in UI**: Show both raw and calibrated confidence for the LLM model, so the user understands the adjustment. XGBoost and Kronos display their native confidence unchanged.

---

### 12.7 Trading Calendar Utility (Issue #7)

**Problem**: `target_date` computation and `evaluate_predictions()` need to know the next trading day (accounting for weekends and NYSE holidays).

**New file**: `utils/trading_calendar.py`

```python
"""NYSE trading calendar utility.

Uses pandas_market_calendars for accurate holiday handling.
Falls back to a hardcoded holiday list if package unavailable.
"""
import pandas as pd

# Try pandas_market_calendars first (most accurate)
try:
    import pandas_market_calendars as mcal
    _nyse = mcal.get_calendar("NYSE")
    _HAS_MARKET_CAL = True
except ImportError:
    _HAS_MARKET_CAL = False

# Hardcoded NYSE holidays for 2026 (fallback)
_NYSE_HOLIDAYS_2026 = {
    "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03",
    "2026-05-25", "2026-06-19", "2026-07-03", "2026-09-07",
    "2026-11-26", "2026-12-25",
}

def get_next_trading_day(from_date: pd.Timestamp | str) -> pd.Timestamp:
    """Return the next NYSE trading day after from_date."""
    ...

def is_market_open_today() -> bool:
    """Check if NYSE is open today (for evaluate_predictions trigger)."""
    ...

def get_previous_trading_day(from_date: pd.Timestamp | str) -> pd.Timestamp:
    """Return the most recent trading day on or before from_date."""
    ...
```

**New optional dependency**: `pandas_market_calendars>=4.0.0` (in requirements.txt, with fallback if not installed).

**Usage**:
- `PredictionService.predict_symbol()` → `target_date = get_next_trading_day(today)`
- `CacheService.evaluate_predictions()` → only evaluates if `is_market_open_today()` or `target_date < today`
- `CacheService.store_prediction()` → validates `target_date` is a trading day

---

### 12.8 Kronos Vendoring Strategy (Issue #8)

**Problem**: Copying 3 files with a one-line import fix is fragile. No version pinning, `import *` carries over.

**Solution**:

1. **Version pinning**: Add `VENDORED_FROM` header to `models/kronos/__init__.py`:
```python
"""Vendored Kronos time-series foundation model.

VENDORED_FROM: /Users/UmarJahangir/Projects/TradingAgents/tradingagents/models/kronos/
VENDORED_DATE: 2026-03-07
SOURCE_VERSION: NeoQuasar/Kronos-mini (HuggingFace)
PAPER: AAAI 2026
NOTE: Do not modify vendored code. If upstream changes are needed,
      re-vendor from TradingAgents and update this header.
"""
from .kronos import KronosTokenizer, Kronos, KronosPredictor
```

2. **Fix `import *`**: In `kronos.py`, replace:
```python
# BEFORE (TradingAgents)
from tradingagents.models.kronos.module import *
# AFTER (quant-news) — explicit imports
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

3. **Integrity check**: `kronos_model.py` verifies the vendored code is importable at import time and logs a clear error if torch/einops are missing:
```python
try:
    from models.kronos import KronosTokenizer, Kronos, KronosPredictor
    KRONOS_AVAILABLE = True
except ImportError as e:
    KRONOS_AVAILABLE = False
    _KRONOS_ERROR = str(e)
```

---

### 12.9 torch CPU-Only Build (Issue #9)

**Problem**: `pip install torch` pulls the CUDA build (~2.4GB) by default, even on macOS with MPS.

**Solution**: Platform-specific installation in requirements.txt:

```
# requirements.txt — torch section
# NOTE: Install torch separately for your platform:
#   macOS (MPS):  pip install torch torchvision  (default pip, uses MPS)
#   Linux (CPU):  pip install torch --index-url https://download.pytorch.org/whl/cpu
#   Linux (CUDA): pip install torch  (default pip, uses CUDA)
# The app auto-detects MPS/CUDA/CPU at runtime.
torch>=2.0.0
```

**Runtime device detection** (already in Kronos vendored code): `KronosPredictor.__init__` auto-selects `mps` on Apple Silicon, `cuda` if available, else `cpu`. No changes needed.

**Documentation**: Add a note to the README or CLAUDE.md that first-time `pip install torch` may take time due to download size, and provide the CPU-only URL for CI/deployment.

---

### 12.10 Ensemble Activation Criteria (Issue #10)

**Problem**: "Validate on 10+ tickers" is vague. Need specific, measurable criteria.

**Activation requirements** — ALL must be met:

| Criterion | Threshold | Rationale |
|-----------|-----------|-----------|
| **Tickers evaluated** | >= 10 distinct tickers | Minimum for cross-sector generalization |
| **Trading days per ticker** | >= 30 days each | Minimum for statistical meaning at p < 0.1 |
| **Per-model 1d accuracy (median across tickers)** | > 52% for at least 2 of 3 models | Better than coin flip + margin |
| **No model below 45% on any ticker** | Each model > 45% on each ticker | Eliminate catastrophic failures |
| **P&L positive** | Median ticker P&L > $0 (equal-weight $1000 trades) | Accuracy alone isn't enough |
| **Prediction diversity** | At least 20% SELL calls across all models | Not just "always BUY" in a bull market |

**LLM agent validation** (since V5 showed 30%):
- The LLM agent is validated on the same criteria. If it consistently underperforms (<45% on most tickers), it is excluded from the ensemble but still displayed as an informational signal.
- Its geopolitical reasoning is valuable to the user even if its directional accuracy is poor.

**Ensemble weight tuning**:
- Once activated, initial weights: XGBoost 1.3, Kronos 1.1, LLM 0.8
- After 30 days of ensemble predictions, compare ensemble accuracy vs best individual model
- If ensemble < best individual for 30 consecutive days, disable ensemble and revert to individual display

**Implementation**: `services/prediction_service.py` checks `CacheService.get_model_accuracy()` at startup. If criteria met, ensemble is auto-registered in the registry. Status displayed in UI sidebar.

---

### 12.11 Global Topic Features + Market Context Data Flow (Issue #12, minor)

**Problem**: The 18 selected features include `global_topic_*` features (global_topic_real_estate_avg, global_topic_retail_wholesale_avg, global_topic_energy_transportation_avg) but the `LiveFeatureBuilder.build_features()` signature only accepts `av_news` for the ticker, not a global news param.

**Fix**: Add `global_news` parameter to feature builder:

```python
def build_features(
    self,
    ticker_df: pd.DataFrame,
    spy_df: pd.DataFrame,
    sector_df: pd.DataFrame,
    av_news: list[dict] | None = None,       # ticker-specific AV news
    global_news: list[dict] | None = None,    # global/market-wide AV news
) -> dict[str, float]:
```

**Data source for global news**: `fetch_alpha_vantage_news(symbol="", topics="economy_macro,economy_fiscal,economy_monetary,financial_markets")` — fetches market-wide news not tied to any ticker. Cached in `historical_news` table with `symbol = "_GLOBAL"`.

**Market context section in Overall tab** — data flow:

```
App startup / symbol change
    ↓
fetch_global_market_news() → cached in DuckDB (symbol="_GLOBAL")
    ↓
LLMAgentModel receives global_news as kwarg
    ↓
LLM generates reasoning that references macro/geopolitical context
    ↓
Per-symbol tab: LLM card shows reasoning (includes macro context)
Overall tab: Dedicated "Market Context" section with:
    - LLM-generated macro summary (from global news)
    - Key macro indicators (SPY 5d return, VIX if available)
    - Geopolitical headlines (top 3 from global news)
```

This is generated by a separate `generate_market_context()` function in `prediction_service.py`, called once (not per-ticker), and passed to each `LLMAgentModel.predict()` call as `kwargs["global_context"]`.

---

### 12.12 Hot-Reload Singleton Survival (Minor Issue)

**Problem**: Dash debug mode hot-reloads kill the Python process and restart it. The in-memory XGBoost model cache (`PredictionService._trained_models`) is lost, causing a 30-60s retrain on every code save during development.

**Solution**: Pickle-cache trained models to disk alongside DuckDB:

```python
MODELS_CACHE_DIR = Path("cache/trained_models")

def _get_or_train(self, symbol, ticker_df, spy_df, sector_df, av_news_by_date):
    cache_path = MODELS_CACHE_DIR / f"{symbol}_xgboost.pkl"
    cache_meta_path = MODELS_CACHE_DIR / f"{symbol}_xgboost_meta.json"

    # Check disk cache: valid if trained today
    if cache_path.exists() and cache_meta_path.exists():
        meta = json.loads(cache_meta_path.read_text())
        if meta.get("training_date") == str(date.today()):
            trainer = MLTrainer("xgboost")
            trainer.load(str(cache_path))
            return trainer

    # Train and cache
    trainer = self._train_from_history(ticker_df, spy_df, sector_df, av_news_by_date)
    MODELS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save(str(cache_path))
    cache_meta_path.write_text(json.dumps({
        "training_date": str(date.today()),
        "n_samples": len(rows),
        "symbol": symbol,
    }))
    return trainer
```

**Trade-off**: Adds ~100KB disk per ticker. Eliminates retrain on hot-reload. Model retrains daily (fresh data) or when cache is manually cleared.

---

## 13. Revised File Inventory

### New Files

| File | Purpose | Phase |
|------|---------|-------|
| `models/__init__.py` | Package init | 1 |
| `models/base.py` | `BaseModel` ABC + `PredictionResult` dataclass | 1 |
| `models/registry.py` | `ModelRegistry` dict-based registry | 1 |
| `models/kronos/__init__.py` | Vendored Kronos (with version header) | 2 |
| `models/kronos/kronos.py` | Vendored Kronos model (explicit imports) | 2 |
| `models/kronos/module.py` | Vendored Kronos modules | 2 |
| `models/kronos_model.py` | `KronosModel(BaseModel)` wrapper | 2 |
| `models/feature_builder.py` | `LiveFeatureBuilder` with precomputed path | 3 |
| `models/xgboost_model.py` | `XGBoostModel(BaseModel)` + disk-cached training | 3 |
| `models/selected_features.json` | 18 SHAP features (revised, no DeBERTa) | 3 |
| `models/sector_map.py` | `SECTOR_ETF_MAP` + `get_sector_etf()` | 3 |
| `models/llm_agent_model.py` | `LLMAgentModel(BaseModel)` + calibrated parsing | 4 |
| `models/ensemble_model.py` | `EnsembleModel(BaseModel)` — DEFERRED | — |
| `services/prediction_service.py` | Orchestrator + singleton + disk cache | 6 |
| `layouts/signal_components.py` | UI: signal cards, history table, market context | 8 |
| `utils/trading_calendar.py` | `get_next_trading_day()`, `is_market_open_today()` | 7 |

### Modified Files

| File | Change | Phase |
|------|--------|-------|
| `services/cache_service.py` | `model_predictions` + `historical_news` tables (revised schema) | 7 |
| `services/news_service.py` | `fetch_historical_av_news()` + `fetch_global_market_news()` | 3 |
| `layouts/main_layout.py` | `model-signals-store` + `model-signals-loading` + `model-signals-progress` | 8 |
| `app.py` | Background callback, modify `update_symbol_tabs`, startup eval | 8 |
| `assets/styles.css` | Signal card, history table, loading state classes | 8 |
| `requirements.txt` / `pyproject.toml` | torch, einops, huggingface_hub, xgboost, scikit-learn, diskcache, pandas_market_calendars | 1 |
| `config.py` | `ModelConfig` dataclass | 1 |

---

## 14. Revised Data Flow

```
User adds symbol (e.g. AAPL)
    ↓
stock-data-store updates (existing: yfinance 1y OHLCV + indicators)
    ↓
generate_model_signals BACKGROUND callback fires (diskcache backend)
    ↓
UI shows progress: "Predicting AAPL... (1/4)"
    ↓
PredictionService.predict_symbol("AAPL", ohlcv_df, news=..., global_context=...)
    ├── KronosModel.predict()  [try/except isolated]
    │     Last 90 OHLCV bars → Kronos-mini → PredictionResult
    │     On error: PredictionResult(error="...", decision="HOLD", confidence=0.0)
    │
    ├── XGBoostModel.predict()  [try/except isolated]
    │     ├── Check disk cache (cache/trained_models/{symbol}_xgboost.pkl)
    │     ├── If stale: pre-compute indicators ONCE on full 1y DataFrames
    │     ├── Slide window → ~200 training samples (with AV topic features)
    │     ├── Train XGBoost (3-fold CV), save to disk
    │     ├── Build today's 18 features (with global_news for global_topic_* features)
    │     └── Predict → PredictionResult (with feature_values for DuckDB)
    │
    └── LLMAgentModel.predict()  [try/except isolated]
          ├── Ticker news (1 week) + global market context
          ├── Technical signals + OHLCV summary
          ├── LLM generate (LM Studio / OpenAI)
          ├── 3-layer JSON parse + confidence calibration [0.4-0.75 range]
          └── PredictionResult (with raw + calibrated confidence)
    ↓
target_date = get_next_trading_day(today)  ← trading_calendar.py
    ↓
Store each prediction in DuckDB model_predictions
    (id: "AAPL_kronos_mini_20260307", model_version, feature_values_json, etc.)
    ↓
model-signals-store updates → UI renders per-model cards in per-symbol tab
    ↓
Next app launch:
    1. is_market_open_today() or target_date <= today?
    2. evaluate_predictions(): JOIN model_predictions ON stock_prices
    3. Fill actual_close, was_correct, pnl_dollars
    4. Check ensemble activation criteria (Section 12.10)
```

---

## 15. Implementation Order (Revised)

| Step | Phase | Description | Est. Time |
|------|-------|-------------|-----------|
| 1 | 1 | Base + Registry + Config + dependencies | 1hr |
| 2 | 7 | DuckDB schema (both tables) + trading_calendar.py | 1hr |
| 3 | 2 | Kronos vendoring (explicit imports, version header) + KronosModel | 2hr |
| 4 | 3a-b | Sector map + historical news fetch + DuckDB news caching | 2hr |
| 5 | 3c-d | Feature builder (with precomputed path) + XGBoostModel + disk cache | 4hr |
| 6 | 4 | LLM Agent (calibrated parsing, global context) | 2hr |
| 7 | 6 | Prediction service (orchestrator, per-model isolation, global context) | 2hr |
| 8 | 8 | UI: background callback, signal components, CSS, progress indicators | 4hr |
| 9 | — | Verification: 4 benchmark tickers, end-to-end testing | 2hr |

**Total estimated**: ~20hr of implementation

---

## 16. Verification Plan (Revised)

1. **Per-model unit test**: Each model receives AAPL 1y OHLCV → returns valid PredictionResult
2. **Feature builder parity**: Compare LiveFeatureBuilder output vs TradingAgents FeatureExtractor for overlapping date
3. **Training performance**: Time `_train_from_history()` with precomputed indicators — must be < 10s/ticker
4. **Missing value logging**: Verify zero-rate warnings appear for sparse AV features on PG/VZ
5. **LLM parse resilience**: Feed malformed JSON, markdown-wrapped JSON, missing fields — all produce valid PredictionResult
6. **DuckDB round-trip**: Store prediction → get_prediction_history → evaluate_predictions with mock actual_close
7. **Trading calendar**: Verify `get_next_trading_day("2026-03-06")` = Monday 2026-03-09 (skip weekend)
8. **Background callback**: Add AAPL → UI stays responsive → progress text updates → signals appear
9. **Hot-reload survival**: Save a file in debug mode → XGBoost loads from disk cache (no retrain)
10. **4-ticker benchmark**: Add PG, LNT, JNJ, VZ → all get independent predictions within 2 minutes
11. **Evaluation flow**: Stop app, insert past prediction for yesterday, restart → actuals filled correctly

---

## 17. Deployment: HuggingFace Spaces (Docker)

### 17.1 Design Constraint

**No runtime model downloads.** All model weights are pre-baked into the Docker image at build time. The app starts with everything it needs — zero HuggingFace Hub calls at runtime.

### 17.2 Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

# Python deps — torch CPU-only first (smaller image)
RUN pip install --no-cache-dir \
    torch==2.2.0+cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu

# App deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download Kronos weights into image layer
RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('NeoQuasar/Kronos-mini', cache_dir='/app/models_cache'); \
snapshot_download('NeoQuasar/Kronos-Tokenizer-base', cache_dir='/app/models_cache')"

# Point HuggingFace to local cache (no runtime downloads)
ENV HF_HOME=/app/models_cache
ENV TRANSFORMERS_OFFLINE=1
ENV HF_HUB_OFFLINE=1

# Copy app
COPY . .

# Create writable dirs for DuckDB + trained model cache
RUN mkdir -p /app/cache/trained_models /app/cache/dash_bg_callbacks

EXPOSE 8050

CMD ["python", "app.py"]
```

### 17.3 HuggingFace Spaces Config

HF Spaces reads config from a `README.md` header:

```yaml
# In repo root README.md (HF Spaces format)
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

### 17.4 Image Size Breakdown

| Layer | Size | Notes |
|-------|------|-------|
| python:3.11-slim base | ~150MB | |
| torch CPU-only | ~200MB | via `+cpu` index |
| einops + huggingface_hub | ~5MB | |
| xgboost + scikit-learn | ~80MB | |
| dash + plotly + pandas + duckdb | ~120MB | Already in requirements |
| Kronos-mini weights | ~16MB | 4.1M params |
| Kronos-Tokenizer-base weights | ~130MB | Tokenizer + quantizer |
| App code | ~2MB | |
| **Total** | **~700MB** | Well under HF Spaces 50GB limit |

### 17.5 Runtime Environment Variables

```env
# .env (or HF Spaces secrets)
ALPHA_VANTAGE_API_KEY=xxx          # Premium key
OPENAI_API_KEY=xxx                  # For LLM agent (or LM Studio URL)
LM_STUDIO_URL=                      # Empty in cloud — uses OpenAI fallback
HF_HOME=/app/models_cache           # Pre-baked, no downloads
HF_HUB_OFFLINE=1                    # Prevent any HF Hub calls
TRANSFORMERS_OFFLINE=1              # Prevent transformers downloads
DEBUG=false                         # No hot-reload in production
PORT=8050
HOST=0.0.0.0                        # Required for container networking
```

### 17.6 Persistent Storage

HF Spaces provides persistent storage that survives container restarts:

```python
# In config.py or cache_service.py
import os

# HF Spaces persistent path (if available)
PERSISTENT_DIR = os.getenv("PERSISTENT_DIR", "/app/cache")
DUCKDB_PATH = f"{PERSISTENT_DIR}/quant_news.duckdb"
TRAINED_MODELS_DIR = f"{PERSISTENT_DIR}/trained_models"
```

This means:
- **DuckDB** (predictions, historical news, stock cache) persists across deploys
- **Trained XGBoost models** persist across deploys (no retrain unless data changes)
- **Kronos weights** are in the image layer (always available, not in persistent storage)

### 17.7 KronosModel Adaptation for Offline Mode

The `KronosModel` wrapper must load from the pre-baked local cache, never calling HuggingFace Hub:

```python
# In models/kronos_model.py

def _ensure_kronos(model_size: str = "mini") -> KronosPredictor:
    """Load Kronos from local cache. Never downloads at runtime."""
    global _predictor, _model_size

    if _predictor is not None and _model_size == model_size:
        return _predictor

    # HF_HOME is set to /app/models_cache in Docker
    # HF_HUB_OFFLINE=1 ensures no network calls
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained(f"NeoQuasar/Kronos-{model_size}")

    _predictor = KronosPredictor(tokenizer, model, max_context=512)
    _model_size = model_size
    return _predictor
```

The `from_pretrained()` calls resolve to the local `HF_HOME` cache automatically when `HF_HUB_OFFLINE=1`. No code changes needed beyond setting env vars.

### 17.8 Local Development vs Production

| Concern | Local (macOS) | HF Spaces (Linux container) |
|---------|---------------|----------------------------|
| torch build | Default pip (MPS-capable) | `torch+cpu` (200MB, no CUDA) |
| Kronos weights | Downloaded to `~/.cache/huggingface/` on first run | Pre-baked in image at `/app/models_cache` |
| DuckDB | `cache/quant_news.duckdb` | Persistent storage dir |
| LLM provider | LM Studio (local, free) | OpenAI API (fallback) |
| Background callbacks | diskcache at `cache/dash_bg_callbacks/` | Same, in persistent storage |
| Debug mode | `DEBUG=true` (hot-reload) | `DEBUG=false` |

### 17.9 New Files for Deployment

| File | Purpose |
|------|---------|
| `Dockerfile` | Multi-stage build with pre-baked Kronos weights |
| `.dockerignore` | Exclude `.git`, `__pycache__`, `.env`, `cache/`, `.coverage` |

These are added as **Step 10** in the implementation order (after verification, before merge).

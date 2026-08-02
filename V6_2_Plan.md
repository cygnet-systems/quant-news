# Add DeBERTa, LightGBM & Live Ensemble Models

## Context

The strategy framework (Steps 1-10 above) is **fully implemented**. Currently 3 models produce predictions: Kronos, XGBoost, LLM Agent. The V6 research doc identifies 3 additional models that were deferred: DeBERTa sentiment, LightGBM, and a live Ensemble combiner. User requested all 3 be activated so the UI shows 6 signal cards per symbol (5 individual + 1 ensemble).

**Important: Individual predictions are NOT replaced by the ensemble.** All 5 individual models (Kronos, XGBoost, LightGBM, DeBERTa, LLM Agent) each produce and store their own independent prediction with their own decision, confidence, and model-specific details. The Ensemble is a 6th model that runs after the other 5 and produces an additional combined signal. The UI renders all 6 as separate signal cards so the user can see both individual model opinions AND the ensemble consensus side by side.

## Key Findings from V6 Research

- **DeBERTa**: `mrm8488/deberta-v3-ft-financial-news-sentiment-analysis` — 73% BUY bias on raw AV news. Critical: must filter by `relevance_score >= 0.7` to get usable signal. Use as standalone model, NOT as XGBoost feature (V6 showed it reduces accuracy).
- **LightGBM**: Isomorphic to XGBoost — same 18 SHAP features from `models/feature_builder.py`, swap `XGBClassifier` → `LGBMClassifier`. Better P&L (+$1,787 vs +$482) but worse OOS stability.
- **Live Ensemble**: Real-time `BaseModel` subclass that runs after all individual models complete. Weighted majority vote across all 5 models.

## Implementation Steps

### Step 1: Dependencies & Config
**File:** `requirements.txt`
- Add `transformers>=4.35.0`, `lightgbm>=4.0.0`

**File:** `config.py`
- Add to `ModelConfig`: `DEBERTA_MODEL_NAME`, `DEBERTA_RELEVANCE_THRESHOLD` (0.7), `DEBERTA_BUY_THRESHOLD` (0.6), `DEBERTA_SELL_THRESHOLD` (0.4)
- Add to `ModelConfig`: `LIGHTGBM_PARAMS` dict (num_leaves=31, learning_rate=0.05, n_estimators=200, verbose=-1)
- Default ensemble config (used as initial values for the UI, NOT hardcoded overrides):
  - `ENSEMBLE_DEFAULT_ENABLED`: `["kronos_mini", "xgboost_shap"]` (only best models on by default)
  - `ENSEMBLE_DEFAULT_WEIGHTS`: `{"kronos_mini": 1.0, "xgboost_shap": 1.0, "lightgbm": 1.0, "deberta_sentiment": 1.0, "llm_agent": 1.0}` (all weights start equal — user adjusts from UI)

### Step 2: LightGBM Model
**New file:** `models/lightgbm_model.py`
- Mirror `models/xgboost_model.py` exactly — same 18 features from `feature_builder.build_features()`
- Swap `XGBClassifier` → `LGBMClassifier`
- Same disk-cache pattern for trained model
- `name = "lightgbm"`, priority same as xgboost

### Step 3: DeBERTa Sentiment Model
**New file:** `models/deberta_model.py`
- HuggingFace `pipeline("sentiment-analysis", model=MODEL.DEBERTA_MODEL_NAME)`
- Input: cached AV news for symbol, filtered by `relevance_score >= MODEL.DEBERTA_RELEVANCE_THRESHOLD`
- Aggregate sentiment scores across filtered articles → BUY/SELL/HOLD
- Thresholds: avg positive score > `DEBERTA_BUY_THRESHOLD` → BUY, < `DEBERTA_SELL_THRESHOLD` → SELL
- `confidence_type = "self_reported"` (same as LLM agent)
- Lazy-load pipeline on first call to avoid startup delay
- If no relevant articles found → return HOLD with low confidence

### Step 4: Live Ensemble Model
**New file:** `models/ensemble_model.py`
- `EnsembleModel(BaseModel)` with `name = "ensemble"`
- `predict()` receives `other_results: dict[str, PredictionResult]` AND `ensemble_config: dict` via kwargs
- `ensemble_config` comes from the UI store: `{"enabled_models": [...], "weights": {...}}`
- Only considers models listed in `enabled_models`; uses weights from `weights` dict
- Weighted vote logic (same math as `strategies/ensemble_vote.py`): BUY=+1, SELL=-1, HOLD=0, multiply by weight, normalize
- Needs ≥2 non-error enabled results to produce signal, otherwise HOLD
- Returns `PredictionResult` with transparent metadata: `{"votes": {"kronos_mini": "BUY", ...}, "weights_used": {...}, "weighted_score": 0.65, "models_enabled": [...], "models_excluded": [...]}`

### Step 5: Prediction Service Updates
**File:** `services/prediction_service.py`
- Update execution phases:
  - Phase 1 (sequential): Kronos (needs market data first)
  - Phase 2 (parallel): XGBoost, LightGBM, LLM Agent, DeBERTa
  - Phase 3 (sequential): Ensemble (depends on all Phase 1+2 results)
- Pass Phase 1+2 results to Ensemble via `other_results` kwarg
- Store all 6 predictions in DuckDB

### Step 6: UI — Ensemble Configuration (Offcanvas Drawer Pattern)

**Design rationale**: Follows TradingView's per-component gear icon pattern and Bloomberg's "conceal complexity" principle. Default view is transparent (weight bar shows composition at a glance). Full controls available on demand via drawer. No arbitrary hidden configs.

**A. Ensemble Card (inline summary — always visible)**
**File:** `layouts/signal_components.py`
- Add display names: `"lightgbm" → "LightGBM"`, `"deberta_sentiment" → "DeBERTa"`, `"ensemble" → "Ensemble"`
- Ensemble card is visually distinct: full-width (spans both columns), subtle highlight border
- Card header: "Ensemble" label + decision badge + **gear icon button** (opens config drawer)
- Card body shows:
  - Confidence bar (same as other cards)
  - **Weight composition bar** — horizontal CSS flex bar showing relative model contributions, color-coded per model (e.g. Kronos=teal, XGBoost=blue, etc.), with model initials inside each segment. Disabled models are absent from the bar. Pure CSS (`display: flex`, each segment's `flex-grow` = normalized weight). Provides transparency without any controls.
  - **Vote summary row** — compact inline: "K: BUY | XG: BUY | LG: SELL | D: HOLD | LLM: BUY" with color-coded text. Disabled models shown as dimmed/strikethrough.

**B. Offcanvas Config Drawer**
**File:** `layouts/components.py` — new `create_ensemble_config_drawer()` function
- `dbc.Offcanvas(placement="end", style={"width": "380px"})` — slides in from right
- Triggered by gear icon on ensemble card
- Content per model (one row each):
  - `dbc.Switch` — enable/disable (checked = included in ensemble)
  - Model name + its current decision shown inline (e.g. "Kronos — BUY")
  - `dcc.Slider(min=0.1, max=2.0, step=0.1, marks={0.5: "0.5", 1.0: "1.0", 1.5: "1.5", 2.0: "2.0"})` — weight slider
  - `dcc.Input(type="number", min=0.1, max=2.0, step=0.1)` — precise keyboard entry (synced with slider)
  - Slider disabled/greyed when model switch is off
- Footer: "Reset to Defaults" button (restores `ENSEMBLE_DEFAULT_ENABLED` + `ENSEMBLE_DEFAULT_WEIGHTS`)
- Default state: only Kronos + XGBoost enabled, all weights at 1.0

**C. State Management**
**File:** `layouts/main_layout.py`
- Add `dcc.Store(id="ensemble-config-store", data={"enabled_models": ["kronos_mini", "xgboost_shap"], "weights": {"kronos_mini": 1.0, "xgboost_shap": 1.0, "lightgbm": 1.0, "deberta_sentiment": 1.0, "llm_agent": 1.0}})` — persists user's ensemble configuration
- Initial values from `config.ENSEMBLE_DEFAULT_*`

**D. Callbacks**
**File:** `app.py`
- `sync_ensemble_config` callback: reads switches + sliders → writes to `ensemble-config-store`
- `recompute_ensemble` callback: when `ensemble-config-store` or `model-signals-store` changes → re-run ensemble weighted vote client-side → update ensemble card in `model-signals-store`
- `toggle_ensemble_drawer` callback: gear icon click → open/close offcanvas

**E. Other Signal Cards (individual models)**
- Each of the 5 individual cards shows independently as before:
  - **Kronos**: decision, confidence, predicted close price (existing)
  - **XGBoost**: decision, confidence, up probability, training samples (existing)
  - **LightGBM**: decision, confidence, up probability, training samples (same layout as XGBoost)
  - **DeBERTa**: decision, confidence, articles analyzed count, avg sentiment score
  - **LLM Agent**: decision, confidence, reasoning text (existing)

**F. CSS**
**File:** `assets/styles.css`
- `.signal-cards-grid` — 2-column grid for individual cards, ensemble card spans full width below
- `.ensemble-weight-bar` — flex container with color-coded segments
- `.ensemble-card` — full-width, subtle accent border to distinguish from individual cards
- `.ensemble-vote-summary` — compact inline vote display
- Offcanvas internal styling: slider rows, model switches, responsive layout

### Step 7: Strategy Weights Update
**File:** `strategies/ensemble_vote.py`
- Add `"lightgbm"` and `"deberta_sentiment"` to `_MODEL_WEIGHTS`
- Accept optional `config_override` parameter so strategy evaluations can also use custom weights

## Files to Modify/Create
- `requirements.txt` — add 2 deps
- `config.py` — add DeBERTa/LightGBM config + ensemble defaults (NOT overrides)
- `models/lightgbm_model.py` — NEW, mirrors xgboost_model.py
- `models/deberta_model.py` — NEW, HF pipeline wrapper
- `models/ensemble_model.py` — NEW, weighted vote combiner (reads config from kwargs, not hardcoded)
- `services/prediction_service.py` — 3-phase execution, pass ensemble_config to ensemble model
- `layouts/signal_components.py` — 3 new display names, ensemble card with weight bar + vote summary + gear icon
- `layouts/components.py` — new `create_ensemble_config_drawer()` (Offcanvas with switches + sliders)
- `layouts/main_layout.py` — add `ensemble-config-store`, offcanvas component
- `strategies/ensemble_vote.py` — 2 new model weights, optional config_override param
- `assets/styles.css` — 6-card grid, ensemble card full-width, weight bar, offcanvas styling
- `app.py` — ensemble config callbacks (sync store, recompute ensemble, toggle drawer)

## Key Patterns to Reuse
- `models/xgboost_model.py` → template for LightGBM (identical structure)
- `models/llm_agent_model.py` → `confidence_type = "self_reported"` pattern for DeBERTa
- `models/base.py:PredictionResult` → all models return this
- `services/prediction_service.py` → existing Phase 1/2 parallel execution pattern
- `strategies/ensemble_vote.py` → weighted vote math for live ensemble model
- `services/cache_service.py` → `get_news_for_symbol()` for DeBERTa input
- `layouts/components.py:create_indicator_toggles()` → pattern for ensemble switches
- `dbc.Offcanvas` → Dash Bootstrap native drawer component
- Existing Raw Data modal in `main_layout.py` → callback pattern for open/close

## Verification
1. `python -c "from models.lightgbm_model import LightGBMModel; print(LightGBMModel().name)"` → "lightgbm"
2. `python -c "from models.deberta_model import DeBERTaModel; print(DeBERTaModel().name)"` → "deberta_sentiment"
3. `python -c "from models.ensemble_model import EnsembleModel; print(EnsembleModel().name)"` → "ensemble"
4. Run app, trigger prediction → 5 individual signal cards + 1 full-width ensemble card per symbol
5. Ensemble card shows weight composition bar (only Kronos + XGBoost colored by default)
6. Ensemble card shows vote summary with all 5 models (disabled ones dimmed)
7. Click gear icon → Offcanvas drawer slides in from right
8. Toggle LightGBM switch ON → ensemble card updates live with new weighted result
9. Adjust XGBoost weight slider to 1.5 → weight bar and ensemble decision update
10. Click "Reset to Defaults" → reverts to Kronos + XGBoost only, all weights at 1.0
11. DeBERTa card shows articles analyzed count and avg sentiment score
12. `python scripts/evaluate.py --list` → shows strategies with all model names
13. Playwright: verify all cards render, drawer opens/closes, sliders work, no console errors

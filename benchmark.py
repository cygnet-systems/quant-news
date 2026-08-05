"""Walk-forward backtest with strict lookahead prevention.

For each test day T:
  1. Truncate ALL data to [0..T] — no future data visible
  2. XGBoost: retrain from scratch on data[:T], predict day T+1
  3. Kronos: feed data[:T], predict day T+1
  4. LLM: feed data[:T], predict day T+1
  5. Record actual direction and P&L on day T+1

Usage:
    python benchmark.py [--symbols AAPL,MSFT] [--days 20] [--mc-samples 5]
"""

import argparse
import logging
import os
import sys
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Prevent HF network calls — weights are cached
os.environ["HF_HUB_OFFLINE"] = "1"

from models.base import compute_pnl, POSITION_SIZE_USD
from models.feature_builder import LiveFeatureBuilder, SELECTED_FEATURES
from services.analytics import add_indicators_to_df

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class DayResult:
    date: str
    model: str
    symbol: str
    decision: str
    confidence: float
    up_probability: float
    actual_return_pct: float
    actual_direction: str  # "UP" or "DOWN"
    correct: bool
    pnl: float
    predicted_close: float | None = None
    news_count: int | None = None


@dataclass
class BenchmarkResults:
    results: list[DayResult] = field(default_factory=list)

    def add(self, r: DayResult) -> None:
        self.results.append(r)

    def summary(self) -> pd.DataFrame:
        if not self.results:
            return pd.DataFrame()
        df = pd.DataFrame([vars(r) for r in self.results])
        return df

    def print_report(self) -> None:
        df = self.summary()
        if df.empty:
            print("No results to report.")
            return

        print("\n" + "=" * 80)
        print("  WALK-FORWARD BACKTEST RESULTS (Strict No-Lookahead)")
        print("=" * 80)

        for model in df["model"].unique():
            mdf = df[df["model"] == model]
            n = len(mdf)
            # Only count BUY/SELL predictions for accuracy (HOLD is neutral)
            active = mdf[mdf["decision"] != "HOLD"]
            n_active = len(active)
            n_correct = active["correct"].sum() if n_active > 0 else 0
            accuracy = n_correct / n_active * 100 if n_active > 0 else 0
            total_pnl = mdf["pnl"].sum()
            avg_conf = mdf["confidence"].mean()

            buy_count = len(mdf[mdf["decision"] == "BUY"])
            sell_count = len(mdf[mdf["decision"] == "SELL"])
            hold_count = len(mdf[mdf["decision"] == "HOLD"])

            win_rate_all = mdf["correct"].mean() * 100

            print(f"\n  --- {model} ---")
            print(f"  Days tested:     {n}")
            print(f"  BUY/SELL/HOLD:   {buy_count}/{sell_count}/{hold_count}")
            print(f"  Active accuracy: {accuracy:.1f}% ({n_correct}/{n_active} BUY/SELL correct)")
            print(f"  All-day accuracy:{win_rate_all:.1f}% (incl HOLD=neutral)")
            print(f"  Total P&L:       ${total_pnl:+,.2f} (${POSITION_SIZE_USD:.0f}/trade)")
            print(f"  Avg confidence:  {avg_conf:.0%}")

            # Per-symbol breakdown
            for symbol in mdf["symbol"].unique():
                sdf = mdf[mdf["symbol"] == symbol]
                s_active = sdf[sdf["decision"] != "HOLD"]
                s_acc = s_active["correct"].mean() * 100 if len(s_active) > 0 else 0
                s_pnl = sdf["pnl"].sum()
                print(f"    {symbol}: {s_acc:.0f}% active acc, ${s_pnl:+,.2f} P&L, "
                      f"{len(s_active)}/{len(sdf)} active days")

        print("\n" + "=" * 80)

        # Market baseline
        for symbol in df["symbol"].unique():
            sdf = df[df["symbol"] == symbol].drop_duplicates(subset=["date"])
            buy_hold_pnl = sdf["actual_return_pct"].sum() / 100 * POSITION_SIZE_USD
            up_days = (sdf["actual_direction"] == "UP").sum()
            print(f"  Baseline {symbol}: buy-and-hold P&L=${buy_hold_pnl:+,.2f}, "
                  f"{up_days}/{len(sdf)} up days ({up_days/len(sdf)*100:.0f}%)")

        print("=" * 80 + "\n")


def run_xgboost_backtest(
    symbol: str,
    ticker_full: pd.DataFrame,
    spy_full: pd.DataFrame,
    sector_full: pd.DataFrame,
    test_indices: list[int],
    results: BenchmarkResults,
) -> None:
    """Walk-forward XGBoost: retrain at each step on data[:T]."""
    from config import MODEL
    from xgboost import XGBClassifier

    XGBOOST_PARAMS = {
        "n_estimators": MODEL.XGBOOST_N_ESTIMATORS,
        "max_depth": MODEL.XGBOOST_MAX_DEPTH,
        "learning_rate": MODEL.XGBOOST_LEARNING_RATE,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "eval_metric": "logloss",
        "random_state": 42,
        "verbosity": 0,
    }

    builder = LiveFeatureBuilder()
    threshold = MODEL.LABEL_AMBIGUITY_THRESHOLD
    min_window = 60

    for t in test_indices:
        # Train on data[:t] only — strictly no future data
        rows, labels = [], []
        for i in range(min_window, t):
            close_today = float(ticker_full["Close"].iloc[i])
            close_next = float(ticker_full["Close"].iloc[i + 1])
            if close_today <= 0:
                continue
            pct = (close_next - close_today) / close_today
            if threshold > 0.0 and abs(pct) < threshold:
                continue
            try:
                features = builder.build_features(
                    ticker_full.iloc[:i + 1],
                    spy_full.iloc[:i + 1],
                    sector_full.iloc[:i + 1],
                )
                rows.append(features)
                labels.append(1 if pct > 0 else 0)
            except Exception:
                continue

        if len(rows) < MODEL.MIN_TRAINING_SAMPLES:
            continue

        X = pd.DataFrame(rows)[SELECTED_FEATURES]
        y = pd.Series(labels)
        clf = XGBClassifier(**XGBOOST_PARAMS)
        clf.fit(X, y)

        # Predict day T using features at day T (no T+1 data)
        try:
            today_features = builder.build_features(
                ticker_full.iloc[:t + 1],
                spy_full.iloc[:t + 1],
                sector_full.iloc[:t + 1],
            )
            X_pred = pd.DataFrame([today_features])[SELECTED_FEATURES]
            prob = clf.predict_proba(X_pred)[0]
            up_prob = float(prob[1]) if len(prob) > 1 else 0.5
        except Exception:
            continue

        if up_prob > MODEL.BUY_THRESHOLD:
            decision = "BUY"
        elif up_prob < MODEL.SELL_THRESHOLD:
            decision = "SELL"
        else:
            decision = "HOLD"

        confidence = max(up_prob, 1 - up_prob)

        # Actual outcome: day T+1
        close_t = float(ticker_full["Close"].iloc[t])
        close_t1 = float(ticker_full["Close"].iloc[t + 1])
        actual_ret = (close_t1 - close_t) / close_t * 100
        actual_dir = "UP" if close_t1 > close_t else "DOWN"

        correct = (
            (decision == "BUY" and actual_dir == "UP") or
            (decision == "SELL" and actual_dir == "DOWN") or
            (decision == "HOLD")
        )
        pnl = compute_pnl(decision, close_t, close_t1)

        results.add(DayResult(
            date=str(ticker_full.index[t])[:10],
            model="xgboost_shap",
            symbol=symbol,
            decision=decision,
            confidence=round(confidence, 3),
            up_probability=round(up_prob, 4),
            actual_return_pct=round(actual_ret, 4),
            actual_direction=actual_dir,
            correct=correct,
            pnl=round(pnl, 2),
        ))


def run_kronos_backtest(
    symbol: str,
    ticker_df: pd.DataFrame,
    test_indices: list[int],
    results: BenchmarkResults,
    mc_samples: int = 5,
) -> None:
    """Walk-forward Kronos: feed data[:T], predict next day."""
    try:
        from models.kronos_model import KronosModel, _get_predictor, KRONOS_AVAILABLE
        from config import MODEL
    except ImportError:
        print("  Kronos not available, skipping.", flush=True)
        return

    if not KRONOS_AVAILABLE:
        print("  Kronos not available (torch missing), skipping.", flush=True)
        return

    predictor = _get_predictor()  # Load once
    context_bars = MODEL.KRONOS_CONTEXT_BARS

    from utils.trading_calendar import get_next_trading_day

    for t in test_indices:
        # Truncate to day T
        df_trunc = ticker_df.iloc[:t + 1].tail(context_bars).copy()
        df_trunc = df_trunc.rename(columns={
            "Open": "open", "High": "high", "Low": "low",
            "Close": "close", "Volume": "volume",
        })
        df_trunc = df_trunc.dropna(subset=["open", "high", "low", "close"])
        if len(df_trunc) < 30:
            continue

        x_ts = pd.DatetimeIndex(df_trunc.index)
        last_date = x_ts[-1].date()
        next_day = get_next_trading_day(last_date)
        y_ts = pd.DatetimeIndex([pd.Timestamp(next_day)])

        try:
            pred_closes = []
            for _ in range(mc_samples):
                pred_df = predictor.predict(
                    df_trunc[["open", "high", "low", "close", "volume"]],
                    x_ts, y_ts,
                    pred_len=1,
                    T=MODEL.KRONOS_TEMPERATURE,
                    top_k=0, top_p=0.9, sample_count=1, verbose=False,
                )
                pred_closes.append(float(pred_df["close"].iloc[0]))
        except Exception as e:
            logger.warning(f"Kronos failed at {ticker_df.index[t]}: {e}")
            continue

        current_close = float(df_trunc["close"].iloc[-1])
        pred_median = float(np.median(pred_closes))
        up_count = sum(1 for c in pred_closes if c > current_close)
        up_prob = up_count / mc_samples

        if up_prob > MODEL.BUY_THRESHOLD:
            decision = "BUY"
        elif up_prob < MODEL.SELL_THRESHOLD:
            decision = "SELL"
        else:
            decision = "HOLD"

        confidence = max(up_prob, 1 - up_prob)

        # Actual outcome
        close_t = float(ticker_df["Close"].iloc[t])
        close_t1 = float(ticker_df["Close"].iloc[t + 1])
        actual_ret = (close_t1 - close_t) / close_t * 100
        actual_dir = "UP" if close_t1 > close_t else "DOWN"

        correct = (
            (decision == "BUY" and actual_dir == "UP") or
            (decision == "SELL" and actual_dir == "DOWN") or
            (decision == "HOLD")
        )
        pnl = compute_pnl(decision, close_t, close_t1)

        results.add(DayResult(
            date=str(ticker_df.index[t])[:10],
            model="kronos_mini",
            symbol=symbol,
            decision=decision,
            confidence=round(confidence, 3),
            up_probability=round(up_prob, 4),
            actual_return_pct=round(actual_ret, 4),
            actual_direction=actual_dir,
            correct=correct,
            pnl=round(pnl, 2),
            predicted_close=round(pred_median, 2),
        ))


def run_all_models_backtest(
    symbol: str,
    ticker_df: pd.DataFrame,
    spy_df: pd.DataFrame,
    test_indices: list[int],
    results: BenchmarkResults,
    *,
    skip: set[str] | None = None,
) -> None:
    """Walk-forward across the whole registry, including the ensemble.

    The per-model runners above cover Kronos/XGBoost/the single agent, but
    nothing here exercised `ensemble` -- which is the thing that actually
    ships a decision. Delegating to prediction_service means the ensemble is
    fed exactly the member results it would see live, and the Kronos-before-
    XGBoost phase ordering (torch/MPS deadlock) is handled there rather than
    duplicated.
    """
    from services.prediction_service import PredictionService

    service = PredictionService()
    skip = skip or set()
    to_run = {m for m in service._registry.list_models() if m not in skip}
    if not to_run:
        return

    for t in test_indices:
        df_trunc = ticker_df.iloc[:t + 1]              # no future bars
        as_of = str(ticker_df.index[t])[:10]
        spy_trunc = spy_df[spy_df.index <= ticker_df.index[t]] if spy_df is not None else None

        try:
            preds = service.predict_symbol_no_store(
                symbol, df_trunc,
                spy_df=spy_trunc,
                as_of=as_of,
                models_to_run=to_run,
                run_ensemble=True,
            )
        except Exception as e:
            print(f"    {as_of} all-models ERROR: {e}", flush=True)
            continue

        close_t = float(ticker_df["Close"].iloc[t])
        close_t1 = float(ticker_df["Close"].iloc[t + 1])
        actual_ret = (close_t1 - close_t) / close_t * 100
        actual_dir = "UP" if close_t1 > close_t else "DOWN"

        for model_name, r in (preds or {}).items():
            decision = r.get("decision", "HOLD")
            correct = (
                (decision == "BUY" and actual_dir == "UP") or
                (decision == "SELL" and actual_dir == "DOWN") or
                (decision == "HOLD")
            )
            results.add(DayResult(
                date=as_of,
                model=model_name,
                symbol=symbol,
                decision=decision,
                confidence=round(float(r.get("confidence") or 0.0), 3),
                up_probability=round(float(r.get("up_probability") or 0.5), 4),
                actual_return_pct=round(actual_ret, 4),
                actual_direction=actual_dir,
                correct=correct,
                pnl=round(compute_pnl(decision, close_t, close_t1), 2),
                news_count=(r.get("details") or {}).get("news_count"),
            ))

        summary = ", ".join(
            f"{m}={(r.get('decision') or '?')}" for m, r in sorted((preds or {}).items())
        )
        print(f"    {as_of} [all-models] {summary}", flush=True)


def run_llm_backtest(
    symbol: str,
    ticker_df: pd.DataFrame,
    test_indices: list[int],
    results: BenchmarkResults,
    *,
    use_news: bool = True,
    use_reflection: bool = False,
    label: str = "trading_agents",
    model_name: str | None = None,
) -> None:
    """Walk-forward single-agent research: feed data[:T], predict day T+1.

    `use_news=False` runs the news-ablation arm (Tier 1 A/B): the agent sees
    the identical technicals/fundamentals but an empty news block, so any change
    in decisions/accuracy vs the `use_news=True` arm is attributable to the
    point-in-time news coverage the fix adds.
    """
    from models.trading_agents_model import TradingAgentsModel
    model = TradingAgentsModel()
    if not model.is_ready():
        print("  Single-agent model not ready (no ANTHROPIC_API_KEY), skipping.",
              flush=True)
        return

    for t in test_indices:
        # Truncate to day T — no future data
        df_trunc = ticker_df.iloc[:t + 1]
        as_of = str(ticker_df.index[t])[:10]

        result = model.predict(
            symbol, df_trunc,
            as_of=as_of, use_news=use_news, use_reflection=use_reflection,
            model=model_name,
        )

        close_t = float(ticker_df["Close"].iloc[t])
        close_t1 = float(ticker_df["Close"].iloc[t + 1])
        actual_ret = (close_t1 - close_t) / close_t * 100
        actual_dir = "UP" if close_t1 > close_t else "DOWN"

        correct = (
            (result.decision == "BUY" and actual_dir == "UP") or
            (result.decision == "SELL" and actual_dir == "DOWN") or
            (result.decision == "HOLD")
        )
        pnl = compute_pnl(result.decision, close_t, close_t1)

        results.add(DayResult(
            date=as_of,
            model=label,
            symbol=symbol,
            decision=result.decision,
            confidence=round(result.confidence, 3),
            up_probability=round(result.up_probability, 4),
            actual_return_pct=round(actual_ret, 4),
            actual_direction=actual_dir,
            correct=correct,
            pnl=round(pnl, 2),
            news_count=result.details.get("news_count"),
        ))
        if result.error:
            print(f"    {as_of} [{label}] ERROR: {result.error}", flush=True)
        else:
            nc = result.details.get("news_count", 0)
            print(f"    {as_of} [{label}] {result.decision} "
                  f"({result.confidence:.0%}), {nc} articles", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Walk-forward backtest")
    parser.add_argument("--symbols", default="AAPL", help="Comma-separated symbols")
    parser.add_argument("--days", type=int, default=20, help="Test window (trading days)")
    parser.add_argument("--mc-samples", type=int, default=5, help="Kronos MC samples")
    parser.add_argument("--skip-kronos", action="store_true", help="Skip Kronos model")
    parser.add_argument("--skip-llm", action="store_true", help="Skip LLM model")
    parser.add_argument("--skip-xgboost", action="store_true", help="Skip XGBoost model")
    parser.add_argument("--llm-ab", action="store_true",
                        help="Run the single-agent news A/B (with-news vs no-news arms)")
    parser.add_argument("--llm-reflect", action="store_true",
                        help="Add a reflection arm (news-on + reflection-on) to compare "
                             "against the news-on baseline")
    parser.add_argument("--all-models", action="store_true",
                        help="Run the full registry (incl. ensemble) per test day "
                             "via prediction_service, alongside any LLM arms")
    parser.add_argument("--llm-model", default=None,
                        help="Override LLM model id for the single-agent backtest "
                             "(e.g. claude-haiku-4-5-20251001 for a cheap A/B)")
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",")]
    test_days = args.days

    print(f"Walk-forward backtest: {symbols}, {test_days} test days", flush=True)
    print(f"MC samples (Kronos): {args.mc_samples}", flush=True)
    print(f"Position size: ${POSITION_SIZE_USD:.0f}", flush=True)
    print()

    from services.stock_data import fetch_stock_data

    # Fetch SPY once
    print("Fetching SPY...", flush=True)
    spy_raw = fetch_stock_data("SPY", period="1y")
    spy_full = add_indicators_to_df(spy_raw.copy())

    bench = BenchmarkResults()

    for symbol in symbols:
        print(f"\n{'='*40}", flush=True)
        print(f"  Benchmarking {symbol}", flush=True)
        print(f"{'='*40}", flush=True)

        ticker_raw = fetch_stock_data(symbol, period="1y")
        ticker_full = add_indicators_to_df(ticker_raw.copy())

        # Resolve sector ETF
        from models.sector_map import get_sector_etf
        sector_etf = get_sector_etf(symbol)
        try:
            sector_raw = fetch_stock_data(sector_etf, period="1y")
            sector_full = add_indicators_to_df(sector_raw.copy())
        except Exception:
            sector_full = spy_full  # Fallback

        n = len(ticker_full)
        # Test indices: last `test_days` days (excluding very last day which has no T+1)
        test_start = max(80, n - test_days - 1)  # Need 80+ bars for indicators+training
        test_end = n - 1  # Exclusive — last index with a T+1 available
        test_indices = list(range(test_start, test_end))
        actual_test_days = len(test_indices)

        print(f"  Data: {n} bars, test window: indices [{test_start}..{test_end})")
        print(f"  Test dates: {str(ticker_full.index[test_start])[:10]} to "
              f"{str(ticker_full.index[test_end-1])[:10]}")
        print(f"  Actual test days: {actual_test_days}", flush=True)

        # --all-models already walks Kronos and XGBoost through the registry,
        # so the dedicated arms below would run them a SECOND time and append
        # both passes under the same model label — doubling their reported
        # sample counts (20 "days tested" for a 10-day window) and making the
        # results look better powered than they are. The LLM arm was already
        # de-duplicated this way; these two were not.
        dedicated_arms = not args.all_models

        # Kronos FIRST — must load torch before XGBoost pickle (MPS deadlock)
        if dedicated_arms and not args.skip_kronos:
            print(f"\n  Running Kronos walk-forward ({actual_test_days} days, "
                  f"{args.mc_samples} MC samples each)...", flush=True)
            t0 = time.time()
            run_kronos_backtest(
                symbol, ticker_raw, test_indices, bench,
                mc_samples=args.mc_samples,
            )
            print(f"  Kronos done in {time.time()-t0:.1f}s", flush=True)

        # XGBoost
        if dedicated_arms and not args.skip_xgboost:
            print(f"\n  Running XGBoost walk-forward ({actual_test_days} retrains)...",
                  flush=True)
            t0 = time.time()
            run_xgboost_backtest(
                symbol, ticker_full, spy_full, sector_full, test_indices, bench,
            )
            print(f"  XGBoost done in {time.time()-t0:.1f}s", flush=True)

        # Full registry incl. ensemble (runs before the LLM arms so Kronos
        # loads torch first -- same constraint prediction_service enforces).
        if args.all_models:
            print(f"\n  All-models walk-forward ({actual_test_days} days, "
                  f"incl. ensemble)...", flush=True)
            t0 = time.time()
            run_all_models_backtest(
                symbol, ticker_raw, spy_raw, test_indices, bench,
                # the single agent is covered by the dedicated A/B arms below;
                # running it here too would double the LLM spend
                skip={"trading_agents"},
            )
            print(f"  All-models done in {time.time()-t0:.1f}s", flush=True)

        # LLM (single-agent research)
        if not args.skip_llm:
            t0 = time.time()
            if args.llm_ab or args.llm_reflect:
                print(f"\n  Single-agent arms ({actual_test_days} days each): "
                      f"news{'+/-' if args.llm_ab else ''}"
                      f"{' +reflect' if args.llm_reflect else ''}...", flush=True)
                # Baseline: news on, reflection off.
                run_llm_backtest(symbol, ticker_raw, test_indices, bench,
                                 use_news=True, label="trading_agents",
                                 model_name=args.llm_model)
                if args.llm_ab:  # news ablation
                    run_llm_backtest(symbol, ticker_raw, test_indices, bench,
                                     use_news=False, label="trading_agents_nonews",
                                     model_name=args.llm_model)
                if args.llm_reflect:  # reflection ablation (news on + reflection on)
                    run_llm_backtest(symbol, ticker_raw, test_indices, bench,
                                     use_news=True, use_reflection=True,
                                     label="trading_agents_reflect",
                                     model_name=args.llm_model)
            else:
                print(f"\n  Running single-agent walk-forward ({actual_test_days} days)...",
                      flush=True)
                run_llm_backtest(symbol, ticker_raw, test_indices, bench,
                                 model_name=args.llm_model)
            print(f"  LLM done in {time.time()-t0:.1f}s", flush=True)

    bench.print_report()


if __name__ == "__main__":
    main()

"""Evaluation service orchestrator.

Bridges model predictions → strategy evaluations → vectorbt metrics.
Runs automatically via APScheduler or manually via scripts/evaluate.py.
"""

import logging
from collections import defaultdict
from datetime import datetime
from typing import Optional

from models.base import compute_pnl
from services.cache_service import get_cache
from strategies.base import BaseStrategy
from strategies.registry import StrategyRegistry

logger = logging.getLogger(__name__)

_service_instance = None


class EvaluationService:
    """Orchestrates strategy evaluation against model predictions."""

    def __init__(self) -> None:
        self._registry = StrategyRegistry()
        logger.info(f"Strategies registered: {self._registry.list_strategies()}")

    def run_evaluation(
        self, strategy_name: Optional[str] = None
    ) -> dict[str, int]:
        """Run strategies against unevaluated predictions.

        1. Fills actual_close on pending predictions (existing logic).
        2. For each strategy, evaluates predictions it hasn't processed yet.
        3. Refreshes vectorbt metrics for strategies with new evaluations.

        Args:
            strategy_name: Run only this strategy. None = all.

        Returns:
            Dict mapping strategy_name to count of new evaluations.
        """
        cache = get_cache()

        # Step 1: Fill actual_close on pending predictions
        n_evaluated = cache.evaluate_predictions()
        if n_evaluated > 0:
            logger.info(f"Filled actual_close for {n_evaluated} predictions")

        # Step 2: Run each strategy
        results: dict[str, int] = {}
        for name, strategy in self._registry:
            if strategy_name and name != strategy_name:
                continue
            count = self._evaluate_strategy(strategy, cache)
            results[name] = count

        # Step 3: Refresh vectorbt metrics for strategies with new evaluations
        for name, count in results.items():
            if count > 0:
                self._refresh_metrics(name, cache)

        return results

    def _evaluate_strategy(self, strategy: BaseStrategy, cache) -> int:
        """Run a single strategy against its unevaluated predictions."""
        pending = cache.get_unevaluated_predictions_for_strategy(strategy.name)
        if not pending:
            return 0

        if strategy.requires_context:
            return self._evaluate_with_context(strategy, pending, cache)
        return self._evaluate_individual(strategy, pending, cache)

    def _evaluate_individual(
        self, strategy: BaseStrategy, predictions: list[dict], cache
    ) -> int:
        """Evaluate predictions one at a time (most strategies)."""
        evaluations = []
        for pred in predictions:
            signal = strategy.evaluate(pred)
            ev = self._build_evaluation(pred, strategy, signal)
            evaluations.append(ev)

        return cache.store_strategy_evaluations(evaluations)

    def _evaluate_with_context(
        self, strategy: BaseStrategy, predictions: list[dict], cache
    ) -> int:
        """Evaluate with context (ensemble strategies).

        Groups predictions by (symbol, target_date) and passes the group
        as context. Evaluates once per group, storing one result per
        prediction in the group.
        """
        # Group by (symbol, target_date)
        groups: dict[tuple, list[dict]] = defaultdict(list)
        for pred in predictions:
            key = (pred["symbol"], str(pred["target_date"]))
            groups[key].append(pred)

        evaluations = []
        for (_symbol, _date), group in groups.items():
            # Evaluate once with full context
            signal = strategy.evaluate(group[0], context=group)

            # Store one evaluation per prediction in the group
            for pred in group:
                ev = self._build_evaluation(pred, strategy, signal)
                evaluations.append(ev)

        return cache.store_strategy_evaluations(evaluations)

    def _build_evaluation(
        self, prediction: dict, strategy: BaseStrategy, signal
    ) -> dict:
        """Build an evaluation dict for storage."""
        entry_price = prediction.get("previous_close")
        exit_price = prediction.get("actual_close")

        pnl = None
        was_correct = None
        if (
            signal.action in ("BUY", "SELL")
            and entry_price
            and exit_price
            and entry_price > 0
        ):
            pnl = compute_pnl(
                signal.action, entry_price, exit_price, signal.position_size
            )
            if signal.action == "BUY":
                was_correct = exit_price > entry_price
            elif signal.action == "SELL":
                was_correct = exit_price < entry_price

        return {
            "id": f"{prediction['id']}_{strategy.name}",
            "prediction_id": prediction["id"],
            "strategy_name": strategy.name,
            "strategy_version": strategy.version,
            "action": signal.action,
            "position_size": signal.position_size,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pnl_dollars": pnl,
            "was_correct": was_correct,
            "metadata": signal.metadata,
        }

    def _refresh_metrics(self, strategy_name: str, cache) -> None:
        """Recompute per-symbol metrics for a strategy.

        These are independent one-session directional calls, not a held
        position, so they are scored as a series of per-trade returns rather
        than fed to vectorbt as an equity curve. The previous version built a
        `close` series out of each trade's exit_price, took BUY as an entry
        and SELL as an exit of a long, and reported whatever came back. On
        disjoint one-day calls that is not an equity curve, and it produced
        results like Sharpe -11.18 on 2 trades and a symbol showing 20% win
        rate with Sharpe +1.72 and a positive return at the same time.

        Ratio metrics are withheld below MIN_TRADES_FOR_RATIOS: a Sharpe from
        a handful of trades is noise, and the UI omits a metric that is None
        rather than printing a number nobody should act on.
        """
        from config import STRATEGY

        symbols = cache.get_strategy_symbols(strategy_name)
        if not symbols:
            return

        try:
            import numpy as np
            import pandas as pd
        except ImportError:
            logger.warning("numpy/pandas unavailable, skipping metrics refresh")
            return

        min_ratio_n = getattr(STRATEGY, "MIN_TRADES_FOR_RATIOS", 30)

        for symbol in symbols:
            series = cache.get_evaluated_signal_series(strategy_name, symbol)
            if len(series) < STRATEGY.MIN_TRADES_FOR_METRICS:
                continue

            try:
                df = pd.DataFrame(series)
                entry = df["entry_price"].astype(float)
                exit_ = df["exit_price"].astype(float)
                side = np.where(df["action"].str.upper() == "SELL", -1.0, 1.0)
                ret = side * (exit_ - entry) / entry.replace(0, np.nan)
                # .to_numpy() matters: handing pandas a Series plus a new index
                # REINDEXES against it rather than relabelling, and the old
                # positional index matches no timestamp, so every value becomes
                # NaN and the whole series silently drops to empty.
                ret = pd.Series(ret.to_numpy(),
                                index=pd.to_datetime(df["target_date"])).dropna()
                if ret.empty:
                    continue

                # One equity curve from the day's mean return, so several calls
                # on the same date count once rather than compounding serially.
                daily = ret.groupby(ret.index).mean().sort_index()
                equity = (1.0 + daily).cumprod()
                drawdown = equity / equity.cummax() - 1.0

                n = int(len(ret))
                win_rate = float((ret > 0).mean() * 100.0)
                total_return = float((equity.iloc[-1] - 1.0) * 100.0)
                max_dd = float(drawdown.min() * 100.0)

                sharpe = sortino = None
                if n >= min_ratio_n and daily.std(ddof=1) > 0:
                    sharpe = float(daily.mean() / daily.std(ddof=1) * np.sqrt(252))
                    downside = daily[daily < 0]
                    if len(downside) > 1 and downside.std(ddof=1) > 0:
                        sortino = float(
                            daily.mean() / downside.std(ddof=1) * np.sqrt(252)
                        )

                # Tail and significance stats. Mean per-trade return this small
                # is dominated by its tail, so a hit rate on its own overstates
                # how much the series actually says.
                se = float(ret.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
                t_stat = float(ret.mean() / se) if se > 0 else None
                var95 = float(ret.quantile(0.05) * 100.0)
                tail = ret[ret <= ret.quantile(0.05)]
                cvar95 = float(tail.mean() * 100.0) if len(tail) else None

                cache.store_strategy_metrics(
                    strategy_name,
                    symbol,
                    "all",
                    {
                        "sharpe_ratio": _safe_float(sharpe),
                        "sortino_ratio": _safe_float(sortino),
                        "max_drawdown": _safe_float(max_dd),
                        "win_rate": _safe_float(win_rate),
                        "total_pnl": _safe_float(total_return),
                        "total_trades": n,
                        "mean_return_bp": _safe_float(ret.mean() * 1e4),
                        "t_stat": _safe_float(t_stat),
                        "var95_pct": _safe_float(var95),
                        "cvar95_pct": _safe_float(cvar95),
                        "skew": _safe_float(ret.skew()) if n > 2 else None,
                        "excess_kurtosis": _safe_float(ret.kurt()) if n > 3 else None,
                        "ratios_withheld_below_n": None if n >= min_ratio_n else min_ratio_n,
                    },
                )
            except Exception as e:
                logger.warning(
                    f"metrics computation failed for {strategy_name}/{symbol}: {e}"
                )

    def refresh_all_metrics(self) -> None:
        """Recompute vectorbt metrics for all strategies."""
        cache = get_cache()
        for name, _ in self._registry:
            self._refresh_metrics(name, cache)

    def backfill(self, strategy_name: Optional[str] = None) -> dict[str, int]:
        """Delete existing evaluations and re-run all strategies.

        Args:
            strategy_name: Backfill only this strategy. None = all.

        Returns:
            Dict mapping strategy_name to count of new evaluations.
        """
        cache = get_cache()

        # Delete existing evaluations
        deleted = cache.delete_strategy_evaluations(strategy_name)
        if deleted:
            logger.info(f"Deleted {deleted} existing evaluations for backfill")

        # Re-run evaluation
        return self.run_evaluation(strategy_name=strategy_name)


def _safe_float(value) -> Optional[float]:
    """Convert to float, returning None for NaN/±inf/None.

    Infinity must be caught here: vectorbt returns inf Sharpe/Sortino for a
    zero-variance series (e.g. one trade), json.dumps happily emits the
    non-standard token "Infinity", and Postgres rejects it — the whole
    strategy_metrics insert then fails.
    """
    if value is None:
        return None
    try:
        import math
        f = float(value)
        return round(f, 4) if math.isfinite(f) else None
    except (ValueError, TypeError):
        return None


def get_evaluation_service() -> EvaluationService:
    """Get the singleton evaluation service instance."""
    global _service_instance
    if _service_instance is None:
        _service_instance = EvaluationService()
    return _service_instance

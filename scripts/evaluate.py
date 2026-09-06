#!/usr/bin/env python3
"""Manual evaluation and backfill script.

Usage:
    python scripts/evaluate.py                          # Evaluate all pending
    python scripts/evaluate.py --strategy directional   # Single strategy only
    python scripts/evaluate.py --backfill               # Re-run all strategies
    python scripts/evaluate.py --refresh-metrics        # Recompute vectorbt stats
    python scripts/evaluate.py --list                   # List registered strategies
    python scripts/evaluate.py --rebase --dry-run       # Scored rows whose closes the
                                                        # cache has since rebased
    python scripts/evaluate.py --rebase                 # Re-score them, rebuild strategy
                                                        # evaluations and metrics
"""

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate model predictions")
    parser.add_argument("--strategy", help="Run specific strategy only")
    parser.add_argument(
        "--backfill",
        action="store_true",
        help="Delete existing evaluations and re-run all",
    )
    parser.add_argument(
        "--refresh-metrics",
        action="store_true",
        help="Recompute vectorbt metrics only (no new evaluations)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List registered strategies and exit",
    )
    parser.add_argument(
        "--rebase",
        action="store_true",
        help="Re-score predictions whose entry/exit closes the price cache "
             "has since rebased (dividend or split across the row); checks "
             "all history, not just recent rows",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="With --rebase: report the rows that would move, change nothing",
    )
    args = parser.parse_args()

    from services.evaluation_service import get_evaluation_service

    service = get_evaluation_service()

    if args.rebase:
        from services.cache_service import get_cache
        cache = get_cache()
        moved = cache.rebase_scored_predictions(since_days=None,
                                                dry_run=args.dry_run)
        verb = "would move" if args.dry_run else "moved"
        print(f"{len(moved)} scored prediction(s) {verb}")
        for m in moved:
            after = "" if args.dry_run else (
                f" -> correct={m['was_correct_after']} pnl={m['pnl_after']:+.2f}")
            print(f"  {m['symbol']:6} {m['model']:20} {m['decision']:4} "
                  f"{m['target_date']}  entry {m['previous_close']:.4f}"
                  f"->{m['entry']:.4f}  exit {m['actual_close']:.4f}"
                  f"->{m['exit']:.4f}  was correct={m['was_correct']} "
                  f"pnl={m['pnl'] if m['pnl'] is None else round(m['pnl'], 2)}"
                  f"{after}")
        if moved and not args.dry_run:
            dropped = cache.delete_strategy_evaluations_for([m["id"] for m in moved])
            print(f"Dropped {dropped} strategy evaluation(s); rebuilding...")
            results = service.run_evaluation()
            for name, count in results.items():
                print(f"  {name}: {count} evaluations rebuilt")
        return

    if args.list:
        for name, strategy in service._registry:
            print(f"  {name} v{strategy.version}"
                  f" (context={strategy.requires_context})")
        return

    if args.refresh_metrics:
        print("Refreshing vectorbt metrics...")
        service.refresh_all_metrics()
        print("Done.")
        return

    if args.backfill:
        print(f"Backfilling {'all strategies' if not args.strategy else args.strategy}...")
        results = service.backfill(strategy_name=args.strategy)
    else:
        results = service.run_evaluation(strategy_name=args.strategy)

    if not results:
        print("No strategies registered.")
        return

    total = 0
    for name, count in results.items():
        print(f"  {name}: {count} new evaluations")
        total += count
    print(f"Total: {total} evaluations")


if __name__ == "__main__":
    main()

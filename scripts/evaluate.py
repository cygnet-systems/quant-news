#!/usr/bin/env python3
"""Manual evaluation and backfill script.

Usage:
    python scripts/evaluate.py                          # Evaluate all pending
    python scripts/evaluate.py --strategy directional   # Single strategy only
    python scripts/evaluate.py --backfill               # Re-run all strategies
    python scripts/evaluate.py --refresh-metrics        # Recompute vectorbt stats
    python scripts/evaluate.py --list                   # List registered strategies
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
    args = parser.parse_args()

    from services.evaluation_service import get_evaluation_service

    service = get_evaluation_service()

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

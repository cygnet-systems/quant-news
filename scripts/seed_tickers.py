#!/usr/bin/env python3
"""Fill the tickers lookup cache: index constituents plus every symbol the
platform has already run.

Usage:
    python scripts/seed_tickers.py              # both seeders
    python scripts/seed_tickers.py --no-indexes # history only (offline)

The scheduled weekly job (kind ticker_refresh) runs the same two seeders;
this is the hand-run for a fresh database or a machine without the job.
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402,F401  (load_dotenv side effect: DB URL)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-indexes", action="store_true",
                        help="skip the S&P 500 / Russell 2000 fetch")
    parser.add_argument("-q", "--quiet", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO,
                        format="%(levelname)-8s %(name)s: %(message)s")

    from sqlalchemy import func, select

    from db.models import Ticker
    from db.session import get_session
    from services import ticker_service

    indexes = 0 if args.no_indexes else ticker_service.seed_from_indexes()
    history = ticker_service.seed_from_history()
    with get_session() as session:
        total = session.execute(
            select(func.count()).select_from(Ticker)).scalar()
    print(f"indexes: {indexes} symbols written")
    print(f"history: {history} symbols written")
    print(f"tickers table: {total} rows")
    if not args.no_indexes and indexes == 0:
        print("index lists could not be fetched (offline?); "
              "history rows were still written", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())

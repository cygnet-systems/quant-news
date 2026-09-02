#!/usr/bin/env python3
"""Scheduled Full Analysis / evaluation for the quant-news dashboard.

Runs the same pipeline as the dashboard's Full Analysis button with no
browser attached, writing as the public user (anonymous, owner_uid NULL,
is_public true), so everything shows up in History, the Scoreboard and the
Activity Log exactly like a UI run.

Usage:
    python scripts/daily_analysis.py analyze                # next session, default watchlist
    python scripts/daily_analysis.py analyze --symbols PANW --target 2026-08-03
    python scripts/daily_analysis.py evaluate               # score everything that has closed

Exit codes: 0 on success, 1 when a stage produced nothing usable.
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Imported for its load_dotenv() side effect: the analyze/evaluate paths pull
# config in deep inside the pipeline, but `cost` and `notify-test` read
# os.environ directly and would otherwise see an empty local environment and
# report every variable as missing.
import config  # noqa: E402,F401

# Weights are cached; a scheduled run must not stall on a HF network call.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

# The 20-symbol watchlist this routine was set up for.
DEFAULT_SYMBOLS = [
    "PANW", "BAC", "VZ", "HWM", "DOC", "HPQ", "LUV", "TPL", "MPWR", "MCD",
    "ROP", "ETR", "CMS", "XYZ", "HIG", "IP", "FLEX", "MET", "FIS", "TYL",
]


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )
    logging.getLogger("services").setLevel(logging.INFO)
    logging.getLogger("models").setLevel(logging.INFO)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command",
                        choices=["analyze", "evaluate", "replay", "alpha-lab",
                                 "cost", "notify-test"])
    parser.add_argument("--event-move", type=float, default=5.0,
                        help="alpha-lab: one-day move (%%) that defines an "
                             "event (default 5)")
    parser.add_argument("--gate-p", type=float, default=0.55,
                        help="alpha-lab: calibrated-probability trade gate "
                             "(default 0.55)")
    parser.add_argument("--top-frac", type=float, default=6.0,
                        help="alpha-lab: rank basket = n_symbols/this "
                             "(default 6 -> top/bottom sixth)")
    parser.add_argument("--symbols", help="Comma-separated (default: watchlist)")
    parser.add_argument("--target", help="Target session YYYY-MM-DD "
                                         "(default: next unresolved session)")
    parser.add_argument("--lookback", type=int,
                        default=config.MODEL.NEWS_LOOKBACK_DAYS,
                        help=f"News window in days (default: "
                             f"{config.MODEL.NEWS_LOOKBACK_DAYS})")
    parser.add_argument("--max-articles", type=int, default=None,
                        help="Per-symbol cap: keep the newest N articles of "
                             "the window, 0 = all (default: config "
                             "NEWS_MAX_ARTICLES)")
    parser.add_argument("--report-model", default=None,
                        help="Research/report model "
                             "(default: config REPORT_MODEL)")
    parser.add_argument("--recs-model", default=None,
                        help="Synthesis model (default: config RECOMMENDATIONS_MODEL)")
    parser.add_argument("--news-filter", default=None,
                        choices=["lookback", "overnight"],
                        help="News window formula: 'lookback' (past N days) or "
                             "'overnight' (anchor close 16:00 ET to target "
                             "open 09:30 ET, relevance >= "
                             "NEWS_OVERNIGHT_RELEVANCE). Default: config "
                             "NEWS_FILTER_MODE")
    parser.add_argument("--models", default=None,
                        help="Comma-separated model ids to run (default: all)")
    parser.add_argument("--depth", default="thesis", choices=["thesis", "standard"],
                        help="Report depth: thesis (company thesis) or standard")
    parser.add_argument("--recs", default="auto", choices=["auto", "signals", "off"],
                        help="Recommendation basis: auto (text analysis + "
                             "predictions), signals (predictions only), off")
    parser.add_argument("--evidence", default=None,
                        help="Comma-separated evidence blocks for the research "
                             "prompt (options,quality,investigation,political); "
                             "'none' for no blocks. Default: config DEFAULT_EVIDENCE")
    parser.add_argument("--ensemble-json", default=None,
                        help="Ensemble config as JSON: {method, min_agree, "
                             "enabled_models, weights}. Default: config")
    parser.add_argument("--no-ensemble", action="store_true",
                        help="Skip the ensemble model")
    parser.add_argument("--tools", default="",
                        help="Comma-separated run tools, e.g. web_research "
                             "(lets the investigation search the open web). "
                             "Default: none: the scheduled job's form and the "
                             "Run dialog switch this on for next-day runs")
    parser.add_argument("--force", action="store_true",
                        help="Re-run symbols even when an identical analysis "
                             "for this cutoff is already stored")
    parser.add_argument("--days", type=int, default=7,
                        help="cost: lookback window in days (default: 7)")
    parser.add_argument("--only-trading-days", action="store_true",
                        help="No-op when today is not an NYSE session. The "
                             "scheduled job uses this so a market holiday "
                             "doesn't predict tomorrow's close twice.")
    parser.add_argument("--json", action="store_true", help="Print the summary as JSON")
    parser.add_argument("-q", "--quiet", action="store_true", help="Warnings only")
    args = parser.parse_args()

    _configure_logging(not args.quiet)

    from services.analysis_runner import evaluate_pending, run_full_analysis
    from utils.trading_calendar import is_trading_day

    if args.only_trading_days and not is_trading_day(datetime.now().date()):
        today = datetime.now().date()
        # Says so in the summary as well as in prose: the caller mails on what
        # it can parse, and an unparseable no-op reads as a run that produced
        # nothing: which is the same shape as a real failure.
        if args.json:
            print(json.dumps({"no_session": True, "date": today.isoformat()}))
        else:
            print(f"{today} is not an NYSE session, nothing to do.")
        return 0

    if args.command == "evaluate":
        count = evaluate_pending()
        from services.cache_service import get_cache
        backlog = get_cache().evaluation_backlog()
        summary = {"evaluated": count, "backlog": backlog,
                   "at": datetime.now().isoformat()}
        print(json.dumps(summary) if args.json
              else f"Evaluated {count} prediction(s); "
                   f"{backlog['pending_mature']} mature prediction(s) still "
                   f"unscored; {backlog.get('unscorable', 0)} can never score "
                   f"(no usable previous close)")
        # A clean run leaves no mature prediction unscored. A remaining
        # backlog is the "evaluated: 0 looked normal" failure shape, exit 2
        # so the scheduler records it as partial and mails accordingly.
        return 2 if backlog["pending_mature"] else 0

    if args.command == "alpha-lab":
        from services.alpha_lab import run_all
        results = run_all(move_pct=args.event_move, gate=args.gate_p,
                          top_frac=args.top_frac)
        print(json.dumps(results, indent=None if args.json else 2,
                         default=str))
        # Exit 0 either way, "no edge found" is a successful experiment.
        return 0

    if args.command == "replay":
        # Reuses the standalone replay script's engine so the scheduled run
        # and a hand run can never disagree about the methodology.
        from scripts.replay_ensemble_methods import (
            hold_bands, load_cohorts, replay,
        )
        cohorts = load_cohorts(args.target)
        if not cohorts:
            summary = {"replayed": 0, "note": "no member votes stored yet"}
        else:
            outcome = replay(cohorts, hold_bands(cohorts),
                             min_agree=2)
            best = max(outcome.items(),
                       key=lambda kv: kv[1].get("net_pnl",
                                                kv[1].get("pnl", 0)))
            summary = {"replayed": sum(len(v) for v in cohorts.values()),
                       "methods": outcome, "best_method": best[0]}
        print(json.dumps(summary, indent=None if args.json else 2,
                         default=str))
        return 0

    if args.command == "notify-test":
        from services import notify_service

        # Names only: never echo the values. This is meant to be run in a
        # deployment console, where the output is not necessarily private.
        present = [k for k in ("AZURE_TENANT_ID", "AZURE_CLIENT_ID",
                               "AZURE_CLIENT_SECRET", "NOTIFY_FROM_EMAIL",
                               "NOTIFY_TO_EMAIL") if os.environ.get(k)]
        missing = [k for k in ("AZURE_TENANT_ID", "AZURE_CLIENT_ID",
                               "AZURE_CLIENT_SECRET", "NOTIFY_FROM_EMAIL",
                               "NOTIFY_TO_EMAIL") if not os.environ.get(k)]
        print(f"Set:     {', '.join(present) or 'none'}")
        print(f"Missing: {', '.join(missing) or 'none'}")
        # The sender and recipients are not secrets and are the two values
        # most likely to be wrong, so show them.
        print(f"From:    {os.environ.get('NOTIFY_FROM_EMAIL', ', ')}")
        print(f"To:      {os.environ.get('NOTIFY_TO_EMAIL', ', ')}")
        if missing:
            print("\nNotifications are OFF until every variable is set.")
            return 1

        ok = notify_service.send_test()
        # The failure reason is logged by the sender, which knows the status
        # code and can name the specific fix; repeating a guess here would
        # only compete with it.
        print("\nSent." if ok else "\nSend FAILED, see the logged reason above.")
        return 0 if ok else 1

    if args.command == "cost":
        from config import LLM_PRICING_VERIFIED_ON
        from services.usage_service import summarize

        rows = summarize(days=args.days)
        if args.json:
            print(json.dumps(rows, indent=2, default=str))
            return 0
        if not rows:
            print(f"No LLM calls recorded in the last {args.days} days.")
            return 0
        print(f"{'DAY':11}{'STAGE':16}{'MODEL':18}{'CALLS':>6}"
              f"{'IN':>10}{'OUT':>9}{'COST':>10}")
        total = 0.0
        unpriced = 0
        for r in rows:
            cost = r["cost"] or 0.0
            total += cost
            unpriced += r["unpriced"] or 0
            print(f"{str(r['day']):11}{r['stage'][:15]:16}{(r['model'] or '?')[:17]:18}"
                  f"{r['calls']:>6}{r['in_tok'] or 0:>10,}{r['out_tok'] or 0:>9,}"
                  f"{cost:>10.4f}")
        print(f"\nTotal: ${total:.4f} over {args.days} days "
              f"(rates last verified {LLM_PRICING_VERIFIED_ON})")
        if unpriced:
            print(f"WARNING: {unpriced} call(s) had no price for their model, "
                  f"tokens counted, cost excluded. Add them to LLM_PRICING.")
        return 0

    symbols = ([s.strip().upper() for s in args.symbols.split(",") if s.strip()]
               if args.symbols else DEFAULT_SYMBOLS)
    models = ({m.strip() for m in args.models.split(",") if m.strip()}
              if args.models else None)

    evidence = None
    if args.evidence is not None:
        evidence = [e.strip() for e in args.evidence.split(",")
                    if e.strip() and e.strip().lower() != "none"]
    ensemble_config = json.loads(args.ensemble_json) if args.ensemble_json else None

    summary = run_full_analysis(
        symbols,
        target=args.target,
        lookback_days=args.lookback,
        max_articles=args.max_articles,
        report_model=args.report_model,
        recs_model=args.recs_model,
        include_thesis=args.depth != "standard",
        models=models,
        force=args.force,
        news_filter=args.news_filter,
        evidence=evidence,
        tools=[t.strip() for t in args.tools.split(",") if t.strip()],
        recs_mode=args.recs,
        ensemble_config=ensemble_config,
        run_ensemble=not args.no_ensemble,
    )

    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        if summary.get("error"):
            print(f"Failed: {summary['error']}")
        else:
            print(f"Target {summary['target_date']} (data through {summary['as_of']}): "
                  f"{summary['predictions_stored']} predictions stored in "
                  f"{summary['duration_s']}s")
            if summary.get("skipped"):
                print(f"Skipped (no price data): {', '.join(summary['skipped'])}")
            for sym, action in (summary.get("actions") or {}).items():
                print(f"  {sym}: {action}")
            for reason in summary.get("degraded") or []:
                print(f"  PARTIAL: {reason}")

    # Three outcomes, not two. A run that stored predictions but lost whole
    # models, or produced no synthesis, is neither a success nor a failure. 
    # collapsing it into either one is what hid a two-model outage for a full
    # cycle. 2 is the caller's cue to record it as partial.
    if summary.get("error") or not summary.get("predictions_stored"):
        return 1
    return 2 if summary.get("degraded") else 0


if __name__ == "__main__":
    sys.exit(main())

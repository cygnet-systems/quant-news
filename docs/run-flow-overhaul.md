# Run flow overhaul

What changed between `c4c72f0` and the end of the `run-flow-overhaul` branch,
and the rules a maintainer has to keep. Written 2026-09-04, at alembic head
**017**.

Before this work a run existed only as free-text lines on one global diskcache
channel. There was no row to key a run page, a per-owner lock, a cancel or a
completion link on; two users running at once shared one `run_id`; the "run
started" toast wrote into an id that was never mounted, so it had never once
rendered.

## Phases

| # | What it did |
|---|---|
| 0 | `analysis_runs` row per run, per-run progress feed, explicit `run_id` threading, per-owner run lock |
| 0b | Stages dispatched from a memory store; scheduled runs never lock the dialog; never-started runs reaped; lock-checked retry |
| 1 | Run dialog opens empty with presets, additive symbol sources, ticker typeahead; the run row's `config_json` is the source of truth |
| 2 | Topbar run pill with stage counters and ETA, stepper progress panel, completion toast, poll tied to active runs |
| 3 | `/runs/<id>` page with shared board rows; Runs section on Reports |
| 4 | Home splits into Scheduled and This session tabs; call-vs-actual rows |
| 4b | Default preset restores research + synthesis; the track record is scheduled-only |
| 5a | Alpha Vantage intelligence layer: insider transactions, congressional dossier, politician roster |
| 5b | Anomaly-driven dynamic reports with targeted web research |
| 6 | Defect closure, then polish: Analyze-now shortcut, cross-process web-research semaphore, measured duration estimates |

## New tables

| Migration | Table | One line |
|---|---|---|
| 014 | `analysis_runs` | One row per run, manual or scheduled: status lifecycle, symbols, config, per-stage progress and counters as JSON, `job_run_id` linking a scheduled run to its `job_runs` row |
| 015 | `tickers` | Local symbol lookup behind the dialog typeahead: index constituents, every symbol ever run, and names a single price lookup validated |
| 016 | — | Installs the weekly `ticker_refresh` job on databases that already have a schedule (the seed only writes into an empty `scheduled_jobs`) |
| 017 | `congress_trades` | STOCK Act disclosures per symbol, with `visible_from` |
| 017 | `insider_transactions` | Form 4 lines per symbol, with `visible_from` |
| 017 | `politicians`, `politician_aliases` | The Congress roster and the name forms it can be matched by |
| 017 | `av_fetch_log` | Per (function, subject) freshness: last attempt, last success, rows seen, why |

017 also installs the weekly `av_refresh` job under 016's guards. All four
migrations are reversible; verified 2026-09-04 on a throwaway database by
`upgrade head` → `downgrade -1` → `upgrade head` for 017, and `downgrade 013`
→ `upgrade head` for the whole span.

## New services and layout modules

- `services/run_service.py` — the run record: create, status, per-stage
  progress, the per-owner active-run lookup, orphan reaping, and
  `median_duration_s` (measured ETA from finished runs of the same band).
- `services/ticker_service.py` — local symbol lookup: search, validate,
  index seeding, the weekly refresh.
- `services/alpha_vantage.py` — one door to Alpha Vantage: quota, key, and
  the HTTP-200 error bodies that are really failures.
- `services/av_store.py` — stored AV intelligence: the syncs, the
  point-in-time readers, freshness bookkeeping, the weekly job body.
- `services/insider_service.py` — the Form 4 evidence block.
- `services/politician_dossier.py` — who in office is trading the name, and
  who in this run's news is one of them.
- `services/anomaly_service.py` — pure arithmetic over the gathered evidence:
  what is unusual about this symbol today, plus `screened()` and
  `format_anomaly_block()`.
- `layouts/run_pill.py` — the topbar pill: one run, one line, on every route.
- `layouts/progress_panel.py` — the activity panel body: a stepper over one
  run with the log beneath it.
- `layouts/pages/run.py` — the `/runs/<id>` page.

`services/progress_service.py` was rewritten from one global channel to
per-run keys (`_events_key`, `_meta_key`, `_pid_key` …) with an active-run
set, so two owners no longer interleave or reset each other.

## Adding a new evidence block

A block is a piece of prompt context with a name. To add one, touch these,
in this order — skipping any one of them fails quietly rather than loudly:

1. `services/evidence_contract.py` — add the key to `BLOCK_SEVERITY`
   (`REQUIRED` / `EXPECTED` / `OPTIONAL`) and `BLOCK_LABELS`. Severity decides
   what a missing block does: refuse the report, mark the run PARTIAL, or log.
2. `layouts/modals.py` — an `EVIDENCE_OPTIONS` entry. The Deep preset derives
   its evidence list from this, so it picks the block up automatically; the
   Schedule modal shares the same controls through `run_field_defaults()`.
3. `layouts/model_info.py` — an `_EVIDENCE_BLOCK_ROWS` explainer, so the block
   is documented where users read about it.
4. `models/trading_agents_model.py::_build_extra_context` — the branch that
   builds it. **Call `ledger.have(key)` whenever a block is written**, even
   when the source was partly stale. A block that rendered is present
   evidence; only an empty block is a gap.
5. `models/trading_agents_model.py::_scan_and_research` — add the key to the
   `looked` map so `anomaly_service.screened()` can say the category was
   examined. `screened` reads gathered-ness from `ledger.present`, never from
   row counts.
6. `services/anomaly_service.py` — optional. A detector's `evidence_key` must
   be a real `BLOCK_SEVERITY` key, and its `gathered` input keyword must
   appear in `SCREENS`.
7. `config.py` — add to `MODEL.DEFAULT_EVIDENCE` only if the block should be
   on for every run.
8. `services/analysis_runner.py` — bump `PIPELINE_EPOCH`.

Cost rule for anything vendor-backed: call `av_store.ensure_fresh(...)` and
then the point-in-time reader. Never `sync_*` per run.

## Point-in-time rules

Both AV tables carry a `visible_from` column and every reader filters
`visible_from <= as_of`. There is deliberately no unfiltered read helper, and
a missing or unparseable `as_of` raises rather than defaulting to today.

**Congressional trades.** `visible_from` is the filing date, or the
notification date when the vendor has no filing date, or NULL when it has
neither. `av_store.congress_visible_from` is the single definition of
"public" for both congressional readers. A NULL row is stored and never
served: `NULL <= as_of` is never true. That is why the column is nullable
even though the correctness property wants it filled.

**Insider Form 4s.** The payload carries no filing date, so `visible_from` is
the second trading day strictly after the transaction — the SEC filing
deadline, a documented **proxy**, not an observation. It must be described as
a proxy anywhere it reaches a prompt or a report. An insider who filed on day
one is treated as invisible until day two, and a row with no parseable
transaction date is dropped at ingest instead of stored.

The insider block header carries at most one date and never one later than
the run's `as_of`: `store synced <date>` when the sync itself is not in the
future of the cutoff, else `newest row visible from <date>`, else no date.

## Traps

**Alias fabrication.** `av_store.matchable_alias` requires two tokens of at
least two characters before an alias may be matched against article prose.
The roster's 1,144 members contribute 69 aliases starting with the bare
article "a" ("a green", "a gray", "a king"); scanned over prose, "regulators
gave the deal a green light" would name a sitting member as appearing in the
story and the prompt would then assert it as fact. Bare surnames are dropped
for the milder version of the same reason. `politician_by_alias` deliberately
does *not* apply the rule (a filing row handing over "A. Green" must still
resolve) and returns None when an alias matches more than one member.
Relatedly: the dossier's cross-symbol list covers only synced symbols, so the
prompt is told never to write it as "traded nothing else".

**Manual vs scheduled prediction ids.** `cache_service._prediction_id` gives
manual runs their own id space: `SYMBOL_model_YYYYMMDD_<run_id>` for manual,
the historic `SYMBOL_model_YYYYMMDD` for scheduled. Before the split, an
ad-hoc run of a watchlist name landed on the same key as the daily job, the
merge re-stamped the row's `run_id` as manual, the Scheduled board then read
"not run" for a day the job had called, and the merge's score reset discarded
that day's evaluation. Consequence: manual duplicates now exist as separate
rows, so **anything that aggregates predictions into a track record must pass
`kind="scheduled"`** (ORM: `cache_service._by_run_kind`) or `AND
SCHEDULED_ONLY_SQL` (text SQL). The two must keep agreeing that a row with no
run counts as scheduled. That is why Performance defaults to scheduled-only
(`history-run-kind` store, default `"scheduled"`) and why
`get_unevaluated_predictions_for_strategy` and every read in
`calibration_service` are scheduled-only: otherwise each rerun of a symbol
becomes another trade in the win rate and Sharpe.

**Preset rule.** The default preset may never switch off either of the
product's two outputs. Standard shipped once with `recs: "off"`, and because
the Schedule modal carries the dialog's values into `params_json`, every
default run *and* every schedule saved through the dialog produced a research
report with no synthesis. The test
`test_the_default_preset_produces_both_primary_outputs` in
`tests/test_run_dialog.py` pins `scope == "full"`, `trading_agents` in the
model list, and `recs == "auto"`. Quick is the only preset allowed to drop them.
`preset_run_config()` must also stay byte-identical to what the dialog's
confirm records — a test asserts full dict equality, so a new key in the
confirm's `config` needs the same key in `run_field_defaults()`.

**PIPELINE_EPOCH.** `services/analysis_runner.py` — currently
`"2026-09-03.3"`. It is part of the prediction cache key: bump it whenever
what a run *sees* changes (a new evidence block, a prompt rewrite, a fixed
feature), or stored predictions from the old pipeline are silently reused and
the change never reaches a report. Bump it once per shipped change, not once
per commit: within one unshipped phase a second bump buys nothing.

**Every symbol that starts a run is priced first.** The locked ticker-lookup
decision is that an unknown name is accepted only after one price lookup
succeeds. The dialog does that in `_resolve_run_symbols` as chips are added;
the strip's Analyze-now shortcut does it in `analyze_typed_symbol` before
`_start_manual_run`, because `ticker_service.normalize_symbol` is a regex and
says only that the text *could* be a ticker. Skipping it costs twice: the run
fails late at the price fetch, and `_start_manual_run`'s `ensure_symbols` has
already written the typo into the typeahead cache for good. Any new surface
that starts a run owes the same gate.

**The open-web bound has to be cross-process.** A manual run's model stage is
a forked background-callback subprocess (dash's `DiskcacheManager` forks per
invocation), so a `threading.Semaphore` in the web process bounds one run's
own fan-out and nothing between two people running at once.
`rate_limiter.FileSemaphore` is the bound: one lock file per slot under
`RATE_LIMIT_STATE_DIR`, taken with a non-blocking `flock`, so the kernel frees
a slot when a killed worker's descriptor closes. `MODEL.WEB_RESEARCH_CONCURRENCY`
(default 6) is the count; below 1 disables it. Held only around calls that
actually search (`_investigate_uncached` with `web=True`, and `_research_one`),
never around the web-free triage, and never inside `_CACHE_LOCK`.

**The insider cluster reads one span, and it is the priced one.** Both the
window test and the money come from a person's PRICED rows
(`priced_disposed`, `first_priced_disposed`, `latest_priced_disposed`, and the
acquired equivalents, built identically by `insider_service.summarize_insiders`
and `anomaly_service._insider_people`). Asking the two questions of different
filings let a $9M sale from May be reported as a disposal inside the last 30
days because an unpriced grant landed in August. Unpriced rows — grants, gifts,
option exercises, which the vendor sends at share price 0 — are excluded from
the dates, the count and the total alike.

**Others worth knowing.**
- `Output()` has no `allow_optional` (only `Input`/`State` do). A second
  writer to an existing Output needs `allow_duplicate=True` and
  `prevent_initial_call=True`.
- Every string `Output` id must be mounted in some layout. Audited clean at
  the end of this work: 155 string Output ids in `app.py`, 0 unmounted.
- `_start_manual_run` in `app.py` is the only place a manual run takes the
  per-owner lock. A new run surface must go through it, not
  `run_service.create_run`, or it walks past the lock.
- `scheduler_service.run_job`'s finalize block is sacred: an exception there
  once caused 28 phantom reruns. Create and update run rows from the runner
  side.
- `models/trading_agents_model._build_extra_context` returns five values and
  `services/export_service.py` unpacks it inside a bare `except` that turns a
  shape error into the string "Precomputed blocks: unavailable". Grep both
  callers before changing that return.
- `anomaly_service.screened()` takes the gathered map positionally; the old
  `**inputs` keyword form is a TypeError now.
- The research budget is a budget on *searches*: cache hits do not spend it,
  and a claimed slot is refunded when the prefetch pool fills the cache first.

## Verifying a change to any of this

```
/opt/miniconda3/envs/quant-news/bin/python -m pytest -q \
  --ignore=tests/perf_benchmark.py --ignore=tests/platform_evaluation.py \
  -p no:cacheprovider
```

887 passing at the close of this work, plus three failures that predate it and
are not this branch's: `test_confidence_weighted_score` in
`test_ensemble_methods.py`, and two timezone tests in
`test_scheduler_run_bookkeeping.py`.

Importing `app` proves nothing about a callback body. Boot the app
(`PORT=8082 DEBUG=false python app.py`) and POST `/_dash-update-component`
against the real callback, or the route walk only proves the SPA shell was
served.

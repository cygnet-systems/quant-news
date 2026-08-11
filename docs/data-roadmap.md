# Data ingestion roadmap

Status of the information-source expansion decided 2026-08-11. Context: with
the architecture frozen (every added predictor tested to a coin flip), new
*information* is the only remaining lever. Each source below plugs into the
same evaluation machinery (point-in-time discipline, scoreboard, Alpha Lab
pre-registered criteria) and earns or loses its seat on evidence.

## Live

### 1. SEC EDGAR — 8-K events + Form 4 insider transactions
- **Collector:** `CygnetResearchTerminal/cron_jobs/jobs/edgar_events.py`
  (nightly; one daily-index fetch for the whole market + one XML fetch per
  Form 4; ≤8 req/s; requires `SEC_USER_AGENT` env in the cron environment).
- **Tables:** `edgar_events` (every 8-K / Form 4: accession, ticker,
  filed_date, url), `edgar_form4_tx` (parsed insider transactions:
  code P/S, shares, price, value, officer flag).
- **Consumer:** `quant-news/services/terminal_data.py` → `filings_block()`
  renders a point-in-time block (recent 8-Ks + 45-day insider buy/sell
  summary) into the research prompt (`PIPELINE_EPOCH 2026-08-11.2`).
- **Point-in-time rule:** filter on the SEC's own `filed_date` ≤ as-of.
- **v1.1 (not yet built):** 8-K item-code enrichment (one extra fetch per
  filing) so the block can say *what kind* of event (2.02 earnings, 5.02
  leadership departure, 1.01 material agreement…) instead of just "an 8-K
  was filed"; Form 4 10b5-1 plan flag to separate scheduled sales.

### 2. Finviz snapshot history as market context
- **Collector:** existing Terminal `finviz-snapshot` job (~11.5k names/day
  into `finviz_snapshots`, JSONB payload per row).
- **Consumer:** `terminal_data.market_context_block()` — universe breadth
  (% advancing, median relative volume) + the symbol's own screener row
  (RSI, rel volume, perf, short float, inst own) as of the latest snapshot
  ≤ as-of.
- **v1.1 (not yet built):** cross-sectional *features* from the same data
  (symbol's rel-volume percentile vs universe, sector breadth) for the ML
  feature builder — a FEATURE_VERSION bump; test as an Alpha Lab hypothesis
  before shipping the features into the ensemble.

## Deferred — revisit when budget/time allows

### 3. Earnings-call transcripts (paid, ~$30-70/mo — e.g. Financial Modeling Prep)
Why deferred: requires a paid API subscription; no free source with reliable
coverage + timestamps.
Integration sketch when revisited: fetch transcript within minutes of
publication; LLM summary of guidance deltas vs prior quarter (same
digest pattern as the research prompt); event-gated — only fetched for
symbols with earnings inside the hold window, so cost scales with events,
not watchlist size. Pairs with the 5-10 day event-drift hypothesis: if
`event_drift` ever passes, transcripts are the first enrichment to test.

### 4. FRED macro calendar (free)
Why deferred: pure context, no direct signal claim — lower priority than
catalyst-bearing sources.
Integration sketch: pull the release calendar (FOMC, CPI, NFP dates);
stamp `regime.macro_event_today` on predictions made into a release
session; Alpha Lab slice: hit rates on macro days vs quiet days. If the
distributions differ, the schedule can learn to abstain (or size down)
into releases — same abstention machinery as the news gate.

## Rejected (tested or reasoned, 2026-08)
- **More daily-direction models** (CatBoost/TabNet/TSFM clones): correlated
  noise; every added voter tested to a coin flip.
- **Social sentiment** (Reddit/StockTwits): crowded, poor point-in-time
  hygiene, survivorship-heavy tooling.
- **13F cloning as a signal**: quarterly lag mismatches the platform's
  horizon (the Terminal keeps 13F for research context, not prediction).
- **Paid alt-data** (cards/satellite): capacity economics don't fit a
  20-name book.

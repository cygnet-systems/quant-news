# Quant News: presenter's briefing (2026-09-05)

Written from the code on branch `run-flow-overhaul` (working tree, alembic head 018)
and a read-only pull of the production database (Railway, `cygnet-systems/quantnews`,
live at quantnews.cygnetsystems.us). Production CSVs are in
`results/prod_export_2026-09-05/` (see its README for columns).

---

## 1. What it is, in one breath

Quant News is a daily pre-open research desk for a watchlist. Every weekday at 07:00 ET
it takes the previous session's close as the data cutoff, gathers point-in-time evidence
for 20 symbols, runs five independent models plus an ensemble, writes one LLM research
report per symbol, synthesizes everything into a BUY/SELL/HOLD action per symbol with a
scored probability, and mails the calls. At 18:00 ET it scores every call against the
real close. Every prompt, token, dollar and decision is stored, so the platform can
measure itself, and the honest headline of that measurement is that no arm has
demonstrated alpha at scale. The product's value is the evidence assembly, the
lookahead discipline and the audit trail, not a proven edge.

Stack: Dash 4 on FastAPI/uvicorn (single worker), Postgres via SQLAlchemy, MinIO/S3 for
report archives, APScheduler in-process, Kronos (torch), XGBoost, LightGBM, DeBERTa
(HuggingFace), OpenAI GPT-5.6 Luna for research/investigation/synthesis with Anthropic
fallback. Cygnet SSO shared with CygnetResearchTerminal.

---

## 2. One scheduled run, stage by stage

`services/analysis_runner.py::_run_stages` is the single pipeline; the Run dialog, the
CLI (`scripts/daily_analysis.py analyze`) and the scheduler all call it.

1. **Resolve dates.** Target = next trading session; cutoff (as-of) = last close.
   `utils/trading_calendar.py` (NYSE calendar).
2. **Market data.** 2y daily bars per symbol via yfinance, written to `stock_prices`
   (this is also what the evaluator scores against). Indicators computed (Wilder RSI/ATR,
   MACD, SMAs, Bollinger).
3. **News window.** One contract for every path (`services/news_window.py`):
   Alpha Vantage NEWS_SENTIMENT, paginated, half-open UTC window `[as_of - N days,
   as_of + 1 day)`, default 14 days, cap 50 newest articles, or an "overnight" mode
   (16:00 ET close to 09:30 ET open, relevance >= 0.7). Undated articles are dropped on
   historical windows. "Source down" and "quiet week" are tracked separately
   (`news_status` ok / empty / unavailable). The exact articles shown are archived as a
   `news_snapshot` report so any past call can be audited against its inputs.
4. **Model predictions** (`run_predictions`), in a fixed order because of an MPS/torch
   deadlock: Kronos first, then XGBoost, LightGBM, DeBERTa, the research agent
   (TradingAgents), then the ensemble over the five. A symbol already analysed for this
   cutoff against the same closing bar is reused, not re-run. Web research is stripped
   on the path if the target date is in the past (a backtest must not see the open web).
5. **Persist + auto-evaluate** predictions (`model_predictions`).
6. **Portfolio AI report** (`build_ai_report`): one LLM call over all symbols' news and
   validated metrics; per-symbol text analysis is the research report itself, merged in.
7. **Synthesis** (`run_recommendations`): one LLM call reads every report digest and every
   model's vote and returns the action per symbol. Stored as its own model
   (`recommendation_synthesis`) so it is scored like any other.
8. **Archive** the report JSON to object storage; assess completeness (PARTIAL if any
   expected evidence block was missing, a model failed, news was down, or the spend
   ceiling stopped purchases); mail the outcome.

Other scheduled jobs: `daily_evaluation` 18:00 ET, `alpha_lab` Mon/Wed/Fri 20:00 (three
pre-registered edge hypotheses re-tested against the growing history), `av_refresh`
Sunday 05:30 (congressional trades, Form 4s, Congress roster), `ticker_refresh` Sunday
06:00 (S&P 500 + Russell 2000 symbol cache).

Measured on production: a successful scheduled daily_analysis takes a median 48 minutes
(max 69) for 20 symbols, most of it serial LLM and vendor time.

---

## 3. Data ingest: every source and its point-in-time rule

| Evidence | Source | Point-in-time rule | Severity if missing |
|---|---|---|---|
| OHLCV bars, technicals, validated metrics (ATR, support/resistance, R:R, volume vs avg, drawdown) | yfinance | frame truncated to as-of; rejected if the last bar is >10 days stale | required |
| SPY regime (BULL/BEAR/MIXED by 50/200 SMA) | yfinance | truncated to as-of | required |
| News | Alpha Vantage NEWS_SENTIMENT (paginated) | half-open UTC window; undated dropped on historical runs | source failure is required (report refused); an empty window is fine |
| Business profile | yfinance info | static identity, allowed on historical runs | expected |
| Sector/industry ETF context and peers | `models/sector_map.py` from the symbol's own sector metadata, SPDR ETFs | truncated to as-of | expected |
| Fundamentals | yfinance statements | fiscal periods ending on or before as-of; snapshot fields labelled | expected |
| Event calendar (earnings, ex-div) | yfinance | inside the 1-5 day hold window gates conviction to <= 0.6 | expected |
| Options put/call, whole chain and by expiry | Alpha Vantage HISTORICAL_OPTIONS (chain as it stood that day) | same call live and in backtests; volume < 500 contracts flagged as noise | expected |
| Quality screen (Bad Apples, 20 pass/fail checks + news red flags) | ported from CygnetResearchTerminal, yfinance | histories truncated; snapshot checks labelled | expected |
| Situation & investigation | GPT-5.6 Luna with OpenAI hosted web search | live runs only; classifier runs web-free on backtests | expected |
| SEC filings (8-K, Form 4 index), finviz market breadth | CygnetResearchTerminal's historical Postgres (read-only bridge) | filtered on SEC filed_date / snapshot_date | optional |
| Congressional trades (STOCK Act), 13F holder flows | Alpha Vantage, stored weekly in `congress_trades` | visible_from = filing date (or notification date); NULL never served | optional |
| Insider Form 4 lines | Alpha Vantage INSIDER_TRANSACTIONS, stored weekly | visible_from = second trading day after the transaction (SEC deadline, stated as a proxy); $0 rows are grants/exercises, never "purchases" | optional |
| Congressional dossier (who the member is, seat, tenure, other trades, members named in this run's news) | stored roster + aliases (1,144 members) | alias must have two tokens >= 2 chars; ambiguous aliases never matched | optional |
| Prior report (continuity) and measured track record | own DB | prior report must predate this run's start; hit rate measured only through as-of | optional |

The severity classes (`services/evidence_contract.py`) decide behaviour: required missing
means the report is not written; expected missing means the report is written, the gap
is stated inside the prompt, the footer, the stored details and the run is marked
PARTIAL; optional is just logged. This exists because of the 2026-09-01 BHF report,
which read as complete while written from a fraction of the configured evidence.

---

## 4. The models

| Model | What it reads | How it decides |
|---|---|---|
| Kronos-mini | last 90 daily bars | time-series foundation model, 20 Monte Carlo samples of a 3-day path; direction from the median predicted close |
| XGBoost SHAP | 18 SHAP-selected features (price/indicator + AV news topic/sentiment features), walk-forward trained on the symbol's own history, min 30 samples | classifier, BUY >= 0.55 up-probability, SELL <= 0.45, else HOLD |
| LightGBM | same feature set | same thresholds |
| DeBERTa sentiment | AV headlines with ticker relevance >= 0.7 | `mrm8488/deberta-v3-ft-financial-news-sentiment-analysis`; abstains (HOLD, coverage not a call) when no relevant article |
| TradingAgents (research agent) | everything in section 3 | one long structured LLM report; the Verdict block is the scored call |
| Ensemble | the five votes | four selectable methods (confidence_weighted default, majority, prob_mean, agreement); equal weights; confidence = weighted mean up-probability |

The research agent's stored `confidence` is deliberately NOT the LLM's stated conviction.
It is a grounded number (measured hit rate once >= 8 resolved outcomes exist, otherwise a
0.5 placeholder labelled "no track record yet"). The stated conviction survives only as
metadata, because the platform measured that stated confidence carries no calibration
signal.

---

## 5. The research prompt and its dynamic behaviour

`models/single_agent.py`. Two messages.

**System message (static, symbol-free, ~4k tokens, byte-identical every symbol and day
so the provider's prefix cache serves it):**

- Role: decide at the close of the decision date whether the ticker moves UP or DOWN in
  the next session; thesis window 1-5 sessions; ignore long-term arguments.
- Grounding rules: every number verbatim from a data block or "not in data"; precomputed
  blocks are the source of truth; conflicting blocks are named, never averaged; no +/-
  on price levels; indicators describe what already happened and can never be the reason
  for a call on their own; group claims must hold for every named member; no browsing,
  the investigation block is the only web-derived input and is cited like news; every
  news claim carries outlet and date inline; never reference the instructions; the
  ALL-CAPS verdict fields are machine keys and never appear in prose.
- Reasoning order: Step 0 Situation (how PENDING_ACQUISITION, LEGAL_REGULATORY_OVERHANG,
  EARNINGS_EVENT, MOMENTUM_ONLY etc. change the reading of everything else), Step 1 SPY
  regime, 1b sector regime, 2 idiosyncratic technicals/fundamentals, 2b positioning and
  flows (options, Congress, insiders, 13F, short interest) with explicit anti-over-reading
  rules (disclosure lag, one member is noise, only clusters of distinct executives count,
  $0 rows are not purchases), 3 combine, 3b map 2-4 scenarios with probabilities, 4 gates
  (earnings in window caps conviction at 0.6; >20% drawdown needs a cause or "cause
  unknown: elevated risk").
- Decisiveness, what CONVICTION means (a judgement, not a measurement), levels are
  approximate and must be rounded to a tick and stamped "as of".
- Voice rules (no em dashes, no "not just X, it's Y", banned words, varied sentence
  length, no padding).
- Required output: the Verdict block first (FINAL TRANSACTION PROPOSAL, CONVICTION,
  MEASURED ACCURACY copied word for word from the user message, REASSESS_TO_BUY,
  MOVE_TO_SELL triggers, three bullets), then numbered sections each ending with a
  ticker-specific "**Read:**" line, then a mandatory fenced JSON epilogue (stance,
  sentiment_alignment, watch_items, scenarios, optional company_thesis).

**User message (per symbol, per day):** the ticker and date, a SITUATION line (what the
investigation classified it as, or an instruction to self-classify), the one literal
measured-accuracy sentence, then blocks: business profile, price action (15 bars),
technicals, fundamentals, news (up to 25 articles spread across the window's time
strata, not just the newest), SPY, sector ETF, then PRECOMPUTED METRICS & EVENTS
(anomaly blocks first, situation block second, then metrics, events, peers, SPY regime,
filings, options, quality, political, insiders, dossier, and an "Evidence NOT available"
list), and finally the numbered section list.

**What makes it dynamic:**

1. **The section list is built from what stood out.** The fixed frame is Situation &
   Key Figures, Technicals, News & Catalysts, Fundamentals, Positioning & Flows, Peer
   Comparison, Business Context, Market & Sector Backdrop (max three sentences), Bull vs
   Bear, Scenarios, Risk, Trade Plan. Each detected anomaly is inserted as its own
   numbered section right after Situation, with a three-part contract: (a) what the
   evidence shows, quoting only that block, (b) what the research found with source and
   date, or "not researched" and stop, (c) what it implies and what would prove it wrong.
2. **Quiet symbols get a short report, and the wording is honest about why.** Three
   distinct headers: nothing stood out in the categories that were screened (named, with
   the unscreened ones listed as "not looked at"), nothing was screened at all (so
   nothing is known), or the scan failed (so nothing is known and the report says so).
3. **The situation line changes how every block is read** (a cash-deal spread is the
   question, not the SMAs).
4. **Evidence gaps travel into the prompt**, so the model says what it could not see
   rather than reasoning as if absent inputs were neutral.
5. **The measured-accuracy sentence is computed, never phrased by the model**
   (`calibration_service.track_record_sentence`, scheduled-only rows, 90 days, minimum 20
   scored non-HOLD calls or it says "not enough history").
6. **Continuity is a post-pass.** The prior report is looked up but never shown to the
   research call (no anchoring). After the decision is extracted, a separate small call
   writes the "SINCE LAST REPORT" line comparing stance and whether prior triggers fired.
7. **Validation and retry.** If the Verdict anchors or the epilogue are missing the call
   is retried once; a stance that contradicts the Verdict is overridden and flagged.
   A deterministic figure audit (`utils/figure_check.py`) checks every cited number
   against the prompt; a grounded ratio under 0.8 emits an error into the run feed.
8. **Provenance footer** computed from what was actually assembled: bars, sources, news
   count and span, prior report date, situation, anomaly count and how many were
   web-researched, and "WRITTEN WITHOUT expected evidence" when degraded.

---

## 6. The investigation stage and the anomaly scan

**Anomaly scan (`services/anomaly_service.py`)** runs first because it is free
arithmetic over blocks already built, and it is the gate that decides whether a paid
web investigation is bought at all (`INVESTIGATE_ONLY_ANOMALIES`; on a 20-name list, 14
to 16 names were paying a search to learn nothing was happening). Six screens:

- options skew: put/call outside the 0.5-1.0 neutral band
- near-dated expiry diverging >= 50% from the whole chain (event hedging)
- insider cluster: >= 3 distinct executives on the same side, priced rows only, inside
  30 days, at >= 2x the symbol's own base rate (tuned on 104 symbol-days: 87 fired before
  the gates, 16 after)
- congressional activity, with extra weight for a second member or a trade against trend
- quality-screen failures (>= 5 fails, "bad apple" at 8)
- news spike: >= 4 articles at >= 2.5x the symbol's own daily norm
- plus positioning against the price trend (options + insiders vs a 20-day move >= 3%)

Severity 0-1 is an ordering key; at most 4 anomalies per symbol, each becomes a section
and a research question. A detector with missing inputs returns nothing (never a
neutral-looking filler).

**Investigation (`services/investigation_service.py`)**: one tool-using GPT-5.6 Luna call
per flagged symbol. System prompt: investigative equity analyst writing the situational
brief a desk reads before any chart; every finding sourced (outlet, date, URL); fact
separated from "inference:"; primary documents preferred; never invent prices or dates.
User prompt supplies profile, headlines, filings, quality screen, last close, and asks
for JSON: situation (one of nine labels), confidence, one_line, deal (acquirer, offer
price, consideration, approvals, break risk), key_figures (who decides the outcome and
their record), findings (6-10), dated_events, open_questions. The spread to a cash offer
is computed by code. Web-free triage runs first; a web-enabled call is bought only when
the name is not plain momentum. Then each anomaly question gets its own narrow search
call ("answer this one question, nothing else; 'no reporting found' is a valid answer")
with a run-level budget on searches and a floor of 2 searches per question.

Cost reality (measured 2026-09-04/05): web search is billed at $10 per 1,000 calls plus
about 8k input tokens of search content per search, so search is ~73-78% of a live run's
cost; prompt text and model choice are cents. `INVESTIGATION_MAX_SEARCHES` went 6 to 3
today; the ETR benchmark rerun came in at $0.139 vs $0.165 per symbol with identical
verdict and contract compliance.

---

## 7. What a typical report looks like (ETR, target 2026-09-08, run today)

Investigation: LEGAL_REGULATORY_OVERHANG, high confidence, 10 sourced findings (Justia
docket, Arkansas Advocate, Axios, Entergy filings), 4 key figures (the presiding federal
judge, the CEO, the Alphabet counterparty), 4 dated events (lawsuit filed 09-01, TRO
denied 09-02, PI schedule TBD, earnings 11-04), 6 open questions. Two anomalies fired:
the 09-18 expiry at 1.00 puts per call against 0.23 for the chain (335% above), and a
5.1x news-volume spike.

Report shape: Verdict (SELL, conviction 0.56, MEASURED ACCURACY quoted as "not enough
history, 2 scored non-HOLD calls in 90 days, 20 is the minimum", triggers at ~$110 and
~$104.25), then 13 sections: 1 Situation & Key Figures, 2 the expiry divergence anomaly,
3 the news spike anomaly, 4 Technicals, 5 News & Catalysts, 6 Fundamentals, 7
Positioning & Flows (named insiders, $2.78M disposed vs $1.23M acquired, institutional
adds), 8 Business Context, 9 Market & Sector Backdrop, 10 Bull vs Bear, 11 Scenarios
(45% headline-driven decline, 35% range, 20% upside), 12 Risk, 13 Trade Plan. 13 Read
lines, 70 cited figures all traceable to the prompt, zero em dashes, zero banned words.
The whole symbol cost $0.165 and 169 seconds; the six model votes split 2 BUY, 2 SELL, 2
HOLD, which is typical.

---

## 8. How the synthesis works

`services/llm_service.py::generate_recommendations`, one call per run over all symbols.

System prompt: "senior portfolio strategist" receiving two independent analyses (the
research reports and the quant votes). It is told, in plain terms, what each number is:
report conviction is self-assessed and unscored; the "track-record weight" is a measured
hit rate, and 0.50 means "no history yet", not a neutral view; "MEASURED ACCURACY" is the
only number checked against outcomes. It is also given the platform's own measured
findings as constraints: model agreement is not evidence (consensus days performed
WORSE, the models are correlated momentum readers), stated confidence has been
anti-calibrated, hit rates sit near 50% at scale. Role rules: disagreement between
report and models is the most valuable signal and must be explained; options and quality
lines shade conviction but never flip direction; a Situation line makes the call about
the situation, not the trend; a report written without expected evidence lowers
p_correct; scenario probabilities must be consistent with p_correct.

Per-symbol input block: research digest (verdict + the sections most relevant to the
decision, budgeted 1.5k-6k chars by symbol count), situation and deal terms, missing
evidence, the AI-report sentiment fields, options positioning and day-over-day P/C
shift, quality screen, then every model's vote with its number labelled (raw uncalibrated
score vs measured weight vs placeholder) and, where history exists, "calls at this score
have resolved X% correct".

Output JSON: `overall` (summary, portfolio_action, key_conflicts, risk_assessment,
watch_items) and `by_symbol` with action, **p_correct** (0.50-0.75 scale; replaced the
HIGH/MEDIUM/LOW labels because 4 weeks of data showed HIGH hit 60% and MEDIUM 64%),
reasoning, key_level, change_trigger, conflicts, model_notes. Each action is stored as a
`recommendation_synthesis` prediction with confidence = p_correct, so the synthesis is
scored exactly like a model. Identical evidence hashes to the same key, so a rerun on the
same inputs is served from cache. Fallback to Anthropic on empty/unparseable output.

---

## 9. How calls are scored

`cache_service.evaluate_predictions`, after each target session closes:

- BUY correct if target close > previous close; SELL correct if lower.
- HOLD correct if |move| <= the symbol's own band (median absolute daily return over
  bars strictly before the target session; the band and move are stored on the row).
  By construction about half of moves land inside the band, so a chronic holder scores
  near 50%; HOLDs are excluded from the "active" hit rates below.
- P&L: fixed $1,000 notional per active call, gross of costs.
- NaN/null prices are guarded; skipped rows are logged, never silently dropped.

---

## 10. Production results since collection began

Source: `model_predictions` on Railway, 4,126 rows, 57 symbols, 30 prediction dates,
2026-07-13 to 2026-09-03. Two eras are mixed in the table and must be separated when you
present:

- **2026-07-13 to 07-16, 31-34 symbols:** the lookahead-audited Russell 2000 diversity
  backtest, written into the production database. Not live calls.
- **2026-07-24 to 07-31:** early manual/prod runs on the first watchlist (8-20 symbols).
- **2026-08-03 onward, 20 symbols every weekday:** the scheduled live era (PANW, BAC, VZ,
  HWM, DOC, HPQ, LUV, TPL, MPWR, MCD, ROP, ETR, CMS, XYZ, HIG, IP, FLEX, MET, FIS, TYL).
  The research arm and synthesis have no rows after 09-01 (OpenAI credits ran out and the
  Anthropic fallback crashed on an unpinned SDK; both fixed on the branch, and the
  09-02..09-04 backfill loop produced only partial runs).

**All history (active = non-HOLD, scored):**

| Model | Rows | Active scored | Hit rate | z vs 50% | HOLD share | P&L | P&L/trade |
|---|---|---|---|---|---|---|---|
| kronos_mini | 637 | 508 | 50.6% | +0.27 | 20% | +$57 | +$0.11 |
| lightgbm | 637 | 558 | 50.7% | +0.34 | 12% | -$390 | -$0.70 |
| xgboost_shap | 637 | 521 | 50.1% | +0.04 | 18% | -$352 | -$0.68 |
| deberta_sentiment | 624 | 190 | 47.9% | -0.58 | 70% | +$47 | +$0.25 |
| trading_agents | 485 | 286 | 46.2% | -1.30 | 41% | -$241 | -$0.84 |
| recommendation_synthesis | 469 | 201 | 49.3% | -0.21 | 57% | -$241 | -$1.20 |
| ensemble | 637 | 518 | 48.8% | -0.53 | 19% | -$757 | -$1.46 |
| buy-and-hold, same symbol-days | 657 | 657 | 49.5% up-days | | | +$367 | +$0.56 |

**Scheduled live era only (2026-08-03 onward, 481 symbol-days, a soft tape: 47.8% up-days,
baseline -$247):**

| Model | Active scored | Hit rate | z | P&L |
|---|---|---|---|---|
| kronos_mini | 382 | 52.6% | +1.02 | +$154 |
| lightgbm | 423 | 49.2% | -0.34 | -$526 |
| xgboost_shap | 381 | 48.6% | -0.56 | -$389 |
| deberta_sentiment | 136 | 47.1% | -0.69 | -$85 |
| trading_agents | 253 | 45.1% | -1.57 | -$427 |
| recommendation_synthesis | 190 | 48.9% | -0.29 | -$331 |
| ensemble | 405 | 47.4% | -1.04 | -$780 |

How to say it: every arm is inside about 1.5 standard errors of a coin flip, in both
directions. Kronos's positive month is its structural SELL bias meeting a down-tilted
tape (the same pattern the June alpha verdict and the July R2000 test documented). The
ensemble is the worst dollar performer, which matches the measured finding that the
members are correlated momentum readers, so their consensus adds correlation, not
information. The synthesis, which is the product's headline output, is 49% on 201 active
calls and holds 57% of the time. Weekly hit rates swing between 16% and 69% for the same
model, which is what n≈20 a week looks like; do not present any single week.

**Calibration:** no model's stated confidence rises monotonically with realized hit
rate. XGBoost's 0.75-0.85 bucket resolved 45.6%; LightGBM's 0.75-0.85 bucket 48.7%; the
ensemble's >0.85 bucket 47.2%. The research agent's stated conviction was never used as a
score, so its rows sit at the 0.5 placeholder (280 of 286 active calls).

**Concentration:** P&L is a few names. Synthesis: PANW alone -$231 of -$241 (11 calls,
27%). Kronos: IP +$197 and XYZ +$172 (80% on 20 calls) against MPWR -$207 and FLEX -$190.
Ensemble: MPWR -$290, PANW -$260. Single-name luck dominates at this sample size.

**Caveats to state up front:** gross of transaction costs (a $0.11 to $1.46 per $1,000
edge is well inside spread and slippage); FEATURE_VERSION and prompt changes mean pre-
and post-08-06 tree models and pre- and post-09-02 reports are not the same system;
HOLD scoring was added in August; and the platform's own rule is not to trust anything
under ~150 samples per arm.

**Operations:** 242 job runs. Scheduled daily_analysis succeeded 18 times (median 48
min); 21 scheduled evaluations succeeded; 119 backfill "partial" and 30 backfill "error"
runs, most from the 09-02..09-04 loop that re-ran one session 109 times; 28 interrupted
catch-ups on 08-11/12 from the finalize KeyError. Alpha Lab has run 12 times, no
hypothesis has crossed its pre-registered bar.

**Cost:** $48.67 of priced LLM tokens over the whole history (8,360 calls; research
$30.75, synthesis $11.30, investigation $5.73, AI report $0.89), of which $10.48 is the
08-28 trade-date backfill loop (6,860 calls). A normal 20-symbol day before web research
was 22 calls at about $0.74 at the old rates (about $0.12 at today's Luna prices). A full
live day with web research measured about $1.14, three quarters of it search fees. Rows
from 09-02 onward show $0 because the served model was unpriced in the rate table at
the time; the ledger marks them as unpriced rather than free.

---

## 11. Pivotal changes since August (chronological)

**Aug 2, platform goes live.** V6 platform commit: five models, Postgres persistence,
Cygnet SSO, Railway deployment.

**Aug 5-7, it becomes a routine, not a demo.** In-app APScheduler with advisory lock,
catch-up and per-job registry; cost telemetry (`llm_usage` with rates copied onto each
row); mail on outcome; torch pin fix that had silently disabled Kronos and DeBERTa in
production; 07:00 ET default; a model abstention counts as coverage, not a partial run;
scoreboard keyed on the session predicted.

**Aug 6, pipeline audit fixes (commit fffc435).** News features were 0.0 at training
time (topics_json key bug, so the tree models were price-only); AV rate-limit responses
silently produced newsless training; prod ensemble was a two-model vote saturating at 1.0;
HOLD band had lookahead and was never persisted; evaluator skipped silently; missed days
unrecoverable. FEATURE_VERSION bump, paginated news, configurable overnight window.

**Aug 6, Reports UX overhaul.** Per-symbol research as the unit of the product; research
digest into synthesis; bull/bear section; richer fundamentals.

**Aug 10-11, grounding and honesty.** Confidences, P&L and report figures grounded in
evaluated history (the measured-accuracy line, grounded confidence for the research
arm); terminal evidence (options P/C, Bad Apples) as blocks; ensemble method selector;
Alpha Lab with pre-registered criteria; SEC filings and finviz context; the scheduler
finalize KeyError outage (28 phantom reruns) and its fix.

**Sep 1-2, every run inspectable and the BHF post-mortem.** `/trace` page, run
watchdog, ET timestamps; the investigation stage (situation classification + web
research) written after a BHF report gave a momentum SELL on a stock under a cash
takeover; the evidence contract (required/expected/optional, refuse to write without
required blocks); web research as a run tool that is never on for backtests; GPT-5.6
Luna as the research and investigation model; one news window contract; kill the whole
process group on job timeout; em dashes and AI-writing tells removed from prompts and
copy; continuity moved to a post-pass so the research call never sees the prior stance;
OOM memory levers after three container deaths.

**Sep 2-4, run flow overhaul (phases 0-6).** `analysis_runs` row per run with per-owner
lock and per-run progress; Run dialog with presets and ticker typeahead; topbar run pill,
stepper and completion toast; `/runs/<id>` page; Home split into Scheduled vs This
session with call-vs-actual rows; default preset restores research + synthesis; track
record is scheduled-only. Then the Alpha Vantage intelligence layer (insider
transactions, congressional dossier, politician roster, weekly refresh, point-in-time
visible_from) and anomaly-driven dynamic reports with targeted web research. Screening
honesty (a report never says "nothing stands out" about a category it did not fetch),
insider-cluster retune, load bounds, docs.

**Sep 4-5, incidents and cost.** Backfill loop (one session re-run 109 times because the
ledger counted only owner-stamped rows) fixed with a per-date attempt cap; OpenAI
credits exhausted and the Anthropic fallback died on an unpinned SDK (pinned, and
sampling parameters are dropped and re-asked rather than losing an answer); spend
ceiling per run; web search priced and prompt-cache hits measured; investigate only the
symbols something stood out for; research prompt split into a cache-stable system
message and a per-symbol user message (uncommitted in the working tree as of this
writing, benchmarked on ETR: -16% cost, identical verdict and contract compliance).

---

## 12. Questions you will get, and honest answers

- **Does it make money?** No demonstrated edge. Every arm is within noise of 50% on
  200-560 active calls each, gross of costs, and P&L is concentrated in one or two names.
  The platform says this about itself inside every report and inside the synthesis
  prompt.
- **Then what is it for?** A reproducible, lookahead-safe evidence desk with a full audit
  trail: every input archived, every figure checked against its prompt, every call
  scored, every dollar attributed. It is the measurement layer you need before you can
  claim alpha, and it has already killed several plausible hypotheses (news density,
  model consensus, conviction labels, options P/C as a timing signal, Bad Apples as a
  direction signal).
- **Why an LLM report at all if it is a coin flip?** The report's job is to state the
  situation, the dated catalysts, who is positioned which way and what would falsify the
  call, with sources. That is useful to a reader even when the direction is not
  predictable; and it is scored so the claim can be tested rather than assumed.
- **What would change the verdict?** Alpha Lab's pre-registered tests: a rank-spread
  |t| >= 2 over >= 40 days, event drift with >= 300 events, or a calibration gate that
  beats ungated by 3pp on >= 100 gated trades. None has passed.
- **Biggest lessons.** Short windows mislead (a 15-day run looked great and 200 days
  erased it); absence must never be asserted from evidence that was not fetched; a
  safety net that is never exercised is not a safety net; search fees, not tokens, are
  the bill.

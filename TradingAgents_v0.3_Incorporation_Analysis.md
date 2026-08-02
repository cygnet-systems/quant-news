# TradingAgents v0.3.x → quant-news: Incorporation Analysis

**Date:** 2026-07-20
**Scope:** Research the latest TradingAgents (fork `MUmarJ/TradingAgents`, `main` == upstream `TauricResearch` v0.3.1) and decide what to fold into quant-news — prompts, filter/lookahead logic, report generation, and the reflection loop. Ignore what doesn't fit our case.

**One-line verdict:** The real, defensible wins are **lookahead-safety and backtest validity**, **LLM-layer robustness**, and **explainability (reflection + report content)** — *not* alpha. Per our own ALPHA VERDICT and R2000 tests, the models are coin-flips; no prompt or debate topology changes that. And our TA integration is **currently broken**, which forces an architecture decision before anything else.

---

## 0. Headline: our TradingAgents integration is dead right now

- `/Users/UmarJahangir/Projects/TradingAgents` is our **fork**. `main` is now sitting exactly on upstream **v0.3.1** (0 ahead / 0 behind).
- `models/trading_agents_model.py` imports `from tradingagents.baselines.single_agent import SingleAgentBaseline`. That module **only exists on the fork's `feat/ace_alpaca` branch** (added in `bf77aa9`), not on `main`. Verified at runtime: `ModuleNotFoundError`. So `TRADING_AGENTS_AVAILABLE = False`, `is_ready()` → `False`, and **every TA prediction silently short-circuits to HOLD + error**.
- Even if we restored `baselines/`, the baseline was written against a **pre-v0.3.0 internal API** and references symbols upstream has since **moved/renamed/deleted**:
  - `tradingagents.llm.requires_responses_api` → gone
  - `tradingagents.agents.utils.agent_utils.normalize_content` + flat `get_*` tools → moved into `core_stock_tools` / `news_data_tools` / `technical_indicators_tools` / `fundamental_data_tools`
  - `tradingagents.graph.signal_processing.extract_decision_from_text` / `extract_confidence_from_text` → replaced by `SignalProcessor.process_signal()` (a thin `parse_rating` adapter)
  - Also needs `langgraph.checkpoint.sqlite`, not installed in the `quant-news` conda env.

**Implication:** "incorporate the latest changes" cannot be a `git pull`. TradingAgents' *internal* API churns every release, and we depend on a non-public submodule of it via an unpinned, un-vendored sibling checkout. **The strategic fix is to stop depending on TA internals** and lift the good, self-contained ideas into quant-news's own services (see §6, Tier 0).

---

## 1. What we actually use today (baseline)

We use **only** `SingleAgentBaseline` — the framework's *single-LLM-call baseline*, i.e. the thing TradingAgents' own paper is designed to *beat*. We do **not** use `TradingAgentsGraph.propagate`, the analyst team, the bull/bear debate, the risk debate, or reflection (grep for `TradingAgentsGraph`/`propagate`/`final_trade_decision` in quant-news = 0 hits).

Our `models/trading_agents_model.py`:
- Copies `DEFAULT_CONFIG`, sets `llm_provider=anthropic`, `deep_think_llm=claude-sonnet-4-6`, `data_vendors={core/tech/fundamentals: yfinance, news: alpha_vantage}`.
- Derives `trade_date` from `ohlcv_df.index[-1]` (ignores the `as_of` kwarg — works only because callers pre-truncate the frame).
- **Computes its own grounding blocks** (this is our bolt-on, not TA): `compute_trading_metrics` (ATR, realized vol, S/R, R:R, vol ratio, drawdown), `get_upcoming_events` (earnings/ex-div gate → "cap confidence at 0.6"), `compute_peer_relative_strength` (vs sector peers, `as_of`-truncated), and an inline SPY regime block (50/200 SMA, `spy[spy.index <= trade_date]`).
- Passes them as `extra_context` to `SingleAgentBaseline.analyze(...)`.
- Extracts `decision`/`confidence` from the returned dict; **fabricates `up_probability = 0.5 ± confidence*0.3`** (HOLD always exactly 0.5).

**The baseline's prompt (`SINGLE_AGENT_PROMPT`) is genuinely good — arguably better than upstream's per-analyst prompts for our 1–5 day horizon.** It already contains, natively:
- **Lookahead discipline in-prompt:** "Only consider information available BEFORE 9:30 AM ET on {date}… ignore anything published after {date} 09:30 ET."
- **DATA DISCIPLINE / anti-hallucination:** "Every price/indicator/statistic MUST come verbatim from the data blocks… If a value is not in the data, write 'not in data' instead of guessing."
- **Explicit regime → sector → idiosyncratic → combine decision framework**, event gate, drawdown-causality rule, "compute don't assert," seasonality check.
- **Falsification triggers**: `REASSESS_TO_BUY` / `MOVE_TO_SELL` — concrete, checkable conditions. (Upstream has nothing equivalent.)
- **Confidence semantics** defined as calibrated probability with a "prefer HOLD at 0.5" rule.

So the comparison is **not** "our thin prompt vs upstream's rich prompts." It's "our already-rich single-agent prompt vs upstream's specific, portable *techniques*." We should cherry-pick techniques, not wholesale-replace.

---

## 2. What's genuinely best in upstream v0.3.x (the capture)

### 2A. Filter / lookahead logic — the crown jewels (`tradingagents/dataflows/`)

These are the highest-value, most portable ideas, and they map directly onto bugs our own lookahead audits chase.

1. **Half-open UTC news window `[start, end+1day)` with default-exclude for undated items in historical runs** (`yfinance_news.py::_in_news_window`, `_as_utc`; fix `40774ca`, #1126):
   - Normalize *all* operands to UTC first. Naive → assume UTC (`.replace`); aware → convert (`.astimezone`). The old bug stripped tzinfo without converting → off-by-a-day drops.
   - Lower bound inclusive, **upper bound exclusive**: `start <= pub < end + 1 day`. Keeps all of `end` day (through 23:59:59) but rejects the article stamped exactly midnight-after.
   - **Undated article (`pub_date is None`) is kept only if the window reaches the present** (`end >= now - 1 day`). In a backtest you can't prove it isn't future news → **default to exclude**. This is the single most important anti-leak rule.
   - Flat yfinance articles carry epoch `providerPublishTime` — parse as `datetime.fromtimestamp(ts, tz=utc)`, not host-local, so they become filterable instead of silently bypassing the filter.

2. **Stale-OHLCV rejection** (`stockstats_utils.py::_assert_ohlcv_not_stale`, `MAX_OHLCV_STALE_DAYS=10`; fix `9fd54f8`, #1021): if the latest row is > 10 calendar days before the requested date, **raise** `NoMarketDataError("… stale — refusing to use it")` instead of feeding a year-old partial frame that passes the empty-check. Catches "present but wrong," which empty-checks miss.

3. **Same-day cache TTL** (`_needs_same_day_refresh`, `OHLCV_CACHE_TTL_SECONDS=900`; fix `d78c698`, #1150): historical dates are immutable → cache forever; the **current day's bar is provisional** (Yahoo publishes a running intraday candle whose Close ≠ the close) → refetch if the cache file's mtime > 15 min. Key insight: a *present* row can still be a partial candle you can't distinguish by inspection.

4. **Look-ahead slice + exclusive-end fix** (`load_ohlcv`): widen the fetch by +1 day (yfinance `end` is exclusive), then slice `data[data["Date"] <= curr_date]` *after*. General lesson: know your vendor's inclusive/exclusive end, widen, slice after.

5. **Fundamentals fiscal-date filter** (`alpha_vantage_fundamentals.py::_filter_reports_by_date`, fix `3570f2e`, #1115; and yfinance `filter_financials_by_date`): drop reports where `fiscalDateEnding > curr_date` (or DataFrame columns > cutoff). The fixed bug is instructive — the payload was a JSON *string* but the guard checked `isinstance(dict)`, so the filter never ran and future reports leaked. **Caveat:** it keys on *fiscal-period-end*, not *filing/publish* date, so a report can still surface a few days early vs. when it was actually public.

6. **Verified market snapshot** (`market_data_validator.py::build_verified_market_snapshot`): compute exact latest OHLCV-on-or-before-date + a fixed indicator set deterministically, re-apply the cutoff defensively, cap recent-closes at 30 rows, and instruct the LLM to treat it as source of truth and **flag conflicts rather than reconcile**. Anti-confabulation for LLMs-over-numbers.

7. **Behavior-based error taxonomy + single no-data sentinel** (`errors.py`): `VendorError` → `NoMarketDataError` / `VendorRateLimitError` / `VendorNotConfiguredError`. Empty and stale share one type (differ only in `detail`). The router returns an explicit `NO_DATA_AVAILABLE: … do not estimate or fabricate values` string to the model. Principle: *number of error types = number of distinct reactions, not causes.*

8. **Path-traversal guard** (`safe_ticker_component`) on any LLM/CLI-supplied ticker used in a cache filename.

### 2B. Prompt-craft techniques (`tradingagents/agents/`)

- **Schema-field-as-instruction** (`schemas.py`): put output rules (enum, "2–4 sentences", score↔band mapping) in Pydantic `Field(description=…)`, not the prompt body — provider-portable and DRY. They iterate on field descriptions the same way you'd iterate on a prompt (fix `7aef10a`).
- **`get_verified_market_snapshot` grounding** (§2A.6) enforced from the *prompt*: "treat as source of truth… flag the discrepancy rather than inventing a reconciled number… do not claim support/resistance bounces or exact % moves unless directly supported by tool output with concrete dates and prices." Best anti-hallucination move in the repo.
- **Sentiment redesign** (`sentiment_analyst.py`): stop tool-calling; **pre-fetch News + StockTwits + Reddit into the prompt** inside `<start_of_x>…<end_of_x>` delimiters (the old prompt "demanded social-media analysis but the only tool was Yahoo news → the LLM fabricated Reddit/X content under pressure"). Plus 8 numbered heuristics (bullish/bearish ratio thresholds, cross-source divergence, engagement-weighting, opinion-vs-event, "be honest about data limits → flag in confidence").
- **Anti-Hold-bias nudge** (`research_manager.py` / `portfolio_manager.py`): "Commit to a clear stance whenever the debate's strongest arguments warrant one; reserve Hold for genuinely balanced evidence."
- **`NO_EXTERNAL_TOOLS` constant** for schema-only agents (fix `030b434`, #1130): "Use only the evidence in this prompt. Do not call external tools… if something is missing, say so explicitly." Stops a primed model from emitting a stray `web_search` call that discards the structured attempt and costs a round-trip.
- **Date at top of prompt** (fix `2b2d685`): put "Today's date is {current_date}; treat it as 'now' for all analysis and tool-call date ranges" in the *header*, not buried after a long indicator block, or weak models anchor to their training cutoff.
- **Instrument-identity anchor** (`build_instrument_context`): "Use this exact ticker… Resolved identity: {name; sector; exchange}. Do not substitute a different company/ticker unless a tool result disproves it." Stops the wrong-company cascade.
- **Prompts as tested code**: `test_news_analyst_prompt.py`, `test_structured_agent_prompts.py` assert against the *rendered* prompt and the *real* tool signature. Rare and good.

### 2C. Structured output + deterministic parse (`schemas.py` / `structured.py` / `rating.py`)

- Decision agents use `with_structured_output(schema)` → render Pydantic back to a fixed markdown shape → `parse_rating()` reads the rating out with **zero extra LLM calls**.
- **Graceful fallback** (`invoke_structured_or_freetext`): structured call; if it returns `None` (a thinking model answered in prose) or throws (bad JSON, transient error) → one-shot plain `llm.invoke()`. Pipeline never blocks.
- `_coerce_optional_float` coerces `"", "n/a", "tbd", "unknown"` → `None` so a model writing "N/A" into a numeric field doesn't fail validation.
- **One centralized rating vocabulary** (`RATINGS_5_TIER`) shared by every producer/consumer → no regex drift. 5-tier Buy/Overweight/Hold/Underweight/Sell (conviction baked into the tier); Trader uses coarser 3-tier Buy/Hold/Sell.

### 2D. Deferred, alpha-grounded reflection loop (`graph/reflection.py`, `agents/utils/memory.py`) — the standout *capability*

Solves "you can't reflect on an outcome you don't have yet":
1. **Write (no LLM):** at run end, append the decision to a plain markdown log (`~/.tradingagents/memory/trading_memory.md`) as `pending`, tagged with the parsed rating. Atomic writes (`temp + os.replace`), idempotency guard, optional rotation of *resolved* entries.
2. **Resolve (deferred):** at the *start* of the next same-ticker run, fetch realized return + **alpha vs benchmark** (yfinance, 5-day hold, benchmark by exchange suffix: `.T`→^N225, default SPY), ask the LLM for a **2–4 sentence lesson** (was the directional call right — cite the alpha figure; which part of the thesis held/failed; one concrete lesson). Fail-open: missing price → stays pending, retries next run.
3. **Inject:** feed the last 5 same-ticker decisions (full) + 3 most-recent cross-ticker lessons (reflection-only) into the next Portfolio Manager prompt.
- **No embeddings / vector DB** — retrieval is exact-ticker + recency. Learning signal is realized alpha, not self-grading.

### 2E. Report tree (`tradingagents/reporting.py::write_report_tree`, fix `a0120e1`)

~90 lines, no deps, shared by CLI + API. Numbered subdirs of per-agent markdown + one consolidated `complete_report.md`:
```
1_analysts/{market,sentiment,news,fundamentals}.md
2_research/{bull,bear,manager}.md
3_trading/trader.md
4_risk/{aggressive,conservative,neutral}.md
5_portfolio/decision.md
complete_report.md   (header + roman-numeraled sections, each section guarded)
```

---

## 3. Side-by-side: upstream vs our case

| Area | Upstream v0.3.1 | quant-news today | Gap / fit |
|---|---|---|---|
| **News window** | Half-open UTC `[start, end+1)`, undated→exclude in backtest, epoch parsed as UTC | `app.py` string cutoff `published_at[:19] <= "{date}T23:59:59"` (naive, ISO-assumed); `news_service` default window is **rolling 7-day-from-*now***; historical cache wired only to XGB/LGBM | **Port upstream window.** Also fix the **coverage gap**: LLM/TA/AI-report backtests older than ~7d run on *empty* news → invalidates those backtests |
| **Fundamentals lookahead** | `fiscalDateEnding <= curr_date` | TA path pulls **today's** fundamentals for backtest dates (leak); our own layer doesn't feed fundamentals | Apply filter *if/when* we own the data layer (Tier 0-C) |
| **Stale OHLCV** | Reject latest row > 10d before requested | None (we truncate but don't detect stale/partial frames) | Cheap add; low risk |
| **Anti-hallucination** | `get_verified_market_snapshot` tool = source of truth; "flag conflicts, don't reconcile" | We inject validated metric blocks + strong "cite verbatim" rules already | We're ~80% there; add the "flag conflicts, don't reconcile / no claimed bounces" clause |
| **Structured output** | `with_structured_output` + render-back + one-shot free-text fallback + centralized rating vocab | Hand-rolled 3-layer regex JSON parse + deterministic fallback (already decent); Luna had empty-content-on-200 failures | Adopt the *fallback discipline*; our regex parse is fine, but `max_completion_tokens` headroom + one-shot retry is the same idea we already half-have |
| **Reflection/memory** | Deferred alpha-grounded lesson loop, re-injected into next prompt | **None** — reports don't learn | Genuinely new capability. We already compute realized/alpha in `evaluation_service` → the plumbing is half-built |
| **Report content** | Numbered tree + per-agent markdown + guarded sections | Two HTML→PDF generators (`report_service.py`); some interpolations **not HTML-escaped** (real bug) | Borrow section structure; fix escaping |
| **Decision prompt** | Generic per-analyst prompts | Our `SINGLE_AGENT_PROMPT` already has regime framework, lookahead discipline, falsification triggers | **We're ahead here.** Only cherry-pick specific clauses |
| **up_probability** | n/a | Fabricated `0.5 ± conf*0.3` fed to ensemble/P&L as if calibrated | Pre-existing quant-news issue; flag, don't pretend it's a probability |

---

## 4. Ignore the bad (does not fit our case)

- **The full LangGraph multi-agent debate graph.** Heavy machinery (StateGraph, ToolNode, conditional routers, SQLite checkpointer, analyst-plan/wall-time trackers) — and **our own report-layer backtest already showed the multi-agent TradingAgents scored 33% on active calls (−$31.85), vs Luna synthesis 62%.** The debate did *not* help us. The *logic* (who speaks, when to stop) is ~30 lines of `conditional_logic.py` you can inline as a plain loop if ever wanted. Skip the engine.
- **Debate context duplication** — all four analyst reports re-injected into every bull/bear/risk turn + growing history. Very token-heavy, largely redundant.
- **Depending on TA internal APIs** — churns every release; already broke us. Decouple.
- **Provider-kwargs / `create_llm_client` plumbing** — we have our own multi-provider `llm_service` (LM Studio > Anthropic > OpenAI). Redundant.
- **SQLite checkpoint/resume, symbol alias tables (forex/CFD), config singleton** — framework-specific; not our surface.
- **Verbose static indicator taxonomy** re-sent every market-analyst call — knowledge base masquerading as a per-call system prompt.

---

## 5. Honesty check (grounding in our own results)

Per `MEMORY.md`: ALPHA VERDICT (200 model-days) and R2000 diversity test (620 preds) both say **no demonstrable alpha; all models 49–56% hit (none > 2 SE from 50%); thin edges are gross of costs and vanish net.** The report-layer backtest put Luna at 62% on 26 active calls and the *full* TradingAgents multi-agent at 33%.

**Therefore:** none of the items below are expected to create returns. Their value is:
- **Backtest validity** (lookahead-safety) — so our own conclusions are trustworthy. *This is the highest-value bucket*, because we make product decisions off these backtests and have already spent effort on lookahead audits.
- **Robustness** — fewer empty-content / parse failures in the LLM/Luna path.
- **Trustworthiness & explainability** — grounded reports, and reports that reference prior calls (reflection).

Do **not** frame any of this as an alpha play. Frame it as correctness + polish.

---

## 6. Incorporation plan (prioritized)

### Tier 0 — Decide the architecture, unblock the integration (must be first)
The TA model is dead. Three options:

- **(A) Re-vendor + pin the fork baseline.** Copy `feat/ace_alpaca`'s `baselines/single_agent.py` into quant-news, port its broken imports to the v0.3.1 API. Keeps behavior; still couples us to TA's moving internals for the data tools.
- **(B) Migrate to upstream's real API.** `SingleAgentBaseline` doesn't exist upstream; we'd either drive the full `TradingAgentsGraph` (heavy, scored 33% for us) or rebuild a single-agent on the new split tools. Most work, least payoff.
- **(C) ✅ Recommended — decouple. Reimplement the single-agent in quant-news.** Move the excellent `SINGLE_AGENT_PROMPT` + decision/confidence/trigger extraction into a quant-news module (e.g. `models/single_agent_prompt.py` + logic in `trading_agents_model.py`), fed by **our own** `services/stock_data.py` + `services/news_service.py` + `services/llm_service.py`. We already compute the grounding blocks ourselves; the only thing TA supplied was data-fetch + prompt-format + LLM-call, all of which we can do natively. **Removes the fragile cross-repo dependency entirely and is where "adopt the best prompt" naturally lands.** Rename the model surface if "TradingAgents" is now a misnomer.

### Tier 1 — Lookahead-safety & backtest validity (highest defensible value)
1. **Port the half-open UTC news window** (§2A.1) into `news_service.py`; replace the fragile `app.py` string cutoff. Default undated → exclude in historical runs.
2. **Close the news coverage gap:** wire point-in-time news (AV `time_from/time_to`, or the existing 3-month `historical_news` cache) into the **TA / AI-report** path, not just XGB/LGBM. Right now LLM backtests > ~7 days old run on empty news.
3. **Fundamentals fiscal-date filter** (§2A.5) — apply in our own data layer once decoupled (Tier 0-C), since the TA path currently leaks today's fundamentals into backtest dates.
4. **Stale-OHLCV guard** (§2A.2) — reject latest row > N days before requested date.

### Tier 2 — LLM-layer robustness & grounding
5. **Structured-output fallback discipline** for `generate_recommendations` (Luna) and `summarize_news_structured`: keep our regex parse, but formalize the one-shot free-text fallback + `max_completion_tokens` headroom we already partly have; consider `with_structured_output` where the provider supports it.
6. **Grounding clause**: add upstream's "treat the validated block as source of truth; flag conflicts rather than reconcile; don't claim support/resistance bounces or exact % moves without dated data" to `summarize_news_structured`.
7. **Cheap prompt tweaks**: `NO_EXTERNAL_TOOLS` guard where we ask for JSON-only; anti-Hold-bias nudge; date-at-top.

### Tier 3 — New capability (additive, medium effort)
8. **Deferred alpha-grounded reflection loop** (§2D), reusing `evaluation_service`'s realized/alpha computation. Store decision → resolve outcome next run → 2–4 sentence lesson → re-inject recent lessons into the analysis prompt and surface in the report. *Caveat: expect explainability/calibration gains, not returns.*

### Tier 4 — Report polish (low effort)
9. Adopt the numbered section structure / richer content for the TA report; **fix the unescaped HTML interpolations** in `_build_report_html` / `generate_ta_report_pdf` (real bug — a stray `<`/`&` in model output breaks layout).

### Explicitly skipped
Full LangGraph debate graph, checkpointer, analyst-plan machinery, provider plumbing, debate context duplication, and any continued dependence on TA internal APIs (see §4).

---

## 7. Suggested first move

Tier 0-C (decouple) + Tier 1.1–1.2 (news window + coverage gap) together fix both the **breakage** and the **most consequential backtest-validity bug** in one focused change, and they're prerequisites for everything else. Everything in Tiers 2–4 is independent and can follow incrementally.

---

## 8. IMPLEMENTED (2026-07-24)

Architecture decision taken: **Option A (decouple)** — most robust (no external dep) and most swappable (research strategy behind the `BaseModel` seam). All four tiers implemented.

### New / changed files
- **`services/news_window.py`** (new) — reusable half-open UTC news window. `as_utc`, `in_news_window` (`[start, end+1day)`, undated→exclude-in-history), `filter_articles_as_of`, `av_time_bounds`, `fetch_point_in_time_news`. Smoke-tested against the exact upstream edge cases (upper-bound-exclusive at midnight-after, offset-converted-not-truncated, undated excluded in backtest / kept live).
- **`models/single_agent.py`** (new) — `SingleAgentResearch`, a fully self-contained port of our `feat/ace_alpaca` `SINGLE_AGENT_PROMPT`, fed by our own `stock_data` / `news_window` / `llm_service`. **No `tradingagents` import.** Includes the ported stale-OHLCV guard (`MAX_OHLCV_STALE_DAYS=10`), the fiscal-date fundamentals filter (`fiscalDateEnding <= as_of`), computed technicals/price/news blocks, decision/confidence/trigger extraction, and an in-process frame/fundamentals cache so a walk-forward run doesn't re-fetch SPY/sector every day. Prompt correctness fix: framed "at the close of {date}, predict the next session" to match how the backtest scores it (removes the old intraday-leak ambiguity). Also folds in the Tier-2 grounding clause ("flag conflicts, don't reconcile"), anti-Hold nudge, and no-tools instruction.
- **`models/trading_agents_model.py`** (rewritten) — thin wrapper over `SingleAgentResearch`. Dropped the dead `tradingagents.baselines.single_agent` import and its readiness gate. Threads `as_of` / `news` / `use_news` / `use_reflection` via kwargs. Keeps the model name `trading_agents` for DB/report back-compat. Still computes the validated extra-context blocks (metrics/events/peers/SPY regime).
- **`services/reflection_service.py`** (new, Tier 3) — deferred alpha-grounded reflection loop over an append-only JSONL store (atomic writes, idempotent). `record_pending` / `realized_alpha` (raw + alpha vs SPY) / `resolve_pending` (only outcomes ≤ as_of → lookahead-safe) / `get_past_context` (recent same-ticker full + cross-ticker reflection-only). Unit-tested in isolation with an injected reflect_fn. Wired into the model wrapper **opt-in** (`use_reflection`, default off) so it never contaminates the news A/B.
- **`services/llm_service.py`** — added the "treat validated block as source of truth; flag conflicts rather than reconcile; no undated bounce/%-move claims" grounding clause to `summarize_news_structured`.
- **`app.py`** — replaced the two fragile naive string-slice news cutoffs (`published_at[:19] <= "…"`) with `filter_articles_as_of` (tz-aware, upper-bound-exclusive, undated-safe).
- **`services/report_service.py`** (Tier 4) — added `import html` + `_esc()` and escaped every LLM-derived free-text interpolation in the `_build_report_html` family (exec summary, AI analysis, recommendations) — a real layout-break/injection bug (verified neutralized against `<script>`/`&` input). Relabelled the `trading_agents` model type to "Single-agent LLM research."
- **`benchmark.py`** — `run_llm_backtest` now threads `as_of`/`use_news`/`model`; added `--llm-ab` (with-news vs no-news arms) and `--llm-model` flags; per-day logging of decision + article count.

### Verification done
- News-window invariants: PASS (upper-bound exclusive, undated excluded in history, offset converted).
- Reflection loop unit tests: PASS (idempotent record, deferred resolution only after hold window, alpha-grounded, same+cross-ticker injection).
- HTML escaping: PASS (hostile tags neutralized).
- End-to-end single prediction (haiku, as-of 2026-06-30 — 3+ weeks old): decision extracted, **50 point-in-time articles** where the old rolling-7-day path returned **0** → the coverage gap is closed. All changed modules import clean.

### A/B backtest — does the Tier-1 news coverage actually change decisions?
Method: AAPL/NVDA/TSLA, last 6 sessions each, two arms over identical technicals/fundamentals — `trading_agents` (point-in-time news) vs `trading_agents_nonews` (empty news block). Model: claude-haiku-4-5. **This tests decision-divergence, not alpha** (n is far below the ~150 our own ALPHA VERDICT requires for any skill claim; see `MEMORY.md`).

**Primary result — the fix changes behavior (hypothesis confirmed):**
Point-in-time news changed the agent's decision on **8 of 18 days (44%)** vs the identical no-news inputs (AAPL 3/6, NVDA 2/6, TSLA 3/6). The coverage gap was **not** cosmetic — it was silently removing a signal that alters ~44% of calls. Every with-news day pulled a real point-in-time article set (~48–50 articles) for dates weeks in the past, where the pre-fix rolling-7-day path returned zero.

**Outcome table (n=18/arm, 11 active — far below significance):**

| arm | BUY/SELL/HOLD | active acc | all-day acc | total P&L |
|---|---|---|---|---|
| `trading_agents` (with PIT news) | 2/9/7 | 54.5% (6/11) | 72.2% | −$126.76 |
| `trading_agents_nonews` (ablation) | 0/11/7 | 72.7% (8/11) | 83.3% | +$250.00 |

Per-symbol P&L: with-news AAPL +$24.53 / NVDA −$2.32 / TSLA −$148.97; no-news AAPL +$32.96 / NVDA +$19.81 / TSLA +$197.23. Baselines (buy-and-hold): AAPL −$17.50, NVDA −$16.64, **TSLA −$197.24 (1/6 up days — a sharp down week).**

**Honest interpretation — do NOT read this as "news hurts":**
1. **n=18 is noise.** Our own `MEMORY.md` ALPHA VERDICT requires ~150 samples before any skill claim; a 15-day run once looked great and 200 days erased it. This run is an order of magnitude too small and single-week.
2. **The gap is one name in one regime.** The entire delta is TSLA: the no-news arm went **SELL 6/6** into a −$197 down week and banked +$197; the with-news arm hesitated (HOLD on 07-16/17, a wrong BUY on 07-22). That is the exact **"structural SELL bias wins in a down tape = regime luck, not skill"** artifact the R2000 test documents — being blindly short was the winning trade that week, and *less* directional conviction (which the news induced) cost P&L. It is regime fit, not evidence about news quality.
3. **AV free-tier throttling** degraded the last two TSLA with-news days to 8 and 7 articles (rate limit), so even the "with-news" arm was partially starved late — a practical note: a dense multi-day backtest needs cached PIT news or a paid AV tier.

**Conclusion:** the engineering is validated (the fix demonstrably and safely changes ~44% of decisions and closes the >7-day empty-news gap), but **whether it improves returns is unproven and unprovable at this scale** — a proper verdict needs a ≥150-sample, multi-regime, cost-aware run (the kind `MEMORY.md` already prescribes). This run reproduces the known short-window trap rather than measuring alpha, which is itself the correct, expected outcome given our prior results.

---

## 9. Follow-ups implemented (round 2)

### 9A. No fabricated probability / confidence (integrity)
The model previously exported a **fabricated** `up_probability = 0.5 ± confidence·0.3` and passed the **LLM's self-reported CONFIDENCE** straight into scoring. Two consumers made that damaging: the ensemble *model* weights each vote by `confidence` (`effective = weight · model_conf`), and the `confidence_threshold` strategy gates on it — so a confident-sounding (or lying) LLM over-weighted itself. Our own `MEMORY.md` records that these conviction labels "carried no calibration signal."

Fixed (`models/trading_agents_model.py::_grounded_confidence`):
- **Nothing is fabricated.** The LLM's own number is kept **only** as `details["stated_conviction"]`, explicitly labelled uncalibrated and **never used for scoring**.
- `confidence` (the field the ensemble/threshold consume) = the model's **realized directional hit-rate** from resolved outcomes (`reflection_service.empirical_edge`, or an `empirical_accuracy` passed by the caller from the eval DB) **once ≥8 resolved active calls exist**; **otherwise a neutral 0.5** — an honest "no earned track record," so the model can neither over- nor under-weight itself on words alone.
- `up_probability` leans in the decision's direction **only by the earned edge** (`hit_rate − 0.5`, floored at 0); with no history it stays 0.5 (a declared *direction*, not an invented *probability*).
- The UI (`signal_components.py`) now labels it "Realized reliability" / "Directional (no track record)" instead of a bare "Confidence."

This is the "something better than a self-reported probability" the brief asked for: a grounded, self-correcting reliability weight that a model *earns*.

### 9B. Swappable research backend (future TradingAgents ingestion)
`models/research_backend.py` defines a `ResearchAgent` protocol and a registry. `models/trading_agents_model.py` now calls `get_research_agent(...)` instead of a concrete class. Backend is chosen by `predict(backend=…)` kwarg > `RESEARCH_BACKEND` env > `MODEL.RESEARCH_BACKEND` (default `single_agent`). `SingleAgentResearch` satisfies the protocol; `TradingAgentsAdapter` is a **loud stub** (raises with instructions, never a silent HOLD). **To adopt a future TradingAgents release:** pin the version, implement the adapter's `analyze()` against its *public* API, set `RESEARCH_BACKEND=tradingagents` — no caller, ensemble, persistence, or report change. (Verified: default→in-tree, adapter fails loudly, unknown→ValueError.)

### 9C. Large paid-AV backtest (news at scale + reflection A/B)
Run: AAPL/NVDA/TSLA/JPM/XOM/WMT (regime-diverse), last 15 sessions each, three arms — `trading_agents` (news on), `trading_agents_nonews` (news ablation), `trading_agents_reflect` (news on + reflection on). Paid AV (no rate-limit starvation), in-process PIT-news cache. Still below the ≥150-per-arm ideal (~90/arm), but 5× the first run with real regime spread.

**Results (90 model-days/arm, 0 errors; confidence honestly neutral 50% throughout — too few 5-day outcomes resolved in-window to ground it):**

| arm | BUY/SELL/HOLD | active acc | all-day acc | total P&L |
|---|---|---|---|---|
| `trading_agents` (news on) | 11/38/41 | 38.8% (19/49) | 66.7% | +$46.59 |
| `trading_agents_nonews` | 8/38/44 | 47.8% (22/46) | 73.3% | +$178.98 |
| `trading_agents_reflect` (**invalid — see below**) | 8/55/27 | 42.9% (27/63) | 60.0% | +$16.65 |

**Directive 1 — news at scale.** All arms sit **below 50% active accuracy** (38.8 / 47.8 / 42.9%). At n≈46–63 active, SE≈7%, so none is distinguishable from a coin flip — the ALPHA VERDICT pattern, again. Notably, **news underperformed no-news for the second independent time** (small run: 54.5 vs 72.7; here: 38.8 vs 47.8). That is now a *pattern worth taking seriously as a hypothesis*: the point-in-time news, as fed (≤50 headline+summary items, relevance ≥0.5), may add noise rather than directional signal at a 1–5 day horizon — the LLM chases narrative against momentum. It is **not** yet a conclusion (n too small; the news-arm damage is again concentrated in one name — AAPL 29% vs 86% no-news), and it does **not** undercut the lookahead-safety work (that is about *validity of the backtest*, not returns). **Actionable:** treat "should the model even read news for short-horizon calls?" as an open, testable question — the ablation harness (`--llm-ab`) now answers it cheaply at scale whenever you want.

**Directive 2 — reflection arm was INVALID (bug found + fixed).** Inspecting the reflection store exposed a real bug in the reflector: it was given the realized return but **not the decision direction**, so it inferred correctness from the return's sign — labelling a **SELL that preceded a +6.2% move as a "correct long."** The reflection arm was therefore trained on **actively wrong lessons**, which both explains its non-benefit and invalidates it as a test. Fixed in `reflection_service.py`: the reflector now receives the explicit decision and the correctness rule (BUY correct iff return>0; SELL iff return<0), verified on the exact failing case. (Note: the *grounded confidence* `empirical_edge` was already correct — it computes hit-rate from return signs, so it was unaffected; only the LLM lesson *text* was wrong.) A **clean reflection A/B is re-running** with the fix.

<!-- REFLECT_RETEST_PLACEHOLDER -->

**Bottom line so far:** at ~90/arm across six regime-diverse names, the decoupled + lookahead-safe + honest-confidence model is a coin flip on direction (as expected — no model here has shown alpha), news shows a repeat tendency to *not help*, and reflection needs the clean re-test before any verdict. The delivered value remains **correctness, validity, robustness, and integrity** — not returns.



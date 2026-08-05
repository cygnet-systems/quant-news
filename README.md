# QuantNews

Real-time stock analysis dashboard with LLM-powered news insights and technical indicators.

![Dashboard Demo](assets/demo.png)

## Features

- **Multi-stock comparison** with normalized % change charts
- **Technical indicators**: SMA (20/50/200), Bollinger Bands, MACD, RSI
- **Per-symbol news tabs** with AI-powered recommendation banners
- **Sentiment analysis** with visual breakdowns and confidence scores
- **Live news feed** with structured LLM summaries
- **Data export** to Parquet format with DuckDB caching

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env  # Add your API keys

# Run the dashboard (uvicorn/ASGI — `python app.py` wraps this)
python app.py
# or explicitly:
uvicorn app:server --host 127.0.0.1 --port 8050 --workers 1 --loop asyncio
```

Open http://127.0.0.1:8050 in your browser.

> **Single worker only.** The app runs Dash 4 on its FastAPI backend under
> uvicorn. `--workers` must stay at 1: torch/MPS models, per-process caches,
> and the APScheduler all live in process memory. Concurrency comes from the
> event loop + threadpools (hot callbacks are `async def`), not from workers.
> `--loop asyncio` is required (plotly_cloud's nest_asyncio hook cannot patch
> uvloop).

## Cygnet SSO & data visibility

QuantNews is a Cygnet Systems sister application (alongside
CygnetResearchTerminal) designed to sit behind a shared SSO portal:

- **Anonymous use is fully supported** — all data is public by default.
  Signing in only attaches ownership (`owner_uid`) to predictions, research
  reports, recommendation runs, and catalog entries; per-user privacy is a
  flip of `is_public` when the product needs it.
- **Login**: the sidebar "Sign in" chip posts to `/auth/login`, validated
  against the shared Cygnet `users` table (same schema/password hashing as
  CygnetResearchTerminal). Sessions are server-side rows (`sessions` table),
  7-day TTL, referenced by an itsdangerous-signed `qn_session` cookie.
- **Portal handoff**: the SSO portal signs a user's raw session token with
  the shared `SESSION_COOKIE_SECRET_KEY` (salt `cygnet-sso-handoff`, 60s
  validity) and redirects to `/sso/login?token=…&next=/`; the app validates
  the session in the shared store and sets its own cookie. This is the
  cross-app SSO mechanism — `*.up.railway.app` subdomains cannot share
  cookies (public-suffix rule), so trust travels via the signed token.
- `/auth/whoami` returns the caller's identity — the portal catalog can use
  it to show signed-in state per app.

**Railway deployment (per app)**: set `SESSION_COOKIE_SECRET_KEY` (identical
across all Cygnet apps), `AUTH_DATABASE_URL` (the shared auth Postgres),
`DATABASE_URL`, and the S3 vars. `alembic upgrade head` runs from the
entrypoint; the auth tables are created automatically if absent.

## Scheduled runs

**The app schedules itself.** An APScheduler instance starts with the server
(ASGI lifespan → `services/scheduler_service.py`) and runs two jobs:

| Job | Default | What it does |
|---|---|---|
| `daily_analysis` | 08:00 ET, weekdays | Full Analysis on the watchlist, targeting that morning's session |
| `daily_evaluation` | 18:00 ET, weekdays | Scores predictions whose target session has closed, then runs the strategies |

Schedules live in the `scheduled_jobs` table and are edited from **History →
Scheduled Jobs**: time, days, timezone, symbol list, enable/disable, plus
*Run now* and the last run's output. No cron, no launchd, nothing tied to a
particular machine — deploying the app deploys the schedule.

Four properties make it safe on an always-on deploy:

- **Advisory lock** — every run takes a Postgres advisory lock on its job id,
  so a rollout with two live instances cannot double-fire an expensive job.
- **Catch-up by date** — on startup, a job whose window passed today with no
  success recorded *for today* runs once. A restart across 08:00 still
  produces the day's analysis; a crash loop does not re-run it repeatedly.
- **Database sync** — the live triggers are reconciled with the stored
  schedule every 60s, so an edit made in another instance (or straight in the
  database) takes effect without a restart.
- **Subprocess execution** — jobs shell out to the CLI below, keeping model
  memory and a ~10-minute CPU burn out of the process serving the UI.

The same pipeline is available directly, which is what the jobs invoke:

```bash
python scripts/daily_analysis.py analyze                       # next session, watchlist
python scripts/daily_analysis.py analyze --symbols PANW --target 2026-08-03
python scripts/daily_analysis.py evaluate                      # score what has closed
```

It writes anonymously, so everything it produces is public and appears in
History, the Scoreboard and the Activity Log.

**Two instances can run schedulers at once.** They read one schedule from the
database and re-sync every 60s, the advisory lock means only one executes a
given run, and the reuse caches make a second run idempotent if it ever
happens. `SCHEDULER_ENABLED=0` is a per-process stand-down for the case that
does not cover — a process sharing the production database that should not
spend money, such as the app running on a laptop pointed at prod. To stop a
job for everyone, disable the job itself in the UI. A stood-down process
reports `scheduling_disabled` and stays healthy, so it reads differently from
a crashed scheduler.

### Knowing it actually ran

`GET /healthz` reports the schedule, not just the process:

```json
{"scheduler_running": true, "overdue": [], "healthy": true,
 "jobs": [{"id": "daily_analysis", "next_run_at": "…", "last_success_date": "…"}]}
```

It returns **503** when the scheduler thread is dead or a job's window has
passed (plus a timeout's grace) with no success recorded that day — so an
uptime monitor pointed here reports "today's analysis didn't happen", which a
plain page check would miss entirely. A watchdog logs the same condition to
the Activity Log every 30 minutes.

### Email notifications

Set the five `AZURE_*` / `NOTIFY_*` variables (see `.env.example`) and each
scheduled run mails its outcome. The transport is inherited from
CygnetResearchTerminal's `cron_jobs/notify.py` — Microsoft Graph `sendMail`
with the same variable names, so one Azure app registration serves both apps.
With any of them missing, notifications stay off and runs are unaffected.

| Mail | When | Contains |
|---|---|---|
| Pre-open calls | after the morning analysis | the call per symbol, buy/sell/hold counts, predictions stored, duration, LLM cost |
| Results | after the evening evaluation | hit rate and P&L per model, the synthesis call per symbol with right/wrong |
| Job failed | any non-zero exit | the captured output tail |
| Job overdue | watchdog | which job missed its window (once per day, not every 30 min) |

**On Railway:** services are always-on by default; the scale-to-zero
"Serverless" mode is opt-in per service and should stay **off** here. Sleep is
judged on *outbound* traffic, and this app's connection pool plus the
scheduler's 60s sync mean it never idles — so enabling it would save nothing
and add cold-start risk. The real interruptions are deploys, crashes and plan
limits: catch-up covers the first, `/healthz` surfaces the rest.

Re-running an analysis nothing has changed under costs nothing: a symbol whose
models already ran for this cutoff — against the same closing bar — is reused,
and the report and synthesis are served from their input-hash caches. A repeat
run finishes in seconds with zero API calls. `--force` re-runs anyway.

## Cost telemetry

Every LLM call is recorded in `llm_usage`: exact token counts from the provider
response, the $/Mtok rates applied, the derived cost, and which stage spent it
(`research` / `ai_report` / `recommendations`). Nothing in that table is ever
fed back into a prompt.

```bash
python scripts/daily_analysis.py cost --days 7
```

Rates live in `LLM_PRICING` in [config.py](config.py) and are copied onto each
row as it is written, so repricing a model never rewrites the cost of calls
already made. A model with no entry records its tokens with a NULL cost and is
counted separately in the report — an unpriced call is visibly unpriced rather
than quietly free. **Verify the seeded rates against current provider pricing
before treating spend numbers as exact.**

## Configuration

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key for AI summaries |
| `ALPHA_VANTAGE_API_KEY` | Alpha Vantage API key for enhanced news coverage |

Alternatively, connect to a local LLM via LM Studio on port 1234. yfinance is used for news if no Alpha Vantage key is provided.

## Tech Stack

Dash + Plotly | DuckDB | yfinance | Bootstrap 5 (Darkly)

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

## Configuration

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key for AI summaries |
| `ALPHA_VANTAGE_API_KEY` | Alpha Vantage API key for enhanced news coverage |

Alternatively, connect to a local LLM via LM Studio on port 1234. yfinance is used for news if no Alpha Vantage key is provided.

## Tech Stack

Dash + Plotly | DuckDB | yfinance | Bootstrap 5 (Darkly)

"""LLM service for news summarization and analysis.

This module provides integration with LM Studio (local), Anthropic Claude,
and OpenAI (cloud) for generating news summaries and sentiment analysis.
"""

import json
import logging
import os
import re
import threading
from typing import Optional

import requests
from openai import OpenAI

from config import API, MODEL

logger = logging.getLogger(__name__)

# Claude models that reject `temperature`/`top_p` and the old fixed-budget
# thinking config; they use adaptive thinking + `output_config.effort` instead.
# Sending temperature to one of these is a hard 400, so the request must be
# built differently rather than degrading gracefully.
_NO_SAMPLING_PARAMS = ("claude-opus-5", "claude-opus-4-7", "claude-opus-4-8",
                       "claude-sonnet-5", "claude-fable-5", "claude-mythos-5")


def _rejects_sampling_params(model: str) -> bool:
    """True if `model` 400s on temperature / budget_tokens."""
    return any(model.startswith(p) for p in _NO_SAMPLING_PARAMS)


def _first_text(response) -> Optional[str]:
    """Extract the first text block, skipping thinking blocks."""
    for block in response.content:
        if block.type == "text":
            return block.text
    return None


def _record_usage(usage_out: Optional[dict], response, provider: str,
                  model: str, duration_ms: Optional[int] = None) -> None:
    """Normalise a response's token counts, then fan them out two ways.

    1. ``usage_out``, when a caller passed one — an out-parameter rather than a
       changed return type, because ``generate()`` returns a plain string to a
       couple of dozen call sites and only the report path wants the counts.
    2. The durable ``llm_usage`` table, ALWAYS. Cost telemetry that depended on
       callers opting in would only ever measure the paths someone remembered
       to instrument; this is the one funnel every provider path returns
       through, so recording here means no call goes uncounted.

    Anthropic reports input_tokens/output_tokens; the OpenAI-compatible path
    reports prompt_tokens/completion_tokens. Both are normalised here so
    callers never branch on provider. Never raises — telemetry must not be
    able to fail a generation that already succeeded.
    """
    in_tok = out_tok = 0
    try:
        usage = getattr(response, "usage", None)
        if usage is not None:
            if provider == "anthropic":
                raw_in = getattr(usage, "input_tokens", None)
                raw_out = getattr(usage, "output_tokens", None)
            else:
                raw_in = getattr(usage, "prompt_tokens", None)
                raw_out = getattr(usage, "completion_tokens", None)
            in_tok = int(raw_in) if raw_in is not None else 0
            out_tok = int(raw_out) if raw_out is not None else 0

        if usage_out is not None:
            usage_out.update({
                "input_tokens": in_tok,
                "output_tokens": out_tok,
                "model": model,
                "provider": provider,
            })
    except Exception as e:  # pragma: no cover - telemetry is best-effort
        logger.debug("usage capture failed: %s", e)

    from services import usage_service
    usage_service.record(
        model=model, provider=provider,
        input_tokens=in_tok, output_tokens=out_tok,
        duration_ms=duration_ms,
    )


# Default model per provider — single source of truth for the three copies of
# this mapping that used to live in _get_model / generate / failover.
_DEFAULT_MODELS = {
    "lm_studio": "local-model",
    "anthropic": "claude-haiku-4-5-20251001",
    "openai": "gpt-3.5-turbo",
}


class LLMService:
    """Service for LLM-powered text generation.

    Supports LM Studio (local), Anthropic Claude, and OpenAI API fallback.

    Concurrency: called from ThreadPoolExecutor workers, async callbacks (via
    threads), and the prediction subprocess. The active (client, provider)
    pair lives in a single tuple attribute so reads and failover swaps are
    atomic reference operations — methods must never mutate client/provider
    piecewise.

    Attributes:
        client: OpenAI or Anthropic client instance (read-only property).
        provider: Current provider ('lm_studio', 'anthropic', or 'openai').
    """

    def __init__(self) -> None:
        """Initialize LLM service with auto-detection of providers."""
        self._client_cache: dict = {}
        self._cache_lock = threading.Lock()
        self._active: tuple = (None, None)  # (client, provider)
        self._initialize_client()

    @property
    def client(self):
        return self._active[0]

    @property
    def provider(self) -> Optional[str]:
        return self._active[1]

    def _make_client(self, provider: str):
        """Construct (and cache) an SDK client for a provider.

        Clients get an explicit timeout — the SDK default of 600s pins a
        worker thread on a hung socket. SDK retries stay at 1 because
        generate() layers its own transient retry on top.
        """
        with self._cache_lock:
            cached = self._client_cache.get(provider)
            if cached is not None:
                return cached
            if provider == "lm_studio":
                client = OpenAI(base_url=API.LM_STUDIO_URL, api_key="not-needed",
                                timeout=API.LLM_TIMEOUT, max_retries=1)
            elif provider == "anthropic":
                import anthropic
                client = anthropic.Anthropic(api_key=API.ANTHROPIC_API_KEY,
                                             timeout=API.LLM_TIMEOUT, max_retries=1)
            elif provider == "openai":
                client = OpenAI(base_url=API.OPENAI_BASE_URL, api_key=API.OPENAI_API_KEY,
                                timeout=API.LLM_TIMEOUT, max_retries=1)
            else:
                return None
            self._client_cache[provider] = client
            return client

    def _initialize_client(self) -> None:
        """Pick a provider with priority: LM Studio > Anthropic > OpenAI."""
        if self._is_lm_studio_available():
            self._active = (self._make_client("lm_studio"), "lm_studio")
            return

        if API.ANTHROPIC_API_KEY:
            try:
                self._active = (self._make_client("anthropic"), "anthropic")
                return
            except Exception:
                pass

        if API.OPENAI_API_KEY:
            self._active = (self._make_client("openai"), "openai")
            return

        self._active = (None, None)

    def _is_lm_studio_available(self) -> bool:
        """Check if LM Studio is running AND has an LLM loaded in memory.

        The OpenAI-compatible /v1/models endpoint lists every *downloaded*
        model regardless of load state, so a 200 (or a non-empty list)
        there isn't enough — LM Studio still 400s on every completion with
        "No models loaded". LM Studio's native /api/v0/models endpoint
        exposes a per-model "state" field, so we require at least one llm
        with state == "loaded". If that endpoint isn't reachable we treat
        LM Studio as unavailable and fall through to a cloud provider
        rather than silently degrading to a broken local one.

        Returns:
            True only if an LLM is actually loaded and ready to serve.
        """
        try:
            base = API.LM_STUDIO_URL.rstrip("/")
            if base.endswith("/v1"):
                base = base[:-3]
            native_url = base.rstrip("/") + "/api/v0/models"
            response = requests.get(native_url, timeout=2)
            if response.status_code != 200:
                return False
            models = response.json().get("data", [])
            return any(
                m.get("state") == "loaded" and m.get("type") == "llm"
                for m in models
            )
        except Exception:
            return False

    def is_available(self) -> bool:
        """Check if any LLM provider is available.

        Returns:
            True if an LLM provider is configured and responding.
        """
        return self.provider is not None

    def get_provider_info(self) -> dict:
        """Get information about the current LLM provider.

        Returns:
            Dictionary with provider name and status.
        """
        return {
            "provider": self.provider or "none",
            "available": self.is_available(),
            "lm_studio_url": API.LM_STUDIO_URL,
            "has_anthropic_key": bool(API.ANTHROPIC_API_KEY),
            "has_openai_key": bool(API.OPENAI_API_KEY),
        }

    def _get_model(self) -> str:
        """Get the appropriate model name for the provider.

        Returns:
            Model identifier string.
        """
        return _DEFAULT_MODELS.get(self.provider or "", "")

    def _get_client_for_provider(self, provider: str):
        """Return a cached client for a specific provider.

        Returns (client, provider_str) or (None, None) if unavailable.
        """
        if provider == "openai" and API.OPENAI_API_KEY:
            return self._make_client("openai"), "openai"
        if provider == "anthropic" and API.ANTHROPIC_API_KEY:
            return self._make_client("anthropic"), "anthropic"
        if provider == "lm_studio" and self._is_lm_studio_available():
            return self._make_client("lm_studio"), "lm_studio"
        return None, None

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 500,
        temperature: float = 0.7,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        usage_out: Optional[dict] = None,
        **kwargs,
    ) -> Optional[str]:
        """Generate text using the LLM.

        Args:
            prompt: User prompt/question.
            system_prompt: Optional system instructions.
            max_tokens: Maximum tokens in response.
            temperature: Creativity parameter (0-1).
            model: Override model name (uses default for provider if None).
            provider: Override provider ('openai', 'anthropic', 'lm_studio').
                      Creates a temporary client; does not mutate self.
            usage_out: Optional dict, populated in place with input_tokens,
                      output_tokens, model and provider for the call that
                      actually produced the text — including after a failover,
                      so recorded cost matches the model that was really used.
            **kwargs: Provider-specific params (e.g. reasoning_effort for OpenAI).

        Returns:
            Generated text or None if unavailable.
        """
        # Single atomic read: never look at self.client/self.provider
        # separately, another thread's failover could swap them mid-read.
        use_client, use_provider = self._active
        use_model = model

        if provider and provider != use_provider:
            use_client, use_provider = self._get_client_for_provider(provider)
            if use_client is None:
                logger.warning(f"Requested provider '{provider}' is not available")
                return None

        if not use_provider or use_client is None:
            return None

        if not use_model:
            use_model = _DEFAULT_MODELS.get(use_provider, "")

        import time as _time
        _call_t0 = _time.time()

        def _elapsed_ms() -> int:
            return int((_time.time() - _call_t0) * 1000)

        try:
            if use_provider == "anthropic":
                api_kwargs = {
                    "model": use_model,
                    "max_tokens": max_tokens,
                    "messages": [{"role": "user", "content": prompt}],
                }
                if system_prompt:
                    api_kwargs["system"] = system_prompt

                if _rejects_sampling_params(use_model):
                    # Adaptive thinking + effort; temperature is not accepted.
                    if kwargs.get("reasoning_effort") == "max":
                        api_kwargs["thinking"] = {"type": "adaptive"}
                        api_kwargs["output_config"] = {"effort": "max"}
                    else:
                        # These models think by default, and max_tokens caps
                        # thinking + text together -- leaving it on silently
                        # truncates the answer (stop_reason=max_tokens) and
                        # callers lose the DECISION line. Callers pass
                        # max_tokens as a *response* budget, so opt out.
                        api_kwargs["thinking"] = {"type": "disabled"}
                else:
                    api_kwargs["temperature"] = temperature
                    if kwargs.get("reasoning_effort") == "max":
                        api_kwargs["temperature"] = 1.0
                        api_kwargs["thinking"] = {
                            "type": "enabled", "budget_tokens": max_tokens * 2,
                        }

                response = use_client.messages.create(**api_kwargs)
                if getattr(response, "stop_reason", None) == "max_tokens":
                    # Truncation was previously silent; on report prompts the
                    # decision footer is the first casualty, so make it loud.
                    logger.warning(
                        f"LLM response truncated at max_tokens={max_tokens} "
                        f"(model={use_model}) — output is incomplete"
                    )
                _record_usage(usage_out, response, "anthropic", use_model,
                              duration_ms=_elapsed_ms())
                return _first_text(response)

            # OpenAI-compatible path (LM Studio, OpenAI)
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            api_kwargs = {
                "model": use_model,
                "messages": messages,
            }
            is_reasoning = bool(kwargs.get("reasoning_effort") and use_provider == "openai")
            if is_reasoning:
                # Reasoning models spend completion budget on reasoning FIRST.
                # max_completion_tokens == desired content size returns HTTP 200
                # with EMPTY content once reasoning eats the budget — give
                # headroom on top of the requested content size.
                effort = kwargs["reasoning_effort"]
                if effort == "max":
                    effort = "high"
                headroom = {"low": 2, "medium": 4, "high": 8}.get(effort, 4)
                api_kwargs["max_completion_tokens"] = max(max_tokens * headroom, 16000)
                api_kwargs["reasoning_effort"] = effort
            else:
                api_kwargs["max_tokens"] = max_tokens
                api_kwargs["temperature"] = temperature

            # Transient errors (401 permission flaps, 429, 5xx) shouldn't kill
            # a whole pipeline — retry once with a short backoff.
            try:
                response = use_client.chat.completions.create(**api_kwargs)
            except Exception as api_err:
                msg = str(api_err)
                transient = any(t in msg for t in ("401", "429", "500", "502", "503",
                                                   "insufficient permissions", "overloaded"))
                if not transient:
                    raise
                logger.warning(f"Transient OpenAI error, retrying in 3s: {msg[:120]}")
                _time.sleep(3)
                response = use_client.chat.completions.create(**api_kwargs)

            content = response.choices[0].message.content

            # Empty content despite HTTP 200 = reasoning consumed the budget.
            # Retry once at low effort with a large budget rather than failing.
            if is_reasoning and not (content or "").strip():
                finish = getattr(response.choices[0], "finish_reason", "?")
                logger.warning(
                    f"Reasoning model returned empty content (finish_reason={finish}) — "
                    f"retrying with low effort"
                )
                api_kwargs["reasoning_effort"] = "low"
                api_kwargs["max_completion_tokens"] = max(max_tokens * 4, 20000)
                response = use_client.chat.completions.create(**api_kwargs)
                content = response.choices[0].message.content

            _record_usage(usage_out, response, use_provider, use_model,
                          duration_ms=_elapsed_ms())
            return content

        except Exception as e:
            logger.warning(f"LLM generation error ({use_provider}): {e}")

            # Auto-failover only when using default provider (not overrides).
            # Work entirely in locals; publish the new active pair only after
            # a success, as one atomic tuple swap.
            if not provider and use_provider == "lm_studio":
                logger.info("LM Studio failed, attempting cloud failover")

                if API.ANTHROPIC_API_KEY:
                    try:
                        fo_client, _ = self._get_client_for_provider("anthropic")
                        result = self._generate_anthropic(
                            fo_client, prompt, system_prompt, max_tokens, temperature,
                            usage_out=usage_out,
                        )
                        self._active = (fo_client, "anthropic")
                        logger.info("Failover to Anthropic succeeded, keeping provider")
                        return result
                    except Exception as e2:
                        logger.warning(f"Anthropic failover failed: {e2}")

                if API.OPENAI_API_KEY:
                    try:
                        fo_client, _ = self._get_client_for_provider("openai")
                        messages = []
                        if system_prompt:
                            messages.append({"role": "system", "content": system_prompt})
                        messages.append({"role": "user", "content": prompt})
                        response = fo_client.chat.completions.create(
                            model=_DEFAULT_MODELS["openai"],
                            messages=messages,
                            max_tokens=max_tokens,
                            temperature=temperature,
                        )
                        self._active = (fo_client, "openai")
                        logger.info("Failover to OpenAI succeeded, keeping provider")
                        _record_usage(usage_out, response, "openai",
                                      _DEFAULT_MODELS["openai"])
                        return response.choices[0].message.content
                    except Exception as e3:
                        logger.warning(f"OpenAI failover failed: {e3}")

            # A failed call still consumed wall-clock and, on a mid-stream
            # error, possibly tokens the provider will bill. Record it as a
            # zero-token failure so the stage's call count matches reality
            # instead of quietly dropping the attempt.
            from services import usage_service
            usage_service.record(
                model=use_model, provider=use_provider,
                input_tokens=0, output_tokens=0,
                duration_ms=_elapsed_ms(), ok=False, error=str(e),
            )
            return None

    def _generate_anthropic(
        self,
        client,
        prompt: str,
        system_prompt: Optional[str],
        max_tokens: int,
        temperature: float,
        usage_out: Optional[dict] = None,
    ) -> Optional[str]:
        """Generate text using Anthropic Claude API (explicit client — used
        by failover before the active pair is swapped)."""
        kwargs = {
            "model": _DEFAULT_MODELS["anthropic"],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        response = client.messages.create(**kwargs)
        _record_usage(usage_out, response, "anthropic", _DEFAULT_MODELS["anthropic"])
        return response.content[0].text

    def summarize_news(
        self,
        articles: list[dict],
        symbol: str,
    ) -> Optional[str]:
        """Generate a summary of news articles.

        Args:
            articles: List of article dictionaries with 'title' and 'summary'.
            symbol: Stock symbol for context.

        Returns:
            AI-generated summary or None if unavailable.
        """
        if not articles:
            return None

        # Build context from articles
        article_text = "\n".join([
            f"- {a.get('title', '')}: {a.get('summary', '')[:200]}"
            for a in articles[:10]  # Limit to 10 articles
        ])

        system_prompt = """You are an objective financial news analyst. Synthesize news into a clear, actionable conclusion.

RULES:
- Only state facts from the provided articles. Never fabricate.
- Commit to ONE clear recommendation. No hedging or "on the other hand" statements.
- If news is mixed, pick the direction with stronger evidence.
- Be concise. No filler. No caveats after your recommendation.
- Use the EXACT markdown format provided. Do not deviate."""

        prompt = f"""Analyze recent news for {symbol}:

{article_text}

Respond using this EXACT markdown format:

### Key Developments

[2-3 sentences on the most important news. Be specific about events, numbers, or catalysts mentioned.]

---

### Market Sentiment

**[BULLISH / BEARISH / NEUTRAL]**

[One sentence explaining why based on news tone and content.]

---

### Recommendation

> **[BULLISH / CAUTIOUS BULLISH / NEUTRAL / CAUTIOUS BEARISH / BEARISH]**

[One sentence explaining your recommendation. No counterpoints.]"""

        return self.generate(prompt, system_prompt, max_tokens=450)

    def summarize_news_structured(
        self,
        articles: list[dict],
        symbols: list[str],
        stock_data: Optional[dict] = None,
        as_of_date: Optional[str] = None,
        extra_blocks: Optional[dict] = None,
        include_thesis: bool = False,
        model: Optional[str] = None,
        provider: Optional[str] = None,
    ) -> Optional[dict]:
        """Generate a structured analysis grounded in financial data and news.

        Args:
            articles: List of article dicts with title, summary, sentiment,
                      sentiment_score, ticker_relevance_score, etc.
            symbols: List of stock symbols for context.
            stock_data: Optional dict keyed by symbol with 'metrics', 'signals',
                        and 'info' sub-dicts from the stock-data-store.
            as_of_date: Analysis cut-off date (ISO). When set, the model is told
                        to reason as of that date only.
            extra_blocks: Optional dict of pre-formatted validated prompt blocks:
                          {'metrics': str, 'events': str, 'peers': str, 'profile': str}.
            include_thesis: When True, request the company_thesis section
                            (background, perception, catalysts, regime risks).

        Returns:
            Dictionary with structured analysis data or None if unavailable.
        """
        if not articles:
            return None

        article_text = "\n".join([
            f"- [{a.get('sentiment', 'unknown').upper()}] "
            f"(relevance:{a.get('ticker_relevance_score', 'N/A')}, "
            f"score:{a.get('sentiment_score', 'N/A')}) "
            f"{a.get('title', '')}: {a.get('summary', '')[:200]}"
            for a in articles[:15]
        ])

        sentiment_counts = {"bullish": 0, "neutral": 0, "bearish": 0}
        scores = []
        for a in articles:
            s = a.get("sentiment", "neutral").lower()
            if "bullish" in s:
                sentiment_counts["bullish"] += 1
            elif "bearish" in s:
                sentiment_counts["bearish"] += 1
            else:
                sentiment_counts["neutral"] += 1
            if a.get("sentiment_score") is not None:
                scores.append(a["sentiment_score"])

        avg_score = sum(scores) / len(scores) if scores else None

        symbols_str = ", ".join(symbols) if symbols else "the stocks"

        financial_context = ""
        if stock_data:
            for sym in symbols:
                sym_data = stock_data.get(sym, {})
                metrics = sym_data.get("metrics", {})
                signals = sym_data.get("signals", {})
                info = sym_data.get("info", {})

                if not metrics and not info:
                    continue

                lines = [f"\n--- {sym} Financial Data ---"]

                if info:
                    # Every line below is emitted ONLY when its data is really
                    # present. The as-of-safe flows (Full Analysis, any
                    # backtest) deliberately strip live quote fields so the
                    # model cannot see the session being predicted; printing
                    # them unconditionally rendered the absence as
                    # "Current Price: $0.00 | Volume: 0 | P/E: N/A" for every
                    # symbol, and the analyst model — correctly — refused to
                    # analyze and reported a broken data feed instead.
                    lines.append(f"Company: {info.get('name', sym)} | Sector: {info.get('sector', 'N/A')} | Industry: {info.get('industry', 'N/A')}")
                    if info.get("market_cap") or info.get("pe_ratio") or info.get("dividend_yield"):
                        mcap = (f"${info['market_cap']:,.0f}" if info.get("market_cap")
                                else "N/A")
                        lines.append(f"Market Cap: {mcap} | P/E Ratio: {info.get('pe_ratio', 'N/A')} | Dividend Yield: {info.get('dividend_yield', 'N/A')}")
                    if info.get("current_price"):
                        lines.append(f"Current Price: ${info['current_price']:.2f} | Previous Close: ${info.get('previous_close', 0):.2f} | Day Change: {info.get('day_change_percent', 0):.2f}%")
                    # yfinance's 52w figures are raw exchange prices; every
                    # indicator here runs on dividend-adjusted history, so
                    # label the basis or the LLM flags a phantom discrepancy.
                    if info.get("fifty_two_week_high"):
                        lines.append(f"52-Week High: ${info['fifty_two_week_high']:.2f} | 52-Week Low: ${info.get('fifty_two_week_low', 0):.2f} (exchange figures, unadjusted)")
                    if info.get("volume") or info.get("avg_volume"):
                        lines.append(f"Volume: {info.get('volume', 0):,} | Avg Volume: {info.get('avg_volume', 0):,}")

                if metrics:
                    lines.append(f"Period Return: {metrics.get('total_return', 'N/A')}% | Volatility: {metrics.get('volatility', 'N/A')}%")
                    lines.append(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 'N/A')} | Max Drawdown: {metrics.get('max_drawdown', 'N/A')}%")
                    lines.append(f"Win Rate: {metrics.get('win_rate', 'N/A')}% | Best Day: {metrics.get('best_day', 'N/A')}% | Worst Day: {metrics.get('worst_day', 'N/A')}%")
                    lines.append(f"Price Range: ${metrics.get('start_price', 'N/A')} → ${metrics.get('end_price', 'N/A')} ({metrics.get('start_date', '')} to {metrics.get('end_date', '')})")

                if signals:
                    signal_parts = []
                    for key, val in signals.items():
                        if isinstance(val, dict):
                            signal_parts.append(f"{key}: {val.get('signal', str(val))}" +
                                                (f" ({val.get('value', ''):.1f})" if 'value' in val else ""))
                    if signal_parts:
                        lines.append(f"Technical Signals: {' | '.join(signal_parts)}")

                financial_context += "\n".join(lines)

        sentiment_summary = (
            f"Sentiment counts: {sentiment_counts['bullish']} bullish, "
            f"{sentiment_counts['neutral']} neutral, {sentiment_counts['bearish']} bearish"
        )
        if avg_score is not None:
            sentiment_summary += f" | Average sentiment score: {avg_score:.3f} (scale: -1 bearish to +1 bullish)"

        system_prompt = """You are a senior equity research analyst. Produce a grounded analysis using ALL provided data — financial metrics, technical signals, AND news. Respond with ONLY valid JSON.

CRITICAL RULES:
- Your response must be parseable JSON with no additional text, no markdown code blocks.
- Ground your recommendation in the FINANCIAL DATA: price action, valuation (P/E), technicals (RSI, MACD, trend), and risk metrics (volatility, drawdown).
- Every number you cite must come from the provided data. Never estimate or invent a figure.
- News sentiment should CONFIRM or CHALLENGE the technical/fundamental picture, not replace it.
- Confidence is your estimated probability (0.0-1.0) that the recommendation direction is correct; high only when technicals, fundamentals, AND sentiment agree.
- If signals conflict (e.g., bullish news but overbought RSI), lower confidence and note the divergence.
- DRAWDOWN CAUSALITY: if the stock is down >20% from its period high, state a causal hypothesis with evidence from the news/fundamentals, or write "cause unknown — elevated risk".
- If a VALIDATED METRICS block is present, use its ATR/support/resistance/R:R arithmetic for any risk-reward statement instead of qualitative claims.
- Treat any VALIDATED block as the source of truth. If two data sources conflict, FLAG the discrepancy rather than inventing a reconciled number. Do not claim historical support/resistance bounces or exact percentage moves unless a data block states them with concrete dates and prices."""

        as_of_line = f"\nANALYSIS AS-OF DATE: {as_of_date}. Reason only with information available on or before this date.\n" if as_of_date else ""

        eb = extra_blocks or {}
        validated_context = ""
        for key, label in (("profile", "COMPANY PROFILE"), ("metrics", "VALIDATED METRICS"),
                           ("events", "EVENT CALENDAR"), ("peers", "PEER RELATIVE STRENGTH")):
            if eb.get(key):
                validated_context += f"\n{label}:\n{eb[key]}\n"

        thesis_schema = ""
        if include_thesis:
            thesis_schema = (
                ', "company_thesis": {'
                '"perception": "2-3 sentences: how the market currently perceives this company based on the news coverage, vs. what the company claims to be/do (from the profile)", '
                '"goal_alignment": "1-2 sentences: do its recent activities in the news align with its stated business goals?", '
                '"positive_catalysts": ["2-4 concrete actions/events that would move the stock UP"], '
                '"negative_catalysts": ["2-4 concrete actions/events that would move the stock DOWN"], '
                '"regime_risks": "1-2 sentences on systematic/regime risks at play (rates, sector rotation, AI displacement, regulation)"'
                '}'
            )

        prompt = f"""Analyze {symbols_str} using the following data:{as_of_line}
{financial_context if financial_context else "(No financial data available — analysis limited to news only)"}
{validated_context}
NEWS ARTICLES:
{article_text}

{sentiment_summary}

BREVITY IS A HARD CONSTRAINT: the sentence counts below are maximums, not
targets. When analyzing multiple symbols, summarize across them — do not
narrate each symbol separately. Total response under 450 words.

Respond with this exact JSON structure (no markdown, no extra text):

{{"recommendation": "BULLISH|CAUTIOUS_BULLISH|NEUTRAL|CAUTIOUS_BEARISH|BEARISH", "confidence": 0.0-1.0, "key_developments": "3-4 sentences covering price action, key technicals, and most important news. Reference specific numbers.", "developments_read": "1-2 sentences: what those developments MEAN for a position holder — interpretation, not a restatement of the numbers.", "market_sentiment": "BULLISH|NEUTRAL|BEARISH", "sentiment_explanation": "One sentence on how news sentiment aligns or conflicts with the technical/fundamental picture.", "risk_factors": "1-2 key risks from the data (valuation, volatility, technical weakness, negative news).", "risks_read": "One sentence: which single risk is most live right now and what would trigger it.", "watch_items": ["2-3 short, concrete things to monitor next (a level, a date, a metric) — each checkable"]{thesis_schema}}}"""

        # Compilation provenance — computed from what was actually assembled
        # (never model-asserted) and stamped on every outcome including the
        # sentiment fallback, so readers always see how the analysis was built.
        src_meta = {
            "articles": len(articles),
            "validated_blocks": sorted(k for k in (extra_blocks or {}) if (extra_blocks or {}).get(k)),
            "financial_data": bool(financial_context),
            "as_of": as_of_date,
            "analysis_tier": "news_summary",
        }

        # Reasoning models (Luna) bill reasoning as output tokens; passing
        # reasoning_effort routes generate() through its max_completion_tokens
        # headroom logic so the JSON actually closes.
        gen_kwargs: dict = {}
        if (model or "").startswith("gpt-"):
            gen_kwargs["reasoning_effort"] = "medium"
            provider = provider or "openai"

        # Measured three times: without an explicit brevity constraint the
        # model expands to fill any cap (truncated at 1000, 1600, AND 2400
        # on a 5-symbol overall call), and a truncated JSON silently degrades
        # to the sentiment fallback. The prompt now hard-caps ~450 words
        # (~700 tokens); 3600/3200 is genuine headroom, not a target.
        response = self.generate(
            prompt, system_prompt,
            max_tokens=3600 if include_thesis else 3200,
            temperature=0.3,
            model=model,
            provider=provider,
            **gen_kwargs,
        )

        if response:
            try:
                # Clean and parse JSON using regex for robustness
                clean = response.strip()

                # Use regex to extract JSON object from response
                # This handles markdown code blocks and extra text
                match = re.search(r'\{.*\}', clean, re.DOTALL)
                if match:
                    clean = match.group(0)

                result = json.loads(clean)

                # Validate and normalize the result
                valid_recommendations = ["BULLISH", "CAUTIOUS_BULLISH", "NEUTRAL", "CAUTIOUS_BEARISH", "BEARISH"]
                rec = result.get("recommendation", "NEUTRAL").upper().replace(" ", "_")
                if rec not in valid_recommendations:
                    rec = "NEUTRAL"
                result["recommendation"] = rec

                # Ensure confidence is a float between 0 and 1
                conf = result.get("confidence", 0.5)
                if isinstance(conf, str):
                    try:
                        conf = float(conf)
                    except ValueError:
                        conf = 0.5
                result["confidence"] = max(0.0, min(1.0, conf))

                result["model_used"] = model or self._get_model()
                result["provider_used"] = provider or self.provider
                result["sources"] = src_meta
                return result

            except (json.JSONDecodeError, KeyError, TypeError) as e:
                logger.warning(f"Error parsing LLM response: {e}")
                # Return a fallback based on sentiment counts
                fb = self._fallback_analysis(sentiment_counts, articles)
                if fb:
                    fb["sources"] = {**src_meta, "analysis_tier": "sentiment_fallback"}
                return fb

        fb = self._fallback_analysis(sentiment_counts, articles)
        if fb:
            fb["sources"] = {**src_meta, "analysis_tier": "sentiment_fallback"}
        return fb

    def _fallback_analysis(
        self,
        sentiment_counts: dict,
        articles: list[dict],
    ) -> dict:
        """Generate fallback analysis when LLM fails or is unavailable.

        Args:
            sentiment_counts: Dictionary with bullish/neutral/bearish counts.
            articles: List of article dictionaries.

        Returns:
            Fallback analysis dictionary.
        """
        total = sum(sentiment_counts.values())
        if total == 0:
            return {
                "recommendation": "NEUTRAL",
                "confidence": 0.3,
                "key_developments": "No news articles available for analysis.",
                "market_sentiment": "NEUTRAL",
                "sentiment_explanation": "Insufficient data to determine sentiment.",
            }

        bullish_pct = sentiment_counts["bullish"] / total
        bearish_pct = sentiment_counts["bearish"] / total

        # Determine recommendation based on sentiment ratio
        if bullish_pct > 0.6:
            rec = "BULLISH"
            sentiment = "BULLISH"
            confidence = min(0.9, 0.5 + bullish_pct * 0.4)
        elif bullish_pct > 0.4:
            rec = "CAUTIOUS_BULLISH"
            sentiment = "BULLISH"
            confidence = 0.6
        elif bearish_pct > 0.6:
            rec = "BEARISH"
            sentiment = "BEARISH"
            confidence = min(0.9, 0.5 + bearish_pct * 0.4)
        elif bearish_pct > 0.4:
            rec = "CAUTIOUS_BEARISH"
            sentiment = "BEARISH"
            confidence = 0.6
        else:
            rec = "NEUTRAL"
            sentiment = "NEUTRAL"
            confidence = 0.5

        # Get top headlines for key developments
        top_titles = [a.get("title", "") for a in articles[:3] if a.get("title")]
        key_dev = ". ".join(top_titles[:2]) if top_titles else "Recent news coverage on these stocks."

        return {
            "recommendation": rec,
            "confidence": confidence,
            "key_developments": key_dev,
            "market_sentiment": sentiment,
            "sentiment_explanation": f"Based on {total} articles: {sentiment_counts['bullish']} bullish, {sentiment_counts['neutral']} neutral, {sentiment_counts['bearish']} bearish.",
        }

    def analyze_sentiment(
        self,
        text: str,
    ) -> Optional[dict]:
        """Analyze sentiment of text.

        Args:
            text: Text to analyze.

        Returns:
            Dictionary with sentiment analysis or None.
        """
        system_prompt = """You are a sentiment analysis expert for financial text.
Analyze the sentiment and respond ONLY with a JSON object containing:
- sentiment: "bullish", "bearish", or "neutral"
- confidence: a number between 0 and 1
- reasoning: a brief one-sentence explanation"""

        prompt = f"""Analyze the sentiment of this financial text:

"{text}"

Respond with JSON only."""

        response = self.generate(prompt, system_prompt, max_tokens=150, temperature=0.3)

        if response:
            try:
                # Try to parse JSON from response
                import json
                # Handle potential markdown code blocks
                clean_response = response.strip()
                if clean_response.startswith("```"):
                    clean_response = clean_response.split("```")[1]
                    if clean_response.startswith("json"):
                        clean_response = clean_response[4:]
                return json.loads(clean_response)
            except Exception:
                return {"sentiment": "neutral", "confidence": 0.5, "raw": response}

        return None

    def generate_market_insight(
        self,
        symbol: str,
        price_data: dict,
        signals: dict,
        news_sentiment: Optional[str] = None,
    ) -> Optional[str]:
        """Generate comprehensive market insight.

        Args:
            symbol: Stock symbol.
            price_data: Dictionary with price metrics.
            signals: Dictionary with technical signals.
            news_sentiment: Optional news sentiment summary.

        Returns:
            AI-generated market insight or None.
        """
        system_prompt = """You are a professional market analyst.
Provide clear, concise insights based on technical and fundamental data.
Avoid making specific predictions or recommendations.
Focus on factual observations and key levels to watch."""

        # Build context
        price_context = f"""
Symbol: {symbol}
Current Price: ${price_data.get('end_price', 'N/A')}
1Y Return: {price_data.get('total_return', 'N/A')}%
Volatility: {price_data.get('volatility', 'N/A')}%
Max Drawdown: {price_data.get('max_drawdown', 'N/A')}%
"""

        signal_context = ""
        if signals:
            signal_lines = []
            for key, val in signals.items():
                if isinstance(val, dict):
                    signal_lines.append(f"- {key}: {val.get('signal', str(val))}")
            signal_context = "\nTechnical Signals:\n" + "\n".join(signal_lines)

        news_context = ""
        if news_sentiment:
            news_context = f"\nNews Sentiment: {news_sentiment}"

        prompt = f"""Based on the following data, provide a brief market insight for {symbol}:

{price_context}{signal_context}{news_context}

Provide a 3-4 sentence analysis covering:
1. Current technical position
2. Key levels to watch
3. Notable observations"""

        return self.generate(prompt, system_prompt, max_tokens=400)

    # Model description constant — extend when new models are added
    _MODEL_DESCRIPTIONS = {
        "kronos_mini": "Kronos: Time-series probabilistic forecasting using Monte Carlo sampling of future price trajectories",
        "xgboost_shap": "XGBoost: Gradient-boosted decision trees on 18 engineered features (technicals + SPY correlation + news sentiment)",
        "lightgbm": "LightGBM: Alternative gradient-boosted model with leaf-wise growth (same feature set, different learning bias)",
        "deberta_sentiment": "DeBERTa: Transformer NLP model analyzing financial news article sentiment",
        "trading_agents": "TradingAgents: LLM-based research agent that synthesizes technicals, fundamentals, and news into a single trade decision",
        "ensemble": "Ensemble: Configurable weighted majority vote across enabled models",
    }

    def generate_recommendations(
        self,
        ai_analysis: dict,
        model_signals: dict,
        symbols: list[str],
        basis: str = "news+signals",
        model_override: Optional[str] = None,
    ) -> Optional[dict]:
        """Synthesize the chosen evidence into actionable recommendations.

        Uses the configured recommendations model (default: GPT 5.6 Luna).
        `basis` controls what evidence goes into the prompt and is stamped
        onto the result so History records what a recommendation was built
        from: "research+signals" (verdict-first research reports),
        "news+signals" (shallow analysis), or "signals" (predictions only —
        no text analysis at all).
        """
        if not symbols:
            return None
        signals_only = basis == "signals"

        model_desc_block = "\n".join(
            f"  - {desc}" for desc in self._MODEL_DESCRIPTIONS.values()
        )

        system_prompt = f"""You are a senior portfolio strategist. You receive two independent analyses:

1. AI REPORT: News-based sentiment analysis with recommendation, confidence, key developments, and risk factors per symbol.
2. MODEL PREDICTIONS: Machine learning model outputs (BUY/SELL/HOLD with confidence scores) from multiple quantitative models.

AVAILABLE MODELS:
{model_desc_block}

YOUR ROLE:
- Synthesize both inputs into specific, actionable recommendations per symbol
- When the AI report and model predictions DISAGREE, this is the most valuable signal — explain WHY they disagree and which to trust in this context
- Explain which models are most relevant for each symbol's situation
- Provide an overall portfolio-level summary
- Be direct and specific. No hedging.

Respond with ONLY valid JSON matching this schema:
{{
  "overall": {{
    "summary": "2-3 sentence portfolio overview",
    "portfolio_action": "One-line actionable directive",
    "key_conflicts": ["List of notable disagreements between report and models"],
    "risk_assessment": "1-2 sentences on aggregate risk",
    "watch_items": ["2-3 short portfolio-level things to monitor next — each a checkable level, date, or metric"]
  }},
  "by_symbol": {{
    "<SYMBOL>": {{
      "action": "BUY|SELL|HOLD",
      "conviction": "HIGH|MEDIUM|LOW",
      "reasoning": "2-3 sentences explaining the recommendation, referencing specific data",
      "key_level": "The single price level that matters most for this call (from the data, e.g. 'support $59.35')",
      "change_trigger": "One concrete condition that would flip this action",
      "conflicts": ["Any disagreements between report sentiment and model signals for this symbol"],
      "model_notes": "Which models to weight most for this symbol and why"
    }}
  }}
}}"""

        # Build per-symbol data block
        symbol_blocks = []
        analysis_by_sym = ai_analysis.get("by_symbol", {})
        overall = ai_analysis.get("overall", {})

        for symbol in symbols:
            lines = [f"=== {symbol} ==="]

            # AI Report data (suppressed entirely on a signals-only basis)
            sym_report = {} if signals_only else analysis_by_sym.get(symbol, {})
            research = (sym_report.get("research") or {}) if sym_report else {}
            if research.get("raw_response"):
                # The verdict block is the distilled research call — triggers
                # and headline reasoning in ~600 chars.
                verdict = research["raw_response"]
                cut = verdict.find("\n### ")
                lines.append("Research report verdict:")
                lines.append("  " + verdict[:cut if 0 < cut <= 900 else 600]
                              .strip().replace("\n", "\n  "))
            if sym_report:
                lines.append(f"AI Report: {sym_report.get('recommendation', 'N/A')} "
                             f"(confidence: {sym_report.get('confidence', 'N/A')})")
                lines.append(f"  Sentiment: {sym_report.get('market_sentiment', 'N/A')}")
                kd = sym_report.get("key_developments", "")
                if kd:
                    lines.append(f"  Key developments: {kd[:300]}")
                rf = sym_report.get("risk_factors", "")
                if rf:
                    lines.append(f"  Risk factors: {rf[:200]}")
                thesis = sym_report.get("company_thesis") or {}
                if thesis:
                    if thesis.get("perception"):
                        lines.append(f"  Market perception: {str(thesis['perception'])[:250]}")
                    if thesis.get("regime_risks"):
                        lines.append(f"  Regime/systematic risks: {str(thesis['regime_risks'])[:200]}")
            else:
                lines.append("AI Report: Not available")

            # Model prediction data
            sym_signals = model_signals.get(symbol, {})
            if sym_signals:
                for model_name, result in sym_signals.items():
                    if not isinstance(result, dict) or model_name.startswith("_"):
                        continue
                    decision = result.get("decision", "N/A")
                    display = self._MODEL_DESCRIPTIONS.get(model_name, model_name).split(":")[0]
                    # A model may legitimately decline to state either number
                    # (the research arm publishes no up-probability until it
                    # has a track record, and a signal read back from storage
                    # carries NULL rather than a default). `.get(k, default)`
                    # does not cover that — the key is present and None — and
                    # formatting None raised, taking the whole synthesis down.
                    # Say "n/a" instead of inventing a 50%.
                    confidence = result.get("confidence")
                    up_prob = result.get("up_probability")
                    conf_str = f"{confidence:.0%}" if confidence is not None else "n/a"
                    up_str = f"{up_prob:.0%}" if up_prob is not None else "n/a"
                    lines.append(f"  {display}: {decision} (conf: {conf_str}, up: {up_str})")
                    triggers = (result.get("details") or {}).get("triggers") or {}
                    if triggers.get("reassess_to_buy"):
                        lines.append(f"    TA reassess-to-BUY trigger: {triggers['reassess_to_buy'][:150]}")
                    if triggers.get("move_to_sell"):
                        lines.append(f"    TA move-to-SELL trigger: {triggers['move_to_sell'][:150]}")
            else:
                lines.append("Model Predictions: Not available")

            symbol_blocks.append("\n".join(lines))

        # Overall AI report
        overall_line = ""
        if overall:
            overall_line = (
                f"\nOVERALL AI REPORT: {overall.get('recommendation', 'N/A')} "
                f"(confidence: {overall.get('confidence', 'N/A')})\n"
                f"  Sentiment: {overall.get('market_sentiment', 'N/A')}\n"
            )

        user_prompt = (
            f"Analyze the following {len(symbols)} symbols and provide recommendations:\n\n"
            + overall_line + "\n"
            + "\n\n".join(symbol_blocks)
        )

        # Caller-selected synthesis model (modal); config default otherwise.
        primary_model = model_override or MODEL.RECOMMENDATIONS_MODEL
        primary_provider = ("openai" if primary_model.startswith("gpt-")
                            else "anthropic")
        primary_kwargs = {}
        if primary_provider == "openai":
            primary_kwargs["reasoning_effort"] = MODEL.RECOMMENDATIONS_REASONING_EFFORT

        logger.info(
            f"Generating recommendations for {len(symbols)} symbols "
            f"via {primary_provider}/{primary_model}"
        )

        raw = self.generate(
            user_prompt,
            system_prompt,
            max_tokens=MODEL.RECOMMENDATIONS_MAX_TOKENS,
            temperature=MODEL.RECOMMENDATIONS_TEMPERATURE,
            model=primary_model,
            provider=primary_provider,
            **primary_kwargs,
        )
        synthesis_model = primary_model

        if not raw and API.ANTHROPIC_API_KEY:
            # Primary synthesis model failed even after retries — a completed
            # AI report + prediction run shouldn't be wasted. Fall back.
            fallback = os.getenv("RECOMMENDATIONS_FALLBACK_MODEL", "claude-sonnet-4-6")
            if fallback == primary_model:
                fallback = "claude-sonnet-5"
            logger.warning(
                f"{primary_model} failed — falling back to {fallback}"
            )
            raw = self.generate(
                user_prompt,
                system_prompt,
                max_tokens=MODEL.RECOMMENDATIONS_MAX_TOKENS,
                temperature=MODEL.RECOMMENDATIONS_TEMPERATURE,
                model=fallback,
                provider="anthropic",
            )
            synthesis_model = fallback

        if not raw:
            logger.warning("Recommendations model returned empty response")
            return None

        try:
            clean = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`")
            match = re.search(r"\{.*\}", clean, re.DOTALL)
            if not match:
                logger.warning("No JSON found in recommendations response")
                return None
            parsed = json.loads(match.group())
            if "overall" not in parsed or "by_symbol" not in parsed:
                logger.warning("Recommendations JSON missing required keys")
                return None
            parsed["model_used"] = synthesis_model
            parsed["provider_used"] = ("anthropic"
                                       if synthesis_model != MODEL.RECOMMENDATIONS_MODEL
                                       else MODEL.RECOMMENDATIONS_PROVIDER)
            parsed["basis"] = basis
            return parsed
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"Failed to parse recommendations: {e}")
            return None


# Singleton instance
_llm_instance: Optional[LLMService] = None
_llm_instance_lock = threading.Lock()


def get_llm() -> LLMService:
    """Get the singleton LLM service instance (thread-safe lazy init).

    Returns:
        LLMService instance.
    """
    global _llm_instance
    if _llm_instance is None:
        with _llm_instance_lock:
            if _llm_instance is None:
                _llm_instance = LLMService()
    return _llm_instance

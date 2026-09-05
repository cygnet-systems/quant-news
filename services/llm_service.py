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


# OpenAI model families that took `max_tokens` away in favour of
# `max_completion_tokens`: the whole gpt-5 line and the o-series reasoners.
# Sending the old name is a hard 400, so the choice belongs to the MODEL.
_COMPLETION_TOKEN_MODELS = ("gpt-5", "o1", "o3", "o4")


def _uses_completion_tokens(model: str) -> bool:
    """True if `model` wants max_completion_tokens rather than max_tokens."""
    return any(model.startswith(p) for p in _COMPLETION_TOKEN_MODELS)


def _first_text(response) -> Optional[str]:
    """Extract the first text block, skipping thinking blocks."""
    for block in response.content:
        if block.type == "text":
            return block.text
    return None


def _record_usage(usage_out: Optional[dict], response, provider: str,
                  model: str, duration_ms: Optional[int] = None) -> Optional[int]:
    """Normalise a response's token counts, then fan them out two ways.

    1. ``usage_out``, when a caller passed one. An out-parameter rather than a
       changed return type, because ``generate()`` returns a plain string to a
       couple of dozen call sites and only the report path wants the counts.
    2. The durable ``llm_usage`` table, ALWAYS. Cost telemetry that depended on
       callers opting in would only ever measure the paths someone remembered
       to instrument; this is the one funnel every provider path returns
       through, so recording here means no call goes uncounted.

    Anthropic reports input_tokens/output_tokens; the OpenAI-compatible path
    reports prompt_tokens/completion_tokens. Both are normalised here so
    callers never branch on provider. Never raises: telemetry must not be
    able to fail a generation that already succeeded.

    Returns the llm_usage row id (or None) so the trace row for the same
    physical call can link to its cost record.
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
    return usage_service.record(
        model=model, provider=provider,
        input_tokens=in_tok, output_tokens=out_tok,
        duration_ms=duration_ms,
    )


def _trace_params(api_kwargs: dict) -> dict:
    """The request parameters as sent, minus the bodies (those get their own
    Text columns) and minus the model (its own column too)."""
    return {k: v for k, v in api_kwargs.items()
            if k not in ("messages", "system", "model")}


def _calibrated(model_name: str, raw_confidence) -> Optional[float]:
    """What this model's raw confidence has historically resolved to, or None.

    The synthesis prompt shows this beside the raw score so the strategist can
    see that a 92% from a gradient-boosted model is not a 92% chance of being
    right. None (too little evaluated history) is stated as such, never filled.
    """
    try:
        from services.calibration_service import calibrate
        return calibrate(model_name, float(raw_confidence))
    except Exception:
        return None


# Default model per provider, single source of truth for the three copies of
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
    atomic reference operations, methods must never mutate client/provider
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

        Clients get an explicit timeout. The SDK default of 600s pins a
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
        there isn't enough. LM Studio still 400s on every completion with
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
                      actually produced the text, including after a failover,
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

        # Trace bookkeeping: one llm_traces row per PHYSICAL API call,
        # retries and failover attempts included. `last_call` describes the
        # attempt in flight; `open` distinguishes "raised before its trace was
        # written" (the outer except records it) from "already traced".
        attempt_n = 0
        last_call: dict = {}

        def _begin_attempt(provider_: str, model_: str, api_kwargs: dict) -> None:
            nonlocal attempt_n
            attempt_n += 1
            last_call.update(provider=provider_, model=model_,
                             params=_trace_params(api_kwargs),
                             t0=_time.time(), open=True)

        def _trace(response_text: Optional[str], *, ok: bool = True,
                   error: Optional[str] = None,
                   usage_id: Optional[int] = None) -> None:
            """Record the in-flight attempt. Captured before any parsing, so
            a truncated/unparseable response is preserved verbatim. Never
            breaks the call path, record_llm_call swallows its own errors."""
            last_call["open"] = False
            try:
                from services.trace_service import record_llm_call
                record_llm_call(
                    provider=last_call.get("provider"),
                    model=last_call.get("model"),
                    system_prompt=system_prompt, prompt=prompt,
                    response=response_text, params=last_call.get("params"),
                    attempt=attempt_n, ok=ok, error=error,
                    duration_ms=int((_time.time()
                                     - last_call.get("t0", _call_t0)) * 1000),
                    usage_id=usage_id,
                )
            except Exception as trace_err:  # pragma: no cover - best-effort
                logger.debug("llm trace failed: %s", trace_err)

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

                _begin_attempt("anthropic", use_model, api_kwargs)
                try:
                    response = use_client.messages.create(**api_kwargs)
                except TypeError as e:
                    # The SDK, not the API, rejected a parameter: an installed
                    # anthropic version whose messages.create() has no such
                    # keyword. This is the fallback path's own failure mode --
                    # on 2026-09-04 the OpenAI account ran out of credits, the
                    # recommendations synthesis fell back to Anthropic exactly
                    # as designed, and the fallback died on
                    # "Messages.create() got an unexpected keyword argument
                    # 'temperature'", losing the whole day's synthesis to the
                    # safety net rather than to the outage. Sampling params are
                    # a preference; the answer is not. Drop them and re-ask.
                    dropped = [k for k in ("temperature", "top_p", "top_k")
                               if k in api_kwargs]
                    if not dropped or "unexpected keyword" not in str(e):
                        raise
                    logger.warning(
                        "anthropic SDK rejected %s (%s); retrying without "
                        "sampling parameters", ", ".join(dropped), e)
                    for k in dropped:
                        api_kwargs.pop(k, None)
                    response = use_client.messages.create(**api_kwargs)
                if getattr(response, "stop_reason", None) == "max_tokens":
                    # Truncation was previously silent; on report prompts the
                    # decision footer is the first casualty, so make it loud.
                    logger.warning(
                        f"LLM response truncated at max_tokens={max_tokens} "
                        f"(model={use_model}): output is incomplete"
                    )
                usage_id = _record_usage(usage_out, response, "anthropic",
                                         use_model, duration_ms=_elapsed_ms())
                text = _first_text(response)
                _trace(text, usage_id=usage_id)
                return text

            # OpenAI-compatible path (LM Studio, OpenAI)
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            api_kwargs = {
                "model": use_model,
                "messages": messages,
            }
            # Which token parameter this model accepts is a property of the
            # MODEL, not of whether this particular caller asked for an
            # effort. Keying it off reasoning_effort meant every gpt-5.6-luna
            # call that did not pass one sent `max_tokens` and took a hard
            # 400 ("Unsupported parameter: 'max_tokens' is not supported with
            # this model"). That was invisible while the account was out of
            # credits -- the 429 came first -- and surfaced the moment
            # billing was restored on 2026-09-04.
            effort = kwargs.get("reasoning_effort") if use_provider == "openai" else None
            if _uses_completion_tokens(use_model) or effort:
                # Reasoning models spend completion budget on reasoning FIRST.
                # max_completion_tokens == desired content size returns HTTP 200
                # with EMPTY content once reasoning eats the budget, give
                # headroom on top of the requested content size.
                if effort == "max":
                    effort = "high"
                headroom = {"low": 2, "medium": 4, "high": 8}.get(effort, 4)
                api_kwargs["max_completion_tokens"] = max(max_tokens * headroom, 16000)
                if effort:
                    api_kwargs["reasoning_effort"] = effort
                # These models reject temperature the same way they reject
                # max_tokens, so it is not sent here either.
            else:
                api_kwargs["max_tokens"] = max_tokens
                api_kwargs["temperature"] = temperature

            # Transient errors (401 permission flaps, 429, 5xx) shouldn't kill
            # a whole pipeline, retry once with a short backoff.
            _begin_attempt(use_provider, use_model, api_kwargs)
            try:
                response = use_client.chat.completions.create(**api_kwargs)
            except Exception as api_err:
                msg = str(api_err)
                transient = any(t in msg for t in ("401", "429", "500", "502", "503",
                                                   "insufficient permissions", "overloaded"))
                if not transient:
                    raise
                _trace(None, ok=False, error=msg)
                logger.warning(f"Transient OpenAI error, retrying in 3s: {msg[:120]}")
                _time.sleep(3)
                _begin_attempt(use_provider, use_model, api_kwargs)
                response = use_client.chat.completions.create(**api_kwargs)

            content = response.choices[0].message.content

            # Empty content despite HTTP 200 = reasoning consumed the budget.
            # Retry once at low effort with a large budget rather than failing.
            # Keyed on the request actually built the reasoning way, which is
            # what "the budget was spent thinking" means; a caller that passed
            # no effort can still be on a model that reasons by default.
            if "max_completion_tokens" in api_kwargs and not (content or "").strip():
                finish = getattr(response.choices[0], "finish_reason", "?")
                logger.warning(
                    f"Reasoning model returned empty content (finish_reason={finish}): "
                    f"retrying with low effort"
                )
                # The empty answer is itself evidence, keep its row.
                _trace(content, error=f"empty content (finish_reason={finish})")
                if "reasoning_effort" in api_kwargs:
                    api_kwargs["reasoning_effort"] = "low"
                api_kwargs["max_completion_tokens"] = max(max_tokens * 4, 20000)
                _begin_attempt(use_provider, use_model, api_kwargs)
                response = use_client.chat.completions.create(**api_kwargs)
                content = response.choices[0].message.content

            usage_id = _record_usage(usage_out, response, use_provider,
                                     use_model, duration_ms=_elapsed_ms())
            _trace(content, usage_id=usage_id)
            return content

        except Exception as e:
            logger.warning(f"LLM generation error ({use_provider}): {e}")

            # The attempt that raised has not been traced yet (the success
            # paths close their own), record it before anything else so
            # failover attempts stack on top with incremented numbers.
            if last_call.get("open"):
                _trace(None, ok=False, error=str(e))

            # Auto-failover only when using default provider (not overrides).
            # Work entirely in locals; publish the new active pair only after
            # a success, as one atomic tuple swap.
            if not provider and use_provider == "lm_studio":
                logger.info("LM Studio failed, attempting cloud failover")

                if API.ANTHROPIC_API_KEY:
                    try:
                        fo_client, _ = self._get_client_for_provider("anthropic")
                        fo_kwargs = {"model": _DEFAULT_MODELS["anthropic"],
                                     "max_tokens": max_tokens,
                                     "temperature": temperature}
                        _begin_attempt("anthropic", _DEFAULT_MODELS["anthropic"],
                                       fo_kwargs)
                        result = self._generate_anthropic(
                            fo_client, prompt, system_prompt, max_tokens, temperature,
                            usage_out=usage_out,
                        )
                        self._active = (fo_client, "anthropic")
                        logger.info("Failover to Anthropic succeeded, keeping provider")
                        _trace(result)
                        return result
                    except Exception as e2:
                        if last_call.get("open"):
                            _trace(None, ok=False, error=str(e2))
                        logger.warning(f"Anthropic failover failed: {e2}")

                if API.OPENAI_API_KEY:
                    try:
                        fo_client, _ = self._get_client_for_provider("openai")
                        messages = []
                        if system_prompt:
                            messages.append({"role": "system", "content": system_prompt})
                        messages.append({"role": "user", "content": prompt})
                        fo_kwargs = {
                            "model": _DEFAULT_MODELS["openai"],
                            "messages": messages,
                            "max_tokens": max_tokens,
                            "temperature": temperature,
                        }
                        _begin_attempt("openai", _DEFAULT_MODELS["openai"],
                                       fo_kwargs)
                        response = fo_client.chat.completions.create(**fo_kwargs)
                        self._active = (fo_client, "openai")
                        logger.info("Failover to OpenAI succeeded, keeping provider")
                        usage_id = _record_usage(usage_out, response, "openai",
                                                 _DEFAULT_MODELS["openai"])
                        content = response.choices[0].message.content
                        _trace(content, usage_id=usage_id)
                        return content
                    except Exception as e3:
                        if last_call.get("open"):
                            _trace(None, ok=False, error=str(e3))
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

    def generate_with_web_search(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        *,
        model: Optional[str] = None,
        max_tokens: int = 6000,
        max_searches: int = 8,
        allowed_domains: Optional[list[str]] = None,
        usage_out: Optional[dict] = None,
        max_rounds: int = 4,
    ) -> dict:
        """One Anthropic call with the server-side web_search tool.

        The model decides what to search; Anthropic runs the searches and
        returns results and citations inside the same response. When the
        server pauses a long tool loop (``stop_reason == "pause_turn"``) the
        accumulated content is sent back and the turn continues, up to
        ``max_rounds`` physical calls. Every physical call is recorded in
        llm_usage and llm_traces like any other.

        Returns {"text", "searches", "sources": [{url, title}], "stop_reason",
        "model"}; raises on provider errors (the caller decides whether the
        block is required, expected or optional). The provider follows the
        model name: gpt-* runs on OpenAI's Responses API with its hosted
        web_search tool, anything else on Anthropic's web_search server
        tool. Both do the browsing on the provider's side.
        """
        use_model = model or MODEL.INVESTIGATION_MODEL
        if use_model.startswith("gpt-"):
            return self._web_search_openai(
                prompt, system_prompt, model=use_model, max_tokens=max_tokens,
                max_searches=max_searches, usage_out=usage_out)
        client, use_provider = self._get_client_for_provider("anthropic")
        if client is None:
            raise RuntimeError("web research needs an Anthropic API key")

        tools = [{"type": "web_search_20260209", "name": "web_search",
                  "max_uses": int(max_searches)}]
        if allowed_domains:
            tools[0]["allowed_domains"] = list(allowed_domains)

        import time as _time
        messages: list = [{"role": "user", "content": prompt}]
        text_parts: list[str] = []
        sources: dict[str, str] = {}
        searches = 0
        total_in = total_out = 0
        stop_reason = None

        for round_n in range(1, max_rounds + 1):
            api_kwargs: dict = {
                "model": use_model,
                "max_tokens": max_tokens,
                "messages": messages,
                "tools": tools,
            }
            if system_prompt:
                api_kwargs["system"] = system_prompt
            if not _rejects_sampling_params(use_model):
                api_kwargs["temperature"] = 0.2
            t0 = _time.time()
            try:
                # Streamed, with its own timeout: a tool loop with several
                # searches runs minutes, past the 180s the plain calls use,
                # and a non-streaming request that long trips the SDK's
                # HTTP timeout. The final message is assembled by the SDK.
                with client.with_options(
                        timeout=API.INVESTIGATION_TIMEOUT).messages.stream(
                        **api_kwargs) as stream:
                    response = stream.get_final_message()
            except Exception as e:
                try:
                    from services.trace_service import record_llm_call
                    record_llm_call(
                        provider="anthropic", model=use_model,
                        system_prompt=system_prompt, prompt=prompt,
                        response=None, params=_trace_params(api_kwargs),
                        attempt=round_n, ok=False, error=str(e),
                        duration_ms=int((_time.time() - t0) * 1000))
                except Exception:
                    pass
                raise

            stop_reason = getattr(response, "stop_reason", None)
            round_text: list[str] = []
            for block in response.content:
                btype = getattr(block, "type", "")
                if btype == "text":
                    round_text.append(block.text)
                    for cit in (getattr(block, "citations", None) or []):
                        url = getattr(cit, "url", None)
                        if url:
                            sources.setdefault(url, getattr(cit, "title", "") or "")
                elif btype == "server_tool_use":
                    # The dynamic-filtering search variant also runs code
                    # execution server-side; only count the searches.
                    if getattr(block, "name", "") == "web_search":
                        searches += 1
                elif btype == "web_search_tool_result":
                    content = getattr(block, "content", None)
                    # A list is results; an object is an error envelope.
                    if isinstance(content, list):
                        for r in content:
                            url = getattr(r, "url", None)
                            if url:
                                sources.setdefault(
                                    url, getattr(r, "title", "") or "")
                    else:
                        code = getattr(content, "error_code", None)
                        if code:
                            round_text.append(f"[web search error: {code}]")
            text_parts.extend(round_text)

            usage = getattr(response, "usage", None)
            in_tok = int(getattr(usage, "input_tokens", 0) or 0)
            out_tok = int(getattr(usage, "output_tokens", 0) or 0)
            total_in += in_tok
            total_out += out_tok
            duration_ms = int((_time.time() - t0) * 1000)
            from services import usage_service
            usage_id = usage_service.record(
                model=use_model, provider="anthropic",
                input_tokens=in_tok, output_tokens=out_tok,
                duration_ms=duration_ms)
            try:
                from services.trace_service import record_llm_call
                record_llm_call(
                    provider="anthropic", model=use_model,
                    system_prompt=system_prompt,
                    prompt=prompt if round_n == 1 else
                    f"[continuation round {round_n} after pause_turn]",
                    response="\n".join(round_text) or None,
                    params={**_trace_params(api_kwargs),
                            "searches_so_far": searches,
                            "stop_reason": stop_reason},
                    attempt=round_n, ok=stop_reason != "refusal",
                    error=(f"refusal: {getattr(getattr(response, 'stop_details', None), 'category', None)}"
                           if stop_reason == "refusal" else None),
                    duration_ms=duration_ms, usage_id=usage_id)
            except Exception as trace_err:  # pragma: no cover
                logger.debug("web-search trace failed: %s", trace_err)

            if stop_reason == "pause_turn":
                messages.append({"role": "assistant",
                                 "content": response.content})
                continue
            break

        if usage_out is not None:
            usage_out.update({"input_tokens": total_in,
                              "output_tokens": total_out,
                              "model": use_model, "provider": "anthropic"})
        if stop_reason == "refusal":
            raise RuntimeError("web research call was refused by the provider")
        if stop_reason == "max_tokens":
            logger.warning(f"web research truncated at max_tokens={max_tokens} "
                           f"(model={use_model})")
        return {
            "text": "\n".join(p for p in text_parts if p),
            "searches": searches,
            "sources": [{"url": u, "title": t} for u, t in sources.items()],
            "stop_reason": stop_reason,
            "model": use_model,
        }

    def _web_search_openai(
        self,
        prompt: str,
        system_prompt: Optional[str],
        *,
        model: str,
        max_tokens: int,
        max_searches: int,
        usage_out: Optional[dict] = None,
    ) -> dict:
        """OpenAI Responses API with the hosted ``web_search`` tool.

        One request; the model searches as it sees fit (there is no per-call
        search cap in the tool, so the prompt's own discipline and
        ``max_tool_calls`` bound it). Citations arrive as ``url_citation``
        annotations on the message; search calls as ``web_search_call``
        output items. Recorded in llm_usage/llm_traces like every call.
        """
        client, _ = self._get_client_for_provider("openai")
        if client is None:
            raise RuntimeError("web research on a gpt-* model needs an OpenAI API key")
        import time as _time
        api_kwargs: dict = {
            "model": model,
            "input": prompt,
            "tools": [{"type": "web_search"}],
            "max_tool_calls": int(max_searches),
            "max_output_tokens": max_tokens,
            "reasoning": {"effort": MODEL.INVESTIGATION_OPENAI_EFFORT},
        }
        if system_prompt:
            api_kwargs["instructions"] = system_prompt
        t0 = _time.time()
        try:
            response = client.with_options(
                timeout=API.INVESTIGATION_TIMEOUT).responses.create(**api_kwargs)
        except Exception as e:
            try:
                from services.trace_service import record_llm_call
                record_llm_call(
                    provider="openai", model=model, system_prompt=system_prompt,
                    prompt=prompt, response=None,
                    params={k: v for k, v in api_kwargs.items()
                            if k not in ("input", "instructions", "model")},
                    attempt=1, ok=False, error=str(e),
                    duration_ms=int((_time.time() - t0) * 1000))
            except Exception:
                pass
            raise

        searches = 0
        sources: dict[str, str] = {}
        text_parts: list[str] = []
        for item in response.output:
            itype = getattr(item, "type", "")
            if itype == "web_search_call":
                searches += 1
            elif itype == "message":
                for part in getattr(item, "content", None) or []:
                    if getattr(part, "type", "") == "output_text":
                        text_parts.append(part.text)
                        for ann in getattr(part, "annotations", None) or []:
                            url = getattr(ann, "url", None)
                            if url:
                                sources.setdefault(url, getattr(ann, "title", "") or "")
        status = getattr(response, "status", None)
        incomplete = getattr(response, "incomplete_details", None)
        stop_reason = (getattr(incomplete, "reason", None) if incomplete
                       else ("end_turn" if status == "completed" else status))
        usage = getattr(response, "usage", None)
        in_tok = int(getattr(usage, "input_tokens", 0) or 0)
        out_tok = int(getattr(usage, "output_tokens", 0) or 0)
        duration_ms = int((_time.time() - t0) * 1000)
        from services import usage_service
        usage_id = usage_service.record(model=model, provider="openai",
                                        input_tokens=in_tok, output_tokens=out_tok,
                                        duration_ms=duration_ms)
        try:
            from services.trace_service import record_llm_call
            record_llm_call(
                provider="openai", model=model, system_prompt=system_prompt,
                prompt=prompt, response="\n".join(text_parts) or None,
                params={k: v for k, v in api_kwargs.items()
                        if k not in ("input", "instructions", "model")}
                | {"searches": searches, "stop_reason": stop_reason},
                attempt=1, ok=bool(text_parts), duration_ms=duration_ms,
                usage_id=usage_id)
        except Exception as trace_err:  # pragma: no cover
            logger.debug("web-search trace failed: %s", trace_err)
        if usage_out is not None:
            usage_out.update({"input_tokens": in_tok, "output_tokens": out_tok,
                              "model": model, "provider": "openai"})
        if stop_reason == "max_output_tokens":
            logger.warning(f"web research truncated at max_output_tokens={max_tokens} "
                           f"(model={model})")
        return {"text": "\n".join(text_parts), "searches": searches,
                "sources": [{"url": u, "title": t} for u, t in sources.items()],
                "stop_reason": stop_reason, "model": model}

    def _generate_anthropic(
        self,
        client,
        prompt: str,
        system_prompt: Optional[str],
        max_tokens: int,
        temperature: float,
        usage_out: Optional[dict] = None,
    ) -> Optional[str]:
        """Generate text using Anthropic Claude API (explicit client, used
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

        # Ten articles sampled across the window, not the newest ten.
        from services.news_window import select_spread
        article_text = "\n".join([
            f"- {str(a.get('published_at') or '')[:10]} {a.get('title', '')}: "
            f"{(a.get('summary') or '')[:200]}"
            for a in select_spread(articles, 10)
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

        # Spread across symbols AND across the window. This used to be
        # ``articles[:15]`` off a newest-first list. With a 20-symbol run
        # that was fifteen headlines, mostly one symbol's, from the last day
        # or two, whatever window the user had picked.
        from config import MODEL as _M
        from services.news_window import article_span, select_spread
        budget = _M.NEWS_SYNTHESIS_ARTICLES
        by_symbol: dict[str, list] = {}
        for a in articles:
            by_symbol.setdefault(a.get("symbol") or "?", []).append(a)
        per_symbol = (max(1, budget // len(by_symbol)) if budget else 0)
        shown: list = []
        for sym in sorted(by_symbol):
            shown.extend(select_spread(by_symbol[sym], per_symbol))
        oldest, newest = article_span(shown)
        article_text = (
            f"({len(shown)} of {len(articles)} articles shown, sampled across "
            f"each symbol's window {oldest} → {newest}; newest first per symbol)\n"
        ) + "\n".join([
            f"- {a.get('symbol') or '?'} "
            f"{str(a.get('published_at') or '?')[:10]} "
            f"[{(a.get('sentiment') or 'unknown').upper()}] "
            f"(relevance:{a.get('ticker_relevance_score', 'N/A')}, "
            f"score:{a.get('sentiment_score', 'N/A')}) "
            f"{a.get('title', '')}: {(a.get('summary') or '')[:200]}"
            for a in shown
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
                    # symbol, and the analyst model, correctly, refused to
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

        system_prompt = """You are a senior equity research analyst. Produce a grounded analysis using ALL provided data. Financial metrics, technical signals, AND news. Respond with ONLY valid JSON.

CRITICAL RULES:
- Your response must be parseable JSON with no additional text, no markdown code blocks.
- Ground your recommendation in the FINANCIAL DATA: price action, valuation (P/E), technicals (RSI, MACD, trend), and risk metrics (volatility, drawdown).
- Every number you cite must come from the provided data. Never estimate or invent a figure.
- News sentiment should CONFIRM or CHALLENGE the technical/fundamental picture, not replace it.
- Confidence is your estimated probability (0.0-1.0) that the recommendation direction is correct; high only when technicals, fundamentals, AND sentiment agree.
- If signals conflict (e.g., bullish news but overbought RSI), lower confidence and note the divergence.
- DRAWDOWN CAUSALITY: if the stock is down >20% from its period high, state a causal hypothesis with evidence from the news/fundamentals, or write "cause unknown, elevated risk".
- If a VALIDATED METRICS block is present, use its ATR/support/resistance/R:R arithmetic for any risk-reward statement instead of qualitative claims.
- Treat any VALIDATED block as the source of truth. If two data sources conflict, FLAG the discrepancy rather than inventing a reconciled number. Do not claim historical support/resistance bounces or exact percentage moves unless a data block states them with concrete dates and prices."""

        as_of_line = f"\nANALYSIS AS-OF DATE: {as_of_date}. Reason only with information available on or before this date.\n" if as_of_date else ""

        eb = extra_blocks or {}
        validated_context = ""
        for key, label in (("profile", "COMPANY PROFILE"), ("metrics", "VALIDATED METRICS"),
                           ("events", "EVENT CALENDAR"), ("peers", "PEER RELATIVE STRENGTH"),
                           ("options", "OPTIONS POSITIONING"),
                           ("quality", "QUALITY SCREEN (BAD APPLES)")):
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
{financial_context if financial_context else "(No financial data available. Analysis limited to news only)"}
{validated_context}
NEWS ARTICLES:
{article_text}

{sentiment_summary}

BREVITY IS A HARD CONSTRAINT: the sentence counts below are maximums, not
targets. When analyzing multiple symbols, summarize across them, do not
narrate each symbol separately. Total response under 450 words.

Respond with this exact JSON structure (no markdown, no extra text):

{{"recommendation": "BULLISH|CAUTIOUS_BULLISH|NEUTRAL|CAUTIOUS_BEARISH|BEARISH", "confidence": 0.0-1.0, "key_developments": "3-4 sentences covering price action, key technicals, and most important news. Reference specific numbers.", "developments_read": "1-2 sentences: what those developments MEAN for a position holder. Interpretation, not a restatement of the numbers.", "market_sentiment": "BULLISH|NEUTRAL|BEARISH", "sentiment_explanation": "One sentence on how news sentiment aligns or conflicts with the technical/fundamental picture.", "risk_factors": "1-2 key risks from the data (valuation, volatility, technical weakness, negative news).", "risks_read": "One sentence: which single risk is most live right now and what would trigger it.", "watch_items": ["2-3 short, concrete things to monitor next (a level, a date, a metric), each checkable"]{thesis_schema}}}"""

        # Compilation provenance: computed from what was actually assembled
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
                # String-aware balanced-brace scan (the greedy regex this
                # replaced matched first-{ to LAST-} and "parsed" truncated
                # responses into a json.loads failure).
                clean = _extract_json_object(response) or response.strip()
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

        # Key developments sampled across the window, not the newest three.
        from services.news_window import select_spread
        top_titles = [a.get("title", "") for a in select_spread(articles, 3)
                      if a.get("title")]
        key_dev = ". ".join(top_titles[:2]) if top_titles else "Recent news coverage on these stocks."

        return {
            "recommendation": rec,
            "confidence": confidence,
            "key_developments": key_dev,
            "market_sentiment": sentiment,
            "sentiment_explanation": f"Based on {total} articles: {sentiment_counts['bullish']} bullish, {sentiment_counts['neutral']} neutral, {sentiment_counts['bearish']} bearish.",
        }

    # Model description constant, extend when new models are added
    _MODEL_DESCRIPTIONS = {
        "kronos_mini": "Kronos: Time-series probabilistic forecasting using Monte Carlo sampling of future price trajectories",
        "xgboost_shap": "XGBoost: Gradient-boosted decision trees on 18 engineered features (technicals + SPY correlation + news sentiment)",
        "lightgbm": "LightGBM: Alternative gradient-boosted model with leaf-wise growth (same feature set, different learning bias)",
        "deberta_sentiment": "DeBERTa: Transformer NLP model analyzing financial news article sentiment",
        "trading_agents": "TradingAgents: LLM-based research agent that synthesizes technicals, fundamentals, and news into a single trade decision",
        "ensemble": "Ensemble: Configurable weighted majority vote across enabled models",
    }

    @staticmethod
    def _research_digest(raw: str, budget: int) -> str:
        """The verdict plus the decision-relevant sections of a research report.

        The report's preamble (the Verdict block) is always kept whole; the
        `### ` sections are then added in decision-relevance order until the
        character budget runs out. Sections that are pure input restatement
        (business context, market regime) are dropped, the synthesis prompt
        already carries the model signals and the shallow report fields.
        """
        from models.single_agent import strip_epilogue

        parts = strip_epilogue(raw).split("\n### ")
        verdict = parts[0].strip()
        remaining = budget - len(verdict)

        priority = ("situation", "scenario", "bull", "risk", "trade plan",
                    "positioning", "news", "technical")

        def rank(section: str) -> int:
            heading = section.split("\n", 1)[0].lower()
            return next((i for i, kw in enumerate(priority) if kw in heading),
                        len(priority))

        keep = []
        for section in sorted(parts[1:], key=rank):
            if rank(section) == len(priority) or remaining <= 0:
                continue
            text = "### " + section.strip()
            if len(text) > remaining:
                text = text[:remaining].rsplit("\n", 1)[0]
                if len(text) < 200:
                    continue
            keep.append(text)
            remaining -= len(text)

        return "\n\n".join([verdict] + keep)

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
        "news+signals" (shallow analysis), or "signals" (predictions only,
        no text analysis at all).
        """
        if not symbols:
            return None
        signals_only = basis == "signals"

        model_desc_block = "\n".join(
            f"  - {desc}" for desc in self._MODEL_DESCRIPTIONS.values()
        )

        system_prompt = f"""You are a senior portfolio strategist. You receive two independent analyses:

1. AI REPORT: Per-symbol research, a verdict plus the report sections most relevant to the decision (bull/bear case, risk, trade plan, news), and/or news-based sentiment fields (recommendation, confidence, key developments, risk factors).
2. MODEL PREDICTIONS: Machine learning model outputs (BUY/SELL/HOLD with confidence scores) from multiple quantitative models.

AVAILABLE MODELS:
{model_desc_block}

WHAT THE NUMBERS IN THE INPUT MEAN (read this before interpreting any of them):
- "conviction" is the research report's OWN stated probability that its direction is right. It is a judgement the report made about itself. It has never been scored, so it carries no information about how often that report is actually correct.
- "track-record weight" is NOT a model saying how sure it is. It is the model's measured directional hit rate on resolved past calls, used as a weight. A value of exactly 0.50 shown as "no track record yet" means the model has not accumulated enough scored calls to earn a weight. It is a placeholder for missing evidence, NOT a claim of 50/50 conviction, and NOT the report "mirroring" anything. Never reason from a 0.50 weight as if the report expressed a neutral view.
- "up" is the model's probability that the next close is higher. Where it sits at 0.50 alongside a stated direction, the model has declared a direction without an earned edge.
- A "MEASURED ACCURACY" line inside a research report quotes the platform's evaluated hit rate with a sample size. It is the only number here that has been checked against outcomes. Where it says no rate can be stated, the sample was too thin. Do not fill one in.
- If a field is "n/a", it was not reported. Say so; do not substitute 50%.

WHAT THIS PLATFORM HAS ACTUALLY MEASURED (use these findings, do not contradict them):
- Model agreement is not evidence of a better call. Measured on this platform's own resolved predictions, days on which the models agreed performed WORSE than days they split. The models are correlated momentum readers, so consensus mostly means they all read the same trend, not that the trend is more likely to continue. Never describe unanimity or a 5-model consensus as "highest-conviction" or as a reason to size up. Where they agree, look for the input they are all leaning on and ask whether it is one signal counted several times.
- Stated confidence on this platform has historically been anti-calibrated: higher stated numbers have not produced higher hit rates. Weight evidence, not stated certainty.
- Directional hit rates across models sit close to 50% over large samples. Treat any thesis that implies a large, reliable edge as suspect.

YOUR ROLE:
- Synthesize both inputs into specific, actionable recommendations per symbol
- When the AI report and model predictions DISAGREE, this is the most valuable signal. Explain WHY they disagree and which to trust in this context
- Where OPTIONS POSITIONING or QUALITY SCREEN lines are present, factor them in: a put-tilted chain or a high quality-screen fail count argues for lower conviction and tighter risk on bullish calls (and vice versa). They are context that shades conviction, never a standalone reason to flip a direction
- Where a "Situation" line is present, the call is about how that situation resolves (a pending deal's completion odds and spread to the offer, a regulator's decision, an earnings print), not about trend. Models that only read price are less relevant there; say so in model_notes, and let key_level/change_trigger reference the offer price or the dated event rather than a moving average
- Where a report says it was WRITTEN WITHOUT expected evidence, lower p_correct for that symbol and name the missing evidence in conflicts; do not treat the absence as neutral
- Where a report carries a Scenarios section, reason from its outcome map: your p_correct for the ACTION should be consistent with the probability the report gave the scenarios that action wins in, and the reasoning should name the most likely path. Where it carries Positioning & Flows, say which positioned group is exposed if the call is wrong
- "p_correct" is a probability, not a mood: your estimate that the ACTION direction is right for the next session's close, on the 0.50-0.75 scale where 0.50 means "no edge over a coin flip". You will be scored against realized outcomes. A persistent gap between your stated p_correct and your hit rate is a defect. State 0.50-0.55 freely; earn anything above 0.65.
- Explain which models are most relevant for each symbol's situation
- Provide an overall portfolio-level summary
- Be direct and specific. No hedging.
- Voice: plain sentences a desk would read. No em dashes or en dashes as punctuation (use commas, periods, parentheses). No "not just X, it's Y". No "delve", "dive", "landscape", "unlock", "navigate", "robust", "seamless". No closing recap.

Respond with ONLY valid JSON matching this schema:
{{
  "overall": {{
    "summary": "2-3 sentence portfolio overview",
    "portfolio_action": "One-line actionable directive",
    "key_conflicts": ["List of notable disagreements between report and models"],
    "risk_assessment": "1-2 sentences on aggregate risk",
    "watch_items": ["2-3 short portfolio-level things to monitor next. Each a checkable level, date, or metric"]
  }},
  "by_symbol": {{
    "<SYMBOL>": {{
      "action": "BUY|SELL|HOLD",
      "p_correct": 0.55,
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

        # Research digest budget per symbol. The synthesis used to see only
        # the ~600-char verdict block. The analysis sections the research
        # model spent its tokens on were discarded before the one call whose
        # job is to weigh them against the quant signals. Scale with symbol
        # count so a 20-name portfolio still fits the context comfortably.
        research_budget = max(1500, min(6000, 45000 // max(1, len(symbols))))

        for symbol in symbols:
            lines = [f"=== {symbol} ==="]

            # AI Report data (suppressed entirely on a signals-only basis)
            sym_report = {} if signals_only else analysis_by_sym.get(symbol, {})
            research = (sym_report.get("research") or {}) if sym_report else {}
            if research.get("raw_response"):
                digest = self._research_digest(research["raw_response"],
                                               research_budget)
                lines.append("Research report (verdict + key sections):")
                lines.append("  " + digest.replace("\n", "\n  "))
            inv = research.get("investigation") or {}
            if inv.get("situation"):
                deal = inv.get("deal") or {}
                deal_str = ""
                if deal.get("present"):
                    offer = deal.get("offer_price")
                    deal_str = (f"; deal: {deal.get('acquirer') or 'n/a'}"
                                + (f" at ${float(offer):.2f}" if isinstance(offer, (int, float)) else "")
                                + (f", gross spread {inv['spread_pct']:+.1f}%"
                                   if inv.get("spread_pct") is not None else ""))
                lines.append(
                    f"Situation ({'web-researched' if inv.get('web') else 'classified from supplied evidence'}): "
                    f"{inv['situation']} ({inv.get('situation_confidence', 'n/a')}): "
                    f"{str(inv.get('one_line') or '')[:220]}{deal_str}")
            # "evidence" carries two shapes: the ledger dict ({"gaps": [...]})
            # from the evidence contract, and the plain list of evidence names
            # the runner stores on older/merged research entries. The list
            # shape crashed every synthesis on 2026-09-02 (AttributeError
            # before the LLM call, so even the fallback never ran).
            _ev = research.get("evidence")
            gaps = [g for g in ((_ev.get("gaps") if isinstance(_ev, dict)
                                 else None) or [])
                    if isinstance(g, dict) and g.get("severity") == "expected"]
            if gaps:
                lines.append("Research report WRITTEN WITHOUT expected evidence: "
                             + "; ".join(f"{g.get('label')} ({g.get('reason')})"
                                         for g in gaps))
            if sym_report:
                # Two distinct numbers, spelled out. The unlabeled
                # "(confidence: 0.5)" this used to print was read by the
                # synthesis model as the report expressing a neutral view,
                # when it is actually the "no track record yet" placeholder.
                weight = sym_report.get("confidence")
                conviction = sym_report.get("stated_conviction")
                weight_str = (
                    f"{weight:.0%}" + (": no track record yet, placeholder"
                                       if float(weight) == 0.5 else "")
                    if isinstance(weight, (int, float)) else "n/a")
                conv_str = (f"{conviction:.2f}"
                            if isinstance(conviction, (int, float)) else "n/a")
                lines.append(
                    f"AI Report: {sym_report.get('recommendation', 'N/A')} "
                    f"(report's own conviction: {conv_str}; "
                    f"measured track-record weight: {weight_str})")
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

            # Options positioning and quality screen: validated data, not text
            # analysis: included even on a signals-only basis.
            ctx_entry = analysis_by_sym.get(symbol, {})
            pos = ctx_entry.get("positioning") or {}
            if pos.get("pc_volume") is not None:
                lines.append(
                    f"Options positioning (as of {pos.get('as_of', 'n/a')}): "
                    f"P/C volume {pos['pc_volume']:.2f}, "
                    f"P/C open-interest {pos['pc_oi']:.2f}: {pos.get('read', '')}"
                    if pos.get("pc_oi") is not None else
                    f"Options positioning (as of {pos.get('as_of', 'n/a')}): "
                    f"P/C volume {pos['pc_volume']:.2f}: {pos.get('read', '')}")
            delta = ctx_entry.get("positioning_delta") or {}
            if delta.get("shift") is not None:
                direction = ("toward puts" if delta["shift"] > 0
                             else "toward calls" if delta["shift"] < 0
                             else "unchanged")
                lines.append(
                    f"Options flow (day-over-day): P/C open-interest "
                    f"{delta['put_call_oi_prev']:.2f} → "
                    f"{delta['put_call_oi_now']:.2f} "
                    f"({delta['shift']:+.2f}, {direction}) since "
                    f"{delta['prev_session']}")
            quality = ctx_entry.get("quality") or {}
            if quality.get("total_checks"):
                failed = ", ".join(f["check"] for f in
                                   (quality.get("failed_checks") or [])[:6])
                lines.append(
                    f"Quality screen (Bad Apples, as of {quality.get('as_of', 'n/a')}): "
                    f"{str(quality.get('flag', '')).upper()}: "
                    f"{quality['total_fails']}/{quality['total_checks']} checks failed"
                    + (f" ({failed})" if failed else ""))

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
                    # does not cover that, the key is present and None, and
                    # formatting None raised, taking the whole synthesis down.
                    # Say "n/a" instead of inventing a 50%.
                    confidence = result.get("confidence")
                    up_prob = result.get("up_probability")
                    # Name which KIND of number this is. The research arm
                    # publishes a measured track-record weight (0.5 = none
                    # earned yet); every other model publishes its own raw,
                    # uncalibrated score. Printing both as "conf:" is what led
                    # the synthesis to read a 0.5 placeholder as a stated view.
                    if model_name == "trading_agents":
                        conf_label = "measured track-record weight"
                        placeholder = (confidence is not None
                                       and float(confidence) == 0.5)
                    else:
                        conf_label = "raw model score, uncalibrated"
                        placeholder = False
                    conf_str = f"{confidence:.0%}" if confidence is not None else "n/a"
                    if placeholder:
                        conf_str += ": no track record yet, placeholder"
                    elif confidence is not None:
                        cal = _calibrated(model_name, confidence)
                        conf_str += (f" (calls at this score have resolved "
                                     f"{cal:.0%} correct)" if cal is not None
                                     else " (no calibration history yet)")
                    up_str = f"{up_prob:.0%}" if up_prob is not None else "n/a"
                    lines.append(f"  {display}: {decision} "
                                 f"({conf_label}: {conf_str}, up: {up_str})")
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
        synthesis_provider = primary_provider
        parsed = _parse_recommendations_json(raw) if raw else None

        if parsed is None and API.ANTHROPIC_API_KEY:
            # Primary synthesis failed, empty response OR unusable JSON (a
            # truncated payload used to drop the whole day's synthesis on the
            # floor here). A completed AI report + prediction run shouldn't
            # be wasted, so re-ask once on the fallback model.
            fallback = MODEL.RECOMMENDATIONS_FALLBACK_MODEL
            if fallback == primary_model:
                fallback = ("claude-sonnet-4-6" if primary_model == "claude-sonnet-5"
                            else "claude-sonnet-5")
            logger.warning(
                f"{primary_model} produced no usable synthesis "
                f"({'empty' if not raw else 'unparseable'}): "
                f"falling back to {fallback}"
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
            synthesis_provider = "anthropic"
            parsed = _parse_recommendations_json(raw) if raw else None

        if parsed is None:
            logger.warning("Recommendations synthesis produced no usable JSON "
                           "from any model")
            return None

        parsed["model_used"] = synthesis_model
        parsed["provider_used"] = synthesis_provider
        parsed["basis"] = basis
        return parsed


def _extract_json_object(text: str) -> Optional[str]:
    """First balanced top-level {...} in ``text``, string-aware.

    The old greedy ``\\{.*\\}`` regex matched from the first ``{`` to the LAST
    ``}`` in the buffer, so a response truncated mid-object "matched" and then
    failed json.loads: and a code fence stripped globally could corrupt JSON
    whose string values themselves contained backticks. Scanning braces with
    string/escape tracking sidesteps both.
    """
    depth = 0
    start = None
    in_str = False
    esc = False
    for i, c in enumerate(text):
        if esc:
            esc = False
            continue
        if in_str:
            if c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            if depth == 0:
                start = i
            depth += 1
        elif c == "}" and depth:
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def _parse_recommendations_json(raw: str) -> Optional[dict]:
    """Parse a synthesis response into its recommendations dict, or None."""
    candidate = _extract_json_object(raw)
    if candidate is None:
        logger.warning("No complete JSON object in recommendations response "
                       f"({len(raw)} chars: likely truncated output)")
        return None
    try:
        parsed = json.loads(candidate)
    except (json.JSONDecodeError, TypeError) as e:
        logger.warning(f"Failed to parse recommendations: {e}")
        return None
    if not isinstance(parsed, dict) or "overall" not in parsed \
            or "by_symbol" not in parsed:
        logger.warning("Recommendations JSON missing required keys")
        return None
    return parsed


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

"""Which token parameter an OpenAI request carries is the MODEL's property.

gpt-5.x and the o-series removed `max_tokens` in favour of
`max_completion_tokens`; sending the old name is a hard 400. The choice used
to be keyed off whether the caller passed `reasoning_effort`, so every
gpt-5.6-luna call that did not pass one 400'd. It stayed invisible while the
OpenAI account was out of credits (the 429 came first) and surfaced the
moment billing was restored on 2026-09-04, failing the run at the research
stage.
"""

from unittest.mock import MagicMock

import pytest

from services import llm_service as ls


class TestModelPredicate:
    @pytest.mark.parametrize("model", [
        "gpt-5.6-luna", "gpt-5", "gpt-5.6", "o1-preview", "o3-mini", "o4-mini",
    ])
    def test_new_families_want_completion_tokens(self, model):
        assert ls._uses_completion_tokens(model) is True

    @pytest.mark.parametrize("model", [
        "gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "local-model",
    ])
    def test_older_models_keep_max_tokens(self, model):
        assert ls._uses_completion_tokens(model) is False


class TestRequestShape:
    """What actually reaches chat.completions.create."""

    def _capture(self, monkeypatch, model):
        """A service whose only live part is the OpenAI client it calls."""
        seen = {}
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = "ok"
        resp.choices[0].finish_reason = "stop"

        def _create(**api_kwargs):
            seen.clear()
            seen.update(api_kwargs)
            return resp

        client = MagicMock()
        client.chat.completions.create = _create

        svc = ls.LLMService.__new__(ls.LLMService)
        svc._active = (client, "openai")
        svc._get_client_for_provider = lambda p: (client, "openai")
        # Tracing and usage accounting both hit the database; neither is
        # what these tests are about.
        monkeypatch.setattr(ls, "_record_usage", lambda *a, **k: None)
        monkeypatch.setattr(ls, "_trace_call", lambda *a, **k: None,
                            raising=False)
        return seen, svc, client

    def test_a_gpt5_call_without_an_effort_still_sends_completion_tokens(
            self, monkeypatch):
        """The exact 2026-09-04 shape: no reasoning_effort, gpt-5 model."""
        seen, svc, client = self._capture(monkeypatch, "gpt-5.6-luna")
        ls.LLMService.generate(
            svc, "hi", None, max_tokens=800, temperature=0.4,
            model="gpt-5.6-luna", provider="openai")
        assert "max_completion_tokens" in seen
        assert "max_tokens" not in seen, (
            "max_tokens on a gpt-5 model is a hard 400")
        assert "temperature" not in seen, (
            "these models reject temperature the same way")
        assert "reasoning_effort" not in seen, (
            "an effort the caller never asked for must not be invented")

    def test_an_effort_is_still_honoured(self, monkeypatch):
        seen, svc, client = self._capture(monkeypatch, "gpt-5.6-luna")
        ls.LLMService.generate(
            svc, "hi", None, max_tokens=800, temperature=0.4,
            model="gpt-5.6-luna", provider="openai", reasoning_effort="high")
        assert seen["reasoning_effort"] == "high"
        assert seen["max_completion_tokens"] >= 16000

    def test_an_older_model_keeps_max_tokens_and_temperature(self, monkeypatch):
        seen, svc, client = self._capture(monkeypatch, "gpt-4o")
        ls.LLMService.generate(
            svc, "hi", None, max_tokens=800, temperature=0.4,
            model="gpt-4o", provider="openai")
        assert seen["max_tokens"] == 800
        assert seen["temperature"] == 0.4
        assert "max_completion_tokens" not in seen

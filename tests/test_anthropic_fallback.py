"""Tests for the OpenRouter → Anthropic-direct fallback in researcher.models.

Covers:
  - 402 from OpenRouter triggers fallback (no retries)
  - 200 OK from OpenRouter does NOT trigger fallback
  - Non-Anthropic models (gemini, deepseek) get substituted to claude-haiku-4-5
  - Model substitution map handles claude-* passthrough
  - Token usage from fallback flows back into the cost tracker
"""
from __future__ import annotations

import asyncio
import os
import sys
import types
from pathlib import Path

import pytest

# Ensure the repo root is on sys.path so `deep_research` imports resolve.
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from deep_research import models as researcher_models  # noqa: E402
from deep_research.models import (  # noqa: E402
    CostTracker,
    _substitute_for_anthropic_direct,
    query_openrouter,
    query_anthropic_direct,
    query_llm,
)


# ---------------------------------------------------------------------------
# Pure-function: substitution table
# ---------------------------------------------------------------------------

def test_substitution_table():
    assert _substitute_for_anthropic_direct("google/gemini-2.5-flash") == "claude-haiku-4-5"
    assert _substitute_for_anthropic_direct("deepseek/deepseek-chat-v3") == "claude-haiku-4-5"
    assert _substitute_for_anthropic_direct("local/ollama") == "claude-haiku-4-5"
    assert _substitute_for_anthropic_direct("claude-opus-4") == "claude-opus-4-5"
    assert _substitute_for_anthropic_direct("claude-sonnet-4") == "claude-sonnet-4-5"
    assert _substitute_for_anthropic_direct("claude-haiku-4") == "claude-haiku-4-5"
    # Already-direct ids passthrough
    assert _substitute_for_anthropic_direct("claude-haiku-4-5") == "claude-haiku-4-5"
    # Unknown vendor → safe haiku
    assert _substitute_for_anthropic_direct("xai/grok-7") == "claude-haiku-4-5"


# ---------------------------------------------------------------------------
# Helpers — fake httpx responses + clients
# ---------------------------------------------------------------------------

class _FakeResponse:
    def __init__(self, status_code: int, json_payload: dict | None = None, text: str = ""):
        self.status_code = status_code
        self._json = json_payload or {}
        self.text = text or str(self._json)

    @property
    def is_success(self) -> bool:
        return 200 <= self.status_code < 300

    def json(self) -> dict:
        return self._json


class _FakeAsyncClient:
    """Drop-in for httpx.AsyncClient that returns a preset response."""

    def __init__(self, response: _FakeResponse):
        self._response = response

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, *args, **kwargs):
        return self._response


def _patch_httpx(monkeypatch, response: _FakeResponse):
    def _factory(*args, **kwargs):
        return _FakeAsyncClient(response)
    monkeypatch.setattr(researcher_models.httpx, "AsyncClient", _factory)


# ---------------------------------------------------------------------------
# Test: 402 triggers fallback immediately, no retries
# ---------------------------------------------------------------------------

def test_402_triggers_fallback(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-key")
    _patch_httpx(monkeypatch, _FakeResponse(402, text="Insufficient credits"))

    fallback_calls: list[tuple] = []

    async def fake_fallback(prompt, model, max_tokens=4096):
        fallback_calls.append((prompt[:30], model, max_tokens))
        return ("FALLBACK_TEXT", 100, 50, "claude-haiku-4-5")

    monkeypatch.setattr(researcher_models, "query_anthropic_direct", fake_fallback)

    text, in_tok, out_tok, model_used = asyncio.run(
        query_openrouter("hello world", "google/gemini-2.5-flash", max_tokens=512)
    )

    assert text == "FALLBACK_TEXT"
    assert in_tok == 100
    assert out_tok == 50
    assert model_used == "claude-haiku-4-5"
    assert len(fallback_calls) == 1, "fallback must fire exactly once on 402"
    assert fallback_calls[0][1] == "google/gemini-2.5-flash"


# ---------------------------------------------------------------------------
# Test: 200 OK skips fallback entirely
# ---------------------------------------------------------------------------

def test_200_skips_fallback(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-key")
    _patch_httpx(monkeypatch, _FakeResponse(
        200,
        json_payload={
            "choices": [{"message": {"content": "OPENROUTER_TEXT"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20},
        },
    ))

    async def boom_fallback(*a, **kw):
        raise AssertionError("fallback must NOT be called on 200 OK")

    monkeypatch.setattr(researcher_models, "query_anthropic_direct", boom_fallback)

    text, in_tok, out_tok, model_used = asyncio.run(
        query_openrouter("hello world", "google/gemini-2.5-flash", max_tokens=512)
    )

    assert text == "OPENROUTER_TEXT"
    assert in_tok == 10
    assert out_tok == 20
    # On the 200 path, model_used reflects the OpenRouter model the caller asked for.
    assert model_used == "google/gemini-2.5-flash"


# ---------------------------------------------------------------------------
# Test: model substitution inside query_anthropic_direct
# ---------------------------------------------------------------------------

def test_model_substitution_in_anthropic_direct(monkeypatch):
    """When called with a non-Anthropic model, query_anthropic_direct must
    substitute it to claude-haiku-4-5 before calling the SDK."""
    monkeypatch.setenv("ANTHROPIC_API_KEY_SUB1", "sk-ant-api03-fake-test-key")
    # Ensure we don't pick up an OAuth token
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY_SUB2", raising=False)

    captured: dict = {}

    class _FakeUsage:
        input_tokens = 42
        output_tokens = 17

    class _FakeBlock:
        text = "FAKE_ANTHROPIC_REPLY"

    class _FakeMessage:
        content = [_FakeBlock()]
        usage = _FakeUsage()

    class _FakeMessages:
        def create(self, *, model, max_tokens, messages):
            captured["model"] = model
            captured["max_tokens"] = max_tokens
            captured["messages"] = messages
            return _FakeMessage()

    class _FakeAnthropicClient:
        def __init__(self, api_key):
            captured["api_key"] = api_key
            self.messages = _FakeMessages()

    fake_anthropic = types.ModuleType("anthropic")
    fake_anthropic.Anthropic = _FakeAnthropicClient
    fake_anthropic.NotFoundError = type("NotFoundError", (Exception,), {})

    monkeypatch.setitem(sys.modules, "anthropic", fake_anthropic)

    text, in_tok, out_tok, model_used = asyncio.run(
        query_anthropic_direct("hello", "google/gemini-2.5-flash", max_tokens=256)
    )

    assert text == "FAKE_ANTHROPIC_REPLY"
    assert in_tok == 42
    assert out_tok == 17
    assert model_used == "claude-haiku-4-5"
    assert captured["model"] == "claude-haiku-4-5"
    assert captured["api_key"] == "sk-ant-api03-fake-test-key"


def test_anthropic_direct_rejects_oauth_token(monkeypatch):
    """OAuth tokens (sk-ant-oat...) must NOT be used for direct SDK calls."""
    monkeypatch.delenv("ANTHROPIC_API_KEY_SUB1", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-oat01-fake-oauth-token")
    monkeypatch.delenv("ANTHROPIC_API_KEY_SUB2", raising=False)

    fake_anthropic = types.ModuleType("anthropic")
    fake_anthropic.Anthropic = lambda api_key: None
    fake_anthropic.NotFoundError = type("NotFoundError", (Exception,), {})
    monkeypatch.setitem(sys.modules, "anthropic", fake_anthropic)

    with pytest.raises(Exception, match="no usable ANTHROPIC_API_KEY"):
        asyncio.run(query_anthropic_direct("hi", "claude-haiku-4", max_tokens=128))


# ---------------------------------------------------------------------------
# Test: cost tracker records the substituted model on fallback path
# ---------------------------------------------------------------------------

def test_cost_tracker_records_substituted_model(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-key")
    _patch_httpx(monkeypatch, _FakeResponse(402, text="Insufficient credits"))

    async def fake_fallback(prompt, model, max_tokens=4096):
        return ("WRITER_OUTPUT", 1000, 500, "claude-haiku-4-5")

    monkeypatch.setattr(researcher_models, "query_anthropic_direct", fake_fallback)

    tracker = CostTracker()
    text = asyncio.run(query_llm(
        prompt="write a report",
        role="writer",
        profile="multi",
        cost_tracker=tracker,
    ))

    assert text == "WRITER_OUTPUT"
    assert len(tracker.calls) == 1
    call = tracker.calls[0]
    # The cost tracker should see the SUBSTITUTED model (not the original deepseek model).
    assert call["model"] == "claude-haiku-4-5"
    assert call["input_tokens"] == 1000
    assert call["output_tokens"] == 500
    # Cost should be > 0 since claude-haiku-4-5 is in MODEL_COSTS
    assert call["cost_usd"] > 0

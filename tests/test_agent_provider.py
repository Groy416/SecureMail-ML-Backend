from __future__ import annotations

import json

import pytest

from api.agent.provider import (
    AgentProviderError,
    OpenAICompatibleProvider,
    build_agent_user_content,
)
from api.schemas import AgentMemoryState


class FakeResponse:
    def __init__(self, body: bytes):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self, limit: int) -> bytes:
        return self.body


def completion_body(payload: dict) -> bytes:
    return json.dumps({"choices": [{"message": {"content": json.dumps(payload)}}]}).encode()


def advisory_payload() -> dict:
    return {
        "answer": "ok",
        "recommendations": ["Verify the chain."],
        "evidence": [],
        "active_step": {
            "id": "verify-chain",
            "title": "Verify the chain",
            "status": "in_progress",
            "evidence": [],
        },
        "memory_update": {
            "summary": "Verifying the chain.",
            "facts": [],
            "completed_steps": [],
            "active_step": {
                "id": "verify-chain",
                "title": "Verify the chain",
                "status": "in_progress",
                "evidence": [],
            },
            "pending_questions": [],
        },
    }


def test_provider_posts_chat_completion_without_exposing_secret():
    captured = {}

    def opener(request, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["auth"] = request.get_header("Authorization")
        return FakeResponse(completion_body(advisory_payload()))

    provider = OpenAICompatibleProvider("openai", "model", "secret", "https://example.test/v1", 3, 1024, opener)
    result = provider.generate("system", "user")

    assert result.answer == "ok"
    assert result.memory_update.active_step.id == "verify-chain"
    assert captured == {"url": "https://example.test/v1/chat/completions", "timeout": 3, "auth": "Bearer secret"}


def test_provider_accepts_fenced_json_response():
    content = f"```json\n{json.dumps(advisory_payload())}\n```"
    body = json.dumps({"choices": [{"message": {"content": content}}]}).encode()
    provider = OpenAICompatibleProvider("groq", "model", "secret", "https://example.test/v1", 3, 4096, lambda *_args, **_kwargs: FakeResponse(body))
    assert provider.generate("system", "user").answer == "ok"


def test_user_content_rejects_over_budget_payload():
    with pytest.raises(AgentProviderError, match="context_too_large"):
        build_agent_user_content(
            "risk", "question", {"x": "y"}, AgentMemoryState(summary="x" * 6000), 1000
        )


def test_provider_rejects_oversized_response():
    provider = OpenAICompatibleProvider("openai", "model", "secret", "https://example.test/v1", 3, 10, lambda *_args, **_kwargs: FakeResponse(b"x" * 20))
    with pytest.raises(AgentProviderError, match="provider_response_too_large"):
        provider.generate("system", "user")


def test_provider_rejects_invalid_advisory():
    provider = OpenAICompatibleProvider("openai", "model", "secret", "https://example.test/v1", 3, 1024, lambda *_args, **_kwargs: FakeResponse(b'{"choices":[{"message":{"content":"not-json"}}]}'))
    with pytest.raises(AgentProviderError, match="provider_invalid_response"):
        provider.generate("system", "user")


def test_factory_supports_groq(monkeypatch):
    monkeypatch.setattr("api.agent.provider.settings.AGENT_PROVIDER", "groq")
    monkeypatch.setattr("api.agent.provider.settings.AGENT_MODEL", "llama-test")
    monkeypatch.setattr("api.agent.provider.settings.AGENT_API_KEY", "secret")
    monkeypatch.setattr("api.agent.provider.settings.AGENT_BASE_URL", None)

    from api.agent.provider import create_agent_provider

    provider = create_agent_provider()
    assert provider is not None
    assert provider.provider == "groq"
    assert provider.base_url == "https://api.groq.com/openai/v1"

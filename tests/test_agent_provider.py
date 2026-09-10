from __future__ import annotations

import pytest

from api.agent.provider import AgentProviderError, OpenAICompatibleProvider


class FakeResponse:
    def __init__(self, body: bytes):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self, limit: int) -> bytes:
        return self.body


def test_provider_posts_chat_completion_without_exposing_secret():
    captured = {}

    def opener(request, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["auth"] = request.get_header("Authorization")
        return FakeResponse(b'{"choices":[{"message":{"content":"{\\"answer\\":\\"ok\\",\\"recommendations\\":[],\\"evidence\\":[]}"}}]}')

    provider = OpenAICompatibleProvider("openai", "model", "secret", "https://example.test/v1", 3, 1024, opener)
    result = provider.generate("system", "user")

    assert result.answer == "ok"
    assert captured == {"url": "https://example.test/v1/chat/completions", "timeout": 3, "auth": "Bearer secret"}


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

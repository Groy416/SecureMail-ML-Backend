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


def gemini_body(payload: dict) -> bytes:
    return json.dumps({"candidates": [{"content": {"parts": [{"text": json.dumps(payload)}]}}]}).encode()


def gemini_body_with_thought(payload: dict) -> bytes:
    return json.dumps(
        {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"thought": True, "text": "I need to inspect the advisory fields."},
                            {"text": json.dumps(payload)},
                        ]
                    }
                }
            ]
        }
    ).encode()


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


def test_provider_rejects_invalid_advisory_json():
    provider = OpenAICompatibleProvider("openai", "model", "secret", "https://example.test/v1", 3, 1024, lambda *_args, **_kwargs: FakeResponse(b'{"choices":[{"message":{"content":"not-json"}}]}'))
    with pytest.raises(AgentProviderError, match="provider_invalid_json"):
        provider.generate("system", "user")


def test_provider_rejects_advisory_schema_mismatch():
    provider = OpenAICompatibleProvider(
        "openai",
        "model",
        "secret",
        "https://example.test/v1",
        3,
        1024,
        lambda *_args, **_kwargs: FakeResponse(completion_body({"answer": "ok", "memory_update": []})),
    )
    with pytest.raises(AgentProviderError, match="provider_schema_mismatch"):
        provider.generate("system", "user")


def test_gemini_uses_native_generate_content_json_mode():
    captured = {}

    def opener(request, timeout):
        captured["url"] = request.full_url
        captured["api_key"] = request.get_header("X-goog-api-key")
        captured["payload"] = json.loads(request.data)
        return FakeResponse(gemini_body(advisory_payload()))

    provider = OpenAICompatibleProvider("gemini", "gemini-test", "secret", "https://example.test/v1/openai", 3, 4096, opener)
    assert provider.generate("system", "user").answer == "ok"

    assert captured["url"] == "https://example.test/v1/models/gemini-test:generateContent"
    assert captured["api_key"] == "secret"
    assert captured["payload"]["systemInstruction"] == {"parts": [{"text": "system"}]}
    assert captured["payload"]["contents"] == [{"role": "user", "parts": [{"text": "user"}]}]
    generation = captured["payload"]["generationConfig"]
    assert generation["temperature"] == 0.6
    assert generation["topP"] == 0.95
    assert generation["maxOutputTokens"] == 2048
    assert generation["responseMimeType"] == "application/json"
    assert generation["responseJsonSchema"] == {
        "type": "object",
        "required": ["answer", "active_step", "memory_update"],
        "properties": {
            "answer": {"type": "string"},
            "recommendations": {"type": "array", "items": {"type": "string"}},
            "evidence": {"type": "array", "items": {"type": "string"}},
            "active_step": {
                "type": "object",
                "required": ["id", "title", "status", "evidence"],
                "properties": {
                    "id": {"type": "string"},
                    "title": {"type": "string"},
                    "status": {"type": "string", "enum": ["proposed", "in_progress", "waiting_for_result", "completed"]},
                    "evidence": {"type": "array", "items": {"type": "string"}},
                },
            },
            "memory_update": {
                "type": "object",
                "required": ["summary", "facts", "completed_steps", "active_step", "pending_questions"],
                "properties": {
                    "summary": {"type": "string"},
                    "facts": {"type": "array", "items": {"type": "string"}},
                    "completed_steps": {"type": "array", "items": {"type": "string"}},
                    "active_step": {
                        "type": "object",
                        "required": ["id", "title", "status", "evidence"],
                        "properties": {
                            "id": {"type": "string"},
                            "title": {"type": "string"},
                            "status": {"type": "string", "enum": ["proposed", "in_progress", "waiting_for_result", "completed"]},
                            "evidence": {"type": "array", "items": {"type": "string"}},
                        },
                    },
                    "pending_questions": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
    }


def test_gemini_ignores_thought_parts_before_the_json_response():
    provider = OpenAICompatibleProvider(
        "gemini",
        "gemini-test",
        "secret",
        "https://example.test/v1",
        3,
        4096,
        lambda *_args, **_kwargs: FakeResponse(gemini_body_with_thought(advisory_payload())),
    )

    assert provider.generate("system", "user").answer == "ok"


def test_factory_supports_gemini(monkeypatch):
    monkeypatch.setattr("api.agent.provider.settings.AGENT_PROVIDER", "gemini")
    monkeypatch.setattr("api.agent.provider.settings.AGENT_MODEL", "gemini-test")
    monkeypatch.setattr("api.agent.provider.settings.AGENT_API_KEY", "secret")
    monkeypatch.setattr("api.agent.provider.settings.AGENT_BASE_URL", None)

    from api.agent.provider import create_agent_provider

    provider = create_agent_provider()
    assert provider is not None
    assert provider.provider == "gemini"
    assert provider.base_url == "https://generativelanguage.googleapis.com/v1beta"


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

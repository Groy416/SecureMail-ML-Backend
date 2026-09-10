from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from pydantic import ValidationError

from api.config import settings
from api.schemas import AgentAdvisory, AgentMemoryState

DEFAULT_BASE_URLS = {
    "openai": "https://api.openai.com/v1",
    "groq": "https://api.groq.com/openai/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta",
}


class AgentProviderError(RuntimeError):
    """A safe, stable provider failure code."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def build_agent_user_content(
    section: str,
    question: str,
    context: dict[str, Any],
    memory: AgentMemoryState,
    max_chars: int,
) -> str:
    payload = json.dumps(
        {
            "section": section,
            "question": question,
            "memory": memory.model_dump(mode="json"),
            "context": context,
        },
        separators=(",", ":"),
    )
    if len(payload.encode("utf-8")) > max_chars:
        raise AgentProviderError("context_too_large")
    return payload


class AgentProvider(Protocol):
    provider: str
    model: str

    def generate(self, system: str, user: str) -> AgentAdvisory: ...


@dataclass(frozen=True)
class OpenAICompatibleProvider:
    provider: str
    model: str
    api_key: str
    base_url: str
    timeout_seconds: float
    max_response_bytes: int
    opener: Callable[..., Any] | None = None

    def generate(self, system: str, user: str) -> AgentAdvisory:
        if self.provider == "gemini":
            return self._generate_gemini(system, user)

        provider_options = {"reasoning_effort": "default"} if self.provider == "groq" else {}
        payload = json.dumps(
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": 0.6,
                "top_p": 0.95,
                "max_completion_tokens": 2048,
                "stream": False,
                **provider_options,
            }
        ).encode("utf-8")
        request = Request(
            f"{self.base_url.rstrip('/')}/chat/completions",
            data=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "securemail-agent/0.1",
            },
            method="POST",
        )
        raw = self._read_response(request)
        try:
            body = json.loads(raw)
            content = body["choices"][0]["message"]["content"]
            return self._parse_advisory(content)
        except (ValueError, KeyError, IndexError, TypeError):
            raise AgentProviderError("provider_invalid_response") from None

    def _generate_gemini(self, system: str, user: str) -> AgentAdvisory:
        base_url = self.base_url.rstrip("/")
        if base_url.endswith("/openai"):
            base_url = base_url.removesuffix("/openai")
        payload = json.dumps(
            {
                "systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": user}]}],
                "generationConfig": {
                    "temperature": 0.6,
                    "topP": 0.95,
                    "maxOutputTokens": 2048,
                    "responseMimeType": "application/json",
                },
            }
        ).encode("utf-8")
        request = Request(
            f"{base_url}/models/{quote(self.model, safe='')}:generateContent",
            data=payload,
            headers={
                "x-goog-api-key": self.api_key,
                "Content-Type": "application/json",
                "User-Agent": "securemail-agent/0.1",
            },
            method="POST",
        )
        raw = self._read_response(request)
        try:
            body = json.loads(raw)
            content = body["candidates"][0]["content"]["parts"][0]["text"]
            return self._parse_advisory(content)
        except (ValueError, KeyError, IndexError, TypeError):
            raise AgentProviderError("provider_invalid_response") from None

    def _read_response(self, request: Request) -> bytes:
        try:
            with (self.opener or urlopen)(request, timeout=self.timeout_seconds) as response:
                raw = response.read(self.max_response_bytes + 1)
        except HTTPError as exc:
            raise AgentProviderError("provider_http_error") from exc
        except (TimeoutError, URLError, OSError):
            raise AgentProviderError("provider_unavailable") from None
        if len(raw) > self.max_response_bytes:
            raise AgentProviderError("provider_response_too_large")
        return raw

    @staticmethod
    def _parse_advisory(content: Any) -> AgentAdvisory:
        try:
            if isinstance(content, str):
                content = content.strip()
                if content.startswith("```"):
                    content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()
                content = json.loads(content)
            return AgentAdvisory.model_validate(content)
        except (ValueError, TypeError, ValidationError):
            raise AgentProviderError("provider_invalid_response") from None


def create_agent_provider() -> AgentProvider | None:
    provider = (settings.AGENT_PROVIDER or "").lower()
    model = settings.AGENT_MODEL
    api_key = settings.AGENT_API_KEY
    if not provider or not model or not api_key:
        return None
    if provider not in DEFAULT_BASE_URLS:
        return None
    return OpenAICompatibleProvider(
        provider=provider,
        model=model,
        api_key=api_key,
        base_url=settings.AGENT_BASE_URL or DEFAULT_BASE_URLS[provider],
        timeout_seconds=settings.AGENT_TIMEOUT_SECONDS,
        max_response_bytes=settings.AGENT_MAX_RESPONSE_BYTES,
    )

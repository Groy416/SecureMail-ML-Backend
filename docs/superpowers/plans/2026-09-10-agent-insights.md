# Authorized Agent Insights Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an authenticated, read-only AI-insights endpoint that sends only allowlisted analysis context to an OpenAI-compatible provider configured for OpenAI or Groq.

**Architecture:** Keep the agent feature isolated under `api/agent/`: a context builder owns the safe section allowlist, a provider module owns bounded OpenAI-compatible HTTP, and route/schema code owns authentication and response envelopes. The existing ML verdict and deterministic findings remain authoritative; provider failures return explicit degraded responses.

**Tech Stack:** Python 3.11, FastAPI, Pydantic v2, standard-library `urllib`, pytest, existing API-key authentication.

**Spec:** `docs/superpowers/specs/2026-09-10-agent-insights-design.md`

## Global Constraints

- The endpoint is `POST /api/v1/agent/insights`.
- Accept only existing safe `analysis-response.v1` or persisted `AnalysisRecordResponse` context; never arbitrary `section_data`.
- Allowed sections are `overview`, `risk`, `tls`, `certificate`, and `findings`.
- Do not add a provider SDK; use the standard library for the first non-streaming adapter.
- Do not send or persist email bodies, credentials, tokens, private keys, raw PCAP, or authorization headers.
- Do not change authoritative ML/rule verdicts or execute remediation.
- Use one provider call with bounded timeout and bounded response size; no retries.
- Never log or return API keys, prompts, provider response bodies, or sensitive context.

---

### Task 1: Add typed agent contracts and safe section context

**Files:**
- Create: `api/agent/__init__.py`
- Create: `api/agent/context.py`
- Modify: `api/schemas.py` (add request/response models near the existing API envelopes)
- Test: `tests/test_agent_context.py`

**Interfaces:**
- Consumes: existing `AnalysisResponse` fields and existing result payload shape.
- Produces: `AgentSection`, `AgentInsightRequest`, `AgentInsightResponse`, `build_agent_context(analysis: AnalysisResponse, section: AgentSection) -> dict[str, object]`, and `allowed_evidence(analysis: AnalysisResponse) -> set[str]`.

- [ ] **Step 1: Write failing tests for contract validation and field filtering**

```python
from api.agent.context import build_agent_context
from api.schemas import AnalysisResponse


def test_risk_context_excludes_unknown_and_sensitive_fields(sample_analysis):
    sample_analysis["result"]["secret"] = "must-not-forward"
    sample_analysis["session"]["raw_record"] = {"password": "must-not-forward"}
    analysis = AnalysisResponse.model_validate(sample_analysis)

    context = build_agent_context(analysis, "risk")

    assert context["risk"]["class"] == "high"
    assert "secret" not in str(context)
    assert "password" not in str(context)


def test_evidence_allowlist_contains_only_analysis_findings(sample_analysis):
    analysis = AnalysisResponse.model_validate(sample_analysis)
    assert allowed_evidence(analysis) == {"TLS-001"}
```

- [ ] **Step 2: Run the focused tests and confirm the new imports/functions fail**

Run: `uv run pytest tests/test_agent_context.py -q`
Expected: collection failure because `api.agent.context` and agent models do not exist.

- [ ] **Step 3: Implement the versioned Pydantic envelopes and section allowlist**

Use `Literal` for the five sections and for `status`. Bound `question` to a finite length (maximum 4,000 characters). Accept either live `analysis-response.v1` or persisted `AnalysisRecordResponse`, normalize both through explicit fields, and build a new dictionary per section; never return `analysis.model_dump()` or recursively copy unknown fields. Include only safe identifiers, protocol/posture, risk fields, deterministic finding summaries, TLS details, certificate details, and diagnostics as specified by the design.

- [ ] **Step 4: Run the focused tests and confirm filtering passes**

Run: `uv run pytest tests/test_agent_context.py -q`
Expected: PASS.

- [ ] **Step 5: Commit the contracts and context boundary**

```bash
git add api/agent api/schemas.py tests/test_agent_context.py
git commit -m "feat: add safe agent insight contracts"
```

### Task 2: Implement the bounded OpenAI-compatible provider

**Files:**
- Create: `api/agent/provider.py`
- Modify: `api/config.py` (add agent settings)
- Test: `tests/test_agent_provider.py`

**Interfaces:**
- Consumes: `AgentInsight` response schema from Task 1 and agent settings.
- Produces: `AgentProvider` protocol, `OpenAICompatibleProvider`, `AgentProviderError`, and `create_agent_provider() -> AgentProvider | None`.

- [ ] **Step 1: Write failing provider tests with a patched standard-library opener**

```python
from api.agent.provider import OpenAICompatibleProvider, AgentProviderError


def test_provider_posts_chat_completion_without_logging_secret(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["authorization"] = request.get_header("Authorization")
        captured["timeout"] = timeout
        return FakeHttpResponse(b'{"choices":[{"message":{"content":"{\\"answer\\":\\"ok\\",\\"recommendations\\":[],\\"evidence\\":[]}"}}]}')

    monkeypatch.setattr("api.agent.provider.urlopen", fake_urlopen)
    provider = OpenAICompatibleProvider("openai", "model", "secret", "https://example.test/v1", 3, 1024)
    result = provider.generate("system", "user")

    assert result.answer == "ok"
    assert captured["url"] == "https://example.test/v1/chat/completions"
    assert captured["timeout"] == 3


def test_provider_rejects_oversized_response(monkeypatch):
    monkeypatch.setattr("api.agent.provider.urlopen", lambda request, timeout: FakeHttpResponse(b"x" * 20))
    provider = OpenAICompatibleProvider("openai", "model", "secret", "https://example.test/v1", 3, 10)
    with pytest.raises(AgentProviderError, match="response_too_large"):
        provider.generate("system", "user")
```

- [ ] **Step 2: Run the focused tests and confirm provider implementation is missing**

Run: `uv run pytest tests/test_agent_provider.py -q`
Expected: collection failure because `api.agent.provider` does not exist.

- [ ] **Step 3: Implement the provider protocol and settings factory**

Use `urllib.request.Request` and `urlopen`. POST JSON to `<base_url>/chat/completions` with `Authorization: Bearer <AGENT_API_KEY>`, `Content-Type: application/json`, one system message, one user message, the configured model, and deterministic generation settings only if required by the provider. Read at most `AGENT_MAX_RESPONSE_BYTES + 1`; raise stable errors for timeout, HTTP failure, invalid JSON, missing choices/content, and oversized bodies. Parse the model content as JSON and validate it as the advisory payload. Redact secrets from all exception text. Return `None` from the factory when agent configuration is incomplete.

- [ ] **Step 4: Run the focused provider tests**

Run: `uv run pytest tests/test_agent_provider.py -q`
Expected: PASS.

- [ ] **Step 5: Commit the provider boundary**

```bash
git add api/agent/provider.py api/config.py tests/test_agent_provider.py
git commit -m "feat: add bounded openai-compatible agent provider"
```

### Task 3: Wire the authenticated insights route with explicit degraded states

**Files:**
- Create: `api/agent/routes.py`
- Modify: `api/routes.py` (include the agent router)
- Test: `tests/test_agent_api.py`

**Interfaces:**
- Consumes: `AgentInsightRequest`, `build_agent_context`, `allowed_evidence`, and `AgentProvider` from Tasks 1–2.
- Produces: authenticated `POST /api/v1/agent/insights` returning `AgentInsightResponse` or normal `401`/`422` errors.

- [ ] **Step 1: Write failing API tests for auth, success, filtering, and degraded provider failure**

```python
def test_agent_requires_api_key(client, sample_analysis, monkeypatch):
    monkeypatch.setenv("SECUREMAIL_API_KEY", "required")
    response = client.post("/api/v1/agent/insights", json=agent_request(sample_analysis))
    assert response.status_code == 401


def test_agent_returns_structured_advisory(client, sample_analysis, fake_provider):
    response = client.post(
        "/api/v1/agent/insights",
        headers={"Authorization": "test-key"},
        json=agent_request(sample_analysis),
    )
    assert response.status_code == 200
    assert response.json()["status"] == "complete"
    assert response.json()["evidence"] == ["TLS-001"]


def test_agent_provider_failure_is_degraded(client, sample_analysis, failing_provider):
    response = client.post(
        "/api/v1/agent/insights",
        headers={"Authorization": "test-key"},
        json=agent_request(sample_analysis),
    )
    assert response.status_code == 200
    assert response.json()["status"] == "degraded"
    assert response.json()["answer"] is None
    assert response.json()["diagnostics"]["errors"] == ["provider_unavailable"]
```

- [ ] **Step 2: Run the focused API tests and confirm the route is absent**

Run: `uv run pytest tests/test_agent_api.py -q`
Expected: FAIL with 404 or missing test fixtures before route wiring.

- [ ] **Step 3: Implement the route and provider injection seam**

Create an `APIRouter(prefix="/agent")`. Depend on the existing API-key dependency or equivalent `authenticate_request` check before constructing context/provider calls. Generate a request ID. Validate the request model, build section context, construct server-owned instructions, and call the provider exactly once. Filter returned evidence against `allowed_evidence`; if the provider references an unknown evidence ID, return degraded with `invalid_evidence` rather than accepting it. Return a complete response only after advisory validation succeeds. Return degraded responses for missing configuration and `AgentProviderError` using stable non-secret diagnostic codes.

Include the router under the existing `/api/v1` router without changing existing endpoint behavior.

- [ ] **Step 4: Run focused API tests and existing API regression tests**

Run: `uv run pytest tests/test_agent_api.py tests/test_api.py -q`
Expected: PASS, with any model-bundle skips matching the existing suite.

- [ ] **Step 5: Commit the endpoint integration**

```bash
git add api/agent/routes.py api/routes.py tests/test_agent_api.py
git commit -m "feat: add authenticated agent insights endpoint"
```

### Task 4: Document configuration and run the complete verification set

**Files:**
- Modify: `docs/api.md` (agent endpoint, configuration, privacy/degraded behavior)
- Modify: `.env.example` (non-secret agent settings and placeholder key)
- Test: `tests/test_agent_api.py` (configuration-disabled case if not already covered)

**Interfaces:**
- Consumes: completed endpoint and configuration from Tasks 1–3.
- Produces: operator documentation and final verification evidence.

- [ ] **Step 1: Add the disabled-configuration test**

```python
def test_agent_without_provider_configuration_is_degraded(client, sample_analysis, monkeypatch):
    monkeypatch.delenv("AGENT_API_KEY", raising=False)
    response = client.post(
        "/api/v1/agent/insights",
        headers={"Authorization": "test-key"},
        json=agent_request(sample_analysis),
    )
    assert response.status_code == 200
    assert response.json()["status"] == "degraded"
    assert response.json()["diagnostics"]["errors"] == ["agent_not_configured"]
```

- [ ] **Step 2: Run the new disabled-configuration test**

Run: `uv run pytest tests/test_agent_api.py::test_agent_without_provider_configuration_is_degraded -q`
Expected: PASS.

- [ ] **Step 3: Document setup and boundaries**

Add the endpoint to `docs/api.md`, including the request example, allowed sections, OpenAI/Groq environment configuration, safe-context restriction, no-retry behavior, and degraded response semantics. Add only non-secret variable names/defaults to `.env.example`; do not add a real key.

- [ ] **Step 4: Run formatting/static checks available in the repository**

Run: `uv run python -m compileall api tests`; `git diff --check`
Expected: exit code 0.

- [ ] **Step 5: Run targeted and full tests**

Run: `uv run pytest tests/test_agent_context.py tests/test_agent_provider.py tests/test_agent_api.py -q` then `uv run pytest -q`
Expected: all applicable tests pass; Docker-gated tests may skip according to existing behavior.

- [ ] **Step 6: Inspect the final diff and commit documentation**

```bash
git diff --stat HEAD~3..HEAD
git diff --check
git status --short
git add docs/api.md .env.example tests/test_agent_api.py
git commit -m "docs: document agent insights configuration"
```

## Plan Self-Review

- Spec coverage: contracts/context (Task 1), provider/configuration (Task 2), route/auth/degraded behavior (Task 3), documentation and full verification (Task 4).
- Security coverage: API-key gate, allowlisted fields, no sensitive forwarding/logging, bounded HTTP, no tools/actions, no retries.
- Type consistency: route consumes `AgentInsightRequest`, context returns `dict[str, object]`, provider returns validated advisory data, and response uses `AgentInsightResponse`.
- Deferred items remain outside this plan: streaming, history, persistence, failover, tools, and dashboard integration.

# Focused SecureMailScope Agent Memory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the stateless insights endpoint into a read-only SecureMailScope Agent chatbot with bounded rolling memory and one-step-at-a-time guidance per analysis history item.

**Architecture:** Persist one structured `AgentMemoryRecord` per `analysis.request_id`; do not persist messages or transcripts. The route loads safe analysis context and memory, calls the existing OpenAI-compatible provider once, applies deterministic scope/focus policy to the typed advisory, then persists the replacement memory state and returns the answer plus chatbot state.

**Tech Stack:** Python 3.11, FastAPI, Pydantic v2, SQLAlchemy async sessions, SQLite/Postgres JSON columns, standard-library `urllib`, pytest.

**Spec:** `docs/superpowers/specs/2026-09-10-agent-insights-design.md`

## Global Constraints

- The endpoint is `POST /api/v1/agent/insights`.
- The agent identity is always `SecureMailScope Agent`.
- Memory is keyed by `analysis.request_id`; each history item has an isolated thread.
- Store only bounded structured rolling memory; never store raw questions, full answers, prompts, provider responses, email content, or credentials.
- First analysis response proposes exactly one active verification step.
- A follow-up such as `yes, start with step 1` stays on the existing step.
- The server does not advance steps until an explicit completion/result signal is present.
- Unrelated code-generation requests return the fixed scope response without a provider call or memory mutation.
- Deterministic findings, evidence, and existing verdict remain authoritative.
- Keep the Python provider adapter; do not add a TypeScript service or Vercel AI SDK runtime.
- Memory summary maximum is 6,000 characters; assembled provider context maximum is 24,000 characters.
- Provider failures are bounded, non-secret, and never retried.

---

### Task 1: Add typed memory/focus contracts and persistence model

**Files:**
- Modify: `api/schemas.py` near the existing agent schemas
- Modify: `api/database.py` after `AnalysisRecord`
- Modify: `api/config.py` agent settings
- Test: `tests/test_agent_memory.py`

**Interfaces:**
- Consumes: existing `AnalysisResponse`, `AnalysisRecordResponse`, and `AgentSection`.
- Produces: `AgentStepStatus`, `AgentActiveStep`, `AgentMemoryState`, `AgentInsightResponse` memory fields, `AgentAdvisory.memory_update`, and `AgentMemoryRecord`.

- [ ] **Step 1: Write failing schema/model tests**

```python
from api.schemas import AgentActiveStep, AgentAdvisory, AgentMemoryState


def test_memory_state_is_bounded_and_typed():
    state = AgentMemoryState(
        summary="CERT-003 is being verified.",
        facts=["CERT-003 is deterministic"],
        active_step=AgentActiveStep(
            id="verify-certificate-chain",
            title="Verify the certificate chain",
            status="in_progress",
            evidence=["CERT-003"],
        ),
    )
    assert state.active_step.id == "verify-certificate-chain"
    assert state.completed_steps == []


def test_advisory_requires_memory_update():
    state = AgentMemoryState(summary="CERT-003 is being verified.")
    advisory = AgentAdvisory(
        answer="Verify the chain.",
        recommendations=["Verify the chain."],
        evidence=["CERT-003"],
        active_step=None,
        memory_update=state,
    )
    assert advisory.memory_update.summary == "CERT-003 is being verified."
```

- [ ] **Step 2: Run the focused tests and verify the new contracts are absent**

Run: `uv run pytest tests/test_agent_memory.py -q`
Expected: collection or validation failure because the new agent memory types do not exist.

- [ ] **Step 3: Implement bounded Pydantic contracts and the SQLAlchemy row**

Add these exact bounded fields:

```python
class AgentMemoryState(BaseModel):
    summary: str = Field(default="", max_length=6000)
    facts: list[str] = Field(default_factory=list, max_length=12)
    completed_steps: list[str] = Field(default_factory=list, max_length=12)
    active_step: AgentActiveStep | None = None
    pending_questions: list[str] = Field(default_factory=list, max_length=8)
```

`AgentActiveStep` has `id`, `title`, `status` (`proposed`, `in_progress`, `waiting_for_result`, `completed`), and up to 10 evidence strings. `AgentAdvisory` gains `active_step` and required `memory_update`. `AgentInsightResponse` gains `agent_name`, `thread_id`, `memory_revision`, `memory_persisted`, and nullable `active_step`; keep `recommendations` for compatibility but cap provider output to one focused recommendation.

Add `AgentMemoryRecord` with a unique indexed `analysis_request_id`, JSON `memory`, integer `revision`, and timezone-aware `created_at`/`updated_at`. `Base.metadata.create_all` in `init_db()` must create it for new and existing deployments. Add `AGENT_MEMORY_MAX_CHARS=6000` and `AGENT_CONTEXT_MAX_CHARS=24000` settings.

- [ ] **Step 4: Run schema/model tests**

Run: `uv run pytest tests/test_agent_memory.py -q`
Expected: PASS.

- [ ] **Step 5: Commit the contracts and persistence model**

```bash
git add api/schemas.py api/database.py api/config.py tests/test_agent_memory.py
git commit -m "feat: add focused agent memory contracts"
```

### Task 2: Implement memory repository, scope guard, and step transition policy

**Files:**
- Create: `api/agent/memory.py`
- Create: `api/agent/policy.py`
- Test: `tests/test_agent_memory.py`
- Test: `tests/test_agent_policy.py`

**Interfaces:**
- Consumes: `AsyncSession`, `AgentMemoryRecord`, `AgentMemoryState`, and safe evidence from `api.agent.context`.
- Produces: `load_memory(db, analysis_request_id) -> tuple[AgentMemoryState, int]`, `save_memory(db, analysis_request_id, state, revision) -> int`, `is_out_of_scope_request(question) -> bool`, `scope_response() -> AgentAdvisory`, and `apply_focus_policy(previous, proposed, question, allowed_evidence) -> AgentMemoryState`.

- [ ] **Step 1: Write failing repository and policy tests**

```python
import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from api.database import Base
from api.agent.memory import load_memory, save_memory
from api.agent.policy import apply_focus_policy, is_out_of_scope_request
from api.schemas import AgentActiveStep, AgentMemoryState


@pytest.mark.anyio
async def test_memory_round_trips_without_chat_transcript():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    async with async_sessionmaker(engine, expire_on_commit=False)() as db_session:
        state = AgentMemoryState(summary="Verify CERT-003.")
        revision = await save_memory(db_session, "analysis-1", state, 0)
        loaded, loaded_revision = await load_memory(db_session, "analysis-1")
    await engine.dispose()
    assert loaded == state
    assert loaded_revision == revision
    assert not hasattr(loaded, "messages")


def test_confirmation_keeps_the_current_step():
    current = AgentMemoryState(
        summary="Verify the chain.",
        active_step=AgentActiveStep(
            id="verify-chain", title="Verify the chain", status="in_progress", evidence=["CERT-003"]
        ),
    )
    proposed = AgentMemoryState(
        summary="Now review model disagreement.",
        active_step=AgentActiveStep(
            id="review-models", title="Review model disagreement", status="in_progress", evidence=[]
        ),
    )
    result = apply_focus_policy(current, proposed, "yes, start with step 1", {"CERT-003"})
    assert result.active_step.id == "verify-chain"


def test_python_request_is_out_of_scope():
    assert is_out_of_scope_request("write a Python script to parse this") is True
    assert is_out_of_scope_request("how do I verify CERT-003?") is False
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `uv run pytest tests/test_agent_memory.py tests/test_agent_policy.py -q`
Expected: FAIL because repository/policy functions are not implemented.

- [ ] **Step 3: Implement the repository with revisioned replacement state**

`load_memory` returns an empty `AgentMemoryState` and revision `0` when no row exists. `save_memory` validates the state, serializes only `state.model_dump(mode="json")`, updates an existing row under a row lock, increments the revision, and flushes before returning. Use the unique `analysis_request_id` constraint; never add a message table or store the request/answer text.

- [ ] **Step 4: Implement deterministic scope and focus policy**

Use a small server-owned classifier for unrelated code-generation verbs combined with language/file terms (`write`, `generate`, `debug`, `implement` plus `python`, `javascript`, `script`, or `code`). Return the exact fixed response for out-of-scope requests:

```text
I’m SecureMailScope Agent. I can explain this security analysis and guide the current verification step, but writing unrelated Python code is outside my capability.
```

`apply_focus_policy` must:

1. Keep the stored active step for ordinary confirmations.
2. Preserve the current step when the question contains no explicit completion/result signal.
3. Accept a proposed next step only when the question explicitly reports completion/result words such as `verified`, `confirmed`, `completed`, `found`, or `result` and the proposed state includes the prior step in `completed_steps`.
4. Filter every state evidence value against `allowed_evidence`.
5. Never change authoritative risk/finding fields.

- [ ] **Step 5: Run repository and policy tests**

Run: `uv run pytest tests/test_agent_memory.py tests/test_agent_policy.py -q`
Expected: PASS.

- [ ] **Step 6: Commit memory/policy behavior**

```bash
git add api/agent/memory.py api/agent/policy.py tests/test_agent_memory.py tests/test_agent_policy.py
 git commit -m "feat: add bounded agent memory and focus policy"
```

### Task 3: Extend the provider payload for focused chatbot responses

**Files:**
- Modify: `api/agent/provider.py`
- Modify: `api/agent/context.py` (bounded context serialization helper)
- Modify: `tests/test_agent_provider.py`

**Interfaces:**
- Consumes: `AgentAdvisory`, `AgentMemoryState`, `AgentActiveStep`, and existing OpenAI/Groq provider settings.
- Produces: provider parsing for `memory_update` and `active_step`, plus `build_agent_user_content(section, question, context, memory, max_chars) -> str`.

- [ ] **Step 1: Write failing provider tests for memory and bounded payloads**

```python
import json
import pytest
from api.agent.provider import AgentProviderError, OpenAICompatibleProvider
from tests.test_agent_provider import FakeResponse
from api.agent.provider import build_agent_user_content
from api.schemas import AgentMemoryState


def test_provider_parses_memory_update_and_active_step():
    completion = {
        "choices": [{"message": {"content": json.dumps({
        "answer": "Verify the chain.",
        "recommendations": ["Verify the chain."],
        "evidence": ["CERT-003"],
        "active_step": {"id": "verify-chain", "title": "Verify the chain", "status": "in_progress", "evidence": ["CERT-003"]},
        "memory_update": {"summary": "Verifying CERT-003.", "facts": ["CERT-003 is deterministic"], "completed_steps": [], "active_step": {"id": "verify-chain", "title": "Verify the chain", "status": "in_progress", "evidence": ["CERT-003"]}, "pending_questions": []},
    })}}]}
    provider = OpenAICompatibleProvider(
        "groq", "model", "secret", "https://example.test/v1", 3, 10000,
        lambda *_args, **_kwargs: FakeResponse(json.dumps(completion).encode()),
    )
    advisory = provider.generate("system", "user")
    assert advisory.memory_update.active_step.id == "verify-chain"


def test_user_content_rejects_over_budget_payload():
    with pytest.raises(AgentProviderError, match="context_too_large"):
        build_agent_user_content("risk", "question", {"x": "y"}, AgentMemoryState(summary="x" * 6000), 1000)
```

- [ ] **Step 2: Run focused provider tests and verify the new fields fail**

Run: `uv run pytest tests/test_agent_provider.py -q`
Expected: FAIL because the advisory schema/parser and bounded builder do not yet support memory.

- [ ] **Step 3: Extend the provider parser and server-owned instructions**

Keep the current Groq-compatible request shape (`temperature`, `top_p`, `max_completion_tokens`, `stream=false`, and Groq `reasoning_effort`). Update instructions to require JSON containing `answer`, at most one recommendation, `evidence`, `active_step`, and `memory_update`. Parse fenced JSON as today and validate the typed nested state. The provider must not receive the API key or raw request envelope.

Implement `build_agent_user_content` with compact JSON serialization. If the assembled payload exceeds `AGENT_CONTEXT_MAX_CHARS`, return `AgentProviderError("context_too_large")` before the network call; do not silently drop authoritative analysis fields.

- [ ] **Step 4: Run provider tests**

Run: `uv run pytest tests/test_agent_provider.py -q`
Expected: PASS.

- [ ] **Step 5: Commit provider contract changes**

```bash
git add api/agent/provider.py api/agent/context.py tests/test_agent_provider.py
git commit -m "feat: add focused memory to agent provider"
```

### Task 4: Integrate async chatbot memory into the insights route

**Files:**
- Modify: `api/agent/routes.py`
- Modify: `api/agent/policy.py` if transition handling needs route-specific completion parsing
- Modify: `tests/test_agent_api.py`

**Interfaces:**
- Consumes: `load_memory`, `save_memory`, `build_agent_context`, `build_agent_user_content`, `create_agent_provider`, `apply_focus_policy`, and existing auth dependencies.
- Produces: async `POST /api/v1/agent/insights` with complete chatbot state and explicit degraded/provider/memory diagnostics.

- [ ] **Step 1: Write failing API tests for the conversation flow**

```python
def post_insight(client, analysis, question):
    return client.post(
        "/api/v1/agent/insights",
        json={"analysis": analysis, "section": "risk", "question": question},
    )


def test_first_turn_persists_one_active_step(client, sample_analysis):
    response = post_insight(client, sample_analysis, "Explain the risk drivers.")
    body = response.json()
    assert body["agent_name"] == "SecureMailScope Agent"
    assert body["active_step"]["status"] == "in_progress"
    assert body["memory_revision"] == 1
    assert body["memory_persisted"] is True


def test_confirmation_stays_on_same_step(client, history_analysis):
    first = post_insight(client, history_analysis, "Explain the risk.")
    second = post_insight(client, history_analysis, "yes, start with step 1")
    assert second.json()["active_step"]["id"] == first.json()["active_step"]["id"]
    assert second.json()["memory_revision"] == 2


def test_out_of_scope_request_does_not_call_provider(client, history_analysis):
    response = post_insight(client, history_analysis, "write Python code for this")
    assert response.json()["agent_name"] == "SecureMailScope Agent"
    assert "outside my capability" in response.json()["answer"]
    assert response.json()["memory_persisted"] is False
```

- [ ] **Step 2: Run the new API tests and verify the stateless route fails them**

Run: `uv run pytest tests/test_agent_api.py -q`
Expected: FAIL because the current route is synchronous/stateless and does not return memory fields.

- [ ] **Step 3: Convert the route to async and wire the memory flow**

Add `db: AsyncSession = Depends(get_db)`. Authenticate before provider/memory work. Use `analysis.request_id` as `thread_id`, load memory, run the scope guard, and return the fixed scope response before provider creation for unrelated coding requests. For in-scope requests, call the synchronous provider through `await asyncio.to_thread(provider.generate, system, user_content)`.

After provider validation, call `apply_focus_policy`, persist the replacement state, and return `AgentInsightResponse` with `agent_name`, provider/model, revision, persistence status, and active step. On provider failure return the existing degraded shape. On memory persistence/update failure, retain the prior memory, return the validated answer with `memory_persisted=false`, and add only a stable diagnostic.

- [ ] **Step 4: Run focused API tests and existing API regression tests**

Run: `uv run pytest tests/test_agent_api.py tests/test_api.py -q`
Expected: PASS, including existing live/persisted analysis input behavior.

- [ ] **Step 5: Commit route integration**

```bash
git add api/agent/routes.py api/agent/policy.py tests/test_agent_api.py
 git commit -m "feat: make agent insights a focused chatbot"
```

### Task 5: Document memory behavior, deploy configuration, and verify the full boundary

**Files:**
- Modify: `docs/api.md`
- Modify: `.env.example`
- Modify: `deploy/vps/env.example`
- Modify: `compose.yaml` if the new settings are not already passed through
- Test: `tests/test_agent_memory.py`, `tests/test_agent_api.py`

**Interfaces:**
- Consumes: completed chatbot endpoint and `AGENT_MEMORY_MAX_CHARS`/`AGENT_CONTEXT_MAX_CHARS` settings.
- Produces: operator/frontend documentation and deployment-ready configuration.

- [ ] **Step 1: Add configuration and persistence-boundary assertions**

```python
def test_memory_limit_is_configured():
    state = AgentMemoryState(summary="x" * 6000)
    assert len(state.summary) == 6000


def test_history_item_is_the_thread_key(client, history_analysis):
    response = client.post(
        "/api/v1/agent/insights",
        json={"analysis": history_analysis, "section": "risk", "question": "Explain the risk."},
    )
    assert response.json()["thread_id"] == "analysis-request"
```

- [ ] **Step 2: Document the chatbot request/response and state machine**

Update `docs/api.md` to explain that frontend callers pass the same history record on every turn, the backend keys memory by `analysis.request_id`, only one active step is guided at a time, code-generation requests are out of scope, and memory contains no transcript. Document `AGENT_MEMORY_MAX_CHARS` and `AGENT_CONTEXT_MAX_CHARS`.

Add the same non-secret settings to `.env.example` and `deploy/vps/env.example`. Ensure `compose.yaml` passes both settings to the API container alongside existing `AGENT_*` provider settings.

- [ ] **Step 3: Run all focused checks**

Run: `uv run pytest tests/test_agent_memory.py tests/test_agent_policy.py tests/test_agent_provider.py tests/test_agent_context.py tests/test_agent_api.py -q`; `uv run python -m compileall -q api tests`; `git diff --check`
Expected: all focused tests pass and compile/diff checks exit 0.

- [ ] **Step 4: Run the full regression suite**

Run: `uv run pytest -q`
Expected: all applicable tests pass; Docker-gated tests may skip according to existing behavior.

- [ ] **Step 5: Verify deployment interpolation without printing secrets**

Run:

```bash
tmp=$(mktemp)
docker compose config --format json > "$tmp"
uv run python - "$tmp" <<'PY'
import json, sys
config = json.load(open(sys.argv[1]))
env = config["services"]["api"]["environment"]
print({
    key: "<set>" if key == "AGENT_API_KEY" and env.get(key) else env.get(key)
    for key in (
        "AGENT_PROVIDER", "AGENT_MODEL", "AGENT_API_KEY", "AGENT_BASE_URL",
        "AGENT_MEMORY_MAX_CHARS", "AGENT_CONTEXT_MAX_CHARS",
    )
})
PY
rm -f "$tmp"
```

Expected: provider/model/base URL and memory/context limits are present; the API key is reported only as `<set>`.

- [ ] **Step 6: Inspect and commit the final documentation/configuration diff**

```bash
git diff --stat
git diff --check
git status --short
git add docs/api.md .env.example deploy/vps/env.example compose.yaml tests/test_agent_memory.py tests/test_agent_api.py
git commit -m "docs: document focused agent memory flow"
```

## Plan Self-Review

- Spec coverage: contracts/database (Task 1), persistence/scope/focus policy (Task 2), provider/context limits (Task 3), chatbot route and state transitions (Task 4), docs/deployment/full verification (Task 5).
- Chatbot behavior: first turn analyzes one risk path; confirmation stays on the active step; explicit result is required before advancement; out-of-scope code requests get a fixed role response.
- Privacy: no transcript, prompt, full answer, email data, credentials, or raw provider response is persisted or logged.
- Provider compatibility: existing standard-library OpenAI/Groq adapter remains; no Vercel AI SDK runtime is introduced.
- Failure handling: provider failures degrade without fabricated content; memory failures preserve valid answers and expose `memory_persisted=false`.
- Type consistency: `AgentAdvisory.memory_update` is `AgentMemoryState`; route passes it through `apply_focus_policy`; repository stores the same JSON state and returns its revision.
- Plan review passed; all implementation steps specify files, interfaces, commands, and expected outcomes.

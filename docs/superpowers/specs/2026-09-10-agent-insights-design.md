# Authorized Agent Insights Design

## Status

Approved design extension; implementation not started.

## Objective

Add a read-only, authenticated SecureMailScope chatbot agent that explains an existing analysis, guides the analyst through one verification step at a time, and retains a bounded rolling memory per history item without storing the full chat transcript. The endpoint supports OpenAI and Groq through one OpenAI-compatible provider boundary while deterministic ML/rule output remains authoritative.

## Scope

### In scope

- `POST /api/v1/agent/insights`.
- API-key authentication using the existing `SECUREMAIL_API_KEY` boundary.
- Live `analysis-response.v1` and persisted `AnalysisRecordResponse` input support.
- Typed response and memory contracts with versioned envelopes.
- Allowlisted analysis context plus bounded rolling memory keyed by `analysis.request_id`.
- One focused active verification step at a time.
- Explicit completion/result gating before advancing to another step.
- Fixed SecureMailScope Agent identity and deterministic out-of-scope refusal for unrelated code-generation requests.
- OpenAI-compatible, non-streaming provider adapter configurable for OpenAI or Groq.
- Bounded context, timeout, response size, and memory limits.
- Persistence of only structured rolling memory; no full chat messages, prompts, provider responses, email content, or credentials.
- Unit/API tests using typed fake providers; no live provider calls in tests.
- Configuration and API documentation.

### Out of scope

- Streaming responses.
- Agent tools, tool loops, MCP, browsing, or external actions.
- Automatic remediation or mutation of analysis findings.
- Accepting raw PCAP, email content, credentials, tokens, or arbitrary section data.
- Persisting a chat transcript, raw questions, raw answers, prompts, or provider responses.
- Changing the existing ML verdict, risk class, deterministic rules, or analysis persistence flow.
- A separate TypeScript service or Vercel AI SDK runtime. The existing Python service owns persistence and provider calls; the SDK's message-compaction concepts are not required for a summary-only memory.

## API contract

### Request

`POST /api/v1/agent/insights`

```json
{
  "schema_version": "agent-insight-request.v1",
  "analysis": { "...analysis-response.v1 or AnalysisRecordResponse...": true },
  "section": "risk",
  "question": "Start with step 1."
}
```

`section` is one of `overview`, `risk`, `tls`, `certificate`, or `findings`.
`question` is bounded to 4,000 characters and treated as untrusted text.
The supplied analysis validates as either the live `analysis-response.v1` shape or the persisted `AnalysisRecordResponse` shape returned by history endpoints. Persisted records are normalized from stored risk, trigger, model, TLS, and certificate fields. The server rebuilds model context from an allowlist and never serializes the whole request.

The memory key is the analysis item's `request_id`. Each history item has an isolated chatbot thread. The frontend does not need to invent a second conversation ID.

### Response

```json
{
  "schema_version": "agent-insight-response.v1",
  "request_id": "uuid",
  "status": "complete",
  "agent_name": "SecureMailScope Agent",
  "section": "risk",
  "answer": "Start by verifying the certificate chain for CERT-003...",
  "recommendations": ["Verify the certificate chain for CERT-003."],
  "evidence": ["CERT-003", "pcap:tls"],
  "provider": "groq",
  "model": "configured-model",
  "thread_id": "analysis-request-id",
  "memory_revision": 2,
  "memory_persisted": true,
  "active_step": {
    "id": "verify-certificate-chain",
    "title": "Verify the certificate chain",
    "status": "in_progress"
  },
  "diagnostics": {}
}
```

`status` is `complete` or `degraded`. A provider/configuration failure returns an explicit degraded response without fabricated AI content. A validated answer remains `status=complete` when memory persistence or memory-update validation fails, with `memory_persisted=false` and a stable diagnostic; losing memory must not discard a valid read-only explanation.

The `agent_name` is server-owned and always `SecureMailScope Agent`. Evidence values are restricted to finding IDs or safe evidence references present in the supplied analysis. The AI cannot emit or overwrite the authoritative verdict.

## Chatbot focus policy

The agent is a security-analysis chatbot, not a general coding or operations assistant.

### First analysis turn

The first successful turn explains the authoritative risk drivers and proposes exactly one active verification step. It may include supporting evidence, but it must not present a parallel list of unrelated next steps.

### Follow-up guidance

A follow-up such as `yes, start with step 1` keeps the existing active step and responds with:

- why the step matters;
- a short, ordered checklist for that step;
- what evidence/result the analyst should bring back; and
- no new step until the current step is explicitly completed or a result is supplied.

The server owns the focus state. A model response that silently changes the active step before completion is rejected or normalized back to the stored step.

### Completion and advancement

The next step may be proposed only when the current user message contains an explicit completion/result signal and the current step has a stored or validated result. `completed` is not inferred from a generic acknowledgement. Until then, `active_step` remains unchanged and the response stays focused on that step.

### Out-of-scope requests

A deterministic scope guard runs before the provider call for unrelated code-generation requests such as writing Python scripts. It returns a server-authored response:

> I’m SecureMailScope Agent. I can explain this security analysis and guide the current verification step, but writing unrelated Python code is outside my capability.

The refusal is a normal chatbot response, not an empty response and not a provider failure. It does not advance or corrupt memory.

## Rolling memory contract

Only one bounded state document is persisted per analysis request. It contains no transcript:

```json
{
  "summary": "The analyst is investigating CERT-003, an invalid certificate chain.",
  "facts": ["CERT-003 is a deterministic finding"],
  "completed_steps": [],
  "active_step": {
    "id": "verify-certificate-chain",
    "title": "Verify the certificate chain",
    "status": "in_progress",
    "evidence": ["CERT-003", "pcap:tls"]
  },
  "pending_questions": ["Is the presented chain trusted by the intended client?"]
}
```

Each model response includes a typed `memory_update`. It is a replacement state, not an append operation. The server validates evidence against the current analysis, enforces field/list/character bounds, and preserves the stored active step when the current request has not completed it.

The database row is keyed uniquely by `analysis_request_id` and stores:

- structured memory JSON;
- monotonic revision;
- created and updated timestamps.

No raw question, full answer, provider response, prompt, or prior message list is stored. Memory is treated as untrusted context when sent back to the model; it cannot override deterministic findings or system policy.

## Safe context and window policy

The provider input contains only:

1. server-owned SecureMailScope Agent instructions;
2. the bounded rolling memory;
3. the current section-specific analysis context; and
4. the current user question.

The analysis context continues to exclude raw records, packet contents, email data, credentials, private keys, authorization headers, and unknown top-level fields. The memory summary is capped at 6,000 characters and the assembled user context is capped at 24,000 characters before provider submission. Oversized model memory updates are rejected, the prior memory is retained, and the validated answer is returned with `memory_persisted=false`; context is never allowed to grow with conversation length.

## Provider boundary

Define a typed provider protocol with one operation: generate the focused advisory and replacement memory state from server instructions and one bounded user payload. Implement one OpenAI-compatible HTTP adapter using the standard library. Both providers use the same `/chat/completions` shape; provider selection changes configuration, not application behavior.

Configuration:

- `AGENT_PROVIDER`: `openai` or `groq`.
- `AGENT_MODEL`: required when the agent endpoint is enabled.
- `AGENT_API_KEY`: server-side secret; never logged or returned.
- `AGENT_BASE_URL`: provider API base URL, defaulting to the selected provider's documented base URL.
- `AGENT_TIMEOUT_SECONDS`: bounded request timeout, default 20 seconds.
- `AGENT_MAX_RESPONSE_BYTES`: bounded response body, default 256 KiB.
- `AGENT_MEMORY_MAX_CHARS`: rolling memory ceiling, default 6,000.
- `AGENT_CONTEXT_MAX_CHARS`: assembled provider context ceiling, default 24,000.

The endpoint is degraded when configuration is incomplete, the request times out, the upstream returns a non-success response, the body exceeds the limit, JSON is invalid, or the advisory/memory schema fails validation. Logs contain request ID, provider, model, and stable error code only; no prompt, response, API key, or sensitive context.

## Data flow

1. Authenticate with the existing API-key dependency.
2. Validate the versioned request and live/persisted analysis shape.
3. Run the deterministic scope guard.
4. Load the memory row for `analysis.request_id`.
5. Build the section-specific allowlisted analysis context.
6. Assemble bounded instructions, memory, context, and current question.
7. Call the configured provider once through a bounded worker hop.
8. Parse and validate the advisory and replacement memory state.
9. Validate evidence and enforce the active-step transition policy.
10. Persist the replacement memory row atomically with its next revision; retain prior memory when persistence or memory bounds fail.
11. Return the answer, focused state, evidence, and memory persistence metadata.

No retries are used: duplicate AI calls can incur cost and are not idempotent from a billing perspective. The endpoint remains read-only and cannot execute remediation.

## Testing and acceptance

Acceptance requires:

- First turn returns a useful analysis, exactly one active step, and a persisted revision.
- A follow-up using the same persisted history item receives the prior summary and stays on the active step.
- `yes, start with step 1` produces focused guidance for step 1 and does not advance.
- A generic acknowledgement cannot advance the step; an explicit completion/result can.
- A request to write Python code returns the fixed SecureMailScope Agent scope response without a provider call or memory mutation.
- Persisted memory contains only bounded structured state and never a transcript, prompt, or full answer.
- Memory and assembled context limits are enforced.
- Evidence cannot reference findings or safe evidence refs absent from the analysis.
- Provider selection/configuration remains unchanged for OpenAI and Groq.
- Provider/configuration failures remain explicit and non-secret.
- Existing API tests continue to pass.
- Documentation describes the chatbot contract, memory retention boundary, focus policy, configuration, and degraded behavior.

## Deferred

Streaming, provider failover, tool execution, persisted insight audit records, cross-analysis memory, user-authenticated memory ownership, and dashboard UI changes are deferred until the bounded per-analysis chatbot flow is proven.

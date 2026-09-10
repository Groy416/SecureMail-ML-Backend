# Authorized Agent Insights Design

## Status

Approved design; implementation not started.

## Objective

Add a read-only, authenticated AI-insights endpoint that explains an existing SecureMail analysis and recommends analyst next steps. The endpoint must support OpenAI and Groq through one OpenAI-compatible provider boundary while keeping deterministic ML/rule output authoritative.

## Scope

### In scope

- `POST /api/v1/agent/insights`.
- API-key authentication using the existing `SECUREMAIL_API_KEY` boundary.
- Typed request/response models with versioned envelopes.
- Allowlisted section context derived from an existing live `analysis-response.v1` or persisted `AnalysisRecordResponse` payload.
- OpenAI-compatible, non-streaming provider adapter configurable for OpenAI or Groq.
- Bounded timeout, response-size limit, safe degraded states, and structured diagnostics.
- Unit/API tests using a typed fake provider; no live provider calls in tests.
- Configuration and API documentation.

### Out of scope

- Streaming responses.
- Agent tools, tool loops, MCP, browsing, or external actions.
- Automatic remediation or mutation of analysis findings.
- Accepting raw PCAP, email content, credentials, tokens, or arbitrary section data.
- Persisting prompts, provider responses, or sensitive context.
- Changing the existing ML verdict, risk class, deterministic rules, or analysis persistence flow.

## API contract

### Request

`POST /api/v1/agent/insights`

```json
{
  "schema_version": "agent-insight-request.v1",
  "analysis": { "...existing analysis-response.v1 or AnalysisRecordResponse...": true },
  "section": "risk",
  "question": "Explain the main risk drivers and what should be checked next."
}
```

`section` is one of `overview`, `risk`, `tls`, `certificate`, or `findings`.
`question` is bounded to a finite length and treated as untrusted text.
The supplied analysis must validate as either the live `analysis-response.v1` shape or the persisted `AnalysisRecordResponse` shape returned by history endpoints. Persisted records are normalized from stored risk, trigger, model, TLS, and certificate fields. The implementation must rebuild the model context from an allowlist rather than serialize the whole request.

### Response

```json
{
  "schema_version": "agent-insight-response.v1",
  "request_id": "uuid",
  "status": "complete",
  "section": "risk",
  "answer": "...",
  "recommendations": ["..."],
  "evidence": ["TLS-001"],
  "provider": "openai",
  "model": "configured-model",
  "diagnostics": {}
}
```

`status` is `complete` or `degraded`. A degraded response has no fabricated answer; `answer` is `null`, recommendations and evidence are empty, and diagnostics contains a stable non-secret error code. HTTP authentication/validation errors remain normal `401`/`422` errors.

The provider response must validate into the advisory shape. Evidence values are restricted to finding IDs or allowlisted safe field references present in the supplied analysis. The AI cannot emit or overwrite the authoritative verdict.

## Safe context boundary

The context builder selects only the fields needed for the requested section:

- `overview`: request/session identifiers safe for display, protocol, posture, verdict, risk score, action, finding summaries, and coverage/diagnostics.
- `risk`: verdict, score, action, model/rule signals, and finding summaries.
- `tls`: protocol, TLS details, cryptographic posture, and TLS-related findings.
- `certificate`: certificate details and certificate-related findings.
- `findings`: deterministic finding IDs, titles, severity, conditions, evidence references, and remediation guidance already present in the result.

It excludes raw records, provenance details not needed for explanation, packet contents, email data, credentials, private keys, authorization headers, and arbitrary top-level fields. The fixed system instruction says the context is evidence, not instructions; findings/rules are authoritative; uncertainty must be stated; and no action may be claimed as executed.

## Provider boundary

Define a small typed provider protocol with one operation: generate an advisory from system instructions and user content. Implement one OpenAI-compatible HTTP adapter using the standard library. Both providers use the same `/chat/completions` shape; provider selection changes configuration, not application behavior.

Configuration:

- `AGENT_PROVIDER`: `openai` or `groq`.
- `AGENT_MODEL`: required when the agent endpoint is enabled.
- `AGENT_API_KEY`: server-side secret; never logged or returned.
- `AGENT_BASE_URL`: provider API base URL, defaulting to the selected provider's documented base URL.
- `AGENT_TIMEOUT_SECONDS`: bounded request timeout, default 20 seconds.
- `AGENT_MAX_RESPONSE_BYTES`: bounded response body, default 256 KiB.

The endpoint is degraded when configuration is incomplete, the request times out, the upstream returns a non-success response, the body exceeds the limit, JSON is invalid, or the advisory schema fails validation. Logs contain request ID, provider, model, and stable error code only; no prompt, response, API key, or sensitive context.

## Data flow

1. Authenticate with the existing API-key dependency.
2. Generate a request ID.
3. Validate the versioned request and safe analysis response.
4. Build the section-specific allowlisted context.
5. Construct server-owned instructions and untrusted user question.
6. Call the configured provider once with a bounded timeout and response size.
7. Parse and validate the structured advisory.
8. Return complete or explicit degraded response.

No retries are used in the first slice: duplicate AI calls can incur cost and are not idempotent from a billing perspective. The request is read-only and is not persisted.

## Testing and acceptance

Acceptance requires:

- Valid authenticated request returns a validated complete response with a fake provider.
- Missing/invalid API key is rejected before provider invocation.
- Each section only receives its allowlisted fields; sensitive and unknown fields are absent.
- Provider is selected/configured without changing endpoint code.
- Timeout, non-success response, malformed JSON, oversized response, and invalid advisory are degraded without leaking provider data.
- AI evidence cannot reference findings absent from the analysis.
- Existing API tests continue to pass.
- Documentation describes configuration, contract, privacy boundary, and degraded behavior.

## Deferred

Streaming, conversation history, persisted insight audit records, provider failover, tool execution, and UI integration are deferred until the non-streaming read-only contract is proven.

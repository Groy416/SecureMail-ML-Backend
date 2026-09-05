# SecureMailScope Frontend–API–ML Dashboard Design

**Status:** Draft design spec for review  
**Date:** 2026-09-05  
**Audience:** Internal security analysts  
**Implementation status:** Design only. The repository currently contains the Python ML pipeline, contracts, and product CLI; it does not yet contain the HTTP API or web frontend.

## 1. Purpose

Define the end-to-end flow chart and frontend dashboard contract for SecureMailScope, from an analyst action in the browser through the API boundary, ML inference, policy precedence, and evidence-linked output.

The existing ML specification remains authoritative for model behavior and security policy:

- [`docs/specs/spec.md`](../../specs/spec.md), especially sections 2, 5, 8, 17, 20, 22, and 23.
- [`README.md`](../../../README.md), especially **API handoff**, **Data contracts**, and **End-to-end flow**.

This document specifies the missing product boundary without changing `session-features.v1` or `ml-result.v1`.

## 2. Design objective and acceptance check

### Objective

Provide an analyst-facing, evidence-first workspace that can submit one authorized session-feature record, display the resulting policy decision, and make model disagreement or degraded operation visible.

### Acceptance check

A representative valid session-feature fixture must be able to follow this path:

```text
browser action
  -> authenticated API request
  -> payload/privacy gate
  -> session-features.v1 validation
  -> pinned model bundle + calibration
  -> ml.pipeline.predict_session(...)
  -> deterministic rules + calibrated XGBoost/Random Forest fusion
  -> optional anomaly signal + explanations
  -> ml-result.v1 response
  -> dashboard risk, findings, evidence, signals, and diagnostics
```

The flow must also show a rejected sensitive payload and a valid degraded result without inventing a prediction or explanation.

## 3. Scope

### In scope

- A contract-first frontend-to-API-to-ML flow chart.
- A synchronous, single-session analysis API boundary.
- An analyst dashboard for one current analysis.
- Risk summary, action, rule findings, evidence references, model signals, feature-level explanations, and diagnostics.
- Loading, empty, success, normal/no-finding, rejected, failed, and degraded states.
- Privacy, authorization, accessibility, responsive layout, and observable failure requirements.
- Testable acceptance criteria for the future API and frontend.

### Out of scope

- Raw PCAP upload or parsing by the browser/API scorer.
- Email bodies, subjects, attachments, credentials, cookies, tokens, TLS key logs, or private keys.
- Automatic blocking, remediation, server configuration changes, or mailbox actions.
- Online training, learned stacking, or model promotion.
- Multi-user administration, tenant management, RBAC beyond an authenticated analyst boundary, or public sharing.
- Durable analysis history, batch orchestration, live streaming, or a packet-payload viewer.
- Replacing the deterministic rule layer with an LLM or UI logic.

The first product slice may use an authorized session-feature fixture or an upstream passive extractor. It must not turn a PCAP upload into an implicit API feature.

## 4. Product principles

1. **Evidence first, AI second.** Deterministic findings and their evidence establish the policy floor. Model output prioritizes and adds context.
2. **The browser presents policy; it does not make policy.** Severity, action, model participation, and degraded status come from the API result.
3. **Raw outputs remain visible.** The UI must not collapse XGBoost, Random Forest, anomaly, and rule signals into one unexplained score.
4. **Unknown is not safe.** Missing, disabled, unavailable, or unsupported signals are shown as such; they are never rendered as a confident negative.
5. **No sensitive-content path.** The UI and API operate on validated session metadata only.
6. **Analyst density over decoration.** Use familiar tables, tabs, disclosure panels, status badges, and clear next actions. Avoid an AI chat surface, decorative charts, and motion that does not aid triage.

## 5. End-to-end flow chart specification

The following Mermaid chart is the source-of-truth shape for the planned integration. Solid arrows are the normal synchronous path. Dashed arrows are explicit degraded or rejected paths.

```mermaid
flowchart LR
    Analyst[Internal analyst] --> Dashboard[Web dashboard]

    Dashboard -->|select authorized session or submit safe fixture| Request[POST /api/v1/analyses]
    Request --> Auth[Authenticate request and authorize capture scope]
    Auth -.->|401 / 403| AuthError[Access error]
    Auth --> Gate[Payload privacy and size gate]
    Gate -.->|sensitive field or forbidden payload| Reject[Rejected request]
    Gate --> Validate[Validate session-features.v1]
    Validate -.->|schema or provenance error| Invalid[Validation error]

    Validate --> Runtime[Load pinned ModelBundle and CalibrationState]
    Runtime -.->|artifact missing or incompatible| RuntimeError[ML unavailable; preserve explicit failure]
    Runtime --> Predict[ml.pipeline.predict_session(...)]

    subgraph Pipeline[Existing Python ML pipeline]
        Predict --> Rules[extract_rule_findings(record)]
        Predict --> Models[predict_model_outputs(...)]
        Models --> Calibrate[Calibrated class probabilities]
        Calibrate --> Fuse[0.60 XGBoost + 0.40 Random Forest]
        Rules --> Policy[Policy precedence and final action]
        Fuse --> Policy
        Models -.->|optional configured model| Anomaly[Anomaly score and threshold]
        Anomaly --> Policy
        Predict --> Explain[explain_session(...)]
        Policy --> Result[MLResult / ml-result.v1]
        Explain -.->|unavailable| ExplainDiagnostic[Explanation diagnostic]
        Explain --> Result
    end

    Result --> Envelope[API response envelope with safe session context]
    RuntimeError -.-> RuleOnly[Optional rule-only degraded result]
    RuleOnly -.-> Envelope
    Reject -.-> Envelope
    Invalid -.-> Envelope
    AuthError -.-> Envelope
    Envelope --> Dashboard

    Dashboard --> Summary[Risk, score, action, policy source]
    Dashboard --> Findings[Rule findings and evidence references]
    Dashboard --> Signals[Model agreement and anomaly state]
    Dashboard --> Explanations[Feature explanations and disclaimer]
    Dashboard --> Diagnostics[Degraded state and recovery action]
```

### 5.1 Ownership by stage

| Stage | Owner | Contract or rule | Failure behavior |
|---|---|---|---|
| Analyst input | Frontend | Safe session-feature record or authorized session selection | Client-side shape check only; server remains authoritative |
| Authentication | API | Deployment identity and capture authorization | Reject unauthenticated or out-of-scope access |
| Privacy gate | API | Existing forbidden-payload policy in `ml.product` | Reject without echoing the sensitive value |
| Schema gate | API | `validate_session(..., strict=True)` and `session-features.v1` | Return structured validation error |
| Runtime loading | API | `load_model_bundle(...)` and `load_calibration_state(...)` | Fail closed for ML; expose artifact/version diagnostic |
| Rule extraction | ML pipeline | `ml.rules.extract_rule_findings` | Preserve evidence-backed finding output |
| Model inference | ML pipeline | `ml.pipeline.predict_session` | Show participating models and diagnostics |
| Calibration/fusion | ML pipeline | Existing calibrated 60/40 XGBoost/Random Forest baseline | Never recompute or reinterpret in UI |
| Policy decision | ML pipeline | Rule severity is the minimum final severity | UI displays, never overrides |
| Explanation | ML pipeline | TreeSHAP/anomaly contribution contract | Return result with explanation-unavailable diagnostic |
| API response | API | `ml-result.v1` plus response envelope | Preserve result and diagnostics separately |
| Rendering | Frontend | Typed response mapping | No fake values; show explicit empty/degraded states |

### 5.2 Current repository alignment

The planned API should call the existing pure boundary:

```python
ml.pipeline.predict_session(
    bundle,
    calibration,
    record,
)
```

That function currently merges deterministic findings, invokes model fusion, and attempts explanations. The API must not duplicate rule severity calculation or implement a second fusion algorithm. The active bundle is `Model_XG_RF`; its default active supervised fusion is calibrated XGBoost plus Random Forest. Isolation Forest is a legacy/optional compatibility path and may be disabled in the active bundle.

## 6. API boundary specification

The API is a thin, authenticated adapter around the Python contracts. Endpoint names below are proposed for the future integration; they do not claim to exist in the current repository.

### 6.1 Synchronous analysis endpoint

```http
POST /api/v1/analyses
Content-Type: application/json
Authorization: deployment-managed credential
```

Request shape:

```text
AnalysisRequest {
  schema_version: "analysis-request.v1"
  record: SessionFeatureRecord // existing session-features.v1 contract
}
```

The request contract is a transport envelope. The inner record must use the existing `SessionFeatureRecord` shape. The inference adapter may supply neutral inference labels when labels are absent, as the current `ml.product.inference_record` path does; client-provided labels must never control the result or be treated as ground truth.

Successful response shape:

```text
AnalysisResponse {
  schema_version: "analysis-response.v1"
  request_id: string
  status: "complete" | "degraded"
  session: SafeSessionContext
  result: MLResult // existing ml-result.v1 contract
  diagnostics: map<string, list<string>>
}

SafeSessionContext {
  capture_id: string
  flow_id: string
  session_id: string
  source_type: SourceType
  protocol: Protocol
  src_port: integer
  dst_port: integer
  observations: fields from MODEL_INPUT_FEATURES only
}

ErrorResponse {
  schema_version: "analysis-response.v1"
  request_id: string
  status: "rejected" | "failed"
  error: { code: string, message: string }
  diagnostics: map<string, list<string>>
}
```

`result` is the existing `MLResult` serialized with its aliases, including `risk.class`. `session.observations` is a read-only presentation copy of `MODEL_INPUT_FEATURES` so the dashboard can explain what was assessed. The API may include the required provenance identifiers in `SafeSessionContext`, but no extra `SessionFeatures` fields, payload content, or captured message content may be returned.

### 6.2 Response status

| `status` | Meaning | UI behavior |
|---|---|---|
| `complete` | Valid result with expected configured outputs | Render normal result |
| `degraded` | Result exists but a model, explanation, category, or evidence capability is unavailable | Render result plus prominent diagnostic and limitation |
| `rejected` | Request failed auth, privacy, schema, or provenance validation | Show correction/access action; do not render a risk result |
| `failed` | Backend could not produce a result | Show failure ID and bounded retry action; do not fabricate values |

HTTP behavior:

- `200` for `complete` and `degraded` responses with a result.
- `400` or `422` for malformed, forbidden, or invalid input, according to the API framework’s validation convention.
- `401` for missing/invalid authentication and `403` for unauthorized capture scope.
- `409` for a client/runtime contract mismatch that can be corrected by selecting a compatible bundle or record.
- `500`/`503` for server or model-artifact failures, with a sanitized diagnostic and request ID.

The API must not echo rejected sensitive values, raw captured content, secrets, or full untrusted exception traces.

### 6.3 API invariants

- Accept only one validated session-feature record per synchronous request in the MVP.
- Reject PCAP files, email content, credentials, key material, and forbidden nested keys before model code runs.
- Load a pinned, checksum-validated model bundle; do not train or mutate artifacts during a request.
- Use bounded request size and one explicit manual retry from the UI; no infinite client retry loop.
- Preserve `model_bundle_version`, rule evidence references, model diagnostics, and explanation diagnostics.
- Return deterministic JSON suitable for fixture-based tests.
- Use a request ID for logs and UI support without logging the sensitive request body.

## 7. ML and output contract

### 7.1 Pipeline mapping

| Pipeline responsibility | Existing implementation | Dashboard interpretation |
|---|---|---|
| Input validation | `ml.schema.validate_session` | The session is accepted/rejected, not “low risk” when invalid |
| Deterministic findings | `ml.rules.extract_rule_findings` | Authoritative evidence-backed findings |
| Supervised models | XGBoost and Random Forest in `ml.models` | Separate model predictions and class probabilities |
| Calibration | `ml.calibration` | Calibrated probabilities/scores only |
| Fusion | `ml.fusion` | Advisory ensemble signal; do not hide component outputs |
| Policy precedence | `ml.fusion.fuse_session` | Final severity cannot be lower than the highest rule severity |
| Anomaly | Optional Isolation Forest path | `disabled`, `calibrated`, or unavailable must be explicit |
| Explainability | `ml.explain` through `ml.pipeline` | Feature contribution with model disclaimer and evidence refs |
| Final result | `MLResult` | API’s authoritative render model |

### 7.2 Required result display

The dashboard must be able to render these fields from `ml-result.v1`:

- `risk.class`, `risk.score`, `risk.source`, and `risk.minimum_rule_severity`.
- `action` as the next analyst action, not as an automatic remediation command.
- `anomaly.status`, `detected`, `score`, `threshold`, and `baseline_id` when present.
- XGBoost and Random Forest predicted classes, risk probabilities, and class probability maps.
- Optional Isolation Forest output when it participates.
- Every rule finding’s ID, severity, title, and evidence references.
- Evidence references attached to the result and explanations.
- Supervised/anomaly explanations, observed values, contribution direction, feature view, model, and the explanation disclaimer.
- `model_bundle_version` and all diagnostics, including unavailable models or explanations.

No UI copy may call a model probability a certainty, verdict, or proof of attacker intent.

## 8. Frontend dashboard design

### 8.1 Information architecture

The MVP is a single-session analyst workspace rather than a history or administration product.

```text
SecureMailScope
├── Analyze session
│   ├── Session input / authorized session selector
│   └── Submission and validation state
└── Current analysis
    ├── Decision summary
    ├── Findings
    ├── Evidence references
    ├── Model signals
    ├── Explanations
    ├── Observed feature views
    └── Diagnostics and limitations
```

Durable session history, queueing, saved reports, and fleet-wide aggregates are deferred until a storage/read-model contract is approved.

### 8.2 Primary screen layout

**Shell**

- Desktop-first analyst console with a collapsible left navigation rail.
- Header shows product name, current capture/session identifier, source badge, model bundle version, and an explicit privacy boundary label: `Session metadata only`.
- Use a neutral, high-contrast canvas; reserve semantic colors for severity and state.
- Use a sans-serif UI font and a monospace treatment for IDs, ports, versions, hashes, and evidence references.

**Analysis workspace**

1. **Input bar** — authorized session selector or safe session-feature fixture action; primary button is `Analyze session`.
2. **Decision banner** — final risk class, score, action, policy source, and rule severity floor. The action must be visible without scrolling.
3. **Finding list** — compact table with severity, finding ID, title, evidence count, and disclosure control. Empty state explicitly says `No deterministic findings`.
4. **Evidence panel** — grouped evidence references with source labels (`pcap`, `session`, `scenario`) and packet ranges/fields when the API provides them. No raw payload viewer.
5. **Model signals** — side-by-side XGBoost and Random Forest results, class probability bars, calibrated score labels, and optional anomaly state. Model absence is a state, not a zero bar.
6. **Explanation panel** — top feature contributions grouped by feature view (`protocol_session`, `tls`, `certificate`), observed value, direction, model, and evidence references. Keep the disclaimer adjacent to the panel title.
7. **Observed features** — disclosure sections for protocol/session, TLS negotiation, and certificate posture. Show null as `Not observed`, never as an empty string.
8. **Diagnostics** — expandable technical details for analysts: request ID, bundle version, unavailable components, unknown categories, explanation failures, and recovery action.

### 8.3 Visual and interaction contract

- Use severity text plus color, icon, and a consistent left-border/accent; never use color alone.
- Severity order is informational → low → medium → high → critical, matching `RISK_SEVERITY_ORDER`.
- Use stable labels: `Final risk`, `Policy floor`, `Model advisory`, `Anomaly signal`, `Evidence`, `Explanation unavailable`, and `Not observed`.
- The primary action is always clear: analyze, inspect evidence, or review a limitation. Do not present `Fix`, `Block`, or `Remediate` controls.
- Findings and model panels are keyboard-operable disclosure sections with visible focus.
- Do not auto-refresh or poll in the synchronous MVP. A failed request offers one manual retry with the same request ID context.
- Preserve expanded/collapsed state while the analyst moves between sections.
- On narrow screens, stack the decision banner, findings, signals, evidence, explanations, and features in that order; keep the action and final risk at the top.

### 8.4 Frontend data ownership

| UI concern | Source of truth |
|---|---|
| Risk class and score | `result.risk` |
| Analyst action | `result.action` |
| Policy precedence | `result.risk.source` and `minimum_rule_severity` |
| Finding title/severity | `result.rule_findings` |
| Evidence | Result/finding/explanation evidence refs and safe API session context |
| Model participation | `result.model_outputs` and `result.diagnostics` |
| Explanation availability | `result.explanations` and explanation diagnostics |
| Loading/error state | API response status and transport state |
| Labels, colors, and ordering | Frontend presentation constants aligned to `RiskLabel`; never inferred from score in the browser |

## 9. Required UI states

| State | Visible behavior | Prohibited behavior |
|---|---|---|
| Empty | Explain accepted input, privacy boundary, and `Analyze session` action | Do not show placeholder risk values |
| Editing/validation | Show field-level safe validation and missing required fields | Do not claim the record is safe |
| Loading | Disable duplicate submit, show progress label, retain request context | Do not animate fake model progress or auto-retry indefinitely |
| Complete with findings | Show final severity, action, findings, evidence, and model signals | Do not hide rule evidence behind the score |
| Complete with no findings | Show `No deterministic findings` and advisory model result | Do not imply absence of findings proves safety |
| Degraded result | Show result plus warning explaining the missing/disabled component | Do not turn disabled anomaly/explanation into a negative signal |
| Rejected | Show sanitized reason and correction/access action | Do not echo forbidden values or raw payload |
| Backend unavailable | Show request ID, bounded retry, and contact/operator path | Do not show stale result as current |
| Explanation unavailable | Keep risk/result visible and mark explanation section unavailable | Do not invent natural-language rationale |
| Unknown category | Keep result visible and list the diagnostic | Do not silently drop or relabel the category |

## 10. Security, privacy, and accessibility requirements

### Security and privacy

- Require authenticated access and an authorization check for the selected capture/session scope.
- Allow only validated session metadata through the browser/API path.
- Reuse the existing forbidden-payload policy and strict schema validation.
- Do not log request bodies, email content, credentials, tokens, key logs, private keys, or raw packet payloads.
- Escape all evidence, scenario, and feature strings before rendering; treat captured/scenario text as untrusted data.
- Use same-origin or explicitly allowlisted API origins and standard CSRF protection where cookie authentication is used.
- Do not expose model bundle files, calibration files, internal filesystem paths, or full stack traces to the browser.
- Preserve provenance and evidence references; do not allow client input to lower a deterministic finding severity.

### Accessibility

- Meet WCAG 2.2 AA for contrast, keyboard access, focus indication, semantics, and status announcements.
- Pair every severity color with text and a non-color indicator.
- Use real headings, table headers, buttons, and disclosure semantics rather than clickable generic containers.
- Announce submission, completion, rejection, and degraded status to assistive technology.
- Keep evidence/reference text selectable and readable at increased text size.

## 11. Verification plan

### Contract/API checks

- Valid `session-features.v1` fixture returns `analysis-response.v1` with `status=complete` and a valid `ml-result.v1` result.
- A critical deterministic rule remains critical even when model output is lower.
- A model-only high-risk result remains advisory.
- A disabled or missing optional anomaly model is shown in `anomaly.status`/diagnostics and does not become a false negative.
- Explanation failure returns the result with an explanation diagnostic and no fabricated explanation.
- Unknown categorical values continue according to the preprocessing contract and remain visible in diagnostics.
- Sensitive nested keys, PCAP input, and malformed provenance are rejected without echoing their values.
- Model bundle/version mismatch is explicit and does not run inference.
- Repeated requests with the same fixture and pinned bundle produce deterministic output.

### Frontend checks

- Empty → submit → loading → complete flow renders the final action and risk without scrolling.
- Findings show severity, IDs, titles, and evidence references.
- Model disagreement is visible without opening developer tools.
- Degraded, rejected, unavailable, and explanation-unavailable states are visually distinct and actionable.
- Null/unknown values render as `Not observed`/`Unknown`, not blank or safe.
- Keyboard-only navigation reaches the submit button, disclosures, evidence references, and retry action.
- Representative desktop and narrow viewport layouts do not clip the decision banner, tables, or diagnostics.
- Browser console has no errors during the representative flow.

### Flowchart check

The implementation review must compare the running call path against the Mermaid chart and update either the chart or implementation plan when a boundary changes. The chart is not a substitute for API tests.

## 12. Delivery boundaries and next step

### Planned implementation slices

1. Define and test the API request/response envelopes around the existing Python result contracts.
2. Implement the authenticated synchronous analysis adapter with privacy/schema gates and pinned runtime loading.
3. Build the single-session dashboard against deterministic fixtures.
4. Add browser verification for success, rejected, and degraded paths.
5. Add durable history, session indexing, and batch analysis only after a separate storage contract is approved.

### Explicit current gaps

- No HTTP API exists in this repository.
- No frontend package or frontend test runner exists in this repository.
- The active ML result contains compact evidence-reference strings; a packet-range viewer requires a separate evidence resolver and is not part of this MVP.
- The current active bundle disables Isolation Forest; the UI must support its explicit disabled state without requiring it for the MVP.

This spec does not change the existing ML contracts. Any implementation plan must preserve the current boundary: validated session-feature JSON enters the scorer; `MLResult` leaves it; rules remain authoritative; sensitive content never enters the system.

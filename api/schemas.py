"""Request and response envelopes for the SecureMail-ML HTTP API.

Follows the contracts defined in the frontend-API-ML dashboard design spec:
docs/superpowers/specs/2026-09-05-frontend-api-ml-dashboard-design.md
"""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from ml.schema import MODEL_INPUT_FEATURES, Protocol, SourceType


class AnalysisRequest(BaseModel):
    """Transport envelope for a single session analysis."""

    schema_version: Literal["analysis-request.v1"] = "analysis-request.v1"
    record: dict[str, Any] = Field(
        ...,
        description="A session-features.v1 record as a raw JSON object.",
    )


class SafeSessionContext(BaseModel):
    """Read-only presentation copy of provenance and observed features.

    Contains only identifiers and MODEL_INPUT_FEATURES so the dashboard
    can display what was assessed without leaking sensitive or extra fields.
    """

    capture_id: str
    flow_id: str
    session_id: str
    source_type: SourceType
    protocol: Protocol
    src_port: int
    dst_port: int
    observations: dict[str, Any] = Field(
        default_factory=dict,
        description="Model input features from the assessed session.",
    )


class AuthLoginRequest(BaseModel):
    email: str = Field(..., min_length=3, max_length=320)
    password: str = Field(..., min_length=1, max_length=256)


class AuthRegisterRequest(BaseModel):
    email: str = Field(..., min_length=3, max_length=320)
    password: str = Field(..., min_length=1, max_length=256)
    display_name: Optional[str] = Field(None, max_length=120)


class ProfileUpdateRequest(BaseModel):
    display_name: str = Field(..., min_length=1, max_length=120)


class ProfileResponse(BaseModel):
    id: int
    email: str
    display_name: str


class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    profile: ProfileResponse


class LogoutResponse(BaseModel):
    logged_out: bool


class ErrorDetail(BaseModel):
    """Structured error information."""

    code: str
    message: str


AgentSection = Literal["overview", "risk", "tls", "certificate", "findings"]


class AnalysisResponse(BaseModel):
    """Successful analysis response envelope."""

    schema_version: str = "analysis-response.v1"
    request_id: str
    status: Literal["complete", "degraded"]
    session: SafeSessionContext
    result: dict[str, Any]
    tls_details: dict[str, Any] | None = None
    certificate_details: dict[str, Any] | None = None
    diagnostics: dict[str, list[str]] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    """Error response envelope for rejected or failed requests."""

    schema_version: str = "analysis-response.v1"
    request_id: str
    status: Literal["rejected", "failed"]
    error: ErrorDetail
    diagnostics: dict[str, list[str]] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    """Health and readiness check response."""

    status: str
    bundle_loaded: bool
    bundle_version: str | None = None
    calibration_loaded: bool
    model_names: list[str] = Field(default_factory=list)


class BundleInfoResponse(BaseModel):
    """Active model bundle metadata."""

    bundle_version: str
    model_names: list[str]
    feature_count: int
    feature_names: list[str]
    calibration_version: str


class RuleInfoItem(BaseModel):
    """Description of a known deterministic rule."""

    finding_id: str
    severity: str
    title: str
    condition: str


class RulesResponse(BaseModel):
    """List of all known deterministic rule IDs."""

    rules: list[RuleInfoItem]


class ValidationRequest(BaseModel):
    """Validation request envelope."""

    schema_version: Literal["validation-request.v1"] = "validation-request.v1"
    record: dict[str, Any] = Field(
        ...,
        description="A session-features.v1 record to validate.",
    )


class ValidationResponse(BaseModel):
    """Validation response envelope."""

    schema_version: Literal["validation-response.v1"] = "validation-response.v1"
    request_id: str
    valid: bool
    status: Literal["valid", "invalid", "rejected"]
    errors: list[str] = Field(default_factory=list)
    diagnostics: dict[str, list[str]] = Field(default_factory=dict)



def cryptographic_posture(observations: dict[str, Any]) -> str:
    """Bucket TLS/certificate observations for dashboard aggregates."""
    handshake = observations.get("handshake_success")
    tls = observations.get("tls_version")
    cipher_family = observations.get("cipher_family")
    forward_secrecy = observations.get("forward_secrecy")
    handshake_failures = observations.get("handshake_failures")
    expired = observations.get("cert_expired")
    mismatch = observations.get("hostname_mismatch")
    chain_valid = observations.get("cert_chain_valid")
    key_len = observations.get("cert_key_length_bits")
    if handshake is False and isinstance(handshake_failures, (int, float)) and handshake_failures > 0:
        return "handshake_failed"
    if tls in {"TLS1.0", "TLS1.1"} or cipher_family in {"3DES", "RC4"}:
        return "deprecated"
    weak_key = isinstance(key_len, (int, float)) and key_len < 2048
    if expired or mismatch or chain_valid is False or forward_secrecy is False or weak_key:
        return "weak"
    if tls == "TLS1.3":
        return "modern"
    if tls == "TLS1.2":
        return "adequate"
    return "unknown"


def build_safe_session_context(record: Any) -> SafeSessionContext:
    """Extract a SafeSessionContext from a validated SessionFeatureRecord."""
    features = record.features
    observations: dict[str, Any] = {}
    for feature_name in MODEL_INPUT_FEATURES:
        value = getattr(features, feature_name, None)
        if value is not None:
            # Convert enums to their string value for JSON serialization
            if hasattr(value, "value"):
                value = value.value
            observations[feature_name] = value

    return SafeSessionContext(
        capture_id=record.provenance.capture_id,
        flow_id=record.provenance.flow_id,
        session_id=record.provenance.session_id,
        source_type=record.provenance.source_type,
        protocol=features.protocol,
        src_port=features.src_port,
        dst_port=features.dst_port,
        observations=observations,
    )


class CaptureSessionPreview(BaseModel):
    session_id: str
    protocol: Protocol
    src_port: int
    dst_port: int
    tls_version: str | None
    cipher_suite: str | None
    starttls_advertised: bool
    starttls_used: bool
    handshake_success: bool
    cert_present: bool | None
    cert_expired: bool | None
    hostname_mismatch: bool | None
    tls_details: dict[str, Any] | None = None
    certificate_details: dict[str, Any] | None = None
    evidence_refs: list[dict[str, Any]]
    checked_views: list[Literal["protocol_session", "tls", "certificate"]]


CaptureJobStatus = Literal["queued", "running", "complete", "empty", "failed"]


class CaptureResponse(BaseModel):
    """Durable PCAP extraction job status."""

    schema_version: Literal["capture-job.v1"] = "capture-job.v1"
    job_id: str
    capture_id: str
    status: CaptureJobStatus
    pcap_sha256: str
    filename: str
    attempts: int = 0
    session_count: int = 0
    processed_sessions: int = 0
    progress: float | None = None
    error_code: str | None = None
    diagnostics: dict[str, list[str]] = Field(default_factory=dict)


class CaptureJobListResponse(BaseModel):
    jobs: list[CaptureResponse]
    total: int
    skip: int
    limit: int


class CaptureSessionListResponse(BaseModel):
    capture_id: str
    skip: int = 0
    limit: int = 200
    total: int
    sessions: list[CaptureSessionPreview]
    records: list[dict[str, Any]]


# ---------------------------------------------------------------------------
#  Database CRUD & Synthetic Data Schemas
# ---------------------------------------------------------------------------


class AnalysisRecordResponse(BaseModel):
    """Serialised view of a single analysis DB row."""

    id: int
    request_id: str
    session_id: str
    client_id: str | None
    capture_id: str | None = None
    protocol: str | None = None
    posture: str | None = None
    timestamp: str
    record_count: int
    evidence_ref_count: int = 0
    risk_score: float
    final_verdict: str
    rule_score: float | None = None
    rule_triggers_count: int
    trigger_details: list[Any]
    ml_scores: dict[str, Any]
    explanations: dict[str, Any]
    model_bundle: dict[str, Any]
    tls_details: dict[str, Any] | None = None
    certificate_details: dict[str, Any] | None = None
    is_synthetic: bool
    source_label: str | None


class AgentInsightRequest(BaseModel):
    """Request for a read-only explanation of a live or persisted analysis."""

    schema_version: Literal["agent-insight-request.v1"] = "agent-insight-request.v1"
    analysis: AnalysisResponse | AnalysisRecordResponse
    section: AgentSection
    question: str = Field(..., min_length=1, max_length=4000)


class AgentAdvisory(BaseModel):
    """Provider-produced advisory payload before the API envelope is added."""

    answer: str = Field(..., min_length=1, max_length=12000)
    recommendations: list[str] = Field(default_factory=list, max_length=10)
    evidence: list[str] = Field(default_factory=list, max_length=20)


class AgentInsightResponse(BaseModel):
    """Advisory AI response; it never replaces the authoritative analysis."""

    schema_version: Literal["agent-insight-response.v1"] = "agent-insight-response.v1"
    request_id: str
    status: Literal["complete", "degraded"]
    section: AgentSection
    answer: str | None = None
    recommendations: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    provider: str | None = None
    model: str | None = None
    diagnostics: dict[str, list[str]] = Field(default_factory=dict)


class AnalysisListResponse(BaseModel):
    """Paginated list of analysis records."""

    total: int
    skip: int
    limit: int
    records: list[AnalysisRecordResponse]


class CaptureAnalysisListResponse(BaseModel):
    job_id: str
    total: int
    skip: int
    limit: int
    records: list[AnalysisRecordResponse]


class SyntheticAnalysisRequest(BaseModel):
    """Body for inserting a synthetic analysis row without running ML inference.

    Use this to load test fixtures, boundary cases, or labelled examples
    directly into the database. All ML fields must be supplied manually.
    """

    session_id: str = Field(..., description="Arbitrary session identifier for this synthetic record.")
    client_id: str | None = Field(default=None, description="Optional client / capture identifier.")
    source_label: str | None = Field(
        default="synthetic",
        description="Human-readable label for the data source (e.g. 'synthetic', 'test-fixture-v1').",
    )
    risk_score: float = Field(..., ge=0.0, le=1.0, description="Synthetic risk score in [0, 1].")
    final_verdict: str = Field(
        ...,
        description="Verdict label. One of: benign, suspicious, malicious, informational.",
    )
    rule_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional rule score. Null when no separate rule-score contract exists.",
    )
    rule_triggers_count: int = Field(default=0, ge=0)
    trigger_details: list[Any] = Field(default_factory=list, description="Optional rule trigger detail objects.")
    ml_scores: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional per-model raw scores (xgboost, random_forest, etc.).",
    )
    explanations: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional SHAP / feature importance explanations.",
    )
    model_bundle: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional model bundle metadata snapshot.",
    )


class SyntheticAnalysisResponse(BaseModel):
    """Confirmation that a synthetic analysis record was inserted."""

    request_id: str
    status: str = "created"
    message: str
    record_id: int


class ValidationRecordResponse(BaseModel):
    """Serialised view of a single validation DB row."""

    id: int
    request_id: str
    session_id: str
    timestamp: str
    valid: bool
    record_count: int
    issues_count: int
    issues: dict[str, Any]


class ValidationListResponse(BaseModel):
    """Paginated list of validation records."""

    total: int
    skip: int
    limit: int
    records: list[ValidationRecordResponse]


class VerdictCount(BaseModel):
    """A single verdict label and its count."""

    verdict: str
    count: int


class PostureCount(BaseModel):
    """A cryptographic posture bucket and its count."""

    posture: str
    count: int


class StatsResponse(BaseModel):
    """Aggregate statistics across stored analysis records."""

    total_analyses: int
    total_synthetic: int
    total_real: int
    flagged_sessions: int
    evidence_archived: int
    avg_risk_score: float | None
    verdict_distribution: list[VerdictCount]
    cryptographic_posture_distribution: list[PostureCount]
    total_validations: int
    validation_pass_rate: float | None


class DeleteResponse(BaseModel):
    """Confirmation of a delete operation."""

    deleted: int
    message: str


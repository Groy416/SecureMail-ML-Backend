"""Request and response envelopes for the SecureMail-ML HTTP API.

Follows the contracts defined in the frontend-API-ML dashboard design spec:
docs/superpowers/specs/2026-09-05-frontend-api-ml-dashboard-design.md
"""
from __future__ import annotations

from typing import Any, Literal

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


class ErrorDetail(BaseModel):
    """Structured error information."""

    code: str
    message: str


class AnalysisResponse(BaseModel):
    """Successful analysis response envelope."""

    schema_version: str = "analysis-response.v1"
    request_id: str
    status: Literal["complete", "degraded"]
    session: SafeSessionContext
    result: dict[str, Any]
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


# ---------------------------------------------------------------------------
#  Database CRUD & Synthetic Data Schemas
# ---------------------------------------------------------------------------


class AnalysisRecordResponse(BaseModel):
    """Serialised view of a single analysis DB row."""

    id: int
    request_id: str
    session_id: str
    client_id: str | None
    timestamp: str
    record_count: int
    risk_score: float
    final_verdict: str
    rule_score: float
    rule_triggers_count: int
    trigger_details: list[Any]
    ml_scores: dict[str, Any]
    explanations: dict[str, Any]
    model_bundle: dict[str, Any]
    is_synthetic: bool
    source_label: str | None


class AnalysisListResponse(BaseModel):
    """Paginated list of analysis records."""

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
    rule_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Synthetic rule/deterministic score.")
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


class StatsResponse(BaseModel):
    """Aggregate statistics across all analysis records in the DB."""

    total_analyses: int
    total_synthetic: int
    total_real: int
    avg_risk_score: float | None
    verdict_distribution: list[VerdictCount]
    total_validations: int
    validation_pass_rate: float | None  # 0.0–1.0


class DeleteResponse(BaseModel):
    """Confirmation of a delete operation."""

    deleted: int
    message: str


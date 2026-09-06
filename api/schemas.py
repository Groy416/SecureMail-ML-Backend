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

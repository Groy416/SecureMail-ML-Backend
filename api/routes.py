"""Route handlers for the SecureMail-ML HTTP API.

Implements the API boundary specification from:
docs/superpowers/specs/2026-09-05-frontend-api-ml-dashboard-design.md
"""
from __future__ import annotations

import logging
import os
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ml.pipeline import predict_session
from ml.product import PayloadRejected, inference_record

from api.database import AnalysisRecord, CaptureSession, ValidationRecord
from api.dependencies import (
    authenticate_request,
    generate_request_id,
    get_runtime,
    get_db,
    require_api_key,
)
from api.schemas import (
    AnalysisRequest,
    AnalysisResponse,
    BundleInfoResponse,
    ErrorDetail,
    ErrorResponse,
    HealthResponse,
    RuleInfoItem,
    RulesResponse,
    ValidationRequest,
    ValidationResponse,
    build_safe_session_context,
    cryptographic_posture,
)

logger = logging.getLogger("securemailscope.api")

router = APIRouter(prefix="/api/v1", tags=["v1"])


# --------------------------------------------------------------------------- #
#  Health / metadata endpoints                                                #
# --------------------------------------------------------------------------- #


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health and readiness check",
)
def health() -> HealthResponse:
    """Return the API health status and whether the ML runtime is loaded."""
    try:
        runtime = get_runtime()
        return HealthResponse(
            status="healthy",
            bundle_loaded=True,
            bundle_version=runtime.bundle.version,
            calibration_loaded=True,
            model_names=[
                "xgboost",
                "random_forest",
                *(["isolation_forest"] if runtime.bundle.isolation_forest is not None else []),
            ],
        )
    except RuntimeError:
        return HealthResponse(
            status="unavailable",
            bundle_loaded=False,
            calibration_loaded=False,
        )


@router.get(
    "/bundle",
    response_model=BundleInfoResponse,
    dependencies=[Depends(require_api_key)],
    summary="Active model bundle metadata",
)
def bundle_info() -> BundleInfoResponse:
    """Return metadata about the currently loaded model bundle."""
    runtime = get_runtime()
    bundle = runtime.bundle
    model_names = ["xgboost", "random_forest"]
    if bundle.isolation_forest is not None:
        model_names.append("isolation_forest")

    return BundleInfoResponse(
        bundle_version=bundle.version,
        model_names=model_names,
        feature_count=len(bundle.preprocessor.feature_names),
        feature_names=list(bundle.preprocessor.feature_names),
        calibration_version=runtime.calibration.version,
    )


# Known deterministic rules with descriptions for documentation.
KNOWN_RULES: list[RuleInfoItem] = [
    RuleInfoItem(
        finding_id="TLS-001",
        severity="critical",
        title="Deprecated TLS version",
        condition="TLS version is 1.0 or 1.1",
    ),
    RuleInfoItem(
        finding_id="TLS-002",
        severity="critical",
        title="Weak cipher suite",
        condition="Cipher family is 3DES or RC4",
    ),
    RuleInfoItem(
        finding_id="FS-001",
        severity="high",
        title="Forward secrecy unavailable",
        condition="Successful handshake without forward secrecy",
    ),
    RuleInfoItem(
        finding_id="CERT-001",
        severity="high",
        title="Expired certificate",
        condition="Certificate is expired",
    ),
    RuleInfoItem(
        finding_id="CERT-002",
        severity="high",
        title="Weak RSA certificate key",
        condition="RSA certificate key below 2048 bits",
    ),
    RuleInfoItem(
        finding_id="CERT-003",
        severity="high",
        title="Invalid certificate chain",
        condition="Certificate chain validation failed",
    ),
    RuleInfoItem(
        finding_id="CERT-004",
        severity="high",
        title="Certificate hostname mismatch",
        condition="Certificate hostname does not match",
    ),
    RuleInfoItem(
        finding_id="STLS-001",
        severity="high",
        title="STARTTLS advertised but unused",
        condition="STARTTLS is advertised but not used",
    ),
    RuleInfoItem(
        finding_id="STLS-002",
        severity="high",
        title="STARTTLS handshake failure",
        condition="STARTTLS used but handshake failed",
    ),
    RuleInfoItem(
        finding_id="ANOM-001",
        severity="high",
        title="Repeated TLS handshake failures",
        condition="Three or more handshake failures",
    ),
    RuleInfoItem(
        finding_id="ANOM-002",
        severity="medium",
        title="Unusual cipher negotiation",
        condition="RC4 or ChaCha20-Poly1305 cipher negotiation",
    ),
    RuleInfoItem(
        finding_id="ANOM-003",
        severity="medium",
        title="Unexpected TLS version",
        condition="TLS version is 1.1",
    ),
    RuleInfoItem(
        finding_id="ANOM-004",
        severity="high",
        title="Multiple TLS renegotiations",
        condition="Two or more TLS renegotiations",
    ),
]


@router.get(
    "/rules",
    response_model=RulesResponse,
    dependencies=[Depends(require_api_key)],
    summary="List known deterministic rules",
)
def list_rules() -> RulesResponse:
    """Return all known deterministic rule IDs with descriptions."""
    return RulesResponse(rules=KNOWN_RULES)


# --------------------------------------------------------------------------- #
#  Analysis endpoint                                                          #
# --------------------------------------------------------------------------- #


@router.post(
    "/analyses",
    response_model=AnalysisResponse,
    responses={
        422: {"model": ErrorResponse, "description": "Rejected request"},
        500: {"model": ErrorResponse, "description": "Server error"},
        503: {"model": ErrorResponse, "description": "ML runtime unavailable"},
    },
    summary="Analyze a single session",
    description=(
        "Submit one session-features.v1 record for synchronous ML analysis. "
        "Returns risk classification, deterministic findings, model signals, "
        "feature explanations, and diagnostics."
    ),
)
async def analyze_session(
    body: AnalysisRequest,
    request: Request,
    authorization: str | None = Header(default=None),
    db: AsyncSession = Depends(get_db),
) -> AnalysisResponse | ErrorResponse:
    """Synchronous single-session analysis endpoint.

    Flow:
    1. Authenticate request
    2. Privacy gate (reject sensitive payload fields)
    3. Schema validation (build SessionFeatureRecord)
    4. Load pinned model bundle and calibration
    5. Run ml.pipeline.predict_session(...)
    6. Persist analysis results to DB
    7. Build response envelope
    """
    request_id = generate_request_id()

    # --- Step 1: Authentication ---
    if not authenticate_request(authorization):
        raise HTTPException(
            status_code=401,
            detail=ErrorResponse(
                request_id=request_id,
                status="rejected",
                error=ErrorDetail(
                    code="authentication_required",
                    message="Valid authorization credentials are required.",
                ),
            ).model_dump(mode="json"),
        )

    # --- Step 2 & 3: Privacy gate + Schema validation ---
    try:
        record = inference_record(body.record)
    except PayloadRejected as exc:
        raise HTTPException(
            status_code=422,
            detail=ErrorResponse(
                request_id=request_id,
                status="rejected",
                error=ErrorDetail(
                    code="payload_rejected",
                    message=str(exc),
                ),
            ).model_dump(mode="json"),
        )
    except ValidationError as exc:
        # Extract clean validation error messages without full traces
        error_messages = []
        for error in exc.errors():
            location = " -> ".join(str(loc) for loc in error["loc"])
            error_messages.append(f"{location}: {error['msg']}")

        raise HTTPException(
            status_code=422,
            detail=ErrorResponse(
                request_id=request_id,
                status="rejected",
                error=ErrorDetail(
                    code="validation_error",
                    message="; ".join(error_messages),
                ),
                diagnostics={"validation": error_messages},
            ).model_dump(mode="json"),
        )
    except (ValueError, TypeError) as exc:
        raise HTTPException(
            status_code=422,
            detail=ErrorResponse(
                request_id=request_id,
                status="rejected",
                error=ErrorDetail(
                    code="validation_error",
                    message=str(exc),
                ),
            ).model_dump(mode="json"),
        )

    # --- Step 4: Load runtime ---
    try:
        runtime = get_runtime()
    except RuntimeError:
        raise HTTPException(
            status_code=503,
            detail=ErrorResponse(
                request_id=request_id,
                status="failed",
                error=ErrorDetail(
                    code="runtime_unavailable",
                    message=(
                        "The ML runtime is not available. "
                        "The model bundle may be missing or failed to load."
                    ),
                ),
            ).model_dump(mode="json"),
        )

    # --- Step 5: Inference ---
    try:
        result = predict_session(
            runtime.bundle,
            runtime.calibration,
            record,
        )
    except Exception as exc:
        logger.exception(
            "Inference failed for request %s: %s",
            request_id,
            type(exc).__name__,
        )
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                request_id=request_id,
                status="failed",
                error=ErrorDetail(
                    code="inference_error",
                    message=(
                        "An error occurred during ML inference. "
                        "Please contact support with the request ID."
                    ),
                ),
                diagnostics={
                    "error_type": [type(exc).__name__],
                },
            ).model_dump(mode="json"),
        )

    # --- Step 6: Build response & Persist to DB ---
    result_dict = result.model_dump(mode="json", by_alias=True)
    session_context = build_safe_session_context(record)

    # Determine status: degraded if any diagnostics contain unavailable signals
    all_diagnostics = result.diagnostics.copy()
    status: str = "complete"
    for key, values in all_diagnostics.items():
        if any("unavailable" in v for v in values):
            status = "degraded"
            break

    protocol = (
        session_context.protocol.value
        if hasattr(session_context.protocol, "value")
        else str(session_context.protocol)
    )
    findings = list(result.rule_findings)
    try:
        db_record = AnalysisRecord(
            request_id=request_id,
            session_id=session_context.session_id,
            client_id=session_context.capture_id,
            capture_id=session_context.capture_id,
            protocol=protocol,
            posture=cryptographic_posture(session_context.observations),
            evidence_ref_count=len(result.evidence_refs),
            record_count=1,
            risk_score=result.risk.score,
            final_verdict=result.risk.risk_class.value,
            rule_score=None,
            rule_triggers_count=len(findings),
            trigger_details=jsonable_encoder(findings),
            ml_scores=jsonable_encoder(result.model_outputs),
            explanations=jsonable_encoder(result.explanations),
            model_bundle={"version": result.model_bundle_version},
        )
        db.add(db_record)
        await db.flush()
        capture_session = await db.scalar(
            select(CaptureSession).where(
                CaptureSession.capture_id == session_context.capture_id,
                CaptureSession.session_id == session_context.session_id,
            )
        )
        if capture_session is not None:
            capture_session.analysis_request_id = request_id
        await db.commit()
    except Exception:
        logger.exception("Failed to persist AnalysisRecord for request %s", request_id)
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                request_id=request_id,
                status="failed",
                error=ErrorDetail(
                    code="persistence_error",
                    message=(
                        "Analysis succeeded but could not be stored. "
                        "Retry with the same payload or contact support with the request ID."
                    ),
                ),
            ).model_dump(mode="json"),
        )

    return AnalysisResponse(
        request_id=request_id,
        status=status,
        session=session_context,
        result=result_dict,
        diagnostics=all_diagnostics,
    )


@router.post(
    "/validate",
    response_model=ValidationResponse,
    dependencies=[Depends(require_api_key)],
    summary="Validate a session record payload",
    description=(
        "Pre-flight payload and schema validation without running ML inference. "
        "Checks for privacy gate violations, forbidden fields, and schema syntax."
    ),
)
async def validate_session_payload(
    body: ValidationRequest,
    db: AsyncSession = Depends(get_db),
) -> ValidationResponse:
    """Pre-flight validation endpoint for frontends."""
    request_id = generate_request_id()

    try:
        record = inference_record(body.record)
        res = ValidationResponse(
            request_id=request_id,
            valid=True,
            status="valid",
            errors=[],
            diagnostics={},
        )
        session_context = build_safe_session_context(record)
        session_id_val = session_context.session_id
        issues_dict = {}
        valid_val = True
    except PayloadRejected as exc:
        res = ValidationResponse(
            request_id=request_id,
            valid=False,
            status="rejected",
            errors=[str(exc)],
            diagnostics={"privacy_gate": [str(exc)]},
        )
        session_id_val = body.record.get("metadata", {}).get("session_id", "unknown") if isinstance(body.record, dict) else "unknown"
        issues_dict = {"privacy_gate": [str(exc)]}
        valid_val = False
    except ValidationError as exc:
        errors = [f"{' -> '.join(str(loc) for loc in e['loc'])}: {e['msg']}" for e in exc.errors()]
        res = ValidationResponse(
            request_id=request_id,
            valid=False,
            status="invalid",
            errors=errors,
            diagnostics={"schema_validation": errors},
        )
        session_id_val = body.record.get("metadata", {}).get("session_id", "unknown") if isinstance(body.record, dict) else "unknown"
        issues_dict = {"schema_validation": errors}
        valid_val = False
    except (ValueError, TypeError) as exc:
        res = ValidationResponse(
            request_id=request_id,
            valid=False,
            status="invalid",
            errors=[str(exc)],
            diagnostics={"schema_validation": [str(exc)]},
        )
        session_id_val = "unknown"
        issues_dict = {"schema_validation": [str(exc)]}
        valid_val = False

    try:
        db_record = ValidationRecord(
            request_id=request_id,
            session_id=session_id_val,
            valid=valid_val,
            record_count=1,
            issues_count=len(res.errors),
            issues=issues_dict,
        )
        db.add(db_record)
        await db.flush()
        await db.commit()
    except Exception:
        logger.exception("Failed to persist ValidationRecord for request %s", request_id)
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                request_id=request_id,
                status="failed",
                error=ErrorDetail(
                    code="persistence_error",
                    message="Validation completed but could not be stored.",
                ),
            ).model_dump(mode="json"),
        )

    return res

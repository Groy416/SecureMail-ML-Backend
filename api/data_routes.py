"""Database CRUD and synthetic data routes for the SecureMail-ML API.

Provides read access to persisted analysis/validation records, aggregate
statistics, synthetic data insertion for testing, and bulk-delete for
resetting the database between test runs.

Endpoints
---------
GET    /api/v1/analyses                  List analysis records (paginated + filtered)
GET    /api/v1/analyses/stats            Aggregate statistics
GET    /api/v1/analyses/{request_id}     Get a single analysis record
DELETE /api/v1/analyses/{request_id}     Delete a single analysis record
POST   /api/v1/analyses/synthetic        Insert a synthetic analysis row
GET    /api/v1/validations               List validation records (paginated)
DELETE /api/v1/analyses                  Bulk-delete all analysis records (confirm=true required)
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import delete, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import Select

from api.database import AnalysisRecord, ValidationRecord
from api.dependencies import generate_request_id, get_db, require_api_key
from api.schemas import (
    AnalysisListResponse,
    AnalysisRecordResponse,
    DeleteResponse,
    PostureCount,
    StatsResponse,
    SyntheticAnalysisRequest,
    SyntheticAnalysisResponse,
    ValidationListResponse,
    ValidationRecordResponse,
    VerdictCount,
)

FLAGGED_VERDICTS = frozenset({"high", "critical"})

logger = logging.getLogger("securemailscope.api.data")

data_router = APIRouter(
    prefix="/api/v1",
    tags=["data"],
    dependencies=[Depends(require_api_key)],
)


# ---------------------------------------------------------------------------
#  Helper: ORM row → response schema
# ---------------------------------------------------------------------------

def _row_to_analysis(row: AnalysisRecord) -> AnalysisRecordResponse:
    return AnalysisRecordResponse(
        id=row.id,
        request_id=row.request_id,
        session_id=row.session_id,
        client_id=row.client_id,
        capture_id=row.capture_id,
        protocol=row.protocol,
        posture=row.posture,
        timestamp=row.timestamp.isoformat(),
        record_count=row.record_count,
        evidence_ref_count=row.evidence_ref_count,
        risk_score=row.risk_score,
        final_verdict=row.final_verdict,
        rule_score=row.rule_score,
        rule_triggers_count=row.rule_triggers_count,
        trigger_details=row.trigger_details if row.trigger_details is not None else [],
        ml_scores=row.ml_scores if row.ml_scores is not None else {},
        explanations=row.explanations if row.explanations is not None else {},
        model_bundle=row.model_bundle if row.model_bundle is not None else {},
        tls_details=row.tls_details,
        certificate_details=row.certificate_details,
        is_synthetic=row.is_synthetic,
        source_label=row.source_label,
    )


def _apply_analysis_filters(
    stmt: Select[Any],
    *,
    verdict: Optional[str],
    session_id: Optional[str],
    is_synthetic: Optional[bool],
    capture_id: Optional[str],
    protocol: Optional[str],
    posture: Optional[str],
    from_ts: Optional[datetime],
    to_ts: Optional[datetime],
) -> Select[Any]:
    if verdict:
        stmt = stmt.where(AnalysisRecord.final_verdict == verdict)
    if session_id:
        stmt = stmt.where(AnalysisRecord.session_id == session_id)
    if is_synthetic is not None:
        stmt = stmt.where(AnalysisRecord.is_synthetic == is_synthetic)
    if capture_id:
        stmt = stmt.where(AnalysisRecord.capture_id == capture_id)
    if protocol:
        stmt = stmt.where(AnalysisRecord.protocol == protocol)
    if posture:
        stmt = stmt.where(AnalysisRecord.posture == posture)
    if from_ts is not None:
        stmt = stmt.where(AnalysisRecord.timestamp >= from_ts)
    if to_ts is not None:
        stmt = stmt.where(AnalysisRecord.timestamp <= to_ts)
    return stmt


def _row_to_validation(row: ValidationRecord) -> ValidationRecordResponse:
    return ValidationRecordResponse(
        id=row.id,
        request_id=row.request_id,
        session_id=row.session_id,
        timestamp=row.timestamp.isoformat(),
        valid=row.valid,
        record_count=row.record_count,
        issues_count=row.issues_count,
        issues=row.issues if row.issues is not None else {},
    )


# ---------------------------------------------------------------------------
#  GET /analyses  —  paginated list with optional filters
# ---------------------------------------------------------------------------

@data_router.get(
    "/analyses",
    response_model=AnalysisListResponse,
    summary="List analysis records",
    description=(
        "Return a paginated list of persisted analysis records. "
        "Optionally filter by verdict, session, capture, protocol, posture, "
        "synthetic flag, and an inclusive `from`/`to` timestamp range."
    ),
)
async def list_analyses(
    skip: int = Query(default=0, ge=0, description="Number of records to skip (offset)."),
    limit: int = Query(default=50, ge=1, le=200, description="Maximum records to return (max 200)."),
    verdict: Optional[str] = Query(default=None, description="Filter by final_verdict (e.g. 'critical')."),
    session_id: Optional[str] = Query(default=None, description="Filter by exact session_id."),
    is_synthetic: Optional[bool] = Query(default=None, description="Filter by is_synthetic flag."),
    capture_id: Optional[str] = Query(default=None, description="Filter by capture_id."),
    protocol: Optional[str] = Query(default=None, description="Filter by protocol (SMTP, IMAP, POP3)."),
    posture: Optional[str] = Query(default=None, description="Filter by cryptographic posture bucket."),
    from_ts: Optional[datetime] = Query(default=None, alias="from", description="Inclusive start timestamp (ISO-8601)."),
    to_ts: Optional[datetime] = Query(default=None, alias="to", description="Inclusive end timestamp (ISO-8601)."),
    db: AsyncSession = Depends(get_db),
) -> AnalysisListResponse:
    filters = dict(
        verdict=verdict,
        session_id=session_id,
        is_synthetic=is_synthetic,
        capture_id=capture_id,
        protocol=protocol,
        posture=posture,
        from_ts=from_ts,
        to_ts=to_ts,
    )
    stmt = _apply_analysis_filters(select(AnalysisRecord), **filters)
    count_stmt = _apply_analysis_filters(select(func.count()).select_from(AnalysisRecord), **filters)

    total_result = await db.execute(count_stmt)
    total = total_result.scalar() or 0

    stmt = stmt.order_by(AnalysisRecord.timestamp.desc()).offset(skip).limit(limit)
    result = await db.execute(stmt)
    rows = result.scalars().all()

    return AnalysisListResponse(
        total=total,
        skip=skip,
        limit=limit,
        records=[_row_to_analysis(r) for r in rows],
    )


# ---------------------------------------------------------------------------
#  GET /analyses/stats  —  aggregate statistics
# ---------------------------------------------------------------------------

@data_router.get(
    "/analyses/stats",
    response_model=StatsResponse,
    summary="Aggregate analysis statistics",
    description=(
        "Return aggregate statistics across all stored analysis records: "
        "total count, flagged sessions, evidence refs, average risk score, "
        "verdict and cryptographic-posture distributions. Optional `from`/`to` "
        "limit the analysis aggregates; validation totals stay global."
    ),
)
async def analyses_stats(
    from_ts: Optional[datetime] = Query(default=None, alias="from", description="Inclusive start timestamp (ISO-8601)."),
    to_ts: Optional[datetime] = Query(default=None, alias="to", description="Inclusive end timestamp (ISO-8601)."),
    capture_id: Optional[str] = Query(default=None),
    protocol: Optional[str] = Query(default=None),
    db: AsyncSession = Depends(get_db),
) -> StatsResponse:
    filters = dict(
        verdict=None,
        session_id=None,
        is_synthetic=None,
        capture_id=capture_id,
        protocol=protocol,
        posture=None,
        from_ts=from_ts,
        to_ts=to_ts,
    )
    total_res = await db.execute(
        _apply_analysis_filters(select(func.count()).select_from(AnalysisRecord), **filters)
    )
    total_analyses = total_res.scalar() or 0

    synth_res = await db.execute(
        _apply_analysis_filters(
            select(func.count()).select_from(AnalysisRecord),
            **{**filters, "is_synthetic": True},
        )
    )
    total_synthetic = synth_res.scalar() or 0
    total_real = total_analyses - total_synthetic

    flagged_res = await db.execute(
        _apply_analysis_filters(select(func.count()).select_from(AnalysisRecord), **filters).where(
            or_(
                AnalysisRecord.final_verdict.in_(FLAGGED_VERDICTS),
                AnalysisRecord.rule_triggers_count > 0,
            )
        )
    )
    flagged_sessions = flagged_res.scalar() or 0

    evidence_res = await db.execute(
        _apply_analysis_filters(
            select(func.coalesce(func.sum(AnalysisRecord.evidence_ref_count), 0)),
            **filters,
        )
    )
    evidence_archived = int(evidence_res.scalar() or 0)

    avg_res = await db.execute(
        _apply_analysis_filters(select(func.avg(AnalysisRecord.risk_score)), **filters)
    )
    avg_risk = avg_res.scalar()

    dist_res = await db.execute(
        _apply_analysis_filters(
            select(AnalysisRecord.final_verdict, func.count().label("cnt")),
            **filters,
        )
        .group_by(AnalysisRecord.final_verdict)
        .order_by(func.count().desc())
    )
    verdict_distribution = [
        VerdictCount(verdict=row.final_verdict, count=row.cnt)
        for row in dist_res.all()
    ]

    posture_res = await db.execute(
        _apply_analysis_filters(
            select(AnalysisRecord.posture, func.count().label("cnt")).where(
                AnalysisRecord.posture.is_not(None)
            ),
            **filters,
        )
        .group_by(AnalysisRecord.posture)
        .order_by(func.count().desc())
    )
    cryptographic_posture_distribution = [
        PostureCount(posture=row.posture, count=row.cnt)
        for row in posture_res.all()
        if row.posture is not None
    ]

    val_total_res = await db.execute(select(func.count()).select_from(ValidationRecord))
    total_validations = val_total_res.scalar() or 0

    val_pass_res = await db.execute(
        select(func.count()).select_from(ValidationRecord).where(ValidationRecord.valid == True)
    )
    total_pass = val_pass_res.scalar() or 0
    pass_rate = (total_pass / total_validations) if total_validations > 0 else None

    return StatsResponse(
        total_analyses=total_analyses,
        total_synthetic=total_synthetic,
        total_real=total_real,
        flagged_sessions=flagged_sessions,
        evidence_archived=evidence_archived,
        avg_risk_score=float(avg_risk) if avg_risk is not None else None,
        verdict_distribution=verdict_distribution,
        cryptographic_posture_distribution=cryptographic_posture_distribution,
        total_validations=total_validations,
        validation_pass_rate=pass_rate,
    )


# ---------------------------------------------------------------------------
#  GET /analyses/{request_id}  —  fetch single record
# ---------------------------------------------------------------------------

@data_router.get(
    "/analyses/{request_id}",
    response_model=AnalysisRecordResponse,
    summary="Get a single analysis record",
    description="Fetch a persisted analysis record by its unique request_id.",
)
async def get_analysis(
    request_id: str,
    db: AsyncSession = Depends(get_db),
) -> AnalysisRecordResponse:
    result = await db.execute(
        select(AnalysisRecord).where(AnalysisRecord.request_id == request_id)
    )
    row = result.scalar_one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail=f"Analysis record '{request_id}' not found.")
    return _row_to_analysis(row)


# ---------------------------------------------------------------------------
#  DELETE /analyses/{request_id}  —  delete single record
# ---------------------------------------------------------------------------

@data_router.delete(
    "/analyses/{request_id}",
    response_model=DeleteResponse,
    summary="Delete a single analysis record",
    description="Permanently delete one analysis record by request_id.",
)
async def delete_analysis(
    request_id: str,
    db: AsyncSession = Depends(get_db),
) -> DeleteResponse:
    result = await db.execute(
        select(AnalysisRecord).where(AnalysisRecord.request_id == request_id)
    )
    row = result.scalar_one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail=f"Analysis record '{request_id}' not found.")
    await db.delete(row)
    return DeleteResponse(deleted=1, message=f"Analysis record '{request_id}' deleted.")


# ---------------------------------------------------------------------------
#  POST /analyses/synthetic  —  insert synthetic analysis row
# ---------------------------------------------------------------------------

@data_router.post(
    "/analyses/synthetic",
    response_model=SyntheticAnalysisResponse,
    status_code=201,
    summary="Insert a synthetic analysis record",
    description=(
        "Insert a hand-crafted analysis result directly into the database "
        "without running ML inference. Useful for loading test fixtures, "
        "boundary cases, and labelled examples for dashboard testing."
    ),
)
async def create_synthetic_analysis(
    body: SyntheticAnalysisRequest,
    db: AsyncSession = Depends(get_db),
) -> SyntheticAnalysisResponse:
    request_id = generate_request_id()

    # Validate verdict value
    allowed_verdicts = {"benign", "suspicious", "malicious", "informational"}
    if body.final_verdict.lower() not in allowed_verdicts:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid final_verdict '{body.final_verdict}'. Must be one of: {sorted(allowed_verdicts)}",
        )

    db_record = AnalysisRecord(
        request_id=request_id,
        session_id=body.session_id,
        client_id=body.client_id,
        record_count=1,
        risk_score=body.risk_score,
        final_verdict=body.final_verdict.lower(),
        rule_score=body.rule_score,
        rule_triggers_count=body.rule_triggers_count,
        trigger_details=body.trigger_details,
        ml_scores=body.ml_scores,
        explanations=body.explanations,
        model_bundle=body.model_bundle,
        is_synthetic=True,
        source_label=body.source_label or "synthetic",
    )
    db.add(db_record)
    await db.flush()  # Get the auto-generated ID before commit
    record_id = db_record.id
    await db.commit()

    logger.info(
        "Synthetic analysis record created: request_id=%s session_id=%s verdict=%s",
        request_id,
        body.session_id,
        body.final_verdict,
    )

    return SyntheticAnalysisResponse(
        request_id=request_id,
        status="created",
        message=f"Synthetic analysis record inserted with verdict='{body.final_verdict}' and risk_score={body.risk_score}.",
        record_id=record_id,
    )


# ---------------------------------------------------------------------------
#  GET /validations  —  paginated list
# ---------------------------------------------------------------------------

@data_router.get(
    "/validations",
    response_model=ValidationListResponse,
    summary="List validation records",
    description=(
        "Return a paginated list of persisted pre-flight validation records. "
        "Optionally filter by `valid` status."
    ),
)
async def list_validations(
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=200),
    valid: Optional[bool] = Query(default=None, description="Filter by pass/fail status."),
    db: AsyncSession = Depends(get_db),
) -> ValidationListResponse:
    stmt = select(ValidationRecord)
    count_stmt = select(func.count()).select_from(ValidationRecord)

    if valid is not None:
        stmt = stmt.where(ValidationRecord.valid == valid)
        count_stmt = count_stmt.where(ValidationRecord.valid == valid)

    total_result = await db.execute(count_stmt)
    total = total_result.scalar() or 0

    stmt = stmt.order_by(ValidationRecord.timestamp.desc()).offset(skip).limit(limit)
    result = await db.execute(stmt)
    rows = result.scalars().all()

    return ValidationListResponse(
        total=total,
        skip=skip,
        limit=limit,
        records=[_row_to_validation(r) for r in rows],
    )


# ---------------------------------------------------------------------------
#  DELETE /analyses  —  bulk delete (requires ?confirm=true)
# ---------------------------------------------------------------------------

@data_router.delete(
    "/analyses",
    response_model=DeleteResponse,
    summary="Bulk-delete all analysis records",
    description=(
        "Permanently delete ALL analysis records from the database. "
        "Requires the query parameter `confirm=true` to prevent accidental wipes. "
        "Useful for resetting between test runs."
    ),
)
async def bulk_delete_analyses(
    confirm: bool = Query(
        default=False,
        description="Must be `true` to confirm the bulk delete operation.",
    ),
    synthetic_only: bool = Query(
        default=False,
        description="If `true`, only delete synthetic records; leave real ML runs intact.",
    ),
    db: AsyncSession = Depends(get_db),
) -> DeleteResponse:
    if not confirm:
        raise HTTPException(
            status_code=400,
            detail="Bulk delete requires ?confirm=true to prevent accidental data loss.",
        )

    if synthetic_only:
        count_res = await db.execute(
            select(func.count()).select_from(AnalysisRecord).where(AnalysisRecord.is_synthetic == True)
        )
        deleted_count = count_res.scalar() or 0
        await db.execute(delete(AnalysisRecord).where(AnalysisRecord.is_synthetic == True))
        msg = f"Deleted {deleted_count} synthetic analysis records."
    else:
        count_res = await db.execute(select(func.count()).select_from(AnalysisRecord))
        deleted_count = count_res.scalar() or 0
        await db.execute(delete(AnalysisRecord))
        msg = f"Deleted all {deleted_count} analysis records."

    logger.warning("Bulk delete executed: %s", msg)
    return DeleteResponse(deleted=deleted_count, message=msg)

"""Authenticated durable PCAP extraction jobs."""
from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from api.database import AnalysisRecord, CaptureJob, CaptureSession
from api.dependencies import get_db, require_api_key
from api.schemas import (
    AnalysisRecordResponse,
    CaptureAnalysisListResponse,
    CaptureJobListResponse,
    CaptureResponse,
    CaptureSessionListResponse,
    CaptureSessionPreview,
)

MAX_CAPTURE_BYTES = 20 * 1024 * 1024
READ_CHUNK_BYTES = 64 * 1024
CAPTURE_DIR = Path(os.environ.get("SECUREMAIL_CAPTURE_DIR", "/tmp/securemail-captures"))
PCAP_MAGIC = frozenset(
    {
        b"\xd4\xc3\xb2\xa1",
        b"\xa1\xb2\xc3\xd4",
        b"\x4d\x3c\xb2\xa1",
        b"\xa1\xb2\x3c\x4d",
        b"\x0a\x0d\x0d\x0a",
    }
)

capture_router = APIRouter(
    prefix="/api/v1",
    tags=["captures"],
    dependencies=[Depends(require_api_key)],
)


def _status(status: str) -> str:
    # Existing installations used "extracted" before jobs became asynchronous.
    return "complete" if status == "extracted" else status


def _response(job: CaptureJob) -> CaptureResponse:
    return CaptureResponse(
        job_id=job.capture_id,
        capture_id=job.capture_id,
        status=_status(job.status),
        pcap_sha256=job.pcap_sha256,
        filename=job.filename,
        attempts=job.attempts,
        session_count=job.session_count,
        processed_sessions=job.processed_sessions,
        progress=job.progress,
        error_code=job.error_code,
        diagnostics=job.diagnostics or {},
    )


def _analysis_response(row: AnalysisRecord) -> AnalysisRecordResponse:
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
        trigger_details=row.trigger_details or [],
        ml_scores=row.ml_scores or {},
        explanations=row.explanations or {},
        model_bundle=row.model_bundle or {},
        tls_details=row.tls_details,
        certificate_details=row.certificate_details,
        is_synthetic=row.is_synthetic,
        source_label=row.source_label,
    )


async def _write_capture(file: UploadFile) -> tuple[Path, str, str]:
    filename = Path(file.filename or "capture.pcap").name[:255] or "capture.pcap"
    CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".upload-", dir=CAPTURE_DIR)
    os.close(descriptor)
    path = Path(temporary_name)
    digest = hashlib.sha256()
    total = 0
    try:
        with path.open("wb") as destination:
            while chunk := await file.read(READ_CHUNK_BYTES):
                total += len(chunk)
                if total > MAX_CAPTURE_BYTES:
                    raise HTTPException(status_code=413, detail="PCAP exceeds the 20 MiB upload limit.")
                digest.update(chunk)
                destination.write(chunk)
        with path.open("rb") as stored:
            magic = stored.read(4)
        if total < 4 or magic not in PCAP_MAGIC:
            raise HTTPException(status_code=422, detail="Upload is not a PCAP or PCAPNG file.")
        return path, digest.hexdigest(), filename
    except Exception:
        path.unlink(missing_ok=True)
        raise
    finally:
        await file.close()


async def _get_job(job_id: str, db: AsyncSession) -> CaptureJob:
    job = await db.scalar(select(CaptureJob).where(CaptureJob.capture_id == job_id))
    if job is None:
        raise HTTPException(status_code=404, detail="Capture job not found.")
    return job


@capture_router.post(
    "/capture-jobs",
    response_model=CaptureResponse,
    status_code=202,
    summary="Queue one authorized PCAP for extraction",
)
@capture_router.post(
    "/captures",
    response_model=CaptureResponse,
    status_code=202,
    include_in_schema=False,
)
async def create_capture_job(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
) -> CaptureResponse:
    """Store a bounded PCAP temporarily and enqueue its hash-idempotent job."""
    temporary_path, pcap_sha256, filename = await _write_capture(file)
    capture_id = f"pcap-{pcap_sha256[:16]}"
    final_path = CAPTURE_DIR / f"{capture_id}{'.pcapng' if filename.lower().endswith('.pcapng') else '.pcap'}"
    try:
        existing = await db.scalar(select(CaptureJob).where(CaptureJob.capture_id == capture_id))
        if existing is not None:
            temporary_path.unlink(missing_ok=True)
            return _response(existing)

        os.replace(temporary_path, final_path)
        job = CaptureJob(
            capture_id=capture_id,
            filename=filename,
            pcap_sha256=pcap_sha256,
            status="queued",
            input_path=str(final_path),
            attempts=0,
            max_attempts=2,
            session_count=0,
            processed_sessions=0,
            progress=None,
            diagnostics={},
        )
        db.add(job)
        try:
            await db.commit()
        except IntegrityError:
            await db.rollback()
            existing = await db.scalar(select(CaptureJob).where(CaptureJob.capture_id == capture_id))
            if existing is None:
                raise
            final_path.unlink(missing_ok=True)
            return _response(existing)
        return _response(job)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        final_path.unlink(missing_ok=True)
        raise


@capture_router.get(
    "/capture-jobs",
    response_model=CaptureJobListResponse,
    summary="List PCAP extraction jobs",
)
async def list_capture_jobs(
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
) -> CaptureJobListResponse:
    total = int((await db.scalar(select(func.count()).select_from(CaptureJob))) or 0)
    jobs = list(
        (
            await db.scalars(
                select(CaptureJob)
                .order_by(CaptureJob.created_at.desc())
                .offset(skip)
                .limit(limit)
            )
        ).all()
    )
    return CaptureJobListResponse(
        jobs=[_response(job) for job in jobs],
        total=total,
        skip=skip,
        limit=limit,
    )


@capture_router.get(
    "/capture-jobs/{job_id}",
    response_model=CaptureResponse,
    summary="Get PCAP extraction job status",
)
@capture_router.get(
    "/captures/{job_id}",
    response_model=CaptureResponse,
    include_in_schema=False,
)
async def capture_job_status(job_id: str, db: AsyncSession = Depends(get_db)) -> CaptureResponse:
    return _response(await _get_job(job_id, db))


@capture_router.get(
    "/capture-jobs/{job_id}/sessions",
    response_model=CaptureSessionListResponse,
    summary="Return extracted session JSON for one job",
)
@capture_router.get(
    "/captures/{job_id}/sessions",
    response_model=CaptureSessionListResponse,
    include_in_schema=False,
)
async def list_capture_sessions(
    job_id: str,
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=200, ge=1, le=500),
    db: AsyncSession = Depends(get_db),
) -> CaptureSessionListResponse:
    job = await _get_job(job_id, db)
    total = int(
        (
            await db.scalar(
                select(func.count()).select_from(CaptureSession).where(CaptureSession.capture_id == job.capture_id)
            )
        )
        or 0
    )
    sessions = list(
        (
            await db.scalars(
                select(CaptureSession)
                .where(CaptureSession.capture_id == job.capture_id)
                .order_by(CaptureSession.id)
                .offset(skip)
                .limit(limit)
            )
        ).all()
    )
    return CaptureSessionListResponse(
        capture_id=job.capture_id,
        skip=skip,
        limit=limit,
        total=total,
        sessions=[CaptureSessionPreview.model_validate(item.preview) for item in sessions],
        records=[item.record for item in sessions],
    )


@capture_router.get(
    "/capture-jobs/{job_id}/results",
    response_model=CaptureAnalysisListResponse,
    summary="List persisted analysis results for one job",
)
async def list_capture_results(
    job_id: str,
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
) -> CaptureAnalysisListResponse:
    job = await _get_job(job_id, db)
    total = int(
        (
            await db.scalar(
                select(func.count()).select_from(AnalysisRecord).where(AnalysisRecord.capture_id == job.capture_id)
            )
        )
        or 0
    )
    rows = list(
        (
            await db.scalars(
                select(AnalysisRecord)
                .where(AnalysisRecord.capture_id == job.capture_id)
                .order_by(AnalysisRecord.timestamp.desc())
                .offset(skip)
                .limit(limit)
            )
        ).all()
    )
    return CaptureAnalysisListResponse(
        job_id=job.capture_id,
        total=total,
        skip=skip,
        limit=limit,
        records=[_analysis_response(row) for row in rows],
    )

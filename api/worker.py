"""Postgres-backed PCAP extraction worker."""
from __future__ import annotations

import asyncio
import logging
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

from sqlalchemy import select, update, delete

from api.database import CaptureJob, CaptureSession
from api.dependencies import AsyncSessionLocal, init_db
from ml.pcap import PcapExtractionUnavailable, extract_authorized_capture

logger = logging.getLogger("securemailscope.worker")
POLL_SECONDS = float(os.environ.get("SECUREMAIL_CAPTURE_POLL_SECONDS", "1"))
EXTRACTION_TIMEOUT_SECONDS = float(
    os.environ.get("SECUREMAIL_CAPTURE_TIMEOUT_SECONDS", "120")
)
LEASE_TIMEOUT_SECONDS = max(EXTRACTION_TIMEOUT_SECONDS * 2, 300)


async def _claim_next_job() -> int | None:
    async with AsyncSessionLocal() as db:
        now = datetime.now(UTC)
        stale_before = now - timedelta(seconds=LEASE_TIMEOUT_SECONDS)
        await db.execute(
            update(CaptureJob)
            .where(
                CaptureJob.status == "running",
                CaptureJob.started_at.is_not(None),
                CaptureJob.started_at < stale_before,
            )
            .values(
                status="queued",
                started_at=None,
                error_code="worker_lease_expired",
                diagnostics={"worker": ["Previous worker lease expired; job requeued."]},
            )
        )
        job = await db.scalar(
            select(CaptureJob)
            .where(
                CaptureJob.status == "queued",
                CaptureJob.attempts < CaptureJob.max_attempts,
            )
            .order_by(CaptureJob.created_at)
            .limit(1)
            .with_for_update(skip_locked=True)
        )
        if job is None:
            await db.commit()
            return None
        job.status = "running"
        job.attempts = (job.attempts or 0) + 1
        job.started_at = now
        job.finished_at = None
        job.progress = 0.0
        job.error_code = None
        job.diagnostics = {}
        await db.commit()
        return job.id


async def _mark_failure(job_id: int, error_code: str, *, retryable: bool) -> None:
    async with AsyncSessionLocal() as db:
        job = await db.get(CaptureJob, job_id)
        if job is None:
            return
        should_retry = retryable and (job.attempts or 0) < (job.max_attempts or 1)
        job.status = "queued" if should_retry else "failed"
        job.error_code = error_code
        job.diagnostics = {"worker": ["PCAP extraction failed."]}
        job.finished_at = None if should_retry else datetime.now(UTC)
        job.progress = None
        if not should_retry and job.input_path:
            Path(job.input_path).unlink(missing_ok=True)
            job.input_path = None
        await db.commit()


async def _process_job(job_id: int) -> None:
    async with AsyncSessionLocal() as db:
        job = await db.get(CaptureJob, job_id)
        if job is None or job.status != "running":
            return
        input_path = Path(job.input_path) if job.input_path else None
        capture_id = job.capture_id

    if input_path is None or not input_path.is_file():
        await _mark_failure(job_id, "capture_file_missing", retryable=False)
        return

    try:
        records = await asyncio.wait_for(
            asyncio.to_thread(
                extract_authorized_capture,
                input_path,
                environment_id="api_worker",
            ),
            timeout=EXTRACTION_TIMEOUT_SECONDS,
        )
    except PcapExtractionUnavailable:
        await _mark_failure(job_id, "extractor_unavailable", retryable=True)
        return
    except (ValueError, TimeoutError, asyncio.TimeoutError):
        await _mark_failure(job_id, "extraction_invalid_or_timed_out", retryable=True)
        return
    except Exception:
        logger.exception("Unexpected extraction failure for capture job %s", capture_id)
        await _mark_failure(job_id, "extraction_error", retryable=True)
        return

    try:
        async with AsyncSessionLocal() as db:
            job = await db.get(CaptureJob, job_id)
            if job is None:
                return
            await db.execute(delete(CaptureSession).where(CaptureSession.capture_id == capture_id))
            for record in records:
                preview = _preview(record)
                db.add(
                    CaptureSession(
                        capture_id=capture_id,
                        session_id=record.provenance.session_id,
                        protocol=record.features.protocol.value,
                        dest_port=record.features.dst_port,
                        preview=preview,
                        record=record.model_dump(mode="json"),
                    )
                )
            job.status = "complete" if records else "empty"
            job.session_count = len(records)
            job.processed_sessions = len(records)
            job.progress = 100.0
            job.finished_at = datetime.now(UTC)
            job.input_path = None
            job.diagnostics = {}
            await db.commit()
    except Exception:
        logger.exception("Failed to persist extraction for capture job %s", capture_id)
        await _mark_failure(job_id, "session_persistence_error", retryable=True)
        return
    finally:
        input_path.unlink(missing_ok=True)


def _preview(record) -> dict[str, object]:
    features = record.features
    return {
        "session_id": record.provenance.session_id,
        "protocol": features.protocol.value,
        "src_port": features.src_port,
        "dst_port": features.dst_port,
        "tls_version": features.tls_version,
        "cipher_suite": features.cipher_suite,
        "starttls_advertised": features.starttls_advertised,
        "starttls_used": features.starttls_used,
        "handshake_success": features.handshake_success,
        "cert_present": features.cert_present,
        "cert_expired": features.cert_expired,
        "hostname_mismatch": features.hostname_mismatch,
        "tls_details": features.tls_details,
        "certificate_details": features.certificate_details,
        "evidence_refs": [ref.model_dump(mode="json") for ref in record.provenance.evidence_refs],
        "checked_views": ["protocol_session", "tls", "certificate"],
    }


async def run_once() -> bool:
    """Claim and process at most one job; return whether work was performed."""
    job_id = await _claim_next_job()
    if job_id is None:
        return False
    await _process_job(job_id)
    return True


async def run_worker() -> None:
    await init_db()
    while True:
        try:
            if not await run_once():
                await asyncio.sleep(POLL_SECONDS)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Capture worker loop failed")
            await asyncio.sleep(POLL_SECONDS)


if __name__ == "__main__":
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
    asyncio.run(run_worker())

"""Tests for authenticated, bounded PCAP extraction endpoints."""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.app import create_app
from api.dependencies import reset_runtime
from ml.schema import (
    AnomalyLabel,
    EvidenceReference,
    EvidenceSource,
    Protocol,
    Provenance,
    RiskLabel,
    SessionFeatureRecord,
    SessionFeatures,
    SessionLabels,
    SourceType,
)

PCAP_HEADER = b"\xd4\xc3\xb2\xa1" + (b"\x00" * 20)


def _record() -> SessionFeatureRecord:
    return SessionFeatureRecord(
        schema_version="session-features.v1",
        provenance=Provenance(
            capture_id="pcap-aabbccddeeff0011",
            flow_id="pcap-aabbccddeeff0011:stream:7",
            session_id="pcap-aabbccddeeff0011:stream:7",
            source_type=SourceType.AUTHORIZED_CAPTURE,
            scenario_id="authorized_capture",
            environment_id="api_upload",
            evidence_refs=[
                EvidenceReference(
                    source=EvidenceSource.PCAP,
                    stream_id=7,
                    packet_start=10,
                    packet_end=20,
                    fields=["tcp.stream", "tls.handshake.type"],
                )
            ],
        ),
        features=SessionFeatures(
            protocol=Protocol.SMTP,
            src_port=50000,
            dst_port=25,
            starttls_advertised=True,
            starttls_used=True,
            handshake_success=True,
            handshake_failures=0,
            renegotiation_count=0,
            session_duration_seconds=1.0,
            packet_count=11,
            byte_count=1024,
            retransmission_count=0,
            out_of_order_count=0,
            tls_version="TLS1.3",
            cipher_suite="TLS_AES_256_GCM_SHA384",
            cipher_family="AES-GCM",
            key_exchange="ECDHE",
            signature_algorithm="RSA-PSS",
            forward_secrecy=True,
            cert_present=True,
            cert_valid=True,
            cert_expired=False,
            cert_expires_in_days=90.0,
            cert_chain_valid=True,
            hostname_mismatch=False,
            cert_key_algorithm="RSA",
            cert_key_length_bits=2048,
            cert_signature_algorithm="SHA256-RSA",
        ),
        labels=SessionLabels(
            risk_label=RiskLabel.INFORMATIONAL,
            anomaly_label=AnomalyLabel.NORMAL,
        ),
    )


@pytest.fixture()
def client():
    reset_runtime()
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client
    reset_runtime()


def test_rejects_a_non_pcap_upload(client: TestClient) -> None:
    response = client.post(
        "/api/v1/captures",
        files={"file": ("not-a-capture.txt", b"not a pcap", "application/octet-stream")},
    )

    assert response.status_code == 422


def test_duplicate_upload_does_not_leave_a_temporary_file(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    upload = {"file": ("duplicate.pcap", PCAP_HEADER + b"duplicate", "application/vnd.tcpdump.pcap")}
    first = client.post("/api/v1/capture-jobs", files=upload)
    assert first.status_code == 202

    second = client.post(
        "/api/v1/capture-jobs",
        files={"file": ("duplicate-again.pcap", PCAP_HEADER + b"duplicate", "application/vnd.tcpdump.pcap")},
    )

    assert second.status_code == 202
    from api.capture_routes import CAPTURE_DIR

    assert not list(CAPTURE_DIR.glob(".upload-*"))

    monkeypatch.setattr("api.worker.extract_authorized_capture", lambda *_args, **_kwargs: [])
    from api.worker import run_once

    assert asyncio.run(run_once()) is True


def test_upload_queues_a_job_and_worker_persists_session_json(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = _record()
    monkeypatch.setattr(
        "api.worker.extract_authorized_capture",
        lambda *_args, **_kwargs: [record],
    )

    response = client.post(
        "/api/v1/capture-jobs",
        files={"file": ("mail.pcap", PCAP_HEADER + b"job", "application/vnd.tcpdump.pcap")},
    )

    assert response.status_code == 202
    payload = response.json()
    assert payload["status"] == "queued"
    assert payload["job_id"] == payload["capture_id"]
    assert payload["session_count"] == 0

    status = client.get(f"/api/v1/capture-jobs/{payload['job_id']}")
    assert status.status_code == 200
    assert status.json()["status"] == "queued"

    from api.worker import run_once

    assert asyncio.run(run_once()) is True

    completed = client.get(f"/api/v1/capture-jobs/{payload['job_id']}")
    assert completed.status_code == 200
    assert completed.json()["status"] == "complete"
    assert completed.json()["session_count"] == 1

    sessions = client.get(f"/api/v1/capture-jobs/{payload['job_id']}/sessions")
    assert sessions.status_code == 200
    assert sessions.json()["records"][0]["provenance"]["session_id"] == record.provenance.session_id

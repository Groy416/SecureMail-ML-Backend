from __future__ import annotations

from pathlib import Path

import pytest

from ml.product import (
    DEFAULT_BUNDLE,
    PayloadRejected,
    inference_record,
    load_runtime,
    score_session,
)
from ml.schema import RiskLabel, SourceType


def _authorized_payload(**feature_overrides: object) -> dict[str, object]:
    features = {
        "protocol": "SMTP",
        "src_port": 50000,
        "dst_port": 25,
        "starttls_advertised": True,
        "starttls_used": True,
        "handshake_success": True,
        "handshake_failures": 0,
        "renegotiation_count": 0,
        "session_duration_seconds": 1.0,
        "packet_count": 10,
        "byte_count": 1000,
        "retransmission_count": 0,
        "out_of_order_count": 0,
        "tls_version": "TLS1.0",
        "cipher_suite": "TLS_RSA_WITH_AES_128_CBC_SHA",
        "cipher_family": "AES-CBC",
        "key_exchange": "RSA",
        "signature_algorithm": "SHA1-RSA",
        "forward_secrecy": False,
        "cert_present": True,
        "cert_valid": True,
        "cert_expired": False,
        "cert_expires_in_days": 200.0,
        "cert_chain_valid": True,
        "hostname_mismatch": False,
        "cert_key_algorithm": "RSA",
        "cert_key_length_bits": 2048,
        "cert_signature_algorithm": "SHA1-RSA",
    }
    features.update(feature_overrides)
    return {
        "schema_version": "session-features.v1",
        "provenance": {
            "capture_id": "live-1",
            "flow_id": "flow-1",
            "session_id": "sess-1",
            "source_type": "authorized_capture",
            "scenario_id": "authorized_live",
            "environment_id": "roundcap",
            "evidence_refs": [
                {
                    "source": "pcap",
                    "stream_id": 1,
                    "packet_start": 1,
                    "packet_end": 10,
                    "fields": ["tls.handshake.type"],
                }
            ],
        },
        "features": features,
    }


def test_product_defaults_to_the_packet_backed_shadow_bundle() -> None:
    assert Path(DEFAULT_BUNDLE) == Path("models/grouped-105-capture-pcap")


def test_product_rejects_mail_body_payloads() -> None:
    payload = _authorized_payload()
    payload["body"] = "please ignore this message"

    with pytest.raises(PayloadRejected, match="body"):
        inference_record(payload)


def test_product_accepts_authorized_capture_without_training_labels() -> None:
    record = inference_record(_authorized_payload())

    assert record.provenance.source_type is SourceType.AUTHORIZED_CAPTURE
    assert record.labels.risk_label is RiskLabel.INFORMATIONAL


def test_product_scores_frozen_bundle_with_rules() -> None:
    bundle_dir = Path("models/grouped-105-capture-pcap")
    if not (bundle_dir / "manifest.json").is_file():
        pytest.skip("frozen production bundle is not present")
    bundle, calibration = load_runtime(bundle_dir)
    result = score_session(bundle, calibration, inference_record(_authorized_payload()))

    assert result.risk.risk_class is RiskLabel.CRITICAL
    assert result.action.value == "critical_review"
    assert any(finding.finding_id == "TLS-001" for finding in result.rule_findings)

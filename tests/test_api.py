"""Tests for the SecureMail-ML HTTP API.

Uses FastAPI's TestClient to verify the analysis endpoint, health checks,
bundle metadata, rules listing, validation, and error handling.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from api.app import create_app
from api.dependencies import load_ml_runtime, reset_runtime


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

BUNDLE_DIR = Path("models/Model_XG_RF")
BUNDLE_AVAILABLE = (BUNDLE_DIR / "manifest.json").is_file()


@pytest.fixture(scope="module")
def client():
    """Create a TestClient with the ML runtime loaded once for the module."""
    reset_runtime()
    os.environ.pop("SECUREMAIL_API_KEY", None)
    app = create_app()
    with TestClient(app) as c:
        yield c
    reset_runtime()


def _valid_authorized_payload(**feature_overrides: Any) -> dict[str, Any]:
    """Return a minimal valid authorized-capture session payload."""
    features: dict[str, Any] = {
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
        "tls_version": "TLS1.3",
        "cipher_suite": "TLS_AES_256_GCM_SHA384",
        "cipher_family": "AES-GCM",
        "key_exchange": "ECDHE",
        "signature_algorithm": "RSA-PSS",
        "forward_secrecy": True,
        "cert_present": True,
        "cert_valid": True,
        "cert_expired": False,
        "cert_expires_in_days": 200.0,
        "cert_chain_valid": True,
        "hostname_mismatch": False,
        "cert_key_algorithm": "RSA",
        "cert_key_length_bits": 2048,
        "cert_signature_algorithm": "SHA256-RSA",
    }
    features.update(feature_overrides)
    return {
        "schema_version": "session-features.v1",
        "provenance": {
            "capture_id": "live-api-test",
            "flow_id": "flow-1",
            "session_id": "sess-api-test-1",
            "source_type": "authorized_capture",
            "scenario_id": "authorized_live",
            "environment_id": "api_test",
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


def _deprecated_tls_payload() -> dict[str, Any]:
    """Return a valid payload with deprecated TLS 1.0 — triggers TLS-001."""
    return _valid_authorized_payload(
        tls_version="TLS1.0",
        cipher_suite="TLS_RSA_WITH_AES_128_CBC_SHA",
        cipher_family="AES-CBC",
        key_exchange="RSA",
        signature_algorithm="SHA1-RSA",
        forward_secrecy=False,
    )


def _analysis_request(record: dict[str, Any]) -> dict[str, Any]:
    """Wrap a record in the AnalysisRequest envelope."""
    return {
        "schema_version": "analysis-request.v1",
        "record": record,
    }


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------


class TestHealth:
    def test_health_returns_status(self, client: TestClient) -> None:
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ("healthy", "unavailable")

    @pytest.mark.skipif(not BUNDLE_AVAILABLE, reason="Model bundle not present")
    def test_health_shows_bundle_info_when_loaded(
        self, client: TestClient
    ) -> None:
        response = client.get("/api/v1/health")
        data = response.json()
        assert data["bundle_loaded"] is True
        assert data["bundle_version"] is not None
        assert len(data["model_names"]) > 0


# ---------------------------------------------------------------------------
# Bundle metadata endpoint
# ---------------------------------------------------------------------------


class TestBundleInfo:
    @pytest.mark.skipif(not BUNDLE_AVAILABLE, reason="Model bundle not present")
    def test_bundle_returns_metadata(self, client: TestClient) -> None:
        response = client.get("/api/v1/bundle")
        assert response.status_code == 200
        data = response.json()
        assert "bundle_version" in data
        assert "model_names" in data
        assert "feature_count" in data
        assert data["feature_count"] > 0
        assert len(data["feature_names"]) == data["feature_count"]
        assert "calibration_version" in data


# ---------------------------------------------------------------------------
# Rules endpoint
# ---------------------------------------------------------------------------


class TestRules:
    def test_rules_returns_all_known_rules(self, client: TestClient) -> None:
        response = client.get("/api/v1/rules")
        assert response.status_code == 200
        data = response.json()
        rules = data["rules"]
        assert len(rules) >= 13  # at least the 13 documented rules
        finding_ids = [rule["finding_id"] for rule in rules]
        assert "TLS-001" in finding_ids
        assert "TLS-002" in finding_ids
        assert "CERT-001" in finding_ids
        assert "ANOM-004" in finding_ids

    def test_each_rule_has_required_fields(self, client: TestClient) -> None:
        response = client.get("/api/v1/rules")
        for rule in response.json()["rules"]:
            assert rule["finding_id"]
            assert rule["severity"]
            assert rule["title"]
            assert rule["condition"]


# ---------------------------------------------------------------------------
# Analysis endpoint — happy path
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not BUNDLE_AVAILABLE, reason="Model bundle not present")
class TestAnalysisHappyPath:
    def test_valid_session_returns_complete(self, client: TestClient) -> None:
        payload = _analysis_request(_valid_authorized_payload())
        response = client.post("/api/v1/analyses", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["schema_version"] == "analysis-response.v1"
        assert data["status"] in ("complete", "degraded")
        assert data["request_id"]

    def test_response_contains_ml_result(self, client: TestClient) -> None:
        payload = _analysis_request(_valid_authorized_payload())
        response = client.post("/api/v1/analyses", json=payload)
        data = response.json()
        result = data["result"]
        assert result["schema_version"] == "ml-result.v1"
        assert "class" in result["risk"]
        assert "score" in result["risk"]
        assert "source" in result["risk"]
        assert "action" in result

    def test_response_contains_session_context(
        self, client: TestClient
    ) -> None:
        payload = _analysis_request(_valid_authorized_payload())
        response = client.post("/api/v1/analyses", json=payload)
        session = response.json()["session"]
        assert session["capture_id"] == "live-api-test"
        assert session["session_id"] == "sess-api-test-1"
        assert session["source_type"] == "authorized_capture"
        assert session["protocol"] == "SMTP"
        assert session["src_port"] == 50000
        assert session["dst_port"] == 25
        assert "observations" in session
        assert session["observations"]["tls_version"] == "TLS1.3"

    def test_deterministic_rule_appears_in_result(
        self, client: TestClient
    ) -> None:
        """TLS 1.0 should trigger the TLS-001 critical finding."""
        payload = _analysis_request(_deprecated_tls_payload())
        response = client.post("/api/v1/analyses", json=payload)
        assert response.status_code == 200
        data = response.json()
        result = data["result"]
        finding_ids = [f["finding_id"] for f in result["rule_findings"]]
        assert "TLS-001" in finding_ids
        # Policy: rules cannot be downgraded
        assert result["risk"]["class"] == "critical"

    def test_request_id_is_unique(self, client: TestClient) -> None:
        payload = _analysis_request(_valid_authorized_payload())
        r1 = client.post("/api/v1/analyses", json=payload)
        r2 = client.post("/api/v1/analyses", json=payload)
        assert r1.json()["request_id"] != r2.json()["request_id"]


# ---------------------------------------------------------------------------
# Analysis endpoint — payload rejection
# ---------------------------------------------------------------------------


class TestAnalysisPayloadRejection:
    @pytest.mark.skipif(
        not BUNDLE_AVAILABLE, reason="Model bundle not present"
    )
    def test_rejects_body_field(self, client: TestClient) -> None:
        record = _valid_authorized_payload()
        record["body"] = "this is an email body"
        payload = _analysis_request(record)
        response = client.post("/api/v1/analyses", json=payload)
        assert response.status_code == 422
        detail = response.json()["detail"]
        assert detail["status"] == "rejected"
        assert detail["error"]["code"] == "payload_rejected"

    @pytest.mark.skipif(
        not BUNDLE_AVAILABLE, reason="Model bundle not present"
    )
    def test_rejects_password_field(self, client: TestClient) -> None:
        record = _valid_authorized_payload()
        record["password"] = "s3cret"
        payload = _analysis_request(record)
        response = client.post("/api/v1/analyses", json=payload)
        assert response.status_code == 422
        detail = response.json()["detail"]
        assert "password" in detail["error"]["message"]

    @pytest.mark.skipif(
        not BUNDLE_AVAILABLE, reason="Model bundle not present"
    )
    def test_rejects_nested_sensitive_field(self, client: TestClient) -> None:
        record = _valid_authorized_payload()
        record["metadata"] = {"token": "abc123"}
        payload = _analysis_request(record)
        response = client.post("/api/v1/analyses", json=payload)
        assert response.status_code == 422
        detail = response.json()["detail"]
        assert detail["error"]["code"] == "payload_rejected"


# ---------------------------------------------------------------------------
# Analysis endpoint — validation errors
# ---------------------------------------------------------------------------


class TestAnalysisValidation:
    @pytest.mark.skipif(
        not BUNDLE_AVAILABLE, reason="Model bundle not present"
    )
    def test_rejects_empty_record(self, client: TestClient) -> None:
        payload = _analysis_request({})
        response = client.post("/api/v1/analyses", json=payload)
        assert response.status_code == 422

    @pytest.mark.skipif(
        not BUNDLE_AVAILABLE, reason="Model bundle not present"
    )
    def test_rejects_missing_features(self, client: TestClient) -> None:
        record = _valid_authorized_payload()
        del record["features"]
        payload = _analysis_request(record)
        response = client.post("/api/v1/analyses", json=payload)
        assert response.status_code == 422

    @pytest.mark.skipif(
        not BUNDLE_AVAILABLE, reason="Model bundle not present"
    )
    def test_rejects_negative_packet_count(self, client: TestClient) -> None:
        record = _valid_authorized_payload(packet_count=-1)
        payload = _analysis_request(record)
        response = client.post("/api/v1/analyses", json=payload)
        assert response.status_code == 422

    @pytest.mark.skipif(
        not BUNDLE_AVAILABLE, reason="Model bundle not present"
    )
    def test_rejects_invalid_protocol(self, client: TestClient) -> None:
        record = _valid_authorized_payload(protocol="FTP")
        payload = _analysis_request(record)
        response = client.post("/api/v1/analyses", json=payload)
        assert response.status_code == 422

    @pytest.mark.skipif(
        not BUNDLE_AVAILABLE, reason="Model bundle not present"
    )
    def test_rejects_missing_request_body(self, client: TestClient) -> None:
        response = client.post("/api/v1/analyses")
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# Analysis endpoint — authentication
# ---------------------------------------------------------------------------


class TestAnalysisAuthentication:
    def test_rejects_unauthenticated_when_key_required(self) -> None:
        """When SECUREMAIL_API_KEY is set, requests without it are rejected."""
        reset_runtime()
        os.environ["SECUREMAIL_API_KEY"] = "test-secret-key"
        try:
            app = create_app()
            with TestClient(app) as c:
                payload = _analysis_request(_valid_authorized_payload())
                response = c.post("/api/v1/analyses", json=payload)
                assert response.status_code == 401
                detail = response.json()["detail"]
                assert detail["error"]["code"] == "authentication_required"
        finally:
            os.environ.pop("SECUREMAIL_API_KEY", None)
            reset_runtime()

    @pytest.mark.skipif(
        not BUNDLE_AVAILABLE, reason="Model bundle not present"
    )
    def test_accepts_valid_api_key(self) -> None:
        reset_runtime()
        os.environ["SECUREMAIL_API_KEY"] = "test-secret-key"
        try:
            app = create_app()
            with TestClient(app) as c:
                payload = _analysis_request(_valid_authorized_payload())
                response = c.post(
                    "/api/v1/analyses",
                    json=payload,
                    headers={"Authorization": "test-secret-key"},
                )
                assert response.status_code == 200
        finally:
            os.environ.pop("SECUREMAIL_API_KEY", None)
            reset_runtime()


# ---------------------------------------------------------------------------
# Error response structure
# ---------------------------------------------------------------------------


class TestErrorResponseStructure:
    @pytest.mark.skipif(
        not BUNDLE_AVAILABLE, reason="Model bundle not present"
    )
    def test_error_contains_request_id(self, client: TestClient) -> None:
        record = _valid_authorized_payload()
        record["body"] = "sensitive content"
        payload = _analysis_request(record)
        response = client.post("/api/v1/analyses", json=payload)
        detail = response.json()["detail"]
        assert "request_id" in detail
        assert detail["request_id"]

    @pytest.mark.skipif(
        not BUNDLE_AVAILABLE, reason="Model bundle not present"
    )
    def test_error_does_not_echo_sensitive_values(
        self, client: TestClient
    ) -> None:
        """The error response must not echo the rejected payload values."""
        record = _valid_authorized_payload()
        record["body"] = "this is private email content that must not leak"
        payload = _analysis_request(record)
        response = client.post("/api/v1/analyses", json=payload)
        response_text = json.dumps(response.json())
        assert "this is private email content" not in response_text


# ---------------------------------------------------------------------------
# Pre-flight validation endpoint tests
# ---------------------------------------------------------------------------


class TestValidationEndpoint:
    def test_validate_valid_payload(self, client: TestClient) -> None:
        record = _valid_authorized_payload()
        response = client.post("/api/v1/validate", json={"record": record})
        assert response.status_code == 200
        data = response.json()
        assert data["schema_version"] == "validation-response.v1"
        assert data["valid"] is True
        assert data["status"] == "valid"
        assert data["errors"] == []

    def test_validate_sensitive_payload_rejected(self, client: TestClient) -> None:
        record = _valid_authorized_payload()
        record["body"] = "sensitive content"
        response = client.post("/api/v1/validate", json={"record": record})
        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is False
        assert data["status"] == "rejected"
        assert len(data["errors"]) > 0

    def test_validate_invalid_schema(self, client: TestClient) -> None:
        record = {"invalid": "payload"}
        response = client.post("/api/v1/validate", json={"record": record})
        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is False
        assert data["status"] == "invalid"
        assert len(data["errors"]) > 0


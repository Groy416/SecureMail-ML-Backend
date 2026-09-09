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

    def test_posted_analysis_appears_in_stats_and_list(self, client: TestClient) -> None:
        payload = _analysis_request(_deprecated_tls_payload())
        posted = client.post("/api/v1/analyses", json=payload)
        assert posted.status_code == 200
        request_id = posted.json()["request_id"]
        result = posted.json()["result"]

        stats = client.get("/api/v1/analyses/stats")
        assert stats.status_code == 200
        body = stats.json()
        assert body["total_analyses"] >= 1
        assert body["avg_risk_score"] is None or 0.0 <= body["avg_risk_score"] <= 1.0
        assert body["flagged_sessions"] >= 1
        assert body["evidence_archived"] >= 1
        verdicts = {row["verdict"] for row in body["verdict_distribution"]}
        assert result["risk"]["class"] in verdicts
        postures = {row["posture"] for row in body["cryptographic_posture_distribution"]}
        assert "deprecated" in postures

        listing = client.get("/api/v1/analyses?limit=50")
        assert listing.status_code == 200
        records = listing.json()["records"]
        match = next(row for row in records if row["request_id"] == request_id)
        assert match["final_verdict"] == result["risk"]["class"]
        assert match["risk_score"] == result["risk"]["score"]
        assert match["capture_id"] == "live-api-test"
        assert match["protocol"] == "SMTP"
        assert match["posture"] == "deprecated"
        assert match["rule_score"] is None


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

    def test_data_reads_require_the_raw_api_key(self) -> None:
        reset_runtime()
        os.environ["SECUREMAIL_API_KEY"] = "test-secret-key"
        try:
            app = create_app()
            with TestClient(app) as c:
                assert c.get("/api/v1/health").status_code == 200
                assert c.get("/api/v1/analyses/stats").status_code == 401
                assert c.get("/api/v1/bundle").status_code == 401
                assert c.get("/api/v1/rules").status_code == 401
                assert c.post("/api/v1/validate", json={"record": {}}).status_code == 401
                assert c.get(
                    "/api/v1/analyses/stats",
                    headers={"Authorization": "test-secret-key"},
                ).status_code == 200
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


# ---------------------------------------------------------------------------
# Data CRUD & Synthetic endpoints
# ---------------------------------------------------------------------------


class TestDataRoutes:
    """Tests for the database CRUD and synthetic data API endpoints."""

    # ------------------------------------------------------------------
    # POST /analyses/synthetic
    # ------------------------------------------------------------------

    def test_create_synthetic_analysis_malicious(self, client: TestClient) -> None:
        """Insert a synthetic malicious record and verify 201 + fields."""
        payload = {
            "session_id": "synth-sess-001",
            "client_id": "test-client",
            "source_label": "pytest-fixture",
            "risk_score": 0.95,
            "final_verdict": "malicious",
            "rule_score": 0.8,
            "rule_triggers_count": 3,
            "trigger_details": [{"finding_id": "TLS-001", "severity": "critical"}],
            "ml_scores": {"xgboost": 0.97, "random_forest": 0.93},
            "explanations": {"top_feature": "tls_version"},
            "model_bundle": {"version": "test-v1"},
        }
        response = client.post("/api/v1/analyses/synthetic", json=payload)
        assert response.status_code == 201
        data = response.json()
        assert data["status"] == "created"
        assert "request_id" in data
        assert "record_id" in data
        assert data["record_id"] > 0

    def test_create_synthetic_analysis_benign(self, client: TestClient) -> None:
        """Insert a synthetic benign record."""
        payload = {
            "session_id": "synth-sess-002",
            "risk_score": 0.05,
            "final_verdict": "benign",
        }
        response = client.post("/api/v1/analyses/synthetic", json=payload)
        assert response.status_code == 201
        data = response.json()
        assert data["status"] == "created"

    def test_create_synthetic_analysis_invalid_verdict(self, client: TestClient) -> None:
        """Reject unknown verdict values."""
        payload = {
            "session_id": "synth-sess-bad",
            "risk_score": 0.5,
            "final_verdict": "unknown_verdict",
        }
        response = client.post("/api/v1/analyses/synthetic", json=payload)
        assert response.status_code == 422

    def test_create_synthetic_analysis_score_out_of_range(self, client: TestClient) -> None:
        """Reject risk_score > 1.0."""
        payload = {
            "session_id": "synth-sess-bad2",
            "risk_score": 1.5,
            "final_verdict": "benign",
        }
        response = client.post("/api/v1/analyses/synthetic", json=payload)
        assert response.status_code == 422

    # ------------------------------------------------------------------
    # GET /analyses
    # ------------------------------------------------------------------

    def test_list_analyses_returns_list(self, client: TestClient) -> None:
        """List analyses returns a paginated response."""
        response = client.get("/api/v1/analyses")
        assert response.status_code == 200
        data = response.json()
        assert "total" in data
        assert "records" in data
        assert isinstance(data["records"], list)
        assert data["skip"] == 0
        assert data["limit"] == 50

    def test_list_analyses_filter_by_verdict(self, client: TestClient) -> None:
        """Filter analyses by verdict."""
        response = client.get("/api/v1/analyses?verdict=malicious")
        assert response.status_code == 200
        data = response.json()
        for record in data["records"]:
            assert record["final_verdict"] == "malicious"

    def test_list_analyses_filter_synthetic(self, client: TestClient) -> None:
        """Filter analyses to show only synthetic records."""
        response = client.get("/api/v1/analyses?is_synthetic=true")
        assert response.status_code == 200
        data = response.json()
        for record in data["records"]:
            assert record["is_synthetic"] is True

    def test_list_and_stats_date_filters(self, client: TestClient) -> None:
        client.post("/api/v1/analyses/synthetic", json={
            "session_id": "date-filter-sess",
            "risk_score": 0.2,
            "final_verdict": "informational",
        })
        future = client.get("/api/v1/analyses?from=2099-01-01T00:00:00Z")
        assert future.status_code == 200
        assert future.json()["total"] == 0
        future_stats = client.get("/api/v1/analyses/stats?from=2099-01-01T00:00:00Z")
        assert future_stats.status_code == 200
        assert future_stats.json()["total_analyses"] == 0
        past = client.get("/api/v1/analyses?from=2000-01-01T00:00:00Z")
        assert past.json()["total"] >= 1

    def test_list_analyses_pagination(self, client: TestClient) -> None:
        """Pagination skip/limit are reflected in the response."""
        response = client.get("/api/v1/analyses?skip=0&limit=1")
        assert response.status_code == 200
        data = response.json()
        assert len(data["records"]) <= 1
        assert data["limit"] == 1

    # ------------------------------------------------------------------
    # GET /analyses/{request_id}
    # ------------------------------------------------------------------

    def test_get_analysis_by_request_id(self, client: TestClient) -> None:
        """Fetch a specific analysis record by its request_id."""
        # First create one so we have a known request_id
        payload = {
            "session_id": "synth-get-test",
            "risk_score": 0.42,
            "final_verdict": "suspicious",
        }
        create_resp = client.post("/api/v1/analyses/synthetic", json=payload)
        assert create_resp.status_code == 201
        request_id = create_resp.json()["request_id"]

        response = client.get(f"/api/v1/analyses/{request_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["request_id"] == request_id
        assert data["final_verdict"] == "suspicious"
        assert data["is_synthetic"] is True

    def test_get_analysis_not_found(self, client: TestClient) -> None:
        """Return 404 for unknown request_id."""
        response = client.get("/api/v1/analyses/nonexistent-id-xyz")
        assert response.status_code == 404

    # ------------------------------------------------------------------
    # GET /analyses/stats
    # ------------------------------------------------------------------

    def test_stats_returns_expected_fields(self, client: TestClient) -> None:
        """Stats endpoint returns all expected aggregate fields."""
        response = client.get("/api/v1/analyses/stats")
        assert response.status_code == 200
        data = response.json()
        assert "total_analyses" in data
        assert "total_synthetic" in data
        assert "total_real" in data
        assert "avg_risk_score" in data
        assert "verdict_distribution" in data
        assert "cryptographic_posture_distribution" in data
        assert "flagged_sessions" in data
        assert "evidence_archived" in data
        assert "total_validations" in data
        assert "validation_pass_rate" in data

    def test_stats_synthetic_count_increases(self, client: TestClient) -> None:
        """Inserting a synthetic record increases total_synthetic."""
        before = client.get("/api/v1/analyses/stats").json()["total_synthetic"]
        client.post("/api/v1/analyses/synthetic", json={
            "session_id": "stats-test-sess",
            "risk_score": 0.1,
            "final_verdict": "benign",
        })
        after = client.get("/api/v1/analyses/stats").json()["total_synthetic"]
        assert after >= before + 1

    # ------------------------------------------------------------------
    # GET /validations
    # ------------------------------------------------------------------

    def test_list_validations_returns_list(self, client: TestClient) -> None:
        """Validations list endpoint returns paginated structure."""
        response = client.get("/api/v1/validations")
        assert response.status_code == 200
        data = response.json()
        assert "total" in data
        assert "records" in data
        assert isinstance(data["records"], list)

    # ------------------------------------------------------------------
    # DELETE /analyses/{request_id}
    # ------------------------------------------------------------------

    def test_delete_analysis_record(self, client: TestClient) -> None:
        """Delete a specific analysis record by request_id."""
        create_resp = client.post("/api/v1/analyses/synthetic", json={
            "session_id": "delete-test-sess",
            "risk_score": 0.3,
            "final_verdict": "informational",
        })
        request_id = create_resp.json()["request_id"]

        del_resp = client.delete(f"/api/v1/analyses/{request_id}")
        assert del_resp.status_code == 200
        assert del_resp.json()["deleted"] == 1

        # Confirm it's gone
        assert client.get(f"/api/v1/analyses/{request_id}").status_code == 404

    # ------------------------------------------------------------------
    # DELETE /analyses (bulk)
    # ------------------------------------------------------------------

    def test_bulk_delete_requires_confirm(self, client: TestClient) -> None:
        """Bulk delete without confirm=true should return 400."""
        response = client.delete("/api/v1/analyses")
        assert response.status_code == 400

    def test_bulk_delete_synthetic_only(self, client: TestClient) -> None:
        """Bulk delete with synthetic_only=true only removes synthetic rows."""
        # Insert a synthetic record so there's something to delete
        client.post("/api/v1/analyses/synthetic", json={
            "session_id": "bulk-del-test",
            "risk_score": 0.5,
            "final_verdict": "suspicious",
        })
        response = client.delete("/api/v1/analyses?confirm=true&synthetic_only=true")
        assert response.status_code == 200
        data = response.json()
        assert data["deleted"] >= 1



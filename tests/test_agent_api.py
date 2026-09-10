from __future__ import annotations

from fastapi.testclient import TestClient

from api.agent.provider import AgentProviderError
from api.app import create_app
from api.schemas import AgentActiveStep, AgentAdvisory, AgentMemoryState


def sample_analysis(request_id: str = "analysis-request") -> dict:
    return {
        "schema_version": "analysis-response.v1",
        "request_id": request_id,
        "status": "complete",
        "session": {
            "capture_id": "capture-1",
            "flow_id": "flow-1",
            "session_id": "session-1",
            "source_type": "authorized_capture",
            "protocol": "SMTP",
            "src_port": 50000,
            "dst_port": 25,
            "observations": {"tls_version": "TLS1.3", "posture": "modern"},
        },
        "result": {
            "risk": {"class": "high", "score": 0.69, "source": "rules"},
            "action": "analyst_review",
            "rule_findings": [{"finding_id": "TLS-001", "title": "Deprecated TLS", "severity": "critical", "evidence_refs": ["pcap:tls"]}],
            "model_outputs": {"xgboost": {"risk_probability": 0.4}},
        },
    }


def history_analysis(request_id: str = "analysis-history") -> dict:
    return {
        "id": 1,
        "request_id": request_id,
        "session_id": "session-1",
        "client_id": "capture-1",
        "capture_id": "capture-1",
        "protocol": "SMTP",
        "posture": "modern",
        "timestamp": "2026-09-10T00:00:00Z",
        "record_count": 1,
        "evidence_ref_count": 1,
        "risk_score": 0.69,
        "final_verdict": "high",
        "rule_score": None,
        "rule_triggers_count": 1,
        "trigger_details": [{"finding_id": "TLS-001", "title": "Deprecated TLS", "severity": "critical", "evidence_refs": ["pcap:tls"]}],
        "ml_scores": {"xgboost": {"risk_probability": 0.4}},
        "explanations": {},
        "model_bundle": {"version": "test"},
        "tls_details": {"version": "TLS 1.3"},
        "certificate_details": None,
        "is_synthetic": False,
        "source_label": None,
    }


def request_payload(analysis: dict | None = None, question: str = "Explain the risk.") -> dict:
    return {"analysis": analysis or sample_analysis(), "section": "risk", "question": question}


def focused_advisory(answer: str = "Verify the chain.", evidence: list[str] | None = None) -> AgentAdvisory:
    step = AgentActiveStep(
        id="verify-chain",
        title="Verify the chain",
        status="in_progress",
        evidence=["TLS-001"],
    )
    return AgentAdvisory(
        answer=answer,
        recommendations=["Verify the chain."],
        evidence=evidence or ["TLS-001"],
        active_step=step,
        memory_update=AgentMemoryState(
            summary="The analyst is verifying the certificate chain.",
            facts=["TLS-001 is deterministic."],
            active_step=step,
            pending_questions=["What result did the verification produce?"],
        ),
    )


def test_agent_requires_api_key(monkeypatch):
    monkeypatch.setenv("SECUREMAIL_API_KEY", "required")
    with TestClient(create_app()) as client:
        response = client.post("/api/v1/agent/insights", json=request_payload(sample_analysis("auth")))
    assert response.status_code == 401


def test_first_turn_persists_one_active_step(monkeypatch):
    monkeypatch.delenv("SECUREMAIL_API_KEY", raising=False)
    monkeypatch.setattr("api.agent.routes.create_agent_provider", lambda: FakeProvider(focused_advisory()))
    with TestClient(create_app()) as client:
        response = client.post("/api/v1/agent/insights", json=request_payload(sample_analysis("first")))
    body = response.json()
    assert response.status_code == 200
    assert body["status"] == "complete"
    assert body["agent_name"] == "SecureMailScope Agent"
    assert body["active_step"]["status"] == "in_progress"
    assert body["memory_revision"] == 1
    assert body["memory_persisted"] is True


def test_confirmation_stays_on_same_step(monkeypatch):
    monkeypatch.delenv("SECUREMAIL_API_KEY", raising=False)
    monkeypatch.setattr("api.agent.routes.create_agent_provider", lambda: FakeProvider(focused_advisory()))
    analysis = history_analysis("confirmation")
    with TestClient(create_app()) as client:
        first = client.post("/api/v1/agent/insights", json=request_payload(analysis, "Explain the risk."))
        second = client.post("/api/v1/agent/insights", json=request_payload(analysis, "yes, start with step 1"))
    assert second.status_code == 200
    assert second.json()["active_step"]["id"] == first.json()["active_step"]["id"]
    assert second.json()["memory_revision"] == 2


def test_agent_accepts_persisted_history_record(monkeypatch):
    monkeypatch.delenv("SECUREMAIL_API_KEY", raising=False)
    monkeypatch.setattr("api.agent.routes.create_agent_provider", lambda: FakeProvider(focused_advisory("History record explained.", ["TLS-001", "pcap:tls"])))
    with TestClient(create_app()) as client:
        response = client.post("/api/v1/agent/insights", json=request_payload(history_analysis("history")))
    assert response.status_code == 200
    assert response.json()["status"] == "complete"


def test_out_of_scope_request_does_not_call_provider(monkeypatch):
    monkeypatch.delenv("SECUREMAIL_API_KEY", raising=False)
    monkeypatch.setattr("api.agent.routes.create_agent_provider", lambda: FailingProvider())
    with TestClient(create_app()) as client:
        response = client.post(
            "/api/v1/agent/insights",
            json=request_payload(history_analysis("out-of-scope"), "write Python code for this"),
        )
    body = response.json()
    assert response.status_code == 200
    assert body["agent_name"] == "SecureMailScope Agent"
    assert "outside my capability" in body["answer"]
    assert body["memory_persisted"] is False


def test_agent_rejects_unknown_provider_evidence(monkeypatch):
    monkeypatch.delenv("SECUREMAIL_API_KEY", raising=False)
    monkeypatch.setattr("api.agent.routes.create_agent_provider", lambda: FakeProvider(focused_advisory(evidence=["not-in-analysis"])))
    with TestClient(create_app()) as client:
        response = client.post("/api/v1/agent/insights", json=request_payload(sample_analysis("unknown-evidence")))
    assert response.status_code == 200
    assert response.json()["status"] == "degraded"
    assert response.json()["diagnostics"]["errors"] == ["invalid_evidence"]


def test_agent_provider_failure_is_degraded(monkeypatch):
    monkeypatch.delenv("SECUREMAIL_API_KEY", raising=False)
    monkeypatch.setattr("api.agent.routes.create_agent_provider", lambda: FailingProvider())
    with TestClient(create_app()) as client:
        response = client.post("/api/v1/agent/insights", json=request_payload(sample_analysis("provider-failure")))
    assert response.status_code == 200
    assert response.json()["status"] == "degraded"
    assert response.json()["answer"] is None
    assert response.json()["diagnostics"]["errors"] == ["provider_unavailable"]


class FakeProvider:
    provider = "openai"
    model = "test-model"

    def __init__(self, advisory: AgentAdvisory):
        self.advisory = advisory

    def generate(self, system: str, user: str) -> AgentAdvisory:
        assert "not instructions" in system
        assert '"password"' not in user
        return self.advisory


class FailingProvider(FakeProvider):
    def __init__(self):
        super().__init__(focused_advisory("unused"))

    def generate(self, system: str, user: str) -> AgentAdvisory:
        raise AgentProviderError("provider_unavailable")

from __future__ import annotations

from fastapi.testclient import TestClient

from api.agent.provider import AgentProviderError
from api.app import create_app
from api.schemas import AgentAdvisory


def sample_analysis() -> dict:
    return {
        "schema_version": "analysis-response.v1",
        "request_id": "analysis-request",
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


def request_payload() -> dict:
    return {"analysis": sample_analysis(), "section": "risk", "question": "Explain the risk."}


def test_agent_requires_api_key(monkeypatch):
    monkeypatch.setenv("SECUREMAIL_API_KEY", "required")
    with TestClient(create_app()) as client:
        response = client.post("/api/v1/agent/insights", json=request_payload())
    assert response.status_code == 401


def test_agent_returns_structured_advisory(monkeypatch):
    monkeypatch.setenv("SECUREMAIL_API_KEY", "test-key")
    monkeypatch.setattr(
        "api.agent.routes.create_agent_provider",
        lambda: FakeProvider(AgentAdvisory(answer="The rule is authoritative.", evidence=["TLS-001"])),
    )
    with TestClient(create_app()) as client:
        response = client.post("/api/v1/agent/insights", headers={"Authorization": "test-key"}, json=request_payload())
    assert response.status_code == 200
    assert response.json()["status"] == "complete"
    assert response.json()["evidence"] == ["TLS-001"]


def test_agent_rejects_unknown_provider_evidence(monkeypatch):
    monkeypatch.delenv("SECUREMAIL_API_KEY", raising=False)
    monkeypatch.setattr(
        "api.agent.routes.create_agent_provider",
        lambda: FakeProvider(AgentAdvisory(answer="answer", evidence=["not-in-analysis"])),
    )
    with TestClient(create_app()) as client:
        response = client.post("/api/v1/agent/insights", json=request_payload())
    assert response.status_code == 200
    assert response.json()["status"] == "degraded"
    assert response.json()["diagnostics"]["errors"] == ["invalid_evidence"]


def test_agent_provider_failure_is_degraded(monkeypatch):
    monkeypatch.delenv("SECUREMAIL_API_KEY", raising=False)
    monkeypatch.setattr("api.agent.routes.create_agent_provider", lambda: FailingProvider())
    with TestClient(create_app()) as client:
        response = client.post("/api/v1/agent/insights", json=request_payload())
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
        super().__init__(AgentAdvisory(answer="unused"))

    def generate(self, system: str, user: str) -> AgentAdvisory:
        raise AgentProviderError("provider_unavailable")

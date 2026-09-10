from __future__ import annotations

from api.agent.context import allowed_evidence, build_agent_context
from api.schemas import AnalysisResponse
from tests.test_agent_api import sample_analysis


def test_risk_context_excludes_unknown_and_sensitive_fields():
    payload = sample_analysis()
    payload["result"]["secret"] = "must-not-forward"
    payload["session"]["raw_record"] = {"password": "must-not-forward"}
    analysis = AnalysisResponse.model_validate(payload)

    context = build_agent_context(analysis, "risk")

    assert context["risk"]["class"] == "high"
    assert "secret" not in str(context)
    assert "password" not in str(context)


def test_evidence_allowlist_contains_only_analysis_findings():
    analysis = AnalysisResponse.model_validate(sample_analysis())
    assert allowed_evidence(analysis) == {"TLS-001", "risk.class", "risk.score", "action"}

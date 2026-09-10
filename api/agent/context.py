from __future__ import annotations

from typing import Any

from api.schemas import AgentSection, AnalysisRecordResponse, AnalysisResponse

AgentAnalysis = AnalysisResponse | AnalysisRecordResponse


def _result(analysis: AgentAnalysis) -> dict[str, Any]:
    if isinstance(analysis, AnalysisRecordResponse):
        return {
            "risk": {
                "class": analysis.final_verdict,
                "score": analysis.risk_score,
                "source": "persisted_analysis",
            },
            "action": None,
            "rule_findings": analysis.trigger_details,
            "model_outputs": analysis.ml_scores,
        }
    return analysis.result


def _findings(analysis: AgentAnalysis) -> list[dict[str, Any]]:
    result = _result(analysis)
    findings = result.get("rule_findings", result.get("findings", []))
    if not isinstance(findings, list):
        return []
    safe: list[dict[str, Any]] = []
    for finding in findings:
        if not isinstance(finding, dict):
            continue
        item = {
            key: finding[key]
            for key in ("finding_id", "title", "severity", "condition", "evidence_refs", "remediation")
            if key in finding
        }
        if item.get("finding_id"):
            safe.append(item)
    return safe


def _safe_details(details: dict[str, Any] | None, keys: tuple[str, ...]) -> dict[str, Any]:
    if not details:
        return {}
    return {key: details[key] for key in keys if key in details}


def _safe_result(analysis: AgentAnalysis) -> dict[str, Any]:
    result = _result(analysis)
    risk = result.get("risk")
    return {
        "risk": {
            key: risk[key]
            for key in ("class", "score", "source")
            if isinstance(risk, dict) and key in risk
        },
        "action": result.get("action"),
        "findings": _findings(analysis),
        "model_signals": result.get("model_signals", result.get("model_outputs", {})),
        "diagnostics": result.get("diagnostics", {}),
    }


def build_agent_context(analysis: AgentAnalysis, section: AgentSection) -> dict[str, Any]:
    """Build a section-specific, non-sensitive context for a live or persisted analysis."""
    if isinstance(analysis, AnalysisRecordResponse):
        protocol = analysis.protocol
        observations: dict[str, Any] = {}
        tls_details = analysis.tls_details
        certificate_details = analysis.certificate_details
        session = {
            "session_id": analysis.session_id,
            "protocol": protocol,
            "posture": analysis.posture,
        }
    else:
        protocol = analysis.session.protocol.value
        observations = analysis.session.observations
        tls_details = analysis.tls_details
        certificate_details = analysis.certificate_details
        session = {
            "session_id": analysis.session.session_id,
            "protocol": protocol,
            "posture": observations.get("posture"),
        }

    result = _safe_result(analysis)
    if section == "overview":
        return {"session": session, "analysis": result}
    if section == "risk":
        return {
            "risk": result["risk"],
            "action": result["action"],
            "findings": result["findings"],
            "model_signals": result["model_signals"],
        }
    if section == "tls":
        return {
            "protocol": protocol,
            "observations": {
                key: observations[key]
                for key in (
                    "tls_version",
                    "cipher_suite",
                    "cipher_family",
                    "key_exchange",
                    "forward_secrecy",
                    "handshake_success",
                )
                if key in observations
            },
            "tls_details": _safe_details(
                tls_details,
                (
                    "version",
                    "cipher_suite",
                    "key_exchange",
                    "forward_secrecy",
                    "encryption",
                    "mac",
                    "posture_rating",
                ),
            ),
            "findings": [
                finding
                for finding in result["findings"]
                if str(finding["finding_id"]).startswith(("TLS-", "STLS-", "FS-", "ANOM-"))
            ],
        }
    if section == "certificate":
        return {
            "certificate_details": _safe_details(
                certificate_details,
                (
                    "domain",
                    "issuer",
                    "status",
                    "valid_from",
                    "valid_until",
                    "key_algorithm",
                    "signature_algorithm",
                    "chain",
                ),
            ),
            "observations": {
                key: observations[key]
                for key in (
                    "cert_present",
                    "cert_valid",
                    "cert_expired",
                    "cert_chain_valid",
                    "hostname_mismatch",
                    "cert_key_algorithm",
                    "cert_key_length_bits",
                )
                if key in observations
            },
            "findings": [
                finding
                for finding in result["findings"]
                if str(finding["finding_id"]).startswith("CERT-")
            ],
        }
    return {"findings": result["findings"]}


def allowed_evidence(analysis: AgentAnalysis) -> set[str]:
    """Return finding IDs and existing safe evidence references the model may cite."""
    evidence: set[str] = {"risk.class", "risk.score", "action"}
    for finding in _findings(analysis):
        evidence.add(str(finding["finding_id"]))
        for key in ("evidence_refs", "citations"):
            values = finding.get(key, [])
            if isinstance(values, list):
                evidence.update(value for value in values if isinstance(value, str) and value)
    return evidence

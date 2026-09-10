from __future__ import annotations

from typing import Any

from api.schemas import AgentSection, AnalysisResponse


def _findings(analysis: AnalysisResponse) -> list[dict[str, Any]]:
    findings = analysis.result.get("rule_findings", analysis.result.get("findings", []))
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


def _safe_result(analysis: AnalysisResponse) -> dict[str, Any]:
    result = analysis.result
    risk = result.get("risk")
    return {
        "risk": {
            key: risk[key]
            for key in ("class", "score", "source")
            if isinstance(risk, dict) and key in risk
        },
        "action": result.get("action"),
        "findings": _findings(analysis),
        "model_signals": result.get("model_signals", {}),
        "diagnostics": analysis.diagnostics,
    }


def build_agent_context(analysis: AnalysisResponse, section: AgentSection) -> dict[str, Any]:
    """Build a section-specific, non-sensitive context for the agent."""
    session = analysis.session
    result = _safe_result(analysis)
    if section == "overview":
        return {
            "session": {
                "session_id": session.session_id,
                "protocol": session.protocol.value,
                "posture": session.observations.get("posture"),
            },
            "analysis": result,
        }
    if section == "risk":
        return {"risk": result["risk"], "action": result["action"], "findings": result["findings"], "model_signals": result["model_signals"]}
    if section == "tls":
        return {
            "protocol": session.protocol.value,
            "observations": {
                key: session.observations[key]
                for key in ("tls_version", "cipher_suite", "cipher_family", "key_exchange", "forward_secrecy", "handshake_success")
                if key in session.observations
            },
            "tls_details": _safe_details(
                analysis.tls_details,
                ("version", "cipher_suite", "key_exchange", "forward_secrecy", "encryption", "mac", "posture_rating"),
            ),
            "findings": [finding for finding in result["findings"] if str(finding["finding_id"]).startswith(("TLS-", "STLS-", "FS-", "ANOM-"))],
        }
    if section == "certificate":
        return {
            "certificate_details": _safe_details(
                analysis.certificate_details,
                ("domain", "issuer", "status", "valid_from", "valid_until", "key_algorithm", "signature_algorithm", "chain"),
            ),
            "observations": {
                key: session.observations[key]
                for key in ("cert_present", "cert_valid", "cert_expired", "cert_chain_valid", "hostname_mismatch", "cert_key_algorithm", "cert_key_length_bits")
                if key in session.observations
            },
            "findings": [finding for finding in result["findings"] if str(finding["finding_id"]).startswith("CERT-")],
        }
    return {"findings": result["findings"]}


def allowed_evidence(analysis: AnalysisResponse) -> set[str]:
    """Return finding IDs and safe field references the model may cite."""
    evidence = {str(finding["finding_id"]) for finding in _findings(analysis)}
    evidence.update({"risk.class", "risk.score", "action"})
    return evidence

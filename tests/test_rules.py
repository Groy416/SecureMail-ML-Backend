from __future__ import annotations

from ml.dataset import CATALOG_BY_ID, _build_record
from ml.fusion import highest_rule_severity
from ml.rules import extract_rule_findings
from ml.schema import (
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


def _record(**feature_overrides: object) -> SessionFeatureRecord:
    features = dict(
        protocol=Protocol.SMTP,
        src_port=50000,
        dst_port=25,
        starttls_advertised=True,
        starttls_used=True,
        handshake_success=True,
        handshake_failures=0,
        renegotiation_count=0,
        session_duration_seconds=1.0,
        packet_count=10,
        byte_count=1000,
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
        cert_expires_in_days=200.0,
        cert_chain_valid=True,
        hostname_mismatch=False,
        cert_key_algorithm="RSA",
        cert_key_length_bits=2048,
        cert_signature_algorithm="SHA256-RSA",
    )
    features.update(feature_overrides)
    return SessionFeatureRecord(
        schema_version="session-features.v1",
        provenance=Provenance(
            capture_id="cap",
            flow_id="flow",
            session_id="sess",
            source_type=SourceType.SYNTHETIC_FEATURE,
            scenario_id="fixture",
            environment_id="lab_train",
            generator_seed=7,
            evidence_refs=[
                EvidenceReference(source=EvidenceSource.SCENARIO, fields=["cert_chain_valid"])
            ],
        ),
        features=SessionFeatures.model_validate(features),
        labels=SessionLabels(risk_label=RiskLabel.INFORMATIONAL, anomaly_label=0),
    )


def test_starttls_without_an_explicit_failure_is_not_a_handshake_failure() -> None:
    findings = extract_rule_findings(
        _record(
            handshake_success=False,
            handshake_failures=0,
            tls_version=None,
            cipher_suite=None,
            cipher_family=None,
            key_exchange=None,
            signature_algorithm=None,
            forward_secrecy=None,
        )
    )

    assert "STLS-002" not in {finding.finding_id for finding in findings}


def test_invalid_chain_with_repeated_failures_is_critical() -> None:
    findings = extract_rule_findings(
        _record(cert_valid=False, cert_chain_valid=False, handshake_failures=3)
    )
    by_id = {finding.finding_id: finding for finding in findings}

    assert by_id["CRIT-001"].severity is RiskLabel.CRITICAL
    assert highest_rule_severity(findings) is RiskLabel.CRITICAL


def test_invalid_chain_alone_stays_high() -> None:
    findings = extract_rule_findings(_record(cert_valid=False, cert_chain_valid=False))
    by_id = {finding.finding_id: finding for finding in findings}

    assert "CRIT-001" not in by_id
    assert by_id["CERT-003"].severity is RiskLabel.HIGH
    assert highest_rule_severity(findings) is RiskLabel.HIGH


def test_repeated_failures_alone_stay_high() -> None:
    findings = extract_rule_findings(
        _record(
            handshake_success=False,
            handshake_failures=4,
            tls_version=None,
            cipher_suite=None,
            cipher_family=None,
            key_exchange=None,
            signature_algorithm=None,
            forward_secrecy=None,
            cert_present=False,
            cert_valid=None,
            cert_expired=None,
            cert_expires_in_days=None,
            cert_chain_valid=None,
            hostname_mismatch=None,
            cert_key_algorithm=None,
            cert_key_length_bits=None,
            cert_signature_algorithm=None,
        )
    )
    by_id = {finding.finding_id: finding for finding in findings}

    assert "CRIT-001" not in by_id
    assert by_id["ANOM-001"].severity is RiskLabel.HIGH
    assert highest_rule_severity(findings) is RiskLabel.HIGH


def test_catalog_combo_scenario_expects_the_critical_rule() -> None:
    record = _build_record(
        item=CATALOG_BY_ID["invalid_chain_repeated_failures"],
        master_seed=7,
        environment_id="lab_train",
        repetition_index=0,
    )
    finding_ids = [finding.finding_id for finding in extract_rule_findings(record)]

    assert record.labels.risk_label is RiskLabel.CRITICAL
    assert "CRIT-001" in record.labels.expected_finding_ids
    assert "CRIT-001" in finding_ids
    assert highest_rule_severity(extract_rule_findings(record)) is RiskLabel.CRITICAL

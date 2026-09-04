from __future__ import annotations

from collections.abc import Iterable

from ml.fusion import RuleFinding
from ml.schema import RiskLabel, SessionFeatureRecord


def _evidence(
    record: SessionFeatureRecord,
    fields: Iterable[str],
) -> list[str]:
    required = set(fields)
    refs = [
        f"{reference.source.value}:{','.join(reference.fields)}"
        for reference in record.provenance.evidence_refs
        if required & set(reference.fields)
    ]
    if refs:
        return refs
    return [
        f"scenario:{record.provenance.scenario_id}:{','.join(sorted(required))}"
    ]


def _finding(
    record: SessionFeatureRecord,
    finding_id: str,
    severity: RiskLabel,
    title: str,
    fields: tuple[str, ...],
) -> RuleFinding:
    return RuleFinding(
        finding_id=finding_id,
        severity=severity,
        title=title,
        evidence_refs=_evidence(record, fields),
    )


def extract_rule_findings(record: SessionFeatureRecord) -> list[RuleFinding]:
    features = record.features
    findings: list[RuleFinding] = []

    if features.tls_version in {"TLS1.0", "TLS1.1"}:
        findings.append(
            _finding(
                record,
                "TLS-001",
                RiskLabel.CRITICAL,
                "Deprecated TLS version",
                ("tls_version",),
            )
        )
    if features.cipher_family in {"3DES", "RC4"}:
        findings.append(
            _finding(
                record,
                "TLS-002",
                RiskLabel.CRITICAL,
                "Weak cipher suite",
                ("cipher_suite", "cipher_family"),
            )
        )
    if features.handshake_success and features.forward_secrecy is False:
        findings.append(
            _finding(
                record,
                "FS-001",
                RiskLabel.HIGH,
                "Forward secrecy unavailable",
                ("key_exchange", "forward_secrecy"),
            )
        )
    if features.cert_expired is True:
        findings.append(
            _finding(
                record,
                "CERT-001",
                RiskLabel.HIGH,
                "Expired certificate",
                ("cert_expired", "cert_expires_in_days"),
            )
        )
    if (
        features.cert_key_algorithm == "RSA"
        and features.cert_key_length_bits is not None
        and features.cert_key_length_bits < 2048
    ):
        findings.append(
            _finding(
                record,
                "CERT-002",
                RiskLabel.HIGH,
                "Weak RSA certificate key",
                ("cert_key_algorithm", "cert_key_length_bits"),
            )
        )
    if features.cert_chain_valid is False:
        findings.append(
            _finding(
                record,
                "CERT-003",
                RiskLabel.HIGH,
                "Invalid certificate chain",
                ("cert_chain_valid",),
            )
        )
    if features.hostname_mismatch is True:
        findings.append(
            _finding(
                record,
                "CERT-004",
                RiskLabel.HIGH,
                "Certificate hostname mismatch",
                ("hostname_mismatch",),
            )
        )
    if features.starttls_advertised and not features.starttls_used:
        findings.append(
            _finding(
                record,
                "STLS-001",
                RiskLabel.HIGH,
                "STARTTLS advertised but unused",
                ("starttls_advertised", "starttls_used"),
            )
        )
    if features.starttls_used and not features.handshake_success:
        findings.append(
            _finding(
                record,
                "STLS-002",
                RiskLabel.HIGH,
                "STARTTLS handshake failure",
                ("starttls_used", "handshake_success"),
            )
        )
    if features.handshake_failures >= 3:
        findings.append(
            _finding(
                record,
                "ANOM-001",
                RiskLabel.HIGH,
                "Repeated TLS handshake failures",
                ("handshake_failures",),
            )
        )
    if features.cipher_family in {"RC4", "CHACHA20-POLY1305"}:
        findings.append(
            _finding(
                record,
                "ANOM-002",
                RiskLabel.MEDIUM,
                "Unusual cipher negotiation",
                ("cipher_suite", "cipher_family"),
            )
        )
    if features.tls_version == "TLS1.1":
        findings.append(
            _finding(
                record,
                "ANOM-003",
                RiskLabel.MEDIUM,
                "Unexpected TLS version",
                ("tls_version",),
            )
        )
    if features.renegotiation_count >= 2:
        findings.append(
            _finding(
                record,
                "ANOM-004",
                RiskLabel.HIGH,
                "Multiple TLS renegotiations",
                ("renegotiation_count",),
            )
        )
    return findings


if __name__ == "__main__":
    from ml.dataset import generate_feature_dataset, split_dataset

    split = split_dataset(
        generate_feature_dataset({"master_seed": 420042, "session_count": 500}),
        {"random_seed": 420042},
    )
    weak = next(
        record
        for record in split.train
        if record.provenance.scenario_id == "deprecated_tls"
    )
    assert [finding.finding_id for finding in extract_rule_findings(weak)] == [
        "TLS-001",
        "FS-001",
    ]
    print(weak.provenance.session_id)

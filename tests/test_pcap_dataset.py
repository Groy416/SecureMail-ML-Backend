from __future__ import annotations

import hashlib

import pytest

from ml.dataset import (
    SplitArtifact,
    assemble_pcap_dataset,
    records_hash,
    split_dataset,
    validate_dataset_run,
    write_dataset_run,
)
from ml.models import train_model_bundle
from ml.schema import (
    AnomalyLabel,
    EvidenceReference,
    EvidenceSource,
    Protocol,
    Provenance,
    RiskLabel,
    ScenarioManifest,
    SessionFeatureRecord,
    SessionFeatures,
    SessionLabels,
    SourceType,
)


def _manifest(scenario_id: str, label: RiskLabel) -> ScenarioManifest:
    return ScenarioManifest.model_validate(
        {
            "schema_version": "scenario.v1",
            "scenario_id": scenario_id,
            "family": "normal_baseline" if label is RiskLabel.INFORMATIONAL else "cryptographic_weakness",
            "description": scenario_id,
            "protocol": "SMTP",
            "risk_label": label.value,
            "anomaly_label": int(label is not RiskLabel.INFORMATIONAL),
            "tls": {
                "version": "TLS1.3",
                "cipher_suite": "TLS_AES_256_GCM_SHA384",
                "key_exchange": "ECDHE",
                "forward_secrecy": True,
                "signature_algorithm": "RSA-PSS",
            },
            "certificate": {"state": "valid", "key_algorithm": "RSA", "key_length_bits": 2048},
            "client_behavior": {
                "starttls_advertised": True,
                "starttls_used": True,
                "handshake_attempts": 1,
            },
            "variation": {
                "duration_seconds": [1.0, 2.0],
                "packet_count": [10, 20],
                "byte_count": [1000, 2000],
            },
            "repetitions": 1,
            "seed": 7,
        }
    )


def _record(manifest: ScenarioManifest, environment_id: str) -> SessionFeatureRecord:
    capture_id = f"capture-{manifest.scenario_id}"
    return SessionFeatureRecord(
        schema_version="session-features.v1",
        provenance=Provenance(
            capture_id=capture_id,
            flow_id=f"{capture_id}:flow",
            session_id=f"{capture_id}:session",
            source_type=SourceType.SYNTHETIC_PCAP,
            scenario_id=manifest.scenario_id,
            environment_id=environment_id,
            parameter_hash="a" * 64,
            generator_seed=7,
            evidence_refs=[
                EvidenceReference(
                    source=EvidenceSource.PCAP,
                    stream_id=1,
                    packet_start=1,
                    packet_end=10,
                    fields=["tcp.stream", "tls.handshake.type"],
                )
            ],
        ),
        features=SessionFeatures(
            protocol=Protocol.SMTP,
            src_port=50000,
            dst_port=587,
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
            cert_expires_in_days=1.0,
            cert_chain_valid=True,
            hostname_mismatch=False,
            cert_key_algorithm="RSA",
            cert_key_length_bits=2048,
            cert_signature_algorithm="SHA256-RSA",
        ),
        labels=SessionLabels(
            risk_label=manifest.risk_label,
            anomaly_label=manifest.anomaly_label,
        ),
    )


def test_synthetic_pcap_provenance_requires_a_parameter_hash() -> None:
    with pytest.raises(ValueError, match="parameter_hash"):
        Provenance(
            capture_id="capture",
            flow_id="flow",
            session_id="session",
            source_type=SourceType.SYNTHETIC_PCAP,
            scenario_id="scenario",
            environment_id="environment",
            generator_seed=7,
            evidence_refs=[
                EvidenceReference(
                    source=EvidenceSource.PCAP,
                    stream_id=1,
                    packet_start=1,
                    packet_end=1,
                    fields=["tcp.stream"],
                )
            ],
        )


def test_pcap_dataset_persists_capture_hashes_and_scenario_manifests(tmp_path) -> None:
    labels = tuple(RiskLabel)
    environments = ("lab_train", "lab_calibration", "lab_test")
    manifests = [
        _manifest(f"{environment}-{label.value}", label)
        for environment in environments
        for label in labels
    ]
    records = [
        _record(manifest, environment)
        for environment in environments
        for manifest in manifests
        if manifest.scenario_id.startswith(environment)
    ]
    capture_hashes = {
        record.provenance.capture_id: hashlib.sha256(
            record.provenance.capture_id.encode()
        ).hexdigest()
        for record in records
    }

    dataset = assemble_pcap_dataset(
        {
            "mode": "synthetic_pcap",
            "master_seed": 7,
            "session_count": len(records),
            "environment_ids": environments,
            "calibration_environment_id": "lab_calibration",
            "evaluation_environment_id": "lab_test",
        },
        records,
        manifests,
        capture_hashes,
    )
    split = split_dataset(
        dataset,
        {
            "random_seed": 7,
            "validation_environment_id": "lab_calibration",
            "test_environment_id": "lab_test",
        },
    )
    run = write_dataset_run(dataset, split, tmp_path)

    assert dataset.mode == "synthetic_pcap"
    assert validate_dataset_run(run.path)["record_count"] == len(records)
    assert (run.path / "pcap_manifest.json").is_file()
    assert (run.path / "scenario_manifest.jsonl").is_file()


def test_training_rejects_an_incomplete_risk_label_set() -> None:
    manifest = _manifest("normal-only", RiskLabel.INFORMATIONAL)
    records = [_record(manifest, "lab_train")]
    split = SplitArtifact(
        train=records,
        validation=[],
        test=[],
        group_key="environment_id",
        sha256=records_hash(records),
    )

    with pytest.raises(ValueError, match="all risk labels"):
        train_model_bundle(split, {"random_seed": 7})

from __future__ import annotations

import ssl
import subprocess
import sys

import pytest

from datasets.lab.scenarios import resolve_runtime_profile
from datasets.lab.client import configure_tls_context
from datasets.lab.runner import LabRun, assemble_successful_runs, finalize_run
from datasets.lab.server import load_runtime_profile
from ml.dataset import CATALOG
from ml.schema import (
    EvidenceReference,
    EvidenceSource,
    Protocol,
    Provenance,
    ScenarioManifest,
    SessionFeatureRecord,
    SessionFeatures,
    SessionLabels,
    SourceType,
)


def test_runtime_profile_is_seeded_and_hashable() -> None:
    manifest = ScenarioManifest.model_validate(CATALOG[0]["manifest"])

    first = resolve_runtime_profile(manifest, Protocol.SMTP, "lab_seed_0001", 7, 0)
    second = resolve_runtime_profile(manifest, Protocol.SMTP, "lab_seed_0001", 7, 0)

    assert first.profile_sha256 == second.profile_sha256
    assert first.destination_port == 2525
    assert first.client_mode == "starttls"
    assert first.tls_maximum_version == "TLS1.3"
    assert first.scenario.protocol is Protocol.SMTP
    assert first.scenario.scenario_id == "normal_tls13_valid-smtp"


def test_renegotiation_profile_requires_explicit_unsupported_status() -> None:
    manifest = ScenarioManifest.model_validate(
        next(item["manifest"] for item in CATALOG if item["manifest"]["scenario_id"] == "multiple_renegotiations")
    )

    profile = resolve_runtime_profile(manifest, Protocol.SMTP, "lab_seed_0001", 7, 0)

    assert profile.requires_renegotiation is True


def test_client_tls_context_uses_the_runtime_profile() -> None:
    manifest = ScenarioManifest.model_validate(
        next(item["manifest"] for item in CATALOG if item["manifest"]["scenario_id"] == "deprecated_tls")
    )
    profile = resolve_runtime_profile(manifest, Protocol.SMTP, "lab_seed_0001", 7, 0)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)

    configure_tls_context(context, profile.model_dump(mode="json"))

    assert context.minimum_version is ssl.TLSVersion.TLSv1
    assert context.maximum_version is ssl.TLSVersion.TLSv1


def test_importing_server_does_not_start_listeners() -> None:
    completed = subprocess.run(
        [sys.executable, "-c", "import datasets.lab.server"],
        capture_output=True,
        text=True,
        timeout=2,
    )

    assert completed.returncode == 0


def test_load_runtime_profile_rejects_a_missing_profile(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_runtime_profile(tmp_path / "runtime_profile.json")


def test_finalize_run_marks_missing_pcap_unsupported(tmp_path) -> None:
    result = finalize_run(tmp_path, "unsupported_in_lab", "cipher rejected")

    assert result.status == "unsupported_in_lab"
    assert result.pcap_sha256 is None
    assert (tmp_path / "run_manifest.json").is_file()


def _record_for_profile(profile, capture_id: str) -> SessionFeatureRecord:
    return SessionFeatureRecord(
        schema_version="session-features.v1",
        provenance=Provenance(
            capture_id=capture_id,
            flow_id=f"{capture_id}:flow",
            session_id=f"{capture_id}:session",
            source_type=SourceType.SYNTHETIC_PCAP,
            scenario_id=profile.scenario.scenario_id,
            environment_id=profile.environment_id,
            generator_seed=profile.derived_seed,
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
            protocol=profile.protocol,
            src_port=50000,
            dst_port=profile.destination_port,
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
            risk_label=profile.scenario.risk_label,
            anomaly_label=profile.scenario.anomaly_label,
        ),
    )


def test_assemble_successful_runs_excludes_unsupported_runs(monkeypatch, tmp_path) -> None:
    manifests = [
        ScenarioManifest.model_validate(item["manifest"])
        for item in CATALOG[:3]
    ]
    profiles = [
        resolve_runtime_profile(manifest, Protocol.SMTP, f"lab_{index}", 7, 0)
        for index, manifest in enumerate(manifests)
    ]
    successful_runs = [
        LabRun(tmp_path / f"success-{index}", "success", f"{index:064x}", profile=profile)
        for index, profile in enumerate(profiles, start=1)
    ]
    unsupported = LabRun(tmp_path / "unsupported", "unsupported_in_lab", None, profile=profiles[0])

    monkeypatch.setattr(
        "datasets.lab.runner.extract_run",
        lambda run: [_record_for_profile(run.profile, run.path.name)],
    )
    dataset = assemble_successful_runs(
        [*successful_runs, unsupported],
        {
            "mode": "synthetic_pcap",
            "master_seed": 7,
            "session_count": 999,
            "environment_ids": tuple(profile.environment_id for profile in profiles),
            "calibration_environment_id": profiles[1].environment_id,
            "evaluation_environment_id": profiles[2].environment_id,
        },
    )

    assert dataset.mode == "synthetic_pcap"
    assert dataset.pcap_sha256 == {
        run.path.name: run.pcap_sha256 for run in successful_runs
    }

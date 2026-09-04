from __future__ import annotations

import shutil
import subprocess

import pytest

from datasets.lab.runner import extract_run, run_scenario
from datasets.lab.scenarios import resolve_runtime_profile
from ml.dataset import CATALOG
from ml.schema import Protocol, ScenarioManifest, SourceType


def _docker_available() -> bool:
    return bool(shutil.which("docker")) and subprocess.run(
        ["docker", "info"], capture_output=True
    ).returncode == 0


def test_hostname_mismatch_is_packet_backed_and_not_a_completed_handshake(tmp_path) -> None:
    if not _docker_available():
        pytest.skip("Docker daemon is unavailable")
    manifest = ScenarioManifest.model_validate(
        next(item["manifest"] for item in CATALOG if item["manifest"]["scenario_id"] == "hostname_mismatch")
    )
    profile = resolve_runtime_profile(manifest, Protocol.SMTP, "lab_seed_0001", 7, 0)

    records = extract_run(run_scenario(profile, tmp_path))

    assert len(records) == 1
    assert records[0].features.handshake_success is False
    assert records[0].features.cert_chain_valid is True
    assert records[0].features.hostname_mismatch is True
    assert records[0].features.cert_valid is False


@pytest.mark.parametrize("protocol", [Protocol.SMTP, Protocol.IMAP, Protocol.POP3])
def test_valid_profile_runs_to_packet_backed_session(tmp_path, protocol: Protocol) -> None:
    if not _docker_available():
        pytest.skip("Docker daemon is unavailable")
    manifest = ScenarioManifest.model_validate(CATALOG[0]["manifest"])
    profile = resolve_runtime_profile(manifest, protocol, "lab_seed_0001", 7, 0)

    run = run_scenario(profile, tmp_path)
    records = extract_run(run)

    assert run.status == "success"
    assert len(records) == 1
    assert records[0].features.protocol is protocol
    assert records[0].provenance.source_type is SourceType.SYNTHETIC_PCAP
    assert records[0].provenance.evidence_refs[0].packet_start is not None

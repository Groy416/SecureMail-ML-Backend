from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

from datasets.lab.scenarios import LabRuntimeProfile, resolve_runtime_profile
from ml.dataset import CATALOG, DatasetConfig, assemble_pcap_dataset
from ml.pcap import extract_sessions
from ml.schema import Protocol, ScenarioManifest, SessionFeatureRecord

LAB_ROOT = Path(__file__).parent
COMPOSE_FILE = LAB_ROOT / "compose.yaml"
PCAP_FILENAME = "capture.pcap"
RunStatus = Literal["success", "unsupported_in_lab", "failed"]


@dataclass(frozen=True)
class LabRun:
    path: Path
    status: RunStatus
    pcap_sha256: str | None
    detail: str | None = None
    profile: LabRuntimeProfile | None = None


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _image_ids() -> list[str]:
    if not shutil.which("docker"):
        return []
    completed = subprocess.run(
        ["docker", "image", "inspect", "lab-mail-lab:latest", "--format", "{{.Id}}"],
        text=True,
        capture_output=True,
    )
    return [line for line in completed.stdout.splitlines() if line]


def _load_profile(path: Path) -> LabRuntimeProfile | None:
    profile_path = path / "runtime_profile.json"
    return (
        LabRuntimeProfile.model_validate_json(profile_path.read_text())
        if profile_path.is_file()
        else None
    )


def finalize_run(
    path: str | Path,
    status: RunStatus,
    detail: str | None = None,
) -> LabRun:
    run_path = Path(path)
    run_path.mkdir(parents=True, exist_ok=True)
    profile = _load_profile(run_path)
    pcap_path = run_path / PCAP_FILENAME
    pcap_sha256 = _file_hash(pcap_path) if status == "success" and pcap_path.is_file() and pcap_path.stat().st_size else None
    if status == "success" and not pcap_sha256:
        status = "failed"
        detail = "successful run did not produce a non-empty PCAP"
    manifest = {
        "schema_version": "lab-run.v1",
        "status": status,
        "detail": detail,
        "pcap_sha256": pcap_sha256,
        "profile": profile.model_dump(mode="json") if profile else None,
        "platform": platform.platform(),
        "container_image_ids": _image_ids(),
    }
    (run_path / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return LabRun(run_path, status, pcap_sha256, detail, profile)


def _existing_run(path: Path) -> LabRun | None:
    manifest_path = path / "run_manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text())
    return LabRun(
        path,
        manifest["status"],
        manifest.get("pcap_sha256"),
        manifest.get("detail"),
        _load_profile(path),
    )


def run_scenario(profile: LabRuntimeProfile, root: str | Path = "datasets/lab/runs") -> LabRun:
    run_path = Path(root) / profile.profile_sha256[:16]
    existing = _existing_run(run_path)
    if existing:
        return existing
    run_path.mkdir(parents=True)
    (run_path / "runtime_profile.json").write_text(
        profile.model_dump_json(indent=2) + "\n"
    )
    if profile.requires_renegotiation:
        return finalize_run(
            run_path,
            "unsupported_in_lab",
            "Python ssl does not expose TLS renegotiation controls",
        )
    if not shutil.which("docker"):
        return finalize_run(run_path, "failed", "Docker CLI is unavailable")
    environment = {
        **os.environ,
        "LAB_RUN_DIR": str(run_path.resolve()),
        "PCAP_PATH": f"/captures/{PCAP_FILENAME}",
        "MAIL_HOST": profile.service,
    }
    compose = [
        "docker", "compose", "-f", str(COMPOSE_FILE), "--project-name",
        f"sml{profile.profile_sha256[:12]}",
    ]
    services = ["mail-core", "capture"] if profile.service == "mail-core" else ["legacy-lab"]
    infrastructure = subprocess.run(
        [*compose, "--profile", "legacy", "up", "--build", "--detach", "--wait", *services],
        text=True,
        capture_output=True,
        env=environment,
    )
    legacy_capture = None
    if infrastructure.returncode == 0 and profile.service == "legacy-lab":
        container_id = subprocess.run(
            [*compose, "ps", "-q", "legacy-lab"], text=True, capture_output=True, env=environment
        ).stdout.strip()
        if container_id:
            legacy_capture = subprocess.Popen(
                ["docker", "exec", container_id, "tcpdump", "-U", "-i", "eth0", "-w", f"/captures/{PCAP_FILENAME}", "tcp port 25 or tcp port 143 or tcp port 110"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
    client_service = "legacy-client" if profile.service == "legacy-lab" else "client"
    completed = (
        subprocess.run(
            [*compose, "run", "--rm", "--no-deps", client_service],
            text=True,
            capture_output=True,
            env=environment,
        )
        if infrastructure.returncode == 0
        else infrastructure
    )
    if legacy_capture is not None:
        legacy_capture.terminate()
        legacy_capture.wait(timeout=10)
    subprocess.run(
        [*compose, "down", "--remove-orphans"],
        text=True,
        capture_output=True,
        env=environment,
    )
    status_path = run_path / "runtime_status.json"
    if status_path.is_file():
        status = json.loads(status_path.read_text())
        if status.get("status") == "unsupported_in_lab":
            return finalize_run(run_path, "unsupported_in_lab", status.get("detail"))
    if completed.returncode:
        return finalize_run(run_path, "failed", (completed.stderr or completed.stdout)[-2000:])
    return finalize_run(run_path, "success")


def extract_run(run: LabRun) -> list[SessionFeatureRecord]:
    if run.status != "success":
        raise ValueError(f"cannot extract a {run.status} lab run")
    profile = run.profile or _load_profile(run.path)
    if profile is None:
        raise ValueError("successful lab run is missing its runtime profile")
    capture_path = run.path / PCAP_FILENAME
    if not capture_path.is_file():
        raise FileNotFoundError(capture_path)
    trusted_ca_path = run.path / "ca.pem"
    return extract_sessions(
        capture_path,
        scenario=profile.scenario,
        environment_id=profile.environment_id,
        generator_seed=profile.derived_seed,
        parameter_hash=profile.profile_sha256,
        destination_port=profile.destination_port,
        trusted_ca_path=trusted_ca_path if trusted_ca_path.is_file() else None,
        expected_hostname="mail-core" if trusted_ca_path.is_file() else None,
    )


def assemble_successful_runs(
    runs: Sequence[LabRun],
    config: DatasetConfig | dict[str, object],
):
    successful = [run for run in runs if run.status == "success"]
    if not successful:
        raise ValueError("at least one successful lab run is required")
    records: list[SessionFeatureRecord] = []
    manifests: dict[str, ScenarioManifest] = {}
    pcap_sha256: dict[str, str] = {}
    for run in successful:
        if run.pcap_sha256 is None:
            raise ValueError("successful lab run is missing its PCAP SHA-256")
        extracted = extract_run(run)
        if not extracted:
            raise ValueError("successful lab run produced no extracted sessions")
        profile = run.profile or _load_profile(run.path)
        if profile is None:
            raise ValueError("successful lab run is missing its runtime profile")
        for record in extracted:
            capture_id = record.provenance.capture_id
            existing_hash = pcap_sha256.setdefault(capture_id, run.pcap_sha256)
            if existing_hash != run.pcap_sha256:
                raise ValueError(f"capture hash conflict: {capture_id}")
            manifests[record.provenance.scenario_id] = profile.scenario
        records.extend(extracted)
    parsed = config if isinstance(config, DatasetConfig) else DatasetConfig.model_validate(config)
    return assemble_pcap_dataset(
        parsed.model_copy(update={"session_count": len(records)}),
        records,
        list(manifests.values()),
        pcap_sha256,
    )


def _catalog_manifest(scenario_id: str) -> ScenarioManifest:
    for item in CATALOG:
        if item["manifest"]["scenario_id"] == scenario_id:
            return ScenarioManifest.model_validate(item["manifest"])
    raise KeyError(f"unknown scenario: {scenario_id}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--protocol", choices=[protocol.value for protocol in Protocol], required=True)
    parser.add_argument("--environment", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--repetition", type=int, default=0)
    parser.add_argument("--root", default="datasets/lab/runs")
    args = parser.parse_args()
    profile = resolve_runtime_profile(
        _catalog_manifest(args.scenario),
        Protocol(args.protocol),
        args.environment,
        args.seed,
        args.repetition,
    )
    result = run_scenario(profile, args.root)
    print(json.dumps({"path": str(result.path), "status": result.status, "pcap_sha256": result.pcap_sha256}))


if __name__ == "__main__":
    main()

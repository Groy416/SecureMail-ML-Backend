from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Literal, Sequence

from datasets.lab.matrix import (
    ENVIRONMENTS,
    MATRIX_CAPTURES,
    MATRIX_SESSIONS,
    SESSIONS_PER_CAPTURE,
    build_training_matrix,
)
from datasets.lab.scenarios import LabRuntimeProfile, resolve_runtime_profile
from ml.dataset import CATALOG, DatasetArtifact, DatasetConfig, assemble_pcap_dataset
from ml.pcap import EXTRACTOR_VERSION, extract_sessions
from ml.schema import Protocol, RiskLabel, ScenarioManifest, SessionFeatureRecord

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
    images = ("lab-mail-lab:latest", "docker.io/mailserver/docker-mailserver")
    image_ids: list[str] = []
    for image in images:
        completed = subprocess.run(
            ["docker", "image", "inspect", image, "--format", "{{.Id}}"],
            text=True,
            capture_output=True,
        )
        image_ids.extend(line for line in completed.stdout.splitlines() if line)
    return sorted(set(image_ids))


def _dependency_versions() -> dict[str, str]:
    return {
        package: version(package)
        for package in ("numpy", "pandas", "pyarrow", "pydantic", "scikit-learn", "xgboost")
    }


def _source_revision() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=LAB_ROOT.parent.parent,
            check=True,
            text=True,
            capture_output=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None


def _scenario_hash(profile: LabRuntimeProfile | None) -> str | None:
    if profile is None:
        return None
    payload = json.dumps(
        profile.scenario.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _compose_down(
    compose: list[str],
    environment: dict[str, str],
) -> None:
    subprocess.run(
        [
            *compose,
            "--profile",
            "legacy",
            "down",
            "--timeout",
            "15",
            "--remove-orphans",
        ],
        text=True,
        capture_output=True,
        env=environment,
    )


def _stop_legacy_capture(
    container_id: str,
    process: subprocess.Popen | None,
) -> None:
    subprocess.run(
        ["docker", "exec", container_id, "pkill", "-TERM", "-x", "tcpdump"],
        text=True,
        capture_output=True,
    )
    if process is None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


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
        "profile_sha256": profile.profile_sha256 if profile else None,
        "parameter_hash": profile.profile_sha256 if profile else None,
        "scenario_manifest_sha256": _scenario_hash(profile),
        "extractor_version": EXTRACTOR_VERSION,
        "dependency_versions": _dependency_versions(),
        "source_revision": _source_revision(),
        "platform": platform.platform(),
        "container_image_ids": _image_ids(),
    }
    (run_path / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return LabRun(run_path, status, pcap_sha256, detail, profile)


def run_training_matrix(
    master_seed: int,
    root: str | Path = "datasets/lab/runs",
    on_run_complete: Callable[[int, int, LabRun], None] | None = None,
) -> tuple[LabRun, ...]:
    profiles = build_training_matrix(master_seed)
    if len(profiles) != MATRIX_CAPTURES:
        raise ValueError(f"training matrix requires {MATRIX_CAPTURES} profiles")
    runs: list[LabRun] = []
    for profile in profiles:
        result = run_scenario(profile, root)
        if result.status == "failed":
            result = run_scenario(profile, Path(root) / "retry-1")
        runs.append(result)
        if on_run_complete is not None:
            on_run_complete(len(runs), len(profiles), result)
    return tuple(runs)


def validate_training_matrix(
    runs: Sequence[LabRun],
    dataset: DatasetArtifact,
) -> None:
    if len(runs) != MATRIX_CAPTURES:
        raise ValueError(f"training matrix requires {MATRIX_CAPTURES} profiles")
    successful = [run for run in runs if run.status == "success"]
    if len(successful) != MATRIX_CAPTURES:
        raise ValueError(
            f"training matrix requires {MATRIX_CAPTURES} successful captures"
        )
    profiles = [run.profile for run in successful]
    if any(profile is None for profile in profiles):
        raise ValueError("successful capture is missing its runtime profile")
    if len({profile.profile_sha256 for profile in profiles if profile}) != MATRIX_CAPTURES:
        raise ValueError("training matrix profiles must be unique")
    if dataset.mode != "synthetic_pcap":
        raise ValueError("training matrix dataset must be synthetic_pcap")
    if len(dataset.records) != MATRIX_SESSIONS:
        raise ValueError(
            f"training matrix requires {MATRIX_SESSIONS} extracted records"
        )

    capture_counts = Counter(record.provenance.capture_id for record in dataset.records)
    if len(capture_counts) != MATRIX_CAPTURES or any(
        count != SESSIONS_PER_CAPTURE for count in capture_counts.values()
    ):
        raise ValueError(
            f"each capture must contain {SESSIONS_PER_CAPTURE} extracted sessions"
        )
    for environment_id in ENVIRONMENTS:
        labels = {
            record.labels.risk_label
            for record in dataset.records
            if record.provenance.environment_id == environment_id
        }
        if labels != set(RiskLabel):
            raise ValueError(
                f"{environment_id} must contain all five risk labels"
            )

    for attribute in ("environment_id", "scenario_id", "capture_id", "parameter_hash"):
        groups = [
            {
                getattr(record.provenance, attribute)
                for record in dataset.records
                if record.provenance.environment_id == environment_id
            }
            for environment_id in ENVIRONMENTS
        ]
        if any(left & right for index, left in enumerate(groups) for right in groups[index + 1:]):
            raise ValueError(f"{attribute} leakage detected across matrix environments")


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
    run_path.mkdir(parents=True, exist_ok=True)
    (run_path / "runtime_profile.json").write_text(
        profile.model_dump_json(indent=2) + "\n"
    )
    mail_config = run_path / "mail-config"
    mail_config.mkdir(exist_ok=True)
    (mail_config / "postfix-accounts.cf").touch()
    (mail_config / "dovecot-quotas.cf").touch()
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
    _compose_down(compose, environment)
    legacy_capture = None
    legacy_container_id = None
    completed = None
    try:
        infrastructure = subprocess.run(
            [
                *compose,
                "--profile",
                "legacy",
                "up",
                "--build",
                "--detach",
                "--wait",
                *services,
            ],
            text=True,
            capture_output=True,
            env=environment,
        )
        if infrastructure.returncode == 0 and profile.service == "legacy-lab":
            legacy_container_id = subprocess.run(
                [*compose, "ps", "-q", "legacy-lab"],
                text=True,
                capture_output=True,
                env=environment,
            ).stdout.strip()
            if legacy_container_id:
                legacy_capture = subprocess.Popen(
                    [
                        "docker",
                        "exec",
                        legacy_container_id,
                        "tcpdump",
                        "-U",
                        "-i",
                        "eth0",
                        "-w",
                        f"/captures/{PCAP_FILENAME}",
                        "tcp port 25 or tcp port 143 or tcp port 110",
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
        client_service = "legacy-client" if profile.service == "legacy-lab" else "client"
        completed = (
            subprocess.run(
                [*compose, "run", "--build", "--rm", "--no-deps", client_service],
                text=True,
                capture_output=True,
                env=environment,
            )
            if infrastructure.returncode == 0
            else infrastructure
        )
    finally:
        if legacy_container_id:
            _stop_legacy_capture(legacy_container_id, legacy_capture)
        _compose_down(compose, environment)

    if completed is None:
        return finalize_run(run_path, "failed", "lab run did not complete")
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
    on_capture_complete: Callable[[int, int], None] | None = None,
):
    successful = [run for run in runs if run.status == "success"]
    if not successful:
        raise ValueError("at least one successful lab run is required")
    records: list[SessionFeatureRecord] = []
    manifests: dict[str, ScenarioManifest] = {}
    pcap_sha256: dict[str, str] = {}
    for completed, run in enumerate(successful, start=1):
        if run.pcap_sha256 is None:
            raise ValueError("successful lab run is missing its PCAP SHA-256")
        extracted = extract_run(run)
        if not extracted:
            raise ValueError("successful lab run produced no extracted sessions")
        if on_capture_complete is not None:
            on_capture_complete(completed, len(successful))
        profile = run.profile or _load_profile(run.path)
        if profile is None:
            raise ValueError("successful lab run is missing its runtime profile")
        if len(extracted) != profile.connection_count:
            raise ValueError(
                f"{profile.scenario.scenario_id} expected "
                f"{profile.connection_count} extracted sessions; got {len(extracted)}"
            )
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

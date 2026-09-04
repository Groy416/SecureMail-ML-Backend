from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from datasets.lab.matrix import MATRIX_SLOTS, _PORTS
from datasets.lab.runner import extract_run, run_scenario
from datasets.lab.scenarios import resolve_runtime_profile
from ml.dataset import CATALOG_BY_ID, SplitArtifact, _build_record, records_hash
from ml.schema import Protocol, ScenarioManifest, SessionFeatureRecord

HOLDOUT_SEED = 424242
HOLDOUT_SESSIONS_PER_CAPTURE = 40
HOLDOUT_FAMILIES: tuple[str, ...] = (
    "weak_rsa_key",
    "starttls_handshake_failure",
    "repeated_handshake_failures",
    "combined_critical_weaknesses",
)
PCAP_HOLDOUT_ROOT = Path("datasets/lab/runs-pcap-holdout")
PCAP_HOLDOUT_PROFILES: tuple[tuple[str, Protocol], ...] = (
    ("normal_tls13_valid", Protocol.SMTP),
    ("hostname_mismatch", Protocol.SMTP),
    ("invalid_certificate_chain", Protocol.SMTP),
)
PCAP_SESSION_COUNT = 2


def _catalog_manifest(scenario_id: str) -> ScenarioManifest:
    return ScenarioManifest.model_validate(CATALOG_BY_ID[scenario_id]["manifest"])


def generate_heldout_family_records(master_seed: int = HOLDOUT_SEED) -> list[SessionFeatureRecord]:
    overlap = set(HOLDOUT_FAMILIES) & set(MATRIX_SLOTS)
    if overlap:
        raise ValueError(f"holdout families must not appear in the training matrix: {sorted(overlap)}")
    records: list[SessionFeatureRecord] = []
    for family_index, base_id in enumerate(HOLDOUT_FAMILIES):
        source = CATALOG_BY_ID[base_id]
        for protocol_index, protocol in enumerate(Protocol):
            scenario_id = f"{base_id}-{protocol.value.lower()}-holdout-{master_seed}"
            capture_id = f"cap-holdout-{base_id}-{protocol.value.lower()}"
            parameter_hash = hashlib.sha256(
                json.dumps(
                    {
                        "base_id": base_id,
                        "protocol": protocol.value,
                        "partition": "family_holdout",
                        "master_seed": master_seed,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest()
            manifest = dict(source["manifest"])
            manifest["scenario_id"] = scenario_id
            manifest["protocol"] = protocol.value
            features = dict(source["features"])
            features["protocol"] = protocol.value
            features["dst_port"] = _PORTS[protocol]
            item = {
                "manifest": manifest,
                "features": features,
                "evidence_fields": list(source["evidence_fields"]),
                "base_scenario_id": base_id,
            }
            for session_index in range(HOLDOUT_SESSIONS_PER_CAPTURE):
                records.append(
                    _build_record(
                        item=item,
                        master_seed=master_seed + family_index + protocol_index,
                        environment_id="lab_test",
                        repetition_index=session_index,
                        capture_id=capture_id,
                        parameter_hash=parameter_hash,
                    )
                )
    return records


def eval_split(records: list[SessionFeatureRecord]) -> SplitArtifact:
    if not records:
        raise ValueError("holdout evaluation requires extracted records")
    return SplitArtifact(
        train=[],
        validation=[],
        test=records,
        group_key="environment_id",
        sha256=records_hash(records),
    )


def docker_available() -> bool:
    return bool(shutil.which("docker")) and subprocess.run(
        ["docker", "info"], capture_output=True
    ).returncode == 0


def pcap_holdout_profiles():
    profiles = []
    for index, (scenario_id, protocol) in enumerate(PCAP_HOLDOUT_PROFILES):
        profiles.append(
            resolve_runtime_profile(
                _catalog_manifest(scenario_id),
                protocol,
                "lab_test",
                HOLDOUT_SEED,
                index,
                connection_count=PCAP_SESSION_COUNT,
                scenario_suffix=f"pcap-holdout-{index}",
            )
        )
    return tuple(profiles)


def collect_pcap_holdout_records(root: str | Path = PCAP_HOLDOUT_ROOT) -> dict[str, object]:
    runs = [run_scenario(profile, root) for profile in pcap_holdout_profiles()]
    records: list[SessionFeatureRecord] = []
    statuses = []
    for run in runs:
        statuses.append(
            {
                "path": str(run.path),
                "status": run.status,
                "detail": run.detail,
                "scenario_id": None if run.profile is None else run.profile.scenario.scenario_id,
            }
        )
        if run.status == "success":
            records.extend(extract_run(run))
    return {"records": records, "runs": statuses}

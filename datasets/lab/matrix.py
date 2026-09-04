from __future__ import annotations

import hashlib
import json
from typing import Mapping

from datasets.lab.scenarios import LabRuntimeProfile, resolve_runtime_profile
from ml.dataset import (
    CATALOG_BY_ID,
    DatasetArtifact,
    DatasetConfig,
    _build_record,
    records_hash,
)
from ml.schema import Protocol, ScenarioManifest

ENVIRONMENTS: tuple[str, ...] = ("lab_train", "lab_calibration", "lab_test")
CAPTURES_PER_ENVIRONMENT = 23
SESSIONS_PER_CAPTURE = 73
MATRIX_CAPTURES = len(ENVIRONMENTS) * CAPTURES_PER_ENVIRONMENT
MATRIX_SESSIONS = MATRIX_CAPTURES * SESSIONS_PER_CAPTURE
PROTOCOLS: tuple[Protocol, ...] = (Protocol.SMTP, Protocol.IMAP, Protocol.POP3)
_PORTS = {Protocol.SMTP: 25, Protocol.IMAP: 143, Protocol.POP3: 110}

# Eight distinct normal profiles per environment so Isolation Forest sees
# varied benign TLS/certificate/protocol traffic, then overlapping
# low/medium/high/critical families in every split.
MATRIX_SLOTS: tuple[str, ...] = (
    "normal_tls13_valid",
    "normal_tls12_valid",
    "starttls_used_successfully",
    "normal_tls13_aes128",
    "normal_tls12_aes256",
    "normal_tls13_ecdsa",
    "normal_tls12_rsa4096",
    "normal_tls12_aes_cbc",
    "certificate_expiry_advisory",
    "certificate_expiry_warning",
    "certificate_expires_soon",
    "unusual_cipher_negotiation",
    "uncommon_chacha20_negotiation",
    "uncommon_chacha20_ecdsa",
    "unusual_chacha20_rsa4096",
    "expired_certificate",
    "invalid_certificate_chain",
    "hostname_mismatch",
    "starttls_advertised_unused",
    "rsa_no_forward_secrecy",
    "deprecated_tls",
    "weak_cipher",
    "invalid_chain_repeated_failures",
)


def _catalog_manifest(scenario_id: str) -> ScenarioManifest:
    return ScenarioManifest.model_validate(CATALOG_BY_ID[scenario_id]["manifest"])


def build_training_matrix(master_seed: int) -> tuple[LabRuntimeProfile, ...]:
    if len(MATRIX_SLOTS) != CAPTURES_PER_ENVIRONMENT:
        raise ValueError("matrix slot count must equal captures per environment")
    profiles: list[LabRuntimeProfile] = []
    for environment_index, environment_id in enumerate(ENVIRONMENTS):
        for slot_index, scenario_id in enumerate(MATRIX_SLOTS):
            protocol = PROTOCOLS[(environment_index + slot_index) % len(PROTOCOLS)]
            profiles.append(
                resolve_runtime_profile(
                    _catalog_manifest(scenario_id),
                    protocol,
                    environment_id,
                    master_seed,
                    slot_index,
                    connection_count=SESSIONS_PER_CAPTURE,
                    scenario_suffix=environment_id,
                )
            )
    return tuple(profiles)


def _capture_item(
    base_id: str,
    protocol: Protocol,
    scenario_id: str,
    environment_index: int,
) -> dict[str, object]:
    source = CATALOG_BY_ID[base_id]
    manifest = dict(source["manifest"])
    manifest["scenario_id"] = scenario_id
    manifest["protocol"] = protocol.value
    features = dict(source["features"])
    features["protocol"] = protocol.value
    features["dst_port"] = _PORTS[protocol]
    if base_id.startswith("normal_") or base_id == "starttls_used_successfully":
        features["cert_expires_in_days"] = float(features["cert_expires_in_days"]) + (
            19 * environment_index
        )
    return {
        "manifest": manifest,
        "features": features,
        "evidence_fields": list(source["evidence_fields"]),
        "base_scenario_id": base_id,
    }


def generate_grouped_feature_dataset(
    master_seed: int,
    config: DatasetConfig | Mapping[str, object] | None = None,
) -> DatasetArtifact:
    parsed = (
        DatasetConfig.model_validate(config)
        if config is not None
        else DatasetConfig.model_validate(
            {
                "mode": "synthetic_feature",
                "master_seed": master_seed,
                "session_count": MATRIX_SESSIONS,
                "environment_ids": ENVIRONMENTS,
                "calibration_environment_id": "lab_calibration",
                "evaluation_environment_id": "lab_test",
            }
        )
    )
    records = []
    for environment_index, environment_id in enumerate(ENVIRONMENTS):
        for slot_index, base_id in enumerate(MATRIX_SLOTS):
            protocol = PROTOCOLS[(environment_index + slot_index) % len(PROTOCOLS)]
            scenario_id = f"{base_id}-{protocol.value.lower()}-{environment_id}"
            capture_id = f"cap-{environment_id}-{slot_index:02d}-{protocol.value.lower()}"
            parameter_hash = hashlib.sha256(
                json.dumps(
                    {
                        "base_id": base_id,
                        "protocol": protocol.value,
                        "environment_id": environment_id,
                        "slot_index": slot_index,
                        "master_seed": master_seed,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest()
            item = _capture_item(base_id, protocol, scenario_id, environment_index)
            for session_index in range(SESSIONS_PER_CAPTURE):
                records.append(
                    _build_record(
                        item=item,
                        master_seed=master_seed,
                        environment_id=environment_id,
                        repetition_index=session_index,
                        capture_id=capture_id,
                        parameter_hash=parameter_hash,
                    )
                )
    if len(records) != parsed.session_count:
        raise ValueError("grouped matrix session count does not match configuration")
    return DatasetArtifact(
        dataset_version=parsed.dataset_version,
        mode=parsed.mode,
        master_seed=parsed.master_seed,
        environment_ids=parsed.environment_ids,
        records=records,
        scenario_manifests=[
            ScenarioManifest.model_validate(item["manifest"])
            for item in [
                _capture_item(
                    base_id,
                    PROTOCOLS[(environment_index + slot_index) % len(PROTOCOLS)],
                    f"{base_id}-{PROTOCOLS[(environment_index + slot_index) % len(PROTOCOLS)].value.lower()}-{environment_id}",
                    environment_index,
                )
                for environment_index, environment_id in enumerate(ENVIRONMENTS)
                for slot_index, base_id in enumerate(MATRIX_SLOTS)
            ]
        ],
        sha256=records_hash(records),
    )

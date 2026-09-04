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
TRAIN_ROTATIONS = 3
EVAL_ROTATIONS = 3
CAL_ROTATIONS = 1
SESSIONS_PER_CAPTURE = 73
TRAIN_SEED = 420042
CALIBRATION_SEED = 420059
EVAL_SEED = 420043
PROTOCOLS: tuple[Protocol, ...] = (Protocol.SMTP, Protocol.IMAP, Protocol.POP3)
_PORTS = {Protocol.SMTP: 25, Protocol.IMAP: 143, Protocol.POP3: 110}

MATRIX_SLOTS: tuple[str, ...] = (
    "normal_tls13_valid",
    "normal_tls12_valid",
    "starttls_used_successfully",
    "starttls_used_tls12",
    "normal_tls13_aes128",
    "normal_tls12_aes256",
    "normal_tls13_ecdsa",
    "normal_tls12_ecdsa",
    "normal_tls12_rsa4096",
    "normal_tls13_rsa4096",
    "normal_tls12_aes_cbc",
    "certificate_expiry_advisory",
    "certificate_expiry_warning",
    "certificate_expires_soon",
    "unusual_cipher_negotiation",
    "uncommon_chacha20_negotiation",
    "uncommon_chacha20_ecdsa",
    "unusual_chacha20_rsa4096",
    "expired_certificate",
    "expired_certificate_tls12",
    "expired_certificate_ecdsa",
    "invalid_certificate_chain",
    "invalid_certificate_chain_tls12",
    "hostname_mismatch",
    "hostname_mismatch_tls12",
    "starttls_advertised_unused",
    "rsa_no_forward_secrecy",
    "deprecated_tls",
    "weak_cipher",
    "invalid_chain_repeated_failures",
    "weak_rsa_key",
    "weak_rsa_key_tls12",
    "starttls_handshake_failure",
    "repeated_handshake_failures",
    "combined_critical_weaknesses",
)
SLOTS_PER_ROTATION = len(MATRIX_SLOTS)


def family_from_scenario(
    scenario_id: str,
    families: tuple[str, ...] | None = None,
) -> str:
    pool = families if families is not None else MATRIX_SLOTS
    matches = [
        family
        for family in sorted(pool, key=len, reverse=True)
        if scenario_id == family or scenario_id.startswith(f"{family}-")
    ]
    if not matches:
        raise ValueError(f"scenario_id {scenario_id!r} does not match a known family")
    return matches[0]
TRAIN_CAPTURES = SLOTS_PER_ROTATION * TRAIN_ROTATIONS
EVAL_CAPTURES = SLOTS_PER_ROTATION * EVAL_ROTATIONS
CAL_CAPTURES = SLOTS_PER_ROTATION * CAL_ROTATIONS
TRAIN_SESSIONS = TRAIN_CAPTURES * SESSIONS_PER_CAPTURE
EVAL_SESSIONS = EVAL_CAPTURES * SESSIONS_PER_CAPTURE
CAL_SESSIONS = CAL_CAPTURES * SESSIONS_PER_CAPTURE
MATRIX_CAPTURES = TRAIN_CAPTURES + CAL_CAPTURES + EVAL_CAPTURES
MATRIX_SESSIONS = TRAIN_SESSIONS + CAL_SESSIONS + EVAL_SESSIONS

_PARTITIONS: tuple[tuple[str, int, int, int], ...] = (
    ("lab_train", TRAIN_SEED, TRAIN_ROTATIONS, 0),
    ("lab_calibration", CALIBRATION_SEED, CAL_ROTATIONS, 1),
    ("lab_test", EVAL_SEED, EVAL_ROTATIONS, 2),
)


def _catalog_manifest(scenario_id: str) -> ScenarioManifest:
    return ScenarioManifest.model_validate(CATALOG_BY_ID[scenario_id]["manifest"])


def _iter_partition_captures(
    environment_id: str,
    master_seed: int,
    rotations: int,
    protocol_offset: int,
):
    capture_index = 0
    for rotation in range(rotations):
        for slot_index, scenario_id in enumerate(MATRIX_SLOTS):
            protocol = PROTOCOLS[(protocol_offset + rotation + slot_index) % len(PROTOCOLS)]
            yield capture_index, slot_index, scenario_id, protocol
            capture_index += 1


def build_training_matrix(master_seed: int) -> tuple[LabRuntimeProfile, ...]:
    if len(MATRIX_SLOTS) != SLOTS_PER_ROTATION:
        raise ValueError("matrix slot count must equal slots per rotation")
    del master_seed
    profiles: list[LabRuntimeProfile] = []
    for environment_id, seed, rotations, protocol_offset in _PARTITIONS:
        for capture_index, slot_index, scenario_id, protocol in _iter_partition_captures(
            environment_id, seed, rotations, protocol_offset
        ):
            profiles.append(
                resolve_runtime_profile(
                    _catalog_manifest(scenario_id),
                    protocol,
                    environment_id,
                    seed,
                    capture_index,
                    connection_count=SESSIONS_PER_CAPTURE,
                    scenario_suffix=f"{environment_id}-{seed}-{capture_index:03d}",
                )
            )
    return tuple(profiles)


def _capture_item(
    base_id: str,
    protocol: Protocol,
    scenario_id: str,
    flavor: int,
) -> dict[str, object]:
    source = CATALOG_BY_ID[base_id]
    manifest = dict(source["manifest"])
    manifest["scenario_id"] = scenario_id
    manifest["protocol"] = protocol.value
    features = dict(source["features"])
    features["protocol"] = protocol.value
    features["dst_port"] = _PORTS[protocol]
    if base_id.startswith("normal_") or base_id in {
        "starttls_used_successfully",
        "starttls_used_tls12",
    }:
        features["cert_expires_in_days"] = float(features["cert_expires_in_days"]) + (
            19 * flavor
        )
    return {
        "manifest": manifest,
        "features": features,
        "evidence_fields": list(source["evidence_fields"]),
        "base_scenario_id": base_id,
    }


def _partition_records(
    environment_id: str,
    master_seed: int,
    rotations: int,
    protocol_offset: int,
) -> list:
    records = []
    for capture_index, slot_index, base_id, protocol in _iter_partition_captures(
        environment_id, master_seed, rotations, protocol_offset
    ):
        scenario_id = (
            f"{base_id}-{protocol.value.lower()}-{environment_id}-{master_seed}-{capture_index:03d}"
        )
        capture_id = f"cap-{environment_id}-{master_seed}-{capture_index:03d}-{protocol.value.lower()}"
        parameter_hash = hashlib.sha256(
            json.dumps(
                {
                    "base_id": base_id,
                    "protocol": protocol.value,
                    "environment_id": environment_id,
                    "capture_index": capture_index,
                    "slot_index": slot_index,
                    "master_seed": master_seed,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        item = _capture_item(base_id, protocol, scenario_id, protocol_offset)
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
    return records


def generate_grouped_feature_dataset(
    master_seed: int,
    config: DatasetConfig | Mapping[str, object] | None = None,
) -> DatasetArtifact:
    del master_seed
    parsed = (
        DatasetConfig.model_validate(config)
        if config is not None
        else DatasetConfig.model_validate(
            {
                "mode": "synthetic_feature",
                "master_seed": TRAIN_SEED,
                "session_count": MATRIX_SESSIONS,
                "environment_ids": ENVIRONMENTS,
                "calibration_environment_id": "lab_calibration",
                "evaluation_environment_id": "lab_test",
            }
        )
    )
    records = []
    manifests = []
    for environment_id, seed, rotations, protocol_offset in _PARTITIONS:
        records.extend(
            _partition_records(environment_id, seed, rotations, protocol_offset)
        )
        for capture_index, slot_index, base_id, protocol in _iter_partition_captures(
            environment_id, seed, rotations, protocol_offset
        ):
            del slot_index
            scenario_id = (
                f"{base_id}-{protocol.value.lower()}-{environment_id}-{seed}-{capture_index:03d}"
            )
            manifests.append(
                ScenarioManifest.model_validate(
                    _capture_item(base_id, protocol, scenario_id, protocol_offset)["manifest"]
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
        scenario_manifests=manifests,
        sha256=records_hash(records),
    )

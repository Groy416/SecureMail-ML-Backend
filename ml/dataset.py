from __future__ import annotations

import hashlib
import json
import platform
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from random import Random
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import Field, model_validator
from sklearn.model_selection import GroupShuffleSplit

from ml.schema import (
    ContractModel,
    ScenarioManifest,
    SessionFeatureRecord,
    validate_scenario,
    validate_session,
)

BUCKET_ORDER: tuple[str, ...] = (
    "normal",
    "single_weakness",
    "behavioral_anomaly",
    "combined",
)

DEFAULT_VARIATION: dict[str, list[float | int]] = {
    "duration_seconds": [1.0, 8.0],
    "packet_count": [80, 260],
    "byte_count": [8000, 45000],
}

_TLS13 = {
    "version": "TLS1.3",
    "cipher_suite": "TLS_AES_256_GCM_SHA384",
    "key_exchange": "ECDHE",
    "forward_secrecy": True,
    "signature_algorithm": "RSA-PSS",
}
_TLS12 = {
    "version": "TLS1.2",
    "cipher_suite": "TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256",
    "key_exchange": "ECDHE",
    "forward_secrecy": True,
    "signature_algorithm": "SHA256-RSA",
}
_CERT_OK = {
    "state": "valid",
    "key_algorithm": "RSA",
    "key_length_bits": 2048,
}
_STARTTLS_USED = {
    "starttls_advertised": True,
    "starttls_used": True,
    "handshake_attempts": 1,
}
_TLS_NULL = {
    "tls_version": None,
    "cipher_suite": None,
    "cipher_family": None,
    "key_exchange": None,
    "signature_algorithm": None,
    "forward_secrecy": None,
}
_CERT_NULL = {
    "cert_present": False,
    "cert_valid": None,
    "cert_expired": None,
    "cert_expires_in_days": None,
    "cert_chain_valid": None,
    "hostname_mismatch": None,
    "cert_key_algorithm": None,
    "cert_key_length_bits": None,
    "cert_signature_algorithm": None,
}


class DatasetDistribution(ContractModel):
    normal: float = 0.60
    single_weakness: float = 0.25
    behavioral_anomaly: float = 0.10
    combined: float = 0.05

    @model_validator(mode="after")
    def validate_weights(self) -> DatasetDistribution:
        total = (
            self.normal
            + self.single_weakness
            + self.behavioral_anomaly
            + self.combined
        )
        if abs(total - 1.0) > 1e-9:
            raise ValueError("distribution weights must sum to 1.0")
        if min(self.model_dump().values()) < 0:
            raise ValueError("distribution weights must be >= 0")
        return self


class DatasetConfig(ContractModel):
    dataset_version: Literal["session-dataset.v1"] = "session-dataset.v1"
    mode: Literal["synthetic_feature", "synthetic_pcap"] = "synthetic_feature"
    master_seed: int = Field(ge=0)
    session_count: int = Field(gt=0)
    environment_ids: tuple[str, ...] = (
        "lab_seed_0001",
        "lab_seed_0002",
        "lab_seed_0003",
        "lab_seed_0004",
        "lab_seed_0005",
        "lab_seed_0006",
    )
    evaluation_environment_id: str = "lab_seed_0004"
    calibration_environment_id: str = "lab_seed_0002"
    distribution: DatasetDistribution = Field(
        default_factory=DatasetDistribution
    )

    @model_validator(mode="after")
    def validate_environments(self) -> DatasetConfig:
        if len(set(self.environment_ids)) < 3:
            raise ValueError("at least three unique environment IDs are required")
        if self.evaluation_environment_id not in self.environment_ids:
            raise ValueError("evaluation_environment_id must be configured")
        if self.calibration_environment_id not in self.environment_ids:
            raise ValueError("calibration_environment_id must be configured")
        if self.calibration_environment_id == self.evaluation_environment_id:
            raise ValueError("calibration and evaluation environments must differ")
        return self


class DatasetArtifact(ContractModel):
    dataset_version: Literal["session-dataset.v1"]
    mode: Literal["synthetic_feature", "synthetic_pcap"]
    master_seed: int
    environment_ids: tuple[str, ...]
    records: list[SessionFeatureRecord]
    scenario_manifests: list[ScenarioManifest] = Field(default_factory=list)
    pcap_sha256: dict[str, str] = Field(default_factory=dict)
    sha256: str

    @model_validator(mode="after")
    def validate_records(self) -> DatasetArtifact:
        if self.sha256 != records_hash(self.records):
            raise ValueError("dataset sha256 does not match records")
        manifests = {manifest.scenario_id: manifest for manifest in self.scenario_manifests}
        if len(manifests) != len(self.scenario_manifests):
            raise ValueError("scenario manifests must have unique scenario IDs")
        if set(manifests) != {record.provenance.scenario_id for record in self.records}:
            raise ValueError("scenario manifests must match record scenario IDs")
        for record in self.records:
            manifest = manifests[record.provenance.scenario_id]
            if (
                record.features.protocol is not manifest.protocol
                or record.labels.risk_label is not manifest.risk_label
                or record.labels.anomaly_label is not manifest.anomaly_label
            ):
                raise ValueError("record does not match its scenario manifest")
        if self.mode == "synthetic_feature":
            if self.pcap_sha256 or any(
                record.provenance.source_type.value != "synthetic_feature"
                for record in self.records
            ):
                raise ValueError("synthetic_feature datasets cannot contain PCAP provenance")
        else:
            capture_ids = {record.provenance.capture_id for record in self.records}
            if (
                set(self.pcap_sha256) != capture_ids
                or any(
                    not re.fullmatch(r"[0-9a-f]{64}", digest)
                    for digest in self.pcap_sha256.values()
                )
                or any(
                    record.provenance.source_type.value != "synthetic_pcap"
                    for record in self.records
                )
            ):
                raise ValueError("synthetic_pcap datasets require one SHA-256 per PCAP capture")
        return self


class SplitConfig(ContractModel):
    train: float = 0.70
    validation: float = 0.15
    test: float = 0.15
    random_seed: int = Field(ge=0)
    test_environment_id: str | None = "lab_seed_0004"
    validation_environment_id: str | None = "lab_seed_0002"

    @model_validator(mode="after")
    def validate_ratios(self) -> SplitConfig:
        ratios = (self.train, self.validation, self.test)
        if min(ratios) <= 0 or abs(sum(ratios) - 1.0) > 1e-9:
            raise ValueError("split ratios must be positive and sum to 1.0")
        return self


class SplitArtifact(ContractModel):
    train: list[SessionFeatureRecord]
    validation: list[SessionFeatureRecord]
    test: list[SessionFeatureRecord]
    group_key: Literal["environment_id"]
    sha256: str


@dataclass(frozen=True)
class DatasetRun:
    path: Path
    run_id: str
    manifest: dict[str, Any]


def scenario_seed(
    master_seed: int,
    scenario_id: str,
    repetition_index: int,
) -> int:
    payload = f"{master_seed}:{scenario_id}:{repetition_index}".encode()
    return int.from_bytes(
        hashlib.sha256(payload).digest()[:8],
        "big",
    ) % (2**63)


EVALUATION_SCENARIO_IDS = frozenset(
    {
        "normal_tls13_valid",
        "certificate_expires_soon",
        "unusual_cipher_negotiation",
        "expired_certificate",
        "combined_critical_weaknesses",
    }
)
CALIBRATION_SCENARIO_IDS = frozenset(
    {
        "normal_tls12_valid",
        "certificate_expiry_warning",
        "unexpected_tls_version",
        "weak_rsa_key",
        "weak_cipher",
    }
)


def environment_for(
    master_seed: int,
    scenario_id: str,
    environment_ids: tuple[str, ...],
    evaluation_environment_id: str,
    calibration_environment_id: str,
) -> str:
    if len(environment_ids) < 3:
        raise ValueError("at least three environment IDs are required")
    if scenario_id in EVALUATION_SCENARIO_IDS:
        return evaluation_environment_id
    if scenario_id in CALIBRATION_SCENARIO_IDS:
        return calibration_environment_id
    reserved = {evaluation_environment_id, calibration_environment_id}
    environment = environment_ids[
        scenario_seed(master_seed, scenario_id, 0) % len(environment_ids)
    ]
    while environment in reserved:
        environment = environment_ids[
            (environment_ids.index(environment) + 1) % len(environment_ids)
        ]
    return environment


def records_hash(records: list[SessionFeatureRecord]) -> str:
    blob = json.dumps(
        [record.model_dump(mode="json") for record in records],
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(blob).hexdigest()


def _ok_features(**overrides: Any) -> dict[str, Any]:
    features = {
        "protocol": "SMTP",
        "dst_port": 587,
        "starttls_advertised": True,
        "starttls_used": True,
        "handshake_success": True,
        "handshake_failures": 0,
        "renegotiation_count": 0,
        "tls_version": "TLS1.3",
        "cipher_suite": "TLS_AES_256_GCM_SHA384",
        "cipher_family": "AES-GCM",
        "key_exchange": "ECDHE",
        "signature_algorithm": "RSA-PSS",
        "forward_secrecy": True,
        "cert_present": True,
        "cert_valid": True,
        "cert_expired": False,
        "cert_expires_in_days": 241.0,
        "cert_chain_valid": True,
        "hostname_mismatch": False,
        "cert_key_algorithm": "RSA",
        "cert_key_length_bits": 2048,
        "cert_signature_algorithm": "SHA256-RSA",
    }
    features.update(overrides)
    return features


def _entry(
    *,
    scenario_id: str,
    family: str,
    description: str,
    risk_label: str,
    anomaly_label: int,
    expected_finding_ids: list[str],
    tls: dict[str, Any],
    certificate: dict[str, Any] | None,
    client_behavior: dict[str, Any],
    features: dict[str, Any],
    evidence_fields: list[str],
) -> dict[str, Any]:
    return {
        "manifest": {
            "schema_version": "scenario.v1",
            "scenario_id": scenario_id,
            "family": family,
            "description": description,
            "protocol": "SMTP",
            "risk_label": risk_label,
            "anomaly_label": anomaly_label,
            "expected_finding_ids": expected_finding_ids,
            "tls": tls,
            "certificate": certificate,
            "client_behavior": client_behavior,
            "variation": DEFAULT_VARIATION,
            "repetitions": 1,
            "seed": 0,
        },
        "features": features,
        "evidence_fields": evidence_fields,
    }


CATALOG: tuple[dict[str, Any], ...] = (
    _entry(
        scenario_id="normal_tls13_valid",
        family="normal_baseline",
        description="TLS 1.3 with a valid certificate",
        risk_label="informational",
        anomaly_label=0,
        expected_finding_ids=[],
        tls=_TLS13,
        certificate=_CERT_OK,
        client_behavior=_STARTTLS_USED,
        features=_ok_features(),
        evidence_fields=["tls_version", "cert_valid"],
    ),
    _entry(
        scenario_id="normal_tls12_valid",
        family="normal_baseline",
        description="TLS 1.2 with a valid certificate",
        risk_label="informational",
        anomaly_label=0,
        expected_finding_ids=[],
        tls=_TLS12,
        certificate=_CERT_OK,
        client_behavior=_STARTTLS_USED,
        features=_ok_features(
            tls_version="TLS1.2",
            cipher_suite="TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256",
            cipher_family="AES-GCM",
            signature_algorithm="SHA256-RSA",
        ),
        evidence_fields=["tls_version", "cert_valid"],
    ),
    _entry(
        scenario_id="starttls_used_successfully",
        family="starttls",
        description="STARTTLS advertised and completed",
        risk_label="informational",
        anomaly_label=0,
        expected_finding_ids=[],
        tls=_TLS13,
        certificate=_CERT_OK,
        client_behavior=_STARTTLS_USED,
        features=_ok_features(),
        evidence_fields=["starttls_advertised", "starttls_used"],
    ),
    _entry(
        scenario_id="normal_tls13_aes128",
        family="normal_baseline",
        description="TLS 1.3 AES-128-GCM with a valid certificate",
        risk_label="informational",
        anomaly_label=0,
        expected_finding_ids=[],
        tls={
            "version": "TLS1.3",
            "cipher_suite": "TLS_AES_128_GCM_SHA256",
            "key_exchange": "ECDHE",
            "forward_secrecy": True,
            "signature_algorithm": "RSA-PSS",
        },
        certificate=_CERT_OK,
        client_behavior=_STARTTLS_USED,
        features=_ok_features(
            cipher_suite="TLS_AES_128_GCM_SHA256",
            cipher_family="AES-GCM",
        ),
        evidence_fields=["tls_version", "cipher_suite"],
    ),
    _entry(
        scenario_id="normal_tls12_aes256",
        family="normal_baseline",
        description="TLS 1.2 AES-256-GCM with a valid certificate",
        risk_label="informational",
        anomaly_label=0,
        expected_finding_ids=[],
        tls={
            "version": "TLS1.2",
            "cipher_suite": "TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384",
            "key_exchange": "ECDHE",
            "forward_secrecy": True,
            "signature_algorithm": "SHA256-RSA",
        },
        certificate=_CERT_OK,
        client_behavior=_STARTTLS_USED,
        features=_ok_features(
            tls_version="TLS1.2",
            cipher_suite="TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384",
            cipher_family="AES-GCM",
            signature_algorithm="SHA256-RSA",
        ),
        evidence_fields=["tls_version", "cipher_suite"],
    ),
    _entry(
        scenario_id="normal_tls13_ecdsa",
        family="normal_baseline",
        description="TLS 1.3 with a valid ECDSA certificate",
        risk_label="informational",
        anomaly_label=0,
        expected_finding_ids=[],
        tls=_TLS13,
        certificate={"state": "valid", "key_algorithm": "ECDSA", "key_length_bits": 256},
        client_behavior=_STARTTLS_USED,
        features=_ok_features(
            cert_key_algorithm="ECDSA",
            cert_key_length_bits=256,
            cert_signature_algorithm="SHA256-ECDSA",
            signature_algorithm="ECDSA",
        ),
        evidence_fields=["cert_key_algorithm", "tls_version"],
    ),
    _entry(
        scenario_id="normal_tls12_rsa4096",
        family="normal_baseline",
        description="TLS 1.2 with a valid 4096-bit RSA certificate",
        risk_label="informational",
        anomaly_label=0,
        expected_finding_ids=[],
        tls=_TLS12,
        certificate={"state": "valid", "key_algorithm": "RSA", "key_length_bits": 4096},
        client_behavior=_STARTTLS_USED,
        features=_ok_features(
            tls_version="TLS1.2",
            cipher_suite="TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256",
            cipher_family="AES-GCM",
            signature_algorithm="SHA256-RSA",
            cert_key_length_bits=4096,
        ),
        evidence_fields=["cert_key_length_bits", "tls_version"],
    ),
    _entry(
        scenario_id="normal_tls12_ecdsa",
        family="normal_baseline",
        description="TLS 1.2 with a valid ECDSA certificate",
        risk_label="informational",
        anomaly_label=0,
        expected_finding_ids=[],
        tls={
            "version": "TLS1.2",
            "cipher_suite": "TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256",
            "key_exchange": "ECDHE",
            "forward_secrecy": True,
            "signature_algorithm": "ECDSA",
        },
        certificate={"state": "valid", "key_algorithm": "ECDSA", "key_length_bits": 256},
        client_behavior=_STARTTLS_USED,
        features=_ok_features(
            tls_version="TLS1.2",
            cipher_suite="TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256",
            cipher_family="AES-GCM",
            signature_algorithm="ECDSA",
            cert_key_algorithm="ECDSA",
            cert_key_length_bits=256,
            cert_signature_algorithm="SHA256-ECDSA",
        ),
        evidence_fields=["cert_key_algorithm", "tls_version"],
    ),
    _entry(
        scenario_id="normal_tls13_rsa4096",
        family="normal_baseline",
        description="TLS 1.3 with a valid 4096-bit RSA certificate",
        risk_label="informational",
        anomaly_label=0,
        expected_finding_ids=[],
        tls=_TLS13,
        certificate={"state": "valid", "key_algorithm": "RSA", "key_length_bits": 4096},
        client_behavior=_STARTTLS_USED,
        features=_ok_features(cert_key_length_bits=4096),
        evidence_fields=["cert_key_length_bits", "tls_version"],
    ),
    _entry(
        scenario_id="starttls_used_tls12",
        family="starttls",
        description="STARTTLS advertised and completed over TLS 1.2",
        risk_label="informational",
        anomaly_label=0,
        expected_finding_ids=[],
        tls=_TLS12,
        certificate=_CERT_OK,
        client_behavior=_STARTTLS_USED,
        features=_ok_features(
            tls_version="TLS1.2",
            cipher_suite="TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256",
            cipher_family="AES-GCM",
            signature_algorithm="SHA256-RSA",
        ),
        evidence_fields=["starttls_advertised", "tls_version"],
    ),
    _entry(
        scenario_id="normal_tls12_aes_cbc",
        family="normal_baseline",
        description="TLS 1.2 AES-CBC with a valid certificate",
        risk_label="informational",
        anomaly_label=0,
        expected_finding_ids=[],
        tls={
            "version": "TLS1.2",
            "cipher_suite": "TLS_ECDHE_RSA_WITH_AES_128_CBC_SHA256",
            "key_exchange": "ECDHE",
            "forward_secrecy": True,
            "signature_algorithm": "SHA256-RSA",
        },
        certificate=_CERT_OK,
        client_behavior=_STARTTLS_USED,
        features=_ok_features(
            tls_version="TLS1.2",
            cipher_suite="TLS_ECDHE_RSA_WITH_AES_128_CBC_SHA256",
            cipher_family="AES-CBC",
            signature_algorithm="SHA256-RSA",
        ),
        evidence_fields=["tls_version", "cipher_family"],
    ),
    _entry(
        scenario_id="deprecated_tls",
        family="cryptographic_weakness",
        description="Deprecated TLS 1.0 without forward secrecy",
        risk_label="critical",
        anomaly_label=1,
        expected_finding_ids=["TLS-001", "FS-001"],
        tls={
            "version": "TLS1.0",
            "cipher_suite": "TLS_RSA_WITH_AES_128_CBC_SHA",
            "key_exchange": "RSA",
            "forward_secrecy": False,
            "signature_algorithm": "SHA1-RSA",
        },
        certificate=_CERT_OK,
        client_behavior=_STARTTLS_USED,
        features=_ok_features(
            tls_version="TLS1.0",
            cipher_suite="TLS_RSA_WITH_AES_128_CBC_SHA",
            cipher_family="AES-CBC",
            key_exchange="RSA",
            signature_algorithm="SHA1-RSA",
            forward_secrecy=False,
        ),
        evidence_fields=["tls_version", "forward_secrecy"],
    ),
    _entry(
        scenario_id="weak_cipher",
        family="cryptographic_weakness",
        description="Weak 3DES cipher suite",
        risk_label="critical",
        anomaly_label=1,
        expected_finding_ids=["TLS-002", "FS-001"],
        tls={
            "version": "TLS1.2",
            "cipher_suite": "TLS_RSA_WITH_3DES_EDE_CBC_SHA",
            "key_exchange": "RSA",
            "forward_secrecy": False,
            "signature_algorithm": "SHA1-RSA",
        },
        certificate=_CERT_OK,
        client_behavior=_STARTTLS_USED,
        features=_ok_features(
            tls_version="TLS1.2",
            cipher_suite="TLS_RSA_WITH_3DES_EDE_CBC_SHA",
            cipher_family="3DES",
            key_exchange="RSA",
            signature_algorithm="SHA1-RSA",
            forward_secrecy=False,
        ),
        evidence_fields=["cipher_suite", "cipher_family"],
    ),
    _entry(
        scenario_id="invalid_chain_repeated_failures",
        family="combined",
        description="Invalid certificate chain with repeated TLS handshake failures",
        risk_label="critical",
        anomaly_label=1,
        expected_finding_ids=["CRIT-001", "CERT-003", "ANOM-001"],
        tls=_TLS13,
        certificate={"state": "invalid_chain", "key_algorithm": "RSA", "key_length_bits": 2048},
        client_behavior=_STARTTLS_USED,
        features=_ok_features(
            cert_valid=False,
            cert_chain_valid=False,
            handshake_failures=3,
        ),
        evidence_fields=["cert_chain_valid", "handshake_failures"],
    ),
    _entry(
        scenario_id="rsa_no_forward_secrecy",
        family="cryptographic_weakness",
        description="RSA key exchange without forward secrecy",
        risk_label="high",
        anomaly_label=1,
        expected_finding_ids=["FS-001"],
        tls={
            "version": "TLS1.2",
            "cipher_suite": "TLS_RSA_WITH_AES_128_GCM_SHA256",
            "key_exchange": "RSA",
            "forward_secrecy": False,
            "signature_algorithm": "SHA256-RSA",
        },
        certificate=_CERT_OK,
        client_behavior=_STARTTLS_USED,
        features=_ok_features(
            tls_version="TLS1.2",
            cipher_suite="TLS_RSA_WITH_AES_128_GCM_SHA256",
            cipher_family="AES-GCM",
            key_exchange="RSA",
            signature_algorithm="SHA256-RSA",
            forward_secrecy=False,
        ),
        evidence_fields=["key_exchange", "forward_secrecy"],
    ),
    _entry(
        scenario_id="expired_certificate",
        family="certificate_posture",
        description="Expired server certificate",
        risk_label="high",
        anomaly_label=1,
        expected_finding_ids=["CERT-001"],
        tls=_TLS13,
        certificate={
            "state": "expired",
            "key_algorithm": "RSA",
            "key_length_bits": 2048,
        },
        client_behavior=_STARTTLS_USED,
        features=_ok_features(
            cert_valid=False,
            cert_expired=True,
            cert_expires_in_days=-12.0,
        ),
        evidence_fields=["cert_expired", "cert_expires_in_days"],
    ),
    _entry(
        scenario_id="expired_certificate_tls12",
        family="certificate_posture",
        description="Expired server certificate on TLS 1.2",
        risk_label="high",
        anomaly_label=1,
        expected_finding_ids=["CERT-001"],
        tls=_TLS12,
        certificate={
            "state": "expired",
            "key_algorithm": "RSA",
            "key_length_bits": 2048,
        },
        client_behavior=_STARTTLS_USED,
        features=_ok_features(
            tls_version="TLS1.2",
            cipher_suite="TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256",
            cipher_family="AES-GCM",
            signature_algorithm="SHA256-RSA",
            cert_valid=False,
            cert_expired=True,
            cert_expires_in_days=-12.0,
        ),
        evidence_fields=["cert_expired", "tls_version"],
    ),
    _entry(
        scenario_id="expired_certificate_ecdsa",
        family="certificate_posture",
        description="Expired ECDSA server certificate",
        risk_label="high",
        anomaly_label=1,
        expected_finding_ids=["CERT-001"],
        tls=_TLS13,
        certificate={
            "state": "expired",
            "key_algorithm": "ECDSA",
            "key_length_bits": 256,
        },
        client_behavior=_STARTTLS_USED,
        features=_ok_features(
            cert_valid=False,
            cert_expired=True,
            cert_expires_in_days=-18.0,
            signature_algorithm="ECDSA",
            cert_key_algorithm="ECDSA",
            cert_key_length_bits=256,
            cert_signature_algorithm="SHA256-ECDSA",
        ),
        evidence_fields=["cert_expired", "cert_key_algorithm"],
    ),
    _entry(
        scenario_id="invalid_certificate_chain",
        family="certificate_posture",
        description="Invalid certificate chain",
        risk_label="high",
        anomaly_label=1,
        expected_finding_ids=["CERT-003"],
        tls=_TLS13,
        certificate={
            "state": "invalid_chain",
            "key_algorithm": "RSA",
            "key_length_bits": 2048,
        },
        client_behavior=_STARTTLS_USED,
        features=_ok_features(
            cert_valid=False,
            cert_chain_valid=False,
        ),
        evidence_fields=["cert_chain_valid"],
    ),
    _entry(
        scenario_id="invalid_certificate_chain_tls12",
        family="certificate_posture",
        description="Invalid certificate chain on TLS 1.2",
        risk_label="high",
        anomaly_label=1,
        expected_finding_ids=["CERT-003"],
        tls=_TLS12,
        certificate={
            "state": "invalid_chain",
            "key_algorithm": "RSA",
            "key_length_bits": 2048,
        },
        client_behavior=_STARTTLS_USED,
        features=_ok_features(
            tls_version="TLS1.2",
            cipher_suite="TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256",
            cipher_family="AES-GCM",
            signature_algorithm="SHA256-RSA",
            cert_valid=False,
            cert_chain_valid=False,
        ),
        evidence_fields=["cert_chain_valid", "tls_version"],
    ),
    _entry(
        scenario_id="hostname_mismatch",
        family="certificate_posture",
        description="Certificate hostname mismatch",
        risk_label="high",
        anomaly_label=1,
        expected_finding_ids=["CERT-004"],
        tls=_TLS13,
        certificate={
            "state": "hostname_mismatch",
            "key_algorithm": "RSA",
            "key_length_bits": 2048,
        },
        client_behavior=_STARTTLS_USED,
        features=_ok_features(hostname_mismatch=True),
        evidence_fields=["hostname_mismatch"],
    ),
    _entry(
        scenario_id="hostname_mismatch_tls12",
        family="certificate_posture",
        description="Certificate hostname mismatch on TLS 1.2",
        risk_label="high",
        anomaly_label=1,
        expected_finding_ids=["CERT-004"],
        tls=_TLS12,
        certificate={
            "state": "hostname_mismatch",
            "key_algorithm": "RSA",
            "key_length_bits": 2048,
        },
        client_behavior=_STARTTLS_USED,
        features=_ok_features(
            tls_version="TLS1.2",
            cipher_suite="TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256",
            cipher_family="AES-GCM",
            signature_algorithm="SHA256-RSA",
            hostname_mismatch=True,
        ),
        evidence_fields=["hostname_mismatch", "tls_version"],
    ),
    _entry(
        scenario_id="weak_rsa_key",
        family="certificate_posture",
        description="1024-bit RSA certificate key",
        risk_label="high",
        anomaly_label=1,
        expected_finding_ids=["CERT-002"],
        tls=_TLS13,
        certificate={
            "state": "weak_key",
            "key_algorithm": "RSA",
            "key_length_bits": 1024,
        },
        client_behavior=_STARTTLS_USED,
        features=_ok_features(cert_key_length_bits=1024),
        evidence_fields=["cert_key_length_bits"],
    ),
    _entry(
        scenario_id="weak_rsa_key_tls12",
        family="certificate_posture",
        description="1024-bit RSA certificate key on TLS 1.2",
        risk_label="high",
        anomaly_label=1,
        expected_finding_ids=["CERT-002"],
        tls=_TLS12,
        certificate={
            "state": "weak_key",
            "key_algorithm": "RSA",
            "key_length_bits": 1024,
        },
        client_behavior=_STARTTLS_USED,
        features=_ok_features(
            tls_version="TLS1.2",
            cipher_suite="TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256",
            cipher_family="AES-GCM",
            signature_algorithm="SHA256-RSA",
            cert_key_length_bits=1024,
        ),
        evidence_fields=["cert_key_length_bits", "tls_version"],
    ),
    _entry(
        scenario_id="starttls_advertised_unused",
        family="starttls",
        description="STARTTLS advertised but not used",
        risk_label="high",
        anomaly_label=1,
        expected_finding_ids=["STLS-001"],
        tls=_TLS13,
        certificate=None,
        client_behavior={
            "starttls_advertised": True,
            "starttls_used": False,
            "handshake_attempts": 0,
        },
        features=_ok_features(
            starttls_used=False,
            handshake_success=False,
            **_TLS_NULL,
            **_CERT_NULL,
        ),
        evidence_fields=["starttls_advertised", "starttls_used"],
    ),
    _entry(
        scenario_id="starttls_handshake_failure",
        family="starttls",
        description="STARTTLS handshake failure",
        risk_label="high",
        anomaly_label=1,
        expected_finding_ids=["STLS-002"],
        tls=_TLS13,
        certificate=None,
        client_behavior={
            "starttls_advertised": True,
            "starttls_used": True,
            "handshake_attempts": 1,
        },
        features=_ok_features(
            handshake_success=False,
            handshake_failures=1,
            **_TLS_NULL,
            **_CERT_NULL,
        ),
        evidence_fields=["starttls_used", "handshake_success"],
    ),
    _entry(
        scenario_id="repeated_handshake_failures",
        family="behavioral_anomaly",
        description="Repeated TLS handshake failures",
        risk_label="high",
        anomaly_label=1,
        expected_finding_ids=["ANOM-001"],
        tls=_TLS13,
        certificate=None,
        client_behavior={
            "starttls_advertised": True,
            "starttls_used": True,
            "handshake_attempts": 4,
        },
        features=_ok_features(
            handshake_success=False,
            handshake_failures=4,
            **_TLS_NULL,
            **_CERT_NULL,
        ),
        evidence_fields=["handshake_failures"],
    ),
    _entry(
        scenario_id="unusual_cipher_negotiation",
        family="behavioral_anomaly",
        description="Unusual cipher negotiation",
        risk_label="medium",
        anomaly_label=1,
        expected_finding_ids=["ANOM-002"],
        tls={
            "version": "TLS1.2",
            "cipher_suite": "TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305_SHA256",
            "key_exchange": "ECDHE",
            "forward_secrecy": True,
            "signature_algorithm": "SHA256-RSA",
        },
        certificate=_CERT_OK,
        client_behavior=_STARTTLS_USED,
        features=_ok_features(
            tls_version="TLS1.2",
            cipher_suite="TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305_SHA256",
            cipher_family="CHACHA20-POLY1305",
            signature_algorithm="SHA256-RSA",
        ),
        evidence_fields=["cipher_suite", "cipher_family"],
    ),
    _entry(
        scenario_id="unexpected_tls_version",
        family="behavioral_anomaly",
        description="Unexpected TLS 1.1 version",
        risk_label="medium",
        anomaly_label=1,
        expected_finding_ids=["ANOM-003"],
        tls={
            "version": "TLS1.1",
            "cipher_suite": "TLS_ECDHE_RSA_WITH_AES_128_CBC_SHA",
            "key_exchange": "ECDHE",
            "forward_secrecy": True,
            "signature_algorithm": "SHA1-RSA",
        },
        certificate=_CERT_OK,
        client_behavior=_STARTTLS_USED,
        features=_ok_features(
            tls_version="TLS1.1",
            cipher_suite="TLS_ECDHE_RSA_WITH_AES_128_CBC_SHA",
            cipher_family="AES-CBC",
            signature_algorithm="SHA1-RSA",
        ),
        evidence_fields=["tls_version"],
    ),
    _entry(
        scenario_id="uncommon_chacha20_negotiation",
        family="behavioral_anomaly",
        description="Uncommon TLS ChaCha20 cipher negotiation",
        risk_label="medium",
        anomaly_label=1,
        expected_finding_ids=["ANOM-002"],
        tls={
            "version": "TLS1.3",
            "cipher_suite": "TLS_CHACHA20_POLY1305_SHA256",
            "key_exchange": "ECDHE",
            "forward_secrecy": True,
            "signature_algorithm": "RSA-PSS",
        },
        certificate=_CERT_OK,
        client_behavior=_STARTTLS_USED,
        features=_ok_features(
            cipher_suite="TLS_CHACHA20_POLY1305_SHA256",
            cipher_family="CHACHA20-POLY1305",
        ),
        evidence_fields=["cipher_suite", "cipher_family"],
    ),
    _entry(
        scenario_id="uncommon_chacha20_ecdsa",
        family="behavioral_anomaly",
        description="ChaCha20 negotiation with an ECDSA certificate",
        risk_label="medium",
        anomaly_label=1,
        expected_finding_ids=["ANOM-002"],
        tls={
            "version": "TLS1.3",
            "cipher_suite": "TLS_CHACHA20_POLY1305_SHA256",
            "key_exchange": "ECDHE",
            "forward_secrecy": True,
            "signature_algorithm": "ECDSA",
        },
        certificate={"state": "valid", "key_algorithm": "ECDSA", "key_length_bits": 256},
        client_behavior=_STARTTLS_USED,
        features=_ok_features(
            cipher_suite="TLS_CHACHA20_POLY1305_SHA256",
            cipher_family="CHACHA20-POLY1305",
            signature_algorithm="ECDSA",
            cert_key_algorithm="ECDSA",
            cert_key_length_bits=256,
            cert_signature_algorithm="SHA256-ECDSA",
        ),
        evidence_fields=["cipher_suite", "cert_key_algorithm"],
    ),
    _entry(
        scenario_id="unusual_chacha20_rsa4096",
        family="behavioral_anomaly",
        description="TLS 1.2 ChaCha20 with a 4096-bit RSA certificate",
        risk_label="medium",
        anomaly_label=1,
        expected_finding_ids=["ANOM-002"],
        tls={
            "version": "TLS1.2",
            "cipher_suite": "TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305_SHA256",
            "key_exchange": "ECDHE",
            "forward_secrecy": True,
            "signature_algorithm": "SHA256-RSA",
        },
        certificate={"state": "valid", "key_algorithm": "RSA", "key_length_bits": 4096},
        client_behavior=_STARTTLS_USED,
        features=_ok_features(
            tls_version="TLS1.2",
            cipher_suite="TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305_SHA256",
            cipher_family="CHACHA20-POLY1305",
            signature_algorithm="SHA256-RSA",
            cert_key_length_bits=4096,
        ),
        evidence_fields=["cipher_suite", "cert_key_length_bits"],
    ),
    _entry(
        scenario_id="multiple_renegotiations",
        family="behavioral_anomaly",
        description="Multiple TLS renegotiations",
        risk_label="high",
        anomaly_label=1,
        expected_finding_ids=["ANOM-004"],
        tls=_TLS12,
        certificate=_CERT_OK,
        client_behavior=_STARTTLS_USED,
        features=_ok_features(
            tls_version="TLS1.2",
            cipher_suite="TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256",
            cipher_family="AES-GCM",
            signature_algorithm="SHA256-RSA",
            renegotiation_count=3,
        ),
        evidence_fields=["renegotiation_count"],
    ),
    _entry(
        scenario_id="certificate_expiry_advisory",
        family="certificate_posture",
        description="Valid certificate with a low-risk future expiry advisory",
        risk_label="low",
        anomaly_label=0,
        expected_finding_ids=[],
        tls=_TLS13,
        certificate={
            "state": "expiry_advisory",
            "key_algorithm": "RSA",
            "key_length_bits": 2048,
        },
        client_behavior=_STARTTLS_USED,
        features=_ok_features(cert_expires_in_days=75.0),
        evidence_fields=["cert_expires_in_days"],
    ),
    _entry(
        scenario_id="certificate_expiry_warning",
        family="certificate_posture",
        description="Valid certificate with an early expiry warning",
        risk_label="low",
        anomaly_label=0,
        expected_finding_ids=[],
        tls=_TLS13,
        certificate={
            "state": "expiry_warning",
            "key_algorithm": "RSA",
            "key_length_bits": 2048,
        },
        client_behavior=_STARTTLS_USED,
        features=_ok_features(cert_expires_in_days=45.0),
        evidence_fields=["cert_expires_in_days"],
    ),
    _entry(
        scenario_id="certificate_expires_soon",
        family="certificate_posture",
        description="Valid certificate approaching expiry",
        risk_label="low",
        anomaly_label=0,
        expected_finding_ids=[],
        tls=_TLS13,
        certificate={
            "state": "expiring_soon",
            "key_algorithm": "RSA",
            "key_length_bits": 2048,
        },
        client_behavior=_STARTTLS_USED,
        features=_ok_features(cert_expires_in_days=14.0),
        evidence_fields=["cert_expires_in_days"],
    ),
    _entry(
        scenario_id="combined_critical_weaknesses",
        family="combined",
        description="Deprecated TLS, weak cipher, and expired certificate",
        risk_label="critical",
        anomaly_label=1,
        expected_finding_ids=["TLS-001", "TLS-002", "CERT-001"],
        tls={
            "version": "TLS1.0",
            "cipher_suite": "TLS_RSA_WITH_3DES_EDE_CBC_SHA",
            "key_exchange": "RSA",
            "forward_secrecy": False,
            "signature_algorithm": "SHA1-RSA",
        },
        certificate={
            "state": "expired",
            "key_algorithm": "RSA",
            "key_length_bits": 1024,
        },
        client_behavior=_STARTTLS_USED,
        features=_ok_features(
            tls_version="TLS1.0",
            cipher_suite="TLS_RSA_WITH_3DES_EDE_CBC_SHA",
            cipher_family="3DES",
            key_exchange="RSA",
            signature_algorithm="SHA1-RSA",
            forward_secrecy=False,
            cert_valid=False,
            cert_expired=True,
            cert_expires_in_days=-30.0,
            cert_key_length_bits=1024,
        ),
        evidence_fields=["tls_version", "cipher_suite", "cert_expired"],
    ),
    _entry(
        scenario_id="expired_hostname_mismatch",
        family="combined",
        description="Expired certificate with a hostname mismatch",
        risk_label="high",
        anomaly_label=1,
        expected_finding_ids=["CERT-001", "CERT-004"],
        tls=_TLS13,
        certificate={
            "state": "expired_hostname_mismatch",
            "key_algorithm": "RSA",
            "key_length_bits": 2048,
        },
        client_behavior=_STARTTLS_USED,
        features=_ok_features(
            cert_valid=False,
            cert_expired=True,
            cert_expires_in_days=-9.0,
            hostname_mismatch=True,
        ),
        evidence_fields=["cert_expired", "hostname_mismatch"],
    ),
    _entry(
        scenario_id="weak_rsa_no_forward_secrecy",
        family="combined",
        description="1024-bit RSA key without forward secrecy",
        risk_label="high",
        anomaly_label=1,
        expected_finding_ids=["CERT-002", "FS-001"],
        tls={
            "version": "TLS1.2",
            "cipher_suite": "TLS_RSA_WITH_AES_128_GCM_SHA256",
            "key_exchange": "RSA",
            "forward_secrecy": False,
            "signature_algorithm": "SHA256-RSA",
        },
        certificate={
            "state": "weak_key",
            "key_algorithm": "RSA",
            "key_length_bits": 1024,
        },
        client_behavior=_STARTTLS_USED,
        features=_ok_features(
            tls_version="TLS1.2",
            cipher_suite="TLS_RSA_WITH_AES_128_GCM_SHA256",
            cipher_family="AES-GCM",
            key_exchange="RSA",
            signature_algorithm="SHA256-RSA",
            forward_secrecy=False,
            cert_key_length_bits=1024,
        ),
        evidence_fields=["cert_key_length_bits", "forward_secrecy"],
    ),
    _entry(
        scenario_id="expired_invalid_chain",
        family="combined",
        description="Expired certificate with an invalid chain",
        risk_label="high",
        anomaly_label=1,
        expected_finding_ids=["CERT-001", "CERT-003"],
        tls=_TLS13,
        certificate={
            "state": "expired_invalid_chain",
            "key_algorithm": "RSA",
            "key_length_bits": 2048,
        },
        client_behavior=_STARTTLS_USED,
        features=_ok_features(
            cert_valid=False,
            cert_expired=True,
            cert_expires_in_days=-21.0,
            cert_chain_valid=False,
        ),
        evidence_fields=["cert_expired", "cert_chain_valid"],
    ),
)

CATALOG_BY_ID: dict[str, dict[str, Any]] = {
    item["manifest"]["scenario_id"]: item for item in CATALOG
}

BUCKET_SCENARIO_IDS: dict[str, tuple[str, ...]] = {
    "normal": (
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
    ),
    "single_weakness": (
        "deprecated_tls",
        "weak_cipher",
        "rsa_no_forward_secrecy",
        "expired_certificate",
        "expired_certificate_tls12",
        "expired_certificate_ecdsa",
        "invalid_certificate_chain",
        "invalid_certificate_chain_tls12",
        "hostname_mismatch",
        "hostname_mismatch_tls12",
        "weak_rsa_key",
        "weak_rsa_key_tls12",
        "starttls_advertised_unused",
        "starttls_handshake_failure",
        "certificate_expiry_advisory",
        "certificate_expiry_warning",
        "certificate_expires_soon",
    ),
    "behavioral_anomaly": (
        "repeated_handshake_failures",
        "unusual_cipher_negotiation",
        "unexpected_tls_version",
        "uncommon_chacha20_negotiation",
        "uncommon_chacha20_ecdsa",
        "unusual_chacha20_rsa4096",
        "multiple_renegotiations",
    ),
    "combined": (
        "combined_critical_weaknesses",
        "invalid_chain_repeated_failures",
        "expired_hostname_mismatch",
        "weak_rsa_no_forward_secrecy",
        "expired_invalid_chain",
    ),
}


def _validate_catalog() -> None:
    for item in CATALOG:
        validate_scenario(item["manifest"])
        if item["features"].get("cert_expired") is True and item[
            "manifest"
        ]["scenario_id"].startswith("normal_"):
            raise ValueError("normal scenarios cannot set cert_expired=true")


_validate_catalog()


def allocate_counts(
    session_count: int,
    distribution: DatasetDistribution,
) -> dict[str, int]:
    weights = [getattr(distribution, bucket) for bucket in BUCKET_ORDER]
    exact = [session_count * weight for weight in weights]
    base = [int(value) for value in exact]
    remainder = session_count - sum(base)
    ranked = sorted(
        range(len(base)),
        key=lambda index: (exact[index] - base[index], -index),
        reverse=True,
    )
    for index in ranked[:remainder]:
        base[index] += 1
    return dict(zip(BUCKET_ORDER, base, strict=True))


def _sample_range(rng: Random, bounds: list[Any]) -> Any:
    low, high = bounds
    if isinstance(low, int) and isinstance(high, int):
        return rng.randint(low, high)
    return rng.uniform(float(low), float(high))


def _build_record(
    *,
    item: dict[str, Any],
    master_seed: int,
    environment_id: str,
    repetition_index: int,
    capture_id: str | None = None,
    parameter_hash: str | None = None,
) -> SessionFeatureRecord:
    manifest = ScenarioManifest.model_validate(item["manifest"])
    base_id = str(item.get("base_scenario_id", item["manifest"]["scenario_id"]))
    derived_seed = scenario_seed(
        master_seed,
        manifest.scenario_id,
        repetition_index,
    )
    rng = Random(derived_seed)
    features = dict(item["features"])
    features["src_port"] = rng.randint(49152, 65535)
    features["session_duration_seconds"] = _sample_range(
        rng,
        manifest.variation.duration_seconds,
    )
    features["packet_count"] = _sample_range(
        rng,
        manifest.variation.packet_count,
    )
    features["byte_count"] = _sample_range(
        rng,
        manifest.variation.byte_count,
    )
    features["retransmission_count"] = rng.randint(0, 2)
    features["out_of_order_count"] = rng.randint(0, 2)
    if base_id in {
        "repeated_handshake_failures",
        "invalid_chain_repeated_failures",
    }:
        features["handshake_failures"] = rng.randint(3, 8)
    if base_id == "multiple_renegotiations":
        features["renegotiation_count"] = rng.randint(2, 6)
    if base_id == "starttls_handshake_failure":
        features["handshake_failures"] = rng.randint(1, 3)

    suffix = f"{manifest.scenario_id}_{repetition_index:06d}"
    resolved_capture = capture_id or f"cap_{suffix}"
    record = {
        "schema_version": "session-features.v1",
        "provenance": {
            "capture_id": resolved_capture,
            "flow_id": f"{resolved_capture}:flow_{repetition_index:04d}",
            "session_id": (
                f"{resolved_capture}:sess_{repetition_index:04d}"
                if capture_id
                else f"sess_{suffix}"
            ),
            "source_type": "synthetic_feature",
            "scenario_id": manifest.scenario_id,
            "environment_id": environment_id,
            "parameter_hash": parameter_hash,
            "generator_seed": derived_seed,
            "evidence_refs": [
                {
                    "source": "scenario",
                    "fields": item["evidence_fields"],
                }
            ],
        },
        "features": features,
        "labels": {
            "risk_label": manifest.risk_label.value,
            "anomaly_label": manifest.anomaly_label.value,
            "expected_finding_ids": list(manifest.expected_finding_ids),
        },
    }
    return validate_session(record)


def generate_feature_dataset(
    config: DatasetConfig | Mapping[str, Any],
) -> DatasetArtifact:
    parsed = (
        config
        if isinstance(config, DatasetConfig)
        else DatasetConfig.model_validate(config)
    )
    if parsed.mode != "synthetic_feature":
        raise ValueError("only synthetic_feature generation is implemented")

    counts = allocate_counts(parsed.session_count, parsed.distribution)
    records: list[SessionFeatureRecord] = []
    for bucket in BUCKET_ORDER:
        scenario_ids = BUCKET_SCENARIO_IDS[bucket]
        for index in range(counts[bucket]):
            scenario_id = scenario_ids[index % len(scenario_ids)]
            records.append(
                _build_record(
                    item=CATALOG_BY_ID[scenario_id],
                    master_seed=parsed.master_seed,
                    environment_id=environment_for(
                        parsed.master_seed,
                        scenario_id,
                        parsed.environment_ids,
                        parsed.evaluation_environment_id,
                        parsed.calibration_environment_id,
                    ),
                    repetition_index=index,
                )
            )

    return DatasetArtifact(
        dataset_version=parsed.dataset_version,
        mode=parsed.mode,
        master_seed=parsed.master_seed,
        environment_ids=parsed.environment_ids,
        records=records,
        scenario_manifests=[
            ScenarioManifest.model_validate(item["manifest"])
            for item in CATALOG
            if item["manifest"]["scenario_id"]
            in {record.provenance.scenario_id for record in records}
        ],
        sha256=records_hash(records),
    )


def assemble_pcap_dataset(
    config: DatasetConfig | Mapping[str, Any],
    records: Sequence[SessionFeatureRecord],
    scenario_manifests: Sequence[ScenarioManifest | Mapping[str, Any]],
    pcap_sha256: Mapping[str, str],
) -> DatasetArtifact:
    parsed = (
        config
        if isinstance(config, DatasetConfig)
        else DatasetConfig.model_validate(config)
    )
    materialized_records = list(records)
    if parsed.mode != "synthetic_pcap":
        raise ValueError("PCAP assembly requires mode='synthetic_pcap'")
    if len(materialized_records) != parsed.session_count:
        raise ValueError("session_count must equal the number of extracted PCAP records")
    if {record.provenance.environment_id for record in materialized_records} != set(
        parsed.environment_ids
    ):
        raise ValueError("PCAP records must cover exactly the configured environments")
    manifests = [
        manifest
        if isinstance(manifest, ScenarioManifest)
        else ScenarioManifest.model_validate(manifest)
        for manifest in scenario_manifests
    ]
    return DatasetArtifact(
        dataset_version=parsed.dataset_version,
        mode=parsed.mode,
        master_seed=parsed.master_seed,
        environment_ids=parsed.environment_ids,
        records=materialized_records,
        scenario_manifests=manifests,
        pcap_sha256=dict(pcap_sha256),
        sha256=records_hash(materialized_records),
    )


def split_dataset(
    dataset: DatasetArtifact,
    config: SplitConfig | Mapping[str, Any],
) -> SplitArtifact:
    parsed = (
        config
        if isinstance(config, SplitConfig)
        else SplitConfig.model_validate(config)
    )
    groups = [record.provenance.environment_id for record in dataset.records]
    if len(set(groups)) < 3:
        raise ValueError("at least three environment groups are required to split")

    indices = list(range(len(dataset.records)))
    if parsed.test_environment_id is None:
        train_validation, test = next(
            GroupShuffleSplit(
                n_splits=1,
                test_size=parsed.test,
                random_state=parsed.random_seed,
            ).split(indices, groups=groups)
        )
    else:
        test = np.asarray(
            [
                index
                for index, group in enumerate(groups)
                if group == parsed.test_environment_id
            ]
        )
        if len(test) == 0:
            raise ValueError("test_environment_id has no generated records")
        train_validation = np.asarray(
            [index for index, group in enumerate(groups) if group != parsed.test_environment_id]
        )
    if parsed.validation_environment_id is None:
        train_validation_groups = [groups[index] for index in train_validation]
        validation_ratio = parsed.validation / (parsed.train + parsed.validation)
        train_local, validation_local = next(
            GroupShuffleSplit(
                n_splits=1,
                test_size=validation_ratio,
                random_state=parsed.random_seed + 1,
            ).split(train_validation, groups=train_validation_groups)
        )
        train = [dataset.records[train_validation[index]] for index in train_local]
        validation = [
            dataset.records[train_validation[index]] for index in validation_local
        ]
    else:
        if parsed.validation_environment_id == parsed.test_environment_id:
            raise ValueError("validation and test environments must differ")
        validation = [
            dataset.records[index]
            for index in train_validation
            if groups[index] == parsed.validation_environment_id
        ]
        if not validation:
            raise ValueError("validation_environment_id has no generated records")
        train = [
            dataset.records[index]
            for index in train_validation
            if groups[index] != parsed.validation_environment_id
        ]
    test_records = [dataset.records[index] for index in test]

    for attribute in ("environment_id", "scenario_id", "capture_id"):
        split_groups = [
            {getattr(record.provenance, attribute) for record in partition}
            for partition in (train, validation, test_records)
        ]
        if any(
            left & right
            for position, left in enumerate(split_groups)
            for right in split_groups[position + 1 :]
        ):
            raise RuntimeError(f"{attribute} leakage detected across splits")
    parameter_groups = [
        {
            record.provenance.parameter_hash
            for record in partition
            if record.provenance.parameter_hash
        }
        for partition in (train, validation, test_records)
    ]
    if any(parameter_groups) and any(
        left & right
        for position, left in enumerate(parameter_groups)
        for right in parameter_groups[position + 1 :]
    ):
        raise RuntimeError("parameter_hash leakage detected across splits")

    split_payload = {
        "train": [record.provenance.session_id for record in train],
        "validation": [record.provenance.session_id for record in validation],
        "test": [record.provenance.session_id for record in test_records],
    }
    split_hash = hashlib.sha256(
        json.dumps(split_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return SplitArtifact(
        train=train,
        validation=validation,
        test=test_records,
        group_key="environment_id",
        sha256=split_hash,
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        )
    )


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    pq.write_table(pa.Table.from_pylist(rows), path)


def write_dataset_run(
    dataset: DatasetArtifact,
    split: SplitArtifact,
    root: str | Path = "datasets/runs",
    run_id: str | None = None,
) -> DatasetRun:
    resolved_run_id = run_id or f"{dataset.mode.replace('_', '-')}-{dataset.sha256[:12]}"
    run_path = Path(root) / resolved_run_id
    manifest_path = run_path / "run_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if (
            manifest.get("dataset_sha256") == dataset.sha256
            and manifest.get("split_sha256") == split.sha256
        ):
            return DatasetRun(run_path, resolved_run_id, manifest)
        raise FileExistsError(f"run already exists with different content: {run_path}")
    if run_path.exists():
        raise FileExistsError(f"run directory already exists: {run_path}")

    extracted_path = run_path / "extracted"
    splits_path = run_path / "splits"
    extracted_path.mkdir(parents=True)
    splits_path.mkdir()
    (run_path / "pcaps").mkdir()

    records = [record.model_dump(mode="json") for record in dataset.records]
    _write_jsonl(extracted_path / "session_features.jsonl", records)
    _write_parquet(extracted_path / "session_features.parquet", records)
    _write_jsonl(
        run_path / "scenario_manifest.jsonl",
        [manifest.model_dump(mode="json") for manifest in dataset.scenario_manifests],
    )
    if dataset.pcap_sha256:
        (run_path / "pcap_manifest.json").write_text(
            json.dumps(dataset.pcap_sha256, indent=2, sort_keys=True) + "\n"
        )

    for name, partition in (
        ("train", split.train),
        ("validation", split.validation),
        ("test", split.test),
    ):
        _write_parquet(
            splits_path / f"{name}.parquet",
            [record.model_dump(mode="json") for record in partition],
        )

    files = sorted(
        path for path in run_path.rglob("*") if path.is_file()
    )
    checksums = {
        str(path.relative_to(run_path)): _file_hash(path) for path in files
    }
    (run_path / "checksums.sha256").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in checksums.items())
    )
    manifest = {
        "dataset_version": dataset.dataset_version,
        "mode": dataset.mode,
        "run_id": resolved_run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "master_seed": dataset.master_seed,
        "environment_ids": list(dataset.environment_ids),
        "record_count": len(dataset.records),
        "dataset_sha256": dataset.sha256,
        "split_sha256": split.sha256,
        "split_group_key": split.group_key,
        "pcap_sha256": dataset.pcap_sha256,
        "dependency_versions": {
            package: version(package)
            for package in ("numpy", "pandas", "pyarrow", "pydantic", "scikit-learn")
        },
        "platform": platform.platform(),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return DatasetRun(run_path, resolved_run_id, manifest)


def validate_dataset_run(path: str | Path) -> dict[str, Any]:
    run_path = Path(path)
    manifest = json.loads((run_path / "run_manifest.json").read_text())
    records = [
        validate_session(json.loads(line))
        for line in (run_path / "extracted" / "session_features.jsonl")
        .read_text()
        .splitlines()
    ]
    if len(records) != manifest["record_count"]:
        raise ValueError("dataset record count does not match run manifest")
    if records_hash(records) != manifest["dataset_sha256"]:
        raise ValueError("dataset hash does not match run manifest")

    partitions: dict[str, list[SessionFeatureRecord]] = {}
    for name in ("train", "validation", "test"):
        rows = pq.read_table(run_path / "splits" / f"{name}.parquet").to_pylist()
        partitions[name] = [validate_session(row) for row in rows]

    split_payload = {
        name: [record.provenance.session_id for record in partitions[name]]
        for name in ("train", "validation", "test")
    }
    split_hash = hashlib.sha256(
        json.dumps(split_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if split_hash != manifest["split_sha256"]:
        raise ValueError("split hash does not match run manifest")

    source_ids = {record.provenance.session_id for record in records}
    split_ids = [
        record.provenance.session_id
        for partition in partitions.values()
        for record in partition
    ]
    if len(split_ids) != len(set(split_ids)) or set(split_ids) != source_ids:
        raise ValueError("saved splits do not partition the dataset exactly once")

    for attribute in ("environment_id", "scenario_id", "capture_id", "parameter_hash"):
        split_groups = [
            {
                value
                for value in (
                    getattr(record.provenance, attribute)
                    for record in partitions[name]
                )
                if value is not None
            }
            for name in ("train", "validation", "test")
        ]
        if any(
            left & right
            for position, left in enumerate(split_groups)
            for right in split_groups[position + 1 :]
        ):
            raise ValueError(f"{attribute} leakage detected across saved splits")

    checksums = (run_path / "checksums.sha256").read_text().splitlines()
    for line in checksums:
        digest, relative_path = line.split("  ", maxsplit=1)
        if _file_hash(run_path / relative_path) != digest:
            raise ValueError(f"checksum mismatch: {relative_path}")
    return manifest


def load_dataset_run(
    path: str | Path,
) -> tuple[DatasetArtifact, SplitArtifact]:
    run_path = Path(path)
    manifest = validate_dataset_run(run_path)
    records = [
        validate_session(json.loads(line))
        for line in (run_path / "extracted" / "session_features.jsonl")
        .read_text()
        .splitlines()
    ]
    scenario_manifests = [
        ScenarioManifest.model_validate(json.loads(line))
        for line in (run_path / "scenario_manifest.jsonl")
        .read_text()
        .splitlines()
    ]
    dataset = DatasetArtifact(
        dataset_version=manifest["dataset_version"],
        mode=manifest["mode"],
        master_seed=manifest["master_seed"],
        environment_ids=tuple(manifest["environment_ids"]),
        records=records,
        scenario_manifests=scenario_manifests,
        pcap_sha256=dict(manifest.get("pcap_sha256", {})),
        sha256=manifest["dataset_sha256"],
    )
    partitions = {
        name: [
            validate_session(row)
            for row in pq.read_table(
                run_path / "splits" / f"{name}.parquet"
            ).to_pylist()
        ]
        for name in ("train", "validation", "test")
    }
    split = SplitArtifact(
        train=partitions["train"],
        validation=partitions["validation"],
        test=partitions["test"],
        group_key=manifest["split_group_key"],
        sha256=manifest["split_sha256"],
    )
    return dataset, split


if __name__ == "__main__":
    first = generate_feature_dataset(
        {"master_seed": 420042, "session_count": 32}
    )
    second = generate_feature_dataset(
        {"master_seed": 420042, "session_count": 32}
    )
    first_split = split_dataset(first, {"random_seed": 420042})
    second_split = split_dataset(second, {"random_seed": 420042})
    assert first.sha256 == second.sha256
    assert first_split.sha256 == second_split.sha256
    assert len(first.records) == 32
    assert all(
        record.provenance.source_type.value == "synthetic_feature"
        for record in first.records
    )
    assert not (
        {record.provenance.scenario_id for record in first_split.train}
        & {record.provenance.scenario_id for record in first_split.validation}
        & {record.provenance.scenario_id for record in first_split.test}
    )
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as directory:
        run = write_dataset_run(first, first_split, directory)
        assert validate_dataset_run(run.path)["record_count"] == 32
    print(first.sha256, first_split.sha256)

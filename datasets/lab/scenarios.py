from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import Field, model_validator

from ml.dataset import scenario_seed
from ml.schema import ContractModel, Protocol, ScenarioManifest

_PROFILE_OVERRIDES: dict[str, dict[str, object]] = {
    "deprecated_tls": {
        "cipher_string": "AES128-SHA:@SECLEVEL=0",
    },
    "weak_cipher": {
        "tls_minimum_version": "TLS1.2",
        "tls_maximum_version": "TLS1.2",
        "cipher_string": "DES-CBC3-SHA:@SECLEVEL=0",
    },
    "rsa_no_forward_secrecy": {
        "tls_minimum_version": "TLS1.2",
        "tls_maximum_version": "TLS1.2",
        "cipher_string": "AES128-SHA:@SECLEVEL=0",
    },
    "starttls_advertised_unused": {"client_mode": "unused_starttls"},
    "starttls_handshake_failure": {"client_mode": "abort_starttls"},
    "repeated_handshake_failures": {
        "client_mode": "abort_starttls",
        "connection_count": 3,
    },
    "unusual_cipher_negotiation": {
        "tls_minimum_version": "TLS1.2",
        "tls_maximum_version": "TLS1.2",
        "cipher_string": "ECDHE-RSA-CHACHA20-POLY1305",
    },
    "uncommon_chacha20_negotiation": {
        "tls_minimum_version": "TLS1.2",
        "tls_maximum_version": "TLS1.2",
        "cipher_string": "ECDHE-RSA-CHACHA20-POLY1305",
    },
    "multiple_renegotiations": {"requires_renegotiation": True},
    "combined_critical_weaknesses": {
        "certificate_mode": "unknown_ca",
        "connection_count": 3,
    },
}

_CERTIFICATE_MODE: dict[str, str] = {
    "expired_certificate": "expired",
    "invalid_certificate_chain": "unknown_ca",
    "hostname_mismatch": "hostname_mismatch",
    "weak_rsa_key": "weak_rsa",
    "combined_critical_weaknesses": "unknown_ca",
}


class LabRuntimeProfile(ContractModel):
    schema_version: Literal["lab-runtime-profile.v1"] = "lab-runtime-profile.v1"
    scenario: ScenarioManifest
    protocol: Protocol
    environment_id: str
    derived_seed: int = Field(ge=0)
    destination_port: int
    client_mode: Literal["starttls", "unused_starttls", "abort_starttls"]
    starttls_advertised: bool
    starttls_accepted: bool
    tls_minimum_version: Literal["TLS1.0", "TLS1.1", "TLS1.2", "TLS1.3"]
    tls_maximum_version: Literal["TLS1.0", "TLS1.1", "TLS1.2", "TLS1.3"]
    cipher_string: str | None = None
    certificate_mode: Literal[
        "valid",
        "expired",
        "unknown_ca",
        "hostname_mismatch",
        "weak_rsa",
    ] = "valid"
    certificate_validity_days: int = Field(gt=0)
    connection_count: int = Field(default=1, gt=0)
    requires_renegotiation: bool = False
    command_delay_milliseconds: int = Field(ge=0, le=100)
    service: Literal["mail-core", "legacy-lab"] = "mail-core"
    profile_sha256: str

    @model_validator(mode="after")
    def validate_profile_hash(self) -> LabRuntimeProfile:
        payload = self.model_dump(mode="json", exclude={"profile_sha256"})
        expected = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if self.profile_sha256 != expected:
            raise ValueError("runtime profile SHA-256 does not match its payload")
        if self.scenario.protocol is not self.protocol:
            raise ValueError("runtime profile protocol must match scenario protocol")
        return self


def _protocol_scenario(manifest: ScenarioManifest, protocol: Protocol) -> ScenarioManifest:
    suffix = protocol.value.lower()
    return manifest.model_copy(
        update={
            "scenario_id": f"{manifest.scenario_id}-{suffix}",
            "description": f"{manifest.description} ({protocol.value} lab profile)",
            "protocol": protocol,
        }
    )


def _certificate_validity_days(scenario_id: str) -> int:
    if scenario_id == "certificate_expires_soon":
        return 7
    if scenario_id == "certificate_expiry_warning":
        return 30
    if scenario_id == "certificate_expiry_advisory":
        return 90
    return 365


def resolve_runtime_profile(
    manifest: ScenarioManifest,
    protocol: Protocol,
    environment_id: str,
    master_seed: int,
    repetition_index: int,
) -> LabRuntimeProfile:
    scenario = _protocol_scenario(manifest, protocol)
    derived_seed = scenario_seed(master_seed, scenario.scenario_id, repetition_index)
    base_id = manifest.scenario_id
    runtime: dict[str, object] = {
        "schema_version": "lab-runtime-profile.v1",
        "scenario": scenario,
        "protocol": protocol,
        "environment_id": environment_id,
        "derived_seed": derived_seed,
        "destination_port": {
            Protocol.SMTP: 25,
            Protocol.IMAP: 143,
            Protocol.POP3: 110,
        }[protocol],
        "client_mode": "starttls",
        "starttls_advertised": manifest.client_behavior.starttls_advertised,
        "starttls_accepted": True,
        "tls_minimum_version": manifest.tls.version,
        "tls_maximum_version": manifest.tls.version,
        "cipher_string": None,
        "certificate_mode": _CERTIFICATE_MODE.get(base_id, "valid"),
        "certificate_validity_days": _certificate_validity_days(base_id),
        "connection_count": 1,
        "requires_renegotiation": False,
        "command_delay_milliseconds": derived_seed % 21,
        "service": "legacy-lab" if base_id in {"weak_rsa_key", "expired_certificate"} else "mail-core",
    }
    runtime.update(_PROFILE_OVERRIDES.get(base_id, {}))
    if runtime["client_mode"] == "unused_starttls":
        runtime["starttls_accepted"] = False
    payload = dict(runtime)
    canonical_payload = {
        **payload,
        "scenario": scenario.model_dump(mode="json"),
        "protocol": protocol.value,
    }
    profile_sha256 = hashlib.sha256(
        json.dumps(canonical_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return LabRuntimeProfile.model_validate(
        {**payload, "profile_sha256": profile_sha256}
    )

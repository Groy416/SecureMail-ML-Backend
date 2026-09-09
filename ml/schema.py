from __future__ import annotations
from enum import Enum
from typing import Any, Literal, Mapping, Self, Tuple

from pydantic import (
    BaseModel,
    Field,
    ConfigDict,
    FiniteFloat,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveInt,
    StrictBool,
    StrictInt,
    StrictFloat,
    StringConstraints,
    model_validator,
)
from typing_extensions import Annotated, TypedDict, NotRequired

NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
NonEmptyString = NonEmptyStr
Port = Annotated[int, Field(ge=0, le=65535)]


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

class Protocol(str, Enum):
    SMTP = "SMTP"
    IMAP = "IMAP"
    POP3 = "POP3"

class SourceType(str, Enum):
    SYNTHETIC_FEATURE = "synthetic_feature"
    SYNTHETIC_PCAP = "synthetic_pcap"
    AUTHORIZED_CAPTURE = "authorized_capture"

class EvidenceSource(str, Enum):
    PCAP = "pcap"
    SESSION = "session"
    SCENARIO = "scenario"

class RiskLabel(str, Enum):
    INFORMATIONAL = "informational"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AnomalyLabel(int, Enum):
    NORMAL = 0
    ANOMALY = 1

class ModelAction(str, Enum):
    NO_ACTION = "no_action"
    MONITOR = "monitor"
    ANALYST_REVIEW = "analyst_review"
    PRIORITIZE = "prioritize"
    CRITICAL_REVIEW = "critical_review"

class FeatureView(str, Enum):
    PROTOCOL_SESSION = "protocol_session"
    TLS = "tls"
    CERTIFICATE = "certificate"

RISK_SEVERITY_ORDER: Tuple[RiskLabel, ...] = (
    RiskLabel.INFORMATIONAL,
    RiskLabel.LOW,
    RiskLabel.MEDIUM,
    RiskLabel.HIGH,
    RiskLabel.CRITICAL,
)

PROTOCOL_SESSION_FEATURES: tuple[str, ...] = (
   "protocol",
   "src_port",
   "dst_port",
   "starttls_advertised",
   "starttls_used",
   "handshake_success",
   "handshake_failures",
   "renegotiation_count",
   "session_duration_seconds",
   "packet_count",
   "byte_count",
   "retransmission_count",
   "out_of_order_count",
)

TLS_NEGOTIATION_FEATURES: tuple[str, ...] = (
    "tls_version",
    "cipher_suite",
    "cipher_family",
    "key_exchange",
    "signature_algorithm",
    "forward_secrecy",
)

CERTIFICATE_POSTURE_FEATURES: tuple[str, ...] = (
    "cert_present",
    "cert_valid",
    "cert_expired",
    "cert_expires_in_days",
    "cert_chain_valid",
    "hostname_mismatch",
    "cert_key_algorithm",
    "cert_key_length_bits",
    "cert_signature_algorithm",
)

FEATURE_VIEWS: dict[FeatureView, tuple[str, ...]] = {
    FeatureView.PROTOCOL_SESSION: PROTOCOL_SESSION_FEATURES,
    FeatureView.TLS: TLS_NEGOTIATION_FEATURES,
    FeatureView.CERTIFICATE: CERTIFICATE_POSTURE_FEATURES,
}

MODEL_INPUT_FEATURES: tuple[str, ...] = tuple(
    feature
    for view_features in FEATURE_VIEWS.values()
    for feature in view_features
)


class EvidenceReference(ContractModel):
    source: EvidenceSource
    stream_id: NonNegativeInt | None = None
    packet_start: NonNegativeInt | None = None
    packet_end: NonNegativeInt | None = None
    fields: list[NonEmptyString] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_packet_range(self) -> Self:
        if (self.packet_start is None) != (self.packet_end is None):
            raise ValueError(
                "packet_start and packet_end must be provided together"
            )

        if (
            self.packet_start is not None
            and self.packet_end is not None
            and self.packet_end < self.packet_start
        ):
            raise ValueError("packet_end must be >= packet_start")

        if self.source is EvidenceSource.PCAP and self.packet_start is None:
            raise ValueError("pcap evidence requires a packet range")

        return self


class Provenance(ContractModel):
    capture_id: NonEmptyString
    flow_id: NonEmptyString
    session_id: NonEmptyString
    source_type: SourceType
    scenario_id: NonEmptyString
    environment_id: NonEmptyString
    parameter_hash: NonEmptyString | None = None
    generator_seed: NonNegativeInt | None = None
    evidence_refs: list[EvidenceReference] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_generated_seed(self) -> Self:
        generated_sources = {
            SourceType.SYNTHETIC_FEATURE,
            SourceType.SYNTHETIC_PCAP,
        }
        if (
            self.source_type in generated_sources
            and self.generator_seed is None
        ):
            raise ValueError(
                "synthetic records require generator_seed"
            )

        if (
            self.source_type not in generated_sources
            and self.generator_seed is not None
        ):
            raise ValueError(
                "real records must not include generator_seed"
            )

        if (
            self.source_type is SourceType.SYNTHETIC_PCAP
            and self.parameter_hash is None
        ):
            raise ValueError("synthetic_pcap records require parameter_hash")

        if (
            self.source_type is SourceType.SYNTHETIC_FEATURE
            and not any(
                ref.source is EvidenceSource.SCENARIO
                for ref in self.evidence_refs
            )
        ):
            raise ValueError(
                "synthetic_feature records require scenario evidence"
            )

        if (
            self.source_type
            in {SourceType.SYNTHETIC_PCAP, SourceType.AUTHORIZED_CAPTURE}
            and not any(
                ref.source is EvidenceSource.PCAP
                for ref in self.evidence_refs
            )
        ):
            raise ValueError(
                "PCAP-derived records require pcap evidence"
            )

        return self


class SessionFeatures(ContractModel):
    model_config = ConfigDict(extra="allow")

    protocol: Protocol
    src_port: Port
    dst_port: Port

    starttls_advertised: StrictBool
    starttls_used: StrictBool
    handshake_success: StrictBool
    handshake_failures: NonNegativeInt
    renegotiation_count: NonNegativeInt

    session_duration_seconds: NonNegativeFloat
    packet_count: NonNegativeInt
    byte_count: NonNegativeInt
    retransmission_count: NonNegativeInt
    out_of_order_count: NonNegativeInt

    tls_version: NonEmptyString | None = None
    cipher_suite: NonEmptyString | None = None
    cipher_family: NonEmptyString | None = None
    key_exchange: NonEmptyString | None = None
    signature_algorithm: NonEmptyString | None = None
    forward_secrecy: StrictBool | None = None

    cert_present: StrictBool | None = None
    cert_valid: StrictBool | None = None
    cert_expired: StrictBool | None = None
    cert_expires_in_days: FiniteFloat | None = None
    cert_chain_valid: StrictBool | None = None
    hostname_mismatch: StrictBool | None = None
    cert_key_algorithm: NonEmptyString | None = None
    cert_key_length_bits: PositiveInt | None = None
    cert_signature_algorithm: NonEmptyString | None = None

    # Evidence-derived presentation metadata; excluded from MODEL_INPUT_FEATURES.
    tls_details: dict[str, Any] | None = None
    certificate_details: dict[str, Any] | None = None

    @model_validator(mode="after")
    def validate_tls_fields(self) -> Self:
        tls_fields = (
            self.tls_version,
            self.cipher_suite,
            self.cipher_family,
            self.key_exchange,
            self.signature_algorithm,
            self.forward_secrecy,
        )

        if self.handshake_success and any(
            value is None for value in tls_fields
        ):
            raise ValueError(
                "successful TLS handshakes require negotiated TLS fields"
            )

        if not self.handshake_success and any(
            value is not None for value in tls_fields
        ):
            raise ValueError(
                "negotiated TLS fields must be null when the handshake fails"
            )

        return self

    @model_validator(mode="after")
    def validate_certificate_fields(self) -> Self:
        certificate_metadata = (
            self.cert_valid,
            self.cert_expired,
            self.cert_expires_in_days,
            self.cert_chain_valid,
            self.hostname_mismatch,
            self.cert_key_algorithm,
            self.cert_key_length_bits,
            self.cert_signature_algorithm,
        )

        if self.cert_present is not True and any(
            value is not None for value in certificate_metadata
        ):
            raise ValueError(
                "certificate metadata requires cert_present=true"
            )

        if self.cert_expired is True and self.cert_valid is True:
            raise ValueError(
                "an expired certificate cannot be marked valid"
            )

        return self


class SessionLabels(ContractModel):
    risk_label: RiskLabel
    anomaly_label: AnomalyLabel
    expected_finding_ids: list[NonEmptyString] = Field(
        default_factory=list
    )


class SessionFeatureRecord(ContractModel):
    model_config = ConfigDict(extra="allow")

    schema_version: Literal["session-features.v1"]
    provenance: Provenance
    features: SessionFeatures
    labels: SessionLabels


def validate_session(
    record: Mapping[str, Any],
    *,
    strict: bool = False,
) -> SessionFeatureRecord:
    if strict:
        return SessionFeatureRecord.model_validate(
            record,
            extra="forbid",
        )

    return SessionFeatureRecord.model_validate(record)


class ScenarioFamily(str, Enum):
    NORMAL_BASELINE = "normal_baseline"
    CRYPTOGRAPHIC_WEAKNESS = "cryptographic_weakness"
    CERTIFICATE_POSTURE = "certificate_posture"
    STARTTLS = "starttls"
    BEHAVIORAL_ANOMALY = "behavioral_anomaly"
    COMBINED = "combined"


class ScenarioTLSConfig(ContractModel):
    version: NonEmptyString
    cipher_suite: NonEmptyString
    key_exchange: NonEmptyString
    forward_secrecy: StrictBool
    signature_algorithm: NonEmptyString | None = None


class ScenarioCertificateConfig(ContractModel):
    state: NonEmptyString
    key_algorithm: NonEmptyString | None = None
    key_length_bits: PositiveInt | None = None


class ScenarioClientBehavior(ContractModel):
    starttls_advertised: StrictBool
    starttls_used: StrictBool
    handshake_attempts: NonNegativeInt

    @model_validator(mode="after")
    def validate_handshake_attempts(self) -> Self:
        if self.starttls_used and self.handshake_attempts == 0:
            raise ValueError(
                "STARTTLS usage requires at least one handshake attempt"
            )

        if not self.starttls_used and self.handshake_attempts != 0:
            raise ValueError(
                "unused STARTTLS must have zero handshake attempts"
            )

        return self


class ScenarioVariation(ContractModel):
    duration_seconds: list[NonNegativeFloat] = Field(
        min_length=2,
        max_length=2,
    )
    packet_count: list[NonNegativeInt] = Field(
        min_length=2,
        max_length=2,
    )
    byte_count: list[NonNegativeInt] = Field(
        min_length=2,
        max_length=2,
    )

    @model_validator(mode="after")
    def validate_ranges(self) -> Self:
        ranges = (
            ("duration_seconds", self.duration_seconds),
            ("packet_count", self.packet_count),
            ("byte_count", self.byte_count),
        )

        for name, values in ranges:
            if values[0] > values[1]:
                raise ValueError(
                    f"{name} lower bound must be <= upper bound"
                )

        return self


class ScenarioManifest(ContractModel):
    schema_version: Literal["scenario.v1"]
    scenario_id: NonEmptyString
    family: ScenarioFamily
    description: NonEmptyString
    protocol: Protocol

    risk_label: RiskLabel
    anomaly_label: AnomalyLabel
    expected_finding_ids: list[NonEmptyString] = Field(
        default_factory=list
    )

    tls: ScenarioTLSConfig
    certificate: ScenarioCertificateConfig | None = None
    client_behavior: ScenarioClientBehavior
    variation: ScenarioVariation

    repetitions: PositiveInt
    seed: NonNegativeInt


def validate_scenario(
    manifest: Mapping[str, Any],
) -> ScenarioManifest:
    return ScenarioManifest.model_validate(manifest)

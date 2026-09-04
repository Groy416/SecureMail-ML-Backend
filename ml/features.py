from __future__ import annotations

from typing import Any, Mapping

from ml.schema import (
   CERTIFICATE_POSTURE_FEATURES,
   FEATURE_VIEWS,
   MODEL_INPUT_FEATURES,
   PROTOCOL_SESSION_FEATURES,
   TLS_NEGOTIATION_FEATURES,
   FeatureView,
   SessionFeatureRecord,
   SessionFeatures,
   validate_session,
)


NUMERIC_FEATURES: tuple[str, ...] = (
   "src_port",
   "dst_port",
   "handshake_failures",
   "renegotiation_count",
   "session_duration_seconds",
   "packet_count",
   "byte_count",
   "retransmission_count",
   "out_of_order_count",
   "cert_expires_in_days",
   "cert_key_length_bits",
)

BOOLEAN_FEATURES: tuple[str, ...] = (
   "starttls_advertised",
   "starttls_used",
   "handshake_success",
   "forward_secrecy",
   "cert_present",
   "cert_valid",
   "cert_expired",
   "cert_chain_valid",
   "hostname_mismatch",
)

CATEGORICAL_FEATURES: tuple[str, ...] = (
   "protocol",
   "tls_version",
   "cipher_suite",
   "cipher_family",
   "key_exchange",
   "signature_algorithm",
   "cert_key_algorithm",
   "cert_signature_algorithm",
)

NULLABLE_FEATURES: frozenset[str] = frozenset(
   TLS_NEGOTIATION_FEATURES + CERTIFICATE_POSTURE_FEATURES
)   

EXCLUDED_FROM_MODEL: frozenset[str] = frozenset(
   {
       "capture_id",
       "flow_id",
       "session_id",
       "source_type",
       "scenario_id",
       "environment_id",
       "generator_seed",
       "parameter_hash",
       "source_ip",
       "destination_ip",
       "risk_label",
       "anomaly_label",
       "expected_finding_ids",
   }
)

FEATURE_DISPLAY_NAMES: dict[str, str] = {
   "protocol": "Protocol",
   "src_port": "Source port",
   "dst_port": "Destination port",
   "starttls_advertised": "STARTTLS advertised",
   "starttls_used": "STARTTLS used",
   "handshake_success": "Handshake success",
   "handshake_failures": "Handshake failures",
   "renegotiation_count": "Renegotiation count",
   "session_duration_seconds": "Session duration",
   "packet_count": "Packet count",
   "byte_count": "Byte count",
   "retransmission_count": "Retransmissions",
   "out_of_order_count": "Out-of-order packets",
   "tls_version": "TLS version",
   "cipher_suite": "Cipher suite",
   "cipher_family": "Cipher family",
   "key_exchange": "Key exchange",
   "signature_algorithm": "Signature algorithm",
   "forward_secrecy": "Forward secrecy",
   "cert_present": "Certificate present",
   "cert_valid": "Certificate valid",
   "cert_expired": "Certificate expired",
   "cert_expires_in_days": "Certificate days to expiry",
   "cert_chain_valid": "Certificate chain valid",
   "hostname_mismatch": "Hostname mismatch",
   "cert_key_algorithm": "Certificate key algorithm",
   "cert_key_length_bits": "Certificate key length",
   "cert_signature_algorithm": "Certificate signature algorithm",
}

FEATURE_VIEW_BY_NAME: dict[str, FeatureView] = {
   feature: view
   for view, features in FEATURE_VIEWS.items()
   for feature in features
}


def feature_view_for(name: str) -> FeatureView:
   try:
       return FEATURE_VIEW_BY_NAME[name]
   except KeyError as exc:
       raise KeyError(f"unknown model feature: {name}") from exc


def extract_model_features(
   record: SessionFeatureRecord | Mapping[str, Any],
) -> dict[str, Any]:
   session = (
       record
       if isinstance(record, SessionFeatureRecord)
       else validate_session(record)
   )
   dumped = session.features.model_dump()
   return {name: dumped.get(name) for name in MODEL_INPUT_FEATURES}  



def source_feature_map() -> dict[str, dict[str, str]]:
   return {
       name: {
           "source_feature": name,
           "feature_view": feature_view_for(name).value,
           "display_name": FEATURE_DISPLAY_NAMES[name],
       }
       for name in MODEL_INPUT_FEATURES
   }


assert set(MODEL_INPUT_FEATURES) == (
   set(PROTOCOL_SESSION_FEATURES) | set(TLS_NEGOTIATION_FEATURES) | set(CERTIFICATE_POSTURE_FEATURES)
)
assert set(MODEL_INPUT_FEATURES) == (
   set(NUMERIC_FEATURES)
   | set(BOOLEAN_FEATURES)
   | set(CATEGORICAL_FEATURES)
)
assert EXCLUDED_FROM_MODEL.isdisjoint(MODEL_INPUT_FEATURES)
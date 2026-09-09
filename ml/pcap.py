from __future__ import annotations

import csv
import hashlib
import os
import re
import shutil
import ssl
import subprocess
import tempfile
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable, Mapping

from cryptography import x509
from cryptography.hazmat.primitives.asymmetric import dsa, ec, ed25519, ed448, rsa
from cryptography.x509.oid import NameOID

from ml.schema import (
    AnomalyLabel,
    EvidenceReference,
    EvidenceSource,
    Protocol,
    Provenance,
    RiskLabel,
    ScenarioManifest,
    SessionFeatureRecord,
    SessionFeatures,
    SessionLabels,
    SourceType,
)

_LAB_TSHARK_IMAGE = "lab-mail-lab:latest"
_FIELDS = (
    "frame.number",
    "frame.time_epoch",
    "tcp.stream",
    "ip.src",
    "tcp.srcport",
    "ip.dst",
    "tcp.dstport",
    "frame.len",
    "tcp.payload",
    "tcp.analysis.retransmission",
    "tcp.analysis.out_of_order",
    "tls.handshake.type",
    "tls.handshake.version",
    "tls.handshake.extensions.supported_version",
    "tls.handshake.ciphersuite",
    "tls.handshake.sig_hash_alg",
    "tls.handshake.certificate",
    "tls.record.content_type",
    "tls.alert_message.level",
)
_TLS_VERSION = {"0x0304": "TLS1.3", "0x0303": "TLS1.2", "0x0302": "TLS1.1", "0x0301": "TLS1.0"}
_CIPHER_SUITE = {
    "0x000a": "TLS_RSA_WITH_3DES_EDE_CBC_SHA",
    "0x002f": "TLS_RSA_WITH_AES_128_CBC_SHA",
    "0x0035": "TLS_RSA_WITH_AES_256_CBC_SHA",
    "0x1301": "TLS_AES_128_GCM_SHA256",
    "0x1302": "TLS_AES_256_GCM_SHA384",
    "0x1303": "TLS_CHACHA20_POLY1305_SHA256",
    "0xc02f": "TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256",
    "0xc030": "TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384",
}
_SIGNATURE_ALGORITHM = {"0x0804": "RSA-PSS", "0x0401": "SHA256-RSA", "0x0501": "SHA384-RSA"}
EXTRACTOR_VERSION = "pcap-extractor.v1"
TSHARK_TIMEOUT_SECONDS = 120
MAX_TSHARK_OUTPUT_BYTES = 64 * 1024 * 1024
_MAIL_PORTS: dict[str, tuple[Protocol, int]] = {
    "25": (Protocol.SMTP, 25),
    "465": (Protocol.SMTP, 465),
    "587": (Protocol.SMTP, 587),
    "143": (Protocol.IMAP, 143),
    "993": (Protocol.IMAP, 993),
    "110": (Protocol.POP3, 110),
    "995": (Protocol.POP3, 995),
}


class PcapExtractionUnavailable(RuntimeError):
    """Raised when the required passive parser is not available."""


def _command_output(command: list[str]) -> str:
    try:
        result = subprocess.run(
            command,
            check=True,
            text=True,
            capture_output=True,
            timeout=TSHARK_TIMEOUT_SECONDS,
        )
        if len(result.stdout.encode("utf-8")) > MAX_TSHARK_OUTPUT_BYTES:
            raise PcapExtractionUnavailable("tshark output exceeded the worker limit")
        return result.stdout
    except FileNotFoundError as exc:
        raise PcapExtractionUnavailable(f"parser command not found: {command[0]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise PcapExtractionUnavailable("tshark extraction timed out") from exc
    except subprocess.CalledProcessError as exc:
        raise PcapExtractionUnavailable(exc.stderr.strip() or "tshark extraction failed") from exc


def _tshark_command(path: Path, keylog_path: Path | None) -> list[str]:
    tshark = shutil.which("tshark")
    if tshark:
        command = [tshark, "-r", str(path)]
        if keylog_path:
            command.extend(["-o", f"tls.keylog_file:{keylog_path}"])
        return command
    if not shutil.which("docker"):
        raise PcapExtractionUnavailable("tshark or Docker with the local lab parser is required")
    image_check = subprocess.run(
        ["docker", "image", "inspect", _LAB_TSHARK_IMAGE],
        text=True,
        capture_output=True,
    )
    if image_check.returncode:
        raise PcapExtractionUnavailable(
            "tshark is unavailable and local lab parser image is not built"
        )
    if keylog_path and keylog_path.parent != path.parent:
        raise PcapExtractionUnavailable(
            "Docker parser requires the PCAP and lab key log in the same directory"
        )
    command = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{path.parent.resolve()}:/captures:ro",
        _LAB_TSHARK_IMAGE,
        "tshark",
        "-r",
        f"/captures/{path.name}",
    ]
    if keylog_path:
        command.extend(["-o", f"tls.keylog_file:/captures/{keylog_path.name}"])
    return command


def _load_packets(
    path: Path,
    *,
    decode_port: int,
    protocol: Protocol,
    keylog_path: Path | None,
) -> list[dict[str, str]]:
    command = _tshark_command(path, keylog_path)
    dissector = "smtp" if protocol is Protocol.SMTP else "imap" if protocol is Protocol.IMAP else "pop"
    command.extend(
        [
            "-d",
            f"tcp.port=={decode_port},{dissector}",
            "-T",
            "fields",
            "-E",
            "header=y",
            "-E",
            "separator=\t",
        ]
    )
    for field in _FIELDS:
        command.extend(["-e", field])
    return list(csv.DictReader(_command_output(command).splitlines(), delimiter="\t"))


def _first(values: Iterable[str]) -> str | None:
    return next((value for value in values if value), None)


def _has_handshake_type(rows: Iterable[dict[str, str]], handshake_type: str) -> bool:
    return any(
        handshake_type in row.get("tls.handshake.type", "").split(",")
        for row in rows
    )


def client_handshake_succeeded(
    rows: Iterable[dict[str, str]],
    client_source_port: str,
) -> bool:
    materialized = list(rows)
    return _has_handshake_type(materialized, "2") and any(
        row.get("tcp.srcport") == client_source_port
        and "20" in row.get("tls.handshake.type", "").split(",")
        for row in materialized
    )


def classify_handshake(
    rows: Iterable[dict[str, str]],
    client_source_port: str,
) -> tuple[bool, int]:
    """Classify TLS without treating encrypted Finished messages as failures."""
    materialized = list(rows)
    if client_handshake_succeeded(materialized, client_source_port):
        return True, 0

    has_server_hello = _has_handshake_type(materialized, "2")
    fatal_alerts = [
        row for row in materialized if row.get("tls.alert_message.level") == "2"
    ]
    if not has_server_hello:
        return False, len(fatal_alerts)

    fatal_alert_frames = [
        int(row["frame.number"]) for row in fatal_alerts if row.get("frame.number")
    ]
    first_fatal_alert = min(fatal_alert_frames) if fatal_alert_frames else None
    application_data_frames = [
        int(row["frame.number"])
        for row in materialized
        if "23" in row.get("tls.record.content_type", "").split(",")
        and row.get("frame.number")
    ]
    if application_data_frames and (
        not fatal_alerts
        or (first_fatal_alert is not None and min(application_data_frames) < first_fatal_alert)
    ):
        return True, 0

    client_change_cipher = any(
        row.get("tcp.srcport") == client_source_port
        and "20" in row.get("tls.record.content_type", "").split(",")
        for row in materialized
    )
    server_change_cipher = any(
        row.get("tcp.srcport") != client_source_port
        and "20" in row.get("tls.record.content_type", "").split(",")
        for row in materialized
    )
    if first_fatal_alert is None and client_change_cipher and server_change_cipher:
        return True, 0
    if fatal_alerts:
        return False, len(fatal_alerts)
    # ponytail: packet-level completion heuristic; key-log-backed Finished
    # verification can replace this when exact TLS state is required.
    return False, 0


def _cipher_family(cipher_suite: str) -> str:
    if "CHACHA20" in cipher_suite:
        return "CHACHA20-POLY1305"
    if "GCM" in cipher_suite:
        return "AES-GCM"
    if "CBC" in cipher_suite:
        return "AES-CBC"
    return "UNKNOWN"


def _key_exchange(tls_version: str, cipher_suite: str) -> str:
    if tls_version == "TLS1.3" or "ECDHE" in cipher_suite:
        return "ECDHE"
    if "DHE" in cipher_suite:
        return "DHE"
    if "RSA" in cipher_suite:
        return "RSA"
    return "UNKNOWN"


def _encryption(cipher_suite: str) -> str:
    normalized = cipher_suite.upper().replace("-", "_")
    if "AES_256_GCM" in normalized:
        return "AES-256-GCM"
    if "AES_128_GCM" in normalized:
        return "AES-128-GCM"
    if "AES_256_CBC" in normalized:
        return "AES-256-CBC"
    if "AES_128_CBC" in normalized:
        return "AES-128-CBC"
    if "CHACHA20" in normalized:
        return "ChaCha20-Poly1305"
    if "3DES" in normalized:
        return "3DES"
    if "RC4" in normalized:
        return "RC4"
    return "UNKNOWN"


def _mac(tls_version: str, cipher_suite: str) -> str:
    normalized = cipher_suite.upper()
    if tls_version == "TLS1.3" or "GCM" in normalized or "CHACHA20" in normalized:
        return "AEAD"
    for digest in ("SHA512", "SHA384", "SHA256", "SHA1"):
        if digest in normalized:
            return digest
    return "UNKNOWN"


def _tls_details(
    tls_version: str,
    cipher_suite: str,
    key_exchange: str,
    forward_secrecy: bool,
) -> dict[str, object]:
    normalized_version = tls_version.replace("TLS", "TLS ", 1) if not tls_version.startswith("TLS ") else tls_version
    normalized_cipher = cipher_suite.upper()
    if normalized_version in {"TLS 1.0", "TLS 1.1"} or any(
        weak_cipher in normalized_cipher for weak_cipher in ("3DES", "RC4")
    ):
        posture_rating = "Deprecated"
    elif not forward_secrecy or key_exchange == "RSA":
        posture_rating = "Weak"
    elif normalized_version == "TLS 1.3" or any(
        strong_cipher in normalized_cipher for strong_cipher in ("GCM", "CHACHA20")
    ):
        posture_rating = "Strong"
    else:
        posture_rating = "Adequate"
    return {
        "version": normalized_version,
        "cipher_suite": cipher_suite,
        "key_exchange": key_exchange,
        "forward_secrecy": forward_secrecy,
        "encryption": _encryption(cipher_suite),
        "mac": _mac(tls_version, cipher_suite),
        "posture_rating": posture_rating,
    }


def _common_name(name: x509.Name) -> str:
    attributes = name.get_attributes_for_oid(NameOID.COMMON_NAME)
    return str(attributes[0].value) if attributes else "UNKNOWN"


def _public_key_details(public_key: object) -> tuple[str, int | None]:
    if isinstance(public_key, rsa.RSAPublicKey):
        return "RSA", public_key.key_size
    if isinstance(public_key, ec.EllipticCurvePublicKey):
        return "ECDSA", public_key.key_size
    if isinstance(public_key, dsa.DSAPublicKey):
        return "DSA", public_key.key_size
    if isinstance(public_key, (ed25519.Ed25519PublicKey, ed448.Ed448PublicKey)):
        return type(public_key).__name__.replace("PublicKey", "").upper(), None
    return "UNKNOWN", getattr(public_key, "key_size", None)


def _certificate_details(der_hex: str) -> dict[str, object]:
    try:
        certificates = [
            x509.load_der_x509_certificate(bytes.fromhex(value))
            for value in der_hex.split(",")
            if value
        ]
    except (TypeError, ValueError) as exc:
        raise PcapExtractionUnavailable("tshark returned malformed certificate DER") from exc
    if not certificates:
        raise PcapExtractionUnavailable("tshark returned an empty certificate")

    leaf = certificates[0]
    now = datetime.now(UTC)
    not_before = leaf.not_valid_before_utc
    not_after = leaf.not_valid_after_utc
    if now < not_before:
        status = "NOT_YET_VALID"
    elif now > not_after:
        status = "EXPIRED"
    else:
        status = "VALID"
    key_algorithm, key_size = _public_key_details(leaf.public_key())
    key_description = f"{key_algorithm}{f' {key_size} bit' if key_size else ''}"
    signature_hash = leaf.signature_hash_algorithm
    signature_algorithm = (
        f"{signature_hash.name.upper()}-{key_algorithm}"
        if signature_hash is not None
        else key_algorithm
    )

    chain: list[dict[str, object]] = []
    for index, certificate in enumerate(reversed(certificates)):
        if len(certificates) == 1:
            level = "leaf"
        elif index == 0:
            level = "root"
        elif index == len(certificates) - 1:
            level = "leaf"
        else:
            level = "intermediate"
        chain.append(
            {
                "level": level,
                "subject": _common_name(certificate.subject),
                "valid": certificate.not_valid_before_utc <= now <= certificate.not_valid_after_utc,
            }
        )

    return {
        "domain": _common_name(leaf.subject),
        "issuer": _common_name(leaf.issuer),
        "status": status,
        "valid_from": not_before.astimezone(UTC).isoformat().replace("+00:00", "Z"),
        "valid_until": not_after.astimezone(UTC).isoformat().replace("+00:00", "Z"),
        "key_algorithm": key_description,
        "signature_algorithm": signature_algorithm,
        "chain": chain,
    }


def _openssl_certificate_metadata(der_hex: str) -> dict[str, object]:
    try:
        # tshark emits repeated certificate fields as comma-separated DER values;
        # certificate posture is derived from the leaf certificate.
        der = bytes.fromhex(der_hex.split(",", 1)[0])
    except ValueError as exc:
        raise PcapExtractionUnavailable("tshark returned malformed certificate DER") from exc
    with tempfile.TemporaryDirectory() as directory:
        certificate_path = Path(directory) / "certificate.pem"
        certificate_path.write_text(ssl.DER_cert_to_PEM_cert(der))
        decoded = ssl._ssl._test_decode_cert(str(certificate_path))
        text = _command_output(["openssl", "x509", "-in", str(certificate_path), "-noout", "-text"])
    key_match = re.search(r"Public Key Algorithm:\s*([^\n]+).*?Public-Key:\s*\((\d+) bit\)", text, re.DOTALL)
    signature_match = re.search(r"Signature Algorithm:\s*([^\n]+)", text)
    expires_at = datetime.fromtimestamp(ssl.cert_time_to_seconds(decoded["notAfter"]), UTC)
    now = datetime.now(UTC)
    # ponytail: day granularity removes capture-clock jitter; use a recorded
    # capture timestamp if sub-day expiry precision becomes a requirement.
    remaining_days = (expires_at - now).total_seconds() / 86400
    key_algorithm = "RSA" if key_match and "rsa" in key_match.group(1).lower() else "UNKNOWN"
    signature = signature_match.group(1).strip().upper() if signature_match else "UNKNOWN"
    signature = signature.replace("WITHRSAENCRYPTION", "-RSA")
    return {
        "cert_present": True,
        "cert_expired": expires_at <= now,
        "cert_expires_in_days": (
            -1.0 if remaining_days <= 0 else float(round(remaining_days))
        ),
        "cert_key_algorithm": key_algorithm,
        "cert_key_length_bits": int(key_match.group(2)) if key_match else None,
        "cert_signature_algorithm": signature,
        "certificate_details": _certificate_details(der_hex),
    }


def _trusted_certificate_fields(
    der_hex: str,
    *,
    trusted_ca_path: Path | None,
    expected_hostname: str | None,
) -> dict[str, bool | None]:
    if not trusted_ca_path or not expected_hostname:
        return {"cert_valid": None, "cert_chain_valid": None, "hostname_mismatch": None}
    der = bytes.fromhex(der_hex)
    with tempfile.TemporaryDirectory() as directory:
        certificate_path = Path(directory) / "certificate.pem"
        certificate_path.write_text(ssl.DER_cert_to_PEM_cert(der))
        verified = subprocess.run(
            ["openssl", "verify", "-CAfile", str(trusted_ca_path), str(certificate_path)],
            text=True,
            capture_output=True,
        ).returncode == 0
        hostname_matches = subprocess.run(
            ["openssl", "x509", "-in", str(certificate_path), "-noout", "-checkhost", expected_hostname],
            text=True,
            capture_output=True,
        ).returncode == 0
    expired = bool(_openssl_certificate_metadata(der_hex)["cert_expired"])
    mismatch = not hostname_matches
    return {
        "cert_valid": verified and not mismatch and not expired,
        "cert_chain_valid": verified,
        "hostname_mismatch": mismatch,
    }


def detect_mail_ports(pcap_path: str | Path) -> list[tuple[Protocol, int]]:
    """Return supported mail destination ports observed by TShark."""
    path = Path(pcap_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    command = _tshark_command(path, None)
    command.extend(["-T", "fields", "-e", "tcp.dstport"])
    return sorted(
        {_MAIL_PORTS[port] for port in _command_output(command).splitlines() if port in _MAIL_PORTS},
        key=lambda item: (item[0].value, item[1]),
    )


def _authorized_manifest(protocol: Protocol) -> ScenarioManifest:
    """Neutral extraction metadata; authorized-capture labels are forced in extract_sessions."""
    return ScenarioManifest.model_validate(
        {
            "schema_version": "scenario.v1",
            "scenario_id": "authorized_capture",
            "family": "normal_baseline",
            "description": "Authorized passive capture",
            "protocol": protocol.value,
            "risk_label": "informational",
            "anomaly_label": 0,
            "tls": {
                "version": "TLS1.3",
                "cipher_suite": "TLS_AES_256_GCM_SHA384",
                "key_exchange": "ECDHE",
                "forward_secrecy": True,
            },
            "certificate": {"state": "unknown"},
            "client_behavior": {
                "starttls_advertised": True,
                "starttls_used": True,
                "handshake_attempts": 1,
            },
            "variation": {
                "duration_seconds": [0.0, 1.0],
                "packet_count": [0, 1],
                "byte_count": [0, 1],
            },
            "repetitions": 1,
            "seed": 0,
        }
    )


def extract_authorized_capture(
    pcap_path: str | Path,
    *,
    environment_id: str,
) -> list[SessionFeatureRecord]:
    """Extract supported mail sessions without client-controlled labels or key logs."""
    records: list[SessionFeatureRecord] = []
    for protocol, destination_port in detect_mail_ports(pcap_path):
        records.extend(
            extract_sessions(
                pcap_path,
                scenario=_authorized_manifest(protocol),
                environment_id=environment_id,
                destination_port=destination_port,
                source_type=SourceType.AUTHORIZED_CAPTURE,
            )
        )
    return records


def extract_sessions(
    pcap_path: str | Path,
    *,
    scenario: ScenarioManifest | Mapping[str, object],
    environment_id: str,
    destination_port: int,
    generator_seed: int | None = None,
    parameter_hash: str | None = None,
    trusted_ca_path: str | Path | None = None,
    expected_hostname: str | None = None,
    source_type: SourceType = SourceType.SYNTHETIC_PCAP,
) -> list[SessionFeatureRecord]:
    """Extract sessions from a PCAP for one protocol and destination port.

    Synthetic lab rows keep catalog labels. Authorized captures get informational
    labels and must not carry ``generator_seed``. Certificate validity fields are
    populated only when a CA and expected hostname are supplied.
    """
    path = Path(pcap_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    manifest = scenario if isinstance(scenario, ScenarioManifest) else ScenarioManifest.model_validate(scenario)
    keylog_path = path.parent / "tls.keys"
    packets = _load_packets(
        path,
        decode_port=destination_port,
        protocol=manifest.protocol,
        keylog_path=keylog_path if keylog_path.is_file() else None,
    )
    streams: dict[str, list[dict[str, str]]] = defaultdict(list)
    for packet in packets:
        if packet["tcp.stream"]:
            streams[packet["tcp.stream"]].append(packet)
    capture_id = f"pcap-{hashlib.sha256(path.read_bytes()).hexdigest()[:16]}"
    records: list[SessionFeatureRecord] = []
    for stream_id, rows in sorted(streams.items(), key=lambda item: int(item[0])):
        client_rows = [row for row in rows if row["tcp.dstport"] == str(destination_port)]
        if not client_rows:
            continue
        first = client_rows[0]
        plaintext = b"".join(
            bytes.fromhex(row["tcp.payload"])
            for row in rows
            if row["tcp.payload"] and all(c in "0123456789abcdefABCDEF" for c in row["tcp.payload"])
        )
        handshake_success, handshake_failures = classify_handshake(
            rows,
            first["tcp.srcport"],
        )
        tls_version_code = _first(
            row["tls.handshake.extensions.supported_version"]
            or row["tls.handshake.version"]
            for row in rows
            if _has_handshake_type((row,), "2")
        )
        tls_version = _TLS_VERSION.get(tls_version_code or "", "UNKNOWN") if handshake_success else None
        cipher_code = _first(
            row["tls.handshake.ciphersuite"]
            for row in rows
            if _has_handshake_type((row,), "2")
        )
        cipher_suite = _CIPHER_SUITE.get(cipher_code or "", f"TLS_{cipher_code}") if handshake_success else None
        signature_code = _first(row["tls.handshake.sig_hash_alg"] for row in rows)
        certificate_der = _first(row["tls.handshake.certificate"] for row in rows)
        certificate = (
            _openssl_certificate_metadata(certificate_der) if certificate_der else {"cert_present": False}
        )
        if certificate_der:
            certificate.update(
                _trusted_certificate_fields(
                    certificate_der,
                    trusted_ca_path=Path(trusted_ca_path) if trusted_ca_path else None,
                    expected_hostname=expected_hostname,
                )
            )
        start = float(rows[0]["frame.time_epoch"])
        end = float(rows[-1]["frame.time_epoch"])
        packet_numbers = [int(row["frame.number"]) for row in rows]
        key_exchange = _key_exchange(tls_version or "", cipher_suite or "")
        tls_details = (
            _tls_details(tls_version, cipher_suite, key_exchange, key_exchange in {"ECDHE", "DHE"})
            if handshake_success and tls_version and cipher_suite
            else None
        )
        records.append(
            SessionFeatureRecord(
                schema_version="session-features.v1",
                provenance=Provenance(
                    capture_id=capture_id,
                    flow_id=f"{capture_id}:stream:{stream_id}",
                    session_id=f"{capture_id}:stream:{stream_id}",
                    source_type=source_type,
                    scenario_id=manifest.scenario_id,
                    environment_id=environment_id,
                    parameter_hash=parameter_hash,
                    generator_seed=generator_seed,
                    evidence_refs=[
                        EvidenceReference(
                            source=EvidenceSource.PCAP,
                            stream_id=int(stream_id),
                            packet_start=min(packet_numbers),
                            packet_end=max(packet_numbers),
                            fields=[
                                "tcp.stream",
                                "tcp.payload",
                                "tls.handshake.type",
                                "tls.handshake.extensions.supported_version",
                                "tls.handshake.ciphersuite",
                                "tls.handshake.certificate",
                                "tls.record.content_type",
                                "tls.alert_message.level",
                            ],
                        )
                    ],
                ),
                features=SessionFeatures(
                    protocol=manifest.protocol,
                    src_port=int(first["tcp.srcport"]),
                    dst_port=destination_port,
                    starttls_advertised=b"STARTTLS" in plaintext,
                    starttls_used=b"STARTTLS" in plaintext and _has_handshake_type(rows, "1"),
                    handshake_success=handshake_success,
                    handshake_failures=handshake_failures,
                    renegotiation_count=0,
                    session_duration_seconds=max(0.0, end - start),
                    packet_count=len(rows),
                    byte_count=sum(int(row["frame.len"]) for row in rows),
                    retransmission_count=sum(bool(row["tcp.analysis.retransmission"]) for row in rows),
                    out_of_order_count=sum(bool(row["tcp.analysis.out_of_order"]) for row in rows),
                    tls_version=tls_version,
                    cipher_suite=cipher_suite,
                    cipher_family=_cipher_family(cipher_suite) if cipher_suite else None,
                    key_exchange=key_exchange if handshake_success else None,
                    signature_algorithm=_SIGNATURE_ALGORITHM.get(signature_code or "", "UNKNOWN") if handshake_success else None,
                    forward_secrecy=key_exchange in {"ECDHE", "DHE"} if handshake_success else None,
                    cert_present=certificate["cert_present"],
                    cert_valid=certificate.get("cert_valid"),
                    cert_expired=certificate.get("cert_expired"),
                    cert_expires_in_days=certificate.get("cert_expires_in_days"),
                    cert_chain_valid=certificate.get("cert_chain_valid"),
                    hostname_mismatch=certificate.get("hostname_mismatch"),
                    cert_key_algorithm=certificate.get("cert_key_algorithm"),
                    cert_key_length_bits=certificate.get("cert_key_length_bits"),
                    cert_signature_algorithm=certificate.get("cert_signature_algorithm"),
                    tls_details=tls_details,
                    certificate_details=certificate.get("certificate_details"),
                ),
                labels=SessionLabels(
                    risk_label=RiskLabel.INFORMATIONAL
                    if source_type is SourceType.AUTHORIZED_CAPTURE
                    else manifest.risk_label,
                    anomaly_label=AnomalyLabel.NORMAL
                    if source_type is SourceType.AUTHORIZED_CAPTURE
                    else manifest.anomaly_label,
                    expected_finding_ids=[]
                    if source_type is SourceType.AUTHORIZED_CAPTURE
                    else list(manifest.expected_finding_ids),
                ),
            )
        )
    return records


if __name__ == "__main__":
    from ml.dataset import CATALOG

    pcap = Path("datasets/lab/captures/synthetic_mail.pcap")
    if pcap.is_file():
        normal_smtp = next(item["manifest"] for item in CATALOG if item["manifest"]["scenario_id"] == "normal_tls13_valid")
        records = extract_sessions(
            pcap,
            scenario=normal_smtp,
            environment_id="lab_smoke",
            generator_seed=420042,
            parameter_hash="lab-smoke",
            destination_port=2525,
            trusted_ca_path=pcap.parent / "ca.pem",
            expected_hostname="mail-lab",
        )
        assert len(records) == 1 and records[0].features.handshake_success
        print(records[0].model_dump_json(indent=2))

from __future__ import annotations

import imaplib
import json
import os
import smtplib
import socket
import ssl
import time
from pathlib import Path
from typing import Any

HOST = "mail-lab"
TLS_VERSION = {
    "TLS1.0": ssl.TLSVersion.TLSv1,
    "TLS1.1": ssl.TLSVersion.TLSv1_1,
    "TLS1.2": ssl.TLSVersion.TLSv1_2,
    "TLS1.3": ssl.TLSVersion.TLSv1_3,
}


def load_runtime_profile(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def configure_tls_context(context: ssl.SSLContext, profile: dict[str, Any]) -> None:
    context.minimum_version = TLS_VERSION[profile["tls_minimum_version"]]
    context.maximum_version = TLS_VERSION[profile["tls_maximum_version"]]
    if profile.get("cipher_string"):
        context.set_ciphers(profile["cipher_string"])


def tls_context(profile: dict[str, Any]) -> ssl.SSLContext:
    context = ssl.create_default_context(cafile="/captures/ca.pem")
    configure_tls_context(context, profile)
    context.keylog_filename = "/captures/tls.keys"
    return context


def _record_unsupported(profile: dict[str, Any], error: ssl.SSLError) -> None:
    if profile["tls_maximum_version"] in {"TLS1.0", "TLS1.1"} or profile.get("cipher_string"):
        Path("/captures/runtime_status.json").write_text(
            json.dumps(
                {"status": "unsupported_in_lab", "detail": str(error)},
                sort_keys=True,
            )
            + "\n"
        )


def _smtp_abort_starttls(port: int) -> None:
    with socket.create_connection((HOST, port), timeout=10) as client:
        client.recv(4096)
        client.sendall(b"EHLO client\r\n")
        client.recv(4096)
        client.sendall(b"STARTTLS\r\n")
        client.recv(4096)


def smtp_starttls(profile: dict[str, Any]) -> None:
    port = int(profile["destination_port"])
    if profile["client_mode"] == "abort_starttls":
        _smtp_abort_starttls(port)
        return
    with smtplib.SMTP(HOST, port, timeout=10) as client:
        client.ehlo()
        if profile["client_mode"] == "starttls":
            try:
                client.starttls(context=tls_context(profile))
                client.ehlo()
            except ssl.SSLCertVerificationError:
                return
            except ssl.SSLError as exc:
                _record_unsupported(profile, exc)
                return


def _imap_abort_starttls(port: int) -> None:
    with socket.create_connection((HOST, port), timeout=10) as client:
        client.recv(4096)
        client.sendall(b"a1 STARTTLS\r\n")
        client.recv(4096)


def imap_starttls(profile: dict[str, Any]) -> None:
    port = int(profile["destination_port"])
    if profile["client_mode"] == "abort_starttls":
        _imap_abort_starttls(port)
        return
    client = imaplib.IMAP4(HOST, port, timeout=10)
    try:
        if profile["client_mode"] == "starttls":
            try:
                client.starttls(ssl_context=tls_context(profile))
            except ssl.SSLCertVerificationError:
                return
            except ssl.SSLError as exc:
                _record_unsupported(profile, exc)
                return
    finally:
        try:
            client.logout()
        except (imaplib.IMAP4.error, OSError):
            pass


def main() -> None:
    profile = load_runtime_profile(
        os.environ.get("RUNTIME_PROFILE_PATH", "/captures/runtime_profile.json")
    )
    handler = smtp_starttls if profile["protocol"] == "SMTP" else imap_starttls
    for _ in range(int(profile["connection_count"])):
        handler(profile)
        time.sleep(int(profile["command_delay_milliseconds"]) / 1000)
    time.sleep(1)


if __name__ == "__main__":
    main()

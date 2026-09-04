from __future__ import annotations

import imaplib
import json
import os
import poplib
import smtplib
import socket
import ssl
import subprocess
import time
from pathlib import Path
from typing import Any

HOST = os.environ.get("MAIL_HOST", "mail-core")
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


def tls_context(
    profile: dict[str, Any],
    *,
    verify: bool = True,
) -> ssl.SSLContext:
    if verify:
        context = ssl.create_default_context(cafile="/captures/ca.pem")
    else:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
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


def smtp_starttls(profile: dict[str, Any], *, verify: bool = True) -> None:
    port = int(profile["destination_port"])
    if profile["client_mode"] == "abort_starttls":
        _smtp_abort_starttls(port)
        return
    with smtplib.SMTP(HOST, port, timeout=10) as client:
        client.ehlo()
        if profile["client_mode"] == "starttls":
            try:
                client.starttls(context=tls_context(profile, verify=verify))
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


def imap_starttls(profile: dict[str, Any], *, verify: bool = True) -> None:
    port = int(profile["destination_port"])
    if profile["client_mode"] == "abort_starttls":
        _imap_abort_starttls(port)
        return
    client = imaplib.IMAP4(HOST, port, timeout=10)
    try:
        if profile["client_mode"] == "starttls":
            try:
                client.starttls(ssl_context=tls_context(profile, verify=verify))
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


def pop3_starttls(profile: dict[str, Any], *, verify: bool = True) -> None:
    port = int(profile["destination_port"])
    if profile["client_mode"] == "abort_starttls":
        with socket.create_connection((HOST, port), timeout=10) as client:
            client.recv(4096)
            client.sendall(b"STLS\r\n")
            client.recv(4096)
        return
    client = poplib.POP3(HOST, port, timeout=10)
    try:
        if profile["client_mode"] == "starttls":
            try:
                client.stls(context=tls_context(profile, verify=verify))
            except ssl.SSLCertVerificationError:
                return
            except ssl.SSLError as exc:
                _record_unsupported(profile, exc)
                return
    finally:
        try:
            client.quit()
        except (poplib.error_proto, OSError):
            pass


def legacy_starttls(profile: dict[str, Any]) -> None:
    if profile["tls_minimum_version"] == "TLS1.3":
        handler = {
            "SMTP": smtp_starttls,
            "IMAP": imap_starttls,
            "POP3": pop3_starttls,
        }[profile["protocol"]]
        handler(profile, verify=False)
        return

    protocol = profile["protocol"].lower()
    command = [
        "openssl",
        "s_client",
        "-connect",
        f"{HOST}:{profile['destination_port']}",
        "-starttls",
        protocol,
        {"TLS1.0": "-tls1", "TLS1.1": "-tls1_1", "TLS1.2": "-tls1_2"}[
            profile["tls_minimum_version"]
        ],
    ]
    if profile.get("cipher_string"):
        command.extend(["-cipher", profile["cipher_string"].split(":@")[0]])
    command.append("-brief")
    completed = subprocess.run(
        command,
        input="Q\n",
        text=True,
        capture_output=True,
        timeout=15,
    )
    if completed.returncode:
        Path("/captures/runtime_status.json").write_text(
            json.dumps(
                {
                    "status": "unsupported_in_lab",
                    "detail": completed.stderr[-1000:],
                },
                sort_keys=True,
            )
            + "\n"
        )


def main() -> None:
    profile = load_runtime_profile(
        os.environ.get("RUNTIME_PROFILE_PATH", "/captures/runtime_profile.json")
    )
    handler = {
        "SMTP": smtp_starttls,
        "IMAP": imap_starttls,
        "POP3": pop3_starttls,
    }[profile["protocol"]]
    for _ in range(int(profile["connection_count"])):
        (legacy_starttls if profile.get("service") == "legacy-lab" else handler)(profile)
        time.sleep(int(profile["command_delay_milliseconds"]) / 1000)
    time.sleep(1)


if __name__ == "__main__":
    main()

from __future__ import annotations

import asyncio
import json
import os
import ssl
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Callable

CAPTURE_DIR = Path("/captures")
CERTIFICATE = Path("/tmp/certs/cert.pem")
PRIVATE_KEY = Path("/tmp/certs/key.pem")
TLS_VERSION = {
    "TLS1.0": ssl.TLSVersion.TLSv1,
    "TLS1.1": ssl.TLSVersion.TLSv1_1,
    "TLS1.2": ssl.TLSVersion.TLSv1_2,
    "TLS1.3": ssl.TLSVersion.TLSv1_3,
}


class UnsupportedLabScenario(RuntimeError):
    pass


def load_runtime_profile(path: str | Path) -> dict[str, Any]:
    profile = json.loads(Path(path).read_text())
    required = {
        "scenario",
        "protocol",
        "client_mode",
        "starttls_advertised",
        "starttls_accepted",
        "tls_minimum_version",
        "tls_maximum_version",
        "certificate_mode",
    }
    missing = sorted(required - set(profile))
    if missing:
        raise ValueError(f"runtime profile is missing: {missing}")
    return profile


def _run_openssl(*arguments: str) -> None:
    try:
        subprocess.run(
            ["openssl", *arguments],
            check=True,
            text=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as exc:
        raise UnsupportedLabScenario(exc.stderr.strip() or "OpenSSL rejected profile") from exc


def _write_status(status: str, detail: str | None = None) -> None:
    payload: dict[str, str] = {"status": status}
    if detail:
        payload["detail"] = detail
    (CAPTURE_DIR / "runtime_status.json").write_text(
        json.dumps(payload, sort_keys=True) + "\n"
    )


def _openssl_timestamp(value: datetime) -> str:
    return value.strftime("%y%m%d%H%M%SZ")


def _sign_expired_certificate(
    *,
    cert_dir: Path,
    signing_ca: Path,
    signing_key: Path,
    csr: Path,
) -> None:
    config = cert_dir / "expired-ca.cnf"
    config.write_text(
        "[ca]\n"
        "default_ca = local\n"
        "[local]\n"
        f"database = {cert_dir / 'index.txt'}\n"
        f"new_certs_dir = {cert_dir}\n"
        f"certificate = {signing_ca}\n"
        f"private_key = {signing_key}\n"
        f"serial = {cert_dir / 'serial'}\n"
        "default_md = sha256\n"
        "default_days = 365\n"
        "policy = local_policy\n"
        "copy_extensions = copy\n"
        "unique_subject = no\n"
        "[local_policy]\n"
        "commonName = supplied\n"
    )
    (cert_dir / "index.txt").write_text("")
    (cert_dir / "serial").write_text("01\n")
    now = datetime.now(UTC)
    _run_openssl(
        "ca",
        "-batch",
        "-config",
        str(config),
        "-in",
        str(csr),
        "-out",
        str(CERTIFICATE),
        "-extfile",
        str(cert_dir / "server.ext"),
        "-startdate",
        _openssl_timestamp(now - timedelta(days=2)),
        "-enddate",
        _openssl_timestamp(now - timedelta(minutes=1)),
    )


def provision_certificates(profile: dict[str, Any]) -> None:
    cert_dir = Path("/tmp/certs")
    cert_dir.mkdir(parents=True, exist_ok=True)
    CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    _run_openssl(
        "req",
        "-x509",
        "-newkey",
        "rsa:2048",
        "-nodes",
        "-keyout",
        str(cert_dir / "ca-key.pem"),
        "-out",
        str(CAPTURE_DIR / "ca.pem"),
        "-subj",
        "/CN=SecureMailScope-Lab-CA",
        "-days",
        str(profile.get("certificate_validity_days", 365)),
    )
    certificate_mode = profile["certificate_mode"]
    key_bits = "1024" if certificate_mode == "weak_rsa" else "2048"
    hostname = os.environ.get("LAB_SERVER_NAME", "mail-core")
    server_name = f"not-{hostname}" if certificate_mode == "hostname_mismatch" else hostname
    _run_openssl(
        "req",
        "-newkey",
        f"rsa:{key_bits}",
        "-nodes",
        "-keyout",
        str(PRIVATE_KEY),
        "-out",
        str(cert_dir / "server.csr"),
        "-subj",
        f"/CN={server_name}",
    )
    (cert_dir / "server.ext").write_text(f"subjectAltName=DNS:{server_name}\n")
    signing_ca = CAPTURE_DIR / "ca.pem"
    signing_key = cert_dir / "ca-key.pem"
    if certificate_mode == "unknown_ca":
        signing_ca = cert_dir / "untrusted-ca.pem"
        signing_key = cert_dir / "untrusted-ca-key.pem"
        _run_openssl(
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-keyout",
            str(signing_key),
            "-out",
            str(signing_ca),
            "-subj",
            "/CN=SecureMailScope-Untrusted-CA",
            "-days",
            str(profile.get("certificate_validity_days", 365)),
        )
    if certificate_mode == "expired":
        _sign_expired_certificate(
            cert_dir=cert_dir,
            signing_ca=signing_ca,
            signing_key=signing_key,
            csr=cert_dir / "server.csr",
        )
    else:
        _run_openssl(
            "x509",
            "-req",
            "-in",
            str(cert_dir / "server.csr"),
            "-CA",
            str(signing_ca),
            "-CAkey",
            str(signing_key),
            "-CAcreateserial",
            "-out",
            str(CERTIFICATE),
            "-days",
            str(profile.get("certificate_validity_days", 365)),
            "-sha256",
            "-extfile",
            str(cert_dir / "server.ext"),
        )


def tls_context(profile: dict[str, Any]) -> ssl.SSLContext:
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    try:
        context.minimum_version = TLS_VERSION[profile["tls_minimum_version"]]
        context.maximum_version = TLS_VERSION[profile["tls_maximum_version"]]
        if profile.get("cipher_string"):
            context.set_ciphers(profile["cipher_string"])
        context.load_cert_chain(CERTIFICATE, PRIVATE_KEY)
    except (KeyError, ssl.SSLError) as exc:
        raise UnsupportedLabScenario(str(exc)) from exc
    return context


async def smtp_session(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    profile: dict[str, Any],
) -> None:
    writer.write(b"220 mail-lab ESMTP\r\n")
    await writer.drain()
    encrypted = False
    context = tls_context(profile)
    while line := await reader.readline():
        command = line.decode("ascii", "replace").strip().upper()
        if command.startswith(("EHLO", "HELO")):
            writer.write(
                b"250-mail-lab\r\n"
                + (
                    b"250-STARTTLS\r\n"
                    if profile["starttls_advertised"] and not encrypted
                    else b""
                )
                + b"250 HELP\r\n"
            )
        elif command == "STARTTLS" and not encrypted:
            if not profile["starttls_accepted"]:
                writer.write(b"454 TLS unavailable\r\n")
            else:
                writer.write(b"220 Ready to start TLS\r\n")
                await writer.drain()
                await writer.start_tls(context)
                encrypted = True
        elif command == "QUIT":
            writer.write(b"221 Bye\r\n")
            await writer.drain()
            break
        else:
            writer.write(b"250 OK\r\n")
        await writer.drain()
    writer.close()
    await writer.wait_closed()


async def imap_session(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    profile: dict[str, Any],
) -> None:
    writer.write(b"* OK mail-lab IMAP4rev1\r\n")
    await writer.drain()
    encrypted = False
    context = tls_context(profile)
    while line := await reader.readline():
        parts = line.decode("ascii", "replace").strip().split(maxsplit=2)
        if len(parts) < 2:
            continue
        tag, command = parts[0], parts[1].upper()
        if command == "CAPABILITY":
            capabilities = "IMAP4rev1"
            if profile["starttls_advertised"] and not encrypted:
                capabilities += " STARTTLS"
            writer.write(
                f"* CAPABILITY {capabilities}\r\n{tag} OK CAPABILITY completed\r\n".encode()
            )
        elif command == "STARTTLS" and not encrypted:
            if not profile["starttls_accepted"]:
                writer.write(f"{tag} NO TLS unavailable\r\n".encode())
            else:
                writer.write(f"{tag} OK Begin TLS negotiation\r\n".encode())
                await writer.drain()
                await writer.start_tls(context)
                encrypted = True
        elif command == "LOGOUT":
            writer.write(f"* BYE\r\n{tag} OK LOGOUT completed\r\n".encode())
            await writer.drain()
            break
        else:
            writer.write(f"{tag} OK\r\n".encode())
        await writer.drain()
    writer.close()
    await writer.wait_closed()


async def safe_session(
    handler: Callable[[asyncio.StreamReader, asyncio.StreamWriter, dict[str, Any]], Any],
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    profile: dict[str, Any],
) -> None:
    try:
        await handler(reader, writer, profile)
    except (ConnectionError, ssl.SSLError):
        pass
    finally:
        if not writer.is_closing():
            writer.close()
        try:
            await writer.wait_closed()
        except ConnectionError:
            pass


async def main() -> None:
    profile = load_runtime_profile(
        os.environ.get("RUNTIME_PROFILE_PATH", "/captures/runtime_profile.json")
    )
    try:
        provision_certificates(profile)
        tls_context(profile)
    except UnsupportedLabScenario as exc:
        _write_status("unsupported_in_lab", str(exc))
        return
    smtp = await asyncio.start_server(
        lambda reader, writer: safe_session(smtp_session, reader, writer, profile),
        "0.0.0.0",
        2525,
    )
    imap = await asyncio.start_server(
        lambda reader, writer: safe_session(imap_session, reader, writer, profile),
        "0.0.0.0",
        1143,
    )
    async with smtp, imap:
        await asyncio.gather(smtp.serve_forever(), imap.serve_forever())


if __name__ == "__main__":
    asyncio.run(main())

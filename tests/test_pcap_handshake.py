from __future__ import annotations

from pathlib import Path

import ml.pcap as pcap
from api.schemas import cryptographic_posture
from ml.pcap import client_handshake_succeeded, classify_handshake
from ml.schema import Protocol


def test_handshake_requires_a_client_finished_message() -> None:
    rows = [
        {"tcp.srcport": "2525", "tls.handshake.type": "2,11,14"},
        {"tcp.srcport": "50000", "tls.handshake.type": ""},
    ]

    assert client_handshake_succeeded(rows, "50000") is False


def test_handshake_accepts_a_client_finished_message() -> None:
    rows = [
        {"tcp.srcport": "2525", "tls.handshake.type": "2,11,14"},
        {"tcp.srcport": "50000", "tls.handshake.type": "16,20"},
    ]

    assert client_handshake_succeeded(rows, "50000") is True


def test_encrypted_tls_transition_counts_as_a_successful_handshake() -> None:
    rows = [
        {
            "tcp.srcport": "50000",
            "tls.handshake.type": "1",
            "tls.record.content_type": "22,20",
            "tls.alert_message.level": "",
        },
        {
            "tcp.srcport": "25",
            "tls.handshake.type": "2",
            "tls.record.content_type": "22,20",
            "tls.alert_message.level": "",
        },
    ]

    assert classify_handshake(rows, "50000") == (True, 0)


def test_fatal_tls_alert_counts_as_a_handshake_failure() -> None:
    rows = [
        {
            "tcp.srcport": "50000",
            "tls.handshake.type": "1",
            "tls.record.content_type": "22",
            "tls.alert_message.level": "",
        },
        {
            "tcp.srcport": "25",
            "tls.handshake.type": "",
            "tls.record.content_type": "21",
            "tls.alert_message.level": "2",
        },
    ]

    assert classify_handshake(rows, "50000") == (False, 1)


def test_missing_finished_message_without_failure_is_unknown() -> None:
    rows = [
        {
            "tcp.srcport": "50000",
            "tls.handshake.type": "1",
            "tls.record.content_type": "22",
            "tls.alert_message.level": "",
        },
        {
            "tcp.srcport": "25",
            "tls.handshake.type": "2",
            "tls.record.content_type": "22",
            "tls.alert_message.level": "",
        },
    ]

    assert classify_handshake(rows, "50000") == (False, 0)
    assert cryptographic_posture({"handshake_success": False, "handshake_failures": 0}) == "unknown"
    assert cryptographic_posture({"handshake_success": False, "handshake_failures": 1}) == "handshake_failed"


def test_tshark_command_has_a_bounded_timeout(monkeypatch) -> None:
    seen = {}

    def run(command, **kwargs):
        seen.update(kwargs)
        return pcap.subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(pcap.subprocess, "run", run)

    pcap._command_output(["tshark"])

    assert seen["timeout"] == pcap.TSHARK_TIMEOUT_SECONDS


def test_detect_mail_ports_keeps_only_supported_destinations(
    tmp_path: Path,
    monkeypatch,
) -> None:
    capture = tmp_path / "capture.pcap"
    capture.write_bytes(b"pcap")
    monkeypatch.setattr(pcap, "_tshark_command", lambda *_args: ["tshark"])
    monkeypatch.setattr(pcap, "_command_output", lambda _command: "25\n143\n443\n25\n")

    assert pcap.detect_mail_ports(capture) == [
        (Protocol.IMAP, 143),
        (Protocol.SMTP, 25),
    ]

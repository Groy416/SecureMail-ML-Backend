from __future__ import annotations

from ml.pcap import client_handshake_succeeded


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

from __future__ import annotations

import ssl
from pathlib import Path

from ml.pcap import _certificate_details, _tls_details


def test_tls_details_normalize_negotiated_values() -> None:
    assert _tls_details("TLS1.3", "TLS_AES_256_GCM_SHA384", "ECDHE", True) == {
        "version": "TLS 1.3",
        "cipher_suite": "TLS_AES_256_GCM_SHA384",
        "key_exchange": "ECDHE",
        "forward_secrecy": True,
        "encryption": "AES-256-GCM",
        "mac": "AEAD",
        "posture_rating": "Strong",
    }


def test_certificate_details_include_x509_identity_and_chain() -> None:
    pem = Path("datasets/lab/captures/ca.pem").read_text()
    details = _certificate_details(ssl.PEM_cert_to_DER_cert(pem).hex())

    assert details["domain"]
    assert details["issuer"]
    assert details["valid_from"].endswith("Z")
    assert details["valid_until"].endswith("Z")
    assert details["key_algorithm"]
    assert details["signature_algorithm"]
    assert details["chain"]
    assert details["chain"][0]["level"] == "leaf"

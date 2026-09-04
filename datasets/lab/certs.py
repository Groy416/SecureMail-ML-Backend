from __future__ import annotations

import os
import shutil
from pathlib import Path

from server import CERTIFICATE, PRIVATE_KEY, load_runtime_profile, provision_certificates

profile = load_runtime_profile(os.environ.get("RUNTIME_PROFILE_PATH", "/captures/runtime_profile.json"))
provision_certificates(profile)
if profile["service"] == "legacy-lab":
    config = Path("/captures/mail-config")
    config.mkdir(exist_ok=True)
    (config / "dovecot.cf").write_text(
        "ssl_min_protocol = TLSv1\n"
        "ssl_cipher_list = ALL:@SECLEVEL=0\n"
    )
    (config / "postfix-main.cf").write_text(
        "smtpd_tls_ciphers = medium\n"
        "smtpd_tls_mandatory_ciphers = medium\n"
        "smtpd_tls_exclude_ciphers =\n"
        "smtpd_tls_protocols = !SSLv2,!SSLv3\n"
        "smtpd_tls_mandatory_protocols = !SSLv2,!SSLv3\n"
    )
certs = Path("/captures/certs")
certs.mkdir(exist_ok=True)
shutil.copy2(CERTIFICATE, certs / "cert.pem")
shutil.copy2(PRIVATE_KEY, certs / "key.pem")

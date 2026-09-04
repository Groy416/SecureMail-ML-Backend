# Full PCAP training-matrix design

## Goal

Finish the unsupported crypto profiles, add POP3 STARTTLS, generate 5,010
packet-backed synthetic rows, assemble a grouped dataset, and train the
existing XGBoost, Random Forest, and Isolation Forest bundle.

## Legacy endpoint

Replace the temporary Python protocol harness with an internal `mail-core`
Compose service running Postfix for SMTP and Dovecot for IMAP/POP3. Add a
separate `legacy-lab` service and matching synthetic client. It runs only
profiles that need weak RSA or expired certificates. The modern
Postfix/Dovecot service remains the default for TLS 1.2/1.3 profiles. The
matrix uses TLS 1.2 `AES128-SHA` with RSA key exchange for legacy-crypto
coverage; 3DES remains an explicit `unsupported_in_lab` result and is not a
required training capture.

The legacy service receives the same immutable runtime-profile JSON. It emits
`unsupported_in_lab` if OpenSSL cannot load the requested provider, cipher,
key size, or protocol. It never emits an imitation feature record.

Expired certificates are issued with explicit historical `notBefore` and
`notAfter` values through an OpenSSL CA configuration. Weak RSA uses a 1024-bit
server key only inside the legacy service. Clients use the matching private CA
and selected protocol/cipher settings; no external network, credentials, or
message data are used.

## POP3

The modern and legacy services expose POP3 STARTTLS on an internal lab port.
The client uses Python `poplib` for normal and unused STARTTLS behavior and a
raw socket for an intentional upgrade abort. The passive extractor recognizes
POP3 on the configured port and emits the existing canonical `Protocol.POP3`
session record, with no new model fields.

## Matrix

Run 15 captures: one distinct scenario for each risk label in each of train,
calibration, and held-out test environments. Each capture contains 334 client
sessions, yielding exactly 5,010 extracted rows before any explicit rejected
or unsupported attempt.

- Informational: normal TLS 1.3, normal TLS 1.2, successful STARTTLS.
- Low: certificate-expiry advisory, warning, and soon-expiring profiles.
- Medium: unusual cipher, unexpected TLS version, uncommon ChaCha20.
- High: certificate/STARTTLS/RSA-forward-secrecy profiles.
- Critical: deprecated TLS, weak cipher, combined critical weaknesses.

Protocol assignments rotate SMTP, IMAP, and POP3 across the 15 captures.
Resolved manifests have protocol-specific scenario IDs. Provenance records a
SHA-256 parameter-combination hash for the immutable runtime profile. No
scenario ID, environment ID, parameter-combination hash, or capture ID occurs
in more than one split.

Within one capture, client sessions vary deterministic command count and timing
from the profile seed and session index without changing their label. Captures
are the grouping unit: all 134 rows from one PCAP stay in one split.

## Dataset and training

The runner writes each run's PCAP hash, scenario-manifest hash, profile hash,
container image digests, extractor version, dependency versions, seed, status,
and source revision when Git metadata is available. Successful runs are
extracted and passed to `assemble_successful_runs`; the artifact is persisted
with `write_dataset_run`.

Train only if all 15 captures succeeded and the assembled dataset has exactly
5,010 rows. Use the environment, scenario, parameter-hash, and capture grouped
split; train the current model bundle, calibrate it on the calibration
environment, evaluate it on the held-out environment, and save the model bundle
plus synthetic-only evaluation report. Do not claim the result represents
production performance. Preserve and report the current fusion critical-recall
result; do not tune it against the held-out test set.

## Verification

- Unit tests: legacy-profile selection, POP3 profile resolution, historical
  certificate configuration, batch count, and refusal of incomplete matrix.
- Docker integration: Postfix/Dovecot valid POP3, weak-cipher legacy capture
  or explicit unsupported status, and expired-certificate capture or explicit
  unsupported status.
- Full matrix: exactly 15 successful capture manifests, exactly 5,010 records,
  checksum-valid dataset, no environment/scenario/parameter/capture leakage,
  train/calibrate/evaluate/save/load bundle round-trip.

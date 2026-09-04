# SMTP/IMAP PCAP scenario-matrix design

## Goal

Generate reproducible, packet-backed SMTP and IMAP session records for the
existing ML pipeline. Every record is labeled by a controlled scenario manifest;
no label is inferred from model output or packet heuristics.

## Boundary

The lab remains an internal Docker network with no published ports, real
mailboxes, credentials, or message bodies. The current headless clients remain
the primary generator. Roundcube is deferred to a separate valid-TLS smoke
profile because it needs authenticated mail services and cannot reliably create
legacy or failing TLS sessions.

POP3 is deferred.

## Components

- `datasets/lab/scenarios.py`: maps a `ScenarioManifest` to a runtime profile.
  It contains behavior not representable as model features: protocol, TLS
  versions/ciphers, certificate posture, whether STARTTLS is advertised or
  accepted, command count, timing jitter, and expected support.
- `datasets/lab/runner.py`: creates one isolated Compose run per scenario and
  environment. It writes the scenario manifest, run metadata, PCAP SHA-256,
  image IDs, and final status.
- `datasets/lab/server.py` and `client.py`: consume the runtime profile rather
  than hard-coded TLS 1.3 behavior.
- `ml/pcap.py`: extracts each completed capture with the scenario manifest and
  preserves packet ranges as evidence.
- `ml.dataset.assemble_pcap_dataset`: accepts only extracted
  `synthetic_pcap` records with matching manifests and capture hashes.

## Scenario support policy

The first runnable matrix includes valid TLS 1.3 and TLS 1.2, STARTTLS unused,
STARTTLS handshake failure, invalid certificate chain, hostname mismatch,
expired certificate, RSA without forward secrecy, and behavioral repetition.

Legacy TLS and weak ciphers use a profile that requests the exact OpenSSL
setting. If the container rejects it, the runner writes `unsupported_in_lab`
with the requested settings and error. It emits no PCAP-backed feature record
for that attempt. This also applies to weak RSA keys or any unsafe combination
refused by the local runtime.

Certificate-failure profiles use TLS 1.2 so certificate metadata remains
passively visible. Valid TLS 1.3 profiles use the lab-only key log for packet
extraction. The key log is ignored and never treated as a production capture
capability.

## Data flow

1. Select a catalog scenario, protocol, environment ID, and derived seed.
2. Resolve an immutable runtime profile and a protocol-specific scenario manifest.
   The resolved manifest has the selected protocol and an ID suffixed with
   `-smtp` or `-imap`; this prevents protocol rows from sharing a scenario ID.
3. Run the isolated mail service, client, and `tcpdump` capture.
4. On success, calculate the PCAP SHA-256 and call `extract_sessions`.
5. Persist run metadata, extracted record JSON, and status.
6. Assemble only successful records into a `synthetic_pcap` dataset. Its
   `session_count` is the actual successful extracted-record count. Grouped
   splitting retains environment, scenario, and capture isolation.

A failed or unsupported scenario is a run result, not a substituted
feature-only row.

## Validation

Unit tests cover profile resolution, deterministic run metadata, and refusal
of unsupported settings. A Docker integration test covers one valid SMTP and
one valid IMAP profile from manifest to validated session record. Dataset tests
cover PCAP hashes/manifests and refuse incomplete training labels.

## Training gate

A trainable PCAP dataset needs at least three isolated environments and all five
risk labels in the training partition. Calibration and test environments remain
held out. The current single valid-TLS smoke capture is integration evidence,
not a training dataset.

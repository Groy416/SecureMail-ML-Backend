# Profile-driven synthetic mail lab

This directory contains the isolated Docker lab used to produce packet-backed SMTP, IMAP, and POP3 STARTTLS observations. It is a capture fixture, not a production mail deployment and not a complete training-matrix generator.

## Run one scenario

From the repository root:

```bash
uv run python -m datasets.lab.runner \
  --scenario normal_tls13_valid \
  --protocol SMTP \
  --environment lab_seed_0001 \
  --seed 420042
```

The latest committed runner accepts `SMTP` and `IMAP`. POP3 support is implemented in `client.py`, `scenarios.py`, Compose, and the passive parser; the current working tree also contains a pending one-line runner change that exposes `--protocol POP3`. The same working tree adds a preliminary `legacy-lab` Compose profile, `Dockerfile.legacy`, and runner service selection/capture path; these follow-up changes are not yet committed or covered by the integration suite.

The runner resolves a protocol-specific `lab-runtime-profile.v1`, derives a stable seed, hashes the profile, creates `datasets/lab/runs/<profile-hash>/`, starts the Compose services, runs the client, tears the project down, and writes a status manifest. Use `--repetition N` to derive another profile hash and `--root PATH` to change the run root.

Current internal ports are:

- SMTP: 25
- IMAP: 143
- POP3: 110

## Compose topology

```text
cert-init + account-init
          │
          ├── mail-core (digest-pinned Docker Mailserver)
          │       └── capture sidecar
          └── legacy-lab (optional legacy OpenSSL profile)
                  └── runner-started tcpdump capture

client (SMTP/IMAP/POP3 STARTTLS) connects to the selected service
```

- `cert-init` provisions a run-scoped private CA and server certificate.
- `account-init` creates only the synthetic `lab@mail-core.lab.test` account required by Docker Mailserver setup.
- `mail-core` runs the pinned Docker Mailserver image with Postfix/Dovecot and POP3 enabled.
- `capture` shares the mail service network namespace and captures TCP ports 25, 143, and 110.
- `client` selects the protocol and TLS behavior from `runtime_profile.json`.

The network is `internal: true` and no host ports are published. The client performs STARTTLS handshakes; it does not send message bodies or deliver external mail. Capture requires `NET_ADMIN` and `NET_RAW` inside the isolated capture container.

## Run artifacts

Successful and unsupported runs are stored under the profile hash:

```text
datasets/lab/runs/<profile-hash>/
├── runtime_profile.json
├── run_manifest.json
├── runtime_status.json            # when a service reports unsupported_in_lab
├── capture.pcap                   # success only
├── ca.pem
├── tls.keys
├── certs/                         # generated server certificate/key
└── mail-{config,data,state}/      # Docker Mailserver runtime state
```

`run_manifest.json` contains the `lab-run.v1` status, detail, PCAP SHA-256 when present, runtime profile, platform, and locally inspected image IDs. `finalize_run` turns a missing/empty successful PCAP into `failed`. Profiles rejected by OpenSSL/TLS are recorded as `unsupported_in_lab` and must not produce a feature row.

Runs are idempotent by profile hash: an existing manifest is returned rather than overwritten. All run directories and key/capture material are ignored by Git. Remove them only when regenerating:

```bash
rm -rf datasets/lab/runs/*
```

## Scenario profiles

`scenarios.py` converts a catalog `ScenarioManifest` plus protocol, environment, seed, and repetition into an immutable runtime profile. The profile includes:

- protocol-specific destination port;
- TLS minimum and maximum versions;
- optional cipher string;
- client mode (`starttls`, `unused_starttls`, or `abort_starttls`);
- certificate mode and validity period;
- connection count and deterministic command delay; and
- `profile_sha256`, which is passed to PCAP extraction as `Provenance.parameter_hash`.

The profile resolver can represent normal TLS 1.2/1.3, STARTTLS-unused/abort behavior, certificate-chain/hostname/weak-key modes, RSA key exchange without forward secrecy, unusual ciphers, repeated connections, and renegotiation requests. The Compose file has an optional `legacy-lab` profile, and the current uncommitted runner selects it from the profile’s `service` field and starts tcpdump inside that container. This path is preliminary and not covered by the integration suite. Local OpenSSL or the selected mail server may reject TLS 1.0/1.1, 3DES, expired certificates, or renegotiation; those outcomes remain explicit `unsupported_in_lab` states.

## Extract and assemble

Only successful runs can be extracted:

```python
from datasets.lab.runner import assemble_successful_runs, extract_run

records = extract_run(run)
# Pass successful LabRun objects to assemble_successful_runs(...) when building
# a synthetic_pcap DatasetArtifact across environments.
```

`extract_run` uses the runtime profile’s scenario, environment, derived seed, parameter hash, destination port, run CA, and `mail-core` hostname. `ml.pcap.extract_sessions` then requires a client Finished message for a successful TLS handshake and preserves packet-range evidence. `assemble_successful_runs` excludes unsupported/failed runs and supplies capture hashes and scenario manifests to `ml.dataset.assemble_pcap_dataset`.

The runner handles one scenario per invocation; `scripts/train_evaluate_grouped_matrix.py` is the batch entry point. The current matrix has 35 scenario slots across 245 profiles: 105 train captures, 35 calibration captures, and 105 test captures, with 73 sessions per capture and 17,885 requested sessions total. The packet-backed MVP workflow is documented in [`docs/superpowers/plans/2026-09-04-mvp-synthetic-pcap-shadow.md`](../../docs/superpowers/plans/2026-09-04-mvp-synthetic-pcap-shadow.md).

## Direct parser fallback

`ml.pcap` prefers a host `tshark`. If it is unavailable, build the parser image expected by the adapter:

```bash
docker build -t lab-mail-lab:latest datasets/lab
```

The older `datasets/lab/captures/synthetic_mail.pcap` fixture and `entrypoint.sh` remain for compatibility with the module self-check. New captures should use `datasets.lab.runner` and the profile-scoped run directory.

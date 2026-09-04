# Full PCAP Training Matrix Implementation Plan

> **Status:** Superseded by [`2026-09-04-mvp-synthetic-pcap-shadow.md`](2026-09-04-mvp-synthetic-pcap-shadow.md). The active matrix is 35 scenario slots, 245 profiles, and 17,885 sessions.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Historical plan for producing a packet-backed SMTP/IMAP/POP3 dataset from isolated Postfix/Dovecot and legacy services, then training, calibrating, evaluating, saving, and reloading the ML bundle.

**Architecture:** The active plan uses the host runner's immutable protocol profiles and current 35-slot matrix. Compose runs `docker-mailserver` for Postfix/Dovecot normal scenarios and an isolated legacy-compatible service for refused crypto profiles. Each of 245 profiles contains 73 deterministic client sessions; successful captures are extracted, grouped, persisted, and fed into the existing model pipeline.

**Tech Stack:** Python 3.11, Pydantic, Docker Compose, Docker Mailserver (Postfix/Dovecot), OpenSSL, tcpdump, tshark, scikit-learn, XGBoost, pytest.

**Spec:** `docs/superpowers/specs/2026-09-04-full-pcap-training-matrix-design.md`

## Global Constraints

- Use only internal Docker networks, synthetic domains/accounts, and synthetic traffic. Never use external recipients, production passwords, email bodies, or public legacy endpoints.
- Pin the Docker Mailserver image digest after the initial pull and record all image digests per run.
- A weak/legacy/expired profile that cannot be observed must be `unsupported_in_lab`; it produces no session row.
- Record source type, scenario-manifest hash, runtime-profile parameter hash, capture hash, extractor version, dependency versions, and source revision when available.
- Group split isolation covers environment ID, scenario ID, parameter-combination hash, and capture ID.
- Generate exactly 245 successful captures × 73 sessions = 17,885 rows before training. Do not pad rows or copy a capture across splits.
- Evaluation is synthetic-only and must retain/report the fusion critical-recall result without test-set tuning.
- This repository has Git metadata. Preserve existing staged changes and do not claim verification that was not run.

---

## Task 1: Provenance and grouped-split metadata

**Files:**
- Modify: `ml/schema.py`
- Modify: `ml/dataset.py`
- Modify: `tests/test_pcap_dataset.py`

**Interfaces:**
- Produces: `Provenance.parameter_hash: str` for synthetic PCAP records and `SplitArtifact` validation over parameter/capture IDs.
- Consumes: profile SHA-256 from `LabRuntimeProfile`.

- [ ] Write failing tests that reject a synthetic PCAP record without `parameter_hash` and reject a split sharing either `parameter_hash` or `capture_id`.
- [ ] Run: `uv run python -m pytest tests/test_pcap_dataset.py -q`; expect the new tests to fail.
- [ ] Add `parameter_hash` to provenance, require it for `synthetic_pcap`, include it in `EXCLUDED_FROM_MODEL`, and check it in `split_dataset` and `validate_dataset_run`.
- [ ] Run the focused test file; expect PASS.

## Task 2: POP3 and batched profile resolution

**Files:**
- Modify: `datasets/lab/scenarios.py`
- Modify: `datasets/lab/runner.py`
- Create: `tests/test_lab_matrix.py`

**Interfaces:**
- Produces: `resolve_runtime_profile(..., protocol=Protocol.POP3, session_count=334)` and `build_training_matrix(seed) -> tuple[LabRuntimeProfile, ...]`.
- Matrix output: exactly 245 profiles, five labels in each of `lab_train`, `lab_calibration`, and `lab_test`, each with unique scenario/profile/capture hashes and 73 sessions.

- [ ] Write failing tests asserting POP3 maps to port 110 and that the matrix contains 245 profiles / 17,885 requested sessions with no reused scenario ID, parameter hash, or environment-scenario pair.
- [ ] Run: `uv run python -m pytest tests/test_lab_matrix.py -q`; expect FAIL.
- [ ] Add POP3 profile support, a `session_count` field replacing the fixed connection count, and the current 245-profile catalog assignment across the three environments; rotate SMTP, IMAP, and POP3.
- [ ] Run the focused tests; expect PASS.

## Task 3: Postfix/Dovecot and legacy Compose services

**Files:**
- Modify: `datasets/lab/compose.yaml`
- Create: `datasets/lab/Dockerfile.legacy`
- Create: `datasets/lab/openssl-legacy.cnf`
- Modify: `datasets/lab/runner.py`
- Modify: `datasets/lab/client.py`
- Create: `tests/test_lab_integration.py` additions

**Interfaces:**
- Produces: internal `mail-core` Postfix/Dovecot service, `legacy-lab` service, and ephemeral synthetic account provisioning.
- Consumes: profile service selection (`modern` or `legacy`), run-scoped TLS credentials, and session count.

- [ ] Write Docker-gated tests for authenticated SMTP STARTTLS, IMAP STARTTLS, and POP3 STARTTLS against `mail-core`; test legacy weak-cipher and expired-certificate profiles as either extracted success or explicit unsupported status.
- [ ] Run the integration file; expect FAIL because the service topology is absent.
- [ ] Configure a digest-pinned Docker Mailserver service with `PERMIT_DOCKER=network`, `SSL_TYPE=manual`, manual run-scoped certificates, SMTP submission, IMAP, and `ENABLE_POP3=1`; publish no host ports.
- [ ] Make the runner create a synthetic run-only account through Docker Mailserver setup, pass its credentials only as container environment values, and delete them from the persisted run manifest.
- [ ] Implement a legacy service based on the same mail stack plus `openssl-legacy.cnf`; issue historical certificates through a local OpenSSL CA config and use the legacy provider only in that internal service.
- [ ] Extend `client.py` with authenticated `poplib` behavior and deterministic command/timing variation by `(derived_seed, session_index)`; no message body is sent.
- [ ] Run the integration tests; expect PASS or explicit `unsupported_in_lab` only for profiles OpenSSL cannot expose.

## Task 4: PCAP extraction and run provenance

**Files:**
- Modify: `ml/pcap.py`
- Modify: `datasets/lab/runner.py`
- Modify: `tests/test_pcap_handshake.py`
- Modify: `tests/test_lab_integration.py`

**Interfaces:**
- Produces: POP3 records, parameter hashes in provenance, and run manifests containing scenario hash, profile hash, image digests, extractor version, dependency versions, source revision, and PCAP hash.

- [ ] Write failing POP3 extraction and run-manifest tests.
- [ ] Run the focused tests; expect FAIL.
- [ ] Decode POP3 on configured ports, preserve direct/STARTTLS evidence, pass `profile_sha256` as `parameter_hash`, and use profile scenario manifest hashes in the run manifest.
- [ ] Add `EXTRACTOR_VERSION` and deterministic source-revision lookup that returns `null` outside Git; do not error because this workspace lacks Git.
- [ ] Run focused extraction/integration tests; expect PASS.

## Task 5: Matrix generation, dataset persistence, and training

**Files:**
- Modify: `datasets/lab/runner.py`
- Modify: `ml/dataset.py` only if persistence needs the new metadata
- Create: `tests/test_full_pcap_matrix.py`
- Create: `configs/training.pcap.json`

**Interfaces:**
- Produces: `generate_training_matrix(seed, root) -> DatasetRun`, a persisted 17,885-row dataset, saved model bundle, and synthetic evaluation report.

- [ ] Write failing test with a temporary runner fixture asserting that matrix generation refuses fewer than 245 successful captures or any count other than 17,885.
- [ ] Run the test; expect FAIL.
- [ ] Implement batch execution with bounded retries (one retry per failed transient Compose run), explicit status manifests, and no retry for `unsupported_in_lab`.
- [ ] Assemble only successful records, call `split_dataset`, `write_dataset_run`, `train_model_bundle`, `calibrate_scores`, `evaluate_bundle`, and `save_model_bundle` using `configs/training.pcap.json`.
- [ ] Reject training unless all three partitions contain all five labels, capture/scenario/parameter/environment groups are disjoint, and the record count is exactly 17,885.
- [ ] Run the fixture test; expect PASS.

## Task 6: Full lab generation and verification

**Files:**
- Modify only when checks identify a defect in Tasks 1–5.

- [ ] Run: `uv run python -m pytest -q`; expect PASS.
- [ ] Run: `uv run python -m py_compile ml/*.py datasets/lab/*.py tests/test_*.py && uv lock --check`; expect PASS.
- [ ] Run the matrix command with seed `420042`; verify 245 `success` manifests, 17,885 rows, checksums, no split leakage, saved bundle reload, and evaluation report.
- [ ] Inspect metrics; report the synthetic-only result and the fusion critical-recall value without tuning on the test partition.

## Plan self-review

- Spec coverage: Postfix/Dovecot, POP3, legacy handling, all required provenance, grouped splitting, 5,000+ size, packet-backed extraction, and training/evaluation are assigned.
- Placeholder scan: each task names files, tests, expected failure, implementation boundary, and verification.
- Type consistency: runtime profile hashes become `Provenance.parameter_hash`; `LabRun` feeds PCAP extraction; extracted rows feed the existing PCAP dataset assembler; the persisted split feeds the existing train/calibrate/evaluate/save sequence.

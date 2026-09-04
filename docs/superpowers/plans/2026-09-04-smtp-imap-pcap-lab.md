# SMTP/IMAP PCAP Scenario Matrix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate manifest-labeled SMTP and IMAP PCAP runs that can be extracted into `synthetic_pcap` records for model training.

**Architecture:** A host-side resolver converts a `ScenarioManifest` plus protocol and seed into a hashed runtime profile. Docker services consume only that JSON profile and write a PCAP, lab-only key log, and status metadata to one run directory. A host-side runner owns Compose execution, explicit unsupported outcomes, PCAP extraction, and dataset-ready manifests.

**Tech Stack:** Python 3.11 stdlib, Pydantic models already in `ml.schema`, Docker Compose, OpenSSL, tcpdump, tshark, pytest.

**Spec:** `docs/superpowers/specs/2026-09-04-smtp-imap-pcap-lab-design.md`

## Global Constraints

- Docker network is internal-only; do not publish ports or use real mailbox credentials, email bodies, or external recipients.
- Keep headless SMTP/IMAP clients as the generator; Roundcube is not part of this scenario matrix.
- Every successful record uses `SourceType.SYNTHETIC_PCAP` and PCAP evidence ranges.
- Labels are copied from `ScenarioManifest`; packet parsing must not assign labels.
- A requested crypto setting that the container rejects must result in `unsupported_in_lab`, never a synthetic substitute row.
- Keep `datasets/lab/captures/` ignored; TLS key logs are lab-only and never production input.
- Run tests with `uv run python -m pytest`, not the `pytest` console script.
- This directory has no Git metadata. Do not initialize Git, commit, or claim a Git-based code review.

---

## File map

- Create `datasets/lab/scenarios.py`: runtime-profile contract and deterministic manifest-to-profile resolver.
- Create `datasets/lab/runner.py`: one-run Compose orchestration, artifact hashes, status JSON, and extraction handoff.
- Modify `datasets/lab/compose.yaml`: parameterize the host run directory and PCAP path; retain internal network/no ports.
- Modify `datasets/lab/entrypoint.sh`: capture the configured SMTP/IMAP ports and preserve a clean PCAP on shutdown.
- Modify `datasets/lab/server.py`: provision profile-specific CA/certificate/TLS contexts and expose the requested STARTTLS behavior.
- Modify `datasets/lab/client.py`: produce profile-specific SMTP/IMAP behavior, including unused STARTTLS and intentionally aborted negotiation.
- Modify `ml/pcap.py`: accept implicit TLS and completed/failed STARTTLS flows without inventing certificate trust fields.
- Create `tests/test_lab_scenarios.py`: resolver and status-file unit tests.
- Create `tests/test_lab_integration.py`: Docker-gated manifest → PCAP → extracted-record tests for one SMTP and one IMAP valid-TLS run.
- Modify `datasets/lab/README.md`: exact scenario-runner command, artifact layout, and unsupported status semantics.

## Task 1: Deterministic runtime profiles

**Files:**
- Create: `datasets/lab/scenarios.py`
- Test: `tests/test_lab_scenarios.py`

**Interfaces:**
- Consumes: `ml.schema.ScenarioManifest`, `ml.schema.Protocol`, `ml.dataset.scenario_seed`.
- Produces: `LabRuntimeProfile`, `resolve_runtime_profile(manifest, protocol, environment_id, master_seed, repetition_index) -> LabRuntimeProfile`.
- Ruling: resolution copies the catalog manifest with `protocol` set to the selected protocol and `scenario_id` suffixed `-smtp` or `-imap`, because `DatasetArtifact` rejects a record whose protocol differs from its manifest and grouped splits reject scenario reuse.
- Used by: `datasets/lab/runner.py`, server, client, and integration tests.

- [ ] **Step 1: Write the failing profile determinism test**

```python
from datasets.lab.scenarios import resolve_runtime_profile
from ml.dataset import CATALOG
from ml.schema import Protocol, ScenarioManifest


def test_runtime_profile_is_seeded_and_hashable() -> None:
    manifest = ScenarioManifest.model_validate(CATALOG[0]["manifest"])
    first = resolve_runtime_profile(manifest, Protocol.SMTP, "lab_seed_0001", 7, 0)
    second = resolve_runtime_profile(manifest, Protocol.SMTP, "lab_seed_0001", 7, 0)

    assert first.profile_sha256 == second.profile_sha256
    assert first.destination_port == 2525
    assert first.client_mode == "starttls"
    assert first.tls_maximum_version == "TLS1.3"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/test_lab_scenarios.py::test_runtime_profile_is_seeded_and_hashable -q`

Expected: FAIL because `datasets.lab.scenarios` does not exist.

- [ ] **Step 3: Write the minimal profile contract and resolver**

```python
class LabRuntimeProfile(ContractModel):
    scenario: ScenarioManifest
    protocol: Protocol
    environment_id: str
    derived_seed: int
    destination_port: int
    client_mode: Literal["starttls", "unused_starttls", "abort_starttls"]
    starttls_advertised: bool
    starttls_accepted: bool
    tls_minimum_version: str
    tls_maximum_version: str
    cipher_string: str | None
    certificate_mode: Literal["valid", "expired", "unknown_ca", "hostname_mismatch", "weak_rsa"]
    connection_count: int = Field(gt=0)
    profile_sha256: str


def resolve_runtime_profile(...)-> LabRuntimeProfile:
    # Copy the manifest with a protocol-specific ID before deriving profile values.
    # Derive profile values only from that manifest and scenario_seed.
    # Select 2525 for SMTP and 1143 for IMAP.
    # Map STARTTLS and behavioral scenario IDs to client_mode/connection_count.
    # Use TLS 1.2 for certificate-failure profiles and explicit legacy profiles.
```

`profile_sha256` is the SHA-256 of the profile payload without its own hash,
serialized with sorted keys and compact separators. Unsupported status is not
resolved here because it depends on the container crypto runtime.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m pytest tests/test_lab_scenarios.py::test_runtime_profile_is_seeded_and_hashable -q`

Expected: PASS.

## Task 2: Profile-driven mail service and client

**Files:**
- Modify: `datasets/lab/compose.yaml`
- Modify: `datasets/lab/entrypoint.sh`
- Modify: `datasets/lab/server.py`
- Modify: `datasets/lab/client.py`
- Test: `tests/test_lab_scenarios.py`

**Interfaces:**
- Consumes: `/captures/runtime_profile.json` serialized from `LabRuntimeProfile`.
- Produces: a PCAP at `/captures/capture.pcap`, local CA at `/captures/ca.pem`, and optional `/captures/tls.keys`.
- Used by: `datasets/lab/runner.py`.

- [ ] **Step 1: Write the failing profile-load test**

```python
from datasets.lab.server import load_runtime_profile


def test_load_runtime_profile_rejects_a_missing_profile(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_runtime_profile(tmp_path / "runtime_profile.json")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/test_lab_scenarios.py::test_load_runtime_profile_rejects_a_missing_profile -q`

Expected: FAIL because `load_runtime_profile` does not exist.

- [ ] **Step 3: Implement profile loading and runtime behavior**

```python
def load_runtime_profile(path: str | Path) -> LabRuntimeProfile:
    return LabRuntimeProfile.model_validate_json(Path(path).read_text())


def tls_context(profile: LabRuntimeProfile) -> ssl.SSLContext:
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.minimum_version = TLS_VERSION[profile.tls_minimum_version]
    context.maximum_version = TLS_VERSION[profile.tls_maximum_version]
    if profile.cipher_string:
        context.set_ciphers(profile.cipher_string)
    context.load_cert_chain(CERTIFICATE, PRIVATE_KEY)
    return context
```

- Provision a valid private CA/server SAN pair for `valid`.
- For `unknown_ca`, sign the server certificate with an untrusted synthetic CA.
- For `hostname_mismatch`, issue a CA-signed synthetic certificate without the `mail-lab` SAN.
- For `expired` and `weak_rsa`, request the configured OpenSSL certificate. On command or TLS-context failure, write `unsupported_in_lab` rather than starting the service.
- Have SMTP omit `250-STARTTLS` for `starttls_advertised=False`; reject an accepted upgrade only when the profile requests failure.
- Have the client use `client_mode`: negotiate, leave STARTTLS unused, or issue STARTTLS then close before TLS bytes. `connection_count` repeats the same behavior without mail content.
- Change Compose bind mounts to `${LAB_RUN_DIR:-./captures}:/captures`; preserve the internal-only network and no host ports.
- Capture both lab service ports into `${PCAP_PATH:-/captures/capture.pcap}` and wait one second before client exit so tcpdump drains.

- [ ] **Step 4: Run profile unit tests to verify they pass**

Run: `uv run python -m pytest tests/test_lab_scenarios.py -q`

Expected: PASS.

## Task 3: Explicit Compose run orchestration

**Files:**
- Create: `datasets/lab/runner.py`
- Test: `tests/test_lab_scenarios.py`

**Interfaces:**
- Consumes: `LabRuntimeProfile`, `datasets/lab/compose.yaml`.
- Produces: `LabRun` with `status` in `success`, `unsupported_in_lab`, or `failed`; `run_manifest.json`; PCAP SHA-256 only for success.
- Used by: integration test and future batch generation command.

- [ ] **Step 1: Write the failing unsupported-status test**

```python
from datasets.lab.runner import finalize_run


def test_finalize_run_marks_missing_pcap_unsupported(tmp_path) -> None:
    result = finalize_run(tmp_path, "unsupported_in_lab", "cipher rejected")

    assert result.status == "unsupported_in_lab"
    assert result.pcap_sha256 is None
    assert (tmp_path / "run_manifest.json").is_file()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/test_lab_scenarios.py::test_finalize_run_marks_missing_pcap_unsupported -q`

Expected: FAIL because `datasets.lab.runner` does not exist.

- [ ] **Step 3: Implement runner and status manifest**

```python
@dataclass(frozen=True)
class LabRun:
    path: Path
    status: Literal["success", "unsupported_in_lab", "failed"]
    pcap_sha256: str | None


def run_scenario(profile: LabRuntimeProfile, root: str | Path) -> LabRun:
    # Create root/<profile_sha256>/runtime_profile.json atomically.
    # Invoke docker compose with LAB_RUN_DIR and PCAP_PATH in its environment.
    # Read service status JSON; hash capture.pcap only when it is nonempty.
    # Return unsupported only for recorded OpenSSL/TLS configuration refusal.
```

`run_manifest.json` must include schema version, scenario manifest, profile
hash, environment ID, derived seed, status, error text when present, PCAP hash,
platform, and `docker image inspect` IDs when Docker is available. It must not
include the key-log contents or credential fields.

- [ ] **Step 4: Run runner unit tests to verify they pass**

Run: `uv run python -m pytest tests/test_lab_scenarios.py -q`

Expected: PASS.

## Task 4: Extraction handoff and integration coverage

**Files:**
- Modify: `ml/pcap.py`
- Create: `tests/test_lab_integration.py`

**Interfaces:**
- Consumes: `LabRun`, `LabRuntimeProfile`, `ml.pcap.extract_sessions`.
- Produces: validated `SessionFeatureRecord` values with `synthetic_pcap` provenance and packet-range evidence.
- Used by: `ml.dataset.assemble_pcap_dataset`.

- [ ] **Step 1: Write the failing Docker-gated SMTP/IMAP tests**

```python
@pytest.mark.parametrize("protocol", [Protocol.SMTP, Protocol.IMAP])
def test_valid_profile_runs_to_packet_backed_session(tmp_path, protocol) -> None:
    run = run_scenario(valid_profile(protocol), tmp_path)
    assert run.status == "success"
    records = extract_run(run)
    assert len(records) == 1
    assert records[0].features.protocol is protocol
    assert records[0].provenance.source_type is SourceType.SYNTHETIC_PCAP
    assert records[0].provenance.evidence_refs[0].packet_start is not None
```

Skip with `pytest.skip` only when `docker info` fails; do not skip a configured
Docker run that fails.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/test_lab_integration.py -q`

Expected: FAIL because `run_scenario` and `extract_run` are missing.

- [ ] **Step 3: Implement extraction handoff**

```python
def extract_run(run: LabRun) -> list[SessionFeatureRecord]:
    # Require status == "success" and load runtime_profile.json.
    # Call extract_sessions(capture.pcap, scenario=profile.scenario,
    # environment_id=profile.environment_id, generator_seed=profile.derived_seed,
    # destination_port=profile.destination_port,
    # trusted_ca_path=ca.pem only for certificate_mode == "valid",
    # expected_hostname="mail-lab" only for certificate_mode == "valid").
```

Update `extract_sessions` so direct SMTPS/IMAPS flows are represented with
`starttls_advertised=False` and `starttls_used=False`, while failed STARTTLS
flows retain null negotiated TLS fields and packet evidence. Keep certificate
trust fields `None` unless the runner supplies a trusted CA and expected
hostname.

- [ ] **Step 4: Run integration tests to verify they pass**

Run: `uv run python -m pytest tests/test_lab_integration.py -q`

Expected: PASS for SMTP and IMAP on a running local Docker daemon.

## Task 5: Dataset-ready matrix API and documentation

**Files:**
- Modify: `datasets/lab/runner.py`
- Modify: `datasets/lab/README.md`
- Test: `tests/test_lab_scenarios.py`

**Interfaces:**
- Consumes: successful `LabRun` values.
- Produces: `assemble_successful_runs(runs, config) -> DatasetArtifact`.
- Used by: the next training command once the matrix has sufficient successful environments and labels.

- [ ] **Step 1: Write the failing successful-run assembly test**

```python
def test_assemble_successful_runs_excludes_unsupported_runs(tmp_path) -> None:
    dataset = assemble_successful_runs(
        [successful_run(tmp_path), unsupported_run(tmp_path)],
        pcap_dataset_config(),
    )

    assert dataset.mode == "synthetic_pcap"
    assert dataset.pcap_sha256 == {"capture-success": SUCCESS_HASH}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/test_lab_scenarios.py::test_assemble_successful_runs_excludes_unsupported_runs -q`

Expected: FAIL because `assemble_successful_runs` does not exist.

- [ ] **Step 3: Implement matrix assembly and usage documentation**

```python
def assemble_successful_runs(
    runs: Sequence[LabRun],
    config: DatasetConfig | Mapping[str, Any],
) -> DatasetArtifact:
    # Extract only status == "success" runs.
    # Reject an empty success set.
    # Set config.session_count to the successful extracted-record count.
    # Pass extracted records, profiles' ScenarioManifests, and PCAP hashes to
    # assemble_pcap_dataset; never add unsupported or failed runs as rows.
```

Document these commands:

```bash
uv run python -m datasets.lab.runner --scenario normal_tls13_valid --protocol SMTP --environment lab_seed_0001 --seed 420042
uv run python -m datasets.lab.runner --scenario normal_tls13_valid --protocol IMAP --environment lab_seed_0001 --seed 420042
```

Document the artifact files, the distinction between `success`,
`unsupported_in_lab`, and `failed`, and the requirement to build a full
three-environment, five-label matrix before invoking `train_model_bundle`.

- [ ] **Step 4: Run matrix unit tests to verify they pass**

Run: `uv run python -m pytest tests/test_lab_scenarios.py -q`

Expected: PASS.

## Task 6: Full verification

**Files:**
- Modify only if a failed check identifies a defect in Tasks 1–5.

- [ ] **Step 1: Run focused tests**

Run:

```bash
uv run python -m pytest tests/test_lab_scenarios.py tests/test_lab_integration.py tests/test_pcap_dataset.py -q
```

Expected: PASS; Docker integration tests may be skipped only if the daemon is unavailable.

- [ ] **Step 2: Run existing module contracts**

Run:

```bash
uv run python -m ml.dataset
uv run python -m ml.models
uv run python -m ml.pipeline
uv run python -m ml.pcap
```

Expected: each exits 0.

- [ ] **Step 3: Run full suite and syntax/lock checks**

Run:

```bash
uv run python -m pytest -q
uv run python -m py_compile ml/*.py datasets/lab/*.py tests/test_*.py
uv lock --check
```

Expected: all commands exit 0.

- [ ] **Step 4: Inspect changed files without Git**

Run:

```bash
find datasets/lab ml tests docs/superpowers -type f -newer docs/superpowers/specs/2026-09-04-smtp-imap-pcap-lab-design.md -print
```

Expected: only the planned lab, parser, dataset, test, and documentation paths are listed.

## Plan self-review

- Spec coverage: Tasks 1–5 cover manifest-driven profiles, isolated Compose execution, explicit unsupported outcomes, packet-backed extraction, scenario labels, dataset handoff, and SMTP/IMAP integration. Roundcube and POP3 remain intentionally out of scope.
- Placeholder scan: all tasks specify files, test names, commands, interfaces, and expected outcomes. No unresolved implementation markers are present.
- Type consistency: `LabRuntimeProfile` is the profile passed into `run_scenario`; it contains the protocol-specific `ScenarioManifest`. `LabRun` is consumed by `extract_run` and `assemble_successful_runs`; `DatasetArtifact` is created only through the existing `assemble_pcap_dataset` contract.

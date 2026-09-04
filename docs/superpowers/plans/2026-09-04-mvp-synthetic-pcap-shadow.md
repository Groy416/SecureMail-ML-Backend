# SecureMailScope MVP Synthetic PCAP Shadow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the current 35-slot synthetic packet lab reproducibly train and evaluate a shadow-scoring bundle, correct Isolation Forest policy behavior, and verify the session-feature-only production boundary.

**Architecture:** `datasets.lab.matrix.build_training_matrix()` remains the deterministic profile catalog. The lab runner executes isolated Docker profiles and `ml.pcap.extract_sessions()` converts successful PCAPs into `SessionFeatureRecord` rows. The persisted packet-backed dataset is split by environment, scenario, parameter hash, and capture; XGBoost, Random Forest, and normal-only Isolation Forest are trained offline, calibrated on `lab_calibration`, evaluated on `lab_test`, and saved as the versioned `grouped-105-capture-pcap` shadow bundle. `ml.product` continues to accept validated session-feature JSONL only; packet capture, message content, credentials, and online training remain outside the scorer.

**Tech Stack:** Python 3.11, `uv`, Pydantic, pandas/PyArrow, scikit-learn, XGBoost, joblib, pytest, Docker Compose, tcpdump, tshark.

**Spec:** `docs/specs/spec.md`

**Supersedes:** The stale 15-capture/5,010-row assumptions in `docs/superpowers/specs/2026-09-04-full-pcap-training-matrix-design.md`, `docs/superpowers/plans/2026-09-04-full-pcap-training-matrix.md`, and related README text.

## Global Constraints

- Production MVP is shadow-only: no blocking, quarantine, configuration changes, or remediation.
- Training data is synthetic and packet-backed; no authorized real/live data is available for this run.
- The current matrix is authoritative: 35 scenario slots, 3 train rotations, 1 calibration rotation, 3 test rotations, 73 sessions per capture, 245 profiles, and 17,885 records when complete.
- The full gate is 105 successful `lab_train` captures, 35 successful `lab_calibration` captures, 105 successful `lab_test` captures, and exactly 17,885 extracted records.
- Every successful capture must yield exactly 73 extracted sessions; unsupported profiles are recorded as `unsupported_in_lab` and are never padded, copied, or relabeled.
- All train/calibration/test groups remain disjoint by environment, scenario ID, parameter hash, and capture ID.
- Isolation Forest is fit only on normal training rows. An IF-only flag creates `analyst_review`; it never creates a critical finding.
- Calibration and fusion weights are chosen from the calibration partition only. The test partition is read only for final evaluation.
- `ml.product` and `ml.pipeline.predict_session` consume session-feature records only. No product path accepts PCAP bytes, email bodies, credentials, cookies, key logs, or private keys.
- Synthetic metrics are reported as synthetic-only and are not production-performance claims.
- Retraining is offline, versioned, and manually/CI approved; no online learning or automatic live-data promotion.
- Existing staged changes are preserved. Do not run reset, clean, checkout, or destructive deletion against the working tree or existing generated artifacts.

---

### Task 1: Reconcile the 35-slot matrix contract

**Files:**
- Modify: `datasets/lab/matrix.py`
- Test: `tests/test_lab_matrix.py`
- Modify: `README.md`
- Modify: `datasets/lab/README.md`
- Modify: `scripts/evaluate_pcap_bundle.py`
- Modify: `docs/superpowers/specs/2026-09-04-full-pcap-training-matrix-design.md`
- Modify: `docs/superpowers/plans/2026-09-04-full-pcap-training-matrix.md`

**Interfaces:**
- Consumes: `build_training_matrix(master_seed: int)` and current `MATRIX_SLOTS`.
- Produces: deterministic counts of 35 slots, 245 profiles, and 17,885 requested sessions.

- [ ] **Step 1: Record the current baseline without editing staged work.**

Run:

```bash
uv run python - <<'PY'
from datasets.lab.matrix import MATRIX_CAPTURES, MATRIX_SESSIONS, MATRIX_SLOTS, build_training_matrix
profiles = build_training_matrix(420042)
print(len(MATRIX_SLOTS), len(profiles), MATRIX_CAPTURES, MATRIX_SESSIONS)
PY
```

Expected baseline: `35 245 245 17885`.

- [ ] **Step 2: Add the matrix contract assertions.**

```python
def test_matrix_uses_the_current_35_slot_contract() -> None:
    profiles = build_training_matrix(420042)

    assert len(MATRIX_SLOTS) == 35
    assert len(profiles) == 245
    assert MATRIX_SESSIONS == 17_885
    assert len({profile.profile_sha256 for profile in profiles}) == 245
    assert len({profile.scenario.scenario_id for profile in profiles}) == 245
    assert {profile.protocol for profile in profiles} == set(Protocol)
```

Also assert that calling `build_training_matrix(420043)` changes profile hashes while calling it twice with `420042` produces identical hashes.

- [ ] **Step 3: Make the function argument real.** Remove the `del master_seed` behavior and derive the three partition seeds from the supplied master seed while preserving the existing `420042` output: train uses `master_seed`, calibration uses `master_seed + 17`, and test uses `master_seed + 1`.

- [ ] **Step 4: Replace every stale 15/5,010/69 reference.** Document 35 slots, 245 profiles, 17,885 sessions, and the distinction between 105 training captures and 245 total captures. Keep `grouped-105-capture` because it names the training partition.

- [ ] **Step 5: Run the focused matrix checks.**

```bash
uv run pytest tests/test_lab_matrix.py -q
```

Expected: PASS with no reference to the obsolete 15-capture target.

---

### Task 2: Complete packet-backed matrix execution and provenance

**Files:**
- Modify: `datasets/lab/runner.py`
- Modify: `datasets/lab/client.py`
- Modify: `datasets/lab/compose.yaml`
- Modify: `ml/pcap.py`
- Modify: `ml/dataset.py`
- Test: `tests/test_full_pcap_matrix.py`
- Modify: `tests/test_lab_integration.py`
- Modify: `tests/test_lab_scenarios.py`
- Modify: `tests/test_pcap_dataset.py`

**Interfaces:**
- Consumes: `build_training_matrix()`, `run_scenario()`, and `extract_sessions()`.
- Produces:
  - `run_training_matrix(master_seed: int, root: str | Path) -> tuple[LabRun, ...]`
  - `validate_training_matrix(runs: Sequence[LabRun], dataset: DatasetArtifact) -> None`
  - run manifests containing profile/scenario/capture/extractor/dependency provenance.

- [ ] **Step 1: Add unit tests for the completion gate before changing orchestration.**

```python
def test_training_matrix_gate_rejects_an_incomplete_profile_list(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("datasets.lab.runner.build_training_matrix", lambda seed: ())

    with pytest.raises(ValueError, match="245 profiles"):
        run_training_matrix(420042, tmp_path)
```

Add a second no-Docker test that passes one `LabRun(status="success")` whose profile expects 73 sessions and whose extracted record list contains 72 records; `validate_training_matrix()` must raise with `73 extracted sessions` in the message.

- [ ] **Step 2: Add `run_training_matrix()`.** Iterate the 245 profiles in deterministic order, call `run_scenario()` once, retry a `failed` run at most once under a separate retry directory, and never retry `unsupported_in_lab`. Return every status so the caller can report failures instead of silently dropping them.

- [ ] **Step 3: Add `validate_training_matrix()`.** Require exactly 245 profiles, 245 successful runs, 17,885 records, 73 records per successful capture, all five risk labels in each partition, and disjoint environment/scenario/parameter/capture groups. Raise a specific `ValueError` naming the first failed gate.

- [ ] **Step 4: Harden extraction and run manifests.** Add `EXTRACTOR_VERSION`, preserve `profile_sha256` as `parameter_hash`, hash the scenario manifest, record the PCAP SHA-256, dependency versions, Docker image IDs/digests, and a nullable source revision. A source-revision lookup returns `None` when Git is unavailable. Do not persist the synthetic account password or any raw payload.

- [ ] **Step 5: Verify protocol coverage.** Keep SMTP/IMAP/POP3 rotation, ensure POP3 uses port 110 and the `poplib` STARTTLS path, and ensure each client variation is deterministic from `(derived_seed, session_index)`. Add Docker-gated tests for one valid SMTP, IMAP, and POP3 profile; weak/legacy profiles must be either packet-backed or explicitly `unsupported_in_lab`.

- [ ] **Step 6: Run focused non-Docker checks.**

```bash
uv run pytest tests/test_pcap_dataset.py tests/test_lab_scenarios.py tests/test_lab_matrix.py tests/test_full_pcap_matrix.py -q
uv run python -m py_compile ml/*.py datasets/lab/*.py tests/test_*.py
```

Expected: PASS; Docker-gated cases may skip only when the daemon is unavailable.

---

### Task 3: Correct Isolation Forest policy and calibration behavior

**Files:**
- Modify: `ml/fusion.py`
- Modify: `ml/calibration.py`
- Test: `tests/test_fusion.py`
- Modify: `tests/test_calibration.py`

**Interfaces:**
- Consumes: `ModelOutputs`, `CalibrationState`, and normal-only IF scores.
- Produces: an `MLResult` with normalized anomaly score, threshold, `flagged`, baseline ID, and explicit analyst-review behavior.

- [ ] **Step 1: Change the failing policy test.** In the existing `test_mail_check_risk_ignores_unusable_isolation_forest` fixture, keep the current record/output/calibration setup and change the final assertions to:

```python
assert result.anomaly.detected is True
assert result.model_outputs["isolation_forest"]["flagged"] is True
assert result.risk.risk_class is RiskLabel.LOW
assert result.action is ModelAction.ANALYST_REVIEW
```

- [ ] **Step 2: Implement the baseline fusion contract.** Set default weights to XGBoost `0.50`, Random Forest `0.30`, and Isolation Forest `0.20`. Keep the raw IF score separate from normalized anomaly score; do not treat the anomaly score as a class probability.

- [ ] **Step 3: Implement policy precedence.** An IF-only flag changes the action to `analyst_review` but cannot raise the risk class above the configured model-fusion result. A deterministic critical finding still returns `critical_review` even when IF is unavailable. A degenerate normal calibration records `unavailable:degenerate_normal_calibration` and contributes zero to fusion.

- [ ] **Step 4: Add calibration boundary tests.** Verify that normal-only training rows are the only rows passed to `IsolationForest.fit`, that the normal percentile range is persisted, and that normalized scores and thresholds survive save/load unchanged.

- [ ] **Step 5: Run the focused policy checks.**

```bash
uv run pytest tests/test_fusion.py tests/test_calibration.py -q
```

Expected: PASS with explicit IF-only analyst review and no critical escalation.

---

### Task 4: Make evaluation compare models instead of hiding them behind rules

**Files:**
- Modify: `ml/evaluate.py`
- Modify: `ml/ablation.py`
- Modify: `ml/dataset.py`
- Test: `tests/test_evaluation.py`
- Create: `configs/training.pcap.json`

**Interfaces:**
- Consumes: persisted grouped packet-backed splits and validation-fitted calibration.
- Produces: an `EvaluationReport` containing individual classifier metrics, model-only fusion, rule-backed policy fusion, IF metrics/status, and per-scenario results.

- [ ] **Step 1: Add a report test that distinguishes model-only from rule-backed fusion.** The test must use a small deterministic split and assert that `report.model_fusion` is computed with no deterministic findings while `report.fusion` retains the existing rule-backed result.

- [ ] **Step 2: Extend `EvaluationReport` with `model_fusion` and `scenario_breakdown`.** Preserve the existing `fusion` field for policy results. For each exact `scenario_id`, report session count, true label counts, model-only accuracy, policy accuracy, critical recall when present, and IF anomaly detection rate.

- [ ] **Step 3: Add complete IF reporting.** Include status, raw normal/anomalous score distributions, normalized threshold, contamination configuration, precision, recall, F1, and normal false-positive rate. Use the explicit degraded status when calibration is degenerate.

- [ ] **Step 4: Add `load_dataset_run(path: str | Path) -> tuple[DatasetArtifact, SplitArtifact]`.** Read `extracted/session_features.jsonl` and the three Parquet split files, validate every record, verify the dataset and split hashes, reconstruct the manifests and PCAP hash map from the run files, and return the same `DatasetArtifact`/`SplitArtifact` contracts used by training. Evaluation scripts must consume this persisted packet-backed run rather than regenerate feature-only rows.

- [ ] **Step 5: Add the training configuration file.** Store the existing XGBoost/Random Forest/Isolation Forest settings, seed `420042`, MinMax/encoding settings, and fusion weights `0.50/0.30/0.20`. Parse the file with the existing Pydantic model contracts; do not add a configuration framework.

```json
{
  "random_seed": 420042,
  "xgboost_n_estimators": 300,
  "xgboost_max_depth": 5,
  "xgboost_learning_rate": 0.05,
  "xgboost_subsample": 0.8,
  "xgboost_colsample_bytree": 0.8,
  "random_forest_n_estimators": 300,
  "random_forest_min_samples_leaf": 2,
  "n_jobs": 1,
  "isolation_contamination": "auto",
  "fusion": {
    "xgboost_weight": 0.5,
    "random_forest_weight": 0.3,
    "isolation_forest_weight": 0.2
  }
}
```

- [ ] **Step 6: Run evaluation tests.**

```bash
uv run pytest tests/test_evaluation.py tests/test_calibration.py -q
```

Expected: PASS; no test row is used for threshold or fusion-weight tuning.

---

### Task 5: Train and verify the session-feature-only shadow artifact

**Files:**
- Modify: `scripts/train_evaluate_grouped_matrix.py`
- Modify: `scripts/evaluate_pcap_bundle.py`
- Modify: `ml/product.py`
- Test: `tests/test_product.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: `run_training_matrix()`, `assemble_successful_runs()`, `split_dataset()`, and `configs/training.pcap.json`.
- Produces: `datasets/runs/<run_id>`, `models/grouped-105-capture-pcap`, a synthetic-only evaluation report, and a JSONL shadow scoring command.

- [ ] **Step 1: Add a script-level gate test using fake `LabRun` results.** The script must refuse to train unless all 245 captures succeeded, all 17,885 rows were extracted, and each partition contains all five labels.

- [ ] **Step 2: Replace feature-only generation in the training script.** Build the 245 profiles, run the packet lab, assemble only successful records, split with `lab_calibration` as validation and `lab_test` as test, persist the dataset, train, calibrate, evaluate, and save.

- [ ] **Step 3: Make artifact writes non-destructive.** Do not remove an existing bundle directory. Write the packet-backed artifact to `models/grouped-105-capture-pcap`; if that path exists, verify its checksums and bundle ID match the deterministic training result, otherwise fail with a conflict. Leave the existing `models/grouped-105-capture` artifact untouched as a rollback candidate. Reuse identical dataset/evaluation directories idempotently.

- [ ] **Step 4: Keep the product path feature-only.** Preserve `ml.product` JSONL input validation and forbidden-key checks. Add a test asserting an authorized payload produces anomaly fields and model diagnostics without accepting PCAP, body, password, token, or key-log fields.

- [ ] **Step 5: Update the evaluation script to load the persisted packet-backed run and bundle.** It must not call `generate_grouped_feature_dataset()` for the production benchmark.

- [ ] **Step 6: Run the product checks.**

```bash
uv run pytest tests/test_product.py tests/test_fusion.py -q
```

Expected: PASS; missing/corrupt model artifacts fail closed with a visible nonzero error rather than a confident result.

---

### Task 6: One full verification pass and release-readiness report

**Files:**
- Modify only files identified by the preceding checks.
- Create/update: `evals/evaluation-*/metrics.json` only for the verified deterministic run.

**Interfaces:**
- Consumes: the completed packet-backed run, bundle, and evaluation report.
- Produces: evidence for the synthetic shadow MVP and an explicit list of unverified external deployment items.

- [ ] **Step 1: Run the whole repository checks once.**

```bash
uv run pytest -q
uv run python -m py_compile ml/*.py datasets/lab/*.py tests/test_*.py
uv lock --check
git diff --check
```

Expected: all tests pass, compilation succeeds, the lockfile is valid, and no whitespace errors remain in the final diff.

- [ ] **Step 2: Run the full synthetic matrix only when Docker is available.**

```bash
uv run python scripts/train_evaluate_grouped_matrix.py --seed 420042 --root datasets/lab/runs
```

Verify with a read-only check that there are 245 profile manifests, 245 successful runs, 17,885 records, valid checksums, disjoint groups, a reloadable `grouped-105-capture-pcap` bundle, and an evaluation report containing model-only fusion, policy fusion, IF metrics, and scenario breakdown.

- [ ] **Step 3: Run the shadow scorer against an authorized-shaped fixture.**

```bash
uv run python -m ml.product \
  --bundle models/grouped-105-capture-pcap \
  --input sessions.jsonl \
  --output results.jsonl
```

Verify that the result contains `risk`, `action`, `anomaly`, `model_outputs`, `rule_findings`, and diagnostics, and that the input contains no packet bytes or sensitive content.

- [ ] **Step 4: Report the result accurately.** State the exact changed paths and commands, report synthetic metrics separately, include fusion critical recall and IF false-positive rate, state whether Docker was available, and explicitly leave real/live validation and external deployment unverified because no authorized live data or deployment target was provided.

- [ ] **Step 5: Stop at the local artifact boundary.** Do not deploy externally, add an HTTP framework, enable online retraining, or promote synthetic rows as production evidence without a separately approved target and authorized data contract.

## Acceptance Check

The MVP is ready for internal shadow use only when the 35-slot packet-backed matrix completes with 245 successful captures and 17,885 rows, grouped leakage checks pass, all five labels exist in every partition, IF-only anomalies produce analyst review, model-only and rule-backed evaluations are both present, the versioned packet-backed bundle reloads by checksum, and `ml.product` scores session-feature JSONL without accepting sensitive payloads. Synthetic results must remain explicitly non-production evidence.

## Deferred by decision

- Authorized real/live capture validation and any real-data training promotion.
- External deployment wiring because no deployment target was named.
- Learned stacking, online learning, blocking/remediation, dashboard work, and raw-PCAP ingestion by the scorer.

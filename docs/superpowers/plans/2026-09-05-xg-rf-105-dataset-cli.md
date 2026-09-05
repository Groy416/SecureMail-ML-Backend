# XGBoost/Random Forest 105-Capture Training Workflow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the active packet-backed Isolation Forest workflow with calibrated 60/40 XGBoost/Random Forest fusion trained on 105 captures, persisted as `Dataset-17K` and `Model_XG_RF`, with visible CLI progress and tabular evaluation output.

**Architecture:** Keep the existing 245-capture source matrix and environment-group split: 105 train, 35 calibration, and 105 test captures. Add an explicit two-model bundle mode while retaining read compatibility for legacy bundles. The training script reuses a validated named dataset when present, otherwise captures/extracts once and writes it under the readable name; all active scoring and evaluation use calibrated XGBoost/Random Forest probability-vector fusion.

**Tech Stack:** Python 3.11, Pydantic contracts, scikit-learn, XGBoost, PyArrow, joblib, pytest, standard-library CLI formatting/timing.

**Spec:** `docs/superpowers/specs/2026-09-05-xg-rf-105-dataset-cli-design.md`

## Global Constraints

- Train only XGBoost and Random Forest on the 105-capture `lab_train` partition.
- Fit probability calibration only on the 35-capture `lab_calibration` partition.
- Evaluate only after fitting on the 105-capture `lab_test` partition.
- Use fusion weights XGBoost `0.60` and Random Forest `0.40`; Isolation Forest has no active weight or artifact.
- Preserve existing bundles and generated data; do not delete or overwrite them.
- Keep the workflow synthetic-PCAP-only, offline, shadow-only, and dependency-free beyond the locked project dependencies.
- Keep deterministic rules separate from model fusion; report model fusion and rule-backed policy separately.

---

### Task 1: Add an explicit two-model bundle mode

**Files:**
- Modify: `ml/models.py`
- Modify: `ml/calibration.py`
- Modify: `ml/explain.py`
- Test: `tests/test_pcap_dataset.py`
- Test: `tests/test_calibration.py`

**Interfaces:**
- `ModelConfig.enable_isolation_forest: bool = False` selects the new active mode.
- `ModelBundle.isolation_forest` and `ModelBundle.isolation_preprocessor` become optional for new bundles while remaining loadable for legacy bundles.
- `ModelOutputs.isolation_forest` becomes optional.
- New bundles use `MODEL_BUNDLE_VERSION = "ml-bundle.v2"` and include `model_names: ["xgboost", "random_forest"]` in `manifest.json`.
- `calibrate_scores()` fits XGBoost/Random Forest calibrators for every bundle and returns an explicit disabled anomaly state when no Isolation Forest output exists.

- [ ] **Step 1: Write failing model contract tests.** Add a small split test that trains with `{"random_seed": 7, "enable_isolation_forest": False}` and asserts the returned bundle has no Isolation Forest objects. Add a save/load test asserting the manifest model names contain exactly XGBoost and Random Forest and that no `isolation_forest.joblib` is written.

- [ ] **Step 2: Run the tests and verify the expected failure.**

```bash
uv run pytest -q tests/test_pcap_dataset.py -k "isolation or save" 
```

Expected: FAIL because the current trainer always creates and saves Isolation Forest.

- [ ] **Step 3: Implement the optional legacy fields.** Keep existing legacy files readable, but guard Isolation Forest fitting, prediction, preprocessing, baseline metadata, and artifact writes behind `enable_isolation_forest`. Set the new bundle version and manifest model names for newly trained bundles; accept both `ml-bundle.v1` and `ml-bundle.v2` while loading.

- [ ] **Step 4: Make calibration explicit for the two-model mode.** Preserve the XGBoost/Random Forest isotonic calibrators. When outputs have no Isolation Forest value, set `anomaly_enabled=False`, keep anomaly threshold fields absent/`None`, and do not read or write Isolation Forest thresholds for the new bundle.

- [ ] **Step 5: Make explanations safe without Isolation Forest.** Return an empty `anomaly_contributions` list when the bundle has no Isolation Forest instead of dereferencing an absent preprocessor/model.

- [ ] **Step 6: Run the task tests.**

```bash
uv run pytest -q tests/test_pcap_dataset.py tests/test_calibration.py
```

Expected: PASS, including legacy fixture coverage and the new no-Isolation-Forest assertions.

---

### Task 2: Replace scalar/Isolation Forest fusion with calibrated two-model probability fusion

**Files:**
- Modify: `ml/fusion.py`
- Modify: `ml/evaluate.py`
- Modify: `ml/ablation.py`
- Modify: `ml/schema.py` if the anomaly result needs an explicit disabled state
- Test: `tests/test_fusion.py`
- Test: `tests/test_evaluation.py`

**Interfaces:**
- `FusionConfig` defaults to `xgboost_weight=0.60`, `random_forest_weight=0.40`, and no active Isolation Forest weight.
- Add a pure helper `fused_class_probabilities(output: ModelOutputs, calibration: CalibrationState, config: FusionConfig | None = None) -> dict[RiskLabel, float]` that normalizes the two configured supervised weights and combines calibrated per-class probabilities.
- `fuse_session()` derives the risk class from the fused probability-vector argmax and the scalar risk score from its class centers. Rules may still raise final severity.
- `EvaluationReport` retains a disabled anomaly status for schema visibility but its active model comparison contains XGBoost, Random Forest, model fusion, and rule policy only.

- [ ] **Step 1: Write failing fusion tests.** Replace the old Isolation Forest weight assertion with a 60/40 assertion. Add a deterministic fixture where XGBoost and Random Forest disagree and assert `fused_class_probabilities()` follows the configured weighted class probabilities. Add a test that a two-model result contains no Isolation Forest model output and reports an explicit disabled anomaly state.

- [ ] **Step 2: Run the tests and verify the expected failure.**

```bash
uv run pytest -q tests/test_fusion.py tests/test_evaluation.py
```

Expected: FAIL because the current fusion requires/scales Isolation Forest and uses fixed scalar thresholds.

- [ ] **Step 3: Implement calibrated probability-vector fusion.** Combine the calibrated XGBoost and Random Forest probability dictionaries, divide by the active supervised weight sum, use the fused vector argmax for `model_risk`, and compute the reported score with the existing ordered class centers. Do not add Isolation Forest scores to the new model risk.

- [ ] **Step 4: Preserve policy precedence.** Keep deterministic rule findings as a minimum final severity and retain the advisory/model-derived source labels. For two-model results, expose no Isolation Forest score as an active model and use the typed disabled state rather than fabricated anomaly values.

- [ ] **Step 5: Update evaluation and ablation output.** Remove active Isolation Forest calculations from two-model evaluation paths, keep the disabled status in the serialized report, and make ablation output report `status="disabled:isolation_forest_removed"` rather than dereferencing missing metrics.

- [ ] **Step 6: Run the focused tests.**

```bash
uv run pytest -q tests/test_fusion.py tests/test_evaluation.py tests/test_calibration.py
```

Expected: PASS with model-only and rule-policy reports separated.

---

### Task 3: Persist the named 17K dataset and train on the 105-capture split

**Files:**
- Modify: `scripts/train_evaluate_grouped_matrix.py`
- Modify: `scripts/evaluate_pcap_bundle.py`
- Modify: `scripts/evaluate_prod_holdouts.py`
- Modify: `ml/product.py`
- Test: `tests/test_training_script.py`
- Test: `tests/test_evaluate_script.py`
- Test: `tests/test_product.py`

**Interfaces:**
- Defaults: `DATASET_RUN = datasets/runs/Dataset-17K`, `BUNDLE_DIR = models/Model_XG_RF`, `CONFIG_PATH = configs/training.pcap.json`.
- `train_packet_matrix()` first loads and validates `Dataset-17K` when it exists; otherwise it runs the existing matrix, assembles the PCAP dataset, and calls `write_dataset_run(..., run_id="Dataset-17K")`.
- The loaded/saved split must be exactly 7,665 train, 2,555 calibration, and 7,665 test records, with 245 capture hashes and 17,885 total records.
- Product default bundle becomes `models/Model_XG_RF`; the old default path remains available only when explicitly passed.

- [ ] **Step 1: Write failing named-artifact tests.** Assert the training script defaults point to `Dataset-17K` and `Model_XG_RF`. Add a fixture that writes a valid small named dataset and verifies the loader path is reused instead of invoking matrix capture. Assert the full-size split gate reports the three expected record counts.

- [ ] **Step 2: Run the tests and verify the expected failure.**

```bash
uv run pytest -q tests/test_training_script.py tests/test_evaluate_script.py tests/test_product.py
```

Expected: FAIL because current defaults use generated hash names and `grouped-105-capture-pcap`.

- [ ] **Step 3: Implement named-run reuse and split validation.** Load `Dataset-17K` with `load_dataset_run()` when its manifest exists; otherwise retain the existing matrix path and write the explicit run ID. Validate the exact environment split before training so XGBoost and Random Forest receive only `lab_train`, calibration reads only `lab_calibration`, and evaluation reads only `lab_test`.

- [ ] **Step 4: Update configuration and bundle defaults.** Set `random_seed=420042`, remove Isolation Forest training settings from the active configuration, set fusion to XGBoost `0.60` and Random Forest `0.40`, and route product/evaluation scripts to `Model_XG_RF`.

- [ ] **Step 5: Materialize the named dataset without regenerating PCAPs.** Load the validated existing 17,885-record synthetic PCAP run and call `write_dataset_run(dataset, split, "datasets/runs", run_id="Dataset-17K")`. If the target path already contains identical hashes, reuse it; never overwrite a conflicting run.

- [ ] **Step 6: Run the task tests.**

```bash
uv run pytest -q tests/test_training_script.py tests/test_evaluate_script.py tests/test_product.py
```

Expected: PASS with the new default artifact names and exact split gates.

---

### Task 4: Add visible phase progress, ETA, and fixed-width evaluation output

**Files:**
- Modify: `datasets/lab/runner.py`
- Modify: `scripts/train_evaluate_grouped_matrix.py`
- Modify: `scripts/evaluate_pcap_bundle.py`
- Modify: `scripts/evaluate_prod_holdouts.py`
- Modify: `ml/evaluate.py`
- Test: `tests/test_training_script.py`
- Test: `tests/test_evaluate_script.py`

**Interfaces:**
- `run_training_matrix(..., on_run_complete: Callable[[int, int, LabRun], None] | None = None)` reports each terminal capture status.
- `assemble_successful_runs(..., on_capture_complete: Callable[[int, int], None] | None = None)` reports each extracted capture.
- Add pure `format_evaluation_table(report: EvaluationReport, fusion_config: FusionConfig) -> str` with rows for XGBoost, Random Forest, model fusion, and rule policy and columns for accuracy, macro-F1, weighted-F1, critical precision, and critical recall.
- The default training command is one line: `uv run python scripts/train_evaluate_grouped_matrix.py`.

- [ ] **Step 1: Write failing CLI tests.** Assert a formatted evaluation string contains the formula `Fusion = 0.60 × XGBoost + 0.40 × Random Forest` and the four row labels. Assert the progress formatter includes completed/total, elapsed, ETA, and status counts.

- [ ] **Step 2: Run the tests and verify the expected failure.**

```bash
uv run pytest -q tests/test_training_script.py tests/test_evaluate_script.py
```

Expected: FAIL because current scripts print JSON/final lines only and runner callbacks do not exist.

- [ ] **Step 3: Add runner callbacks.** Call the capture callback after each `run_scenario()` result and the extraction callback after each successful `extract_run()` result. Leave callback arguments optional so existing callers retain behavior.

- [ ] **Step 4: Implement the standard-library progress reporter.** Print named phases for configuration, matrix capture, PCAP extraction, validation, split, training, calibration, evaluation, and persistence. Use a carriage-return bar for callback-driven phases, calculate ETA only from completed work, and print explicit elapsed time when ETA is unavailable.

- [ ] **Step 5: Implement the evaluation table.** Format percentages deterministically to two decimals, print the fusion formula, include model-only and rule-policy rows, and avoid third-party table dependencies. Use the same formatter from all evaluation CLIs.

- [ ] **Step 6: Run the task tests and a one-shot CLI check.**

```bash
uv run pytest -q tests/test_training_script.py tests/test_evaluate_script.py
uv run python scripts/evaluate_pcap_bundle.py --dataset-run datasets/runs/Dataset-17K --bundle models/Model_XG_RF
```

Expected: PASS and a readable table printed after the persisted bundle is available.

---

### Task 5: Update documentation, train the named bundle, and verify artifacts

**Files:**
- Modify: `README.md`
- Modify: `docs/specs/spec.md`
- Modify: `configs/training.pcap.json`
- Create: `datasets/runs/Dataset-17K/` (generated, ignored)
- Create: `models/Model_XG_RF/` (generated, ignored)
- Create: `evals/evaluation-*/metrics.json` (generated, ignored or staged according to repository state)

- [ ] **Step 1: Update documentation and config.** Document the 105/35/105 split, two-model 60/40 fusion, named artifacts, short command, progress output, and the synthetic-only limitation. Remove active Isolation Forest claims from the new workflow while documenting old bundles as legacy artifacts.

- [ ] **Step 2: Run the complete focused software checks.**

```bash
uv run pytest -q tests/test_models.py tests/test_pcap_dataset.py tests/test_calibration.py tests/test_fusion.py tests/test_evaluation.py tests/test_training_script.py tests/test_evaluate_script.py tests/test_product.py
uv run python -m compileall -q ml datasets/lab scripts tests
```

Expected: PASS; if `tests/test_models.py` is absent, run the listed existing test files without it.

- [ ] **Step 3: Run the short training command.**

```bash
uv run python scripts/train_evaluate_grouped_matrix.py
```

Expected: the CLI prints all named phases, progress/ETA during capture or extraction, the `Dataset-17K` path, the `Model_XG_RF` path, the evaluation path, and the fixed-width table.

- [ ] **Step 4: Verify the persisted dataset.**

```bash
uv run python - <<'PY'
from ml.dataset import validate_dataset_run
m = validate_dataset_run("datasets/runs/Dataset-17K")
assert m["record_count"] == 17885
assert len(m["pcap_sha256"]) == 245
print("Dataset-17K validated", m["record_count"], "records", len(m["pcap_sha256"]), "PCAPs")
PY
```

- [ ] **Step 5: Verify the new bundle and evaluation report.**

```bash
uv run python - <<'PY'
import json
from pathlib import Path
from ml.calibration import load_calibration_state
from ml.models import load_model_bundle
bundle = load_model_bundle("models/Model_XG_RF")
calibration = load_calibration_state("models/Model_XG_RF")
manifest = json.loads(Path("models/Model_XG_RF/manifest.json").read_text())
assert manifest["model_names"] == ["xgboost", "random_forest"]
assert not Path("models/Model_XG_RF/isolation_forest.joblib").exists()
assert calibration.version
print("Model_XG_RF reloaded", bundle.bundle_id)
PY
find evals -maxdepth 2 -path '*/metrics.json' -print | sort | tail -1
```

- [ ] **Step 6: Inspect the diff and report evidence.** Run `git diff --check`, `git status --short`, and the final focused/full test commands. Report old artifacts preserved, generated artifact paths, exact metrics, any unverified live/deployment work, and one next step.

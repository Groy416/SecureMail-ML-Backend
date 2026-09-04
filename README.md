# SecureMailScope ML

SecureMailScope ML is a Python library-first pipeline for passive email-network security assessment. It turns reconstructed SMTP/IMAP/POP3 session observations into:

- supervised cryptographic-risk predictions;
- an unsupervised behavioral-anomaly signal;
- deterministic, evidence-linked security findings;
- calibrated and fused model results; and
- feature-level explanations and evaluation reports.

The governing principle is **evidence first, AI second**. The ML output is advisory. Deterministic protocol, TLS, certificate, and policy rules remain authoritative for security findings.

> **Current status:** the core schema, feature-only dataset factory, PCAP dataset assembly, preprocessing, XGBoost, Random Forest, Isolation Forest, calibration, fusion, rules, explanations, evaluation, ablations, and a small Docker capture lab are implemented. The repository is not yet a packaged command-line application. See [Spec alignment](#spec-alignment) for the exact gaps between the target specification and the current code.

## Security and product boundary

This project is designed for authorized, passive analysis only.

- It does not read or store email bodies, subjects, attachments, passwords, cookies, tokens, or production credentials.
- It must not block mail, change server configuration, or automatically remediate a finding.
- The synthetic lab uses a private Docker network, synthetic names, and short-lived certificates.
- Weak or legacy TLS scenarios must stay isolated. If a scenario cannot be negotiated by the lab crypto stack, it must be recorded as unsupported rather than represented as observed traffic.
- Authorized real captures are validation data, not initial training data.
- Generated PCAPs, TLS key logs, certificates, model bundles, and dataset runs are local artifacts and are ignored by the repository’s `.gitignore`.

## Quick start

### Requirements

- Python 3.11 or newer.
- [`uv`](https://docs.astral.sh/uv/) for environment and dependency management.
- Docker and Docker Compose only for the optional PCAP lab.
- `tshark` or the local `lab-mail-lab:latest` parser image only for PCAP extraction.

The project dependencies are locked in `uv.lock`. The current environment uses the versions declared in `pyproject.toml` (NumPy, pandas, PyArrow, Pydantic, scikit-learn, XGBoost, SHAP, joblib, and pytest).

### Install

From the repository root:

```bash
uv sync
```

### Run the test suite

```bash
uv run pytest
```

The current automated suite is intentionally small: `tests/test_pcap_dataset.py` verifies PCAP-mode dataset assembly, capture-hash/scenario-manifest persistence, split writing, run validation, and rejection of a training split with incomplete risk labels. It does not yet cover every item in the specification’s test plan.

For a concise result:

```bash
uv run pytest -q
```

### Run the module smoke checks

The modules are currently the executable examples and smoke checks. They generate a deterministic 500-session feature dataset and train models where required:

```bash
for module in \
  ml.dataset \
  ml.preprocess \
  ml.models \
  ml.fusion \
  ml.evaluate \
  ml.ablation \
  ml.explain \
  ml.pipeline
do
  uv run python -m "$module"
done
```

`ml.pcap` is separate because it needs a capture and a parser:

```bash
uv run python -m ml.pcap
```

That smoke check reads `datasets/lab/captures/synthetic_mail.pcap` when it exists. It uses the local `tshark` binary first and otherwise the Docker image described in [PCAP extraction](#pcap-extraction).

There is no current `python -m securemailscope.ml ...` CLI. The CLI shape in `docs/specs/spec.md` is a future interface, not a command that can be run today.

## Repository layout

```text
.
├── README.md
├── pyproject.toml                 # project metadata and dependencies
├── uv.lock                        # locked dependency graph
├── .python-version                # Python 3.11
├── .gitignore                     # local environments and generated artifacts
├── ml/
│   ├── schema.py                  # Pydantic contracts and enums
│   ├── features.py                # feature views and model-input map
│   ├── dataset.py                 # catalog, generators, PCAP assembly, splits, persistence
│   ├── preprocess.py              # imputation, encoding, scaling, feature map
│   ├── models.py                  # XGBoost, Random Forest, Isolation Forest, bundles
│   ├── calibration.py             # isotonic calibration and anomaly normalization
│   ├── rules.py                   # deterministic evidence-backed findings
│   ├── fusion.py                  # model fusion, policy precedence, result contract
│   ├── explain.py                 # TreeSHAP and anomaly baseline perturbation
│   ├── evaluate.py                # metrics and evaluation report persistence
│   ├── ablation.py                # feature-view ablation reports
│   ├── pcap.py                    # tshark/Docker PCAP-to-session adapter
│   └── pipeline.py                # inference orchestration with explanations
├── datasets/
│   └── lab/                       # isolated SMTP/IMAP STARTTLS Docker lab
├── evals/                         # checked-in evaluation snapshots
├── tests/                         # current pytest coverage
└── docs/specs/spec.md             # implementation target and contracts
```

The following directories are intentionally absent until generated: `models/` and `datasets/runs/`. They are ignored because they contain local model and dataset artifacts, not source code.

## End-to-end flow

The real code path is:

```text
feature config or PCAP
        │
        ├── generate_feature_dataset(...)          # deterministic feature rows
        └── extract_sessions(...)                 # packet-backed rows
                         │
                         ▼
              DatasetArtifact / SessionFeatureRecord
                         │
                         ▼
              split_dataset(...)                   # environment grouping
                         │
                         ▼
              build_feature_matrix(...)            # fit on train only
                         │
                         ├── XGBoost risk classifier
                         ├── Random Forest risk classifier
                         └── Isolation Forest normal baseline
                                      │
                                      ▼
              calibrate_scores(validation outputs, validation records)
                                      │
                                      ▼
              deterministic rules + weighted fusion
                                      │
                                      ▼
              ml.pipeline.predict_session(...)
                                      │
                         ├── MLResult
                         └── TreeSHAP/anomaly explanations
```

Labels in synthetic data come from the scenario manifest. They are never inferred from the prediction being evaluated. Evidence references travel with each record and are copied into rule findings and final results where available.

## Data contracts

### `SessionFeatureRecord`

`ml.schema.SessionFeatureRecord` is the canonical unit of inference. It contains:

- `schema_version`: currently `session-features.v1`;
- `provenance`: capture, flow, and session IDs; source type; scenario and environment IDs; optional generated seed; and one or more evidence references;
- `features`: validated protocol/session, TLS, and certificate posture fields; and
- `labels`: ordered risk label, anomaly label, and expected deterministic finding IDs.

The supported protocol values are `SMTP`, `IMAP`, and `POP3`. The supported source types are `synthetic_feature`, `synthetic_pcap`, and `authorized_capture`. The current feature generator emits SMTP `synthetic_feature` records; the PCAP adapter emits `synthetic_pcap` records.

The risk order is:

```text
informational < low < medium < high < critical
```

The possible result actions are `no_action`, `monitor`, `analyst_review`, `prioritize`, and `critical_review`.

Pydantic rejects impossible values such as negative durations, packet counts, or byte counts. Successful TLS handshakes require negotiated TLS fields. Failed handshakes require those negotiated fields to be null. Certificate metadata is only accepted when `cert_present=true`. Synthetic records require a seed and scenario evidence; PCAP-derived records require a packet evidence reference. `validate_session(record, strict=True)` additionally rejects unknown top-level fields. The feature model keeps explicitly unknown feature fields available, but the model matrix only selects the documented input list.

### Evidence references

An evidence reference identifies a source (`pcap`, `session`, or `scenario`), optional stream and packet bounds, and the fields it supports. PCAP evidence must include a packet range. Rule and explanation output uses compact references such as `pcap:tls.handshake.certificate` or `scenario:tls_version`.

## Feature views and preprocessing

The model input is the union of three views. Identifiers, labels, source IPs, and other provenance are not predictive features.

### Protocol/session view

- `protocol`
- `src_port`, `dst_port`
- `starttls_advertised`, `starttls_used`
- `handshake_success`, `handshake_failures`
- `renegotiation_count`
- `session_duration_seconds`
- `packet_count`, `byte_count`
- `retransmission_count`, `out_of_order_count`

### TLS negotiation view

- `tls_version`
- `cipher_suite`, `cipher_family`
- `key_exchange`
- `signature_algorithm`
- `forward_secrecy`

### Certificate posture view

- `cert_present`, `cert_valid`, `cert_expired`
- `cert_expires_in_days`
- `cert_chain_valid`, `hostname_mismatch`
- `cert_key_algorithm`, `cert_key_length_bits`
- `cert_signature_algorithm`

There are 28 source features: 11 numeric, 9 boolean, and 8 categorical. The current default preprocessing pipeline is:

```text
numeric  -> median imputation fitted on training rows
         -> MinMaxScaler(0, 1, clip=True)
boolean  -> True/False mapped to 1.0/0.0, then scaled with numeric features
categorical
         -> constant "__MISSING__" imputation
         -> OneHotEncoder(handle_unknown="ignore", sparse_output=False)
```

The fitted `ColumnTransformer`, source feature order, transformed feature names, and source-to-view feature map are stored in `PreprocessorState`. Validation, test, and inference rows must use that fitted state; the scaler is never refit at inference time. Unknown categories are ignored by the encoder and returned in matrix diagnostics instead of crashing prediction. A 500-row default smoke run currently produces a 46-column matrix; the exact one-hot width depends on categories observed by the training split.

Pass `source_features=(...)` to `build_feature_matrix` or `ModelConfig` for a deliberate feature-view ablation. The default model uses all three views.

## Dataset generation

### Feature-only mode

The current fast generator is `generate_feature_dataset`. It requires a non-negative `master_seed` and positive `session_count`:

```python
from ml.dataset import generate_feature_dataset, split_dataset


dataset = generate_feature_dataset(
    {
        "mode": "synthetic_feature",
        "master_seed": 420042,
        "session_count": 500,
    }
)
split = split_dataset(dataset, {"random_seed": 420042})
```

The default bucket distribution is:

- `normal`: 60%;
- `single_weakness`: 25%;
- `behavioral_anomaly`: 10%;
- `combined`: 5%.

`allocate_counts` uses largest-remainder allocation so the bucket counts add exactly to `session_count`. Within a bucket, scenario IDs are selected cyclically. Each generated row is currently an SMTP session with destination port 587. The generator varies the source port, duration, packet count, byte count, retransmissions, and out-of-order count within the scenario variation ranges. Scenario-specific behavior also varies repeated handshake failures, renegotiations, and handshake-failure counts without changing the ground-truth label.

The catalog currently contains 21 scenarios:

- Normal: `normal_tls13_valid`, `normal_tls12_valid`, `starttls_used_successfully`.
- Single weakness/advisory: `deprecated_tls`, `weak_cipher`, `rsa_no_forward_secrecy`, `expired_certificate`, `invalid_certificate_chain`, `hostname_mismatch`, `weak_rsa_key`, `starttls_advertised_unused`, `starttls_handshake_failure`, `certificate_expiry_advisory`, `certificate_expiry_warning`, `certificate_expires_soon`.
- Behavioral anomaly: `repeated_handshake_failures`, `unusual_cipher_negotiation`, `unexpected_tls_version`, `uncommon_chacha20_negotiation`, `multiple_renegotiations`.
- Combined: `combined_critical_weaknesses`.

Each catalog entry has a validated `scenario.v1` manifest, an expected risk/anomaly label, expected finding IDs, TLS and certificate configuration, client behavior, and variation ranges.

### Reproducibility and environment assignment

For repetition `i`, the derived seed is the first eight bytes of:

```text
SHA-256("<master_seed>:<scenario_id>:<repetition_index>") mod 2^63
```

That seed drives Python’s `Random` instance. The record stores the derived seed, scenario ID, environment ID, and a dataset hash. The default environment IDs are `lab_seed_0001` through `lab_seed_0006`. The code reserves `lab_seed_0004` for evaluation and `lab_seed_0002` for calibration. Certain scenarios are deterministically assigned to those environments; other scenarios are assigned to one of the remaining environments. This keeps repeated scenario IDs out of multiple environment groups.

### Splitting and leakage checks

`split_dataset` defaults to:

- test environment: `lab_seed_0004`;
- validation environment: `lab_seed_0002`; and
- train: all remaining configured environments.

The `train`, `validation`, and `test` ratio fields are used when the corresponding environment IDs are set to `None` and `GroupShuffleSplit` must choose groups. The default path is therefore environment holdout, not a row-level 70/15/15 random split.

The function rejects fewer than three environment groups and checks that both `environment_id` and `scenario_id` are disjoint across partitions. The split hash is based on the ordered session IDs in each partition. The current implementation does not yet expose explicit parameter-combination hashes or capture-level grouping as separate split keys; see [Known gaps](#known-gaps).

### PCAP-mode dataset assembly

PCAP parsing and dataset assembly are separate operations. After `extract_sessions` has produced validated records, use:

```python
from ml.dataset import assemble_pcap_dataset

pcap_dataset = assemble_pcap_dataset(
    {
        "mode": "synthetic_pcap",
        "master_seed": 420042,
        "session_count": len(records),
        "environment_ids": ("lab_train", "lab_calibration", "lab_test"),
        "calibration_environment_id": "lab_calibration",
        "evaluation_environment_id": "lab_test",
    },
    records,
    scenario_manifests,
    pcap_sha256_by_capture_id,
)
```

Assembly validates that the mode is `synthetic_pcap`, the count matches, all configured environments are represented, scenario manifests match the records, and exactly one 64-character SHA-256 digest exists per capture ID. The current code does not generate legacy/weak PCAPs from a scenario manifest; the Docker lab currently produces only valid TLS 1.2+ traffic.

### Persisting a dataset run

```python
from ml.dataset import validate_dataset_run, write_dataset_run

run = write_dataset_run(dataset, split, root="datasets/runs")
validate_dataset_run(run.path)
```

A run is written as:

```text
<root>/<run_id>/
├── run_manifest.json
├── scenario_manifest.jsonl
├── pcap_manifest.json             # PCAP mode only
├── pcaps/                         # populated by a future/owning capture workflow
├── extracted/
│   ├── session_features.jsonl
│   └── session_features.parquet
├── splits/
│   ├── train.parquet
│   ├── validation.parquet
│   └── test.parquet
└── checksums.sha256
```

The run manifest stores the dataset and split hashes, mode, seed, environments, dependency versions, platform, and PCAP hashes when present. Validation rechecks the dataset hash, split hash, exact one-time partitioning, environment/scenario leakage, and saved-file checksums. Existing matching runs are reused; a conflicting run ID raises instead of overwriting data.

## Models

All three models receive the same combined preprocessed vector in the default baseline. They are independent: no model prediction is passed as an input feature to another model.

### XGBoost risk classifier

`ml.models.train_model_bundle` trains `XGBClassifier` with a multiclass probability objective:

```text
objective: multi:softprob
eval_metric: mlogloss
tree_method: hist
n_estimators: 300
max_depth: 5
learning_rate: 0.05
subsample: 0.8
colsample_bytree: 0.8
reg_lambda: 1.0
random_state: configured seed
n_jobs: configured worker count
```

Its target is the five ordered risk labels. Training rejects a split that is missing any risk label, then uses a contiguous class mapping for XGBoost and restores the five-label probability dictionary. XGBoost is the primary supervised signal, but its output is still advisory and is calibrated before fusion. The current code does not pass explicit sample weights to XGBoost; Random Forest’s balancing strategy is the only built-in class-imbalance treatment.

### Random Forest risk classifier

`RandomForestClassifier` is an independent supervised comparator and ensemble-diversity source:

```text
n_estimators: 300
class_weight: balanced
min_samples_leaf: 2
random_state: configured seed
n_jobs: configured worker count
```

It predicts the same risk labels. Its `classes_` output is expanded into the same five-label probability contract so consumers do not need model-specific class handling.

### Isolation Forest

`IsolationForest` is not a risk classifier. It detects behavior that differs from the learned normal baseline:

- fit rows are only training records with `risk_label=informational` and `anomaly_label=0`;
- the raw `decision_function` score is retained, where a higher value is more normal;
- validation normal scores define `normal_low` (1st percentile) and `normal_high` (99th percentile);
- the normalized anomaly score is `clip((normal_high - raw_score) / (normal_high - normal_low), 0, 1)`, so higher is more anomalous;
- the anomaly threshold is the 99th percentile of normalized validation-normal scores; and
- if the normal score range is degenerate, anomaly scoring is explicitly disabled and diagnostics report `unavailable:degenerate_normal_calibration`.

An Isolation Forest flag means “outside the learned normal behavior.” It is not proof of maliciousness and does not alone create a critical finding.

## Calibration, fusion, and policy

### Calibration

`calibrate_scores` receives aligned model outputs and records, normally from the validation partition. XGBoost and Random Forest each get one-vs-rest `IsotonicRegression` calibrators per risk class. Calibrated probabilities are renormalized and converted to a scalar risk probability using class centers:

```text
informational = 0.00
low           = 0.25
medium        = 0.50
high          = 0.75
critical      = 1.00
```

Isolation Forest is calibrated separately as an anomaly score; it is never treated as a class probability.

### Weighted fusion

The default `FusionConfig` is:

```text
0.50 * calibrated XGBoost risk probability
+0.30 * calibrated Random Forest risk probability
+0.20 * normalized Isolation Forest anomaly score
```

The ensemble score maps back to a risk class at these boundaries:

```text
[0.000, 0.125) informational
[0.125, 0.375) low
[0.375, 0.625) medium
[0.625, 0.875) high
[0.875, 1.000] critical
```

Weights must be non-negative and sum to 1. Learned stacking is not implemented. The fixed weights are a baseline, not an optimization claim.

### Deterministic rules

`ml.rules.extract_rule_findings` recomputes evidence-backed findings from features. Current IDs and conditions are:

- `TLS-001`: TLS 1.0 or 1.1, critical.
- `TLS-002`: 3DES or RC4 cipher family, critical.
- `FS-001`: successful handshake without forward secrecy, high.
- `CERT-001`: expired certificate, high.
- `CERT-002`: RSA certificate key below 2048 bits, high.
- `CERT-003`: invalid certificate chain, high.
- `CERT-004`: hostname mismatch, high.
- `STLS-001`: STARTTLS advertised but unused, high.
- `STLS-002`: STARTTLS used but handshake failed, high.
- `ANOM-001`: at least three handshake failures, high.
- `ANOM-002`: RC4 or ChaCha20-Poly1305 negotiation, medium.
- `ANOM-003`: TLS 1.1, medium.
- `ANOM-004`: at least two renegotiations, high.

A rule finding must have evidence. The highest rule severity becomes the minimum final severity. Model output may raise priority but cannot downgrade a deterministic finding. Model-only results are labeled `model_ensemble_advisory`.

### Result actions

The current action selection is ordered as follows:

1. A critical deterministic rule produces `critical_review`.
2. An anomaly flag produces `analyst_review`.
3. Informational model risk produces `no_action`.
4. Low or medium model risk produces `monitor`.
5. Other model risk produces `prioritize`.

`ml.pipeline.predict_session` merges automatically extracted findings with any supplied findings by finding ID, performs fusion, then attempts explanations. If explanation generation fails, it preserves the prediction and adds a typed diagnostic such as `unavailable:...`; it does not fabricate an explanation.

## Inference and training recipe

The current public interfaces are Python functions rather than a CLI:

```python
from ml.calibration import calibrate_scores
from ml.dataset import generate_feature_dataset, split_dataset
from ml.evaluate import evaluate_bundle, save_evaluation_report
from ml.models import predict_model_outputs, train_model_bundle
from ml.pipeline import predict_session


dataset = generate_feature_dataset({"master_seed": 420042, "session_count": 500})
split = split_dataset(dataset, {"random_seed": 420042})
bundle = train_model_bundle(split, {"random_seed": 420042})

validation_outputs = predict_model_outputs(bundle, split.validation)
calibration = calibrate_scores(validation_outputs, split.validation)
result = predict_session(bundle, calibration, split.test[0])

report = evaluate_bundle(bundle, calibration, split)
report_path = save_evaluation_report(report)

print(result.model_dump(mode="json"))
print(report_path)
```

Important interface details:

- `train_model_bundle` takes a `SplitArtifact`, not a raw `DatasetArtifact`.
- `ml.models.predict_model_outputs` returns batch outputs without fusion or explanations.
- `ml.fusion.predict_session` performs model prediction and fusion but does not run explanations.
- `ml.pipeline.predict_session` is the complete single-session path and is the preferred inference entry point.
- `ml.explain.explain_session` can be called directly when an `MLResult` already exists.

## Explainability

The explanation contract is `explanation.v1`.

### XGBoost and Random Forest

`explain_session` uses local `shap.TreeExplainer` values for the predicted class. One-hot contributions are aggregated back to the original source feature, then ranked by absolute contribution. Each entry includes:

- source feature and feature view;
- observed value;
- signed contribution;
- risk direction;
- model name; and
- matching evidence references where the record has them.

The default `top_k` is 8 per supervised model, so a normal full-view result contains up to 16 supervised entries.

### Isolation Forest

The current anomaly method is `isolation_forest_baseline_perturbation`:

1. calculate the original raw Isolation Forest score;
2. replace one source feature with its stored normal baseline median or mode;
3. recalculate the raw score;
4. use the signed score delta as the contribution; and
5. rank absolute deltas.

The default `top_k` is 8. Explanations describe model behavior, not attacker intent. The current explanation entries do not yet repeat all model, preprocessor, and library version metadata required by the target specification; the bundle manifest remains the artifact-level version source.

## Evaluation and ablations

`evaluate_bundle` evaluates only `split.test`. It reports:

- per-class precision, recall, F1, PR-AUC, and ROC-AUC where defined;
- macro and weighted F1;
- critical precision and recall;
- a multiclass Brier score and confusion matrix;
- Isolation Forest precision, recall, F1, false-positive rate, threshold, and normal/anomalous score distributions; and
- the same label metrics for fixed weighted fusion.

Use:

```python
from ml.ablation import evaluate_feature_view_ablations, save_ablation_report

ablation = evaluate_feature_view_ablations(split, {"random_seed": 420042})
path = save_ablation_report(ablation)
```

Ablations train/evaluate protocol-session-only, TLS-only, and certificate-only bundles. They are comparison experiments, not the default topology.

The checked-in evaluation snapshot at `evals/evaluation-5e0b4f5acb7f/metrics.json` covers 156 test sessions in `lab_seed_0004`. Its reported critical recall is 0.0 for XGBoost, Random Forest, and fusion; the Isolation Forest F1 is 0.107. The snapshot itself records that the results are synthetic-lab-only and that isotonic calibration may overfit the small calibration partition. These numbers are evidence of the current benchmark snapshot, not production performance or a quality claim.

## Model and dataset artifacts

### Model bundles

`save_model_bundle(bundle, directory, calibration=None)` refuses to overwrite an existing directory and writes:

```text
<bundle>/
├── manifest.json
├── preprocessor.joblib
├── xgboost.joblib
├── random_forest.joblib
├── isolation_forest.joblib
├── feature_map.json
├── normal_baseline.json
├── calibration.json             # when calibration is supplied
├── thresholds.json              # when calibration is supplied
├── calibration.joblib           # when calibration is supplied
└── checksums.sha256
```

The manifest records bundle/version IDs, training split hash, normal-baseline count, feature names, configuration, dependency versions, and whether calibration is included. `load_model_bundle` verifies listed checksums and the bundle version before loading. `save_calibration_state` and `load_calibration_state` manage calibration artifacts separately when needed.

The current bundle does not yet write the specification’s `policy_version.json` or `metrics.json`, and it stores dependency versions for provenance without enforcing them at load time.

### Evaluation artifacts

`save_evaluation_report` writes a content-addressed `evals/evaluation-<digest>/metrics.json`. `save_ablation_report` writes `evals/ablation-<digest>/metrics.json`. Existing snapshots are kept as reference artifacts; reruns may produce new digest directories.

## Docker synthetic mail lab

The lab is optional and does not feed the ML models automatically. Run it from the repository root:

```bash
docker compose -f datasets/lab/compose.yaml up --build --abort-on-container-exit
```

The lab does the following:

1. The `mail-lab` container creates a one-day private CA and a `mail-lab` SAN certificate.
2. It starts an asyncio SMTP server on port 2525 and IMAP server on port 1143.
3. It starts `tcpdump` on the internal interface.
4. The client performs one SMTP STARTTLS and one IMAP STARTTLS flow using the generated CA.
5. The client writes a TLS key log for local dissection.
6. The capture is written to `datasets/lab/captures/synthetic_mail.pcap`.

The Docker network is marked `internal: true`; the Compose services publish no host ports and send no messages to external recipients. The lab uses `NET_ADMIN` and `NET_RAW` only for its private capture process.

Clean up the services after a run:

```bash
docker compose -f datasets/lab/compose.yaml down --remove-orphans
```

Generated `ca.pem`, `ca.srl`, `tls.keys`, and `synthetic_mail.pcap` are ignored. Remove them only when you intend to regenerate the capture:

```bash
rm -f datasets/lab/captures/*
```

### PCAP extraction

`ml.pcap.extract_sessions` is a passive adapter. It:

- selects a TCP dissector for SMTP, IMAP, or POP3;
- reads packet fields with `tshark`;
- groups packets by `tcp.stream`;
- identifies STARTTLS and server handshakes from packet data;
- maps known TLS/cipher/signature codes to canonical names;
- extracts certificate metadata with OpenSSL/Python SSL; and
- preserves stream, packet-range, and field evidence references.

The parser chooses a host `tshark` binary first. If none exists, it requires Docker and an image named `lab-mail-lab:latest`:

```bash
docker build -t lab-mail-lab:latest datasets/lab
```

A TLS key log is used when `tls.keys` is beside the PCAP. Certificate validity, chain validity, and hostname mismatch are populated only when both a trusted CA path and expected hostname are supplied; otherwise those fields remain null. Unknown TLS values are represented as `UNKNOWN`, not silently guessed.

Example:

```python
from ml.dataset import CATALOG
from ml.pcap import extract_sessions

scenario = next(
    item["manifest"]
    for item in CATALOG
    if item["manifest"]["scenario_id"] == "normal_tls13_valid"
)
records = extract_sessions(
    "datasets/lab/captures/synthetic_mail.pcap",
    scenario=scenario,
    environment_id="lab_smoke",
    generator_seed=420042,
    destination_port=2525,
    trusted_ca_path="datasets/lab/captures/ca.pem",
    expected_hostname="mail-lab",
)
```

The adapter currently labels extracted rows as `synthetic_pcap` and does not create an `authorized_capture` dataset. It also has limited packet-derived handling for retries and renegotiations; the full scenario runner and legacy-compatible lab cases remain future work.

## Git initialization and repository hygiene

This checkout has been initialized as a Git repository on the `main` branch. No commit or remote is created automatically.

Inspect the initial state with:

```bash
git status --short --ignored
```

The root `.gitignore` excludes:

- `.venv`, bytecode, pytest/tool caches, coverage output, and OS files;
- `.env` files and local credentials;
- generated `models/`, `datasets/runs/`, `graphify-out/`, and `*.joblib` artifacts; and
- lab captures, PCAP/PCAPNG files, and TLS key logs.

`pyproject.toml`, `uv.lock`, source files, specs, tests, and the checked-in evaluation snapshots are not ignored. Review generated data before staging anything; do not force-add credentials, key logs, or captures containing sensitive traffic.

## Spec alignment

The authoritative target is [`docs/specs/spec.md`](docs/specs/spec.md). The spec has been updated to identify the implementation as partial rather than “not implemented.” The current code covers these areas:

- Pydantic session/scenario contracts and validation.
- Three feature views, model feature selection, and transformed-feature mapping.
- Deterministic feature-only scenario generation with reproducible hashes.
- PCAP-mode dataset assembly with scenario manifests and capture hashes.
- Environment-held-out train/validation/test splits with environment/scenario leakage checks.
- Training-only imputation, one-hot encoding, and MinMax scaling.
- XGBoost, Random Forest, and normal-only Isolation Forest model adapters.
- Isotonic supervised calibration, percentile anomaly normalization, and fixed weighted fusion.
- Deterministic rules with evidence and policy precedence.
- TreeSHAP and Isolation Forest baseline-perturbation explanations.
- Evaluation metrics, saved reports, feature-view ablations, and model/dataset checksums.
- A private Docker SMTP/IMAP STARTTLS smoke lab and a tshark/OpenSSL PCAP adapter.

### Known gaps

The following specification items are deliberately not described as complete:

1. There is no packaged `securemailscope.ml` CLI or YAML configuration loader; callers use Python APIs and mappings.
2. The feature generator does not create `synthetic_pcap` traffic. PCAP mode assembles already-extracted rows, while the lab only exercises valid TLS 1.2+ SMTP/IMAP STARTTLS.
3. POP3 is represented in the schema and parser selection but is not exercised by the Docker lab. Postfix, Dovecot, Roundcube, scenario execution, and legacy-compatible endpoints are not present.
4. Splitting checks environment and scenario overlap, but parameter-combination and capture grouping are not first-class persisted keys.
5. The model bundle omits `policy_version.json` and `metrics.json`; its manifest is smaller than the target artifact contract and does not enforce dependency-version compatibility.
6. Evaluation does not yet include a per-scenario/family breakdown or calibration curves, and learned stacking is intentionally disabled.
7. Explanations do not repeat every model/preprocessor/explanation-library version on each entry.
8. The repository has two focused pytest tests, not the complete schema/dataset/preprocessing/model/policy/explainability/end-to-end matrix listed in the spec.

These gaps are important boundaries: do not present the current lab capture or checked-in metrics as evidence that the full target system has been delivered.

## Development workflow

For a change:

1. Read the relevant section of `docs/specs/spec.md` and the affected module.
2. Keep labels, policy rules, provenance, and security decisions in code rather than prompts or UI.
3. Run `uv run pytest`.
4. Run the affected module smoke check; use the Docker/PCAP check only when the capture boundary changed.
5. Inspect `git diff` and confirm no generated artifacts or secrets are staged.

The smallest useful next step toward the target is to add the remaining contract-focused tests around the existing public APIs before introducing a CLI or a larger lab runner.

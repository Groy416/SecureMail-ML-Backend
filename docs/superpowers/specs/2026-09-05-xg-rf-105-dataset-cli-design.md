# XGBoost/Random Forest 105-Capture Training Workflow

## Goal

Create a readable, repeatable two-model training workflow that trains XGBoost and Random Forest on the 105-capture training partition, calibrates on the separate 35-capture partition, evaluates once on the 105-capture test partition, and presents progress and metrics clearly in the CLI.

## Decisions

- The existing 245-capture matrix remains the source: 105 train captures, 35 calibration captures, and 105 test captures; each capture contributes 73 sessions.
- The new persisted dataset is `datasets/runs/Dataset-17K` with 17,885 session records.
- The new active model bundle is `models/Model_XG_RF`.
- XGBoost and Random Forest are the only active trained models. Isolation Forest is not fitted, persisted, calibrated, fused, evaluated, or used for explanations in the new bundle.
- The old generated bundles remain untouched. The loader may continue reading legacy bundles when their artifacts are present, but the new bundle is versioned and identifies its active model names.
- Calibration uses only the 35-capture `lab_calibration` partition. The 105-capture `lab_test` partition remains read-only until final evaluation.
- Fusion uses calibrated per-class probability vectors with XGBoost weight `0.60` and Random Forest weight `0.40`. The reported scalar risk score is the expected class-center score of the fused vector; the class is the fused-vector argmax.
- Deterministic rules remain a separate policy layer and may raise the final minimum severity. The CLI reports model fusion separately from rule-backed policy fusion.
- The training command defaults to the named dataset and model paths so the normal command is `uv run python scripts/train_evaluate_grouped_matrix.py`.
- Progress output uses the standard library: named phases, a carriage-return progress bar for capture/extraction phases, elapsed time, bounded ETA from completed work, and explicit terminal status counts.
- Evaluation output uses a dependency-free fixed-width table and prints the formula `Fusion = 0.60 × XGBoost + 0.40 × Random Forest`.

## Boundaries

- No online learning, live capture, remediation, blocking, quarantine, or new dependency.
- No test-row tuning. Any future weight tuning must use grouped calibration data and be recorded separately from the final test report.
- Existing `models/*` and generated datasets are not deleted or overwritten.

## Acceptance checks

1. A complete named dataset validates with 245 PCAP hashes, 17,885 records, and split sizes 7,665 / 2,555 / 7,665 for train/calibration/test.
2. `models/Model_XG_RF` contains XGBoost, Random Forest, preprocessing, calibration, manifest, and checksums, with no Isolation Forest artifact.
3. A reloaded named bundle produces the same XGBoost/Random Forest predictions as the in-memory bundle.
4. The default training command prints phase progress, elapsed/ETA information during long phases, the dataset/model/evaluation paths, and a fixed-width metrics table.
5. Focused tests cover two-model training, calibration partition boundaries, probability-vector fusion, disabled Isolation Forest state, named dataset reuse, progress formatting, and evaluation table formatting.
6. The full non-Docker test suite and Python compilation pass; Docker capture verification is run only if the named dataset is absent.

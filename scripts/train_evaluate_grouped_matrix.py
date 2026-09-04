"""Train, calibrate on lab_calibration only, then evaluate lab_test last."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets.lab.matrix import generate_grouped_feature_dataset
from ml.calibration import calibrate_scores
from ml.dataset import split_dataset, write_dataset_run
from ml.evaluate import evaluate_bundle, save_evaluation_report
from ml.models import predict_model_outputs, save_model_bundle, train_model_bundle

SEED = 420042
BUNDLE_DIR = Path("models/grouped-69-capture")


def main() -> None:
    dataset = generate_grouped_feature_dataset(SEED)
    split = split_dataset(
        dataset,
        {
            "random_seed": SEED,
            "validation_environment_id": "lab_calibration",
            "test_environment_id": "lab_test",
        },
    )
    write_dataset_run(dataset, split, "datasets/runs")
    bundle = train_model_bundle(split, {"random_seed": SEED})
    calibration = calibrate_scores(
        predict_model_outputs(bundle, split.validation),
        split.validation,
    )
    report = evaluate_bundle(bundle, calibration, split)
    if BUNDLE_DIR.exists():
        shutil.rmtree(BUNDLE_DIR)
    saved = save_model_bundle(bundle, BUNDLE_DIR, calibration)
    evaluation_path = save_evaluation_report(report, "evals")
    print(saved)
    print(evaluation_path)
    print(
        "fusion_critical_recall",
        report.fusion["critical_recall"],
        "xgboost_accuracy",
        report.classifiers["xgboost"]["accuracy"],
        "isolation_recall",
        report.isolation_forest["recall"],
        "test_sessions",
        report.test_session_count,
        "test_environments",
        report.test_environment_ids,
    )


if __name__ == "__main__":
    main()

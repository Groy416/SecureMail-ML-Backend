"""Evaluate the grouped 69-capture lab matrix against its saved bundle."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets.lab.matrix import generate_grouped_feature_dataset
from ml.calibration import load_calibration_state
from ml.dataset import split_dataset
from ml.evaluate import evaluate_bundle, save_evaluation_report
from ml.models import load_model_bundle

SEED = 420042
BUNDLE = "models/grouped-105-capture"

dataset = generate_grouped_feature_dataset(SEED)
split = split_dataset(
    dataset,
    {
        "random_seed": SEED,
        "validation_environment_id": "lab_calibration",
        "test_environment_id": "lab_test",
    },
)
report = evaluate_bundle(
    load_model_bundle(BUNDLE),
    load_calibration_state(BUNDLE),
    split,
)
print(save_evaluation_report(report, "evals"))

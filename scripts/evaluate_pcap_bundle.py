"""Re-evaluate the saved PCAP bundle against its deterministic lab matrix."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets.lab.runner import assemble_successful_runs, build_training_matrix, run_scenario
from ml.calibration import load_calibration_state
from ml.dataset import split_dataset
from ml.evaluate import evaluate_bundle, save_evaluation_report
from ml.models import load_model_bundle

SEED = 420042
BUNDLE = "models/pcap-b23f490d266e-v2"

runs = [run_scenario(profile) for profile in build_training_matrix(SEED)]
if any(run.status != "success" for run in runs):
    raise SystemExit("all 15 matrix captures must succeed before evaluation")
dataset = assemble_successful_runs(runs, {
    "mode": "synthetic_pcap", "master_seed": SEED, "session_count": 5010,
    "environment_ids": ("lab_train", "lab_calibration", "lab_test"),
    "calibration_environment_id": "lab_calibration", "evaluation_environment_id": "lab_test",
})
split = split_dataset(dataset, {"random_seed": SEED, "validation_environment_id": "lab_calibration", "test_environment_id": "lab_test"})
report = evaluate_bundle(load_model_bundle(BUNDLE), load_calibration_state(BUNDLE), split)
print(save_evaluation_report(report, "evals"))

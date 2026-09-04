"""Train and evaluate the persisted synthetic packet-backed matrix."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets.lab.matrix import ENVIRONMENTS, MATRIX_SESSIONS
from datasets.lab.runner import (
    LabRun,
    assemble_successful_runs,
    run_training_matrix,
    validate_training_matrix,
)
from ml.calibration import calibrate_scores, load_calibration_state
from ml.dataset import (
    DatasetArtifact,
    DatasetConfig,
    DatasetRun,
    SplitArtifact,
    split_dataset,
    write_dataset_run,
)
from ml.evaluate import EvaluationReport, evaluate_bundle, save_evaluation_report
from ml.fusion import FusionConfig
from ml.models import (
    ModelBundle,
    ModelConfig,
    load_model_bundle,
    predict_model_outputs,
    save_model_bundle,
    train_model_bundle,
)

SEED = 420042
CONFIG_PATH = Path("configs/training.pcap.json")
LAB_ROOT = Path("datasets/lab/runs")
DATASET_ROOT = Path("datasets/runs")
BUNDLE_DIR = Path("models/grouped-105-capture-pcap")


def load_training_config(
    path: str | Path = CONFIG_PATH,
) -> tuple[ModelConfig, FusionConfig]:
    payload = json.loads(Path(path).read_text())
    fusion = FusionConfig.model_validate(payload.pop("fusion", {}))
    return ModelConfig.model_validate(payload), fusion


def ensure_training_matrix_ready(
    runs: Sequence[LabRun],
    dataset: DatasetArtifact,
) -> None:
    validate_training_matrix(runs, dataset)


def _dataset_config(seed: int) -> DatasetConfig:
    return DatasetConfig.model_validate(
        {
            "mode": "synthetic_pcap",
            "master_seed": seed,
            "session_count": MATRIX_SESSIONS,
            "environment_ids": ENVIRONMENTS,
            "calibration_environment_id": "lab_calibration",
            "evaluation_environment_id": "lab_test",
        }
    )


def _reuse_or_save_bundle(
    bundle: ModelBundle,
    calibration,
    directory: str | Path,
) -> Path:
    path = Path(directory)
    if not path.exists():
        return save_model_bundle(bundle, path, calibration)
    existing = load_model_bundle(path)
    existing_calibration = load_calibration_state(path)
    if existing.bundle_id != bundle.bundle_id:
        raise FileExistsError(f"model bundle conflicts with existing path: {path}")
    if (
        existing_calibration.normal_low != calibration.normal_low
        or existing_calibration.normal_high != calibration.normal_high
        or existing_calibration.anomaly_threshold != calibration.anomaly_threshold
        or existing_calibration.anomaly_enabled != calibration.anomaly_enabled
    ):
        raise FileExistsError(f"calibration conflicts with existing path: {path}")
    return path


def train_packet_matrix(
    *,
    seed: int = SEED,
    lab_root: str | Path = LAB_ROOT,
    dataset_root: str | Path = DATASET_ROOT,
    bundle_dir: str | Path = BUNDLE_DIR,
    config_path: str | Path = CONFIG_PATH,
) -> tuple[DatasetRun, Path, Path, EvaluationReport]:
    model_config, fusion_config = load_training_config(config_path)
    if model_config.random_seed != seed:
        model_config = model_config.model_copy(update={"random_seed": seed})
    runs = run_training_matrix(seed, lab_root)
    dataset_config = _dataset_config(seed)
    dataset = assemble_successful_runs(runs, dataset_config)
    ensure_training_matrix_ready(runs, dataset)
    split = split_dataset(
        dataset,
        {
            "random_seed": seed,
            "validation_environment_id": "lab_calibration",
            "test_environment_id": "lab_test",
        },
    )
    dataset_run = write_dataset_run(dataset, split, dataset_root)
    bundle = train_model_bundle(split, model_config)
    calibration = calibrate_scores(
        predict_model_outputs(bundle, split.validation),
        split.validation,
    )
    report = evaluate_bundle(
        bundle,
        calibration,
        split,
        fusion_config=fusion_config,
    )
    saved_bundle = _reuse_or_save_bundle(bundle, calibration, bundle_dir)
    evaluation_path = save_evaluation_report(report, "evals")
    return dataset_run, saved_bundle, evaluation_path, report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--root", default=str(LAB_ROOT))
    parser.add_argument("--dataset-root", default=str(DATASET_ROOT))
    parser.add_argument("--bundle", default=str(BUNDLE_DIR))
    parser.add_argument("--config", default=str(CONFIG_PATH))
    args = parser.parse_args(argv)
    configured_model, _ = load_training_config(args.config)
    seed = configured_model.random_seed if args.seed is None else args.seed
    dataset_run, saved_bundle, evaluation_path, report = train_packet_matrix(
        seed=seed,
        lab_root=args.root,
        dataset_root=args.dataset_root,
        bundle_dir=args.bundle,
        config_path=args.config,
    )
    print("dataset", dataset_run.path)
    print("bundle", saved_bundle)
    print("evaluation", evaluation_path)
    print(
        "fusion_critical_recall",
        report.fusion["critical_recall"],
        "model_fusion_critical_recall",
        report.model_fusion["critical_recall"],
        "isolation_recall",
        report.isolation_forest["recall"],
        "test_sessions",
        report.test_session_count,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

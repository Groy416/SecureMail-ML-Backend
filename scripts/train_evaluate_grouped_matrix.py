"""Train and evaluate the persisted synthetic packet-backed matrix."""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets.lab.matrix import ENVIRONMENTS, MATRIX_SESSIONS, SESSIONS_PER_CAPTURE
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
    load_dataset_run,
    split_dataset,
    write_dataset_run,
)
from ml.evaluate import (
    EvaluationReport,
    evaluate_bundle,
    format_evaluation_table,
    save_evaluation_report,
)
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
DATASET_RUN = DATASET_ROOT / "Dataset-17K"
BUNDLE_DIR = Path("models/Model_XG_RF")
EXPECTED_DATASET_RECORDS = 17_885
EXPECTED_CAPTURE_COUNT = 245
EXPECTED_SPLIT_COUNTS = (7_665, 2_555, 7_665)


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


def validate_named_dataset(
    dataset: DatasetArtifact,
    split: SplitArtifact,
    *,
    expected_seed: int | None = None,
) -> None:
    if dataset.mode != "synthetic_pcap":
        raise ValueError("named training dataset must be synthetic_pcap")
    if expected_seed is not None and dataset.master_seed != expected_seed:
        raise ValueError(
            f"named training dataset seed {dataset.master_seed} does not match {expected_seed}"
        )
    if len(dataset.records) != EXPECTED_DATASET_RECORDS:
        raise ValueError(
            f"named training dataset requires {EXPECTED_DATASET_RECORDS} records"
        )
    if len(dataset.pcap_sha256) != EXPECTED_CAPTURE_COUNT:
        raise ValueError(
            f"named training dataset requires {EXPECTED_CAPTURE_COUNT} PCAP hashes"
        )
    capture_counts = Counter(record.provenance.capture_id for record in dataset.records)
    if (
        set(capture_counts) != set(dataset.pcap_sha256)
        or set(capture_counts.values()) != {SESSIONS_PER_CAPTURE}
    ):
        raise ValueError(
            f"named training dataset must contain {SESSIONS_PER_CAPTURE} sessions per capture"
        )
    split_counts = tuple(
        len(partition) for partition in (split.train, split.validation, split.test)
    )
    if split_counts != EXPECTED_SPLIT_COUNTS:
        raise ValueError(
            "named training dataset split must be "
            f"{EXPECTED_SPLIT_COUNTS[0]}/{EXPECTED_SPLIT_COUNTS[1]}/{EXPECTED_SPLIT_COUNTS[2]} records"
        )
    expected_environments = ("lab_train", "lab_calibration", "lab_test")
    expected_captures = (105, 35, 105)
    for partition, environment, capture_count in zip(
        (split.train, split.validation, split.test),
        expected_environments,
        expected_captures,
        strict=True,
    ):
        if {record.provenance.environment_id for record in partition} != {environment}:
            raise ValueError(f"{environment} split contains records from another environment")
        if len({record.provenance.capture_id for record in partition}) != capture_count:
            raise ValueError(f"{environment} split must contain {capture_count} captures")


def _format_duration(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return (
        f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        if hours
        else f"{minutes:02d}:{seconds:02d}"
    )


def format_progress(
    phase: str,
    *,
    completed: int,
    total: int,
    elapsed_seconds: float,
    status_counts: dict[str, int] | None = None,
) -> str:
    if total <= 0:
        raise ValueError("progress total must be positive")
    completed = max(0, min(completed, total))
    width = 20
    filled = int(width * completed / total)
    bar = "#" * filled + "." * (width - filled)
    eta = "n/a" if completed == 0 else _format_duration(
        elapsed_seconds * (total - completed) / completed
    )
    statuses = " ".join(
        f"{name}={count}"
        for name, count in sorted((status_counts or {}).items())
    )
    suffix = f" statuses={statuses}" if statuses else ""
    return (
        f"{phase}: [{bar}] {completed}/{total} "
        f"elapsed={_format_duration(elapsed_seconds)} ETA={eta}{suffix}"
    )


class _ProgressReporter:
    def __init__(self) -> None:
        self.started = time.monotonic()
        self.status_counts: Counter[str] = Counter()

    def phase(self, name: str) -> None:
        print(
            f"\n[{name}] elapsed="
            f"{_format_duration(time.monotonic() - self.started)} ETA=n/a"
        )

    def matrix_complete(self, completed: int, total: int, result: LabRun) -> None:
        self.status_counts[result.status] += 1
        print(
            "\r" + format_progress(
                "matrix capture",
                completed=completed,
                total=total,
                elapsed_seconds=time.monotonic() - self.started,
                status_counts=dict(self.status_counts),
            ),
            end="",
            flush=True,
        )
        if completed == total:
            print()

    def extraction_complete(self, completed: int, total: int) -> None:
        print(
            "\r" + format_progress(
                "PCAP extraction",
                completed=completed,
                total=total,
                elapsed_seconds=time.monotonic() - self.started,
                status_counts={"success": completed},
            ),
            end="",
            flush=True,
        )
        if completed == total:
            print()


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
    show_progress: bool = False,
) -> tuple[DatasetRun, Path, Path, EvaluationReport]:
    reporter = _ProgressReporter() if show_progress else None
    if reporter:
        reporter.phase("configuration")
    model_config, fusion_config = load_training_config(config_path)
    if model_config.random_seed != seed:
        model_config = model_config.model_copy(update={"random_seed": seed})

    named_path = Path(dataset_root) / DATASET_RUN.name
    manifest_path = named_path / "run_manifest.json"
    if manifest_path.is_file():
        if reporter:
            reporter.phase("matrix capture: reused Dataset-17K")
        dataset, split = load_dataset_run(named_path)
        validate_named_dataset(dataset, split, expected_seed=seed)
        dataset_run = DatasetRun(
            named_path,
            named_path.name,
            json.loads(manifest_path.read_text()),
        )
    else:
        if reporter:
            reporter.phase("matrix capture")
        runs = run_training_matrix(
            seed,
            lab_root,
            on_run_complete=reporter.matrix_complete if reporter else None,
        )
        if reporter:
            reporter.phase("PCAP extraction")
        dataset_config = _dataset_config(seed)
        dataset = assemble_successful_runs(
            runs,
            dataset_config,
            on_capture_complete=reporter.extraction_complete if reporter else None,
        )
        ensure_training_matrix_ready(runs, dataset)
        split = split_dataset(
            dataset,
            {
                "random_seed": seed,
                "validation_environment_id": "lab_calibration",
                "test_environment_id": "lab_test",
            },
        )
        validate_named_dataset(dataset, split, expected_seed=seed)
        dataset_run = write_dataset_run(
            dataset,
            split,
            dataset_root,
            run_id=DATASET_RUN.name,
        )

    if reporter:
        reporter.phase("validation")
        print(
            f"dataset records={len(dataset.records)} captures={len(dataset.pcap_sha256)} "
            f"split={len(split.train)}/{len(split.validation)}/{len(split.test)}"
        )
        reporter.phase("split")
        reporter.phase("training")
    bundle = train_model_bundle(split, model_config)
    if reporter:
        reporter.phase("calibration")
    calibration = calibrate_scores(
        predict_model_outputs(bundle, split.validation),
        split.validation,
    )
    if reporter:
        reporter.phase("evaluation")
    report = evaluate_bundle(
        bundle,
        calibration,
        split,
        fusion_config=fusion_config,
    )
    if reporter:
        reporter.phase("persistence")
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
    configured_model, configured_fusion = load_training_config(args.config)
    seed = configured_model.random_seed if args.seed is None else args.seed
    dataset_run, saved_bundle, evaluation_path, report = train_packet_matrix(
        seed=seed,
        lab_root=args.root,
        dataset_root=args.dataset_root,
        bundle_dir=args.bundle,
        config_path=args.config,
        show_progress=True,
    )
    print("dataset", dataset_run.path)
    print("bundle", saved_bundle)
    print("evaluation", evaluation_path)
    print(format_evaluation_table(report, configured_fusion))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

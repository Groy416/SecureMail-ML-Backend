"""Evaluate a persisted packet-backed training run against its saved bundle."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.calibration import load_calibration_state
from ml.dataset import load_dataset_run
from ml.evaluate import evaluate_bundle, format_evaluation_table, save_evaluation_report
from ml.models import load_model_bundle

DATASET_RUN = Path("datasets/runs/Dataset-17K")
BUNDLE = Path("models/Model_XG_RF")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-run", default=DATASET_RUN, type=Path)
    parser.add_argument("--bundle", default=BUNDLE, type=Path)
    parser.add_argument("--output-root", default=Path("evals"), type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _, split = load_dataset_run(args.dataset_run)
    report = evaluate_bundle(
        load_model_bundle(args.bundle),
        load_calibration_state(args.bundle),
        split,
    )
    print(format_evaluation_table(report))
    print(save_evaluation_report(report, args.output_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

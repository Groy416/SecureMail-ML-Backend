"""Evaluate the frozen bundle on held-out families and optional PCAP captures."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets.lab.holdout import (
    collect_pcap_holdout_records,
    docker_available,
    eval_split,
    generate_heldout_family_records,
)
from ml.calibration import load_calibration_state
from ml.evaluate import evaluate_bundle, save_evaluation_report
from ml.models import load_model_bundle

BUNDLE = "models/grouped-69-capture"


def _print_report(title: str, path: Path, report) -> None:
    print(title)
    print(path)
    print(
        json.dumps(
            {
                "test_session_count": report.test_session_count,
                "fusion_accuracy": report.fusion["accuracy"],
                "fusion_critical_recall": report.fusion["critical_recall"],
                "xgboost_accuracy": report.classifiers["xgboost"]["accuracy"],
                "random_forest_accuracy": report.classifiers["random_forest"]["accuracy"],
            },
            indent=2,
        )
    )


def main() -> None:
    bundle = load_model_bundle(BUNDLE)
    calibration = load_calibration_state(BUNDLE)

    family_split = eval_split(generate_heldout_family_records())
    family_report = evaluate_bundle(bundle, calibration, family_split)
    family_path = save_evaluation_report(family_report, "evals")
    _print_report("held-out families (not in the 5k trainer)", family_path, family_report)

    if not docker_available():
        print("pcap holdout skipped: Docker is unavailable")
        return
    captured = collect_pcap_holdout_records()
    print("pcap runs", json.dumps(captured["runs"], indent=2))
    if not captured["records"]:
        print("pcap holdout produced no successful sessions")
        return
    pcap_report = evaluate_bundle(bundle, calibration, eval_split(captured["records"]))
    pcap_path = save_evaluation_report(pcap_report, "evals")
    _print_report("pcap holdout (frozen bundle, packet-backed)", pcap_path, pcap_report)


if __name__ == "__main__":
    main()

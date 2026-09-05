from __future__ import annotations

from pathlib import Path

import pytest

from scripts.train_evaluate_grouped_matrix import (
    BUNDLE_DIR,
    DATASET_RUN,
    ensure_training_matrix_ready,
    format_progress,
    load_training_config,
)


def test_training_script_rejects_an_incomplete_matrix() -> None:
    with pytest.raises(ValueError, match="245 profiles"):
        ensure_training_matrix_ready((), None)


def test_training_script_loads_the_pcap_training_config() -> None:
    model_config, fusion_config = load_training_config(
        Path("configs/training.pcap.json")
    )

    assert model_config.random_seed == 420042
    assert fusion_config.xgboost_weight == 0.6
    assert fusion_config.random_forest_weight == 0.4
    assert fusion_config.isolation_forest_weight == 0.0


def test_training_script_uses_readable_artifact_defaults() -> None:
    assert DATASET_RUN == Path("datasets/runs/Dataset-17K")
    assert BUNDLE_DIR == Path("models/Model_XG_RF")


def test_progress_formatter_reports_elapsed_eta_and_status_counts() -> None:
    output = format_progress(
        "matrix capture",
        completed=2,
        total=4,
        elapsed_seconds=10.0,
        status_counts={"success": 1, "failed": 1},
    )

    assert "2/4" in output
    assert "elapsed=00:10" in output
    assert "ETA=00:10" in output
    assert "success=1" in output and "failed=1" in output

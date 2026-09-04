from __future__ import annotations

from pathlib import Path

import pytest

from scripts.train_evaluate_grouped_matrix import (
    ensure_training_matrix_ready,
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
    assert fusion_config.xgboost_weight == 0.5
    assert fusion_config.random_forest_weight == 0.3
    assert fusion_config.isolation_forest_weight == 0.2

from __future__ import annotations

import numpy as np

from ml.calibration import calibrate_scores, threshold_for_target_recall
from ml.dataset import generate_feature_dataset, split_dataset
from ml.features import ISOLATION_FOREST_FEATURES
from ml.models import predict_model_outputs, train_model_bundle


def test_isolation_forest_features_drop_session_jitter() -> None:
    assert "session_duration_seconds" not in ISOLATION_FOREST_FEATURES
    assert "packet_count" not in ISOLATION_FOREST_FEATURES
    assert "tls_version" in ISOLATION_FOREST_FEATURES
    assert "cert_chain_valid" in ISOLATION_FOREST_FEATURES


def test_threshold_meets_calibration_recall_target() -> None:
    scores = np.asarray([0.1, 0.2, 0.3, 0.8, 0.9, 1.0])
    labels = np.asarray([0, 0, 0, 1, 1, 1])

    threshold = threshold_for_target_recall(scores, labels, 0.8)

    recall = float(np.mean(scores[labels == 1] >= threshold))
    assert recall >= 0.8
    assert threshold >= 0.8


def test_two_model_calibration_disables_isolation_forest() -> None:
    dataset = generate_feature_dataset(
        {"master_seed": 420042, "session_count": 500}
    )
    split = split_dataset(dataset, {"random_seed": 420042})
    bundle = train_model_bundle(
        split,
        {
            "random_seed": 420042,
            "enable_isolation_forest": False,
            "xgboost_n_estimators": 2,
            "random_forest_n_estimators": 2,
        },
    )

    state = calibrate_scores(
        predict_model_outputs(bundle, split.validation),
        split.validation,
    )

    assert state.anomaly_enabled is False
    assert state.normal_low is None
    assert state.normal_high is None
    assert state.anomaly_threshold is None

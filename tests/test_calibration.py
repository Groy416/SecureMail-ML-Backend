from __future__ import annotations

import numpy as np

from ml.calibration import threshold_for_target_recall
from ml.features import ISOLATION_FOREST_FEATURES


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

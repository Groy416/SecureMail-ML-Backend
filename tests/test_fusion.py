from __future__ import annotations

import pytest
from sklearn.isotonic import IsotonicRegression

from ml.calibration import CalibrationState
from ml.fusion import FusionConfig, fuse_session
from ml.models import (
    RISK_LABELS,
    ClassifierOutput,
    IsolationForestOutput,
    ModelOutputs,
)
from ml.schema import ModelAction, RiskLabel
from tests.test_rules import _record


def _identity_calibrators() -> tuple[IsotonicRegression, ...]:
    calibrators = []
    for _ in RISK_LABELS:
        calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        calibrator.fit([0.0, 1.0], [0.0, 1.0])
        calibrators.append(calibrator)
    return tuple(calibrators)


def test_isolation_forest_only_flag_requires_analyst_review() -> None:
    probabilities = {label: 0.0 for label in RISK_LABELS}
    probabilities[RiskLabel.INFORMATIONAL] = 1.0
    output = ModelOutputs(
        xgboost=ClassifierOutput(
            predicted_class=RiskLabel.INFORMATIONAL,
            class_probabilities=probabilities,
        ),
        random_forest=ClassifierOutput(
            predicted_class=RiskLabel.INFORMATIONAL,
            class_probabilities=probabilities,
        ),
        isolation_forest=IsolationForestOutput(raw_score=-1.0),
        diagnostics={},
    )
    calibration = CalibrationState(
        xgboost=_identity_calibrators(),
        random_forest=_identity_calibrators(),
        normal_low=-0.1,
        normal_high=0.1,
        anomaly_threshold=0.5,
        anomaly_enabled=True,
    )

    result = fuse_session(_record(), output, calibration)

    assert result.risk.risk_class is RiskLabel.LOW
    assert result.action is ModelAction.ANALYST_REVIEW
    assert result.anomaly.detected is True
    assert result.model_outputs["isolation_forest"]["flagged"] is True


def test_default_fusion_weights_include_isolation_forest() -> None:
    config = FusionConfig()

    assert config.xgboost_weight == 0.50
    assert config.random_forest_weight == 0.30
    assert config.isolation_forest_weight == 0.20


def test_non_flagged_isolation_score_does_not_lower_model_risk() -> None:
    probabilities = {label: 0.0 for label in RISK_LABELS}
    probabilities[RiskLabel.HIGH] = 1.0
    output = ModelOutputs(
        xgboost=ClassifierOutput(
            predicted_class=RiskLabel.HIGH,
            class_probabilities=probabilities,
        ),
        random_forest=ClassifierOutput(
            predicted_class=RiskLabel.HIGH,
            class_probabilities=probabilities,
        ),
        isolation_forest=IsolationForestOutput(raw_score=0.1),
        diagnostics={},
    )
    calibration = CalibrationState(
        xgboost=_identity_calibrators(),
        random_forest=_identity_calibrators(),
        normal_low=-0.1,
        normal_high=0.1,
        anomaly_threshold=0.5,
        anomaly_enabled=True,
    )

    result = fuse_session(_record(), output, calibration)

    assert result.anomaly.detected is False
    assert result.risk.risk_class is RiskLabel.HIGH
    assert result.risk.score == pytest.approx(0.75)

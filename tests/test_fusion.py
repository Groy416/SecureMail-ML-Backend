from __future__ import annotations

import pytest
from sklearn.isotonic import IsotonicRegression

from ml.calibration import CalibrationState
from ml.fusion import FusionConfig, fuse_session, fused_class_probabilities
from ml.models import RISK_LABELS, ClassifierOutput, ModelOutputs
from ml.schema import RiskLabel
from tests.test_rules import _record


def _identity_calibrators() -> tuple[IsotonicRegression, ...]:
    calibrators = []
    for _ in RISK_LABELS:
        calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        calibrator.fit([0.0, 1.0], [0.0, 1.0])
        calibrators.append(calibrator)
    return tuple(calibrators)


def _two_model_output(
    xgboost: dict[RiskLabel, float],
    random_forest: dict[RiskLabel, float],
) -> ModelOutputs:
    return ModelOutputs(
        xgboost=ClassifierOutput(
            predicted_class=max(xgboost, key=xgboost.get),
            class_probabilities=xgboost,
        ),
        random_forest=ClassifierOutput(
            predicted_class=max(random_forest, key=random_forest.get),
            class_probabilities=random_forest,
        ),
        diagnostics={},
    )


def test_two_model_result_marks_isolation_forest_disabled() -> None:
    probabilities = {label: 0.0 for label in RISK_LABELS}
    probabilities[RiskLabel.HIGH] = 1.0
    output = _two_model_output(probabilities, probabilities)
    calibration = CalibrationState(
        xgboost=_identity_calibrators(),
        random_forest=_identity_calibrators(),
    )

    result = fuse_session(_record(), output, calibration)

    assert result.risk.risk_class is RiskLabel.HIGH
    assert result.anomaly.status == "disabled"
    assert result.anomaly.detected is False
    assert "isolation_forest" not in result.model_outputs


def test_default_fusion_weights_use_xgboost_and_random_forest() -> None:
    config = FusionConfig()

    assert config.xgboost_weight == 0.60
    assert config.random_forest_weight == 0.40
    assert config.isolation_forest_weight == 0.0


def test_fused_class_probabilities_use_configured_model_weights() -> None:
    xgboost = {label: 0.0 for label in RISK_LABELS}
    random_forest = {label: 0.0 for label in RISK_LABELS}
    xgboost[RiskLabel.HIGH] = 1.0
    random_forest[RiskLabel.CRITICAL] = 1.0
    output = _two_model_output(xgboost, random_forest)

    probabilities = fused_class_probabilities(
        output,
        CalibrationState(
            xgboost=_identity_calibrators(),
            random_forest=_identity_calibrators(),
        ),
        FusionConfig(xgboost_weight=0.6, random_forest_weight=0.4),
    )

    assert probabilities[RiskLabel.HIGH] == pytest.approx(0.6)
    assert probabilities[RiskLabel.CRITICAL] == pytest.approx(0.4)


def test_two_model_fusion_uses_the_fused_class() -> None:
    probabilities = {label: 0.0 for label in RISK_LABELS}
    probabilities[RiskLabel.HIGH] = 1.0
    output = _two_model_output(probabilities, probabilities)
    calibration = CalibrationState(
        xgboost=_identity_calibrators(),
        random_forest=_identity_calibrators(),
    )

    result = fuse_session(_record(), output, calibration)

    assert result.risk.risk_class is RiskLabel.HIGH
    assert result.risk.score == pytest.approx(0.75)

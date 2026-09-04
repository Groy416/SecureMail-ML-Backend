from __future__ import annotations

from sklearn.isotonic import IsotonicRegression

from ml.calibration import CalibrationState
from ml.fusion import fuse_session
from ml.models import (
    RISK_LABELS,
    ClassifierOutput,
    IsolationForestOutput,
    ModelOutputs,
)
from ml.schema import RiskLabel
from tests.test_rules import _record


def _identity_calibrators() -> tuple[IsotonicRegression, ...]:
    calibrators = []
    for _ in RISK_LABELS:
        calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        calibrator.fit([0.0, 1.0], [0.0, 1.0])
        calibrators.append(calibrator)
    return tuple(calibrators)


def test_mail_check_risk_ignores_unusable_isolation_forest() -> None:
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

    assert result.risk.risk_class is RiskLabel.INFORMATIONAL
    assert result.action.value == "no_action"
    assert result.anomaly.detected is True

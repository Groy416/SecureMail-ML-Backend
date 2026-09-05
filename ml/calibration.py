from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression

from ml.models import ModelOutputs, RISK_LABELS
from ml.schema import RiskLabel, SessionFeatureRecord

CLASS_CENTERS: dict[RiskLabel, float] = {
    RiskLabel.INFORMATIONAL: 0.0,
    RiskLabel.LOW: 0.25,
    RiskLabel.MEDIUM: 0.50,
    RiskLabel.HIGH: 0.75,
    RiskLabel.CRITICAL: 1.0,
}
CALIBRATION_VERSION = "calibration.v2"


@dataclass
class CalibrationState:
    xgboost: tuple[IsotonicRegression, ...]
    random_forest: tuple[IsotonicRegression, ...]
    normal_low: float | None = None
    normal_high: float | None = None
    anomaly_threshold: float | None = None
    anomaly_enabled: bool = False
    baseline_id: str = "normal-baseline.v1"
    version: str = CALIBRATION_VERSION


def risk_probability(class_probabilities: dict[RiskLabel, float]) -> float:
    return float(
        sum(
            probability * CLASS_CENTERS[label]
            for label, probability in class_probabilities.items()
        )
    )


def _probability_matrix(
    outputs: Sequence[ModelOutputs],
    model_name: str,
) -> np.ndarray:
    return np.asarray(
        [
            [
                getattr(output, model_name).class_probabilities[label]
                for label in RISK_LABELS
            ]
            for output in outputs
        ],
        dtype=float,
    )


def _fit_calibrators(
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> tuple[IsotonicRegression, ...]:
    return tuple(
        IsotonicRegression(
            y_min=0.0,
            y_max=1.0,
            out_of_bounds="clip",
        ).fit(probabilities[:, index], (labels == index).astype(float))
        for index in range(len(RISK_LABELS))
    )


def normalized_anomaly_score(
    raw_score: float,
    state: CalibrationState,
) -> float:
    if not state.anomaly_enabled:
        return 0.0
    if state.normal_low is None or state.normal_high is None:
        raise ValueError("normal calibration bounds are required")
    denominator = state.normal_high - state.normal_low
    if denominator <= 0:
        raise ValueError("normal calibration scores must have non-zero range")
    return float(
        np.clip(
            (state.normal_high - raw_score) / denominator,
            0.0,
            1.0,
        )
    )


def threshold_for_target_recall(
    scores: np.ndarray,
    labels: np.ndarray,
    target_recall: float,
) -> float:
    if not 0 < target_recall <= 1:
        raise ValueError("target_recall must be in (0, 1]")
    anomalous = scores[labels == 1]
    if len(anomalous) == 0:
        return 1.0
    best_threshold = float(np.quantile(anomalous, 1.0 - target_recall))
    best_fpr = 1.0
    has_normal = (labels == 0).any()
    for threshold in np.unique(scores):
        recall = float(np.mean(anomalous >= threshold))
        if recall + 1e-12 < target_recall:
            continue
        false_positive_rate = (
            float(np.mean(scores[labels == 0] >= threshold)) if has_normal else 0.0
        )
        if false_positive_rate < best_fpr or (
            np.isclose(false_positive_rate, best_fpr) and threshold > best_threshold
        ):
            best_fpr = false_positive_rate
            best_threshold = float(threshold)
    return best_threshold


def calibrate_scores(
    outputs: Sequence[ModelOutputs],
    records: Sequence[SessionFeatureRecord],
) -> CalibrationState:
    if len(outputs) != len(records) or not records:
        raise ValueError("calibration outputs and records must be non-empty and aligned")

    labels = np.asarray(
        [RISK_LABELS.index(record.labels.risk_label) for record in records],
        dtype=int,
    )
    xgboost_calibrators = _fit_calibrators(
        _probability_matrix(outputs, "xgboost"), labels
    )
    random_forest_calibrators = _fit_calibrators(
        _probability_matrix(outputs, "random_forest"), labels
    )
    if all(output.isolation_forest is None for output in outputs):
        return CalibrationState(
            xgboost=xgboost_calibrators,
            random_forest=random_forest_calibrators,
        )
    if any(output.isolation_forest is None for output in outputs):
        raise ValueError("Isolation Forest outputs must be aligned across calibration rows")

    normal_raw_scores = np.asarray(
        [
            output.isolation_forest.raw_score
            for output, record in zip(outputs, records, strict=True)
            if record.labels.risk_label is RiskLabel.INFORMATIONAL
            and record.labels.anomaly_label.value == 0
        ],
        dtype=float,
    )
    if len(normal_raw_scores) < 2:
        raise ValueError("calibration requires at least two normal baseline rows")

    normal_low, normal_high = np.percentile(normal_raw_scores, [1, 99])
    anomaly_enabled = normal_high > normal_low
    state = CalibrationState(
        xgboost=xgboost_calibrators,
        random_forest=random_forest_calibrators,
        normal_low=float(normal_low),
        normal_high=float(normal_high),
        anomaly_threshold=1.0,
        anomaly_enabled=anomaly_enabled,
    )
    if anomaly_enabled:
        normalized_normal_scores = np.asarray(
            [
                normalized_anomaly_score(output.isolation_forest.raw_score, state)
                for output, record in zip(outputs, records, strict=True)
                if record.labels.risk_label is RiskLabel.INFORMATIONAL
                and record.labels.anomaly_label.value == 0
            ],
            dtype=float,
        )
        state.anomaly_threshold = float(np.percentile(normalized_normal_scores, 99))
    return state


def calibrated_class_probabilities(
    output: ModelOutputs,
    state: CalibrationState,
) -> tuple[dict[RiskLabel, float], dict[RiskLabel, float]]:
    calibrated: list[dict[RiskLabel, float]] = []
    for model_name, calibrators in (
        ("xgboost", state.xgboost),
        ("random_forest", state.random_forest),
    ):
        raw = getattr(output, model_name).class_probabilities
        values = np.asarray(
            [
                calibrator.predict([raw[label]])[0]
                for label, calibrator in zip(RISK_LABELS, calibrators, strict=True)
            ],
            dtype=float,
        )
        total = values.sum()
        if total <= 0:
            values = np.asarray([raw[label] for label in RISK_LABELS], dtype=float)
            total = values.sum()
        values /= total
        calibrated.append(
            {label: float(values[index]) for index, label in enumerate(RISK_LABELS)}
        )
    return calibrated[0], calibrated[1]


def calibrated_risk_probability(
    output: ModelOutputs,
    state: CalibrationState,
) -> tuple[float, float]:
    xgboost, random_forest = calibrated_class_probabilities(output, state)
    return risk_probability(xgboost), risk_probability(random_forest)


def _calibrator_metadata(
    calibrators: tuple[IsotonicRegression, ...],
) -> dict[str, dict[str, list[float]]]:
    return {
        label.value: {
            "x_thresholds": calibrator.X_thresholds_.tolist(),
            "y_thresholds": calibrator.y_thresholds_.tolist(),
        }
        for label, calibrator in zip(RISK_LABELS, calibrators, strict=True)
    }


def save_calibration_state(state: CalibrationState, directory: str | Path) -> None:
    path = Path(directory)
    calibration_path = path / "calibration.json"
    thresholds_path = path / "thresholds.json"
    state_path = path / "calibration.joblib"
    if any(target.exists() for target in (calibration_path, thresholds_path, state_path)):
        raise FileExistsError(f"calibration artifacts already exist: {path}")
    path.mkdir(parents=True, exist_ok=True)
    calibration_path.write_text(
        json.dumps(
            {
                "version": state.version,
                "method": "isotonic_ovr_class_probability",
                "xgboost": _calibrator_metadata(state.xgboost),
                "random_forest": _calibrator_metadata(state.random_forest),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    if state.anomaly_enabled:
        thresholds_path.write_text(
            json.dumps(
                {
                    "baseline_id": state.baseline_id,
                    "normal_low": state.normal_low,
                    "normal_high": state.normal_high,
                    "anomaly_threshold": state.anomaly_threshold,
                    "anomaly_enabled": True,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
    joblib.dump(state, state_path)


def load_calibration_state(directory: str | Path) -> CalibrationState:
    path = Path(directory)
    metadata = json.loads((path / "calibration.json").read_text())
    state = joblib.load(path / "calibration.joblib")
    if not isinstance(state, CalibrationState):
        raise ValueError("artifact is not a SecureMailScope calibration state")
    if metadata["version"] != CALIBRATION_VERSION or state.version != CALIBRATION_VERSION:
        raise ValueError("unsupported calibration version")
    return state

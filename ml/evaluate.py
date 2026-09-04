from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from ml.calibration import (
    CalibrationState,
    calibrated_class_probabilities,
    normalized_anomaly_score,
)
from ml.dataset import SplitArtifact, records_hash
from ml.fusion import fuse_session
from ml.models import (
    MODEL_BUNDLE_VERSION,
    RISK_LABELS,
    RISK_LABEL_TO_INDEX,
    ModelBundle,
    ModelOutputs,
    predict_model_outputs,
)
from ml.rules import extract_rule_findings
from ml.schema import RiskLabel

EVALUATION_VERSION = "evaluation.v1"


@dataclass(frozen=True)
class EvaluationReport:
    schema_version: str
    model_bundle_version: str
    model_bundle_id: str
    calibration_version: str
    test_split_sha256: str
    test_session_count: int
    test_environment_ids: tuple[str, ...]
    classifiers: dict[str, dict[str, Any]]
    isolation_forest: dict[str, Any]
    fusion: dict[str, Any]
    limitations: tuple[str, ...]


def _probability_matrix(
    outputs: list[ModelOutputs],
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


def _calibrated_probability_matrix(
    outputs: list[ModelOutputs],
    calibration: CalibrationState,
    model_name: str,
) -> np.ndarray:
    position = 0 if model_name == "xgboost" else 1
    return np.asarray(
        [
            [
                calibrated_class_probabilities(output, calibration)[position][label]
                for label in RISK_LABELS
            ]
            for output in outputs
        ],
        dtype=float,
    )


def _optional_binary_metric(
    y_true: np.ndarray,
    scores: np.ndarray,
    metric: str,
) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    if metric == "pr_auc":
        return float(average_precision_score(y_true, scores))
    return float(roc_auc_score(y_true, scores))


def _label_metrics(predictions: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    indexes = list(range(len(RISK_LABELS)))
    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        predictions,
        labels=indexes,
        zero_division=0,
    )
    present = [index for index in indexes if support[index] > 0]
    per_class = {
        label.value: {
            "precision": float(precision[index]) if support[index] else None,
            "recall": float(recall[index]) if support[index] else None,
            "f1": float(f1[index]) if support[index] else None,
            "support": int(support[index]),
        }
        for index, label in enumerate(RISK_LABELS)
    }
    return {
        "accuracy": float(np.mean(predictions == labels)),
        "per_class": per_class,
        "macro_f1": (
            float(f1_score(labels, predictions, labels=present, average="macro"))
            if present
            else None
        ),
        "weighted_f1": (
            float(f1_score(labels, predictions, labels=present, average="weighted"))
            if present
            else None
        ),
        "critical_precision": per_class[RiskLabel.CRITICAL.value]["precision"],
        "critical_recall": per_class[RiskLabel.CRITICAL.value]["recall"],
        "confusion_matrix": confusion_matrix(
            labels,
            predictions,
            labels=indexes,
        ).tolist(),
    }


def _classifier_metrics(
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> dict[str, Any]:
    predictions = probabilities.argmax(axis=1)
    metrics = _label_metrics(predictions, labels)
    for index, label in enumerate(RISK_LABELS):
        binary_labels = (labels == index).astype(int)
        metrics["per_class"][label.value].update(
            {
                "pr_auc": _optional_binary_metric(
                    binary_labels,
                    probabilities[:, index],
                    "pr_auc",
                ),
                "roc_auc": _optional_binary_metric(
                    binary_labels,
                    probabilities[:, index],
                    "roc_auc",
                ),
            }
        )
    one_hot = np.eye(len(RISK_LABELS), dtype=float)[labels]
    metrics["multiclass_brier_score"] = float(
        np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))
    )
    return metrics


def _score_distribution(values: np.ndarray) -> dict[str, float] | None:
    if len(values) == 0:
        return None
    return {
        "count": float(len(values)),
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
    }


def evaluate_bundle(
    bundle: ModelBundle,
    calibration: CalibrationState,
    split: SplitArtifact,
) -> EvaluationReport:
    if bundle.version != MODEL_BUNDLE_VERSION:
        raise ValueError("unsupported model bundle version")
    outputs = predict_model_outputs(bundle, split.test)
    labels = np.asarray(
        [RISK_LABEL_TO_INDEX[record.labels.risk_label] for record in split.test],
        dtype=int,
    )
    anomaly_labels = np.asarray(
        [record.labels.anomaly_label.value for record in split.test],
        dtype=int,
    )
    classifiers = {
        "xgboost": _classifier_metrics(
            _calibrated_probability_matrix(outputs, calibration, "xgboost"),
            labels,
        ),
        "random_forest": _classifier_metrics(
            _calibrated_probability_matrix(outputs, calibration, "random_forest"),
            labels,
        ),
    }
    fused_results = [
        fuse_session(record, output, calibration, extract_rule_findings(record))
        for record, output in zip(split.test, outputs, strict=True)
    ]
    fusion_labels = np.asarray(
        [RISK_LABEL_TO_INDEX[result.risk.risk_class] for result in fused_results],
        dtype=int,
    )
    raw_scores = np.asarray(
        [output.isolation_forest.raw_score for output in outputs],
        dtype=float,
    )
    normal_mask = anomaly_labels == 0
    if calibration.anomaly_enabled:
        anomaly_scores = np.asarray(
            [normalized_anomaly_score(score, calibration) for score in raw_scores],
            dtype=float,
        )
        anomaly_predictions = (anomaly_scores >= calibration.anomaly_threshold).astype(int)
        anomaly_precision, anomaly_recall, anomaly_f1, _ = precision_recall_fscore_support(
            anomaly_labels,
            anomaly_predictions,
            labels=[1],
            zero_division=0,
        )
        isolation_metrics: dict[str, Any] = {
            "status": "calibrated",
            "threshold": calibration.anomaly_threshold,
            "precision": float(anomaly_precision[0]),
            "recall": float(anomaly_recall[0]),
            "f1": float(anomaly_f1[0]),
            "false_positive_rate": (
                float(anomaly_predictions[normal_mask].mean())
                if normal_mask.any()
                else None
            ),
        }
    else:
        isolation_metrics = {
            "status": "unavailable:degenerate_normal_calibration",
            "threshold": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "false_positive_rate": None,
        }
    isolation_metrics.update(
        {
            "normal": _score_distribution(raw_scores[anomaly_labels == 0]),
            "anomalous": _score_distribution(raw_scores[anomaly_labels == 1]),
        }
    )
    return EvaluationReport(
        schema_version=EVALUATION_VERSION,
        model_bundle_version=bundle.version,
        model_bundle_id=bundle.bundle_id,
        calibration_version=calibration.version,
        test_split_sha256=records_hash(list(split.test)),
        test_session_count=len(split.test),
        test_environment_ids=tuple(
            sorted({record.provenance.environment_id for record in split.test})
        ),
        classifiers=classifiers,
        isolation_forest=isolation_metrics,
        fusion=_label_metrics(fusion_labels, labels),
        limitations=(
            "Scores are synthetic-lab-only and must not be presented as production performance.",
            "Class-probability calibration is isotonic and may overfit this small synthetic calibration partition.",
            "Learned stacking is disabled until grouped out-of-fold normal-baseline coverage exists.",
        ),
    )


def save_evaluation_report(
    report: EvaluationReport,
    root: str | Path = "evals",
) -> Path:
    payload = asdict(report)
    evaluation_hash = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    path = Path(root) / f"evaluation-{evaluation_hash[:12]}"
    metrics_path = path / "metrics.json"
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if metrics_path.exists():
        if metrics_path.read_text() == encoded:
            return path
        raise FileExistsError(f"evaluation directory conflicts: {path}")
    if path.exists():
        raise FileExistsError(f"evaluation directory already exists: {path}")
    path.mkdir(parents=True)
    metrics_path.write_text(encoded)
    return path


if __name__ == "__main__":
    from tempfile import TemporaryDirectory

    from ml.calibration import calibrate_scores
    from ml.dataset import generate_feature_dataset, split_dataset
    from ml.models import train_model_bundle

    split = split_dataset(
        generate_feature_dataset({"master_seed": 420042, "session_count": 500}),
        {"random_seed": 420042},
    )
    bundle = train_model_bundle(split, {"random_seed": 420042})
    calibration = calibrate_scores(
        predict_model_outputs(bundle, split.validation),
        split.validation,
    )
    report = evaluate_bundle(bundle, calibration, split)
    assert set(report.classifiers) == {"xgboost", "random_forest"}
    assert report.fusion["critical_recall"] is not None
    with TemporaryDirectory() as directory:
        path = save_evaluation_report(report, directory)
        assert (path / "metrics.json").is_file()
    print(report.test_session_count, report.fusion["critical_recall"])

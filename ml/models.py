from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import joblib
import numpy as np
from pydantic import Field, model_validator
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from xgboost import XGBClassifier

from ml.dataset import SplitArtifact, records_hash
from ml.features import (
    BOOLEAN_FEATURES,
    CATEGORICAL_FEATURES,
    ISOLATION_FOREST_FEATURES,
    MODEL_INPUT_FEATURES,
    NUMERIC_FEATURES,
    extract_model_features,
)
from ml.preprocess import (
    PreprocessorState,
    build_feature_matrix,
    load_preprocessor,
    save_preprocessor,
)
from ml.schema import ContractModel, RiskLabel, SessionFeatureRecord

if TYPE_CHECKING:
    from ml.calibration import CalibrationState

RISK_LABELS: tuple[RiskLabel, ...] = (
    RiskLabel.INFORMATIONAL,
    RiskLabel.LOW,
    RiskLabel.MEDIUM,
    RiskLabel.HIGH,
    RiskLabel.CRITICAL,
)
RISK_LABEL_TO_INDEX = {label: index for index, label in enumerate(RISK_LABELS)}
MODEL_BUNDLE_VERSION = "ml-bundle.v2"
SUPPORTED_MODEL_BUNDLE_VERSIONS = {"ml-bundle.v1", MODEL_BUNDLE_VERSION}


class ModelConfig(ContractModel):
    random_seed: int = Field(ge=0)
    enable_isolation_forest: bool = False
    xgboost_n_estimators: int = Field(default=300, gt=0)
    xgboost_max_depth: int = Field(default=5, gt=0)
    xgboost_learning_rate: float = Field(default=0.05, gt=0)
    xgboost_subsample: float = Field(default=0.8, gt=0, le=1)
    xgboost_colsample_bytree: float = Field(default=0.8, gt=0, le=1)
    random_forest_n_estimators: int = Field(default=300, gt=0)
    random_forest_min_samples_leaf: int = Field(default=2, gt=0)
    n_jobs: int = 1
    isolation_contamination: float | str = "auto"
    source_features: tuple[str, ...] | None = None
    isolation_source_features: tuple[str, ...] | None = None

    @model_validator(mode="after")
    def validate_runtime_options(self) -> ModelConfig:
        if self.n_jobs == 0:
            raise ValueError("n_jobs cannot be zero")
        for selected in (self.source_features, self.isolation_source_features):
            if selected is None:
                continue
            unknown = set(selected) - set(MODEL_INPUT_FEATURES)
            if not selected or unknown:
                raise ValueError(f"invalid source features: {sorted(unknown)}")
        if (
            isinstance(self.isolation_contamination, str)
            and self.isolation_contamination != "auto"
        ):
            raise ValueError("isolation_contamination must be 'auto' or a float")
        if (
            isinstance(self.isolation_contamination, float)
            and not 0 < self.isolation_contamination <= 0.5
        ):
            raise ValueError("isolation_contamination must be in (0, 0.5]")
        return self


class ClassifierOutput(ContractModel):
    predicted_class: RiskLabel
    class_probabilities: dict[RiskLabel, float]


class IsolationForestOutput(ContractModel):
    raw_score: float


class ModelOutputs(ContractModel):
    xgboost: ClassifierOutput
    random_forest: ClassifierOutput
    isolation_forest: IsolationForestOutput | None = None
    diagnostics: dict[str, list[str]]


@dataclass
class ModelBundle:
    bundle_id: str
    config: ModelConfig
    preprocessor: PreprocessorState
    xgboost: XGBClassifier
    xgboost_class_indices: tuple[int, ...]
    random_forest: RandomForestClassifier
    training_split_sha256: str
    normal_baseline_count: int = 0
    normal_baseline_features: dict[str, object] | None = None
    isolation_preprocessor: PreprocessorState | None = None
    isolation_forest: IsolationForest | None = None
    version: str = MODEL_BUNDLE_VERSION


def _normal_baseline_features(
    records: Sequence[SessionFeatureRecord],
    source_features: Sequence[str],
) -> dict[str, object]:
    if not records:
        raise ValueError("normal baseline records are required")
    rows = [extract_model_features(record) for record in records]
    baseline: dict[str, object] = {}
    for name in NUMERIC_FEATURES:
        if name not in source_features:
            continue
        values = [row[name] for row in rows if row[name] is not None]
        if not values:
            baseline[name] = None
            continue
        median = np.median(values)
        baseline[name] = (
            int(round(float(median)))
            if isinstance(values[0], int) and not isinstance(values[0], bool)
            else float(median)
        )
    for name in BOOLEAN_FEATURES:
        if name not in source_features:
            continue
        values = [row[name] for row in rows if row[name] is not None]
        if not values:
            baseline[name] = None
            continue
        true_count = sum(values)
        baseline[name] = true_count * 2 >= len(values)
    for name in CATEGORICAL_FEATURES:
        if name not in source_features:
            continue
        values = [row[name] for row in rows if row[name] is not None]
        if not values:
            baseline[name] = None
            continue
        counts = {value: values.count(value) for value in set(values)}
        baseline[name] = min(
            counts,
            key=lambda value: (-counts[value], str(value)),
        )
    return baseline


def _labels(records: Sequence[SessionFeatureRecord]) -> np.ndarray:
    return np.asarray(
        [RISK_LABEL_TO_INDEX[record.labels.risk_label] for record in records],
        dtype=np.int64,
    )


def _five_class_probabilities(
    model: XGBClassifier | RandomForestClassifier,
    matrix: np.ndarray,
    class_indices: Sequence[int],
) -> np.ndarray:
    predicted = model.predict_proba(matrix)
    probabilities = np.zeros((len(matrix), len(RISK_LABELS)), dtype=float)
    for position, class_index in enumerate(class_indices):
        probabilities[:, class_index] = predicted[:, position]
    return probabilities


def _classifier_output(probabilities: np.ndarray) -> ClassifierOutput:
    class_index = int(np.argmax(probabilities))
    return ClassifierOutput(
        predicted_class=RISK_LABELS[class_index],
        class_probabilities={
            label: float(probabilities[index])
            for index, label in enumerate(RISK_LABELS)
        },
    )


def train_model_bundle(
    split: SplitArtifact,
    config: ModelConfig | Mapping[str, Any],
) -> ModelBundle:
    parsed = (
        config
        if isinstance(config, ModelConfig)
        else ModelConfig.model_validate(config)
    )
    labels = _labels(split.train)
    missing_labels = set(range(len(RISK_LABELS))) - set(labels.tolist())
    if missing_labels:
        missing_names = [RISK_LABELS[index].value for index in sorted(missing_labels)]
        raise ValueError(
            f"training split must contain all risk labels; missing: {missing_names}"
        )
    training = build_feature_matrix(
        split.train,
        source_features=parsed.source_features,
    )
    xgboost_class_indices = tuple(sorted(set(labels.tolist())))
    xgboost_label_indices = {
        class_index: position
        for position, class_index in enumerate(xgboost_class_indices)
    }
    xgboost_labels = np.asarray(
        [xgboost_label_indices[label] for label in labels],
        dtype=np.int64,
    )
    xgboost = XGBClassifier(
        objective="multi:softprob",
        num_class=len(xgboost_class_indices),
        eval_metric="mlogloss",
        tree_method="hist",
        n_estimators=parsed.xgboost_n_estimators,
        max_depth=parsed.xgboost_max_depth,
        learning_rate=parsed.xgboost_learning_rate,
        subsample=parsed.xgboost_subsample,
        colsample_bytree=parsed.xgboost_colsample_bytree,
        reg_lambda=1.0,
        random_state=parsed.random_seed,
        n_jobs=parsed.n_jobs,
    ).fit(training.matrix, xgboost_labels)
    random_forest = RandomForestClassifier(
        n_estimators=parsed.random_forest_n_estimators,
        class_weight="balanced",
        min_samples_leaf=parsed.random_forest_min_samples_leaf,
        random_state=parsed.random_seed,
        n_jobs=parsed.n_jobs,
    ).fit(training.matrix, labels)
    isolation_training = None
    isolation_forest = None
    normal_mask = np.zeros(len(split.train), dtype=bool)
    if parsed.enable_isolation_forest:
        normal_mask = np.asarray(
            [
                record.labels.risk_label is RiskLabel.INFORMATIONAL
                and record.labels.anomaly_label.value == 0
                for record in split.train
            ],
            dtype=bool,
        )
        if not normal_mask.any():
            raise ValueError("Isolation Forest requires normal baseline training rows")
        isolation_training = build_feature_matrix(
            split.train,
            source_features=parsed.isolation_source_features or ISOLATION_FOREST_FEATURES,
        )
        isolation_forest = IsolationForest(
            contamination=parsed.isolation_contamination,
            random_state=parsed.random_seed,
            n_jobs=parsed.n_jobs,
        ).fit(isolation_training.matrix[normal_mask])

    training_hash = records_hash(list(split.train))
    bundle_id = hashlib.sha256(
        json.dumps(
            {
                "config": parsed.model_dump(mode="json"),
                "training_split_sha256": training_hash,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()[:16]
    return ModelBundle(
        bundle_id=f"{MODEL_BUNDLE_VERSION}.{bundle_id}",
        config=parsed,
        preprocessor=training.fit_state,
        xgboost=xgboost,
        xgboost_class_indices=xgboost_class_indices,
        random_forest=random_forest,
        training_split_sha256=training_hash,
        normal_baseline_count=int(normal_mask.sum()),
        normal_baseline_features=(
            _normal_baseline_features(
                [
                    record
                    for record, is_normal in zip(split.train, normal_mask, strict=True)
                    if is_normal
                ],
                training.fit_state.source_features,
            )
            if isolation_training is not None
            else None
        ),
        isolation_preprocessor=(
            isolation_training.fit_state if isolation_training is not None else None
        ),
        isolation_forest=isolation_forest,
    )


def predict_model_outputs(
    bundle: ModelBundle,
    records: Sequence[SessionFeatureRecord],
) -> list[ModelOutputs]:
    matrix = build_feature_matrix(records, bundle.preprocessor)
    xgboost_probabilities = _five_class_probabilities(
        bundle.xgboost,
        matrix.matrix,
        bundle.xgboost_class_indices,
    )
    random_forest_probabilities = _five_class_probabilities(
        bundle.random_forest,
        matrix.matrix,
        tuple(int(index) for index in bundle.random_forest.classes_),
    )
    anomaly_scores = None
    if bundle.isolation_forest is not None and bundle.isolation_preprocessor is not None:
        isolation_matrix = build_feature_matrix(records, bundle.isolation_preprocessor)
        anomaly_scores = bundle.isolation_forest.decision_function(isolation_matrix.matrix)
    return [
        ModelOutputs(
            xgboost=_classifier_output(xgboost_probabilities[index]),
            random_forest=_classifier_output(random_forest_probabilities[index]),
            isolation_forest=(
                IsolationForestOutput(raw_score=float(anomaly_scores[index]))
                if anomaly_scores is not None
                else None
            ),
            diagnostics=matrix.diagnostics,
        )
        for index in range(len(records))
    ]


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_model_bundle(
    bundle: ModelBundle,
    directory: str | Path,
    calibration: CalibrationState | None = None,
) -> Path:
    path = Path(directory)
    if path.exists():
        raise FileExistsError(f"model bundle directory already exists: {path}")
    path.mkdir(parents=True)
    save_preprocessor(bundle.preprocessor, path / "preprocessor.joblib")
    if bundle.isolation_preprocessor is not None and bundle.isolation_forest is not None:
        save_preprocessor(bundle.isolation_preprocessor, path / "isolation_preprocessor.joblib")
        joblib.dump(bundle.isolation_forest, path / "isolation_forest.joblib")
    joblib.dump(bundle.xgboost, path / "xgboost.joblib")
    joblib.dump(bundle.random_forest, path / "random_forest.joblib")
    (path / "feature_map.json").write_text(
        json.dumps(bundle.preprocessor.feature_map, indent=2, sort_keys=True) + "\n"
    )
    if bundle.normal_baseline_features is not None:
        (path / "normal_baseline.json").write_text(
            json.dumps(bundle.normal_baseline_features, indent=2, sort_keys=True) + "\n"
        )
    if calibration is not None:
        from ml.calibration import save_calibration_state

        save_calibration_state(calibration, path)

    manifest = {
        "bundle_id": bundle.bundle_id,
        "bundle_version": bundle.version,
        "model_names": [
            "xgboost",
            "random_forest",
            *(["isolation_forest"] if bundle.isolation_forest is not None else []),
        ],
        "training_split_sha256": bundle.training_split_sha256,
        "normal_baseline_count": bundle.normal_baseline_count,
        "normal_baseline_feature_count": len(bundle.normal_baseline_features or {}),
        "calibration_included": calibration is not None,
        "xgboost_class_indices": list(bundle.xgboost_class_indices),
        "feature_names": list(bundle.preprocessor.feature_names),
        "config": bundle.config.model_dump(mode="json"),
        "dependency_versions": {
            package: version(package)
            for package in ("joblib", "scikit-learn", "xgboost")
        },
    }
    (path / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    files = sorted(file for file in path.iterdir() if file.is_file())
    (path / "checksums.sha256").write_text(
        "".join(
            f"{_file_hash(file)}  {file.name}\n" for file in files
        )
    )
    return path


def load_model_bundle(directory: str | Path) -> ModelBundle:
    path = Path(directory)
    for line in (path / "checksums.sha256").read_text().splitlines():
        digest, filename = line.split("  ", maxsplit=1)
        if _file_hash(path / filename) != digest:
            raise ValueError(f"checksum mismatch: {filename}")
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest["bundle_version"] not in SUPPORTED_MODEL_BUNDLE_VERSIONS:
        raise ValueError("unsupported model bundle version")
    isolation_preprocessor_path = path / "isolation_preprocessor.joblib"
    isolation_forest_path = path / "isolation_forest.joblib"
    has_isolation = isolation_preprocessor_path.is_file() and isolation_forest_path.is_file()
    baseline_path = path / "normal_baseline.json"
    return ModelBundle(
        bundle_id=manifest["bundle_id"],
        version=manifest["bundle_version"],
        config=ModelConfig.model_validate(manifest["config"]),
        preprocessor=load_preprocessor(path / "preprocessor.joblib"),
        xgboost=joblib.load(path / "xgboost.joblib"),
        xgboost_class_indices=tuple(manifest["xgboost_class_indices"]),
        random_forest=joblib.load(path / "random_forest.joblib"),
        training_split_sha256=manifest["training_split_sha256"],
        normal_baseline_count=manifest.get("normal_baseline_count", 0),
        normal_baseline_features=(
            json.loads(baseline_path.read_text()) if baseline_path.is_file() else None
        ),
        isolation_preprocessor=(
            load_preprocessor(isolation_preprocessor_path) if has_isolation else None
        ),
        isolation_forest=joblib.load(isolation_forest_path) if has_isolation else None,
    )


if __name__ == "__main__":
    from tempfile import TemporaryDirectory

    from ml.dataset import generate_feature_dataset, split_dataset

    dataset = generate_feature_dataset(
        {"master_seed": 420042, "session_count": 500}
    )
    split = split_dataset(dataset, {"random_seed": 420042})
    bundle = train_model_bundle(split, {"random_seed": 420042})
    predictions = predict_model_outputs(bundle, split.validation)
    assert len(predictions) == len(split.validation)
    assert all(
        len(prediction.xgboost.class_probabilities) == len(RISK_LABELS)
        and len(prediction.random_forest.class_probabilities) == len(RISK_LABELS)
        for prediction in predictions
    )
    with TemporaryDirectory() as directory:
        saved = save_model_bundle(bundle, Path(directory) / "bundle")
        reloaded = load_model_bundle(saved)
        assert predict_model_outputs(reloaded, split.validation) == predictions
    print(bundle.bundle_id, len(predictions), bundle.normal_baseline_count)

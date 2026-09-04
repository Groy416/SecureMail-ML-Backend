from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from ml.features import (
    BOOLEAN_FEATURES,
    CATEGORICAL_FEATURES,
    FEATURE_DISPLAY_NAMES,
    MODEL_INPUT_FEATURES,
    NUMERIC_FEATURES,
    extract_model_features,
    feature_view_for,
)
from ml.schema import SessionFeatureRecord, validate_session

PREPROCESSOR_VERSION = "preprocessor.v1"
NUMERIC_MODEL_FEATURES = NUMERIC_FEATURES + BOOLEAN_FEATURES


@dataclass
class PreprocessorState:
    transformer: ColumnTransformer
    source_features: tuple[str, ...]
    feature_names: tuple[str, ...]
    feature_map: dict[str, dict[str, str]]
    version: str = PREPROCESSOR_VERSION


@dataclass
class MatrixArtifact:
    matrix: np.ndarray
    feature_names: tuple[str, ...]
    feature_map: dict[str, dict[str, str]]
    fit_state: PreprocessorState
    diagnostics: dict[str, list[str]]


def _records_to_frame(
    records: Sequence[SessionFeatureRecord | Mapping[str, Any]],
) -> pd.DataFrame:
    if not records:
        raise ValueError("at least one session record is required")
    rows = [
        extract_model_features(
            record if isinstance(record, SessionFeatureRecord) else validate_session(record)
        )
        for record in records
    ]
    frame = pd.DataFrame(rows)
    for name in NUMERIC_FEATURES:
        frame[name] = pd.to_numeric(frame[name], errors="raise")
    for name in BOOLEAN_FEATURES:
        frame[name] = frame[name].map({True: 1.0, False: 0.0})
    for name in CATEGORICAL_FEATURES:
        frame[name] = frame[name].map(lambda value: getattr(value, "value", value))
    return frame


def _select_features(
    source_features: Sequence[str] | None,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    selected = tuple(source_features or MODEL_INPUT_FEATURES)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("source features must be non-empty and unique")
    unknown = set(selected) - set(MODEL_INPUT_FEATURES)
    if unknown:
        raise ValueError(f"unknown model features: {sorted(unknown)}")
    numeric = tuple(name for name in NUMERIC_MODEL_FEATURES if name in selected)
    categorical = tuple(name for name in CATEGORICAL_FEATURES if name in selected)
    return selected, numeric, categorical


def _new_transformer(
    numeric_features: tuple[str, ...],
    categorical_features: tuple[str, ...],
) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", MinMaxScaler(feature_range=(0, 1), clip=True)),
                    ]
                ),
                list(numeric_features),
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="constant",
                                fill_value="__MISSING__",
                            ),
                        ),
                        (
                            "encoder",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                sparse_output=False,
                            ),
                        ),
                    ]
                ),
                list(categorical_features),
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def _feature_map(
    transformer: ColumnTransformer,
    numeric_features: tuple[str, ...],
    categorical_features: tuple[str, ...],
) -> dict[str, dict[str, str]]:
    names = list(transformer.get_feature_names_out())
    feature_map: dict[str, dict[str, str]] = {}
    position = 0
    for source_feature in numeric_features:
        transformed_name = names[position]
        feature_map[transformed_name] = {
            "source_feature": source_feature,
            "feature_view": feature_view_for(source_feature).value,
            "display_name": FEATURE_DISPLAY_NAMES[source_feature],
        }
        position += 1
    encoder = transformer.named_transformers_["categorical"].named_steps["encoder"]
    for source_feature, categories in zip(
        categorical_features,
        encoder.categories_,
        strict=True,
    ):
        for _ in categories:
            transformed_name = names[position]
            feature_map[transformed_name] = {
                "source_feature": source_feature,
                "feature_view": feature_view_for(source_feature).value,
                "display_name": FEATURE_DISPLAY_NAMES[source_feature],
            }
            position += 1
    if position != len(names):
        raise RuntimeError("transformed feature map is incomplete")
    return feature_map


def _unknown_categories(
    frame: pd.DataFrame,
    state: PreprocessorState,
) -> dict[str, list[str]]:
    categorical_features = tuple(
        name for name in CATEGORICAL_FEATURES if name in state.source_features
    )
    encoder = state.transformer.named_transformers_["categorical"].named_steps["encoder"]
    diagnostics: dict[str, list[str]] = {}
    for source_feature, categories in zip(
        categorical_features,
        encoder.categories_,
        strict=True,
    ):
        known = {str(category) for category in categories}
        observed = {str(value) for value in frame[source_feature].dropna().unique()}
        unknown = sorted(observed - known)
        if unknown:
            diagnostics[source_feature] = unknown
    return diagnostics


def build_feature_matrix(
    records: Sequence[SessionFeatureRecord | Mapping[str, Any]],
    fit_state: PreprocessorState | None = None,
    source_features: Sequence[str] | None = None,
) -> MatrixArtifact:
    frame = _records_to_frame(records)
    if fit_state is None:
        selected, numeric_features, categorical_features = _select_features(
            source_features
        )
        transformer = _new_transformer(numeric_features, categorical_features)
        matrix = transformer.fit_transform(frame)
        feature_names = tuple(str(name) for name in transformer.get_feature_names_out())
        fit_state = PreprocessorState(
            transformer=transformer,
            source_features=selected,
            feature_names=feature_names,
            feature_map=_feature_map(
                transformer,
                numeric_features,
                categorical_features,
            ),
        )
        diagnostics: dict[str, list[str]] = {}
    else:
        if source_features is not None and tuple(source_features) != fit_state.source_features:
            raise ValueError("source features do not match the fitted preprocessor")
        matrix = fit_state.transformer.transform(frame)
        diagnostics = _unknown_categories(frame, fit_state)
    return MatrixArtifact(
        matrix=np.asarray(matrix, dtype=float),
        feature_names=fit_state.feature_names,
        feature_map=fit_state.feature_map,
        fit_state=fit_state,
        diagnostics=diagnostics,
    )


def save_preprocessor(state: PreprocessorState, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(state, target)


def load_preprocessor(path: str | Path) -> PreprocessorState:
    state = joblib.load(path)
    if not isinstance(state, PreprocessorState):
        raise ValueError("artifact is not a SecureMailScope preprocessor")
    if state.version != PREPROCESSOR_VERSION:
        raise ValueError(f"unsupported preprocessor version: {state.version}")
    return state


if __name__ == "__main__":
    from ml.dataset import generate_feature_dataset, split_dataset

    split = split_dataset(
        generate_feature_dataset({"master_seed": 420042, "session_count": 500}),
        {"random_seed": 420042},
    )
    training = build_feature_matrix(split.train)
    validation = build_feature_matrix(split.validation, training.fit_state)
    protocol_only = build_feature_matrix(
        split.train,
        source_features=("protocol", "src_port", "dst_port"),
    )
    unknown = split.validation[0].model_copy(deep=True)
    unknown.features.tls_version = "TLS9.9"
    unknown_matrix = build_feature_matrix([unknown], training.fit_state)
    assert training.matrix.shape[1] == len(training.feature_names)
    assert validation.matrix.shape[1] == training.matrix.shape[1]
    assert protocol_only.matrix.shape[1] < training.matrix.shape[1]
    assert np.all((0.0 <= training.matrix) & (training.matrix <= 1.0))
    assert unknown_matrix.diagnostics["tls_version"] == ["TLS9.9"]
    with TemporaryDirectory() as directory:
        path = Path(directory) / "preprocessor.joblib"
        save_preprocessor(training.fit_state, path)
        assert np.array_equal(
            validation.matrix,
            build_feature_matrix(split.validation, load_preprocessor(path)).matrix,
        )
    print(training.matrix.shape, validation.matrix.shape, protocol_only.matrix.shape)

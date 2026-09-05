from __future__ import annotations

from typing import Any

import numpy as np
import shap
from pydantic import Field

from ml.fusion import MLResult
from ml.models import ModelBundle
from ml.preprocess import build_feature_matrix
from ml.schema import (
    ContractModel,
    Protocol,
    RiskLabel,
    RISK_SEVERITY_ORDER,
    SessionFeatureRecord,
)

EXPLANATION_VERSION = "explanation.v1"


class ExplanationEntry(ContractModel):
    feature: str
    feature_view: str
    observed_value: object
    contribution: float
    direction: str
    model: str
    evidence_refs: list[str]


class ExplanationSet(ContractModel):
    version: str = EXPLANATION_VERSION
    supervised: list[ExplanationEntry]
    anomaly_contributions: list[ExplanationEntry]
    disclaimer: str = (
        "These explanations describe model behavior and are not proof of attacker intent."
    )


def _evidence_refs(
    record: SessionFeatureRecord,
    source_feature: str,
) -> list[str]:
    return [
        f"{reference.source.value}:{','.join(reference.fields)}"
        for reference in record.provenance.evidence_refs
        if source_feature in reference.fields
    ]


def _direction(contribution: float, predicted_class: RiskLabel) -> str:
    severity = RISK_SEVERITY_ORDER.index(predicted_class)
    if severity < RISK_SEVERITY_ORDER.index(RiskLabel.MEDIUM):
        return (
            "increases_risk" if contribution < 0 else "decreases_risk"
        )
    return "increases_risk" if contribution > 0 else "decreases_risk"


def _class_position(
    bundle: ModelBundle,
    model_name: str,
    predicted_class: RiskLabel,
) -> int:
    class_index = RISK_SEVERITY_ORDER.index(predicted_class)
    if model_name == "xgboost":
        return bundle.xgboost_class_indices.index(class_index)
    return list(bundle.random_forest.classes_).index(class_index)


def _tree_entries(
    bundle: ModelBundle,
    record: SessionFeatureRecord,
    result: MLResult,
    model_name: str,
    top_k: int,
) -> list[ExplanationEntry]:
    matrix = build_feature_matrix([record], bundle.preprocessor)
    model = getattr(bundle, model_name)
    model_output = result.model_outputs[model_name]
    predicted_class = RiskLabel(model_output["predicted_class"])
    class_position = _class_position(bundle, model_name, predicted_class)
    shap_values = np.asarray(shap.TreeExplainer(model)(matrix.matrix).values)
    if shap_values.ndim != 3 or shap_values.shape[0] != 1:
        raise ValueError("unexpected TreeSHAP output shape")
    contributions = shap_values[0, :, class_position]

    aggregated: dict[str, float] = {}
    for transformed_name, contribution in zip(
        matrix.feature_names,
        contributions,
        strict=True,
    ):
        source = matrix.feature_map[transformed_name]["source_feature"]
        aggregated[source] = aggregated.get(source, 0.0) + float(contribution)

    observed = record.features.model_dump(mode="json")
    ranked = sorted(
        aggregated.items(),
        key=lambda item: abs(item[1]),
        reverse=True,
    )[:top_k]
    return [
        ExplanationEntry(
            feature=source,
            feature_view=bundle.preprocessor.feature_map[
                next(
                    name
                    for name, metadata in bundle.preprocessor.feature_map.items()
                    if metadata["source_feature"] == source
                )
            ]["feature_view"],
            observed_value=observed[source],
            contribution=contribution,
            direction=_direction(contribution, predicted_class),
            model=model_name,
            evidence_refs=_evidence_refs(record, source),
        )
        for source, contribution in ranked
    ]


def _anomaly_entries(
    bundle: ModelBundle,
    record: SessionFeatureRecord,
    result: MLResult,
    top_k: int,
) -> list[ExplanationEntry]:
    if (
        bundle.isolation_preprocessor is None
        or bundle.isolation_forest is None
        or bundle.normal_baseline_features is None
        or result.anomaly.status != "calibrated"
    ):
        return []
    observed = record.features.model_dump(mode="json")
    original_raw_score = result.anomaly.raw_score
    contributions: list[ExplanationEntry] = []
    for source_feature in bundle.isolation_preprocessor.source_features:
        baseline_value = bundle.normal_baseline_features[source_feature]
        if source_feature == "protocol" and baseline_value is not None:
            baseline_value = Protocol(baseline_value)
        perturbed_record = record.model_copy(deep=True)
        setattr(perturbed_record.features, source_feature, baseline_value)
        perturbed_matrix = build_feature_matrix(
            [perturbed_record],
            bundle.isolation_preprocessor,
        )
        perturbed_raw_score = float(
            bundle.isolation_forest.decision_function(perturbed_matrix.matrix)[0]
        )
        contribution = perturbed_raw_score - original_raw_score
        metadata = next(
            metadata
            for metadata in bundle.isolation_preprocessor.feature_map.values()
            if metadata["source_feature"] == source_feature
        )
        contributions.append(
            ExplanationEntry(
                feature=source_feature,
                feature_view=metadata["feature_view"],
                observed_value=observed[source_feature],
                contribution=contribution,
                direction=(
                    "increases_risk"
                    if contribution > 0
                    else "decreases_risk"
                ),
                model="isolation_forest_baseline_perturbation",
                evidence_refs=_evidence_refs(record, source_feature),
            )
        )
    return sorted(
        contributions,
        key=lambda entry: abs(entry.contribution),
        reverse=True,
    )[:top_k]


def explain_session(
    bundle: ModelBundle,
    record: SessionFeatureRecord,
    result: MLResult,
    top_k: int = 8,
) -> ExplanationSet:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    return ExplanationSet(
        supervised=(
            _tree_entries(bundle, record, result, "xgboost", top_k)
            + _tree_entries(bundle, record, result, "random_forest", top_k)
        ),
        anomaly_contributions=_anomaly_entries(bundle, record, result, top_k),
    )


if __name__ == "__main__":
    from ml.calibration import calibrate_scores
    from ml.dataset import generate_feature_dataset, split_dataset
    from ml.fusion import predict_session
    from ml.models import predict_model_outputs, train_model_bundle

    split = split_dataset(
        generate_feature_dataset({"master_seed": 420042, "session_count": 500}),
        {"random_seed": 420042},
    )
    bundle = train_model_bundle(split, {"random_seed": 420042})
    calibration = calibrate_scores(
        predict_model_outputs(bundle, split.validation),
        split.validation,
    )
    result = predict_session(bundle, calibration, split.test[0])
    explanations = explain_session(bundle, split.test[0], result)
    assert len(explanations.supervised) == 16
    assert not explanations.anomaly_contributions
    print(explanations.model_dump(mode="json"))

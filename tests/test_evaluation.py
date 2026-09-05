from __future__ import annotations

from ml.calibration import calibrate_scores
from ml.dataset import generate_feature_dataset, split_dataset
from ml.evaluate import evaluate_bundle
from ml.fusion import FusionConfig
from ml.models import predict_model_outputs, train_model_bundle


def test_evaluation_separates_model_fusion_from_rule_policy() -> None:
    split = split_dataset(
        generate_feature_dataset({"master_seed": 420042, "session_count": 500}),
        {"random_seed": 420042},
    )
    bundle = train_model_bundle(
        split,
        {
            "random_seed": 420042,
            "xgboost_n_estimators": 2,
            "random_forest_n_estimators": 2,
        },
    )
    calibration = calibrate_scores(
        predict_model_outputs(bundle, split.validation),
        split.validation,
    )

    report = evaluate_bundle(
        bundle,
        calibration,
        split,
        fusion_config=FusionConfig(),
    )

    assert report.model_fusion["confusion_matrix"]
    assert report.fusion["confusion_matrix"]
    assert report.scenario_breakdown
    assert report.isolation_forest["status"] == "disabled:isolation_forest_removed"
    assert all(
        values["isolation_forest"]["status"] == "disabled:isolation_forest_removed"
        for values in report.scenario_breakdown.values()
    )

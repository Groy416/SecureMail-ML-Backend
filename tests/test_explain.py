from __future__ import annotations

from ml.calibration import calibrate_scores
from ml.dataset import generate_feature_dataset, split_dataset
from ml.explain import explain_session
from ml.fusion import fuse_session
from ml.models import predict_model_outputs, train_model_bundle


def test_two_model_explanations_have_no_isolation_forest_entries() -> None:
    split = split_dataset(
        generate_feature_dataset(
            {"master_seed": 420042, "session_count": 500}
        ),
        {"random_seed": 420042},
    )
    bundle = train_model_bundle(
        split,
        {
            "random_seed": 420042,
            "enable_isolation_forest": False,
            "xgboost_n_estimators": 2,
            "random_forest_n_estimators": 2,
        },
    )
    outputs = predict_model_outputs(bundle, split.test[:1])
    calibration = calibrate_scores(
        predict_model_outputs(bundle, split.validation),
        split.validation,
    )
    result = fuse_session(split.test[0], outputs[0], calibration)

    explanations = explain_session(bundle, split.test[0], result)

    assert explanations.anomaly_contributions == []
    assert explanations.supervised

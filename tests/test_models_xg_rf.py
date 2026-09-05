from __future__ import annotations

import json
from pathlib import Path

from ml.calibration import calibrate_scores
from ml.dataset import generate_feature_dataset, split_dataset
from ml.models import (
    load_model_bundle,
    predict_model_outputs,
    save_model_bundle,
    train_model_bundle,
)


def _two_model_split():
    dataset = generate_feature_dataset(
        {"master_seed": 420042, "session_count": 500}
    )
    return split_dataset(dataset, {"random_seed": 420042})


def test_two_model_bundle_does_not_fit_or_persist_isolation_forest(
    tmp_path: Path,
) -> None:
    split = _two_model_split()
    bundle = train_model_bundle(
        split,
        {
            "random_seed": 420042,
            "enable_isolation_forest": False,
            "xgboost_n_estimators": 2,
            "random_forest_n_estimators": 2,
        },
    )

    assert bundle.isolation_forest is None
    calibration = calibrate_scores(
        predict_model_outputs(bundle, split.validation),
        split.validation,
    )
    before_reload = predict_model_outputs(bundle, split.test[:3])
    path = save_model_bundle(bundle, tmp_path / "Model_XG_RF", calibration)
    manifest = json.loads((path / "manifest.json").read_text())

    assert manifest["model_names"] == ["xgboost", "random_forest"]
    assert not (path / "isolation_forest.joblib").exists()
    assert not (path / "isolation_preprocessor.joblib").exists()
    assert not (path / "normal_baseline.json").exists()
    assert not (path / "thresholds.json").exists()
    reloaded = load_model_bundle(path)
    after_reload = predict_model_outputs(reloaded, split.test[:3])
    assert reloaded.isolation_forest is None
    assert [item.model_dump() for item in before_reload] == [
        item.model_dump() for item in after_reload
    ]

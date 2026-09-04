from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from ml.calibration import calibrate_scores
from ml.dataset import SplitArtifact
from ml.evaluate import evaluate_bundle
from ml.features import FEATURE_VIEWS
from ml.models import ModelConfig, predict_model_outputs, train_model_bundle
from ml.schema import FeatureView

ABLATION_VERSION = "feature-ablation.v1"


@dataclass(frozen=True)
class AblationReport:
    version: str
    views: dict[str, dict[str, Any]]
    limitations: tuple[str, ...]


def evaluate_feature_view_ablations(
    split: SplitArtifact,
    config: ModelConfig | Mapping[str, Any],
) -> AblationReport:
    base = config if isinstance(config, ModelConfig) else ModelConfig.model_validate(config)
    views: dict[str, dict[str, Any]] = {}
    for view, source_features in FEATURE_VIEWS.items():
        view_config = ModelConfig.model_validate(
            {
                **base.model_dump(mode="json"),
                "source_features": source_features,
            }
        )
        bundle = train_model_bundle(split, view_config)
        calibration = calibrate_scores(
            predict_model_outputs(bundle, split.validation),
            split.validation,
        )
        evaluation = evaluate_bundle(bundle, calibration, split)
        views[view.value] = {
            "bundle_id": bundle.bundle_id,
            "source_features": list(source_features),
            "xgboost": {
                "macro_f1": evaluation.classifiers["xgboost"]["macro_f1"],
                "critical_recall": evaluation.classifiers["xgboost"]["critical_recall"],
            },
            "random_forest": {
                "macro_f1": evaluation.classifiers["random_forest"]["macro_f1"],
                "critical_recall": evaluation.classifiers["random_forest"]["critical_recall"],
            },
            "fusion": {
                "macro_f1": evaluation.fusion["macro_f1"],
                "critical_recall": evaluation.fusion["critical_recall"],
            },
            "isolation_forest": {
                "f1": evaluation.isolation_forest["f1"],
                "false_positive_rate": evaluation.isolation_forest["false_positive_rate"],
            },
        }
    return AblationReport(
        version=ABLATION_VERSION,
        views=views,
        limitations=(
            "Ablations use only synthetic data and are not production-performance claims.",
            "Learned stacking is disabled until grouped out-of-fold normal-baseline coverage exists.",
        ),
    )


def save_ablation_report(
    report: AblationReport,
    root: str | Path = "evals",
) -> Path:
    payload = asdict(report)
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    path = Path(root) / f"ablation-{digest[:12]}"
    target = path / "metrics.json"
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if target.exists():
        if target.read_text() == encoded:
            return path
        raise FileExistsError(f"ablation directory conflicts: {path}")
    if path.exists():
        raise FileExistsError(f"ablation directory already exists: {path}")
    path.mkdir(parents=True)
    target.write_text(encoded)
    return path


if __name__ == "__main__":
    from tempfile import TemporaryDirectory

    from ml.dataset import generate_feature_dataset, split_dataset

    split = split_dataset(
        generate_feature_dataset({"master_seed": 420042, "session_count": 500}),
        {"random_seed": 420042},
    )
    report = evaluate_feature_view_ablations(split, {"random_seed": 420042})
    assert set(report.views) == {view.value for view in FeatureView}
    with TemporaryDirectory() as directory:
        assert (save_ablation_report(report, directory) / "metrics.json").is_file()
    print({name: values["fusion"] for name, values in report.views.items()})

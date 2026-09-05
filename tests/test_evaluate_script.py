from __future__ import annotations

from ml.evaluate import EvaluationReport, format_evaluation_table
from ml.fusion import FusionConfig
from scripts.evaluate_pcap_bundle import build_parser


def test_pcap_evaluation_defaults_to_named_artifacts() -> None:
    args = build_parser().parse_args([])

    assert args.dataset_run.name == "Dataset-17K"
    assert args.bundle.name == "Model_XG_RF"


def test_evaluation_table_is_fixed_width_and_names_the_two_model_fusion() -> None:
    metrics = {
        "accuracy": 0.98,
        "macro_f1": 0.97,
        "weighted_f1": 0.98,
        "critical_precision": 1.0,
        "critical_recall": 0.8333,
    }
    report = EvaluationReport(
        schema_version="evaluation.v1",
        model_bundle_version="ml-bundle.v2",
        model_bundle_id="bundle",
        calibration_version="calibration.v2",
        test_split_sha256="split",
        test_session_count=1,
        test_environment_ids=("lab_test",),
        classifiers={"xgboost": metrics, "random_forest": metrics},
        isolation_forest={"status": "disabled:isolation_forest_removed"},
        model_fusion=metrics,
        fusion=metrics,
        scenario_breakdown={},
        limitations=(),
    )

    table = format_evaluation_table(report, FusionConfig())

    assert "Fusion = 0.60 × XGBoost + 0.40 × Random Forest" in table
    assert all(label in table for label in ("XGBoost", "Random Forest", "Model fusion", "Rule policy"))
    assert "Accuracy" in table and "Critical recall" in table


def test_pcap_evaluation_requires_a_persisted_dataset_run() -> None:
    args = build_parser().parse_args(
        [
            "--dataset-run",
            "datasets/runs/synthetic-pcap-example",
            "--bundle",
            "models/grouped-105-capture",
        ]
    )

    assert args.dataset_run.name == "synthetic-pcap-example"
    assert args.bundle.name == "grouped-105-capture"

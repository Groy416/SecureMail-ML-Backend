from __future__ import annotations

from collections.abc import Sequence

from ml.calibration import CalibrationState
from ml.explain import explain_session
from ml.fusion import MLResult, RuleFinding, predict_session as fuse_session
from ml.models import ModelBundle
from ml.rules import extract_rule_findings
from ml.schema import SessionFeatureRecord


def predict_session(
    bundle: ModelBundle,
    calibration: CalibrationState,
    record: SessionFeatureRecord,
    findings: Sequence[RuleFinding] = (),
) -> MLResult:
    merged_findings = {
        finding.finding_id: finding
        for finding in (*extract_rule_findings(record), *findings)
    }
    result = fuse_session(
        bundle,
        calibration,
        record,
        tuple(merged_findings.values()),
    )
    try:
        explanations = explain_session(bundle, record, result)
    except Exception as exc:
        diagnostics = {
            **result.diagnostics,
            "explanations": [f"unavailable:{type(exc).__name__}"],
        }
        return result.model_copy(update={"diagnostics": diagnostics})
    return result.model_copy(
        update={"explanations": explanations.model_dump(mode="json")}
    )


if __name__ == "__main__":
    from ml.calibration import calibrate_scores
    from ml.dataset import generate_feature_dataset, split_dataset
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
    assert result.explanations["version"] == "explanation.v1"
    assert not result.diagnostics.get("explanations")
    print(result.session_id, len(result.explanations["supervised"]))

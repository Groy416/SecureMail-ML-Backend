from __future__ import annotations

from typing import Literal, Sequence

from pydantic import ConfigDict, Field

from ml.calibration import (
    CalibrationState,
    calibrated_class_probabilities,
    normalized_anomaly_score,
    risk_probability,
)
from ml.models import (
    MODEL_BUNDLE_VERSION,
    ModelBundle,
    ModelOutputs,
    predict_model_outputs,
)
from ml.schema import (
    ContractModel,
    ModelAction,
    RiskLabel,
    RISK_SEVERITY_ORDER,
    SessionFeatureRecord,
)


class FusionConfig(ContractModel):
    xgboost_weight: float = Field(default=0.60, ge=0, le=1)
    random_forest_weight: float = Field(default=0.40, ge=0, le=1)
    isolation_forest_weight: float = Field(default=0.0, ge=0, le=1)

    def __init__(self, **data: object) -> None:
        super().__init__(**data)
        if abs(
            self.xgboost_weight
            + self.random_forest_weight
            + self.isolation_forest_weight
            - 1.0
        ) > 1e-9:
            raise ValueError("fusion weights must sum to 1.0")


class RuleFinding(ContractModel):
    finding_id: str = Field(min_length=1)
    severity: RiskLabel
    title: str = Field(min_length=1)
    evidence_refs: list[str] = Field(min_length=1)


class SupervisedRiskOutput(ContractModel):
    predicted_class: RiskLabel
    risk_probability: float = Field(ge=0, le=1)
    class_probabilities: dict[RiskLabel, float]


class AnomalyResult(ContractModel):
    status: Literal["calibrated", "disabled"] = "calibrated"
    detected: bool
    score: float = Field(ge=0, le=1)
    raw_score: float | None = None
    threshold: float | None = Field(default=None, ge=0, le=1)
    baseline_id: str | None = None


class RiskResult(ContractModel):
    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)

    risk_class: RiskLabel = Field(alias="class", serialization_alias="class")
    score: float = Field(ge=0, le=1)
    source: str
    minimum_rule_severity: RiskLabel | None = None


class MLResult(ContractModel):
    model_config = ConfigDict(serialize_by_alias=True)

    schema_version: str = "ml-result.v1"
    capture_id: str
    session_id: str
    model_bundle_version: str
    risk: RiskResult
    anomaly: AnomalyResult
    model_outputs: dict[str, object]
    rule_findings: list[RuleFinding]
    explanations: dict[str, object] = Field(default_factory=dict)
    action: ModelAction
    evidence_refs: list[str]
    diagnostics: dict[str, list[str]] = Field(default_factory=dict)


def risk_for_score(score: float) -> RiskLabel:
    if score < 0.125:
        return RiskLabel.INFORMATIONAL
    if score < 0.375:
        return RiskLabel.LOW
    if score < 0.625:
        return RiskLabel.MEDIUM
    if score < 0.875:
        return RiskLabel.HIGH
    return RiskLabel.CRITICAL


def highest_rule_severity(
    findings: Sequence[RuleFinding],
) -> RiskLabel | None:
    if not findings:
        return None
    return max(
        (finding.severity for finding in findings),
        key=RISK_SEVERITY_ORDER.index,
    )


def _action(
    risk: RiskLabel,
    anomaly_detected: bool,
    minimum_rule_severity: RiskLabel | None,
) -> ModelAction:
    if minimum_rule_severity is RiskLabel.CRITICAL:
        return ModelAction.CRITICAL_REVIEW
    if anomaly_detected:
        return ModelAction.ANALYST_REVIEW
    if risk in {RiskLabel.INFORMATIONAL}:
        return ModelAction.NO_ACTION
    if risk in {RiskLabel.LOW, RiskLabel.MEDIUM}:
        return ModelAction.MONITOR
    return ModelAction.PRIORITIZE


def _weighted_class_probabilities(
    xgboost: dict[RiskLabel, float],
    random_forest: dict[RiskLabel, float],
    config: FusionConfig,
) -> dict[RiskLabel, float]:
    active_weight = config.xgboost_weight + config.random_forest_weight
    if active_weight <= 0:
        raise ValueError("XGBoost and Random Forest weights must be positive")
    return {
        label: (
            config.xgboost_weight * xgboost[label]
            + config.random_forest_weight * random_forest[label]
        )
        / active_weight
        for label in RISK_SEVERITY_ORDER
    }


def fused_class_probabilities(
    output: ModelOutputs,
    calibration: CalibrationState,
    config: FusionConfig | None = None,
) -> dict[RiskLabel, float]:
    fusion = config or FusionConfig()
    xgboost, random_forest = calibrated_class_probabilities(output, calibration)
    return _weighted_class_probabilities(xgboost, random_forest, fusion)


def _session_evidence(record: SessionFeatureRecord) -> list[str]:
    return [
        f"{reference.source.value}:{','.join(reference.fields)}"
        for reference in record.provenance.evidence_refs
    ]


def fuse_session(
    record: SessionFeatureRecord,
    output: ModelOutputs,
    calibration: CalibrationState,
    findings: Sequence[RuleFinding] = (),
    config: FusionConfig | None = None,
    model_bundle_version: str = MODEL_BUNDLE_VERSION,
) -> MLResult:
    fusion = config or FusionConfig()
    xgboost_probabilities, random_forest_probabilities = (
        calibrated_class_probabilities(output, calibration)
    )
    fused_probabilities = _weighted_class_probabilities(
        xgboost_probabilities,
        random_forest_probabilities,
        fusion,
    )
    ensemble_score = risk_probability(fused_probabilities)
    model_risk = max(
        fused_probabilities,
        key=fused_probabilities.get,
    )
    if output.isolation_forest is not None and calibration.anomaly_enabled:
        anomaly_score = normalized_anomaly_score(
            output.isolation_forest.raw_score,
            calibration,
        )
        anomaly_detected = (
            calibration.anomaly_threshold is not None
            and anomaly_score >= calibration.anomaly_threshold
        )
        anomaly = AnomalyResult(
            status="calibrated",
            detected=anomaly_detected,
            score=anomaly_score,
            raw_score=output.isolation_forest.raw_score,
            threshold=calibration.anomaly_threshold,
            baseline_id=calibration.baseline_id,
        )
    else:
        anomaly_score = 0.0
        anomaly_detected = False
        anomaly = AnomalyResult(
            status="disabled",
            detected=False,
            score=0.0,
            baseline_id=None,
        )
    minimum_rule_severity = highest_rule_severity(findings)
    final_risk = max(
        (model_risk, minimum_rule_severity)
        if minimum_rule_severity is not None
        else (model_risk,),
        key=RISK_SEVERITY_ORDER.index,
    )
    if minimum_rule_severity is None:
        source = "model_ensemble_advisory"
    elif final_risk is minimum_rule_severity:
        source = "deterministic_rule"
    else:
        source = "deterministic_rule_plus_ensemble"

    evidence_refs = _session_evidence(record)
    evidence_refs.extend(
        reference
        for finding in findings
        for reference in finding.evidence_refs
    )
    isolation_notes: list[str] = []
    if anomaly.status == "disabled":
        isolation_notes.append(
            "disabled:isolation_forest_removed"
            if output.isolation_forest is None
            else "unavailable:degenerate_normal_calibration"
        )
    return MLResult(
        capture_id=record.provenance.capture_id,
        session_id=record.provenance.session_id,
        model_bundle_version=model_bundle_version,
        risk=RiskResult(
            **{
                "class": final_risk,
                "score": ensemble_score,
                "source": source,
                "minimum_rule_severity": minimum_rule_severity,
            }
        ),
        anomaly=anomaly,
        model_outputs={
            "xgboost": SupervisedRiskOutput(
                predicted_class=max(
                    xgboost_probabilities,
                    key=xgboost_probabilities.get,
                ),
                risk_probability=risk_probability(xgboost_probabilities),
                class_probabilities=xgboost_probabilities,
            ).model_dump(mode="json"),
            "random_forest": SupervisedRiskOutput(
                predicted_class=max(
                    random_forest_probabilities,
                    key=random_forest_probabilities.get,
                ),
                risk_probability=risk_probability(random_forest_probabilities),
                class_probabilities=random_forest_probabilities,
            ).model_dump(mode="json"),
            **(
                {
                    "isolation_forest": {
                        "anomaly_score": anomaly_score,
                        "flagged": anomaly_detected,
                    }
                }
                if anomaly.status == "calibrated"
                else {}
            ),
        },
        rule_findings=list(findings),
        action=_action(
            final_risk,
            anomaly_detected,
            minimum_rule_severity,
        ),
        evidence_refs=list(dict.fromkeys(evidence_refs)),
        diagnostics={
            **output.diagnostics,
            **({"isolation_forest": isolation_notes} if isolation_notes else {}),
        },
    )


def predict_session(
    bundle: ModelBundle,
    calibration: CalibrationState,
    record: SessionFeatureRecord,
    findings: Sequence[RuleFinding] = (),
) -> MLResult:
    output = predict_model_outputs(bundle, [record])[0]
    return fuse_session(
        record,
        output,
        calibration,
        findings,
        model_bundle_version=bundle.version,
    )


if __name__ == "__main__":
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
    result = predict_session(
        bundle,
        calibration,
        split.test[0],
        [
            RuleFinding(
                finding_id="TLS-001",
                severity=RiskLabel.CRITICAL,
                title="Deprecated TLS version",
                evidence_refs=["scenario:tls_version"],
            )
        ],
    )
    assert result.risk.risk_class is RiskLabel.CRITICAL
    assert result.action is ModelAction.CRITICAL_REVIEW
    print(result.model_dump(mode="json"))

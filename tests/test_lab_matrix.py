from __future__ import annotations

from collections import Counter, defaultdict

from datasets.lab.matrix import (
    CAL_SESSIONS,
    ENVIRONMENTS,
    EVAL_SESSIONS,
    MATRIX_CAPTURES,
    MATRIX_SESSIONS,
    MATRIX_SLOTS,
    SESSIONS_PER_CAPTURE,
    TRAIN_SESSIONS,
    build_training_matrix,
    generate_grouped_feature_dataset,
)
from ml.dataset import split_dataset
from ml.fusion import highest_rule_severity
from ml.rules import extract_rule_findings
from ml.schema import RiskLabel


def test_training_matrix_keeps_train_and_eval_as_separate_sets() -> None:
    profiles = build_training_matrix(420042)

    assert len(profiles) == MATRIX_CAPTURES
    assert len({profile.profile_sha256 for profile in profiles}) == MATRIX_CAPTURES
    assert {profile.environment_id for profile in profiles} == set(ENVIRONMENTS)
    counts = Counter(profile.environment_id for profile in profiles)
    assert counts["lab_train"] * SESSIONS_PER_CAPTURE == TRAIN_SESSIONS
    assert counts["lab_test"] * SESSIONS_PER_CAPTURE == EVAL_SESSIONS
    assert counts["lab_calibration"] * SESSIONS_PER_CAPTURE == CAL_SESSIONS


def test_grouped_dataset_trains_on_five_thousand_and_evals_on_another_set() -> None:
    dataset = generate_grouped_feature_dataset(420042)
    split = split_dataset(
        dataset,
        {
            "random_seed": 420042,
            "validation_environment_id": "lab_calibration",
            "test_environment_id": "lab_test",
        },
    )

    assert len(dataset.records) == MATRIX_SESSIONS
    assert len(split.train) == TRAIN_SESSIONS == 5037
    assert len(split.test) == EVAL_SESSIONS == 5037
    assert len(split.validation) == CAL_SESSIONS
    assert not {
        record.provenance.capture_id for record in split.train
    } & {record.provenance.capture_id for record in split.test}
    captures = defaultdict(set)
    for record in dataset.records:
        captures[record.provenance.capture_id].add(record.provenance.session_id)
    assert all(len(sessions) == SESSIONS_PER_CAPTURE for sessions in captures.values())
    for partition in (split.train, split.validation, split.test):
        assert {record.labels.risk_label for record in partition} == set(RiskLabel)
        normals = {
            record.provenance.capture_id
            for record in partition
            if record.labels.risk_label is RiskLabel.INFORMATIONAL
        }
        assert len(normals) >= 8


def test_grouped_labels_match_rule_severity() -> None:
    dataset = generate_grouped_feature_dataset(420042)
    seen_bases: set[str] = set()
    for record in dataset.records:
        matched = next(
            slot
            for slot in sorted(MATRIX_SLOTS, key=len, reverse=True)
            if record.provenance.scenario_id.startswith(f"{slot}-")
        )
        if matched in seen_bases:
            continue
        seen_bases.add(matched)
        findings = extract_rule_findings(record)
        maximum = highest_rule_severity(findings)
        if record.labels.risk_label is RiskLabel.INFORMATIONAL:
            assert maximum is None
        elif record.labels.risk_label is RiskLabel.LOW:
            assert maximum is None
        else:
            assert maximum is record.labels.risk_label
    assert seen_bases == set(MATRIX_SLOTS)

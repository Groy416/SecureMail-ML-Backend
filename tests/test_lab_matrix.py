from __future__ import annotations

from collections import Counter, defaultdict

from datasets.lab.matrix import (
    CAPTURES_PER_ENVIRONMENT,
    ENVIRONMENTS,
    MATRIX_CAPTURES,
    MATRIX_SESSIONS,
    MATRIX_SLOTS,
    SESSIONS_PER_CAPTURE,
    build_training_matrix,
    generate_grouped_feature_dataset,
)
from ml.dataset import split_dataset
from ml.fusion import highest_rule_severity
from ml.rules import extract_rule_findings
from ml.schema import RiskLabel


def test_training_matrix_has_distinct_grouped_captures() -> None:
    profiles = build_training_matrix(420042)

    assert len(profiles) == MATRIX_CAPTURES == 69
    assert len({profile.profile_sha256 for profile in profiles}) == MATRIX_CAPTURES
    assert len({profile.scenario.scenario_id for profile in profiles}) == MATRIX_CAPTURES
    assert {profile.environment_id for profile in profiles} == set(ENVIRONMENTS)
    assert all(profile.connection_count == SESSIONS_PER_CAPTURE for profile in profiles)
    by_environment = Counter(profile.environment_id for profile in profiles)
    assert by_environment == {environment: CAPTURES_PER_ENVIRONMENT for environment in ENVIRONMENTS}


def test_grouped_dataset_overlaps_label_families_with_distinct_captures() -> None:
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
    assert len({record.provenance.capture_id for record in dataset.records}) == MATRIX_CAPTURES
    captures = defaultdict(set)
    for record in dataset.records:
        captures[record.provenance.capture_id].add(record.provenance.session_id)
    assert all(len(sessions) == SESSIONS_PER_CAPTURE for sessions in captures.values())

    for partition in (split.train, split.validation, split.test):
        labels = {record.labels.risk_label for record in partition}
        assert labels == set(RiskLabel)
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
        base_id = record.provenance.scenario_id.rsplit("-", 2)[0]
        if base_id in seen_bases:
            continue
        seen_bases.add(base_id)
        findings = extract_rule_findings(record)
        maximum = highest_rule_severity(findings)
        if record.labels.risk_label is RiskLabel.INFORMATIONAL:
            assert maximum is None
        elif record.labels.risk_label is RiskLabel.LOW:
            assert maximum is None
        else:
            assert maximum is record.labels.risk_label
    assert seen_bases == set(MATRIX_SLOTS)

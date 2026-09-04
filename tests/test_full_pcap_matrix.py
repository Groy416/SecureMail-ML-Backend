from __future__ import annotations

import json

import pytest

from datasets.lab.matrix import generate_grouped_feature_dataset
from datasets.lab.runner import (
    finalize_run,
    run_training_matrix,
    validate_training_matrix,
)
from datasets.lab.scenarios import resolve_runtime_profile
from ml.dataset import CATALOG
from ml.schema import Protocol, ScenarioManifest


def test_training_matrix_rejects_an_incomplete_profile_list(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("datasets.lab.runner.build_training_matrix", lambda seed: ())

    with pytest.raises(ValueError, match="245 profiles"):
        run_training_matrix(420042, tmp_path)


def test_validation_rejects_an_incomplete_success_set() -> None:
    dataset = generate_grouped_feature_dataset(420042)

    with pytest.raises(ValueError, match="245 profiles"):
        validate_training_matrix((), dataset)


def test_run_manifest_contains_packet_provenance(tmp_path) -> None:
    manifest = ScenarioManifest.model_validate(CATALOG[0]["manifest"])
    profile = resolve_runtime_profile(manifest, Protocol.SMTP, "lab_train", 420042, 0)
    (tmp_path / "runtime_profile.json").write_text(
        profile.model_dump_json(indent=2) + "\n"
    )

    finalize_run(tmp_path, "unsupported_in_lab", "cipher rejected")
    saved = json.loads((tmp_path / "run_manifest.json").read_text())

    assert saved["profile_sha256"] == profile.profile_sha256
    assert saved["scenario_manifest_sha256"]
    assert saved["extractor_version"]
    assert "dependency_versions" in saved
    assert "container_image_ids" in saved
    assert "source_revision" in saved

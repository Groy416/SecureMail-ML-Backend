from __future__ import annotations

from datasets.lab.holdout import HOLDOUT_FAMILIES, generate_heldout_family_records
from datasets.lab.matrix import MATRIX_SLOTS


def test_heldout_families_are_absent_from_the_training_matrix() -> None:
    assert not set(HOLDOUT_FAMILIES) & set(MATRIX_SLOTS)
    records = generate_heldout_family_records()
    assert records
    assert all(record.provenance.environment_id == "lab_test" for record in records)
    bases = set()
    for record in records:
        matched = next(
            family
            for family in sorted(HOLDOUT_FAMILIES, key=len, reverse=True)
            if record.provenance.scenario_id.startswith(f"{family}-")
        )
        bases.add(matched)
    assert bases == set(HOLDOUT_FAMILIES)

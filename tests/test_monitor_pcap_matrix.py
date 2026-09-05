from __future__ import annotations

import json
from pathlib import Path

from scripts.monitor_pcap_matrix import read_progress


def _write_profile(path: Path, profile_sha256: str, scenario_id: str) -> None:
    path.mkdir(parents=True)
    (path / "runtime_profile.json").write_text(
        json.dumps(
            {
                "profile_sha256": profile_sha256,
                "scenario": {"scenario_id": scenario_id},
            }
        )
    )


def _write_manifest(path: Path, status: str) -> None:
    (path / "run_manifest.json").write_text(json.dumps({"status": status}))


def test_read_progress_counts_retry_once_and_tracks_incomplete_runs(tmp_path: Path) -> None:
    _write_profile(tmp_path / "profile-a", "a", "normal-a")
    _write_manifest(tmp_path / "profile-a", "success")

    _write_profile(tmp_path / "profile-b", "b", "normal-b")
    _write_manifest(tmp_path / "profile-b", "failed")
    _write_profile(tmp_path / "retry-1" / "profile-b", "b", "normal-b")
    _write_manifest(tmp_path / "retry-1" / "profile-b", "success")

    _write_profile(tmp_path / "profile-c", "c", "normal-c")
    snapshot = read_progress(
        tmp_path,
        expected_profiles=3,
        sessions_per_capture=73,
        bundle=tmp_path / "bundle",
    )

    assert snapshot.profiles_seen == 3
    assert snapshot.completed_profiles == 2
    assert snapshot.status_counts == {
        "success": 2,
        "failed": 0,
        "unsupported_in_lab": 0,
        "incomplete": 1,
        "invalid": 0,
    }
    assert snapshot.successful_sessions == 146
    assert snapshot.expected_sessions == 219
    assert snapshot.bundle_present is False

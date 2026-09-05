"""Show progress for a synthetic packet-backed training matrix."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets.lab.matrix import MATRIX_CAPTURES, SESSIONS_PER_CAPTURE

TERMINAL_STATUSES = ("success", "failed", "unsupported_in_lab")
STATUS_NAMES = (*TERMINAL_STATUSES, "incomplete", "invalid")


@dataclass(frozen=True)
class ProgressSnapshot:
    profiles_seen: int
    completed_profiles: int
    status_counts: dict[str, int]
    successful_sessions: int
    expected_sessions: int
    latest_scenario: str | None
    bundle_present: bool


def _json_object(path: Path) -> dict[str, object] | None:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _run_status(run_path: Path, profile: dict[str, object] | None) -> str:
    if profile is None:
        return "invalid"
    manifest_path = run_path / "run_manifest.json"
    if not manifest_path.is_file():
        return "incomplete"
    manifest = _json_object(manifest_path)
    status = manifest.get("status") if manifest is not None else None
    return status if status in STATUS_NAMES[:-2] else "invalid"


def _profile_key(
    root: Path,
    run_path: Path,
    profile: dict[str, object] | None,
) -> str:
    value = profile.get("profile_sha256") if profile is not None else None
    return value if isinstance(value, str) and value else str(run_path.relative_to(root))


def _retry_rank(root: Path, run_path: Path) -> tuple[int, str]:
    relative = run_path.relative_to(root)
    is_retry = any(part.startswith("retry-") for part in relative.parts[:-1])
    return (int(is_retry), str(relative))


def read_progress(
    root: str | Path,
    *,
    expected_profiles: int = MATRIX_CAPTURES,
    sessions_per_capture: int = SESSIONS_PER_CAPTURE,
    bundle: str | Path = "models/grouped-105-capture-pcap",
) -> ProgressSnapshot:
    root_path = Path(root)
    runs: dict[str, tuple[tuple[int, str], str, str | None, float]] = {}

    if root_path.is_dir():
        for profile_path in root_path.rglob("runtime_profile.json"):
            run_path = profile_path.parent
            profile = _json_object(profile_path)
            key = _profile_key(root_path, run_path, profile)
            rank = _retry_rank(root_path, run_path)
            existing = runs.get(key)
            if existing is not None and rank < existing[0]:
                continue
            scenario = profile.get("scenario") if profile is not None else None
            scenario_id = (
                scenario.get("scenario_id")
                if isinstance(scenario, dict)
                else None
            )
            scenario_id = scenario_id if isinstance(scenario_id, str) else None
            runs[key] = (
                rank,
                _run_status(run_path, profile),
                scenario_id,
                run_path.stat().st_mtime,
            )

    status_counts = Counter(status for _, status, _, _ in runs.values())
    counts = {status: status_counts.get(status, 0) for status in STATUS_NAMES}
    latest = max(runs.values(), key=lambda item: item[3], default=None)
    successes = counts["success"]
    return ProgressSnapshot(
        profiles_seen=len(runs),
        completed_profiles=sum(counts[status] for status in TERMINAL_STATUSES),
        status_counts=counts,
        successful_sessions=successes * sessions_per_capture,
        expected_sessions=expected_profiles * sessions_per_capture,
        latest_scenario=latest[2] if latest is not None else None,
        bundle_present=Path(bundle).is_dir(),
    )


def active_lab_containers() -> tuple[str, ...] | None:
    try:
        completed = subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    return tuple(
        name for name in completed.stdout.splitlines() if name.startswith("sml")
    )


def format_progress(
    snapshot: ProgressSnapshot,
    *,
    expected_profiles: int,
    bundle: str | Path,
    containers: Sequence[str] | None,
) -> str:
    counts = snapshot.status_counts
    matrix_ready = snapshot.completed_profiles == expected_profiles and counts["success"] == expected_profiles
    container_text = "unavailable" if containers is None else str(len(containers))
    lines = [
        f"captures: {snapshot.completed_profiles}/{expected_profiles} completed "
        f"({snapshot.profiles_seen} seen)",
        "status: "
        f"success={counts['success']} "
        f"failed={counts['failed']} "
        f"unsupported={counts['unsupported_in_lab']} "
        f"incomplete={counts['incomplete']} "
        f"invalid={counts['invalid']}",
        f"successful session slots: {snapshot.successful_sessions}/{snapshot.expected_sessions}",
        f"matrix ready: {'yes' if matrix_ready else 'no'}",
        f"model bundle: {'present' if snapshot.bundle_present else 'absent'} ({bundle})",
        f"active lab containers: {container_text}",
    ]
    if snapshot.latest_scenario:
        lines.append(f"latest scenario: {snapshot.latest_scenario}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="datasets/lab/runs")
    parser.add_argument("--bundle", default="models/grouped-105-capture-pcap")
    parser.add_argument("--expected-profiles", type=int, default=MATRIX_CAPTURES)
    parser.add_argument("--sessions-per-capture", type=int, default=SESSIONS_PER_CAPTURE)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=10.0)
    args = parser.parse_args(argv)
    if args.expected_profiles <= 0 or args.sessions_per_capture <= 0:
        parser.error("profile and session counts must be positive")
    if args.interval <= 0:
        parser.error("interval must be positive")

    try:
        while True:
            snapshot = read_progress(
                args.root,
                expected_profiles=args.expected_profiles,
                sessions_per_capture=args.sessions_per_capture,
                bundle=args.bundle,
            )
            if args.watch and sys.stdout.isatty():
                print("\033[2J\033[H", end="")
            print(
                format_progress(
                    snapshot,
                    expected_profiles=args.expected_profiles,
                    bundle=args.bundle,
                    containers=active_lab_containers(),
                )
            )
            if not args.watch:
                return 0
            time.sleep(args.interval)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())

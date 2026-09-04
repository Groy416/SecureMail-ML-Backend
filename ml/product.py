from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from ml.calibration import CalibrationState, load_calibration_state
from ml.fusion import MLResult, predict_session as fuse_predict
from ml.models import ModelBundle, load_model_bundle
from ml.rules import extract_rule_findings
from ml.schema import (
    AnomalyLabel,
    RiskLabel,
    SessionFeatureRecord,
    SessionLabels,
    SourceType,
    validate_session,
)

DEFAULT_BUNDLE = "models/grouped-105-capture"
FORBIDDEN_PAYLOAD_KEYS = frozenset(
    {
        "attachment",
        "attachments",
        "authorization",
        "body",
        "cookie",
        "credentials",
        "email_body",
        "html",
        "keylog",
        "mail_body",
        "passwd",
        "password",
        "private_key",
        "secret",
        "sslkeylogfile",
        "subject",
        "token",
    }
)


class PayloadRejected(ValueError):
    pass


def _walk_forbidden(payload: object, path: str = "") -> str | None:
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            name = str(key).lower()
            here = f"{path}.{name}" if path else name
            if name in FORBIDDEN_PAYLOAD_KEYS:
                return here
            found = _walk_forbidden(value, here)
            if found:
                return found
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            found = _walk_forbidden(value, f"{path}[{index}]")
            if found:
                return found
    return None


def reject_sensitive_payload(payload: Mapping[str, Any]) -> None:
    found = _walk_forbidden(payload)
    if found:
        raise PayloadRejected(
            f"refused field {found}: mail content, credentials, and key material are not allowed"
        )


def inference_record(payload: Mapping[str, Any]) -> SessionFeatureRecord:
    reject_sensitive_payload(payload)
    data = dict(payload)
    data.setdefault("schema_version", "session-features.v1")
    labels = data.get("labels")
    if labels is None:
        data["labels"] = {
            "risk_label": RiskLabel.INFORMATIONAL.value,
            "anomaly_label": AnomalyLabel.NORMAL.value,
            "expected_finding_ids": [],
        }
    record = validate_session(data, strict=True)
    if record.provenance.source_type is SourceType.AUTHORIZED_CAPTURE:
        if record.provenance.generator_seed is not None:
            raise PayloadRejected("authorized captures must not include generator_seed")
    return record


def score_session(
    bundle: ModelBundle,
    calibration: CalibrationState,
    record: SessionFeatureRecord,
) -> MLResult:
    return fuse_predict(
        bundle,
        calibration,
        record,
        extract_rule_findings(record),
    )


def score_payloads(
    bundle: ModelBundle,
    calibration: CalibrationState,
    payloads: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for index, payload in enumerate(payloads):
        try:
            record = inference_record(payload)
            result = score_session(bundle, calibration, record)
            results.append(
                {
                    "ok": True,
                    "index": index,
                    "session_id": result.session_id,
                    "result": result.model_dump(mode="json", by_alias=True),
                }
            )
        except (PayloadRejected, ValueError) as exc:
            results.append({"ok": False, "index": index, "error": str(exc)})
    return results


def load_runtime(bundle_dir: str | Path = DEFAULT_BUNDLE) -> tuple[ModelBundle, CalibrationState]:
    path = Path(bundle_dir)
    return load_model_bundle(path), load_calibration_state(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise PayloadRejected(f"{path}:{line_number}: invalid JSON") from exc
        if not isinstance(payload, dict):
            raise PayloadRejected(f"{path}:{line_number}: each line must be a JSON object")
        rows.append(payload)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Score authorized session feature records. Does not accept PCAPs or mail content.",
    )
    parser.add_argument("--bundle", default=DEFAULT_BUNDLE)
    parser.add_argument("--input", required=True, help="JSONL of session-features.v1 objects")
    parser.add_argument("--output", help="JSONL results path (default stdout)")
    args = parser.parse_args(argv)
    bundle, calibration = load_runtime(args.bundle)
    rows = score_payloads(bundle, calibration, _read_jsonl(Path(args.input)))
    encoded = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    if args.output:
        Path(args.output).write_text(encoded)
    else:
        sys.stdout.write(encoded)
    return 0 if all(row["ok"] for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())

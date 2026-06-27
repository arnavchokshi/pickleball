#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCHEMA = ROOT / "corrections" / "schema.json"
ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
JSON_POINTER_PATTERN = re.compile(r"^(/([^/~]|~[01])*)+$")
TOP_LEVEL_FIELDS = {"schema_version", "manifest_id", "created_at", "description", "corrections"}
CORRECTION_FIELDS = {"id", "target", "operation", "value", "confidence", "reason", "annotator", "created_at"}
TARGET_FIELDS = {"artifact", "clip_id", "frame_index", "t_s", "path"}
VALUE_REQUIRED_OPERATIONS = {"set", "replace", "append"}
OPERATIONS = VALUE_REQUIRED_OPERATIONS | {"delete"}


def validate_manifest(path: str | Path, schema_path: str | Path = DEFAULT_SCHEMA) -> dict[str, Any]:
    manifest_path = Path(path)
    payload = _read_json(manifest_path)
    _read_json(Path(schema_path))

    errors = _schema_errors(payload)
    errors.extend(_semantic_errors(payload))
    if errors:
        raise ValueError("\n".join(errors))

    corrections = payload["corrections"]
    return {
        "schema_version": payload["schema_version"],
        "manifest_id": payload["manifest_id"],
        "path": str(manifest_path),
        "correction_count": len(corrections),
        "correction_ids": [correction["id"] for correction in corrections],
    }


def _schema_errors(payload: Any) -> list[str]:
    if not isinstance(payload, dict):
        return ["manifest must be an object"]

    errors: list[str] = []
    errors.extend(_unknown_fields("", payload, TOP_LEVEL_FIELDS))
    for field in ("schema_version", "manifest_id", "created_at", "corrections"):
        if field not in payload:
            errors.append(f"{field}: required property is missing")

    if payload.get("schema_version") != 1:
        errors.append("schema_version: must equal 1")
    _validate_id(errors, "manifest_id", payload.get("manifest_id"))
    _validate_date_time(errors, "created_at", payload.get("created_at"))
    if "description" in payload and (not isinstance(payload["description"], str) or not payload["description"]):
        errors.append("description: must be a non-empty string")

    corrections = payload.get("corrections")
    if not isinstance(corrections, list) or not corrections:
        errors.append("corrections: must be a non-empty array")
        return errors

    for index, correction in enumerate(corrections):
        errors.extend(_correction_errors(index, correction))
    return errors


def _correction_errors(index: int, correction: Any) -> list[str]:
    prefix = f"corrections/{index}"
    if not isinstance(correction, dict):
        return [f"{prefix}: must be an object"]

    errors: list[str] = []
    errors.extend(_unknown_fields(prefix, correction, CORRECTION_FIELDS))
    for field in ("id", "target", "operation", "reason", "annotator"):
        if field not in correction:
            errors.append(f"{prefix}/{field}: required property is missing")

    _validate_id(errors, f"{prefix}/id", correction.get("id"))
    operation = correction.get("operation")
    if operation not in OPERATIONS:
        errors.append(f"{prefix}/operation: must be one of append, delete, replace, set")
    if operation in VALUE_REQUIRED_OPERATIONS and "value" not in correction:
        errors.append(f"{prefix}/value: required property is missing")
    if operation == "delete" and "value" in correction:
        errors.append(f"{prefix}/value: must not be present for delete operations")

    if "confidence" in correction:
        confidence = correction["confidence"]
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
            errors.append(f"{prefix}/confidence: must be a number between 0 and 1")

    for field in ("reason", "annotator"):
        if field in correction and (not isinstance(correction[field], str) or not correction[field]):
            errors.append(f"{prefix}/{field}: must be a non-empty string")
    _validate_date_time(errors, f"{prefix}/created_at", correction.get("created_at"), required=False)
    errors.extend(_target_errors(index, correction.get("target")))
    return errors


def _target_errors(index: int, target: Any) -> list[str]:
    prefix = f"corrections/{index}/target"
    if not isinstance(target, dict):
        return [f"{prefix}: must be an object"]

    errors: list[str] = []
    errors.extend(_unknown_fields(prefix, target, TARGET_FIELDS))
    for field in ("artifact", "path"):
        if field not in target:
            errors.append(f"{prefix}/{field}: required property is missing")
    if "artifact" in target and (not isinstance(target["artifact"], str) or not target["artifact"]):
        errors.append(f"{prefix}/artifact: must be a non-empty string")
    _validate_id(errors, f"{prefix}/clip_id", target.get("clip_id"), required=False)

    frame_index = target.get("frame_index")
    if "frame_index" in target and (isinstance(frame_index, bool) or not isinstance(frame_index, int) or frame_index < 0):
        errors.append(f"{prefix}/frame_index: must be a non-negative integer")
    t_s = target.get("t_s")
    if "t_s" in target and (isinstance(t_s, bool) or not isinstance(t_s, (int, float)) or t_s < 0):
        errors.append(f"{prefix}/t_s: must be a non-negative number")

    pointer = target.get("path")
    if isinstance(pointer, str):
        if not JSON_POINTER_PATTERN.match(pointer):
            errors.append(f"{prefix}/path: does not match JSON Pointer format")
    elif "path" in target:
        errors.append(f"{prefix}/path: must be a string")
    return errors


def _unknown_fields(prefix: str, payload: dict[str, Any], allowed: set[str]) -> list[str]:
    errors = []
    for field in sorted(set(payload) - allowed):
        path = f"{prefix}/{field}" if prefix else field
        errors.append(f"{path}: additional property is not allowed")
    return errors


def _validate_id(errors: list[str], path: str, value: Any, *, required: bool = True) -> None:
    if value is None and not required:
        return
    if not isinstance(value, str) or not ID_PATTERN.match(value):
        errors.append(f"{path}: must match ^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _validate_date_time(errors: list[str], path: str, value: Any, *, required: bool = True) -> None:
    if value is None and not required:
        return
    if not isinstance(value, str):
        errors.append(f"{path}: must be an RFC 3339 date-time string")
        return
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        errors.append(f"{path}: must be an RFC 3339 date-time string")


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"{path} does not exist") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON: {exc}") from exc


def _semantic_errors(payload: Any) -> list[str]:
    if not isinstance(payload, dict):
        return []

    corrections = payload.get("corrections")
    if not isinstance(corrections, list):
        return []

    errors: list[str] = []
    seen_ids: set[str] = set()
    for index, correction in enumerate(corrections):
        if not isinstance(correction, dict):
            continue

        correction_id = correction.get("id")
        if isinstance(correction_id, str):
            if correction_id in seen_ids:
                errors.append(f"corrections/{index}/id duplicate correction id: {correction_id}")
            seen_ids.add(correction_id)

        target = correction.get("target")
        if not isinstance(target, dict):
            continue

        artifact = target.get("artifact")
        if isinstance(artifact, str) and _is_unsafe_relative_path(artifact):
            errors.append(
                f"corrections/{index}/target/artifact must be relative and stay within the workspace"
            )

    return errors


def _is_unsafe_relative_path(value: str) -> bool:
    path = Path(value)
    return path.is_absolute() or ".." in path.parts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a manual corrections manifest.")
    parser.add_argument("manifest", type=Path, help="Path to a corrections manifest JSON file.")
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA, help="Path to corrections schema.json.")
    args = parser.parse_args(argv)

    try:
        summary = validate_manifest(args.manifest, args.schema)
    except ValueError as exc:
        print("ERROR: corrections manifest failed validation:", file=sys.stderr)
        for line in str(exc).splitlines():
            print(f"- {line}", file=sys.stderr)
        return 1

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

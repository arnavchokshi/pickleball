#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
SHOT_LABELS = (
    "serve",
    "fh_shot",
    "bh_shot",
    "fh_drive",
    "bh_drive",
    "dink",
    "lob",
    "overhead",
    "third_shot_drop",
    "reset_block",
)
SOURCE_TYPES = ("audio_snapped_pose", "manual_review", "synthetic_aug")
SPLITS = ("train", "val", "test")
TOP_LEVEL_FIELDS = {"schema_version", "dataset_id", "description", "created_at", "notes", "entries"}
ENTRY_FIELDS = {
    "id",
    "path",
    "split",
    "shot_label",
    "source_type",
    "fps",
    "contact_time_ms",
    "window_ms",
    "player_id",
    "notes",
}
ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def validate_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    errors: list[str] = []
    payload = _read_json(manifest_path, errors)
    manifest_dir = manifest_path.parent

    entries: list[dict[str, Any]] = []
    if isinstance(payload, dict):
        errors.extend(_top_level_errors(payload))
        raw_entries = payload.get("entries")
        if isinstance(raw_entries, list):
            seen_ids: set[str] = set()
            for index, entry in enumerate(raw_entries):
                if isinstance(entry, dict):
                    entries.append(entry)
                errors.extend(_entry_errors(index, entry, manifest_dir, seen_ids))
    elif payload is not None:
        errors.append("manifest: must be an object")

    coverage_counts = _coverage_counts(entries)
    coverage_gaps = _coverage_gaps(coverage_counts)
    valid = not errors

    return {
        "schema_version": SCHEMA_VERSION,
        "manifest": str(manifest_path),
        "valid": valid,
        "dataset_ready": valid and not coverage_gaps,
        "entry_count": len(entries),
        "coverage_counts": coverage_counts,
        "coverage_gaps": coverage_gaps,
        "errors": errors,
    }


def _read_json(path: Path, errors: list[str]) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        errors.append(f"manifest: does not exist: {path}")
    except json.JSONDecodeError as exc:
        errors.append(f"manifest: invalid JSON: {exc}")
    return None


def _top_level_errors(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    errors.extend(_unknown_fields("", payload, TOP_LEVEL_FIELDS))

    if payload.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version: must equal {SCHEMA_VERSION}")

    if "dataset_id" in payload:
        _validate_id(errors, "dataset_id", payload["dataset_id"])
    for field in ("description", "created_at"):
        if field in payload and (not isinstance(payload[field], str) or not payload[field]):
            errors.append(f"{field}: must be a non-empty string")
    if "notes" in payload and not isinstance(payload["notes"], str):
        errors.append("notes: must be a string")

    entries = payload.get("entries")
    if not isinstance(entries, list):
        errors.append("entries: must be an array")
    return errors


def _entry_errors(index: int, entry: Any, manifest_dir: Path, seen_ids: set[str]) -> list[str]:
    prefix = f"entries/{index}"
    if not isinstance(entry, dict):
        return [f"{prefix}: must be an object"]

    errors: list[str] = []
    errors.extend(_unknown_fields(prefix, entry, ENTRY_FIELDS))

    for field in ("id", "path", "split", "shot_label", "source_type"):
        if field not in entry:
            errors.append(f"{prefix}/{field}: required property is missing")

    if "id" in entry:
        _validate_id(errors, f"{prefix}/id", entry["id"])
        if isinstance(entry["id"], str):
            if entry["id"] in seen_ids:
                errors.append(f"{prefix}/id duplicate entry id: {entry['id']}")
            seen_ids.add(entry["id"])

    if entry.get("shot_label") not in SHOT_LABELS:
        errors.append(f"{prefix}/shot_label: must be one of {', '.join(sorted(SHOT_LABELS))}")

    if entry.get("source_type") not in SOURCE_TYPES:
        errors.append(f"{prefix}/source_type: must be one of {', '.join(sorted(SOURCE_TYPES))}")

    if entry.get("split") not in SPLITS:
        errors.append(f"{prefix}/split: must be one of {', '.join(sorted(SPLITS))}")

    path_value = entry.get("path")
    if isinstance(path_value, str) and path_value:
        target = _resolve_safe_relative_path(path_value, manifest_dir)
        if target is None:
            errors.append(f"{prefix}/path: must be relative and stay within the manifest directory")
        elif not target.is_file():
            errors.append(f"{prefix}/path: file does not exist: {path_value}")
    elif "path" in entry:
        errors.append(f"{prefix}/path: must be a non-empty string")

    if "fps" in entry and not _is_positive_number(entry["fps"]):
        errors.append(f"{prefix}/fps: must be a positive number")
    if "contact_time_ms" in entry and not _is_nonnegative_number(entry["contact_time_ms"]):
        errors.append(f"{prefix}/contact_time_ms: must be a non-negative number")
    if "window_ms" in entry and not _is_positive_number(entry["window_ms"]):
        errors.append(f"{prefix}/window_ms: must be a positive number")
    if "player_id" in entry:
        _validate_id(errors, f"{prefix}/player_id", entry["player_id"])
    if "notes" in entry and not isinstance(entry["notes"], str):
        errors.append(f"{prefix}/notes: must be a string")

    return errors


def _coverage_counts(entries: list[dict[str, Any]]) -> dict[str, Any]:
    label_counts = Counter(entry.get("shot_label") for entry in entries if entry.get("shot_label") in SHOT_LABELS)
    source_counts = Counter(entry.get("source_type") for entry in entries if entry.get("source_type") in SOURCE_TYPES)
    split_counts = Counter(entry.get("split") for entry in entries if entry.get("split") in SPLITS)
    by_split_shot_label: dict[str, dict[str, int]] = {}

    for entry in entries:
        split = entry.get("split")
        shot_label = entry.get("shot_label")
        if split not in SPLITS or shot_label not in SHOT_LABELS:
            continue
        by_split_shot_label.setdefault(split, {})
        by_split_shot_label[split][shot_label] = by_split_shot_label[split].get(shot_label, 0) + 1

    return {
        "shot_label": {label: label_counts.get(label, 0) for label in SHOT_LABELS},
        "source_type": {source_type: source_counts.get(source_type, 0) for source_type in SOURCE_TYPES},
        "split": {split: split_counts.get(split, 0) for split in sorted(SPLITS)},
        "by_split_shot_label": {
            split: dict(sorted(counts.items())) for split, counts in sorted(by_split_shot_label.items())
        },
    }


def _coverage_gaps(coverage_counts: dict[str, Any]) -> list[str]:
    gaps: list[str] = []
    split_counts = coverage_counts["split"]
    if split_counts.get("val", 0) == 0:
        gaps.append("missing val entries")
    if split_counts.get("test", 0) == 0:
        gaps.append("missing test entries")

    label_counts = coverage_counts["shot_label"]
    missing_labels = sorted(label for label, count in label_counts.items() if count == 0)
    if missing_labels:
        gaps.append(f"missing key shot classes: {', '.join(missing_labels)}")
    return gaps


def _unknown_fields(prefix: str, payload: dict[str, Any], allowed: set[str]) -> list[str]:
    errors = []
    for field in sorted(set(payload) - allowed):
        path = f"{prefix}/{field}" if prefix else field
        errors.append(f"{path}: additional property is not allowed")
    return errors


def _validate_id(errors: list[str], path: str, value: Any) -> None:
    if not isinstance(value, str) or not ID_PATTERN.match(value):
        errors.append(f"{path}: must match ^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _resolve_safe_relative_path(value: str, root: Path) -> Path | None:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        return None
    root_resolved = root.resolve()
    target = (root / path).resolve()
    try:
        target.relative_to(root_resolved)
    except ValueError:
        return None
    return target


def _is_positive_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and value > 0


def _is_nonnegative_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and value >= 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a shot-class dataset manifest.")
    parser.add_argument("manifest", type=Path, help="Path to a shot-class dataset manifest JSON file.")
    args = parser.parse_args(argv)

    summary = validate_manifest(args.manifest)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

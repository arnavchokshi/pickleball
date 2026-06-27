#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
SOURCE_TYPES = {"ball_track", "pop_audio"}
SPLITS = {"train", "val", "test"}
TOP_LEVEL_FIELDS = {"schema_version", "entries", "dataset_id", "description", "created_at", "notes"}
ENTRY_FIELDS = {"id", "path", "split", "source_type", "frame_rate", "sample_rate", "notes"}
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

    coverage_counts = _coverage_counts(entries)
    coverage_gaps = _coverage_gaps(coverage_counts["source_type"])
    valid = not errors

    return {
        "schema_version": SCHEMA_VERSION,
        "manifest": str(manifest_path),
        "valid": valid,
        "dataset_ready": valid and not coverage_gaps,
        "entry_count": len(entries) if valid else len(entries),
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


def _top_level_errors(payload: Any) -> list[str]:
    if not isinstance(payload, dict):
        return ["manifest: must be an object"]

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

    for field in ("id", "path", "split", "source_type"):
        if field not in entry:
            errors.append(f"{prefix}/{field}: required property is missing")

    if "id" in entry:
        _validate_id(errors, f"{prefix}/id", entry["id"])
        if isinstance(entry["id"], str):
            if entry["id"] in seen_ids:
                errors.append(f"{prefix}/id duplicate entry id: {entry['id']}")
            seen_ids.add(entry["id"])

    source_type = entry.get("source_type")
    if source_type not in SOURCE_TYPES:
        errors.append(f"{prefix}/source_type: must be one of ball_track, pop_audio")

    split = entry.get("split")
    if split not in SPLITS:
        errors.append(f"{prefix}/split: must be one of test, train, val")

    path_value = entry.get("path")
    if isinstance(path_value, str):
        target = _resolve_safe_relative_path(path_value, manifest_dir)
        if target is None:
            errors.append(f"{prefix}/path: must be relative and stay within the manifest directory")
        elif not target.is_file():
            errors.append(f"{prefix}/path: file does not exist: {path_value}")
    elif "path" in entry:
        errors.append(f"{prefix}/path: must be a string")

    frame_rate = entry.get("frame_rate")
    if "frame_rate" in entry and not _is_positive_number(frame_rate):
        errors.append(f"{prefix}/frame_rate: must be a positive number")

    sample_rate = entry.get("sample_rate")
    if "sample_rate" in entry and not _is_positive_integer(sample_rate):
        errors.append(f"{prefix}/sample_rate: must be a positive integer")
    if source_type == "pop_audio" and "sample_rate" in entry and sample_rate != 44100:
        errors.append(f"{prefix}/sample_rate: pop_audio sources must be 44100 Hz when provided")

    if "notes" in entry and not isinstance(entry["notes"], str):
        errors.append(f"{prefix}/notes: must be a string")

    return errors


def _coverage_counts(entries: list[dict[str, Any]]) -> dict[str, Any]:
    source_counts = Counter(entry.get("source_type") for entry in entries if entry.get("source_type") in SOURCE_TYPES)
    split_counts = Counter(entry.get("split") for entry in entries if entry.get("split") in SPLITS)
    by_split_source: dict[str, dict[str, int]] = {}
    for entry in entries:
        split = entry.get("split")
        source_type = entry.get("source_type")
        if split not in SPLITS or source_type not in SOURCE_TYPES:
            continue
        by_split_source.setdefault(split, {})
        by_split_source[split][source_type] = by_split_source[split].get(source_type, 0) + 1

    return {
        "source_type": {source_type: source_counts.get(source_type, 0) for source_type in sorted(SOURCE_TYPES)},
        "split": {split: split_counts.get(split, 0) for split in sorted(SPLITS)},
        "by_split_source_type": {
            split: dict(sorted(counts.items())) for split, counts in sorted(by_split_source.items())
        },
    }


def _coverage_gaps(source_type_counts: dict[str, int]) -> list[str]:
    gaps = []
    if source_type_counts.get("ball_track", 0) == 0:
        gaps.append("missing ball_track entries")
    if source_type_counts.get("pop_audio", 0) == 0:
        gaps.append("missing pop_audio entries")
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


def _is_positive_integer(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value > 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a ball-track and pop-audio dataset manifest.")
    parser.add_argument("manifest", type=Path, help="Path to a dataset manifest JSON file.")
    args = parser.parse_args(argv)

    summary = validate_manifest(args.manifest)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

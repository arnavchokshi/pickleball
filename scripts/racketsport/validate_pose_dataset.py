#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
SOURCE_TYPES = ("amass", "athletepose3d", "bedlam2", "caltennis", "emdb_eval", "rich")
FINE_TUNE_LADDER = ("bedlam2", "athletepose3d", "caltennis", "rich", "amass")
SPLITS = ("eval", "test", "train", "val")
TOP_LEVEL_FIELDS = {"schema_version", "dataset_id", "sources", "notes"}
SOURCE_FIELDS = {
    "id",
    "source_type",
    "path",
    "split",
    "fps",
    "frame_count",
    "joint_set",
    "license",
    "notes",
}
ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def validate_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    payload = _read_json(manifest_path)
    base_dir = manifest_path.parent.resolve()

    errors = _schema_errors(payload)
    if isinstance(payload, dict):
        errors.extend(_semantic_errors(payload, base_dir))
    if errors:
        raise ValueError("\n".join(errors))

    sources = payload["sources"]
    source_counts = Counter(source["source_type"] for source in sources)
    split_counts = Counter(source["split"] for source in sources)
    source_type_counts = {source_type: source_counts.get(source_type, 0) for source_type in SOURCE_TYPES}
    split_type_counts = {split: split_counts.get(split, 0) for split in SPLITS}
    coverage = _coverage_summary(source_type_counts)

    return {
        "valid": True,
        "schema_version": payload["schema_version"],
        "dataset_id": payload["dataset_id"],
        "manifest": str(manifest_path),
        "total_sources": len(sources),
        "source_type_counts": source_type_counts,
        "split_counts": split_type_counts,
        "dataset_ready": not coverage["gaps"],
        "coverage_summary": coverage,
    }


def _coverage_summary(source_type_counts: dict[str, int]) -> dict[str, Any]:
    missing_fine_tune_sources = [
        source_type for source_type in FINE_TUNE_LADDER if source_type_counts.get(source_type, 0) == 0
    ]
    gaps = [f"missing fine-tune ladder source: {source_type}" for source_type in missing_fine_tune_sources]
    if source_type_counts.get("emdb_eval", 0) == 0:
        gaps.append("no emdb_eval entries registered for eval coverage")

    return {
        "fine_tune_ladder": list(FINE_TUNE_LADDER),
        "missing_fine_tune_sources": missing_fine_tune_sources,
        "has_emdb_eval": source_type_counts.get("emdb_eval", 0) > 0,
        "gaps": gaps,
    }


def _schema_errors(payload: Any) -> list[str]:
    if not isinstance(payload, dict):
        return ["manifest must be an object"]

    errors: list[str] = []
    errors.extend(_unknown_fields("", payload, TOP_LEVEL_FIELDS))
    for field in ("schema_version", "dataset_id", "sources"):
        if field not in payload:
            errors.append(f"{field}: required property is missing")

    if payload.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version: must equal {SCHEMA_VERSION}")
    _validate_id(errors, "dataset_id", payload.get("dataset_id"))
    _validate_string_list(errors, "notes", payload.get("notes"), required=False)

    sources = payload.get("sources")
    if not isinstance(sources, list):
        errors.append("sources: must be an array")
        return errors

    seen_ids: set[str] = set()
    for index, source in enumerate(sources):
        errors.extend(_source_errors(index, source, seen_ids))
    return errors


def _source_errors(index: int, source: Any, seen_ids: set[str]) -> list[str]:
    prefix = f"sources/{index}"
    if not isinstance(source, dict):
        return [f"{prefix}: must be an object"]

    errors: list[str] = []
    errors.extend(_unknown_fields(prefix, source, SOURCE_FIELDS))
    for field in ("id", "source_type", "path", "split"):
        if field not in source:
            errors.append(f"{prefix}/{field}: required property is missing")

    source_id = source.get("id")
    _validate_id(errors, f"{prefix}/id", source_id)
    if isinstance(source_id, str):
        if source_id in seen_ids:
            errors.append(f"{prefix}/id duplicate source id: {source_id}")
        seen_ids.add(source_id)

    if source.get("source_type") not in SOURCE_TYPES:
        errors.append(f"{prefix}/source_type: must be one of {', '.join(SOURCE_TYPES)}")
    if source.get("split") not in SPLITS:
        errors.append(f"{prefix}/split: must be one of {', '.join(SPLITS)}")

    source_path = source.get("path")
    if not isinstance(source_path, str) or not source_path:
        errors.append(f"{prefix}/path: must be a non-empty string")

    if "fps" in source:
        fps = source["fps"]
        if isinstance(fps, bool) or not isinstance(fps, (int, float)) or fps <= 0:
            errors.append(f"{prefix}/fps: must be a positive number")
    if "frame_count" in source:
        frame_count = source["frame_count"]
        if isinstance(frame_count, bool) or not isinstance(frame_count, int) or frame_count <= 0:
            errors.append(f"{prefix}/frame_count: must be a positive integer")
    for field in ("joint_set", "license"):
        if field in source and (not isinstance(source[field], str) or not source[field]):
            errors.append(f"{prefix}/{field}: must be a non-empty string")
    _validate_string_list(errors, f"{prefix}/notes", source.get("notes"), required=False)
    return errors


def _semantic_errors(payload: dict[str, Any], base_dir: Path) -> list[str]:
    sources = payload.get("sources")
    if not isinstance(sources, list):
        return []

    errors: list[str] = []
    for index, source in enumerate(sources):
        if not isinstance(source, dict):
            continue

        raw_path = source.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            continue

        relative_path = Path(raw_path)
        if _is_unsafe_relative_path(relative_path):
            errors.append(f"sources/{index}/path must be relative and stay within the manifest directory")
            continue

        candidate = base_dir / relative_path
        try:
            candidate.resolve(strict=False).relative_to(base_dir)
        except ValueError:
            errors.append(f"sources/{index}/path must be relative and stay within the manifest directory")
            continue

        if not candidate.is_file():
            errors.append(f"sources/{index}/path file does not exist: {raw_path}")
    return errors


def _unknown_fields(prefix: str, payload: dict[str, Any], allowed: set[str]) -> list[str]:
    errors = []
    for field in sorted(set(payload) - allowed):
        path = f"{prefix}/{field}" if prefix else field
        errors.append(f"{path}: additional property is not allowed")
    return errors


def _validate_id(errors: list[str], path: str, value: Any) -> None:
    if not isinstance(value, str) or not ID_PATTERN.match(value):
        errors.append(f"{path}: must match ^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _validate_string_list(errors: list[str], path: str, value: Any, *, required: bool) -> None:
    if value is None and not required:
        return
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        errors.append(f"{path}: must be an array of non-empty strings")


def _is_unsafe_relative_path(path: Path) -> bool:
    return path.is_absolute() or ".." in path.parts


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"{path} does not exist") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON: {exc}") from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a pose/body dataset source manifest.")
    parser.add_argument("manifest", type=Path, help="Path to a pose dataset source manifest JSON file.")
    args = parser.parse_args(argv)

    try:
        summary = validate_manifest(args.manifest)
    except ValueError as exc:
        print("ERROR: pose dataset manifest failed validation:", file=sys.stderr)
        for line in str(exc).splitlines():
            print(f"- {line}", file=sys.stderr)
        return 1

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

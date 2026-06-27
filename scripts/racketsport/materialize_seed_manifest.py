#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


VIDEO_SUFFIXES = (".mp4", ".mov", ".m4v", ".avi", ".mkv")
METADATA_FIELDS = (
    "camera_height",
    "camera_angle",
    "play_type",
    "environment",
    "frame_rate_fps",
    "duration_s",
    "racket_gt",
)
SOURCE_FILE_FIELDS = ("source_file", "downloaded_file", "local_file", "file")


def _load_source_manifest(manifest_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("source manifest must be a JSON object with a clips list")
    rows = payload.get("clips")
    if not isinstance(rows, list):
        raise ValueError("source manifest must contain a clips list")
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"clip row {index} must be an object")
        if not isinstance(row.get("name"), str) or not row["name"]:
            raise ValueError(f"clip row {index} must have a non-empty name")
        if not isinstance(row.get("metadata"), dict):
            raise ValueError(f"clip row {index} must have a metadata object")
    return rows


def _source_file_from_row(row: dict[str, Any]) -> str | None:
    for field in SOURCE_FILE_FIELDS:
        value = row.get(field)
        if isinstance(value, str) and value:
            return value
    return None


def _resolve_source_path(
    *,
    row: dict[str, Any],
    downloaded_source_root: Path,
    allow_missing: bool,
) -> tuple[Path, bool]:
    explicit_source = _source_file_from_row(row)
    if explicit_source is not None:
        source_path = downloaded_source_root / explicit_source
        if source_path.is_file() or allow_missing:
            return source_path, not source_path.is_file()
        raise FileNotFoundError(source_path)

    name = row["name"]
    candidates = [downloaded_source_root / f"{name}{suffix}" for suffix in VIDEO_SUFFIXES]
    present = [candidate for candidate in candidates if candidate.is_file()]
    if len(present) == 1:
        return present[0], False
    if len(present) > 1:
        matches = ", ".join(str(candidate) for candidate in present)
        raise ValueError(f"multiple source files found for {name}: {matches}")
    if allow_missing:
        return downloaded_source_root / f"{name}.mp4", True
    raise FileNotFoundError(downloaded_source_root / f"{name}<video suffix>")


def _registrar_row(
    *,
    source_path: Path,
    source_row: dict[str, Any],
) -> dict[str, Any]:
    metadata = source_row["metadata"]
    row: dict[str, Any] = {"source": str(source_path), "name": source_row["name"]}
    for field in METADATA_FIELDS:
        if field in metadata:
            row[field] = metadata[field]
    return row


def materialize_seed_manifest(
    *,
    source_manifest_path: Path,
    output_manifest_path: Path,
    downloaded_source_root: Path,
    allow_missing: bool = False,
) -> dict[str, Any]:
    source_rows = _load_source_manifest(source_manifest_path)
    registrar_rows: list[dict[str, Any]] = []
    missing: list[str] = []

    for source_row in source_rows:
        source_path, is_missing = _resolve_source_path(
            row=source_row,
            downloaded_source_root=downloaded_source_root,
            allow_missing=allow_missing,
        )
        if is_missing:
            missing.append(source_row["name"])
        registrar_rows.append(_registrar_row(source_path=source_path, source_row=source_row))

    output = {
        "schema_version": 1,
        "source_manifest": str(source_manifest_path),
        "downloaded_source_root": str(downloaded_source_root),
        "clips": registrar_rows,
    }
    output_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    output_manifest_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return {
        "source_manifest": str(source_manifest_path),
        "output_manifest": str(output_manifest_path),
        "downloaded_source_root": str(downloaded_source_root),
        "requested_count": len(source_rows),
        "materialized_count": len(registrar_rows),
        "missing_count": len(missing),
        "missing_clips": missing,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a DATA-1 source seed manifest into a registrar-compatible "
            "manifest for already-downloaded local source files."
        )
    )
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--downloaded-source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Write expected source paths even when downloaded files are not present.",
    )
    args = parser.parse_args()

    summary = materialize_seed_manifest(
        source_manifest_path=args.source_manifest,
        output_manifest_path=args.output,
        downloaded_source_root=args.downloaded_source_root,
        allow_missing=args.allow_missing,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

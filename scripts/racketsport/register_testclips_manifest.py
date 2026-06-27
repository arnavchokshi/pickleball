#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from register_testclip import register_testclip
from threed.racketsport.testclips import TestClipMetadata, build_testclip_manifest


METADATA_FIELDS = {
    "camera_height",
    "camera_angle",
    "play_type",
    "environment",
    "frame_rate_fps",
    "duration_s",
    "racket_gt",
}
ROW_FIELDS = {"source", "name", "symlink"} | METADATA_FIELDS


def _load_rows(manifest_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = payload.get("clips") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError("manifest must be a JSON list or an object with a clips list")
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"clip row {index} must be an object")
        extra_fields = sorted(set(row) - ROW_FIELDS)
        if extra_fields:
            raise ValueError(f"clip row {index} has unknown fields: {', '.join(extra_fields)}")
    return rows


def _source_path(manifest_path: Path, source_value: str) -> Path:
    source = Path(source_value)
    if source.is_absolute():
        return source
    return manifest_path.parent / source


def register_manifest(
    *,
    manifest_path: Path,
    root: Path,
    symlink: bool = False,
    continue_on_error: bool = False,
) -> dict[str, Any]:
    rows = _load_rows(manifest_path)
    clips: list[dict[str, Any]] = []
    failed_count = 0

    for index, row in enumerate(rows):
        try:
            metadata = TestClipMetadata(
                schema_version=1,
                **{field: row[field] for field in METADATA_FIELDS if field in row},
            )
            result = register_testclip(
                source=_source_path(manifest_path, row["source"]),
                root=root,
                name=row["name"],
                metadata=metadata,
                symlink=bool(row.get("symlink", symlink)),
            )
            clips.append({"clip": result["clip"], "status": "registered", **result})
        except Exception as exc:
            failed_count += 1
            clips.append(
                {
                    "clip": str(row.get("name", f"row_{index}")),
                    "status": "failed",
                    "error": str(exc),
                }
            )
            if not continue_on_error:
                break

    dataset_manifest = build_testclip_manifest(root)
    return {
        "manifest": str(manifest_path),
        "root": str(root),
        "requested_count": len(rows),
        "registered_count": sum(1 for clip in clips if clip["status"] == "registered"),
        "failed_count": failed_count,
        "clips": clips,
        "dataset_total_clips": dataset_manifest.total_clips,
        "metadata_ready_clips": dataset_manifest.metadata_ready_clips,
        "ready_clips": dataset_manifest.ready_clips,
        "dataset_ready": dataset_manifest.dataset_ready,
        "coverage_gaps": dataset_manifest.coverage_gaps,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Register DATA-1 candidate test clips from a JSON manifest.")
    parser.add_argument("--manifest", type=Path, required=True, help="JSON list or object with a clips list.")
    parser.add_argument("--root", type=Path, default=Path("data/testclips"))
    parser.add_argument("--symlink", action="store_true", help="Symlink sources by default instead of copying them.")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue registering later rows after a row fails.",
    )
    args = parser.parse_args()

    summary = register_manifest(
        manifest_path=args.manifest,
        root=args.root,
        symlink=args.symlink,
        continue_on_error=args.continue_on_error,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["failed_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

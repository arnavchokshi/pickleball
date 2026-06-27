#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.testclips import REQUIRED_LABEL_FILES, TestClipManifest, build_testclip_manifest


SCHEMA_VERSION = 1
EXPECTED_DRAFT_STATUS = "draft_manual_annotation"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _frame_pack(frames_root: Path | None, clip_name: str) -> dict[str, Any]:
    if frames_root is None:
        return {
            "manifest_path": None,
            "present": False,
            "frame_count": 0,
            "sample_every_frames": None,
            "warnings": ["frames_root not provided"],
        }

    manifest_path = frames_root / clip_name / "label_frame_manifest.json"
    if not manifest_path.is_file():
        return {
            "manifest_path": str(manifest_path),
            "present": False,
            "frame_count": 0,
            "sample_every_frames": None,
            "warnings": ["frame-pack manifest missing"],
        }

    try:
        manifest = _read_json(manifest_path)
    except json.JSONDecodeError as exc:
        return {
            "manifest_path": str(manifest_path),
            "present": False,
            "frame_count": 0,
            "sample_every_frames": None,
            "warnings": [f"frame-pack manifest invalid JSON: {exc}"],
        }

    frames = manifest.get("frames", []) if isinstance(manifest, dict) else []
    warnings: list[str] = []
    if not isinstance(manifest, dict):
        warnings.append("frame-pack manifest must be an object")
    elif not isinstance(frames, list):
        warnings.append("frame-pack manifest frames field must be a list")
        frames = []

    raw_frame_count = manifest.get("frame_count", len(frames)) if isinstance(manifest, dict) else 0
    frame_count = len(frames)
    if isinstance(raw_frame_count, int) and raw_frame_count >= 0:
        frame_count = raw_frame_count
    else:
        warnings.append("frame-pack manifest frame_count must be a non-negative integer")

    return {
        "manifest_path": str(manifest_path),
        "present": not warnings,
        "frame_count": frame_count,
        "sample_every_frames": manifest.get("sample_every_frames") if isinstance(manifest, dict) else None,
        "warnings": warnings,
    }


def _annotation_items(payload: dict[str, Any], errors: list[str]) -> int:
    annotation = payload.get("annotation")
    if not isinstance(annotation, dict):
        errors.append("annotation must be an object")
        return 0

    items = annotation.get("items")
    if items is None:
        return 0
    if not isinstance(items, list):
        errors.append("annotation.items must be a list")
        return 0
    return len(items)


def _target_file_ok(payload: dict[str, Any], expected_label_file: str, errors: list[str]) -> bool:
    annotation = payload.get("annotation")
    actual_target = annotation.get("target_file") if isinstance(annotation, dict) else None
    if actual_target != expected_label_file:
        errors.append(f"annotation.target_file must equal {expected_label_file}")
        return False
    return True


def _audit_draft_file(path: Path, label_file: str) -> dict[str, Any]:
    if not path.is_file():
        return {
            "path": str(path),
            "present": False,
            "json_valid": False,
            "schema_version_ok": False,
            "status_ok": False,
            "target_file_ok": False,
            "annotation_item_count": 0,
            "warnings": [],
            "errors": ["missing required label draft"],
        }

    warnings: list[str] = []
    errors: list[str] = []
    try:
        payload = _read_json(path)
    except json.JSONDecodeError as exc:
        return {
            "path": str(path),
            "present": True,
            "json_valid": False,
            "schema_version_ok": False,
            "status_ok": False,
            "target_file_ok": False,
            "annotation_item_count": 0,
            "warnings": warnings,
            "errors": [f"invalid JSON: {exc}"],
        }

    if not isinstance(payload, dict):
        return {
            "path": str(path),
            "present": True,
            "json_valid": True,
            "schema_version_ok": False,
            "status_ok": False,
            "target_file_ok": False,
            "annotation_item_count": 0,
            "warnings": warnings,
            "errors": ["draft payload must be an object"],
        }

    schema_version_ok = payload.get("schema_version") == SCHEMA_VERSION
    if not schema_version_ok:
        errors.append(f"schema_version must equal {SCHEMA_VERSION}")

    status_ok = payload.get("status") == EXPECTED_DRAFT_STATUS
    if not status_ok:
        errors.append(f"status must equal {EXPECTED_DRAFT_STATUS}")

    target_file_ok = _target_file_ok(payload, label_file, errors)
    annotation_item_count = _annotation_items(payload, errors)
    _draft_frame_warnings(payload, warnings)

    return {
        "path": str(path),
        "present": True,
        "json_valid": True,
        "schema_version_ok": schema_version_ok,
        "status_ok": status_ok,
        "target_file_ok": target_file_ok,
        "annotation_item_count": annotation_item_count,
        "warnings": warnings,
        "errors": errors,
    }


def _draft_frame_warnings(payload: dict[str, Any], warnings: list[str]) -> None:
    frames = payload.get("frames")
    if frames is None:
        warnings.append("draft frames context missing")
        return
    if not isinstance(frames, dict):
        warnings.append("draft frames context must be an object")
        return
    if "frame_count" in frames and not isinstance(frames["frame_count"], int):
        warnings.append("draft frames.frame_count must be an integer")
    if "frames" in frames and not isinstance(frames["frames"], list):
        warnings.append("draft frames.frames must be a list")


def _clip_report(clip: TestClipManifest, drafts_root: Path, frames_root: Path | None) -> dict[str, Any]:
    drafts_dir = drafts_root / clip.name / "labels"
    frame_pack = _frame_pack(frames_root, clip.name)
    drafts = {
        label_file: _audit_draft_file(drafts_dir / label_file, label_file)
        for label_file in REQUIRED_LABEL_FILES
    }
    missing_required = [label for label, draft in drafts.items() if not draft["present"]]
    errors = [
        f"{label}: {error}"
        for label, draft in drafts.items()
        for error in draft["errors"]
    ]
    warnings = list(frame_pack["warnings"])
    warnings.extend(
        f"{label}: {warning}"
        for label, draft in drafts.items()
        for warning in draft["warnings"]
    )

    return {
        "name": clip.name,
        "clip_dir": str(clip.path),
        "draft_labels_dir": str(drafts_dir),
        "dataset_labels_dir": str(clip.labels_dir),
        "status": "ready" if not errors else "not_ready",
        "missing_required_label_drafts": missing_required,
        "frame_pack": frame_pack,
        "drafts": drafts,
        "warnings": warnings,
        "errors": errors,
    }


def audit_label_drafts(*, root: Path, drafts_root: Path, frames_root: Path | None = None) -> dict[str, Any]:
    manifest = build_testclip_manifest(root)
    clip_reports = [
        _clip_report(clip=clip, drafts_root=drafts_root, frames_root=frames_root)
        for clip in manifest.clips
    ]
    ready_clips = sum(1 for clip in clip_reports if clip["status"] == "ready")
    missing_required = sum(len(clip["missing_required_label_drafts"]) for clip in clip_reports)
    invalid_drafts = sum(
        1
        for clip in clip_reports
        for draft in clip["drafts"].values()
        if draft["present"] and draft["errors"]
    )
    warning_count = sum(len(clip["warnings"]) for clip in clip_reports)
    errors: list[str] = []
    if not manifest.root_exists:
        errors.append(f"testclip root does not exist: {root}")
    if manifest.total_clips == 0:
        errors.append("no test clips discovered")
    errors.extend(
        f"{clip['name']}: {error}"
        for clip in clip_reports
        for error in clip["errors"]
    )

    ready = manifest.root_exists and manifest.total_clips > 0 and ready_clips == manifest.total_clips
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "racketsport_label_draft_audit",
        "status": "ready" if ready else "not_ready",
        "root": str(root),
        "drafts_root": str(drafts_root),
        "frames_root": str(frames_root) if frames_root is not None else None,
        "dataset_labels_written": False,
        "required_label_files": list(REQUIRED_LABEL_FILES),
        "counts": {
            "total_clips": manifest.total_clips,
            "ready_clips": ready_clips,
            "not_ready_clips": manifest.total_clips - ready_clips,
            "missing_required_drafts": missing_required,
            "invalid_drafts": invalid_drafts,
            "warnings": warning_count,
        },
        "warnings": [
            f"{clip['name']}: {warning}"
            for clip in clip_reports
            for warning in clip["warnings"]
        ],
        "errors": errors,
        "clips": {clip["name"]: clip for clip in clip_reports},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit DATA-1 draft label files before dataset-label promotion.")
    parser.add_argument("--root", type=Path, default=Path("data/testclips"))
    parser.add_argument("--drafts-root", type=Path, default=Path("runs/label_drafts"))
    parser.add_argument("--frames-root", type=Path, help="Optional output root from extract_label_frames.py.")
    args = parser.parse_args()

    report = audit_label_drafts(root=args.root, drafts_root=args.drafts_root, frames_root=args.frames_root)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

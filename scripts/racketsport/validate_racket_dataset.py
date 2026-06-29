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
SOURCE_TYPES = ("racketvision", "synthetic_blenderproc", "aruco_gt")
SPLITS = ("eval", "test", "train", "val")
ANNOTATION_FIELDS = {"keypoints", "corners", "pose_labels", "segmentation_masks"}
TOP_LEVEL_FIELDS = {"schema_version", "dataset_id", "sources", "notes"}
SOURCE_FIELDS = {
    "id",
    "source_type",
    "path",
    "split",
    "annotations",
    "camera",
    "fps",
    "frame_count",
    "license",
    "marker",
    "notes",
}
ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
KEYPOINT_NAMES = ("top", "bottom", "handle")
CORNER_NAMES = ("top_left", "top_right", "bottom_right", "bottom_left")
POSE_LABEL_FORMATS = ("aruco_marker_pose", "matrix_4x4", "quaternion_translation", "rvec_tvec")
SEGMENTATION_MASK_FORMATS = ("coco_rle", "png", "polygon")


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
    counts = Counter(source["source_type"] for source in sources)
    split_counts = Counter(source["split"] for source in sources)
    source_type_counts = {source_type: counts.get(source_type, 0) for source_type in SOURCE_TYPES}
    split_type_counts = {split: split_counts.get(split, 0) for split in SPLITS}
    annotation_counts = _annotation_counts(sources)
    coverage = _coverage_summary(source_type_counts, split_type_counts, sources)
    content_gaps = _content_gaps(sources, base_dir)

    return {
        "valid": True,
        "schema_version": payload["schema_version"],
        "dataset_id": payload["dataset_id"],
        "manifest": str(manifest_path),
        "total_sources": len(sources),
        "source_type_counts": source_type_counts,
        "split_counts": split_type_counts,
        "annotation_counts": annotation_counts,
        "dataset_ready": not coverage["gaps"] and not content_gaps,
        "coverage_summary": coverage,
        "content_gaps": content_gaps,
    }


def _annotation_counts(sources: list[dict[str, Any]]) -> dict[str, int]:
    counts = {field: 0 for field in ANNOTATION_FIELDS}
    for source in sources:
        annotations = source.get("annotations")
        if not isinstance(annotations, dict):
            continue
        for field in ANNOTATION_FIELDS:
            if field in annotations:
                counts[field] += 1
    return dict(sorted(counts.items()))


def _coverage_summary(
    source_type_counts: dict[str, int],
    split_counts: dict[str, int],
    sources: list[dict[str, Any]],
) -> dict[str, Any]:
    missing_source_types = [
        source_type for source_type in SOURCE_TYPES if source_type_counts.get(source_type, 0) == 0
    ]
    missing_splits = [split for split in SPLITS if split_counts.get(split, 0) == 0]
    has_aruco_gt_eval = any(
        source.get("source_type") == "aruco_gt" and source.get("split") == "eval" for source in sources
    )

    gaps = [f"missing source type: {source_type}" for source_type in missing_source_types]
    gaps.extend(f"missing split: {split}" for split in missing_splits)
    if not has_aruco_gt_eval:
        gaps.append("no aruco_gt eval entries registered for racket face-angle GT coverage")

    return {
        "required_source_types": list(SOURCE_TYPES),
        "required_splits": list(SPLITS),
        "missing_source_types": missing_source_types,
        "missing_splits": missing_splits,
        "has_aruco_gt_eval": has_aruco_gt_eval,
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
    for field in ("id", "source_type", "path", "split", "annotations"):
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

    source_type = source.get("source_type")
    errors.extend(_annotation_errors(prefix, source.get("annotations"), source_type))

    if "camera" in source and not isinstance(source["camera"], dict):
        errors.append(f"{prefix}/camera: must be an object")
    if "marker" in source and not isinstance(source["marker"], dict):
        errors.append(f"{prefix}/marker: must be an object")
    if source_type == "aruco_gt" and "marker" not in source:
        errors.append(f"{prefix}/marker: required for aruco_gt sources")
    if "fps" in source:
        fps = source["fps"]
        if isinstance(fps, bool) or not isinstance(fps, (int, float)) or fps <= 0:
            errors.append(f"{prefix}/fps: must be a positive number")
    if "frame_count" in source:
        frame_count = source["frame_count"]
        if isinstance(frame_count, bool) or not isinstance(frame_count, int) or frame_count <= 0:
            errors.append(f"{prefix}/frame_count: must be a positive integer")
    if "license" in source and (not isinstance(source["license"], str) or not source["license"]):
        errors.append(f"{prefix}/license: must be a non-empty string")
    _validate_string_list(errors, f"{prefix}/notes", source.get("notes"), required=False)
    return errors


def _annotation_errors(prefix: str, annotations: Any, source_type: Any) -> list[str]:
    path = f"{prefix}/annotations"
    if not isinstance(annotations, dict):
        return [f"{path}: must be an object"]

    errors: list[str] = []
    errors.extend(_unknown_fields(path, annotations, ANNOTATION_FIELDS))

    if "keypoints" not in annotations:
        errors.append(f"{path}/keypoints: required property is missing")
    else:
        errors.extend(_keypoint_errors(f"{path}/keypoints", annotations["keypoints"]))

    if "corners" not in annotations:
        errors.append(f"{path}/corners: required property is missing")
    else:
        errors.extend(_corner_errors(f"{path}/corners", annotations["corners"]))

    if "pose_labels" in annotations:
        errors.extend(_pose_label_errors(f"{path}/pose_labels", annotations["pose_labels"]))
    elif source_type in {"aruco_gt", "synthetic_blenderproc"}:
        errors.append(f"{path}/pose_labels: required for {source_type} sources")

    if "segmentation_masks" in annotations:
        errors.extend(_segmentation_mask_errors(f"{path}/segmentation_masks", annotations["segmentation_masks"]))

    return errors


def _keypoint_errors(path: str, value: Any) -> list[str]:
    if not isinstance(value, dict):
        return [f"{path}: must be an object"]

    errors: list[str] = []
    errors.extend(_unknown_fields(path, value, {"names", "dimensions"}))
    names = value.get("names")
    if not _is_string_list(names):
        errors.append(f"{path}/names: must be an array of non-empty strings")
    elif any(name not in names for name in KEYPOINT_NAMES):
        errors.append(f"{path}/names: must include {', '.join(KEYPOINT_NAMES)}")
    dimensions = value.get("dimensions")
    if isinstance(dimensions, bool) or dimensions not in (2, 3):
        errors.append(f"{path}/dimensions: must be 2 or 3")
    return errors


def _corner_errors(path: str, value: Any) -> list[str]:
    if not isinstance(value, dict):
        return [f"{path}: must be an object"]

    errors: list[str] = []
    errors.extend(_unknown_fields(path, value, {"names", "dimensions"}))
    names = value.get("names")
    if not _is_string_list(names):
        errors.append(f"{path}/names: must be an array of non-empty strings")
    elif list(names) != list(CORNER_NAMES):
        errors.append(f"{path}/names: must equal {', '.join(CORNER_NAMES)}")
    dimensions = value.get("dimensions")
    if isinstance(dimensions, bool) or dimensions not in (2, 3):
        errors.append(f"{path}/dimensions: must be 2 or 3")
    return errors


def _pose_label_errors(path: str, value: Any) -> list[str]:
    if not isinstance(value, dict):
        return [f"{path}: must be an object"]

    errors: list[str] = []
    errors.extend(_unknown_fields(path, value, {"format", "coordinate_frame"}))
    if value.get("format") not in POSE_LABEL_FORMATS:
        errors.append(f"{path}/format: must be one of {', '.join(POSE_LABEL_FORMATS)}")
    if not isinstance(value.get("coordinate_frame"), str) or not value.get("coordinate_frame"):
        errors.append(f"{path}/coordinate_frame: must be a non-empty string")
    return errors


def _segmentation_mask_errors(path: str, value: Any) -> list[str]:
    if not isinstance(value, dict):
        return [f"{path}: must be an object"]

    errors: list[str] = []
    errors.extend(_unknown_fields(path, value, {"format", "path"}))
    if value.get("format") not in SEGMENTATION_MASK_FORMATS:
        errors.append(f"{path}/format: must be one of {', '.join(SEGMENTATION_MASK_FORMATS)}")
    mask_path = value.get("path")
    if not isinstance(mask_path, str) or not mask_path:
        errors.append(f"{path}/path: must be a non-empty string")
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

        annotations = source.get("annotations")
        if not isinstance(annotations, dict):
            continue
        masks = annotations.get("segmentation_masks")
        if not isinstance(masks, dict):
            continue
        raw_mask_path = masks.get("path")
        if not isinstance(raw_mask_path, str) or not raw_mask_path:
            continue
        errors.extend(
            _safe_existing_file_errors(
                f"sources/{index}/annotations/segmentation_masks/path",
                raw_mask_path,
                base_dir,
            )
        )
    return errors


def _safe_existing_file_errors(path: str, value: str, base_dir: Path) -> list[str]:
    relative_path = Path(value)
    if _is_unsafe_relative_path(relative_path):
        return [f"{path}: must be relative and stay within the manifest directory"]

    candidate = base_dir / relative_path
    try:
        candidate.resolve(strict=False).relative_to(base_dir)
    except ValueError:
        return [f"{path}: must be relative and stay within the manifest directory"]

    if not candidate.is_file():
        return [f"{path}: file does not exist: {value}"]
    return []


def _content_gaps(sources: list[dict[str, Any]], base_dir: Path) -> list[str]:
    placeholder_json = 0
    placeholder_masks = 0
    for source in sources:
        raw_path = source.get("path")
        if isinstance(raw_path, str):
            target = base_dir / raw_path
            if target.suffix.lower() == ".json" and _is_placeholder_json(target):
                placeholder_json += 1
        annotations = source.get("annotations")
        masks = annotations.get("segmentation_masks") if isinstance(annotations, dict) else None
        raw_mask_path = masks.get("path") if isinstance(masks, dict) else None
        if isinstance(raw_mask_path, str):
            mask_path = base_dir / raw_mask_path
            if mask_path.is_file() and not _has_png_signature(mask_path):
                placeholder_masks += 1
    gaps: list[str] = []
    if placeholder_json:
        gaps.append(f"sources contain placeholder JSON only: {placeholder_json}")
    if placeholder_masks:
        gaps.append(f"segmentation masks contain placeholder bytes: {placeholder_masks}")
    return gaps


def _is_placeholder_json(path: Path) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return payload == {} or payload == []


def _has_png_signature(path: Path) -> bool:
    try:
        return path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    except OSError:
        return False


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
    if not _is_string_list(value):
        errors.append(f"{path}: must be an array of non-empty strings")


def _is_string_list(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) and item for item in value)


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
    parser = argparse.ArgumentParser(description="Validate a racket data source manifest.")
    parser.add_argument("manifest", type=Path, help="Path to a racket dataset source manifest JSON file.")
    args = parser.parse_args(argv)

    try:
        summary = validate_manifest(args.manifest)
    except ValueError as exc:
        print("ERROR: racket dataset manifest failed validation:", file=sys.stderr)
        for line in str(exc).splitlines():
            print(f"- {line}", file=sys.stderr)
        return 1

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

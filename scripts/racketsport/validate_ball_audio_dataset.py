#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
SOURCE_TYPES = ("pop_audio", "roboflow_ball_xy", "tracknet_ball_xy")
BALL_SOURCE_TYPES = {"roboflow_ball_xy", "tracknet_ball_xy"}
CLASS_LABELS = ("ball_occluded", "ball_visible", "negative", "pop")
BALL_CLASS_LABELS = {"ball_occluded", "ball_visible"}
AUDIO_CLASS_LABELS = {"negative", "pop"}
LABEL_FORMATS = ("roboflow_xy", "tracknet_xy")
SOURCE_LABEL_FORMAT = {
    "roboflow_ball_xy": "roboflow_xy",
    "tracknet_ball_xy": "tracknet_xy",
}
SPLITS = ("test", "train", "val")
KEY_SPLITS = ("val", "test")
KEY_SOURCE_TYPES = SOURCE_TYPES
KEY_CLASS_LABELS = CLASS_LABELS
KNOWN_AUGMENTATIONS = (
    "amplitude",
    "background_subtraction",
    "color_jitter",
    "copy_paste",
    "fps_sim",
    "h264_artifact",
    "jpeg_artifact",
    "mixup",
    "motion_blur",
    "noise_snr",
    "occlusion",
    "pitch_shift",
    "rir",
    "specaugment",
    "time_shift",
    "time_stretch",
)
TOP_LEVEL_FIELDS = {"schema_version", "entries", "dataset_id", "description", "created_at", "notes"}
ENTRY_FIELDS = {
    "id",
    "path",
    "split",
    "source_type",
    "source_name",
    "class_label",
    "label_format",
    "frame_rate",
    "sample_rate",
    "audio_format",
    "duration_ms",
    "visibility",
    "augmentations",
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
    content_gaps = _content_gaps(entries, manifest_dir) if not errors else []
    valid = not errors

    return {
        "schema_version": SCHEMA_VERSION,
        "manifest": str(manifest_path),
        "valid": valid,
        "dataset_ready": valid and not coverage_gaps and not content_gaps,
        "entry_count": len(entries) if valid else len(entries),
        "coverage_counts": coverage_counts,
        "coverage_gaps": coverage_gaps,
        "content_gaps": content_gaps,
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

    for field in ("id", "path", "split", "source_type", "class_label"):
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
        errors.append(f"{prefix}/source_type: must be one of {', '.join(SOURCE_TYPES)}")

    split = entry.get("split")
    if split not in SPLITS:
        errors.append(f"{prefix}/split: must be one of {', '.join(SPLITS)}")

    class_label = entry.get("class_label")
    if class_label not in CLASS_LABELS:
        errors.append(f"{prefix}/class_label: must be one of {', '.join(CLASS_LABELS)}")
    elif source_type in BALL_SOURCE_TYPES and class_label not in BALL_CLASS_LABELS:
        errors.append(f"{prefix}/class_label: ball sources must use ball_occluded or ball_visible")
    elif source_type == "pop_audio" and class_label not in AUDIO_CLASS_LABELS:
        errors.append(f"{prefix}/class_label: pop_audio sources must use negative or pop")

    label_format = entry.get("label_format")
    expected_label_format = SOURCE_LABEL_FORMAT.get(source_type)
    if expected_label_format is not None:
        if "label_format" not in entry:
            errors.append(f"{prefix}/label_format: required for {source_type} sources")
        elif label_format != expected_label_format:
            errors.append(f"{prefix}/label_format: {source_type} sources must use {expected_label_format}")
    elif "label_format" in entry and label_format not in LABEL_FORMATS:
        errors.append(f"{prefix}/label_format: must be one of {', '.join(LABEL_FORMATS)}")

    path_value = entry.get("path")
    if isinstance(path_value, str) and path_value:
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
    if source_type == "pop_audio":
        if "sample_rate" not in entry:
            errors.append(f"{prefix}/sample_rate: required for pop_audio sources")
        elif sample_rate != 44100:
            errors.append(f"{prefix}/sample_rate: pop_audio sources must be 44100 Hz")
    if "audio_format" in entry and entry["audio_format"] != "wav":
        errors.append(f"{prefix}/audio_format: must be wav")
    if "duration_ms" in entry and not _is_positive_number(entry["duration_ms"]):
        errors.append(f"{prefix}/duration_ms: must be a positive number")
    if "visibility" in entry and not _is_visibility_value(entry["visibility"]):
        errors.append(f"{prefix}/visibility: must be 0, 1, or 2")
    if "source_name" in entry and (not isinstance(entry["source_name"], str) or not entry["source_name"]):
        errors.append(f"{prefix}/source_name: must be a non-empty string")
    if "augmentations" in entry:
        errors.extend(_augmentation_errors(prefix, entry["augmentations"]))

    if "notes" in entry and not isinstance(entry["notes"], str):
        errors.append(f"{prefix}/notes: must be a string")

    return errors


def _coverage_counts(entries: list[dict[str, Any]]) -> dict[str, Any]:
    source_counts = Counter(entry.get("source_type") for entry in entries if entry.get("source_type") in SOURCE_TYPES)
    class_counts = Counter(entry.get("class_label") for entry in entries if entry.get("class_label") in CLASS_LABELS)
    split_counts = Counter(entry.get("split") for entry in entries if entry.get("split") in SPLITS)
    by_split_source: dict[str, dict[str, int]] = {}
    by_split_class: dict[str, dict[str, int]] = {}
    for entry in entries:
        split = entry.get("split")
        source_type = entry.get("source_type")
        class_label = entry.get("class_label")
        if split in SPLITS and source_type in SOURCE_TYPES:
            by_split_source.setdefault(split, {})
            by_split_source[split][source_type] = by_split_source[split].get(source_type, 0) + 1
        if split in SPLITS and class_label in CLASS_LABELS:
            by_split_class.setdefault(split, {})
            by_split_class[split][class_label] = by_split_class[split].get(class_label, 0) + 1

    return {
        "source_type": {source_type: source_counts.get(source_type, 0) for source_type in SOURCE_TYPES},
        "class_label": {class_label: class_counts.get(class_label, 0) for class_label in CLASS_LABELS},
        "split": {split: split_counts.get(split, 0) for split in SPLITS},
        "by_split_source_type": {
            split: dict(sorted(counts.items())) for split, counts in sorted(by_split_source.items())
        },
        "by_split_class_label": {
            split: dict(sorted(counts.items())) for split, counts in sorted(by_split_class.items())
        },
    }


def _coverage_gaps(coverage_counts: dict[str, Any]) -> list[str]:
    gaps: list[str] = []
    split_counts = coverage_counts["split"]
    for split in KEY_SPLITS:
        if split_counts.get(split, 0) == 0:
            gaps.append(f"missing {split} entries")

    source_counts = coverage_counts["source_type"]
    missing_sources = [source_type for source_type in KEY_SOURCE_TYPES if source_counts.get(source_type, 0) == 0]
    if missing_sources:
        gaps.append(f"missing source types: {', '.join(missing_sources)}")

    class_counts = coverage_counts["class_label"]
    missing_classes = [class_label for class_label in KEY_CLASS_LABELS if class_counts.get(class_label, 0) == 0]
    if missing_classes:
        gaps.append(f"missing key classes: {', '.join(missing_classes)}")
    return gaps


def _content_gaps(entries: list[dict[str, Any]], manifest_dir: Path) -> list[str]:
    placeholder_ball_json = 0
    invalid_wav = 0
    for entry in entries:
        path_value = entry.get("path")
        if not isinstance(path_value, str):
            continue
        target = _resolve_safe_relative_path(path_value, manifest_dir)
        if target is None:
            continue
        source_type = entry.get("source_type")
        if source_type in BALL_SOURCE_TYPES and target.suffix.lower() == ".json" and _is_placeholder_json(target):
            placeholder_ball_json += 1
        if source_type == "pop_audio" and not _is_wav_container(target):
            invalid_wav += 1
    gaps: list[str] = []
    if placeholder_ball_json:
        gaps.append(f"ball labels contain placeholder JSON only: {placeholder_ball_json}")
    if invalid_wav:
        gaps.append(f"audio files are not WAV containers: {invalid_wav}")
    return gaps


def _is_placeholder_json(path: Path) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return payload == {} or payload == []


def _is_wav_container(path: Path) -> bool:
    try:
        header = path.read_bytes()[:12]
    except OSError:
        return False
    return len(header) == 12 and header[:4] == b"RIFF" and header[8:12] == b"WAVE"


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


def _is_visibility_value(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value in {0, 1, 2}


def _augmentation_errors(prefix: str, value: Any) -> list[str]:
    if not isinstance(value, list) or not value:
        return [f"{prefix}/augmentations: must be a non-empty array of known augmentation names"]
    errors: list[str] = []
    for index, item in enumerate(value):
        if item not in KNOWN_AUGMENTATIONS:
            errors.append(
                f"{prefix}/augmentations/{index}: must be one of {', '.join(KNOWN_AUGMENTATIONS)}"
            )
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a ball-track and pop-audio dataset manifest.")
    parser.add_argument("manifest", type=Path, help="Path to a dataset manifest JSON file.")
    args = parser.parse_args(argv)

    summary = validate_manifest(args.manifest)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

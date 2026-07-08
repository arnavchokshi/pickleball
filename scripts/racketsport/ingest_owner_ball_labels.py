#!/usr/bin/env python3
"""Ingest owner CVAT image exports into the sparse BALL reviewed corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
from xml.etree import ElementTree

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.racketsport.train_ball_stage2 import (  # noqa: E402
    load_cvat_annotations_from_export_clip,
    sparse_tracknet_labels_from_annotations,
)
from threed.racketsport.cvat_video import write_cvat_video_annotations  # noqa: E402
from threed.racketsport.schemas import (  # noqa: E402
    BALL_VISIBILITY_LEVELS,
    BallVisibilityLevel,
    CvatVideoAnnotationSummary,
    CvatVideoAnnotations,
    CvatVideoBox,
    CvatVideoFrame,
    CvatVideoTask,
    CvatVideoTrackSummary,
)


ARTIFACT_TYPE = "racketsport_owner_ball_label_ingest"
DEFAULT_PROTECTED_PATTERNS = (
    "pwxNwFfYQlQ",
    "vQhtz8l6VqU",
    "outdoor_webcam_iynbd",
    "indoor_doubles_fwuks",
)
DEFAULT_SOURCE_CLASSES = {
    "73VurrTKCZ8": "outdoor_day_multicam",
    "Ezz6HDNHlnk": "outdoor_night_fenced",
    "HyUqT7zFiwk": "indoor_court_level",
    "wBu8bC4OfUY": "outdoor_night_tennis_overlay",
    "_L0HVmAlCQI": "outdoor_night_tennis_overlay",
    "zwCtH_i1_S4": "outdoor_day_broadcast_overlay",
}


class OwnerBallLabelIngestError(ValueError):
    """Raised for invalid owner label ingest inputs."""


class ProtectedPatternError(OwnerBallLabelIngestError):
    """Raised when an owner export row maps to protected held-out data."""


@dataclass(frozen=True)
class ImageRow:
    image_id: int
    image_name: str
    source_id: str
    clip_id: str
    frame_index: int
    disagreement_type: str
    width: int
    height: int
    boxes: tuple[CvatVideoBox, ...]

    @property
    def row_key(self) -> str:
        return f"{self.clip_id}:{self.frame_index:06d}"


def build_reviewed_corpus(
    *,
    base_cvat_export_root: str | Path,
    export_zips: Sequence[str | Path],
    labelpack_manifest: str | Path,
    out_root: str | Path,
    protected_patterns: Sequence[str] = DEFAULT_PROTECTED_PATTERNS,
) -> dict[str, Any]:
    """Build a new sparse reviewed corpus from prior harvest labels plus owner exports."""

    base_root = Path(base_cvat_export_root)
    out = Path(out_root)
    reviewed_root = out / "reviewed_corpus"
    if not base_root.is_dir():
        raise FileNotFoundError(f"missing base CVAT export root: {base_root}")
    if not export_zips:
        raise OwnerBallLabelIngestError("at least one owner --export-zip is required")

    labelpack = _load_json(Path(labelpack_manifest))
    source_classes = _source_classes(labelpack)
    base_payloads = _load_base_payloads(base_root)
    payloads: dict[str, dict[str, Any]] = {}
    for clip_id, annotations in base_payloads.items():
        payload = _payload_to_dict(annotations)
        # The CVAT zip importer uses a temp extraction path as source_path.
        # Preserve deterministic corpus bytes by recording the stable source dir.
        payload["source_path"] = str(base_root / clip_id)
        payloads[clip_id] = payload
    base_reviewed_row_count = sum(_reviewed_row_count(payload) for payload in payloads.values())
    base_counts = _counts_for_payloads(payloads, source_classes=source_classes)

    protected_matches: list[dict[str, str]] = []
    zip_reports: list[dict[str, Any]] = []
    new_row_keys: set[str] = set()
    skipped_rows: list[dict[str, Any]] = []
    added_rows: list[ImageRow] = []

    for zip_path_raw in export_zips:
        zip_path = Path(zip_path_raw)
        rows, zip_report = _parse_cvat_images_zip(zip_path)
        protected_matches.extend(_protected_matches(rows, protected_patterns=protected_patterns))
        if protected_matches:
            raise ProtectedPatternError(
                "protected held-out pattern matched owner export rows: "
                + ", ".join(sorted({match["pattern"] for match in protected_matches}))
            )
        _reconcile_labelpack_session(zip_report, rows, labelpack)
        skip_counts: Counter[str] = Counter()
        added_count = 0
        for row in rows:
            if len(row.boxes) > 1:
                _record_skip(skipped_rows, row, "multiple_ball_boxes", zip_path=zip_path)
                skip_counts["multiple_ball_boxes"] += 1
                continue
            payload = payloads.get(row.clip_id)
            if payload is None:
                payload = _new_clip_payload(row, source_path=str(zip_path))
                payloads[row.clip_id] = payload
            existing_reviewed = set(payload.get("reviewed_frame_indices") or [])
            if row.frame_index in existing_reviewed:
                existing_sig = _reviewed_signature(payload, row.frame_index)
                row_sig = _row_signature(row)
                if existing_sig == row_sig:
                    reason = "duplicate_existing_reviewed_same_absent" if row_sig[0] == "absent" else "duplicate_existing_reviewed_same_box"
                else:
                    reason = "duplicate_existing_reviewed_conflict"
                _record_skip(skipped_rows, row, reason, zip_path=zip_path)
                skip_counts[reason] += 1
                continue
            _merge_row(payload, row, source_path=str(zip_path))
            new_row_keys.add(row.row_key)
            added_rows.append(row)
            added_count += 1
        zip_report["added_reviewed_rows"] = added_count
        zip_report["skip_reason_counts"] = dict(sorted(skip_counts.items()))
        zip_report["skipped_row_count"] = sum(skip_counts.values())
        zip_reports.append(zip_report)

    for payload in payloads.values():
        _finalize_payload(payload)

    _write_corpus(reviewed_root, payloads)
    counts = _counts_for_payloads(payloads, source_classes=source_classes)
    new_counts = _counts_for_payloads(payloads, source_classes=source_classes, only_row_keys=new_row_keys)
    loso_manifest = _write_loso_manifest(out / "loso_fold_manifest.json", payloads, source_classes=source_classes)
    md5_manifest = _write_md5_manifest(out / "corpus_md5_manifest.json", reviewed_root, counts)
    manifest_md5 = _file_md5(out / "corpus_md5_manifest.json")
    report = {
        "schema_version": 1,
        "artifact_type": ARTIFACT_TYPE,
        "status": "reviewed_corpus_rebuilt",
        "ball_verified": False,
        "heldout_touched": False,
        "promotion_claimed": False,
        "base_cvat_export_root": str(base_root),
        "labelpack_manifest": str(labelpack_manifest),
        "reviewed_corpus_root": str(reviewed_root),
        "manifest_md5": manifest_md5,
        "md5_manifest": md5_manifest,
        "protected_scan": {
            "status": "NO_MATCH",
            "patterns": list(protected_patterns),
            "match_count": 0,
            "matches": [],
        },
        "zip_reports": zip_reports,
        "skipped_rows": skipped_rows,
        "counts": counts,
        "base_counts": base_counts,
        "new_counts": new_counts,
        "fold_manifest_path": str(out / "loso_fold_manifest.json"),
        "fold_manifest": loso_manifest,
        "summary": {
            "base_reviewed_row_count": base_reviewed_row_count,
            "new_reviewed_row_count": new_counts["totals"]["reviewed_row_count"],
            "new_positive_row_count": new_counts["totals"]["positive_row_count"],
            "new_negative_row_count": new_counts["totals"]["negative_row_count"],
            "total_reviewed_row_count": counts["totals"]["reviewed_row_count"],
            "total_positive_row_count": counts["totals"]["positive_row_count"],
            "total_negative_row_count": counts["totals"]["negative_row_count"],
            "throughput_bar_min_frames": 10_000,
            "throughput_bar_max_frames": 20_000,
            "owner_label_rate_frames_per_hour": 240,
        },
        "limitations": [
            "This corpus is internal-val/training-side data plumbing only.",
            "Protected Outdoor/Indoor held-out labels are not read or written.",
            "Rows with multiple ball boxes are skipped as ambiguous for the single-ball stage-2 consumer.",
            "Existing reviewed rows are not overwritten; conflicts are reported as skipped rows.",
        ],
    }
    _write_json(out / "report.json", report)
    return report


def _load_base_payloads(base_root: Path) -> dict[str, CvatVideoAnnotations]:
    payloads: dict[str, CvatVideoAnnotations] = {}
    for clip_dir in sorted(path for path in base_root.iterdir() if path.is_dir()):
        annotations = load_cvat_annotations_from_export_clip(clip_dir)
        if annotations.clip_id in payloads:
            raise OwnerBallLabelIngestError(f"duplicate base clip_id: {annotations.clip_id}")
        payloads[annotations.clip_id] = annotations
    if not payloads:
        raise OwnerBallLabelIngestError(f"no base CVAT clip dirs found under {base_root}")
    return payloads


def _parse_cvat_images_zip(zip_path: Path) -> tuple[list[ImageRow], dict[str, Any]]:
    if not zip_path.is_file():
        raise FileNotFoundError(f"missing owner CVAT export zip: {zip_path}")
    with zipfile.ZipFile(zip_path) as archive:
        try:
            raw_xml = archive.read("annotations.xml")
        except KeyError as exc:
            raise OwnerBallLabelIngestError(f"{zip_path} is missing annotations.xml") from exc
    root = ElementTree.fromstring(raw_xml)
    if _text(root.find("version")) != "1.1":
        raise OwnerBallLabelIngestError(f"{zip_path} annotations.xml must be CVAT images version 1.1")
    job = root.find("./meta/job")
    if job is None:
        raise OwnerBallLabelIngestError(f"{zip_path} annotations.xml missing meta/job")
    job_size = _optional_int(_text(job.find("size"))) or 0
    rows: list[ImageRow] = []
    box_count = 0
    box_count_by_image: Counter[int] = Counter()
    visibility_levels: Counter[str] = Counter()
    center_conventions: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    clip_counts: Counter[str] = Counter()
    disagreement_counts: Counter[str] = Counter()
    for image in root.findall("image"):
        row = _parse_image_row(image)
        rows.append(row)
        source_counts[row.source_id] += 1
        clip_counts[row.clip_id] += 1
        disagreement_counts[row.disagreement_type] += 1
        box_count += len(row.boxes)
        box_count_by_image[len(row.boxes)] += 1
        for box in row.boxes:
            visibility_levels[box.visibility_level or "none"] += 1
            center_conventions[box.center_convention or "unknown"] += 1
    report = {
        "zip_path": str(zip_path),
        "session_id": _session_id_from_zip(zip_path),
        "job_id": _optional_int(_text(job.find("id"))),
        "job_size": job_size,
        "job_mode": _text(job.find("mode")) or None,
        "image_count": len(rows),
        "ball_box_count": box_count,
        "zero_box_image_count": box_count_by_image[0],
        "single_box_image_count": box_count_by_image[1],
        "multi_box_image_count": sum(count for box_num, count in box_count_by_image.items() if box_num > 1),
        "source_counts": dict(sorted(source_counts.items())),
        "clip_counts": dict(sorted(clip_counts.items())),
        "disagreement_type_counts": dict(sorted(disagreement_counts.items())),
        "visibility_level_counts": dict(sorted(visibility_levels.items())),
        "center_convention_counts": dict(sorted(center_conventions.items())),
        "schema_checks": {
            "version": "1.1",
            "job_size_matches_image_count": job_size == len(rows),
        },
    }
    return rows, report


def _parse_image_row(image: ElementTree.Element) -> ImageRow:
    image_name = _required_attr(image, "name")
    source_id, clip_id, frame_index, disagreement_type = _parse_image_name(image_name)
    image_id = _int_attr(image, "id", minimum=0)
    width = _int_attr(image, "width", minimum=1)
    height = _int_attr(image, "height", minimum=1)
    boxes = tuple(
        _parse_image_box(box, frame_index=frame_index, track_id=0)
        for box in image.findall("box")
        if _required_attr(box, "label").strip().lower() == "ball"
    )
    return ImageRow(
        image_id=image_id,
        image_name=image_name,
        source_id=source_id,
        clip_id=clip_id,
        frame_index=frame_index,
        disagreement_type=disagreement_type,
        width=width,
        height=height,
        boxes=boxes,
    )


def _parse_image_name(name: str) -> tuple[str, str, int, str]:
    stem = Path(name).stem
    parts = stem.split("__")
    if len(parts) < 5:
        raise OwnerBallLabelIngestError(f"unexpected owner label image name: {name}")
    source_id = parts[1]
    clip_id = parts[2]
    frame_token = parts[3]
    if not re.fullmatch(r"f\d+", frame_token):
        raise OwnerBallLabelIngestError(f"unexpected owner label frame token in {name}: {frame_token}")
    disagreement_type = parts[4].replace("_", "-")
    return source_id, clip_id, int(frame_token[1:]), disagreement_type


def _parse_image_box(box: ElementTree.Element, *, frame_index: int, track_id: int) -> CvatVideoBox:
    visibility_level = _normalize_visibility_level(_shape_attributes(box).get("visibility_level"))
    if visibility_level is None:
        raise OwnerBallLabelIngestError(f"ball box at frame {frame_index} missing visibility_level")
    x1 = _float_attr(box, "xtl")
    y1 = _float_attr(box, "ytl")
    x2 = _float_attr(box, "xbr")
    y2 = _float_attr(box, "ybr")
    blur_fields = _shape_ball_blur_fields(box)
    return CvatVideoBox(
        track_id=track_id,
        label="ball",
        frame_index=frame_index,
        bbox_xyxy=(x1, y1, x2, y2),
        bbox_xywh=(x1, y1, x2 - x1, y2 - y1),
        keyframe=True,
        occluded=_boolish(box.attrib.get("occluded")),
        source=box.attrib.get("source") or "file",
        visibility_level=visibility_level,
        **blur_fields,
    )


def _shape_ball_blur_fields(node: ElementTree.Element) -> dict[str, Any]:
    attributes = _shape_attributes(node)
    fields: dict[str, Any] = {}
    for name in ("center_convention", "blur_label_quality"):
        value = attributes.get(name)
        if value:
            fields[name] = value
    for name in ("blur_angle_deg", "blur_length_px", "blur_width_px"):
        value = attributes.get(name)
        if value:
            fields[name] = float(value)
    return fields


def _new_clip_payload(row: ImageRow, *, source_path: str) -> dict[str, Any]:
    size = row.frame_index + 1
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_cvat_video_annotations",
        "clip_id": row.clip_id,
        "source_format": "cvat_images_1_1",
        "source_path": source_path,
        "reviewed_frame_indices": [],
        "reviewed_frame_indices_source": "explicit",
        "task": {
            "task_id": None,
            "name": row.clip_id,
            "size": size,
            "mode": "annotation",
            "start_frame": 0,
            "stop_frame": size - 1,
            "frame_filter": None,
            "original_size": [row.width, row.height],
            "source": f"{row.clip_id}.mp4",
            "dumped": None,
        },
        "frames": [{"frame_index": index, "boxes": [], "visibility_levels_by_label": {}} for index in range(size)],
        "tracks": [],
        "summary": {
            "frame_count": size,
            "visible_box_count": 0,
            "outside_box_count": 0,
            "labels": ["ball"],
            "track_count_by_label": {},
            "visible_box_count_by_label": {},
        },
    }


def _merge_row(payload: dict[str, Any], row: ImageRow, *, source_path: str) -> None:
    _ensure_frame(payload, row.frame_index, width=row.width, height=row.height)
    payload["source_format"] = "cvat_images_1_1"
    source_paths = str(payload.get("source_path") or "")
    if source_path not in source_paths.split(" | "):
        payload["source_path"] = f"{source_paths} | {source_path}" if source_paths else source_path
    frame = payload["frames"][row.frame_index]
    frame["boxes"] = [
        box for box in frame.get("boxes", [])
        if str(box.get("label", "")).strip().lower() != "ball"
    ]
    frame["boxes"].extend([box.model_dump(mode="json") for box in row.boxes])
    visibility_levels = dict(frame.get("visibility_levels_by_label") or {})
    if row.boxes:
        level = row.boxes[0].visibility_level
        if level is not None:
            visibility_levels["ball"] = level
    else:
        visibility_levels.pop("ball", None)
    frame["visibility_levels_by_label"] = visibility_levels
    reviewed = set(payload.get("reviewed_frame_indices") or [])
    reviewed.add(row.frame_index)
    payload["reviewed_frame_indices"] = sorted(reviewed)
    payload["reviewed_frame_indices_source"] = "explicit"


def _ensure_frame(payload: dict[str, Any], frame_index: int, *, width: int, height: int) -> None:
    frames = payload.setdefault("frames", [])
    while len(frames) <= frame_index:
        frames.append({"frame_index": len(frames), "boxes": [], "visibility_levels_by_label": {}})
    task = payload.setdefault("task", {})
    if int(task.get("size") or 0) <= frame_index:
        task["size"] = frame_index + 1
        task["stop_frame"] = frame_index
    task.setdefault("original_size", [width, height])


def _reviewed_signature(payload: Mapping[str, Any], frame_index: int) -> tuple[Any, ...]:
    frame = payload.get("frames", [])[frame_index]
    ball_boxes = [box for box in frame.get("boxes", []) if str(box.get("label", "")).strip().lower() == "ball"]
    level = (frame.get("visibility_levels_by_label") or {}).get("ball")
    if not ball_boxes:
        return ("absent", level)
    if len(ball_boxes) > 1:
        return ("multiple", len(ball_boxes))
    box = ball_boxes[0]
    return (
        "box",
        tuple(round(float(value), 3) for value in box["bbox_xyxy"]),
        box.get("visibility_level") or level,
    )


def _row_signature(row: ImageRow) -> tuple[Any, ...]:
    if not row.boxes:
        return ("absent", None)
    if len(row.boxes) > 1:
        return ("multiple", len(row.boxes))
    box = row.boxes[0]
    return (
        "box",
        tuple(round(float(value), 3) for value in box.bbox_xyxy),
        box.visibility_level,
    )


def _record_skip(skipped_rows: list[dict[str, Any]], row: ImageRow, reason: str, *, zip_path: Path) -> None:
    skipped_rows.append(
        {
            "zip_path": str(zip_path),
            "reason": reason,
            "image_name": row.image_name,
            "source_id": row.source_id,
            "clip_id": row.clip_id,
            "frame_index": row.frame_index,
            "ball_box_count": len(row.boxes),
        }
    )


def _finalize_payload(payload: dict[str, Any]) -> None:
    frames = payload.get("frames", [])
    for index, frame in enumerate(frames):
        frame["boxes"] = sorted(
            [dict(box) for box in frame.get("boxes", [])],
            key=_box_sort_key,
        )
        frame["visibility_levels_by_label"] = dict(
            sorted((frame.get("visibility_levels_by_label") or {}).items())
        )
        frame["frame_index"] = index
    visible_by_label: Counter[str] = Counter()
    track_stats: dict[tuple[str, int], dict[str, Any]] = {}
    for frame in frames:
        frame_index = int(frame["frame_index"])
        for box in frame.get("boxes", []):
            label = str(box["label"])
            track_id = int(box["track_id"])
            visible_by_label[label] += 1
            stats = track_stats.setdefault(
                (label, track_id),
                {
                    "track_id": track_id,
                    "label": label,
                    "visible_box_count": 0,
                    "outside_box_count": 0,
                    "keyframe_count": 0,
                    "first_visible_frame": frame_index,
                    "last_visible_frame": frame_index,
                },
            )
            stats["visible_box_count"] += 1
            if box.get("keyframe"):
                stats["keyframe_count"] += 1
            stats["first_visible_frame"] = min(int(stats["first_visible_frame"]), frame_index)
            stats["last_visible_frame"] = max(int(stats["last_visible_frame"]), frame_index)
    labels = sorted(visible_by_label) or ["ball"]
    payload["tracks"] = [
        CvatVideoTrackSummary(**stats).model_dump(mode="json")
        for _, stats in sorted(track_stats.items(), key=lambda item: (item[0][0], item[0][1]))
    ]
    payload["summary"] = CvatVideoAnnotationSummary(
        frame_count=len(frames),
        visible_box_count=sum(visible_by_label.values()),
        outside_box_count=0,
        labels=labels,
        track_count_by_label={
            label: sum(1 for track_label, _ in track_stats if track_label == label)
            for label in labels
        },
        visible_box_count_by_label={label: visible_by_label[label] for label in labels if visible_by_label[label]},
    ).model_dump(mode="json")
    payload["task"]["size"] = len(frames)
    payload["task"]["stop_frame"] = max(0, len(frames) - 1)
    payload["reviewed_frame_indices"] = sorted(
        int(frame_index) for frame_index in payload.get("reviewed_frame_indices", [])
    )


def _box_sort_key(box: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(box.get("label", "")),
        int(box.get("track_id") or 0),
        int(box.get("frame_index") or 0),
        tuple(round(float(value), 6) for value in box.get("bbox_xyxy", [])),
        str(box.get("visibility_level") or ""),
        str(box.get("source") or ""),
    )


def _write_corpus(reviewed_root: Path, payloads: Mapping[str, Mapping[str, Any]]) -> None:
    reviewed_root.mkdir(parents=True, exist_ok=True)
    for clip_id, payload in sorted(payloads.items()):
        annotations = CvatVideoAnnotations.model_validate(payload)
        write_cvat_video_annotations(reviewed_root / clip_id / "reviewed_boxes.json", annotations)


def _counts_for_payloads(
    payloads: Mapping[str, Mapping[str, Any]],
    *,
    source_classes: Mapping[str, str],
    only_row_keys: set[str] | None = None,
) -> dict[str, Any]:
    per_source: dict[str, Counter[str]] = defaultdict(Counter)
    per_clip: dict[str, Counter[str]] = defaultdict(Counter)
    class_counts: Counter[str] = Counter()
    visibility_counts: Counter[str] = Counter()
    totals: Counter[str] = Counter()
    for clip_id, payload in sorted(payloads.items()):
        annotations = CvatVideoAnnotations.model_validate(payload)
        labels = sparse_tracknet_labels_from_annotations(annotations)
        source_id = _source_id_from_clip(clip_id)
        source_class = source_classes.get(source_id, "unknown")
        for label in labels:
            row_key = f"{clip_id}:{int(label.frame):06d}"
            if only_row_keys is not None and row_key not in only_row_keys:
                continue
            present_key = "positive" if int(label.visibility) == 1 else "negative"
            visibility_key = label.visibility_level or "none"
            totals["reviewed_row_count"] += 1
            totals[f"{present_key}_row_count"] += 1
            class_counts[present_key] += 1
            visibility_counts[visibility_key] += 1
            per_source[source_id]["reviewed_row_count"] += 1
            per_source[source_id][f"{present_key}_row_count"] += 1
            per_source[source_id][f"visibility:{visibility_key}"] += 1
            per_source[source_id]["source_class:" + source_class] += 1
            per_clip[clip_id]["reviewed_row_count"] += 1
            per_clip[clip_id][f"{present_key}_row_count"] += 1
            per_clip[clip_id][f"visibility:{visibility_key}"] += 1
    return {
        "totals": {
            "reviewed_row_count": totals["reviewed_row_count"],
            "positive_row_count": totals["positive_row_count"],
            "negative_row_count": totals["negative_row_count"],
        },
        "per_source": {key: dict(value) for key, value in sorted(per_source.items())},
        "per_clip": {key: dict(value) for key, value in sorted(per_clip.items())},
        "per_class": dict(sorted(class_counts.items())),
        "visibility_level_counts": dict(sorted(visibility_counts.items())),
    }


def _write_loso_manifest(
    path: Path,
    payloads: Mapping[str, Mapping[str, Any]],
    *,
    source_classes: Mapping[str, str],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for clip_id, payload in sorted(payloads.items()):
        annotations = CvatVideoAnnotations.model_validate(payload)
        source_id = _source_id_from_clip(clip_id)
        source_class = source_classes.get(source_id, "unknown")
        for label in sparse_tracknet_labels_from_annotations(annotations):
            rows.append(
                {
                    "row_key": f"{clip_id}:{int(label.frame):06d}",
                    "clip_id": clip_id,
                    "source_id": source_id,
                    "source_class": source_class,
                    "frame_index": int(label.frame),
                    "ball_present": bool(label.visibility == 1),
                    "visibility_level": label.visibility_level or "none",
                }
            )
    all_keys = {row["row_key"] for row in rows}
    folds = []
    for source_id in sorted({row["source_id"] for row in rows}):
        val_keys = sorted(row["row_key"] for row in rows if row["source_id"] == source_id)
        train_keys = sorted(all_keys - set(val_keys))
        source_class = source_classes.get(source_id, "unknown")
        folds.append(
            {
                "source_id": source_id,
                "source_class": source_class,
                "is_outdoor_fold": "outdoor" in source_class,
                "train_row_count": len(train_keys),
                "val_row_count": len(val_keys),
                "train_row_keys": train_keys,
                "val_row_keys": val_keys,
                "disjoint": set(train_keys).isdisjoint(val_keys),
            }
        )
    manifest = {
        "schema_version": 1,
        "artifact_type": "racketsport_ball_loso_fold_manifest",
        "status": "internal_val_only",
        "heldout_touched": False,
        "row_count": len(rows),
        "source_count": len(folds),
        "folds": folds,
        "fold_disjointness": {
            "all_disjoint": all(fold["disjoint"] for fold in folds),
            "checked_fold_count": len(folds),
        },
        "rows": rows,
    }
    _write_json(path, manifest)
    return manifest


def _write_md5_manifest(path: Path, reviewed_root: Path, counts: Mapping[str, Any]) -> dict[str, Any]:
    files = [
        {
            "relative_path": file.relative_to(reviewed_root.parent).as_posix(),
            "md5": _file_md5(file),
            "bytes": file.stat().st_size,
        }
        for file in sorted(reviewed_root.glob("*/reviewed_boxes.json"))
    ]
    manifest = {
        "schema_version": 1,
        "artifact_type": "racketsport_reviewed_corpus_md5_manifest",
        "files": files,
        "counts": counts,
    }
    _write_json(path, manifest)
    return manifest


def _reconcile_labelpack_session(report: dict[str, Any], rows: Sequence[ImageRow], labelpack: Mapping[str, Any]) -> None:
    session_id = report.get("session_id")
    sessions = {
        session.get("session_id"): session
        for session in labelpack.get("ball_sessions", [])
        if isinstance(session, Mapping)
    }
    session = sessions.get(session_id)
    if session is None:
        report["labelpack_reconcile"] = {"status": "missing_session", "session_id": session_id}
        return
    actual_clip_counts = Counter(row.clip_id for row in rows)
    actual_type_counts = Counter(row.disagreement_type for row in rows)
    expected_type_counts = {
        str(key).replace("_", "-"): int(value)
        for key, value in dict(session.get("disagreement_type_counts") or {}).items()
    }
    report["labelpack_reconcile"] = {
        "status": "PASS"
        if int(session.get("frame_count") or -1) == len(rows)
        and dict(sorted(actual_clip_counts.items())) == dict(sorted((session.get("clip_counts") or {}).items()))
        and dict(sorted(actual_type_counts.items())) == dict(sorted(expected_type_counts.items()))
        else "FAIL",
        "expected_frame_count": session.get("frame_count"),
        "actual_frame_count": len(rows),
        "expected_clip_counts": session.get("clip_counts") or {},
        "actual_clip_counts": dict(sorted(actual_clip_counts.items())),
        "expected_disagreement_type_counts": expected_type_counts,
        "actual_disagreement_type_counts": dict(sorted(actual_type_counts.items())),
    }


def _protected_matches(rows: Sequence[ImageRow], *, protected_patterns: Sequence[str]) -> list[dict[str, str]]:
    matches: list[dict[str, str]] = []
    lowered_patterns = [(pattern, pattern.lower()) for pattern in protected_patterns]
    for row in rows:
        haystack = " ".join([row.image_name, row.source_id, row.clip_id]).lower()
        for pattern, lowered in lowered_patterns:
            if lowered in haystack:
                matches.append(
                    {
                        "pattern": pattern,
                        "image_name": row.image_name,
                        "source_id": row.source_id,
                        "clip_id": row.clip_id,
                    }
                )
    return matches


def _source_classes(labelpack: Mapping[str, Any]) -> dict[str, str]:
    classes = dict(DEFAULT_SOURCE_CLASSES)
    for session in labelpack.get("ball_sessions", []):
        if isinstance(session, Mapping):
            for source_id, source_class in dict(session.get("source_classes") or {}).items():
                classes[str(source_id)] = str(source_class)
    court = labelpack.get("court_kp_relabel")
    if isinstance(court, Mapping):
        for source_id, source_class in dict(court.get("source_classes") or {}).items():
            classes[str(source_id)] = str(source_class)
    return classes


def _reviewed_row_count(payload: Mapping[str, Any]) -> int:
    reviewed = payload.get("reviewed_frame_indices")
    if reviewed is not None:
        return len(reviewed)
    return len(payload.get("frames", []))


def _payload_to_dict(annotations: CvatVideoAnnotations) -> dict[str, Any]:
    return annotations.model_dump(mode="json")


def _source_id_from_clip(clip_id: str) -> str:
    if "_rally_" not in clip_id:
        return clip_id
    return clip_id.split("_rally_", 1)[0]


def _session_id_from_zip(path: Path) -> str | None:
    match = re.search(r"session[_-](\d+)", path.name)
    if not match:
        return None
    return f"ball_session_{int(match.group(1)):02d}"


def _normalize_visibility_level(value: str | None) -> BallVisibilityLevel | None:
    if value is None or not value.strip():
        return None
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized not in BALL_VISIBILITY_LEVELS:
        raise OwnerBallLabelIngestError(
            f"visibility_level must be one of {', '.join(BALL_VISIBILITY_LEVELS)}: {value}"
        )
    return normalized  # type: ignore[return-value]


def _shape_attributes(node: ElementTree.Element) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for attribute in node.findall("attribute"):
        name = attribute.attrib.get("name", "").strip()
        value = _text(attribute)
        if name and value:
            parsed[name] = value
    return parsed


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text(node: ElementTree.Element | None) -> str:
    if node is None or node.text is None:
        return ""
    return node.text.strip()


def _required_attr(node: ElementTree.Element, name: str) -> str:
    value = node.attrib.get(name, "").strip()
    if not value:
        raise OwnerBallLabelIngestError(f"missing required attribute: {name}")
    return value


def _int_attr(node: ElementTree.Element, name: str, *, minimum: int) -> int:
    value = _required_attr(node, name)
    number = _optional_int(value)
    if number is None or number < minimum:
        raise OwnerBallLabelIngestError(f"{name} must be an integer >= {minimum}: {value}")
    return number


def _float_attr(node: ElementTree.Element, name: str) -> float:
    value = _required_attr(node, name)
    try:
        return float(value)
    except ValueError as exc:
        raise OwnerBallLabelIngestError(f"{name} must be a float: {value}") from exc


def _optional_int(value: str) -> int | None:
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _boolish(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes"}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ingest owner CVAT images 1.1 BALL annotations into a new sparse reviewed corpus.",
    )
    parser.add_argument("--base-cvat-export-root", type=Path, required=True)
    parser.add_argument("--export-zip", type=Path, action="append", required=True)
    parser.add_argument("--labelpack-manifest", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        report = build_reviewed_corpus(
            base_cvat_export_root=args.base_cvat_export_root,
            export_zips=args.export_zip,
            labelpack_manifest=args.labelpack_manifest,
            out_root=args.out_root,
        )
    except Exception as exc:
        print(f"owner BALL label ingest failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": report["status"],
                "reviewed_corpus_root": report["reviewed_corpus_root"],
                "manifest_md5": report["manifest_md5"],
                "summary": report["summary"],
                "fold_manifest_path": report["fold_manifest_path"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

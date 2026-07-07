"""Build TrackNetV3 label artifacts from reviewed CVAT ball boxes."""

from __future__ import annotations

import csv
import io
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence

from .eval_guard import assert_not_training_on_eval_clip
from .schemas import (
    BALL_VISIBILITY_LEVELS,
    BALL_VISIBILITY_WBCE_WEIGHTS,
    LEGACY_BALL_VISIBILITY_MAPPING,
    BallVisibilityLevel,
    CvatVideoAnnotations,
    CvatVideoFrame,
    validate_artifact_file,
)


ARTIFACT_TYPE = "racketsport_ball_tracknet_cvat_dataset"
MANIFEST_JSON = "ball_tracknet_cvat_dataset_manifest.json"
MANIFEST_MD = "ball_tracknet_cvat_dataset_manifest.md"
TRACKNET_COLUMNS = ("Frame", "Visibility", "X", "Y")
CENTER_CONVENTION_VALUES = ("blur_midpoint", "disk_center", "unknown")
BLUR_LABEL_QUALITY_VALUES = ("absent", "clear", "unknown", "weak")
LEGACY_VISIBILITY_STATES = ("legacy_visible", "legacy_hidden")
SPLIT_ORDER = ("train", "val", "test")
TRAIN_AUGMENTATION_PROFILES: dict[str, tuple[dict[str, Any], ...]] = {
    "codec_motion_v1": (
        {
            "name": "jpeg_q55_brightness_1_06_contrast_1_08_color_0_94",
            "jpeg_quality": 55,
            "brightness": 1.06,
            "contrast": 1.08,
            "color": 0.94,
        },
        {
            "name": "horizontal_motion3_jpeg_q45_brightness_0_94_contrast_1_10",
            "motion_blur": "horizontal_3",
            "jpeg_quality": 45,
            "brightness": 0.94,
            "contrast": 1.10,
            "color": 0.98,
        },
        {
            "name": "vertical_motion3_jpeg_q60_brightness_1_10_color_0_90",
            "motion_blur": "vertical_3",
            "jpeg_quality": 60,
            "brightness": 1.10,
            "contrast": 1.02,
            "color": 0.90,
        },
    ),
}


@dataclass(frozen=True)
class TrackNetCvatLabel:
    frame: int
    visibility: int
    x: float
    y: float
    source: str
    center_convention: str | None = None
    blur_angle_deg: float | None = None
    blur_length_px: float | None = None
    blur_width_px: float | None = None
    blur_label_quality: str | None = None
    visibility_level: BallVisibilityLevel | None = None
    wbce_weight: int | None = None
    legacy_visibility_state: str | None = None


def dense_tracknet_labels_from_cvat(reviewed_boxes_path: str | Path) -> list[TrackNetCvatLabel]:
    """Return one TrackNet ``Frame,Visibility,X,Y`` row per reviewed CVAT frame."""

    annotations = _load_cvat_video_annotations(reviewed_boxes_path)
    return _dense_tracknet_labels_from_annotations(annotations)


def build_ball_tracknet_cvat_dataset(
    *,
    cvat_root: str | Path,
    yolo_manifest: str | Path,
    out_dir: str | Path,
    fps: float = 60.0,
    clips: Sequence[str] | None = None,
    materialize_frames: bool = False,
    video_paths: Mapping[str, str | Path] | None = None,
    hard_negative_plan: str | Path | None = None,
    hard_negative_context_frames: int = 0,
    hard_negative_repeat: int = 1,
    train_augmentation_profile: str | None = None,
    train_augmentation_repeat: int = 1,
) -> dict[str, Any]:
    """Write dense TrackNetV3 CSV labels plus JSON/Markdown manifest artifacts."""

    fps = _positive_float(fps, "fps")
    cvat_base = Path(cvat_root)
    yolo_manifest_path = Path(yolo_manifest)
    out = Path(out_dir)
    split_map = _split_map_from_yolo_manifest(yolo_manifest_path)
    selected_clips = _selected_clips(split_map, clips)
    # Eval-clip integrity gate (fail closed): this function writes the CSV/frame
    # artifacts that TrackNet fine-tuning consumes directly as training and
    # checkpoint-selection ("val" split) input, so it counts as training-input
    # creation. No protected eval clip may appear in any split built here --
    # see threed/racketsport/eval_guard.py for the policy this enforces.
    assert_not_training_on_eval_clip(selected_clips, allow_internal_val=False)
    normalized_video_paths = _normalize_video_paths(video_paths)
    hard_negative_spec = _load_hard_negative_plan(hard_negative_plan) if hard_negative_plan is not None else None
    train_augmentation_spec = _normalize_train_augmentation(
        profile=train_augmentation_profile,
        repeat=train_augmentation_repeat,
    )
    if hard_negative_spec is not None:
        _require_new_empty_out_dir(out, reason="hard-negative materialization")
        hard_negative_context_frames = _nonnegative_int(hard_negative_context_frames, "hard_negative_context_frames")
        hard_negative_repeat = _positive_int(hard_negative_repeat, "hard_negative_repeat")
        _validate_hard_negative_split_roles(hard_negative_spec, split_map=split_map, selected_clips=selected_clips)
    if train_augmentation_spec is not None:
        if not materialize_frames:
            raise ValueError("train augmentation requires --materialize-frames")
        _require_new_empty_out_dir(out, reason="train augmentation")
    if materialize_frames:
        _require_video_paths(selected_clips, normalized_video_paths)
    out.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": ARTIFACT_TYPE,
        "status": _dataset_status(
            materialize_frames=materialize_frames,
            hard_negative_plan=hard_negative_spec is not None,
            train_augmentation=train_augmentation_spec is not None,
        ),
        "ball_verified": False,
        "cvat_root": str(cvat_base),
        "source_yolo_manifest": str(yolo_manifest_path),
        "out_dir": str(out),
        "fps": fps,
        "tracknet_columns": list(TRACKNET_COLUMNS),
        "label_source": "reviewed CVAT ball boxes; frames without a visible ball box are explicit hidden negatives",
        "splits": {},
        "label_counts": {
            "clip_count": 0,
            "frame_count": 0,
            "reviewed_visible_ball_frame_count": 0,
            "reviewed_hidden_frame_count": 0,
        },
        "visibility_level_counts": _empty_visibility_level_counts(),
        "wbce_weight_counts": _empty_wbce_weight_counts(),
        "legacy_visibility_mapping": LEGACY_BALL_VISIBILITY_MAPPING,
        "blur_annotation_summary": _empty_blur_annotation_summary(),
        "leakage_checks": _leakage_checks(split_map, selected_clips),
        "limitations": _limitations(
            materialize_frames=materialize_frames,
            train_augmentation=train_augmentation_spec is not None,
        ),
    }
    if hard_negative_spec is not None:
        manifest["hard_negative_plan"] = {
            "source_plan": str(Path(hard_negative_plan).resolve() if Path(hard_negative_plan).is_absolute() else Path(hard_negative_plan)),
            "context_frames": hard_negative_context_frames,
            "repeat": hard_negative_repeat,
            "train_clips": [clip for clip in hard_negative_spec.get("train_clips", []) if clip in selected_clips],
            "validation_clips_held_out": [clip for clip in hard_negative_spec.get("validation_clips", []) if clip in selected_clips],
            "generated_window_count": 0,
            "unique_window_count": 0,
            "oversampled_frame_count": 0,
            "cache_policy": "hard_negative_plan_requires_new_empty_out_dir; generated frame dirs and median.npz files are not reused from prior datasets",
        }
    if train_augmentation_spec is not None:
        manifest["train_augmentation"] = {
            "profile": train_augmentation_spec["profile"],
            "repeat": train_augmentation_spec["repeat"],
            "applies_to_splits": ["train"],
            "source_sample_count": 0,
            "generated_sample_count": 0,
            "source_sample_types": [],
            "label_policy": "CSV labels are copied from reviewed CVAT-derived rows; no synthetic ball labels are generated.",
            "cache_policy": "train augmentation requires a new empty out_dir so transformed frame caches cannot be stale.",
        }

    match_index = 1
    for split in _ordered_splits(split_map):
        split_rows: list[dict[str, Any]] = []
        for clip in sorted(selected_clips):
            if split_map[clip] != split:
                continue
            reviewed_path = cvat_base / clip / "reviewed_boxes.json"
            annotations = _load_cvat_video_annotations(reviewed_path)
            labels = _dense_tracknet_labels_from_annotations(annotations)
            rally_id = f"{match_index}_01_00"
            match_name = f"match{match_index}"
            match_dir = out / split / match_name
            csv_dir = match_dir / ("corrected_csv" if split == "test" else "csv")
            csv_path = csv_dir / f"{rally_id}_ball.csv"
            visibility_metadata_path = _visibility_metadata_path(csv_path, rally_id=rally_id)
            frame_dir = match_dir / "frame" / rally_id
            _write_tracknet_csv(csv_path, labels)
            _write_visibility_metadata_json(visibility_metadata_path, labels)
            source_video_path = normalized_video_paths.get(clip)
            if materialize_frames:
                assert source_video_path is not None
                _extract_frames(source_video_path, frame_dir, len(labels))
                _write_median(frame_dir, len(labels))
            row = _clip_manifest_row(
                annotations=annotations,
                labels=labels,
                reviewed_path=reviewed_path,
                csv_path=csv_path,
                visibility_metadata_path=visibility_metadata_path,
                split=split,
                match_name=match_name,
                rally_id=rally_id,
                frame_dir=frame_dir,
                frames_materialized=materialize_frames,
                source_video_path=source_video_path,
            )
            row["training_sample_type"] = "dense_clip"
            split_rows.append(row)
            _add_counts(manifest["label_counts"], row)
            _add_visibility_counts(manifest["visibility_level_counts"], row["visibility_level_counts"])
            _add_wbce_weight_counts(manifest["wbce_weight_counts"], row["wbce_weight_counts"])
            _add_blur_annotation_counts(manifest["blur_annotation_summary"], row["blur_annotation_summary"])
            match_index += 1
        if split_rows:
            manifest["splits"][split] = split_rows

    if hard_negative_spec is not None:
        hard_negative_rows, next_match_index = _build_hard_negative_rows(
            plan=hard_negative_spec,
            cvat_base=cvat_base,
            out=out,
            split_map=split_map,
            selected_clips=selected_clips,
            video_paths=normalized_video_paths,
            materialize_frames=materialize_frames,
            context_frames=hard_negative_context_frames,
            repeat=hard_negative_repeat,
            first_match_index=match_index,
        )
        match_index = next_match_index
        if hard_negative_rows:
            manifest["splits"].setdefault("train", []).extend(hard_negative_rows)
            for row in hard_negative_rows:
                _add_counts(manifest["label_counts"], row)
                _add_visibility_counts(manifest["visibility_level_counts"], row["visibility_level_counts"])
                _add_wbce_weight_counts(manifest["wbce_weight_counts"], row["wbce_weight_counts"])
                _add_blur_annotation_counts(manifest["blur_annotation_summary"], row["blur_annotation_summary"])
        hard_negative_manifest = manifest["hard_negative_plan"]
        hard_negative_manifest["generated_window_count"] = len(hard_negative_rows)
        hard_negative_manifest["hard_negative_window_count"] = sum(
            1 for row in hard_negative_rows if row.get("detector_error_kind") == "hidden_false_positive"
        )
        hard_negative_manifest["visible_error_window_count"] = sum(
            1 for row in hard_negative_rows if row.get("detector_error_kind") in {"visible_miss", "visible_mislocalized"}
        )
        hard_negative_manifest["unique_window_count"] = len(
            {
                (row["clip"], row["source_frame_start"], row["source_frame_end"], row.get("detector_error_kind"))
                for row in hard_negative_rows
            }
        )
        hard_negative_manifest["oversampled_frame_count"] = sum(int(row["frame_count"]) for row in hard_negative_rows)

    if train_augmentation_spec is not None:
        source_train_rows = list(manifest["splits"].get("train", []))
        train_augmented_rows, next_match_index = _build_train_augmentation_rows(
            source_rows=source_train_rows,
            spec=train_augmentation_spec,
            out=out,
            first_match_index=match_index,
        )
        match_index = next_match_index
        if train_augmented_rows:
            manifest["splits"].setdefault("train", []).extend(train_augmented_rows)
            for row in train_augmented_rows:
                _add_counts(manifest["label_counts"], row)
                _add_visibility_counts(manifest["visibility_level_counts"], row["visibility_level_counts"])
                _add_wbce_weight_counts(manifest["wbce_weight_counts"], row["wbce_weight_counts"])
                _add_blur_annotation_counts(manifest["blur_annotation_summary"], row["blur_annotation_summary"])
        train_augmentation_manifest = manifest["train_augmentation"]
        train_augmentation_manifest["source_sample_count"] = len(source_train_rows)
        train_augmentation_manifest["generated_sample_count"] = len(train_augmented_rows)
        train_augmentation_manifest["source_sample_types"] = sorted(
            {str(row.get("training_sample_type", "")) for row in source_train_rows if row.get("training_sample_type")}
        )

    manifest["label_counts"]["clip_count"] = sum(len(rows) for rows in manifest["splits"].values())
    manifest["next_match_index"] = match_index
    manifest["next_gpu_commands"] = _next_gpu_commands(manifest)
    manifest_json = out / MANIFEST_JSON
    manifest_md = out / MANIFEST_MD
    manifest_json.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest_md.write_text(render_ball_tracknet_cvat_dataset_markdown(manifest), encoding="utf-8")
    manifest["manifest_json"] = str(manifest_json)
    manifest["manifest_md"] = str(manifest_md)
    return manifest


def render_ball_tracknet_cvat_dataset_markdown(manifest: Mapping[str, Any]) -> str:
    counts = manifest.get("label_counts", {})
    blur_summary = manifest.get("blur_annotation_summary", {})
    visibility_counts = manifest.get("visibility_level_counts", {})
    wbce_counts = manifest.get("wbce_weight_counts", {})
    status = str(manifest.get("status", ""))
    lines = [
        "# BALL TrackNet CVAT Dataset",
        "",
        f"Status: `{manifest.get('status')}`",
        "",
        "BALL is not verified by this artifact. It prepares dense supervised TrackNetV3 labels from reviewed CVAT ball boxes.",
        "",
        "## Counts",
        "",
        f"- Clips: {counts.get('clip_count', 0)}",
        f"- Frames: {counts.get('frame_count', 0)}",
        f"- Visible ball frames: {counts.get('reviewed_visible_ball_frame_count', 0)}",
        f"- Hidden negative frames: {counts.get('reviewed_hidden_frame_count', 0)}",
        "",
        "## Visibility Levels",
        "",
        f"- clear: {visibility_counts.get('clear', 0)}",
        f"- partial: {visibility_counts.get('partial', 0)}",
        f"- full: {visibility_counts.get('full', 0)}",
        f"- out_of_frame: {visibility_counts.get('out_of_frame', 0)}",
        f"- legacy visible: {visibility_counts.get('legacy_visible', 0)}",
        f"- legacy hidden: {visibility_counts.get('legacy_hidden', 0)}",
        f"- WBCE weight 1: {wbce_counts.get('1', 0)}",
        f"- WBCE weight 2: {wbce_counts.get('2', 0)}",
        f"- WBCE weight 3: {wbce_counts.get('3', 0)}",
        f"- unweighted legacy: {wbce_counts.get('unweighted_legacy', 0)}",
        "",
        "## Blur Annotation Summary",
        "",
        f"- Visible ball frames: {blur_summary.get('visible_ball_frame_count', 0)}",
        f"- Blur angle labels: {blur_summary.get('blur_angle_labeled_count', 0)}",
        f"- Blur length labels: {blur_summary.get('blur_length_labeled_count', 0)}",
        f"- Blur width labels: {blur_summary.get('blur_width_labeled_count', 0)}",
        "",
        "## Splits",
        "",
        "| Split | Clip | Frames | Visible | Hidden | CSV |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for split, rows in manifest.get("splits", {}).items():
        for row in rows:
            lines.append(
                "| {split} | {clip} | {frame_count} | {visible} | {hidden} | `{csv}` |".format(
                    split=split,
                    clip=row.get("clip", ""),
                    frame_count=row.get("frame_count", 0),
                    visible=row.get("visible_label_frames", 0),
                    hidden=row.get("hidden_label_frames", 0),
                    csv=row.get("csv", ""),
                )
            )
    train_augmentation = manifest.get("train_augmentation")
    if isinstance(train_augmentation, Mapping):
        lines.extend(
            [
                "",
                "## Train Augmentation",
                "",
                f"- Profile: `{train_augmentation.get('profile')}`",
                f"- Repeat: {train_augmentation.get('repeat')}",
                f"- Source samples: {train_augmentation.get('source_sample_count')}",
                f"- Generated samples: {train_augmentation.get('generated_sample_count')}",
                "- Applies only to the `train` split.",
                "- CSV labels are copied unchanged from reviewed CVAT-derived rows; no synthetic ball labels are generated.",
            ]
        )
    lines.extend(["", "## Next GPU Commands", ""])
    for command in manifest.get("next_gpu_commands", []):
        lines.extend(["```bash", str(command), "```", ""])
    lines.extend(
        [
            "## Limits",
            "",
            "- This is a label artifact, not a model checkpoint.",
            "- Frame directories were materialized." if status.startswith("tracknet_dataset_materialized") else "- Frame directories were not materialized.",
            "- Do not claim BALL verified until a trained checkpoint improves reviewed-label held-out metrics over the A100 baseline.",
            "- Do not promote validation accuracy alone; rerun the CVAT benchmark and report F1@20, precision, recall, hidden-FP rate, and teleports.",
            "",
        ]
    )
    return "\n".join(lines)


def _dense_tracknet_labels_from_annotations(annotations: CvatVideoAnnotations) -> list[TrackNetCvatLabel]:
    frames_by_index = {frame.frame_index: frame for frame in annotations.frames}
    frame_count = _annotation_frame_count(annotations)
    labels: list[TrackNetCvatLabel] = []
    for frame_index in range(frame_count):
        frame = frames_by_index.get(frame_index)
        ball_boxes = [box for box in (frame.boxes if frame is not None else []) if box.label == "ball"]
        if len(ball_boxes) > 1:
            raise ValueError(f"multiple ball boxes in {annotations.clip_id} frame {frame_index}")
        frame_visibility_level = _frame_ball_visibility_level(frame)
        if not ball_boxes:
            if frame_visibility_level in {"clear", "partial"}:
                raise ValueError(f"{annotations.clip_id} frame {frame_index} has {frame_visibility_level} without a ball box")
            labels.append(
                TrackNetCvatLabel(
                    frame=frame_index,
                    visibility=0,
                    x=0.0,
                    y=0.0,
                    source="reviewed_cvat_ball_visibility_level" if frame_visibility_level is not None else "reviewed_hidden",
                    visibility_level=frame_visibility_level,
                    wbce_weight=_visibility_wbce_weight(frame_visibility_level),
                    legacy_visibility_state=None if frame_visibility_level is not None else "legacy_hidden",
                )
            )
            continue
        box = ball_boxes[0]
        visibility_level = _merge_box_and_frame_visibility_level(
            box.visibility_level,
            frame_visibility_level,
            clip_id=annotations.clip_id,
            frame_index=frame_index,
        )
        if visibility_level in {"full", "out_of_frame"}:
            labels.append(
                TrackNetCvatLabel(
                    frame=frame_index,
                    visibility=0,
                    x=0.0,
                    y=0.0,
                    source="reviewed_cvat_ball_visibility_level",
                    center_convention=box.center_convention,
                    blur_angle_deg=box.blur_angle_deg,
                    blur_length_px=box.blur_length_px,
                    blur_width_px=box.blur_width_px,
                    blur_label_quality=box.blur_label_quality,
                    visibility_level=visibility_level,
                    wbce_weight=_visibility_wbce_weight(visibility_level),
                )
            )
            continue
        x, y, width, height = box.bbox_xywh
        labels.append(
            TrackNetCvatLabel(
                frame=frame_index,
                visibility=1,
                x=float(x) + float(width) * 0.5,
                y=float(y) + float(height) * 0.5,
                source="reviewed_cvat_ball_box",
                center_convention=box.center_convention,
                blur_angle_deg=box.blur_angle_deg,
                blur_length_px=box.blur_length_px,
                blur_width_px=box.blur_width_px,
                blur_label_quality=box.blur_label_quality,
                visibility_level=visibility_level,
                wbce_weight=_visibility_wbce_weight(visibility_level),
                legacy_visibility_state=None if visibility_level is not None else "legacy_visible",
            )
        )
    return labels


def _load_cvat_video_annotations(path: str | Path) -> CvatVideoAnnotations:
    parsed = validate_artifact_file("cvat_video_annotations", path)
    if not isinstance(parsed, CvatVideoAnnotations):
        raise ValueError(f"reviewed boxes artifact did not parse as CvatVideoAnnotations: {path}")
    return parsed


def _annotation_frame_count(annotations: CvatVideoAnnotations) -> int:
    frame_indexes = [frame.frame_index for frame in annotations.frames]
    max_frame = max(frame_indexes, default=-1) + 1
    return max(int(annotations.task.size), max_frame)


def _frame_ball_visibility_level(frame: CvatVideoFrame | None) -> BallVisibilityLevel | None:
    if frame is None:
        return None
    return frame.visibility_levels_by_label.get("ball")


def _merge_box_and_frame_visibility_level(
    box_level: BallVisibilityLevel | None,
    frame_level: BallVisibilityLevel | None,
    *,
    clip_id: str,
    frame_index: int,
) -> BallVisibilityLevel | None:
    if box_level is not None and frame_level is not None and box_level != frame_level:
        raise ValueError(
            f"{clip_id} frame {frame_index} has conflicting ball visibility levels: {box_level} vs {frame_level}"
        )
    return box_level or frame_level


def _visibility_wbce_weight(visibility_level: BallVisibilityLevel | None) -> int | None:
    if visibility_level is None:
        return None
    return BALL_VISIBILITY_WBCE_WEIGHTS[visibility_level]


def _write_tracknet_csv(path: Path, labels: Sequence[TrackNetCvatLabel]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(TRACKNET_COLUMNS)
        for label in labels:
            writer.writerow([label.frame, label.visibility, f"{label.x:.3f}", f"{label.y:.3f}"])


def _visibility_metadata_path(csv_path: Path, *, rally_id: str) -> Path:
    return csv_path.with_name(f"{rally_id}_visibility.json")


def _write_visibility_metadata_json(path: Path, labels: Sequence[TrackNetCvatLabel]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "artifact_type": "racketsport_ball_visibility_metadata",
        "visibility_levels": list(BALL_VISIBILITY_LEVELS),
        "wbce_weights": dict(BALL_VISIBILITY_WBCE_WEIGHTS),
        "legacy_visibility_mapping": LEGACY_BALL_VISIBILITY_MAPPING,
        "rows": [
            {
                "frame": int(label.frame),
                "tracknet_visibility": int(label.visibility),
                "x": float(label.x),
                "y": float(label.y),
                "visibility_level": label.visibility_level,
                "wbce_weight": label.wbce_weight,
                "legacy_visibility_state": label.legacy_visibility_state,
                "source": label.source,
            }
            for label in labels
        ],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _clip_manifest_row(
    *,
    annotations: CvatVideoAnnotations,
    labels: Sequence[TrackNetCvatLabel],
    reviewed_path: Path,
    csv_path: Path,
    visibility_metadata_path: Path,
    split: str,
    match_name: str,
    rally_id: str,
    frame_dir: Path,
    frames_materialized: bool,
    source_video_path: Path | None,
) -> dict[str, Any]:
    visible = [label for label in labels if label.visibility == 1]
    hidden = len(labels) - len(visible)
    widths = []
    heights = []
    for frame in annotations.frames:
        for box in frame.boxes:
            if box.label == "ball":
                widths.append(float(box.bbox_xywh[2]))
                heights.append(float(box.bbox_xywh[3]))
    return {
        "clip": annotations.clip_id,
        "split": split,
        "match": match_name,
        "rally_id": rally_id,
        "reviewed_boxes": str(reviewed_path),
        "csv": str(csv_path),
        "visibility_metadata_json": str(visibility_metadata_path),
        "frame_dir": str(frame_dir),
        "frames_materialized": bool(frames_materialized),
        "source_video_path": str(source_video_path) if source_video_path is not None else None,
        "frame_count": len(labels),
        "visible_label_frames": len(visible),
        "hidden_label_frames": hidden,
        "first_visible_frame": visible[0].frame if visible else None,
        "last_visible_frame": visible[-1].frame if visible else None,
        "original_size": list(annotations.task.original_size),
        "source_video": annotations.task.source,
        "ball_bbox_width_px_median": median(widths) if widths else None,
        "ball_bbox_height_px_median": median(heights) if heights else None,
        "blur_annotation_summary": _blur_annotation_summary(labels),
        "visibility_level_counts": _visibility_level_counts(labels),
        "wbce_weight_counts": _wbce_weight_counts(labels),
    }


def _split_map_from_yolo_manifest(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError(f"YOLO manifest rows must be a list: {path}")
    by_clip: dict[str, set[str]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"YOLO manifest row {index} must be an object")
        clip = row.get("clip_id")
        split = row.get("split")
        if not isinstance(clip, str) or not clip:
            raise ValueError(f"YOLO manifest row {index} requires clip_id")
        if not isinstance(split, str) or not split:
            raise ValueError(f"YOLO manifest row {index} requires split")
        by_clip.setdefault(clip, set()).add(split)
    leakage = {clip: sorted(splits) for clip, splits in by_clip.items() if len(splits) > 1}
    if leakage:
        details = ", ".join(f"{clip}={splits}" for clip, splits in sorted(leakage.items()))
        raise ValueError(f"split leakage in YOLO manifest: {details}")
    return {clip: next(iter(splits)) for clip, splits in sorted(by_clip.items())}


def _normalize_video_paths(video_paths: Mapping[str, str | Path] | None) -> dict[str, Path]:
    if video_paths is None:
        return {}
    normalized: dict[str, Path] = {}
    for clip, path in video_paths.items():
        if not clip:
            raise ValueError("video path clip id must be non-empty")
        video_path = Path(path)
        if not video_path.is_file():
            raise FileNotFoundError(f"missing video for {clip}: {video_path}")
        normalized[str(clip)] = video_path
    return normalized


def _require_video_paths(selected_clips: Sequence[str], video_paths: Mapping[str, Path]) -> None:
    missing = [clip for clip in selected_clips if clip not in video_paths]
    if missing:
        raise ValueError(f"--materialize-frames requires --video for clip(s): {', '.join(missing)}")


def _selected_clips(split_map: Mapping[str, str], clips: Sequence[str] | None) -> list[str]:
    if clips is None:
        return sorted(split_map)
    selected = sorted(dict.fromkeys(clips))
    missing = [clip for clip in selected if clip not in split_map]
    if missing:
        raise ValueError(f"selected clip(s) missing from YOLO manifest: {', '.join(missing)}")
    return selected


def _load_hard_negative_plan(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        raise ValueError("hard negative plan path is required")
    plan_path = Path(path)
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    if payload.get("artifact_type") != "racketsport_ball_hard_negative_iteration_plan":
        raise ValueError(f"unexpected hard negative plan artifact_type in {plan_path}")
    if payload.get("ball_verified") is True or payload.get("promotion_claimed") is True:
        raise ValueError("hard negative plans must not claim BALL promotion")
    clips = payload.get("clips")
    if not isinstance(clips, dict):
        raise ValueError("hard negative plan requires clips object")
    for key in ("train_clips", "validation_clips"):
        value = payload.get(key)
        if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
            raise ValueError(f"hard negative plan requires {key} list of clip ids")
    return payload


def _validate_hard_negative_split_roles(
    plan: Mapping[str, Any],
    *,
    split_map: Mapping[str, str],
    selected_clips: Sequence[str],
) -> None:
    selected = set(selected_clips)
    for clip in plan.get("train_clips", []):
        if clip in selected and split_map.get(clip) != "train":
            raise ValueError(f"hard negative train clip is not in train split: {clip}={split_map.get(clip)}")
    for clip in plan.get("validation_clips", []):
        if clip in selected and split_map.get(clip) == "train":
            raise ValueError(f"hard negative validation clip must stay held out, not train: {clip}")


def _ordered_splits(split_map: Mapping[str, str]) -> list[str]:
    present = set(split_map.values())
    ordered = [split for split in SPLIT_ORDER if split in present]
    ordered.extend(sorted(present - set(ordered)))
    return ordered


def _leakage_checks(split_map: Mapping[str, str], selected_clips: Sequence[str]) -> dict[str, Any]:
    split_counts: dict[str, int] = {}
    for clip in selected_clips:
        split = split_map[clip]
        split_counts[split] = split_counts.get(split, 0) + 1
    return {
        "clips_with_multiple_splits": [],
        "disjoint_clip_splits": True,
        "split_clip_counts": dict(sorted(split_counts.items())),
    }


def _empty_blur_annotation_summary() -> dict[str, Any]:
    return {
        "visible_ball_frame_count": 0,
        "center_convention_counts": {value: 0 for value in CENTER_CONVENTION_VALUES},
        "blur_label_quality_counts": {value: 0 for value in BLUR_LABEL_QUALITY_VALUES},
        "blur_angle_labeled_count": 0,
        "blur_length_labeled_count": 0,
        "blur_width_labeled_count": 0,
    }


def _blur_annotation_summary(labels: Sequence[TrackNetCvatLabel]) -> dict[str, Any]:
    summary = _empty_blur_annotation_summary()
    for label in labels:
        if label.visibility != 1:
            continue
        summary["visible_ball_frame_count"] += 1
        center_convention = label.center_convention or "unknown"
        blur_label_quality = label.blur_label_quality or "unknown"
        _increment_required_count(summary["center_convention_counts"], center_convention, "center_convention")
        _increment_required_count(summary["blur_label_quality_counts"], blur_label_quality, "blur_label_quality")
        if label.blur_angle_deg is not None:
            summary["blur_angle_labeled_count"] += 1
        if label.blur_length_px is not None:
            summary["blur_length_labeled_count"] += 1
        if label.blur_width_px is not None:
            summary["blur_width_labeled_count"] += 1
    return summary


def _empty_visibility_level_counts() -> dict[str, int]:
    counts = {value: 0 for value in BALL_VISIBILITY_LEVELS}
    counts.update({value: 0 for value in LEGACY_VISIBILITY_STATES})
    return counts


def _visibility_level_counts(labels: Sequence[TrackNetCvatLabel]) -> dict[str, int]:
    counts = _empty_visibility_level_counts()
    for label in labels:
        if label.visibility_level is not None:
            counts[label.visibility_level] += 1
        elif label.legacy_visibility_state is not None:
            counts[label.legacy_visibility_state] += 1
    return counts


def _empty_wbce_weight_counts() -> dict[str, int]:
    return {"1": 0, "2": 0, "3": 0, "unweighted_legacy": 0}


def _wbce_weight_counts(labels: Sequence[TrackNetCvatLabel]) -> dict[str, int]:
    counts = _empty_wbce_weight_counts()
    for label in labels:
        if label.wbce_weight is None:
            counts["unweighted_legacy"] += 1
        else:
            counts[str(label.wbce_weight)] += 1
    return counts


def _increment_required_count(counts: dict[str, int], value: str, field_name: str) -> None:
    if value not in counts:
        raise ValueError(f"unsupported {field_name}: {value}")
    counts[value] += 1


def _add_blur_annotation_counts(target: dict[str, Any], source: Mapping[str, Any]) -> None:
    target["visible_ball_frame_count"] = int(target.get("visible_ball_frame_count", 0)) + int(
        source.get("visible_ball_frame_count", 0)
    )
    for key in ("center_convention_counts", "blur_label_quality_counts"):
        target_counts = target[key]
        source_counts = source.get(key, {})
        if not isinstance(target_counts, dict) or not isinstance(source_counts, Mapping):
            raise ValueError(f"invalid blur annotation count map: {key}")
        for value, count in source_counts.items():
            value_key = str(value)
            if value_key not in target_counts:
                raise ValueError(f"unsupported {key}: {value_key}")
            target_counts[value_key] = int(target_counts[value_key]) + int(count)
    for key in ("blur_angle_labeled_count", "blur_length_labeled_count", "blur_width_labeled_count"):
        target[key] = int(target.get(key, 0)) + int(source.get(key, 0))


def _add_visibility_counts(target: dict[str, int], source: Mapping[str, int]) -> None:
    for key, value in source.items():
        if key not in target:
            raise ValueError(f"unsupported visibility count key: {key}")
        target[key] = int(target[key]) + int(value)


def _add_wbce_weight_counts(target: dict[str, int], source: Mapping[str, int]) -> None:
    for key, value in source.items():
        if key not in target:
            raise ValueError(f"unsupported WBCE weight count key: {key}")
        target[key] = int(target[key]) + int(value)


def _add_counts(counts: dict[str, Any], row: Mapping[str, Any]) -> None:
    counts["frame_count"] = int(counts.get("frame_count", 0)) + int(row["frame_count"])
    counts["reviewed_visible_ball_frame_count"] = int(counts.get("reviewed_visible_ball_frame_count", 0)) + int(
        row["visible_label_frames"]
    )
    counts["reviewed_hidden_frame_count"] = int(counts.get("reviewed_hidden_frame_count", 0)) + int(
        row["hidden_label_frames"]
    )


def _build_hard_negative_rows(
    *,
    plan: Mapping[str, Any],
    cvat_base: Path,
    out: Path,
    split_map: Mapping[str, str],
    selected_clips: Sequence[str],
    video_paths: Mapping[str, Path],
    materialize_frames: bool,
    context_frames: int,
    repeat: int,
    first_match_index: int,
) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    match_index = first_match_index
    selected = set(selected_clips)
    train_clips = [clip for clip in plan.get("train_clips", []) if clip in selected and split_map.get(clip) == "train"]
    clip_entries = plan.get("clips", {})
    assert isinstance(clip_entries, Mapping)

    for clip in sorted(train_clips):
        reviewed_path = cvat_base / clip / "reviewed_boxes.json"
        annotations = _load_cvat_video_annotations(reviewed_path)
        labels = _dense_tracknet_labels_from_annotations(annotations)
        windows = _detector_error_windows_for_clip(
            plan_clip=clip_entries.get(clip, {}),
            frame_count=len(labels),
            context_frames=context_frames,
        )
        source_video_path = video_paths.get(clip)
        for window_index, (error_kind, start_frame, end_frame, original_start, original_end) in enumerate(windows, start=1):
            for repeat_index in range(1, repeat + 1):
                rally_id = f"{match_index}_01_00"
                match_name = f"match{match_index}"
                match_dir = out / "train" / match_name
                csv_path = match_dir / "csv" / f"{rally_id}_ball.csv"
                visibility_metadata_path = _visibility_metadata_path(csv_path, rally_id=rally_id)
                frame_dir = match_dir / "frame" / rally_id
                window_labels = _slice_labels_for_window(labels, start_frame=start_frame, end_frame=end_frame)
                _write_tracknet_csv(csv_path, window_labels)
                _write_visibility_metadata_json(visibility_metadata_path, window_labels)
                if materialize_frames:
                    if source_video_path is None:
                        raise ValueError(f"--materialize-frames requires --video for hard-negative clip: {clip}")
                    _extract_frame_window(source_video_path, frame_dir, start_frame, len(window_labels))
                    _write_median(frame_dir, len(window_labels))
                row = _clip_manifest_row(
                    annotations=annotations,
                    labels=window_labels,
                    reviewed_path=reviewed_path,
                    csv_path=csv_path,
                    visibility_metadata_path=visibility_metadata_path,
                    split="train",
                    match_name=match_name,
                    rally_id=rally_id,
                    frame_dir=frame_dir,
                    frames_materialized=materialize_frames,
                    source_video_path=source_video_path,
                )
                row.update(
                    {
                        "training_sample_type": (
                            "hard_negative_oversample"
                            if error_kind == "hidden_false_positive"
                            else "detector_error_oversample"
                        ),
                        "detector_error_kind": error_kind,
                        "source_frame_start": start_frame,
                        "source_frame_end": end_frame,
                        "hard_negative_original_start": original_start,
                        "hard_negative_original_end": original_end,
                        "hard_negative_window_index": window_index,
                        "hard_negative_repeat_index": repeat_index,
                    }
                )
                rows.append(row)
                match_index += 1
    return rows, match_index


def _detector_error_windows_for_clip(
    *,
    plan_clip: object,
    frame_count: int,
    context_frames: int,
) -> list[tuple[str, int, int, int, int]]:
    windows: list[tuple[str, int, int, int, int]] = []
    for key, kind in (
        ("hard_negative_hidden_fp_ranges", "hidden_false_positive"),
        ("visible_miss_ranges", "visible_miss"),
        ("visible_mislocalized_ranges", "visible_mislocalized"),
    ):
        windows.extend(
            (kind, start, end, original_start, original_end)
            for start, end, original_start, original_end in _error_windows_for_key(
                plan_clip=plan_clip,
                key=key,
                frame_count=frame_count,
                context_frames=context_frames,
            )
        )
    return sorted(windows, key=lambda item: (item[1], item[2], item[0]))


def _hard_negative_windows_for_clip(
    *,
    plan_clip: object,
    frame_count: int,
    context_frames: int,
) -> list[tuple[int, int, int, int]]:
    return _error_windows_for_key(
        plan_clip=plan_clip,
        key="hard_negative_hidden_fp_ranges",
        frame_count=frame_count,
        context_frames=context_frames,
    )


def _error_windows_for_key(
    *,
    plan_clip: object,
    key: str,
    frame_count: int,
    context_frames: int,
) -> list[tuple[int, int, int, int]]:
    if not isinstance(plan_clip, Mapping):
        return []
    ranges = plan_clip.get(key, [])
    if not isinstance(ranges, list):
        raise ValueError(f"{key} must be a list")
    expanded: list[tuple[int, int, int, int]] = []
    for index, item in enumerate(ranges):
        if not isinstance(item, Mapping):
            raise ValueError(f"{key} range {index} must be an object")
        start = _frame_int(item.get("start"), f"{key} range {index} start")
        end = _frame_int(item.get("end"), f"{key} range {index} end")
        if end < start:
            raise ValueError(f"{key} range {index} end before start")
        expanded_start = max(0, start - context_frames)
        expanded_end = min(frame_count - 1, end + context_frames)
        if expanded_start <= expanded_end:
            expanded.append((expanded_start, expanded_end, start, end))
    return _merge_hard_negative_windows(expanded)


def _merge_hard_negative_windows(windows: Sequence[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    merged: list[tuple[int, int, int, int]] = []
    for start, end, original_start, original_end in sorted(windows):
        if not merged or start > merged[-1][1] + 1:
            merged.append((start, end, original_start, original_end))
            continue
        previous_start, previous_end, previous_original_start, previous_original_end = merged[-1]
        merged[-1] = (
            previous_start,
            max(previous_end, end),
            min(previous_original_start, original_start),
            max(previous_original_end, original_end),
        )
    return merged


def _slice_labels_for_window(
    labels: Sequence[TrackNetCvatLabel],
    *,
    start_frame: int,
    end_frame: int,
) -> list[TrackNetCvatLabel]:
    sliced: list[TrackNetCvatLabel] = []
    for relative_frame, label in enumerate(labels[start_frame : end_frame + 1]):
        sliced.append(
            TrackNetCvatLabel(
                frame=relative_frame,
                visibility=label.visibility,
                x=label.x,
                y=label.y,
                source=f"{label.source};hard_negative_window",
                center_convention=label.center_convention,
                blur_angle_deg=label.blur_angle_deg,
                blur_length_px=label.blur_length_px,
                blur_width_px=label.blur_width_px,
                blur_label_quality=label.blur_label_quality,
                visibility_level=label.visibility_level,
                wbce_weight=label.wbce_weight,
                legacy_visibility_state=label.legacy_visibility_state,
            )
        )
    return sliced


def _normalize_train_augmentation(*, profile: str | None, repeat: int) -> dict[str, Any] | None:
    if profile is None:
        return None
    if profile not in TRAIN_AUGMENTATION_PROFILES:
        supported = ", ".join(sorted(TRAIN_AUGMENTATION_PROFILES))
        raise ValueError(f"unsupported train augmentation profile: {profile}; supported profiles: {supported}")
    return {"profile": profile, "repeat": _positive_int(repeat, "train_augmentation_repeat")}


def _build_train_augmentation_rows(
    *,
    source_rows: Sequence[Mapping[str, Any]],
    spec: Mapping[str, Any],
    out: Path,
    first_match_index: int,
) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    match_index = first_match_index
    profile = str(spec["profile"])
    repeat = int(spec["repeat"])
    for source_row in source_rows:
        if source_row.get("split") != "train":
            continue
        if source_row.get("frames_materialized") is not True:
            raise ValueError("train augmentation requires materialized train frame directories")
        source_csv = Path(str(source_row.get("csv", "")))
        source_frame_dir = Path(str(source_row.get("frame_dir", "")))
        if not source_csv.is_file():
            raise FileNotFoundError(f"missing source CSV for train augmentation: {source_csv}")
        if not source_frame_dir.is_dir():
            raise FileNotFoundError(f"missing source frame dir for train augmentation: {source_frame_dir}")
        source_sample_type = str(source_row.get("training_sample_type", "dense_clip"))
        source_visibility_metadata = Path(str(source_row.get("visibility_metadata_json", "")))
        if not source_visibility_metadata.is_file():
            raise FileNotFoundError(f"missing source visibility metadata for train augmentation: {source_visibility_metadata}")
        for repeat_index in range(1, repeat + 1):
            recipe = _augmentation_recipe(profile, repeat_index=repeat_index)
            rally_id = f"{match_index}_01_00"
            match_name = f"match{match_index}"
            match_dir = out / "train" / match_name
            csv_path = match_dir / "csv" / f"{rally_id}_ball.csv"
            visibility_metadata_path = _visibility_metadata_path(csv_path, rally_id=rally_id)
            frame_dir = match_dir / "frame" / rally_id
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_csv, csv_path)
            visibility_metadata_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_visibility_metadata, visibility_metadata_path)
            _augment_frame_dir(
                source_frame_dir,
                frame_dir,
                frame_count=int(source_row["frame_count"]),
                recipe=recipe,
            )
            row = dict(source_row)
            row.update(
                {
                    "match": match_name,
                    "rally_id": rally_id,
                    "csv": str(csv_path),
                    "visibility_metadata_json": str(visibility_metadata_path),
                    "frame_dir": str(frame_dir),
                    "frames_materialized": True,
                    "training_sample_type": f"visual_augmented_{source_sample_type}",
                    "source_training_sample_type": source_sample_type,
                    "source_match": source_row.get("match"),
                    "source_rally_id": source_row.get("rally_id"),
                    "source_csv": source_row.get("csv"),
                    "source_visibility_metadata_json": source_row.get("visibility_metadata_json"),
                    "source_frame_dir": source_row.get("frame_dir"),
                    "augmentation_profile": profile,
                    "augmentation_repeat_index": repeat_index,
                    "augmentation_recipe": recipe["name"],
                    "augmentation_recipe_index": recipe["profile_index"],
                }
            )
            rows.append(row)
            match_index += 1
    return rows, match_index


def _augmentation_recipe(profile: str, *, repeat_index: int) -> dict[str, Any]:
    if repeat_index <= 0:
        raise ValueError("repeat_index must be positive")
    recipes = TRAIN_AUGMENTATION_PROFILES.get(profile)
    if recipes is None:
        supported = ", ".join(sorted(TRAIN_AUGMENTATION_PROFILES))
        raise ValueError(f"unsupported train augmentation profile: {profile}; supported profiles: {supported}")
    profile_index = (repeat_index - 1) % len(recipes)
    recipe = dict(recipes[profile_index])
    recipe["profile"] = profile
    recipe["profile_index"] = profile_index + 1
    return recipe


def _augment_frame_dir(source_frame_dir: Path, target_frame_dir: Path, *, frame_count: int, recipe: Mapping[str, Any]) -> None:
    if frame_count < 0:
        raise ValueError("frame_count must be non-negative")
    target_frame_dir.mkdir(parents=True, exist_ok=True)
    if frame_count == 0:
        return
    from PIL import Image, ImageEnhance, ImageFilter

    for frame_index in range(frame_count):
        source_path = source_frame_dir / f"{frame_index}.png"
        if not source_path.is_file():
            raise FileNotFoundError(f"missing source frame for train augmentation: {source_path}")
        with Image.open(source_path) as source_image:
            image = source_image.convert("RGB")
        blur = recipe.get("motion_blur")
        if blur == "horizontal_3":
            image = image.filter(ImageFilter.Kernel((3, 3), [0, 0, 0, 1, 1, 1, 0, 0, 0], scale=3))
        elif blur == "vertical_3":
            image = image.filter(ImageFilter.Kernel((3, 3), [0, 1, 0, 0, 1, 0, 0, 1, 0], scale=3))
        elif blur is not None:
            raise ValueError(f"unsupported train augmentation motion_blur: {blur}")
        image = ImageEnhance.Brightness(image).enhance(float(recipe.get("brightness", 1.0)))
        image = ImageEnhance.Contrast(image).enhance(float(recipe.get("contrast", 1.0)))
        image = ImageEnhance.Color(image).enhance(float(recipe.get("color", 1.0)))
        jpeg_quality = recipe.get("jpeg_quality")
        if jpeg_quality is not None:
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=int(jpeg_quality))
            buffer.seek(0)
            with Image.open(buffer) as compressed_image:
                image = compressed_image.convert("RGB")
        image.save(target_frame_dir / f"{frame_index}.png")
    _write_median(target_frame_dir, frame_count)


def _next_gpu_commands(manifest: Mapping[str, Any]) -> list[str]:
    out_dir = str(manifest.get("out_dir"))
    video_by_clip: dict[str, str] = {}
    for rows in manifest.get("splits", {}).values():
        for row in rows:
            video_path = row.get("source_video_path")
            clip = row.get("clip")
            if isinstance(clip, str) and video_path and clip not in video_by_clip:
                video_by_clip[clip] = str(video_path)
    video_args = [f"--video {clip}={path}" for clip, path in sorted(video_by_clip.items())]
    materialize = " --materialize-frames" if video_args else ""
    output_arg = f"--out-dir {out_dir}"
    if "hard_negative_plan" in manifest or "train_augmentation" in manifest:
        output_arg = f"--out-dir <new_empty_out_dir_like_{Path(out_dir).name}>"
    return [
        "python scripts/racketsport/build_ball_tracknet_cvat_dataset.py "
        f"--cvat-root {manifest.get('cvat_root')} "
        f"--yolo-manifest {manifest.get('source_yolo_manifest')} "
        f"{output_arg}"
        f"{_hard_negative_cli_args(manifest)}"
        f"{_train_augmentation_cli_args(manifest)}"
        f"{materialize}"
        f"{(' ' + ' '.join(video_args)) if video_args else ''}",
        "cd /workspace/runs/pickleball_pretraining/TrackNetV3_finetune_repo && "
        "/opt/conda/envs/fast_sam_3d_body/bin/python train.py "
        "--model_name TrackNet --seq_len 8 --epochs 8 --batch_size 10 --optim Adam "
        "--learning_rate 0.0001 --bg_mode concat --alpha 0.5 --resume_training "
        "--save_dir /workspace/runs/pickleball_pretraining/tracknetv3_full_cvat_dense_hidden_20260630/train --verbose",
    ]


def _hard_negative_cli_args(manifest: Mapping[str, Any]) -> str:
    hard_negative = manifest.get("hard_negative_plan")
    if not isinstance(hard_negative, Mapping):
        return ""
    return (
        f" --hard-negative-plan {hard_negative.get('source_plan')}"
        f" --hard-negative-context-frames {hard_negative.get('context_frames')}"
        f" --hard-negative-repeat {hard_negative.get('repeat')}"
    )


def _train_augmentation_cli_args(manifest: Mapping[str, Any]) -> str:
    train_augmentation = manifest.get("train_augmentation")
    if not isinstance(train_augmentation, Mapping):
        return ""
    return (
        f" --train-augmentation-profile {train_augmentation.get('profile')}"
        f" --train-augmentation-repeat {train_augmentation.get('repeat')}"
    )


def _positive_float(value: object, name: str) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be positive") from exc
    if number <= 0.0:
        raise ValueError(f"{name} must be positive")
    return number


def _positive_int(value: object, name: str) -> int:
    try:
        number = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if number <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return number


def _nonnegative_int(value: object, name: str) -> int:
    try:
        number = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be non-negative") from exc
    if number < 0:
        raise ValueError(f"{name} must be non-negative")
    return number


def _frame_int(value: object, name: str) -> int:
    number = _nonnegative_int(value, name)
    return number


def _require_new_empty_out_dir(out: Path, *, reason: str = "materialization") -> None:
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"{reason} requires a new empty out_dir to avoid stale TrackNet frame/median caches: {out}")


def _dataset_status(*, materialize_frames: bool, hard_negative_plan: bool, train_augmentation: bool) -> str:
    if hard_negative_plan and train_augmentation:
        return "tracknet_dataset_materialized_hard_negative_train_augmented"
    if train_augmentation:
        return "tracknet_dataset_materialized_train_augmented"
    if hard_negative_plan and materialize_frames:
        return "tracknet_dataset_materialized_hard_negative_augmented"
    if hard_negative_plan:
        return "labels_prepared_hard_negative_augmented_frames_not_materialized"
    return "tracknet_dataset_materialized" if materialize_frames else "labels_prepared_frames_not_materialized"


def _extract_frames(video_path: Path, frame_dir: Path, frame_count: int) -> None:
    if frame_count < 0:
        raise ValueError("frame_count must be non-negative")
    if frame_count == 0:
        frame_dir.mkdir(parents=True, exist_ok=True)
        return
    if (frame_dir / "0.png").is_file() and (frame_dir / f"{frame_count - 1}.png").is_file():
        return
    frame_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-i",
            str(video_path),
            "-start_number",
            "0",
            str(frame_dir / "%d.png"),
        ],
        check=True,
    )
    missing = [index for index in (0, frame_count - 1) if not (frame_dir / f"{index}.png").is_file()]
    if missing:
        raise RuntimeError(f"frame extraction did not produce expected frame(s): {missing}")


def _extract_frame_window(video_path: Path, frame_dir: Path, start_frame: int, frame_count: int) -> None:
    if start_frame < 0:
        raise ValueError("start_frame must be non-negative")
    if frame_count < 0:
        raise ValueError("frame_count must be non-negative")
    if frame_count == 0:
        frame_dir.mkdir(parents=True, exist_ok=True)
        return
    if (frame_dir / "0.png").is_file() and (frame_dir / f"{frame_count - 1}.png").is_file():
        return
    frame_dir.mkdir(parents=True, exist_ok=True)
    end_frame = start_frame + frame_count - 1
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-i",
            str(video_path),
            "-vf",
            f"select=between(n\\,{start_frame}\\,{end_frame}),setpts=N/FRAME_RATE/TB",
            "-vsync",
            "0",
            "-start_number",
            "0",
            str(frame_dir / "%d.png"),
        ],
        check=True,
    )
    missing = [index for index in (0, frame_count - 1) if not (frame_dir / f"{index}.png").is_file()]
    if missing:
        raise RuntimeError(f"frame window extraction did not produce expected frame(s): {missing}")


def _write_median(frame_dir: Path, frame_count: int) -> None:
    median_path = frame_dir / "median.npz"
    if median_path.is_file():
        return
    import numpy as np
    from PIL import Image

    samples = []
    stride = max(1, frame_count // 64)
    for frame in range(0, frame_count, stride):
        path = frame_dir / f"{frame}.png"
        if path.is_file():
            samples.append(np.asarray(Image.open(path).convert("RGB"), dtype=np.float32))
    if not samples:
        raise FileNotFoundError(f"no extracted frames found for median: {frame_dir}")
    median_image = np.median(np.stack(samples, axis=0), axis=0).astype("uint8")
    np.savez(median_path, median=median_image)


def _limitations(*, materialize_frames: bool, train_augmentation: bool) -> list[str]:
    limitations = ["This artifact prepares supervised labels only; it does not train or verify BALL."]
    if materialize_frames:
        limitations.append("Frame images were materialized for TrackNetV3 training, but no model has been trained by this builder.")
    else:
        limitations.append(
            "Frame images are not materialized by this builder run; rerun with --materialize-frames and --video clip=path before training."
        )
    if train_augmentation:
        limitations.append("Train-only visual augmentation copies reviewed labels unchanged and does not add new reviewed BALL evidence.")
    limitations.append("Validation uses the split declared by the source YOLO manifest, so held-out metrics must be reported separately after GPU training.")
    return limitations


__all__ = [
    "ARTIFACT_TYPE",
    "MANIFEST_JSON",
    "MANIFEST_MD",
    "TRACKNET_COLUMNS",
    "TrackNetCvatLabel",
    "build_ball_tracknet_cvat_dataset",
    "dense_tracknet_labels_from_cvat",
    "render_ball_tracknet_cvat_dataset_markdown",
]

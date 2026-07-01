"""Build TrackNetV3 label artifacts from reviewed CVAT ball boxes."""

from __future__ import annotations

import csv
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence

from .schemas import CvatVideoAnnotations, validate_artifact_file


ARTIFACT_TYPE = "racketsport_ball_tracknet_cvat_dataset"
MANIFEST_JSON = "ball_tracknet_cvat_dataset_manifest.json"
MANIFEST_MD = "ball_tracknet_cvat_dataset_manifest.md"
TRACKNET_COLUMNS = ("Frame", "Visibility", "X", "Y")
SPLIT_ORDER = ("train", "val", "test")


@dataclass(frozen=True)
class TrackNetCvatLabel:
    frame: int
    visibility: int
    x: float
    y: float
    source: str


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
) -> dict[str, Any]:
    """Write dense TrackNetV3 CSV labels plus JSON/Markdown manifest artifacts."""

    fps = _positive_float(fps, "fps")
    cvat_base = Path(cvat_root)
    yolo_manifest_path = Path(yolo_manifest)
    out = Path(out_dir)
    split_map = _split_map_from_yolo_manifest(yolo_manifest_path)
    selected_clips = _selected_clips(split_map, clips)
    normalized_video_paths = _normalize_video_paths(video_paths)
    hard_negative_spec = _load_hard_negative_plan(hard_negative_plan) if hard_negative_plan is not None else None
    if hard_negative_spec is not None:
        _require_new_empty_out_dir(out)
        hard_negative_context_frames = _nonnegative_int(hard_negative_context_frames, "hard_negative_context_frames")
        hard_negative_repeat = _positive_int(hard_negative_repeat, "hard_negative_repeat")
        _validate_hard_negative_split_roles(hard_negative_spec, split_map=split_map, selected_clips=selected_clips)
    if materialize_frames:
        _require_video_paths(selected_clips, normalized_video_paths)
    out.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": ARTIFACT_TYPE,
        "status": _dataset_status(materialize_frames=materialize_frames, hard_negative_plan=hard_negative_spec is not None),
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
        "leakage_checks": _leakage_checks(split_map, selected_clips),
        "limitations": _limitations(materialize_frames=materialize_frames),
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
            frame_dir = match_dir / "frame" / rally_id
            _write_tracknet_csv(csv_path, labels)
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
        if not ball_boxes:
            labels.append(TrackNetCvatLabel(frame=frame_index, visibility=0, x=0.0, y=0.0, source="reviewed_hidden"))
            continue
        box = ball_boxes[0]
        x, y, width, height = box.bbox_xywh
        labels.append(
            TrackNetCvatLabel(
                frame=frame_index,
                visibility=1,
                x=float(x) + float(width) * 0.5,
                y=float(y) + float(height) * 0.5,
                source="reviewed_cvat_ball_box",
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


def _write_tracknet_csv(path: Path, labels: Sequence[TrackNetCvatLabel]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(TRACKNET_COLUMNS)
        for label in labels:
            writer.writerow([label.frame, label.visibility, f"{label.x:.3f}", f"{label.y:.3f}"])


def _clip_manifest_row(
    *,
    annotations: CvatVideoAnnotations,
    labels: Sequence[TrackNetCvatLabel],
    reviewed_path: Path,
    csv_path: Path,
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
                frame_dir = match_dir / "frame" / rally_id
                window_labels = _slice_labels_for_window(labels, start_frame=start_frame, end_frame=end_frame)
                _write_tracknet_csv(csv_path, window_labels)
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
            )
        )
    return sliced


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
    if "hard_negative_plan" in manifest:
        output_arg = f"--out-dir <new_empty_out_dir_like_{Path(out_dir).name}>"
    return [
        "python scripts/racketsport/build_ball_tracknet_cvat_dataset.py "
        f"--cvat-root {manifest.get('cvat_root')} "
        f"--yolo-manifest {manifest.get('source_yolo_manifest')} "
        f"{output_arg}"
        f"{_hard_negative_cli_args(manifest)}"
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


def _require_new_empty_out_dir(out: Path) -> None:
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"hard-negative materialization requires a new empty out_dir to avoid stale TrackNet frame/median caches: {out}")


def _dataset_status(*, materialize_frames: bool, hard_negative_plan: bool) -> str:
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


def _limitations(*, materialize_frames: bool) -> list[str]:
    limitations = ["This artifact prepares supervised labels only; it does not train or verify BALL."]
    if materialize_frames:
        limitations.append("Frame images were materialized for TrackNetV3 training, but no model has been trained by this builder.")
    else:
        limitations.append(
            "Frame images are not materialized by this builder run; rerun with --materialize-frames and --video clip=path before training."
        )
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

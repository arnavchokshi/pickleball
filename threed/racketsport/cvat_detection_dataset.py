"""Export reviewed CVAT video boxes into YOLO detection datasets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .schemas import CvatVideoAnnotations, validate_artifact_file


DETECTOR_PRESETS: dict[str, dict[str, int]] = {
    "player": {"player": 0},
    "paddle": {"paddle": 0},
    "ball": {"ball": 0},
    "combined": {"player": 0, "paddle": 1, "ball": 2},
}


@dataclass(frozen=True)
class ClipSpec:
    clip_id: str
    video_path: Path
    reviewed_boxes_path: Path


def class_map_for_preset(preset: str) -> dict[str, int]:
    try:
        return dict(DETECTOR_PRESETS[preset])
    except KeyError as exc:
        raise ValueError(f"unsupported detector preset: {preset}") from exc


def yolo_label_line_from_xywh(
    *,
    bbox_xywh: Sequence[float],
    image_width: int,
    image_height: int,
    class_id: int,
) -> str:
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    if len(bbox_xywh) != 4:
        raise ValueError("bbox_xywh must have four values")
    x, y, width, height = [float(value) for value in bbox_xywh]
    x1 = max(0.0, x)
    y1 = max(0.0, y)
    x2 = min(float(image_width), x + width)
    y2 = min(float(image_height), y + height)
    clipped_width = x2 - x1
    clipped_height = y2 - y1
    if clipped_width <= 0.0 or clipped_height <= 0.0:
        raise ValueError("bbox is outside image bounds")
    cx = (x1 + x2) * 0.5 / float(image_width)
    cy = (y1 + y2) * 0.5 / float(image_height)
    norm_w = clipped_width / float(image_width)
    norm_h = clipped_height / float(image_height)
    return f"{int(class_id)} {cx:.6f} {cy:.6f} {norm_w:.6f} {norm_h:.6f}"


def export_cvat_detection_yolo_dataset(
    *,
    clips: Sequence[ClipSpec],
    out_dir: Path,
    class_map: Mapping[str, int],
    split_mode: str,
    val_clips: Sequence[str] = (),
    val_every: int = 5,
    frame_stride: int = 1,
    jpeg_quality: int = 95,
) -> dict[str, Any]:
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for YOLO dataset export") from exc
    if not class_map:
        raise ValueError("class_map must not be empty")
    if sorted(class_map.values()) != list(range(len(class_map))):
        raise ValueError("class_map ids must be contiguous starting at 0")
    if split_mode not in {"alternating", "by_clip"}:
        raise ValueError(f"unsupported split mode: {split_mode}")
    if val_every <= 1 and split_mode == "alternating":
        raise ValueError("val_every must be > 1 for alternating split")
    if frame_stride <= 0:
        raise ValueError("frame_stride must be positive")

    for split in ("train", "val"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    selected_labels = set(class_map)
    val_clip_set = set(val_clips)
    image_count = 0
    label_count = 0
    rows: list[dict[str, Any]] = []
    label_counts_by_name = {label: 0 for label in sorted(class_map, key=lambda name: class_map[name])}
    split_image_counts = {"train": 0, "val": 0}
    split_label_counts = {"train": 0, "val": 0}

    for clip in clips:
        annotations = validate_artifact_file("cvat_video_annotations", clip.reviewed_boxes_path)
        if not isinstance(annotations, CvatVideoAnnotations):
            raise ValueError(f"reviewed boxes artifact did not parse as CvatVideoAnnotations: {clip.reviewed_boxes_path}")
        boxes_by_frame = {
            frame.frame_index: [box for box in frame.boxes if box.label in selected_labels]
            for frame in annotations.frames
        }
        cap = cv2.VideoCapture(str(clip.video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"cannot open video: {clip.video_path}")
        try:
            frame_index = 0
            while frame_index < len(annotations.frames):
                ok, frame = cap.read()
                if not ok:
                    break
                boxes = boxes_by_frame.get(frame_index, [])
                if boxes and frame_index % frame_stride == 0:
                    height, width = frame.shape[:2]
                    split = _split_for_frame(
                        clip_id=clip.clip_id,
                        frame_index=frame_index,
                        split_mode=split_mode,
                        val_clips=val_clip_set,
                        val_every=val_every,
                    )
                    stem = f"{_safe_token(clip.clip_id)}_{frame_index:06d}"
                    image_path = out_dir / "images" / split / f"{stem}.jpg"
                    label_path = out_dir / "labels" / split / f"{stem}.txt"
                    lines: list[str] = []
                    row_label_counts = {label: 0 for label in sorted(class_map, key=lambda name: class_map[name])}
                    for box in boxes:
                        try:
                            lines.append(
                                yolo_label_line_from_xywh(
                                    bbox_xywh=box.bbox_xywh,
                                    image_width=width,
                                    image_height=height,
                                    class_id=class_map[box.label],
                                )
                            )
                        except ValueError:
                            continue
                        label_counts_by_name[box.label] += 1
                        row_label_counts[box.label] += 1
                    if lines:
                        ok_write = cv2.imwrite(str(image_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
                        if not ok_write:
                            raise RuntimeError(f"failed to write image: {image_path}")
                        label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
                        image_count += 1
                        label_count += len(lines)
                        split_image_counts[split] += 1
                        split_label_counts[split] += len(lines)
                        rows.append(
                            {
                                "clip_id": clip.clip_id,
                                "frame_index": frame_index,
                                "split": split,
                                "image_path": str(image_path),
                                "label_path": str(label_path),
                                "label_count": len(lines),
                                "label_counts_by_name": row_label_counts,
                            }
                        )
                frame_index += 1
        finally:
            cap.release()

    data_yaml = out_dir / "data.yaml"
    names_by_id = {class_id: label for label, class_id in class_map.items()}
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {out_dir.resolve()}",
                "train: images/train",
                "val: images/val",
                f"nc: {len(class_map)}",
                "names:",
                *[f"  {class_id}: {names_by_id[class_id]}" for class_id in sorted(names_by_id)],
                "",
            ]
        ),
        encoding="utf-8",
    )
    summary = {
        "schema_version": 1,
        "artifact_type": "racketsport_cvat_detection_yolo_dataset",
        "out_dir": str(out_dir),
        "data_yaml": str(data_yaml),
        "class_map": dict(sorted(class_map.items(), key=lambda item: item[1])),
        "image_count": image_count,
        "label_count": label_count,
        "label_counts_by_name": label_counts_by_name,
        "split_mode": split_mode,
        "val_clips": sorted(val_clip_set),
        "val_every": val_every,
        "frame_stride": frame_stride,
        "split_image_counts": split_image_counts,
        "split_label_counts": split_label_counts,
        "rows": rows,
    }
    (out_dir / "manifest.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _split_for_frame(
    *,
    clip_id: str,
    frame_index: int,
    split_mode: str,
    val_clips: set[str],
    val_every: int,
) -> str:
    if split_mode == "alternating":
        return "val" if frame_index % val_every == 0 else "train"
    if split_mode == "by_clip":
        if not val_clips:
            raise ValueError("by_clip split requires at least one --val-clip")
        return "val" if clip_id in val_clips else "train"
    raise ValueError(f"unsupported split mode: {split_mode}")


def _safe_token(value: str) -> str:
    token = "".join(char if char.isalnum() else "_" for char in str(value).strip().lower())
    token = "_".join(part for part in token.split("_") if part)
    if not token:
        raise ValueError("empty token")
    return token


__all__ = [
    "ClipSpec",
    "class_map_for_preset",
    "export_cvat_detection_yolo_dataset",
    "yolo_label_line_from_xywh",
]

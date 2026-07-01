"""Deterministic sanity checks for reviewed CVAT video annotations."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any

from .schemas import CvatVideoAnnotations, CvatVideoBox


def build_annotation_sanity_report(
    annotations: CvatVideoAnnotations,
    *,
    expected_players: int = 4,
    long_gap_frames: int = 30,
    jump_factor: float = 8.0,
) -> dict[str, Any]:
    if expected_players < 0:
        raise ValueError("expected_players must be nonnegative")
    if long_gap_frames < 0:
        raise ValueError("long_gap_frames must be nonnegative")
    if jump_factor <= 0.0:
        raise ValueError("jump_factor must be positive")

    width, height = annotations.task.original_size
    labels: dict[str, dict[str, Any]] = {}
    warnings: list[str] = []
    player_counts: Counter[int] = Counter()
    boxes_by_label_track: dict[str, dict[int, list[CvatVideoBox]]] = defaultdict(lambda: defaultdict(list))

    for frame in annotations.frames:
        player_count = sum(1 for box in frame.boxes if box.label == "player")
        player_counts[player_count] += 1
        for box in frame.boxes:
            boxes_by_label_track[box.label][box.track_id].append(box)

    for label, tracks in sorted(boxes_by_label_track.items()):
        label_report: dict[str, Any] = {
            "visible_box_count": 0,
            "track_count": len(tracks),
            "out_of_bounds_box_count": 0,
            "min_bbox_area_px2": None,
            "max_bbox_area_px2": None,
            "tracks": {},
        }
        areas: list[float] = []
        for track_id, boxes in sorted(tracks.items()):
            sorted_boxes = sorted(boxes, key=lambda item: item.frame_index)
            track_report = _track_report(
                sorted_boxes,
                image_width=width,
                image_height=height,
                long_gap_frames=long_gap_frames,
                jump_factor=jump_factor,
            )
            label_report["tracks"][str(track_id)] = track_report
            label_report["visible_box_count"] += len(sorted_boxes)
            label_report["out_of_bounds_box_count"] += track_report["out_of_bounds_box_count"]
            areas.extend(track_report["bbox_areas_px2"])
            if track_report["long_gaps"]:
                warnings.append(f"{label} track {track_id} has {len(track_report['long_gaps'])} long visible gaps")
            if track_report["weird_jumps"]:
                warnings.append(f"{label} track {track_id} has {len(track_report['weird_jumps'])} large center jumps")
            if track_report["out_of_bounds_box_count"]:
                warnings.append(f"{label} track {track_id} has {track_report['out_of_bounds_box_count']} out-of-bounds boxes")
        if areas:
            label_report["min_bbox_area_px2"] = min(areas)
            label_report["max_bbox_area_px2"] = max(areas)
        for track_report in label_report["tracks"].values():
            track_report.pop("bbox_areas_px2", None)
        labels[label] = label_report

    frames_with_expected = player_counts.get(expected_players, 0)
    if expected_players and frames_with_expected != len(annotations.frames):
        warnings.append(
            f"player count equals expected {expected_players} on {frames_with_expected}/{len(annotations.frames)} frames"
        )

    return {
        "schema_version": 1,
        "artifact_type": "racketsport_cvat_annotation_sanity",
        "clip_id": annotations.clip_id,
        "source_path": annotations.source_path,
        "frame_count": len(annotations.frames),
        "task_size": annotations.task.size,
        "task_stop_frame": annotations.task.stop_frame,
        "outside_box_count": annotations.summary.outside_box_count,
        "expected_players": expected_players,
        "frames_with_expected_players": frames_with_expected,
        "player_frame_count_histogram": {str(key): player_counts[key] for key in sorted(player_counts)},
        "visible_box_count_by_label": annotations.summary.visible_box_count_by_label,
        "track_count_by_label": annotations.summary.track_count_by_label,
        "labels": labels,
        "warnings": warnings,
    }


def _track_report(
    boxes: list[CvatVideoBox],
    *,
    image_width: int,
    image_height: int,
    long_gap_frames: int,
    jump_factor: float,
) -> dict[str, Any]:
    long_gaps: list[dict[str, int]] = []
    weird_jumps: list[dict[str, float | int]] = []
    out_of_bounds_count = 0
    areas: list[float] = []
    previous: CvatVideoBox | None = None
    for box in boxes:
        x, y, width, height = box.bbox_xywh
        areas.append(float(width) * float(height))
        if x < 0.0 or y < 0.0 or x + width > image_width or y + height > image_height:
            out_of_bounds_count += 1
        if previous is not None:
            gap = box.frame_index - previous.frame_index - 1
            if gap >= long_gap_frames:
                long_gaps.append({"from_frame": previous.frame_index, "to_frame": box.frame_index, "gap_frames": gap})
            jump_px = _center_distance(previous, box)
            scale = max(_bbox_diagonal(previous), _bbox_diagonal(box), 1.0)
            if jump_px > scale * jump_factor:
                weird_jumps.append(
                    {
                        "from_frame": previous.frame_index,
                        "to_frame": box.frame_index,
                        "jump_px": round(jump_px, 3),
                        "scale_px": round(scale, 3),
                    }
                )
        previous = box

    return {
        "visible_box_count": len(boxes),
        "first_visible_frame": boxes[0].frame_index if boxes else None,
        "last_visible_frame": boxes[-1].frame_index if boxes else None,
        "long_gaps": long_gaps,
        "weird_jumps": weird_jumps,
        "out_of_bounds_box_count": out_of_bounds_count,
        "bbox_areas_px2": areas,
    }


def _center_distance(a: CvatVideoBox, b: CvatVideoBox) -> float:
    ax, ay, aw, ah = a.bbox_xywh
    bx, by, bw, bh = b.bbox_xywh
    return math.hypot((ax + aw * 0.5) - (bx + bw * 0.5), (ay + ah * 0.5) - (by + bh * 0.5))


def _bbox_diagonal(box: CvatVideoBox) -> float:
    _, _, width, height = box.bbox_xywh
    return math.hypot(float(width), float(height))


__all__ = ["build_annotation_sanity_report"]

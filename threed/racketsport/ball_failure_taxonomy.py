"""Frame-level BALL failure taxonomy from reviewed CVAT labels and ball tracks."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from .ball_overlay import load_ball_track
from .schemas import BallFrame, CvatVideoAnnotations


ARTIFACT_TYPE = "racketsport_ball_failure_taxonomy"
ACTIONABLE_FAILURE_CLASSES = ("visible_miss", "visible_mislocalized", "hidden_false_positive")
PRIMARY_CLASS_PRIORITY = (
    "visible_miss",
    "visible_mislocalized",
    "hidden_false_positive",
    "visible_hit",
    "hidden_true_negative",
)
BLOCKED_CLASSES = {
    "likely_line_glint": "requires court-line geometry or image evidence",
    "high_blur_or_fast_motion": "requires reviewed blur fields or frame-difference evidence",
}


def build_ball_failure_taxonomy(
    *,
    ball_track_path: str | Path,
    cvat_labels_path: str | Path,
    candidate_name: str,
    f1_radius_px: float = 20.0,
    teleport_px_per_frame: float = 160.0,
    max_jump_gap_frames: int = 3,
) -> dict[str, Any]:
    """Classify every evaluated frame into BALL failure buckets.

    The taxonomy is intentionally evidence-limited. It uses reviewed CVAT ball
    boxes, reviewed player/paddle boxes, and the candidate ball track. It does
    not guess court-line glints or blur unless those inputs are present.
    """

    if f1_radius_px <= 0.0 or not math.isfinite(float(f1_radius_px)):
        raise ValueError("f1_radius_px must be positive and finite")
    if teleport_px_per_frame <= 0.0 or not math.isfinite(float(teleport_px_per_frame)):
        raise ValueError("teleport_px_per_frame must be positive and finite")
    if max_jump_gap_frames < 1:
        raise ValueError("max_jump_gap_frames must be >= 1")

    track = load_ball_track(ball_track_path)
    labels = _load_cvat_labels(cvat_labels_path)
    samples_by_index = _track_samples_by_frame_index(track)
    evaluated_frame_count = min(max(samples_by_index, default=-1) + 1, len(labels.frames))
    frames = labels.frames[:evaluated_frame_count]
    ball_sizes = [
        _ball_size_px(box)
        for frame in frames
        for box in frame.boxes
        if _is_label(box, "ball")
    ]
    far_size_threshold = _nearest_percentile(ball_sizes, 25.0)
    near_size_threshold = _nearest_percentile(ball_sizes, 75.0)
    speed_by_index = _prediction_speed_by_index(
        samples_by_index,
        evaluated_frame_count=evaluated_frame_count,
        max_jump_gap_frames=max_jump_gap_frames,
    )

    frame_rows: list[dict[str, Any]] = []
    class_counts: Counter[str] = Counter()
    actionable_counts: Counter[str] = Counter()
    for frame in frames:
        frame_index = int(frame.frame_index)
        sample = samples_by_index.get(frame_index)
        ball_boxes = [box for box in frame.boxes if _is_label(box, "ball")]
        distractor_boxes = [box for box in frame.boxes if _is_label(box, "player") or _is_label(box, "paddle")]
        row = _classify_frame(
            frame_index=frame_index,
            sample=sample,
            ball_boxes=ball_boxes,
            distractor_boxes=distractor_boxes,
            f1_radius_px=float(f1_radius_px),
            far_size_threshold=far_size_threshold,
            near_size_threshold=near_size_threshold,
            speed_px_per_frame=speed_by_index.get(frame_index),
            teleport_px_per_frame=float(teleport_px_per_frame),
        )
        frame_rows.append(row)
        class_counts.update(row["classes"])
        actionable_counts.update(cls for cls in row["classes"] if cls in ACTIONABLE_FAILURE_CLASSES)

    return {
        "schema_version": 1,
        "artifact_type": ARTIFACT_TYPE,
        "status": "TESTED-ON-REAL-DATA",
        "ball_verified": False,
        "clip": labels.clip_id,
        "candidate": candidate_name,
        "ball_track_path": str(ball_track_path),
        "cvat_labels_path": str(cvat_labels_path),
        "f1_radius_px": float(f1_radius_px),
        "teleport_px_per_frame": float(teleport_px_per_frame),
        "max_jump_gap_frames": int(max_jump_gap_frames),
        "evaluated_frame_count": evaluated_frame_count,
        "size_bins": {
            "size_px": "max(width, height) of reviewed ball box",
            "far_camera_threshold_px": far_size_threshold,
            "near_camera_threshold_px": near_size_threshold,
        },
        "summary": {
            "class_counts": dict(sorted(class_counts.items())),
            "actionable_failure_counts": dict(sorted(actionable_counts.items())),
            "worst_visible_mislocalized": _top_frames(frame_rows, primary_class="visible_mislocalized"),
            "highest_conf_hidden_false_positive": _top_frames(
                frame_rows,
                primary_class="hidden_false_positive",
                key="prediction_conf",
            ),
        },
        "blocked_classes": dict(BLOCKED_CLASSES),
        "frames": frame_rows,
        "notes": [
            "This taxonomy scores an existing candidate against reviewed CVAT labels only; it does not run inference or verify BALL.",
            "near_camera/far_camera are reviewed ball-size bins, not real 3D depth.",
            "likely_player_or_paddle is based on prediction point overlap with reviewed player/paddle boxes.",
        ],
    }


def write_ball_failure_taxonomy(
    *,
    ball_track_path: str | Path,
    cvat_labels_path: str | Path,
    candidate_name: str,
    out_json: str | Path,
    out_markdown: str | Path | None = None,
    f1_radius_px: float = 20.0,
    teleport_px_per_frame: float = 160.0,
    max_jump_gap_frames: int = 3,
) -> dict[str, Any]:
    taxonomy = build_ball_failure_taxonomy(
        ball_track_path=ball_track_path,
        cvat_labels_path=cvat_labels_path,
        candidate_name=candidate_name,
        f1_radius_px=f1_radius_px,
        teleport_px_per_frame=teleport_px_per_frame,
        max_jump_gap_frames=max_jump_gap_frames,
    )
    out = Path(out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(taxonomy, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if out_markdown is not None:
        Path(out_markdown).write_text(render_ball_failure_taxonomy_markdown(taxonomy), encoding="utf-8")
    return taxonomy


def render_ball_failure_taxonomy_markdown(taxonomy: Mapping[str, Any]) -> str:
    summary = taxonomy.get("summary", {})
    class_counts = summary.get("class_counts", {}) if isinstance(summary, Mapping) else {}
    actionable_counts = summary.get("actionable_failure_counts", {}) if isinstance(summary, Mapping) else {}
    lines = [
        "# BALL Failure Taxonomy",
        "",
        f"Candidate: `{taxonomy.get('candidate')}`",
        f"Clip: `{taxonomy.get('clip')}`",
        "",
        "BALL is not verified by this report. It classifies failure modes for an existing candidate.",
        "",
        "## Actionable Failures",
        "",
    ]
    if isinstance(actionable_counts, Mapping) and actionable_counts:
        for key, value in actionable_counts.items():
            lines.append(f"- `{key}`: `{value}`")
    else:
        lines.append("- none")
    lines.extend(["", "## All Classes", ""])
    if isinstance(class_counts, Mapping):
        for key, value in class_counts.items():
            lines.append(f"- `{key}`: `{value}`")
    blocked = taxonomy.get("blocked_classes")
    if isinstance(blocked, Mapping):
        lines.extend(["", "## Unavailable Classes", ""])
        for key, reason in blocked.items():
            lines.append(f"- `{key}`: {reason}")
    lines.append("")
    return "\n".join(lines)


def _load_cvat_labels(path: str | Path) -> CvatVideoAnnotations:
    labels_path = Path(path)
    if not labels_path.is_file():
        raise ValueError(f"missing CVAT labels file: {labels_path}")
    try:
        payload = json.loads(labels_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid CVAT labels JSON: {labels_path}: {exc}") from exc
    try:
        return CvatVideoAnnotations.model_validate(payload)
    except Exception as exc:
        raise ValueError(f"invalid CVAT labels schema: {labels_path}: {exc}") from exc


def _track_samples_by_frame_index(track: Any) -> dict[int, BallFrame]:
    return {int(round(float(frame.t) * float(track.fps))): frame for frame in track.frames}


def _classify_frame(
    *,
    frame_index: int,
    sample: BallFrame | None,
    ball_boxes: Sequence[Any],
    distractor_boxes: Sequence[Any],
    f1_radius_px: float,
    far_size_threshold: float | None,
    near_size_threshold: float | None,
    speed_px_per_frame: float | None,
    teleport_px_per_frame: float,
) -> dict[str, Any]:
    classes: list[str] = []
    error_px: float | None = None
    nearest_label_center: list[float] | None = None
    nearest_label_size_px: float | None = None
    has_visible_prediction = sample is not None and bool(sample.visible)
    prediction_xy = [float(sample.xy[0]), float(sample.xy[1])] if sample is not None else None
    prediction_conf = float(sample.conf) if sample is not None else None

    if ball_boxes:
        if has_visible_prediction and sample is not None:
            nearest_box, error_px = _nearest_ball_box(sample.xy, ball_boxes)
            nearest_label_center = list(_box_center(nearest_box))
            nearest_label_size_px = _ball_size_px(nearest_box)
            classes.append("visible_hit" if error_px <= f1_radius_px else "visible_mislocalized")
        else:
            nearest_box = ball_boxes[0]
            nearest_label_center = list(_box_center(nearest_box))
            nearest_label_size_px = _ball_size_px(nearest_box)
            classes.append("visible_miss")

        if nearest_label_size_px is not None:
            if far_size_threshold is not None and nearest_label_size_px <= far_size_threshold:
                classes.append("far_camera")
            if near_size_threshold is not None and nearest_label_size_px >= near_size_threshold:
                classes.append("near_camera")
        if any(bool(box.occluded) for box in ball_boxes):
            classes.append("occluded_or_contact")
    elif has_visible_prediction and sample is not None:
        classes.append("hidden_false_positive")
        if any(_point_in_box(sample.xy, box.bbox_xyxy) for box in distractor_boxes):
            classes.append("likely_player_or_paddle")
    else:
        classes.append("hidden_true_negative")

    if speed_px_per_frame is not None and speed_px_per_frame > teleport_px_per_frame:
        classes.append("high_blur_or_fast_motion")

    return {
        "frame_index": frame_index,
        "primary_class": _primary_class(classes),
        "classes": classes,
        "prediction_xy": prediction_xy,
        "prediction_conf": prediction_conf,
        "error_px": error_px,
        "nearest_label_center": nearest_label_center,
        "nearest_label_size_px": nearest_label_size_px,
        "speed_px_per_frame": speed_px_per_frame,
    }


def _nearest_ball_box(point: Sequence[float], boxes: Sequence[Any]) -> tuple[Any, float]:
    ranked = [(box, _distance(point, _box_center(box))) for box in boxes]
    return min(ranked, key=lambda item: item[1])


def _box_center(box: Any) -> tuple[float, float]:
    x1, y1, x2, y2 = box.bbox_xyxy
    return ((float(x1) + float(x2)) / 2.0, (float(y1) + float(y2)) / 2.0)


def _ball_size_px(box: Any) -> float:
    _, _, width, height = box.bbox_xywh
    return max(float(width), float(height))


def _is_label(box: Any, label: str) -> bool:
    return str(box.label).strip().lower() == label


def _point_in_box(point: Sequence[float], box_xyxy: Sequence[float]) -> bool:
    x, y = float(point[0]), float(point[1])
    x1, y1, x2, y2 = (float(value) for value in box_xyxy)
    return x1 <= x <= x2 and y1 <= y <= y2


def _distance(a: Sequence[float], b: Sequence[float]) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def _nearest_percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    index = int(round((len(ordered) - 1) * float(percentile) / 100.0))
    return ordered[max(0, min(len(ordered) - 1, index))]


def _prediction_speed_by_index(
    samples_by_index: Mapping[int, BallFrame],
    *,
    evaluated_frame_count: int,
    max_jump_gap_frames: int,
) -> dict[int, float]:
    visible = sorted(
        (index, frame)
        for index, frame in samples_by_index.items()
        if index < evaluated_frame_count and frame.visible
    )
    speeds: dict[int, float] = {}
    for (prev_index, prev_frame), (index, frame) in zip(visible, visible[1:]):
        gap = index - prev_index
        if 0 < gap <= max_jump_gap_frames:
            speeds[index] = _distance(prev_frame.xy, frame.xy) / float(gap)
    return speeds


def _primary_class(classes: Sequence[str]) -> str:
    for candidate in PRIMARY_CLASS_PRIORITY:
        if candidate in classes:
            return candidate
    return classes[0] if classes else "unknown"


def _top_frames(frame_rows: Sequence[Mapping[str, Any]], *, primary_class: str, key: str = "error_px", limit: int = 20) -> list[dict[str, Any]]:
    rows = [row for row in frame_rows if row.get("primary_class") == primary_class and row.get(key) is not None]
    rows.sort(key=lambda row: float(row.get(key) or 0.0), reverse=True)
    return [
        {
            "frame_index": int(row["frame_index"]),
            "value": float(row[key]),
            "prediction_xy": row.get("prediction_xy"),
            "classes": row.get("classes", []),
        }
        for row in rows[:limit]
    ]

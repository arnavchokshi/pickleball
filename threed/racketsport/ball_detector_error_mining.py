"""Mine reviewed detector errors into BALL fine-tuning iteration artifacts."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .ball_overlay import load_ball_track
from .schemas import BallFrame, CvatVideoAnnotations


@dataclass(frozen=True)
class DetectorTrackInput:
    clip: str
    candidate: str
    path: Path
    split: str


def mine_ball_detector_errors(
    *,
    cvat_root: str | Path,
    tracks: Sequence[DetectorTrackInput],
    out_json: str | Path,
    radius_px: float = 20.0,
) -> dict[str, Any]:
    """Write a hard-negative/error-mining plan from reviewed labels and model tracks."""

    radius = _positive_float(radius_px, "radius_px")
    cvat_base = Path(cvat_root)
    out_path = Path(out_json)
    normalized_tracks = _normalize_tracks(tracks)

    clip_reports: dict[str, Any] = {}
    train_clips: list[str] = []
    validation_clips: list[str] = []
    totals = {
        "total_clip_count": 0,
        "total_visible_label_count": 0,
        "total_hidden_label_count": 0,
        "total_true_positive_count_at_radius": 0,
        "total_false_negative_count_at_radius": 0,
        "total_false_positive_count": 0,
        "total_visible_miss_count": 0,
        "total_visible_mislocalized_count": 0,
    }

    for track_input in normalized_tracks:
        report = _mine_clip_errors(cvat_base=cvat_base, track_input=track_input, radius_px=radius)
        clip_reports[track_input.clip] = report
        totals["total_clip_count"] += 1
        metrics = report["metrics"]
        totals["total_visible_label_count"] += int(metrics["visible_label_count"])
        totals["total_hidden_label_count"] += int(metrics["hidden_label_count"])
        totals["total_true_positive_count_at_radius"] += int(metrics["true_positive_count_at_radius"])
        totals["total_false_negative_count_at_radius"] += int(metrics["false_negative_count_at_radius"])
        totals["total_false_positive_count"] += int(metrics["false_positive_count"])
        totals["total_visible_miss_count"] += int(metrics["visible_miss_count"])
        totals["total_visible_mislocalized_count"] += int(metrics["visible_mislocalized_count"])
        if track_input.split == "train":
            train_clips.append(track_input.clip)
        else:
            validation_clips.append(track_input.clip)

    plan = {
        "schema_version": 1,
        "artifact_type": "racketsport_ball_hard_negative_iteration_plan",
        "status": "TESTED-ON-REAL-DATA",
        "ball_verified": False,
        "not_ground_truth": True,
        "promotion_claimed": False,
        "cvat_root": str(cvat_base),
        "radius_px": radius,
        "train_clips": train_clips,
        "validation_clips": validation_clips,
        "clips": clip_reports,
        "summary": _summary_with_rates(totals),
        "notes": [
            "This artifact is mined from reviewed CVAT labels and existing detector outputs only.",
            "It is fine-tuning input and error evidence, not a BALL gate pass or ground-truth ball track.",
            "Validation clips are listed for diagnostics only; the TrackNet dataset builder must not train on validation_only_do_not_train clips.",
        ],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return plan


def _mine_clip_errors(*, cvat_base: Path, track_input: DetectorTrackInput, radius_px: float) -> dict[str, Any]:
    reviewed_path = cvat_base / track_input.clip / "reviewed_boxes.json"
    labels = _load_cvat_labels(reviewed_path)
    track = load_ball_track(track_input.path)
    samples = _samples_by_frame_index(track.frames, fps=float(track.fps))
    frame_count = _label_frame_count(labels)
    centers_by_frame = _ball_centers_by_frame(labels)

    true_positive_frames: list[dict[str, Any]] = []
    visible_miss_frames: list[int] = []
    visible_mislocalized_frames: list[dict[str, Any]] = []
    hidden_false_positive_frames: list[dict[str, Any]] = []

    for frame_index in range(frame_count):
        centers = centers_by_frame.get(frame_index, [])
        sample = samples.get(frame_index)
        has_prediction = sample is not None and bool(sample.visible)
        if centers:
            if not has_prediction:
                visible_miss_frames.append(frame_index)
                continue
            assert sample is not None
            distance = min(_distance(sample.xy, center) for center in centers)
            row = {
                "frame": frame_index,
                "distance_px": _round3(distance),
                "conf": _round3(float(sample.conf)),
                "xy": [_round3(float(sample.xy[0])), _round3(float(sample.xy[1]))],
                "label_xy": [_round3(float(centers[0][0])), _round3(float(centers[0][1]))],
            }
            if distance <= radius_px:
                true_positive_frames.append(row)
            else:
                visible_mislocalized_frames.append(row)
            continue
        if has_prediction:
            assert sample is not None
            hidden_false_positive_frames.append(
                {
                    "frame": frame_index,
                    "conf": _round3(float(sample.conf)),
                    "xy": [_round3(float(sample.xy[0])), _round3(float(sample.xy[1]))],
                }
            )

    visible_label_count = len(centers_by_frame)
    hidden_label_count = max(0, frame_count - visible_label_count)
    false_negative_count = len(visible_miss_frames) + len(visible_mislocalized_frames)
    metrics = {
        "frame_count": frame_count,
        "track_frame_count": max(samples, default=-1) + 1,
        "visible_label_count": visible_label_count,
        "hidden_label_count": hidden_label_count,
        "true_positive_count_at_radius": len(true_positive_frames),
        "false_negative_count_at_radius": false_negative_count,
        "false_positive_count": len(hidden_false_positive_frames),
        "visible_miss_count": len(visible_miss_frames),
        "visible_mislocalized_count": len(visible_mislocalized_frames),
        "recall_at_radius": _ratio(len(true_positive_frames), visible_label_count),
        "hidden_false_positive_rate": _ratio(len(hidden_false_positive_frames), hidden_label_count),
    }
    split_role = "train_hard_negative_candidate" if track_input.split == "train" else "validation_only_do_not_train"
    return {
        "clip": track_input.clip,
        "candidate": track_input.candidate,
        "split": track_input.split,
        "split_role": split_role,
        "track_path": str(track_input.path),
        "reviewed_boxes": str(reviewed_path),
        "source": str(track.source),
        "fps": float(track.fps),
        "radius_px": radius_px,
        "metrics": metrics,
        "hard_negative_hidden_fp_frames": hidden_false_positive_frames,
        "hard_negative_hidden_fp_ranges": _hidden_fp_ranges(hidden_false_positive_frames),
        "visible_miss_frames": visible_miss_frames,
        "visible_miss_ranges": _plain_ranges(visible_miss_frames),
        "visible_mislocalized_frames": visible_mislocalized_frames,
        "visible_mislocalized_ranges": _distance_ranges(visible_mislocalized_frames),
        "true_positive_frames": true_positive_frames,
    }


def _load_cvat_labels(path: Path) -> CvatVideoAnnotations:
    if not path.is_file():
        raise FileNotFoundError(f"missing reviewed CVAT labels: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return CvatVideoAnnotations.model_validate(payload)


def _label_frame_count(labels: CvatVideoAnnotations) -> int:
    max_frame = max((frame.frame_index for frame in labels.frames), default=-1) + 1
    return max(int(labels.task.size), max_frame)


def _samples_by_frame_index(frames: Sequence[BallFrame], *, fps: float) -> dict[int, BallFrame]:
    return {int(round(float(frame.t) * fps)): frame for frame in frames}


def _ball_centers_by_frame(labels: CvatVideoAnnotations) -> dict[int, list[tuple[float, float]]]:
    centers: dict[int, list[tuple[float, float]]] = {}
    for frame in labels.frames:
        ball_centers: list[tuple[float, float]] = []
        for box in frame.boxes:
            if box.label.strip().lower() != "ball":
                continue
            x1, y1, x2, y2 = box.bbox_xyxy
            ball_centers.append(((float(x1) + float(x2)) * 0.5, (float(y1) + float(y2)) * 0.5))
        if ball_centers:
            centers[int(frame.frame_index)] = ball_centers
    return centers


def _hidden_fp_ranges(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return _ranges_with_metric(rows, metric_name="conf", output_name="max_conf")


def _distance_ranges(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return _ranges_with_metric(rows, metric_name="distance_px", output_name="max_distance_px")


def _plain_ranges(frames: Sequence[int]) -> list[dict[str, int]]:
    ranges: list[dict[str, int]] = []
    for start, end, indexes in _contiguous_groups(frames):
        ranges.append({"start": start, "end": end, "count": len(indexes)})
    return ranges


def _ranges_with_metric(
    rows: Sequence[Mapping[str, Any]],
    *,
    metric_name: str,
    output_name: str,
) -> list[dict[str, Any]]:
    by_frame = {int(row["frame"]): row for row in rows}
    ranges: list[dict[str, Any]] = []
    for start, end, indexes in _contiguous_groups(by_frame):
        max_value = max(float(by_frame[index][metric_name]) for index in indexes)
        ranges.append({"start": start, "end": end, "count": len(indexes), output_name: _round3(max_value)})
    return ranges


def _contiguous_groups(frames: Sequence[int] | Mapping[int, Any]) -> list[tuple[int, int, list[int]]]:
    indexes = sorted(int(index) for index in (frames.keys() if isinstance(frames, Mapping) else frames))
    if not indexes:
        return []
    groups: list[tuple[int, int, list[int]]] = []
    start = previous = indexes[0]
    current = [indexes[0]]
    for index in indexes[1:]:
        if index == previous + 1:
            current.append(index)
            previous = index
            continue
        groups.append((start, previous, current))
        start = previous = index
        current = [index]
    groups.append((start, previous, current))
    return groups


def _normalize_tracks(tracks: Sequence[DetectorTrackInput]) -> list[DetectorTrackInput]:
    if not tracks:
        raise ValueError("at least one detector track is required")
    normalized: list[DetectorTrackInput] = []
    seen: set[str] = set()
    for item in tracks:
        clip = item.clip.strip()
        candidate = item.candidate.strip()
        split = item.split.strip().lower()
        if not clip:
            raise ValueError("track clip must be non-empty")
        if clip in seen:
            raise ValueError(f"duplicate detector track clip: {clip}")
        seen.add(clip)
        if not candidate:
            raise ValueError(f"track candidate must be non-empty for clip {clip}")
        if split not in {"train", "val", "validation", "test", "eval"}:
            raise ValueError(f"split must be train/val/test/eval for clip {clip}: {split}")
        normalized_split = "val" if split in {"validation", "eval"} else split
        path = Path(item.path)
        if not path.is_file():
            raise FileNotFoundError(f"missing detector track for {clip}: {path}")
        normalized.append(DetectorTrackInput(clip=clip, candidate=candidate, path=path, split=normalized_split))
    return normalized


def _summary_with_rates(totals: Mapping[str, int]) -> dict[str, Any]:
    visible = int(totals["total_visible_label_count"])
    hidden = int(totals["total_hidden_label_count"])
    true_positive = int(totals["total_true_positive_count_at_radius"])
    false_positive = int(totals["total_false_positive_count"])
    false_negative = int(totals["total_false_negative_count_at_radius"])
    summary = dict(totals)
    summary["micro_recall_at_radius"] = _ratio(true_positive, visible)
    summary["micro_hidden_false_positive_rate"] = _ratio(false_positive, hidden)
    summary["micro_precision_at_radius"] = _ratio(true_positive, true_positive + false_positive)
    summary["micro_f1_at_radius"] = _f1(summary["micro_precision_at_radius"], summary["micro_recall_at_radius"])
    summary["blocked_reason"] = (
        "detector_errors_mined_for_fine_tuning; this is not a gate pass and must be followed by real training plus held-out benchmark"
    )
    return summary


def _distance(xy: Sequence[float], center: Sequence[float]) -> float:
    return math.hypot(float(xy[0]) - float(center[0]), float(xy[1]) - float(center[1]))


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator


def _f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None or precision + recall <= 0.0:
        return None
    return 2.0 * precision * recall / (precision + recall)


def _positive_float(value: object, name: str) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be positive") from exc
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be positive")
    return number


def _round3(value: float) -> float:
    return round(float(value), 3)


__all__ = ["DetectorTrackInput", "mine_ball_detector_errors"]

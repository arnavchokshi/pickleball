"""Benchmark ball-track candidates against sparse human review labels.

The reviewed clicks are labels only.  This module never feeds them back into a
tracker; it measures whether a candidate would have found the target ball on
future clips without per-video human correction.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .ball_identity_filter import BallClickReview, load_ball_click_review
from .ball_overlay import load_ball_track
from .schemas import BallFrame, BallTrack


@dataclass(frozen=True)
class BallCandidate:
    clip: str
    name: str
    path: Path
    category: str = "generalizable"


def benchmark_ball_track_candidate(
    *,
    ball_track_path: str | Path,
    clicks_path: str | Path,
    candidate_name: str,
    category: str = "generalizable",
    hit_radius_px: float = 36.0,
    teleport_px_per_frame: float = 160.0,
    max_jump_gap_frames: int = 3,
) -> dict[str, Any]:
    """Score one candidate track against sparse click labels."""

    if hit_radius_px < 0.0:
        raise ValueError("hit_radius_px must be >= 0")
    if teleport_px_per_frame <= 0.0 or not math.isfinite(teleport_px_per_frame):
        raise ValueError("teleport_px_per_frame must be > 0")
    if max_jump_gap_frames < 1:
        raise ValueError("max_jump_gap_frames must be >= 1")

    track = load_ball_track(ball_track_path)
    review = load_ball_click_review(clicks_path)
    samples_by_index = _track_samples_by_frame_index(track)
    label_metrics = _label_metrics(
        samples_by_index=samples_by_index,
        review=review,
        hit_radius_px=hit_radius_px,
    )
    jitter_metrics = _jitter_metrics(
        samples_by_index=samples_by_index,
        teleport_px_per_frame=teleport_px_per_frame,
        max_jump_gap_frames=max_jump_gap_frames,
    )
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_ball_tracker_benchmark_candidate",
        "status": "scored_not_gate_verified",
        "clip": review.clip,
        "candidate": candidate_name,
        "category": category,
        "ball_track_path": str(ball_track_path),
        "clicks_path": str(clicks_path),
        "hit_radius_px": float(hit_radius_px),
        "teleport_px_per_frame": float(teleport_px_per_frame),
        "max_jump_gap_frames": int(max_jump_gap_frames),
        "frame_count": len(track.frames),
        "visible_frame_count": sum(1 for frame in track.frames if frame.visible),
        "label_metrics": label_metrics,
        "jitter_metrics": jitter_metrics,
        "quality_score": _quality_score(label_metrics, jitter_metrics),
    }


def benchmark_ball_tracker_candidates(
    *,
    candidates: list[BallCandidate],
    review_root: str | Path,
    hit_radius_px: float = 36.0,
    teleport_px_per_frame: float = 160.0,
    max_jump_gap_frames: int = 3,
) -> dict[str, Any]:
    """Score many candidates and aggregate by candidate name."""

    review_base = Path(review_root)
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        clicks_path = review_base / candidate.clip / "ball_points.json"
        rows.append(
            benchmark_ball_track_candidate(
                ball_track_path=candidate.path,
                clicks_path=clicks_path,
                candidate_name=candidate.name,
                category=candidate.category,
                hit_radius_px=hit_radius_px,
                teleport_px_per_frame=teleport_px_per_frame,
                max_jump_gap_frames=max_jump_gap_frames,
            )
        )

    return {
        "schema_version": 1,
        "artifact_type": "racketsport_ball_tracker_benchmark",
        "status": "scored_not_gate_verified",
        "candidate_count": len(rows),
        "clip_count": len({row["clip"] for row in rows}),
        "hit_radius_px": float(hit_radius_px),
        "teleport_px_per_frame": float(teleport_px_per_frame),
        "max_jump_gap_frames": int(max_jump_gap_frames),
        "results": rows,
        "aggregate": _aggregate(rows),
    }


def write_ball_tracker_benchmark(
    *,
    candidates: list[BallCandidate],
    review_root: str | Path,
    out_json: str | Path,
    out_markdown: str | Path | None = None,
    hit_radius_px: float = 36.0,
    teleport_px_per_frame: float = 160.0,
    max_jump_gap_frames: int = 3,
) -> dict[str, Any]:
    summary = benchmark_ball_tracker_candidates(
        candidates=candidates,
        review_root=review_root,
        hit_radius_px=hit_radius_px,
        teleport_px_per_frame=teleport_px_per_frame,
        max_jump_gap_frames=max_jump_gap_frames,
    )
    json_path = Path(out_json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if out_markdown is not None:
        md_path = Path(out_markdown)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(render_ball_tracker_benchmark_markdown(summary), encoding="utf-8")
    return summary


def render_ball_tracker_benchmark_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Ball Tracker Benchmark",
        "",
        "Sparse human clicks are used only as held-out review labels. They are not consumed by generalizable candidates.",
        "",
        "## Aggregate",
        "",
        "| Candidate | Category | Clips | Hit recall | Median err px | P90 err px | Hidden FP | P95 jump px | Teleports | Score |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, row in sorted(summary["aggregate"].items()):
        lines.append(
            "| {name} | {category} | {clips} | {recall} | {median} | {p90} | {hidden_fp} | {jump} | {teleports} | {score} |".format(
                name=name,
                category=row["category"],
                clips=row["clip_count"],
                recall=_fmt(row.get("mean_visible_hit_recall")),
                median=_fmt(row.get("mean_median_error_px")),
                p90=_fmt(row.get("mean_p90_error_px")),
                hidden_fp=_fmt(row.get("mean_hidden_false_positive_rate")),
                jump=_fmt(row.get("mean_p95_step_px")),
                teleports=_fmt(row.get("total_teleport_count"), digits=0),
                score=_fmt(row.get("mean_quality_score")),
            )
        )

    lines.extend(
        [
            "",
            "## Per Clip",
            "",
            "| Clip | Candidate | Category | Visible labels hit | Median err px | P90 err px | Hidden FP | P95 jump px | Teleports | Score |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in sorted(summary["results"], key=lambda item: (item["clip"], item["candidate"])):
        label = row["label_metrics"]
        jitter = row["jitter_metrics"]
        lines.append(
            "| {clip} | {candidate} | {category} | {hits}/{labels} | {median} | {p90} | {hidden_fp} | {jump} | {teleports} | {score} |".format(
                clip=row["clip"],
                candidate=row["candidate"],
                category=row["category"],
                hits=label["visible_hit_count"],
                labels=label["visible_label_count"],
                median=_fmt(label.get("median_error_px")),
                p90=_fmt(label.get("p90_error_px")),
                hidden_fp=_fmt(label.get("hidden_false_positive_rate")),
                jump=_fmt(jitter.get("p95_step_px")),
                teleports=_fmt(jitter.get("teleport_count"), digits=0),
                score=_fmt(row.get("quality_score")),
            )
        )
    lines.append("")
    return "\n".join(lines)


def _track_samples_by_frame_index(track: BallTrack) -> dict[int, BallFrame]:
    return {int(round(float(frame.t) * float(track.fps))): frame for frame in track.frames}


def _label_metrics(
    *,
    samples_by_index: dict[int, BallFrame],
    review: BallClickReview,
    hit_radius_px: float,
) -> dict[str, Any]:
    radii = (5.0, 10.0, 20.0, 40.0)
    distances: list[float] = []
    visible_prediction_count = 0
    visible_hit_count = 0
    hit_counts_by_radius = {radius: 0 for radius in radii}
    for item in review.visible_items:
        if item.xy is None:
            continue
        frame = samples_by_index.get(item.frame_index)
        if frame is None or not frame.visible:
            continue
        visible_prediction_count += 1
        distance_px = _distance(frame.xy, item.xy)
        distances.append(distance_px)
        if distance_px <= hit_radius_px:
            visible_hit_count += 1
        for radius in radii:
            if distance_px <= radius:
                hit_counts_by_radius[radius] += 1

    hidden_false_positive_count = 0
    for item in review.hidden_items:
        frame = samples_by_index.get(item.frame_index)
        if frame is not None and frame.visible:
            hidden_false_positive_count += 1

    hidden_true_negative_count = len(review.hidden_items) - hidden_false_positive_count
    f1_radius = 20.0
    f1_tp = hit_counts_by_radius[f1_radius]
    f1_fp = hidden_false_positive_count
    f1_fn = len(review.visible_items) - f1_tp
    precision_at_20 = _ratio(f1_tp, f1_tp + f1_fp)
    recall_at_20 = _ratio(f1_tp, f1_tp + f1_fn)
    label_f1_at_20 = _f1(precision_at_20, recall_at_20)
    hidden_true_negative_rate = _ratio(hidden_true_negative_count, len(review.hidden_items))
    balanced_accuracy_at_20 = _mean(
        value
        for value in (
            recall_at_20,
            hidden_true_negative_rate,
        )
    )
    return {
        "visible_label_count": len(review.visible_items),
        "visible_prediction_count": visible_prediction_count,
        "visible_hit_count": visible_hit_count,
        "visible_presence_recall": _ratio(visible_prediction_count, len(review.visible_items)),
        "visible_hit_recall": _ratio(visible_hit_count, len(review.visible_items)),
        "visible_recall_at_5px": _ratio(hit_counts_by_radius[5.0], len(review.visible_items)),
        "visible_recall_at_10px": _ratio(hit_counts_by_radius[10.0], len(review.visible_items)),
        "visible_recall_at_20px": recall_at_20,
        "visible_recall_at_40px": _ratio(hit_counts_by_radius[40.0], len(review.visible_items)),
        "median_error_px": _percentile(distances, 50) if distances else None,
        "p90_error_px": _percentile(distances, 90) if distances else None,
        "p95_error_px": _percentile(distances, 95) if distances else None,
        "hit_radius_px": float(hit_radius_px),
        "hidden_label_count": len(review.hidden_items),
        "hidden_false_positive_count": hidden_false_positive_count,
        "hidden_false_positive_rate": _ratio(hidden_false_positive_count, len(review.hidden_items)),
        "hidden_true_negative_count": hidden_true_negative_count,
        "hidden_true_negative_rate": hidden_true_negative_rate,
        "precision_at_20px": precision_at_20,
        "label_f1_at_20px": label_f1_at_20,
        "balanced_accuracy_at_20px": balanced_accuracy_at_20,
        "pending_label_count": len(review.pending_items),
    }


def _jitter_metrics(
    *,
    samples_by_index: dict[int, BallFrame],
    teleport_px_per_frame: float,
    max_jump_gap_frames: int,
) -> dict[str, Any]:
    visible = sorted((index, frame) for index, frame in samples_by_index.items() if frame.visible)
    steps: list[float] = []
    speeds: list[float] = []
    teleport_count = 0
    max_step_px = 0.0
    for (prev_index, prev_frame), (index, frame) in zip(visible, visible[1:]):
        gap = index - prev_index
        if gap <= 0:
            continue
        distance_px = _distance(prev_frame.xy, frame.xy)
        steps.append(distance_px)
        speed = distance_px / float(gap)
        speeds.append(speed)
        max_step_px = max(max_step_px, distance_px)
        if gap <= max_jump_gap_frames and speed > teleport_px_per_frame:
            teleport_count += 1

    return {
        "visible_segment_count": len(visible),
        "step_count": len(steps),
        "median_step_px": _percentile(steps, 50) if steps else None,
        "p95_step_px": _percentile(steps, 95) if steps else None,
        "max_step_px": max_step_px if steps else None,
        "median_speed_px_per_frame": _percentile(speeds, 50) if speeds else None,
        "p95_speed_px_per_frame": _percentile(speeds, 95) if speeds else None,
        "teleport_count": teleport_count,
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["candidate"]), []).append(row)

    aggregate: dict[str, Any] = {}
    for name, group in grouped.items():
        label_rows = [row["label_metrics"] for row in group]
        jitter_rows = [row["jitter_metrics"] for row in group]
        aggregate[name] = {
            "category": _first_category(group),
            "clip_count": len({row["clip"] for row in group}),
            "mean_visible_hit_recall": _mean(row.get("visible_hit_recall") for row in label_rows),
            "mean_visible_presence_recall": _mean(row.get("visible_presence_recall") for row in label_rows),
            "mean_median_error_px": _mean(row.get("median_error_px") for row in label_rows),
            "mean_p90_error_px": _mean(row.get("p90_error_px") for row in label_rows),
            "mean_hidden_false_positive_rate": _mean(row.get("hidden_false_positive_rate") for row in label_rows),
            "mean_p95_step_px": _mean(row.get("p95_step_px") for row in jitter_rows),
            "total_teleport_count": sum(int(row.get("teleport_count") or 0) for row in jitter_rows),
            "mean_quality_score": _mean(row.get("quality_score") for row in group),
        }
    return aggregate


def _quality_score(label_metrics: dict[str, Any], jitter_metrics: dict[str, Any]) -> float:
    hit_recall = float(label_metrics.get("visible_hit_recall") or 0.0)
    hidden_fp = float(label_metrics.get("hidden_false_positive_rate") or 0.0)
    teleport_penalty = min(0.35, float(jitter_metrics.get("teleport_count") or 0) * 0.01)
    p90_error = label_metrics.get("p90_error_px")
    error_penalty = min(0.35, float(p90_error) / 400.0) if p90_error is not None else 0.35
    return round(hit_recall - (0.45 * hidden_fp) - teleport_penalty - error_penalty, 6)


def _first_category(rows: list[dict[str, Any]]) -> str:
    categories = {str(row["category"]) for row in rows}
    if len(categories) == 1:
        return next(iter(categories))
    return "mixed"


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / float(denominator)


def _f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None:
        return None
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _mean(values: Any) -> float | None:
    numbers = [float(value) for value in values if value is not None]
    if not numbers:
        return None
    return sum(numbers) / len(numbers)


def _distance(a: Any, b: Any) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[int(position)]
    alpha = position - lower
    return ordered[lower] * (1.0 - alpha) + ordered[upper] * alpha


def _fmt(value: Any, *, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{digits}f}"


__all__ = [
    "BallCandidate",
    "benchmark_ball_track_candidate",
    "benchmark_ball_tracker_candidates",
    "render_ball_tracker_benchmark_markdown",
    "write_ball_tracker_benchmark",
]

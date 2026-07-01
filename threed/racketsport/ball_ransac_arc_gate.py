"""Deterministic quadratic-arc RANSAC gate for BALL tracks."""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path
from typing import Any, Sequence

from .ball_overlay import load_ball_track
from .schemas import BallTrack


ARTIFACT_TYPE = "racketsport_ball_ransac_arc_recovery"
STATUS_TESTED = "TESTED-ON-REAL-DATA"


def filter_ball_track_ransac_arcs(
    *,
    ball_track_path: str | Path,
    max_residual_px: float = 5.0,
    min_fit_points: int = 5,
    max_gap_frames: int = 6,
    max_trials: int = 2000,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if max_residual_px <= 0.0 or not math.isfinite(float(max_residual_px)):
        raise ValueError("max_residual_px must be > 0")
    if min_fit_points < 3:
        raise ValueError("min_fit_points must be >= 3")
    if max_gap_frames < 0:
        raise ValueError("max_gap_frames must be >= 0")
    if max_trials <= 0:
        raise ValueError("max_trials must be > 0")

    track = load_ball_track(ball_track_path)
    payload = track.model_dump(mode="json")
    visible_before = sum(1 for frame in payload["frames"] if bool(frame["visible"]))
    segments = _visible_segments(payload["frames"], max_gap_frames=max_gap_frames)
    evaluated_segment_count = 0
    skipped_short_segment_count = 0
    rejected_ransac_outlier_count = 0
    max_surviving_residual_px = 0.0

    for segment in segments:
        if len(segment) < min_fit_points:
            skipped_short_segment_count += 1
            continue
        result = _best_quadratic_arc(
            segment,
            max_residual_px=float(max_residual_px),
            min_fit_points=int(min_fit_points),
            max_trials=int(max_trials),
        )
        if result is None:
            skipped_short_segment_count += 1
            continue
        evaluated_segment_count += 1
        inliers = result["inlier_indices"]
        residuals = result["residuals"]
        for sample in segment:
            residual = residuals[sample["payload_index"]]
            if sample["payload_index"] in inliers:
                max_surviving_residual_px = max(max_surviving_residual_px, residual)
                continue
            _hide_frame(payload["frames"][sample["payload_index"]])
            rejected_ransac_outlier_count += 1

    BallTrack.model_validate(payload)
    visible_after = sum(1 for frame in payload["frames"] if bool(frame["visible"]))
    summary = {
        "schema_version": 1,
        "artifact_type": ARTIFACT_TYPE,
        "status": STATUS_TESTED,
        "source_ball_track": str(ball_track_path),
        "coordinate_model": "image_xy_quadratic_time",
        "max_residual_px": float(max_residual_px),
        "min_fit_points": int(min_fit_points),
        "max_gap_frames": int(max_gap_frames),
        "max_trials": int(max_trials),
        "frame_count": len(payload["frames"]),
        "visible_before": visible_before,
        "visible_after": visible_after,
        "segment_count": len(segments),
        "evaluated_segment_count": evaluated_segment_count,
        "skipped_short_segment_count": skipped_short_segment_count,
        "rejected_ransac_outlier_count": rejected_ransac_outlier_count,
        "recovered_count": 0,
        "max_surviving_residual_px": float(max_surviving_residual_px),
        "uses_human_clicks": False,
        "not_ground_truth": True,
    }
    return payload, summary


def write_ransac_arc_filtered_ball_track(
    *,
    ball_track_path: str | Path,
    out_path: str | Path,
    summary_path: str | Path,
    max_residual_px: float = 5.0,
    min_fit_points: int = 5,
    max_gap_frames: int = 6,
    max_trials: int = 2000,
) -> dict[str, Any]:
    payload, summary = filter_ball_track_ransac_arcs(
        ball_track_path=ball_track_path,
        max_residual_px=max_residual_px,
        min_fit_points=min_fit_points,
        max_gap_frames=max_gap_frames,
        max_trials=max_trials,
    )
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    summary_out = Path(summary_path)
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _visible_segments(frames: Sequence[dict[str, Any]], *, max_gap_frames: int) -> list[list[dict[str, Any]]]:
    segments: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    previous_index: int | None = None
    for index, frame in enumerate(frames):
        if not bool(frame["visible"]):
            continue
        if previous_index is not None and index - previous_index > max_gap_frames + 1:
            if current:
                segments.append(current)
            current = []
        current.append(
            {
                "payload_index": index,
                "time_index": float(index),
                "xy": [float(frame["xy"][0]), float(frame["xy"][1])],
            }
        )
        previous_index = index
    if current:
        segments.append(current)
    return segments


def _best_quadratic_arc(
    samples: Sequence[dict[str, Any]],
    *,
    max_residual_px: float,
    min_fit_points: int,
    max_trials: int,
) -> dict[str, Any] | None:
    best: dict[str, Any] | None = None
    for triplet in _candidate_triplets(samples, max_trials=max_trials):
        model = _fit_quadratic_triplet(triplet)
        if model is None:
            continue
        residuals = {
            sample["payload_index"]: _residual_px(model, sample)
            for sample in samples
        }
        inliers = {
            payload_index
            for payload_index, residual in residuals.items()
            if residual <= max_residual_px
        }
        if len(inliers) < min_fit_points:
            continue
        score = (
            len(inliers),
            -sum(residuals[index] for index in inliers) / float(len(inliers)),
        )
        if best is None or score > best["score"]:
            best = {"model": model, "residuals": residuals, "inlier_indices": inliers, "score": score}
    return best


def _candidate_triplets(samples: Sequence[dict[str, Any]], *, max_trials: int) -> list[tuple[dict[str, Any], ...]]:
    triplets = list(itertools.combinations(samples, 3))
    if len(triplets) <= max_trials:
        return triplets
    step = max(1, len(triplets) // max_trials)
    return triplets[::step][:max_trials]


def _fit_quadratic_triplet(triplet: Sequence[dict[str, Any]]) -> dict[str, tuple[float, float, float]] | None:
    times = [float(sample["time_index"]) for sample in triplet]
    if len(set(times)) < 3:
        return None
    xs = [float(sample["xy"][0]) for sample in triplet]
    ys = [float(sample["xy"][1]) for sample in triplet]
    coeff_x = _quadratic_coefficients(times, xs)
    coeff_y = _quadratic_coefficients(times, ys)
    if coeff_x is None or coeff_y is None:
        return None
    return {"x": coeff_x, "y": coeff_y}


def _quadratic_coefficients(times: Sequence[float], values: Sequence[float]) -> tuple[float, float, float] | None:
    t0, t1, t2 = times
    v0, v1, v2 = values
    denominator = (t0 - t1) * (t0 - t2) * (t1 - t2)
    if denominator == 0.0:
        return None
    a = (t2 * (v1 - v0) + t1 * (v0 - v2) + t0 * (v2 - v1)) / denominator
    b = (t2 * t2 * (v0 - v1) + t1 * t1 * (v2 - v0) + t0 * t0 * (v1 - v2)) / denominator
    c = (
        t1 * t2 * (t1 - t2) * v0
        + t2 * t0 * (t2 - t0) * v1
        + t0 * t1 * (t0 - t1) * v2
    ) / denominator
    return float(a), float(b), float(c)


def _residual_px(model: dict[str, tuple[float, float, float]], sample: dict[str, Any]) -> float:
    t = float(sample["time_index"])
    pred_x = _eval_quadratic(model["x"], t)
    pred_y = _eval_quadratic(model["y"], t)
    return math.hypot(pred_x - float(sample["xy"][0]), pred_y - float(sample["xy"][1]))


def _eval_quadratic(coefficients: tuple[float, float, float], t: float) -> float:
    a, b, c = coefficients
    return a * t * t + b * t + c


def _hide_frame(frame: dict[str, Any]) -> None:
    frame["visible"] = False
    frame["conf"] = 0.0
    frame["approx"] = False
    frame.pop("world_xyz", None)
    frame.pop("speed_mps", None)


__all__ = [
    "ARTIFACT_TYPE",
    "filter_ball_track_ransac_arcs",
    "write_ransac_arc_filtered_ball_track",
]

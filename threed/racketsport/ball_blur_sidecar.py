"""CPU blur sidecar for existing ball tracks.

This module estimates image-plane blur only. It does not infer spin and does
not change detection metrics.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .ball_overlay import load_ball_track
from .schemas import BallFrame


ARTIFACT_TYPE = "racketsport_ball_blur_sidecar"


def estimate_blur_from_points(points_xy: np.ndarray) -> dict[str, Any] | None:
    """Estimate streak midpoint/orientation/extent from foreground points."""

    points = np.asarray(points_xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
        return None
    center = points.mean(axis=0)
    centered = points - center
    covariance = np.cov(centered, rowvar=False, bias=True)
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    major_value = max(float(values[order[0]]), 0.0)
    minor_value = max(float(values[order[1]]), 0.0)
    major_vector = vectors[:, order[0]]
    angle_deg = math.degrees(math.atan2(float(major_vector[1]), float(major_vector[0]))) % 180.0
    projections = centered @ major_vector
    length_px = float(projections.max() - projections.min()) if len(projections) else 0.0
    width_px = 2.0 * math.sqrt(minor_value)
    quality = "clear" if len(points) >= 8 and length_px >= 4.0 else "weak"
    return {
        "center_xy": [round(float(center[0]), 6), round(float(center[1]), 6)],
        "blur_angle_deg": round(angle_deg, 6),
        "blur_length_px": round(length_px, 6),
        "blur_width_px": round(width_px, 6),
        "point_count": int(len(points)),
        "major_variance": round(major_value, 6),
        "minor_variance": round(minor_value, 6),
        "quality": quality,
    }


def build_ball_blur_sidecar(
    *,
    video_path: str | Path,
    ball_track_path: str | Path,
    max_frames: int | None = None,
    crop_radius_px: int = 24,
    min_abs_delta: int = 12,
    threshold_percentile: float = 95.0,
) -> dict[str, Any]:
    """Estimate per-frame blur attributes around current ball candidates."""

    if crop_radius_px < 2:
        raise ValueError("crop_radius_px must be >= 2")
    if max_frames is not None and max_frames < 1:
        raise ValueError("max_frames must be positive")
    if not 0.0 < threshold_percentile < 100.0:
        raise ValueError("threshold_percentile must be in (0, 100)")

    cv2 = _cv2()
    video = Path(video_path)
    if not video.is_file():
        raise ValueError(f"missing video: {video}")
    track = load_ball_track(ball_track_path)
    samples_by_index = _track_samples_by_frame_index(track)
    velocity_angles = _velocity_angles_by_index(samples_by_index)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f"cannot open video: {video}")

    records: list[dict[str, Any]] = []
    previous_gray: np.ndarray | None = None
    frame_index = 0
    try:
        while max_frames is None or frame_index < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            sample = samples_by_index.get(frame_index)
            if previous_gray is not None and sample is not None and sample.visible:
                record = _estimate_frame_blur(
                    gray=gray,
                    previous_gray=previous_gray,
                    frame_index=frame_index,
                    sample=sample,
                    velocity_angle_deg=velocity_angles.get(frame_index),
                    crop_radius_px=crop_radius_px,
                    min_abs_delta=min_abs_delta,
                    threshold_percentile=threshold_percentile,
                )
                if record is not None:
                    records.append(record)
            previous_gray = gray
            frame_index += 1
    finally:
        cap.release()

    clear = [record for record in records if record["quality"] == "clear"]
    aligned = [record["angle_delta_to_track_velocity_deg"] for record in clear if record.get("angle_delta_to_track_velocity_deg") is not None]
    return {
        "schema_version": 1,
        "artifact_type": ARTIFACT_TYPE,
        "status": "TESTED-ON-REAL-DATA",
        "ball_verified": False,
        "video_path": str(video),
        "ball_track_path": str(ball_track_path),
        "crop_radius_px": int(crop_radius_px),
        "min_abs_delta": int(min_abs_delta),
        "threshold_percentile": float(threshold_percentile),
        "decoded_frame_count": int(frame_index),
        "record_count": len(records),
        "summary": {
            "clear_count": len(clear),
            "weak_count": len(records) - len(clear),
            "mean_angle_delta_to_track_velocity_deg": _mean(aligned),
            "median_angle_delta_to_track_velocity_deg": _percentile(aligned, 50.0) if aligned else None,
            "usable_as_velocity_prior": bool(aligned and (_percentile(aligned, 50.0) or 180.0) <= 45.0),
        },
        "frames": records,
        "notes": [
            "Frame-difference blur is a candidate/arc prior only; it is not a detector improvement or spin estimate.",
            "Quality depends on exposure, background motion, and whether the current candidate is the true ball.",
        ],
    }


def write_ball_blur_sidecar(
    *,
    video_path: str | Path,
    ball_track_path: str | Path,
    out_json: str | Path,
    max_frames: int | None = None,
    crop_radius_px: int = 24,
    min_abs_delta: int = 12,
    threshold_percentile: float = 95.0,
) -> dict[str, Any]:
    payload = build_ball_blur_sidecar(
        video_path=video_path,
        ball_track_path=ball_track_path,
        max_frames=max_frames,
        crop_radius_px=crop_radius_px,
        min_abs_delta=min_abs_delta,
        threshold_percentile=threshold_percentile,
    )
    out = Path(out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _estimate_frame_blur(
    *,
    gray: np.ndarray,
    previous_gray: np.ndarray,
    frame_index: int,
    sample: BallFrame,
    velocity_angle_deg: float | None,
    crop_radius_px: int,
    min_abs_delta: int,
    threshold_percentile: float,
) -> dict[str, Any] | None:
    x, y = float(sample.xy[0]), float(sample.xy[1])
    height, width = gray.shape[:2]
    x1 = max(0, int(round(x)) - crop_radius_px)
    y1 = max(0, int(round(y)) - crop_radius_px)
    x2 = min(width, int(round(x)) + crop_radius_px + 1)
    y2 = min(height, int(round(y)) + crop_radius_px + 1)
    if x2 <= x1 or y2 <= y1:
        return None
    diff = np.abs(gray[y1:y2, x1:x2].astype(np.int16) - previous_gray[y1:y2, x1:x2].astype(np.int16))
    threshold = max(float(min_abs_delta), float(np.percentile(diff, threshold_percentile)))
    ys, xs = np.nonzero(diff >= threshold)
    if len(xs) < 3:
        return None
    points = np.column_stack([xs.astype(np.float64) + x1, ys.astype(np.float64) + y1])
    estimate = estimate_blur_from_points(points)
    if estimate is None:
        return None
    angle_delta = _angle_delta_180(estimate["blur_angle_deg"], velocity_angle_deg) if velocity_angle_deg is not None else None
    return {
        "frame_index": int(frame_index),
        "track_xy": [round(x, 6), round(y, 6)],
        "track_conf": float(sample.conf),
        "velocity_angle_deg": velocity_angle_deg,
        "angle_delta_to_track_velocity_deg": angle_delta,
        **estimate,
    }


def _track_samples_by_frame_index(track: Any) -> dict[int, BallFrame]:
    return {int(round(float(frame.t) * float(track.fps))): frame for frame in track.frames}


def _velocity_angles_by_index(samples_by_index: Mapping[int, BallFrame]) -> dict[int, float]:
    visible = sorted((index, frame) for index, frame in samples_by_index.items() if frame.visible)
    angles: dict[int, float] = {}
    for (prev_index, prev), (index, frame) in zip(visible, visible[1:]):
        if index - prev_index != 1:
            continue
        dx = float(frame.xy[0]) - float(prev.xy[0])
        dy = float(frame.xy[1]) - float(prev.xy[1])
        if dx == 0.0 and dy == 0.0:
            continue
        angles[index] = math.degrees(math.atan2(dy, dx)) % 180.0
    return angles


def _angle_delta_180(a: float, b: float | None) -> float | None:
    if b is None:
        return None
    delta = abs((float(a) - float(b)) % 180.0)
    return round(min(delta, 180.0 - delta), 6)


def _mean(values: Sequence[float]) -> float | None:
    return sum(float(value) for value in values) / len(values) if values else None


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[int(position)]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _cv2() -> Any:
    try:
        import cv2  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise RuntimeError("opencv-python is required for the blur sidecar") from exc
    return cv2


__all__ = [
    "build_ball_blur_sidecar",
    "estimate_blur_from_points",
    "write_ball_blur_sidecar",
]

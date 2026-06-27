"""Target-court filtering for ball tracks in multi-court videos."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

from .ball_overlay import load_ball_track
from .court_calibration import project_planar_points
from .court_templates import get_court_template
from .schemas import BallTrack, CourtCalibration


def load_court_calibration(path: str | Path) -> CourtCalibration:
    calibration_path = Path(path)
    if not calibration_path.is_file():
        raise ValueError(f"missing court_calibration file: {calibration_path}")
    try:
        payload = json.loads(calibration_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid court_calibration JSON: {calibration_path}: {exc}") from exc
    try:
        return CourtCalibration.model_validate(payload)
    except Exception as exc:
        raise ValueError(f"invalid court_calibration schema: {calibration_path}: {exc}") from exc


def build_target_court_polygon(
    calibration: CourtCalibration,
    *,
    target_size: tuple[int, int] | None = None,
) -> list[list[float]]:
    """Project the calibrated target court into the requested image space."""

    template = get_court_template(calibration.sport)
    polygon = project_planar_points(calibration.homography, template.corners_m)
    if target_size is None:
        return polygon

    source_width, source_height = _calibration_image_size(calibration)
    target_width, target_height = target_size
    if target_width <= 0 or target_height <= 0:
        raise ValueError("target_size values must be > 0")
    scale_x = float(target_width) / source_width
    scale_y = float(target_height) / source_height
    return [[float(x) * scale_x, float(y) * scale_y] for x, y in polygon]


def point_in_polygon_with_margin(
    point: Iterable[float],
    polygon: Iterable[Iterable[float]],
    *,
    margin_px: float,
) -> bool:
    pt = [float(value) for value in point]
    poly = [[float(value) for value in vertex] for vertex in polygon]
    if len(pt) != 2:
        raise ValueError("point must have two values")
    if len(poly) < 3:
        raise ValueError("polygon must contain at least three vertices")
    if margin_px < 0.0:
        raise ValueError("margin_px must be >= 0")
    if _point_in_polygon(pt, poly):
        return True
    return _distance_to_polygon(pt, poly) <= float(margin_px)


def filter_ball_track_to_target_court(
    *,
    ball_track_path: str | Path,
    calibration: CourtCalibration,
    target_size: tuple[int, int] | None,
    margin_px: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if margin_px < 0.0:
        raise ValueError("margin_px must be >= 0")

    track = load_ball_track(ball_track_path)
    target_court_polygon = build_target_court_polygon(calibration, target_size=target_size)
    payload = track.model_dump(mode="json")
    visible_before = 0
    visible_after = 0
    rejected_outside = 0

    for frame in payload["frames"]:
        if not bool(frame["visible"]):
            continue
        visible_before += 1
        if point_in_polygon_with_margin(frame["xy"], target_court_polygon, margin_px=margin_px):
            visible_after += 1
            continue
        frame["visible"] = False
        frame["conf"] = 0.0
        rejected_outside += 1

    BallTrack.model_validate(payload)
    summary = {
        "schema_version": 1,
        "artifact_type": "racketsport_ball_target_court_filter",
        "status": "filtered_not_gate_verified",
        "source_ball_track": str(ball_track_path),
        "sport": calibration.sport,
        "target_size": list(target_size) if target_size is not None else None,
        "margin_px": float(margin_px),
        "target_court_polygon": target_court_polygon,
        "frame_count": len(payload["frames"]),
        "visible_before": visible_before,
        "visible_after": visible_after,
        "rejected_outside_target_court": rejected_outside,
        "not_ground_truth": True,
    }
    return payload, summary


def write_filtered_ball_track(
    *,
    ball_track_path: str | Path,
    calibration_path: str | Path,
    out_path: str | Path,
    summary_path: str | Path,
    target_size: tuple[int, int] | None,
    margin_px: float,
) -> dict[str, Any]:
    calibration = load_court_calibration(calibration_path)
    payload, summary = filter_ball_track_to_target_court(
        ball_track_path=ball_track_path,
        calibration=calibration,
        target_size=target_size,
        margin_px=margin_px,
    )

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    summary_out = Path(summary_path)
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _calibration_image_size(calibration: CourtCalibration) -> tuple[float, float]:
    width = float(calibration.intrinsics.cx) * 2.0
    height = float(calibration.intrinsics.cy) * 2.0
    if not math.isfinite(width) or not math.isfinite(height) or width <= 0.0 or height <= 0.0:
        raise ValueError("cannot infer calibration image size from intrinsics")
    return width, height


def _point_in_polygon(point: list[float], polygon: list[list[float]]) -> bool:
    x, y = point
    inside = False
    prev_x, prev_y = polygon[-1]
    for curr_x, curr_y in polygon:
        if _point_on_segment(point, [prev_x, prev_y], [curr_x, curr_y]):
            return True
        crosses = (curr_y > y) != (prev_y > y)
        if crosses:
            x_intersection = (prev_x - curr_x) * (y - curr_y) / (prev_y - curr_y) + curr_x
            if x <= x_intersection:
                inside = not inside
        prev_x, prev_y = curr_x, curr_y
    return inside


def _point_on_segment(point: list[float], start: list[float], end: list[float]) -> bool:
    return _distance_to_segment(point, start, end) <= 1e-9


def _distance_to_polygon(point: list[float], polygon: list[list[float]]) -> float:
    return min(
        _distance_to_segment(point, polygon[idx - 1], polygon[idx])
        for idx in range(len(polygon))
    )


def _distance_to_segment(point: list[float], start: list[float], end: list[float]) -> float:
    px, py = point
    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1
    segment_length_sq = dx * dx + dy * dy
    if math.isclose(segment_length_sq, 0.0):
        return math.hypot(px - x1, py - y1)
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / segment_length_sq))
    nearest_x = x1 + t * dx
    nearest_y = y1 + t * dy
    return math.hypot(px - nearest_x, py - nearest_y)

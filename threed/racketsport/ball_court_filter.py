"""Target-court filtering for ball tracks in multi-court videos."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

from .ball_overlay import load_ball_track
from .coordinates import (
    CoordinateSpace,
    homography_pixel_space,
    project_world_xy_points,
    require_same_raster_space,
    resolve_homography_pixel_convention,
    scale_raster_points,
    unproject_image_points_to_world,
)
from .court_calibration import calibration_image_size
from .court_templates import get_court_template
from .schemas import BallTrack, CourtCalibration

STATUS_TESTED = "TESTED-ON-REAL-DATA"


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
    calibration_space = _calibration_homography_space(calibration)
    polygon = project_world_xy_points(
        calibration.homography,
        template.corners_m,
        input_space=CoordinateSpace.WORLD_XY_HOMOGRAPHY_M,
        output_space=calibration_space,
        homography_space=calibration_space,
    )
    if target_size is None:
        return polygon

    source_width, source_height = _calibration_image_size(calibration)
    target_width, target_height = target_size
    if target_width <= 0 or target_height <= 0:
        raise ValueError("target_size values must be > 0")
    return scale_raster_points(
        polygon,
        source_size=(source_width, source_height),
        target_size=(float(target_width), float(target_height)),
        input_space=calibration_space,
        output_space=CoordinateSpace.PIXELS_PREVIEW_SCALED,
    )


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


def point_in_polygon_with_margin_typed(
    point: Iterable[float],
    polygon: Iterable[Iterable[float]],
    *,
    margin_px: float,
    point_space: CoordinateSpace,
    polygon_space: CoordinateSpace,
) -> bool:
    """Declare the raster convention used by both pixel-domain operands."""

    require_same_raster_space(point_space, polygon_space)
    return point_in_polygon_with_margin(point, polygon, margin_px=margin_px)


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
    target_space = (
        CoordinateSpace.PIXELS_PREVIEW_SCALED if target_size is not None else _calibration_homography_space(calibration)
    )
    payload = track.model_dump(mode="json")
    visible_before = 0
    visible_after = 0
    rejected_outside = 0

    for frame in payload["frames"]:
        if not bool(frame["visible"]):
            continue
        visible_before += 1
        if point_in_polygon_with_margin_typed(
            frame["xy"],
            target_court_polygon,
            margin_px=margin_px,
            point_space=target_space,
            polygon_space=target_space,
        ):
            visible_after += 1
            continue
        frame["visible"] = False
        frame["conf"] = 0.0
        rejected_outside += 1

    BallTrack.model_validate(payload)
    summary = {
        "schema_version": 1,
        "artifact_type": "racketsport_ball_target_court_filter",
        "status": STATUS_TESTED,
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


def filter_ball_track_to_target_court_metric_margin(
    *,
    ball_track_path: str | Path,
    calibration: CourtCalibration,
    target_size: tuple[int, int] | None,
    margin_m: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Drop visible detections outside the regulation court plus a metric margin."""

    if margin_m < 0.0 or not math.isfinite(float(margin_m)):
        raise ValueError("margin_m must be >= 0")

    track = load_ball_track(ball_track_path)
    payload = track.model_dump(mode="json")
    visible_before = 0
    visible_after = 0
    rejected_outside = 0

    for frame in payload["frames"]:
        if not bool(frame["visible"]):
            continue
        visible_before += 1
        world_xy = _target_image_xy_to_world_xy(frame["xy"], calibration=calibration, target_size=target_size)
        if _world_xy_in_court_with_margin(world_xy, calibration=calibration, margin_m=float(margin_m)):
            visible_after += 1
            continue
        frame["visible"] = False
        frame["conf"] = 0.0
        rejected_outside += 1

    BallTrack.model_validate(payload)
    summary = {
        "schema_version": 1,
        "artifact_type": "racketsport_ball_target_court_metric_filter",
        "status": STATUS_TESTED,
        "source_ball_track": str(ball_track_path),
        "sport": calibration.sport,
        "target_size": list(target_size) if target_size is not None else None,
        "court_margin_m": float(margin_m),
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
    margin_m: float | None = None,
) -> dict[str, Any]:
    calibration = load_court_calibration(calibration_path)
    if margin_m is None:
        payload, summary = filter_ball_track_to_target_court(
            ball_track_path=ball_track_path,
            calibration=calibration,
            target_size=target_size,
            margin_px=margin_px,
        )
    else:
        payload, summary = filter_ball_track_to_target_court_metric_margin(
            ball_track_path=ball_track_path,
            calibration=calibration,
            target_size=target_size,
            margin_m=margin_m,
        )

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    summary_out = Path(summary_path)
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _calibration_image_size(calibration: CourtCalibration) -> tuple[float, float]:
    width, height = calibration_image_size(calibration)
    if not math.isfinite(width) or not math.isfinite(height) or width <= 0.0 or height <= 0.0:
        raise ValueError("cannot infer calibration image size from intrinsics")
    return width, height


def _target_image_xy_to_world_xy(
    xy: Any,
    *,
    calibration: CourtCalibration,
    target_size: tuple[int, int] | None,
) -> list[float]:
    x, y = float(xy[0]), float(xy[1])
    calibration_space = _calibration_homography_space(calibration)
    if target_size is not None:
        calibration_width, calibration_height = _calibration_image_size(calibration)
        target_width, target_height = target_size
        if target_width <= 0 or target_height <= 0:
            raise ValueError("target_size values must be > 0")
        x, y = scale_raster_points(
            [[x, y]],
            source_size=(float(target_width), float(target_height)),
            target_size=(calibration_width, calibration_height),
            input_space=CoordinateSpace.PIXELS_PREVIEW_SCALED,
            output_space=calibration_space,
        )[0]
    world = unproject_image_points_to_world(
        calibration.homography,
        [[x, y]],
        input_space=calibration_space,
        homography_space=calibration_space,
        output_space=CoordinateSpace.WORLD_XY_HOMOGRAPHY_M,
    )[0]
    return [float(world[0]), float(world[1])]


def _calibration_homography_space(calibration: CourtCalibration) -> CoordinateSpace:
    payload = calibration.model_dump(mode="python", exclude_none=True)
    convention = resolve_homography_pixel_convention(payload, default="raw_pixels")
    return homography_pixel_space(convention)


def _world_xy_in_court_with_margin(
    world_xy: list[float],
    *,
    calibration: CourtCalibration,
    margin_m: float,
) -> bool:
    template = get_court_template(calibration.sport)
    half_width = template.width_m / 2.0 + margin_m
    half_length = template.length_m / 2.0 + margin_m
    x, y = float(world_xy[0]), float(world_xy[1])
    return -half_width <= x <= half_width and -half_length <= y <= half_length


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

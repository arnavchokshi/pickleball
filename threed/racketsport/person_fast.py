"""YOLO plus tracker fast-tier person detection entrypoints."""

from __future__ import annotations

from dataclasses import dataclass

from .court_calibration import project_image_points_to_world
from .court_templates import Sport, get_court_template
from .court_zones import classify_point
from .schemas import CourtCalibration


@dataclass(frozen=True)
class PersonDetection:
    bbox_xyxy: tuple[float, float, float, float]
    confidence: float
    foot_world_xy: list[float]


def person_detection_from_bbox(
    calibration: CourtCalibration,
    *,
    bbox_xyxy: tuple[float, float, float, float],
    confidence: float,
) -> PersonDetection:
    x1, y1, x2, y2 = [float(value) for value in bbox_xyxy]
    if x2 < x1 or y2 < y1:
        raise ValueError("bbox_xyxy must be ordered as x1, y1, x2, y2")

    foot_image = [(x1 + x2) / 2.0, y2]
    foot_world_xy = project_image_points_to_world(calibration.homography, [foot_image])[0]
    return PersonDetection(
        bbox_xyxy=(x1, y1, x2, y2),
        confidence=float(confidence),
        foot_world_xy=foot_world_xy,
    )


def court_polygon_filter(
    detections: list[PersonDetection],
    *,
    sport: Sport,
    margin_m: float = 0.0,
) -> list[PersonDetection]:
    if margin_m < 0.0:
        raise ValueError("margin_m must be non-negative")
    if margin_m == 0.0:
        return [detection for detection in detections if classify_point(sport, detection.foot_world_xy) is not None]
    return [detection for detection in detections if _inside_court_footprint(sport, detection.foot_world_xy, margin_m=margin_m)]


def _inside_court_footprint(sport: Sport, world_xy: list[float], *, margin_m: float) -> bool:
    template = get_court_template(sport)
    half_width_m = template.width_m / 2.0 + margin_m
    half_length_m = template.length_m / 2.0 + margin_m
    x, y = world_xy
    return -half_width_m <= x <= half_width_m and -half_length_m <= y <= half_length_m

"""YOLO plus tracker fast-tier person detection entrypoints."""

from __future__ import annotations

from dataclasses import dataclass

from .court_calibration import project_image_points_to_world
from .court_templates import Sport
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


def court_polygon_filter(detections: list[PersonDetection], *, sport: Sport) -> list[PersonDetection]:
    return [detection for detection in detections if classify_point(sport, detection.foot_world_xy) is not None]

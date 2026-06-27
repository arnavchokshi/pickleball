"""YOLO plus tracker fast-tier person detection entrypoints."""

from __future__ import annotations

from dataclasses import dataclass

from .court_templates import Sport
from .court_zones import classify_point


@dataclass(frozen=True)
class PersonDetection:
    bbox_xyxy: tuple[float, float, float, float]
    confidence: float
    foot_world_xy: list[float]


def court_polygon_filter(detections: list[PersonDetection], *, sport: Sport) -> list[PersonDetection]:
    return [detection for detection in detections if classify_point(sport, detection.foot_world_xy) is not None]

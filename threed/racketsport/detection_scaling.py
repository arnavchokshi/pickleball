"""Utilities for converting detection boxes between video and calibration pixels."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


def scale_detection_payload_bboxes(
    payload: dict[str, Any],
    *,
    scale_x: float,
    scale_y: float,
) -> dict[str, Any]:
    if scale_x <= 0 or scale_y <= 0:
        raise ValueError("scale_x and scale_y must be positive")
    if scale_x == 1.0 and scale_y == 1.0:
        return deepcopy(payload)

    scaled = deepcopy(payload)
    frames = scaled.get("frames", [])
    if not isinstance(frames, list):
        return scaled
    for frame in frames:
        if not isinstance(frame, dict):
            continue
        detections = frame.get("detections", [])
        if not isinstance(detections, list):
            continue
        for detection in detections:
            if not isinstance(detection, dict):
                continue
            for field in ("bbox", "bbox_xyxy"):
                bbox = detection.get(field)
                if isinstance(bbox, list | tuple) and len(bbox) == 4:
                    detection[field] = _scale_bbox_xyxy(bbox, scale_x=scale_x, scale_y=scale_y)
    return scaled


def _scale_bbox_xyxy(bbox: list[Any] | tuple[Any, ...], *, scale_x: float, scale_y: float) -> list[float]:
    x1, y1, x2, y2 = [float(value) for value in bbox]
    return [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y]


__all__ = ["scale_detection_payload_bboxes"]

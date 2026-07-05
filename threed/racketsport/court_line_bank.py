"""Shared court line candidate helpers.

This module starts with the shape-normalization glue needed by OpenCV callers.
Additional detector adapters live here as they graduate from benchmark-local
experiments into proposal artifacts.
"""

from __future__ import annotations

import math
from typing import Any


LineSegment = tuple[float, float, float, float]


def normalize_hough_lines_p(raw_lines: Any) -> list[LineSegment]:
    """Return OpenCV HoughLinesP output as flat x1,y1,x2,y2 tuples."""

    if raw_lines is None:
        return []

    lines: list[LineSegment] = []
    for row in raw_lines:
        values = row[0] if hasattr(row, "__len__") and len(row) == 1 else row
        if len(values) != 4:
            continue
        x1, y1, x2, y2 = values
        lines.append((float(x1), float(y1), float(x2), float(y2)))
    return lines


def build_line_bank_from_image(image: Any, *, max_segments: int = 256) -> dict[str, Any]:
    """Extract normalized line candidates from a BGR or grayscale image."""

    cv2 = __import__("cv2")
    if image is None or not hasattr(image, "shape") or len(image.shape) < 2:
        raise ValueError("image must be a grayscale or BGR image array")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    raw_hough = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=math.pi / 180.0,
        threshold=35,
        minLineLength=max(20, min(int(image.shape[0]), int(image.shape[1])) // 12),
        maxLineGap=12,
    )
    segments = sorted(normalize_hough_lines_p(raw_hough), key=_segment_length, reverse=True)[:max_segments]
    return {
        "segments": [
            {
                "xyxy": [x1, y1, x2, y2],
                "length": _segment_length((x1, y1, x2, y2)),
                "detector": "hough",
            }
            for x1, y1, x2, y2 in segments
        ],
        "metadata": {
            "detectors": {
                "hough": {"available": True, "count": len(segments)},
                "lsd": {"available": False, "reason": "not_implemented"},
                "fast_line": {"available": False, "reason": "not_implemented"},
                "skimage": {"available": False, "reason": "not_implemented"},
                "elsed": {"available": False, "reason": "import_not_attempted"},
                "deeplsd": {"available": False, "reason": "import_not_attempted"},
            }
        },
    }


def _segment_length(segment: LineSegment) -> float:
    x1, y1, x2, y2 = segment
    return float(math.hypot(x2 - x1, y2 - y1))

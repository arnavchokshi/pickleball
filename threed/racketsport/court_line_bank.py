"""Shared court line candidate helpers.

This module starts with the shape-normalization glue needed by OpenCV callers.
Additional detector adapters live here as they graduate from benchmark-local
experiments into proposal artifacts.

CAL-GEO 2026-07-05 extended this module with the real multi-detector line
bank (Hough + LSD + optional Fast Line Detector merge/dedupe), vanishing-point
family clustering, candidate cross/sideline grouping, and the projected-line
pixel/distance/color evidence scorers used by the real detector_v2 solver.
These are adapted from the proven `court_finding_technology_benchmark`
regulation-line machinery (289.5px mean floor median deployable prototype)
so the product-facing detector can reuse evidence that is already known to
work, instead of re-deriving it from scratch.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence


LineSegment = tuple[float, float, float, float]
LineABC = tuple[float, float, float]


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


# ---------------------------------------------------------------------------
# Real merged multi-detector line bank (hough + lsd + fast-line-detector).
# ---------------------------------------------------------------------------


def _gray(image_bgr: Any) -> Any:
    import cv2  # type: ignore[import-not-found]
    import numpy as np

    if image_bgr is None or not hasattr(image_bgr, "shape") or len(image_bgr.shape) < 2:
        raise ValueError("image_bgr must be an image array")
    if len(image_bgr.shape) == 2:
        return image_bgr.astype(np.uint8)
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)


def _segment_item(x1: float, y1: float, x2: float, y2: float, *, source: str) -> dict[str, Any]:
    dx = x2 - x1
    dy = y2 - y1
    return {
        "p1": [round(float(x1), 3), round(float(y1), 3)],
        "p2": [round(float(x2), 3), round(float(y2), 3)],
        "length_px": round(float(math.hypot(dx, dy)), 3),
        "angle_deg": round(float(math.degrees(math.atan2(dy, dx))), 3),
        "source": source,
    }


def extract_hough_line_segments(image_bgr: Any) -> list[dict[str, Any]]:
    import cv2  # type: ignore[import-not-found]

    gray = _gray(image_bgr)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    raw = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=math.pi / 180.0,
        threshold=32,
        minLineLength=max(18, min(gray.shape[0], gray.shape[1]) // 14),
        maxLineGap=10,
    )
    segments: list[dict[str, Any]] = []
    for x1, y1, x2, y2 in normalize_hough_lines_p(raw):
        item = _segment_item(x1, y1, x2, y2, source="opencv_hough")
        if item["length_px"] >= 18.0:
            segments.append(item)
    segments.sort(key=lambda item: float(item["length_px"]), reverse=True)
    return segments[:128]


def extract_lsd_line_segments(image_bgr: Any) -> list[dict[str, Any]]:
    import cv2  # type: ignore[import-not-found]

    gray = _gray(image_bgr)
    detector = cv2.createLineSegmentDetector()
    raw = detector.detect(gray)[0]
    if raw is None:
        return []
    segments: list[dict[str, Any]] = []
    for line in raw.reshape(-1, 4):
        x1, y1, x2, y2 = [float(value) for value in line]
        item = _segment_item(x1, y1, x2, y2, source="opencv_lsd")
        if item["length_px"] >= 18.0:
            segments.append(item)
    segments.sort(key=lambda item: float(item["length_px"]), reverse=True)
    return segments[:128]


def extract_fast_line_segments(image_bgr: Any) -> list[dict[str, Any]]:
    """Fast Line Detector segments. Returns [] fail-closed when unavailable."""

    try:
        import cv2  # type: ignore[import-not-found]

        gray = _gray(image_bgr)
        detector = cv2.ximgproc.createFastLineDetector()
        raw = detector.detect(gray)
    except Exception:  # pragma: no cover - environment dependent
        return []
    if raw is None:
        return []
    segments: list[dict[str, Any]] = []
    for line in raw.reshape(-1, 4):
        x1, y1, x2, y2 = [float(value) for value in line]
        item = _segment_item(x1, y1, x2, y2, source="opencv_fast_line_detector")
        if item["length_px"] >= 18.0:
            segments.append(item)
    segments.sort(key=lambda item: float(item["length_px"]), reverse=True)
    return segments[:128]


def dedupe_line_segments(segments: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for segment in sorted(segments, key=lambda item: float(item["length_px"]), reverse=True):
        mid = segment_midpoint(segment)
        angle = float(segment["angle_deg"])
        duplicate = False
        for existing in selected:
            existing_mid = segment_midpoint(existing)
            if math.dist(mid, existing_mid) <= 8.0 and angle_diff_mod_180(angle, float(existing["angle_deg"])) <= 4.0:
                duplicate = True
                break
        if not duplicate:
            selected.append(dict(segment))
        if len(selected) >= 192:
            break
    return selected


def build_merged_line_bank(image_bgr: Any, *, max_segments: int = 192) -> dict[str, Any]:
    """Merge hough + lsd + fast-line-detector candidates into one deduped bank.

    This is the real Stage-1(c) line bank: multiple classical detectors merged
    and deduped, ready for vanishing-point clustering and cross/sideline
    grouping. Individual detector failures (e.g. fast-line-detector missing
    from the OpenCV build) degrade gracefully to an empty contribution.
    """

    hough = extract_hough_line_segments(image_bgr)
    lsd = extract_lsd_line_segments(image_bgr)
    fast_line = extract_fast_line_segments(image_bgr)
    merged = dedupe_line_segments(hough + lsd + fast_line)[:max_segments]
    return {
        "segments": merged,
        "metadata": {
            "detectors": {
                "hough": {"available": True, "count": len(hough)},
                "lsd": {"available": True, "count": len(lsd)},
                "fast_line": {"available": bool(fast_line), "count": len(fast_line)},
            },
            "merged_count": len(merged),
        },
    }


def cluster_line_family_directions(segments: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return up to two dominant vanishing-point-ish line directions.

    Thin wrapper around `net_anchor_court.cluster_ground_line_directions` so the
    detector can VP-lock cross-court vs sideline family assignment instead of a
    fixed image-axis split, which is what steep/low cameras need.
    """

    from .net_anchor_court import cluster_ground_line_directions

    raw_segments = [
        [[float(segment["p1"][0]), float(segment["p1"][1])], [float(segment["p2"][0]), float(segment["p2"][1])]]
        for segment in segments
        if _is_xy(segment.get("p1")) and _is_xy(segment.get("p2"))
    ]
    return cluster_ground_line_directions(raw_segments)


def _is_xy(value: Any) -> bool:
    return isinstance(value, Sequence) and len(value) == 2


# ---------------------------------------------------------------------------
# Line geometry helpers (ax + by + c = 0 normal form).
# ---------------------------------------------------------------------------


def line_from_segment(segment: Mapping[str, Any]) -> LineABC:
    p1 = segment.get("p1")
    p2 = segment.get("p2")
    if not _is_xy(p1) or not _is_xy(p2):
        raise ValueError("segment must contain p1 and p2")
    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])
    a = y1 - y2
    b = x2 - x1
    c = x1 * y2 - x2 * y1
    norm = math.hypot(a, b)
    if norm <= 1e-6:
        raise ValueError("zero-length segment")
    a, b, c = a / norm, b / norm, c / norm
    if a < 0.0 or (abs(a) <= 1e-9 and b < 0.0):
        a, b, c = -a, -b, -c
    return (float(a), float(b), float(c))


def line_segment_perpendicular_offset(line: LineABC, segment: Mapping[str, Any]) -> float:
    p1 = segment.get("p1")
    p2 = segment.get("p2")
    if not _is_xy(p1) or not _is_xy(p2):
        return float("inf")
    a, b, c = line
    return (abs(a * float(p1[0]) + b * float(p1[1]) + c) + abs(a * float(p2[0]) + b * float(p2[1]) + c)) / 2.0


def line_intersection(first: LineABC, second: LineABC) -> tuple[float, float]:
    a1, b1, c1 = first
    a2, b2, c2 = second
    det = a1 * b2 - a2 * b1
    if abs(det) <= 1e-6:
        raise ValueError("parallel lines do not intersect")
    x = (b1 * c2 - b2 * c1) / det
    y = (c1 * a2 - c2 * a1) / det
    return (float(x), float(y))


def line_y_at_x(line: LineABC, x: float, *, fallback: float) -> float:
    a, b, c = line
    if abs(b) <= 1e-6:
        return float(fallback)
    return float((-a * x - c) / b)


def line_x_at_y(line: LineABC, y: float, *, fallback: float) -> float:
    a, b, c = line
    if abs(a) <= 1e-6:
        return float(fallback)
    return float((-b * y - c) / a)


def point_is_finite(point: Sequence[float]) -> bool:
    return math.isfinite(float(point[0])) and math.isfinite(float(point[1]))


def point_inside_loose_bounds(point: Sequence[float], *, width: int, height: int) -> bool:
    margin = max(width, height) * 0.50
    return -margin <= point[0] <= width + margin and -margin <= point[1] <= height + margin


def segment_midpoint(segment: Mapping[str, Any]) -> tuple[float, float]:
    p1 = segment["p1"]
    p2 = segment["p2"]
    return ((float(p1[0]) + float(p2[0])) / 2.0, (float(p1[1]) + float(p2[1])) / 2.0)


def angle_diff_mod_180(a: float, b: float) -> float:
    diff = abs((a - b) % 180.0)
    return min(diff, 180.0 - diff)


def point_line_distance(point: Sequence[float], line_p1: Sequence[float], line_p2: Sequence[float]) -> float:
    x0, y0 = point
    x1, y1 = line_p1
    x2, y2 = line_p2
    numerator = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
    denominator = math.hypot(y2 - y1, x2 - x1)
    return numerator / denominator if denominator > 1e-6 else float("inf")


def projection_t(point: Sequence[float], line_p1: Sequence[float], line_p2: Sequence[float]) -> float:
    px, py = point
    x1, y1 = line_p1
    x2, y2 = line_p2
    dx = x2 - x1
    dy = y2 - y1
    denom = dx * dx + dy * dy
    if denom <= 1e-6:
        return 0.0
    return ((px - x1) * dx + (py - y1) * dy) / denom


def segment_overlap_fraction_along_line(
    c1: Sequence[float],
    c2: Sequence[float],
    reviewed_p1: Sequence[float],
    reviewed_p2: Sequence[float],
) -> float:
    t1 = projection_t(c1, reviewed_p1, reviewed_p2)
    t2 = projection_t(c2, reviewed_p1, reviewed_p2)
    low = max(0.0, min(t1, t2))
    high = min(1.0, max(t1, t2))
    return max(0.0, high - low)


# ---------------------------------------------------------------------------
# Candidate cross/sideline grouping (dedupe collinear segments into families).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CandidateLine:
    line: LineABC
    segment: dict[str, Any]
    support_length_px: float
    source_segment_count: int
    angle_deg: float


def group_candidate_lines(
    segments: Sequence[Mapping[str, Any]],
    *,
    width: float,
    height: float,
) -> list[CandidateLine]:
    """Merge near-collinear segments into candidate line families.

    Adapted from the proven benchmark `_candidate_line_groups`: segments within
    4 degrees and a small perpendicular offset are merged into one candidate
    line, ranked by total supporting pixel length.
    """

    groups: list[list[Mapping[str, Any]]] = []
    for segment in sorted(segments, key=lambda item: float(item.get("length_px") or 0.0), reverse=True):
        try:
            line = line_from_segment(segment)
        except ValueError:
            continue
        assigned = False
        for group in groups:
            if (
                angle_diff_mod_180(float(segment["angle_deg"]), float(group[0]["angle_deg"])) <= 4.0
                and line_segment_perpendicular_offset(line, group[0]) <= max(10.0, min(width, height) * 0.018)
            ):
                group.append(segment)
                assigned = True
                break
        if not assigned:
            groups.append([segment])
    merged: list[CandidateLine] = []
    for group in groups:
        reference = max(group, key=lambda item: float(item.get("length_px") or 0.0))
        line = line_from_segment(reference)
        support = sum(float(item.get("length_px") or 0.0) for item in group)
        merged.append(
            CandidateLine(
                line=line,
                segment=dict(reference),
                support_length_px=float(support),
                source_segment_count=len(group),
                angle_deg=float(reference["angle_deg"]),
            )
        )
    return sorted(merged, key=lambda item: item.support_length_px, reverse=True)


def best_segment_support_for_line(
    reviewed_p1: Sequence[float],
    reviewed_p2: Sequence[float],
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    rx1, ry1 = reviewed_p1
    rx2, ry2 = reviewed_p2
    r_len = math.hypot(rx2 - rx1, ry2 - ry1)
    if r_len <= 1e-6:
        return None
    r_angle = math.degrees(math.atan2(ry2 - ry1, rx2 - rx1))
    best: dict[str, Any] | None = None
    best_cost = float("inf")
    for candidate in candidates:
        p1 = candidate.get("p1")
        p2 = candidate.get("p2")
        if not _is_xy(p1) or not _is_xy(p2):
            continue
        c1 = (float(p1[0]), float(p1[1]))
        c2 = (float(p2[0]), float(p2[1]))
        c_len = math.dist(c1, c2)
        if c_len <= 1e-6:
            continue
        c_angle = math.degrees(math.atan2(c2[1] - c1[1], c2[0] - c1[0]))
        angle_diff = angle_diff_mod_180(r_angle, c_angle)
        mean_distance = (
            point_line_distance(c1, reviewed_p1, reviewed_p2) + point_line_distance(c2, reviewed_p1, reviewed_p2)
        ) / 2.0
        overlap_fraction = segment_overlap_fraction_along_line(c1, c2, reviewed_p1, reviewed_p2)
        midpoint = ((c1[0] + c2[0]) / 2.0, (c1[1] + c2[1]) / 2.0)
        midpoint_t = projection_t(midpoint, reviewed_p1, reviewed_p2)
        outside_penalty = max(0.0, -midpoint_t, midpoint_t - 1.0) * 100.0
        cost = mean_distance + angle_diff * 2.0 - overlap_fraction * 20.0 + outside_penalty
        if cost < best_cost:
            best_cost = cost
            best = {
                "p1": [round(c1[0], 3), round(c1[1], 3)],
                "p2": [round(c2[0], 3), round(c2[1], 3)],
                "source": candidate.get("source"),
                "length_px": candidate.get("length_px"),
                "angle_diff_deg": round(angle_diff, 3),
                "mean_perpendicular_distance_px": round(mean_distance, 3),
                "overlap_fraction": round(overlap_fraction, 4),
                "midpoint_t": round(midpoint_t, 4),
                "support_cost": round(cost, 3),
            }
    return best


# ---------------------------------------------------------------------------
# Dual-line support test with off-frame visibility states (CAL-GEO round 2).
# ---------------------------------------------------------------------------


def clip_segment_to_image(
    p1: Sequence[float],
    p2: Sequence[float],
    *,
    width: int,
    height: int,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Liang-Barsky clip of segment p1-p2 to the image rectangle.

    Returns the clipped (p1, p2), or None when the segment lies entirely
    outside the frame.
    """

    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])
    dx = x2 - x1
    dy = y2 - y1
    t_min, t_max = 0.0, 1.0
    for p, q in (
        (-dx, x1 - 0.0),
        (dx, float(width - 1) - x1),
        (-dy, y1 - 0.0),
        (dy, float(height - 1) - y1),
    ):
        if abs(p) < 1e-12:
            if q < 0:
                return None
            continue
        t = q / p
        if p < 0:
            t_min = max(t_min, t)
        else:
            t_max = min(t_max, t)
        if t_min > t_max:
            return None
    return ((x1 + t_min * dx, y1 + t_min * dy), (x1 + t_max * dx, y1 + t_max * dy))


def evaluate_projected_template_lines(
    keypoints: Mapping[str, Sequence[float]],
    *,
    endpoint_pairs: Mapping[str, tuple[str, str]],
    segments: Sequence[Mapping[str, Any]],
    pixel_mask: Any,
    image_size: tuple[int, int],
    min_in_frame_fraction: float = 0.22,
) -> dict[str, Any]:
    """Round-2 DUAL-LINE SUPPORT TEST with off-frame visibility states.

    For every projected template line:
    - Clip the projected segment to the frame. If the in-image portion is
      absent or tiny, the line is UNOBSERVABLE: zero penalty, recorded as
      such (a template line outside the camera view cannot be evidence for
      or against a hypothesis).
    - Otherwise the line is OBSERVABLE and MUST have real support: either
      paint-pixel support sampled along the in-image portion, or a matching
      segment in the (persistent) line bank with compatible angle/offset and
      meaningful overlap. An observable projected line with ~zero support is
      exactly the signature of a depth-compressed/exploded mis-assignment
      (e.g. kitchen-as-baseline projects its NVZ into unpainted mid-court),
      and callers apply a heavy penalty per unsupported observable line.

    Returns per-line records (status/in_frame_fraction/support fractions/
    best matching bank segment) plus summary counts, so the same structure
    can be stored in artifacts (verify metrics + review UI).
    """

    width, height = int(image_size[0]), int(image_size[1])
    mask_h = int(pixel_mask.shape[0]) if pixel_mask is not None else 0
    mask_w = int(pixel_mask.shape[1]) if pixel_mask is not None else 0
    per_line: dict[str, dict[str, Any]] = {}
    observable = 0
    supported = 0
    unobservable = 0
    for line_name, (p1_name, p2_name) in endpoint_pairs.items():
        raw_p1 = keypoints.get(p1_name)
        raw_p2 = keypoints.get(p2_name)
        if not _is_xy(raw_p1) or not _is_xy(raw_p2):
            per_line[line_name] = {"status": "missing_projected_endpoint", "endpoint_names": [p1_name, p2_name]}
            continue
        p1 = (float(raw_p1[0]), float(raw_p1[1]))
        p2 = (float(raw_p2[0]), float(raw_p2[1]))
        full_length = math.dist(p1, p2)
        clipped = clip_segment_to_image(p1, p2, width=width, height=height)
        if clipped is None or full_length <= 1e-6:
            unobservable += 1
            per_line[line_name] = {
                "status": "unobservable",
                "endpoint_names": [p1_name, p2_name],
                "in_frame_fraction": 0.0,
                "in_frame_length_px": 0.0,
            }
            continue
        c1, c2 = clipped
        in_frame_length = math.dist(c1, c2)
        in_frame_fraction = in_frame_length / full_length if full_length > 1e-6 else 0.0
        # Unobservable strictly means the line PROJECTS OUT OF FRAME (fix 2).
        # A fully-in-frame line that is merely short (e.g. because the whole
        # hypothesized court is small) is very much observable -- treating it
        # as unobservable was a measured loophole that let depth-compressed
        # fits exempt their own centerlines from the support requirement.
        if in_frame_fraction < min_in_frame_fraction:
            unobservable += 1
            per_line[line_name] = {
                "status": "unobservable",
                "endpoint_names": [p1_name, p2_name],
                "in_frame_fraction": round(in_frame_fraction, 4),
                "in_frame_length_px": round(in_frame_length, 2),
            }
            continue

        observable += 1
        samples = sample_points_on_segment(c1, c2, spacing_px=5.0, min_count=16, max_count=96)
        paint_hits = 0
        for x, y in samples:
            ix, iy = int(round(x)), int(round(y))
            if 0 <= ix < mask_w and 0 <= iy < mask_h and int(pixel_mask[iy, ix]) > 0:
                paint_hits += 1
        paint_ratio = paint_hits / float(len(samples)) if samples else 0.0

        best = best_segment_support_for_line(c1, c2, segments)
        segment_supported = bool(
            best is not None
            and best["angle_diff_deg"] <= 10.0
            and best["mean_perpendicular_distance_px"] <= 14.0
            and best["overlap_fraction"] >= 0.15
        )
        paint_supported = paint_ratio >= 0.32
        is_supported = paint_supported or segment_supported
        if is_supported:
            supported += 1
        per_line[line_name] = {
            "status": "supported" if is_supported else "unsupported",
            "endpoint_names": [p1_name, p2_name],
            "in_frame_fraction": round(in_frame_fraction, 4),
            "in_frame_length_px": round(in_frame_length, 2),
            "paint_support_ratio": round(paint_ratio, 4),
            "paint_supported": paint_supported,
            "segment_supported": segment_supported,
            "best_segment": best,
        }
    return {
        "per_line": per_line,
        "observable_count": int(observable),
        "supported_count": int(supported),
        "unsupported_count": int(observable - supported),
        "unobservable_count": int(unobservable),
        "supported_fraction": round(supported / observable, 4) if observable else None,
    }


# ---------------------------------------------------------------------------
# Projected-line pixel/distance/color evidence scoring against real images.
# ---------------------------------------------------------------------------


def court_line_pixel_mask(image_bgr: Any, *, dilation_px: int) -> Any:
    import cv2  # type: ignore[import-not-found]
    import numpy as np

    if len(image_bgr.shape) == 2:
        bgr = cv2.cvtColor(image_bgr.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    else:
        bgr = image_bgr.astype(np.uint8)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    value = hsv[:, :, 2]
    saturation = hsv[:, :, 1]
    lightness = lab[:, :, 0]
    local_value = cv2.GaussianBlur(value, (0, 0), 7.0)
    local_lightness = cv2.GaussianBlur(lightness, (0, 0), 7.0)
    bright_or_colored = (
        ((value.astype(np.int16) - local_value.astype(np.int16)) >= 24)
        | ((lightness.astype(np.int16) - local_lightness.astype(np.int16)) >= 22)
        | ((value >= 165) & (saturation >= 45))
        | ((value >= 205) & (saturation <= 70))
    )
    mask = bright_or_colored.astype(np.uint8) * 255
    if dilation_px > 1:
        kernel_size = max(3, int(dilation_px) | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


def line_pixel_distance_transform(mask: Any) -> Any:
    import cv2  # type: ignore[import-not-found]
    import numpy as np

    line_pixels = (mask > 0).astype(np.uint8)
    inverse = (1 - line_pixels).astype(np.uint8) * 255
    return cv2.distanceTransform(inverse, cv2.DIST_L2, 3)


def sample_points_on_segment(
    p1: Sequence[float],
    p2: Sequence[float],
    *,
    spacing_px: float,
    min_count: int,
    max_count: int,
) -> list[tuple[float, float]]:
    length = math.dist(p1, p2)
    if length <= 1e-6:
        return [tuple(p1)]
    count = max(min_count, min(max_count, int(math.ceil(length / max(1.0, spacing_px))) + 1))
    return [
        (
            p1[0] + (p2[0] - p1[0]) * index / float(count - 1),
            p1[1] + (p2[1] - p1[1]) * index / float(count - 1),
        )
        for index in range(count)
    ]


def sample_line_pixels(image_bgr: Any, mask: Any, p1: Sequence[float], p2: Sequence[float]) -> list[tuple[float, float, float]]:
    height, width = int(mask.shape[0]), int(mask.shape[1])
    pixels: list[tuple[float, float, float]] = []
    for x, y in sample_points_on_segment(p1, p2, spacing_px=7.0, min_count=10, max_count=72):
        ix = int(round(x))
        iy = int(round(y))
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                px = ix + dx
                py = iy + dy
                if 0 <= px < width and 0 <= py < height and int(mask[py, px]) > 0:
                    value = image_bgr[py, px]
                    if len(image_bgr.shape) == 2:
                        scalar = float(value)
                        pixels.append((scalar, scalar, scalar))
                    else:
                        pixels.append((float(value[0]), float(value[1]), float(value[2])))
    return pixels[:512]


def line_color_cluster(mean_bgr: Sequence[float]) -> str:
    import cv2  # type: ignore[import-not-found]
    import numpy as np

    bgr = np.array([[[int(round(max(0.0, min(255.0, float(value))))) for value in mean_bgr]]], dtype=np.uint8)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[0, 0]
    hue, saturation, value = int(hsv[0]), int(hsv[1]), int(hsv[2])
    if value >= 175 and saturation <= 42:
        return "white"
    if value < 85:
        return "dark"
    hue_bin = int(round(hue / 10.0) * 10) % 180
    sat_bin = "high_sat" if saturation >= 70 else "low_sat"
    return f"hue_{hue_bin:03d}_{sat_bin}"


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    position = (max(0.0, min(100.0, float(percentile))) / 100.0) * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def score_projected_line_pixels_against_image(
    image_bgr: Any,
    keypoints: Mapping[str, Sequence[float]],
    *,
    endpoint_pairs: Mapping[str, tuple[str, str]],
    line_width_px: int = 5,
    line_pixel_mask: Any | None = None,
) -> dict[str, Any]:
    """Score projected regulation floor lines against local high-contrast line pixels."""

    if image_bgr is None or not hasattr(image_bgr, "shape") or len(image_bgr.shape) < 2:
        return {"available": False, "reason": "invalid_image"}
    mask = line_pixel_mask if line_pixel_mask is not None else court_line_pixel_mask(image_bgr, dilation_px=max(1, int(line_width_px)))
    height, width = int(mask.shape[0]), int(mask.shape[1])
    per_line: dict[str, dict[str, Any]] = {}
    ratios: list[float] = []
    supported_count = 0
    evaluated_count = 0
    for line_name, (p1_name, p2_name) in endpoint_pairs.items():
        raw_p1 = keypoints.get(p1_name)
        raw_p2 = keypoints.get(p2_name)
        if not _is_xy(raw_p1) or not _is_xy(raw_p2):
            per_line[line_name] = {"status": "missing_projected_endpoint", "endpoint_names": [p1_name, p2_name]}
            continue
        p1 = (float(raw_p1[0]), float(raw_p1[1]))
        p2 = (float(raw_p2[0]), float(raw_p2[1]))
        samples = sample_points_on_segment(p1, p2, spacing_px=5.0, min_count=16, max_count=96)
        supported_samples = 0
        inside_samples = 0
        for x, y in samples:
            ix = int(round(x))
            iy = int(round(y))
            if 0 <= ix < width and 0 <= iy < height:
                inside_samples += 1
                if int(mask[iy, ix]) > 0:
                    supported_samples += 1
        ratio = supported_samples / float(len(samples)) if samples else 0.0
        inside_ratio = inside_samples / float(len(samples)) if samples else 0.0
        supported = ratio >= 0.34 and inside_ratio >= 0.45
        evaluated_count += 1
        if supported:
            supported_count += 1
        ratios.append(ratio)
        per_line[line_name] = {
            "status": "supported" if supported else "unsupported",
            "endpoint_names": [p1_name, p2_name],
            "sample_count": len(samples),
            "inside_image_sample_count": int(inside_samples),
            "line_pixel_sample_count": int(supported_samples),
            "line_pixel_support_ratio": round(float(ratio), 4),
            "inside_image_ratio": round(float(inside_ratio), 4),
        }
    return {
        "available": evaluated_count > 0,
        "mode": "local_high_contrast_value_mask",
        "evaluated_line_count": int(evaluated_count),
        "supported_line_pixel_count": int(supported_count),
        "mean_line_pixel_support_ratio": round(float(sum(ratios) / len(ratios)), 4) if ratios else 0.0,
        "mask_support_ratio": round(float((mask > 0).sum()) / float(mask.size), 6) if mask.size else 0.0,
        "per_line": per_line,
    }


def score_projected_line_distance_transform_against_image(
    image_bgr: Any,
    keypoints: Mapping[str, Sequence[float]],
    *,
    endpoint_pairs: Mapping[str, tuple[str, str]],
    line_pixel_mask: Any | None = None,
    distance_map: Any | None = None,
) -> dict[str, Any]:
    """Score projected regulation lines by distance to the nearest line-like pixel."""

    if image_bgr is None or not hasattr(image_bgr, "shape") or len(image_bgr.shape) < 2:
        return {"available": False, "reason": "invalid_image"}
    mask = line_pixel_mask if line_pixel_mask is not None else court_line_pixel_mask(image_bgr, dilation_px=3)
    distances = distance_map if distance_map is not None else line_pixel_distance_transform(mask)
    height, width = int(mask.shape[0]), int(mask.shape[1])
    per_line: dict[str, dict[str, Any]] = {}
    line_means: list[float] = []
    line_p95s: list[float] = []
    supported_count = 0
    evaluated_count = 0
    for line_name, (p1_name, p2_name) in endpoint_pairs.items():
        raw_p1 = keypoints.get(p1_name)
        raw_p2 = keypoints.get(p2_name)
        if not _is_xy(raw_p1) or not _is_xy(raw_p2):
            per_line[line_name] = {"status": "missing_projected_endpoint", "endpoint_names": [p1_name, p2_name]}
            continue
        p1 = (float(raw_p1[0]), float(raw_p1[1]))
        p2 = (float(raw_p2[0]), float(raw_p2[1]))
        samples = sample_points_on_segment(p1, p2, spacing_px=5.0, min_count=16, max_count=96)
        sample_distances: list[float] = []
        inside_samples = 0
        for x, y in samples:
            ix = int(round(x))
            iy = int(round(y))
            if 0 <= ix < width and 0 <= iy < height:
                inside_samples += 1
                sample_distances.append(float(distances[iy, ix]))
        if sample_distances:
            mean_distance = sum(sample_distances) / float(len(sample_distances))
            p95_distance = _percentile(sample_distances, 95.0)
        else:
            mean_distance = float(max(width, height))
            p95_distance = float(max(width, height))
        inside_ratio = inside_samples / float(len(samples)) if samples else 0.0
        supported = mean_distance <= 5.0 and p95_distance <= 12.0 and inside_ratio >= 0.45
        evaluated_count += 1
        if supported:
            supported_count += 1
        line_means.append(mean_distance)
        line_p95s.append(p95_distance)
        per_line[line_name] = {
            "status": "supported" if supported else "unsupported",
            "endpoint_names": [p1_name, p2_name],
            "sample_count": len(samples),
            "inside_image_sample_count": int(inside_samples),
            "inside_image_ratio": round(float(inside_ratio), 4),
            "mean_distance_px": round(float(mean_distance), 4),
            "p95_distance_px": round(float(p95_distance), 4),
        }
    return {
        "available": evaluated_count > 0,
        "mode": "distance_transform_local_high_contrast_mask",
        "evaluated_line_count": int(evaluated_count),
        "distance_supported_line_count": int(supported_count),
        "mean_projected_line_distance_px": round(float(sum(line_means) / len(line_means)), 4) if line_means else 0.0,
        "p95_projected_line_distance_px": round(_percentile(line_p95s, 95.0), 4) if line_p95s else 0.0,
        "mask_support_ratio": round(float((mask > 0).sum()) / float(mask.size), 6) if mask.size else 0.0,
        "per_line": per_line,
    }


def score_line_color_consistency_for_assignment(
    image_bgr: Any,
    line_assignment: Mapping[str, Any],
    *,
    line_pixel_mask: Any | None = None,
) -> dict[str, Any]:
    """Estimate whether assigned pickleball lines belong to one local color layer."""

    if image_bgr is None or not hasattr(image_bgr, "shape") or len(image_bgr.shape) < 2:
        return {"available": False, "reason": "invalid_image"}
    mask = line_pixel_mask if line_pixel_mask is not None else court_line_pixel_mask(image_bgr, dilation_px=2)
    per_line: dict[str, dict[str, Any]] = {}
    clusters: dict[str, int] = {}
    sampled_count = 0
    for name, raw in sorted(line_assignment.items(), key=lambda item: str(item[0])):
        segment = raw.segment if isinstance(raw, CandidateLine) else raw
        p1 = segment.get("p1") if isinstance(segment, Mapping) else None
        p2 = segment.get("p2") if isinstance(segment, Mapping) else None
        if not _is_xy(p1) or not _is_xy(p2):
            per_line[str(name)] = {"status": "invalid_segment"}
            continue
        pixels = sample_line_pixels(image_bgr, mask, (float(p1[0]), float(p1[1])), (float(p2[0]), float(p2[1])))
        if not pixels:
            per_line[str(name)] = {"status": "no_line_pixels", "sample_count": 0}
            continue
        mean_bgr = [sum(float(pixel[index]) for pixel in pixels) / float(len(pixels)) for index in range(3)]
        cluster = line_color_cluster(mean_bgr)
        clusters[cluster] = clusters.get(cluster, 0) + 1
        sampled_count += 1
        per_line[str(name)] = {
            "status": "sampled",
            "sample_count": len(pixels),
            "mean_bgr": [round(float(value), 3) for value in mean_bgr],
            "color_cluster": cluster,
        }
    if sampled_count == 0:
        return {"available": False, "reason": "no_line_pixels_sampled", "sampled_line_count": 0, "per_line": per_line}
    dominant_cluster, dominant_count = max(clusters.items(), key=lambda item: item[1])
    dominant_fraction = dominant_count / float(sampled_count)
    distinct_count = len(clusters)
    mixed_penalty = max(0.0, (distinct_count - 1) * 18.0 + (1.0 - dominant_fraction) * 42.0)
    return {
        "available": True,
        "sampled_line_count": int(sampled_count),
        "distinct_color_cluster_count": int(distinct_count),
        "dominant_color_cluster": dominant_cluster,
        "dominant_color_cluster_fraction": round(float(dominant_fraction), 4),
        "cluster_counts": dict(sorted(clusters.items())),
        "mixed_layer_penalty": round(float(mixed_penalty), 4),
        "per_line": per_line,
    }


# ---------------------------------------------------------------------------
# Subpixel finite-width paint-band centerlines (CAL rank 2, additive opt-in).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PaintCenterlineSample:
    """One supported paint-center observation and its normal uncertainty."""

    xy: tuple[float, float]
    left_edge_xy: tuple[float, float]
    right_edge_xy: tuple[float, float]
    normal: tuple[float, float]
    normal_variance_px2: float
    normal_covariance_px2: tuple[tuple[float, float], tuple[float, float]]
    band_width_px: float
    contrast: float
    edge_fit_residual: float


@dataclass(frozen=True)
class PaintCenterlineCandidate:
    """Subpixel centerline evidence for one finite-width bright paint band."""

    endpoints: tuple[tuple[float, float], tuple[float, float]]
    sampled_points: tuple[PaintCenterlineSample, ...]
    support_length_px: float
    coverage_fraction: float
    band_width_px: float
    band_width_p90_px: float
    polarity: str
    polarity_score: float
    family_hint: str
    orientation_hint: str
    angle_deg: float
    provider: str
    preprocessing: str

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible fixed candidate payload."""

        return {
            "endpoints": [[float(value) for value in point] for point in self.endpoints],
            "sampled_points": [
                {
                    "xy": list(sample.xy),
                    "left_edge_xy": list(sample.left_edge_xy),
                    "right_edge_xy": list(sample.right_edge_xy),
                    "normal": list(sample.normal),
                    "normal_variance_px2": sample.normal_variance_px2,
                    "normal_covariance_px2": [list(row) for row in sample.normal_covariance_px2],
                    "band_width_px": sample.band_width_px,
                    "contrast": sample.contrast,
                    "edge_fit_residual": sample.edge_fit_residual,
                }
                for sample in self.sampled_points
            ],
            "support_length_px": self.support_length_px,
            "coverage_fraction": self.coverage_fraction,
            "band_width_px": self.band_width_px,
            "band_width_p90_px": self.band_width_p90_px,
            "polarity": self.polarity,
            "polarity_score": self.polarity_score,
            "family_hint": self.family_hint,
            "orientation_hint": self.orientation_hint,
            "angle_deg": self.angle_deg,
            "provider": self.provider,
            "preprocessing": self.preprocessing,
        }


@dataclass(frozen=True)
class LegacyPaintEvidenceSample:
    """One legacy evidence location to refine without changing its coverage."""

    xy: tuple[float, float]
    normal: tuple[float, float]
    source_id: str | None = None


@dataclass(frozen=True)
class HybridPaintEvidenceSample:
    """No-drop legacy sample with optional paired-edge subpixel refinement."""

    xy: tuple[float, float]
    legacy_xy: tuple[float, float]
    normal: tuple[float, float]
    normal_variance_px2: float
    normal_covariance_px2: tuple[tuple[float, float], tuple[float, float]]
    provenance: str
    band_width_px: float | None
    contrast: float | None
    edge_fit_residual: float | None
    source_id: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "xy": list(self.xy),
            "legacy_xy": list(self.legacy_xy),
            "normal": list(self.normal),
            "normal_variance_px2": self.normal_variance_px2,
            "normal_covariance_px2": [list(row) for row in self.normal_covariance_px2],
            "provenance": self.provenance,
            "band_width_px": self.band_width_px,
            "contrast": self.contrast,
            "edge_fit_residual": self.edge_fit_residual,
            "source_id": self.source_id,
        }


@dataclass(frozen=True)
class HybridPaintEvidenceSegment:
    """Legacy segment whose samples are all preserved and selectively refined."""

    legacy_endpoints: tuple[tuple[float, float], tuple[float, float]]
    endpoints: tuple[tuple[float, float], tuple[float, float]]
    sampled_points: tuple[HybridPaintEvidenceSample, ...]
    support_length_px: float
    band_refined_fraction: float


def refine_legacy_paint_samples(
    image_bgr: Any,
    legacy_samples: Sequence[LegacyPaintEvidenceSample | Mapping[str, Any]],
    *,
    seed_calibration: Any | None = None,
    preprocessing: str = "raw",
    max_refined_band_width_px: float = 16.0,
) -> list[HybridPaintEvidenceSample]:
    """Refine confirmed paint bands and preserve every declined legacy sample.

    Output cardinality and order always equal the input. A local profile moves
    a sample only when both signed-gradient edges, bright-on-dark polarity, and
    the applicable 2-inch width prior confirm. Otherwise the original point is
    emitted exactly with ``provenance='legacy_raw'`` and deliberately higher
    normal covariance. Bands above 10 px additionally require unsaturated
    contrast and very low fit variance; the default 16 px hard cap makes broad
    bands conservative fallbacks. Callers may raise it explicitly only after
    measuring the same no-regression gate.
    """

    import cv2  # type: ignore[import-not-found]
    import numpy as np

    if image_bgr is None or not hasattr(image_bgr, "shape") or len(image_bgr.shape) < 2:
        raise ValueError("image_bgr must be an image array")
    if preprocessing not in {"raw", "shadow_compensated"}:
        raise ValueError(f"unsupported paint centerline preprocessing: {preprocessing}")
    if max_refined_band_width_px <= 0.0:
        raise ValueError("max_refined_band_width_px must be positive")
    normalized = [_coerce_legacy_paint_sample(sample) for sample in legacy_samples]
    if not normalized:
        return []

    working = image_bgr
    if preprocessing == "shadow_compensated":
        from .overlapping_court_calibration import shadow_removal_preprocess

        working, _metadata = shadow_removal_preprocess(image_bgr)
    gray = cv2.GaussianBlur(_gray(working), (0, 0), 0.8).astype(np.float32)

    width_windows: list[tuple[float, float]] = []
    unit_normals: list[Any] = []
    for sample in normalized:
        normal = np.asarray(sample.normal, dtype=np.float64)
        length = float(np.linalg.norm(normal))
        if length <= 1e-9:
            raise ValueError("legacy sample normal must be non-zero")
        normal /= length
        unit_normals.append(normal)
        expected_width = _seed_expected_paint_width_px(seed_calibration, sample.xy, normal)
        width_windows.append(_paint_width_window(gray.shape[:2], expected_width))

    scan_radius = int(math.ceil(max(maximum for _minimum, maximum in width_windows) * 1.25 + 3.0))
    scan_radius = max(7, min(80, scan_radius))
    offsets = np.arange(-scan_radius, scan_radius + 1, dtype=np.float64)
    points = np.asarray([sample.xy for sample in normalized], dtype=np.float64)
    normals = np.asarray(unit_normals, dtype=np.float64)
    map_x = (points[:, 0:1] + normals[:, 0:1] * offsets[None, :]).astype(np.float32)
    map_y = (points[:, 1:2] + normals[:, 1:2] * offsets[None, :]).astype(np.float32)
    profiles = cv2.remap(
        gray,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=float("nan"),
    )

    results: list[HybridPaintEvidenceSample] = []
    for sample, normal, profile, (min_width, max_width) in zip(
        normalized, normals, profiles, width_windows, strict=True
    ):
        fit = _fit_bright_band_profile(profile, offsets, min_width=min_width, max_width=max_width)
        if fit is not None:
            left_offset, right_offset, contrast, edge_residual, normal_variance = fit
            # The legacy observation must lie on or inside the confirmed band;
            # this prevents a nearby parallel stripe from stealing refinement.
            contains_legacy = left_offset <= 1.25 and right_offset >= -1.25
            center_offset = (left_offset + right_offset) * 0.5
            local_limit = max(1.5, (right_offset - left_offset) * 0.65)
            high_confidence_width = (right_offset - left_offset) <= max_refined_band_width_px
            width = right_offset - left_offset
            broad_band_is_precise = width <= 10.0 or (contrast <= 120.0 and normal_variance <= 0.008)
            if contains_legacy and high_confidence_width and broad_band_is_precise and abs(center_offset) <= local_limit:
                refined = np.asarray(sample.xy, dtype=np.float64) + center_offset * normal
                covariance = normal_variance * np.outer(normal, normal)
                results.append(
                    HybridPaintEvidenceSample(
                        xy=(float(refined[0]), float(refined[1])),
                        legacy_xy=sample.xy,
                        normal=(float(normal[0]), float(normal[1])),
                        normal_variance_px2=float(normal_variance),
                        normal_covariance_px2=(
                            (float(covariance[0, 0]), float(covariance[0, 1])),
                            (float(covariance[1, 0]), float(covariance[1, 1])),
                        ),
                        provenance="band_refined",
                        band_width_px=float(right_offset - left_offset),
                        contrast=float(contrast),
                        edge_fit_residual=float(edge_residual),
                        source_id=sample.source_id,
                    )
                )
                continue

        raw_variance = max(4.0, (max_width * 0.5) ** 2 + 1.0)
        covariance = raw_variance * np.outer(normal, normal)
        results.append(
            HybridPaintEvidenceSample(
                xy=sample.xy,
                legacy_xy=sample.xy,
                normal=(float(normal[0]), float(normal[1])),
                normal_variance_px2=float(raw_variance),
                normal_covariance_px2=(
                    (float(covariance[0, 0]), float(covariance[0, 1])),
                    (float(covariance[1, 0]), float(covariance[1, 1])),
                ),
                provenance="legacy_raw",
                band_width_px=None,
                contrast=None,
                edge_fit_residual=None,
                source_id=sample.source_id,
            )
        )
    return results


def refine_legacy_paint_segments(
    image_bgr: Any,
    legacy_segments: Sequence[Any],
    *,
    seed_calibration: Any | None = None,
    preprocessing: str = "raw",
    sample_spacing_px: float = 8.0,
    max_samples_per_segment: int = 8,
    max_refinement_segments: int = 96,
    max_refined_band_width_px: float = 16.0,
) -> list[HybridPaintEvidenceSegment]:
    """Sample a bounded proposal set and preserve every legacy segment.

    The longest ``max_refinement_segments`` proposals receive bounded normal
    scans. Remaining segments are emitted unchanged with high-covariance raw
    endpoints, so runtime is bounded without a coverage drop.
    """

    if sample_spacing_px <= 0.0:
        raise ValueError("sample_spacing_px must be positive")
    if max_samples_per_segment < 2:
        raise ValueError("max_samples_per_segment must be at least two")
    if max_refinement_segments <= 0:
        raise ValueError("max_refinement_segments must be positive")
    prepared: list[
        tuple[int, tuple[tuple[float, float], tuple[float, float]], float, tuple[LegacyPaintEvidenceSample, ...]]
    ] = []
    for segment_index, raw_segment in enumerate(legacy_segments):
        p1, p2 = _coerce_legacy_segment(raw_segment)
        length = math.dist(p1, p2)
        if length <= 1e-6:
            continue
        direction = ((p2[0] - p1[0]) / length, (p2[1] - p1[1]) / length)
        normal = (-direction[1], direction[0])
        count = max(2, min(max_samples_per_segment, int(math.ceil(length / sample_spacing_px)) + 1))
        samples: list[LegacyPaintEvidenceSample] = []
        for sample_index in range(count):
            t = sample_index / float(count - 1)
            samples.append(
                LegacyPaintEvidenceSample(
                    xy=(p1[0] + (p2[0] - p1[0]) * t, p1[1] + (p2[1] - p1[1]) * t),
                    normal=normal,
                    source_id=f"segment_{segment_index}:sample_{sample_index}",
                )
            )
        prepared.append((segment_index, (p1, p2), length, tuple(samples)))

    selected_indexes = {
        segment_index
        for segment_index, _endpoints, _length, _samples in sorted(
            prepared, key=lambda row: row[2], reverse=True
        )[:max_refinement_segments]
    }
    flattened: list[LegacyPaintEvidenceSample] = []
    selected_slices: dict[int, tuple[int, int]] = {}
    for segment_index, _endpoints, _length, samples in prepared:
        if segment_index not in selected_indexes:
            continue
        start = len(flattened)
        flattened.extend(samples)
        selected_slices[segment_index] = (start, len(flattened))
    refined = refine_legacy_paint_samples(
        image_bgr,
        flattened,
        seed_calibration=seed_calibration,
        preprocessing=preprocessing,
        max_refined_band_width_px=max_refined_band_width_px,
    )
    results: list[HybridPaintEvidenceSegment] = []
    for segment_index, legacy_endpoints, length, legacy_samples in prepared:
        selected_slice = selected_slices.get(segment_index)
        if selected_slice is None:
            samples = tuple(
                _legacy_raw_hybrid_sample(
                    sample,
                    image_shape=image_bgr.shape[:2],
                    seed_calibration=seed_calibration,
                )
                for sample in (legacy_samples[0], legacy_samples[-1])
            )
        else:
            start, stop = selected_slice
            samples = tuple(refined[start:stop])
        refined_count = sum(sample.provenance == "band_refined" for sample in samples)
        results.append(
            HybridPaintEvidenceSegment(
                legacy_endpoints=legacy_endpoints,
                endpoints=(samples[0].xy, samples[-1].xy),
                sampled_points=samples,
                support_length_px=length,
                band_refined_fraction=refined_count / float(len(samples)),
            )
        )
    return results


def _legacy_raw_hybrid_sample(
    sample: LegacyPaintEvidenceSample,
    *,
    image_shape: Sequence[int],
    seed_calibration: Any | None,
) -> HybridPaintEvidenceSample:
    import numpy as np

    normal = np.asarray(sample.normal, dtype=np.float64)
    normal /= max(float(np.linalg.norm(normal)), 1e-12)
    expected_width = _seed_expected_paint_width_px(seed_calibration, sample.xy, normal)
    _minimum, maximum = _paint_width_window(image_shape, expected_width)
    variance = max(4.0, (maximum * 0.5) ** 2 + 1.0)
    covariance = variance * np.outer(normal, normal)
    return HybridPaintEvidenceSample(
        xy=sample.xy,
        legacy_xy=sample.xy,
        normal=(float(normal[0]), float(normal[1])),
        normal_variance_px2=float(variance),
        normal_covariance_px2=(
            (float(covariance[0, 0]), float(covariance[0, 1])),
            (float(covariance[1, 0]), float(covariance[1, 1])),
        ),
        provenance="legacy_raw",
        band_width_px=None,
        contrast=None,
        edge_fit_residual=None,
        source_id=sample.source_id,
    )


def _coerce_legacy_paint_sample(
    sample: LegacyPaintEvidenceSample | Mapping[str, Any],
) -> LegacyPaintEvidenceSample:
    if isinstance(sample, LegacyPaintEvidenceSample):
        return sample
    if not isinstance(sample, Mapping) or not _is_xy(sample.get("xy")) or not _is_xy(sample.get("normal")):
        raise ValueError("legacy sample must contain xy and normal pairs")
    return LegacyPaintEvidenceSample(
        xy=(float(sample["xy"][0]), float(sample["xy"][1])),
        normal=(float(sample["normal"][0]), float(sample["normal"][1])),
        source_id=str(sample["source_id"]) if sample.get("source_id") is not None else None,
    )


def _coerce_legacy_segment(raw_segment: Any) -> tuple[tuple[float, float], tuple[float, float]]:
    if isinstance(raw_segment, Mapping):
        p1 = raw_segment.get("p1")
        p2 = raw_segment.get("p2")
    elif isinstance(raw_segment, Sequence) and len(raw_segment) == 2:
        p1, p2 = raw_segment
    else:
        raise ValueError("legacy segment must contain two endpoints")
    if not _is_xy(p1) or not _is_xy(p2):
        raise ValueError("legacy segment endpoints must be xy pairs")
    return ((float(p1[0]), float(p1[1])), (float(p2[0]), float(p2[1])))


def detect_paint_centerline_candidates(
    image_bgr: Any,
    *,
    seed_calibration: Any | None = None,
    provider: str = "classical_paired_edges",
    preprocessing: str = "raw",
    min_support_length_px: float | None = None,
    max_candidates: int = 96,
) -> list[PaintCenterlineCandidate]:
    """Detect finite-width bright court paint as paired signed-gradient edges.

    This API is intentionally separate from the legacy Hough/LSD APIs. It does
    not change their defaults. ``seed_calibration`` is optional; when supplied,
    its world-to-image homography converts the regulation 2-inch paint width to
    a local pixel-width prior. Without a seed, a deliberately loose image-scale
    prior is used. ``preprocessing='shadow_compensated'`` is an explicit A/B
    arm and imports the existing deterministic shadow normalizer lazily.

    The classical provider merges Hough and locally available OpenCV LSD only
    to propose orientations/extents; every returned line is located from the
    two opposite signed-gradient edges. The LSD-only proposal arm is explicit via
    ``provider='opencv_lsd_paired_edges'`` and uses the same paired-edge fit.
    """

    import cv2  # type: ignore[import-not-found]
    import numpy as np

    if image_bgr is None or not hasattr(image_bgr, "shape") or len(image_bgr.shape) < 2:
        raise ValueError("image_bgr must be an image array")
    if provider not in {"classical_paired_edges", "opencv_lsd_paired_edges"}:
        raise ValueError(f"unsupported paint centerline provider: {provider}")
    if preprocessing not in {"raw", "shadow_compensated"}:
        raise ValueError(f"unsupported paint centerline preprocessing: {preprocessing}")
    if max_candidates <= 0:
        raise ValueError("max_candidates must be positive")

    working = image_bgr
    if preprocessing == "shadow_compensated":
        from .overlapping_court_calibration import shadow_removal_preprocess

        working, _metadata = shadow_removal_preprocess(image_bgr)
    gray_u8 = _gray(working)
    height, width = gray_u8.shape[:2]
    if height <= 0 or width <= 0:
        return []
    gray = cv2.GaussianBlur(gray_u8, (0, 0), 0.8).astype(np.float32)
    minimum_support = (
        float(min_support_length_px)
        if min_support_length_px is not None
        else max(18.0, min(width, height) * 0.045)
    )
    if provider == "opencv_lsd_paired_edges":
        proposal_rows = extract_lsd_line_segments(working)
    else:
        # The primary classical arm unions two locally available proposal
        # mechanisms before the shared paired-edge fit. Hough contributes
        # long stable extents; LSD recovers shorter/blurred paint fragments.
        proposal_rows = dedupe_line_segments(
            extract_hough_line_segments(working) + extract_lsd_line_segments(working)
        )
    proposal_rows = [row for row in proposal_rows if float(row.get("length_px", 0.0)) >= minimum_support]

    candidates: list[PaintCenterlineCandidate] = []
    for proposal in proposal_rows[:96]:
        candidate = _paint_candidate_from_proposal(
            gray,
            proposal,
            seed_calibration=seed_calibration,
            provider=provider,
            preprocessing=preprocessing,
            minimum_support=minimum_support,
        )
        if candidate is not None:
            candidates.append(candidate)
    return _dedupe_paint_centerlines(candidates, max_candidates=max_candidates)


def _paint_candidate_from_proposal(
    gray: Any,
    proposal: Mapping[str, Any],
    *,
    seed_calibration: Any | None,
    provider: str,
    preprocessing: str,
    minimum_support: float,
) -> PaintCenterlineCandidate | None:
    import cv2  # type: ignore[import-not-found]
    import numpy as np

    p1 = np.asarray(proposal["p1"], dtype=np.float64)
    p2 = np.asarray(proposal["p2"], dtype=np.float64)
    delta = p2 - p1
    proposal_length = float(np.linalg.norm(delta))
    if proposal_length < minimum_support:
        return None
    direction = delta / proposal_length
    normal = np.asarray([-direction[1], direction[0]], dtype=np.float64)
    midpoint = (p1 + p2) * 0.5
    expected_width = _seed_expected_paint_width_px(seed_calibration, midpoint, normal)
    min_width, max_width = _paint_width_window(gray.shape[:2], expected_width)
    scan_radius = int(math.ceil(max(7.0, max_width * 1.25 + 3.0)))

    sample_count = max(8, min(56, int(math.ceil(proposal_length / 8.0)) + 1))
    tangential = np.linspace(0.0, proposal_length, sample_count, dtype=np.float64)
    offsets = np.arange(-scan_radius, scan_radius + 1, dtype=np.float64)
    base = p1[None, :] + tangential[:, None] * direction[None, :]
    map_x = (base[:, 0:1] + offsets[None, :] * normal[0]).astype(np.float32)
    map_y = (base[:, 1:2] + offsets[None, :] * normal[1]).astype(np.float32)
    profiles = cv2.remap(
        gray,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=float("nan"),
    )

    raw_samples: list[tuple[float, PaintCenterlineSample]] = []
    for sample_index, profile in enumerate(profiles):
        fit = _fit_bright_band_profile(profile, offsets, min_width=min_width, max_width=max_width)
        if fit is None:
            continue
        left_offset, right_offset, contrast, edge_residual, normal_variance = fit
        center_offset = (left_offset + right_offset) * 0.5
        center = base[sample_index] + center_offset * normal
        left = base[sample_index] + left_offset * normal
        right = base[sample_index] + right_offset * normal
        covariance = normal_variance * np.outer(normal, normal)
        raw_samples.append(
            (
                float(tangential[sample_index]),
                PaintCenterlineSample(
                    xy=(float(center[0]), float(center[1])),
                    left_edge_xy=(float(left[0]), float(left[1])),
                    right_edge_xy=(float(right[0]), float(right[1])),
                    normal=(float(normal[0]), float(normal[1])),
                    normal_variance_px2=float(normal_variance),
                    normal_covariance_px2=(
                        (float(covariance[0, 0]), float(covariance[0, 1])),
                        (float(covariance[1, 0]), float(covariance[1, 1])),
                    ),
                    band_width_px=float(right_offset - left_offset),
                    contrast=float(contrast),
                    edge_fit_residual=float(edge_residual),
                ),
            )
        )
    if len(raw_samples) < max(5, int(math.ceil(sample_count * 0.22))):
        return None

    points = np.asarray([sample.xy for _, sample in raw_samples], dtype=np.float64)
    center = np.median(points, axis=0)
    centered = points - center
    _values, vectors = np.linalg.eigh(centered.T @ centered)
    fitted_direction = vectors[:, -1]
    if float(np.dot(fitted_direction, direction)) < 0.0:
        fitted_direction = -fitted_direction
    fitted_normal = np.asarray([-fitted_direction[1], fitted_direction[0]], dtype=np.float64)
    residuals = np.abs(centered @ fitted_normal)
    median_residual = float(np.median(residuals))
    mad = float(np.median(np.abs(residuals - median_residual)))
    residual_limit = max(0.65, median_residual + 3.5 * max(0.08, 1.4826 * mad))
    keep = residuals <= residual_limit
    supported = [row for row, accepted in zip(raw_samples, keep, strict=True) if bool(accepted)]
    if len(supported) < max(5, int(math.ceil(sample_count * 0.20))):
        return None

    points = np.asarray([sample.xy for _, sample in supported], dtype=np.float64)
    center = np.mean(points, axis=0)
    centered = points - center
    _values, vectors = np.linalg.eigh(centered.T @ centered)
    fitted_direction = vectors[:, -1]
    if float(np.dot(fitted_direction, direction)) < 0.0:
        fitted_direction = -fitted_direction
    fitted_normal = np.asarray([-fitted_direction[1], fitted_direction[0]], dtype=np.float64)
    projections = centered @ fitted_direction
    endpoint_a = center + float(np.min(projections)) * fitted_direction
    endpoint_b = center + float(np.max(projections)) * fitted_direction
    support_length = _gap_aware_support_length([position for position, _sample in supported], proposal_length, sample_count)
    coverage = min(1.0, support_length / max(proposal_length, 1e-6))
    if support_length < minimum_support:
        return None

    adjusted_samples: list[PaintCenterlineSample] = []
    covariance_axis = np.outer(fitted_normal, fitted_normal)
    for _position, sample in supported:
        covariance = sample.normal_variance_px2 * covariance_axis
        adjusted_samples.append(
            PaintCenterlineSample(
                xy=sample.xy,
                left_edge_xy=sample.left_edge_xy,
                right_edge_xy=sample.right_edge_xy,
                normal=(float(fitted_normal[0]), float(fitted_normal[1])),
                normal_variance_px2=sample.normal_variance_px2,
                normal_covariance_px2=(
                    (float(covariance[0, 0]), float(covariance[0, 1])),
                    (float(covariance[1, 0]), float(covariance[1, 1])),
                ),
                band_width_px=sample.band_width_px,
                contrast=sample.contrast,
                edge_fit_residual=sample.edge_fit_residual,
            )
        )
    widths = [sample.band_width_px for sample in adjusted_samples]
    contrasts = [sample.contrast for sample in adjusted_samples]
    angle = math.degrees(math.atan2(float(fitted_direction[1]), float(fitted_direction[0])))
    if angle < -90.0:
        angle += 180.0
    elif angle > 90.0:
        angle -= 180.0
    orientation, family = _paint_orientation_hints(angle)
    return PaintCenterlineCandidate(
        endpoints=(
            (float(endpoint_a[0]), float(endpoint_a[1])),
            (float(endpoint_b[0]), float(endpoint_b[1])),
        ),
        sampled_points=tuple(adjusted_samples),
        support_length_px=float(support_length),
        coverage_fraction=float(coverage),
        band_width_px=float(_percentile(widths, 50.0)),
        band_width_p90_px=float(_percentile(widths, 90.0)),
        polarity="bright_band_on_darker_sides",
        polarity_score=float(_percentile(contrasts, 25.0)),
        family_hint=family,
        orientation_hint=orientation,
        angle_deg=float(angle),
        provider=provider,
        preprocessing=preprocessing,
    )


def _fit_bright_band_profile(
    raw_profile: Any,
    offsets: Any,
    *,
    min_width: float,
    max_width: float,
) -> tuple[float, float, float, float, float] | None:
    import numpy as np

    profile = np.asarray(raw_profile, dtype=np.float64)
    finite = np.isfinite(profile)
    if int(finite.sum()) < 9:
        return None
    if not bool(finite.all()):
        valid_indexes = np.flatnonzero(finite)
        profile = np.interp(np.arange(len(profile)), valid_indexes, profile[valid_indexes])
    gradient = np.gradient(profile)
    min_pixels = max(1, int(math.floor(min_width)))
    max_pixels = min(len(profile) - 4, int(math.ceil(max_width)))
    rising_candidates = _strong_local_peak_indexes(gradient, minimum=2.0, limit=7)
    falling_candidates = _strong_local_peak_indexes(-gradient, minimum=2.0, limit=7)
    best: tuple[float, int, int, float, float] | None = None
    for left_index in rising_candidates:
        if left_index < 2 or left_index >= len(profile) - min_pixels - 2:
            continue
        rising = float(gradient[left_index])
        right_start = left_index + min_pixels
        right_stop = min(len(profile) - 2, left_index + max_pixels + 1)
        for right_index in falling_candidates:
            if not (right_start <= right_index < right_stop):
                continue
            falling = float(-gradient[right_index])
            inside = profile[left_index : right_index + 1]
            left_outside = profile[max(0, left_index - 3) : left_index]
            right_outside = profile[right_index + 1 : min(len(profile), right_index + 4)]
            if left_outside.size == 0 or right_outside.size == 0:
                continue
            inside_level = float(np.median(inside))
            left_level = float(np.median(left_outside))
            right_level = float(np.median(right_outside))
            contrast_left = inside_level - left_level
            contrast_right = inside_level - right_level
            polarity_contrast = min(contrast_left, contrast_right)
            if polarity_contrast < 7.0:
                continue
            filled_threshold = max(left_level, right_level) + 0.45 * polarity_contrast
            if float(np.mean(inside >= filled_threshold)) < 0.55:
                continue
            edge_balance = min(rising, falling) / max(rising, falling)
            score = polarity_contrast * math.sqrt(max(0.0, rising * falling)) * edge_balance
            if best is None or score > best[0]:
                best = (score, left_index, right_index, polarity_contrast, inside_level)
    if best is None:
        return None
    _score, left_index, right_index, contrast, inside_level = best
    rising_response = gradient
    falling_response = -gradient
    left_position, left_residual = _subpixel_parabolic_peak(rising_response, left_index)
    right_position, right_residual = _subpixel_parabolic_peak(falling_response, right_index)
    left_offset = float(np.interp(left_position, np.arange(len(offsets)), offsets))
    right_offset = float(np.interp(right_position, np.arange(len(offsets)), offsets))
    width = right_offset - left_offset
    if not (min_width <= width <= max_width):
        return None
    outside = np.concatenate(
        [
            profile[max(0, left_index - 4) : left_index],
            profile[right_index + 1 : min(len(profile), right_index + 5)],
        ]
    )
    noise = float(np.median(np.abs(outside - np.median(outside)))) * 1.4826 if outside.size else 0.0
    left_strength = max(1.0, float(rising_response[left_index]))
    right_strength = max(1.0, float(falling_response[right_index]))
    left_sigma = 0.04 + 0.45 * (noise + left_residual + 0.5) / (left_strength + contrast + 1.0)
    right_sigma = 0.04 + 0.45 * (noise + right_residual + 0.5) / (right_strength + contrast + 1.0)
    normal_variance = max(0.0016, (left_sigma * left_sigma + right_sigma * right_sigma) / 4.0)
    edge_residual = (left_residual + right_residual) * 0.5
    # ``inside_level`` is kept in the computation above to ensure both edge
    # contrasts are measured against one common band interior.
    _ = inside_level
    return (left_offset, right_offset, float(contrast), float(edge_residual), float(normal_variance))


def _strong_local_peak_indexes(response: Any, *, minimum: float, limit: int) -> list[int]:
    import numpy as np

    values = np.asarray(response, dtype=np.float64)
    if len(values) < 3:
        return []
    indexes = np.flatnonzero(
        (values[1:-1] >= values[:-2])
        & (values[1:-1] >= values[2:])
        & (values[1:-1] > float(minimum))
    ) + 1
    if len(indexes) > limit:
        strongest = np.argpartition(values[indexes], -limit)[-limit:]
        indexes = indexes[strongest]
    return sorted((int(index) for index in indexes), key=lambda index: float(values[index]), reverse=True)


def _subpixel_parabolic_peak(response: Any, index: int) -> tuple[float, float]:
    import numpy as np

    values = np.asarray(response, dtype=np.float64)
    if index <= 0 or index >= len(values) - 1:
        return (float(index), 1.0)
    left, center, right = float(values[index - 1]), float(values[index]), float(values[index + 1])
    denominator = left - 2.0 * center + right
    delta = 0.0 if abs(denominator) <= 1e-9 else 0.5 * (left - right) / denominator
    delta = max(-0.75, min(0.75, delta))
    peak = center - 0.25 * (left - right) * delta
    fitted = np.asarray(
        [
            peak + 0.5 * denominator * (-1.0 - delta) ** 2,
            peak + 0.5 * denominator * (0.0 - delta) ** 2,
            peak + 0.5 * denominator * (1.0 - delta) ** 2,
        ],
        dtype=np.float64,
    )
    residual = float(np.sqrt(np.mean((np.asarray([left, center, right]) - fitted) ** 2)))
    return (float(index) + float(delta), residual)


def _paint_width_window(image_shape: Sequence[int], expected_width: float | None) -> tuple[float, float]:
    height, width = int(image_shape[0]), int(image_shape[1])
    scale = max(0.35, min(height, width) / 1080.0)
    if expected_width is None or not math.isfinite(expected_width):
        return (max(1.2, 1.5 * scale), max(12.0, 30.0 * scale))
    return (
        max(0.35, expected_width * 0.40),
        max(3.2, min(64.0, expected_width * 2.1)),
    )


def _seed_expected_paint_width_px(seed_calibration: Any | None, point: Any, image_normal: Any) -> float | None:
    if seed_calibration is None:
        return None
    import numpy as np

    raw_homography = (
        seed_calibration.get("homography")
        if isinstance(seed_calibration, Mapping)
        else getattr(seed_calibration, "homography", None)
    )
    if raw_homography is None:
        return None
    try:
        homography = np.asarray(raw_homography, dtype=np.float64).reshape(3, 3)
        inverse = np.linalg.inv(homography)
        image_h = np.asarray([float(point[0]), float(point[1]), 1.0], dtype=np.float64)
        world_h = inverse @ image_h
        world = world_h[:2] / world_h[2]
        epsilon_m = 0.01

        def project(world_xy: Any) -> Any:
            projected = homography @ np.asarray([float(world_xy[0]), float(world_xy[1]), 1.0])
            return projected[:2] / projected[2]

        origin = project(world)
        jacobian = np.column_stack(
            [
                (project(world + np.asarray([epsilon_m, 0.0])) - origin) / epsilon_m,
                (project(world + np.asarray([0.0, epsilon_m])) - origin) / epsilon_m,
            ]
        )
        meters_per_normal_pixel = float(np.linalg.norm(np.linalg.solve(jacobian, image_normal)))
        if meters_per_normal_pixel <= 1e-9:
            return None
        return float(0.0508 / meters_per_normal_pixel)
    except (ValueError, TypeError, np.linalg.LinAlgError, ZeroDivisionError):
        return None


def _gap_aware_support_length(positions: Sequence[float], proposal_length: float, sample_count: int) -> float:
    ordered = sorted(float(position) for position in positions)
    if not ordered:
        return 0.0
    nominal_step = proposal_length / max(1, sample_count - 1)
    support = nominal_step
    for first, second in zip(ordered, ordered[1:]):
        gap = second - first
        if gap <= nominal_step * 1.65:
            support += gap
    return min(proposal_length, support)


def _paint_orientation_hints(angle_deg: float) -> tuple[str, str]:
    absolute = abs(angle_deg)
    if absolute <= 15.0:
        return ("near_horizontal", "cross_court_candidate")
    if absolute >= 75.0:
        return ("near_vertical", "longitudinal_candidate")
    slope = "positive" if angle_deg > 0.0 else "negative"
    dominant = "longitudinal_candidate" if absolute > 45.0 else "cross_court_candidate"
    return (f"diagonal_{slope}", dominant)


def _dedupe_paint_centerlines(
    candidates: Sequence[PaintCenterlineCandidate],
    *,
    max_candidates: int,
) -> list[PaintCenterlineCandidate]:
    def quality(candidate: PaintCenterlineCandidate) -> float:
        variance = _percentile(
            [sample.normal_variance_px2 for sample in candidate.sampled_points],
            50.0,
        )
        return candidate.support_length_px * candidate.coverage_fraction * candidate.polarity_score / max(0.01, variance)

    selected: list[PaintCenterlineCandidate] = []
    for candidate in sorted(candidates, key=quality, reverse=True):
        candidate_mid = (
            (candidate.endpoints[0][0] + candidate.endpoints[1][0]) * 0.5,
            (candidate.endpoints[0][1] + candidate.endpoints[1][1]) * 0.5,
        )
        duplicate = False
        for existing in selected:
            if angle_diff_mod_180(candidate.angle_deg, existing.angle_deg) > 3.0:
                continue
            existing_mid = (
                (existing.endpoints[0][0] + existing.endpoints[1][0]) * 0.5,
                (existing.endpoints[0][1] + existing.endpoints[1][1]) * 0.5,
            )
            existing_line = (
                existing.sampled_points[0].normal[0],
                existing.sampled_points[0].normal[1],
                -existing.sampled_points[0].normal[0] * existing_mid[0]
                - existing.sampled_points[0].normal[1] * existing_mid[1],
            )
            offset = abs(
                existing_line[0] * candidate_mid[0]
                + existing_line[1] * candidate_mid[1]
                + existing_line[2]
            )
            if offset <= max(1.25, 0.35 * max(existing.band_width_px, candidate.band_width_px)):
                duplicate = True
                break
        if not duplicate:
            selected.append(candidate)
        if len(selected) >= max_candidates:
            break
    return selected

"""White-line court keypoint detection from a single video frame."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

from threed.racketsport.court_calibration import homography_from_planar_points, project_planar_points
from threed.racketsport.court_keypoint_net import PICKLEBALL_KEYPOINTS, PICKLEBALL_KEYPOINT_BY_NAME


Segment = tuple[tuple[float, float], tuple[float, float]]


@dataclass(frozen=True)
class DetectedCourtKeypoints:
    keypoints: dict[str, dict[str, Any]]
    semantic_lines: dict[str, list[list[float]]]
    raw_segment_count: int
    merged_line_count: int
    confidence: float


@dataclass(frozen=True)
class _Line:
    a: float
    b: float
    c: float
    angle_deg: float


@dataclass(frozen=True)
class _LineGroup:
    line: _Line
    segment: Segment
    support_length_px: float
    source_segment_count: int


REQUIRED_SEMANTIC_LINES: tuple[str, ...] = (
    "far_baseline",
    "far_nvz",
    "net",
    "near_nvz",
    "near_baseline",
    "left_sideline",
    "centerline",
    "right_sideline",
)


def detect_court_keypoints_from_image(
    image: Any,
    *,
    cv2_module: Any | None = None,
    white_threshold: int = 200,
) -> DetectedCourtKeypoints:
    """Detect the pickleball court taxonomy from visible white court lines."""

    cv2 = cv2_module or _cv2()
    if image is None or not hasattr(image, "shape") or len(image.shape) < 2:
        raise ValueError("image must be an OpenCV-style array")
    height, width = image.shape[:2]
    if width <= 0 or height <= 0:
        raise ValueError("image must have positive dimensions")

    mask = _white_line_mask(image, cv2_module=cv2, white_threshold=white_threshold)
    segments = _detect_hough_segments(mask, cv2_module=cv2)
    merged_lines = _merge_segments(segments)
    if len(merged_lines) < 4:
        raise ValueError("not enough court-line candidates were detected")

    high_support_candidate = _try_high_support_near_strip_keypoints(
        merged_lines,
        mask=mask,
        cv2_module=cv2,
        width=float(width),
        height=float(height),
    )
    if high_support_candidate is not None:
        keypoints, semantic_segments, confidence = high_support_candidate
        return DetectedCourtKeypoints(
            keypoints=keypoints,
            semantic_lines=semantic_segments,
            raw_segment_count=len(segments),
            merged_line_count=len(merged_lines),
            confidence=confidence,
        )

    semantic_candidate = _try_semantic_line_keypoints(
        merged_lines,
        mask=mask,
        cv2_module=cv2,
        width=float(width),
        height=float(height),
    )
    if semantic_candidate is not None:
        keypoints, semantic_segments, confidence = semantic_candidate
        return DetectedCourtKeypoints(
            keypoints=keypoints,
            semantic_lines=semantic_segments,
            raw_segment_count=len(segments),
            merged_line_count=len(merged_lines),
            confidence=confidence,
        )

    near_strip = _select_near_strip_lines(merged_lines, width=float(width), height=float(height))
    refined = {
        name: _refine_line_from_mask(mask, group.line, cv2_module=cv2, max_distance_px=6.0)
        for name, group in near_strip.items()
    }
    keypoints = _keypoints_from_near_strip_homography(refined)
    semantic_segments = _semantic_segments_from_keypoints(keypoints)
    confidence = min(1.0, sum(group.support_length_px for group in near_strip.values()) / (float(width) * 2.0))
    return DetectedCourtKeypoints(
        keypoints=keypoints,
        semantic_lines=semantic_segments,
        raw_segment_count=len(segments),
        merged_line_count=len(merged_lines),
        confidence=confidence,
    )


def _try_semantic_line_keypoints(
    groups: Sequence[_LineGroup],
    *,
    mask: Any,
    cv2_module: Any,
    width: float,
    height: float,
) -> tuple[dict[str, dict[str, Any]], dict[str, list[list[float]]], float] | None:
    try:
        semantic_groups = _select_semantic_lines(groups, width=width, height=height)
        semantic_lines = {
            name: _refine_line_from_mask(mask, group.line, cv2_module=cv2_module, max_distance_px=6.0)
            for name, group in semantic_groups.items()
        }
        semantic_segments = {
            name: _segment_for_image(line, width=width, height=height)
            for name, line in semantic_lines.items()
        }
        keypoints = keypoints_from_semantic_lines(semantic_segments)
    except ValueError:
        return None
    if not _keypoints_are_plausible(keypoints, width=width, height=height):
        return None
    confidence = min(1.0, sum(group.support_length_px for group in semantic_groups.values()) / (float(width) * 3.0))
    return keypoints, semantic_segments, confidence


def _try_high_support_near_strip_keypoints(
    groups: Sequence[_LineGroup],
    *,
    mask: Any,
    cv2_module: Any,
    width: float,
    height: float,
) -> tuple[dict[str, dict[str, Any]], dict[str, list[list[float]]], float] | None:
    side_pool = sorted(
        [
            group
            for group in groups
            if abs(group.line.angle_deg) > 5.0 and _segment_length(group.segment) > width * 0.05
        ],
        key=lambda group: group.support_length_px,
        reverse=True,
    )[:4]
    if len(side_pool) < 2:
        return None

    best: tuple[float, dict[str, dict[str, Any]], dict[str, list[list[float]]], float] | None = None
    for left_index, first in enumerate(side_pool):
        for second in side_pool[left_index + 1 :]:
            left, right = sorted((first, second), key=lambda group: _line_x_at(group.line, height * 0.52))
            cross_candidates = [
                group
                for group in groups
                if abs(group.line.angle_deg) <= 8.0
                and _segment_length(group.segment) > width * 0.04
                and _cross_line_intersections_are_sane(group.line, left.line, right.line, width=width, height=height)
            ]
            cross_candidates = sorted(
                _dedupe_parallel_groups(cross_candidates, x_for_order=width * 0.35),
                key=lambda group: _mean_side_intersection_y(group.line, left.line, right.line),
            )
            if len(cross_candidates) < 2:
                continue
            min_cross_separation_px = max(40.0, height * 0.08)
            cross_y = {
                group: _mean_side_intersection_y(group.line, left.line, right.line)
                for group in cross_candidates
            }
            for near_baseline in cross_candidates[1:]:
                baseline_y = cross_y[near_baseline]
                for near_nvz in cross_candidates:
                    separation = baseline_y - cross_y[near_nvz]
                    if separation < min_cross_separation_px:
                        continue
                    refined = {
                        "left_sideline": _refine_line_from_mask(mask, left.line, cv2_module=cv2_module, max_distance_px=6.0),
                        "right_sideline": _refine_line_from_mask(mask, right.line, cv2_module=cv2_module, max_distance_px=6.0),
                        "near_nvz": _refine_line_from_mask(mask, near_nvz.line, cv2_module=cv2_module, max_distance_px=6.0),
                        "near_baseline": _refine_line_from_mask(
                            mask,
                            near_baseline.line,
                            cv2_module=cv2_module,
                            max_distance_px=6.0,
                        ),
                    }
                    try:
                        keypoints = _keypoints_from_near_strip_homography(refined)
                    except ValueError:
                        continue
                    if not _keypoints_are_plausible(keypoints, width=width, height=height):
                        continue
                    semantic_segments = _semantic_segments_from_keypoints(keypoints)
                    score = (
                        left.support_length_px
                        + right.support_length_px
                        + near_nvz.support_length_px
                        + near_baseline.support_length_px
                    )
                    confidence = min(1.0, score / (float(width) * 2.0))
                    if best is None or score > best[0]:
                        best = (score, keypoints, semantic_segments, confidence)
    if best is None:
        return None
    return best[1], best[2], best[3]


def _keypoints_are_plausible(keypoints: Mapping[str, Mapping[str, Any]], *, width: float, height: float) -> bool:
    margin = max(width, height) * 0.25
    for keypoint in keypoints.values():
        xy = keypoint.get("xy")
        if not isinstance(xy, list) or len(xy) != 2:
            return False
        x, y = float(xy[0]), float(xy[1])
        if not (math.isfinite(x) and math.isfinite(y)):
            return False
        if x < -margin or x > width + margin or y < -margin or y > height + margin:
            return False

    try:
        far_left = _xy_from_keypoint(keypoints["far_left_corner"])
        far_right = _xy_from_keypoint(keypoints["far_right_corner"])
        near_left = _xy_from_keypoint(keypoints["near_left_corner"])
        near_right = _xy_from_keypoint(keypoints["near_right_corner"])
        near_nvz_left = _xy_from_keypoint(keypoints["near_nvz_left"])
        near_nvz_right = _xy_from_keypoint(keypoints["near_nvz_right"])
    except (KeyError, ValueError):
        return False

    if far_left[0] >= far_right[0] or near_left[0] >= near_right[0]:
        return False
    far_y = (far_left[1] + far_right[1]) / 2.0
    near_nvz_y = (near_nvz_left[1] + near_nvz_right[1]) / 2.0
    near_y = (near_left[1] + near_right[1]) / 2.0
    return far_y < near_nvz_y < near_y


def keypoints_from_semantic_lines(semantic_lines: Mapping[str, Sequence[Sequence[float]]]) -> dict[str, dict[str, Any]]:
    """Convert semantic court lines into the 15-keypoint pickleball taxonomy."""

    missing = [name for name in REQUIRED_SEMANTIC_LINES if name not in semantic_lines]
    if missing:
        raise ValueError(f"missing semantic court lines: {', '.join(missing)}")
    lines = {
        name: _line_from_segment(_segment_from_raw(semantic_lines[name]))
        for name in REQUIRED_SEMANTIC_LINES
    }

    points = {
        "near_left_corner": _intersection(lines["left_sideline"], lines["near_baseline"]),
        "near_baseline_center": _intersection(lines["centerline"], lines["near_baseline"]),
        "near_right_corner": _intersection(lines["right_sideline"], lines["near_baseline"]),
        "far_right_corner": _intersection(lines["right_sideline"], lines["far_baseline"]),
        "far_baseline_center": _intersection(lines["centerline"], lines["far_baseline"]),
        "far_left_corner": _intersection(lines["left_sideline"], lines["far_baseline"]),
        "near_nvz_left": _intersection(lines["left_sideline"], lines["near_nvz"]),
        "near_nvz_center": _intersection(lines["centerline"], lines["near_nvz"]),
        "near_nvz_right": _intersection(lines["right_sideline"], lines["near_nvz"]),
        "net_left_sideline": _intersection(lines["left_sideline"], lines["net"]),
        "net_center": _intersection(lines["centerline"], lines["net"]),
        "net_right_sideline": _intersection(lines["right_sideline"], lines["net"]),
        "far_nvz_left": _intersection(lines["left_sideline"], lines["far_nvz"]),
        "far_nvz_center": _intersection(lines["centerline"], lines["far_nvz"]),
        "far_nvz_right": _intersection(lines["right_sideline"], lines["far_nvz"]),
    }
    return {
        name: {
            "xy": [float(xy[0]), float(xy[1])],
            "confidence": 1.0,
            "source": "auto_white_line_intersections",
        }
        for name, xy in points.items()
    }


def _keypoints_from_near_strip_homography(lines: Mapping[str, _Line]) -> dict[str, dict[str, Any]]:
    left = lines["left_sideline"]
    right = lines["right_sideline"]
    near_nvz = lines["near_nvz"]
    near_baseline = lines["near_baseline"]
    image_points = [
        _intersection(left, near_nvz),
        _intersection(right, near_nvz),
        _intersection(left, near_baseline),
        _intersection(right, near_baseline),
    ]
    world_points = [
        PICKLEBALL_KEYPOINT_BY_NAME["near_nvz_left"].world_xyz_m,
        PICKLEBALL_KEYPOINT_BY_NAME["near_nvz_right"].world_xyz_m,
        PICKLEBALL_KEYPOINT_BY_NAME["near_left_corner"].world_xyz_m,
        PICKLEBALL_KEYPOINT_BY_NAME["near_right_corner"].world_xyz_m,
    ]
    homography = homography_from_planar_points(world_points, image_points)
    projected = project_planar_points(homography, [point.world_xyz_m for point in PICKLEBALL_KEYPOINTS])
    return {
        point.name: {
            "xy": [float(xy[0]), float(xy[1])],
            "confidence": 1.0,
            "source": "auto_white_line_near_strip_homography",
        }
        for point, xy in zip(PICKLEBALL_KEYPOINTS, projected, strict=True)
    }


def _semantic_segments_from_keypoints(keypoints: Mapping[str, Mapping[str, Any]]) -> dict[str, list[list[float]]]:
    return {
        "far_baseline": [_xy_from_keypoint(keypoints["far_left_corner"]), _xy_from_keypoint(keypoints["far_right_corner"])],
        "far_nvz": [_xy_from_keypoint(keypoints["far_nvz_left"]), _xy_from_keypoint(keypoints["far_nvz_right"])],
        "net": [_xy_from_keypoint(keypoints["net_left_sideline"]), _xy_from_keypoint(keypoints["net_right_sideline"])],
        "near_nvz": [_xy_from_keypoint(keypoints["near_nvz_left"]), _xy_from_keypoint(keypoints["near_nvz_right"])],
        "near_baseline": [_xy_from_keypoint(keypoints["near_left_corner"]), _xy_from_keypoint(keypoints["near_right_corner"])],
        "left_sideline": [_xy_from_keypoint(keypoints["far_left_corner"]), _xy_from_keypoint(keypoints["near_left_corner"])],
        "centerline": [
            _xy_from_keypoint(keypoints["far_baseline_center"]),
            _xy_from_keypoint(keypoints["near_baseline_center"]),
        ],
        "right_sideline": [_xy_from_keypoint(keypoints["far_right_corner"]), _xy_from_keypoint(keypoints["near_right_corner"])],
    }


def _xy_from_keypoint(keypoint: Mapping[str, Any]) -> list[float]:
    xy = keypoint.get("xy")
    if not isinstance(xy, list) or len(xy) != 2:
        raise ValueError("keypoint is missing xy")
    return [float(xy[0]), float(xy[1])]


def _white_line_mask(image: Any, *, cv2_module: Any, white_threshold: int) -> Any:
    cv2 = cv2_module
    import numpy as np

    threshold = int(max(0, min(255, white_threshold)))
    if len(image.shape) == 2:
        mask = (image >= threshold).astype(np.uint8) * 255
    else:
        bgr = image.astype(np.int16)
        channel_min = bgr.min(axis=2)
        channel_max = bgr.max(axis=2)
        mask = ((channel_min >= threshold) & ((channel_max - channel_min) <= 80)).astype(np.uint8) * 255

    height = int(mask.shape[0])
    mask[: int(round(height * 0.25)), :] = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def _detect_hough_segments(mask: Any, *, cv2_module: Any) -> list[Segment]:
    cv2 = cv2_module
    height, width = mask.shape[:2]
    edges = cv2.Canny(mask, 50, 150)
    raw_lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=math.pi / 180.0,
        threshold=45,
        minLineLength=max(40, int(round(min(width, height) * 0.12))),
        maxLineGap=max(12, int(round(max(width, height) * 0.015))),
    )
    if raw_lines is None:
        return []
    segments: list[Segment] = []
    for raw in raw_lines.reshape(-1, 4):
        x1, y1, x2, y2 = [float(value) for value in raw]
        if math.hypot(x2 - x1, y2 - y1) >= 40.0:
            segments.append(((x1, y1), (x2, y2)))
    return segments


def _merge_segments(segments: Sequence[Segment]) -> list[_LineGroup]:
    groups: list[list[Segment]] = []
    for segment in sorted(segments, key=_segment_length, reverse=True):
        line = _line_from_segment(segment)
        for group in groups:
            reference = _line_from_segment(_longest_segment(group))
            if _angle_delta(line.angle_deg, reference.angle_deg) <= 3.0 and _line_distance(line, reference) <= 14.0:
                group.append(segment)
                break
        else:
            groups.append([segment])

    merged = [_line_group(group) for group in groups]
    return sorted(merged, key=lambda group: group.support_length_px, reverse=True)


def _select_near_strip_lines(groups: Sequence[_LineGroup], *, width: float, height: float) -> dict[str, _LineGroup]:
    longitudinal_candidates = [
        group for group in groups if group.line.angle_deg > 5.0 and _segment_length(group.segment) > width * 0.05
    ]
    if len(longitudinal_candidates) < 2:
        raise ValueError("could not classify left/right court sideline candidates")

    y_ref = height * 0.52
    ordered_longitudinal = sorted(longitudinal_candidates, key=lambda group: _line_x_at(group.line, y_ref))
    left = ordered_longitudinal[0]
    right = ordered_longitudinal[-1]

    cross_candidates = [
        group
        for group in groups
        if group.line.angle_deg <= 5.0
        and _segment_length(group.segment) > width * 0.04
        and _cross_line_intersections_are_sane(group.line, left.line, right.line, width=width, height=height)
    ]
    cross_candidates = _dedupe_parallel_groups(cross_candidates, x_for_order=width * 0.35)
    if len(cross_candidates) < 2:
        raise ValueError("could not classify near NVZ and near baseline candidates")
    ordered_cross = sorted(
        cross_candidates,
        key=lambda group: _mean_side_intersection_y(group.line, left.line, right.line),
    )
    return {
        "left_sideline": left,
        "right_sideline": right,
        "near_nvz": ordered_cross[-2],
        "near_baseline": ordered_cross[-1],
    }


def _select_semantic_lines(groups: Sequence[_LineGroup], *, width: float, height: float) -> dict[str, _LineGroup]:
    cross_candidates = [
        group for group in groups if group.line.angle_deg <= 5.0 and _segment_length(group.segment) > width * 0.06
    ]
    longitudinal_candidates = [
        group for group in groups if group.line.angle_deg > 5.0 and _segment_length(group.segment) > width * 0.06
    ]
    if len(cross_candidates) < 5 or len(longitudinal_candidates) < 3:
        raise ValueError("could not classify enough semantic court-line candidates")

    y_ref = height * 0.52
    ordered_longitudinal = sorted(
        longitudinal_candidates,
        key=lambda group: _line_x_at(group.line, y_ref),
    )
    left = ordered_longitudinal[0]
    right_sideline = ordered_longitudinal[-1]
    center_target = (_line_x_at(left.line, y_ref) + _line_x_at(right_sideline.line, y_ref)) / 2.0
    centerline = min(
        ordered_longitudinal[1:-1],
        key=lambda group: abs(_line_x_at(group.line, y_ref) - center_target),
    )

    x_ref = width * 0.35
    horizontal_pool = _dedupe_parallel_groups(
        [
            group
            for group in cross_candidates
            if _cross_line_intersections_are_sane(group.line, left.line, right_sideline.line, width=width, height=height)
        ],
        x_for_order=x_ref,
    )
    if len(horizontal_pool) < 5:
        horizontal_pool = _dedupe_parallel_groups(cross_candidates, x_for_order=x_ref)
    if len(horizontal_pool) < 5:
        raise ValueError("could not classify the far/net/NVZ/baseline court-line candidates")
    ordered_cross = sorted(horizontal_pool, key=lambda group: _mean_side_intersection_y(group.line, left.line, right_sideline.line))
    far_baseline, far_nvz, net, near_nvz = ordered_cross[:4]
    near_baseline = ordered_cross[-1]

    if len({id(group) for group in (far_baseline, far_nvz, net, near_nvz, near_baseline)}) < 5:
        raise ValueError("semantic cross-court lines were not distinct")

    return {
        "far_baseline": far_baseline,
        "far_nvz": far_nvz,
        "net": net,
        "near_nvz": near_nvz,
        "near_baseline": near_baseline,
        "left_sideline": left,
        "centerline": centerline,
        "right_sideline": right_sideline,
    }

def _dedupe_parallel_groups(groups: Sequence[_LineGroup], *, x_for_order: float) -> list[_LineGroup]:
    ordered = sorted(groups, key=lambda group: (round(_line_y_at(group.line, x_for_order) / 12.0), -group.support_length_px))
    selected: list[_LineGroup] = []
    for group in ordered:
        y = _line_y_at(group.line, x_for_order)
        if any(abs(y - _line_y_at(existing.line, x_for_order)) < 18.0 for existing in selected):
            continue
        selected.append(group)
    return selected


def _refine_line_from_mask(mask: Any, line: _Line, *, cv2_module: Any, max_distance_px: float = 10.0) -> _Line:
    cv2 = cv2_module
    import numpy as np

    ys, xs = np.nonzero(mask)
    if len(xs) < 2:
        return line
    distances = np.abs(line.a * xs.astype(np.float64) + line.b * ys.astype(np.float64) + line.c)
    keep = distances <= max_distance_px
    if int(keep.sum()) < 8:
        return line
    points = np.column_stack([xs[keep].astype(np.float32), ys[keep].astype(np.float32)])
    vx, vy, x0, y0 = [float(value) for value in cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01).reshape(-1)]
    return _line_from_point_direction((x0, y0), (vx, vy))


def _line_group(segments: Sequence[Segment]) -> _LineGroup:
    points = [point for segment in segments for point in segment]
    line = _fit_line(points)
    segment = _extent_segment(points, line)
    return _LineGroup(
        line=line,
        segment=segment,
        support_length_px=sum(_segment_length(segment) for segment in segments),
        source_segment_count=len(segments),
    )


def _fit_line(points: Sequence[tuple[float, float]]) -> _Line:
    mean_x = sum(point[0] for point in points) / len(points)
    mean_y = sum(point[1] for point in points) / len(points)
    sxx = sum((point[0] - mean_x) ** 2 for point in points)
    syy = sum((point[1] - mean_y) ** 2 for point in points)
    sxy = sum((point[0] - mean_x) * (point[1] - mean_y) for point in points)
    angle = 0.5 * math.atan2(2.0 * sxy, sxx - syy)
    return _line_from_point_direction((mean_x, mean_y), (math.cos(angle), math.sin(angle)))


def _line_from_segment(segment: Segment) -> _Line:
    (x1, y1), (x2, y2) = segment
    return _line_from_point_direction((x1, y1), (x2 - x1, y2 - y1))


def _line_from_point_direction(point: tuple[float, float], direction: tuple[float, float]) -> _Line:
    dx, dy = direction
    length = math.hypot(dx, dy)
    if length <= 1e-9:
        raise ValueError("line direction must be non-zero")
    dx /= length
    dy /= length
    angle = math.degrees(math.atan2(dy, dx))
    if angle < -90.0:
        angle += 180.0
        dx = -dx
        dy = -dy
    elif angle > 90.0:
        angle -= 180.0
        dx = -dx
        dy = -dy
    a = -dy
    b = dx
    c = -(a * point[0] + b * point[1])
    if c < 0.0:
        a, b, c = -a, -b, -c
    return _Line(a=a, b=b, c=c, angle_deg=angle)


def _segment_from_raw(raw: Sequence[Sequence[float]]) -> Segment:
    if len(raw) != 2:
        raise ValueError("semantic line segment must have exactly two points")
    return (
        (float(raw[0][0]), float(raw[0][1])),
        (float(raw[1][0]), float(raw[1][1])),
    )


def _intersection(first: _Line, second: _Line) -> tuple[float, float]:
    det = first.a * second.b - second.a * first.b
    if abs(det) <= 1e-9:
        raise ValueError("semantic court lines are parallel")
    x = (first.b * second.c - second.b * first.c) / det
    y = (second.a * first.c - first.a * second.c) / det
    return (float(x), float(y))


def _segment_for_image(line: _Line, *, width: float, height: float) -> list[list[float]]:
    points: list[tuple[float, float]] = []
    for x in (0.0, width - 1.0):
        if abs(line.b) > 1e-9:
            y = -(line.a * x + line.c) / line.b
            if -height <= y <= height * 2.0:
                points.append((x, y))
    for y in (0.0, height - 1.0):
        if abs(line.a) > 1e-9:
            x = -(line.b * y + line.c) / line.a
            if -width <= x <= width * 2.0:
                points.append((x, y))
    if len(points) < 2:
        point = (-line.a * line.c, -line.b * line.c)
        dx, dy = line.b, -line.a
        points = [(point[0] - dx * width, point[1] - dy * width), (point[0] + dx * width, point[1] + dy * width)]
    first, second = max(
        ((p0, p1) for idx, p0 in enumerate(points) for p1 in points[idx + 1 :]),
        key=lambda pair: math.dist(pair[0], pair[1]),
    )
    return [[float(first[0]), float(first[1])], [float(second[0]), float(second[1])]]


def _extent_segment(points: Sequence[tuple[float, float]], line: _Line) -> Segment:
    dx, dy = line.b, -line.a
    anchor = (-line.a * line.c, -line.b * line.c)
    projections = [((point[0] - anchor[0]) * dx + (point[1] - anchor[1]) * dy, point) for point in points]
    min_projection = min(value for value, _ in projections)
    max_projection = max(value for value, _ in projections)
    return (
        (anchor[0] + dx * min_projection, anchor[1] + dy * min_projection),
        (anchor[0] + dx * max_projection, anchor[1] + dy * max_projection),
    )


def _longest_segment(segments: Sequence[Segment]) -> Segment:
    return max(segments, key=_segment_length)


def _segment_length(segment: Segment) -> float:
    return math.dist(segment[0], segment[1])


def _angle_delta(first: float, second: float) -> float:
    delta = abs(first - second)
    return min(delta, 180.0 - delta)


def _line_distance(first: _Line, second: _Line) -> float:
    dot = first.a * second.a + first.b * second.b
    second_c = second.c if dot >= 0.0 else -second.c
    return abs(first.c - second_c)


def _line_y_at(line: _Line, x: float) -> float:
    if abs(line.b) <= 1e-9:
        return float("inf")
    return float(-(line.a * x + line.c) / line.b)


def _line_x_at(line: _Line, y: float) -> float:
    if abs(line.a) <= 1e-9:
        return float("inf")
    return float(-(line.b * y + line.c) / line.a)


def _cross_line_intersections_are_sane(
    cross: _Line,
    left: _Line,
    right: _Line,
    *,
    width: float,
    height: float,
) -> bool:
    try:
        left_xy = _intersection(cross, left)
        right_xy = _intersection(cross, right)
    except ValueError:
        return False
    if left_xy[0] > right_xy[0]:
        return False
    x_margin = max(20.0, width * 0.05)
    y_margin = max(20.0, height * 0.05)
    return (
        -x_margin <= left_xy[0] <= width + x_margin
        and -x_margin <= right_xy[0] <= width + x_margin
        and -y_margin <= left_xy[1] <= height + y_margin
        and -y_margin <= right_xy[1] <= height + y_margin
    )


def _mean_side_intersection_y(cross: _Line, left: _Line, right: _Line) -> float:
    left_xy = _intersection(cross, left)
    right_xy = _intersection(cross, right)
    return (left_xy[1] + right_xy[1]) / 2.0


def _min_x(segment: Segment) -> float:
    return min(segment[0][0], segment[1][0])


def _max_x(segment: Segment) -> float:
    return max(segment[0][0], segment[1][0])


def _max_y(segment: Segment) -> float:
    return max(segment[0][1], segment[1][1])


def _cv2() -> Any:
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("court line keypoint detection requires opencv-python") from exc
    return cv2

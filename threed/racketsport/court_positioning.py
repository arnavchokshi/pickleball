"""Metric court positioning primitives for the ARKit-floor-plane path.

These helpers implement the CPU geometry contract from the court positioning
spec. They do not load detector or pose models; model-backed stages must call
these functions with real keypoints/feet after undistortion.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Literal, Mapping, Sequence

from threed.racketsport.court_keypoint_net import PICKLEBALL_KEYPOINT_BY_NAME

LINE_WIDTH_M = 0.0508
PICKLEBALL_HALF_WIDTH_M = 3.048
PICKLEBALL_HALF_LENGTH_M = 6.7056
PICKLEBALL_NVZ_M = 2.1336
SCALE_DRIFT_GATE = 0.02
PLACEMENT_RESIDUAL_GATE_M = 0.03


MetricConfidence = Literal["high", "med", "low"]
CaptureQualityGrade = Literal["good", "warn", "poor"]
BoundaryName = Literal["near_kitchen", "far_kitchen", "sideline", "baseline", "centerline"]
Decision = Literal["in", "out", "kitchen", "too_close_to_call"]


@dataclass(frozen=True)
class CameraFloorGeometry:
    intrinsics: Mapping[str, Any]
    camera_origin_world: Sequence[float]
    R_world_camera: Sequence[Sequence[float]]
    floor_plane_point: Sequence[float]
    floor_plane_normal: Sequence[float]


@dataclass(frozen=True)
class ResidualErrorMeters:
    median: float
    p95: float
    max: float


@dataclass(frozen=True)
class MetricCourtPlacement:
    T_world_court: list[list[float]]
    scale_estimate: float
    residual_error_m: ResidualErrorMeters
    metric_confidence: MetricConfidence
    gate_failures: tuple[str, ...]
    solved_keypoints: tuple[str, ...]


@dataclass(frozen=True)
class CourtDecisionInput:
    boundary: BoundaryName
    foot_court_xy: Sequence[float]
    sigma_p_m: float
    metric_confidence: MetricConfidence
    capture_quality_grade: CaptureQualityGrade = "good"
    z_sigma: float = 2.0


@dataclass(frozen=True)
class CourtBoundaryDecision:
    boundary: BoundaryName
    decision: Decision
    signed_dist_m: float
    sigma_p_m: float
    metric_confidence: MetricConfidence
    capture_quality_grade: CaptureQualityGrade


@dataclass(frozen=True)
class CourtGateInput:
    reprojection_p95_px: float
    metric_confidence: MetricConfidence
    keypoint_inlier_count: int
    required_line_recovered: bool
    capture_quality_grade: CaptureQualityGrade
    drift_or_recalibration: bool
    requested_line_call_decision: Decision | None = None


def back_project_pixel_to_floor(pixel_uv: Sequence[float], geometry: CameraFloorGeometry) -> list[float]:
    """Back-project an undistorted pixel through the camera and onto the floor plane."""

    u, v = _xy(pixel_uv, "pixel_uv")
    fx = _positive_float(geometry.intrinsics.get("fx"), "intrinsics.fx")
    fy = _positive_float(geometry.intrinsics.get("fy"), "intrinsics.fy")
    cx = _finite_float(geometry.intrinsics.get("cx"), "intrinsics.cx")
    cy = _finite_float(geometry.intrinsics.get("cy"), "intrinsics.cy")
    origin = _vec3(geometry.camera_origin_world, "camera_origin_world")
    plane_point = _vec3(geometry.floor_plane_point, "floor_plane_point")
    plane_normal = _unit(_vec3(geometry.floor_plane_normal, "floor_plane_normal"), "floor_plane_normal")
    rotation = _mat3(geometry.R_world_camera, "R_world_camera")

    direction_camera = [(u - cx) / fx, (v - cy) / fy, 1.0]
    direction_world = _mat_vec(rotation, direction_camera)
    denom = _dot(plane_normal, direction_world)
    if math.isclose(denom, 0.0, abs_tol=1e-12):
        raise ValueError("camera ray is parallel to the floor plane")
    distance = _dot(plane_normal, _sub(plane_point, origin)) / denom
    if distance < 0.0:
        raise ValueError("floor plane intersection is behind the camera")
    return [origin[idx] + distance * direction_world[idx] for idx in range(3)]


def estimate_ground_sample_distance(pixel_uv: Sequence[float], geometry: CameraFloorGeometry) -> float:
    """Estimate local ground meters per pixel from finite differences on the floor plane."""

    u, v = _xy(pixel_uv, "pixel_uv")
    center = back_project_pixel_to_floor([u, v], geometry)
    right = back_project_pixel_to_floor([u + 1.0, v], geometry)
    down = back_project_pixel_to_floor([u, v + 1.0], geometry)
    return (_distance(center, right) + _distance(center, down)) / 2.0


def estimate_position_uncertainty(
    *,
    pixel_error_px: float,
    gsd_m_per_px: float,
    plane_sigma_m: float,
    calibration_sigma_m: float,
) -> float:
    """Combine pixel, ARKit plane, and calibration terms into one positional sigma."""

    pixel_error = _nonnegative_float(pixel_error_px, "pixel_error_px")
    gsd = _nonnegative_float(gsd_m_per_px, "gsd_m_per_px")
    plane_sigma = _nonnegative_float(plane_sigma_m, "plane_sigma_m")
    calibration_sigma = _nonnegative_float(calibration_sigma_m, "calibration_sigma_m")
    return math.sqrt((pixel_error * gsd) ** 2 + plane_sigma**2 + calibration_sigma**2)


def solve_metric_court_placement(world_keypoints: Mapping[str, Sequence[float]]) -> MetricCourtPlacement:
    """Solve a scale-locked 2D rigid court placement from canonical keypoints to world points."""

    names = tuple(name for name in PICKLEBALL_KEYPOINT_BY_NAME if name in world_keypoints)
    if len(names) < 3:
        raise ValueError("metric court placement requires at least 3 known keypoints")

    court_xy = [
        (
            float(PICKLEBALL_KEYPOINT_BY_NAME[name].world_xyz_m[0]),
            float(PICKLEBALL_KEYPOINT_BY_NAME[name].world_xyz_m[1]),
        )
        for name in names
    ]
    world_xyz = [_vec3(world_keypoints[name], f"world_keypoints.{name}") for name in names]
    world_xy = [(point[0], point[1]) for point in world_xyz]
    court_center = _mean2(court_xy)
    world_center = _mean2(world_xy)

    numerator_sin = 0.0
    numerator_cos = 0.0
    denominator = 0.0
    for court, world in zip(court_xy, world_xy, strict=True):
        qx = court[0] - court_center[0]
        qy = court[1] - court_center[1]
        px = world[0] - world_center[0]
        py = world[1] - world_center[1]
        numerator_sin += qx * py - qy * px
        numerator_cos += qx * px + qy * py
        denominator += qx * qx + qy * qy
    if math.isclose(denominator, 0.0, abs_tol=1e-12):
        raise ValueError("court keypoints are degenerate")

    theta = math.atan2(numerator_sin, numerator_cos)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    scale_numerator = 0.0
    for court, world in zip(court_xy, world_xy, strict=True):
        qx = court[0] - court_center[0]
        qy = court[1] - court_center[1]
        rotated_x = cos_t * qx - sin_t * qy
        rotated_y = sin_t * qx + cos_t * qy
        px = world[0] - world_center[0]
        py = world[1] - world_center[1]
        scale_numerator += px * rotated_x + py * rotated_y
    scale_estimate = scale_numerator / denominator

    tx = world_center[0] - (cos_t * court_center[0] - sin_t * court_center[1])
    ty = world_center[1] - (sin_t * court_center[0] + cos_t * court_center[1])
    tz = sum(point[2] for point in world_xyz) / len(world_xyz)
    transform = [
        [cos_t, -sin_t, 0.0, tx],
        [sin_t, cos_t, 0.0, ty],
        [0.0, 0.0, 1.0, tz],
        [0.0, 0.0, 0.0, 1.0],
    ]

    residuals = [
        _distance(transform_court_to_world(PICKLEBALL_KEYPOINT_BY_NAME[name].world_xyz_m, transform), point)
        for name, point in zip(names, world_xyz, strict=True)
    ]
    residual_error = ResidualErrorMeters(
        median=_percentile(residuals, 50),
        p95=_percentile(residuals, 95),
        max=max(residuals),
    )
    gate_failures: list[str] = []
    if abs(scale_estimate - 1.0) > SCALE_DRIFT_GATE:
        gate_failures.append("scale_drift_gt_2pct")
    if residual_error.p95 > PLACEMENT_RESIDUAL_GATE_M:
        gate_failures.append("residual_p95_gt_3cm")
    confidence: MetricConfidence = "high" if not gate_failures else "low"
    return MetricCourtPlacement(
        T_world_court=transform,
        scale_estimate=float(scale_estimate),
        residual_error_m=residual_error,
        metric_confidence=confidence,
        gate_failures=tuple(gate_failures),
        solved_keypoints=names,
    )


def transform_court_to_world(court_xyz: Sequence[float], T_world_court: Sequence[Sequence[float]]) -> list[float]:
    point = _vec3(court_xyz, "court_xyz")
    transform = _mat4(T_world_court, "T_world_court")
    return [
        transform[row][0] * point[0]
        + transform[row][1] * point[1]
        + transform[row][2] * point[2]
        + transform[row][3]
        for row in range(3)
    ]


def decide_court_boundary(decision_input: CourtDecisionInput) -> CourtBoundaryDecision:
    sigma = _nonnegative_float(decision_input.sigma_p_m, "sigma_p_m")
    z_sigma = _positive_float(decision_input.z_sigma, "z_sigma")
    x, y = _xy(decision_input.foot_court_xy, "foot_court_xy")
    if decision_input.capture_quality_grade not in {"good", "warn", "poor"}:
        raise ValueError("capture_quality_grade must be good, warn, or poor")
    if decision_input.boundary == "near_kitchen":
        signed_dist = _signed_distance_to_rect(
            x,
            y,
            x_min=-PICKLEBALL_HALF_WIDTH_M - LINE_WIDTH_M / 2.0,
            x_max=PICKLEBALL_HALF_WIDTH_M + LINE_WIDTH_M / 2.0,
            y_min=-PICKLEBALL_NVZ_M - LINE_WIDTH_M / 2.0,
            y_max=LINE_WIDTH_M / 2.0,
        )
    elif decision_input.boundary == "far_kitchen":
        signed_dist = _signed_distance_to_rect(
            x,
            y,
            x_min=-PICKLEBALL_HALF_WIDTH_M - LINE_WIDTH_M / 2.0,
            x_max=PICKLEBALL_HALF_WIDTH_M + LINE_WIDTH_M / 2.0,
            y_min=-LINE_WIDTH_M / 2.0,
            y_max=PICKLEBALL_NVZ_M + LINE_WIDTH_M / 2.0,
        )
    elif decision_input.boundary == "sideline":
        signed_dist = abs(x) - (PICKLEBALL_HALF_WIDTH_M + LINE_WIDTH_M / 2.0)
    elif decision_input.boundary == "baseline":
        signed_dist = abs(y) - (PICKLEBALL_HALF_LENGTH_M + LINE_WIDTH_M / 2.0)
    elif decision_input.boundary == "centerline":
        signed_dist = abs(x) - LINE_WIDTH_M / 2.0
    else:
        raise ValueError(f"unsupported boundary: {decision_input.boundary}")

    if decision_input.metric_confidence != "high" or decision_input.capture_quality_grade == "poor":
        outcome: Decision = "too_close_to_call"
    elif abs(signed_dist) <= sigma * z_sigma:
        outcome = "too_close_to_call"
    elif decision_input.boundary in {"near_kitchen", "far_kitchen"} and signed_dist > 0.0:
        outcome = "kitchen"
    elif decision_input.boundary in {"sideline", "baseline"} and signed_dist > 0.0:
        outcome = "out"
    elif decision_input.boundary == "centerline" and signed_dist > 0.0:
        outcome = "out"
    else:
        outcome = "in"
    return CourtBoundaryDecision(
        boundary=decision_input.boundary,
        decision=outcome,
        signed_dist_m=float(signed_dist),
        sigma_p_m=sigma,
        metric_confidence=decision_input.metric_confidence,
        capture_quality_grade=decision_input.capture_quality_grade,
    )


def court_escalation_reasons(gate: CourtGateInput) -> tuple[str, ...]:
    """Return Stage H escalation reasons without hiding low-confidence results."""

    reasons: list[str] = []
    if _nonnegative_float(gate.reprojection_p95_px, "reprojection_p95_px") > 5.0:
        reasons.append("court_reprojection_p95_gt_5px")
    if gate.metric_confidence != "high":
        reasons.append("metric_confidence_not_high")
    if _nonnegative_int(gate.keypoint_inlier_count, "keypoint_inlier_count") < 10:
        reasons.append("keypoint_inlier_count_lt_10")
    if not gate.required_line_recovered:
        reasons.append("required_line_unrecovered")
    if gate.capture_quality_grade == "poor":
        reasons.append("capture_quality_poor")
    elif gate.capture_quality_grade not in {"good", "warn"}:
        raise ValueError("capture_quality_grade must be good, warn, or poor")
    if gate.drift_or_recalibration:
        reasons.append("drift_or_recalibration")
    if gate.requested_line_call_decision == "too_close_to_call":
        reasons.append("line_call_too_close_to_call")
    return tuple(reasons)


def _signed_distance_to_rect(
    x: float,
    y: float,
    *,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> float:
    inside = x_min <= x <= x_max and y_min <= y <= y_max
    if inside:
        return min(x - x_min, x_max - x, y - y_min, y_max - y)
    dx = max(x_min - x, 0.0, x - x_max)
    dy = max(y_min - y, 0.0, y - y_max)
    return -math.hypot(dx, dy)


def _xy(value: Sequence[float], name: str) -> tuple[float, float]:
    if len(value) != 2:
        raise ValueError(f"{name} must contain 2 values")
    return (_finite_float(value[0], f"{name}[0]"), _finite_float(value[1], f"{name}[1]"))


def _vec3(value: Sequence[float], name: str) -> list[float]:
    if len(value) != 3:
        raise ValueError(f"{name} must contain 3 values")
    return [_finite_float(value[idx], f"{name}[{idx}]") for idx in range(3)]


def _mat3(value: Sequence[Sequence[float]], name: str) -> list[list[float]]:
    if len(value) != 3:
        raise ValueError(f"{name} must contain 3 rows")
    return [_vec3(row, f"{name}[{idx}]") for idx, row in enumerate(value)]


def _mat4(value: Sequence[Sequence[float]], name: str) -> list[list[float]]:
    if len(value) != 4:
        raise ValueError(f"{name} must contain 4 rows")
    matrix: list[list[float]] = []
    for row_idx, row in enumerate(value):
        if len(row) != 4:
            raise ValueError(f"{name}[{row_idx}] must contain 4 values")
        matrix.append([_finite_float(row[col_idx], f"{name}[{row_idx}][{col_idx}]") for col_idx in range(4)])
    return matrix


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _positive_float(value: Any, name: str) -> float:
    result = _finite_float(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _nonnegative_float(value: Any, name: str) -> float:
    result = _finite_float(value, name)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be a non-negative integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _unit(vector: Sequence[float], name: str) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if math.isclose(norm, 0.0, abs_tol=1e-12):
        raise ValueError(f"{name} must be non-zero")
    return [float(value) / norm for value in vector]


def _mat_vec(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> list[float]:
    return [sum(matrix[row][col] * vector[col] for col in range(3)) for row in range(3)]


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(float(a) * float(b) for a, b in zip(left, right, strict=True))


def _sub(left: Sequence[float], right: Sequence[float]) -> list[float]:
    return [float(a) - float(b) for a, b in zip(left, right, strict=True)]


def _distance(left: Sequence[float], right: Sequence[float]) -> float:
    return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(left, right, strict=True)))


def _mean2(points: Sequence[tuple[float, float]]) -> tuple[float, float]:
    return (
        sum(point[0] for point in points) / len(points),
        sum(point[1] for point in points) / len(points),
    )


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

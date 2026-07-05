"""Overlapping-court calibration helpers.

These utilities are intentionally opt-in. They target shared tennis/pickleball
courts where pickleball lines are painted in a distinct high-saturation color,
and they should fail closed when that assumption is false.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from .schemas import PICKLEBALL_COURT_KEYPOINT_NAMES


@dataclass(frozen=True)
class HSVPaintRange:
    name: str
    lower: tuple[int, int, int]
    upper: tuple[int, int, int]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("HSV paint range name must be non-empty")
        _validate_hsv_triplet(self.lower, "lower")
        _validate_hsv_triplet(self.upper, "upper")
        if any(low > high for low, high in zip(self.lower, self.upper, strict=True)):
            raise ValueError("HSV lower bounds must be <= upper bounds")


@dataclass(frozen=True)
class LineClusterConfig:
    hough_threshold: int = 34
    min_line_length_px: int = 36
    max_line_gap_px: int = 12
    angle_tolerance_deg: float = 5.0
    line_distance_tolerance_px: float = 12.0
    max_clusters: int = 4

    def __post_init__(self) -> None:
        if self.hough_threshold <= 0:
            raise ValueError("hough_threshold must be positive")
        if self.min_line_length_px <= 0:
            raise ValueError("min_line_length_px must be positive")
        if self.max_line_gap_px < 0:
            raise ValueError("max_line_gap_px must be non-negative")
        if self.angle_tolerance_deg <= 0.0:
            raise ValueError("angle_tolerance_deg must be positive")
        if self.line_distance_tolerance_px <= 0.0:
            raise ValueError("line_distance_tolerance_px must be positive")
        if self.max_clusters <= 0:
            raise ValueError("max_clusters must be positive")


@dataclass(frozen=True)
class BoundaryCluster:
    orientation: str
    line: tuple[float, float, float]
    segment: tuple[tuple[float, float], tuple[float, float]]
    support_length_px: float
    source_segment_count: int
    angle_deg: float

    def to_segment_payload(self, *, source: str, cluster_id: int) -> dict[str, Any]:
        return {
            "p1": [round(float(self.segment[0][0]), 3), round(float(self.segment[0][1]), 3)],
            "p2": [round(float(self.segment[1][0]), 3), round(float(self.segment[1][1]), 3)],
            "length_px": round(float(_segment_length(self.segment)), 3),
            "angle_deg": round(float(self.angle_deg), 3),
            "source": source,
            "cluster_id": int(cluster_id),
            "orientation": self.orientation,
            "cluster_support_length_px": round(float(self.support_length_px), 3),
            "source_segment_count": int(self.source_segment_count),
        }


@dataclass(frozen=True)
class BoundaryClusters:
    clusters: list[BoundaryCluster]
    raw_segment_count: int
    mask_support_ratio: float

    def to_payload(self, *, source: str) -> dict[str, Any]:
        return {
            "raw_segment_count": int(self.raw_segment_count),
            "boundary_cluster_count": len(self.clusters),
            "mask_support_ratio": round(float(self.mask_support_ratio), 6),
            "segments": [
                cluster.to_segment_payload(source=source, cluster_id=index)
                for index, cluster in enumerate(self.clusters)
            ],
        }


def _validate_hsv_triplet(value: tuple[int, int, int], name: str) -> None:
    if len(value) != 3:
        raise ValueError(f"{name} must be an HSV triplet")
    h, s, v = value
    if not (0 <= int(h) <= 179 and 0 <= int(s) <= 255 and 0 <= int(v) <= 255):
        raise ValueError(f"{name} must use OpenCV HSV ranges H=[0,179], S/V=[0,255]")


DEFAULT_PICKLEBALL_PAINT_RANGES: tuple[HSVPaintRange, ...] = (
    HSVPaintRange("yellow_pickleball_paint", (18, 70, 90), (42, 255, 255)),
    HSVPaintRange("blue_pickleball_paint", (88, 60, 70), (132, 255, 255)),
    HSVPaintRange("light_green_pickleball_paint", (44, 80, 120), (86, 255, 255)),
)

NET_TOP_KEYPOINT_NAMES = frozenset({"net_left_sideline", "net_center", "net_right_sideline"})
FLOOR_HOMOGRAPHY_KEYPOINT_NAMES: tuple[str, ...] = tuple(
    name for name in PICKLEBALL_COURT_KEYPOINT_NAMES if name not in NET_TOP_KEYPOINT_NAMES
)
CORNER_SEED_KEYPOINT_NAMES: tuple[str, ...] = (
    "near_left_corner",
    "near_right_corner",
    "far_right_corner",
    "far_left_corner",
)
DEFAULT_POINT_LINE_WEIGHT = 0.35
POINT_LINE_WEIGHT_SWEEP: tuple[float, ...] = (
    0.01,
    0.02,
    0.035,
    0.05,
    0.075,
    0.1,
    0.15,
    0.2,
    DEFAULT_POINT_LINE_WEIGHT,
    0.5,
    0.8,
    1.2,
)
DEFAULT_LINE_PIXEL_SAMPLE_COUNT = 9
LINE_INTERSECTION_QUALITY_GATE_PROFILES: tuple[dict[str, Any], ...] = (
    {
        "profile_id": "balanced_overlap20_dist16_angle10",
        "min_overlap_fraction": 0.20,
        "max_mean_perpendicular_distance_px": 16.0,
        "max_angle_diff_deg": 10.0,
    },
    {
        "profile_id": "tight_overlap35_dist12_angle8",
        "min_overlap_fraction": 0.35,
        "max_mean_perpendicular_distance_px": 12.0,
        "max_angle_diff_deg": 8.0,
    },
    {
        "profile_id": "very_tight_overlap50_dist8_angle6",
        "min_overlap_fraction": 0.50,
        "max_mean_perpendicular_distance_px": 8.0,
        "max_angle_diff_deg": 6.0,
    },
    {
        "profile_id": "tight_overlap35_dist12_angle8_model24",
        "min_overlap_fraction": 0.35,
        "max_mean_perpendicular_distance_px": 12.0,
        "max_angle_diff_deg": 8.0,
        "max_model_to_line_intersection_delta_px": 24.0,
    },
    {
        "profile_id": "tight_overlap35_dist12_angle8_model20",
        "min_overlap_fraction": 0.35,
        "max_mean_perpendicular_distance_px": 12.0,
        "max_angle_diff_deg": 8.0,
        "max_model_to_line_intersection_delta_px": 20.0,
    },
)
NEURAL_KEYPOINT_METRICS_GLOB = "court_keypoint_metrics.json"
MOBILENET_V3_KEYPOINT_CHECKPOINT_GLOBS: tuple[str, ...] = (
    "runs/**/mobilenet_v3_court_keypoint_regressor*.pt",
    "runs/**/court_mobilenet_v3*_regressor*.pt",
)
METRIC_PLANE_TOP_RESIDUAL_REFIT_DROP_COUNT = 3
METRIC_PLANE_TOP_RESIDUAL_REFIT_MAX_PROGRESSION_DROP_COUNT = 5
METRIC_PLANE_TOP_RESIDUAL_LINE_OVERRIDE_DROP_COUNT = 4
FLOOR_LINE_KEYPOINT_PAIRS: dict[str, tuple[str, str]] = {
    "near_baseline": ("near_left_corner", "near_right_corner"),
    "far_baseline": ("far_left_corner", "far_right_corner"),
    "left_sideline": ("near_left_corner", "far_left_corner"),
    "right_sideline": ("near_right_corner", "far_right_corner"),
    "near_nvz": ("near_nvz_left", "near_nvz_right"),
    "far_nvz": ("far_nvz_left", "far_nvz_right"),
    "near_centerline": ("near_baseline_center", "near_nvz_center"),
    "far_centerline": ("far_nvz_center", "far_baseline_center"),
}
FLOOR_KEYPOINT_LINE_INTERSECTIONS: dict[str, tuple[str, str]] = {
    "near_left_corner": ("near_baseline", "left_sideline"),
    "near_right_corner": ("near_baseline", "right_sideline"),
    "far_left_corner": ("far_baseline", "left_sideline"),
    "far_right_corner": ("far_baseline", "right_sideline"),
    "near_baseline_center": ("near_baseline", "near_centerline"),
    "far_baseline_center": ("far_baseline", "far_centerline"),
    "near_nvz_left": ("near_nvz", "left_sideline"),
    "near_nvz_right": ("near_nvz", "right_sideline"),
    "near_nvz_center": ("near_nvz", "near_centerline"),
    "far_nvz_left": ("far_nvz", "left_sideline"),
    "far_nvz_right": ("far_nvz", "right_sideline"),
    "far_nvz_center": ("far_nvz", "far_centerline"),
}
LINE_INTERSECTION_OVERRIDE_CENTER_KEYPOINTS = frozenset(
    {"near_baseline_center", "far_baseline_center", "near_nvz_center", "far_nvz_center"}
)


def hsv_paint_mask(
    image_bgr: Any,
    *,
    ranges: Sequence[HSVPaintRange] = DEFAULT_PICKLEBALL_PAINT_RANGES,
    morphology_kernel_px: int = 3,
) -> Any:
    """Return a binary mask for explicitly configured colored court paint."""

    cv2 = _cv2()
    np = _np()
    if not ranges:
        raise ValueError("at least one HSV paint range is required")
    image = _as_uint8_bgr(image_bgr)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for paint_range in ranges:
        lower = np.asarray(paint_range.lower, dtype=np.uint8)
        upper = np.asarray(paint_range.upper, dtype=np.uint8)
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))
    if morphology_kernel_px > 1:
        kernel_size = int(morphology_kernel_px)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def apply_near_side_net_crop(
    mask_or_image: Any,
    *,
    net_evidence: Mapping[str, Any] | None,
    preserve_margin_px: int = 8,
) -> tuple[Any, dict[str, Any]]:
    """Zero everything above the net top-tape estimate, retaining the near side."""

    np = _np()
    if mask_or_image is None or not hasattr(mask_or_image, "shape") or len(mask_or_image.shape) < 2:
        raise ValueError("mask_or_image must be an image array")
    output = np.array(mask_or_image, copy=True)
    height = int(output.shape[0])
    if height <= 0:
        raise ValueError("mask_or_image must have positive height")
    cutoff = _net_cutoff_y(net_evidence)
    if cutoff is None:
        return output, {"applied": False, "reason": "missing_net_y_evidence"}
    y_min = max(0, min(height, int(math.floor(float(cutoff) - float(preserve_margin_px)))))
    if output.ndim == 2:
        output[:y_min, :] = 0
    else:
        output[:y_min, :, :] = 0
    return output, {
        "applied": True,
        "cutoff_y": round(float(cutoff), 3),
        "y_min_retained": int(y_min),
        "preserve_margin_px": int(preserve_margin_px),
        "assumption": "near_side_is_below_net_in_image_coordinates",
    }


def clustered_hough_boundaries(
    mask: Any,
    *,
    config: LineClusterConfig | None = None,
) -> BoundaryClusters:
    """Detect and cluster Hough line segments into likely court-boundary lines."""

    cfg = config or LineClusterConfig()
    mask_u8 = _as_uint8_mask(mask)
    segments = _hough_segments_from_mask(mask_u8, config=cfg)
    groups: list[list[tuple[tuple[float, float], tuple[float, float]]]] = []
    for segment in sorted(segments, key=_segment_length, reverse=True):
        line = _line_from_segment(segment)
        angle = _segment_angle_deg(segment)
        assigned = False
        for group in groups:
            reference = _line_from_segment(_longest_segment(group))
            reference_angle = _segment_angle_deg(_longest_segment(group))
            if (
                _angle_diff_mod_180(angle, reference_angle) <= cfg.angle_tolerance_deg
                and _line_segment_distance(reference, segment) <= cfg.line_distance_tolerance_px
            ):
                group.append(segment)
                assigned = True
                break
        if not assigned:
            groups.append([segment])

    clusters = [_boundary_cluster(group) for group in groups if group]
    selected = _select_boundary_clusters(clusters, max_clusters=cfg.max_clusters)
    return BoundaryClusters(
        clusters=selected,
        raw_segment_count=len(segments),
        mask_support_ratio=_support_ratio(mask_u8),
    )


def detect_hsv_paint_hough_segments(
    image_bgr: Any,
    *,
    ranges: Sequence[HSVPaintRange] = DEFAULT_PICKLEBALL_PAINT_RANGES,
    net_evidence: Mapping[str, Any] | None = None,
    use_near_side_crop: bool = False,
    technology_id: str = "opencv_hsv_paint_hough",
    config: LineClusterConfig | None = None,
) -> dict[str, Any]:
    """Return JSON-safe HSV+Hough line evidence for benchmark adapters."""

    mask = hsv_paint_mask(image_bgr, ranges=ranges)
    crop_evidence: dict[str, Any] = {"applied": False}
    if use_near_side_crop:
        mask, crop_evidence = apply_near_side_net_crop(mask, net_evidence=net_evidence)
    boundaries = clustered_hough_boundaries(mask, config=config)
    payload = boundaries.to_payload(source=technology_id)
    segments = payload["segments"]
    return {
        "technology_id": technology_id,
        "available": True,
        "candidate_count": len(segments),
        "segments": segments,
        "paint_mask": {
            "mode": "strict_hsv_ranges",
            "range_names": [paint_range.name for paint_range in ranges],
            "support_ratio": round(float(_support_ratio(mask)), 6),
        },
        "net_crop": crop_evidence,
        "raw_segment_count": payload["raw_segment_count"],
        "boundary_cluster_count": payload["boundary_cluster_count"],
    }


def load_labelme_court_keypoints(path: str | Path) -> dict[str, Any]:
    """Load LabelMe point annotations into the canonical court-keypoint taxonomy."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    shapes = payload.get("shapes")
    if not isinstance(shapes, list):
        raise ValueError(f"{path}: LabelMe payload must contain a shapes list")
    keypoints: dict[str, list[float]] = {}
    for shape in shapes:
        if not isinstance(shape, Mapping):
            continue
        label = str(shape.get("label") or "")
        if label not in PICKLEBALL_COURT_KEYPOINT_NAMES:
            continue
        shape_type = str(shape.get("shape_type") or "point")
        points = shape.get("points")
        if shape_type != "point":
            raise ValueError(f"{path}: court keypoint {label} must use LabelMe point shape")
        if not isinstance(points, Sequence) or not points:
            raise ValueError(f"{path}: court keypoint {label} is missing point coordinates")
        xy = _point2(points[0], f"shapes[{label}].points[0]")
        keypoints[label] = [xy[0], xy[1]]
    image_width = _positive_int(payload.get("imageWidth"), "imageWidth")
    image_height = _positive_int(payload.get("imageHeight"), "imageHeight")
    missing = [name for name in PICKLEBALL_COURT_KEYPOINT_NAMES if name not in keypoints]
    return {
        "artifact_type": "labelme_court_keypoint_points",
        "image_path": str(payload.get("imagePath") or ""),
        "image_size": [image_width, image_height],
        "keypoints": keypoints,
        "visible_keypoint_count": len(keypoints),
        "missing_keypoints": missing,
    }


def refine_image_to_world_homography_lm(
    image_points_px: Sequence[Sequence[float]],
    world_points_ft: Sequence[Sequence[float]],
    *,
    seed_point_indexes: Sequence[int] | None = None,
    target_mean_residual_ft: float = 0.2,
) -> dict[str, Any]:
    """Refine an image->court-world homography with LM world-space residuals."""

    from scipy.optimize import least_squares

    from .court_calibration import homography_from_planar_points, project_planar_points

    image_points = [_point2(point, "image_points_px") for point in image_points_px]
    world_points = [_point2(point, "world_points_ft") for point in world_points_ft]
    if len(image_points) != len(world_points) or len(image_points) < 4:
        raise ValueError("LM homography refinement requires at least 4 paired points")
    if len(image_points) * 2 < 8:
        raise ValueError("LM homography refinement needs at least as many residuals as parameters")

    if seed_point_indexes is None:
        seed_indexes = tuple(range(len(image_points)))
    else:
        seed_indexes = tuple(int(index) for index in seed_point_indexes)
    if len(seed_indexes) < 4:
        raise ValueError("seed_point_indexes must contain at least 4 indexes")
    if any(index < 0 or index >= len(image_points) for index in seed_indexes):
        raise ValueError("seed_point_indexes contains an out-of-range index")

    seed_image = [image_points[index] for index in seed_indexes]
    seed_world = [world_points[index] for index in seed_indexes]
    initial_h = homography_from_planar_points(seed_image, seed_world)

    def pack(homography: Sequence[Sequence[float]]) -> list[float]:
        return [
            float(homography[0][0]),
            float(homography[0][1]),
            float(homography[0][2]),
            float(homography[1][0]),
            float(homography[1][1]),
            float(homography[1][2]),
            float(homography[2][0]),
            float(homography[2][1]),
        ]

    def unpack(params: Sequence[float]) -> list[list[float]]:
        return [
            [float(params[0]), float(params[1]), float(params[2])],
            [float(params[3]), float(params[4]), float(params[5])],
            [float(params[6]), float(params[7]), 1.0],
        ]

    def residuals(params: Sequence[float]) -> list[float]:
        projected = project_planar_points(unpack(params), image_points)
        values: list[float] = []
        for predicted, expected in zip(projected, world_points, strict=True):
            values.append(float(predicted[0]) - float(expected[0]))
            values.append(float(predicted[1]) - float(expected[1]))
        return values

    initial_params = pack(initial_h)
    initial_residuals = residuals(initial_params)
    result = least_squares(residuals, initial_params, method="lm")
    optimized_h = unpack(result.x)
    optimized_residuals = residuals(result.x)
    initial_distances = _paired_residual_distances(initial_residuals)
    optimized_distances = _paired_residual_distances(optimized_residuals)
    return {
        "method": "scipy_least_squares_lm",
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "target_mean_residual_ft": float(target_mean_residual_ft),
        "seed_point_count": len(seed_indexes),
        "point_count": len(image_points),
        "initial_h_image_to_world": _round_matrix(initial_h),
        "optimized_h_image_to_world": _round_matrix(optimized_h),
        "initial_mean_residual_ft": round(_mean(initial_distances), 6),
        "optimized_mean_residual_ft": round(_mean(optimized_distances), 6),
        "initial_p95_residual_ft": round(_percentile(initial_distances, 95), 6),
        "optimized_p95_residual_ft": round(_percentile(optimized_distances, 95), 6),
        "optimized_passes_target": _mean(optimized_distances) <= float(target_mean_residual_ft),
    }


def fit_joint_distorted_camera_lm(
    object_points_m: Sequence[Sequence[float]],
    image_points_px: Sequence[Sequence[float]],
    *,
    image_size: tuple[int | float, int | float],
) -> dict[str, Any]:
    """Jointly fit focal length, pose, and radial distortion from court points."""

    cv2 = _cv2()
    np = _np()
    from scipy.optimize import least_squares

    obj = _object_points3(object_points_m)
    img = _image_points2(image_points_px)
    if obj.shape[0] != img.shape[0] or obj.shape[0] < 5:
        raise ValueError("joint distorted camera fit requires at least 5 paired points")
    width, height = _image_size2(image_size)
    cx, cy = width / 2.0, height / 2.0
    initial = _initial_camera_parameter_vector(cv2, np, obj, img, width=width, height=height)

    def residuals(params: Sequence[float]) -> Any:
        projected = _project_with_camera_params(cv2, np, obj, params, cx=cx, cy=cy)
        return (projected - img).reshape(-1)

    initial_residuals = residuals(initial)
    result = least_squares(residuals, initial, method="lm", max_nfev=1200)
    optimized_residuals = residuals(result.x)
    return _camera_fit_payload(
        result.x,
        cx=cx,
        cy=cy,
        method="joint_focal_pose_radial_lm",
        success=bool(result.success),
        status=int(result.status),
        message=str(result.message),
        initial_residuals=initial_residuals,
        optimized_residuals=optimized_residuals,
        point_count=int(obj.shape[0]),
        line_observation_count=0,
    )


def fit_joint_camera_point_line_lm(
    object_points_m: Sequence[Sequence[float]],
    image_points_px: Sequence[Sequence[float]],
    *,
    image_size: tuple[int | float, int | float],
    line_observations: Sequence[Mapping[str, Any]],
    line_weight: float = 1.0,
) -> dict[str, Any]:
    """Fit camera parameters using point reprojection and line-distance residuals."""

    if line_weight <= 0.0:
        raise ValueError("line_weight must be positive")
    cv2 = _cv2()
    np = _np()
    from scipy.optimize import least_squares

    obj = _object_points3(object_points_m)
    img = _image_points2(image_points_px)
    if obj.shape[0] != img.shape[0] or obj.shape[0] < 5:
        raise ValueError("joint point-line fit requires at least 5 paired points")
    parsed_lines = [_line_observation_arrays(observation) for observation in line_observations]
    if not parsed_lines:
        raise ValueError("joint point-line fit requires at least one line observation")
    line_pixel_sample_count = sum(int(image_samples.shape[0]) for _world_line, image_samples in parsed_lines)
    line_pixels_per_observation = line_pixel_sample_count / len(parsed_lines)
    width, height = _image_size2(image_size)
    cx, cy = width / 2.0, height / 2.0
    point_only = fit_joint_distorted_camera_lm(object_points_m, image_points_px, image_size=(width, height))
    initial = _camera_params_from_fit(point_only)

    def residuals(params: Sequence[float]) -> Any:
        projected = _project_with_camera_params(cv2, np, obj, params, cx=cx, cy=cy)
        point_residuals = (projected - img).reshape(-1)
        line_residuals: list[float] = []
        for world_line, image_segment in parsed_lines:
            projected_line = _project_with_camera_params(cv2, np, world_line, params, cx=cx, cy=cy)
            line_residuals.extend(_image_segment_to_projected_line_residuals(projected_line, image_segment))
        return np.concatenate(
            [
                point_residuals,
                np.asarray(line_residuals, dtype=np.float64) * float(line_weight),
            ]
        )

    initial_residuals = residuals(initial)
    result = least_squares(residuals, initial, method="lm", max_nfev=1600)
    optimized_residuals = residuals(result.x)
    payload = _camera_fit_payload(
        result.x,
        cx=cx,
        cy=cy,
        method="joint_point_line_focal_pose_radial_lm",
        success=bool(result.success),
        status=int(result.status),
        message=str(result.message),
        initial_residuals=initial_residuals,
        optimized_residuals=optimized_residuals,
        point_count=int(obj.shape[0]),
        line_observation_count=len(parsed_lines),
    )
    payload.update(
        {
            "line_residual_mode": "sampled_line_pixels_to_projected_model_line",
            "line_pixel_sample_count": int(line_pixel_sample_count),
            "line_pixel_samples_per_observation": round(float(line_pixels_per_observation), 6),
        }
    )
    return payload


def fit_metric_plane_camera_lm(
    object_points_m: Sequence[Sequence[float]],
    image_points_px: Sequence[Sequence[float]],
    *,
    image_size: tuple[int | float, int | float],
    robust_f_scale_m: float = 0.02,
) -> dict[str, Any]:
    """Fit camera parameters by minimizing court-plane backprojection residuals.

    The pixel-reprojection fit is a good camera-model objective, but the CAL gate
    we care about is inverse court-space residual. This solver starts from the
    pixel fit, then runs LM on z=0 backprojection errors in meters with a soft-L1
    residual transform so noisy reviewed keypoints do not dominate the solve.
    """

    if robust_f_scale_m <= 0.0:
        raise ValueError("robust_f_scale_m must be positive")
    cv2 = _cv2()
    np = _np()
    from scipy.optimize import least_squares

    obj = _object_points3(object_points_m)
    img = _image_points2(image_points_px)
    if obj.shape[0] != img.shape[0] or obj.shape[0] < 5:
        raise ValueError("metric-plane camera fit requires at least 5 paired points")
    width, height = _image_size2(image_size)
    cx, cy = width / 2.0, height / 2.0
    point_fit = fit_joint_distorted_camera_lm(object_points_m, image_points_px, image_size=(width, height))
    initial = _camera_params_from_fit(point_fit)
    target_xy = obj[:, :2]

    def world_plane_residuals(params: Sequence[float]) -> Any:
        predicted = _image_points_to_world_plane_with_camera_params(cv2, np, img, params, cx=cx, cy=cy)
        residual = (predicted[:, :2] - target_xy).reshape(-1)
        if not np.all(np.isfinite(residual)):
            return np.full(obj.shape[0] * 2, 1e3, dtype=np.float64)
        return residual

    def objective_residuals(params: Sequence[float]) -> Any:
        return _soft_l1_residual_transform(world_plane_residuals(params), scale=float(robust_f_scale_m))

    initial_world_residuals = world_plane_residuals(initial)
    result = least_squares(objective_residuals, initial, method="lm", max_nfev=2200)
    optimized_world_residuals = world_plane_residuals(result.x)
    initial_projected = _project_with_camera_params(cv2, np, obj, initial, cx=cx, cy=cy)
    optimized_projected = _project_with_camera_params(cv2, np, obj, result.x, cx=cx, cy=cy)
    payload = _camera_fit_payload(
        result.x,
        cx=cx,
        cy=cy,
        method="metric_plane_focal_pose_radial_soft_l1_lm",
        success=bool(result.success),
        status=int(result.status),
        message=str(result.message),
        initial_residuals=(initial_projected - img).reshape(-1),
        optimized_residuals=(optimized_projected - img).reshape(-1),
        point_count=int(obj.shape[0]),
        line_observation_count=0,
    )
    initial_world_distances = _paired_residual_distances(initial_world_residuals)
    optimized_world_distances = _paired_residual_distances(optimized_world_residuals)
    payload.update(
        {
            "objective": "world_plane_backprojection_m",
            "robust_loss": "soft_l1_residual_transform",
            "robust_f_scale_m": round(float(robust_f_scale_m), 6),
            "initial_world_plane_rmse_m": round(_rmse(initial_world_distances), 6),
            "optimized_world_plane_rmse_m": round(_rmse(optimized_world_distances), 6),
            "optimized_world_plane_median_m": round(_median(optimized_world_distances), 6),
            "optimized_world_plane_p95_m": round(_percentile(optimized_world_distances, 95), 6),
        }
    )
    return payload


def fit_full_intrinsics_metric_plane_camera_lm(
    object_points_m: Sequence[Sequence[float]],
    image_points_px: Sequence[Sequence[float]],
    *,
    image_size: tuple[int | float, int | float],
    robust_f_scale_m: float = 0.02,
    pixel_residual_weight_m_per_px: float = 0.0005,
) -> dict[str, Any]:
    """Diagnostic camera fit with fx/fy/cx/cy, pose, and radial distortion free.

    A single planar view cannot identify full intrinsics robustly. This exists
    to measure whether extra intrinsics can explain residuals, not to promote a
    production calibration model.
    """

    if robust_f_scale_m <= 0.0:
        raise ValueError("robust_f_scale_m must be positive")
    if pixel_residual_weight_m_per_px < 0.0:
        raise ValueError("pixel_residual_weight_m_per_px must be non-negative")
    cv2 = _cv2()
    np = _np()
    from scipy.optimize import least_squares

    obj = _object_points3(object_points_m)
    img = _image_points2(image_points_px)
    if obj.shape[0] != img.shape[0] or obj.shape[0] < 6:
        raise ValueError("full-intrinsics metric-plane camera fit requires at least 6 paired points")
    width, height = _image_size2(image_size)
    seed_fit = fit_metric_plane_camera_lm(
        object_points_m,
        image_points_px,
        image_size=(width, height),
        robust_f_scale_m=robust_f_scale_m,
    )
    intrinsics = seed_fit["intrinsics"]
    distortion = seed_fit["distortion"]
    extrinsics = seed_fit["extrinsics"]
    initial = np.asarray(
        [
            math.log(max(1.0, float(intrinsics["fx"]))),
            math.log(max(1.0, float(intrinsics.get("fy", intrinsics["fx"])))),
            float(intrinsics["cx"]),
            float(intrinsics["cy"]),
            float(extrinsics["rvec"][0]),
            float(extrinsics["rvec"][1]),
            float(extrinsics["rvec"][2]),
            float(extrinsics["tvec"][0]),
            float(extrinsics["tvec"][1]),
            float(extrinsics["tvec"][2]),
            float(distortion.get("k1", 0.0)),
            float(distortion.get("k2", 0.0)),
        ],
        dtype=np.float64,
    )
    target_xy = obj[:, :2]

    def world_plane_residuals(params: Sequence[float]) -> Any:
        predicted = _image_points_to_world_plane_with_full_camera_params(cv2, np, img, params)
        residual = (predicted[:, :2] - target_xy).reshape(-1)
        if not np.all(np.isfinite(residual)):
            return np.full(obj.shape[0] * 2, 1e3, dtype=np.float64)
        return residual

    def objective_residuals(params: Sequence[float]) -> Any:
        world_residual = _soft_l1_residual_transform(world_plane_residuals(params), scale=float(robust_f_scale_m))
        if pixel_residual_weight_m_per_px <= 0.0:
            return world_residual
        projected = _project_with_full_camera_params(cv2, np, obj, params)
        pixel_residual = (projected - img).reshape(-1) * float(pixel_residual_weight_m_per_px)
        if not np.all(np.isfinite(pixel_residual)):
            pixel_residual = np.full(obj.shape[0] * 2, 1e3, dtype=np.float64)
        return np.concatenate([world_residual, pixel_residual])

    initial_world_residuals = world_plane_residuals(initial)
    result = least_squares(objective_residuals, initial, method="lm", max_nfev=3200)
    optimized_world_residuals = world_plane_residuals(result.x)
    initial_projected = _project_with_full_camera_params(cv2, np, obj, initial)
    optimized_projected = _project_with_full_camera_params(cv2, np, obj, result.x)
    payload = _full_camera_fit_payload(
        result.x,
        method="full_intrinsics_metric_plane_pose_radial_soft_l1_lm",
        success=bool(result.success),
        status=int(result.status),
        message=str(result.message),
        initial_residuals=(initial_projected - img).reshape(-1),
        optimized_residuals=(optimized_projected - img).reshape(-1),
        point_count=int(obj.shape[0]),
    )
    initial_world_distances = _paired_residual_distances(initial_world_residuals)
    optimized_world_distances = _paired_residual_distances(optimized_world_residuals)
    payload.update(
        {
            "diagnostic_only": True,
            "promotes_calibration": False,
            "objective": "world_plane_backprojection_m_plus_weak_pixel_reprojection",
            "robust_loss": "soft_l1_residual_transform",
            "robust_f_scale_m": round(float(robust_f_scale_m), 6),
            "pixel_residual_weight_m_per_px": round(float(pixel_residual_weight_m_per_px), 9),
            "initial_world_plane_rmse_m": round(_rmse(initial_world_distances), 6),
            "optimized_world_plane_rmse_m": round(_rmse(optimized_world_distances), 6),
            "optimized_world_plane_median_m": round(_median(optimized_world_distances), 6),
            "optimized_world_plane_p95_m": round(_percentile(optimized_world_distances, 95), 6),
            "identifiability_notes": [
                "single planar views cannot uniquely identify full intrinsics",
                "diagnostic-only: do not use as CAL pass criterion without independent validation",
            ],
        }
    )
    return payload


def project_world_points_with_distortion_fit(
    object_points_m: Sequence[Sequence[float]],
    fit: Mapping[str, Any],
) -> list[list[float]]:
    """Project 3D world points using a fit payload returned by the joint solvers."""

    cv2 = _cv2()
    np = _np()
    obj = _object_points3(object_points_m)
    projected = _project_with_fit_payload(cv2, np, obj, fit)
    return projected.tolist()


def image_points_to_world_plane_with_distortion_fit(
    image_points_px: Sequence[Sequence[float]],
    fit: Mapping[str, Any],
) -> list[list[float]]:
    """Back-project image points through a camera fit onto the court plane z=0."""

    cv2 = _cv2()
    np = _np()
    image_points = _image_points2(image_points_px).reshape(-1, 1, 2)
    intrinsics = fit.get("intrinsics") if isinstance(fit.get("intrinsics"), Mapping) else {}
    distortion = fit.get("distortion") if isinstance(fit.get("distortion"), Mapping) else {}
    extrinsics = fit.get("extrinsics") if isinstance(fit.get("extrinsics"), Mapping) else {}
    fx = _finite_float(intrinsics.get("fx"), "fit.intrinsics.fx")
    fy = _finite_float(intrinsics.get("fy", fx), "fit.intrinsics.fy")
    cx = _finite_float(intrinsics.get("cx"), "fit.intrinsics.cx")
    cy = _finite_float(intrinsics.get("cy"), "fit.intrinsics.cy")
    rvec = extrinsics.get("rvec")
    tvec = extrinsics.get("tvec")
    if not isinstance(rvec, Sequence) or len(rvec) != 3 or not isinstance(tvec, Sequence) or len(tvec) != 3:
        raise ValueError("fit payload is missing extrinsics.rvec/tvec")
    k = np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    dist = np.asarray(
        [
            _finite_float(distortion.get("k1", 0.0), "fit.distortion.k1"),
            _finite_float(distortion.get("k2", 0.0), "fit.distortion.k2"),
            0.0,
            0.0,
        ],
        dtype=np.float64,
    )
    normalized = cv2.undistortPoints(image_points, k, dist).reshape(-1, 2)
    rotation, _ = cv2.Rodrigues(np.asarray([float(value) for value in rvec], dtype=np.float64).reshape(3, 1))
    translation = np.asarray([float(value) for value in tvec], dtype=np.float64).reshape(3)
    camera_center_world = -rotation.T @ translation
    points: list[list[float]] = []
    for x_norm, y_norm in normalized:
        ray_camera = np.asarray([float(x_norm), float(y_norm), 1.0], dtype=np.float64)
        ray_world = rotation.T @ ray_camera
        if abs(float(ray_world[2])) <= 1e-9:
            raise ValueError("camera ray is parallel to court plane")
        scale = -float(camera_center_world[2]) / float(ray_world[2])
        world = camera_center_world + ray_world * scale
        points.append([float(world[0]), float(world[1]), 0.0])
    return points


def shadow_removal_preprocess(image_bgr: Any) -> tuple[Any, dict[str, Any]]:
    """Compensate broad shadows before classical line detection.

    This is a deterministic local-illumination normalizer. It is deliberately
    exposed as a preprocessing fallback until a real pretrained shadow-removal
    model is added and verified on court footage.
    """

    cv2 = _cv2()
    np = _np()
    image = _as_uint8_bgr(image_bgr)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0].astype(np.float32)
    illumination = cv2.GaussianBlur(l_channel, (0, 0), sigmaX=21.0, sigmaY=21.0)
    reference = float(np.percentile(illumination, 82))
    scale = reference / np.maximum(illumination, 8.0)
    corrected_l = np.clip(l_channel * scale * 1.28, 0, 255).astype(np.uint8)
    corrected_lab = lab.copy()
    corrected_lab[:, :, 0] = corrected_l
    corrected = cv2.cvtColor(corrected_lab, cv2.COLOR_LAB2BGR)
    return corrected, {
        "available": True,
        "method": "lab_luminance_local_illumination_compensation",
        "mode": "lab_luminance_shadow_compensation",
        "pretrained_model_used": False,
        "reference_l_percentile": round(reference, 3),
    }


def pretrained_shadow_removal_preprocess(
    image_bgr: Any,
    *,
    model_path: str | Path | None = None,
    env_var: str = "PICKLEBALL_SHADOW_REMOVAL_TORCHSCRIPT",
) -> tuple[Any, dict[str, Any]]:
    """Run an explicitly configured pretrained TorchScript shadow remover.

    This path is intentionally fail-closed. If no real model weights are
    configured, callers receive the original image and evidence explaining that
    no pretrained model was used.
    """

    image = _as_uint8_bgr(image_bgr)
    raw_model_path = "" if model_path is None else str(model_path)
    if not raw_model_path:
        raw_model_path = os.environ.get(env_var, "")
    base_evidence: dict[str, Any] = {
        "available": False,
        "method": "torchscript_image_to_image_shadow_removal",
        "mode": "pretrained_ml_shadow_removal",
        "pretrained_model_used": False,
        "framework": "torchscript",
        "env_var": env_var,
        "candidate_model_families": ["ShadowFormer", "SID", "DHAN"],
        "candidate_model_references": [
            "https://github.com/guolanqing/shadowformer",
            "https://github.com/cvlab-stonybrook/SID",
            "https://github.com/mducducd/Shadow-Removal",
        ],
    }
    if not raw_model_path:
        base_evidence["reason"] = "missing_pretrained_shadow_removal_model"
        return image, base_evidence
    resolved_path = Path(raw_model_path)
    if not resolved_path.is_file():
        base_evidence.update(
            {
                "reason": "pretrained_shadow_removal_model_not_found",
                "model_path": str(resolved_path),
            }
        )
        return image, base_evidence

    try:
        import torch  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - depends on optional runtime
        base_evidence.update(
            {
                "reason": "torch_unavailable_for_pretrained_shadow_removal",
                "error": str(exc),
                "model_path": str(resolved_path),
            }
        )
        return image, base_evidence

    cv2 = _cv2()
    np = _np()
    try:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = torch.jit.load(str(resolved_path), map_location=device)
        model.eval()
        with torch.no_grad():
            output = model(tensor.to(device))
        if isinstance(output, (tuple, list)):
            output = output[0]
        if not hasattr(output, "detach"):
            raise ValueError("TorchScript shadow model output must be a tensor")
        output_tensor = output.detach().float().cpu()
        if output_tensor.ndim == 3:
            output_tensor = output_tensor.unsqueeze(0)
        if output_tensor.ndim != 4 or int(output_tensor.shape[0]) != 1 or int(output_tensor.shape[1]) != 3:
            raise ValueError("TorchScript shadow model output must have shape [1,3,H,W] or [3,H,W]")
        if int(output_tensor.shape[2]) != int(image.shape[0]) or int(output_tensor.shape[3]) != int(image.shape[1]):
            output_tensor = torch.nn.functional.interpolate(
                output_tensor,
                size=(int(image.shape[0]), int(image.shape[1])),
                mode="bilinear",
                align_corners=False,
            )
        output_rgb = (
            output_tensor.squeeze(0).permute(1, 2, 0).clamp(0.0, 1.0).numpy() * 255.0
        ).round().astype(np.uint8)
        output_bgr = cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR)
    except Exception as exc:
        base_evidence.update(
            {
                "reason": "pretrained_shadow_removal_inference_failed",
                "error": str(exc),
                "model_path": str(resolved_path),
            }
        )
        return image, base_evidence

    base_evidence.update(
        {
            "available": True,
            "pretrained_model_used": True,
            "model_path": str(resolved_path),
            "model_sha256": _sha256_file(resolved_path),
            "device": str(device),
            "input_size": [int(image.shape[1]), int(image.shape[0])],
            "output_size": [int(output_bgr.shape[1]), int(output_bgr.shape[0])],
        }
    )
    return output_bgr, base_evidence


def build_lm_homography_reviewed_label_report(
    *,
    eval_root: str | Path,
    out_path: str | Path | None = None,
    target_mean_residual_ft: float = 0.2,
) -> dict[str, Any]:
    """Evaluate corner-seed vs LM-refined homography on reviewed full labels."""

    from .court_calibration_metric15 import (
        aggregate_reviewed_keypoints_native_px,
        load_reviewed_court_keypoints_15pt,
    )
    from .court_keypoint_net import PICKLEBALL_KEYPOINT_BY_NAME
    from .court_templates import FT_TO_M

    root = Path(eval_root)
    if not root.is_dir():
        raise ValueError(f"eval_root does not exist or is not a directory: {root}")
    repo_root = _calibration_repo_root(root)
    neural_keypoint_evidence = _neural_keypoint_checkpoint_evidence(repo_root=repo_root)
    mobilenet_v3_keypoint_evidence = _mobilenet_v3_keypoint_checkpoint_evidence(
        repo_root=repo_root,
        eval_root=root,
    )
    best_neural_candidate = neural_keypoint_evidence.get("best_real_label_candidate")
    best_neural_metric = (
        best_neural_candidate.get("candidate_metric_value_px")
        if isinstance(best_neural_candidate, Mapping)
        else None
    )
    best_mobilenet_candidate = mobilenet_v3_keypoint_evidence.get("best_candidate")
    best_mobilenet_metric = (
        best_mobilenet_candidate.get("median_error_px")
        if isinstance(best_mobilenet_candidate, Mapping)
        else None
    )

    results: list[dict[str, Any]] = []
    partial_excluded: list[dict[str, Any]] = []
    sample_count = 0
    for clip_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        sample_count += 1
        labels = clip_dir / "labels"
        full_label = labels / "court_keypoints.json"
        partial_label = labels / "court_keypoints_partial.json"
        if not full_label.exists():
            if partial_label.exists():
                partial_excluded.append(
                    {
                        "clip": clip_dir.name,
                        "label_path": str(partial_label),
                        "reason": "partial_visible_labels_not_full_15pt_metric_homography",
                    }
                )
            continue

        reviewed = load_reviewed_court_keypoints_15pt(full_label)
        aggregated, _stdev, native_size = aggregate_reviewed_keypoints_native_px(reviewed)
        image_points = [list(aggregated[name]) for name in FLOOR_HOMOGRAPHY_KEYPOINT_NAMES]
        object_points_m = [
            list(PICKLEBALL_KEYPOINT_BY_NAME[name].world_xyz_m)
            for name in FLOOR_HOMOGRAPHY_KEYPOINT_NAMES
        ]
        all15_image_points = [list(aggregated[name]) for name in PICKLEBALL_COURT_KEYPOINT_NAMES]
        all15_object_points_m = [
            list(PICKLEBALL_KEYPOINT_BY_NAME[name].world_xyz_m)
            for name in PICKLEBALL_COURT_KEYPOINT_NAMES
        ]
        world_points_ft = [
            [
                float(PICKLEBALL_KEYPOINT_BY_NAME[name].world_xyz_m[0]) / FT_TO_M,
                float(PICKLEBALL_KEYPOINT_BY_NAME[name].world_xyz_m[1]) / FT_TO_M,
            ]
            for name in FLOOR_HOMOGRAPHY_KEYPOINT_NAMES
        ]
        seed_indexes = tuple(FLOOR_HOMOGRAPHY_KEYPOINT_NAMES.index(name) for name in CORNER_SEED_KEYPOINT_NAMES)
        lm = refine_image_to_world_homography_lm(
            image_points,
            world_points_ft,
            seed_point_indexes=seed_indexes,
            target_mean_residual_ft=target_mean_residual_ft,
        )
        distorted_camera = fit_joint_distorted_camera_lm(
            object_points_m,
            image_points,
            image_size=(native_size[0], native_size[1]),
        )
        distorted_residual_ft = _world_plane_residual_summary_ft(
            object_points_m,
            image_points,
            distorted_camera,
            meters_to_feet=1.0 / FT_TO_M,
        )
        all15_camera = fit_joint_distorted_camera_lm(
            all15_object_points_m,
            all15_image_points,
            image_size=(native_size[0], native_size[1]),
        )
        all15_floor_residual_ft = _world_plane_residual_summary_ft(
            object_points_m,
            image_points,
            all15_camera,
            meters_to_feet=1.0 / FT_TO_M,
        )
        metric_plane_camera = fit_metric_plane_camera_lm(
            object_points_m,
            image_points,
            image_size=(native_size[0], native_size[1]),
        )
        full_intrinsics_metric_plane_camera = fit_full_intrinsics_metric_plane_camera_lm(
            object_points_m,
            image_points,
            image_size=(native_size[0], native_size[1]),
        )
        metric_plane_residual_ft = _world_plane_residual_summary_ft(
            object_points_m,
            image_points,
            metric_plane_camera,
            meters_to_feet=1.0 / FT_TO_M,
        )
        full_intrinsics_metric_plane_residual_ft = _world_plane_residual_summary_ft(
            object_points_m,
            image_points,
            full_intrinsics_metric_plane_camera,
            meters_to_feet=1.0 / FT_TO_M,
        )
        metric_plane_residual_details_ft = _world_plane_residual_details_ft(
            FLOOR_HOMOGRAPHY_KEYPOINT_NAMES,
            object_points_m,
            image_points,
            metric_plane_camera,
            meters_to_feet=1.0 / FT_TO_M,
        )
        review_line_observations = _review_line_observations_for_reviewed_clip(
            full_label,
            aggregated=aggregated,
            image_size=(native_size[0], native_size[1]),
            keypoint_by_name=PICKLEBALL_KEYPOINT_BY_NAME,
        )
        model_projected_line_observations = _model_projected_line_observations_for_reviewed_clip(
            full_label,
            keypoint_names=FLOOR_HOMOGRAPHY_KEYPOINT_NAMES,
            object_points_m=object_points_m,
            model_fit=full_intrinsics_metric_plane_camera,
            image_size=(native_size[0], native_size[1]),
            keypoint_by_name=PICKLEBALL_KEYPOINT_BY_NAME,
        )
        top_residual_refit_diagnostic = _metric_plane_top_residual_refit_diagnostic(
            FLOOR_HOMOGRAPHY_KEYPOINT_NAMES,
            object_points_m,
            image_points,
            image_size=(native_size[0], native_size[1]),
            residual_details_ft=metric_plane_residual_details_ft,
            line_observations=review_line_observations,
            meters_to_feet=1.0 / FT_TO_M,
            drop_count=METRIC_PLANE_TOP_RESIDUAL_REFIT_DROP_COUNT,
        )
        top_residual_refit_progression = _metric_plane_top_residual_refit_progression(
            FLOOR_HOMOGRAPHY_KEYPOINT_NAMES,
            object_points_m,
            image_points,
            image_size=(native_size[0], native_size[1]),
            residual_details_ft=metric_plane_residual_details_ft,
            line_observations=review_line_observations,
            meters_to_feet=1.0 / FT_TO_M,
            max_drop_count=METRIC_PLANE_TOP_RESIDUAL_REFIT_MAX_PROGRESSION_DROP_COUNT,
        )
        metric_plane_outlier_candidates = _metric_plane_outlier_review_candidates(
            FLOOR_HOMOGRAPHY_KEYPOINT_NAMES,
            object_points_m,
            image_points,
            metric_plane_camera,
            metric_plane_residual_details_ft,
            line_observations=review_line_observations,
        )
        line_intersection_override_oracle = _metric_plane_line_intersection_override_oracle(
            FLOOR_HOMOGRAPHY_KEYPOINT_NAMES,
            object_points_m,
            image_points,
            image_size=(native_size[0], native_size[1]),
            candidates=metric_plane_outlier_candidates,
            baseline_mean_residual_ft=metric_plane_residual_ft["mean_residual_ft"],
            meters_to_feet=1.0 / FT_TO_M,
        )
        top_residual_line_intersection_override_oracle = _metric_plane_top_residual_line_intersection_override_oracle(
            FLOOR_HOMOGRAPHY_KEYPOINT_NAMES,
            object_points_m,
            image_points,
            image_size=(native_size[0], native_size[1]),
            progression=top_residual_refit_progression,
            drop_count=METRIC_PLANE_TOP_RESIDUAL_LINE_OVERRIDE_DROP_COUNT,
            baseline_mean_residual_ft=metric_plane_residual_ft["mean_residual_ft"],
            meters_to_feet=1.0 / FT_TO_M,
        )
        top_residual_relaxed_line_intersection_override_oracle = (
            _metric_plane_top_residual_line_intersection_override_oracle(
                FLOOR_HOMOGRAPHY_KEYPOINT_NAMES,
                object_points_m,
                image_points,
                image_size=(native_size[0], native_size[1]),
                progression=top_residual_refit_progression,
                drop_count=METRIC_PLANE_TOP_RESIDUAL_LINE_OVERRIDE_DROP_COUNT,
                baseline_mean_residual_ft=metric_plane_residual_ft["mean_residual_ft"],
                meters_to_feet=1.0 / FT_TO_M,
                strict_support_required=False,
                skip_center_keypoints=False,
                source_strategy="top_residual_relaxed_all_available_line_intersections",
            )
        )
        full_intrinsics_top_residual_line_intersection_override_oracle = (
            _metric_plane_top_residual_line_intersection_override_oracle(
                FLOOR_HOMOGRAPHY_KEYPOINT_NAMES,
                object_points_m,
                image_points,
                image_size=(native_size[0], native_size[1]),
                progression=top_residual_refit_progression,
                drop_count=METRIC_PLANE_TOP_RESIDUAL_LINE_OVERRIDE_DROP_COUNT,
                baseline_mean_residual_ft=full_intrinsics_metric_plane_residual_ft["mean_residual_ft"],
                meters_to_feet=1.0 / FT_TO_M,
                source_strategy="top_residual_strict_endpoint_line_intersections_full_intrinsics_fit",
                camera_fit_model="full_intrinsics_metric_plane",
            )
        )
        full_intrinsics_all_strict_line_intersection_override_oracle = (
            _all_strict_line_intersection_override_oracle(
                FLOOR_HOMOGRAPHY_KEYPOINT_NAMES,
                object_points_m,
                image_points,
                image_size=(native_size[0], native_size[1]),
                line_observations=review_line_observations,
                model_fit=full_intrinsics_metric_plane_camera,
                baseline_mean_residual_ft=full_intrinsics_metric_plane_residual_ft["mean_residual_ft"],
                meters_to_feet=1.0 / FT_TO_M,
                camera_fit_model="full_intrinsics_metric_plane",
            )
        )
        full_intrinsics_quality_gated_line_intersection_override_sweep = (
            _quality_gated_line_intersection_override_sweep(
                FLOOR_HOMOGRAPHY_KEYPOINT_NAMES,
                object_points_m,
                image_points,
                image_size=(native_size[0], native_size[1]),
                line_observations=review_line_observations,
                model_fit=full_intrinsics_metric_plane_camera,
                baseline_mean_residual_ft=full_intrinsics_metric_plane_residual_ft["mean_residual_ft"],
                meters_to_feet=1.0 / FT_TO_M,
                camera_fit_model="full_intrinsics_metric_plane",
            )
        )
        full_intrinsics_model_projected_quality_gated_line_intersection_override_sweep = (
            _quality_gated_line_intersection_override_sweep(
                FLOOR_HOMOGRAPHY_KEYPOINT_NAMES,
                object_points_m,
                image_points,
                image_size=(native_size[0], native_size[1]),
                line_observations=model_projected_line_observations,
                model_fit=full_intrinsics_metric_plane_camera,
                baseline_mean_residual_ft=full_intrinsics_metric_plane_residual_ft["mean_residual_ft"],
                meters_to_feet=1.0 / FT_TO_M,
                camera_fit_model="full_intrinsics_metric_plane",
                source_strategy="model_projected_quality_gated_endpoint_line_intersections",
                selection_mode="model_projection_quality_profile_endpoint_intersections_without_residual_ranking",
                line_observation_source="model_projected_line_observations",
                line_reference_source="model_projection",
                uses_reviewed_line_positions_for_matching=False,
            )
        )
        point_line_camera = _point_line_fit_for_reviewed_clip(
            full_label,
            aggregated=aggregated,
            object_points_m=object_points_m,
            image_points=image_points,
            image_size=(native_size[0], native_size[1]),
            keypoint_by_name=PICKLEBALL_KEYPOINT_BY_NAME,
            meters_to_feet=1.0 / FT_TO_M,
        )
        safe_selected_camera = _safe_selected_camera_payload(
            distorted_camera=distorted_camera,
            distorted_residual_ft=distorted_residual_ft,
            metric_plane_camera=metric_plane_camera,
            metric_plane_residual_ft=metric_plane_residual_ft,
            point_line_camera=point_line_camera,
        )
        results.append(
            {
                "clip": clip_dir.name,
                "label_path": str(full_label),
                "label_kind": "full_15pt",
                "image_size": [int(round(native_size[0])), int(round(native_size[1]))],
                "floor_keypoint_count": len(FLOOR_HOMOGRAPHY_KEYPOINT_NAMES),
                "corner_seed_keypoints": list(CORNER_SEED_KEYPOINT_NAMES),
                "corner_seed_mean_residual_ft": lm["initial_mean_residual_ft"],
                "corner_seed_p95_residual_ft": lm["initial_p95_residual_ft"],
                "lm_optimized_mean_residual_ft": lm["optimized_mean_residual_ft"],
                "lm_optimized_p95_residual_ft": lm["optimized_p95_residual_ft"],
                "lm_passes_target": lm["optimized_passes_target"],
                "lm_status": {
                    "success": lm["success"],
                    "status": lm["status"],
                    "method": lm["method"],
                },
                "distorted_camera": {
                    "method": distorted_camera["method"],
                    "success": distorted_camera["success"],
                    "optimized_reprojection_rmse_px": distorted_camera["optimized_reprojection_rmse_px"],
                    "optimized_reprojection_median_px": distorted_camera["optimized_reprojection_median_px"],
                    "optimized_reprojection_p95_px": distorted_camera["optimized_reprojection_p95_px"],
                    "mean_residual_ft": distorted_residual_ft["mean_residual_ft"],
                    "median_residual_ft": distorted_residual_ft["median_residual_ft"],
                    "p95_residual_ft": distorted_residual_ft["p95_residual_ft"],
                    "fx": distorted_camera["intrinsics"]["fx"],
                    "k1": distorted_camera["distortion"]["k1"],
                    "k2": distorted_camera["distortion"]["k2"],
                },
                "all15_camera": {
                    "method": all15_camera["method"],
                    "success": all15_camera["success"],
                    "point_count": all15_camera["point_count"],
                    "floor_keypoint_count": len(FLOOR_HOMOGRAPHY_KEYPOINT_NAMES),
                    "net_keypoint_count": len(NET_TOP_KEYPOINT_NAMES),
                    "optimized_reprojection_rmse_px": all15_camera["optimized_reprojection_rmse_px"],
                    "optimized_reprojection_median_px": all15_camera["optimized_reprojection_median_px"],
                    "optimized_reprojection_p95_px": all15_camera["optimized_reprojection_p95_px"],
                    "floor_mean_residual_ft": all15_floor_residual_ft["mean_residual_ft"],
                    "floor_median_residual_ft": all15_floor_residual_ft["median_residual_ft"],
                    "floor_p95_residual_ft": all15_floor_residual_ft["p95_residual_ft"],
                    "fx": all15_camera["intrinsics"]["fx"],
                    "k1": all15_camera["distortion"]["k1"],
                    "k2": all15_camera["distortion"]["k2"],
                },
                "metric_plane_camera": {
                    "method": metric_plane_camera["method"],
                    "success": metric_plane_camera["success"],
                    "objective": metric_plane_camera["objective"],
                    "robust_loss": metric_plane_camera["robust_loss"],
                    "robust_f_scale_m": metric_plane_camera["robust_f_scale_m"],
                    "optimized_reprojection_rmse_px": metric_plane_camera["optimized_reprojection_rmse_px"],
                    "optimized_world_plane_rmse_m": metric_plane_camera["optimized_world_plane_rmse_m"],
                    "mean_residual_ft": metric_plane_residual_ft["mean_residual_ft"],
                    "median_residual_ft": metric_plane_residual_ft["median_residual_ft"],
                    "p95_residual_ft": metric_plane_residual_ft["p95_residual_ft"],
                    "per_keypoint_residual_ft": metric_plane_residual_details_ft["per_keypoint_residual_ft"],
                    "worst_keypoint": metric_plane_residual_details_ft["worst_keypoint"],
                    "trimmed_mean_residual_ft_drop_worst_1": metric_plane_residual_details_ft[
                        "trimmed_mean_residual_ft_drop_worst_1"
                    ],
                    "trimmed_mean_residual_ft_drop_worst_2": metric_plane_residual_details_ft[
                        "trimmed_mean_residual_ft_drop_worst_2"
                    ],
                    "trimmed_mean_residual_ft_drop_worst_3": metric_plane_residual_details_ft[
                        "trimmed_mean_residual_ft_drop_worst_3"
                    ],
                    "trimmed_residual_diagnostic_only": True,
                    "top_residual_refit_diagnostic": top_residual_refit_diagnostic,
                    "top_residual_refit_progression": top_residual_refit_progression,
                    "outlier_review_candidates": metric_plane_outlier_candidates,
                    "line_intersection_override_oracle": line_intersection_override_oracle,
                    "top_residual_line_intersection_override_oracle": top_residual_line_intersection_override_oracle,
                    "top_residual_relaxed_line_intersection_override_oracle": (
                        top_residual_relaxed_line_intersection_override_oracle
                    ),
                    "fx": metric_plane_camera["intrinsics"]["fx"],
                    "k1": metric_plane_camera["distortion"]["k1"],
                    "k2": metric_plane_camera["distortion"]["k2"],
                },
                "full_intrinsics_metric_plane_camera": {
                    "method": full_intrinsics_metric_plane_camera["method"],
                    "success": full_intrinsics_metric_plane_camera["success"],
                    "diagnostic_only": True,
                    "promotes_calibration": False,
                    "objective": full_intrinsics_metric_plane_camera["objective"],
                    "robust_loss": full_intrinsics_metric_plane_camera["robust_loss"],
                    "robust_f_scale_m": full_intrinsics_metric_plane_camera["robust_f_scale_m"],
                    "pixel_residual_weight_m_per_px": full_intrinsics_metric_plane_camera[
                        "pixel_residual_weight_m_per_px"
                    ],
                    "optimized_reprojection_rmse_px": full_intrinsics_metric_plane_camera[
                        "optimized_reprojection_rmse_px"
                    ],
                    "optimized_world_plane_rmse_m": full_intrinsics_metric_plane_camera[
                        "optimized_world_plane_rmse_m"
                    ],
                    "mean_residual_ft": full_intrinsics_metric_plane_residual_ft["mean_residual_ft"],
                    "median_residual_ft": full_intrinsics_metric_plane_residual_ft["median_residual_ft"],
                    "p95_residual_ft": full_intrinsics_metric_plane_residual_ft["p95_residual_ft"],
                    "fx": full_intrinsics_metric_plane_camera["intrinsics"]["fx"],
                    "fy": full_intrinsics_metric_plane_camera["intrinsics"]["fy"],
                    "cx": full_intrinsics_metric_plane_camera["intrinsics"]["cx"],
                    "cy": full_intrinsics_metric_plane_camera["intrinsics"]["cy"],
                    "k1": full_intrinsics_metric_plane_camera["distortion"]["k1"],
                    "k2": full_intrinsics_metric_plane_camera["distortion"]["k2"],
                    "identifiability_notes": full_intrinsics_metric_plane_camera["identifiability_notes"],
                    "top_residual_line_intersection_override_oracle": (
                        full_intrinsics_top_residual_line_intersection_override_oracle
                    ),
                    "all_strict_endpoint_line_intersection_override_oracle": (
                        full_intrinsics_all_strict_line_intersection_override_oracle
                    ),
                    "quality_gated_line_intersection_override_sweep": (
                        full_intrinsics_quality_gated_line_intersection_override_sweep
                    ),
                    "model_projected_quality_gated_line_intersection_override_sweep": (
                        full_intrinsics_model_projected_quality_gated_line_intersection_override_sweep
                    ),
                },
                "point_line_camera": point_line_camera,
                "safe_selected_camera": safe_selected_camera,
            }
        )

    corner_means = [float(result["corner_seed_mean_residual_ft"]) for result in results]
    lm_means = [float(result["lm_optimized_mean_residual_ft"]) for result in results]
    distorted_rmses = [
        float(result["distorted_camera"]["optimized_reprojection_rmse_px"])
        for result in results
        if result.get("distorted_camera")
    ]
    distorted_world_means = [
        float(result["distorted_camera"]["mean_residual_ft"])
        for result in results
        if result.get("distorted_camera", {}).get("mean_residual_ft") is not None
    ]
    all15_rmses = [
        float(result["all15_camera"]["optimized_reprojection_rmse_px"])
        for result in results
        if result.get("all15_camera")
    ]
    all15_floor_world_means = [
        float(result["all15_camera"]["floor_mean_residual_ft"])
        for result in results
        if result.get("all15_camera", {}).get("floor_mean_residual_ft") is not None
    ]
    metric_plane_rmses = [
        float(result["metric_plane_camera"]["optimized_reprojection_rmse_px"])
        for result in results
        if result.get("metric_plane_camera")
    ]
    metric_plane_world_means = [
        float(result["metric_plane_camera"]["mean_residual_ft"])
        for result in results
        if result.get("metric_plane_camera", {}).get("mean_residual_ft") is not None
    ]
    full_intrinsics_metric_plane_rmses = [
        float(result["full_intrinsics_metric_plane_camera"]["optimized_reprojection_rmse_px"])
        for result in results
        if result.get("full_intrinsics_metric_plane_camera")
    ]
    full_intrinsics_metric_plane_world_means = [
        float(result["full_intrinsics_metric_plane_camera"]["mean_residual_ft"])
        for result in results
        if result.get("full_intrinsics_metric_plane_camera", {}).get("mean_residual_ft") is not None
    ]
    metric_plane_all_keypoint_residuals = [
        float(residual)
        for result in results
        if isinstance(result.get("metric_plane_camera"), Mapping)
        for residual in result["metric_plane_camera"].get("per_keypoint_residual_ft", {}).values()
    ]
    metric_plane_trimmed_worst3_means = [
        float(result["metric_plane_camera"]["trimmed_mean_residual_ft_drop_worst_3"])
        for result in results
        if result.get("metric_plane_camera", {}).get("trimmed_mean_residual_ft_drop_worst_3") is not None
    ]
    metric_plane_top_residual_refits = [
        result["metric_plane_camera"].get("top_residual_refit_diagnostic", {})
        for result in results
        if isinstance(result.get("metric_plane_camera"), Mapping)
        and isinstance(result["metric_plane_camera"].get("top_residual_refit_diagnostic"), Mapping)
    ]
    metric_plane_top_residual_refit_inlier_means = [
        float(refit["inlier_mean_residual_ft"])
        for refit in metric_plane_top_residual_refits
        if refit.get("status") == "scored" and refit.get("inlier_mean_residual_ft") is not None
    ]
    metric_plane_top_residual_refit_all_label_means = [
        float(refit["all_label_mean_residual_ft"])
        for refit in metric_plane_top_residual_refits
        if refit.get("status") == "scored" and refit.get("all_label_mean_residual_ft") is not None
    ]
    metric_plane_top_residual_progression_summary = _top_residual_refit_progression_summary(
        results,
        target_mean_residual_ft=target_mean_residual_ft,
    )
    metric_plane_outlier_candidate_counts = [
        len(result["metric_plane_camera"].get("outlier_review_candidates", []))
        for result in results
        if isinstance(result.get("metric_plane_camera"), Mapping)
    ]
    metric_plane_line_intersection_candidate_count = sum(
        1
        for result in results
        if isinstance(result.get("metric_plane_camera"), Mapping)
        for candidate in result["metric_plane_camera"].get("outlier_review_candidates", [])
        if candidate.get("line_intersection_available") is True
    )
    metric_plane_line_override_oracles = [
        result["metric_plane_camera"].get("line_intersection_override_oracle", {})
        for result in results
        if isinstance(result.get("metric_plane_camera"), Mapping)
        and isinstance(result["metric_plane_camera"].get("line_intersection_override_oracle"), Mapping)
    ]
    metric_plane_line_override_candidate_count = sum(
        int(oracle.get("override_candidate_count", 0))
        for oracle in metric_plane_line_override_oracles
    )
    metric_plane_line_override_means = [
        float(oracle["mean_residual_ft"])
        for oracle in metric_plane_line_override_oracles
        if oracle.get("status") == "scored" and oracle.get("mean_residual_ft") is not None
    ]
    metric_plane_line_override_original_reviewed_means = [
        float(oracle["original_reviewed_mean_residual_ft"])
        for oracle in metric_plane_line_override_oracles
        if oracle.get("status") == "scored" and oracle.get("original_reviewed_mean_residual_ft") is not None
    ]
    metric_plane_top_line_override_oracles = [
        result["metric_plane_camera"].get("top_residual_line_intersection_override_oracle", {})
        for result in results
        if isinstance(result.get("metric_plane_camera"), Mapping)
        and isinstance(
            result["metric_plane_camera"].get("top_residual_line_intersection_override_oracle"),
            Mapping,
        )
    ]
    metric_plane_top_line_override_candidate_count = sum(
        int(oracle.get("override_candidate_count", 0))
        for oracle in metric_plane_top_line_override_oracles
    )
    metric_plane_top_line_override_means = [
        float(oracle["mean_residual_ft"])
        for oracle in metric_plane_top_line_override_oracles
        if oracle.get("status") == "scored" and oracle.get("mean_residual_ft") is not None
    ]
    metric_plane_top_line_override_original_reviewed_means = [
        float(oracle["original_reviewed_mean_residual_ft"])
        for oracle in metric_plane_top_line_override_oracles
        if oracle.get("status") == "scored" and oracle.get("original_reviewed_mean_residual_ft") is not None
    ]
    metric_plane_relaxed_top_line_override_oracles = [
        result["metric_plane_camera"].get("top_residual_relaxed_line_intersection_override_oracle", {})
        for result in results
        if isinstance(result.get("metric_plane_camera"), Mapping)
        and isinstance(
            result["metric_plane_camera"].get("top_residual_relaxed_line_intersection_override_oracle"),
            Mapping,
        )
    ]
    metric_plane_relaxed_top_line_override_candidate_count = sum(
        int(oracle.get("override_candidate_count", 0))
        for oracle in metric_plane_relaxed_top_line_override_oracles
    )
    metric_plane_relaxed_top_line_override_means = [
        float(oracle["mean_residual_ft"])
        for oracle in metric_plane_relaxed_top_line_override_oracles
        if oracle.get("status") == "scored" and oracle.get("mean_residual_ft") is not None
    ]
    metric_plane_relaxed_top_line_override_original_reviewed_means = [
        float(oracle["original_reviewed_mean_residual_ft"])
        for oracle in metric_plane_relaxed_top_line_override_oracles
        if oracle.get("status") == "scored" and oracle.get("original_reviewed_mean_residual_ft") is not None
    ]
    full_intrinsics_top_line_override_oracles = [
        result["full_intrinsics_metric_plane_camera"].get("top_residual_line_intersection_override_oracle", {})
        for result in results
        if isinstance(result.get("full_intrinsics_metric_plane_camera"), Mapping)
        and isinstance(
            result["full_intrinsics_metric_plane_camera"].get("top_residual_line_intersection_override_oracle"),
            Mapping,
        )
    ]
    full_intrinsics_top_line_override_candidate_count = sum(
        int(oracle.get("override_candidate_count", 0))
        for oracle in full_intrinsics_top_line_override_oracles
    )
    full_intrinsics_top_line_override_means = [
        float(oracle["mean_residual_ft"])
        for oracle in full_intrinsics_top_line_override_oracles
        if oracle.get("status") == "scored" and oracle.get("mean_residual_ft") is not None
    ]
    full_intrinsics_top_line_override_original_reviewed_means = [
        float(oracle["original_reviewed_mean_residual_ft"])
        for oracle in full_intrinsics_top_line_override_oracles
        if oracle.get("status") == "scored" and oracle.get("original_reviewed_mean_residual_ft") is not None
    ]
    full_intrinsics_all_strict_line_override_oracles = [
        result["full_intrinsics_metric_plane_camera"].get(
            "all_strict_endpoint_line_intersection_override_oracle",
            {},
        )
        for result in results
        if isinstance(result.get("full_intrinsics_metric_plane_camera"), Mapping)
        and isinstance(
            result["full_intrinsics_metric_plane_camera"].get(
                "all_strict_endpoint_line_intersection_override_oracle"
            ),
            Mapping,
        )
    ]
    full_intrinsics_all_strict_line_override_candidate_count = sum(
        int(oracle.get("override_candidate_count", 0))
        for oracle in full_intrinsics_all_strict_line_override_oracles
    )
    full_intrinsics_all_strict_line_override_means = [
        float(oracle["mean_residual_ft"])
        for oracle in full_intrinsics_all_strict_line_override_oracles
        if oracle.get("status") == "scored" and oracle.get("mean_residual_ft") is not None
    ]
    full_intrinsics_all_strict_line_override_original_reviewed_means = [
        float(oracle["original_reviewed_mean_residual_ft"])
        for oracle in full_intrinsics_all_strict_line_override_oracles
        if oracle.get("status") == "scored" and oracle.get("original_reviewed_mean_residual_ft") is not None
    ]
    full_intrinsics_quality_gated_line_override_sweeps = [
        result["full_intrinsics_metric_plane_camera"].get("quality_gated_line_intersection_override_sweep", {})
        for result in results
        if isinstance(result.get("full_intrinsics_metric_plane_camera"), Mapping)
        and isinstance(
            result["full_intrinsics_metric_plane_camera"].get("quality_gated_line_intersection_override_sweep"),
            Mapping,
        )
    ]
    full_intrinsics_quality_gated_line_override_summary = _quality_gated_line_override_profile_summary(
        full_intrinsics_quality_gated_line_override_sweeps,
        full_clip_count=len(results),
    )
    full_intrinsics_quality_gated_best = full_intrinsics_quality_gated_line_override_summary["best_profile"]
    full_intrinsics_model_projected_quality_gated_line_override_sweeps = [
        result["full_intrinsics_metric_plane_camera"].get(
            "model_projected_quality_gated_line_intersection_override_sweep", {}
        )
        for result in results
        if isinstance(result.get("full_intrinsics_metric_plane_camera"), Mapping)
        and isinstance(
            result["full_intrinsics_metric_plane_camera"].get(
                "model_projected_quality_gated_line_intersection_override_sweep"
            ),
            Mapping,
        )
    ]
    full_intrinsics_model_projected_quality_gated_line_override_summary = (
        _quality_gated_line_override_profile_summary(
            full_intrinsics_model_projected_quality_gated_line_override_sweeps,
            full_clip_count=len(results),
        )
    )
    full_intrinsics_model_projected_quality_gated_best = (
        full_intrinsics_model_projected_quality_gated_line_override_summary["best_profile"]
    )
    point_line_rmses = [
        float(result["point_line_camera"]["optimized_reprojection_rmse_px"])
        for result in results
        if result.get("point_line_camera", {}).get("status") == "fit"
    ]
    point_line_world_means = [
        float(result["point_line_camera"]["mean_residual_ft"])
        for result in results
        if result.get("point_line_camera", {}).get("status") == "fit"
        and result.get("point_line_camera", {}).get("mean_residual_ft") is not None
    ]
    point_line_sample_counts = [
        int(result["point_line_camera"]["line_pixel_sample_count"])
        for result in results
        if result.get("point_line_camera", {}).get("status") == "fit"
        and result.get("point_line_camera", {}).get("line_pixel_sample_count") is not None
    ]
    point_line_samples_per_observation = [
        float(result["point_line_camera"]["line_pixel_samples_per_observation"])
        for result in results
        if result.get("point_line_camera", {}).get("status") == "fit"
        and result.get("point_line_camera", {}).get("line_pixel_samples_per_observation") is not None
    ]
    point_line_best_world_means = [
        float(result["point_line_camera"]["best_weighted_camera"]["mean_residual_ft"])
        for result in results
        if result.get("point_line_camera", {}).get("status") == "fit"
        and isinstance(result.get("point_line_camera", {}).get("best_weighted_camera"), Mapping)
        and result["point_line_camera"]["best_weighted_camera"].get("mean_residual_ft") is not None
    ]
    point_line_pair_oracle_world_means = [
        float(result["point_line_camera"]["pair_subset_oracle"]["mean_residual_ft"])
        for result in results
        if result.get("point_line_camera", {}).get("status") == "fit"
        and isinstance(result.get("point_line_camera", {}).get("pair_subset_oracle"), Mapping)
        and result["point_line_camera"]["pair_subset_oracle"].get("mean_residual_ft") is not None
    ]
    safe_selected_world_means = [
        float(result["safe_selected_camera"]["mean_residual_ft"])
        for result in results
        if result.get("safe_selected_camera", {}).get("mean_residual_ft") is not None
    ]
    safe_source_counts = _source_counts(
        str(result["safe_selected_camera"].get("source"))
        for result in results
        if isinstance(result.get("safe_selected_camera"), Mapping)
    )
    summary = {
        "sample_count": sample_count,
        "full_15pt_clip_count": len(results),
        "partial_excluded_count": len(partial_excluded),
        "lm_target_mean_residual_ft": float(target_mean_residual_ft),
        "corner_seed_mean_residual_ft_mean": round(_mean(corner_means), 6),
        "corner_seed_mean_residual_ft_median": round(_median(corner_means), 6),
        "lm_optimized_mean_residual_ft_mean": round(_mean(lm_means), 6),
        "lm_optimized_mean_residual_ft_median": round(_median(lm_means), 6),
        "lm_target_pass_clip_count": sum(1 for result in results if result["lm_passes_target"]),
        "lm_mean_residual_delta_ft_mean": round(_mean([a - b for a, b in zip(corner_means, lm_means, strict=True)]), 6),
        "distorted_camera_rmse_px_mean": None if not distorted_rmses else round(_mean(distorted_rmses), 6),
        "distorted_camera_rmse_px_median": None if not distorted_rmses else round(_median(distorted_rmses), 6),
        "distorted_camera_mean_residual_ft_mean": None if not distorted_world_means else round(_mean(distorted_world_means), 6),
        "distorted_camera_mean_residual_ft_median": None if not distorted_world_means else round(_median(distorted_world_means), 6),
        "all15_camera_rmse_px_mean": None if not all15_rmses else round(_mean(all15_rmses), 6),
        "all15_camera_rmse_px_median": None if not all15_rmses else round(_median(all15_rmses), 6),
        "all15_camera_floor_mean_residual_ft_mean": None if not all15_floor_world_means else round(_mean(all15_floor_world_means), 6),
        "all15_camera_floor_mean_residual_ft_median": None if not all15_floor_world_means else round(_median(all15_floor_world_means), 6),
        "metric_plane_camera_rmse_px_mean": None if not metric_plane_rmses else round(_mean(metric_plane_rmses), 6),
        "metric_plane_camera_rmse_px_median": None if not metric_plane_rmses else round(_median(metric_plane_rmses), 6),
        "metric_plane_camera_mean_residual_ft_mean": None if not metric_plane_world_means else round(_mean(metric_plane_world_means), 6),
        "metric_plane_camera_mean_residual_ft_median": None if not metric_plane_world_means else round(_median(metric_plane_world_means), 6),
        "full_intrinsics_metric_plane_rmse_px_mean": (
            None if not full_intrinsics_metric_plane_rmses else round(_mean(full_intrinsics_metric_plane_rmses), 6)
        ),
        "full_intrinsics_metric_plane_mean_residual_ft_mean": (
            None
            if not full_intrinsics_metric_plane_world_means
            else round(_mean(full_intrinsics_metric_plane_world_means), 6)
        ),
        "full_intrinsics_metric_plane_mean_residual_ft_median": (
            None
            if not full_intrinsics_metric_plane_world_means
            else round(_median(full_intrinsics_metric_plane_world_means), 6)
        ),
        "full_intrinsics_metric_plane_diagnostic_only": True,
        "metric_plane_global_trimmed_worst8_mean_residual_ft": _trimmed_mean_or_none(
            metric_plane_all_keypoint_residuals,
            drop_worst_count=8,
        ),
        "metric_plane_global_trimmed_worst8_diagnostic_only": True,
        "metric_plane_per_clip_trimmed_worst3_mean_residual_ft_mean": (
            None if not metric_plane_trimmed_worst3_means else round(_mean(metric_plane_trimmed_worst3_means), 6)
        ),
        "metric_plane_top_residual_refit_drop3_inlier_mean_residual_ft_mean": (
            None
            if not metric_plane_top_residual_refit_inlier_means
            else round(_mean(metric_plane_top_residual_refit_inlier_means), 6)
        ),
        "metric_plane_top_residual_refit_drop3_all_label_mean_residual_ft_mean": (
            None
            if not metric_plane_top_residual_refit_all_label_means
            else round(_mean(metric_plane_top_residual_refit_all_label_means), 6)
        ),
        "metric_plane_top_residual_refit_drop3_diagnostic_only": True,
        "metric_plane_top_residual_refit_min_drop_count_for_mean_target": metric_plane_top_residual_progression_summary[
            "min_drop_count_for_mean_target"
        ],
        "metric_plane_top_residual_refit_min_drop_count_for_all_clips_target": metric_plane_top_residual_progression_summary[
            "min_drop_count_for_all_clips_target"
        ],
        "metric_plane_top_residual_refit_drop4_inlier_mean_residual_ft_mean": metric_plane_top_residual_progression_summary[
            "drop4_inlier_mean_residual_ft_mean"
        ],
        "metric_plane_top_residual_refit_drop4_inlier_mean_residual_ft_max": metric_plane_top_residual_progression_summary[
            "drop4_inlier_mean_residual_ft_max"
        ],
        "metric_plane_top_residual_refit_drop5_inlier_mean_residual_ft_mean": metric_plane_top_residual_progression_summary[
            "drop5_inlier_mean_residual_ft_mean"
        ],
        "metric_plane_top_residual_refit_drop5_inlier_mean_residual_ft_max": metric_plane_top_residual_progression_summary[
            "drop5_inlier_mean_residual_ft_max"
        ],
        "metric_plane_top_residual_refit_drop5_all_label_mean_residual_ft_mean": metric_plane_top_residual_progression_summary[
            "drop5_all_label_mean_residual_ft_mean"
        ],
        "metric_plane_top_residual_refit_drop4_line_status_counts": metric_plane_top_residual_progression_summary[
            "drop4_line_intersection_status_counts"
        ],
        "metric_plane_top_residual_refit_drop5_line_status_counts": metric_plane_top_residual_progression_summary[
            "drop5_line_intersection_status_counts"
        ],
        "metric_plane_top_residual_refit_progression_diagnostic_only": True,
        "metric_plane_outlier_review_candidate_count": sum(metric_plane_outlier_candidate_counts),
        "metric_plane_line_intersection_review_candidate_count": metric_plane_line_intersection_candidate_count,
        "metric_plane_line_intersection_override_candidate_count": metric_plane_line_override_candidate_count,
        "metric_plane_line_intersection_override_mean_residual_ft_mean": (
            None if not metric_plane_line_override_means else round(_mean(metric_plane_line_override_means), 6)
        ),
        "metric_plane_line_intersection_override_original_reviewed_mean_residual_ft_mean": (
            None
            if not metric_plane_line_override_original_reviewed_means
            else round(_mean(metric_plane_line_override_original_reviewed_means), 6)
        ),
        "metric_plane_line_intersection_override_diagnostic_only": True,
        "metric_plane_top_residual_line_override_candidate_count": metric_plane_top_line_override_candidate_count,
        "metric_plane_top_residual_line_override_mean_residual_ft_mean": (
            None if not metric_plane_top_line_override_means else round(_mean(metric_plane_top_line_override_means), 6)
        ),
        "metric_plane_top_residual_line_override_original_reviewed_mean_residual_ft_mean": (
            None
            if not metric_plane_top_line_override_original_reviewed_means
            else round(_mean(metric_plane_top_line_override_original_reviewed_means), 6)
        ),
        "metric_plane_top_residual_line_override_diagnostic_only": True,
        "metric_plane_top_residual_relaxed_line_override_candidate_count": (
            metric_plane_relaxed_top_line_override_candidate_count
        ),
        "metric_plane_top_residual_relaxed_line_override_mean_residual_ft_mean": (
            None
            if not metric_plane_relaxed_top_line_override_means
            else round(_mean(metric_plane_relaxed_top_line_override_means), 6)
        ),
        "metric_plane_top_residual_relaxed_line_override_original_reviewed_mean_residual_ft_mean": (
            None
            if not metric_plane_relaxed_top_line_override_original_reviewed_means
            else round(_mean(metric_plane_relaxed_top_line_override_original_reviewed_means), 6)
        ),
        "metric_plane_top_residual_relaxed_line_override_diagnostic_only": True,
        "full_intrinsics_top_residual_line_override_candidate_count": (
            full_intrinsics_top_line_override_candidate_count
        ),
        "full_intrinsics_top_residual_line_override_mean_residual_ft_mean": (
            None if not full_intrinsics_top_line_override_means else round(_mean(full_intrinsics_top_line_override_means), 6)
        ),
        "full_intrinsics_top_residual_line_override_original_reviewed_mean_residual_ft_mean": (
            None
            if not full_intrinsics_top_line_override_original_reviewed_means
            else round(_mean(full_intrinsics_top_line_override_original_reviewed_means), 6)
        ),
        "full_intrinsics_top_residual_line_override_diagnostic_only": True,
        "full_intrinsics_all_strict_line_override_candidate_count": (
            full_intrinsics_all_strict_line_override_candidate_count
        ),
        "full_intrinsics_all_strict_line_override_mean_residual_ft_mean": (
            None
            if not full_intrinsics_all_strict_line_override_means
            else round(_mean(full_intrinsics_all_strict_line_override_means), 6)
        ),
        "full_intrinsics_all_strict_line_override_original_reviewed_mean_residual_ft_mean": (
            None
            if not full_intrinsics_all_strict_line_override_original_reviewed_means
            else round(_mean(full_intrinsics_all_strict_line_override_original_reviewed_means), 6)
        ),
        "full_intrinsics_all_strict_line_override_diagnostic_only": True,
        "full_intrinsics_quality_gated_line_override_profile_count": (
            full_intrinsics_quality_gated_line_override_summary["profile_count"]
        ),
        "full_intrinsics_quality_gated_line_override_best_profile_id": (
            None if full_intrinsics_quality_gated_best is None else full_intrinsics_quality_gated_best["profile_id"]
        ),
        "full_intrinsics_quality_gated_line_override_best_candidate_count": (
            None
            if full_intrinsics_quality_gated_best is None
            else full_intrinsics_quality_gated_best["override_candidate_count"]
        ),
        "full_intrinsics_quality_gated_line_override_best_mean_residual_ft_mean": (
            None
            if full_intrinsics_quality_gated_best is None
            else full_intrinsics_quality_gated_best["mean_residual_ft_mean"]
        ),
        "full_intrinsics_quality_gated_line_override_best_mean_residual_ft_max": (
            None
            if full_intrinsics_quality_gated_best is None
            else full_intrinsics_quality_gated_best["mean_residual_ft_max"]
        ),
        "full_intrinsics_quality_gated_line_override_best_original_reviewed_mean_residual_ft_mean": (
            None
            if full_intrinsics_quality_gated_best is None
            else full_intrinsics_quality_gated_best["original_reviewed_mean_residual_ft_mean"]
        ),
        "full_intrinsics_quality_gated_line_override_profiles": (
            full_intrinsics_quality_gated_line_override_summary["profiles"]
        ),
        "full_intrinsics_quality_gated_line_override_diagnostic_only": True,
        "full_intrinsics_model_projected_quality_gated_line_override_profile_count": (
            full_intrinsics_model_projected_quality_gated_line_override_summary["profile_count"]
        ),
        "full_intrinsics_model_projected_quality_gated_line_override_best_profile_id": (
            None
            if full_intrinsics_model_projected_quality_gated_best is None
            else full_intrinsics_model_projected_quality_gated_best["profile_id"]
        ),
        "full_intrinsics_model_projected_quality_gated_line_override_best_candidate_count": (
            None
            if full_intrinsics_model_projected_quality_gated_best is None
            else full_intrinsics_model_projected_quality_gated_best["override_candidate_count"]
        ),
        "full_intrinsics_model_projected_quality_gated_line_override_best_mean_residual_ft_mean": (
            None
            if full_intrinsics_model_projected_quality_gated_best is None
            else full_intrinsics_model_projected_quality_gated_best["mean_residual_ft_mean"]
        ),
        "full_intrinsics_model_projected_quality_gated_line_override_best_mean_residual_ft_max": (
            None
            if full_intrinsics_model_projected_quality_gated_best is None
            else full_intrinsics_model_projected_quality_gated_best["mean_residual_ft_max"]
        ),
        "full_intrinsics_model_projected_quality_gated_line_override_best_original_reviewed_mean_residual_ft_mean": (
            None
            if full_intrinsics_model_projected_quality_gated_best is None
            else full_intrinsics_model_projected_quality_gated_best["original_reviewed_mean_residual_ft_mean"]
        ),
        "full_intrinsics_model_projected_quality_gated_line_override_profiles": (
            full_intrinsics_model_projected_quality_gated_line_override_summary["profiles"]
        ),
        "full_intrinsics_model_projected_quality_gated_line_override_diagnostic_only": True,
        "full_intrinsics_model_projected_quality_gated_uses_reviewed_line_positions_for_matching": False,
        "point_line_fit_clip_count": len(point_line_rmses),
        "point_line_camera_rmse_px_mean": None if not point_line_rmses else round(_mean(point_line_rmses), 6),
        "point_line_camera_mean_residual_ft_mean": None if not point_line_world_means else round(_mean(point_line_world_means), 6),
        "point_line_camera_mean_residual_ft_median": None if not point_line_world_means else round(_median(point_line_world_means), 6),
        "point_line_segment_pixel_sample_count_mean": (
            None if not point_line_sample_counts else round(_mean(point_line_sample_counts), 6)
        ),
        "point_line_segment_pixel_samples_per_observation_mean": (
            None if not point_line_samples_per_observation else round(_mean(point_line_samples_per_observation), 6)
        ),
        "point_line_weight_sweep_candidate_weights": [float(value) for value in POINT_LINE_WEIGHT_SWEEP],
        "point_line_weight_sweep_best_mean_residual_ft_mean": (
            None if not point_line_best_world_means else round(_mean(point_line_best_world_means), 6)
        ),
        "point_line_weight_sweep_best_mean_residual_ft_median": (
            None if not point_line_best_world_means else round(_median(point_line_best_world_means), 6)
        ),
        "point_line_pair_subset_oracle_mean_residual_ft_mean": (
            None if not point_line_pair_oracle_world_means else round(_mean(point_line_pair_oracle_world_means), 6)
        ),
        "point_line_pair_subset_oracle_mean_residual_ft_median": (
            None if not point_line_pair_oracle_world_means else round(_median(point_line_pair_oracle_world_means), 6)
        ),
        "neural_keypoint_checkpoint_candidate_count": neural_keypoint_evidence["candidate_count"],
        "neural_keypoint_real_label_candidate_count": neural_keypoint_evidence["real_label_candidate_count"],
        "neural_keypoint_gate_pass_count": neural_keypoint_evidence["gate_pass_count"],
        "neural_keypoint_best_real_median_px": best_neural_metric,
        "neural_keypoint_diagnostic_only": True,
        "mobilenet_v3_keypoint_checkpoint_candidate_count": mobilenet_v3_keypoint_evidence["candidate_count"],
        "mobilenet_v3_keypoint_scored_candidate_count": mobilenet_v3_keypoint_evidence["scored_candidate_count"],
        "mobilenet_v3_keypoint_best_median_px": best_mobilenet_metric,
        "mobilenet_v3_keypoint_status": mobilenet_v3_keypoint_evidence["status"],
        "safe_selected_camera_mean_residual_ft_mean": (
            None if not safe_selected_world_means else round(_mean(safe_selected_world_means), 6)
        ),
        "safe_selected_camera_mean_residual_ft_median": (
            None if not safe_selected_world_means else round(_median(safe_selected_world_means), 6)
        ),
        "safe_selected_camera_source_counts": safe_source_counts,
    }
    report = {
        "schema_version": 1,
        "artifact_type": "racketsport_overlapping_court_calibration_eval",
        "status": "ran_not_verified",
        "verified": False,
        "not_cal3_verified": True,
        "summary": summary,
        "neural_keypoint_checkpoint_evidence": neural_keypoint_evidence,
        "mobilenet_v3_keypoint_checkpoint_evidence": mobilenet_v3_keypoint_evidence,
        "partial_excluded": partial_excluded,
        "results": results,
        "notes": [
            "LM homography residuals are measured on reviewed court keypoint labels, not raw color-mask pixels.",
            "This evaluates the proposed LM refinement seam but does not promote no-tap CAL-3 calibration.",
            "Metric-plane trimmed residuals are diagnostic-only outlier analysis and must not be used as CAL pass criteria.",
            "Neural court-keypoint checkpoints are scored as diagnostic evidence only unless their reviewed-label gate passes.",
            "MobileNetV3 court-keypoint checkpoints are opt-in diagnostic evidence and are reported separately from heatmap checkpoints.",
        ],
    }
    if out_path is not None:
        output = Path(out_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def render_metric_plane_outlier_review_packet(
    *,
    report_path: str | Path,
    eval_root: str | Path,
    out_dir: str | Path,
    crop_radius_px: int = 96,
    max_candidates: int | None = None,
    cv2_module: Any | None = None,
) -> dict[str, Any]:
    """Render fail-closed visual crops for metric-plane outlier candidates."""

    if crop_radius_px <= 0:
        raise ValueError("crop_radius_px must be positive")
    if max_candidates is not None and max_candidates <= 0:
        raise ValueError("max_candidates must be positive when provided")
    report_file = Path(report_path)
    report = json.loads(report_file.read_text(encoding="utf-8"))
    if report.get("artifact_type") != "racketsport_overlapping_court_calibration_eval":
        raise ValueError("report must be a racketsport overlapping-court calibration eval artifact")

    cv2 = cv2_module or _cv2()
    np = _np()
    root = Path(eval_root)
    output_dir = Path(out_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    candidates_seen = 0
    items: list[dict[str, Any]] = []
    contact_tiles: list[Any] = []
    frame_cache: dict[tuple[str, int], Any] = {}
    clip_names: set[str] = set()

    for result in report.get("results", []):
        if max_candidates is not None and candidates_seen >= max_candidates:
            break
        if not isinstance(result, Mapping):
            continue
        clip = str(result.get("clip") or "")
        if not clip:
            continue
        clip_names.add(clip)
        metric_plane = result.get("metric_plane_camera")
        if not isinstance(metric_plane, Mapping):
            continue
        candidates = metric_plane.get("outlier_review_candidates")
        if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
            continue
        label_path = _resolve_report_label_path(result, eval_root=root)
        frame_index = _metric_packet_frame_index(label_path)
        source_video = root / clip / "source.mp4"
        cache_key = (clip, frame_index)
        if cache_key not in frame_cache:
            frame_cache[cache_key] = _read_video_frame(source_video, frame_index, cv2)
        frame = frame_cache[cache_key]

        for candidate_index, raw_candidate in enumerate(candidates):
            if max_candidates is not None and candidates_seen >= max_candidates:
                break
            if not isinstance(raw_candidate, Mapping):
                continue
            candidate = dict(raw_candidate)
            reviewed_px = _point2(candidate.get("reviewed_image_px"), "candidate.reviewed_image_px")
            model_px = _point2(candidate.get("model_projected_image_px"), "candidate.model_projected_image_px")
            line_px = None
            if candidate.get("line_intersection_available") is True:
                try:
                    line_px = _point2(candidate.get("line_intersection_image_px"), "candidate.line_intersection_image_px")
                except ValueError:
                    line_px = None
            center = _packet_crop_center([reviewed_px, model_px, line_px])
            crop, crop_box = _crop_around_point(frame, center, radius_px=crop_radius_px)
            annotated = crop.copy()
            _draw_metric_plane_outlier_crop(
                cv2,
                annotated,
                candidate=candidate,
                crop_origin=(crop_box[0], crop_box[1]),
            )

            keypoint = _safe_filename_token(str(candidate.get("keypoint") or "keypoint"))
            image_name = f"{candidates_seen + 1:03d}_{_safe_filename_token(clip)}_{keypoint}.jpg"
            image_path = images_dir / image_name
            if not cv2.imwrite(str(image_path), annotated):
                raise RuntimeError(f"failed writing outlier review crop: {image_path}")
            contact_tiles.append(_review_packet_tile(annotated, crop_radius_px * 2, np))
            line_support = _line_intersection_support(candidate)
            items.append(
                {
                    "review_id": f"metric_plane_outlier_{candidates_seen + 1:03d}",
                    "clip": clip,
                    "keypoint": str(candidate.get("keypoint") or ""),
                    "source_video": str(source_video),
                    "frame_index": frame_index,
                    "image": str(image_path),
                    "crop_box_xyxy": [int(value) for value in crop_box],
                    "residual_ft": candidate.get("residual_ft"),
                    "model_delta_px": candidate.get("model_delta_px"),
                    "line_intersection_available": candidate.get("line_intersection_available") is True,
                    "line_intersection_support": line_support,
                    "reviewed_image_px": candidate.get("reviewed_image_px"),
                    "model_projected_image_px": candidate.get("model_projected_image_px"),
                    "line_intersection_image_px": candidate.get("line_intersection_image_px"),
                    "diagnostic_only": True,
                    "candidate_rank_in_clip": candidate_index + 1,
                }
            )
            candidates_seen += 1

    contact_sheet_path: Path | None = None
    if contact_tiles:
        contact_sheet = _review_packet_contact_sheet(contact_tiles, np)
        contact_sheet_path = output_dir / "metric_plane_outlier_contact_sheet.jpg"
        if not cv2.imwrite(str(contact_sheet_path), contact_sheet):
            raise RuntimeError(f"failed writing outlier review contact sheet: {contact_sheet_path}")

    line_support_counts = _source_counts([str(item["line_intersection_support"]) for item in items])
    summary = {
        "schema_version": 1,
        "artifact_type": "racketsport_metric_plane_outlier_review_packet",
        "status": "needs_human_review" if items else "no_outlier_candidates",
        "verified": False,
        "not_cal3_verified": True,
        "diagnostic_only": True,
        "source_report": str(report_file),
        "eval_root": str(root),
        "out_dir": str(output_dir),
        "item_count": len(items),
        "candidate_count": len(items),
        "clip_count": len(clip_names),
        "line_intersection_item_count": sum(1 for item in items if item["line_intersection_available"]),
        "line_intersection_support_counts": line_support_counts,
        "contact_sheet": None if contact_sheet_path is None else str(contact_sheet_path),
        "legend_bgr": {
            "reviewed_label": [0, 220, 0],
            "metric_plane_projection": [0, 0, 255],
            "line_intersection_candidate": [255, 255, 0],
        },
        "items": items,
        "notes": [
            "This packet visualizes residual outliers only; it does not mutate labels or promote CAL-3.",
            "Line-intersection support is diagnostic and must be human-reviewed before any relabel or promotion.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metric_plane_outlier_review_packet.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def render_metric_plane_top_residual_refit_review_packet(
    *,
    report_path: str | Path,
    eval_root: str | Path,
    out_dir: str | Path,
    crop_radius_px: int = 96,
    max_candidates: int | None = None,
    drop_count: int | None = None,
    cv2_module: Any | None = None,
) -> dict[str, Any]:
    """Render fail-closed crops for keypoints excluded by the top-residual refit diagnostic."""

    if crop_radius_px <= 0:
        raise ValueError("crop_radius_px must be positive")
    if max_candidates is not None and max_candidates <= 0:
        raise ValueError("max_candidates must be positive when provided")
    if drop_count is not None and drop_count < 0:
        raise ValueError("drop_count must be non-negative when provided")
    report_file = Path(report_path)
    report = json.loads(report_file.read_text(encoding="utf-8"))
    if report.get("artifact_type") != "racketsport_overlapping_court_calibration_eval":
        raise ValueError("report must be a racketsport overlapping-court calibration eval artifact")

    cv2 = cv2_module or _cv2()
    np = _np()
    root = Path(eval_root)
    output_dir = Path(out_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    items: list[dict[str, Any]] = []
    contact_tiles: list[Any] = []
    frame_cache: dict[tuple[str, int], Any] = {}
    clip_names: set[str] = set()
    inlier_means: list[float] = []
    all_label_means: list[float] = []

    for result in report.get("results", []):
        if max_candidates is not None and len(items) >= max_candidates:
            break
        if not isinstance(result, Mapping):
            continue
        clip = str(result.get("clip") or "")
        if not clip:
            continue
        metric_plane = result.get("metric_plane_camera")
        if not isinstance(metric_plane, Mapping):
            continue
        diagnostic = _top_residual_refit_payload_for_drop_count(metric_plane, drop_count=drop_count)
        if not isinstance(diagnostic, Mapping) or diagnostic.get("status") != "scored":
            continue
        dropped = diagnostic.get("dropped_keypoints")
        if not isinstance(dropped, Sequence) or isinstance(dropped, (str, bytes)):
            continue
        clip_names.add(clip)
        inlier_mean = _optional_number(diagnostic.get("inlier_mean_residual_ft"))
        all_label_mean = _optional_number(diagnostic.get("all_label_mean_residual_ft"))
        if inlier_mean is not None:
            inlier_means.append(inlier_mean)
        if all_label_mean is not None:
            all_label_means.append(all_label_mean)
        details_by_name = _top_residual_detail_by_keypoint(diagnostic)
        outlier_by_name = _outlier_candidate_by_keypoint(metric_plane)

        label_path = _resolve_report_label_path(result, eval_root=root)
        frame_index = _metric_packet_frame_index(label_path)
        source_video = root / clip / "source.mp4"
        cache_key = (clip, frame_index)
        if cache_key not in frame_cache:
            frame_cache[cache_key] = _read_video_frame(source_video, frame_index, cv2)
        frame = frame_cache[cache_key]

        for dropped_index, raw_name in enumerate(dropped):
            if max_candidates is not None and len(items) >= max_candidates:
                break
            name = str(raw_name)
            detail = details_by_name.get(name, {})
            outlier = outlier_by_name.get(name, {})
            try:
                reviewed_px = _point2(
                    detail.get("reviewed_image_px", outlier.get("reviewed_image_px")),
                    f"{name}.reviewed_image_px",
                )
                model_px = _point2(
                    detail.get("refit_projected_image_px", outlier.get("model_projected_image_px")),
                    f"{name}.refit_projected_image_px",
                )
            except ValueError:
                continue
            line_px = None
            line_source = detail if detail.get("line_intersection_available") is True else outlier
            if line_source.get("line_intersection_available") is True:
                try:
                    line_px = _point2(line_source.get("line_intersection_image_px"), "line_intersection_image_px")
                except ValueError:
                    line_px = None
            candidate_for_drawing = dict(outlier)
            for line_key in (
                "expected_line_names",
                "line_names",
                "line_intersection_status",
                "line_support_modes",
                "line_intersection_delta_px",
                "model_to_line_intersection_delta_px",
                "missing_line_names",
            ):
                if detail.get(line_key) is not None:
                    candidate_for_drawing[line_key] = detail.get(line_key)
            source_residual = (
                detail.get("source_residual_ft")
                if detail.get("source_residual_ft") is not None
                else _top_residual_value_for_keypoint(diagnostic, name)
            )
            candidate_for_drawing.update(
                {
                    "keypoint": name,
                    "residual_ft": source_residual,
                    "reviewed_image_px": [round(reviewed_px[0], 3), round(reviewed_px[1], 3)],
                    "model_projected_image_px": [round(model_px[0], 3), round(model_px[1], 3)],
                    "model_delta_px": round(math.dist(reviewed_px, model_px), 3),
                    "line_intersection_available": line_px is not None,
                }
            )
            if line_px is not None:
                candidate_for_drawing["line_intersection_image_px"] = [round(line_px[0], 3), round(line_px[1], 3)]
            center = _packet_crop_center([reviewed_px, model_px, line_px])
            crop, crop_box = _crop_around_point(frame, center, radius_px=crop_radius_px)
            annotated = crop.copy()
            _draw_metric_plane_outlier_crop(
                cv2,
                annotated,
                candidate=candidate_for_drawing,
                crop_origin=(crop_box[0], crop_box[1]),
            )
            image_name = f"{len(items) + 1:03d}_{_safe_filename_token(clip)}_{_safe_filename_token(name)}.jpg"
            image_path = images_dir / image_name
            if not cv2.imwrite(str(image_path), annotated):
                raise RuntimeError(f"failed writing top-residual refit review crop: {image_path}")
            contact_tiles.append(_review_packet_tile(annotated, crop_radius_px * 2, np))
            line_support = _line_intersection_support(candidate_for_drawing)
            expected_line_names = _string_sequence(
                candidate_for_drawing.get("expected_line_names", candidate_for_drawing.get("line_names"))
            )
            line_status = str(
                candidate_for_drawing.get(
                    "line_intersection_status",
                    "available" if line_px is not None else "missing_line_intersection",
                )
            )
            items.append(
                {
                    "review_id": f"metric_plane_top_residual_refit_{len(items) + 1:03d}",
                    "clip": clip,
                    "keypoint": name,
                    "source_video": str(source_video),
                    "frame_index": frame_index,
                    "image": str(image_path),
                    "crop_box_xyxy": [int(value) for value in crop_box],
                    "candidate_rank_in_clip": dropped_index + 1,
                    "source_residual_ft": source_residual,
                    "refit_model_delta_px": detail.get("refit_model_delta_px", candidate_for_drawing["model_delta_px"]),
                    "refit_inlier_mean_residual_ft": diagnostic.get("inlier_mean_residual_ft"),
                    "refit_all_label_mean_residual_ft": diagnostic.get("all_label_mean_residual_ft"),
                    "line_intersection_available": line_px is not None,
                    "line_intersection_status": line_status,
                    "expected_line_names": expected_line_names,
                    "line_intersection_support": line_support,
                    "reviewed_image_px": candidate_for_drawing["reviewed_image_px"],
                    "refit_projected_image_px": candidate_for_drawing["model_projected_image_px"],
                    "line_intersection_image_px": candidate_for_drawing.get("line_intersection_image_px"),
                    "diagnostic_only": True,
                    "promotes_calibration": False,
                }
            )

    contact_sheet_path: Path | None = None
    if contact_tiles:
        contact_sheet = _review_packet_contact_sheet(contact_tiles, np)
        contact_sheet_path = output_dir / "metric_plane_top_residual_refit_contact_sheet.jpg"
        if not cv2.imwrite(str(contact_sheet_path), contact_sheet):
            raise RuntimeError(f"failed writing top-residual refit contact sheet: {contact_sheet_path}")

    line_support_counts = _source_counts([str(item["line_intersection_support"]) for item in items])
    line_status_counts = _source_counts([str(item["line_intersection_status"]) for item in items])
    summary = {
        "schema_version": 1,
        "artifact_type": "racketsport_metric_plane_top_residual_refit_review_packet",
        "status": "needs_human_review" if items else "no_top_residual_refit_candidates",
        "verified": False,
        "not_cal3_verified": True,
        "diagnostic_only": True,
        "promotes_calibration": False,
        "source_report": str(report_file),
        "eval_root": str(root),
        "out_dir": str(output_dir),
        "item_count": len(items),
        "candidate_count": len(items),
        "clip_count": len(clip_names),
        "line_intersection_item_count": sum(1 for item in items if item["line_intersection_available"]),
        "line_intersection_support_counts": line_support_counts,
        "line_intersection_status_counts": line_status_counts,
        "drop_count": drop_count,
        "inlier_mean_residual_ft_mean": None if not inlier_means else round(_mean(inlier_means), 6),
        "all_label_mean_residual_ft_mean": None if not all_label_means else round(_mean(all_label_means), 6),
        "contact_sheet": None if contact_sheet_path is None else str(contact_sheet_path),
        "legend_bgr": {
            "reviewed_label": [0, 220, 0],
            "refit_projection": [0, 0, 255],
            "line_intersection_candidate": [255, 255, 0],
        },
        "items": items,
        "notes": [
            "This packet visualizes keypoints excluded by a diagnostic residual refit; it does not mutate labels.",
            "The inlier-only refit score is label-selected and must not be used as a CAL pass criterion.",
            "Human review or independent evidence is required before any excluded keypoint can affect calibration scoring.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metric_plane_top_residual_refit_review_packet.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _hough_segments_from_mask(mask: Any, *, config: LineClusterConfig) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    cv2 = _cv2()
    raw = cv2.HoughLinesP(
        mask,
        rho=1,
        theta=math.pi / 180.0,
        threshold=int(config.hough_threshold),
        minLineLength=int(config.min_line_length_px),
        maxLineGap=int(config.max_line_gap_px),
    )
    if raw is None:
        return []
    segments: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for x1, y1, x2, y2 in raw.reshape(-1, 4):
        segment = ((float(x1), float(y1)), (float(x2), float(y2)))
        if _segment_length(segment) >= float(config.min_line_length_px):
            segments.append(segment)
    return segments


def _boundary_cluster(segments: Sequence[tuple[tuple[float, float], tuple[float, float]]]) -> BoundaryCluster:
    points = [point for segment in segments for point in segment]
    line = _fit_line(points)
    segment = _extent_segment(points, line)
    angle = _line_angle_deg(line)
    orientation = "cross" if _angle_diff_mod_180(angle, 0.0) <= 25.0 else "longitudinal"
    return BoundaryCluster(
        orientation=orientation,
        line=line,
        segment=segment,
        support_length_px=sum(_segment_length(segment) for segment in segments),
        source_segment_count=len(segments),
        angle_deg=angle,
    )


def _select_boundary_clusters(clusters: Sequence[BoundaryCluster], *, max_clusters: int) -> list[BoundaryCluster]:
    ordered = sorted(clusters, key=lambda cluster: cluster.support_length_px, reverse=True)
    cross = [cluster for cluster in ordered if cluster.orientation == "cross"][:2]
    longitudinal = [cluster for cluster in ordered if cluster.orientation == "longitudinal"][:2]
    selected = [*cross, *longitudinal]
    if len(selected) < max_clusters:
        used = {id(cluster) for cluster in selected}
        selected.extend(cluster for cluster in ordered if id(cluster) not in used)
    return selected[:max_clusters]


def _fit_line(points: Sequence[tuple[float, float]]) -> tuple[float, float, float]:
    if len(points) < 2:
        raise ValueError("line fit requires at least 2 points")
    mean_x = sum(point[0] for point in points) / len(points)
    mean_y = sum(point[1] for point in points) / len(points)
    sxx = sum((point[0] - mean_x) ** 2 for point in points)
    syy = sum((point[1] - mean_y) ** 2 for point in points)
    sxy = sum((point[0] - mean_x) * (point[1] - mean_y) for point in points)
    angle = 0.5 * math.atan2(2.0 * sxy, sxx - syy)
    return _line_from_point_direction((mean_x, mean_y), (math.cos(angle), math.sin(angle)))


def _line_from_segment(segment: tuple[tuple[float, float], tuple[float, float]]) -> tuple[float, float, float]:
    return _line_from_point_direction(segment[0], (segment[1][0] - segment[0][0], segment[1][1] - segment[0][1]))


def _line_from_point_direction(point: tuple[float, float], direction: tuple[float, float]) -> tuple[float, float, float]:
    dx, dy = direction
    length = math.hypot(dx, dy)
    if length <= 1e-9:
        raise ValueError("line direction must be non-zero")
    dx /= length
    dy /= length
    a = -dy
    b = dx
    c = -(a * point[0] + b * point[1])
    if a < 0.0 or (abs(a) <= 1e-9 and b < 0.0):
        a, b, c = -a, -b, -c
    return (float(a), float(b), float(c))


def _extent_segment(
    points: Sequence[tuple[float, float]],
    line: tuple[float, float, float],
) -> tuple[tuple[float, float], tuple[float, float]]:
    a, b, c = line
    dx, dy = b, -a
    anchor = (-a * c, -b * c)
    projections = [((point[0] - anchor[0]) * dx + (point[1] - anchor[1]) * dy, point) for point in points]
    min_projection = min(value for value, _ in projections)
    max_projection = max(value for value, _ in projections)
    return (
        (anchor[0] + dx * min_projection, anchor[1] + dy * min_projection),
        (anchor[0] + dx * max_projection, anchor[1] + dy * max_projection),
    )


def _line_segment_distance(
    line: tuple[float, float, float],
    segment: tuple[tuple[float, float], tuple[float, float]],
) -> float:
    a, b, c = line
    return (
        abs(a * segment[0][0] + b * segment[0][1] + c)
        + abs(a * segment[1][0] + b * segment[1][1] + c)
    ) / 2.0


def _longest_segment(
    segments: Sequence[tuple[tuple[float, float], tuple[float, float]]],
) -> tuple[tuple[float, float], tuple[float, float]]:
    return max(segments, key=_segment_length)


def _segment_length(segment: tuple[tuple[float, float], tuple[float, float]]) -> float:
    return math.dist(segment[0], segment[1])


def _segment_angle_deg(segment: tuple[tuple[float, float], tuple[float, float]]) -> float:
    return math.degrees(math.atan2(segment[1][1] - segment[0][1], segment[1][0] - segment[0][0]))


def _line_angle_deg(line: tuple[float, float, float]) -> float:
    a, b, _c = line
    angle = math.degrees(math.atan2(-a, b))
    while angle <= -90.0:
        angle += 180.0
    while angle > 90.0:
        angle -= 180.0
    return float(angle)


def _angle_diff_mod_180(first: float, second: float) -> float:
    diff = abs((float(first) - float(second)) % 180.0)
    return min(diff, 180.0 - diff)


def _support_ratio(mask: Any) -> float:
    np = _np()
    total = int(mask.size)
    if total <= 0:
        return 0.0
    return float(np.count_nonzero(mask)) / float(total)


def _as_uint8_bgr(image: Any) -> Any:
    cv2 = _cv2()
    np = _np()
    if image is None or not hasattr(image, "shape") or len(image.shape) < 2:
        raise ValueError("image must be an OpenCV-style array")
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    elif arr.ndim == 3 and arr.shape[2] >= 3:
        arr = arr[:, :, :3]
    else:
        raise ValueError("image must have shape HxW or HxWx3")
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _as_uint8_mask(mask: Any) -> Any:
    np = _np()
    if mask is None or not hasattr(mask, "shape") or len(mask.shape) != 2:
        raise ValueError("mask must be a 2D array")
    arr = np.asarray(mask)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return ((arr > 0).astype(np.uint8)) * 255


def _net_cutoff_y(net_evidence: Mapping[str, Any] | None) -> float | None:
    if not net_evidence:
        return None
    top_tape = net_evidence.get("top_tape_line")
    if isinstance(top_tape, Sequence) and not isinstance(top_tape, (str, bytes)) and top_tape:
        ys = []
        for point in top_tape:
            try:
                xy = _point2(point, "top_tape_line")
            except ValueError:
                continue
            ys.append(xy[1])
        if ys:
            return sum(ys) / len(ys)
    roi = net_evidence.get("roi")
    if isinstance(roi, Mapping) and roi.get("y_min") is not None and roi.get("y_max") is not None:
        return (float(roi["y_min"]) + float(roi["y_max"])) / 2.0
    return None


def _point2(value: Any, name: str) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) < 2:
        raise ValueError(f"{name} must be a two-item point")
    x = _finite_float(value[0], f"{name}[0]")
    y = _finite_float(value[1], f"{name}[1]")
    return [x, y]


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _paired_residual_distances(residuals: Sequence[float]) -> list[float]:
    values = [float(value) for value in residuals]
    return [math.hypot(values[index], values[index + 1]) for index in range(0, len(values), 2)]


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(float(value) for value in values) / len(values)


def _source_counts(values: Sequence[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts


def _string_sequence(value: Any) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [str(item) for item in value]


def _median(values: Sequence[float]) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * float(percentile) / 100.0
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _round_matrix(matrix: Sequence[Sequence[float]]) -> list[list[float]]:
    return [[round(float(value), 9) for value in row] for row in matrix]


def _object_points3(points: Sequence[Sequence[float]]) -> Any:
    np = _np()
    rows: list[list[float]] = []
    for point in points:
        if not isinstance(point, Sequence) or isinstance(point, (str, bytes)) or len(point) < 3:
            raise ValueError("object points must be 3D")
        rows.append([
            _finite_float(point[0], "object_point.x"),
            _finite_float(point[1], "object_point.y"),
            _finite_float(point[2], "object_point.z"),
        ])
    if not rows:
        raise ValueError("object points must be non-empty")
    return np.asarray(rows, dtype=np.float64)


def _image_points2(points: Sequence[Sequence[float]]) -> Any:
    np = _np()
    rows = [_point2(point, "image_point") for point in points]
    if not rows:
        raise ValueError("image points must be non-empty")
    return np.asarray(rows, dtype=np.float64)


def _image_size2(image_size: tuple[int | float, int | float]) -> tuple[float, float]:
    if len(image_size) != 2:
        raise ValueError("image_size must be a width/height pair")
    width = _finite_float(image_size[0], "image_size.width")
    height = _finite_float(image_size[1], "image_size.height")
    if width <= 0.0 or height <= 0.0:
        raise ValueError("image_size dimensions must be positive")
    return width, height


def _initial_camera_parameter_vector(cv2: Any, np: Any, obj: Any, img: Any, *, width: float, height: float) -> Any:
    cx, cy = width / 2.0, height / 2.0
    best_rmse = float("inf")
    best: tuple[float, Any, Any] | None = None
    for focal in np.linspace(max(width, height) * 0.45, max(width, height) * 3.5, 31):
        k = np.asarray([[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
        ok, rvec, tvec = cv2.solvePnP(obj, img, k, None, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            continue
        projected, _ = cv2.projectPoints(obj, rvec, tvec, k, None)
        projected = projected.reshape(-1, 2)
        rmse = float(np.sqrt(np.mean(np.sum((projected - img) ** 2, axis=1))))
        if rmse < best_rmse:
            best_rmse = rmse
            best = (float(focal), rvec.reshape(3), tvec.reshape(3))
    if best is None:
        raise ValueError("could not seed camera pose for joint LM fit")
    focal, rvec, tvec = best
    return np.asarray(
        [
            math.log(max(1.0, focal)),
            float(rvec[0]),
            float(rvec[1]),
            float(rvec[2]),
            float(tvec[0]),
            float(tvec[1]),
            float(tvec[2]),
            0.0,
            0.0,
        ],
        dtype=np.float64,
    )


def _project_with_camera_params(cv2: Any, np: Any, obj: Any, params: Sequence[float], *, cx: float, cy: float) -> Any:
    values = np.asarray(params, dtype=np.float64)
    focal = float(math.exp(float(values[0])))
    rvec = values[1:4].reshape(3, 1)
    tvec = values[4:7].reshape(3, 1)
    k1 = float(values[7])
    k2 = float(values[8])
    k = np.asarray([[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    dist = np.asarray([k1, k2, 0.0, 0.0], dtype=np.float64)
    projected, _ = cv2.projectPoints(obj, rvec, tvec, k, dist)
    return projected.reshape(-1, 2)


def _project_with_full_camera_params(cv2: Any, np: Any, obj: Any, params: Sequence[float]) -> Any:
    values = np.asarray(params, dtype=np.float64)
    fx = float(math.exp(float(values[0])))
    fy = float(math.exp(float(values[1])))
    cx = float(values[2])
    cy = float(values[3])
    rvec = values[4:7].reshape(3, 1)
    tvec = values[7:10].reshape(3, 1)
    k1 = float(values[10])
    k2 = float(values[11])
    k = np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    dist = np.asarray([k1, k2, 0.0, 0.0], dtype=np.float64)
    projected, _ = cv2.projectPoints(obj, rvec, tvec, k, dist)
    return projected.reshape(-1, 2)


def _project_with_fit_payload(cv2: Any, np: Any, obj: Any, fit: Mapping[str, Any]) -> Any:
    intrinsics = fit.get("intrinsics") if isinstance(fit.get("intrinsics"), Mapping) else {}
    distortion = fit.get("distortion") if isinstance(fit.get("distortion"), Mapping) else {}
    extrinsics = fit.get("extrinsics") if isinstance(fit.get("extrinsics"), Mapping) else {}
    rvec = extrinsics.get("rvec")
    tvec = extrinsics.get("tvec")
    if not isinstance(rvec, Sequence) or len(rvec) != 3 or not isinstance(tvec, Sequence) or len(tvec) != 3:
        raise ValueError("fit payload is missing extrinsics.rvec/tvec")
    fx = _finite_float(intrinsics.get("fx"), "fit.intrinsics.fx")
    fy = _finite_float(intrinsics.get("fy", fx), "fit.intrinsics.fy")
    cx = _finite_float(intrinsics.get("cx"), "fit.intrinsics.cx")
    cy = _finite_float(intrinsics.get("cy"), "fit.intrinsics.cy")
    k = np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    dist = np.asarray(
        [
            _finite_float(distortion.get("k1", 0.0), "fit.distortion.k1"),
            _finite_float(distortion.get("k2", 0.0), "fit.distortion.k2"),
            0.0,
            0.0,
        ],
        dtype=np.float64,
    )
    projected, _ = cv2.projectPoints(
        obj,
        np.asarray([float(value) for value in rvec], dtype=np.float64).reshape(3, 1),
        np.asarray([float(value) for value in tvec], dtype=np.float64).reshape(3, 1),
        k,
        dist,
    )
    return projected.reshape(-1, 2)


def _image_points_to_world_plane_with_camera_params(
    cv2: Any,
    np: Any,
    image_points_px: Any,
    params: Sequence[float],
    *,
    cx: float,
    cy: float,
) -> Any:
    values = np.asarray(params, dtype=np.float64)
    focal = float(math.exp(float(values[0])))
    rvec = values[1:4].reshape(3, 1)
    tvec = values[4:7].reshape(3, 1)
    k1 = float(values[7])
    k2 = float(values[8])
    k = np.asarray([[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    dist = np.asarray([k1, k2, 0.0, 0.0], dtype=np.float64)
    normalized = cv2.undistortPoints(np.asarray(image_points_px, dtype=np.float64).reshape(-1, 1, 2), k, dist)
    rays = np.concatenate(
        [normalized.reshape(-1, 2), np.ones((normalized.shape[0], 1), dtype=np.float64)],
        axis=1,
    )
    rotation, _ = cv2.Rodrigues(rvec)
    rotation_inv = rotation.T
    camera_origin_term = rotation_inv @ tvec.reshape(3)
    points: list[Any] = []
    for ray in rays:
        world_direction = rotation_inv @ ray
        if abs(float(world_direction[2])) <= 1e-9:
            points.append([float("nan"), float("nan"), float("nan")])
            continue
        scale = float(camera_origin_term[2]) / float(world_direction[2])
        world = scale * world_direction - camera_origin_term
        points.append(world)
    return np.asarray(points, dtype=np.float64)


def _image_points_to_world_plane_with_full_camera_params(
    cv2: Any,
    np: Any,
    image_points_px: Any,
    params: Sequence[float],
) -> Any:
    values = np.asarray(params, dtype=np.float64)
    fx = float(math.exp(float(values[0])))
    fy = float(math.exp(float(values[1])))
    cx = float(values[2])
    cy = float(values[3])
    rvec = values[4:7].reshape(3, 1)
    tvec = values[7:10].reshape(3, 1)
    k1 = float(values[10])
    k2 = float(values[11])
    k = np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    dist = np.asarray([k1, k2, 0.0, 0.0], dtype=np.float64)
    normalized = cv2.undistortPoints(np.asarray(image_points_px, dtype=np.float64).reshape(-1, 1, 2), k, dist)
    rays = np.concatenate(
        [normalized.reshape(-1, 2), np.ones((normalized.shape[0], 1), dtype=np.float64)],
        axis=1,
    )
    rotation, _ = cv2.Rodrigues(rvec)
    rotation_inv = rotation.T
    camera_origin_term = rotation_inv @ tvec.reshape(3)
    points: list[Any] = []
    for ray in rays:
        world_direction = rotation_inv @ ray
        if abs(float(world_direction[2])) <= 1e-9:
            points.append([float("nan"), float("nan"), float("nan")])
            continue
        scale = float(camera_origin_term[2]) / float(world_direction[2])
        world = scale * world_direction - camera_origin_term
        points.append(world)
    return np.asarray(points, dtype=np.float64)


def _soft_l1_residual_transform(values: Any, *, scale: float) -> Any:
    np = _np()
    raw = np.asarray(values, dtype=np.float64)
    if scale <= 0.0:
        raise ValueError("scale must be positive")
    squared_loss = 2.0 * scale * scale * (np.sqrt(1.0 + (raw / scale) ** 2) - 1.0)
    return np.sign(raw) * np.sqrt(np.maximum(squared_loss, 0.0))


def _camera_fit_payload(
    params: Sequence[float],
    *,
    cx: float,
    cy: float,
    method: str,
    success: bool,
    status: int,
    message: str,
    initial_residuals: Sequence[float],
    optimized_residuals: Sequence[float],
    point_count: int,
    line_observation_count: int,
) -> dict[str, Any]:
    values = [float(value) for value in params]
    focal = float(math.exp(values[0]))
    initial_distances = _paired_residual_distances(initial_residuals[: point_count * 2])
    optimized_distances = _paired_residual_distances(optimized_residuals[: point_count * 2])
    return {
        "method": method,
        "success": bool(success),
        "status": int(status),
        "message": message,
        "point_count": int(point_count),
        "line_observation_count": int(line_observation_count),
        "intrinsics": {
            "fx": round(focal, 6),
            "fy": round(focal, 6),
            "cx": round(float(cx), 6),
            "cy": round(float(cy), 6),
        },
        "distortion": {
            "k1": round(values[7], 9),
            "k2": round(values[8], 9),
        },
        "extrinsics": {
            "rvec": [round(value, 9) for value in values[1:4]],
            "tvec": [round(value, 9) for value in values[4:7]],
        },
        "initial_reprojection_rmse_px": round(_rmse(initial_distances), 6),
        "optimized_reprojection_rmse_px": round(_rmse(optimized_distances), 6),
        "optimized_reprojection_median_px": round(_median(optimized_distances), 6),
        "optimized_reprojection_p95_px": round(_percentile(optimized_distances, 95), 6),
        "total_initial_residual_rmse": round(_rmse([float(value) for value in initial_residuals]), 6),
        "total_optimized_residual_rmse": round(_rmse([float(value) for value in optimized_residuals]), 6),
    }


def _full_camera_fit_payload(
    params: Sequence[float],
    *,
    method: str,
    success: bool,
    status: int,
    message: str,
    initial_residuals: Sequence[float],
    optimized_residuals: Sequence[float],
    point_count: int,
) -> dict[str, Any]:
    values = [float(value) for value in params]
    fx = float(math.exp(values[0]))
    fy = float(math.exp(values[1]))
    initial_distances = _paired_residual_distances(initial_residuals[: point_count * 2])
    optimized_distances = _paired_residual_distances(optimized_residuals[: point_count * 2])
    return {
        "method": method,
        "success": bool(success),
        "status": int(status),
        "message": message,
        "point_count": int(point_count),
        "line_observation_count": 0,
        "intrinsics": {
            "fx": round(fx, 6),
            "fy": round(fy, 6),
            "cx": round(values[2], 6),
            "cy": round(values[3], 6),
        },
        "distortion": {
            "k1": round(values[10], 9),
            "k2": round(values[11], 9),
        },
        "extrinsics": {
            "rvec": [round(value, 9) for value in values[4:7]],
            "tvec": [round(value, 9) for value in values[7:10]],
        },
        "initial_reprojection_rmse_px": round(_rmse(initial_distances), 6),
        "optimized_reprojection_rmse_px": round(_rmse(optimized_distances), 6),
        "optimized_reprojection_median_px": round(_median(optimized_distances), 6),
        "optimized_reprojection_p95_px": round(_percentile(optimized_distances, 95), 6),
        "total_initial_residual_rmse": round(_rmse([float(value) for value in initial_residuals]), 6),
        "total_optimized_residual_rmse": round(_rmse([float(value) for value in optimized_residuals]), 6),
    }


def _camera_params_from_fit(fit: Mapping[str, Any]) -> Any:
    np = _np()
    intrinsics = fit.get("intrinsics") if isinstance(fit.get("intrinsics"), Mapping) else {}
    distortion = fit.get("distortion") if isinstance(fit.get("distortion"), Mapping) else {}
    extrinsics = fit.get("extrinsics") if isinstance(fit.get("extrinsics"), Mapping) else {}
    rvec = extrinsics.get("rvec")
    tvec = extrinsics.get("tvec")
    if not isinstance(rvec, Sequence) or len(rvec) != 3 or not isinstance(tvec, Sequence) or len(tvec) != 3:
        raise ValueError("fit payload is missing extrinsics.rvec/tvec")
    focal = _finite_float(intrinsics.get("fx"), "fit.intrinsics.fx")
    return np.asarray(
        [
            math.log(max(1.0, focal)),
            _finite_float(rvec[0], "fit.rvec[0]"),
            _finite_float(rvec[1], "fit.rvec[1]"),
            _finite_float(rvec[2], "fit.rvec[2]"),
            _finite_float(tvec[0], "fit.tvec[0]"),
            _finite_float(tvec[1], "fit.tvec[1]"),
            _finite_float(tvec[2], "fit.tvec[2]"),
            _finite_float(distortion.get("k1", 0.0), "fit.distortion.k1"),
            _finite_float(distortion.get("k2", 0.0), "fit.distortion.k2"),
        ],
        dtype=np.float64,
    )


def _line_observation_arrays(observation: Mapping[str, Any]) -> tuple[Any, Any]:
    np = _np()
    world_line = observation.get("world_line_m")
    image_segment = observation.get("image_segment_px")
    if not isinstance(world_line, Sequence) or len(world_line) != 2:
        raise ValueError("line observation must include two world_line_m endpoints")
    if not isinstance(image_segment, Sequence) or len(image_segment) != 2:
        raise ValueError("line observation must include two image_segment_px endpoints")
    world = _object_points3(world_line)  # type: ignore[arg-type]
    sampled_points = observation.get("sampled_image_points_px")
    if isinstance(sampled_points, Sequence) and not isinstance(sampled_points, (str, bytes)) and len(sampled_points) >= 2:
        image = np.asarray(
            [_point2(point, "sampled_image_points_px") for point in sampled_points],
            dtype=np.float64,
        )
    else:
        image = np.asarray([_point2(point, "image_segment_px") for point in image_segment], dtype=np.float64)
    return world, image


def _line_intersection_from_observations(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
) -> tuple[float, float] | None:
    first_segment = first.get("image_segment_px")
    second_segment = second.get("image_segment_px")
    if not isinstance(first_segment, Sequence) or not isinstance(second_segment, Sequence):
        return None
    if len(first_segment) != 2 or len(second_segment) != 2:
        return None
    first_p1 = _point2(first_segment[0], "first.image_segment_px[0]")
    first_p2 = _point2(first_segment[1], "first.image_segment_px[1]")
    second_p1 = _point2(second_segment[0], "second.image_segment_px[0]")
    second_p2 = _point2(second_segment[1], "second.image_segment_px[1]")
    try:
        first_line = _line_from_segment((first_p1, first_p2))
        second_line = _line_from_segment((second_p1, second_p2))
    except ValueError:
        return None
    a1, b1, c1 = first_line
    a2, b2, c2 = second_line
    determinant = a1 * b2 - a2 * b1
    if abs(determinant) <= 1e-9:
        return None
    x = (b1 * c2 - b2 * c1) / determinant
    y = (c1 * a2 - c2 * a1) / determinant
    if not math.isfinite(x) or not math.isfinite(y):
        return None
    return (float(x), float(y))


def _line_intersection_diagnostic_for_keypoint(
    name: str,
    reviewed_px: tuple[float, float],
    model_px: tuple[float, float],
    observations_by_name: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    line_names = FLOOR_KEYPOINT_LINE_INTERSECTIONS.get(name)
    if line_names is None:
        return {
            "expected_line_names": [],
            "line_names": [],
            "line_intersection_status": "no_expected_line_mapping",
            "line_intersection_available": False,
        }

    expected = [str(line_name) for line_name in line_names]
    diagnostic: dict[str, Any] = {
        "expected_line_names": expected,
        "line_names": expected,
        "line_intersection_available": False,
    }
    missing = [line_name for line_name in expected if line_name not in observations_by_name]
    if len(missing) == len(expected):
        diagnostic.update(
            {
                "line_intersection_status": "missing_both_expected_line_observations",
                "missing_line_names": missing,
            }
        )
        return diagnostic
    if missing:
        diagnostic.update(
            {
                "line_intersection_status": f"missing_expected_line_observation:{missing[0]}",
                "missing_line_names": missing,
            }
        )
        return diagnostic

    first_observation = observations_by_name[expected[0]]
    second_observation = observations_by_name[expected[1]]
    quality_payloads = [
        payload
        for payload in (
            _line_observation_quality_payload(first_observation),
            _line_observation_quality_payload(second_observation),
        )
        if payload is not None
    ]
    if len(quality_payloads) == 2:
        diagnostic.update(_line_intersection_quality_summary(quality_payloads))
    intersection = _line_intersection_from_observations(first_observation, second_observation)
    if intersection is None:
        diagnostic["line_intersection_status"] = "degenerate_expected_line_observations"
        diagnostic["line_support_modes"] = [
            str(first_observation.get("support_mode", "unknown")),
            str(second_observation.get("support_mode", "unknown")),
        ]
        return diagnostic

    diagnostic.update(
        {
            "line_intersection_status": "available",
            "line_support_modes": [
                str(first_observation.get("support_mode", "unknown")),
                str(second_observation.get("support_mode", "unknown")),
            ],
            "line_intersection_available": True,
            "line_intersection_image_px": [round(intersection[0], 3), round(intersection[1], 3)],
            "line_intersection_delta_px": round(math.dist(reviewed_px, intersection), 3),
            "model_to_line_intersection_delta_px": round(math.dist(model_px, intersection), 3),
        }
    )
    return diagnostic


def _line_observation_quality_payload(observation: Mapping[str, Any]) -> dict[str, float] | None:
    quality = observation.get("quality")
    if isinstance(quality, Mapping):
        source = quality
    else:
        source = observation
    try:
        return {
            "angle_diff_deg": round(_finite_float(source.get("angle_diff_deg"), "line_quality.angle_diff_deg"), 3),
            "mean_perpendicular_distance_px": round(
                _finite_float(
                    source.get("mean_perpendicular_distance_px"),
                    "line_quality.mean_perpendicular_distance_px",
                ),
                3,
            ),
            "overlap_fraction": round(_finite_float(source.get("overlap_fraction"), "line_quality.overlap_fraction"), 6),
        }
    except ValueError:
        return None


def _line_intersection_quality_summary(quality_payloads: Sequence[Mapping[str, float]]) -> dict[str, Any]:
    if len(quality_payloads) != 2:
        return {}
    angles = [float(payload["angle_diff_deg"]) for payload in quality_payloads]
    distances = [float(payload["mean_perpendicular_distance_px"]) for payload in quality_payloads]
    overlaps = [float(payload["overlap_fraction"]) for payload in quality_payloads]
    return {
        "line_quality": [dict(payload) for payload in quality_payloads],
        "line_quality_max_angle_diff_deg": round(max(angles), 3),
        "line_quality_max_mean_perpendicular_distance_px": round(max(distances), 3),
        "line_quality_min_overlap_fraction": round(min(overlaps), 6),
    }


def _line_intersection_quality_gate_decision(
    diagnostic: Mapping[str, Any],
    quality_profile: Mapping[str, Any] | None,
) -> str:
    if quality_profile is None:
        return "passed"
    try:
        max_angle = _finite_float(diagnostic.get("line_quality_max_angle_diff_deg"), "line_quality_max_angle_diff_deg")
        max_distance = _finite_float(
            diagnostic.get("line_quality_max_mean_perpendicular_distance_px"),
            "line_quality_max_mean_perpendicular_distance_px",
        )
        min_overlap = _finite_float(
            diagnostic.get("line_quality_min_overlap_fraction"),
            "line_quality_min_overlap_fraction",
        )
        threshold_angle = _finite_float(quality_profile.get("max_angle_diff_deg"), "quality_profile.max_angle_diff_deg")
        threshold_distance = _finite_float(
            quality_profile.get("max_mean_perpendicular_distance_px"),
            "quality_profile.max_mean_perpendicular_distance_px",
        )
        threshold_overlap = _finite_float(
            quality_profile.get("min_overlap_fraction"),
            "quality_profile.min_overlap_fraction",
        )
    except ValueError:
        return "missing_quality"
    if max_angle > threshold_angle or max_distance > threshold_distance or min_overlap < threshold_overlap:
        return "failed"
    if quality_profile.get("max_model_to_line_intersection_delta_px") is not None:
        try:
            model_delta = _finite_float(
                diagnostic.get("model_to_line_intersection_delta_px"),
                "model_to_line_intersection_delta_px",
            )
            threshold_model_delta = _finite_float(
                quality_profile.get("max_model_to_line_intersection_delta_px"),
                "quality_profile.max_model_to_line_intersection_delta_px",
            )
        except ValueError:
            return "missing_quality"
        if model_delta > threshold_model_delta:
            return "failed"
    return "passed"


def _world_plane_residual_summary_ft(
    object_points_m: Sequence[Sequence[float]],
    image_points_px: Sequence[Sequence[float]],
    fit: Mapping[str, Any],
    *,
    meters_to_feet: float,
) -> dict[str, float]:
    np = _np()
    target = _object_points3(object_points_m)
    predicted = np.asarray(image_points_to_world_plane_with_distortion_fit(image_points_px, fit), dtype=np.float64)
    if predicted.shape[0] != target.shape[0]:
        raise ValueError("backprojected point count does not match object point count")
    residuals_ft = [
        float(np.linalg.norm(predicted[index, :2] - target[index, :2]) * float(meters_to_feet))
        for index in range(target.shape[0])
    ]
    return {
        "mean_residual_ft": round(_mean(residuals_ft), 6),
        "median_residual_ft": round(_median(residuals_ft), 6),
        "p95_residual_ft": round(_percentile(residuals_ft, 95), 6),
    }


def _world_plane_residual_details_ft(
    keypoint_names: Sequence[str],
    object_points_m: Sequence[Sequence[float]],
    image_points_px: Sequence[Sequence[float]],
    fit: Mapping[str, Any],
    *,
    meters_to_feet: float,
) -> dict[str, Any]:
    np = _np()
    names = [str(name) for name in keypoint_names]
    target = _object_points3(object_points_m)
    predicted = np.asarray(image_points_to_world_plane_with_distortion_fit(image_points_px, fit), dtype=np.float64)
    if predicted.shape[0] != target.shape[0] or len(names) != target.shape[0]:
        raise ValueError("keypoint, object point, and backprojected point counts must match")
    residuals = [
        float(np.linalg.norm(predicted[index, :2] - target[index, :2]) * float(meters_to_feet))
        for index in range(target.shape[0])
    ]
    per_keypoint = {name: round(residual, 6) for name, residual in zip(names, residuals, strict=True)}
    worst_index = max(range(len(residuals)), key=lambda index: residuals[index])
    payload: dict[str, Any] = {
        "per_keypoint_residual_ft": per_keypoint,
        "worst_keypoint": {
            "name": names[worst_index],
            "residual_ft": round(residuals[worst_index], 6),
        },
    }
    for drop_count in (1, 2, 3):
        payload[f"trimmed_mean_residual_ft_drop_worst_{drop_count}"] = _trimmed_mean_or_none(
            residuals,
            drop_worst_count=drop_count,
        )
    return payload


def _metric_plane_outlier_review_candidates(
    keypoint_names: Sequence[str],
    object_points_m: Sequence[Sequence[float]],
    image_points_px: Sequence[Sequence[float]],
    fit: Mapping[str, Any],
    residual_details_ft: Mapping[str, Any],
    *,
    line_observations: Sequence[Mapping[str, Any]],
    residual_threshold_ft: float = 0.45,
    max_candidates: int = 8,
) -> list[dict[str, Any]]:
    if residual_threshold_ft <= 0.0:
        raise ValueError("residual_threshold_ft must be positive")
    if max_candidates <= 0:
        raise ValueError("max_candidates must be positive")
    per_keypoint = residual_details_ft.get("per_keypoint_residual_ft")
    if not isinstance(per_keypoint, Mapping):
        return []
    names = [str(name) for name in keypoint_names]
    projected = project_world_points_with_distortion_fit(object_points_m, fit)
    observations_by_name = {str(observation.get("name")): observation for observation in line_observations}
    candidates: list[dict[str, Any]] = []
    for name, image_point, model_point in zip(names, image_points_px, projected, strict=True):
        residual = per_keypoint.get(name)
        if residual is None or float(residual) < float(residual_threshold_ft):
            continue
        reviewed_px = _point2(image_point, f"{name}.reviewed_image_px")
        model_px = _point2(model_point, f"{name}.model_projected_image_px")
        candidate: dict[str, Any] = {
            "diagnostic_only": True,
            "keypoint": name,
            "residual_ft": round(float(residual), 6),
            "reviewed_image_px": [round(reviewed_px[0], 3), round(reviewed_px[1], 3)],
            "model_projected_image_px": [round(model_px[0], 3), round(model_px[1], 3)],
            "model_delta_px": round(math.dist(reviewed_px, model_px), 3),
        }
        candidate.update(
            _line_intersection_diagnostic_for_keypoint(name, reviewed_px, model_px, observations_by_name)
        )
        candidates.append(candidate)
    candidates.sort(key=lambda item: float(item["residual_ft"]), reverse=True)
    return candidates[:max_candidates]


def _metric_plane_top_residual_refit_diagnostic(
    keypoint_names: Sequence[str],
    object_points_m: Sequence[Sequence[float]],
    image_points_px: Sequence[Sequence[float]],
    *,
    image_size: tuple[float, float],
    residual_details_ft: Mapping[str, Any],
    line_observations: Sequence[Mapping[str, Any]] = (),
    meters_to_feet: float,
    drop_count: int,
) -> dict[str, Any]:
    names = [str(name) for name in keypoint_names]
    if drop_count <= 0:
        raise ValueError("drop_count must be positive")
    if len(names) <= drop_count:
        return {
            "diagnostic_only": True,
            "promotes_calibration": False,
            "status": "skipped",
            "reason": "drop_count_exhausts_keypoints",
            "drop_count": int(drop_count),
        }
    per_keypoint = residual_details_ft.get("per_keypoint_residual_ft")
    if not isinstance(per_keypoint, Mapping):
        return {
            "diagnostic_only": True,
            "promotes_calibration": False,
            "status": "skipped",
            "reason": "missing_per_keypoint_residuals",
            "drop_count": int(drop_count),
        }
    name_order = {name: index for index, name in enumerate(names)}
    residual_items: list[tuple[str, float]] = []
    for name in names:
        residual = per_keypoint.get(name)
        if residual is None:
            continue
        residual_items.append((name, float(residual)))
    if len(residual_items) <= drop_count:
        return {
            "diagnostic_only": True,
            "promotes_calibration": False,
            "status": "skipped",
            "reason": "too_few_scored_keypoints",
            "drop_count": int(drop_count),
            "scored_keypoint_count": len(residual_items),
        }
    residual_items.sort(key=lambda item: (-item[1], name_order[item[0]]))
    dropped = residual_items[:drop_count]
    dropped_names = {name for name, _residual in dropped}
    keep_indexes = [index for index, name in enumerate(names) if name not in dropped_names]
    if len(keep_indexes) < 6:
        return {
            "diagnostic_only": True,
            "promotes_calibration": False,
            "status": "skipped",
            "reason": "too_few_inlier_keypoints_for_camera_fit",
            "drop_count": int(drop_count),
            "inlier_keypoint_count": len(keep_indexes),
        }
    inlier_object_points = [object_points_m[index] for index in keep_indexes]
    inlier_image_points = [image_points_px[index] for index in keep_indexes]
    fit = fit_metric_plane_camera_lm(
        inlier_object_points,
        inlier_image_points,
        image_size=image_size,
    )
    inlier_residual_ft = _world_plane_residual_summary_ft(
        inlier_object_points,
        inlier_image_points,
        fit,
        meters_to_feet=meters_to_feet,
    )
    all_label_residual_ft = _world_plane_residual_summary_ft(
        object_points_m,
        image_points_px,
        fit,
        meters_to_feet=meters_to_feet,
    )
    refit_projected = project_world_points_with_distortion_fit(object_points_m, fit)
    observations_by_name = {str(observation.get("name")): observation for observation in line_observations}
    dropped_details: list[dict[str, Any]] = []
    for name, residual in dropped:
        index = name_order[name]
        reviewed_px = _point2(image_points_px[index], f"{name}.reviewed_image_px")
        projected_px = _point2(refit_projected[index], f"{name}.refit_projected_image_px")
        detail = {
            "keypoint": name,
            "source_residual_ft": round(residual, 6),
            "reviewed_image_px": [round(reviewed_px[0], 3), round(reviewed_px[1], 3)],
            "refit_projected_image_px": [round(projected_px[0], 3), round(projected_px[1], 3)],
            "refit_model_delta_px": round(math.dist(reviewed_px, projected_px), 3),
        }
        detail.update(_line_intersection_diagnostic_for_keypoint(name, reviewed_px, projected_px, observations_by_name))
        dropped_details.append(detail)
    return {
        "diagnostic_only": True,
        "promotes_calibration": False,
        "status": "scored",
        "selection_mode": "drop_largest_full_label_metric_plane_residuals",
        "method": fit["method"],
        "drop_count": int(drop_count),
        "inlier_keypoint_count": len(keep_indexes),
        "dropped_keypoints": [name for name, _residual in dropped],
        "dropped_keypoint_residual_ft": {name: round(residual, 6) for name, residual in dropped},
        "dropped_keypoint_details": dropped_details,
        "inlier_mean_residual_ft": inlier_residual_ft["mean_residual_ft"],
        "inlier_median_residual_ft": inlier_residual_ft["median_residual_ft"],
        "inlier_p95_residual_ft": inlier_residual_ft["p95_residual_ft"],
        "all_label_mean_residual_ft": all_label_residual_ft["mean_residual_ft"],
        "all_label_median_residual_ft": all_label_residual_ft["median_residual_ft"],
        "all_label_p95_residual_ft": all_label_residual_ft["p95_residual_ft"],
        "notes": [
            "This diagnostic selects outliers using reviewed-label residuals, so it is not a production calibration source.",
            "The inlier-only residual quantifies whether a small label/evidence subset blocks the 0.2 ft target.",
            "The all-label residual remains the non-promoted check against the original reviewed labels.",
        ],
    }


def _metric_plane_top_residual_refit_progression(
    keypoint_names: Sequence[str],
    object_points_m: Sequence[Sequence[float]],
    image_points_px: Sequence[Sequence[float]],
    *,
    image_size: tuple[float, float],
    residual_details_ft: Mapping[str, Any],
    line_observations: Sequence[Mapping[str, Any]] = (),
    meters_to_feet: float,
    max_drop_count: int,
) -> list[dict[str, Any]]:
    if max_drop_count < 0:
        raise ValueError("max_drop_count must be non-negative")
    names = [str(name) for name in keypoint_names]
    per_keypoint = residual_details_ft.get("per_keypoint_residual_ft")
    if not isinstance(per_keypoint, Mapping):
        return []
    name_order = {name: index for index, name in enumerate(names)}
    residual_items: list[tuple[str, float]] = []
    for name in names:
        residual = per_keypoint.get(name)
        if residual is None:
            continue
        residual_items.append((name, float(residual)))
    if not residual_items:
        return []
    residual_items.sort(key=lambda item: (-item[1], name_order[item[0]]))
    observations_by_name = {str(observation.get("name")): observation for observation in line_observations}

    progression: list[dict[str, Any]] = []
    for drop_count in range(max_drop_count + 1):
        if len(names) <= drop_count:
            break
        dropped = residual_items[:drop_count]
        dropped_names = {name for name, _residual in dropped}
        keep_indexes = [index for index, name in enumerate(names) if name not in dropped_names]
        if len(keep_indexes) < 6:
            break
        inlier_object_points = [object_points_m[index] for index in keep_indexes]
        inlier_image_points = [image_points_px[index] for index in keep_indexes]
        fit = fit_metric_plane_camera_lm(
            inlier_object_points,
            inlier_image_points,
            image_size=image_size,
        )
        inlier_residual_ft = _world_plane_residual_summary_ft(
            inlier_object_points,
            inlier_image_points,
            fit,
            meters_to_feet=meters_to_feet,
        )
        all_label_residual_ft = _world_plane_residual_summary_ft(
            object_points_m,
            image_points_px,
            fit,
            meters_to_feet=meters_to_feet,
        )
        refit_projected = project_world_points_with_distortion_fit(object_points_m, fit)
        dropped_details: list[dict[str, Any]] = []
        for name, residual in dropped:
            index = name_order[name]
            reviewed_px = _point2(image_points_px[index], f"{name}.reviewed_image_px")
            projected_px = _point2(refit_projected[index], f"{name}.refit_projected_image_px")
            detail = {
                "keypoint": name,
                "source_residual_ft": round(residual, 6),
                "reviewed_image_px": [round(reviewed_px[0], 3), round(reviewed_px[1], 3)],
                "refit_projected_image_px": [round(projected_px[0], 3), round(projected_px[1], 3)],
                "refit_model_delta_px": round(math.dist(reviewed_px, projected_px), 3),
            }
            detail.update(_line_intersection_diagnostic_for_keypoint(name, reviewed_px, projected_px, observations_by_name))
            dropped_details.append(detail)
        progression.append(
            {
                "diagnostic_only": True,
                "promotes_calibration": False,
                "status": "scored",
                "selection_mode": "drop_largest_full_label_metric_plane_residuals",
                "method": fit["method"],
                "drop_count": int(drop_count),
                "inlier_keypoint_count": len(keep_indexes),
                "dropped_keypoints": [name for name, _residual in dropped],
                "dropped_keypoint_residual_ft": {name: round(residual, 6) for name, residual in dropped},
                "dropped_keypoint_details": dropped_details,
                "inlier_mean_residual_ft": inlier_residual_ft["mean_residual_ft"],
                "inlier_median_residual_ft": inlier_residual_ft["median_residual_ft"],
                "inlier_p95_residual_ft": inlier_residual_ft["p95_residual_ft"],
                "all_label_mean_residual_ft": all_label_residual_ft["mean_residual_ft"],
                "all_label_median_residual_ft": all_label_residual_ft["median_residual_ft"],
                "all_label_p95_residual_ft": all_label_residual_ft["p95_residual_ft"],
            }
        )
    return progression


def _top_residual_refit_progression_summary(
    results: Sequence[Mapping[str, Any]],
    *,
    target_mean_residual_ft: float,
) -> dict[str, Any]:
    by_drop: dict[int, list[dict[str, Any]]] = {}
    for result in results:
        metric_plane = result.get("metric_plane_camera")
        if not isinstance(metric_plane, Mapping):
            continue
        progression = metric_plane.get("top_residual_refit_progression")
        if not isinstance(progression, Sequence) or isinstance(progression, (str, bytes)):
            continue
        for raw_item in progression:
            if not isinstance(raw_item, Mapping) or raw_item.get("status") != "scored":
                continue
            drop_count = raw_item.get("drop_count")
            if isinstance(drop_count, bool) or not isinstance(drop_count, int):
                continue
            by_drop.setdefault(int(drop_count), []).append(dict(raw_item))

    min_mean_drop: int | None = None
    min_all_clips_drop: int | None = None
    full_clip_count = len(results)
    for drop_count in sorted(by_drop):
        values = [
            float(item["inlier_mean_residual_ft"])
            for item in by_drop[drop_count]
            if item.get("inlier_mean_residual_ft") is not None
        ]
        if len(values) != full_clip_count or not values:
            continue
        if min_mean_drop is None and _mean(values) <= float(target_mean_residual_ft):
            min_mean_drop = int(drop_count)
        if min_all_clips_drop is None and max(values) <= float(target_mean_residual_ft):
            min_all_clips_drop = int(drop_count)

    drop4_values = [
        float(item["inlier_mean_residual_ft"])
        for item in by_drop.get(4, [])
        if item.get("inlier_mean_residual_ft") is not None
    ]
    drop4_all_label_values = [
        float(item["all_label_mean_residual_ft"])
        for item in by_drop.get(4, [])
        if item.get("all_label_mean_residual_ft") is not None
    ]
    drop5_values = [
        float(item["inlier_mean_residual_ft"])
        for item in by_drop.get(5, [])
        if item.get("inlier_mean_residual_ft") is not None
    ]
    drop5_all_label_values = [
        float(item["all_label_mean_residual_ft"])
        for item in by_drop.get(5, [])
        if item.get("all_label_mean_residual_ft") is not None
    ]
    drop4_line_status_counts = _source_counts(
        str(detail.get("line_intersection_status", "unknown"))
        for item in by_drop.get(4, [])
        if isinstance(item.get("dropped_keypoint_details"), Sequence)
        and not isinstance(item.get("dropped_keypoint_details"), (str, bytes))
        for detail in item["dropped_keypoint_details"]
        if isinstance(detail, Mapping)
    )
    drop5_line_status_counts = _source_counts(
        str(detail.get("line_intersection_status", "unknown"))
        for item in by_drop.get(5, [])
        if isinstance(item.get("dropped_keypoint_details"), Sequence)
        and not isinstance(item.get("dropped_keypoint_details"), (str, bytes))
        for detail in item["dropped_keypoint_details"]
        if isinstance(detail, Mapping)
    )
    return {
        "min_drop_count_for_mean_target": min_mean_drop,
        "min_drop_count_for_all_clips_target": min_all_clips_drop,
        "drop4_inlier_mean_residual_ft_mean": None if not drop4_values else round(_mean(drop4_values), 6),
        "drop4_inlier_mean_residual_ft_max": None if not drop4_values else round(max(drop4_values), 6),
        "drop4_all_label_mean_residual_ft_mean": (
            None if not drop4_all_label_values else round(_mean(drop4_all_label_values), 6)
        ),
        "drop4_line_intersection_status_counts": drop4_line_status_counts,
        "drop5_inlier_mean_residual_ft_mean": None if not drop5_values else round(_mean(drop5_values), 6),
        "drop5_inlier_mean_residual_ft_max": None if not drop5_values else round(max(drop5_values), 6),
        "drop5_all_label_mean_residual_ft_mean": (
            None if not drop5_all_label_values else round(_mean(drop5_all_label_values), 6)
        ),
        "drop5_line_intersection_status_counts": drop5_line_status_counts,
    }


def _all_strict_line_intersection_override_oracle(
    keypoint_names: Sequence[str],
    object_points_m: Sequence[Sequence[float]],
    image_points_px: Sequence[Sequence[float]],
    *,
    image_size: tuple[float, float],
    line_observations: Sequence[Mapping[str, Any]],
    model_fit: Mapping[str, Any],
    baseline_mean_residual_ft: float,
    meters_to_feet: float,
    camera_fit_model: str = "metric_plane",
    quality_profile: Mapping[str, Any] | None = None,
    source_strategy: str = "all_strict_endpoint_line_intersections",
    selection_mode: str = "all_canonical_endpoint_intersections_without_residual_ranking",
    line_observation_source: str = "review_line_observations",
    line_reference_source: str | None = None,
    uses_reviewed_line_positions_for_matching: bool = True,
) -> dict[str, Any]:
    names = [str(name) for name in keypoint_names]
    fit_model = str(camera_fit_model)
    if fit_model not in {"metric_plane", "full_intrinsics_metric_plane"}:
        raise ValueError(f"unsupported all-strict override camera_fit_model: {fit_model}")

    observations_by_name = {str(observation.get("name")): observation for observation in line_observations}
    projected = project_world_points_with_distortion_fit(object_points_m, model_fit)
    override_points = [list(_point2(point, "all_strict_line_override.image_point")) for point in image_points_px]
    overrides: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    skipped_missing = 0
    skipped_non_strict = 0
    skipped_center_keypoints = 0
    skipped_quality_gate = 0
    missing_quality = 0

    for index, name in enumerate(names):
        reviewed_px = _point2(image_points_px[index], f"{name}.reviewed_image_px")
        model_px = _point2(projected[index], f"{name}.model_projected_image_px")
        diagnostic = _line_intersection_diagnostic_for_keypoint(
            name,
            reviewed_px,
            model_px,
            observations_by_name,
        )
        diagnostic["keypoint"] = name
        diagnostics.append(diagnostic)
        is_center_keypoint = name in LINE_INTERSECTION_OVERRIDE_CENTER_KEYPOINTS
        if is_center_keypoint:
            skipped_center_keypoints += 1
            continue
        if diagnostic.get("line_intersection_available") is not True:
            skipped_missing += 1
            continue
        support_modes_raw = diagnostic.get("line_support_modes")
        support_modes = (
            [str(value) for value in support_modes_raw]
            if isinstance(support_modes_raw, Sequence) and not isinstance(support_modes_raw, (str, bytes))
            else []
        )
        has_strict_support = bool(support_modes) and all(mode == "overlapping_segment" for mode in support_modes)
        if not has_strict_support:
            skipped_non_strict += 1
            continue
        quality_gate = _line_intersection_quality_gate_decision(diagnostic, quality_profile)
        if quality_gate == "missing_quality":
            missing_quality += 1
            skipped_quality_gate += 1
            continue
        if quality_gate == "failed":
            skipped_quality_gate += 1
            continue
        line_point = _point2(diagnostic.get("line_intersection_image_px"), "line_intersection_image_px")
        override_points[index] = line_point
        overrides.append(
            {
                "keypoint": name,
                "reviewed_image_px": [round(reviewed_px[0], 3), round(reviewed_px[1], 3)],
                "line_intersection_image_px": [round(line_point[0], 3), round(line_point[1], 3)],
                "line_intersection_support": _line_intersection_support(diagnostic),
                "line_support_modes": support_modes,
                "line_intersection_delta_px": diagnostic.get("line_intersection_delta_px"),
                "model_to_line_intersection_delta_px": diagnostic.get("model_to_line_intersection_delta_px"),
                "line_quality_max_angle_diff_deg": diagnostic.get("line_quality_max_angle_diff_deg"),
                "line_quality_max_mean_perpendicular_distance_px": diagnostic.get(
                    "line_quality_max_mean_perpendicular_distance_px"
                ),
                "line_quality_min_overlap_fraction": diagnostic.get("line_quality_min_overlap_fraction"),
            }
        )

    base_payload = {
        "diagnostic_only": True,
        "promotes_calibration": False,
        "mutates_reviewed_labels": False,
        "source": str(line_observation_source),
        "source_strategy": str(source_strategy),
        "selection_mode": str(selection_mode),
        "uses_residual_rank_selection": False,
        "uses_reviewed_line_positions_for_matching": bool(uses_reviewed_line_positions_for_matching),
        "line_reference_source": line_reference_source,
        "camera_fit_model": fit_model,
        "quality_profile": dict(quality_profile) if quality_profile is not None else None,
        "quality_profile_id": (
            str(quality_profile.get("profile_id"))
            if isinstance(quality_profile, Mapping) and quality_profile.get("profile_id") is not None
            else None
        ),
        "strict_support_required": True,
        "skips_center_keypoints": True,
        "available_line_intersection_count": sum(
            1 for diagnostic in diagnostics if diagnostic.get("line_intersection_available") is True
        ),
        "override_candidate_count": len(overrides),
        "skipped_missing_line_intersection_count": skipped_missing,
        "skipped_non_strict_line_intersection_count": skipped_non_strict,
        "skipped_center_keypoint_count": skipped_center_keypoints,
        "skipped_quality_gate_count": skipped_quality_gate,
        "missing_quality_count": missing_quality,
        "line_intersection_status_counts": _source_counts(
            str(diagnostic.get("line_intersection_status", "unknown")) for diagnostic in diagnostics
        ),
        "line_intersection_support_counts": _source_counts(
            str(item["line_intersection_support"]) for item in overrides
        ),
    }
    if not overrides:
        return base_payload | {
            "status": "skipped",
            "reason": "no_strict_endpoint_line_intersections",
        }

    fit = (
        fit_full_intrinsics_metric_plane_camera_lm(
            object_points_m,
            override_points,
            image_size=image_size,
        )
        if fit_model == "full_intrinsics_metric_plane"
        else fit_metric_plane_camera_lm(
            object_points_m,
            override_points,
            image_size=image_size,
        )
    )
    override_residual_ft = _world_plane_residual_summary_ft(
        object_points_m,
        override_points,
        fit,
        meters_to_feet=meters_to_feet,
    )
    original_reviewed_residual_ft = _world_plane_residual_summary_ft(
        object_points_m,
        image_points_px,
        fit,
        meters_to_feet=meters_to_feet,
    )
    override_details_ft = _world_plane_residual_details_ft(
        names,
        object_points_m,
        override_points,
        fit,
        meters_to_feet=meters_to_feet,
    )
    delta_ft_vs_reviewed_baseline = round(
        float(baseline_mean_residual_ft) - float(override_residual_ft["mean_residual_ft"]),
        6,
    )
    original_reviewed_delta_ft_vs_reviewed_baseline = round(
        float(baseline_mean_residual_ft) - float(original_reviewed_residual_ft["mean_residual_ft"]),
        6,
    )
    return base_payload | {
        "status": "scored",
        "method": fit["method"],
        "mean_residual_ft": override_residual_ft["mean_residual_ft"],
        "median_residual_ft": override_residual_ft["median_residual_ft"],
        "p95_residual_ft": override_residual_ft["p95_residual_ft"],
        "original_reviewed_mean_residual_ft": original_reviewed_residual_ft["mean_residual_ft"],
        "original_reviewed_median_residual_ft": original_reviewed_residual_ft["median_residual_ft"],
        "original_reviewed_p95_residual_ft": original_reviewed_residual_ft["p95_residual_ft"],
        "delta_ft_vs_reviewed_baseline": delta_ft_vs_reviewed_baseline,
        "original_reviewed_delta_ft_vs_reviewed_baseline": original_reviewed_delta_ft_vs_reviewed_baseline,
        f"delta_ft_vs_{fit_model}_reviewed_baseline": delta_ft_vs_reviewed_baseline,
        f"original_reviewed_delta_ft_vs_{fit_model}_reviewed_baseline": (
            original_reviewed_delta_ft_vs_reviewed_baseline
        ),
        "per_keypoint_residual_ft": override_details_ft["per_keypoint_residual_ft"],
        "worst_keypoint": override_details_ft["worst_keypoint"],
        "overrides": overrides,
        "notes": [
            "Scores endpoint line intersections without residual-rank selection.",
            "Line observations are matched using reviewed geometry, so this remains diagnostic-only.",
        ],
    }


def _quality_gated_line_intersection_override_sweep(
    keypoint_names: Sequence[str],
    object_points_m: Sequence[Sequence[float]],
    image_points_px: Sequence[Sequence[float]],
    *,
    image_size: tuple[float, float],
    line_observations: Sequence[Mapping[str, Any]],
    model_fit: Mapping[str, Any],
    baseline_mean_residual_ft: float,
    meters_to_feet: float,
    camera_fit_model: str = "metric_plane",
    profiles: Sequence[Mapping[str, Any]] = LINE_INTERSECTION_QUALITY_GATE_PROFILES,
    source_strategy: str = "quality_gated_endpoint_line_intersections",
    selection_mode: str = "quality_profile_endpoint_intersections_without_residual_ranking",
    line_observation_source: str = "review_line_observations",
    line_reference_source: str | None = "reviewed_keypoints",
    uses_reviewed_line_positions_for_matching: bool = True,
) -> dict[str, Any]:
    profile_results = [
        _all_strict_line_intersection_override_oracle(
            keypoint_names,
            object_points_m,
            image_points_px,
            image_size=image_size,
            line_observations=line_observations,
            model_fit=model_fit,
            baseline_mean_residual_ft=baseline_mean_residual_ft,
            meters_to_feet=meters_to_feet,
            camera_fit_model=camera_fit_model,
            quality_profile=profile,
            source_strategy=source_strategy,
            selection_mode=selection_mode,
            line_observation_source=line_observation_source,
            line_reference_source=line_reference_source,
            uses_reviewed_line_positions_for_matching=uses_reviewed_line_positions_for_matching,
        )
        for profile in profiles
    ]
    scored = [
        result
        for result in profile_results
        if result.get("status") == "scored" and result.get("mean_residual_ft") is not None
    ]
    best = min(scored, key=lambda item: float(item["mean_residual_ft"])) if scored else None
    return {
        "diagnostic_only": True,
        "promotes_calibration": False,
        "mutates_reviewed_labels": False,
        "uses_residual_rank_selection": False,
        "uses_reviewed_line_positions_for_matching": bool(uses_reviewed_line_positions_for_matching),
        "line_reference_source": line_reference_source,
        "source": str(line_observation_source),
        "profile_selection_metric": "lowest_temporary_override_mean_residual_ft",
        "profile_selection_is_diagnostic": True,
        "profile_count": len(profile_results),
        "scored_profile_count": len(scored),
        "best_profile": best,
        "profiles": profile_results,
        "notes": [
            "Profiles are fixed line-quality thresholds, but choosing the best profile by reviewed-label outcome is diagnostic-only.",
            "Line observations are still matched from reviewed geometry in this evaluator.",
        ],
    }


def _quality_gated_line_override_profile_summary(
    sweeps: Sequence[Mapping[str, Any]],
    *,
    full_clip_count: int,
) -> dict[str, Any]:
    profile_rows: dict[str, list[Mapping[str, Any]]] = {}
    for sweep in sweeps:
        raw_profiles = sweep.get("profiles")
        if not isinstance(raw_profiles, Sequence) or isinstance(raw_profiles, (str, bytes)):
            continue
        for profile in raw_profiles:
            if not isinstance(profile, Mapping):
                continue
            profile_id = profile.get("quality_profile_id")
            if profile_id is None:
                continue
            profile_rows.setdefault(str(profile_id), []).append(profile)

    summaries: list[dict[str, Any]] = []
    for profile_id in sorted(profile_rows):
        rows = profile_rows[profile_id]
        scored = [
            row
            for row in rows
            if row.get("status") == "scored" and row.get("mean_residual_ft") is not None
        ]
        means = [float(row["mean_residual_ft"]) for row in scored]
        original_means = [
            float(row["original_reviewed_mean_residual_ft"])
            for row in scored
            if row.get("original_reviewed_mean_residual_ft") is not None
        ]
        candidate_count = sum(int(row.get("override_candidate_count", 0)) for row in rows)
        first_profile = next((row.get("quality_profile") for row in rows if isinstance(row.get("quality_profile"), Mapping)), {})
        summaries.append(
            {
                "profile_id": profile_id,
                "quality_profile": dict(first_profile) if isinstance(first_profile, Mapping) else {},
                "clip_count": len(rows),
                "scored_clip_count": len(scored),
                "complete_clip_coverage": len(scored) == int(full_clip_count),
                "override_candidate_count": candidate_count,
                "mean_residual_ft_mean": None if not means else round(_mean(means), 6),
                "mean_residual_ft_max": None if not means else round(max(means), 6),
                "original_reviewed_mean_residual_ft_mean": (
                    None if not original_means else round(_mean(original_means), 6)
                ),
            }
        )

    scored_summaries = [
        summary
        for summary in summaries
        if summary["complete_clip_coverage"] is True and summary["mean_residual_ft_mean"] is not None
    ]
    best = min(scored_summaries, key=lambda item: float(item["mean_residual_ft_mean"])) if scored_summaries else None
    return {
        "profile_count": len(summaries),
        "best_profile": best,
        "profiles": summaries,
    }


def _metric_plane_line_intersection_override_oracle(
    keypoint_names: Sequence[str],
    object_points_m: Sequence[Sequence[float]],
    image_points_px: Sequence[Sequence[float]],
    *,
    image_size: tuple[float, float],
    candidates: Sequence[Mapping[str, Any]],
    baseline_mean_residual_ft: float,
    meters_to_feet: float,
) -> dict[str, Any]:
    names = [str(name) for name in keypoint_names]
    name_to_index = {name: index for index, name in enumerate(names)}
    override_points = [list(_point2(point, "line_override.image_point")) for point in image_points_px]
    overrides: list[dict[str, Any]] = []
    skipped_non_strict = 0
    skipped_center_keypoints = 0
    for candidate in candidates:
        if candidate.get("line_intersection_available") is not True:
            continue
        name = str(candidate.get("keypoint") or "")
        support_modes_raw = candidate.get("line_support_modes")
        support_modes = (
            [str(value) for value in support_modes_raw]
            if isinstance(support_modes_raw, Sequence) and not isinstance(support_modes_raw, (str, bytes))
            else []
        )
        has_strict_support = bool(support_modes) and all(mode == "overlapping_segment" for mode in support_modes)
        is_center_keypoint = name in LINE_INTERSECTION_OVERRIDE_CENTER_KEYPOINTS
        if not has_strict_support:
            skipped_non_strict += 1
        if is_center_keypoint:
            skipped_center_keypoints += 1
        if is_center_keypoint or not has_strict_support:
            continue
        index = name_to_index.get(name)
        if index is None:
            continue
        try:
            line_point = _point2(candidate.get("line_intersection_image_px"), "line_intersection_image_px")
            reviewed_point = _point2(candidate.get("reviewed_image_px"), "reviewed_image_px")
        except ValueError:
            continue
        override_points[index] = line_point
        overrides.append(
            {
                "keypoint": name,
                "reviewed_image_px": [round(reviewed_point[0], 3), round(reviewed_point[1], 3)],
                "line_intersection_image_px": [round(line_point[0], 3), round(line_point[1], 3)],
                "line_intersection_support": _line_intersection_support(candidate),
                "line_support_modes": support_modes,
                "line_intersection_delta_px": candidate.get("line_intersection_delta_px"),
                "model_to_line_intersection_delta_px": candidate.get("model_to_line_intersection_delta_px"),
                "source_residual_ft": candidate.get("residual_ft"),
            }
        )
    if not overrides:
        return {
            "diagnostic_only": True,
            "mutates_reviewed_labels": False,
            "status": "skipped",
            "reason": "no_line_intersection_outlier_candidates",
            "override_candidate_count": 0,
            "default_strategy": "endpoint_intersections_only",
            "strict_support_required": True,
            "skipped_non_strict_line_intersection_count": skipped_non_strict,
            "skipped_center_keypoint_count": skipped_center_keypoints,
        }

    fit = fit_metric_plane_camera_lm(
        object_points_m,
        override_points,
        image_size=image_size,
    )
    override_residual_ft = _world_plane_residual_summary_ft(
        object_points_m,
        override_points,
        fit,
        meters_to_feet=meters_to_feet,
    )
    original_reviewed_residual_ft = _world_plane_residual_summary_ft(
        object_points_m,
        image_points_px,
        fit,
        meters_to_feet=meters_to_feet,
    )
    override_details_ft = _world_plane_residual_details_ft(
        names,
        object_points_m,
        override_points,
        fit,
        meters_to_feet=meters_to_feet,
    )
    return {
        "diagnostic_only": True,
        "mutates_reviewed_labels": False,
        "status": "scored",
        "method": fit["method"],
        "override_candidate_count": len(overrides),
        "default_strategy": "endpoint_intersections_only",
        "strict_support_required": True,
        "skipped_non_strict_line_intersection_count": skipped_non_strict,
        "skipped_center_keypoint_count": skipped_center_keypoints,
        "line_intersection_support_counts": _source_counts(
            str(item["line_intersection_support"]) for item in overrides
        ),
        "mean_residual_ft": override_residual_ft["mean_residual_ft"],
        "median_residual_ft": override_residual_ft["median_residual_ft"],
        "p95_residual_ft": override_residual_ft["p95_residual_ft"],
        "original_reviewed_mean_residual_ft": original_reviewed_residual_ft["mean_residual_ft"],
        "original_reviewed_median_residual_ft": original_reviewed_residual_ft["median_residual_ft"],
        "original_reviewed_p95_residual_ft": original_reviewed_residual_ft["p95_residual_ft"],
        "delta_ft_vs_metric_plane_reviewed_baseline": round(
            float(baseline_mean_residual_ft) - float(override_residual_ft["mean_residual_ft"]),
            6,
        ),
        "original_reviewed_delta_ft_vs_metric_plane_reviewed_baseline": round(
            float(baseline_mean_residual_ft) - float(original_reviewed_residual_ft["mean_residual_ft"]),
            6,
        ),
        "per_keypoint_residual_ft": override_details_ft["per_keypoint_residual_ft"],
        "worst_keypoint": override_details_ft["worst_keypoint"],
        "overrides": overrides,
        "notes": [
            "Scores a temporary line-intersection observation set only; reviewed label JSON is unchanged.",
            "This is an upper-bound diagnostic for human review, not a production calibration source.",
        ],
    }


def _metric_plane_top_residual_line_intersection_override_oracle(
    keypoint_names: Sequence[str],
    object_points_m: Sequence[Sequence[float]],
    image_points_px: Sequence[Sequence[float]],
    *,
    image_size: tuple[float, float],
    progression: Sequence[Mapping[str, Any]],
    drop_count: int,
    baseline_mean_residual_ft: float,
    meters_to_feet: float,
    strict_support_required: bool = True,
    skip_center_keypoints: bool = True,
    source_strategy: str = "top_residual_strict_endpoint_line_intersections",
    camera_fit_model: str = "metric_plane",
) -> dict[str, Any]:
    names = [str(name) for name in keypoint_names]
    name_to_index = {name: index for index, name in enumerate(names)}
    fit_model = str(camera_fit_model)
    if fit_model not in {"metric_plane", "full_intrinsics_metric_plane"}:
        raise ValueError(f"unsupported top-residual override camera_fit_model: {fit_model}")
    selected: Mapping[str, Any] | None = None
    for item in progression:
        if not isinstance(item, Mapping) or item.get("status") != "scored":
            continue
        if item.get("drop_count") == int(drop_count):
            selected = item
            break
    if selected is None:
        return {
            "diagnostic_only": True,
            "promotes_calibration": False,
            "mutates_reviewed_labels": False,
            "source": "top_residual_refit_progression",
            "status": "skipped",
            "reason": "missing_requested_drop_count",
            "drop_count": int(drop_count),
            "override_candidate_count": 0,
            "strict_support_required": bool(strict_support_required),
            "skips_center_keypoints": bool(skip_center_keypoints),
            "source_strategy": str(source_strategy),
            "camera_fit_model": fit_model,
        }

    raw_details = selected.get("dropped_keypoint_details")
    details = (
        [detail for detail in raw_details if isinstance(detail, Mapping)]
        if isinstance(raw_details, Sequence) and not isinstance(raw_details, (str, bytes))
        else []
    )
    override_points = [list(_point2(point, "top_line_override.image_point")) for point in image_points_px]
    overrides: list[dict[str, Any]] = []
    skipped_missing = 0
    skipped_non_strict = 0
    skipped_center_keypoints = 0
    for detail in details:
        if detail.get("line_intersection_available") is not True:
            skipped_missing += 1
            continue
        name = str(detail.get("keypoint") or "")
        support_modes_raw = detail.get("line_support_modes")
        support_modes = (
            [str(value) for value in support_modes_raw]
            if isinstance(support_modes_raw, Sequence) and not isinstance(support_modes_raw, (str, bytes))
            else []
        )
        has_strict_support = bool(support_modes) and all(mode == "overlapping_segment" for mode in support_modes)
        is_center_keypoint = name in LINE_INTERSECTION_OVERRIDE_CENTER_KEYPOINTS
        if not has_strict_support:
            skipped_non_strict += 1
        if is_center_keypoint:
            skipped_center_keypoints += 1
        if (skip_center_keypoints and is_center_keypoint) or (strict_support_required and not has_strict_support):
            continue
        index = name_to_index.get(name)
        if index is None:
            continue
        try:
            line_point = _point2(detail.get("line_intersection_image_px"), "line_intersection_image_px")
            reviewed_point = _point2(detail.get("reviewed_image_px"), "reviewed_image_px")
        except ValueError:
            continue
        override_points[index] = line_point
        overrides.append(
            {
                "keypoint": name,
                "reviewed_image_px": [round(reviewed_point[0], 3), round(reviewed_point[1], 3)],
                "line_intersection_image_px": [round(line_point[0], 3), round(line_point[1], 3)],
                "line_intersection_support": _line_intersection_support(detail),
                "line_support_modes": support_modes,
                "line_intersection_delta_px": detail.get("line_intersection_delta_px"),
                "model_to_line_intersection_delta_px": detail.get("model_to_line_intersection_delta_px"),
                "source_residual_ft": detail.get("source_residual_ft"),
            }
        )

    if not overrides:
        no_override_reason = (
            "no_strict_top_residual_line_intersections"
            if strict_support_required
            else "no_relaxed_top_residual_line_intersections"
        )
        return {
            "diagnostic_only": True,
            "promotes_calibration": False,
            "mutates_reviewed_labels": False,
            "source": "top_residual_refit_progression",
            "status": "skipped",
            "reason": no_override_reason,
            "drop_count": int(drop_count),
            "available_line_intersection_count": sum(
                1 for detail in details if detail.get("line_intersection_available") is True
            ),
            "override_candidate_count": 0,
            "strict_support_required": bool(strict_support_required),
            "skips_center_keypoints": bool(skip_center_keypoints),
            "source_strategy": str(source_strategy),
            "camera_fit_model": fit_model,
            "skipped_missing_line_intersection_count": skipped_missing,
            "skipped_non_strict_line_intersection_count": skipped_non_strict,
            "skipped_center_keypoint_count": skipped_center_keypoints,
        }

    fit = (
        fit_full_intrinsics_metric_plane_camera_lm(
            object_points_m,
            override_points,
            image_size=image_size,
        )
        if fit_model == "full_intrinsics_metric_plane"
        else fit_metric_plane_camera_lm(
            object_points_m,
            override_points,
            image_size=image_size,
        )
    )
    override_residual_ft = _world_plane_residual_summary_ft(
        object_points_m,
        override_points,
        fit,
        meters_to_feet=meters_to_feet,
    )
    original_reviewed_residual_ft = _world_plane_residual_summary_ft(
        object_points_m,
        image_points_px,
        fit,
        meters_to_feet=meters_to_feet,
    )
    override_details_ft = _world_plane_residual_details_ft(
        names,
        object_points_m,
        override_points,
        fit,
        meters_to_feet=meters_to_feet,
    )
    delta_ft_vs_reviewed_baseline = round(
        float(baseline_mean_residual_ft) - float(override_residual_ft["mean_residual_ft"]),
        6,
    )
    original_reviewed_delta_ft_vs_reviewed_baseline = round(
        float(baseline_mean_residual_ft) - float(original_reviewed_residual_ft["mean_residual_ft"]),
        6,
    )
    baseline_delta_key = f"delta_ft_vs_{fit_model}_reviewed_baseline"
    original_baseline_delta_key = f"original_reviewed_delta_ft_vs_{fit_model}_reviewed_baseline"
    return {
        "diagnostic_only": True,
        "promotes_calibration": False,
        "mutates_reviewed_labels": False,
        "source": "top_residual_refit_progression",
        "status": "scored",
        "method": fit["method"],
        "camera_fit_model": fit_model,
        "drop_count": int(drop_count),
        "source_strategy": str(source_strategy),
        "available_line_intersection_count": sum(
            1 for detail in details if detail.get("line_intersection_available") is True
        ),
        "override_candidate_count": len(overrides),
        "strict_support_required": bool(strict_support_required),
        "skips_center_keypoints": bool(skip_center_keypoints),
        "skipped_missing_line_intersection_count": skipped_missing,
        "skipped_non_strict_line_intersection_count": skipped_non_strict,
        "skipped_center_keypoint_count": skipped_center_keypoints,
        "line_intersection_status_counts": _source_counts(
            str(detail.get("line_intersection_status", "unknown")) for detail in details
        ),
        "line_intersection_support_counts": _source_counts(
            str(item["line_intersection_support"]) for item in overrides
        ),
        "mean_residual_ft": override_residual_ft["mean_residual_ft"],
        "median_residual_ft": override_residual_ft["median_residual_ft"],
        "p95_residual_ft": override_residual_ft["p95_residual_ft"],
        "original_reviewed_mean_residual_ft": original_reviewed_residual_ft["mean_residual_ft"],
        "original_reviewed_median_residual_ft": original_reviewed_residual_ft["median_residual_ft"],
        "original_reviewed_p95_residual_ft": original_reviewed_residual_ft["p95_residual_ft"],
        "delta_ft_vs_reviewed_baseline": delta_ft_vs_reviewed_baseline,
        "original_reviewed_delta_ft_vs_reviewed_baseline": original_reviewed_delta_ft_vs_reviewed_baseline,
        baseline_delta_key: delta_ft_vs_reviewed_baseline,
        original_baseline_delta_key: original_reviewed_delta_ft_vs_reviewed_baseline,
        "per_keypoint_residual_ft": override_details_ft["per_keypoint_residual_ft"],
        "worst_keypoint": override_details_ft["worst_keypoint"],
        "overrides": overrides,
        "notes": [
            "Scores a temporary top-residual line-intersection observation set only; reviewed label JSON is unchanged.",
            "This is selected from reviewed-label residuals and is diagnostic-only, not a production calibration source.",
        ],
    }


def _trimmed_mean_or_none(values: Sequence[float], *, drop_worst_count: int) -> float | None:
    if drop_worst_count < 0:
        raise ValueError("drop_worst_count must be non-negative")
    finite = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not finite or drop_worst_count >= len(finite):
        return None
    kept = finite[: len(finite) - drop_worst_count] if drop_worst_count else finite
    return round(_mean(kept), 6)


def _point_line_weight_sweep(
    object_points_m: Sequence[Sequence[float]],
    image_points: Sequence[Sequence[float]],
    *,
    image_size: tuple[float, float],
    observations: Sequence[Mapping[str, Any]],
    meters_to_feet: float,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for weight in POINT_LINE_WEIGHT_SWEEP:
        fit = fit_joint_camera_point_line_lm(
            object_points_m,
            image_points,
            image_size=image_size,
            line_observations=observations,
            line_weight=float(weight),
        )
        residual_ft = _world_plane_residual_summary_ft(
            object_points_m,
            image_points,
            fit,
            meters_to_feet=meters_to_feet,
        )
        items.append(
            {
                "line_weight": round(float(weight), 6),
                "method": fit["method"],
                "success": fit["success"],
                "line_residual_mode": fit["line_residual_mode"],
                "line_pixel_sample_count": fit["line_pixel_sample_count"],
                "line_pixel_samples_per_observation": fit["line_pixel_samples_per_observation"],
                "optimized_reprojection_rmse_px": fit["optimized_reprojection_rmse_px"],
                "total_optimized_residual_rmse": fit["total_optimized_residual_rmse"],
                "mean_residual_ft": residual_ft["mean_residual_ft"],
                "median_residual_ft": residual_ft["median_residual_ft"],
                "p95_residual_ft": residual_ft["p95_residual_ft"],
            }
        )
    return items


def _point_line_pair_subset_oracle(
    object_points_m: Sequence[Sequence[float]],
    image_points: Sequence[Sequence[float]],
    *,
    image_size: tuple[float, float],
    observations: Sequence[Mapping[str, Any]],
    meters_to_feet: float,
) -> dict[str, Any]:
    if len(observations) < 2:
        return {
            "diagnostic_only": True,
            "available": False,
            "reason": "fewer_than_two_line_observations",
            "searched_subset_count": 0,
        }
    best: dict[str, Any] | None = None
    searched = 0
    for first_index in range(len(observations) - 1):
        for second_index in range(first_index + 1, len(observations)):
            subset = [observations[first_index], observations[second_index]]
            searched += 1
            for weight in POINT_LINE_WEIGHT_SWEEP:
                fit = fit_joint_camera_point_line_lm(
                    object_points_m,
                    image_points,
                    image_size=image_size,
                    line_observations=subset,
                    line_weight=float(weight),
                )
                residual_ft = _world_plane_residual_summary_ft(
                    object_points_m,
                    image_points,
                    fit,
                    meters_to_feet=meters_to_feet,
                )
                item = {
                    "diagnostic_only": True,
                    "available": True,
                    "line_weight": round(float(weight), 6),
                    "line_names": [str(observation["name"]) for observation in subset],
                    "method": fit["method"],
                    "success": fit["success"],
                    "line_residual_mode": fit["line_residual_mode"],
                    "line_pixel_sample_count": fit["line_pixel_sample_count"],
                    "line_pixel_samples_per_observation": fit["line_pixel_samples_per_observation"],
                    "optimized_reprojection_rmse_px": fit["optimized_reprojection_rmse_px"],
                    "total_optimized_residual_rmse": fit["total_optimized_residual_rmse"],
                    "mean_residual_ft": residual_ft["mean_residual_ft"],
                    "median_residual_ft": residual_ft["median_residual_ft"],
                    "p95_residual_ft": residual_ft["p95_residual_ft"],
                }
                if best is None or float(item["mean_residual_ft"]) < float(best["mean_residual_ft"]):
                    best = item
    assert best is not None
    best["searched_subset_count"] = int(searched)
    best["searched_weight_count"] = len(POINT_LINE_WEIGHT_SWEEP)
    return best


def _safe_selected_camera_payload(
    *,
    distorted_camera: Mapping[str, Any],
    distorted_residual_ft: Mapping[str, float],
    metric_plane_camera: Mapping[str, Any],
    metric_plane_residual_ft: Mapping[str, float],
    point_line_camera: Mapping[str, Any],
) -> dict[str, Any]:
    baseline = {
        "source": "distorted_camera",
        "selection_mode": "reviewed_label_diagnostic_only",
        "method": distorted_camera["method"],
        "optimized_reprojection_rmse_px": distorted_camera["optimized_reprojection_rmse_px"],
        "mean_residual_ft": distorted_residual_ft["mean_residual_ft"],
        "median_residual_ft": distorted_residual_ft["median_residual_ft"],
        "p95_residual_ft": distorted_residual_ft["p95_residual_ft"],
        "improvement_ft_vs_distorted_camera": 0.0,
    }
    candidates = [baseline]
    if metric_plane_residual_ft.get("mean_residual_ft") is not None:
        candidates.append(
            {
                "source": "metric_plane_camera",
                "selection_mode": "reviewed_label_diagnostic_only",
                "method": metric_plane_camera.get("method"),
                "optimized_reprojection_rmse_px": metric_plane_camera.get("optimized_reprojection_rmse_px"),
                "optimized_world_plane_rmse_m": metric_plane_camera.get("optimized_world_plane_rmse_m"),
                "mean_residual_ft": metric_plane_residual_ft["mean_residual_ft"],
                "median_residual_ft": metric_plane_residual_ft["median_residual_ft"],
                "p95_residual_ft": metric_plane_residual_ft["p95_residual_ft"],
                "improvement_ft_vs_distorted_camera": round(
                    float(distorted_residual_ft["mean_residual_ft"]) - float(metric_plane_residual_ft["mean_residual_ft"]),
                    6,
                ),
            }
        )
    best = point_line_camera.get("best_weighted_camera")
    if isinstance(best, Mapping) and best.get("mean_residual_ft") is not None:
        candidates.append(
            {
                "source": "point_line_weight_sweep",
                "selection_mode": "reviewed_label_diagnostic_only",
                "method": point_line_camera.get("method"),
                "line_weight": best.get("line_weight"),
                "optimized_reprojection_rmse_px": best.get("optimized_reprojection_rmse_px"),
                "mean_residual_ft": round(float(best["mean_residual_ft"]), 6),
                "median_residual_ft": best.get("median_residual_ft"),
                "p95_residual_ft": best.get("p95_residual_ft"),
                "improvement_ft_vs_distorted_camera": round(
                    float(distorted_residual_ft["mean_residual_ft"]) - float(best["mean_residual_ft"]),
                    6,
                ),
            }
        )
    baseline_mean = float(distorted_residual_ft["mean_residual_ft"])
    selected = min(candidates, key=lambda item: float(item["mean_residual_ft"]))
    if selected is not baseline and float(selected["mean_residual_ft"]) < baseline_mean:
        return selected
    baseline["reason"] = "no_candidate_better_than_distorted_camera"
    if isinstance(best, Mapping) and best.get("mean_residual_ft") is not None:
        baseline["best_point_line_mean_residual_ft"] = round(float(best["mean_residual_ft"]), 6)
        baseline["best_point_line_line_weight"] = best.get("line_weight")
    if metric_plane_residual_ft.get("mean_residual_ft") is not None:
        baseline["metric_plane_mean_residual_ft"] = round(float(metric_plane_residual_ft["mean_residual_ft"]), 6)
    return baseline


def _neural_keypoint_checkpoint_evidence(*, repo_root: Path) -> dict[str, Any]:
    runs_root = repo_root / "runs"
    if not runs_root.is_dir():
        return {
            "diagnostic_only": True,
            "promotes_calibration": False,
            "status": "unavailable",
            "reason": "missing_runs_directory",
            "candidate_count": 0,
            "real_label_candidate_count": 0,
            "gate_pass_count": 0,
            "best_real_label_candidate": None,
            "candidates": [],
        }

    candidates: list[dict[str, Any]] = []
    for metrics_path in _recursive_files_named(runs_root, NEURAL_KEYPOINT_METRICS_GLOB):
        candidate = _neural_keypoint_candidate_from_metrics(metrics_path, repo_root=repo_root)
        if candidate is not None:
            candidates.append(candidate)

    real_label_candidates = [
        candidate
        for candidate in candidates
        if candidate.get("candidate_metric_name") == "after.real_keypoint_median_px"
        and candidate.get("candidate_metric_value_px") is not None
    ]
    best = (
        min(real_label_candidates, key=lambda item: float(item["candidate_metric_value_px"]))
        if real_label_candidates
        else None
    )
    return {
        "diagnostic_only": True,
        "promotes_calibration": False,
        "status": "scored" if candidates else "no_checkpoint_metrics",
        "repo_root": str(repo_root),
        "metrics_glob": f"runs/**/{NEURAL_KEYPOINT_METRICS_GLOB}",
        "candidate_count": len(candidates),
        "real_label_candidate_count": len(real_label_candidates),
        "gate_pass_count": sum(1 for candidate in candidates if candidate.get("gate_passed") is True),
        "best_real_label_candidate": best,
        "candidates": candidates,
        "notes": [
            "Neural court-keypoint evidence is read from existing training metrics only.",
            "These checkpoint metrics do not promote calibration unless the reviewed-label gate passes separately.",
        ],
    }


def _mobilenet_v3_keypoint_checkpoint_evidence(*, repo_root: Path, eval_root: Path) -> dict[str, Any]:
    checkpoint_paths = _mobilenet_v3_keypoint_checkpoint_paths(repo_root=repo_root)
    base: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "mobilenet_v3_court_keypoint_regressor_checkpoint_evidence",
        "diagnostic_only": True,
        "promotes_calibration": False,
        "verified": False,
        "not_cal3_verified": True,
        "repo_root": str(repo_root),
        "checkpoint_globs": list(MOBILENET_V3_KEYPOINT_CHECKPOINT_GLOBS),
        "candidate_count": len(checkpoint_paths),
        "scored_candidate_count": 0,
        "best_candidate": None,
        "candidates": [],
        "notes": [
            "MobileNetV3 direct-regression checkpoints must be explicit local artifacts.",
            "A missing checkpoint is recorded as unavailable; it is not inferred from heatmap-model metrics.",
        ],
    }
    if not checkpoint_paths:
        base.update({"status": "unavailable", "reason": "missing_checkpoint_candidates"})
        return base

    candidate_reports: list[dict[str, Any]] = []
    fallback_checkpoint_paths: list[Path] = []
    for checkpoint_path in checkpoint_paths:
        candidate = _mobilenet_v3_candidate_from_training_metrics(checkpoint_path, repo_root=repo_root)
        if candidate is None:
            fallback_checkpoint_paths.append(checkpoint_path)
        else:
            candidate_reports.append(candidate)

    rows: list[dict[str, Any]] = []
    if fallback_checkpoint_paths:
        try:
            from scripts.racketsport.train_court_keypoint_heatmap import load_real_court_keypoint_labels

            rows = load_real_court_keypoint_labels(eval_root)
        except Exception as exc:
            candidate_reports.extend(
                {
                    "status": "unavailable",
                    "reason": "reviewed_label_load_failed",
                    "checkpoint": str(checkpoint_path),
                    "source": "all_reviewed_rows_fallback",
                    "error": str(exc),
                    "diagnostic_only": True,
                    "promotes_calibration": False,
                    "verified": False,
                    "not_cal3_verified": True,
                    "promotion_blockers": [
                        "diagnostic_only",
                        "not_cal3_verified",
                        "gate_failed",
                        "reviewed_label_load_failed",
                    ],
                }
                for checkpoint_path in fallback_checkpoint_paths
            )
        else:
            from .court_detector_v2_model import evaluate_mobilenet_v3_court_keypoint_regressor_checkpoint

            candidate_reports.extend(
                evaluate_mobilenet_v3_court_keypoint_regressor_checkpoint(
                    checkpoint_path=checkpoint_path,
                    rows=rows,
                    device="cpu",
                )
                | {"source": "all_reviewed_rows_fallback"}
                for checkpoint_path in fallback_checkpoint_paths
            )
    scored = [
        candidate
        for candidate in candidate_reports
        if candidate.get("status") == "scored" and candidate.get("median_error_px") is not None
    ]
    best = min(scored, key=lambda item: float(item["median_error_px"])) if scored else None
    base.update(
        {
            "status": "scored" if scored else "unavailable",
            "reason": None if scored else "no_scored_checkpoint_candidates",
            "reviewed_row_count": len(rows) if rows else None,
            "scored_candidate_count": len(scored),
            "best_candidate": best,
            "candidates": candidate_reports,
        }
    )
    return base


def _mobilenet_v3_candidate_from_training_metrics(checkpoint_path: Path, *, repo_root: Path) -> dict[str, Any] | None:
    metrics_path = checkpoint_path.with_name("mobilenet_v3_court_keypoint_metrics.json")
    if not metrics_path.is_file():
        return None
    try:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, Mapping):
        return None
    if payload.get("artifact_type") != "mobilenet_v3_court_keypoint_regressor_training_report":
        return None
    evaluation = payload.get("evaluation")
    if not isinstance(evaluation, Mapping):
        return None
    candidate_checkpoint = Path(str(payload.get("checkpoint") or checkpoint_path))
    if not candidate_checkpoint.is_absolute():
        candidate_checkpoint = repo_root / candidate_checkpoint
    pck_at_5px = _optional_number(evaluation.get("pck_at_5px"))
    return {
        "source": "sibling_training_metrics",
        "metrics_path": str(metrics_path),
        "checkpoint": str(checkpoint_path),
        "resolved_checkpoint": str(candidate_checkpoint) if candidate_checkpoint.is_file() else str(checkpoint_path),
        "checkpoint_exists": checkpoint_path.is_file(),
        "status": str(evaluation.get("status") or payload.get("status") or "unknown"),
        "diagnostic_only": True,
        "promotes_calibration": False,
        "verified": False,
        "not_cal3_verified": True,
        "architecture": payload.get("architecture"),
        "coordinate_mode": payload.get("coordinate_mode"),
        "train_row_count": _optional_int(payload.get("train_row_count")),
        "holdout_row_count": _optional_int(payload.get("holdout_row_count")),
        "train_clip_names": payload.get("train_clip_names") if isinstance(payload.get("train_clip_names"), list) else [],
        "holdout_clip_names": (
            payload.get("holdout_clip_names") if isinstance(payload.get("holdout_clip_names"), list) else []
        ),
        "mean_error_px": _round_optional_number(evaluation.get("mean_error_px")),
        "median_error_px": _round_optional_number(evaluation.get("median_error_px")),
        "p95_error_px": _round_optional_number(evaluation.get("p95_error_px")),
        "pck_at_5px": None if pck_at_5px is None else round(pck_at_5px, 6),
        "evaluated_keypoint_count": _optional_int(evaluation.get("evaluated_keypoint_count")),
        "gate_passed": evaluation.get("gate_passed") is True,
        "promotion_blockers": _mobilenet_training_candidate_blockers(payload, evaluation=evaluation),
    }


def _mobilenet_training_candidate_blockers(
    payload: Mapping[str, Any],
    *,
    evaluation: Mapping[str, Any],
) -> list[str]:
    blockers = ["diagnostic_only", "not_cal3_verified"]
    if evaluation.get("gate_passed") is not True:
        blockers.append("gate_failed")
    if evaluation.get("status") != "scored":
        blockers.append("missing_scored_holdout")
    if _optional_int(payload.get("holdout_row_count")) in (None, 0):
        blockers.append("missing_holdout_rows")
    return blockers


def _mobilenet_v3_keypoint_checkpoint_paths(*, repo_root: Path) -> list[Path]:
    paths: set[Path] = set()
    runs_root = repo_root / "runs"
    if not runs_root.is_dir():
        return []
    for path in _recursive_files(runs_root):
        if _is_mobilenet_v3_keypoint_checkpoint_path(path):
            paths.add(path)
    return sorted(paths)


def _is_mobilenet_v3_keypoint_checkpoint_path(path: Path) -> bool:
    name = path.name
    return (
        name.startswith("mobilenet_v3_court_keypoint_regressor")
        or (name.startswith("court_mobilenet_v3") and "_regressor" in name)
    ) and path.suffix == ".pt"


def _recursive_files_named(root: Path, name: str) -> list[Path]:
    return sorted(path for path in _recursive_files(root) if path.name == name)


def _recursive_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for dirpath, _dirnames, filenames in os.walk(root, onerror=lambda _error: None):
        directory = Path(dirpath)
        for filename in filenames:
            files.append(directory / filename)
    return sorted(files)


def _neural_keypoint_candidate_from_metrics(metrics_path: Path, *, repo_root: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, Mapping):
        return None
    if payload.get("checkpoint") is None and payload.get("gate") is None and payload.get("after") is None:
        return None

    gate = payload.get("gate") if isinstance(payload.get("gate"), Mapping) else {}
    after = payload.get("after") if isinstance(payload.get("after"), Mapping) else {}
    before = payload.get("before") if isinstance(payload.get("before"), Mapping) else {}
    architecture = payload.get("architecture") if isinstance(payload.get("architecture"), Mapping) else {}
    postprocess = payload.get("postprocess") if isinstance(payload.get("postprocess"), Mapping) else {}
    checkpoint_text = payload.get("checkpoint")
    checkpoint_path = _resolve_neural_checkpoint_path(
        metrics_path=metrics_path,
        checkpoint_text=checkpoint_text,
        repo_root=repo_root,
    )
    real_holdout_count = _optional_int(payload.get("real_holdout_count"))
    candidate_metric_value = _optional_number(after.get("real_keypoint_median_px"))
    promotion_blockers = _neural_candidate_promotion_blockers(
        payload,
        gate=gate,
        checkpoint_exists=checkpoint_path is not None,
        candidate_metric_value=candidate_metric_value,
    )
    holdout_artifacts = [
        item for item in payload.get("holdout_artifacts", []) if isinstance(item, Mapping)
    ] if isinstance(payload.get("holdout_artifacts"), Sequence) and not isinstance(
        payload.get("holdout_artifacts"), (str, bytes)
    ) else []
    holdout_medians = [
        value
        for value in (_optional_number(item.get("median_keypoint_reprojection_px")) for item in holdout_artifacts)
        if value is not None
    ]
    holdout_p95s = [
        value
        for value in (_optional_number(item.get("p95_keypoint_reprojection_px")) for item in holdout_artifacts)
        if value is not None
    ]

    return {
        "metrics_path": str(metrics_path),
        "status": str(payload.get("status") or "unknown"),
        "diagnostic_only": True,
        "promotes_calibration": False,
        "not_cal3_verified": payload.get("not_cal3_verified", True) is not False,
        "checkpoint": str(checkpoint_text) if checkpoint_text is not None else None,
        "resolved_checkpoint": None if checkpoint_path is None else str(checkpoint_path),
        "checkpoint_exists": checkpoint_path is not None,
        "architecture": architecture.get("name"),
        "network_architecture": architecture.get("network_architecture"),
        "prediction_mode": postprocess.get("prediction_mode") or after.get("prediction_mode"),
        "gate_metric": gate.get("metric"),
        "gate_value": _optional_number(gate.get("value")),
        "gate_passed": gate.get("passed") is True,
        "gate_threshold": _optional_number(gate.get("threshold")),
        "gate_pck_threshold_px": _optional_number(gate.get("pck_threshold_px")),
        "labels_independent_human_frames": _optional_int(
            gate.get("independent_reviewed_frame_count", payload.get("labels_independent_human_frames"))
        ),
        "labels_static_camera_copy_frame_count": _optional_int(
            gate.get("copied_frame_count", payload.get("labels_static_camera_copy_frame_count"))
        ),
        "labels_synthetic_frame_count": _optional_int(
            gate.get("synthetic_frame_count", payload.get("labels_synthetic_frame_count"))
        ),
        "real_train_count": _optional_int(payload.get("real_train_count")),
        "real_holdout_count": real_holdout_count,
        "candidate_metric_name": "after.real_keypoint_median_px" if candidate_metric_value is not None else None,
        "candidate_metric_value_px": None if candidate_metric_value is None else round(candidate_metric_value, 6),
        "after_real_keypoint_mean_px": _round_optional_number(after.get("real_keypoint_mean_px")),
        "after_real_keypoint_median_px": _round_optional_number(after.get("real_keypoint_median_px")),
        "after_real_keypoint_p95_px": _round_optional_number(after.get("real_keypoint_p95_px")),
        "after_real_keypoint_pck_at_5px": _round_optional_number(after.get("real_keypoint_pck_at_5px")),
        "before_real_keypoint_mean_px": _round_optional_number(before.get("real_keypoint_mean_px")),
        "before_real_keypoint_median_px": _round_optional_number(before.get("real_keypoint_median_px")),
        "before_real_keypoint_p95_px": _round_optional_number(before.get("real_keypoint_p95_px")),
        "best_holdout_artifact_median_px": None if not holdout_medians else round(min(holdout_medians), 6),
        "best_holdout_artifact_p95_px": None if not holdout_p95s else round(min(holdout_p95s), 6),
        "holdout_artifact_count": len(holdout_artifacts),
        "promotion_blockers": promotion_blockers,
    }


def _neural_candidate_promotion_blockers(
    payload: Mapping[str, Any],
    *,
    gate: Mapping[str, Any],
    checkpoint_exists: bool,
    candidate_metric_value: float | None,
) -> list[str]:
    blockers = ["diagnostic_only", "not_cal3_verified"]
    if str(payload.get("status") or "") != "phase_verified":
        blockers.append("trained_not_phase_verified")
    if gate.get("passed") is not True:
        blockers.append("gate_failed")
    if not checkpoint_exists:
        blockers.append("missing_checkpoint")
    if candidate_metric_value is None:
        blockers.append("missing_real_label_median_px")
    if _optional_int(payload.get("real_holdout_count")) in (None, 0):
        blockers.append("missing_real_holdout")
    return blockers


def _resolve_neural_checkpoint_path(
    *,
    metrics_path: Path,
    checkpoint_text: Any,
    repo_root: Path,
) -> Path | None:
    if not isinstance(checkpoint_text, str) or not checkpoint_text:
        return None
    raw = Path(checkpoint_text)
    candidates: list[Path] = []
    if raw.is_absolute():
        candidates.append(raw)
        parts = raw.parts
        if "runs" in parts:
            runs_index = parts.index("runs")
            candidates.append(repo_root / Path(*parts[runs_index:]))
    else:
        candidates.append(repo_root / raw)
        candidates.append(metrics_path.parent / raw.name)
    candidates.append(metrics_path.with_name("court_keypoint_heatmap.pt"))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _calibration_repo_root(eval_root: Path) -> Path:
    resolved = eval_root.resolve()
    for candidate in (resolved, *resolved.parents):
        if (candidate / "runs").is_dir() and (candidate / "eval_clips").exists():
            return candidate
    return Path.cwd()


def _optional_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return number


def _round_optional_number(value: Any) -> float | None:
    number = _optional_number(value)
    return None if number is None else round(number, 6)


def _optional_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return int(value)


def _sweep_item_for_weight(items: Sequence[Mapping[str, Any]], weight: float) -> dict[str, Any]:
    if not items:
        raise ValueError("point-line weight sweep is empty")
    target = float(weight)
    best = min(items, key=lambda item: abs(float(item["line_weight"]) - target))
    return dict(best)


def _point_line_fit_for_reviewed_clip(
    label_path: Path,
    *,
    aggregated: Mapping[str, tuple[float, float]],
    object_points_m: Sequence[Sequence[float]],
    image_points: Sequence[Sequence[float]],
    image_size: tuple[float, float],
    keypoint_by_name: Mapping[str, Any],
    meters_to_feet: float,
) -> dict[str, Any]:
    technology_id = "opencv_hough_lsd_temporal_persistent"
    try:
        evidence = _temporal_line_evidence_for_reviewed_clip(label_path, technology_id=technology_id)
        segments = _scale_segments_to_image_size(
            evidence.get("segments", []),
            source_image_size=_evidence_image_size(evidence),
            target_image_size=image_size,
        )
        observations = _line_observations_from_segments(
            aggregated=aggregated,
            segments=segments,
            keypoint_by_name=keypoint_by_name,
        )
        if len(observations) < 2:
            return {
                "status": "skipped",
                "reason": "fewer_than_two_supported_hough_line_observations",
                "supported_line_observation_count": len(observations),
                "line_candidate_technology_id": technology_id,
                "raw_segment_count": len(segments),
                "temporal_frame_count": evidence.get("temporal_frame_count"),
            }
        weight_sweep = _point_line_weight_sweep(
            object_points_m,
            image_points,
            image_size=image_size,
            observations=observations,
            meters_to_feet=meters_to_feet,
        )
        pair_subset_oracle = _point_line_pair_subset_oracle(
            object_points_m,
            image_points,
            image_size=image_size,
            observations=observations,
            meters_to_feet=meters_to_feet,
        )
        default_item = _sweep_item_for_weight(weight_sweep, DEFAULT_POINT_LINE_WEIGHT)
        best_item = min(weight_sweep, key=lambda item: float(item["mean_residual_ft"]))
        default_improvement = round(
            float(best_item["mean_residual_ft"]) - float(default_item["mean_residual_ft"]),
            6,
        )
        return {
            "status": "fit",
            "method": default_item["method"],
            "success": default_item["success"],
            "supported_line_observation_count": len(observations),
            "line_candidate_technology_id": technology_id,
            "line_weight": default_item["line_weight"],
            "line_residual_mode": default_item["line_residual_mode"],
            "line_pixel_sample_count": default_item["line_pixel_sample_count"],
            "line_pixel_samples_per_observation": default_item["line_pixel_samples_per_observation"],
            "raw_segment_count": len(segments),
            "temporal_frame_count": evidence.get("temporal_frame_count"),
            "optimized_reprojection_rmse_px": default_item["optimized_reprojection_rmse_px"],
            "mean_residual_ft": default_item["mean_residual_ft"],
            "median_residual_ft": default_item["median_residual_ft"],
            "p95_residual_ft": default_item["p95_residual_ft"],
            "total_optimized_residual_rmse": default_item["total_optimized_residual_rmse"],
            "line_names": [str(observation["name"]) for observation in observations],
            "weight_sweep": weight_sweep,
            "best_weighted_camera": dict(best_item),
            "best_weighted_delta_ft_vs_default": default_improvement,
            "pair_subset_oracle": pair_subset_oracle,
        }
    except Exception as exc:
        return {"status": "error", "line_candidate_technology_id": technology_id, "reason": str(exc)}


def _review_line_observations_for_reviewed_clip(
    label_path: Path,
    *,
    aggregated: Mapping[str, tuple[float, float]],
    image_size: tuple[float, float],
    keypoint_by_name: Mapping[str, Any],
) -> list[dict[str, Any]]:
    technology_id = "opencv_hough_lsd_temporal_persistent"
    try:
        evidence = _temporal_line_evidence_for_reviewed_clip(label_path, technology_id=technology_id)
        segments = _scale_segments_to_image_size(
            evidence.get("segments", []),
            source_image_size=_evidence_image_size(evidence),
            target_image_size=image_size,
        )
        return _line_observations_from_segments(
            aggregated=aggregated,
            segments=segments,
            keypoint_by_name=keypoint_by_name,
        )
    except Exception:
        return []


def _model_projected_line_observations_for_reviewed_clip(
    label_path: Path,
    *,
    keypoint_names: Sequence[str],
    object_points_m: Sequence[Sequence[float]],
    model_fit: Mapping[str, Any],
    image_size: tuple[float, float],
    keypoint_by_name: Mapping[str, Any],
) -> list[dict[str, Any]]:
    technology_id = "opencv_hough_lsd_temporal_persistent"
    try:
        evidence = _temporal_line_evidence_for_reviewed_clip(label_path, technology_id=technology_id)
        segments = _scale_segments_to_image_size(
            evidence.get("segments", []),
            source_image_size=_evidence_image_size(evidence),
            target_image_size=image_size,
        )
        return _line_observations_from_projected_model(
            keypoint_names=keypoint_names,
            object_points_m=object_points_m,
            model_fit=model_fit,
            segments=segments,
            keypoint_by_name=keypoint_by_name,
        )
    except Exception:
        return []


def _resolve_report_label_path(result: Mapping[str, Any], *, eval_root: Path) -> Path | None:
    raw = result.get("label_path")
    if isinstance(raw, str) and raw:
        direct = Path(raw)
        if direct.is_file():
            return direct
        cwd_relative = Path.cwd() / direct
        if cwd_relative.is_file():
            return cwd_relative
    clip = str(result.get("clip") or "")
    if clip:
        fallback = eval_root / clip / "labels" / "court_keypoints.json"
        if fallback.is_file():
            return fallback
    return None


def _top_residual_detail_by_keypoint(diagnostic: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    raw_details = diagnostic.get("dropped_keypoint_details")
    if not isinstance(raw_details, Sequence) or isinstance(raw_details, (str, bytes)):
        return {}
    details: dict[str, dict[str, Any]] = {}
    for raw in raw_details:
        if not isinstance(raw, Mapping):
            continue
        name = str(raw.get("keypoint") or "")
        if name:
            details[name] = dict(raw)
    return details


def _top_residual_refit_payload_for_drop_count(
    metric_plane: Mapping[str, Any],
    *,
    drop_count: int | None,
) -> Mapping[str, Any] | None:
    if drop_count is None:
        diagnostic = metric_plane.get("top_residual_refit_diagnostic")
        return diagnostic if isinstance(diagnostic, Mapping) else None
    progression = metric_plane.get("top_residual_refit_progression")
    if isinstance(progression, Sequence) and not isinstance(progression, (str, bytes)):
        for raw_item in progression:
            if not isinstance(raw_item, Mapping):
                continue
            raw_drop_count = raw_item.get("drop_count")
            if isinstance(raw_drop_count, bool) or not isinstance(raw_drop_count, int):
                continue
            if int(raw_drop_count) == int(drop_count):
                return raw_item
    diagnostic = metric_plane.get("top_residual_refit_diagnostic")
    if isinstance(diagnostic, Mapping) and diagnostic.get("drop_count") == drop_count:
        return diagnostic
    return None


def _outlier_candidate_by_keypoint(metric_plane: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    raw_candidates = metric_plane.get("outlier_review_candidates")
    if not isinstance(raw_candidates, Sequence) or isinstance(raw_candidates, (str, bytes)):
        return {}
    candidates: dict[str, dict[str, Any]] = {}
    for raw in raw_candidates:
        if not isinstance(raw, Mapping):
            continue
        name = str(raw.get("keypoint") or "")
        if name:
            candidates[name] = dict(raw)
    return candidates


def _top_residual_value_for_keypoint(diagnostic: Mapping[str, Any], keypoint: str) -> float | None:
    residuals = diagnostic.get("dropped_keypoint_residual_ft")
    if not isinstance(residuals, Mapping):
        return None
    return _round_optional_number(residuals.get(keypoint))


def _metric_packet_frame_index(label_path: Path | None) -> int:
    if label_path is None:
        return 0
    calibration_path = label_path.with_name("court_calibration_metric15pt.json")
    if calibration_path.is_file():
        payload = json.loads(calibration_path.read_text(encoding="utf-8"))
        solved = payload.get("solved_over_frames")
        if isinstance(solved, Sequence) and not isinstance(solved, (str, bytes)) and solved:
            try:
                return max(0, int(solved[0]))
            except (TypeError, ValueError):
                return 0
    return 0


def _read_video_frame(video_path: Path, frame_index: int, cv2: Any) -> Any:
    if not video_path.is_file():
        raise ValueError(f"missing source video: {video_path}")
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise ValueError(f"cannot open source video: {video_path}")
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        index = max(0, int(frame_index))
        if total_frames > 0 and index >= total_frames:
            raise ValueError(f"frame_index {index} is outside source video: {video_path}")
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if not ok or frame is None:
            raise ValueError(f"failed reading frame {index} from {video_path}")
        return frame
    finally:
        capture.release()


def _packet_crop_center(points: Sequence[Sequence[float] | None]) -> tuple[float, float]:
    valid = [tuple(_point2(point, "packet_crop_center.point")) for point in points if point is not None]
    if not valid:
        raise ValueError("at least one crop point is required")
    return (_mean([point[0] for point in valid]), _mean([point[1] for point in valid]))


def _crop_around_point(frame: Any, center: tuple[float, float], *, radius_px: int) -> tuple[Any, tuple[int, int, int, int]]:
    if frame is None or not hasattr(frame, "shape") or len(frame.shape) < 2:
        raise ValueError("frame must be an OpenCV-style image")
    height, width = int(frame.shape[0]), int(frame.shape[1])
    if height <= 0 or width <= 0:
        raise ValueError("frame must have positive size")
    radius = int(radius_px)
    cx, cy = center
    x1 = max(0, int(round(cx)) - radius)
    y1 = max(0, int(round(cy)) - radius)
    x2 = min(width, int(round(cx)) + radius)
    y2 = min(height, int(round(cy)) + radius)
    if x2 <= x1:
        x2 = min(width, x1 + 1)
    if y2 <= y1:
        y2 = min(height, y1 + 1)
    return frame[y1:y2, x1:x2].copy(), (x1, y1, x2, y2)


def _draw_metric_plane_outlier_crop(
    cv2: Any,
    crop: Any,
    *,
    candidate: Mapping[str, Any],
    crop_origin: tuple[int, int],
) -> None:
    line_type = getattr(cv2, "LINE_AA", 16)
    reviewed = _local_crop_point(candidate.get("reviewed_image_px"), crop_origin)
    model = _local_crop_point(candidate.get("model_projected_image_px"), crop_origin)
    cv2.line(crop, reviewed, model, (0, 0, 255), 1, line_type)
    _draw_packet_marker(cv2, crop, reviewed, color=(0, 220, 0), label="reviewed")
    _draw_packet_marker(cv2, crop, model, color=(0, 0, 255), label="model")
    if candidate.get("line_intersection_available") is True and candidate.get("line_intersection_image_px") is not None:
        try:
            line_point = _local_crop_point(candidate.get("line_intersection_image_px"), crop_origin)
            cv2.line(crop, reviewed, line_point, (255, 255, 0), 1, line_type)
            _draw_packet_marker(cv2, crop, line_point, color=(255, 255, 0), label="line")
        except ValueError:
            pass
    header = f"{candidate.get('keypoint', 'keypoint')}  {candidate.get('residual_ft', '?')} ft"
    _draw_packet_text(cv2, crop, header, (6, 16), color=(255, 255, 255))
    _draw_packet_text(
        cv2,
        crop,
        f"line: {_short_line_support_label(_line_intersection_support(candidate))}",
        (6, 34),
        color=(220, 220, 220),
    )


def _local_crop_point(point: Any, crop_origin: tuple[int, int]) -> tuple[int, int]:
    x, y = _point2(point, "crop.point")
    return (int(round(x - crop_origin[0])), int(round(y - crop_origin[1])))


def _draw_packet_marker(
    cv2: Any,
    crop: Any,
    point: tuple[int, int],
    *,
    color: tuple[int, int, int],
    label: str,
) -> None:
    line_type = getattr(cv2, "LINE_AA", 16)
    cv2.circle(crop, point, 7, (0, 0, 0), 2, line_type)
    cv2.circle(crop, point, 5, color, 2, line_type)
    _draw_packet_text(cv2, crop, label, (point[0] + 8, point[1] - 6), color=color)


def _draw_packet_text(
    cv2: Any,
    crop: Any,
    text: str,
    origin: tuple[int, int],
    *,
    color: tuple[int, int, int],
) -> None:
    font = getattr(cv2, "FONT_HERSHEY_SIMPLEX", 0)
    line_type = getattr(cv2, "LINE_AA", 16)
    safe_origin = (max(0, int(origin[0])), max(12, int(origin[1])))
    cv2.putText(crop, text, (safe_origin[0] + 1, safe_origin[1] + 1), font, 0.42, (0, 0, 0), 2, line_type)
    cv2.putText(crop, text, safe_origin, font, 0.42, color, 1, line_type)


def _review_packet_tile(image: Any, tile_size: int, np: Any) -> Any:
    tile = np.full((int(tile_size), int(tile_size), 3), 18, dtype=np.uint8)
    height = min(int(image.shape[0]), int(tile_size))
    width = min(int(image.shape[1]), int(tile_size))
    tile[:height, :width] = image[:height, :width]
    return tile


def _review_packet_contact_sheet(tiles: Sequence[Any], np: Any) -> Any:
    if not tiles:
        raise ValueError("tiles must be non-empty")
    columns = min(4, len(tiles))
    rows = int(math.ceil(len(tiles) / columns))
    tile_height, tile_width = int(tiles[0].shape[0]), int(tiles[0].shape[1])
    sheet = np.full((rows * tile_height, columns * tile_width, 3), 12, dtype=np.uint8)
    for index, tile in enumerate(tiles):
        row = index // columns
        column = index % columns
        y1 = row * tile_height
        x1 = column * tile_width
        sheet[y1 : y1 + tile_height, x1 : x1 + tile_width] = tile
    return sheet


def _line_intersection_support(candidate: Mapping[str, Any], *, tie_px: float = 2.0) -> str:
    if candidate.get("line_intersection_available") is not True:
        return "missing_line_intersection"
    try:
        reviewed_to_line = _finite_float(candidate.get("line_intersection_delta_px"), "line_intersection_delta_px")
        model_to_line = _finite_float(
            candidate.get("model_to_line_intersection_delta_px"),
            "model_to_line_intersection_delta_px",
        )
    except ValueError:
        return "unusable_line_intersection_metrics"
    if reviewed_to_line + tie_px < model_to_line:
        return "reviewed_label_closer_to_line"
    if model_to_line + tie_px < reviewed_to_line:
        return "model_projection_closer_to_line"
    return "ambiguous_or_tie"


def _short_line_support_label(value: str) -> str:
    labels = {
        "missing_line_intersection": "none",
        "model_projection_closer_to_line": "model",
        "reviewed_label_closer_to_line": "review",
        "ambiguous_or_tie": "tie",
        "unusable_line_intersection_metrics": "bad",
    }
    return labels.get(value, value[:16])


def _safe_filename_token(value: str) -> str:
    token = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value.strip())
    return token.strip("_") or "item"


def _load_first_review_frame(label_path: Path) -> Any | None:
    cv2 = _cv2()
    payload = json.loads(label_path.read_text(encoding="utf-8"))
    frames = payload.get("frames") if isinstance(payload.get("frames"), Mapping) else {}
    frame_dir = frames.get("frame_dir") if isinstance(frames, Mapping) else None
    if not frame_dir:
        return None
    root = Path(frame_dir)
    if not root.is_absolute():
        root = Path.cwd() / root
    frame_paths = sorted(root.glob("*.jpg"))
    if not frame_paths:
        return None
    return cv2.imread(str(frame_paths[0]), cv2.IMREAD_COLOR)


def _temporal_line_evidence_for_reviewed_clip(label_path: Path, *, technology_id: str) -> dict[str, Any]:
    from .court_finding_technology_benchmark import detect_temporal_line_candidates_for_input

    frame_input = _review_frame_input_path(label_path)
    if frame_input is None:
        frame = _load_first_review_frame(label_path)
        if frame is None:
            return {
                "technology_id": "opencv_hough",
                "available": False,
                "candidate_count": 0,
                "segments": [],
                "reason": "no_readable_review_frame",
            }
        segments = _plain_hough_segments(frame)
        return {
            "technology_id": "opencv_hough",
            "available": True,
            "candidate_count": len(segments),
            "segments": segments,
            "temporal_frame_count": 1,
            "image_size": [int(frame.shape[1]), int(frame.shape[0])],
        }
    return detect_temporal_line_candidates_for_input(frame_input, technology_id=technology_id)


def _review_frame_input_path(label_path: Path) -> Path | None:
    payload = json.loads(label_path.read_text(encoding="utf-8"))
    frames = payload.get("frames") if isinstance(payload.get("frames"), Mapping) else {}
    frame_dir = frames.get("frame_dir") if isinstance(frames, Mapping) else None
    if not frame_dir:
        return None
    root = Path(frame_dir)
    if not root.is_absolute():
        root = Path.cwd() / root
    return root if root.exists() else None


def _evidence_image_size(evidence: Mapping[str, Any]) -> tuple[float, float] | None:
    raw = evidence.get("image_size")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or len(raw) != 2:
        return None
    width = _finite_float(raw[0], "line_evidence.image_size.width")
    height = _finite_float(raw[1], "line_evidence.image_size.height")
    if width <= 0.0 or height <= 0.0:
        return None
    return (width, height)


def _scale_segments_to_image_size(
    segments: Sequence[Mapping[str, Any]],
    *,
    source_image_size: tuple[float, float] | None,
    target_image_size: tuple[float, float],
) -> list[dict[str, Any]]:
    target_width, target_height = _image_size2(target_image_size)
    if source_image_size is None:
        scale_x = 1.0
        scale_y = 1.0
    else:
        source_width, source_height = _image_size2(source_image_size)
        scale_x = target_width / source_width
        scale_y = target_height / source_height
    scaled: list[dict[str, Any]] = []
    for segment in segments:
        p1 = segment.get("p1")
        p2 = segment.get("p2")
        if not isinstance(p1, Sequence) or not isinstance(p2, Sequence) or len(p1) != 2 or len(p2) != 2:
            continue
        x1, y1 = float(p1[0]) * scale_x, float(p1[1]) * scale_y
        x2, y2 = float(p2[0]) * scale_x, float(p2[1]) * scale_y
        item = dict(segment)
        item["p1"] = [x1, y1]
        item["p2"] = [x2, y2]
        item["length_px"] = round(math.hypot(x2 - x1, y2 - y1), 3)
        item["candidate_to_fit_image_scale"] = [round(scale_x, 6), round(scale_y, 6)]
        scaled.append(item)
    return scaled


def _plain_hough_segments(image_bgr: Any) -> list[dict[str, Any]]:
    cv2 = _cv2()
    np = _np()
    image = _as_uint8_bgr(image_bgr)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 45, 145)
    height, width = gray.shape[:2]
    raw = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=38,
        minLineLength=max(24, int(round(width * 0.045))),
        maxLineGap=max(10, int(round(width * 0.018))),
    )
    if raw is None:
        return []
    segments: list[dict[str, Any]] = []
    for x1, y1, x2, y2 in raw.reshape(-1, 4):
        length = math.hypot(float(x2) - float(x1), float(y2) - float(y1))
        if length < 20.0:
            continue
        segments.append(
            {
                "p1": [float(x1), float(y1)],
                "p2": [float(x2), float(y2)],
                "length_px": round(length, 3),
                "angle_deg": round(math.degrees(math.atan2(float(y2) - float(y1), float(x2) - float(x1))), 3),
            }
        )
    segments.sort(key=lambda item: float(item["length_px"]), reverse=True)
    return segments[:96]


def _line_observations_from_segments(
    *,
    aggregated: Mapping[str, tuple[float, float]],
    segments: Sequence[Mapping[str, Any]],
    keypoint_by_name: Mapping[str, Any],
) -> list[dict[str, Any]]:
    return _line_observations_from_reference_points(
        reference_points=aggregated,
        segments=segments,
        keypoint_by_name=keypoint_by_name,
        reference_source="reviewed_keypoints",
    )


def _line_observations_from_projected_model(
    *,
    keypoint_names: Sequence[str],
    object_points_m: Sequence[Sequence[float]],
    model_fit: Mapping[str, Any],
    segments: Sequence[Mapping[str, Any]],
    keypoint_by_name: Mapping[str, Any],
) -> list[dict[str, Any]]:
    projected = project_world_points_with_distortion_fit(object_points_m, model_fit)
    if len(projected) != len(keypoint_names):
        raise ValueError("projected point count must match keypoint_names")
    reference_points = {
        str(name): _point2(projected[index], f"{name}.projected_image_px")
        for index, name in enumerate(keypoint_names)
    }
    return _line_observations_from_reference_points(
        reference_points=reference_points,
        segments=segments,
        keypoint_by_name=keypoint_by_name,
        reference_source="model_projection",
    )


def _line_observations_from_reference_points(
    *,
    reference_points: Mapping[str, Sequence[float]],
    segments: Sequence[Mapping[str, Any]],
    keypoint_by_name: Mapping[str, Any],
    reference_source: str,
) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    for line_name, (p1_name, p2_name) in FLOOR_LINE_KEYPOINT_PAIRS.items():
        if p1_name not in reference_points or p2_name not in reference_points:
            continue
        reference_p1 = _point2(reference_points[p1_name], f"{line_name}.reference_p1")
        reference_p2 = _point2(reference_points[p2_name], f"{line_name}.reference_p2")
        best = _best_segment_for_reviewed_line(reference_p1, reference_p2, segments)
        if best is None:
            continue
        support_mode = _line_observation_support_mode(line_name, best)
        if support_mode is None:
            continue
        observations.append(
            {
                "name": line_name,
                "world_line_m": [
                    list(keypoint_by_name[p1_name].world_xyz_m),
                    list(keypoint_by_name[p2_name].world_xyz_m),
                ],
                "image_segment_px": [best["p1"], best["p2"]],
                "sampled_image_points_px": _sample_segment_points(
                    best["p1"],
                    best["p2"],
                    sample_count=DEFAULT_LINE_PIXEL_SAMPLE_COUNT,
                ),
                "quality": _line_candidate_quality_payload(best),
                "support_mode": support_mode,
                "reference_source": str(reference_source),
            }
        )
    return observations


def _sample_segment_points(
    p1: Sequence[float],
    p2: Sequence[float],
    *,
    sample_count: int,
) -> list[list[float]]:
    if sample_count < 2:
        raise ValueError("sample_count must be at least 2")
    start = _point2(p1, "sample_segment.p1")
    end = _point2(p2, "sample_segment.p2")
    samples: list[list[float]] = []
    for index in range(sample_count):
        t = index / float(sample_count - 1)
        x = start[0] * (1.0 - t) + end[0] * t
        y = start[1] * (1.0 - t) + end[1] * t
        samples.append([round(x, 3), round(y, 3)])
    return samples


def _line_observation_support_mode(line_name: str, best: Mapping[str, Any]) -> str | None:
    angle_diff = float(best["angle_diff_deg"])
    mean_distance = float(best["mean_perpendicular_distance_px"])
    overlap = float(best["overlap_fraction"])
    if angle_diff <= 14.0 and mean_distance <= 22.0 and overlap >= 0.05:
        return "overlapping_segment"
    if line_name in {"near_centerline", "far_centerline"} and angle_diff <= 7.0 and mean_distance <= 26.0:
        return "centerline_collinear_segment"
    return None


def _line_candidate_quality_payload(candidate: Mapping[str, Any]) -> dict[str, float]:
    return {
        "angle_diff_deg": round(_finite_float(candidate.get("angle_diff_deg"), "line_candidate.angle_diff_deg"), 3),
        "mean_perpendicular_distance_px": round(
            _finite_float(candidate.get("mean_perpendicular_distance_px"), "line_candidate.mean_perpendicular_distance_px"),
            3,
        ),
        "overlap_fraction": round(_finite_float(candidate.get("overlap_fraction"), "line_candidate.overlap_fraction"), 6),
    }


def _best_segment_for_reviewed_line(
    reviewed_p1: tuple[float, float],
    reviewed_p2: tuple[float, float],
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    rx1, ry1 = reviewed_p1
    rx2, ry2 = reviewed_p2
    r_angle = math.degrees(math.atan2(ry2 - ry1, rx2 - rx1))
    best: dict[str, Any] | None = None
    best_cost = float("inf")
    for candidate in candidates:
        p1 = candidate.get("p1")
        p2 = candidate.get("p2")
        if not isinstance(p1, Sequence) or not isinstance(p2, Sequence) or len(p1) != 2 or len(p2) != 2:
            continue
        c1 = (float(p1[0]), float(p1[1]))
        c2 = (float(p2[0]), float(p2[1]))
        c_angle = math.degrees(math.atan2(c2[1] - c1[1], c2[0] - c1[0]))
        angle_diff = _angle_diff_mod_180(r_angle, c_angle)
        mean_distance = (_point_line_distance(c1, reviewed_p1, reviewed_p2) + _point_line_distance(c2, reviewed_p1, reviewed_p2)) / 2.0
        overlap = _segment_overlap_fraction(c1, c2, reviewed_p1, reviewed_p2)
        cost = mean_distance + angle_diff * 2.0 - overlap * 20.0
        if cost < best_cost:
            best_cost = cost
            best = {
                "p1": [round(c1[0], 3), round(c1[1], 3)],
                "p2": [round(c2[0], 3), round(c2[1], 3)],
                "angle_diff_deg": round(angle_diff, 3),
                "mean_perpendicular_distance_px": round(mean_distance, 3),
                "overlap_fraction": round(overlap, 4),
            }
    return best


def _point_line_distance(point: tuple[float, float], line_p1: tuple[float, float], line_p2: tuple[float, float]) -> float:
    x0, y0 = point
    x1, y1 = line_p1
    x2, y2 = line_p2
    numerator = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
    denominator = math.hypot(y2 - y1, x2 - x1)
    return numerator / denominator if denominator > 1e-6 else float("inf")


def _projection_t(point: tuple[float, float], line_p1: tuple[float, float], line_p2: tuple[float, float]) -> float:
    dx = line_p2[0] - line_p1[0]
    dy = line_p2[1] - line_p1[1]
    denom = dx * dx + dy * dy
    if denom <= 1e-6:
        return 0.0
    return ((point[0] - line_p1[0]) * dx + (point[1] - line_p1[1]) * dy) / denom


def _segment_overlap_fraction(
    c1: tuple[float, float],
    c2: tuple[float, float],
    reviewed_p1: tuple[float, float],
    reviewed_p2: tuple[float, float],
) -> float:
    t1 = _projection_t(c1, reviewed_p1, reviewed_p2)
    t2 = _projection_t(c2, reviewed_p1, reviewed_p2)
    low = max(0.0, min(t1, t2))
    high = min(1.0, max(t1, t2))
    return max(0.0, high - low)


def _image_segment_to_projected_line_residuals(projected_line: Any, image_segment: Any) -> list[float]:
    p0 = projected_line[0]
    p1 = projected_line[1]
    dx = float(p1[0] - p0[0])
    dy = float(p1[1] - p0[1])
    denom = math.hypot(dx, dy)
    if denom <= 1e-6:
        return [1000.0, 1000.0]
    residuals = []
    for point in image_segment:
        residuals.append(((float(point[0]) - float(p0[0])) * dy - (float(point[1]) - float(p0[1])) * dx) / denom)
    return residuals


def _rmse(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return math.sqrt(sum(float(value) * float(value) for value in values) / len(values))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cv2() -> Any:
    import cv2  # type: ignore[import-not-found]

    return cv2


def _np() -> Any:
    import numpy as np  # type: ignore[import-not-found]

    return np

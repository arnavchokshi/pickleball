"""Overlapping-court calibration helpers.

These utilities are intentionally opt-in. They target shared tennis/pickleball
courts where pickleball lines are painted in a distinct high-saturation color,
and they should fail closed when that assumption is false.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
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
FLOOR_LINE_KEYPOINT_PAIRS: dict[str, tuple[str, str]] = {
    "near_baseline": ("near_left_corner", "near_right_corner"),
    "far_baseline": ("far_left_corner", "far_right_corner"),
    "left_sideline": ("near_left_corner", "far_left_corner"),
    "right_sideline": ("near_right_corner", "far_right_corner"),
    "near_nvz": ("near_nvz_left", "near_nvz_right"),
    "far_nvz": ("far_nvz_left", "far_nvz_right"),
}


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
    return _camera_fit_payload(
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


def project_world_points_with_distortion_fit(
    object_points_m: Sequence[Sequence[float]],
    fit: Mapping[str, Any],
) -> list[list[float]]:
    """Project 3D world points using a fit payload returned by the joint solvers."""

    cv2 = _cv2()
    np = _np()
    obj = _object_points3(object_points_m)
    projected = _project_with_camera_params(
        cv2,
        np,
        obj,
        _camera_params_from_fit(fit),
        cx=float(fit["intrinsics"]["cx"]),
        cy=float(fit["intrinsics"]["cy"]),
    )
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
        metric_plane_residual_ft = _world_plane_residual_summary_ft(
            object_points_m,
            image_points,
            metric_plane_camera,
            meters_to_feet=1.0 / FT_TO_M,
        )
        metric_plane_residual_details_ft = _world_plane_residual_details_ft(
            FLOOR_HOMOGRAPHY_KEYPOINT_NAMES,
            object_points_m,
            image_points,
            metric_plane_camera,
            meters_to_feet=1.0 / FT_TO_M,
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
                    "fx": metric_plane_camera["intrinsics"]["fx"],
                    "k1": metric_plane_camera["distortion"]["k1"],
                    "k2": metric_plane_camera["distortion"]["k2"],
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
        "metric_plane_global_trimmed_worst8_mean_residual_ft": _trimmed_mean_or_none(
            metric_plane_all_keypoint_residuals,
            drop_worst_count=8,
        ),
        "metric_plane_global_trimmed_worst8_diagnostic_only": True,
        "metric_plane_per_clip_trimmed_worst3_mean_residual_ft_mean": (
            None if not metric_plane_trimmed_worst3_means else round(_mean(metric_plane_trimmed_worst3_means), 6)
        ),
        "point_line_fit_clip_count": len(point_line_rmses),
        "point_line_camera_rmse_px_mean": None if not point_line_rmses else round(_mean(point_line_rmses), 6),
        "point_line_camera_mean_residual_ft_mean": None if not point_line_world_means else round(_mean(point_line_world_means), 6),
        "point_line_camera_mean_residual_ft_median": None if not point_line_world_means else round(_median(point_line_world_means), 6),
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
        "partial_excluded": partial_excluded,
        "results": results,
        "notes": [
            "LM homography residuals are measured on reviewed court keypoint labels, not raw color-mask pixels.",
            "This evaluates the proposed LM refinement seam but does not promote no-tap CAL-3 calibration.",
            "Metric-plane trimmed residuals are diagnostic-only outlier analysis and must not be used as CAL pass criteria.",
        ],
    }
    if out_path is not None:
        output = Path(out_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


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
    image = np.asarray([_point2(point, "image_segment_px") for point in image_segment], dtype=np.float64)
    return world, image


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
    observations: list[dict[str, Any]] = []
    for line_name, (p1_name, p2_name) in FLOOR_LINE_KEYPOINT_PAIRS.items():
        reviewed_p1 = tuple(float(value) for value in aggregated[p1_name])
        reviewed_p2 = tuple(float(value) for value in aggregated[p2_name])
        best = _best_segment_for_reviewed_line(reviewed_p1, reviewed_p2, segments)
        if best is None:
            continue
        if not (
            float(best["angle_diff_deg"]) <= 14.0
            and float(best["mean_perpendicular_distance_px"]) <= 22.0
            and float(best["overlap_fraction"]) >= 0.05
        ):
            continue
        observations.append(
            {
                "name": line_name,
                "world_line_m": [
                    list(keypoint_by_name[p1_name].world_xyz_m),
                    list(keypoint_by_name[p2_name].world_xyz_m),
                ],
                "image_segment_px": [best["p1"], best["p2"]],
            }
        )
    return observations


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


def _cv2() -> Any:
    import cv2  # type: ignore[import-not-found]

    return cv2


def _np() -> Any:
    import numpy as np  # type: ignore[import-not-found]

    return np

"""Court keypoint detector primitives.

This module contains the deterministic validation, geometry, and lightweight
heatmap helpers used by the court-keypoint training and calibration paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import math
from typing import Any, Mapping, Sequence

from threed.racketsport.court_calibration import homography_from_planar_points, project_planar_points
from threed.racketsport.court_templates import ft_to_m, get_court_template, xyz_ft_to_m


@dataclass(frozen=True)
class CourtKeypoint:
    name: str
    world_xyz_m: tuple[float, float, float]
    description: str


@dataclass(frozen=True)
class SyntheticRenderConfig:
    viewpoint_count: int
    height_m: tuple[float, float]
    tilt_deg: tuple[float, float]
    focal_mm_eq: tuple[float, float]
    image_size: tuple[int, int]
    augmentations: tuple[str, ...]


@dataclass(frozen=True)
class TrainingPlanConfig:
    synthetic: SyntheticRenderConfig
    tennis_frames: int
    pickleball_frames: int
    device: str
    checkpoint_policy: str


@dataclass(frozen=True)
class HeatmapDecode:
    x: float
    y: float
    score: float


@dataclass(frozen=True)
class KeypointPrediction:
    image_xy: tuple[float, float]
    confidence: float
    heatmap_score: float | None = None


@dataclass(frozen=True)
class SolvePnPCorrespondences:
    keypoint_names: tuple[str, ...]
    object_points_m: tuple[tuple[float, float, float], ...]
    image_points_px: tuple[tuple[float, float], ...]
    confidence: tuple[float, ...]


def make_court_keypoint_heatmap_model(keypoint_count: int, *, architecture: str = "encoder_decoder_v1") -> Any:
    if isinstance(keypoint_count, bool) or not isinstance(keypoint_count, int) or keypoint_count <= 0:
        raise ValueError("keypoint_count must be a positive integer")
    if architecture not in {"encoder_decoder_v1", "local_conv_v1"}:
        raise ValueError("unsupported court keypoint heatmap architecture")
    import torch.nn as nn
    import torch.nn.functional as F

    if architecture == "local_conv_v1":
        return nn.Sequential(
            nn.Conv2d(3, 24, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(24, 48, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(48, 48, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(48, keypoint_count, 1),
        )

    class CourtKeypointEncoderDecoder(nn.Module):
        def __init__(self, outputs: int) -> None:
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 96, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(96, 128, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(128, 128, 3, padding=1),
                nn.ReLU(),
            )
            self.decoder = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(128, 96, 3, padding=1),
                nn.ReLU(),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(96, 64, 3, padding=1),
                nn.ReLU(),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(64, 32, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, outputs, 1),
            )

        def forward(self, x: Any) -> Any:
            logits = self.decoder(self.encoder(x))
            if logits.shape[-2:] != x.shape[-2:]:
                logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
            return logits

    return CourtKeypointEncoderDecoder(keypoint_count)


def court_keypoint_probabilities(logits: Any, *, activation: str = "spatial_softmax") -> Any:
    if activation == "sigmoid":
        return logits.sigmoid()
    if activation != "spatial_softmax":
        raise ValueError("unsupported court keypoint heatmap activation")
    import torch.nn.functional as F

    if logits.ndim < 3:
        return F.softmax(logits, dim=-1)
    flat = logits.reshape(*logits.shape[:-2], -1)
    return F.softmax(flat, dim=-1).reshape_as(logits)


def court_keypoint_heatmap_loss(logits: Any, target: Any, mask: Any, *, foreground_weight: float = 20.0) -> Any:
    if foreground_weight <= 0:
        raise ValueError("foreground_weight must be positive")
    import torch.nn.functional as F

    target = target.clamp(0.0, 1.0)
    target_flat = target.reshape(*target.shape[:-2], -1)
    logits_flat = logits.reshape(*logits.shape[:-2], -1)
    mask_flat = mask.reshape(*mask.shape[:-2], -1)
    channel_mask = (mask_flat.amax(dim=-1) > 0).to(logits.dtype)
    target_distribution = target_flat / target_flat.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    loss = -(target_distribution * F.log_softmax(logits_flat, dim=-1)).sum(dim=-1)
    return (loss * channel_mask).sum() / channel_mask.sum().clamp_min(1.0)


_PICKLEBALL_TEMPLATE = get_court_template("pickleball")


def _pickleball_net_keypoint_xyz_ft(x_ft: float) -> tuple[float, float, float]:
    return (
        ft_to_m(x_ft),
        0.0,
        _PICKLEBALL_TEMPLATE.post_net_height_m,
    )


PICKLEBALL_KEYPOINTS: tuple[CourtKeypoint, ...] = (
    CourtKeypoint("near_left_corner", tuple(xyz_ft_to_m(-10.0, -22.0)), "near baseline left sideline corner"),
    CourtKeypoint("near_baseline_center", tuple(xyz_ft_to_m(0.0, -22.0)), "near baseline at centerline"),
    CourtKeypoint("near_right_corner", tuple(xyz_ft_to_m(10.0, -22.0)), "near baseline right sideline corner"),
    CourtKeypoint("far_right_corner", tuple(xyz_ft_to_m(10.0, 22.0)), "far baseline right sideline corner"),
    CourtKeypoint("far_baseline_center", tuple(xyz_ft_to_m(0.0, 22.0)), "far baseline at centerline"),
    CourtKeypoint("far_left_corner", tuple(xyz_ft_to_m(-10.0, 22.0)), "far baseline left sideline corner"),
    CourtKeypoint("near_nvz_left", tuple(xyz_ft_to_m(-10.0, -7.0)), "near NVZ line at left sideline"),
    CourtKeypoint("near_nvz_center", tuple(xyz_ft_to_m(0.0, -7.0)), "near NVZ line at centerline"),
    CourtKeypoint("near_nvz_right", tuple(xyz_ft_to_m(10.0, -7.0)), "near NVZ line at right sideline"),
    CourtKeypoint("net_left_sideline", _pickleball_net_keypoint_xyz_ft(-10.0), "net top at left sideline"),
    CourtKeypoint("net_center", _pickleball_net_keypoint_xyz_ft(0.0), "net top at centerline"),
    CourtKeypoint("net_right_sideline", _pickleball_net_keypoint_xyz_ft(10.0), "net top at right sideline"),
    CourtKeypoint("far_nvz_left", tuple(xyz_ft_to_m(-10.0, 7.0)), "far NVZ line at left sideline"),
    CourtKeypoint("far_nvz_center", tuple(xyz_ft_to_m(0.0, 7.0)), "far NVZ line at centerline"),
    CourtKeypoint("far_nvz_right", tuple(xyz_ft_to_m(10.0, 7.0)), "far NVZ line at right sideline"),
)

PICKLEBALL_KEYPOINT_BY_NAME: dict[str, CourtKeypoint] = {point.name: point for point in PICKLEBALL_KEYPOINTS}
COURT_CORNER_LABEL_TO_KEYPOINT: dict[str, str] = {
    "near_left": "near_left_corner",
    "near_right": "near_right_corner",
    "far_right": "far_right_corner",
    "far_left": "far_left_corner",
}

DEFAULT_SYNTHETIC_RENDER_CONFIG = SyntheticRenderConfig(
    viewpoint_count=200,
    height_m=(1.0, 4.0),
    tilt_deg=(10.0, 80.0),
    focal_mm_eq=(28.0, 90.0),
    image_size=(1920, 1080),
    augmentations=("colors", "shadows", "glare", "occluded_corners"),
)

ALLOWED_TRAINING_DEVICES = {"cpu", "cuda", "mps"}
ALLOWED_CHECKPOINT_POLICIES = {
    "none",
    "local_only",
    "scratch",
    "courtkeynet_mit",
    "download_tenniscourtdetector",
    "ultralytics_yolo11_pose",
}


def validate_synthetic_render_config(config: Mapping[str, Any]) -> SyntheticRenderConfig:
    """Validate the CAL-3 synthetic-render recipe bounds without rendering."""

    viewpoint_count = _int_field(config, "viewpoint_count", DEFAULT_SYNTHETIC_RENDER_CONFIG.viewpoint_count)
    if not 50 <= viewpoint_count <= 500:
        raise ValueError("viewpoint_count must be in the CAL-3 synthetic recipe range [50, 500]")

    height_m = _range_field(config, "height_m", DEFAULT_SYNTHETIC_RENDER_CONFIG.height_m)
    if height_m[0] < 1.0 or height_m[1] > 4.0:
        raise ValueError("height_m must stay within the recipe range [1.0, 4.0]")

    tilt_deg = _range_field(config, "tilt_deg", DEFAULT_SYNTHETIC_RENDER_CONFIG.tilt_deg)
    if tilt_deg[0] < 10.0 or tilt_deg[1] > 80.0:
        raise ValueError("tilt_deg must stay within the recipe range [10.0, 80.0]")

    focal_mm_eq = _range_field(config, "focal_mm_eq", DEFAULT_SYNTHETIC_RENDER_CONFIG.focal_mm_eq)
    if focal_mm_eq[0] < 28.0 or focal_mm_eq[1] > 90.0:
        raise ValueError("focal_mm_eq must stay within the recipe range [28.0, 90.0]")

    image_size = _image_size_field(config, "image_size", DEFAULT_SYNTHETIC_RENDER_CONFIG.image_size)
    augmentations = _string_tuple_field(
        config,
        "augmentations",
        DEFAULT_SYNTHETIC_RENDER_CONFIG.augmentations,
    )

    return SyntheticRenderConfig(
        viewpoint_count=viewpoint_count,
        height_m=height_m,
        tilt_deg=tilt_deg,
        focal_mm_eq=focal_mm_eq,
        image_size=image_size,
        augmentations=augmentations,
    )


def validate_training_plan_config(config: Mapping[str, Any]) -> TrainingPlanConfig:
    """Validate a court-keypoint training-plan description."""

    synthetic_raw = config.get("synthetic", DEFAULT_SYNTHETIC_RENDER_CONFIG)
    if isinstance(synthetic_raw, SyntheticRenderConfig):
        synthetic = synthetic_raw
    elif isinstance(synthetic_raw, Mapping):
        synthetic = validate_synthetic_render_config(synthetic_raw)
    else:
        raise ValueError("synthetic must be a SyntheticRenderConfig or mapping")

    tennis_frames = _int_field(config, "tennis_frames", 8800)
    if tennis_frames < 0:
        raise ValueError("tennis_frames must be non-negative")

    pickleball_frames = _int_field(config, "pickleball_frames", 300)
    if not 200 <= pickleball_frames <= 500:
        raise ValueError("pickleball_frames must be in the fine-tune recipe range [200, 500]")

    device = str(config.get("device", "cpu")).lower()
    if device not in ALLOWED_TRAINING_DEVICES:
        allowed = ", ".join(sorted(ALLOWED_TRAINING_DEVICES))
        raise ValueError(f"device must be one of: {allowed}")

    checkpoint_policy = str(config.get("checkpoint_policy", "none"))
    if checkpoint_policy not in ALLOWED_CHECKPOINT_POLICIES:
        allowed = ", ".join(sorted(ALLOWED_CHECKPOINT_POLICIES))
        raise ValueError(f"checkpoint_policy must be one of: {allowed}")

    return TrainingPlanConfig(
        synthetic=synthetic,
        tennis_frames=tennis_frames,
        pickleball_frames=pickleball_frames,
        device=device,
        checkpoint_policy=checkpoint_policy,
    )


def keypoint_labels_from_court_corners(corners: Mapping[str, Any]) -> dict[str, list[float]]:
    """Expand four labeled court corners into the full pickleball keypoint layout."""

    world_pts: list[tuple[float, float, float]] = []
    image_pts: list[tuple[float, float]] = []
    for corner_name, keypoint_name in COURT_CORNER_LABEL_TO_KEYPOINT.items():
        keypoint = PICKLEBALL_KEYPOINT_BY_NAME[keypoint_name]
        image_xy = _xy_field(corners.get(corner_name), f"court_corners.{corner_name}")
        world_pts.append(keypoint.world_xyz_m)
        image_pts.append(image_xy)

    homography = homography_from_planar_points(world_pts, image_pts)
    projected = project_planar_points(homography, [point.world_xyz_m for point in PICKLEBALL_KEYPOINTS])
    return {
        point.name: [float(projected_xy[0]), float(projected_xy[1])]
        for point, projected_xy in zip(PICKLEBALL_KEYPOINTS, projected, strict=True)
    }


def refine_keypoint_xy_with_planar_homography(
    keypoints: Mapping[str, Sequence[float]],
    *,
    max_inlier_error_px: float = 30.0,
    min_inliers: int = 8,
) -> dict[str, list[float]]:
    """Correct scattered per-channel outliers using regulation court geometry.

    The heatmap model predicts each keypoint channel independently, but a valid court
    must be explained by one planar homography. When enough raw predictions agree on
    that geometry, this projects the canonical 15-point template through the consensus
    homography. If consensus is weak, raw predictions are returned unchanged.
    """

    if max_inlier_error_px <= 0:
        raise ValueError("max_inlier_error_px must be positive")
    if min_inliers < 4:
        raise ValueError("min_inliers must be at least 4")

    candidates: list[tuple[str, tuple[float, float], tuple[float, float]]] = []
    for point in PICKLEBALL_KEYPOINTS:
        xy = keypoints.get(point.name)
        if xy is None:
            continue
        candidates.append(
            (
                point.name,
                (point.world_xyz_m[0], point.world_xyz_m[1]),
                _xy_field(xy, f"{point.name}.xy"),
            )
        )
    raw = {name: [image_xy[0], image_xy[1]] for name, _, image_xy in candidates}
    if len(candidates) < max(4, min_inliers):
        return raw

    best: tuple[int, float, tuple[str, ...], list[list[float]]] | None = None
    all_world = [item[1] for item in candidates]
    for subset in combinations(candidates, 4):
        world_subset = [item[1] for item in subset]
        image_subset = [item[2] for item in subset]
        if _max_triangle_area(world_subset) <= 1e-6 or _max_triangle_area(image_subset) <= 1e-3:
            continue
        try:
            homography = homography_from_planar_points(world_subset, image_subset)
            projected = project_planar_points(homography, all_world)
        except ValueError:
            continue

        residuals = [
            math.hypot(float(projected_xy[0]) - image_xy[0], float(projected_xy[1]) - image_xy[1])
            for projected_xy, (_, _, image_xy) in zip(projected, candidates, strict=True)
        ]
        inlier_indexes = [idx for idx, residual in enumerate(residuals) if residual <= max_inlier_error_px]
        if len(inlier_indexes) < min_inliers:
            continue
        mean_inlier_error = sum(residuals[idx] for idx in inlier_indexes) / len(inlier_indexes)
        inlier_names = tuple(candidates[idx][0] for idx in inlier_indexes)
        candidate = (len(inlier_indexes), mean_inlier_error, inlier_names, homography)
        if best is None or candidate[0] > best[0] or (candidate[0] == best[0] and candidate[1] < best[1]):
            best = candidate

    if best is None:
        return raw

    inlier_name_set = set(best[2])
    inlier_candidates = [item for item in candidates if item[0] in inlier_name_set]
    homography = best[3]
    try:
        homography = homography_from_planar_points(
            [item[1] for item in inlier_candidates],
            [item[2] for item in inlier_candidates],
        )
        projected_all = project_planar_points(
            homography,
            [(point.world_xyz_m[0], point.world_xyz_m[1]) for point in PICKLEBALL_KEYPOINTS],
        )
    except ValueError:
        return raw

    return {
        point.name: [float(projected_xy[0]), float(projected_xy[1])]
        for point, projected_xy in zip(PICKLEBALL_KEYPOINTS, projected_all, strict=True)
    }


def decode_subpixel_heatmap(heatmap: Sequence[Sequence[float]]) -> HeatmapDecode:
    """Decode a 2D heatmap with a local parabolic subpixel refinement."""

    rows = _rectangular_finite_heatmap(heatmap)
    max_y = 0
    max_x = 0
    max_score = rows[0][0]
    for y, row in enumerate(rows):
        for x, value in enumerate(row):
            if value > max_score:
                max_y = y
                max_x = x
                max_score = value

    x = float(max_x) + _axis_offset_x(rows, max_y, max_x)
    y = float(max_y) + _axis_offset_y(rows, max_y, max_x)
    return HeatmapDecode(x=x, y=y, score=max_score)


def validate_heatmap_prediction_payload(payload: Mapping[str, Any]) -> dict[str, KeypointPrediction]:
    """Validate predicted heatmaps or direct image points keyed by taxonomy name."""

    keypoints = payload.get("keypoints")
    if not isinstance(keypoints, Mapping):
        raise ValueError("payload must contain a keypoints mapping")

    predictions: dict[str, KeypointPrediction] = {}
    for name, raw_prediction in keypoints.items():
        if name not in PICKLEBALL_KEYPOINT_BY_NAME:
            raise ValueError(f"unknown keypoint: {name}")
        if not isinstance(raw_prediction, Mapping):
            raise ValueError(f"{name} prediction must be a mapping")

        confidence = _confidence(raw_prediction.get("confidence", 1.0), f"{name}.confidence")
        heatmap_score = None
        if "heatmap" in raw_prediction:
            decoded = decode_subpixel_heatmap(raw_prediction["heatmap"])
            image_xy = (decoded.x, decoded.y)
            heatmap_score = decoded.score
        elif "xy" in raw_prediction:
            image_xy = _xy_field(raw_prediction["xy"], f"{name}.xy")
        else:
            raise ValueError(f"{name} must include either heatmap or xy")

        predictions[str(name)] = KeypointPrediction(
            image_xy=image_xy,
            confidence=confidence,
            heatmap_score=heatmap_score,
        )

    return predictions


def keypoints_to_solvepnp_correspondences(
    predictions: Mapping[str, KeypointPrediction],
    *,
    min_confidence: float = 0.5,
) -> SolvePnPCorrespondences:
    """Convert validated predictions into ordered object/image point pairs."""

    threshold = _confidence(min_confidence, "min_confidence")
    names: list[str] = []
    object_points: list[tuple[float, float, float]] = []
    image_points: list[tuple[float, float]] = []
    confidences: list[float] = []

    for keypoint in PICKLEBALL_KEYPOINTS:
        prediction = predictions.get(keypoint.name)
        if prediction is None or prediction.confidence < threshold:
            continue
        names.append(keypoint.name)
        object_points.append(keypoint.world_xyz_m)
        image_points.append(prediction.image_xy)
        confidences.append(prediction.confidence)

    if len(names) < 4:
        raise ValueError("solvePnP correspondences require at least 4 confident keypoints")

    return SolvePnPCorrespondences(
        keypoint_names=tuple(names),
        object_points_m=tuple(object_points),
        image_points_px=tuple(image_points),
        confidence=tuple(confidences),
    )


def _int_field(config: Mapping[str, Any], name: str, default: int) -> int:
    value = config.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _range_field(config: Mapping[str, Any], name: str, default: tuple[float, float]) -> tuple[float, float]:
    raw_value = config.get(name, default)
    if not isinstance(raw_value, Sequence) or isinstance(raw_value, (str, bytes)) or len(raw_value) != 2:
        raise ValueError(f"{name} must be a two-item range")
    low = _finite_float(raw_value[0], name)
    high = _finite_float(raw_value[1], name)
    if low > high:
        raise ValueError(f"{name} range must be ascending")
    return (low, high)


def _image_size_field(config: Mapping[str, Any], name: str, default: tuple[int, int]) -> tuple[int, int]:
    raw_value = config.get(name, default)
    if not isinstance(raw_value, Sequence) or isinstance(raw_value, (str, bytes)) or len(raw_value) != 2:
        raise ValueError(f"{name} must be [width, height]")
    width = raw_value[0]
    height = raw_value[1]
    if isinstance(width, bool) or isinstance(height, bool) or not isinstance(width, int) or not isinstance(height, int):
        raise ValueError(f"{name} must contain integer width and height")
    if width <= 0 or height <= 0:
        raise ValueError(f"{name} must contain positive dimensions")
    return (width, height)


def _string_tuple_field(config: Mapping[str, Any], name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw_value = config.get(name, default)
    if not isinstance(raw_value, Sequence) or isinstance(raw_value, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of non-empty strings")
    values = tuple(raw_value)
    if not values or any(not isinstance(item, str) or not item for item in values):
        raise ValueError(f"{name} must contain non-empty strings")
    return values


def _rectangular_finite_heatmap(heatmap: Sequence[Sequence[float]]) -> tuple[tuple[float, ...], ...]:
    if not isinstance(heatmap, Sequence) or isinstance(heatmap, (str, bytes)) or not heatmap:
        raise ValueError("heatmap must be a non-empty rectangular 2D sequence")

    rows: list[tuple[float, ...]] = []
    expected_width: int | None = None
    for row in heatmap:
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)) or not row:
            raise ValueError("heatmap must be a non-empty rectangular 2D sequence")
        values = tuple(_finite_float(value, "heatmap value") for value in row)
        if expected_width is None:
            expected_width = len(values)
        elif len(values) != expected_width:
            raise ValueError("heatmap must be rectangular")
        rows.append(values)
    return tuple(rows)


def _axis_offset_x(rows: tuple[tuple[float, ...], ...], y: int, x: int) -> float:
    if x <= 0 or x >= len(rows[0]) - 1:
        return 0.0
    return _parabolic_offset(rows[y][x - 1], rows[y][x], rows[y][x + 1])


def _axis_offset_y(rows: tuple[tuple[float, ...], ...], y: int, x: int) -> float:
    if y <= 0 or y >= len(rows) - 1:
        return 0.0
    return _parabolic_offset(rows[y - 1][x], rows[y][x], rows[y + 1][x])


def _parabolic_offset(left: float, center: float, right: float) -> float:
    denominator = left - (2.0 * center) + right
    if denominator >= 0.0 or math.isclose(denominator, 0.0):
        return 0.0
    offset = 0.5 * (left - right) / denominator
    return max(-0.5, min(0.5, offset))


def _max_triangle_area(points: Sequence[tuple[float, float]]) -> float:
    best = 0.0
    for a, b, c in combinations(points, 3):
        area = abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])) / 2.0
        best = max(best, area)
    return best


def _xy_field(raw_value: Any, name: str) -> tuple[float, float]:
    if not isinstance(raw_value, Sequence) or isinstance(raw_value, (str, bytes)) or len(raw_value) != 2:
        raise ValueError(f"{name} must be a two-item image coordinate")
    return (_finite_float(raw_value[0], name), _finite_float(raw_value[1], name))


def _confidence(value: Any, name: str) -> float:
    confidence = _finite_float(value, name)
    if not 0.0 <= confidence <= 1.0:
        raise ValueError(f"{name} confidence must be in [0, 1]")
    return confidence


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result

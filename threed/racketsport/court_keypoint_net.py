"""CPU-only court keypoint detector scaffold primitives.

This module intentionally does not train or load a model. It provides the
deterministic validation and geometry helpers that a future CAL-3 training
script can use before handing correspondences to the existing calibration path.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

from threed.racketsport.court_templates import xyz_ft_to_m


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


PICKLEBALL_KEYPOINTS: tuple[CourtKeypoint, ...] = (
    CourtKeypoint("near_left_corner", tuple(xyz_ft_to_m(-10.0, -22.0)), "near baseline left sideline corner"),
    CourtKeypoint("near_right_corner", tuple(xyz_ft_to_m(10.0, -22.0)), "near baseline right sideline corner"),
    CourtKeypoint("far_right_corner", tuple(xyz_ft_to_m(10.0, 22.0)), "far baseline right sideline corner"),
    CourtKeypoint("far_left_corner", tuple(xyz_ft_to_m(-10.0, 22.0)), "far baseline left sideline corner"),
    CourtKeypoint("near_nvz_left", tuple(xyz_ft_to_m(-10.0, -7.0)), "near NVZ line at left sideline"),
    CourtKeypoint("near_nvz_center", tuple(xyz_ft_to_m(0.0, -7.0)), "near NVZ line at centerline"),
    CourtKeypoint("near_nvz_right", tuple(xyz_ft_to_m(10.0, -7.0)), "near NVZ line at right sideline"),
    CourtKeypoint("net_left_sideline", tuple(xyz_ft_to_m(-10.0, 0.0)), "net line at left sideline"),
    CourtKeypoint("net_center", tuple(xyz_ft_to_m(0.0, 0.0)), "net line at centerline"),
    CourtKeypoint("net_right_sideline", tuple(xyz_ft_to_m(10.0, 0.0)), "net line at right sideline"),
    CourtKeypoint("far_nvz_left", tuple(xyz_ft_to_m(-10.0, 7.0)), "far NVZ line at left sideline"),
    CourtKeypoint("far_nvz_center", tuple(xyz_ft_to_m(0.0, 7.0)), "far NVZ line at centerline"),
    CourtKeypoint("far_nvz_right", tuple(xyz_ft_to_m(10.0, 7.0)), "far NVZ line at right sideline"),
)

PICKLEBALL_KEYPOINT_BY_NAME: dict[str, CourtKeypoint] = {point.name: point for point in PICKLEBALL_KEYPOINTS}

DEFAULT_SYNTHETIC_RENDER_CONFIG = SyntheticRenderConfig(
    viewpoint_count=200,
    height_m=(1.0, 4.0),
    tilt_deg=(10.0, 80.0),
    focal_mm_eq=(28.0, 90.0),
    image_size=(1920, 1080),
    augmentations=("colors", "shadows", "glare", "occluded_corners"),
)

ALLOWED_CHECKPOINT_POLICIES = {"none", "local_only", "scratch"}


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
    """Validate a CPU-only training-plan description for future orchestration."""

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
    if device != "cpu":
        raise ValueError("device must be 'cpu' for this scaffold")

    checkpoint_policy = str(config.get("checkpoint_policy", "none"))
    if checkpoint_policy not in ALLOWED_CHECKPOINT_POLICIES:
        raise ValueError("checkpoint_policy must not download or select external checkpoints")

    return TrainingPlanConfig(
        synthetic=synthetic,
        tennis_frames=tennis_frames,
        pickleball_frames=pickleball_frames,
        device=device,
        checkpoint_policy=checkpoint_policy,
    )


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
    values = tuple(str(item) for item in raw_value)
    if not values or any(not item for item in values):
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

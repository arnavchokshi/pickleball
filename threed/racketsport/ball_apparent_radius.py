"""Deterministic image-based apparent-radius measurements for tracked balls.

This candidate is deliberately independent of detector heatmaps and pipeline
solvers.  It selects a current-frame Hough circle using an adaptive Lab
foreground extent, then validates that circle against the minor-axis spread
and velocity alignment of a local frame-difference blob.  The temporal minor
axis is insensitive to inter-frame displacement along the major axis, while
the current-frame shape gate makes motion blur an explicit abstention instead
of silently inflating the radius.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any, Literal

import numpy as np


TEMPORAL_METHOD = "hough_circle_temporal_blob_gated_v1"
SINGLE_FRAME_METHOD = "hough_circle_edges_v1"


class RadiusSkipReason(StrEnum):
    INVALID_FRAME = "invalid_frame"
    INVALID_CENTER = "invalid_center"
    CROP_TRUNCATED = "crop_truncated"
    FOREGROUND_NOT_SEPARABLE = "foreground_not_separable"
    MOTION_BLUR = "motion_blur"
    NO_SUPPORTED_CIRCLE = "no_supported_circle"
    TEMPORAL_FOREGROUND_WEAK = "temporal_foreground_weak"
    TEMPORAL_FOREGROUND_MISALIGNED = "temporal_foreground_misaligned"


@dataclass(frozen=True)
class ApparentRadiusConfig:
    crop_radius_px: int = 24
    min_radius_px: int = 3
    max_radius_px: int = 16
    max_center_refinement_px: float = 8.0
    hough_canny_high: float = 60.0
    hough_accumulator_threshold: float = 10.0
    foreground_lab_margin: float = 5.0
    max_foreground_axis_ratio: float = 2.0
    min_foreground_pixels: int = 8
    temporal_min_abs_delta: float = 12.0
    temporal_threshold_percentile: float = 95.0
    temporal_min_points: int = 8
    max_temporal_velocity_angle_delta_deg: float = 30.0


@dataclass(frozen=True)
class ApparentRadiusMeasurement:
    status: Literal["measured", "skipped"]
    input_center_xy_px: tuple[float, float]
    center_xy_px: tuple[float, float] | None
    radius_px: float | None
    confidence: float
    method: str
    skip_reason: RadiusSkipReason | None
    crop_xyxy_px: tuple[int, int, int, int] | None
    gates: dict[str, float | int | bool | str | None]
    provenance: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.skip_reason is not None:
            payload["skip_reason"] = self.skip_reason.value
        return payload


def estimate_apparent_radius(
    frame: np.ndarray,
    ball_center_xy: tuple[float, float],
    *,
    previous_frame: np.ndarray | None = None,
    previous_ball_center_xy: tuple[float, float] | None = None,
    config: ApparentRadiusConfig | None = None,
) -> ApparentRadiusMeasurement:
    """Measure apparent radius from image pixels near an existing 2D center.

    A previous frame and center enable preferred temporal validation. Without
    them, the supported current-frame Hough radius is returned. Every rejection
    has a typed reason and no rejected result carries a radius.
    """

    cfg = config or ApparentRadiusConfig()
    center = _finite_center(ball_center_xy)
    if center is None:
        return _skipped(ball_center_xy, RadiusSkipReason.INVALID_CENTER)
    array = np.asarray(frame)
    if array.ndim != 3 or array.shape[2] != 3 or array.dtype != np.uint8:
        return _skipped(center, RadiusSkipReason.INVALID_FRAME)
    crop_data = _crop(array, center, cfg.crop_radius_px)
    if crop_data is None:
        return _skipped(center, RadiusSkipReason.CROP_TRUNCATED)
    crop, crop_box, local_center = crop_data
    cv2 = _cv2()

    shape = _seeded_foreground_shape(crop, local_center, cfg, cv2)
    if shape is None:
        return _skipped(
            center,
            RadiusSkipReason.FOREGROUND_NOT_SEPARABLE,
            crop_box=crop_box,
        )
    if float(shape["axis_ratio"]) > cfg.max_foreground_axis_ratio:
        return _skipped(
            center,
            RadiusSkipReason.MOTION_BLUR,
            crop_box=crop_box,
            gates={
                "foreground_axis_ratio": float(shape["axis_ratio"]),
                "max_foreground_axis_ratio": cfg.max_foreground_axis_ratio,
            },
        )

    circle = _supported_hough_circle(
        crop,
        local_center,
        float(shape["area_radius_px"]),
        cfg,
        cv2,
    )
    if circle is None:
        return _skipped(
            center,
            RadiusSkipReason.NO_SUPPORTED_CIRCLE,
            crop_box=crop_box,
            gates={"foreground_axis_ratio": float(shape["axis_ratio"])},
        )
    circle_x, circle_y, hough_radius = circle
    measured_center = (float(crop_box[0] + circle_x), float(crop_box[1] + circle_y))
    center_residual = math.hypot(measured_center[0] - center[0], measured_center[1] - center[1])

    temporal = None
    if previous_frame is not None and previous_ball_center_xy is not None:
        previous = np.asarray(previous_frame)
        previous_center = _finite_center(previous_ball_center_xy)
        if previous.shape == array.shape and previous.dtype == np.uint8 and previous_center is not None:
            previous_crop = previous[crop_box[1] : crop_box[3], crop_box[0] : crop_box[2]]
            temporal = _temporal_minor_axis(
                crop,
                previous_crop,
                center,
                previous_center,
                cfg,
                cv2,
            )

    common_gates: dict[str, float | int | bool | str | None] = {
        "foreground_axis_ratio": float(shape["axis_ratio"]),
        "foreground_pixel_count": int(shape["pixel_count"]),
        "center_refinement_px": center_residual,
        "hough_radius_px": float(hough_radius),
    }
    if temporal is not None and temporal.get("reason") is not None:
        reason = RadiusSkipReason(str(temporal["reason"]))
        return _skipped(
            center,
            reason,
            crop_box=crop_box,
            gates={**common_gates, **temporal["gates"]},
        )

    contrast = float(shape["lab_contrast"])
    shape_score = float(np.clip((cfg.max_foreground_axis_ratio - float(shape["axis_ratio"])) / (cfg.max_foreground_axis_ratio - 1.0), 0.0, 1.0))
    center_score = float(np.clip(1.0 - center_residual / cfg.max_center_refinement_px, 0.0, 1.0))
    contrast_score = float(np.clip(contrast / 60.0, 0.0, 1.0))

    if temporal is not None:
        radius = float(hough_radius)
        temporal_radius = float(temporal["radius_proxy_px"])
        alignment_score = float(np.clip(1.0 - float(temporal["angle_delta_deg"]) / cfg.max_temporal_velocity_angle_delta_deg, 0.0, 1.0))
        radius_agreement = float(np.clip(1.0 - abs(radius - temporal_radius) / max(radius, temporal_radius, 1.0), 0.0, 1.0))
        confidence = float(np.clip(0.25 * shape_score + 0.20 * center_score + 0.20 * contrast_score + 0.20 * alignment_score + 0.15 * radius_agreement, 0.0, 1.0))
        gates = {
            **common_gates,
            **temporal["gates"],
            "radius_agreement": radius_agreement,
        }
        method = TEMPORAL_METHOD
        provenance = {
            "radius_source": "current-frame grayscale gradient Hough circle selected by adaptive foreground extent",
            "temporal_validation": "minor-axis covariance and velocity alignment of thresholded local absolute frame difference",
            "sharpness_gate": "adaptive Lab foreground axis ratio plus current-frame Hough support",
            "heatmap_used": False,
        }
    else:
        radius = float(hough_radius)
        confidence = float(np.clip(0.40 * shape_score + 0.30 * center_score + 0.30 * contrast_score, 0.0, 1.0))
        gates = common_gates
        method = SINGLE_FRAME_METHOD
        provenance = {
            "radius_source": "current-frame grayscale gradient Hough circle",
            "sharpness_gate": "adaptive Lab foreground axis ratio",
            "heatmap_used": False,
        }

    return ApparentRadiusMeasurement(
        status="measured",
        input_center_xy_px=center,
        center_xy_px=measured_center,
        radius_px=radius,
        confidence=confidence,
        method=method,
        skip_reason=None,
        crop_xyxy_px=crop_box,
        gates=gates,
        provenance=provenance,
    )


def _seeded_foreground_shape(
    crop: np.ndarray,
    local_center: tuple[float, float],
    cfg: ApparentRadiusConfig,
    cv2: Any,
) -> dict[str, float | int] | None:
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float64)
    yy, xx = np.mgrid[: crop.shape[0], : crop.shape[1]]
    radial = np.hypot(xx - local_center[0], yy - local_center[1])
    seed_pixels = lab[radial <= 2.2]
    background_pixels = lab[(radial >= cfg.crop_radius_px - 6) & (radial <= cfg.crop_radius_px - 1)]
    if len(seed_pixels) == 0 or len(background_pixels) == 0:
        return None
    seed = np.median(seed_pixels, axis=0)
    background = np.median(background_pixels, axis=0)
    seed_distance = np.linalg.norm(lab - seed, axis=2)
    background_distance = np.linalg.norm(lab - background, axis=2)
    mask = ((seed_distance + cfg.foreground_lab_margin < background_distance) & (radial <= cfg.max_radius_px + 2)).astype(np.uint8)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    candidates: list[tuple[float, int, int]] = []
    for label in range(1, int(count)):
        pixel_count = int(stats[label, cv2.CC_STAT_AREA])
        distance = math.hypot(
            float(centroids[label][0]) - local_center[0],
            float(centroids[label][1]) - local_center[1],
        )
        if pixel_count >= cfg.min_foreground_pixels and distance <= cfg.max_center_refinement_px:
            candidates.append((distance, -pixel_count, label))
    if not candidates:
        return None
    label = min(candidates)[2]
    ys, xs = np.where(labels == label)
    covariance = np.cov(np.column_stack([xs, ys]), rowvar=False, bias=True)
    eigenvalues = np.linalg.eigvalsh(covariance)
    if len(eigenvalues) != 2 or float(eigenvalues[0]) <= 1e-9:
        return None
    return {
        "axis_ratio": math.sqrt(float(eigenvalues[1]) / float(eigenvalues[0])),
        "pixel_count": len(xs),
        "lab_contrast": float(np.linalg.norm(seed - background)),
        "area_radius_px": math.sqrt(len(xs) / math.pi),
    }


def _supported_hough_circle(
    crop: np.ndarray,
    local_center: tuple[float, float],
    expected_radius_px: float,
    cfg: ApparentRadiusConfig,
    cv2: Any,
) -> tuple[float, float, float] | None:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (0, 0), 1.0)
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.0,
        minDist=5.0,
        param1=cfg.hough_canny_high,
        param2=cfg.hough_accumulator_threshold,
        minRadius=cfg.min_radius_px,
        maxRadius=cfg.max_radius_px,
    )
    if circles is None:
        return None
    supported = [
        tuple(float(value) for value in circle)
        for circle in circles[0]
        if math.hypot(float(circle[0]) - local_center[0], float(circle[1]) - local_center[1])
        <= cfg.max_center_refinement_px
    ]
    if not supported:
        return None
    return min(
        supported,
        key=lambda circle: (
            abs(circle[2] - expected_radius_px),
            math.hypot(circle[0] - local_center[0], circle[1] - local_center[1]),
            -circle[2],
        ),
    )


def _temporal_minor_axis(
    crop: np.ndarray,
    previous_crop: np.ndarray,
    center: tuple[float, float],
    previous_center: tuple[float, float],
    cfg: ApparentRadiusConfig,
    cv2: Any,
) -> dict[str, Any]:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    previous_gray = cv2.cvtColor(previous_crop, cv2.COLOR_BGR2GRAY)
    delta = np.abs(gray.astype(np.int16) - previous_gray.astype(np.int16))
    threshold = max(cfg.temporal_min_abs_delta, float(np.percentile(delta, cfg.temporal_threshold_percentile)))
    ys, xs = np.nonzero(delta >= threshold)
    if len(xs) < cfg.temporal_min_points:
        return {
            "reason": RadiusSkipReason.TEMPORAL_FOREGROUND_WEAK.value,
            "gates": {"temporal_point_count": len(xs), "temporal_threshold": threshold},
        }
    points = np.column_stack([xs, ys]).astype(np.float64)
    covariance = np.cov(points, rowvar=False, bias=True)
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    minor_value = max(float(values[order[1]]), 0.0)
    major_vector = vectors[:, order[0]]
    velocity = np.asarray(center, dtype=np.float64) - np.asarray(previous_center, dtype=np.float64)
    velocity_norm = float(np.linalg.norm(velocity))
    if velocity_norm <= 1e-9:
        angle_delta = 0.0
    else:
        velocity_angle = math.degrees(math.atan2(float(velocity[1]), float(velocity[0]))) % 180.0
        major_angle = math.degrees(math.atan2(float(major_vector[1]), float(major_vector[0]))) % 180.0
        angle_delta = _angle_delta_180(major_angle, velocity_angle)
    gates = {
        "temporal_point_count": len(xs),
        "temporal_threshold": threshold,
        "temporal_velocity_angle_delta_deg": angle_delta,
    }
    if angle_delta > cfg.max_temporal_velocity_angle_delta_deg:
        return {
            "reason": RadiusSkipReason.TEMPORAL_FOREGROUND_MISALIGNED.value,
            "gates": gates,
        }
    return {
        "reason": None,
        "radius_proxy_px": 2.0 * math.sqrt(minor_value),
        "angle_delta_deg": angle_delta,
        "gates": gates,
    }


def _angle_delta_180(a: float, b: float) -> float:
    delta = abs((float(a) - float(b)) % 180.0)
    return min(delta, 180.0 - delta)


def _crop(
    frame: np.ndarray,
    center: tuple[float, float],
    crop_radius: int,
) -> tuple[np.ndarray, tuple[int, int, int, int], tuple[float, float]] | None:
    x0 = int(round(center[0])) - crop_radius
    y0 = int(round(center[1])) - crop_radius
    x1 = x0 + 2 * crop_radius + 1
    y1 = y0 + 2 * crop_radius + 1
    if x0 < 0 or y0 < 0 or x1 > frame.shape[1] or y1 > frame.shape[0]:
        return None
    return frame[y0:y1, x0:x1], (x0, y0, x1, y1), (center[0] - x0, center[1] - y0)


def _finite_center(value: Any) -> tuple[float, float] | None:
    try:
        if len(value) != 2:
            return None
        center = (float(value[0]), float(value[1]))
    except (TypeError, ValueError):
        return None
    return center if all(math.isfinite(item) for item in center) else None


def _skipped(
    center: Any,
    reason: RadiusSkipReason,
    *,
    crop_box: tuple[int, int, int, int] | None = None,
    gates: dict[str, float | int | bool | str | None] | None = None,
) -> ApparentRadiusMeasurement:
    finite = _finite_center(center) or (0.0, 0.0)
    return ApparentRadiusMeasurement(
        status="skipped",
        input_center_xy_px=finite,
        center_xy_px=None,
        radius_px=None,
        confidence=0.0,
        method=TEMPORAL_METHOD,
        skip_reason=reason,
        crop_xyxy_px=crop_box,
        gates=dict(gates or {}),
        provenance={"heatmap_used": False},
    )


def _cv2() -> Any:
    try:
        import cv2  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise RuntimeError("opencv-python is required for apparent-radius estimation") from exc
    return cv2


__all__ = [
    "ApparentRadiusConfig",
    "ApparentRadiusMeasurement",
    "RadiusSkipReason",
    "SINGLE_FRAME_METHOD",
    "TEMPORAL_METHOD",
    "estimate_apparent_radius",
]

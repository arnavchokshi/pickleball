"""Project 2D pose foot points onto the metric court floor plane."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, Mapping, Sequence

from threed.racketsport.court_positioning import (
    CameraFloorGeometry,
    MetricConfidence,
    back_project_pixel_to_floor,
    estimate_ground_sample_distance,
    estimate_position_uncertainty,
)
from threed.racketsport.footlock import ContactHysteresis, FootContactObservation, classify_contact

FootSide = Literal["L", "R"]
FootPointName = Literal["ankle", "heel", "toe"]


@dataclass(frozen=True)
class FootImageObservation:
    side: FootSide
    pixels: Mapping[str, Sequence[float]]
    confidence: float
    height_m: float = 0.0
    previous_court_xy: Sequence[float] | None = None
    dt_s: float | None = None
    previous_contact: bool = False


@dataclass(frozen=True)
class GroundedFoot:
    side: FootSide
    court_xy: list[float]
    world_xyz: list[float]
    height_m: float
    contact: bool
    sigma_p_m: float
    confidence: float
    source_points: tuple[str, ...]


def ground_player_feet(
    observations: Sequence[FootImageObservation],
    *,
    geometry: CameraFloorGeometry,
    T_world_court: Sequence[Sequence[float]],
    metric_confidence: MetricConfidence,
    pixel_error_px: float = 1.5,
    plane_sigma_m: float = 0.012,
    calibration_sigma_m: float = 0.018,
    hysteresis: ContactHysteresis = ContactHysteresis(),
) -> list[GroundedFoot]:
    return [
        ground_foot_observation(
            observation,
            geometry=geometry,
            T_world_court=T_world_court,
            metric_confidence=metric_confidence,
            pixel_error_px=pixel_error_px,
            plane_sigma_m=plane_sigma_m,
            calibration_sigma_m=calibration_sigma_m,
            hysteresis=hysteresis,
        )
        for observation in observations
    ]


def ground_foot_observation(
    observation: FootImageObservation,
    *,
    geometry: CameraFloorGeometry,
    T_world_court: Sequence[Sequence[float]],
    metric_confidence: MetricConfidence,
    pixel_error_px: float = 1.5,
    plane_sigma_m: float = 0.012,
    calibration_sigma_m: float = 0.018,
    hysteresis: ContactHysteresis = ContactHysteresis(),
) -> GroundedFoot:
    if observation.side not in {"L", "R"}:
        raise ValueError("foot side must be 'L' or 'R'")
    confidence = _unit_interval(observation.confidence, "confidence")
    height_m = _finite_float(observation.height_m, "height_m")
    source_names = tuple(name for name in ("ankle", "heel", "toe") if name in observation.pixels)
    if "ankle" not in source_names:
        raise ValueError("foot observation must include an ankle pixel")

    world_points = [back_project_pixel_to_floor(observation.pixels[name], geometry) for name in source_names]
    world_xyz = [
        sum(point[axis] for point in world_points) / len(world_points)
        for axis in range(3)
    ]
    court_xyz = _transform_world_to_court(world_xyz, T_world_court)
    gsd_values = [estimate_ground_sample_distance(observation.pixels[name], geometry) for name in source_names]
    gsd_m_per_px = sum(gsd_values) / len(gsd_values)
    sigma_p_m = estimate_position_uncertainty(
        pixel_error_px=pixel_error_px,
        gsd_m_per_px=gsd_m_per_px,
        plane_sigma_m=plane_sigma_m,
        calibration_sigma_m=calibration_sigma_m,
    )

    speed_mps = _speed_mps(court_xyz[:2], observation.previous_court_xy, observation.dt_s)
    contact = classify_contact(
        FootContactObservation(height_m=max(0.0, height_m), speed_mps=speed_mps, confidence=confidence),
        previous_contact=observation.previous_contact,
        hysteresis=hysteresis,
    )
    if metric_confidence != "high":
        contact = False

    return GroundedFoot(
        side=observation.side,
        court_xy=[float(court_xyz[0]), float(court_xyz[1])],
        world_xyz=[float(value) for value in world_xyz],
        height_m=height_m,
        contact=contact,
        sigma_p_m=sigma_p_m,
        confidence=confidence,
        source_points=source_names,
    )


def _transform_world_to_court(world_xyz: Sequence[float], T_world_court: Sequence[Sequence[float]]) -> list[float]:
    point = _vec3(world_xyz, "world_xyz")
    transform = _mat4(T_world_court, "T_world_court")
    rotation = [[transform[row][col] for col in range(3)] for row in range(3)]
    translation = [transform[row][3] for row in range(3)]
    shifted = [point[idx] - translation[idx] for idx in range(3)]
    return [
        rotation[0][axis] * shifted[0] + rotation[1][axis] * shifted[1] + rotation[2][axis] * shifted[2]
        for axis in range(3)
    ]


def _speed_mps(
    current_court_xy: Sequence[float],
    previous_court_xy: Sequence[float] | None,
    dt_s: float | None,
) -> float:
    if previous_court_xy is None or dt_s is None:
        return 0.0
    dt = _positive_float(dt_s, "dt_s")
    previous = _xy(previous_court_xy, "previous_court_xy")
    current = _xy(current_court_xy, "current_court_xy")
    return math.hypot(current[0] - previous[0], current[1] - previous[1]) / dt


def _unit_interval(value: float, name: str) -> float:
    result = _finite_float(value, name)
    if result < 0.0 or result > 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return result


def _positive_float(value: float, name: str) -> float:
    result = _finite_float(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _finite_float(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _xy(value: Sequence[float], name: str) -> tuple[float, float]:
    if len(value) != 2:
        raise ValueError(f"{name} must contain 2 values")
    return (_finite_float(value[0], f"{name}[0]"), _finite_float(value[1], f"{name}[1]"))


def _vec3(value: Sequence[float], name: str) -> list[float]:
    if len(value) != 3:
        raise ValueError(f"{name} must contain 3 values")
    return [_finite_float(value[idx], f"{name}[{idx}]") for idx in range(3)]


def _mat4(value: Sequence[Sequence[float]], name: str) -> list[list[float]]:
    if len(value) != 4:
        raise ValueError(f"{name} must contain 4 rows")
    matrix: list[list[float]] = []
    for row_idx, row in enumerate(value):
        if len(row) != 4:
            raise ValueError(f"{name}[{row_idx}] must contain 4 values")
        matrix.append([_finite_float(row[col_idx], f"{name}[{row_idx}][{col_idx}]") for col_idx in range(4)])
    return matrix

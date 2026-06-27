"""Biomechanics metric primitives."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any


MetricDict = dict[str, dict[str, Any]]


def balance_score_metric(
    center_of_mass_xy: Sequence[float],
    support_points_xy: Sequence[Sequence[float]],
    *,
    conf: float = 1.0,
) -> MetricDict:
    """Return a bounded balance score from COM against a support bbox proxy."""

    com_x, com_y = _xy(center_of_mass_xy, name="center_of_mass_xy")
    support = [_xy(point, name="support_points_xy[]") for point in support_points_xy]
    if len(support) < 2:
        raise ValueError("support_points_xy must contain at least 2 points")

    xs = [point[0] for point in support]
    ys = [point[1] for point in support]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    width_m = max_x - min_x
    depth_m = max_y - min_y
    proxy_scale_m = max(width_m, depth_m)
    if proxy_scale_m <= 0.0:
        raise ValueError("support_points_xy must span a non-zero support proxy")

    dx_m = max(min_x - com_x, 0.0, com_x - max_x)
    dy_m = max(min_y - com_y, 0.0, com_y - max_y)
    outside_distance_m = math.hypot(dx_m, dy_m)
    score = max(0.0, min(1.0, 1.0 - outside_distance_m / proxy_scale_m))
    return {
        "balance_score": _metric_payload(
            value=score,
            conf=conf,
            unit="score",
            source="cpu_com_support_proxy",
        )
    }


def x_factor_angle_metric(
    shoulder_vector_xy: Sequence[float],
    hip_vector_xy: Sequence[float],
    *,
    conf: float = 1.0,
) -> MetricDict:
    """Return signed smallest shoulder-vs-hip separation in degrees."""

    shoulder_x, shoulder_y = _nonzero_xy(shoulder_vector_xy, name="shoulder_vector_xy")
    hip_x, hip_y = _nonzero_xy(hip_vector_xy, name="hip_vector_xy")
    cross = hip_x * shoulder_y - hip_y * shoulder_x
    dot = hip_x * shoulder_x + hip_y * shoulder_y
    angle_deg = math.degrees(math.atan2(cross, dot))
    return {
        "x_factor_deg": _metric_payload(
            value=angle_deg,
            conf=conf,
            unit="deg",
            source="cpu_shoulder_hip_vectors",
        )
    }


def _metric_payload(*, value: float, conf: float, unit: str, source: str) -> dict[str, Any]:
    return {
        "value": float(value),
        "conf": _confidence(conf),
        "unit": unit,
        "gated": False,
        "source": source,
    }


def _nonzero_xy(value: Sequence[float], *, name: str) -> tuple[float, float]:
    x, y = _xy(value, name=name)
    if math.hypot(x, y) == 0.0:
        raise ValueError(f"{name} must be non-zero")
    return x, y


def _xy(value: Sequence[float], *, name: str) -> tuple[float, float]:
    if len(value) != 2:
        raise ValueError(f"{name} must be world_xy with exactly 2 values")
    x = _finite_float(value[0], name=f"{name}[0]")
    y = _finite_float(value[1], name=f"{name}[1]")
    return x, y


def _confidence(value: float) -> float:
    conf = _finite_float(value, name="conf")
    if not 0.0 <= conf <= 1.0:
        raise ValueError("conf must be between 0 and 1")
    return conf


def _finite_float(value: float, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    return numeric

"""CPU-only 3D ball physics reconstruction primitives."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Iterable, Sequence


@dataclass(frozen=True)
class BallSample3D:
    """Single reconstructed ball sample in world coordinates."""

    t: float
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class BounceEvent:
    """Simple bounce event that can be projected into BallTrack.bounces."""

    t: float
    world_xy: tuple[float, float]
    sample_index: int
    z_min: float


@dataclass(frozen=True)
class ParabolaFit3D:
    """Linear x/y plus parabolic z fit over a short ball segment."""

    t0: float
    x0: float
    y0: float
    z0: float
    vx: float
    vy: float
    vz: float
    accel_z: float
    inlier_indices: tuple[int, ...]
    outlier_indices: tuple[int, ...]
    rms_residual_m: float
    max_residual_m: float

    def predict(self, t: float) -> tuple[float, float, float]:
        dt = t - self.t0
        return (
            self.x0 + self.vx * dt,
            self.y0 + self.vy * dt,
            self.z0 + self.vz * dt + 0.5 * self.accel_z * dt * dt,
        )

    def residual_m(self, sample: BallSample3D) -> float:
        x, y, z = self.predict(sample.t)
        return sqrt((sample.x - x) ** 2 + (sample.y - y) ** 2 + (sample.z - z) ** 2)


def fit_parabola_segment(
    samples: Sequence[BallSample3D],
    *,
    residual_threshold_m: float | None = None,
    min_inliers: int = 3,
) -> ParabolaFit3D:
    """Fit linear x/y and constant-acceleration height to a short segment.

    If ``residual_threshold_m`` is provided, the largest residual outlier is
    removed deterministically until all retained samples are under threshold or
    ``min_inliers`` would be violated.
    """

    if len(samples) < min_inliers:
        raise ValueError("fit_parabola_segment requires at least min_inliers samples")
    if min_inliers < 3:
        raise ValueError("min_inliers must be at least 3 for a height parabola")
    if residual_threshold_m is not None and residual_threshold_m < 0.0:
        raise ValueError("residual_threshold_m must be non-negative")

    active = list(range(len(samples)))
    outliers: list[int] = []

    while True:
        fit = _fit_indices(samples, active, tuple(sorted(outliers)))
        residuals = [(idx, fit.residual_m(samples[idx])) for idx in active]
        max_idx, max_residual = max(residuals, key=lambda item: (item[1], -item[0]))
        if residual_threshold_m is None or max_residual <= residual_threshold_m:
            return fit
        if len(active) <= min_inliers:
            return fit
        active.remove(max_idx)
        outliers.append(max_idx)


def detect_bounce_events(
    samples: Sequence[BallSample3D],
    *,
    min_vertical_speed_mps: float = 0.0,
    min_separation_s: float = 0.05,
) -> tuple[BounceEvent, ...]:
    """Detect bounces from a z local minimum with downward-to-upward velocity."""

    if len(samples) < 3:
        return ()
    ordered = sorted(enumerate(samples), key=lambda item: (item[1].t, item[0]))
    events: list[BounceEvent] = []
    last_t: float | None = None

    for ordered_idx in range(1, len(ordered) - 1):
        _, prev_sample = ordered[ordered_idx - 1]
        original_idx, sample = ordered[ordered_idx]
        _, next_sample = ordered[ordered_idx + 1]

        before_dt = sample.t - prev_sample.t
        after_dt = next_sample.t - sample.t
        if before_dt <= 0.0 or after_dt <= 0.0:
            continue

        vz_before = (sample.z - prev_sample.z) / before_dt
        vz_after = (next_sample.z - sample.z) / after_dt
        is_z_minimum = sample.z <= prev_sample.z and sample.z <= next_sample.z
        has_sign_change = vz_before < -min_vertical_speed_mps and vz_after > min_vertical_speed_mps
        separated = last_t is None or sample.t - last_t >= min_separation_s

        if is_z_minimum and has_sign_change and separated:
            events.append(
                BounceEvent(
                    t=sample.t,
                    world_xy=(sample.x, sample.y),
                    sample_index=original_idx,
                    z_min=sample.z,
                )
            )
            last_t = sample.t

    return tuple(events)


def project_bounces_to_ball_track(events: Iterable[BounceEvent]) -> list[dict[str, object]]:
    """Return dictionaries compatible with the existing BallTrack.bounces field."""

    return [
        {"t": event.t, "world_xy": [event.world_xy[0], event.world_xy[1]]}
        for event in sorted(events, key=lambda event: event.t)
    ]


def _fit_indices(
    samples: Sequence[BallSample3D],
    indices: Sequence[int],
    outlier_indices: tuple[int, ...],
) -> ParabolaFit3D:
    if len(indices) < 3:
        raise ValueError("at least 3 inliers are required for a height parabola")

    t0 = samples[indices[0]].t
    linear_rows = [[1.0, samples[idx].t - t0] for idx in indices]
    parabola_rows = [
        [1.0, samples[idx].t - t0, (samples[idx].t - t0) ** 2]
        for idx in indices
    ]

    x0, vx = _least_squares(linear_rows, [samples[idx].x for idx in indices])
    y0, vy = _least_squares(linear_rows, [samples[idx].y for idx in indices])
    z0, vz, z_quadratic = _least_squares(parabola_rows, [samples[idx].z for idx in indices])
    accel_z = 2.0 * z_quadratic

    provisional = ParabolaFit3D(
        t0=t0,
        x0=x0,
        y0=y0,
        z0=z0,
        vx=vx,
        vy=vy,
        vz=vz,
        accel_z=accel_z,
        inlier_indices=tuple(indices),
        outlier_indices=outlier_indices,
        rms_residual_m=0.0,
        max_residual_m=0.0,
    )
    residuals = [provisional.residual_m(samples[idx]) for idx in indices]
    rms = sqrt(sum(residual * residual for residual in residuals) / len(residuals))
    max_residual = max(residuals)
    return ParabolaFit3D(
        t0=t0,
        x0=x0,
        y0=y0,
        z0=z0,
        vx=vx,
        vy=vy,
        vz=vz,
        accel_z=accel_z,
        inlier_indices=tuple(indices),
        outlier_indices=outlier_indices,
        rms_residual_m=rms,
        max_residual_m=max_residual,
    )


def _least_squares(rows: Sequence[Sequence[float]], values: Sequence[float]) -> tuple[float, ...]:
    if not rows:
        raise ValueError("least squares requires at least one row")
    width = len(rows[0])
    normal = [[0.0 for _ in range(width)] for _ in range(width)]
    rhs = [0.0 for _ in range(width)]

    for row, value in zip(rows, values):
        if len(row) != width:
            raise ValueError("least squares rows must have consistent width")
        for i in range(width):
            rhs[i] += row[i] * value
            for j in range(width):
                normal[i][j] += row[i] * row[j]

    return tuple(_solve_linear_system(normal, rhs))


def _solve_linear_system(matrix: list[list[float]], rhs: list[float]) -> list[float]:
    n = len(rhs)
    for col in range(n):
        pivot_row = max(range(col, n), key=lambda row: abs(matrix[row][col]))
        if abs(matrix[pivot_row][col]) < 1e-12:
            raise ValueError("cannot fit parabola to singular sample times")
        if pivot_row != col:
            matrix[col], matrix[pivot_row] = matrix[pivot_row], matrix[col]
            rhs[col], rhs[pivot_row] = rhs[pivot_row], rhs[col]

        pivot = matrix[col][col]
        for j in range(col, n):
            matrix[col][j] /= pivot
        rhs[col] /= pivot

        for row in range(n):
            if row == col:
                continue
            factor = matrix[row][col]
            if factor == 0.0:
                continue
            for j in range(col, n):
                matrix[row][j] -= factor * matrix[col][j]
            rhs[row] -= factor * rhs[col]

    return rhs


__all__ = [
    "BallSample3D",
    "BounceEvent",
    "ParabolaFit3D",
    "detect_bounce_events",
    "fit_parabola_segment",
    "project_bounces_to_ball_track",
]

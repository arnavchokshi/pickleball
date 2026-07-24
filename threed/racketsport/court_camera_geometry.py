"""Small, dependency-light camera utilities for structured court calibration.

The v1 product assumes a static pinhole camera with at most a bounded first
radial-distortion coefficient.  This module keeps that assumption explicit and
provides the uncertainty propagation needed by a future ``court_lock.json``
consumer.  It deliberately does not choose calibration authority or mutate raw
observations.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np


DEFAULT_K1_BOUNDS: tuple[float, float] = (-0.45, 0.25)


class RadialDistortionAmbiguityError(ValueError):
    """Raised when a distorted radius has no stable central-branch inverse."""


@dataclass(frozen=True)
class PinholeIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float

    def __post_init__(self) -> None:
        values = (self.fx, self.fy, self.cx, self.cy)
        if any(isinstance(value, bool) or not math.isfinite(float(value)) for value in values):
            raise ValueError("camera intrinsics must be finite numbers")
        if float(self.fx) <= 0.0 or float(self.fy) <= 0.0:
            raise ValueError("fx and fy must be positive")

    @property
    def matrix(self) -> np.ndarray:
        return np.asarray(
            [[float(self.fx), 0.0, float(self.cx)], [0.0, float(self.fy), float(self.cy)], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )


@dataclass(frozen=True)
class RadialUndistortionDiagnostics:
    converged: tuple[bool, ...]
    ambiguous: tuple[bool, ...]
    iterations: tuple[int, ...]
    k1: float
    k1_bounds: tuple[float, float]

    @property
    def all_converged(self) -> bool:
        return all(self.converged)

    @property
    def any_ambiguous(self) -> bool:
        return any(self.ambiguous)


def validate_bounded_k1(
    k1: float,
    *,
    bounds: tuple[float, float] = DEFAULT_K1_BOUNDS,
) -> float:
    """Validate, but never silently clip, a first radial coefficient."""

    if isinstance(k1, bool) or not math.isfinite(float(k1)):
        raise ValueError("k1 must be a finite number")
    lower, upper = _validate_bounds(bounds)
    value = float(k1)
    if not lower <= value <= upper:
        raise ValueError(f"k1={value} is outside configured bounds [{lower}, {upper}]")
    return value


def distort_normalized_points_radial_k1(
    points_xy: Sequence[Sequence[float]] | np.ndarray,
    *,
    k1: float,
    bounds: tuple[float, float] = DEFAULT_K1_BOUNDS,
) -> np.ndarray:
    """Apply ``x_d=x(1+k1*r^2)`` in normalized camera coordinates."""

    points = _points2(points_xy, name="points_xy")
    coefficient = validate_bounded_k1(k1, bounds=bounds)
    radius_squared = np.sum(points * points, axis=1, keepdims=True)
    return points * (1.0 + coefficient * radius_squared)


def distort_pixels_radial_k1(
    pixels_xy: Sequence[Sequence[float]] | np.ndarray,
    intrinsics: PinholeIntrinsics,
    *,
    k1: float,
    bounds: tuple[float, float] = DEFAULT_K1_BOUNDS,
) -> np.ndarray:
    """Distort undistorted pixels with the bounded radial-k1 model."""

    pixels = _points2(pixels_xy, name="pixels_xy")
    normalized = _normalize_pixels(pixels, intrinsics)
    distorted = distort_normalized_points_radial_k1(normalized, k1=k1, bounds=bounds)
    return _denormalize_pixels(distorted, intrinsics)


def project_camera_points_radial_k1(
    points_camera_xyz: Sequence[Sequence[float]] | np.ndarray,
    intrinsics: PinholeIntrinsics,
    *,
    k1: float,
    bounds: tuple[float, float] = DEFAULT_K1_BOUNDS,
) -> np.ndarray:
    """Project positive-depth camera-frame points into distorted pixels."""

    points = np.asarray(points_camera_xyz, dtype=np.float64)
    if points.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("points_camera_xyz must be a finite Nx3 array")
    if np.any(points[:, 2] <= 0.0):
        raise ValueError("all camera-frame points must have positive depth")
    normalized = points[:, :2] / points[:, 2:3]
    distorted = distort_normalized_points_radial_k1(normalized, k1=k1, bounds=bounds)
    return _denormalize_pixels(distorted, intrinsics)


def undistort_pixels_radial_k1(
    pixels_xy: Sequence[Sequence[float]] | np.ndarray,
    intrinsics: PinholeIntrinsics,
    *,
    k1: float,
    bounds: tuple[float, float] = DEFAULT_K1_BOUNDS,
    strict: bool = True,
    tolerance: float = 1.0e-12,
    max_iterations: int = 80,
) -> tuple[np.ndarray, RadialUndistortionDiagnostics]:
    """Invert radial-k1 pixels on the stable central branch.

    Negative ``k1`` has a finite turning radius.  Measurements at or beyond
    that turning point do not have a stable central-branch inverse and are
    marked ambiguous.  ``strict=True`` raises instead of returning NaNs, which
    prevents an ambiguous fisheye-like observation from silently entering a
    court lock.
    """

    if tolerance <= 0.0 or not math.isfinite(float(tolerance)):
        raise ValueError("tolerance must be positive and finite")
    if isinstance(max_iterations, bool) or int(max_iterations) <= 0:
        raise ValueError("max_iterations must be a positive integer")
    coefficient = validate_bounded_k1(k1, bounds=bounds)
    pixels = _points2(pixels_xy, name="pixels_xy")
    distorted = _normalize_pixels(pixels, intrinsics)
    result = np.empty_like(distorted)
    converged: list[bool] = []
    ambiguous: list[bool] = []
    iteration_counts: list[int] = []

    for index, point in enumerate(distorted):
        radius_distorted = float(np.linalg.norm(point))
        if radius_distorted <= tolerance or abs(coefficient) <= tolerance:
            result[index] = point
            converged.append(True)
            ambiguous.append(False)
            iteration_counts.append(0)
            continue

        is_ambiguous = False
        if coefficient < 0.0:
            turning_radius = math.sqrt(-1.0 / (3.0 * coefficient))
            max_distorted_radius = turning_radius * (1.0 + coefficient * turning_radius**2)
            if radius_distorted >= max_distorted_radius - tolerance:
                is_ambiguous = True
                if strict:
                    raise RadialDistortionAmbiguityError(
                        "distorted radius lies at or beyond the radial-k1 turning point "
                        f"(point_index={index}, rd={radius_distorted:.9g}, limit={max_distorted_radius:.9g})"
                    )
                result[index] = np.asarray([math.nan, math.nan], dtype=np.float64)
                converged.append(False)
                ambiguous.append(True)
                iteration_counts.append(0)
                continue
            lower = 0.0
            upper = turning_radius
        else:
            lower = 0.0
            upper = max(1.0, radius_distorted)
            while _distorted_radius(upper, coefficient) < radius_distorted:
                upper *= 2.0
                if upper > 1.0e6:
                    is_ambiguous = True
                    break
            if is_ambiguous:
                if strict:
                    raise RadialDistortionAmbiguityError(
                        f"could not bracket radial-k1 inverse for point_index={index}"
                    )
                result[index] = np.asarray([math.nan, math.nan], dtype=np.float64)
                converged.append(False)
                ambiguous.append(True)
                iteration_counts.append(0)
                continue

        radius_undistorted = 0.0
        did_converge = False
        used_iterations = 0
        for iteration in range(1, int(max_iterations) + 1):
            used_iterations = iteration
            radius_undistorted = 0.5 * (lower + upper)
            mapped = _distorted_radius(radius_undistorted, coefficient)
            if abs(mapped - radius_distorted) <= tolerance:
                did_converge = True
                break
            if mapped < radius_distorted:
                lower = radius_undistorted
            else:
                upper = radius_undistorted
        if not did_converge:
            did_converge = abs(
                _distorted_radius(radius_undistorted, coefficient) - radius_distorted
            ) <= max(tolerance * 10.0, 1.0e-10)
        if not did_converge and strict:
            raise RadialDistortionAmbiguityError(
                f"radial-k1 inverse did not converge for point_index={index}"
            )
        if did_converge:
            result[index] = point * (radius_undistorted / radius_distorted)
        else:
            result[index] = np.asarray([math.nan, math.nan], dtype=np.float64)
        converged.append(did_converge)
        ambiguous.append(is_ambiguous or not did_converge)
        iteration_counts.append(used_iterations)

    return _denormalize_pixels(result, intrinsics), RadialUndistortionDiagnostics(
        converged=tuple(converged),
        ambiguous=tuple(ambiguous),
        iterations=tuple(iteration_counts),
        k1=coefficient,
        k1_bounds=_validate_bounds(bounds),
    )


def propagate_covariance(
    jacobian: Sequence[Sequence[float]] | np.ndarray,
    covariance: Sequence[Sequence[float]] | np.ndarray,
) -> np.ndarray:
    """Return ``J covariance J.T`` after finite/symmetry/PSD validation."""

    jac = np.asarray(jacobian, dtype=np.float64)
    cov = _validated_covariance(covariance, name="covariance")
    if jac.ndim != 2 or jac.shape[1] != cov.shape[0] or not np.isfinite(jac).all():
        raise ValueError("jacobian must be a finite MxN matrix matching covariance")
    propagated = jac @ cov @ jac.T
    propagated = 0.5 * (propagated + propagated.T)
    if np.linalg.eigvalsh(propagated).min(initial=0.0) < -1.0e-8:
        raise ValueError("propagated covariance is not positive semidefinite")
    return propagated


def project_planar_point_with_covariance(
    homography_image_from_court: Sequence[Sequence[float]] | np.ndarray,
    court_xy: Sequence[float] | np.ndarray,
    transform_covariance: Sequence[Sequence[float]] | np.ndarray,
    *,
    court_covariance: Sequence[Sequence[float]] | np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Project one court point and propagate an 8-parameter homography covariance.

    The parameter order is ``h00,h01,h02,h10,h11,h12,h20,h21`` with ``h22``
    normalized to one.  Optional court-coordinate uncertainty is propagated in
    addition to transform uncertainty.
    """

    homography = _normalized_homography(homography_image_from_court)
    point = np.asarray(court_xy, dtype=np.float64)
    if point.shape != (2,) or not np.isfinite(point).all():
        raise ValueError("court_xy must contain two finite numbers")
    transform_cov = _validated_covariance(transform_covariance, name="transform_covariance", size=8)
    x, y = float(point[0]), float(point[1])
    denominator = float(homography[2, 0] * x + homography[2, 1] * y + 1.0)
    if abs(denominator) <= 1.0e-12:
        raise ValueError("court point projects to infinity")
    numerator_u = float(homography[0, 0] * x + homography[0, 1] * y + homography[0, 2])
    numerator_v = float(homography[1, 0] * x + homography[1, 1] * y + homography[1, 2])
    u = numerator_u / denominator
    v = numerator_v / denominator
    inverse_denominator = 1.0 / denominator
    jac_transform = np.asarray(
        [
            [
                x * inverse_denominator,
                y * inverse_denominator,
                inverse_denominator,
                0.0,
                0.0,
                0.0,
                -u * x * inverse_denominator,
                -u * y * inverse_denominator,
            ],
            [
                0.0,
                0.0,
                0.0,
                x * inverse_denominator,
                y * inverse_denominator,
                inverse_denominator,
                -v * x * inverse_denominator,
                -v * y * inverse_denominator,
            ],
        ],
        dtype=np.float64,
    )
    point_covariance = propagate_covariance(jac_transform, transform_cov)
    if court_covariance is not None:
        court_cov = _validated_covariance(court_covariance, name="court_covariance", size=2)
        jac_point = np.asarray(
            [
                [
                    (homography[0, 0] * denominator - numerator_u * homography[2, 0]) / denominator**2,
                    (homography[0, 1] * denominator - numerator_u * homography[2, 1]) / denominator**2,
                ],
                [
                    (homography[1, 0] * denominator - numerator_v * homography[2, 0]) / denominator**2,
                    (homography[1, 1] * denominator - numerator_v * homography[2, 1]) / denominator**2,
                ],
            ],
            dtype=np.float64,
        )
        point_covariance += propagate_covariance(jac_point, court_cov)
    return np.asarray([u, v], dtype=np.float64), 0.5 * (point_covariance + point_covariance.T)


def distort_pixel_with_covariance_radial_k1(
    pixel_xy: Sequence[float] | np.ndarray,
    covariance_px2: Sequence[Sequence[float]] | np.ndarray,
    intrinsics: PinholeIntrinsics,
    *,
    k1: float,
    k1_variance: float = 0.0,
    bounds: tuple[float, float] = DEFAULT_K1_BOUNDS,
) -> tuple[np.ndarray, np.ndarray]:
    """Distort one pixel and propagate pixel plus k1 uncertainty."""

    coefficient = validate_bounded_k1(k1, bounds=bounds)
    if isinstance(k1_variance, bool) or not math.isfinite(float(k1_variance)) or float(k1_variance) < 0.0:
        raise ValueError("k1_variance must be non-negative and finite")
    point = np.asarray(pixel_xy, dtype=np.float64)
    if point.shape != (2,) or not np.isfinite(point).all():
        raise ValueError("pixel_xy must contain two finite numbers")
    covariance = _validated_covariance(covariance_px2, name="covariance_px2", size=2)
    normalized = _normalize_pixels(point.reshape(1, 2), intrinsics)[0]
    x, y = float(normalized[0]), float(normalized[1])
    radius_squared = x * x + y * y
    scale = 1.0 + coefficient * radius_squared
    jac_normalized = np.asarray(
        [
            [scale + 2.0 * coefficient * x * x, 2.0 * coefficient * x * y],
            [2.0 * coefficient * x * y, scale + 2.0 * coefficient * y * y],
        ],
        dtype=np.float64,
    )
    to_normalized = np.diag([1.0 / float(intrinsics.fx), 1.0 / float(intrinsics.fy)])
    to_pixels = np.diag([float(intrinsics.fx), float(intrinsics.fy)])
    jac_pixel = to_pixels @ jac_normalized @ to_normalized
    output_covariance = propagate_covariance(jac_pixel, covariance)
    if float(k1_variance) > 0.0:
        jac_k1 = np.asarray(
            [[float(intrinsics.fx) * x * radius_squared], [float(intrinsics.fy) * y * radius_squared]],
            dtype=np.float64,
        )
        output_covariance += jac_k1 @ np.asarray([[float(k1_variance)]]) @ jac_k1.T
    distorted = distort_pixels_radial_k1(point.reshape(1, 2), intrinsics, k1=coefficient, bounds=bounds)[0]
    return distorted, 0.5 * (output_covariance + output_covariance.T)


def _validate_bounds(bounds: tuple[float, float]) -> tuple[float, float]:
    if len(bounds) != 2:
        raise ValueError("k1 bounds must contain exactly two values")
    lower, upper = float(bounds[0]), float(bounds[1])
    if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
        raise ValueError("k1 bounds must be finite and strictly ordered")
    return lower, upper


def _points2(points: Sequence[Sequence[float]] | np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(points, dtype=np.float64)
    if array.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 2 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite Nx2 array")
    return array


def _normalize_pixels(points: np.ndarray, intrinsics: PinholeIntrinsics) -> np.ndarray:
    return np.column_stack(
        [
            (points[:, 0] - float(intrinsics.cx)) / float(intrinsics.fx),
            (points[:, 1] - float(intrinsics.cy)) / float(intrinsics.fy),
        ]
    )


def _denormalize_pixels(points: np.ndarray, intrinsics: PinholeIntrinsics) -> np.ndarray:
    return np.column_stack(
        [
            points[:, 0] * float(intrinsics.fx) + float(intrinsics.cx),
            points[:, 1] * float(intrinsics.fy) + float(intrinsics.cy),
        ]
    )


def _distorted_radius(radius: float, k1: float) -> float:
    return radius * (1.0 + k1 * radius * radius)


def _validated_covariance(
    covariance: Sequence[Sequence[float]] | np.ndarray,
    *,
    name: str,
    size: int | None = None,
) -> np.ndarray:
    matrix = np.asarray(covariance, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be a square matrix")
    if size is not None and matrix.shape != (size, size):
        raise ValueError(f"{name} must have shape {size}x{size}")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} must contain only finite values")
    if not np.allclose(matrix, matrix.T, atol=1.0e-10, rtol=1.0e-10):
        raise ValueError(f"{name} must be symmetric")
    eigenvalues = np.linalg.eigvalsh(matrix)
    if eigenvalues.min(initial=0.0) < -1.0e-9:
        raise ValueError(f"{name} must be positive semidefinite")
    return 0.5 * (matrix + matrix.T)


def _normalized_homography(homography: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    matrix = np.asarray(homography, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("homography_image_from_court must be a finite 3x3 matrix")
    if abs(float(np.linalg.det(matrix))) <= 1.0e-12:
        raise ValueError("homography_image_from_court must be nonsingular")
    if abs(float(matrix[2, 2])) <= 1.0e-12:
        raise ValueError("homography h22 must be nonzero for the 8-parameter convention")
    return matrix / float(matrix[2, 2])


__all__ = [
    "DEFAULT_K1_BOUNDS",
    "PinholeIntrinsics",
    "RadialDistortionAmbiguityError",
    "RadialUndistortionDiagnostics",
    "distort_normalized_points_radial_k1",
    "distort_pixel_with_covariance_radial_k1",
    "distort_pixels_radial_k1",
    "project_camera_points_radial_k1",
    "project_planar_point_with_covariance",
    "propagate_covariance",
    "undistort_pixels_radial_k1",
    "validate_bounded_k1",
]

"""Bounded joint planar-homography and radial-k1 refinement."""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

from threed.racketsport.court_camera_geometry import (
    DEFAULT_K1_BOUNDS,
    PinholeIntrinsics,
    distort_pixels_radial_k1,
    undistort_pixels_radial_k1,
    validate_bounded_k1,
)


def refine_planar_homography_and_k1(
    world_xy: Sequence[Sequence[float]] | np.ndarray,
    observed_pixels_xy: Sequence[Sequence[float]] | np.ndarray,
    intrinsics: PinholeIntrinsics,
    *,
    weights: Sequence[float] | np.ndarray | None = None,
    k1_bounds: tuple[float, float] = DEFAULT_K1_BOUNDS,
    k1_initial: float = 0.0,
    grid_steps: int = 41,
) -> dict[str, Any]:
    """Jointly fit an undistorted floor homography and bounded radial coefficient."""

    world = _points(world_xy, "world_xy")
    observed = _points(observed_pixels_xy, "observed_pixels_xy")
    if len(world) != len(observed) or len(world) < 4:
        raise ValueError("world and observed points must match with at least four rows")
    if weights is None:
        parsed_weights = np.ones(len(world), dtype=np.float64)
    else:
        parsed_weights = np.asarray(weights, dtype=np.float64)
        if parsed_weights.shape != (len(world),) or not np.isfinite(parsed_weights).all() or np.any(parsed_weights <= 0.0):
            raise ValueError("weights must be finite positive values matching the observations")
    if isinstance(grid_steps, bool) or not isinstance(grid_steps, int) or grid_steps < 3:
        raise ValueError("grid_steps must be an integer of at least three")
    lower, upper = (float(k1_bounds[0]), float(k1_bounds[1]))
    validate_bounded_k1(lower, bounds=k1_bounds)
    validate_bounded_k1(upper, bounds=k1_bounds)
    initial = validate_bounded_k1(k1_initial, bounds=k1_bounds)
    grid = sorted(set(np.linspace(lower, upper, grid_steps).tolist() + [0.0, initial]))
    candidates: list[dict[str, Any]] = []
    for coefficient in grid:
        try:
            undistorted, diagnostics = undistort_pixels_radial_k1(
                observed,
                intrinsics,
                k1=coefficient,
                bounds=k1_bounds,
                strict=True,
            )
            if not diagnostics.all_converged:
                continue
            homography = _weighted_homography(world, undistorted, parsed_weights)
            projected_undistorted = _project(homography, world)
            projected_raw = distort_pixels_radial_k1(
                projected_undistorted,
                intrinsics,
                k1=coefficient,
                bounds=k1_bounds,
            )
        except ValueError:
            continue
        residuals = np.linalg.norm(projected_raw - observed, axis=1)
        robust = _weighted_robust_cost(residuals, parsed_weights)
        candidates.append(
            {
                "k1": float(coefficient),
                "cost": robust,
                "homography": homography,
                "projected_raw": projected_raw,
                "residuals": residuals,
            }
        )
    if not candidates:
        raise ValueError("no bounded radial-distortion hypothesis was numerically valid")
    candidates.sort(key=lambda row: (float(row["cost"]), abs(float(row["k1"])), float(row["k1"])))
    best = candidates[0]
    costs = np.asarray([float(row["cost"]) for row in candidates], dtype=np.float64)
    k1s = np.asarray([float(row["k1"]) for row in candidates], dtype=np.float64)
    tolerance = max(0.25, float(best["cost"]) * 0.10)
    plausible = k1s[costs <= float(best["cost"]) + tolerance]
    k1_variance = float(np.var(plausible)) if len(plausible) > 1 else 0.0
    margin = None if len(candidates) < 2 else float(candidates[1]["cost"] - best["cost"])
    return {
        "homography_undistorted_from_court": best["homography"].tolist(),
        "k1": float(best["k1"]),
        "k1_variance": k1_variance,
        "projected_pixels_xy": best["projected_raw"].tolist(),
        "residuals_px": best["residuals"].tolist(),
        "median_residual_px": float(np.median(best["residuals"])),
        "p95_residual_px": float(np.percentile(best["residuals"], 95)),
        "hypothesis_margin": margin,
        "candidate_count": len(candidates),
        "bounds": [lower, upper],
        "uncertainty_status": "ambiguous" if k1_variance > 0.01 else "bounded",
    }


def _points(value: Any, name: str) -> np.ndarray:
    points = np.asarray(value, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError(f"{name} must be a finite Nx2 array")
    return points


def _weighted_homography(world: np.ndarray, image: np.ndarray, weights: np.ndarray) -> np.ndarray:
    world_n, world_t = _normalize(world, weights)
    image_n, image_t = _normalize(image, weights)
    rows: list[list[float]] = []
    for (x, y), (u, v), weight in zip(world_n, image_n, weights, strict=True):
        scale = math.sqrt(float(weight))
        rows.append([-x * scale, -y * scale, -scale, 0.0, 0.0, 0.0, u * x * scale, u * y * scale, u * scale])
        rows.append([0.0, 0.0, 0.0, -x * scale, -y * scale, -scale, v * x * scale, v * y * scale, v * scale])
    matrix = np.asarray(rows, dtype=np.float64)
    _, _, right_t = np.linalg.svd(matrix, full_matrices=True)
    normalized_h = right_t[-1].reshape(3, 3)
    homography = np.linalg.inv(image_t) @ normalized_h @ world_t
    if abs(float(homography[2, 2])) <= 1.0e-12 or np.linalg.matrix_rank(homography) < 3:
        raise ValueError("weighted DLT produced a degenerate homography")
    return homography / float(homography[2, 2])


def _normalize(points: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    total = float(weights.sum())
    center = (points * weights[:, None]).sum(axis=0) / total
    centered = points - center
    mean_distance = float((np.linalg.norm(centered, axis=1) * weights).sum() / total)
    if mean_distance <= 1.0e-9:
        raise ValueError("point set is degenerate")
    scale = math.sqrt(2.0) / mean_distance
    transform = np.asarray(
        [[scale, 0.0, -scale * center[0]], [0.0, scale, -scale * center[1]], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    homogeneous = np.column_stack((points, np.ones(len(points))))
    normalized = (transform @ homogeneous.T).T
    return normalized[:, :2] / normalized[:, 2:3], transform


def _project(homography: np.ndarray, world: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack((world, np.ones(len(world))))
    projected = (homography @ homogeneous.T).T
    if np.any(np.abs(projected[:, 2]) <= 1.0e-12):
        raise ValueError("homography projected a point to infinity")
    return projected[:, :2] / projected[:, 2:3]


def _weighted_robust_cost(residuals: np.ndarray, weights: np.ndarray) -> float:
    scale = max(float(np.median(residuals)) * 1.4826, 1.0)
    normalized = residuals / scale
    losses = np.where(normalized <= 2.5, 0.5 * normalized**2, 2.5 * normalized - 3.125)
    return float((losses * weights).sum() / weights.sum())


__all__ = ["refine_planar_homography_and_k1"]

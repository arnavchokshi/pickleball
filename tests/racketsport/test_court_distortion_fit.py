from __future__ import annotations

import numpy as np
import pytest

from threed.racketsport.court_camera_geometry import (
    PinholeIntrinsics,
    distort_pixels_radial_k1,
)
from threed.racketsport.court_distortion_fit import refine_planar_homography_and_k1
from threed.racketsport.court_structured_solver import FLOOR_KEYPOINT_NAMES, FLOOR_WORLD_XY_M


def _project(homography: np.ndarray, world: np.ndarray) -> np.ndarray:
    values = (homography @ np.column_stack((world, np.ones(len(world)))).T).T
    return values[:, :2] / values[:, 2:3]


def test_joint_point_fit_recovers_known_bounded_radial_k1() -> None:
    intrinsics = PinholeIntrinsics(fx=720.0, fy=700.0, cx=640.0, cy=360.0)
    world = np.asarray([FLOOR_WORLD_XY_M[name] for name in FLOOR_KEYPOINT_NAMES])
    homography = np.asarray(
        [[72.0, 8.0, 640.0], [-4.0, 35.0, 360.0], [0.004, -0.012, 1.0]],
        dtype=np.float64,
    )
    true_k1 = -0.24
    observed = distort_pixels_radial_k1(
        _project(homography, world),
        intrinsics,
        k1=true_k1,
    )
    result = refine_planar_homography_and_k1(
        world,
        observed,
        intrinsics,
        k1_bounds=(-0.4, 0.2),
        grid_steps=61,
    )
    assert result["k1"] == pytest.approx(true_k1, abs=0.02)
    assert result["p95_residual_px"] < 0.2
    assert result["candidate_count"] >= 40


def test_distortion_fit_rejects_too_few_points() -> None:
    intrinsics = PinholeIntrinsics(fx=700.0, fy=700.0, cx=320.0, cy=180.0)
    with pytest.raises(ValueError, match="at least four"):
        refine_planar_homography_and_k1(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            [[10.0, 10.0], [20.0, 10.0], [10.0, 20.0]],
            intrinsics,
        )

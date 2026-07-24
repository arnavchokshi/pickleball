from __future__ import annotations

import numpy as np
import pytest

from threed.racketsport.court_camera_geometry import (
    PinholeIntrinsics,
    RadialDistortionAmbiguityError,
    distort_pixel_with_covariance_radial_k1,
    distort_pixels_radial_k1,
    project_camera_points_radial_k1,
    project_planar_point_with_covariance,
    undistort_pixels_radial_k1,
    validate_bounded_k1,
)


def test_bounded_radial_k1_pixel_roundtrip_and_camera_projection() -> None:
    intrinsics = PinholeIntrinsics(fx=960.0, fy=940.0, cx=640.0, cy=360.0)
    undistorted = np.asarray(
        [[640.0, 360.0], [800.0, 420.0], [390.0, 245.0], [1060.0, 590.0]],
        dtype=np.float64,
    )
    distorted = distort_pixels_radial_k1(undistorted, intrinsics, k1=-0.28)
    recovered, diagnostics = undistort_pixels_radial_k1(distorted, intrinsics, k1=-0.28)

    assert diagnostics.all_converged
    assert not diagnostics.any_ambiguous
    assert recovered == pytest.approx(undistorted, abs=1.0e-7)

    camera_points = np.column_stack(
        [
            (undistorted[:, 0] - intrinsics.cx) / intrinsics.fx,
            (undistorted[:, 1] - intrinsics.cy) / intrinsics.fy,
            np.ones(len(undistorted), dtype=np.float64),
        ]
    )
    projected = project_camera_points_radial_k1(camera_points, intrinsics, k1=-0.28)
    assert projected == pytest.approx(distorted, abs=1.0e-10)


def test_radial_k1_rejects_out_of_bounds_and_unstable_inverse() -> None:
    intrinsics = PinholeIntrinsics(fx=1000.0, fy=1000.0, cx=500.0, cy=400.0)
    with pytest.raises(ValueError, match="outside configured bounds"):
        validate_bounded_k1(-0.46)

    # For k1=-0.45 the central monotonic branch reaches only rd~=0.5738.
    impossible_distorted_pixel = np.asarray([[1200.0, 400.0]], dtype=np.float64)
    with pytest.raises(RadialDistortionAmbiguityError, match="turning point"):
        undistort_pixels_radial_k1(
            impossible_distorted_pixel,
            intrinsics,
            k1=-0.45,
            strict=True,
        )

    recovered, diagnostics = undistort_pixels_radial_k1(
        impossible_distorted_pixel,
        intrinsics,
        k1=-0.45,
        strict=False,
    )
    assert diagnostics.any_ambiguous
    assert not diagnostics.all_converged
    assert np.isnan(recovered).all()


def test_homography_and_k1_covariance_propagation_is_psd_and_uncertainty_sensitive() -> None:
    homography = np.asarray(
        [[72.0, 4.0, 640.0], [1.5, 31.0, 355.0], [0.002, -0.004, 1.0]],
        dtype=np.float64,
    )
    small_transform_covariance = np.eye(8, dtype=np.float64) * 1.0e-6
    large_transform_covariance = small_transform_covariance * 100.0
    point_small, covariance_small = project_planar_point_with_covariance(
        homography,
        (3.048, 6.7056),
        small_transform_covariance,
    )
    point_large, covariance_large = project_planar_point_with_covariance(
        homography,
        (3.048, 6.7056),
        large_transform_covariance,
    )
    assert point_large == pytest.approx(point_small)
    assert np.trace(covariance_large) == pytest.approx(np.trace(covariance_small) * 100.0)
    assert np.linalg.eigvalsh(covariance_small).min() >= -1.0e-12

    intrinsics = PinholeIntrinsics(fx=960.0, fy=940.0, cx=640.0, cy=360.0)
    _, covariance_known_k1 = distort_pixel_with_covariance_radial_k1(
        point_small,
        np.eye(2, dtype=np.float64),
        intrinsics,
        k1=-0.2,
        k1_variance=0.0,
    )
    _, covariance_uncertain_k1 = distort_pixel_with_covariance_radial_k1(
        point_small,
        np.eye(2, dtype=np.float64),
        intrinsics,
        k1=-0.2,
        k1_variance=0.01,
    )
    assert np.trace(covariance_uncertain_k1) > np.trace(covariance_known_k1)
    assert np.linalg.eigvalsh(covariance_uncertain_k1).min() >= -1.0e-12

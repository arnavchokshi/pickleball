from __future__ import annotations

import numpy as np
import pytest

from threed.racketsport.court_camera_geometry import (
    PinholeIntrinsics,
    project_camera_points_radial_k1,
)
from threed.racketsport.court_net_stage import (
    NET_CENTER_HEIGHT_M,
    NET_POINT_NAMES,
    NET_POST_HEIGHT_M,
    NET_WORLD_XYZ_M,
    apply_bounded_net_residual,
    decompose_floor_homography,
    evaluate_net_reprojection,
    project_regulation_net_top,
)


def _fixture() -> tuple[np.ndarray, PinholeIntrinsics, np.ndarray, np.ndarray]:
    intrinsics = PinholeIntrinsics(fx=900.0, fy=880.0, cx=640.0, cy=360.0)
    rotation = np.eye(3, dtype=np.float64)
    translation = np.asarray([0.2, -0.1, 20.0], dtype=np.float64)
    homography = intrinsics.matrix @ np.column_stack(
        (rotation[:, 0], rotation[:, 1], translation)
    )
    homography /= homography[2, 2]
    return homography, intrinsics, rotation, translation


def test_net_stage_recovers_pose_and_projects_regulation_36_34_36_heights() -> None:
    homography, intrinsics, rotation, translation = _fixture()
    recovered_rotation, recovered_translation = decompose_floor_homography(
        homography, intrinsics
    )
    assert recovered_rotation == pytest.approx(rotation, abs=1.0e-9)
    assert recovered_translation == pytest.approx(translation, abs=1.0e-9)

    result = project_regulation_net_top(
        homography,
        intrinsics,
        transform_covariance=np.eye(8) * 1.0e-8,
        floor_court_confidence=0.9,
    )
    expected = project_camera_points_radial_k1(
        (rotation @ NET_WORLD_XYZ_M.T).T + translation,
        intrinsics,
        k1=0.0,
    )
    actual = np.asarray([result["keypoints_xy"][name] for name in NET_POINT_NAMES])
    assert actual == pytest.approx(expected, abs=1.0e-7)
    assert result["world_xyz_m"]["net_left_sideline"][2] == pytest.approx(
        NET_POST_HEIGHT_M
    )
    assert result["world_xyz_m"]["net_center"][2] == pytest.approx(
        NET_CENTER_HEIGHT_M
    )
    assert result["world_xyz_m"]["net_right_sideline"][2] == pytest.approx(
        NET_POST_HEIGHT_M
    )
    assert result["floor_homography_fit_used_net_top"] is False
    assert result["measurement_valid"] is False


def test_net_confidence_declines_with_distortion_uncertainty() -> None:
    homography, intrinsics, _, _ = _fixture()
    stable = project_regulation_net_top(
        homography,
        intrinsics,
        transform_covariance=np.eye(8) * 1.0e-8,
        k1_variance=0.0,
        floor_court_confidence=0.9,
    )
    ambiguous = project_regulation_net_top(
        homography,
        intrinsics,
        transform_covariance=np.eye(8) * 1.0e-8,
        k1_variance=0.05,
        floor_court_confidence=0.9,
    )
    assert ambiguous["net_confidence"] < stable["net_confidence"]


def test_net_residual_is_bounded_and_evaluation_is_exact_semantic() -> None:
    base = {name: [100.0 + index * 10.0, 50.0] for index, name in enumerate(NET_POINT_NAMES)}
    corrected = apply_bounded_net_residual(
        base,
        {
            "net_left_sideline": [1000.0, 0.0],
            "net_center": [0.0, 2.0],
            "net_right_sideline": [-1000.0, 0.0],
        },
        max_residual_px=12.0,
    )
    assert corrected["net_left_sideline"][0] <= base["net_left_sideline"][0] + 12.0
    assert corrected["net_right_sideline"][0] >= base["net_right_sideline"][0] - 12.0
    report = evaluate_net_reprojection(corrected, base)
    assert set(report["point_error_px"]) == set(NET_POINT_NAMES)
    assert report["point_error_px"]["net_center"] == pytest.approx(
        abs(corrected["net_center"][1] - base["net_center"][1])
    )

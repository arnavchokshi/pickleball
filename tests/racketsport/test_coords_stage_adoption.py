from __future__ import annotations

import cv2
import numpy as np
import pytest

from threed.racketsport import ball_court_filter, ball_inout_uncertainty, ball_physics3d, coordinates, placement
from threed.racketsport.court_calibration import (
    project_image_points_to_world,
    project_planar_points,
)
from threed.racketsport.court_templates import get_court_template
from threed.racketsport.schemas import CourtCalibration
from threed.racketsport.virtual_world import _ball_world_xyz


def _distorted_calibration() -> CourtCalibration:
    template = get_court_template("pickleball")
    image_corners = [[140.0, 930.0], [1760.0, 900.0], [1510.0, 170.0], [390.0, 190.0]]
    from threed.racketsport.court_calibration import homography_from_planar_points

    homography = homography_from_planar_points(template.corners_m, image_corners)
    return CourtCalibration.model_validate(
        {
            "schema_version": 1,
            "sport": "pickleball",
            "homography": homography,
            "intrinsics": {
                "fx": 1187.5,
                "fy": 1179.25,
                "cx": 951.0,
                "cy": 537.5,
                "dist": [0.17, -0.08, 0.002, -0.001],
                "source": "distorted_synthetic_stage_adoption",
            },
            "extrinsics": {
                "R": [[0.9998, -0.0175, 0.009], [0.0174, 0.9998, 0.004], [-0.0091, -0.0038, 0.99995]],
                "t": [0.25, -0.1, 14.5],
                "camera_height_m": 14.5,
            },
            "reprojection_error_px": {"median": 0.0, "p95": 0.0},
            "capture_quality": {"grade": "warn", "reasons": ["synthetic"]},
            "image_pts": image_corners,
            "world_pts": template.corners_m,
        }
    )


def test_homography_declarations_preserve_legacy_default_and_fail_on_conflict() -> None:
    assert coordinates.resolve_homography_pixel_convention({}) == "raw_pixels"
    assert (
        coordinates.resolve_homography_pixel_convention(
            {
                "coordinate_contract": {
                    "homography_pixel_convention": "undistorted_pixels",
                    "homography_output_space": "pixels_undistorted_native",
                }
            }
        )
        == "undistorted_pixels"
    )
    with pytest.raises(ValueError, match="conflicting homography pixel declarations"):
        coordinates.resolve_homography_pixel_convention(
            {
                "homography_pixel_convention": "raw_pixels",
                "coordinate_contract": {"homography_pixel_convention": "undistorted_pixels"},
            }
        )


def test_placement_distorted_synthetic_undistort_and_unproject_match_legacy_exactly() -> None:
    calibration = _distorted_calibration()
    k = np.asarray(
        [
            [calibration.intrinsics.fx, 0.0, calibration.intrinsics.cx],
            [0.0, calibration.intrinsics.fy, calibration.intrinsics.cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    dist = np.asarray(calibration.intrinsics.dist, dtype=np.float64)
    raw = [1732.25, 811.75]
    legacy_pixel = cv2.undistortPoints(
        np.asarray([[raw]], dtype=np.float64), k, dist, P=k
    )[0, 0]
    inverse = np.linalg.inv(np.asarray(calibration.homography, dtype=float))
    legacy_h = inverse @ np.array([float(legacy_pixel[0]), float(legacy_pixel[1]), 1.0], dtype=float)
    legacy_world = legacy_h[:2] / legacy_h[2]

    typed_pixel = placement.undistort_pixel(
        raw,
        k,
        calibration.intrinsics.dist,
        input_space=coordinates.CoordinateSpace.PIXELS_RAW_NATIVE,
        output_space=coordinates.CoordinateSpace.PIXELS_UNDISTORTED_NATIVE,
    )
    signal = placement._signal_from_pixel(
        name="synthetic",
        pixel_xy=raw,
        confidence=1.0,
        sigma_px=2.0,
        side="near",
        homography=np.asarray(calibration.homography, dtype=float),
        camera_matrix=k,
        dist=calibration.intrinsics.dist,
        undistort_applied=True,
        homography_pixel_space=coordinates.CoordinateSpace.PIXELS_UNDISTORTED_NATIVE,
        config=placement.PlacementConfig(),
    )

    assert typed_pixel == [float(legacy_pixel[0]), float(legacy_pixel[1])]
    assert np.array_equal(np.asarray(signal["xy"]), legacy_world)
    with pytest.raises(ValueError, match="coordinate-space mismatch"):
        placement.homography_world_covariance(
            calibration.homography,
            typed_pixel,
            sigma_px=2.0,
            pixel_space=coordinates.CoordinateSpace.PIXELS_RAW_NATIVE,
            homography_space=coordinates.CoordinateSpace.PIXELS_UNDISTORTED_NATIVE,
        )


def test_ball_court_filter_distorted_synthetic_projection_and_scaling_match_legacy_exactly() -> None:
    calibration = _distorted_calibration()
    template = get_court_template("pickleball")
    legacy_native = project_planar_points(calibration.homography, template.corners_m)
    legacy_preview = [[float(x) * (960.0 / 1902.0), float(y) * (540.0 / 1075.0)] for x, y in legacy_native]

    typed_native = ball_court_filter.build_target_court_polygon(calibration)
    typed_preview = ball_court_filter.build_target_court_polygon(calibration, target_size=(960, 540))
    raw_point = [1650.0, 750.0]
    legacy_world = project_image_points_to_world(calibration.homography, [raw_point])[0]
    typed_world = ball_court_filter._target_image_xy_to_world_xy(
        raw_point,
        calibration=calibration,
        target_size=None,
    )

    assert typed_native == legacy_native
    assert typed_preview == legacy_preview
    assert typed_world == legacy_world
    with pytest.raises(ValueError, match="coordinate-space mismatch"):
        ball_court_filter.point_in_polygon_with_margin_typed(
            raw_point,
            typed_preview,
            margin_px=0.0,
            point_space=coordinates.CoordinateSpace.PIXELS_RAW_NATIVE,
            polygon_space=coordinates.CoordinateSpace.PIXELS_PREVIEW_SCALED,
        )


def test_ball_arc_distorted_synthetic_camera_projection_matches_legacy_exactly() -> None:
    calibration = _distorted_calibration()
    world = np.asarray([[-3.1, -6.7, 0.0], [0.2, 0.4, 1.15], [2.9, 6.5, 0.0]], dtype=float)
    rotation = np.asarray(calibration.extrinsics.R, dtype=float)
    translation = np.asarray(calibration.extrinsics.t, dtype=float)
    camera_points = (rotation @ world.T).T + translation
    depth = camera_points[:, 2]
    depth = np.where(np.abs(depth) < 1e-9, 1e-9, depth)
    legacy = np.column_stack(
        [
            calibration.intrinsics.fx * camera_points[:, 0] / depth + calibration.intrinsics.cx,
            calibration.intrinsics.fy * camera_points[:, 1] / depth + calibration.intrinsics.cy,
        ]
    )
    camera = {
        "intrinsics": calibration.intrinsics,
        "rotation": rotation,
        "translation": translation,
        "reference_space": coordinates.CoordinateSpace.PIXELS_RAW_NATIVE,
    }

    typed = ball_physics3d._project_world_array(world, camera=camera, np_module=np)
    assert np.array_equal(typed, legacy)
    camera["reference_space"] = coordinates.CoordinateSpace.WORLD_XY_HOMOGRAPHY_M
    with pytest.raises(ValueError, match="projection reference must be a raster space"):
        ball_physics3d._project_world_array(world, camera=camera, np_module=np)


def test_ball_inout_distorted_synthetic_missing_declaration_defaults_and_wrong_space_fails() -> None:
    calibration = _distorted_calibration()
    world = np.asarray(calibration.world_pts, dtype=np.float64)
    k = np.asarray(
        [
            [calibration.intrinsics.fx, 0.0, calibration.intrinsics.cx],
            [0.0, calibration.intrinsics.fy, calibration.intrinsics.cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    rvec, _ = cv2.Rodrigues(np.asarray(calibration.extrinsics.R, dtype=np.float64))
    distorted, _ = cv2.projectPoints(
        world,
        rvec,
        np.asarray(calibration.extrinsics.t, dtype=np.float64),
        k,
        np.asarray(calibration.intrinsics.dist, dtype=np.float64),
    )
    image = distorted.reshape(-1, 2).tolist()

    legacy_default = ball_inout_uncertainty.solve_manual_corner_camera_pose(image, world.tolist())
    typed_explicit = ball_inout_uncertainty.solve_manual_corner_camera_pose(
        image,
        world.tolist(),
        object_space=coordinates.CoordinateSpace.WORLD_COURT_NETCENTER_Z_UP_M,
        image_reference_space=coordinates.CoordinateSpace.PIXELS_RAW_NATIVE,
        projected_space=coordinates.CoordinateSpace.PIXELS_UNDISTORTED_NATIVE,
    )
    assert typed_explicit == legacy_default
    with pytest.raises(ValueError, match="solvePnP object space"):
        ball_inout_uncertainty.solve_manual_corner_camera_pose(
            image,
            world.tolist(),
            object_space=coordinates.CoordinateSpace.CAMERA_M,
        )


def test_virtual_world_distorted_synthetic_unprojection_matches_legacy_and_fails_wrong_space() -> None:
    calibration = _distorted_calibration()
    frame = {"t": 0.0, "xy": [951.0, 537.5], "conf": 1.0, "visible": True}
    legacy_world = project_image_points_to_world(calibration.homography, [frame["xy"]])[0]
    typed = _ball_world_xyz(
        frame,
        calibration=calibration,
        ball_world_policy="court_plane_approx_for_review_only",
    )
    expected = [float(legacy_world[0]), float(legacy_world[1]), 0.0]

    assert typed == expected
    with pytest.raises(ValueError, match="coordinate-space mismatch"):
        _ball_world_xyz(
            frame,
            calibration=calibration,
            ball_world_policy="court_plane_approx_for_review_only",
            image_space=coordinates.CoordinateSpace.PIXELS_PREVIEW_SCALED,
        )

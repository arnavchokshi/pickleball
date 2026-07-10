from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from threed.racketsport import coordinates, mhr_decode
from threed.racketsport.schemas import CameraIntrinsics


def test_coordinate_vocabulary_and_homography_conventions_are_stable() -> None:
    assert {space.value for space in coordinates.CoordinateSpace} >= {
        "pixels_raw_native",
        "pixels_undistorted_native",
        "pixels_preview_scaled",
        "camera_m",
        "body_camera_root_relative_m",
        "world_court_netcenter_z_up_m",
        "world_xy_homography_m",
    }
    assert coordinates.HOMOGRAPHY_PIXEL_CONVENTIONS == ("raw_pixels", "undistorted_pixels")


def test_coordinate_enum_is_python310_compatible_and_string_like() -> None:
    source = Path(coordinates.__file__).read_text(encoding="utf-8")
    assert re.search(r"from\s+enum\s+import[^\n]*\bStrEnum\b", source) is None
    for member in coordinates.CoordinateSpace:
        assert isinstance(member, str)
        assert member == member.value
        assert str(member) == member.value


def test_module_import_is_lightweight() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json,sys; import threed.racketsport.coordinates; "
                "print(json.dumps({'torch': 'torch' in sys.modules, 'cv2': 'cv2' in sys.modules}))"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout) == {"torch": False, "cv2": False}


def test_extrinsic_inverse_and_point_round_trip_are_vectorized() -> None:
    rotation = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    translation = np.asarray([2.0, -3.0, 4.0])
    points = np.asarray([[1.0, 2.0, 3.0], [-4.0, 5.0, 6.0]])

    camera_to_world, center = coordinates.invert_extrinsics(rotation, translation)
    assert camera_to_world == pytest.approx(rotation.T)
    assert center == pytest.approx(-(rotation.T @ translation))
    camera = coordinates.world_to_camera_points(points, rotation, translation)
    assert coordinates.camera_to_world_points(camera, rotation, translation) == pytest.approx(points)


def test_translation_policy_and_mhr_wrapper_preserve_exactly_once_behavior() -> None:
    points = [[1.0, 2.0, 3.0]]
    assert coordinates.apply_translation_once(points, [0.25, -0.5, 1.0]) == [[1.25, 1.5, 4.0]]
    assert coordinates.apply_translation_once([[1.25, 1.5, 4.0]], [0.25, -0.5, 1.0], True) == [
        [1.25, 1.5, 4.0]
    ]
    assert mhr_decode.apply_pred_cam_t_once(points, pred_cam_t=[0.25, -0.5, 1.0]) == [[1.25, 1.5, 4.0]]
    with pytest.raises(ValueError, match="pred_cam_t must be a 3-vector"):
        mhr_decode.apply_pred_cam_t_once(points, pred_cam_t=[1.0, 2.0])


def test_blessed_camera_matrix_builder_delegates_to_court_calibration() -> None:
    intrinsics = CameraIntrinsics(fx=1000.0, fy=900.0, cx=640.0, cy=360.0, dist=[], source="unit")
    assert coordinates.camera_matrix_from_intrinsics(intrinsics) == [
        [1000.0, 0.0, 640.0],
        [0.0, 900.0, 360.0],
        [0.0, 0.0, 1.0],
    ]

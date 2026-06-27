from __future__ import annotations

import json
import subprocess
import sys

import pytest

from threed.racketsport.court_calibration import (
    CALIBRATION_REPROJECTION_MEDIAN_GATE_PX,
    CALIBRATION_REPROJECTION_P95_GATE_PX,
    calibration_from_manual_taps,
    calibration_from_manual_tap_frames,
    camera_matrix_from_intrinsics,
    homography_from_planar_points,
    manual_tap_correspondences,
    passes_reprojection_gate,
    project_image_points_to_world,
    project_world_points,
    project_planar_points,
    reprojection_error,
    solve_camera_pose,
)
from threed.racketsport.court_templates import FT_TO_M
from threed.racketsport.schemas import CameraIntrinsics, CaptureSidecar, CourtCalibration, CourtExtrinsics, CourtZones, NetPlane, validate_artifact_file


def _capture_sidecar_payload() -> dict:
    return {
        "schema_version": 1,
        "device_tier": "B_standard",
        "device_model": "iPhone16,2",
        "fps": 120,
        "format": "hevc",
        "resolution": [1920, 1080],
        "orientation": "landscape",
        "locked": {
            "exposure_s": 0.001,
            "iso": 320,
            "focus": 0.7,
            "wb_locked": True,
        },
        "intrinsics": {
            "fx": 1000.0,
            "fy": 1010.0,
            "cx": 960.0,
            "cy": 540.0,
            "dist": [0.0, 0.0, 0.0, 0.0],
            "source": "arkit",
        },
        "arkit_camera_pose": {
            "R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "t": [0.0, 0.0, 15.0],
        },
        "court_plane": {"point": [0.0, 0.0, 0.0], "normal": [0.0, 0.0, 1.0]},
        "manual_court_taps": [[756.8, 88.4896], [1163.2, 88.4896], [1163.2, 991.5104], [756.8, 991.5104]],
        "gravity": [0.0, -1.0, 0.0],
        "lidar_depth_refs": [],
        "ondevice_pose_track": "ondevice_pose.json",
        "capture_quality": {"grade": "good", "reasons": []},
    }


def test_camera_matrix_uses_sidecar_intrinsics():
    sidecar = CaptureSidecar.model_validate(_capture_sidecar_payload())

    assert camera_matrix_from_intrinsics(sidecar.intrinsics) == [
        [1000.0, 0.0, 960.0],
        [0.0, 1010.0, 540.0],
        [0.0, 0.0, 1.0],
    ]


def test_manual_tap_correspondences_map_to_template_corners():
    sidecar = CaptureSidecar.model_validate(_capture_sidecar_payload())

    image_pts, world_pts = manual_tap_correspondences(sidecar, sport="pickleball")

    assert image_pts == sidecar.manual_court_taps
    assert world_pts == [
        [-10.0 * FT_TO_M, -22.0 * FT_TO_M, 0.0],
        [10.0 * FT_TO_M, -22.0 * FT_TO_M, 0.0],
        [10.0 * FT_TO_M, 22.0 * FT_TO_M, 0.0],
        [-10.0 * FT_TO_M, 22.0 * FT_TO_M, 0.0],
    ]


def test_manual_tap_correspondences_require_four_taps():
    payload = _capture_sidecar_payload()
    payload["manual_court_taps"] = [[100.0, 900.0], [1820.0, 900.0], [1720.0, 120.0]]
    sidecar = CaptureSidecar.model_validate(payload)

    with pytest.raises(ValueError, match="at least 4 manual court taps"):
        manual_tap_correspondences(sidecar, sport="pickleball")


def test_reprojection_error_reports_median_and_p95():
    observed = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]
    projected = [[0.0, 0.0], [13.0, 4.0], [10.0, 22.0], [0.0, 10.0]]

    stats = reprojection_error(observed, projected)

    assert stats.median == pytest.approx(2.5)
    assert stats.p95 == pytest.approx(10.95)
    assert passes_reprojection_gate(stats)
    assert CALIBRATION_REPROJECTION_MEDIAN_GATE_PX == pytest.approx(8.0)
    assert CALIBRATION_REPROJECTION_P95_GATE_PX == pytest.approx(15.0)


def test_homography_maps_court_world_to_image_pixels():
    homography = homography_from_planar_points(
        world_pts=[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        image_pts=[[10.0, 20.0], [30.0, 20.0], [30.0, 25.0], [10.0, 25.0]],
    )

    assert project_planar_points(homography, [[1.0, 0.5, 0.0]])[0] == pytest.approx([20.0, 22.5])


def test_image_pixels_project_back_to_court_world_plane():
    homography = homography_from_planar_points(
        world_pts=[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        image_pts=[[10.0, 20.0], [30.0, 20.0], [30.0, 25.0], [10.0, 25.0]],
    )

    world_points = project_image_points_to_world(homography, [[20.0, 22.5]])

    assert world_points[0] == pytest.approx([1.0, 0.5])


def test_manual_calibration_artifact_is_schema_valid_and_gate_scored(tmp_path):
    sidecar_path = tmp_path / "capture_sidecar.json"
    sidecar_path.write_text(json.dumps(_capture_sidecar_payload()), encoding="utf-8")

    calibration = calibration_from_manual_taps(sidecar_path, sport="pickleball")

    assert isinstance(calibration, CourtCalibration)
    assert calibration.sport == "pickleball"
    assert calibration.intrinsics.source == "arkit"
    assert calibration.reprojection_error_px.median == pytest.approx(0.0)
    assert calibration.reprojection_error_px.p95 == pytest.approx(0.0)
    assert calibration.capture_quality.grade == "good"
    assert calibration.extrinsics.camera_height_m == pytest.approx(15.0)
    assert len(calibration.image_pts) == 4
    assert len(calibration.world_pts) == 4


def test_multiframe_manual_taps_average_static_correspondences(tmp_path):
    paths = []
    jitters = [
        [[-2.0, 1.0], [2.0, -1.0], [2.0, 1.0], [-2.0, -1.0]],
        [[0.0, -2.0], [0.0, 2.0], [0.0, -2.0], [0.0, 2.0]],
        [[2.0, 1.0], [-2.0, -1.0], [-2.0, 1.0], [2.0, -1.0]],
    ]
    base = _capture_sidecar_payload()["manual_court_taps"]
    for idx, jitter in enumerate(jitters):
        payload = _capture_sidecar_payload()
        payload["manual_court_taps"] = [
            [point[0] + delta[0], point[1] + delta[1]] for point, delta in zip(base, jitter, strict=True)
        ]
        path = tmp_path / f"sidecar_{idx}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        paths.append(path)

    calibration = calibration_from_manual_tap_frames(paths, sport="pickleball")

    for actual, expected in zip(calibration.image_pts, base, strict=True):
        assert actual == pytest.approx(expected)
    assert calibration.reprojection_error_px.median == pytest.approx(0.0)


def test_calibrate_cli_writes_calibration_zones_and_net_plane(tmp_path):
    sidecar_path = tmp_path / "capture_sidecar.json"
    out_dir = tmp_path / "calib"
    sidecar_path.write_text(json.dumps(_capture_sidecar_payload()), encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/calibrate.py",
            "--sidecar",
            str(sidecar_path),
            "--sport",
            "pickleball",
            "--out",
            str(out_dir),
        ],
        check=True,
    )

    calibration = validate_artifact_file("court_calibration", out_dir / "court_calibration.json")
    zones = validate_artifact_file("court_zones", out_dir / "court_zones.json")
    net = validate_artifact_file("net_plane", out_dir / "net_plane.json")

    assert isinstance(calibration, CourtCalibration)
    assert isinstance(zones, CourtZones)
    assert isinstance(net, NetPlane)


def test_solve_camera_pose_uses_opencv_when_available():
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    intrinsics = CameraIntrinsics(fx=1000.0, fy=1000.0, cx=960.0, cy=540.0, dist=[], source="arkit")
    world_pts = [
        [-3.0, -6.0, 0.0],
        [3.0, -6.0, 0.0],
        [3.0, 6.0, 0.0],
        [-3.0, 6.0, 0.0],
        [0.0, -3.0, 0.0],
        [0.0, 3.0, 0.0],
    ]
    expected_rvec = np.zeros((3, 1), dtype=np.float64)
    expected_tvec = np.array([[0.2], [0.1], [15.0]], dtype=np.float64)
    image_pts, _ = cv2.projectPoints(
        np.asarray(world_pts, dtype=np.float64),
        expected_rvec,
        expected_tvec,
        np.asarray(camera_matrix_from_intrinsics(intrinsics), dtype=np.float64),
        None,
    )

    extrinsics = solve_camera_pose(world_pts, image_pts.reshape(-1, 2).tolist(), intrinsics)

    assert extrinsics.t == pytest.approx([0.2, 0.1, 15.0], abs=1e-4)
    assert extrinsics.camera_height_m == pytest.approx(15.0, abs=1e-4)
    assert extrinsics.R[0] == pytest.approx([1.0, 0.0, 0.0], abs=1e-4)


def test_project_world_points_uses_camera_extrinsics_and_intrinsics():
    intrinsics = CameraIntrinsics(fx=1000.0, fy=1000.0, cx=960.0, cy=540.0, dist=[], source="arkit")
    extrinsics = CourtExtrinsics(
        R=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        t=[0.0, 0.0, 15.0],
        camera_height_m=15.0,
    )

    projected = project_world_points(extrinsics, intrinsics, [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])

    assert projected[0] == pytest.approx([960.0, 540.0])
    assert projected[1] == pytest.approx([1160.0, 540.0])


def test_manual_calibration_prefers_solved_pose_when_opencv_is_available(tmp_path):
    pytest.importorskip("cv2")
    pytest.importorskip("numpy")
    payload = _capture_sidecar_payload()
    payload["arkit_camera_pose"]["t"] = [0.0, 0.0, 12.0]
    sidecar_path = tmp_path / "capture_sidecar.json"
    sidecar_path.write_text(json.dumps(payload), encoding="utf-8")

    calibration = calibration_from_manual_taps(sidecar_path, sport="pickleball")

    assert calibration.extrinsics.t == pytest.approx([0.0, 0.0, 15.0], abs=1e-4)
    assert calibration.extrinsics.camera_height_m == pytest.approx(15.0, abs=1e-4)

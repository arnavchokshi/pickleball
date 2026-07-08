from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from threed.racketsport.court_calibration import metric_calibration_from_sidecar_and_keypoints
from threed.racketsport.court_keypoint_net import PICKLEBALL_KEYPOINT_BY_NAME, PICKLEBALL_KEYPOINTS
from threed.racketsport.court_positioning_artifacts import build_court_keypoints_artifact
from threed.racketsport.owner_capture_intake import inject_device_profile_distortion
from threed.racketsport.profile_registry import DeviceProfile, update_profile


def test_metric_calibration_undistorts_distorted_edge_keypoints_toward_truth(tmp_path: Path) -> None:
    pytest.importorskip("cv2")
    sidecar = _metric_sidecar_payload(dist=[-0.24, 0.08, 0.0, 0.0])
    keypoints = _distorted_keypoints_payload(sidecar)
    truth_name = "far_left_corner"
    truth_xy = PICKLEBALL_KEYPOINT_BY_NAME[truth_name].world_xyz_m[:2]
    ground_sample_m_per_px = 10.0 / sidecar["intrinsics"]["fx"]

    undistorted_path = tmp_path / "capture_sidecar_distorted.json"
    zero_dist_path = tmp_path / "capture_sidecar_zero_dist.json"
    keypoints_path = tmp_path / "court_keypoints.json"
    undistorted_path.write_text(json.dumps(sidecar), encoding="utf-8")
    zero_sidecar = json.loads(json.dumps(sidecar))
    zero_sidecar["intrinsics"]["dist"] = [0.0, 0.0, 0.0, 0.0]
    zero_dist_path.write_text(json.dumps(zero_sidecar), encoding="utf-8")
    keypoints_path.write_text(json.dumps(keypoints), encoding="utf-8")

    with_dist = metric_calibration_from_sidecar_and_keypoints(undistorted_path, keypoints_path, sport="pickleball")
    zero_dist = metric_calibration_from_sidecar_and_keypoints(zero_dist_path, keypoints_path, sport="pickleball")

    idx = with_dist.solved_over_frames and _keypoint_index(truth_name)
    assert idx is not False
    with_dist_error_m = _xy_distance(with_dist.world_pts[idx], truth_xy)
    zero_dist_error_m = _xy_distance(zero_dist.world_pts[idx], truth_xy)
    improvement_px_equiv = (zero_dist_error_m - with_dist_error_m) / ground_sample_m_per_px

    assert "undistort_applied" in with_dist.capture_quality.reasons
    assert with_dist_error_m < 0.025
    assert improvement_px_equiv > 2.0


def test_metric_calibration_entrypoint_changes_edge_floor_point_when_dist_is_nonzero(tmp_path: Path) -> None:
    sidecar = _metric_sidecar_payload(dist=[-0.24, 0.08, 0.0, 0.0])
    keypoints_path = tmp_path / "court_keypoints.json"
    nonzero_path = tmp_path / "capture_sidecar_nonzero.json"
    zero_path = tmp_path / "capture_sidecar_zero.json"
    nonzero_path.write_text(json.dumps(sidecar), encoding="utf-8")
    zero_sidecar = json.loads(json.dumps(sidecar))
    zero_sidecar["intrinsics"]["dist"] = [0.0, 0.0, 0.0, 0.0]
    zero_path.write_text(json.dumps(zero_sidecar), encoding="utf-8")
    keypoints_path.write_text(json.dumps(_distorted_keypoints_payload(sidecar)), encoding="utf-8")

    nonzero = metric_calibration_from_sidecar_and_keypoints(nonzero_path, keypoints_path, sport="pickleball")
    zero = metric_calibration_from_sidecar_and_keypoints(zero_path, keypoints_path, sport="pickleball")

    edge_idx = _keypoint_index("far_left_corner")
    ground_sample_m_per_px = 10.0 / sidecar["intrinsics"]["fx"]
    delta_px_equiv = _xy_distance(nonzero.world_pts[edge_idx], zero.world_pts[edge_idx]) / ground_sample_m_per_px

    assert delta_px_equiv > 2.0


def test_owner_intake_injects_matching_device_profile_distortion_and_leaves_unmatched_zero(tmp_path: Path) -> None:
    profiles_root = tmp_path / "profiles"
    account_id = "owner_1"
    profile_dist = [0.011, -0.023, 0.001, -0.002]
    update_profile(account_id, _device_profile(account_id=account_id, dist=profile_dist), profiles_root=profiles_root)
    sidecar = _owner_sidecar(lens="wide", zoom=1.0)

    injected = inject_device_profile_distortion(sidecar, account_id=account_id, profiles_root=profiles_root)

    assert injected["intrinsics"]["dist"] == pytest.approx(profile_dist)
    assert sidecar["intrinsics"]["dist"] == [0.0, 0.0, 0.0, 0.0]

    unmatched = inject_device_profile_distortion(
        _owner_sidecar(lens="telephoto", zoom=1.0),
        account_id=account_id,
        profiles_root=profiles_root,
    )
    assert unmatched["intrinsics"]["dist"] == [0.0, 0.0, 0.0, 0.0]


def _metric_sidecar_payload(*, dist: list[float]) -> dict:
    return {
        "schema_version": 1,
        "device_tier": "B_standard",
        "device_model": "iPhone16,2",
        "fps": 120,
        "format": "hevc",
        "resolution": [1280, 720],
        "orientation": "landscape",
        "camera_lens": "wide",
        "locked": {
            "exposure_s": 0.001,
            "iso": 320,
            "focus": 0.7,
            "wb_locked": True,
        },
        "intrinsics": {
            "fx": 600.0,
            "fy": 600.0,
            "cx": 640.0,
            "cy": 360.0,
            "dist": dist,
            "source": "arkit",
        },
        "arkit_camera_pose": {
            "R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]],
            "t": [1.2, -0.8, 10.0],
        },
        "court_plane": {"point": [0.0, 0.0, 0.0], "normal": [0.0, 0.0, 1.0]},
        "manual_court_taps": [],
        "gravity": [0.0, -1.0, 0.0],
        "lidar_depth_refs": [],
        "capture_quality": {"grade": "good", "reasons": []},
    }


def _distorted_keypoints_payload(sidecar: dict) -> dict:
    intrinsics = sidecar["intrinsics"]
    dist = intrinsics["dist"]
    camera_origin = sidecar["arkit_camera_pose"]["t"]
    keypoints = {}
    for point in PICKLEBALL_KEYPOINTS:
        x = (point.world_xyz_m[0] - camera_origin[0]) / camera_origin[2]
        y = (point.world_xyz_m[1] - camera_origin[1]) / camera_origin[2]
        r2 = x * x + y * y
        radial = 1.0 + dist[0] * r2 + dist[1] * r2 * r2
        x_distorted = x * radial
        y_distorted = y * radial
        keypoints[point.name] = {
            "uv": [
                intrinsics["cx"] + intrinsics["fx"] * x_distorted,
                intrinsics["cy"] + intrinsics["fy"] * y_distorted,
            ],
            "confidence": 0.95,
            "inlier_frames": [0, 15, 29],
            "recovered": False,
        }
    return build_court_keypoints_artifact(
        frame_indexes=[0, 15, 29],
        keypoints=keypoints,
        target_court_score=0.99,
        source="synthetic_distorted_keypoints",
    )


def _device_profile(*, account_id: str, dist: list[float]) -> DeviceProfile:
    trace = {"source_clip_id": "charuco_owner_wide_1x", "source_clip_ref": "runs/charuco/wide.mov"}
    retention = {
        "scope": "account_lifetime",
        "delete_with_source_clip": True,
        "delete_with_source_profile": True,
        "retention_days": None,
        "legal_basis": "owner_setup",
    }
    return DeviceProfile(
        schema_version=1,
        artifact_type="racketsport_device_profile",
        account_id=account_id,
        profile_id="iphone16_wide",
        display_name="iPhone 16 Wide",
        version=1,
        source_trace=trace,
        retention=retention,
        device_key="iphone16-owner",
        intrinsics_by_lens_zoom=[
            {
                "lens": "wide",
                "zoom": 1.0,
                "intrinsics": {
                    "fx": 610.0,
                    "fy": 609.0,
                    "cx": 640.0,
                    "cy": 360.0,
                    "dist": dist,
                    "source": "charuco_sweep",
                },
                "source_trace": trace,
            }
        ],
        exposure_constant=1.0,
    )


def _owner_sidecar(*, lens: str, zoom: float) -> dict:
    return {
        "device_key": "iphone16-owner",
        "camera_lens": lens,
        "camera_zoom": zoom,
        "intrinsics": {
            "fx": 600.0,
            "fy": 600.0,
            "cx": 640.0,
            "cy": 360.0,
            "dist": [0.0, 0.0, 0.0, 0.0],
            "source": "arkit",
        },
    }


def _keypoint_index(name: str) -> int:
    return [point.name for point in PICKLEBALL_KEYPOINTS].index(name)


def _xy_distance(point: list[float] | tuple[float, ...], truth_xy: tuple[float, float] | list[float]) -> float:
    return math.hypot(float(point[0]) - float(truth_xy[0]), float(point[1]) - float(truth_xy[1]))

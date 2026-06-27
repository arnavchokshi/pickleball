from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from threed.racketsport.schemas import (
    CaptureSidecar,
    CourtCalibration,
    validate_artifact_file,
)


def test_capture_sidecar_schema_accepts_documented_payload(tmp_path):
    payload = {
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
            "fy": 1000.0,
            "cx": 960.0,
            "cy": 540.0,
            "dist": [0.0, 0.0, 0.0, 0.0],
            "source": "arkit",
        },
        "arkit_camera_pose": {
            "R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "t": [0.0, 1.5, 0.0],
        },
        "court_plane": {"point": [0.0, 0.0, 0.0], "normal": [0.0, 1.0, 0.0]},
        "manual_court_taps": [[10.0, 10.0], [100.0, 10.0], [100.0, 80.0], [10.0, 80.0]],
        "gravity": [0.0, -1.0, 0.0],
        "lidar_depth_refs": [],
        "ondevice_pose_track": "ondevice_pose.json",
        "capture_quality": {"grade": "good", "reasons": []},
    }
    path = tmp_path / "capture_sidecar.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    parsed = validate_artifact_file("capture_sidecar", path)

    assert isinstance(parsed, CaptureSidecar)
    assert parsed.fps == 120
    assert parsed.capture_quality.grade == "good"


def test_court_calibration_requires_current_schema_version():
    payload = {
        "schema_version": 2,
        "sport": "pickleball",
        "homography": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "intrinsics": {"fx": 1, "fy": 1, "cx": 0, "cy": 0, "dist": [], "source": "manual"},
        "extrinsics": {"R": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "t": [0, 0, 0], "camera_height_m": 1.4},
        "reprojection_error_px": {"median": 2.0, "p95": 5.0},
        "capture_quality": {"grade": "good", "reasons": []},
        "image_pts": [[0, 0], [1, 0], [1, 1], [0, 1]],
        "world_pts": [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
    }

    with pytest.raises(ValidationError):
        CourtCalibration.model_validate(payload)


def test_validate_artifact_file_rejects_unknown_artifact(tmp_path):
    path = tmp_path / "unknown.json"
    path.write_text("{}", encoding="utf-8")

    with pytest.raises(KeyError):
        validate_artifact_file("not_real", path)

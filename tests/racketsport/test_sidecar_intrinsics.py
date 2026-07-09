from __future__ import annotations

import json

import pytest

from threed.racketsport.intrinsics import get_intrinsics
from threed.racketsport.schemas import CameraIntrinsics, CaptureSidecar
from threed.racketsport.sidecar import load_capture_sidecar


def _capture_sidecar_payload() -> dict:
    return {
        "schema_version": 1,
        "provenance": "live_recording",
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


def test_load_capture_sidecar_reads_direct_sidecar_path(tmp_path):
    sidecar_path = tmp_path / "capture_sidecar.json"
    sidecar_path.write_text(json.dumps(_capture_sidecar_payload()), encoding="utf-8")

    sidecar = load_capture_sidecar(sidecar_path)

    assert isinstance(sidecar, CaptureSidecar)
    assert sidecar.device_model == "iPhone16,2"
    assert sidecar.provenance == "live_recording"


def test_load_capture_sidecar_reads_capture_sidecar_from_clip_directory(tmp_path):
    clip_dir = tmp_path / "clip_001"
    clip_dir.mkdir()
    (clip_dir / "capture_sidecar.json").write_text(json.dumps(_capture_sidecar_payload()), encoding="utf-8")

    sidecar = load_capture_sidecar(clip_dir)

    assert isinstance(sidecar, CaptureSidecar)
    assert sidecar.fps == 120


def test_load_capture_sidecar_fails_closed_when_clip_directory_has_no_sidecar(tmp_path):
    clip_dir = tmp_path / "clip_001"
    clip_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="capture_sidecar.json"):
        load_capture_sidecar(clip_dir)


def test_get_intrinsics_returns_sidecar_camera_intrinsics_from_clip_directory(tmp_path):
    clip_dir = tmp_path / "clip_001"
    clip_dir.mkdir()
    (clip_dir / "capture_sidecar.json").write_text(json.dumps(_capture_sidecar_payload()), encoding="utf-8")

    intrinsics = get_intrinsics(clip_dir)

    assert isinstance(intrinsics, CameraIntrinsics)
    assert intrinsics.fx == pytest.approx(1000.0)
    assert intrinsics.source == "arkit"


def test_get_intrinsics_accepts_direct_sidecar_path(tmp_path):
    sidecar_path = tmp_path / "capture_sidecar.json"
    sidecar_path.write_text(json.dumps(_capture_sidecar_payload()), encoding="utf-8")

    intrinsics = get_intrinsics(sidecar_path)

    assert isinstance(intrinsics, CameraIntrinsics)
    assert intrinsics.fy == pytest.approx(1010.0)

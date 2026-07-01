from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from threed.racketsport.ball_capture_protocol import build_ball_capture_protocol_report


def _write_test_video(path: Path, *, size: str = "1920x1080", fps: int = 60, audio: bool = True) -> None:
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is required to synthesize capture-protocol test video")
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"testsrc=size={size}:rate={fps}:duration=0.2",
    ]
    if audio:
        command.extend(["-f", "lavfi", "-i", "sine=frequency=1000:sample_rate=48000:duration=0.2"])
    command.extend(["-pix_fmt", "yuv420p", "-shortest", str(path)])
    subprocess.run(command, check=True)


def _capture_sidecar_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": 1,
        "device_tier": "B_standard",
        "device_model": "iPhone16,2",
        "fps": 60,
        "format": "hevc",
        "resolution": [1920, 1080],
        "orientation": "landscape",
        "locked": {"exposure_s": 0.001, "iso": 320, "focus": 0.7, "wb_locked": True},
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
            "t": [0.0, 0.0, 1.6],
        },
        "court_plane": {"point": [0.0, 0.0, 0.0], "normal": [0.0, 0.0, 1.0]},
        "manual_court_taps": [[756.8, 88.4896], [1163.2, 88.4896], [1163.2, 991.5104], [756.8, 991.5104]],
        "gravity": [0.0, -1.0, 0.0],
        "lidar_depth_refs": [],
        "ondevice_pose_track": None,
        "capture_quality": {"grade": "good", "reasons": []},
        "hdr_enabled": False,
        "video_stabilization_enabled": False,
        "exposure_locked": True,
        "focus_locked": True,
        "tripod_height_m": 1.6,
        "full_court_visible": True,
        "court_lock_passed": True,
        "ball_high_contrast": True,
        "audio_recorded": True,
    }
    payload.update(overrides)
    return payload


def test_ball_capture_protocol_fails_closed_when_sidecar_is_missing(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    _write_test_video(video)

    report = build_ball_capture_protocol_report(video_path=video, sidecar_path=None)

    assert report["status"] == "TESTED-ON-REAL-DATA"
    assert report["gate_result"] == "fail"
    assert report["blocked_reason"] == "missing_capture_sidecar"
    assert "missing_capture_sidecar" in report["violations"]
    assert report["video"]["resolution"] == [1920, 1080]
    assert report["video"]["fps"] == pytest.approx(60.0)
    assert report["video"]["audio_present"] is True


def test_ball_capture_protocol_reports_video_violations_when_sidecar_is_missing(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    _write_test_video(video, size="1280x720", fps=30, audio=False)

    report = build_ball_capture_protocol_report(video_path=video, sidecar_path=None)

    assert report["gate_result"] == "fail"
    assert report["blocked_reason"] == "missing_capture_sidecar"
    assert set(report["violations"]) >= {
        "missing_capture_sidecar",
        "resolution_below_1080p",
        "fps_below_60",
        "audio_missing",
    }


def test_ball_capture_protocol_passes_strict_sidecar_and_real_video(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    sidecar = tmp_path / "capture_sidecar.json"
    _write_test_video(video)
    sidecar.write_text(json.dumps(_capture_sidecar_payload()), encoding="utf-8")

    report = build_ball_capture_protocol_report(video_path=video, sidecar_path=sidecar)

    assert report["status"] == "TESTED-ON-REAL-DATA"
    assert report["gate_result"] == "pass"
    assert report["blocked_reason"] is None
    assert report["violations"] == []


def test_ball_capture_protocol_requires_arkit_or_manual_court_seed(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    sidecar = tmp_path / "capture_sidecar.json"
    _write_test_video(video)
    sidecar.write_text(
        json.dumps(
            _capture_sidecar_payload(
                arkit_camera_pose=None,
                court_plane=None,
                manual_court_taps=[],
            )
        ),
        encoding="utf-8",
    )

    report = build_ball_capture_protocol_report(video_path=video, sidecar_path=sidecar)

    assert report["gate_result"] == "fail"
    assert report["blocked_reason"] == "capture_protocol_failed"
    assert "court_calibration_seed_missing" in report["violations"]


def test_ball_capture_protocol_rejects_incomplete_manual_court_taps(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    sidecar = tmp_path / "capture_sidecar.json"
    _write_test_video(video)
    sidecar.write_text(
        json.dumps(
            _capture_sidecar_payload(
                arkit_camera_pose=None,
                court_plane=None,
                manual_court_taps=[[100.0, 100.0], [200.0, 100.0], [200.0, 200.0]],
            )
        ),
        encoding="utf-8",
    )

    report = build_ball_capture_protocol_report(video_path=video, sidecar_path=sidecar)

    assert report["gate_result"] == "fail"
    assert report["blocked_reason"] == "capture_protocol_failed"
    assert "manual_court_taps_incomplete" in report["violations"]


def test_ball_capture_protocol_reports_every_spec_violation(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    sidecar = tmp_path / "capture_sidecar.json"
    _write_test_video(video, size="1280x720", fps=30, audio=False)
    sidecar.write_text(
        json.dumps(
            _capture_sidecar_payload(
                fps=30,
                resolution=[1280, 720],
                orientation="portrait",
                locked={"exposure_s": 1 / 250, "iso": 640, "focus": 0.2, "wb_locked": False},
                hdr_enabled=True,
                video_stabilization_enabled=True,
                exposure_locked=False,
                focus_locked=False,
                tripod_height_m=1.2,
                full_court_visible=False,
                court_lock_passed=False,
                ball_high_contrast=False,
                audio_recorded=False,
                capture_quality={"grade": "poor", "reasons": ["too_dark"]},
            )
        ),
        encoding="utf-8",
    )

    report = build_ball_capture_protocol_report(video_path=video, sidecar_path=sidecar)

    assert report["gate_result"] == "fail"
    assert report["blocked_reason"] == "capture_protocol_failed"
    assert set(report["violations"]) >= {
        "resolution_below_1080p",
        "fps_below_60",
        "orientation_not_landscape",
        "shutter_slower_than_1_500",
        "hdr_enabled",
        "video_stabilization_enabled",
        "exposure_not_locked",
        "focus_not_locked",
        "white_balance_not_locked",
        "tripod_height_below_1_5m",
        "full_court_not_visible",
        "court_lock_failed",
        "ball_low_contrast",
        "audio_missing",
        "capture_quality_poor",
    }


def test_ball_capture_protocol_cli_writes_report(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    out = tmp_path / "capture_protocol_report.json"
    _write_test_video(video)

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/validate_ball_capture.py",
            "--video",
            str(video),
            "--out",
            str(out),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    assert json.loads(completed.stdout)["blocked_reason"] == "missing_capture_sidecar"
    assert json.loads(out.read_text(encoding="utf-8"))["gate_result"] == "fail"

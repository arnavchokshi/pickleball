from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from threed.racketsport.ball_court_calibration_gate import build_ball_court_calibration_gate_report
from threed.racketsport.schemas import PICKLEBALL_COURT_KEYPOINT_NAMES


def _write_test_video(path: Path, *, size: str = "1920x1080", fps: int = 60) -> None:
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is required to synthesize calibration-gate test video")
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=size={size}:rate={fps}:duration=0.2",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
    )


def _base_calibration_payload(**overrides: object) -> dict[str, object]:
    image_pts = [[500.0 + idx * 10.0, 300.0 + idx * 5.0] for idx in range(len(PICKLEBALL_COURT_KEYPOINT_NAMES))]
    world_pts = [[float(idx), float(idx % 3), 0.0] for idx in range(len(PICKLEBALL_COURT_KEYPOINT_NAMES))]
    payload: dict[str, object] = {
        "schema_version": 1,
        "sport": "pickleball",
        "coordinate_frame": "court_netcenter_z_up_m",
        "T_world_court": [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "homography": [[1.0, 0.0, 500.0], [0.0, 1.0, 300.0], [0.0, 0.0, 1.0]],
        "intrinsics": {
            "fx": 1000.0,
            "fy": 1010.0,
            "cx": 960.0,
            "cy": 540.0,
            "dist": [0.0, 0.0, 0.0, 0.0],
            "source": "arkit",
        },
        "image_size": [1920, 1080],
        "extrinsics": {
            "R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "t": [0.0, 0.0, 10.0],
            "camera_height_m": 10.0,
        },
        "reprojection_error_px": {"median": 2.0, "p95": 4.0},
        "per_keypoint_residual_px": [1.0] * len(PICKLEBALL_COURT_KEYPOINT_NAMES),
        "metric_confidence": "high",
        "gsd_model": {
            "type": "analytic_ray_plane",
            "plane_sigma_m": 0.012,
            "calibration_sigma_m": 0.02,
            "samples": [{"court_xy": [0.0, 0.0], "gsd_m_per_px": 0.01, "sigma_p_m": 0.02}],
        },
        "capture_quality": {"grade": "good", "reasons": []},
        "image_pts": image_pts,
        "world_pts": world_pts,
        "source": "arkit_plane_keypoint_metric_solve_v1",
        "solved_over_frames": [0, 15, 29],
    }
    payload.update(overrides)
    return payload


def test_ball_court_calibration_gate_passes_trusted_metric_calibration_and_real_video(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    calibration = tmp_path / "court_calibration.json"
    _write_test_video(video)
    calibration.write_text(json.dumps(_base_calibration_payload()), encoding="utf-8")

    report = build_ball_court_calibration_gate_report(calibration_path=calibration, video_path=video)

    assert report["status"] == "TESTED-ON-REAL-DATA"
    assert report["milestone"] == "M3 Court"
    assert report["gate_result"] == "pass"
    assert report["blocked_reason"] is None
    assert report["violations"] == []
    assert report["reprojection_error_px"] == {"median": 2.0, "p95": 4.0}
    assert report["calibration_source"] == "arkit_plane_keypoint_metric_solve_v1"
    assert report["metric_confidence"] == "high"
    assert report["not_ground_truth"] is True


def test_ball_court_calibration_gate_fails_untrusted_prototype_calibration(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    calibration = tmp_path / "court_calibration.json"
    _write_test_video(video)
    prototype_payload = _base_calibration_payload(
        coordinate_frame=None,
        T_world_court=None,
        per_keypoint_residual_px=None,
        metric_confidence=None,
        gsd_model=None,
        source=None,
        solved_over_frames=None,
        intrinsics={
            "fx": 1000.0,
            "fy": 1010.0,
            "cx": 960.0,
            "cy": 540.0,
            "dist": [0.0, 0.0, 0.0, 0.0],
            "source": "estimated_from_review_frame",
        },
        capture_quality={
            "grade": "warn",
            "reasons": ["prototype_human_review_corners", "estimated_intrinsics", "corrected_unverified"],
        },
    )
    calibration.write_text(json.dumps(prototype_payload), encoding="utf-8")

    report = build_ball_court_calibration_gate_report(calibration_path=calibration, video_path=video)

    assert report["status"] == "TESTED-ON-REAL-DATA"
    assert report["gate_result"] == "fail"
    assert report["blocked_reason"] == "court_calibration_gate_failed"
    assert set(report["violations"]) >= {
        "calibration_source_missing",
        "metric_confidence_missing",
        "metric_fields_incomplete",
        "intrinsics_not_trusted",
        "capture_quality_warn",
        "capture_quality_unverified:prototype_human_review_corners",
        "capture_quality_unverified:estimated_intrinsics",
        "capture_quality_unverified:corrected_unverified",
    }


def test_ball_court_calibration_gate_cli_writes_failed_report(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    calibration = tmp_path / "court_calibration.json"
    out = tmp_path / "m3_report.json"
    _write_test_video(video)
    calibration.write_text(
        json.dumps(
            _base_calibration_payload(
                metric_confidence="med",
                capture_quality={"grade": "warn", "reasons": ["manual_fallback_unverified"]},
            )
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/validate_ball_court_calibration.py",
            "--calibration",
            str(calibration),
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
    assert json.loads(completed.stdout)["gate_result"] == "fail"
    assert json.loads(out.read_text(encoding="utf-8"))["metric_confidence"] == "med"

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from threed.racketsport.ball_inout_gate import build_ball_inout_gate_report


def _write_test_video(path: Path, *, size: str = "1920x1080", fps: int = 60) -> None:
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is required to synthesize in/out-gate test video")
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


def _ball_track_payload(*, bounces: list[dict[str, object]] | None = None) -> dict[str, object]:
    return {
        "schema_version": 1,
        "fps": 60.0,
        "source": "fused",
        "frames": [{"t": 1.0, "xy": [120.0, 210.0], "conf": 0.83, "visible": True}],
        "bounces": bounces
        if bounces is not None
        else [
            {
                "t": 1.0,
                "frame": 60,
                "world_xy": [1.2, 2.4],
                "contact_xy_img": [120.0, 210.0],
                "p_bounce": 0.82,
                "margin_m": 0.18,
                "uncertainty_m": 0.06,
                "confidence": 0.75,
                "call": "in",
                "nearest_line": "right_sideline",
                "region": "near",
                "dominant_uncertainty_term": "localization",
            },
            {
                "t": 1.5,
                "frame": 90,
                "world_xy": [1.5, 2.6],
                "contact_xy_img": [140.0, 220.0],
                "p_bounce": 0.78,
                "margin_m": 0.03,
                "uncertainty_m": 0.08,
                "confidence": 0.2727272727,
                "call": "too_close_to_call",
                "nearest_line": "far_baseline",
                "region": "far",
                "dominant_uncertainty_term": "depth",
            },
        ],
    }


def _m4_report_payload(
    *,
    gate_result: str = "pass",
    ball_track_path: str | None = None,
    bounce_count: int | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "racketsport_ball_bounce_gate_report",
        "milestone": "M4 Bounce",
        "status": "TESTED-ON-REAL-DATA",
        "gate_result": gate_result,
        "blocked_reason": None if gate_result == "pass" else "ball_bounce_gate_failed",
    }
    if ball_track_path is not None:
        payload["ball_track_path"] = ball_track_path
    if bounce_count is not None:
        payload["bounce_count"] = bounce_count
    return payload


def _reviewed_inout_payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_reviewed_ball_inout",
        "fps": 60.0,
        "calls": [{"frame": 60, "call": "in"}, {"frame": 90, "call": "in"}],
    }


def test_ball_inout_gate_passes_uncertainty_rule_gray_zone_and_reviewed_agreement(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    track = tmp_path / "ball_track.json"
    m4 = tmp_path / "m4_report.json"
    reviewed = tmp_path / "reviewed_inout.json"
    _write_test_video(video)
    track.write_text(json.dumps(_ball_track_payload()), encoding="utf-8")
    m4.write_text(json.dumps(_m4_report_payload(ball_track_path=str(track), bounce_count=2)), encoding="utf-8")
    reviewed.write_text(json.dumps(_reviewed_inout_payload()), encoding="utf-8")

    report = build_ball_inout_gate_report(
        ball_track_path=track,
        video_path=video,
        m4_bounce_report_path=m4,
        reviewed_inout_path=reviewed,
    )

    assert report["status"] == "TESTED-ON-REAL-DATA"
    assert report["milestone"] == "M5 In/out"
    assert report["gate_result"] == "pass"
    assert report["blocked_reason"] is None
    assert report["confident_call_count"] == 1
    assert report["gray_zone_rate"] == pytest.approx(0.5)
    assert report["review_agreement"]["confident_agreement_rate"] == pytest.approx(1.0)
    assert report["near_far_split"]["near"]["confident"] == 1
    assert report["near_far_split"]["far"]["gray"] == 1
    assert report["violations"] == []


def test_ball_inout_gate_surfaces_uncertainty_breakdown_when_present(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    track = tmp_path / "ball_track.json"
    m4 = tmp_path / "m4_report.json"
    reviewed = tmp_path / "reviewed_inout.json"
    _write_test_video(video)
    payload = _ball_track_payload()
    payload["bounces"][0]["uncertainty_breakdown"] = {
        "method": "camera_geometry_elevation_parallax_v1",
        "sigma_reproj_m": 0.0,
        "sigma_depth_m": 0.15,
        "sigma_ballradius_m": 0.02,
        "sigma_localization_m": 0.01,
        "camera_height_m": 4.2,
        "h_max_m": 0.2,
        "v_z_ref_mps": 6.23,
        "dt_s": 0.033,
        "frames_window": 2.0,
        "binding_axis": "x",
        "pose_source": "manual_corner_focal_length_search_v1",
    }
    track.write_text(json.dumps(payload), encoding="utf-8")
    m4.write_text(json.dumps(_m4_report_payload(ball_track_path=str(track), bounce_count=2)), encoding="utf-8")
    reviewed.write_text(json.dumps(_reviewed_inout_payload()), encoding="utf-8")

    report = build_ball_inout_gate_report(
        ball_track_path=track,
        video_path=video,
        m4_bounce_report_path=m4,
        reviewed_inout_path=reviewed,
    )

    with_breakdown = report["calls"][0]
    without_breakdown = report["calls"][1]
    assert with_breakdown["uncertainty_breakdown"]["method"] == "camera_geometry_elevation_parallax_v1"
    assert with_breakdown["uncertainty_breakdown"]["sigma_depth_m"] == pytest.approx(0.15)
    assert without_breakdown["uncertainty_breakdown"] is None


def test_ball_inout_gate_allows_out_calls_without_in_court_region(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    track = tmp_path / "ball_track.json"
    m4 = tmp_path / "m4_report.json"
    reviewed = tmp_path / "reviewed_inout.json"
    _write_test_video(video)
    track.write_text(
        json.dumps(
            _ball_track_payload(
                bounces=[
                    {
                        "t": 1.0,
                        "frame": 60,
                        "world_xy": [-0.2, 2.4],
                        "contact_xy_img": [120.0, 210.0],
                        "p_bounce": 0.82,
                        "margin_m": -0.18,
                        "uncertainty_m": 0.06,
                        "confidence": 0.75,
                        "call": "out",
                        "nearest_line": "left_sideline",
                        "dominant_uncertainty_term": "manual_corner_homography_projection",
                    }
                ],
            )
        ),
        encoding="utf-8",
    )
    m4.write_text(json.dumps(_m4_report_payload(ball_track_path=str(track), bounce_count=1)), encoding="utf-8")
    reviewed.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_type": "racketsport_reviewed_ball_inout",
                "fps": 60.0,
                "calls": [{"frame": 60, "call": "out"}],
            }
        ),
        encoding="utf-8",
    )

    report = build_ball_inout_gate_report(
        ball_track_path=track,
        video_path=video,
        m4_bounce_report_path=m4,
        reviewed_inout_path=reviewed,
    )

    assert "region_missing" not in report["violations"]
    assert report["gate_result"] == "pass"


def test_ball_inout_gate_allows_gray_calls_without_in_court_region(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    track = tmp_path / "ball_track.json"
    m4 = tmp_path / "m4_report.json"
    reviewed = tmp_path / "reviewed_inout.json"
    _write_test_video(video)
    payload = _ball_track_payload()
    payload["bounces"][1].pop("region")
    track.write_text(json.dumps(payload), encoding="utf-8")
    m4.write_text(json.dumps(_m4_report_payload(ball_track_path=str(track), bounce_count=2)), encoding="utf-8")
    reviewed.write_text(json.dumps(_reviewed_inout_payload()), encoding="utf-8")

    report = build_ball_inout_gate_report(
        ball_track_path=track,
        video_path=video,
        m4_bounce_report_path=m4,
        reviewed_inout_path=reviewed,
    )

    assert "region_missing" not in report["violations"]
    assert report["gate_result"] == "pass"


def test_ball_inout_gate_fails_closed_without_bounces_m4_pass_or_review_labels(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    track = tmp_path / "ball_track.json"
    m4 = tmp_path / "m4_report.json"
    _write_test_video(video)
    track.write_text(json.dumps(_ball_track_payload(bounces=[])), encoding="utf-8")
    m4.write_text(json.dumps(_m4_report_payload(gate_result="fail")), encoding="utf-8")

    report = build_ball_inout_gate_report(
        ball_track_path=track,
        video_path=video,
        m4_bounce_report_path=m4,
        reviewed_inout_path=tmp_path / "missing_reviewed_inout.json",
    )

    assert report["status"] == "TESTED-ON-REAL-DATA"
    assert report["gate_result"] == "fail"
    assert report["blocked_reason"] == "ball_inout_gate_failed"
    assert set(report["violations"]) >= {
        "m4_bounce_gate_not_passed",
        "ball_track_has_no_bounces",
        "missing_reviewed_inout_labels",
        "no_confident_inout_calls",
    }


def test_ball_inout_gate_rejects_unbound_m4_report_and_untyped_review_labels(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    track = tmp_path / "ball_track.json"
    m4 = tmp_path / "m4_report.json"
    reviewed = tmp_path / "reviewed_inout.json"
    _write_test_video(video)
    track.write_text(json.dumps(_ball_track_payload()), encoding="utf-8")
    m4_payload = _m4_report_payload()
    m4_payload["ball_track_path"] = str(tmp_path / "different_ball_track.json")
    m4_payload["bounce_count"] = 99
    m4.write_text(json.dumps(m4_payload), encoding="utf-8")
    reviewed_payload = _reviewed_inout_payload()
    reviewed_payload.pop("artifact_type")
    reviewed.write_text(json.dumps(reviewed_payload), encoding="utf-8")

    report = build_ball_inout_gate_report(
        ball_track_path=track,
        video_path=video,
        m4_bounce_report_path=m4,
        reviewed_inout_path=reviewed,
    )

    assert report["gate_result"] == "fail"
    assert set(report["violations"]) >= {
        "m4_bounce_gate_ball_track_mismatch",
        "m4_bounce_gate_bounce_count_mismatch",
        "reviewed_inout_artifact_type_invalid",
    }


def test_ball_inout_gate_cli_writes_failed_report(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    track = tmp_path / "ball_track.json"
    out = tmp_path / "m5_report.json"
    _write_test_video(video)
    track.write_text(json.dumps(_ball_track_payload(bounces=[])), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/validate_ball_inout.py",
            "--ball-track",
            str(track),
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
    assert json.loads(out.read_text(encoding="utf-8"))["blocked_reason"] == "ball_inout_gate_failed"

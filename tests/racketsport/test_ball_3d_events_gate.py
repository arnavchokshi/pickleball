from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from threed.racketsport.ball_3d_events_gate import build_ball_3d_events_gate_report


def _write_test_video(path: Path, *, size: str = "1920x1080", fps: int = 60) -> None:
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is required to synthesize 3D/events-gate test video")
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


def _ball_track_payload(*, complete_3d: bool = True) -> dict[str, object]:
    frames: list[dict[str, object]] = []
    for frame, t, xy, xyz, speed in [
        (58, 0.966667, [114.0, 205.0], [0.8, 2.0, 1.1], 12.0),
        (60, 1.000000, [120.0, 210.0], [1.2, 2.4, 0.8], 16.0),
        (63, 1.050000, [128.0, 216.0], [1.7, 2.9, 0.0], 14.0),
    ]:
        del frame
        item = {"t": t, "xy": xy, "conf": 0.83, "visible": True}
        if complete_3d:
            item.update({"world_xyz": xyz, "spin_rpm": -820.0, "speed_mps": speed})
        frames.append(item)
    return {
        "schema_version": 1,
        "fps": 60.0,
        "source": "fused",
        "frames": frames,
        "bounces": [
            {
                "t": 1.05,
                "frame": 63,
                "world_xy": [1.7, 2.9],
                "contact_xy_img": [128.0, 216.0],
                "p_bounce": 0.82,
                "margin_m": 0.18,
                "uncertainty_m": 0.06,
                "confidence": 0.75,
                "call": "in",
                "nearest_line": "right_sideline",
                "region": "near",
                "dominant_uncertainty_term": "localization",
            }
        ]
        if complete_3d
        else [],
    }


def _gate_report_payload(
    *,
    artifact_type: str,
    milestone: str,
    gate_result: str = "pass",
    ball_track_path: str | None = None,
    bounce_count: int | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": artifact_type,
        "milestone": milestone,
        "status": "TESTED-ON-REAL-DATA",
        "gate_result": gate_result,
        "blocked_reason": None if gate_result == "pass" else f"{milestone.lower()}_failed",
    }
    if ball_track_path is not None:
        payload["ball_track_path"] = ball_track_path
    if bounce_count is not None:
        payload["bounce_count"] = bounce_count
    return payload


def _physics_segments_payload(*, input_ball_track_path: str | None = None) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "racketsport_ball_physics_segments",
        "model": "gravity_drag_magnus_ode_v1",
        "solver_command": "python scripts/racketsport/reconstruct_ball_3d_events.py --ball-track ...",
        "segments": [
            {
                "start_t": 1.0,
                "end_t": 1.05,
                "uses_drag": True,
                "uses_magnus": True,
                "fit_residual_px": 3.0,
                "boundary_constraints": ["contact", "bounce_z0"],
                "peak_speed_mps": 18.46,
                "avg_speed_mps": 14.75,
                "peak_speed_mph": 41.3,
                "avg_speed_mph": 33.0,
                "spin_sign": "negative",
                "spin_rpm_estimate": -820.0,
            }
        ],
    }
    if input_ball_track_path is not None:
        payload["input_ball_track_path"] = input_ball_track_path
    return payload


def _contact_windows_payload() -> dict[str, object]:
    base = {
        "frame": 60,
        "confidence": 0.86,
        "sources": {"audio": 0.95, "wrist_vel": 0.80, "ball_inflection": 0.76},
        "window": {"t0": 0.965, "t1": 1.035, "importance": 0.91},
    }
    return {
        "schema_version": 1,
        "events": [
            {"type": "contact", "t": 1.0, "player_id": 0, **base},
            {"type": "net_cross", "t": 1.02, "player_id": None, **{**base, "frame": 61}},
            {"type": "into_net", "t": 1.03, "player_id": None, **{**base, "frame": 62}},
        ],
    }


def _reviewed_contacts_payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_reviewed_ball_contacts",
        "fps": 60.0,
        "contacts": [{"frame": 60, "t": 1.0}],
    }


def test_ball_3d_events_gate_passes_complete_physics_contact_and_net_artifacts(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    track = tmp_path / "ball_track.json"
    m4 = tmp_path / "m4_report.json"
    m5 = tmp_path / "m5_report.json"
    physics = tmp_path / "physics_segments.json"
    contact_windows = tmp_path / "contact_windows.json"
    reviewed_contacts = tmp_path / "reviewed_contacts.json"
    _write_test_video(video)
    track.write_text(json.dumps(_ball_track_payload()), encoding="utf-8")
    m4.write_text(
        json.dumps(
            _gate_report_payload(
                artifact_type="racketsport_ball_bounce_gate_report",
                milestone="M4 Bounce",
                ball_track_path=str(track),
                bounce_count=1,
            )
        ),
        encoding="utf-8",
    )
    m5.write_text(
        json.dumps(
            _gate_report_payload(
                artifact_type="racketsport_ball_inout_gate_report",
                milestone="M5 In/out",
                ball_track_path=str(track),
                bounce_count=1,
            )
        ),
        encoding="utf-8",
    )
    physics.write_text(json.dumps(_physics_segments_payload(input_ball_track_path=str(track))), encoding="utf-8")
    contact_windows.write_text(json.dumps(_contact_windows_payload()), encoding="utf-8")
    reviewed_contacts.write_text(json.dumps(_reviewed_contacts_payload()), encoding="utf-8")

    report = build_ball_3d_events_gate_report(
        ball_track_path=track,
        video_path=video,
        m4_bounce_report_path=m4,
        m5_inout_report_path=m5,
        physics_segments_path=physics,
        contact_windows_path=contact_windows,
        reviewed_contacts_path=reviewed_contacts,
    )

    assert report["status"] == "TESTED-ON-REAL-DATA"
    assert report["milestone"] == "M6 3D/spin/events"
    assert report["gate_result"] == "pass"
    assert report["blocked_reason"] is None
    assert report["trajectory_3d"]["world_xyz_frame_count"] == 3
    assert report["spin_speed"]["frame_speed_count"] == 3
    assert report["physics_segments"]["segment_count"] == 1
    assert report["contact_timing"]["max_abs_delta_frames"] == pytest.approx(0.0)
    assert report["contact_timing"]["max_abs_audio_delta_ms"] == pytest.approx(0.0)
    assert report["events"]["net_cross_count"] == 1
    assert report["events"]["into_net_count"] == 1
    assert report["violations"] == []
    assert report["not_ground_truth"] is True


def test_ball_3d_events_gate_fails_closed_without_upstream_3d_or_event_artifacts(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    track = tmp_path / "ball_track.json"
    m4 = tmp_path / "m4_report.json"
    m5 = tmp_path / "m5_report.json"
    _write_test_video(video)
    track.write_text(json.dumps(_ball_track_payload(complete_3d=False)), encoding="utf-8")
    m4.write_text(
        json.dumps(
            _gate_report_payload(
                artifact_type="racketsport_ball_bounce_gate_report",
                milestone="M4 Bounce",
                gate_result="fail",
            )
        ),
        encoding="utf-8",
    )
    m5.write_text(
        json.dumps(
            _gate_report_payload(
                artifact_type="racketsport_ball_inout_gate_report",
                milestone="M5 In/out",
                gate_result="fail",
            )
        ),
        encoding="utf-8",
    )

    report = build_ball_3d_events_gate_report(
        ball_track_path=track,
        video_path=video,
        m4_bounce_report_path=m4,
        m5_inout_report_path=m5,
        physics_segments_path=tmp_path / "missing_physics_segments.json",
        contact_windows_path=tmp_path / "missing_contact_windows.json",
        reviewed_contacts_path=tmp_path / "missing_reviewed_contacts.json",
    )

    assert report["status"] == "TESTED-ON-REAL-DATA"
    assert report["gate_result"] == "fail"
    assert report["blocked_reason"] == "ball_3d_events_gate_failed"
    assert set(report["violations"]) >= {
        "m4_bounce_gate_not_passed",
        "m5_inout_gate_not_passed",
        "ball_track_has_no_bounces",
        "no_world_xyz_frames",
        "no_spin_estimates",
        "no_speed_estimates",
        "missing_physics_segments",
        "missing_contact_windows",
        "missing_reviewed_contact_labels",
        "no_contact_events",
        "no_net_cross_events",
    }


def test_ball_3d_events_gate_rejects_unbound_upstream_physics_and_review_artifacts(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    track = tmp_path / "ball_track.json"
    m4 = tmp_path / "m4_report.json"
    m5 = tmp_path / "m5_report.json"
    physics = tmp_path / "physics_segments.json"
    contact_windows = tmp_path / "contact_windows.json"
    reviewed_contacts = tmp_path / "reviewed_contacts.json"
    _write_test_video(video)
    track.write_text(json.dumps(_ball_track_payload()), encoding="utf-8")
    m4.write_text(
        json.dumps(
            _gate_report_payload(
                artifact_type="racketsport_ball_bounce_gate_report",
                milestone="M4 Bounce",
                ball_track_path=str(tmp_path / "different_ball_track.json"),
                bounce_count=99,
            )
        ),
        encoding="utf-8",
    )
    m5.write_text(
        json.dumps(
            _gate_report_payload(
                artifact_type="racketsport_ball_inout_gate_report",
                milestone="M5 In/out",
                ball_track_path=str(tmp_path / "different_ball_track.json"),
                bounce_count=99,
            )
        ),
        encoding="utf-8",
    )
    physics.write_text(json.dumps(_physics_segments_payload(input_ball_track_path=str(tmp_path / "different_ball_track.json"))), encoding="utf-8")
    contact_windows.write_text(json.dumps(_contact_windows_payload()), encoding="utf-8")
    reviewed_payload = _reviewed_contacts_payload()
    reviewed_payload.pop("artifact_type")
    reviewed_contacts.write_text(json.dumps(reviewed_payload), encoding="utf-8")

    report = build_ball_3d_events_gate_report(
        ball_track_path=track,
        video_path=video,
        m4_bounce_report_path=m4,
        m5_inout_report_path=m5,
        physics_segments_path=physics,
        contact_windows_path=contact_windows,
        reviewed_contacts_path=reviewed_contacts,
    )

    assert report["gate_result"] == "fail"
    assert set(report["violations"]) >= {
        "m4_bounce_gate_ball_track_mismatch",
        "m4_bounce_gate_bounce_count_mismatch",
        "m5_inout_gate_ball_track_mismatch",
        "m5_inout_gate_bounce_count_mismatch",
        "physics_segments_input_track_mismatch",
        "reviewed_contacts_artifact_type_invalid",
    }


def test_ball_3d_events_gate_cli_writes_failed_report(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    track = tmp_path / "ball_track.json"
    out = tmp_path / "m6_report.json"
    _write_test_video(video)
    track.write_text(json.dumps(_ball_track_payload(complete_3d=False)), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/validate_ball_3d_events.py",
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
    assert json.loads(out.read_text(encoding="utf-8"))["blocked_reason"] == "ball_3d_events_gate_failed"

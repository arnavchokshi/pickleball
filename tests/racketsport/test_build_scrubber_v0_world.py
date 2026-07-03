from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tests.racketsport.calibration_fixtures import minimal_calibration_image_pts, minimal_calibration_world_pts
from threed.racketsport.schemas import VirtualWorld, validate_artifact_file


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _court_calibration() -> dict:
    return {
        "schema_version": 1,
        "sport": "pickleball",
        "homography": [[1.0, 0.0, 960.0], [0.0, 1.0, 540.0], [0.0, 0.0, 1.0]],
        "intrinsics": {"fx": 1000.0, "fy": 1000.0, "cx": 960.0, "cy": 540.0, "dist": [], "source": "estimated_from_review_frame"},
        "extrinsics": {
            "R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "t": [0.0, 0.0, 12.0],
            "camera_height_m": 12.0,
        },
        "reprojection_error_px": {"median": 1.2, "p95": 3.4},
        "capture_quality": {"grade": "warn", "reasons": ["prototype_human_review_corners", "corrected_unverified"]},
        "image_pts": minimal_calibration_image_pts(),
        "world_pts": minimal_calibration_world_pts(),
    }


def _tracks() -> dict:
    return {
        "schema_version": 1,
        "fps": 60.0,
        "players": [
            {
                "id": 3,
                "side": "far",
                "role": "left",
                "frames": [{"t": 0.0, "bbox": [10.0, 10.0, 40.0, 240.0], "world_xy": [-1.0, 4.0], "conf": 0.9}],
            },
            {
                "id": 7,
                "side": "near",
                "role": "left",
                "frames": [{"t": 0.0, "bbox": [100.0, 100.0, 140.0, 240.0], "world_xy": [0.25, -2.0], "conf": 0.91}],
            },
        ],
        "rally_spans": [],
    }


def _body_world_label_packet() -> dict:
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_body_world_label_packet",
        "clip": "burlington_test",
        "not_ground_truth": True,
        "trusted_for_world_mpjpe": False,
        "joint_names": ["nose"],
        "samples": [
            {
                "sample_id": "frame_000000_player_3",
                "frame_index": 0,
                "t": 0.0,
                "player_id": 3,
                "track_world_xy": [-1.0, 4.0],
                "predicted_joints_world": [[-1.0, 4.0, 1.6]],
                "joint_conf": [0.7],
                "joint_count": 1,
                "review_required": True,
            }
        ],
    }


def _ball_track() -> dict:
    return {
        "schema_version": 1,
        "fps": 60.0,
        "source": "tracknet",
        "frames": [{"t": 0.0, "xy": [320.0, 240.0], "conf": 0.83, "visible": True, "world_xyz": [0.1, -0.8, 0.9]}],
        "bounces": [],
    }


def _body_gate_report_clip() -> dict:
    return {
        "clip": "burlington_test_run",
        "status": "blocked",
        "full_clip_body_gate": {"passed": True},
        "body_grounding_quality": {"status": "pass"},
        "body_review_overlay_alignment": {
            "status": "pass",
            "rendered_count": 20,
            "sample_count": 20,
            "resolved_warning_sample_count": 3,
            "unresolved_warning_sample_count": 0,
        },
        "world_mpjpe": {"blockers": ["missing_world_mpjpe_gate"]},
    }


def _run_cli(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "scripts/racketsport/build_scrubber_v0_world.py", *args],
        check=False,
        capture_output=True,
        text=True,
    )


def test_build_scrubber_v0_world_wires_trust_bands_from_real_artifacts(tmp_path: Path) -> None:
    run_dir = tmp_path / "clip_a"
    court = _write_json(run_dir / "court_calibration.json", _court_calibration())
    tracks = _write_json(run_dir / "tracks.json", _tracks())
    packet = _write_json(run_dir / "body_world_label_packet.json", _body_world_label_packet())
    ball = _write_json(run_dir / "ball_track.json", _ball_track())
    gate_report = _write_json(run_dir / "body_gate_report.json", {"clips": [_body_gate_report_clip()]})
    out = run_dir / "virtual_world.json"
    trust_band_report_out = run_dir / "trust_bands.json"

    completed = _run_cli(
        [
            "--clip",
            "clip_a",
            "--court-calibration",
            str(court),
            "--tracks",
            str(tracks),
            "--body-world-label-packet",
            str(packet),
            "--ball-track",
            str(ball),
            "--body-gate-report",
            str(gate_report),
            "--track-idf1",
            "0.8904",
            "--track-evidence",
            "runs/phase2/trk_offline_authority_20260701T205912Z/",
            "--out",
            str(out),
            "--trust-band-report-out",
            str(trust_band_report_out),
        ]
    )

    assert completed.returncode == 0, completed.stderr
    parsed = validate_artifact_file("virtual_world", out)
    assert isinstance(parsed, VirtualWorld)
    assert parsed.court.trust_band is not None
    assert parsed.court.trust_band.badge == "preview"

    players_by_id = {player.id: player for player in parsed.players}
    assert players_by_id[3].representation == "joints"
    assert players_by_id[3].trust_band is not None
    assert players_by_id[3].trust_band.badge == "preview"
    assert players_by_id[3].trust_band.stage == "BODY"
    assert players_by_id[7].representation == "track_only"
    assert players_by_id[7].trust_band is not None
    assert players_by_id[7].trust_band.badge == "low_confidence"
    assert players_by_id[7].trust_band.stage == "TRK"

    assert parsed.ball.trust_band is not None
    assert parsed.ball.trust_band.badge == "low_confidence"

    stdout_payload = json.loads(completed.stdout)
    assert stdout_payload["trust_bands"]["body"]["badge"] == "preview"
    assert stdout_payload["trust_bands"]["track"]["badge"] == "low_confidence"

    trust_band_report = json.loads(trust_band_report_out.read_text(encoding="utf-8"))
    assert trust_band_report["trust_bands"]["court"]["badge"] == "preview"


def test_build_scrubber_v0_world_requires_disambiguation_for_multi_clip_gate_report(tmp_path: Path) -> None:
    run_dir = tmp_path / "clip_a"
    court = _write_json(run_dir / "court_calibration.json", _court_calibration())
    tracks = _write_json(run_dir / "tracks.json", _tracks())
    gate_report = _write_json(
        run_dir / "body_gate_report.json",
        {"clips": [_body_gate_report_clip(), {**_body_gate_report_clip(), "clip": "other_run"}]},
    )
    out = run_dir / "virtual_world.json"

    completed = _run_cli(
        [
            "--clip",
            "clip_a",
            "--court-calibration",
            str(court),
            "--tracks",
            str(tracks),
            "--body-gate-report",
            str(gate_report),
            "--out",
            str(out),
        ]
    )

    assert completed.returncode == 1
    assert "multiple clips" in completed.stderr

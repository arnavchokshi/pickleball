from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from threed.racketsport.ball_world_speed_gate import filter_ball_track_world_speed
from threed.racketsport.court_calibration import homography_from_planar_points, project_planar_points
from threed.racketsport.court_templates import get_court_template
from threed.racketsport.schemas import BallTrack, CourtCalibration, validate_artifact_file


def _calibration_for_image_court(width: float = 420.0, height: float = 880.0) -> CourtCalibration:
    template = get_court_template("pickleball")
    image_corners = [[0.0, height], [width, height], [width, 0.0], [0.0, 0.0]]
    homography = homography_from_planar_points(template.corners_m, image_corners)
    return CourtCalibration.model_validate(
        {
            "schema_version": 1,
            "sport": "pickleball",
            "intrinsics": {
                "fx": width * 1.2,
                "fy": width * 1.2,
                "cx": width / 2.0,
                "cy": height / 2.0,
                "dist": [],
                "source": "test",
            },
            "extrinsics": {
                "R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                "t": [0.0, 0.0, 10.0],
                "camera_height_m": 10.0,
            },
            "homography": homography,
            "image_pts": image_corners,
            "world_pts": template.corners_m,
            "reprojection_error_px": {"median": 0.0, "p95": 0.0},
            "capture_quality": {"grade": "warn", "reasons": ["test"]},
        }
    )


def _write_track_from_world_points(path: Path, calibration: CourtCalibration, world_xy: list[list[float]]) -> None:
    image_points = project_planar_points(calibration.homography, world_xy)
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "fps": 30.0,
                "source": "fused",
                "frames": [
                    {"t": index / 30.0, "xy": xy, "conf": 0.9, "visible": True}
                    for index, xy in enumerate(image_points)
                ],
                "bounces": [],
            }
        ),
        encoding="utf-8",
    )


def test_world_speed_gate_rejects_links_over_30_mps_and_preserves_schema(tmp_path: Path) -> None:
    calibration = _calibration_for_image_court()
    track_path = tmp_path / "ball_track.json"
    _write_track_from_world_points(
        track_path,
        calibration,
        [
            [0.0, 0.0],
            [0.5, 0.0],
            [4.0, 0.0],
            [1.5, 0.0],
        ],
    )

    payload, summary = filter_ball_track_world_speed(
        ball_track_path=track_path,
        calibration=calibration,
        target_size=(420, 880),
        max_world_speed_mps=30.0,
        base_jump_px=60.0,
    )

    filtered = BallTrack.model_validate(payload)
    assert filtered.frames[0].visible is True
    assert filtered.frames[1].visible is True
    assert filtered.frames[2].visible is False
    assert filtered.frames[2].conf == pytest.approx(0.0)
    assert filtered.frames[3].visible is True
    assert summary["artifact_type"] == "racketsport_ball_world_speed_gate"
    assert summary["status"] == "TESTED-ON-REAL-DATA"
    assert summary["coordinate_model"] == "court_plane_xy"
    assert summary["max_world_speed_mps"] == pytest.approx(30.0)
    assert summary["rejected_world_speed_count"] == 1
    assert summary["max_observed_world_speed_mps"] == pytest.approx(105.0)
    assert summary["uses_human_clicks"] is False
    assert summary["not_ground_truth"] is True


def test_world_speed_gate_cli_writes_track_and_summary(tmp_path: Path) -> None:
    calibration = _calibration_for_image_court()
    calibration_path = tmp_path / "court_calibration.json"
    track_path = tmp_path / "ball_track.json"
    out_path = tmp_path / "ball_speed_filtered.json"
    summary_path = tmp_path / "speed_summary.json"
    calibration_path.write_text(json.dumps(calibration.model_dump(mode="json")), encoding="utf-8")
    _write_track_from_world_points(track_path, calibration, [[0.0, 0.0], [0.5, 0.0], [4.0, 0.0]])

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/filter_ball_world_speed.py",
            "--ball-track",
            str(track_path),
            "--calibration",
            str(calibration_path),
            "--target-size",
            "420",
            "880",
            "--max-world-speed-mps",
            "30",
            "--base-jump-px",
            "60",
            "--out",
            str(out_path),
            "--summary-out",
            str(summary_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert json.loads(completed.stdout)["rejected_world_speed_count"] == 1
    assert isinstance(validate_artifact_file("ball_track", out_path), BallTrack)
    written = json.loads(summary_path.read_text(encoding="utf-8"))
    assert written["artifact_type"] == "racketsport_ball_world_speed_gate"
    assert written["status"] == "TESTED-ON-REAL-DATA"

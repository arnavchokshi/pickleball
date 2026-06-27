from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from threed.racketsport.schemas import (
    CameraIntrinsics,
    CaptureQuality,
    CourtCalibration,
    CourtExtrinsics,
    ReprojectionError,
    Tracks,
    validate_artifact_file,
)


def _write_calibration(path: Path) -> None:
    calibration = CourtCalibration(
        schema_version=1,
        sport="pickleball",
        homography=[[100.0, 0.0, 1000.0], [0.0, 100.0, 1000.0], [0.0, 0.0, 1.0]],
        intrinsics=CameraIntrinsics(fx=1000.0, fy=1000.0, cx=960.0, cy=540.0, dist=[], source="manual"),
        extrinsics=CourtExtrinsics(
            R=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            t=[0.0, 0.0, 15.0],
            camera_height_m=15.0,
        ),
        reprojection_error_px=ReprojectionError(median=0.0, p95=0.0),
        capture_quality=CaptureQuality(grade="good", reasons=[]),
        image_pts=[],
        world_pts=[],
    )
    path.write_text(calibration.model_dump_json(), encoding="utf-8")


def _write_detections(path: Path, frames: list[dict]) -> None:
    path.write_text(json.dumps({"fps": 30.0, "frames": frames}), encoding="utf-8")


def _run_track_cli(tmp_path: Path, detections: Path, calibration: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/track.py",
            "--detections",
            str(detections),
            "--calibration",
            str(calibration),
            "--out",
            str(tmp_path / "tracks.json"),
        ],
        check=False,
        text=True,
        capture_output=True,
    )


def test_track_cli_writes_schema_valid_tracks_from_precomputed_detections(tmp_path):
    calibration = tmp_path / "court_calibration.json"
    detections = tmp_path / "detections.json"
    _write_calibration(calibration)
    _write_detections(
        detections,
        [
            {"frame": 0, "detections": [{"bbox": [1080.0, 900.0, 1120.0, 1000.0], "conf": 0.91, "class": "person", "player_id": 7}]},
            {"frame": 1, "detections": [{"bbox": [1084.0, 900.0, 1124.0, 1000.0], "conf": 0.89, "class": 0, "player_id": 7}]},
        ],
    )

    result = _run_track_cli(tmp_path, detections, calibration)

    assert result.returncode == 0, result.stderr
    parsed = validate_artifact_file("tracks", tmp_path / "tracks.json")
    assert isinstance(parsed, Tracks)
    assert parsed.fps == 30.0
    assert [player.id for player in parsed.players] == [7]
    assert [frame.t for frame in parsed.players[0].frames] == [0.0, 1.0 / 30.0]
    assert parsed.players[0].frames[0].world_xy == [1.0, 0.0]


def test_track_cli_filters_outside_court_detections(tmp_path):
    calibration = tmp_path / "court_calibration.json"
    detections = tmp_path / "detections.json"
    _write_calibration(calibration)
    _write_detections(
        detections,
        [
            {
                "frame": 0,
                "detections": [
                    {"bbox": [1080.0, 900.0, 1120.0, 1000.0], "conf": 0.91, "class": "person", "track_id": 1},
                    {"bbox": [2980.0, 900.0, 3020.0, 1000.0], "conf": 0.99, "class": "person", "track_id": 2},
                ],
            }
        ],
    )

    result = _run_track_cli(tmp_path, detections, calibration)

    assert result.returncode == 0, result.stderr
    payload = json.loads((tmp_path / "tracks.json").read_text(encoding="utf-8"))
    assert [player["id"] for player in payload["players"]] == [1]
    assert "outside_court=1" in result.stderr


def test_track_cli_fails_cleanly_when_inputs_are_missing(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/track.py",
            "--detections",
            str(tmp_path / "missing_detections.json"),
            "--calibration",
            str(tmp_path / "missing_calibration.json"),
            "--out",
            str(tmp_path / "tracks.json"),
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "missing detections file" in result.stderr
    assert not (tmp_path / "tracks.json").exists()

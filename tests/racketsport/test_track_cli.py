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
from scripts.racketsport.track import build_tracks


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


def _run_track_cli(
    tmp_path: Path,
    detections: Path,
    calibration: Path,
    *,
    extra_args: list[str] | None = None,
) -> subprocess.CompletedProcess[str]:
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
            *(extra_args or []),
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


def test_track_cli_accepts_configurable_court_runoff_margin(tmp_path):
    calibration = tmp_path / "court_calibration.json"
    detections = tmp_path / "detections.json"
    _write_calibration(calibration)
    _write_detections(
        detections,
        [
            {
                "frame": 0,
                "detections": [
                    {"bbox": [1320.0, 900.0, 1360.0, 1000.0], "conf": 0.91, "class": "person", "track_id": 1},
                ],
            }
        ],
    )

    strict = _run_track_cli(tmp_path, detections, calibration)
    with_margin = _run_track_cli(tmp_path, detections, calibration, extra_args=["--court-margin-m", "0.5"])

    assert strict.returncode == 0, strict.stderr
    assert "outside_court=1" in strict.stderr
    assert with_margin.returncode == 0, with_margin.stderr
    payload = json.loads((tmp_path / "tracks.json").read_text(encoding="utf-8"))
    assert [player["id"] for player in payload["players"]] == [1]
    assert "court_margin_m=0.5" in with_margin.stderr


def test_track_cli_caps_doubles_to_four_stable_on_court_players_and_labels_roles(tmp_path):
    calibration = tmp_path / "court_calibration.json"
    detections = tmp_path / "detections.json"
    _write_calibration(calibration)

    # Bottom-center pixels map through the test homography to these court points:
    # 1 near-left, 2 near-right, 3 far-left, 4 far-right, 5 one-frame extra.
    _write_detections(
        detections,
        [
            {
                "frame": 0,
                "detections": [
                    {"bbox": [780.0, 520.0, 820.0, 600.0], "conf": 0.91, "class": "person", "track_id": 1},
                    {"bbox": [1180.0, 520.0, 1220.0, 600.0], "conf": 0.90, "class": "person", "track_id": 2},
                    {"bbox": [780.0, 1120.0, 820.0, 1400.0], "conf": 0.89, "class": "person", "track_id": 3},
                    {"bbox": [1180.0, 1120.0, 1220.0, 1400.0], "conf": 0.88, "class": "person", "track_id": 4},
                    {"bbox": [980.0, 720.0, 1020.0, 1000.0], "conf": 0.99, "class": "person", "track_id": 5},
                ],
            },
            {
                "frame": 1,
                "detections": [
                    {"bbox": [782.0, 520.0, 822.0, 600.0], "conf": 0.91, "class": "person", "track_id": 1},
                    {"bbox": [1182.0, 520.0, 1222.0, 600.0], "conf": 0.90, "class": "person", "track_id": 2},
                    {"bbox": [782.0, 1120.0, 822.0, 1400.0], "conf": 0.89, "class": "person", "track_id": 3},
                    {"bbox": [1182.0, 1120.0, 1222.0, 1400.0], "conf": 0.88, "class": "person", "track_id": 4},
                ],
            },
        ],
    )

    result = _run_track_cli(tmp_path, detections, calibration, extra_args=["--max-players", "4"])

    assert result.returncode == 0, result.stderr
    payload = json.loads((tmp_path / "tracks.json").read_text(encoding="utf-8"))
    assert [player["id"] for player in payload["players"]] == [1, 2, 3, 4]
    assert [(player["side"], player["role"]) for player in payload["players"]] == [
        ("near", "left"),
        ("near", "right"),
        ("far", "left"),
        ("far", "right"),
    ]
    assert "extra_players_dropped=1" in result.stderr


def test_build_tracks_role_lock_keeps_logical_player_ids_when_raw_ids_fragment(tmp_path):
    calibration_path = tmp_path / "court_calibration.json"
    _write_calibration(calibration_path)
    calibration = validate_artifact_file("court_calibration", calibration_path)
    detections = {
        "fps": 30.0,
        "frames": [
            {
                "frame": 0,
                "detections": [
                    {"bbox": [780.0, 520.0, 820.0, 600.0], "conf": 0.91, "class": "person", "track_id": 10},
                    {"bbox": [1180.0, 520.0, 1220.0, 600.0], "conf": 0.90, "class": "person", "track_id": 20},
                    {"bbox": [780.0, 1320.0, 820.0, 1400.0], "conf": 0.89, "class": "person", "track_id": 30},
                    {"bbox": [1180.0, 1320.0, 1220.0, 1400.0], "conf": 0.88, "class": "person", "track_id": 40},
                ],
            },
            {
                "frame": 1,
                "detections": [
                    {"bbox": [782.0, 520.0, 822.0, 600.0], "conf": 0.91, "class": "person", "track_id": 101},
                    {"bbox": [1182.0, 520.0, 1222.0, 600.0], "conf": 0.90, "class": "person", "track_id": 102},
                    {"bbox": [782.0, 1320.0, 822.0, 1400.0], "conf": 0.89, "class": "person", "track_id": 103},
                    {"bbox": [1182.0, 1320.0, 1222.0, 1400.0], "conf": 0.88, "class": "person", "track_id": 104},
                ],
            },
        ],
    }

    tracks, counts = build_tracks(
        detections,
        calibration,
        max_step_m=2.0,
        max_players=4,
        id_strategy="role_lock",
    )

    assert [player.id for player in tracks.players] == [1, 2, 3, 4]
    assert [(player.side, player.role) for player in tracks.players] == [
        ("near", "left"),
        ("near", "right"),
        ("far", "left"),
        ("far", "right"),
    ]
    assert [len(player.frames) for player in tracks.players] == [2, 2, 2, 2]
    assert counts["id_strategy"] == "role_lock"


def test_build_tracks_role_lock_prefers_high_confidence_players_over_low_confidence_role_anchor(tmp_path):
    calibration_path = tmp_path / "court_calibration.json"
    _write_calibration(calibration_path)
    calibration = validate_artifact_file("court_calibration", calibration_path)
    detections = {
        "fps": 30.0,
        "frames": [
            {
                "frame": 0,
                "detections": [
                    {"bbox": [780.0, 520.0, 820.0, 600.0], "conf": 0.91, "class": "person", "track_id": 1},
                    {"bbox": [1180.0, 520.0, 1220.0, 600.0], "conf": 0.90, "class": "person", "track_id": 2},
                    {"bbox": [780.0, 1320.0, 820.0, 1400.0], "conf": 0.89, "class": "person", "track_id": 3},
                    {"bbox": [1480.0, 1320.0, 1520.0, 1400.0], "conf": 0.88, "class": "person", "track_id": 4},
                    {"bbox": [1180.0, 1320.0, 1220.0, 1400.0], "conf": 0.17, "class": "person", "track_id": 99},
                ],
            }
        ],
    }

    tracks, _counts = build_tracks(
        detections,
        calibration,
        max_step_m=2.0,
        max_players=4,
        court_margin_m=4.0,
        id_strategy="role_lock",
    )

    assert len(tracks.players) == 4
    assert min(player.frames[0].conf for player in tracks.players) == 0.88


def test_build_tracks_auto_role_locks_player_label_payloads_without_tracker_ids(tmp_path):
    calibration_path = tmp_path / "court_calibration.json"
    _write_calibration(calibration_path)
    calibration = validate_artifact_file("court_calibration", calibration_path)
    detections = {
        "schema_version": 1,
        "artifact_type": "racketsport_person_detections",
        "source": "player_labels",
        "fps": 30.0,
        "frames": [
            {
                "frame": 0,
                "detections": [
                    {"bbox": [780.0, 520.0, 820.0, 600.0], "conf": 0.91, "class": "person", "source_id": "p1"},
                    {"bbox": [1180.0, 520.0, 1220.0, 600.0], "conf": 0.90, "class": "person", "source_id": "p2"},
                ],
            },
            {
                "frame": 1,
                "detections": [
                    {"bbox": [1182.0, 520.0, 1222.0, 600.0], "conf": 0.91, "class": "person", "source_id": "p1"},
                    {"bbox": [782.0, 520.0, 822.0, 600.0], "conf": 0.90, "class": "person", "source_id": "p2"},
                ],
            },
        ],
    }

    tracks, counts = build_tracks(
        detections,
        calibration,
        max_step_m=2.0,
        max_players=4,
        id_strategy="auto",
    )

    assert counts["requested_id_strategy"] == "auto"
    assert counts["id_strategy"] == "role_lock"
    assert [player.id for player in tracks.players] == [1, 2]
    assert [(player.side, player.role) for player in tracks.players] == [
        ("near", "left"),
        ("near", "right"),
    ]
    assert [len(player.frames) for player in tracks.players] == [2, 2]


def test_track_cli_caps_singles_to_two_players(tmp_path):
    calibration = tmp_path / "court_calibration.json"
    detections = tmp_path / "detections.json"
    _write_calibration(calibration)
    _write_detections(
        detections,
        [
            {
                "frame": 0,
                "detections": [
                    {"bbox": [780.0, 520.0, 820.0, 600.0], "conf": 0.91, "class": "person", "track_id": 1},
                    {"bbox": [1180.0, 1120.0, 1220.0, 1400.0], "conf": 0.90, "class": "person", "track_id": 2},
                    {"bbox": [980.0, 720.0, 1020.0, 1000.0], "conf": 0.99, "class": "person", "track_id": 3},
                ],
            },
            {
                "frame": 1,
                "detections": [
                    {"bbox": [782.0, 520.0, 822.0, 600.0], "conf": 0.91, "class": "person", "track_id": 1},
                    {"bbox": [1182.0, 1120.0, 1222.0, 1400.0], "conf": 0.90, "class": "person", "track_id": 2},
                ],
            },
        ],
    )

    result = _run_track_cli(tmp_path, detections, calibration, extra_args=["--max-players", "2"])

    assert result.returncode == 0, result.stderr
    payload = json.loads((tmp_path / "tracks.json").read_text(encoding="utf-8"))
    assert [player["id"] for player in payload["players"]] == [1, 2]
    assert [player["role"] for player in payload["players"]] == ["singles", "singles"]
    assert "extra_players_dropped=1" in result.stderr


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

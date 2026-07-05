from __future__ import annotations

import json
import importlib
import importlib.util
import math
import subprocess
import sys
from pathlib import Path


def _virtual_world(ball_frames: list[dict]) -> dict:
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_virtual_world",
        "world_frame": "court_Z0",
        "ball": {"frames": ball_frames},
    }


def _ball_track(ball_frames: list[dict]) -> dict:
    return {
        "schema_version": 1,
        "fps": 30.0,
        "source": "tracknet",
        "frames": ball_frames,
        "bounces": [],
    }


def _builder():
    assert importlib.util.find_spec("threed.racketsport.ball_inflections") is not None
    module = importlib.import_module("threed.racketsport.ball_inflections")
    return module.build_ball_inflections_from_virtual_world


def test_ball_inflection_builder_detects_court_plane_turn() -> None:
    build_ball_inflections_from_virtual_world = _builder()

    payload = build_ball_inflections_from_virtual_world(
        _virtual_world(
            [
                {"t": 0.0, "visible": True, "world_xyz": [0.0, 0.0, 0.0], "conf": 0.9, "approx": True},
                {"t": 0.1, "visible": True, "world_xyz": [1.0, 0.0, 0.0], "conf": 0.8, "approx": True},
                {"t": 0.2, "visible": True, "world_xyz": [1.0, 1.0, 0.0], "conf": 0.7, "approx": True},
            ]
        ),
        min_turn_degrees=35.0,
        min_speed_mps=1.0,
    )

    assert payload["artifact_type"] == "racketsport_ball_inflections"
    assert payload["not_gate_verified"] is True
    assert payload["summary"]["candidate_count"] == 1
    candidate = payload["candidates"][0]
    assert candidate["time_s"] == 0.1
    assert candidate["ball_world_xyz"] == [1.0, 0.0, 0.0]
    assert candidate["turn_angle_deg"] == 90.0
    assert candidate["approx"] is True
    assert 0.0 < candidate["confidence"] <= 1.0


def test_ball_inflection_builder_ignores_straight_or_missing_world_points() -> None:
    build_ball_inflections_from_virtual_world = _builder()

    payload = build_ball_inflections_from_virtual_world(
        _virtual_world(
            [
                {"t": 0.0, "visible": True, "world_xyz": [0.0, 0.0, 0.0], "conf": 0.9, "approx": True},
                {"t": 0.1, "visible": False, "world_xyz": [1.0, 0.0, 0.0], "conf": 0.8, "approx": True},
                {"t": 0.2, "visible": True, "world_xyz": [2.0, 0.0, 0.0], "conf": 0.7, "approx": True},
                {"t": 0.3, "visible": True, "world_xyz": None, "conf": 0.7, "approx": True},
                {"t": 0.4, "visible": True, "world_xyz": [4.0, 0.0, 0.0], "conf": 0.7, "approx": True},
            ]
        ),
        min_turn_degrees=35.0,
        min_speed_mps=1.0,
    )

    assert payload["summary"]["candidate_count"] == 0
    assert payload["summary"]["usable_frame_count"] == 3
    assert payload["candidates"] == []


def test_ball_inflection_builder_suppresses_nearby_duplicate_turns() -> None:
    build_ball_inflections_from_virtual_world = _builder()

    payload = build_ball_inflections_from_virtual_world(
        _virtual_world(
            [
                {"t": 0.0, "visible": True, "world_xyz": [0.0, 0.0, 0.0], "conf": 0.9, "approx": True},
                {"t": 0.1, "visible": True, "world_xyz": [1.0, 0.0, 0.0], "conf": 0.9, "approx": True},
                {"t": 0.2, "visible": True, "world_xyz": [1.0, 1.0, 0.0], "conf": 0.8, "approx": True},
                {"t": 0.3, "visible": True, "world_xyz": [2.0, 1.0, 0.0], "conf": 0.7, "approx": True},
            ]
        ),
        min_turn_degrees=35.0,
        min_speed_mps=1.0,
        min_candidate_separation_s=0.15,
    )

    assert payload["summary"]["candidate_count"] == 1
    assert payload["summary"]["raw_candidate_count"] == 2
    assert payload["candidates"][0]["time_s"] == 0.1


def test_ball_inflection_builder_detects_image_track_turn_without_world_points() -> None:
    module = importlib.import_module("threed.racketsport.ball_inflections")

    payload = module.build_ball_inflections_from_ball_track(
        _ball_track(
            [
                {"t": 0.0, "xy": [0.0, 0.0], "visible": True, "conf": 0.9, "approx": False},
                {"t": 0.1, "xy": [10.0, 0.0], "visible": True, "conf": 0.8, "approx": False},
                {"t": 0.2, "xy": [10.0, 10.0], "visible": True, "conf": 0.7, "approx": False},
            ]
        ),
        min_turn_degrees=35.0,
        min_speed_px_per_s=25.0,
    )

    assert payload["artifact_type"] == "racketsport_ball_inflections"
    assert payload["source"] == "ball_track_image_motion"
    assert payload["requires_additional_cues"] == ["wrist_velocity_peaks"]
    assert payload["summary"]["candidate_count"] == 1
    candidate = payload["candidates"][0]
    assert candidate["time_s"] == 0.1
    assert candidate["frame"] == 1
    assert candidate["ball_image_xy"] == [10.0, 0.0]
    assert candidate["ball_world_xyz"] is None
    assert candidate["ball_world_xyz_source"] == "missing_from_raw_ball_track"
    assert candidate["render_only"] is True
    assert candidate["not_for_detection_metrics"] is True
    assert candidate["turn_angle_deg"] == 90.0
    assert candidate["speed_before_px_s"] == 100.0
    assert candidate["speed_after_px_s"] == 100.0


def test_ball_inflection_builder_detects_image_turn_over_wider_window() -> None:
    module = importlib.import_module("threed.racketsport.ball_inflections")

    payload = module.build_ball_inflections_from_ball_track(
        _ball_track(
            [
                {"t": 0.0, "xy": [0.0, 0.0], "visible": True, "conf": 0.9, "approx": False},
                {"t": 0.1, "xy": [10.0, 0.0], "visible": True, "conf": 0.9, "approx": False},
                {"t": 0.2, "xy": [20.0, 0.0], "visible": True, "conf": 0.9, "approx": False},
                {"t": 0.3, "xy": [30.0, 0.0], "visible": True, "conf": 0.9, "approx": False},
                {"t": 0.4, "xy": [32.0, 1.0], "visible": True, "conf": 0.9, "approx": False},
                {"t": 0.5, "xy": [34.0, 6.0], "visible": True, "conf": 0.9, "approx": False},
                {"t": 0.6, "xy": [36.0, 15.0], "visible": True, "conf": 0.9, "approx": False},
            ]
        ),
        min_turn_degrees=35.0,
        min_speed_px_per_s=25.0,
    )

    assert payload["summary"]["candidate_count"] >= 1
    assert any(candidate["window_frames"] > 1 for candidate in payload["candidates"])


def test_ball_inflection_builder_preserves_close_image_turns_by_default() -> None:
    module = importlib.import_module("threed.racketsport.ball_inflections")

    payload = module.build_ball_inflections_from_ball_track(
        _ball_track(
            [
                {"t": 0.0, "xy": [0.0, 0.0], "visible": True, "conf": 0.9, "approx": False},
                {"t": 0.05, "xy": [10.0, 0.0], "visible": True, "conf": 0.9, "approx": False},
                {"t": 0.10, "xy": [10.0, 10.0], "visible": True, "conf": 0.9, "approx": False},
                {"t": 0.15, "xy": [20.0, 10.0], "visible": True, "conf": 0.9, "approx": False},
                {"t": 0.20, "xy": [20.0, 20.0], "visible": True, "conf": 0.9, "approx": False},
            ]
        ),
        min_turn_degrees=35.0,
        min_speed_px_per_s=25.0,
    )

    assert payload["summary"]["candidate_count"] == 2
    assert [candidate["time_s"] for candidate in payload["candidates"]] == [0.05, 0.1]


def test_ball_inflection_builder_image_default_filters_weaker_turns() -> None:
    module = importlib.import_module("threed.racketsport.ball_inflections")
    turn_degrees = 50.0
    end_xy = [10.0 + 10.0 * math.cos(math.radians(turn_degrees)), 10.0 * math.sin(math.radians(turn_degrees))]
    ball_track = _ball_track(
        [
            {"t": 0.0, "xy": [0.0, 0.0], "visible": True, "conf": 0.9, "approx": False},
            {"t": 0.1, "xy": [10.0, 0.0], "visible": True, "conf": 0.9, "approx": False},
            {"t": 0.2, "xy": end_xy, "visible": True, "conf": 0.9, "approx": False},
        ]
    )

    default_payload = module.build_ball_inflections_from_ball_track(ball_track)
    lower_threshold_payload = module.build_ball_inflections_from_ball_track(ball_track, min_turn_degrees=35.0)

    assert default_payload["summary"]["candidate_count"] == 0
    assert lower_threshold_payload["summary"]["candidate_count"] == 1


def test_build_ball_inflections_cli_writes_artifact(tmp_path: Path) -> None:
    virtual_world = tmp_path / "virtual_world.json"
    out = tmp_path / "ball_inflections.json"
    virtual_world.write_text(
        json.dumps(
            _virtual_world(
                [
                    {"t": 0.0, "visible": True, "world_xyz": [0.0, 0.0, 0.0], "conf": 0.9, "approx": False},
                    {"t": 0.1, "visible": True, "world_xyz": [1.0, 0.0, 0.0], "conf": 0.8, "approx": False},
                    {"t": 0.2, "visible": True, "world_xyz": [1.0, 1.0, 0.0], "conf": 0.7, "approx": False},
                ]
            )
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/build_ball_inflections.py",
            "--virtual-world",
            str(virtual_world),
            "--out",
            str(out),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert completed.stderr == ""
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["summary"]["candidate_count"] == 1
    assert json.loads(completed.stdout)["candidate_count"] == 1


def test_build_ball_inflections_cli_accepts_ball_track_input(tmp_path: Path) -> None:
    ball_track = tmp_path / "ball_track.json"
    out = tmp_path / "ball_inflections.json"
    ball_track.write_text(
        json.dumps(
            _ball_track(
                [
                    {"t": 0.0, "xy": [0.0, 0.0], "visible": True, "conf": 0.9, "approx": False},
                    {"t": 0.1, "xy": [10.0, 0.0], "visible": True, "conf": 0.8, "approx": False},
                    {"t": 0.2, "xy": [10.0, 10.0], "visible": True, "conf": 0.7, "approx": False},
                ]
            )
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/build_ball_inflections.py",
            "--ball-track",
            str(ball_track),
            "--out",
            str(out),
            "--min-turn-degrees",
            "35.0",
            "--min-speed-px-s",
            "25.0",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert completed.stderr == ""
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["source"] == "ball_track_image_motion"
    assert payload["summary"]["candidate_count"] == 1
    assert json.loads(completed.stdout)["candidate_count"] == 1

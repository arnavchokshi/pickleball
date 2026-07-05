from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest


CLI_PATH = "scripts/racketsport/player_court_membership.py"


def _membership_module():
    try:
        return importlib.import_module("threed.racketsport.player_court_membership")
    except ModuleNotFoundError:
        pytest.fail("threed.racketsport.player_court_membership module is missing")


def _calibration() -> dict:
    return {
        "schema_version": 1,
        "sport": "pickleball",
        "source": "synthetic_unit_test",
        "homography": [[100.0, 0.0, 500.0], [0.0, 100.0, 500.0], [0.0, 0.0, 1.0]],
    }


def _bbox_for_world(x_m: float, y_m: float) -> list[float]:
    cx = 100.0 * x_m + 500.0
    bottom_y = 100.0 * y_m + 500.0
    return [cx - 10.0, bottom_y - 80.0, cx + 10.0, bottom_y]


def _player(player_id: int, world_xy: list[tuple[float, float]], *, fps: float = 10.0) -> dict:
    return {
        "id": player_id,
        "frames": [
            {
                "frame_idx": index,
                "t": index / fps,
                "bbox": _bbox_for_world(x_m, y_m),
                "conf": 0.91,
            }
            for index, (x_m, y_m) in enumerate(world_xy)
        ],
    }


def _tracks(players: list[dict], *, fps: float = 10.0) -> dict:
    return {
        "schema_version": 1,
        "fps": fps,
        "clip": "synthetic_membership_unit",
        "players": players,
    }


def test_player_court_membership_classifies_geometry_with_asymmetric_near_apron() -> None:
    compute = _membership_module().compute_player_court_membership
    tracks = _tracks(
        [
            _player(1, [(0.1, -7.2), (0.15, -7.15), (0.2, -7.2), (0.25, -7.1), (0.3, -7.2)]),
            _player(2, [(2.1, 6.02), (2.1, 6.08), (2.1, 6.14), (2.1, 6.2), (2.1, 6.25)]),
            _player(3, [(-1.0, -2.0), (-0.5, -1.0), (0.0, 0.0), (0.5, 1.0), (1.0, 2.0)]),
            _player(4, [(0.0, 8.3), (0.0, 8.32), (0.0, 8.35), (0.0, 8.31), (0.0, 8.34)]),
            _player(5, [(0.0, 0.0)]),
        ]
    )

    payload = compute(tracks, _calibration())

    assert payload["artifact_type"] == "racketsport_player_court_membership"
    assert payload["verified"] is False
    assert payload["not_gate_verified"] is True
    assert payload["camera_motion_used"] is False
    assert payload["per_player"]["1"]["verdict"] == "on_target_court"
    assert payload["per_player"]["1"]["inside_strict_frac"] == pytest.approx(0.0)
    assert payload["per_player"]["1"]["inside_asym_frac"] == pytest.approx(1.0)
    assert payload["per_player"]["2"]["verdict"] == "adjacent_or_spectator"
    assert "far_boundary_camper" in payload["per_player"]["2"]["reasons"]
    assert payload["per_player"]["3"]["verdict"] == "on_target_court"
    assert payload["per_player"]["4"]["verdict"] == "adjacent_or_spectator"
    assert payload["per_player"]["4"]["median_y_m"] == pytest.approx(8.32)
    assert payload["per_player"]["5"]["verdict"] == "uncertain"
    assert "too_few_frames_for_on_target" in payload["per_player"]["5"]["reasons"]


def test_player_court_membership_uses_compensated_camera_motion_by_frame() -> None:
    compute = _membership_module().compute_player_court_membership
    calibration = {
        "schema_version": 1,
        "sport": "pickleball",
        "homography": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    }
    tracks = _tracks(
        [
            {
                "id": 9,
                "frames": [
                    {"frame_idx": 0, "t": 0.0, "bbox": [9.0, -70.0, 11.0, 10.0], "conf": 0.9},
                    {"frame_idx": 1, "t": 0.1, "bbox": [13.0, -70.0, 15.0, 10.0], "conf": 0.9},
                ],
            }
        ]
    )
    camera_motion = {
        "schema_version": 1,
        "artifact_type": "racketsport_camera_motion",
        "reference_frame_idx": 0,
        "frames": [
            {
                "frame_idx": 0,
                "M": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                "compensated": False,
                "model": "identity",
                "reason": "synthetic_uncompensated",
            },
            {
                "frame_idx": 1,
                "M": [[1.0, 0.0, -4.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                "compensated": True,
                "model": "homography",
            },
        ],
    }

    compensated = compute(tracks, calibration, camera_motion)
    static = compute(tracks, calibration)

    assert compensated["camera_motion_used"] is True
    assert compensated["n_compensated_frames_used"] == 1
    assert compensated["n_uncompensated_frames_used"] == 1
    assert compensated["per_player"]["9"]["median_x_m"] == pytest.approx(10.0)
    assert static["per_player"]["9"]["median_x_m"] == pytest.approx(12.0)


def test_player_court_membership_is_deterministic_and_json_round_trips() -> None:
    compute = _membership_module().compute_player_court_membership
    tracks = _tracks([_player(3, [(-1.0, -1.0), (-0.5, -0.5), (0.0, 0.0), (0.5, 0.5)])])

    first = compute(tracks, _calibration())
    second = compute(tracks, _calibration())
    encoded = json.dumps(first, sort_keys=True)

    assert encoded == json.dumps(second, sort_keys=True)
    assert json.loads(encoded) == first
    assert first["thresholds"]["baseline_y_m"] == pytest.approx(6.7056)
    assert first["per_player"]["3"]["speed_p50_mps"] > 0.0


def test_player_court_membership_cli_help_is_registered() -> None:
    completed = subprocess.run(
        [sys.executable, CLI_PATH, "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert CLI_PATH in Path("tests/racketsport/test_player_court_membership.py").read_text(encoding="utf-8")
    assert "--calibration" in completed.stdout
    assert "--camera-motion" in completed.stdout
    assert "--evidence-dir" in completed.stdout

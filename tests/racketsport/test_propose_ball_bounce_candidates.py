from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tests.racketsport.test_ball_arc_solver import BALL_RADIUS_M, GRAVITY, _project, _projection_calibration


def test_propose_ball_bounce_candidates_cli_emits_honest_track_geometry_payload(tmp_path: Path) -> None:
    calibration = _projection_calibration()
    fps = 30.0
    frames = []
    for frame in range(15):
        t = frame / fps
        y_image = 500.0 + 50.0 * frame
        if frame > 7:
            y_image = 500.0 + 50.0 * (14 - frame)
        frames.append({"t": t, "xy": [960.0 + frame, y_image], "conf": 0.95, "visible": True})
    track_path = _write_json(tmp_path / "ball_track.json", {"schema_version": 1, "fps": fps, "source": "synthetic", "frames": frames, "bounces": []})
    calibration_path = _write_json(tmp_path / "court_calibration.json", calibration)
    out_path = tmp_path / "auto_bounce_candidates.json"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/propose_ball_bounce_candidates.py",
            "--clip",
            "synthetic_cusp",
            "--ball-track",
            str(track_path),
            "--court-calibration",
            str(calibration_path),
            "--out",
            str(out_path),
        ],
        cwd=Path(__file__).resolve().parents[2],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["source"] == "track_geometry_candidate"
    assert payload["human_reviewed"] is False
    assert payload["not_ground_truth"] is True
    assert payload["candidate_prediction"] is True
    assert payload["summary"]["final_candidate_count"] >= 1
    assert payload["candidates"][0]["source"] == "track_geometry_candidate"
    assert payload["candidates"][0]["human_reviewed"] is False
    assert payload["candidates"][0]["not_ground_truth"] is True
    assert payload["candidates"][0]["method"] == "image_y_cusp"
    assert "scripts/racketsport/propose_ball_bounce_candidates.py" in Path(__file__).read_text(encoding="utf-8")


def test_bounce_candidate_visible_runs_use_frame_times_when_track_t_missing() -> None:
    from threed.racketsport.ball_bounce_candidates import _visible_runs

    frame_times = {
        "artifact_type": "racketsport_frame_times",
        "frames": [
            {"frame": 0, "pts_s": 0.0},
            {"frame": 1, "pts_s": 0.025},
            {"frame": 2, "pts_s": 0.110},
        ],
    }
    ball_track = {
        "fps": 30.0,
        "frames": [
            {"xy": [100.0, 100.0], "visible": True, "conf": 0.9},
            {"xy": [110.0, 120.0], "visible": True, "conf": 0.9},
            {"xy": [120.0, 135.0], "visible": True, "conf": 0.9},
        ],
    }

    runs = _visible_runs(ball_track, max_gap_frames=2, frame_times=frame_times)

    assert [item["t"] for item in runs[0]] == [0.0, 0.025, 0.110]
    assert runs[0][2]["t"] != pytest.approx(2.0 / 30.0)


def test_gap_ballistic_intersection_emits_candidate_inside_hidden_bounce_gap() -> None:
    from threed.racketsport.ball_bounce_candidates import build_bounce_candidate_payload

    calibration = _projection_calibration()
    fps = 30.0
    bounce_t = 1.0
    start = (-0.2, -2.0, 1.4)
    v_before = (0.5, 2.8, _vz_for_endpoint(start[2], BALL_RADIUS_M, bounce_t))
    bounce_xyz = _no_drag_position(start, v_before, bounce_t)
    v_after = (0.5, 2.5, _vz_for_endpoint(BALL_RADIUS_M, 1.1, 0.45))
    frames = []
    for frame in range(45):
        t = frame / fps
        if t <= bounce_t:
            world = _no_drag_position(start, v_before, t)
        else:
            world = _no_drag_position(bounce_xyz, v_after, t - bounce_t)
        visible = frame not in {29, 30, 31}
        frames.append(
            {
                "t": t,
                "xy": list(_project(calibration, world)) if visible else [0.0, 0.0],
                "conf": 0.95 if visible else 0.0,
                "visible": visible,
            }
        )

    payload = build_bounce_candidate_payload(
        {"schema_version": 1, "fps": fps, "source": "synthetic", "frames": frames, "bounces": []},
        calibration,
        clip_id="synthetic_gap",
    )

    gap_candidates = [item for item in payload["candidates"] if item["method"] == "gap_ballistic_intersection"]
    assert gap_candidates
    assert gap_candidates[0]["frame"] == pytest.approx(round(bounce_t * fps), abs=1)
    assert gap_candidates[0]["source"] == "track_geometry_candidate"
    assert gap_candidates[0]["human_reviewed"] is False
    assert gap_candidates[0]["not_ground_truth"] is True


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _vz_for_endpoint(z0: float, z1: float, dt: float) -> float:
    return (z1 - z0 + 0.5 * GRAVITY * dt * dt) / dt


def _no_drag_position(
    p0: tuple[float, float, float],
    v0: tuple[float, float, float],
    dt: float,
) -> tuple[float, float, float]:
    return (
        p0[0] + v0[0] * dt,
        p0[1] + v0[1] * dt,
        p0[2] + v0[2] * dt - 0.5 * GRAVITY * dt * dt,
    )

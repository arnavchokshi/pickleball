from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.racketsport.rebuild_ball_trail_from_arc_solved import (
    rebuild_physics_filled,
    rebuild_world_ball_field,
)


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _physics_filled() -> dict:
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_ball_track_physics_filled",
        "fps": 30.0,
        "source": "physics_filled",
        "frames": [
            {"t": 0.0, "xy": [1.0, 1.0], "conf": 0.9, "visible": True, "world_xyz": [0.0, 0.0, 0.0]},
            {"t": 1.0 / 30.0, "xy": [2.0, 2.0], "conf": 0.9, "visible": True, "world_xyz": [0.1, 0.0, 0.0]},
            # This frame is a stale raw/non-arc fallback carried over from a
            # pre-BALL-ARC-SOLVER staging pass -- the arc solver has no
            # confident coverage for it (see the matching arc-solved fixture).
            {"t": 2.0 / 30.0, "xy": [3.0, 3.0], "conf": 0.4, "visible": True, "world_xyz": [-5.0, 5.0, 0.0]},
        ],
        "bounces": [],
    }


def _arc_solved() -> dict:
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_ball_track_arc_solved",
        "frames": [
            {"t": 0.0, "band": "anchored_measured", "world_xyz": [0.0, 0.0, 0.0]},
            {"t": 1.0 / 30.0, "band": "arc_interpolated", "world_xyz": [0.1, 0.0, 0.0]},
            {"t": 2.0 / 30.0, "band": "hidden", "world_xyz": None},
        ],
    }


def _confidence_gated_world() -> dict:
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_virtual_world",
        "players": [{"id": 1, "representation": "track_only", "frames": [{"t": 0.0, "marker": "do-not-touch"}]}],
        "ball": {
            "source": "physics_filled",
            "frames": [
                {
                    "t": 0.0,
                    "xy": [1.0, 1.0],
                    "conf": 0.9,
                    "visible": True,
                    "world_xyz": [0.0, 0.0, 0.0],
                    "confidence_provenance": {
                        "band": "measured",
                        "display_band": "measured",
                        "horizon_frames": 0,
                        "predicted_sigma_m": None,
                        "predictor": "source_artifact",
                    },
                },
                {
                    "t": 1.0 / 30.0,
                    "xy": [2.0, 2.0],
                    "conf": 0.9,
                    "visible": True,
                    "world_xyz": [0.1, 0.0, 0.0],
                    "confidence_provenance": {
                        "band": "measured",
                        "display_band": "measured",
                        "horizon_frames": 0,
                        "predicted_sigma_m": None,
                        "predictor": "source_artifact",
                    },
                },
                {
                    "t": 2.0 / 30.0,
                    "xy": [3.0, 3.0],
                    "conf": 0.4,
                    "visible": True,
                    "world_xyz": [-5.0, 5.0, 0.0],
                    "confidence_provenance": {
                        "band": "physics_predicted_low",
                        "display_band": "physics_predicted_low",
                        "horizon_frames": 0,
                        "predicted_sigma_m": None,
                        "predictor": "source_artifact_low_confidence",
                    },
                },
            ],
        },
    }


def test_rebuild_physics_filled_hides_frame_the_arc_solver_never_covered() -> None:
    rebuilt = rebuild_physics_filled(_physics_filled(), _arc_solved())

    assert [frame["world_xyz"] for frame in rebuilt["frames"]] == [
        [0.0, 0.0, 0.0],
        [0.1, 0.0, 0.0],
        None,
    ]
    assert rebuilt["arc_solved_overlay"]["forced_hidden_frame_count"] == 1


def test_rebuild_world_ball_field_only_touches_ball_frames_that_changed() -> None:
    rebuilt_physics_filled = rebuild_physics_filled(_physics_filled(), _arc_solved())
    world = _confidence_gated_world()

    patched, counts = rebuild_world_ball_field(world, rebuilt_physics_filled)

    assert counts == {"changed_frame_count": 1, "newly_hidden_frame_count": 1}
    assert [frame["world_xyz"] for frame in patched["ball"]["frames"]] == [
        [0.0, 0.0, 0.0],
        [0.1, 0.0, 0.0],
        None,
    ]
    # The newly-hidden frame's confidence provenance is corrected to match --
    # a null position must never wear a "measured"/"physics_predicted_low" badge.
    assert patched["ball"]["frames"][2]["confidence_provenance"]["band"] == "hidden_no_prediction"
    assert patched["ball"]["frames"][2]["confidence_provenance"]["display_band"] == "hidden_no_prediction"
    # Untouched frames are the exact same object contents (xy/conf/visible preserved).
    assert patched["ball"]["frames"][0]["confidence_provenance"]["band"] == "measured"
    assert patched["ball"]["frames"][0]["xy"] == [1.0, 1.0]
    # Player data is completely untouched (do_not_touch marker still present, unmodified).
    assert patched["players"] == world["players"]


def test_rebuild_ball_trail_cli_patches_files_in_place(tmp_path: Path) -> None:
    physics_filled_path = _write_json(tmp_path / "ball_track_physics_filled.json", _physics_filled())
    arc_solved_path = _write_json(tmp_path / "ball_track_arc_solved.json", _arc_solved())
    world_path = _write_json(tmp_path / "confidence_gated_world.json", _confidence_gated_world())

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/rebuild_ball_trail_from_arc_solved.py",
            "--physics-filled",
            str(physics_filled_path),
            "--arc-solved",
            str(arc_solved_path),
            "--world",
            str(world_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stderr == ""

    rebuilt_physics_filled = json.loads(physics_filled_path.read_text(encoding="utf-8"))
    assert rebuilt_physics_filled["frames"][2]["world_xyz"] is None

    patched_world = json.loads(world_path.read_text(encoding="utf-8"))
    assert patched_world["ball"]["frames"][2]["world_xyz"] is None
    assert patched_world["players"] == _confidence_gated_world()["players"]

    report = json.loads(completed.stdout)
    assert report["world_ball_field_changes"] == {"changed_frame_count": 1, "newly_hidden_frame_count": 1}


def test_rebuild_ball_trail_cli_supports_separate_output_paths(tmp_path: Path) -> None:
    physics_filled_path = _write_json(tmp_path / "ball_track_physics_filled.json", _physics_filled())
    arc_solved_path = _write_json(tmp_path / "ball_track_arc_solved.json", _arc_solved())
    out_physics_filled = tmp_path / "rebuilt.json"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/rebuild_ball_trail_from_arc_solved.py",
            "--physics-filled",
            str(physics_filled_path),
            "--arc-solved",
            str(arc_solved_path),
            "--out-physics-filled",
            str(out_physics_filled),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    # The original input is untouched when a separate --out-physics-filled is given.
    assert json.loads(physics_filled_path.read_text(encoding="utf-8")) == _physics_filled()
    assert json.loads(out_physics_filled.read_text(encoding="utf-8"))["frames"][2]["world_xyz"] is None

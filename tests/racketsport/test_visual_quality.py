from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from threed.racketsport.external_gt_body_prediction_schema import MHR70_JOINT_NAMES
from threed.racketsport.visual_quality import estimate_integer_lag_frames, measure_visual_quality


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _joints(root_x: float, *, left_foot_x: float | None = None, right_foot_x: float | None = None) -> list[list[float]]:
    joints = [[root_x, 0.0, 1.0] for _name in MHR70_JOINT_NAMES]
    joints[5] = [root_x - 0.1, 0.0, 1.0]
    joints[6] = [root_x + 0.1, 0.0, 1.0]
    joints[13] = [root_x if left_foot_x is None else left_foot_x, -0.1, 0.0]
    joints[14] = [root_x if right_foot_x is None else right_foot_x, 0.1, 0.0]
    joints[15] = [joints[13][0], -0.2, 0.0]
    joints[16] = [joints[14][0], 0.2, 0.0]
    joints[17] = [joints[13][0], -0.05, 0.0]
    joints[20] = [joints[14][0], 0.05, 0.0]
    return joints


def _make_run_dir(tmp_path: Path, *, pop_frame: int | None = None) -> Path:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    frames = []
    placement_frames = []
    world_frames = []
    for frame_idx in range(8):
        root_x = frame_idx * 0.01
        if pop_frame is not None and frame_idx >= pop_frame:
            root_x += 0.45
        left_foot_x = frame_idx * 0.01
        right_foot_x = 1.0
        frame = {
            "frame_idx": frame_idx,
            "t": frame_idx / 30.0,
            "joints_world": _joints(root_x, left_foot_x=left_foot_x, right_foot_x=right_foot_x),
            "transl_world": [root_x, 0.0, 0.0],
            "joint_conf": [0.9] * len(MHR70_JOINT_NAMES),
            "smoothing_flag": ["none"] * len(MHR70_JOINT_NAMES),
        }
        frames.append(frame)
        world_frames.append({**frame, "track_world_xy": [root_x, 0.0]})
        placement_frames.append(
            {
                "frame_idx": frame_idx,
                "t": frame_idx / 30.0,
                "stance": frame_idx < 4,
                "smoothed_world_xy": [root_x, 0.0],
            }
        )
    _write_json(
        run_dir / "skeleton3d.json",
        {
            "schema_version": 1,
            "artifact_type": "racketsport_skeleton3d",
            "fps": 30.0,
            "joint_names": list(MHR70_JOINT_NAMES),
            "players": [{"id": 1, "frames": frames}],
        },
    )
    _write_json(
        run_dir / "virtual_world.json",
        {
            "schema_version": 1,
            "artifact_type": "racketsport_virtual_world",
            "fps": 30.0,
            "joint_names": list(MHR70_JOINT_NAMES),
            "players": [{"id": 1, "frames": world_frames}],
            "summary": {},
        },
    )
    _write_json(
        run_dir / "placement.json",
        {
            "schema_version": 1,
            "artifact_type": "racketsport_placement",
            "fps": 30.0,
            "players": [{"id": 1, "frames": placement_frames}],
            "summary": {"court_bounds_violations": 0},
        },
    )
    _write_json(
        run_dir / "body_joint_quality.json",
        {
            "schema_version": 1,
            "artifact_type": "body_joint_quality",
            "summary": {"temporal_smoothing_reset_count": 2 if pop_frame is None else 3},
        },
    )
    return run_dir


def test_visual_quality_constant_velocity_has_zero_world_jitter(tmp_path: Path) -> None:
    metrics = measure_visual_quality(_make_run_dir(tmp_path))
    player = metrics["players"]["1"]

    assert player["world_jitter_mm_per_frame2"]["root"]["rms"] == pytest.approx(0.0, abs=1e-9)
    assert player["world_jitter_mm_per_frame2"]["feet"]["rms"] == pytest.approx(0.0, abs=1e-9)
    assert player["root_step_m"]["max"] == pytest.approx(0.01)
    assert player["placement_root_step_m"]["max"] == pytest.approx(0.01)
    assert player["foot_slide_mm_per_frame"]["stance"]["p95"] == pytest.approx(10.0)
    assert player["longest_unanchored_foot_run_frames"] == 4
    assert metrics["summary"]["temporal_smoothing_reset_count"] == 2


def test_visual_quality_detects_injected_root_pop(tmp_path: Path) -> None:
    metrics = measure_visual_quality(_make_run_dir(tmp_path, pop_frame=4))
    player = metrics["players"]["1"]

    assert player["root_step_m"]["max"] > 0.40
    assert player["world_jitter_mm_per_frame2"]["root"]["rms"] > 200.0
    assert metrics["summary"]["temporal_smoothing_reset_count"] == 3


def test_visual_quality_root_steps_are_per_frame_across_sample_gaps(tmp_path: Path) -> None:
    run_dir = _make_run_dir(tmp_path)
    virtual_world = json.loads((run_dir / "virtual_world.json").read_text(encoding="utf-8"))
    placement = json.loads((run_dir / "placement.json").read_text(encoding="utf-8"))
    virtual_world["players"][0]["frames"] = [
        {**virtual_world["players"][0]["frames"][0], "frame_idx": 0, "transl_world": [0.0, 0.0, 0.0]},
        {**virtual_world["players"][0]["frames"][4], "frame_idx": 4, "transl_world": [0.4, 0.0, 0.0]},
    ]
    placement["players"][0]["frames"] = [
        {**placement["players"][0]["frames"][0], "frame_idx": 0, "smoothed_world_xy": [0.0, 0.0]},
        {**placement["players"][0]["frames"][4], "frame_idx": 4, "smoothed_world_xy": [0.4, 0.0]},
    ]
    _write_json(run_dir / "virtual_world.json", virtual_world)
    _write_json(run_dir / "placement.json", placement)

    player = measure_visual_quality(run_dir)["players"]["1"]

    assert player["root_step_m"]["max"] == pytest.approx(0.10)
    assert player["placement_root_step_m"]["max"] == pytest.approx(0.10)


def test_visual_quality_lag_estimator_bounds_centered_smoothing_to_zero_frames() -> None:
    original = [0.0, 1.0, 3.0, 1.0, 0.0]
    centered_smoothed = [0.25, 1.25, 2.0, 1.25, 0.25]

    assert estimate_integer_lag_frames(original, centered_smoothed, max_lag_frames=2) == 0


def test_measure_visual_quality_cli_help_direct_reference() -> None:
    command_path = "scripts/racketsport/measure_visual_quality.py"
    completed = subprocess.run(
        [sys.executable, command_path, "--help"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert command_path.endswith("measure_visual_quality.py")
    assert "--run-dir" in completed.stdout
    assert "--out-dir" in completed.stdout

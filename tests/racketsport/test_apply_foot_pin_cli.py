from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from threed.racketsport.external_gt_body_prediction_schema import MHR70_JOINT_NAMES


COMMAND_PATH = "scripts/racketsport/apply_foot_pin.py"


def _frame(frame_idx: int, x: float) -> dict:
    joints = [[0.0, 0.0, 1.0] for _idx in range(70)]
    for idx in (13, 15, 16, 17):
        joints[idx] = [x, 0.0, 0.0]
    for idx in (14, 18, 19, 20):
        joints[idx] = [1.0, 0.0, 0.20]
    return {
        "frame_idx": frame_idx,
        "t": frame_idx / 30.0,
        "track_world_xy": [x, 0.0],
        "floor_world_xyz": [x, 0.0, 0.0],
        "transl_world": [x, 0.0, 0.0],
        "joints_world": joints,
        "joint_conf": [0.95] * 70,
    }


def _world_payload() -> dict:
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_virtual_world",
        "fps": 30.0,
        "world_frame": "court_Z0",
        "joint_names": list(MHR70_JOINT_NAMES),
        "players": [{"id": 1, "frames": [_frame(idx, idx * 0.01) for idx in range(4)]}],
        "ball": {"frames": [{"t": 0.0, "world_xyz": [0.0, 0.0, 1.0]}]},
    }


def _skeleton_payload() -> dict:
    payload = _world_payload()
    payload["artifact_type"] = "racketsport_skeleton3d"
    for player in payload["players"]:
        for frame in player["frames"]:
            frame.pop("track_world_xy", None)
            frame.pop("floor_world_xyz", None)
    return payload


def test_apply_foot_pin_cli_exposes_direct_help_reference() -> None:
    completed = subprocess.run(
        [sys.executable, COMMAND_PATH, "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--skeleton3d" in completed.stdout
    assert "--world" in completed.stdout
    assert "--out-dir" in completed.stdout
    assert "--in-place" in completed.stdout


def test_apply_foot_pin_cli_writes_corrected_copies_without_mutating_inputs(tmp_path: Path) -> None:
    skeleton = _skeleton_payload()
    world = _world_payload()
    original_skeleton = copy.deepcopy(skeleton)
    original_world = copy.deepcopy(world)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    skeleton_path = run_dir / "skeleton3d.json"
    world_path = run_dir / "confidence_gated_world.json"
    skeleton_path.write_text(json.dumps(skeleton), encoding="utf-8")
    world_path.write_text(json.dumps(world), encoding="utf-8")
    out_dir = tmp_path / "foot_pin"

    completed = subprocess.run(
        [
            sys.executable,
            COMMAND_PATH,
            "--skeleton3d",
            str(skeleton_path),
            "--world",
            str(world_path),
            "--out-dir",
            str(out_dir),
            "--taper-frames",
            "0",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    summary = json.loads(completed.stdout)
    corrected_skeleton = json.loads((out_dir / "skeleton3d.json").read_text(encoding="utf-8"))
    corrected_world = json.loads((out_dir / "confidence_gated_world.json").read_text(encoding="utf-8"))
    audit = json.loads((out_dir / "foot_pin_audit.json").read_text(encoding="utf-8"))
    assert summary["out_dir"] == str(out_dir)
    assert audit["summary"]["stance_slide_after_mm"]["median"] == pytest.approx(0.0)
    assert corrected_skeleton["players"][0]["frames"][0]["joints_world"][13][0] == pytest.approx(0.015)
    assert corrected_world["players"][0]["frames"][0]["joints_world"][13][0] == pytest.approx(0.015)
    assert corrected_world["ball"] == original_world["ball"]
    assert json.loads(skeleton_path.read_text(encoding="utf-8")) == original_skeleton
    assert json.loads(world_path.read_text(encoding="utf-8")) == original_world
    assert corrected_world["foot_pin"]["audit_path"] == "foot_pin_audit.json"

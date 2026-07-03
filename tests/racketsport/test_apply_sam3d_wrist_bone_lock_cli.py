from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from threed.racketsport.skeleton3d import SAM3D_BODY_MHR70_SEMANTIC_MAP


def test_apply_sam3d_wrist_bone_lock_cli_exposes_direct_help_reference() -> None:
    command_path = "scripts/racketsport/apply_sam3d_wrist_bone_lock.py"

    completed = subprocess.run(
        [sys.executable, command_path, "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--skeleton3d" in completed.stdout
    assert "--player-bone-lengths" in completed.stdout
    assert "--out" in completed.stdout


def test_apply_sam3d_wrist_bone_lock_cli_writes_locked_payload(tmp_path: Path) -> None:
    command_path = "scripts/racketsport/apply_sam3d_wrist_bone_lock.py"
    left_elbow = SAM3D_BODY_MHR70_SEMANTIC_MAP.joints["left_elbow"]
    left_wrist = SAM3D_BODY_MHR70_SEMANTIC_MAP.joints["left_wrist"]
    joints = [[0.0, 0.0, 0.0] for _idx in range(70)]
    joints[left_elbow] = [1.0, 0.0, 1.0]
    joints[left_wrist] = [2.0, 0.0, 1.0]
    skeleton = {
        "schema_version": 1,
        "artifact_type": "racketsport_skeleton3d",
        "fps": 30.0,
        "world_frame": "court_Z0",
        "source_model": "sam3d_body_joints",
        "joint_names": [f"sam3dbody_joint_{idx:03d}" for idx in range(70)],
        "preview_only": False,
        "players": [{"id": 1, "frames": [{"frame_idx": 0, "t": 0.0, "joints_world": joints, "joint_conf": [1.0] * 70}]}],
        "provenance": {"source": "sam3d_body_joints"},
    }
    bone_lengths = {
        "artifact_type": "bone_calib_canonical_lengths",
        "players": {"1": {"bones": {"left_lower_arm": {"median_m": 0.25}}}},
    }
    skeleton_path = tmp_path / "skeleton3d.json"
    bone_path = tmp_path / "player_bone_lengths.json"
    out_path = tmp_path / "locked_skeleton3d.json"
    skeleton_path.write_text(json.dumps(skeleton), encoding="utf-8")
    bone_path.write_text(json.dumps(bone_lengths), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            command_path,
            "--skeleton3d",
            str(skeleton_path),
            "--player-bone-lengths",
            str(bone_path),
            "--out",
            str(out_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    summary = json.loads(completed.stdout)
    assert summary["out"] == str(out_path)
    locked_frame = payload["players"][0]["frames"][0]
    assert locked_frame["joints_world"][left_wrist] == pytest.approx([1.25, 0.0, 1.0])
    assert payload["provenance"]["sam3d_wrist_bone_lock"]["players"]["1"]["left_lower_arm"]["locked_frame_count"] == 1

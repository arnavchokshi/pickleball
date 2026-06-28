from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from threed.racketsport.physics_world_refinement import build_physics_refinement_from_virtual_world


def _world() -> dict:
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_virtual_world",
        "fps": 30.0,
        "players": [
            {
                "id": 1,
                "frames": [
                    {
                        "t": 0.0,
                        "floor_world_xyz": [0.0, 0.0, 0.0],
                        "transl_world": [0.0, 0.0, 0.0],
                        "min_mesh_z_m": -0.003,
                        "foot_contact": {"left": True, "right": False},
                    },
                    {
                        "t": 1.0 / 30.0,
                        "floor_world_xyz": [0.002, 0.0, 0.0],
                        "transl_world": [0.002, 0.0, 0.0],
                        "min_mesh_z_m": 0.0,
                        "foot_contact": {"left": True, "right": False},
                    },
                ],
            },
            {
                "id": 2,
                "frames": [
                    {
                        "t": 0.0,
                        "floor_world_xyz": [0.45, 0.0, 0.0],
                        "transl_world": [0.45, 0.0, 0.0],
                        "min_mesh_z_m": 0.0,
                        "foot_contact": {"left": False, "right": False},
                    }
                ],
            },
        ],
        "ball": {"frames": [{"t": 0.0}]},
        "summary": {"floor_placed_player_frame_count": 3, "mesh_player_frame_count": 3},
    }


def test_build_physics_refinement_from_virtual_world_packages_contact_constraints() -> None:
    artifact = build_physics_refinement_from_virtual_world(
        clip_id="clip-a",
        virtual_world=_world(),
        requested_mode="auto",
        mjx_available=False,
        pad_frames=0,
    )

    assert artifact["artifact_type"] == "racketsport_physics_refinement"
    assert artifact["clip_id"] == "clip-a"
    assert artifact["physics"] == "cpu_fallback_scaffold"
    assert artifact["foot2_done"] is False
    assert artifact["must_not_mark_done_verified"] is True
    assert artifact["constraint_summary"]["contact_frames"] == 2
    assert artifact["constraint_summary"]["max_contact_slide_m"] == pytest.approx(0.002)
    assert artifact["constraint_summary"]["max_floor_penetration_m"] == pytest.approx(0.003)
    assert artifact["constraint_summary"]["inter_player_penetration_frames"] == 1
    assert artifact["refinement_windows"] == [
        {"start_frame": 0, "end_frame": 1, "player_id": "1", "reason": "left_foot_contact"}
    ]
    assert artifact["source_summary"] == {
        "floor_placed_player_frame_count": 3,
        "mesh_player_frame_count": 3,
        "foot_sample_count": 6,
        "root_sample_count": 3,
    }


def test_physics_refinement_cli_writes_artifact(tmp_path: Path) -> None:
    world_path = tmp_path / "virtual_world.json"
    out = tmp_path / "physics_refinement.json"
    world_path.write_text(json.dumps(_world()), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/build_physics_refinement_from_world.py",
            "--clip",
            "clip-a",
            "--virtual-world",
            str(world_path),
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
    assert payload["artifact_type"] == "racketsport_physics_refinement"
    assert json.loads(completed.stdout)["out"] == str(out)

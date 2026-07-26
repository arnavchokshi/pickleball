from __future__ import annotations

import runpy
from pathlib import Path

import numpy as np


RENDERER = Path(__file__).resolve().parents[2] / "web" / "replay" / "tools" / "render_demo_comparison.py"
MODULE = runpy.run_path(str(RENDERER))
MeshFrame = MODULE["MeshFrame"]
WorldFrame = MODULE["WorldFrame"]
mesh_sample_at = MODULE["mesh_sample_at"]
world_sample_at = MODULE["world_sample_at"]
resolve_fps_multiplier = MODULE["resolve_fps_multiplier"]


def _mesh_frame(x_mm: int, *, window: int = 0):
    vertices = np.full((4, 3), x_mm, dtype=np.int16)
    joints = np.full((70, 3), x_mm, dtype=np.int16)
    return MeshFrame(
        vertices_mm=vertices,
        joints_mm=joints,
        joint_conf=np.ones(70, dtype=np.float32),
        blend_weight=1.0,
        source_window_index=window,
    )


def _world_frame(x_m: float, *, mesh_player_id: int = 1):
    joints = np.full((70, 3), x_m, dtype=np.float32)
    return WorldFrame(
        joints_m=joints,
        joint_conf=np.ones(70, dtype=np.float32),
        mesh_player_id=mesh_player_id,
    )


def test_sparse_60fps_ticks_interpolate_across_measured_33ms_pair() -> None:
    mesh = mesh_sample_at({0: _mesh_frame(0), 2: _mesh_frame(2000)}, 1.0, 60.0)
    world = world_sample_at({0: _world_frame(0), 2: _world_frame(2)}, 1.0, 60.0)

    assert mesh is not None
    assert np.allclose(mesh[0], 1.0)
    assert world is not None
    assert np.allclose(world.joints_m, 1.0)


def test_display_interpolation_refuses_long_or_identity_ambiguous_gap() -> None:
    assert mesh_sample_at({0: _mesh_frame(0), 2: _mesh_frame(2000)}, 1.0, 30.0) is None
    assert mesh_sample_at({0: _mesh_frame(0, window=0), 2: _mesh_frame(2000, window=1)}, 1.0, 60.0) is None
    assert world_sample_at(
        {0: _world_frame(0, mesh_player_id=1), 2: _world_frame(2, mesh_player_id=2)},
        1.0,
        60.0,
    ) is None


def test_native_source_fps_is_default_when_ratio_is_integral() -> None:
    assert resolve_fps_multiplier(30.0, 60.0, None) == 2
    assert resolve_fps_multiplier(60.0, 60.0, None) == 1
    assert resolve_fps_multiplier(30.0, 59.94, None) == 1
    assert resolve_fps_multiplier(30.0, 60.0, 1) == 1

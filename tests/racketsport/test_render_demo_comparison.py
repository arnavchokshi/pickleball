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
baseline_camera = MODULE["baseline_camera"]
draw_translucent_joint_avatar = MODULE["draw_translucent_joint_avatar"]
stabilize_world_frames_for_presentation = MODULE["stabilize_world_frames_for_presentation"]
stabilize_bboxes_for_presentation = MODULE["stabilize_bboxes_for_presentation"]
center_joints_for_studio = MODULE["center_joints_for_studio"]
studio_orbit_angle = MODULE["studio_orbit_angle"]
tracked_person_crop_geometry = MODULE["tracked_person_crop_geometry"]
fit_tracked_person_panel = MODULE["fit_tracked_person_panel"]
source_seek_seconds = MODULE["source_seek_seconds"]
world_sample_at = MODULE["world_sample_at"]


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


def test_joint_only_world_draws_translucent_avatar_without_mesh_geometry() -> None:
    panel = np.full((360, 640, 3), 244, dtype=np.uint8)
    before = panel.copy()
    joints = np.zeros((70, 3), dtype=np.float32)
    joints[:, 1] = 0.5
    joints[:, 2] = 1.0
    joints[69] = [0.0, 0.5, 1.55]
    joints[0] = [0.0, 0.5, 1.78]
    joints[5], joints[6] = [-0.22, 0.5, 1.48], [0.22, 0.5, 1.48]
    joints[9], joints[10] = [-0.16, 0.5, 1.0], [0.16, 0.5, 1.0]
    joints[7], joints[8] = [-0.34, 0.5, 1.2], [0.34, 0.5, 1.2]
    joints[62], joints[41] = [-0.42, 0.5, 0.96], [0.42, 0.5, 0.96]
    joints[11], joints[12] = [-0.17, 0.5, 0.55], [0.17, 0.5, 0.55]
    joints[13], joints[14] = [-0.18, 0.5, 0.08], [0.18, 0.5, 0.08]
    for index in (15, 16, 17):
        joints[index] = [-0.18, 0.64, 0.03]
    for index in (18, 19, 20):
        joints[index] = [0.18, 0.64, 0.03]

    draw_translucent_joint_avatar(
        panel,
        joints,
        np.ones(70, dtype=np.float32),
        (61, 255, 223),
        baseline_camera(640, 360),
    )

    assert np.count_nonzero(panel != before) > 100


def test_presentation_stabilizer_reduces_root_snap_without_changing_pose() -> None:
    frames = {}
    raw_roots = [0.0, 0.05, 0.10, 2.5, 0.20, 0.25, 0.30]
    for frame_idx, root_x in enumerate(raw_roots):
        joints = np.zeros((70, 3), dtype=np.float32)
        joints[:, 0] = root_x
        joints[:, 2] = np.linspace(0.0, 1.8, 70)
        frames[frame_idx] = WorldFrame(
            joints_m=joints,
            joint_conf=np.ones(70, dtype=np.float32),
            mesh_player_id=1,
        )

    stabilized, stats = stabilize_world_frames_for_presentation({1: frames}, 30.0)
    raw_steps = np.abs(np.diff(raw_roots))
    stabilized_roots = np.asarray(
        [stabilized[1][idx].joints_m[9:11, 0].mean() for idx in range(len(raw_roots))]
    )
    stabilized_steps = np.abs(np.diff(stabilized_roots))

    assert stabilized_steps.max() < raw_steps.max() * 0.25
    assert stats["authority"] == "presentation_only"
    assert stats["frames"] == len(raw_roots)
    for frame_idx in frames:
        raw_bones = frames[frame_idx].joints_m[1:] - frames[frame_idx].joints_m[:-1]
        stabilized_bones = (
            stabilized[1][frame_idx].joints_m[1:]
            - stabilized[1][frame_idx].joints_m[:-1]
        )
        assert np.allclose(stabilized_bones, raw_bones)


def test_presentation_stabilizer_never_bridges_missing_gap() -> None:
    frames = {
        frame_idx: _world_frame(float(frame_idx))
        for frame_idx in (0, 1, 2, 20, 21, 22)
    }

    stabilized, stats = stabilize_world_frames_for_presentation({1: frames}, 30.0)

    assert set(stabilized[1]) == set(frames)
    assert 3 not in stabilized[1]
    assert 19 not in stabilized[1]
    assert stats["frames"] == 0


def test_presentation_stabilizer_uses_floor_path_and_preserves_it() -> None:
    frames = {}
    for frame_idx in range(9):
        floor_xy = np.asarray([frame_idx * 0.04, 1.0], dtype=np.float32)
        root_xy = floor_xy + np.asarray([0.2, 0.1], dtype=np.float32)
        if frame_idx == 4:
            root_xy += np.asarray([2.0, -1.0], dtype=np.float32)
        joints = np.zeros((70, 3), dtype=np.float32)
        joints[:, :2] = root_xy
        frames[frame_idx] = WorldFrame(
            joints_m=joints,
            joint_conf=np.ones(70, dtype=np.float32),
            mesh_player_id=1,
            floor_xy_m=floor_xy,
        )

    stabilized, stats = stabilize_world_frames_for_presentation({1: frames}, 30.0)

    assert stats["anchor_segments"] == 1
    assert stats["root_fallback_segments"] == 0
    assert np.allclose(stabilized[1][4].floor_xy_m, frames[4].floor_xy_m)
    stabilized_root = stabilized[1][4].joints_m[9:11, :2].mean(axis=0)
    assert np.linalg.norm(stabilized_root - frames[4].floor_xy_m) < 0.5


def test_body_local_centering_removes_court_xy_but_preserves_pose_and_height() -> None:
    rng = np.random.default_rng(13)
    joints = rng.normal(size=(70, 3)).astype(np.float32)
    translation = np.asarray([3.2, -4.1, 0.7], dtype=np.float32)
    world = joints + translation
    frame = WorldFrame(
        joints_m=world,
        joint_conf=np.ones(70, dtype=np.float32),
        mesh_player_id=1,
        translation_world=translation,
    )

    local = center_joints_for_studio(frame)

    assert local is not None
    hip = world[9:11].mean(axis=0)
    assert np.allclose(local[9:11, :2].mean(axis=0), 0.0, atol=5e-7)
    assert np.allclose(local[:, 2], world[:, 2], atol=5e-7)
    assert np.allclose(local[:, :2] + hip[:2], world[:, :2], atol=5e-7)
    assert np.allclose(
        local[1:] - local[:-1],
        world[1:] - world[:-1],
        atol=5e-7,
    )


def test_tracked_bbox_stabilization_preserves_keys_gaps_and_raw_boxes() -> None:
    frames = {}
    for frame_idx in (*range(8), *range(20, 28)):
        x = float(frame_idx)
        if frame_idx == 4:
            x += 80.0
        bbox = np.asarray([x, 20.0, x + 40.0, 140.0], dtype=np.float32)
        frames[frame_idx] = WorldFrame(
            joints_m=np.zeros((70, 3), dtype=np.float32),
            joint_conf=np.ones(70, dtype=np.float32),
            mesh_player_id=1,
            bbox_xyxy=bbox,
            raw_bbox_xyxy=bbox.copy(),
        )

    stabilized, stats = stabilize_bboxes_for_presentation(frames, 30.0)

    assert set(stabilized) == set(frames)
    assert 8 not in stabilized and 19 not in stabilized
    assert stats["segments"] == 2
    assert stats["smoothed_center_step_p95_px"] < stats["raw_center_step_p95_px"]
    for frame_idx in frames:
        assert np.array_equal(stabilized[frame_idx].raw_bbox_xyxy, frames[frame_idx].raw_bbox_xyxy)


def test_tracked_bbox_stabilization_never_smooths_across_identity_change() -> None:
    frames = {}
    for frame_idx in range(12):
        identity = 1 if frame_idx < 6 else 2
        center_x = 60.0 if identity == 1 else 1000.0
        bbox = np.asarray([center_x - 20, 20, center_x + 20, 140], dtype=np.float32)
        frames[frame_idx] = WorldFrame(
            joints_m=np.zeros((70, 3), dtype=np.float32),
            joint_conf=np.ones(70, dtype=np.float32),
            mesh_player_id=identity,
            bbox_xyxy=bbox,
            raw_bbox_xyxy=bbox.copy(),
        )

    stabilized, stats = stabilize_bboxes_for_presentation(frames, 30.0)

    assert stats["segments"] == 2
    assert stabilized[5].bbox_xyxy[0] < 100
    assert stabilized[6].bbox_xyxy[0] > 900


def test_player_closeup_interpolation_respects_body_cadence_gap() -> None:
    def frame(x: float) -> WorldFrame:
        return WorldFrame(
            joints_m=np.full((70, 3), x, dtype=np.float32),
            joint_conf=np.ones(70, dtype=np.float32),
            mesh_player_id=1,
        )

    assert world_sample_at({0: frame(0), 4: frame(4)}, 2.0, 60.0, max_gap_frames=2) is None
    interpolated = world_sample_at(
        {0: frame(0), 2: frame(2)},
        1.0,
        60.0,
        max_gap_frames=2,
    )
    assert interpolated is not None
    assert np.allclose(interpolated.joints_m, 1.0)


def test_source_seek_includes_mesh_window_base_offset() -> None:
    assert np.isclose(source_seek_seconds(300, 30.0, 2.5), 12.5)


def test_tracked_crop_contains_raw_box_with_guard_margin() -> None:
    smoothed = np.asarray([300.0, 120.0, 420.0, 600.0], dtype=np.float32)
    raw = np.asarray([250.0, 100.0, 455.0, 625.0], dtype=np.float32)

    center, crop_w, crop_h = tracked_person_crop_geometry(
        smoothed,
        raw,
        1280,
        720,
        1920,
        1080,
    )
    crop = np.asarray(
        [center[0] - crop_w / 2, center[1] - crop_h / 2,
         center[0] + crop_w / 2, center[1] + crop_h / 2]
    )
    guard = (raw[2:] - raw[:2]) * 0.10

    assert crop[0] <= raw[0] - guard[0]
    assert crop[1] <= raw[1] - guard[1]
    assert crop[2] >= raw[2] + guard[0]
    assert crop[3] >= raw[3] + guard[1]
    assert np.isclose(crop_w / crop_h, 16 / 9)


def test_tracked_crop_near_edge_stays_inside_real_source_pixels() -> None:
    smoothed = np.asarray([1700.0, 200.0, 1910.0, 850.0], dtype=np.float32)
    raw = np.asarray([1710.0, 190.0, 1918.0, 870.0], dtype=np.float32)

    center, crop_w, crop_h = tracked_person_crop_geometry(
        smoothed,
        raw,
        1280,
        720,
        1920,
        1080,
    )

    assert center[0] - crop_w / 2 >= -1e-6
    assert center[1] - crop_h / 2 >= -1e-6
    assert center[0] + crop_w / 2 <= 1920 + 1e-6
    assert center[1] + crop_h / 2 <= 1080 + 1e-6


def test_tracked_crop_target_marker_changes_only_presentation_pixels() -> None:
    frame = np.full((720, 1280, 3), 96, dtype=np.uint8)
    bbox = np.asarray([460.0, 100.0, 760.0, 650.0], dtype=np.float32)

    plain = fit_tracked_person_panel(frame, bbox, bbox, 1280, 720)
    marked = fit_tracked_person_panel(frame, bbox, bbox, 1280, 720, player_id=1)

    assert plain.shape == marked.shape == (720, 1280, 3)
    assert np.count_nonzero(plain != marked) > 200


def test_studio_angle_schedule_has_four_holds_and_eased_transitions() -> None:
    assert studio_orbit_angle(0.0) == 0.0
    assert studio_orbit_angle(0.21) == 0.0
    assert studio_orbit_angle(0.40) == 45.0
    assert studio_orbit_angle(0.66) == 90.0
    assert studio_orbit_angle(0.90) == 180.0
    assert 0.0 < studio_orbit_angle(0.27) < 45.0
    assert 45.0 < studio_orbit_angle(0.53) < 90.0
    assert 90.0 < studio_orbit_angle(0.79) < 180.0

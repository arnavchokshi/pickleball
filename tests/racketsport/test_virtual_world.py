from __future__ import annotations

import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from tests.racketsport.calibration_fixtures import minimal_calibration_image_pts, minimal_calibration_world_pts
from threed.racketsport.eval_guard import EvalClipLeakError
from threed.racketsport.external_gt_body_prediction_schema import MHR70_JOINT_NAMES
from threed.racketsport.schemas import VirtualWorld, validate_artifact_file
from threed.racketsport.virtual_world import (
    apply_ball_track_arc_solved_overlay,
    build_virtual_world_state,
    build_virtual_world_state_from_run_dir,
)


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _court_calibration() -> dict:
    return {
        "schema_version": 1,
        "sport": "pickleball",
        "homography": [[1.0, 0.0, 960.0], [0.0, 1.0, 540.0], [0.0, 0.0, 1.0]],
        "intrinsics": {"fx": 1000.0, "fy": 1000.0, "cx": 960.0, "cy": 540.0, "dist": [], "source": "manual"},
        "extrinsics": {
            "R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "t": [0.0, 0.0, 12.0],
            "camera_height_m": 12.0,
        },
        "reprojection_error_px": {"median": 1.2, "p95": 3.4},
        "capture_quality": {"grade": "good", "reasons": []},
        "image_pts": minimal_calibration_image_pts(),
        "world_pts": minimal_calibration_world_pts(),
    }


def _metric15_court_calibration() -> dict:
    payload = _court_calibration()
    payload["coordinate_frame"] = "court_netcenter_z_up_m"
    payload["T_world_court"] = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    payload["source"] = "metric_15pt_reviewed"
    payload["metric_confidence"] = "high"
    payload["per_keypoint_residual_px"] = [0.5] * 15
    payload["gsd_model"] = {
        "type": "analytic_ray_plane",
        "plane_sigma_m": 0.0,
        "calibration_sigma_m": 0.0,
        "samples": [],
    }
    payload["solved_over_frames"] = [0]
    payload["intrinsics"]["source"] = "metric_15pt_reviewed"
    payload["capture_quality"] = {
        "grade": "warn",
        "reasons": [
            "single_view_planar_full_calibration",
            "reviewed_15pt_correspondences",
        ],
    }
    return payload


def _tracks() -> dict:
    return {
        "schema_version": 1,
        "fps": 60.0,
        "players": [
            {
                "id": 7,
                "side": "near",
                "role": "left",
                "frames": [{"t": 0.0, "bbox": [100.0, 100.0, 140.0, 240.0], "world_xy": [0.25, -2.0], "conf": 0.91}],
            }
        ],
        "rally_spans": [],
    }


def _tracks_two_frames() -> dict:
    payload = _tracks()
    payload["players"][0]["frames"].append(
        {
            "t": 1.0 / 60.0,
            "bbox": [102.0, 102.0, 142.0, 242.0],
            "world_xy": [0.3, -1.95],
            "conf": 0.89,
        }
    )
    return payload


def _tracks_ten_frame_clip() -> dict:
    return {
        "schema_version": 1,
        "fps": 10.0,
        "players": [
            {
                "id": 7,
                "side": "near",
                "role": "left",
                "frames": [
                    {
                        "t": index / 10.0,
                        "bbox": [100.0 + index, 100.0, 140.0 + index, 240.0],
                        "world_xy": [0.25 + index * 0.01, -2.0],
                        "conf": 0.91,
                    }
                    for index in range(10)
                ],
            }
        ],
        "rally_spans": [],
    }


def _tracks_ragged_two_players() -> dict:
    return {
        "schema_version": 1,
        "fps": 10.0,
        "players": [
            {
                "id": 7,
                "side": "near",
                "role": "left",
                "frames": [
                    {"t": 0.0, "bbox": [100.0, 100.0, 140.0, 240.0], "world_xy": [0.25, -2.0], "conf": 0.91},
                    {"t": 0.1, "bbox": [101.0, 100.0, 141.0, 240.0], "world_xy": [0.26, -2.0], "conf": 0.91},
                    {"t": 0.2, "bbox": [102.0, 100.0, 142.0, 240.0], "world_xy": [0.27, -2.0], "conf": 0.91},
                ],
            },
            {
                "id": 8,
                "side": "near",
                "role": "right",
                "frames": [
                    {"t": 0.0, "bbox": [200.0, 100.0, 240.0, 240.0], "world_xy": [1.25, -2.0], "conf": 0.88},
                    {"t": 0.2, "bbox": [202.0, 100.0, 242.0, 240.0], "world_xy": [1.27, -2.0], "conf": 0.88},
                ],
            },
        ],
        "rally_spans": [],
    }


def _smpl_motion() -> dict:
    return {
        "schema_version": 1,
        "model": "smplx",
        "fps": 60.0,
        "world_frame": "court_Z0",
        "players": [
            {
                "id": 7,
                "betas": [0.0] * 10,
                "skate_free": True,
                "physics": "worldhmr_grounded_not_footlocked",
                "frames": [
                    {
                        "t": 0.0,
                        "global_orient": [0.0, 0.0, 0.0],
                        "body_pose": [0.0] * 63,
                        "left_hand_pose": [],
                        "right_hand_pose": [],
                        "transl_world": [0.25, -2.0, 0.0],
                        "joints_world": [[0.25, -2.0, 0.0], [0.25, -2.0, 1.6]],
                        "mesh_vertices_world": [[0.2, -2.05, 0.0], [0.3, -1.95, 1.7]],
                        "joint_conf": [0.9, 0.88],
                        "foot_contact": {"left": True, "right": True},
                        "grf": None,
                    }
                ],
            }
        ],
    }


def _skeleton3d_second_frame() -> dict:
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_skeleton3d",
        "fps": 60.0,
        "world_frame": "court_Z0",
        "source_model": "sat_hmr_world_joints",
        "joint_names": ["nose", "left_shoulder"],
        "preview_only": True,
        "players": [
            {
                "id": 7,
                "frames": [
                    {
                        "frame_idx": 1,
                        "t": 1.0 / 60.0,
                        "joints_world": [[0.31, -1.95, 1.55], [0.32, -1.96, 1.35]],
                        "joint_conf": [0.73, 0.71],
                    }
                ],
            }
        ],
        "provenance": {"source": "lane_a_skeleton3d"},
    }


def _sam3d_skeleton3d_same_first_frame() -> dict:
    joints = [[float(index), float(index) + 0.1, float(index) + 0.2] for index in range(70)]
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_skeleton3d",
        "fps": 60.0,
        "world_frame": "court_Z0",
        "source_model": "sam3d_body_joints",
        "joint_names": [f"sam3dbody_joint_{index:03d}" for index in range(70)],
        "preview_only": True,
        "players": [
            {
                "id": 7,
                "frames": [
                    {
                        "frame_idx": 0,
                        "t": 0.0,
                        "joints_world": joints,
                        "joint_conf": [0.5 + index / 1000.0 for index in range(70)],
                    }
                ],
            }
        ],
        "provenance": {"source": "sam3d_refined_skeleton3d"},
    }


def _ball_track() -> dict:
    return {
        "schema_version": 1,
        "fps": 60.0,
        "source": "tracknet",
        "frames": [{"t": 0.0, "xy": [320.0, 240.0], "conf": 0.83, "visible": True, "world_xyz": [0.1, -0.8, 0.9]}],
        "bounces": [],
    }


def _ball_track_half_clip() -> dict:
    return {
        "schema_version": 1,
        "fps": 10.0,
        "source": "tracknet",
        "frames": [
            {"t": 0.0, "xy": [320.0, 240.0], "conf": 0.83, "visible": True, "world_xyz": [0.1, -0.8, 0.9]},
            {"t": 0.4, "xy": [321.0, 241.0], "conf": 0.81, "visible": True, "world_xyz": [0.2, -0.7, 0.8]},
        ],
        "bounces": [],
    }


def _ball_track_without_world_xyz() -> dict:
    return {
        "schema_version": 1,
        "fps": 60.0,
        "source": "tracknet",
        "frames": [{"t": 0.0, "xy": [961.0, 542.0], "conf": 0.83, "visible": True}],
        "bounces": [],
    }


def _racket_pose() -> dict:
    return {
        "schema_version": 1,
        "fps": 120.0,
        "world_frame": "court_Z0",
        "translation_unit": "m",
        "players": [
            {
                "id": 7,
                "paddle_dims_in": {"length": 15.5, "width": 7.5},
                "frames": [
                    {
                        "t": 0.0,
                        "pose_se3": {
                            "R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                            "t": [0.45, -1.9, 0.85],
                        },
                        "conf": 0.72,
                        "world_frame": "court_Z0",
                        "translation_unit": "m",
                        "source": "pnp_ippe:court_Z0",
                        "reprojection_error_px": 2.1,
                        "ambiguous": False,
                    }
                ],
                "contacts": [],
            }
        ],
    }


def _physics_footlock() -> dict:
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_physics_footlock",
        "world_frame": "court_Z0",
        "players": [
            {
                "id": 7,
                "frames": [
                    {
                        "t": 0.0,
                        "floor_world_xyz": [0.11, -2.22, 0.07],
                        "foot_contact": {"left": True, "right": False},
                        "contact_locked": True,
                        "trust_band": "corrected",
                        "source": "foot_lock_ik",
                        "grf": [[0.0, 0.0, 1.0]],
                    }
                ],
            }
        ],
    }


def _physics_footlock_actual_schema() -> dict:
    return {
        "schema_version": 1,
        "artifact_type": "physics_footlock",
        "joint_names": ["nose", "left_shoulder"],
        "players": [
            {
                "id": 7,
                "frames": [
                    {
                        "frame_index": 0,
                        "t": 0.0,
                        "joints_world": [[0.31, -2.01, 1.55], [0.32, -2.02, 1.35]],
                        "joint_conf": [0.93, 0.91],
                        "foot_lock": {
                            "frame_index": 0,
                            "player_id": 7,
                            "active_contacts": [
                                {
                                    "foot": "left",
                                    "anchor_position_xyz": [0.21, -2.11, 0.0],
                                    "start_frame_index": 0,
                                    "end_frame_index": 1,
                                }
                            ],
                            "root_delta_xyz": [0.01, 0.0, -0.02],
                            "max_any_joint_displacement_m": 0.02,
                            "max_non_foot_joint_displacement_m": 0.0,
                        },
                    }
                ],
            }
        ],
        "trust_band": {
            "stage": "FOOT/PHYS",
            "gate_id": "foot_slide_floor_penetration_gate",
            "gate_status": "pass",
            "badge": "preview",
            "reason": "max slide passed",
            "evidence_path": "foot_slide_report.json",
        },
    }


def _ball_track_physics_filled() -> dict:
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_ball_track_physics_filled",
        "fps": 60.0,
        "source": "physics_filled",
        "frames": [
            {
                "t": 0.0,
                "xy": [320.0, 240.0],
                "conf": 0.83,
                "visible": True,
                "world_xyz": [0.1, -0.8, 0.9],
                "trust_band": "corrected",
            },
            {
                "t": 1.0 / 60.0,
                "xy": [321.0, 241.0],
                "conf": 0.61,
                "visible": False,
                "world_xyz": [0.2, -0.7, 0.05],
                "trust_band": "interpolated",
            },
            {
                "t": 2.0 / 60.0,
                "xy": [322.0, 242.0],
                "conf": 0.76,
                "visible": True,
                "world_xyz": [0.22, -0.69, 0.42],
                "trust_band": "physics_derived",
            },
        ],
        "bounces": [{"t": 1.0 / 60.0, "world_xy": [0.2, -0.7], "source": "physics_bounce"}],
    }


def _ball_track_arc_solved() -> dict:
    """BALL-ARC-SOLVER output matching `_ball_track_physics_filled`'s timestamps.

    Frame 1 (t=1/60) is outside the solver's event coverage -- it is honestly
    `hidden`, unlike the physics-filled fixture above which still carries a
    raw/interpolated `world_xyz` for that same frame. Frame 2's position also
    deliberately differs from the physics-filled fixture's value so tests can
    assert the arc wins.
    """

    return {
        "schema_version": 1,
        "artifact_type": "racketsport_ball_track_arc_solved",
        "lane": "BALL-ARC-SOLVER",
        "render_only": True,
        "frames": [
            {"t": 0.0, "band": "anchored_measured", "world_xyz": [0.1, -0.8, 0.9]},
            {"t": 1.0 / 60.0, "band": "hidden", "world_xyz": None},
            {"t": 2.0 / 60.0, "band": "arc_interpolated", "world_xyz": [0.5, -0.5, 0.3]},
        ],
    }


def _ball_track_arc_solved_monotonic_flight(*, frame_count: int = 5, fps: float = 60.0) -> dict:
    """A fully-covered arc solution whose x position is strictly increasing.

    A true parabola never reverses its horizontal velocity direction mid
    flight, so this fixture is the "kink-free" ground truth used to prove a
    deviating raw sighting elsewhere in the pipeline cannot leak into the
    rendered trail once the arc-solved overlay is applied.
    """

    return {
        "schema_version": 1,
        "artifact_type": "racketsport_ball_track_arc_solved",
        "lane": "BALL-ARC-SOLVER",
        "render_only": True,
        "frames": [
            {
                "t": index / fps,
                "band": "anchored_measured" if index in (0, frame_count - 1) else "arc_interpolated",
                "world_xyz": [0.1 * index, -0.8 + 0.05 * index, 0.9 - 0.02 * index],
            }
            for index in range(frame_count)
        ],
    }


def _ball_track_physics_filled_with_deviating_mid_flight_sighting(*, frame_count: int = 5, fps: float = 60.0) -> dict:
    """A physics-filled ball track with a raw sighting that kinks mid-segment.

    Every frame matches `_ball_track_arc_solved_monotonic_flight`'s timestamps
    and x/z trend except frame 2, whose x/y jump far off the line -- exactly
    the shape of a bad 2D-lifted sighting or stale non-arc fallback leaking
    into an otherwise clean flight.
    """

    frames = []
    for index in range(frame_count):
        world_xyz = [0.1 * index, -0.8 + 0.05 * index, 0.9 - 0.02 * index]
        if index == 2:
            world_xyz = [-9.0, 9.0, 4.0]
        frames.append(
            {
                "t": index / fps,
                "xy": [320.0 + index, 240.0 + index],
                "conf": 0.8,
                "visible": True,
                "world_xyz": world_xyz,
                "trust_band": "physics_derived",
            }
        )
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_ball_track_physics_filled",
        "fps": fps,
        "source": "physics_filled",
        "frames": frames,
        "bounces": [],
    }


def _horizontal_velocity_sign_changes(points: list[list[float]]) -> int:
    """Count direction reversals of horizontal (x, y) velocity across points.

    A single event-bounded parabola segment must never reverse its
    horizontal velocity direction, so any nonzero count here is a kink.
    """

    deltas = [(b[0] - a[0], b[1] - a[1]) for a, b in zip(points, points[1:])]
    changes = 0
    for axis in (0, 1):
        previous_sign = 0
        for delta in deltas:
            value = delta[axis]
            if abs(value) < 1e-9:
                continue
            sign = 1 if value > 0 else -1
            if previous_sign != 0 and sign != previous_sign:
                changes += 1
            previous_sign = sign
    return changes


def _racket_pose_estimate() -> dict:
    payload = _racket_pose()
    payload["artifact_type"] = "racketsport_racket_pose_estimate"
    payload["players"][0]["frames"][0]["source"] = "physics_derived:ukf:court_Z0"
    payload["players"][0]["frames"][0]["trust_band"] = "physics_derived"
    return payload


def _wrist_proxy_pose_estimate() -> dict:
    payload = _racket_pose_estimate()
    payload["source"] = "wrist_proxy"
    payload["render_only"] = True
    payload["trusted_for_rkt_promotion"] = False
    payload["summary"] = {"hidden_frame_count": 3}
    payload["players"][0]["frames"][0]["source"] = "wrist_proxy"
    payload["players"][0]["frames"][0]["render_only"] = True
    payload["players"][0]["frames"][0]["trust"] = "estimated_from_wrist"
    payload["players"][0]["frames"][0]["trust_band"] = {"status": "estimated_from_wrist"}
    return payload


def _ambiguous_racket_pose() -> dict:
    payload = _racket_pose()
    payload["players"][0]["frames"][0]["source"] = "draft_box:pnp_ippe_preview"
    payload["players"][0]["frames"][0]["ambiguous"] = True
    return payload


def test_build_virtual_world_state_combines_court_players_mesh_ball_and_paddle() -> None:
    world = build_virtual_world_state(
        court_calibration=_court_calibration(),
        tracks=_tracks(),
        smpl_motion=_smpl_motion(),
        ball_track=_ball_track(),
        racket_pose=_racket_pose(),
    )

    assert world["artifact_type"] == "racketsport_virtual_world"
    assert world["world_frame"] == "court_Z0"
    assert world["court"]["sport"] == "pickleball"
    assert world["court"]["coordinate_frame"] == "origin_net_center_x_width_y_length_z_up_m"
    assert "near_nvz" in world["court"]["line_segments"]
    assert world["court"]["net"]["center_height_m"] > 0.8
    assert world["players"][0]["id"] == 7
    assert world["players"][0]["representation"] == "mesh"
    assert world["players"][0]["frames"][0]["track_world_xy"] == [0.25, -2.0]
    assert world["players"][0]["frames"][0]["joint_count"] == 2
    assert world["players"][0]["frames"][0]["mesh_vertex_count"] == 2
    assert world["players"][0]["frames"][0]["mesh_vertices_world"] == []
    assert world["players"][0]["frames"][0]["mesh_ref"] == {
        "artifact": "body_mesh.json",
        "player_id": 7,
        "frame_idx": 0,
        "t": 0.0,
    }
    assert world["players"][0]["frames"][0]["floor_world_xyz"] == [0.25, -2.0, 0.0]
    assert world["players"][0]["frames"][0]["floor_source"] == "track_footpoint+smpl_world"
    assert world["players"][0]["frames"][0]["foot_contact"] == {"left": True, "right": True}
    assert world["players"][0]["frames"][0]["contact_locked"] is True
    assert world["players"][0]["frames"][0]["physics"] == "worldhmr_grounded_not_footlocked"
    assert world["players"][0]["frames"][0]["floor_offset_m"] == pytest.approx(0.0)
    assert world["players"][0]["frames"][0]["min_mesh_z_m"] == pytest.approx(0.0)
    assert world["players"][0]["frames"][0]["floor_penetration_m"] == pytest.approx(0.0)
    assert world["ball"]["frames"][0]["world_xyz"] == [0.1, -0.8, 0.9]
    assert world["paddles"][0]["frames"][0]["pose_se3"]["t"] == [0.45, -1.9, 0.85]
    assert len(world["paddles"][0]["frames"][0]["mesh_vertices_world"]) == 16
    assert len(world["paddles"][0]["frames"][0]["mesh_faces"]) == 24
    first_face_vertex = world["paddles"][0]["frames"][0]["mesh_vertices_world"][0]
    first_back_vertex = world["paddles"][0]["frames"][0]["mesh_vertices_world"][4]
    assert first_face_vertex == pytest.approx([0.35475, -1.70315, 0.856985])
    assert first_back_vertex == pytest.approx([0.35475, -1.70315, 0.843015])
    assert world["summary"] == {
        "player_count": 1,
        "mesh_player_count": 1,
        "mesh_player_frame_count": 1,
        "joint_player_frame_count": 1,
        "track_only_player_frame_count": 0,
        "floor_placed_player_frame_count": 1,
        "floor_contact_player_frame_count": 1,
        "max_floor_penetration_m": 0.0,
        "max_abs_floor_offset_m": 0.0,
        "physics_modes": ["worldhmr_grounded_not_footlocked"],
        "ball_frame_count": 1,
        "approx_ball_frame_count": 0,
        "paddle_player_count": 1,
        "paddle_frame_count": 1,
        "ambiguous_paddle_frame_count": 0,
        "temporal_coverage": {
            "clip_min_t": 0.0,
            "clip_max_t": 0.0,
            "clip_frame_span": 0.016667,
            "ball": {
                "min_t": 0.0,
                "max_t": 0.0,
                "frame_span": 0.016667,
                "coverage_fraction": 1.0,
            },
            "players": [
                {
                    "player_id": 7,
                    "min_t": 0.0,
                    "max_t": 0.0,
                    "frame_span": 0.016667,
                    "coverage_fraction": 1.0,
                    "observed_frame_count": 1,
                    "no_data_frame_count": 0,
                }
            ],
        },
        "warnings": [],
    }


def test_build_virtual_world_state_warns_softly_when_only_embedded_mesh_vertices_are_missing(tmp_path: Path) -> None:
    run_dir = tmp_path / "clip"
    court_path = _write_json(run_dir / "court_calibration.json", _court_calibration())
    _write_json(
        run_dir / "body_mesh_index" / "body_mesh_index.json",
        {
            "artifact_type": "racketsport_body_mesh_index",
            "summary": {"mesh_frame_count": 3},
            "windows": [{"frame_start": 0, "frame_end": 2}],
        },
    )

    world = build_virtual_world_state(
        court_calibration=_court_calibration(),
        tracks=_tracks(),
        skeleton3d=_sam3d_skeleton3d_same_first_frame(),
        ball_track=_ball_track(),
        racket_pose=_racket_pose(),
        placement_calibration_path=court_path,
    )

    assert "missing_embedded_mesh_vertices" in world["summary"]["warnings"]
    assert "missing_mesh_vertices" not in world["summary"]["warnings"]


def test_build_virtual_world_state_keeps_strong_mesh_warning_when_embedded_and_index_are_absent() -> None:
    world = build_virtual_world_state(
        court_calibration=_court_calibration(),
        tracks=_tracks(),
        skeleton3d=_sam3d_skeleton3d_same_first_frame(),
        ball_track=_ball_track(),
        racket_pose=_racket_pose(),
    )

    assert "missing_mesh_vertices" in world["summary"]["warnings"]
    assert "missing_embedded_mesh_vertices" not in world["summary"]["warnings"]


def test_build_virtual_world_state_does_not_warn_when_embedded_mesh_vertices_exist(tmp_path: Path) -> None:
    run_dir = tmp_path / "clip"
    court_path = _write_json(run_dir / "court_calibration.json", _court_calibration())
    _write_json(
        run_dir / "body_mesh_index" / "body_mesh_index.json",
        {
            "artifact_type": "racketsport_body_mesh_index",
            "summary": {"mesh_frame_count": 3},
            "windows": [{"frame_start": 0, "frame_end": 2}],
        },
    )

    world = build_virtual_world_state(
        court_calibration=_court_calibration(),
        tracks=_tracks(),
        smpl_motion=_smpl_motion(),
        ball_track=_ball_track(),
        racket_pose=_racket_pose(),
        placement_calibration_path=court_path,
    )

    assert "missing_mesh_vertices" not in world["summary"]["warnings"]
    assert "missing_embedded_mesh_vertices" not in world["summary"]["warnings"]


def test_build_virtual_world_state_warns_on_ambiguous_paddle_preview() -> None:
    world = build_virtual_world_state(
        court_calibration=_court_calibration(),
        tracks=_tracks(),
        smpl_motion=_smpl_motion(),
        ball_track=_ball_track(),
        racket_pose=_ambiguous_racket_pose(),
    )

    assert world["summary"]["paddle_frame_count"] == 1
    assert world["summary"]["ambiguous_paddle_frame_count"] == 1
    assert world["paddles"][0]["frames"][0]["ambiguous"] is True
    assert "ambiguous_paddle_pose" in world["summary"]["warnings"]


def test_build_virtual_world_state_suppresses_box_derived_paddle_output() -> None:
    racket_pose = _racket_pose()
    racket_pose["players"][0]["frames"][0]["source"] = "label_bbox:cvat_video:paddle:pnp_ippe"

    world = build_virtual_world_state(
        court_calibration=_court_calibration(),
        tracks=_tracks(),
        smpl_motion=_smpl_motion(),
        ball_track=_ball_track(),
        racket_pose=racket_pose,
    )

    assert world["paddles"] == []
    assert world["summary"]["paddle_player_count"] == 0
    assert world["summary"]["paddle_frame_count"] == 0
    assert "missing_paddle_pose" in world["summary"]["warnings"]
    assert "box_derived_paddle_pose_suppressed" in world["summary"]["warnings"]


def test_build_virtual_world_state_marks_wrist_proxy_as_estimated_render_only_paddle() -> None:
    world = build_virtual_world_state(
        court_calibration=_court_calibration(),
        tracks=_tracks(),
        smpl_motion=_smpl_motion(),
        ball_track=_ball_track(),
        racket_pose_estimate=_wrist_proxy_pose_estimate(),
    )

    frame = world["paddles"][0]["frames"][0]
    assert frame["source"].startswith("wrist_proxy")
    assert frame["render_only"] is True
    assert frame["trust_band"]["stage"] == "RKT"
    assert frame["trust_band"]["gate_id"] == "wrist_proxy_estimated_paddle"
    assert frame["trust_band"]["gate_status"] == "estimated_from_wrist"
    assert frame["trust_band"]["badge"] == "low_confidence"
    assert world["paddles"][0]["trust_band"]["gate_status"] == "estimated_from_wrist"
    assert world["summary"]["paddle_player_count"] == 1
    assert world["summary"]["paddle_frame_count"] == 1
    assert world["summary"]["hidden_paddle_frame_count"] == 3
    assert world["summary"]["paddle_source"] == "wrist_proxy_estimated"
    assert "missing_paddle_pose" not in world["summary"]["warnings"]


def test_build_virtual_world_state_outputs_solid_paddle_face_and_handle_mesh() -> None:
    world = build_virtual_world_state(
        court_calibration=_court_calibration(),
        tracks=_tracks(),
        smpl_motion=_smpl_motion(),
        ball_track=_ball_track(),
        racket_pose=_racket_pose(),
    )

    frame = world["paddles"][0]["frames"][0]
    assert len(frame["mesh_vertices_world"]) == 16
    assert len(frame["mesh_faces"]) == 24
    front_face = frame["mesh_vertices_world"][0:4]
    back_face = frame["mesh_vertices_world"][4:8]
    handle_front = frame["mesh_vertices_world"][8:12]
    handle_back = frame["mesh_vertices_world"][12:16]
    assert front_face[0] == pytest.approx([0.35475, -1.70315, 0.856985])
    assert front_face[2] == pytest.approx([0.54525, -2.09685, 0.856985])
    assert back_face[0] == pytest.approx([0.35475, -1.70315, 0.843015])
    assert back_face[2] == pytest.approx([0.54525, -2.09685, 0.843015])
    assert handle_front[0] == pytest.approx([0.434125, -2.09685, 0.856985])
    assert handle_front[2] == pytest.approx([0.465875, -2.2302, 0.856985])
    assert handle_back[0] == pytest.approx([0.434125, -2.09685, 0.843015])
    assert handle_back[2] == pytest.approx([0.465875, -2.2302, 0.843015])


def test_build_virtual_world_state_hides_visible_2d_only_ball_in_replay_mode() -> None:
    world = build_virtual_world_state(
        court_calibration=_court_calibration(),
        tracks=_tracks(),
        ball_track=_ball_track_without_world_xyz(),
    )

    ball_frame = world["ball"]["frames"][0]
    assert ball_frame["visible"] is True
    assert ball_frame["world_xyz"] is None
    assert ball_frame["approx"] is False
    assert world["summary"]["approx_ball_frame_count"] == 0
    assert "unprojected_visible_ball_frames" in world["summary"]["warnings"]


def test_build_virtual_world_state_projects_visible_2d_ball_to_court_plane_only_in_review_mode() -> None:
    world = build_virtual_world_state(
        court_calibration=_court_calibration(),
        tracks=_tracks(),
        ball_track=_ball_track_without_world_xyz(),
        ball_world_policy="court_plane_approx_for_review_only",
    )

    ball_frame = world["ball"]["frames"][0]
    assert ball_frame["world_xyz"] == pytest.approx([1.0, 2.0, 0.0])
    assert ball_frame["approx"] is True
    assert ball_frame["render_only"] is True
    assert ball_frame["not_for_detection_metrics"] is True
    assert world["summary"]["approx_ball_frame_count"] == 1
    assert "unprojected_visible_ball_frames" not in world["summary"]["warnings"]


def test_build_virtual_world_state_rejects_off_court_2d_ball_projection() -> None:
    ball_track = _ball_track_without_world_xyz()
    ball_track["frames"][0]["xy"] = [10000.0, 10000.0]

    world = build_virtual_world_state(
        court_calibration=_court_calibration(),
        tracks=_tracks(),
        ball_track=ball_track,
    )

    ball_frame = world["ball"]["frames"][0]
    assert ball_frame["visible"] is True
    assert ball_frame["world_xyz"] is None
    assert ball_frame["approx"] is False
    assert world["summary"]["approx_ball_frame_count"] == 0
    assert "unprojected_visible_ball_frames" in world["summary"]["warnings"]


def test_apply_ball_track_arc_solved_overlay_prefers_arc_and_hides_uncovered_frames() -> None:
    merged = apply_ball_track_arc_solved_overlay(_ball_track_physics_filled(), _ball_track_arc_solved())

    assert [frame["world_xyz"] for frame in merged["frames"]] == [
        [0.1, -0.8, 0.9],
        None,
        [0.5, -0.5, 0.3],
    ]
    # xy/conf/visible/trust_band are a 2D-detection concern this overlay does not own.
    assert [frame["visible"] for frame in merged["frames"]] == [True, False, True]
    assert [frame["xy"] for frame in merged["frames"]] == [[320.0, 240.0], [321.0, 241.0], [322.0, 242.0]]
    assert merged["arc_solved_overlay"] == {
        "applied": True,
        "source_artifact_type": "racketsport_ball_track_arc_solved",
        "overlaid_frame_count": 2,
        "forced_hidden_frame_count": 1,
    }


def test_apply_ball_track_arc_solved_overlay_is_a_noop_without_an_arc_artifact() -> None:
    physics_filled = _ball_track_physics_filled()
    assert apply_ball_track_arc_solved_overlay(physics_filled, None) == physics_filled
    assert apply_ball_track_arc_solved_overlay(physics_filled, {"frames": "not-a-list"}) == physics_filled
    assert apply_ball_track_arc_solved_overlay(None, _ball_track_arc_solved()) is None


def test_apply_ball_track_arc_solved_overlay_ignores_self_killed_artifacts() -> None:
    physics_filled = _ball_track_physics_filled()
    for status in ("experimental_off", "degenerate_zero_segments"):
        arc_solved = _ball_track_arc_solved()
        arc_solved["status"] = status
        arc_solved["kill_reasons"] = ["solver_self_killed"]

        assert apply_ball_track_arc_solved_overlay(physics_filled, arc_solved) == physics_filled


def test_build_virtual_world_state_excludes_self_killed_ball_arc_overlay() -> None:
    arc_solved = _ball_track_arc_solved()
    arc_solved["status"] = "experimental_off"
    arc_solved["kill_reasons"] = ["physical_sanity_gate_failed"]

    world = build_virtual_world_state(
        court_calibration=_court_calibration(),
        tracks=_tracks(),
        ball_track_physics_filled=_ball_track_physics_filled(),
        ball_track_arc_solved=arc_solved,
    )

    assert [frame["world_xyz"] for frame in world["ball"]["frames"]] == [
        [0.1, -0.8, 0.9],
        [0.2, -0.7, 0.05],
        [0.22, -0.69, 0.42],
    ]
    assert "arc_solved_overlay" not in world["ball"]


def test_build_virtual_world_state_arc_solved_overlay_wins_over_physics_filled_world_xyz() -> None:
    world = build_virtual_world_state(
        court_calibration=_court_calibration(),
        tracks=_tracks(),
        ball_track_physics_filled=_ball_track_physics_filled(),
        ball_track_arc_solved=_ball_track_arc_solved(),
    )

    # Frame 1: the arc solver has no confident coverage here. The raw
    # physics-filled artifact still carries a world_xyz for it -- that value
    # must never reach the rendered world, or a non-analytic sighting has
    # leaked into the ball trail exactly like the owner's bug report.
    assert world["ball"]["frames"][1]["world_xyz"] is None
    # Frame 2: the arc solver's analytic position wins over physics-filled's.
    assert world["ball"]["frames"][2]["world_xyz"] == [0.5, -0.5, 0.3]
    # Frame 0: both agree; still sourced through the overlay correctly.
    assert world["ball"]["frames"][0]["world_xyz"] == [0.1, -0.8, 0.9]


def test_build_virtual_world_state_arc_solved_overlay_renders_kink_free_despite_deviating_raw_sighting() -> None:
    """Regression test for the owner's mid-air-direction-change report.

    `ball_track_physics_filled` alone (as it would be built by an older,
    partial hand-composition step) has a wildly deviating mid-flight
    sighting at frame 2. Without the arc-solved overlay this reverses the
    ball's horizontal velocity twice -- a visible kink. With the overlay
    applied, the rendered world stream must be exactly the arc's monotonic,
    kink-free flight.
    """

    physics_filled = _ball_track_physics_filled_with_deviating_mid_flight_sighting()
    arc_solved = _ball_track_arc_solved_monotonic_flight()

    raw_points = [frame["world_xyz"] for frame in physics_filled["frames"]]
    assert _horizontal_velocity_sign_changes(raw_points) == 4, "fixture must actually kink before the fix"

    world = build_virtual_world_state(
        court_calibration=_court_calibration(),
        tracks=_tracks(),
        ball_track_physics_filled=physics_filled,
        ball_track_arc_solved=arc_solved,
    )

    rendered_points = [frame["world_xyz"] for frame in world["ball"]["frames"]]
    assert rendered_points == [frame["world_xyz"] for frame in arc_solved["frames"]]
    assert _horizontal_velocity_sign_changes(rendered_points) == 0


def test_build_virtual_world_state_from_run_dir_consumes_ball_track_arc_solved(tmp_path: Path) -> None:
    run_dir = tmp_path / "clip_v1_frames_metric15"
    _write_json(run_dir / "court_calibration_metric15pt.json", _metric15_court_calibration())
    _write_json(run_dir / "ball_track_physics_filled.json", _ball_track_physics_filled())
    _write_json(run_dir / "ball_track_arc_solved.json", _ball_track_arc_solved())

    world = build_virtual_world_state_from_run_dir(run_dir)

    assert [frame["world_xyz"] for frame in world["ball"]["frames"]] == [
        [0.1, -0.8, 0.9],
        None,
        [0.5, -0.5, 0.3],
    ]


def test_virtual_world_cli_consumes_ball_track_arc_solved_from_run_dir(tmp_path: Path) -> None:
    run_dir = tmp_path / "clip_v1_frames_metric15"
    _write_json(run_dir / "court_calibration.json", _court_calibration())
    _write_json(run_dir / "court_calibration_metric15pt.json", _metric15_court_calibration())
    _write_json(run_dir / "ball_track_physics_filled.json", _ball_track_physics_filled())
    _write_json(run_dir / "ball_track_arc_solved.json", _ball_track_arc_solved())
    out = run_dir / "virtual_world.json"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/build_virtual_world.py",
            "--run-dir",
            str(run_dir),
            "--out",
            str(out),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert completed.stderr == ""
    emitted = json.loads(out.read_text(encoding="utf-8"))
    assert [frame["world_xyz"] for frame in emitted["ball"]["frames"]] == [
        [0.1, -0.8, 0.9],
        None,
        [0.5, -0.5, 0.3],
    ]


def test_build_virtual_world_state_preserves_track_only_frames_outside_sparse_mesh() -> None:
    world = build_virtual_world_state(
        court_calibration=_court_calibration(),
        tracks=_tracks_two_frames(),
        smpl_motion=_smpl_motion(),
    )

    player = world["players"][0]
    assert player["representation"] == "mesh"
    assert [frame["t"] for frame in player["frames"]] == [0.0, pytest.approx(1.0 / 60.0)]
    assert player["frames"][0]["mesh_vertex_count"] == 2
    assert player["frames"][1]["track_world_xy"] == [0.3, -1.95]
    assert player["frames"][1]["floor_world_xyz"] == [0.3, -1.95, 0.0]
    assert player["frames"][1]["floor_source"] == "track_footpoint"
    assert player["frames"][1]["foot_contact"] is None
    assert player["frames"][1]["contact_locked"] is False
    assert player["frames"][1]["transl_world"] is None
    assert player["frames"][1]["joints_world"] == []
    assert player["frames"][1]["mesh_vertex_count"] == 0
    assert world["summary"]["mesh_player_frame_count"] == 1
    assert world["summary"]["joint_player_frame_count"] == 1
    assert world["summary"]["track_only_player_frame_count"] == 1


def test_build_virtual_world_state_keeps_skeleton_joints_outside_sparse_mesh_frames() -> None:
    world = build_virtual_world_state(
        court_calibration=_court_calibration(),
        tracks=_tracks_two_frames(),
        smpl_motion=_smpl_motion(),
        skeleton3d=_skeleton3d_second_frame(),
    )

    player = world["players"][0]
    assert player["representation"] == "mesh"
    assert [frame["t"] for frame in player["frames"]] == [0.0, pytest.approx(1.0 / 60.0)]

    mesh_frame = player["frames"][0]
    assert mesh_frame["mesh_vertex_count"] == 2
    assert mesh_frame["joints_world"] == [[0.25, -2.0, 0.0], [0.25, -2.0, 1.6]]
    assert mesh_frame["mesh_ref"] == {
        "artifact": "body_mesh.json",
        "player_id": 7,
        "frame_idx": 0,
        "t": 0.0,
    }

    skeleton_frame = player["frames"][1]
    assert skeleton_frame["mesh_vertex_count"] == 0
    assert skeleton_frame["mesh_ref"] is None
    assert skeleton_frame["joints_world"] == [[0.31, -1.95, 1.55], [0.32, -1.96, 1.35]]
    assert skeleton_frame["joint_conf"] == [0.73, 0.71]
    assert skeleton_frame["joint_count"] == 2
    assert skeleton_frame["floor_source"] == "track_footpoint+skeleton_world"
    assert world["summary"]["mesh_player_frame_count"] == 1
    assert world["summary"]["joint_player_frame_count"] == 2
    assert world["summary"]["track_only_player_frame_count"] == 0


def test_build_virtual_world_state_prefers_skeleton3d_joints_over_same_timestamp_smpl() -> None:
    smpl = _smpl_motion()
    smpl["players"][0]["frames"][0]["transl_world"] = [9.0, 9.0, 0.0]
    smpl_fill_frame = deepcopy(smpl["players"][0]["frames"][0])
    smpl_fill_frame["t"] = 1.0 / 60.0
    smpl_fill_frame["frame_idx"] = 1
    smpl_fill_frame["joints_world"] = [[9.0, 9.0, 9.0], [10.0, 10.0, 10.0]]
    smpl_fill_frame["joint_conf"] = [0.2, 0.2]
    smpl["players"][0]["frames"].append(smpl_fill_frame)
    skeleton = _sam3d_skeleton3d_same_first_frame()
    skeleton["players"][0]["frames"][0]["transl_world"] = [8.0, 8.0, 0.0]

    world = build_virtual_world_state(
        court_calibration=_court_calibration(),
        tracks=_tracks_two_frames(),
        smpl_motion=smpl,
        skeleton3d=skeleton,
    )

    assert world["joint_names"] == list(MHR70_JOINT_NAMES)
    player = world["players"][0]
    assert player["joints_source"] == {"skeleton3d": 1, "smpl_fill": 1}
    assert [frame["t"] for frame in player["frames"]] == [0.0, pytest.approx(1.0 / 60.0)]

    same_timestamp = player["frames"][0]
    assert same_timestamp["joints_world"] == skeleton["players"][0]["frames"][0]["joints_world"]
    assert same_timestamp["joint_conf"] == skeleton["players"][0]["frames"][0]["joint_conf"]
    assert same_timestamp["joint_count"] == 70
    assert same_timestamp["mesh_vertex_count"] == 2
    assert same_timestamp["mesh_ref"] == {
        "artifact": "body_mesh.json",
        "player_id": 7,
        "frame_idx": 0,
        "t": 0.0,
    }
    assert same_timestamp["transl_world"] == [8.0, 8.0, 0.0]
    assert same_timestamp["floor_world_xyz"][:2] == same_timestamp["track_world_xy"]
    assert same_timestamp["joints_world"] != _smpl_motion()["players"][0]["frames"][0]["joints_world"]

    smpl_fill = player["frames"][1]
    assert smpl_fill["joints_world"] == [[9.0, 9.0, 9.0], [10.0, 10.0, 10.0]]
    assert smpl_fill["joint_count"] == 2


def test_build_virtual_world_state_rejects_nonfinite_track_world_xy() -> None:
    tracks = _tracks()
    tracks["players"][0]["frames"][0]["world_xy"] = [float("nan"), -2.0]

    with pytest.raises(ValueError, match="finite number"):
        build_virtual_world_state(court_calibration=_court_calibration(), tracks=tracks)


def test_run_dir_rebuild_strips_legacy_skeleton3d_foot_pin_extra(tmp_path: Path) -> None:
    run_dir = tmp_path / "legacy_run"
    _write_json(run_dir / "court_calibration.json", _court_calibration())
    _write_json(run_dir / "tracks.json", _tracks())
    skeleton = _sam3d_skeleton3d_same_first_frame()
    skeleton["foot_pin"] = {
        "version": 1,
        "audit": {
            "artifact_type": "racketsport_foot_pin_audit",
            "summary": {"total_corrected_frame_count": 0},
        },
    }
    _write_json(run_dir / "skeleton3d.json", skeleton)

    world = build_virtual_world_state_from_run_dir(run_dir)

    assert world["players"][0]["frames"][0]["joint_count"] == 70
    assert world["players"][0]["frames"][0]["transl_world"] == [0.25, -2.0, 0.0]


def test_run_dir_rebuild_reuses_legacy_pre_r3_confidence_world_when_available(tmp_path: Path) -> None:
    run_dir = tmp_path / "legacy_run"
    _write_json(run_dir / "court_calibration.json", _court_calibration())
    _write_json(run_dir / "tracks.json", _tracks())
    skeleton = _sam3d_skeleton3d_same_first_frame()
    skeleton["foot_pin"] = {"version": 1, "audit": {"summary": {"total_corrected_frame_count": 12}}}
    _write_json(run_dir / "skeleton3d.json", skeleton)
    legacy_world = build_virtual_world_state(court_calibration=_court_calibration(), tracks=_tracks())
    legacy_world["confidence_gate"] = {"source": "legacy_pre_r3"}
    legacy_world["players"][0]["frames"][0]["transl_world"] = [9.0, 9.0, 0.0]
    _write_json(run_dir / "confidence_gated_world.json", legacy_world)

    world = build_virtual_world_state_from_run_dir(run_dir)

    assert world["confidence_gate"] == {"source": "legacy_pre_r3"}
    assert world["players"][0]["frames"][0]["transl_world"] == [0.25, -2.0, 0.0]
    assert any("legacy_pre_r3_world_reused" in warning for warning in world["summary"]["warnings"])


def test_build_virtual_world_state_reports_temporal_coverage_relative_to_clip_span() -> None:
    world = build_virtual_world_state(
        court_calibration=_court_calibration(),
        tracks=_tracks_ten_frame_clip(),
        ball_track=_ball_track_half_clip(),
    )

    coverage = world["summary"]["temporal_coverage"]
    assert coverage["clip_min_t"] == pytest.approx(0.0)
    assert coverage["clip_max_t"] == pytest.approx(0.9)
    assert coverage["clip_frame_span"] == pytest.approx(1.0)
    assert coverage["ball"] == {
        "min_t": 0.0,
        "max_t": 0.4,
        "frame_span": 0.5,
        "coverage_fraction": 0.5,
    }
    assert coverage["players"] == [
        {
            "player_id": 7,
            "min_t": 0.0,
            "max_t": 0.9,
            "frame_span": 1.0,
            "coverage_fraction": 1.0,
            "observed_frame_count": 10,
            "no_data_frame_count": 0,
        }
    ]


def test_build_virtual_world_state_adds_schema_compliant_no_data_player_frames() -> None:
    world = build_virtual_world_state(court_calibration=_court_calibration(), tracks=_tracks_ragged_two_players())

    players_by_id = {player["id"]: player for player in world["players"]}
    assert [frame["t"] for frame in players_by_id[7]["frames"]] == [0.0, 0.1, 0.2]
    assert [frame["t"] for frame in players_by_id[8]["frames"]] == [0.0, 0.1, 0.2]

    missing_frame = players_by_id[8]["frames"][1]
    assert missing_frame["track_world_xy"] is None
    assert missing_frame["bbox"] is None
    assert missing_frame["joints_world"] == []
    assert missing_frame["joint_count"] == 0
    assert missing_frame["mesh_vertex_count"] == 0
    assert missing_frame["floor_world_xyz"] is None
    assert missing_frame["trust_band"]["stage"] == "WORLD"
    assert missing_frame["trust_band"]["gate_id"] == "world_no_data_placeholder"
    assert missing_frame["trust_band"]["gate_status"] == "no_data"
    assert missing_frame["trust_band"]["badge"] == "low_confidence"
    assert world["summary"]["temporal_coverage"]["players"][1]["no_data_frame_count"] == 1


def _tracks_three_players() -> dict:
    payload = _tracks()
    payload["players"] = [
        {
            "id": 3,
            "side": "far",
            "role": "left",
            "frames": [{"t": 0.0, "bbox": [10.0, 10.0, 40.0, 240.0], "world_xy": [-1.0, 4.0], "conf": 0.9}],
        },
        {
            "id": 4,
            "side": "far",
            "role": "right",
            "frames": [{"t": 0.0, "bbox": [110.0, 10.0, 140.0, 240.0], "world_xy": [1.0, 4.0], "conf": 0.9}],
        },
        {
            "id": 7,
            "side": "near",
            "role": "left",
            "frames": [{"t": 0.0, "bbox": [100.0, 100.0, 140.0, 240.0], "world_xy": [0.25, -2.0], "conf": 0.91}],
        },
    ]
    return payload


def _skeleton3d_two_players() -> dict:
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_skeleton3d",
        "fps": 60.0,
        "world_frame": "court_Z0",
        "source_model": "sat_hmr_world_joints",
        "joint_names": ["nose", "left_shoulder"],
        "preview_only": True,
        "players": [
            {
                "id": 3,
                "frames": [{"frame_idx": 0, "t": 0.0, "joints_world": [[-1.0, 4.0, 1.6], [-1.1, 4.0, 1.4]], "joint_conf": [0.7, 0.65]}],
            },
            {
                "id": 4,
                "frames": [{"frame_idx": 0, "t": 0.0, "joints_world": [[1.0, 4.0, 1.6], [1.1, 4.0, 1.4]], "joint_conf": [0.8, 0.75]}],
            },
        ],
        "provenance": {},
    }


def test_build_virtual_world_state_combines_skeleton3d_with_full_clip_tracks() -> None:
    world = build_virtual_world_state(
        court_calibration=_court_calibration(),
        tracks=_tracks_three_players(),
        skeleton3d=_skeleton3d_two_players(),
    )

    players_by_id = {player["id"]: player for player in world["players"]}
    assert set(players_by_id) == {3, 4, 7}

    # Player 3 has both a track and BODY joints: floor comes from the track footpoint,
    # and the world joints are carried through.
    player3_frame = players_by_id[3]["frames"][0]
    assert players_by_id[3]["representation"] == "joints"
    assert player3_frame["joint_count"] == 2
    assert player3_frame["joints_world"][0] == [-1.0, 4.0, 1.6]
    assert player3_frame["track_world_xy"] == [-1.0, 4.0]
    assert player3_frame["floor_world_xyz"] == [-1.0, 4.0, 0.0]
    assert player3_frame["floor_source"] == "track_footpoint+skeleton_world"

    # Player 7 is tracked but has no BODY output for this clip: track-only fallback.
    player7 = players_by_id[7]
    assert player7["representation"] == "track_only"
    assert player7["frames"][0]["joint_count"] == 0
    assert player7["frames"][0]["floor_world_xyz"] == [0.25, -2.0, 0.0]


def test_build_virtual_world_state_skeleton_only_without_tracks_has_no_floor() -> None:
    world = build_virtual_world_state(
        court_calibration=_court_calibration(),
        skeleton3d=_skeleton3d_two_players(),
    )

    players_by_id = {player["id"]: player for player in world["players"]}
    assert set(players_by_id) == {3, 4}
    assert players_by_id[3]["frames"][0]["floor_world_xyz"] is None
    assert players_by_id[3]["frames"][0]["track_world_xy"] is None


def test_build_virtual_world_state_wires_trust_bands_onto_matching_entities() -> None:
    body_band = {
        "stage": "BODY",
        "gate_id": "body_full_clip_gate+body_review_overlay_alignment",
        "gate_status": "structural_pass_accuracy_unmeasured",
        "badge": "preview",
        "reason": "structural pass, calibration offset under review",
        "evidence_path": "runs/x/body_gate_report.json",
    }
    track_band = {
        "stage": "TRK",
        "gate_id": "trk_idf1_gate",
        "gate_status": "do_not_promote",
        "badge": "low_confidence",
        "reason": "IDF1 below gate",
        "evidence_path": "runs/x/person_track_gt_score.json",
    }
    court_band = {
        "stage": "CAL",
        "gate_id": "court_calibration_pck5px_gate",
        "gate_status": "manual_sidecar_unverified",
        "badge": "preview",
        "reason": "manual corners",
        "evidence_path": "runs/x/court_calibration.json",
    }
    ball_band = {
        "stage": "BALL",
        "gate_id": "ball_m1_f1_at_20_gate",
        "gate_status": "0/8 milestones pass",
        "badge": "low_confidence",
        "reason": "zero-shot only",
        "evidence_path": "runs/x/ball_track.json",
    }
    paddle_band = {
        "stage": "RKT",
        "gate_id": "racket_face_angle_p90_gate",
        "gate_status": "unscoreable_no_gt",
        "badge": "low_confidence",
        "reason": "box-derived only",
        "evidence_path": "runs/x/racket_promotion_audit.json",
    }

    world = build_virtual_world_state(
        court_calibration=_court_calibration(),
        tracks=_tracks_three_players(),
        skeleton3d=_skeleton3d_two_players(),
        ball_track=_ball_track(),
        racket_pose=_racket_pose(),
        trust_bands={"body": body_band, "track": track_band, "court": court_band, "ball": ball_band, "paddle": paddle_band},
    )

    players_by_id = {player["id"]: player for player in world["players"]}
    assert players_by_id[3]["trust_band"]["badge"] == "preview"
    assert players_by_id[3]["trust_band"]["stage"] == "BODY"
    assert players_by_id[7]["trust_band"]["badge"] == "low_confidence"
    assert players_by_id[7]["trust_band"]["stage"] == "TRK"
    assert world["court"]["trust_band"]["badge"] == "preview"
    assert world["ball"]["trust_band"]["badge"] == "low_confidence"
    assert world["paddles"][0]["trust_band"]["badge"] == "low_confidence"


def test_build_virtual_world_state_without_trust_bands_leaves_them_null() -> None:
    world = build_virtual_world_state(court_calibration=_court_calibration(), tracks=_tracks())
    assert world["court"]["trust_band"] is None
    assert world["players"][0]["trust_band"] is None


def test_build_virtual_world_state_from_run_dir_prefers_metric15_and_consumes_physics_artifacts(tmp_path: Path) -> None:
    run_dir = tmp_path / "clip_v1_frames_metric15"
    _write_json(run_dir / "court_calibration.json", _court_calibration())
    _write_json(run_dir / "court_calibration_metric15pt.json", _metric15_court_calibration())
    _write_json(run_dir / "tracks.json", _tracks())
    _write_json(run_dir / "ball_track.json", _ball_track())
    _write_json(run_dir / "racket_pose.json", _racket_pose())
    _write_json(run_dir / "physics_footlock.json", _physics_footlock())
    _write_json(run_dir / "ball_track_physics_filled.json", _ball_track_physics_filled())
    _write_json(run_dir / "racket_pose_estimate.json", _racket_pose_estimate())

    world = build_virtual_world_state_from_run_dir(run_dir)

    placement = world["court"]["placement_calibration"]
    assert placement["source"] == "metric_15pt_reviewed"
    assert placement["intrinsics_source"] == "metric_15pt_reviewed"
    assert placement["capture_quality_grade"] == "warn"
    assert placement["metric_confidence"] == "high"
    assert placement["evidence_path"].endswith("court_calibration_metric15pt.json")

    player_frame = world["players"][0]["frames"][0]
    # Locked stance samples are consumed exactly in X/Y and snapped to court Z=0,
    # not smoothed back toward the raw track footpoint.
    assert player_frame["floor_world_xyz"] == [0.11, -2.22, 0.0]
    assert player_frame["floor_source"] == "physics_footlock_corrected"
    assert player_frame["foot_contact"] == {"left": True, "right": False}
    assert player_frame["contact_locked"] is True
    assert player_frame["physics"] == "physics_footlock"
    assert player_frame["grf"] == [[0.0, 0.0, 1.0]]
    assert player_frame["trust_band"]["stage"] == "PHYS-FOOT"
    assert player_frame["trust_band"]["gate_status"] == "corrected"

    # The physics-filled ball track is used as-authored. The interpolated sample
    # at the bounce time is not temporally smoothed across the bounce.
    assert world["ball"]["source"] == "physics_filled"
    assert [frame["world_xyz"] for frame in world["ball"]["frames"]] == [
        [0.1, -0.8, 0.9],
        [0.2, -0.7, 0.05],
        [0.22, -0.69, 0.42],
    ]
    assert [frame["trust_band"]["gate_status"] for frame in world["ball"]["frames"]] == [
        "corrected",
        "interpolated",
        "physics_derived",
    ]

    assert world["paddles"][0]["frames"][0]["source"].startswith("physics_derived")
    assert world["paddles"][0]["frames"][0]["trust_band"]["stage"] == "PHYS-RACKET"
    assert world["paddles"][0]["frames"][0]["trust_band"]["gate_status"] == "physics_derived"


def test_build_virtual_world_state_from_run_dir_consumes_actual_footlock_schema(tmp_path: Path) -> None:
    run_dir = tmp_path / "clip_v1_frames_metric15"
    skeleton = {
        "schema_version": 1,
        "artifact_type": "racketsport_skeleton3d",
        "fps": 60.0,
        "world_frame": "court_Z0",
        "source_model": "sat_hmr_world_joints",
        "joint_names": ["nose", "left_shoulder"],
        "preview_only": True,
        "players": [
            {
                "id": 7,
                "frames": [
                    {
                        "frame_idx": 0,
                        "t": 0.0,
                        "joints_world": [[0.25, -2.0, 1.6], [0.25, -2.0, 1.4]],
                        "joint_conf": [0.7, 0.65],
                    }
                ],
            }
        ],
        "provenance": {},
    }
    _write_json(run_dir / "court_calibration_metric15pt.json", _metric15_court_calibration())
    _write_json(run_dir / "tracks.json", _tracks())
    _write_json(run_dir / "skeleton3d.json", skeleton)
    _write_json(run_dir / "physics_footlock.json", _physics_footlock_actual_schema())

    world = build_virtual_world_state_from_run_dir(run_dir)

    frame = world["players"][0]["frames"][0]
    assert frame["joints_world"] == [[0.31, -2.01, 1.55], [0.32, -2.02, 1.35]]
    assert frame["joint_conf"] == [0.93, 0.91]
    assert frame["foot_contact"] == {"left": True, "right": False}
    assert frame["contact_locked"] is True
    assert frame["physics"] == "physics_footlock"
    assert frame["trust_band"]["gate_status"] == "corrected"


def test_build_virtual_world_state_from_run_dir_maps_physics_interpolated_ball_source(tmp_path: Path) -> None:
    run_dir = tmp_path / "clip_v1_frames_metric15"
    _write_json(run_dir / "court_calibration_metric15pt.json", _metric15_court_calibration())
    filled = _ball_track_physics_filled()
    filled["frames"][1]["source"] = "physics_interpolated"
    filled["frames"][1]["trust_band"] = {
        "stage": "PHYS-BALLFILL",
        "gate_id": "physics_render_continuity_only",
        "gate_status": "not_a_ball_detection_gate",
        "badge": "low_confidence",
        "reason": "render continuity only",
        "evidence_path": "ball_track_physics_filled.json",
    }
    _write_json(run_dir / "ball_track_physics_filled.json", filled)

    world = build_virtual_world_state_from_run_dir(run_dir)

    assert world["ball"]["frames"][1]["trust_band"]["gate_status"] == "interpolated"
    assert world["ball"]["frames"][1]["trust_band"]["badge"] == "preview"


def test_build_virtual_world_state_from_run_dir_refreshes_court_trust_from_selected_calibration(tmp_path: Path) -> None:
    run_dir = tmp_path / "clip_v1_frames_metric15"
    _write_json(run_dir / "court_calibration_metric15pt.json", _metric15_court_calibration())
    _write_json(run_dir / "tracks.json", _tracks())
    _write_json(
        run_dir / "trust_bands.json",
        {
            "court": {
                "stage": "CAL",
                "gate_id": "court_calibration_pck5px_gate",
                "gate_status": "manual_sidecar_unverified",
                "badge": "preview",
                "reason": "stale manual corner sidecar wording",
                "evidence_path": "old/court_calibration.json",
            }
        },
    )

    world = build_virtual_world_state_from_run_dir(run_dir)

    assert world["court"]["trust_band"]["gate_status"] == "metric15_unverified"
    assert "metric-15pt reviewed calibration" in world["court"]["trust_band"]["reason"]
    assert "manual corner sidecar" not in world["court"]["trust_band"]["reason"]


def test_virtual_world_cli_writes_coverage_and_only_needs_summary_schema_diff(tmp_path: Path) -> None:
    run_dir = tmp_path / "clip_001"
    court = _write_json(run_dir / "court_calibration.json", _court_calibration())
    tracks = _write_json(run_dir / "tracks.json", _tracks())
    smpl = _write_json(run_dir / "smpl_motion.json", _smpl_motion())
    ball = _write_json(run_dir / "ball_track.json", _ball_track())
    racket = _write_json(run_dir / "racket_pose.json", _racket_pose())
    out = run_dir / "virtual_world.json"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/build_virtual_world.py",
            "--court-calibration",
            str(court),
            "--tracks",
            str(tracks),
            "--smpl-motion",
            str(smpl),
            "--ball-track",
            str(ball),
            "--racket-pose",
            str(racket),
            "--out",
            str(out),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert completed.stderr == ""
    emitted = json.loads(out.read_text(encoding="utf-8"))
    coverage = emitted["summary"].pop("temporal_coverage")
    schema_compatible = _write_json(run_dir / "virtual_world_schema_compatible.json", emitted)
    parsed = validate_artifact_file("virtual_world", schema_compatible)
    assert isinstance(parsed, VirtualWorld)
    assert parsed.summary.mesh_player_count == 1
    assert parsed.summary.mesh_player_frame_count == 1
    assert coverage["ball"]["coverage_fraction"] == 1.0
    assert coverage["players"][0]["player_id"] == 7
    assert json.loads(completed.stdout)["summary"]["paddle_frame_count"] == 1


def test_build_virtual_world_state_from_run_dir_refuses_protected_eval_clips_by_default(tmp_path: Path) -> None:
    strict_dir = tmp_path / "outdoor_webcam_iynbd_1500_long_high_baseline"
    _write_json(strict_dir / "court_calibration.json", _court_calibration())

    with pytest.raises(EvalClipLeakError, match="strict held-out eval clip"):
        build_virtual_world_state_from_run_dir(strict_dir)

    internal_dir = tmp_path / "wolverine_mixed_0200_mid_steep_corner"
    _write_json(internal_dir / "court_calibration.json", _court_calibration())

    with pytest.raises(EvalClipLeakError, match="allow_internal_val=True"):
        build_virtual_world_state_from_run_dir(internal_dir)

    world = build_virtual_world_state_from_run_dir(internal_dir, allow_internal_val=True)
    assert world["artifact_type"] == "racketsport_virtual_world"


def test_virtual_world_cli_can_build_from_run_dir_with_best_calibration_and_physics(tmp_path: Path) -> None:
    run_dir = tmp_path / "clip_v1_frames_metric15"
    _write_json(run_dir / "court_calibration.json", _court_calibration())
    _write_json(run_dir / "court_calibration_metric15pt.json", _metric15_court_calibration())
    _write_json(run_dir / "tracks.json", _tracks())
    _write_json(run_dir / "physics_footlock.json", _physics_footlock())
    _write_json(run_dir / "ball_track_physics_filled.json", _ball_track_physics_filled())
    _write_json(run_dir / "racket_pose_estimate.json", _racket_pose_estimate())
    out = run_dir / "virtual_world.json"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/build_virtual_world.py",
            "--run-dir",
            str(run_dir),
            "--out",
            str(out),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert completed.stderr == ""
    parsed = json.loads(out.read_text(encoding="utf-8"))
    assert parsed["court"]["placement_calibration"]["source"] == "metric_15pt_reviewed"
    assert parsed["players"][0]["frames"][0]["trust_band"]["gate_status"] == "corrected"
    assert parsed["ball"]["frames"][1]["trust_band"]["gate_status"] == "interpolated"
    assert parsed["paddles"][0]["frames"][0]["trust_band"]["gate_status"] == "physics_derived"


def test_virtual_world_cli_run_dir_refuses_eval_clip_without_internal_val_flag(tmp_path: Path) -> None:
    run_dir = tmp_path / "wolverine_mixed_0200_mid_steep_corner"
    _write_json(run_dir / "court_calibration.json", _court_calibration())
    out = run_dir / "virtual_world.json"

    blocked = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/build_virtual_world.py",
            "--run-dir",
            str(run_dir),
            "--out",
            str(out),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert blocked.returncode == 1
    assert "allow_internal_val=True" in blocked.stderr
    assert not out.exists()

    allowed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/build_virtual_world.py",
            "--run-dir",
            str(run_dir),
            "--allow-internal-val-run-dir",
            "--out",
            str(out),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert allowed.returncode == 0
    assert out.exists()


def _multi_player_tracks_for_membership() -> dict:
    payload = _tracks()
    payload["players"].append(
        {
            "id": 8,
            "side": "far",
            "role": "right",
            "frames": [{"t": 0.0, "bbox": [200.0, 100.0, 240.0, 240.0], "world_xy": [1.25, 2.0], "conf": 0.88}],
        }
    )
    payload["players"].append(
        {
            "id": 9,
            "side": "far",
            "role": "left",
            "frames": [{"t": 0.0, "bbox": [260.0, 100.0, 300.0, 240.0], "world_xy": [1.8, 2.0], "conf": 0.82}],
        }
    )
    return payload


def _multi_player_smpl_for_membership() -> dict:
    payload = _smpl_motion()
    player_8 = deepcopy(payload["players"][0])
    player_8["id"] = 8
    player_8["frames"][0]["transl_world"] = [1.25, 2.0, 0.0]
    player_8["frames"][0]["joints_world"] = [[1.25, 2.0, 0.0], [1.25, 2.0, 1.6]]
    player_8["frames"][0]["mesh_vertices_world"] = [[1.2, 1.95, 0.0], [1.3, 2.05, 1.7]]
    player_9 = deepcopy(payload["players"][0])
    player_9["id"] = 9
    player_9["frames"][0]["transl_world"] = [1.8, 2.0, 0.0]
    player_9["frames"][0]["joints_world"] = [[1.8, 2.0, 0.0], [1.8, 2.0, 1.6]]
    player_9["frames"][0]["mesh_vertices_world"] = [[1.75, 1.95, 0.0], [1.85, 2.05, 1.7]]
    payload["players"].extend([player_8, player_9])
    return payload


def _membership_payload_for_virtual_world() -> dict:
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_player_court_membership",
        "verified": False,
        "not_gate_verified": True,
        "per_player": {
            "7": {"verdict": "on_target_court", "reasons": []},
            "8": {"verdict": "adjacent_or_spectator", "reasons": ["inside_strict_frac_below_threshold"]},
            "9": {"verdict": "uncertain", "reasons": ["too_few_frames_for_on_target"]},
        },
    }


def test_build_virtual_world_state_excludes_membership_adjacent_players_and_records_preview_provenance(
    tmp_path: Path,
) -> None:
    membership_path = _write_json(tmp_path / "membership.json", _membership_payload_for_virtual_world())
    trust_bands: dict[str, dict | None] = {
        "track": {
            "stage": "TRK",
            "gate_id": "track_preview",
            "gate_status": "preview",
            "badge": "preview",
            "reason": "synthetic",
            "evidence_path": None,
        }
    }

    world = build_virtual_world_state(
        court_calibration=_court_calibration(),
        tracks=_multi_player_tracks_for_membership(),
        smpl_motion=_multi_player_smpl_for_membership(),
        trust_bands=trust_bands,
        membership_path=membership_path,
    )

    assert [player["id"] for player in world["players"]] == [7, 9]
    assert world["summary"]["player_count"] == 2
    assert "player_membership_preview_not_verified" in world["summary"]["warnings"]
    assert "player_membership_excluded_count_1" in world["summary"]["warnings"]
    assert "player_membership_excluded_ids_8" in world["summary"]["warnings"]
    assert "player_membership_uncertain_ids_9" in world["summary"]["warnings"]
    assert trust_bands["player_membership"]["gate_status"] == "membership_preview_not_verified"
    assert trust_bands["player_membership"]["excluded_players"] == [
        {"id": 8, "verdict": "adjacent_or_spectator", "reasons": ["inside_strict_frac_below_threshold"]}
    ]
    assert trust_bands["player_membership"]["uncertain_players"] == [
        {"id": 9, "verdict": "uncertain", "reasons": ["too_few_frames_for_on_target"]}
    ]


def test_build_virtual_world_state_without_membership_path_is_byte_identical() -> None:
    kwargs = {
        "court_calibration": _court_calibration(),
        "tracks": _multi_player_tracks_for_membership(),
        "smpl_motion": _multi_player_smpl_for_membership(),
    }

    before = build_virtual_world_state(**kwargs)
    after = build_virtual_world_state(**kwargs, membership_path=None)

    assert json.dumps(after, sort_keys=True) == json.dumps(before, sort_keys=True)


def test_build_virtual_world_state_from_files_auto_discovers_membership_next_to_tracks(tmp_path: Path) -> None:
    from threed.racketsport.virtual_world import build_virtual_world_state_from_files

    court_path = _write_json(tmp_path / "court_calibration.json", _court_calibration())
    tracks_path = _write_json(tmp_path / "tracks.json", _multi_player_tracks_for_membership())
    smpl_path = _write_json(tmp_path / "smpl_motion.json", _multi_player_smpl_for_membership())
    _write_json(tmp_path / "membership.json", _membership_payload_for_virtual_world())
    trust_bands: dict[str, dict | None] = {}

    world = build_virtual_world_state_from_files(
        court_calibration_path=court_path,
        tracks_path=tracks_path,
        smpl_motion_path=smpl_path,
        trust_bands=trust_bands,
    )

    assert [player["id"] for player in world["players"]] == [7, 9]
    assert trust_bands["player_membership"]["evidence_path"] == str(tmp_path / "membership.json")

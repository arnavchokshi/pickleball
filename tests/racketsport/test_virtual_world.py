from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from threed.racketsport.schemas import VirtualWorld, validate_artifact_file
from threed.racketsport.virtual_world import build_virtual_world_state


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
        "image_pts": [],
        "world_pts": [],
    }


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


def _ball_track() -> dict:
    return {
        "schema_version": 1,
        "fps": 60.0,
        "source": "tracknet",
        "frames": [{"t": 0.0, "xy": [320.0, 240.0], "conf": 0.83, "visible": True, "world_xyz": [0.1, -0.8, 0.9]}],
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
    assert world["paddles"][0]["frames"][0]["mesh_faces"] == [[0, 1, 2], [0, 2, 3]]
    expected_paddle_vertices = [
        [0.35475, -1.70315, 0.85],
        [0.54525, -1.70315, 0.85],
        [0.54525, -2.09685, 0.85],
        [0.35475, -2.09685, 0.85],
    ]
    for actual_vertex, expected_vertex in zip(
        world["paddles"][0]["frames"][0]["mesh_vertices_world"],
        expected_paddle_vertices,
        strict=True,
    ):
        assert actual_vertex == pytest.approx(expected_vertex)
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
        "warnings": [],
    }


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


def test_build_virtual_world_state_projects_visible_2d_ball_to_court_plane() -> None:
    world = build_virtual_world_state(
        court_calibration=_court_calibration(),
        tracks=_tracks(),
        ball_track=_ball_track_without_world_xyz(),
    )

    ball_frame = world["ball"]["frames"][0]
    assert ball_frame["world_xyz"] == pytest.approx([1.0, 2.0, 0.0])
    assert ball_frame["approx"] is True
    assert world["summary"]["approx_ball_frame_count"] == 1


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


def test_virtual_world_cli_writes_registered_schema_artifact(tmp_path: Path) -> None:
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
    parsed = validate_artifact_file("virtual_world", out)
    assert isinstance(parsed, VirtualWorld)
    assert parsed.summary.mesh_player_count == 1
    assert parsed.summary.mesh_player_frame_count == 1
    assert json.loads(completed.stdout)["summary"]["paddle_frame_count"] == 1

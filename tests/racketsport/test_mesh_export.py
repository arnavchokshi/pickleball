from __future__ import annotations

from threed.racketsport.mesh_export import build_body_mesh_export


def _smpl_motion() -> dict:
    return {
        "schema_version": 1,
        "model": "sam3dbody_world_joints",
        "fps": 30.0,
        "world_frame": "court_Z0",
        "joint_names": ["left_wrist"],
        "mesh_faces": [[0, 1, 2]],
        "players": [
            {
                "id": 7,
                "betas": [0.1, 0.2],
                "frames": [
                    {
                        "frame_idx": 12,
                        "t": 0.4,
                        "global_orient": [0.0, 0.0, 0.0],
                        "body_pose": [0.1, 0.2, 0.3],
                        "left_hand_pose": [0.4],
                        "right_hand_pose": [0.5],
                        "transl_world": [1.0, 2.0, 0.0],
                        "track_world_xy": [1.0, 2.0],
                        "temporal_smoothing_reset": False,
                        "joints_world": [[1.0, 2.0, 1.0]],
                        "mesh_vertices_world": [[1.0, 2.0, 0.0], [1.2, 2.1, 1.8]],
                        "joint_conf": [0.9],
                        "foot_contact": {"left": True, "right": False},
                        "grf": None,
                    }
                ],
                "skate_free": True,
                "physics": "worldhmr_floor_contact_footlock_z_snap",
            }
        ],
    }


def test_build_body_mesh_export_keeps_contact_vertices_params_and_faces_ref() -> None:
    payload = build_body_mesh_export(
        _smpl_motion(),
        clip="clip_001",
        body_compute_execution={
            "scheduled_frames": [
                {
                    "frame_idx": 12,
                    "target_player_ids": [7],
                    "source_window_index": 0,
                    "reasons": ["contact_window"],
                }
            ]
        },
    )

    assert payload["artifact_type"] == "racketsport_body_mesh"
    assert payload["clip"] == "clip_001"
    assert payload["world_frame"] == "court_Z0"
    assert payload["faces_ref"] == "mhr_faces_static"
    assert payload["mesh_faces"] == [[0, 1, 2]]
    assert payload["summary"] == {"mesh_frame_count": 1, "player_count": 1, "contact_window_count": 1}
    assert payload["players"][0]["id"] == 7
    assert payload["players"][0]["frames"] == [
        {
            "frame_idx": 12,
            "t": 0.4,
            "source_window_index": 0,
            "blend_weight": 1.0,
            "joints_world": [[1.0, 2.0, 1.0]],
            "joint_conf": [0.9],
            "mesh_vertices_world": [[1.0, 2.0, 0.0], [1.2, 2.1, 1.8]],
            "smplx_params": {
                "global_orient": [0.0, 0.0, 0.0],
                "body_pose": [0.1, 0.2, 0.3],
                "left_hand_pose": [0.4],
                "right_hand_pose": [0.5],
                "betas": [0.1, 0.2],
                "transl_world": [1.0, 2.0, 0.0],
            },
            "reasons": ["contact_window"],
        }
    ]


def test_build_body_mesh_export_keeps_window_metadata_and_raised_cosine_blend_weights() -> None:
    motion = _smpl_motion()
    frame_template = motion["players"][0]["frames"][0]
    motion["players"][0]["frames"] = [
        {**frame_template, "frame_idx": 10, "t": 10.0 / 30.0},
        {**frame_template, "frame_idx": 11, "t": 11.0 / 30.0},
        {**frame_template, "frame_idx": 12, "t": 12.0 / 30.0},
    ]

    payload = build_body_mesh_export(
        motion,
        clip="clip_001",
        body_compute_execution={
            "scheduled_frames": [
                {
                    "frame_idx": frame_idx,
                    "target_player_ids": [7],
                    "source_window_index": 4,
                    "window_frame_start": 10,
                    "window_frame_end": 12,
                    "window_frame_count": 3,
                    "window_t0": 10.0 / 30.0,
                    "window_t1": 13.0 / 30.0,
                    "target_representation": "world_mesh",
                    "fallback_representation": "lane_a_skeleton",
                    "reason_counts": {"contact_window": 3},
                    "max_score": 0.91,
                    "reasons": ["contact_window"],
                }
                for frame_idx in [10, 11, 12]
            ]
        },
    )

    assert payload["windows"] == [
        {
            "source_window_index": 4,
            "frame_start": 10,
            "frame_end": 12,
            "t0": 10.0 / 30.0,
            "t1": 13.0 / 30.0,
            "frame_count": 3,
            "target_player_ids": [7],
            "target_representation": "world_mesh",
            "fallback_representation": "lane_a_skeleton",
            "reason_counts": {"contact_window": 3},
            "max_score": 0.91,
        }
    ]
    assert [frame["blend_weight"] for frame in payload["players"][0]["frames"]] == [0.0, 1.0, 0.0]

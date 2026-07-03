from __future__ import annotations

import pytest

from threed.racketsport.body_world_label_packet_skeleton import skeleton3d_from_body_world_label_packet
from threed.racketsport.schemas import Skeleton3D


def _packet() -> dict:
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_body_world_label_packet",
        "clip": "burlington_best_zero_switch_tracks",
        "not_ground_truth": True,
        "trusted_for_world_mpjpe": False,
        "joint_names": ["nose", "left_shoulder"],
        "samples": [
            {
                "sample_id": "frame_000010_player_4",
                "frame_index": 10,
                "t": 0.166833,
                "player_id": 4,
                "track_world_xy": [1.0, 4.0],
                "predicted_joints_world": [[1.0, 4.0, 1.6], [1.1, 4.0, 1.4]],
                "joint_conf": [0.8, 0.7],
                "joint_count": 2,
                "review_required": True,
            },
            {
                "sample_id": "frame_000005_player_3",
                "frame_index": 5,
                "t": 0.083417,
                "player_id": 3,
                "track_world_xy": [-1.0, 4.0],
                "predicted_joints_world": [[-1.0, 4.0, 1.6], [-1.1, 4.0, 1.4]],
                "joint_conf": [0.6, 0.5],
                "joint_count": 2,
                "review_required": True,
            },
            {
                "sample_id": "frame_000020_player_4",
                "frame_index": 20,
                "t": 0.333667,
                "player_id": 4,
                "track_world_xy": [1.05, 4.0],
                "predicted_joints_world": [[1.05, 4.0, 1.6], [1.15, 4.0, 1.4]],
                "joint_conf": [0.9, 0.85],
                "joint_count": 2,
                "review_required": True,
            },
        ],
    }


def test_skeleton3d_from_body_world_label_packet_groups_by_player_and_sorts_by_time() -> None:
    skeleton = skeleton3d_from_body_world_label_packet(_packet(), fps=59.94005994005994)

    assert skeleton["artifact_type"] == "racketsport_skeleton3d"
    assert skeleton["preview_only"] is True
    assert skeleton["world_frame"] == "court_Z0"
    assert skeleton["joint_names"] == ["nose", "left_shoulder"]
    players_by_id = {player["id"]: player for player in skeleton["players"]}
    assert set(players_by_id) == {3, 4}
    # Player 4 has two frames (frame 10 then frame 20); they must come out time-sorted.
    frames = players_by_id[4]["frames"]
    assert [frame["frame_idx"] for frame in frames] == [10, 20]
    assert frames[0]["transl_world"] == [1.0, 4.0, 0.0]
    assert frames[0]["joints_world"] == [[1.0, 4.0, 1.6], [1.1, 4.0, 1.4]]
    assert frames[0]["joint_conf"] == [0.8, 0.7]
    assert skeleton["provenance"]["not_ground_truth"] is True
    assert skeleton["provenance"]["clip"] == "burlington_best_zero_switch_tracks"

    # Round-trips through the strict Skeleton3D schema.
    Skeleton3D.model_validate(skeleton)


def test_skeleton3d_from_body_world_label_packet_defaults_mismatched_conf_to_zero() -> None:
    packet = _packet()
    packet["samples"][0]["joint_conf"] = [0.9]  # length mismatch vs. 2 joints

    skeleton = skeleton3d_from_body_world_label_packet(packet, fps=30.0)

    player4 = next(player for player in skeleton["players"] if player["id"] == 4)
    frame10 = next(frame for frame in player4["frames"] if frame["frame_idx"] == 10)
    assert frame10["joint_conf"] == [0.0, 0.0]


def test_skeleton3d_from_body_world_label_packet_skips_samples_without_joints() -> None:
    packet = _packet()
    packet["samples"].append(
        {
            "sample_id": "frame_000030_player_4",
            "frame_index": 30,
            "t": 0.5005,
            "player_id": 4,
            "track_world_xy": [1.1, 4.0],
            "predicted_joints_world": [],
            "joint_conf": [],
            "joint_count": 0,
            "review_required": True,
        }
    )

    skeleton = skeleton3d_from_body_world_label_packet(packet, fps=30.0)

    player4 = next(player for player in skeleton["players"] if player["id"] == 4)
    assert [frame["frame_idx"] for frame in player4["frames"]] == [10, 20]


def test_skeleton3d_from_body_world_label_packet_rejects_non_positive_fps() -> None:
    with pytest.raises(ValueError, match="fps must be positive"):
        skeleton3d_from_body_world_label_packet(_packet(), fps=0.0)


def test_skeleton3d_from_body_world_label_packet_rejects_missing_samples() -> None:
    with pytest.raises(ValueError, match="samples must be a list"):
        skeleton3d_from_body_world_label_packet({"joint_names": []}, fps=30.0)

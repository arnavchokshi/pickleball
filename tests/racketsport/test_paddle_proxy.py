from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from threed.racketsport.paddle_proxy import build_paddle_proxy_from_skeleton


def _skeleton_payload(*, right_wrist_conf: float = 0.9, frames: list[dict] | None = None) -> dict:
    joint_names = ["right_elbow", "right_wrist", "right_shoulder", "left_elbow", "left_wrist", "left_shoulder"]
    if frames is None:
        frames = [
            {
                "frame_idx": 0,
                "t": 0.0,
                "joints_world": [
                    [0.0, 0.0, 1.0],
                    [1.0, 0.0, 1.0],
                    [0.0, -0.2, 1.35],
                    [0.0, 0.0, 1.0],
                    [-1.0, 0.0, 1.0],
                    [0.0, 0.2, 1.35],
                ],
                "joint_conf": [0.91, right_wrist_conf, 0.88, 0.87, 0.86, 0.85],
            },
            {
                "frame_idx": 1,
                "t": 0.1,
                "joints_world": [
                    [0.0, 0.0, 1.0],
                    [2.0, 0.0, 1.0],
                    [0.0, -0.2, 1.35],
                    [0.0, 0.0, 1.0],
                    [-1.0, 0.0, 1.0],
                    [0.0, 0.2, 1.35],
                ],
                "joint_conf": [0.91, 0.92, 0.88, 0.87, 0.86, 0.85],
            },
        ]
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_skeleton3d",
        "fps": 10.0,
        "world_frame": "court_Z0",
        "source_model": "synthetic_test",
        "joint_names": joint_names,
        "preview_only": True,
        "players": [{"id": 7, "frames": frames}],
        "provenance": {"source": "unit_test"},
    }


def test_paddle_proxy_offsets_from_dominant_wrist_and_smooths_position() -> None:
    artifact = build_paddle_proxy_from_skeleton(
        _skeleton_payload(),
        clip_id="synthetic",
        dominant_hand="right",
        wrist_offset_m=0.2,
        min_joint_confidence=0.5,
        smoothing_alpha=0.5,
    )

    frames = artifact["players"][0]["frames"]
    assert artifact["source"] == "wrist_proxy"
    assert artifact["render_only"] is True
    assert artifact["trusted_for_rkt_promotion"] is False
    assert artifact["summary"]["estimate_frame_count"] == 2
    assert artifact["summary"]["hidden_frame_count"] == 0
    assert frames[0]["pose_se3"]["t"] == pytest.approx([1.2, 0.0, 1.0])
    assert frames[1]["pose_se3"]["t"] == pytest.approx([1.7, 0.0, 1.0])

    first_rotation = frames[0]["pose_se3"]["R"]
    length_axis = [first_rotation[row][1] for row in range(3)]
    face_normal = [first_rotation[row][2] for row in range(3)]
    assert length_axis == pytest.approx([1.0, 0.0, 0.0])
    assert sum(component * component for component in face_normal) == pytest.approx(1.0)
    assert abs(sum(length_axis[i] * face_normal[i] for i in range(3))) < 1e-9
    assert frames[0]["source"] == "wrist_proxy"
    assert frames[0]["render_only"] is True
    assert frames[0]["trust"] == "estimated_from_wrist"
    assert frames[0]["trust_band"]["status"] == "estimated_from_wrist"


def test_paddle_proxy_emits_grip_attachment_and_render_style_metadata() -> None:
    artifact = build_paddle_proxy_from_skeleton(
        _skeleton_payload(),
        clip_id="synthetic",
        dominant_hand="right",
        wrist_offset_m=0.2,
        min_joint_confidence=0.5,
        smoothing_alpha=1.0,
    )

    frame = artifact["players"][0]["frames"][0]
    assert artifact["parameters"]["render_mesh_style"] == "paddle_face_with_handle"
    assert frame["render_mesh"]["style"] == "paddle_face_with_handle"
    assert frame["render_mesh"]["face_vertex_count"] == 4
    assert frame["render_mesh"]["handle_vertex_count"] == 4
    assert frame["attachment"]["hand_side"] == "right"
    assert frame["attachment"]["joint_basis"] == "right_wrist+right_elbow"
    assert frame["attachment"]["attachment_point_world"] == pytest.approx([1.0, 0.0, 1.0])
    assert frame["attachment"]["grip_center_world"] == pytest.approx([1.04, 0.0, 1.0])
    assert frame["attachment"]["face_center_world"] == pytest.approx(frame["pose_se3"]["t"])
    assert frame["proxy_inputs"]["grip_center_world"] == pytest.approx([1.04, 0.0, 1.0])
    assert frame["proxy_inputs"]["face_center_world"] == pytest.approx([1.2, 0.0, 1.0])


def test_paddle_proxy_physically_places_face_center_from_hand_grip_and_paddle_dimensions() -> None:
    artifact = build_paddle_proxy_from_skeleton(
        _skeleton_payload(),
        clip_id="synthetic",
        dominant_hand="right",
        min_joint_confidence=0.5,
        smoothing_alpha=1.0,
    )

    frame = artifact["players"][0]["frames"][0]

    assert frame["attachment"]["attachment_point_world"] == pytest.approx([1.0, 0.0, 1.0])
    assert frame["attachment"]["grip_center_world"] == pytest.approx([1.04, 0.0, 1.0])
    assert frame["attachment"]["grip_to_face_center_m"] == pytest.approx((15.5 + 5.25) * 0.0254 / 2.0)
    assert frame["pose_se3"]["t"] == pytest.approx([1.303525, 0.0, 1.0])
    assert frame["attachment"]["face_center_world"] == pytest.approx(frame["pose_se3"]["t"])
    assert frame["proxy_inputs"]["placement_model"] == "hand_grip_to_face_center_from_paddle_dimensions"


def test_paddle_proxy_uses_ball_trajectory_change_to_tilt_estimated_face_normal() -> None:
    ball_track = {
        "frames": [
            {"t": 0.0, "world_xyz": [0.0, 0.0, 1.0]},
            {"t": 0.1, "world_xyz": [0.0, 1.0, 1.0]},
            {"t": 0.2, "world_xyz": [0.0, 1.0, 2.0]},
        ]
    }
    artifact = build_paddle_proxy_from_skeleton(
        _skeleton_payload(),
        clip_id="synthetic",
        dominant_hand="right",
        ball_track=ball_track,
        min_joint_confidence=0.5,
        smoothing_alpha=1.0,
    )

    second = artifact["players"][0]["frames"][1]
    face_normal = [second["pose_se3"]["R"][row][2] for row in range(3)]

    assert second["orientation_hint"]["orientation_model"] == "hand_anchor_ball_reflection_estimate"
    assert second["orientation_hint"]["face_normal_solver"] == "ball_reflection_impulse_projected_to_forearm_plane"
    assert second["orientation_hint"]["incoming_ball_direction_world"] == pytest.approx([0.0, 1.0, 0.0])
    assert second["orientation_hint"]["outgoing_ball_direction_world"] == pytest.approx([0.0, 0.0, 1.0])
    assert second["orientation_hint"]["ball_impulse_direction_world"] == pytest.approx([0.0, -0.707106781, 0.707106781])
    assert face_normal == pytest.approx([0.0, -0.707106781, 0.707106781])
    assert second["confidence_provenance"]["predictor"] == "wrist_proxy+ball_reflection_hint"
    assert second["trust_band"]["status"] == "estimated_from_wrist"


def test_paddle_proxy_auto_selects_side_per_player_and_accepts_player_override() -> None:
    frames = [
        {
            "frame_idx": 0,
            "t": 0.0,
            "joints_world": [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [0.0, -0.2, 1.35],
                [0.0, 0.0, 1.0],
                [-1.0, 0.0, 1.0],
                [0.0, 0.2, 1.35],
            ],
            "joint_conf": [0.1, 0.1, 0.88, 0.9, 0.9, 0.85],
        },
        {
            "frame_idx": 1,
            "t": 0.1,
            "joints_world": [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [0.0, -0.2, 1.35],
                [0.0, 0.0, 1.0],
                [-1.1, 0.0, 1.0],
                [0.0, 0.2, 1.35],
            ],
            "joint_conf": [0.1, 0.1, 0.88, 0.9, 0.9, 0.85],
        },
    ]
    artifact = build_paddle_proxy_from_skeleton(
        _skeleton_payload(frames=frames),
        clip_id="synthetic",
        dominant_hand="auto",
        min_joint_confidence=0.5,
        smoothing_alpha=1.0,
    )
    player = artifact["players"][0]
    assert player["dominant_hand"] == "left"
    assert player["hand_selection"]["selected_side"] == "left"
    assert player["hand_selection"]["side_scores"]["left"]["usable_frame_count"] == 2
    assert player["hand_selection"]["side_scores"]["right"]["usable_frame_count"] == 0
    assert artifact["summary"]["dominant_hand_by_player"] == {"7": "left"}

    overridden = build_paddle_proxy_from_skeleton(
        _skeleton_payload(frames=frames),
        clip_id="synthetic",
        dominant_hand="auto",
        dominant_hand_by_player={7: "right"},
        min_joint_confidence=0.05,
        smoothing_alpha=1.0,
    )
    assert overridden["players"][0]["dominant_hand"] == "right"
    assert overridden["players"][0]["hand_selection"]["selection_reason"] == "player_override"


def test_paddle_proxy_records_motion_and_ball_orientation_hints_without_promotion() -> None:
    ball_track = {
        "frames": [
            {"t": 0.0, "world_xyz": [1.0, -0.1, 1.0]},
            {"t": 0.1, "world_xyz": [1.0, 0.4, 1.0]},
            {"t": 0.2, "world_xyz": [1.0, 0.8, 1.0]},
        ]
    }
    artifact = build_paddle_proxy_from_skeleton(
        _skeleton_payload(),
        clip_id="synthetic",
        dominant_hand="right",
        ball_track=ball_track,
        min_joint_confidence=0.5,
        smoothing_alpha=1.0,
    )

    assert artifact["trusted_for_rkt_promotion"] is False
    assert artifact["never_canonical_racket_pose"] is True
    assert "motion_ball_orientation_hints_estimated_only" in artifact["warnings"]
    second = artifact["players"][0]["frames"][1]
    assert second["orientation_hint"]["trust"] == "estimated_from_motion"
    assert second["orientation_hint"]["sources"] == ["wrist_swing", "ball_path"]
    assert second["orientation_hint"]["swing_direction_world"] == pytest.approx([1.0, 0.0, 0.0])
    assert second["orientation_hint"]["ball_direction_world"] == pytest.approx([0.0, 1.0, 0.0])
    assert second["confidence_provenance"]["predictor"] == "wrist_proxy+motion_hint"
    assert second["trust_band"]["status"] == "estimated_from_wrist"


def test_paddle_proxy_hides_low_confidence_or_missing_wrist_frames() -> None:
    artifact = build_paddle_proxy_from_skeleton(
        _skeleton_payload(right_wrist_conf=0.1),
        clip_id="synthetic",
        dominant_hand="right",
        min_joint_confidence=0.5,
        smoothing_alpha=1.0,
    )

    assert artifact["summary"]["input_frame_count"] == 2
    assert artifact["summary"]["estimate_frame_count"] == 1
    assert artifact["summary"]["hidden_frame_count"] == 1
    assert artifact["summary"]["hidden_frame_counts_by_reason"] == {"low_joint_confidence": 1}
    assert artifact["hidden_frames"] == [
        {
            "player_id": 7,
            "frame_idx": 0,
            "t": 0.0,
            "side": "right",
            "reason": "low_joint_confidence",
            "joint_confidence": pytest.approx(0.1),
        }
    ]


def test_paddle_proxy_cli_writes_racket_pose_estimate_compatible_artifact(tmp_path: Path) -> None:
    skeleton_path = tmp_path / "skeleton3d.json"
    out_path = tmp_path / "racket_pose_estimate.json"
    skeleton_path.write_text(json.dumps(_skeleton_payload(), indent=2), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/build_paddle_proxy.py",
            "--skeleton3d",
            str(skeleton_path),
            "--out",
            str(out_path),
            "--clip",
            "synthetic",
            "--dominant-hand",
            "right",
            "--wrist-offset-m",
            "0.2",
            "--smoothing-alpha",
            "1.0",
            "--min-joint-confidence",
            "0.5",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert completed.stderr == ""
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["artifact_type"] == "racketsport_racket_pose_estimate"
    assert payload["source"] == "wrist_proxy"
    assert payload["summary"]["estimate_frame_count"] == 2
    assert json.loads(completed.stdout)["summary"]["estimate_frame_count"] == 2

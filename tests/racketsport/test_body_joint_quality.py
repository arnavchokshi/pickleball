from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from threed.racketsport.body_joint_quality import build_body_joint_quality


def _smpl_motion(*, player_frames: int = 2, joint_count: int = 3) -> dict:
    frames = []
    for index in range(player_frames):
        frames.append(
            {
                "frame_idx": index,
                "t": index / 30.0,
                "transl_world": [0.0 + 0.1 * index, 1.0, 0.0],
                "joints_world": [
                    [0.0 + 0.1 * index, 1.0, 0.0],
                    [0.0 + 0.1 * index, 1.0, 1.0],
                    [0.0 + 0.1 * index, 1.0, 1.7],
                ][:joint_count],
                "joint_conf": [0.9] * joint_count,
                "mesh_vertices_world": [[0.0, 1.0, 0.0], [0.1, 1.0, 1.7]],
                "track_world_xy": [0.0 + 0.1 * index, 1.0],
            }
        )
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_smpl_motion",
        "fps": 30.0,
        "world_frame": "court_Z0",
        "players": [{"id": 7, "frames": frames}],
    }


def _skeleton3d(*, player_frames: int = 2, joint_count: int = 3) -> dict:
    smpl = _smpl_motion(player_frames=player_frames, joint_count=joint_count)
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_skeleton3d",
        "preview_only": True,
        "joint_names": [f"sam3dbody_joint_{index:03d}" for index in range(joint_count)],
        "players": [
            {
                "id": 7,
                "frames": [
                    {
                        "t": frame["t"],
                        "joints_world": frame["joints_world"],
                        "joint_conf": frame["joint_conf"],
                    }
                    for frame in smpl["players"][0]["frames"]
                ],
            }
        ],
    }


def _body_execution(*, scheduled_player_frames: int = 2) -> dict:
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_body_compute_execution",
        "summary": {
            "scheduled_frame_count": scheduled_player_frames,
            "scheduled_player_frame_count": scheduled_player_frames,
            "scheduled_by_target_representation": {"world_mesh": scheduled_player_frames},
        },
    }


def _body_world_label_packet(*, track_world_xy: list[float] | None = None) -> dict:
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_body_world_label_packet",
        "status": "needs_review",
        "not_ground_truth": True,
        "trusted_for_world_mpjpe": False,
        "samples": [
            {
                "sample_id": "frame_000000_player_7",
                "frame_index": 0,
                "t": 0.0,
                "player_id": 7,
                "track_world_xy": track_world_xy or [0.0, 1.0],
                "predicted_joints_world": [[0.0, 1.0, 0.0], [0.0, 1.0, 1.0], [0.0, 1.0, 1.7]],
                "joint_conf": [0.9, 0.9, 0.9],
                "joint_count": 3,
            },
            {
                "sample_id": "frame_000001_player_7",
                "frame_index": 1,
                "t": 1.0 / 30.0,
                "player_id": 7,
                "track_world_xy": track_world_xy or [0.1, 1.0],
                "predicted_joints_world": [[0.1, 1.0, 0.0], [0.1, 1.0, 1.0], [0.1, 1.0, 1.7]],
                "joint_conf": [0.9, 0.9, 0.9],
                "joint_count": 3,
            },
        ],
    }


def test_body_joint_quality_reports_structural_usability_without_accuracy_promotion() -> None:
    payload = build_body_joint_quality(
        clip="clip_001",
        smpl_motion=_smpl_motion(),
        skeleton3d=_skeleton3d(),
        body_compute_execution=_body_execution(),
        min_joint_count=3,
    )

    assert payload["artifact_type"] == "racketsport_body_joint_quality"
    assert payload["status"] == "quality_checked_needs_accuracy_gate"
    assert payload["usable_for_review"] is True
    assert payload["trusted_for_body_promotion"] is False
    assert payload["accuracy_verified"] is False
    assert payload["quality_blockers"] == []
    assert payload["promotion_blockers"] == ["missing_world_mpjpe_gate", "missing_full_clip_body_gate"]
    assert payload["summary"]["joint_frame_count"] == 2
    assert payload["summary"]["joint_count_min"] == 3
    assert payload["summary"]["joint_count_max"] == 3
    assert payload["summary"]["scheduled_player_frame_count"] == 2
    assert payload["summary"]["schedule_coverage_ratio"] == 1.0
    assert payload["summary"]["min_joint_z_m"] == 0.0


def test_body_joint_quality_uses_compact_prediction_packet_when_motion_files_are_absent() -> None:
    payload = build_body_joint_quality(
        clip="clip_001",
        body_world_label_packet=_body_world_label_packet(),
        body_compute_execution=_body_execution(),
        min_joint_count=3,
    )

    assert payload["status"] == "quality_checked_needs_accuracy_gate"
    assert payload["usable_for_review"] is True
    assert payload["world_joints_available"] is True
    assert payload["quality_blockers"] == []
    assert payload["summary"]["joint_source"] == "body_world_label_packet"
    assert payload["summary"]["joint_frame_count"] == 2
    assert payload["summary"]["joint_count_min"] == 3
    assert payload["summary"]["scheduled_player_frame_count"] == 2
    assert payload["summary"]["schedule_coverage_ratio"] == 1.0
    assert payload["summary"]["min_joint_z_m"] == 0.0
    assert payload["summary"]["max_track_anchor_residual_m"] == pytest.approx(0.0)
    assert "missing_smpl_motion_json" not in payload["quality_blockers"]
    assert "missing_skeleton3d_json" not in payload["quality_blockers"]


def test_body_joint_quality_blocks_compact_packet_joints_far_from_track_anchor() -> None:
    payload = build_body_joint_quality(
        clip="clip_001",
        body_world_label_packet=_body_world_label_packet(track_world_xy=[4.0, 1.0]),
        body_compute_execution=_body_execution(),
        min_joint_count=3,
        warn_track_anchor_residual_m=1.0,
        max_track_anchor_residual_for_review_m=3.0,
    )

    assert payload["status"] == "quality_blocked"
    assert payload["usable_for_review"] is False
    assert "track_anchor_residual_too_large" in payload["quality_blockers"]
    assert payload["summary"]["joint_source"] == "body_world_label_packet"
    assert payload["summary"]["max_track_anchor_residual_m"] == pytest.approx(4.0)


def test_body_joint_quality_blocks_incomplete_or_invalid_world_joints() -> None:
    smpl_motion = _smpl_motion(player_frames=1)
    smpl_motion["players"][0]["frames"][0]["joints_world"][1][2] = float("nan")

    payload = build_body_joint_quality(
        clip="clip_001",
        smpl_motion=smpl_motion,
        skeleton3d=_skeleton3d(player_frames=1),
        body_compute_execution=_body_execution(scheduled_player_frames=2),
        min_joint_count=3,
    )

    assert payload["status"] == "quality_blocked"
    assert payload["usable_for_review"] is False
    assert "nonfinite_joint_world" in payload["quality_blockers"]
    assert "scheduled_body_output_incomplete" in payload["quality_blockers"]
    assert payload["summary"]["joint_frame_count"] == 1
    assert payload["summary"]["scheduled_player_frame_count"] == 2
    assert payload["summary"]["schedule_coverage_ratio"] == 0.5


def test_body_joint_quality_blocks_implausible_short_term_root_speed() -> None:
    smpl_motion = _smpl_motion()
    smpl_motion["players"][0]["frames"][1]["transl_world"] = [2.0, 1.0, 0.0]

    payload = build_body_joint_quality(
        clip="clip_001",
        smpl_motion=smpl_motion,
        skeleton3d=_skeleton3d(),
        body_compute_execution=_body_execution(),
        min_joint_count=3,
        max_root_speed_for_review_mps=10.0,
    )

    assert payload["status"] == "quality_blocked"
    assert payload["usable_for_review"] is False
    assert "root_motion_temporal_jump" in payload["quality_blockers"]
    assert payload["summary"]["max_root_speed_mps"] == pytest.approx(60.0)
    assert payload["summary"]["root_motion_temporal_jump_count"] == 1
    assert payload["root_motion_temporal_jumps"] == [
        {
            "player_id": "7",
            "prev_frame_idx": 0,
            "frame_idx": 1,
            "prev_t": pytest.approx(0.0),
            "t": pytest.approx(1.0 / 30.0, abs=1e-6),
            "dt_s": pytest.approx(1.0 / 30.0, abs=1e-6),
            "step_m": pytest.approx(2.0),
            "speed_mps": pytest.approx(60.0),
            "prev_root_world": [0.0, 1.0, 0.0],
            "root_world": [2.0, 1.0, 0.0],
        }
    ]


def test_body_joint_quality_ignores_root_speed_across_temporal_reset() -> None:
    smpl_motion = _smpl_motion()
    smpl_motion["players"][0]["frames"][1]["transl_world"] = [2.0, 1.0, 0.0]
    smpl_motion["players"][0]["frames"][1]["track_world_xy"] = [2.0, 1.0]
    smpl_motion["players"][0]["frames"][1]["temporal_smoothing_reset"] = True

    payload = build_body_joint_quality(
        clip="clip_001",
        smpl_motion=smpl_motion,
        skeleton3d=_skeleton3d(),
        body_compute_execution=_body_execution(),
        min_joint_count=3,
        max_root_speed_for_review_mps=10.0,
    )

    assert payload["status"] == "quality_checked_needs_accuracy_gate"
    assert "root_motion_temporal_jump" not in payload["quality_blockers"]
    assert payload["summary"]["temporal_smoothing_reset_count"] == 1
    assert payload["summary"]["max_root_speed_mps"] is None
    assert payload["summary"]["root_motion_temporal_jump_count"] == 0
    assert payload["root_motion_temporal_jumps"] == []


def test_body_joint_quality_warns_then_blocks_track_anchor_residual() -> None:
    warning_motion = _smpl_motion(player_frames=1)
    warning_motion["players"][0]["frames"][0]["transl_world"] = [2.0, 1.0, 0.0]
    warning_motion["players"][0]["frames"][0]["track_world_xy"] = [0.0, 1.0]

    warning_payload = build_body_joint_quality(
        clip="clip_001",
        smpl_motion=warning_motion,
        skeleton3d=_skeleton3d(player_frames=1),
        body_compute_execution=_body_execution(scheduled_player_frames=1),
        min_joint_count=3,
        warn_track_anchor_residual_m=1.0,
        max_track_anchor_residual_for_review_m=3.0,
    )

    assert warning_payload["status"] == "quality_checked_needs_accuracy_gate"
    assert "track_anchor_residual_high" in warning_payload["warnings"]
    assert warning_payload["summary"]["max_track_anchor_residual_m"] == pytest.approx(2.0)

    blocked_motion = _smpl_motion(player_frames=1)
    blocked_motion["players"][0]["frames"][0]["transl_world"] = [4.0, 1.0, 0.0]
    blocked_motion["players"][0]["frames"][0]["track_world_xy"] = [0.0, 1.0]

    blocked_payload = build_body_joint_quality(
        clip="clip_001",
        smpl_motion=blocked_motion,
        skeleton3d=_skeleton3d(player_frames=1),
        body_compute_execution=_body_execution(scheduled_player_frames=1),
        min_joint_count=3,
        warn_track_anchor_residual_m=1.0,
        max_track_anchor_residual_for_review_m=3.0,
    )

    assert blocked_payload["status"] == "quality_blocked"
    assert "track_anchor_residual_too_large" in blocked_payload["quality_blockers"]


def test_body_joint_quality_allows_isolated_track_anchor_outlier_in_long_clip() -> None:
    smpl_motion = _smpl_motion(player_frames=200)
    smpl_motion["players"][0]["frames"][2]["track_world_xy"] = [4.0, 1.0]

    payload = build_body_joint_quality(
        clip="clip_001",
        smpl_motion=smpl_motion,
        skeleton3d=_skeleton3d(player_frames=200),
        body_compute_execution=_body_execution(scheduled_player_frames=200),
        min_joint_count=3,
        warn_track_anchor_residual_m=1.0,
        max_track_anchor_residual_for_review_m=3.0,
    )

    assert payload["status"] == "quality_checked_needs_accuracy_gate"
    assert payload["usable_for_review"] is True
    assert "track_anchor_residual_outliers" in payload["warnings"]
    assert "track_anchor_residual_too_large" not in payload["quality_blockers"]
    assert payload["summary"]["track_anchor_residual_over_review_count"] == 1
    assert payload["summary"]["track_anchor_residual_over_review_ratio"] == pytest.approx(0.005)


def test_build_body_joint_quality_cli_writes_report(tmp_path: Path) -> None:
    smpl = tmp_path / "smpl_motion.json"
    skeleton = tmp_path / "skeleton3d.json"
    execution = tmp_path / "body_compute_execution.json"
    out = tmp_path / "body_joint_quality.json"
    smpl.write_text(json.dumps(_smpl_motion()), encoding="utf-8")
    skeleton.write_text(json.dumps(_skeleton3d()), encoding="utf-8")
    execution.write_text(json.dumps(_body_execution()), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/build_body_joint_quality.py",
            "--clip",
            "clip_001",
            "--smpl-motion",
            str(smpl),
            "--skeleton3d",
            str(skeleton),
            "--body-compute-execution",
            str(execution),
            "--out",
            str(out),
            "--min-joint-count",
            "3",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert completed.stderr == ""
    assert json.loads(completed.stdout)["status"] == "quality_checked_needs_accuracy_gate"
    assert json.loads(out.read_text(encoding="utf-8"))["summary"]["joint_frame_count"] == 2

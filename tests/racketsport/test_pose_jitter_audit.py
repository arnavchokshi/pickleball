from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.racketsport.audit_pose_temporal_jitter import main as audit_pose_temporal_jitter_main
from threed.racketsport.pose_fast import LANE_A_RTMW3D_JOINT_NAMES
from threed.racketsport.pose_temporal import (
    compare_wrist_peak_timing,
    compute_pose_jitter_audit,
    refine_lane_a_skeleton3d,
)


FIXTURE = Path("tests/racketsport/fixtures/pose_jitter_real_excerpt_skeleton3d.json")


def _idx(name: str) -> int:
    return list(LANE_A_RTMW3D_JOINT_NAMES).index(name)


def _base_joints() -> list[list[float]]:
    return [[0.0, 0.0, 1.0] for _name in LANE_A_RTMW3D_JOINT_NAMES]


def _frame(frame_idx: int, joints: list[list[float]], *, conf: list[float] | None = None) -> dict:
    return {
        "frame_idx": frame_idx,
        "t": frame_idx / 30.0,
        "joints_world": joints,
        "joint_conf": conf or [0.9 for _name in LANE_A_RTMW3D_JOINT_NAMES],
    }


def _skeleton(frames: list[dict]) -> dict:
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_skeleton3d",
        "fps": 30.0,
        "world_frame": "court_Z0",
        "source_model": "rtmw3d_x",
        "joint_names": list(LANE_A_RTMW3D_JOINT_NAMES),
        "preview_only": False,
        "players": [{"id": 7, "frames": frames}],
        "provenance": {"lane": "POSE-SMOOTH-test"},
    }


def test_full_body_one_euro_covers_all_65_joints_and_marks_low_confidence() -> None:
    shoulder_idx = _idx("left_shoulder")
    hand_idx = _idx("left_hand_00")
    toe_idx = _idx("left_big_toe")
    hip_idx = _idx("right_hip")
    frames = []
    for frame_idx, offset in enumerate([0.0, 1.0, 0.0]):
        joints = _base_joints()
        joints[shoulder_idx] = [offset, 0.0, 1.5]
        joints[hand_idx] = [offset, 0.0, 1.4]
        joints[toe_idx] = [offset, 0.0, 0.0]
        conf = [0.9 for _name in LANE_A_RTMW3D_JOINT_NAMES]
        conf[hip_idx] = 0.1
        frames.append(_frame(frame_idx, joints, conf=conf))

    refined = refine_lane_a_skeleton3d(_skeleton(frames), fps=30.0)

    temporal = refined["provenance"]["temporal_refine"]
    assert temporal["one_euro"]["applied_joint_groups"] == ["core_body", "feet", "hands", "wrists"]
    assert temporal["one_euro"]["filtered_joint_count"] == 65
    out_frame = refined["players"][0]["frames"][1]
    assert out_frame["joints_world"][shoulder_idx][0] < 1.0
    assert out_frame["joints_world"][hand_idx][0] < 1.0
    assert out_frame["joints_world"][toe_idx][0] < 1.0
    assert len(out_frame["smoothing_flag"]) == len(LANE_A_RTMW3D_JOINT_NAMES)
    assert "low_confidence_joint" in out_frame["smoothing_flag"][hip_idx]


def test_pose_temporal_flags_and_damps_core_single_frame_teleport_and_sustained_speed() -> None:
    shoulder_idx = _idx("left_shoulder")
    frames = []
    for frame_idx, offset in enumerate([0.0, 2.0, 2.2]):
        joints = _base_joints()
        joints[shoulder_idx] = [offset, 0.0, 1.5]
        frames.append(_frame(frame_idx, joints))

    refined = refine_lane_a_skeleton3d(
        _skeleton(frames),
        fps=30.0,
        one_euro_mincutoff=1000.0,
        one_euro_beta=0.0,
    )

    output_frames = refined["players"][0]["frames"]
    assert output_frames[1]["joints_world"][shoulder_idx][0] <= 0.11
    assert "single_frame_jump_clamped" in output_frames[1]["smoothing_flag"][shoulder_idx]
    assert "core_speed_clamped" in output_frames[2]["smoothing_flag"][shoulder_idx]
    counts = refined["provenance"]["temporal_refine"]["smoothing_flags"]
    assert counts["single_frame_jump_clamped"] >= 1
    assert counts["core_speed_clamped"] >= 1


def test_wrist_peak_timing_survives_synthetic_fast_swing_within_one_frame() -> None:
    shoulder_idx = _idx("left_shoulder")
    elbow_idx = _idx("left_elbow")
    wrist_idx = _idx("left_wrist")
    frames = []
    for frame_idx, x in enumerate([0.0, 0.05, 0.15, 0.35, 0.82, 1.28, 1.58, 1.72, 1.77]):
        joints = _base_joints()
        joints[shoulder_idx] = [x - 0.6, 0.0, 1.2]
        joints[elbow_idx] = [x - 0.3, 0.0, 1.15]
        joints[wrist_idx] = [x, 0.0, 1.1]
        frames.append(_frame(frame_idx, joints))
    original = _skeleton(frames)

    refined = refine_lane_a_skeleton3d(original, fps=30.0)
    timing = compare_wrist_peak_timing(original, refined, top_k=1)

    assert timing["status"] == "pass"
    assert timing["max_abs_delta_frames"] <= 1
    assert timing["comparisons"][0]["joint_name"] == "left_wrist"


def test_jitter_audit_before_after_on_stored_real_wolverine_excerpt() -> None:
    skeleton = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert skeleton["provenance"]["source_player_id"] == 4

    before = compute_pose_jitter_audit(skeleton)
    refined = refine_lane_a_skeleton3d(skeleton, fps=30.0)
    after = compute_pose_jitter_audit(refined)

    assert before["group_stats"]["core_body"]["p90_frame_displacement_m"] > 1.0
    assert after["group_stats"]["core_body"]["p90_frame_displacement_m"] < 0.3
    assert refined["provenance"]["temporal_refine"]["smoothing_flags"]["single_frame_jump_clamped"] > 0


def test_pose_temporal_jitter_cli_writes_reference_artifacts(tmp_path: Path) -> None:
    exit_code = audit_pose_temporal_jitter_main(
        [
            "--skeleton3d",
            str(FIXTURE),
            "--out-dir",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    summary = json.loads((tmp_path / "pose_temporal_jitter_summary.json").read_text(encoding="utf-8"))
    smoothed = json.loads((tmp_path / "skeleton3d_pose_smooth.json").read_text(encoding="utf-8"))
    assert summary["status"] == "pass"
    assert summary["before"]["group_stats"]["core_body"]["p90_frame_displacement_m"] > 1.0
    assert summary["after"]["group_stats"]["core_body"]["p90_frame_displacement_m"] < 0.3
    assert summary["wrist_peak_timing"]["max_abs_delta_frames"] <= 1
    assert "smoothing_flag" in smoothed["players"][0]["frames"][0]

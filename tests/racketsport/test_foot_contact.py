from __future__ import annotations

import pytest

from threed.racketsport.foot_contact import (
    ContactThresholds,
    SkeletonFrame,
    detect_contact_phases,
    measure_contact_metrics,
    resolve_foot_joint_indices,
)


JOINT_NAMES_65 = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_big_toe",
    "left_small_toe",
    "left_heel",
    "right_big_toe",
    "right_small_toe",
    "right_heel",
)


def _frame(frame_index: int, *, left_x: float, left_z: float, right_x: float = 1.0, right_z: float = 0.30) -> SkeletonFrame:
    joints = [[0.0, 0.0, 1.0] for _ in JOINT_NAMES_65]
    for idx in (15, 17, 18, 19):
        joints[idx] = [left_x, 0.0, left_z]
    for idx in (16, 20, 21, 22):
        joints[idx] = [right_x, 0.0, right_z]
    joints[0] = [0.2 + left_x, 0.0, 1.7]
    return SkeletonFrame(
        player_id="p1",
        frame_index=frame_index,
        t=frame_index / 30.0,
        joints_world=joints,
        joint_conf=[0.9] * len(joints),
    )


def test_resolve_foot_joint_indices_prefers_payload_names_for_65_joint_schema():
    indices = resolve_foot_joint_indices(JOINT_NAMES_65, joint_count=len(JOINT_NAMES_65))

    assert indices.left == (15, 17, 18, 19)
    assert indices.right == (16, 20, 21, 22)


def test_detect_contact_phases_uses_floor_height_speed_and_hysteresis():
    frames = [
        _frame(0, left_x=0.000, left_z=0.020),
        _frame(1, left_x=0.002, left_z=0.025),
        _frame(2, left_x=0.004, left_z=0.055),
        _frame(3, left_x=0.006, left_z=0.070),
        _frame(4, left_x=0.009, left_z=0.130),
    ]

    phases = detect_contact_phases(
        frames,
        joint_names=JOINT_NAMES_65,
        thresholds=ContactThresholds(
            enter_height_m=0.030,
            exit_height_m=0.080,
            enter_speed_mps=0.25,
            exit_speed_mps=0.40,
            min_phase_frames=2,
        ),
    )

    assert len(phases) == 1
    assert phases[0].foot == "left"
    assert phases[0].start_frame_index == 0
    assert phases[0].end_frame_index == 3
    assert phases[0].frame_count == 4


def test_measure_contact_metrics_reports_phase_slide_and_foot_penetration():
    frames = [
        _frame(0, left_x=0.000, left_z=0.000),
        _frame(1, left_x=0.020, left_z=-0.010),
        _frame(2, left_x=0.035, left_z=0.010),
        _frame(3, left_x=0.100, left_z=0.200),
    ]
    phases = detect_contact_phases(
        frames[:3],
        joint_names=JOINT_NAMES_65,
        thresholds=ContactThresholds(
            enter_height_m=0.040,
            exit_height_m=0.070,
            enter_speed_mps=1.20,
            exit_speed_mps=1.20,
            min_phase_frames=2,
        ),
    )

    metrics = measure_contact_metrics(frames, phases, joint_names=JOINT_NAMES_65)

    assert metrics.phase_metrics[0].slide_mm == pytest.approx(35.0)
    assert metrics.penetration.max_penetration_mm == pytest.approx(10.0)
    assert metrics.summary_by_player["p1"].phase_count == 1
    assert metrics.summary_by_player["p1"].max_slide_mm == pytest.approx(35.0)

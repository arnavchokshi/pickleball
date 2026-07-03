from __future__ import annotations

import copy

import pytest

from tests.racketsport.test_foot_contact import JOINT_NAMES_65, _frame
from threed.racketsport.foot_contact import ContactThresholds, detect_contact_phases, measure_contact_metrics
from threed.racketsport.foot_lock_solver import FootLockSolverSettings, solve_foot_lock


def test_solve_foot_lock_pins_stance_phase_to_anchor_and_removes_penetration():
    frames = [
        _frame(0, left_x=0.000, left_z=0.000),
        _frame(1, left_x=0.015, left_z=-0.010),
        _frame(2, left_x=0.030, left_z=0.008),
        _frame(3, left_x=0.045, left_z=0.010),
        _frame(4, left_x=0.300, left_z=0.200),
    ]
    original_airborne = copy.deepcopy(frames[-1])
    phases = detect_contact_phases(
        frames[:4],
        joint_names=JOINT_NAMES_65,
        thresholds=ContactThresholds(
            enter_height_m=0.040,
            exit_height_m=0.070,
            enter_speed_mps=1.50,
            exit_speed_mps=1.50,
            min_phase_frames=2,
        ),
    )

    result = solve_foot_lock(
        frames,
        phases,
        joint_names=JOINT_NAMES_65,
        settings=FootLockSolverSettings(root_translation_weight=0.5),
    )
    solved_metrics = measure_contact_metrics(result.frames, phases, joint_names=JOINT_NAMES_65)

    assert solved_metrics.summary_by_player["p1"].max_slide_mm <= 3.0
    assert solved_metrics.penetration.max_penetration_mm == pytest.approx(0.0)
    assert result.frames[-1].joints_world == original_airborne.joints_world
    assert result.max_non_foot_joint_displacement_m > 0.0
    assert result.max_non_foot_joint_displacement_m < 0.10


def test_solve_foot_lock_leaves_all_airborne_sequence_unchanged():
    frames = [
        _frame(0, left_x=0.000, left_z=0.300, right_x=1.0, right_z=0.320),
        _frame(1, left_x=0.200, left_z=0.350, right_x=1.2, right_z=0.360),
    ]

    result = solve_foot_lock(frames, [], joint_names=JOINT_NAMES_65)

    assert result.frames == frames
    assert result.max_non_foot_joint_displacement_m == pytest.approx(0.0)


def test_solve_foot_lock_clamps_penetrating_foot_joints_even_without_contact_phase():
    frames = [
        _frame(0, left_x=0.000, left_z=-0.030, right_x=1.0, right_z=0.250),
    ]

    result = solve_foot_lock(frames, [], joint_names=JOINT_NAMES_65)

    metrics = measure_contact_metrics(result.frames, [], joint_names=JOINT_NAMES_65)
    assert metrics.penetration.max_penetration_mm == pytest.approx(0.0)
    assert result.frames[0].joints_world[15][2] == pytest.approx(0.0)
    assert result.frames[0].joints_world[0] == frames[0].joints_world[0]

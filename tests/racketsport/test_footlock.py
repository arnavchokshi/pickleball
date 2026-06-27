from __future__ import annotations

import pytest

from threed.racketsport.footlock import (
    FootContactObservation,
    FootKinematics,
    classify_contact_sequence,
    foot_lock_metrics,
    snap_stance_foot,
)


def test_classify_contact_sequence_uses_height_speed_confidence_hysteresis():
    observations = [
        FootContactObservation(height_m=0.030, speed_mps=0.10, confidence=0.90),
        FootContactObservation(height_m=0.018, speed_mps=0.12, confidence=0.90),
        FootContactObservation(height_m=0.045, speed_mps=0.30, confidence=0.90),
        FootContactObservation(height_m=0.055, speed_mps=0.10, confidence=0.90),
        FootContactObservation(height_m=0.010, speed_mps=0.10, confidence=0.20),
    ]

    contacts = classify_contact_sequence(observations)

    assert contacts == [False, True, True, False, False]


def test_snap_stance_foot_projects_z_and_zeros_velocity_without_mutating_input():
    sample = FootKinematics(
        position_xyz=[1.25, -0.50, -0.04],
        velocity_xyz=[0.30, -0.10, 0.20],
        contact=True,
    )

    locked = snap_stance_foot(sample, court_z_m=0.0)

    assert locked is not sample
    assert locked.position_xyz == pytest.approx([1.25, -0.50, 0.0])
    assert locked.velocity_xyz == pytest.approx([0.0, 0.0, 0.0])
    assert sample.position_xyz == pytest.approx([1.25, -0.50, -0.04])
    assert sample.velocity_xyz == pytest.approx([0.30, -0.10, 0.20])


def test_snap_stance_foot_leaves_swing_foot_unchanged():
    sample = FootKinematics(
        position_xyz=[1.25, -0.50, 0.12],
        velocity_xyz=[0.30, -0.10, 0.20],
        contact=False,
    )

    unlocked = snap_stance_foot(sample, court_z_m=0.0)

    assert unlocked.position_xyz == pytest.approx(sample.position_xyz)
    assert unlocked.velocity_xyz == pytest.approx(sample.velocity_xyz)


def test_foot_lock_metrics_report_max_contact_slide_and_penetration():
    samples = [
        FootKinematics(position_xyz=[0.0, 0.0, 0.01], velocity_xyz=[0.0, 0.0, 0.0], contact=True),
        FootKinematics(position_xyz=[0.03, 0.04, -0.02], velocity_xyz=[0.0, 0.0, 0.0], contact=True),
        FootKinematics(position_xyz=[0.20, 0.00, -0.03], velocity_xyz=[0.0, 0.0, 0.0], contact=False),
        FootKinematics(position_xyz=[0.30, 0.00, -0.01], velocity_xyz=[0.0, 0.0, 0.0], contact=True),
        FootKinematics(position_xyz=[0.36, 0.08, 0.02], velocity_xyz=[0.0, 0.0, 0.0], contact=True),
    ]

    metrics = foot_lock_metrics(samples, court_z_m=0.0)

    assert metrics.max_slide_m == pytest.approx(0.10)
    assert metrics.max_penetration_m == pytest.approx(0.03)
    assert metrics.contact_frames == 4
    assert metrics.scaffold == "cpu_foot_lock_primitives_no_smpl_ik"

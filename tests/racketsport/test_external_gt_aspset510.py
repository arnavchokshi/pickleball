from __future__ import annotations

import pytest

from threed.racketsport.eval.body_gate_report import CORE_BODY_JOINT_NAMES
from threed.racketsport.external_gt_aspset510 import (
    ASPSET17J_JOINT_NAMES,
    ASPSET17J_NOT_COMPARABLE_CORE_JOINT_NAMES,
    ASPSET17J_ONLY_JOINT_NAMES,
    SHARED_CORE_JOINT_NAMES,
    build_external_gt_label_samples,
    select_shared_core_joints,
)


def test_shared_core_joint_names_is_exactly_the_limb_intersection() -> None:
    # ASPset-510's native skeleton has no facial landmarks (nose/eyes/ears) and our
    # CORE_BODY_JOINT_NAMES (COCO-17) has no torso/head landmarks (pelvis/spine/neck/
    # head/head_top) -- the only genuinely shared names are the 12 limb joints.
    expected = (
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle",
    )
    assert SHARED_CORE_JOINT_NAMES == expected
    assert set(SHARED_CORE_JOINT_NAMES).issubset(set(CORE_BODY_JOINT_NAMES))
    assert set(SHARED_CORE_JOINT_NAMES).issubset(set(ASPSET17J_JOINT_NAMES))


def test_not_comparable_and_aspset_only_sets_are_disjoint_and_complete() -> None:
    assert set(ASPSET17J_NOT_COMPARABLE_CORE_JOINT_NAMES) == set(CORE_BODY_JOINT_NAMES) - set(SHARED_CORE_JOINT_NAMES)
    assert set(ASPSET17J_ONLY_JOINT_NAMES) == set(ASPSET17J_JOINT_NAMES) - set(SHARED_CORE_JOINT_NAMES)
    # nose/eyes/ears -- ASPset has no facial landmarks at all.
    assert set(ASPSET17J_NOT_COMPARABLE_CORE_JOINT_NAMES) == {
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    }


def test_select_shared_core_joints_orders_by_core_body_joint_names() -> None:
    # ASPset's own c3d marker order (right side first) differs from our CORE order
    # (left side first) -- the selection must reorder, not just filter.
    frame = {name: (float(index), 0.0, 0.0) for index, name in enumerate(ASPSET17J_JOINT_NAMES)}
    selected = select_shared_core_joints(frame)
    assert len(selected) == 12
    for name, point in zip(SHARED_CORE_JOINT_NAMES, selected):
        assert point == frame[name]


def test_select_shared_core_joints_rejects_missing_joint() -> None:
    frame = {name: (0.0, 0.0, 0.0) for name in ASPSET17J_JOINT_NAMES if name != "left_wrist"}
    with pytest.raises(KeyError):
        select_shared_core_joints(frame)


def test_build_external_gt_label_samples_shape_and_provenance() -> None:
    frames_mm = [
        {name: (100.0 * index, 200.0 * index, 300.0 * index) for index, name in enumerate(ASPSET17J_JOINT_NAMES)}
        for _ in range(3)
    ]
    samples = build_external_gt_label_samples(
        frames_joint_positions_mm=frames_mm,
        frame_indices=[10, 11, 12],
        player_id=1,
        clip_id="1e28-0001",
        subject_id="1e28",
    )
    assert len(samples) == 3
    for sample, frame_index in zip(samples, [10, 11, 12]):
        assert sample["frame_index"] == frame_index
        assert sample["player_id"] == 1
        assert sample["accepted"] is True
        assert sample["label_source"] == "external_ground_truth"
        assert sample["joint_names"] == list(SHARED_CORE_JOINT_NAMES)
        assert len(sample["joints_world"]) == 12
        assert sample["external_gt_provenance"]["dataset"] == "aspset-510"
        assert sample["external_gt_provenance"]["clip_id"] == "1e28-0001"
        assert sample["external_gt_provenance"]["subject_id"] == "1e28"
        # mm -> m conversion.
        for value in sample["joints_world"]:
            for coordinate in value:
                assert isinstance(coordinate, float)


def test_build_external_gt_label_samples_converts_millimeters_to_meters() -> None:
    frame_mm = {name: (1000.0, 2000.0, 3000.0) for name in ASPSET17J_JOINT_NAMES}
    samples = build_external_gt_label_samples(
        frames_joint_positions_mm=[frame_mm],
        frame_indices=[0],
        player_id=1,
        clip_id="c",
        subject_id="s",
    )
    for point in samples[0]["joints_world"]:
        assert point == pytest.approx([1.0, 2.0, 3.0])

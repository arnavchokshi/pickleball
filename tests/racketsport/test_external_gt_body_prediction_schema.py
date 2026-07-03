from __future__ import annotations

from threed.racketsport.eval.body_gate_report import CORE_BODY_JOINT_NAMES
from threed.racketsport.external_gt_aspset510 import SHARED_CORE_JOINT_NAMES
from threed.racketsport.external_gt_body_prediction_schema import MHR70_JOINT_NAMES


def test_mhr70_joint_names_has_70_unique_entries() -> None:
    assert len(MHR70_JOINT_NAMES) == 70
    assert len(set(MHR70_JOINT_NAMES)) == 70


def test_mhr70_joint_names_covers_every_core_body_joint_name() -> None:
    mhr_set = set(MHR70_JOINT_NAMES)
    for name in CORE_BODY_JOINT_NAMES:
        assert name in mhr_set, f"{name!r} missing from MHR70_JOINT_NAMES"


def test_mhr70_joint_names_covers_every_shared_core_joint_name() -> None:
    mhr_set = set(MHR70_JOINT_NAMES)
    for name in SHARED_CORE_JOINT_NAMES:
        assert name in mhr_set, f"{name!r} missing from MHR70_JOINT_NAMES"


def test_mhr70_wrists_are_not_at_coco_positions() -> None:
    # Regression guard for the exact bug this module exists to prevent: MHR70 does NOT
    # put wrists at COCO-17 index 9/10 (that's hips in MHR70); positional matching
    # between a raw MHR70 array and a COCO-order GT array would silently swap joints.
    assert MHR70_JOINT_NAMES.index("left_hip") == 9
    assert MHR70_JOINT_NAMES.index("right_hip") == 10
    assert MHR70_JOINT_NAMES.index("right_wrist") == 41
    assert MHR70_JOINT_NAMES.index("left_wrist") == 62

from __future__ import annotations

from threed.racketsport.joint_schema import BODY65_JOINT_NAMES, WHOLEBODY_133_JOINT_NAMES


def test_body65_joint_names_keep_body_feet_hands_and_drop_face() -> None:
    assert len(WHOLEBODY_133_JOINT_NAMES) == 133
    assert len(BODY65_JOINT_NAMES) == 65
    assert "left_wrist" in BODY65_JOINT_NAMES
    assert "right_wrist" in BODY65_JOINT_NAMES
    assert "face_00" not in BODY65_JOINT_NAMES
    assert "left_hand_20" in BODY65_JOINT_NAMES
    assert "right_hand_20" in BODY65_JOINT_NAMES

from __future__ import annotations

import pytest

from threed.racketsport.skeleton3d import (
    SAM3D_BODY_MHR70_SEMANTIC_MAP,
    SemanticJointMap,
    semanticize_skeleton_payload,
    validate_semantic_joint_map,
)


def _generic_sam3d_payload(joint_count: int = 70) -> dict[str, object]:
    return {
        "schema_version": 1,
        "joint_names": [f"sam3dbody_joint_{index:03d}" for index in range(joint_count)],
        "preview_only": True,
        "players": [
            {
                "id": 2,
                "frames": [
                    {
                        "t": 1.0,
                        "joints_world": [[float(index), 0.0, 1.0] for index in range(joint_count)],
                        "joint_conf": [round(0.5 + (index / 1000), 3) for index in range(joint_count)],
                    }
                ],
            }
        ],
    }


def test_semanticize_sam3d_mhr70_payload_reorders_named_joints_and_confidence() -> None:
    payload = semanticize_skeleton_payload(_generic_sam3d_payload())

    assert payload is not None
    assert payload["joint_names"] == list(SAM3D_BODY_MHR70_SEMANTIC_MAP.joints)
    assert payload["semantic_joint_source"] == "sam3d_body_mhr70_v1"
    frame = payload["players"][0]["frames"][0]
    names = payload["joint_names"]
    assert frame["joints_world"][names.index("left_wrist")] == [62.0, 0.0, 1.0]
    assert frame["joints_world"][names.index("right_wrist")] == [41.0, 0.0, 1.0]
    assert frame["joint_conf"][names.index("left_wrist")] == 0.562
    assert frame["joint_conf"][names.index("right_wrist")] == 0.541


def test_semanticize_payload_preserves_already_semantic_payload() -> None:
    payload = {
        "schema_version": 1,
        "joint_names": ["left_shoulder", "right_shoulder", "left_wrist", "right_wrist"],
        "players": [{"id": 1, "frames": [{"t": 0.0, "joints_world": [[0, 0, 0]] * 4, "joint_conf": [0.9] * 4}]}],
    }

    semantic = semanticize_skeleton_payload(payload)

    assert semantic is not None
    assert semantic["joint_names"] == payload["joint_names"]
    assert semantic["semantic_joint_source"] == "already_semantic"


@pytest.mark.parametrize(
    "joint_map, message",
    [
        (SemanticJointMap(name="bad", source_joint_count=70, joints={"left_wrist": 1, "right_wrist": 1}), "duplicate"),
        (SemanticJointMap(name="bad", source_joint_count=70, joints={"left_wrist": 70, "right_wrist": 41}), "out of range"),
        (SemanticJointMap(name="bad", source_joint_count=70, joints={"left_wrist": 62}), "missing required"),
    ],
)
def test_validate_semantic_joint_map_rejects_unsafe_maps(joint_map: SemanticJointMap, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        validate_semantic_joint_map(joint_map)


def test_semanticize_wrong_joint_count_fails_closed() -> None:
    assert semanticize_skeleton_payload(_generic_sam3d_payload(joint_count=69)) is None

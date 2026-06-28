"""Semantic adapters for BODY skeleton payloads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


REQUIRED_SHOT_SEMANTIC_JOINTS = ("left_wrist", "right_wrist")


@dataclass(frozen=True)
class SemanticJointMap:
    name: str
    source_joint_count: int
    joints: Mapping[str, int]


SAM3D_BODY_MHR70_SEMANTIC_MAP = SemanticJointMap(
    name="sam3d_body_mhr70_v1",
    source_joint_count=70,
    joints={
        "left_shoulder": 5,
        "right_shoulder": 6,
        "left_elbow": 7,
        "right_elbow": 8,
        "left_hip": 9,
        "right_hip": 10,
        "left_knee": 11,
        "right_knee": 12,
        "left_ankle": 13,
        "right_ankle": 14,
        "right_wrist": 41,
        "left_wrist": 62,
    },
)


def semanticize_skeleton_payload(
    payload: Mapping[str, Any] | None,
    *,
    joint_map: SemanticJointMap | None = None,
) -> dict[str, Any] | None:
    """Return a semantic-joint payload, or ``None`` when mapping is unsafe."""

    if payload is None:
        return None
    joint_names = payload.get("joint_names")
    if not isinstance(joint_names, Sequence) or isinstance(joint_names, (str, bytes)):
        if joint_map is None:
            return None
        return _apply_joint_map(payload, joint_map)

    normalized_names = [_normalize_name(name) for name in joint_names]
    if all(name in normalized_names for name in REQUIRED_SHOT_SEMANTIC_JOINTS):
        output = dict(payload)
        output["semantic_joint_source"] = str(payload.get("semantic_joint_source", "already_semantic"))
        return output

    selected_map = joint_map
    if selected_map is None and _looks_like_sam3d_mhr70_names(joint_names):
        selected_map = SAM3D_BODY_MHR70_SEMANTIC_MAP
    if selected_map is None:
        return None
    return _apply_joint_map(payload, selected_map)


def validate_semantic_joint_map(joint_map: SemanticJointMap) -> None:
    if not joint_map.name:
        raise ValueError("semantic joint map name is required")
    if joint_map.source_joint_count <= 0:
        raise ValueError("semantic joint map source_joint_count must be positive")
    for required in REQUIRED_SHOT_SEMANTIC_JOINTS:
        if required not in joint_map.joints:
            raise ValueError(f"semantic joint map missing required joint: {required}")
    indexes = list(joint_map.joints.values())
    if len(indexes) != len(set(indexes)):
        raise ValueError("semantic joint map has duplicate source indexes")
    for name, index in joint_map.joints.items():
        if not isinstance(name, str) or not name:
            raise ValueError("semantic joint map joint names must be non-empty strings")
        if isinstance(index, bool) or not isinstance(index, int):
            raise ValueError(f"semantic joint map index for {name} must be an integer")
        if index < 0 or index >= joint_map.source_joint_count:
            raise ValueError(f"semantic joint map index for {name} out of range")


def _apply_joint_map(payload: Mapping[str, Any], joint_map: SemanticJointMap) -> dict[str, Any] | None:
    validate_semantic_joint_map(joint_map)
    joint_names = payload.get("joint_names")
    if isinstance(joint_names, Sequence) and not isinstance(joint_names, (str, bytes)):
        if len(joint_names) != joint_map.source_joint_count:
            return None

    semantic_names = list(joint_map.joints)
    players = payload.get("players", [])
    if not isinstance(players, Sequence) or isinstance(players, (str, bytes)):
        return None

    semantic_players: list[dict[str, Any]] = []
    for player in players:
        if not isinstance(player, Mapping):
            continue
        semantic_frames: list[dict[str, Any]] = []
        frames = player.get("frames", [])
        if not isinstance(frames, Sequence) or isinstance(frames, (str, bytes)):
            continue
        for frame in frames:
            if not isinstance(frame, Mapping):
                continue
            joints = frame.get("joints_world")
            if not isinstance(joints, Sequence) or isinstance(joints, (str, bytes)):
                continue
            if len(joints) != joint_map.source_joint_count:
                return None
            semantic_frame = dict(frame)
            semantic_frame["joints_world"] = [joints[joint_map.joints[name]] for name in semantic_names]
            conf = frame.get("joint_conf")
            if isinstance(conf, Sequence) and not isinstance(conf, (str, bytes)) and len(conf) == joint_map.source_joint_count:
                semantic_frame["joint_conf"] = [conf[joint_map.joints[name]] for name in semantic_names]
            semantic_frames.append(semantic_frame)
        if semantic_frames:
            semantic_player = dict(player)
            semantic_player["frames"] = semantic_frames
            semantic_players.append(semantic_player)

    if not semantic_players:
        return None
    output = dict(payload)
    output["joint_names"] = semantic_names
    output["semantic_joint_source"] = joint_map.name
    output["source_joint_count"] = joint_map.source_joint_count
    output["players"] = semantic_players
    return output


def _looks_like_sam3d_mhr70_names(joint_names: Sequence[Any]) -> bool:
    if len(joint_names) != SAM3D_BODY_MHR70_SEMANTIC_MAP.source_joint_count:
        return False
    return all(str(name) == f"sam3dbody_joint_{index:03d}" for index, name in enumerate(joint_names))


def _normalize_name(value: Any) -> str:
    text = str(value).lower()
    return "_".join(part for part in text.replace("-", "_").split("_") if part)


__all__ = [
    "REQUIRED_SHOT_SEMANTIC_JOINTS",
    "SAM3D_BODY_MHR70_SEMANTIC_MAP",
    "SemanticJointMap",
    "semanticize_skeleton_payload",
    "validate_semantic_joint_map",
]

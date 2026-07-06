"""Shared skeleton joint-name layouts used by BODY and legacy artifacts."""

from __future__ import annotations


BODY_17_JOINT_NAMES: tuple[str, ...] = (
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
)

FOOT_6_JOINT_NAMES: tuple[str, ...] = (
    "left_big_toe",
    "left_small_toe",
    "left_heel",
    "right_big_toe",
    "right_small_toe",
    "right_heel",
)

FACE_68_JOINT_NAMES: tuple[str, ...] = tuple(f"face_{idx:02d}" for idx in range(68))
LEFT_HAND_21_JOINT_NAMES: tuple[str, ...] = tuple(f"left_hand_{idx:02d}" for idx in range(21))
RIGHT_HAND_21_JOINT_NAMES: tuple[str, ...] = tuple(f"right_hand_{idx:02d}" for idx in range(21))

WHOLEBODY_133_JOINT_NAMES: tuple[str, ...] = (
    BODY_17_JOINT_NAMES
    + FOOT_6_JOINT_NAMES
    + FACE_68_JOINT_NAMES
    + LEFT_HAND_21_JOINT_NAMES
    + RIGHT_HAND_21_JOINT_NAMES
)

BODY65_JOINT_INDEXES: tuple[int, ...] = tuple(range(17)) + tuple(range(17, 23)) + tuple(range(91, 133))
BODY65_JOINT_NAMES: tuple[str, ...] = tuple(WHOLEBODY_133_JOINT_NAMES[idx] for idx in BODY65_JOINT_INDEXES)
SUPPORT_FOOT_JOINT_NAMES = frozenset(FOOT_6_JOINT_NAMES)


__all__ = [
    "BODY65_JOINT_INDEXES",
    "BODY65_JOINT_NAMES",
    "BODY_17_JOINT_NAMES",
    "FACE_68_JOINT_NAMES",
    "FOOT_6_JOINT_NAMES",
    "LEFT_HAND_21_JOINT_NAMES",
    "RIGHT_HAND_21_JOINT_NAMES",
    "SUPPORT_FOOT_JOINT_NAMES",
    "WHOLEBODY_133_JOINT_NAMES",
]

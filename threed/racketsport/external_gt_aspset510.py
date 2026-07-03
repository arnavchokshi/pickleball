"""ASPset-510 joint-schema mapping for the external-ground-truth BODY validation lane.

Owner decision (2026-07-02, binding): validate the BODY pipeline against a public 3D-GT
dataset via the gate's `external_ground_truth` label source. Dataset choice, license, and
verified download mechanism are recorded in
`runs/body_external_gt_20260702T*/DATASET_SELECTION.md`; this module implements only the
joint-schema mapping between ASPset-510's native skeleton and this project's
`CORE_BODY_JOINT_NAMES` (17-joint core-body set, see
`threed.racketsport.eval.body_gate_report`).

**The honest joint-coverage caveat (must be surfaced anywhere this data is used).**
ASPset-510's native ``aspset_17j`` skeleton (`anibali/posekit`
`Aspset17jSkeleton`, confirmed against a real downloaded `.c3d` file for clip
``1e28-0001``) has **no facial landmarks at all** -- it uses
``head``/``head_top`` instead of ``nose``/``left_eye``/``right_eye``/``left_ear``/
``right_ear``. Our own core-17 set (COCO-17-style) has no ``pelvis``/``spine``/``neck``/
``head``/``head_top``. The intersection -- the only joints genuinely comparable between
the two schemas -- is exactly the **12 limb joints**: both shoulders, elbows, wrists,
hips, knees, ankles. ``SHARED_CORE_JOINT_NAMES`` is that 12-joint set, always emitted in
`CORE_BODY_JOINT_NAMES` order (left-first) even though ASPset's own raw ``.c3d`` marker
order is right-first -- see `select_shared_core_joints`.

This means external-GT world-MPJPE scored via this dataset covers **12 of the 17**
core-body joints (limbs only, no head/face) -- a real, meaningful, but partial subset.
This is reported honestly rather than silently treated as "the full core-17."
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from threed.racketsport.eval.body_gate_report import CORE_BODY_JOINT_NAMES

DATASET_NAME = "aspset-510"
DATASET_LICENSE = "CC0-1.0"

# Native ASPset-510 / posekit `Aspset17jSkeleton` joint order, confirmed against a real
# downloaded `.c3d` file (`ASPset-510/test/joints_3d/1e28/1e28-0001.c3d`, ezc3d
# POINT.LABELS): right-side-first, then left-side, then torso/head.
ASPSET17J_JOINT_NAMES: tuple[str, ...] = (
    "right_ankle", "right_knee", "right_hip",
    "right_wrist", "right_elbow", "right_shoulder",
    "left_ankle", "left_knee", "left_hip",
    "left_wrist", "left_elbow", "left_shoulder",
    "head_top", "head", "neck",
    "spine", "pelvis",
)

# The only joints present in *both* schemas -- always reported in CORE_BODY_JOINT_NAMES
# (left-first COCO) order, not ASPset's native (right-first) order.
SHARED_CORE_JOINT_NAMES: tuple[str, ...] = tuple(
    name for name in CORE_BODY_JOINT_NAMES if name in set(ASPSET17J_JOINT_NAMES)
)

# Our core-17 joints ASPset-510 cannot supply ground truth for (facial landmarks).
ASPSET17J_NOT_COMPARABLE_CORE_JOINT_NAMES: tuple[str, ...] = tuple(
    name for name in CORE_BODY_JOINT_NAMES if name not in set(SHARED_CORE_JOINT_NAMES)
)

# ASPset-510 joints with no equivalent in our core-17 schema (torso/head landmarks).
ASPSET17J_ONLY_JOINT_NAMES: tuple[str, ...] = tuple(
    name for name in ASPSET17J_JOINT_NAMES if name not in set(SHARED_CORE_JOINT_NAMES)
)

MILLIMETERS_PER_METER = 1000.0


def select_shared_core_joints(
    frame_joint_positions: Mapping[str, Sequence[float]],
) -> list[tuple[float, float, float]]:
    """Reorder a per-frame ASPset joint-name -> xyz mapping into SHARED_CORE_JOINT_NAMES order.

    Raises ``KeyError`` if any of the 12 shared joints is missing, so a partially
    occluded/incomplete mocap frame fails loudly instead of silently shipping a short
    or misaligned vector.
    """

    return [tuple(float(v) for v in frame_joint_positions[name]) for name in SHARED_CORE_JOINT_NAMES]


def build_external_gt_label_samples(
    *,
    frames_joint_positions_mm: Sequence[Mapping[str, Sequence[float]]],
    frame_indices: Sequence[int],
    player_id: int,
    clip_id: str,
    subject_id: str,
    camera_id: str = "",
) -> list[dict[str, Any]]:
    """Build `body_world_joints.json`-style sample dicts from raw ASPset mocap frames.

    Input joint positions are in millimeters (ASPset-510's native `.c3d` unit, confirmed
    from the real downloaded file's `POINT.UNITS` field); output ``joints_world`` is in
    meters (matching this project's own `body_world_joints.json` convention).
    """

    if len(frames_joint_positions_mm) != len(frame_indices):
        raise ValueError(
            f"frames_joint_positions_mm length {len(frames_joint_positions_mm)} != "
            f"frame_indices length {len(frame_indices)}"
        )
    samples: list[dict[str, Any]] = []
    for frame_joints_mm, frame_index in zip(frames_joint_positions_mm, frame_indices):
        joints_m = [
            [coordinate / MILLIMETERS_PER_METER for coordinate in point]
            for point in select_shared_core_joints(frame_joints_mm)
        ]
        samples.append(
            {
                "sample_id": f"{clip_id}_frame_{frame_index:06d}_player_{player_id}",
                "frame_index": int(frame_index),
                "player_id": int(player_id),
                "accepted": True,
                "label_source": "external_ground_truth",
                "joint_names": list(SHARED_CORE_JOINT_NAMES),
                "joints_world": joints_m,
                "external_gt_provenance": {
                    "dataset": DATASET_NAME,
                    "license": DATASET_LICENSE,
                    "clip_id": clip_id,
                    "subject_id": subject_id,
                    "camera_id": camera_id,
                    "not_comparable_core_joint_names": list(ASPSET17J_NOT_COMPARABLE_CORE_JOINT_NAMES),
                },
            }
        )
    return samples


__all__ = [
    "ASPSET17J_JOINT_NAMES",
    "ASPSET17J_NOT_COMPARABLE_CORE_JOINT_NAMES",
    "ASPSET17J_ONLY_JOINT_NAMES",
    "DATASET_LICENSE",
    "DATASET_NAME",
    "MILLIMETERS_PER_METER",
    "SHARED_CORE_JOINT_NAMES",
    "build_external_gt_label_samples",
    "select_shared_core_joints",
]

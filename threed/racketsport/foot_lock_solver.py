"""Deterministic foot-lock solver for measured world-frame skeletons.

This is not full biomechanical IK. During detected stance phases it applies a
small root translation, then a weighted leg-chain residual so the stance foot's
low contact point is pinned to its phase anchor. The tradeoff is explicit:
large residuals can look cosmetic and can create knee pops, so the result
reports non-foot joint displacement instead of pretending the correction is
physically solved.
"""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass
from typing import Sequence

from threed.racketsport.foot_contact import (
    ContactPhase,
    FootJointIndices,
    SkeletonFrame,
    foot_contact_point,
    resolve_foot_joint_indices,
)


@dataclass(frozen=True)
class FootLockSolverSettings:
    root_translation_weight: float = 0.50
    knee_residual_weight: float = 0.35
    hip_residual_weight: float = 0.15
    court_z_m: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class FrameCorrection:
    player_id: str | int
    frame_index: int
    active_contacts: list[dict[str, object]]
    root_delta_xyz: tuple[float, float, float]
    max_any_joint_displacement_m: float
    max_non_foot_joint_displacement_m: float

    def to_dict(self) -> dict[str, object]:
        return {
            "player_id": self.player_id,
            "frame_index": self.frame_index,
            "active_contacts": self.active_contacts,
            "root_delta_xyz": list(self.root_delta_xyz),
            "max_any_joint_displacement_m": self.max_any_joint_displacement_m,
            "max_non_foot_joint_displacement_m": self.max_non_foot_joint_displacement_m,
        }


@dataclass(frozen=True)
class FootLockResult:
    frames: list[SkeletonFrame]
    frame_corrections: list[FrameCorrection]
    max_any_joint_displacement_m: float
    max_non_foot_joint_displacement_m: float
    settings: FootLockSolverSettings

    def to_dict(self) -> dict[str, object]:
        return {
            "frame_corrections": [correction.to_dict() for correction in self.frame_corrections],
            "max_any_joint_displacement_m": self.max_any_joint_displacement_m,
            "max_non_foot_joint_displacement_m": self.max_non_foot_joint_displacement_m,
            "settings": self.settings.to_dict(),
        }


def solve_foot_lock(
    frames: Sequence[SkeletonFrame],
    phases: Sequence[ContactPhase],
    *,
    joint_names: Sequence[str],
    settings: FootLockSolverSettings = FootLockSolverSettings(),
) -> FootLockResult:
    _validate_settings(settings)
    if not frames:
        return FootLockResult(
            frames=[],
            frame_corrections=[],
            max_any_joint_displacement_m=0.0,
            max_non_foot_joint_displacement_m=0.0,
            settings=settings,
        )
    indices = resolve_foot_joint_indices(joint_names, joint_count=len(frames[0].joints_world))
    eligible_phases = [phase for phase in phases if not _phase_is_weak(phase)]
    active_by_frame = _active_phases_by_frame(eligible_phases)
    corrected_frames: list[SkeletonFrame] = []
    corrections: list[FrameCorrection] = []
    max_any = 0.0
    max_non_foot = 0.0

    for frame in frames:
        original_joints = copy.deepcopy(frame.joints_world)
        joints = copy.deepcopy(frame.joints_world)
        active = active_by_frame.get((frame.player_id, frame.frame_index), [])
        root_delta = _root_delta(frame, active, indices, settings)
        if active:
            _translate_joints(joints, root_delta)
            for phase in active:
                _apply_leg_residual(joints, phase, indices, joint_names, settings)
        _clamp_foot_penetration(joints, indices, court_z_m=settings.court_z_m)

        frame_any, frame_non_foot = _frame_displacement(original_joints, joints, indices.all())
        max_any = max(max_any, frame_any)
        max_non_foot = max(max_non_foot, frame_non_foot)
        corrected = SkeletonFrame(
            player_id=frame.player_id,
            frame_index=frame.frame_index,
            t=frame.t,
            joints_world=joints,
            joint_conf=copy.deepcopy(frame.joint_conf),
            source=frame.source,
        )
        corrected_frames.append(corrected)
        if active or frame_any > 0:
            corrections.append(
                FrameCorrection(
                    player_id=frame.player_id,
                    frame_index=frame.frame_index,
                    active_contacts=[_contact_metadata(phase) for phase in active],
                    root_delta_xyz=root_delta,
                    max_any_joint_displacement_m=frame_any,
                    max_non_foot_joint_displacement_m=frame_non_foot,
                )
            )

    return FootLockResult(
        frames=corrected_frames,
        frame_corrections=corrections,
        max_any_joint_displacement_m=max_any,
        max_non_foot_joint_displacement_m=max_non_foot,
        settings=settings,
    )


def _root_delta(
    frame: SkeletonFrame,
    active: Sequence[ContactPhase],
    indices: FootJointIndices,
    settings: FootLockSolverSettings,
) -> tuple[float, float, float]:
    if not active or settings.root_translation_weight == 0:
        return (0.0, 0.0, 0.0)
    deltas: list[tuple[float, float]] = []
    for phase in active:
        current = foot_contact_point(frame, indices.for_foot(phase.foot))
        anchor = phase.anchor_position_xyz
        deltas.append((anchor[0] - current[0], anchor[1] - current[1]))
    dx = sum(delta[0] for delta in deltas) / len(deltas) * settings.root_translation_weight
    dy = sum(delta[1] for delta in deltas) / len(deltas) * settings.root_translation_weight
    return (dx, dy, 0.0)


def _apply_leg_residual(
    joints: list[list[float]],
    phase: ContactPhase,
    indices: FootJointIndices,
    joint_names: Sequence[str],
    settings: FootLockSolverSettings,
) -> None:
    foot_indices = indices.for_foot(phase.foot)
    frame = SkeletonFrame(
        player_id=phase.player_id,
        frame_index=phase.start_frame_index,
        t=None,
        joints_world=joints,
    )
    current = foot_contact_point(frame, foot_indices)
    anchor = phase.anchor_position_xyz
    residual = (
        anchor[0] - current[0],
        anchor[1] - current[1],
        settings.court_z_m - current[2],
    )
    weights = {index: 1.0 for index in foot_indices}
    name_to_index = {name: index for index, name in enumerate(_effective_joint_names(joint_names, len(joints)))}
    knee_index = name_to_index.get(f"{phase.foot}_knee")
    hip_index = name_to_index.get(f"{phase.foot}_hip")
    if knee_index is not None:
        weights.setdefault(knee_index, settings.knee_residual_weight)
    if hip_index is not None:
        weights.setdefault(hip_index, settings.hip_residual_weight)
    for index, weight in weights.items():
        if index >= len(joints):
            continue
        joints[index][0] += residual[0] * weight
        joints[index][1] += residual[1] * weight
        joints[index][2] += residual[2] * weight


def _translate_joints(joints: list[list[float]], delta_xyz: tuple[float, float, float]) -> None:
    if delta_xyz == (0.0, 0.0, 0.0):
        return
    for joint in joints:
        joint[0] += delta_xyz[0]
        joint[1] += delta_xyz[1]
        joint[2] += delta_xyz[2]


def _clamp_foot_penetration(joints: list[list[float]], indices: FootJointIndices, *, court_z_m: float) -> None:
    for index in indices.all():
        if index < len(joints) and joints[index][2] < court_z_m:
            joints[index][2] = court_z_m


def _frame_displacement(
    original: Sequence[Sequence[float]],
    corrected: Sequence[Sequence[float]],
    foot_indices: Sequence[int],
) -> tuple[float, float]:
    foot_set = set(foot_indices)
    max_any = 0.0
    max_non_foot = 0.0
    for index, (before, after) in enumerate(zip(original, corrected, strict=True)):
        distance = math.sqrt(
            (after[0] - before[0]) ** 2 + (after[1] - before[1]) ** 2 + (after[2] - before[2]) ** 2
        )
        max_any = max(max_any, distance)
        if index not in foot_set:
            max_non_foot = max(max_non_foot, distance)
    return max_any, max_non_foot


def _active_phases_by_frame(phases: Sequence[ContactPhase]) -> dict[tuple[str | int, int], list[ContactPhase]]:
    active: dict[tuple[str | int, int], list[ContactPhase]] = {}
    for phase in phases:
        for frame_index in phase.frame_indices:
            active.setdefault((phase.player_id, frame_index), []).append(phase)
    return active


def _contact_metadata(phase: ContactPhase) -> dict[str, object]:
    return {
        "foot": phase.foot,
        "start_frame_index": phase.start_frame_index,
        "end_frame_index": phase.end_frame_index,
        "anchor_position_xyz": list(phase.anchor_position_xyz),
        "foot_assignment": phase.foot_assignment,
        "weak": phase.weak,
        "demoted": phase.demoted,
        "source": phase.source,
    }


def _phase_is_weak(phase: ContactPhase) -> bool:
    return bool(phase.weak or phase.demoted or phase.foot_assignment == "bilateral_from_player_stance")


def _effective_joint_names(joint_names: Sequence[str], joint_count: int) -> tuple[str, ...]:
    if joint_count == 70:
        from threed.racketsport.external_gt_body_prediction_schema import MHR70_JOINT_NAMES

        return MHR70_JOINT_NAMES
    return tuple(joint_names)


def _validate_settings(settings: FootLockSolverSettings) -> None:
    if settings.root_translation_weight < 0 or settings.root_translation_weight > 1:
        raise ValueError("root_translation_weight must be in [0, 1]")
    if settings.knee_residual_weight < 0 or settings.knee_residual_weight > 1:
        raise ValueError("knee_residual_weight must be in [0, 1]")
    if settings.hip_residual_weight < 0 or settings.hip_residual_weight > 1:
        raise ValueError("hip_residual_weight must be in [0, 1]")


__all__ = [
    "FootLockResult",
    "FootLockSolverSettings",
    "FrameCorrection",
    "solve_foot_lock",
]

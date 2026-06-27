"""CPU-only physics-refinement scaffold primitives.

This module prepares and validates FOOT-2 inputs/outputs without running
PhysPT, MuJoCo, MJX, PHC/PULSE, or MultiPhys. It exists so downstream eval and
orchestration code can exchange deterministic artifacts while the real physics
stack remains unclaimed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral
from typing import Any, Sequence


SCAFFOLD_NOTE = "cpu_physics_refinement_scaffold_no_sim"
SUPPORTED_MODES = frozenset({"auto", "cpu_fallback", "physpt", "mujoco_mjx_required"})


@dataclass(frozen=True)
class ContactWindow:
    start_frame: int
    end_frame: int
    player_id: str
    reason: str

    def __post_init__(self) -> None:
        start_frame = _non_negative_int(self.start_frame, name="start_frame")
        end_frame = _non_negative_int(self.end_frame, name="end_frame")
        if end_frame < start_frame:
            raise ValueError("end_frame must be greater than or equal to start_frame")
        if not self.player_id:
            raise ValueError("player_id is required")
        if not self.reason:
            raise ValueError("reason is required")
        object.__setattr__(self, "start_frame", start_frame)
        object.__setattr__(self, "end_frame", end_frame)
        object.__setattr__(self, "player_id", str(self.player_id))
        object.__setattr__(self, "reason", str(self.reason))


@dataclass(frozen=True)
class MotionRefinementRequest:
    clip_id: str
    frame_rate_hz: float
    total_frames: int
    player_ids: tuple[str, ...]
    requested_mode: str = "auto"

    def __post_init__(self) -> None:
        if not self.clip_id:
            raise ValueError("clip_id is required")
        frame_rate_hz = _finite_float(self.frame_rate_hz, name="frame_rate_hz")
        total_frames = _positive_int(self.total_frames, name="total_frames")
        if isinstance(self.player_ids, (str, bytes)):
            raise ValueError("player_ids must be a sequence")
        player_ids = tuple(str(player_id) for player_id in self.player_ids)
        if frame_rate_hz <= 0.0:
            raise ValueError("frame_rate_hz must be positive")
        if not player_ids:
            raise ValueError("player_ids must not be empty")
        if len(set(player_ids)) != len(player_ids):
            raise ValueError("player_ids must be unique")
        if any(not player_id for player_id in player_ids):
            raise ValueError("player_ids must not contain empty values")
        if self.requested_mode not in SUPPORTED_MODES:
            raise ValueError(f"unsupported requested_mode: {self.requested_mode}")
        object.__setattr__(self, "clip_id", str(self.clip_id))
        object.__setattr__(self, "frame_rate_hz", frame_rate_hz)
        object.__setattr__(self, "total_frames", total_frames)
        object.__setattr__(self, "player_ids", player_ids)
        object.__setattr__(self, "requested_mode", str(self.requested_mode))


@dataclass(frozen=True)
class FootContactSample:
    frame_index: int
    player_id: str
    foot: str
    position_xyz: tuple[float, float, float]
    contact: bool

    def __post_init__(self) -> None:
        frame_index = _non_negative_int(self.frame_index, name="frame_index")
        if not self.player_id:
            raise ValueError("player_id is required")
        if not self.foot:
            raise ValueError("foot is required")
        if not isinstance(self.contact, bool):
            raise ValueError("contact must be a bool")
        object.__setattr__(self, "frame_index", frame_index)
        object.__setattr__(self, "player_id", str(self.player_id))
        object.__setattr__(self, "foot", str(self.foot))
        object.__setattr__(self, "position_xyz", _validate_vector3(self.position_xyz, name="position_xyz"))


@dataclass(frozen=True)
class PlayerRootSample:
    frame_index: int
    player_id: str
    center_xyz: tuple[float, float, float]
    radius_m: float

    def __post_init__(self) -> None:
        frame_index = _non_negative_int(self.frame_index, name="frame_index")
        if not self.player_id:
            raise ValueError("player_id is required")
        radius_m = _finite_float(self.radius_m, name="radius_m")
        if radius_m < 0.0:
            raise ValueError("radius_m must be non-negative")
        object.__setattr__(self, "frame_index", frame_index)
        object.__setattr__(self, "player_id", str(self.player_id))
        object.__setattr__(self, "center_xyz", _validate_vector3(self.center_xyz, name="center_xyz"))
        object.__setattr__(self, "radius_m", radius_m)


@dataclass(frozen=True)
class ConstraintSummary:
    contact_frames: int
    max_contact_slide_m: float
    max_floor_penetration_m: float
    inter_player_penetration_frames: int
    max_inter_player_penetration_m: float
    scaffold: str = SCAFFOLD_NOTE


@dataclass(frozen=True)
class ExecutionPlan:
    mode: str
    requires_mjx: bool
    will_run_mjx: bool
    reason: str
    scaffold: str = SCAFFOLD_NOTE


def select_refinement_windows(
    windows: Sequence[ContactWindow],
    *,
    total_frames: int,
    pad_frames: int,
) -> tuple[ContactWindow, ...]:
    """Pad, clamp, sort, and merge contact-driven refinement windows."""

    total_frames = _positive_int(total_frames, name="total_frames")
    pad_frames = _non_negative_int(pad_frames, name="pad_frames")

    padded = [
        ContactWindow(
            start_frame=max(0, window.start_frame - pad_frames),
            end_frame=min(total_frames - 1, window.end_frame + pad_frames),
            player_id=window.player_id,
            reason=window.reason,
        )
        for window in windows
    ]
    padded.sort(key=lambda window: (window.player_id, window.start_frame, window.end_frame, window.reason))

    merged: list[ContactWindow] = []
    for window in padded:
        if not merged:
            merged.append(window)
            continue

        previous = merged[-1]
        if window.player_id == previous.player_id and window.start_frame <= previous.end_frame + 1:
            merged[-1] = ContactWindow(
                start_frame=previous.start_frame,
                end_frame=max(previous.end_frame, window.end_frame),
                player_id=previous.player_id,
                reason=_merge_reasons(previous.reason, window.reason),
            )
        else:
            merged.append(window)

    return tuple(sorted(merged, key=lambda window: (window.start_frame, window.end_frame, window.player_id)))


def summarize_constraints(
    foot_samples: Sequence[FootContactSample],
    player_roots: Sequence[PlayerRootSample] = (),
    *,
    floor_z_m: float = 0.0,
) -> ConstraintSummary:
    """Summarize floor/contact and inter-player constraints without simulation."""

    floor_z_m = _finite_float(floor_z_m, name="floor_z_m")
    contact_frames = 0
    max_contact_slide_m = 0.0
    max_floor_penetration_m = 0.0
    previous_contact: dict[tuple[str, str], FootContactSample] = {}

    ordered_foot_samples = sorted(
        foot_samples,
        key=lambda sample: (sample.player_id, sample.foot, sample.frame_index),
    )
    for sample in ordered_foot_samples:
        max_floor_penetration_m = max(max_floor_penetration_m, max(0.0, floor_z_m - sample.position_xyz[2]))
        key = (sample.player_id, sample.foot)

        if not sample.contact:
            previous_contact.pop(key, None)
            continue

        contact_frames += 1
        previous = previous_contact.get(key)
        if previous is not None:
            max_contact_slide_m = max(max_contact_slide_m, _horizontal_distance(previous.position_xyz, sample.position_xyz))
        previous_contact[key] = sample

    penetration_frames, max_inter_player_penetration_m = _summarize_player_penetration(player_roots)

    return ConstraintSummary(
        contact_frames=contact_frames,
        max_contact_slide_m=max_contact_slide_m,
        max_floor_penetration_m=max_floor_penetration_m,
        inter_player_penetration_frames=penetration_frames,
        max_inter_player_penetration_m=max_inter_player_penetration_m,
    )


def choose_execution_plan(request: MotionRefinementRequest, *, mjx_available: bool) -> ExecutionPlan:
    """Choose a scaffold execution plan without importing or running MJX."""

    mode = request.requested_mode
    if mode == "mujoco_mjx_required":
        if mjx_available:
            return ExecutionPlan(
                mode="mujoco_mjx_required",
                requires_mjx=True,
                will_run_mjx=False,
                reason="MJX runtime is available, but this CPU scaffold does not run simulation.",
            )
        return ExecutionPlan(
            mode="blocked_mjx_required",
            requires_mjx=True,
            will_run_mjx=False,
            reason="MJX runtime is required for flagship FOOT-2 physics; scaffold did not run MJX.",
        )

    if mode == "physpt":
        return ExecutionPlan(
            mode="physpt_scaffold",
            requires_mjx=False,
            will_run_mjx=False,
            reason="PhysPT requested, but this CPU scaffold only packages inputs and constraint summaries.",
        )

    return ExecutionPlan(
        mode="cpu_fallback",
        requires_mjx=False,
        will_run_mjx=False,
        reason="Using deterministic CPU fallback scaffold; no physics model or simulator was run.",
    )


def package_refinement_artifact(
    request: MotionRefinementRequest,
    windows: Sequence[ContactWindow],
    summary: ConstraintSummary,
    plan: ExecutionPlan,
) -> dict[str, object]:
    """Build a schema-friendly artifact for downstream FOOT-2 eval gates."""

    return {
        "clip_id": request.clip_id,
        "frame_rate_hz": request.frame_rate_hz,
        "total_frames": request.total_frames,
        "player_ids": list(request.player_ids),
        "skate_free": False,
        "physics": "cpu_fallback_scaffold",
        "foot2_done": False,
        "must_not_mark_done_verified": True,
        "scaffold": SCAFFOLD_NOTE,
        "requested_mode": request.requested_mode,
        "refinement_windows": [_window_to_dict(window) for window in windows],
        "constraint_summary": _summary_to_dict(summary),
        "execution_plan": _plan_to_dict(plan),
    }


def _summarize_player_penetration(player_roots: Sequence[PlayerRootSample]) -> tuple[int, float]:
    by_frame: dict[int, list[PlayerRootSample]] = {}
    for root in player_roots:
        by_frame.setdefault(root.frame_index, []).append(root)

    penetration_frames = 0
    max_penetration_m = 0.0
    for roots in by_frame.values():
        frame_has_penetration = False
        ordered = sorted(roots, key=lambda root: root.player_id)
        for left_index, left in enumerate(ordered):
            for right in ordered[left_index + 1 :]:
                center_distance_m = _horizontal_distance(left.center_xyz, right.center_xyz)
                penetration_m = max(0.0, left.radius_m + right.radius_m - center_distance_m)
                if penetration_m > 0.0:
                    frame_has_penetration = True
                    max_penetration_m = max(max_penetration_m, penetration_m)
        if frame_has_penetration:
            penetration_frames += 1

    return penetration_frames, max_penetration_m


def _merge_reasons(left: str, right: str) -> str:
    reasons = []
    for reason in (*left.split("+"), *right.split("+")):
        if reason not in reasons:
            reasons.append(reason)
    return "+".join(reasons)


def _horizontal_distance(left_xyz: Sequence[float], right_xyz: Sequence[float]) -> float:
    return math.hypot(left_xyz[0] - right_xyz[0], left_xyz[1] - right_xyz[1])


def _window_to_dict(window: ContactWindow) -> dict[str, object]:
    return {
        "start_frame": window.start_frame,
        "end_frame": window.end_frame,
        "player_id": window.player_id,
        "reason": window.reason,
    }


def _summary_to_dict(summary: ConstraintSummary) -> dict[str, object]:
    return {
        "contact_frames": summary.contact_frames,
        "max_contact_slide_m": summary.max_contact_slide_m,
        "max_floor_penetration_m": summary.max_floor_penetration_m,
        "inter_player_penetration_frames": summary.inter_player_penetration_frames,
        "max_inter_player_penetration_m": summary.max_inter_player_penetration_m,
        "scaffold": summary.scaffold,
    }


def _plan_to_dict(plan: ExecutionPlan) -> dict[str, object]:
    return {
        "mode": plan.mode,
        "requires_mjx": plan.requires_mjx,
        "will_run_mjx": plan.will_run_mjx,
        "reason": plan.reason,
        "scaffold": plan.scaffold,
    }


def _validate_vector3(values: Sequence[float], *, name: str) -> tuple[float, float, float]:
    if len(values) != 3:
        raise ValueError(f"{name} must be a 3-vector")
    return (
        _finite_float(values[0], name=f"{name}/0"),
        _finite_float(values[1], name=f"{name}/1"),
        _finite_float(values[2], name=f"{name}/2"),
    )


def _non_negative_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a non-negative integer")
    value_int = int(value)
    if value_int < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value_int


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be positive")
    value_int = int(value)
    if value_int <= 0:
        raise ValueError(f"{name} must be positive")
    return value_int


def _finite_float(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number

"""Adaptive BODY compute scheduling helpers."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

from .best_stack import load_best_stack_manifest
from .schemas import Tracks
from .trust_band import TRUST_BADGES


ARTIFACT_TYPE = "racketsport_body_compute_execution"
SCHEMA_VERSION = 1
DEFAULT_MAX_TRACK_SPEED_FOR_BODY_MPS = 10.0
DEFAULT_MAX_BBOX_CENTER_SPEED_FOR_BODY_DIAG_S = 30.0
DEFAULT_MAX_BBOX_CENTER_STEP_FOR_BODY_PX = 300.0
DEFAULT_MAX_TRACK_WORLD_STEP_FOR_BBOX_JITTER_M = 3.5
DEFAULT_BODY_SKELETON_STRIDE = int(load_best_stack_manifest().value("body.skeleton_stride"))
UNSAFE_TRACK_CONTINUITY_REASON = "unsafe_track_continuity"
MISSING_FRAME_COMPUTE_PLAN_REASON = "missing_frame_compute_plan"
BODY_SKELETON_STRIDE_SKIP_REASON = "body_skeleton_stride_skip"
SAM3D_BODY_JOINTS_ALL_TRACKED_REASON = "sam3d_body_joints_all_tracked"
SAM3D_BODY_JOINTS_SOURCE = "sam3d_body_joints"
TIER2_BODY_JOINTS_TIER = "tier2_body_joints"
TIER2_BODY_JOINTS_REPRESENTATION = "body_joints"
TIER1_MESH_REPRESENTATION = "world_mesh"
CONTACT_DENSE_PROFILE_MODE = "contact_dense"
DEFAULT_CONTACT_DENSE_PAD_S = 0.5
CONTACT_DENSE_HITTER_WINDOW_REASON = "contact_dense_hitter_window"
CONTACT_DENSE_TRIGGER_REASONS = frozenset(
    {
        "ball_aware_contact",
        "high_confidence_swing",
        "contact_window",
        "reviewed_contact_targeted_body",
    }
)
UNIFORM_MESH_SELECTION_REASON = "uniform_mesh_coverage"


def build_body_compute_execution(
    tracks: Tracks,
    *,
    frame_plan_path: str | Path | None = None,
    max_frames: int | None = None,
    include_tier2_body_joints: bool = False,
    contact_dense_pad_s: float = DEFAULT_CONTACT_DENSE_PAD_S,
    skeleton_stride: int = DEFAULT_BODY_SKELETON_STRIDE,
    max_track_speed_for_body_mps: float = DEFAULT_MAX_TRACK_SPEED_FOR_BODY_MPS,
    max_bbox_center_speed_for_body_diag_s: float = DEFAULT_MAX_BBOX_CENTER_SPEED_FOR_BODY_DIAG_S,
    max_bbox_center_step_for_body_px: float = DEFAULT_MAX_BBOX_CENTER_STEP_FOR_BODY_PX,
    max_track_world_step_for_bbox_jitter_m: float = DEFAULT_MAX_TRACK_WORLD_STEP_FOR_BBOX_JITTER_M,
) -> dict[str, Any]:
    """Return the BODY frames that should invoke deep mesh compute.

    If ``frame_plan_path`` exists, only frames inside ``deep_mesh_windows`` are
    scheduled. Other plan frames are preserved as skipped review records. When
    no plan exists, BODY fails closed by skipping all mesh work; Lane B must be
    driven by an explicit frame plan.
    """

    if max_frames is not None and max_frames < 0:
        raise ValueError("max_frames must be non-negative")
    if contact_dense_pad_s < 0.0:
        raise ValueError("contact_dense_pad_s must be non-negative")
    if skeleton_stride <= 0:
        raise ValueError("skeleton_stride must be positive")
    if max_track_speed_for_body_mps <= 0.0:
        raise ValueError("max_track_speed_for_body_mps must be positive")
    if max_bbox_center_speed_for_body_diag_s <= 0.0:
        raise ValueError("max_bbox_center_speed_for_body_diag_s must be positive")
    if max_bbox_center_step_for_body_px <= 0.0:
        raise ValueError("max_bbox_center_step_for_body_px must be positive")
    if max_track_world_step_for_bbox_jitter_m <= 0.0:
        raise ValueError("max_track_world_step_for_bbox_jitter_m must be positive")

    track_lookup = _track_lookup(tracks)
    safe_track_lookup, track_continuity = _body_safe_track_lookup(
        tracks,
        track_lookup=track_lookup,
        max_track_speed_for_body_mps=max_track_speed_for_body_mps,
        max_bbox_center_speed_for_body_diag_s=max_bbox_center_speed_for_body_diag_s,
        max_bbox_center_step_for_body_px=max_bbox_center_step_for_body_px,
        max_track_world_step_for_bbox_jitter_m=max_track_world_step_for_bbox_jitter_m,
    )
    plan_path = Path(frame_plan_path) if frame_plan_path is not None else None
    if plan_path is not None and plan_path.is_file():
        return _execution_from_frame_plan(
            tracks,
            plan_path=plan_path,
            track_lookup=track_lookup,
            safe_track_lookup=safe_track_lookup,
            track_continuity=track_continuity,
            max_frames=max_frames,
            include_tier2_body_joints=include_tier2_body_joints,
            contact_dense_pad_s=contact_dense_pad_s,
            skeleton_stride=int(skeleton_stride),
        )
    return _execution_without_frame_plan(
        tracks,
        track_lookup=track_lookup,
        safe_track_lookup=safe_track_lookup,
        track_continuity=track_continuity,
        include_tier2_body_joints=include_tier2_body_joints,
        skeleton_stride=int(skeleton_stride),
    )


def build_empty_body_compute_execution(
    tracks: Tracks,
    *,
    mode: str,
    source_plan: str | None,
    max_track_speed_for_body_mps: float = DEFAULT_MAX_TRACK_SPEED_FOR_BODY_MPS,
    max_bbox_center_speed_for_body_diag_s: float = DEFAULT_MAX_BBOX_CENTER_SPEED_FOR_BODY_DIAG_S,
    max_bbox_center_step_for_body_px: float = DEFAULT_MAX_BBOX_CENTER_STEP_FOR_BODY_PX,
    max_track_world_step_for_bbox_jitter_m: float = DEFAULT_MAX_TRACK_WORLD_STEP_FOR_BBOX_JITTER_M,
    skeleton_stride: int = DEFAULT_BODY_SKELETON_STRIDE,
) -> dict[str, Any]:
    track_lookup = _track_lookup(tracks)
    _safe_track_lookup, track_continuity = _body_safe_track_lookup(
        tracks,
        track_lookup=track_lookup,
        max_track_speed_for_body_mps=max_track_speed_for_body_mps,
        max_bbox_center_speed_for_body_diag_s=max_bbox_center_speed_for_body_diag_s,
        max_bbox_center_step_for_body_px=max_bbox_center_step_for_body_px,
        max_track_world_step_for_bbox_jitter_m=max_track_world_step_for_bbox_jitter_m,
    )
    return _execution_payload(
        tracks,
        mode=mode,
        source_plan=source_plan,
        scheduled=[],
        skipped=[],
        track_continuity=track_continuity,
        skeleton_stride=int(skeleton_stride),
    )


def body_frame_batches_from_execution(
    tracks: Tracks,
    execution: Mapping[str, Any],
) -> list[tuple[int, list[tuple[int, Any]]]]:
    """Convert a BODY execution manifest into runner frame batches."""

    track_lookup = _track_lookup(tracks)
    batches: list[tuple[int, list[tuple[int, Any]]]] = []
    for frame in execution.get("scheduled_frames", []):
        frame_idx = int(frame["frame_idx"])
        target_ids = {int(player_id) for player_id in frame.get("target_player_ids", [])}
        active = track_lookup.get(frame_idx, [])
        if target_ids:
            active = [(player_id, track_frame) for player_id, track_frame in active if player_id in target_ids]
        if active:
            batches.append((frame_idx, active))
    # Ascending frame order so the static-intrinsics baseline is always the
    # clip's earliest scheduled frame and size-mismatch errors name the later
    # offending frame, regardless of deep-mesh-window scheduling order.
    batches.sort(key=lambda item: item[0])
    return batches


def write_body_compute_execution(path: str | Path, execution: Mapping[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(execution, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _execution_from_frame_plan(
    tracks: Tracks,
    *,
    plan_path: Path,
    track_lookup: dict[int, list[tuple[int, Any]]],
    safe_track_lookup: dict[int, list[tuple[int, Any]]],
    track_continuity: dict[str, Any],
    max_frames: int | None,
    include_tier2_body_joints: bool,
    contact_dense_pad_s: float,
    skeleton_stride: int,
) -> dict[str, Any]:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    frame_lookup = {int(frame["frame_idx"]): frame for frame in plan.get("frames", [])}
    if _plan_uses_contact_dense_profile(plan):
        scheduled, skipped, profile = _contact_dense_execution_frames(
            tracks,
            plan=plan,
            frame_lookup=frame_lookup,
            track_lookup=track_lookup,
            safe_track_lookup=safe_track_lookup,
            contact_dense_pad_s=contact_dense_pad_s,
        )
        if include_tier2_body_joints:
            tier2_scheduled, tier2_skipped = _tier2_body_joint_scheduled_frames(
                tracks,
                frame_lookup=frame_lookup,
                track_lookup=track_lookup,
                safe_track_lookup=safe_track_lookup,
                already_scheduled_targets=_scheduled_target_keys(scheduled),
                skeleton_stride=skeleton_stride,
            )
            scheduled.extend(tier2_scheduled)
            skipped.extend(tier2_skipped)
        scheduled, limit_skipped = _apply_max_frames(scheduled, max_frames=max_frames)
        skipped.extend(limit_skipped)
        return _execution_payload(
            tracks,
            mode="adaptive_frame_compute_plan",
            source_plan=str(plan_path),
            scheduled=scheduled,
            skipped=sorted(skipped, key=lambda item: (int(item["frame_idx"]), str(item["skip_reason"]))),
            track_continuity=track_continuity,
            mesh_density_profile=profile,
            skeleton_stride=skeleton_stride,
        )

    scheduled: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    scheduled_indexes: set[int] = set()
    continuity_skipped_indexes: set[int] = set()

    for window_index, window in enumerate(plan.get("deep_mesh_windows", [])):
        frame_start = int(window["frame_start"])
        frame_end = int(window["frame_end"])
        window_target_ids = [int(player_id) for player_id in window.get("target_player_ids", [])]
        for frame_idx in range(frame_start, frame_end + 1):
            active = track_lookup.get(frame_idx, [])
            if not active:
                continue
            frame_plan = frame_lookup.get(frame_idx, {})
            target_ids = _target_player_ids(active, window_target_ids, frame_plan)
            if not target_ids:
                continue
            safe_ids = {player_id for player_id, _frame in safe_track_lookup.get(frame_idx, [])}
            safe_target_ids = [player_id for player_id in target_ids if player_id in safe_ids]
            unsafe_target_ids = [player_id for player_id in target_ids if player_id not in safe_ids]
            if unsafe_target_ids:
                continuity_skipped_indexes.add(frame_idx)
                skipped.append(
                    _continuity_skipped_frame(
                        tracks,
                        frame_idx=frame_idx,
                        target_player_ids=unsafe_target_ids,
                        active_player_ids=[player_id for player_id, _frame in active],
                        player_targets=_selected_player_targets(frame_plan, unsafe_target_ids),
                    )
                )
            if not safe_target_ids:
                continue
            scheduled_indexes.add(frame_idx)
            scheduled.append(
                _scheduled_mesh_frame(
                    frame_idx=frame_idx,
                    t=frame_idx / tracks.fps,
                    frame_plan=frame_plan,
                    target_player_ids=safe_target_ids,
                    active_player_ids=[player_id for player_id, _frame in active],
                    source_window_index=window_index,
                    window_frame_start=frame_start,
                    window_frame_end=frame_end,
                    window_frame_count=int(window.get("frame_count", frame_end - frame_start + 1)),
                    window_t0=float(window.get("t0", frame_start / tracks.fps)),
                    window_t1=float(window.get("t1", (frame_end + 1) / tracks.fps)),
                    fallback_representation=str(window.get("fallback_representation", "lane_a_skeleton")),
                    reason_counts=dict(window.get("reason_counts", {})),
                    reasons=list(frame_plan.get("reasons", [])),
                    max_score=float(window.get("max_score", frame_plan.get("score", 0.0))),
                    player_targets=_selected_player_targets(frame_plan, safe_target_ids),
                )
            )

    cadence_skipped_indexes: set[int] = set()
    if include_tier2_body_joints:
        tier2_scheduled, tier2_skipped = _tier2_body_joint_scheduled_frames(
            tracks,
            frame_lookup=frame_lookup,
            track_lookup=track_lookup,
            safe_track_lookup=safe_track_lookup,
            already_scheduled_targets=_scheduled_target_keys(scheduled),
            skeleton_stride=skeleton_stride,
        )
        scheduled.extend(tier2_scheduled)
        skipped.extend(tier2_skipped)
        scheduled_indexes.update(int(frame["frame_idx"]) for frame in tier2_scheduled)
        cadence_skipped_indexes.update(int(frame["frame_idx"]) for frame in tier2_skipped)

    for frame_idx, frame_plan in sorted(frame_lookup.items()):
        if frame_idx in scheduled_indexes or frame_idx in continuity_skipped_indexes or frame_idx in cadence_skipped_indexes:
            continue
        tier = str(frame_plan.get("recommended_tier", "unknown"))
        target_representation = str(frame_plan.get("target_representation", "unknown"))
        skipped.append(
            {
                "frame_idx": frame_idx,
                "t": float(frame_plan.get("t", frame_idx / tracks.fps)),
                "recommended_tier": tier,
                "target_representation": target_representation,
                "skip_reason": _skip_reason(tier=tier, target_representation=target_representation),
                "reasons": list(frame_plan.get("reasons", [])),
                "active_player_ids": [int(player_id) for player_id in frame_plan.get("active_player_ids", [])],
                "player_targets": _all_player_targets(frame_plan),
            }
        )

    scheduled, limit_skipped = _apply_max_frames(scheduled, max_frames=max_frames)
    skipped.extend(limit_skipped)
    return _execution_payload(
        tracks,
        mode="adaptive_frame_compute_plan",
        source_plan=str(plan_path),
        scheduled=scheduled,
        skipped=sorted(skipped, key=lambda item: (int(item["frame_idx"]), str(item["skip_reason"]))),
        track_continuity=track_continuity,
        skeleton_stride=skeleton_stride,
    )


def _plan_uses_contact_dense_profile(plan: Mapping[str, Any]) -> bool:
    policy = plan.get("mesh_coverage_policy")
    if not isinstance(policy, Mapping):
        return False
    return str(policy.get("mode", "")) == "ball_aware"


def _scheduled_mesh_frame(
    *,
    frame_idx: int,
    t: float,
    frame_plan: Mapping[str, Any],
    target_player_ids: list[int],
    active_player_ids: list[int],
    source_window_index: int,
    window_frame_start: int,
    window_frame_end: int,
    window_frame_count: int,
    window_t0: float,
    window_t1: float,
    fallback_representation: str,
    reason_counts: Mapping[str, Any],
    reasons: list[Any],
    max_score: float,
    player_targets: list[dict[str, Any]],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "frame_idx": frame_idx,
        "t": t,
        "recommended_tier": str(frame_plan.get("recommended_tier", "deep_mesh")),
        "target_representation": TIER1_MESH_REPRESENTATION,
        "target_player_ids": target_player_ids,
        "active_player_ids": active_player_ids,
        "source_window_index": source_window_index,
        "window_frame_start": window_frame_start,
        "window_frame_end": window_frame_end,
        "window_frame_count": window_frame_count,
        "window_t0": window_t0,
        "window_t1": window_t1,
        "fallback_representation": fallback_representation,
        "reason_counts": dict(reason_counts),
        "reasons": list(reasons),
        "max_score": max_score,
        "player_targets": player_targets,
    }
    trust_badge = _optional_trust_badge(frame_plan.get("trust_badge"))
    if trust_badge is not None:
        payload["trust_badge"] = trust_badge
    return payload


def _optional_trust_badge(value: Any) -> str | None:
    if value is None:
        return None
    badge = str(value)
    if badge not in TRUST_BADGES:
        raise ValueError(f"BODY frame trust_badge must be one of {TRUST_BADGES}, got {badge!r}")
    return badge


def _contact_dense_execution_frames(
    tracks: Tracks,
    *,
    plan: Mapping[str, Any],
    frame_lookup: dict[int, Any],
    track_lookup: dict[int, list[tuple[int, Any]]],
    safe_track_lookup: dict[int, list[tuple[int, Any]]],
    contact_dense_pad_s: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    pad_frames = int(round(contact_dense_pad_s * tracks.fps))
    uniform_targets_by_frame = _targets_from_existing_deep_mesh_windows(plan, frame_lookup=frame_lookup, track_lookup=track_lookup)
    dense_targets_by_frame: dict[int, set[int]] = {}
    contact_seed_frames: dict[int, set[int]] = {}
    fallback_seed_frame_count = 0

    for frame_idx, frame_plan in sorted(frame_lookup.items()):
        seed_target_ids, used_fallback = _contact_dense_seed_target_ids(frame_plan)
        if not seed_target_ids:
            continue
        if used_fallback:
            fallback_seed_frame_count += 1
        contact_seed_frames[frame_idx] = set(seed_target_ids)
        for dense_frame_idx in range(max(0, frame_idx - pad_frames), frame_idx + pad_frames + 1):
            if dense_frame_idx not in frame_lookup:
                continue
            active_ids = {player_id for player_id, _frame in track_lookup.get(dense_frame_idx, [])}
            for player_id in seed_target_ids:
                if player_id in active_ids:
                    dense_targets_by_frame.setdefault(dense_frame_idx, set()).add(player_id)

    combined_targets_by_frame: dict[int, set[int]] = {}
    for frame_idx, target_ids in uniform_targets_by_frame.items():
        combined_targets_by_frame.setdefault(frame_idx, set()).update(target_ids)
    for frame_idx, target_ids in dense_targets_by_frame.items():
        combined_targets_by_frame.setdefault(frame_idx, set()).update(target_ids)

    if contact_seed_frames:
        status = "applied"
        notes = [
            "contact_dense applied inside existing ball_aware BODY execution plumbing: hitter targets are scheduled on every tracked frame within the contact pad; existing selected ball_aware/uniform windows remain the sparse continuity floor."
        ]
    else:
        status = "uniform_fallback_missing_contact_evidence"
        notes = [
            "contact_dense found no contact-attributed ball_aware/contact trigger frames; falling back to existing ball_aware/uniform mesh windows."
        ]
    if fallback_seed_frame_count:
        notes.append(
            "one or more contact_dense seeds lacked per-player attribution in player_targets; used active/player-target fallback from frame_compute_plan rather than fabricating nearest-player evidence in body_compute."
        )

    frame_window_ids = _contact_dense_window_ids(
        sorted(combined_targets_by_frame),
        combined_targets_by_frame=combined_targets_by_frame,
        dense_targets_by_frame=dense_targets_by_frame,
        uniform_targets_by_frame=uniform_targets_by_frame,
    )
    window_bounds = _window_bounds_from_frame_window_ids(frame_window_ids)
    window_reason_counts = _window_reason_counts(
        frame_window_ids,
        frame_lookup=frame_lookup,
        dense_targets_by_frame=dense_targets_by_frame,
        uniform_targets_by_frame=uniform_targets_by_frame,
    )

    scheduled: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    scheduled_indexes: set[int] = set()
    continuity_skipped_indexes: set[int] = set()
    for frame_idx in sorted(combined_targets_by_frame):
        active = track_lookup.get(frame_idx, [])
        if not active:
            continue
        target_ids = sorted(combined_targets_by_frame[frame_idx])
        safe_ids = {player_id for player_id, _frame in safe_track_lookup.get(frame_idx, [])}
        safe_target_ids = [player_id for player_id in target_ids if player_id in safe_ids]
        unsafe_target_ids = [player_id for player_id in target_ids if player_id not in safe_ids]
        frame_plan = frame_lookup.get(frame_idx, {})
        if unsafe_target_ids:
            continuity_skipped_indexes.add(frame_idx)
            skipped.append(
                _continuity_skipped_frame(
                    tracks,
                    frame_idx=frame_idx,
                    target_player_ids=unsafe_target_ids,
                    active_player_ids=[player_id for player_id, _frame in active],
                    player_targets=_contact_dense_player_targets(
                        frame_plan,
                        active=active,
                        target_ids=unsafe_target_ids,
                        dense_target_ids=dense_targets_by_frame.get(frame_idx, set()),
                        uniform_target_ids=uniform_targets_by_frame.get(frame_idx, set()),
                    ),
                )
            )
        if not safe_target_ids:
            continue
        scheduled_indexes.add(frame_idx)
        source_window_index = frame_window_ids[frame_idx]
        window_start, window_end = window_bounds[source_window_index]
        reasons = _contact_dense_frame_reasons(
            frame_plan,
            frame_idx=frame_idx,
            dense_target_ids=dense_targets_by_frame.get(frame_idx, set()),
            uniform_target_ids=uniform_targets_by_frame.get(frame_idx, set()),
        )
        scheduled.append(
            _scheduled_mesh_frame(
                frame_idx=frame_idx,
                t=frame_idx / tracks.fps,
                frame_plan=frame_plan,
                target_player_ids=safe_target_ids,
                active_player_ids=[player_id for player_id, _frame in active],
                source_window_index=source_window_index,
                window_frame_start=window_start,
                window_frame_end=window_end,
                window_frame_count=window_end - window_start + 1,
                window_t0=window_start / tracks.fps,
                window_t1=(window_end + 1) / tracks.fps,
                fallback_representation="lane_a_skeleton",
                reason_counts=window_reason_counts[source_window_index],
                reasons=reasons,
                max_score=_contact_dense_frame_score(frame_plan, frame_idx=frame_idx, dense_targets_by_frame=dense_targets_by_frame),
                player_targets=_contact_dense_player_targets(
                    frame_plan,
                    active=active,
                    target_ids=safe_target_ids,
                    dense_target_ids=dense_targets_by_frame.get(frame_idx, set()),
                    uniform_target_ids=uniform_targets_by_frame.get(frame_idx, set()),
                ),
            )
        )

    for frame_idx, frame_plan in sorted(frame_lookup.items()):
        if frame_idx in scheduled_indexes or frame_idx in continuity_skipped_indexes:
            continue
        tier = str(frame_plan.get("recommended_tier", "unknown"))
        target_representation = str(frame_plan.get("target_representation", "unknown"))
        skipped.append(
            {
                "frame_idx": frame_idx,
                "t": float(frame_plan.get("t", frame_idx / tracks.fps)),
                "recommended_tier": tier,
                "target_representation": target_representation,
                "skip_reason": _skip_reason(tier=tier, target_representation=target_representation),
                "reasons": list(frame_plan.get("reasons", [])),
                "active_player_ids": [int(player_id) for player_id in frame_plan.get("active_player_ids", [])],
                "player_targets": _all_player_targets(frame_plan),
            }
        )

    profile = {
        "mode": CONTACT_DENSE_PROFILE_MODE,
        "status": status,
        "source_policy_mode": "ball_aware",
        "contact_dense_pad_s": float(contact_dense_pad_s),
        "contact_dense_pad_frames": pad_frames,
        "contact_seed_frame_count": len(contact_seed_frames),
        "contact_seed_player_frame_count": sum(len(targets) for targets in contact_seed_frames.values()),
        "contact_dense_frame_count": len(dense_targets_by_frame),
        "contact_dense_player_frame_count": sum(len(targets) for targets in dense_targets_by_frame.values()),
        "uniform_floor_frame_count": len(uniform_targets_by_frame),
        "uniform_floor_player_frame_count": sum(len(targets) for targets in uniform_targets_by_frame.values()),
        "scheduled_world_mesh_frame_count": len(combined_targets_by_frame),
        "scheduled_world_mesh_player_frame_count": sum(len(targets) for targets in combined_targets_by_frame.values()),
        "unattributed_contact_seed_frame_count": fallback_seed_frame_count,
        "notes": notes,
    }
    return scheduled, skipped, profile


def _targets_from_existing_deep_mesh_windows(
    plan: Mapping[str, Any],
    *,
    frame_lookup: Mapping[int, Any],
    track_lookup: Mapping[int, list[tuple[int, Any]]],
) -> dict[int, set[int]]:
    targets_by_frame: dict[int, set[int]] = {}
    for window in plan.get("deep_mesh_windows", []) or []:
        if not isinstance(window, Mapping):
            continue
        try:
            frame_start = int(window["frame_start"])
            frame_end = int(window["frame_end"])
        except (KeyError, TypeError, ValueError):
            continue
        window_target_ids = [int(player_id) for player_id in window.get("target_player_ids", [])]
        for frame_idx in range(frame_start, frame_end + 1):
            active = list(track_lookup.get(frame_idx, []))
            if not active:
                continue
            frame_plan = frame_lookup.get(frame_idx, {})
            frame_reasons = {str(reason) for reason in frame_plan.get("reasons", [])} if isinstance(frame_plan, Mapping) else set()
            if frame_reasons & CONTACT_DENSE_TRIGGER_REASONS:
                continue
            selected = _target_player_ids(active, window_target_ids, frame_plan if isinstance(frame_plan, Mapping) else {})
            if selected:
                targets_by_frame.setdefault(frame_idx, set()).update(selected)
    return targets_by_frame


def _contact_dense_seed_target_ids(frame_plan: Mapping[str, Any]) -> tuple[set[int], bool]:
    frame_reasons = {str(reason) for reason in frame_plan.get("reasons", [])}
    target_ids: set[int] = set()
    for target in _all_player_targets(frame_plan):
        target_reasons = {str(reason) for reason in target.get("reasons", [])}
        if target_reasons & CONTACT_DENSE_TRIGGER_REASONS and target.get("player_id") is not None:
            target_ids.add(int(target["player_id"]))
    if target_ids:
        return target_ids, False
    if frame_reasons & CONTACT_DENSE_TRIGGER_REASONS:
        fallback_ids = {
            int(target["player_id"])
            for target in _all_player_targets(frame_plan)
            if target.get("player_id") is not None
        }
        if not fallback_ids:
            fallback_ids = {int(player_id) for player_id in frame_plan.get("active_player_ids", [])}
        return fallback_ids, True
    return set(), False


def _contact_dense_window_ids(
    frame_indexes: list[int],
    *,
    combined_targets_by_frame: Mapping[int, set[int]],
    dense_targets_by_frame: Mapping[int, set[int]],
    uniform_targets_by_frame: Mapping[int, set[int]],
) -> dict[int, int]:
    frame_window_ids: dict[int, int] = {}
    current_window = -1
    previous_frame_idx: int | None = None
    previous_signature: tuple[tuple[int, ...], bool, bool] | None = None
    for frame_idx in frame_indexes:
        signature = (
            tuple(sorted(combined_targets_by_frame.get(frame_idx, set()))),
            bool(dense_targets_by_frame.get(frame_idx)),
            bool(uniform_targets_by_frame.get(frame_idx)),
        )
        if previous_frame_idx is None or frame_idx != previous_frame_idx + 1 or signature != previous_signature:
            current_window += 1
        frame_window_ids[frame_idx] = current_window
        previous_frame_idx = frame_idx
        previous_signature = signature
    return frame_window_ids


def _window_bounds_from_frame_window_ids(frame_window_ids: Mapping[int, int]) -> dict[int, tuple[int, int]]:
    bounds: dict[int, tuple[int, int]] = {}
    for frame_idx, window_id in frame_window_ids.items():
        if window_id not in bounds:
            bounds[window_id] = (frame_idx, frame_idx)
        else:
            start, end = bounds[window_id]
            bounds[window_id] = (min(start, frame_idx), max(end, frame_idx))
    return bounds


def _window_reason_counts(
    frame_window_ids: Mapping[int, int],
    *,
    frame_lookup: Mapping[int, Any],
    dense_targets_by_frame: Mapping[int, set[int]],
    uniform_targets_by_frame: Mapping[int, set[int]],
) -> dict[int, dict[str, int]]:
    counts: dict[int, dict[str, int]] = {}
    for frame_idx, window_id in frame_window_ids.items():
        frame_counts = counts.setdefault(window_id, {})
        reasons = _contact_dense_frame_reasons(
            frame_lookup.get(frame_idx, {}),
            frame_idx=frame_idx,
            dense_target_ids=dense_targets_by_frame.get(frame_idx, set()),
            uniform_target_ids=uniform_targets_by_frame.get(frame_idx, set()),
        )
        for reason in reasons:
            frame_counts[reason] = frame_counts.get(reason, 0) + 1
    return {window_id: dict(sorted(reason_counts.items())) for window_id, reason_counts in counts.items()}


def _contact_dense_frame_reasons(
    frame_plan: Mapping[str, Any],
    *,
    frame_idx: int,
    dense_target_ids: set[int],
    uniform_target_ids: set[int],
) -> list[str]:
    reasons: list[str] = []
    if dense_target_ids:
        reasons.append(CONTACT_DENSE_HITTER_WINDOW_REASON)
    frame_reasons = [str(reason) for reason in frame_plan.get("reasons", [])]
    for reason in frame_reasons:
        if reason in CONTACT_DENSE_TRIGGER_REASONS and reason not in reasons:
            reasons.append(reason)
    if uniform_target_ids:
        uniform_reason_present = False
        for reason in frame_reasons:
            if reason == UNIFORM_MESH_SELECTION_REASON:
                uniform_reason_present = True
            if reason not in reasons and (reason == UNIFORM_MESH_SELECTION_REASON or not dense_target_ids):
                reasons.append(reason)
        if not uniform_reason_present and UNIFORM_MESH_SELECTION_REASON not in reasons:
            reasons.append(UNIFORM_MESH_SELECTION_REASON)
    return reasons or list(frame_reasons)


def _contact_dense_frame_score(
    frame_plan: Mapping[str, Any],
    *,
    frame_idx: int,
    dense_targets_by_frame: Mapping[int, set[int]],
) -> float:
    base = float(frame_plan.get("score", 0.0) or 0.0)
    if dense_targets_by_frame.get(frame_idx):
        base = max(base, 0.9)
    return round(min(base, 1.0), 3)


def _contact_dense_player_targets(
    frame_plan: Mapping[str, Any],
    *,
    active: list[tuple[int, Any]],
    target_ids: list[int],
    dense_target_ids: set[int],
    uniform_target_ids: set[int],
) -> list[dict[str, Any]]:
    active_by_id = {int(player_id): track_frame for player_id, track_frame in active}
    original_by_id = {
        int(target["player_id"]): dict(target)
        for target in _all_player_targets(frame_plan)
        if target.get("player_id") is not None
    }
    targets: list[dict[str, Any]] = []
    for player_id in sorted(int(target_id) for target_id in target_ids):
        track_frame = active_by_id.get(player_id)
        original = original_by_id.get(player_id, {})
        reasons = [str(reason) for reason in original.get("reasons", [])]
        if player_id in dense_target_ids and CONTACT_DENSE_HITTER_WINDOW_REASON not in reasons:
            reasons.insert(0, CONTACT_DENSE_HITTER_WINDOW_REASON)
        if player_id in uniform_target_ids and UNIFORM_MESH_SELECTION_REASON not in reasons:
            reasons.append(UNIFORM_MESH_SELECTION_REASON)
        score = float(original.get("score", 0.0) or 0.0)
        if player_id in dense_target_ids:
            score = max(score, 0.9)
        if player_id in uniform_target_ids:
            score = max(score, 1.0)
        targets.append(
            {
                "player_id": player_id,
                "track_conf": round(float(original.get("track_conf", getattr(track_frame, "conf", 0.0))), 3),
                "score": round(min(score, 1.0), 3),
                "recommended_tier": "deep_mesh",
                "target_representation": TIER1_MESH_REPRESENTATION,
                "reasons": reasons or _contact_dense_frame_reasons(
                    frame_plan,
                    frame_idx=int(frame_plan.get("frame_idx", 0)),
                    dense_target_ids={player_id} if player_id in dense_target_ids else set(),
                    uniform_target_ids={player_id} if player_id in uniform_target_ids else set(),
                ),
            }
        )
    return targets


def _execution_without_frame_plan(
    tracks: Tracks,
    *,
    track_lookup: dict[int, list[tuple[int, Any]]],
    safe_track_lookup: dict[int, list[tuple[int, Any]]],
    track_continuity: dict[str, Any],
    include_tier2_body_joints: bool,
    skeleton_stride: int,
) -> dict[str, Any]:
    skipped: list[dict[str, Any]] = []
    scheduled: list[dict[str, Any]] = []
    base_frame_indexes = _base_skeleton_frame_indexes(track_lookup, skeleton_stride=skeleton_stride)
    for frame_idx, active in sorted(track_lookup.items()):
        active_player_ids = [player_id for player_id, _frame in active]
        safe_active = safe_track_lookup.get(frame_idx, [])
        safe_ids = {player_id for player_id, _frame in safe_active}
        unsafe_ids = [player_id for player_id in active_player_ids if player_id not in safe_ids]
        if include_tier2_body_joints and safe_active and frame_idx in base_frame_indexes:
            scheduled.append(
                _tier2_body_joint_frame(
                    tracks,
                    frame_idx=frame_idx,
                    active=active,
                    safe_active=safe_active,
                    frame_plan={},
                )
            )
        elif include_tier2_body_joints and safe_active:
            skipped.append(
                _skeleton_stride_skipped_frame(
                    tracks,
                    frame_idx=frame_idx,
                    active=active,
                    skipped_active=safe_active,
                )
            )
        elif safe_active:
            skipped.append(
                {
                    "frame_idx": frame_idx,
                    "t": frame_idx / tracks.fps,
                    "recommended_tier": "deep_mesh",
                    "target_representation": "world_mesh",
                    "target_player_ids": [player_id for player_id, _frame in safe_active],
                    "active_player_ids": active_player_ids,
                    "source_window_index": None,
                    "reason_counts": {},
                    "skip_reason": MISSING_FRAME_COMPUTE_PLAN_REASON,
                    "reasons": [MISSING_FRAME_COMPUTE_PLAN_REASON],
                    "max_score": None,
                    "player_targets": _track_player_targets(
                        safe_active,
                        reason=MISSING_FRAME_COMPUTE_PLAN_REASON,
                    ),
                }
            )
        if unsafe_ids:
            unsafe_id_set = set(unsafe_ids)
            unsafe_active = [(player_id, track_frame) for player_id, track_frame in active if player_id in unsafe_id_set]
            skipped.append(
                _continuity_skipped_frame(
                    tracks,
                    frame_idx=frame_idx,
                    target_player_ids=unsafe_ids,
                    active_player_ids=active_player_ids,
                    player_targets=_track_player_targets(unsafe_active),
                )
            )
    return _execution_payload(
        tracks,
        mode="sam3d_tier2_body_joints_without_frame_compute_plan"
        if include_tier2_body_joints
        else "lane_b_requires_frame_compute_plan",
        source_plan=None,
        scheduled=scheduled,
        skipped=skipped,
        track_continuity=track_continuity,
        skeleton_stride=skeleton_stride,
    )


def _scheduled_target_keys(scheduled: list[dict[str, Any]]) -> set[tuple[int, int]]:
    keys: set[tuple[int, int]] = set()
    for frame in scheduled:
        frame_idx = int(frame["frame_idx"])
        for player_id in frame.get("target_player_ids", []):
            keys.add((frame_idx, int(player_id)))
    return keys


def _tier2_body_joint_scheduled_frames(
    tracks: Tracks,
    *,
    frame_lookup: dict[int, Any],
    track_lookup: dict[int, list[tuple[int, Any]]],
    safe_track_lookup: dict[int, list[tuple[int, Any]]],
    already_scheduled_targets: set[tuple[int, int]],
    skeleton_stride: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    scheduled: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    base_frame_indexes = _base_skeleton_frame_indexes(track_lookup, skeleton_stride=skeleton_stride)
    for frame_idx, active in sorted(track_lookup.items()):
        unscheduled_safe_active = [
            (player_id, track_frame)
            for player_id, track_frame in safe_track_lookup.get(frame_idx, [])
            if (frame_idx, player_id) not in already_scheduled_targets
        ]
        if not unscheduled_safe_active:
            continue
        if frame_idx not in base_frame_indexes:
            skipped.append(
                _skeleton_stride_skipped_frame(
                    tracks,
                    frame_idx=frame_idx,
                    active=active,
                    skipped_active=unscheduled_safe_active,
                )
            )
            continue
        safe_active = unscheduled_safe_active
        if not safe_active:
            continue
        scheduled.append(
            _tier2_body_joint_frame(
                tracks,
                frame_idx=frame_idx,
                active=active,
                safe_active=safe_active,
                frame_plan=frame_lookup.get(frame_idx, {}),
            )
        )
    return scheduled, skipped


def _base_skeleton_frame_indexes(
    track_lookup: Mapping[int, list[tuple[int, Any]]],
    *,
    skeleton_stride: int,
) -> set[int]:
    frame_indexes = sorted(int(frame_idx) for frame_idx in track_lookup)
    if not frame_indexes:
        return set()
    if skeleton_stride <= 1:
        return set(frame_indexes)
    anchor = frame_indexes[0]
    return {frame_idx for frame_idx in frame_indexes if (frame_idx - anchor) % skeleton_stride == 0}


def _skeleton_stride_skipped_frame(
    tracks: Tracks,
    *,
    frame_idx: int,
    active: list[tuple[int, Any]],
    skipped_active: list[tuple[int, Any]],
) -> dict[str, Any]:
    target_ids = [int(player_id) for player_id, _frame in skipped_active]
    return {
        "frame_idx": frame_idx,
        "t": frame_idx / tracks.fps,
        "recommended_tier": TIER2_BODY_JOINTS_TIER,
        "target_representation": TIER2_BODY_JOINTS_REPRESENTATION,
        "skip_reason": BODY_SKELETON_STRIDE_SKIP_REASON,
        "reasons": [BODY_SKELETON_STRIDE_SKIP_REASON],
        "target_player_ids": target_ids,
        "active_player_ids": [int(player_id) for player_id, _frame in active],
        "player_targets": _tier2_stride_skipped_targets(skipped_active),
    }


def _tier2_body_joint_frame(
    tracks: Tracks,
    *,
    frame_idx: int,
    active: list[tuple[int, Any]],
    safe_active: list[tuple[int, Any]],
    frame_plan: Mapping[str, Any],
) -> dict[str, Any]:
    target_ids = [int(player_id) for player_id, _frame in safe_active]
    return {
        "frame_idx": frame_idx,
        "t": float(frame_plan.get("t", frame_idx / tracks.fps)) if isinstance(frame_plan, Mapping) else frame_idx / tracks.fps,
        "recommended_tier": TIER2_BODY_JOINTS_TIER,
        "target_representation": TIER2_BODY_JOINTS_REPRESENTATION,
        "target_player_ids": target_ids,
        "active_player_ids": [int(player_id) for player_id, _frame in active],
        "source_window_index": None,
        "fallback_representation": "lane_a_skeleton",
        "reason_counts": {SAM3D_BODY_JOINTS_ALL_TRACKED_REASON: len(target_ids)},
        "reasons": [SAM3D_BODY_JOINTS_ALL_TRACKED_REASON],
        "max_score": float(frame_plan.get("score", 0.0) or 0.0) if isinstance(frame_plan, Mapping) else 0.0,
        "source": SAM3D_BODY_JOINTS_SOURCE,
        "player_targets": _tier2_player_targets(frame_plan, safe_active),
    }


def _tier2_player_targets(frame_plan: Mapping[str, Any], safe_active: list[tuple[int, Any]]) -> list[dict[str, Any]]:
    plan_targets = _selected_player_targets(frame_plan, [player_id for player_id, _frame in safe_active])
    if plan_targets:
        targets = []
        for target in plan_targets:
            targets.append(
                {
                    **target,
                    "recommended_tier": TIER2_BODY_JOINTS_TIER,
                    "target_representation": TIER2_BODY_JOINTS_REPRESENTATION,
                    "source": SAM3D_BODY_JOINTS_SOURCE,
                    "reasons": [SAM3D_BODY_JOINTS_ALL_TRACKED_REASON],
                }
            )
        return targets
    return [
        {
            "player_id": int(player_id),
            "track_conf": float(track_frame.conf),
            "score": 0.0,
            "recommended_tier": TIER2_BODY_JOINTS_TIER,
            "target_representation": TIER2_BODY_JOINTS_REPRESENTATION,
            "source": SAM3D_BODY_JOINTS_SOURCE,
            "reasons": [SAM3D_BODY_JOINTS_ALL_TRACKED_REASON],
        }
        for player_id, track_frame in safe_active
    ]


def _tier2_stride_skipped_targets(skipped_active: list[tuple[int, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "player_id": int(player_id),
            "track_conf": round(float(track_frame.conf), 3),
            "score": 0.0,
            "recommended_tier": TIER2_BODY_JOINTS_TIER,
            "target_representation": TIER2_BODY_JOINTS_REPRESENTATION,
            "source": SAM3D_BODY_JOINTS_SOURCE,
            "reasons": [BODY_SKELETON_STRIDE_SKIP_REASON],
        }
        for player_id, track_frame in skipped_active
    ]


def _cadence_summary(
    tracks: Tracks,
    *,
    scheduled: list[dict[str, Any]],
    scheduled_player_frame_count: int,
    skeleton_stride: int,
) -> dict[str, Any]:
    track_lookup = _track_lookup(tracks)
    total_track_frame_count = len(track_lookup)
    total_track_player_frame_count = sum(len(active) for active in track_lookup.values())
    base_frame_indexes = _base_skeleton_frame_indexes(track_lookup, skeleton_stride=skeleton_stride)
    scheduled_frame_indexes = {int(frame["frame_idx"]) for frame in scheduled}
    scheduled_frame_count = len(scheduled_frame_indexes)
    effective_stride = (
        round(total_track_frame_count / scheduled_frame_count, 3)
        if total_track_frame_count > 0 and scheduled_frame_count > 0
        else None
    )
    effective_player_stride = (
        round(total_track_player_frame_count / scheduled_player_frame_count, 3)
        if total_track_player_frame_count > 0 and scheduled_player_frame_count > 0
        else None
    )
    base_skeleton_player_frame_count = sum(len(track_lookup.get(frame_idx, [])) for frame_idx in base_frame_indexes)
    return {
        "base_skeleton_stride": int(skeleton_stride),
        "effective_stride": effective_stride,
        "effective_player_stride": effective_player_stride,
        "total_track_frame_count": total_track_frame_count,
        "total_track_player_frame_count": total_track_player_frame_count,
        "base_skeleton_frame_count": len(base_frame_indexes),
        "base_skeleton_player_frame_count": base_skeleton_player_frame_count,
        "event_extra_frame_count": len(scheduled_frame_indexes - base_frame_indexes),
        "scheduled_vs_total_frame_count": {
            "scheduled": scheduled_frame_count,
            "total": total_track_frame_count,
        },
        "scheduled_vs_total_player_frame_count": {
            "scheduled": scheduled_player_frame_count,
            "total": total_track_player_frame_count,
        },
    }


def _execution_payload(
    tracks: Tracks,
    *,
    mode: str,
    source_plan: str | None,
    scheduled: list[dict[str, Any]],
    skipped: list[dict[str, Any]],
    track_continuity: dict[str, Any],
    skeleton_stride: int,
    mesh_density_profile: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    scheduled_by_target_representation: dict[str, int] = {}
    scheduled_by_reason: dict[str, int] = {}
    scheduled_targeted_reviewed_contact_frame_count = 0
    scheduled_coverage_incomplete_frame_count = 0
    tier1_mesh_player_frame_count = 0
    tier2_body_joint_player_frame_count = 0
    for frame in scheduled:
        target_representation = str(frame.get("target_representation", "unknown"))
        scheduled_by_target_representation[target_representation] = scheduled_by_target_representation.get(target_representation, 0) + 1
        target_count = len(frame.get("target_player_ids", []))
        if target_representation == TIER1_MESH_REPRESENTATION:
            tier1_mesh_player_frame_count += target_count
        if target_representation == TIER2_BODY_JOINTS_REPRESENTATION:
            tier2_body_joint_player_frame_count += target_count
        reasons = [str(reason) for reason in frame.get("reasons", []) if reason]
        if "reviewed_contact_targeted_body" in reasons:
            scheduled_targeted_reviewed_contact_frame_count += 1
        if "missing_expected_players" in reasons:
            scheduled_coverage_incomplete_frame_count += 1
        for reason in reasons:
            scheduled_by_reason[reason] = scheduled_by_reason.get(reason, 0) + 1

    skipped_by_tier: dict[str, int] = {}
    skipped_by_target_representation: dict[str, int] = {}
    skipped_by_reason: dict[str, int] = {}
    for frame in skipped:
        tier = str(frame.get("recommended_tier", "unknown"))
        skipped_by_tier[tier] = skipped_by_tier.get(tier, 0) + 1
        target_representation = str(frame.get("target_representation", "unknown"))
        skipped_by_target_representation[target_representation] = (
            skipped_by_target_representation.get(target_representation, 0) + 1
        )
        reasons = frame.get("reasons", [])
        if isinstance(reasons, list):
            for reason in reasons:
                reason_key = str(reason)
                skipped_by_reason[reason_key] = skipped_by_reason.get(reason_key, 0) + 1
    scheduled_player_frame_count = sum(len(frame.get("target_player_ids", [])) for frame in scheduled)
    cadence_summary = _cadence_summary(
        tracks,
        scheduled=scheduled,
        scheduled_player_frame_count=scheduled_player_frame_count,
        skeleton_stride=skeleton_stride,
    )
    track_continuity_skipped_player_frame_count = sum(
        len(frame.get("target_player_ids", []))
        for frame in skipped
        if frame.get("skip_reason") == UNSAFE_TRACK_CONTINUITY_REASON
    )
    max_track_speed_for_body_mps = float(track_continuity["max_track_speed_for_body_mps"])
    max_bbox_center_speed_for_body_diag_s = float(track_continuity["max_bbox_center_speed_for_body_diag_s"])
    max_bbox_center_step_for_body_px = float(track_continuity["max_bbox_center_step_for_body_px"])
    max_track_world_step_for_bbox_jitter_m = float(track_continuity["max_track_world_step_for_bbox_jitter_m"])
    payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "mode": mode,
        "source_plan": source_plan,
        "fps": tracks.fps,
        "scheduled_frames": scheduled,
        "skipped_frames": skipped,
        "track_continuity": track_continuity,
        "summary": {
            "scheduled_frame_count": len(scheduled),
            "scheduled_player_frame_count": scheduled_player_frame_count,
            **cadence_summary,
            "scheduled_by_target_representation": dict(sorted(scheduled_by_target_representation.items())),
            "scheduled_by_reason": dict(sorted(scheduled_by_reason.items())),
            "scheduled_targeted_reviewed_contact_frame_count": scheduled_targeted_reviewed_contact_frame_count,
            "scheduled_coverage_incomplete_frame_count": scheduled_coverage_incomplete_frame_count,
            "tier1_mesh_player_frame_count": tier1_mesh_player_frame_count,
            "tier2_body_joint_player_frame_count": tier2_body_joint_player_frame_count,
            "skipped_frame_count": len(skipped),
            "skipped_by_tier": dict(sorted(skipped_by_tier.items())),
            "skipped_by_target_representation": dict(sorted(skipped_by_target_representation.items())),
            "skipped_by_reason": dict(sorted(skipped_by_reason.items())),
            "max_track_speed_for_body_mps": max_track_speed_for_body_mps,
            "max_bbox_center_speed_for_body_diag_s": max_bbox_center_speed_for_body_diag_s,
            "max_bbox_center_step_for_body_px": max_bbox_center_step_for_body_px,
            "max_track_world_step_for_bbox_jitter_m": max_track_world_step_for_bbox_jitter_m,
            "track_continuity_status": track_continuity["status"],
            "track_continuity_temporal_jump_count": int(track_continuity["temporal_jump_count"]),
            "track_continuity_world_anchor_jitter_player_frame_count": int(
                track_continuity["world_anchor_jitter_count"]
            ),
            "track_continuity_skipped_player_frame_count": track_continuity_skipped_player_frame_count,
        },
    }
    if mesh_density_profile is not None:
        payload["mesh_density_profile"] = dict(mesh_density_profile)
    return payload


def _body_safe_track_lookup(
    tracks: Tracks,
    *,
    track_lookup: dict[int, list[tuple[int, Any]]],
    max_track_speed_for_body_mps: float,
    max_bbox_center_speed_for_body_diag_s: float,
    max_bbox_center_step_for_body_px: float,
    max_track_world_step_for_bbox_jitter_m: float,
) -> tuple[dict[int, list[tuple[int, Any]]], dict[str, Any]]:
    safe_lookup: dict[int, list[tuple[int, Any]]] = {}
    last_safe_by_player: dict[int, tuple[int, float, list[float], list[float]]] = {}
    temporal_jumps: list[dict[str, Any]] = []
    world_anchor_jitter: list[dict[str, Any]] = []
    total_player_frame_count = 0
    safe_player_frame_count = 0

    for frame_idx, active in sorted(track_lookup.items()):
        t = frame_idx / tracks.fps
        for player_id, track_frame in active:
            total_player_frame_count += 1
            xy = [float(value) for value in track_frame.world_xy]
            bbox = [float(value) for value in track_frame.bbox]
            previous = last_safe_by_player.get(player_id)
            if previous is not None:
                prev_frame_idx, prev_t, prev_xy, prev_bbox = previous
                dt = t - prev_t
                step = math.dist(prev_xy, xy) if dt > 0.0 else math.inf
                speed = step / dt if dt > 0.0 else math.inf
                bbox_step_px = _bbox_center_step_px(prev_bbox, bbox)
                bbox_center_speed_diag_s = (
                    bbox_step_px / _average_bbox_diagonal_px(prev_bbox, bbox) / dt
                    if dt > 0.0
                    else math.inf
                )
                if dt <= 0.0 or speed > max_track_speed_for_body_mps:
                    continuity_record = _continuity_record(
                        player_id=player_id,
                        prev_frame_idx=prev_frame_idx,
                        frame_idx=frame_idx,
                        prev_t=prev_t,
                        t=t,
                        dt=dt,
                        step=step,
                        speed=speed,
                        prev_xy=prev_xy,
                        xy=xy,
                        bbox_step_px=bbox_step_px,
                        bbox_center_speed_diag_s=bbox_center_speed_diag_s,
                    )
                    if (
                        dt > 0.0
                        and bbox_center_speed_diag_s <= max_bbox_center_speed_for_body_diag_s
                        and bbox_step_px <= max_bbox_center_step_for_body_px
                        and step <= max_track_world_step_for_bbox_jitter_m
                    ):
                        world_anchor_jitter.append(continuity_record)
                    else:
                        temporal_jumps.append(continuity_record)
                        continue
            safe_lookup.setdefault(frame_idx, []).append((player_id, track_frame))
            last_safe_by_player[player_id] = (frame_idx, t, xy, bbox)
            safe_player_frame_count += 1

    status = "blocked" if temporal_jumps else "warning" if world_anchor_jitter else "ok"
    return safe_lookup, {
        "status": status,
        "max_track_speed_for_body_mps": float(max_track_speed_for_body_mps),
        "max_bbox_center_speed_for_body_diag_s": float(max_bbox_center_speed_for_body_diag_s),
        "max_bbox_center_step_for_body_px": float(max_bbox_center_step_for_body_px),
        "max_track_world_step_for_bbox_jitter_m": float(max_track_world_step_for_bbox_jitter_m),
        "total_player_frame_count": total_player_frame_count,
        "safe_player_frame_count": safe_player_frame_count,
        "skipped_player_frame_count": total_player_frame_count - safe_player_frame_count,
        "temporal_jump_count": len(temporal_jumps),
        "world_anchor_jitter_count": len(world_anchor_jitter),
        "temporal_jumps": temporal_jumps[:50],
        "world_anchor_jitter": world_anchor_jitter[:50],
    }


def _continuity_record(
    *,
    player_id: int,
    prev_frame_idx: int,
    frame_idx: int,
    prev_t: float,
    t: float,
    dt: float,
    step: float,
    speed: float,
    prev_xy: list[float],
    xy: list[float],
    bbox_step_px: float,
    bbox_center_speed_diag_s: float,
) -> dict[str, Any]:
    return {
        "player_id": int(player_id),
        "prev_frame_idx": int(prev_frame_idx),
        "frame_idx": int(frame_idx),
        "prev_t": round(prev_t, 6),
        "t": round(t, 6),
        "dt_s": round(dt, 6),
        "step_m": round(step, 6) if math.isfinite(step) else None,
        "speed_mps": round(speed, 6) if math.isfinite(speed) else None,
        "world_speed_mps": round(speed, 6) if math.isfinite(speed) else None,
        "prev_world_xy": [round(value, 6) for value in prev_xy],
        "world_xy": [round(value, 6) for value in xy],
        "bbox_center_step_px": round(bbox_step_px, 6) if math.isfinite(bbox_step_px) else None,
        "bbox_center_speed_diag_s": round(bbox_center_speed_diag_s, 6)
        if math.isfinite(bbox_center_speed_diag_s)
        else None,
    }


def _bbox_center_step_px(previous_bbox: list[float], bbox: list[float]) -> float:
    prev_center = _bbox_center(previous_bbox)
    center = _bbox_center(bbox)
    return math.dist(prev_center, center)


def _bbox_center(bbox: list[float]) -> list[float]:
    return [(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0]


def _average_bbox_diagonal_px(previous_bbox: list[float], bbox: list[float]) -> float:
    return max((_bbox_diagonal_px(previous_bbox) + _bbox_diagonal_px(bbox)) / 2.0, 1e-6)


def _bbox_diagonal_px(bbox: list[float]) -> float:
    return math.hypot(bbox[2] - bbox[0], bbox[3] - bbox[1])


def _track_lookup(tracks: Tracks) -> dict[int, list[tuple[int, Any]]]:
    by_frame: dict[int, list[tuple[int, Any]]] = {}
    for player in tracks.players:
        for frame in player.frames:
            frame_idx = int(round(frame.t * tracks.fps))
            by_frame.setdefault(frame_idx, []).append((player.id, frame))
    return by_frame


def _continuity_skipped_frame(
    tracks: Tracks,
    *,
    frame_idx: int,
    target_player_ids: list[int],
    active_player_ids: list[int],
    player_targets: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "frame_idx": frame_idx,
        "t": frame_idx / tracks.fps,
        "recommended_tier": "deep_mesh",
        "target_representation": "world_mesh",
        "skip_reason": UNSAFE_TRACK_CONTINUITY_REASON,
        "reasons": [UNSAFE_TRACK_CONTINUITY_REASON],
        "target_player_ids": target_player_ids,
        "active_player_ids": active_player_ids,
        "player_targets": player_targets,
    }


def _track_player_targets(
    active: list[tuple[int, Any]],
    *,
    reason: str = "no_frame_compute_plan",
) -> list[dict[str, Any]]:
    return [
        {
            "player_id": player_id,
            "track_conf": round(float(track_frame.conf), 3),
            "score": 1.0,
            "recommended_tier": "deep_mesh",
            "target_representation": "world_mesh",
            "reasons": [reason],
        }
        for player_id, track_frame in active
    ]


def _target_player_ids(
    active: list[tuple[int, Any]],
    window_target_ids: list[int],
    frame_plan: Mapping[str, Any],
) -> list[int]:
    active_ids = {player_id for player_id, _frame in active}
    requested = set(window_target_ids) or {int(player_id) for player_id in frame_plan.get("active_player_ids", [])}
    if not requested:
        requested = active_ids
    return sorted(active_ids & requested)


def _skip_reason(*, tier: str, target_representation: str) -> str:
    if target_representation == "manual_review_required" or tier == "human_review":
        return "manual_review_required"
    if tier == "skeleton_preview":
        return "preview_tier_only"
    if tier == "baseline":
        return "baseline_tier"
    return "not_in_deep_mesh_window"


def _apply_max_frames(
    scheduled: list[dict[str, Any]],
    *,
    max_frames: int | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if max_frames is None or len(scheduled) <= max_frames:
        return scheduled, []
    kept = scheduled[:max_frames]
    skipped = [
        {
            "frame_idx": int(frame["frame_idx"]),
            "t": float(frame["t"]),
            "recommended_tier": str(frame.get("recommended_tier", "deep_mesh")),
            "target_representation": str(frame.get("target_representation", "world_mesh")),
            "skip_reason": "max_frames_limit",
            "reasons": list(frame.get("reasons", [])),
            "active_player_ids": [int(player_id) for player_id in frame.get("active_player_ids", [])],
            "player_targets": _all_player_targets(frame),
        }
        for frame in scheduled[max_frames:]
    ]
    return kept, skipped


def _selected_player_targets(frame_plan: Mapping[str, Any], target_ids: list[int]) -> list[dict[str, Any]]:
    wanted = {int(player_id) for player_id in target_ids}
    return [
        dict(target)
        for target in _all_player_targets(frame_plan)
        if int(target.get("player_id", -1)) in wanted
    ]


def _all_player_targets(frame_plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    targets = frame_plan.get("player_targets", [])
    if not isinstance(targets, list):
        return []
    return [dict(target) for target in targets if isinstance(target, Mapping)]


__all__ = [
    "build_body_compute_execution",
    "build_empty_body_compute_execution",
    "body_frame_batches_from_execution",
    "write_body_compute_execution",
]

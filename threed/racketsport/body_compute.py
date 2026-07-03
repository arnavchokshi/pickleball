"""Adaptive BODY compute scheduling helpers."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

from .schemas import Tracks


ARTIFACT_TYPE = "racketsport_body_compute_execution"
SCHEMA_VERSION = 1
DEFAULT_MAX_TRACK_SPEED_FOR_BODY_MPS = 10.0
DEFAULT_MAX_BBOX_CENTER_SPEED_FOR_BODY_DIAG_S = 30.0
DEFAULT_MAX_BBOX_CENTER_STEP_FOR_BODY_PX = 300.0
DEFAULT_MAX_TRACK_WORLD_STEP_FOR_BBOX_JITTER_M = 3.5
UNSAFE_TRACK_CONTINUITY_REASON = "unsafe_track_continuity"
MISSING_FRAME_COMPUTE_PLAN_REASON = "missing_frame_compute_plan"
SAM3D_BODY_JOINTS_ALL_TRACKED_REASON = "sam3d_body_joints_all_tracked"
SAM3D_BODY_JOINTS_SOURCE = "sam3d_body_joints"
TIER2_BODY_JOINTS_TIER = "tier2_body_joints"
TIER2_BODY_JOINTS_REPRESENTATION = "body_joints"
TIER1_MESH_REPRESENTATION = "world_mesh"


def build_body_compute_execution(
    tracks: Tracks,
    *,
    frame_plan_path: str | Path | None = None,
    max_frames: int | None = None,
    include_tier2_body_joints: bool = False,
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
        )
    return _execution_without_frame_plan(
        tracks,
        track_lookup=track_lookup,
        safe_track_lookup=safe_track_lookup,
        track_continuity=track_continuity,
        include_tier2_body_joints=include_tier2_body_joints,
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
) -> dict[str, Any]:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    frame_lookup = {int(frame["frame_idx"]): frame for frame in plan.get("frames", [])}
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
                {
                    "frame_idx": frame_idx,
                    "t": frame_idx / tracks.fps,
                    "recommended_tier": str(frame_plan.get("recommended_tier", "deep_mesh")),
                    "target_representation": TIER1_MESH_REPRESENTATION,
                    "target_player_ids": safe_target_ids,
                    "active_player_ids": [player_id for player_id, _frame in active],
                    "source_window_index": window_index,
                    "window_frame_start": frame_start,
                    "window_frame_end": frame_end,
                    "window_frame_count": int(window.get("frame_count", frame_end - frame_start + 1)),
                    "window_t0": float(window.get("t0", frame_start / tracks.fps)),
                    "window_t1": float(window.get("t1", (frame_end + 1) / tracks.fps)),
                    "fallback_representation": str(window.get("fallback_representation", "lane_a_skeleton")),
                    "reason_counts": dict(window.get("reason_counts", {})),
                    "reasons": list(frame_plan.get("reasons", [])),
                    "max_score": float(window.get("max_score", frame_plan.get("score", 0.0))),
                    "player_targets": _selected_player_targets(frame_plan, safe_target_ids),
                }
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

    if include_tier2_body_joints:
        scheduled.extend(
            _tier2_body_joint_scheduled_frames(
                tracks,
                frame_lookup=frame_lookup,
                track_lookup=track_lookup,
                safe_track_lookup=safe_track_lookup,
                already_scheduled_targets=_scheduled_target_keys(scheduled),
            )
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
    )


def _execution_without_frame_plan(
    tracks: Tracks,
    *,
    track_lookup: dict[int, list[tuple[int, Any]]],
    safe_track_lookup: dict[int, list[tuple[int, Any]]],
    track_continuity: dict[str, Any],
    include_tier2_body_joints: bool,
) -> dict[str, Any]:
    skipped: list[dict[str, Any]] = []
    scheduled: list[dict[str, Any]] = []
    for frame_idx, active in sorted(track_lookup.items()):
        active_player_ids = [player_id for player_id, _frame in active]
        safe_active = safe_track_lookup.get(frame_idx, [])
        safe_ids = {player_id for player_id, _frame in safe_active}
        unsafe_ids = [player_id for player_id in active_player_ids if player_id not in safe_ids]
        if include_tier2_body_joints and safe_active:
            scheduled.append(
                _tier2_body_joint_frame(
                    tracks,
                    frame_idx=frame_idx,
                    active=active,
                    safe_active=safe_active,
                    frame_plan={},
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
) -> list[dict[str, Any]]:
    scheduled: list[dict[str, Any]] = []
    for frame_idx, active in sorted(track_lookup.items()):
        safe_active = [
            (player_id, track_frame)
            for player_id, track_frame in safe_track_lookup.get(frame_idx, [])
            if (frame_idx, player_id) not in already_scheduled_targets
        ]
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
    return scheduled


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


def _execution_payload(
    tracks: Tracks,
    *,
    mode: str,
    source_plan: str | None,
    scheduled: list[dict[str, Any]],
    skipped: list[dict[str, Any]],
    track_continuity: dict[str, Any],
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
    track_continuity_skipped_player_frame_count = sum(
        len(frame.get("target_player_ids", []))
        for frame in skipped
        if frame.get("skip_reason") == UNSAFE_TRACK_CONTINUITY_REASON
    )
    max_track_speed_for_body_mps = float(track_continuity["max_track_speed_for_body_mps"])
    max_bbox_center_speed_for_body_diag_s = float(track_continuity["max_bbox_center_speed_for_body_diag_s"])
    max_bbox_center_step_for_body_px = float(track_continuity["max_bbox_center_step_for_body_px"])
    max_track_world_step_for_bbox_jitter_m = float(track_continuity["max_track_world_step_for_bbox_jitter_m"])
    return {
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
    "body_frame_batches_from_execution",
    "write_body_compute_execution",
]

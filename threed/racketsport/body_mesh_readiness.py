"""CPU-only BODY mesh readiness audit for review packets."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = 1
ARTIFACT_TYPE = "racketsport_body_mesh_readiness"


def build_body_mesh_readiness(
    *,
    clip: str,
    smpl_motion: Mapping[str, Any] | None = None,
    skeleton3d: Mapping[str, Any] | None = None,
    frame_compute_plan: Mapping[str, Any] | None = None,
    body_compute_execution: Mapping[str, Any] | None = None,
    body_full_clip_gate: Mapping[str, Any] | None = None,
    smpl_motion_path: str | None = None,
    skeleton3d_path: str | None = None,
    frame_compute_plan_path: str | None = None,
    body_compute_execution_path: str | None = None,
    body_full_clip_gate_path: str | None = None,
) -> dict[str, Any]:
    """Report whether existing BODY artifacts contain real mesh vertices."""

    mesh_stats = _smpl_mesh_stats(smpl_motion)
    joints_stats = _joints_stats(smpl_motion=smpl_motion, skeleton3d=skeleton3d)
    representation_plan = _representation_plan(
        frame_compute_plan=frame_compute_plan,
        body_compute_execution=body_compute_execution,
        body_full_clip_gate=body_full_clip_gate,
        mesh_stats=mesh_stats,
        joints_stats=joints_stats,
    )
    blockers: list[str] = []
    warnings: list[str] = []

    if smpl_motion is None:
        blockers.append("missing_smpl_motion_json")
    elif mesh_stats["mesh_frame_count"] == 0:
        blockers.append("joints_only_no_mesh_vertices")
    else:
        blockers.extend(["missing_world_mpjpe_gate", *_full_clip_gate_blockers(body_full_clip_gate)])
        warnings.append("mesh_not_accuracy_verified")

    if skeleton3d is None and smpl_motion is None:
        blockers.append("missing_skeleton3d_json")

    if mesh_stats["mesh_frame_count"] == 0:
        warnings.append("missing_mesh_vertices")
        if joints_stats["joints_frame_count"] > 0:
            warnings.append("joints_preview_only")
        else:
            warnings.append("missing_body_joints")

    if smpl_motion is not None and mesh_stats["mesh_frame_count"] == 0:
        blockers.append("missing_world_mpjpe_gate")

    blockers.extend(str(blocker) for blocker in representation_plan["blockers"])
    warnings.extend(str(warning) for warning in representation_plan["warnings"])

    world_mesh_available = mesh_stats["mesh_frame_count"] > 0
    if world_mesh_available:
        status = "mesh_available_needs_accuracy_gate"
    elif joints_stats["joints_frame_count"] > 0:
        status = "joints_only_no_mesh"
    else:
        status = "missing_body_output"

    summary = {
        "player_count": mesh_stats["player_count"],
        "mesh_player_count": mesh_stats["mesh_player_count"],
        "mesh_frame_count": mesh_stats["mesh_frame_count"],
        "mesh_vertex_count_min": mesh_stats["mesh_vertex_count_min"],
        "mesh_vertex_count_max": mesh_stats["mesh_vertex_count_max"],
        "joints_player_count": joints_stats["joints_player_count"],
        "joints_frame_count": joints_stats["joints_frame_count"],
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "clip": clip,
        "status": status,
        "world_mesh_available": world_mesh_available,
        "representation_decision": _representation_decision(representation_plan),
        "trusted_for_body_promotion": False,
        "smpl_motion_path": smpl_motion_path or "",
        "skeleton3d_path": skeleton3d_path or "",
        "frame_compute_plan_path": frame_compute_plan_path or "",
        "body_compute_execution_path": body_compute_execution_path or "",
        "body_full_clip_gate_path": body_full_clip_gate_path or "",
        "summary": summary,
        "representation_plan": representation_plan,
        "blockers": _dedupe(blockers),
        "warnings": _dedupe(warnings),
        "execution": {
            "cpu_only": True,
            "uses_gpu": False,
            "runs_body_model": False,
            "creates_synthetic_mesh_from_joints": False,
            "claims_accuracy_verified": False,
        },
    }


def build_body_mesh_readiness_from_paths(
    *,
    clip: str,
    smpl_motion_path: str | Path | None,
    skeleton3d_path: str | Path | None,
    frame_compute_plan_path: str | Path | None = None,
    body_compute_execution_path: str | Path | None = None,
    body_full_clip_gate_path: str | Path | None = None,
) -> dict[str, Any]:
    smpl_payload = _read_optional_json(smpl_motion_path)
    skeleton_payload = _read_optional_json(skeleton3d_path)
    frame_plan_payload = _read_optional_json(frame_compute_plan_path)
    execution_payload = _read_optional_json(body_compute_execution_path)
    full_clip_gate_payload = _read_optional_json(body_full_clip_gate_path)
    return build_body_mesh_readiness(
        clip=clip,
        smpl_motion=smpl_payload,
        skeleton3d=skeleton_payload,
        frame_compute_plan=frame_plan_payload,
        body_compute_execution=execution_payload,
        body_full_clip_gate=full_clip_gate_payload,
        smpl_motion_path=str(smpl_motion_path or ""),
        skeleton3d_path=str(skeleton3d_path or ""),
        frame_compute_plan_path=str(frame_compute_plan_path or ""),
        body_compute_execution_path=str(body_compute_execution_path or ""),
        body_full_clip_gate_path=str(body_full_clip_gate_path or ""),
    )


def write_body_mesh_readiness(path: str | Path, payload: Mapping[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _smpl_mesh_stats(payload: Mapping[str, Any] | None) -> dict[str, int]:
    players = payload.get("players") if isinstance(payload, Mapping) else None
    if not isinstance(players, list):
        return {
            "player_count": 0,
            "mesh_player_count": 0,
            "mesh_frame_count": 0,
            "mesh_vertex_count_min": 0,
            "mesh_vertex_count_max": 0,
        }
    mesh_player_ids: set[int] = set()
    vertex_counts: list[int] = []
    for player in players:
        if not isinstance(player, Mapping):
            continue
        player_id = int(player.get("id", len(mesh_player_ids) + 1))
        frames = player.get("frames")
        if not isinstance(frames, list):
            continue
        for frame in frames:
            if not isinstance(frame, Mapping):
                continue
            vertices = frame.get("mesh_vertices_world")
            if _is_vertex_list(vertices):
                mesh_player_ids.add(player_id)
                vertex_counts.append(len(vertices))
    return {
        "player_count": len([player for player in players if isinstance(player, Mapping)]),
        "mesh_player_count": len(mesh_player_ids),
        "mesh_frame_count": len(vertex_counts),
        "mesh_vertex_count_min": min(vertex_counts) if vertex_counts else 0,
        "mesh_vertex_count_max": max(vertex_counts) if vertex_counts else 0,
    }


def _joints_stats(
    *,
    smpl_motion: Mapping[str, Any] | None,
    skeleton3d: Mapping[str, Any] | None,
) -> dict[str, int]:
    skeleton_players = skeleton3d.get("players") if isinstance(skeleton3d, Mapping) else None
    if isinstance(skeleton_players, list):
        return _count_joint_frames(skeleton_players)
    smpl_players = smpl_motion.get("players") if isinstance(smpl_motion, Mapping) else None
    if isinstance(smpl_players, list):
        return _count_joint_frames(smpl_players)
    return {"joints_player_count": 0, "joints_frame_count": 0}


def _count_joint_frames(players: list[Any]) -> dict[str, int]:
    player_ids: set[int] = set()
    frame_count = 0
    for player in players:
        if not isinstance(player, Mapping):
            continue
        frames = player.get("frames")
        if not isinstance(frames, list):
            continue
        player_id = int(player.get("id", len(player_ids) + 1))
        for frame in frames:
            if isinstance(frame, Mapping) and _is_vertex_list(frame.get("joints_world")):
                player_ids.add(player_id)
                frame_count += 1
    return {"joints_player_count": len(player_ids), "joints_frame_count": frame_count}


def _representation_plan(
    *,
    frame_compute_plan: Mapping[str, Any] | None,
    body_compute_execution: Mapping[str, Any] | None,
    body_full_clip_gate: Mapping[str, Any] | None,
    mesh_stats: Mapping[str, int],
    joints_stats: Mapping[str, int],
) -> dict[str, Any]:
    plan_summary = frame_compute_plan.get("summary") if isinstance(frame_compute_plan, Mapping) else None
    execution_summary = body_compute_execution.get("summary") if isinstance(body_compute_execution, Mapping) else None

    by_player_target = (
        plan_summary.get("by_player_target_representation") if isinstance(plan_summary, Mapping) else None
    )
    requested_world_mesh_frame_count = _int_summary(plan_summary, "deep_mesh_frame_count")
    requested_world_mesh_player_target_count = _int_mapping_value(by_player_target, "world_mesh")
    scheduled_world_mesh_frame_count = _scheduled_world_mesh_frame_count(execution_summary)
    scheduled_world_mesh_player_frame_count = (
        _int_summary(execution_summary, "tier1_mesh_player_frame_count")
        if isinstance(execution_summary, Mapping) and "tier1_mesh_player_frame_count" in execution_summary
        else _int_summary(execution_summary, "scheduled_player_frame_count")
    )
    available_mesh_frame_count = int(mesh_stats.get("mesh_frame_count", 0))
    available_joint_frame_count = int(joints_stats.get("joints_frame_count", 0))
    lane_a_skeleton_target_count = _int_mapping_value(by_player_target, "lane_a_skeleton") + _int_mapping_value(
        by_player_target,
        "joints_or_preview_mesh",
    )
    manual_review_required_target_count = _int_mapping_value(by_player_target, "manual_review_required")

    blockers: list[str] = []
    warnings: list[str] = []
    world_mesh_required = requested_world_mesh_frame_count > 0 or scheduled_world_mesh_frame_count > 0

    if world_mesh_required:
        if available_mesh_frame_count > 0:
            warnings.append("mesh_not_accuracy_verified")
        if available_mesh_frame_count == 0:
            blockers.append("world_mesh_required_but_missing")
            warnings.append("world_mesh_required_but_missing")
        elif available_mesh_frame_count < max(requested_world_mesh_frame_count, scheduled_world_mesh_frame_count):
            blockers.append("world_mesh_demand_exceeds_available_mesh")
            warnings.append("world_mesh_demand_exceeds_available_mesh")
        blockers.extend(["missing_world_mpjpe_gate", *_full_clip_gate_blockers(body_full_clip_gate)])
    elif frame_compute_plan is not None:
        blockers.append("no_trusted_world_mesh_triggers")
        warnings.append("world_mesh_not_requested_by_current_frame_plan")
        if manual_review_required_target_count > 0:
            blockers.append("manual_review_required_before_mesh")

    return {
        "requested_world_mesh_frame_count": requested_world_mesh_frame_count,
        "requested_world_mesh_player_target_count": requested_world_mesh_player_target_count,
        "scheduled_world_mesh_frame_count": scheduled_world_mesh_frame_count,
        "scheduled_world_mesh_player_frame_count": scheduled_world_mesh_player_frame_count,
        "available_mesh_frame_count": available_mesh_frame_count,
        "available_joint_frame_count": available_joint_frame_count,
        "lane_a_skeleton_target_count": lane_a_skeleton_target_count,
        "manual_review_required_target_count": manual_review_required_target_count,
        "blockers": _dedupe(blockers),
        "warnings": _dedupe(warnings),
    }


def _representation_decision(representation_plan: Mapping[str, Any]) -> str:
    requested = int(representation_plan.get("requested_world_mesh_frame_count", 0))
    scheduled = int(representation_plan.get("scheduled_world_mesh_frame_count", 0))
    available = int(representation_plan.get("available_mesh_frame_count", 0))
    if requested == 0 and scheduled == 0:
        return "no_world_mesh_requested"
    if available == 0:
        return "world_mesh_required_missing_output"
    return "world_mesh_required_available_unverified"


def _full_clip_gate_blockers(body_full_clip_gate: Mapping[str, Any] | None) -> list[str]:
    if body_full_clip_gate is None:
        return ["missing_full_clip_body_gate"]
    if body_full_clip_gate.get("passed") is True:
        return []
    blockers = body_full_clip_gate.get("blockers")
    if isinstance(blockers, list):
        parsed = [str(blocker) for blocker in blockers if str(blocker)]
        if parsed:
            return parsed
    return ["full_clip_body_gate_failed"]


def _scheduled_world_mesh_frame_count(summary: Any) -> int:
    if not isinstance(summary, Mapping):
        return 0
    scheduled_by_target = summary.get("scheduled_by_target_representation")
    if isinstance(scheduled_by_target, Mapping):
        return _int_mapping_value(scheduled_by_target, "world_mesh")
    return _int_summary(summary, "scheduled_frame_count")


def _int_summary(summary: Any, key: str) -> int:
    if not isinstance(summary, Mapping):
        return 0
    return _non_negative_int(summary.get(key))


def _int_mapping_value(mapping: Any, key: str) -> int:
    if not isinstance(mapping, Mapping):
        return 0
    return _non_negative_int(mapping.get(key))


def _non_negative_int(value: Any) -> int:
    if isinstance(value, bool) or value is None:
        return 0
    try:
        number = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, number)


def _is_vertex_list(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and all(_is_vector3(item) for item in value)


def _is_vector3(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 3
        and all(not isinstance(item, bool) and isinstance(item, int | float) and math.isfinite(float(item)) for item in value)
    )


def _read_optional_json(path: str | Path | None) -> Mapping[str, Any] | None:
    if path is None:
        return None
    resolved = Path(path)
    if not resolved.is_file():
        return None
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{resolved} must contain a JSON object")
    return payload


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        if value not in out:
            out.append(value)
    return out

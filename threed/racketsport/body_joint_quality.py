"""CPU-only BODY joint quality checks for world-grounded review output."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = 1
ARTIFACT_TYPE = "racketsport_body_joint_quality"
PROMOTION_BLOCKERS = ["missing_world_mpjpe_gate", "missing_full_clip_body_gate"]


def build_body_joint_quality(
    *,
    clip: str,
    smpl_motion: Mapping[str, Any] | None = None,
    skeleton3d: Mapping[str, Any] | None = None,
    body_compute_execution: Mapping[str, Any] | None = None,
    body_world_label_packet: Mapping[str, Any] | None = None,
    smpl_motion_path: str | None = None,
    skeleton3d_path: str | None = None,
    body_compute_execution_path: str | None = None,
    body_world_label_packet_path: str | None = None,
    min_joint_count: int = 17,
    floor_z_tolerance_m: float = 0.15,
    max_root_speed_for_review_mps: float = 10.0,
    warn_track_anchor_residual_m: float = 1.5,
    max_track_anchor_residual_for_review_m: float = 3.0,
    max_track_anchor_residual_outlier_count_for_review: int = 1,
    max_track_anchor_residual_outlier_ratio_for_review: float = 0.01,
) -> dict[str, Any]:
    """Report whether BODY world joints are structurally usable for review.

    This is deliberately not an accuracy gate. It checks that the world-joint
    artifact is finite, complete relative to the BODY schedule, and plausible
    enough to hand to review/replay tooling while preserving fail-closed
    promotion blockers for labeled world-MPJPE and full-clip coverage.
    """

    if min_joint_count < 1:
        raise ValueError("min_joint_count must be positive")
    if floor_z_tolerance_m < 0.0:
        raise ValueError("floor_z_tolerance_m must be non-negative")
    if max_root_speed_for_review_mps <= 0.0:
        raise ValueError("max_root_speed_for_review_mps must be positive")
    if warn_track_anchor_residual_m <= 0.0:
        raise ValueError("warn_track_anchor_residual_m must be positive")
    if max_track_anchor_residual_for_review_m <= 0.0:
        raise ValueError("max_track_anchor_residual_for_review_m must be positive")
    if warn_track_anchor_residual_m > max_track_anchor_residual_for_review_m:
        raise ValueError("warn_track_anchor_residual_m cannot exceed max_track_anchor_residual_for_review_m")
    if max_track_anchor_residual_outlier_count_for_review < 0:
        raise ValueError("max_track_anchor_residual_outlier_count_for_review must be non-negative")
    if (
        max_track_anchor_residual_outlier_ratio_for_review < 0.0
        or max_track_anchor_residual_outlier_ratio_for_review > 1.0
    ):
        raise ValueError("max_track_anchor_residual_outlier_ratio_for_review must be between 0 and 1")

    smpl_joint_stats = _joint_stats(smpl_motion)
    packet_joint_stats = _packet_joint_stats(body_world_label_packet)
    if smpl_joint_stats["joint_frame_count"] > 0:
        joint_stats = smpl_joint_stats
        joint_source = "smpl_motion"
    elif packet_joint_stats["joint_frame_count"] > 0:
        joint_stats = packet_joint_stats
        joint_source = "body_world_label_packet"
    else:
        joint_stats = smpl_joint_stats
        joint_source = ""
    skeleton_stats = _joint_stats(skeleton3d)
    execution_stats = _execution_stats(body_compute_execution)
    quality_blockers: list[str] = []
    warnings: list[str] = []

    if smpl_motion is None and joint_source != "body_world_label_packet":
        quality_blockers.append("missing_smpl_motion_json")
    if skeleton3d is None and joint_source != "body_world_label_packet":
        quality_blockers.append("missing_skeleton3d_json")
    if joint_stats["joint_frame_count"] == 0:
        quality_blockers.append("no_world_joint_frames")
    if joint_stats["nonfinite_joint_count"] > 0:
        quality_blockers.append("nonfinite_joint_world")
    if joint_stats["invalid_joint_count"] > 0:
        quality_blockers.append("invalid_joint_world")
    if joint_stats["joint_count_min"] and joint_stats["joint_count_min"] < min_joint_count:
        quality_blockers.append("joint_count_below_minimum")
    if joint_stats["inconsistent_joint_count"] > 0:
        quality_blockers.append("inconsistent_joint_count")
    if joint_stats["joint_conf_mismatch_count"] > 0:
        quality_blockers.append("joint_conf_length_mismatch")
    if joint_stats["min_joint_z_m"] is not None and joint_stats["min_joint_z_m"] < -floor_z_tolerance_m:
        quality_blockers.append("joint_below_court_floor")
    root_speed_check_requires_reset_metadata = joint_source == "body_world_label_packet"
    root_speed_check_enabled = (
        not root_speed_check_requires_reset_metadata or joint_stats["temporal_smoothing_reset_metadata_present"]
    )
    root_motion_temporal_jumps = (
        [
            event
            for event in joint_stats["root_motion_events"]
            if event["speed_mps"] > max_root_speed_for_review_mps
        ]
        if root_speed_check_enabled
        else []
    )
    if (
        root_speed_check_enabled
        and joint_stats["max_root_speed_mps"] is not None
        and joint_stats["max_root_speed_mps"] > max_root_speed_for_review_mps
    ):
        quality_blockers.append("root_motion_temporal_jump")
    elif (
        not root_speed_check_enabled
        and joint_stats["max_root_speed_mps"] is not None
        and joint_stats["max_root_speed_mps"] > max_root_speed_for_review_mps
    ):
        warnings.append("compact_packet_root_speed_unchecked_without_reset_metadata")
    track_anchor_residuals = joint_stats["track_anchor_residuals"]
    residual_over_review_count = sum(
        1 for value in track_anchor_residuals if value > max_track_anchor_residual_for_review_m
    )
    residual_over_review_ratio = (
        residual_over_review_count / len(track_anchor_residuals) if track_anchor_residuals else 0.0
    )
    residual_outliers_block_review = residual_over_review_count > 0 and (
        residual_over_review_count > max_track_anchor_residual_outlier_count_for_review
        or residual_over_review_ratio > max_track_anchor_residual_outlier_ratio_for_review
    )
    if residual_outliers_block_review:
        quality_blockers.append("track_anchor_residual_too_large")
    elif residual_over_review_count > 0:
        warnings.append("track_anchor_residual_outliers")
    elif (
        joint_stats["max_track_anchor_residual_m"] is not None
        and joint_stats["max_track_anchor_residual_m"] > warn_track_anchor_residual_m
    ):
        warnings.append("track_anchor_residual_high")

    scheduled_player_frame_count = execution_stats["scheduled_player_frame_count"]
    joint_frame_count = joint_stats["joint_frame_count"]
    if scheduled_player_frame_count > 0 and joint_frame_count < scheduled_player_frame_count:
        quality_blockers.append("scheduled_body_output_incomplete")

    if smpl_motion is not None and skeleton3d is None:
        warnings.append("missing_skeleton3d_preview")
    if skeleton_stats["joint_frame_count"] and skeleton_stats["joint_frame_count"] != joint_frame_count:
        warnings.append("skeleton_frame_count_differs_from_smpl")
    if joint_stats["mesh_frame_count"] < joint_frame_count:
        warnings.append("mesh_vertices_missing_for_some_joint_frames")
    if joint_stats["missing_joint_conf_count"] > 0:
        warnings.append("missing_joint_confidence")

    quality_blockers = _dedupe(quality_blockers)
    usable_for_review = not quality_blockers and joint_frame_count > 0
    status = "quality_checked_needs_accuracy_gate" if usable_for_review else "quality_blocked"
    promotion_blockers = list(PROMOTION_BLOCKERS)

    summary = {
        "player_count": joint_stats["player_count"],
        "joint_player_count": joint_stats["joint_player_count"],
        "joint_frame_count": joint_frame_count,
        "joint_count_min": joint_stats["joint_count_min"],
        "joint_count_max": joint_stats["joint_count_max"],
        "skeleton_joint_frame_count": skeleton_stats["joint_frame_count"],
        "scheduled_frame_count": execution_stats["scheduled_frame_count"],
        "scheduled_player_frame_count": scheduled_player_frame_count,
        "joint_source": joint_source,
        "schedule_coverage_ratio": _coverage_ratio(joint_frame_count, scheduled_player_frame_count),
        "min_joint_z_m": joint_stats["min_joint_z_m"],
        "max_joint_z_m": joint_stats["max_joint_z_m"],
        "mesh_frame_count": joint_stats["mesh_frame_count"],
        "mesh_vertex_count_min": joint_stats["mesh_vertex_count_min"],
        "mesh_vertex_count_max": joint_stats["mesh_vertex_count_max"],
        "nonfinite_joint_count": joint_stats["nonfinite_joint_count"],
        "invalid_joint_count": joint_stats["invalid_joint_count"],
        "joint_conf_mismatch_count": joint_stats["joint_conf_mismatch_count"],
        "missing_joint_conf_count": joint_stats["missing_joint_conf_count"],
        "inconsistent_joint_count": joint_stats["inconsistent_joint_count"],
        "max_frame_gap_s": joint_stats["max_frame_gap_s"],
        "max_root_step_m": joint_stats["max_root_step_m"],
        "max_root_speed_mps": joint_stats["max_root_speed_mps"],
        "max_root_speed_for_review_mps": max_root_speed_for_review_mps,
        "root_motion_temporal_jump_count": len(root_motion_temporal_jumps),
        "temporal_smoothing_reset_count": joint_stats["temporal_smoothing_reset_count"],
        "temporal_smoothing_reset_metadata_present": joint_stats["temporal_smoothing_reset_metadata_present"],
        "max_track_anchor_residual_m": joint_stats["max_track_anchor_residual_m"],
        "warn_track_anchor_residual_m": warn_track_anchor_residual_m,
        "max_track_anchor_residual_for_review_m": max_track_anchor_residual_for_review_m,
        "track_anchor_residual_count": len(track_anchor_residuals),
        "track_anchor_residual_over_review_count": residual_over_review_count,
        "track_anchor_residual_over_review_ratio": round(residual_over_review_ratio, 6),
        "max_track_anchor_residual_outlier_count_for_review": max_track_anchor_residual_outlier_count_for_review,
        "max_track_anchor_residual_outlier_ratio_for_review": max_track_anchor_residual_outlier_ratio_for_review,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "clip": clip,
        "status": status,
        "usable_for_review": usable_for_review,
        "world_joints_available": joint_frame_count > 0,
        "accuracy_verified": False,
        "trusted_for_body_promotion": False,
        "smpl_motion_path": smpl_motion_path or "",
        "skeleton3d_path": skeleton3d_path or "",
        "body_compute_execution_path": body_compute_execution_path or "",
        "body_world_label_packet_path": body_world_label_packet_path or "",
        "summary": summary,
        "quality_blockers": quality_blockers,
        "promotion_blockers": promotion_blockers,
        "blockers": _dedupe([*quality_blockers, *promotion_blockers]),
        "warnings": _dedupe(warnings),
        "root_motion_temporal_jumps": root_motion_temporal_jumps[:20],
        "execution": {
            "cpu_only": True,
            "uses_gpu": False,
            "runs_body_model": False,
            "claims_accuracy_verified": False,
        },
    }


def build_body_joint_quality_from_paths(
    *,
    clip: str,
    smpl_motion_path: str | Path | None,
    skeleton3d_path: str | Path | None,
    body_compute_execution_path: str | Path | None = None,
    body_world_label_packet_path: str | Path | None = None,
    min_joint_count: int = 17,
    floor_z_tolerance_m: float = 0.15,
    max_root_speed_for_review_mps: float = 10.0,
    warn_track_anchor_residual_m: float = 1.5,
    max_track_anchor_residual_for_review_m: float = 3.0,
    max_track_anchor_residual_outlier_count_for_review: int = 1,
    max_track_anchor_residual_outlier_ratio_for_review: float = 0.01,
) -> dict[str, Any]:
    smpl_motion = _read_optional_json(smpl_motion_path)
    skeleton3d = _read_optional_json(skeleton3d_path)
    body_compute_execution = _read_optional_json(body_compute_execution_path)
    body_world_label_packet = _read_optional_json(body_world_label_packet_path)
    return build_body_joint_quality(
        clip=clip,
        smpl_motion=smpl_motion,
        skeleton3d=skeleton3d,
        body_compute_execution=body_compute_execution,
        body_world_label_packet=body_world_label_packet,
        smpl_motion_path=str(smpl_motion_path or ""),
        skeleton3d_path=str(skeleton3d_path or ""),
        body_compute_execution_path=str(body_compute_execution_path or ""),
        body_world_label_packet_path=str(body_world_label_packet_path or ""),
        min_joint_count=min_joint_count,
        floor_z_tolerance_m=floor_z_tolerance_m,
        max_root_speed_for_review_mps=max_root_speed_for_review_mps,
        warn_track_anchor_residual_m=warn_track_anchor_residual_m,
        max_track_anchor_residual_for_review_m=max_track_anchor_residual_for_review_m,
        max_track_anchor_residual_outlier_count_for_review=max_track_anchor_residual_outlier_count_for_review,
        max_track_anchor_residual_outlier_ratio_for_review=max_track_anchor_residual_outlier_ratio_for_review,
    )


def write_body_joint_quality(path: str | Path, payload: Mapping[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _packet_joint_stats(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    samples = payload.get("samples") if isinstance(payload, Mapping) else None
    if not isinstance(samples, list):
        return _empty_joint_stats()

    players: dict[str, list[dict[str, Any]]] = {}
    for sample in samples:
        if not isinstance(sample, Mapping):
            continue
        player_id = _sample_player_id(sample)
        joints = sample.get("predicted_joints_world")
        if not isinstance(joints, list) or not joints:
            continue
        frame: dict[str, Any] = {
            "frame_idx": _frame_idx_or_none(sample.get("frame_index")),
            "t": sample.get("t"),
            "joints_world": joints,
            "joint_conf": sample.get("joint_conf"),
            "track_world_xy": sample.get("track_world_xy"),
        }
        root = _packet_root_world(joints)
        if root is not None:
            frame["transl_world"] = root
        if sample.get("temporal_smoothing_reset") is True:
            frame["temporal_smoothing_reset"] = True
        players.setdefault(player_id, []).append(frame)

    motion = {
        "players": [
            {"id": player_id, "frames": sorted(frames, key=lambda frame: frame.get("t") or 0.0)}
            for player_id, frames in sorted(players.items())
        ]
    }
    stats = _joint_stats(motion)
    stats["temporal_smoothing_reset_metadata_present"] = any(
        isinstance(sample, Mapping) and "temporal_smoothing_reset" in sample for sample in samples
    )
    return stats


def _joint_stats(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    players = payload.get("players") if isinstance(payload, Mapping) else None
    if not isinstance(players, list):
        return _empty_joint_stats()

    player_count = 0
    joint_player_ids: set[str] = set()
    joint_counts: list[int] = []
    mesh_vertex_counts: list[int] = []
    finite_z_values: list[float] = []
    nonfinite_joint_count = 0
    invalid_joint_count = 0
    joint_conf_mismatch_count = 0
    missing_joint_conf_count = 0
    times_by_player: dict[str, list[float]] = {}
    roots_by_player: dict[str, list[tuple[float, list[float], int | None, bool]]] = {}
    track_residuals: list[float] = []
    temporal_smoothing_reset_count = 0

    for fallback_index, player in enumerate(players, start=1):
        if not isinstance(player, Mapping):
            continue
        player_count += 1
        player_id = str(player.get("id", fallback_index))
        frames = player.get("frames")
        if not isinstance(frames, list):
            continue
        for frame in frames:
            if not isinstance(frame, Mapping):
                continue
            joints = frame.get("joints_world")
            if not isinstance(joints, list) or not joints:
                continue

            joint_player_ids.add(player_id)
            joint_counts.append(len(joints))
            t = _finite_float_or_none(frame.get("t"))
            if t is not None:
                times_by_player.setdefault(player_id, []).append(t)

            for joint in joints:
                if not _is_vector3_shape(joint):
                    invalid_joint_count += 1
                    continue
                coords = [float(value) for value in joint]
                if all(math.isfinite(value) for value in coords):
                    finite_z_values.append(coords[2])
                else:
                    nonfinite_joint_count += 1

            joint_conf = frame.get("joint_conf")
            if joint_conf is None:
                missing_joint_conf_count += 1
            elif not isinstance(joint_conf, list) or len(joint_conf) != len(joints):
                joint_conf_mismatch_count += 1

            vertices = frame.get("mesh_vertices_world")
            if isinstance(vertices, list) and vertices and all(_is_finite_vector3(vertex) for vertex in vertices):
                mesh_vertex_counts.append(len(vertices))

            root = _finite_vector3_or_none(frame.get("transl_world"))
            if root is not None and t is not None:
                temporal_smoothing_reset = frame.get("temporal_smoothing_reset") is True
                if temporal_smoothing_reset:
                    temporal_smoothing_reset_count += 1
                roots_by_player.setdefault(player_id, []).append(
                    (t, root, _frame_idx_or_none(frame.get("frame_idx")), temporal_smoothing_reset)
                )
            track_xy = _finite_vector2_or_none(frame.get("track_world_xy"))
            if root is not None and track_xy is not None:
                track_residuals.append(math.dist(root[:2], track_xy))

    max_frame_gap_s = _max_frame_gap(times_by_player)
    max_root_step_m = _max_root_step(roots_by_player)
    max_root_speed_mps = _max_root_speed(roots_by_player)
    root_motion_events = _root_motion_events(roots_by_player)
    return {
        "player_count": player_count,
        "joint_player_count": len(joint_player_ids),
        "joint_frame_count": len(joint_counts),
        "joint_count_min": min(joint_counts) if joint_counts else 0,
        "joint_count_max": max(joint_counts) if joint_counts else 0,
        "mesh_frame_count": len(mesh_vertex_counts),
        "mesh_vertex_count_min": min(mesh_vertex_counts) if mesh_vertex_counts else 0,
        "mesh_vertex_count_max": max(mesh_vertex_counts) if mesh_vertex_counts else 0,
        "min_joint_z_m": min(finite_z_values) if finite_z_values else None,
        "max_joint_z_m": max(finite_z_values) if finite_z_values else None,
        "nonfinite_joint_count": nonfinite_joint_count,
        "invalid_joint_count": invalid_joint_count,
        "joint_conf_mismatch_count": joint_conf_mismatch_count,
        "missing_joint_conf_count": missing_joint_conf_count,
        "inconsistent_joint_count": 1 if len(set(joint_counts)) > 1 else 0,
        "max_frame_gap_s": max_frame_gap_s,
        "max_root_step_m": max_root_step_m,
        "max_root_speed_mps": max_root_speed_mps,
        "root_motion_events": root_motion_events,
        "temporal_smoothing_reset_count": temporal_smoothing_reset_count,
        "temporal_smoothing_reset_metadata_present": temporal_smoothing_reset_count > 0,
        "max_track_anchor_residual_m": max(track_residuals) if track_residuals else None,
        "track_anchor_residuals": track_residuals,
    }


def _empty_joint_stats() -> dict[str, Any]:
    return {
        "player_count": 0,
        "joint_player_count": 0,
        "joint_frame_count": 0,
        "joint_count_min": 0,
        "joint_count_max": 0,
        "mesh_frame_count": 0,
        "mesh_vertex_count_min": 0,
        "mesh_vertex_count_max": 0,
        "min_joint_z_m": None,
        "max_joint_z_m": None,
        "nonfinite_joint_count": 0,
        "invalid_joint_count": 0,
        "joint_conf_mismatch_count": 0,
        "missing_joint_conf_count": 0,
        "inconsistent_joint_count": 0,
        "max_frame_gap_s": None,
        "max_root_step_m": None,
        "max_root_speed_mps": None,
        "root_motion_events": [],
        "temporal_smoothing_reset_count": 0,
        "temporal_smoothing_reset_metadata_present": False,
        "max_track_anchor_residual_m": None,
        "track_anchor_residuals": [],
    }


def _execution_stats(payload: Mapping[str, Any] | None) -> dict[str, int]:
    summary = payload.get("summary") if isinstance(payload, Mapping) else None
    if not isinstance(summary, Mapping):
        return {"scheduled_frame_count": 0, "scheduled_player_frame_count": 0}
    return {
        "scheduled_frame_count": _non_negative_int(summary.get("scheduled_frame_count")),
        "scheduled_player_frame_count": _non_negative_int(summary.get("scheduled_player_frame_count")),
    }


def _coverage_ratio(available: int, scheduled: int) -> float:
    if scheduled <= 0:
        return 1.0 if available > 0 else 0.0
    return min(1.0, available / scheduled)


def _sample_player_id(sample: Mapping[str, Any]) -> str:
    value = sample.get("player_id")
    parsed = _frame_idx_or_none(value)
    return str(parsed if parsed is not None else value or "unknown")


def _packet_root_world(joints: Any) -> list[float] | None:
    if not isinstance(joints, list):
        return None
    for joint in joints:
        root = _finite_vector3_or_none(joint)
        if root is not None:
            return root
    return None


def _max_frame_gap(times_by_player: Mapping[str, list[float]]) -> float | None:
    gaps: list[float] = []
    for times in times_by_player.values():
        ordered = sorted(times)
        gaps.extend(
            later - earlier
            for earlier, later in zip(ordered, ordered[1:])
            if math.isfinite(later - earlier)
        )
    return max(gaps) if gaps else None


def _max_root_step(
    roots_by_player: Mapping[str, list[tuple[float, list[float], int | None, bool]]],
) -> float | None:
    steps: list[float] = []
    for roots in roots_by_player.values():
        ordered = sorted(roots, key=lambda item: item[0])
        for (_, prev_root, _, _), (_, next_root, _, next_reset) in zip(ordered, ordered[1:]):
            if next_reset:
                continue
            steps.append(math.dist(prev_root, next_root))
    return max(steps) if steps else None


def _max_root_speed(
    roots_by_player: Mapping[str, list[tuple[float, list[float], int | None, bool]]],
) -> float | None:
    speeds: list[float] = []
    for roots in roots_by_player.values():
        ordered = sorted(roots, key=lambda item: item[0])
        for (prev_t, prev_root, _, _), (next_t, next_root, _, next_reset) in zip(ordered, ordered[1:]):
            if next_reset:
                continue
            dt = next_t - prev_t
            if dt <= 0.0 or not math.isfinite(dt):
                continue
            speeds.append(math.dist(prev_root, next_root) / dt)
    return max(speeds) if speeds else None


def _root_motion_events(
    roots_by_player: Mapping[str, list[tuple[float, list[float], int | None, bool]]],
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for player_id, roots in roots_by_player.items():
        ordered = sorted(roots, key=lambda item: item[0])
        for (prev_t, prev_root, prev_frame_idx, _), (next_t, next_root, frame_idx, next_reset) in zip(
            ordered, ordered[1:]
        ):
            if next_reset:
                continue
            dt = next_t - prev_t
            if dt <= 0.0 or not math.isfinite(dt):
                continue
            step = math.dist(prev_root, next_root)
            speed = step / dt
            if not math.isfinite(speed):
                continue
            events.append(
                {
                    "player_id": player_id,
                    "prev_frame_idx": prev_frame_idx,
                    "frame_idx": frame_idx,
                    "prev_t": round(prev_t, 6),
                    "t": round(next_t, 6),
                    "dt_s": round(dt, 6),
                    "step_m": round(step, 6),
                    "speed_mps": round(speed, 6),
                    "prev_root_world": [round(value, 6) for value in prev_root],
                    "root_world": [round(value, 6) for value in next_root],
                }
            )
    return sorted(events, key=lambda item: item["speed_mps"], reverse=True)


def _is_vector3_shape(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 3
        and all(not isinstance(item, bool) and isinstance(item, int | float) for item in value)
    )


def _is_finite_vector3(value: Any) -> bool:
    return _is_vector3_shape(value) and all(math.isfinite(float(item)) for item in value)


def _finite_vector3_or_none(value: Any) -> list[float] | None:
    if not _is_finite_vector3(value):
        return None
    return [float(item) for item in value]


def _finite_vector2_or_none(value: Any) -> list[float] | None:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, int | float) for item in value)
    ):
        return None
    coords = [float(item) for item in value]
    return coords if all(math.isfinite(item) for item in coords) else None


def _finite_float_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _frame_idx_or_none(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _non_negative_int(value: Any) -> int:
    if isinstance(value, bool) or value is None:
        return 0
    try:
        number = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, number)


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

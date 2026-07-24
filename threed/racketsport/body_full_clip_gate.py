"""Fail-closed BODY full-clip coverage gate."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = 1
ARTIFACT_TYPE = "racketsport_body_full_clip_gate"
DEFAULT_MIN_COVERAGE = 0.98
DEFAULT_COVERAGE_DENOMINATOR_POLICY = "all_tracked_player_frames"
ELIGIBLE_SCHEDULED_MEASURED_COVERAGE_POLICY = "eligible_scheduled_measured_samples"


def build_body_full_clip_gate(
    *,
    clip: str,
    tracks: Mapping[str, Any] | None = None,
    body_compute_execution: Mapping[str, Any] | None = None,
    body_joint_quality: Mapping[str, Any] | None = None,
    contact_splice: Mapping[str, Any] | None = None,
    runtime_timing: Mapping[str, Any] | None = None,
    tracks_path: str | None = None,
    body_compute_execution_path: str | None = None,
    body_joint_quality_path: str | None = None,
    contact_splice_path: str | None = None,
    runtime_timing_path: str | None = None,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
) -> dict[str, Any]:
    """Build the artifact consumed by BODY gate reports.

    This gate is structural coverage only. It does not measure accuracy and it
    does not clear the world-MPJPE gate.
    """

    if min_coverage <= 0.0 or min_coverage > 1.0:
        raise ValueError("min_coverage must be in (0, 1]")

    tracked_player_frames = _tracked_player_frame_count(tracks)
    execution_summary = body_compute_execution.get("summary") if isinstance(body_compute_execution, Mapping) else {}
    quality_summary = body_joint_quality.get("summary") if isinstance(body_joint_quality, Mapping) else {}
    scheduled_player_frames = _non_negative_int(_mapping_value(execution_summary, "scheduled_player_frame_count"))
    requested_coverage_policy = str(
        _mapping_value(execution_summary, "coverage_denominator_policy") or DEFAULT_COVERAGE_DENOMINATOR_POLICY
    )
    if requested_coverage_policy == ELIGIBLE_SCHEDULED_MEASURED_COVERAGE_POLICY:
        coverage_denominator_policy = requested_coverage_policy
        coverage_denominator_player_frames = _non_negative_int(
            _mapping_value(execution_summary, "coverage_denominator_player_frame_count")
        )
    else:
        coverage_denominator_policy = DEFAULT_COVERAGE_DENOMINATOR_POLICY
        coverage_denominator_player_frames = tracked_player_frames
    mesh_joint_player_frames = _non_negative_int(_mapping_value(quality_summary, "joint_frame_count"))
    skeleton_joint_player_frames = _non_negative_int(_mapping_value(quality_summary, "skeleton_joint_frame_count"))
    joint_player_frames = skeleton_joint_player_frames or mesh_joint_player_frames
    coverage = (
        joint_player_frames / coverage_denominator_player_frames
        if coverage_denominator_player_frames
        else 0.0
    )
    contact_summary = contact_splice.get("summary") if isinstance(contact_splice, Mapping) else {}
    scheduled_contacts = _non_negative_int(_mapping_value(contact_summary, "scheduled_contact_count"))
    mesh_contact_count = _non_negative_int(_mapping_value(contact_summary, "spliced_contact_count"))
    mesh_unavailable_contacts = _non_negative_int(_mapping_value(contact_summary, "mesh_unavailable_count"))
    fallback_spliced_contacts = _non_negative_int(_mapping_value(contact_summary, "fallback_spliced_count"))
    accounted_contacts = min(scheduled_contacts, mesh_contact_count + mesh_unavailable_contacts)
    contact_mesh_coverage = (accounted_contacts / scheduled_contacts) if scheduled_contacts else None
    clip_duration_s = _clip_duration_seconds(tracks)
    runtime_seconds = _runtime_seconds(runtime_timing)
    latency_seconds_per_video_minute = (
        round(runtime_seconds / (clip_duration_s / 60.0), 6)
        if runtime_seconds is not None and clip_duration_s > 0.0
        else None
    )
    quality_blockers = _string_list(body_joint_quality.get("quality_blockers")) if isinstance(body_joint_quality, Mapping) else []
    quality_usable = bool(body_joint_quality.get("usable_for_review")) if isinstance(body_joint_quality, Mapping) else False
    blockers: list[str] = []
    warnings: list[str] = []

    if tracks is None:
        blockers.append("missing_tracks_json")
    if body_compute_execution is None:
        blockers.append("missing_body_compute_execution")
    if body_joint_quality is None:
        blockers.append("missing_body_joint_quality")
    if tracked_player_frames == 0:
        blockers.append("no_tracked_player_frames")
    if (
        coverage_denominator_policy == ELIGIBLE_SCHEDULED_MEASURED_COVERAGE_POLICY
        and coverage_denominator_player_frames == 0
    ):
        blockers.append("no_eligible_scheduled_measured_samples")
    if body_joint_quality is not None and (not quality_usable or quality_blockers):
        blockers.append("body_joint_quality_blocked")
    if coverage_denominator_player_frames > 0 and coverage < min_coverage:
        blockers.append("full_clip_body_coverage_below_threshold")
    if scheduled_contacts > 0 and accounted_contacts < scheduled_contacts:
        blockers.append("contact_mesh_or_unavailable_coverage_incomplete")
    if (
        coverage_denominator_policy == DEFAULT_COVERAGE_DENOMINATOR_POLICY
        and scheduled_player_frames < tracked_player_frames
    ):
        warnings.append("body_not_scheduled_for_all_tracked_player_frames")

    blockers = _dedupe(blockers)
    passed = not blockers
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "clip": clip,
        "passed": passed,
        "coverage": round(coverage, 6),
        "contact_mesh_coverage": round(contact_mesh_coverage, 6) if contact_mesh_coverage is not None else None,
        "latency_seconds_per_video_minute": latency_seconds_per_video_minute,
        "evaluated_frame_count": joint_player_frames,
        "min_coverage": min_coverage,
        "tracked_player_frame_count": tracked_player_frames,
        "coverage_denominator_policy": coverage_denominator_policy,
        "coverage_denominator_player_frame_count": coverage_denominator_player_frames,
        "evaluated_player_frame_count": joint_player_frames,
        "summary": {
            "tracked_player_frame_count": tracked_player_frames,
            "coverage_denominator_policy": coverage_denominator_policy,
            "coverage_denominator_player_frame_count": coverage_denominator_player_frames,
            "scheduled_player_frame_count": scheduled_player_frames,
            "joint_player_frame_count": joint_player_frames,
            "mesh_joint_player_frame_count": mesh_joint_player_frames,
            "skeleton_joint_player_frame_count": skeleton_joint_player_frames,
            "scheduled_contact_count": scheduled_contacts,
            "contact_mesh_frame_count": mesh_contact_count,
            "mesh_unavailable_contact_count": mesh_unavailable_contacts,
            "fallback_spliced_contact_count": fallback_spliced_contacts,
            "contact_mesh_accounted_count": accounted_contacts,
            "clip_duration_s": round(clip_duration_s, 6),
            "body_runtime_seconds": runtime_seconds,
            "latency_seconds_per_video_minute": latency_seconds_per_video_minute,
            "quality_status": str(body_joint_quality.get("status", "")) if isinstance(body_joint_quality, Mapping) else "",
            "quality_usable_for_review": quality_usable,
            "quality_blockers": quality_blockers,
        },
        "blockers": blockers,
        "warnings": _dedupe(warnings),
        "paths": {
            "tracks": tracks_path or "",
            "body_compute_execution": body_compute_execution_path or "",
            "body_joint_quality": body_joint_quality_path or "",
            "contact_splice": contact_splice_path or "",
            "runtime_timing": runtime_timing_path or "",
        },
        "execution": {
            "cpu_only": True,
            "uses_gpu": False,
            "runs_body_model": False,
            "claims_accuracy_verified": False,
        },
    }


def build_body_full_clip_gate_from_paths(
    *,
    clip: str,
    tracks_path: str | Path | None,
    body_compute_execution_path: str | Path | None,
    body_joint_quality_path: str | Path | None,
    contact_splice_path: str | Path | None = None,
    runtime_timing_path: str | Path | None = None,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
) -> dict[str, Any]:
    return build_body_full_clip_gate(
        clip=clip,
        tracks=_read_optional_json(tracks_path),
        body_compute_execution=_read_optional_json(body_compute_execution_path),
        body_joint_quality=_read_optional_json(body_joint_quality_path),
        contact_splice=_read_optional_json(contact_splice_path),
        runtime_timing=_read_optional_json(runtime_timing_path),
        tracks_path=str(tracks_path or ""),
        body_compute_execution_path=str(body_compute_execution_path or ""),
        body_joint_quality_path=str(body_joint_quality_path or ""),
        contact_splice_path=str(contact_splice_path or ""),
        runtime_timing_path=str(runtime_timing_path or ""),
        min_coverage=min_coverage,
    )


def write_body_full_clip_gate(path: str | Path, payload: Mapping[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _tracked_player_frame_count(tracks: Mapping[str, Any] | None) -> int:
    players = tracks.get("players") if isinstance(tracks, Mapping) else None
    if not isinstance(players, list):
        return 0
    total = 0
    for player in players:
        if not isinstance(player, Mapping):
            continue
        frames = player.get("frames")
        if isinstance(frames, list):
            total += len(frames)
    return total


def _clip_duration_seconds(tracks: Mapping[str, Any] | None) -> float:
    if not isinstance(tracks, Mapping):
        return 0.0
    fps = _positive_float(tracks.get("fps"))
    players = tracks.get("players")
    if not isinstance(players, list):
        return 0.0
    frame_counts: list[int] = []
    max_time: float | None = None
    for player in players:
        if not isinstance(player, Mapping):
            continue
        frames = player.get("frames")
        if not isinstance(frames, list) or not frames:
            continue
        frame_counts.append(len(frames))
        for frame in frames:
            if not isinstance(frame, Mapping):
                continue
            try:
                frame_time = float(frame.get("t"))
            except (TypeError, ValueError):
                continue
            max_time = frame_time if max_time is None else max(max_time, frame_time)
    if fps > 0.0 and frame_counts:
        return max(frame_counts) / fps
    if max_time is not None:
        return max_time
    return 0.0


def _runtime_seconds(runtime_timing: Mapping[str, Any] | None) -> float | None:
    if not isinstance(runtime_timing, Mapping):
        return None
    for key in ("body_wall_seconds", "wall_seconds", "elapsed_seconds", "runtime_seconds"):
        value = _positive_float(runtime_timing.get(key))
        if value > 0.0:
            return value
    return None


def _positive_float(value: Any) -> float:
    if isinstance(value, bool):
        return 0.0
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if number > 0.0 else 0.0


def _read_optional_json(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    candidate = Path(path)
    if not candidate.is_file():
        return None
    return json.loads(candidate.read_text(encoding="utf-8"))


def _mapping_value(value: Any, key: str) -> Any:
    return value.get(key) if isinstance(value, Mapping) else None


def _non_negative_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    try:
        number = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, number)


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item)]


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        if value not in out:
            out.append(value)
    return out

"""Adaptive BODY compute scheduling helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .schemas import Tracks


ARTIFACT_TYPE = "racketsport_body_compute_execution"
SCHEMA_VERSION = 1


def build_body_compute_execution(
    tracks: Tracks,
    *,
    frame_plan_path: str | Path | None = None,
    max_frames: int | None = None,
) -> dict[str, Any]:
    """Return the BODY frames that should invoke deep mesh compute.

    If ``frame_plan_path`` exists, only frames inside ``deep_mesh_windows`` are
    scheduled. Other plan frames are preserved as skipped review records. When
    no plan exists, the runner keeps its historical behavior and schedules every
    tracked frame.
    """

    if max_frames is not None and max_frames < 0:
        raise ValueError("max_frames must be non-negative")

    track_lookup = _track_lookup(tracks)
    plan_path = Path(frame_plan_path) if frame_plan_path is not None else None
    if plan_path is not None and plan_path.is_file():
        return _execution_from_frame_plan(tracks, plan_path=plan_path, track_lookup=track_lookup, max_frames=max_frames)
    return _execution_from_tracks(tracks, track_lookup=track_lookup, max_frames=max_frames)


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
    max_frames: int | None,
) -> dict[str, Any]:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    frame_lookup = {int(frame["frame_idx"]): frame for frame in plan.get("frames", [])}
    scheduled: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    scheduled_indexes: set[int] = set()

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
            scheduled_indexes.add(frame_idx)
            scheduled.append(
                {
                    "frame_idx": frame_idx,
                    "t": frame_idx / tracks.fps,
                    "target_representation": "world_mesh",
                    "target_player_ids": target_ids,
                    "active_player_ids": [player_id for player_id, _frame in active],
                    "source_window_index": window_index,
                    "reason_counts": dict(window.get("reason_counts", {})),
                    "reasons": list(frame_plan.get("reasons", [])),
                    "max_score": float(window.get("max_score", frame_plan.get("score", 0.0))),
                    "player_targets": _selected_player_targets(frame_plan, target_ids),
                }
            )

    for frame_idx, frame_plan in sorted(frame_lookup.items()):
        if frame_idx in scheduled_indexes:
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
    )


def _execution_from_tracks(
    tracks: Tracks,
    *,
    track_lookup: dict[int, list[tuple[int, Any]]],
    max_frames: int | None,
) -> dict[str, Any]:
    scheduled = [
        {
            "frame_idx": frame_idx,
            "t": frame_idx / tracks.fps,
            "target_representation": "world_mesh",
            "target_player_ids": [player_id for player_id, _frame in active],
            "active_player_ids": [player_id for player_id, _frame in active],
            "source_window_index": None,
            "reason_counts": {},
            "reasons": ["no_frame_compute_plan"],
            "max_score": None,
            "player_targets": [
                {
                    "player_id": player_id,
                    "track_conf": round(float(track_frame.conf), 3),
                    "score": 1.0,
                    "recommended_tier": "deep_mesh",
                    "target_representation": "world_mesh",
                    "reasons": ["no_frame_compute_plan"],
                }
                for player_id, track_frame in active
            ],
        }
        for frame_idx, active in sorted(track_lookup.items())
    ]
    scheduled, skipped = _apply_max_frames(scheduled, max_frames=max_frames)
    return _execution_payload(
        tracks,
        mode="all_track_frames",
        source_plan=None,
        scheduled=scheduled,
        skipped=skipped,
    )


def _execution_payload(
    tracks: Tracks,
    *,
    mode: str,
    source_plan: str | None,
    scheduled: list[dict[str, Any]],
    skipped: list[dict[str, Any]],
) -> dict[str, Any]:
    scheduled_by_target_representation: dict[str, int] = {}
    scheduled_by_reason: dict[str, int] = {}
    scheduled_targeted_reviewed_contact_frame_count = 0
    scheduled_coverage_incomplete_frame_count = 0
    for frame in scheduled:
        target_representation = str(frame.get("target_representation", "unknown"))
        scheduled_by_target_representation[target_representation] = scheduled_by_target_representation.get(target_representation, 0) + 1
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
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "mode": mode,
        "source_plan": source_plan,
        "fps": tracks.fps,
        "scheduled_frames": scheduled,
        "skipped_frames": skipped,
        "summary": {
            "scheduled_frame_count": len(scheduled),
            "scheduled_player_frame_count": scheduled_player_frame_count,
            "scheduled_by_target_representation": dict(sorted(scheduled_by_target_representation.items())),
            "scheduled_by_reason": dict(sorted(scheduled_by_reason.items())),
            "scheduled_targeted_reviewed_contact_frame_count": scheduled_targeted_reviewed_contact_frame_count,
            "scheduled_coverage_incomplete_frame_count": scheduled_coverage_incomplete_frame_count,
            "skipped_frame_count": len(skipped),
            "skipped_by_tier": dict(sorted(skipped_by_tier.items())),
            "skipped_by_target_representation": dict(sorted(skipped_by_target_representation.items())),
            "skipped_by_reason": dict(sorted(skipped_by_reason.items())),
        },
    }


def _track_lookup(tracks: Tracks) -> dict[int, list[tuple[int, Any]]]:
    by_frame: dict[int, list[tuple[int, Any]]] = {}
    for player in tracks.players:
        for frame in player.frames:
            frame_idx = int(round(frame.t * tracks.fps))
            by_frame.setdefault(frame_idx, []).append((player.id, frame))
    return by_frame


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
            "recommended_tier": "deep_mesh",
            "target_representation": "world_mesh",
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

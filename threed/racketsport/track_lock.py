"""N-lock and identity persistence helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class TrackCandidate:
    track_id: int
    world_xy: list[float]
    confidence: float


@dataclass(frozen=True)
class TrackLockUpdate:
    locked: list[TrackCandidate]
    accepted: bool
    notes: list[str]


def n_lock(candidates: list[TrackCandidate], *, count: int) -> list[TrackCandidate]:
    if count <= 0:
        raise ValueError("count must be positive")
    if len(candidates) < count:
        raise ValueError(f"need {count} candidates to lock tracks")
    return sorted(candidates, key=lambda candidate: (-candidate.confidence, candidate.track_id))[:count]


def ground_step_plausible(previous_world_xy: list[float], next_world_xy: list[float], *, max_step_m: float) -> bool:
    if max_step_m <= 0:
        raise ValueError("max_step_m must be positive")
    distance = math.hypot(next_world_xy[0] - previous_world_xy[0], next_world_xy[1] - previous_world_xy[1])
    return distance <= max_step_m


def update_track_lock(
    previous_locked: list[TrackCandidate],
    candidates: list[TrackCandidate],
    *,
    max_step_m: float,
) -> TrackLockUpdate:
    if max_step_m <= 0:
        raise ValueError("max_step_m must be positive")
    if not previous_locked:
        return TrackLockUpdate(locked=[], accepted=False, notes=["no_locked_tracks"])

    current_by_id: dict[int, TrackCandidate] = {}
    duplicate_current_ids: set[int] = set()
    for candidate in candidates:
        if candidate.track_id in current_by_id:
            duplicate_current_ids.add(candidate.track_id)
        else:
            current_by_id[candidate.track_id] = candidate
    if duplicate_current_ids:
        duplicate_ids = ",".join(str(track_id) for track_id in sorted(duplicate_current_ids))
        return TrackLockUpdate(locked=previous_locked, accepted=False, notes=[f"duplicate_current_track:{duplicate_ids}"])

    seen_previous_ids: set[int] = set()
    duplicate_previous_ids: set[int] = set()
    for candidate in previous_locked:
        if candidate.track_id in seen_previous_ids:
            duplicate_previous_ids.add(candidate.track_id)
        seen_previous_ids.add(candidate.track_id)
    if duplicate_previous_ids:
        duplicate_ids = ",".join(str(track_id) for track_id in sorted(duplicate_previous_ids))
        return TrackLockUpdate(locked=previous_locked, accepted=False, notes=[f"duplicate_locked_track:{duplicate_ids}"])

    updated: list[TrackCandidate] = []
    rejection_notes: list[str] = []
    for previous in previous_locked:
        current = current_by_id.get(previous.track_id)
        if current is None:
            rejection_notes.append(f"missing_locked_track:{previous.track_id}")
            continue
        if not ground_step_plausible(previous.world_xy, current.world_xy, max_step_m=max_step_m):
            rejection_notes.append(f"implausible_ground_step:{previous.track_id}")
            continue
        updated.append(current)

    if rejection_notes:
        return TrackLockUpdate(locked=previous_locked, accepted=False, notes=rejection_notes)

    locked_ids = ",".join(str(candidate.track_id) for candidate in previous_locked)
    return TrackLockUpdate(locked=updated, accepted=True, notes=[f"preserved_locked_ids:{locked_ids}"])

"""N-lock and identity persistence helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class TrackCandidate:
    track_id: int
    world_xy: list[float]
    confidence: float


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

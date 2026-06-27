"""Doubles side and role assignment."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Literal

from .track_lock import TrackCandidate

CourtSide = Literal["near", "far"]
LateralRole = Literal["left", "right"]


@dataclass(frozen=True)
class DoublesIdentity:
    track_id: int
    side: CourtSide
    role: LateralRole
    label: str | None = None


def assign_doubles_roles(candidates: list[TrackCandidate]) -> dict[int, DoublesIdentity]:
    identities: dict[int, DoublesIdentity] = {}
    for candidate in candidates:
        side: CourtSide = "near" if candidate.world_xy[1] < 0.0 else "far"
        role: LateralRole = "left" if candidate.world_xy[0] < 0.0 else "right"
        identities[candidate.track_id] = DoublesIdentity(track_id=candidate.track_id, side=side, role=role)
    return identities


def coach_anchor(
    candidates: list[TrackCandidate],
    *,
    anchor_world_xy: list[float],
    label: str,
    max_distance_m: float,
) -> DoublesIdentity:
    if max_distance_m <= 0:
        raise ValueError("max_distance_m must be positive")
    if not candidates:
        raise ValueError("no tracks to anchor")

    nearest = min(
        candidates,
        key=lambda candidate: math.hypot(candidate.world_xy[0] - anchor_world_xy[0], candidate.world_xy[1] - anchor_world_xy[1]),
    )
    distance = math.hypot(nearest.world_xy[0] - anchor_world_xy[0], nearest.world_xy[1] - anchor_world_xy[1])
    if distance > max_distance_m:
        raise ValueError("no track within anchor radius")

    side: CourtSide = "near" if nearest.world_xy[1] < 0.0 else "far"
    role: LateralRole = "left" if nearest.world_xy[0] < 0.0 else "right"
    return DoublesIdentity(track_id=nearest.track_id, side=side, role=role, label=label)


def apply_coach_anchor(
    identities: dict[int, DoublesIdentity],
    candidates: list[TrackCandidate],
    *,
    anchor_world_xy: list[float],
    label: str,
    max_distance_m: float,
) -> dict[int, DoublesIdentity]:
    if max_distance_m <= 0:
        raise ValueError("max_distance_m must be positive")
    if not identities:
        raise ValueError("no identities to anchor")

    eligible = [candidate for candidate in candidates if candidate.track_id in identities]
    if not eligible:
        raise ValueError("no identity within anchor radius")

    nearest = min(
        eligible,
        key=lambda candidate: math.hypot(candidate.world_xy[0] - anchor_world_xy[0], candidate.world_xy[1] - anchor_world_xy[1]),
    )
    distance = math.hypot(nearest.world_xy[0] - anchor_world_xy[0], nearest.world_xy[1] - anchor_world_xy[1])
    if distance > max_distance_m:
        raise ValueError("no identity within anchor radius")

    anchored = dict(identities)
    anchored[nearest.track_id] = replace(identities[nearest.track_id], label=label)
    return anchored

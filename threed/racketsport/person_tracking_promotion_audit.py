"""Audit person tracks before treating tracker candidates as canonical-safe."""

from __future__ import annotations

from typing import Any

from .court_templates import Sport, get_court_template
from .schemas import Tracks


def build_person_tracking_promotion_audit(
    *,
    tracks: Tracks,
    clip: str,
    variant: str,
    court_margin_m: float,
    max_players: int,
    total_frames: int | None = None,
    labeled_gate_passed: bool = False,
    sport: Sport = "pickleball",
) -> dict[str, Any]:
    if court_margin_m < 0.0:
        raise ValueError("court_margin_m must be non-negative")
    if max_players <= 0:
        raise ValueError("max_players must be positive")

    track_frame_count = sum(len(player.frames) for player in tracks.players)
    outside = _outside_court_frames(tracks, sport=sport)
    safety_blockers: list[str] = []
    if court_margin_m > 0.0:
        safety_blockers.append("widened_court_margin_diagnostic_only")
    if outside["outside_court_frame_count"] > 0:
        safety_blockers.append("off_court_player_frames_present")
    if len(tracks.players) < max_players:
        safety_blockers.append("missing_expected_players")
    if len(tracks.players) > max_players:
        safety_blockers.append("too_many_player_tracks")

    safe_for_canonical_review = not safety_blockers
    promotion_blockers = list(safety_blockers)
    if not labeled_gate_passed:
        promotion_blockers.append("labeled_idf1_spectator_gate_missing")
    trusted_for_trk_promotion = safe_for_canonical_review and labeled_gate_passed

    return {
        "schema_version": 1,
        "artifact_type": "racketsport_person_tracking_promotion_audit",
        "clip": clip,
        "variant": variant,
        "status": _status(
            safe_for_canonical_review=safe_for_canonical_review,
            trusted_for_trk_promotion=trusted_for_trk_promotion,
        ),
        "safe_for_canonical_review": safe_for_canonical_review,
        "diagnostic_only": bool(safety_blockers),
        "trusted_for_trk_promotion": trusted_for_trk_promotion,
        "labeled_gate_passed": bool(labeled_gate_passed),
        "max_players": int(max_players),
        "player_count": len(tracks.players),
        "court_margin_m": float(court_margin_m),
        "safety_blockers": safety_blockers,
        "promotion_blockers": promotion_blockers,
        "track_safety": {
            **outside,
            "track_frame_count": track_frame_count,
            "outside_court_frame_rate": (outside["outside_court_frame_count"] / track_frame_count)
            if track_frame_count
            else 0.0,
        },
        "coverage": _coverage_summary(tracks, total_frames=total_frames, target_players=max_players),
        "canonical_policy": {
            "requires_strict_court_margin_m": 0.0,
            "requires_no_off_court_player_frames": True,
            "requires_labeled_idf1_spectator_gate": True,
            "notes": "widened court margins are diagnostics only until labeled spectator/ID gates pass",
        },
        "recommended_action": _recommended_action(
            safe_for_canonical_review=safe_for_canonical_review,
            trusted_for_trk_promotion=trusted_for_trk_promotion,
        ),
    }


def _outside_court_frames(tracks: Tracks, *, sport: Sport) -> dict[str, Any]:
    template = get_court_template(sport)
    half_width_m = template.width_m / 2.0
    half_length_m = template.length_m / 2.0
    outside_count = 0
    outside_player_ids: set[int] = set()
    epsilon = 1e-6
    for player in tracks.players:
        for frame in player.frames:
            x, y = [float(value) for value in frame.world_xy]
            if (
                x < -half_width_m - epsilon
                or x > half_width_m + epsilon
                or y < -half_length_m - epsilon
                or y > half_length_m + epsilon
            ):
                outside_count += 1
                outside_player_ids.add(player.id)
    return {
        "outside_court_frame_count": outside_count,
        "outside_court_player_ids": sorted(outside_player_ids),
    }


def _coverage_summary(tracks: Tracks, *, total_frames: int | None, target_players: int) -> dict[str, Any]:
    inferred_total_frames = _infer_total_frames(tracks)
    frame_count = int(total_frames) if total_frames is not None else inferred_total_frames
    frame_count = max(0, frame_count)
    active_counts = [0 for _ in range(frame_count)]
    for player in tracks.players:
        for frame in player.frames:
            frame_index = int(round(float(frame.t) * float(tracks.fps)))
            if 0 <= frame_index < frame_count:
                active_counts[frame_index] += 1
    target_frames = sum(1 for count in active_counts if count >= target_players)
    return {
        "total_frames": frame_count,
        "frames_with_any_player": sum(1 for count in active_counts if count > 0),
        "target_players": target_players,
        "target_player_frames": target_frames,
        "target_player_frame_rate": (target_frames / frame_count) if frame_count else 0.0,
        "mean_active_players": (sum(active_counts) / frame_count) if frame_count else 0.0,
    }


def _infer_total_frames(tracks: Tracks) -> int:
    max_index = -1
    for player in tracks.players:
        for frame in player.frames:
            max_index = max(max_index, int(round(float(frame.t) * float(tracks.fps))))
    return max_index + 1


def _status(*, safe_for_canonical_review: bool, trusted_for_trk_promotion: bool) -> str:
    if trusted_for_trk_promotion:
        return "trusted_for_trk_promotion"
    if safe_for_canonical_review:
        return "canonical_candidate_not_gate_verified"
    return "diagnostic_only"


def _recommended_action(*, safe_for_canonical_review: bool, trusted_for_trk_promotion: bool) -> str:
    if trusted_for_trk_promotion:
        return "eligible for canonical TRK promotion"
    if safe_for_canonical_review:
        return "keep as canonical-safe candidate and run labeled IDF1/spectator/ID-switch gates"
    return "keep diagnostic-only; do not promote to canonical tracks.json"


__all__ = ["build_person_tracking_promotion_audit"]

"""Adaptive frame compute-rating primitives.

This module ranks frames for heavier downstream compute. It intentionally does
not schedule GPU work; it emits an inspectable plan that later runners can use.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

from pydantic import ValidationError

from .schemas import BallTrack, ContactWindows, Tracks, validate_artifact_file


SCHEMA_VERSION = 1
ARTIFACT_TYPE = "racketsport_frame_compute_plan"
DEFAULT_MESH_COVERAGE_MODE = "hybrid"
DEFAULT_TARGET_MESH_FRAME_BUDGET = 200
MESH_COVERAGE_MODES = ("contact_only", "uniform", "hybrid", "ball_aware")
CONTACT_BOOST_SELECTION_REASON = "contact_boost"
UNIFORM_MESH_SELECTION_REASON = "uniform_mesh_coverage"
MESH_FALLBACK_REASON_ELIGIBLE_ZERO_ALL_MANUAL_REVIEW = "eligible_zero_all_manual_review"

# --- ball-aware tier-1 (SAM-3D mesh) scheduling ---------------------------
#
# OWNER DIRECTIVE (2026-07-03): tier-1 mesh moments must be BALL-AWARE, not
# triggered by raw wrist-cue windows. `mesh_coverage_mode="ball_aware"` is
# built from exactly three signals, all of which must be provided by the
# caller (they are optional artifacts, not part of the strict schema
# registry -- both are explicitly render_only/not_for_detection_metrics
# advisory outputs of the ball-arc solver, so they are read defensively as
# plain mappings, never pydantic-validated):
#   (a) `ball_aware_events`  -- events_selected.json from
#       scripts/racketsport/solve_ball_arcs.py: physically-validated contact
#       events only (residual-reduction + physical-plausibility checked),
#       typically a handful per clip vs. dozens of raw wrist-cue candidates.
#   (b) `ball_track_arc_solved` -- ball_track_arc_solved.json from the same
#       solver: per-frame arc-solved ball world position. A frame triggers
#       when an active player's court-plane world position is within
#       `ball_proximity_m` (horizontal XY distance -- Tracks carries no
#       player height) of the ball's world_xyz projected to XY.
#   (c) high-confidence swing cues -- the existing wrist+ball fused
#       `contact_windows` events, but only those at/above
#       `high_confidence_swing_floor` confidence (NOT every raw cue).
# When `mesh_coverage_mode="ball_aware"`, the legacy `contact_window` reason
# (all raw fused wrist-cue events regardless of confidence) is suppressed
# from scoring/candidate selection entirely -- see the per-frame loop below.
DEFAULT_BALL_PROXIMITY_M = 1.5
DEFAULT_HIGH_CONFIDENCE_SWING_FLOOR = 0.6
DEFAULT_BALL_AWARE_PADDING_S = 0.08
BALL_AWARE_CONTACT_REASON = "ball_aware_contact"
BALL_PROXIMITY_REASON = "ball_proximity"
HIGH_CONFIDENCE_SWING_REASON = "high_confidence_swing"
BALL_AWARE_TRIGGER_REASONS = (BALL_AWARE_CONTACT_REASON, BALL_PROXIMITY_REASON, HIGH_CONFIDENCE_SWING_REASON)
BALL_AWARE_BOOST_SELECTION_REASON = "ball_aware_boost"
# Each of the three trigger scores independently clears the >=0.5
# deep_mesh threshold in _recommended_tier (mirroring how the legacy
# contact_window/missing_expected_players reasons are each >=0.5 alone),
# so a frame reachable ONLY by proximity or ONLY by a high-confidence
# swing still becomes a ball_aware mesh candidate on its own, not just
# when stacked with another reason. Relative ordering (contact > proximity
# > swing) matches the owner's priority for _select_priority_indexes
# budget tie-breaking.
BALL_AWARE_CONTACT_SCORE = 0.90
BALL_PROXIMITY_SCORE = 0.65
HIGH_CONFIDENCE_SWING_SCORE = 0.55


def build_frame_compute_plan(
    tracks: Tracks | Mapping[str, Any],
    *,
    ball_track: BallTrack | Mapping[str, Any] | None = None,
    contact_windows: ContactWindows | Mapping[str, Any] | None = None,
    expected_players: int = 4,
    low_track_confidence: float = 0.5,
    low_ball_confidence: float = 0.4,
    contact_padding_s: float = 0.08,
    mesh_coverage_mode: str = DEFAULT_MESH_COVERAGE_MODE,
    target_mesh_frame_budget: int | None = DEFAULT_TARGET_MESH_FRAME_BUDGET,
    ball_aware_events: Mapping[str, Any] | None = None,
    ball_track_arc_solved: Mapping[str, Any] | None = None,
    ball_proximity_m: float = DEFAULT_BALL_PROXIMITY_M,
    high_confidence_swing_floor: float = DEFAULT_HIGH_CONFIDENCE_SWING_FLOOR,
    ball_aware_padding_s: float = DEFAULT_BALL_AWARE_PADDING_S,
) -> dict[str, Any]:
    """Return a CPU-only frame priority plan for adaptive deep compute.

    ``ball_aware_events``/``ball_track_arc_solved`` are the ball-arc-solver's
    advisory outputs (events_selected.json / ball_track_arc_solved.json).
    They are read as plain mappings (not schema-validated -- both artifacts
    self-describe as render_only/not_for_detection_metrics/candidate,
    outside the strict-artifact registry) and only matter when
    ``mesh_coverage_mode="ball_aware"`` selects mesh candidates from them;
    see the module docstring block above for the exact trigger semantics.
    """

    if expected_players <= 0:
        raise ValueError("expected_players must be positive")
    if not 0.0 <= low_track_confidence <= 1.0:
        raise ValueError("low_track_confidence must be between 0 and 1")
    if not 0.0 <= low_ball_confidence <= 1.0:
        raise ValueError("low_ball_confidence must be between 0 and 1")
    if contact_padding_s < 0.0:
        raise ValueError("contact_padding_s must be non-negative")
    mesh_coverage_mode = str(mesh_coverage_mode)
    if mesh_coverage_mode not in MESH_COVERAGE_MODES:
        raise ValueError(f"mesh_coverage_mode must be one of {', '.join(MESH_COVERAGE_MODES)}")
    if target_mesh_frame_budget is not None and target_mesh_frame_budget <= 0:
        raise ValueError("target_mesh_frame_budget must be positive when provided")
    if ball_proximity_m <= 0.0:
        raise ValueError("ball_proximity_m must be positive")
    if not 0.0 <= high_confidence_swing_floor <= 1.0:
        raise ValueError("high_confidence_swing_floor must be between 0 and 1")
    if ball_aware_padding_s < 0.0:
        raise ValueError("ball_aware_padding_s must be non-negative")

    tracks_obj = _tracks(tracks)
    if tracks_obj.fps <= 0.0:
        raise ValueError("tracks.fps must be positive")
    ball_obj = _ball_track(ball_track)
    contacts_obj = _contact_windows(contact_windows)
    ball_aware_only = mesh_coverage_mode == "ball_aware"

    track_frames = _track_frames_by_index(tracks_obj)
    ball_frames = _ball_frames_by_index(ball_obj, fps=tracks_obj.fps)
    contact_spans = _contact_spans(contacts_obj, padding_s=contact_padding_s)
    # The three ball-aware trigger signals are opt-in via mesh_coverage_mode:
    # only computed/scored under "ball_aware" so that passing
    # ball_aware_events/ball_track_arc_solved (or having a high-confidence
    # contact_windows event) never changes scoring/selection for the other
    # (pre-existing) coverage modes.
    ball_aware_contact_spans = _ball_aware_contact_spans(ball_aware_events, padding_s=ball_aware_padding_s) if ball_aware_only else []
    high_confidence_swing_spans = (
        _high_confidence_swing_spans(contacts_obj, floor=high_confidence_swing_floor, padding_s=ball_aware_padding_s)
        if ball_aware_only
        else []
    )
    proximity_frame_players = (
        _ball_proximity_frame_players(
            ball_track_arc_solved,
            track_frames=track_frames,
            fps=tracks_obj.fps,
            proximity_m=ball_proximity_m,
        )
        if ball_aware_only
        else {}
    )
    frame_indexes = sorted(
        set(track_frames)
        | set(ball_frames)
        | {_frame_index(span["t"], tracks_obj.fps) for span in contact_spans}
        | {_frame_index(span["t"], tracks_obj.fps) for span in ball_aware_contact_spans}
        | {_frame_index(span["t"], tracks_obj.fps) for span in high_confidence_swing_spans}
        | set(proximity_frame_players)
    )

    frames: list[dict[str, Any]] = []
    for frame_idx in frame_indexes:
        t = frame_idx / tracks_obj.fps
        active_tracks = track_frames.get(frame_idx, [])
        reasons: list[str] = []
        score = 0.0

        frame_contact_spans = _contact_spans_for_time(t, contact_spans)
        frame_ba_contact_spans = _contact_spans_for_time(t, ball_aware_contact_spans)
        frame_swing_spans = _contact_spans_for_time(t, high_confidence_swing_spans)
        frame_proximity_players = proximity_frame_players.get(frame_idx, frozenset())
        active_player_ids = {player_id for player_id, _frame in active_tracks}
        reviewed_target_player_ids = _reviewed_target_player_ids(frame_contact_spans, active_player_ids)

        if frame_contact_spans and not ball_aware_only:
            reasons.append("contact_window")
            score += 0.55
        if frame_ba_contact_spans:
            reasons.append(BALL_AWARE_CONTACT_REASON)
            score += BALL_AWARE_CONTACT_SCORE
        if frame_proximity_players:
            reasons.append(BALL_PROXIMITY_REASON)
            score += BALL_PROXIMITY_SCORE
        if frame_swing_spans:
            reasons.append(HIGH_CONFIDENCE_SWING_REASON)
            score += HIGH_CONFIDENCE_SWING_SCORE

        missing_players = max(0, expected_players - len(active_tracks))
        if missing_players >= 2 and reviewed_target_player_ids:
            reasons.append("missing_expected_players")
            reasons.append("reviewed_contact_targeted_body")
        elif missing_players >= 2:
            reasons.append("missing_expected_players")
            score += 0.65

        min_track_conf = min((float(frame.conf) for _player_id, frame in active_tracks), default=0.0)
        if active_tracks and min_track_conf < low_track_confidence:
            reasons.append("low_track_confidence")
            score += 0.25

        ball_frame = ball_frames.get(frame_idx)
        ball_conf = float(ball_frame.conf) if ball_frame is not None else None
        if ball_frame is not None and (not ball_frame.visible or ball_frame.conf < low_ball_confidence):
            ball_reason = "ball_uncertain"
            reasons.append(ball_reason)
            score += 0.20
        elif ball_obj is not None and ball_frame is None:
            ball_reason = "ball_missing"
            reasons.append(ball_reason)
            score += 0.20
        else:
            ball_reason = None

        recommended_tier = _recommended_tier(reasons, score)
        target_representation = _target_representation(recommended_tier)
        player_targets = _player_targets(
            active_tracks,
            frame_contact_spans=frame_contact_spans,
            missing_players=missing_players,
            low_track_confidence=low_track_confidence,
            ball_reason=ball_reason,
            reviewed_target_player_ids=reviewed_target_player_ids,
            frame_ba_contact_spans=frame_ba_contact_spans,
            frame_proximity_players=frame_proximity_players,
            frame_swing_spans=frame_swing_spans,
            include_contact_window=not ball_aware_only,
        )
        frames.append(
            {
                "frame_idx": frame_idx,
                "t": t,
                "score": round(min(score, 1.0), 3),
                "recommended_tier": recommended_tier,
                "target_representation": target_representation,
                "reasons": reasons,
                "active_players": len(active_tracks),
                "active_player_ids": sorted(player_id for player_id, _frame in active_tracks),
                "missing_players": missing_players,
                "min_track_conf": min_track_conf,
                "ball_conf": ball_conf,
                "player_targets": player_targets,
                "tier_rationale": {
                    "base_recommended_tier": recommended_tier,
                    "base_target_representation": target_representation,
                    "coverage_policy_mode": mesh_coverage_mode,
                    "mesh_selected": recommended_tier == "deep_mesh",
                    "selection_reasons": [CONTACT_BOOST_SELECTION_REASON]
                    if recommended_tier == "deep_mesh" and "contact_window" in reasons
                    else [],
                },
            }
        )

    mesh_coverage_policy = _apply_mesh_coverage_policy(
        frames,
        tracks_obj=tracks_obj,
        mode=mesh_coverage_mode,
        target_budget=target_mesh_frame_budget,
    )
    deep_mesh_windows = _deep_mesh_windows(frames, fps=tracks_obj.fps)
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "fps": tracks_obj.fps,
        "expected_players": expected_players,
        "mesh_coverage_policy": mesh_coverage_policy,
        "frame_count": len(frames),
        "frames": frames,
        "deep_mesh_windows": deep_mesh_windows,
        "summary": _summary(frames, deep_mesh_windows=deep_mesh_windows, mesh_coverage_policy=mesh_coverage_policy),
    }


def build_frame_compute_plan_from_files(
    *,
    tracks_path: str | Path,
    ball_track_path: str | Path | None = None,
    contact_windows_path: str | Path | None = None,
    expected_players: int = 4,
    mesh_coverage_mode: str = DEFAULT_MESH_COVERAGE_MODE,
    target_mesh_frame_budget: int | None = DEFAULT_TARGET_MESH_FRAME_BUDGET,
    ball_aware_events_path: str | Path | None = None,
    ball_track_arc_solved_path: str | Path | None = None,
    ball_proximity_m: float = DEFAULT_BALL_PROXIMITY_M,
    high_confidence_swing_floor: float = DEFAULT_HIGH_CONFIDENCE_SWING_FLOOR,
) -> dict[str, Any]:
    tracks = validate_artifact_file("tracks", Path(tracks_path))
    if not isinstance(tracks, Tracks):
        raise ValueError("tracks artifact did not parse as Tracks")

    ball_track = None
    if ball_track_path is not None:
        parsed = validate_artifact_file("ball_track", Path(ball_track_path))
        if not isinstance(parsed, BallTrack):
            raise ValueError("ball track artifact did not parse as BallTrack")
        ball_track = parsed

    contact_windows = None
    if contact_windows_path is not None:
        parsed = validate_artifact_file("contact_windows", Path(contact_windows_path))
        if not isinstance(parsed, ContactWindows):
            raise ValueError("contact windows artifact did not parse as ContactWindows")
        contact_windows = parsed

    # events_selected.json / ball_track_arc_solved.json are advisory
    # ball-arc-solver outputs, not part of the strict-artifact registry
    # (both self-describe render_only/not_for_detection_metrics); read
    # as plain JSON rather than through validate_artifact_file.
    ball_aware_events = None
    if ball_aware_events_path is not None:
        ball_aware_events = json.loads(Path(ball_aware_events_path).read_text(encoding="utf-8"))

    ball_track_arc_solved = None
    if ball_track_arc_solved_path is not None:
        ball_track_arc_solved = json.loads(Path(ball_track_arc_solved_path).read_text(encoding="utf-8"))

    return build_frame_compute_plan(
        tracks,
        ball_track=ball_track,
        contact_windows=contact_windows,
        expected_players=expected_players,
        mesh_coverage_mode=mesh_coverage_mode,
        target_mesh_frame_budget=target_mesh_frame_budget,
        ball_aware_events=ball_aware_events,
        ball_track_arc_solved=ball_track_arc_solved,
        ball_proximity_m=ball_proximity_m,
        high_confidence_swing_floor=high_confidence_swing_floor,
    )


def write_frame_compute_plan(path: str | Path, plan: Mapping[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _track_frames_by_index(tracks: Tracks) -> dict[int, list[tuple[int, Any]]]:
    by_index: dict[int, list[tuple[int, Any]]] = {}
    for player in tracks.players:
        for frame in player.frames:
            by_index.setdefault(_frame_index(frame.t, tracks.fps), []).append((player.id, frame))
    return by_index


def _ball_frames_by_index(ball_track: BallTrack | None, *, fps: float) -> dict[int, Any]:
    if ball_track is None:
        return {}
    return {_frame_index(frame.t, fps): frame for frame in ball_track.frames}


def _contact_spans(contact_windows: ContactWindows | None, *, padding_s: float) -> list[dict[str, float]]:
    if contact_windows is None:
        return []
    spans = []
    for event in contact_windows.events:
        spans.append(
            {
                "t": float(event.t),
                "t0": max(0.0, float(event.window.t0) - padding_s),
                "t1": float(event.window.t1) + padding_s,
                "importance": float(event.window.importance),
                "player_id": event.player_id,
                "human_review": event.sources.human_review,
            }
        )
    return spans


def _contact_spans_for_time(t: float, spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [span for span in spans if span["t0"] <= t <= span["t1"]]


def _ball_aware_contact_spans(
    ball_aware_events: Mapping[str, Any] | None, *, padding_s: float
) -> list[dict[str, Any]]:
    """Physically-validated contact events from events_selected.json.

    Only ``kind == "contact"`` entries in the top-level ``selected`` list
    count (bounces/rally-endpoints are ball-only events, not player-body
    trigger moments). These are the owner's "~6 real contacts" -- the
    ball-arc solver's residual-reduction + physical-plausibility filtered
    subset, never the raw wrist-reach-prior candidate pool.
    """
    if not isinstance(ball_aware_events, Mapping):
        return []
    spans: list[dict[str, Any]] = []
    for event in ball_aware_events.get("selected", []) or []:
        if not isinstance(event, Mapping) or event.get("kind") != "contact":
            continue
        t = float(event.get("t", 0.0))
        player_id = event.get("player_id")
        spans.append(
            {
                "t": t,
                "t0": max(0.0, t - padding_s),
                "t1": t + padding_s,
                "player_id": int(player_id) if player_id is not None else None,
                "anchor_id": event.get("anchor_id"),
                "candidate_confidence": float(event.get("candidate_confidence", 0.0)),
            }
        )
    return spans


def _high_confidence_swing_spans(
    contact_windows: ContactWindows | None, *, floor: float, padding_s: float
) -> list[dict[str, Any]]:
    """contact_windows events at/above ``floor`` confidence only.

    This reuses the same fused wrist+ball cue substrate as the legacy
    ``contact_window`` reason, but excludes every event below the
    confidence floor -- the point is to keep a swing-cue signal available
    under ball_aware scheduling without letting raw low-confidence wrist
    noise drive mesh placement.
    """
    if contact_windows is None:
        return []
    spans: list[dict[str, Any]] = []
    for event in contact_windows.events:
        if float(event.confidence) < floor:
            continue
        spans.append(
            {
                "t": float(event.t),
                "t0": max(0.0, float(event.window.t0) - padding_s),
                "t1": float(event.window.t1) + padding_s,
                "player_id": event.player_id,
                "confidence": float(event.confidence),
            }
        )
    return spans


def _ball_proximity_frame_players(
    ball_track_arc_solved: Mapping[str, Any] | None,
    *,
    track_frames: dict[int, list[tuple[int, Any]]],
    fps: float,
    proximity_m: float,
) -> dict[int, frozenset[int]]:
    """frame_idx -> set of player_ids within ``proximity_m`` of the arc-solved ball.

    Distance is horizontal (world XY court-plane) only: Tracks.TrackFrame
    carries a 2D ``world_xy`` (no player height), so this is a ground-plane
    proximity, not a full 3D reach distance. ``ball_track_arc_solved.json``
    frames with ``world_xyz is None`` (the solver's ``hidden`` band -- no
    trusted position) are skipped.
    """
    if not isinstance(ball_track_arc_solved, Mapping):
        return {}
    ball_frames = ball_track_arc_solved.get("frames")
    if not isinstance(ball_frames, list):
        return {}

    result: dict[int, set[int]] = {}
    for ball_frame in ball_frames:
        if not isinstance(ball_frame, Mapping):
            continue
        world_xyz = ball_frame.get("world_xyz")
        t = ball_frame.get("t")
        if world_xyz is None or t is None:
            continue
        frame_idx = _frame_index(float(t), fps)
        active = track_frames.get(frame_idx)
        if not active:
            continue
        bx, by = float(world_xyz[0]), float(world_xyz[1])
        near_players = {
            player_id
            for player_id, frame in active
            if math.hypot(float(frame.world_xy[0]) - bx, float(frame.world_xy[1]) - by) < proximity_m
        }
        if near_players:
            result[frame_idx] = near_players
    return {frame_idx: frozenset(players) for frame_idx, players in result.items()}


def _frame_index(t: float, fps: float) -> int:
    return int(round(float(t) * fps))


def _recommended_tier(reasons: list[str], score: float) -> str:
    if "reviewed_contact_targeted_body" in reasons:
        return "deep_mesh"
    if "missing_expected_players" in reasons:
        return "human_review"
    if score >= 0.5:
        return "deep_mesh"
    if score > 0.0:
        return "skeleton_preview"
    return "baseline"


def _target_representation(recommended_tier: str) -> str:
    if recommended_tier == "human_review":
        return "manual_review_required"
    if recommended_tier == "deep_mesh":
        return "world_mesh"
    if recommended_tier == "skeleton_preview":
        return "lane_a_skeleton"
    return "track_only"


def _player_targets(
    active_tracks: list[tuple[int, Any]],
    *,
    frame_contact_spans: list[dict[str, Any]],
    missing_players: int,
    low_track_confidence: float,
    ball_reason: str | None,
    reviewed_target_player_ids: set[int],
    frame_ba_contact_spans: list[dict[str, Any]] = (),
    frame_proximity_players: frozenset[int] = frozenset(),
    frame_swing_spans: list[dict[str, Any]] = (),
    include_contact_window: bool = True,
) -> list[dict[str, Any]]:
    if not active_tracks:
        return []

    if missing_players >= 2 and reviewed_target_player_ids:
        targets: list[dict[str, Any]] = []
        for player_id, frame in sorted(active_tracks):
            if player_id not in reviewed_target_player_ids:
                continue
            reasons = ["contact_window", "reviewed_contact_targeted_body"]
            track_conf = float(frame.conf)
            score = 0.55
            if track_conf < low_track_confidence:
                reasons.append("low_track_confidence")
                score += 0.25
            if ball_reason is not None:
                reasons.append(ball_reason)
                score += 0.20
            targets.append(
                _player_target(
                    player_id=player_id,
                    track_conf=track_conf,
                    score=score,
                    reasons=reasons,
                )
            )
        return targets

    if missing_players >= 2:
        return [
            _player_target(
                player_id=player_id,
                track_conf=float(frame.conf),
                score=0.65,
                reasons=["missing_expected_players"],
            )
            for player_id, frame in sorted(active_tracks)
        ]

    contact_player_ids = {
        int(span["player_id"])
        for span in frame_contact_spans
        if span.get("player_id") is not None
    }
    unassigned_contact = any(span.get("player_id") is None for span in frame_contact_spans)

    ba_contact_player_ids = {
        int(span["player_id"])
        for span in frame_ba_contact_spans
        if span.get("player_id") is not None
    }
    ba_contact_unassigned = any(span.get("player_id") is None for span in frame_ba_contact_spans)

    swing_player_ids = {
        int(span["player_id"])
        for span in frame_swing_spans
        if span.get("player_id") is not None
    }
    swing_unassigned = any(span.get("player_id") is None for span in frame_swing_spans)

    any_trigger_spans_this_frame = bool(
        frame_contact_spans or frame_ba_contact_spans or frame_swing_spans or frame_proximity_players
    )

    targets: list[dict[str, Any]] = []
    for player_id, frame in sorted(active_tracks):
        reasons: list[str] = []
        score = 0.0
        track_conf = float(frame.conf)

        if include_contact_window and frame_contact_spans and (unassigned_contact or player_id in contact_player_ids):
            reasons.append("contact_window")
            score += 0.55

        if frame_ba_contact_spans and (ba_contact_unassigned or player_id in ba_contact_player_ids):
            reasons.append(BALL_AWARE_CONTACT_REASON)
            score += BALL_AWARE_CONTACT_SCORE

        if player_id in frame_proximity_players:
            reasons.append(BALL_PROXIMITY_REASON)
            score += BALL_PROXIMITY_SCORE

        if frame_swing_spans and (swing_unassigned or player_id in swing_player_ids):
            reasons.append(HIGH_CONFIDENCE_SWING_REASON)
            score += HIGH_CONFIDENCE_SWING_SCORE

        if track_conf < low_track_confidence:
            reasons.append("low_track_confidence")
            score += 0.25

        if ball_reason is not None and (reasons or not any_trigger_spans_this_frame):
            reasons.append(ball_reason)
            score += 0.20

        targets.append(
            _player_target(
                player_id=player_id,
                track_conf=track_conf,
                score=score,
                reasons=reasons,
            )
        )
    return targets


def _reviewed_target_player_ids(frame_contact_spans: list[dict[str, Any]], active_player_ids: set[int]) -> set[int]:
    return {
        int(span["player_id"])
        for span in frame_contact_spans
        if span.get("player_id") is not None
        and int(span["player_id"]) in active_player_ids
        and float(span.get("human_review") or 0.0) > 0.0
    }


def _player_target(
    *,
    player_id: int,
    track_conf: float,
    score: float,
    reasons: list[str],
) -> dict[str, Any]:
    rounded_score = round(min(score, 1.0), 3)
    recommended_tier = _recommended_tier(reasons, rounded_score)
    return {
        "player_id": int(player_id),
        "track_conf": round(track_conf, 3),
        "score": rounded_score,
        "recommended_tier": recommended_tier,
        "target_representation": _target_representation(recommended_tier),
        "reasons": reasons,
    }


def _apply_mesh_coverage_policy(
    frames: list[dict[str, Any]],
    *,
    tracks_obj: Tracks,
    mode: str,
    target_budget: int | None,
) -> dict[str, Any]:
    base_deep_indexes = {
        int(frame["frame_idx"])
        for frame in frames
        if frame.get("recommended_tier") == "deep_mesh"
    }
    contact_candidate_indexes = {
        int(frame["frame_idx"])
        for frame in frames
        if frame.get("recommended_tier") == "deep_mesh"
        and (
            "contact_window" in {str(reason) for reason in frame.get("reasons", [])}
            or "reviewed_contact_targeted_body" in {str(reason) for reason in frame.get("reasons", [])}
        )
    }
    ball_aware_candidate_indexes = {
        int(frame["frame_idx"])
        for frame in frames
        if frame.get("recommended_tier") == "deep_mesh"
        and set(BALL_AWARE_TRIGGER_REASONS) & {str(reason) for reason in frame.get("reasons", [])}
    }
    eligible_indexes = [
        int(frame["frame_idx"])
        for frame in frames
        if int(frame.get("active_players", 0)) > 0
        and frame.get("target_representation") != "manual_review_required"
        and _frame_in_rally_spans(frame, tracks_obj=tracks_obj)
    ]
    eligible_index_set = set(eligible_indexes)
    budget = len(eligible_indexes) if target_budget is None else min(int(target_budget), len(eligible_indexes))

    contact_selected: set[int] = set()
    uniform_selected: set[int] = set()
    mesh_fallback: dict[str, Any] | None = None
    if mode == "contact_only":
        contact_selected = set(base_deep_indexes & eligible_index_set)
        if target_budget is not None and len(contact_selected) > budget:
            contact_selected = set(_select_priority_indexes(frames, contact_selected, budget))
    elif mode == "uniform":
        uniform_selected = set(
            _select_uniform_indexes(
                sorted(eligible_index_set),
                target_count=budget,
                tracks_obj=tracks_obj,
            )
        )
    elif mode == "ball_aware":
        candidate_budget = min(len(ball_aware_candidate_indexes), budget)
        contact_selected = set(
            _select_priority_indexes(frames, ball_aware_candidate_indexes & eligible_index_set, candidate_budget)
        )
        remaining_budget = max(0, budget - len(contact_selected))
        if remaining_budget:
            uniform_pool = sorted(eligible_index_set - contact_selected)
            uniform_selected = set(
                _select_uniform_indexes(
                    uniform_pool,
                    target_count=remaining_budget,
                    tracks_obj=tracks_obj,
                )
            )
        if not contact_selected and not uniform_selected:
            fallback_pool = _manual_review_mesh_fallback_pool(frames, tracks_obj=tracks_obj)
            if fallback_pool:
                fallback_budget = len(fallback_pool) if target_budget is None else min(int(target_budget), len(fallback_pool))
                uniform_selected = set(
                    _select_uniform_indexes(
                        fallback_pool,
                        target_count=fallback_budget,
                        tracks_obj=tracks_obj,
                    )
                )
                if uniform_selected:
                    mesh_fallback = {
                        "engaged": True,
                        "reason": MESH_FALLBACK_REASON_ELIGIBLE_ZERO_ALL_MANUAL_REVIEW,
                        "selected": len(uniform_selected),
                        "policy": "uniform_stride",
                    }
    else:
        contact_budget = min(len(contact_candidate_indexes), budget)
        contact_selected = set(_select_priority_indexes(frames, contact_candidate_indexes & eligible_index_set, contact_budget))
        remaining_budget = max(0, budget - len(contact_selected))
        if remaining_budget:
            uniform_pool = sorted(eligible_index_set - contact_selected)
            uniform_selected = set(
                _select_uniform_indexes(
                    uniform_pool,
                    target_count=remaining_budget,
                    tracks_obj=tracks_obj,
                )
            )

    boost_reason = BALL_AWARE_BOOST_SELECTION_REASON if mode == "ball_aware" else CONTACT_BOOST_SELECTION_REASON
    selected_indexes = contact_selected | uniform_selected
    for frame in frames:
        frame_idx = int(frame["frame_idx"])
        base_tier = str(frame["tier_rationale"]["base_recommended_tier"])
        base_representation = str(frame["tier_rationale"]["base_target_representation"])
        if frame_idx in selected_indexes:
            selection_reasons: list[str] = []
            if frame_idx in contact_selected:
                selection_reasons.append(boost_reason)
            if frame_idx in uniform_selected:
                selection_reasons.append(UNIFORM_MESH_SELECTION_REASON)
            if mesh_fallback is None or frame_idx not in uniform_selected:
                _promote_frame_to_mesh(frame, selection_reasons=selection_reasons)
        else:
            _restore_non_selected_tier(
                frame,
                base_tier=base_tier,
                base_representation=base_representation,
            )
        frame["tier_rationale"] = {
            "base_recommended_tier": base_tier,
            "base_target_representation": base_representation,
            "coverage_policy_mode": mode,
            "mesh_selected": frame_idx in selected_indexes,
            "selection_reasons": _selection_reasons_for_frame(
                frame_idx, contact_selected, uniform_selected, boost_reason=boost_reason
            ),
        }

    uniform_stride = _uniform_stride(uniform_selected)
    selected_count = len(selected_indexes)
    budget_limited_pool_count = len(_manual_review_mesh_fallback_pool(frames, tracks_obj=tracks_obj)) if mesh_fallback else len(eligible_indexes)
    policy = {
        "mode": mode,
        "target_mesh_frame_budget": target_budget,
        "eligible_mesh_frame_count": len(eligible_indexes),
        "selected_mesh_frame_count": selected_count,
        "contact_candidate_frame_count": len(contact_candidate_indexes & eligible_index_set),
        "uniform_selected_frame_count": len(uniform_selected),
        "contact_selected_frame_count": len(contact_selected),
        "rally_span_count": _rally_span_count(tracks_obj),
        "uniform_stride_frames": uniform_stride,
        "budget_limited": selected_count < budget_limited_pool_count,
        "ball_aware_candidate_frame_count": len(ball_aware_candidate_indexes & eligible_index_set) if mode == "ball_aware" else None,
        "ball_aware_trigger_source_counts": _ball_aware_trigger_source_counts(frames, contact_selected, uniform_selected)
        if mode == "ball_aware"
        else None,
    }
    if mesh_fallback is not None:
        policy["mesh_fallback"] = mesh_fallback
    return policy


def _manual_review_mesh_fallback_pool(frames: list[Mapping[str, Any]], *, tracks_obj: Tracks) -> list[int]:
    if not any(_frame_has_ball_evidence(frame) for frame in frames):
        return []
    active_rally_indexes = [
        int(frame["frame_idx"])
        for frame in frames
        if int(frame.get("active_players", 0)) > 0 and _frame_in_rally_spans(frame, tracks_obj=tracks_obj)
    ]
    if not active_rally_indexes:
        return []
    active_rally_lookup = {int(frame["frame_idx"]): frame for frame in frames if int(frame["frame_idx"]) in active_rally_indexes}
    if not all(
        active_rally_lookup[frame_idx].get("target_representation") == "manual_review_required"
        for frame_idx in active_rally_indexes
    ):
        return []
    return sorted(active_rally_indexes)


def _frame_has_ball_evidence(frame: Mapping[str, Any]) -> bool:
    if frame.get("ball_conf") is not None:
        return True
    reasons = {str(reason) for reason in frame.get("reasons", [])}
    return bool(reasons & {"ball_missing", "ball_uncertain"})


def _selection_reasons_for_frame(
    frame_idx: int, contact_selected: set[int], uniform_selected: set[int], *, boost_reason: str = CONTACT_BOOST_SELECTION_REASON
) -> list[str]:
    reasons: list[str] = []
    if frame_idx in contact_selected:
        reasons.append(boost_reason)
    if frame_idx in uniform_selected:
        reasons.append(UNIFORM_MESH_SELECTION_REASON)
    return reasons


def _ball_aware_trigger_source_counts(
    frames: list[dict[str, Any]], contact_selected: set[int], uniform_selected: set[int]
) -> dict[str, int]:
    """Trigger-source distribution for the scheduled (selected) mesh frames.

    A frame can carry more than one ball-aware reason (e.g. a physically
    validated contact that also happens to be a proximity window); counts
    below are overlapping tallies per source, not a mutually exclusive
    partition, so they can sum to more than ``selected_mesh_frame_count``.
    """
    frame_lookup = {int(frame["frame_idx"]): frame for frame in frames}
    counts = {"events": 0, "proximity": 0, "swing": 0, "uniform_fill": len(uniform_selected)}
    reason_key = {
        BALL_AWARE_CONTACT_REASON: "events",
        BALL_PROXIMITY_REASON: "proximity",
        HIGH_CONFIDENCE_SWING_REASON: "swing",
    }
    for frame_idx in contact_selected:
        frame_reasons = {str(reason) for reason in frame_lookup.get(frame_idx, {}).get("reasons", [])}
        for reason, key in reason_key.items():
            if reason in frame_reasons:
                counts[key] += 1
    return counts


def _promote_frame_to_mesh(frame: dict[str, Any], *, selection_reasons: list[str]) -> None:
    frame["recommended_tier"] = "deep_mesh"
    frame["target_representation"] = "world_mesh"
    if UNIFORM_MESH_SELECTION_REASON in selection_reasons:
        frame["score"] = round(max(float(frame.get("score", 0.0)), 1.0), 3)
        reasons = [str(reason) for reason in frame.get("reasons", [])]
        if UNIFORM_MESH_SELECTION_REASON not in reasons:
            reasons.append(UNIFORM_MESH_SELECTION_REASON)
        frame["reasons"] = reasons
        frame["player_targets"] = _uniform_mesh_player_targets(frame)


def _restore_non_selected_tier(
    frame: dict[str, Any],
    *,
    base_tier: str,
    base_representation: str,
) -> None:
    if base_representation == "world_mesh":
        reasons = [str(reason) for reason in frame.get("reasons", [])]
        tier = "skeleton_preview" if reasons else "baseline"
        representation = _target_representation(tier)
        frame["recommended_tier"] = tier
        frame["target_representation"] = representation
        frame["player_targets"] = [
            _demote_player_target(target, tier=tier, representation=representation)
            for target in frame.get("player_targets", [])
            if isinstance(target, Mapping)
        ]
        return
    frame["recommended_tier"] = base_tier
    frame["target_representation"] = base_representation


def _demote_player_target(target: Mapping[str, Any], *, tier: str, representation: str) -> dict[str, Any]:
    payload = dict(target)
    if payload.get("target_representation") == "world_mesh":
        payload["recommended_tier"] = tier
        payload["target_representation"] = representation
    return payload


def _uniform_mesh_player_targets(frame: Mapping[str, Any]) -> list[dict[str, Any]]:
    existing_targets = {
        int(target.get("player_id")): target
        for target in frame.get("player_targets", [])
        if isinstance(target, Mapping) and target.get("player_id") is not None
    }
    targets: list[dict[str, Any]] = []
    for player_id in sorted(int(player_id) for player_id in frame.get("active_player_ids", [])):
        track_conf = float(existing_targets.get(player_id, {}).get("track_conf", 0.0))
        targets.append(
            _player_target(
                player_id=player_id,
                track_conf=track_conf,
                score=1.0,
                reasons=[UNIFORM_MESH_SELECTION_REASON],
            )
        )
    return targets


def _frame_in_rally_spans(frame: Mapping[str, Any], *, tracks_obj: Tracks) -> bool:
    if not tracks_obj.rally_spans:
        return True
    t = float(frame["t"])
    return any(float(span.t0) <= t <= float(span.t1) for span in tracks_obj.rally_spans)


def _select_priority_indexes(
    frames: list[Mapping[str, Any]],
    candidates: set[int],
    target_count: int,
) -> list[int]:
    if target_count <= 0:
        return []
    frame_lookup = {int(frame["frame_idx"]): frame for frame in frames}
    ordered = sorted(
        candidates,
        key=lambda frame_idx: (-float(frame_lookup.get(frame_idx, {}).get("score", 0.0)), frame_idx),
    )
    selected = sorted(ordered[:target_count])
    return selected


def _select_uniform_indexes(
    indexes: list[int],
    *,
    target_count: int,
    tracks_obj: Tracks,
) -> list[int]:
    if target_count <= 0 or not indexes:
        return []
    if target_count >= len(indexes):
        return sorted(indexes)
    spans = _rally_frame_spans(indexes, tracks_obj=tracks_obj)
    allocations = _allocate_counts_by_span(spans, target_count)
    selected: set[int] = set()
    for span_indexes, count in zip(spans, allocations):
        selected.update(_evenly_select_indexes(span_indexes, count))
    if len(selected) < target_count:
        remaining = [index for index in indexes if index not in selected]
        selected.update(_evenly_select_indexes(remaining, target_count - len(selected)))
    if len(selected) > target_count:
        selected = set(_evenly_select_indexes(sorted(selected), target_count))
    return sorted(selected)


def _rally_frame_spans(indexes: list[int], *, tracks_obj: Tracks) -> list[list[int]]:
    ordered = sorted(indexes)
    if not ordered:
        return []
    if not tracks_obj.rally_spans:
        return [ordered]
    spans: list[list[int]] = []
    for span in tracks_obj.rally_spans:
        span_indexes = [
            index
            for index in ordered
            if float(span.t0) <= index / tracks_obj.fps <= float(span.t1)
        ]
        if span_indexes:
            spans.append(span_indexes)
    return spans or [ordered]


def _allocate_counts_by_span(spans: list[list[int]], target_count: int) -> list[int]:
    if not spans:
        return []
    total = sum(len(span) for span in spans)
    if target_count >= total:
        return [len(span) for span in spans]
    raw_allocations = [(len(span) * target_count / total) for span in spans]
    allocations = [min(len(span), int(value)) for span, value in zip(spans, raw_allocations)]
    remaining = target_count - sum(allocations)
    remainders = sorted(
        range(len(spans)),
        key=lambda index: (-(raw_allocations[index] - int(raw_allocations[index])), -len(spans[index]), index),
    )
    for span_index in remainders:
        if remaining <= 0:
            break
        if allocations[span_index] < len(spans[span_index]):
            allocations[span_index] += 1
            remaining -= 1
    return allocations


def _evenly_select_indexes(indexes: list[int], target_count: int) -> list[int]:
    ordered = sorted(indexes)
    if target_count <= 0 or not ordered:
        return []
    if target_count >= len(ordered):
        return ordered
    if target_count == 1:
        return [ordered[len(ordered) // 2]]
    selected: list[int] = []
    last_index = len(ordered) - 1
    for slot in range(target_count):
        selected.append(ordered[round(slot * last_index / (target_count - 1))])
    return selected


def _uniform_stride(selected_indexes: set[int]) -> int | None:
    ordered = sorted(selected_indexes)
    if len(ordered) < 2:
        return None
    return int(round((ordered[-1] - ordered[0]) / (len(ordered) - 1)))


def _rally_span_count(tracks_obj: Tracks) -> int:
    return len(tracks_obj.rally_spans) if tracks_obj.rally_spans else 1


def _deep_mesh_windows(frames: list[Mapping[str, Any]], *, fps: float) -> list[dict[str, Any]]:
    windows: list[dict[str, Any]] = []
    current: list[Mapping[str, Any]] = []
    last_frame_idx: int | None = None

    for frame in sorted(frames, key=lambda item: int(item["frame_idx"])):
        frame_idx = int(frame["frame_idx"])
        if not _frame_selected_for_mesh_compute(frame):
            if current:
                windows.append(_deep_mesh_window(current, fps=fps))
                current = []
                last_frame_idx = None
            continue
        if current and last_frame_idx is not None and frame_idx != last_frame_idx + 1:
            windows.append(_deep_mesh_window(current, fps=fps))
            current = []
        current.append(frame)
        last_frame_idx = frame_idx

    if current:
        windows.append(_deep_mesh_window(current, fps=fps))
    return windows


def _deep_mesh_window(frames: list[Mapping[str, Any]], *, fps: float) -> dict[str, Any]:
    frame_start = int(frames[0]["frame_idx"])
    frame_end = int(frames[-1]["frame_idx"])
    target_player_ids = _deep_mesh_window_target_player_ids(frames)
    return {
        "frame_start": frame_start,
        "frame_end": frame_end,
        "t0": frame_start / fps,
        "t1": (frame_end + 1) / fps,
        "frame_count": len(frames),
        "target_representation": "world_mesh",
        "fallback_representation": "lane_a_skeleton",
        "target_player_ids": target_player_ids,
        "reason_counts": _reason_counts(frames),
        "max_score": max(float(frame["score"]) for frame in frames),
    }


def _frame_selected_for_mesh_compute(frame: Mapping[str, Any]) -> bool:
    if frame.get("recommended_tier") == "deep_mesh":
        return True
    tier_rationale = frame.get("tier_rationale")
    return isinstance(tier_rationale, Mapping) and bool(tier_rationale.get("mesh_selected"))


def _deep_mesh_window_target_player_ids(frames: list[Mapping[str, Any]]) -> list[int]:
    world_mesh_targets = {
        int(target["player_id"])
        for frame in frames
        for target in frame.get("player_targets", [])
        if target.get("target_representation") == "world_mesh"
    }
    if world_mesh_targets:
        return sorted(world_mesh_targets)
    return sorted(
        {
            int(player_id)
            for frame in frames
            if _frame_selected_for_mesh_compute(frame)
            for player_id in frame.get("active_player_ids", [])
        }
    )


def _reason_counts(frames: list[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for frame in frames:
        for reason in frame.get("reasons", []):
            reason_text = str(reason)
            counts[reason_text] = counts.get(reason_text, 0) + 1
    return dict(sorted(counts.items()))


def _summary(
    frames: list[Mapping[str, Any]],
    *,
    deep_mesh_windows: list[Mapping[str, Any]],
    mesh_coverage_policy: Mapping[str, Any],
) -> dict[str, Any]:
    by_tier: dict[str, int] = {}
    by_reason: dict[str, int] = {}
    by_player_target_representation: dict[str, int] = {}
    targeted_reviewed_contact_frame_count = 0
    coverage_incomplete_deep_mesh_frame_count = 0
    for frame in frames:
        tier = str(frame["recommended_tier"])
        by_tier[tier] = by_tier.get(tier, 0) + 1
        frame_reasons = [str(reason) for reason in frame.get("reasons", [])]
        if "reviewed_contact_targeted_body" in frame_reasons:
            targeted_reviewed_contact_frame_count += 1
        if tier == "deep_mesh" and "missing_expected_players" in frame_reasons:
            coverage_incomplete_deep_mesh_frame_count += 1
        for reason in frame.get("reasons", []):
            reason_text = str(reason)
            by_reason[reason_text] = by_reason.get(reason_text, 0) + 1
        for target in frame.get("player_targets", []):
            if not isinstance(target, Mapping):
                continue
            representation = str(target.get("target_representation", "unknown"))
            by_player_target_representation[representation] = by_player_target_representation.get(representation, 0) + 1
    return {
        "by_tier": dict(sorted(by_tier.items())),
        "by_reason": dict(sorted(by_reason.items())),
        "by_player_target_representation": dict(sorted(by_player_target_representation.items())),
        "max_score": max((float(frame["score"]) for frame in frames), default=0.0),
        "deep_mesh_window_count": len(deep_mesh_windows),
        "deep_mesh_frame_count": sum(int(window["frame_count"]) for window in deep_mesh_windows),
        "world_mesh_frame_count": sum(int(window["frame_count"]) for window in deep_mesh_windows),
        "human_review_frame_count": by_tier.get("human_review", 0),
        "targeted_reviewed_contact_frame_count": targeted_reviewed_contact_frame_count,
        "coverage_incomplete_deep_mesh_frame_count": coverage_incomplete_deep_mesh_frame_count,
        "mesh_coverage_mode": str(mesh_coverage_policy.get("mode")),
        "mesh_coverage_fraction": round(
            float(mesh_coverage_policy.get("selected_mesh_frame_count", 0)) / len(frames),
            6,
        )
        if frames
        else 0.0,
    }


def _tracks(value: Tracks | Mapping[str, Any]) -> Tracks:
    if isinstance(value, Tracks):
        return value
    try:
        return Tracks.model_validate(value)
    except ValidationError as exc:
        raise ValueError(f"tracks failed validation: {exc}") from exc


def _ball_track(value: BallTrack | Mapping[str, Any] | None) -> BallTrack | None:
    if value is None or isinstance(value, BallTrack):
        return value
    try:
        return BallTrack.model_validate(value)
    except ValidationError as exc:
        raise ValueError(f"ball_track failed validation: {exc}") from exc


def _contact_windows(value: ContactWindows | Mapping[str, Any] | None) -> ContactWindows | None:
    if value is None or isinstance(value, ContactWindows):
        return value
    try:
        return ContactWindows.model_validate(value)
    except ValidationError as exc:
        raise ValueError(f"contact_windows failed validation: {exc}") from exc


__all__ = [
    "DEFAULT_MESH_COVERAGE_MODE",
    "DEFAULT_TARGET_MESH_FRAME_BUDGET",
    "DEFAULT_BALL_PROXIMITY_M",
    "DEFAULT_HIGH_CONFIDENCE_SWING_FLOOR",
    "DEFAULT_BALL_AWARE_PADDING_S",
    "MESH_COVERAGE_MODES",
    "BALL_AWARE_TRIGGER_REASONS",
    "BALL_AWARE_CONTACT_REASON",
    "BALL_PROXIMITY_REASON",
    "HIGH_CONFIDENCE_SWING_REASON",
    "build_frame_compute_plan",
    "build_frame_compute_plan_from_files",
    "write_frame_compute_plan",
]

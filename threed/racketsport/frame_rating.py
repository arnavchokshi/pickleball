"""Adaptive frame compute-rating primitives.

This module ranks frames for heavier downstream compute. It intentionally does
not schedule GPU work; it emits an inspectable plan that later runners can use.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from pydantic import ValidationError

from .schemas import BallTrack, ContactWindows, Tracks, validate_artifact_file


SCHEMA_VERSION = 1
ARTIFACT_TYPE = "racketsport_frame_compute_plan"


def build_frame_compute_plan(
    tracks: Tracks | Mapping[str, Any],
    *,
    ball_track: BallTrack | Mapping[str, Any] | None = None,
    contact_windows: ContactWindows | Mapping[str, Any] | None = None,
    expected_players: int = 4,
    low_track_confidence: float = 0.5,
    low_ball_confidence: float = 0.4,
    contact_padding_s: float = 0.08,
) -> dict[str, Any]:
    """Return a CPU-only frame priority plan for adaptive deep compute."""

    if expected_players <= 0:
        raise ValueError("expected_players must be positive")
    if not 0.0 <= low_track_confidence <= 1.0:
        raise ValueError("low_track_confidence must be between 0 and 1")
    if not 0.0 <= low_ball_confidence <= 1.0:
        raise ValueError("low_ball_confidence must be between 0 and 1")
    if contact_padding_s < 0.0:
        raise ValueError("contact_padding_s must be non-negative")

    tracks_obj = _tracks(tracks)
    if tracks_obj.fps <= 0.0:
        raise ValueError("tracks.fps must be positive")
    ball_obj = _ball_track(ball_track)
    contacts_obj = _contact_windows(contact_windows)

    track_frames = _track_frames_by_index(tracks_obj)
    ball_frames = _ball_frames_by_index(ball_obj, fps=tracks_obj.fps)
    contact_spans = _contact_spans(contacts_obj, padding_s=contact_padding_s)
    frame_indexes = sorted(set(track_frames) | set(ball_frames) | set(_frame_index(span["t"], tracks_obj.fps) for span in contact_spans))

    frames: list[dict[str, Any]] = []
    for frame_idx in frame_indexes:
        t = frame_idx / tracks_obj.fps
        active_tracks = track_frames.get(frame_idx, [])
        reasons: list[str] = []
        score = 0.0

        frame_contact_spans = _contact_spans_for_time(t, contact_spans)
        active_player_ids = {player_id for player_id, _frame in active_tracks}
        reviewed_target_player_ids = _reviewed_target_player_ids(frame_contact_spans, active_player_ids)

        if frame_contact_spans:
            reasons.append("contact_window")
            score += 0.55

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
        player_targets = _player_targets(
            active_tracks,
            frame_contact_spans=frame_contact_spans,
            missing_players=missing_players,
            low_track_confidence=low_track_confidence,
            ball_reason=ball_reason,
            reviewed_target_player_ids=reviewed_target_player_ids,
        )
        frames.append(
            {
                "frame_idx": frame_idx,
                "t": t,
                "score": round(min(score, 1.0), 3),
                "recommended_tier": recommended_tier,
                "target_representation": _target_representation(recommended_tier),
                "reasons": reasons,
                "active_players": len(active_tracks),
                "active_player_ids": sorted(player_id for player_id, _frame in active_tracks),
                "missing_players": missing_players,
                "min_track_conf": min_track_conf,
                "ball_conf": ball_conf,
                "player_targets": player_targets,
            }
        )

    deep_mesh_windows = _deep_mesh_windows(frames, fps=tracks_obj.fps)
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "fps": tracks_obj.fps,
        "expected_players": expected_players,
        "frame_count": len(frames),
        "frames": frames,
        "deep_mesh_windows": deep_mesh_windows,
        "summary": _summary(frames, deep_mesh_windows=deep_mesh_windows),
    }


def build_frame_compute_plan_from_files(
    *,
    tracks_path: str | Path,
    ball_track_path: str | Path | None = None,
    contact_windows_path: str | Path | None = None,
    expected_players: int = 4,
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

    return build_frame_compute_plan(
        tracks,
        ball_track=ball_track,
        contact_windows=contact_windows,
        expected_players=expected_players,
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

    targets: list[dict[str, Any]] = []
    for player_id, frame in sorted(active_tracks):
        reasons: list[str] = []
        score = 0.0
        track_conf = float(frame.conf)

        if frame_contact_spans and (unassigned_contact or player_id in contact_player_ids):
            reasons.append("contact_window")
            score += 0.55

        if track_conf < low_track_confidence:
            reasons.append("low_track_confidence")
            score += 0.25

        if ball_reason is not None and (reasons or not frame_contact_spans):
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


def _deep_mesh_windows(frames: list[Mapping[str, Any]], *, fps: float) -> list[dict[str, Any]]:
    windows: list[dict[str, Any]] = []
    current: list[Mapping[str, Any]] = []
    last_frame_idx: int | None = None

    for frame in sorted(frames, key=lambda item: int(item["frame_idx"])):
        frame_idx = int(frame["frame_idx"])
        if frame.get("recommended_tier") != "deep_mesh":
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
    target_player_ids = sorted(
        {
            int(target["player_id"])
            for frame in frames
            for target in frame.get("player_targets", [])
            if target.get("target_representation") == "world_mesh"
        }
    )
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


def _reason_counts(frames: list[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for frame in frames:
        for reason in frame.get("reasons", []):
            reason_text = str(reason)
            counts[reason_text] = counts.get(reason_text, 0) + 1
    return dict(sorted(counts.items()))


def _summary(frames: list[Mapping[str, Any]], *, deep_mesh_windows: list[Mapping[str, Any]]) -> dict[str, Any]:
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
        "human_review_frame_count": by_tier.get("human_review", 0),
        "targeted_reviewed_contact_frame_count": targeted_reviewed_contact_frame_count,
        "coverage_incomplete_deep_mesh_frame_count": coverage_incomplete_deep_mesh_frame_count,
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
    "build_frame_compute_plan",
    "build_frame_compute_plan_from_files",
    "write_frame_compute_plan",
]

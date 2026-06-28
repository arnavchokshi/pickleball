"""CPU-only ball trajectory inflection cue generation.

These cues are review/support signals for contact-window fusion. They are not
BALL accuracy gates and are not trusted without audio and wrist-velocity cues.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


ARTIFACT_TYPE = "racketsport_ball_inflections"
DEFAULT_MIN_TURN_DEGREES = 45.0
DEFAULT_MIN_SPEED_MPS = 0.75
DEFAULT_MAX_NEIGHBOR_GAP_S = 0.2
DEFAULT_MIN_CANDIDATE_SEPARATION_S = 0.15


def build_ball_inflections_from_virtual_world(
    virtual_world: Mapping[str, Any],
    *,
    min_turn_degrees: float = DEFAULT_MIN_TURN_DEGREES,
    min_speed_mps: float = DEFAULT_MIN_SPEED_MPS,
    max_neighbor_gap_s: float = DEFAULT_MAX_NEIGHBOR_GAP_S,
    min_candidate_separation_s: float = DEFAULT_MIN_CANDIDATE_SEPARATION_S,
) -> dict[str, Any]:
    """Build a review-only ball-inflection cue artifact from court-frame ball path."""

    min_turn_degrees = _require_finite(min_turn_degrees, "min_turn_degrees")
    min_speed_mps = _require_finite(min_speed_mps, "min_speed_mps")
    max_neighbor_gap_s = _require_finite(max_neighbor_gap_s, "max_neighbor_gap_s")
    min_candidate_separation_s = _require_finite(min_candidate_separation_s, "min_candidate_separation_s")
    if min_turn_degrees < 0.0:
        raise ValueError("min_turn_degrees must be non-negative")
    if min_speed_mps < 0.0:
        raise ValueError("min_speed_mps must be non-negative")
    if max_neighbor_gap_s <= 0.0:
        raise ValueError("max_neighbor_gap_s must be positive")
    if min_candidate_separation_s < 0.0:
        raise ValueError("min_candidate_separation_s must be non-negative")

    frames = _usable_ball_frames(virtual_world)
    raw_candidates: list[dict[str, Any]] = []
    for previous, current, following in zip(frames, frames[1:], frames[2:]):
        dt_before = current["time_s"] - previous["time_s"]
        dt_after = following["time_s"] - current["time_s"]
        if dt_before <= 0.0 or dt_after <= 0.0:
            continue
        if dt_before > max_neighbor_gap_s or dt_after > max_neighbor_gap_s:
            continue

        before = _vector_delta(previous["world_xyz"], current["world_xyz"])
        after = _vector_delta(current["world_xyz"], following["world_xyz"])
        speed_before = _norm(before) / dt_before
        speed_after = _norm(after) / dt_after
        if min(speed_before, speed_after) < min_speed_mps:
            continue

        turn_degrees = _turn_angle_degrees(before, after)
        if turn_degrees < min_turn_degrees:
            continue

        confidence = _candidate_confidence(current, previous, following, turn_degrees)
        raw_candidates.append(
            {
                "time_s": current["time_s"],
                "frame": current.get("frame"),
                "ball_world_xyz": [round(float(value), 9) for value in current["world_xyz"]],
                "confidence": confidence,
                "turn_angle_deg": round(turn_degrees, 6),
                "speed_before_mps": round(speed_before, 6),
                "speed_after_mps": round(speed_after, 6),
                "approx": bool(current.get("approx", False) or previous.get("approx", False) or following.get("approx", False)),
                "source": "virtual_world_court_plane_ball_path",
            }
        )
    candidates = _suppress_nearby_candidates(raw_candidates, min_separation_s=min_candidate_separation_s)

    return {
        "schema_version": 1,
        "artifact_type": ARTIFACT_TYPE,
        "source": "virtual_world_court_plane_ball_path",
        "world_frame": str(virtual_world.get("world_frame", "court_Z0")),
        "not_gate_verified": True,
        "requires_additional_cues": ["audio_onsets", "wrist_velocity_peaks"],
        "summary": {
            "usable_frame_count": len(frames),
            "candidate_count": len(candidates),
            "raw_candidate_count": len(raw_candidates),
            "min_turn_degrees": min_turn_degrees,
            "min_speed_mps": min_speed_mps,
            "max_neighbor_gap_s": max_neighbor_gap_s,
            "min_candidate_separation_s": min_candidate_separation_s,
        },
        "candidates": candidates,
    }


def build_ball_inflections_from_file(
    virtual_world_path: str | Path,
    *,
    min_turn_degrees: float = DEFAULT_MIN_TURN_DEGREES,
    min_speed_mps: float = DEFAULT_MIN_SPEED_MPS,
    max_neighbor_gap_s: float = DEFAULT_MAX_NEIGHBOR_GAP_S,
    min_candidate_separation_s: float = DEFAULT_MIN_CANDIDATE_SEPARATION_S,
) -> dict[str, Any]:
    payload = _read_json(Path(virtual_world_path))
    if not isinstance(payload, Mapping):
        raise ValueError("virtual_world.json must contain an object")
    return build_ball_inflections_from_virtual_world(
        payload,
        min_turn_degrees=min_turn_degrees,
        min_speed_mps=min_speed_mps,
        max_neighbor_gap_s=max_neighbor_gap_s,
        min_candidate_separation_s=min_candidate_separation_s,
    )


def write_ball_inflections(path: str | Path, payload: Mapping[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _usable_ball_frames(virtual_world: Mapping[str, Any]) -> list[dict[str, Any]]:
    ball = virtual_world.get("ball")
    raw_frames = ball.get("frames") if isinstance(ball, Mapping) else None
    if not isinstance(raw_frames, list):
        raise ValueError("virtual_world ball.frames must be a list")

    frames: list[dict[str, Any]] = []
    for index, frame in enumerate(raw_frames):
        if not isinstance(frame, Mapping):
            continue
        if frame.get("visible") is not True:
            continue
        world_xyz = frame.get("world_xyz")
        if not isinstance(world_xyz, Sequence) or isinstance(world_xyz, (str, bytes)) or len(world_xyz) != 3:
            continue
        try:
            point = tuple(_require_finite(component, "world_xyz") for component in world_xyz)
            time_s = _require_finite(frame.get("t", frame.get("time_s")), "t")
            confidence = _confidence(frame.get("conf", frame.get("confidence", 0.0)))
        except ValueError:
            continue
        if time_s < 0.0:
            continue
        frames.append(
            {
                "time_s": time_s,
                "frame": frame.get("frame", index),
                "world_xyz": point,
                "confidence": confidence,
                "approx": bool(frame.get("approx", False)),
            }
        )
    return sorted(frames, key=lambda item: (item["time_s"], item["frame"]))


def _candidate_confidence(
    current: Mapping[str, Any],
    previous: Mapping[str, Any],
    following: Mapping[str, Any],
    turn_degrees: float,
) -> float:
    min_confidence = min(float(previous["confidence"]), float(current["confidence"]), float(following["confidence"]))
    turn_score = min(1.0, turn_degrees / 90.0)
    return round(max(0.0, min(1.0, min_confidence * turn_score)), 6)


def _suppress_nearby_candidates(candidates: list[dict[str, Any]], *, min_separation_s: float) -> list[dict[str, Any]]:
    if min_separation_s == 0.0:
        return sorted(candidates, key=lambda item: item["time_s"])
    kept: list[dict[str, Any]] = []
    for candidate in sorted(candidates, key=lambda item: item["time_s"]):
        if not kept or float(candidate["time_s"]) - float(kept[-1]["time_s"]) >= min_separation_s:
            kept.append(candidate)
            continue
        if _candidate_rank(candidate) > _candidate_rank(kept[-1]):
            kept[-1] = candidate
    return kept


def _candidate_rank(candidate: Mapping[str, Any]) -> tuple[float, float, float]:
    return (
        float(candidate.get("confidence", 0.0)),
        float(candidate.get("turn_angle_deg", 0.0)),
        min(float(candidate.get("speed_before_mps", 0.0)), float(candidate.get("speed_after_mps", 0.0))),
    )


def _vector_delta(left: Sequence[float], right: Sequence[float]) -> tuple[float, float, float]:
    return tuple(float(b) - float(a) for a, b in zip(left, right))


def _norm(vector: Sequence[float]) -> float:
    return math.sqrt(sum(float(component) ** 2 for component in vector))


def _turn_angle_degrees(before: Sequence[float], after: Sequence[float]) -> float:
    before_norm = _norm(before)
    after_norm = _norm(after)
    if before_norm == 0.0 or after_norm == 0.0:
        return 0.0
    cosine = sum(float(a) * float(b) for a, b in zip(before, after)) / (before_norm * after_norm)
    cosine = max(-1.0, min(1.0, cosine))
    return math.degrees(math.acos(cosine))


def _confidence(value: Any) -> float:
    value = _require_finite(value, "confidence")
    if not 0.0 <= value <= 1.0:
        raise ValueError("confidence must be in [0, 1]")
    return value


def _require_finite(value: Any, name: str) -> float:
    if value is None:
        raise ValueError(f"{name} is required")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON: {exc}") from exc


__all__ = [
    "ARTIFACT_TYPE",
    "build_ball_inflections_from_file",
    "build_ball_inflections_from_virtual_world",
    "write_ball_inflections",
]

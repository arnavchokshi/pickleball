"""Soft player-proximity confidence prior for ball candidates.

This module is intentionally a postprocess over an existing ``ball_track``
artifact. It never deletes frames, never flips ``visible`` flags, and never
thresholds candidates. It only multiplies each visible candidate's confidence
by a smooth factor derived from distance to the nearest player box. Downstream
thresholding remains the same separate step used by existing BALL benchmarks.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .ball_overlay import load_ball_track
from .schemas import BallTrack, Tracks, validate_artifact_file

ARTIFACT_TYPE = "racketsport_ball_player_proximity_prior"
SCHEMA_VERSION = 1
DEFAULT_STRENGTH = 0.35
DEFAULT_INFLUENCE_DIAG_FRACTION = 1.5


@dataclass(frozen=True)
class BallPlayerProximityPriorConfig:
    """Parameters for the soft confidence prior.

    ``strength`` is the maximum fractional confidence reduction at/on a player
    box. A value of 0.35 means the strongest possible factor is 0.65. Values
    must stay below 1.0 so this prior never turns confidence into a hard zero.

    ``influence_diag_fraction`` controls how far from a player box the prior
    tapers to no effect, in units of the nearest player box diagonal.
    """

    strength: float = DEFAULT_STRENGTH
    influence_diag_fraction: float = DEFAULT_INFLUENCE_DIAG_FRACTION

    def __post_init__(self) -> None:
        strength = float(self.strength)
        influence = float(self.influence_diag_fraction)
        if not math.isfinite(strength) or not 0.0 <= strength < 1.0:
            raise ValueError("strength must be a finite value in [0, 1)")
        if not math.isfinite(influence) or influence <= 0.0:
            raise ValueError("influence_diag_fraction must be a finite positive value")

    def as_dict(self) -> dict[str, float]:
        return {
            "strength": float(self.strength),
            "influence_diag_fraction": float(self.influence_diag_fraction),
            "min_factor": 1.0 - float(self.strength),
        }


def apply_ball_player_proximity_prior(
    *,
    ball_track_path: str | Path,
    tracks_path: str | Path,
    config: BallPlayerProximityPriorConfig | None = None,
) -> dict[str, Any]:
    """Return a new BallTrack payload plus a provenance report.

    The returned ``ball_track`` is schema-valid. It has the same frame count,
    timestamps, coordinates, source, bounces, and ``visible`` flags as the
    input; only frame ``conf`` values may change.
    """

    cfg = config or BallPlayerProximityPriorConfig()
    source_ball_track = Path(ball_track_path)
    source_tracks = Path(tracks_path)
    track = load_ball_track(source_ball_track)
    tracks = validate_artifact_file("tracks", source_tracks)
    if not isinstance(tracks, Tracks):
        raise ValueError(f"{source_tracks} did not parse as Tracks")

    original_payload = track.model_dump(mode="json")
    output_payload = track.model_dump(mode="json")
    player_boxes_by_frame = _player_boxes_by_frame(tracks)
    frame_rows: list[dict[str, Any]] = []
    adjusted_frame_count = 0
    factors: list[float] = []

    for frame_payload, frame in zip(output_payload["frames"], track.frames):
        ball_frame_index = int(round(float(frame.t) * float(track.fps)))
        tracks_frame_index = int(round(float(frame.t) * float(tracks.fps)))
        player_boxes = player_boxes_by_frame.get(tracks_frame_index, [])
        if frame.visible:
            proximity = _nearest_player_proximity(tuple(float(value) for value in frame.xy), player_boxes)
            factor = _confidence_factor(
                proximity.distance_in_box_diagonals,
                strength=float(cfg.strength),
                influence_diag_fraction=float(cfg.influence_diag_fraction),
            )
        else:
            proximity = _nearest_player_proximity(tuple(float(value) for value in frame.xy), player_boxes)
            factor = 1.0
        old_conf = float(frame.conf)
        new_conf = _clamp_unit(old_conf * factor)
        frame_payload["conf"] = new_conf
        if factor < 1.0 - 1e-12:
            adjusted_frame_count += 1
        factors.append(factor)
        frame_rows.append(
            {
                "frame": ball_frame_index,
                "tracks_frame": tracks_frame_index,
                "t": float(frame.t),
                "visible": bool(frame.visible),
                "xy": [float(frame.xy[0]), float(frame.xy[1])],
                "original_conf": old_conf,
                "adjusted_conf": new_conf,
                "factor": factor,
                "distance_to_nearest_player_box_px": proximity.distance_px,
                "nearest_player_box_diag_px": proximity.box_diag_px,
                "distance_in_box_diagonals": proximity.distance_in_box_diagonals,
                "player_box_count": len(player_boxes),
            }
        )

    BallTrack.model_validate(output_payload)
    report = _report(
        source_ball_track=source_ball_track,
        source_tracks=source_tracks,
        config=cfg,
        original_payload=original_payload,
        output_payload=output_payload,
        frame_rows=frame_rows,
        adjusted_frame_count=adjusted_frame_count,
        factors=factors,
    )
    return {"ball_track": output_payload, "report": report}


def apply_ball_player_proximity_prior_from_files(
    *,
    ball_track_path: str | Path,
    tracks_path: str | Path,
    out_ball_track_path: str | Path,
    out_report_path: str | Path,
    config: BallPlayerProximityPriorConfig | None = None,
) -> dict[str, Any]:
    """Apply the prior and write ``ball_track.json`` plus a provenance report."""

    result = apply_ball_player_proximity_prior(ball_track_path=ball_track_path, tracks_path=tracks_path, config=config)
    out_ball_track = Path(out_ball_track_path)
    out_report = Path(out_report_path)
    _write_json(out_ball_track, result["ball_track"])
    _write_json(out_report, result["report"])
    return result["report"]


@dataclass(frozen=True)
class _Proximity:
    distance_px: float | None
    box_diag_px: float | None
    distance_in_box_diagonals: float | None


def _report(
    *,
    source_ball_track: Path,
    source_tracks: Path,
    config: BallPlayerProximityPriorConfig,
    original_payload: Mapping[str, Any],
    output_payload: Mapping[str, Any],
    frame_rows: list[dict[str, Any]],
    adjusted_frame_count: int,
    factors: list[float],
) -> dict[str, Any]:
    additive_safe = _additive_safety_summary(original_payload, output_payload)
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "status": "applied",
        "source_ball_track_path": str(source_ball_track),
        "source_tracks_path": str(source_tracks),
        "config": config.as_dict(),
        "frame_count": len(frame_rows),
        "adjusted_frame_count": adjusted_frame_count,
        "min_factor": min(factors) if factors else None,
        "max_factor": max(factors) if factors else None,
        "mean_factor": (sum(factors) / len(factors)) if factors else None,
        "additive_safe": additive_safe,
        "downstream_thresholding_required": True,
        "not_ground_truth": True,
        "promotion_claimed": False,
        "notes": [
            "Soft confidence prior only: no frames are deleted and visible flags are unchanged.",
            "Run the existing downstream confidence thresholding/benchmark step after this artifact to measure any effect.",
        ],
        "frames": frame_rows,
    }


def _additive_safety_summary(original_payload: Mapping[str, Any], output_payload: Mapping[str, Any]) -> dict[str, bool]:
    original_frames = list(original_payload.get("frames", []))
    output_frames = list(output_payload.get("frames", []))
    frame_count_preserved = len(original_frames) == len(output_frames)
    visible_flags_preserved = frame_count_preserved and all(
        bool(before.get("visible")) == bool(after.get("visible")) for before, after in zip(original_frames, output_frames)
    )
    only_confidence_changed = frame_count_preserved
    if only_confidence_changed:
        for before, after in zip(original_frames, output_frames):
            before_without_conf = {key: value for key, value in before.items() if key != "conf"}
            after_without_conf = {key: value for key, value in after.items() if key != "conf"}
            if before_without_conf != after_without_conf:
                only_confidence_changed = False
                break
    return {
        "frame_count_preserved": frame_count_preserved,
        "visible_flags_preserved": visible_flags_preserved,
        "only_confidence_changed": only_confidence_changed,
    }


def _player_boxes_by_frame(tracks: Tracks) -> dict[int, list[tuple[float, float, float, float]]]:
    by_frame: dict[int, list[tuple[float, float, float, float]]] = {}
    for player in tracks.players:
        for frame in player.frames:
            frame_index = int(round(float(frame.t) * float(tracks.fps)))
            by_frame.setdefault(frame_index, []).append(tuple(float(value) for value in frame.bbox))
    return by_frame


def _nearest_player_proximity(
    point: tuple[float, float], player_boxes: list[tuple[float, float, float, float]]
) -> _Proximity:
    if not player_boxes:
        return _Proximity(distance_px=None, box_diag_px=None, distance_in_box_diagonals=None)
    best_distance = math.inf
    best_diag = 0.0
    for box in player_boxes:
        diag = _box_diagonal(box)
        if diag <= 0.0:
            continue
        distance = _point_to_box_distance(point, box)
        if distance < best_distance:
            best_distance = distance
            best_diag = diag
    if not math.isfinite(best_distance) or best_diag <= 0.0:
        return _Proximity(distance_px=None, box_diag_px=None, distance_in_box_diagonals=None)
    return _Proximity(
        distance_px=best_distance,
        box_diag_px=best_diag,
        distance_in_box_diagonals=best_distance / best_diag,
    )


def _confidence_factor(
    distance_in_box_diagonals: float | None,
    *,
    strength: float,
    influence_diag_fraction: float,
) -> float:
    if distance_in_box_diagonals is None:
        return 1.0
    if distance_in_box_diagonals >= influence_diag_fraction:
        return 1.0
    x = _clamp_unit(distance_in_box_diagonals / influence_diag_fraction)
    proximity_weight = 1.0 - _smoothstep(x)
    return _clamp_unit(1.0 - strength * proximity_weight)


def _smoothstep(x: float) -> float:
    value = _clamp_unit(x)
    return value * value * (3.0 - 2.0 * value)


def _point_to_box_distance(point: tuple[float, float], box: tuple[float, float, float, float]) -> float:
    x, y = point
    x1, y1, x2, y2 = box
    dx = max(x1 - x, 0.0, x - x2)
    dy = max(y1 - y, 0.0, y - y2)
    return math.hypot(dx, dy)


def _box_diagonal(box: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = box
    return math.hypot(max(0.0, x2 - x1), max(0.0, y2 - y1))


def _clamp_unit(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


__all__ = [
    "ARTIFACT_TYPE",
    "BallPlayerProximityPriorConfig",
    "apply_ball_player_proximity_prior",
    "apply_ball_player_proximity_prior_from_files",
]

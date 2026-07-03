"""Bucket ball-detector false positives by distance to the nearest player box.

Tests the owner's explicit hypothesis ("ball FPs cluster on paddles/limbs")
against existing benchmark artifacts. This is a pure post-hoc diagnostic: it
reuses an already-scored ball track (predictions) plus already-reviewed CVAT
ball labels plus an already-computed player ``tracks.json`` -- it does not run
any model, does not write back into any promoted artifact, and never treats
its own output as a gate pass. See ``ball_detector_error_mining.py`` for the
sibling per-clip FP/TP/miss mining this borrows its "hidden false positive"
definition from (an unmatched, model-predicted-visible ball sample on a frame
with no reviewed ball label).

Unsafe-flow note (see the synergy audit): this diagnostic exists to decide
*whether* a player-proximity signal is worth building into the ball pipeline,
not to build it. If the distribution shows FPs cluster near player boxes, the
audit's own guidance applies: any resulting prior must be a *soft* confidence
multiplier, never a hard veto -- the current best TRK candidate is itself
``do_not_promote`` (spectator/off-court FPs, ID switches on some clips), and a
hard veto built on top of still-imperfect track boxes risks suppressing a
correct ball detection the same way the rejected WASB ``stable_veto`` fusion
candidate did (recall collapsed to 0.445 vs ~0.51-0.56).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .ball_overlay import load_ball_track
from .schemas import CvatVideoAnnotations, Tracks, validate_artifact_file

ARTIFACT_TYPE = "racketsport_ball_fp_player_proximity"
SCHEMA_VERSION = 1

ON_PLAYER = "on_player"
NEAR_PLAYER = "near_player"
FAR_FIELD = "far_field"
NO_PLAYERS_IN_FRAME = "no_players_in_frame"

DEFAULT_ON_PLAYER_DIAG_FRACTION = 0.25
DEFAULT_NEAR_PLAYER_DIAG_FRACTION = 1.5


@dataclass(frozen=True)
class ClipProximityInput:
    clip: str
    ball_track_path: Path
    reviewed_boxes_path: Path
    tracks_path: Path


def bucket_ball_fps_by_player_proximity(
    *,
    clips: Sequence[ClipProximityInput],
    on_player_diag_fraction: float = DEFAULT_ON_PLAYER_DIAG_FRACTION,
    near_player_diag_fraction: float = DEFAULT_NEAR_PLAYER_DIAG_FRACTION,
) -> dict[str, Any]:
    """Bucket every hidden ball-detector FP across ``clips`` by distance to the
    nearest player box, expressed in units of that player's box diagonal.

    Source-only / diagnostic-only: reads already-computed ball tracks,
    already-reviewed CVAT ball labels, and an already-computed player
    ``tracks.json`` per clip; writes no artifact any downstream stage
    consumes and makes no promotion claim.
    """
    if on_player_diag_fraction <= 0.0:
        raise ValueError("on_player_diag_fraction must be positive")
    if near_player_diag_fraction <= on_player_diag_fraction:
        raise ValueError("near_player_diag_fraction must be greater than on_player_diag_fraction")

    per_clip: dict[str, Any] = {}
    all_rows: list[dict[str, Any]] = []
    for clip_input in clips:
        rows = _clip_fp_proximity_rows(
            clip_input,
            on_player_diag_fraction=on_player_diag_fraction,
            near_player_diag_fraction=near_player_diag_fraction,
        )
        per_clip[clip_input.clip] = {
            "ball_track_path": str(clip_input.ball_track_path),
            "reviewed_boxes_path": str(clip_input.reviewed_boxes_path),
            "tracks_path": str(clip_input.tracks_path),
            "false_positive_count": len(rows),
            "bucket_counts": _bucket_counts(rows),
            "rows": rows,
        }
        all_rows.extend(rows)

    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "not_ground_truth": True,
        "source_only": True,
        "promotion_claimed": False,
        "purpose": (
            "Test whether ball-detector false positives cluster near player boxes "
            "(paddles/limbs), the pre-condition for building a soft player-proximity "
            "confidence prior. Diagnostic only; never a gate pass."
        ),
        "on_player_diag_fraction": on_player_diag_fraction,
        "near_player_diag_fraction": near_player_diag_fraction,
        "bucket_definitions": {
            ON_PLAYER: f"distance to nearest player box <= {on_player_diag_fraction} x that box's diagonal (inside or immediately adjacent)",
            NEAR_PLAYER: f"{on_player_diag_fraction} x diag < distance <= {near_player_diag_fraction} x diag",
            FAR_FIELD: f"distance > {near_player_diag_fraction} x diag from every player box",
            NO_PLAYERS_IN_FRAME: "no player track box exists on this frame to measure distance against",
        },
        "clips": per_clip,
        "combined": {
            "false_positive_count": len(all_rows),
            "bucket_counts": _bucket_counts(all_rows),
            "bucket_fractions": _bucket_fractions(all_rows),
        },
    }


def bucket_ball_fps_by_player_proximity_from_files(
    *,
    clips: Sequence[Mapping[str, str]],
    on_player_diag_fraction: float = DEFAULT_ON_PLAYER_DIAG_FRACTION,
    near_player_diag_fraction: float = DEFAULT_NEAR_PLAYER_DIAG_FRACTION,
) -> dict[str, Any]:
    """File-path convenience wrapper. Each ``clips`` entry needs ``clip``,
    ``ball_track``, ``reviewed_boxes``, and ``tracks`` string paths."""

    parsed = [
        ClipProximityInput(
            clip=str(entry["clip"]),
            ball_track_path=Path(entry["ball_track"]),
            reviewed_boxes_path=Path(entry["reviewed_boxes"]),
            tracks_path=Path(entry["tracks"]),
        )
        for entry in clips
    ]
    return bucket_ball_fps_by_player_proximity(
        clips=parsed,
        on_player_diag_fraction=on_player_diag_fraction,
        near_player_diag_fraction=near_player_diag_fraction,
    )


def write_ball_fp_player_proximity(path: str | Path, report: Mapping[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _clip_fp_proximity_rows(
    clip_input: ClipProximityInput,
    *,
    on_player_diag_fraction: float,
    near_player_diag_fraction: float,
) -> list[dict[str, Any]]:
    ball_track = load_ball_track(clip_input.ball_track_path)
    labels = _load_cvat_labels(clip_input.reviewed_boxes_path)
    tracks = validate_artifact_file("tracks", clip_input.tracks_path)
    if not isinstance(tracks, Tracks):
        raise ValueError(f"{clip_input.tracks_path} did not parse as Tracks")

    labeled_ball_frames = _labeled_ball_frame_indexes(labels)
    label_frame_count = _label_frame_count(labels)
    player_boxes_by_frame = _player_boxes_by_frame(tracks)

    rows: list[dict[str, Any]] = []
    for frame in ball_track.frames:
        if not frame.visible:
            continue
        frame_idx = int(round(float(frame.t) * ball_track.fps))
        if frame_idx >= label_frame_count:
            # Outside the reviewed label horizon: cannot classify as a hidden
            # FP one way or the other, so it is excluded (not counted as a FP).
            continue
        if frame_idx in labeled_ball_frames:
            continue  # a reviewed ball label exists on this frame; not a hidden FP
        row = _fp_row(
            clip=clip_input.clip,
            frame_idx=frame_idx,
            xy=(float(frame.xy[0]), float(frame.xy[1])),
            conf=float(frame.conf),
            player_boxes=player_boxes_by_frame.get(frame_idx, []),
            on_player_diag_fraction=on_player_diag_fraction,
            near_player_diag_fraction=near_player_diag_fraction,
        )
        rows.append(row)
    return rows


def _fp_row(
    *,
    clip: str,
    frame_idx: int,
    xy: tuple[float, float],
    conf: float,
    player_boxes: list[tuple[float, float, float, float]],
    on_player_diag_fraction: float,
    near_player_diag_fraction: float,
) -> dict[str, Any]:
    if not player_boxes:
        return {
            "clip": clip,
            "frame": frame_idx,
            "xy": [round(xy[0], 3), round(xy[1], 3)],
            "conf": round(conf, 3),
            "distance_to_nearest_player_box_px": None,
            "nearest_player_box_diag_px": None,
            "distance_in_box_diagonals": None,
            "bucket": NO_PLAYERS_IN_FRAME,
        }
    best_distance = math.inf
    best_diag = 0.0
    for box in player_boxes:
        distance = _point_to_box_distance(xy, box)
        diag = _box_diagonal(box)
        if distance < best_distance:
            best_distance = distance
            best_diag = diag
    diag_units = best_distance / best_diag if best_diag > 0.0 else math.inf
    bucket = _bucket_for_diag_units(
        diag_units,
        on_player_diag_fraction=on_player_diag_fraction,
        near_player_diag_fraction=near_player_diag_fraction,
    )
    return {
        "clip": clip,
        "frame": frame_idx,
        "xy": [round(xy[0], 3), round(xy[1], 3)],
        "conf": round(conf, 3),
        "distance_to_nearest_player_box_px": round(best_distance, 3),
        "nearest_player_box_diag_px": round(best_diag, 3),
        "distance_in_box_diagonals": round(diag_units, 3) if math.isfinite(diag_units) else None,
        "bucket": bucket,
    }


def _bucket_for_diag_units(diag_units: float, *, on_player_diag_fraction: float, near_player_diag_fraction: float) -> str:
    if diag_units <= on_player_diag_fraction:
        return ON_PLAYER
    if diag_units <= near_player_diag_fraction:
        return NEAR_PLAYER
    return FAR_FIELD


def _point_to_box_distance(point: tuple[float, float], box: tuple[float, float, float, float]) -> float:
    x, y = point
    x1, y1, x2, y2 = box
    dx = max(x1 - x, 0.0, x - x2)
    dy = max(y1 - y, 0.0, y - y2)
    return math.hypot(dx, dy)


def _box_diagonal(box: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = box
    return math.hypot(max(0.0, x2 - x1), max(0.0, y2 - y1))


def _player_boxes_by_frame(tracks: Tracks) -> dict[int, list[tuple[float, float, float, float]]]:
    by_frame: dict[int, list[tuple[float, float, float, float]]] = {}
    for player in tracks.players:
        for frame in player.frames:
            frame_idx = int(round(float(frame.t) * tracks.fps))
            by_frame.setdefault(frame_idx, []).append(tuple(float(value) for value in frame.bbox))  # type: ignore[arg-type]
    return by_frame


def _load_cvat_labels(path: Path) -> CvatVideoAnnotations:
    if not path.is_file():
        raise FileNotFoundError(f"missing reviewed CVAT labels: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return CvatVideoAnnotations.model_validate(payload)


def _label_frame_count(labels: CvatVideoAnnotations) -> int:
    max_frame = max((frame.frame_index for frame in labels.frames), default=-1) + 1
    return max(int(labels.task.size), max_frame)


def _labeled_ball_frame_indexes(labels: CvatVideoAnnotations) -> set[int]:
    indexes: set[int] = set()
    for frame in labels.frames:
        if any(box.label.strip().lower() == "ball" for box in frame.boxes):
            indexes.add(int(frame.frame_index))
    return indexes


def _bucket_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = {ON_PLAYER: 0, NEAR_PLAYER: 0, FAR_FIELD: 0, NO_PLAYERS_IN_FRAME: 0}
    for row in rows:
        bucket = str(row["bucket"])
        counts[bucket] = counts.get(bucket, 0) + 1
    return counts


def _bucket_fractions(rows: Sequence[Mapping[str, Any]]) -> dict[str, float | None]:
    total = len(rows)
    counts = _bucket_counts(rows)
    if total == 0:
        return {bucket: None for bucket in counts}
    return {bucket: round(count / total, 4) for bucket, count in counts.items()}


__all__ = [
    "ClipProximityInput",
    "bucket_ball_fps_by_player_proximity",
    "bucket_ball_fps_by_player_proximity_from_files",
    "write_ball_fp_player_proximity",
]

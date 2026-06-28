"""Fuse ball-track candidates without using human click labels."""

from __future__ import annotations

import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any

from .ball_overlay import load_ball_track
from .schemas import BallTrack


def fuse_ball_tracks_with_verifiers(
    *,
    primary_ball_track_path: str | Path,
    stable_ball_track_path: str | Path,
    verifier_ball_track_paths: list[str | Path],
    outlier_distance_px: float = 100.0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Keep a stable backbone and add primary detections confirmed by verifiers.

    The sparse click labels are deliberately not inputs.  This is meant for
    fusing independent model outputs, for example TrackNetV3 plus VballNet.
    """

    if outlier_distance_px < 0.0 or not math.isfinite(float(outlier_distance_px)):
        raise ValueError("outlier_distance_px must be >= 0")
    if not verifier_ball_track_paths:
        raise ValueError("at least one verifier ball track is required")

    primary = load_ball_track(primary_ball_track_path)
    stable = load_ball_track(stable_ball_track_path)
    verifiers = [load_ball_track(path) for path in verifier_ball_track_paths]
    _require_compatible_fps(primary, stable, "stable")
    for index, verifier in enumerate(verifiers):
        _require_compatible_fps(primary, verifier, f"verifier/{index}")

    payload = deepcopy(primary.model_dump(mode="json"))
    primary_samples = _samples_by_index(primary)
    stable_samples = _samples_by_index(stable)
    verifier_samples = [_samples_by_index(verifier) for verifier in verifiers]
    output_samples = _payload_samples_by_index(payload, fps=float(primary.fps))

    kept_stable_count = 0
    kept_primary_consensus_count = 0
    added_verifier_consensus_count = 0
    suppressed_primary_count = 0

    for frame_index, out_frame in output_samples.items():
        primary_frame = primary_samples.get(frame_index)
        stable_frame = stable_samples.get(frame_index)
        verifier_frames = [samples.get(frame_index) for samples in verifier_samples]

        if stable_frame is not None and stable_frame.visible:
            _copy_frame(out_frame, stable_frame.model_dump(mode="json"))
            kept_stable_count += 1
            continue

        if primary_frame is not None and primary_frame.visible:
            if _any_close_visible(primary_frame.xy, verifier_frames, threshold_px=outlier_distance_px):
                kept_primary_consensus_count += 1
                continue
            _hide_frame(out_frame)
            suppressed_primary_count += 1
            continue

        consensus_xy = _verifier_consensus_xy(verifier_frames, threshold_px=outlier_distance_px)
        if consensus_xy is not None:
            out_frame["xy"] = [float(consensus_xy[0]), float(consensus_xy[1])]
            out_frame["conf"] = 0.5
            out_frame["visible"] = True
            out_frame["approx"] = True
            out_frame.pop("world_xyz", None)
            added_verifier_consensus_count += 1
            continue

        _hide_frame(out_frame)

    BallTrack.model_validate(payload)
    visible_before = sum(1 for frame in primary.frames if frame.visible)
    visible_after = sum(1 for frame in payload["frames"] if bool(frame["visible"]))
    summary = {
        "schema_version": 1,
        "artifact_type": "racketsport_ball_model_fusion",
        "status": "fused_not_gate_verified",
        "primary_ball_track": str(primary_ball_track_path),
        "stable_ball_track": str(stable_ball_track_path),
        "verifier_ball_tracks": [str(path) for path in verifier_ball_track_paths],
        "frame_count": len(payload["frames"]),
        "visible_before": visible_before,
        "visible_after": visible_after,
        "kept_stable_count": kept_stable_count,
        "kept_primary_consensus_count": kept_primary_consensus_count,
        "added_verifier_consensus_count": added_verifier_consensus_count,
        "suppressed_primary_count": suppressed_primary_count,
        "outlier_distance_px": float(outlier_distance_px),
        "uses_human_clicks": False,
        "not_ground_truth": True,
    }
    return payload, summary


def write_fused_ball_track(
    *,
    primary_ball_track_path: str | Path,
    stable_ball_track_path: str | Path,
    verifier_ball_track_paths: list[str | Path],
    out_path: str | Path,
    summary_path: str | Path,
    outlier_distance_px: float = 100.0,
) -> dict[str, Any]:
    payload, summary = fuse_ball_tracks_with_verifiers(
        primary_ball_track_path=primary_ball_track_path,
        stable_ball_track_path=stable_ball_track_path,
        verifier_ball_track_paths=verifier_ball_track_paths,
        outlier_distance_px=outlier_distance_px,
    )
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary_out = Path(summary_path)
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _samples_by_index(track: BallTrack) -> dict[int, Any]:
    return {int(round(float(frame.t) * float(track.fps))): frame for frame in track.frames}


def _payload_samples_by_index(payload: dict[str, Any], *, fps: float) -> dict[int, dict[str, Any]]:
    return {int(round(float(frame["t"]) * fps)): frame for frame in payload["frames"]}


def _require_compatible_fps(reference: BallTrack, other: BallTrack, label: str) -> None:
    if not math.isclose(float(reference.fps), float(other.fps), rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError(f"{label} fps must match primary fps")


def _copy_frame(target: dict[str, Any], source: dict[str, Any]) -> None:
    target.clear()
    target.update(deepcopy(source))


def _hide_frame(frame: dict[str, Any]) -> None:
    frame["visible"] = False
    frame["conf"] = 0.0
    frame["approx"] = False
    frame.pop("world_xyz", None)


def _any_close_visible(xy: list[float], frames: list[Any | None], *, threshold_px: float) -> bool:
    return any(
        frame is not None and frame.visible and _distance(xy, frame.xy) <= threshold_px
        for frame in frames
    )


def _verifier_consensus_xy(frames: list[Any | None], *, threshold_px: float) -> tuple[float, float] | None:
    visible = [frame for frame in frames if frame is not None and frame.visible]
    if len(visible) < 2:
        return None
    for index, left in enumerate(visible):
        for right in visible[index + 1 :]:
            if _distance(left.xy, right.xy) <= threshold_px:
                return (
                    (float(left.xy[0]) + float(right.xy[0])) / 2.0,
                    (float(left.xy[1]) + float(right.xy[1])) / 2.0,
                )
    return None


def _distance(left: list[float], right: list[float]) -> float:
    return math.hypot(float(left[0]) - float(right[0]), float(left[1]) - float(right[1]))


__all__ = [
    "fuse_ball_tracks_with_verifiers",
    "write_fused_ball_track",
]

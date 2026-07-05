"""2D image/court-plane bounce detection for BALL tracks."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .ball_manual_court_inout import manual_court_projection_from_corners
from .court_calibration import project_image_points_to_world
from .court_templates import Sport
from .schemas import BallTrack


STATUS_TESTED = "TESTED-ON-REAL-DATA"
ALGORITHM = "image_velocity_inflection_court_plane_2d_v1"


def detect_2d_bounces_from_ball_track(
    ball_track: Mapping[str, Any] | str | Path,
    court_corners: Mapping[str, Any] | str | Path,
    *,
    target_image_size: Sequence[int],
    sport: Sport = "pickleball",
    min_p_bounce: float = 0.5,
    min_separation_s: float = 0.10,
    min_vertical_delta_px: float = 4.0,
    min_candidate_t_s: float = 0.20,
    max_gap_fill_frames: int = 6,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Detect court-plane bounces from image-y velocity sign changes.

    ``target_image_size`` is the (width, height) pixel space the ball
    track's own xy points live in (almost always native source-video
    resolution). Required: see threed/racketsport/ball_manual_court_inout.py's
    module docstring for the pixel-space contract court corner artifacts
    must satisfy and why this can no longer be inferred.
    """

    threshold = _unit_interval(min_p_bounce, "min_p_bounce")
    min_separation = _nonnegative_finite(min_separation_s, "min_separation_s")
    min_vertical_delta = _nonnegative_finite(min_vertical_delta_px, "min_vertical_delta_px")
    min_candidate_t = _nonnegative_finite(min_candidate_t_s, "min_candidate_t_s")
    if max_gap_fill_frames < 0:
        raise ValueError("max_gap_fill_frames must be >= 0")
    track = BallTrack.model_validate(_load_json_or_mapping(ball_track))
    projection = manual_court_projection_from_corners(court_corners, sport=sport, target_image_size=target_image_size)
    samples = _trajectory_samples(track, max_gap_fill_frames=max_gap_fill_frames)
    candidates: list[dict[str, Any]] = []
    accepted: list[dict[str, Any]] = []
    last_accepted_t: float | None = None
    max_sample_gap_frames = max_gap_fill_frames + 1

    for prev, cur, nxt in zip(samples, samples[1:], samples[2:], strict=False):
        sample_gap_ok = (cur["frame"] - prev["frame"]) <= max_sample_gap_frames and (nxt["frame"] - cur["frame"]) <= max_sample_gap_frames
        dy_prev = cur["xy"][1] - prev["xy"][1]
        dy_next = nxt["xy"][1] - cur["xy"][1]
        vertical_sign_change = (
            abs(dy_prev) >= min_vertical_delta
            and abs(dy_next) >= min_vertical_delta
            and dy_prev > 0.0
            and dy_next < 0.0
            and cur["visible"]
            and cur["t"] >= min_candidate_t
        )
        curvature_px = abs(dy_next - dy_prev)
        p_bounce = _bounce_probability(dy_prev, dy_next, min_vertical_delta=min_vertical_delta)
        candidate = {
            "frame": cur["frame"],
            "t": cur["t"],
            "contact_xy_img": cur["xy"],
            "dy_prev_px": dy_prev,
            "dy_next_px": dy_next,
            "curvature_px": curvature_px,
            "p_bounce": p_bounce,
            "vertical_sign_change": vertical_sign_change,
            "sample_gap_ok": sample_gap_ok,
            "visible": cur["visible"],
            "gap_fill": cur["gap_fill"],
        }
        candidates.append(candidate)
        if not vertical_sign_change or p_bounce < threshold:
            continue
        if last_accepted_t is not None and (cur["t"] - last_accepted_t) < min_separation:
            continue
        world_xy = project_image_points_to_world(projection["homography"], [cur["xy"]])[0]
        accepted.append(
            {
                "t": cur["t"],
                "frame": cur["frame"],
                "world_xy": [float(world_xy[0]), float(world_xy[1])],
                "contact_xy_img": cur["xy"],
                "p_bounce": p_bounce,
                "source": ALGORITHM,
                "not_ground_truth": True,
                "render_only": True,
                "not_for_detection_metrics": True,
            }
        )
        last_accepted_t = cur["t"]

    payload = track.model_dump(mode="json")
    payload["bounces"] = accepted
    parsed = BallTrack.model_validate(payload)
    detector = {
        "schema_version": 1,
        "artifact_type": "racketsport_ball_bounce_2d_output",
        "status": STATUS_TESTED,
        "algorithm": ALGORITHM,
        "sport": sport,
        "probability_threshold": threshold,
        "min_separation_s": min_separation,
        "min_vertical_delta_px": min_vertical_delta,
        "min_candidate_t_s": min_candidate_t,
        "max_gap_fill_frames": int(max_gap_fill_frames),
        "input_ball_track_path": str(ball_track) if isinstance(ball_track, str | Path) else None,
        "court_corners_path": str(court_corners) if isinstance(court_corners, str | Path) else None,
        "candidate_count": len(candidates),
        "accepted_bounce_count": len(accepted),
        "accepted_bounces": accepted,
        "candidates": candidates,
        "projection": projection,
        "blocked_reason": None if accepted else "no_2d_bounces_detected",
        "not_ground_truth": True,
    }
    return parsed.model_dump(mode="json"), detector


def write_2d_bounce_ball_track(
    *,
    ball_track_path: str | Path,
    court_corners_path: str | Path,
    out: str | Path,
    detector_out: str | Path,
    target_image_size: Sequence[int],
    sport: Sport = "pickleball",
    min_p_bounce: float = 0.5,
    min_separation_s: float = 0.10,
    min_vertical_delta_px: float = 4.0,
    min_candidate_t_s: float = 0.20,
    max_gap_fill_frames: int = 6,
    command: str | None = None,
) -> dict[str, Any]:
    payload, detector = detect_2d_bounces_from_ball_track(
        ball_track_path,
        court_corners_path,
        target_image_size=target_image_size,
        sport=sport,
        min_p_bounce=min_p_bounce,
        min_separation_s=min_separation_s,
        min_vertical_delta_px=min_vertical_delta_px,
        min_candidate_t_s=min_candidate_t_s,
        max_gap_fill_frames=max_gap_fill_frames,
    )
    if command is not None:
        detector["command"] = command
    out_path = Path(out)
    detector_path = Path(detector_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    detector_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    detector_path.write_text(json.dumps(detector, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return detector


def _trajectory_samples(track: BallTrack, *, max_gap_fill_frames: int) -> list[dict[str, Any]]:
    frame_rows: list[dict[str, Any]] = []
    for index, frame in enumerate(track.frames):
        frame_index = int(round(float(frame.t) * float(track.fps)))
        frame_rows.append(
            {
                "index": index,
                "t": float(frame.t),
                "frame": frame_index,
                "xy": [float(frame.xy[0]), float(frame.xy[1])],
                "conf": float(frame.conf),
                "visible": bool(frame.visible),
            }
        )
    visible_indexes = [index for index, row in enumerate(frame_rows) if row["visible"]]
    bounded_gap_indexes: set[int] = set()
    for left, right in zip(visible_indexes, visible_indexes[1:], strict=False):
        hidden_count = right - left - 1
        if hidden_count <= 0 or hidden_count > max_gap_fill_frames:
            continue
        bounded_gap_indexes.update(range(left + 1, right))

    samples: list[dict[str, Any]] = []
    for index, row in enumerate(frame_rows):
        if not row["visible"] and index not in bounded_gap_indexes:
            continue
        samples.append(
            {
                "t": row["t"],
                "frame": row["frame"],
                "xy": row["xy"],
                "conf": row["conf"],
                "visible": row["visible"],
                "gap_fill": not row["visible"],
            }
        )
    return sorted(samples, key=lambda item: (item["t"], item["frame"]))


def _bounce_probability(dy_prev: float, dy_next: float, *, min_vertical_delta: float) -> float:
    if min_vertical_delta <= 0.0:
        min_vertical_delta = 1.0
    if dy_prev <= 0.0 or dy_next >= 0.0:
        return 0.0
    score = (abs(dy_prev) + abs(dy_next)) / (10.0 * min_vertical_delta)
    return max(0.0, min(1.0, 0.5 + score))


def _load_json_or_mapping(value: Mapping[str, Any] | str | Path) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    path = Path(value)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _unit_interval(value: float, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return number


def _nonnegative_finite(value: float, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be finite and >= 0")
    return number


__all__ = [
    "ALGORITHM",
    "detect_2d_bounces_from_ball_track",
    "write_2d_bounce_ball_track",
]

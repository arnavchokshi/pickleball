"""Manual tap-track fallback for ball trajectories.

These helpers apply reviewer taps to BallTrack-friendly frame dictionaries.
They are CPU-only and do not invoke any detector or learned model.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

from threed.racketsport.ball_tracknet import BallFrameDict, ball_frame, validate_ball_frame


def apply_tap_track_corrections(
    frames: Sequence[Mapping[str, object]],
    corrections: Sequence[Mapping[str, object]],
    *,
    time_tolerance_s: float,
    min_visible_conf: float = 0.0,
) -> list[BallFrameDict]:
    """Apply tap corrections by frame index or nearest timestamp.

    A correction may provide ``frame_index``, ``t``, or both. When both are
    present, the frame index wins only if its timestamp agrees within
    ``time_tolerance_s``.
    """

    tolerance = _require_non_negative_float(time_tolerance_s, "time_tolerance_s")
    corrected = [validate_ball_frame(frame, min_visible_conf=min_visible_conf) for frame in frames]

    for correction in corrections:
        index = _match_frame_index(corrected, correction, time_tolerance_s=tolerance)
        base = corrected[index]
        xy = correction.get("xy", base["xy"])
        conf = correction.get("conf", 1.0 if "xy" in correction else base["conf"])
        visible = correction.get("visible", True if "xy" in correction else base["visible"])
        world_xyz = correction.get("world_xyz", base.get("world_xyz"))

        updated = ball_frame(
            t=float(base["t"]),
            xy=xy,  # type: ignore[arg-type]
            conf=conf,  # type: ignore[arg-type]
            visible=visible,  # type: ignore[arg-type]
            world_xyz=world_xyz,  # type: ignore[arg-type]
            approx=correction.get("approx", False),  # type: ignore[arg-type]
            min_visible_conf=min_visible_conf,
        )
        corrected[index] = updated

    return corrected


def _match_frame_index(
    frames: Sequence[Mapping[str, object]],
    correction: Mapping[str, object],
    *,
    time_tolerance_s: float,
) -> int:
    has_index = "frame_index" in correction
    has_t = "t" in correction
    if not has_index and not has_t:
        raise ValueError("tap correction requires frame_index or t")

    if has_index:
        frame_index = _require_frame_index(correction["frame_index"], len(frames))
        if has_t:
            correction_t = _require_finite_float(correction["t"], "t")
            frame_t = float(frames[frame_index]["t"])
            if abs(frame_t - correction_t) > time_tolerance_s:
                raise ValueError("frame_index/t mismatch exceeds time_tolerance_s")
        return frame_index

    correction_t = _require_finite_float(correction["t"], "t")
    candidates = [
        (index, abs(float(frame["t"]) - correction_t))
        for index, frame in enumerate(frames)
        if abs(float(frame["t"]) - correction_t) <= time_tolerance_s
    ]
    if not candidates:
        raise ValueError("no frame within time_tolerance_s")
    candidates.sort(key=lambda item: (item[1], item[0]))
    if len(candidates) > 1 and math.isclose(candidates[0][1], candidates[1][1], rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("ambiguous frame match within time_tolerance_s")
    return candidates[0][0]


def _require_frame_index(value: object, frame_count: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("frame_index must be an integer")
    if value < 0 or value >= frame_count:
        raise ValueError("frame_index out of range")
    return value


def _require_non_negative_float(value: object, name: str) -> float:
    number = _require_finite_float(value, name)
    if number < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return number


def _require_finite_float(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite")
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


__all__ = ["apply_tap_track_corrections"]

"""CPU-only ball-track primitives for TrackNet-compatible artifacts.

This module intentionally provides deterministic containers and interpolation
helpers only. It does not load TrackNet weights, train a model, or use a GPU.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence


BallFrameDict = dict[str, object]


def ball_frame(
    *,
    t: float,
    xy: Sequence[float],
    conf: float,
    visible: bool,
    world_xyz: Sequence[float] | None = None,
    approx: bool = False,
    min_visible_conf: float = 0.0,
) -> BallFrameDict:
    """Return a dictionary compatible with ``schemas.BallFrame``."""

    frame: BallFrameDict = {
        "t": _require_finite_float(t, "t"),
        "xy": list(_validate_vector(xy, "xy", length=2)),
        "conf": _require_confidence(conf, "conf"),
        "visible": _require_bool(visible, "visible"),
        "approx": _require_bool(approx, "approx"),
    }
    if world_xyz is not None:
        frame["world_xyz"] = list(_validate_vector(world_xyz, "world_xyz", length=3))
    return validate_ball_frame(frame, min_visible_conf=min_visible_conf)


def validate_ball_frame(frame: Mapping[str, object], *, min_visible_conf: float = 0.0) -> BallFrameDict:
    """Validate and normalize one BallTrack frame dictionary."""

    min_conf = _require_confidence(min_visible_conf, "min_visible_conf")
    normalized: BallFrameDict = {
        "t": _require_finite_float(frame.get("t"), "t"),
        "xy": list(_validate_vector(frame.get("xy"), "xy", length=2)),
        "conf": _require_confidence(frame.get("conf"), "conf"),
        "visible": _require_bool(frame.get("visible"), "visible"),
        "approx": _require_bool(frame.get("approx", False), "approx"),
    }

    if "world_xyz" in frame and frame.get("world_xyz") is not None:
        normalized["world_xyz"] = list(_validate_vector(frame.get("world_xyz"), "world_xyz", length=3))
    if "spin_rpm" in frame and frame.get("spin_rpm") is not None:
        normalized["spin_rpm"] = _require_finite_float(frame.get("spin_rpm"), "spin_rpm")

    if normalized["visible"] and normalized["conf"] < min_conf:
        raise ValueError("visible frames require conf >= min_visible_conf")
    return normalized


def interpolate_short_gaps(
    frames: Sequence[Mapping[str, object]],
    *,
    max_gap_frames: int = 2,
    min_visible_conf: float = 0.0,
) -> list[BallFrameDict]:
    """Linearly fill short invisible runs bracketed by visible samples.

    Filled samples are marked ``visible=True`` and ``approx=True``. Longer gaps,
    leading gaps, and trailing gaps are preserved unchanged.
    """

    if max_gap_frames < 0:
        raise ValueError("max_gap_frames must be non-negative")
    normalized = [validate_ball_frame(frame, min_visible_conf=min_visible_conf) for frame in frames]
    filled = [dict(frame) for frame in normalized]

    left_index = _next_visible_index(normalized, start=0)
    while left_index is not None:
        right_index = _next_visible_index(normalized, start=left_index + 1)
        if right_index is None:
            break

        gap_count = right_index - left_index - 1
        if 0 < gap_count <= max_gap_frames:
            left = normalized[left_index]
            right = normalized[right_index]
            for index in range(left_index + 1, right_index):
                ratio = (float(normalized[index]["t"]) - float(left["t"])) / (float(right["t"]) - float(left["t"]))
                replacement = dict(normalized[index])
                replacement["xy"] = _interpolate_vector(left["xy"], right["xy"], ratio)
                replacement["conf"] = min(float(left["conf"]), float(right["conf"]))
                replacement["visible"] = True
                replacement["approx"] = True
                if "world_xyz" in left and "world_xyz" in right:
                    replacement["world_xyz"] = _interpolate_vector(left["world_xyz"], right["world_xyz"], ratio)
                else:
                    replacement.pop("world_xyz", None)
                filled[index] = replacement

        left_index = right_index

    return filled


def _next_visible_index(frames: Sequence[Mapping[str, object]], *, start: int) -> int | None:
    for index in range(start, len(frames)):
        if frames[index]["visible"]:
            return index
    return None


def _interpolate_vector(left: object, right: object, ratio: float) -> list[float]:
    left_values = tuple(float(value) for value in left)  # type: ignore[arg-type]
    right_values = tuple(float(value) for value in right)  # type: ignore[arg-type]
    return [left_value + (right_value - left_value) * ratio for left_value, right_value in zip(left_values, right_values)]


def _require_bool(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a bool")
    return value


def _require_confidence(value: object, name: str) -> float:
    number = _require_finite_float(value, name)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be between 0.0 and 1.0")
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


def _validate_vector(values: object, name: str, *, length: int) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a {length}-vector")
    try:
        vector = tuple(values)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValueError(f"{name} must be a {length}-vector") from exc
    if len(vector) != length:
        raise ValueError(f"{name} must be a {length}-vector")
    return tuple(_require_finite_float(value, f"{name}/{index}") for index, value in enumerate(vector))


__all__ = [
    "BallFrameDict",
    "ball_frame",
    "interpolate_short_gaps",
    "validate_ball_frame",
]

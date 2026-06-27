"""Contact-window post-processing helpers."""

from __future__ import annotations

import math
from typing import Mapping


def build_contact_event(
    *,
    t: float,
    frame: int,
    confidence: float,
    sources: Mapping[str, float],
    t0: float,
    t1: float,
    importance: float,
    player_id: int | None = None,
) -> dict[str, object]:
    """Return a dictionary compatible with the ContactEvent schema."""

    t = _require_finite(t, "t")
    t0 = _require_finite(t0, "t0")
    t1 = _require_finite(t1, "t1")
    confidence = _require_confidence(confidence, "confidence")
    importance = _require_confidence(importance, "importance")
    if frame < 0:
        raise ValueError("frame must be non-negative")
    if t1 < t0:
        raise ValueError("t1 must be greater than or equal to t0")

    return {
        "type": "contact",
        "t": t,
        "frame": int(frame),
        "player_id": player_id,
        "confidence": confidence,
        "sources": {
            "audio": _require_confidence(sources.get("audio"), "sources.audio"),
            "wrist_vel": _require_confidence(sources.get("wrist_vel"), "sources.wrist_vel"),
            "ball_inflection": _require_confidence(sources.get("ball_inflection"), "sources.ball_inflection"),
        },
        "window": {"t0": t0, "t1": t1, "importance": importance},
    }


def build_contact_windows_artifact(events: list[dict[str, object]]) -> dict[str, object]:
    """Return a ContactWindows-compatible artifact dictionary."""

    return {"schema_version": 1, "events": events}


def _require_finite(value: float | None, name: str) -> float:
    if value is None:
        raise ValueError(f"{name} is required")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _require_confidence(value: float | None, name: str) -> float:
    value = _require_finite(value, name)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0.0 and 1.0")
    return value


__all__ = ["build_contact_event", "build_contact_windows_artifact"]

"""Review-only contact-window candidates from prototype event labels."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping


ARTIFACT_TYPE = "racketsport_contact_window_candidates"
SCHEMA_VERSION = 1
FRAME_RE = re.compile(r"frame_(\d+)\.[A-Za-z0-9]+$")


def build_contact_window_candidates_from_label_events(
    events_path: str | Path,
    *,
    fps: float | None = None,
    pre_s: float = 0.08,
    post_s: float = 0.08,
) -> dict[str, Any]:
    """Build review candidates without creating trusted ``contact_windows.json``."""

    if pre_s < 0.0 or post_s < 0.0:
        raise ValueError("pre_s and post_s must be non-negative")
    path = Path(events_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    clip = _clip_name(payload, fallback=path.parents[1].name if len(path.parents) > 1 else path.parent.name)
    resolved_fps = _fps(payload, fps=fps)
    items = payload.get("annotation", {}).get("items", [])
    if not isinstance(items, list):
        raise ValueError("events annotation.items must be a list")

    candidates: list[dict[str, Any]] = []
    rejected_item_count = 0
    for index, item in enumerate(items):
        if not isinstance(item, Mapping):
            rejected_item_count += 1
            continue
        event_type = str(item.get("type", "")).strip()
        if event_type not in {"contact", "bounce", "net_cross"}:
            rejected_item_count += 1
            continue
        frame_idx = _frame_index(item.get("frame"))
        if frame_idx is None:
            rejected_item_count += 1
            continue
        t = frame_idx / resolved_fps
        confidence = _confidence(item.get("confidence"))
        candidates.append(
            {
                "review_id": str(item.get("review_id") or f"event_{index:04d}"),
                "type": event_type,
                "frame": frame_idx,
                "t": t,
                "xy_px": _xy_px(item.get("xy_px")),
                "source_label": str(item.get("label", event_type)),
                "source_status": str(item.get("status", "unknown")),
                "source_confidence": confidence,
                "candidate_confidence": confidence,
                "window": {
                    "t0": max(0.0, t - pre_s),
                    "t1": t + post_s,
                    "importance": confidence,
                },
            }
        )

    candidates.sort(key=lambda candidate: (float(candidate["t"]), int(candidate["frame"]), str(candidate["review_id"])))
    by_type = _counts(candidate["type"] for candidate in candidates)
    by_status = _counts(candidate["source_status"] for candidate in candidates)
    uncertainty_flags = payload.get("confidence", {}).get("uncertainty_flags", [])
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "clip": clip,
        "fps": resolved_fps,
        "source_event_path": str(path),
        "not_gate_verified": True,
        "trusted_for_body": False,
        "promotion_target": "contact_windows.json",
        "candidates": candidates,
        "summary": {
            "candidate_count": len(candidates),
            "rejected_item_count": rejected_item_count,
            "by_type": by_type,
            "by_status": by_status,
            "uncertainty_flags": [str(flag) for flag in uncertainty_flags] if isinstance(uncertainty_flags, list) else [],
        },
    }


def write_contact_window_candidates(path: str | Path, payload: Mapping[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _clip_name(payload: Mapping[str, Any], *, fallback: str) -> str:
    clip = payload.get("clip")
    if isinstance(clip, Mapping) and clip.get("name"):
        return str(clip["name"])
    if isinstance(clip, str) and clip:
        return clip
    return fallback


def _fps(payload: Mapping[str, Any], *, fps: float | None) -> float:
    value = fps
    if value is None:
        clip = payload.get("clip")
        if isinstance(clip, Mapping):
            metadata = clip.get("metadata")
            if isinstance(metadata, Mapping):
                value = metadata.get("frame_rate_fps")  # type: ignore[assignment]
    if value is None:
        raise ValueError("fps is required when clip.metadata.frame_rate_fps is missing")
    value = float(value)
    if value <= 0.0:
        raise ValueError("fps must be positive")
    return value


def _frame_index(value: Any) -> int | None:
    if isinstance(value, int) and value >= 0:
        return value
    if isinstance(value, float) and value >= 0.0 and value.is_integer():
        return int(value)
    if isinstance(value, str):
        match = FRAME_RE.search(value)
        if match:
            return int(match.group(1))
    return None


def _confidence(value: Any) -> float:
    confidence = 0.0 if value is None else float(value)
    if confidence < 0.0:
        return 0.0
    if confidence > 1.0:
        return 1.0
    return confidence


def _xy_px(value: Any) -> list[float] | None:
    if not isinstance(value, list | tuple) or len(value) != 2:
        return None
    return [float(value[0]), float(value[1])]


def _counts(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


__all__ = [
    "ARTIFACT_TYPE",
    "build_contact_window_candidates_from_label_events",
    "write_contact_window_candidates",
]

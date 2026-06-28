"""Convert draft racket label boxes into strict four-corner candidate artifacts."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

from .schemas import RacketCandidates


DEFAULT_PADDLE_DIMS_IN = {"length": 16.0, "width": 8.0}


def racket_labels_to_candidates(
    labels_payload: Mapping[str, Any],
    *,
    fps: float,
    paddle_dims_in: Mapping[str, float] | None = None,
    min_confidence: float = 0.0,
    include_uncertain: bool = True,
) -> tuple[dict[str, Any], dict[str, int]]:
    """Build schema-valid ``racket_candidates.json`` from draft racket label boxes."""

    if fps <= 0.0:
        raise ValueError("fps must be positive")
    if not 0.0 <= min_confidence <= 1.0:
        raise ValueError("min_confidence must be in [0, 1]")

    dims = dict(paddle_dims_in or DEFAULT_PADDLE_DIMS_IN)
    by_player: dict[int, list[dict[str, Any]]] = {}
    counts = {"accepted": 0, "skipped_status": 0, "skipped_confidence": 0, "skipped_invalid": 0}
    for item in _payload_items(labels_payload):
        status = str(item.get("status", "accepted"))
        if status == "rejected" or (status == "uncertain" and not include_uncertain):
            counts["skipped_status"] += 1
            continue
        try:
            conf = float(item.get("confidence", item.get("conf", 1.0)))
            if conf < min_confidence:
                counts["skipped_confidence"] += 1
                continue
            frame_idx = _frame_index(item.get("frame"))
            player_id = _player_id(item.get("player_id", item.get("id")))
            corners = _bbox_corners_px(_bbox_xyxy(item))
            source = _candidate_source(item)
        except (TypeError, ValueError):
            counts["skipped_invalid"] += 1
            continue
        by_player.setdefault(player_id, []).append(
            {
                "t": frame_idx / float(fps),
                "corners_px": corners,
                "conf": conf,
                "source": source,
            }
        )
        counts["accepted"] += 1

    if counts["accepted"] == 0:
        raise ValueError(f"no racket candidates accepted; counts={counts}")

    payload = {
        "schema_version": 1,
        "artifact_type": "racketsport_racket_candidates",
        "fps": float(fps),
        "players": [
            {
                "id": player_id,
                "paddle_dims_in": dims,
                "frames": sorted(frames, key=lambda frame: float(frame["t"])),
            }
            for player_id, frames in sorted(by_player.items())
        ],
    }
    return RacketCandidates.model_validate(payload).model_dump(mode="json"), counts


def load_json_object(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def write_racket_candidates(path: str | Path, candidates: Mapping[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(candidates, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _payload_items(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    annotation = payload.get("annotation")
    if isinstance(annotation, Mapping) and isinstance(annotation.get("items"), list):
        return [item for item in annotation["items"] if isinstance(item, dict)]
    if isinstance(payload.get("items"), list):
        return [item for item in payload["items"] if isinstance(item, dict)]
    return []


def _bbox_xyxy(item: Mapping[str, Any]) -> list[float]:
    raw_xyxy = item.get("bbox_xyxy")
    if isinstance(raw_xyxy, list | tuple) and len(raw_xyxy) == 4:
        x1, y1, x2, y2 = [float(value) for value in raw_xyxy]
        if x2 <= x1 or y2 <= y1:
            raise ValueError("bbox_xyxy must be ordered with positive area")
        return [x1, y1, x2, y2]

    raw_xywh = item.get("bbox")
    if isinstance(raw_xywh, list | tuple) and len(raw_xywh) == 4:
        x, y, w, h = [float(value) for value in raw_xywh]
        if w <= 0.0 or h <= 0.0:
            raise ValueError("bbox width/height must be positive")
        return [x, y, x + w, y + h]

    raise ValueError("label item requires bbox_xyxy or bbox")


def _bbox_corners_px(bbox_xyxy: list[float]) -> list[list[float]]:
    x1, y1, x2, y2 = bbox_xyxy
    return [
        [x1, y1],
        [x2, y1],
        [x2, y2],
        [x1, y2],
    ]


def _frame_index(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("frame must be numeric or frame filename")
    if isinstance(value, int):
        if value < 0:
            raise ValueError("frame must be non-negative")
        return value
    if isinstance(value, float):
        if value < 0:
            raise ValueError("frame must be non-negative")
        return int(value)
    if isinstance(value, str):
        match = re.search(r"(\d+)", value)
        if match:
            return max(0, int(match.group(1)) - 1)
    raise ValueError("frame is missing or invalid")


def _player_id(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, bool):
        raise ValueError("player_id must be integer-like")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        match = re.search(r"-?\d+", value)
        if match:
            return int(match.group(0))
    raise ValueError("player_id must be integer-like")


def _candidate_source(item: Mapping[str, Any]) -> str:
    source = item.get("source") or item.get("teacher_model") or item.get("class_name") or "unknown"
    return f"label_bbox:{source}"


__all__ = [
    "DEFAULT_PADDLE_DIMS_IN",
    "load_json_object",
    "racket_labels_to_candidates",
    "write_racket_candidates",
]

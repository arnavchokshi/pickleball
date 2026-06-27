"""Adapters from prototype label artifacts into tracking detection contracts."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def player_labels_to_detections(
    labels_payload: dict[str, Any],
    *,
    fps: float,
    include_uncertain: bool = False,
    preserve_label_ids: bool = False,
) -> dict[str, Any]:
    """Convert reviewed/teacher player label items into the `track.py` input shape."""

    if fps <= 0:
        raise ValueError("fps must be positive")

    by_frame: dict[int, list[dict[str, Any]]] = {}
    counts = {"accepted": 0, "skipped_status": 0, "skipped_invalid": 0}
    for item in _payload_items(labels_payload):
        status = str(item.get("status", "accepted"))
        if status == "uncertain" and not include_uncertain:
            counts["skipped_status"] += 1
            continue
        try:
            frame = _frame_index(item.get("frame"))
            detection = _detection_from_label_item(item, preserve_label_ids=preserve_label_ids)
        except ValueError:
            counts["skipped_invalid"] += 1
            continue
        by_frame.setdefault(frame, []).append(detection)
        counts["accepted"] += 1

    return {
        "schema_version": 1,
        "artifact_type": "racketsport_person_detections",
        "source": "player_labels",
        "fps": float(fps),
        "frames": [
            {"frame": frame, "detections": detections}
            for frame, detections in sorted(by_frame.items())
            if detections
        ],
        "counts": counts,
        "qualitative_status": "prototype_teacher_detections_not_verified",
    }


def load_json_object(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def fps_from_manifest(path: str | Path | None) -> float | None:
    if path is None:
        return None
    manifest_path = Path(path)
    if not manifest_path.is_file():
        return None
    payload = load_json_object(manifest_path)
    metadata = payload.get("clip", {}).get("metadata", {})
    if isinstance(metadata, dict):
        fps = metadata.get("frame_rate_fps")
        if isinstance(fps, int | float) and fps > 0:
            return float(fps)
    return None


def write_detections(path: str | Path, detections: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(detections, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _payload_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
    annotation = payload.get("annotation")
    if isinstance(annotation, dict) and isinstance(annotation.get("items"), list):
        return [item for item in annotation["items"] if isinstance(item, dict)]
    if isinstance(payload.get("items"), list):
        return [item for item in payload["items"] if isinstance(item, dict)]
    return []


def _detection_from_label_item(item: dict[str, Any], *, preserve_label_ids: bool) -> dict[str, Any]:
    bbox = _bbox_xyxy(item)
    detection: dict[str, Any] = {
        "bbox_xyxy": bbox,
        "conf": float(item.get("confidence", item.get("conf", 1.0))),
        "class": "person",
    }
    if item.get("id") is not None:
        detection["source_id"] = str(item["id"])
        if preserve_label_ids:
            detection["temp_track_id"] = str(item["id"])
    if item.get("review_id") is not None:
        detection["source_review_id"] = str(item["review_id"])
    return detection


def _bbox_xyxy(item: dict[str, Any]) -> list[float]:
    raw_xyxy = item.get("bbox_xyxy")
    if isinstance(raw_xyxy, list | tuple) and len(raw_xyxy) == 4:
        x1, y1, x2, y2 = [float(value) for value in raw_xyxy]
        if x2 < x1 or y2 < y1:
            raise ValueError("bbox_xyxy must be ordered")
        return [x1, y1, x2, y2]

    raw_xywh = item.get("bbox")
    if isinstance(raw_xywh, list | tuple) and len(raw_xywh) == 4:
        x, y, w, h = [float(value) for value in raw_xywh]
        if w < 0 or h < 0:
            raise ValueError("bbox width/height must be non-negative")
        return [x, y, x + w, y + h]

    raise ValueError("label item requires bbox_xyxy or bbox")


def _frame_index(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("frame must be numeric or frame filename")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        match = re.search(r"(\d+)", value)
        if match:
            return max(0, int(match.group(1)) - 1)
    raise ValueError("frame is missing or invalid")


__all__ = [
    "fps_from_manifest",
    "load_json_object",
    "player_labels_to_detections",
    "write_detections",
]

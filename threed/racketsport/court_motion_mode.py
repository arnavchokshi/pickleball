"""Classify camera motion for court proposal/tap constraints."""

from __future__ import annotations


def classify_motion_mode(frame_transforms: list[dict[str, float]]) -> dict[str, object]:
    if not frame_transforms:
        return {"motion_mode": "unknown", "reasons": ["no_motion_evidence"]}
    low_inliers = any(item.get("inliers", 0) < 20 for item in frame_transforms)
    max_translation = max(item.get("translation_px", 0.0) for item in frame_transforms)
    max_rotation = max(item.get("rotation_deg", 0.0) for item in frame_transforms)
    if low_inliers:
        return {"motion_mode": "untrusted_motion", "reasons": ["low_inliers"]}
    if max_translation <= 2.0 and max_rotation <= 0.2:
        return {"motion_mode": "static", "reasons": []}
    return {"motion_mode": "moving", "reasons": ["camera_motion_detected"]}

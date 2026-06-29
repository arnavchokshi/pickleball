"""Detector-only oracle metrics for labeled person boxes."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from .mobile_person_eval import _bbox_iou
from .schemas import PersonGroundTruth


Candidate = dict[str, Any]


def detections_payload_to_candidates(payload: Mapping[str, Any]) -> dict[int, list[Candidate]]:
    """Convert tiled detector payload rows into source-video xywh candidates."""

    by_frame: dict[int, list[Candidate]] = {}
    for frame in payload.get("frames", []):
        frame_index = int(frame.get("frame", frame.get("frame_index", 0)))
        candidates: list[Candidate] = []
        for detection in frame.get("detections", []):
            raw_bbox = detection.get("bbox") or detection.get("bbox_xywh")
            if raw_bbox is None or len(raw_bbox) != 4:
                continue
            if "bbox_xywh" in detection:
                x, y, width, height = [float(value) for value in raw_bbox]
            else:
                x1, y1, x2, y2 = [float(value) for value in raw_bbox]
                width = x2 - x1
                height = y2 - y1
                x = x1
                y = y1
            if width <= 0.0 or height <= 0.0:
                continue
            candidates.append(
                {
                    "bbox_xywh": [x, y, width, height],
                    "confidence": float(detection.get("conf", detection.get("confidence", 0.0))),
                }
            )
        candidates.sort(key=lambda item: -float(item["confidence"]))
        by_frame[frame_index] = candidates
    return by_frame


def score_detector_oracle(
    ground_truth: PersonGroundTruth,
    candidates_by_frame: Mapping[int, Sequence[Candidate]],
    *,
    candidate_limits: Iterable[int] = (4, 8, 12, 20),
    iou_thresholds: Iterable[float] = (0.3, 0.5),
) -> dict[str, Any]:
    """Score whether each GT person exists anywhere in the top-N detector pool."""

    limits = sorted({int(limit) for limit in candidate_limits})
    thresholds = sorted({float(threshold) for threshold in iou_thresholds})
    if not limits or any(limit <= 0 for limit in limits):
        raise ValueError("candidate_limits must contain positive integers")
    if not thresholds or any(threshold <= 0.0 or threshold > 1.0 for threshold in thresholds):
        raise ValueError("iou_thresholds must be in (0, 1]")

    labels_by_frame = {
        int(frame.frame_index): [label for label in frame.labels if not label.ignored]
        for frame in ground_truth.frames
    }
    gt_detections = sum(len(labels) for labels in labels_by_frame.values())
    output: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "racketsport_person_detector_oracle_metrics",
        "clip_id": ground_truth.clip_id,
        "gt_detections": gt_detections,
        "candidate_limits": {},
    }
    for limit in limits:
        limit_report: dict[str, Any] = {}
        for threshold in thresholds:
            hits = 0
            per_track: dict[int, dict[str, int]] = {}
            for frame_index, labels in labels_by_frame.items():
                candidates = list(candidates_by_frame.get(frame_index, []))[:limit]
                candidate_boxes = [candidate["bbox_xywh"] for candidate in candidates]
                for label in labels:
                    track_id = int(label.track_id)
                    track_stats = per_track.setdefault(track_id, {"hits": 0, "total": 0})
                    track_stats["total"] += 1
                    matched = bool(candidate_boxes) and max(_bbox_iou(label.bbox_xywh, bbox) for bbox in candidate_boxes) >= threshold
                    if matched:
                        hits += 1
                        track_stats["hits"] += 1
            limit_report[_threshold_key(threshold)] = {
                "hits": hits,
                "total": gt_detections,
                "recall": _safe_divide(hits, gt_detections),
                "per_track": {
                    str(track_id): {
                        "hits": stats["hits"],
                        "total": stats["total"],
                        "recall": _safe_divide(stats["hits"], stats["total"]),
                    }
                    for track_id, stats in sorted(per_track.items())
                },
            }
        output["candidate_limits"][str(limit)] = limit_report
    return output


def _threshold_key(threshold: float) -> str:
    return f"iou_{threshold:.2f}"


def _safe_divide(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


__all__ = ["detections_payload_to_candidates", "score_detector_oracle"]

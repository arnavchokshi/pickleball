"""Mobile person-tracking metrics for iPhone fast-tier candidates."""

from __future__ import annotations

import itertools
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .schemas import MobilePersonTrackingMetrics, OnDevicePersonTracks, PersonGroundTruth, PersonLabel


def score_mobile_person_tracks(
    ground_truth: PersonGroundTruth,
    predictions: OnDevicePersonTracks,
    *,
    iou_threshold: float = 0.5,
    expected_players: int | None = None,
) -> MobilePersonTrackingMetrics:
    if ground_truth.clip_id != predictions.clip_id:
        raise ValueError(f"clip mismatch: {ground_truth.clip_id!r} != {predictions.clip_id!r}")
    if not 0.0 < iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be in (0, 1]")

    gt_by_frame = {frame.frame_index: [label for label in frame.labels if not label.ignored] for frame in ground_truth.frames}
    ignored_by_frame = {frame.frame_index: [label for label in frame.labels if label.ignored] for frame in ground_truth.frames}
    pred_by_frame = {frame.frame_index: frame.detections for frame in predictions.frames}
    frame_indexes = sorted(set(gt_by_frame) | set(pred_by_frame))
    expected = expected_players or ground_truth.summary.max_valid_players_per_frame

    matches = 0
    false_positives = 0
    false_negatives = 0
    id_switches = 0
    expected_player_frames = 0
    exact_expected_player_frames = 0
    gt_detections = 0
    pred_detections = 0
    last_pred_for_gt: dict[int, int] = {}
    association_counts: dict[tuple[int, int], int] = defaultdict(int)

    for frame_index in frame_indexes:
        gt_labels = gt_by_frame.get(frame_index, [])
        ignored_labels = ignored_by_frame.get(frame_index, [])
        pred_labels = pred_by_frame.get(frame_index, [])
        gt_detections += len(gt_labels)
        if len(gt_labels) == expected:
            expected_player_frames += 1

        frame_matches = _match_frame(gt_labels, pred_labels, iou_threshold=iou_threshold)
        matches += len(frame_matches)
        false_negatives += len(gt_labels) - len(frame_matches)
        matched_pred_indexes = {pred_index for _, pred_index, _ in frame_matches}
        scored_pred_count = len(frame_matches)
        for gt_index, pred_index, _ in frame_matches:
            gt_id = int(gt_labels[gt_index].track_id)
            pred_id = int(pred_labels[pred_index].track_id)
            previous_pred_id = last_pred_for_gt.get(gt_id)
            if previous_pred_id is not None and previous_pred_id != pred_id:
                id_switches += 1
            last_pred_for_gt[gt_id] = pred_id
            association_counts[(gt_id, pred_id)] += 1

        for pred_index, pred in enumerate(pred_labels):
            if pred_index in matched_pred_indexes:
                continue
            if _overlaps_ignored(pred.bbox_xywh, ignored_labels, threshold=iou_threshold):
                continue
            scored_pred_count += 1
            false_positives += 1
        pred_detections += scored_pred_count
        if len(gt_labels) == expected and scored_pred_count == expected:
            exact_expected_player_frames += 1

    idtp = _best_identity_true_positive_count(association_counts)
    idfp = max(0, pred_detections - idtp)
    idfn = max(0, gt_detections - idtp)
    idf1 = _safe_divide(2 * idtp, (2 * idtp) + idfp + idfn)
    mota = 1.0 - _safe_divide(false_negatives + false_positives + id_switches, gt_detections)
    precision = _safe_divide(matches, matches + false_positives)
    recall = _safe_divide(matches, matches + false_negatives)
    expected_coverage = _safe_divide(exact_expected_player_frames, expected_player_frames)

    return MobilePersonTrackingMetrics(
        schema_version=1,
        artifact_type="racketsport_mobile_person_tracking_metrics",
        clip_id=ground_truth.clip_id,
        candidate=predictions.candidate,
        iou_threshold=iou_threshold,
        frames=len(frame_indexes),
        gt_detections=gt_detections,
        pred_detections=pred_detections,
        matches=matches,
        false_positives=false_positives,
        false_negatives=false_negatives,
        id_switches=id_switches,
        idf1=idf1,
        mota=mota,
        precision=precision,
        recall=recall,
        expected_players=expected,
        expected_player_coverage=expected_coverage,
        expected_player_frames=expected_player_frames,
        exact_expected_player_frames=exact_expected_player_frames,
    )


def write_mobile_person_metrics(path: str | Path, metrics: MobilePersonTrackingMetrics) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = metrics.model_dump(mode="json")
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _match_frame(gt_labels: list[PersonLabel], pred_labels: list[Any], *, iou_threshold: float) -> list[tuple[int, int, float]]:
    if not gt_labels or not pred_labels:
        return []
    candidate_scores: list[tuple[float, int, int]] = []
    for gt_index, gt in enumerate(gt_labels):
        for pred_index, pred in enumerate(pred_labels):
            iou = _bbox_iou(gt.bbox_xywh, pred.bbox_xywh)
            if iou >= iou_threshold:
                candidate_scores.append((iou, gt_index, pred_index))
    return [(gt_index, pred_index, iou) for iou, gt_index, pred_index in _best_unique_matches(candidate_scores)]


def _best_unique_matches(candidate_scores: list[tuple[float, int, int]]) -> list[tuple[float, int, int]]:
    best: list[tuple[float, int, int]] = []
    best_key = (-1, -1.0)
    for size in range(len(candidate_scores) + 1):
        for subset in itertools.combinations(candidate_scores, size):
            gt_indexes = [item[1] for item in subset]
            pred_indexes = [item[2] for item in subset]
            if len(set(gt_indexes)) != len(gt_indexes) or len(set(pred_indexes)) != len(pred_indexes):
                continue
            key = (len(subset), sum(item[0] for item in subset))
            if key > best_key:
                best_key = key
                best = list(subset)
    return best


def _best_identity_true_positive_count(association_counts: dict[tuple[int, int], int]) -> int:
    gt_ids = sorted({gt_id for gt_id, _ in association_counts})
    pred_ids = sorted({pred_id for _, pred_id in association_counts})
    if not gt_ids or not pred_ids:
        return 0
    best = 0
    for size in range(1, min(len(gt_ids), len(pred_ids)) + 1):
        for gt_subset in itertools.combinations(gt_ids, size):
            for pred_subset in itertools.permutations(pred_ids, size):
                total = 0
                for gt_id, pred_id in zip(gt_subset, pred_subset, strict=False):
                    total += association_counts.get((gt_id, pred_id), 0)
                best = max(best, total)
    return best


def _overlaps_ignored(bbox_xywh: tuple[float, float, float, float], ignored_labels: list[PersonLabel], *, threshold: float) -> bool:
    return any(_bbox_iou(bbox_xywh, ignored.bbox_xywh) >= threshold for ignored in ignored_labels)


def _bbox_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2 = ax1 + aw
    ay2 = ay1 + ah
    bx2 = bx1 + bw
    by2 = by1 + bh
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    intersection = iw * ih
    if intersection <= 0.0:
        return 0.0
    union = (aw * ah) + (bw * bh) - intersection
    return intersection / union if union > 0.0 else 0.0


def _safe_divide(numerator: float | int, denominator: float | int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


__all__ = ["score_mobile_person_tracks", "write_mobile_person_metrics"]

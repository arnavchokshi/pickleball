"""Type-aware one-to-one event matching and peak selection."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from statistics import mean, median
from typing import Iterable, Sequence

import torch


@dataclass(frozen=True)
class Event:
    frame: int
    class_id: int
    score: float = 1.0


def peak_pick(logits: torch.Tensor, *, threshold: float = 0.5, nms_radius: int = 2) -> list[Event]:
    """Pick non-background local maxima independently for HIT and BOUNCE."""

    if logits.ndim != 2 or logits.shape[1] != 3:
        raise ValueError("expected logits [T,3]")
    probabilities = logits.softmax(-1)
    events: list[Event] = []
    for class_id in (1, 2):
        scores = probabilities[:, class_id]
        candidates = [
            i for i, score in enumerate(scores.tolist())
            if score >= threshold
            and score >= float(scores[max(0, i - 1):min(len(scores), i + 2)].max())
        ]
        kept: list[int] = []
        for index in sorted(candidates, key=lambda i: (-float(scores[i]), i)):
            if all(abs(index - prior) > nms_radius for prior in kept):
                kept.append(index)
        events.extend(Event(i, class_id, float(scores[i])) for i in sorted(kept))
    return sorted(events, key=lambda event: (event.frame, event.class_id))


def greedy_match(
    predictions: Sequence[Event], ground_truth: Sequence[Event], *, tolerance_frames: int
) -> dict[str, object]:
    """Greedy highest-score prediction to nearest same-type unmatched GT."""

    matched_gt: set[int] = set()
    pairs: list[tuple[Event, Event]] = []
    for pred in sorted(predictions, key=lambda event: (-event.score, event.frame, event.class_id)):
        choices = [
            (abs(pred.frame - gt.frame), index, gt)
            for index, gt in enumerate(ground_truth)
            if index not in matched_gt and gt.class_id == pred.class_id
            and abs(pred.frame - gt.frame) <= tolerance_frames
        ]
        if choices:
            _, index, gt = min(choices, key=lambda item: (item[0], item[2].frame, item[1]))
            matched_gt.add(index)
            pairs.append((pred, gt))
    return {"pairs": pairs, "tp": len(pairs), "fp": len(predictions) - len(pairs),
            "fn": len(ground_truth) - len(pairs)}


def event_metrics(
    predictions: Iterable[Event], ground_truth: Iterable[Event], *, tolerance_frames: int, fps: float
) -> dict[str, object]:
    predictions, ground_truth = list(predictions), list(ground_truth)
    by_class: dict[str, object] = {}
    for class_id, name in ((1, "HIT"), (2, "BOUNCE")):
        matched = greedy_match(
            [e for e in predictions if e.class_id == class_id],
            [e for e in ground_truth if e.class_id == class_id],
            tolerance_frames=tolerance_frames,
        )
        tp, fp, fn = int(matched["tp"]), int(matched["fp"]), int(matched["fn"])
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        errors = [(pred.frame - gt.frame) * 1000.0 / fps for pred, gt in matched["pairs"]]
        by_class[name] = {
            "tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "timing_error_ms": {
                "count": len(errors), "mean_signed": mean(errors) if errors else None,
                "median_signed": median(errors) if errors else None,
                "mean_absolute": mean(map(abs, errors)) if errors else None,
                "max_absolute": max(map(abs, errors)) if errors else None,
            },
        }
    return {
        "tolerance_frames": tolerance_frames,
        "tolerance_ms": tolerance_frames * 1000.0 / fps,
        "per_class": by_class,
    }

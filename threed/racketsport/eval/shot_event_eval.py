from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class ShotEventMatch:
    truth_id: str
    prediction_id: str
    dt_s: float
    truth_label: str
    predicted_label: str
    confidence: float
    top2_labels: tuple[str, ...]
    gated: bool


@dataclass(frozen=True)
class ShotEventMatches:
    matched: list[ShotEventMatch]
    unmatched_truth_ids: list[str]
    unmatched_prediction_ids: list[str]


def match_shot_events(
    truth_events: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    *,
    tolerance_s: float,
) -> ShotEventMatches:
    """Greedily pair each truth shot to the nearest unused prediction in time."""

    if tolerance_s < 0:
        raise ValueError("tolerance_s must be non-negative")

    used_prediction_indexes: set[int] = set()
    matched: list[ShotEventMatch] = []
    unmatched_truth_ids: list[str] = []

    for truth in truth_events:
        truth_t = _event_time(truth)
        best_index: int | None = None
        best_dt: float | None = None
        for index, prediction in enumerate(predictions):
            if index in used_prediction_indexes:
                continue
            dt_s = abs(_event_time(prediction) - truth_t)
            if dt_s > tolerance_s:
                continue
            if best_dt is None or dt_s < best_dt:
                best_dt = dt_s
                best_index = index

        if best_index is None or best_dt is None:
            unmatched_truth_ids.append(_event_id(truth))
            continue

        used_prediction_indexes.add(best_index)
        prediction = predictions[best_index]
        matched.append(
            ShotEventMatch(
                truth_id=_event_id(truth),
                prediction_id=_event_id(prediction),
                dt_s=round(best_dt, 6),
                truth_label=_truth_label(truth),
                predicted_label=_prediction_label(prediction),
                confidence=_confidence(prediction),
                top2_labels=_top2_labels(prediction),
                gated=bool(prediction.get("gated", False)),
            )
        )

    unmatched_prediction_ids = [
        _event_id(prediction)
        for index, prediction in enumerate(predictions)
        if index not in used_prediction_indexes
    ]
    return ShotEventMatches(
        matched=matched,
        unmatched_truth_ids=unmatched_truth_ids,
        unmatched_prediction_ids=unmatched_prediction_ids,
    )


def score_shot_events(
    truth_events: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    *,
    tolerance_s: float,
) -> dict[str, Any]:
    matches = match_shot_events(truth_events, predictions, tolerance_s=tolerance_s)
    truth_labels = sorted({_truth_label(event) for event in truth_events})
    top1_correct = sum(1 for match in matches.matched if match.predicted_label == match.truth_label)
    top2_correct = sum(1 for match in matches.matched if match.truth_label in match.top2_labels)

    confusion: dict[str, dict[str, int]] = {}
    for match in matches.matched:
        confusion.setdefault(match.truth_label, {})
        confusion[match.truth_label][match.predicted_label] = (
            confusion[match.truth_label].get(match.predicted_label, 0) + 1
        )

    by_label = {
        label: _label_scores(label, matches.matched)
        for label in truth_labels
    }

    sample_count = len(truth_events)
    unknown_count = sum(1 for match in matches.matched if match.predicted_label == "unknown")
    gated_count = sum(1 for match in matches.matched if match.gated)
    return {
        "sample_count": sample_count,
        "matched_count": len(matches.matched),
        "unmatched_truth_count": len(matches.unmatched_truth_ids),
        "unmatched_prediction_count": len(matches.unmatched_prediction_ids),
        "top1_correct": top1_correct,
        "top2_correct": top2_correct,
        "accuracy": _ratio(top1_correct, sample_count),
        "top2_accuracy": _ratio(top2_correct, sample_count),
        "macro_f1": _mean([scores["f1"] for scores in by_label.values()]),
        "unknown_count": unknown_count,
        "unknown_rate": _ratio(unknown_count, sample_count),
        "gated_count": gated_count,
        "gated_rate": _ratio(gated_count, sample_count),
        "confusion": {label: dict(sorted(counts.items())) for label, counts in sorted(confusion.items())},
        "by_label": by_label,
        "calibration": _calibration(matches.matched),
        "matches": [
            {
                "truth_id": match.truth_id,
                "prediction_id": match.prediction_id,
                "dt_s": match.dt_s,
                "truth_label": match.truth_label,
                "predicted_label": match.predicted_label,
                "confidence": match.confidence,
                "top2": list(match.top2_labels),
                "gated": match.gated,
            }
            for match in matches.matched
        ],
        "unmatched_truth_ids": matches.unmatched_truth_ids,
        "unmatched_prediction_ids": matches.unmatched_prediction_ids,
    }


def _label_scores(label: str, matches: Sequence[ShotEventMatch]) -> dict[str, float | int]:
    true_positive = sum(1 for match in matches if match.truth_label == label and match.predicted_label == label)
    false_positive = sum(1 for match in matches if match.truth_label != label and match.predicted_label == label)
    false_negative = sum(1 for match in matches if match.truth_label == label and match.predicted_label != label)
    precision = _ratio(true_positive, true_positive + false_positive)
    recall = _ratio(true_positive, true_positive + false_negative)
    return {
        "support": true_positive + false_negative,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": _f1(precision, recall),
    }


def _calibration(matches: Sequence[ShotEventMatch]) -> dict[str, dict[str, float | int]]:
    buckets = [
        ("0.00-0.50", 0.0, 0.5),
        ("0.50-0.70", 0.5, 0.7),
        ("0.70-0.85", 0.7, 0.85),
        ("0.85-1.00", 0.85, 1.0000001),
    ]
    payload: dict[str, dict[str, float | int]] = {}
    for name, low, high in buckets:
        items = [match for match in matches if low <= match.confidence < high]
        if not items:
            continue
        correct = sum(1 for match in items if match.truth_label == match.predicted_label)
        payload[name] = {
            "count": len(items),
            "correct": correct,
            "accuracy": _ratio(correct, len(items)),
            "mean_confidence": _mean([match.confidence for match in items]),
        }
    return payload


def _event_time(event: Mapping[str, Any]) -> float:
    if "t" in event:
        return float(event["t"])
    if "time_s" in event:
        return float(event["time_s"])
    raise ValueError("shot event requires t or time_s")


def _event_id(event: Mapping[str, Any]) -> str:
    return str(event.get("id", ""))


def _truth_label(event: Mapping[str, Any]) -> str:
    return str(event.get("shot_label", event.get("type", "unknown")))


def _prediction_label(event: Mapping[str, Any]) -> str:
    return str(event.get("type", event.get("shot_label", "unknown")))


def _confidence(event: Mapping[str, Any]) -> float:
    return float(event.get("type_conf", event.get("confidence", 0.0)))


def _top2_labels(event: Mapping[str, Any]) -> tuple[str, ...]:
    labels: list[str] = []
    for item in event.get("top2", []):
        if isinstance(item, Mapping):
            labels.append(str(item.get("type", item.get("label", "unknown"))))
        elif isinstance(item, Sequence) and not isinstance(item, (str, bytes)) and item:
            labels.append(str(item[0]))
    return tuple(labels[:2])


def _ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _f1(precision: float, recall: float) -> float:
    return 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _load_event_list(path: Path, keys: Sequence[str]) -> list[Mapping[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, Mapping):
        items = None
        for key in keys:
            candidate = payload.get(key)
            if isinstance(candidate, list):
                items = candidate
                break
        if items is None:
            raise ValueError(f"{path} must contain one of: {', '.join(keys)}")
    else:
        raise ValueError(f"{path} must contain a JSON object or array")

    events: list[Mapping[str, Any]] = []
    for index, item in enumerate(items):
        if not isinstance(item, Mapping):
            raise ValueError(f"{path}:{index} must be a JSON object")
        events.append(item)
    return events


def main() -> int:
    parser = argparse.ArgumentParser(description="Score predicted shot events against reviewed truth events.")
    parser.add_argument("--truth", type=Path, required=True, help="Truth JSON list or object containing events/items/shots.")
    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="Prediction JSON list or object containing shots/predictions/events.",
    )
    parser.add_argument("--out", type=Path, required=True, help="Output JSON score path.")
    parser.add_argument("--tolerance-s", type=float, default=0.10)
    args = parser.parse_args()

    truth = _load_event_list(args.truth, ("events", "items", "shots", "truth"))
    predictions = _load_event_list(args.predictions, ("shots", "predictions", "events", "items"))
    payload = score_shot_events(truth, predictions, tolerance_s=args.tolerance_s)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


__all__ = ["ShotEventMatch", "ShotEventMatches", "match_shot_events", "score_shot_events"]


if __name__ == "__main__":
    raise SystemExit(main())

"""Deterministic rally-structure selection over saved event-head scores.

This module is deliberately disconnected from model inference and the product
pipeline.  It accepts JSON-compatible payloads containing saved per-frame
logits or probabilities and emits traceable, typed anchors only when the raw
candidate rate is already physically plausible.  ``VERIFIED=0`` remains
binding and callers must opt in explicitly.
"""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass
from statistics import mean
from typing import Any, Mapping, Sequence


CLASS_NAMES = ("background", "HIT", "BOUNCE")


class SequenceDPError(ValueError):
    """Raised when a saved-score input violates the frozen schema contract."""


@dataclass(frozen=True)
class SequenceDPConfig:
    """Pre-registered parameters; the CLI intentionally exposes no overrides."""

    candidate_threshold: float = 0.05
    nms_radius_frames: int = 2
    min_hit_spacing_s: float = 0.5
    min_event_rate_hz: float = 0.3
    max_event_rate_hz: float = 1.0
    max_hit_rate_hz: float = 1.0
    same_side_penalty: float = 1.0


DEFAULT_CONFIG = SequenceDPConfig()


@dataclass(frozen=True)
class _Candidate:
    candidate_id: int
    frame: int
    class_id: int
    probability: float
    background_probability: float
    side: str | None
    unary_score: float

    @property
    def class_name(self) -> str:
        return CLASS_NAMES[self.class_id]


@dataclass(frozen=True)
class _Path:
    score: float
    selected: tuple[int, ...]


def _require_finite_number(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SequenceDPError(f"{context} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise SequenceDPError(f"{context} must be a finite number")
    return result


def _probabilities(clip: Mapping[str, Any], clip_index: int) -> list[list[float]]:
    has_probabilities = "probabilities" in clip
    has_logits = "logits" in clip
    if has_probabilities == has_logits:
        raise SequenceDPError(
            f"clips[{clip_index}] must contain exactly one of probabilities or logits"
        )
    field = "probabilities" if has_probabilities else "logits"
    rows = clip[field]
    if not isinstance(rows, list) or not rows:
        raise SequenceDPError(f"clips[{clip_index}].{field} must be a non-empty array")
    output: list[list[float]] = []
    for frame, row in enumerate(rows):
        if not isinstance(row, list) or len(row) != len(CLASS_NAMES):
            raise SequenceDPError(
                f"clips[{clip_index}].{field}[{frame}] must contain three class scores"
            )
        values = [
            _require_finite_number(value, f"clips[{clip_index}].{field}[{frame}]")
            for value in row
        ]
        if has_probabilities:
            if any(value < 0.0 or value > 1.0 for value in values):
                raise SequenceDPError(
                    f"clips[{clip_index}].probabilities[{frame}] must lie in [0,1]"
                )
            if not math.isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=1e-5):
                raise SequenceDPError(
                    f"clips[{clip_index}].probabilities[{frame}] must sum to one"
                )
            output.append(values)
        else:
            maximum = max(values)
            exponentials = [math.exp(value - maximum) for value in values]
            denominator = sum(exponentials)
            output.append([value / denominator for value in exponentials])
    return output


def _validate_payload(payload: Mapping[str, Any]) -> list[tuple[Mapping[str, Any], list[list[float]]]]:
    if payload.get("schema_version") != 1:
        raise SequenceDPError("schema_version must be 1")
    if payload.get("artifact_type") != "event_head_sequence_input":
        raise SequenceDPError("artifact_type must be event_head_sequence_input")
    if payload.get("class_names") != list(CLASS_NAMES):
        raise SequenceDPError(f"class_names must be {list(CLASS_NAMES)!r}")
    if payload.get("ground_truth_policy", "none") not in {"none", "dense_exhaustive"}:
        raise SequenceDPError("ground_truth_policy must be none or dense_exhaustive")
    clips = payload.get("clips")
    if not isinstance(clips, list) or not clips:
        raise SequenceDPError("clips must be a non-empty array")

    dense_ground_truth = payload.get("ground_truth_policy") == "dense_exhaustive"
    validated: list[tuple[Mapping[str, Any], list[list[float]]]] = []
    for clip_index, clip in enumerate(clips):
        if not isinstance(clip, Mapping):
            raise SequenceDPError(f"clips[{clip_index}] must be an object")
        fps = _require_finite_number(clip.get("fps"), f"clips[{clip_index}].fps")
        if fps <= 0.0:
            raise SequenceDPError(f"clips[{clip_index}].fps must be positive")
        probabilities = _probabilities(clip, clip_index)
        frame_count = len(probabilities)

        sides = clip.get("hit_side_by_frame")
        if sides is not None:
            if not isinstance(sides, list) or len(sides) != frame_count:
                raise SequenceDPError(
                    f"clips[{clip_index}].hit_side_by_frame must match the score length"
                )
            invalid = [value for value in sides if value not in (None, "A", "B")]
            if invalid:
                raise SequenceDPError(
                    f"clips[{clip_index}].hit_side_by_frame values must be A, B, or null"
                )

        rallies = clip.get("rally_spans")
        if not isinstance(rallies, list) or not rallies:
            raise SequenceDPError(f"clips[{clip_index}].rally_spans must be non-empty")
        previous_end = -1
        rally_ids: set[str] = set()
        for rally_index, rally in enumerate(rallies):
            if not isinstance(rally, Mapping):
                raise SequenceDPError(
                    f"clips[{clip_index}].rally_spans[{rally_index}] must be an object"
                )
            rally_id = rally.get("rally_id")
            if not isinstance(rally_id, str) or not rally_id or rally_id in rally_ids:
                raise SequenceDPError(
                    f"clips[{clip_index}] rally_id values must be non-empty and unique"
                )
            rally_ids.add(rally_id)
            start = rally.get("start_frame")
            end = rally.get("end_frame")
            if isinstance(start, bool) or not isinstance(start, int):
                raise SequenceDPError(f"rally {rally_id} start_frame must be an integer")
            if isinstance(end, bool) or not isinstance(end, int):
                raise SequenceDPError(f"rally {rally_id} end_frame must be an integer")
            if start < 0 or end <= start or end > frame_count:
                raise SequenceDPError(
                    f"rally {rally_id} must use a non-empty half-open span inside the clip"
                )
            if start < previous_end:
                raise SequenceDPError(f"rally {rally_id} overlaps the previous rally span")
            previous_end = end

        ground_truth = clip.get("ground_truth", [])
        if not isinstance(ground_truth, list):
            raise SequenceDPError(f"clips[{clip_index}].ground_truth must be an array")
        if dense_ground_truth and clip.get("ground_truth_complete") is not True:
            raise SequenceDPError(
                f"clips[{clip_index}] must assert ground_truth_complete for a dense judge"
            )
        for event_index, event in enumerate(ground_truth):
            if not isinstance(event, Mapping):
                raise SequenceDPError(
                    f"clips[{clip_index}].ground_truth[{event_index}] must be an object"
                )
            frame = event.get("frame")
            if isinstance(frame, bool) or not isinstance(frame, int) or not 0 <= frame < frame_count:
                raise SequenceDPError(
                    f"clips[{clip_index}].ground_truth[{event_index}].frame is invalid"
                )
            if event.get("class") not in CLASS_NAMES[1:]:
                raise SequenceDPError(
                    f"clips[{clip_index}].ground_truth[{event_index}].class is invalid"
                )
            if dense_ground_truth and sum(
                int(rally["start_frame"]) <= frame < int(rally["end_frame"])
                for rally in rallies
            ) != 1:
                raise SequenceDPError(
                    f"dense ground truth at clips[{clip_index}].ground_truth[{event_index}] "
                    "must belong to exactly one rally"
                )
        validated.append((clip, probabilities))
    return validated


def _peak_candidates(
    probabilities: Sequence[Sequence[float]], *, start: int, end: int,
    sides: Sequence[str | None] | None, config: SequenceDPConfig,
) -> list[_Candidate]:
    raw: list[tuple[int, int, float]] = []
    for class_id in (1, 2):
        candidate_frames = [
            frame
            for frame in range(start, end)
            if probabilities[frame][class_id] >= config.candidate_threshold
            and probabilities[frame][class_id]
            >= max(
                probabilities[nearby][class_id]
                for nearby in range(max(start, frame - 1), min(end, frame + 2))
            )
        ]
        kept: list[int] = []
        for frame in sorted(
            candidate_frames, key=lambda value: (-probabilities[value][class_id], value)
        ):
            if all(abs(frame - prior) > config.nms_radius_frames for prior in kept):
                kept.append(frame)
        raw.extend((frame, class_id, probabilities[frame][class_id]) for frame in kept)

    candidates: list[_Candidate] = []
    epsilon = 1e-9
    for candidate_id, (frame, class_id, probability) in enumerate(
        sorted(raw, key=lambda value: (value[0], value[1]))
    ):
        background = probabilities[frame][0]
        candidates.append(
            _Candidate(
                candidate_id=candidate_id,
                frame=frame,
                class_id=class_id,
                probability=float(probability),
                background_probability=float(background),
                side=(sides[frame] if sides is not None and class_id == 1 else None),
                unary_score=math.log(max(float(probability), epsilon))
                - math.log(max(float(background), epsilon)),
            )
        )
    return candidates


def _candidate_json(candidate: _Candidate) -> dict[str, Any]:
    result: dict[str, Any] = {
        "frame": candidate.frame,
        "class": candidate.class_name,
        "probability": candidate.probability,
        "background_probability": candidate.background_probability,
        "score_trace": {
            "frame": candidate.frame,
            "class_index": candidate.class_id,
        },
    }
    if candidate.side is not None:
        result["side"] = candidate.side
    return result


def _better(candidate: _Path, incumbent: _Path | None) -> bool:
    if incumbent is None:
        return True
    if candidate.score > incumbent.score + 1e-12:
        return True
    if incumbent.score > candidate.score + 1e-12:
        return False
    return candidate.selected < incumbent.selected


def _select_candidates(
    candidates: Sequence[_Candidate], *, fps: float, min_count: int,
    max_count: int, max_hits: int, config: SequenceDPConfig,
) -> _Path | None:
    by_id = {candidate.candidate_id: candidate for candidate in candidates}
    states: dict[tuple[int | None, int, int], _Path] = {(None, 0, 0): _Path(0.0, ())}
    frame_groups: dict[int, list[_Candidate]] = {}
    for candidate in candidates:
        frame_groups.setdefault(candidate.frame, []).append(candidate)
    min_hit_spacing_frames = math.ceil(config.min_hit_spacing_s * fps - 1e-12)

    for frame in sorted(frame_groups):
        updated = dict(states)  # Skip this frame.
        for key, path in states.items():
            last_hit_id, count, hit_count = key
            for candidate in frame_groups[frame]:
                next_count = count + 1
                next_hit_count = hit_count + int(candidate.class_id == 1)
                if next_count > max_count or next_hit_count > max_hits:
                    continue
                next_last_hit_id = last_hit_id
                score = path.score + candidate.unary_score
                if candidate.class_id == 1:
                    if last_hit_id is not None:
                        prior = by_id[last_hit_id]
                        if candidate.frame - prior.frame < min_hit_spacing_frames:
                            continue
                        if (
                            prior.side is not None
                            and candidate.side is not None
                            and prior.side == candidate.side
                        ):
                            score -= config.same_side_penalty
                    next_last_hit_id = candidate.candidate_id
                next_key = (next_last_hit_id, next_count, next_hit_count)
                proposal = _Path(score, path.selected + (candidate.candidate_id,))
                if _better(proposal, updated.get(next_key)):
                    updated[next_key] = proposal
        states = updated

    feasible = [
        path for (_last_hit, count, _hits), path in states.items()
        if min_count <= count <= max_count
    ]
    best: _Path | None = None
    for path in feasible:
        if _better(path, best):
            best = path
    return best


def _apply_rally(
    probabilities: Sequence[Sequence[float]], rally: Mapping[str, Any], *, fps: float,
    sides: Sequence[str | None] | None, config: SequenceDPConfig,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    start, end = int(rally["start_frame"]), int(rally["end_frame"])
    duration_s = (end - start) / fps
    candidates = _peak_candidates(
        probabilities, start=start, end=end, sides=sides, config=config
    )
    raw_predictions = [_candidate_json(candidate) for candidate in candidates]
    raw_rate_hz = len(candidates) / duration_s
    min_count = math.ceil(config.min_event_rate_hz * duration_s - 1e-12)
    max_count = math.floor(config.max_event_rate_hz * duration_s + 1e-12)
    max_hits = math.floor(config.max_hit_rate_hz * duration_s + 1e-12)
    report: dict[str, Any] = {
        "rally_id": rally["rally_id"],
        "start_frame": start,
        "end_frame": end,
        "duration_s": duration_s,
        "raw_rate_hz": raw_rate_hz,
        "raw_predictions": raw_predictions,
        "selected_predictions": None,
        "selected_rate_hz": None,
        "objective_score": None,
        "count_bounds": {"minimum_events": min_count, "maximum_events": max_count,
                         "maximum_hits": max_hits},
    }
    if not config.min_event_rate_hz <= raw_rate_hz <= config.max_event_rate_hz:
        report["status"] = "ineligible_raw_rate"
        report["reason"] = "raw candidate rate is outside the pre-registered physical band"
        return report, []
    if min_count > max_count:
        report["status"] = "ineligible_discrete_rate_band"
        report["reason"] = "rally duration has no integer count inside the physical rate band"
        return report, []

    best = _select_candidates(
        candidates, fps=fps, min_count=min_count, max_count=max_count,
        max_hits=max_hits, config=config,
    )
    if best is None:
        report["status"] = "ineligible_constraints"
        report["reason"] = "saved candidates cannot satisfy all hard constraints"
        return report, []
    selected = [_candidate_json(candidates[candidate_id]) for candidate_id in best.selected]
    report.update({
        "status": "applied",
        "selected_predictions": selected,
        "selected_rate_hz": len(selected) / duration_s,
        "objective_score": best.score,
    })
    return report, selected


def _greedy_match(
    predictions: Sequence[Mapping[str, Any]], ground_truth: Sequence[Mapping[str, Any]],
    *, class_name: str, tolerance_frames: int,
) -> tuple[int, int, int, list[int]]:
    class_predictions = sorted(
        [event for event in predictions if event["class"] == class_name],
        key=lambda event: (-float(event["probability"]), int(event["frame"])),
    )
    class_ground_truth = [event for event in ground_truth if event["class"] == class_name]
    matched_gt: set[int] = set()
    errors: list[int] = []
    for prediction in class_predictions:
        choices = [
            (abs(int(prediction["frame"]) - int(truth["frame"])), index)
            for index, truth in enumerate(class_ground_truth)
            if index not in matched_gt
            and abs(int(prediction["frame"]) - int(truth["frame"])) <= tolerance_frames
        ]
        if choices:
            _, index = min(choices)
            matched_gt.add(index)
            errors.append(int(prediction["frame"]) - int(class_ground_truth[index]["frame"]))
    tp = len(matched_gt)
    return tp, len(class_predictions) - tp, len(class_ground_truth) - tp, errors


def _aggregate_metrics(
    units: Sequence[
        tuple[Sequence[Mapping[str, Any]], Sequence[Mapping[str, Any]], float]
    ],
    *, tolerance_frames: int,
) -> dict[str, Any]:
    totals: dict[str, dict[str, Any]] = {
        class_name: {"tp": 0, "fp": 0, "fn": 0, "errors_ms": []}
        for class_name in CLASS_NAMES[1:]
    }
    for predictions, ground_truth, fps in units:
        for class_name in CLASS_NAMES[1:]:
            tp, fp, fn, errors = _greedy_match(
                predictions, ground_truth, class_name=class_name,
                tolerance_frames=tolerance_frames,
            )
            totals[class_name]["tp"] += tp
            totals[class_name]["fp"] += fp
            totals[class_name]["fn"] += fn
            totals[class_name]["errors_ms"].extend(
                error * 1000.0 / fps for error in errors
            )

    per_class: dict[str, dict[str, Any]] = {}
    micro = {"tp": 0, "fp": 0, "fn": 0}
    for class_name in CLASS_NAMES[1:]:
        tp = int(totals[class_name]["tp"])
        fp = int(totals[class_name]["fp"])
        fn = int(totals[class_name]["fn"])
        errors_ms = list(totals[class_name]["errors_ms"])
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        per_class[class_name] = {
            "tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "mean_absolute_timing_error_ms": mean(map(abs, errors_ms)) if errors_ms else None,
        }
        micro["tp"] += tp
        micro["fp"] += fp
        micro["fn"] += fn
    precision = micro["tp"] / (micro["tp"] + micro["fp"]) if micro["tp"] + micro["fp"] else 0.0
    recall = micro["tp"] / (micro["tp"] + micro["fn"]) if micro["tp"] + micro["fn"] else 0.0
    return {
        "tolerance_frames": tolerance_frames,
        "micro": {
            **micro, "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        },
        "per_class": per_class,
    }


def _metric_delta(raw: Mapping[str, Any], selected: Mapping[str, Any]) -> dict[str, Any]:
    def differences(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
        output = {
            metric: float(right[metric]) - float(left[metric])
            for metric in ("precision", "recall", "f1")
        }
        raw_timing = left.get("mean_absolute_timing_error_ms")
        selected_timing = right.get("mean_absolute_timing_error_ms")
        output["mean_absolute_timing_error_ms"] = (
            float(selected_timing) - float(raw_timing)
            if raw_timing is not None and selected_timing is not None else None
        )
        return output

    return {
        "micro": differences(raw["micro"], selected["micro"]),
        "per_class": {
            class_name: differences(raw["per_class"][class_name], selected["per_class"][class_name])
            for class_name in CLASS_NAMES[1:]
        },
    }


def _evaluation(output: Mapping[str, Any]) -> dict[str, Any]:
    if output.get("ground_truth_policy") != "dense_exhaustive":
        return {"scoreable": False, "reason": "ground_truth_policy is not dense_exhaustive"}
    raw_units: list[
        tuple[Sequence[Mapping[str, Any]], Sequence[Mapping[str, Any]], float]
    ] = []
    selected_units: list[
        tuple[Sequence[Mapping[str, Any]], Sequence[Mapping[str, Any]], float]
    ] = []
    for clip in output["clips"]:
        for rally in clip["sequence_dp_rallies"]:
            rally_truth = [
                event
                for event in clip.get("ground_truth", [])
                if int(rally["start_frame"]) <= int(event["frame"]) < int(rally["end_frame"])
            ]
            if rally["status"] != "applied":
                return {
                    "scoreable": False,
                    "reason": "DP was not applied to every dense rally",
                    "ineligible_rally_id": rally["rally_id"],
                }
            raw_units.append(
                (rally["raw_predictions"], rally_truth, float(clip["fps"]))
            )
            selected_units.append(
                (rally["selected_predictions"], rally_truth, float(clip["fps"]))
            )
    sweep = []
    for tolerance in (1, 2, 5):
        raw_metrics = _aggregate_metrics(raw_units, tolerance_frames=tolerance)
        selected_metrics = _aggregate_metrics(selected_units, tolerance_frames=tolerance)
        sweep.append({
            "raw": raw_metrics,
            "dp": selected_metrics,
            "delta_dp_minus_raw": _metric_delta(raw_metrics, selected_metrics),
        })
    return {
        "scoreable": True,
        "tolerance_sweep": sweep,
    }


def apply_event_sequence_dp(
    payload: Mapping[str, Any], *, enabled: bool = False,
    config: SequenceDPConfig = DEFAULT_CONFIG,
) -> dict[str, Any]:
    """Apply the fixed DP, or return an equal deep copy when default-OFF.

    Disabled mode intentionally performs no schema validation so it remains a
    true passthrough.  Enabled mode never mutates the input mapping.
    """

    if not enabled:
        return copy.deepcopy(dict(payload))
    validated = _validate_payload(payload)
    output = copy.deepcopy(dict(payload))
    output["artifact_type"] = "event_head_sequence_dp_output"
    output["verified"] = False
    output["sequence_dp"] = {
        "enabled": True,
        "default_on": False,
        "parameters": asdict(config),
        "constraints_source": "runs/lanes/rally_dp_20260720/spec.md",
    }
    for clip_output, (clip, probabilities) in zip(output["clips"], validated, strict=True):
        sides = clip.get("hit_side_by_frame")
        reports: list[dict[str, Any]] = []
        anchors: list[dict[str, Any]] = []
        for rally in clip["rally_spans"]:
            report, selected = _apply_rally(
                probabilities, rally, fps=float(clip["fps"]), sides=sides, config=config
            )
            reports.append(report)
            anchors.extend(selected)
        clip_output["sequence_dp_rallies"] = reports
        clip_output["typed_event_anchors"] = sorted(
            anchors, key=lambda event: (event["frame"], event["class"])
        )
    output["sequence_dp_evaluation"] = _evaluation(output)
    return output


def selected_constraint_violations(
    rally_report: Mapping[str, Any], *, fps: float,
    config: SequenceDPConfig = DEFAULT_CONFIG,
) -> list[str]:
    """Return hard-constraint violations for tests and artifact audits."""

    selected = rally_report.get("selected_predictions")
    if selected is None:
        return ["selection_not_applied"]
    violations: list[str] = []
    duration = float(rally_report["duration_s"])
    rate = len(selected) / duration
    if not config.min_event_rate_hz <= rate <= config.max_event_rate_hz:
        violations.append("event_rate")
    hits = [event for event in selected if event["class"] == "HIT"]
    if len(hits) > math.floor(config.max_hit_rate_hz * duration + 1e-12):
        violations.append("maximum_hits")
    minimum_frames = math.ceil(config.min_hit_spacing_s * fps - 1e-12)
    if any(
        int(right["frame"]) - int(left["frame"]) < minimum_frames
        for left, right in zip(hits, hits[1:])
    ):
        violations.append("hit_spacing")
    if len({int(event["frame"]) for event in selected}) != len(selected):
        violations.append("multiple_events_per_frame")
    return violations


__all__ = [
    "CLASS_NAMES", "DEFAULT_CONFIG", "SequenceDPConfig", "SequenceDPError",
    "apply_event_sequence_dp", "selected_constraint_violations",
]

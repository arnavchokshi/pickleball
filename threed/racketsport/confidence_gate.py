"""Confidence-gated physics provenance for virtual-world artifacts.

The gate is additive: it annotates existing values and propagates render-only
physics-fill flags, but it does not mutate measured/corrected coordinates from
upstream artifacts.
"""

from __future__ import annotations

import copy
from collections import Counter
from dataclasses import dataclass
from math import sqrt
from typing import Any, Iterable, Mapping, Sequence


BAND_MEASURED = "measured"
BAND_PHYSICS_PREDICTED = "physics_predicted"
BAND_PHYSICS_PREDICTED_WARN = "physics_predicted_warn"
BAND_PHYSICS_PREDICTED_LOW = "physics_predicted_low"
BAND_PHYSICS_CORRECTED = "physics_corrected"
BAND_PHYSICS_CORRECTED_WARN = "physics_corrected_warn"
BAND_HIDDEN_NO_ANCHOR = "hidden_no_anchor"
BAND_HIDDEN_NO_PREDICTION = "hidden_no_prediction"


@dataclass(frozen=True)
class ConfidenceGateConfig:
    confidence_threshold: float = 0.5
    short_gap_max_frames: int = 12
    hysteresis_frames: int = 3
    max_non_target_displacement_m: float = 0.15


@dataclass(frozen=True)
class ConfidenceSpan:
    kind: str
    start: int
    end: int
    horizon_frames: int


@dataclass(frozen=True)
class CorrectionSanityResult:
    band: str
    max_non_target_displacement_m: float
    warning: str | None = None


def classify_low_confidence_spans(
    confidences: Sequence[float | None],
    *,
    threshold: float,
    short_gap_max_frames: int,
) -> list[ConfidenceSpan]:
    """Classify contiguous below-threshold spans.

    A short gap requires both a previous and a following high-confidence anchor.
    Longer spans with a previous anchor become forward predictions. Prefix spans
    with no previous anchor are hidden.
    """

    spans: list[ConfidenceSpan] = []
    index = 0
    count = len(confidences)
    while index < count:
        if _is_high_confidence(confidences[index], threshold):
            index += 1
            continue
        start = index
        while index < count and not _is_high_confidence(confidences[index], threshold):
            index += 1
        end = index - 1
        span_len = end - start + 1
        has_left_anchor = start > 0 and _is_high_confidence(confidences[start - 1], threshold)
        has_right_anchor = index < count and _is_high_confidence(confidences[index], threshold)
        if not has_left_anchor:
            kind = "NO_ANCHOR"
        elif span_len <= short_gap_max_frames and has_right_anchor:
            kind = "SHORT_GAP"
        else:
            kind = "LONG_GAP"
        spans.append(ConfidenceSpan(kind=kind, start=start, end=end, horizon_frames=span_len))
    return spans


def apply_hysteresis(raw_bands: Sequence[str], *, min_consecutive: int) -> list[str]:
    if min_consecutive <= 1 or len(raw_bands) <= 1:
        return list(raw_bands)
    smoothed = [raw_bands[0]]
    current = raw_bands[0]
    pending: str | None = None
    pending_count = 0
    for band in raw_bands[1:]:
        if band == current:
            pending = None
            pending_count = 0
            smoothed.append(current)
            continue
        if band == pending:
            pending_count += 1
        else:
            pending = band
            pending_count = 1
        if pending_count >= min_consecutive:
            current = band
            pending = None
            pending_count = 0
        smoothed.append(current)
    return smoothed


def band_from_sigma(
    sigma_m: float | None,
    *,
    entity: str,
    horizon_frames: int,
    calibration_curves: Mapping[str, Any] | None,
) -> str:
    if sigma_m is None:
        return BAND_HIDDEN_NO_PREDICTION
    bucket = horizon_bucket(horizon_frames)
    entity_curves = (calibration_curves or {}).get(entity, {})
    bucket_curves = {}
    if isinstance(entity_curves, Mapping):
        buckets = entity_curves.get("horizon_buckets", {})
        if isinstance(buckets, Mapping):
            bucket_curves = buckets.get(bucket, {}) or {}
    p50 = _optional_float(bucket_curves.get("p50_m", bucket_curves.get("p50")))
    p90 = _optional_float(bucket_curves.get("p90_m", bucket_curves.get("p90")))
    if p50 is None or p90 is None:
        return BAND_PHYSICS_PREDICTED_LOW
    if sigma_m <= p50:
        return BAND_PHYSICS_PREDICTED
    if sigma_m <= p90:
        return BAND_PHYSICS_PREDICTED_WARN
    return BAND_PHYSICS_PREDICTED_LOW


def horizon_bucket(horizon_frames: int) -> str:
    if horizon_frames <= 3:
        return "1-3"
    if horizon_frames <= 8:
        return "4-8"
    if horizon_frames <= 15:
        return "9-15"
    return "16+"


def apply_correction_sanity_gate(
    original_joints_world: Sequence[Sequence[float]],
    corrected_joints_world: Sequence[Sequence[float]],
    *,
    target_joint_indices: Iterable[int],
    base_band: str,
    max_non_target_displacement_m: float,
) -> CorrectionSanityResult:
    target_indices = set(target_joint_indices)
    max_displacement = 0.0
    for index, (original, corrected) in enumerate(zip(original_joints_world, corrected_joints_world)):
        if index in target_indices:
            continue
        displacement = _distance(original, corrected)
        max_displacement = max(max_displacement, displacement)
    if max_displacement > max_non_target_displacement_m:
        return CorrectionSanityResult(
            band=BAND_PHYSICS_CORRECTED_WARN,
            max_non_target_displacement_m=max_displacement,
            warning=(
                f"non_target_joint_displacement {max_displacement:.3f}m exceeds "
                f"{max_non_target_displacement_m:.3f}m"
            ),
        )
    return CorrectionSanityResult(band=base_band, max_non_target_displacement_m=max_displacement)


def apply_confidence_gate_to_world(
    virtual_world: Mapping[str, Any],
    *,
    ball_track_physics_filled: Mapping[str, Any] | None,
    physics_footlock: Mapping[str, Any] | None,
    racket_pose_estimate: Mapping[str, Any] | None,
    contact_windows: Mapping[str, Any] | None,
    calibration_curves: Mapping[str, Any] | None,
    config: ConfidenceGateConfig | None = None,
) -> dict[str, Any]:
    cfg = config or ConfidenceGateConfig()
    gated = copy.deepcopy(dict(virtual_world))
    fps = _optional_float(gated.get("fps")) or 30.0

    ball_counts = _annotate_ball_frames(gated, ball_track_physics_filled, calibration_curves, cfg)
    player_counts = _annotate_player_frames(gated, physics_footlock, calibration_curves, cfg)
    paddle_counts = _annotate_paddle_frames(gated, racket_pose_estimate, contact_windows)

    gated["confidence_gate"] = {
        "schema_version": 1,
        "artifact_type": "racketsport_confidence_gate_overlay",
        "fps": fps,
        "config": {
            "confidence_threshold": cfg.confidence_threshold,
            "short_gap_max_frames": cfg.short_gap_max_frames,
            "hysteresis_frames": cfg.hysteresis_frames,
            "max_non_target_displacement_m": cfg.max_non_target_displacement_m,
        },
        "counts_by_entity_band": {
            "ball": dict(ball_counts),
            "player_joints": dict(player_counts),
            "paddle": dict(paddle_counts),
        },
        "policy": {
            "additive_only": True,
            "existing_values_preserved": True,
            "filled_frames_render_only": True,
            "protected_eval_labels_used": False,
        },
    }
    return gated


def summarize_bands(confidence_gated_world: Mapping[str, Any]) -> dict[str, dict[str, int]]:
    gate = confidence_gated_world.get("confidence_gate")
    if isinstance(gate, Mapping):
        counts = gate.get("counts_by_entity_band")
        if isinstance(counts, Mapping):
            return {str(entity): dict(value) for entity, value in counts.items() if isinstance(value, Mapping)}
    return {}


def _annotate_ball_frames(
    world: dict[str, Any],
    ball_track_physics_filled: Mapping[str, Any] | None,
    calibration_curves: Mapping[str, Any] | None,
    config: ConfidenceGateConfig,
) -> Counter[str]:
    ball = world.get("ball")
    if not isinstance(ball, dict):
        return Counter()
    frames = ball.get("frames")
    if not isinstance(frames, list):
        return Counter()
    fill_frames = []
    if isinstance(ball_track_physics_filled, Mapping) and isinstance(ball_track_physics_filled.get("frames"), list):
        fill_frames = ball_track_physics_filled["frames"]
    counts: Counter[str] = Counter()
    annotated_frames: list[dict[str, Any]] = []
    for index, frame in enumerate(frames):
        if not isinstance(frame, dict):
            continue
        fill_frame = fill_frames[index] if index < len(fill_frames) and isinstance(fill_frames[index], Mapping) else {}
        physics_fill = fill_frame.get("physics_fill") if isinstance(fill_frame.get("physics_fill"), Mapping) else {}
        is_filled = fill_frame.get("source") == "physics_interpolated" or bool(physics_fill)
        horizon_frames = int(physics_fill.get("gap_distance_frames", 0) or 0)
        sigma = _optional_float(physics_fill.get("uncertainty_m"))
        if is_filled:
            band = band_from_sigma(
                sigma,
                entity="ball",
                horizon_frames=max(1, horizon_frames),
                calibration_curves=calibration_curves,
            )
            predictor = "BallBallisticAdapter"
            frame["render_only"] = bool(physics_fill.get("render_only", fill_frame.get("render_only", True)))
            frame["not_for_detection_metrics"] = bool(
                physics_fill.get("not_for_detection_metrics", fill_frame.get("not_for_detection_metrics", True))
            )
        elif frame.get("world_xyz") is not None and _is_high_confidence(frame.get("conf"), config.confidence_threshold):
            band = BAND_MEASURED
            predictor = "source_artifact"
        elif frame.get("world_xyz") is not None:
            band = BAND_PHYSICS_PREDICTED_LOW
            predictor = "source_artifact_low_confidence"
        else:
            band = BAND_HIDDEN_NO_ANCHOR if index == 0 else BAND_HIDDEN_NO_PREDICTION
            predictor = "none"
        frame["confidence_provenance"] = {
            "band": band,
            "predictor": predictor,
            "horizon_frames": horizon_frames,
            "predicted_sigma_m": sigma,
        }
        counts[band] += 1
        annotated_frames.append(frame)
    _add_display_bands(annotated_frames, min_consecutive=config.hysteresis_frames)
    return counts


def _annotate_player_frames(
    world: dict[str, Any],
    physics_footlock: Mapping[str, Any] | None,
    calibration_curves: Mapping[str, Any] | None,
    config: ConfidenceGateConfig,
) -> Counter[str]:
    del physics_footlock, calibration_curves
    players = world.get("players")
    if not isinstance(players, list):
        return Counter()
    counts: Counter[str] = Counter()
    for player in players:
        if not isinstance(player, dict):
            continue
        annotated_frames: list[dict[str, Any]] = []
        for frame in player.get("frames", []) or []:
            if not isinstance(frame, dict):
                continue
            trust_band = frame.get("trust_band") if isinstance(frame.get("trust_band"), Mapping) else {}
            gate_status = str(trust_band.get("gate_status", ""))
            predictor = "source_artifact"
            sigma = None
            if gate_status in {"corrected", "contact_locked"} or frame.get("physics") == "physics_footlock":
                band = BAND_PHYSICS_CORRECTED
                predictor = "FootContactLockAdapter"
                sigma = 0.02
                source_joints = frame.get("source_joints_world")
                if isinstance(source_joints, list) and isinstance(frame.get("joints_world"), list):
                    sanity = apply_correction_sanity_gate(
                        source_joints,
                        frame["joints_world"],
                        target_joint_indices=_foot_joint_indices(frame),
                        base_band=band,
                        max_non_target_displacement_m=config.max_non_target_displacement_m,
                    )
                    band = sanity.band
            else:
                mean_conf = _mean_confidence(frame.get("joint_conf"))
                band = BAND_MEASURED if mean_conf is not None and mean_conf >= config.confidence_threshold else BAND_PHYSICS_PREDICTED_LOW
            frame["confidence_provenance"] = {
                "band": band,
                "predictor": predictor,
                "horizon_frames": 0,
                "predicted_sigma_m": sigma,
            }
            counts[band] += 1
            annotated_frames.append(frame)
        _add_display_bands(annotated_frames, min_consecutive=config.hysteresis_frames)
    return counts


def _annotate_paddle_frames(
    world: dict[str, Any],
    racket_pose_estimate: Mapping[str, Any] | None,
    contact_windows: Mapping[str, Any] | None,
) -> Counter[str]:
    counts: Counter[str] = Counter()
    paddles = world.get("paddles")
    if isinstance(paddles, list):
        for paddle in paddles:
            if not isinstance(paddle, dict):
                continue
            annotated_frames: list[dict[str, Any]] = []
            for frame in paddle.get("frames", []) or []:
                if not isinstance(frame, dict):
                    continue
                frame["confidence_provenance"] = {
                    "band": BAND_MEASURED if frame.get("conf", 0.0) >= 0.5 else BAND_PHYSICS_PREDICTED_LOW,
                    "predictor": "source_artifact",
                    "horizon_frames": 0,
                    "predicted_sigma_m": None,
                }
                counts[frame["confidence_provenance"]["band"]] += 1
                annotated_frames.append(frame)
            _add_display_bands(annotated_frames, min_consecutive=3)
    if not counts:
        skipped = 0
        if isinstance(racket_pose_estimate, Mapping):
            summary = racket_pose_estimate.get("summary")
            if isinstance(summary, Mapping):
                skipped = int(summary.get("skipped_contact_count", summary.get("reviewed_contact_count", 0)) or 0)
        if not skipped and isinstance(contact_windows, Mapping) and isinstance(contact_windows.get("events"), list):
            skipped = len(contact_windows["events"])
        if skipped:
            counts[BAND_HIDDEN_NO_PREDICTION] = skipped
    return counts


def _add_display_bands(frames: Sequence[dict[str, Any]], *, min_consecutive: int) -> None:
    raw_bands = []
    for frame in frames:
        provenance = frame.get("confidence_provenance")
        if not isinstance(provenance, dict):
            continue
        raw_bands.append(str(provenance.get("band")))
    if len(raw_bands) != len(frames):
        return
    display_bands = apply_hysteresis(raw_bands, min_consecutive=min_consecutive)
    for frame, display_band in zip(frames, display_bands):
        provenance = frame.get("confidence_provenance")
        if isinstance(provenance, dict):
            provenance["display_band"] = display_band


def _foot_joint_indices(frame: Mapping[str, Any]) -> set[int]:
    foot_contact = frame.get("foot_contact")
    indices: set[int] = set()
    if isinstance(foot_contact, Mapping):
        for value in foot_contact.get("joint_indices", []) or []:
            if isinstance(value, int):
                indices.add(value)
    return indices


def _is_high_confidence(confidence: Any, threshold: float) -> bool:
    value = _optional_float(confidence)
    return value is not None and value >= threshold


def _optional_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if numeric != numeric or numeric in (float("inf"), float("-inf")):
        return None
    return numeric


def _distance(a: Sequence[float], b: Sequence[float]) -> float:
    return sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def _mean_confidence(values: Any) -> float | None:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or not values:
        return None
    numeric = [_optional_float(value) for value in values]
    valid = [value for value in numeric if value is not None]
    if not valid:
        return None
    return sum(valid) / len(valid)


__all__ = [
    "ConfidenceGateConfig",
    "ConfidenceSpan",
    "CorrectionSanityResult",
    "apply_confidence_gate_to_world",
    "apply_correction_sanity_gate",
    "apply_hysteresis",
    "band_from_sigma",
    "classify_low_confidence_spans",
    "horizon_bucket",
    "summarize_bands",
]

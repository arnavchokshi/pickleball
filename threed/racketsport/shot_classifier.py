"""CPU-only shot-classification scaffold primitives.

This module intentionally does not train or run BST/PoseConv3D models. It only
validates inputs, packages deterministic feature-window metadata, gates predicted
labels, and emits metrics-style shot sequences for later model integration.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral
from typing import Any, Sequence


ALLOWED_SHOT_LABELS = (
    "serve",
    "fh_shot",
    "bh_shot",
    "fh_drive",
    "bh_drive",
    "dink",
    "lob",
    "overhead",
    "third_shot_drop",
    "reset_block",
)
UNKNOWN_SHOT_LABEL = "unknown"
DEFAULT_MIN_CONTACT_CONFIDENCE = 0.50
DEFAULT_MIN_PREDICTION_CONFIDENCE = 0.65
DEFAULT_PRE_WINDOW_S = 0.45
DEFAULT_POST_WINDOW_S = 0.45


@dataclass(frozen=True)
class ShotCandidate:
    candidate_id: str
    player_id: int
    contact_t: float
    contact_frame: int
    contact_confidence: float
    audio_event_id: str | None
    pose_track_id: str | None
    ball_event_id: str | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate_id", str(self.candidate_id))
        object.__setattr__(self, "player_id", _int_like(self.player_id, "player_id"))
        object.__setattr__(self, "contact_t", _float_like(self.contact_t, "contact_t"))
        object.__setattr__(self, "contact_frame", _int_like(self.contact_frame, "contact_frame"))
        object.__setattr__(self, "contact_confidence", _float_like(self.contact_confidence, "contact_confidence"))
        if self.audio_event_id is not None:
            object.__setattr__(self, "audio_event_id", str(self.audio_event_id))
        if self.pose_track_id is not None:
            object.__setattr__(self, "pose_track_id", str(self.pose_track_id))
        if self.ball_event_id is not None:
            object.__setattr__(self, "ball_event_id", str(self.ball_event_id))


@dataclass(frozen=True)
class ShotCandidateValidation:
    accepted: bool
    reasons: list[str]


@dataclass(frozen=True)
class ShotPrediction:
    candidate: ShotCandidate
    label: str
    confidence: float
    top2: Sequence[tuple[str, float]] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.candidate, ShotCandidate):
            raise ValueError("candidate must be a ShotCandidate")
        object.__setattr__(self, "label", str(self.label))
        object.__setattr__(self, "confidence", _float_like(self.confidence, "confidence"))

        top2: list[tuple[str, float]] = []
        for index, item in enumerate(self.top2):
            if not isinstance(item, Sequence) or isinstance(item, (str, bytes)) or len(item) != 2:
                raise ValueError(f"top2/{index} must be a label/confidence pair")
            top2.append((str(item[0]), _float_like(item[1], f"top2/{index}.confidence")))
        object.__setattr__(self, "top2", tuple(top2))


def validate_shot_candidate(
    candidate: ShotCandidate,
    *,
    min_contact_confidence: float = DEFAULT_MIN_CONTACT_CONFIDENCE,
) -> ShotCandidateValidation:
    """Validate candidate references before any shot label can be claimed."""

    min_contact_confidence = _confidence_threshold(min_contact_confidence, "min_contact_confidence")
    if min_contact_confidence < 0 or min_contact_confidence > 1:
        raise ValueError("min_contact_confidence must be in [0, 1]")

    reasons: list[str] = []
    if not candidate.candidate_id:
        reasons.append("candidate_id is required")
    if candidate.player_id <= 0:
        reasons.append("player_id must be positive")
    if candidate.contact_t < 0:
        reasons.append("contact_t must be non-negative")
    if candidate.contact_frame < 0:
        reasons.append("contact_frame must be non-negative")
    if not _is_confidence(candidate.contact_confidence):
        reasons.append("contact_confidence must be in [0, 1]")
    elif candidate.contact_confidence < min_contact_confidence:
        reasons.append(f"contact_confidence below {min_contact_confidence:.2f}")
    if not candidate.audio_event_id:
        reasons.append("audio_event_id is required")
    if not candidate.pose_track_id:
        reasons.append("pose_track_id is required")
    if not candidate.ball_event_id:
        reasons.append("ball_event_id is required")

    return ShotCandidateValidation(accepted=not reasons, reasons=reasons)


def build_feature_window(
    candidate: ShotCandidate,
    *,
    pre_s: float = DEFAULT_PRE_WINDOW_S,
    post_s: float = DEFAULT_POST_WINDOW_S,
    fps: float,
) -> dict[str, object]:
    """Build deterministic metadata for later pose/audio/ball feature extraction."""

    validation = validate_shot_candidate(candidate)
    if not validation.accepted:
        raise ValueError("; ".join(validation.reasons))
    pre_s = _float_like(pre_s, "pre_s")
    post_s = _float_like(post_s, "post_s")
    fps = _float_like(fps, "fps")
    if pre_s < 0:
        raise ValueError("pre_s must be non-negative")
    if post_s < 0:
        raise ValueError("post_s must be non-negative")
    if fps <= 0:
        raise ValueError("fps must be positive")

    start_frame = max(0, candidate.contact_frame - round(pre_s * fps))
    end_frame = candidate.contact_frame + round(post_s * fps)

    return {
        "candidate_id": candidate.candidate_id,
        "player_id": candidate.player_id,
        "center_t": _round_time(candidate.contact_t),
        "start_t": _round_time(max(0.0, candidate.contact_t - pre_s)),
        "end_t": _round_time(candidate.contact_t + post_s),
        "center_frame": candidate.contact_frame,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "fps": float(fps),
        "sources": {
            "audio_event_id": candidate.audio_event_id,
            "pose_track_id": candidate.pose_track_id,
            "ball_event_id": candidate.ball_event_id,
        },
    }


def gate_prediction(
    prediction: ShotPrediction,
    *,
    min_confidence: float = DEFAULT_MIN_PREDICTION_CONFIDENCE,
) -> dict[str, object]:
    """Return a fail-closed shot-label payload for a candidate prediction."""

    min_confidence = _confidence_threshold(min_confidence, "min_confidence")

    reasons: list[str] = []
    if prediction.label not in ALLOWED_SHOT_LABELS:
        reasons.append(f"label must be one of {_allowed_label_text()}")
    if not _is_confidence(prediction.confidence):
        reasons.append("confidence must be in [0, 1]")
    elif prediction.confidence < min_confidence:
        reasons.append(f"confidence below {min_confidence:.2f}")

    top2 = []
    for index, (label, confidence) in enumerate(prediction.top2):
        if label not in ALLOWED_SHOT_LABELS:
            reasons.append(f"top2/{index} label must be one of {_allowed_label_text()}")
        if not _is_confidence(confidence):
            reasons.append(f"top2/{index} confidence must be in [0, 1]")
        top2.append({"type": label, "confidence": float(confidence)})

    candidate_validation = validate_shot_candidate(prediction.candidate)
    reasons.extend(candidate_validation.reasons)

    gated = bool(reasons)
    payload: dict[str, object] = {
        "type": UNKNOWN_SHOT_LABEL if gated else prediction.label,
        "type_conf": float(prediction.confidence),
        "gated": gated,
        "gate_reasons": reasons,
    }
    if gated:
        payload["original_type"] = prediction.label
    payload["top2"] = top2
    return payload


def build_shot_sequence_payload(
    *,
    clip_id: str,
    predictions: Sequence[ShotPrediction],
    fps: float,
    min_confidence: float = DEFAULT_MIN_PREDICTION_CONFIDENCE,
    pre_s: float = DEFAULT_PRE_WINDOW_S,
    post_s: float = DEFAULT_POST_WINDOW_S,
) -> dict[str, object]:
    """Package gated predictions in the metrics artifact's player/shot style."""

    if not clip_id:
        raise ValueError("clip_id is required")
    fps = _float_like(fps, "fps")
    grouped: dict[int, list[dict[str, object]]] = {}
    for prediction in sorted(
        predictions,
        key=lambda item: (
            item.candidate.player_id,
            item.candidate.contact_t,
            item.candidate.candidate_id,
        ),
    ):
        gate_payload = gate_prediction(prediction, min_confidence=min_confidence)
        shot_payload = {
            "t": _round_time(prediction.candidate.contact_t),
            **gate_payload,
            "window": build_feature_window(
                prediction.candidate,
                pre_s=pre_s,
                post_s=post_s,
                fps=fps,
            ),
        }
        grouped.setdefault(prediction.candidate.player_id, []).append(shot_payload)

    return {
        "schema_version": 1,
        "clip_id": clip_id,
        "classifier": {
            "name": "shot_classifier_cpu_scaffold",
            "scaffold_only": True,
            "model_training_complete": False,
        },
        "players": [
            {"id": player_id, "shots": grouped[player_id]}
            for player_id in sorted(grouped)
        ],
    }


def _is_confidence(value: float) -> bool:
    if isinstance(value, bool):
        return False
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and 0.0 <= number <= 1.0


def _confidence_threshold(value: Any, name: str) -> float:
    number = _float_like(value, name)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return number


def _int_like(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _float_like(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _allowed_label_text() -> str:
    return ", ".join(sorted(ALLOWED_SHOT_LABELS))


def _round_time(value: float) -> float:
    return round(float(value), 6)


__all__ = [
    "ALLOWED_SHOT_LABELS",
    "UNKNOWN_SHOT_LABEL",
    "ShotCandidate",
    "ShotCandidateValidation",
    "ShotPrediction",
    "build_feature_window",
    "build_shot_sequence_payload",
    "gate_prediction",
    "validate_shot_candidate",
]

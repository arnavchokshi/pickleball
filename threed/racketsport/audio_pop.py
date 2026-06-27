"""CPU-only audio contact timing primitives.

This module intentionally contains deterministic timing helpers only. It does
not run an audio-pop detector, classifier, model, or training pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


DEFAULT_SPEED_OF_SOUND_MPS = 343.0


def _require_finite(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def sound_travel_delay_seconds(distance_m: float, *, speed_of_sound_mps: float = DEFAULT_SPEED_OF_SOUND_MPS) -> float:
    """Return acoustic travel time in seconds for a metric distance."""

    distance_m = _require_finite(distance_m, "distance_m")
    speed_of_sound_mps = _require_finite(speed_of_sound_mps, "speed_of_sound_mps")
    if distance_m < 0.0:
        raise ValueError("distance_m must be non-negative")
    if speed_of_sound_mps <= 0.0:
        raise ValueError("speed_of_sound_mps must be positive")
    return distance_m / speed_of_sound_mps


def correct_audio_onset_to_court_time(
    observed_time_s: float,
    *,
    distance_m: float,
    speed_of_sound_mps: float = DEFAULT_SPEED_OF_SOUND_MPS,
) -> float:
    """Shift a camera/audio observed onset back to estimated court time."""

    observed_time_s = _require_finite(observed_time_s, "observed_time_s")
    if observed_time_s < 0.0:
        raise ValueError("observed_time_s must be non-negative")
    return observed_time_s - sound_travel_delay_seconds(distance_m, speed_of_sound_mps=speed_of_sound_mps)


@dataclass(frozen=True)
class OnsetCandidate:
    """A lightweight onset candidate from a CPU-side heuristic or upstream source."""

    time_s: float
    score: float
    source: str = "audio_pop"
    raw_time_s: float | None = None

    def __post_init__(self) -> None:
        time_s = _require_finite(self.time_s, "time_s")
        score = _require_finite(self.score, "score")
        if time_s < 0.0:
            raise ValueError("time_s must be non-negative")
        if not 0.0 <= score <= 1.0:
            raise ValueError("score must be between 0.0 and 1.0")
        if not self.source:
            raise ValueError("source must be non-empty")
        if self.raw_time_s is not None:
            raw_time_s = _require_finite(self.raw_time_s, "raw_time_s")
            if raw_time_s < 0.0:
                raise ValueError("raw_time_s must be non-negative")


def mel_window_bounds(
    contact_time_s: float,
    *,
    sample_rate_hz: int,
    hop_length: int,
    pre_s: float,
    post_s: float,
    total_frames: int | None = None,
) -> tuple[int, int]:
    """Return clamped half-open mel frame bounds around a contact timestamp."""

    contact_time_s = _require_finite(contact_time_s, "contact_time_s")
    pre_s = _require_finite(pre_s, "pre_s")
    post_s = _require_finite(post_s, "post_s")
    if contact_time_s < 0.0:
        raise ValueError("contact_time_s must be non-negative")
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive")
    if hop_length <= 0:
        raise ValueError("hop_length must be positive")
    if pre_s < 0.0:
        raise ValueError("pre_s must be non-negative")
    if post_s < 0.0:
        raise ValueError("post_s must be non-negative")
    if total_frames is not None and total_frames < 0:
        raise ValueError("total_frames must be non-negative")

    frames_per_second = sample_rate_hz / hop_length
    start_frame = max(0, math.floor((contact_time_s - pre_s) * frames_per_second))
    end_frame = max(start_frame, math.ceil((contact_time_s + post_s) * frames_per_second) + 1)
    if total_frames is not None:
        start_frame = min(start_frame, total_frames)
        end_frame = min(end_frame, total_frames)
    return start_frame, end_frame


def fuse_audio_onsets_to_court_time(
    candidates: list[OnsetCandidate] | tuple[OnsetCandidate, ...],
    *,
    player_camera_distance_m: float,
    speed_of_sound_mps: float = DEFAULT_SPEED_OF_SOUND_MPS,
) -> list[OnsetCandidate]:
    """Return onset candidates shifted from raw audio time to court time."""

    return [
        OnsetCandidate(
            time_s=correct_audio_onset_to_court_time(
                candidate.time_s,
                distance_m=player_camera_distance_m,
                speed_of_sound_mps=speed_of_sound_mps,
            ),
            score=candidate.score,
            source=f"{candidate.source}:court_time",
            raw_time_s=candidate.raw_time_s if candidate.raw_time_s is not None else candidate.time_s,
        )
        for candidate in candidates
    ]


__all__ = [
    "DEFAULT_SPEED_OF_SOUND_MPS",
    "OnsetCandidate",
    "correct_audio_onset_to_court_time",
    "fuse_audio_onsets_to_court_time",
    "mel_window_bounds",
    "sound_travel_delay_seconds",
]

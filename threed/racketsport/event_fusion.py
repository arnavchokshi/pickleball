"""Fusion of audio, ball, and body cues into contact windows.

This module intentionally contains deterministic CPU-only primitives. It does
not run ML models, cross-attention, or learned event ranking.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from threed.racketsport.contact_windows import build_contact_event, build_contact_windows_artifact


DEFAULT_MAX_TIME_DELTA_S = 0.035
DEFAULT_PRE_WINDOW_S = 0.035
DEFAULT_POST_WINDOW_S = 0.055


@dataclass(frozen=True)
class WristVelocityPeak:
    """CPU-side wrist velocity peak near a possible racket contact."""

    time_s: float
    player_id: int
    wrist_world_xyz: tuple[float, float, float]
    speed_mps: float
    confidence: float

    def __post_init__(self) -> None:
        _require_non_negative_time(self.time_s, "time_s")
        _require_vector3(self.wrist_world_xyz, "wrist_world_xyz")
        speed_mps = _require_finite(self.speed_mps, "speed_mps")
        if speed_mps < 0.0:
            raise ValueError("speed_mps must be non-negative")
        _require_confidence(self.confidence, "confidence")


@dataclass(frozen=True)
class BallInflectionCandidate:
    """Ball trajectory inflection near a possible racket contact."""

    time_s: float
    ball_world_xyz: tuple[float, float, float]
    confidence: float

    def __post_init__(self) -> None:
        _require_non_negative_time(self.time_s, "time_s")
        _require_vector3(self.ball_world_xyz, "ball_world_xyz")
        _require_confidence(self.confidence, "confidence")


@dataclass(frozen=True)
class AudioOnsetCandidate:
    """Normalized audio onset cue used by the fusion helper."""

    time_s: float
    confidence: float

    def __post_init__(self) -> None:
        _require_non_negative_time(self.time_s, "time_s")
        _require_confidence(self.confidence, "confidence")


def fuse_contact_windows(
    *,
    fps: float,
    audio_onsets: Sequence[AudioOnsetCandidate | Mapping[str, Any] | Any],
    wrist_velocity_peaks: Sequence[WristVelocityPeak],
    ball_inflections: Sequence[BallInflectionCandidate],
    require_audio: bool = True,
    max_time_delta_s: float = DEFAULT_MAX_TIME_DELTA_S,
    pre_s: float = DEFAULT_PRE_WINDOW_S,
    post_s: float = DEFAULT_POST_WINDOW_S,
) -> dict[str, object]:
    """Fuse required audio, wrist, and ball cues into ContactWindows dicts.

    Events fail closed: a contact is emitted only when all three required source
    families have candidates inside the temporal gate.
    """

    fps = _require_finite(fps, "fps")
    max_time_delta_s = _require_finite(max_time_delta_s, "max_time_delta_s")
    pre_s = _require_finite(pre_s, "pre_s")
    post_s = _require_finite(post_s, "post_s")
    if fps <= 0.0:
        raise ValueError("fps must be positive")
    if max_time_delta_s < 0.0:
        raise ValueError("max_time_delta_s must be non-negative")
    if pre_s < 0.0 or post_s < 0.0:
        raise ValueError("pre_s and post_s must be non-negative")

    normalized_audio = sorted(
        (_coerce_audio_onset(candidate) for candidate in audio_onsets),
        key=lambda candidate: candidate.time_s,
    )
    sorted_ball = sorted(ball_inflections, key=lambda candidate: candidate.time_s)
    sorted_wrist = sorted(wrist_velocity_peaks, key=lambda candidate: (candidate.time_s, candidate.player_id))

    if require_audio and not normalized_audio:
        return build_contact_windows_artifact([])
    if not sorted_ball or not sorted_wrist:
        return build_contact_windows_artifact([])

    events: list[dict[str, object]] = []
    used_ball_indices: set[int] = set()
    used_wrist_indices: set[int] = set()

    if not require_audio:
        for ball_idx, ball in enumerate(sorted_ball):
            if ball_idx in used_ball_indices:
                continue
            wrist_match = _nearest_wrist_to_ball_only_match(
                ball=ball,
                wrists=sorted_wrist,
                used_indices=used_wrist_indices,
                max_time_delta_s=max_time_delta_s,
            )
            if wrist_match is None:
                continue

            wrist_idx, wrist = wrist_match
            source_times = (wrist.time_s, ball.time_s)
            event_t = (min(source_times) + max(source_times)) / 2.0
            source_confidences = {
                "wrist_vel": wrist.confidence,
                "ball_inflection": ball.confidence,
            }
            confidence = round(sum(source_confidences.values()) / len(source_confidences), 12)
            frame = max(0, int(round(event_t * fps)))

            events.append(
                build_contact_event(
                    t=event_t,
                    frame=frame,
                    player_id=wrist.player_id,
                    confidence=confidence,
                    sources=source_confidences,
                    t0=max(0.0, event_t - pre_s),
                    t1=event_t + post_s,
                    importance=confidence,
                )
            )
            used_ball_indices.add(ball_idx)
            used_wrist_indices.add(wrist_idx)

        events.sort(key=lambda event: (float(event["t"]), int(event["frame"])))
        return build_contact_windows_artifact(events)

    for audio in normalized_audio:
        ball_match = _nearest_ball_match(audio, sorted_ball, used_ball_indices, max_time_delta_s)
        if ball_match is None:
            continue

        ball_idx, ball = ball_match
        wrist_match = _nearest_wrist_to_ball_match(
            audio=audio,
            ball=ball,
            wrists=sorted_wrist,
            used_indices=used_wrist_indices,
            max_time_delta_s=max_time_delta_s,
        )
        if wrist_match is None:
            continue

        wrist_idx, wrist = wrist_match
        source_times = (audio.time_s, wrist.time_s, ball.time_s)
        event_t = (min(source_times) + max(source_times)) / 2.0
        source_confidences = {
            "audio": audio.confidence,
            "wrist_vel": wrist.confidence,
            "ball_inflection": ball.confidence,
        }
        confidence = round(sum(source_confidences.values()) / len(source_confidences), 12)
        frame = max(0, int(round(event_t * fps)))

        events.append(
            build_contact_event(
                t=event_t,
                frame=frame,
                player_id=wrist.player_id,
                confidence=confidence,
                sources=source_confidences,
                t0=max(0.0, event_t - pre_s),
                t1=event_t + post_s,
                importance=confidence,
            )
        )
        used_ball_indices.add(ball_idx)
        used_wrist_indices.add(wrist_idx)

    events.sort(key=lambda event: (float(event["t"]), int(event["frame"])))
    return build_contact_windows_artifact(events)


def fuse_contact_windows_from_cue_payloads(
    *,
    fps: float,
    audio_onsets_payload: Any,
    wrist_velocity_peaks_payload: Any,
    ball_inflections_payload: Any,
    require_audio: bool = True,
    max_time_delta_s: float = DEFAULT_MAX_TIME_DELTA_S,
    pre_s: float = DEFAULT_PRE_WINDOW_S,
    post_s: float = DEFAULT_POST_WINDOW_S,
) -> dict[str, object]:
    """Fuse canonical cue artifact payloads into a ContactWindows artifact."""

    audio_onsets = _items(audio_onsets_payload, keys=("onsets", "audio_onsets", "items"))
    wrist_velocity_peaks = [
        WristVelocityPeak(
            time_s=float(item["time_s"]),
            player_id=int(item["player_id"]),
            wrist_world_xyz=tuple(item["wrist_world_xyz"]),
            speed_mps=float(item["speed_mps"]),
            confidence=float(item["confidence"]),
        )
        for item in _items(wrist_velocity_peaks_payload, keys=("peaks", "wrist_velocity_peaks", "items"))
    ]
    ball_inflections = [
        BallInflectionCandidate(
            time_s=float(item["time_s"]),
            ball_world_xyz=tuple(item["ball_world_xyz"]),
            confidence=float(item["confidence"]),
        )
        for item in _items(ball_inflections_payload, keys=("candidates", "ball_inflections", "items"))
    ]
    return fuse_contact_windows(
        fps=fps,
        audio_onsets=audio_onsets,
        wrist_velocity_peaks=wrist_velocity_peaks,
        ball_inflections=ball_inflections,
        require_audio=require_audio,
        max_time_delta_s=max_time_delta_s,
        pre_s=pre_s,
        post_s=post_s,
    )


def fuse_contact_windows_from_cue_files(
    *,
    fps: float,
    audio_onsets_path: str | Path | None,
    wrist_velocity_peaks_path: str | Path,
    ball_inflections_path: str | Path,
    require_audio: bool = True,
    max_time_delta_s: float = DEFAULT_MAX_TIME_DELTA_S,
    pre_s: float = DEFAULT_PRE_WINDOW_S,
    post_s: float = DEFAULT_POST_WINDOW_S,
) -> dict[str, object]:
    """Read cue artifact files and fuse them into a ContactWindows artifact."""

    return fuse_contact_windows_from_cue_payloads(
        fps=fps,
        audio_onsets_payload=_read_json(Path(audio_onsets_path)) if audio_onsets_path is not None else [],
        wrist_velocity_peaks_payload=_read_json(Path(wrist_velocity_peaks_path)),
        ball_inflections_payload=_read_json(Path(ball_inflections_path)),
        require_audio=require_audio,
        max_time_delta_s=max_time_delta_s,
        pre_s=pre_s,
        post_s=post_s,
    )


def _coerce_audio_onset(candidate: AudioOnsetCandidate | Mapping[str, Any] | Any) -> AudioOnsetCandidate:
    if isinstance(candidate, AudioOnsetCandidate):
        return candidate
    if isinstance(candidate, Mapping):
        time_s = candidate.get("time_s", candidate.get("t", candidate.get("time")))
        confidence = candidate.get("confidence", candidate.get("score", candidate.get("conf")))
    else:
        time_s = getattr(candidate, "time_s", getattr(candidate, "t", None))
        confidence = getattr(candidate, "confidence", getattr(candidate, "score", getattr(candidate, "conf", None)))
    return AudioOnsetCandidate(
        time_s=_require_non_negative_time(time_s, "audio_onset.time_s"),
        confidence=_require_confidence(confidence, "audio_onset.confidence"),
    )


def _nearest_ball_match(
    audio: AudioOnsetCandidate,
    candidates: Sequence[BallInflectionCandidate],
    used_indices: set[int],
    max_time_delta_s: float,
) -> tuple[int, BallInflectionCandidate] | None:
    matches = [
        (idx, candidate)
        for idx, candidate in enumerate(candidates)
        if idx not in used_indices and abs(candidate.time_s - audio.time_s) <= max_time_delta_s
    ]
    if not matches:
        return None
    return min(matches, key=lambda item: (abs(item[1].time_s - audio.time_s), -item[1].confidence, item[0]))


def _nearest_wrist_to_ball_match(
    *,
    audio: AudioOnsetCandidate,
    ball: BallInflectionCandidate,
    wrists: Sequence[WristVelocityPeak],
    used_indices: set[int],
    max_time_delta_s: float,
) -> tuple[int, WristVelocityPeak] | None:
    center_t = (audio.time_s + ball.time_s) / 2.0
    matches = [
        (idx, candidate)
        for idx, candidate in enumerate(wrists)
        if idx not in used_indices
        and abs(candidate.time_s - audio.time_s) <= max_time_delta_s
        and abs(candidate.time_s - ball.time_s) <= max_time_delta_s
    ]
    if not matches:
        return None
    return min(
        matches,
        key=lambda item: (
            _distance_m(item[1].wrist_world_xyz, ball.ball_world_xyz),
            abs(item[1].time_s - center_t),
            -item[1].confidence,
            item[1].player_id,
            item[0],
        ),
    )


def _nearest_wrist_to_ball_only_match(
    *,
    ball: BallInflectionCandidate,
    wrists: Sequence[WristVelocityPeak],
    used_indices: set[int],
    max_time_delta_s: float,
) -> tuple[int, WristVelocityPeak] | None:
    matches = [
        (idx, candidate)
        for idx, candidate in enumerate(wrists)
        if idx not in used_indices and abs(candidate.time_s - ball.time_s) <= max_time_delta_s
    ]
    if not matches:
        return None
    return min(
        matches,
        key=lambda item: (
            _distance_m(item[1].wrist_world_xyz, ball.ball_world_xyz),
            abs(item[1].time_s - ball.time_s),
            -item[1].confidence,
            item[1].player_id,
            item[0],
        ),
    )


def _distance_m(left: Sequence[float], right: Sequence[float]) -> float:
    return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(left, right)))


def _require_non_negative_time(value: float | None, name: str) -> float:
    value = _require_finite(value, name)
    if value < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _require_confidence(value: float | None, name: str) -> float:
    value = _require_finite(value, name)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0.0 and 1.0")
    return value


def _require_vector3(value: Sequence[float], name: str) -> tuple[float, float, float]:
    if len(value) != 3:
        raise ValueError(f"{name} must be a 3-vector")
    return tuple(_require_finite(component, name) for component in value)


def _require_finite(value: float | None, name: str) -> float:
    if value is None:
        raise ValueError(f"{name} is required")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _items(payload: Any, *, keys: tuple[str, ...]) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, Mapping):
        items = next((payload[key] for key in keys if isinstance(payload.get(key), list)), None)
        if items is None:
            raise ValueError(f"cue artifact must contain one of: {', '.join(keys)}")
    else:
        raise ValueError("cue artifact must be an object or array")
    if not all(isinstance(item, dict) for item in items):
        raise ValueError("cue artifact items must be objects")
    return list(items)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid cue artifact JSON: {path}: {exc}") from exc


__all__ = [
    "AudioOnsetCandidate",
    "BallInflectionCandidate",
    "DEFAULT_MAX_TIME_DELTA_S",
    "DEFAULT_POST_WINDOW_S",
    "DEFAULT_PRE_WINDOW_S",
    "WristVelocityPeak",
    "fuse_contact_windows",
    "fuse_contact_windows_from_cue_files",
    "fuse_contact_windows_from_cue_payloads",
]

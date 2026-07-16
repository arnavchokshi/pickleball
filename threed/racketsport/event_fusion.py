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
from threed.racketsport.io_decode import nearest_frame_for_time, time_for_frame


DEFAULT_MAX_TIME_DELTA_S = 0.035
DEFAULT_PRE_WINDOW_S = 0.035
DEFAULT_POST_WINDOW_S = 0.055
DEFAULT_WRIST_ONLY_PRE_WINDOW_S = 0.12
DEFAULT_WRIST_ONLY_POST_WINDOW_S = 0.18
DEFAULT_WRIST_ONLY_MIN_SEPARATION_S = 0.18
DEFAULT_WRIST_ONLY_CONFIDENCE_CAP = 0.35
WRIST_ONLY_TRUST_BAND_NOTE = "wrist-cue-only, unverified"
LOW_TRUST_BALL_INFLECTION_CONFIDENCE_THRESHOLD = 0.25
LOW_TRUST_BALL_INFLECTION_NOTE = "low-trust ball-inflection cue, unverified"
IMAGE_SPACE_BALL_INFLECTION_NOTE = "image-space ball-inflection cue, unverified"
POP_LIKELIHOOD_LOG_BOUND = 0.20
POP_WINDOW_SCALE_BOUND = 0.10


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
    ball_world_xyz: tuple[float, float, float] | None
    confidence: float

    def __post_init__(self) -> None:
        _require_non_negative_time(self.time_s, "time_s")
        if self.ball_world_xyz is not None:
            _require_vector3(self.ball_world_xyz, "ball_world_xyz")
        _require_confidence(self.confidence, "confidence")


@dataclass(frozen=True)
class AudioOnsetCandidate:
    """Normalized audio onset cue used by the fusion helper."""

    time_s: float
    confidence: float
    features: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        _require_non_negative_time(self.time_s, "time_s")
        _require_confidence(self.confidence, "confidence")
        if self.features is not None:
            if not isinstance(self.features, Mapping):
                raise ValueError("features must be a mapping when present")
            # Copy, but do not normalize, clip, or otherwise rewrite the raw
            # onset evidence. Fusion reads a bounded view of pop_band_ratio.
            object.__setattr__(self, "features", dict(self.features))


def fuse_contact_windows(
    *,
    fps: float,
    frame_times: Any = None,
    audio_onsets: Sequence[AudioOnsetCandidate | Mapping[str, Any] | Any],
    wrist_velocity_peaks: Sequence[WristVelocityPeak],
    ball_inflections: Sequence[BallInflectionCandidate],
    require_audio: bool = True,
    max_time_delta_s: float = DEFAULT_MAX_TIME_DELTA_S,
    pre_s: float = DEFAULT_PRE_WINDOW_S,
    post_s: float = DEFAULT_POST_WINDOW_S,
    allow_wrist_only_contact_hints: bool = False,
    wrist_only_pre_s: float = DEFAULT_WRIST_ONLY_PRE_WINDOW_S,
    wrist_only_post_s: float = DEFAULT_WRIST_ONLY_POST_WINDOW_S,
    wrist_only_min_separation_s: float = DEFAULT_WRIST_ONLY_MIN_SEPARATION_S,
    wrist_only_confidence_cap: float = DEFAULT_WRIST_ONLY_CONFIDENCE_CAP,
) -> dict[str, object]:
    """Fuse required audio, wrist, and ball cues into ContactWindows dicts.

    Events fail closed: a contact is emitted only when all three required source
    families have candidates inside the temporal gate.
    """

    fps = _require_finite(fps, "fps")
    max_time_delta_s = _require_finite(max_time_delta_s, "max_time_delta_s")
    pre_s = _require_finite(pre_s, "pre_s")
    post_s = _require_finite(post_s, "post_s")
    wrist_only_pre_s = _require_finite(wrist_only_pre_s, "wrist_only_pre_s")
    wrist_only_post_s = _require_finite(wrist_only_post_s, "wrist_only_post_s")
    wrist_only_min_separation_s = _require_finite(
        wrist_only_min_separation_s,
        "wrist_only_min_separation_s",
    )
    wrist_only_confidence_cap = _require_confidence(
        wrist_only_confidence_cap,
        "wrist_only_confidence_cap",
    )
    if fps <= 0.0:
        raise ValueError("fps must be positive")
    if max_time_delta_s < 0.0:
        raise ValueError("max_time_delta_s must be non-negative")
    if pre_s < 0.0 or post_s < 0.0:
        raise ValueError("pre_s and post_s must be non-negative")
    if wrist_only_pre_s < 0.0 or wrist_only_post_s < 0.0:
        raise ValueError("wrist_only_pre_s and wrist_only_post_s must be non-negative")
    if wrist_only_min_separation_s < 0.0:
        raise ValueError("wrist_only_min_separation_s must be non-negative")

    normalized_audio = sorted(
        (_coerce_audio_onset(candidate, fps=fps, frame_times=frame_times) for candidate in audio_onsets),
        key=lambda candidate: candidate.time_s,
    )
    sorted_ball = sorted(ball_inflections, key=lambda candidate: candidate.time_s)
    sorted_wrist = sorted(wrist_velocity_peaks, key=lambda candidate: (candidate.time_s, candidate.player_id))

    if not sorted_wrist:
        return build_contact_windows_artifact([])
    if require_audio and not normalized_audio:
        if allow_wrist_only_contact_hints:
            return _wrist_only_contact_hints(
                fps=fps,
                frame_times=frame_times,
                wrists=sorted_wrist,
                pre_s=wrist_only_pre_s,
                post_s=wrist_only_post_s,
                min_separation_s=wrist_only_min_separation_s,
                confidence_cap=wrist_only_confidence_cap,
            )
        return build_contact_windows_artifact([])
    if not sorted_ball:
        if allow_wrist_only_contact_hints:
            return _wrist_only_contact_hints(
                fps=fps,
                frame_times=frame_times,
                wrists=sorted_wrist,
                pre_s=wrist_only_pre_s,
                post_s=wrist_only_post_s,
                min_separation_s=wrist_only_min_separation_s,
                confidence_cap=wrist_only_confidence_cap,
            )
        return build_contact_windows_artifact([])

    events: list[dict[str, object]] = []
    used_ball_indices: set[int] = set()
    used_wrist_indices: set[int] = set()

    # Optional audio is still evidence.  ``require_audio=False`` means each
    # visual wrist+ball proposal survives when no nearby audio candidate
    # exists; a nearby audio candidate augments that proposal instead of
    # becoming a new hard gate.
    if not require_audio:
        used_audio_indices: set[int] = set()
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
            audio_match = _nearest_audio_to_visual_match(
                ball=ball,
                wrist=wrist,
                candidates=normalized_audio,
                used_indices=used_audio_indices,
                max_time_delta_s=max_time_delta_s,
            )
            audio_idx, audio = audio_match if audio_match is not None else (None, None)
            source_times = (
                (wrist.time_s, ball.time_s, audio.time_s)
                if audio is not None
                else (wrist.time_s, ball.time_s)
            )
            event_t = (min(source_times) + max(source_times)) / 2.0
            source_confidences = {
                "wrist_vel": wrist.confidence,
                "ball_inflection": ball.confidence,
            }
            if audio is not None:
                source_confidences["audio"] = audio.confidence
            confidence = round(sum(source_confidences.values()) / len(source_confidences), 12)
            event_pre_s = pre_s
            event_post_s = post_s
            if audio is not None:
                confidence, event_pre_s, event_post_s = _apply_audio_pop_soft_evidence(
                    confidence=confidence,
                    pre_s=pre_s,
                    post_s=post_s,
                    audio=audio,
                )
            frame = _event_frame(event_t, fps=fps, frame_times=frame_times)

            events.append(
                build_contact_event(
                    t=event_t,
                    frame=frame,
                    player_id=wrist.player_id,
                    confidence=confidence,
                    sources=source_confidences,
                    t0=max(0.0, event_t - event_pre_s),
                    t1=event_t + event_post_s,
                    importance=confidence,
                    trust_band_note=_ball_contact_trust_note(ball),
                )
            )
            used_ball_indices.add(ball_idx)
            used_wrist_indices.add(wrist_idx)
            if audio_idx is not None:
                used_audio_indices.add(audio_idx)

        events.sort(key=lambda event: (float(event["t"]), int(event["frame"])))
        if events or not allow_wrist_only_contact_hints:
            return build_contact_windows_artifact(events)
        return _wrist_only_contact_hints(
            fps=fps,
            frame_times=frame_times,
            wrists=sorted_wrist,
            pre_s=wrist_only_pre_s,
            post_s=wrist_only_post_s,
            min_separation_s=wrist_only_min_separation_s,
            confidence_cap=wrist_only_confidence_cap,
        )

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
        frame = _event_frame(event_t, fps=fps, frame_times=frame_times)

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
                trust_band_note=_ball_contact_trust_note(ball),
            )
        )
        used_ball_indices.add(ball_idx)
        used_wrist_indices.add(wrist_idx)

    events.sort(key=lambda event: (float(event["t"]), int(event["frame"])))
    if events or not allow_wrist_only_contact_hints:
        return build_contact_windows_artifact(events)
    return _wrist_only_contact_hints(
        fps=fps,
        frame_times=frame_times,
        wrists=sorted_wrist,
        pre_s=wrist_only_pre_s,
        post_s=wrist_only_post_s,
        min_separation_s=wrist_only_min_separation_s,
        confidence_cap=wrist_only_confidence_cap,
    )


def fuse_contact_windows_from_cue_payloads(
    *,
    fps: float,
    frame_times: Any = None,
    audio_onsets_payload: Any,
    wrist_velocity_peaks_payload: Any,
    ball_inflections_payload: Any,
    require_audio: bool = True,
    max_time_delta_s: float = DEFAULT_MAX_TIME_DELTA_S,
    pre_s: float = DEFAULT_PRE_WINDOW_S,
    post_s: float = DEFAULT_POST_WINDOW_S,
    allow_wrist_only_contact_hints: bool = False,
    wrist_only_pre_s: float = DEFAULT_WRIST_ONLY_PRE_WINDOW_S,
    wrist_only_post_s: float = DEFAULT_WRIST_ONLY_POST_WINDOW_S,
    wrist_only_min_separation_s: float = DEFAULT_WRIST_ONLY_MIN_SEPARATION_S,
    wrist_only_confidence_cap: float = DEFAULT_WRIST_ONLY_CONFIDENCE_CAP,
) -> dict[str, object]:
    """Fuse canonical cue artifact payloads into a ContactWindows artifact."""

    audio_onsets = _items(audio_onsets_payload, keys=("onsets", "audio_onsets", "items"))
    wrist_velocity_peaks = [
        WristVelocityPeak(
            time_s=_cue_time_s(item, fps=fps, frame_times=frame_times, name="wrist_velocity_peak.time_s"),
            player_id=int(item["player_id"]),
            wrist_world_xyz=tuple(item["wrist_world_xyz"]),
            speed_mps=float(item["speed_mps"]),
            confidence=float(item["confidence"]),
        )
        for item in _items(wrist_velocity_peaks_payload, keys=("peaks", "wrist_velocity_peaks", "items"))
    ]
    ball_inflections = [
        BallInflectionCandidate(
            time_s=_cue_time_s(item, fps=fps, frame_times=frame_times, name="ball_inflection.time_s"),
            ball_world_xyz=_optional_vector3(item.get("ball_world_xyz")),
            confidence=float(item["confidence"]),
        )
        for item in _items(ball_inflections_payload, keys=("candidates", "ball_inflections", "items"))
    ]
    return fuse_contact_windows(
        fps=fps,
        frame_times=frame_times,
        audio_onsets=audio_onsets,
        wrist_velocity_peaks=wrist_velocity_peaks,
        ball_inflections=ball_inflections,
        require_audio=require_audio,
        max_time_delta_s=max_time_delta_s,
        pre_s=pre_s,
        post_s=post_s,
        allow_wrist_only_contact_hints=allow_wrist_only_contact_hints,
        wrist_only_pre_s=wrist_only_pre_s,
        wrist_only_post_s=wrist_only_post_s,
        wrist_only_min_separation_s=wrist_only_min_separation_s,
        wrist_only_confidence_cap=wrist_only_confidence_cap,
    )


def fuse_contact_windows_from_cue_files(
    *,
    fps: float,
    frame_times: Any = None,
    audio_onsets_path: str | Path | None,
    wrist_velocity_peaks_path: str | Path,
    ball_inflections_path: str | Path | None,
    require_audio: bool = True,
    max_time_delta_s: float = DEFAULT_MAX_TIME_DELTA_S,
    pre_s: float = DEFAULT_PRE_WINDOW_S,
    post_s: float = DEFAULT_POST_WINDOW_S,
    allow_wrist_only_contact_hints: bool = False,
    wrist_only_pre_s: float = DEFAULT_WRIST_ONLY_PRE_WINDOW_S,
    wrist_only_post_s: float = DEFAULT_WRIST_ONLY_POST_WINDOW_S,
    wrist_only_min_separation_s: float = DEFAULT_WRIST_ONLY_MIN_SEPARATION_S,
    wrist_only_confidence_cap: float = DEFAULT_WRIST_ONLY_CONFIDENCE_CAP,
) -> dict[str, object]:
    """Read cue artifact files and fuse them into a ContactWindows artifact."""

    return fuse_contact_windows_from_cue_payloads(
        fps=fps,
        frame_times=frame_times,
        audio_onsets_payload=_read_json(Path(audio_onsets_path)) if audio_onsets_path is not None else [],
        wrist_velocity_peaks_payload=_read_json(Path(wrist_velocity_peaks_path)),
        ball_inflections_payload=_read_json(Path(ball_inflections_path)) if ball_inflections_path is not None else [],
        require_audio=require_audio,
        max_time_delta_s=max_time_delta_s,
        pre_s=pre_s,
        post_s=post_s,
        allow_wrist_only_contact_hints=allow_wrist_only_contact_hints,
        wrist_only_pre_s=wrist_only_pre_s,
        wrist_only_post_s=wrist_only_post_s,
        wrist_only_min_separation_s=wrist_only_min_separation_s,
        wrist_only_confidence_cap=wrist_only_confidence_cap,
    )


def _wrist_only_contact_hints(
    *,
    fps: float,
    frame_times: Any,
    wrists: Sequence[WristVelocityPeak],
    pre_s: float,
    post_s: float,
    min_separation_s: float,
    confidence_cap: float,
) -> dict[str, object]:
    events: list[dict[str, object]] = []
    for wrist in _suppress_wrist_only_peaks(wrists, min_separation_s=min_separation_s):
        confidence = round(min(confidence_cap, wrist.confidence), 12)
        event_t = wrist.time_s
        events.append(
            build_contact_event(
                t=event_t,
                frame=_event_frame(event_t, fps=fps, frame_times=frame_times),
                player_id=wrist.player_id,
                confidence=confidence,
                sources={"wrist_vel": wrist.confidence, "ball_inflection": 0.0},
                t0=max(0.0, event_t - pre_s),
                t1=event_t + post_s,
                importance=confidence,
                trust_band_note=WRIST_ONLY_TRUST_BAND_NOTE,
            )
        )
    events.sort(key=lambda event: (float(event["t"]), int(event["player_id"] or 0), int(event["frame"])))
    return build_contact_windows_artifact(events)


def _suppress_wrist_only_peaks(
    wrists: Sequence[WristVelocityPeak],
    *,
    min_separation_s: float,
) -> list[WristVelocityPeak]:
    grouped: dict[int, list[WristVelocityPeak]] = {}
    for wrist in wrists:
        grouped.setdefault(wrist.player_id, []).append(wrist)

    kept: list[WristVelocityPeak] = []
    for group in grouped.values():
        current: list[WristVelocityPeak] = []
        for wrist in sorted(group, key=lambda item: item.time_s):
            if not current or wrist.time_s - current[-1].time_s >= min_separation_s:
                current.append(wrist)
                continue
            if _wrist_rank(wrist) > _wrist_rank(current[-1]):
                current[-1] = wrist
        kept.extend(current)
    return sorted(kept, key=lambda item: (item.time_s, item.player_id))


def _wrist_rank(wrist: WristVelocityPeak) -> tuple[float, float]:
    return wrist.speed_mps, wrist.confidence


def _coerce_audio_onset(
    candidate: AudioOnsetCandidate | Mapping[str, Any] | Any,
    *,
    fps: float,
    frame_times: Any,
) -> AudioOnsetCandidate:
    if isinstance(candidate, AudioOnsetCandidate):
        return candidate
    if isinstance(candidate, Mapping):
        corrected_time_s = candidate.get("corrected_time_s")
        time_s = (
            _require_non_negative_time(corrected_time_s, "audio_onset.corrected_time_s")
            if corrected_time_s is not None
            else _cue_time_s(candidate, fps=fps, frame_times=frame_times, name="audio_onset.time_s")
        )
        confidence = candidate.get("confidence", candidate.get("score", candidate.get("conf")))
        features = candidate.get("features")
    else:
        time_s = getattr(candidate, "time_s", getattr(candidate, "t", None))
        confidence = getattr(candidate, "confidence", getattr(candidate, "score", getattr(candidate, "conf", None)))
        features = getattr(candidate, "features", None)
    return AudioOnsetCandidate(
        time_s=_require_non_negative_time(time_s, "audio_onset.time_s"),
        confidence=_require_confidence(confidence, "audio_onset.confidence"),
        features=features,
    )


def _apply_audio_pop_soft_evidence(
    *,
    confidence: float,
    pre_s: float,
    post_s: float,
    audio: AudioOnsetCandidate,
) -> tuple[float, float, float]:
    """Apply bounded advisory pop evidence without making audio a gate.

    The legacy visual-plus-onset confidence is treated as prior odds. A raw
    ``pop_band_ratio`` is converted to a bounded likelihood ratio in
    ``[exp(-0.2), exp(0.2)]``; it is not averaged with cue confidences. The
    same centered evidence scales each window side by at most ten percent.
    Missing features return the exact legacy values for byte parity.
    """

    if audio.features is None or "pop_band_ratio" not in audio.features:
        return confidence, pre_s, post_s
    raw_ratio = _require_finite(audio.features["pop_band_ratio"], "audio_onset.features.pop_band_ratio")
    bounded_ratio = min(1.0, max(0.0, raw_ratio))
    centered = 2.0 * bounded_ratio - 1.0
    if confidence <= 0.0 or confidence >= 1.0:
        adjusted_confidence = confidence
    else:
        prior_odds = confidence / (1.0 - confidence)
        adjusted_odds = prior_odds * math.exp(POP_LIKELIHOOD_LOG_BOUND * centered)
        adjusted_confidence = adjusted_odds / (1.0 + adjusted_odds)
    window_scale = 1.0 - POP_WINDOW_SCALE_BOUND * centered
    return (
        round(adjusted_confidence, 12),
        pre_s * window_scale,
        post_s * window_scale,
    )


def _cue_time_s(item: Mapping[str, Any], *, fps: float, frame_times: Any, name: str) -> float:
    time_s = item.get("time_s", item.get("t", item.get("time")))
    if time_s is not None:
        return _require_non_negative_time(time_s, name)
    frame = item.get("frame", item.get("frame_index", item.get("frame_idx")))
    if frame is None:
        raise ValueError(f"{name} is required")
    return _require_non_negative_time(
        time_for_frame(int(frame), frame_times=frame_times, fps=fps),
        name,
    )


def _event_frame(event_t: float, *, fps: float, frame_times: Any) -> int:
    return max(0, nearest_frame_for_time(float(event_t), frame_times=frame_times, fps=fps))


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


def _nearest_audio_to_visual_match(
    *,
    ball: BallInflectionCandidate,
    wrist: WristVelocityPeak,
    candidates: Sequence[AudioOnsetCandidate],
    used_indices: set[int],
    max_time_delta_s: float,
) -> tuple[int, AudioOnsetCandidate] | None:
    center_t = (ball.time_s + wrist.time_s) / 2.0
    matches = [
        (idx, candidate)
        for idx, candidate in enumerate(candidates)
        if idx not in used_indices
        and abs(candidate.time_s - ball.time_s) <= max_time_delta_s
        and abs(candidate.time_s - wrist.time_s) <= max_time_delta_s
    ]
    if not matches:
        return None
    return min(
        matches,
        key=lambda item: (
            abs(item[1].time_s - center_t),
            -item[1].confidence,
            item[0],
        ),
    )


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
            _distance_to_ball(item[1], ball),
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
            _distance_to_ball(item[1], ball),
            abs(item[1].time_s - ball.time_s),
            -item[1].confidence,
            item[1].player_id,
            item[0],
        ),
    )


def _distance_to_ball(wrist: WristVelocityPeak, ball: BallInflectionCandidate) -> float:
    if ball.ball_world_xyz is None:
        return math.inf
    return _distance_m(wrist.wrist_world_xyz, ball.ball_world_xyz)


def _ball_contact_trust_note(ball: BallInflectionCandidate) -> str | None:
    if ball.ball_world_xyz is None:
        return IMAGE_SPACE_BALL_INFLECTION_NOTE
    if ball.confidence < LOW_TRUST_BALL_INFLECTION_CONFIDENCE_THRESHOLD:
        return LOW_TRUST_BALL_INFLECTION_NOTE
    return None


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


def _optional_vector3(value: Sequence[float] | None) -> tuple[float, float, float] | None:
    if value is None:
        return None
    return _require_vector3(value, "ball_world_xyz")


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
    "DEFAULT_WRIST_ONLY_CONFIDENCE_CAP",
    "DEFAULT_WRIST_ONLY_MIN_SEPARATION_S",
    "DEFAULT_WRIST_ONLY_POST_WINDOW_S",
    "DEFAULT_WRIST_ONLY_PRE_WINDOW_S",
    "IMAGE_SPACE_BALL_INFLECTION_NOTE",
    "LOW_TRUST_BALL_INFLECTION_CONFIDENCE_THRESHOLD",
    "LOW_TRUST_BALL_INFLECTION_NOTE",
    "WristVelocityPeak",
    "WRIST_ONLY_TRUST_BAND_NOTE",
    "fuse_contact_windows",
    "fuse_contact_windows_from_cue_files",
    "fuse_contact_windows_from_cue_payloads",
]

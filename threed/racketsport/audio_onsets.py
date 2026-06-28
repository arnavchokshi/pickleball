"""CPU-only audio onset cue generation for contact review.

These cues are support signals for event fusion. They are not audio-pop
classifier outputs and are not gate-verified contact events.
"""

from __future__ import annotations

import json
import math
import struct
import subprocess
import wave
from pathlib import Path
from typing import Any, Mapping, Sequence


ARTIFACT_TYPE = "racketsport_audio_onsets"
DEFAULT_SAMPLE_RATE_HZ = 16_000
DEFAULT_FRAME_SIZE_S = 0.020
DEFAULT_HOP_S = 0.005
DEFAULT_THRESHOLD_SCORE = 0.55
DEFAULT_MIN_SEPARATION_S = 0.080


def build_audio_onsets_from_wav(
    wav_path: str | Path,
    *,
    threshold_score: float = DEFAULT_THRESHOLD_SCORE,
    frame_size_s: float = DEFAULT_FRAME_SIZE_S,
    hop_s: float = DEFAULT_HOP_S,
    min_separation_s: float = DEFAULT_MIN_SEPARATION_S,
    clip: str | None = None,
    frame_rate: float | None = None,
    analysis_sample_rate_hz: int | None = None,
) -> dict[str, Any]:
    """Build review-only onset cues from a PCM WAV file."""

    path = Path(wav_path)
    samples, sample_rate_hz = _read_wav_mono(path)
    return build_audio_onsets_from_samples(
        samples,
        sample_rate_hz=sample_rate_hz,
        source="wav_audio_energy",
        source_path=path,
        threshold_score=threshold_score,
        frame_size_s=frame_size_s,
        hop_s=hop_s,
        min_separation_s=min_separation_s,
        clip=clip,
        frame_rate=frame_rate,
        analysis_sample_rate_hz=analysis_sample_rate_hz,
    )


def build_audio_onsets_from_video(
    video_path: str | Path,
    *,
    sample_rate_hz: int = DEFAULT_SAMPLE_RATE_HZ,
    analysis_sample_rate_hz: int | None = None,
    threshold_score: float = DEFAULT_THRESHOLD_SCORE,
    frame_size_s: float = DEFAULT_FRAME_SIZE_S,
    hop_s: float = DEFAULT_HOP_S,
    min_separation_s: float = DEFAULT_MIN_SEPARATION_S,
    start_s: float = 0.0,
    duration_s: float | None = None,
    clip: str | None = None,
    frame_rate: float | None = None,
) -> dict[str, Any]:
    """Build onset cues from a video's first audio stream, or an explicit blocker."""

    path = Path(video_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    start_s = _require_non_negative(start_s, "start_s")
    if duration_s is not None:
        duration_s = _require_positive(duration_s, "duration_s")
    audio_stream = _ffprobe_audio_stream(path)
    if audio_stream is None:
        return _artifact(
            status="blocked",
            source="video_audio_stream",
            source_path=path,
            sample_rate_hz=None,
            duration_s=0.0,
            raw_peak_count=0,
            onsets=[],
            blockers=["no_audio_stream"],
            warnings=["audio_stream_missing"],
            clip=clip,
            frame_rate=frame_rate,
            thresholds={
                "threshold_score": threshold_score,
                "frame_size_s": frame_size_s,
                "hop_s": hop_s,
                "min_separation_s": min_separation_s,
                "start_s": start_s,
                "duration_limit_s": duration_s,
                "analysis_sample_rate_hz": analysis_sample_rate_hz or sample_rate_hz,
            },
        )

    samples = _decode_video_audio_mono(path, sample_rate_hz=sample_rate_hz, start_s=start_s, duration_s=duration_s)
    return build_audio_onsets_from_samples(
        samples,
        sample_rate_hz=sample_rate_hz,
        source="video_audio_energy",
        source_path=path,
        threshold_score=threshold_score,
        frame_size_s=frame_size_s,
        hop_s=hop_s,
        min_separation_s=min_separation_s,
        clip=clip,
        frame_rate=frame_rate,
        analysis_sample_rate_hz=analysis_sample_rate_hz or sample_rate_hz,
        source_metadata={
            "codec_name": audio_stream.get("codec_name"),
            "source_sample_rate_hz": _optional_int(audio_stream.get("sample_rate")),
            "channels": _optional_int(audio_stream.get("channels")),
            "window_start_s": start_s,
            "window_duration_s": duration_s,
        },
    )


def build_audio_onsets_from_samples(
    samples: Sequence[float],
    *,
    sample_rate_hz: int,
    source: str,
    source_path: str | Path,
    threshold_score: float = DEFAULT_THRESHOLD_SCORE,
    frame_size_s: float = DEFAULT_FRAME_SIZE_S,
    hop_s: float = DEFAULT_HOP_S,
    min_separation_s: float = DEFAULT_MIN_SEPARATION_S,
    source_metadata: Mapping[str, Any] | None = None,
    clip: str | None = None,
    frame_rate: float | None = None,
    analysis_sample_rate_hz: int | None = None,
) -> dict[str, Any]:
    """Build onset cues from mono floating-point samples in [-1, 1]."""

    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive")
    threshold_score = _require_unit(threshold_score, "threshold_score")
    frame_size_s = _require_positive(frame_size_s, "frame_size_s")
    hop_s = _require_positive(hop_s, "hop_s")
    min_separation_s = _require_non_negative(min_separation_s, "min_separation_s")
    if not source:
        raise ValueError("source must be non-empty")

    frame_size = max(1, int(round(frame_size_s * sample_rate_hz)))
    hop = max(1, int(round(hop_s * sample_rate_hz)))
    duration_s = len(samples) / sample_rate_hz
    energies: list[tuple[int, float]] = []
    for start in range(0, max(0, len(samples) - frame_size + 1), hop):
        window = samples[start : start + frame_size]
        rms = math.sqrt(sum(float(sample) ** 2 for sample in window) / len(window))
        energies.append((start, rms))

    if len(energies) < 2:
        return _artifact(
            status="blocked",
            source=source,
            source_path=source_path,
            sample_rate_hz=sample_rate_hz,
            duration_s=duration_s,
            raw_peak_count=0,
            onsets=[],
            blockers=["insufficient_audio_samples"],
            warnings=["insufficient_audio_samples"],
            clip=clip,
            frame_rate=frame_rate,
            thresholds={
                "threshold_score": threshold_score,
                "frame_size_s": frame_size_s,
                "hop_s": hop_s,
                "min_separation_s": min_separation_s,
                "analysis_sample_rate_hz": analysis_sample_rate_hz or sample_rate_hz,
            },
            source_metadata=source_metadata,
        )

    deltas = [max(0.0, energies[idx][1] - energies[idx - 1][1]) for idx in range(1, len(energies))]
    max_delta = max(deltas) if deltas else 0.0
    raw: list[dict[str, Any]] = []
    if max_delta > 0.0:
        for idx, delta in enumerate(deltas, start=1):
            score = delta / max_delta
            left = deltas[idx - 2] / max_delta if idx >= 2 else -1.0
            right = deltas[idx] / max_delta if idx < len(deltas) else -1.0
            if score < threshold_score or score < left or score < right:
                continue
            start = energies[idx][0]
            onset_sample = min(len(samples), start + frame_size - hop)
            raw.append(
                {
                    "time_s": round(onset_sample / sample_rate_hz, 9),
                    "raw_time_s": round(onset_sample / sample_rate_hz, 9),
                    "score": round(score, 6),
                    "source": "audio_energy_onset",
                    "window_start_s": round(start / sample_rate_hz, 9),
                    "window_end_s": round((start + frame_size) / sample_rate_hz, 9),
                }
            )
    onsets = _suppress_nearby_onsets(raw, min_separation_s=min_separation_s)
    blockers = [] if onsets else ["no_audio_energy_peaks"]
    warnings = ["energy_heuristic_not_classifier"]
    if blockers:
        warnings.extend(blockers)
    return _artifact(
        status="review_only" if onsets else "blocked",
        source=source,
        source_path=source_path,
        sample_rate_hz=sample_rate_hz,
        duration_s=duration_s,
        raw_peak_count=len(raw),
        onsets=onsets,
        blockers=blockers,
        warnings=warnings,
        clip=clip,
        frame_rate=frame_rate,
        thresholds={
            "threshold_score": threshold_score,
            "frame_size_s": frame_size_s,
            "hop_s": hop_s,
            "min_separation_s": min_separation_s,
            "analysis_sample_rate_hz": analysis_sample_rate_hz or sample_rate_hz,
        },
        source_metadata=source_metadata,
    )


def write_audio_onsets(path: str | Path, payload: Mapping[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _artifact(
    *,
    status: str,
    source: str,
    source_path: str | Path,
    sample_rate_hz: int | None,
    duration_s: float,
    raw_peak_count: int,
    onsets: list[dict[str, Any]],
    blockers: list[str],
    warnings: list[str],
    thresholds: Mapping[str, Any],
    source_metadata: Mapping[str, Any] | None = None,
    clip: str | None = None,
    frame_rate: float | None = None,
) -> dict[str, Any]:
    enriched_onsets = _attach_nearest_frames(onsets, frame_rate=frame_rate)
    return {
        "schema_version": 1,
        "artifact_type": ARTIFACT_TYPE,
        "status": status,
        **({"clip": clip} if clip else {}),
        **({"frame_rate": float(frame_rate)} if frame_rate is not None else {}),
        "source": source,
        "source_path": str(source_path),
        "sample_rate_hz": sample_rate_hz,
        "not_gate_verified": True,
        "trusted_for_contact": False,
        "blockers": blockers,
        "warnings": warnings,
        "summary": {
            "duration_s": round(duration_s, 9),
            "raw_peak_count": raw_peak_count,
            "onset_count": len(enriched_onsets),
            **dict(thresholds),
        },
        "source_metadata": dict(source_metadata or {}),
        "onsets": enriched_onsets,
    }


def _attach_nearest_frames(onsets: list[dict[str, Any]], *, frame_rate: float | None) -> list[dict[str, Any]]:
    if frame_rate is None:
        return onsets
    if frame_rate <= 0:
        raise ValueError("frame_rate must be positive")
    enriched: list[dict[str, Any]] = []
    for onset in onsets:
        item = dict(onset)
        item["nearest_frame"] = int(round(float(item["time_s"]) * frame_rate))
        enriched.append(item)
    return enriched


def _suppress_nearby_onsets(candidates: list[dict[str, Any]], *, min_separation_s: float) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for candidate in sorted(candidates, key=lambda item: (float(item["time_s"]), -float(item["score"]))):
        if not kept or float(candidate["time_s"]) - float(kept[-1]["time_s"]) >= min_separation_s:
            kept.append(candidate)
            continue
        if float(candidate["score"]) > float(kept[-1]["score"]):
            kept[-1] = candidate
    return kept


def _read_wav_mono(path: Path) -> tuple[list[float], int]:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate_hz = wav.getframerate()
        frame_count = wav.getnframes()
        raw = wav.readframes(frame_count)
    if channels <= 0:
        raise ValueError("wav must have at least one channel")
    if sample_width not in {1, 2, 4}:
        raise ValueError("wav sample width must be 8, 16, or 32 bits")
    samples = _decode_pcm(raw, sample_width=sample_width)
    mono: list[float] = []
    for index in range(0, len(samples), channels):
        mono.append(sum(samples[index : index + channels]) / channels)
    return mono, sample_rate_hz


def _decode_pcm(raw: bytes, *, sample_width: int) -> list[float]:
    if sample_width == 1:
        return [(byte - 128) / 128.0 for byte in raw]
    if sample_width == 2:
        count = len(raw) // 2
        return [value / 32768.0 for value in struct.unpack(f"<{count}h", raw[: count * 2])]
    count = len(raw) // 4
    return [value / 2147483648.0 for value in struct.unpack(f"<{count}i", raw[: count * 4])]


def _ffprobe_audio_stream(path: Path) -> Mapping[str, Any] | None:
    completed = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    for stream in payload.get("streams", []):
        if isinstance(stream, Mapping) and stream.get("codec_type") == "audio":
            return stream
    return None


def _decode_video_audio_mono(path: Path, *, sample_rate_hz: int, start_s: float, duration_s: float | None) -> list[float]:
    command = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error"]
    if start_s > 0.0:
        command.extend(["-ss", str(start_s)])
    command.extend(["-i", str(path)])
    if duration_s is not None:
        command.extend(["-t", str(duration_s)])
    command.extend(
        [
            "-map",
            "0:a:0",
            "-ac",
            "1",
            "-ar",
            str(sample_rate_hz),
            "-f",
            "s16le",
            "-",
        ]
    )
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
    )
    return _decode_pcm(completed.stdout, sample_width=2)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _require_unit(value: float, name: str) -> float:
    value = _require_finite(value, name)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return value


def _require_positive(value: float, name: str) -> float:
    value = _require_finite(value, name)
    if value <= 0.0:
        raise ValueError(f"{name} must be positive")
    return value


def _require_non_negative(value: float, name: str) -> float:
    value = _require_finite(value, name)
    if value < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _require_finite(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


__all__ = [
    "ARTIFACT_TYPE",
    "build_audio_onsets_from_samples",
    "build_audio_onsets_from_video",
    "build_audio_onsets_from_wav",
    "write_audio_onsets",
]

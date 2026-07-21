"""CPU-only v2 audio-pop onset cue generation for contact review.

This module is intentionally separate from ``audio_onsets.py`` so the older
energy heuristic remains unchanged while M4 audio diagnostics can compare a
pop-tuned detector.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import ndimage, signal

from threed.racketsport.audio_onsets import (
    ARTIFACT_TYPE,
    DEFAULT_SPEED_OF_SOUND_MPS,
    _decode_video_audio_mono,
    _ffprobe_audio_stream,
    _optional_int,
    _read_wav_mono,
    finalize_audio_onset_timing,
)


DETECTOR_VERSION = "audio_onset_pop_v2"
DEFAULT_ANALYSIS_SAMPLE_RATE_HZ = 24_000
DEFAULT_BANDPASS_LOW_HZ = 1_000.0
DEFAULT_BANDPASS_HIGH_HZ = 6_000.0
DEFAULT_FRAME_SIZE_S = 0.006
DEFAULT_HOP_S = 0.001
DEFAULT_MIN_SEPARATION_S = 0.080
DEFAULT_THRESHOLD_MAD = 4.0
DEFAULT_ADAPTIVE_WINDOW_S = 0.500
DEFAULT_MIN_POP_BAND_RATIO = 0.10
DEFAULT_MIN_SPECTRAL_EVIDENCE = 0.5
DEFAULT_MIN_HFC_EVIDENCE = 0.7


def build_audio_onsets_v2_from_wav(
    wav_path: str | Path,
    *,
    analysis_sample_rate_hz: int = DEFAULT_ANALYSIS_SAMPLE_RATE_HZ,
    bandpass_low_hz: float = DEFAULT_BANDPASS_LOW_HZ,
    bandpass_high_hz: float = DEFAULT_BANDPASS_HIGH_HZ,
    frame_size_s: float = DEFAULT_FRAME_SIZE_S,
    hop_s: float = DEFAULT_HOP_S,
    min_separation_s: float = DEFAULT_MIN_SEPARATION_S,
    threshold_mad: float = DEFAULT_THRESHOLD_MAD,
    adaptive_window_s: float = DEFAULT_ADAPTIVE_WINDOW_S,
    min_pop_band_ratio: float = DEFAULT_MIN_POP_BAND_RATIO,
    min_spectral_evidence: float = DEFAULT_MIN_SPECTRAL_EVIDENCE,
    min_hfc_evidence: float = DEFAULT_MIN_HFC_EVIDENCE,
    clip: str | None = None,
    frame_rate: float | None = None,
    source_to_microphone_distance_m: float | None = None,
    distance_uncertainty_m: float = 0.0,
    speed_of_sound_mps: float = DEFAULT_SPEED_OF_SOUND_MPS,
    media_sha256: str | None = None,
    pts_source: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build v2 review-only onset cues from a PCM WAV file."""

    path = Path(wav_path)
    samples, sample_rate_hz = _read_wav_mono(path)
    return build_audio_onsets_v2_from_samples(
        samples,
        sample_rate_hz=sample_rate_hz,
        source="wav_audio_pop_v2",
        source_path=path,
        analysis_sample_rate_hz=analysis_sample_rate_hz,
        bandpass_low_hz=bandpass_low_hz,
        bandpass_high_hz=bandpass_high_hz,
        frame_size_s=frame_size_s,
        hop_s=hop_s,
        min_separation_s=min_separation_s,
        threshold_mad=threshold_mad,
        adaptive_window_s=adaptive_window_s,
        min_pop_band_ratio=min_pop_band_ratio,
        min_spectral_evidence=min_spectral_evidence,
        min_hfc_evidence=min_hfc_evidence,
        clip=clip,
        frame_rate=frame_rate,
        source_to_microphone_distance_m=source_to_microphone_distance_m,
        distance_uncertainty_m=distance_uncertainty_m,
        speed_of_sound_mps=speed_of_sound_mps,
        media_sha256=media_sha256,
        pts_source=pts_source,
    )


def build_audio_onsets_v2_from_video(
    video_path: str | Path,
    *,
    analysis_sample_rate_hz: int = DEFAULT_ANALYSIS_SAMPLE_RATE_HZ,
    bandpass_low_hz: float = DEFAULT_BANDPASS_LOW_HZ,
    bandpass_high_hz: float = DEFAULT_BANDPASS_HIGH_HZ,
    frame_size_s: float = DEFAULT_FRAME_SIZE_S,
    hop_s: float = DEFAULT_HOP_S,
    min_separation_s: float = DEFAULT_MIN_SEPARATION_S,
    threshold_mad: float = DEFAULT_THRESHOLD_MAD,
    adaptive_window_s: float = DEFAULT_ADAPTIVE_WINDOW_S,
    min_pop_band_ratio: float = DEFAULT_MIN_POP_BAND_RATIO,
    min_spectral_evidence: float = DEFAULT_MIN_SPECTRAL_EVIDENCE,
    min_hfc_evidence: float = DEFAULT_MIN_HFC_EVIDENCE,
    start_s: float = 0.0,
    duration_s: float | None = None,
    clip: str | None = None,
    frame_rate: float | None = None,
    source_to_microphone_distance_m: float | None = None,
    distance_uncertainty_m: float = 0.0,
    speed_of_sound_mps: float = DEFAULT_SPEED_OF_SOUND_MPS,
    media_sha256: str | None = None,
    pts_source: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build v2 onset cues from a video's first audio stream, or a blocker."""

    path = Path(video_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    start_s = _require_non_negative(start_s, "start_s")
    if duration_s is not None:
        duration_s = _require_positive(duration_s, "duration_s")
    audio_stream = _ffprobe_audio_stream(path)
    thresholds = _threshold_summary(
        analysis_sample_rate_hz=analysis_sample_rate_hz,
        bandpass_low_hz=bandpass_low_hz,
        bandpass_high_hz=bandpass_high_hz,
        frame_size_s=frame_size_s,
        hop_s=hop_s,
        min_separation_s=min_separation_s,
        threshold_mad=threshold_mad,
        adaptive_window_s=adaptive_window_s,
        min_pop_band_ratio=min_pop_band_ratio,
        min_spectral_evidence=min_spectral_evidence,
        min_hfc_evidence=min_hfc_evidence,
        start_s=start_s,
        duration_limit_s=duration_s,
    )
    if audio_stream is None:
        return _artifact(
            status="blocked",
            source="video_audio_pop_v2",
            source_path=path,
            sample_rate_hz=None,
            duration_s=0.0,
            raw_peak_count=0,
            onsets=[],
            blockers=["no_audio_stream"],
            warnings=["audio_stream_missing"],
            thresholds=thresholds,
            clip=clip,
            frame_rate=frame_rate,
            source_to_microphone_distance_m=source_to_microphone_distance_m,
            distance_uncertainty_m=distance_uncertainty_m,
            speed_of_sound_mps=speed_of_sound_mps,
            media_sha256=media_sha256,
            pts_source=pts_source,
        )

    samples = _decode_video_audio_mono(
        path,
        sample_rate_hz=analysis_sample_rate_hz,
        start_s=start_s,
        duration_s=duration_s,
    )
    return build_audio_onsets_v2_from_samples(
        samples,
        sample_rate_hz=analysis_sample_rate_hz,
        source="video_audio_pop_v2",
        source_path=path,
        analysis_sample_rate_hz=analysis_sample_rate_hz,
        bandpass_low_hz=bandpass_low_hz,
        bandpass_high_hz=bandpass_high_hz,
        frame_size_s=frame_size_s,
        hop_s=hop_s,
        min_separation_s=min_separation_s,
        threshold_mad=threshold_mad,
        adaptive_window_s=adaptive_window_s,
        min_pop_band_ratio=min_pop_band_ratio,
        min_spectral_evidence=min_spectral_evidence,
        min_hfc_evidence=min_hfc_evidence,
        time_offset_s=start_s,
        source_metadata={
            "codec_name": audio_stream.get("codec_name"),
            "source_sample_rate_hz": _optional_int(audio_stream.get("sample_rate")),
            "channels": _optional_int(audio_stream.get("channels")),
            "window_start_s": start_s,
            "window_duration_s": duration_s,
        },
        clip=clip,
        frame_rate=frame_rate,
        source_to_microphone_distance_m=source_to_microphone_distance_m,
        distance_uncertainty_m=distance_uncertainty_m,
        speed_of_sound_mps=speed_of_sound_mps,
        media_sha256=media_sha256,
        pts_source=pts_source,
    )


def build_audio_onsets_v2_from_samples(
    samples: Sequence[float],
    *,
    sample_rate_hz: int,
    source: str,
    source_path: str | Path,
    analysis_sample_rate_hz: int | None = None,
    bandpass_low_hz: float = DEFAULT_BANDPASS_LOW_HZ,
    bandpass_high_hz: float = DEFAULT_BANDPASS_HIGH_HZ,
    frame_size_s: float = DEFAULT_FRAME_SIZE_S,
    hop_s: float = DEFAULT_HOP_S,
    min_separation_s: float = DEFAULT_MIN_SEPARATION_S,
    threshold_mad: float = DEFAULT_THRESHOLD_MAD,
    adaptive_window_s: float = DEFAULT_ADAPTIVE_WINDOW_S,
    min_pop_band_ratio: float = DEFAULT_MIN_POP_BAND_RATIO,
    min_spectral_evidence: float = DEFAULT_MIN_SPECTRAL_EVIDENCE,
    min_hfc_evidence: float = DEFAULT_MIN_HFC_EVIDENCE,
    time_offset_s: float = 0.0,
    source_metadata: Mapping[str, Any] | None = None,
    clip: str | None = None,
    frame_rate: float | None = None,
    source_to_microphone_distance_m: float | None = None,
    distance_uncertainty_m: float = 0.0,
    speed_of_sound_mps: float = DEFAULT_SPEED_OF_SOUND_MPS,
    media_sha256: str | None = None,
    pts_source: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build v2 onset cues from mono floating-point samples in [-1, 1]."""

    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive")
    if analysis_sample_rate_hz is not None and analysis_sample_rate_hz <= 0:
        raise ValueError("analysis_sample_rate_hz must be positive")
    if not source:
        raise ValueError("source must be non-empty")
    bandpass_low_hz = _require_positive(bandpass_low_hz, "bandpass_low_hz")
    bandpass_high_hz = _require_positive(bandpass_high_hz, "bandpass_high_hz")
    frame_size_s = _require_positive(frame_size_s, "frame_size_s")
    hop_s = _require_positive(hop_s, "hop_s")
    min_separation_s = _require_non_negative(min_separation_s, "min_separation_s")
    threshold_mad = _require_positive(threshold_mad, "threshold_mad")
    adaptive_window_s = _require_positive(adaptive_window_s, "adaptive_window_s")
    min_pop_band_ratio = _require_unit(min_pop_band_ratio, "min_pop_band_ratio")
    min_spectral_evidence = _require_non_negative(min_spectral_evidence, "min_spectral_evidence")
    min_hfc_evidence = _require_non_negative(min_hfc_evidence, "min_hfc_evidence")
    time_offset_s = _require_non_negative(time_offset_s, "time_offset_s")

    target_sample_rate_hz = analysis_sample_rate_hz or sample_rate_hz
    sample_array = _as_float_array(samples)
    if target_sample_rate_hz != sample_rate_hz:
        sample_array = _resample(sample_array, source_rate_hz=sample_rate_hz, target_rate_hz=target_sample_rate_hz)
    duration_s = len(sample_array) / target_sample_rate_hz
    thresholds = _threshold_summary(
        analysis_sample_rate_hz=target_sample_rate_hz,
        bandpass_low_hz=bandpass_low_hz,
        bandpass_high_hz=bandpass_high_hz,
        frame_size_s=frame_size_s,
        hop_s=hop_s,
        min_separation_s=min_separation_s,
        threshold_mad=threshold_mad,
        adaptive_window_s=adaptive_window_s,
        min_pop_band_ratio=min_pop_band_ratio,
        min_spectral_evidence=min_spectral_evidence,
        min_hfc_evidence=min_hfc_evidence,
        start_s=time_offset_s,
        duration_limit_s=None,
    )

    onsets, raw_peak_count = _detect_onsets(
        sample_array,
        sample_rate_hz=target_sample_rate_hz,
        bandpass_low_hz=bandpass_low_hz,
        bandpass_high_hz=bandpass_high_hz,
        frame_size_s=frame_size_s,
        hop_s=hop_s,
        min_separation_s=min_separation_s,
        threshold_mad=threshold_mad,
        adaptive_window_s=adaptive_window_s,
        min_pop_band_ratio=min_pop_band_ratio,
        min_spectral_evidence=min_spectral_evidence,
        min_hfc_evidence=min_hfc_evidence,
        time_offset_s=time_offset_s,
    )
    blockers = [] if onsets else ["no_audio_pop_v2_peaks"]
    warnings = ["pop_transient_heuristic_not_classifier", "cue_not_gate_verified"]
    if blockers:
        warnings.extend(blockers)

    return _artifact(
        status="review_only" if onsets else "blocked",
        source=source,
        source_path=source_path,
        sample_rate_hz=target_sample_rate_hz,
        duration_s=duration_s,
        raw_peak_count=raw_peak_count,
        onsets=onsets,
        blockers=blockers,
        warnings=warnings,
        thresholds=thresholds,
        source_metadata={
            **dict(source_metadata or {}),
            "input_sample_rate_hz": sample_rate_hz,
        },
        clip=clip,
        frame_rate=frame_rate,
        source_to_microphone_distance_m=source_to_microphone_distance_m,
        distance_uncertainty_m=distance_uncertainty_m,
        speed_of_sound_mps=speed_of_sound_mps,
        media_sha256=media_sha256,
        pts_source=pts_source,
    )


def write_audio_onsets_v2(path: str | Path, payload: Mapping[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _detect_onsets(
    samples: np.ndarray,
    *,
    sample_rate_hz: int,
    bandpass_low_hz: float,
    bandpass_high_hz: float,
    frame_size_s: float,
    hop_s: float,
    min_separation_s: float,
    threshold_mad: float,
    adaptive_window_s: float,
    min_pop_band_ratio: float,
    min_spectral_evidence: float,
    min_hfc_evidence: float,
    time_offset_s: float,
) -> tuple[list[dict[str, Any]], int]:
    frame_size = max(16, int(round(frame_size_s * sample_rate_hz)))
    hop = max(1, int(round(hop_s * sample_rate_hz)))
    if len(samples) < frame_size * 2:
        return [], 0

    centered = samples - float(np.mean(samples)) if len(samples) else samples
    raw_frames = _frame_signal(centered, frame_size=frame_size, hop=hop)
    filtered = _bandpass(samples, sample_rate_hz=sample_rate_hz, low_hz=bandpass_low_hz, high_hz=bandpass_high_hz)
    frames = _frame_signal(filtered, frame_size=frame_size, hop=hop)
    if len(frames) < 3:
        return [], 0

    window = np.hanning(frame_size)
    spectrum = np.abs(np.fft.rfft(frames * window, axis=1))
    raw_spectrum = np.abs(np.fft.rfft(raw_frames * window, axis=1))
    freqs = np.fft.rfftfreq(frame_size, d=1.0 / sample_rate_hz)
    band = (freqs >= bandpass_low_hz) & (freqs <= min(bandpass_high_hz, sample_rate_hz * 0.5 - 1.0))
    low_band = (freqs >= 50.0) & (freqs < bandpass_low_hz)
    if not np.any(band):
        return [], 0

    raw_pop_power = np.sum(raw_spectrum[:, band] ** 2, axis=1)
    raw_low_power = np.sum(raw_spectrum[:, low_band] ** 2, axis=1) if np.any(low_band) else np.zeros_like(raw_pop_power)
    pop_band_ratio = raw_pop_power / np.maximum(raw_pop_power + raw_low_power, 1e-12)
    band_mag = spectrum[:, band]
    spectral_flux = np.r_[0.0, np.sum(np.maximum(0.0, band_mag[1:] - band_mag[:-1]), axis=1)]
    hfc_weights = np.maximum(freqs[band] / bandpass_low_hz, 1.0)
    high_frequency_content = np.sum((band_mag**2) * hfc_weights, axis=1)
    band_rms = np.sqrt(np.mean(frames**2, axis=1))
    band_energy_delta = np.r_[0.0, np.maximum(0.0, np.diff(band_rms))]

    window_frames = max(3, int(round(adaptive_window_s / hop_s)))
    flux_z = _adaptive_positive_z(spectral_flux, window_frames=window_frames)
    hfc_z = _adaptive_positive_z(high_frequency_content, window_frames=window_frames)
    energy_z = _adaptive_positive_z(band_energy_delta, window_frames=window_frames)
    onset_strength = 0.45 * flux_z + 0.35 * hfc_z + 0.20 * energy_z

    distance_frames = max(1, int(round(min_separation_s / hop_s)))
    peaks, properties = signal.find_peaks(onset_strength, height=threshold_mad, distance=distance_frames)
    peaks = np.asarray(
        [
            int(peak)
            for peak in peaks
            if pop_band_ratio[int(peak)] >= min_pop_band_ratio
            and flux_z[int(peak)] + hfc_z[int(peak)] >= min_spectral_evidence
            and hfc_z[int(peak)] >= min_hfc_evidence
        ],
        dtype=int,
    )
    if len(peaks) == 0:
        return [], 0

    peak_strengths = onset_strength[peaks]
    score_norm = max(float(np.percentile(peak_strengths, 95)), threshold_mad, 1e-9)
    onsets: list[dict[str, Any]] = []
    for peak in peaks:
        sub_frame = peak + _parabolic_offset(onset_strength, int(peak))
        analysis_time_s = max(0.0, (sub_frame * hop + frame_size * 0.5) / sample_rate_hz)
        time_s = time_offset_s + analysis_time_s
        score = min(1.0, max(0.0, float(onset_strength[peak]) / score_norm))
        onsets.append(
            {
                "time_s": round(time_s, 9),
                "raw_time_s": round(time_s, 9),
                "analysis_time_s": round(analysis_time_s, 9),
                "score": round(score, 6),
                "onset_strength": round(float(onset_strength[peak]), 6),
                "source": "audio_pop_v2",
                "window_start_s": round(time_offset_s + (peak * hop) / sample_rate_hz, 9),
                "window_end_s": round(time_offset_s + (peak * hop + frame_size) / sample_rate_hz, 9),
                "features": {
                    "spectral_flux": round(float(flux_z[peak]), 6),
                    "high_frequency_content": round(float(hfc_z[peak]), 6),
                    "band_energy_delta": round(float(energy_z[peak]), 6),
                    "pop_band_ratio": round(float(pop_band_ratio[peak]), 6),
                },
            }
        )

    onsets.sort(key=lambda item: float(item["time_s"]))
    return onsets, int(len(peaks))


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
    source_to_microphone_distance_m: float | None = None,
    distance_uncertainty_m: float = 0.0,
    speed_of_sound_mps: float = DEFAULT_SPEED_OF_SOUND_MPS,
    media_sha256: str | None = None,
    pts_source: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    ordered_onsets, timing = finalize_audio_onset_timing(
        onsets,
        source_to_microphone_distance_m=source_to_microphone_distance_m,
        distance_uncertainty_m=distance_uncertainty_m,
        speed_of_sound_mps=speed_of_sound_mps,
    )
    enriched_onsets = _attach_nearest_frames(ordered_onsets, frame_rate=frame_rate)
    identity = _dependency_identity_fields(
        media_sha256=media_sha256,
        pts_source=pts_source,
    )
    return {
        "schema_version": 1,
        "artifact_type": ARTIFACT_TYPE,
        "detector_version": DETECTOR_VERSION,
        "status": status,
        **({"clip": clip} if clip else {}),
        **({"frame_rate": float(frame_rate)} if frame_rate is not None else {}),
        "source": source,
        "source_path": str(source_path),
        **identity,
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
        "timing": timing,
        "onsets": enriched_onsets,
    }


def _dependency_identity_fields(
    *, media_sha256: str | None, pts_source: Mapping[str, Any] | None
) -> dict[str, Any]:
    """Return additive media/PTS identities for SHA-bound downstream consumers."""

    identity: dict[str, Any] = {}
    if media_sha256 is not None:
        if not isinstance(media_sha256, str) or len(media_sha256) != 64:
            raise ValueError("media_sha256 must be a 64-character SHA-256 digest")
        identity.update({
            "media_sha256": media_sha256,
            "source_video_sha256": media_sha256,
        })
    if pts_source is not None:
        pts_identity = dict(pts_source)
        pts_sha256 = pts_identity.get("sha256")
        if not isinstance(pts_sha256, str) or len(pts_sha256) != 64:
            raise ValueError("pts_source.sha256 must be a 64-character SHA-256 digest")
        identity.update({
            "pts_source": pts_identity,
            "frame_times_sha256": pts_sha256,
        })
    return identity


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


def _as_float_array(samples: Sequence[float]) -> np.ndarray:
    array = np.asarray(samples, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError("samples must be mono")
    if len(array) == 0:
        return array
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(array, -1.0, 1.0)


def _resample(samples: np.ndarray, *, source_rate_hz: int, target_rate_hz: int) -> np.ndarray:
    divisor = math.gcd(source_rate_hz, target_rate_hz)
    up = target_rate_hz // divisor
    down = source_rate_hz // divisor
    return signal.resample_poly(samples, up, down)


def _bandpass(samples: np.ndarray, *, sample_rate_hz: int, low_hz: float, high_hz: float) -> np.ndarray:
    nyquist = sample_rate_hz * 0.5
    high_hz = min(high_hz, nyquist * 0.95)
    if not 0.0 < low_hz < high_hz:
        raise ValueError("bandpass frequencies must satisfy 0 < low < high < Nyquist")
    centered = samples - float(np.mean(samples)) if len(samples) else samples
    sos = signal.butter(4, [low_hz, high_hz], btype="bandpass", fs=sample_rate_hz, output="sos")
    padlen = min(len(centered) - 1, 3 * (2 * len(sos) + 1))
    if padlen > 0:
        return signal.sosfiltfilt(sos, centered, padlen=padlen)
    return signal.sosfilt(sos, centered)


def _frame_signal(samples: np.ndarray, *, frame_size: int, hop: int) -> np.ndarray:
    frame_count = 1 + (len(samples) - frame_size) // hop
    if frame_count <= 0:
        return np.empty((0, frame_size), dtype=np.float64)
    shape = (frame_count, frame_size)
    strides = (samples.strides[0] * hop, samples.strides[0])
    return np.lib.stride_tricks.as_strided(samples, shape=shape, strides=strides).copy()


def _adaptive_positive_z(values: np.ndarray, *, window_frames: int) -> np.ndarray:
    values = np.log1p(np.maximum(values.astype(np.float64), 0.0))
    if len(values) == 0:
        return values
    if window_frames % 2 == 0:
        window_frames += 1
    baseline = ndimage.median_filter(values, size=window_frames, mode="nearest")
    deviation = np.abs(values - baseline)
    local_mad = ndimage.median_filter(deviation, size=window_frames, mode="nearest")
    global_median = float(np.median(values))
    global_mad = float(np.median(np.abs(values - global_median)))
    scale = np.maximum(1.4826 * local_mad, 0.25 * 1.4826 * global_mad)
    scale = np.maximum(scale, 1e-6)
    return np.maximum(0.0, (values - baseline) / scale)


def _parabolic_offset(values: np.ndarray, index: int) -> float:
    if index <= 0 or index >= len(values) - 1:
        return 0.0
    left = float(values[index - 1])
    center = float(values[index])
    right = float(values[index + 1])
    denominator = left - 2.0 * center + right
    if abs(denominator) < 1e-12:
        return 0.0
    return float(np.clip(0.5 * (left - right) / denominator, -0.5, 0.5))


def _threshold_summary(
    *,
    analysis_sample_rate_hz: int,
    bandpass_low_hz: float,
    bandpass_high_hz: float,
    frame_size_s: float,
    hop_s: float,
    min_separation_s: float,
    threshold_mad: float,
    adaptive_window_s: float,
    min_pop_band_ratio: float,
    min_spectral_evidence: float,
    min_hfc_evidence: float,
    start_s: float,
    duration_limit_s: float | None,
) -> dict[str, Any]:
    return {
        "analysis_sample_rate_hz": int(analysis_sample_rate_hz),
        "bandpass_low_hz": float(bandpass_low_hz),
        "bandpass_high_hz": float(bandpass_high_hz),
        "frame_size_s": float(frame_size_s),
        "hop_s": float(hop_s),
        "min_separation_s": float(min_separation_s),
        "threshold_mad": float(threshold_mad),
        "adaptive_window_s": float(adaptive_window_s),
        "min_pop_band_ratio": float(min_pop_band_ratio),
        "min_spectral_evidence": float(min_spectral_evidence),
        "min_hfc_evidence": float(min_hfc_evidence),
        "start_s": float(start_s),
        "duration_limit_s": duration_limit_s,
    }


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


def _require_unit(value: float, name: str) -> float:
    value = _require_finite(value, name)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return value


def _require_finite(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


__all__ = [
    "DETECTOR_VERSION",
    "build_audio_onsets_v2_from_samples",
    "build_audio_onsets_v2_from_video",
    "build_audio_onsets_v2_from_wav",
    "write_audio_onsets_v2",
]

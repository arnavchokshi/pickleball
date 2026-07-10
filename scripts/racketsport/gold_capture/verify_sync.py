#!/usr/bin/env python3
"""Measure LED-flash or audio-clap alignment across two or three capture clips."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


@dataclass(frozen=True)
class EventMeasurement:
    clip: Path
    method: str
    event_time_seconds: float
    source_fps: float | None
    frame_count: int | None
    event_frame_float: float | None
    signal_to_noise: float
    details: dict[str, Any]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _frame_pts(path: Path) -> list[float]:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return []
    completed = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "frame=best_effort_timestamp_time",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    frames = json.loads(completed.stdout).get("frames", [])
    return [float(frame["best_effort_timestamp_time"]) for frame in frames if "best_effort_timestamp_time" in frame]


def _parse_roi(value: str | None, width: int, height: int) -> tuple[int, int, int, int]:
    if value is None:
        return (0, 0, width, height)
    parts = [int(item) for item in value.split(",")]
    if len(parts) != 4:
        raise ValueError("ROI must be x,y,width,height")
    x, y, roi_width, roi_height = parts
    if x < 0 or y < 0 or roi_width <= 0 or roi_height <= 0 or x + roi_width > width or y + roi_height > height:
        raise ValueError(f"ROI {value!r} is outside {width}x{height}")
    return x, y, roi_width, roi_height


def measure_led(path: Path, *, roi: str | None = None) -> EventMeasurement:
    import cv2  # type: ignore[import-not-found]
    import numpy as np

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"cannot open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    values: list[float] = []
    timestamps: list[float] = []
    resolved_roi: tuple[int, int, int, int] | None = None
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if resolved_roi is None:
            resolved_roi = _parse_roi(roi, int(frame.shape[1]), int(frame.shape[0]))
        x, y, width, height = resolved_roi
        gray = cv2.cvtColor(frame[y : y + height, x : x + width], cv2.COLOR_BGR2GRAY)
        values.append(float(np.mean(gray)))
        timestamps.append(float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0)
    capture.release()
    if len(values) < 5:
        raise ValueError(f"{path}: need at least five frames")

    pts = _frame_pts(path)
    if len(pts) == len(values):
        timestamps = pts
    elif not timestamps or any(timestamps[index] <= timestamps[index - 1] for index in range(1, len(timestamps))):
        if fps <= 0.0:
            raise ValueError(f"{path}: no usable PTS or FPS")
        timestamps = [index / fps for index in range(len(values))]

    signal = np.asarray(values, dtype=np.float64)
    rises = np.diff(signal)
    rise_index = int(np.argmax(rises)) + 1
    baseline = float(np.median(signal[: max(1, rise_index)]))
    lit_window = signal[rise_index : min(len(signal), rise_index + 3)]
    lit_level = float(np.median(lit_window))
    rise = lit_level - baseline
    noise = float(1.4826 * np.median(np.abs(rises - np.median(rises))))
    if rise < 5.0:
        raise ValueError(f"{path}: no LED rise of at least 5 luma detected (rise={rise:.3f})")
    previous = float(signal[rise_index - 1])
    current = float(signal[rise_index])
    target = baseline + 0.5 * rise
    fraction = 0.5 if math.isclose(current, previous) else min(1.0, max(0.0, (target - previous) / (current - previous)))
    event_time = timestamps[rise_index - 1] + fraction * (timestamps[rise_index] - timestamps[rise_index - 1])
    frame_float = (rise_index - 1) + fraction
    return EventMeasurement(
        clip=path,
        method="led",
        event_time_seconds=event_time,
        source_fps=fps if fps > 0.0 else None,
        frame_count=len(values),
        event_frame_float=frame_float,
        signal_to_noise=rise / max(noise, 1e-6),
        details={
            "roi_xywh": list(resolved_roi or ()),
            "baseline_luma": baseline,
            "lit_luma": lit_level,
            "rise_luma": rise,
            "timestamp_source": "ffprobe_best_effort_pts" if len(pts) == len(values) else "opencv_or_fps",
        },
    )


def measure_audio(path: Path, *, source_distance_m: float | None = None) -> EventMeasurement:
    import numpy as np

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required for audio-clap analysis")
    sample_rate = 48_000
    completed = subprocess.run(
        [ffmpeg, "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", str(sample_rate), "-f", "f32le", "-"],
        check=True,
        capture_output=True,
    )
    samples = np.frombuffer(completed.stdout, dtype="<f4").astype(np.float64)
    if samples.size < sample_rate // 10:
        raise ValueError(f"{path}: missing or too-short audio track")
    window = max(1, sample_rate // 1000)
    envelope = np.sqrt(np.convolve(samples * samples, np.ones(window) / window, mode="same"))
    baseline_count = max(window, min(samples.size // 4, sample_rate))
    baseline = envelope[:baseline_count]
    median = float(np.median(baseline))
    noise = float(1.4826 * np.median(np.abs(baseline - median)))
    peak_index = int(np.argmax(envelope))
    threshold = median + max(8.0 * noise, 0.05 * float(envelope[peak_index]))
    search_start = max(0, peak_index - sample_rate // 20)
    crossings = np.flatnonzero(envelope[search_start : peak_index + 1] >= threshold)
    if crossings.size == 0:
        raise ValueError(f"{path}: no clap onset detected")
    onset_index = search_start + int(crossings[0])
    raw_time = onset_index / sample_rate
    propagation = (source_distance_m / 343.0) if source_distance_m is not None else 0.0
    return EventMeasurement(
        clip=path,
        method="audio",
        event_time_seconds=raw_time - propagation,
        source_fps=None,
        frame_count=None,
        event_frame_float=None,
        signal_to_noise=float(envelope[peak_index]) / max(noise, 1e-9),
        details={
            "sample_rate_hz": sample_rate,
            "raw_onset_seconds": raw_time,
            "source_distance_m": source_distance_m,
            "acoustic_propagation_correction_seconds": propagation,
            "speed_of_sound_mps_assumed": 343.0,
        },
    )


def _method_report(measurements: Sequence[EventMeasurement], *, gate_fps: float) -> dict[str, Any]:
    reference = measurements[0]
    offsets = [measurement.event_time_seconds - reference.event_time_seconds for measurement in measurements]
    max_pairwise = max(offsets) - min(offsets)
    threshold = 0.5 / gate_fps
    return {
        "method": reference.method,
        "reference_clip": reference.clip.as_posix(),
        "gate_fps": gate_fps,
        "half_frame_threshold_seconds": threshold,
        "max_pairwise_offset_seconds": max_pairwise,
        "max_pairwise_offset_frames": max_pairwise * gate_fps,
        "gate_pass": max_pairwise <= threshold,
        "measurements": [
            {
                "clip": item.clip.as_posix(),
                "immutable_raw_reference": {
                    "uri": item.clip.as_posix(),
                    "sha256": _sha256(item.clip),
                    "size_bytes": item.clip.stat().st_size,
                },
                "event_time_seconds": item.event_time_seconds,
                "offset_from_reference_seconds": item.event_time_seconds - reference.event_time_seconds,
                "source_fps": item.source_fps,
                "frame_count": item.frame_count,
                "event_frame_float": item.event_frame_float,
                "signal_to_noise": item.signal_to_noise,
                "details": item.details,
            }
            for item in measurements
        ],
    }


def analyze_clips(
    clips: Sequence[Path],
    *,
    method: str,
    gate_fps: float,
    roi: str | None = None,
    audio_distances_m: Sequence[float] | None = None,
) -> dict[str, Any]:
    if len(clips) not in (2, 3):
        raise ValueError("provide exactly two or three clips")
    if gate_fps <= 0.0:
        raise ValueError("gate FPS must be positive")
    for clip in clips:
        if not clip.is_file():
            raise ValueError(f"missing clip: {clip}")
    if audio_distances_m is not None and len(audio_distances_m) != len(clips):
        raise ValueError("provide one --audio-distance-m per clip")

    reports: dict[str, Any] = {}
    if method in {"led", "both"}:
        reports["led"] = _method_report([measure_led(clip, roi=roi) for clip in clips], gate_fps=gate_fps)
    if method in {"audio", "both"}:
        distances = list(audio_distances_m) if audio_distances_m is not None else [None] * len(clips)
        reports["audio"] = _method_report(
            [measure_audio(clip, source_distance_m=distance) for clip, distance in zip(clips, distances, strict=True)],
            gate_fps=gate_fps,
        )

    primary = reports["led"] if "led" in reports else reports["audio"]
    warnings: list[str] = []
    if "audio" in reports and audio_distances_m is None:
        warnings.append("audio offsets are uncorrected for source-to-camera acoustic travel distance")
    return {
        "schema_version": 1,
        "artifact_type": "gold_capture_sync_verification",
        "status": "pass" if primary["gate_pass"] else "fail",
        "primary_gate_method": primary["method"],
        "gate_pass": primary["gate_pass"],
        "methods": reports,
        "warnings": warnings,
        "product_boundary": "The product remains monocular; extra cameras, markers, and surveys are GT-only.",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("clips", nargs="+", type=Path, help="Exactly two or three clips containing the same sync event.")
    parser.add_argument("--method", choices=("led", "audio", "both"), default="led")
    parser.add_argument("--gate-fps", type=float, default=240.0, help="Frame rate defining the half-frame gate.")
    parser.add_argument("--roi", help="Optional shared LED ROI as x,y,width,height.")
    parser.add_argument("--audio-distance-m", action="append", type=float, help="Clap-source distance for each clip.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = analyze_clips(
            args.clips,
            method=args.method,
            gate_fps=args.gate_fps,
            roi=args.roi,
            audio_distances_m=args.audio_distance_m,
        )
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
        parser.exit(2, f"{parser.prog}: error: {exc}\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0 if report["gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

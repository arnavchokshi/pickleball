from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .capture_quality import score_capture_quality
from .schemas import CaptureQuality


@dataclass(frozen=True)
class FrameSource:
    """Phase 0 clip metadata returned by the decode/probe layer."""

    path: Path
    width: int
    height: int
    fps: float
    duration_s: float
    frame_count: int | None
    audio_sample_rate: int | None
    fps_out: float | None = None

    def to_frames_meta(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "clip_path": str(self.path),
            "resolution": [self.width, self.height],
            "fps": self.fps,
            "fps_out": self.fps_out,
            "duration_s": self.duration_s,
            "frame_count": self.frame_count,
            "audio_sample_rate": self.audio_sample_rate,
        }


@dataclass(frozen=True)
class ClipQualityProbe:
    """Sampled clip-level QC metrics extracted before expensive model stages."""

    sample_width: int
    sample_height: int
    sampled_frames: int
    qc_decode_fps: float
    blur_laplacian_var: float
    luminance_mean: float
    luminance_std_fraction: float
    capture_quality: CaptureQuality

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_resolution": [self.sample_width, self.sample_height],
            "sampled_frames": self.sampled_frames,
            "qc_decode_fps": self.qc_decode_fps,
            "blur_laplacian_var": self.blur_laplacian_var,
            "luminance_mean": self.luminance_mean,
            "luminance_std_fraction": self.luminance_std_fraction,
            "capture_quality": self.capture_quality.model_dump(),
        }


@dataclass(frozen=True)
class DecodeBenchmark:
    """Wall-clock decode throughput for Phase 0 backend validation."""

    backend: str
    elapsed_s: float
    duration_s: float
    frame_count: int
    decode_fps: float
    realtime_factor: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "elapsed_s": self.elapsed_s,
            "duration_s": self.duration_s,
            "frame_count": self.frame_count,
            "decode_fps": self.decode_fps,
            "realtime_factor": self.realtime_factor,
        }


def _parse_rational(value: str | None) -> float | None:
    if not value or value == "0/0":
        return None
    if "/" not in value:
        return float(value)
    numerator, denominator = value.split("/", 1)
    denominator_f = float(denominator)
    if denominator_f == 0:
        return None
    return float(numerator) / denominator_f


def _run_ffprobe(path: Path) -> dict[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError("ffprobe is required for Phase 0 clip probing") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"ffprobe failed for {path}: {exc.stderr.strip()}") from exc
    return json.loads(completed.stdout)


def _scaled_sample_size(width: int, height: int, max_width: int) -> tuple[int, int]:
    sample_width = max(2, min(width, max_width))
    if sample_width % 2:
        sample_width -= 1
    sample_height = max(2, round(height * sample_width / width))
    if sample_height % 2:
        sample_height += 1
    return sample_width, sample_height


def _variance(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return sum((value - mean) ** 2 for value in values) / len(values)


def _laplacian_variance(gray: bytes, width: int, height: int) -> float:
    if width < 3 or height < 3:
        return 0.0

    values: list[float] = []
    for y in range(1, height - 1):
        row = y * width
        row_above = (y - 1) * width
        row_below = (y + 1) * width
        for x in range(1, width - 1):
            center = gray[row + x]
            laplacian = (
                4 * center
                - gray[row + x - 1]
                - gray[row + x + 1]
                - gray[row_above + x]
                - gray[row_below + x]
            )
            values.append(float(laplacian))
    return _variance(values)


def _mean_luminance(gray: bytes) -> float:
    if not gray:
        return 0.0
    return sum(gray) / (len(gray) * 255.0)


def probe_clip(path: str | Path, *, fps_out: float | None = None) -> FrameSource:
    clip_path = Path(path)
    if not clip_path.exists():
        raise FileNotFoundError(clip_path)

    payload = _run_ffprobe(clip_path)
    streams = payload.get("streams", [])
    video_stream = next((stream for stream in streams if stream.get("codec_type") == "video"), None)
    if video_stream is None:
        raise ValueError(f"no video stream found in {clip_path}")

    audio_stream = next((stream for stream in streams if stream.get("codec_type") == "audio"), None)
    fps = _parse_rational(video_stream.get("avg_frame_rate")) or _parse_rational(
        video_stream.get("r_frame_rate")
    )
    if fps is None:
        raise ValueError(f"could not determine frame rate for {clip_path}")

    duration_s = float(video_stream.get("duration") or payload.get("format", {}).get("duration") or 0.0)
    raw_frame_count = video_stream.get("nb_frames")
    frame_count = int(raw_frame_count) if raw_frame_count not in (None, "N/A") else None
    if frame_count is None and duration_s > 0:
        frame_count = round(duration_s * fps)

    sample_rate: int | None = None
    if audio_stream is not None and audio_stream.get("sample_rate"):
        sample_rate = int(audio_stream["sample_rate"])

    return FrameSource(
        path=clip_path,
        width=int(video_stream["width"]),
        height=int(video_stream["height"]),
        fps=fps,
        duration_s=duration_s,
        frame_count=frame_count,
        audio_sample_rate=sample_rate,
        fps_out=fps_out,
    )


def analyze_clip_qc(
    path: str | Path,
    *,
    source: FrameSource | None = None,
    sample_fps: float = 1.0,
    max_frames: int = 30,
    max_width: int = 160,
) -> ClipQualityProbe:
    """Sample frames with ffmpeg and compute deterministic ingest QC signals."""

    if sample_fps <= 0:
        raise ValueError("sample_fps must be positive")
    if max_frames <= 0:
        raise ValueError("max_frames must be positive")

    clip_path = Path(path)
    source = source or probe_clip(clip_path)
    sample_width, sample_height = _scaled_sample_size(source.width, source.height, max_width)
    frame_bytes = sample_width * sample_height
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(clip_path),
        "-vf",
        f"fps={sample_fps},scale={sample_width}:{sample_height},format=gray",
        "-frames:v",
        str(max_frames),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "-",
    ]

    started = time.perf_counter()
    try:
        completed = subprocess.run(command, check=True, capture_output=True)
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg is required for Phase 0 clip QC") from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg QC failed for {clip_path}: {stderr}") from exc

    elapsed = max(time.perf_counter() - started, 1e-9)
    payload = completed.stdout
    sampled_frames = len(payload) // frame_bytes
    if sampled_frames == 0:
        raise ValueError(f"no QC sample frames decoded from {clip_path}")

    blur_values: list[float] = []
    luminance_values: list[float] = []
    for index in range(sampled_frames):
        start = index * frame_bytes
        gray = payload[start : start + frame_bytes]
        blur_values.append(_laplacian_variance(gray, sample_width, sample_height))
        luminance_values.append(_mean_luminance(gray))

    luminance_mean = sum(luminance_values) / len(luminance_values)
    luminance_std_fraction = _variance(luminance_values) ** 0.5
    blur_laplacian_var = sum(blur_values) / len(blur_values)
    quality = score_capture_quality(
        blur_laplacian_var=blur_laplacian_var,
        luminance_mean=luminance_mean,
        luminance_std_fraction=luminance_std_fraction,
        fps=source.fps,
    )

    return ClipQualityProbe(
        sample_width=sample_width,
        sample_height=sample_height,
        sampled_frames=sampled_frames,
        qc_decode_fps=sampled_frames / elapsed,
        blur_laplacian_var=blur_laplacian_var,
        luminance_mean=luminance_mean,
        luminance_std_fraction=luminance_std_fraction,
        capture_quality=quality,
    )


def measure_decode_throughput(path: str | Path, *, backend: str = "cpu") -> DecodeBenchmark:
    """Measure ffmpeg video decode throughput without materializing frames."""

    if backend not in {"cpu", "cuda"}:
        raise ValueError(f"Unsupported decode backend: {backend}")

    clip_path = Path(path)
    source = probe_clip(clip_path)
    command = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error"]
    if backend == "cuda":
        command.extend(["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"])
    command.extend(["-i", str(clip_path), "-map", "0:v:0", "-an", "-f", "null", "-"])

    started = time.perf_counter()
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg is required for Phase 0 decode benchmarking") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"ffmpeg {backend} decode failed for {clip_path}: {exc.stderr.strip()}") from exc

    elapsed_s = max(time.perf_counter() - started, 1e-9)
    frame_count = source.frame_count or max(1, round(source.duration_s * source.fps))
    decode_fps = frame_count / elapsed_s
    realtime_factor = source.duration_s / elapsed_s if source.duration_s > 0 else 0.0

    return DecodeBenchmark(
        backend=backend,
        elapsed_s=elapsed_s,
        duration_s=source.duration_s,
        frame_count=frame_count,
        decode_fps=decode_fps,
        realtime_factor=realtime_factor,
    )


def decode_clip(path: str | Path, fps_out: float | None = None) -> FrameSource:
    """Return Phase 0 metadata for a clip.

    Full NVDEC frame iteration is wired in later once the GPU worker environment
    is finalized. Phase 0 callers can still use this deterministic metadata
    contract for ingestion and schema tests.
    """

    return probe_clip(path, fps_out=fps_out)

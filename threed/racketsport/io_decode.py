from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


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


def decode_clip(path: str | Path, fps_out: float | None = None) -> FrameSource:
    """Return Phase 0 metadata for a clip.

    Full NVDEC frame iteration is wired in later once the GPU worker environment
    is finalized. Phase 0 callers can still use this deterministic metadata
    contract for ingestion and schema tests.
    """

    return probe_clip(path, fps_out=fps_out)


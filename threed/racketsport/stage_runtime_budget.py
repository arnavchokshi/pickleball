"""Per-stage runtime budget: normalize measured costs to seconds-per-minute-of-video.

GLUE-3 speed-budget lane. This module answers "how expensive is each offline
pipeline stage, normalized so stages can be compared apples-to-apples" using
**measured** numbers, not guesses:

- decode / calibration / world build are cheap, CPU-only, sub-second
  operations, so this module measures them **fresh, locally** (see
  ``measure_decode_cost``, ``measure_calibration_cost``,
  ``measure_world_build_cost``) -- there is no reason to trust a stale number
  when a real one costs nothing to produce.
- detection+tracking / ball inference (TrackNetV3, WASB) / BODY mesh /
  person-ReID association are GPU-bound and were already measured in prior
  A100 runs elsewhere in this repo. This module does **not** re-run those
  models (that would require spinning up a GPU, out of scope for a
  runtime-budget lane); instead it loads the already-recorded evidence JSON
  and normalizes it with the same formula used for the cheap stages, so every
  stage lands in the same unit: **seconds of compute per minute of video**.

All the ``*_to_seconds_per_minute_video`` helpers are pure unit-conversion
functions (independently unit-tested); the ``load_*_evidence`` functions read
a specific already-existing run artifact and extract the field(s) needed,
raising ``KeyError``/``ValueError`` loudly rather than silently defaulting if
the artifact shape changes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .io_decode import measure_decode_throughput

SECONDS_PER_MINUTE = 60.0


@dataclass(frozen=True)
class StageCost:
    """One row of the runtime budget table, already normalized to seconds/minute-of-video."""

    stage: str
    seconds_per_minute_video: float
    basis: str
    source: str
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "seconds_per_minute_video": round(self.seconds_per_minute_video, 3),
            "compute_minutes_per_video_minute": round(self.seconds_per_minute_video / SECONDS_PER_MINUTE, 4),
            "basis": self.basis,
            "source": self.source,
            "notes": self.notes,
        }


# --- Pure unit-conversion helpers (independently testable, no I/O) -----------------


def per_frame_ms_to_seconds_per_minute_video(ms_per_frame: float, *, video_fps: float) -> float:
    """E.g. TrackNetV3/WASB: a fixed cost paid once per decoded video frame."""

    if ms_per_frame < 0:
        raise ValueError("ms_per_frame must be non-negative")
    if video_fps <= 0:
        raise ValueError("video_fps must be positive")
    frames_per_minute = video_fps * SECONDS_PER_MINUTE
    return (ms_per_frame / 1000.0) * frames_per_minute


def per_unit_ms_to_seconds_per_minute_video(ms_per_unit: float, *, units_per_video_second: float) -> float:
    """General form for costs keyed to a schedule other than raw video frames,
    e.g. BODY's "person-frames" (tracked-player crops), which scale with
    scheduling density (event-triggered vs dense-replay vs naive-all-players).
    """

    if ms_per_unit < 0:
        raise ValueError("ms_per_unit must be non-negative")
    if units_per_video_second < 0:
        raise ValueError("units_per_video_second must be non-negative")
    units_per_minute = units_per_video_second * SECONDS_PER_MINUTE
    return (ms_per_unit / 1000.0) * units_per_minute


def fps_to_seconds_per_minute_video(processing_fps: float, *, video_fps: float) -> float:
    """E.g. detection+tracking reported as "N processed fps" against source video_fps."""

    if processing_fps <= 0:
        raise ValueError("processing_fps must be positive")
    if video_fps <= 0:
        raise ValueError("video_fps must be positive")
    frames_per_minute = video_fps * SECONDS_PER_MINUTE
    return frames_per_minute / processing_fps


def wall_seconds_to_seconds_per_minute_video(wall_seconds: float, clip_seconds_processed: float) -> float:
    """E.g. a measured (wall_seconds, video_seconds_processed) pair from any stage's own run log."""

    if wall_seconds < 0:
        raise ValueError("wall_seconds must be non-negative")
    if clip_seconds_processed <= 0:
        raise ValueError("clip_seconds_processed must be positive")
    return (wall_seconds / clip_seconds_processed) * SECONDS_PER_MINUTE


def fixed_cost_amortized_per_minute_video(fixed_seconds: float, *, video_minutes: float) -> float:
    """A one-time-per-job cost (e.g. model setup/warmup, per-clip calibration)
    spread across the length of the job it is paid for once."""

    if fixed_seconds < 0:
        raise ValueError("fixed_seconds must be non-negative")
    if video_minutes <= 0:
        raise ValueError("video_minutes must be positive")
    return fixed_seconds / video_minutes


# --- Fresh, cheap, local measurements -----------------------------------------------


def measure_decode_cost(clip_path: str | Path, *, backend: str = "cpu") -> StageCost:
    """Fresh local ffmpeg decode-throughput benchmark (no model inference)."""

    bench = measure_decode_throughput(clip_path, backend=backend)
    spm = wall_seconds_to_seconds_per_minute_video(bench.elapsed_s, bench.duration_s)
    return StageCost(
        stage="decode",
        seconds_per_minute_video=spm,
        basis=f"fresh local ffmpeg {backend} decode benchmark ({bench.frame_count} frames, {bench.duration_s:.2f}s clip)",
        source=str(clip_path),
        notes=f"decode_fps={bench.decode_fps:.1f} realtime_factor={bench.realtime_factor:.1f}x elapsed_s={bench.elapsed_s:.3f}",
    )


def stage_cost_from_wall_clock(
    *,
    stage: str,
    wall_seconds: float,
    clip_seconds_processed: float,
    basis: str,
    source: str,
    notes: str = "",
) -> StageCost:
    """Build a StageCost from a directly-timed (wall_seconds, clip_seconds) pair,
    e.g. the calibration/world-build CLI timings captured by the caller with
    ``time.perf_counter()`` around a subprocess or direct function call."""

    spm = wall_seconds_to_seconds_per_minute_video(wall_seconds, clip_seconds_processed)
    return StageCost(stage=stage, seconds_per_minute_video=spm, basis=basis, source=source, notes=notes)


# --- Evidence loaders: extract already-recorded numbers from existing run JSON -----


def _read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def load_tracknet_metadata_evidence(path: str | Path, *, video_fps: float) -> StageCost:
    """Read a ``tracknet_metadata.json`` (TrackNetV3 A100/local run) and normalize
    its measured (wall_seconds, video_seconds_processed) to seconds/minute-of-video."""

    payload = _read_json(path)
    runtime = payload["runtime"]
    wall_seconds = float(runtime["wall_seconds"])
    video_seconds_processed = float(runtime["video_seconds_processed"])
    spm = wall_seconds_to_seconds_per_minute_video(wall_seconds, video_seconds_processed)
    ms_per_frame = (wall_seconds / float(runtime["processed_frame_count"])) * 1000.0
    return StageCost(
        stage="ball_inference_tracknetv3",
        seconds_per_minute_video=spm,
        basis="measured wall_seconds/video_seconds_processed from tracknet_metadata.json",
        source=str(path),
        notes=(
            f"ms_per_frame={ms_per_frame:.1f} processed_frame_count={runtime['processed_frame_count']} "
            f"batch_size={runtime.get('batch_size')}"
        ),
    )


def load_wasb_metadata_evidence(path: str | Path) -> StageCost:
    """Read a ``wasb_metadata.json`` A100 run and normalize its effective_fps.

    Two schema shapes exist in this repo: some runs report ``effective_fps``
    at the top level, others nest it under ``runtime.effective_fps`` (along
    with ``wall_seconds``/``processed_frame_count``); both are handled.
    """

    payload = _read_json(path)
    runtime = payload.get("runtime") if isinstance(payload.get("runtime"), dict) else {}
    effective_fps_raw = payload.get("effective_fps", runtime.get("effective_fps"))
    if effective_fps_raw is None:
        raise KeyError(f"{path} has no 'effective_fps' (checked top level and 'runtime')")
    effective_fps = float(effective_fps_raw)
    frame_count = int(payload.get("frame_count", runtime.get("processed_frame_count", 0)))
    fps = float(payload.get("fps") or runtime.get("source_video_fps") or 60.0)
    spm = fps_to_seconds_per_minute_video(effective_fps, video_fps=fps)
    return StageCost(
        stage="ball_inference_wasb_verifier",
        seconds_per_minute_video=spm,
        basis="measured effective_fps from wasb_metadata.json",
        source=str(path),
        notes=f"effective_fps={effective_fps:.2f} frame_count={frame_count}",
    )


def load_offline_person_authority_evidence(path: str | Path, *, video_fps: float) -> StageCost:
    """Read an ``offline_authority_summary.json`` (global person-ReID association)
    and normalize its measured ``effective_fps`` (unique output video-frame
    indices covered / wall_time_s -- see ``offline_person_authority._effective_fps``)
    to seconds/minute-of-video, the same way TrackNetV3/WASB/detection-tracking
    "processed fps" evidence is normalized.
    """

    payload = _read_json(path)
    wall_time_s = float(payload["wall_time_s"])
    effective_fps = float(payload["effective_fps"])
    spm = fps_to_seconds_per_minute_video(effective_fps, video_fps=video_fps)
    return StageCost(
        stage="association_reid_global",
        seconds_per_minute_video=spm,
        basis="measured effective_fps from offline_authority_summary.json (global_association + CPU ReID embedding)",
        source=str(path),
        notes=(
            f"effective_fps={effective_fps:.3f} wall_time_s={wall_time_s:.1f} "
            f"reid_device={payload.get('config', {}).get('reid_device')}"
        ),
    )


def load_body_cost_model_evidence(path: str | Path, *, scenario_person_frames_per_video_second: float) -> StageCost:
    """Read the committed BODY cost-model assumptions JSON
    (``runs/gpu_pipeline_cost_model_20260701/model_assumptions.json``) and
    normalize the measured A100 ms/person-frame at a given scheduling density."""

    payload = _read_json(path)
    ms_per_person_frame = float(payload["body_a100_fast_seconds_per_person_frame"]) * 1000.0
    spm = per_unit_ms_to_seconds_per_minute_video(
        ms_per_person_frame, units_per_video_second=scenario_person_frames_per_video_second
    )
    return StageCost(
        stage=f"body_mesh_a100_{scenario_person_frames_per_video_second:g}pframes_per_s",
        seconds_per_minute_video=spm,
        basis="measured A100 ms/person-frame from model_assumptions.json, scaled by a scheduling scenario",
        source=str(path),
        notes=(
            f"ms_per_person_frame={ms_per_person_frame:.1f} "
            f"person_frames_per_video_second={scenario_person_frames_per_video_second:g} "
            f"setup_seconds={payload.get('body_setup_seconds')}"
        ),
    )


def load_detection_tracking_fps_evidence(path: str | Path, *, video_fps: float, fps_key: str = "fps") -> StageCost:
    """Read a person-tracking ``metrics.json``/substrate summary that records a
    top-level or ``timing.fps`` processed-fps figure."""

    payload = _read_json(path)
    fps_value = payload.get(fps_key)
    if fps_value is None and isinstance(payload.get("timing"), dict):
        fps_value = payload["timing"].get(fps_key)
    if fps_value is None:
        raise KeyError(f"{path} has no '{fps_key}' fps field")
    spm = fps_to_seconds_per_minute_video(float(fps_value), video_fps=video_fps)
    return StageCost(
        stage="detection_tracking",
        seconds_per_minute_video=spm,
        basis="measured processed fps from a person-tracking benchmark/substrate run",
        source=str(path),
        notes=f"processed_fps={float(fps_value):.2f}",
    )

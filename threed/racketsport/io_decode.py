from __future__ import annotations

import json
import math
import subprocess
import time
from collections import OrderedDict
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Mapping, Sequence

from .capture_quality import score_capture_quality
from .schemas import CaptureQuality
from .timebase import (
    ClockDomain,
    CorrectedPTS,
    CorrectionProvenance,
    FrameAbsenceReason,
    FrameAvailability,
    FrameAvailabilityStatus,
    FrameTime,
    RawEncodedPTS,
    RollingShutterModelOutcome,
    SensorClockMappingOutcome,
    SensorClockMappingStatus,
    TimeBasis,
    TimebaseContract,
    TimebaseValidationError,
    build_rolling_shutter_model_from_sidecar,
    build_sensor_clock_mapping_from_sidecar,
)


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


@dataclass(frozen=True)
class FrameTimeResolution:
    time_s: float
    time_basis: str
    provenance: str
    fallback_used: bool


@dataclass(frozen=True)
class TimebaseArtifactBuild:
    legacy_frame_times: dict[str, Any]
    contract: TimebaseContract | None
    evidence: dict[str, Any]


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


def _run_ffprobe_frames(path: Path) -> dict[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-print_format",
        "json",
        "-show_entries",
        (
            "frame=best_effort_timestamp,best_effort_timestamp_time,pts,pts_time,"
            "pkt_pts,pkt_pts_time,duration,duration_time,pkt_duration_time"
        ),
        str(path),
    ]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError("ffprobe is required for frame time probing") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"ffprobe frame probing failed for {path}: {exc.stderr.strip()}") from exc
    return json.loads(completed.stdout)


def _video_stream(payload: Mapping[str, Any], clip_path: Path) -> Mapping[str, Any]:
    streams = payload.get("streams", [])
    video_stream = next((stream for stream in streams if stream.get("codec_type") == "video"), None)
    if video_stream is None:
        raise ValueError(f"no video stream found in {clip_path}")
    return video_stream


def _finite_float_or_none(value: Any) -> float | None:
    if value in (None, "N/A"):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _frame_pts_from_ffprobe_frames(payload: Mapping[str, Any]) -> list[float]:
    frames = payload.get("frames")
    if not isinstance(frames, Sequence) or isinstance(frames, (str, bytes)):
        return []
    pts_values: list[float] = []
    for frame in frames:
        if not isinstance(frame, Mapping):
            continue
        pts = (
            _finite_float_or_none(frame.get("best_effort_timestamp_time"))
            or _finite_float_or_none(frame.get("pts_time"))
            or _finite_float_or_none(frame.get("pkt_pts_time"))
        )
        if pts is None:
            continue
        pts_values.append(float(pts))
    if not pts_values:
        return []
    monotonic: list[float] = []
    previous: float | None = None
    for pts in pts_values:
        if previous is not None and pts < previous:
            return []
        monotonic.append(pts)
        previous = pts
    return monotonic


def _stream_frame_count(video_stream: Mapping[str, Any], duration_s: float, fps: float) -> int:
    raw_frame_count = video_stream.get("nb_frames")
    if raw_frame_count not in (None, "N/A"):
        return int(raw_frame_count)
    if duration_s > 0 and fps > 0:
        return max(1, round(duration_s * fps))
    return 0


def _legacy_frame_time_table(
    clip_path: Path,
    stream_payload: Mapping[str, Any],
    frame_payload: Mapping[str, Any],
) -> dict[str, Any]:
    """The pre-contract frame_times.json behavior, intentionally byte-stable."""

    video_stream = _video_stream(stream_payload, clip_path)
    fps = _parse_rational(video_stream.get("avg_frame_rate")) or _parse_rational(
        video_stream.get("r_frame_rate")
    )
    r_frame_rate = _parse_rational(video_stream.get("r_frame_rate"))
    if fps is None:
        raise ValueError(f"could not determine frame rate for {clip_path}")
    duration_s = float(video_stream.get("duration") or stream_payload.get("format", {}).get("duration") or 0.0)

    pts_values = _frame_pts_from_ffprobe_frames(frame_payload)
    source_start_pts_s: float | None = None
    if pts_values:
        provenance = "ffprobe_pts"
        source_start_pts_s = float(pts_values[0])
        frame_times = [float(value) - source_start_pts_s for value in pts_values]
        fps_assumed = None
    else:
        provenance = "constant_fps_assumed"
        fps_assumed = r_frame_rate or fps
        frame_count = _stream_frame_count(video_stream, duration_s, fps_assumed)
        frame_times = [index / fps_assumed for index in range(frame_count)]

    return {
        "schema_version": 1,
        "artifact_type": "racketsport_frame_times",
        "clip_path": str(clip_path),
        "provenance": provenance,
        "trust_band": "pts" if provenance == "ffprobe_pts" else "constant_fps_assumed",
        "fps": float(fps),
        "r_frame_rate": float(r_frame_rate) if r_frame_rate is not None else None,
        "duration_s": duration_s,
        "frame_count": len(frame_times),
        "source_start_pts_s": source_start_pts_s,
        "fps_assumed": float(fps_assumed) if fps_assumed is not None else None,
        "frames": [
            {"frame": index, "pts_s": round(float(pts_s), 9)}
            for index, pts_s in enumerate(frame_times)
        ],
    }


def _time_base_ticks(video_stream: Mapping[str, Any]) -> tuple[int, int]:
    value = video_stream.get("time_base")
    if not isinstance(value, str) or "/" not in value:
        raise TimebaseValidationError("ffprobe video stream did not declare an exact time_base")
    numerator_text, denominator_text = value.split("/", 1)
    try:
        numerator = int(numerator_text)
        denominator = int(denominator_text)
    except ValueError as exc:
        raise TimebaseValidationError(f"invalid ffprobe time_base {value!r}") from exc
    if numerator <= 0 or denominator <= 0:
        raise TimebaseValidationError(f"invalid ffprobe time_base {value!r}")
    return numerator, denominator


def _exact_decimal_ticks(value: str) -> tuple[int, int]:
    try:
        decimal = Decimal(value)
    except (InvalidOperation, ValueError) as exc:
        raise TimebaseValidationError(f"invalid ffprobe decimal timestamp {value!r}") from exc
    if not decimal.is_finite():
        raise TimebaseValidationError(f"non-finite ffprobe decimal timestamp {value!r}")
    sign, digits, exponent = decimal.as_tuple()
    coefficient = int("".join(str(digit) for digit in digits) or "0")
    if sign:
        coefficient = -coefficient
    if exponent >= 0:
        return coefficient * (10**exponent), 1
    return coefficient, 10 ** (-exponent)


def _exact_frame_observation(
    frame: Mapping[str, Any],
    *,
    frame_index: int,
    time_base: tuple[int, int] | None,
) -> tuple[RawEncodedPTS | None, dict[str, Any]]:
    integer_fields = (
        ("best_effort_timestamp", "best_effort_timestamp_time"),
        ("pts", "pts_time"),
        ("pkt_pts", "pkt_pts_time"),
    )
    decimal_string: str | None = None
    for integer_field, decimal_field in integer_fields:
        raw_integer = frame.get(integer_field)
        raw_decimal = frame.get(decimal_field)
        if raw_decimal not in (None, "N/A"):
            decimal_string = str(raw_decimal)
        if raw_integer in (None, "N/A") or time_base is None:
            continue
        try:
            integer_timestamp = int(raw_integer)
        except (TypeError, ValueError):
            continue
        numerator, denominator = time_base
        duration_raw = frame.get("duration")
        duration_ticks = None
        if duration_raw not in (None, "N/A"):
            try:
                parsed_duration = int(duration_raw) * numerator
            except (TypeError, ValueError):
                parsed_duration = 0
            duration_ticks = parsed_duration if parsed_duration > 0 else None
        raw = RawEncodedPTS(
            frame_index=frame_index,
            pts_ticks=integer_timestamp * numerator,
            timescale=denominator,
            duration_ticks=duration_ticks,
        )
        return raw, {
            "frame_index": frame_index,
            "timestamp_field": integer_field,
            "source_timestamp_decimal": decimal_string,
            "conversion_method": "ffprobe_integer_timestamp_times_stream_time_base",
            "pts_ticks": raw.pts_ticks,
            "timescale": raw.timescale,
            "duration_ticks": raw.duration_ticks,
        }

    for _integer_field, decimal_field in integer_fields:
        raw_decimal = frame.get(decimal_field)
        if raw_decimal in (None, "N/A"):
            continue
        ticks, timescale = _exact_decimal_ticks(str(raw_decimal))
        raw = RawEncodedPTS(frame_index, ticks, timescale)
        return raw, {
            "frame_index": frame_index,
            "timestamp_field": decimal_field,
            "source_timestamp_decimal": str(raw_decimal),
            "conversion_method": "exact_decimal_string_to_integer_ticks",
            "pts_ticks": ticks,
            "timescale": timescale,
            "duration_ticks": None,
        }
    return None, {
        "frame_index": frame_index,
        "timestamp_field": None,
        "source_timestamp_decimal": None,
        "conversion_method": "unavailable",
        "pts_ticks": None,
        "timescale": None,
        "duration_ticks": None,
    }


def build_timebase_artifacts(
    path: str | Path,
    *,
    capture_id: str,
    capture_sidecar: Mapping[str, Any] | str | Path | None = None,
) -> TimebaseArtifactBuild:
    """Build canonical raw-PTS evidence beside the byte-stable legacy table."""

    clip_path = Path(path)
    if not clip_path.exists():
        raise FileNotFoundError(clip_path)
    stream_payload = _run_ffprobe(clip_path)
    frame_payload = _run_ffprobe_frames(clip_path)
    video_stream = _video_stream(stream_payload, clip_path)
    legacy = _legacy_frame_time_table(clip_path, stream_payload, frame_payload)
    raw_frames = frame_payload.get("frames")
    if not isinstance(raw_frames, Sequence) or isinstance(raw_frames, (str, bytes)):
        raw_frames = []
    try:
        time_base = _time_base_ticks(video_stream)
    except TimebaseValidationError:
        time_base = None

    observations: list[tuple[RawEncodedPTS | None, dict[str, Any]]] = []
    for frame_index, raw_frame in enumerate(raw_frames):
        if not isinstance(raw_frame, Mapping):
            observations.append((None, {
                "frame_index": frame_index,
                "timestamp_field": None,
                "source_timestamp_decimal": None,
                "conversion_method": "invalid_ffprobe_frame_record",
                "pts_ticks": None,
                "timescale": None,
                "duration_ticks": None,
            }))
            continue
        observations.append(_exact_frame_observation(raw_frame, frame_index=frame_index, time_base=time_base))

    stream_frame_count = _stream_frame_count(video_stream, float(legacy["duration_s"]), float(legacy["fps"]))
    accounted_frame_count = max(stream_frame_count, len(observations))
    if isinstance(capture_sidecar, (str, Path)):
        sidecar_payload: Mapping[str, Any] | None = json.loads(Path(capture_sidecar).read_text(encoding="utf-8"))
    else:
        sidecar_payload = capture_sidecar
    sensor_outcome: SensorClockMappingOutcome = build_sensor_clock_mapping_from_sidecar(sidecar_payload)
    rolling_shutter_outcome: RollingShutterModelOutcome = build_rolling_shutter_model_from_sidecar(
        sidecar_payload
    )

    has_exact_pts = any(raw is not None for raw, _meta in observations)
    contract: TimebaseContract | None = None
    unavailable_indices: list[int] = []
    non_monotonic_indices: list[int] = []
    if has_exact_pts:
        contract_frames: list[FrameTime] = []
        previous_raw: RawEncodedPTS | None = None
        legacy_pts = [float(item["pts_s"]) for item in legacy["frames"]]
        first_accepted_seconds: float | None = None
        for frame_index in range(accounted_frame_count):
            raw = observations[frame_index][0] if frame_index < len(observations) else None
            if raw is None:
                unavailable_indices.append(frame_index)
                contract_frames.append(FrameTime(
                    frame_index,
                    FrameAvailability(FrameAvailabilityStatus.MISSING, FrameAbsenceReason.PTS_UNAVAILABLE),
                    None,
                ))
                continue
            if previous_raw is not None and raw.pts_ticks * previous_raw.timescale <= previous_raw.pts_ticks * raw.timescale:
                non_monotonic_indices.append(frame_index)
                contract_frames.append(FrameTime(
                    frame_index,
                    FrameAvailability(FrameAvailabilityStatus.DROPPED, FrameAbsenceReason.PTS_UNAVAILABLE),
                    None,
                ))
                continue
            previous_raw = raw
            if first_accepted_seconds is None:
                first_accepted_seconds = raw.seconds
            if legacy["provenance"] == "ffprobe_pts" and frame_index < len(legacy_pts):
                corrected_time_s = legacy_pts[frame_index]
                method = "legacy_rebase_to_first_table_pts_round9"
            else:
                corrected_time_s = round(raw.seconds - first_accepted_seconds, 9)
                method = "rebase_to_first_frame_round9"
            correction = CorrectionProvenance(
                method=method,
                source_clock=ClockDomain.MEDIA,
                target_clock=ClockDomain.MEDIA,
                model_id="ffprobe_pts_rebase_round9_v1",
                offset_s=corrected_time_s - raw.seconds,
                drift_ppm=0.0,
                uncertainty_s=0.5e-9,
            )
            contract_frames.append(FrameTime(
                frame_index,
                FrameAvailability(FrameAvailabilityStatus.PRESENT),
                raw,
                CorrectedPTS(corrected_time_s, correction),
            ))
        mappings = (
            (sensor_outcome.mapping,)
            if sensor_outcome.status is SensorClockMappingStatus.MAPPED and sensor_outcome.mapping is not None
            else ()
        )
        contract = TimebaseContract(
            capture_id=capture_id,
            frames=tuple(contract_frames),
            audio_times=(),
            acoustic_models=(),
            sensor_clock_mappings=mappings,
            rolling_shutter_model=rolling_shutter_outcome.model,
        )

    evidence = {
        "schema_version": 1,
        "artifact_type": "racketsport_timebase_decode_evidence",
        "capture_id": capture_id,
        "clip_path": str(clip_path),
        "timing_declaration": (
            "ffprobe_pts_with_explicit_availability" if contract is not None else "constant_fps_assumed"
        ),
        "raw_pts_authority": contract is not None,
        "timebase_contract_emitted": contract is not None,
        "raw_pts_observations": [meta for _raw, meta in observations],
        "count_consistency": {
            "stream_frame_count": stream_frame_count,
            "ffprobe_reported_frame_count": len(observations),
            "canonical_accounted_frame_count": accounted_frame_count,
            "legacy_frame_count": int(legacy["frame_count"]),
            "unavailable_frame_indices": unavailable_indices,
            "non_monotonic_frame_indices": non_monotonic_indices,
            "all_source_frames_accounted": accounted_frame_count >= max(stream_frame_count, len(observations)),
            "legacy_difference_explanation": (
                None
                if int(legacy["frame_count"]) == accounted_frame_count
                else "legacy extraction semantics retained byte-for-byte; canonical frames independently account for ffprobe/stream records"
            ),
        },
        "legacy_compatibility": {
            "artifact": "frame_times.json",
            "values_unchanged": True,
            "derived_method": (
                "legacy_rebase_to_first_table_pts_round9"
                if legacy["provenance"] == "ffprobe_pts"
                else "constant_fps_assumed"
            ),
        },
        "sensor_clock_mapping_outcome": sensor_outcome.to_dict(),
        "rolling_shutter_model_outcome": rolling_shutter_outcome.to_dict(),
    }
    return TimebaseArtifactBuild(legacy, contract, evidence)


def build_frame_time_table(path: str | Path) -> dict[str, Any]:
    """Build a per-frame presentation timestamp table for a clip.

    The preferred path is ffprobe's per-frame PTS/best-effort timestamp output.
    When timestamps are unavailable, the table is still emitted from the
    stream's r_frame_rate with ``provenance=constant_fps_assumed`` so downstream
    consumers can treat that timing as lower trust instead of silently assuming
    CFR.
    """

    clip_path = Path(path)
    if not clip_path.exists():
        raise FileNotFoundError(clip_path)

    stream_payload = _run_ffprobe(clip_path)
    return _legacy_frame_time_table(clip_path, stream_payload, _run_ffprobe_frames(clip_path))


def write_frame_time_table(path: str | Path, out_path: str | Path) -> dict[str, Any]:
    table = build_frame_time_table(path)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(table, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return table


def write_timebase_artifacts(
    path: str | Path,
    frame_times_out_path: str | Path,
    contract_out_path: str | Path,
    evidence_out_path: str | Path,
    *,
    capture_id: str,
    capture_sidecar: Mapping[str, Any] | str | Path | None = None,
) -> TimebaseArtifactBuild:
    build = build_timebase_artifacts(path, capture_id=capture_id, capture_sidecar=capture_sidecar)
    frame_times_out = Path(frame_times_out_path)
    frame_times_out.parent.mkdir(parents=True, exist_ok=True)
    frame_times_out.write_text(
        json.dumps(build.legacy_frame_times, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    evidence_out = Path(evidence_out_path)
    evidence_out.parent.mkdir(parents=True, exist_ok=True)
    evidence_out.write_text(json.dumps(build.evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    contract_out = Path(contract_out_path)
    if build.contract is None:
        contract_out.unlink(missing_ok=True)
    else:
        contract_out.write_bytes(build.contract.to_json_bytes())
    return build


def load_frame_time_table(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("frame time table must contain a JSON object")
    if payload.get("artifact_type") != "racketsport_frame_times":
        raise ValueError("frame time table must have artifact_type='racketsport_frame_times'")
    frame_time_lookup(payload)
    return payload


_FRAME_TIME_LOOKUP_CACHE_MAXSIZE = 32
_FRAME_TIME_LOOKUP_CACHE: "OrderedDict[tuple[Any, ...], tuple[Any, dict[int, float]]]" = OrderedDict()


def _frame_time_lookup_cache_key(frame_times: Any) -> tuple[Any, ...] | None:
    """A cheap-to-compute cache key for inputs worth memoizing, else ``None``.

    Only keys costing O(1) relative to table size are used: a filesystem
    stat tuple for on-disk tables (so a rewritten file on the same path is
    never served stale) and object identity for in-memory containers (the
    identity hit is re-checked against a retained reference before being
    trusted -- see ``_frame_time_lookup_from_cache`` -- so a reused id()
    can never produce a stale result). Anything else falls back to the
    uncached path, unchanged from before.
    """
    if isinstance(frame_times, (str, Path)):
        try:
            resolved = Path(frame_times)
            stat = resolved.stat()
        except OSError:
            return None
        return ("path", str(resolved), stat.st_mtime_ns, stat.st_size)
    if isinstance(frame_times, (Mapping, Sequence)) and not isinstance(frame_times, (str, bytes)):
        return ("id", id(frame_times))
    return None


def _frame_time_lookup_from_cache(frame_times: Any, cache_key: tuple[Any, ...]) -> dict[int, float] | None:
    entry = _FRAME_TIME_LOOKUP_CACHE.get(cache_key)
    if entry is None:
        return None
    cached_source, cached_result = entry
    if cache_key[0] == "id" and cached_source is not frame_times:
        # id() was reused by an unrelated object after the original entry
        # was evicted; treat this as a miss rather than trust a coincidence.
        return None
    _FRAME_TIME_LOOKUP_CACHE.move_to_end(cache_key)
    return cached_result


def _frame_time_lookup_store(frame_times: Any, cache_key: tuple[Any, ...], result: dict[int, float]) -> None:
    # Retain a strong reference to `frame_times` alongside the id()-based key
    # so the source object cannot be garbage-collected (and its id reused by
    # an unrelated object) while the entry is live in the cache.
    _FRAME_TIME_LOOKUP_CACHE[cache_key] = (frame_times, result)
    _FRAME_TIME_LOOKUP_CACHE.move_to_end(cache_key)
    while len(_FRAME_TIME_LOOKUP_CACHE) > _FRAME_TIME_LOOKUP_CACHE_MAXSIZE:
        _FRAME_TIME_LOOKUP_CACHE.popitem(last=False)


def frame_time_lookup(frame_times: Any) -> dict[int, float]:
    """Normalize supported frame-time payloads into ``{frame_index: pts_s}``.

    Perf note: hot-path callers (e.g. ``wasb_csv_to_ball_track``) invoke this
    once per *output frame* with the same ``frame_times`` argument unchanged
    across the whole loop. Re-validating/re-parsing that argument from
    scratch on every call made data builds effectively O(frame_count^2)
    (observed ~10 frames/sec on a real multi-hour build). This function now
    memoizes the normalized result for a bounded set of recently-seen inputs
    so repeat calls with an unchanged table are O(1) after the first call;
    every accept/reject decision and every returned value is still produced
    by the exact same logic in ``_frame_time_lookup_uncached`` below -- only
    the *work* of redoing it is skipped on a cache hit.
    """

    if frame_times is None:
        return {}
    cache_key = _frame_time_lookup_cache_key(frame_times)
    if cache_key is not None:
        cached_result = _frame_time_lookup_from_cache(frame_times, cache_key)
        if cached_result is not None:
            return cached_result
    result = _frame_time_lookup_uncached(frame_times)
    if cache_key is not None:
        _frame_time_lookup_store(frame_times, cache_key, result)
    return result


def _frame_time_lookup_uncached(frame_times: Any) -> dict[int, float]:
    """Byte-for-byte the original (pre-cache) normalization logic."""

    if isinstance(frame_times, (str, Path)):
        return _frame_time_lookup_uncached(load_frame_time_table(frame_times))
    if isinstance(frame_times, Mapping):
        frames = frame_times.get("frames")
        if isinstance(frames, Sequence) and not isinstance(frames, (str, bytes)):
            output: dict[int, float] = {}
            for index, item in enumerate(frames):
                if isinstance(item, Mapping):
                    frame_index = int(item.get("frame", item.get("frame_index", item.get("frame_idx", index))))
                    pts_s = _finite_float_or_none(item.get("pts_s", item.get("t", item.get("time_s"))))
                else:
                    frame_index = index
                    pts_s = _finite_float_or_none(item)
                if pts_s is None or frame_index < 0:
                    continue
                output[frame_index] = float(pts_s)
            return output
        output = {}
        for key, value in frame_times.items():
            try:
                frame_index = int(key)
            except (TypeError, ValueError):
                continue
            pts_s = _finite_float_or_none(value)
            if pts_s is not None and frame_index >= 0:
                output[frame_index] = float(pts_s)
        return output
    if isinstance(frame_times, Sequence) and not isinstance(frame_times, (str, bytes)):
        output = {}
        for index, item in enumerate(frame_times):
            if isinstance(item, Mapping):
                frame_index = int(item.get("frame", item.get("frame_index", item.get("frame_idx", index))))
                pts_s = _finite_float_or_none(item.get("pts_s", item.get("t", item.get("time_s"))))
            else:
                frame_index = index
                pts_s = _finite_float_or_none(item)
            if pts_s is not None and frame_index >= 0:
                output[frame_index] = float(pts_s)
        return output
    raise ValueError("unsupported frame time table shape")


def time_for_frame(
    frame_index: int,
    *,
    frame_times: Any = None,
    fps: float | None = None,
    timebase_contract: TimebaseContract | Mapping[str, Any] | str | Path | None = None,
    return_provenance: bool = False,
) -> float | FrameTimeResolution:
    frame_index = int(frame_index)
    if timebase_contract is not None:
        if isinstance(timebase_contract, TimebaseContract):
            contract = timebase_contract
        elif isinstance(timebase_contract, (str, Path)):
            contract = TimebaseContract.from_json_bytes(Path(timebase_contract).read_bytes())
        elif isinstance(timebase_contract, Mapping):
            contract = TimebaseContract.from_dict(timebase_contract)
        else:
            raise ValueError("unsupported timebase contract shape")
        frame = (
            contract.frames[frame_index]
            if 0 <= frame_index < len(contract.frames)
            and contract.frames[frame_index].frame_index == frame_index
            else next((item for item in contract.frames if item.frame_index == frame_index), None)
        )
        if frame is None:
            raise ValueError(f"timebase contract does not account for frame {frame_index}")
        if frame.availability.status is not FrameAvailabilityStatus.PRESENT:
            reason = frame.availability.reason.value if frame.availability.reason is not None else "unknown"
            raise ValueError(f"timebase contract declares frame {frame_index} unavailable: {reason}")
        if frame.corrected_pts is None:
            raise ValueError(f"timebase contract has no derived corrected PTS for frame {frame_index}")
        result = FrameTimeResolution(
            time_s=frame.time_s(TimeBasis.CORRECTED_PTS),
            time_basis=TimeBasis.CORRECTED_PTS.value,
            provenance=frame.corrected_pts.provenance.method,
            fallback_used=False,
        )
        return result if return_provenance else result.time_s

    lookup = frame_time_lookup(frame_times)
    if frame_index in lookup:
        result = FrameTimeResolution(lookup[frame_index], "legacy_pts_s", "frame_times_table", False)
        return result if return_provenance else result.time_s
    if lookup:
        raise ValueError(f"frame_times table does not contain frame {frame_index}; refusing silent CFR fallback")
    if fps is None or fps <= 0:
        raise ValueError("fps is required when frame_times does not contain frame")
    result = FrameTimeResolution(
        frame_index / float(fps),
        "constant_fps_assumed",
        "explicit_fps_argument",
        True,
    )
    return result if return_provenance else result.time_s


def nearest_frame_for_time(time_s: float, *, frame_times: Any = None, fps: float | None = None) -> int:
    lookup = frame_time_lookup(frame_times)
    if lookup:
        return min(lookup, key=lambda frame: (abs(lookup[frame] - float(time_s)), frame))
    if fps is None or fps <= 0:
        raise ValueError("fps is required when frame_times is unavailable")
    return max(0, int(round(float(time_s) * float(fps))))


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
    """Compatibility alias for ``probe_clip`` metadata-only clip inspection."""

    return probe_clip(path, fps_out=fps_out)

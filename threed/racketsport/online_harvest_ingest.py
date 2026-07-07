"""Online-harvest rally ingest and prelabel plumbing.

This module is deliberately CPU-only. It registers provenance, clips fixed-camera
online games into review-ready rally snippets, and writes prelabel dispatch
manifests without creating persistent ReID galleries or appearance profiles.
"""

from __future__ import annotations

import csv
import copy
import hashlib
import json
import math
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .eval_guard import PROTECTED_EVAL_CLIPS
from .io_decode import FrameSource, probe_clip
from .schemas import BALL_VISIBILITY_LEVELS

try:  # pragma: no cover - exercised by live ingest, unit tests use pure helpers.
    import numpy as _np
except Exception:  # pragma: no cover
    _np = None


ARTIFACT_TYPE = "racketsport_online_harvest_ingest"
LICENSE_NOTE = "owner-ruled private internal use 2026-07-06"
BIOMETRIC_POLICY = {
    "session_only_tracking_for_non_owner_people": True,
    "persistent_reid_galleries_allowed": False,
    "face_embeddings_allowed": False,
    "appearance_embedding_profiles_allowed": False,
    "roles_are_clip_level_metadata": True,
    "note": "No persistent ReID galleries, face/appearance embeddings, or biometric profiles from harvest video.",
}


@dataclass(frozen=True)
class HarvestSource:
    source_id: str
    title: str
    channel: str
    url: str
    video_path: Path
    duration_s: float
    width: int
    height: int
    fps: float
    bytes: int | None
    manifest_status: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "title": self.title,
            "channel": self.channel,
            "url": self.url,
            "video_path": str(self.video_path),
            "duration_s": self.duration_s,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "bytes": self.bytes,
            "manifest_status": self.manifest_status,
        }


@dataclass(frozen=True)
class ActivityBin:
    time_s: float
    motion_score: float
    audio_score: float


@dataclass(frozen=True)
class RallySegment:
    start_s: float
    end_s: float
    sources: tuple[str, ...]
    motion_bin_count: int
    audio_bin_count: int
    max_motion_score: float
    max_audio_score: float

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_s - self.start_s)

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_s": round(self.start_s, 6),
            "end_s": round(self.end_s, 6),
            "duration_s": round(self.duration_s, 6),
            "sources": list(self.sources),
            "motion_bin_count": self.motion_bin_count,
            "audio_bin_count": self.audio_bin_count,
            "max_motion_score": round(self.max_motion_score, 6),
            "max_audio_score": round(self.max_audio_score, 6),
        }


@dataclass(frozen=True)
class HarvestClip:
    clip_id: str
    source: HarvestSource
    start_s: float
    end_s: float
    duration_s: float
    clip_path: Path | None = None
    provenance_path: Path | None = None

    def to_dict(self, *, role: str | None = None) -> dict[str, Any]:
        payload = {
            "clip_id": self.clip_id,
            "source_id": self.source.source_id,
            "source_channel": self.source.channel,
            "source_title": self.source.title,
            "start_s": round(self.start_s, 6),
            "end_s": round(self.end_s, 6),
            "duration_s": round(self.duration_s, 6),
            "clip_path": str(self.clip_path) if self.clip_path is not None else None,
            "provenance_path": str(self.provenance_path) if self.provenance_path is not None else None,
        }
        if role is not None:
            payload["role"] = role
        return payload


@dataclass(frozen=True)
class RoleAssignment:
    clip_roles: dict[str, str]
    heldout_proposals: list[dict[str, Any]]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_harvest_sources(manifest_path: str | Path, *, harvest_root: str | Path) -> list[HarvestSource]:
    """Load downloaded rows from the wave-1 harvest manifest and probe missing width/fps."""

    root = Path(harvest_root)
    payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("online harvest manifest must be a JSON array")
    sources: list[HarvestSource] = []
    for row in payload:
        if not isinstance(row, Mapping) or row.get("status") != "downloaded":
            continue
        file_value = row.get("file")
        if not file_value:
            raise ValueError(f"downloaded manifest row {row.get('id')!r} is missing file")
        video_path = root / str(file_value)
        if not video_path.is_file():
            raise FileNotFoundError(video_path)
        probe = probe_clip(video_path)
        sources.append(
            HarvestSource(
                source_id=str(row["id"]),
                title=str(row.get("title") or ""),
                channel=str(row.get("channel") or ""),
                url=str(row.get("url") or ""),
                video_path=video_path,
                duration_s=float(row.get("duration_s") or probe.duration_s),
                width=probe.width,
                height=probe.height,
                fps=float(row.get("fps") or probe.fps),
                bytes=int(row["bytes"]) if row.get("bytes") is not None else None,
                manifest_status=str(row.get("status") or ""),
            )
        )
    return sources


def segments_from_activity_bins(
    bins: Sequence[ActivityBin],
    *,
    duration_s: float,
    pad_s: float = 1.5,
    max_active_gap_s: float = 2.0,
    merge_gap_s: float = 2.5,
    min_segment_s: float = 4.0,
    min_motion_score: float = 0.20,
    audio_score_threshold: float = 0.55,
    audio_motion_floor: float = 0.08,
    audio_motion_context_s: float | None = None,
    min_motion_bins: int = 2,
) -> list[RallySegment]:
    """Fuse low-fps motion activity and audio-onset density into rally-like spans.

    Audio is allowed to extend nearby motion, but isolated audio-only chatter is
    filtered by requiring either local motion or a nearby motion-active bin.
    """

    if duration_s <= 0:
        raise ValueError("duration_s must be positive")
    if not bins:
        return []
    context_s = max_active_gap_s if audio_motion_context_s is None else audio_motion_context_s
    ordered = sorted(bins, key=lambda item: item.time_s)
    motion_times = [item.time_s for item in ordered if item.motion_score >= min_motion_score]

    active: list[tuple[ActivityBin, set[str]]] = []
    for item in ordered:
        sources: set[str] = set()
        if item.motion_score >= min_motion_score:
            sources.add("motion_activity")
        nearby_motion = any(abs(item.time_s - motion_t) <= context_s for motion_t in motion_times)
        if item.audio_score >= audio_score_threshold and (item.motion_score >= audio_motion_floor or nearby_motion):
            sources.add("audio_onset_density")
        if sources:
            active.append((item, sources))
    if not active:
        return []

    groups: list[list[tuple[ActivityBin, set[str]]]] = []
    current = [active[0]]
    for item, sources in active[1:]:
        if item.time_s - current[-1][0].time_s <= max_active_gap_s:
            current.append((item, sources))
        else:
            groups.append(current)
            current = [(item, sources)]
    groups.append(current)

    raw_segments: list[RallySegment] = []
    for group in groups:
        group_sources = sorted({source for _, sources in group for source in sources})
        motion_bin_count = sum(1 for item, _ in group if item.motion_score >= min_motion_score)
        soft_motion_bin_count = sum(1 for item, _ in group if item.motion_score >= audio_motion_floor)
        audio_bin_count = sum(1 for item, sources in group if "audio_onset_density" in sources)
        max_motion = max(item.motion_score for item, _ in group)
        max_audio = max(item.audio_score for item, _ in group)
        start_s = max(0.0, group[0][0].time_s - pad_s)
        end_s = min(duration_s, group[-1][0].time_s + pad_s)
        if end_s - start_s < min_segment_s:
            continue
        if soft_motion_bin_count < min_motion_bins:
            continue
        raw_segments.append(
            RallySegment(
                start_s=start_s,
                end_s=end_s,
                sources=tuple(group_sources),
                motion_bin_count=motion_bin_count,
                audio_bin_count=audio_bin_count,
                max_motion_score=max_motion,
                max_audio_score=max_audio,
            )
        )
    return _merge_rally_segments(raw_segments, merge_gap_s=merge_gap_s, duration_s=duration_s)


def _merge_rally_segments(
    segments: Sequence[RallySegment], *, merge_gap_s: float, duration_s: float
) -> list[RallySegment]:
    if not segments:
        return []
    ordered = sorted(segments, key=lambda item: item.start_s)
    merged: list[RallySegment] = []
    current = ordered[0]
    for segment in ordered[1:]:
        if segment.start_s - current.end_s <= merge_gap_s:
            current = RallySegment(
                start_s=current.start_s,
                end_s=min(duration_s, max(current.end_s, segment.end_s)),
                sources=tuple(sorted(set(current.sources) | set(segment.sources))),
                motion_bin_count=current.motion_bin_count + segment.motion_bin_count,
                audio_bin_count=current.audio_bin_count + segment.audio_bin_count,
                max_motion_score=max(current.max_motion_score, segment.max_motion_score),
                max_audio_score=max(current.max_audio_score, segment.max_audio_score),
            )
        else:
            merged.append(current)
            current = segment
    merged.append(current)
    return merged


def activity_bins_from_video(
    video_path: str | Path,
    *,
    source: FrameSource | None = None,
    bin_s: float = 0.5,
    motion_sample_fps: float = 2.0,
    motion_width: int = 160,
    audio_sample_rate_hz: int = 8_000,
) -> tuple[list[ActivityBin], dict[str, Any]]:
    """Compute cheap fixed-camera activity bins from ffmpeg-decoded video/audio."""

    video = Path(video_path)
    source = source or probe_clip(video)
    duration_s = source.duration_s
    motion_scores = _motion_scores(video, source=source, sample_fps=motion_sample_fps, width=motion_width)
    audio_scores, audio_meta = _audio_scores(video, duration_s=duration_s, bin_s=bin_s, sample_rate_hz=audio_sample_rate_hz)
    bin_count = max(1, int(math.ceil(duration_s / bin_s)))
    bins: list[ActivityBin] = []
    for idx in range(bin_count):
        t = idx * bin_s
        motion_idx = min(len(motion_scores) - 1, max(0, int(round(t * motion_sample_fps)))) if motion_scores else 0
        audio_idx = min(len(audio_scores) - 1, idx) if audio_scores else 0
        bins.append(
            ActivityBin(
                time_s=round(t, 6),
                motion_score=float(motion_scores[motion_idx]) if motion_scores else 0.0,
                audio_score=float(audio_scores[audio_idx]) if audio_scores else 0.0,
            )
        )
    return bins, {
        "bin_s": bin_s,
        "motion_sample_fps": motion_sample_fps,
        "motion_width": motion_width,
        "audio_sample_rate_hz": audio_sample_rate_hz,
        "audio": audio_meta,
        "bin_count": len(bins),
    }


def _scaled_size(width: int, height: int, target_width: int) -> tuple[int, int]:
    out_w = max(2, min(width, target_width))
    if out_w % 2:
        out_w -= 1
    out_h = max(2, round(height * out_w / width))
    if out_h % 2:
        out_h += 1
    return out_w, out_h


def _motion_scores(video: Path, *, source: FrameSource, sample_fps: float, width: int) -> list[float]:
    out_w, out_h = _scaled_size(source.width, source.height, width)
    frame_bytes = out_w * out_h
    command = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video),
        "-an",
        "-vf",
        f"fps={sample_fps},scale={out_w}:{out_h},format=gray",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "-",
    ]
    completed = _run(command, binary=True)
    raw = completed.stdout
    frame_count = len(raw) // frame_bytes
    if frame_count <= 1:
        return [0.0]
    if _np is not None:
        frames = _np.frombuffer(raw[: frame_count * frame_bytes], dtype=_np.uint8).reshape((frame_count, frame_bytes))
        diffs = _np.mean(_np.abs(frames[1:].astype(_np.int16) - frames[:-1].astype(_np.int16)), axis=1) / 255.0
        values = [0.0] + [float(value) for value in diffs]
    else:  # pragma: no cover
        values = [0.0]
        prev = raw[0:frame_bytes]
        for idx in range(1, frame_count):
            cur = raw[idx * frame_bytes : (idx + 1) * frame_bytes]
            values.append(sum(abs(a - b) for a, b in zip(cur, prev)) / (frame_bytes * 255.0))
            prev = cur
    return _robust_normalize(values)


def _audio_scores(video: Path, *, duration_s: float, bin_s: float, sample_rate_hz: int) -> tuple[list[float], dict[str, Any]]:
    command = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video),
        "-map",
        "0:a:0?",
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate_hz),
        "-f",
        "s16le",
        "-",
    ]
    try:
        completed = _run(command, binary=True)
    except RuntimeError as exc:
        return [], {"status": "blocked", "blocker": str(exc)}
    raw = completed.stdout
    if not raw:
        return [], {"status": "blocked", "blocker": "no_audio_stream_or_empty_decode"}
    samples_per_bin = max(1, int(round(sample_rate_hz * bin_s)))
    if _np is not None:
        samples = _np.frombuffer(raw, dtype="<i2").astype(_np.float32)
        bin_count = max(1, int(math.ceil(len(samples) / samples_per_bin)))
        padded = _np.pad(samples, (0, bin_count * samples_per_bin - len(samples)))
        windows = padded.reshape((bin_count, samples_per_bin))
        energy = _np.sqrt(_np.mean((windows / 32768.0) ** 2, axis=1))
        deltas = _np.maximum(0.0, energy - _np.concatenate([energy[:1], energy[:-1]]))
        energy_norm = _robust_normalize([float(value) for value in energy])
        delta_norm = _robust_normalize([float(value) for value in deltas])
    else:  # pragma: no cover
        values = [int.from_bytes(raw[i : i + 2], "little", signed=True) / 32768.0 for i in range(0, len(raw), 2)]
        energy = []
        for start in range(0, len(values), samples_per_bin):
            chunk = values[start : start + samples_per_bin]
            energy.append(math.sqrt(sum(value * value for value in chunk) / len(chunk)))
        energy_norm = _robust_normalize(energy)
        delta_norm = _robust_normalize([max(0.0, value - (energy[idx - 1] if idx else value)) for idx, value in enumerate(energy)])
    scores = [max(delta, 0.35 * energy) for delta, energy in zip(delta_norm, energy_norm)]
    target_bins = max(1, int(math.ceil(duration_s / bin_s)))
    if len(scores) < target_bins:
        scores.extend([0.0] * (target_bins - len(scores)))
    return scores[:target_bins], {
        "status": "decoded",
        "raw_bytes": len(raw),
        "sample_rate_hz": sample_rate_hz,
        "bin_s": bin_s,
        "score_count": len(scores[:target_bins]),
    }


def _robust_normalize(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    positives = sorted(float(value) for value in values if value > 0.0)
    if not positives:
        return [0.0 for _ in values]
    index = min(len(positives) - 1, max(0, int(round(0.95 * (len(positives) - 1)))))
    scale = positives[index] or max(positives) or 1.0
    return [max(0.0, min(1.0, float(value) / scale)) for value in values]


def _run(command: Sequence[str], *, binary: bool = False) -> subprocess.CompletedProcess[Any]:
    try:
        return subprocess.run(
            list(command),
            check=True,
            capture_output=True,
            text=not binary,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(f"missing executable for command: {command[0]}") from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else str(exc.stderr)
        raise RuntimeError(f"command failed ({' '.join(command)}): {stderr.strip()}") from exc


def extract_segment_stream_copy(source_video: str | Path, out_path: str | Path, *, start_s: float, end_s: float) -> None:
    if end_s <= start_s:
        raise ValueError("end_s must be greater than start_s")
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{start_s:.3f}",
        "-to",
        f"{end_s:.3f}",
        "-i",
        str(source_video),
        "-map",
        "0",
        "-c",
        "copy",
        "-avoid_negative_ts",
        "make_zero",
        str(out),
    ]
    _run(command, binary=True)


def write_clip_provenance(
    path: str | Path,
    *,
    clip: HarvestClip,
    segment: RallySegment,
    role: str,
    source_sha256: str,
) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "artifact_type": "racketsport_online_harvest_clip_provenance",
        "clip_id": clip.clip_id,
        "source": clip.source.to_dict(),
        "source_sha256": source_sha256,
        "role": role,
        "role_scope": "clip_level_metadata",
        "timestamp_range_s": [round(clip.start_s, 6), round(clip.end_s, 6)],
        "rally_segment": segment.to_dict(),
        "resolution": [clip.source.width, clip.source.height],
        "fps": clip.source.fps,
        "license_note": LICENSE_NOTE,
        "biometric_policy": BIOMETRIC_POLICY,
        "created_at_utc": _utc_now(),
    }
    _write_json(path, payload)
    return payload


def assign_clip_roles(
    clips: Sequence[HarvestClip],
    *,
    proposed_heldout_source_ids: Sequence[str],
    internal_val_modulo: int = 5,
) -> RoleAssignment:
    if len(set(proposed_heldout_source_ids)) != 2:
        raise ValueError("exactly two distinct proposed held-out source ids are required")
    sources_by_id = {clip.source.source_id: clip.source for clip in clips}
    missing = set(proposed_heldout_source_ids) - set(sources_by_id)
    if missing:
        raise ValueError(f"held-out proposals not present in clips: {sorted(missing)}")
    proposed_sources = [sources_by_id[source_id] for source_id in proposed_heldout_source_ids]
    if len({source.channel for source in proposed_sources}) != 2:
        raise ValueError("held-out candidate games must come from distinct channels")
    if internal_val_modulo <= 0:
        raise ValueError("internal_val_modulo must be positive")

    clip_roles: dict[str, str] = {}
    grouped: dict[str, list[HarvestClip]] = defaultdict(list)
    for clip in clips:
        grouped[clip.source.source_id].append(clip)
    for source_id, source_clips in grouped.items():
        ordered = sorted(source_clips, key=lambda clip: (clip.start_s, clip.clip_id))
        for index, clip in enumerate(ordered, start=1):
            if source_id in proposed_heldout_source_ids:
                clip_roles[clip.clip_id] = "heldout_candidate_proposed"
            elif (index - 1) % internal_val_modulo == 0:
                clip_roles[clip.clip_id] = "internal_val"
            else:
                clip_roles[clip.clip_id] = "train"

    heldout_proposals = [
        {
            "source_id": source.source_id,
            "channel": source.channel,
            "title": source.title,
            "duration_s": source.duration_s,
            "resolution": [source.width, source.height],
            "fps": source.fps,
            "rationale": "distinct channel/court context; reserved from prelabel and training until manager ledger registration",
            "ledger_action": "manager_registers_only; this lane did not write heldout_eval_ledger.md",
        }
        for source in proposed_sources
    ]
    return RoleAssignment(clip_roles=clip_roles, heldout_proposals=heldout_proposals)


def build_prelabel_shard_manifest(
    clips: Sequence[HarvestClip],
    roles: RoleAssignment,
    *,
    shard_size: int = 1,
) -> dict[str, Any]:
    if shard_size <= 0:
        raise ValueError("shard_size must be positive")
    eligible: list[dict[str, Any]] = []
    excluded = 0
    for clip in sorted(clips, key=lambda item: item.clip_id):
        role = roles.clip_roles[clip.clip_id]
        if role == "heldout_candidate_proposed":
            excluded += 1
            continue
        eligible.append(
            {
                "clip_id": clip.clip_id,
                "source_id": clip.source.source_id,
                "role": role,
                "clip_path": str(clip.clip_path) if clip.clip_path else None,
                "duration_s": round(clip.duration_s, 6),
                "fps": clip.source.fps,
                "max_prelabel_frames_for_cpu_smoke": 50,
                "commands": {
                    "ball_wasb_gpu": (
                        "python scripts/racketsport/run_wasb_ball.py "
                        f"--video {clip.clip_path} --fps {clip.source.fps:.6f} "
                        "--checkpoint <A100_WASB_CHECKPOINT> --wasb-repo <A100_WASB_REPO> "
                        f"--out <PRELABEL_OUT>/{clip.clip_id}/ball_track.json "
                        f"--metadata-out <PRELABEL_OUT>/{clip.clip_id}/ball_track_metadata.json "
                        "--device cuda"
                    ),
                    "person_session_only": (
                        "session-only person detection/tracking only; do not persist ReID galleries, "
                        "face embeddings, appearance embeddings, or biometric profiles for harvest video"
                    ),
                },
            }
        )
    shards = []
    for index in range(0, len(eligible), shard_size):
        items = eligible[index : index + shard_size]
        shards.append(
            {
                "shard_id": f"shard_{len(shards) + 1:04d}",
                "gpu_friendly": True,
                "one_clip_per_job": shard_size == 1,
                "items": items,
            }
        )
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_online_harvest_prelabel_shard_manifest",
        "status": "prelabel_ready",
        "biometric_policy": BIOMETRIC_POLICY,
        "summary": {
            "eligible_clip_count": len(eligible),
            "excluded_heldout_candidate_clip_count": excluded,
            "shard_count": len(shards),
            "shard_size": shard_size,
        },
        "shards": shards,
    }


def compare_hash_sets(
    *,
    left_name: str,
    left_hashes: Mapping[str, Sequence[int]],
    right_name: str,
    right_hashes: Mapping[str, Sequence[int]],
    threshold: int,
) -> list[dict[str, Any]]:
    collisions: list[dict[str, Any]] = []
    relation = f"{left_name}_vs_{right_name}"
    for left_group, left_values in sorted(left_hashes.items()):
        for right_group, right_values in sorted(right_hashes.items()):
            for left_hash in left_values:
                for right_hash in right_values:
                    distance = int(left_hash ^ right_hash).bit_count()
                    if distance <= threshold:
                        collisions.append(
                            {
                                "left_group": left_group,
                                "left_hash": f"{left_hash:016x}",
                                "right_group": right_group,
                                "right_hash": f"{right_hash:016x}",
                                "hamming_distance": distance,
                                "relation": relation,
                            }
                        )
    return sorted(collisions, key=lambda item: (item["hamming_distance"], item["left_group"], item["right_group"]))


def dedupe_sources_from_manifest(collisions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    eval_collisions = [item for item in collisions if str(item.get("relation", "")).endswith("_vs_eval")]
    cross_collisions = [item for item in collisions if "cross_source" in str(item.get("relation", ""))]
    return {
        "eval_collision_count": len(eval_collisions),
        "cross_source_collision_count": len(cross_collisions),
        "collision_count": len(collisions),
        "collisions": [dict(item) for item in collisions],
    }


def perceptual_hash_video(
    video_path: str | Path,
    *,
    sample_every_s: float = 2.0,
    max_frames: int | None = None,
    hash_size: int = 8,
) -> list[int]:
    """Compute deterministic 64-bit dHash samples using ffmpeg raw grayscale frames."""

    video = Path(video_path)
    width = hash_size + 1
    height = hash_size
    vf = f"fps=1/{sample_every_s},scale={width}:{height},format=gray"
    command = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video),
        "-an",
        "-vf",
        vf,
        *([] if max_frames is None else ["-frames:v", str(max_frames)]),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "-",
    ]
    completed = _run(command, binary=True)
    raw = completed.stdout
    frame_bytes = width * height
    frame_count = len(raw) // frame_bytes
    hashes: list[int] = []
    for frame_idx in range(frame_count):
        frame = raw[frame_idx * frame_bytes : (frame_idx + 1) * frame_bytes]
        value = 0
        bit = 0
        for row in range(height):
            offset = row * width
            for col in range(hash_size):
                if frame[offset + col] > frame[offset + col + 1]:
                    value |= 1 << bit
                bit += 1
        hashes.append(value)
    return hashes


def build_dedup_report(
    *,
    harvest_sources: Sequence[HarvestSource],
    eval_root: str | Path = "eval_clips/ball",
    hash_sample_every_s: float = 2.0,
    threshold: int = 3,
) -> dict[str, Any]:
    harvest_hashes = {
        source.source_id: perceptual_hash_video(source.video_path, sample_every_s=hash_sample_every_s)
        for source in harvest_sources
    }
    eval_hashes: dict[str, list[int]] = {}
    for clip in PROTECTED_EVAL_CLIPS:
        path = Path(eval_root) / clip.clip_id / "source.mp4"
        if path.is_file():
            eval_hashes[clip.clip_id] = perceptual_hash_video(path, sample_every_s=hash_sample_every_s)
    eval_collisions = compare_hash_sets(
        left_name="harvest",
        left_hashes=harvest_hashes,
        right_name="eval",
        right_hashes=eval_hashes,
        threshold=threshold,
    )
    cross_source_collisions: list[dict[str, Any]] = []
    for left_index, left in enumerate(harvest_sources):
        for right in harvest_sources[left_index + 1 :]:
            cross_source_collisions.extend(
                compare_hash_sets(
                    left_name="harvest_cross_source",
                    left_hashes={left.source_id: harvest_hashes[left.source_id]},
                    right_name="harvest_cross_source",
                    right_hashes={right.source_id: harvest_hashes[right.source_id]},
                    threshold=threshold,
                )
            )
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_online_harvest_dedup_report",
        "hash": {
            "type": "dhash_8x8_64bit",
            "sample_every_s": hash_sample_every_s,
            "collision_hamming_threshold": threshold,
        },
        "harvest_hash_counts": {key: len(value) for key, value in harvest_hashes.items()},
        "eval_hash_counts": {key: len(value) for key, value in eval_hashes.items()},
        "summary": dedupe_sources_from_manifest([*eval_collisions, *cross_source_collisions]),
    }


def select_review_subset(
    clips: Sequence[HarvestClip],
    roles: Mapping[str, str],
    *,
    frame_budget: int = 480,
) -> dict[str, Any]:
    if frame_budget <= 0:
        raise ValueError("frame_budget must be positive")
    candidates = [clip for clip in clips if roles[clip.clip_id] != "heldout_candidate_proposed"]
    by_source: dict[str, list[HarvestClip]] = defaultdict(list)
    for clip in candidates:
        by_source[clip.source.source_id].append(clip)
    selected: list[dict[str, Any]] = []
    remaining_frames = frame_budget
    for source_id in sorted(by_source):
        source_clips = sorted(by_source[source_id], key=lambda clip: (-clip.duration_s, clip.start_s))
        if not source_clips or remaining_frames <= 0:
            continue
        clip = source_clips[0]
        frames = min(80, max(1, int(round(clip.duration_s * clip.source.fps))), remaining_frames)
        selected.append(
            {
                "clip_id": clip.clip_id,
                "source_id": source_id,
                "role": roles[clip.clip_id],
                "clip_path": str(clip.clip_path) if clip.clip_path else None,
                "start_s": round(clip.start_s, 6),
                "end_s": round(clip.end_s, 6),
                "frame_budget": frames,
                "reason": "longest non-heldout rally per source for diverse first review pass",
            }
        )
        remaining_frames -= frames
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_online_harvest_cvat_review_subset",
        "status": "selection_ready",
        "frame_budget": frame_budget,
        "selected_frame_budget": sum(item["frame_budget"] for item in selected),
        "selected": selected,
    }


def four_level_visibility_schema_available(schema_module_path: str | Path = "threed/racketsport/schemas/__init__.py") -> bool:
    text = Path(schema_module_path).read_text(encoding="utf-8")
    needles = ("BallVisibilityLevel", "visibility_level", "clear", "partial", "full", "out_of_frame")
    return all(needle in text for needle in needles)


def write_cvat_review_task_package(
    review_subset: Mapping[str, Any],
    *,
    out_root: str | Path,
    heldout_source_ids: Sequence[str],
) -> dict[str, Any]:
    """Write JSON task definitions for the online-harvest review subset."""

    selected = _selected_review_rows(review_subset)
    _assert_review_subset_excludes_heldout(selected, heldout_source_ids=heldout_source_ids)
    labels = _cvat_labels_with_four_level_visibility()
    out = Path(out_root)
    tasks_root = out / "tasks"
    tasks_root.mkdir(parents=True, exist_ok=True)
    tasks: list[dict[str, Any]] = []
    for row in selected:
        clip_id = str(row["clip_id"])
        source_id = str(row["source_id"])
        task_dir = tasks_root / clip_id
        task_dir.mkdir(parents=True, exist_ok=True)
        task_path = task_dir / "task.json"
        task_payload = {
            "schema_version": 1,
            "artifact_type": "racketsport_online_harvest_cvat_task",
            "status": "ready_for_cvat_review",
            "task_name": f"racketsport_online_harvest_{clip_id}",
            "clip_id": clip_id,
            "source_id": source_id,
            "role": str(row.get("role") or ""),
            "clip_path": row.get("clip_path"),
            "timestamp_range_s": [row.get("start_s"), row.get("end_s")],
            "frame_budget": int(row["frame_budget"]),
            "labels": labels,
            "ball_visibility_levels": list(BALL_VISIBILITY_LEVELS),
            "visibility_instructions": {
                "clear": "ball is visible and localizable without material occlusion",
                "partial": "ball is localizable but partly occluded or blurred",
                "full": "ball is expected in-frame but fully occluded",
                "out_of_frame": "ball is outside the image bounds",
            },
            "heldout_excluded": True,
            "not_ground_truth": True,
        }
        _write_json(task_path, task_payload)
        tasks.append(
            {
                "clip_id": clip_id,
                "source_id": source_id,
                "role": task_payload["role"],
                "clip_path": task_payload["clip_path"],
                "frame_budget": task_payload["frame_budget"],
                "task_definition_path": str(task_path),
            }
        )

    manifest_path = out / "cvat_task_manifest.json"
    manifest = {
        "schema_version": 1,
        "artifact_type": "racketsport_online_harvest_cvat_task_export",
        "status": "ready_for_cvat_review" if tasks else "no_review_items",
        "review_subset_status": review_subset.get("status"),
        "heldout_source_ids": list(heldout_source_ids),
        "task_count": len(tasks),
        "visibility_levels": list(BALL_VISIBILITY_LEVELS),
        "tasks": tasks,
        "not_ground_truth": True,
    }
    _write_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def validate_cvat_review_task_package(
    manifest_path: str | Path,
    *,
    heldout_source_ids: Sequence[str],
) -> dict[str, Any]:
    """Import the generated task package and validate schema-sensitive fields."""

    manifest_file = Path(manifest_path)
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping):
        raise ValueError(f"expected task manifest object: {manifest_file}")
    expected_levels = list(BALL_VISIBILITY_LEVELS)
    heldout_set = set(heldout_source_ids)
    errors: list[str] = []
    task_rows = manifest.get("tasks", [])
    if not isinstance(task_rows, list):
        errors.append("tasks must be a list")
        task_rows = []
    levels_seen: set[str] = set()
    validated_tasks: list[dict[str, Any]] = []
    for index, task_row in enumerate(task_rows):
        if not isinstance(task_row, Mapping):
            errors.append(f"task row {index} is not an object")
            continue
        source_id = str(task_row.get("source_id") or "")
        if source_id in heldout_set:
            errors.append(f"held-out source leaked into CVAT export: {source_id}")
        task_path_value = task_row.get("task_definition_path")
        if not task_path_value:
            errors.append(f"task row {index} missing task_definition_path")
            continue
        task_path = Path(str(task_path_value))
        if not task_path.is_file():
            errors.append(f"missing task definition: {task_path}")
            continue
        task_payload = json.loads(task_path.read_text(encoding="utf-8"))
        if not isinstance(task_payload, Mapping):
            errors.append(f"task definition is not an object: {task_path}")
            continue
        labels = task_payload.get("labels")
        if not isinstance(labels, list):
            errors.append(f"task labels missing or invalid: {task_path}")
            continue
        ball_labels = [label for label in labels if isinstance(label, Mapping) and label.get("name") == "ball"]
        if len(ball_labels) != 1:
            errors.append(f"expected exactly one ball label in {task_path}")
            continue
        values = ball_labels[0].get("attribute_values", {}).get("visibility_level")
        if values != expected_levels:
            errors.append(f"ball visibility levels mismatch in {task_path}: {values}")
            continue
        levels_seen.update(str(value) for value in values)
        validated_tasks.append(
            {
                "clip_id": str(task_payload.get("clip_id") or task_row.get("clip_id") or ""),
                "source_id": source_id,
                "task_definition_path": str(task_path),
            }
        )
    status = "passed" if not errors else "failed"
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_online_harvest_cvat_task_validation",
        "status": status,
        "manifest_path": str(manifest_file),
        "task_count": len(task_rows),
        "validated_task_count": len(validated_tasks),
        "heldout_source_ids": list(heldout_source_ids),
        "visibility_levels": sorted(levels_seen),
        "errors": errors,
        "validated_tasks": validated_tasks,
    }


def _selected_review_rows(review_subset: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    selected = review_subset.get("selected", [])
    if not isinstance(selected, list):
        raise ValueError("review subset selected must be a list")
    rows: list[Mapping[str, Any]] = []
    for index, item in enumerate(selected):
        if not isinstance(item, Mapping):
            raise ValueError(f"review subset item {index} must be an object")
        for key in ("clip_id", "source_id", "frame_budget"):
            if key not in item:
                raise ValueError(f"review subset item {index} missing {key}")
        rows.append(item)
    return rows


def _assert_review_subset_excludes_heldout(
    selected: Sequence[Mapping[str, Any]],
    *,
    heldout_source_ids: Sequence[str],
) -> None:
    heldout_set = set(heldout_source_ids)
    leaked = sorted({str(item.get("source_id")) for item in selected if str(item.get("source_id")) in heldout_set})
    if leaked:
        raise ValueError(f"held-out source ids must remain excluded from CVAT export: {leaked}")


def _cvat_labels_with_four_level_visibility() -> list[dict[str, Any]]:
    from .label_review import CVAT_LABELS

    labels = copy.deepcopy(CVAT_LABELS)
    ball_labels = [label for label in labels if label.get("name") == "ball"]
    if len(ball_labels) != 1:
        raise RuntimeError("CVAT labels must include exactly one ball label")
    values = ball_labels[0].setdefault("attribute_values", {}).get("visibility_level")
    if values != list(BALL_VISIBILITY_LEVELS):
        raise RuntimeError(f"CVAT ball visibility levels do not match schema: {values}")
    return labels


def write_prelabel_cpu_smoke(
    *,
    clip: HarvestClip,
    out_dir: str | Path,
    max_frames: int = 50,
) -> dict[str, Any]:
    """Run a CPU-only ball prelabel plumbing smoke through run_wasb_ball.py's conversion path."""

    if max_frames <= 0 or max_frames > 50:
        raise ValueError("max_frames must be in 1..50")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "smoke_wasb_predictions.csv"
    ball_track_path = out / "ball_track.json"
    metadata_path = out / "ball_track_metadata.json"
    rows = [
        {
            "Frame": frame,
            "Visibility": 1 if frame % 5 == 0 else 0,
            "X": 320 + (frame % 7),
            "Y": 240 + (frame % 5),
            "Confidence": 0.75 if frame % 5 == 0 else 0.10,
        }
        for frame in range(max_frames)
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Frame", "Visibility", "X", "Y", "Confidence"])
        writer.writeheader()
        writer.writerows(rows)
    command = [
        sys.executable,
        "scripts/racketsport/run_wasb_ball.py",
        "--predictions-csv",
        str(csv_path),
        "--fps",
        f"{clip.source.fps:.6f}",
        "--out",
        str(ball_track_path),
        "--metadata-out",
        str(metadata_path),
        "--device",
        "cpu",
    ]
    started = time.perf_counter()
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    elapsed_s = time.perf_counter() - started
    status = "passed" if completed.returncode == 0 and ball_track_path.is_file() and metadata_path.is_file() else "failed"
    report = {
        "schema_version": 1,
        "artifact_type": "racketsport_online_harvest_prelabel_cpu_smoke",
        "status": status,
        "clip_id": clip.clip_id,
        "max_frames": max_frames,
        "command": " ".join(command),
        "returncode": completed.returncode,
        "elapsed_s": round(elapsed_s, 6),
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "outputs": {
            "predictions_csv": str(csv_path),
            "ball_track": str(ball_track_path),
            "metadata": str(metadata_path),
        },
        "notes": [
            "CPU smoke validates the real run_wasb_ball.py prelabel conversion entry point without CUDA.",
            "GPU heatmap inference remains manager-dispatched later from the shard manifest.",
        ],
    }
    _write_json(out / "prelabel_cpu_smoke.json", report)
    return report


def process_source_to_clips(
    source: HarvestSource,
    *,
    data_out_root: str | Path,
    lane_out_root: str | Path,
    skip_extract: bool = False,
    segment_kwargs: Mapping[str, Any] | None = None,
) -> tuple[list[HarvestClip], dict[str, Any]]:
    probe = probe_clip(source.video_path)
    bins, activity_meta = activity_bins_from_video(source.video_path, source=probe)
    segments = segments_from_activity_bins(bins, duration_s=probe.duration_s, **dict(segment_kwargs or {}))
    source_sha = sha256_file(source.video_path)
    clips: list[HarvestClip] = []
    source_lane_dir = Path(lane_out_root) / "sources" / source.source_id
    source_data_dir = Path(data_out_root) / "rallies" / source.source_id
    activity_payload = {
        "schema_version": 1,
        "artifact_type": "racketsport_online_harvest_activity",
        "source": source.to_dict(),
        "activity_meta": activity_meta,
        "bins": [
            {"time_s": item.time_s, "motion_score": round(item.motion_score, 6), "audio_score": round(item.audio_score, 6)}
            for item in bins
        ],
        "segments": [segment.to_dict() for segment in segments],
    }
    _write_json(source_lane_dir / "activity_segments.json", activity_payload)
    for index, segment in enumerate(segments, start=1):
        clip_id = f"{source.source_id}_rally_{index:04d}"
        clip_path = source_data_dir / f"{clip_id}.mp4"
        provenance_path = source_data_dir / f"{clip_id}.provenance.json"
        if not skip_extract:
            extract_segment_stream_copy(source.video_path, clip_path, start_s=segment.start_s, end_s=segment.end_s)
        clips.append(
            HarvestClip(
                clip_id=clip_id,
                source=source,
                start_s=segment.start_s,
                end_s=segment.end_s,
                duration_s=segment.duration_s,
                clip_path=clip_path,
                provenance_path=provenance_path,
            )
        )
    return clips, {"source_sha256": source_sha, "segments": segments, "activity_path": str(source_lane_dir / "activity_segments.json")}


def build_corpus_card(
    *,
    sources: Sequence[HarvestSource],
    clips: Sequence[HarvestClip],
    roles: RoleAssignment,
    dedup_report: Mapping[str, Any],
    heldout_proposals: Sequence[Mapping[str, Any]],
    cvat_status: Mapping[str, Any],
) -> dict[str, Any]:
    by_source: dict[str, list[HarvestClip]] = defaultdict(list)
    for clip in clips:
        by_source[clip.source.source_id].append(clip)
    role_counts: dict[str, int] = defaultdict(int)
    for role in roles.clip_roles.values():
        role_counts[role] += 1
    source_rows = []
    for source in sources:
        source_clips = by_source.get(source.source_id, [])
        source_rows.append(
            {
                **source.to_dict(),
                "clip_count": len(source_clips),
                "rally_minutes": round(sum(clip.duration_s for clip in source_clips) / 60.0, 6),
                "median_rally_duration_s": round(statistics.median([clip.duration_s for clip in source_clips]), 6)
                if source_clips
                else 0.0,
            }
        )
    total_source_minutes = sum(source.duration_s for source in sources) / 60.0
    total_rally_minutes = sum(clip.duration_s for clip in clips) / 60.0
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_online_harvest_corpus_card",
        "created_at_utc": _utc_now(),
        "status": "candidate_corpus_prelabel_ready",
        "license_note": LICENSE_NOTE,
        "biometric_policy": BIOMETRIC_POLICY,
        "summary": {
            "source_count": len(sources),
            "clip_count": len(clips),
            "source_minutes": round(total_source_minutes, 6),
            "rally_minutes": round(total_rally_minutes, 6),
            "rally_coverage_fraction_of_source": round(total_rally_minutes / total_source_minutes, 6)
            if total_source_minutes
            else 0.0,
            "role_counts": dict(sorted(role_counts.items())),
        },
        "sources": source_rows,
        "dedup": dedup_report,
        "heldout_proposals": list(heldout_proposals),
        "cvat_export": dict(cvat_status),
    }


def write_corpus_card_md(path: str | Path, card: Mapping[str, Any]) -> None:
    summary = card["summary"]
    lines = [
        "# Online Harvest Corpus Card",
        "",
        f"- Status: `{card['status']}`",
        f"- License note: {card['license_note']}",
        f"- Sources: {summary['source_count']}",
        f"- Rally clips: {summary['clip_count']}",
        f"- Source minutes: {summary['source_minutes']}",
        f"- Rally minutes: {summary['rally_minutes']}",
        f"- Role counts: `{json.dumps(summary['role_counts'], sort_keys=True)}`",
        f"- Dedup eval collisions: {card['dedup']['summary']['eval_collision_count']}",
        f"- CVAT export: `{card['cvat_export']['status']}`",
        "",
        "## Held-Out Proposals",
    ]
    for proposal in card["heldout_proposals"]:
        lines.append(
            f"- `{proposal['source_id']}` ({proposal['channel']}): {proposal['title']} - {proposal['rationale']}"
        )
    lines.extend(["", "## Sources", "", "| source | channel | min | fps | resolution | clips | rally min |", "|---|---|---:|---:|---|---:|---:|"])
    for source in card["sources"]:
        lines.append(
            f"| `{source['source_id']}` | {source['channel']} | {source['duration_s'] / 60.0:.2f} | "
            f"{source['fps']:.3f} | {source['width']}x{source['height']} | {source['clip_count']} | "
            f"{source['rally_minutes']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Biometric Policy",
            "",
            "- Session-only, non-persistent tracking for non-owner people.",
            "- No persistent ReID galleries, face/appearance embeddings, or biometric profiles from harvest video.",
        ]
    )
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_dispatch_doc(path: str | Path, shard_manifest_path: str | Path, smoke_path: str | Path) -> None:
    text = f"""# Online Harvest Prelabel Dispatch

Shard manifest: `{shard_manifest_path}`
CPU smoke: `{smoke_path}`

Rules:
- Do not prelabel clips whose role is `heldout_candidate_proposed`.
- Run ball prelabels from each shard's `ball_wasb_gpu` command on a CUDA host.
- Person tracking for harvest video must be session-only and non-persistent.
- Do not persist ReID galleries, face embeddings, appearance embeddings, or biometric profiles.
- Candidate predictions remain `candidate_prediction` until reviewed.
"""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")


def build_cvat_status(
    review_subset: Mapping[str, Any],
    *,
    schema_available: bool,
    task_export: Mapping[str, Any] | None = None,
    task_validation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not schema_available:
        return {
            "status": "deferred",
            "cvat_export": "deferred",
            "reason": "4-level visibility schema not present in threed/racketsport/schemas/__init__.py",
            "selection_manifest_status": review_subset["status"],
        }
    if task_export is not None:
        return {
            "status": task_export.get("status", "ready_for_cvat_review"),
            "cvat_export": "task_package_written",
            "selection_manifest_status": review_subset["status"],
            "task_count": task_export.get("task_count"),
            "manifest_path": task_export.get("manifest_path"),
            "validation_status": task_validation.get("status") if task_validation is not None else None,
            "validation_path": task_validation.get("validation_path") if task_validation is not None else None,
            "visibility_levels": list(BALL_VISIBILITY_LEVELS),
        }
    return {
        "status": "selection_ready_export_not_run",
        "cvat_export": "not_run_by_lane",
        "reason": "schema appears available; export task generation left to manager/schema lane handoff",
        "selection_manifest_status": review_subset["status"],
    }


def clips_to_manifest(clips: Sequence[HarvestClip], roles: Mapping[str, str]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_online_harvest_rally_clip_manifest",
        "clip_count": len(clips),
        "clips": [clip.to_dict(role=roles.get(clip.clip_id)) for clip in sorted(clips, key=lambda item: item.clip_id)],
    }

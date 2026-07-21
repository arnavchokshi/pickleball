#!/usr/bin/env python3
"""Build the repair-gated pb.vision event-teacher staging corpus.

pb.vision events are noisy teacher observations, never human ground truth.  The
builder maps the teacher export's explicit frame clock through timestamps to
the nearest encoded source PTS when PTS are available.  Sources without local
media remain non-training placeholders until the VM verifies their PTS and
independent audio-onset/ball-velocity-kink agreement.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import statistics
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_ROOT = ROOT / "data/pbvision_gallery_20260719"
DEFAULT_MEDIA_ROOT = ROOT / "data/pbv_replay_20260720"
DEFAULT_FRAME_TIMES_ROOT = (
    ROOT
    / "runs/lanes/pbv_replay_20260720/vm_pull"
)
DEFAULT_OUTPUT_DIR = ROOT / "runs/lanes/pbv_corpus_rebuild_20260720"
DEFAULT_SEED = 20260720
DEFAULT_WINDOW_FRAMES = 64
DEFAULT_WINDOW_STRIDE = 32
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
LICENSE_POSTURE = "pbvision_signed_full_usage"
SOURCE_NAME = "pbvision_teacher_predictions"
PLAUSIBLE_EVENT_RATE_RANGE = (0.3, 1.0)
TEACHER_CONFIDENCE_MIN = 0.5
AGREEMENT_MAX_DELTA_S = 0.035
FILTERING_POLICY_ID = "pbvision_teacher_filter_v2_20260720"
BUILDER_PATH = Path(__file__).resolve()

# These ids are permanently compare-only across every derivative.
COMPARE_ONLY_HOLDOUTS = {
    "83gyqyc10y8f": (
        "Pre-existing pbvision_11min head-to-head benchmark demo; training on it "
        "would invalidate the comparison."
    ),
    "iottnc0h3ekn": (
        "Gallery venue-diversity hold-out: indoor wood-floor court at 59.94 fps."
    ),
    "o4dee9dn0ccr": (
        "Gallery venue-diversity hold-out: outdoor court with strong shadows."
    ),
}

ABC_WEIGHTING_POLICY = {
    "policy_id": "pbvision_abc_weighting_frozen_20260720",
    "pseudo_weight_by_independent_agreement_count": {
        "0": 0.0,
        "1": 0.25,
        ">=2": 0.5,
    },
    "independent_signal_families": ["audio_onset", "ball_velocity_kink"],
    "pbvision_confidence_alone_weight": 0.0,
    "pbvision_confidence_role": (
        "reject_below_filter_threshold_only; never adds agreement or positive weight"
    ),
    "uncertain_frame_treatment": "ignored_not_background",
    "normalized_aggregate_pseudo_loss_cap_relative_to_human_loss": 1.0,
    "selection_data": "owner_validation_only",
    "arm_a": "zero_teacher_owner_only",
    "arm_b": "agreement_filtered_teacher",
    "arm_c": (
        "same_rows_pixels_classes_weights_and_budget_as_B; shuffle only event time "
        "within immutable source rally"
    ),
}

FILTERING_POLICY = {
    "policy_id": FILTERING_POLICY_ID,
    "teacher_confidence_min_inclusive": TEACHER_CONFIDENCE_MIN,
    "low_confidence_action": "reject_and_mask_unknown",
    "unsupported_net_action": "context_only_and_mask_unknown",
    "pending_agreement_action": "mask_unknown_until_vm_agreement_pass",
    "agreement_max_abs_delta_s": AGREEMENT_MAX_DELTA_S,
    "holdout_action": "permanent_compare_only_never_training_eligible",
    "timebase_action": (
        "teacher_frame_to_timestamp_then_nearest_encoded_source_pts; nominal "
        "provenance mapping only when media is absent"
    ),
    "abc_weighting": ABC_WEIGHTING_POLICY,
}

ROW_SCHEMA_KEYS = frozenset({
    "source",
    "video",
    "source_video",
    "video_path",
    "media_present",
    "split",
    "fps",
    "source_start_frame",
    "num_frames",
    "event_counts",
    "inventory_event_count",
    "events",
    "loss_validity_mask",
    "unknown_frame_mask",
    "sample_weight",
    "agreement_count",
    "needs_agreement_pass",
    "training_eligible",
    "source_video_sha256",
    "parent_identity",
    "source_lineage_key",
    "timebase_conversion",
    "license_id",
    "license_posture",
})


class CorpusBuildError(ValueError):
    """Raised when an input cannot satisfy the repair-gated corpus contract."""


@dataclass(frozen=True)
class SourceTiming:
    fps: float
    num_frames: int
    duration_seconds: float
    pts_seconds: tuple[float, ...] | None
    basis: str
    needs_pts_verify: bool
    frame_times_sha256: str | None
    pts_media_binding: Mapping[str, Any]


@dataclass(frozen=True)
class ParsedVideo:
    video_id: str
    title: str
    duration_seconds: float
    fps: float
    num_frames: int
    teacher_fps: float
    export_version: str
    cv_version: Mapping[str, Any]
    events: tuple[dict[str, Any], ...]
    media_path: Path | None
    media_path_display: str | None
    source_video_sha256: str
    parent_identity: str
    source_lineage_key: str
    export_sha256: str
    cv_version_sha256: str
    frame_times_sha256: str | None
    pts_media_binding: Mapping[str, Any]
    timebase_conversion: Mapping[str, Any]


def _load_json(path: Path) -> Any:
    if not path.is_file():
        raise CorpusBuildError(f"required input missing: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CorpusBuildError(f"invalid JSON: {path}: {exc}") from exc


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: Any) -> str:
    raw = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return _sha256_bytes(raw)


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def _finite_float(value: Any, *, field: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise CorpusBuildError(f"{field} must be numeric, got {value!r}") from exc
    if not math.isfinite(parsed):
        raise CorpusBuildError(f"{field} must be finite, got {value!r}")
    return parsed


def _positive_float(value: Any, *, field: str) -> float:
    parsed = _finite_float(value, field=field)
    if parsed <= 0.0:
        raise CorpusBuildError(f"{field} must be positive, got {parsed}")
    return parsed


def _integer(value: Any, *, field: str) -> int:
    if isinstance(value, bool):
        raise CorpusBuildError(f"{field} must be an integer, got {value!r}")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise CorpusBuildError(f"{field} must be an integer, got {value!r}") from exc
    if isinstance(value, float) and parsed != value:
        raise CorpusBuildError(f"{field} must be integral, got {value!r}")
    if isinstance(value, str) and str(parsed) != value.strip():
        raise CorpusBuildError(f"{field} must be integral, got {value!r}")
    return parsed


def _gallery_entries(input_root: Path) -> tuple[dict[str, dict[str, Any]], str]:
    path = input_root / "MANIFEST.json"
    raw = path.read_bytes() if path.is_file() else b""
    if not raw:
        raise CorpusBuildError(f"source identity manifest missing: {path}")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise CorpusBuildError(f"invalid JSON: {path}: {exc}") from exc
    videos = payload.get("videos") if isinstance(payload, dict) else None
    if not isinstance(videos, list):
        raise CorpusBuildError(f"videos array missing: {path}")
    entries: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(videos):
        if not isinstance(item, dict) or not isinstance(item.get("video_id"), str):
            raise CorpusBuildError(f"invalid videos[{index}] in {path}")
        video_id = item["video_id"]
        if video_id in entries:
            raise CorpusBuildError(f"duplicate source identity for {video_id}")
        source_sha = item.get("video_sha256")
        if (
            not isinstance(source_sha, str)
            or len(source_sha) != 64
            or any(char not in "0123456789abcdef" for char in source_sha)
        ):
            raise CorpusBuildError(f"invalid video_sha256 for {video_id}")
        entries[video_id] = item
    return entries, _sha256_bytes(raw)


def _discover_media(video_dir: Path, media_root: Path | None) -> Path | None:
    candidates = [video_dir / "max.mp4", video_dir / "source.mp4"]
    if media_root is not None:
        candidates.extend([
            media_root / video_dir.name / "max.mp4",
            media_root / video_dir.name / "source.mp4",
        ])
    return next((path for path in candidates if path.is_file()), None)


def _discover_frame_times(video_dir: Path, frame_times_root: Path | None) -> Path | None:
    direct = video_dir / "frame_times.json"
    matches = [direct] if direct.is_file() else []
    if frame_times_root is not None and frame_times_root.is_dir():
        matches.extend(sorted(frame_times_root.rglob(f"{video_dir.name}/frame_times.json")))
    unique = sorted({path.resolve() for path in matches})
    if len(unique) > 1:
        hashes = {_sha256_file(path) for path in unique}
        if len(hashes) != 1:
            raise CorpusBuildError(
                f"conflicting frame_times.json artifacts for {video_dir.name}: {unique}"
            )
    return unique[0] if unique else None


def _validate_pts(
    payload: Any,
    *,
    video_id: str,
    source_path: Path,
    source_video_sha256: str,
    media_path: Path | None,
) -> SourceTiming:
    if not isinstance(payload, dict) or not isinstance(payload.get("frames"), list):
        raise CorpusBuildError(f"invalid frame-times payload for {video_id}: {source_path}")
    declared_count = _integer(payload.get("frame_count"), field=f"{video_id}.frame_count")
    frames = payload["frames"]
    if declared_count < 1 or len(frames) != declared_count:
        raise CorpusBuildError(
            f"frame-times count mismatch for {video_id}: {len(frames)} != {declared_count}"
        )
    pts: list[float] = []
    for expected_frame, item in enumerate(frames):
        if not isinstance(item, dict) or _integer(
            item.get("frame"), field=f"{video_id}.frames[{expected_frame}].frame"
        ) != expected_frame:
            raise CorpusBuildError(f"non-contiguous frame-times indices for {video_id}")
        pts.append(
            _finite_float(
                item.get("pts_s"), field=f"{video_id}.frames[{expected_frame}].pts_s"
            )
        )
    if pts[0] < 0.0 or any(right <= left for left, right in zip(pts, pts[1:])):
        raise CorpusBuildError(f"frame PTS must be non-negative and strictly increasing: {video_id}")
    fps = _positive_float(payload.get("fps"), field=f"{video_id}.frame_times.fps")
    duration = _positive_float(
        payload.get("duration_s"), field=f"{video_id}.frame_times.duration_s"
    )
    declared_media_sha = payload.get("source_video_sha256", payload.get("media_sha256"))
    if declared_media_sha is not None and declared_media_sha != source_video_sha256:
        raise CorpusBuildError(
            f"frame-times media SHA-256 mismatch for {video_id}: "
            f"{declared_media_sha} != {source_video_sha256}"
        )
    if media_path is None and declared_media_sha is None:
        raise CorpusBuildError(
            f"frame-times artifact is not media-bound for {video_id}: "
            "add source_video_sha256 or stage the matching media"
        )
    frame_times_sha256 = _sha256_file(source_path)
    binding_payload = {
        "source_video_sha256": source_video_sha256,
        "frame_times_sha256": frame_times_sha256,
    }
    return SourceTiming(
        fps=fps,
        num_frames=declared_count,
        duration_seconds=duration,
        pts_seconds=tuple(pts),
        basis="encoded_pts_frame_times",
        needs_pts_verify=False,
        frame_times_sha256=frame_times_sha256,
        pts_media_binding={
            "binding_schema_version": 1,
            "status": "sha256_bound",
            "source_video_sha256": source_video_sha256,
            "media_path": _portable_path(media_path) if media_path is not None else None,
            "media_sha256_verified_from_file": media_path is not None,
            "frame_times_path": _portable_path(source_path),
            "frame_times_sha256": frame_times_sha256,
            "frame_times_declares_media_sha256": declared_media_sha is not None,
            "binding_sha256": _canonical_sha256(binding_payload),
        },
    )


def _probe_media_pts(
    media_path: Path, *, video_id: str, source_video_sha256: str
) -> SourceTiming:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=avg_frame_rate,duration:frame=best_effort_timestamp_time",
        "-of",
        "json",
        str(media_path),
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
    except FileNotFoundError as exc:
        raise CorpusBuildError(
            f"ffprobe is required to derive encoded PTS for local media: {media_path}"
        ) from exc
    if completed.returncode != 0:
        raise CorpusBuildError(
            f"ffprobe failed for {video_id}: {completed.stderr.strip()}"
        )
    try:
        payload = json.loads(completed.stdout)
        streams = payload["streams"]
        frames = payload["frames"]
        numerator, denominator = streams[0]["avg_frame_rate"].split("/", 1)
        fps = float(numerator) / float(denominator)
        pts = tuple(float(frame["best_effort_timestamp_time"]) for frame in frames)
        duration = float(streams[0]["duration"])
    except (KeyError, IndexError, TypeError, ValueError, ZeroDivisionError) as exc:
        raise CorpusBuildError(f"ffprobe metadata malformed for {video_id}") from exc
    if not pts or any(right <= left for left, right in zip(pts, pts[1:])):
        raise CorpusBuildError(f"ffprobe returned invalid PTS for {video_id}")
    return SourceTiming(
        fps=_positive_float(fps, field=f"{video_id}.media.fps"),
        num_frames=len(pts),
        duration_seconds=_positive_float(duration, field=f"{video_id}.media.duration"),
        pts_seconds=pts,
        basis="encoded_pts_ffprobe",
        needs_pts_verify=False,
        frame_times_sha256=None,
        pts_media_binding={
            "binding_schema_version": 1,
            "status": "direct_media_probe",
            "source_video_sha256": source_video_sha256,
            "media_path": _portable_path(media_path),
            "media_sha256_verified_from_file": True,
            "frame_times_path": None,
            "frame_times_sha256": None,
            "frame_times_declares_media_sha256": False,
            "binding_sha256": _canonical_sha256({
                "source_video_sha256": source_video_sha256,
                "frame_times_sha256": None,
            }),
        },
    )


def _source_timing(
    *,
    video_id: str,
    provenance: Mapping[str, Any],
    metadata: Mapping[str, Any],
    media_path: Path | None,
    frame_times_path: Path | None,
    source_video_sha256: str,
) -> SourceTiming:
    if frame_times_path is not None:
        return _validate_pts(
            _load_json(frame_times_path),
            video_id=video_id,
            source_path=frame_times_path,
            source_video_sha256=source_video_sha256,
            media_path=media_path,
        )
    if media_path is not None:
        return _probe_media_pts(
            media_path, video_id=video_id, source_video_sha256=source_video_sha256
        )
    fps_value = provenance.get("fps_reported", metadata.get("fps"))
    duration_value = provenance.get("duration_sec_reported", metadata.get("secs"))
    fps = _positive_float(fps_value, field=f"{video_id}.provenance.fps_reported")
    duration = _positive_float(
        duration_value, field=f"{video_id}.provenance.duration_sec_reported"
    )
    return SourceTiming(
        fps=fps,
        num_frames=max(1, int(math.floor(duration * fps + 0.5))),
        duration_seconds=duration,
        pts_seconds=None,
        basis="provenance_declared_nominal_cfr_needs_pts_verify",
        needs_pts_verify=True,
        frame_times_sha256=None,
        pts_media_binding={
            "binding_schema_version": 1,
            "status": "missing_pts_and_media_binding",
            "source_video_sha256": source_video_sha256,
            "media_path": None,
            "media_sha256_verified_from_file": False,
            "frame_times_path": None,
            "frame_times_sha256": None,
            "frame_times_declares_media_sha256": False,
            "binding_sha256": None,
        },
    )


def _nearest_pts_frame(timestamp_s: float, pts_seconds: Sequence[float]) -> int:
    insertion = bisect.bisect_left(pts_seconds, timestamp_s)
    if insertion <= 0:
        return 0
    if insertion >= len(pts_seconds):
        return len(pts_seconds) - 1
    before = insertion - 1
    after = insertion
    return before if (
        timestamp_s - pts_seconds[before] <= pts_seconds[after] - timestamp_s
    ) else after


def _map_timestamp(timestamp_s: float, timing: SourceTiming) -> tuple[int, float]:
    if timing.pts_seconds is not None:
        frame = _nearest_pts_frame(timestamp_s, timing.pts_seconds)
        return frame, timing.pts_seconds[frame]
    frame = int(math.floor(timestamp_s * timing.fps + 0.5))
    frame = min(max(frame, 0), timing.num_frames - 1)
    return frame, frame / timing.fps


def parse_pbvision_video(
    video_dir: Path,
    *,
    manifest_entry: Mapping[str, Any] | None = None,
    media_root: Path | None = DEFAULT_MEDIA_ROOT,
    frame_times_root: Path | None = DEFAULT_FRAME_TIMES_ROOT,
) -> ParsedVideo:
    """Parse one export and map teacher timestamps onto the source timebase."""

    video_id = video_dir.name
    export_path = video_dir / "cv_export.json"
    cv_version_path = video_dir / "api_get_cv_version.json"
    export = _load_json(export_path)
    metadata_payload = _load_json(video_dir / "api_get_metadata.json")
    provenance = _load_json(video_dir / "video_provenance.json")
    cv_version = _load_json(cv_version_path)
    if not all(isinstance(item, dict) for item in (export, metadata_payload, provenance, cv_version)):
        raise CorpusBuildError(f"expected object inputs for {video_id}")

    if provenance.get("video_id") != video_id:
        raise CorpusBuildError(
            f"video id mismatch for {video_dir}: {provenance.get('video_id')!r}"
        )
    metadata = metadata_payload.get("metadata")
    if not isinstance(metadata, dict):
        raise CorpusBuildError(f"metadata object missing: {video_dir}")
    if manifest_entry is None:
        manifest_entries, _ = _gallery_entries(video_dir.parent)
        manifest_entry = manifest_entries.get(video_id)
    if not isinstance(manifest_entry, Mapping):
        raise CorpusBuildError(f"{video_id} is absent from source MANIFEST.json")
    source_sha = str(manifest_entry.get("video_sha256", ""))

    camera = export.get("camera")
    teacher_fps = _positive_float(
        camera.get("fps") if isinstance(camera, dict) else None,
        field=f"{video_id}.cv_export.camera.fps",
    )
    export_version = export.get("version")
    if not isinstance(export_version, str) or not export_version:
        raise CorpusBuildError(f"export version missing for {video_id}")

    gallery_card = provenance.get("gallery_card")
    title = gallery_card.get("title") if isinstance(gallery_card, dict) else None
    if not isinstance(title, str) or not title.strip():
        title = video_id

    media_path = _discover_media(video_dir, media_root)
    if media_path is not None:
        observed_sha = _sha256_file(media_path)
        if observed_sha != source_sha:
            raise CorpusBuildError(
                f"local media SHA-256 mismatch for {video_id}: {observed_sha} != {source_sha}"
            )
    frame_times_path = _discover_frame_times(video_dir, frame_times_root)
    timing = _source_timing(
        video_id=video_id,
        provenance=provenance,
        metadata=metadata,
        media_path=media_path,
        frame_times_path=frame_times_path,
        source_video_sha256=source_sha,
    )

    sessions = export.get("sessions")
    if not isinstance(sessions, list) or not sessions:
        raise CorpusBuildError(f"sessions array missing or empty: {video_id}")
    parent_identity = f"pbvision:{video_id}:sha256:{source_sha}"
    lineage_key = _sha256_bytes(parent_identity.encode())

    teacher_events: list[dict[str, Any]] = []
    observed_teacher_frames: set[tuple[int, str]] = set()
    for session_index, session in enumerate(sessions):
        rallies = session.get("rallies") if isinstance(session, dict) else None
        if not isinstance(rallies, list):
            raise CorpusBuildError(f"{video_id}.sessions[{session_index}].rallies is not an array")
        for rally_index, rally in enumerate(rallies):
            if not isinstance(rally, dict) or not isinstance(rally.get("frames"), list):
                raise CorpusBuildError(
                    f"{video_id}.sessions[{session_index}].rallies[{rally_index}] is malformed"
                )
            rally_start = _integer(
                rally.get("frame_index"),
                field=f"{video_id}.sessions[{session_index}].rallies[{rally_index}].frame_index",
            )
            if rally_start < 0:
                raise CorpusBuildError(f"negative rally frame_index in {video_id}")
            rally_frame_count = len(rally["frames"])
            if rally_frame_count:
                rally_source_start, _ = _map_timestamp(rally_start / teacher_fps, timing)
                rally_source_last, _ = _map_timestamp(
                    (rally_start + rally_frame_count - 1) / teacher_fps, timing
                )
                rally_source_end_exclusive = min(
                    timing.num_frames, max(rally_source_start + 1, rally_source_last + 1)
                )
            else:
                rally_source_start = 0
                rally_source_end_exclusive = 0
            for local_frame, frame_payload in enumerate(rally["frames"]):
                if not isinstance(frame_payload, dict):
                    raise CorpusBuildError(f"non-object frame payload in {video_id}")
                balls = frame_payload.get("balls")
                if not isinstance(balls, dict):
                    raise CorpusBuildError(f"balls object missing in {video_id}")
                selected = balls.get("selected")
                if selected in (None, "ball"):
                    continue
                if selected not in {"shot", "bounce", "net"}:
                    raise CorpusBuildError(
                        f"unknown selected ball action {selected!r} in {video_id}"
                    )
                actions = frame_payload.get("actions")
                action = actions.get(selected) if isinstance(actions, dict) else None
                if not isinstance(action, dict):
                    raise CorpusBuildError(
                        f"selected action payload {selected!r} missing in {video_id}"
                    )
                confidence = _finite_float(
                    action.get("confidence"), field=f"{video_id}.{selected}.confidence"
                )
                if not 0.0 <= confidence <= 1.0:
                    raise CorpusBuildError(
                        f"confidence outside [0,1] in {video_id}: {confidence}"
                    )
                teacher_frame = rally_start + local_frame
                dedupe_key = (teacher_frame, selected)
                if dedupe_key in observed_teacher_frames:
                    raise CorpusBuildError(
                        f"duplicate selected event at teacher frame {teacher_frame} "
                        f"({selected}) in {video_id}"
                    )
                observed_teacher_frames.add(dedupe_key)
                teacher_timestamp_s = teacher_frame / teacher_fps
                source_frame, source_pts_s = _map_timestamp(teacher_timestamp_s, timing)
                mapped_class = {"shot": "HIT", "bounce": "BOUNCE", "net": None}[selected]
                low_confidence = confidence < TEACHER_CONFIDENCE_MIN
                event_id = _sha256_bytes(
                    f"{lineage_key}:{session_index}:{rally_index}:{teacher_frame}:{selected}".encode()
                )
                teacher_events.append({
                    "event_id": event_id,
                    "video_id": video_id,
                    "source_video_sha256": source_sha,
                    "parent_identity": parent_identity,
                    "source_lineage_key": lineage_key,
                    "session_index": session_index,
                    "rally_index": rally_index,
                    "rally_source_start_frame": rally_source_start,
                    "rally_source_end_frame_exclusive": rally_source_end_exclusive,
                    "teacher_event_type": selected.upper(),
                    "manifest_class": mapped_class,
                    "teacher_frame": teacher_frame,
                    "teacher_fps": teacher_fps,
                    "teacher_timestamp_s": teacher_timestamp_s,
                    "source_frame": source_frame,
                    "frame": source_frame,
                    "source_pts_s": source_pts_s,
                    "time_seconds": source_pts_s,
                    "mapping_abs_error_s": abs(source_pts_s - teacher_timestamp_s),
                    "teacher_confidence": confidence,
                    "confidence": confidence,
                    "below_teacher_confidence_min": low_confidence,
                    "agreement_count": 0,
                    "independent_agreements": [],
                    "pseudo_weight": 0.0,
                    "needs_agreement_pass": mapped_class is not None and not low_confidence,
                    "training_eligible": False,
                    "unknown_for_loss": True,
                    "filter_decision": (
                        "context_only_unsupported_net"
                        if mapped_class is None
                        else "rejected_low_teacher_confidence"
                        if low_confidence
                        else "pending_independent_agreement"
                    ),
                    "action": action,
                    "ball_context": balls.get(selected),
                    "court_context": frame_payload.get("court"),
                })

    teacher_events.sort(
        key=lambda item: (
            item["source_frame"], item["teacher_frame"], item["teacher_event_type"]
        )
    )
    if teacher_events and teacher_events[-1]["source_frame"] >= timing.num_frames:
        raise CorpusBuildError(f"mapped event exceeds source frame count for {video_id}")
    conversion = {
        "teacher_timebase": "cv_export.camera.fps",
        "teacher_fps": teacher_fps,
        "timestamp_formula": "teacher_frame / teacher_fps",
        "source_timebase": timing.basis,
        "source_fps": timing.fps,
        "mapping": (
            "argmin(abs(encoded_source_pts - teacher_timestamp_s))"
            if timing.pts_seconds is not None
            else "nearest_nominal_source_frame_from_provenance_fps"
        ),
        "needs_pts_verify": timing.needs_pts_verify,
        "frame_times_sha256": timing.frame_times_sha256,
        "pts_media_binding": dict(timing.pts_media_binding),
    }
    return ParsedVideo(
        video_id=video_id,
        title=title,
        duration_seconds=timing.duration_seconds,
        fps=timing.fps,
        num_frames=timing.num_frames,
        teacher_fps=teacher_fps,
        export_version=export_version,
        cv_version=dict(cv_version),
        events=tuple(teacher_events),
        media_path=media_path,
        media_path_display=_portable_path(media_path) if media_path is not None else None,
        source_video_sha256=source_sha,
        parent_identity=parent_identity,
        source_lineage_key=lineage_key,
        export_sha256=_sha256_file(export_path),
        cv_version_sha256=_sha256_file(cv_version_path),
        frame_times_sha256=timing.frame_times_sha256,
        pts_media_binding=dict(timing.pts_media_binding),
        timebase_conversion=conversion,
    )


def _split_counts(group_count: int) -> dict[str, int]:
    raw = {split: group_count * ratio for split, ratio in SPLIT_RATIOS.items()}
    counts = {split: int(value) for split, value in raw.items()}
    remaining = group_count - sum(counts.values())
    split_order = list(SPLIT_RATIOS)
    order = sorted(
        SPLIT_RATIOS,
        key=lambda split: (-(raw[split] - counts[split]), split_order.index(split)),
    )
    for split in order[:remaining]:
        counts[split] += 1
    return counts


def assign_source_splits(
    lineage_by_video: Mapping[str, str] | Iterable[str], *, seed: int
) -> dict[str, str]:
    """Assign source-disjoint splits from immutable lineage identities."""

    if isinstance(lineage_by_video, Mapping):
        lineage = dict(lineage_by_video)
    else:
        # Compatibility for callers that only have ids; production passes the
        # content-derived lineage mapping and tests assert that path.
        lineage = {video_id: video_id for video_id in lineage_by_video}
    if len(lineage.values()) != len(set(lineage.values())):
        raise CorpusBuildError("duplicate immutable source lineage in eligible inputs")
    ordered = sorted(
        lineage,
        key=lambda video_id: (
            hashlib.sha256(f"{seed}:{lineage[video_id]}".encode()).hexdigest(),
            lineage[video_id],
            video_id,
        ),
    )
    counts = _split_counts(len(ordered))
    assignment: dict[str, str] = {}
    offset = 0
    for split in ("train", "val", "test"):
        for video_id in ordered[offset:offset + counts[split]]:
            assignment[video_id] = split
        offset += counts[split]
    return assignment


def _window_starts(num_frames: int, *, window_frames: int, stride_frames: int) -> list[int]:
    tail_start = max(0, num_frames - window_frames)
    starts = list(range(0, tail_start + 1, stride_frames))
    if not starts or starts[-1] != tail_start:
        starts.append(tail_start)
    return starts


def _window_projection(
    row: Mapping[str, Any], *, window_frames: int, stride_frames: int
) -> tuple[int, int]:
    starts = _window_starts(
        int(row["num_frames"]), window_frames=window_frames, stride_frames=stride_frames
    )
    event_frames = [int(event["frame"]) for event in row["events"]]
    positive = sum(
        any(start <= frame < start + window_frames for frame in event_frames)
        for start in starts
    )
    return len(starts), positive


def _sequence_regularity(frames: list[int], fps: float) -> dict[str, Any]:
    if len(frames) < 3:
        return {
            "event_count": len(frames),
            "median_gap_seconds": None,
            "gap_coefficient_of_variation": None,
            "dominant_frame_gap_fraction": None,
            "constant_rate_degenerate": False,
        }
    gaps = [right - left for left, right in zip(frames, frames[1:])]
    mean_gap = statistics.fmean(gaps)
    gap_cv = statistics.pstdev(gaps) / mean_gap if mean_gap > 0.0 else 0.0
    dominant_fraction = max(gaps.count(gap) for gap in set(gaps)) / len(gaps)
    degenerate = len(gaps) >= 20 and (gap_cv <= 0.10 or dominant_fraction >= 0.80)
    return {
        "event_count": len(frames),
        "median_gap_seconds": statistics.median(gaps) / fps,
        "gap_coefficient_of_variation": gap_cv,
        "dominant_frame_gap_fraction": dominant_fraction,
        "constant_rate_degenerate": degenerate,
    }


def _video_report(parsed: ParsedVideo, *, compare_only: bool, split: str | None) -> dict[str, Any]:
    counts = {
        event_type: sum(event["teacher_event_type"] == event_type for event in parsed.events)
        for event_type in ("SHOT", "BOUNCE", "NET")
    }
    mapped = [event for event in parsed.events if event["manifest_class"] is not None]
    event_rate = len(mapped) / parsed.duration_seconds
    hit_frames = [event["frame"] for event in mapped if event["manifest_class"] == "HIT"]
    mapped_frames = [event["frame"] for event in mapped]
    hit_regularity = _sequence_regularity(hit_frames, parsed.fps)
    combined_regularity = _sequence_regularity(mapped_frames, parsed.fps)
    low, high = PLAUSIBLE_EVENT_RATE_RANGE
    return {
        "video_id": parsed.video_id,
        "title": parsed.title,
        "source_video_sha256": parsed.source_video_sha256,
        "parent_identity": parsed.parent_identity,
        "source_lineage_key": parsed.source_lineage_key,
        "source_fps": parsed.fps,
        "teacher_fps": parsed.teacher_fps,
        "duration_seconds": parsed.duration_seconds,
        "num_frames": parsed.num_frames,
        "compare_only": compare_only,
        "training_eligible": False,
        "split": split,
        "media_present": parsed.media_path is not None,
        "needs_pts_verify": parsed.timebase_conversion["needs_pts_verify"],
        "timebase_conversion": parsed.timebase_conversion,
        "teacher_event_counts": {
            "HIT_from_selected_shot": counts["SHOT"],
            "BOUNCE_from_selected_bounce": counts["BOUNCE"],
            "NET_context_only": counts["NET"],
        },
        "manifest_event_count": len(mapped),
        "low_confidence_mapped_events": sum(
            event["below_teacher_confidence_min"] for event in mapped
        ),
        "pending_agreement_events": sum(event["needs_agreement_pass"] for event in mapped),
        "manifest_events_per_second": event_rate,
        "rate_outside_plausible_range": not low <= event_rate <= high,
        "constant_rate_degenerate": bool(
            hit_regularity["constant_rate_degenerate"]
            or combined_regularity["constant_rate_degenerate"]
        ),
        "regularity": {"HIT": hit_regularity, "HIT_plus_BOUNCE": combined_regularity},
    }


def _manifest_event(event: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "event_id": event["event_id"],
        "class": event["manifest_class"],
        "frame": event["frame"],
        "source_pts_s": event["source_pts_s"],
        "teacher_frame": event["teacher_frame"],
        "teacher_fps": event["teacher_fps"],
        "teacher_timestamp_s": event["teacher_timestamp_s"],
        "teacher_confidence": event["teacher_confidence"],
        "agreement_count": 0,
        "independent_agreements": [],
        "pseudo_weight": 0.0,
        "needs_agreement_pass": event["needs_agreement_pass"],
        "training_eligible": False,
        "unknown_for_loss": True,
        "filter_decision": event["filter_decision"],
        "session_index": event["session_index"],
        "rally_index": event["rally_index"],
        "rally_source_start_frame": event["rally_source_start_frame"],
        "rally_source_end_frame_exclusive": event[
            "rally_source_end_frame_exclusive"
        ],
    }


def _provenance_header(
    parsed_videos: Sequence[ParsedVideo], *, gallery_manifest_sha256: str
) -> dict[str, Any]:
    return {
        "source_manifest": "data/pbvision_gallery_20260719/MANIFEST.json",
        "source_manifest_sha256": gallery_manifest_sha256,
        "builder": {
            "path": "scripts/racketsport/build_pbvision_event_corpus.py",
            "sha256": _sha256_file(BUILDER_PATH),
        },
        "filtering_policy": FILTERING_POLICY,
        "filtering_policy_sha256": _canonical_sha256(FILTERING_POLICY),
        "sources": [
            {
                "video_id": parsed.video_id,
                "compare_only": parsed.video_id in COMPARE_ONLY_HOLDOUTS,
                "training_eligible": False,
                "source_video_sha256": parsed.source_video_sha256,
                "parent_identity": parsed.parent_identity,
                "source_lineage_key": parsed.source_lineage_key,
                "cv_export_version": parsed.export_version,
                "cv_export_sha256": parsed.export_sha256,
                "cv_version": parsed.cv_version,
                "get_cv_version_sha256": parsed.cv_version_sha256,
                "frame_times_sha256": parsed.frame_times_sha256,
                "pts_media_binding": parsed.pts_media_binding,
            }
            for parsed in sorted(parsed_videos, key=lambda item: item.video_id)
        ],
    }


def build_corpus(
    input_root: Path,
    *,
    media_root: Path | None = DEFAULT_MEDIA_ROOT,
    frame_times_root: Path | None = DEFAULT_FRAME_TIMES_ROOT,
    seed: int = DEFAULT_SEED,
    window_frames: int = DEFAULT_WINDOW_FRAMES,
    stride_frames: int = DEFAULT_WINDOW_STRIDE,
) -> tuple[
    dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]
]:
    """Build raw staging manifest, report, eligible context, compare-only context."""

    if window_frames < 1 or stride_frames < 1:
        raise CorpusBuildError("window_frames and stride_frames must be positive")
    manifest_entries, gallery_manifest_sha256 = _gallery_entries(input_root)
    export_paths = sorted(input_root.glob("*/cv_export.json"))
    if not export_paths:
        raise CorpusBuildError(f"no */cv_export.json inputs found under {input_root}")
    parsed_videos = [
        parse_pbvision_video(
            path.parent,
            manifest_entry=manifest_entries.get(path.parent.name),
            media_root=media_root,
            frame_times_root=frame_times_root,
        )
        for path in export_paths
    ]
    ids = [parsed.video_id for parsed in parsed_videos]
    if len(ids) != len(set(ids)):
        raise CorpusBuildError("duplicate video ids in input")

    eligible = [item for item in parsed_videos if item.video_id not in COMPARE_ONLY_HOLDOUTS]
    split_assignment = assign_source_splits(
        {item.video_id: item.source_lineage_key for item in eligible}, seed=seed
    )
    rows: list[dict[str, Any]] = []
    contexts: list[dict[str, Any]] = []
    compare_contexts: list[dict[str, Any]] = []
    per_video: list[dict[str, Any]] = []
    for parsed in parsed_videos:
        compare_only = parsed.video_id in COMPARE_ONLY_HOLDOUTS
        split = None if compare_only else split_assignment[parsed.video_id]
        per_video.append(_video_report(parsed, compare_only=compare_only, split=split))
        destination = compare_contexts if compare_only else contexts
        for event in parsed.events:
            destination.append({
                **event,
                "compare_only": compare_only,
                "training_eligible": False,
                "needs_agreement_pass": False if compare_only else event["needs_agreement_pass"],
                "filter_decision": (
                    "permanent_compare_only_denylist"
                    if compare_only
                    else event["filter_decision"]
                ),
            })
        if compare_only:
            continue
        manifest_events = sorted(
            (_manifest_event(event) for event in parsed.events if event["manifest_class"] is not None),
            key=lambda event: (event["frame"], event["class"], event["event_id"]),
        )
        hit_count = sum(event["class"] == "HIT" for event in manifest_events)
        bounce_count = sum(event["class"] == "BOUNCE" for event in manifest_events)
        unknown_frames = {event["frame"] for event in parsed.events}
        row = {
            "source": SOURCE_NAME,
            "video": parsed.video_id,
            "source_video": parsed.video_id,
            "video_path": parsed.media_path_display,
            "media_present": parsed.media_path is not None,
            "split": split,
            "fps": parsed.fps,
            "source_start_frame": 0,
            "num_frames": parsed.num_frames,
            "event_counts": {"HIT": hit_count, "BOUNCE": bounce_count, "background": 0},
            "inventory_event_count": len(manifest_events),
            "events": manifest_events,
            "loss_validity_mask": [True, True, True],
            "unknown_frame_mask": [frame in unknown_frames for frame in range(parsed.num_frames)],
            # A raw/pending row must fail the current fine-tune row-weight gate.
            "sample_weight": 0.0,
            "agreement_count": 0,
            "needs_agreement_pass": any(
                event["needs_agreement_pass"] for event in manifest_events
            ),
            "training_eligible": False,
            "source_video_sha256": parsed.source_video_sha256,
            "parent_identity": parsed.parent_identity,
            "source_lineage_key": parsed.source_lineage_key,
            "timebase_conversion": parsed.timebase_conversion,
            "license_id": "owner_signed_full_usage_2026-07-20",
            "license_posture": LICENSE_POSTURE,
        }
        if set(row) != ROW_SCHEMA_KEYS:
            raise CorpusBuildError(f"internal row-schema drift: {sorted(set(row) ^ ROW_SCHEMA_KEYS)}")
        rows.append(row)

    def sort_key(event: Mapping[str, Any]) -> tuple[Any, ...]:
        return (
            event["video_id"], event["source_frame"], event["teacher_frame"],
            event["teacher_event_type"],
        )
    contexts.sort(key=sort_key)
    compare_contexts.sort(key=sort_key)
    rows.sort(key=lambda row: row["source_video"])
    per_video.sort(key=lambda item: item["video_id"])

    split_projection: dict[str, dict[str, int]] = {}
    for split in ("train", "val", "test"):
        split_rows = [row for row in rows if row["split"] == split]
        window_counts = [
            _window_projection(row, window_frames=window_frames, stride_frames=stride_frames)
            for row in split_rows
        ]
        split_projection[split] = {
            "source_videos": len(split_rows),
            "manifest_events": sum(row["inventory_event_count"] for row in split_rows),
            "projected_sliding_windows": sum(total for total, _ in window_counts),
            "projected_positive_windows": sum(positive for _, positive in window_counts),
            "currently_materializable_windows": sum(
                total for row, (total, _) in zip(split_rows, window_counts)
                if row["media_present"] and not row["timebase_conversion"]["needs_pts_verify"]
            ),
            "training_eligible_windows_before_agreement": 0,
        }

    provenance = _provenance_header(
        parsed_videos, gallery_manifest_sha256=gallery_manifest_sha256
    )
    mapped_events = [event for row in rows for event in row["events"]]
    manifest = {
        "schema_version": 2,
        "artifact_type": "event_head_pbvision_teacher_staging_dataset_manifest",
        "verified": False,
        "training_ready": False,
        "teacher_derived": True,
        "ground_truth": False,
        "seed": seed,
        "config": {
            "split_unit": "immutable_source_lineage_key",
            "split_ratios": SPLIT_RATIOS,
            "window_frames": window_frames,
            "window_stride_frames": stride_frames,
            "event_mapping": {
                "selected_shot": "HIT",
                "selected_bounce": "BOUNCE",
                "selected_net": "context_only_and_unknown_not_background",
            },
            "unknown_frame_mask_semantics": (
                "true means ignore frame for loss; loss_validity_mask remains class-level"
            ),
        },
        "classes": {"0": "background", "1": "HIT", "2": "BOUNCE"},
        "image_size": 224,
        "decode_policy": "on_the_fly_no_frame_cache",
        "license_posture": LICENSE_POSTURE,
        "permanent_compare_only_denylist": sorted(COMPARE_ONLY_HOLDOUTS),
        "abc_weighting_policy": ABC_WEIGHTING_POLICY,
        "provenance": provenance,
        "totals": {
            "source_videos": len(rows),
            "media_present_videos": sum(row["media_present"] for row in rows),
            "pts_verified_videos": sum(
                not row["timebase_conversion"]["needs_pts_verify"] for row in rows
            ),
            "needs_pts_verify_videos": sum(
                row["timebase_conversion"]["needs_pts_verify"] for row in rows
            ),
            "HIT": sum(row["event_counts"]["HIT"] for row in rows),
            "BOUNCE": sum(row["event_counts"]["BOUNCE"] for row in rows),
            "manifest_events": len(mapped_events),
            "low_confidence_rejected_events": sum(
                event["filter_decision"] == "rejected_low_teacher_confidence"
                for event in mapped_events
            ),
            "needs_agreement_pass_events": sum(
                event["needs_agreement_pass"] for event in mapped_events
            ),
            "training_eligible_events": 0,
        },
        "rows": rows,
    }
    row_ids = {row["source_video"] for row in rows}
    holdouts = [
        {
            "video_id": video_id,
            "reason": reason,
            "present_in_gallery_input": video_id in ids,
            "excluded_from_training_rows": video_id not in row_ids,
            "derivative": (
                "compare_only_teacher_event_context.jsonl"
                if video_id in ids else None
            ),
            "training_eligible": False,
        }
        for video_id, reason in sorted(COMPARE_ONLY_HOLDOUTS.items())
    ]
    report = {
        "schema_version": 2,
        "artifact_type": "pbvision_teacher_event_corpus_report",
        "verified": False,
        "training_ready": False,
        "teacher_derived": True,
        "ground_truth": False,
        "license_posture": LICENSE_POSTURE,
        "input_gallery_videos": len(parsed_videos),
        "included_staging_videos": len(rows),
        "compare_only_holdouts": holdouts,
        "per_video": per_video,
        "window_projection": split_projection,
        "builder_totals": manifest["totals"],
        "blockers": [
            "Nine eligible sources still need encoded-PTS verification on staged media.",
            "No teacher event is training-eligible until the independent audio-onset/ball-velocity-kink agreement pass runs.",
            "The schema-v2 loader emits frame_loss_mask; the concurrent fine-tune owner must preserve it through the loss call.",
            "Ultra re-review is required before any A/B/C training use.",
        ],
    }
    return manifest, report, contexts, compare_contexts


def _json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _context_bytes(contexts: Iterable[dict[str, Any]]) -> bytes:
    return b"".join(
        (json.dumps(item, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
        for item in contexts
    )


def _filtered_placeholder(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "artifact_type": "event_head_pbvision_agreement_filtered_dataset_manifest",
        "verified": False,
        "training_ready": False,
        "teacher_derived": True,
        "ground_truth": False,
        "classes": manifest["classes"],
        "image_size": manifest["image_size"],
        "license_posture": manifest["license_posture"],
        "config": {
            "window_frames": manifest["config"]["window_frames"],
            "window_stride_frames": manifest["config"]["window_stride_frames"],
            "split": "train_only",
            "materialization": "event_centered_rows_after_vm_agreement_pass",
            "unknown_frame_mask_semantics": manifest["config"][
                "unknown_frame_mask_semantics"
            ],
        },
        "abc_weighting_policy": manifest["abc_weighting_policy"],
        "permanent_compare_only_denylist": manifest[
            "permanent_compare_only_denylist"
        ],
        "provenance": manifest["provenance"],
        "source_staging_manifest_sha256": None,
        "needs_agreement_pass": True,
        "pending_event_count": manifest["totals"]["needs_agreement_pass_events"],
        "rows": [],
        "blocked_reason": (
            "placeholder only: VM must verify all source PTS, compute independent "
            "audio-onset/ball-velocity-kink agreements, materialize train-only "
            "event-centered rows, "
            "and apply the unknown-frame loader hunk"
        ),
    }


def _abc_stage_markdown() -> str:
    return f"""# pb.vision A/B/C agreement staging procedure

Status: `VERIFIED=0`; staging-only. Do not train until the resulting manifests,
loader hunk, media identities, and this corpus pass ULTRA re-review.

## Frozen invariants

- Permanent compare-only ids: {', '.join(f'`{item}`' for item in sorted(COMPARE_ONLY_HOLDOUTS))}. Reject them before reading any derivative row.
- Validate every staged MP4 against `source_video_sha256`; build/validate monotonic encoded `frame_times.json`; remap every teacher timestamp with nearest encoded PTS. Refuse any row with `needs_pts_verify=true`.
- pb.vision confidence is retained for audit. Confidence below `{TEACHER_CONFIDENCE_MIN}` rejects and UNKNOWN-masks the event, but confidence never creates an agreement and by itself earns weight zero.
- Use source PTS for all matching. The frozen maximum absolute match delta is `{AGREEMENT_MAX_DELTA_S:.3f}` seconds, matching `event_fusion.DEFAULT_MAX_TIME_DELTA_S` at corpus freeze.

## Exact independent-agreement pass

1. For each eligible source clip, load the current `audio_onsets_v2.json` and image-space `ball_inflections.json` produced against the same source SHA and `frame_times.json`. Their paths are explicit `build_abc_arm_manifests.py` parameters. Missing, unhashed, stale, differently timed, or non-monotonic evidence blocks that clip.
2. Within each signal family independently, form all teacher-cue candidates with absolute PTS delta <= `{AGREEMENT_MAX_DELTA_S:.3f}` seconds. Resolve one-to-one conflicts deterministically by `(absolute_delta_s, teacher_event_id, cue_time_s, cue_stable_id)`; one cue cannot agree with two teacher events.
3. Set each mapped HIT/BOUNCE event's `independent_agreements` to the sorted subset of `audio_onset` and `ball_velocity_kink`; set `agreement_count=len(independent_agreements)`. pb.vision is never counted. Do not combine multiple cues from one family into multiple agreements.
4. Apply the frozen weight exactly: count 0 -> `pseudo_weight=0` and UNKNOWN/ignored; count 1 -> `0.25`; count >=2 -> `0.5`. A rejected low-confidence event stays weight 0 even if a cue is nearby. Uncertain/rejected events are ignored, never converted to background.
5. Materialize `pbvision_filtered_teacher_manifest.json` as train-only, one 64-frame event-centered row per accepted event. Preserve source SHA, parent identity, lineage, class, pixels, rally, confidence, agreement details, and PTS. Set row `sample_weight` to the focal event weight. The focal event is positive; every rejected/pending non-focal teacher frame in the window has `unknown_frame_mask=true`.
6. Materialize arm C from B only after B is frozen: preserve byte-identical source rows/video paths, pixels, classes, weights, agreement metadata, row count, and optimizer/update budget; deterministically shuffle only the focal event time within the same immutable source rally. Record seed and B manifest SHA.
7. Run A with zero teacher rows. Run B/C with normalized aggregate pseudo loss capped at human loss (`--pseudo-weight-cap 1.0`). Select checkpoints and thresholds only on the fixed owner validation set; pseudo validation and compare-only clips cannot select.

## Required pre-dispatch assertions

- no holdout id has any `training_eligible=true` derivative;
- every B/C row is `split=train`, media-present, SHA-matched, PTS-verified, and has `sample_weight` 0.25 or 0.5 consistent with `agreement_count`;
- every UNKNOWN frame reaches loss as ignored by the reviewed loader hunk;
- B/C have identical row/pixel/class/weight budgets and differ only in within-rally event time;
- normalized total pseudo loss never exceeds normalized human loss;
- `VERIFIED=0` and teacher-not-GT provenance survive into every output.
"""


def _vm_abc_run_markdown() -> str:
    return rf"""# VM A/B/C staging and materialization runbook

Status: `VERIFIED=0`. This procedure builds training inputs only. It must not
run a GT scorer, frozen gate, protected evaluation, or checkpoint selection.

## 1. Stage media and freeze its identity

For every train-split row in the rebuilt teacher corpus, stage the MP4 at an
explicit path and verify it before any derived artifact is built:

```bash
sha256sum "$MEDIA_PATH"
# Must equal row.source_video_sha256 exactly.
```

Stop on a missing file, decode failure, or mismatch. Never repair the manifest
to match an unexpected file.

## 2. Build encoded PTS and bind it to that MP4

Generate monotonic `frame_times.json` from the staged MP4, then add the exact
`source_video_sha256` to that JSON (or keep the MP4 staged so the corpus builder
can verify the pair directly). Rebuild the teacher corpus with explicit media
and frame-times roots. The resulting row must contain a
`timebase_conversion.pts_media_binding` with:

- `status=sha256_bound`;
- the staged media SHA;
- the frame-times artifact SHA; and
- `binding_sha256=sha256(canonical(media_sha, frame_times_sha))`.

```bash
sha256sum "$MEDIA_PATH" "$FRAME_TIMES_PATH"
.venv/bin/python scripts/racketsport/build_pbvision_event_corpus.py \
  --input-root data/pbvision_gallery_20260719 \
  --media-root "$MEDIA_ROOT" \
  --frame-times-root "$FRAME_TIMES_ROOT" \
  --output-dir "$CORPUS_OUT"
```

Stop if any train row still has `needs_pts_verify=true` or lacks the explicit
PTS/media binding.

## 3. Build audio-onset artifacts from the same MP4

```bash
.venv/bin/python scripts/racketsport/build_audio_onsets_v2.py \
  --input "$MEDIA_PATH" --clip "$VIDEO_ID" --frame-rate "$FPS" \
  --out "$AUDIO_ONSETS_PATH"
sha256sum "$MEDIA_PATH" "$FRAME_TIMES_PATH" "$AUDIO_ONSETS_PATH"
```

Record this exact audio artifact path in the materializer arguments. Audio is
an agreement family only; it never supplies the HIT/BOUNCE class.

## 4. Build 2D ball-velocity-kink artifacts

First build the per-clip 2D ball track against the same media/PTS identity,
then derive image-space velocity/direction kinks:

```bash
# Produce $BALL_TRACK_PATH with the frozen BALL-2D command selected by the manager.
.venv/bin/python scripts/racketsport/build_ball_inflections.py \
  --ball-track "$BALL_TRACK_PATH" --frame-times "$FRAME_TIMES_PATH" \
  --out "$BALL_KINKS_PATH"
sha256sum "$MEDIA_PATH" "$FRAME_TIMES_PATH" "$BALL_TRACK_PATH" "$BALL_KINKS_PATH"
```

Do not substitute pb.vision ball/court output; it is not independent evidence.

## 5. Run deterministic agreement and materialize B/C

Pass every consumed path explicitly as `VIDEO_ID=PATH`. The CLI emits
`VM_ABC_NEEDS.json`, `arm_b_manifest.json`, `arm_c_manifest.json`, and an input
binding ledger containing every media/artifact SHA:

```bash
.venv/bin/python scripts/racketsport/build_abc_arm_manifests.py \
  --teacher-manifest "$CORPUS_OUT/manifest.json" \
  --output-dir "$ABC_OUT" --seed {DEFAULT_SEED} \
  --media "$VIDEO_ID=$MEDIA_PATH" \
  --frame-times "$VIDEO_ID=$FRAME_TIMES_PATH" \
  --audio-onsets "$VIDEO_ID=$AUDIO_ONSETS_PATH" \
  --ball-velocity-kinks "$VIDEO_ID=$BALL_KINKS_PATH"
sha256sum "$CORPUS_OUT/manifest.json" "$ABC_OUT"/*.json
```

Repeat each path flag once per train clip. Agreement is one-to-one within each
signal family at the frozen `{AGREEMENT_MAX_DELTA_S:.3f}s` tolerance. Count 0 is
ignored/omitted from B rows; count 1 receives 0.25; count >=2 receives 0.5. C
keeps B's rows, pixels, class counts, weights, and agreement metadata, changing
only focal event time within the same rally using the frozen seed.

## 6. Pre-dispatch checks

- every recorded SHA recomputes exactly on the VM;
- B/C manifests remain `verified=false` and teacher-derived, never GT;
- every B/C row is schema v2 and has a 64-entry UNKNOWN mask;
- every B/C media path decodes and matches its source SHA;
- B/C row/pixel/class/weight budgets match exactly;
- no protected or compare-only scorer/eval file was read; and
- stop for ULTRA review before training or any scored run.
"""


def _corpus_notes(manifest_sha256: str, report: Mapping[str, Any]) -> str:
    totals = report["builder_totals"]
    return f"""# pb.vision teacher corpus repair notes

Status: `VERIFIED=0`, `training_ready=false`. This is a teacher-derived staging
corpus, not human ground truth. ULTRA re-review is mandatory before training.

- Manifest SHA-256: `{manifest_sha256}`
- Eligible source rows: {totals['source_videos']}
- HIT / BOUNCE inventory: {totals['HIT']} / {totals['BOUNCE']}
- PTS-verified / needs-PTS-verify sources: {totals['pts_verified_videos']} / {totals['needs_pts_verify_videos']}
- Low-confidence rejected events: {totals['low_confidence_rejected_events']}
- Pending independent-agreement events: {totals['needs_agreement_pass_events']}
- Training-eligible events now: 0

`teacher_event_context.jsonl` contains only non-holdout source events.
Holdout observations, when locally present, are segregated into
`compare_only_teacher_event_context.jsonl` with `training_eligible=false`.
The empty filtered manifest is an intentional fail-closed placeholder until the
VM agreement pass described in `ABC_STAGE.md` completes.
"""


def _loader_required_diff() -> str:
    return """# REQUIRED MANAGER HUNK — NOT APPLIED BY pbv_corpus_rebuild_20260720
# Concurrent owners must adapt this against their final datasets.py and
# finetune_event_head.py, add tests, and ULTRA-review it before corpus use.
diff --git a/threed/racketsport/event_head/datasets.py b/threed/racketsport/event_head/datasets.py
@@ class WindowSpec:
+    unknown_frame_mask: tuple[bool, ...] = ()
@@ validate_current_manifest(manifest)
-        manifest.get("schema_version") != 1
+        manifest.get("schema_version") not in {1, 2}
@@ validate_current_manifest(row)
+        # Parse and validate fps/source_start_frame/num_frames/events even when
+        # media_present=false; the current early continue must move below this.
+        unknown = row.get("unknown_frame_mask")
+        if manifest["schema_version"] == 2 and unknown is None:
+            raise DatasetFormatError("schema v2 rows require unknown_frame_mask")
+        if unknown is not None and (
+            not isinstance(unknown, list)
+            or len(unknown) != num_frames
+            or not all(isinstance(value, bool) for value in unknown)
+        ):
+            raise DatasetFormatError(
+                "unknown_frame_mask must be one bool per source frame"
+            )
@@ manifest_windows(... WindowSpec(...))
+                unknown_frame_mask=tuple(
+                    row.get("unknown_frame_mask", [False] * row_frames)[
+                        local_start:local_start + window_frames
+                    ]
+                ),
@@ manifest_event_centered_windows(... WindowSpec(...))
+            unknown_frame_mask=tuple(
+                row.get("unknown_frame_mask", [False] * int(row["num_frames"]))[
+                    local_start:local_start + window_frames
+                ]
+            ),
@@ EventWindowDataset.__getitem__
+        if spec.unknown_frame_mask and len(spec.unknown_frame_mask) != spec.num_frames:
+            raise DatasetFormatError("window unknown_frame_mask length mismatch")
+        frame_loss_mask = ~torch.tensor(
+            spec.unknown_frame_mask or (False,) * spec.num_frames,
+            dtype=torch.bool,
+        )
         return {
             "frames": frames,
             "targets": targets,
+            "frame_loss_mask": frame_loss_mask,
diff --git a/scripts/racketsport/finetune_event_head.py b/scripts/racketsport/finetune_event_head.py
@@ _validate_row(row)
+    unknown = row.get("unknown_frame_mask")
+    if role == "pseudo" and unknown is None:
+        raise FineTuneInputError(
+            f"pseudo row {index} must carry unknown_frame_mask", 20
+        )
+    if unknown is not None and (
+        not isinstance(unknown, list)
+        or len(unknown) != int(row["num_frames"])
+        or not all(isinstance(value, bool) for value in unknown)
+    ):
+        raise FineTuneInputError(
+            f"{role} row {index} has invalid unknown_frame_mask", 20
+        )
@@ _weighted_window(... WindowSpec(...))
+            unknown_frame_mask=tuple(
+                row.get("unknown_frame_mask", [False] * int(row["num_frames"]))[
+                    local_start:local_start + window_frames
+                ]
+            ),
@@ weighted_masked_cross_entropy(...)
+    frame_loss_mask: torch.Tensor,
@@
     valid_target = validity_mask.gather(1, targets).bool()
+    if frame_loss_mask.shape != targets.shape:
+        raise ValueError("frame_loss_mask must be [B,T]")
+    valid_target &= frame_loss_mask.bool()
@@ _merge_batches
-        "frames", "targets", "validity_mask", "sample_weight", "is_pseudo",
+        "frames", "targets", "validity_mask", "frame_loss_mask", "sample_weight", "is_pseudo",
@@ training call
+            frame_loss_mask=batch["frame_loss_mask"].to(device),

# Required tests: rejected/pending teacher frame has zero gradient; adjacent
# background retains gradient; class-level loss_validity_mask behavior is
# unchanged; accepted 0.25/0.5 event-centered rows retain their sample weight;
# normalized aggregate pseudo loss remains capped at human loss.
"""


def write_corpus_artifacts(
    output_dir: Path,
    manifest: dict[str, Any],
    report: dict[str, Any],
    contexts: list[dict[str, Any]],
    compare_contexts: list[dict[str, Any]],
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_bytes = _json_bytes(manifest)
    manifest_sha256 = _sha256_bytes(manifest_bytes)
    context_bytes = _context_bytes(contexts)
    compare_bytes = _context_bytes(compare_contexts)
    filtered = _filtered_placeholder(manifest)
    filtered["source_staging_manifest_sha256"] = manifest_sha256
    artifacts = {
        "manifest.json": manifest_bytes,
        "corpus_report.json": _json_bytes({
            **report,
            "manifest_sha256": manifest_sha256,
            "teacher_event_context_sha256": _sha256_bytes(context_bytes),
            "compare_only_teacher_event_context_sha256": _sha256_bytes(compare_bytes),
        }),
        "teacher_event_context.jsonl": context_bytes,
        "compare_only_teacher_event_context.jsonl": compare_bytes,
        "pbvision_filtered_teacher_manifest.json": _json_bytes(filtered),
        "ABC_STAGE.md": _abc_stage_markdown().encode(),
        "VM_ABC_RUN.md": _vm_abc_run_markdown().encode(),
        "CORPUS_NOTES.md": _corpus_notes(manifest_sha256, report).encode(),
        "LOADER_CHANGE_REQUIRED.diff": _loader_required_diff().encode(),
    }
    for name, content in artifacts.items():
        (output_dir / name).write_bytes(content)
    return {
        "manifest_sha256": manifest_sha256,
        "teacher_event_context_sha256": _sha256_bytes(context_bytes),
        "compare_only_teacher_event_context_sha256": _sha256_bytes(compare_bytes),
        "filtered_placeholder_sha256": _sha256_bytes(
            artifacts["pbvision_filtered_teacher_manifest.json"]
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--media-root", type=Path, default=DEFAULT_MEDIA_ROOT)
    parser.add_argument("--frame-times-root", type=Path, default=DEFAULT_FRAME_TIMES_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--window-frames", type=int, default=DEFAULT_WINDOW_FRAMES)
    parser.add_argument("--stride-frames", type=int, default=DEFAULT_WINDOW_STRIDE)
    args = parser.parse_args()
    try:
        built = build_corpus(
            args.input_root,
            media_root=args.media_root,
            frame_times_root=args.frame_times_root,
            seed=args.seed,
            window_frames=args.window_frames,
            stride_frames=args.stride_frames,
        )
        manifest = built[0]
        hashes = write_corpus_artifacts(args.output_dir, *built)
    except (CorpusBuildError, OSError) as exc:
        parser.exit(2, f"pb.vision teacher corpus rejected: {exc}\n")
    print(json.dumps({
        "output_dir": _portable_path(args.output_dir),
        "included_videos": len(manifest["rows"]),
        "manifest_events": manifest["totals"]["manifest_events"],
        "training_eligible_events": 0,
        "verified": False,
        "training_ready": False,
        "teacher_derived": True,
        **hashes,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

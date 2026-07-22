#!/usr/bin/env python3
"""Materialize SHA-bound pb.vision A/B/C pseudo-label manifests.

This is a data-construction tool, not a scorer. It consumes only a rebuilt
teacher corpus plus explicitly parameterized media, frame-times, audio-onset,
and image-space ball-velocity-kink artifacts. Arm B keeps agreement-supported
teacher events; arm C keeps the same pixel windows, classes, weights, and update
budget while moving each focal label within its immutable source rally.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import sys
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.event_head.datasets import (  # noqa: E402
    DatasetFormatError,
    validate_current_manifest,
)
from threed.racketsport.ball_inflections import (  # noqa: E402
    build_ball_inflections_from_ball_track_file,
)


DEFAULT_TEACHER_MANIFEST = (
    ROOT / "runs/lanes/pbv_corpus_rebuild_20260720/manifest.json"
)
DEFAULT_OUTPUT_DIR = ROOT / "runs/lanes/w1b_abc_loader_20260721/abc_materialized"
DEFAULT_SEED = 20260720
DEFAULT_MAX_DELTA_S = 0.035
AUDIO_NULL_SHIFT_COUNT = 20
AUDIO_NULL_MIN_ABS_OFFSET_S = 1.0
AUDIO_NULL_OFFSET_QUANTUM_S = 1_000_000_000
AUDIO_NULL_MIN_OBSERVED_MATCHES = 2
EXPECTED_WINDOW_FRAMES = 64
SIGNAL_FAMILIES = ("audio_onset", "ball_velocity_kink")
EXPECTED_AUDIO_CONTRACT = {
    "schema_version": 1,
    "artifact_type": "racketsport_audio_onsets",
    "detector_version": "audio_onset_pop_v2",
    "source": "video_audio_pop_v2",
}
EXPECTED_BALL_CONTRACT = {
    "schema_version": 1,
    "artifact_type": "racketsport_ball_inflections",
    "source": "ball_track_image_motion",
    "world_frame": "image_xy",
}
COMPARE_ONLY_HOLDOUTS = {"83gyqyc10y8f", "iottnc0h3ekn", "o4dee9dn0ccr"}
SCRIPT_PATH = Path(__file__).resolve()


class ABCMaterializationError(ValueError):
    """Raised when an A/B/C input cannot satisfy the frozen contract."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_unchanged_file(
    path: Path, *, expected_sha256: str, label: str
) -> None:
    try:
        current_sha256 = _sha256_file(path)
    except OSError as exc:
        raise ABCMaterializationError(
            f"{label} disappeared during materialization: {path}"
        ) from exc
    if current_sha256 != expected_sha256:
        raise ABCMaterializationError(
            f"{label} changed during materialization: "
            f"{current_sha256} != {expected_sha256}"
        )


def _json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _canonical_sha256(payload: Any) -> str:
    return _sha256_bytes(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    )


def _normalized_non_path_identity_text(payload: Any) -> str:
    parts: list[str] = []

    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                normalized_key = "".join(
                    character
                    for character in str(key).lower()
                    if character.isalnum()
                )
                if normalized_key == "path" and isinstance(child, str):
                    continue
                parts.append(normalized_key)
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
        elif isinstance(value, str):
            parts.append("".join(
                character for character in value.lower() if character.isalnum()
            ))

    visit(payload)
    return "".join(parts)


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def _load_json(path: Path, *, label: str) -> tuple[Any, bytes]:
    if not path.is_file():
        raise ABCMaterializationError(f"{label} is missing: {path}")
    raw = path.read_bytes()
    try:
        return json.loads(raw), raw
    except json.JSONDecodeError as exc:
        raise ABCMaterializationError(f"{label} is invalid JSON: {path}: {exc}") from exc


def _finite_nonnegative(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ABCMaterializationError(f"{field} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise ABCMaterializationError(f"{field} must be finite and nonnegative")
    return parsed


def _finite_number(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ABCMaterializationError(f"{field} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ABCMaterializationError(f"{field} must be finite")
    return parsed


def _positive_integer(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ABCMaterializationError(f"{field} must be a positive integer")
    return value


def _sha256_digest(value: Any, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ABCMaterializationError(
            f"{field} must be a lowercase 64-character SHA-256 digest"
        )
    return value


def parse_path_bindings(values: Sequence[str], *, flag: str) -> dict[str, Path]:
    """Parse repeatable VIDEO_ID=PATH CLI values without guessing paths."""

    bindings: dict[str, Path] = {}
    for value in values:
        video_id, separator, raw_path = value.partition("=")
        if not separator or not video_id or not raw_path:
            raise ABCMaterializationError(
                f"{flag} must use VIDEO_ID=PATH, got {value!r}"
            )
        if video_id in bindings:
            raise ABCMaterializationError(f"duplicate {flag} binding for {video_id}")
        bindings[video_id] = Path(raw_path)
    return bindings


def _validate_teacher_manifest(manifest: Any) -> list[dict[str, Any]]:
    if not isinstance(manifest, dict):
        raise ABCMaterializationError("teacher manifest must be an object")
    if (
        type(manifest.get("schema_version")) is not int
        or manifest.get("schema_version") != 2
        or manifest.get("artifact_type")
        != "event_head_pbvision_teacher_staging_dataset_manifest"
        or manifest.get("teacher_derived") is not True
        or manifest.get("ground_truth") is not False
        or manifest.get("verified") is not False
    ):
        raise ABCMaterializationError(
            "teacher manifest must be the unverified schema-v2 pb.vision staging corpus"
        )
    try:
        validate_current_manifest(manifest)
    except DatasetFormatError as exc:
        raise ABCMaterializationError(f"teacher manifest schema rejected: {exc}") from exc
    denied = set(manifest.get("permanent_compare_only_denylist", []))
    if denied != COMPARE_ONLY_HOLDOUTS:
        raise ABCMaterializationError("teacher manifest compare-only denylist drift")
    source_records = manifest.get("provenance", {}).get("sources")
    if not isinstance(source_records, list):
        raise ABCMaterializationError("teacher manifest lacks provenance.sources")
    compare_records = [
        item for item in source_records
        if isinstance(item, dict) and item.get("compare_only") is True
    ]
    present_ids = {
        item.get("video_id") for item in source_records if isinstance(item, dict)
    }
    if {item.get("video_id") for item in compare_records} != (
        COMPARE_ONLY_HOLDOUTS & present_ids
    ):
        raise ABCMaterializationError("compare-only provenance inventory drift")
    compare_hashes = {item.get("source_video_sha256") for item in compare_records}
    if any(not isinstance(value, str) or len(value) != 64 for value in compare_hashes):
        raise ABCMaterializationError("compare-only provenance lacks source SHA-256")
    rows = [row for row in manifest["rows"] if row["split"] == "train"]
    if not rows:
        raise ABCMaterializationError("teacher manifest has no train-split rows")
    if any(row["source_video"] in COMPARE_ONLY_HOLDOUTS for row in rows):
        raise ABCMaterializationError("compare-only source reached train rows")
    if any(row["source_video_sha256"] in compare_hashes for row in rows):
        raise ABCMaterializationError("compare-only media SHA reached train rows")
    if len({row["source_video"] for row in rows}) != len(rows):
        raise ABCMaterializationError("teacher train rows must be one row per source video")
    window_frames = manifest.get("config", {}).get("window_frames")
    if window_frames != EXPECTED_WINDOW_FRAMES:
        raise ABCMaterializationError(
            f"teacher window context must be {EXPECTED_WINDOW_FRAMES}, got {window_frames}"
        )
    return sorted(rows, key=lambda row: str(row["source_video"]))


def build_vm_needs(
    manifest: Mapping[str, Any],
    *,
    teacher_manifest_path: Path,
    media_paths: Mapping[str, Path] | None = None,
    frame_times_paths: Mapping[str, Path] | None = None,
    audio_paths: Mapping[str, Path] | None = None,
    ball_paths: Mapping[str, Path] | None = None,
) -> dict[str, Any]:
    """Describe every per-clip artifact the VM must stage before materialization."""

    rows = _validate_teacher_manifest(dict(manifest))
    provided = {
        "media": media_paths or {},
        "frame_times": frame_times_paths or {},
        "audio_onsets": audio_paths or {},
        "ball_velocity_kinks": ball_paths or {},
    }
    clips = []
    for row in rows:
        video_id = str(row["source_video"])
        requirements = {}
        for name, flag in (
            ("media", "--media"),
            ("frame_times", "--frame-times"),
            ("audio_onsets", "--audio-onsets"),
            ("ball_velocity_kinks", "--ball-velocity-kinks"),
        ):
            path = provided[name].get(video_id)
            requirements[name] = {
                "cli_flag": flag,
                "required": True,
                "provided": path is not None,
                "path": _portable_path(path) if path is not None else None,
                "sha256_required": True,
            }
        clips.append({
            "video_id": video_id,
            "source_video_sha256": row["source_video_sha256"],
            "source_lineage_key": row["source_lineage_key"],
            "split": "train",
            "required_artifacts": requirements,
        })
    return {
        "schema_version": 1,
        "artifact_type": "pbvision_abc_vm_needs",
        "verified": False,
        "no_scoring": True,
        "teacher_manifest": {
            "path": _portable_path(teacher_manifest_path),
            "sha256": _sha256_file(teacher_manifest_path),
        },
        "permanent_compare_only_sha256_denylist": sorted(
            item["source_video_sha256"]
            for item in manifest["provenance"]["sources"]
            if item.get("compare_only") is True
        ),
        "required_train_clips": len(clips),
        "clips": clips,
    }


def _require_exact_bindings(
    rows: Sequence[Mapping[str, Any]],
    *,
    label: str,
    bindings: Mapping[str, Path],
) -> None:
    expected = {str(row["source_video"]) for row in rows}
    missing = sorted(expected - set(bindings))
    extra = sorted(set(bindings) - expected)
    if missing or extra:
        raise ABCMaterializationError(
            f"{label} bindings must exactly cover train clips; missing={missing}, extra={extra}"
        )


def _validate_frame_times(
    path: Path, *, row: Mapping[str, Any], media_sha256: str
) -> tuple[list[float], float, dict[str, Any], bytes]:
    payload, raw = _load_json(path, label="frame-times artifact")
    if not isinstance(payload, dict) or not isinstance(payload.get("frames"), list):
        raise ABCMaterializationError(f"invalid frame-times payload: {path}")
    if (
        type(payload.get("schema_version")) is not int
        or payload.get("schema_version") != 1
        or payload.get("artifact_type") != "racketsport_frame_times"
    ):
        raise ABCMaterializationError(
            "frame-times artifact must be schema-v1 racketsport_frame_times"
        )
    frames = payload["frames"]
    declared_frame_count = _positive_integer(
        payload.get("frame_count"), field="frame_times.frame_count"
    )
    if (
        declared_frame_count != len(frames)
        or declared_frame_count != int(row["num_frames"])
    ):
        raise ABCMaterializationError(
            f"frame-times count mismatch for {row['source_video']}: "
            f"declared={declared_frame_count}, frames={len(frames)}, "
            f"row={row['num_frames']}"
        )
    times: list[float] = []
    for expected, item in enumerate(frames):
        if (
            not isinstance(item, dict)
            or type(item.get("frame")) is not int
            or item.get("frame") != expected
        ):
            raise ABCMaterializationError(
                f"frame-times indices are not contiguous for {row['source_video']}"
            )
        times.append(
            _finite_nonnegative(
                item.get("pts_s"), field=f"frame_times.frames[{expected}].pts_s"
            )
        )
    if any(right <= left for left, right in zip(times, times[1:])):
        raise ABCMaterializationError(
            f"frame-times PTS are not strictly increasing for {row['source_video']}"
        )
    if len(times) < 2:
        raise ABCMaterializationError(
            "frame-times requires at least two verified PTS values for duration"
        )
    declared_duration_s = _finite_nonnegative(
        payload.get("duration_s"), field="frame_times.duration_s"
    )
    if declared_duration_s <= 0.0:
        raise ABCMaterializationError("frame_times.duration_s must be positive")
    pts_span_s = times[-1] - times[0]
    terminal_reference_s = times[-1] - times[-2]
    terminal_frame_duration_s = terminal_reference_s
    circular_period_s = pts_span_s + terminal_frame_duration_s
    declared_duration_error_s = declared_duration_s - circular_period_s
    cadence_tolerance_s = max(0.000005, terminal_reference_s * 0.001)
    if abs(declared_duration_error_s) > cadence_tolerance_s:
        raise ABCMaterializationError(
            "frame-times duration is inconsistent with verified terminal PTS "
            f"for {row['source_video']}: declared={declared_duration_s}, "
            f"PTS-derived={circular_period_s}, "
            f"tolerance={cadence_tolerance_s}"
        )
    frame_times_sha256 = _sha256_bytes(raw)
    binding = row.get("timebase_conversion", {}).get("pts_media_binding")
    expected_binding_sha = _canonical_sha256({
        "source_video_sha256": media_sha256,
        "frame_times_sha256": frame_times_sha256,
    })
    if (
        not isinstance(binding, dict)
        or binding.get("status") != "sha256_bound"
        or binding.get("source_video_sha256") != media_sha256
        or binding.get("frame_times_sha256") != frame_times_sha256
        or binding.get("binding_sha256") != expected_binding_sha
        or row.get("timebase_conversion", {}).get("needs_pts_verify") is not False
    ):
        raise ABCMaterializationError(
            f"PTS artifact is not SHA-bound to staged media for {row['source_video']}"
        )
    declared_media_values = {
        key: _sha256_digest(payload[key], field=f"frame_times.{key}")
        for key in ("source_video_sha256", "media_sha256")
        if payload.get(key) is not None
    }
    if not declared_media_values:
        raise ABCMaterializationError(
            f"frame-times must declare staged media SHA for {row['source_video']}"
        )
    if any(value != media_sha256 for value in declared_media_values.values()):
        raise ABCMaterializationError(
            f"frame-times declares the wrong media SHA for {row['source_video']}"
        )
    declared_media_sha = next(iter(declared_media_values.values()))
    return times, circular_period_s, {
        "path": _portable_path(path),
        "sha256": frame_times_sha256,
        "schema_version": 1,
        "artifact_type": "racketsport_frame_times",
        "frame_count": declared_frame_count,
        "declared_media_sha256": declared_media_sha,
        "staged_media_sha256": media_sha256,
        "media_sha256": media_sha256,
        "binding_sha256": expected_binding_sha,
        "duration_s": declared_duration_s,
        "declared_duration_s": declared_duration_s,
        "declared_duration_error_s": declared_duration_error_s,
        "pts_span_s": pts_span_s,
        "terminal_pts_interval_s": terminal_reference_s,
        "validated_terminal_frame_duration_s": terminal_frame_duration_s,
        "circular_period_s": circular_period_s,
        "pts_origin_s": times[0],
    }, raw


def _cue_time(item: Mapping[str, Any], *, field: str) -> float:
    for key in ("corrected_time_s", "time_s", "pts_s", "source_pts_s"):
        if key in item:
            return _finite_nonnegative(item[key], field=f"{field}.{key}")
    raise ABCMaterializationError(f"{field} has no corrected_time_s/time_s/pts_s")


def _require_exact_contract(
    payload: Mapping[str, Any],
    *,
    family: str,
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    mismatches = {
        key: {"expected": expected_value, "declared": payload.get(key)}
        for key, expected_value in expected.items()
        if (
            type(payload.get(key)) is not type(expected_value)
            or payload.get(key) != expected_value
        )
    }
    if mismatches:
        raise ABCMaterializationError(
            f"{family} artifact contract mismatch: {mismatches}"
        )
    if "family" in payload and payload.get("family") != family:
        raise ABCMaterializationError(
            f"{family} artifact declares incompatible family {payload.get('family')!r}"
        )
    return dict(expected)


def _declared_media_identity(
    payload: Mapping[str, Any], *, family: str, media_sha256: str
) -> str:
    declared = {
        key: _sha256_digest(payload[key], field=f"{family}.{key}")
        for key in ("source_video_sha256", "media_sha256")
        if payload.get(key) is not None
    }
    if not declared:
        raise ABCMaterializationError(
            f"{family} artifact must declare source_video_sha256/media_sha256"
        )
    if any(value != media_sha256 for value in declared.values()):
        raise ABCMaterializationError(f"{family} artifact media SHA mismatch")
    return next(iter(declared.values()))


def _validate_pts_source(
    payload: Mapping[str, Any],
    *,
    family: str,
    frame_times_path: Path,
    frame_times_sha256: str,
    media_sha256: str,
) -> dict[str, str]:
    pts_source = payload.get("pts_source")
    if not isinstance(pts_source, Mapping):
        raise ABCMaterializationError(f"{family} artifact lacks pts_source identity")
    declared_path = pts_source.get("path")
    if not isinstance(declared_path, str) or not declared_path.strip():
        raise ABCMaterializationError(f"{family}.pts_source.path must be nonempty")
    if Path(declared_path).resolve() != frame_times_path.resolve():
        raise ABCMaterializationError(
            f"{family} pts_source.path does not identify the staged frame-times artifact"
        )
    declared_pts_sha = _sha256_digest(
        pts_source.get("sha256"), field=f"{family}.pts_source.sha256"
    )
    declared_pts_media_sha = _sha256_digest(
        pts_source.get("source_video_sha256"),
        field=f"{family}.pts_source.source_video_sha256",
    )
    if declared_pts_sha != frame_times_sha256:
        raise ABCMaterializationError(f"{family} pts_source SHA mismatch")
    if declared_pts_media_sha != media_sha256:
        raise ABCMaterializationError(f"{family} pts_source media SHA mismatch")
    return {
        "path": declared_path,
        "sha256": declared_pts_sha,
        "source_video_sha256": declared_pts_media_sha,
    }


def _validate_ball_track_source(
    payload: Mapping[str, Any],
    *,
    media_sha256: str,
    frame_times_sha256: str,
) -> tuple[dict[str, Any], bytes]:
    ball_chain_text = _normalized_non_path_identity_text(payload)
    if "pbvision" in ball_chain_text:
        raise ABCMaterializationError(
            "pb.vision-derived provenance is forbidden in the BALL artifact chain"
        )
    if "audio" in ball_chain_text:
        raise ABCMaterializationError(
            "audio-derived provenance is forbidden in the BALL artifact chain"
        )
    ball_track_source = payload.get("ball_track_source")
    if not isinstance(ball_track_source, Mapping):
        raise ABCMaterializationError(
            "ball_velocity_kink artifact lacks ball_track_source provenance"
        )
    declared_path = ball_track_source.get("path")
    if not isinstance(declared_path, str) or not declared_path.strip():
        raise ABCMaterializationError("ball_track_source.path must be nonempty")
    declared_sha = _sha256_digest(
        ball_track_source.get("sha256"), field="ball_track_source.sha256"
    )
    upstream_path = Path(declared_path)
    ball_track, ball_track_raw = _load_json(
        upstream_path, label="ball_track_source"
    )
    if _sha256_bytes(ball_track_raw) != declared_sha:
        raise ABCMaterializationError("ball_track_source SHA mismatch")
    if not isinstance(ball_track, Mapping):
        raise ABCMaterializationError("ball_track_source must contain an object")
    if (
        type(ball_track.get("schema_version")) is not int
        or ball_track.get("schema_version") != 1
    ):
        raise ABCMaterializationError(
            "ball_track_source must be a schema-v1 WASB track"
        )
    declared_source = ball_track.get("source")
    normalized_track_text = _normalized_non_path_identity_text(ball_track)
    if "pbvision" in normalized_track_text:
        raise ABCMaterializationError(
            "pb.vision-derived ball tracks are forbidden as independent agreement"
        )
    if declared_source != "wasb":
        raise ABCMaterializationError(
            "ball_track_source.source must be exactly 'wasb'"
        )
    if "audio" in normalized_track_text:
        raise ABCMaterializationError(
            "audio-derived provenance is forbidden in a WASB ball track"
        )
    nested_sources: list[tuple[str, Any]] = []

    def collect_nested_sources(value: Any, path: tuple[str, ...] = ()) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                key_text = str(key)
                child_path = (*path, key_text)
                if path and key_text.lower() == "source":
                    nested_sources.append((".".join(child_path), child))
                collect_nested_sources(child, child_path)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                collect_nested_sources(child, (*path, str(index)))

    collect_nested_sources(ball_track)
    contradictory_sources = [
        {"path": path, "declared": value}
        for path, value in nested_sources
        if value != "wasb"
    ]
    if contradictory_sources:
        raise ABCMaterializationError(
            "ball_track_source contains contradictory nested source provenance: "
            f"{contradictory_sources}"
        )
    declared_track_media_sha = _declared_media_identity(
        ball_track,
        family="ball_track_source",
        media_sha256=media_sha256,
    )
    declared_track_pts_sha = _sha256_digest(
        ball_track.get("frame_times_sha256"),
        field="ball_track_source.frame_times_sha256",
    )
    if declared_track_pts_sha != frame_times_sha256:
        raise ABCMaterializationError("ball_track_source frame-times SHA mismatch")
    if not isinstance(ball_track.get("frames"), list):
        raise ABCMaterializationError("ball_track_source.frames must be an array")
    return {
        "path": declared_path,
        "sha256": declared_sha,
        "declared_source": declared_source,
        "declared_artifact_type": ball_track.get("artifact_type"),
        "declared_media_sha256": declared_track_media_sha,
        "declared_frame_times_sha256": declared_track_pts_sha,
        "frame_count": len(ball_track["frames"]),
    }, ball_track_raw


def _validate_ball_derivation(
    payload: Mapping[str, Any],
    *,
    ball_track_path: Path,
    ball_track_raw: bytes,
    ball_track_sha256: str,
    frame_times_path: Path,
    frame_times_raw: bytes,
    frame_times_sha256: str,
) -> dict[str, Any]:
    summary = payload.get("summary")
    if not isinstance(summary, Mapping):
        raise ABCMaterializationError(
            "ball_velocity_kink artifact lacks derivation summary"
        )
    parameter_names = (
        "min_turn_degrees",
        "min_speed_px_per_s",
        "max_neighbor_gap_s",
        "min_candidate_separation_s",
    )
    parameters = {
        name: _finite_nonnegative(
            summary.get(name), field=f"ball_velocity_kink.summary.{name}"
        )
        for name in parameter_names
    }
    if parameters["max_neighbor_gap_s"] <= 0.0:
        raise ABCMaterializationError(
            "ball_velocity_kink.summary.max_neighbor_gap_s must be positive"
        )
    try:
        with tempfile.TemporaryDirectory(prefix="abc-ball-derivation-") as temp_dir:
            snapshot_dir = Path(temp_dir)
            ball_track_snapshot = snapshot_dir / "ball_track.json"
            frame_times_snapshot = snapshot_dir / "frame_times.json"
            ball_track_snapshot.write_bytes(ball_track_raw)
            frame_times_snapshot.write_bytes(frame_times_raw)
            rebuilt = build_ball_inflections_from_ball_track_file(
                ball_track_snapshot,
                frame_times_path=frame_times_snapshot,
                min_turn_degrees=parameters["min_turn_degrees"],
                min_speed_px_per_s=parameters["min_speed_px_per_s"],
                max_neighbor_gap_s=parameters["max_neighbor_gap_s"],
                min_candidate_separation_s=parameters[
                    "min_candidate_separation_s"
                ],
            )
    except (OSError, ValueError, TypeError) as exc:
        raise ABCMaterializationError(
            f"ball_velocity_kink upstream rebuild failed: {exc}"
        ) from exc
    try:
        summary_matches = _canonical_sha256(rebuilt.get("summary")) == (
            _canonical_sha256(dict(summary))
        )
        candidates_match = _canonical_sha256(rebuilt.get("candidates")) == (
            _canonical_sha256(payload.get("candidates"))
        )
    except (TypeError, ValueError) as exc:
        raise ABCMaterializationError(
            f"ball_velocity_kink derivation contains noncanonical JSON: {exc}"
        ) from exc
    if not summary_matches:
        raise ABCMaterializationError(
            "ball_velocity_kink derivation summary does not match upstream rebuild"
        )
    if not candidates_match:
        raise ABCMaterializationError(
            "ball_velocity_kink candidates do not match authenticated upstream rebuild"
        )

    frame_times = json.loads(frame_times_raw)
    frames = frame_times.get("frames") if isinstance(frame_times, Mapping) else None
    if not isinstance(frames, list):
        raise ABCMaterializationError(
            "ball_velocity_kink frame-times lacks frames"
        )
    for candidate_index, candidate in enumerate(payload.get("candidates", [])):
        frame = candidate.get("frame") if isinstance(candidate, Mapping) else None
        if (
            isinstance(frame, bool)
            or not isinstance(frame, int)
            or not 0 <= frame < len(frames)
        ):
            raise ABCMaterializationError(
                f"ball_velocity_kink candidate {candidate_index} frame escaped PTS"
            )
        frame_entry = frames[frame]
        if not isinstance(frame_entry, Mapping):
            raise ABCMaterializationError(
                f"frame-times entry {frame} is not an object"
            )
        candidate_time_s = _cue_time(
            candidate, field=f"ball_velocity_kink[{candidate_index}]"
        )
        frame_pts_s = _finite_nonnegative(
            frame_entry.get("pts_s"), field=f"frame_times.frames[{frame}].pts_s"
        )
        if not math.isclose(
            candidate_time_s, frame_pts_s, rel_tol=0.0, abs_tol=0.000001
        ):
            raise ABCMaterializationError(
                "ball_velocity_kink candidate time is not tied to verified frame PTS"
            )
    _require_unchanged_file(
        ball_track_path,
        expected_sha256=ball_track_sha256,
        label="ball_track_source",
    )
    _require_unchanged_file(
        frame_times_path,
        expected_sha256=frame_times_sha256,
        label="frame-times artifact",
    )
    return {
        "status": "exact_upstream_rebuild_match",
        "candidate_count": len(payload.get("candidates", [])),
        "parameters": parameters,
        "candidate_pts_tolerance_s": 0.000001,
    }


def _cue_items(payload: Any, *, family: str, path: Path) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        raise ABCMaterializationError(f"{family} artifact must be an object: {path}")
    keys = ("onsets",) if family == "audio_onset" else ("candidates",)
    items: Any = None
    for key in keys:
        if key in payload:
            items = payload[key]
            break
    if not isinstance(items, list):
        raise ABCMaterializationError(
            f"{family} artifact must contain one of {keys}: {path}"
        )
    cues: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ABCMaterializationError(f"{family} cue {index} must be an object")
        if "family" in item and item.get("family") != family:
            raise ABCMaterializationError(
                f"{family} cue {index} declares incompatible family "
                f"{item.get('family')!r}"
            )
        time_s = _cue_time(item, field=f"{family}[{index}]")
        if family == "ball_velocity_kink":
            if item.get("source") != "ball_track_image_motion":
                raise ABCMaterializationError(
                    f"ball_velocity_kink cue {index} lacks image-motion source"
                )
            image_xy = item.get("ball_image_xy")
            if (
                not isinstance(image_xy, Sequence)
                or isinstance(image_xy, (str, bytes))
                or len(image_xy) != 2
            ):
                raise ABCMaterializationError(
                    f"ball_velocity_kink cue {index} lacks ball_image_xy provenance"
                )
            normalized_image_xy = [
                _finite_number(value, field=f"ball_velocity_kink[{index}].ball_image_xy")
                for value in image_xy
            ]
            frame = item.get("frame")
            if isinstance(frame, bool) or not isinstance(frame, int) or frame < 0:
                raise ABCMaterializationError(
                    f"ball_velocity_kink cue {index}.frame must be nonnegative integer"
                )
            cue_provenance = {
                "source": "ball_track_image_motion",
                "world_frame": "image_xy",
                "frame": frame,
                "ball_image_xy": normalized_image_xy,
                "source_index": index,
            }
        else:
            if item.get("source") != "audio_pop_v2":
                raise ABCMaterializationError(
                    f"audio_onset cue {index} lacks audio_pop_v2 source"
                )
            cue_provenance = {
                "source": "audio_pop_v2",
                "source_index": index,
            }
        stable_id = item.get("cue_id", item.get("event_id", item.get("id")))
        if stable_id is None:
            stable_id = _canonical_sha256({"index": index, "cue": item})
        cues.append({
            "stable_id": str(stable_id),
            "time_s": time_s,
            "source_index": index,
            "cue_provenance": cue_provenance,
        })
    if any(
        right["time_s"] <= left["time_s"]
        for left, right in zip(cues, cues[1:])
    ):
        raise ABCMaterializationError(
            f"{family} cue times must be strictly increasing: {path}"
        )
    cues.sort(key=lambda item: (item["time_s"], item["stable_id"]))
    return cues


def _load_cues(
    path: Path,
    *,
    family: str,
    video_id: str,
    media_sha256: str,
    frame_times_path: Path,
    frame_times_sha256: str,
    frame_times_raw: bytes,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload, raw = _load_json(path, label=f"{family} artifact")
    if not isinstance(payload, dict):
        raise ABCMaterializationError(f"{family} artifact must be an object")
    declared_video = payload.get("video_id", payload.get("clip"))
    if declared_video is not None and str(declared_video) != video_id:
        raise ABCMaterializationError(
            f"{family} artifact clip mismatch: {declared_video!r} != {video_id!r}"
        )
    expected_contract = (
        EXPECTED_AUDIO_CONTRACT
        if family == "audio_onset"
        else EXPECTED_BALL_CONTRACT
    )
    contract = _require_exact_contract(
        payload, family=family, expected=expected_contract
    )
    declared_media_sha = _declared_media_identity(
        payload, family=family, media_sha256=media_sha256
    )
    declared_pts_sha = _sha256_digest(
        payload.get("frame_times_sha256"),
        field=f"{family}.frame_times_sha256",
    )
    if declared_pts_sha != frame_times_sha256:
        raise ABCMaterializationError(f"{family} artifact frame-times SHA mismatch")
    pts_source = _validate_pts_source(
        payload,
        family=family,
        frame_times_path=frame_times_path,
        frame_times_sha256=frame_times_sha256,
        media_sha256=media_sha256,
    )
    ball_track_validation = (
        _validate_ball_track_source(
            payload,
            media_sha256=media_sha256,
            frame_times_sha256=frame_times_sha256,
        )
        if family == "ball_velocity_kink"
        else None
    )
    ball_track_source = (
        ball_track_validation[0]
        if ball_track_validation is not None
        else None
    )
    ball_track_raw = (
        ball_track_validation[1]
        if ball_track_validation is not None
        else None
    )
    ball_derivation = (
        _validate_ball_derivation(
            payload,
            ball_track_path=Path(ball_track_source["path"]),
            ball_track_raw=ball_track_raw,
            ball_track_sha256=ball_track_source["sha256"],
            frame_times_path=frame_times_path,
            frame_times_raw=frame_times_raw,
            frame_times_sha256=frame_times_sha256,
        )
        if ball_track_source is not None
        else None
    )
    cues = _cue_items(payload, family=family, path=path)
    artifact_sha256 = _sha256_bytes(raw)
    dependency_identity = {
        "artifact_sha256": artifact_sha256,
        "source_video_sha256": media_sha256,
        "frame_times_sha256": frame_times_sha256,
        "pts_source_sha256": pts_source["sha256"],
        **(
            {"ball_track_source_sha256": ball_track_source["sha256"]}
            if ball_track_source is not None
            else {}
        ),
    }
    binding = {
        "path": _portable_path(path),
        "sha256": artifact_sha256,
        "cue_count": len(cues),
        "artifact_contract": contract,
        "declared_media_sha256": declared_media_sha,
        "declared_frame_times_sha256": declared_pts_sha,
        "pts_source": pts_source,
        "dependency_binding_sha256": _canonical_sha256(dependency_identity),
        "cue_provenance_contract": (
            {
                "source": "audio_pop_v2",
                "time_space": "corrected_audio_time_s",
            }
            if family == "audio_onset"
            else {
                "source": "ball_track_image_motion",
                "world_frame": "image_xy",
                "coordinate_field": "ball_image_xy",
            }
        ),
    }
    if ball_track_source is not None:
        binding["ball_track_source"] = ball_track_source
        binding["derivation_validation"] = ball_derivation
    return cues, binding


def _validate_and_index_events(
    events: Any, *, video_id: str
) -> list[tuple[int, Mapping[str, Any]]]:
    if not isinstance(events, list):
        raise ABCMaterializationError(f"events must be an array for {video_id}")
    seen_strings: set[str] = set()
    seen_coerced: dict[str, int] = {}
    for event_index, event in enumerate(events):
        if not isinstance(event, Mapping):
            raise ABCMaterializationError(
                f"event {event_index} must be an object for {video_id}"
            )
        raw_event_id = event.get("event_id")
        if isinstance(raw_event_id, str) and raw_event_id in seen_strings:
            raise ABCMaterializationError(
                f"duplicate event_id {raw_event_id!r} for {video_id}"
            )
        coerced_event_id = str(raw_event_id)
        if coerced_event_id in seen_coerced:
            raise ABCMaterializationError(
                "string-coercion event_id collision for "
                f"{video_id}: indices {seen_coerced[coerced_event_id]} and "
                f"{event_index} both map to {coerced_event_id!r}"
            )
        seen_coerced[coerced_event_id] = event_index
        if isinstance(raw_event_id, str):
            seen_strings.add(raw_event_id)

    indexed: list[tuple[int, Mapping[str, Any]]] = []
    for event_index, event in enumerate(events):
        raw_event_id = event.get("event_id")
        if not isinstance(raw_event_id, str) or not raw_event_id.strip():
            raise ABCMaterializationError(
                f"event_id must be a nonempty string for {video_id} index {event_index}"
            )
        indexed.append((event_index, event))
    return indexed


def _match_family(
    events: Sequence[tuple[int, Mapping[str, Any]]],
    cues: Sequence[Mapping[str, Any]],
    *,
    family: str,
    max_delta_s: float,
) -> dict[int, dict[str, Any]]:
    candidates: list[tuple[float, int, str, float, str, int]] = []
    for event_index, event in events:
        event_id = event["event_id"]
        event_time = _finite_nonnegative(
            event.get("source_pts_s"), field=f"event {event_id}.source_pts_s"
        )
        for cue_index, cue in enumerate(cues):
            delta = abs(float(cue["time_s"]) - event_time)
            if delta <= max_delta_s:
                candidates.append((
                    delta,
                    event_index,
                    event_id,
                    float(cue["time_s"]),
                    str(cue["stable_id"]),
                    cue_index,
                ))
    candidates.sort()
    used_events: set[int] = set()
    used_cues: set[int] = set()
    matches: dict[int, dict[str, Any]] = {}
    for (
        delta,
        event_index,
        event_id,
        cue_time,
        stable_id,
        cue_index,
    ) in candidates:
        if event_index in used_events or cue_index in used_cues:
            continue
        used_events.add(event_index)
        used_cues.add(cue_index)
        matches[event_index] = {
            "family": family,
            "cue_stable_id": stable_id,
            "cue_time_s": cue_time,
            "absolute_delta_s": delta,
            "cue_provenance": deepcopy(cues[cue_index]["cue_provenance"]),
        }
    return matches


def _validated_emitted_agreement(
    agreement: Mapping[str, Any],
    *,
    event_index: int,
    event: Mapping[str, Any],
    max_delta_s: float,
) -> dict[str, Any]:
    event_time_s = _finite_nonnegative(
        event.get("source_pts_s"),
        field=f"event {event['event_id']}.source_pts_s",
    )
    cue_time_s = _finite_nonnegative(
        agreement.get("cue_time_s"),
        field=f"agreement {event['event_id']}.cue_time_s",
    )
    recomputed_delta_s = abs(cue_time_s - event_time_s)
    stored_delta_s = _finite_nonnegative(
        agreement.get("absolute_delta_s"),
        field=f"agreement {event['event_id']}.absolute_delta_s",
    )
    if (
        recomputed_delta_s > max_delta_s + 1e-12
        or not math.isclose(
            recomputed_delta_s, stored_delta_s, rel_tol=0.0, abs_tol=1e-12
        )
    ):
        raise ABCMaterializationError(
            f"agreement delta no longer belongs to current event {event['event_id']}"
        )
    return {
        **dict(agreement),
        "absolute_delta_s": recomputed_delta_s,
        "matched_event_source_pts_s": event_time_s,
        "source_event_index": event_index,
    }


def _audio_null_shift_offsets(
    *, seed: int, video_id: str, video_duration_s: float
) -> list[float]:
    duration_ns = round(video_duration_s * AUDIO_NULL_OFFSET_QUANTUM_S)
    min_offset_ns = round(
        AUDIO_NULL_MIN_ABS_OFFSET_S * AUDIO_NULL_OFFSET_QUANTUM_S
    )
    if duration_ns <= 2 * min_offset_ns:
        raise ABCMaterializationError(
            "audio time-shift null requires video duration > "
            f"{2 * AUDIO_NULL_MIN_ABS_OFFSET_S:.1f}s"
        )
    admissible_residues = duration_ns - 2 * min_offset_ns + 1
    if admissible_residues < AUDIO_NULL_SHIFT_COUNT:
        raise ABCMaterializationError(
            f"audio time-shift null cannot produce {AUDIO_NULL_SHIFT_COUNT} "
            "unique admissible offsets"
        )
    offsets: list[float] = []
    seen_offsets_ns: set[int] = set()
    draw_index = 0
    while len(offsets) < AUDIO_NULL_SHIFT_COUNT and draw_index < 10_000:
        digest = hashlib.sha256(
            f"abc_audio_null_v1:{seed}:{video_id}:{draw_index}".encode()
        ).digest()
        draw_index += 1
        residue_ns = min_offset_ns + (
            int.from_bytes(digest[:8], "big") % admissible_residues
        )
        signed_ns = (
            residue_ns
            if 2 * residue_ns <= duration_ns
            else residue_ns - duration_ns
        )
        if signed_ns in seen_offsets_ns:
            continue
        seen_offsets_ns.add(signed_ns)
        offsets.append(signed_ns / AUDIO_NULL_OFFSET_QUANTUM_S)
    if len(offsets) != AUDIO_NULL_SHIFT_COUNT:
        raise ABCMaterializationError(
            f"audio time-shift null failed to derive {AUDIO_NULL_SHIFT_COUNT} "
            "unique offsets"
        )
    return offsets


def _audio_time_shift_null(
    eligible_events: Sequence[tuple[int, Mapping[str, Any]]],
    audio_cues: Sequence[Mapping[str, Any]],
    observed_matches: Mapping[int, Mapping[str, Any]],
    *,
    video_id: str,
    seed: int,
    max_delta_s: float,
    pts_origin_s: float,
    video_duration_s: float,
) -> dict[str, Any]:
    offsets = _audio_null_shift_offsets(
        seed=seed, video_id=video_id, video_duration_s=video_duration_s
    )
    interval_end_s = pts_origin_s + video_duration_s
    for cue in audio_cues:
        cue_time_s = float(cue["time_s"])
        if not pts_origin_s <= cue_time_s < interval_end_s:
            raise ABCMaterializationError(
                f"audio_onset cue escaped video interval for {video_id}"
            )
    for _, event in eligible_events:
        event_time_s = _finite_nonnegative(
            event.get("source_pts_s"),
            field=f"event {event['event_id']}.source_pts_s",
        )
        if not pts_origin_s <= event_time_s < interval_end_s:
            raise ABCMaterializationError(
                f"eligible event escaped video interval for {video_id}"
            )

    eligible_count = len(eligible_events)
    observed_count = len(observed_matches)
    null_counts: list[int] = []
    for offset_s in offsets:
        shifted_cues = [
            {
                **dict(cue),
                "time_s": pts_origin_s
                + ((float(cue["time_s"]) - pts_origin_s + offset_s) % video_duration_s),
            }
            for cue in audio_cues
        ]
        shifted_cues.sort(key=lambda item: (item["time_s"], item["stable_id"]))
        null_counts.append(len(_match_family(
            eligible_events,
            shifted_cues,
            family="audio_onset",
            max_delta_s=max_delta_s,
        )))

    if eligible_count:
        observed_rate = observed_count / eligible_count
        null_rates = [count / eligible_count for count in null_counts]
    else:
        observed_rate = 0.0
        null_rates = [0.0 for _ in offsets]
    null_max_rate = max(null_rates)
    support_satisfied = observed_count >= AUDIO_NULL_MIN_OBSERVED_MATCHES
    return {
        "eligible_event_count": eligible_count,
        "observed_match_count": observed_count,
        "minimum_observed_match_count": AUDIO_NULL_MIN_OBSERVED_MATCHES,
        "support_satisfied": support_satisfied,
        "pts_origin_s": pts_origin_s,
        "circular_period_s": video_duration_s,
        "observed_match_rate": observed_rate,
        "shift_offsets_s": offsets,
        "unique_shift_count": len(set(offsets)),
        "null_match_rates": null_rates,
        "null_max_rate": null_max_rate,
        "beats_null": support_satisfied and observed_rate > null_max_rate,
    }


def _weight_for_count(count: int) -> float:
    return 0.0 if count == 0 else 0.25 if count == 1 else 0.5


def _event_window_start(event_frame: int, *, num_frames: int) -> int:
    if num_frames < EXPECTED_WINDOW_FRAMES:
        raise ABCMaterializationError(
            f"source has fewer than {EXPECTED_WINDOW_FRAMES} frames"
        )
    return min(
        max(0, event_frame - EXPECTED_WINDOW_FRAMES // 2),
        num_frames - EXPECTED_WINDOW_FRAMES,
    )


def _materialize_b_row(
    source_row: Mapping[str, Any],
    event: Mapping[str, Any],
    *,
    media_path: Path,
    agreements: Sequence[Mapping[str, Any]],
    agreement_count: int,
    sample_weight: float,
    audio_weight_eligible: bool,
) -> dict[str, Any]:
    event_frame = int(event["frame"])
    source_row_start = int(source_row["source_start_frame"])
    start = _event_window_start(event_frame, num_frames=int(source_row["num_frames"]))
    absolute_start = source_row_start + start
    local_frame = event_frame - start
    unknown = list(
        source_row["unknown_frame_mask"][start:start + EXPECTED_WINDOW_FRAMES]
    )
    if len(unknown) != EXPECTED_WINDOW_FRAMES:
        raise ABCMaterializationError("teacher UNKNOWN mask cannot cover a 64-frame window")
    unknown[local_frame] = False
    agreement_list = [dict(item) for item in sorted(
        agreements, key=lambda item: str(item["family"])
    )]
    recorded_agreement_count = len(agreement_list)
    if agreement_count not in {1, 2} or sample_weight != _weight_for_count(
        agreement_count
    ):
        raise ABCMaterializationError(
            "accepted B row has inconsistent weight-bearing agreement count"
        )
    focal_event = {
        **dict(event),
        "frame": local_frame,
        "source_frame": source_row_start + event_frame,
        "agreement_count": agreement_count,
        "recorded_agreement_count": recorded_agreement_count,
        "independent_agreements": agreement_list,
        "pseudo_weight": sample_weight,
        "audio_weight_eligible": audio_weight_eligible,
        "needs_agreement_pass": False,
        "training_eligible": True,
        "unknown_for_loss": False,
        "filter_decision": "accepted_independent_agreement",
    }
    return {
        "source": source_row["source"],
        "video": f"{source_row['source_video']}:{event['event_id']}",
        "source_video": source_row["source_video"],
        "video_path": _portable_path(media_path),
        "media_present": True,
        "split": "train",
        "fps": source_row["fps"],
        "source_start_frame": absolute_start,
        "num_frames": EXPECTED_WINDOW_FRAMES,
        "event_counts": {
            "HIT": int(event["class"] == "HIT"),
            "BOUNCE": int(event["class"] == "BOUNCE"),
            "background": 0,
        },
        "inventory_event_count": 1,
        "events": [focal_event],
        "loss_validity_mask": list(source_row["loss_validity_mask"]),
        "unknown_frame_mask": unknown,
        "sample_weight": sample_weight,
        "agreement_count": agreement_count,
        "recorded_agreement_count": recorded_agreement_count,
        "audio_weight_eligible": audio_weight_eligible,
        "needs_agreement_pass": False,
        "training_eligible": True,
        "source_video_sha256": source_row["source_video_sha256"],
        "parent_identity": source_row["parent_identity"],
        "source_lineage_key": source_row["source_lineage_key"],
        "timebase_conversion": deepcopy(source_row["timebase_conversion"]),
        "focal_event_id": event["event_id"],
        "license_id": source_row["license_id"],
        "license_posture": source_row["license_posture"],
    }


def _placebo_frame(
    row: Mapping[str, Any], *, event: Mapping[str, Any], seed: int
) -> int:
    window_start = int(row["source_start_frame"])
    window_end = window_start + int(row["num_frames"])
    rally_start = int(event["rally_source_start_frame"])
    rally_end = int(event["rally_source_end_frame_exclusive"])
    low = max(window_start, rally_start)
    high = min(window_end, rally_end)
    original = int(event["source_frame"])
    unknown = row["unknown_frame_mask"]
    choices = [
        frame for frame in range(low, high)
        if frame != original and unknown[frame - window_start] is False
    ]
    if not choices:
        raise ABCMaterializationError(
            f"event {event['event_id']} has no alternate loss-valid within-rally placebo frame"
        )
    derived_seed = int(
        hashlib.sha256(f"{seed}:{event['event_id']}".encode()).hexdigest()[:16], 16
    )
    return random.Random(derived_seed).choice(choices)


def _materialize_c_row(
    b_row: Mapping[str, Any], *, frame_times: Sequence[float], seed: int
) -> dict[str, Any]:
    row = deepcopy(dict(b_row))
    event = row["events"][0]
    original_source_frame = int(event["source_frame"])
    shuffled_source_frame = _placebo_frame(row, event=event, seed=seed)
    local_frame = shuffled_source_frame - int(row["source_start_frame"])
    if not 0 <= local_frame < EXPECTED_WINDOW_FRAMES:
        raise ABCMaterializationError("placebo frame escaped its byte-identical pixel window")
    unknown = list(row["unknown_frame_mask"])
    original_local = original_source_frame - int(row["source_start_frame"])
    # The vacated label location remains loss-valid as background. The shuffled
    # location is selected from already-valid frames and receives B's focal
    # treatment, so C changes label time without changing exposure cardinality.
    unknown[original_local] = False
    unknown[local_frame] = False
    row["unknown_frame_mask"] = unknown
    event["placebo_original_source_frame"] = original_source_frame
    event["placebo_original_source_pts_s"] = event["source_pts_s"]
    event["frame"] = local_frame
    event["source_frame"] = shuffled_source_frame
    event["source_pts_s"] = frame_times[shuffled_source_frame]
    event["placebo_seed"] = seed
    event["filter_decision"] = "placebo_time_shuffled_within_rally"
    row["placebo"] = {
        "policy": "same_pixel_window_shift_focal_time_within_source_rally",
        "seed": seed,
        "original_source_frame": original_source_frame,
        "shuffled_source_frame": shuffled_source_frame,
    }
    return row


def _manifest_header(
    *,
    arm: str,
    teacher_manifest_path: Path,
    teacher_sha256: str,
    input_bindings: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    seed: int,
    max_delta_s: float,
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "artifact_type": f"event_head_pbvision_arm_{arm.lower()}_dataset_manifest",
        "verified": False,
        "training_ready": False,
        "teacher_derived": True,
        "ground_truth": False,
        "arm": arm,
        "seed": seed,
        "classes": {"0": "background", "1": "HIT", "2": "BOUNCE"},
        "image_size": 224,
        "license_posture": "pbvision_signed_full_usage",
        "config": {
            "window_frames": EXPECTED_WINDOW_FRAMES,
            "split": "train_only",
            "agreement_max_abs_delta_s": max_delta_s,
            "agreement_signal_families": list(SIGNAL_FAMILIES),
            "arm_b_required_agreement_family": "ball_velocity_kink",
            "audio_only_rejection_reason": "audio_only_no_physical_cue",
            "agreement_count_semantics": (
                "weight-bearing families after per-video audio null gating"
            ),
            "recorded_agreement_count_semantics": (
                "raw len(independent_agreements), including weight-inert audio"
            ),
            "pseudo_weight_by_agreement_count": {
                "0": 0.0,
                "1": 0.25,
                ">=2": 0.5,
            },
            "audio_time_shift_null": {
                "shift_count": AUDIO_NULL_SHIFT_COUNT,
                "unique_shift_count_required": AUDIO_NULL_SHIFT_COUNT,
                "minimum_absolute_offset_s": AUDIO_NULL_MIN_ABS_OFFSET_S,
                "minimum_observed_match_count": AUDIO_NULL_MIN_OBSERVED_MATCHES,
                "singleton_policy": "weight_ineligible",
                "offset_derivation": "sha256(seed,video_id,draw_index)",
                "comparison": (
                    "observed_match_count >= minimum and observed_match_rate "
                    "> max(null_match_rates)"
                ),
            },
            "unknown_frame_mask_semantics": "true means excluded from loss, never background",
            "no_scoring_or_protected_eval": True,
        },
        "permanent_compare_only_denylist": sorted(COMPARE_ONLY_HOLDOUTS),
        "provenance": {
            "teacher_manifest": {
                "path": _portable_path(teacher_manifest_path),
                "sha256": teacher_sha256,
            },
            "materializer": {
                "path": _portable_path(SCRIPT_PATH),
                "sha256": _sha256_file(SCRIPT_PATH),
            },
            "consumed_inputs": list(input_bindings),
        },
        "metadata": {
            "audio_time_shift_null": {
                str(binding["video_id"]): deepcopy(
                    binding["audio_time_shift_null"]
                )
                for binding in input_bindings
            },
        },
        "totals": {
            "rows": len(rows),
            "HIT": sum(row["events"][0]["class"] == "HIT" for row in rows),
            "BOUNCE": sum(row["events"][0]["class"] == "BOUNCE" for row in rows),
            "sample_weight": sum(float(row["sample_weight"]) for row in rows),
        },
        "rows": list(rows),
    }


def _non_placebo_projection(row: Mapping[str, Any]) -> dict[str, Any]:
    projected = deepcopy(dict(row))
    projected.pop("placebo", None)
    event = projected["events"][0]
    for key in (
        "frame",
        "source_frame",
        "source_pts_s",
        "filter_decision",
        "placebo_original_source_frame",
        "placebo_original_source_pts_s",
        "placebo_seed",
    ):
        event.pop(key, None)
    return projected


def _assert_b_c_parity(
    b_rows: Sequence[Mapping[str, Any]], c_rows: Sequence[Mapping[str, Any]]
) -> None:
    if len(b_rows) != len(c_rows):
        raise ABCMaterializationError("B/C row-count parity failed")
    for b_row, c_row in zip(b_rows, c_rows):
        b_event, c_event = b_row["events"][0], c_row["events"][0]
        if b_row["unknown_frame_mask"] != c_row["unknown_frame_mask"]:
            raise ABCMaterializationError(
                "B/C exact UNKNOWN-mask parity failed"
            )
        if _non_placebo_projection(b_row) != _non_placebo_projection(c_row):
            raise ABCMaterializationError(
                "B/C non-placebo field parity failed"
            )
        if b_event["source_frame"] == c_event["source_frame"]:
            raise ABCMaterializationError("C placebo did not move focal event time")
        if (
            c_event.get("placebo_original_source_frame") != b_event["source_frame"]
            or c_event.get("placebo_original_source_pts_s")
            != b_event["source_pts_s"]
            or c_event.get("filter_decision")
            != "placebo_time_shuffled_within_rally"
            or c_event.get("frame")
            != c_event["source_frame"] - c_row["source_start_frame"]
        ):
            raise ABCMaterializationError("C placebo transformation parity failed")
        placebo = c_row.get("placebo")
        if (
            not isinstance(placebo, Mapping)
            or placebo.get("original_source_frame") != b_event["source_frame"]
            or placebo.get("shuffled_source_frame") != c_event["source_frame"]
        ):
            raise ABCMaterializationError("C placebo provenance parity failed")


def materialize_arms(
    teacher_manifest_path: Path,
    *,
    media_paths: Mapping[str, Path],
    frame_times_paths: Mapping[str, Path],
    audio_paths: Mapping[str, Path],
    ball_paths: Mapping[str, Path],
    seed: int = DEFAULT_SEED,
    max_delta_s: float = DEFAULT_MAX_DELTA_S,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Build B/C manifests, decisions, and a complete SHA binding ledger."""

    if not math.isfinite(max_delta_s) or max_delta_s <= 0.0:
        raise ABCMaterializationError("max_delta_s must be finite and positive")
    teacher, teacher_raw = _load_json(teacher_manifest_path, label="teacher manifest")
    rows = _validate_teacher_manifest(teacher)
    for label, bindings in (
        ("media", media_paths),
        ("frame-times", frame_times_paths),
        ("audio-onsets", audio_paths),
        ("ball-velocity-kinks", ball_paths),
    ):
        _require_exact_bindings(rows, label=label, bindings=bindings)

    b_rows: list[dict[str, Any]] = []
    c_rows: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    input_bindings: list[dict[str, Any]] = []
    for source_row in rows:
        video_id = str(source_row["source_video"])
        indexed_events = _validate_and_index_events(
            source_row.get("events"), video_id=video_id
        )
        media_path = media_paths[video_id]
        if not media_path.is_file():
            raise ABCMaterializationError(f"media is missing for {video_id}: {media_path}")
        media_sha = _sha256_file(media_path)
        if media_sha != source_row["source_video_sha256"]:
            raise ABCMaterializationError(
                f"media SHA-256 mismatch for {video_id}: "
                f"{media_sha} != {source_row['source_video_sha256']}"
            )
        (
            frame_times,
            video_duration_s,
            pts_binding,
            frame_times_raw,
        ) = _validate_frame_times(
            frame_times_paths[video_id], row=source_row, media_sha256=media_sha
        )
        audio_cues, audio_binding = _load_cues(
            audio_paths[video_id],
            family="audio_onset",
            video_id=video_id,
            media_sha256=media_sha,
            frame_times_path=frame_times_paths[video_id],
            frame_times_sha256=pts_binding["sha256"],
            frame_times_raw=frame_times_raw,
        )
        ball_cues, ball_binding = _load_cues(
            ball_paths[video_id],
            family="ball_velocity_kink",
            video_id=video_id,
            media_sha256=media_sha,
            frame_times_path=frame_times_paths[video_id],
            frame_times_sha256=pts_binding["sha256"],
            frame_times_raw=frame_times_raw,
        )
        eligible_events = [
            (event_index, event) for event_index, event in indexed_events
            if event.get("needs_agreement_pass") is True
            and event.get("filter_decision") == "pending_independent_agreement"
        ]
        match_by_family = {
            "audio_onset": _match_family(
                eligible_events,
                audio_cues,
                family="audio_onset",
                max_delta_s=max_delta_s,
            ),
            "ball_velocity_kink": _match_family(
                eligible_events,
                ball_cues,
                family="ball_velocity_kink",
                max_delta_s=max_delta_s,
            ),
        }
        audio_time_shift_null = _audio_time_shift_null(
            eligible_events,
            audio_cues,
            match_by_family["audio_onset"],
            video_id=video_id,
            seed=seed,
            max_delta_s=max_delta_s,
            pts_origin_s=frame_times[0],
            video_duration_s=video_duration_s,
        )
        source_b_rows: list[dict[str, Any]] = []
        for event_index, event in indexed_events:
            event_id = event["event_id"]
            agreements = [
                _validated_emitted_agreement(
                    match_by_family[family][event_index],
                    event_index=event_index,
                    event=event,
                    max_delta_s=max_delta_s,
                )
                for family in SIGNAL_FAMILIES
                if event_index in match_by_family[family]
            ]
            recorded_agreement_count = len(agreements)
            families = {str(item["family"]) for item in agreements}
            pending_agreement = (
                event.get("needs_agreement_pass") is True
                and event.get("filter_decision") == "pending_independent_agreement"
            )
            has_audio = "audio_onset" in families
            has_physical_cue = "ball_velocity_kink" in families
            accepted = pending_agreement and has_physical_cue
            audio_weight_eligible = bool(
                accepted and has_audio and audio_time_shift_null["beats_null"]
            )
            agreement_count = (
                (1 if has_physical_cue else 0)
                + (1 if audio_weight_eligible else 0)
                if accepted
                else 0
            )
            weight = _weight_for_count(agreement_count) if accepted else 0.0
            if accepted:
                rejection_reason = None
            elif not pending_agreement:
                rejection_reason = str(event.get("filter_decision"))
            elif has_audio and not has_physical_cue:
                rejection_reason = "audio_only_no_physical_cue"
            else:
                rejection_reason = "zero_independent_agreements"
            decisions.append({
                "video_id": video_id,
                "event_id": event_id,
                "source_event_index": event_index,
                "class": event["class"],
                "source_frame": event["frame"],
                "source_pts_s": event["source_pts_s"],
                "agreement_count": agreement_count,
                "recorded_agreement_count": recorded_agreement_count,
                "independent_agreements": agreements,
                "audio_weight_eligible": audio_weight_eligible,
                "pseudo_weight": weight,
                "accepted_into_arm_b": accepted,
                "rejection_reason": rejection_reason,
            })
            if accepted:
                source_b_rows.append(_materialize_b_row(
                    source_row,
                    event,
                    media_path=media_path,
                    agreements=agreements,
                    agreement_count=agreement_count,
                    sample_weight=weight,
                    audio_weight_eligible=audio_weight_eligible,
                ))
        source_b_rows.sort(key=lambda row: str(row["focal_event_id"]))
        for b_row in source_b_rows:
            b_rows.append(b_row)
            c_rows.append(_materialize_c_row(
                b_row, frame_times=frame_times, seed=seed
            ))
        input_bindings.append({
            "video_id": video_id,
            "source_lineage_key": source_row["source_lineage_key"],
            "media": {"path": _portable_path(media_path), "sha256": media_sha},
            "frame_times": pts_binding,
            "audio_onsets": audio_binding,
            "ball_velocity_kinks": ball_binding,
            "audio_time_shift_null": audio_time_shift_null,
        })

    decisions.sort(key=lambda item: (item["video_id"], item["source_frame"], item["event_id"]))
    input_bindings.sort(key=lambda item: item["video_id"])
    if not b_rows:
        raise ABCMaterializationError("agreement pass accepted zero teacher events")
    _assert_b_c_parity(b_rows, c_rows)
    teacher_sha = _sha256_bytes(teacher_raw)
    b_manifest = _manifest_header(
        arm="B",
        teacher_manifest_path=teacher_manifest_path,
        teacher_sha256=teacher_sha,
        input_bindings=input_bindings,
        rows=b_rows,
        seed=seed,
        max_delta_s=max_delta_s,
    )
    b_sha = _sha256_bytes(_json_bytes(b_manifest))
    c_manifest = _manifest_header(
        arm="C",
        teacher_manifest_path=teacher_manifest_path,
        teacher_sha256=teacher_sha,
        input_bindings=input_bindings,
        rows=c_rows,
        seed=seed,
        max_delta_s=max_delta_s,
    )
    c_manifest["placebo"] = {
        "policy": "same_rows_pixels_classes_weights_shift_time_within_rally",
        "seed": seed,
        "source_arm_b_manifest_sha256": b_sha,
    }
    try:
        validate_current_manifest(b_manifest)
        validate_current_manifest(c_manifest)
    except DatasetFormatError as exc:
        raise ABCMaterializationError(f"materialized manifest schema rejected: {exc}") from exc
    return b_manifest, c_manifest, decisions, input_bindings


def write_materializations(
    output_dir: Path,
    *,
    needs: Mapping[str, Any],
    b_manifest: Mapping[str, Any] | None = None,
    c_manifest: Mapping[str, Any] | None = None,
    decisions: Iterable[Mapping[str, Any]] | None = None,
    input_bindings: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, str]:
    optional_parts = (b_manifest, c_manifest, decisions, input_bindings)
    if any(part is not None for part in optional_parts) and not all(
        part is not None for part in optional_parts
    ):
        raise ABCMaterializationError(
            "full publication requires B, C, decisions, and input bindings together"
        )
    artifacts: dict[str, bytes] = {
        "VM_ABC_NEEDS.json": _json_bytes(needs),
    }
    if b_manifest is not None:
        artifacts["arm_b_manifest.json"] = _json_bytes(b_manifest)
    if c_manifest is not None:
        artifacts["arm_c_manifest.json"] = _json_bytes(c_manifest)
    if decisions is not None:
        artifacts["agreement_decisions.jsonl"] = b"".join(
            json.dumps(item, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
            + b"\n"
            for item in decisions
        )
    if input_bindings is not None:
        artifacts["input_bindings.json"] = _json_bytes({
            "schema_version": 1,
            "artifact_type": "pbvision_abc_input_bindings",
            "verified": False,
            "bindings": list(input_bindings),
        })
    artifact_hashes = {
        name: _sha256_bytes(content)
        for name, content in sorted(artifacts.items())
    }
    completion_name = "materialization_complete.json"
    completion = {
        "schema_version": 1,
        "artifact_type": "pbvision_abc_materialization_complete",
        "complete": True,
        "mode": "full" if b_manifest is not None else "needs_only",
        "artifact_sha256": artifact_hashes,
    }
    artifacts[completion_name] = _json_bytes(completion)
    artifact_hashes[completion_name] = _sha256_bytes(artifacts[completion_name])

    managed_names = {
        "VM_ABC_NEEDS.json",
        "arm_b_manifest.json",
        "arm_c_manifest.json",
        "agreement_decisions.jsonl",
        "input_bindings.json",
        completion_name,
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(exist_ok=True)
    lock_path = output_dir / ".abc_materialization.lock"
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise ABCMaterializationError(
            f"ABC publication already in progress or abandoned: {lock_path}"
        ) from exc
    staging_dir: Path | None = None
    try:
        os.write(lock_fd, f"pid={os.getpid()}\n".encode())
        os.close(lock_fd)
        lock_fd = -1
        stale = sorted(
            name for name in managed_names if (output_dir / name).exists()
        )
        if stale:
            raise ABCMaterializationError(
                "output directory contains prior ABC materialization artifacts; "
                f"use a fresh output directory: {stale}"
            )
        staging_dir = Path(tempfile.mkdtemp(
            prefix=f".{output_dir.name}.staging-", dir=output_dir.parent
        ))
        for name, content in artifacts.items():
            (staging_dir / name).write_bytes(content)
        for name in sorted(set(artifacts) - {completion_name}):
            os.replace(staging_dir / name, output_dir / name)
        # Consumers may trust the set only after this completion record appears.
        os.replace(staging_dir / completion_name, output_dir / completion_name)
        mismatches = {
            name: {
                "expected": expected_sha,
                "actual": _sha256_file(output_dir / name),
            }
            for name, expected_sha in artifact_hashes.items()
            if _sha256_file(output_dir / name) != expected_sha
        }
        if mismatches:
            (output_dir / completion_name).unlink(missing_ok=True)
            raise ABCMaterializationError(
                f"published artifact self-verification failed: {mismatches}"
            )
    finally:
        if lock_fd >= 0:
            os.close(lock_fd)
        if staging_dir is not None:
            shutil.rmtree(staging_dir, ignore_errors=True)
        lock_path.unlink(missing_ok=True)
    return dict(sorted(artifact_hashes.items()))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-manifest", type=Path, default=DEFAULT_TEACHER_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-delta-s", type=float, default=DEFAULT_MAX_DELTA_S)
    parser.add_argument("--needs-only", action="store_true")
    parser.add_argument("--media", action="append", default=[], metavar="VIDEO_ID=PATH")
    parser.add_argument("--frame-times", action="append", default=[], metavar="VIDEO_ID=PATH")
    parser.add_argument("--audio-onsets", action="append", default=[], metavar="VIDEO_ID=PATH")
    parser.add_argument(
        "--ball-velocity-kinks", action="append", default=[], metavar="VIDEO_ID=PATH"
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        media = parse_path_bindings(args.media, flag="--media")
        frame_times = parse_path_bindings(args.frame_times, flag="--frame-times")
        audio = parse_path_bindings(args.audio_onsets, flag="--audio-onsets")
        ball = parse_path_bindings(
            args.ball_velocity_kinks, flag="--ball-velocity-kinks"
        )
        teacher, _ = _load_json(args.teacher_manifest, label="teacher manifest")
        needs = build_vm_needs(
            teacher,
            teacher_manifest_path=args.teacher_manifest,
            media_paths=media,
            frame_times_paths=frame_times,
            audio_paths=audio,
            ball_paths=ball,
        )
        if args.needs_only:
            write_materializations(args.output_dir, needs=needs)
            print(json.dumps({
                "output_dir": _portable_path(args.output_dir),
                "needs_only": True,
                "required_train_clips": needs["required_train_clips"],
                "verified": False,
            }, sort_keys=True))
            return 0
        b_manifest, c_manifest, decisions, bindings = materialize_arms(
            args.teacher_manifest,
            media_paths=media,
            frame_times_paths=frame_times,
            audio_paths=audio,
            ball_paths=ball,
            seed=args.seed,
            max_delta_s=args.max_delta_s,
        )
        hashes = write_materializations(
            args.output_dir,
            needs=needs,
            b_manifest=b_manifest,
            c_manifest=c_manifest,
            decisions=decisions,
            input_bindings=bindings,
        )
    except (ABCMaterializationError, OSError) as exc:
        parser.exit(2, f"A/B/C materialization rejected: {exc}\n")
    print(json.dumps({
        "output_dir": _portable_path(args.output_dir),
        "arm_b_rows": len(b_manifest["rows"]),
        "arm_c_rows": len(c_manifest["rows"]),
        "verified": False,
        "scoring_performed": False,
        "sha256": hashes,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

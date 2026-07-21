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
import random
import sys
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


DEFAULT_TEACHER_MANIFEST = (
    ROOT / "runs/lanes/pbv_corpus_rebuild_20260720/manifest.json"
)
DEFAULT_OUTPUT_DIR = ROOT / "runs/lanes/w1b_abc_loader_20260721/abc_materialized"
DEFAULT_SEED = 20260720
DEFAULT_MAX_DELTA_S = 0.035
EXPECTED_WINDOW_FRAMES = 64
SIGNAL_FAMILIES = ("audio_onset", "ball_velocity_kink")
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


def _json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _canonical_sha256(payload: Any) -> str:
    return _sha256_bytes(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    )


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
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ABCMaterializationError(f"{field} must be numeric") from exc
    if not math.isfinite(parsed) or parsed < 0.0:
        raise ABCMaterializationError(f"{field} must be finite and nonnegative")
    return parsed


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
        manifest.get("schema_version") != 2
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
) -> tuple[list[float], dict[str, Any]]:
    payload, raw = _load_json(path, label="frame-times artifact")
    if not isinstance(payload, dict) or not isinstance(payload.get("frames"), list):
        raise ABCMaterializationError(f"invalid frame-times payload: {path}")
    frames = payload["frames"]
    if len(frames) != int(row["num_frames"]):
        raise ABCMaterializationError(
            f"frame-times count mismatch for {row['source_video']}: "
            f"{len(frames)} != {row['num_frames']}"
        )
    times: list[float] = []
    for expected, item in enumerate(frames):
        if not isinstance(item, dict) or item.get("frame") != expected:
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
    declared_media_sha = payload.get(
        "source_video_sha256", payload.get("media_sha256")
    )
    if declared_media_sha is not None and declared_media_sha != media_sha256:
        raise ABCMaterializationError(
            f"frame-times declares the wrong media SHA for {row['source_video']}"
        )
    return times, {
        "path": _portable_path(path),
        "sha256": frame_times_sha256,
        "media_sha256": media_sha256,
        "binding_sha256": expected_binding_sha,
    }


def _cue_time(item: Mapping[str, Any], *, field: str) -> float:
    for key in ("corrected_time_s", "time_s", "pts_s", "source_pts_s"):
        if key in item:
            return _finite_nonnegative(item[key], field=f"{field}.{key}")
    raise ABCMaterializationError(f"{field} has no corrected_time_s/time_s/pts_s")


def _cue_items(payload: Any, *, family: str, path: Path) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        raise ABCMaterializationError(f"{family} artifact must be an object: {path}")
    keys = ("onsets",) if family == "audio_onset" else ("kinks", "candidates")
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
        time_s = _cue_time(item, field=f"{family}[{index}]")
        stable_id = item.get("cue_id", item.get("event_id", item.get("id")))
        if stable_id is None:
            stable_id = _canonical_sha256({"index": index, "cue": item})
        cues.append({
            "stable_id": str(stable_id),
            "time_s": time_s,
            "source_index": index,
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
    frame_times_sha256: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload, raw = _load_json(path, label=f"{family} artifact")
    if not isinstance(payload, dict):
        raise ABCMaterializationError(f"{family} artifact must be an object")
    declared_video = payload.get("video_id", payload.get("clip"))
    if declared_video is not None and str(declared_video) != video_id:
        raise ABCMaterializationError(
            f"{family} artifact clip mismatch: {declared_video!r} != {video_id!r}"
        )
    declared_media_sha = payload.get(
        "source_video_sha256", payload.get("media_sha256")
    )
    declared_pts_sha = payload.get("frame_times_sha256")
    if declared_media_sha is None or declared_pts_sha is None:
        raise ABCMaterializationError(
            f"{family} artifact must declare source_video_sha256 and "
            "frame_times_sha256; unbound agreement inputs are forbidden"
        )
    if declared_media_sha != media_sha256:
        raise ABCMaterializationError(f"{family} artifact media SHA mismatch")
    if declared_pts_sha != frame_times_sha256:
        raise ABCMaterializationError(f"{family} artifact frame-times SHA mismatch")
    cues = _cue_items(payload, family=family, path=path)
    artifact_sha256 = _sha256_bytes(raw)
    return cues, {
        "path": _portable_path(path),
        "sha256": artifact_sha256,
        "cue_count": len(cues),
        "declared_media_sha256": declared_media_sha,
        "declared_frame_times_sha256": declared_pts_sha,
        "dependency_binding_sha256": _canonical_sha256({
            "artifact_sha256": artifact_sha256,
            "source_video_sha256": media_sha256,
            "frame_times_sha256": frame_times_sha256,
        }),
    }


def _match_family(
    events: Sequence[Mapping[str, Any]],
    cues: Sequence[Mapping[str, Any]],
    *,
    family: str,
    max_delta_s: float,
) -> dict[str, dict[str, Any]]:
    candidates: list[tuple[float, str, float, str, int]] = []
    for event in events:
        event_id = str(event["event_id"])
        event_time = _finite_nonnegative(
            event.get("source_pts_s"), field=f"event {event_id}.source_pts_s"
        )
        for cue_index, cue in enumerate(cues):
            delta = abs(float(cue["time_s"]) - event_time)
            if delta <= max_delta_s:
                candidates.append((
                    delta,
                    event_id,
                    float(cue["time_s"]),
                    str(cue["stable_id"]),
                    cue_index,
                ))
    candidates.sort()
    used_events: set[str] = set()
    used_cues: set[int] = set()
    matches: dict[str, dict[str, Any]] = {}
    for delta, event_id, cue_time, stable_id, cue_index in candidates:
        if event_id in used_events or cue_index in used_cues:
            continue
        used_events.add(event_id)
        used_cues.add(cue_index)
        matches[event_id] = {
            "family": family,
            "cue_stable_id": stable_id,
            "cue_time_s": cue_time,
            "absolute_delta_s": delta,
        }
    return matches


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
    agreement_count = len(agreement_list)
    sample_weight = _weight_for_count(agreement_count)
    if sample_weight == 0.0:
        raise ABCMaterializationError("zero-agreement events cannot become B rows")
    focal_event = {
        **dict(event),
        "frame": local_frame,
        "source_frame": source_row_start + event_frame,
        "agreement_count": agreement_count,
        "independent_agreements": agreement_list,
        "pseudo_weight": sample_weight,
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
            "pseudo_weight_by_agreement_count": {
                "0": 0.0,
                "1": 0.25,
                ">=2": 0.5,
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
        "totals": {
            "rows": len(rows),
            "HIT": sum(row["events"][0]["class"] == "HIT" for row in rows),
            "BOUNCE": sum(row["events"][0]["class"] == "BOUNCE" for row in rows),
            "sample_weight": sum(float(row["sample_weight"]) for row in rows),
        },
        "rows": list(rows),
    }


def _assert_b_c_parity(b_rows: Sequence[Mapping[str, Any]], c_rows: Sequence[Mapping[str, Any]]) -> None:
    if len(b_rows) != len(c_rows):
        raise ABCMaterializationError("B/C row-count parity failed")
    for b_row, c_row in zip(b_rows, c_rows):
        for key in (
            "source_video", "video_path", "source_start_frame", "num_frames",
            "sample_weight", "agreement_count", "focal_event_id",
            "source_video_sha256", "source_lineage_key", "loss_validity_mask",
        ):
            if b_row[key] != c_row[key]:
                raise ABCMaterializationError(f"B/C parity failed for {key}")
        b_event, c_event = b_row["events"][0], c_row["events"][0]
        if b_event["class"] != c_event["class"]:
            raise ABCMaterializationError("B/C class parity failed")
        if b_event["independent_agreements"] != c_event["independent_agreements"]:
            raise ABCMaterializationError("B/C agreement-metadata parity failed")
        b_loss_valid = sum(not bool(masked) for masked in b_row["unknown_frame_mask"])
        c_loss_valid = sum(not bool(masked) for masked in c_row["unknown_frame_mask"])
        if b_loss_valid != c_loss_valid:
            raise ABCMaterializationError(
                "B/C loss-valid frame-count parity failed: "
                f"{b_loss_valid} != {c_loss_valid}"
            )
        if b_event["source_frame"] == c_event["source_frame"]:
            raise ABCMaterializationError("C placebo did not move focal event time")


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
        media_path = media_paths[video_id]
        if not media_path.is_file():
            raise ABCMaterializationError(f"media is missing for {video_id}: {media_path}")
        media_sha = _sha256_file(media_path)
        if media_sha != source_row["source_video_sha256"]:
            raise ABCMaterializationError(
                f"media SHA-256 mismatch for {video_id}: "
                f"{media_sha} != {source_row['source_video_sha256']}"
            )
        frame_times, pts_binding = _validate_frame_times(
            frame_times_paths[video_id], row=source_row, media_sha256=media_sha
        )
        audio_cues, audio_binding = _load_cues(
            audio_paths[video_id],
            family="audio_onset",
            video_id=video_id,
            media_sha256=media_sha,
            frame_times_sha256=pts_binding["sha256"],
        )
        ball_cues, ball_binding = _load_cues(
            ball_paths[video_id],
            family="ball_velocity_kink",
            video_id=video_id,
            media_sha256=media_sha,
            frame_times_sha256=pts_binding["sha256"],
        )
        eligible_events = [
            event for event in source_row["events"]
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
        source_b_rows: list[dict[str, Any]] = []
        for event in source_row["events"]:
            event_id = str(event["event_id"])
            agreements = [
                match_by_family[family][event_id]
                for family in SIGNAL_FAMILIES
                if event_id in match_by_family[family]
            ]
            count = len(agreements)
            weight = _weight_for_count(count)
            accepted = bool(event.get("needs_agreement_pass")) and count > 0
            decisions.append({
                "video_id": video_id,
                "event_id": event_id,
                "class": event["class"],
                "source_frame": event["frame"],
                "source_pts_s": event["source_pts_s"],
                "agreement_count": count,
                "independent_agreements": agreements,
                "pseudo_weight": weight if accepted else 0.0,
                "accepted_into_arm_b": accepted,
                "rejection_reason": None if accepted else (
                    "zero_independent_agreements"
                    if event.get("needs_agreement_pass")
                    else str(event.get("filter_decision"))
                ),
            })
            if accepted:
                source_b_rows.append(_materialize_b_row(
                    source_row, event, media_path=media_path, agreements=agreements
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
    output_dir.mkdir(parents=True, exist_ok=True)
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
    for name, content in artifacts.items():
        (output_dir / name).write_bytes(content)
    return {name: _sha256_bytes(content) for name, content in sorted(artifacts.items())}


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
        write_materializations(args.output_dir, needs=needs)
        if args.needs_only:
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

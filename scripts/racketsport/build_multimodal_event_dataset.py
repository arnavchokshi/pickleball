#!/usr/bin/env python3
"""Build the multimodal event window dataset (WS3.2 training substrate).

Fuse-aligns the three production CPU cue streams — audio_onsets_v2,
wrist_velocity_peaks, ball_inflections — against the two authorized event
label sets (owner 102 rows, corrected 1,189-row teacher corpus) into
deterministic windowed JSONL records with per-modality availability masks.

Window length: 64 frames, matching the E-v2 design's 64-frame context
(owner_102_manifest.json `config.window_frames == 64`; both manifests carry
`num_frames == 64` on every row and place the label event at frame bin 32).

Two subcommands:
  gen-cues  Discover existing cue artifacts; where a labeled video has local
            source media but no artifact, run the existing CPU builders
            (build_audio_onsets_v2.py / build_ball_inflections.py) into the
            lane cue dir. Writes cue_index.json. Never writes into data/.
  build     Deterministically emit records/, coverage.json, COVERAGE.md and
            MANIFEST.sha256.json from the manifests + cue_index. Byte-identical
            re-runs given the same inputs (sorted keys, fixed precision, no
            timestamps).

Hard refusals (fail closed, measurement-only):
  - any teacher event whose independent agreements are audio-only;
  - any teacher record landing in val, or a teacher family overlapping an
    owner VAL family (quarantined and documented instead);
  - any record whose row key matches a protected-seed label id, any train
    record whose window overlaps a protected-seed window, and any record on a
    protected-seed video sha that is not an owner-manifest row.

VERIFIED=0 remains binding: outputs are measurement artifacts, not gate proof.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shlex
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.multimodal_event_dataset import (  # noqa: E402
    ARTIFACT_TYPE,
    SCHEMA_VERSION,
    WINDOW_CENTER_BIN,
    WINDOW_FRAMES,
    canonicalize,
    cues_in_window,
    modality_block,
    round_float,
    write_records_jsonl,
)

CUE_INDEX_ARTIFACT_TYPE = "racketsport_multimodal_cue_index"
COVERAGE_ARTIFACT_TYPE = "racketsport_multimodal_event_dataset_coverage"

# Existing per-source-video audio_onsets_v2 artifacts (content-identical across
# the four 20260722 audio lanes; repair2 carries the latest calibration report).
DEFAULT_AUDIO_ARTIFACT_TEMPLATES = (
    "runs/lanes/ball_audio_repair2_20260722/raw_audio_onsets/{family}.audio_onsets_v2.json",
)
DEFAULT_MEDIA_TEMPLATES = (
    "data/online_harvest_20260706/raw/{family}.mp4",
    "data/pbv_replay_20260720/{family}/max.mp4",
)
DEFAULT_BALL_TRACK_TEMPLATE = "data/online_harvest_20260706/prelabels/{clip_id}/ball_track.json"
DEFAULT_CLIP_AUDIO_TEMPLATE = "data/event_bootstrap_20260713/audio_onsets_v0/{clip_id}.json"

# Clip-to-source time mapping acceptance thresholds (measured, fail closed).
CLIP_MAPPING_TOLERANCE_S = 0.05
CLIP_MAPPING_MIN_MATCHES = 5
CLIP_MAPPING_MAX_ABS_MEDIAN_S = 0.04

AUDIO_ONLY_ERROR = "AUDIO_ONLY_TEACHER_EVENTS_PRESENT"
TEACHER_VAL_ERROR = "TEACHER_ROW_IN_VAL_SPLIT"
PROTECTED_TRAIN_OVERLAP_ERROR = "PROTECTED_OVERLAP_IN_TRAIN_WINDOW"
PROTECTED_IDENTITY_ERROR = "PROTECTED_SEED_IDENTITY_IN_RECORDS"
OWNER_SPLIT_ERROR = "OWNER_SPLIT_DRIFT"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_canonical_json(path: Path, payload: Any) -> str:
    text = json.dumps(canonicalize(payload), sort_keys=True, indent=1, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class Roots:
    """Resolve repo-relative paths against an ordered list of repo roots."""

    def __init__(self, roots: Sequence[Path]):
        self.roots = [Path(root).resolve() for root in roots]

    def resolve(self, rel: str) -> Path | None:
        for root in self.roots:
            candidate = root / rel
            if candidate.exists():
                return candidate
        return None

    def require(self, rel: str, *, role: str) -> Path:
        resolved = self.resolve(rel)
        if resolved is None:
            raise SystemExit(f"INPUT_MISSING: {role}: {rel} not found under any repo root")
        return resolved


class InputLedger:
    """Records the sha256 of every input file consumed by the build."""

    def __init__(self, roots: Roots):
        self.roots = roots
        self.entries: dict[str, dict[str, str]] = {}

    def consume(self, rel: str, *, role: str) -> Path:
        path = self.roots.require(rel, role=role)
        if rel not in self.entries:
            self.entries[rel] = {"path": rel, "role": role, "sha256": _sha256_file(path)}
        return path

    def as_list(self) -> list[dict[str, str]]:
        return [self.entries[key] for key in sorted(self.entries)]


# ---------------------------------------------------------------------------
# Label-set loading and hard assertions
# ---------------------------------------------------------------------------

def teacher_audio_only_event_count(teacher_manifest: Mapping[str, Any]) -> int:
    count = 0
    for row in teacher_manifest["rows"]:
        for event in row["events"]:
            agreements = event.get("independent_agreements") or []
            families = {item.get("family") for item in agreements}
            non_audio = families - {"audio_onset"}
            if not non_audio:
                count += 1
    return count


def owner_split_counts(owner_manifest: Mapping[str, Any]) -> dict[str, int]:
    counts = {"train": 0, "val": 0}
    for row in owner_manifest["rows"]:
        counts[row["split"]] = counts.get(row["split"], 0) + 1
    return counts


def owner_val_families(owner_manifest: Mapping[str, Any]) -> set[str]:
    return {row["source_video"] for row in owner_manifest["rows"] if row["split"] == "val"}


def quarantined_teacher_families(
    teacher_manifest: Mapping[str, Any], owner_manifest: Mapping[str, Any]
) -> set[str]:
    """Teacher families overlapping an owner VAL family (by id or media sha)."""
    val_rows = [row for row in owner_manifest["rows"] if row["split"] == "val"]
    val_ids = {row["source_video"] for row in val_rows}
    val_shas = {row.get("video_sha256") for row in val_rows if row.get("video_sha256")}
    quarantined: set[str] = set()
    for row in teacher_manifest["rows"]:
        family = row["source_video"]
        if family in val_ids or row.get("source_video_sha256") in val_shas:
            quarantined.add(family)
    return quarantined


# ---------------------------------------------------------------------------
# Protected seed identities (refuse-by-identity; answer file is never read)
# ---------------------------------------------------------------------------

def load_protected_identities(selector: Mapping[str, Any], inventory: Mapping[str, Any]) -> dict[str, Any]:
    clip_sha = {
        clip["clip_id"]: clip.get("video_sha256")
        for clip in inventory.get("clips", [])
        if isinstance(clip, Mapping) and clip.get("clip_id")
    }
    label_ids: set[str] = set()
    windows_by_clip: dict[str, list[dict[str, float]]] = {}
    shas: set[str] = set()
    for label in selector.get("labels", []):
        label_id = str(label["label_id"])
        label_ids.add(label_id)
        seed_clip = label_id.split("__", 1)[0]
        window = label.get("window") or {}
        anchor = label.get("anchor") or {}
        entry = {
            "anchor_pts_s": float(anchor.get("pts_s")),
            "end_pts_s": float(window.get("end_pts_s")),
            "label_id": label_id,
            "start_pts_s": float(window.get("start_pts_s")),
        }
        windows_by_clip.setdefault(seed_clip, []).append(entry)
        sha = clip_sha.get(seed_clip)
        if sha:
            shas.add(sha)
    for windows in windows_by_clip.values():
        windows.sort(key=lambda item: item["start_pts_s"])
    return {"label_ids": label_ids, "video_sha256s": shas, "windows_by_clip": windows_by_clip}


def protected_overlaps(
    protected: Mapping[str, Any],
    *,
    clip_id: str | None,
    clip_window_start_s: float | None,
    clip_window_end_s: float | None,
) -> list[str]:
    if clip_id is None or clip_window_start_s is None or clip_window_end_s is None:
        return []
    hits: list[str] = []
    for window in protected["windows_by_clip"].get(clip_id, []):
        if window["start_pts_s"] < clip_window_end_s and clip_window_start_s < window["end_pts_s"]:
            hits.append(window["label_id"])
    return hits


# ---------------------------------------------------------------------------
# Clip-to-source time mapping (measured against independent audio onsets)
# ---------------------------------------------------------------------------

def measure_clip_offset(
    clip_onset_times_s: Sequence[float],
    source_onset_times_s: Sequence[float],
    *,
    clip_start_s: float,
    tolerance_s: float = CLIP_MAPPING_TOLERANCE_S,
) -> dict[str, Any]:
    source_sorted = sorted(source_onset_times_s)
    residuals: list[float] = []
    for time_s in clip_onset_times_s:
        shifted = time_s + clip_start_s
        index = _bisect(source_sorted, shifted)
        best: float | None = None
        for candidate_index in (index - 1, index):
            if 0 <= candidate_index < len(source_sorted):
                delta = source_sorted[candidate_index] - shifted
                if best is None or abs(delta) < abs(best):
                    best = delta
        if best is not None and abs(best) <= tolerance_s:
            residuals.append(best)
    median = statistics.median(residuals) if residuals else None
    verified = (
        len(residuals) >= CLIP_MAPPING_MIN_MATCHES
        and median is not None
        and abs(median) <= CLIP_MAPPING_MAX_ABS_MEDIAN_S
    )
    return {
        "clip_onset_count": len(clip_onset_times_s),
        "clip_start_s_source": round_float(clip_start_s),
        "matched_onset_count": len(residuals),
        "median_residual_s": round_float(median) if median is not None else None,
        "verified": bool(verified),
    }


def _bisect(values: Sequence[float], target: float) -> int:
    low, high = 0, len(values)
    while low < high:
        mid = (low + high) // 2
        if values[mid] < target:
            low = mid + 1
        else:
            high = mid
    return low


# ---------------------------------------------------------------------------
# gen-cues subcommand
# ---------------------------------------------------------------------------

def _probe_media_pts_audio_only(path: Path) -> dict[str, Any]:
    """Minimal replica of the audited lane probe (audio stream only)."""
    commands = [
        ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
         "stream=index,codec_name,codec_type,sample_rate,channels,time_base,start_pts,start_time,duration_ts,duration,nb_frames",
         "-of", "json", str(path)],
        ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_packets", "-show_entries",
         "packet=pts,pts_time,dts_time,duration_time,flags,side_data_list",
         "-read_intervals", "%+#4", "-of", "json", str(path)],
        ["ffprobe", "-v", "error", "-show_entries", "format=start_time,duration", "-of", "json", str(path)],
    ]
    payloads = []
    for command in commands:
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
        payloads.append(json.loads(completed.stdout or "{}"))
    stream_payload, packet_payload, _format_payload = payloads
    streams = stream_payload.get("streams") or []
    if not streams:
        return {"audio_effective_origin_pts_s": None, "commands": [shlex.join(c) for c in commands],
                "status": "NO_AUDIO_STREAM"}
    stream = streams[0]
    sample_rate = float(stream.get("sample_rate") or 0.0)
    origin = None
    for packet in packet_payload.get("packets") or []:
        pts_time = packet.get("pts_time")
        if pts_time is None:
            continue
        packet_pts = float(pts_time)
        skip_samples = 0.0
        for side_data in packet.get("side_data_list") or []:
            if side_data.get("side_data_type") == "Skip Samples":
                skip_samples = float(side_data.get("skip_samples") or 0.0)
                break
        if skip_samples and sample_rate:
            effective = packet_pts + skip_samples / sample_rate
            origin = 0.0 if abs(effective) <= 0.5 / sample_rate else effective
        else:
            origin = packet_pts
        break
    return {
        "audio_effective_origin_pts_s": origin,
        "commands": [shlex.join(command) for command in commands],
        "status": "PTS_PROBED",
    }


def generate_cues(args: argparse.Namespace) -> int:
    roots = Roots([Path(root) for root in args.repo_root])
    owner_manifest = _load_json(roots.require(args.owner_manifest, role="owner manifest"))
    teacher_manifest = _load_json(roots.require(args.teacher_manifest, role="teacher manifest"))
    cue_dir = Path(args.cue_dir)
    cue_dir.mkdir(parents=True, exist_ok=True)
    out_root = Path(args.repo_root[0]).resolve()

    families = sorted(
        {row["source_video"] for row in owner_manifest["rows"]}
        | {row["source_video"] for row in teacher_manifest["rows"]}
    )
    audio_index: dict[str, Any] = {}
    media_index: dict[str, Any] = {}
    for family in families:
        media_rel = None
        media_path = None
        for template in DEFAULT_MEDIA_TEMPLATES:
            rel = template.format(family=family)
            resolved = roots.resolve(rel)
            if resolved is not None:
                media_rel, media_path = rel, resolved
                break
        media_sha = _sha256_file(media_path) if media_path is not None else None
        media_index[family] = {
            "path": media_rel,
            "present": media_path is not None,
            "sha256": media_sha,
        }

        artifact_rel = None
        for template in DEFAULT_AUDIO_ARTIFACT_TEMPLATES:
            rel = template.format(family=family)
            if roots.resolve(rel) is not None:
                artifact_rel = rel
                break
        if artifact_rel is None and media_path is not None:
            built_rel = _relative_to_root(cue_dir / f"{family}.audio_onsets_v2.json", out_root)
            built_path = cue_dir / f"{family}.audio_onsets_v2.json"
            if not built_path.exists():
                probe = _probe_media_pts_audio_only(media_path)
                pts_path = cue_dir / f"{family}.pts_identity.json"
                _write_canonical_json(pts_path, {
                    "artifact_type": "audio_alignment_pts_identity",
                    "audio_effective_origin_pts_s": probe.get("audio_effective_origin_pts_s"),
                    "clip_id": family,
                    "commands": probe.get("commands"),
                    "media_sha256": media_sha,
                    "probe_scope": "audio_stream_only_no_video_frame_decode",
                    "schema_version": 1,
                    "source_video_sha256": media_sha,
                    "video_effective_origin_pts_s": None,
                })
                command = [
                    sys.executable,
                    str(ROOT / "scripts/racketsport/build_audio_onsets_v2.py"),
                    "--input", str(media_path),
                    "--frame-times", str(pts_path),
                    "--out", str(built_path),
                    "--clip", family,
                ]
                completed = subprocess.run(command, check=False, capture_output=True, text=True)
                if completed.returncode != 0:
                    raise SystemExit(
                        f"CUE_BUILD_FAILED: audio_onsets_v2 for {family}: {completed.stderr.strip()[-400:]}"
                    )
            artifact_rel = built_rel
        if artifact_rel is not None:
            artifact_path = roots.resolve(artifact_rel)
            payload = _load_json(artifact_path)
            audio_index[family] = {
                "detector_version": payload.get("detector_version"),
                "media_sha256": payload.get("media_sha256"),
                "path": artifact_rel,
                "sha256": _sha256_file(artifact_path),
                "status": payload.get("status"),
            }

    clip_ids = sorted({row["clip_id"] for row in owner_manifest["rows"] if row.get("clip_id")})
    ball_index: dict[str, Any] = {}
    clip_audio_index: dict[str, Any] = {}
    for clip_id in clip_ids:
        track_rel = DEFAULT_BALL_TRACK_TEMPLATE.format(clip_id=clip_id)
        track_path = roots.resolve(track_rel)
        if track_path is not None:
            built_path = cue_dir / "ball_inflections" / f"{clip_id}.ball_inflections.json"
            if not built_path.exists():
                built_path.parent.mkdir(parents=True, exist_ok=True)
                command = [
                    sys.executable,
                    str(ROOT / "scripts/racketsport/build_ball_inflections.py"),
                    "--ball-track", str(track_path),
                    "--out", str(built_path),
                ]
                completed = subprocess.run(command, check=False, capture_output=True, text=True)
                if completed.returncode != 0:
                    raise SystemExit(
                        f"CUE_BUILD_FAILED: ball_inflections for {clip_id}: {completed.stderr.strip()[-400:]}"
                    )
            ball_index[clip_id] = {
                "ball_track_path": track_rel,
                "ball_track_sha256": _sha256_file(track_path),
                "path": _relative_to_root(built_path, out_root),
                "sha256": _sha256_file(built_path),
            }
        clip_audio_rel = DEFAULT_CLIP_AUDIO_TEMPLATE.format(clip_id=clip_id)
        clip_audio_path = roots.resolve(clip_audio_rel)
        if clip_audio_path is not None:
            clip_audio_index[clip_id] = {
                "path": clip_audio_rel,
                "sha256": _sha256_file(clip_audio_path),
            }

    cue_index = {
        "artifact_type": CUE_INDEX_ARTIFACT_TYPE,
        "audio_onsets_v2": audio_index,
        "ball_inflections": ball_index,
        "clip_audio_onsets_v0": clip_audio_index,
        "media": media_index,
        "schema_version": 1,
        "wrist_velocity_peaks": {},
        "wrist_velocity_peaks_note": (
            "no skeleton3d.json artifact exists locally for any labeled source video; "
            "the wrist_velocity_peaks builder requires skeleton3d upstream, so the wrist "
            "modality is masked no_artifact for every row"
        ),
    }
    _write_canonical_json(Path(args.cue_index), cue_index)
    print(f"wrote {args.cue_index}: audio families={len(audio_index)}/{len(families)}, "
          f"ball clips={len(ball_index)}/{len(clip_ids)}")
    return 0


def _relative_to_root(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root))


# ---------------------------------------------------------------------------
# build subcommand
# ---------------------------------------------------------------------------

def _window_fields(row: Mapping[str, Any]) -> dict[str, Any]:
    fps = float(row["fps"])
    start_frame = int(row["source_start_frame"])
    start_s = start_frame / fps
    return {
        "center_time_s": round_float(start_s + WINDOW_CENTER_BIN / fps),
        "duration_s": round_float(WINDOW_FRAMES / fps),
        "fps": round_float(fps),
        "frames": WINDOW_FRAMES,
        "source_start_frame": start_frame,
        "start_time_s": round_float(start_s),
    }


def _label_fields(row: Mapping[str, Any]) -> dict[str, Any]:
    fps = float(row["fps"])
    start_frame = int(row["source_start_frame"])
    events = row.get("events") or []
    if not events:
        review = row.get("review") or {}
        return {
            "class": "negative",
            "dt_s": None,
            "event_frame": None,
            "event_time_s": None,
            "negative_kind": review.get("decision") or "none",
        }
    event = events[0]
    frame = int(event["frame"])
    return {
        "class": str(event["class"]),
        "dt_s": round_float((frame - WINDOW_CENTER_BIN) / fps),
        "event_frame": frame,
        "event_time_s": round_float((start_frame + frame) / fps),
        "negative_kind": None,
    }


def _audio_modality(
    row: Mapping[str, Any],
    *,
    label_set: str,
    audio_entry: Mapping[str, Any] | None,
    onsets_by_family: Mapping[str, list[Mapping[str, Any]]],
    media_index: Mapping[str, Any],
) -> dict[str, Any]:
    family = row["source_video"]
    if audio_entry is None:
        return modality_block(artifact=None, events=[], absent_reason=None,
                              series_value_key="score", series_value_name="max_onset_score_per_frame_bin")
    artifact = {
        "detector_version": audio_entry.get("detector_version"),
        "media_sha256": audio_entry.get("media_sha256"),
        "path": audio_entry["path"],
        "sha256": audio_entry["sha256"],
        "timebase": "source_video_s",
    }
    if audio_entry.get("status") == "blocked":
        return modality_block(artifact=artifact, events=[], absent_reason="artifact_blocked",
                              series_value_key="score", series_value_name="max_onset_score_per_frame_bin")
    media = media_index.get(family) or {}
    if label_set == "teacher":
        bound = audio_entry.get("media_sha256") == row.get("source_video_sha256")
        artifact["media_binding"] = "row_source_video_sha256" if bound else "mismatch"
    else:
        local_sha = media.get("sha256")
        if local_sha is None:
            bound = True
            artifact["media_binding"] = "recorded_only"
        else:
            bound = audio_entry.get("media_sha256") == local_sha
            artifact["media_binding"] = "verified_local_media" if bound else "mismatch"
    if not bound:
        return modality_block(artifact=artifact, events=[], absent_reason="media_unbound",
                              series_value_key="score", series_value_name="max_onset_score_per_frame_bin")
    fps = float(row["fps"])
    window_start_s = int(row["source_start_frame"]) / fps
    onsets = onsets_by_family.get(family) or []
    events = []
    for onset in onsets:
        for offset, frame_offset in cues_in_window([float(onset["corrected_time_s"])],
                                                   window_start_s=window_start_s, fps=fps):
            events.append({
                "frame_offset": frame_offset,
                "offset_s": round_float(offset),
                "onset_strength": round_float(float(onset.get("onset_strength") or 0.0)),
                "score": round_float(float(onset.get("score") or 0.0)),
            })
    return modality_block(artifact=artifact, events=events, absent_reason=None,
                          series_value_key="score", series_value_name="max_onset_score_per_frame_bin")


def _ball_modality(
    row: Mapping[str, Any],
    *,
    ball_entry: Mapping[str, Any] | None,
    candidates_by_clip: Mapping[str, list[Mapping[str, Any]]],
    clip_mapping: Mapping[str, Any],
) -> dict[str, Any]:
    clip_id = row.get("clip_id")
    if ball_entry is None or clip_id is None:
        return modality_block(artifact=None, events=[], absent_reason=None,
                              series_value_key="confidence",
                              series_value_name="max_candidate_confidence_per_frame_bin")
    mapping = clip_mapping.get(clip_id) or {}
    artifact = {
        "ball_track_path": ball_entry.get("ball_track_path"),
        "ball_track_sha256": ball_entry.get("ball_track_sha256"),
        "clip_start_s_source": mapping.get("clip_start_s_source"),
        "clip_time_mapping": {
            "matched_onset_count": mapping.get("matched_onset_count"),
            "median_residual_s": mapping.get("median_residual_s"),
            "verified": mapping.get("verified"),
        },
        "path": ball_entry["path"],
        "sha256": ball_entry["sha256"],
        "timebase": "clip_s_plus_clip_start_offset",
    }
    if not mapping.get("verified"):
        return modality_block(artifact=artifact, events=[], absent_reason="clip_time_mapping_unverified",
                              series_value_key="confidence",
                              series_value_name="max_candidate_confidence_per_frame_bin")
    fps = float(row["fps"])
    window_start_s = int(row["source_start_frame"]) / fps
    clip_start_s = float(mapping["clip_start_s_source"])
    events = []
    for candidate in candidates_by_clip.get(clip_id) or []:
        source_time_s = float(candidate["time_s"]) + clip_start_s
        for offset, frame_offset in cues_in_window([source_time_s], window_start_s=window_start_s, fps=fps):
            events.append({
                "confidence": round_float(float(candidate.get("confidence") or 0.0)),
                "frame_offset": frame_offset,
                "offset_s": round_float(offset),
                "turn_angle_deg": round_float(float(candidate.get("turn_angle_deg") or 0.0)),
            })
    return modality_block(artifact=artifact, events=events, absent_reason=None,
                          series_value_key="confidence",
                          series_value_name="max_candidate_confidence_per_frame_bin")


def _wrist_modality() -> dict[str, Any]:
    return modality_block(artifact=None, events=[], absent_reason=None,
                          series_value_key="speed_mps",
                          series_value_name="max_wrist_speed_mps_per_frame_bin")


def _usable(block: Mapping[str, Any]) -> bool:
    return bool(block["available"]) or block.get("reason") == "no_signal_in_window"


def build_dataset(args: argparse.Namespace) -> int:
    roots = Roots([Path(root) for root in args.repo_root])
    ledger = InputLedger(roots)

    owner_manifest = _load_json(ledger.consume(args.owner_manifest, role="owner 102-row manifest"))
    teacher_manifest = _load_json(ledger.consume(args.teacher_manifest, role="teacher corrected 1189-row manifest"))
    cue_index = _load_json(ledger.consume(args.cue_index, role="cue artifact index"))
    selector = _load_json(ledger.consume(args.protected_selector, role="protected seed identity selector"))
    inventory = _load_json(ledger.consume(args.protected_inventory, role="event bootstrap clip inventory"))

    # Hard assertion: the corrected teacher corpus carries no audio-only events.
    audio_only = teacher_audio_only_event_count(teacher_manifest)
    if audio_only != 0:
        raise SystemExit(f"{AUDIO_ONLY_ERROR}: measured {audio_only} audio-only teacher events; expected 0")

    # Hard assertion: frozen owner split is intact.
    expected_train, expected_val = (int(part) for part in str(args.expect_owner_split).split(":", 1))
    split_counts = owner_split_counts(owner_manifest)
    if split_counts != {"train": expected_train, "val": expected_val}:
        raise SystemExit(
            f"{OWNER_SPLIT_ERROR}: measured {split_counts}; expected "
            f"{expected_train} train / {expected_val} val"
        )

    # Window-length assertion: every row carries the E-v2 64-frame context.
    for manifest, name in ((owner_manifest, "owner"), (teacher_manifest, "teacher")):
        bad = [row for row in manifest["rows"] if int(row.get("num_frames", -1)) != WINDOW_FRAMES]
        if bad:
            raise SystemExit(f"WINDOW_MISMATCH: {name} manifest has {len(bad)} rows with num_frames != {WINDOW_FRAMES}")

    quarantined_families = quarantined_teacher_families(teacher_manifest, owner_manifest)
    protected = load_protected_identities(selector, inventory)

    audio_index: Mapping[str, Any] = cue_index.get("audio_onsets_v2") or {}
    ball_index: Mapping[str, Any] = cue_index.get("ball_inflections") or {}
    clip_audio_index: Mapping[str, Any] = cue_index.get("clip_audio_onsets_v0") or {}
    media_index: Mapping[str, Any] = cue_index.get("media") or {}

    onsets_by_family: dict[str, list[Mapping[str, Any]]] = {}
    for family, entry in sorted(audio_index.items()):
        payload = _load_json(ledger.consume(entry["path"], role=f"audio_onsets_v2[{family}]"))
        onsets_by_family[family] = list(payload.get("onsets") or [])

    candidates_by_clip: dict[str, list[Mapping[str, Any]]] = {}
    for clip_id, entry in sorted(ball_index.items()):
        payload = _load_json(ledger.consume(entry["path"], role=f"ball_inflections[{clip_id}]"))
        candidates_by_clip[clip_id] = list(payload.get("candidates") or [])
        ledger.consume(entry["ball_track_path"], role=f"ball_track[{clip_id}]")

    # Clip-to-source time mapping, measured against independent audio onsets.
    clip_mapping: dict[str, Any] = {}
    owner_clips = sorted({row["clip_id"]: row["video_path"] for row in owner_manifest["rows"]}.items())
    clip_start_by_id: dict[str, float] = {}
    for clip_id, video_path in owner_clips:
        provenance_rel = str(video_path).replace(".mp4", ".provenance.json")
        provenance_path = roots.resolve(provenance_rel)
        if provenance_path is None:
            continue
        provenance = _load_json(ledger.consume(provenance_rel, role=f"clip_provenance[{clip_id}]"))
        clip_start_s = float((provenance.get("rally_segment") or {}).get("start_s") or 0.0)
        clip_start_by_id[clip_id] = clip_start_s
        family = clip_id.split("_rally_", 1)[0]
        clip_audio_entry = clip_audio_index.get(clip_id)
        family_onsets = onsets_by_family.get(family)
        if clip_audio_entry is None or family_onsets is None:
            clip_mapping[clip_id] = {
                "clip_onset_count": 0,
                "clip_start_s_source": round_float(clip_start_s),
                "matched_onset_count": 0,
                "median_residual_s": None,
                "verified": False,
            }
            continue
        clip_payload = _load_json(ledger.consume(clip_audio_entry["path"], role=f"clip_audio_onsets_v0[{clip_id}]"))
        clip_times = [float(onset["corrected_time_s"]) for onset in clip_payload.get("onsets") or []]
        source_times = [float(onset["corrected_time_s"]) for onset in family_onsets]
        clip_mapping[clip_id] = measure_clip_offset(clip_times, source_times, clip_start_s=clip_start_s)

    records: list[dict[str, Any]] = []
    unbuildable: list[dict[str, Any]] = []
    quarantined_rows: list[dict[str, Any]] = []
    protected_val_overlaps: list[dict[str, Any]] = []

    owner_manifest_ref = {"path": args.owner_manifest, "sha256": ledger.entries[args.owner_manifest]["sha256"]}
    teacher_manifest_ref = {"path": args.teacher_manifest, "sha256": ledger.entries[args.teacher_manifest]["sha256"]}

    for row in owner_manifest["rows"]:
        row_key = str(row["label_id"])
        record_id = f"owner:{row_key}"
        if row_key in protected["label_ids"] or record_id in protected["label_ids"]:
            raise SystemExit(f"{PROTECTED_IDENTITY_ERROR}: owner row key {row_key} matches a protected seed id")
        audio = _audio_modality(row, label_set="owner", audio_entry=audio_index.get(row["source_video"]),
                                onsets_by_family=onsets_by_family, media_index=media_index)
        ball = _ball_modality(row, ball_entry=ball_index.get(row["clip_id"]),
                              candidates_by_clip=candidates_by_clip, clip_mapping=clip_mapping)
        wrist = _wrist_modality()
        window = _window_fields(row)
        clip_id = row.get("clip_id")
        clip_start_s = clip_start_by_id.get(clip_id)
        overlap_ids: list[str] = []
        if clip_start_s is not None:
            clip_window_start = window["start_time_s"] - clip_start_s
            overlap_ids = protected_overlaps(
                protected, clip_id=clip_id,
                clip_window_start_s=clip_window_start,
                clip_window_end_s=clip_window_start + window["duration_s"],
            )
        if overlap_ids and row["split"] == "train":
            raise SystemExit(
                f"{PROTECTED_TRAIN_OVERLAP_ERROR}: owner train row {row_key} window overlaps {overlap_ids}"
            )
        if overlap_ids:
            protected_val_overlaps.append({"protected_label_ids": sorted(overlap_ids), "row_key": row_key})
        if not any(_usable(block) for block in (audio, ball, wrist)):
            unbuildable.append({
                "family": row["source_video"], "label_set": "owner",
                "reason": "missing_cue_artifacts", "row_key": row_key, "split": row["split"],
            })
            continue
        records.append({
            "artifact_type": ARTIFACT_TYPE,
            "family": row["source_video"],
            "label": _label_fields(row),
            "label_set": "owner",
            "modalities": {"audio_onsets_v2": audio, "ball_inflections": ball, "wrist_velocity_peaks": wrist},
            "provenance": {
                "clip_id": clip_id,
                "clip_video_sha256": row.get("video_sha256"),
                "ground_truth": True,
                "label_provenance": "human_gt",
                "manifest": owner_manifest_ref,
                "owner_review": canonicalize(row.get("review")),
                "source_video": row["source_video"],
                "source_video_sha256": None,
                "stratum": row.get("stratum"),
                "video_path": row.get("video_path"),
            },
            "record_id": record_id,
            "row_key": row_key,
            "schema_version": SCHEMA_VERSION,
            "split": row["split"],
            "window": window,
        })

    for row in teacher_manifest["rows"]:
        row_key = str(row["video"])
        record_id = f"teacher:{row_key}"
        family = row["source_video"]
        if row["split"] == "val":
            raise SystemExit(f"{TEACHER_VAL_ERROR}: teacher row {row_key} declares split=val")
        if row_key in protected["label_ids"]:
            raise SystemExit(f"{PROTECTED_IDENTITY_ERROR}: teacher row key {row_key} matches a protected seed id")
        if row.get("source_video_sha256") in protected["video_sha256s"]:
            raise SystemExit(f"{PROTECTED_IDENTITY_ERROR}: teacher row {row_key} carries a protected seed video sha")
        if family in quarantined_families:
            quarantined_rows.append({
                "family": family, "label_set": "teacher",
                "reason": "family_overlaps_owner_val", "row_key": row_key,
            })
            continue
        audio = _audio_modality(row, label_set="teacher", audio_entry=audio_index.get(family),
                                onsets_by_family=onsets_by_family, media_index=media_index)
        ball = _wrist_like_absent_ball()
        wrist = _wrist_modality()
        if not any(_usable(block) for block in (audio, ball, wrist)):
            media = media_index.get(family) or {}
            reason = "missing_media" if not media.get("present") else "missing_cue_artifacts"
            unbuildable.append({
                "family": family, "label_set": "teacher",
                "reason": reason, "row_key": row_key, "split": row["split"],
            })
            continue
        event = row["events"][0]
        records.append({
            "artifact_type": ARTIFACT_TYPE,
            "family": family,
            "label": _label_fields(row),
            "label_set": "teacher",
            "modalities": {"audio_onsets_v2": audio, "ball_inflections": ball, "wrist_velocity_peaks": wrist},
            "provenance": {
                "clip_id": None,
                "clip_video_sha256": None,
                "ground_truth": False,
                "label_provenance": "teacher_derived",
                "manifest": teacher_manifest_ref,
                "source_video": family,
                "source_video_sha256": row.get("source_video_sha256"),
                "teacher": {
                    "agreement_count": event.get("agreement_count"),
                    "audio_weight_eligible": event.get("audio_weight_eligible"),
                    "filter_decision": event.get("filter_decision"),
                    "focal_event_id": row.get("focal_event_id"),
                    "sample_weight": round_float(float(row.get("sample_weight") or 0.0)),
                    "source_lineage_key": row.get("source_lineage_key"),
                    "teacher_confidence": round_float(float(event.get("teacher_confidence") or 0.0)),
                },
                "video_path": None,
            },
            "record_id": record_id,
            "row_key": row_key,
            "schema_version": SCHEMA_VERSION,
            "split": row["split"],
            "window": _window_fields(row),
        })

    out_dir = Path(args.out_dir)
    records_dir = out_dir / "records"
    owner_records = [record for record in records if record["label_set"] == "owner"]
    teacher_records = [record for record in records if record["label_set"] == "teacher"]
    owner_sha = write_records_jsonl(records_dir / "owner_records.jsonl", owner_records)
    teacher_sha = write_records_jsonl(records_dir / "teacher_records.jsonl", teacher_records)

    binding_counts: dict[str, dict[str, int]] = {"owner": {}, "teacher": {}}
    for record in records:
        artifact = record["modalities"]["audio_onsets_v2"]["artifact"]
        if artifact is not None:
            binding = str(artifact.get("media_binding"))
            bucket = binding_counts[record["label_set"]]
            bucket[binding] = bucket.get(binding, 0) + 1
    cue_provenance = {
        "audio_media_binding_counts": binding_counts,
        "audio_onsets_v2_by_family": {
            family: {"media_sha256": entry.get("media_sha256"), "path": entry["path"], "sha256": entry["sha256"]}
            for family, entry in sorted(audio_index.items())
        },
        "ball_inflections_clip_count": len(ball_index),
        "wrist_velocity_peaks_note": cue_index.get("wrist_velocity_peaks_note"),
    }

    coverage = _coverage_payload(
        owner_records=owner_records,
        teacher_records=teacher_records,
        owner_manifest=owner_manifest,
        teacher_manifest=teacher_manifest,
        unbuildable=unbuildable,
        quarantined_rows=quarantined_rows,
        clip_mapping=clip_mapping,
        protected=protected,
        protected_val_overlaps=protected_val_overlaps,
        audio_only=audio_only,
        cue_provenance=cue_provenance,
    )
    coverage_sha = _write_canonical_json(out_dir / "coverage.json", coverage)
    (out_dir / "COVERAGE.md").write_text(_render_coverage_md(coverage), encoding="utf-8")

    manifest_payload = {
        "artifact_type": "racketsport_multimodal_event_dataset_manifest",
        "inputs": ledger.as_list(),
        "outputs": [
            {"path": "records/owner_records.jsonl", "sha256": owner_sha},
            {"path": "records/teacher_records.jsonl", "sha256": teacher_sha},
            {"path": "coverage.json", "sha256": coverage_sha},
        ],
        "schema_version": 1,
    }
    _write_canonical_json(out_dir / "MANIFEST.sha256.json", manifest_payload)
    print(
        f"records: owner={len(owner_records)} teacher={len(teacher_records)} "
        f"unbuildable={len(unbuildable)} quarantined={len(quarantined_rows)}"
    )
    return 0


def _wrist_like_absent_ball() -> dict[str, Any]:
    return modality_block(artifact=None, events=[], absent_reason=None,
                          series_value_key="confidence",
                          series_value_name="max_candidate_confidence_per_frame_bin")


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def _modality_stats(records: list[Mapping[str, Any]], modality: str) -> dict[str, Any]:
    total = len(records)
    with_artifact = sum(1 for record in records if record["modalities"][modality]["artifact"] is not None)
    available = sum(1 for record in records if record["modalities"][modality]["available"])
    return {
        "artifact_rate": round_float(with_artifact / total) if total else None,
        "available_rate": round_float(available / total) if total else None,
        "records_available": available,
        "records_total": total,
        "records_with_artifact": with_artifact,
    }


def _group_stats(records: list[Mapping[str, Any]], key) -> dict[str, Any]:
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        groups.setdefault(str(key(record)), []).append(record)
    return {
        name: {modality: _modality_stats(group, modality) for modality in
               ("audio_onsets_v2", "ball_inflections", "wrist_velocity_peaks")}
        for name, group in sorted(groups.items())
    }


def _coverage_payload(**kwargs: Any) -> dict[str, Any]:
    owner_records = kwargs["owner_records"]
    teacher_records = kwargs["teacher_records"]
    owner_manifest = kwargs["owner_manifest"]
    teacher_manifest = kwargs["teacher_manifest"]
    unbuildable = sorted(kwargs["unbuildable"], key=lambda item: (item["label_set"], item["row_key"]))
    quarantined_rows = sorted(kwargs["quarantined_rows"], key=lambda item: item["row_key"])
    protected = kwargs["protected"]
    modalities = ("audio_onsets_v2", "ball_inflections", "wrist_velocity_peaks")

    def label_set_block(records: list[Mapping[str, Any]], manifest_rows: int) -> dict[str, Any]:
        classes: dict[str, int] = {}
        for record in records:
            classes[record["label"]["class"]] = classes.get(record["label"]["class"], 0) + 1
        event_records = [record for record in records if record["label"]["class"] != "negative"]
        proximity: dict[str, Any] = {}
        for modality in modalities:
            near = sum(
                1 for record in event_records
                if any(
                    abs(int(event["frame_offset"]) - int(record["label"]["event_frame"])) <= 2
                    for event in record["modalities"][modality]["events"]
                )
            )
            proximity[modality] = {
                "event_records": len(event_records),
                "rate": round_float(near / len(event_records)) if event_records else None,
                "records_with_cue_within_2_frames_of_label": near,
            }
        return {
            "label_class_counts": dict(sorted(classes.items())),
            "label_proximity_within_2_frames": proximity,
            "per_family": _group_stats(records, lambda record: record["family"]),
            "per_fps_regime": _group_stats(records, lambda record: record["window"]["fps"]),
            "per_modality": {modality: _modality_stats(records, modality) for modality in modalities},
            "records_emitted": len(records),
            "rows_total": manifest_rows,
        }

    return {
        "artifact_type": COVERAGE_ARTIFACT_TYPE,
        "assertions": {
            "audio_only_teacher_events": kwargs["audio_only"],
            "protected_seed_label_id_count": len(protected["label_ids"]),
            "protected_seed_identity_matches_in_records": 0,
            "protected_train_window_overlaps": 0,
            "protected_val_window_overlaps_measured": sorted(
                kwargs["protected_val_overlaps"], key=lambda item: item["row_key"]
            ),
            "teacher_rows_in_val": 0,
        },
        "clip_time_mapping": dict(sorted(kwargs["clip_mapping"].items())),
        "cue_provenance": kwargs["cue_provenance"],
        "label_sets": {
            "owner": label_set_block(owner_records, len(owner_manifest["rows"])),
            "teacher": label_set_block(teacher_records, len(teacher_manifest["rows"])),
        },
        "not_gate_verified": True,
        "quarantined_rows": quarantined_rows,
        "schema_version": 1,
        "split_table": {
            "owner": {
                "train": sum(1 for record in owner_records if record["split"] == "train"),
                "val": sum(1 for record in owner_records if record["split"] == "val"),
            },
            "teacher": {
                "quarantined": len(quarantined_rows),
                "train": sum(1 for record in teacher_records if record["split"] == "train"),
                "val": 0,
            },
        },
        "unbuildable_rows": unbuildable,
        "verified": False,
        "window": {
            "center_bin": WINDOW_CENTER_BIN,
            "frames": WINDOW_FRAMES,
            "source": (
                "owner_102_manifest.json config.window_frames == 64 (E-v2 64-frame context); "
                "both manifests carry num_frames == 64 on every row"
            ),
        },
    }


def _render_coverage_md(coverage: Mapping[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Multimodal event dataset coverage (WS3.2)")
    lines.append("")
    lines.append("Measurement-only artifact; VERIFIED=0 stands. Deterministic rebuild:")
    lines.append("`scripts/racketsport/build_multimodal_event_dataset.py build` with the inputs")
    lines.append("pinned in `MANIFEST.sha256.json`.")
    lines.append("")
    window = coverage["window"]
    lines.append(f"Window: {window['frames']} frames, label at bin {window['center_bin']}. Source: {window['source']}.")
    lines.append("")
    lines.append("## Per-modality coverage")
    lines.append("")
    lines.append("| label set | rows | records | modality | artifact-bound | signal in window |")
    lines.append("|---|---|---|---|---|---|")
    for label_set in ("owner", "teacher"):
        block = coverage["label_sets"][label_set]
        for modality in ("audio_onsets_v2", "ball_inflections", "wrist_velocity_peaks"):
            stats = block["per_modality"][modality]
            artifact_rate = stats["artifact_rate"]
            available_rate = stats["available_rate"]
            lines.append(
                f"| {label_set} | {block['rows_total']} | {block['records_emitted']} | {modality} "
                f"| {stats['records_with_artifact']}/{stats['records_total']}"
                f" ({'-' if artifact_rate is None else f'{100 * artifact_rate:.1f}%'})"
                f" | {stats['records_available']}/{stats['records_total']}"
                f" ({'-' if available_rate is None else f'{100 * available_rate:.1f}%'}) |"
            )
    lines.append("")
    lines.append("Rates are over EMITTED records; rows never emitted are in the unbuildable ledger below.")
    lines.append("")
    lines.append("## Cue proximity to the label (ceiling predictor)")
    lines.append("")
    lines.append("Fraction of event records with at least one cue of the modality within ±2")
    lines.append("frames of the label frame (the frozen judge's macro-F1@±2 tolerance).")
    lines.append("")
    lines.append("| label set | modality | records with cue within ±2 | rate |")
    lines.append("|---|---|---|---|")
    for label_set in ("owner", "teacher"):
        proximity = coverage["label_sets"][label_set]["label_proximity_within_2_frames"]
        for modality, stats in proximity.items():
            rate = stats["rate"]
            lines.append(
                f"| {label_set} | {modality} | {stats['records_with_cue_within_2_frames_of_label']}"
                f"/{stats['event_records']} | {'-' if rate is None else f'{100 * rate:.1f}%'} |"
            )
    lines.append("")
    lines.append("## Split table")
    lines.append("")
    split = coverage["split_table"]
    lines.append("| label set | train | val | quarantined |")
    lines.append("|---|---|---|---|")
    lines.append(f"| owner | {split['owner']['train']} | {split['owner']['val']} | 0 |")
    lines.append(f"| teacher | {split['teacher']['train']} | {split['teacher']['val']} | {split['teacher']['quarantined']} |")
    lines.append("")
    lines.append("## Assertions (measured while building)")
    lines.append("")
    assertions = coverage["assertions"]
    lines.append(f"- audio-only teacher events: {assertions['audio_only_teacher_events']} (build refuses non-zero)")
    lines.append(f"- protected seed ids checked: {assertions['protected_seed_label_id_count']};"
                 f" identity matches in records: {assertions['protected_seed_identity_matches_in_records']}")
    lines.append(f"- protected train-window overlaps: {assertions['protected_train_window_overlaps']} (build refuses non-zero)")
    lines.append(f"- protected val-window overlaps measured: {len(assertions['protected_val_window_overlaps_measured'])}")
    lines.append(f"- teacher rows in val: {assertions['teacher_rows_in_val']} (build refuses non-zero)")
    lines.append("")
    lines.append("## Cue provenance")
    lines.append("")
    cue = coverage["cue_provenance"]
    lines.append("| family | audio_onsets_v2 artifact | sha256 (first 12) |")
    lines.append("|---|---|---|")
    for family, entry in cue["audio_onsets_v2_by_family"].items():
        lines.append(f"| {family} | {entry['path']} | {entry['sha256'][:12]} |")
    lines.append("")
    lines.append(f"- audio media-binding counts per label set: {json.dumps(cue['audio_media_binding_counts'], sort_keys=True)}")
    lines.append(f"- ball_inflections artifacts (per clip, built from WASB prelabel ball tracks): {cue['ball_inflections_clip_count']}")
    lines.append(f"- wrist_velocity_peaks: {cue['wrist_velocity_peaks_note']}")
    lines.append("")
    lines.append("Per-family and per-fps-regime breakdowns live in coverage.json under")
    lines.append("`label_sets.*.per_family` and `label_sets.*.per_fps_regime`; regime beyond")
    lines.append("family/fps is not determinable from the consumed artifacts.")
    lines.append("")
    lines.append("## Unbuildable rows")
    lines.append("")
    reasons: dict[tuple[str, str, str], int] = {}
    for item in coverage["unbuildable_rows"]:
        key = (item["label_set"], item["family"], item["reason"])
        reasons[key] = reasons.get(key, 0) + 1
    lines.append("| label set | family | reason | rows |")
    lines.append("|---|---|---|---|")
    for (label_set, family, reason), count in sorted(reasons.items()):
        lines.append(f"| {label_set} | {family} | {reason} | {count} |")
    if not reasons:
        lines.append("| - | - | - | 0 |")
    lines.append("")
    lines.append(f"Total unbuildable rows: {len(coverage['unbuildable_rows'])}"
                 f" (full per-row ledger in coverage.json `unbuildable_rows`).")
    lines.append("")
    lines.append("## Clip-to-source time mapping (ball modality)")
    lines.append("")
    lines.append("| clip | clip_start_s | matched onsets | median residual (s) | verified |")
    lines.append("|---|---|---|---|---|")
    for clip_id, mapping in coverage["clip_time_mapping"].items():
        lines.append(
            f"| {clip_id} | {mapping['clip_start_s_source']} | {mapping['matched_onset_count']}"
            f" | {mapping['median_residual_s']} | {mapping['verified']} |"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo-root", action="append", required=True,
                        help="Repo root(s) for resolving repo-relative inputs; first root wins. Repeatable.")
    parser.add_argument("--owner-manifest", required=True, help="Repo-relative owner 102-row manifest path.")
    parser.add_argument("--teacher-manifest", required=True, help="Repo-relative corrected teacher manifest path.")
    parser.add_argument("--cue-index", required=True, help="cue_index.json path (gen-cues writes, build reads).")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("gen-cues", help="Discover/build cue artifacts and write cue_index.json.")
    _add_common(gen)
    gen.add_argument("--cue-dir", required=True, help="Lane directory for newly built cue artifacts.")
    gen.set_defaults(func=generate_cues)

    build = sub.add_parser("build", help="Emit deterministic records + coverage from cue_index.json.")
    _add_common(build)
    build.add_argument("--protected-selector", required=True,
                       help="Repo-relative protected seed identity selector (identity fields only).")
    build.add_argument("--protected-inventory", required=True,
                       help="Repo-relative clip inventory used to map clip ids to video sha256.")
    build.add_argument("--out-dir", required=True, help="Output dataset directory.")
    build.add_argument("--expect-owner-split", default="61:41",
                       help="Frozen owner train:val row counts; build refuses drift (default 61:41).")
    build.set_defaults(func=build_dataset)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

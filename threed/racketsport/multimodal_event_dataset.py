"""Multimodal event window dataset records (WS3.2 training substrate).

Deterministic JSONL records that fuse-align the three production CPU cue
streams (audio_onsets_v2, wrist_velocity_peaks, ball_inflections) against
labeled 64-frame event windows.

Window convention: 64 frames, matching the E-v2 design's 64-frame context
(owner_102_manifest.json config.window_frames == 64; the frozen E1 judge
derives the same window from checkpoint provenance). The label event sits at
frame bin 32 for 1,245/1,248 labeled events, so the row window is the
symmetric context window around the label time; `label.dt_s` records the
signed offset of the label from the window center for the exceptions.

Determinism contract: sorted keys, compact separators, floats rounded to six
decimals, no timestamps, records sorted by record_id. `load_records_jsonl`
re-serializes every line and refuses any byte drift, so a loaded file is
proof of canonical form.

All content is measurement-only; nothing here asserts gate verification
(VERIFIED=0 remains binding).
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

SCHEMA_VERSION = 1
ARTIFACT_TYPE = "racketsport_multimodal_event_window_record"
WINDOW_FRAMES = 64
WINDOW_CENTER_BIN = 32
FLOAT_DECIMALS = 6

LABEL_SETS = ("owner", "teacher")
SPLITS = ("train", "val")
LABEL_CLASSES = ("HIT", "BOUNCE", "negative")
MODALITY_NAMES = ("audio_onsets_v2", "ball_inflections", "wrist_velocity_peaks")
ABSENT_REASONS = (
    "no_artifact",
    "no_signal_in_window",
    "media_unbound",
    "clip_time_mapping_unverified",
    "artifact_blocked",
)

_REQUIRED_TOP_KEYS = (
    "artifact_type",
    "family",
    "label",
    "label_set",
    "modalities",
    "provenance",
    "record_id",
    "row_key",
    "schema_version",
    "split",
    "window",
)


def round_float(value: float) -> float:
    """Fixed-precision float canonicalization (six decimals, no NaN/Inf)."""
    result = round(float(value), FLOAT_DECIMALS)
    if not math.isfinite(result):
        raise ValueError(f"non-finite float refused: {value!r}")
    # Avoid -0.0 so byte output is stable across arithmetic paths.
    return result + 0.0 if result != 0 else 0.0


def canonicalize(value: Any) -> Any:
    """Recursively canonicalize floats; refuse non-JSON types."""
    if isinstance(value, bool) or value is None or isinstance(value, (int, str)):
        return value
    if isinstance(value, float):
        return round_float(value)
    if isinstance(value, Mapping):
        return {str(key): canonicalize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [canonicalize(item) for item in value]
    raise TypeError(f"unsupported record value type: {type(value).__name__}")


def canonical_json_line(record: Mapping[str, Any]) -> str:
    return json.dumps(
        canonicalize(record), sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def _is_rounded(value: float) -> bool:
    return value == round(value, FLOAT_DECIMALS)


def _validate_modality(name: str, block: Any, errors: list[str]) -> None:
    prefix = f"modalities.{name}"
    if not isinstance(block, Mapping):
        errors.append(f"{prefix}: not an object")
        return
    required = {"artifact", "available", "events", "reason", "series"}
    missing = required - set(block)
    if missing:
        errors.append(f"{prefix}: missing keys {sorted(missing)}")
        return
    available = block["available"]
    if not isinstance(available, bool):
        errors.append(f"{prefix}.available: not a bool")
        return
    events = block["events"]
    if not isinstance(events, list):
        errors.append(f"{prefix}.events: not a list")
        return
    if available:
        if block["reason"] is not None:
            errors.append(f"{prefix}.reason: must be null when available")
        if not isinstance(block["artifact"], Mapping):
            errors.append(f"{prefix}.artifact: required when available")
        if len(events) < 1:
            errors.append(f"{prefix}.events: empty although available")
        series = block["series"]
        if not isinstance(series, Mapping):
            errors.append(f"{prefix}.series: required when available")
        else:
            values = series.get("values")
            if series.get("length") != WINDOW_FRAMES:
                errors.append(f"{prefix}.series.length: expected {WINDOW_FRAMES}")
            if not isinstance(values, list) or len(values) != WINDOW_FRAMES:
                errors.append(f"{prefix}.series.values: expected {WINDOW_FRAMES} values")
    else:
        if block["reason"] not in ABSENT_REASONS:
            errors.append(f"{prefix}.reason: {block['reason']!r} not in {ABSENT_REASONS}")
        if events:
            errors.append(f"{prefix}.events: must be empty when masked absent")
        if block["series"] is not None:
            errors.append(f"{prefix}.series: must be null when masked absent (never zero-filled)")
    for index, event in enumerate(events):
        if not isinstance(event, Mapping):
            errors.append(f"{prefix}.events[{index}]: not an object")
            continue
        offset = event.get("offset_s")
        frame_offset = event.get("frame_offset")
        if not isinstance(offset, (int, float)) or isinstance(offset, bool):
            errors.append(f"{prefix}.events[{index}].offset_s: missing")
            continue
        if not isinstance(frame_offset, int) or isinstance(frame_offset, bool):
            errors.append(f"{prefix}.events[{index}].frame_offset: missing")
            continue
        if not 0 <= frame_offset < WINDOW_FRAMES:
            errors.append(f"{prefix}.events[{index}].frame_offset: {frame_offset} out of [0,{WINDOW_FRAMES})")
        if offset < 0:
            errors.append(f"{prefix}.events[{index}].offset_s: negative")


def validate_record(record: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if not isinstance(record, Mapping):
        return ["record is not an object"]
    missing = [key for key in _REQUIRED_TOP_KEYS if key not in record]
    if missing:
        errors.append(f"missing top-level keys: {missing}")
        return errors
    if record["artifact_type"] != ARTIFACT_TYPE:
        errors.append(f"artifact_type: {record['artifact_type']!r}")
    if record["schema_version"] != SCHEMA_VERSION:
        errors.append(f"schema_version: {record['schema_version']!r}")
    if record["label_set"] not in LABEL_SETS:
        errors.append(f"label_set: {record['label_set']!r}")
    if record["split"] not in SPLITS:
        errors.append(f"split: {record['split']!r}")
    expected_id = f"{record['label_set']}:{record['row_key']}"
    if record["record_id"] != expected_id:
        errors.append(f"record_id: {record['record_id']!r} != {expected_id!r}")

    window = record["window"]
    if not isinstance(window, Mapping):
        errors.append("window: not an object")
    else:
        if window.get("frames") != WINDOW_FRAMES:
            errors.append(f"window.frames: expected {WINDOW_FRAMES}")
        fps = window.get("fps")
        if not isinstance(fps, (int, float)) or isinstance(fps, bool) or fps <= 0:
            errors.append("window.fps: not a positive number")
        else:
            start = window.get("start_time_s")
            center = window.get("center_time_s")
            if isinstance(start, (int, float)) and isinstance(center, (int, float)):
                expected_center = start + WINDOW_CENTER_BIN / fps
                if abs(center - expected_center) > 10 ** -(FLOAT_DECIMALS - 1):
                    errors.append("window.center_time_s: inconsistent with start_time_s + 32/fps")

    label = record["label"]
    if not isinstance(label, Mapping):
        errors.append("label: not an object")
    else:
        if label.get("class") not in LABEL_CLASSES:
            errors.append(f"label.class: {label.get('class')!r}")
        if label.get("class") == "negative":
            if label.get("event_frame") is not None or label.get("dt_s") is not None:
                errors.append("label: negative rows must carry null event_frame/dt_s")
        else:
            event_frame = label.get("event_frame")
            if not isinstance(event_frame, int) or not 0 <= event_frame < WINDOW_FRAMES:
                errors.append(f"label.event_frame: {event_frame!r} out of [0,{WINDOW_FRAMES})")
            if not isinstance(label.get("dt_s"), (int, float)):
                errors.append("label.dt_s: missing for event rows")

    modalities = record["modalities"]
    if not isinstance(modalities, Mapping) or sorted(modalities) != sorted(MODALITY_NAMES):
        errors.append(f"modalities: expected exactly {sorted(MODALITY_NAMES)}")
    else:
        for name in MODALITY_NAMES:
            _validate_modality(name, modalities[name], errors)

    provenance = record["provenance"]
    if not isinstance(provenance, Mapping):
        errors.append("provenance: not an object")
    else:
        if provenance.get("label_provenance") not in ("human_gt", "teacher_derived"):
            errors.append(f"provenance.label_provenance: {provenance.get('label_provenance')!r}")
        if not isinstance(provenance.get("ground_truth"), bool):
            errors.append("provenance.ground_truth: not a bool")
        if record["label_set"] == "teacher" and provenance.get("ground_truth") is not False:
            errors.append("provenance.ground_truth: teacher records must be false")
        if record["label_set"] == "teacher" and provenance.get("label_provenance") != "teacher_derived":
            errors.append("provenance.label_provenance: teacher records must be teacher_derived")
        if record["label_set"] == "owner" and provenance.get("label_provenance") != "human_gt":
            errors.append("provenance.label_provenance: owner records must be human_gt")
    return errors


def write_records_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> str:
    """Validate, sort by record_id, and write canonical JSONL. Returns sha256."""
    ordered = sorted(records, key=lambda record: str(record.get("record_id")))
    lines: list[str] = []
    seen_ids: set[str] = set()
    for record in ordered:
        errors = validate_record(record)
        if errors:
            raise ValueError(f"invalid record {record.get('record_id')!r}: {errors}")
        record_id = str(record["record_id"])
        if record_id in seen_ids:
            raise ValueError(f"duplicate record_id: {record_id}")
        seen_ids.add(record_id)
        lines.append(canonical_json_line(record))
    payload = ("\n".join(lines) + "\n") if lines else ""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_records_jsonl(path: Path, *, strict: bool = True) -> list[dict[str, Any]]:
    """Load records; in strict mode refuse non-canonical bytes and invalid rows."""
    records: list[dict[str, Any]] = []
    text = Path(path).read_text(encoding="utf-8")
    if not text:
        return records
    previous_id: str | None = None
    for line_number, line in enumerate(text.splitlines(), start=1):
        record = json.loads(line)
        if strict:
            reserialized = canonical_json_line(record)
            if reserialized != line:
                raise ValueError(f"{path}:{line_number}: non-canonical record bytes")
            errors = validate_record(record)
            if errors:
                raise ValueError(f"{path}:{line_number}: {errors}")
            record_id = str(record["record_id"])
            if previous_id is not None and record_id <= previous_id:
                raise ValueError(f"{path}:{line_number}: records not sorted by record_id")
            previous_id = record_id
        records.append(record)
    return records


def cues_in_window(
    cue_times_s: Iterable[float],
    *,
    window_start_s: float,
    fps: float,
) -> list[tuple[float, int]]:
    """Map absolute cue times to (offset_s, frame_offset) pairs inside the window.

    Half-open membership: 0 <= floor((t - start) * fps) < WINDOW_FRAMES.
    """
    hits: list[tuple[float, int]] = []
    for time_s in cue_times_s:
        offset = time_s - window_start_s
        frame_offset = math.floor(offset * fps + 1e-9)
        if 0 <= frame_offset < WINDOW_FRAMES:
            hits.append((offset, frame_offset))
    hits.sort()
    return hits


def build_series(events: list[Mapping[str, Any]], *, value_key: str, value_name: str) -> dict[str, Any]:
    """Per-frame-bin max-pooled series over the 64-frame window."""
    values = [0.0] * WINDOW_FRAMES
    for event in events:
        bin_index = int(event["frame_offset"])
        value = float(event.get(value_key) or 0.0)
        if value > values[bin_index]:
            values[bin_index] = value
    return {
        "length": WINDOW_FRAMES,
        "value": value_name,
        "values": [round_float(value) for value in values],
    }


def modality_block(
    *,
    artifact: Mapping[str, Any] | None,
    events: list[Mapping[str, Any]],
    absent_reason: str | None,
    series_value_key: str,
    series_value_name: str,
) -> dict[str, Any]:
    """Assemble one modality with explicit availability mask semantics.

    A modality with no artifact, an unbound/blocked artifact, or zero in-window
    cues is masked absent (series null, events empty) — never zero-filled.
    """
    if absent_reason is not None:
        if absent_reason not in ABSENT_REASONS:
            raise ValueError(f"unknown absent reason: {absent_reason}")
        return {
            "artifact": dict(artifact) if artifact is not None else None,
            "available": False,
            "events": [],
            "reason": absent_reason,
            "series": None,
        }
    if artifact is None:
        return {
            "artifact": None,
            "available": False,
            "events": [],
            "reason": "no_artifact",
            "series": None,
        }
    if not events:
        return {
            "artifact": dict(artifact),
            "available": False,
            "events": [],
            "reason": "no_signal_in_window",
            "series": None,
        }
    ordered = sorted(events, key=lambda event: (event["offset_s"], event["frame_offset"]))
    return {
        "artifact": dict(artifact),
        "available": True,
        "events": [canonicalize(event) for event in ordered],
        "reason": None,
        "series": build_series(ordered, value_key=series_value_key, value_name=series_value_name),
    }

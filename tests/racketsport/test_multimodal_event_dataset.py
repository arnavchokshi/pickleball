"""Round-trip and mask-semantics tests for the multimodal event dataset records."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from threed.racketsport.multimodal_event_dataset import (
    ABSENT_REASONS,
    ARTIFACT_TYPE,
    MODALITY_NAMES,
    SCHEMA_VERSION,
    WINDOW_FRAMES,
    build_series,
    canonical_json_line,
    cues_in_window,
    load_records_jsonl,
    modality_block,
    round_float,
    validate_record,
    write_records_jsonl,
)


def _artifact() -> dict:
    return {
        "detector_version": "audio_onset_pop_v2",
        "media_sha256": "a" * 64,
        "path": "runs/lanes/example/audio.json",
        "sha256": "b" * 64,
        "timebase": "source_video_s",
    }


def _record(record_id: str = "owner:els_test_001", split: str = "train") -> dict:
    label_set, row_key = record_id.split(":", 1)
    events = [{"frame_offset": 32, "offset_s": 1.066667, "score": 0.5}]
    return {
        "artifact_type": ARTIFACT_TYPE,
        "family": "wBu8bC4OfUY",
        "label": {"class": "HIT", "dt_s": 0.0, "event_frame": 32, "event_time_s": 79.6, "negative_kind": None},
        "label_set": label_set,
        "modalities": {
            "audio_onsets_v2": modality_block(
                artifact=_artifact(),
                events=events,
                absent_reason=None,
                series_value_key="score",
                series_value_name="max_onset_score_per_frame_bin",
            ),
            "ball_inflections": modality_block(
                artifact=None,
                events=[],
                absent_reason=None,
                series_value_key="confidence",
                series_value_name="max_candidate_confidence_per_frame_bin",
            ),
            "wrist_velocity_peaks": modality_block(
                artifact=None,
                events=[],
                absent_reason=None,
                series_value_key="speed_mps",
                series_value_name="max_wrist_speed_mps_per_frame_bin",
            ),
        },
        "provenance": {
            "clip_id": "wBu8bC4OfUY_rally_0001",
            "clip_video_sha256": "c" * 64,
            "ground_truth": label_set == "owner",
            "label_provenance": "human_gt" if label_set == "owner" else "teacher_derived",
            "manifest": {"path": "runs/lanes/example/manifest.json", "sha256": "d" * 64},
            "source_video": "wBu8bC4OfUY",
        },
        "record_id": record_id,
        "row_key": row_key,
        "schema_version": SCHEMA_VERSION,
        "split": split,
        "window": {
            "center_time_s": 79.6,
            "duration_s": round_float(64 / 30.0),
            "fps": 30.0,
            "frames": WINDOW_FRAMES,
            "source_start_frame": 2356,
            "start_time_s": round_float(2356 / 30.0),
        },
    }


def test_valid_record_passes_validation() -> None:
    assert validate_record(_record()) == []


def test_round_trip_is_byte_identical(tmp_path: Path) -> None:
    records = [_record("owner:els_test_002", "val"), _record("owner:els_test_001", "train")]
    path = tmp_path / "records.jsonl"
    sha_first = write_records_jsonl(path, records)
    first_bytes = path.read_bytes()
    loaded = load_records_jsonl(path)
    assert [record["record_id"] for record in loaded] == ["owner:els_test_001", "owner:els_test_002"]
    sha_second = write_records_jsonl(path, loaded)
    assert path.read_bytes() == first_bytes
    assert sha_first == sha_second


def test_loader_refuses_non_canonical_bytes(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    write_records_jsonl(path, [_record()])
    # Re-write with spaced separators: same JSON value, different bytes.
    record = json.loads(path.read_text().splitlines()[0])
    path.write_text(json.dumps(record, sort_keys=True, separators=(", ", ": ")) + "\n")
    with pytest.raises(ValueError, match="non-canonical"):
        load_records_jsonl(path)


def test_loader_refuses_unsorted_records(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    lines = [
        canonical_json_line(_record("owner:els_test_002")),
        canonical_json_line(_record("owner:els_test_001")),
    ]
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="not sorted"):
        load_records_jsonl(path)


def test_writer_refuses_duplicate_record_ids(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="duplicate"):
        write_records_jsonl(tmp_path / "records.jsonl", [_record(), _record()])


def test_floats_are_fixed_precision_and_finite(tmp_path: Path) -> None:
    record = _record()
    record["window"]["start_time_s"] = 78.53333333333333
    path = tmp_path / "records.jsonl"
    write_records_jsonl(path, [record])
    loaded = load_records_jsonl(path)[0]
    assert loaded["window"]["start_time_s"] == 78.533333
    bad = _record()
    bad["window"]["start_time_s"] = float("nan")
    with pytest.raises(ValueError):
        write_records_jsonl(tmp_path / "bad.jsonl", [bad])


def test_masked_absent_modality_is_never_zero_filled() -> None:
    block = modality_block(
        artifact=_artifact(),
        events=[],
        absent_reason=None,
        series_value_key="score",
        series_value_name="max_onset_score_per_frame_bin",
    )
    assert block["available"] is False
    assert block["reason"] == "no_signal_in_window"
    assert block["series"] is None
    assert block["events"] == []

    no_artifact = modality_block(
        artifact=None,
        events=[],
        absent_reason=None,
        series_value_key="score",
        series_value_name="max_onset_score_per_frame_bin",
    )
    assert no_artifact["reason"] == "no_artifact"
    assert no_artifact["series"] is None

    unbound = modality_block(
        artifact=_artifact(),
        events=[],
        absent_reason="media_unbound",
        series_value_key="score",
        series_value_name="max_onset_score_per_frame_bin",
    )
    assert unbound["available"] is False
    assert unbound["reason"] == "media_unbound"


def test_validation_rejects_zero_filled_absent_series() -> None:
    record = _record()
    record["modalities"]["wrist_velocity_peaks"] = {
        "artifact": None,
        "available": False,
        "events": [],
        "reason": "no_artifact",
        "series": {"length": 64, "value": "max", "values": [0.0] * 64},
    }
    errors = validate_record(record)
    assert any("never zero-filled" in error for error in errors)


def test_validation_rejects_unknown_absent_reason() -> None:
    record = _record()
    record["modalities"]["wrist_velocity_peaks"]["reason"] = "unknown_reason"
    errors = validate_record(record)
    assert any("reason" in error for error in errors)
    assert "unknown_reason" not in ABSENT_REASONS


def test_validation_rejects_teacher_ground_truth_true() -> None:
    record = _record("teacher:143sf3gdwxsa:abc123")
    record["provenance"]["label_provenance"] = "teacher_derived"
    record["provenance"]["ground_truth"] = True
    errors = validate_record(record)
    assert any("ground_truth" in error for error in errors)


def test_cues_in_window_half_open_membership() -> None:
    fps = 30.0
    start = 10.0
    inside = cues_in_window([10.0, 10.5, start + 63.9 / fps], window_start_s=start, fps=fps)
    assert [frame for _, frame in inside] == [0, 15, 63]
    outside = cues_in_window([9.99, start + 64.0 / fps], window_start_s=start, fps=fps)
    assert outside == []


def test_series_max_pools_per_bin() -> None:
    events = [
        {"frame_offset": 4, "offset_s": 0.1, "score": 0.25},
        {"frame_offset": 4, "offset_s": 0.12, "score": 0.75},
    ]
    series = build_series(events, value_key="score", value_name="max_onset_score_per_frame_bin")
    assert series["values"][4] == 0.75
    assert sum(1 for value in series["values"] if value != 0.0) == 1
    assert sorted(MODALITY_NAMES) == list(MODALITY_NAMES)

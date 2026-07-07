from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.racketsport.generate_flight_corpus import (
    BurstDropoutConfig,
    apply_burst_detector_noise,
    build_error_profile_match,
)
from scripts.racketsport.list_scaffold_tools import build_scaffold_tool_index
from threed.racketsport.flight_simulator import DetectorNoiseProfile


EVAL_CALIBRATION = Path(
    "eval_clips/ball/wolverine_mixed_0200_mid_steep_corner/labels/court_calibration_metric15pt.json"
)


def _clean_track(frame_count: int = 120) -> list[dict[str, object]]:
    return [
        {
            "frame": idx,
            "t": idx / 240.0,
            "xy_px": [640.0 + idx * 0.25, 360.0 - idx * 0.1],
            "visible": True,
        }
        for idx in range(frame_count)
    ]


def _run_cli(tmp_path: Path, *, seed: int, stem: str, count: int = 8) -> tuple[Path, Path]:
    out_jsonl = tmp_path / f"{stem}.jsonl"
    report_json = tmp_path / f"{stem}_report.json"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/generate_flight_corpus.py",
            "--count",
            str(count),
            "--seed",
            str(seed),
            "--calibration",
            str(EVAL_CALIBRATION),
            "--out",
            str(out_jsonl),
            "--report",
            str(report_json),
            "--roundtrip-samples",
            "1",
            "--burst-min-frames",
            "3",
            "--burst-max-frames",
            "8",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    stdout = json.loads(completed.stdout)
    assert stdout["artifact_type"] == "racketsport_flight_corpus_generation"
    assert stdout["jsonl"] == str(out_jsonl)
    return out_jsonl, report_json


def test_burst_detector_noise_creates_contiguous_dropout_runs() -> None:
    profile = DetectorNoiseProfile(p95_jitter_px=34.0, recall=0.578, hidden_fp_rate=0.021)
    config = BurstDropoutConfig(enabled=True, burst_dropout_share=0.75, min_frames=3, max_frames=8)

    noisy = apply_burst_detector_noise(
        _clean_track(),
        seed=20260707,
        image_size=(1920, 1080),
        profile=profile,
        burst_config=config,
    )

    model = noisy["dropout_model"]
    burst_lengths = model["burst_lengths"]
    assert burst_lengths
    assert max(burst_lengths) <= config.max_frames
    assert any(length >= config.min_frames for length in burst_lengths)
    assert model["dropped_frame_count"] == len(noisy["dropped_frames"])
    assert model["true_positive_count"] == len(
        [det for det in noisy["detections"] if det["kind"] == "true_positive"]
    )
    assert model["recall"] == pytest.approx(profile.recall, rel=0.20)


def test_error_profile_match_is_side_by_side_with_per_metric_flags() -> None:
    profile = DetectorNoiseProfile(p95_jitter_px=34.0, recall=0.578, hidden_fp_rate=0.021)
    noisy = apply_burst_detector_noise(
        _clean_track(600),
        seed=77,
        image_size=(1920, 1080),
        profile=profile,
        burst_config=BurstDropoutConfig(enabled=True, min_frames=4, max_frames=10),
    )

    match = build_error_profile_match([_clean_track(600)], [noisy], profile)

    assert set(match["metrics"]) == {"jitter_p95_px", "recall", "hidden_fp_rate"}
    for metric in match["metrics"].values():
        assert set(metric) == {"measured", "generated", "relative_error", "within_20_percent"}
        assert metric["within_20_percent"] is True
    assert match["all_within_20_percent"] is True


def test_cli_same_seed_is_byte_identical(tmp_path: Path) -> None:
    first_jsonl, first_report = _run_cli(tmp_path, seed=12345, stem="first")
    second_jsonl, second_report = _run_cli(tmp_path, seed=12345, stem="second")

    assert first_jsonl.read_bytes() == second_jsonl.read_bytes()
    first = json.loads(first_report.read_text(encoding="utf-8"))
    second = json.loads(second_report.read_text(encoding="utf-8"))
    assert first["acceptance"]["noise_profile"] == second["acceptance"]["noise_profile"]
    assert first["phase2"]["burst_dropout"]["summary"] == second["phase2"]["burst_dropout"]["summary"]


def test_cli_records_embed_calibration_for_backprojection(tmp_path: Path) -> None:
    out_jsonl, report_json = _run_cli(tmp_path, seed=20260707, stem="calibration", count=5)
    rows = [json.loads(line) for line in out_jsonl.read_text(encoding="utf-8").splitlines()]
    report = json.loads(report_json.read_text(encoding="utf-8"))

    assert len(rows) == 5
    projection = rows[0]["projection"]
    calibration = projection["calibration"]
    assert projection["schema"] == "CourtCalibration"
    assert calibration["image_size"] == [1920, 1080]
    assert set(calibration) >= {"intrinsics", "extrinsics", "image_size", "coordinate_frame"}
    assert "outdoor_" not in projection["calibration_path"].lower()
    assert "indoor_" not in projection["calibration_path"].lower()
    assert report["phase2"]["calibration_embedded"] is True


def test_bounced_records_are_tagged_unmeasured(tmp_path: Path) -> None:
    out_jsonl, _ = _run_cli(tmp_path, seed=42, stem="bounce", count=12)
    rows = [json.loads(line) for line in out_jsonl.read_text(encoding="utf-8").splitlines()]
    bounced = [row for row in rows if row["truth_3d"]["bounces"]]

    assert bounced
    assert all(row["bounce_params_measured"] is False for row in bounced)
    assert all(
        bounce["bounce_params_measured"] is False
        for row in bounced
        for bounce in row["truth_3d"]["bounces"]
    )


def test_scaffold_index_registers_generate_flight_corpus() -> None:
    index = build_scaffold_tool_index(Path("."))
    tools = {tool["command_path"]: tool for tool in index["tools"]}
    tool = tools["scripts/racketsport/generate_flight_corpus.py"]

    assert tool["category"] == "physics"
    assert tool["workstream"] == "BALL"
    assert tool["task_prefix"] == "P0-7"
    assert tool["direct_cli_reference_test"] is not None

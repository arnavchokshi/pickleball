from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from threed.racketsport.stage_runtime_budget import (
    StageCost,
    fixed_cost_amortized_per_minute_video,
    fps_to_seconds_per_minute_video,
    load_body_cost_model_evidence,
    load_detection_tracking_fps_evidence,
    load_offline_person_authority_evidence,
    load_tracknet_metadata_evidence,
    load_wasb_metadata_evidence,
    measure_decode_cost,
    per_frame_ms_to_seconds_per_minute_video,
    per_unit_ms_to_seconds_per_minute_video,
    stage_cost_from_wall_clock,
    wall_seconds_to_seconds_per_minute_video,
)


# --- pure conversion math ------------------------------------------------------------


def test_per_frame_ms_to_seconds_per_minute_video_basic():
    # 482ms/frame at 60fps: 60*60=3600 frames/minute * 0.482s = 1735.2s/minute-of-video.
    assert per_frame_ms_to_seconds_per_minute_video(482.0, video_fps=60.0) == pytest.approx(1735.2)


def test_per_frame_ms_to_seconds_per_minute_video_rejects_bad_input():
    with pytest.raises(ValueError):
        per_frame_ms_to_seconds_per_minute_video(-1.0, video_fps=60.0)
    with pytest.raises(ValueError):
        per_frame_ms_to_seconds_per_minute_video(10.0, video_fps=0.0)


def test_per_unit_ms_to_seconds_per_minute_video_body_event_triggered():
    # 310ms/person-frame at 24 person-frames/video-second (event-triggered scenario):
    # 24*60=1440 person-frames/minute * 0.310s = 446.4s/minute-of-video.
    result = per_unit_ms_to_seconds_per_minute_video(310.0, units_per_video_second=24.0)
    assert result == pytest.approx(446.4)


def test_fps_to_seconds_per_minute_video_basic():
    # 39.42 processed fps at 60 video fps: 3600/39.42 ~= 91.3s/minute-of-video.
    result = fps_to_seconds_per_minute_video(39.42, video_fps=60.0)
    assert result == pytest.approx(3600.0 / 39.42)


def test_fps_to_seconds_per_minute_video_rejects_nonpositive():
    with pytest.raises(ValueError):
        fps_to_seconds_per_minute_video(0.0, video_fps=60.0)


def test_wall_seconds_to_seconds_per_minute_video_basic():
    # 20s wall for a 10s clip -> 2x realtime -> 120s compute per minute-of-video.
    result = wall_seconds_to_seconds_per_minute_video(20.0, 10.0)
    assert result == pytest.approx(120.0)


def test_wall_seconds_to_seconds_per_minute_video_rejects_zero_clip():
    with pytest.raises(ValueError):
        wall_seconds_to_seconds_per_minute_video(1.0, 0.0)


def test_fixed_cost_amortized_per_minute_video():
    # 16.3s setup amortized over a 10-minute job -> 1.63s/minute-of-video.
    result = fixed_cost_amortized_per_minute_video(16.3, video_minutes=10.0)
    assert result == pytest.approx(1.63)


def test_stage_cost_from_wall_clock_builds_stage_cost():
    cost = stage_cost_from_wall_clock(
        stage="world_build", wall_seconds=2.07, clip_seconds_processed=10.01, basis="fresh timing", source="x"
    )
    assert isinstance(cost, StageCost)
    assert cost.seconds_per_minute_video == pytest.approx(wall_seconds_to_seconds_per_minute_video(2.07, 10.01))
    as_dict = cost.to_dict()
    assert as_dict["stage"] == "world_build"
    assert as_dict["compute_minutes_per_video_minute"] == pytest.approx(as_dict["seconds_per_minute_video"] / 60.0)


# --- evidence loaders (deterministic synthetic fixtures, no GPU/model calls) ---------


def test_load_tracknet_metadata_evidence(tmp_path: Path):
    payload = {
        "runtime": {
            "wall_seconds": 172.88316297900747,
            "video_seconds_processed": 172.88316297900747 / (1151 / 60.0) * (1151 / 60.0),
            "processed_frame_count": 1151,
        }
    }
    # keep video_seconds_processed simple & explicit
    payload["runtime"]["video_seconds_processed"] = 1151 / 60.0
    path = tmp_path / "tracknet_metadata.json"
    path.write_text(json.dumps(payload))

    cost = load_tracknet_metadata_evidence(path, video_fps=60.0)
    assert cost.stage == "ball_inference_tracknetv3"
    expected = wall_seconds_to_seconds_per_minute_video(172.88316297900747, 1151 / 60.0)
    assert cost.seconds_per_minute_video == pytest.approx(expected)


def test_load_wasb_metadata_evidence_top_level_shape(tmp_path: Path):
    payload = {"effective_fps": 34.63729337566924, "frame_count": 1151, "fps": 60.0}
    path = tmp_path / "wasb_metadata.json"
    path.write_text(json.dumps(payload))

    cost = load_wasb_metadata_evidence(path)
    assert cost.stage == "ball_inference_wasb_verifier"
    assert cost.seconds_per_minute_video == pytest.approx(fps_to_seconds_per_minute_video(34.63729337566924, video_fps=60.0))


def test_load_wasb_metadata_evidence_nested_runtime_shape(tmp_path: Path):
    payload = {
        "fps": 60.0,
        "frame_count": 1151,
        "runtime": {
            "effective_fps": 34.63729337566924,
            "processed_frame_count": 1151,
            "source_video_fps": 60.0,
        },
    }
    path = tmp_path / "wasb_metadata.json"
    path.write_text(json.dumps(payload))

    cost = load_wasb_metadata_evidence(path)
    assert cost.seconds_per_minute_video == pytest.approx(fps_to_seconds_per_minute_video(34.63729337566924, video_fps=60.0))


def test_load_offline_person_authority_evidence(tmp_path: Path):
    payload = {
        "effective_fps": 0.474538,
        "wall_time_s": 1264.386594,
        "config": {"reid_device": "cpu"},
    }
    path = tmp_path / "offline_authority_summary.json"
    path.write_text(json.dumps(payload))

    cost = load_offline_person_authority_evidence(path, video_fps=60.0)
    assert cost.stage == "association_reid_global"
    # sanity: this stage should be the most expensive in the whole budget (real repo finding).
    assert cost.seconds_per_minute_video > 1000.0


def test_load_body_cost_model_evidence(tmp_path: Path):
    payload = {"body_a100_fast_seconds_per_person_frame": 0.30977042253521125, "body_setup_seconds": 16.29445}
    path = tmp_path / "model_assumptions.json"
    path.write_text(json.dumps(payload))

    cost = load_body_cost_model_evidence(path, scenario_person_frames_per_video_second=24.0)
    assert cost.seconds_per_minute_video == pytest.approx(446.35, abs=0.5)


def test_load_detection_tracking_fps_evidence_top_level_key(tmp_path: Path):
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps({"fps": 39.42}))
    cost = load_detection_tracking_fps_evidence(path, video_fps=60.0)
    assert cost.stage == "detection_tracking"
    assert cost.seconds_per_minute_video == pytest.approx(fps_to_seconds_per_minute_video(39.42, video_fps=60.0))


def test_load_detection_tracking_fps_evidence_nested_timing_key(tmp_path: Path):
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps({"timing": {"fps": 25.822947}}))
    cost = load_detection_tracking_fps_evidence(path, video_fps=60.0)
    assert cost.seconds_per_minute_video == pytest.approx(fps_to_seconds_per_minute_video(25.822947, video_fps=60.0))


def test_load_detection_tracking_fps_evidence_missing_key_raises(tmp_path: Path):
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps({"other": 1}))
    with pytest.raises(KeyError):
        load_detection_tracking_fps_evidence(path, video_fps=60.0)


# --- decode benchmark: real ffmpeg, tiny synthetic clip ------------------------------


def _write_test_video(path: Path, *, duration_s: float = 0.5, fps: int = 30) -> None:
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is required for decode-cost measurement test")
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"testsrc=size=320x240:rate={fps}:duration={duration_s}",
        "-pix_fmt",
        "yuv420p",
        str(path),
    ]
    subprocess.run(command, check=True)


def test_measure_decode_cost_real_ffmpeg(tmp_path: Path):
    clip = tmp_path / "tiny.mp4"
    _write_test_video(clip)
    cost = measure_decode_cost(clip, backend="cpu")
    assert cost.stage == "decode"
    assert cost.seconds_per_minute_video >= 0.0
    assert "decode_fps" in cost.notes

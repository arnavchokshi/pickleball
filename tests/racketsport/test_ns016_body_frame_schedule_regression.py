from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.racketsport import process_video
from tests.racketsport.test_process_video import _base_options, _make_video, _write_json
from threed.racketsport.orchestrator import StageContext, _precomputed_process_video_body_execution


def _two_frame_tracks() -> dict:
    return {
        "schema_version": 1,
        "fps": 30.0,
        "players": [
            {
                "id": 1,
                "side": "near",
                "role": "left",
                "frames": [
                    {"t": 0.0, "bbox": [10, 10, 50, 50], "world_xy": [0, 0], "conf": 0.9},
                    {"t": 1 / 30, "bbox": [11, 10, 51, 50], "world_xy": [0, 0], "conf": 0.9},
                ],
            }
        ],
        "rally_spans": [],
    }


def test_frames_stage_rejects_and_repairs_incomplete_warm_cache(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    _make_video(video)
    options = _base_options(tmp_path, video=video, court_corners=None)
    options.no_gpu = False
    options.body_skeleton_stride = 1
    pipeline = process_video.ProcessVideoPipeline(options)
    pipeline._stage_ingest()
    _write_json(options.clip_dir / "tracks.json", _two_frame_tracks())
    body_frames = options.clip_dir / "body_frames"
    body_frames.mkdir()
    (body_frames / "frame_000000.jpg").write_bytes(b"stale-partial-cache")

    outcome = pipeline._stage_frames()

    assert outcome.status == "ran"
    assert outcome.metrics["schedule_materialized_equal"] is True
    assert sorted(path.name for path in body_frames.glob("frame_*.jpg")) == [
        "frame_000000.jpg",
        "frame_000001.jpg",
    ]
    assert any("cached body_frames/ rejected" in note for note in outcome.notes)


def test_body_runner_handoff_rejects_execution_outside_frames_schedule(tmp_path: Path) -> None:
    execution = {
        "schema_version": 1,
        "artifact_type": "racketsport_body_compute_execution",
        "summary": {"base_skeleton_stride": 2},
        "scheduled_frames": [{"frame_idx": 5}],
    }
    schedule = {"schema_version": 1, "frame_indexes": [2]}
    (tmp_path / "body_compute_execution.json").write_text(json.dumps(execution), encoding="utf-8")
    (tmp_path / "process_video_frame_schedule.json").write_text(json.dumps(schedule), encoding="utf-8")
    context = StageContext(
        clip="cold_handoff",
        inputs_dir=tmp_path,
        run_dir=tmp_path,
        sport="pickleball",
    )

    with pytest.raises(ValueError, match=r"missing_frames=\[5\]"):
        _precomputed_process_video_body_execution(context, skeleton_stride=2)


def test_body_runner_handoff_consumes_exact_persisted_execution(tmp_path: Path) -> None:
    execution = {
        "schema_version": 1,
        "artifact_type": "racketsport_body_compute_execution",
        "summary": {"base_skeleton_stride": 2},
        "scheduled_frames": [{"frame_idx": 2}, {"frame_idx": 5}],
    }
    schedule = {"schema_version": 1, "frame_indexes": [2, 5, 8]}
    (tmp_path / "body_compute_execution.json").write_text(json.dumps(execution), encoding="utf-8")
    (tmp_path / "process_video_frame_schedule.json").write_text(json.dumps(schedule), encoding="utf-8")
    context = StageContext(
        clip="cold_handoff",
        inputs_dir=tmp_path,
        run_dir=tmp_path,
        sport="pickleball",
    )

    loaded = _precomputed_process_video_body_execution(context, skeleton_stride=2)

    assert loaded == execution

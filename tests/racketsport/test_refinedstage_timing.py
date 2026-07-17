from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts.racketsport import process_video
from tests.racketsport.test_process_video import (
    _ball_track_payload,
    _base_options,
    _contact_windows_payload,
    _court_calibration_payload,
    _make_video,
    _sam3d_skeleton_payload,
    _tracks_payload,
    _write_json,
)


def _refined_pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> process_video.ProcessVideoPipeline:
    video = tmp_path / "clip.mp4"
    _make_video(video)
    options = _base_options(tmp_path, video=video, court_corners=None)
    options.no_ball_arc = False
    options.clip_dir.mkdir(parents=True, exist_ok=True)
    _write_json(options.clip_dir / "contact_windows.json", _contact_windows_payload())
    _write_json(options.clip_dir / "skeleton3d.json", _sam3d_skeleton_payload())
    _write_json(options.clip_dir / "ball_track.json", _ball_track_payload(frame_count=3))
    _write_json(options.clip_dir / "ball_candidates.json", {"schema_version": 1, "frames": []})
    _write_json(options.clip_dir / "ball_inflections.json", {"schema_version": 1, "candidates": []})
    _write_json(options.clip_dir / "wrist_velocity_peaks.json", {"schema_version": 1, "peaks": []})
    _write_json(options.clip_dir / "court_calibration.json", _court_calibration_payload())
    _write_json(options.clip_dir / "tracks.json", _tracks_payload())
    _write_json(options.clip_dir / "frame_times.json", {"schema_version": 1, "frames": []})
    _write_json(options.clip_dir / "timebase_contract.json", {"schema_version": 1, "status": "typed"})
    _write_json(
        options.clip_dir / "audio_onsets_v2.json",
        {
            "schema_version": 1,
            "detector_version": "audio_onsets_v2.test",
            "status": "review_only",
            "summary": {"threshold_mad": 5.0, "min_pop_band_ratio": 0.2},
            "onsets": [],
        },
    )
    monkeypatch.setattr(
        process_video,
        "build_wrist_velocity_peaks_from_file",
        lambda *_args, **_kwargs: {"schema_version": 1, "peaks": []},
    )
    return process_video.ProcessVideoPipeline(options)


def _write_arc_outputs(clip_dir: Path, *, status: str) -> None:
    for name in (
        "ball_bounce_candidates.json",
        "ball_track_arc_solved.json",
        "ball_arc_render.json",
        "ball_flight_sanity.json",
        "ball_chain_manifest.json",
    ):
        _write_json(clip_dir / name, {"schema_version": 1, "status": status})


@pytest.mark.parametrize("dependency_name", ["frame_times.json", "wrist_velocity_peaks.json"])
def test_refined_contact_reuse_refuses_newly_booked_dependency_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dependency_name: str,
) -> None:
    pipeline = _refined_pipeline(tmp_path, monkeypatch)
    assert pipeline._stage_events_refined().status == "ran"
    assert pipeline._stage_events_refined().metrics["reason"] == "dependency_hashes_match"

    dependency_path = pipeline.clip_dir / dependency_name
    payload = json.loads(dependency_path.read_text(encoding="utf-8"))
    payload["reuse_invalidation_probe"] = dependency_name
    _write_json(dependency_path, payload)

    rebuilt = pipeline._stage_events_refined()
    assert rebuilt.status == "ran"
    assert rebuilt.metrics["reuse_refusal_reason"] == process_video.DEPENDENCY_HASH_MISMATCH


def test_cold_then_reuse_run_times_explicit_refined_stages_without_rebuild(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = _refined_pipeline(tmp_path, monkeypatch)
    calls: list[Path | None] = []

    def fake_arc(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs.get("contact_windows_path"))
        _write_arc_outputs(pipeline.clip_dir, status="ran")
        return {"status": "ran", "summary": {"seed_anchor_count": 0}}

    monkeypatch.setattr(process_video, "run_default_ball_arc_chain", fake_arc)

    cold_events = pipeline._run_stage_safely("events_refined", pipeline._stage_events_refined)
    cold_arc = pipeline._run_stage_safely("ball_arc_refined", pipeline._stage_ball_arc_refined)

    assert cold_events.status == "ran"
    assert cold_arc.status == "ran"
    assert cold_events.wall_seconds > 0.0
    assert cold_arc.wall_seconds > 0.0
    assert cold_events.as_dict()["wall_seconds"] >= 0.0
    assert cold_arc.as_dict()["wall_seconds"] >= 0.0
    assert calls == [pipeline.clip_dir / "contact_windows_refined_v1.json"]

    reuse_pipeline = process_video.ProcessVideoPipeline(pipeline.options)
    reuse_events = reuse_pipeline._run_stage_safely("events_refined", reuse_pipeline._stage_events_refined)
    reuse_arc = reuse_pipeline._run_stage_safely("ball_arc_refined", reuse_pipeline._stage_ball_arc_refined)

    assert reuse_events.status == "skipped"
    assert reuse_arc.status == "skipped"
    assert "reused content-addressed stage generation" in reuse_events.notes[0]
    assert "reused content-addressed stage generation" in reuse_arc.notes[0]
    assert calls == [pipeline.clip_dir / "contact_windows_refined_v1.json"]


def test_world_does_not_call_or_absorb_refined_stages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = _refined_pipeline(tmp_path, monkeypatch)

    def hidden_call_is_forbidden() -> process_video.StageOutcome:
        raise AssertionError("refined stage was called from world")

    monkeypatch.setattr(pipeline, "_stage_events_refined", hidden_call_is_forbidden)
    monkeypatch.setattr(pipeline, "_stage_ball_arc_refined", hidden_call_is_forbidden)

    world = pipeline._run_stage_safely("world", pipeline._stage_world)

    assert world.status == "ran"
    assert world.wall_seconds > 0.0
    assert "events_refined" not in world.metrics
    assert "ball_arc_refined" not in world.metrics


def test_track_a_segment_guard_result_is_typed_degraded_and_timed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = _refined_pipeline(tmp_path, monkeypatch)
    assert pipeline._stage_events_refined().status == "ran"

    calls = 0

    def guarded_arc(**_kwargs: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        _write_arc_outputs(pipeline.clip_dir, status="degraded")
        return {
            "status": "degraded",
            "summary": {
                "seed_anchor_count": 2,
                "segment_budget_exceeded_count": 1,
                "segment_budget_exceeded_ids": [7],
                "missing_segment_reasons": {"segment_budget_exceeded": 1},
            },
        }

    monkeypatch.setattr(process_video, "run_default_ball_arc_chain", guarded_arc)

    outcome = pipeline._run_stage_safely("ball_arc_refined", pipeline._stage_ball_arc_refined)

    assert outcome.status == "degraded"
    assert outcome.wall_seconds > 0.0
    assert outcome.metrics["solver_status"] == "degraded"
    assert outcome.metrics["segment_budget_exceeded_count"] == 1
    assert outcome.metrics["segment_budget_exceeded_ids"] == [7]
    assert outcome.metrics["missing_segment_reasons"] == {"segment_budget_exceeded": 1}
    assert "segment_budget_exceeded" in " ".join(outcome.notes)

    reuse_pipeline = process_video.ProcessVideoPipeline(pipeline.options)
    reused = reuse_pipeline._run_stage_safely("ball_arc_refined", reuse_pipeline._stage_ball_arc_refined)
    assert reused.status == "skipped"
    assert reused.metrics["segment_budget_exceeded_count"] == 1
    assert calls == 1


def test_track_a_segment_guard_result_is_also_degraded_on_coarse_arc_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = _refined_pipeline(tmp_path, monkeypatch)

    def guarded_arc(**_kwargs: Any) -> dict[str, Any]:
        _write_arc_outputs(pipeline.clip_dir, status="degraded")
        return {
            "status": "degraded",
            "summary": {
                "segment_budget_exceeded_count": 1,
                "segment_budget_exceeded_ids": [7],
                "missing_segment_reasons": {"segment_budget_exceeded": 1},
            },
        }

    monkeypatch.setattr(process_video, "run_default_ball_arc_chain", guarded_arc)

    outcome = pipeline._run_stage_safely("ball_arc", pipeline._stage_ball_arc)

    assert outcome.status == "degraded"
    assert outcome.wall_seconds > 0.0
    assert outcome.metrics["segment_budget_exceeded_count"] == 1
    assert outcome.metrics["segment_budget_exceeded_ids"] == [7]
    assert outcome.metrics["missing_segment_reasons"] == {"segment_budget_exceeded": 1}


def test_refined_arc_timeout_exception_is_typed_degraded_not_failed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = _refined_pipeline(tmp_path, monkeypatch)
    assert pipeline._stage_events_refined().status == "ran"

    def timeout_arc(**_kwargs: Any) -> dict[str, Any]:
        raise TimeoutError("segment_budget_exceeded")

    monkeypatch.setattr(process_video, "run_default_ball_arc_chain", timeout_arc)

    outcome = pipeline._run_stage_safely("ball_arc_refined", pipeline._stage_ball_arc_refined)

    assert outcome.status == "degraded"
    assert outcome.wall_seconds > 0.0
    assert outcome.metrics["reason"] == "arc_chain_timeout"
    assert outcome.metrics["typed_degraded_reason"] == "segment_budget_exceeded"
    assert "timed out fail-closed" in outcome.notes[0]

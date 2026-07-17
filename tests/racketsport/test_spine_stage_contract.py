from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.racketsport import process_video
from tests.racketsport.test_process_video import _base_options, _make_video, _tracks_payload, _write_json
from threed.racketsport.process_video_body_frames import (
    BodyFrameMaterializationError,
    BodyFrameScheduleError,
)


def _frames_pipeline(tmp_path: Path) -> process_video.ProcessVideoPipeline:
    video = tmp_path / "clip.mp4"
    _make_video(video)
    options = _base_options(tmp_path, video=video, court_corners=None)
    options.no_gpu = False
    options.clip_dir.mkdir(parents=True, exist_ok=True)
    _make_video(options.clip_dir / "source.mp4")
    _write_json(options.clip_dir / "tracks.json", _tracks_payload())
    return process_video.ProcessVideoPipeline(options)


def test_authoritative_graph_projects_exact_23_24_25_stage_contracts() -> None:
    serial = process_video.authoritative_stage_names(
        rally_gating=False,
        verify_viewer=False,
        body_schedule="serial",
    )
    overlap = process_video.authoritative_stage_names(
        rally_gating=False,
        verify_viewer=False,
        body_schedule="overlap",
    )

    assert len(serial) == 23
    assert len(process_video.authoritative_stage_names(rally_gating=True, verify_viewer=False)) == 24
    assert len(process_video.authoritative_stage_names(rally_gating=False, verify_viewer=True)) == 24
    assert len(process_video.authoritative_stage_names(rally_gating=True, verify_viewer=True)) == 25
    assert set(overlap) == set(serial)
    assert serial[6:12] == ("ball", "ball_arc", "events", "ball_fill", "frames", "body")
    assert overlap[6:12] == ("frames", "ball", "ball_arc", "events", "ball_fill", "body")
    assert serial[13:16] == overlap[13:16] == (
        "grounding_refine",
        "placement_trajectory_refine",
        "paddle_pose",
    )
    assert serial[16:19] == overlap[16:19] == ("events_refined", "ball_arc_refined", "world")


def test_typed_optional_absence_degrades_and_execution_continues(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    _make_video(video)
    pipeline = process_video.ProcessVideoPipeline(_base_options(tmp_path, video=video, court_corners=None))
    executed: list[str] = []

    def optional_stage() -> process_video.StageOutcome:
        raise process_video.ExpectedOptionalAbsence(
            "degraded",
            "optional_fixture_absent",
            "optional fixture is intentionally absent",
        )

    def after_stage() -> process_video.StageOutcome:
        executed.append("after")
        return process_video.StageOutcome(stage="after", status="ran", wall_seconds=0.0)

    hard_failed = pipeline._run_stage_list([("optional", optional_stage), ("after", after_stage)])

    assert hard_failed is False
    assert [outcome.status for outcome in pipeline.stage_outcomes] == ["degraded", "ran"]
    assert pipeline.stage_outcomes[0].metrics["expected_optional_absence"]["reason_code"] == "optional_fixture_absent"
    assert executed == ["after"]


@pytest.mark.parametrize(
    "error",
    [
        BodyFrameScheduleError("schedule shortfall"),
        BodyFrameMaterializationError("materialization shortfall"),
    ],
)
def test_frames_typed_schedule_errors_surface_as_hard_stage_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error: RuntimeError,
) -> None:
    pipeline = _frames_pipeline(tmp_path)

    def fail_schedule(*_args, **_kwargs):  # noqa: ANN001
        raise error

    monkeypatch.setattr(process_video, "build_frame_schedule", fail_schedule)

    outcome = pipeline._run_stage_safely("frames", pipeline._stage_frames)

    assert outcome.status == "failed"
    assert "BODY frames-stage schedule validation failed" in " ".join(outcome.notes)
    assert type(error).__name__ in " ".join(outcome.notes)


def test_frames_missing_materializer_validation_is_loud(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline = _frames_pipeline(tmp_path)

    def fake_materialize(**kwargs):  # noqa: ANN001
        return {
            "schedule": kwargs["schedule"],
            "notes": [],
            "frame_count": len(kwargs["schedule"]["frame_indexes"]),
            "total_bytes": 0,
        }

    monkeypatch.setattr(process_video, "materialize_process_video_frames", fake_materialize)

    outcome = pipeline._run_stage_safely("frames", pipeline._stage_frames)

    assert outcome.status == "failed"
    assert "validation.equal=true" in " ".join(outcome.notes)


def test_written_schedule_shortfall_against_current_plan_is_typed_loud() -> None:
    with pytest.raises(BodyFrameScheduleError, match=r"current frame_compute_plan.json.*missing_frames=\[3\]"):
        process_video._assert_frame_schedule_covers_required(
            {"frame_indexes": [0, 1, 2]},
            required_frame_indexes={1, 3},
            frame_compute_plan_present=True,
        )


def test_unexpected_stage_error_artifact_contains_full_traceback(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    _make_video(video)
    pipeline = process_video.ProcessVideoPipeline(_base_options(tmp_path, video=video, court_corners=None))

    def bug() -> process_video.StageOutcome:
        return 1 / 0  # type: ignore[return-value]

    stopped = pipeline._run_stage_list(
        [
            ("bug", bug),
            ("must_not_run", lambda: process_video.StageOutcome("must_not_run", "ran", 0.0)),
        ]
    )

    assert stopped is True
    assert [outcome.stage for outcome in pipeline.stage_outcomes] == ["bug"]
    outcome = pipeline.stage_outcomes[0]
    assert outcome.status == "failed"
    payload = json.loads((pipeline.clip_dir / outcome.artifacts[0]).read_text(encoding="utf-8"))
    assert payload["exception_type"] == "ZeroDivisionError"
    assert "in bug" in payload["traceback"]
    assert "ZeroDivisionError: division by zero" in payload["traceback"]


def test_artifact_validator_programming_error_is_not_reported_as_invalid_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = tmp_path / "tracks.json"
    _write_json(artifact, _tracks_payload())

    def validator_bug(*_args, **_kwargs):  # noqa: ANN001
        raise ZeroDivisionError("validator bug")

    monkeypatch.setattr(process_video, "validate_artifact_file", validator_bug)

    with pytest.raises(ZeroDivisionError, match="validator bug"):
        process_video._valid_artifact("tracks", artifact)

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts.racketsport import process_video
from tests.racketsport.test_placement_trajectory_refine import _phases, _skeleton, _tracks
from tests.racketsport.test_process_video import _base_options, _make_video, _write_json


def _pipeline(tmp_path: Path, *, enabled: bool) -> process_video.ProcessVideoPipeline:
    video = tmp_path / "clip.mp4"
    _make_video(video)
    options = _base_options(tmp_path, video=video, court_corners=None)
    options.placement_trajectory_refine = enabled
    options.placement_trajectory_refine_explicit = enabled
    options.clip_dir.mkdir(parents=True, exist_ok=True)
    return process_video.ProcessVideoPipeline(options)


def _write_stage_inputs(pipeline: process_video.ProcessVideoPipeline) -> dict[str, Path]:
    paths = {
        "skeleton3d": pipeline.clip_dir / "skeleton3d.json",
        "tracks": pipeline.clip_dir / "tracks.json",
        "foot_contact_phases": pipeline.clip_dir / "foot_contact_phases.json",
        "placement": pipeline.clip_dir / "placement.json",
        "grounding_refinement": pipeline.clip_dir / "body_grounding_refinement.json",
    }
    _write_json(paths["skeleton3d"], _skeleton(jitter=[0.0, 0.02, -0.01, 0.0]))
    _write_json(paths["tracks"], _tracks(4))
    _write_json(paths["foot_contact_phases"], _phases([0, 1, 2, 3]))
    _write_json(paths["placement"], {"artifact_type": "racketsport_placement", "players": []})
    _write_json(
        paths["grounding_refinement"],
        {"artifact_type": "racketsport_body_grounding_refinement", "status": "ran"},
    )
    return paths


def test_default_off_is_byte_parity_for_every_preexisting_artifact(tmp_path: Path) -> None:
    pipeline = _pipeline(tmp_path, enabled=False)
    paths = _write_stage_inputs(pipeline)
    before = {name: path.read_bytes() for name, path in paths.items()}

    outcome = pipeline._stage_placement_trajectory_refine()

    assert outcome.status == "skipped"
    assert outcome.metrics["expected_optional_absence"] == {
        "reason_code": "placement_trajectory_refine_disabled",
        "stage_status": "skipped",
    }
    assert {name: path.read_bytes() for name, path in paths.items()} == before
    assert not (pipeline.clip_dir / "placement_trajectory_refined.json").exists()


def test_flag_on_cold_run_emits_preview_artifact_with_typed_provenance_and_keeps_raw_inputs(
    tmp_path: Path,
) -> None:
    pipeline = _pipeline(tmp_path, enabled=True)
    paths = _write_stage_inputs(pipeline)
    before = {name: path.read_bytes() for name, path in paths.items()}

    outcome = pipeline._run_stage_safely(
        "placement_trajectory_refine",
        pipeline._stage_placement_trajectory_refine,
    )

    assert outcome.status == "ran"
    assert outcome.trust_badge == "preview"
    assert outcome.artifacts == ["placement_trajectory_refined.json"]
    assert {name: path.read_bytes() for name, path in paths.items()} == before

    artifact = json.loads(
        (pipeline.clip_dir / "placement_trajectory_refined.json").read_text(encoding="utf-8")
    )
    assert artifact["artifact_type"] == "placement_trajectory_refined"
    assert artifact["preview_band"] is True
    assert artifact["VERIFIED"] == 0
    refinement = artifact["placement_trajectory_refinement"]
    provenance = refinement["provenance"]
    assert provenance["preview_band"] is True
    assert provenance["VERIFIED"] == 0
    assert provenance["coordinate_space"] == {
        "world_frame": "court_Z0",
        "typed": "world_court_netcenter_z_up_m",
    }
    assert provenance["config_identity"]["entry_key"] == "body.placement_trajectory_refine"
    assert provenance["config_identity"]["enablement_source"] == "explicit_flag"
    assert provenance["config_identity"]["entry"]["status"] == "PENDING"
    assert set(provenance["inputs"]) == {
        "skeleton3d",
        "tracks",
        "foot_contact_phases",
        "placement",
        "grounding_refinement",
    }
    frame = artifact["players"][0]["frames"][0]["placement_trajectory_refinement"]
    assert len(frame["covariance_m2"]) == 3
    assert set(frame["provenance"]["evidence"]) == {
        "trk",
        "body",
        "plant",
        "smoothness",
        "court_plane",
    }


@pytest.mark.parametrize(
    ("missing", "phase_payload", "reason_code"),
    [
        ("skeleton3d", None, "placement_trajectory_no_body"),
        ("foot_contact_phases", None, "placement_trajectory_no_plant_windows"),
        (None, {"schema_version": 1, "phases": []}, "placement_trajectory_no_plant_windows"),
    ],
)
def test_expected_optional_absence_is_typed_skip(
    tmp_path: Path,
    missing: str | None,
    phase_payload: dict | None,
    reason_code: str,
) -> None:
    pipeline = _pipeline(tmp_path, enabled=True)
    paths = _write_stage_inputs(pipeline)
    if missing is not None:
        paths[missing].unlink()
    if phase_payload is not None:
        _write_json(paths["foot_contact_phases"], phase_payload)

    outcome = pipeline._stage_placement_trajectory_refine()

    assert outcome.status == "skipped"
    assert outcome.metrics["expected_optional_absence"] == {
        "reason_code": reason_code,
        "stage_status": "skipped",
    }
    assert not (pipeline.clip_dir / "placement_trajectory_refined.json").exists()


def test_existing_malformed_schema_fails_loudly_in_spine_wrapper(tmp_path: Path) -> None:
    pipeline = _pipeline(tmp_path, enabled=True)
    paths = _write_stage_inputs(pipeline)
    malformed = copy.deepcopy(_tracks(4))
    malformed["players"][0]["frames"][0]["conf"] = float("nan")
    _write_json(paths["tracks"], malformed)

    outcome = pipeline._run_stage_safely(
        "placement_trajectory_refine",
        pipeline._stage_placement_trajectory_refine,
    )

    assert outcome.status == "failed"
    assert outcome.metrics["reason_code"] == "unexpected_stage_exception"
    assert "MalformedPlacementInputError" in " ".join(outcome.notes)
    assert not (pipeline.clip_dir / "placement_trajectory_refined.json").exists()
    error = json.loads((pipeline.clip_dir / outcome.artifacts[0]).read_text(encoding="utf-8"))
    assert error["exception_type"] == "MalformedPlacementInputError"


def test_stage_identity_covers_inputs_config_output_and_reuses_only_matching_generation(
    tmp_path: Path,
) -> None:
    pipeline = _pipeline(tmp_path, enabled=True)
    paths = _write_stage_inputs(pipeline)
    spec = pipeline._stage_identity_spec(
        "placement_trajectory_refine",
        pipeline._stage_placement_trajectory_refine,
    )

    assert spec.dependencies == ("tracking", "placement", "body", "grounding_refine")
    assert set(spec.explicit_inputs) == {
        "tracks",
        "placement",
        "skeleton3d",
        "foot_contact_phases",
        "grounding_refinement",
    }
    assert spec.config["body.placement_trajectory_refine"]["enabled"] is True
    assert process_video.RUN_IDENTITY_OUTPUTS["placement_trajectory_refine"] == (
        "placement_trajectory_refined.json",
    )

    first = pipeline._run_stage_safely(
        "placement_trajectory_refine",
        pipeline._stage_placement_trajectory_refine,
    )
    second_pipeline = process_video.ProcessVideoPipeline(pipeline.options)
    second = second_pipeline._run_stage_safely(
        "placement_trajectory_refine",
        second_pipeline._stage_placement_trajectory_refine,
    )
    assert first.status == "ran"
    assert second.status == "skipped"
    assert "content-addressed stage generation" in " ".join(second.notes)

    grounding = json.loads(paths["grounding_refinement"].read_text(encoding="utf-8"))
    grounding["status"] = "changed"
    _write_json(paths["grounding_refinement"], grounding)
    third_pipeline = process_video.ProcessVideoPipeline(pipeline.options)
    third = third_pipeline._run_stage_safely(
        "placement_trajectory_refine",
        third_pipeline._stage_placement_trajectory_refine,
    )
    assert third.status == "ran"


def test_best_stack_enablement_is_the_no_flag_default_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video = tmp_path / "clip.mp4"
    monkeypatch.setattr(process_video, "DEFAULT_PLACEMENT_TRAJECTORY_REFINE", True)
    parser = process_video.build_arg_parser()

    options = process_video.build_options_from_args(
        parser.parse_args(["--video", str(video), "--out", str(tmp_path / "run")])
    )

    assert options.placement_trajectory_refine is True

"""NS-04.6 wiring tests for the DEFAULT-OFF `one_world` fusion stage.

Scope note, binding: these tests prove INTEGRATION mechanics -- that the stage is
reachable, default-off, byte-identical when off, typed-degrading when its inputs
are absent, schema-valid when it runs, and correctly registered in the
content-addressed identity graph. They prove NOTHING about the accuracy of the
refinement. `VERIFIED=0` remains binding and no gate has passed.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts.racketsport import process_video
from threed.racketsport.one_world_v1 import OneWorldV1
from tests.racketsport.test_one_world_core import make_run


ROOT = Path(__file__).resolve().parents[2]


def _make_video(path: Path, *, frame_count: int = 3, fps: float = 30.0) -> None:
    pytest.importorskip("cv2")
    import cv2
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (960, 540))
    for _ in range(frame_count):
        writer.write(np.zeros((540, 960, 3), dtype="uint8"))
    writer.release()


def _options(tmp_path: Path, **overrides: object) -> process_video.PipelineOptions:
    video = tmp_path / "clip.mp4"
    if not video.is_file():
        _make_video(video)
    defaults: dict[str, object] = {
        "video": video,
        "clip": "test_clip",
        "run_dir": tmp_path / "run",
        "skip_ball": True,
        "no_gpu": True,
        "vite_allow_root": tmp_path,
    }
    defaults.update(overrides)
    return process_video.PipelineOptions(**defaults)  # type: ignore[arg-type]


# ----------------------------------------------------------------------
# default-OFF byte identity
# ----------------------------------------------------------------------


def test_the_only_fusion_stage_is_the_default_off_preview_one_world_stage() -> None:
    """Honesty anchor for the Defect A rename: once `world` stopped being sold as
    fusion, the single place in the graph that may claim joint refinement is
    `one_world`, and it must be default-OFF."""

    graph = {node.name: node for node in process_video.AUTHORITATIVE_STAGE_GRAPH}
    assert "one_world" in graph
    assert graph["one_world"].enabled_by == "one_world"

    assert "one_world" not in process_video.authoritative_stage_names(
        rally_gating=False, verify_viewer=False
    )
    assert "one_world" in process_video.authoritative_stage_names(
        rally_gating=False, verify_viewer=False, one_world=True
    )


def test_stage_is_default_off_in_the_manifest_and_in_the_options() -> None:
    entry = process_video.BEST_STACK_MANIFEST.entry(process_video.ONE_WORLD_STACK_KEY)
    assert entry.status == "PENDING"
    assert entry.value["enabled"] is False
    assert entry.value["do_not_promote"] is True
    assert entry.raw["trust_band"] == "preview"
    assert entry.gate is not None
    assert process_video.DEFAULT_ONE_WORLD_ENABLED is False
    assert process_video.PipelineOptions.one_world is False


def test_default_projection_is_byte_identical_to_the_pre_wire_stage_list() -> None:
    """The pre-wire graph. Adding the node must not change any default list."""

    expected_serial = (
        "ingest", "calibration", "input_quality", "tracking", "camera_motion",
        "placement", "ball", "ball_arc", "events", "ball_fill", "frames", "body",
        "placement_refine", "grounding_refine", "placement_trajectory_refine",
        "paddle_pose", "events_refined", "ball_arc_refined", "world",
        "confidence_gate", "match_stats", "coaching_facts", "manifest",
    )
    assert process_video.authoritative_stage_names(
        rally_gating=False, verify_viewer=False, player_selection=False
    ) == expected_serial

    # The counts pinned by tests/racketsport/test_spine_stage_contract.py.
    assert len(process_video.authoritative_stage_names(rally_gating=True, verify_viewer=False)) == 25
    assert len(process_video.authoritative_stage_names(rally_gating=False, verify_viewer=True)) == 25
    assert len(process_video.authoritative_stage_names(rally_gating=True, verify_viewer=True)) == 26

    for preset in ("full", "court_skeletons"):
        for schedule in ("serial", "overlap"):
            names = process_video.authoritative_stage_names(
                rally_gating=True,
                verify_viewer=True,
                body_schedule=schedule,
                pipeline_preset=preset,
            )
            assert "one_world" not in names, (preset, schedule)


def test_enabled_stage_sits_at_order_185_between_world_and_confidence_gate() -> None:
    node = next(n for n in process_video.AUTHORITATIVE_STAGE_GRAPH if n.name == "one_world")
    assert (node.serial_order, node.overlap_order, node.enabled_by) == (185, 185, "one_world")

    for schedule in ("serial", "overlap"):
        names = process_video.authoritative_stage_names(
            rally_gating=False, verify_viewer=False, one_world=True, body_schedule=schedule
        )
        index = names.index("one_world")
        assert names[index - 1] == "world"
        assert names[index + 1] == "confidence_gate"


def test_court_skeletons_preset_never_runs_the_fusion_stage() -> None:
    """The preset has no ball stages at all, so it can never satisfy the required
    inputs. Excluding it keeps the 100% of recent runs that used this preset from
    even reaching the blocked path."""

    assert "one_world" not in process_video.COURT_SKELETON_STAGE_NAMES
    names = process_video.authoritative_stage_names(
        rally_gating=False, verify_viewer=False, one_world=True, pipeline_preset="court_skeletons"
    )
    assert "one_world" not in names


def test_default_off_resolved_config_omits_the_key_entirely(tmp_path: Path) -> None:
    """The identity-preserving contract: while off, the fusion selection is absent
    from the resolved best-stack config, so no other stage's fingerprint moves and
    every existing content-addressed generation stays reusable."""

    off = process_video.resolved_best_stack_config_from_options(_options(tmp_path))
    assert process_video.ONE_WORLD_STACK_KEY not in off
    assert not any("one_world" in key for key in off)

    on = process_video.resolved_best_stack_config_from_options(
        _options(tmp_path, one_world=True)
    )
    assert on[process_video.ONE_WORLD_STACK_KEY]["enabled"] is True
    # Enabling changes exactly one key and nothing else.
    assert {k: v for k, v in on.items() if k != process_video.ONE_WORLD_STACK_KEY} == off


def test_enabling_the_stage_does_not_invalidate_any_other_stage_identity(tmp_path: Path) -> None:
    """A stage that perturbs its siblings' fingerprints would force a full
    recompute of unrelated stages. Every other stage's identity spec must be
    identical with the fusion on and off."""

    off_pipeline = process_video.ProcessVideoPipeline(_options(tmp_path / "off"))
    on_pipeline = process_video.ProcessVideoPipeline(_options(tmp_path / "on", one_world=True))

    shared = [
        name
        for name in process_video.authoritative_stage_names(rally_gating=True, verify_viewer=True)
        if name != "one_world"
    ]
    for name in shared:
        fn = getattr(off_pipeline, f"_stage_{name}")
        off_spec = off_pipeline._stage_identity_spec(name, fn)
        on_spec = on_pipeline._stage_identity_spec(name, getattr(on_pipeline, f"_stage_{name}"))
        assert off_spec.config == on_spec.config, name
        assert off_spec.code == on_spec.code, name
        assert off_spec.dependencies == on_spec.dependencies, name


# ----------------------------------------------------------------------
# identity-graph registration (NS-01.3)
# ----------------------------------------------------------------------


def test_stage_is_registered_in_the_identity_graph() -> None:
    """A stage missing from the identity graph silently reuses stale output."""

    assert "one_world" in process_video.RUN_IDENTITY_DEPENDENCIES
    assert "one_world" in process_video.RUN_IDENTITY_CONFIG_KEYS
    assert "one_world" in process_video.RUN_IDENTITY_OUTPUTS

    deps = process_video.RUN_IDENTITY_DEPENDENCIES["one_world"]
    # It reads the composited world plus the raw evidence the compositor read.
    assert "world" in deps
    assert "ball" in deps
    assert "calibration" in deps

    assert process_video.RUN_IDENTITY_CONFIG_KEYS["one_world"] == (
        process_video.ONE_WORLD_STACK_KEY,
    )
    assert process_video.RUN_IDENTITY_OUTPUTS["one_world"] == (
        process_video.ONE_WORLD_ARTIFACT_NAME,
        process_video.ONE_WORLD_VALIDATION_ARTIFACT_NAME,
    )


def test_nothing_downstream_depends_on_the_fusion(tmp_path: Path) -> None:
    """The fusion must never become product authority. If a downstream stage took
    a dependency on it, an enabled one_world would change the shipped bundle."""

    for stage, deps in process_video.RUN_IDENTITY_DEPENDENCIES.items():
        assert "one_world" not in deps, f"{stage} must not depend on the preview fusion"


def test_identity_spec_tracks_the_stage_inputs_and_enablement(tmp_path: Path) -> None:
    pipeline = process_video.ProcessVideoPipeline(_options(tmp_path, one_world=True))
    spec = pipeline._stage_identity_spec("one_world", pipeline._stage_one_world)

    assert spec.config[process_video.ONE_WORLD_STACK_KEY]["enabled"] is True
    assert spec.config["enabled"] is True
    assert spec.config["enablement_source"] == "best_stack"

    explicit = pipeline._stage_explicit_inputs("one_world")
    for required in process_video.ONE_WORLD_REQUIRED_INPUTS:
        assert required in explicit
    assert "virtual_world.json" in explicit


def test_explicit_flag_is_recorded_as_the_enablement_source(tmp_path: Path) -> None:
    pipeline = process_video.ProcessVideoPipeline(
        _options(tmp_path, one_world=True, one_world_explicit=True)
    )
    options = pipeline._stage_identity_options("one_world")
    assert options["enablement_source"] == "explicit_flag"


# ----------------------------------------------------------------------
# typed degradation on absent inputs
# ----------------------------------------------------------------------


@pytest.mark.parametrize("removed", ["ball_track.json", "court_calibration.json"])
def test_absent_required_input_blocks_typed_and_never_crashes(tmp_path: Path, removed: str) -> None:
    """`one_world_v1.build_one_world` hard-raises FileNotFoundError without
    ball_track.json. The stage must pre-flight that into a typed blocked outcome
    naming the missing file, and must not write any artifact."""

    pipeline = process_video.ProcessVideoPipeline(_options(tmp_path, one_world=True))
    make_run(pipeline.clip_dir)
    (pipeline.clip_dir / removed).unlink()

    outcome = pipeline._run_stage_safely("one_world", pipeline._stage_one_world)

    assert outcome.status == "blocked"
    assert outcome.stage == "one_world"
    assert outcome.metrics["expected_optional_absence"] == {
        "reason_code": "one_world_required_inputs_missing",
        "stage_status": "blocked",
    }
    assert removed in outcome.notes[0]
    assert not (pipeline.clip_dir / process_video.ONE_WORLD_ARTIFACT_NAME).exists()
    assert not (pipeline.clip_dir / process_video.ONE_WORLD_VALIDATION_ARTIFACT_NAME).exists()


def test_completely_empty_clip_dir_blocks_instead_of_raising(tmp_path: Path) -> None:
    pipeline = process_video.ProcessVideoPipeline(_options(tmp_path, one_world=True))
    outcome = pipeline._run_stage_safely("one_world", pipeline._stage_one_world)
    assert outcome.status == "blocked"
    for required in process_video.ONE_WORLD_REQUIRED_INPUTS:
        assert required in outcome.notes[0]


def test_inconsistent_same_run_inputs_degrade_instead_of_failing_the_run(tmp_path: Path) -> None:
    """Real disagreement between same-run artifacts must not take down the
    authoritative bundle: this stage is never authority."""

    pipeline = process_video.ProcessVideoPipeline(_options(tmp_path, one_world=True))
    make_run(pipeline.clip_dir)
    calibration = json.loads((pipeline.clip_dir / "court_calibration.json").read_text())
    calibration["coordinate_frame"] = "some_other_frame"
    (pipeline.clip_dir / "court_calibration.json").write_text(json.dumps(calibration))

    outcome = pipeline._run_stage_safely("one_world", pipeline._stage_one_world)

    assert outcome.status == "degraded"
    assert outcome.metrics["expected_optional_absence"]["reason_code"] == "one_world_input_disagreement"
    assert not (pipeline.clip_dir / process_video.ONE_WORLD_ARTIFACT_NAME).exists()


# ----------------------------------------------------------------------
# stage-on behaviour against a real fixture
# ----------------------------------------------------------------------


def test_enabled_stage_emits_schema_valid_preview_artifacts(tmp_path: Path) -> None:
    pipeline = process_video.ProcessVideoPipeline(_options(tmp_path, one_world=True))
    make_run(pipeline.clip_dir)

    outcome = pipeline._run_stage_safely("one_world", pipeline._stage_one_world)

    assert outcome.status == "ran", outcome.notes
    assert outcome.trust_badge == "preview"
    assert outcome.artifacts == [
        process_video.ONE_WORLD_ARTIFACT_NAME,
        process_video.ONE_WORLD_VALIDATION_ARTIFACT_NAME,
    ]
    assert outcome.metrics["VERIFIED"] == 0
    assert outcome.metrics["preview_only"] is True
    assert outcome.metrics["do_not_promote"] is True
    assert outcome.metrics["authority"] == "never"
    assert outcome.metrics["validation_valid"] is True

    artifact_path = pipeline.clip_dir / process_video.ONE_WORLD_ARTIFACT_NAME
    payload = json.loads(artifact_path.read_text())
    assert payload["VERIFIED"] == 0
    assert payload["preview_only"] is True
    assert payload["render_only"] is True
    assert payload["not_for_training"] is True
    assert payload["raw_inputs_mutated"] is False
    assert payload["trust_band"]["badge"] != "verified"

    validation = json.loads(
        (pipeline.clip_dir / process_video.ONE_WORLD_VALIDATION_ARTIFACT_NAME).read_text()
    )
    assert validation["valid"] is True


def test_emitted_artifact_validates_against_the_committed_public_schema(tmp_path: Path) -> None:
    """`docs/racketsport/one_world_v1_schema.json` is the published contract. It is
    generated from OneWorldV1, so pin that equality and then validate the exact
    emitted bytes through the model."""

    committed = json.loads(
        (ROOT / "docs" / "racketsport" / "one_world_v1_schema.json").read_text(encoding="utf-8")
    )
    assert OneWorldV1.model_json_schema() == committed, (
        "committed one_world_v1 schema drifted from the OneWorldV1 model"
    )

    pipeline = process_video.ProcessVideoPipeline(_options(tmp_path, one_world=True))
    make_run(pipeline.clip_dir)
    pipeline._run_stage_safely("one_world", pipeline._stage_one_world)

    raw = (pipeline.clip_dir / process_video.ONE_WORLD_ARTIFACT_NAME).read_text(encoding="utf-8")
    artifact = OneWorldV1.model_validate_json(raw)
    assert artifact.artifact_type == "racketsport_one_world_v1"
    assert set(json.loads(raw)) == set(committed["required"])


def test_fusion_never_mutates_a_raw_observation(tmp_path: Path) -> None:
    """Standing rule: raw observations are immutable; refinements are separate
    artifacts with their own provenance."""

    pipeline = process_video.ProcessVideoPipeline(_options(tmp_path, one_world=True))
    make_run(pipeline.clip_dir)
    before = {
        path.name: path.read_bytes()
        for path in sorted(pipeline.clip_dir.iterdir())
        if path.is_file()
    }

    pipeline._run_stage_safely("one_world", pipeline._stage_one_world)

    after = {
        path.name: path.read_bytes()
        for path in sorted(pipeline.clip_dir.iterdir())
        if path.is_file()
    }
    for name, payload in before.items():
        assert after[name] == payload, f"{name} was mutated by the fusion stage"
    assert set(after) - set(before) == {
        process_video.ONE_WORLD_ARTIFACT_NAME,
        process_video.ONE_WORLD_VALIDATION_ARTIFACT_NAME,
    }


def test_fusion_output_is_not_credited_to_any_bundle_capability(tmp_path: Path) -> None:
    """The preview fusion must not make a bundle look more complete than it is."""

    clip_dir = tmp_path / "clip"
    clip_dir.mkdir()
    make_run(clip_dir)
    (clip_dir / process_video.ONE_WORLD_ARTIFACT_NAME).write_text("{}", encoding="utf-8")

    missing = process_video._minimum_bundle_missing_capabilities(
        clip_dir=clip_dir, stage_outcomes=[], video_identity=None, pipeline_preset="full"
    )
    names = {item["capability"] for item in missing}
    # virtual_world.json exists in the fixture, so composited_world is satisfied;
    # nothing else may be satisfied by the fusion artifact.
    assert "composited_world" not in names
    assert "ball" not in names  # ball_track.json exists in the fixture
    for capability in ("body", "paddle", "stats", "coaching", "assets", "manifest"):
        assert capability in names, capability


def test_stage_is_deterministic_across_repeat_runs(tmp_path: Path) -> None:
    pipeline = process_video.ProcessVideoPipeline(_options(tmp_path, one_world=True))
    make_run(pipeline.clip_dir)
    artifact_path = pipeline.clip_dir / process_video.ONE_WORLD_ARTIFACT_NAME

    pipeline._stage_one_world()
    first = artifact_path.read_bytes()
    pipeline._stage_one_world()
    assert artifact_path.read_bytes() == first

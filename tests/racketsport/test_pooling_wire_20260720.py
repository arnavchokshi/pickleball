from __future__ import annotations

import bz2
from dataclasses import replace
import hashlib
import inspect
import json
from pathlib import Path
from typing import Any

import pytest

from scripts.racketsport import process_video
from threed.racketsport.court_line_evidence import (
    aggregate_court_line_evidence,
    required_court_line_ids,
    required_court_net_ids,
    select_best_line_observation,
)
from threed.racketsport.court_line_robustness import (
    AssignedCourtLine,
    CourtLineHardeningResult,
    FrameCourtLineEvidence,
    SeedGuidedPaintSample,
    canonical_json_bytes,
    combine_pooled_static_court_line_evidence,
    load_pooled_court_line_evidence_artifact,
    pool_static_semantic_lines,
    proven_court_line_pool_config,
)
from threed.racketsport.schemas import CourtLineEvidence


ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DIR = (
    ROOT / "tests/racketsport/fixtures/pooling_wire_20260720"
)
DRILL_CALIBRATION = FIXTURE_DIR / "drill_court_calibration.json"
DRILL_BASELINE_EVIDENCE = (
    FIXTURE_DIR / "drill_court_line_evidence.json"
)
DRILL_COURT_ZONES = FIXTURE_DIR / "drill_court_zones.json"
DRILL_NET_PLANE = FIXTURE_DIR / "drill_net_plane.json"
DRILL_RAW_POOL_FIXTURE = (
    FIXTURE_DIR / "drill_compact_raw_frame_evidence.json.bz2"
)
DRILL_LEGACY_BASELINE_FIXTURE = (
    FIXTURE_DIR / "drill_legacy_baseline_96.json"
)
DRILL_VIDEO = FIXTURE_DIR / "intentionally_absent_no_decode.mp4"
DRILL_RAW_POOL_SHA256 = (
    "5426dacfce9fe58ca318260c7b9e071fa55d9e45ddf6a41fcc2086ff44a33afe"
)
DRILL_COMPACT_POOL_FIXTURE_SHA256 = (
    "c7fe629058ddbde3f167cba6c350e8dadad29353e4bc576962b15d00ad007fe4"
)
DRILL_BASELINE_EVIDENCE_SHA256 = (
    "ca6c42caac1646d8f5e52b291a0eec506cb984d154010db8c24e22753d892c3e"
)
DRILL_CALIBRATION_SHA256 = (
    "1cabc3d0affad63c5f7e6a90d89dbff0151ea524734b3b895b3a899e6fb2419a"
)
DRILL_LEGACY_BASELINE_FIXTURE_SHA256 = (
    "1278f15fa77134c2a5e7a09cab5150d0c4fa72e439c9a21b0c2681df6b611210"
)


def _load_compact_drill_frames(
    payload: dict[str, Any],
    *,
    seed_calibration: dict[str, Any],
    config: Any,
) -> tuple[FrameCourtLineEvidence, ...]:
    """Expand the lossless pooling fields from the portable Drill fixture."""

    assert payload["artifact_type"] == (
        "pooling_wire_drill_compact_raw_fixture"
    )
    assert payload["seed_calibration_sha256"] == hashlib.sha256(
        canonical_json_bytes(seed_calibration)
    ).hexdigest()
    assert payload["evidence_config_sha256"] == hashlib.sha256(
        canonical_json_bytes(config.evidence_config_dict())
    ).hexdigest()
    line_ids = [str(value) for value in payload["line_ids"]]
    expected_segments = [
        (
            (float(segment[0][0]), float(segment[0][1])),
            (float(segment[1][0]), float(segment[1][1])),
        )
        for segment in payload["expected_segments"]
    ]
    sample_basis = {
        (int(row[0]), int(row[1])): row
        for row in payload["sample_basis"]
    }
    frames: list[FrameCourtLineEvidence] = []
    for frame_row in payload["frames"]:
        frame_index = int(frame_row[0])
        assigned_line_indexes = [int(value) for value in frame_row[2]]
        assignments = tuple(
            AssignedCourtLine(
                line_id=line_ids[line_index],
                candidate_id=(
                    "seed_guided_paired_edges:"
                    f"{line_ids[line_index]}"
                ),
                segment=expected_segments[line_index],
                expected_segment=expected_segments[line_index],
                score=1.0,
                normal_distance_px=0.0,
                angle_delta_deg=0.0,
                overlap_fraction=1.0,
                support_length_px=1.0,
                selection_margin=None,
            )
            for line_index in assigned_line_indexes
        )
        samples: list[SeedGuidedPaintSample] = []
        for sample_row in frame_row[3]:
            line_index = int(sample_row[0])
            sample_index = int(sample_row[1])
            basis = sample_basis[(line_index, sample_index)]
            signed_offset = float(sample_row[2])
            seed_xy = (float(basis[6]), float(basis[7]))
            normal_xy = (float(basis[8]), float(basis[9]))
            samples.append(
                SeedGuidedPaintSample(
                    line_id=line_ids[line_index],
                    sample_index=sample_index,
                    t=float(basis[2]),
                    world_xyz_m=(
                        float(basis[3]),
                        float(basis[4]),
                        float(basis[5]),
                    ),
                    seed_xy=seed_xy,
                    normal_xy=normal_xy,
                    observed_xy=(
                        seed_xy[0] + signed_offset * normal_xy[0],
                        seed_xy[1] + signed_offset * normal_xy[1],
                    ),
                    signed_offset_px=signed_offset,
                    expected_width_px=1.0,
                    band_width_px=float(sample_row[3]),
                    contrast=float(sample_row[4]),
                    edge_strength=float(sample_row[5]),
                    edge_symmetry=1.0,
                    selection_rank=0.0,
                )
            )
        frames.append(
            FrameCourtLineEvidence(
                frame_index=frame_index,
                frame_sha256=str(frame_row[1]),
                image_size=(
                    int(payload["image_size"][0]),
                    int(payload["image_size"][1]),
                ),
                coordinate_space=str(payload["coordinate_space"]),
                distortion_state=str(payload["distortion_state"]),
                provider=config.provider,
                raw_candidates=(),
                assignments=assignments,
                status="accepted" if assignments else "abstained",
                rejection_reasons=(),
                roi_polygon_px=(),
                roi_source=str(payload["roi_source"]),
                template_samples=tuple(samples),
                seed_calibration_sha256=str(
                    payload["seed_calibration_sha256"]
                ),
                template_projection_sha256=str(
                    payload["template_projection_sha256"]
                ),
            )
        )
    return tuple(frames)


@pytest.fixture(scope="module")
def drill_fixture() -> dict[str, Any]:
    calibration = json.loads(DRILL_CALIBRATION.read_text(encoding="utf-8"))
    baseline = CourtLineEvidence.model_validate_json(
        DRILL_BASELINE_EVIDENCE.read_text(encoding="utf-8")
    )
    compressed_bytes = DRILL_RAW_POOL_FIXTURE.read_bytes()
    assert (
        hashlib.sha256(compressed_bytes).hexdigest()
        == DRILL_COMPACT_POOL_FIXTURE_SHA256
    )
    raw_payload = json.loads(bz2.decompress(compressed_bytes))
    assert raw_payload["source_raw_sha256"] == DRILL_RAW_POOL_SHA256
    config = proven_court_line_pool_config()
    frames = _load_compact_drill_frames(
        raw_payload,
        seed_calibration=calibration,
        config=config,
    )
    return {
        "calibration": calibration,
        "baseline": baseline,
        "seed_calibration_sha256": raw_payload[
            "seed_calibration_sha256"
        ],
        "frames": frames,
        "config": config,
    }


def test_cli_and_stage_identity_are_default_off(tmp_path: Path) -> None:
    parser = process_video.build_arg_parser()
    default_args = parser.parse_args(
        ["--video", str(DRILL_VIDEO), "--out", str(tmp_path / "off")]
    )
    enabled_args = parser.parse_args(
        [
            "--video",
            str(DRILL_VIDEO),
            "--out",
            str(tmp_path / "on"),
            "--court-line-evidence-pooling",
        ]
    )
    default_options = process_video.build_options_from_args(default_args)
    enabled_options = process_video.build_options_from_args(enabled_args)

    assert default_options.court_line_evidence_pooling is False
    assert enabled_options.court_line_evidence_pooling is True
    default_pipeline = process_video.ProcessVideoPipeline(default_options)
    enabled_pipeline = process_video.ProcessVideoPipeline(enabled_options)
    default_identity = default_pipeline._stage_identity_options(
        "calibration"
    )
    enabled_identity = enabled_pipeline._stage_identity_options(
        "calibration"
    )
    assert "court_line_evidence_pooling" not in default_identity
    assert enabled_identity["court_line_evidence_pooling"]["enabled"] is True
    assert enabled_identity["court_line_evidence_pooling"]["sample_count"] == 96
    assert set(
        enabled_identity["court_line_evidence_pooling"][
            "process_seam_sha256"
        ]
    ) == {
        "pool",
        "readiness",
        "pretracking_gate",
        "input_quality",
    }
    assert set(
        enabled_identity["court_line_evidence_pooling"]["dependency_sha256"]
    ) == {
        "court_auto_evidence",
        "court_calibration",
        "court_keypoint_net",
        "court_line_evidence",
        "court_line_keypoints",
        "court_proposal_optimizer",
        "court_templates",
        "schemas",
    }
    assert len(
        enabled_identity["court_line_evidence_pooling"][
            "process_video_sha256"
        ]
    ) == 64
    for stage, legacy_sha256 in (
        process_video.COURT_LINE_POOL_DEFAULT_OFF_CALLABLE_SHA256.items()
    ):
        default_spec = default_pipeline._stage_identity_spec(
            stage,
            getattr(default_pipeline, f"_stage_{stage}"),
        )
        enabled_spec = enabled_pipeline._stage_identity_spec(
            stage,
            getattr(enabled_pipeline, f"_stage_{stage}"),
        )
        assert default_spec.code == {
            "callable_sha256": legacy_sha256,
        }
        assert enabled_spec.code["callable_sha256"] != legacy_sha256


def test_default_off_identity_translation_requires_reviewed_callable_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def changed_stage_calibration(
        _pipeline: process_video.ProcessVideoPipeline,
    ) -> process_video.StageOutcome:
        raise AssertionError("identity construction must not invoke the stage")

    changed_sha256 = hashlib.sha256(
        inspect.getsource(changed_stage_calibration).encode("utf-8")
    ).hexdigest()
    assert changed_sha256 != (
        process_video.COURT_LINE_POOL_REVIEWED_POST_WIRE_CALLABLE_SHA256[
            "calibration"
        ]
    )
    monkeypatch.setattr(
        process_video.ProcessVideoPipeline,
        "_stage_calibration",
        changed_stage_calibration,
    )
    pipeline = process_video.ProcessVideoPipeline(
        process_video.PipelineOptions(
            video=DRILL_VIDEO,
            clip="drill",
            run_dir=tmp_path,
        )
    )

    spec = pipeline._stage_identity_spec(
        "calibration",
        pipeline._stage_calibration,
    )

    assert spec.code == {"callable_sha256": changed_sha256}
    assert spec.code["callable_sha256"] != (
        process_video.COURT_LINE_POOL_DEFAULT_OFF_CALLABLE_SHA256[
            "calibration"
        ]
    )


def test_default_off_calibration_reuse_is_byte_identical_golden(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    options = process_video.PipelineOptions(
        video=DRILL_VIDEO,
        clip="drill",
        run_dir=tmp_path,
    )
    pipeline = process_video.ProcessVideoPipeline(options)
    calibration_path = pipeline.clip_dir / "court_calibration.json"
    evidence_path = pipeline.clip_dir / "court_line_evidence.json"
    calibration_bytes = DRILL_CALIBRATION.read_bytes()
    evidence_bytes = DRILL_BASELINE_EVIDENCE.read_bytes()
    calibration_path.write_bytes(calibration_bytes)
    evidence_path.write_bytes(evidence_bytes)

    def forbidden(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("default-OFF path invoked court-line pooling")

    monkeypatch.setattr(
        "threed.racketsport.court_line_robustness.run_proven_court_line_pool_from_video",
        forbidden,
    )
    outcome = pipeline._stage_calibration()

    assert outcome.as_dict() == {
        "stage": "calibration",
        "status": "skipped",
        "wall_seconds": 0.0,
        "notes": ["reusing existing valid court_calibration.json"],
        "artifacts": ["court_calibration.json"],
        "trust_badge": outcome.trust_badge,
        "metrics": {},
    }
    assert calibration_path.read_bytes() == calibration_bytes
    assert evidence_path.read_bytes() == evidence_bytes
    assert not (
        pipeline.clip_dir / process_video.COURT_LINE_POOL_RAW_FRAMES_NAME
    ).exists()
    assert not (
        pipeline.clip_dir / process_video.COURT_LINE_POOLED_NAME
    ).exists()


def test_default_off_cold_external_calibration_matches_frozen_golden(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_artifact_bytes = {
        "court_calibration.json": DRILL_CALIBRATION.read_bytes(),
        "court_zones.json": DRILL_COURT_ZONES.read_bytes(),
        "net_plane.json": DRILL_NET_PLANE.read_bytes(),
        "court_line_evidence.json": DRILL_BASELINE_EVIDENCE.read_bytes(),
    }

    def fake_calibration_spine(**kwargs: Any) -> dict[str, Any]:
        assert kwargs["stage"] == "calibration"
        run_dir = Path(kwargs["run_dir"])
        for name, artifact_bytes in expected_artifact_bytes.items():
            (run_dir / name).write_bytes(artifact_bytes)
        return {"status": process_video.orchestrator.PIPELINE_STATUS_PASS}

    def forbidden(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("cold default-OFF path invoked pooling")

    monkeypatch.setattr(
        process_video.orchestrator,
        "run_pipeline",
        fake_calibration_spine,
    )
    monkeypatch.setattr(
        "threed.racketsport.court_line_robustness.run_proven_court_line_pool_from_video",
        forbidden,
    )
    pipeline = process_video.ProcessVideoPipeline(
        process_video.PipelineOptions(
            video=DRILL_VIDEO,
            clip="drill",
            run_dir=tmp_path,
            court_calibration=DRILL_CALIBRATION,
        )
    )

    outcome = pipeline._stage_calibration()

    assert outcome.status == "ran"
    assert outcome.artifacts == list(expected_artifact_bytes)
    assert "court_line_pooling" not in outcome.metrics
    assert {
        path.name: path.read_bytes()
        for path in pipeline.clip_dir.iterdir()
        if path.is_file()
    } == expected_artifact_bytes
    assert not (
        pipeline.clip_dir / process_video.COURT_LINE_POOL_RAW_FRAMES_NAME
    ).exists()
    assert not (
        pipeline.clip_dir / process_video.COURT_LINE_POOLED_NAME
    ).exists()
    normalized_outcome = outcome.as_dict()
    normalized_outcome["notes"] = [
        note.replace(str(DRILL_CALIBRATION), "<DRILL_CALIBRATION>")
        for note in normalized_outcome["notes"]
    ]
    assert normalized_outcome == {
        "stage": "calibration",
        "status": "ran",
        "wall_seconds": 0.0,
        "notes": [
            (
                "consumed externally-provided court calibration from "
                "<DRILL_CALIBRATION> "
                "(intrinsics.source='estimated_from_declared_court_corners'); "
                "schema + intrinsics.source validated "
                "(threed.racketsport.orchestrator.ExternalCalibrationRunner), "
                "PnP re-derivation from --court-corners/--capture-sidecar "
                "skipped"
            )
        ],
        "artifacts": list(expected_artifact_bytes),
        "trust_badge": "low_confidence",
        "metrics": {
            "reprojection_median_px": 0.0,
            "reprojection_p95_px": 0.0,
            "intrinsics_source": "estimated_from_declared_court_corners",
            "intrinsics_dist_nonzero": False,
        },
    }


def test_default_off_ignores_and_preserves_immutable_pool_sidecars(
    tmp_path: Path,
) -> None:
    options = process_video.PipelineOptions(
        video=DRILL_VIDEO,
        clip="drill",
        run_dir=tmp_path,
    )
    pipeline = process_video.ProcessVideoPipeline(options)
    calibration_path = pipeline.clip_dir / "court_calibration.json"
    evidence_path = pipeline.clip_dir / "court_line_evidence.json"
    calibration_path.write_bytes(DRILL_CALIBRATION.read_bytes())
    evidence_bytes = DRILL_BASELINE_EVIDENCE.read_bytes()
    evidence_path.write_bytes(evidence_bytes)
    raw_sidecar = (
        pipeline.clip_dir / process_video.COURT_LINE_POOL_RAW_FRAMES_NAME
    )
    pooled_sidecar = pipeline.clip_dir / process_video.COURT_LINE_POOLED_NAME
    raw_sidecar.write_text('{"stale": true}\n', encoding="utf-8")
    pooled_sidecar.write_text('{"stale": true}\n', encoding="utf-8")

    outcome = pipeline._stage_calibration()

    assert outcome.status == "skipped"
    assert evidence_path.read_bytes() == evidence_bytes
    assert raw_sidecar.read_bytes() == b'{"stale": true}\n'
    assert pooled_sidecar.read_bytes() == b'{"stale": true}\n'


def test_default_off_reuse_does_not_parse_malformed_evidence(
    tmp_path: Path,
) -> None:
    pipeline = process_video.ProcessVideoPipeline(
        process_video.PipelineOptions(
            video=DRILL_VIDEO,
            clip="drill",
            run_dir=tmp_path,
        )
    )
    calibration_path = pipeline.clip_dir / "court_calibration.json"
    evidence_path = pipeline.clip_dir / "court_line_evidence.json"
    calibration_path.write_bytes(DRILL_CALIBRATION.read_bytes())
    malformed_bytes = b"{malformed raw evidence must remain unread}\\n"
    evidence_path.write_bytes(malformed_bytes)

    outcome = pipeline._stage_calibration()

    assert outcome.status == "skipped"
    assert evidence_path.read_bytes() == malformed_bytes


def test_drill_raw_fixture_replays_exact_pool_and_unchanged_readiness_bar(
    drill_fixture: dict[str, Any],
) -> None:
    frames = drill_fixture["frames"]
    config = drill_fixture["config"]
    calibration = drill_fixture["calibration"]
    baseline = drill_fixture["baseline"]
    baseline_before = canonical_json_bytes(baseline.model_dump(mode="json"))
    assert hashlib.sha256(DRILL_RAW_POOL_FIXTURE.read_bytes()).hexdigest() == (
        DRILL_COMPACT_POOL_FIXTURE_SHA256
    )
    assert hashlib.sha256(DRILL_BASELINE_EVIDENCE.read_bytes()).hexdigest() == (
        DRILL_BASELINE_EVIDENCE_SHA256
    )
    assert hashlib.sha256(DRILL_CALIBRATION.read_bytes()).hexdigest() == (
        DRILL_CALIBRATION_SHA256
    )
    legacy_baseline_bytes = DRILL_LEGACY_BASELINE_FIXTURE.read_bytes()
    assert hashlib.sha256(legacy_baseline_bytes).hexdigest() == (
        DRILL_LEGACY_BASELINE_FIXTURE_SHA256
    )
    legacy_baseline = json.loads(legacy_baseline_bytes)
    assert legacy_baseline["artifact_type"] == (
        "pooling_wire_drill_legacy_baseline_fixture"
    )
    assert legacy_baseline["source_frozen_baseline_sha256"] == (
        "9f425f2d0b80291dfe68e5ce93680b80df3cbbbed377f8cef9cc74f71d338af5"
    )
    legacy_rows = legacy_baseline["frames"]
    assert [int(row[0]) for row in legacy_rows] == [
        frame.frame_index for frame in frames
    ]
    legacy_far_support_frames = sum(bool(row[1]) for row in legacy_rows)
    assert (legacy_far_support_frames, len(legacy_rows)) == (4, 96)

    first = pool_static_semantic_lines(frames, config=config)
    second = pool_static_semantic_lines(
        list(reversed(frames)),
        config=config,
    )

    assert first.canonical_bytes() == second.canonical_bytes()
    assert (
        load_pooled_court_line_evidence_artifact(
            first.as_dict()
        ).canonical_bytes()
        == first.canonical_bytes()
    )
    assert first.status == "accepted"
    assert first.static_consistency is not None
    assert first.static_consistency.status == "accepted"
    assert first.static_consistency.dispersion_bound_px == 3.0
    far = next(
        line for line in first.lines if line.line_id == "far_centerline"
    )
    assert (
        len(
            set(far.contributing_frame_indexes)
            | set(far.heldout_frame_indexes)
        )
        == 63
    )
    assert far.geometry_fit_p90_px == pytest.approx(
        0.35691255314973347,
        abs=1e-12,
    )
    assert far.as_dict()["source"] == "pooled_static"

    readiness = combine_pooled_static_court_line_evidence(
        baseline,
        first,
        seed_calibration=calibration,
        config=config,
    )
    assert canonical_json_bytes(baseline.model_dump(mode="json")) == baseline_before
    assert readiness.status == "accepted"
    assert [line.line_id for line in readiness.added_line_observations] == [
        "far_centerline"
    ]
    added = readiness.added_line_observations[0]
    assert added.source == "pooled_static"
    assert added.confidence == pytest.approx(0.9675323384, abs=1e-9)
    assert added.residual_px.mean == pytest.approx(
        1.483691,
        abs=2e-6,
    )
    assert added.residual_px.p95 == pytest.approx(
        1.702454,
        abs=5e-6,
    )
    aggregate = readiness.effective_evidence.aggregate
    assert aggregate.auto_calibration_ready is True
    assert aggregate.missing_required_line_ids == []
    assert aggregate.missing_required_net_ids == []
    assert aggregate.mean_residual_px == pytest.approx(
        3.6498698057003773,
        abs=1e-12,
    )
    assert aggregate.p95_residual_px == pytest.approx(
        10.253094225879789,
        abs=1e-12,
    )

    selector_signature = inspect.signature(select_best_line_observation)
    assert selector_signature.parameters["min_confidence"].default == 0.5
    assert selector_signature.parameters["max_distance_px"].default == 24.0
    assert (
        selector_signature.parameters["min_visible_fraction"].default == 0.2
    )
    aggregate_signature = inspect.signature(aggregate_court_line_evidence)
    assert aggregate_signature.parameters["min_line_confidence"].default == 0.5
    assert aggregate_signature.parameters["min_net_confidence"].default == 0.5
    assert aggregate_signature.parameters["max_mean_residual_px"].default == 8.0
    assert aggregate_signature.parameters["max_p95_residual_px"].default == 16.0
    assert required_court_line_ids("pickleball") == (
        "near_nvz",
        "far_nvz",
        "near_centerline",
        "far_centerline",
    )
    assert required_court_net_ids("pickleball") == ("top_net",)


def _moving_frame(
    frame: FrameCourtLineEvidence,
    *,
    coherent_offset_px: float,
) -> FrameCourtLineEvidence:
    shifted_samples = []
    for sample in frame.template_samples:
        shifted_offset = sample.signed_offset_px + coherent_offset_px
        shifted_samples.append(
            replace(
                sample,
                signed_offset_px=shifted_offset,
                observed_xy=(
                    sample.seed_xy[0]
                    + shifted_offset * sample.normal_xy[0],
                    sample.seed_xy[1]
                    + shifted_offset * sample.normal_xy[1],
                ),
            )
        )
    return replace(frame, template_samples=tuple(shifted_samples))


def test_moving_camera_fixture_abstains_as_a_whole(
    drill_fixture: dict[str, Any],
) -> None:
    frames = drill_fixture["frames"]
    moving = tuple(
        _moving_frame(
            frame,
            coherent_offset_px=12.0 * position / (len(frames) - 1),
        )
        for position, frame in enumerate(frames)
    )

    pooled = pool_static_semantic_lines(
        moving,
        config=drill_fixture["config"],
    )

    assert pooled.status == "abstained"
    assert pooled.static_consistency is not None
    assert pooled.static_consistency.status == "abstained"
    assert pooled.static_consistency.max_observed_mad_px is not None
    assert pooled.static_consistency.max_observed_mad_px > 3.0
    assert pooled.static_consistency.violating_measurements
    assert any(
        reason.startswith("static_consistency_drift_exceeds_bound:")
        for reason in pooled.rejection_reasons
    )
    readiness = combine_pooled_static_court_line_evidence(
        drill_fixture["baseline"],
        pooled,
        seed_calibration=drill_fixture["calibration"],
        config=drill_fixture["config"],
    )
    assert readiness.status == "abstained"
    assert readiness.added_line_observations == ()
    assert readiness.effective_evidence.aggregate.auto_calibration_ready is False


@pytest.mark.parametrize("tail_frame_count", [1, 2, 8, 16, 20])
def test_sustained_tail_pan_abstains_at_the_same_dispersion_bound(
    drill_fixture: dict[str, Any],
    tail_frame_count: int,
) -> None:
    frames = drill_fixture["frames"]
    moving = tuple(
        _moving_frame(
            frame,
            coherent_offset_px=(
                12.0
                if position >= len(frames) - tail_frame_count
                else 0.0
            ),
        )
        for position, frame in enumerate(frames)
    )

    pooled = pool_static_semantic_lines(
        moving,
        config=drill_fixture["config"],
    )

    assert pooled.status == "abstained"
    guard = pooled.static_consistency
    assert guard is not None
    assert guard.status == "abstained"
    assert guard.dispersion_bound_px == 3.0
    assert guard.max_observed_temporal_span_px is not None
    assert guard.max_observed_temporal_span_px > 3.0
    assert any(
        reason.startswith(
            "static_consistency_temporal_shift_exceeds_bound:"
        )
        for reason in guard.rejection_reasons
    )


@pytest.mark.parametrize("tail_frame_count", [1, 2])
@pytest.mark.parametrize("retained_line_count", [1, 2, 3])
def test_partial_visibility_tail_pan_abstains_typed(
    drill_fixture: dict[str, Any],
    tail_frame_count: int,
    retained_line_count: int,
) -> None:
    frames = drill_fixture["frames"]
    moving: list[FrameCourtLineEvidence] = []
    for position, frame in enumerate(frames):
        if position < len(frames) - tail_frame_count:
            moving.append(frame)
            continue
        shifted = _moving_frame(frame, coherent_offset_px=12.0)
        retained_assignments = shifted.assignments[:retained_line_count]
        retained_line_ids = {
            assignment.line_id for assignment in retained_assignments
        }
        moving.append(
            replace(
                shifted,
                assignments=retained_assignments,
                template_samples=tuple(
                    sample
                    for sample in shifted.template_samples
                    if sample.line_id in retained_line_ids
                ),
            )
        )

    pooled = pool_static_semantic_lines(
        moving,
        config=drill_fixture["config"],
    )

    assert pooled.status == "abstained"
    guard = pooled.static_consistency
    assert guard is not None
    assert guard.status == "abstained"
    assert guard.violating_boundary_degraded_frames == (
        (
            len(frames) - 1,
            frames[-1].frame_index,
            retained_line_count,
        ),
    )
    assert any(
        reason.startswith(
            "static_consistency_boundary_assignment_below_consensus:"
        )
        for reason in guard.rejection_reasons
    )


@pytest.mark.parametrize("frame_position", [1, 45, 94])
@pytest.mark.parametrize("retained_line_count", [1, 2, 3])
def test_partial_visibility_internal_pan_is_still_measured(
    drill_fixture: dict[str, Any],
    frame_position: int,
    retained_line_count: int,
) -> None:
    frames = drill_fixture["frames"]
    moving = list(frames)
    shifted = _moving_frame(
        frames[frame_position],
        coherent_offset_px=12.0,
    )
    retained_assignments = shifted.assignments[:retained_line_count]
    retained_line_ids = {
        assignment.line_id for assignment in retained_assignments
    }
    moving[frame_position] = replace(
        shifted,
        assignments=retained_assignments,
        template_samples=tuple(
            sample
            for sample in shifted.template_samples
            if sample.line_id in retained_line_ids
        ),
    )

    pooled = pool_static_semantic_lines(
        moving,
        config=drill_fixture["config"],
    )

    assert pooled.status == "abstained"
    guard = pooled.static_consistency
    assert guard is not None
    assert guard.status == "abstained"
    assert guard.max_observed_temporal_span_px is not None
    assert guard.max_observed_temporal_span_px > 3.0
    assert any(
        reason.startswith(
            "static_consistency_temporal_shift_exceeds_bound:"
        )
        for reason in guard.rejection_reasons
    )


@pytest.mark.parametrize("usable_line_count", [1, 2, 3])
def test_boundary_guard_counts_usable_lines_not_assignment_objects(
    drill_fixture: dict[str, Any],
    usable_line_count: int,
) -> None:
    frames = drill_fixture["frames"]
    moving = list(frames)
    shifted = _moving_frame(frames[-1], coherent_offset_px=12.0)
    usable_line_ids = {
        assignment.line_id
        for assignment in shifted.assignments[:usable_line_count]
    }
    moving[-1] = replace(
        shifted,
        template_samples=tuple(
            sample
            for sample in shifted.template_samples
            if sample.line_id in usable_line_ids
        ),
    )

    pooled = pool_static_semantic_lines(
        moving,
        config=drill_fixture["config"],
    )

    assert pooled.status == "abstained"
    guard = pooled.static_consistency
    assert guard is not None
    assert guard.status == "abstained"
    assert guard.violating_boundary_degraded_frames == (
        (
            len(frames) - 1,
            frames[-1].frame_index,
            usable_line_count,
        ),
    )


@pytest.mark.parametrize("dropout_frame_count", [1, 2, 40])
def test_tail_assignment_dropout_abstains_typed(
    drill_fixture: dict[str, Any],
    dropout_frame_count: int,
) -> None:
    frames = drill_fixture["frames"]
    dropout_start = len(frames) - dropout_frame_count
    moving_outside_seed_rois = tuple(
        replace(frame, assignments=(), template_samples=())
        if position >= dropout_start
        else frame
        for position, frame in enumerate(frames)
    )

    pooled = pool_static_semantic_lines(
        moving_outside_seed_rois,
        config=drill_fixture["config"],
    )

    assert pooled.status == "abstained"
    guard = pooled.static_consistency
    assert guard is not None
    assert guard.status == "abstained"
    assert guard.violating_assignment_dropout_spans == (
        (
            dropout_start,
            len(frames) - 1,
            dropout_frame_count,
            frames[dropout_start].frame_index,
            frames[-1].frame_index,
        ),
    )
    assert guard.violating_raw_template_dropout_spans == (
        (
            dropout_start,
            len(frames) - 1,
            dropout_frame_count,
            frames[dropout_start].frame_index,
            frames[-1].frame_index,
        ),
    )
    assert any(
        reason.startswith("static_consistency_assignment_dropout_span:")
        for reason in guard.rejection_reasons
    )


@pytest.mark.parametrize("dropout_frame_count", [1, 2, 6])
def test_internal_assignment_dropout_abstains_typed(
    drill_fixture: dict[str, Any],
    dropout_frame_count: int,
) -> None:
    frames = drill_fixture["frames"]
    dropout_start = 45
    dropout_end = dropout_start + dropout_frame_count
    moving_outside_seed_rois = tuple(
        replace(frame, assignments=(), template_samples=())
        if dropout_start <= position < dropout_end
        else frame
        for position, frame in enumerate(frames)
    )

    pooled = pool_static_semantic_lines(
        moving_outside_seed_rois,
        config=drill_fixture["config"],
    )

    assert pooled.status == "abstained"
    guard = pooled.static_consistency
    assert guard is not None
    assert guard.status == "abstained"
    assert guard.violating_raw_template_dropout_spans == (
        (
            dropout_start,
            dropout_end - 1,
            dropout_frame_count,
            frames[dropout_start].frame_index,
            frames[dropout_end - 1].frame_index,
        ),
    )
    assert any(
        reason.startswith(
            "static_consistency_raw_template_dropout_span:"
        )
        for reason in guard.rejection_reasons
    )


@pytest.mark.parametrize("frame_position", [1, 45, 94])
def test_unassigned_internal_pan_is_measured_from_raw_template_samples(
    drill_fixture: dict[str, Any],
    frame_position: int,
) -> None:
    frames = drill_fixture["frames"]
    moving = list(frames)
    moving[frame_position] = replace(
        _moving_frame(
            frames[frame_position],
            coherent_offset_px=12.0,
        ),
        assignments=(),
    )

    pooled = pool_static_semantic_lines(
        moving,
        config=drill_fixture["config"],
    )

    assert pooled.status == "abstained"
    guard = pooled.static_consistency
    assert guard is not None
    assert guard.status == "abstained"
    assert guard.max_observed_temporal_span_px is not None
    assert guard.max_observed_temporal_span_px > 3.0
    assert any(
        measurement.startswith("camera_raw:")
        for measurement in guard.violating_measurements
    )
    assert any(
        reason.startswith(
            "static_consistency_temporal_shift_exceeds_bound:"
        )
        for reason in guard.rejection_reasons
    )


def test_enabled_process_seam_writes_separate_artifacts_and_adds_only_pool_line(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    drill_fixture: dict[str, Any],
) -> None:
    calibration = drill_fixture["calibration"]
    frames = drill_fixture["frames"]
    config = drill_fixture["config"]
    pooled = pool_static_semantic_lines(frames, config=config)
    result = CourtLineHardeningResult(
        config=config,
        seed_calibration_sha256=drill_fixture[
            "seed_calibration_sha256"
        ],
        raw_frame_evidence=frames,
        pooled_evidence=pooled,
        refinement={
            "accepted": True,
            "selection": "candidate",
            "test_only": True,
        },
        candidate_calibration={"must_not_be_consumed": True},
    )
    monkeypatch.setattr(
        "threed.racketsport.court_line_robustness.run_proven_court_line_pool_from_video",
        lambda *_args, **_kwargs: result,
    )
    options = process_video.PipelineOptions(
        video=DRILL_VIDEO,
        clip="drill",
        run_dir=tmp_path,
        court_line_evidence_pooling=True,
    )
    pipeline = process_video.ProcessVideoPipeline(options)
    calibration_path = pipeline.clip_dir / "court_calibration.json"
    evidence_path = pipeline.clip_dir / "court_line_evidence.json"
    calibration_bytes = DRILL_CALIBRATION.read_bytes()
    baseline_payload = json.loads(
        DRILL_BASELINE_EVIDENCE.read_text(encoding="utf-8")
    )
    calibration_path.write_bytes(calibration_bytes)
    evidence_path.write_text(
        json.dumps(baseline_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    baseline_bytes = evidence_path.read_bytes()
    baseline_observations = {
        key: baseline_payload[key]
        for key in (
            "line_observations",
            "keypoint_observations",
            "net_observations",
        )
    }

    notes, artifacts, metrics = pipeline._maybe_pool_court_line_evidence(
        calibration
    )

    assert artifacts == [
        process_video.COURT_LINE_POOL_RAW_FRAMES_NAME,
        process_video.COURT_LINE_POOLED_NAME,
    ]
    assert notes
    assert metrics["court_line_pooling"]["far_centerline_support_frames"] == 63
    assert (
        metrics["court_line_pooling"]["effective_auto_calibration_ready"]
        is True
    )
    assert calibration_path.read_bytes() == calibration_bytes
    assert evidence_path.read_bytes() == baseline_bytes
    effective, readiness_path = pipeline._court_line_evidence_for_readiness()
    assert readiness_path == (
        pipeline.clip_dir / process_video.COURT_LINE_POOLED_NAME
    )
    assert effective["line_observations"][
        : len(baseline_payload["line_observations"])
    ] == baseline_payload["line_observations"]
    assert effective["keypoint_observations"] == baseline_observations[
        "keypoint_observations"
    ]
    assert effective["net_observations"] == baseline_observations[
        "net_observations"
    ]
    assert effective["line_observations"][-1]["line_id"] == "far_centerline"
    assert effective["line_observations"][-1]["source"] == "pooled_static"
    assert effective["aggregate"]["auto_calibration_ready"] is True
    pooled_payload = json.loads(
        (
            pipeline.clip_dir / process_video.COURT_LINE_POOLED_NAME
        ).read_text(encoding="utf-8")
    )
    assert pooled_payload["provenance"]["source"] == "pooled_static"
    assert pooled_payload["provenance"]["sample_count_requested"] == 96
    assert pooled_payload["provenance"]["sample_count_actual"] == 96
    assert (
        pooled_payload["provenance"]["calibration"]["refinement_consumed"]
        is False
    )
    assert pooled_payload["readiness"]["status"] == "accepted"
    raw_payload = json.loads(
        (
            pipeline.clip_dir
            / process_video.COURT_LINE_POOL_RAW_FRAMES_NAME
        ).read_text(encoding="utf-8")
    )
    assert len(raw_payload["frames"]) == 96

    pooled_path = pipeline.clip_dir / process_video.COURT_LINE_POOLED_NAME
    pooled_bytes = pooled_path.read_bytes()
    tampered_readiness = json.loads(pooled_bytes)
    tampered_added = tampered_readiness["readiness"][
        "effective_evidence"
    ]["line_observations"][-1]
    tampered_added["image_segment"] = [[0.0, 0.0], [1.0, 1.0]]
    pooled_path.write_text(
        json.dumps(tampered_readiness, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(
        process_video._HardStageFailure,
        match="does not match recomputation",
    ):
        pipeline._court_line_evidence_for_readiness()

    tampered_pool = json.loads(pooled_bytes)
    serialized_far = next(
        line
        for line in tampered_pool["lines"]
        if line["line_id"] == "far_centerline"
    )
    serialized_far["segment"] = [[0.0, 0.0], [1.0, 1.0]]
    pooled_path.write_text(
        json.dumps(tampered_pool, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(
        process_video._HardStageFailure,
        match="does not match recomputation",
    ):
        pipeline._court_line_evidence_for_readiness()
    pooled_path.write_bytes(pooled_bytes)
    assert evidence_path.read_bytes() == baseline_bytes

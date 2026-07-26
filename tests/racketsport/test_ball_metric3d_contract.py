from __future__ import annotations

import dataclasses
import json

import pytest

from threed.racketsport.ball_metric3d_contract import (
    GT_ARTIFACT_TYPE,
    RAY_STATUS_COMPUTED,
    SCHEMA_VERSION,
    SOLVER_LOG_ARTIFACT_TYPE,
    WORLD_FRAME,
    AnchorEvent,
    CandidateSetSummary,
    ContractValidationError,
    GroundTruthObservation,
    GroundTruthObservationSet,
    SolverFrameObservation,
    SolverObservationLog,
    SourceArtifact,
    WorldRay,
    dumps_contract_json,
    read_ground_truth,
    read_solver_observation_log,
    write_ground_truth,
    write_solver_observation_log,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _gt_observation(t: float, y: float = -2.0, flags: tuple[str, ...] = ("gold",)) -> GroundTruthObservation:
    return GroundTruthObservation(
        timestamp_s=t,
        xyz_world_m=(0.5, y, 1.25),
        sigma_xyz_m=(0.01, 0.03, 0.02),
        cameras_used=("dev_side", "dev_corner"),
        triangulation_residual_px=0.75,
        quality_flags=flags,
    )


def _gt_set(count: int = 3) -> GroundTruthObservationSet:
    # Timestamps exact at FLOAT_DECIMALS so serialization rounding is lossless.
    return GroundTruthObservationSet(
        clip="synthetic_clip",
        observations=tuple(_gt_observation(i * 0.1) for i in range(count)),
    )


def _solver_frame(index: int, *, verdict: str = "accepted") -> SolverFrameObservation:
    return SolverFrameObservation(
        frame_index=index,
        timestamp_s=index * 0.1,
        observation_status="observed",
        pixel_xy=(640.0 + index, 360.0 - index),
        pixel_confidence=0.9,
        ray=WorldRay(origin_m=(0.0, -10.0, 2.0), direction=(0.0, 0.6, 0.8)),
        ray_status=RAY_STATUS_COMPUTED,
        candidate_summary=CandidateSetSummary(
            candidate_count=None,
            selected_residual_px=2.5,
            inlier_sighting=True,
            outlier_pruned=False,
            rescued=False,
        ),
        anchor_events=(
            AnchorEvent(
                anchor_id="bounce_000010",
                kind="bounce",
                status="solver_proposed",
                source="ball_bounce_candidates",
            ),
        )
        if index == 1
        else (),
        solver_verdict=verdict,
        segment_id=0,
        band="anchored_measured",
    )


def _solver_log(count: int = 3) -> SolverObservationLog:
    return SolverObservationLog(
        clip="synthetic_clip",
        frames=tuple(_solver_frame(i) for i in range(count)),
        inputs=(
            SourceArtifact(
                kind="ball_track_arc_solved",
                path="runs/example/ball_track_arc_solved.json",
                sha256="0" * 64,
            ),
        ),
        calibration_sha_verified=True,
        source_clip_id="synthetic_source_clip",
    )


# ---------------------------------------------------------------------------
# Round trips
# ---------------------------------------------------------------------------


def test_ground_truth_round_trip_preserves_evidence():
    original = _gt_set()
    payload = original.to_json_dict()
    assert payload["artifact_type"] == GT_ARTIFACT_TYPE
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["world_frame"] == WORLD_FRAME
    restored = GroundTruthObservationSet.from_json_dict(
        json.loads(dumps_contract_json(payload))
    )
    assert restored == original
    first = restored.observations[0]
    # Evidence must survive: sigma, cameras, residual, quality flags.
    assert first.sigma_xyz_m == (0.01, 0.03, 0.02)
    assert first.cameras_used == ("dev_side", "dev_corner")
    assert first.triangulation_residual_px == 0.75
    assert first.quality_flags == ("gold",)


def test_ground_truth_file_round_trip(tmp_path):
    original = _gt_set()
    path = write_ground_truth(tmp_path / "gt.json", original)
    assert read_ground_truth(path) == original


def test_solver_log_round_trip(tmp_path):
    original = _solver_log()
    payload = original.to_json_dict()
    assert payload["artifact_type"] == SOLVER_LOG_ARTIFACT_TYPE
    restored = SolverObservationLog.from_json_dict(json.loads(dumps_contract_json(payload)))
    assert restored == original
    path = write_solver_observation_log(tmp_path / "log.json", original)
    assert read_solver_observation_log(path) == original
    # Anchor evidence and provenance survive the round trip.
    assert restored.frames[1].anchor_events[0].kind == "bounce"
    assert restored.inputs[0].sha256 == "0" * 64
    assert restored.calibration_sha_verified is True
    assert restored.source_clip_id == "synthetic_source_clip"
    # source_clip_id is optional: absent stays None (older logs load fine).
    payload_without = original.to_json_dict()
    del payload_without["source_clip_id"]
    assert SolverObservationLog.from_json_dict(payload_without).source_clip_id is None


def test_serialization_is_deterministic_bytes():
    first = dumps_contract_json(_gt_set().to_json_dict())
    second = dumps_contract_json(_gt_set().to_json_dict())
    assert first == second
    log_first = dumps_contract_json(_solver_log().to_json_dict())
    log_second = dumps_contract_json(_solver_log().to_json_dict())
    assert log_first == log_second


# ---------------------------------------------------------------------------
# Validation rejections: ground truth
# ---------------------------------------------------------------------------


def test_gt_missing_sigma_field_rejected():
    payload = _gt_set().to_json_dict()
    del payload["observations"][0]["sigma_xyz_m"]
    with pytest.raises(ContractValidationError, match="sigma_xyz_m"):
        GroundTruthObservationSet.from_json_dict(payload)


def test_gt_null_sigma_rejected():
    payload = _gt_set().to_json_dict()
    payload["observations"][0]["sigma_xyz_m"] = None
    with pytest.raises(ContractValidationError, match="sigma_xyz_m"):
        GroundTruthObservationSet.from_json_dict(payload)


def test_gt_nonpositive_sigma_rejected():
    payload = _gt_set().to_json_dict()
    payload["observations"][0]["sigma_xyz_m"] = [0.01, 0.0, 0.02]
    with pytest.raises(ContractValidationError, match="sigma"):
        GroundTruthObservationSet.from_json_dict(payload)


def test_gt_non_monotonic_timestamps_rejected():
    payload = _gt_set().to_json_dict()
    payload["observations"][1]["timestamp_s"] = payload["observations"][2]["timestamp_s"] + 1.0
    with pytest.raises(ContractValidationError, match="non-monotonic"):
        GroundTruthObservationSet.from_json_dict(payload)


def test_gt_duplicate_timestamps_rejected():
    payload = _gt_set().to_json_dict()
    payload["observations"][1]["timestamp_s"] = payload["observations"][0]["timestamp_s"]
    with pytest.raises(ContractValidationError, match="non-monotonic"):
        GroundTruthObservationSet.from_json_dict(payload)


def test_gt_unknown_quality_flag_rejected():
    payload = _gt_set().to_json_dict()
    payload["observations"][0]["quality_flags"] = ["gold", "totally_new_flag"]
    with pytest.raises(ContractValidationError, match="totally_new_flag"):
        GroundTruthObservationSet.from_json_dict(payload)


def test_gt_empty_camera_set_rejected():
    payload = _gt_set().to_json_dict()
    payload["observations"][0]["cameras_used"] = []
    with pytest.raises(ContractValidationError, match="cameras_used"):
        GroundTruthObservationSet.from_json_dict(payload)


def test_gt_negative_triangulation_residual_rejected():
    payload = _gt_set().to_json_dict()
    payload["observations"][0]["triangulation_residual_px"] = -1.0
    with pytest.raises(ContractValidationError, match="triangulation_residual_px"):
        GroundTruthObservationSet.from_json_dict(payload)


def test_gt_unknown_schema_version_rejected():
    payload = _gt_set().to_json_dict()
    payload["schema_version"] = SCHEMA_VERSION + 1
    with pytest.raises(ContractValidationError, match="schema_version"):
        GroundTruthObservationSet.from_json_dict(payload)


def test_gt_wrong_artifact_type_rejected():
    payload = _gt_set().to_json_dict()
    payload["artifact_type"] = "something_else"
    with pytest.raises(ContractValidationError, match="artifact_type"):
        GroundTruthObservationSet.from_json_dict(payload)


def test_gt_optional_covariance_round_trip_and_validation():
    observation = GroundTruthObservation(
        timestamp_s=0.0,
        xyz_world_m=(0.5, -2.0, 1.25),
        sigma_xyz_m=(0.01, 0.03, 0.02),
        cameras_used=("dev_side", "dev_corner"),
        triangulation_residual_px=0.75,
        quality_flags=("gold",),
        covariance_world_m2=((0.0001, 0.0, 0.0), (0.0, 0.0009, 0.0), (0.0, 0.0, 0.0004)),
        triangulation_angle_deg=24.5,
        reviewed=True,
    )
    original = GroundTruthObservationSet(clip="clip", observations=(observation,))
    restored = GroundTruthObservationSet.from_json_dict(
        json.loads(dumps_contract_json(original.to_json_dict()))
    )
    assert restored == original
    # Optional fields default to absent, never fabricated.
    assert _gt_set().observations[0].covariance_world_m2 is None


def test_gt_malformed_covariance_rejected():
    payload = _gt_set().to_json_dict()
    payload["observations"][0]["covariance_world_m2"] = [[1.0, 0.0], [0.0, 1.0]]
    with pytest.raises(ContractValidationError, match="covariance_world_m2"):
        GroundTruthObservationSet.from_json_dict(payload)


def test_gt_negative_variance_rejected():
    payload = _gt_set().to_json_dict()
    payload["observations"][0]["covariance_world_m2"] = [
        [-0.01, 0.0, 0.0],
        [0.0, 0.01, 0.0],
        [0.0, 0.0, 0.01],
    ]
    with pytest.raises(ContractValidationError, match="variance"):
        GroundTruthObservationSet.from_json_dict(payload)


def test_gt_non_finite_position_rejected():
    with pytest.raises(ContractValidationError, match="xyz_world_m"):
        GroundTruthObservation(
            timestamp_s=0.0,
            xyz_world_m=(0.0, float("nan"), 0.0),
            sigma_xyz_m=(0.01, 0.01, 0.01),
            cameras_used=("dev_side",),
            triangulation_residual_px=0.0,
        ).validate()


# ---------------------------------------------------------------------------
# Validation rejections: solver log
# ---------------------------------------------------------------------------


def test_solver_log_non_monotonic_timestamps_rejected():
    payload = _solver_log().to_json_dict()
    payload["frames"][2]["timestamp_s"] = payload["frames"][0]["timestamp_s"]
    with pytest.raises(ContractValidationError, match="non-monotonic"):
        SolverObservationLog.from_json_dict(payload)


def test_solver_log_unknown_observation_status_rejected():
    payload = _solver_log().to_json_dict()
    payload["frames"][0]["observation_status"] = "kind_of_seen"
    with pytest.raises(ContractValidationError, match="observation_status"):
        SolverObservationLog.from_json_dict(payload)


def test_solver_log_unknown_verdict_rejected():
    payload = _solver_log().to_json_dict()
    payload["frames"][0]["solver_verdict"] = "promoted"
    with pytest.raises(ContractValidationError, match="solver_verdict"):
        SolverObservationLog.from_json_dict(payload)


def test_solver_log_empty_source_clip_id_rejected():
    payload = _solver_log().to_json_dict()
    payload["source_clip_id"] = ""
    with pytest.raises(ContractValidationError, match="source_clip_id"):
        SolverObservationLog.from_json_dict(payload)


def test_solver_log_observed_without_pixel_rejected():
    payload = _solver_log().to_json_dict()
    payload["frames"][0]["pixel_xy"] = None
    with pytest.raises(ContractValidationError, match="pixel_xy"):
        SolverObservationLog.from_json_dict(payload)


def test_solver_log_computed_ray_status_without_ray_rejected():
    payload = _solver_log().to_json_dict()
    payload["frames"][0]["ray"] = None
    with pytest.raises(ContractValidationError, match="ray"):
        SolverObservationLog.from_json_dict(payload)


def test_solver_log_ray_with_non_computed_status_rejected():
    payload = _solver_log().to_json_dict()
    payload["frames"][0]["ray_status"] = "calibration_not_sha_verified"
    with pytest.raises(ContractValidationError, match="ray"):
        SolverObservationLog.from_json_dict(payload)


def test_solver_log_non_unit_ray_direction_rejected():
    payload = _solver_log().to_json_dict()
    payload["frames"][0]["ray"]["direction"] = [1.0, 1.0, 1.0]
    with pytest.raises(ContractValidationError, match="unit"):
        SolverObservationLog.from_json_dict(payload)


def test_solver_log_unknown_ray_status_rejected():
    payload = _solver_log().to_json_dict()
    payload["frames"][0]["ray"] = None
    payload["frames"][0]["ray_status"] = "eyeballed"
    with pytest.raises(ContractValidationError, match="ray_status"):
        SolverObservationLog.from_json_dict(payload)


def test_solver_log_non_monotonic_frame_index_rejected():
    # Keep timestamps increasing so only the frame_index ordering is at fault.
    frames = (
        _solver_frame(0),
        dataclasses.replace(_solver_frame(2), timestamp_s=0.1),
        dataclasses.replace(_solver_frame(1), timestamp_s=0.2),
    )
    with pytest.raises(ContractValidationError, match="frame_index"):
        SolverObservationLog(clip="clip", frames=frames).validate()


def test_solver_log_unknown_schema_version_rejected():
    payload = _solver_log().to_json_dict()
    payload["schema_version"] = 99
    with pytest.raises(ContractValidationError, match="schema_version"):
        SolverObservationLog.from_json_dict(payload)

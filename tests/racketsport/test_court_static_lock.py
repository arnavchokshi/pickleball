from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tests.racketsport.json_schema_assertions import assert_matches_json_schema
from threed.racketsport.court_camera_geometry import PinholeIntrinsics
from threed.racketsport.court_static_lock import (
    CourtLockArtifact,
    CourtLockCameraParameters,
    CourtLockDistortion,
    CourtLockEvidenceSummary,
    CourtLockResidual,
    FrameCourtTransform,
    StaticCourtObservation,
    StaticFrameEvidence,
    StaticMotionDiagnostics,
    diagnose_static_camera,
    pool_static_frame_evidence,
    read_court_lock,
    select_static_frame_evidence,
    write_court_lock,
)
from threed.racketsport.schemas import validate_artifact_file


def _homography(*, dx: float = 0.0, dy: float = 0.0) -> tuple[tuple[float, float, float], ...]:
    return (
        (65.0, 2.0, 640.0 + dx),
        (1.0, 29.0, 360.0 + dy),
        (0.001, -0.003, 1.0),
    )


def _observation(frame_index: int, *, outlier: bool = False) -> StaticCourtObservation:
    offset = 80.0 if outlier else 0.15 * ((frame_index % 3) - 1)
    return StaticCourtObservation(
        semantic_name="near_left_corner",
        xy=(221.0 + offset, 474.0 - offset),
        confidence=0.90,
        covariance_px2=((0.5, 0.0), (0.0, 0.5)),
        line_support=0.95,
    )


def _frame(frame_index: int, *, quality: float, outlier: bool = False) -> StaticFrameEvidence:
    return StaticFrameEvidence(
        frame_index=frame_index,
        line_support=quality,
        surface_support=quality,
        visible_fraction=quality,
        sharpness=quality,
        occlusion_fraction=1.0 - quality,
        observations=(_observation(frame_index, outlier=outlier),),
    )


def test_frame_selection_and_pooling_are_deterministic_bounded_and_reject_outlier() -> None:
    frames = [_frame(index, quality=0.65 + 0.02 * index) for index in range(10)]
    frames[7] = _frame(7, quality=0.95, outlier=True)

    forward = select_static_frame_evidence(frames, max_frames=8)
    reversed_selection = select_static_frame_evidence(list(reversed(frames)), max_frames=8)
    assert tuple(frame.frame_index for frame in forward) == tuple(
        frame.frame_index for frame in reversed_selection
    )
    assert len(forward) == 8

    pooled_forward = pool_static_frame_evidence(frames, max_frames=8)
    pooled_reverse = pool_static_frame_evidence(list(reversed(frames)), max_frames=8)
    assert pooled_forward == pooled_reverse
    assert len(pooled_forward.selected_frame_indices) == 8
    assert len(pooled_forward.observations) == 1
    pooled = pooled_forward.observations[0]
    assert 7 in pooled.rejected_frame_indices
    assert pooled.xy == pytest.approx((221.0, 474.0), abs=0.2)
    assert pooled.to_solver_observation()["source"] == "static_evidence_pool"


def test_static_motion_diagnostics_separate_static_pan_and_uncertain_ambiguity() -> None:
    static = diagnose_static_camera(
        [FrameCourtTransform(frame_index=index, homography_image_from_court=_homography()) for index in range(5)]
    )
    assert static.status == "static"
    assert static.static_lock_usable
    assert not static.camera_motion_suspected

    moving = diagnose_static_camera(
        [
            FrameCourtTransform(
                frame_index=index,
                homography_image_from_court=_homography(dx=1.2 * index, dy=0.2 * index),
            )
            for index in range(5)
        ]
    )
    assert moving.status == "moving"
    assert moving.camera_motion_suspected
    assert not moving.static_lock_usable
    assert moving.net_center_displacement_px is not None
    assert moving.net_center_displacement_px > 4.0

    high_uncertainty = tuple(
        tuple(float(value) for value in row)
        for row in (np.eye(8, dtype=np.float64) * 0.02)
    )
    ambiguous = diagnose_static_camera(
        [
            FrameCourtTransform(
                frame_index=index,
                homography_image_from_court=_homography(dx=0.5 * index),
                transform_covariance=high_uncertainty,
            )
            for index in range(5)
        ]
    )
    assert ambiguous.status == "ambiguous"
    assert ambiguous.camera_motion_suspected
    assert not ambiguous.static_lock_usable
    assert ambiguous.drift_uncertainty_px is not None
    assert ambiguous.drift_uncertainty_px > 0.0


def test_court_lock_schema_and_dataclass_serialization_roundtrip_are_byte_stable(tmp_path: Path) -> None:
    diagnostics = diagnose_static_camera(
        [FrameCourtTransform(frame_index=index, homography_image_from_court=_homography()) for index in range(5)]
    )
    covariance = tuple(
        tuple(float(value) for value in row)
        for row in (np.eye(8, dtype=np.float64) * 1.0e-6)
    )
    lock = CourtLockArtifact(
        coordinate_space="pixels_undistorted_native",
        homography_image_from_court=_homography(),
        camera_parameters=CourtLockCameraParameters(
            intrinsics=PinholeIntrinsics(fx=720.0, fy=720.0, cx=640.0, cy=360.0),
            source="iphone_profile",
        ),
        distortion=CourtLockDistortion(
            k1=-0.28,
            k1_variance=0.0004,
            source="reviewed_correspondence_fit",
            optimized=True,
        ),
        transform_covariance=covariance,
        source="multi_frame_point_and_line",
        evidence=CourtLockEvidenceSummary(
            candidate_frame_count=20,
            selected_frame_indices=(0, 10, 20, 30, 40),
            pooled_semantic_count=12,
        ),
        static_motion=diagnostics,
        residual_px=CourtLockResidual(median_px=1.1, p95_px=2.8),
        score_components={"line": 0.8, "point": 0.9},
        checkpoint_sha256="a" * 64,
        measurement_valid=False,
        authority_state="review_only",
        verified=False,
    )
    first_path = tmp_path / "court_lock.json"
    second_path = tmp_path / "court_lock_again.json"
    write_court_lock(lock, first_path)
    loaded = read_court_lock(first_path)
    write_court_lock(loaded, second_path)

    assert loaded == lock
    assert first_path.read_bytes() == second_path.read_bytes()
    schema = json.loads(
        (Path(__file__).parents[2] / "docs/racketsport/court_lock_schema.json").read_text(encoding="utf-8")
    )
    assert_matches_json_schema(loaded.to_dict(), schema)
    assert validate_artifact_file("court_lock", first_path).artifact_type == "racketsport_court_lock"


def test_measurement_valid_refuses_ambiguous_static_state() -> None:
    ambiguous = StaticMotionDiagnostics(
        status="ambiguous",
        camera_motion_suspected=True,
        static_lock_usable=False,
        reference_frame_index=0,
        frame_indices=(0, 1, 2),
        valid_frame_count=3,
        invalid_frame_count=0,
        median_corner_drift_px=1.8,
        p95_corner_drift_px=3.8,
        max_corner_drift_px=4.0,
        net_center_displacement_px=2.9,
        monotonic_trend_px_per_frame=0.1,
        drift_uncertainty_px=0.5,
        reasons=("static_threshold_overlaps_transform_uncertainty",),
    )
    with pytest.raises(ValueError, match="confirmed static camera"):
        CourtLockArtifact(
            coordinate_space="pixels_undistorted_native",
            homography_image_from_court=_homography(),
            camera_parameters=CourtLockCameraParameters(
                intrinsics=PinholeIntrinsics(fx=720.0, fy=720.0, cx=640.0, cy=360.0),
                source="fixture",
            ),
            distortion=CourtLockDistortion(k1=0.0),
            transform_covariance=None,
            source="camera_profile_prior",
            evidence=CourtLockEvidenceSummary(
                candidate_frame_count=3,
                selected_frame_indices=(0, 1, 2),
                pooled_semantic_count=0,
            ),
            static_motion=ambiguous,
            residual_px=CourtLockResidual(median_px=5.0, p95_px=9.0),
            measurement_valid=True,
        )

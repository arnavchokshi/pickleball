from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from pydantic import ValidationError

from threed.racketsport.court_camera_geometry import PinholeIntrinsics
from threed.racketsport.court_keypoint_net import PICKLEBALL_KEYPOINTS
from threed.racketsport.court_lock_visualization import adapt_court_lock_for_visualization
from threed.racketsport.court_static_lock import (
    CourtLockArtifact,
    CourtLockCameraParameters,
    CourtLockDistortion,
    CourtLockEvidenceSummary,
    CourtLockResidual,
    FrameCourtTransform,
    StaticMotionDiagnostics,
    diagnose_static_camera,
)
from threed.racketsport.court_zones import build_court_zones
from threed.racketsport.net_plane import build_net_plane
from threed.racketsport.schemas import CourtCalibration, CourtZones, NetPlane


_H = (
    (76.0, 3.0, 640.0),
    (1.5, 32.0, 430.0),
    (0.0015, -0.004, 1.0),
)


def _lock(
    *,
    coordinate_space: str = "pixels_undistorted_native",
    k1: float = 0.0,
    motion: StaticMotionDiagnostics | None = None,
) -> CourtLockArtifact:
    static = motion or diagnose_static_camera(
        [FrameCourtTransform(frame_index=index, homography_image_from_court=_H) for index in range(4)]
    )
    return CourtLockArtifact(
        coordinate_space=coordinate_space,  # type: ignore[arg-type]
        homography_image_from_court=_H,
        camera_parameters=CourtLockCameraParameters(
            intrinsics=PinholeIntrinsics(fx=920.0, fy=920.0, cx=640.0, cy=360.0),
            source="iphone_profile_fixture",
        ),
        distortion=CourtLockDistortion(
            k1=k1,
            k1_variance=0.0001,
            source="fixture",
            optimized=k1 != 0.0,
        ),
        transform_covariance=tuple(
            tuple(float(value) for value in row)
            for row in (np.eye(8, dtype=np.float64) * 1.0e-6)
        ),
        source="multi_frame_point_and_line",
        evidence=CourtLockEvidenceSummary(
            candidate_frame_count=12,
            selected_frame_indices=(0, 3, 6, 9),
            pooled_semantic_count=12,
        ),
        static_motion=static,
        residual_px=CourtLockResidual(median_px=1.2, p95_px=3.5),
        score_components={"point": 0.9, "line": 0.8},
        measurement_valid=False,
        authority_state="review_only",
        verified=False,
    )


def test_static_lock_adapts_to_existing_artifacts_with_named_floor_only_correspondences() -> None:
    result = adapt_court_lock_for_visualization(_lock(), image_size=(1280, 720))
    payloads = result.artifact_payloads()

    calibration = CourtCalibration.model_validate(payloads["court_calibration.json"])
    zones = CourtZones.model_validate(payloads["court_zones.json"])
    net = NetPlane.model_validate(payloads["net_plane.json"])
    assert calibration.usage == "visualization_only"
    assert calibration.authority_state == "review_only"
    assert calibration.measurement_valid is False
    assert calibration.trust_band == "preview"
    assert "measurement_valid=false" in calibration.capture_quality.reasons
    assert zones == build_court_zones("pickleball")
    assert net == build_net_plane("pickleball")

    expected_floor_names = [
        point.name for point in PICKLEBALL_KEYPOINTS if abs(float(point.world_xyz_m[2])) <= 1.0e-12
    ]
    correspondence_names = [row.semantic_name for row in result.named_floor_correspondences]
    assert correspondence_names == expected_floor_names
    assert not any(name.startswith("net_") for name in correspondence_names)
    assert len(calibration.image_pts) == len(calibration.world_pts) == len(expected_floor_names) == 12
    assert result.metadata_payload()["trust"] == {
        "authority_state": "review_only",
        "measurement_valid": False,
        "usage": "visualization_only",
        "verified": False,
    }


def test_raw_k1_lock_is_undistorted_and_refit_before_visualization_pose() -> None:
    result = adapt_court_lock_for_visualization(
        _lock(coordinate_space="pixels_raw_native", k1=-0.12),
        image_size=(1280, 720),
    )

    diagnostics = result.adapter_diagnostics
    assert diagnostics["source_lock_coordinate_space"] == "pixels_raw_native"
    assert diagnostics["output_homography_coordinate_space"] == "pixels_undistorted_native"
    assert diagnostics["undistortion_refit_p95_px"] <= 5.0
    assert all(row.raw_image_xy is not None for row in result.named_floor_correspondences)
    assert result.court_calibration.intrinsics.dist[0] == pytest.approx(-0.12)
    assert "raw_lock_points_undistorted_before_homography_refit" in (
        result.court_calibration.capture_quality.reasons
    )


def test_visualization_adapter_rejects_motion_high_residual_and_uncertain_distortion() -> None:
    moving = StaticMotionDiagnostics(
        status="moving",
        camera_motion_suspected=True,
        static_lock_usable=False,
        reference_frame_index=0,
        frame_indices=(0, 1, 2),
        valid_frame_count=3,
        invalid_frame_count=0,
        median_corner_drift_px=8.0,
        p95_corner_drift_px=10.0,
        max_corner_drift_px=12.0,
        net_center_displacement_px=10.0,
        monotonic_trend_px_per_frame=5.0,
        drift_uncertainty_px=0.1,
        reasons=("court_projection_drift_exceeds_static_threshold",),
    )
    with pytest.raises(ValueError, match="confirmed static"):
        adapt_court_lock_for_visualization(_lock(motion=moving), image_size=(1280, 720))

    with pytest.raises(ValueError, match="p95 residual"):
        adapt_court_lock_for_visualization(
            replace(_lock(), residual_px=CourtLockResidual(median_px=15.0, p95_px=25.0)),
            image_size=(1280, 720),
        )

    with pytest.raises(ValueError, match="k1 uncertainty"):
        adapt_court_lock_for_visualization(
            replace(
                _lock(),
                distortion=CourtLockDistortion(k1=0.0, k1_variance=0.09, source="uncertain"),
            ),
            image_size=(1280, 720),
        )


def test_court_calibration_visualization_trust_contract_is_fail_closed() -> None:
    valid = adapt_court_lock_for_visualization(_lock(), image_size=(1280, 720)).court_calibration
    payload = valid.model_dump(mode="json")
    payload["measurement_valid"] = True
    with pytest.raises(ValidationError, match="False"):
        CourtCalibration.model_validate(payload)

    incomplete = valid.model_dump(mode="json")
    del incomplete["authority_state"]
    with pytest.raises(ValidationError, match="trust fields must be complete"):
        CourtCalibration.model_validate(incomplete)

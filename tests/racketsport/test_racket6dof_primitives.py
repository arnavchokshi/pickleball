from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from threed.racketsport.schemas import CourtCalibration

from threed.racketsport.racket6dof import (
    SE3PoseConfidence,
    camera_paddle_pose_to_court_world,
    estimate_planar_paddle_pose,
    estimate_planar_paddle_pose_with_diagnostics,
    normalize_face_normal,
    paddle_face_corners_object_cm,
    pose_face_normal,
    project_paddle_corners,
    rebound_consistency,
    smooth_racket_pose_samples,
    validate_contact_point_face_cm,
    validate_paddle_dimensions,
)


def test_validate_paddle_dimensions_accepts_named_or_short_dimension_keys() -> None:
    named = validate_paddle_dimensions({"length": 16.0, "width": 8.0})
    short = validate_paddle_dimensions({"h": 15.5, "w": 7.75})

    assert named.length_in == pytest.approx(16.0)
    assert named.width_in == pytest.approx(8.0)
    assert named.length_cm == pytest.approx(40.64)
    assert named.width_cm == pytest.approx(20.32)
    assert short.length_in == pytest.approx(15.5)
    assert short.width_in == pytest.approx(7.75)


def test_validate_paddle_dimensions_rejects_missing_non_positive_or_non_finite_values() -> None:
    with pytest.raises(ValueError, match="paddle_dims_in"):
        validate_paddle_dimensions({"length": 16.0})

    with pytest.raises(ValueError, match="positive"):
        validate_paddle_dimensions({"length": 16.0, "width": 0.0})

    with pytest.raises(ValueError, match="finite"):
        validate_paddle_dimensions({"length": float("nan"), "width": 8.0})


def test_normalize_face_normal_returns_unit_vector_without_changing_direction() -> None:
    assert normalize_face_normal([0.0, 0.0, 4.0]) == pytest.approx((0.0, 0.0, 1.0))
    assert normalize_face_normal([3.0, 4.0, 0.0]) == pytest.approx((0.6, 0.8, 0.0))

    with pytest.raises(ValueError, match="face_normal"):
        normalize_face_normal([1.0, 0.0])

    with pytest.raises(ValueError, match="non-zero"):
        normalize_face_normal([0.0, 0.0, 0.0])


def test_validate_contact_point_face_cm_accepts_points_inside_paddle_face() -> None:
    dims = validate_paddle_dimensions({"length": 16.0, "width": 8.0})

    assert validate_contact_point_face_cm([0.0, 0.0], dims) == pytest.approx((0.0, 0.0))
    assert validate_contact_point_face_cm([10.16, -20.32], dims) == pytest.approx((10.16, -20.32))


def test_validate_contact_point_face_cm_rejects_points_outside_paddle_face() -> None:
    dims = validate_paddle_dimensions({"length": 16.0, "width": 8.0})

    with pytest.raises(ValueError, match="width"):
        validate_contact_point_face_cm([10.17, 0.0], dims)

    with pytest.raises(ValueError, match="length"):
        validate_contact_point_face_cm([0.0, 20.33], dims)

    with pytest.raises(ValueError, match="contact_point_face_cm"):
        validate_contact_point_face_cm([0.0], dims)


def test_se3_pose_confidence_validates_cpu_only_ukf_placeholder_fields() -> None:
    pose = SE3PoseConfidence(
        R=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        t=[0.1, 0.2, 0.3],
        confidence=0.85,
    )

    assert pose.R == ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    assert pose.t == pytest.approx((0.1, 0.2, 0.3))
    assert pose.confidence == pytest.approx(0.85)
    assert pose.source == "ukf_placeholder"

    with pytest.raises(ValueError, match="R"):
        SE3PoseConfidence(R=[[1.0]], t=[0.0, 0.0, 0.0], confidence=0.5)

    with pytest.raises(ValueError, match="t"):
        SE3PoseConfidence(
            R=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            t=[0.0],
            confidence=0.5,
        )

    with pytest.raises(ValueError, match="confidence"):
        SE3PoseConfidence(
            R=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            t=[0.0, 0.0, 0.0],
            confidence=1.01,
        )


def test_planar_paddle_pose_round_trips_projected_corners() -> None:
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    camera_matrix = [[900.0, 0.0, 320.0], [0.0, 900.0, 240.0], [0.0, 0.0, 1.0]]
    dims = {"length": 16.0, "width": 8.0}
    object_points = np.asarray(paddle_face_corners_object_cm(dims), dtype=np.float64)
    rvec = np.asarray([[0.18], [-0.10], [0.07]], dtype=np.float64)
    tvec = np.asarray([[3.0], [-1.5], [95.0]], dtype=np.float64)
    projected, _ = cv2.projectPoints(object_points, rvec, tvec, np.asarray(camera_matrix), None)
    image_points = projected.reshape(-1, 2).tolist()

    pose = estimate_planar_paddle_pose(image_points, camera_matrix, dims)
    reprojected = project_paddle_corners(pose, camera_matrix, dims)

    max_error = max(
        ((got[0] - want[0]) ** 2 + (got[1] - want[1]) ** 2) ** 0.5
        for got, want in zip(reprojected, image_points)
    )
    assert max_error < 0.75
    assert pose.confidence > 0.8
    assert pose.source == "pnp_ippe"
    assert pose_face_normal(pose)[2] > 0.9


def test_planar_paddle_pose_diagnostics_report_reprojection_and_ambiguity() -> None:
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    camera_matrix = [[900.0, 0.0, 320.0], [0.0, 900.0, 240.0], [0.0, 0.0, 1.0]]
    dims = {"length": 16.0, "width": 8.0}
    object_points = np.asarray(paddle_face_corners_object_cm(dims), dtype=np.float64)
    rvec = np.asarray([[0.18], [-0.10], [0.07]], dtype=np.float64)
    tvec = np.asarray([[3.0], [-1.5], [95.0]], dtype=np.float64)
    projected, _ = cv2.projectPoints(object_points, rvec, tvec, np.asarray(camera_matrix), None)
    clean_points = projected.reshape(-1, 2).tolist()
    noisy_points = [list(point) for point in clean_points]
    noisy_points[2][0] += 12.0
    noisy_points[2][1] -= 8.0

    clean = estimate_planar_paddle_pose_with_diagnostics(clean_points, camera_matrix, dims)
    noisy = estimate_planar_paddle_pose_with_diagnostics(noisy_points, camera_matrix, dims)

    assert clean.pose.source == "pnp_ippe"
    assert clean.reprojection_error_px < 0.75
    assert clean.candidate_count >= 1
    if clean.candidate_count >= 2:
        assert clean.alt_pose is not None
        assert clean.alt_pose.source == "pnp_ippe_alt"
        assert clean.alt_pose.t != clean.pose.t
    assert clean.ambiguity_margin_px is None or clean.ambiguity_margin_px >= 0.0
    assert noisy.reprojection_error_px > clean.reprojection_error_px
    assert noisy.pose.confidence < clean.pose.confidence
    assert len(noisy.candidate_reprojection_errors_px) == noisy.candidate_count


def test_camera_paddle_pose_to_court_world_uses_calibration_extrinsics_and_meter_units() -> None:
    pose = SE3PoseConfidence(
        R=[[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
        t=[100.0, 200.0, 300.0],
        confidence=0.82,
        source="pnp_ippe",
    )
    calibration = SimpleNamespace(
        extrinsics=SimpleNamespace(
            R=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            t=[0.0, 0.0, 1.0],
        )
    )

    world_pose = camera_paddle_pose_to_court_world(pose, calibration, input_translation_unit="cm")

    for got, want in zip(world_pose.R, pose.R, strict=True):
        assert got == pytest.approx(want)
    assert world_pose.t == pytest.approx((1.0, 2.0, 2.0))
    assert world_pose.confidence == pytest.approx(0.82)
    assert world_pose.source == "pnp_ippe:court_Z0"


def test_frozen_real_calibration_camera_world_and_cm_to_m_parity_is_exact() -> None:
    repo = Path(__file__).resolve().parents[2]
    fixture = (
        repo
        / "runs/lanes/w7_critique_20260709/wolv_world"
        / "wolverine_mixed_0200_mid_steep_corner/court_calibration.json"
    )
    assert hashlib.sha256(fixture.read_bytes()).hexdigest() == (
        "fb4e6f7f54d2c40e2c7b491e436261f747240945a6f0d154c4dd943e28edbacf"
    )
    calibration = CourtCalibration.model_validate_json(fixture.read_text(encoding="utf-8"))
    assert "coordinate_contract" not in calibration.model_dump(mode="json")
    pose = SE3PoseConfidence(
        R=[
            [0.9362933635841992, -0.3129918257854679, -0.1593450793079779],
            [0.28962947762551555, 0.9447024859948943, -0.1537919979889642],
            [0.19866933079506122, 0.09784339500725571, 0.975170327201816],
        ],
        t=[12.5, -7.25, 183.75],
        confidence=0.8125,
        source="frozen_ippe",
    )

    world_pose = camera_paddle_pose_to_court_world(
        pose,
        calibration,
        input_translation_unit="cm",
    )
    numeric = {
        "R": world_pose.R,
        "t": world_pose.t,
        "confidence": world_pose.confidence,
        "source": world_pose.source,
    }
    digest = hashlib.sha256(
        json.dumps(numeric, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    assert digest == "a5dd043e7eee0985eeb92686f83f9cff6a8ec35007e7571b50a09a63bb59163c"


def test_smooth_racket_pose_samples_clamps_implausible_frame_jumps() -> None:
    first = SE3PoseConfidence(
        R=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        t=[0.0, 0.0, 100.0],
        confidence=0.95,
    )
    second = SE3PoseConfidence(
        R=[[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        t=[100.0, 0.0, 100.0],
        confidence=0.95,
    )

    smoothed, report = smooth_racket_pose_samples(
        [(0.0, first), (1.0 / 120.0, second)],
        max_translation_speed_per_s=240.0,
        max_angular_speed_deg_s=600.0,
    )

    assert smoothed[1].pose.t[0] == pytest.approx(2.0)
    assert report.translation_clamp_count == 1
    assert report.rotation_clamp_count == 1
    assert report.max_translation_speed_per_s == pytest.approx(12000.0)
    assert report.max_angular_speed_deg_s == pytest.approx(10800.0)


def test_rebound_consistency_requires_ball_velocity_to_cross_face_normal() -> None:
    good = rebound_consistency(
        incoming_velocity=[0.0, 0.0, -8.0],
        outgoing_velocity=[0.0, 0.0, 12.0],
        face_normal=[0.0, 0.0, 1.0],
    )
    bad = rebound_consistency(
        incoming_velocity=[0.0, 0.0, -8.0],
        outgoing_velocity=[0.0, 0.0, -6.0],
        face_normal=[0.0, 0.0, 1.0],
    )

    assert good.consistent is True
    assert good.normal_speed_before == pytest.approx(-8.0)
    assert good.normal_speed_after == pytest.approx(12.0)
    assert bad.consistent is False
    assert "normal_component_did_not_flip" in bad.notes

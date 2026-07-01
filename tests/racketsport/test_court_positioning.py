from __future__ import annotations

import math

import pytest

from threed.racketsport.court_keypoint_net import PICKLEBALL_KEYPOINTS
from threed.racketsport.court_positioning import (
    CameraFloorGeometry,
    CourtGateInput,
    CourtDecisionInput,
    back_project_pixel_to_floor,
    court_escalation_reasons,
    decide_court_boundary,
    estimate_ground_sample_distance,
    estimate_position_uncertainty,
    solve_metric_court_placement,
    transform_court_to_world,
)


def _camera_geometry() -> CameraFloorGeometry:
    return CameraFloorGeometry(
        intrinsics={"fx": 100.0, "fy": 100.0, "cx": 50.0, "cy": 50.0, "dist": []},
        camera_origin_world=[0.0, 0.0, 2.0],
        R_world_camera=[
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, -1.0],
        ],
        floor_plane_point=[0.0, 0.0, 0.0],
        floor_plane_normal=[0.0, 0.0, 1.0],
    )


def test_back_project_pixel_to_floor_uses_intrinsics_pose_and_floor_plane() -> None:
    geometry = _camera_geometry()

    center = back_project_pixel_to_floor([50.0, 50.0], geometry)
    one_focal_length_right = back_project_pixel_to_floor([150.0, 50.0], geometry)

    assert center == pytest.approx([0.0, 0.0, 0.0])
    assert one_focal_length_right == pytest.approx([2.0, 0.0, 0.0])


def test_solve_metric_court_placement_recovers_scale_locked_rigid_transform() -> None:
    theta = math.radians(30.0)
    translation = [4.25, -1.75, 0.0]
    world_points = {
        point.name: [
            math.cos(theta) * point.world_xyz_m[0] - math.sin(theta) * point.world_xyz_m[1] + translation[0],
            math.sin(theta) * point.world_xyz_m[0] + math.cos(theta) * point.world_xyz_m[1] + translation[1],
            0.0,
        ]
        for point in PICKLEBALL_KEYPOINTS
    }

    solve = solve_metric_court_placement(world_points)

    assert solve.metric_confidence == "high"
    assert solve.scale_estimate == pytest.approx(1.0, abs=1e-9)
    assert solve.residual_error_m.p95 < 1e-8
    assert transform_court_to_world(PICKLEBALL_KEYPOINTS[0].world_xyz_m, solve.T_world_court) == pytest.approx(
        world_points[PICKLEBALL_KEYPOINTS[0].name],
        abs=1e-8,
    )


def test_solve_metric_court_placement_fails_closed_when_scale_is_not_regulation() -> None:
    world_points = {
        point.name: [point.world_xyz_m[0] * 1.05, point.world_xyz_m[1] * 1.05, 0.0]
        for point in PICKLEBALL_KEYPOINTS
    }

    solve = solve_metric_court_placement(world_points)

    assert solve.metric_confidence == "low"
    assert solve.scale_estimate == pytest.approx(1.05, rel=1e-3)
    assert "scale_drift_gt_2pct" in solve.gate_failures


def test_uncertainty_model_uses_gsd_plane_and_calibration_terms() -> None:
    geometry = _camera_geometry()

    gsd = estimate_ground_sample_distance([50.0, 50.0], geometry)
    sigma = estimate_position_uncertainty(
        pixel_error_px=1.5,
        gsd_m_per_px=gsd,
        plane_sigma_m=0.012,
        calibration_sigma_m=0.018,
    )

    assert gsd == pytest.approx(0.02, rel=1e-3)
    assert sigma == pytest.approx(math.sqrt((1.5 * 0.02) ** 2 + 0.012**2 + 0.018**2))


def test_kitchen_decision_includes_nvz_line_and_abstains_inside_uncertainty_margin() -> None:
    on_near_nvz_line_strip = CourtDecisionInput(
        boundary="near_kitchen",
        foot_court_xy=[0.0, -2.1336 - 0.010],
        sigma_p_m=0.002,
        metric_confidence="high",
    )
    just_outside_but_uncertain = CourtDecisionInput(
        boundary="near_kitchen",
        foot_court_xy=[0.0, -2.1336 - 0.030],
        sigma_p_m=0.010,
        metric_confidence="high",
    )
    low_confidence_inside = CourtDecisionInput(
        boundary="near_kitchen",
        foot_court_xy=[0.0, -1.0],
        sigma_p_m=0.002,
        metric_confidence="low",
    )

    assert decide_court_boundary(on_near_nvz_line_strip).decision == "kitchen"
    assert decide_court_boundary(just_outside_but_uncertain).decision == "too_close_to_call"
    assert decide_court_boundary(low_confidence_inside).decision == "too_close_to_call"


def test_sideline_and_baseline_decisions_emit_spec_in_out_and_abstain_band() -> None:
    outside_sideline = CourtDecisionInput(
        boundary="sideline",
        foot_court_xy=[3.11, 0.0],
        sigma_p_m=0.005,
        metric_confidence="high",
    )
    inside_baseline = CourtDecisionInput(
        boundary="baseline",
        foot_court_xy=[0.0, 6.65],
        sigma_p_m=0.005,
        metric_confidence="high",
    )
    uncertain_sideline = CourtDecisionInput(
        boundary="sideline",
        foot_court_xy=[3.048 + 0.030, 0.0],
        sigma_p_m=0.010,
        metric_confidence="high",
    )

    outside_call = decide_court_boundary(outside_sideline)
    inside_call = decide_court_boundary(inside_baseline)
    uncertain_call = decide_court_boundary(uncertain_sideline)

    assert outside_call.decision == "out"
    assert outside_call.signed_dist_m > 0.0
    assert inside_call.decision == "in"
    assert inside_call.signed_dist_m < 0.0
    assert uncertain_call.decision == "too_close_to_call"


def test_capture_quality_poor_forces_too_close_to_call_even_with_high_metric_confidence() -> None:
    decision_input = CourtDecisionInput(
        boundary="sideline",
        foot_court_xy=[3.20, 0.0],
        sigma_p_m=0.005,
        metric_confidence="high",
        capture_quality_grade="poor",
    )

    call = decide_court_boundary(decision_input)

    assert call.decision == "too_close_to_call"
    assert call.capture_quality_grade == "poor"


def test_court_escalation_reasons_cover_all_stage_h_failures() -> None:
    gate = CourtGateInput(
        reprojection_p95_px=5.1,
        metric_confidence="med",
        keypoint_inlier_count=9,
        required_line_recovered=False,
        capture_quality_grade="poor",
        drift_or_recalibration=True,
        requested_line_call_decision="too_close_to_call",
    )

    reasons = court_escalation_reasons(gate)

    assert reasons == (
        "court_reprojection_p95_gt_5px",
        "metric_confidence_not_high",
        "keypoint_inlier_count_lt_10",
        "required_line_unrecovered",
        "capture_quality_poor",
        "drift_or_recalibration",
        "line_call_too_close_to_call",
    )

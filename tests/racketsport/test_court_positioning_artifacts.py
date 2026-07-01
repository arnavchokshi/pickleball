from __future__ import annotations

import pytest

from threed.racketsport.court_keypoint_net import PICKLEBALL_KEYPOINTS
from threed.racketsport.court_positioning import CourtBoundaryDecision, solve_metric_court_placement
from threed.racketsport.court_positioning_artifacts import (
    CallsArtifact,
    PlayerGroundArtifact,
    build_call_event,
    build_calls_artifact,
    build_court_keypoints_artifact,
    build_metric_court_calibration_artifact,
    build_player_ground_artifact,
)
from threed.racketsport.schemas import CourtCalibration
from threed.racketsport.player_grounding import GroundedFoot


KEYPOINT_NAMES = (
    "near_left_corner",
    "near_baseline_center",
    "near_right_corner",
    "far_right_corner",
    "far_baseline_center",
    "far_left_corner",
    "near_nvz_left",
    "near_nvz_center",
    "near_nvz_right",
    "net_left_sideline",
    "net_center",
    "net_right_sideline",
    "far_nvz_left",
    "far_nvz_center",
    "far_nvz_right",
)


def test_build_court_keypoints_artifact_emits_stage_b_aggregate_contract() -> None:
    keypoints = {
        name: {
            "xy": [float(index), float(index + 20)],
            "confidence": 0.84,
            "inlier_frames": [0, 15, 29],
            "recovered": name == "far_left_corner",
        }
        for index, name in enumerate(KEYPOINT_NAMES)
    }

    artifact = build_court_keypoints_artifact(
        frame_indexes=[0, 15, 29],
        keypoints=keypoints,
        target_court_score=0.78,
        source="model_aggregate_v1",
    )

    assert artifact["artifact_type"] == "racketsport_court_keypoints"
    assert artifact["coordinate_space"] == "undistorted_source_video_pixels"
    assert len(artifact["keypoints"]) == 15
    assert artifact["keypoints"][5]["name"] == "far_left_corner"
    assert artifact["keypoints"][5]["recovered"] is True


def test_build_court_keypoints_artifact_rejects_missing_canonical_keypoint() -> None:
    keypoints = {
        name: {"xy": [float(index), float(index + 20)], "confidence": 0.84}
        for index, name in enumerate(KEYPOINT_NAMES[:-1])
    }

    with pytest.raises(ValueError, match="15 canonical"):
        build_court_keypoints_artifact(
            frame_indexes=[0, 15, 29],
            keypoints=keypoints,
            target_court_score=0.78,
            source="model_aggregate_v1",
        )


def test_build_metric_court_calibration_artifact_emits_stage_c_spec_fields() -> None:
    world_keypoints = {
        point.name: [point.world_xyz_m[0] + 4.0, point.world_xyz_m[1] - 2.0, 0.0]
        for point in PICKLEBALL_KEYPOINTS
    }
    placement = solve_metric_court_placement(world_keypoints)
    image_keypoints = {
        name: {"uv": [100.0 + index * 3.0, 300.0 + index * 2.0]}
        for index, name in enumerate(placement.solved_keypoints)
    }

    artifact = build_metric_court_calibration_artifact(
        placement=placement,
        intrinsics={"fx": 1000.0, "fy": 1000.0, "cx": 960.0, "cy": 540.0, "dist": [], "source": "arkit"},
        homography=[[1.0, 0.0, 960.0], [0.0, 1.0, 540.0], [0.0, 0.0, 1.0]],
        image_keypoints=image_keypoints,
        extrinsics={
            "R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "t": [0.0, 0.0, 1.6],
            "camera_height_m": 1.6,
        },
        reprojection_error_px={"median": 1.2, "p95": 4.8},
        per_keypoint_residual_px=[1.0] * len(placement.solved_keypoints),
        gsd_model={
            "type": "analytic_ray_plane",
            "plane_sigma_m": 0.012,
            "calibration_sigma_m": 0.018,
            "samples": [{"court_xy": [0.0, -2.1336], "gsd_m_per_px": 0.018, "sigma_p_m": 0.032}],
        },
        capture_quality={"grade": "good", "reasons": []},
        source="arkit_plane_kabsch_ransac_v1",
        solved_over_frames=[0, 15, 29],
    )
    parsed = CourtCalibration.model_validate(artifact)

    assert parsed.coordinate_frame == "court_netcenter_z_up_m"
    assert parsed.metric_confidence == "high"
    assert parsed.T_world_court is not None
    for actual_row, expected_row in zip(parsed.T_world_court, placement.T_world_court, strict=True):
        assert actual_row == pytest.approx(expected_row)
    assert parsed.per_keypoint_residual_px == pytest.approx([1.0] * len(placement.solved_keypoints))
    assert parsed.gsd_model.type == "analytic_ray_plane"
    assert parsed.source == "arkit_plane_kabsch_ransac_v1"
    assert parsed.solved_over_frames == [0, 15, 29]


def test_build_player_ground_artifact_emits_spec_contract_for_both_feet_and_body() -> None:
    left = GroundedFoot(
        side="L",
        court_xy=[-0.25, -2.10],
        world_xyz=[1.0, 2.0, 0.0],
        height_m=0.012,
        contact=True,
        sigma_p_m=0.028,
        confidence=0.91,
        source_points=("ankle", "heel", "toe"),
    )
    right = GroundedFoot(
        side="R",
        court_xy=[0.22, -2.05],
        world_xyz=[1.4, 2.1, 0.0],
        height_m=0.018,
        contact=True,
        sigma_p_m=0.031,
        confidence=0.88,
        source_points=("ankle",),
    )

    artifact = build_player_ground_artifact(
        fps=60.0,
        players=[
            {
                "id": 3,
                "frames": [
                    {
                        "t": 1.25,
                        "feet": [left, right],
                        "root_world": [1.2, 2.05, 0.0],
                        "joints_world": [[1.2, 2.05, 0.9]],
                    }
                ],
            }
        ],
    )

    parsed = PlayerGroundArtifact.model_validate(artifact)

    assert parsed.artifact_type == "racketsport_player_ground"
    assert parsed.players[0].frames[0].feet[0].side == "L"
    assert parsed.players[0].frames[0].feet[0].court_xy == pytest.approx([-0.25, -2.10])
    assert parsed.players[0].frames[0].feet[0].sigma_p_m == pytest.approx(0.028)
    assert parsed.players[0].frames[0].root_world == pytest.approx([1.2, 2.05, 0.0])


def test_player_ground_artifact_requires_both_feet_per_frame() -> None:
    only_left = GroundedFoot(
        side="L",
        court_xy=[-0.25, -2.10],
        world_xyz=[1.0, 2.0, 0.0],
        height_m=0.012,
        contact=True,
        sigma_p_m=0.028,
        confidence=0.91,
        source_points=("ankle",),
    )

    with pytest.raises(ValueError, match="both L and R feet"):
        build_player_ground_artifact(
            fps=60.0,
            players=[{"id": 3, "frames": [{"t": 1.25, "feet": [only_left], "root_world": [1.0, 2.0, 0.0]}]}],
        )


def test_build_calls_artifact_preserves_uncertainty_and_fail_closed_status() -> None:
    decision = CourtBoundaryDecision(
        boundary="sideline",
        decision="out",
        signed_dist_m=0.052,
        sigma_p_m=0.010,
        metric_confidence="high",
        capture_quality_grade="good",
    )
    event = build_call_event(
        t=2.0,
        player_id=3,
        foot="R",
        decision=decision,
        frames=[118, 119, 120],
    )

    artifact = build_calls_artifact([event], source="metric_floor_v1")
    parsed = CallsArtifact.model_validate(artifact)

    assert parsed.artifact_type == "racketsport_court_calls"
    assert parsed.events[0].decision == "out"
    assert parsed.events[0].signed_dist_m == pytest.approx(0.052)
    assert parsed.events[0].sigma_p_m == pytest.approx(0.010)
    assert parsed.summary == {
        "total_events": 1,
        "hard_call_count": 1,
        "too_close_to_call_count": 0,
        "status": "not_gate_verified",
    }


def test_calls_artifact_records_abstain_as_not_hard_call() -> None:
    decision = CourtBoundaryDecision(
        boundary="near_kitchen",
        decision="too_close_to_call",
        signed_dist_m=-0.004,
        sigma_p_m=0.020,
        metric_confidence="low",
        capture_quality_grade="good",
    )

    artifact = build_calls_artifact(
        [build_call_event(t=2.0, player_id=3, foot="L", decision=decision, frames=[118])],
        source="metric_floor_v1",
    )

    parsed = CallsArtifact.model_validate(artifact)
    assert parsed.events[0].decision == "too_close_to_call"
    assert parsed.summary["hard_call_count"] == 0
    assert parsed.summary["too_close_to_call_count"] == 1

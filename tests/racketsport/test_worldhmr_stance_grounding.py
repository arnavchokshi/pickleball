from __future__ import annotations

import pytest

from tests.racketsport.calibration_fixtures import minimal_calibration_image_pts, minimal_calibration_world_pts
from threed.racketsport import worldhmr
from threed.racketsport.schemas import (
    CameraIntrinsics,
    CaptureQuality,
    CourtCalibration,
    CourtExtrinsics,
    ReprojectionError,
)


def _identity_calibration() -> CourtCalibration:
    return CourtCalibration(
        schema_version=1,
        sport="pickleball",
        homography=[[100.0, 0.0, 1000.0], [0.0, 100.0, 1000.0], [0.0, 0.0, 1.0]],
        intrinsics=CameraIntrinsics(fx=1000.0, fy=1000.0, cx=960.0, cy=540.0, dist=[], source="manual"),
        extrinsics=CourtExtrinsics(
            R=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            t=[0.0, 0.0, 0.0],
            camera_height_m=1.5,
        ),
        reprojection_error_px=ReprojectionError(median=0.0, p95=0.0),
        capture_quality=CaptureQuality(grade="good", reasons=[]),
        image_pts=minimal_calibration_image_pts(),
        world_pts=minimal_calibration_world_pts(),
    )


def _sample(frame_idx: int, track_x: float) -> dict:
    return {
        "frame_idx": frame_idx,
        "player_id": 3,
        "t": frame_idx / 30.0,
        "confidence": 0.9,
        "track_world_xy": [track_x, 0.0],
        "joints_camera": [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        "vertices_camera": [[0.2, 0.0, 0.0]],
        "global_orient": [0.0, 0.0, 0.0],
        "body_pose": [0.0, 0.0, 0.0],
        "betas": [0.0],
    }


def test_stance_aware_grounding_anchors_transitions_to_placement_without_speed_clamp() -> None:
    samples = [_sample(0, 0.0), _sample(1, 1.0), _sample(2, 2.0)]
    stance_index = {
        (3, 0): {"stance": True, "velocity": [30.0, 0.0], "covariance_m2": [[0.01, 0.0], [0.0, 0.01]]},
        (3, 1): {"stance": False, "velocity": [30.0, 0.0], "covariance_m2": [[0.01, 0.0], [0.0, 0.01]]},
        (3, 2): {"stance": True, "velocity": [30.0, 0.0], "covariance_m2": [[0.01, 0.0], [0.0, 0.01]]},
    }

    smpl_motion, skeleton3d, metrics = worldhmr.build_body_artifacts_from_fast_sam(
        samples,
        calibration=_identity_calibration(),
        fps=30.0,
        smoothing_alpha=0.1,
        max_root_speed_mps=0.1,
        stance_index=stance_index,
        grounding_anchor_source="placement_track_world_xy",
    )

    frames = smpl_motion["players"][0]["frames"]
    assert [frame["transl_world"][0] for frame in frames] == pytest.approx([0.0, 1.0, 2.0], abs=1e-9)
    assert [frame["joints_world"][0][0] for frame in frames] == pytest.approx([0.0, 1.0, 2.0], abs=1e-9)
    assert [frame["mesh_vertices_world"][0][0] for frame in frames] == pytest.approx([0.2, 1.2, 2.2], abs=1e-9)
    assert metrics["grounding_anchor_source"] == "placement_track_world_xy"
    assert metrics["stance_aware_grounding"]["stance_frame_count"] == 2
    assert metrics["stance_aware_grounding"]["transition_frame_count"] == 1
    assert metrics["stance_aware_grounding"]["transition_anchor_lag_p95_m"] <= 0.10
    assert metrics["stance_aware_grounding"]["transition_anchor_lag_median_m"] <= 0.05
    assert metrics["root_speed_limited_frames"] == 0
    assert skeleton3d["provenance"]["grounding_anchor_source"] == "placement_track_world_xy"


def test_nonfinite_track_world_xy_is_rejected_before_grounding_artifacts() -> None:
    bad_sample = _sample(0, 0.0)
    bad_sample["track_world_xy"] = [float("nan"), 0.0]

    with pytest.raises(ValueError, match="track_world_xy/0 must be finite"):
        worldhmr.build_body_artifacts_from_fast_sam(
            [bad_sample],
            calibration=_identity_calibration(),
            fps=30.0,
        )


def test_stance_aware_grounding_reports_speed_engagement_known_answer() -> None:
    samples = [_sample(0, 0.0), _sample(1, 1.0), _sample(2, 2.0)]
    stance_index = {
        (3, 0): {"stance": True, "velocity": [30.0, 0.0], "covariance_m2": [[0.01, 0.0], [0.0, 0.01]]},
        (3, 1): {"stance": True, "velocity": [30.0, 0.0], "covariance_m2": [[0.01, 0.0], [0.0, 0.01]]},
        (3, 2): {"stance": True, "velocity": [30.0, 0.0], "covariance_m2": [[0.01, 0.0], [0.0, 0.01]]},
    }

    _smpl_motion, _skeleton3d, metrics = worldhmr.build_body_artifacts_from_fast_sam(
        samples,
        calibration=_identity_calibration(),
        fps=30.0,
        max_root_speed_mps=3.0,
        stance_index=stance_index,
        grounding_anchor_source="placement_track_world_xy",
    )

    assert metrics["root_speed_limited_frames"] == 0
    assert metrics["root_speed_anomaly_frames"] == 2
    assert metrics["root_speed_anomaly_fraction_overall"] == pytest.approx(1.0)
    assert metrics["root_speed_clamp_engagement_overall"] == pytest.approx(1.0)
    assert metrics["root_speed_clamp_engagement_by_player"] == {"3": pytest.approx(1.0)}
    assert metrics["stance_aware_grounding"]["root_speed_clamp_engagement_overall"] == pytest.approx(1.0)
    assert metrics["stance_aware_grounding"]["root_speed_clamp_engagement_by_player"] == {"3": pytest.approx(1.0)}


def test_stance_aware_smoother_uses_phase_median_anchor_during_stance() -> None:
    frames = [
        {
            "frame_idx": frame_idx,
            "player_id": 3,
            "t": frame_idx / 30.0,
            "track_world_xy": [track_x, 0.0],
            "transl_world": [track_x, 0.0, 0.0],
            "joints_world": [[track_x, 0.0, 0.0], [track_x, 0.0, 1.0]],
            "vertices_world": [[track_x + 0.2, 0.0, 0.0]],
            "grounding_anchor": "low_joint_cluster",
        }
        for frame_idx, track_x in enumerate([0.00, 0.12, 0.24])
    ]
    stance_index = {
        (3, 0): {"stance": True, "phase_id": "3:left:0", "phase_foot": "left"},
        (3, 1): {"stance": True, "phase_id": "3:left:0", "phase_foot": "left"},
        (3, 2): {"stance": True, "phase_id": "3:left:0", "phase_foot": "left"},
    }

    smoothed, metrics = worldhmr._smooth_grounded_frames_stance_aware(
        frames,
        stance_index=stance_index,
        fps=30.0,
        max_root_speed_mps=None,
        max_track_anchor_smoothing_residual_m=None,
    )

    assert [frame["transl_world"][0] for frame in smoothed] == pytest.approx([0.12, 0.12, 0.12])
    assert [frame["joints_world"][0][0] for frame in smoothed] == pytest.approx([0.12, 0.12, 0.12])
    assert [frame["vertices_world"][0][0] for frame in smoothed] == pytest.approx([0.32, 0.32, 0.32])
    assert metrics["stance_aware_grounding"]["phase_median_anchor_frame_count"] == 3

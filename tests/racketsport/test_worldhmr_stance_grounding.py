from __future__ import annotations

import pytest

from tests.racketsport.test_foot_contact import JOINT_NAMES_65, _frame as _foot_frame
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


def _eligible_stance_info(*, stance: bool = True, phase_id: str = "3:left:0") -> dict:
    info = {
        "stance": bool(stance),
        "velocity": [30.0, 0.0],
        "covariance_m2": [[0.01, 0.0], [0.0, 0.01]],
    }
    if stance:
        info.update(
            {
                "phase_id": phase_id,
                "phase_foot": "left",
                "foot_assignment": "per_foot_keypoint_support",
                "source_phase_foot": "left",
                "min_confidence": 0.95,
                "max_height_m": 0.0,
                "max_speed_mps": 0.10,
                "source_thresholds": {"min_confidence": 0.90},
                "assignment_evidence": {"body_detector_agreement": 0.95},
            }
        )
    return info


def test_stance_aware_grounding_keeps_placement_target_without_breaking_speed_cap() -> None:
    samples = [_sample(0, 0.0), _sample(1, 1.0), _sample(2, 2.0)]
    stance_index = {
        (3, 0): _eligible_stance_info(phase_id="3:left:0"),
        (3, 1): _eligible_stance_info(stance=False),
        (3, 2): _eligible_stance_info(phase_id="3:left:2"),
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
    expected_root_x = [0.0, 0.1 / 30.0, 0.2 / 30.0]
    assert [frame["transl_world"][0] for frame in frames] == pytest.approx(expected_root_x, abs=1e-9)
    assert [frame["joints_world"][0][0] for frame in frames] == pytest.approx(expected_root_x, abs=1e-9)
    assert [frame["mesh_vertices_world"][0][0] for frame in frames] == pytest.approx(
        [value + 0.2 for value in expected_root_x], abs=1e-9
    )
    assert metrics["grounding_anchor_source"] == "placement_track_world_xy"
    assert metrics["stance_aware_grounding"]["stance_frame_count"] == 2
    assert metrics["stance_aware_grounding"]["transition_frame_count"] == 1
    assert metrics["stance_aware_grounding"]["transition_anchor_lag_p95_m"] > 0.90
    assert metrics["root_speed_limited_frames"] == 2
    assert all(
        frame.get("temporal_smoothing_metadata", {}).get("root_speed_limited") is True
        for frame in frames[1:]
    )
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
        (3, 0): _eligible_stance_info(),
        (3, 1): _eligible_stance_info(),
        (3, 2): _eligible_stance_info(),
    }

    smpl_motion, _skeleton3d, metrics = worldhmr.build_body_artifacts_from_fast_sam(
        samples,
        calibration=_identity_calibration(),
        fps=30.0,
        max_root_speed_mps=3.0,
        stance_index=stance_index,
        grounding_anchor_source="placement_track_world_xy",
    )

    frames = smpl_motion["players"][0]["frames"]
    assert [frame["transl_world"][0] for frame in frames] == pytest.approx([0.0, 0.1, 0.2])
    assert metrics["root_speed_limited_frames"] == 2
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
        (3, 0): _eligible_stance_info(phase_id="3:left:0"),
        (3, 1): _eligible_stance_info(phase_id="3:left:0"),
        (3, 2): _eligible_stance_info(phase_id="3:left:0"),
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


def test_stance_aware_grounding_rejects_unqualified_placement_stance_rows() -> None:
    samples = [_sample(0, 0.0), _sample(1, 1.0), _sample(2, 2.0)]
    stance_index = {
        (3, 0): {"stance": True, "source": "placement.json"},
        (3, 1): {"stance": True, "source": "placement.json"},
        (3, 2): {"stance": False, "source": "placement.json"},
    }

    _smpl_motion, _skeleton3d, metrics = worldhmr.build_body_artifacts_from_fast_sam(
        samples,
        calibration=_identity_calibration(),
        fps=30.0,
        smoothing_alpha=0.1,
        max_root_speed_mps=0.1,
        stance_index=stance_index,
        grounding_anchor_source="placement_track_world_xy",
    )

    assert metrics["stance_aware_grounding"]["stance_frame_count"] == 0
    assert metrics["stance_aware_grounding"]["rejected_stance_frame_count"] == 2
    assert metrics["stance_aware_grounding"]["rejected_stance_reasons"] == {"missing_per_foot_assignment": 2}


def test_contact_gate_stream_rows_include_phase_and_frame_provenance() -> None:
    contact_frames = [
        _foot_frame(0, left_x=0.000, left_z=0.000),
        _foot_frame(1, left_x=0.004, left_z=0.000),
        _foot_frame(2, left_x=0.008, left_z=0.000),
    ]
    skeleton3d = {
        "artifact_type": "racketsport_skeleton3d",
        "fps": 30.0,
        "joint_names": list(JOINT_NAMES_65),
        "players": [
            {
                "id": "p1",
                "frames": [
                    {
                        "frame_idx": frame.frame_index,
                        "t": frame.t,
                        "joints_world": frame.joints_world,
                        "joint_conf": frame.joint_conf,
                        "transl_world": [0.0, 0.0, 0.0],
                        "track_world_xy": [0.0, 0.0],
                        "output_source": "unit_test",
                    }
                    for frame in contact_frames
                ],
            }
        ],
    }

    metrics, gate_stream = worldhmr._contact_gate_stream_for_skeleton3d(
        skeleton3d,
        clip="unit_clip",
        threshold_m=0.03,
    )

    assert metrics["phase_metrics"]
    assert gate_stream["artifact_type"] == "foot_lock_gate_stream"
    assert gate_stream["phase_rows"][0]["clip"] == "unit_clip"
    assert gate_stream["phase_rows"][0]["foot_assignment"] == "per_foot_body_contact"
    assert gate_stream["frame_rows"][0]["selected_foot"] in {"left", "right"}
    assert gate_stream["summary"]["frame_row_stride"] == 1
    assert gate_stream["artifact_size_policy"]["max_bytes"] == 20_000_000


def test_real_pipeline_gate_metric_keeps_overthreshold_lock_rows_in_metric() -> None:
    contact_frames = [
        _foot_frame(0, left_x=0.000, left_z=0.000),
        _foot_frame(1, left_x=0.010, left_z=0.000),
        _foot_frame(2, left_x=0.020, left_z=0.000),
        _foot_frame(3, left_x=0.030, left_z=0.000),
        _foot_frame(4, left_x=0.040, left_z=0.000),
    ]
    skeleton3d = {
        "artifact_type": "racketsport_skeleton3d",
        "fps": 30.0,
        "joint_names": list(JOINT_NAMES_65),
        "provenance": {"refined_stance_phase_lock": {"source": "unit_test"}},
        "players": [
            {
                "id": "p1",
                "frames": [
                    {
                        "frame_idx": frame.frame_index,
                        "t": frame.t,
                        "joints_world": frame.joints_world,
                        "joint_conf": frame.joint_conf,
                        "transl_world": [0.0, 0.0, 0.0],
                        "track_world_xy": [0.0, 0.0],
                        "output_source": "unit_test",
                    }
                    for frame in contact_frames
                ],
            }
        ],
    }

    metrics, gate_stream = worldhmr._contact_gate_stream_for_skeleton3d(
        skeleton3d,
        clip="unit_clip",
        threshold_m=0.03,
    )

    assert len(metrics["phase_metrics"]) == 1
    assert metrics["phase_metrics"][0]["slide_mm"] == pytest.approx(40.0)
    assert metrics["max_candidate_phase_slide_m"] == pytest.approx(0.04)
    assert metrics["candidate_phase_rejection_reason_counts"] == {}
    assert gate_stream["summary"]["top_20_phases_by_slide_m"][0]["slide_m"] == pytest.approx(0.04)
    assert gate_stream["summary"]["max_candidate_phase_slide_m"] == pytest.approx(0.04)
    assert gate_stream["summary"]["candidate_phase_rejection_reason_counts"] == {}
    assert gate_stream["summary"]["weak_rejection_reasons"] == {}
    assert gate_stream["phase_rows"][0]["lock_metric_included"] is True
    assert gate_stream["phase_rows"][0]["demoted"] is False
    assert gate_stream["phase_rows"][0]["rejection_reason"] is None

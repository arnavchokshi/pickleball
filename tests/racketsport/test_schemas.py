from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from threed.racketsport.schemas import (
    BallFrame,
    CallsArtifact,
    CaptureSidecar,
    CourtCalibration,
    CourtLineEvidence,
    CourtKeypoints,
    DriftLog,
    ContactWindows,
    ContactWindowCandidates,
    BallTrack,
    MetricValue,
    PlayerGroundArtifact,
    RacketCandidates,
    PhaseEvalMetrics,
    RacketPose,
    ReprojectionError,
    Skeleton3D,
    SmplMotion,
    TrackFrame,
    Tracks,
    VirtualWorld,
    validate_artifact_file,
)


def test_capture_sidecar_schema_accepts_documented_payload(tmp_path):
    payload = {
        "schema_version": 1,
        "device_tier": "B_standard",
        "device_model": "iPhone16,2",
        "fps": 120,
        "format": "hevc",
        "resolution": [1080, 1920],
        "orientation": "portrait",
        "capture_device_orientation": "portrait",
        "video_rotation_angle_degrees": 0,
        "recording_started_at": "2026-06-28T20:30:00Z",
        "recording_duration_s": 4.5,
        "camera_position": "back",
        "camera_lens": "wide",
        "locked": {
            "exposure_s": 0.001,
            "iso": 320,
            "focus": 0.7,
            "wb_locked": True,
        },
        "intrinsics": {
            "fx": 1000.0,
            "fy": 1000.0,
            "cx": 960.0,
            "cy": 540.0,
            "dist": [0.0, 0.0, 0.0, 0.0],
            "source": "arkit",
        },
        "arkit_camera_pose": {
            "R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "t": [0.0, 1.5, 0.0],
        },
        "court_plane": {"point": [0.0, 0.0, 0.0], "normal": [0.0, 1.0, 0.0]},
        "manual_court_taps": [[10.0, 10.0], [100.0, 10.0], [100.0, 80.0], [10.0, 80.0]],
        "gravity": [0.0, -1.0, 0.0],
        "lidar_depth_refs": [],
        "ondevice_pose_track": "ondevice_pose.json",
        "capture_quality": {"grade": "good", "reasons": []},
    }
    path = tmp_path / "capture_sidecar.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    parsed = validate_artifact_file("capture_sidecar", path)

    assert isinstance(parsed, CaptureSidecar)
    assert parsed.fps == 120
    assert parsed.orientation == "portrait"
    assert parsed.capture_device_orientation == "portrait"
    assert parsed.video_rotation_angle_degrees == 0
    assert parsed.recording_duration_s == 4.5
    assert parsed.capture_quality.grade == "good"


def test_court_calibration_requires_current_schema_version():
    payload = {
        "schema_version": 2,
        "sport": "pickleball",
        "homography": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "intrinsics": {"fx": 1, "fy": 1, "cx": 0, "cy": 0, "dist": [], "source": "manual"},
        "extrinsics": {"R": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "t": [0, 0, 0], "camera_height_m": 1.4},
        "reprojection_error_px": {"median": 2.0, "p95": 5.0},
        "capture_quality": {"grade": "good", "reasons": []},
        "image_pts": [[0, 0], [1, 0], [1, 1], [0, 1]],
        "world_pts": [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
    }

    with pytest.raises(ValidationError):
        CourtCalibration.model_validate(payload)


def test_court_calibration_accepts_metric_stage_c_spec_fields(tmp_path: Path) -> None:
    payload = {
        "schema_version": 1,
        "sport": "pickleball",
        "coordinate_frame": "court_netcenter_z_up_m",
        "T_world_court": [
            [1.0, 0.0, 0.0, 4.0],
            [0.0, 1.0, 0.0, -2.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "homography": [[1.0, 0.0, 960.0], [0.0, 1.0, 540.0], [0.0, 0.0, 1.0]],
        "intrinsics": {"fx": 1000.0, "fy": 1000.0, "cx": 960.0, "cy": 540.0, "dist": [], "source": "arkit"},
        "extrinsics": {
            "R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "t": [0.0, 0.0, 1.6],
            "camera_height_m": 1.6,
        },
        "reprojection_error_px": {"median": 1.2, "p95": 4.8},
        "per_keypoint_residual_px": [1.0] * 15,
        "metric_confidence": "high",
        "gsd_model": {
            "type": "analytic_ray_plane",
            "plane_sigma_m": 0.012,
            "calibration_sigma_m": 0.018,
            "samples": [{"court_xy": [0.0, -2.1336], "gsd_m_per_px": 0.018, "sigma_p_m": 0.032}],
        },
        "capture_quality": {"grade": "good", "reasons": []},
        "image_pts": [[100.0, 300.0], [900.0, 300.0], [500.0, 300.0], [500.0, 180.0]],
        "world_pts": [[0.0, 7.0, 0.0], [20.0, 7.0, 0.0], [10.0, 7.0, 0.0], [10.0, 22.0, 0.0]],
        "source": "arkit_plane_kabsch_ransac_v1",
        "solved_over_frames": [0, 15, 29],
    }
    path = tmp_path / "court_calibration.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    parsed = validate_artifact_file("court_calibration", path)

    assert isinstance(parsed, CourtCalibration)
    assert parsed.coordinate_frame == "court_netcenter_z_up_m"
    assert parsed.T_world_court[0][3] == pytest.approx(4.0)
    assert parsed.metric_confidence == "high"
    assert parsed.gsd_model.samples[0].sigma_p_m == pytest.approx(0.032)
    assert parsed.solved_over_frames == [0, 15, 29]

    payload["solved_over_frames"] = [0, -1, 29]
    with pytest.raises(ValidationError, match="solved_over_frames"):
        CourtCalibration.model_validate(payload)


def test_common_vector_fields_reject_wrong_dimensions() -> None:
    calibration_payload = {
        "schema_version": 1,
        "sport": "pickleball",
        "homography": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "intrinsics": {"fx": 1, "fy": 1, "cx": 0, "cy": 0, "dist": [], "source": "manual"},
        "extrinsics": {"R": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "t": [0, 0, 0], "camera_height_m": 1.4},
        "reprojection_error_px": {"median": 2.0, "p95": 5.0},
        "capture_quality": {"grade": "good", "reasons": []},
        "image_pts": [[0, 0], [1, 0], [1, 1], [0, 1]],
        "world_pts": [[0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
    }
    with pytest.raises(ValidationError):
        CourtCalibration.model_validate(calibration_payload)

    ball_payload = {
        "schema_version": 1,
        "fps": 60.0,
        "source": "tap",
        "frames": [{"t": 0.0, "xy": [10.0], "conf": 0.9, "visible": True, "world_xyz": [0.0, 0.0]}],
    }
    with pytest.raises(ValidationError):
        BallTrack.model_validate(ball_payload)

    candidates_payload = {
        "schema_version": 1,
        "artifact_type": "racketsport_contact_window_candidates",
        "clip": "clip-a",
        "fps": 60.0,
        "source_event_path": "labels/events.json",
        "not_gate_verified": True,
        "trusted_for_body": False,
        "promotion_target": "contact_windows.json",
        "candidates": [
            {
                "review_id": "contact-001",
                "type": "contact",
                "frame": 10,
                "t": 0.167,
                "xy_px": [100.0, 200.0, 300.0],
                "source_label": "manual",
                "source_status": "draft",
                "source_confidence": 0.7,
                "candidate_confidence": 0.7,
                "window": {"t0": 0.1, "t1": 0.2, "importance": 0.7},
            }
        ],
        "summary": {
            "candidate_count": 1,
            "rejected_item_count": 0,
            "by_type": {"contact": 1},
            "by_status": {"draft": 1},
            "uncertainty_flags": [],
        },
    }
    with pytest.raises(ValidationError):
        ContactWindowCandidates.model_validate(candidates_payload)


def test_common_numeric_fields_reject_bool_nan_and_infinity() -> None:
    for payload in (
        {"median": float("nan"), "p95": 1.0},
        {"median": 1.0, "p95": float("inf")},
    ):
        with pytest.raises(ValidationError):
            ReprojectionError.model_validate(payload)

    for payload in (
        {"t": True, "bbox": [0.0, 0.0, 10.0, 20.0], "world_xy": [0.0, 0.0], "conf": 0.8},
        {"t": 0.0, "bbox": [0.0, 0.0, 10.0, 20.0], "world_xy": [float("nan"), 0.0], "conf": 0.8},
        {"t": 0.0, "bbox": [0.0, 0.0, 10.0, 20.0], "world_xy": [0.0, 0.0], "conf": float("nan")},
    ):
        with pytest.raises(ValidationError):
            TrackFrame.model_validate(payload)


def test_track_frame_bbox_rejects_inverted_xyxy_boxes() -> None:
    with pytest.raises(ValidationError, match="bbox must be ordered"):
        TrackFrame.model_validate({"t": 0.0, "bbox": [10.0, 20.0, 5.0, 40.0], "world_xy": [0.0, 0.0], "conf": 0.8})


def test_smpl_motion_frame_accepts_raw_track_anchor() -> None:
    parsed = SmplMotion.model_validate(
        {
            "schema_version": 1,
            "model": "smplx",
            "fps": 30.0,
            "world_frame": "court_Z0",
            "players": [
                {
                    "id": 1,
                    "betas": [0.0] * 10,
                    "frames": [
                        {
                            "t": 0.0,
                            "global_orient": [0.0, 0.0, 0.0],
                            "body_pose": [0.0] * 63,
                            "left_hand_pose": [],
                            "right_hand_pose": [],
                            "transl_world": [0.0, 1.0, 0.0],
                            "track_world_xy": [1.25, -2.5],
                            "joints_world": [[0.0, 1.0, 0.0]],
                            "mesh_vertices_world": [],
                            "joint_conf": [0.9],
                            "foot_contact": {"left": False, "right": False},
                            "grf": None,
                        }
                    ],
                    "skate_free": True,
                    "physics": "pending",
                }
            ],
        }
    )

    assert parsed.players[0].frames[0].track_world_xy == [1.25, -2.5]


def test_skeleton3d_accepts_real_lane_a_world_joints() -> None:
    parsed = Skeleton3D.model_validate(
        {
            "schema_version": 1,
            "artifact_type": "racketsport_skeleton3d",
            "joint_names": ["pelvis", "left_wrist", "right_wrist"],
            "preview_only": False,
            "players": [
                {
                    "id": 1,
                    "frames": [
                        {
                            "frame_idx": 0,
                            "t": 0.0,
                            "joints_world": [
                                [0.0, 0.0, 0.9],
                                [-0.35, 0.1, 1.2],
                                [0.35, 0.1, 1.2],
                            ],
                            "joint_conf": [0.99, 0.95, 0.94],
                        }
                    ],
                }
            ],
        }
    )

    assert parsed.preview_only is False


def test_smpl_motion_accepts_sam3dbody_world_joint_representation_without_grf() -> None:
    parsed = SmplMotion.model_validate(
        {
            "schema_version": 1,
            "model": "sam3dbody_world_joints",
            "fps": 30.0,
            "world_frame": "court_Z0",
            "mesh_faces": [[0, 1, 2]],
            "players": [
                {
                    "id": 1,
                    "betas": [],
                    "frames": [
                        {
                            "t": 0.0,
                            "global_orient": [0.0, 0.0, 0.0],
                            "body_pose": [],
                            "left_hand_pose": [],
                            "right_hand_pose": [],
                            "transl_world": [0.0, 1.0, 0.0],
                            "track_world_xy": [0.0, 1.0],
                            "joints_world": [[0.0, 1.0, 0.0]],
                            "mesh_vertices_world": [],
                            "joint_conf": [0.9],
                            "foot_contact": {"left": True, "right": False},
                            "grf": None,
                        }
                    ],
                    "skate_free": False,
                    "physics": "worldhmr_floor_contact_observation_only",
                }
            ],
        }
    )

    assert parsed.model == "sam3dbody_world_joints"
    assert parsed.mesh_faces == [(0, 1, 2)]
    assert parsed.players[0].skate_free is False
    assert parsed.players[0].frames[0].grf is None


    for payload in (
        {"t": True, "xy": [10.0, 20.0], "conf": 0.9, "visible": True},
        {"t": 0.0, "xy": [10.0, 20.0], "conf": float("nan"), "visible": True},
        {"t": 0.0, "xy": [10.0, 20.0], "conf": 0.9, "visible": True, "world_xyz": [0.0, float("inf"), 0.0]},
    ):
        with pytest.raises(ValidationError):
            BallFrame.model_validate(payload)

    with pytest.raises(ValidationError):
        MetricValue.model_validate({"value": 1.0, "conf": float("nan")})


def test_calibration_artifact_rejects_nonfinite_negative_and_empty_geometry_fields() -> None:
    base_payload = {
        "schema_version": 1,
        "sport": "pickleball",
        "homography": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "intrinsics": {"fx": 1000.0, "fy": 1000.0, "cx": 960.0, "cy": 540.0, "dist": [], "source": "manual"},
        "extrinsics": {"R": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "t": [0, 0, 0], "camera_height_m": 1.4},
        "reprojection_error_px": {"median": 2.0, "p95": 5.0},
        "capture_quality": {"grade": "good", "reasons": []},
        "image_pts": [[0, 0], [1, 0], [1, 1], [0, 1]],
        "world_pts": [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
    }

    invalid_payloads = []
    payload = json.loads(json.dumps(base_payload))
    payload["intrinsics"]["fx"] = -1.0
    invalid_payloads.append(payload)
    payload = json.loads(json.dumps(base_payload))
    payload["intrinsics"]["fx"] = float("nan")
    invalid_payloads.append(payload)
    payload = json.loads(json.dumps(base_payload))
    payload["reprojection_error_px"]["median"] = -0.1
    invalid_payloads.append(payload)
    payload = json.loads(json.dumps(base_payload))
    payload["extrinsics"]["camera_height_m"] = 0.0
    invalid_payloads.append(payload)
    payload = json.loads(json.dumps(base_payload))
    payload["image_pts"] = []
    payload["world_pts"] = []
    invalid_payloads.append(payload)
    payload = json.loads(json.dumps(base_payload))
    payload["image_pts"] = [[0, 0], [1, 0], [1, 1], [0, 1]]
    payload["world_pts"] = [[0, 0, 0], [1, 0, 0], [1, 1, 0]]
    invalid_payloads.append(payload)

    for invalid in invalid_payloads:
        with pytest.raises(ValidationError):
            CourtCalibration.model_validate(invalid)


def test_court_line_evidence_rejects_ready_aggregate_without_backing_observations() -> None:
    payload = {
        "schema_version": 1,
        "sport": "pickleball",
        "source": "auto_hough_template",
        "line_observations": [],
        "keypoint_observations": [],
        "net_observations": [],
        "aggregate": {
            "accepted_line_ids": ["near_nvz", "far_nvz", "near_centerline", "far_centerline"],
            "rejected_line_ids": [],
            "missing_required_line_ids": [],
            "missing_required_net_ids": [],
            "mean_residual_px": 2.0,
            "p95_residual_px": 4.0,
            "temporal_stability_px": 3.0,
            "auto_calibration_ready": True,
            "reasons": [],
        },
    }

    with pytest.raises(ValidationError, match="ready court_line_evidence"):
        CourtLineEvidence.model_validate(payload)


def test_validate_artifact_file_rejects_unknown_artifact(tmp_path):
    path = tmp_path / "unknown.json"
    path.write_text("{}", encoding="utf-8")

    with pytest.raises(KeyError):
        validate_artifact_file("not_real", path)


def test_phase_eval_metrics_schema_is_registered(tmp_path):
    payload = {
        "schema_version": 1,
        "phase": "phase1",
        "evaluator": "calib_eval",
        "root": "runs/phase1",
        "labels_root": "data/testclips",
        "status": "blocked",
        "required_artifacts": ["court_calibration.json", "court_zones.json", "net_plane.json"],
        "summary": {
            "total_clips": 1,
            "ready_clips": 0,
            "evaluated_clips": 0,
            "passed_clips": 0,
            "failed_clips": 0,
            "blocked_clips": 1,
        },
        "metrics": {
            "artifact_readiness": {
                "value": False,
                "unit": None,
                "gate": "artifact_check.all_required_artifacts_exist",
                "passed": False,
                "status": "measured",
            }
        },
        "clips": [
            {
                "clip": "clip_001",
                "run_dir": "runs/phase1/clip_001",
                "labels_dir": "data/testclips/clip_001/labels",
                "status": "blocked",
                "missing_label_files": ["events.json"],
                "missing_artifacts": ["court_calibration.json"],
                "metrics": {},
                "notes": [],
            }
        ],
        "notes": ["DATA-1 labels are incomplete"],
    }
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    parsed = validate_artifact_file("phase_eval_metrics", path)

    assert isinstance(parsed, PhaseEvalMetrics)
    assert parsed.status == "blocked"
    assert parsed.clips[0].missing_artifacts == ["court_calibration.json"]


def test_player_ground_schema_is_registered_and_requires_both_feet(tmp_path: Path) -> None:
    payload = {
        "schema_version": 1,
        "artifact_type": "racketsport_player_ground",
        "fps": 60.0,
        "source": "metric_floor_pose_grounding_v1",
        "not_gate_verified": True,
        "players": [
            {
                "id": 1,
                "frames": [
                    {
                        "t": 0.5,
                        "feet": [
                            {
                                "side": "L",
                                "court_xy": [-0.25, -2.1],
                                "height_m": 0.012,
                                "contact": True,
                                "sigma_p_m": 0.028,
                                "confidence": 0.91,
                                "world_xyz": [1.0, 2.0, 0.0],
                                "source_points": ["ankle", "heel", "toe"],
                            },
                            {
                                "side": "R",
                                "court_xy": [0.22, -2.05],
                                "height_m": 0.018,
                                "contact": True,
                                "sigma_p_m": 0.031,
                                "confidence": 0.88,
                                "world_xyz": [1.4, 2.1, 0.0],
                                "source_points": ["ankle"],
                            },
                        ],
                        "root_world": [1.2, 2.05, 0.0],
                        "joints_world": [[1.2, 2.05, 0.9]],
                    }
                ],
            }
        ],
    }
    path = tmp_path / "player_ground.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    parsed = validate_artifact_file("player_ground", path)

    assert isinstance(parsed, PlayerGroundArtifact)
    assert parsed.players[0].frames[0].feet[0].court_xy == pytest.approx([-0.25, -2.1])

    payload["players"][0]["frames"][0]["feet"] = payload["players"][0]["frames"][0]["feet"][:1]
    with pytest.raises(ValidationError, match="both L and R feet"):
        PlayerGroundArtifact.model_validate(payload)


def test_court_calls_schema_is_registered_and_preserves_uncertainty(tmp_path: Path) -> None:
    payload = {
        "schema_version": 1,
        "artifact_type": "racketsport_court_calls",
        "source": "metric_floor_v1",
        "not_gate_verified": True,
        "events": [
            {
                "t": 2.0,
                "player_id": 3,
                "foot": "R",
                "boundary": "sideline",
                "decision": "out",
                "signed_dist_m": 0.052,
                "sigma_p_m": 0.010,
                "frames": [118, 119, 120],
                "metric_confidence": "high",
                "capture_quality_grade": "good",
            },
            {
                "t": 2.2,
                "player_id": 3,
                "foot": "L",
                "boundary": "kitchen",
                "decision": "too_close_to_call",
                "signed_dist_m": -0.004,
                "sigma_p_m": 0.020,
                "frames": [132],
                "metric_confidence": "low",
                "capture_quality_grade": "good",
            },
        ],
        "summary": {
            "total_events": 2,
            "hard_call_count": 1,
            "too_close_to_call_count": 1,
            "status": "not_gate_verified",
        },
    }
    path = tmp_path / "calls.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    parsed = validate_artifact_file("court_calls", path)

    assert isinstance(parsed, CallsArtifact)
    assert parsed.events[0].signed_dist_m == pytest.approx(0.052)
    assert parsed.events[1].decision == "too_close_to_call"

    payload["events"][0]["frames"] = []
    with pytest.raises(ValidationError):
        CallsArtifact.model_validate(payload)


def test_drift_log_schema_is_registered_and_validates_recalibration_spans(tmp_path: Path) -> None:
    payload = {
        "schema_version": 1,
        "artifact_type": "racketsport_drift_log",
        "checks": [
            {"frame": 15, "p95_px": 8.5, "tripped": False},
            {"frame": 30, "p95_px": 8.4, "tripped": False},
            {"frame": 45, "p95_px": 8.6, "tripped": True},
        ],
        "recalibrations": [
            {
                "from_frame": 15,
                "to_frame": 45,
                "reason": "reprojection_drift_3_consecutive",
            }
        ],
    }
    path = tmp_path / "drift_log.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    parsed = validate_artifact_file("drift_log", path)

    assert isinstance(parsed, DriftLog)
    assert parsed.recalibrations[0].from_frame == 15

    payload["recalibrations"][0]["to_frame"] = 14
    with pytest.raises(ValidationError, match="to_frame"):
        DriftLog.model_validate(payload)


def test_court_keypoints_schema_is_registered_and_requires_full_15_point_contract(tmp_path: Path) -> None:
    names = [
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
    ]
    payload = {
        "schema_version": 1,
        "artifact_type": "racketsport_court_keypoints",
        "frame_indexes": [0, 15, 29],
        "coordinate_space": "undistorted_source_video_pixels",
        "keypoints": [
            {
                "name": name,
                "uv": [float(index), float(index + 10)],
                "confidence": 0.91,
                "inlier_frames": [0, 15, 29],
                "recovered": False,
            }
            for index, name in enumerate(names)
        ],
        "target_court_score": 0.82,
        "source": "model_aggregate_v1",
        "not_gate_verified": True,
    }
    path = tmp_path / "court_keypoints.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    parsed = validate_artifact_file("court_keypoints", path)

    assert isinstance(parsed, CourtKeypoints)
    assert parsed.keypoints[0].name == "near_left_corner"
    assert parsed.keypoints[0].uv == pytest.approx([0.0, 10.0])

    payload["keypoints"] = payload["keypoints"][:-1]
    with pytest.raises(ValidationError, match="exactly the 15 canonical"):
        CourtKeypoints.model_validate(payload)


def test_contact_windows_rejects_out_of_range_human_review_source() -> None:
    payload = {
        "schema_version": 1,
        "events": [
            {
                "type": "contact",
                "t": 1.0,
                "frame": 60,
                "player_id": 1,
                "confidence": 0.9,
                "sources": {
                    "audio": 0.0,
                    "wrist_vel": 0.0,
                    "ball_inflection": 0.0,
                    "human_review": 1.5,
                },
                "window": {"t0": 0.92, "t1": 1.08, "importance": 0.9},
            }
        ],
    }

    with pytest.raises(ValidationError):
        ContactWindows.model_validate(payload)


def test_court_line_evidence_schema_validates_semantic_lines_and_net(tmp_path):
    payload = {
        "schema_version": 1,
        "sport": "pickleball",
        "source": "auto_hough_template",
        "line_observations": [
            {
                "line_id": "near_nvz",
                "image_segment": [[100.0, 320.0], [900.0, 320.0]],
                "confidence": 0.91,
                "frame_indexes": [1, 2, 3],
                "residual_px": {"mean": 1.2, "p95": 2.4},
                "visible_fraction": 0.86,
                "source": "hough",
            }
        ],
        "keypoint_observations": [
            {
                "name": "near_baseline_center",
                "image_xy": [500.0, 700.0],
                "confidence": 0.88,
                "frame_indexes": [1, 2, 3],
                "source": "line_intersection",
            }
        ],
        "net_observations": [
            {
                "net_id": "top_net",
                "image_points": [[100.0, 250.0], [500.0, 245.0], [900.0, 250.0]],
                "confidence": 0.84,
                "frame_indexes": [1, 2, 3],
                "residual_px": {"mean": 2.1, "p95": 3.5},
                "source": "net_top_roi",
            }
        ],
        "aggregate": {
            "accepted_line_ids": ["near_nvz"],
            "rejected_line_ids": ["far_centerline"],
            "missing_required_line_ids": ["far_centerline"],
            "mean_residual_px": 1.2,
            "p95_residual_px": 3.5,
            "temporal_stability_px": 0.8,
            "auto_calibration_ready": False,
            "reasons": ["missing_far_centerline"],
        },
    }
    path = tmp_path / "court_line_evidence.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    parsed = validate_artifact_file("court_line_evidence", path)

    assert isinstance(parsed, CourtLineEvidence)
    assert parsed.aggregate.auto_calibration_ready is False
    assert parsed.line_observations[0].line_id == "near_nvz"
    assert parsed.net_observations[0].image_points[1] == pytest.approx([500.0, 245.0])


def test_racket_pose_schema_records_world_frame_units_source_and_projection_error():
    payload = {
        "schema_version": 1,
        "fps": 120.0,
        "world_frame": "court_Z0",
        "translation_unit": "m",
        "players": [
            {
                "id": 1,
                "paddle_dims_in": {"length": 15.5, "width": 7.5},
                "frames": [
                    {
                        "t": 0.0,
                        "pose_se3": {
                            "R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                            "t": [0.4, -0.2, 1.1],
                        },
                        "conf": 0.91,
                        "world_frame": "court_Z0",
                        "translation_unit": "m",
                        "source": "pnp_ippe:court_Z0",
                        "reprojection_error_px": 1.7,
                        "ambiguous": False,
                    }
                ],
                "contacts": [],
            }
        ],
    }

    parsed = RacketPose.model_validate(payload)

    assert parsed.world_frame == "court_Z0"
    assert parsed.translation_unit == "m"
    frame = parsed.players[0].frames[0]
    assert frame.world_frame == "court_Z0"
    assert frame.translation_unit == "m"
    assert frame.source == "pnp_ippe:court_Z0"
    assert frame.reprojection_error_px == pytest.approx(1.7)


def test_racket_candidates_schema_is_registered_and_validates_four_corner_inputs(tmp_path):
    payload = {
        "schema_version": 1,
        "artifact_type": "racketsport_racket_candidates",
        "fps": 120.0,
        "players": [
            {
                "id": 1,
                "paddle_dims_in": {"length": 15.5, "width": 7.5},
                "frames": [
                    {
                        "t": 0.0,
                        "corners_px": [[10.0, 20.0], [30.0, 21.0], [31.0, 70.0], [11.0, 69.0]],
                        "conf": 0.82,
                        "source": "manual_four_corners",
                    }
                ],
            }
        ],
    }
    path = tmp_path / "racket_candidates.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    parsed = validate_artifact_file("racket_candidates", path)

    assert isinstance(parsed, RacketCandidates)
    assert parsed.artifact_type == "racketsport_racket_candidates"
    assert parsed.players[0].frames[0].corners_px[2] == pytest.approx([31.0, 70.0])


def test_racket_candidates_schema_rejects_bad_corner_contract():
    payload = {
        "schema_version": 1,
        "artifact_type": "racketsport_racket_candidates",
        "fps": 120.0,
        "players": [
            {
                "id": 1,
                "paddle_dims_in": {"length": 15.5, "width": 7.5},
                "frames": [
                    {
                        "t": 0.0,
                        "corners_px": [[10.0, 20.0], [30.0, 21.0], [31.0, 70.0]],
                        "conf": 1.2,
                        "source": "manual_four_corners",
                    }
                ],
            }
        ],
    }

    with pytest.raises(ValidationError):
        RacketCandidates.model_validate(payload)


def test_racket_pose_rejects_malformed_geometry():
    payload = {
        "schema_version": 1,
        "fps": 120.0,
        "players": [
            {
                "id": 1,
                "paddle_dims_in": {},
                "frames": [{"t": 0.0, "pose_se3": {"R": [[1.0]], "t": [0.0]}, "conf": 0.9}],
                "contacts": [{"t": 0.0, "contact_point_face_cm": [0.0], "face_normal": [0.0], "conf": 0.8}],
            }
        ],
    }

    with pytest.raises(ValidationError):
        RacketPose.model_validate(payload)


def test_racket_pose_rejects_non_orthonormal_rotation_matrix():
    payload = {
        "schema_version": 1,
        "fps": 120.0,
        "players": [
            {
                "id": 1,
                "paddle_dims_in": {"length": 15.5, "width": 7.5},
                "frames": [
                    {
                        "t": 0.0,
                        "pose_se3": {
                            "R": [[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                            "t": [0.0, 0.0, 1.0],
                        },
                        "conf": 0.91,
                    }
                ],
                "contacts": [],
            }
        ],
    }

    with pytest.raises(ValidationError, match="orthonormal"):
        RacketPose.model_validate(payload)


def test_virtual_world_rejects_vectors_the_replay_viewer_cannot_read() -> None:
    payload = {
        "schema_version": 1,
        "artifact_type": "racketsport_virtual_world",
        "world_frame": "court_Z0",
        "fps": 30.0,
        "court": {
            "sport": "pickleball",
            "coordinate_frame": "court_Z0",
            "length_m": 13.4112,
            "width_m": 6.096,
            "line_segments": {"net": [[-3.048, 0.0, 0.0], [3.048, 0.0]]},
            "net": {
                "endpoints": [[-3.048, 0.0, 0.0, 1.0], [3.048, 0.0, 0.0]],
                "center_height_m": 0.8636,
                "post_height_m": 0.9144,
            },
        },
        "players": [
            {
                "id": 1,
                "side": "near",
                "role": "left",
                "representation": "track_only",
                "frames": [
                    {
                        "t": 0.0,
                        "track_world_xy": [0.1],
                        "joint_count": 0,
                        "mesh_vertex_count": 0,
                        "floor_world_xyz": [0.1, -1.0],
                    }
                ],
            }
        ],
        "ball": {"source": "tap", "frames": [{"t": 0.0, "xy": [10.0], "conf": 0.8, "visible": True}]},
        "paddles": [],
        "summary": {
            "player_count": 1,
            "mesh_player_count": 0,
            "ball_frame_count": 1,
            "paddle_frame_count": 0,
        },
    }

    with pytest.raises(ValidationError):
        VirtualWorld.model_validate(payload)


def test_virtual_world_accepts_all_ball_track_source_enums() -> None:
    payload = {
        "schema_version": 1,
        "artifact_type": "racketsport_virtual_world",
        "world_frame": "court_Z0",
        "fps": 30.0,
        "court": {
            "sport": "pickleball",
            "coordinate_frame": "court_Z0",
            "length_m": 13.4112,
            "width_m": 6.096,
            "line_segments": {"net": [[-3.048, 0.0, 0.0], [3.048, 0.0, 0.0]]},
            "net": {
                "endpoints": [[-3.048, 0.0, 0.0], [3.048, 0.0, 0.0]],
                "center_height_m": 0.8636,
                "post_height_m": 0.9144,
            },
        },
        "players": [],
        "ball": {
            "source": "pbmat",
            "frames": [{"t": 0.0, "xy": [10.0, 20.0], "conf": 0.8, "visible": True, "world_xyz": [0.0, 0.0, 1.0]}],
        },
        "paddles": [],
        "summary": {
            "player_count": 0,
            "mesh_player_count": 0,
            "ball_frame_count": 1,
            "paddle_frame_count": 0,
        },
    }

    parsed = VirtualWorld.model_validate(payload)

    assert parsed.ball.source == "pbmat"


def test_timeseries_schemas_reject_impossible_numeric_values() -> None:
    with pytest.raises(ValidationError):
        BallTrack.model_validate(
            {
                "schema_version": 1,
                "fps": -30.0,
                "source": "tap",
                "frames": [{"t": 0.0, "xy": [10.0, 20.0], "conf": 99.0, "visible": True}],
                "bounces": [{"t": -1.0, "world_xy": [0.0, 0.0]}],
            }
        )

    with pytest.raises(ValidationError):
        ContactWindows.model_validate(
            {
                "schema_version": 1,
                "events": [
                    {
                        "type": "contact",
                        "t": -0.1,
                        "frame": -1,
                        "player_id": 1,
                        "confidence": 2.0,
                        "sources": {"audio": 1.0, "wrist_vel": 1.0, "ball_inflection": 1.0},
                        "window": {"t0": 10.0, "t1": 1.0, "importance": -5.0},
                    }
                ],
            }
        )

    with pytest.raises(ValidationError):
        Tracks.model_validate(
            {
                "schema_version": 1,
                "fps": 0.0,
                "players": [
                    {
                        "id": 1,
                        "side": "near",
                        "role": "left",
                        "frames": [{"t": 0.0, "bbox": [10.0, 20.0, -5.0, 40.0], "world_xy": [0.0, 0.0], "conf": 2.0}],
                    }
                ],
            }
        )

    with pytest.raises(ValidationError):
        RacketPose.model_validate({"schema_version": 1, "fps": 0.0, "players": []})


def test_contact_window_candidates_reject_impossible_values_and_mismatched_summary() -> None:
    payload = {
        "schema_version": 1,
        "artifact_type": "racketsport_contact_window_candidates",
        "clip": "clip-a",
        "fps": 0.0,
        "source_event_path": "labels/events.json",
        "not_gate_verified": True,
        "trusted_for_body": False,
        "promotion_target": "contact_windows.json",
        "candidates": [
            {
                "review_id": "contact-001",
                "type": "contact",
                "frame": -2,
                "t": -1.0,
                "xy_px": [100.0, 200.0],
                "source_label": "manual",
                "source_status": "draft",
                "source_confidence": 2.5,
                "candidate_confidence": -0.5,
                "window": {"t0": 0.2, "t1": 0.1, "importance": 1.5},
            }
        ],
        "summary": {
            "candidate_count": -1,
            "rejected_item_count": -2,
            "by_type": {"contact": -1},
            "by_status": {"draft": 1},
            "uncertainty_flags": [],
        },
    }

    with pytest.raises(ValidationError):
        ContactWindowCandidates.model_validate(payload)

    valid_candidate = {
        "review_id": "contact-001",
        "type": "contact",
        "frame": 2,
        "t": 1.0,
        "xy_px": [100.0, 200.0],
        "source_label": "manual",
        "source_status": "draft",
        "source_confidence": 0.5,
        "candidate_confidence": 0.5,
        "window": {"t0": 0.9, "t1": 1.1, "importance": 0.5},
    }
    payload.update(
        {
            "fps": 60.0,
            "candidates": [valid_candidate],
            "summary": {
                "candidate_count": 2,
                "rejected_item_count": 0,
                "by_type": {"contact": 1},
                "by_status": {"draft": 1},
                "uncertainty_flags": [],
            },
        }
    )
    with pytest.raises(ValidationError, match="candidate_count"):
        ContactWindowCandidates.model_validate(payload)

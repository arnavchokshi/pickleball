from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tests.racketsport.json_schema_assertions import assert_matches_json_schema
from tests.racketsport.test_orchestrator_spine import _sidecar_payload
from threed.racketsport.pipeline_contracts import (
    PIPELINE_STAGE_ORDER,
    PUBLIC_PIPELINE_STAGE_ORDER,
    _ARTIFACT_SCHEMA_BY_FILENAME,
    PipelineContractError,
    build_public_contract_readiness,
    build_readiness_report,
    safe_relative_path,
)


def _touch_all(run_dir: Path, names: list[str]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        (run_dir / name).write_text(json.dumps(_artifact_payload(name)) + "\n", encoding="utf-8")


def _artifact_payload(name: str) -> dict:
    payloads = {
        "court_calibration.json": {
            "schema_version": 1,
            "sport": "pickleball",
            "homography": [[1.0, 0.0, 960.0], [0.0, 1.0, 540.0], [0.0, 0.0, 1.0]],
            "intrinsics": {"fx": 1000.0, "fy": 1000.0, "cx": 960.0, "cy": 540.0, "dist": [], "source": "manual"},
            "extrinsics": {
                "R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                "t": [0.0, 0.0, 15.0],
                "camera_height_m": 15.0,
            },
            "reprojection_error_px": {"median": 2.0, "p95": 4.0},
            "capture_quality": {"grade": "good", "reasons": []},
            "image_pts": [[100.0, 300.0], [900.0, 300.0], [500.0, 300.0], [500.0, 180.0]],
            "world_pts": [[0.0, 7.0, 0.0], [20.0, 7.0, 0.0], [10.0, 7.0, 0.0], [10.0, 22.0, 0.0]],
        },
        "court_zones.json": {"schema_version": 1, "zones": {}},
        "net_plane.json": {
            "schema_version": 1,
            "plane": {"point": [0.0, 0.0, 0.0], "normal": [0.0, 1.0, 0.0]},
            "endpoints": [[-3.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
            "center_height_in": 34.0,
            "post_height_in": 36.0,
        },
        "court_line_evidence.json": _ready_court_line_evidence_payload(),
        "court_keypoints.json": {
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
                for index, name in enumerate(
                    [
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
                )
            ],
            "target_court_score": 0.82,
            "source": "model_aggregate_v1",
            "not_gate_verified": True,
        },
        "tracks.json": {
            "schema_version": 1,
            "fps": 60.0,
            "players": [
                {
                    "id": 1,
                    "side": "near",
                    "role": "left",
                    "frames": [{"t": 0.0, "bbox": [10.0, 20.0, 40.0, 80.0], "world_xy": [0.0, 0.0], "conf": 0.9}],
                }
            ],
            "rally_spans": [],
        },
        "player_ground.json": {
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
                            "t": 0.0,
                            "feet": [
                                {
                                    "side": "L",
                                    "court_xy": [-0.2, -2.1],
                                    "height_m": 0.01,
                                    "contact": True,
                                    "sigma_p_m": 0.03,
                                    "confidence": 0.9,
                                    "world_xyz": [0.0, 0.0, 0.0],
                                    "source_points": ["ankle"],
                                },
                                {
                                    "side": "R",
                                    "court_xy": [0.2, -2.1],
                                    "height_m": 0.01,
                                    "contact": True,
                                    "sigma_p_m": 0.03,
                                    "confidence": 0.9,
                                    "world_xyz": [0.4, 0.0, 0.0],
                                    "source_points": ["ankle"],
                                },
                            ],
                            "root_world": [0.2, 0.0, 0.9],
                            "joints_world": [[0.2, 0.0, 0.9]],
                        }
                    ],
                }
            ],
        },
        "calls.json": {
            "schema_version": 1,
            "artifact_type": "racketsport_court_calls",
            "source": "metric_floor_v1",
            "not_gate_verified": True,
            "events": [
                {
                    "t": 0.0,
                    "player_id": 1,
                    "foot": "L",
                    "boundary": "kitchen",
                    "decision": "too_close_to_call",
                    "signed_dist_m": -0.004,
                    "sigma_p_m": 0.02,
                    "frames": [0],
                    "metric_confidence": "low",
                    "capture_quality_grade": "good",
                }
            ],
            "summary": {
                "total_events": 1,
                "hard_call_count": 0,
                "too_close_to_call_count": 1,
                "status": "not_gate_verified",
            },
        },
        "drift_log.json": {
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
        },
        "smpl_motion.json": {
            "schema_version": 1,
            "model": "smplx",
            "fps": 60.0,
            "world_frame": "court_Z0",
            "players": [
                {
                    "id": 1,
                    "betas": [0.0] * 10,
                    "skate_free": True,
                    "physics": "none",
                    "frames": [
                        {
                            "t": 0.0,
                            "global_orient": [0.0, 0.0, 0.0],
                            "body_pose": [0.0] * 63,
                            "left_hand_pose": [],
                            "right_hand_pose": [],
                            "transl_world": [0.0, 0.0, 0.0],
                            "joints_world": [[0.0, 0.0, 0.0]],
                            "joint_conf": [0.9],
                            "foot_contact": {"left": True, "right": False},
                        }
                    ],
                }
            ],
        },
        "skeleton3d.json": {
            "schema_version": 1,
            "joint_names": ["pelvis"],
            "preview_only": True,
            "players": [{"id": 1, "frames": [{"t": 0.0, "joints_world": [[0.0, 0.0, 0.0]], "joint_conf": [0.9]}]}],
        },
        "body_compute_execution.json": {
            "schema_version": 1,
            "artifact_type": "racketsport_body_compute_execution",
            "mode": "adaptive_frame_compute_plan",
            "scheduled_frames": [
                {
                    "frame_idx": 0,
                    "player_targets": [
                        {
                            "player_id": 1,
                            "track_conf": 0.9,
                            "score": 0.8,
                            "recommended_tier": "deep_mesh",
                            "target_representation": "world_mesh",
                            "reasons": ["test_fixture"],
                        }
                    ],
                }
            ],
            "skipped_frames": [],
            "summary": {
                "scheduled_frame_count": 1,
                "scheduled_player_frame_count": 1,
                "scheduled_by_target_representation": {"world_mesh": 1},
                "skipped_frame_count": 0,
            },
        },
        "body_mesh_readiness.json": {
            "schema_version": 1,
            "artifact_type": "racketsport_body_mesh_readiness",
            "clip": "clip_001",
            "status": "verified",
            "world_mesh_available": True,
            "representation_decision": "world_mesh_required_available_verified",
            "trusted_for_body_promotion": True,
            "summary": {
                "player_count": 1,
                "mesh_player_count": 1,
                "mesh_frame_count": 1,
                "mesh_vertex_count_min": 3,
                "mesh_vertex_count_max": 3,
                "joints_player_count": 1,
                "joints_frame_count": 1,
            },
            "blockers": [],
            "warnings": [],
        },
        "physics_refinement.json": {
            "schema_version": 1,
            "artifact_type": "racketsport_physics_refinement",
            "physics": "cpu_fallback_scaffold",
            "foot2_done": False,
            "must_not_mark_done_verified": True,
            "constraint_summary": {
                "contact_frames": 1,
                "max_contact_slide_m": 0.0,
                "max_floor_penetration_m": 0.0,
                "inter_player_penetration_frames": 0,
                "max_inter_player_penetration_m": 0.0,
            },
            "execution_plan": {"mode": "cpu_fallback", "will_run_mjx": False, "reason": "test fixture"},
        },
        "ball_track.json": {
            "schema_version": 1,
            "fps": 60.0,
            "source": "tracknet",
            "frames": [{"t": 0.0, "xy": [320.0, 240.0], "conf": 0.9, "visible": True}],
            "bounces": [],
        },
        "contact_windows.json": {
            "schema_version": 1,
            "events": [
                {
                    "type": "contact",
                    "t": 0.25,
                    "frame": 15,
                    "player_id": 1,
                    "confidence": 0.88,
                    "sources": {"audio": 0.9, "wrist_vel": 0.8, "ball_inflection": 0.85},
                    "window": {"t0": 0.2, "t1": 0.3, "importance": 1.0},
                }
            ],
        },
        "racket_pose.json": {
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
                                "R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                                "t": [0.0, 0.0, 1.0],
                            },
                            "conf": 0.9,
                        }
                    ],
                    "contacts": [],
                }
            ],
        },
        "racket_pose_readiness.json": {
            "schema_version": 1,
            "artifact_type": "racketsport_racket_pose_readiness",
            "clip": "clip_001",
            "status": "ready_for_rkt_promotion",
            "blockers": [],
        },
        "racket_promotion_audit.json": {
            "schema_version": 1,
            "artifact_type": "racketsport_racket_promotion_audit",
            "clip": "clip_001",
            "trusted_for_rkt_promotion": True,
            "blockers": [],
        },
        "racket_sport_metrics.json": {
            "schema_version": 1,
            "players": [
                {
                    "id": 1,
                    "shots": [
                        {
                            "t": 0.25,
                            "type": "dink",
                            "type_conf": 0.82,
                            "top2": [{"type": "dink", "confidence": 0.82}],
                            "metrics": {"nvz_margin_ft": {"value": -0.5, "conf": 0.86}},
                        }
                    ],
                }
            ],
        },
        "habit_report.json": _habit_report_payload(),
        "coach_report.json": _habit_report_payload(),
        "drill_report.json": {
            "schema_version": 1,
            "drill": "kitchen_dinks",
            "reps": 1,
            "clean_reps": 1,
            "per_rep": [{"t": 0.0, "quality": "clean", "reasons": []}],
        },
    }
    return payloads.get(name, {"schema_version": 1})


def _habit_report_payload() -> dict:
    return {
        "schema_version": 1,
        "sport": "pickleball",
        "coverage": {"overall": 0.82, "skipped_reason_counts": {}},
        "priority_habit_id": "kitchen_foot",
        "replay_ref": None,
        "habits": [
            {
                "id": "kitchen_foot",
                "title": "Kitchen foot",
                "summary": "Foot crossed the NVZ line.",
                "confidence": 0.86,
                "clip_ref": {"t0_sec": 0.2, "t1_sec": 1.2},
                "cue": "Stay balanced before contact.",
                "drill": {"name": "Kitchen reset", "duration_min": 6.0},
            }
        ],
    }


def _ready_court_line_evidence_payload() -> dict:
    return {
        "schema_version": 1,
        "sport": "pickleball",
        "source": "auto_hough_template",
        "line_observations": [
            _court_line_observation("near_nvz", [[100.0, 300.0], [900.0, 300.0]]),
            _court_line_observation("far_nvz", [[120.0, 180.0], [880.0, 180.0]]),
            _court_line_observation("near_centerline", [[500.0, 300.0], [500.0, 700.0]]),
            _court_line_observation("far_centerline", [[500.0, 180.0], [500.0, 40.0]]),
        ],
        "keypoint_observations": [],
        "net_observations": [
            {
                "net_id": "top_net",
                "image_points": [[100.0, 240.0], [500.0, 238.0], [900.0, 240.0]],
                "confidence": 0.88,
                "frame_indexes": [1, 2, 3],
                "residual_px": {"mean": 2.0, "p95": 3.0},
                "source": "net_top_roi",
            }
        ],
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


def _court_line_observation(line_id: str, image_segment: list[list[float]]) -> dict:
    return {
        "line_id": line_id,
        "image_segment": image_segment,
        "confidence": 0.9,
        "frame_indexes": [1, 2, 3],
        "residual_px": {"mean": 1.0, "p95": 2.0},
        "visible_fraction": 0.9,
        "source": "hough_template",
    }


def _aggregate_only_ready_court_line_evidence_payload() -> dict:
    payload = _ready_court_line_evidence_payload()
    payload["line_observations"] = []
    payload["net_observations"] = []
    return payload


def _write_ready_court_line_evidence(run_dir: Path) -> None:
    (run_dir / "court_line_evidence.json").write_text(
        json.dumps(_ready_court_line_evidence_payload())
        + "\n",
        encoding="utf-8",
    )


def test_public_contract_readiness_survives_legacy_cli_removal(tmp_path: Path) -> None:
    run_dir = tmp_path / "public"
    _touch_all(run_dir, ["court_calibration.json", "tracks.json"])
    (run_dir / "capture_sidecar.json").write_text(json.dumps(_sidecar_payload()) + "\n", encoding="utf-8")

    report = build_public_contract_readiness(run_dir, stage="tracks")

    assert report["artifact_type"] == "pickleball_public_pipeline_contract_readiness"
    assert report["status"] == "ready"
    assert report["stage_order"] == PUBLIC_PIPELINE_STAGE_ORDER
    assert [stage["stage"] for stage in report["stages"]] == [
        "capture_sidecar",
        "court_calibration",
        "tracks",
    ]

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/validate_pipeline_artifacts.py",
            "--run-dir",
            str(run_dir),
            "--public-contracts",
            "--stage",
            "tracks",
        ],
        cwd=Path(__file__).resolve().parents[2],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr + completed.stdout
    assert json.loads(completed.stdout)["status"] == "ready"


def test_readiness_report_marks_stage_ready_only_after_dependencies_exist(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "phase11" / "clip_001"
    _touch_all(run_dir, ["tracks.json"])

    report = build_readiness_report(run_dir, stage="tracking")

    assert report["schema_version"] == 1
    assert report["artifact_type"] == "racketsport_pipeline_artifact_readiness"
    assert report["status"] == "not_ready"
    assert report["requested_stage"] == "tracking"
    assert report["stage_order"][:4] == ["calibration", "tracking", "pose", "body"]

    calibration = report["stages"][0]
    tracking = report["stages"][1]
    assert calibration["stage"] == "calibration"
    assert calibration["status"] == "not_ready"
    assert calibration["missing_artifacts"] == [
        "court_calibration.json",
        "court_zones.json",
        "net_plane.json",
        "court_line_evidence.json",
    ]
    assert tracking["stage"] == "tracking"
    assert tracking["present_artifacts"] == ["tracks.json"]
    assert tracking["missing_artifacts"] == []
    assert tracking["status"] == "blocked"
    assert tracking["blocked_by"] == ["calibration"]


def test_readiness_report_is_ready_when_requested_stage_and_dependencies_are_present(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "phase11" / "clip_001"
    required = [
        "court_calibration.json",
        "court_zones.json",
        "net_plane.json",
        "court_line_evidence.json",
        "tracks.json",
        "smpl_motion.json",
        "skeleton3d.json",
        "body_compute_execution.json",
        "body_mesh_readiness.json",
        "physics_refinement.json",
        "ball_track.json",
        "contact_windows.json",
        "racket_pose.json",
        "racket_pose_readiness.json",
        "racket_promotion_audit.json",
    ]
    _touch_all(run_dir, required)

    report = build_readiness_report(run_dir, stage="racket")

    assert report["status"] == "ready"
    # BODY depends on tracking directly; the removed legacy pose stage is no
    # longer part of the racket readiness chain.
    assert [stage["stage"] for stage in report["stages"]] == [
        "calibration",
        "tracking",
        "body",
        "physics",
        "ball_events",
        "racket",
    ]
    assert report["required_artifacts"] == required
    assert report["missing_artifacts"] == []
    assert report["artifact_validation_errors"] == []
    assert all(stage["status"] == "ready" for stage in report["stages"])


def test_readiness_report_rejects_aggregate_only_court_line_evidence(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "phase11" / "clip_001"
    _touch_all(run_dir, ["court_calibration.json", "court_zones.json", "net_plane.json", "court_line_evidence.json", "tracks.json"])
    (run_dir / "court_line_evidence.json").write_text(
        json.dumps(_aggregate_only_ready_court_line_evidence_payload()) + "\n",
        encoding="utf-8",
    )

    report = build_readiness_report(run_dir, stage="tracking")

    assert report["status"] == "not_ready"
    calibration = report["stages"][0]
    assert calibration["status"] == "not_ready"
    assert "court_line_evidence_ready_without_line_observations" in calibration["semantic_blockers"]
    assert "court_line_evidence_ready_without_net_observations" in calibration["semantic_blockers"]


def test_readiness_report_blocks_on_semantically_empty_artifacts(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "phase11" / "clip_001"
    required = [
        "court_calibration.json",
        "court_zones.json",
        "net_plane.json",
        "court_line_evidence.json",
        "tracks.json",
        "smpl_motion.json",
        "skeleton3d.json",
        "body_compute_execution.json",
        "body_mesh_readiness.json",
        "physics_refinement.json",
        "ball_track.json",
        "contact_windows.json",
        "racket_pose.json",
        "racket_pose_readiness.json",
        "racket_promotion_audit.json",
    ]
    _touch_all(run_dir, required)
    (run_dir / "contact_windows.json").write_text(
        json.dumps({"schema_version": 1, "events": []}) + "\n",
        encoding="utf-8",
    )
    (run_dir / "body_compute_execution.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_type": "racketsport_body_compute_execution",
                "mode": "adaptive_frame_compute_plan",
                "scheduled_frames": [],
                "skipped_frames": [],
                "summary": {"scheduled_frame_count": 0},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "body_mesh_readiness.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_type": "racketsport_body_mesh_readiness",
                "representation_decision": "no_world_mesh_requested",
                "trusted_for_body_promotion": False,
                "status": "mesh_available_needs_accuracy_gate",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = build_readiness_report(run_dir, stage="racket")

    assert report["status"] == "not_ready"
    assert report["missing_artifacts"] == []
    assert report["semantic_blockers"] == [
        "body:body_compute_execution_has_no_scheduled_frames",
        "body:body_mesh_no_world_mesh_requested",
        "body:body_mesh_not_trusted_for_promotion",
        "ball_events:contact_windows_has_no_events",
    ]
    body = next(stage for stage in report["stages"] if stage["stage"] == "body")
    ball_events = next(stage for stage in report["stages"] if stage["stage"] == "ball_events")
    racket = next(stage for stage in report["stages"] if stage["stage"] == "racket")
    assert body["status"] == "blocked"
    assert body["semantic_blockers"] == [
        "body_compute_execution_has_no_scheduled_frames",
        "body_mesh_no_world_mesh_requested",
        "body_mesh_not_trusted_for_promotion",
    ]
    assert ball_events["semantic_blockers"] == ["contact_windows_has_no_events"]
    assert racket["status"] == "blocked"
    assert racket["blocked_by"] == ["physics", "ball_events"]


def test_readiness_report_blocks_when_body_schedule_omits_world_mesh_count(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "phase11" / "clip_001"
    required = [
        "court_calibration.json",
        "court_zones.json",
        "net_plane.json",
        "court_line_evidence.json",
        "tracks.json",
        "smpl_motion.json",
        "skeleton3d.json",
        "body_compute_execution.json",
        "body_mesh_readiness.json",
    ]
    _touch_all(run_dir, required)
    (run_dir / "body_compute_execution.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_type": "racketsport_body_compute_execution",
                "mode": "adaptive_frame_compute_plan",
                "scheduled_frames": [{"frame_idx": 0, "player_targets": []}],
                "skipped_frames": [],
                "summary": {"scheduled_frame_count": 1},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = build_readiness_report(run_dir, stage="body")

    assert report["status"] == "not_ready"
    assert report["semantic_blockers"] == [
        "body:body_compute_execution_missing_scheduled_by_target_representation"
    ]
    body = next(stage for stage in report["stages"] if stage["stage"] == "body")
    assert body["status"] == "blocked"
    assert body["semantic_blockers"] == ["body_compute_execution_missing_scheduled_by_target_representation"]


def test_readiness_report_requires_body_and_racket_readiness_artifacts(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "phase11" / "clip_001"
    _touch_all(
        run_dir,
        [
            "court_calibration.json",
            "court_zones.json",
            "net_plane.json",
            "court_line_evidence.json",
            "tracks.json",
            "smpl_motion.json",
            "skeleton3d.json",
            "physics_refinement.json",
            "ball_track.json",
            "contact_windows.json",
            "racket_pose.json",
        ],
    )

    report = build_readiness_report(run_dir, stage="racket")

    assert report["status"] == "not_ready"
    assert report["missing_artifacts"] == [
        "body_compute_execution.json",
        "body_mesh_readiness.json",
        "racket_pose_readiness.json",
        "racket_promotion_audit.json",
    ]


def test_readiness_report_blocks_when_court_line_evidence_is_not_ready(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "phase2" / "clip_001"
    _touch_all(
        run_dir,
        [
            "court_calibration.json",
            "court_zones.json",
            "net_plane.json",
            "court_line_evidence.json",
            "tracks.json",
        ],
    )
    (run_dir / "court_line_evidence.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "sport": "pickleball",
                "source": "auto_hough_template",
                "line_observations": [],
                "keypoint_observations": [],
                "net_observations": [],
                "aggregate": {
                    "accepted_line_ids": [],
                    "rejected_line_ids": [],
                    "auto_calibration_ready": False,
                    "missing_required_line_ids": ["near_nvz"],
                    "missing_required_net_ids": ["top_net"],
                    "mean_residual_px": 20.0,
                    "p95_residual_px": 30.0,
                    "temporal_stability_px": 5.0,
                    "reasons": ["not ready"],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = build_readiness_report(run_dir, stage="tracking")

    assert report["status"] == "not_ready"
    assert report["missing_artifacts"] == []
    assert report["semantic_blockers"] == [
        "calibration:court_line_evidence_not_ready",
        "calibration:court_line_evidence_missing_required_line_near_nvz",
        "calibration:court_line_evidence_missing_required_net_top_net",
    ]
    calibration = report["stages"][0]
    tracking = report["stages"][1]
    assert calibration["status"] == "blocked"
    assert calibration["semantic_blockers"] == [
        "court_line_evidence_not_ready",
        "court_line_evidence_missing_required_line_near_nvz",
        "court_line_evidence_missing_required_net_top_net",
    ]
    assert tracking["status"] == "blocked"
    assert tracking["blocked_by"] == ["calibration"]


def test_readiness_report_blocks_retired_burlington_court_evidence(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "prototype_gate_h100_v2" / "burlington_gold_0300_low_steep_corner"
    _touch_all(
        run_dir,
        [
            "court_calibration.json",
            "court_zones.json",
            "net_plane.json",
            "court_line_evidence.json",
            "tracks.json",
        ],
    )
    (run_dir / "court_line_evidence.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "sport": "pickleball",
                "source": "auto_hough_template",
                "line_observations": [],
                "keypoint_observations": [],
                "net_observations": [],
                "aggregate": {
                    "accepted_line_ids": [],
                    "rejected_line_ids": [],
                    "auto_calibration_ready": False,
                    "missing_required_line_ids": ["near_nvz"],
                    "missing_required_net_ids": ["top_net"],
                    "mean_residual_px": 20.0,
                    "p95_residual_px": 30.0,
                    "temporal_stability_px": 5.0,
                    "reasons": ["retired"],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = build_readiness_report(run_dir, stage="tracking")

    assert report["semantic_blockers"] == ["calibration:court_line_evidence_retired_for_court_calibration"]
    calibration = report["stages"][0]
    tracking = report["stages"][1]
    assert calibration["status"] == "blocked"
    assert calibration["semantic_blockers"] == ["court_line_evidence_retired_for_court_calibration"]
    assert tracking["status"] == "blocked"
    assert tracking["blocked_by"] == ["calibration"]


def test_safe_relative_path_rejects_absolute_and_parent_traversal() -> None:
    assert safe_relative_path("clip_001/court_calibration.json") == Path("clip_001/court_calibration.json")

    for value in ["", ".", "/tmp/court_calibration.json", "../court_calibration.json", "clip/../../tracks.json"]:
        with pytest.raises(PipelineContractError):
            safe_relative_path(value)


def test_court_positioning_artifact_filenames_are_schema_mapped() -> None:
    assert _ARTIFACT_SCHEMA_BY_FILENAME["court_keypoints.json"] == "court_keypoints"
    assert _ARTIFACT_SCHEMA_BY_FILENAME["court_lock.json"] == "court_lock"
    assert _ARTIFACT_SCHEMA_BY_FILENAME["player_ground.json"] == "player_ground"
    assert _ARTIFACT_SCHEMA_BY_FILENAME["calls.json"] == "court_calls"
    assert _ARTIFACT_SCHEMA_BY_FILENAME["drift_log.json"] == "drift_log"


def test_unknown_stage_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(PipelineContractError, match="unknown pipeline stage"):
        build_readiness_report(tmp_path, stage="eval4")


def test_validate_pipeline_artifacts_cli_writes_machine_readable_report(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "phase7" / "clip_001"
    _touch_all(
        run_dir,
        [
            "court_calibration.json",
            "court_zones.json",
            "net_plane.json",
            "court_line_evidence.json",
            "tracks.json",
            "smpl_motion.json",
            "skeleton3d.json",
        ],
    )
    _write_ready_court_line_evidence(run_dir)
    out = tmp_path / "readiness.json"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/validate_pipeline_artifacts.py",
            "--run-dir",
            str(run_dir),
            "--stage",
            "metrics",
            "--out",
            str(out),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    assert "not_ready" in completed.stdout

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["status"] == "not_ready"
    assert payload["requested_stage"] == "metrics"
    assert payload["missing_artifacts"] == [
        "body_compute_execution.json",
        "body_mesh_readiness.json",
        "physics_refinement.json",
        "ball_track.json",
        "contact_windows.json",
        "racket_pose.json",
        "racket_pose_readiness.json",
        "racket_promotion_audit.json",
        "racket_sport_metrics.json",
        "habit_report.json",
    ]
    assert payload["stages"][-1]["stage"] == "metrics"
    assert payload["stages"][-1]["status"] == "not_ready"


def test_readiness_report_matches_checked_in_json_schema(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "phase11" / "clip_001"
    _touch_all(run_dir, ["tracks.json"])

    report = build_readiness_report(run_dir, stage="tracking")
    schema = json.loads(Path("docs/racketsport/pipeline_contracts_schema.json").read_text(encoding="utf-8"))

    assert_matches_json_schema(report, schema)


def test_body_contract_validation_tolerates_absent_smpl_monolith_but_rejects_invalid(tmp_path: Path) -> None:
    """S4 slim mode (write_body_monoliths=False) legally omits smpl_motion.json.

    The body contract validator must accept its ABSENCE (skeleton3d.json et al.
    are the slim evidence) while still failing on a PRESENT-but-invalid file and
    on absence of the hard artifacts.
    """

    from threed.racketsport.orchestrator import _validate_contract_artifacts
    from threed.racketsport.pipeline_contracts import PIPELINE_STAGE_CONTRACTS

    body_contract = next(c for c in PIPELINE_STAGE_CONTRACTS if c.stage == "body")
    assert "smpl_motion.json" in body_contract.required_artifacts

    import pytest

    # Hard artifacts absent -> still fails (absence tolerance is smpl-only).
    with pytest.raises(Exception):
        _validate_contract_artifacts(body_contract, tmp_path)

    # Present-but-invalid smpl_motion.json -> still fails.
    (tmp_path / "smpl_motion.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(Exception):
        _validate_contract_artifacts(body_contract, tmp_path)
    (tmp_path / "smpl_motion.json").unlink()

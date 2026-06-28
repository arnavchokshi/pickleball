from __future__ import annotations

import json
from pathlib import Path

from threed.racketsport.eval import e2e_eval
from threed.racketsport.replay_export import build_replay_review_export_from_virtual_world, write_replay_scene
from threed.racketsport.schemas import (
    CameraIntrinsics,
    CaptureQuality,
    CourtCalibration,
    CourtExtrinsics,
    ReprojectionError,
)


REQUIRED_LABEL_FILES = (
    "court_corners.json",
    "players.json",
    "feet_nvz.json",
    "ball.json",
    "events.json",
    "racket_pose.json",
    "foot_contact.json",
    "coach_habits.json",
    "manual_metrics.json",
)


def _write_ready_clip(labels_root: Path, name: str) -> None:
    labels_dir = labels_root / name / "labels"
    labels_dir.mkdir(parents=True)
    for label in REQUIRED_LABEL_FILES:
        (labels_dir / label).write_text("{}", encoding="utf-8")


def _write_calibration_artifacts(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    calibration = CourtCalibration(
        schema_version=1,
        sport="pickleball",
        homography=[[1.0, 0.0, 960.0], [0.0, 1.0, 540.0], [0.0, 0.0, 1.0]],
        intrinsics=CameraIntrinsics(fx=1000.0, fy=1000.0, cx=960.0, cy=540.0, dist=[], source="arkit"),
        extrinsics=CourtExtrinsics(
            R=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            t=[0.0, 0.0, 15.0],
            camera_height_m=15.0,
        ),
        reprojection_error_px=ReprojectionError(median=2.5, p95=7.0),
        capture_quality=CaptureQuality(grade="good", reasons=[]),
        image_pts=[],
        world_pts=[],
    )
    (run_dir / "court_calibration.json").write_text(calibration.model_dump_json(), encoding="utf-8")
    (run_dir / "court_zones.json").write_text(json.dumps({"schema_version": 1, "zones": {}}), encoding="utf-8")
    (run_dir / "net_plane.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "plane": {"point": [0.0, 0.0, 0.0], "normal": [0.0, 1.0, 0.0]},
                "endpoints": [[-3.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
                "center_height_in": 34.0,
                "post_height_in": 36.0,
            }
        ),
        encoding="utf-8",
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
        ),
        encoding="utf-8",
    )


def _write_tracks_artifact(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "tracks.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "fps": 60.0,
                "players": [
                    {
                        "id": 1,
                        "side": "near",
                        "role": "left",
                        "frames": [{"t": 0.0, "bbox": [100.0, 100.0, 140.0, 240.0], "world_xy": [0.0, 0.0], "conf": 0.9}],
                    }
                ],
                "rally_spans": [],
            }
        ),
        encoding="utf-8",
    )


def _write_smpl_motion_artifact(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "smpl_motion.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model": "smplx",
                "fps": 60.0,
                "world_frame": "court_Z0",
                "players": [
                    {
                        "id": 1,
                        "betas": [0.0] * 10,
                        "skate_free": True,
                        "physics": "physpt",
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
                                "grf": [[0.0, 0.0, 1.0]],
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    (run_dir / "skeleton3d.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "joint_names": ["pelvis"],
                "preview_only": True,
                "players": [
                    {
                        "id": 1,
                        "frames": [{"t": 0.0, "joints_world": [[0.0, 0.0, 0.0]], "joint_conf": [0.9]}],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def _write_physics_refinement_artifact(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "physics_refinement.json").write_text(
        json.dumps(
            {
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
                "execution_plan": {
                    "mode": "cpu_fallback",
                    "will_run_mjx": False,
                    "reason": "test fixture",
                },
            }
        ),
        encoding="utf-8",
    )


def _write_ball_event_artifacts(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "ball_track.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "fps": 60.0,
                "source": "tracknet",
                "frames": [{"t": 0.0, "xy": [320.0, 240.0], "conf": 0.91, "visible": True, "world_xyz": [0.0, 0.0, 1.0]}],
                "bounces": [{"t": 0.5, "world_xy": [0.1, 0.2]}],
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "contact_windows.json").write_text(
        json.dumps(
            {
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
            }
        ),
        encoding="utf-8",
    )


def _write_racket_pose_artifact(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "racket_pose.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "fps": 120.0,
                "players": [
                    {
                        "id": 1,
                        "paddle_dims_in": {"length": 15.5, "width": 7.5, "thickness": 0.55},
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
                        "contacts": [
                            {"t": 0.0, "contact_point_face_cm": [0.0, 0.0], "face_normal": [0.0, 0.0, 1.0], "conf": 0.8}
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def _habit_report_payload() -> dict:
    return {
        "schema_version": 1,
        "sport": "pickleball",
        "coverage": {"overall": 0.82, "skipped_reason_counts": {"ball_uncertain": 1}},
        "priority_habit_id": "kitchen_foot",
        "replay_ref": {"glb_url": "/replay/clip_001?point=1"},
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


def _write_report_artifacts(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "racket_sport_metrics.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "players": [
                    {
                        "id": 1,
                        "shots": [
                            {
                                "t": 0.25,
                                "type": "dink",
                                "type_conf": 0.82,
                                "metrics": {"nvz_margin_ft": {"value": -0.5, "conf": 0.86, "frames": 5}},
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "habit_report.json").write_text(json.dumps(_habit_report_payload()), encoding="utf-8")
    (run_dir / "coach_report.json").write_text(json.dumps(_habit_report_payload()), encoding="utf-8")
    (run_dir / "drill_report.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "drill": "kitchen_dinks",
                "reps": 1,
                "clean_reps": 1,
                "per_rep": [{"t": 0.0, "quality": "clean", "reasons": []}],
            }
        ),
        encoding="utf-8",
    )


def _write_replay_artifacts(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    scene = build_replay_review_export_from_virtual_world(_replay_virtual_world_payload(), export_root=run_dir, point_id=3)
    write_replay_scene(run_dir / "replay_scene.json", scene)


def _replay_virtual_world_payload() -> dict:
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_virtual_world",
        "world_frame": "court_Z0",
        "fps": 30.0,
        "court": {
            "sport": "pickleball",
            "coordinate_frame": "court_Z0",
            "line_segments": {"baseline": [[-3.05, 0.0, 0.0], [3.05, 0.0, 0.0]]},
            "net": {"endpoints": [[-3.05, 6.705, 0.91], [3.05, 6.705, 0.91]]},
        },
        "players": [
            {
                "id": 1,
                "side": "near",
                "role": "left",
                "frames": [
                    {"t": 0.0, "track_world_xy": [0.0, 1.0], "joints_world": [[0.0, 1.0, 1.2]]},
                    {"t": 1.0, "track_world_xy": [0.0, 2.0], "joints_world": [[0.0, 2.0, 1.1]]},
                ],
            },
            {
                "id": 2,
                "side": "far",
                "role": "right",
                "frames": [
                    {"t": 0.0, "track_world_xy": [1.0, 10.0], "joints_world": [[1.0, 10.0, 1.2]]},
                    {"t": 1.0, "track_world_xy": [1.0, 9.0], "joints_world": [[1.0, 9.0, 1.1]]},
                ],
            },
        ],
        "ball": {"source": "fixture", "frames": [{"t": 0.0, "world_xyz": [0.0, 6.0, 0.3], "visible": True}]},
        "paddles": [
            {
                "player_id": 1,
                "frames": [
                    {
                        "t": 0.0,
                        "mesh_vertices_world": [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.1, 0.2, 0.0]],
                        "mesh_faces": [[0, 1, 2]],
                    }
                ],
            }
        ],
    }


def _write_e2e_artifacts(run_dir: Path) -> None:
    _write_calibration_artifacts(run_dir)
    _write_tracks_artifact(run_dir)
    _write_smpl_motion_artifact(run_dir)
    _write_physics_refinement_artifact(run_dir)
    _write_ball_event_artifacts(run_dir)
    _write_racket_pose_artifact(run_dir)
    _write_report_artifacts(run_dir)
    _write_replay_artifacts(run_dir)


def test_e2e_eval_passes_when_all_artifacts_and_glbs_exist(tmp_path: Path) -> None:
    labels_root = tmp_path / "data" / "testclips"
    root = tmp_path / "runs" / "phase11"
    _write_ready_clip(labels_root, "clip_001")
    _write_e2e_artifacts(root / "clip_001")

    payload = e2e_eval.evaluate(root, labels_root)

    assert payload.status == "pass"
    assert payload.clips[0].status == "pass"
    metrics = payload.clips[0].metrics
    assert metrics["required_artifacts_present"].value == 16
    assert metrics["required_artifacts_present"].passed is True
    assert metrics["referenced_glb_files_present"].value == 2
    assert metrics["referenced_glb_files_present"].passed is True
    assert metrics["referenced_glb_files_valid"].value == 2
    assert metrics["referenced_glb_files_valid"].passed is True


def test_e2e_eval_blocks_when_required_artifact_is_missing(tmp_path: Path) -> None:
    labels_root = tmp_path / "data" / "testclips"
    root = tmp_path / "runs" / "phase11"
    _write_ready_clip(labels_root, "clip_001")
    _write_e2e_artifacts(root / "clip_001")
    (root / "clip_001" / "racket_pose.json").unlink()

    payload = e2e_eval.evaluate(root, labels_root)

    assert payload.status == "blocked"
    assert payload.clips[0].status == "blocked"
    assert payload.clips[0].missing_artifacts == ["racket_pose.json"]
    assert payload.clips[0].metrics == {}


def test_e2e_eval_fails_when_referenced_glb_is_missing(tmp_path: Path) -> None:
    labels_root = tmp_path / "data" / "testclips"
    root = tmp_path / "runs" / "phase11"
    _write_ready_clip(labels_root, "clip_001")
    _write_e2e_artifacts(root / "clip_001")
    (root / "clip_001" / "points" / "point_003_review.glb").unlink()

    payload = e2e_eval.evaluate(root, labels_root)

    assert payload.status == "fail"
    assert payload.clips[0].status == "fail"
    assert payload.clips[0].metrics["referenced_glb_files_present"].value == 1
    assert payload.clips[0].metrics["referenced_glb_files_present"].passed is False
    assert "missing referenced GLB files: points/point_003_review.glb" in payload.clips[0].notes


def test_e2e_eval_fails_when_referenced_glb_is_invalid(tmp_path: Path) -> None:
    labels_root = tmp_path / "data" / "testclips"
    root = tmp_path / "runs" / "phase11"
    _write_ready_clip(labels_root, "clip_001")
    _write_e2e_artifacts(root / "clip_001")
    (root / "clip_001" / "points" / "point_003_review.glb").write_bytes(b"glb")

    payload = e2e_eval.evaluate(root, labels_root)

    assert payload.status == "fail"
    assert payload.clips[0].status == "fail"
    assert payload.clips[0].metrics["referenced_glb_files_present"].passed is True
    assert payload.clips[0].metrics["referenced_glb_files_valid"].value == 1
    assert payload.clips[0].metrics["referenced_glb_files_valid"].passed is False
    assert "invalid referenced GLB files: points/point_003_review.glb" in payload.clips[0].notes[0]


def test_e2e_eval_emits_named_numeric_gate_metadata(tmp_path: Path) -> None:
    labels_root = tmp_path / "data" / "testclips"
    root = tmp_path / "runs" / "phase11"
    _write_ready_clip(labels_root, "clip_001")
    _write_e2e_artifacts(root / "clip_001")

    payload = e2e_eval.evaluate(root, labels_root)
    metrics = payload.clips[0].metrics

    assert metrics["required_artifacts_present"].gate == "artifact_check.e2e_required_artifacts_present: == 16"
    assert metrics["required_artifacts_total"].gate == "artifact_check.e2e_required_artifacts_total: == 16"
    assert metrics["referenced_glb_files_present"].gate == "artifact_check.e2e_referenced_glb_files_present: == 2"
    assert metrics["referenced_glb_files_valid"].gate == "artifact_check.e2e_referenced_glb_files_valid: == 2"
    assert all(metric.status == "measured" for metric in metrics.values())

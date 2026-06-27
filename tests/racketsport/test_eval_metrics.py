from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from threed.racketsport.schemas import (
    CameraIntrinsics,
    CaptureQuality,
    CourtCalibration,
    CourtExtrinsics,
    ReprojectionError,
    validate_artifact_file,
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


def _write_calibration_artifacts(run_dir: Path, *, median: float = 2.5, p95: float = 7.0) -> None:
    run_dir.mkdir(parents=True)
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
        reprojection_error_px=ReprojectionError(median=median, p95=p95),
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


def _write_tracks_artifact(run_dir: Path, *, players: int = 2) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "fps": 60.0,
        "players": [
            {
                "id": idx + 1,
                "side": "near" if idx % 2 == 0 else "far",
                "role": "left" if idx % 2 == 0 else "right",
                "frames": [
                    {
                        "t": 0.0,
                        "bbox": [100.0 + idx, 100.0, 140.0 + idx, 240.0],
                        "world_xy": [float(idx), 0.0],
                        "conf": 0.9,
                    }
                ],
            }
            for idx in range(players)
        ],
        "rally_spans": [],
    }
    (run_dir / "tracks.json").write_text(json.dumps(payload), encoding="utf-8")


def _write_body_artifacts(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    smpl_motion = {
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
    }
    skeleton = {
        "schema_version": 1,
        "joint_names": ["pelvis"],
        "preview_only": True,
        "players": [{"id": 1, "frames": [{"t": 0.0, "joints_world": [[0.0, 0.0, 0.0]], "joint_conf": [0.9]}]}],
    }
    (run_dir / "smpl_motion.json").write_text(json.dumps(smpl_motion), encoding="utf-8")
    (run_dir / "skeleton3d.json").write_text(json.dumps(skeleton), encoding="utf-8")


def _write_ball_event_artifacts(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    ball_track = {
        "schema_version": 1,
        "fps": 60.0,
        "source": "tracknet",
        "frames": [{"t": 0.0, "xy": [320.0, 240.0], "conf": 0.91, "visible": True, "world_xyz": [0.0, 0.0, 1.0]}],
        "bounces": [{"t": 0.5, "world_xy": [0.1, 0.2]}],
    }
    contact_windows = {
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
    (run_dir / "ball_track.json").write_text(json.dumps(ball_track), encoding="utf-8")
    (run_dir / "contact_windows.json").write_text(json.dumps(contact_windows), encoding="utf-8")


def _write_racket_pose_artifact(run_dir: Path, *, contacts: int = 1) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
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
                    {"t": float(idx) / 120.0, "contact_point_face_cm": [0.0, 0.0], "face_normal": [0.0, 0.0, 1.0], "conf": 0.8}
                    for idx in range(contacts)
                ],
            }
        ],
    }
    (run_dir / "racket_pose.json").write_text(json.dumps(payload), encoding="utf-8")


def test_calib_eval_passes_when_ready_clip_has_required_artifacts(tmp_path):
    labels_root = tmp_path / "data" / "testclips"
    root = tmp_path / "runs" / "phase1"
    _write_ready_clip(labels_root, "clip_001")
    _write_calibration_artifacts(root / "clip_001")

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "threed.racketsport.eval.calib_eval",
            "--root",
            str(root),
            "--labels",
            str(labels_root),
            "--out",
            str(root / "metrics.json"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
    assert json.loads(completed.stdout)["status"] == "pass"
    assert payload["status"] == "pass"
    assert payload["summary"]["evaluated_clips"] == 1
    assert payload["summary"]["passed_clips"] == 1
    assert payload["metrics"]["artifact_readiness"]["passed"] is True
    assert payload["clips"][0]["metrics"]["reprojection_median_px"]["value"] == 2.5
    validate_artifact_file("phase_eval_metrics", root / "metrics.json")


def test_calib_eval_blocks_when_required_run_artifact_is_missing(tmp_path):
    labels_root = tmp_path / "data" / "testclips"
    root = tmp_path / "runs" / "phase1"
    _write_ready_clip(labels_root, "clip_001")
    _write_calibration_artifacts(root / "clip_001")
    (root / "clip_001" / "net_plane.json").unlink()

    subprocess.run(
        [
            sys.executable,
            "-m",
            "threed.racketsport.eval.calib_eval",
            "--root",
            str(root),
            "--labels",
            str(labels_root),
            "--out",
            str(root / "metrics.json"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    payload = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
    assert payload["status"] == "blocked"
    assert payload["summary"]["blocked_clips"] == 1
    assert payload["clips"][0]["missing_artifacts"] == ["net_plane.json"]


def test_calib_eval_marks_missing_labels_not_measured(tmp_path):
    labels_root = tmp_path / "data" / "testclips"
    root = tmp_path / "runs" / "phase1"
    labels_dir = labels_root / "clip_001" / "labels"
    labels_dir.mkdir(parents=True)
    (labels_dir / "court_corners.json").write_text("{}", encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            "-m",
            "threed.racketsport.eval.calib_eval",
            "--root",
            str(root),
            "--labels",
            str(labels_root),
            "--out",
            str(root / "metrics.json"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    payload = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
    assert payload["status"] == "not_measured"
    assert payload["summary"]["evaluated_clips"] == 0
    assert payload["clips"][0]["status"] == "not_measured"
    assert "players.json" in payload["clips"][0]["missing_label_files"]


def test_track_eval_passes_when_ready_clip_has_tracks_artifact(tmp_path):
    labels_root = tmp_path / "data" / "testclips"
    root = tmp_path / "runs" / "phase2"
    _write_ready_clip(labels_root, "clip_001")
    _write_tracks_artifact(root / "clip_001", players=2)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "threed.racketsport.eval.track_eval",
            "--root",
            str(root),
            "--labels",
            str(labels_root),
            "--out",
            str(root / "metrics.json"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
    assert json.loads(completed.stdout)["status"] == "pass"
    assert payload["phase"] == "phase2"
    assert payload["required_artifacts"] == ["tracks.json"]
    assert payload["clips"][0]["metrics"]["players_detected"]["value"] == 2
    assert payload["clips"][0]["metrics"]["track_frames"]["value"] == 2
    validate_artifact_file("phase_eval_metrics", root / "metrics.json")


def test_track_eval_blocks_when_tracks_artifact_is_missing(tmp_path):
    labels_root = tmp_path / "data" / "testclips"
    root = tmp_path / "runs" / "phase2"
    _write_ready_clip(labels_root, "clip_001")

    subprocess.run(
        [
            sys.executable,
            "-m",
            "threed.racketsport.eval.track_eval",
            "--root",
            str(root),
            "--labels",
            str(labels_root),
            "--out",
            str(root / "metrics.json"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    payload = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
    assert payload["status"] == "blocked"
    assert payload["clips"][0]["missing_artifacts"] == ["tracks.json"]


def test_body_eval_passes_when_ready_clip_has_body_artifacts(tmp_path):
    labels_root = tmp_path / "data" / "testclips"
    root = tmp_path / "runs" / "phase3"
    _write_ready_clip(labels_root, "clip_001")
    _write_body_artifacts(root / "clip_001")

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "threed.racketsport.eval.body_eval",
            "--root",
            str(root),
            "--labels",
            str(labels_root),
            "--out",
            str(root / "metrics.json"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
    assert json.loads(completed.stdout)["status"] == "pass"
    assert payload["phase"] == "phase3"
    assert payload["required_artifacts"] == ["smpl_motion.json", "skeleton3d.json"]
    assert payload["clips"][0]["metrics"]["smpl_players"]["value"] == 1
    assert payload["clips"][0]["metrics"]["smpl_frames"]["value"] == 1
    assert payload["clips"][0]["metrics"]["skeleton_players"]["value"] == 1
    validate_artifact_file("phase_eval_metrics", root / "metrics.json")


def test_body_eval_blocks_when_skeleton_preview_is_missing(tmp_path):
    labels_root = tmp_path / "data" / "testclips"
    root = tmp_path / "runs" / "phase3"
    _write_ready_clip(labels_root, "clip_001")
    _write_body_artifacts(root / "clip_001")
    (root / "clip_001" / "skeleton3d.json").unlink()

    subprocess.run(
        [
            sys.executable,
            "-m",
            "threed.racketsport.eval.body_eval",
            "--root",
            str(root),
            "--labels",
            str(labels_root),
            "--out",
            str(root / "metrics.json"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    payload = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
    assert payload["status"] == "blocked"
    assert payload["clips"][0]["missing_artifacts"] == ["skeleton3d.json"]


def test_ball_event_eval_passes_when_ready_clip_has_ball_and_event_artifacts(tmp_path):
    labels_root = tmp_path / "data" / "testclips"
    root = tmp_path / "runs" / "phase5"
    _write_ready_clip(labels_root, "clip_001")
    _write_ball_event_artifacts(root / "clip_001")

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "threed.racketsport.eval.ball_event_eval",
            "--root",
            str(root),
            "--labels",
            str(labels_root),
            "--out",
            str(root / "metrics.json"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
    assert json.loads(completed.stdout)["status"] == "pass"
    assert payload["phase"] == "phase5"
    assert payload["required_artifacts"] == ["ball_track.json", "contact_windows.json"]
    assert payload["clips"][0]["metrics"]["ball_frames"]["value"] == 1
    assert payload["clips"][0]["metrics"]["contact_events"]["value"] == 1
    assert payload["clips"][0]["metrics"]["bounce_events"]["value"] == 1
    validate_artifact_file("phase_eval_metrics", root / "metrics.json")


def test_ball_event_eval_blocks_when_contact_windows_are_missing(tmp_path):
    labels_root = tmp_path / "data" / "testclips"
    root = tmp_path / "runs" / "phase5"
    _write_ready_clip(labels_root, "clip_001")
    _write_ball_event_artifacts(root / "clip_001")
    (root / "clip_001" / "contact_windows.json").unlink()

    subprocess.run(
        [
            sys.executable,
            "-m",
            "threed.racketsport.eval.ball_event_eval",
            "--root",
            str(root),
            "--labels",
            str(labels_root),
            "--out",
            str(root / "metrics.json"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    payload = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
    assert payload["status"] == "blocked"
    assert payload["clips"][0]["missing_artifacts"] == ["contact_windows.json"]


def test_racket_eval_passes_when_ready_clip_has_racket_pose_artifact(tmp_path):
    labels_root = tmp_path / "data" / "testclips"
    root = tmp_path / "runs" / "phase6"
    _write_ready_clip(labels_root, "clip_001")
    _write_racket_pose_artifact(root / "clip_001", contacts=2)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "threed.racketsport.eval.racket_eval",
            "--root",
            str(root),
            "--labels",
            str(labels_root),
            "--out",
            str(root / "metrics.json"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
    assert json.loads(completed.stdout)["status"] == "pass"
    assert payload["phase"] == "phase6"
    assert payload["required_artifacts"] == ["racket_pose.json"]
    assert payload["clips"][0]["metrics"]["racket_players"]["value"] == 1
    assert payload["clips"][0]["metrics"]["racket_frames"]["value"] == 1
    assert payload["clips"][0]["metrics"]["racket_contacts"]["value"] == 2
    validate_artifact_file("phase_eval_metrics", root / "metrics.json")


def test_racket_eval_blocks_when_racket_pose_artifact_is_missing(tmp_path):
    labels_root = tmp_path / "data" / "testclips"
    root = tmp_path / "runs" / "phase6"
    _write_ready_clip(labels_root, "clip_001")

    subprocess.run(
        [
            sys.executable,
            "-m",
            "threed.racketsport.eval.racket_eval",
            "--root",
            str(root),
            "--labels",
            str(labels_root),
            "--out",
            str(root / "metrics.json"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    payload = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
    assert payload["status"] == "blocked"
    assert payload["clips"][0]["missing_artifacts"] == ["racket_pose.json"]

from __future__ import annotations

import json
from pathlib import Path

from threed.racketsport.eval import ball_event_eval, racket_eval


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
        payload = {}
        if label == "racket_pose.json":
            payload = {
                "schema_version": 1,
                "status": "human_reviewed",
                "not_ground_truth": False,
                "annotation": {"target_file": "racket_pose.json", "items": []},
            }
        (labels_dir / label).write_text(json.dumps(payload), encoding="utf-8")


def _write_incomplete_clip(labels_root: Path, name: str) -> None:
    labels_dir = labels_root / name / "labels"
    labels_dir.mkdir(parents=True)
    (labels_dir / "court_corners.json").write_text("{}", encoding="utf-8")


def _write_ball_event_artifacts(
    run_dir: Path,
    *,
    frames: int = 1,
    contacts: int = 1,
    bounces: int = 1,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    ball_track = {
        "schema_version": 1,
        "fps": 60.0,
        "source": "tracknet",
        "frames": [
            {
                "t": float(idx) / 60.0,
                "xy": [320.0 + idx, 240.0],
                "conf": 0.91,
                "visible": True,
                "world_xyz": [0.0, 0.0, 1.0],
            }
            for idx in range(frames)
        ],
        "bounces": [{"t": 0.5 + idx, "world_xy": [0.1, 0.2]} for idx in range(bounces)],
    }
    contact_windows = {
        "schema_version": 1,
        "events": [
            {
                "type": "contact",
                "t": 0.25 + idx,
                "frame": 15 + idx,
                "player_id": 1,
                "confidence": 0.88,
                "sources": {"audio": 0.9, "wrist_vel": 0.8, "ball_inflection": 0.85},
                "window": {"t0": 0.2 + idx, "t1": 0.3 + idx, "importance": 1.0},
            }
            for idx in range(contacts)
        ],
    }
    (run_dir / "ball_track.json").write_text(json.dumps(ball_track), encoding="utf-8")
    (run_dir / "contact_windows.json").write_text(json.dumps(contact_windows), encoding="utf-8")


def _write_racket_pose_artifact(
    run_dir: Path,
    *,
    players: int = 1,
    frames: int = 1,
    contacts: int = 1,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "fps": 120.0,
        "players": [
            {
                "id": player_idx + 1,
                "paddle_dims_in": {"length": 15.5, "width": 7.5, "thickness": 0.55},
                "frames": [
                    {
                        "t": float(frame_idx) / 120.0,
                        "pose_se3": {
                            "R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                            "t": [0.0, 0.0, 1.0],
                        },
                        "conf": 0.9,
                    }
                    for frame_idx in range(frames)
                ],
                "contacts": [
                    {
                        "t": float(contact_idx) / 120.0,
                        "contact_point_face_cm": [0.0, 0.0],
                        "face_normal": [0.0, 0.0, 1.0],
                        "conf": 0.8,
                    }
                    for contact_idx in range(contacts)
                ],
            }
            for player_idx in range(players)
        ],
    }
    (run_dir / "racket_pose.json").write_text(json.dumps(payload), encoding="utf-8")


def _write_racket_pose_labels(
    labels_root: Path,
    name: str,
    *,
    face_normal: list[float] | None = None,
    contact_point_face_cm: list[float] | None = None,
    not_ground_truth: bool = False,
) -> None:
    labels_dir = labels_root / name / "labels"
    payload = {
        "schema_version": 1,
        "status": "draft_prototype_unverified" if not_ground_truth else "human_reviewed",
        "not_ground_truth": not_ground_truth,
        "annotation": {
            "target_file": "racket_pose.json",
            "items": [
                {
                    "frame_index": 0,
                    "player_id": 1,
                    "face_normal": face_normal or [0.0, 0.0, 1.0],
                    "contact_point_face_cm": contact_point_face_cm or [0.0, 0.0],
                    "status": "accepted",
                }
            ],
        },
    }
    (labels_dir / "racket_pose.json").write_text(json.dumps(payload), encoding="utf-8")


def test_ball_event_eval_fails_zero_event_evidence(tmp_path: Path) -> None:
    labels_root = tmp_path / "data" / "testclips"
    root = tmp_path / "runs" / "phase5"
    _write_ready_clip(labels_root, "clip_001")
    _write_ball_event_artifacts(root / "clip_001", frames=1, contacts=0, bounces=0)

    payload = ball_event_eval.evaluate(root, labels_root)

    assert payload.status == "fail"
    metrics = payload.clips[0].metrics
    assert metrics["ball_frames"].value == 1
    assert metrics["ball_frames"].gate == "presence_check.ball_frames_min: >= 1"
    assert metrics["ball_frames"].passed is True
    assert metrics["ball_frames"].status == "measured"
    assert metrics["contact_events"].value == 0
    assert metrics["contact_events"].gate == "presence_check.ball_contact_events_min: >= 1"
    assert metrics["contact_events"].passed is False
    assert metrics["bounce_events"].value == 0
    assert metrics["bounce_events"].gate == "presence_check.ball_bounce_events_min: >= 1"
    assert metrics["bounce_events"].passed is False


def test_ball_event_eval_fails_when_ball_frame_numeric_gate_fails(tmp_path: Path) -> None:
    labels_root = tmp_path / "data" / "testclips"
    root = tmp_path / "runs" / "phase5"
    _write_ready_clip(labels_root, "clip_001")
    _write_ball_event_artifacts(root / "clip_001", frames=0, contacts=0, bounces=0)

    payload = ball_event_eval.evaluate(root, labels_root)

    assert payload.status == "fail"
    metrics = payload.clips[0].metrics
    assert metrics["ball_frames"].value == 0
    assert metrics["ball_frames"].gate == "presence_check.ball_frames_min: >= 1"
    assert metrics["ball_frames"].passed is False


def test_racket_eval_without_reference_labels_is_not_measured(tmp_path: Path) -> None:
    labels_root = tmp_path / "data" / "testclips"
    root = tmp_path / "runs" / "phase6"
    _write_ready_clip(labels_root, "clip_001")
    _write_racket_pose_artifact(root / "clip_001", players=1, frames=1, contacts=0)

    payload = racket_eval.evaluate(root, labels_root)

    assert payload.status == "not_measured"
    assert payload.clips[0].status == "not_measured"
    metrics = payload.clips[0].metrics
    assert metrics["racket_players"].value == 1
    assert metrics["racket_players"].gate == "artifact_check.racket_players_recorded"
    assert metrics["racket_players"].passed is None
    assert metrics["racket_players"].status == "measured"
    assert metrics["racket_frames"].value == 1
    assert metrics["racket_frames"].gate == "artifact_check.racket_frames_recorded"
    assert metrics["racket_frames"].passed is None
    assert metrics["racket_contacts"].value == 0
    assert metrics["racket_contacts"].gate == "artifact_check.racket_contacts_recorded"
    assert metrics["racket_contacts"].passed is None
    assert metrics["racket_face_angle_p90_error_deg"].status == "not_measured"
    assert metrics["racket_face_angle_p90_error_deg"].passed is None
    assert "reviewed racket pose reference labels" in " ".join(payload.clips[0].notes)


def test_racket_eval_fails_when_reviewed_face_angle_gate_fails(tmp_path: Path) -> None:
    labels_root = tmp_path / "data" / "testclips"
    root = tmp_path / "runs" / "phase6"
    _write_ready_clip(labels_root, "clip_001")
    _write_racket_pose_labels(labels_root, "clip_001", face_normal=[1.0, 0.0, 0.0])
    _write_racket_pose_artifact(root / "clip_001", players=1, frames=1, contacts=1)

    payload = racket_eval.evaluate(root, labels_root)

    assert payload.status == "fail"
    metrics = payload.clips[0].metrics
    assert metrics["racket_face_angle_p90_error_deg"].gate == "label_check.racket_face_angle_p90_error_deg: <= 5.0"
    assert metrics["racket_face_angle_p90_error_deg"].passed is False
    assert metrics["racket_contact_point_p90_error_cm"].passed is True


def test_racket_eval_passes_with_reviewed_reference_pose_and_contact_labels(tmp_path: Path) -> None:
    labels_root = tmp_path / "data" / "testclips"
    root = tmp_path / "runs" / "phase6"
    _write_ready_clip(labels_root, "clip_001")
    _write_racket_pose_labels(labels_root, "clip_001")
    _write_racket_pose_artifact(root / "clip_001", players=1, frames=1, contacts=1)

    payload = racket_eval.evaluate(root, labels_root)

    assert payload.status == "pass"
    metrics = payload.clips[0].metrics
    assert metrics["racket_face_angle_p90_error_deg"].value == 0.0
    assert metrics["racket_face_angle_p90_error_deg"].passed is True
    assert metrics["racket_contact_point_p90_error_cm"].value == 0.0
    assert metrics["racket_contact_point_p90_error_cm"].passed is True


def test_ball_and_racket_evals_keep_incomplete_data_one_clips_not_measured(tmp_path: Path) -> None:
    labels_root = tmp_path / "data" / "testclips"
    _write_incomplete_clip(labels_root, "clip_001")

    ball_payload = ball_event_eval.evaluate(tmp_path / "runs" / "phase5", labels_root)
    racket_payload = racket_eval.evaluate(tmp_path / "runs" / "phase6", labels_root)

    assert ball_payload.status == "not_measured"
    assert ball_payload.clips[0].status == "not_measured"
    assert ball_payload.clips[0].metrics == {}
    assert "players.json" in ball_payload.clips[0].missing_label_files
    assert racket_payload.status == "not_measured"
    assert racket_payload.clips[0].status == "not_measured"
    assert racket_payload.clips[0].metrics == {}
    assert "players.json" in racket_payload.clips[0].missing_label_files

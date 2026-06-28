from __future__ import annotations

import json
from pathlib import Path

from threed.racketsport.eval import ball_event_eval, track_eval


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


def _write_ready_labels(labels_root: Path, clip: str) -> Path:
    labels_dir = labels_root / clip / "labels"
    labels_dir.mkdir(parents=True)
    for filename in REQUIRED_LABEL_FILES:
        (labels_dir / filename).write_text("{}", encoding="utf-8")
    return labels_dir


def _write_tracks(run_dir: Path, *, bbox: list[float]) -> None:
    run_dir.mkdir(parents=True)
    payload = {
        "schema_version": 1,
        "fps": 30.0,
        "players": [
            {
                "id": 1,
                "side": "near",
                "role": "left",
                "frames": [{"t": 0.0, "bbox": bbox, "world_xy": [0.0, 0.0], "conf": 0.9}],
            }
        ],
        "rally_spans": [],
    }
    (run_dir / "tracks.json").write_text(json.dumps(payload), encoding="utf-8")


def _write_player_labels(labels_dir: Path, *, not_ground_truth: bool) -> None:
    payload = {
        "schema_version": 1,
        "status": "draft_prototype_unverified" if not_ground_truth else "human_reviewed",
        "not_ground_truth": not_ground_truth,
        "annotation": {
            "items": [
                {
                    "frame": "frame_000000.jpg",
                    "bbox_xyxy": [10.0, 10.0, 50.0, 90.0],
                    "status": "accepted",
                    "id": "p1",
                }
            ],
            "target_file": "players.json",
        },
    }
    (labels_dir / "players.json").write_text(json.dumps(payload), encoding="utf-8")


def _write_ball_event_artifacts(run_dir: Path, *, xy: list[float]) -> None:
    run_dir.mkdir(parents=True)
    ball_track = {
        "schema_version": 1,
        "fps": 30.0,
        "source": "tracknet",
        "frames": [{"t": 0.0, "xy": xy, "conf": 0.9, "visible": True, "world_xyz": [0.0, 0.0, 1.0]}],
        "bounces": [{"t": 0.1, "world_xy": [0.0, 0.0]}],
    }
    contact_windows = {
        "schema_version": 1,
        "events": [
            {
                "type": "contact",
                "t": 0.0,
                "frame": 0,
                "player_id": 1,
                "confidence": 0.9,
                "sources": {"audio": 0.9, "wrist_vel": 0.8, "ball_inflection": 0.7},
                "window": {"t0": 0.0, "t1": 0.1, "importance": 1.0},
            }
        ],
    }
    (run_dir / "ball_track.json").write_text(json.dumps(ball_track), encoding="utf-8")
    (run_dir / "contact_windows.json").write_text(json.dumps(contact_windows), encoding="utf-8")


def _write_ball_labels(labels_dir: Path, *, not_ground_truth: bool) -> None:
    payload = {
        "schema_version": 1,
        "status": "draft_prototype_unverified" if not_ground_truth else "human_reviewed",
        "not_ground_truth": not_ground_truth,
        "annotation": {
            "items": [
                {
                    "frame": "frame_000000.jpg",
                    "frame_index": 0,
                    "xy_px": [10.0, 10.0],
                    "visible": True,
                    "status": "accepted",
                }
            ],
            "target_file": "ball.json",
        },
    }
    (labels_dir / "ball.json").write_text(json.dumps(payload), encoding="utf-8")


def test_track_eval_fails_when_reviewed_player_labels_are_not_matched(tmp_path: Path) -> None:
    labels_root = tmp_path / "data" / "testclips"
    root = tmp_path / "runs" / "phase2"
    labels_dir = _write_ready_labels(labels_root, "clip_001")
    _write_player_labels(labels_dir, not_ground_truth=False)
    _write_tracks(root / "clip_001", bbox=[200.0, 200.0, 260.0, 320.0])

    payload = track_eval.evaluate(root, labels_root).model_dump(mode="json")
    metrics = payload["clips"][0]["metrics"]

    assert payload["status"] == "fail"
    assert metrics["player_bbox_recall_iou50"]["gate"] == "label_check.track_player_bbox_recall_iou50: >= 0.9"
    assert metrics["player_bbox_recall_iou50"]["passed"] is False
    assert metrics["player_bbox_precision_iou50"]["passed"] is False


def test_track_eval_does_not_gate_on_draft_player_labels(tmp_path: Path) -> None:
    labels_root = tmp_path / "data" / "testclips"
    root = tmp_path / "runs" / "phase2"
    labels_dir = _write_ready_labels(labels_root, "clip_001")
    _write_player_labels(labels_dir, not_ground_truth=True)
    _write_tracks(root / "clip_001", bbox=[10.0, 10.0, 50.0, 90.0])

    payload = track_eval.evaluate(root, labels_root).model_dump(mode="json")
    metrics = payload["clips"][0]["metrics"]

    assert payload["status"] == "pass"
    assert metrics["player_bbox_recall_iou50"]["status"] == "not_measured"
    assert metrics["player_bbox_recall_iou50"]["passed"] is None
    assert "not_ground_truth" in " ".join(payload["clips"][0]["notes"])


def test_ball_event_eval_fails_when_reviewed_ball_labels_are_missed(tmp_path: Path) -> None:
    labels_root = tmp_path / "data" / "testclips"
    root = tmp_path / "runs" / "phase5"
    labels_dir = _write_ready_labels(labels_root, "clip_001")
    _write_ball_labels(labels_dir, not_ground_truth=False)
    _write_ball_event_artifacts(root / "clip_001", xy=[100.0, 100.0])

    payload = ball_event_eval.evaluate(root, labels_root).model_dump(mode="json")
    metrics = payload["clips"][0]["metrics"]

    assert payload["status"] == "fail"
    assert metrics["ball_f1_at_10px"]["gate"] == "label_check.ball_f1_at_10px: >= 0.9"
    assert metrics["ball_f1_at_10px"]["passed"] is False


def test_ball_event_eval_does_not_gate_on_draft_ball_labels(tmp_path: Path) -> None:
    labels_root = tmp_path / "data" / "testclips"
    root = tmp_path / "runs" / "phase5"
    labels_dir = _write_ready_labels(labels_root, "clip_001")
    _write_ball_labels(labels_dir, not_ground_truth=True)
    _write_ball_event_artifacts(root / "clip_001", xy=[10.0, 10.0])

    payload = ball_event_eval.evaluate(root, labels_root).model_dump(mode="json")
    metrics = payload["clips"][0]["metrics"]

    assert payload["status"] == "pass"
    assert metrics["ball_f1_at_10px"]["status"] == "not_measured"
    assert metrics["ball_f1_at_10px"]["passed"] is None
    assert "not_ground_truth" in " ".join(payload["clips"][0]["notes"])

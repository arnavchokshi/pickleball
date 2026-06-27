from __future__ import annotations

import json
from pathlib import Path

from threed.racketsport.eval import body_eval, physics_eval


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


def _write_partial_clip(labels_root: Path, name: str) -> None:
    labels_dir = labels_root / name / "labels"
    labels_dir.mkdir(parents=True)
    (labels_dir / "court_corners.json").write_text("{}", encoding="utf-8")


def _smpl_motion_payload(*, players: int = 1, frames: int = 1, skate_free: bool = True) -> dict:
    return {
        "schema_version": 1,
        "model": "smplx",
        "fps": 60.0,
        "world_frame": "court_Z0",
        "players": [
            {
                "id": idx + 1,
                "betas": [0.0] * 10,
                "skate_free": skate_free,
                "physics": "physpt",
                "frames": [
                    {
                        "t": frame_idx / 60.0,
                        "global_orient": [0.0, 0.0, 0.0],
                        "body_pose": [0.0] * 63,
                        "left_hand_pose": [],
                        "right_hand_pose": [],
                        "transl_world": [0.0, 0.0, 0.0],
                        "joints_world": [[0.0, 0.0, 0.0]],
                        "joint_conf": [0.9],
                        "foot_contact": {"left": frame_idx == 0, "right": False},
                        "grf": [[0.0, 0.0, 1.0]],
                    }
                    for frame_idx in range(frames)
                ],
            }
            for idx in range(players)
        ],
    }


def _write_body_artifacts(run_dir: Path, *, smpl_players: int = 1, skeleton_players: int = 1) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    skeleton = {
        "schema_version": 1,
        "joint_names": ["pelvis"],
        "preview_only": True,
        "players": [
            {"id": idx + 1, "frames": [{"t": 0.0, "joints_world": [[0.0, 0.0, 0.0]], "joint_conf": [0.9]}]}
            for idx in range(skeleton_players)
        ],
    }
    (run_dir / "smpl_motion.json").write_text(
        json.dumps(_smpl_motion_payload(players=smpl_players)),
        encoding="utf-8",
    )
    (run_dir / "skeleton3d.json").write_text(json.dumps(skeleton), encoding="utf-8")


def _write_physics_artifact(run_dir: Path, *, players: int = 1, frames: int = 1, skate_free: bool = True) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "smpl_motion.json").write_text(
        json.dumps(_smpl_motion_payload(players=players, frames=frames, skate_free=skate_free)),
        encoding="utf-8",
    )


def test_body_eval_uses_numeric_gates_for_measured_body_counts(tmp_path: Path) -> None:
    labels_root = tmp_path / "data" / "testclips"
    root = tmp_path / "runs" / "phase3"
    _write_ready_clip(labels_root, "clip_001")
    _write_body_artifacts(root / "clip_001")

    payload = body_eval.evaluate(root, labels_root).model_dump(mode="json")

    metrics = payload["clips"][0]["metrics"]
    assert payload["status"] == "pass"
    assert metrics["smpl_players"]["gate"] == "body_smpl_players_min: >= 1"
    assert metrics["smpl_players"]["passed"] is True
    assert metrics["smpl_frames"]["gate"] == "body_smpl_frames_min: >= 1"
    assert metrics["smpl_frames"]["passed"] is True
    assert metrics["skeleton_players"]["gate"] == "body_skeleton_players_min: >= 1"
    assert metrics["skeleton_players"]["passed"] is True


def test_body_eval_numeric_gate_failure_fails_ready_clip(tmp_path: Path) -> None:
    labels_root = tmp_path / "data" / "testclips"
    root = tmp_path / "runs" / "phase3"
    _write_ready_clip(labels_root, "clip_001")
    _write_body_artifacts(root / "clip_001", smpl_players=0)

    payload = body_eval.evaluate(root, labels_root).model_dump(mode="json")

    metrics = payload["clips"][0]["metrics"]
    assert payload["status"] == "fail"
    assert payload["clips"][0]["status"] == "fail"
    assert metrics["smpl_players"]["value"] == 0
    assert metrics["smpl_players"]["gate"] == "body_smpl_players_min: >= 1"
    assert metrics["smpl_players"]["passed"] is False
    assert metrics["smpl_frames"]["passed"] is False


def test_physics_eval_uses_numeric_gates_for_measured_physics_counts(tmp_path: Path) -> None:
    labels_root = tmp_path / "data" / "testclips"
    root = tmp_path / "runs" / "phase4"
    _write_ready_clip(labels_root, "clip_001")
    _write_physics_artifact(root / "clip_001", frames=2)

    payload = physics_eval.evaluate(root, labels_root).model_dump(mode="json")

    metrics = payload["clips"][0]["metrics"]
    assert payload["status"] == "pass"
    assert metrics["smpl_players"]["gate"] == "physics_smpl_players_min: >= 1"
    assert metrics["smpl_frames"]["gate"] == "physics_smpl_frames_min: >= 1"
    assert metrics["foot_contact_frames"]["gate"] == "physics_foot_contact_frames_observed: >= 0"
    assert metrics["skate_free_players"]["gate"] == "physics_skate_free_players_min: >= 1"
    assert metrics["grf_frames"]["gate"] == "physics_grf_frames_observed: >= 0"
    assert metrics["physics_modes"]["gate"] == "recorded for later physics gates"


def test_physics_eval_numeric_gate_failure_fails_ready_clip(tmp_path: Path) -> None:
    labels_root = tmp_path / "data" / "testclips"
    root = tmp_path / "runs" / "phase4"
    _write_ready_clip(labels_root, "clip_001")
    _write_physics_artifact(root / "clip_001", skate_free=False)

    payload = physics_eval.evaluate(root, labels_root).model_dump(mode="json")

    metrics = payload["clips"][0]["metrics"]
    assert payload["status"] == "fail"
    assert payload["clips"][0]["status"] == "fail"
    assert metrics["skate_free_players"]["value"] == 0
    assert metrics["skate_free_players"]["gate"] == "physics_skate_free_players_min: >= 1"
    assert metrics["skate_free_players"]["passed"] is False


def test_incomplete_data_1_clips_remain_not_measured_without_gate_metrics(tmp_path: Path) -> None:
    labels_root = tmp_path / "data" / "testclips"
    _write_partial_clip(labels_root, "clip_partial")

    body_payload = body_eval.evaluate(tmp_path / "runs" / "phase3", labels_root).model_dump(mode="json")
    physics_payload = physics_eval.evaluate(tmp_path / "runs" / "phase4", labels_root).model_dump(mode="json")

    assert body_payload["status"] == "not_measured"
    assert body_payload["clips"][0]["status"] == "not_measured"
    assert body_payload["clips"][0]["metrics"] == {}
    assert physics_payload["status"] == "not_measured"
    assert physics_payload["clips"][0]["status"] == "not_measured"
    assert physics_payload["clips"][0]["metrics"] == {}

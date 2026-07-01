from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from threed.racketsport.body_world_label_packet import build_body_world_label_packet


def _smpl_motion() -> dict:
    return {
        "schema_version": 1,
        "model": "smplx",
        "fps": 30.0,
        "world_frame": "court_Z0",
        "players": [
            {
                "id": 7,
                "betas": [0.0] * 10,
                "skate_free": True,
                "physics": "pending",
                "frames": [
                    {
                        "t": 1.0 / 30.0,
                        "global_orient": [0.0, 0.0, 0.0],
                        "body_pose": [0.0] * 63,
                        "transl_world": [0.0, 0.0, 0.0],
                        "track_world_xy": [0.0, -3.0],
                        "joints_world": [[0.0, 0.0, 0.1], [0.2, 0.0, 1.4]],
                        "mesh_vertices_world": [[0.0, 0.0, 0.0]],
                        "joint_conf": [0.9, 0.8],
                        "foot_contact": {"left": False, "right": False},
                        "grf": None,
                    }
                ],
            }
        ],
    }


def _skeleton3d() -> dict:
    return {
        "schema_version": 1,
        "preview_only": True,
        "joint_names": ["pelvis", "neck"],
        "players": [
            {
                "id": 7,
                "frames": [
                    {
                        "t": 1.0 / 30.0,
                        "joints_world": [[0.0, 0.0, 0.1], [0.2, 0.0, 1.4]],
                        "joint_conf": [0.9, 0.8],
                    }
                ],
            }
        ],
    }


def _body_compute_execution() -> dict:
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_body_compute_execution",
        "scheduled_frames": [{"frame_idx": 1, "target_player_ids": [7]}],
        "summary": {"scheduled_frame_count": 1, "scheduled_player_frame_count": 1},
    }


def test_body_world_label_packet_lists_prediction_samples_without_claiming_ground_truth() -> None:
    payload = build_body_world_label_packet(
        clip="clip_001",
        smpl_motion=_smpl_motion(),
        skeleton3d=_skeleton3d(),
        body_compute_execution=_body_compute_execution(),
        source_video="source.mp4",
        suggested_label_path="labels/body_world_joints.json",
    )

    assert payload["artifact_type"] == "racketsport_body_world_label_packet"
    assert payload["status"] == "needs_review"
    assert payload["not_ground_truth"] is True
    assert payload["trusted_for_world_mpjpe"] is False
    assert payload["suggested_label_path"] == "labels/body_world_joints.json"
    assert payload["summary"]["sample_count"] == 1
    assert payload["summary"]["joint_count_min"] == 2
    assert payload["joint_names"] == ["pelvis", "neck"]
    sample = payload["samples"][0]
    assert sample["frame_index"] == 1
    assert sample["player_id"] == 7
    assert sample["predicted_joints_world"] == [[0.0, 0.0, 0.1], [0.2, 0.0, 1.4]]
    assert "joints_world" not in sample


def test_body_world_label_packet_prefers_explicit_frame_idx() -> None:
    smpl_motion = _smpl_motion()
    smpl_motion["players"][0]["frames"][0]["frame_idx"] = 1
    smpl_motion["players"][0]["frames"][0]["t"] = 0.0

    payload = build_body_world_label_packet(
        clip="clip_001",
        smpl_motion=smpl_motion,
        skeleton3d=_skeleton3d(),
        body_compute_execution=_body_compute_execution(),
    )

    assert payload["status"] == "needs_review"
    assert payload["samples"][0]["frame_index"] == 1


def test_body_world_label_packet_preserves_temporal_smoothing_reset_metadata() -> None:
    smpl_motion = _smpl_motion()
    smpl_motion["players"][0]["frames"][0]["temporal_smoothing_reset"] = True

    payload = build_body_world_label_packet(
        clip="clip_001",
        smpl_motion=smpl_motion,
        skeleton3d=_skeleton3d(),
        body_compute_execution=_body_compute_execution(),
    )

    assert payload["samples"][0]["temporal_smoothing_reset"] is True


def test_body_world_label_packet_includes_representative_review_plan() -> None:
    smpl_motion = _smpl_motion()
    frames = []
    for frame_index in range(40):
        frame = dict(smpl_motion["players"][0]["frames"][0])
        frame["frame_idx"] = frame_index
        frame["t"] = frame_index / 30.0
        frame["transl_world"] = [float(frame_index) * 0.1, 0.0, 0.0]
        frames.append(frame)
    smpl_motion["players"][0]["frames"] = frames
    body_compute_execution = {
        "schema_version": 1,
        "artifact_type": "racketsport_body_compute_execution",
        "scheduled_frames": [{"frame_idx": frame_index, "target_player_ids": [7]} for frame_index in range(40)],
        "summary": {"scheduled_frame_count": 40, "scheduled_player_frame_count": 40},
    }

    payload = build_body_world_label_packet(
        clip="clip_001",
        smpl_motion=smpl_motion,
        skeleton3d=_skeleton3d(),
        body_compute_execution=body_compute_execution,
    )

    review_plan = payload["review_plan"]
    assert review_plan["expected_sample_count"] == 40
    assert review_plan["required_sample_count"] == 20
    assert review_plan["selected_sample_count"] == 20
    assert review_plan["min_sample_count"] == 20
    assert review_plan["min_coverage_ratio"] == 0.1
    assert len(review_plan["selected_sample_ids"]) == 20
    assert set(review_plan["selected_sample_ids"]).issubset({sample["sample_id"] for sample in payload["samples"]})


def test_build_body_world_label_packet_cli_writes_packet(tmp_path: Path) -> None:
    smpl = tmp_path / "smpl_motion.json"
    skeleton = tmp_path / "skeleton3d.json"
    execution = tmp_path / "body_compute_execution.json"
    out = tmp_path / "body_world_label_packet.json"
    smpl.write_text(json.dumps(_smpl_motion()), encoding="utf-8")
    skeleton.write_text(json.dumps(_skeleton3d()), encoding="utf-8")
    execution.write_text(json.dumps(_body_compute_execution()), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/build_body_world_label_packet.py",
            "--clip",
            "clip_001",
            "--smpl-motion",
            str(smpl),
            "--skeleton3d",
            str(skeleton),
            "--body-compute-execution",
            str(execution),
            "--source-video",
            "source.mp4",
            "--out",
            str(out),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert completed.stderr == ""
    assert json.loads(completed.stdout)["trusted_for_world_mpjpe"] is False
    assert json.loads(out.read_text(encoding="utf-8"))["summary"]["sample_count"] == 1

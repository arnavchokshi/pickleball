from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from threed.racketsport.racket_physics_estimate import build_racket_physics_estimate


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _contact_windows(*times: float) -> dict:
    return {
        "schema_version": 1,
        "events": [
            {
                "type": "contact",
                "t": t,
                "frame": int(round(t * 60.0)),
                "player_id": None,
                "confidence": 1.0,
                "sources": {"human_review": 1.0, "wrist_vel": 0.0, "ball_inflection": 0.0},
                "window": {"t0": t - 0.04, "t1": t + 0.04, "importance": 1.0},
            }
            for t in times
        ],
    }


def _ball_track() -> dict:
    samples = [
        (0.10, [-0.30, 0.0, 1.0]),
        (0.15, [-0.15, 0.0, 1.0]),
        (0.20, [0.0, 0.0, 1.0]),
        (0.25, [-0.15, 0.0, 1.0]),
        (0.30, [-0.30, 0.0, 1.0]),
    ]
    return {
        "schema_version": 1,
        "fps": 60.0,
        "source": "synthetic_filled_world_track",
        "frames": [
            {"t": t, "frame": int(round(t * 60.0)), "world_xyz": xyz, "conf": 0.9, "visible": True}
            for t, xyz in samples
        ],
    }


def _skeleton() -> dict:
    return {
        "schema_version": 1,
        "joint_names": ["right_elbow", "right_wrist", "left_elbow", "left_wrist"],
        "players": [
            {
                "id": 7,
                "frames": [
                    {
                        "t": 0.15,
                        "joints_world": [[-0.42, -0.05, 1.0], [-0.22, -0.05, 1.0], [1.0, 0.0, 1.0], [1.2, 0.0, 1.0]],
                        "joint_conf": [0.9, 0.9, 0.9, 0.9],
                    },
                    {
                        "t": 0.20,
                        "joints_world": [[-0.30, 0.0, 1.0], [-0.15, 0.0, 1.0], [1.0, 0.0, 1.0], [1.2, 0.0, 1.0]],
                        "joint_conf": [0.9, 0.9, 0.9, 0.9],
                    },
                    {
                        "t": 0.25,
                        "joints_world": [[-0.22, 0.05, 1.0], [-0.08, 0.05, 1.0], [1.0, 0.0, 1.0], [1.2, 0.0, 1.0]],
                        "joint_conf": [0.9, 0.9, 0.9, 0.9],
                    },
                ],
            }
        ],
    }


def test_racket_physics_estimate_derives_face_normal_and_wrist_constrained_pose() -> None:
    artifact = build_racket_physics_estimate(
        clip_id="clip_001",
        contact_windows=_contact_windows(0.20),
        ball_track=_ball_track(),
        skeleton3d=_skeleton(),
        ball_source_path="ball_track_filled.json",
        skeleton_source_path="skeleton3d.json",
    )

    assert artifact["artifact_type"] == "racketsport_racket_pose_estimate"
    assert artifact["physics_derived"] is True
    assert artifact["never_canonical_racket_pose"] is True
    assert artifact["summary"]["reviewed_contact_count"] == 1
    assert artifact["summary"]["estimate_count"] == 1
    assert artifact["summary"]["outgoing_hemisphere_fraction"] == 1.0
    assert artifact["summary"]["plausible_reach_fraction"] == 1.0

    estimate = artifact["estimates"][0]
    assert estimate["source"] == "physics_delta_v_wrist_swing_preview"
    assert estimate["face_normal_world"] == pytest.approx([-1.0, 0.0, 0.0], abs=1e-6)
    assert estimate["position_world"] == pytest.approx([0.0, 0.0, 1.0], abs=1e-6)
    assert estimate["selected_wrist"]["player_id"] == 7
    assert estimate["selected_wrist"]["side"] == "right"
    assert estimate["reach"]["plausible"] is True
    assert estimate["uncertainty"]["normal_angle_bound_deg"] > 0.0
    assert estimate["uncertainty"]["restitution_range"] == [0.55, 0.9]
    assert estimate["orientation_basis"]["normal"] == pytest.approx(estimate["face_normal_world"], abs=1e-6)


def test_racket_physics_estimate_fails_closed_when_ball_samples_are_sparse() -> None:
    sparse_ball = {
        "schema_version": 1,
        "fps": 60.0,
        "frames": [
            {"t": 0.18, "frame": 11, "world_xyz": [-0.05, 0.0, 1.0], "visible": True},
            {"t": 0.20, "frame": 12, "world_xyz": [0.0, 0.0, 1.0], "visible": True},
        ],
    }

    artifact = build_racket_physics_estimate(
        clip_id="clip_001",
        contact_windows=_contact_windows(0.20),
        ball_track=sparse_ball,
        skeleton3d=_skeleton(),
    )

    assert artifact["summary"]["reviewed_contact_count"] == 1
    assert artifact["summary"]["estimate_count"] == 0
    assert artifact["summary"]["skipped_contact_count"] == 1
    assert artifact["estimates"] == []
    assert artifact["skipped_contacts"][0]["reason"] == "insufficient_ball_velocity_samples"
    assert "insufficient_ball_velocity_samples" in artifact["blockers"]


def test_racket_physics_estimate_reports_temporal_smoothness_for_multiple_contacts() -> None:
    ball = _ball_track()
    shifted_frames = [
        {
            **frame,
            "t": frame["t"] + 0.20,
            "frame": int(round((frame["t"] + 0.20) * 60.0)),
            "world_xyz": [frame["world_xyz"][0], frame["world_xyz"][1] + 0.05, frame["world_xyz"][2]],
        }
        for frame in _ball_track()["frames"]
    ]
    ball["frames"] = [*ball["frames"], *shifted_frames]
    skeleton = _skeleton()
    skeleton["players"][0]["frames"].extend(
        {
            **frame,
            "t": frame["t"] + 0.20,
            "joints_world": [[joint[0], joint[1] + 0.05, joint[2]] for joint in frame["joints_world"]],
        }
        for frame in _skeleton()["players"][0]["frames"]
    )

    artifact = build_racket_physics_estimate(
        clip_id="clip_001",
        contact_windows=_contact_windows(0.20, 0.40),
        ball_track=ball,
        skeleton3d=skeleton,
    )

    assert artifact["summary"]["estimate_count"] == 2
    assert artifact["summary"]["temporal_smoothness"]["pair_count"] == 1
    assert artifact["summary"]["temporal_smoothness"]["max_adjacent_normal_angle_deg"] == pytest.approx(0.0, abs=1e-6)


def test_build_racket_physics_estimates_cli_writes_preview_artifact(tmp_path: Path) -> None:
    contact_path = _write_json(tmp_path / "contact_windows.json", _contact_windows(0.20))
    ball_path = _write_json(tmp_path / "ball_track_filled.json", _ball_track())
    skeleton_path = _write_json(tmp_path / "skeleton3d.json", _skeleton())
    out_dir = tmp_path / "out"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/build_racket_physics_estimates.py",
            "--clip",
            "clip_001",
            "--contact-windows",
            str(contact_path),
            "--ball-track",
            str(ball_path),
            "--skeleton3d",
            str(skeleton_path),
            "--out-dir",
            str(out_dir),
        ],
        cwd=Path(__file__).resolve().parents[2],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    artifact_path = out_dir / "clip_001" / "racket_pose_estimate.json"
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert payload["artifact_type"] == "racketsport_racket_pose_estimate"
    assert payload["summary"]["estimate_count"] == 1
    stdout = json.loads(completed.stdout)
    assert stdout["summary"]["total_estimates"] == 1
    assert stdout["artifacts"][0]["racket_pose_estimate"] == str(artifact_path)

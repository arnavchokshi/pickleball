from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from threed.racketsport.wrist_velocity_peaks import build_wrist_velocity_peaks_from_skeleton


def _skeleton_payload(*, joint_names: list[str]) -> dict:
    return {
        "schema_version": 1,
        "joint_names": joint_names,
        "preview_only": True,
        "players": [
            {
                "id": 7,
                "frames": [
                    {
                        "t": 0.0,
                        "joints_world": [[0.0, 0.0, 0.0] for _ in joint_names],
                        "joint_conf": [0.9 for _ in joint_names],
                    },
                    {
                        "t": 0.05,
                        "joints_world": [[0.0, 0.0, 0.0] for _ in joint_names],
                        "joint_conf": [0.9 for _ in joint_names],
                    },
                    {
                        "t": 0.10,
                        "joints_world": [[0.0, 0.0, 0.0] for _ in joint_names],
                        "joint_conf": [0.9 for _ in joint_names],
                    },
                ],
            }
        ],
    }


def test_wrist_velocity_builder_detects_semantic_wrist_speed_peak() -> None:
    payload = _skeleton_payload(joint_names=["pelvis", "left_wrist", "right_wrist"])
    payload["players"][0]["frames"][1]["joints_world"][1] = [0.5, 0.0, 1.0]
    payload["players"][0]["frames"][2]["joints_world"][1] = [0.55, 0.0, 1.0]

    artifact = build_wrist_velocity_peaks_from_skeleton(payload, min_speed_mps=4.0, min_separation_s=0.1)

    assert artifact["artifact_type"] == "racketsport_wrist_velocity_peaks"
    assert artifact["status"] == "review_only"
    assert artifact["not_gate_verified"] is True
    assert artifact["trusted_for_contact"] is False
    assert artifact["summary"]["peak_count"] == 1
    assert artifact["joint_mapping"] == {"left_wrist": 1, "right_wrist": 2}
    assert artifact["peaks"][0]["time_s"] == 0.05
    assert artifact["peaks"][0]["player_id"] == 7
    assert artifact["peaks"][0]["wrist_side"] == "left"
    assert artifact["peaks"][0]["source"] == "smoothed_wrist_speed_world_joints"
    assert artifact["peaks"][0]["speed_mps"] >= 4.0
    assert artifact["summary"]["speed_smoothing_window_frames"] == 3


def test_wrist_velocity_builder_blocks_when_wrist_joint_mapping_is_missing() -> None:
    payload = _skeleton_payload(joint_names=["sam3dbody_joint_000", "sam3dbody_joint_001"])

    artifact = build_wrist_velocity_peaks_from_skeleton(payload)

    assert artifact["status"] == "blocked"
    assert artifact["summary"]["peak_count"] == 0
    assert artifact["blockers"] == ["missing_wrist_joint_mapping"]
    assert artifact["warnings"] == ["missing_wrist_joint_mapping"]
    assert artifact["peaks"] == []


def test_wrist_velocity_builder_uses_sam3d_mhr70_semantic_adapter() -> None:
    payload = _skeleton_payload(joint_names=[f"sam3dbody_joint_{index:03d}" for index in range(70)])
    payload["players"][0]["frames"][1]["joints_world"][62] = [0.5, 0.0, 1.0]
    payload["players"][0]["frames"][2]["joints_world"][62] = [0.55, 0.0, 1.0]

    artifact = build_wrist_velocity_peaks_from_skeleton(payload, min_speed_mps=4.0, min_separation_s=0.1)

    assert artifact["status"] == "review_only"
    assert artifact["summary"]["peak_count"] == 1
    assert artifact["joint_mapping"] == {"right_wrist": 10, "left_wrist": 11}
    assert artifact["peaks"][0]["player_id"] == 7
    assert artifact["peaks"][0]["wrist_side"] == "left"


def test_wrist_velocity_builder_suppresses_peaks_per_player_and_side() -> None:
    payload = _skeleton_payload(joint_names=["pelvis", "left_wrist", "right_wrist"])
    payload["players"].append(
        {
            "id": 8,
            "frames": [
                {
                    "t": 0.0,
                    "joints_world": [[0.0, 0.0, 0.0] for _name in payload["joint_names"]],
                    "joint_conf": [0.9 for _name in payload["joint_names"]],
                },
                {
                    "t": 0.05,
                    "joints_world": [[0.0, 0.0, 0.0] for _name in payload["joint_names"]],
                    "joint_conf": [0.9 for _name in payload["joint_names"]],
                },
                {
                    "t": 0.10,
                    "joints_world": [[0.0, 0.0, 0.0] for _name in payload["joint_names"]],
                    "joint_conf": [0.9 for _name in payload["joint_names"]],
                },
            ],
        }
    )
    payload["players"][0]["frames"][1]["joints_world"][1] = [0.45, 0.0, 1.0]
    payload["players"][0]["frames"][2]["joints_world"][1] = [0.50, 0.0, 1.0]
    payload["players"][0]["frames"][1]["joints_world"][2] = [0.30, 0.0, 1.0]
    payload["players"][0]["frames"][2]["joints_world"][2] = [0.35, 0.0, 1.0]
    payload["players"][1]["frames"][1]["joints_world"][1] = [0.70, 0.0, 1.0]
    payload["players"][1]["frames"][2]["joints_world"][1] = [0.75, 0.0, 1.0]

    artifact = build_wrist_velocity_peaks_from_skeleton(payload, min_speed_mps=4.0, min_separation_s=0.1)

    assert artifact["summary"]["peak_count"] == 3
    assert {(peak["player_id"], peak["wrist_side"]) for peak in artifact["peaks"]} == {
        (7, "left"),
        (7, "right"),
        (8, "left"),
    }


def test_build_wrist_velocity_peaks_cli_writes_artifact(tmp_path: Path) -> None:
    skeleton = _skeleton_payload(joint_names=["pelvis", "left_wrist"])
    skeleton["players"][0]["frames"][1]["joints_world"][1] = [0.5, 0.0, 1.0]
    skeleton_path = tmp_path / "skeleton3d.json"
    out_path = tmp_path / "wrist_velocity_peaks.json"
    skeleton_path.write_text(json.dumps(skeleton), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/build_wrist_velocity_peaks.py",
            "--skeleton3d",
            str(skeleton_path),
            "--out",
            str(out_path),
            "--min-speed-mps",
            "4.0",
        ],
        cwd=Path(__file__).resolve().parents[2],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert "wrote" in completed.stdout
    assert payload["artifact_type"] == "racketsport_wrist_velocity_peaks"
    assert payload["summary"]["peak_count"] == 1

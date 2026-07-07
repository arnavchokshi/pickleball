from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _write_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    skeleton = {
        "schema_version": 1,
        "artifact_type": "synthetic_skeleton3d",
        "fps": 30,
        "joint_names": ["left_hip", "right_hip", "left_ankle", "right_ankle"],
        "players": [
            {
                "id": "p1",
                "frames": [
                    {
                        "frame_idx": 0,
                        "t": 0.0,
                        "joints_world": [
                            [1.1, 0.0, 1.05],
                            [1.3, 0.0, 1.05],
                            [1.08, 0.0, 0.05],
                            [1.32, 0.0, 0.05],
                        ],
                        "joint_conf": [1.0, 1.0, 1.0, 1.0],
                    }
                ],
            }
        ],
    }
    tracks = {
        "schema_version": 1,
        "fps": 30,
        "players": [{"id": "p1", "frames": [{"frame_idx": 0, "t": 0.0, "world_xy": [1.0, 0.0]}]}],
    }
    phases = {
        "artifact_type": "foot_contact_phases",
        "schema_version": 1,
        "phase_count": 1,
        "phases": [
            {
                "player_id": "p1",
                "foot": "left",
                "start_frame_index": 0,
                "end_frame_index": 0,
                "frame_indices": [0],
                "frame_count": 1,
                "anchor_position_xyz": [0.0, 0.0, 0.0],
                "min_confidence": 0.95,
                "max_height_m": 0.01,
                "max_speed_mps": 0.10,
                "source_thresholds": {"min_confidence": 0.20},
                "source_phase_foot": "left",
                "foot_assignment": "per_foot_body_contact",
                "assignment_evidence": {"body_detector_agreement": 0.95},
            }
        ],
    }
    skeleton_path = tmp_path / "skeleton3d.json"
    tracks_path = tmp_path / "tracks.json"
    phases_path = tmp_path / "foot_contact_phases.json"
    skeleton_path.write_text(json.dumps(skeleton), encoding="utf-8")
    tracks_path.write_text(json.dumps(tracks), encoding="utf-8")
    phases_path.write_text(json.dumps(phases), encoding="utf-8")
    return skeleton_path, tracks_path, phases_path


def test_refine_body_grounding_cli_exposes_direct_help_reference() -> None:
    command_path = "scripts/racketsport/refine_body_grounding.py"

    completed = subprocess.run(
        [sys.executable, command_path, "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--skeleton" in completed.stdout
    assert "--tracks" in completed.stdout
    assert "--foot-contact-phases" in completed.stdout
    assert "--out-dir" in completed.stdout
    assert "--xy-translation-enabled" in completed.stdout


def test_refine_body_grounding_cli_writes_refined_payload_and_report(tmp_path: Path) -> None:
    command_path = "scripts/racketsport/refine_body_grounding.py"
    skeleton_path, tracks_path, phases_path = _write_inputs(tmp_path)
    out_dir = tmp_path / "grounding"

    completed = subprocess.run(
        [
            sys.executable,
            command_path,
            "--skeleton",
            str(skeleton_path),
            "--tracks",
            str(tracks_path),
            "--foot-contact-phases",
            str(phases_path),
            "--out-dir",
            str(out_dir),
            "--max-correction-warn-m",
            "1.0",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    refined = json.loads((out_dir / "skeleton3d.json").read_text(encoding="utf-8"))
    report = json.loads((out_dir / "body_grounding_refinement.json").read_text(encoding="utf-8"))
    assert refined["players"][0]["frames"][0]["joints_world"][2][2] == 0.0
    assert report["summary"]["foot_plane_residual_m"]["mean_abs_after"] == 0.0


def test_refine_body_grounding_cli_passes_xy_translation_flag(tmp_path: Path) -> None:
    command_path = "scripts/racketsport/refine_body_grounding.py"
    skeleton_path, tracks_path, phases_path = _write_inputs(tmp_path)
    out_dir = tmp_path / "grounding"

    completed = subprocess.run(
        [
            sys.executable,
            command_path,
            "--skeleton",
            str(skeleton_path),
            "--tracks",
            str(tracks_path),
            "--foot-contact-phases",
            str(phases_path),
            "--out-dir",
            str(out_dir),
            "--xy-translation-enabled",
            "false",
            "--max-correction-warn-m",
            "1.0",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    report = json.loads((out_dir / "body_grounding_refinement.json").read_text(encoding="utf-8"))
    assert report["policy"]["xy_translation_enabled"] is False

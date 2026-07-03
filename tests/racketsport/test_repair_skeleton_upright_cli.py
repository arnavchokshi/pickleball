from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _write_inputs(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    skeleton = {
        "schema_version": 1,
        "artifact_type": "racketsport_skeleton3d",
        "fps": 30.0,
        "source_model": "rtmw3d_x",
        "world_frame": "court_Z0",
        "preview_only": False,
        "joint_names": ["nose", "left_hip", "right_hip", "left_ankle", "right_ankle", "left_big_toe", "right_big_toe"],
        "provenance": {"lane": "A"},
        "players": [
            {
                "id": 1,
                "frames": [
                    {
                        "frame_idx": 0,
                        "t": 0.0,
                        "joints_world": [
                            [0.0, 1.6, 0.05],
                            [-0.12, 0.8, 0.05],
                            [0.12, 0.8, 0.05],
                            [-0.12, 0.05, 0.05],
                            [0.12, 0.05, 0.05],
                            [-0.18, 0.0, 0.05],
                            [0.18, 0.0, 0.05],
                        ],
                        "joint_conf": [0.9] * 7,
                    }
                ],
            }
        ],
    }
    calibration = {
        "schema_version": 1,
        "sport": "pickleball",
        "homography": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "intrinsics": {"fx": 100.0, "fy": 100.0, "cx": 50.0, "cy": 50.0, "dist": [], "source": "synthetic"},
        "extrinsics": {
            "R": [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
            "t": [0.0, 0.0, 2.0],
            "camera_height_m": 2.0,
        },
        "reprojection_error_px": {"median": 0.0, "p95": 0.0},
        "capture_quality": {"grade": "good", "reasons": []},
        "image_pts": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        "world_pts": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
    }
    (run_dir / "skeleton3d.json").write_text(json.dumps(skeleton), encoding="utf-8")
    (run_dir / "court_calibration.json").write_text(json.dumps(calibration), encoding="utf-8")


def test_repair_skeleton_upright_cli_exposes_direct_help_reference() -> None:
    command_path = "scripts/racketsport/repair_skeleton_upright.py"

    completed = subprocess.run(
        [sys.executable, command_path, "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--run-dir" in completed.stdout
    assert "--skeleton" in completed.stdout
    assert "--court-calibration" in completed.stdout
    assert "--overlay-scale-suspect-caption" in completed.stdout


def test_repair_skeleton_upright_cli_preserves_original_and_writes_provenance(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_inputs(run_dir)

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/repair_skeleton_upright.py",
            "--run-dir",
            str(run_dir),
            "--z-smoothing-radius",
            "1",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    summary = json.loads(completed.stdout)
    repaired = json.loads((run_dir / "skeleton3d.json").read_text(encoding="utf-8"))
    report = json.loads((run_dir / "skeleton_upright_repair.json").read_text(encoding="utf-8"))
    original = json.loads((run_dir / "skeleton3d.pre_upright.json").read_text(encoding="utf-8"))
    assert summary["selected_convention"] == "offset_row_times_R"
    assert original["players"][0]["frames"][0]["joints_world"][0][2] == 0.05
    assert repaired["players"][0]["frames"][0]["joints_world"][0][2] == 1.6
    assert repaired["provenance"]["skeleton_upright_repair"]["pre_upright_backup"].endswith("skeleton3d.pre_upright.json")
    assert report["metrics_after"]["feet_within_0_35m_rate"] == 1.0

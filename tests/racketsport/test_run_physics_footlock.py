from __future__ import annotations

import json
import subprocess
import sys

from tests.racketsport.test_foot_contact import JOINT_NAMES_65, _frame
from scripts.racketsport.run_physics_footlock import main


def _write_synthetic_skeleton(path):
    frames = [
        _frame(0, left_x=0.000, left_z=0.000),
        _frame(1, left_x=0.020, left_z=-0.010),
        _frame(2, left_x=0.040, left_z=0.005),
        _frame(3, left_x=0.300, left_z=0.250),
    ]
    source = {
        "schema_version": 1,
        "artifact_type": "synthetic_skeleton3d",
        "fps": 30,
        "joint_names": list(JOINT_NAMES_65),
        "players": [
            {
                "id": "p1",
                "frames": [
                    {
                        "frame_idx": frame.frame_index,
                        "t": frame.t,
                        "joints_world": frame.joints_world,
                        "joint_conf": frame.joint_conf,
                    }
                    for frame in frames
                ],
            }
        ],
    }
    path.write_text(json.dumps(source), encoding="utf-8")
    return source


def test_run_physics_footlock_writes_phase_metrics_and_corrected_artifact(tmp_path):
    input_path = tmp_path / "skeleton3d.json"
    source = _write_synthetic_skeleton(input_path)
    out_dir = tmp_path / "phys_foot"

    rc = main(["--input", str(input_path), "--clip-name", "synthetic", "--out-dir", str(out_dir)])

    assert rc == 0
    phases = json.loads((out_dir / "foot_contact_phases.json").read_text(encoding="utf-8"))
    report = json.loads((out_dir / "foot_slide_report.json").read_text(encoding="utf-8"))
    corrected = json.loads((out_dir / "physics_footlock.json").read_text(encoding="utf-8"))

    assert phases["clip"] == "synthetic"
    assert phases["phase_count"] == 1
    assert report["baseline_metrics"]["summary_by_player"]["p1"]["max_slide_mm"] > 3.0
    assert report["solved_metrics"]["summary_by_player"]["p1"]["max_slide_mm"] <= 3.0
    assert corrected["artifact_type"] == "physics_footlock"
    assert corrected["trust_band"]["gate_id"] == "foot_slide_floor_penetration_gate"
    assert corrected["players"][0]["frames"][0]["joints_world"] != source["players"][0]["frames"][1]["joints_world"]


def test_run_physics_footlock_can_execute_by_script_path(tmp_path):
    input_path = tmp_path / "skeleton3d.json"
    _write_synthetic_skeleton(input_path)
    out_dir = tmp_path / "script_path_out"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/run_physics_footlock.py",
            "--input",
            str(input_path),
            "--clip-name",
            "synthetic",
            "--out-dir",
            str(out_dir),
        ],
        check=False,
        cwd="/Users/arnavchokshi/Desktop/pickleball",
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert (out_dir / "physics_footlock.json").exists()

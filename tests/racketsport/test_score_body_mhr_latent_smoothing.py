from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from threed.racketsport.external_gt_body_prediction_schema import MHR70_JOINT_NAMES


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _frame(frame_idx: int, *, root_x: float, wrist_x: float, foot_z: float = 0.0) -> dict:
    index = {name: pos for pos, name in enumerate(MHR70_JOINT_NAMES)}
    joints = [[root_x, 0.0, 1.0] for _ in MHR70_JOINT_NAMES]
    for name in ("left_hip", "right_hip"):
        joints[index[name]] = [root_x, 0.0, 1.0]
    for name in ("left_ankle", "left_big_toe_tip", "left_small_toe_tip", "left_heel"):
        joints[index[name]] = [root_x, -0.15, foot_z]
    for name in ("right_ankle", "right_big_toe_tip", "right_small_toe_tip", "right_heel"):
        joints[index[name]] = [root_x, 0.15, foot_z]
    joints[index["left_wrist"]] = [wrist_x, -0.45, 1.2]
    joints[index["right_wrist"]] = [wrist_x, 0.45, 1.2]
    return {
        "frame_idx": frame_idx,
        "t": frame_idx / 30.0,
        "joints_world": joints,
        "joint_conf": [0.95] * len(MHR70_JOINT_NAMES),
    }


def _world(root_values: list[float], wrist_values: list[float], *, foot_z: float = 0.0) -> dict:
    frames = [
        _frame(idx, root_x=root_x, wrist_x=wrist_values[idx], foot_z=foot_z)
        for idx, root_x in enumerate(root_values)
    ]
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_virtual_world",
        "fps": 30.0,
        "joint_names": list(MHR70_JOINT_NAMES),
        "players": [{"id": "1", "frames": frames}],
        "world_frame": "court",
    }


def _write_run(run_dir: Path, world: dict) -> None:
    placement_frames = [
        {
            "frame_idx": idx,
            "t": idx / 30.0,
            "stance": True,
            "smoothed_world_xy": [0.0, 0.0],
        }
        for idx in range(len(world["players"][0]["frames"]))
    ]
    _write_json(run_dir / "virtual_world.json", world)
    _write_json(run_dir / "skeleton3d.json", world | {"artifact_type": "racketsport_skeleton3d"})
    _write_json(
        run_dir / "placement.json",
        {
            "schema_version": 1,
            "artifact_type": "racketsport_placement",
            "fps": 30.0,
            "players": [{"id": "1", "frames": placement_frames}],
            "summary": {},
        },
    )
    _write_json(
        run_dir / "body_joint_quality.json",
        {
            "schema_version": 1,
            "artifact_type": "racketsport_body_joint_quality",
            "summary": {},
        },
    )


def test_cli_help_references_scaffold_command_path() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/racketsport/score_body_mhr_latent_smoothing.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "candidate-run" in completed.stdout
    assert "proxy-world-joint" in completed.stdout


def test_scaffold_index_places_cli_in_body_category_with_direct_reference() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/list_scaffold_tools.py",
            "--root",
            ".",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    tools = {tool["command_path"]: tool for tool in payload["tools"]}
    entry = tools["scripts/racketsport/score_body_mhr_latent_smoothing.py"]
    assert entry["category"] == "body"
    assert entry["workstream"] == "BODY"
    assert entry["direct_cli_reference_test"] == "tests/racketsport/test_score_body_mhr_latent_smoothing.py"


def test_evaluate_candidate_runs_reports_acceptance_keys(tmp_path: Path) -> None:
    from scripts.racketsport import score_body_mhr_latent_smoothing as scoring

    raw_run = tmp_path / "raw"
    candidate_run = tmp_path / "lambda_0_1"
    raw_root = [0.0, 0.0, 0.045, -0.045, 0.045, 0.0, 0.0]
    smooth_root = [0.0, 0.0, 0.015, -0.015, 0.015, 0.0, 0.0]
    wrist = [0.0, 0.02, 0.08, 0.25, 0.08, 0.02, 0.0]
    _write_run(raw_run, _world(raw_root, wrist, foot_z=-0.01))
    _write_run(candidate_run, _world(smooth_root, wrist, foot_z=-0.01))

    report = scoring.evaluate_candidate_runs(
        raw_run_dir=raw_run,
        candidate_run_dirs={"0.1": candidate_run},
        out_dir=tmp_path / "out",
        wrist_min_peak_speed_mps=0.0,
    )

    assert report["measurement_status"] == "measured"
    assert report["measurement_mode"] == "decoded_candidate_run_dirs"
    player = report["players"]["1"]
    assert player["world_jitter_mm_per_frame2"]["feet"]["raw"]["rms"] > player["world_jitter_mm_per_frame2"]["feet"]["0.1"]["rms"]
    assert player["world_jitter_mm_per_frame2"]["wrists"]["raw"]["rms"] == pytest.approx(
        player["world_jitter_mm_per_frame2"]["wrists"]["0.1"]["rms"]
    )
    assert player["foot_slide_mm_per_frame"]["stance"]["raw"]["p95"] > player["foot_slide_mm_per_frame"]["stance"]["0.1"]["p95"]
    assert player["wrist_peak_delta_frames"]["0.1"]["max_abs_delta_frames"] == 0
    assert player["foot_slide_gate"]["raw"]["threshold_m"] == pytest.approx(0.03)
    assert player["foot_slide_gate"]["0.1"]["predicted_pass"] is True
    assert "phase_penetrates_ground" in player["phase_census"]["raw"]["rejection_reason_counts"]
    assert (tmp_path / "out" / "latent_smoothing_acceptance_report.json").is_file()


def test_run_blocks_without_candidates_or_proxy(tmp_path: Path) -> None:
    from scripts.racketsport import score_body_mhr_latent_smoothing as scoring

    raw_run = tmp_path / "raw"
    _write_run(raw_run, _world([0.0, 0.0, 0.02], [0.0, 0.01, 0.0]))

    parser = scoring.build_arg_parser()
    args = parser.parse_args(["--raw-run-dir", str(raw_run), "--out-dir", str(tmp_path / "out")])
    report = scoring.run(args)

    assert report["measurement_status"] == "blocked"
    assert report["blockers"] == ["missing_candidate_run_dirs"]

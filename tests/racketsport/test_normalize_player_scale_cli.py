from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _calibration() -> dict:
    return {
        "schema_version": 1,
        "sport": "pickleball",
        "image_size": [1920, 1080],
        "homography": [[1.0, 0.0, 960.0], [0.0, 1.0, 540.0], [0.0, 0.0, 1.0]],
        "intrinsics": {"fx": 1000.0, "fy": 1000.0, "cx": 960.0, "cy": 540.0, "dist": [], "source": "synthetic"},
        "extrinsics": {
            "R": [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
            "t": [0.0, 0.0, 10.0],
            "camera_height_m": 10.0,
        },
        "metric_confidence": "high",
    }


def _bbox_for_height(height_m: float, *, court_y_m: float, top: float = 100.0) -> list[float]:
    depth_m = 10.0 - court_y_m
    pixel_height = 1000.0 * height_m / depth_m
    return [500.0, top, 560.0, top + pixel_height]


def _write_run(run_dir: Path, *, frame_count: int = 70, conf: float = 0.9) -> None:
    run_dir.mkdir(parents=True)
    frames = []
    for frame_idx in range(frame_count):
        court_y = -2.0 + 0.02 * frame_idx
        frames.append(
            {
                "frame_idx": frame_idx,
                "t": frame_idx / 30.0,
                "bbox": _bbox_for_height(1.7, court_y_m=court_y),
                "world_xy": [0.0, court_y],
                "conf": conf,
            }
        )
    tracks = {"schema_version": 1, "fps": 30.0, "players": [{"id": 1, "frames": frames}]}
    skeleton = {
        "schema_version": 1,
        "artifact_type": "racketsport_skeleton3d",
        "fps": 30.0,
        "source_model": "synthetic",
        "world_frame": "court_Z0",
        "joint_names": ["nose", "pelvis", "left_ankle", "right_ankle"],
        "players": [
            {
                "id": 1,
                "frames": [
                    {
                        "frame_idx": 0,
                        "t": 0.0,
                        "transl_world": [0.0, 0.0, 0.45],
                        "joints_world": [
                            [0.0, 0.0, 0.9],
                            [0.0, 0.0, 0.45],
                            [-0.1, 0.0, 0.0],
                            [0.1, 0.0, 0.0],
                        ],
                        "joint_conf": [0.9, 0.9, 0.9, 0.9],
                    }
                ],
            }
        ],
    }
    (run_dir / "tracks.json").write_text(json.dumps(tracks), encoding="utf-8")
    (run_dir / "court_calibration.json").write_text(json.dumps(_calibration()), encoding="utf-8")
    (run_dir / "skeleton3d.json").write_text(json.dumps(skeleton), encoding="utf-8")


def test_normalize_player_scale_cli_exposes_direct_help_reference() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/racketsport/normalize_player_scale.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--run-dir" in completed.stdout
    assert "--tracks" in completed.stdout
    assert "--min-confidence" in completed.stdout
    assert "--window-spread-max-m" in completed.stdout


def test_normalize_player_scale_cli_writes_backup_estimate_and_provenance(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_run(run_dir)

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/normalize_player_scale.py",
            "--run-dir",
            str(run_dir),
            "--min-confidence",
            "0.5",
            "--samples-per-window",
            "10",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    summary = json.loads(completed.stdout)
    estimates = json.loads((run_dir / "player_scale_estimates.json").read_text(encoding="utf-8"))
    report = json.loads((run_dir / "player_scale_normalization_report.json").read_text(encoding="utf-8"))
    normalized = json.loads((run_dir / "skeleton3d.json").read_text(encoding="utf-8"))
    backup = json.loads((run_dir / "skeleton3d.pre_player_scale.json").read_text(encoding="utf-8"))
    assert summary["status"] == "normalized"
    assert estimates["players"]["1"]["height_m"] == report["players"]["1"]["estimated_height_m"]
    assert backup["players"][0]["frames"][0]["joints_world"][0][2] == 0.9
    assert report["players"]["1"]["scale_factor"] > 1.8
    assert normalized["provenance"]["player_scale_normalization"]["estimate_path"].endswith("player_scale_estimates.json")


def test_normalize_player_scale_cli_refuses_low_confidence_without_rewriting_skeleton(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_run(run_dir, frame_count=3, conf=0.1)
    original = (run_dir / "skeleton3d.json").read_text(encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/normalize_player_scale.py",
            "--run-dir",
            str(run_dir),
            "--min-confidence",
            "0.8",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    assert "low-confidence" in completed.stderr
    assert (run_dir / "player_scale_estimates.json").is_file()
    assert not (run_dir / "skeleton3d.pre_player_scale.json").exists()
    assert (run_dir / "skeleton3d.json").read_text(encoding="utf-8") == original

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_build_rally_metrics_cli_exposes_direct_help_reference() -> None:
    command_path = "scripts/racketsport/build_rally_metrics.py"

    completed = subprocess.run(
        [sys.executable, command_path, "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--run-dir" in completed.stdout
    assert "--out-dir" in completed.stdout


def test_build_rally_metrics_cli_writes_expected_artifacts(tmp_path: Path) -> None:
    command_path = "scripts/racketsport/build_rally_metrics.py"
    run_dir = tmp_path / "run"
    out_dir = tmp_path / "out"
    run_dir.mkdir()
    (run_dir / "virtual_world.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_type": "racketsport_virtual_world",
                "fps": 10,
                "summary": {"ball_frame_count": 2},
                "players": [{"id": 1, "frames": [{"t": 0.0, "track_world_xy": [0.0, 0.0]}]}],
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "court_zones.json").write_text(
        json.dumps({"schema_version": 1, "zones": {"court": [[-1, -1], [1, -1], [1, 1], [-1, 1]]}}),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [sys.executable, command_path, "--run-dir", str(run_dir), "--out-dir", str(out_dir)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert (out_dir / "rally_metrics.json").exists()
    assert (out_dir / "coaching_card_facts.json").exists()
    stdout = json.loads(completed.stdout)
    assert stdout["rally_metrics"] == str(out_dir / "rally_metrics.json")
    assert stdout["coaching_card_facts"] == str(out_dir / "coaching_card_facts.json")


def test_build_rally_metrics_cli_rejects_missing_run_dir(tmp_path: Path) -> None:
    command_path = "scripts/racketsport/build_rally_metrics.py"

    completed = subprocess.run(
        [
            sys.executable,
            command_path,
            "--run-dir",
            str(tmp_path / "missing"),
            "--out-dir",
            str(tmp_path / "out"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    assert "ERROR: run dir does not exist" in completed.stderr

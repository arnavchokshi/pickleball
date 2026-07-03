from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_apply_confidence_gate_cli_exposes_direct_help_reference() -> None:
    command_path = "scripts/racketsport/apply_confidence_gate.py"

    completed = subprocess.run(
        [sys.executable, command_path, "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--run-dir" in completed.stdout
    assert "--out" in completed.stdout
    assert "--confidence-threshold" in completed.stdout


def test_apply_confidence_gate_cli_rejects_missing_run_dir(tmp_path: Path) -> None:
    command_path = "scripts/racketsport/apply_confidence_gate.py"

    completed = subprocess.run(
        [
            sys.executable,
            command_path,
            "--run-dir",
            str(tmp_path / "missing"),
            "--out",
            str(tmp_path / "out"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    assert "ERROR: run dir does not exist" in completed.stderr


def test_calibrate_confidence_bands_cli_exposes_direct_help_reference() -> None:
    command_path = "scripts/racketsport/calibrate_confidence_bands.py"

    completed = subprocess.run(
        [sys.executable, command_path, "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--run-dir" in completed.stdout
    assert "--out" in completed.stdout
    assert "--confidence-threshold" in completed.stdout


def test_calibrate_confidence_bands_cli_rejects_missing_run_dir(tmp_path: Path) -> None:
    command_path = "scripts/racketsport/calibrate_confidence_bands.py"

    completed = subprocess.run(
        [
            sys.executable,
            command_path,
            "--run-dir",
            str(tmp_path / "missing"),
            "--out",
            str(tmp_path / "calibration_curves.json"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    assert "ERROR: run dir does not exist" in completed.stderr

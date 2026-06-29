from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


PYTHON_CLI_SCRIPTS = sorted([*Path("scripts").glob("*.py"), *Path("scripts/racketsport").glob("*.py")])


@pytest.mark.parametrize("script_path", PYTHON_CLI_SCRIPTS, ids=lambda path: path.as_posix())
def test_python_cli_help_runs_from_repo_root(script_path: Path) -> None:
    completed = subprocess.run(
        [sys.executable, str(script_path), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "usage:" in completed.stdout.lower()


def test_smoke_mujoco_mjx_help_does_not_import_optional_mjx_stack() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/racketsport/smoke_mujoco_mjx.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "Smoke-test MuJoCo MJX stepping" in completed.stdout


@pytest.mark.parametrize(
    ("script_path", "expected"),
    [
        ("scripts/racketsport/run_totnet_ball.py", "Run a public TOTNet checkpoint"),
        ("scripts/racketsport/train_tenniset_shot_baseline.py", "Train a small TenniSet external shot-class baseline"),
    ],
)
def test_optional_model_cli_help_does_not_import_numpy_with_no_site_packages(script_path: str, expected: str) -> None:
    completed = subprocess.run(
        [sys.executable, "-S", script_path, "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert expected in completed.stdout

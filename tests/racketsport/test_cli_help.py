from __future__ import annotations

import subprocess
import sys


def test_autolabel_compatibility_wrapper_help_runs_from_repo_root() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/autolabel.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "Bootstrap draft labels" in completed.stdout


def test_train_court_keypoint_heatmap_help_runs_from_repo_root() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/racketsport/train_court_keypoint_heatmap.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "Train a lightweight pickleball court-keypoint heatmap model" in completed.stdout


def test_smoke_mujoco_mjx_help_does_not_import_optional_mjx_stack() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/racketsport/smoke_mujoco_mjx.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "Smoke-test MuJoCo MJX stepping" in completed.stdout

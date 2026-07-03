from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


PYTHON_CLI_SCRIPTS = sorted([*Path("scripts").glob("*.py"), *Path("scripts/racketsport").glob("*.py")])
EXPLICIT_AUDIT_CLI_SCRIPTS = (
    "scripts/racketsport/apply_ball_physics_fill.py",
    "scripts/racketsport/benchmark_decode.py",
    "scripts/racketsport/benchmark_person_trackers.py",
    "scripts/racketsport/build_external_gt_aspset510_body_inputs.py",
    "scripts/racketsport/build_external_gt_aspset510_labels.py",
    "scripts/racketsport/build_paddle_true_corner_review.py",
    "scripts/racketsport/build_rally_spans.py",
    "scripts/racketsport/build_tiled_raw_pool.py",
    "scripts/racketsport/filter_ball_local_search.py",
    "scripts/racketsport/pool_parity_diagnostics.py",
    "scripts/racketsport/prepare_tracknetv3_finetune_dataset.py",
    "scripts/racketsport/profile_stage_runtime.py",
    "scripts/racketsport/review_input_server.py",
    "scripts/racketsport/rescore_body_ext3_grounding_consistent.py",
    "scripts/racketsport/run_external_gt_aspset510_body_inference.py",
    "scripts/racketsport/run_ball_tracking_eval_suite.py",
    "scripts/racketsport/run_yolo26_teacher.py",
    "scripts/racketsport/score_external_gt_aspset510_body_results.py",
    "scripts/racketsport/train_court_keypoint_heatmap.py",
    "scripts/racketsport/validate_metric_calibration_15pt.py",
    "scripts/racketsport/validate_rally_gating.py",
    "scripts/racketsport/verify_process_video_viewer.py",
)


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


@pytest.mark.parametrize("script_path", EXPLICIT_AUDIT_CLI_SCRIPTS)
def test_audit_tracked_python_cli_help_runs_from_repo_root(script_path: str) -> None:
    assert script_path in {path.as_posix() for path in PYTHON_CLI_SCRIPTS}

    completed = subprocess.run(
        [sys.executable, script_path, "--help"],
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

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest


SHELL_SCRIPTS = [
    Path("scripts/gpu-eval-run.sh"),
    Path("scripts/gpu-train-lock.sh"),
    Path("scripts/racketsport/setup_env.sh"),
    Path("scripts/racketsport/install_fast_sam_env.sh"),
    Path("scripts/racketsport/install_mujoco_mjx_env.sh"),
    Path("scripts/racketsport/run_fast_sam_benchmark.sh"),
]


def _require_flock() -> None:
    if shutil.which("flock") is None:
        pytest.skip("flock is not installed")


def test_shell_scripts_are_executable_and_parse():
    for script in SHELL_SCRIPTS:
        assert script.exists(), script
        assert os.access(script, os.X_OK), script
        subprocess.run(["bash", "-n", str(script)], check=True)


def test_fast_sam_wrapper_records_machine_readable_profile_metrics():
    script = Path("scripts/racketsport/run_fast_sam_benchmark.sh").read_text(encoding="utf-8")

    assert "profile_stdout.log" in script
    assert "benchmark_sam3dbody.py" in script
    assert "--profile-log" in script
    assert "sam3dbody_benchmark.json" in script


def test_fast_sam_wrapper_normalizes_relative_output_dir_before_cd():
    script = Path("scripts/racketsport/run_fast_sam_benchmark.sh").read_text(encoding="utf-8")

    assert 'case "$OUT_DIR" in' in script
    assert 'OUT_DIR="$ROOT/$OUT_DIR"' in script
    assert script.index('case "$OUT_DIR" in') < script.index('cd "$FAST_SAM_ROOT"')


def test_gpu_eval_run_repairs_stale_slot_uuid(tmp_path):
    _require_flock()

    lease_root = tmp_path / "gpu-lease"
    slots = lease_root / "slots"
    slots.mkdir(parents=True)
    (slots / "slot0.lock").write_text("", encoding="utf-8")
    env = {**os.environ, "GPU_LEASE_ROOT": str(lease_root), "CUDA_VISIBLE_DEVICES": "7"}

    completed = subprocess.run(
        ["bash", "scripts/gpu-eval-run.sh", "bash", "-lc", "printf '%s' \"$CUDA_VISIBLE_DEVICES\""],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    assert completed.stdout == "7"
    assert (slots / "slot0.uuid").read_text(encoding="utf-8").strip() == "7"


def test_gpu_eval_run_waits_for_full_gpu_training_lock(tmp_path):
    _require_flock()

    lease_root = tmp_path / "gpu-lease"
    marker = tmp_path / "train-started"
    env = {**os.environ, "GPU_LEASE_ROOT": str(lease_root), "CUDA_VISIBLE_DEVICES": "7"}
    train = subprocess.Popen(
        [
            "bash",
            "scripts/gpu-train-lock.sh",
            "bash",
            "-lc",
            f"printf started > {marker}; sleep 0.4",
        ],
        env=env,
    )

    try:
        deadline = time.monotonic() + 3.0
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert marker.exists()

        start = time.monotonic()
        completed = subprocess.run(
            ["bash", "scripts/gpu-eval-run.sh", "bash", "-lc", "printf eval"],
            check=True,
            capture_output=True,
            text=True,
            env=env,
            timeout=5,
        )
        elapsed = time.monotonic() - start

        assert completed.stdout == "eval"
        assert elapsed >= 0.25
    finally:
        train.wait(timeout=5)


def test_gpu_helpers_fail_closed_when_flock_is_missing(tmp_path):
    if shutil.which("flock") is not None:
        pytest.skip("flock is installed")

    env = {**os.environ, "GPU_LEASE_ROOT": str(tmp_path / "gpu-lease")}
    for script, message in [
        ("scripts/gpu-eval-run.sh", "gpu-eval-run: flock is required"),
        ("scripts/gpu-train-lock.sh", "gpu-train-lock: flock is required"),
    ]:
        completed = subprocess.run(
            ["bash", script, "bash", "-lc", "exit 0"],
            capture_output=True,
            text=True,
            env=env,
        )

        assert completed.returncode == 69
        assert message in completed.stderr

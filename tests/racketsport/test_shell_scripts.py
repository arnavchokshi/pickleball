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
    Path("scripts/download_checkpoints.sh"),
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


def test_fast_sam_wrapper_help_exits_before_environment_checks():
    completed = subprocess.run(
        ["bash", "scripts/racketsport/run_fast_sam_benchmark.sh", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "Usage: scripts/racketsport/run_fast_sam_benchmark.sh [OUT_DIR]" in completed.stdout
    assert completed.stderr == ""


def test_fast_sam_wrapper_rejects_extra_args_before_environment_checks():
    completed = subprocess.run(
        ["bash", "scripts/racketsport/run_fast_sam_benchmark.sh", "out", "extra"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 64
    assert "Usage: scripts/racketsport/run_fast_sam_benchmark.sh [OUT_DIR]" in completed.stderr
    assert "missing Fast-SAM" not in completed.stderr


def test_mujoco_mjx_installer_help_exits_before_environment_checks():
    completed = subprocess.run(
        ["bash", "scripts/racketsport/install_mujoco_mjx_env.sh", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "Usage: scripts/racketsport/install_mujoco_mjx_env.sh" in completed.stdout
    assert completed.stderr == ""


def test_mujoco_mjx_installer_rejects_args_before_environment_checks():
    completed = subprocess.run(
        ["bash", "scripts/racketsport/install_mujoco_mjx_env.sh", "extra"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 64
    assert "Usage: scripts/racketsport/install_mujoco_mjx_env.sh" in completed.stderr


def test_mujoco_mjx_installer_uses_env_path_when_overridden(tmp_path: Path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    conda_log = tmp_path / "conda.log"
    named_env_root = tmp_path / "named-envs"
    conda = fake_bin / "conda"
    conda.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
printf '%s\\n' "$*" >> "$CONDA_LOG"
env_path=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    -p)
      shift
      env_path="$1"
      ;;
    -n)
      shift
      env_path="$CONDA_NAMED_ENV_ROOT/$1"
      ;;
  esac
  shift || true
done
mkdir -p "$env_path/bin"
cat > "$env_path/bin/python" <<'PY'
#!/usr/bin/env bash
exit 0
PY
chmod +x "$env_path/bin/python"
""",
        encoding="utf-8",
    )
    conda.chmod(0o755)
    env_path = tmp_path / "custom-mjx-env"
    env = {
        **os.environ,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "CONDA_LOG": str(conda_log),
        "CONDA_NAMED_ENV_ROOT": str(named_env_root),
        "MUJOCO_MJX_ENV_PATH": str(env_path),
    }

    completed = subprocess.run(
        ["bash", "scripts/racketsport/install_mujoco_mjx_env.sh"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert completed.returncode == 0
    conda_args = conda_log.read_text(encoding="utf-8")
    assert f"-p {env_path}" in conda_args
    assert "-n racketsport_mjx" not in conda_args


def test_gpu_helpers_help_exits_before_lease_side_effects(tmp_path: Path):
    for script in ("scripts/gpu-eval-run.sh", "scripts/gpu-train-lock.sh"):
        lease_root = tmp_path / script.replace("/", "_")
        completed = subprocess.run(
            ["bash", script, "--help"],
            check=False,
            capture_output=True,
            text=True,
            env={**os.environ, "GPU_LEASE_ROOT": str(lease_root)},
        )

        assert completed.returncode == 0
        assert "Usage:" in completed.stdout
        assert completed.stderr == ""
        assert not lease_root.exists()


def test_setup_env_help_and_arg_validation_exit_before_mutation():
    for args, expected_returncode in [(["--help"], 0), (["extra"], 64)]:
        completed = subprocess.run(
            ["bash", "scripts/racketsport/setup_env.sh", *args],
            check=False,
            capture_output=True,
            text=True,
        )

        assert completed.returncode == expected_returncode
        assert "Usage: scripts/racketsport/setup_env.sh" in completed.stdout + completed.stderr
        assert "local Phase 0 environment ready" not in completed.stdout


def test_download_checkpoints_help_exits_before_network_access():
    completed = subprocess.run(
        ["bash", "scripts/download_checkpoints.sh", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "Usage: scripts/download_checkpoints.sh" in completed.stdout
    assert "--verify-only" in completed.stdout
    assert "--dest-root" in completed.stdout
    assert completed.stderr == ""


def test_download_checkpoints_verify_only_uses_sha256(tmp_path: Path):
    dest_root = tmp_path / "checkpoints"
    tracknet_dir = dest_root / "tracknetv3"
    tracknet_dir.mkdir(parents=True)
    (tracknet_dir / "TrackNet_best.pt").write_text("tracknet", encoding="utf-8")
    (tracknet_dir / "InpaintNet_best.pt").write_text("inpaintnet", encoding="utf-8")
    (dest_root / "yolo26n.pt").write_text("yolo26n", encoding="utf-8")
    (dest_root / "yolo26m.pt").write_text("yolo26m", encoding="utf-8")

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_sha256sum = fake_bin / "sha256sum"
    fake_sha256sum.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
case "$1" in
  */TrackNet_best.pt)
    printf '%s  %s\\n' df867641a02712b021f04548ff4b1208ddfdb47f629ab2094ceb978667e83b1a "$1"
    ;;
  */InpaintNet_best.pt)
    printf '%s  %s\\n' 5749b66b8002f3ad9e0af841604004706fc796df30599e6bf01952696009688c "$1"
    ;;
  */yolo26n.pt)
    printf '%s  %s\\n' 9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef "$1"
    ;;
  */yolo26m.pt)
    printf '%s  %s\\n' 401cea9ab23ad19246ff7744859816bc599f350e93c9dd30367b6f0a0745d0b7 "$1"
    ;;
  *)
    exit 2
    ;;
esac
""",
        encoding="utf-8",
    )
    fake_sha256sum.chmod(0o755)

    completed = subprocess.run(
        ["bash", "scripts/download_checkpoints.sh", "--verify-only", "--dest-root", str(dest_root)],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}"},
    )

    assert completed.returncode == 0
    assert "TrackNetV3 checkpoints verified" in completed.stdout
    assert "YOLO detector checkpoints verified" in completed.stdout


def test_download_checkpoints_verify_only_fails_on_hash_mismatch(tmp_path: Path):
    dest_root = tmp_path / "checkpoints"
    tracknet_dir = dest_root / "tracknetv3"
    tracknet_dir.mkdir(parents=True)
    (tracknet_dir / "TrackNet_best.pt").write_text("bad", encoding="utf-8")
    (tracknet_dir / "InpaintNet_best.pt").write_text("bad", encoding="utf-8")
    (dest_root / "yolo26n.pt").write_text("bad", encoding="utf-8")
    (dest_root / "yolo26m.pt").write_text("bad", encoding="utf-8")

    completed = subprocess.run(
        ["bash", "scripts/download_checkpoints.sh", "--verify-only", "--dest-root", str(dest_root)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "sha256 mismatch" in completed.stderr


def test_dockerfile_checks_checkpoint_script_without_downloading_weights():
    dockerfile = Path("Dockerfile").read_text(encoding="utf-8")

    assert "bash -n scripts/download_checkpoints.sh" in dockerfile
    assert "bash scripts/download_checkpoints.sh --help" in dockerfile


def test_fast_sam_installer_help_and_arg_validation_exit_before_environment_checks(tmp_path: Path):
    for args, expected_returncode in [(["--help"], 0), (["extra"], 64)]:
        cache_root = tmp_path / ("cache_" + args[0].lstrip("-"))
        completed = subprocess.run(
            ["bash", "scripts/racketsport/install_fast_sam_env.sh", *args],
            check=False,
            capture_output=True,
            text=True,
            env={**os.environ, "WORKSPACE_CACHE": str(cache_root)},
        )

        assert completed.returncode == expected_returncode
        assert "Usage: scripts/racketsport/install_fast_sam_env.sh" in completed.stdout + completed.stderr
        assert "conda.sh" not in completed.stderr
        assert not cache_root.exists()


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


def test_gpu_eval_run_uses_next_available_precreated_slot(tmp_path):
    _require_flock()

    lease_root = tmp_path / "gpu-lease"
    slots = lease_root / "slots"
    slots.mkdir(parents=True)
    for index in range(2):
        (slots / f"slot{index}.lock").write_text("", encoding="utf-8")
        (slots / f"slot{index}.uuid").write_text(str(index), encoding="utf-8")
    marker = tmp_path / "slot0-held"
    env = {**os.environ, "GPU_LEASE_ROOT": str(lease_root), "CUDA_VISIBLE_DEVICES": "9"}
    holder = subprocess.Popen(
        [
            "flock",
            str(slots / "slot0.lock"),
            "bash",
            "-lc",
            f"printf held > {marker}; sleep 0.4",
        ],
        env=env,
    )

    try:
        deadline = time.monotonic() + 3.0
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert marker.exists()

        completed = subprocess.run(
            ["bash", "scripts/gpu-eval-run.sh", "bash", "-lc", "printf '%s' \"$CUDA_VISIBLE_DEVICES\""],
            check=True,
            capture_output=True,
            text=True,
            env=env,
            timeout=5,
        )

        assert completed.stdout == "1"
    finally:
        holder.wait(timeout=5)


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

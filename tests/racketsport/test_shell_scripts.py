from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
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
    Path("scripts/racketsport/gpu_cold_start.sh"),
    Path("scripts/racketsport/run_fast_sam_benchmark.sh"),
    Path("scripts/fleet/lane_vm_startup.sh"),
]


def _require_flock() -> None:
    if shutil.which("flock") is None:
        pytest.skip("flock is not installed")


def test_shell_scripts_are_executable_and_parse():
    for script in SHELL_SCRIPTS:
        assert script.exists(), script
        assert os.access(script, os.X_OK), script
        subprocess.run(["bash", "-n", str(script)], check=True)


def _run_lane_vm_startup(
    tmp_path: Path,
    *,
    env_compute_mode: str | None = None,
    metadata_compute_mode: str | None = None,
    env_role: str | None = None,
    metadata_role: str | None = None,
    nvidia_smi_exit: int = 0,
    configure_training_gate: bool = False,
    omit_gate_proof_path: bool = False,
) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    nvidia_log = tmp_path / "nvidia-smi.log"

    fake_nvidia_smi = fake_bin / "nvidia-smi"
    fake_nvidia_smi.write_text(
        "#!/usr/bin/env bash\nprintf '%s\\n' \"$*\" >> \"$FAKE_NVIDIA_LOG\"\nexit \"${FAKE_NVIDIA_SMI_EXIT:-0}\"\n",
        encoding="utf-8",
    )
    fake_nvidia_smi.chmod(0o755)

    fake_curl = fake_bin / "curl"
    fake_curl.write_text(
        """#!/usr/bin/env bash
case "$*" in
  *fable-cuda-compute-mode*)
    if [ -n "${FAKE_CUDA_METADATA:-}" ]; then
      printf '%s\\n' "$FAKE_CUDA_METADATA"
    else
      exit 22
    fi
    ;;
  *preempted*)
    printf '%s\\n' FALSE
    ;;
  *fable-role*)
    if [ -n "${FAKE_FABLE_ROLE_METADATA:-}" ]; then
      printf '%s\\n' "$FAKE_FABLE_ROLE_METADATA"
    else
      exit 22
    fi
    ;;
  *)
    exit 22
    ;;
esac
""",
        encoding="utf-8",
    )
    fake_curl.chmod(0o755)

    fake_sleep = fake_bin / "sleep"
    fake_sleep.write_text("#!/usr/bin/env bash\nexit 1\n", encoding="utf-8")
    fake_sleep.chmod(0o755)

    env = {
        **os.environ,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "FAKE_NVIDIA_LOG": str(nvidia_log),
        "FAKE_NVIDIA_SMI_EXIT": str(nvidia_smi_exit),
    }
    for key in (
        "FABLE_REPO_DIR",
        "FABLE_TRAINING_PYTHON",
        "FABLE_TRAINING_INPUT_MANIFEST",
        "FABLE_DATA_LEDGER",
        "FABLE_CACHE_MANIFEST",
        "FABLE_GATE_PROOF",
    ):
        env.pop(key, None)
    if env_compute_mode is not None:
        env["FABLE_CUDA_COMPUTE_MODE"] = env_compute_mode
    else:
        env.pop("FABLE_CUDA_COMPUTE_MODE", None)
    if metadata_compute_mode is not None:
        env["FAKE_CUDA_METADATA"] = metadata_compute_mode
    else:
        env.pop("FAKE_CUDA_METADATA", None)
    if env_role is not None:
        env["FABLE_ROLE"] = env_role
    else:
        env.pop("FABLE_ROLE", None)
    if metadata_role is not None:
        env["FAKE_FABLE_ROLE_METADATA"] = metadata_role
    else:
        env.pop("FAKE_FABLE_ROLE_METADATA", None)
    if configure_training_gate:
        root = Path.cwd().resolve()
        input_manifest = tmp_path / "training_inputs.json"
        input_manifest.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "artifact_type": "training_input_manifest",
                    "inputs": [
                        {
                            "path": str(
                                root
                                / "data/event_labels_owner_20260719/PROVENANCE.json"
                            ),
                            "asset_id": "owner_event_labels_102_20260719",
                        }
                    ],
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        env.update(
            {
                "FABLE_REPO_DIR": str(root),
                "FABLE_TRAINING_PYTHON": sys.executable,
                "FABLE_TRAINING_INPUT_MANIFEST": str(input_manifest),
                "FABLE_DATA_LEDGER": str(root / "runs/manager/data_ledger.json"),
                "FABLE_GATE_PROOF": str(tmp_path / "gate_proof.json"),
            }
        )
        if omit_gate_proof_path:
            env.pop("FABLE_GATE_PROOF")

    completed = subprocess.run(
        ["bash", "scripts/fleet/lane_vm_startup.sh"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        timeout=10,
    )
    calls = nvidia_log.read_text(encoding="utf-8").splitlines() if nvidia_log.exists() else []
    return completed, calls


def test_lane_vm_startup_defaults_pipeline_vms_to_default_compute_mode(tmp_path: Path) -> None:
    completed, calls = _run_lane_vm_startup(tmp_path)

    assert completed.returncode == 0
    assert calls == ["-c DEFAULT"]
    assert "CUDA compute mode DEFAULT" in completed.stdout


@pytest.mark.parametrize(
    ("env_compute_mode", "metadata_compute_mode", "env_role", "metadata_role"),
    [
        ("EXCLUSIVE_PROCESS", None, "training", None),
        (None, "EXCLUSIVE_PROCESS", None, "training"),
    ],
)
def test_lane_vm_startup_allows_explicit_single_context_training_mode(
    tmp_path: Path,
    env_compute_mode: str | None,
    metadata_compute_mode: str | None,
    env_role: str | None,
    metadata_role: str | None,
) -> None:
    completed, calls = _run_lane_vm_startup(
        tmp_path,
        env_compute_mode=env_compute_mode,
        metadata_compute_mode=metadata_compute_mode,
        env_role=env_role,
        metadata_role=metadata_role,
        configure_training_gate=True,
    )

    assert completed.returncode == 0
    assert calls == ["-c EXCLUSIVE_PROCESS"]
    assert "CUDA compute mode EXCLUSIVE_PROCESS" in completed.stdout
    proof = json.loads((tmp_path / "gate_proof.json").read_text(encoding="utf-8"))
    assert proof["status"] == "PASS"


def test_lane_vm_startup_training_role_refuses_without_gate_configuration(
    tmp_path: Path,
) -> None:
    completed, calls = _run_lane_vm_startup(
        tmp_path,
        env_role="training",
    )

    assert completed.returncode != 0
    assert calls == ["-c DEFAULT"]
    assert "TRAINING_INPUT_MANIFEST_REQUIRED" in completed.stderr


def test_lane_vm_startup_training_role_refuses_without_gate_proof_path(
    tmp_path: Path,
) -> None:
    completed, calls = _run_lane_vm_startup(
        tmp_path,
        env_role="training",
        configure_training_gate=True,
        omit_gate_proof_path=True,
    )

    assert completed.returncode != 0
    assert calls == ["-c DEFAULT"]
    assert "GATE_PROOF_PATH_REQUIRED" in completed.stderr


@pytest.mark.parametrize("role", [None, "pipeline", "pickleball-worker"])
def test_lane_vm_startup_rejects_exclusive_process_without_explicit_training_role(
    tmp_path: Path,
    role: str | None,
) -> None:
    completed, calls = _run_lane_vm_startup(
        tmp_path,
        env_compute_mode="EXCLUSIVE_PROCESS",
        env_role=role,
    )

    assert completed.returncode == 64
    assert calls == []
    assert "EXCLUSIVE_PROCESS requires explicit fable-role=training" in completed.stderr


def test_lane_vm_startup_fails_closed_when_compute_mode_set_fails(tmp_path: Path) -> None:
    completed, calls = _run_lane_vm_startup(tmp_path, nvidia_smi_exit=42)

    assert completed.returncode != 0
    assert calls == ["-c DEFAULT"]
    assert "failed to set CUDA compute mode DEFAULT" in completed.stderr


def test_lane_vm_startup_arms_watcher_before_bounded_metadata_lookup() -> None:
    script = Path("scripts/fleet/lane_vm_startup.sh").read_text(encoding="utf-8")

    assert script.index("# 1. Arm the preemption watcher") < script.index("CUDA_COMPUTE_MODE_METADATA")
    assert "--connect-timeout 1 --max-time 2" in script


def test_lane_vm_startup_rejects_unknown_compute_mode_before_nvidia_call(tmp_path: Path) -> None:
    completed, calls = _run_lane_vm_startup(tmp_path, env_compute_mode="SHARED")

    assert completed.returncode == 64
    assert calls == []
    assert "unsupported CUDA compute mode: SHARED" in completed.stderr


def test_fast_sam_wrapper_records_machine_readable_profile_metrics():
    script = Path("scripts/racketsport/run_fast_sam_benchmark.sh").read_text(encoding="utf-8")

    assert "profile_stdout.log" in script
    assert "benchmark_sam3dbody.py" in script
    assert "--profile-log" in script
    assert "sam3dbody_benchmark.json" in script


def test_gpu_cold_start_checks_venv_ensurepip_not_only_venv_importability():
    script = Path("scripts/racketsport/gpu_cold_start.sh").read_text(encoding="utf-8")

    assert "import ensurepip" in script
    assert 'python3.10 -c "import venv"' not in script


def test_gpu_cold_start_body_venv_step_fails_on_install_command_failures():
    script = Path("scripts/racketsport/gpu_cold_start.sh").read_text(encoding="utf-8")
    step = script[script.index("step_build_body_venv() {") : script.index("# --- step 5: fetch")]

    assert "python3.10 -m venv \"$BODY_VENV_DIR\" || return 1" in step
    assert '"$BODY_VENV_DIR/bin/python" -m pip install --upgrade pip || return 1' in step
    assert "pip cache purge || true" not in step


def test_gpu_cold_start_pytest_smoke_is_count_agnostic_and_checks_exit_status():
    script = Path("scripts/racketsport/gpu_cold_start.sh").read_text(encoding="utf-8")
    step = script[script.index("step_pytest_smoke() {") : script.index("# --- step 7: minimal")]

    assert "pytest_status" in step
    assert "13 passed" not in step
    assert "0 failed" in step


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
    sat_hmr_path = dest_root / "body4d" / "sat-hmr" / "weights" / "sat_hmr" / "sat_644_3dpw.pth"
    sat_hmr_path.parent.mkdir(parents=True)
    sat_hmr_path.write_text("sat_hmr", encoding="utf-8")

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
  */sat_644_3dpw.pth)
    printf '%s  %s\\n' 7e1b5e80a967c8f4e1e273156e5272b2a3413caf079c2bc0a038e90c6a0b6dec "$1"
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
    assert "SAT-HMR checkpoint verified" in completed.stdout


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


# --- Finding 6: GPU lock holder metadata + optional wait timeout ---------


def test_gpu_train_lock_writes_and_removes_holder_metadata(tmp_path):
    _require_flock()

    lease_root = tmp_path / "gpu-lease"
    meta_path = lease_root / "full-gpu.lock.meta"
    env = {**os.environ, "GPU_LEASE_ROOT": str(lease_root)}

    completed = subprocess.run(
        ["bash", "scripts/gpu-train-lock.sh", "bash", "-lc", f'cat "{meta_path}"'],
        check=True,
        capture_output=True,
        text=True,
        env=env,
        timeout=10,
    )

    for field in ("pid=", "ppid=", "user=", "host=", "cwd=", "started_at_utc=", "command="):
        assert field in completed.stdout, completed.stdout
    # The metadata file is a diagnostic only, written/removed around the
    # kernel flock (which remains the sole correctness mechanism) -- it must
    # not survive past the command that held the lock.
    assert not meta_path.exists()


def test_gpu_train_lock_timeout_reports_holder_metadata_and_exits_75(tmp_path):
    _require_flock()

    lease_root = tmp_path / "gpu-lease"
    marker = tmp_path / "holder-started"
    env = {**os.environ, "GPU_LEASE_ROOT": str(lease_root)}
    holder = subprocess.Popen(
        ["bash", "scripts/gpu-train-lock.sh", "bash", "-lc", f"printf started > {marker}; sleep 1.5"],
        env=env,
    )
    try:
        deadline = time.monotonic() + 5.0
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert marker.exists()

        completed = subprocess.run(
            ["bash", "scripts/gpu-train-lock.sh", "bash", "-lc", "echo should-not-run"],
            capture_output=True,
            text=True,
            env={**env, "GPU_LOCK_TIMEOUT_S": "1"},
            timeout=10,
        )

        assert completed.returncode == 75
        assert "timed out after 1s" in completed.stderr
        assert "current holder metadata" in completed.stderr
        assert "should-not-run" not in completed.stdout
    finally:
        holder.wait(timeout=10)


def test_gpu_train_lock_still_blocks_forever_by_default_without_timeout_env(tmp_path):
    # Backward compatibility: existing callers that never set
    # GPU_LOCK_TIMEOUT_S must keep waiting indefinitely for the exclusive
    # lock, exactly as before this env var existed.
    _require_flock()

    lease_root = tmp_path / "gpu-lease"
    marker = tmp_path / "holder-started"
    env = {**os.environ, "GPU_LEASE_ROOT": str(lease_root)}
    assert "GPU_LOCK_TIMEOUT_S" not in env
    holder = subprocess.Popen(
        ["bash", "scripts/gpu-train-lock.sh", "bash", "-lc", f"printf started > {marker}; sleep 0.6"],
        env=env,
    )
    try:
        deadline = time.monotonic() + 5.0
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert marker.exists()

        start = time.monotonic()
        completed = subprocess.run(
            ["bash", "scripts/gpu-train-lock.sh", "bash", "-lc", "echo waited-then-ran"],
            check=True,
            capture_output=True,
            text=True,
            env=env,
            timeout=10,
        )
        elapsed = time.monotonic() - start

        assert completed.stdout.strip() == "waited-then-ran"
        assert elapsed >= 0.2
    finally:
        holder.wait(timeout=10)


def test_gpu_eval_run_heartbeat_includes_enriched_metadata_fields(tmp_path):
    _require_flock()

    lease_root = tmp_path / "gpu-lease"
    env = {**os.environ, "GPU_LEASE_ROOT": str(lease_root), "CUDA_VISIBLE_DEVICES": "5"}

    completed = subprocess.run(
        ["bash", "scripts/gpu-eval-run.sh", "bash", "-lc", 'cat "$GPU_LEASE_ROOT"/heartbeat/*'],
        check=True,
        capture_output=True,
        text=True,
        env=env,
        timeout=10,
    )

    # First line's shape stays byte-for-byte compatible with any existing
    # reader; new fields are additive lines after it.
    lines = completed.stdout.splitlines()
    assert lines[0].startswith("pid=") and " slot=" in lines[0] and " ts=" in lines[0]
    assert any(line.startswith("ppid=") for line in lines)
    assert any(line.startswith("started_at_utc=") for line in lines)


def test_gpu_eval_run_timeout_waiting_on_exclusive_train_lock_reports_holder(tmp_path):
    _require_flock()

    lease_root = tmp_path / "gpu-lease"
    marker = tmp_path / "train-started"
    env = {**os.environ, "GPU_LEASE_ROOT": str(lease_root)}
    train = subprocess.Popen(
        ["bash", "scripts/gpu-train-lock.sh", "bash", "-lc", f"printf started > {marker}; sleep 1.5"],
        env=env,
    )
    try:
        deadline = time.monotonic() + 5.0
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert marker.exists()

        completed = subprocess.run(
            ["bash", "scripts/gpu-eval-run.sh", "bash", "-lc", "echo should-not-run"],
            capture_output=True,
            text=True,
            env={**env, "GPU_LOCK_TIMEOUT_S": "1"},
            timeout=10,
        )

        assert completed.returncode == 75
        assert "timed out after 1s" in completed.stderr
        assert "current holder metadata" in completed.stderr
        assert "should-not-run" not in completed.stdout
    finally:
        train.wait(timeout=10)


def test_gpu_lock_scripts_warn_on_stderr_when_falling_back_to_tmpdir_lease_root(tmp_path):
    # A file (not a directory) at the primary lease root path makes
    # `mkdir -p "$LEASE_ROOT"` fail, forcing the ${TMPDIR:-/tmp}/gpu-lease
    # fallback this test wants to exercise -- and isolates that fallback to
    # a per-test TMPDIR so it cannot collide with any other concurrent test
    # or agent using the real /tmp/gpu-lease.
    primary_lease_root = tmp_path / "blocked-lease-root"
    primary_lease_root.write_bytes(b"not a directory")
    fallback_tmpdir = tmp_path / "fallback-tmpdir"
    fallback_tmpdir.mkdir()
    env = {**os.environ, "GPU_LEASE_ROOT": str(primary_lease_root), "TMPDIR": f"{fallback_tmpdir}/"}

    for script, prog_name in [
        ("scripts/gpu-eval-run.sh", "gpu-eval-run"),
        ("scripts/gpu-train-lock.sh", "gpu-train-lock"),
    ]:
        completed = subprocess.run(
            ["bash", script, "bash", "-lc", "exit 0"],
            capture_output=True,
            text=True,
            env=env,
            timeout=10,
        )
        assert f"{prog_name}: WARNING" in completed.stderr, completed.stderr
        assert str(primary_lease_root) in completed.stderr
        assert "TMPDIR" in completed.stderr
        # Either it ran to completion under the fallback root (flock
        # available) or it failed closed because flock is missing --
        # either way the warning above must have been printed first.
        assert completed.returncode in (0, 69)

from __future__ import annotations

import os
import subprocess
from pathlib import Path


SHELL_SCRIPTS = [
    Path("scripts/gpu-eval-run.sh"),
    Path("scripts/gpu-train-lock.sh"),
    Path("scripts/racketsport/setup_env.sh"),
    Path("scripts/racketsport/install_fast_sam_env.sh"),
    Path("scripts/racketsport/install_mujoco_mjx_env.sh"),
    Path("scripts/racketsport/run_fast_sam_benchmark.sh"),
]


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

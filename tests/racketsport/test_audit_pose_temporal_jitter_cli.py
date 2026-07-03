from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_audit_pose_temporal_jitter_cli_exposes_direct_help_reference() -> None:
    command_path = "scripts/racketsport/audit_pose_temporal_jitter.py"

    completed = subprocess.run(
        [sys.executable, command_path, "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--skeleton3d" in completed.stdout
    assert "--out-dir" in completed.stdout
    assert "--low-confidence-threshold" in completed.stdout


def test_audit_pose_temporal_jitter_cli_rejects_missing_skeleton(tmp_path: Path) -> None:
    command_path = "scripts/racketsport/audit_pose_temporal_jitter.py"

    completed = subprocess.run(
        [
            sys.executable,
            command_path,
            "--skeleton3d",
            str(tmp_path / "missing_skeleton3d.json"),
            "--out-dir",
            str(tmp_path / "out"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    assert "No such file or directory" in completed.stderr + completed.stdout

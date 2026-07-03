from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_ingest_owner_capture_cli_exposes_direct_help_reference() -> None:
    command_path = "scripts/racketsport/ingest_owner_capture.py"

    completed = subprocess.run(
        [sys.executable, command_path, "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--manifest" in completed.stdout
    assert "--reviewed-cvat-export" in completed.stdout
    assert "--corpus-manifest" in completed.stdout


def test_ingest_owner_capture_cli_rejects_missing_input(tmp_path: Path) -> None:
    command_path = "scripts/racketsport/ingest_owner_capture.py"

    completed = subprocess.run(
        [
            sys.executable,
            command_path,
            str(tmp_path / "missing_video.mp4"),
            "--manifest",
            str(tmp_path / "manifest.json"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    assert "owner capture ingest failed" in completed.stderr + completed.stdout


def test_prelabel_owner_capture_cli_exposes_direct_help_reference() -> None:
    command_path = "scripts/racketsport/prelabel_owner_capture.py"

    completed = subprocess.run(
        [sys.executable, command_path, "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--capture-id" in completed.stdout
    assert "--dry-run" in completed.stdout
    assert "--owner-data-root" in completed.stdout


def test_prelabel_owner_capture_cli_rejects_unknown_capture_id(tmp_path: Path) -> None:
    command_path = "scripts/racketsport/prelabel_owner_capture.py"

    completed = subprocess.run(
        [
            sys.executable,
            command_path,
            "--capture-id",
            "nonexistent",
            "--manifest",
            str(tmp_path / "manifest.json"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    assert "owner capture prelabel failed" in completed.stderr + completed.stdout

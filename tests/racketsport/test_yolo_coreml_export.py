from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_export_yolo_coreml_dry_run_prints_mobile_command(tmp_path: Path) -> None:
    out_dir = tmp_path / "models_coreml"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/export_yolo_coreml.py",
            "--weights",
            "yolo26n.pt",
            "--out-dir",
            str(out_dir),
            "--imgsz",
            "640",
            "--quantize",
            "8",
            "--dry-run",
        ],
        check=False,
        cwd=Path(__file__).resolve().parents[2],
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    assert "format=coreml" in result.stdout
    assert "imgsz=640" in result.stdout
    assert "batch=1" in result.stdout
    assert "quantize=8" in result.stdout
    assert "int8=True" not in result.stdout
    assert "--out-dir" not in result.stdout


def test_export_yolo_coreml_dry_run_maps_quantize_16_to_half_precision(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/export_yolo_coreml.py",
            "--weights",
            "yolo26n.pt",
            "--out-dir",
            str(tmp_path),
            "--quantize",
            "16",
            "--dry-run",
        ],
        check=False,
        cwd=Path(__file__).resolve().parents[2],
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    assert "quantize=16" in result.stdout
    assert "int8=True" not in result.stdout

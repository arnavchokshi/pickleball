from __future__ import annotations

import json
import subprocess
import sys


def test_manifest_report_summarizes_status_counts():
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/manifest_report.py",
            "--manifest",
            "models/MANIFEST.json",
            "--json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)

    assert payload["total_models"] >= 10
    assert payload["status_counts"]["available_on_h100"] >= 5
    assert payload["commercial_posture_counts"]["research_ok_verify_commercial"] >= 2
    assert "fast_sam_3d_body_dinov3" in payload["declared_h100_checkpoint_files"]
    assert "mujoco_mjx" in payload["declared_h100_inventory"]
    assert "ready_on_h100" not in payload


def test_manifest_report_writes_markdown(tmp_path):
    out = tmp_path / "manifest_report.md"

    subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/manifest_report.py",
            "--manifest",
            "models/MANIFEST.json",
            "--out",
            str(out),
        ],
        check=True,
    )

    body = out.read_text(encoding="utf-8")
    assert "# Model Manifest Report" in body
    assert "Declared H100 inventory" in body
    assert "Ready on H100" not in body
    assert "fast_sam_3d_body_dinov3" in body
    assert "available_runtime_on_h100" in body

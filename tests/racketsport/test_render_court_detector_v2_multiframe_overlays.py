from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_render_court_detector_v2_multiframe_overlays_writes_manifest_and_contact_sheet(tmp_path: Path) -> None:
    out_dir = tmp_path / "overlays"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/render_court_detector_v2_multiframe_overlays.py",
            "--eval-root",
            "eval_clips/ball",
            "--out-dir",
            str(out_dir),
            "--max-frames",
            "6",
            "--top-k",
            "4",
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    manifest = json.loads((out_dir / "overlay_manifest.json").read_text())
    assert manifest["artifact_type"] == "racketsport_court_detector_v2_multiframe_overlay_manifest"
    assert manifest["verified"] is False
    assert manifest["not_cal3_verified"] is True
    assert manifest["clips"]
    assert (out_dir / "contact_sheet.jpg").exists()
    for clip_summary in manifest["clips"]:
        overlay_path = Path(clip_summary["overlay_path"])
        assert overlay_path.exists()

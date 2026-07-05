from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_build_court_proposals_cli_writes_fail_closed_report(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    video = "eval_clips/ball/wolverine_mixed_0200_mid_steep_corner/source.mp4"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/build_court_proposals.py",
            "--video",
            video,
            "--clip",
            "wolverine_mixed_0200_mid_steep_corner",
            "--out-dir",
            str(out_dir),
            "--max-frames",
            "2",
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads((out_dir / "court_proposals.json").read_text())
    assert payload["artifact_type"] == "racketsport_court_proposals"
    assert payload["verified"] is False
    assert payload["not_cal3_verified"] is True
    assert payload["ranking"]["abstain"] is True
    assert payload["proposals"]
    assert all(proposal["verified"] is False for proposal in payload["proposals"])
    assert all(proposal["not_cal3_verified"] is True for proposal in payload["proposals"])

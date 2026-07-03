from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_run_court_detector_v2_img1605_writes_blocked_proposal(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/run_court_detector_v2.py",
            "--input",
            "runs/owner_data/owner_IMG_1605_8a193402780b/prelabels/review_frames/"
            "owner_IMG_1605_8a193402780b/frame_000151.jpg",
            "--clip-id",
            "owner_IMG_1605_8a193402780b",
            "--out-dir",
            str(tmp_path),
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    artifact = json.loads((tmp_path / "court_detector_v2_proposals.json").read_text())
    assert artifact["artifact_type"] == "racketsport_court_detector_v2_proposals"
    assert artifact["promoted"] is False
    assert artifact["verified"] is False
    assert artifact["not_cal3_verified"] is True
    assert "near_left_corner" in artifact["needs_user_input"]


def test_detector_v2_eval_includes_img1605_partial_clip(tmp_path: Path) -> None:
    out = tmp_path / "court_detector_v2_eval.json"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/evaluate_court_detector_v2.py",
            "--eval-root",
            "eval_clips/ball",
            "--include-partial",
            "owner_IMG_1605_8a193402780b",
            "--out",
            str(out),
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(out.read_text())
    assert payload["artifact_type"] == "racketsport_court_detector_v2_eval"
    assert payload["summary"]["partial_clip_count"] == 1
    assert payload["summary"]["promoted_clip_count"] == 0
    assert payload["summary"]["blocked_clip_count"] == 1
    assert payload["verified"] is False
    assert payload["not_cal3_verified"] is True

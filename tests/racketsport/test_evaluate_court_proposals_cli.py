from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_evaluate_court_proposals_cli_keeps_report_unverified(tmp_path: Path) -> None:
    proposal = {
        "artifact_type": "racketsport_court_proposals",
        "schema_version": 1,
        "clip": "unit",
        "status": "ranked_not_verified",
        "verified": False,
        "not_cal3_verified": True,
        "input": {"image_size": [100, 100], "frame_indices": [0], "motion_mode": "static"},
        "assist": {"mode": "none", "tap_points": [], "line_label": None},
        "ranking": {
            "selected_proposal_id": None,
            "selection_reason": "no_proposals",
            "abstain": True,
            "abstain_reasons": ["not_cal3_verified"],
        },
        "proposals": [],
    }
    proposal_path = tmp_path / "court_proposals.json"
    proposal_path.write_text(json.dumps(proposal), encoding="utf-8")
    out_path = tmp_path / "eval.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/evaluate_court_proposals.py",
            "--proposal",
            str(proposal_path),
            "--out",
            str(out_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(out_path.read_text())
    assert payload["verified"] is False
    assert payload["not_cal3_verified"] is True
    assert payload["status"] == "ran_not_verified"

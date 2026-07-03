from __future__ import annotations

import subprocess
import sys


def test_compare_court_proposals_cli_exposes_direct_help_reference() -> None:
    command_path = "scripts/racketsport/compare_court_proposals_to_reviewed_keypoints.py"

    completed = subprocess.run(
        [sys.executable, command_path, "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "usage" in completed.stdout.lower()

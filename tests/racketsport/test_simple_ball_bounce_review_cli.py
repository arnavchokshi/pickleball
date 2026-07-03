from __future__ import annotations

import subprocess
import sys


def test_run_simple_ball_bounce_review_cli_help_runs_from_repo_root() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/racketsport/simple_ball_bounce_review.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "usage:" in completed.stdout.lower()
    assert "--port" in completed.stdout


def test_run_simple_ball_bounce_review_cli_fails_closed_on_invalid_port() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/simple_ball_bounce_review.py",
            "--no-open",
            "--port",
            "not-an-integer",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "invalid int value" in completed.stderr.lower()

from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CLI = "scripts/racketsport/eval_event_head.py"


def test_eval_cli_is_directly_referenced() -> None:
    completed = subprocess.run(
        [str(ROOT / ".venv/bin/python"), CLI, "--help"], cwd=ROOT,
        capture_output=True, text=True, check=False,
    )
    assert completed.returncode == 0
    assert "protected-seed" in completed.stdout

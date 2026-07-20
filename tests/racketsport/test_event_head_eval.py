from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from scripts.racketsport.eval_event_head import _resolve_window_frames


ROOT = Path(__file__).resolve().parents[2]
CLI = "scripts/racketsport/eval_event_head.py"


def test_eval_cli_is_directly_referenced() -> None:
    completed = subprocess.run(
        [str(ROOT / ".venv/bin/python"), CLI, "--help"], cwd=ROOT,
        capture_output=True, text=True, check=False,
    )
    assert completed.returncode == 0
    assert "protected-seed" in completed.stdout
    assert "--window-frames" in completed.stdout


def test_eval_window_defaults_to_checkpoint_training_config_and_rejects_mismatch() -> None:
    payload = {"window_frames": 64, "config": {"window_frames": 64}}
    assert _resolve_window_frames(payload, None) == 64
    assert _resolve_window_frames(payload, 64) == 64
    with pytest.raises(ValueError, match="eval window mismatch"):
        _resolve_window_frames(payload, 15)


def test_eval_window_rejects_internally_inconsistent_checkpoint() -> None:
    with pytest.raises(ValueError, match="window provenance disagrees"):
        _resolve_window_frames({"window_frames": 15, "config": {"window_frames": 64}}, None)

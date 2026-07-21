from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import torch

from scripts.racketsport.finetune_event_head import FineTuneInputError, _assert_checkpoint_context
from threed.racketsport.event_head.model import EventHead, checkpoint_payload


ROOT = Path(__file__).resolve().parents[2]
CLI = "scripts/racketsport/finetune_event_head.py"


def test_legacy_stale_schema_cli_is_removed(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            str(ROOT / ".venv/bin/python"),
            CLI,
            "--reviewed",
            str(tmp_path / "reviewed.jsonl"),
            "--manifest",
            str(tmp_path / "schema_v2.json"),
            "--pretrain",
            str(tmp_path / "checkpoint.pt"),
            "--out",
            str(tmp_path / "out"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "legacy --reviewed/--manifest/--pretrain input was removed" in completed.stderr
    assert "use --owner-manifest and --init-checkpoint-model-only" in completed.stderr


def test_checkpoint_without_window_context_is_never_blessed(tmp_path: Path) -> None:
    model = EventHead(weights="none", feature_dim=8, hidden_dim=8)
    checkpoint = tmp_path / "missing_window_frames.pt"
    torch.save(checkpoint_payload(model, image_size=32), checkpoint)

    with pytest.raises(FineTuneInputError, match="no explicit window_frames"):
        _assert_checkpoint_context(checkpoint, window_frames=64, image_size=32)

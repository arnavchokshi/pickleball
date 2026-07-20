from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import torch

import scripts.racketsport.eval_event_head as eval_event_head
from scripts.racketsport.eval_event_head import _resolve_window_frames, eval_public


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


def test_public_eval_persists_probability_maxima_and_nonfinite_count(monkeypatch) -> None:
    class DiagnosticModel(torch.nn.Module):
        def forward(self, frames: torch.Tensor) -> torch.Tensor:
            logits = torch.zeros(frames.shape[0], frames.shape[1], 3)
            logits[:, :, 0] = 4.0
            logits[:, 1, 1] = 8.0
            logits[:, 2, 2] = 9.0
            logits[:, 0, 0] = float("nan")
            return logits

    monkeypatch.setattr(
        eval_event_head,
        "decode_video_frames",
        lambda _path, indices, image_size: torch.zeros(len(indices), 3, image_size, image_size),
    )
    manifest = {
        "rows": [{
            "split": "val", "media_present": True, "video_path": "/nonexistent.mp4",
            "source_start_frame": 0, "num_frames": 64, "fps": 30.0,
            "events": [{"class": "HIT", "frame": 32}],
            "loss_validity_mask": [True, True, True],
            "source": "synthetic_fixture", "license_posture": "RD_ONLY",
        }],
    }

    result = eval_public(
        DiagnosticModel(), image_size=8, threshold=0.5,
        window_frames=64, max_clips=1, manifest=manifest,
    )
    summary = result["clip_summaries"][0]
    assert summary["nonfinite_probability_count"] == 3
    assert summary["max_probability_by_class"]["background"] == pytest.approx(
        float(torch.softmax(torch.tensor([4.0, 0.0, 0.0]), dim=0)[0])
    )
    assert summary["max_probability_by_class"]["HIT"] == pytest.approx(
        float(torch.softmax(torch.tensor([4.0, 8.0, 0.0]), dim=0)[1])
    )
    assert summary["max_probability_by_class"]["BOUNCE"] == pytest.approx(
        float(torch.softmax(torch.tensor([4.0, 0.0, 9.0]), dim=0)[2])
    )
    assert summary["max_positive_class_probability"] == (
        summary["max_probability_by_class"]["BOUNCE"]
    )

from __future__ import annotations

import inspect
import subprocess
from pathlib import Path

import pytest
import torch

import scripts.racketsport.eval_event_head as eval_event_head
from scripts.racketsport.eval_event_head import (
    _resolve_window_frames,
    eval_owner_val,
    eval_public,
)
from threed.racketsport.event_head.matcher import peak_pick


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


def _row(index: int, *, num_frames: int, event_frames: list[int]) -> dict:
    return {
        "split": "val", "media_present": True, "video_path": f"/nonexistent_{index}.mp4",
        "source_start_frame": index * num_frames, "num_frames": num_frames, "fps": 30.0,
        "events": [{"class": "HIT", "frame": frame} for frame in event_frames],
        "loss_validity_mask": [True, True, True],
        "source": "synthetic_fixture", "license_posture": "RD_ONLY",
    }


def test_public_eval_builds_every_window_at_the_matched_checkpoint_context(monkeypatch) -> None:
    """E-v2 blocker: eval must run at the checkpoint's 64-frame context, never 15.

    The historic harness bug scored a 64-frame-context model on hardcoded
    15-frame windows. This proves the current harness (a) derives the window
    from checkpoint provenance, (b) decodes exactly that many frames per
    evaluated clip even when the source row carries more context, and
    (c) refuses the legacy 15-frame request outright.
    """

    decoded_window_lengths: list[int] = []

    def _fake_decode(_path, indices, image_size):
        decoded_window_lengths.append(len(indices))
        return torch.zeros(len(indices), 3, image_size, image_size)

    monkeypatch.setattr(eval_event_head, "decode_video_frames", _fake_decode)

    class SilentModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            # The judge resolves its device from model parameters; keep one.
            self.device_anchor = torch.nn.Parameter(torch.zeros(1))

        def forward(self, frames: torch.Tensor) -> torch.Tensor:
            logits = torch.zeros(frames.shape[0], frames.shape[1], 3)
            logits[:, :, 0] = 9.0
            return logits

    checkpoint_payload = {"config": {"window_frames": 64}}
    window_frames = _resolve_window_frames(checkpoint_payload, None)
    assert window_frames == 64

    manifest = {"rows": [
        _row(0, num_frames=128, event_frames=[64]),
        _row(1, num_frames=128, event_frames=[100, 20]),
    ]}
    result = eval_public(
        SilentModel(), image_size=8, threshold=0.5,
        window_frames=window_frames, max_clips=4, manifest=manifest,
    )
    assert result["window_frames"] == 64
    assert result["clip_count"] == 2
    assert decoded_window_lengths == [64, 64]

    with pytest.raises(ValueError, match="eval window mismatch"):
        _resolve_window_frames(checkpoint_payload, 15)


def test_owner_val_refuses_row_context_that_differs_from_checkpoint_window() -> None:
    """The frozen owner-41 gate must never score a mismatched-window row."""

    rows = []
    decisions = ["HIT"] * 19 + ["none"] * 22
    for index, decision in enumerate(decisions):
        row = _row(index, num_frames=15, event_frames=[7] if decision == "HIT" else [])
        row["label_id"] = f"owner_{index:02d}"
        row["source_video"] = f"source_{index // 21}"
        rows.append(row)
    manifest = {
        "schema_version": 1,
        "artifact_type": "event_head_owner_reviewed_dataset_manifest",
        "classes": {"0": "background", "1": "HIT", "2": "BOUNCE"},
        "config": {"window_frames": 64},
        "rows": rows,
    }
    with pytest.raises(ValueError, match="owner-val row context must match checkpoint window_frames"):
        eval_owner_val(
            torch.nn.Identity(), image_size=8, threshold=0.5, window_frames=64,
            manifest=manifest, arm="B", seed=0, completed_steps=1, target_steps=1,
        )


def test_frozen_e1_judge_peak_pick_nms_radius_is_2() -> None:
    """E1 frozen judge protocol: NMS radius 2 by default, with radius-2 behavior."""

    assert inspect.signature(peak_pick).parameters["nms_radius"].default == 2

    logits = torch.zeros(20, 3)
    logits[:, 0] = 4.0
    for frame, strength in ((5, 8.0), (7, 7.0), (10, 8.0), (13, 7.5)):
        logits[frame, 1] = strength
    events = peak_pick(logits, threshold=0.5)
    # |7-5| = 2 <= radius -> weaker peak suppressed; |13-10| = 3 > radius -> kept.
    assert [event.frame for event in events if event.class_id == 1] == [5, 10, 13]


def test_public_eval_persists_probability_maxima_and_nonfinite_count(monkeypatch) -> None:
    class DiagnosticModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            # The judge resolves its device from model parameters; the original
            # parameterless fixture raised StopIteration inside _predict before
            # any assertion ran (pre-existing failure, reproduced at origin/main).
            self.device_anchor = torch.nn.Parameter(torch.zeros(1))

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

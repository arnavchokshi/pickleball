from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from scripts.racketsport.build_event_head_anchor_candidates import _window_logits
from threed.racketsport.event_head.datasets import (
    EventWindowDataset,
    WindowSpec,
    deterministic_source_video_holdout,
    manifest_windows,
)


ROOT = Path(__file__).resolve().parents[2]
VIDEO = ROOT / "tests/racketsport/fixtures/event_head/tiny.avi"


class _InputSpy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.inputs: list[torch.Tensor] = []

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        self.inputs.append(frames.detach().cpu().clone())
        batch_size, time_steps = frames.shape[:2]
        return torch.zeros(
            (batch_size, time_steps, 3), dtype=frames.dtype, device=frames.device
        )


def test_dataloader_tensor_is_bit_identical_to_production_inference_preprocessor() -> None:
    spec = WindowSpec(
        video_path=VIDEO,
        start_frame=0,
        num_frames=3,
        fps=10.0,
        events=((1, 1),),
        validity_mask=(True, True, True),
        source="fixture",
        license_posture="TEST_ONLY",
        source_video="tiny",
    )
    batch = next(iter(DataLoader(EventWindowDataset([spec], image_size=32), batch_size=1)))

    spy = _InputSpy()
    inference_windows, _, _ = _window_logits(
        spy,
        VIDEO,
        image_size=32,
        window_frames=3,
        stride=3,
        device="cpu",
        max_seconds=None,
    )
    try:
        start_frame, _ = next(inference_windows)
    finally:
        inference_windows.close()

    assert start_frame == 0
    assert len(spy.inputs) == 1
    assert torch.equal(batch["frames"], spy.inputs[0])


def _construction_snapshot(seed: int) -> bytes:
    rows = []
    for index, source_video in enumerate(("alpha", "bravo", "charlie", "delta", "echo")):
        rows.append({
            "source": "fixture",
            "source_video": source_video,
            "video_path": str(VIDEO),
            "media_present": True,
            "split": "train",
            "fps": 10.0,
            "source_start_frame": index,
            "num_frames": 5,
            "events": [
                {"frame": 1, "class": "HIT"},
                {"frame": 4, "class": "BOUNCE"},
            ],
            "loss_validity_mask": [True, True, True],
            "license_posture": "TEST_ONLY",
        })
    manifest, held_out = deterministic_source_video_holdout(
        {"rows": rows}, seed=seed, holdout_source_count=2
    )
    windows = {
        split: manifest_windows(
            manifest,
            split=split,
            limit=len(rows),
            window_frames=3,
            stride_frames=2,
        )
        for split in ("train", "val")
    }
    payload = {
        "held_out": held_out,
        "row_splits": [
            (row["source_video"], row["split"])
            for row in manifest["rows"]
        ],
        "windows": {
            split: [
                {
                    "source_video": window.source_video,
                    "start_frame": window.start_frame,
                    "num_frames": window.num_frames,
                    "events": window.events,
                    "row_index": window.row_index,
                }
                for window in split_windows
            ]
            for split, split_windows in windows.items()
        },
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def test_same_seed_source_split_and_window_construction_are_byte_identical() -> None:
    first = _construction_snapshot(20260722)
    second = _construction_snapshot(20260722)

    assert first == second
    decoded = json.loads(first)
    assert len(decoded["held_out"]) == 2
    assert {item[1] for item in decoded["row_splits"]} == {"train", "val"}

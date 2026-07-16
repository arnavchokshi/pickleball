from __future__ import annotations

import json
import subprocess
from pathlib import Path

import torch

from threed.racketsport.event_head.datasets import (
    EXPECTED_UNIVERSE,
    EventWindowDataset,
    build_public_manifest,
    decode_video_frames,
    manifest_windows,
    parse_jhong_clip_name,
)


ROOT = Path(__file__).resolve().parents[2]
CLI = "scripts/racketsport/build_event_head_dataset.py"


def test_builder_is_byte_deterministic_and_reconciles_inventory(tmp_path: Path) -> None:
    outputs = [tmp_path / "one.json", tmp_path / "two.json"]
    for output in outputs:
        completed = subprocess.run(
            [str(ROOT / ".venv/bin/python"), CLI, "--out", str(output)],
            cwd=ROOT, capture_output=True, text=True, check=False,
        )
        assert completed.returncode == 0, completed.stderr
    assert outputs[0].read_bytes() == outputs[1].read_bytes()
    manifest = json.loads(outputs[0].read_text())
    assert {name: values["inventory_events"] for name, values in manifest["totals"].items()} == EXPECTED_UNIVERSE


def test_jhong_reference_offset_and_train_eval_preprocessor_parity() -> None:
    manifest = build_public_manifest(ROOT / "data/event_public_20260713")
    window = next(
        item for item in manifest_windows(manifest, split="train", limit=500, window_frames=3)
        if item.source == "jhong93_spot"
    )
    row = next(row for row in manifest["rows"] if row["video_path"] == str(window.video_path)
               and row["split"] == "train" and row["source"] == "jhong93_spot")
    _, parsed_start, parsed_end = parse_jhong_clip_name(row["video"])
    assert parsed_start == row["source_start_frame"]
    assert parsed_end - parsed_start == row["num_frames"]
    training_tensor = EventWindowDataset([window], image_size=48)[0]["frames"]
    inference_tensor = decode_video_frames(
        window.video_path, list(range(window.start_frame, window.start_frame + 3)), image_size=48
    )
    assert torch.equal(training_tensor, inference_tensor)


def test_split_assignment_is_source_video_disjoint() -> None:
    manifest = build_public_manifest(ROOT / "data/event_public_20260713")
    groups: dict[tuple[str, str], set[str]] = {}
    for row in manifest["rows"]:
        groups.setdefault((row["source"], row["source_video"]), set()).add(row["split"])
    assert all(len(splits) == 1 for splits in groups.values())

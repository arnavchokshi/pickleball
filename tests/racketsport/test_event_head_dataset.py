from __future__ import annotations

import json
import subprocess
from pathlib import Path

import torch

from threed.racketsport.event_head.datasets import (
    BOUNCE,
    EXPECTED_UNIVERSE,
    EventWindowDataset,
    build_public_manifest,
    decode_video_frames,
    load_shuttleset_rows,
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
    by_source: dict[str, dict[str, int]] = {}
    for (source, _), splits in groups.items():
        split = next(iter(splits))
        by_source.setdefault(source, {"train": 0, "val": 0, "test": 0})[split] += 1
    assert by_source == {
        "jhong93_spot": {"train": 20, "val": 4, "test": 4},
        "openttgames": {"train": 8, "val": 2, "test": 2},
        "shuttleset": {"train": 31, "val": 7, "test": 6},
    }


def test_manifest_windows_slide_over_full_row_and_union_labels() -> None:
    row = {
        "split": "train", "media_present": True,
        "video_path": "/tmp/source.mp4", "source_start_frame": 100,
        "num_frames": 100, "fps": 30.0,
        "events": [{"frame": 10, "class": "HIT"}, {"frame": 70, "class": "BOUNCE"}],
        "loss_validity_mask": [True, True, False], "source": "fixture",
        "license_posture": "RD_ONLY",
    }
    windows = manifest_windows(
        {"rows": [row]}, split="train", limit=1, window_frames=64, stride_frames=32,
    )
    assert [window.start_frame for window in windows] == [100, 132, 136]
    assert windows[0].events == ((10, 1),)
    assert windows[1].events == ((38, BOUNCE),)
    assert windows[2].events == ((34, BOUNCE),)
    assert all(window.validity_mask == (True, True, False) for window in windows)


def test_shuttleset_glob_ignores_appledouble_sidecars(tmp_path: Path) -> None:
    set_dir = tmp_path / "coachai_shuttleset/ShuttleSet/set/match"
    set_dir.mkdir(parents=True)
    (set_dir / "valid.csv").write_text("frame_num,type\n12,hit\n")
    (set_dir / "._valid.csv").write_bytes(b"\x00\xffnot-csv")
    rows = load_shuttleset_rows(tmp_path, seed=7)
    assert len(rows) == 1
    assert rows[0]["inventory_event_count"] == 1

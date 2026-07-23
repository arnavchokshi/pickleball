from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path

import pytest

from scripts.racketsport.build_owner_event_manifest import (
    DEFAULT_PROTECTED,
    _json_bytes,
    build_owner_manifest,
)
from scripts.racketsport.finetune_event_head import FineTuneInputError, validate_manifests
from threed.racketsport.event_head.datasets import (
    manifest_windows,
    validate_current_manifest,
)


ROOT = Path(__file__).resolve().parents[2]
CLI = "scripts/racketsport/build_owner_event_manifest.py"
OWNER_MANIFEST = ROOT / "runs/lanes/ball_event_abc_20260720/inputs/owner_102_manifest.json"


def test_owner_102_manifest_is_deterministic_current_schema_and_source_held() -> None:
    first = build_owner_manifest()
    second = build_owner_manifest()

    assert _json_bytes(first) == _json_bytes(second) == OWNER_MANIFEST.read_bytes()
    validate_current_manifest(first)
    assert first["totals"]["train_rows"] == 61
    assert first["totals"]["val_rows"] == 41
    assert first["totals"]["split_decision_counts"] == {
        "train": {"ground": 17, "none": 20, "other": 1, "paddle": 23},
        "val": {"ground": 4, "none": 22, "paddle": 15},
    }
    assert first["totals"]["typed_answers"] == 60
    assert first["totals"]["answers_with_coordinates"] == 60
    assert first["totals"]["answers_with_dt"] == 60
    assert first["totals"]["legacy_provenance_counts"]["coordinates"] == 46
    assert first["totals"]["legacy_provenance_counts"]["dt"] == 57
    assert first["protected_seed_check"]["overlap_rows"] == 0
    assert first["protected_seed_check"]["policy"] == (
        "no_protected_frame_in_half_open_training_window"
    )
    assert first["protected_seed_check"]["checked_training_windows"] == 61
    assert first["protected_seed_check"]["adjusted_training_windows"] == 2
    assert all(row["num_frames"] == 64 and row["media_present"] for row in first["rows"])
    source_splits: dict[str, set[str]] = {}
    for row in first["rows"]:
        source_splits.setdefault(row["source_video"], set()).add(row["split"])
    assert all(len(splits) == 1 for splits in source_splits.values())

    assert len(manifest_windows(
        first, split="train", limit=102, window_frames=64
    )) == 40
    assert len(manifest_windows(
        first, split="val", limit=102, window_frames=64
    )) == 19


def test_no_protected_seed_frame_is_inside_any_owner_training_window() -> None:
    manifest = build_owner_manifest()
    protected = json.loads(DEFAULT_PROTECTED.read_text(encoding="utf-8"))["labels"]
    protected_by_sha: dict[str, list[int]] = {}
    for label in protected:
        protected_by_sha.setdefault(label["source"]["video_sha256"], []).append(
            label["anchor"]["frame"]
        )

    for row in manifest["rows"]:
        if row["split"] != "train":
            continue
        start = row["source_start_frame"]
        end = start + row["num_frames"]
        assert all(
            not start <= frame < end
            for frame in protected_by_sha.get(row["video_sha256"], [])
        ), (row["label_id"], start, end)

    adjusted = {
        row["label_id"]: (row["source_start_frame"], row["source_start_frame"] + 64)
        for row in manifest["rows"]
        if row["label_id"] in {"els20260715_004", "els20260715_052"}
    }
    assert adjusted == {
        "els20260715_004": (847, 911),
        "els20260715_052": (1098, 1162),
    }


def test_owner_102_manifest_passes_finetune_and_protected_overlap_guards(
    tmp_path: Path,
) -> None:
    owner, _, pseudo, _ = validate_manifests(
        OWNER_MANIFEST, None, window_frames=64
    )
    assert len(owner["rows"]) == 102
    assert pseudo is None

    protected = json.loads(DEFAULT_PROTECTED.read_text(encoding="utf-8"))["labels"][0]
    overlapping = copy.deepcopy(owner)
    matching = next(
        row for row in overlapping["rows"]
        if row["clip_id"] == protected["source"]["clip_id"]
    )
    matching["source_start_frame"] = protected["anchor"]["frame"]
    bad_path = tmp_path / "overlap.json"
    bad_path.write_text(json.dumps(overlapping), encoding="utf-8")

    with pytest.raises(FineTuneInputError, match="PROTECTED_SEED_WINDOW_OVERLAP"):
        validate_manifests(bad_path, None, window_frames=64)


def test_owner_manifest_cli_emits_the_frozen_artifact(tmp_path: Path) -> None:
    output = tmp_path / "owner_102_manifest.json"
    completed = subprocess.run(
        [str(ROOT / ".venv/bin/python"), CLI, "--out", str(output)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    summary = json.loads(completed.stdout)
    assert summary["rows"] == 102
    assert summary["train_rows"] == 61
    assert summary["val_rows"] == 41
    assert summary["protected_overlap_rows"] == 0
    assert summary["verified"] is False
    assert output.read_bytes() == OWNER_MANIFEST.read_bytes()

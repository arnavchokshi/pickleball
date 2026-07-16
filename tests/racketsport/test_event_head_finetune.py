from __future__ import annotations

import json
import subprocess
from pathlib import Path

import torch

from threed.racketsport.event_head.model import EventHead, checkpoint_payload


ROOT = Path(__file__).resolve().parents[2]
CLI = "scripts/racketsport/finetune_event_head.py"
FIXTURES = ROOT / "tests/racketsport/fixtures/event_head"


def _checkpoint(path: Path) -> None:
    model = EventHead(weights="none", feature_dim=8, hidden_dim=8)
    torch.save(checkpoint_payload(model, license_posture="RD_ONLY", image_size=32), path)


def _run(reviewed: Path, manifest: Path, pretrain: Path, out: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(ROOT / ".venv/bin/python"), CLI, "--reviewed", str(reviewed),
         "--manifest", str(manifest), "--pretrain", str(pretrain), "--out", str(out),
         "--steps", "2", "--image-size", "32", "--window-frames", "3"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )


def test_fixture_finetune_cpu_exit_zero(tmp_path: Path) -> None:
    pretrain = tmp_path / "pretrain.pt"
    _checkpoint(pretrain)
    completed = _run(FIXTURES / "reviewed_labels_v2.jsonl", FIXTURES / "dataset_manifest.json",
                     pretrain, tmp_path / "out")
    assert completed.returncode == 0, completed.stderr
    assert (tmp_path / "out/event_head_finetuned.pt").is_file()


def test_finetune_provenance_hard_fail_exit_codes(tmp_path: Path) -> None:
    pretrain = tmp_path / "pretrain.pt"
    _checkpoint(pretrain)
    fixture_rows = [json.loads(line) for line in (FIXTURES / "reviewed_labels_v2.jsonl").read_text().splitlines()]
    manifest = FIXTURES / "dataset_manifest.json"

    bootstrap = tmp_path / "bootstrap.jsonl"
    bootstrap_row = dict(fixture_rows[0])
    bootstrap_row["provenance"] = dict(bootstrap_row["provenance"], generator_version="event_bootstrap_v0_bad")
    bootstrap.write_text(json.dumps(bootstrap_row) + "\n")
    assert _run(bootstrap, manifest, pretrain, tmp_path / "a").returncode == 21

    seed = json.loads((ROOT / "runs/lanes/event_bootstrap_20260713/spot_check_tier_a_50.json").read_text())["labels"][0]
    overlap = tmp_path / "overlap.jsonl"
    overlap_row = dict(fixture_rows[0], label_id="overlap", video_path=seed["source"]["video_path"],
                       anchor_pts_s=seed["anchor"]["pts_s"], corrected_contact_pts_s=seed["anchor"]["pts_s"])
    overlap.write_text(json.dumps(overlap_row) + "\n")
    assert _run(overlap, manifest, pretrain, tmp_path / "b").returncode == 22

    duplicate = tmp_path / "duplicate.jsonl"
    duplicate.write_text("\n".join(json.dumps(dict(fixture_rows[0], label_id="dup")) for _ in range(2)) + "\n")
    assert _run(duplicate, manifest, pretrain, tmp_path / "c").returncode == 23


def test_missing_reviewed_file_is_actionable(tmp_path: Path) -> None:
    pretrain = tmp_path / "pretrain.pt"
    _checkpoint(pretrain)
    completed = _run(tmp_path / "missing.jsonl", FIXTURES / "dataset_manifest.json", pretrain, tmp_path / "out")
    assert completed.returncode == 2
    assert "run ingest_event_review_results.py first" in completed.stderr

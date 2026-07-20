from __future__ import annotations

import json
import subprocess
from pathlib import Path

from scripts.racketsport.train_event_head import _git_head, run_full


ROOT = Path(__file__).resolve().parents[2]
CLI = "scripts/racketsport/train_event_head.py"


def test_train_cli_reference_and_cpu_smoke(tmp_path: Path) -> None:
    completed = subprocess.run(
        [str(ROOT / ".venv/bin/python"), CLI, "--smoke", "--out", str(tmp_path),
         "--steps", "30", "--image-size", "32", "--window-frames", "3"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert (tmp_path / "smoke_event_head.pt").is_file()
    assert (tmp_path / "train_manifest.json").is_file()
    manifest = json.loads((tmp_path / "train_manifest.json").read_text())
    assert manifest["smoke_verified"] is True
    assert manifest["optimizer_steps"] == 30
    assert manifest["image_size"] == 32
    assert manifest["window_frames"] == 3


def _tiny_public_manifest(path: Path) -> Path:
    video = ROOT / "tests/racketsport/fixtures/event_head/tiny.avi"
    row = {
        "source": "synthetic_fixture", "video": "tiny", "source_video": "tiny",
        "video_path": str(video), "media_present": True, "fps": 10.0,
        "source_start_frame": 0, "num_frames": 10,
        "event_counts": {"HIT": 1, "BOUNCE": 1, "background": 0},
        "inventory_event_count": 2,
        "events": [{"frame": 2, "class": "HIT"}, {"frame": 8, "class": "BOUNCE"}],
        "loss_validity_mask": [True, True, True], "license_posture": "RD_ONLY",
    }
    manifest = {
        "schema_version": 1, "artifact_type": "event_head_public_dataset_manifest",
        "verified": False, "rows": [{**row, "split": "train"}, {**row, "split": "val"}],
    }
    path.write_text(json.dumps(manifest) + "\n")
    return path


def test_full_train_writes_provenance_best_and_last_checkpoints(tmp_path: Path) -> None:
    manifest_path = _tiny_public_manifest(tmp_path / "public_manifest.json")
    out = tmp_path / "full"
    completed = subprocess.run(
        [
            str(ROOT / ".venv/bin/python"), CLI, "--full", "--manifest", str(manifest_path),
            "--device", "cpu", "--out", str(out), "--steps", "2", "--image-size", "32",
            "--window-frames", "3", "--batch-size", "1", "--lr", "0.001",
            "--val-every", "1", "--seed", "17", "--limit-clips", "1",
            "--num-workers", "0", "--prefetch-factor", "2",
        ],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads((out / "train_manifest.json").read_text())
    assert report["mode"] == "full"
    assert report["status"] == "complete"
    assert report["verified"] is False
    assert report["license_posture"] == "RD_ONLY"
    assert report["seed"] == 17
    assert len(report["data_manifest_sha256"]) == 64
    assert report["config"]["limit_clips"] == 1
    assert report["config"]["stride_frames"] == 32
    assert report["dataloader"] == {"num_workers": 0, "prefetch_factor": None}
    assert report["completed_steps"] == 2
    assert report["steps_per_s"] > 0
    assert report["validations"][0]["tolerance_frames"] == 2
    assert Path(report["best_checkpoint"]).is_file()
    assert Path(report["last_checkpoint"]).is_file()


def test_full_train_rejects_protected_or_owner_media(tmp_path: Path) -> None:
    manifest_path = _tiny_public_manifest(tmp_path / "public_manifest.json")
    manifest = json.loads(manifest_path.read_text())
    manifest["rows"][0]["source"] = "jhong93_spot"
    manifest["rows"][0]["video_path"] = "data/event_bootstrap_20260713/tier_a/owner.mp4"
    manifest_path.write_text(json.dumps(manifest) + "\n")
    completed = subprocess.run(
        [
            str(ROOT / ".venv/bin/python"), CLI, "--full", "--manifest", str(manifest_path),
            "--device", "cpu", "--out", str(tmp_path / "out"), "--steps", "1",
        ],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    assert completed.returncode == 3
    assert "protected or owner training input forbidden" in completed.stderr


def test_full_train_wall_stop_is_honest_and_resumable(tmp_path: Path) -> None:
    manifest_path = _tiny_public_manifest(tmp_path / "public_manifest.json")
    partial_out = tmp_path / "partial"
    partial = run_full(
        manifest_path=manifest_path, device_name="cpu", out=partial_out, weights="none",
        steps=2, image_size=32, window_frames=3, batch_size=1, lr=0.001,
        val_every=1, seed=23, max_wall_minutes=1e-12, init_checkpoint=None,
        limit_clips=1, num_workers=0,
    )
    assert partial["status"] == "partial_wall_stop"
    assert partial["honest_partial"] is True
    assert partial["completed_steps"] == 0
    last_checkpoint = Path(partial["last_checkpoint"])
    assert last_checkpoint.is_file()

    resumed = run_full(
        manifest_path=manifest_path, device_name="cpu", out=tmp_path / "resumed",
        weights="none", steps=1, image_size=32, window_frames=3, batch_size=1,
        lr=0.001, val_every=1, seed=23, max_wall_minutes=None,
        init_checkpoint=last_checkpoint, limit_clips=1, num_workers=0,
    )
    assert resumed["status"] == "complete"
    assert resumed["start_step"] == 0
    assert resumed["completed_steps"] == 1
    assert resumed["init_checkpoint"] == str(last_checkpoint)


def test_git_head_is_safe_for_mirror_without_git_metadata(tmp_path: Path) -> None:
    assert _git_head(tmp_path) == "unavailable:no_git_metadata"

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from threed.racketsport.autolabel import (
    COMPATIBILITY_LABEL_FILES,
    PROTOTYPE_GATE_CLIPS,
    PROTOTYPE_LABEL_FILES,
    bootstrap_prototype_gate,
    h100_defaults,
)


def _write_clip(root: Path, frames_root: Path, name: str) -> None:
    clip_dir = root / name
    clip_dir.mkdir(parents=True, exist_ok=True)
    (clip_dir / "clip_metadata.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "camera_height": "high",
                "camera_angle": "shallow_baseline",
                "play_type": "doubles",
                "environment": "outdoor",
                "frame_rate_fps": 60,
                "duration_s": 90.0,
                "racket_gt": False,
            }
        ),
        encoding="utf-8",
    )
    clip_frames = frames_root / name
    clip_frames.mkdir(parents=True, exist_ok=True)
    frames = ["frame_000001.jpg", "frame_000031.jpg", "frame_000061.jpg"]
    for frame in frames:
        (clip_frames / frame).write_bytes(b"fake jpeg")
    (clip_frames / "label_frame_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "clip": name,
                "frame_count": len(frames),
                "frames": frames,
                "sample_every_frames": 30,
                "source_resolution": [1920, 1080],
            }
        ),
        encoding="utf-8",
    )


def test_bootstrap_writes_draft_packages_without_touching_dataset_labels(tmp_path: Path) -> None:
    root = tmp_path / "data" / "testclips"
    frames_root = tmp_path / "runs" / "label_frames"
    for index in range(6):
        _write_clip(root, frames_root, f"candidate_{index + 1:03d}")
    out = tmp_path / "runs" / "label_drafts" / "prototype_gate"

    summary = bootstrap_prototype_gate(root=root, frames_root=frames_root, out=out)

    assert summary["status"] == "draft_ready_for_review"
    assert summary["clip_count"] == 5
    assert summary["dataset_labels_written"] is False
    labels_dir = out / "candidate_001" / "labels"
    expected = set(PROTOTYPE_LABEL_FILES) | set(COMPATIBILITY_LABEL_FILES) | {
        "uncertain_frames.json",
        "prototype_autolabel_manifest.json",
        "status.json",
    }
    assert expected.issubset({path.name for path in labels_dir.glob("*.json")})
    assert not (root / "candidate_001" / "labels").exists()

    ball = json.loads((labels_dir / "ball.json").read_text(encoding="utf-8"))
    assert ball["status"] == "draft_prototype_unverified"
    assert ball["source"]["mode"] == "deterministic_smoke"
    assert ball["confidence"]["verified"] is False
    assert "teacher_model_unavailable" in ball["confidence"]["uncertainty_flags"]
    assert ball["annotation"]["target_file"] == "ball.json"
    assert len(ball["annotation"]["items"]) == 3

    uncertain = json.loads((labels_dir / "uncertain_frames.json").read_text(encoding="utf-8"))
    assert uncertain["status"] == "draft_requires_review"
    assert uncertain["frames"][0]["target_file"]


def test_bootstrap_preserves_teacher_payload_but_still_marks_draft(tmp_path: Path) -> None:
    root = tmp_path / "data" / "testclips"
    frames_root = tmp_path / "runs" / "label_frames"
    _write_clip(root, frames_root, "candidate_001")
    teacher_labels = tmp_path / "runs" / "teachers" / "candidate_001" / "labels"
    teacher_labels.mkdir(parents=True)
    (teacher_labels / "ball.json").write_text(json.dumps({"detections": [{"conf": 0.91}]}), encoding="utf-8")

    out = tmp_path / "runs" / "eval0" / "prototype_gate"
    bootstrap_prototype_gate(root=root, frames_root=frames_root, out=out, teacher_root=tmp_path / "runs" / "teachers")

    ball = json.loads((out / "candidate_001" / "labels" / "ball.json").read_text(encoding="utf-8"))
    assert ball["status"] == "draft_prototype_unverified"
    assert ball["source"]["mode"] == "teacher_artifact"
    assert ball["source"]["teacher_path"] == str(teacher_labels / "ball.json")
    assert ball["annotation"]["teacher_payload"]["detections"][0]["conf"] == 0.91


def test_bootstrap_uses_source_clip_cache_when_testclip_source_is_absent(tmp_path: Path) -> None:
    root = tmp_path / "data" / "testclips"
    frames_root = tmp_path / "runs" / "label_frames"
    clip = "candidate_001"
    _write_clip(root, frames_root, clip)
    cached = tmp_path / "data" / "source_clips" / f"{clip}.mp4"
    cached.parent.mkdir(parents=True)
    cached.write_bytes(b"fake mp4")
    (root / clip / "source.mp4").unlink(missing_ok=True)

    out = tmp_path / "runs" / "label_drafts" / "prototype_gate"
    bootstrap_prototype_gate(root=root, frames_root=frames_root, out=out)

    manifest = json.loads((out / clip / "labels" / "prototype_autolabel_manifest.json").read_text(encoding="utf-8"))
    assert manifest["clip"]["source_video"] == str(cached)


def test_bootstrap_refuses_dataset_output_and_h100_defaults_are_scoped(tmp_path: Path) -> None:
    root = tmp_path / "data" / "testclips"
    frames_root = tmp_path / "runs" / "label_frames"
    _write_clip(root, frames_root, "candidate_001")

    with pytest.raises(ValueError, match="refusing to write prototype drafts"):
        bootstrap_prototype_gate(root=root, frames_root=frames_root, out=root)

    defaults = h100_defaults(output_space="eval0")
    assert defaults["root"] == Path("/workspace/pickleball/data/testclips")
    assert defaults["frames_root"] == Path("/workspace/pickleball/runs/label_frames")
    assert defaults["out"] == Path("runs/eval0/prototype_gate")
    assert defaults["clip_names"] == list(PROTOTYPE_GATE_CLIPS)


def test_prototype_gate_clip_order_and_cli(tmp_path: Path) -> None:
    root = tmp_path / "data" / "testclips"
    frames_root = tmp_path / "runs" / "label_frames"
    for name in reversed(PROTOTYPE_GATE_CLIPS):
        _write_clip(root, frames_root, name)
    out = tmp_path / "runs" / "label_drafts" / "prototype_gate"

    summary = bootstrap_prototype_gate(root=root, frames_root=frames_root, out=out)
    assert [clip["clip"] for clip in summary["clips"]] == list(PROTOTYPE_GATE_CLIPS)

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/run_prototype_autolabel.py",
            "--root",
            str(root),
            "--frames-root",
            str(frames_root),
            "--out",
            str(out),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["artifact_type"] == "racketsport_prototype_autolabel_run"
    assert payload["clip_count"] == 5

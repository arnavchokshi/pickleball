from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from threed.racketsport.label_overlay import PROTOTYPE_GATE_CLIPS, render_label_overlays


def _make_video(path: Path, *, frames: int = 4) -> None:
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 4.0, (64, 48))
    if not writer.isOpened():
        pytest.skip("OpenCV cannot write mp4")
    for index in range(frames):
        writer.write(np.full((48, 64, 3), 30 + index * 8, dtype=np.uint8))
    writer.release()


def _write_draft(path: Path, target: str, items: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"schema_version": 1, "status": "draft_manual_annotation", "annotation": {"target_file": target, "items": items}}),
        encoding="utf-8",
    )


def _decoded_count(path: Path) -> int:
    cv2 = pytest.importorskip("cv2")
    cap = cv2.VideoCapture(str(path))
    count = 0
    try:
        while True:
            ok, _ = cap.read()
            if not ok:
                return count
            count += 1
    finally:
        cap.release()


def test_render_label_overlays_draws_available_layers_and_index(tmp_path: Path) -> None:
    video = tmp_path / "candidate_001.mp4"
    labels = tmp_path / "labels"
    out_root = tmp_path / "runs" / "eval0" / "prototype_gate"
    _make_video(video)
    _write_draft(labels / "court_corners.json", "court_corners.json", [{"id": "near_left", "xy_px": [8, 40]}])
    _write_draft(labels / "players.json", "players.json", [{"frame": 1, "id": "p1", "bbox": [14, 12, 18, 26]}])
    _write_draft(labels / "ball.json", "ball.json", [{"frame": 0, "xy_px": [18, 20]}, {"frame": 1, "xy_px": [28, 22]}])
    _write_draft(labels / "events.json", "events.json", [{"frame": 1, "type": "contact", "xy_px": [30, 24], "label": "hit"}])
    _write_draft(labels / "racket_pose.json", "racket_pose.json", [{"frame": 1, "keypoints_px": [[32, 20], [40, 24]]}])
    _write_draft(labels / "foot_contact.json", "foot_contact.json", [{"frame": 2, "foot": "left", "xy_px": [22, 38]}])

    summary = render_label_overlays(video_path=video, draft_label_dir=labels, output_root=out_root, clip_name="candidate_001", write_markdown=True)

    compare_dir = out_root / "candidate_001" / "compare"
    overlay_path = compare_dir / "all_labels_overlay.mp4"
    assert summary["status"] == "rendered"
    assert set(summary["available_layers"]) == {"court", "players", "ball", "events", "racket", "foot_contact"}
    assert summary["frame_count"] == 4
    assert overlay_path.stat().st_size > 0
    assert _decoded_count(overlay_path) == 4
    assert (compare_dir / "label_overlay_index.json").is_file()
    assert "all_labels_overlay.mp4" in (compare_dir / "label_overlay_index.md").read_text(encoding="utf-8")


def test_render_label_overlays_tolerates_sparse_labels_and_cli_defaults(tmp_path: Path) -> None:
    video = tmp_path / "data" / "testclips" / PROTOTYPE_GATE_CLIPS[0] / "source.mp4"
    labels = tmp_path / "runs" / "eval0" / "prototype_gate" / PROTOTYPE_GATE_CLIPS[0] / "labels"
    _make_video(video, frames=2)
    _write_draft(labels / "ball.json", "ball.json", [{"frame": 0, "xy_px": [20, 18]}])

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/render_label_overlays.py",
            "--root",
            str(tmp_path),
            "--clip",
            PROTOTYPE_GATE_CLIPS[0],
            "--markdown",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads(completed.stdout)
    clip_summary = summary["clips"][0]
    assert clip_summary["available_layers"] == ["ball"]
    assert clip_summary["qualitative_status"] == "prototype_not_gate_verified"
    assert (tmp_path / "runs" / "eval0" / "prototype_gate" / PROTOTYPE_GATE_CLIPS[0] / "compare" / "all_labels_overlay.mp4").is_file()

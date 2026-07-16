from __future__ import annotations

import json
import subprocess
from pathlib import Path

import torch

from threed.racketsport.event_head.model import EventHead, checkpoint_payload


ROOT = Path(__file__).resolve().parents[2]
CLI = "scripts/racketsport/build_event_head_anchor_candidates.py"


def test_anchor_candidate_cli_emits_frozen_track_a_schema(tmp_path: Path) -> None:
    checkpoint = tmp_path / "event_head.pt"
    video = ROOT / "tests/racketsport/fixtures/event_head/tiny.avi"
    out = tmp_path / "anchors.json"
    model = EventHead(weights="none", feature_dim=8, hidden_dim=8)
    torch.save(checkpoint_payload(
        model, image_size=32, window_frames=3, license_posture="RD_ONLY",
        pretrain_data="synthetic_fixture_only",
    ), checkpoint)
    completed = subprocess.run(
        [
            str(ROOT / ".venv/bin/python"), CLI, "--checkpoint", str(checkpoint),
            "--video", str(video), "--out", str(out), "--threshold", "0.0",
            "--nms-radius-frames", "1", "--device", "cpu", "--stride", "2",
            "--max-seconds", "0.8", "--video-provenance", "synthetic_fixture",
        ],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    artifact = json.loads(out.read_text())
    assert list(artifact) == [
        "artifact_type", "schema_version", "source_video", "video_provenance",
        "never_training", "review_only", "verified", "model", "config", "events",
        "counts", "honest_limits",
    ]
    assert artifact["artifact_type"] == "event_head_contact_anchor_candidates"
    assert artifact["schema_version"] == 1
    assert artifact["video_provenance"] == "synthetic_fixture"
    assert artifact["never_training"] is True
    assert artifact["review_only"] is True
    assert artifact["verified"] is False
    assert set(artifact["source_video"]) == {"path", "sha256"}
    assert len(artifact["source_video"]["sha256"]) == 64
    assert set(artifact["model"]) == {
        "checkpoint_path", "checkpoint_sha256", "license_posture", "pretrain_data",
    }
    assert artifact["model"]["license_posture"] == "RD_ONLY"
    assert artifact["model"]["pretrain_data"] == "synthetic_fixture_only"
    assert artifact["config"] == {
        "threshold": 0.0, "nms_radius_frames": 1, "stride": 2,
        "image_size": 32, "window_frames": 3, "fps": 10.0,
        "pts_convention": "normalized_to_first_video_pts",
    }
    assert artifact["counts"] == {
        name: sum(event["class"] == name for event in artifact["events"])
        for name in ("HIT", "BOUNCE")
    }
    assert artifact["events"]
    for event in artifact["events"]:
        assert list(event) == ["frame_idx", "pts_s", "class", "score"]
        assert event["class"] in {"HIT", "BOUNCE"}
        assert event["pts_s"] == event["frame_idx"] / artifact["config"]["fps"]

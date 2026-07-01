from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from threed.racketsport.body_world_label_review_bundle import build_body_world_label_review_bundle


def _packet() -> dict:
    samples = [
        {
            "sample_id": "frame_000001_player_7",
            "frame_index": 1,
            "t": 1.0 / 30.0,
            "player_id": 7,
            "track_world_xy": [0.0, -3.0],
            "predicted_joints_world": [[0.0, 0.0, 0.1], [0.2, 0.0, 1.4]],
            "joint_conf": [0.9, 0.8],
            "joint_count": 2,
            "review_required": True,
        },
        {
            "sample_id": "frame_000004_player_8",
            "frame_index": 4,
            "t": 4.0 / 30.0,
            "player_id": 8,
            "track_world_xy": [1.0, -2.0],
            "predicted_joints_world": [[1.0, 0.0, 0.1], [1.2, 0.0, 1.4]],
            "joint_conf": [0.7, 0.6],
            "joint_count": 2,
            "review_required": True,
        },
    ]
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_body_world_label_packet",
        "clip": "clip_001",
        "status": "needs_review",
        "not_ground_truth": True,
        "trusted_for_world_mpjpe": False,
        "source_video": "source.mp4",
        "suggested_label_path": "labels/body_world_joints.json",
        "joint_names": ["pelvis", "neck"],
        "samples": samples,
        "review_plan": {
            "expected_sample_count": 2,
            "required_sample_count": 1,
            "selected_sample_count": 1,
            "selected_sample_ids": ["frame_000004_player_8"],
            "min_sample_count": 20,
            "min_coverage_ratio": 0.1,
        },
        "summary": {"sample_count": 2, "player_count": 2, "frame_count": 2, "joint_count_min": 2, "joint_count_max": 2},
    }


def test_body_world_label_review_bundle_writes_selected_queue_and_safe_template(tmp_path: Path) -> None:
    frames = tmp_path / "body_frames"
    frames.mkdir()
    (frames / "frame_000004.jpg").write_bytes(b"fake jpeg")
    out = tmp_path / "review_bundle"

    manifest = build_body_world_label_review_bundle(
        packet=_packet(),
        body_frames_dir=frames,
        out_dir=out,
        packet_path=tmp_path / "body_world_label_packet.json",
    )

    assert manifest["artifact_type"] == "racketsport_body_world_label_review_bundle"
    assert manifest["status"] == "ready_for_review"
    assert manifest["selected_sample_count"] == 1
    assert manifest["missing_frame_count"] == 0
    assert manifest["final_label_path"] == str(out.parent / "labels" / "body_world_joints.json")
    assert manifest["finalize_command"] == (
        "python scripts/racketsport/finalize_body_world_labels.py "
        f"--template {out / 'body_world_joints.template.json'} "
        f"--out {out.parent / 'labels' / 'body_world_joints.json'} "
        f"--report-out {out / 'body_world_label_finalization.json'}"
    )
    assert (out / "frames" / "frame_000004.jpg").read_bytes() == b"fake jpeg"
    queue = json.loads((out / "body_world_label_review_queue.json").read_text(encoding="utf-8"))
    assert queue["samples"][0]["sample_id"] == "frame_000004_player_8"
    assert queue["samples"][0]["image_path"] == str(out / "frames" / "frame_000004.jpg")
    template = json.loads((out / "body_world_joints.template.json").read_text(encoding="utf-8"))
    assert template["artifact_type"] == "racketsport_body_world_joints_labels"
    assert template["status"] == "draft_review_template"
    assert template["not_ground_truth"] is True
    assert template["trusted_for_world_mpjpe"] is False
    assert template["samples"][0]["accepted"] is False
    assert template["samples"][0]["joints_world"] == []
    assert template["samples"][0]["predicted_joints_world"] == [[1.0, 0.0, 0.1], [1.2, 0.0, 1.4]]


def test_body_world_label_review_bundle_blocks_when_selected_frame_image_is_missing(tmp_path: Path) -> None:
    manifest = build_body_world_label_review_bundle(
        packet=_packet(),
        body_frames_dir=tmp_path / "missing_body_frames",
        out_dir=tmp_path / "review_bundle",
    )

    assert manifest["status"] == "blocked_missing_review_frames"
    assert manifest["missing_frame_count"] == 1
    assert manifest["missing_frames"][0]["frame_index"] == 4
    assert manifest["missing_frames"][0]["sample_id"] == "frame_000004_player_8"


def test_build_body_world_label_review_bundle_cli_writes_manifest(tmp_path: Path) -> None:
    packet = tmp_path / "body_world_label_packet.json"
    frames = tmp_path / "body_frames"
    out = tmp_path / "review_bundle"
    packet.write_text(json.dumps(_packet()), encoding="utf-8")
    frames.mkdir()
    (frames / "frame_000004.jpg").write_bytes(b"fake jpeg")

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/build_body_world_label_review_bundle.py",
            "--packet",
            str(packet),
            "--body-frames-dir",
            str(frames),
            "--out-dir",
            str(out),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert completed.stderr == ""
    summary = json.loads(completed.stdout)
    assert summary["status"] == "ready_for_review"
    assert summary["selected_sample_count"] == 1
    assert (out / "body_world_label_review_bundle.json").is_file()

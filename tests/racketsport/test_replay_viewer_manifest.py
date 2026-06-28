from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from threed.racketsport.replay_viewer_manifest import build_replay_viewer_manifest, write_replay_viewer_manifest


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def test_build_replay_viewer_manifest_links_video_world_and_non_promoting_labels(tmp_path: Path) -> None:
    run_dir = tmp_path / "clip_a"
    video = run_dir / "tracknet_smoke_0000_0010" / "input_0000_0010.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"video")
    virtual_world = _write_json(
        run_dir / "virtual_world.json",
        {"schema_version": 1, "artifact_type": "racketsport_virtual_world", "world_frame": "court_Z0"},
    )
    labels = _write_json(
        run_dir / "labels" / "players.json",
        {
            "schema_version": 1,
            "not_ground_truth": True,
            "annotation": {"items": [{"frame": "frame_000001.jpg", "bbox_xyxy": [1, 2, 3, 4]}]},
        },
    )
    person_gt = _write_json(
        tmp_path / "labels" / "task_1" / "person_ground_truth.json",
        {
            "schema_version": 1,
            "artifact_type": "racketsport_person_ground_truth",
            "clip_id": "task_1",
            "frames": [],
        },
    )
    physics = _write_json(
        run_dir / "physics_refinement.json",
        {
            "schema_version": 1,
            "artifact_type": "racketsport_physics_refinement",
            "physics": "cpu_fallback_scaffold",
        },
    )
    contacts = _write_json(run_dir / "contact_windows.json", {"schema_version": 1, "events": []})

    manifest = build_replay_viewer_manifest(
        clip="clip_a",
        video_path=video,
        virtual_world_path=virtual_world,
        player_labels_path=labels,
        physics_refinement_path=physics,
        contact_windows_path=contacts,
        annotation_sources=[person_gt],
    )

    assert manifest["artifact_type"] == "racketsport_replay_viewer_manifest"
    assert manifest["clip"] == "clip_a"
    assert manifest["video_url"].startswith("/@fs/")
    assert manifest["virtual_world_url"].startswith("/@fs/")
    assert manifest["physics_refinement_url"].startswith("/@fs/")
    assert manifest["contact_windows_url"].startswith("/@fs/")
    assert manifest["label_overlays"] == [
        {
            "kind": "player_boxes",
            "label": "prototype player boxes",
            "url": manifest["label_overlays"][0]["url"],
            "trusted_for_metrics": False,
            "not_ground_truth": True,
        }
    ]
    assert manifest["annotation_sources"][0]["kind"] == "person_ground_truth"
    assert manifest["annotation_sources"][0]["trusted_for_metrics"] is True


def test_replay_viewer_manifest_cli_writes_manifest(tmp_path: Path) -> None:
    run_dir = tmp_path / "clip_a"
    video = run_dir / "video.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"video")
    _write_json(run_dir / "virtual_world.json", {"artifact_type": "racketsport_virtual_world"})
    _write_json(run_dir / "labels" / "players.json", {"not_ground_truth": True, "annotation": {"items": []}})
    out = run_dir / "replay_viewer_manifest.json"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/build_replay_viewer_manifest.py",
            "--clip",
            "clip_a",
            "--video",
            str(video),
            "--virtual-world",
            str(run_dir / "virtual_world.json"),
            "--player-labels",
            str(run_dir / "labels" / "players.json"),
            "--out",
            str(out),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert completed.stderr == ""
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["clip"] == "clip_a"
    assert json.loads(completed.stdout)["out"] == str(out)

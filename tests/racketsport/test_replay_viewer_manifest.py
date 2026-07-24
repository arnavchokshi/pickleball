from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from threed.racketsport.replay_viewer_manifest import build_replay_viewer_manifest, write_replay_viewer_manifest
from threed.racketsport.schemas import ReplayViewerManifest, validate_artifact_file


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
            "source_format": "cvat_mot_1_1",
            "source_path": "task_1.zip",
            "fps": 30.0,
            "frames": [],
            "summary": {
                "frame_count": 0,
                "valid_label_count": 0,
                "ignored_label_count": 0,
                "track_ids": [],
                "max_valid_players_per_frame": 0,
            },
        },
    )
    replay_scene = _write_json(
        run_dir / "replay_scene.json",
        {
            "schema_version": 1,
            "world_frame": "court_Z0",
            "fps": 30.0,
            "court_glb": "court_review.glb",
            "players": [],
            "points": [],
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
    ball_inflections = _write_json(
        run_dir / "ball_inflections.json",
        {"schema_version": 1, "artifact_type": "racketsport_ball_inflections", "candidates": []},
    )
    body_mesh = _write_json(
        run_dir / "body_mesh.json",
        {
            "schema_version": 1,
            "artifact_type": "racketsport_body_mesh",
            "clip": "clip_a",
            "model": "sam3dbody_world_joints",
            "fps": 30.0,
            "world_frame": "court_Z0",
            "faces_ref": "mhr_faces_static",
            "mesh_faces": [[0, 1, 2]],
            "joint_names": ["left_wrist"],
            "players": [],
            "summary": {"mesh_frame_count": 0, "player_count": 0, "contact_window_count": 0},
        },
    )
    body_mesh_index = _write_json(
        run_dir / "body_mesh_index.json",
        {
            "schema_version": 1,
            "artifact_type": "racketsport_body_mesh_index",
            "clip": "clip_a",
            "model": "sam3dbody_world_joints",
            "fps": 30.0,
            "world_frame": "court_Z0",
            "faces_ref": "mhr_faces_static",
            "faces_url": "body_mesh_faces.json",
            "windows": [],
            "summary": {"window_count": 0, "mesh_frame_count": 0, "player_count": 0, "faces_count": 0},
        },
    )
    court_evidence = _write_json(
        run_dir / "court_lock_visualization_adapter.json",
        {"schema_version": 1, "artifact_type": "racketsport_court_lock_visualization_adapter"},
    )
    court_calibration = _write_json(
        run_dir / "court_calibration.json",
        {"schema_version": 1, "sport": "pickleball"},
    )
    skeleton_evidence = _write_json(
        run_dir / "sam3d_keypoints_2d.json",
        {"schema_version": 1, "artifact_type": "racketsport_sam3d_keypoints_2d"},
    )

    manifest = build_replay_viewer_manifest(
        clip="clip_a",
        video_path=video,
        virtual_world_path=virtual_world,
        player_labels_path=labels,
        replay_scene_path=replay_scene,
        physics_refinement_path=physics,
        contact_windows_path=contacts,
        ball_inflections_path=ball_inflections,
        body_mesh_path=body_mesh,
        body_mesh_index_path=body_mesh_index,
        court_evidence_path=court_evidence,
        court_calibration_path=court_calibration,
        skeleton_evidence_path=skeleton_evidence,
        annotation_sources=[person_gt],
        vite_allow_root=tmp_path,
    )

    assert manifest["artifact_type"] == "racketsport_replay_viewer_manifest"
    assert manifest["clip"] == "clip_a"
    assert manifest["video_url"].startswith("/@fs/")
    assert manifest["virtual_world_url"].startswith("/@fs/")
    assert manifest["replay_scene_url"].startswith("/@fs/")
    assert manifest["body_mesh_url"].startswith("/@fs/")
    assert manifest["body_mesh_index_url"].startswith("/@fs/")
    assert manifest["court_evidence_url"].startswith("/@fs/")
    assert manifest["court_calibration_url"].startswith("/@fs/")
    assert manifest["skeleton_evidence_url"].startswith("/@fs/")
    assert manifest["physics_refinement_url"].startswith("/@fs/")
    assert manifest["contact_windows_url"].startswith("/@fs/")
    assert manifest["ball_inflections_url"].startswith("/@fs/")
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
    assert isinstance(ReplayViewerManifest.model_validate(manifest), ReplayViewerManifest)


def test_replay_viewer_manifest_rejects_files_outside_default_vite_allow_root(tmp_path: Path) -> None:
    run_dir = tmp_path / "clip_a"
    video = run_dir / "video.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"video")
    virtual_world = _write_json(run_dir / "virtual_world.json", {"artifact_type": "racketsport_virtual_world"})

    with pytest.raises(ValueError, match="outside Vite allow root"):
        build_replay_viewer_manifest(
            clip="clip_a",
            video_path=video,
            virtual_world_path=virtual_world,
        )


def test_replay_viewer_manifest_rejects_unvalidated_person_ground_truth_sources(tmp_path: Path) -> None:
    run_dir = tmp_path / "clip_a"
    video = run_dir / "video.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"video")
    virtual_world = _write_json(run_dir / "virtual_world.json", {"artifact_type": "racketsport_virtual_world"})
    invalid_person_gt = _write_json(
        tmp_path / "labels" / "task_1" / "person_ground_truth.json",
        {
            "schema_version": 1,
            "artifact_type": "racketsport_person_ground_truth",
            "clip_id": "task_1",
            "frames": [],
        },
    )

    try:
        build_replay_viewer_manifest(
            clip="clip_a",
            video_path=video,
            virtual_world_path=virtual_world,
            annotation_sources=[invalid_person_gt],
            vite_allow_root=tmp_path,
        )
    except Exception as exc:
        assert "source_format" in str(exc)
    else:
        raise AssertionError("invalid person_ground_truth source should not be marked trusted")


def test_replay_viewer_manifest_requires_explicit_label_trust_status(tmp_path: Path) -> None:
    run_dir = tmp_path / "clip_a"
    video = run_dir / "video.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"video")
    virtual_world = _write_json(run_dir / "virtual_world.json", {"artifact_type": "racketsport_virtual_world"})
    labels = _write_json(run_dir / "labels" / "players.json", {"annotation": {"items": []}})

    try:
        build_replay_viewer_manifest(
            clip="clip_a",
            video_path=video,
            virtual_world_path=virtual_world,
            player_labels_path=labels,
            vite_allow_root=tmp_path,
        )
    except ValueError as exc:
        assert "not_ground_truth" in str(exc)
    else:
        raise AssertionError("player labels without explicit not_ground_truth should not be trusted")


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
            "--vite-allow-root",
            str(tmp_path),
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
    assert isinstance(validate_artifact_file("replay_viewer_manifest", out), ReplayViewerManifest)


def test_replay_viewer_manifest_links_optional_review_coaching_and_rally_artifacts(tmp_path: Path) -> None:
    run_dir = tmp_path / "clip_a"
    video = run_dir / "video.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"video")
    virtual_world = _write_json(run_dir / "virtual_world.json", {"artifact_type": "racketsport_virtual_world"})
    reviewed_bounces = _write_json(
        run_dir / "reviewed_ball_bounces.json",
        {"artifact_type": "racketsport_reviewed_ball_bounces", "status": "human_reviewed", "bounces": []},
    )
    coaching_facts = _write_json(run_dir / "coaching_card_facts.json", {"artifact_type": "racketsport_coaching_card_facts"})
    rally_spans = _write_json(
        run_dir / "rally_spans.json",
        {"artifact_type": "racketsport_rally_spans", "not_ground_truth": True, "spans": []},
    )

    manifest = build_replay_viewer_manifest(
        clip="clip_a",
        video_path=video,
        virtual_world_path=virtual_world,
        reviewed_bounces_path=reviewed_bounces,
        coaching_card_facts_path=coaching_facts,
        rally_spans_path=rally_spans,
        vite_allow_root=tmp_path,
    )

    assert manifest["reviewed_bounces_url"].endswith("/reviewed_ball_bounces.json")
    assert manifest["coaching_card_facts_url"].endswith("/coaching_card_facts.json")
    assert manifest["rally_spans_url"].endswith("/rally_spans.json")


def test_replay_viewer_manifest_links_ball_arc_render_artifact(tmp_path: Path) -> None:
    run_dir = tmp_path / "clip_a"
    video = run_dir / "video.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"video")
    virtual_world = _write_json(run_dir / "virtual_world.json", {"artifact_type": "racketsport_virtual_world"})
    ball_arc_render = _write_json(
        run_dir / "ball_arc_render.json",
        {
            "schema_version": 1,
            "artifact_type": "racketsport_ball_arc_render",
            "clip_id": "clip_a",
            "source_artifact": "ball_track_arc_solved.json",
            "solver_status": "ran",
            "render_only": True,
            "not_for_detection_metrics": True,
            "trusted_for_ball_detection_metrics": False,
            "segments": [],
            "bridges": [],
            "samples": [],
            "summary": {"segment_count": 0, "sample_count": 0, "bridge_sample_count": 0, "rally_span_count": 0},
        },
    )

    manifest = build_replay_viewer_manifest(
        clip="clip_a",
        video_path=video,
        virtual_world_path=virtual_world,
        ball_arc_render_path=ball_arc_render,
        vite_allow_root=tmp_path,
    )

    assert manifest["ball_arc_render_url"].endswith("/ball_arc_render.json")
    assert isinstance(ReplayViewerManifest.model_validate(manifest), ReplayViewerManifest)


def test_replay_viewer_manifest_cli_writes_optional_review_coaching_and_rally_urls(tmp_path: Path) -> None:
    run_dir = tmp_path / "clip_a"
    video = run_dir / "video.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"video")
    _write_json(run_dir / "virtual_world.json", {"artifact_type": "racketsport_virtual_world"})
    _write_json(run_dir / "reviewed_ball_bounces.json", {"artifact_type": "racketsport_reviewed_ball_bounces", "status": "human_reviewed", "bounces": []})
    _write_json(run_dir / "coaching_card_facts.json", {"artifact_type": "racketsport_coaching_card_facts"})
    _write_json(run_dir / "rally_spans.json", {"artifact_type": "racketsport_rally_spans", "not_ground_truth": True, "spans": []})
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
            "--reviewed-bounces",
            str(run_dir / "reviewed_ball_bounces.json"),
            "--coaching-card-facts",
            str(run_dir / "coaching_card_facts.json"),
            "--rally-spans",
            str(run_dir / "rally_spans.json"),
            "--vite-allow-root",
            str(tmp_path),
            "--out",
            str(out),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["reviewed_bounces_url"].endswith("/reviewed_ball_bounces.json")
    assert payload["coaching_card_facts_url"].endswith("/coaching_card_facts.json")
    assert payload["rally_spans_url"].endswith("/rally_spans.json")


def test_replay_viewer_manifest_writer_validates_schema(tmp_path: Path) -> None:
    out = tmp_path / "replay_viewer_manifest.json"
    with pytest.raises(Exception, match="virtual_world_url"):
        write_replay_viewer_manifest(
            out,
            {
                "schema_version": 1,
                "artifact_type": "racketsport_replay_viewer_manifest",
                "clip": "clip_a",
                "video_url": "/@fs/tmp/clip_a/input.mp4",
                "label_overlays": [],
                "annotation_sources": [],
                "notes": [],
            },
        )

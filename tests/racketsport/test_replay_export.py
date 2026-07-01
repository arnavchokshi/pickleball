from __future__ import annotations

import json
import subprocess
import sys

import pytest

from threed.racketsport.replay_export import (
    audit_replay_export_manifest,
    build_replay_review_export_from_virtual_world,
    build_replay_export_manifest,
    inspect_glb_file,
    validate_replay_export_manifest,
    write_replay_scene,
)
from threed.racketsport.schemas import ReplayScene


def _virtual_world_payload() -> dict:
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_virtual_world",
        "world_frame": "court_Z0",
        "fps": 30.0,
        "court": {
            "sport": "pickleball",
            "coordinate_frame": "court_Z0",
            "length_m": 13.41,
            "width_m": 6.1,
            "line_segments": {
                "near_baseline": [[-3.05, 0.0, 0.0], [3.05, 0.0, 0.0]],
                "left_sideline": [[-3.05, 0.0, 0.0], [-3.05, 13.41, 0.0]],
            },
            "net": {"endpoints": [[-3.05, 6.705, 0.91], [3.05, 6.705, 0.91]], "center_height_m": 0.86, "post_height_m": 0.91},
        },
        "players": [
            {
                "id": 1,
                "side": "near",
                "role": "left",
                "representation": "world_mesh",
                "frames": [
                    {
                        "t": 0.0,
                        "track_world_xy": [0.0, 1.0],
                        "joints_world": [[0.0, 1.0, 1.2]],
                        "mesh_vertices_world": [[0.0, 1.0, 0.0], [0.1, 1.0, 0.0]],
                    },
                    {"t": 1 / 30, "track_world_xy": [0.2, 1.2], "joints_world": [[0.2, 1.2, 1.1]]},
                ],
            }
        ],
        "ball": {"source": "tracknet", "frames": [{"t": 0.0, "world_xyz": [0.0, 6.0, 0.3], "visible": True, "conf": 0.9}]},
        "paddles": [
            {
                "player_id": 1,
                "paddle_dims_in": {"length": 16.0, "width": 8.0},
                "frames": [
                    {
                        "t": 0.0,
                        "mesh_vertices_world": [[0, 0, 0], [0.1, 0, 0], [0.1, 0.2, 0], [0, 0.2, 0]],
                        "mesh_faces": [[0, 1, 2], [0, 2, 3]],
                        "ambiguous": True,
                    }
                ],
            }
        ],
    }


def test_build_replay_export_manifest_returns_replayscene_with_measured_sizes(tmp_path):
    export_root = tmp_path / "clip_001"
    export_root.mkdir()
    (export_root / "court.glb").write_bytes(b"court")
    (export_root / "points").mkdir()
    (export_root / "points" / "point_1.glb").write_bytes(b"a" * 2_500_000)
    (export_root / "points" / "point_2.glb").write_bytes(b"b" * 1_250_000)

    scene = build_replay_export_manifest(
        export_root=export_root,
        court_glb="court.glb",
        point_glbs=[
            {"id": 1, "t0": 0.0, "t1": 3.5, "glb_path": "points/point_1.glb"},
            {"id": 2, "t0": 3.5, "t1": 7.0, "glb_path": "points/point_2.glb"},
        ],
        players=[2, 1],
        fps=60,
    )

    assert isinstance(scene, ReplayScene)
    assert scene.model_dump(mode="json") == {
        "schema_version": 1,
        "world_frame": "court_Z0",
        "fps": 60.0,
        "court_glb": "court.glb",
        "players": [1, 2],
        "points": [
            {"id": 1, "t0": 0.0, "t1": 3.5, "glb_url": "points/point_1.glb", "size_mb": 2.5},
            {"id": 2, "t0": 3.5, "t1": 7.0, "glb_url": "points/point_2.glb", "size_mb": 1.25},
        ],
    }


def test_validate_replay_export_manifest_rejects_unsafe_relative_paths(tmp_path):
    export_root = tmp_path / "clip_001"
    export_root.mkdir()
    (export_root / "court.glb").write_bytes(b"court")

    manifest = {
        "schema_version": 1,
        "world_frame": "court_Z0",
        "fps": 30.0,
        "court_glb": "court.glb",
        "players": [1],
        "points": [{"id": 1, "t0": 0.0, "t1": 1.0, "glb_url": "../point_1.glb", "size_mb": 1.0}],
    }

    with pytest.raises(ValueError, match="points/0/glb_url must be relative and stay within export_root"):
        validate_replay_export_manifest(export_root, manifest)


def test_build_replay_export_manifest_rejects_missing_referenced_files(tmp_path):
    export_root = tmp_path / "clip_001"
    export_root.mkdir()
    (export_root / "court.glb").write_bytes(b"court")

    with pytest.raises(FileNotFoundError, match="points/missing.glb"):
        build_replay_export_manifest(
            export_root=export_root,
            court_glb="court.glb",
            point_glbs=[{"id": 1, "t0": 0.0, "t1": 1.0, "glb_path": "points/missing.glb"}],
            players=[1],
            fps=30.0,
        )


def test_validate_replay_export_manifest_rejects_invalid_point_timing(tmp_path):
    export_root = tmp_path / "clip_001"
    export_root.mkdir()
    (export_root / "court.glb").write_bytes(b"court")
    (export_root / "points").mkdir()
    (export_root / "points" / "point_1.glb").write_bytes(b"a")
    (export_root / "points" / "point_2.glb").write_bytes(b"b")

    backwards_point = {
        "schema_version": 1,
        "world_frame": "court_Z0",
        "fps": 30.0,
        "court_glb": "court.glb",
        "players": [1],
        "points": [{"id": 1, "t0": 2.0, "t1": 1.0, "glb_url": "points/point_1.glb", "size_mb": 0.000001}],
    }
    overlapping_points = {
        **backwards_point,
        "points": [
            {"id": 1, "t0": 0.0, "t1": 2.0, "glb_url": "points/point_1.glb", "size_mb": 0.000001},
            {"id": 2, "t0": 1.5, "t1": 3.0, "glb_url": "points/point_2.glb", "size_mb": 0.000001},
        ],
    }

    with pytest.raises(ValueError, match="points/0/t1 must be greater than points/0/t0"):
        validate_replay_export_manifest(export_root, backwards_point)
    with pytest.raises(ValueError, match="points/1/t0 must be greater than or equal to previous point t1"):
        validate_replay_export_manifest(export_root, overlapping_points)


def test_build_replay_review_export_from_virtual_world_writes_glbs_and_scene(tmp_path):
    virtual_world = _virtual_world_payload()

    scene = build_replay_review_export_from_virtual_world(virtual_world, export_root=tmp_path)
    write_replay_scene(tmp_path / "replay_scene.json", scene)
    payload = scene.model_dump(mode="json")

    assert isinstance(scene, ReplayScene)
    assert payload["court_glb"] == "court_review.glb"
    assert payload["players"] == [1]
    assert payload["points"][0]["glb_url"] == "points/point_001_review.glb"
    assert (tmp_path / "court_review.glb").read_bytes()[:4] == b"glTF"
    assert (tmp_path / payload["points"][0]["glb_url"]).read_bytes()[:4] == b"glTF"
    assert payload["points"][0]["size_mb"] == round((tmp_path / payload["points"][0]["glb_url"]).stat().st_size / 1_000_000, 6)
    assert validate_replay_export_manifest(tmp_path, scene) == scene
    assert json.loads((tmp_path / "replay_scene.json").read_text(encoding="utf-8")) == payload


def test_validate_replay_export_manifest_rejects_static_review_glbs_for_production(tmp_path):
    scene = build_replay_review_export_from_virtual_world(_virtual_world_payload(), export_root=tmp_path)

    assert validate_replay_export_manifest(tmp_path, scene) == scene
    with pytest.raises(ValueError, match="review_static_glb_export"):
        validate_replay_export_manifest(tmp_path, scene, require_production=True)


def test_validate_replay_export_manifest_rejects_empty_production_scene(tmp_path):
    scene = build_replay_review_export_from_virtual_world(_virtual_world_payload(), export_root=tmp_path)
    payload = scene.model_dump(mode="json")
    payload["players"] = []
    payload["points"] = []

    audit = audit_replay_export_manifest(tmp_path, payload)

    assert audit["production_replay_ready"] is False
    assert "missing_replay_players" in audit["blockers"]
    assert "missing_replay_points" in audit["blockers"]
    with pytest.raises(ValueError, match="missing_replay_players"):
        validate_replay_export_manifest(tmp_path, payload, require_production=True)


def test_inspect_glb_file_reports_generated_review_scene_structure(tmp_path):
    scene = build_replay_review_export_from_virtual_world(_virtual_world_payload(), export_root=tmp_path)

    court_info = inspect_glb_file(tmp_path / scene.court_glb)
    point_info = inspect_glb_file(tmp_path / scene.points[0].glb_url)

    assert court_info["valid"] is True
    assert court_info["version"] == 2
    assert court_info["asset_version"] == "2.0"
    assert court_info["mesh_count"] >= 1
    assert court_info["primitive_count"] >= 1
    assert court_info["json_chunk_bytes"] > 0
    assert point_info["valid"] is True
    assert point_info["node_count"] >= 1
    assert point_info["bin_chunk_bytes"] > 0


def test_inspect_glb_file_rejects_placeholder_bytes(tmp_path):
    bad_glb = tmp_path / "point_3.glb"
    bad_glb.write_bytes(b"glb")

    with pytest.raises(ValueError, match="invalid GLB magic"):
        inspect_glb_file(bad_glb)


def test_build_replay_review_export_cli_writes_scene_and_glbs(tmp_path):
    virtual_world = tmp_path / "virtual_world_paddle_preview.json"
    virtual_world.write_text(json.dumps(_virtual_world_payload()), encoding="utf-8")
    out_dir = tmp_path / "replay_review"
    scene_out = tmp_path / "replay_scene.json"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/build_replay_review_export.py",
            "--virtual-world",
            str(virtual_world),
            "--out-dir",
            str(out_dir),
            "--scene-out",
            str(scene_out),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads(completed.stdout)
    scene = json.loads(scene_out.read_text(encoding="utf-8"))

    assert summary["schema_version"] == 1
    assert summary["scene_out"] == str(scene_out)
    assert scene["court_glb"] == "replay_review/court_review.glb"
    assert (scene_out.parent / scene["court_glb"]).is_file()
    assert (scene_out.parent / scene["points"][0]["glb_url"]).is_file()

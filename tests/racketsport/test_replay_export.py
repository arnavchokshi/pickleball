from __future__ import annotations

import pytest

from threed.racketsport.replay_export import (
    build_replay_export_manifest,
    validate_replay_export_manifest,
)
from threed.racketsport.schemas import ReplayScene


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

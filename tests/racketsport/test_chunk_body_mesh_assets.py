from __future__ import annotations

import gzip
import json
import subprocess
import sys
from pathlib import Path

from scripts.racketsport.chunk_body_mesh_assets import chunk_body_mesh_assets


FIXTURES = Path("tests/racketsport/fixtures")


def test_chunk_body_mesh_assets_writes_shared_faces_index_and_quantized_window_chunks(tmp_path: Path) -> None:
    out_dir = tmp_path / "chunks"

    report = chunk_body_mesh_assets(
        body_mesh_path=FIXTURES / "body_mesh_real_excerpt.json",
        frame_compute_plan_path=FIXTURES / "frame_compute_plan_real_excerpt.json",
        out_dir=out_dir,
        quantization_scale=100,
    )

    index = json.loads((out_dir / "body_mesh_index.json").read_text(encoding="utf-8"))
    faces = json.loads((out_dir / "body_mesh_faces.json").read_text(encoding="utf-8"))

    assert report["chunk_count"] == 2
    assert index["artifact_type"] == "racketsport_body_mesh_index"
    assert index["faces_url"] == "body_mesh_faces.json"
    assert index["summary"] == {
        "faces_count": 2,
        "mesh_frame_count": 2,
        "player_count": 2,
        "window_count": 2,
    }
    assert faces["mesh_faces"] == [[0, 1, 2], [0, 2, 3]]
    assert faces["faces_ref"] == "mhr_faces_static"

    first_window = index["windows"][0]
    assert first_window["player_ids"] == [7]
    assert first_window["target_player_ids"] == [7]
    assert first_window["encoding"] == "gzip_int16_world_vertices_v1"
    assert first_window["quantization"]["scale"] == 100
    assert first_window["players"][0]["id"] == 7
    assert first_window["players"][0]["frames"][0]["vertex_count"] == 4
    assert first_window["players"][0]["frames"][0]["joint_count"] == 2
    assert (out_dir / first_window["url"]).is_file()
    assert first_window["byte_size"] == (out_dir / first_window["url"]).stat().st_size

    decoded = gzip.decompress((out_dir / first_window["url"]).read_bytes())
    expected_int16_values = (4 + 2) * 3
    assert len(decoded) == expected_int16_values * 2


def test_chunk_body_mesh_assets_cli_has_direct_path_reference_and_writes_report(tmp_path: Path) -> None:
    out_dir = tmp_path / "cli_chunks"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/chunk_body_mesh_assets.py",
            "--body-mesh",
            str(FIXTURES / "body_mesh_real_excerpt.json"),
            "--frame-compute-plan",
            str(FIXTURES / "frame_compute_plan_real_excerpt.json"),
            "--out-dir",
            str(out_dir),
            "--quantization-scale",
            "100",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report["chunk_count"] == 2
    assert report["index_path"].endswith("body_mesh_index.json")
    assert (out_dir / "body_mesh_index.json").is_file()
    assert (out_dir / "body_mesh_chunk_report.json").is_file()

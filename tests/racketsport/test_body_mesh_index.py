from __future__ import annotations

import gzip
import json
import os
import struct
import subprocess
import sys
import time
from pathlib import Path

import pytest

from threed.racketsport import body_mesh_index as body_mesh_index_module
from threed.racketsport.body_mesh_index import (
    build_body_mesh_index,
    build_body_mesh_index_from_arrays,
    build_body_mesh_index_from_payload,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _body_mesh_payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_body_mesh",
        "clip": "clip_a",
        "model": "sam3dbody_world_joints",
        "fps": 30.0,
        "world_frame": "court_Z0",
        "faces_ref": "mhr_faces_static",
        "mesh_faces": [[0, 1, 2]],
        "joint_names": ["joint_0"],
        "windows": [
            {
                "source_window_index": 0,
                "frame_start": 5,
                "frame_end": 6,
                "t0": 5 / 30.0,
                "t1": 7 / 30.0,
                "frame_count": 2,
                "target_player_ids": [7],
                "target_representation": "world_mesh",
                "fallback_representation": "lane_a_skeleton",
                "reason_counts": {"contact_window": 2},
                "max_score": 0.8,
            }
        ],
        "players": [
            {
                "id": 7,
                "frames": [
                    {
                        "frame_idx": 5,
                        "t": 5 / 30.0,
                        "source_window_index": 0,
                        "blend_weight": 0.5,
                        "mesh_vertices_world": [[1.0, 2.0, 0.3], [1.1, 2.1, 0.4], [1.2, 2.2, 0.5]],
                        "joints_world": [[0.1, 0.2, 1.0]],
                        "joint_conf": [0.9],
                        "smplx_params": {},
                        "reasons": ["contact_window"],
                    },
                    {
                        "frame_idx": 6,
                        "t": 6 / 30.0,
                        "source_window_index": 0,
                        "blend_weight": 1.0,
                        "mesh_vertices_world": [[1.3, 2.3, 0.6], [1.4, 2.4, 0.7], [1.5, 2.5, 0.8]],
                        "joints_world": [[0.2, 0.3, 1.1]],
                        "joint_conf": [0.8],
                        "smplx_params": {},
                        "reasons": ["contact_window", "proximity"],
                    },
                ],
            }
        ],
        "summary": {"mesh_frame_count": 2, "player_count": 1, "contact_window_count": 1},
    }


def test_build_body_mesh_index_splits_mesh_into_viewer_schema_and_chunks(tmp_path: Path) -> None:
    body_mesh_path = tmp_path / "clip" / "body_mesh.json"
    out_dir = tmp_path / "indexed"
    _write_json(body_mesh_path, _body_mesh_payload())

    result = build_body_mesh_index(body_mesh_path, out_dir=out_dir, quantization_scale=100)

    index = json.loads((out_dir / "body_mesh_index.json").read_text(encoding="utf-8"))
    faces = json.loads((out_dir / "body_mesh_faces.json").read_text(encoding="utf-8"))
    chunk_path = out_dir / "body_mesh_chunks" / "window_000.bin.gz"
    raw_values = struct.unpack("<" + "h" * (len(gzip.decompress(chunk_path.read_bytes())) // 2), gzip.decompress(chunk_path.read_bytes()))

    assert result["index_path"] == out_dir / "body_mesh_index.json"
    assert faces == {
        "schema_version": 1,
        "artifact_type": "racketsport_body_mesh_faces",
        "faces_ref": "mhr_faces_static",
        "mesh_faces": [[0, 1, 2]],
    }
    assert index["artifact_type"] == "racketsport_body_mesh_index"
    assert index["faces_url"] == "body_mesh_faces.json"
    assert index["summary"] == {"window_count": 1, "mesh_frame_count": 2, "player_count": 1, "faces_count": 1}
    assert index["windows"][0]["url"] == "body_mesh_chunks/window_000.bin.gz"
    assert index["windows"][0]["encoding"] == "gzip_int16_delta_world_vertices_v2"
    assert index["windows"][0]["byte_size"] == chunk_path.stat().st_size
    assert index["windows"][0]["player_frame_count"] == 2
    assert index["windows"][0]["player_ids"] == [7]
    assert index["windows"][0]["players"][0]["frames"][0] == {
        "frame_idx": 5,
        "t": 5 / 30.0,
        "source_window_index": 0,
        "blend_weight": 0.5,
        "vertex_count": 3,
        "joint_count": 1,
        "joint_conf": [0.9],
        "reasons": ["contact_window"],
    }
    assert raw_values[:12] == (100, 200, 30, 110, 210, 40, 120, 220, 50, 10, 20, 100)
    assert index["windows"][0]["players"][0]["frames"][1]["delta_from_previous"] is True
    assert raw_values[12:] == (30, 30, 30, 30, 30, 30, 30, 30, 30, 10, 10, 10)


def test_build_body_mesh_index_is_deterministic(tmp_path: Path) -> None:
    body_mesh_path = tmp_path / "clip" / "body_mesh.json"
    _write_json(body_mesh_path, _body_mesh_payload())

    build_body_mesh_index(body_mesh_path, out_dir=tmp_path / "first", quantization_scale=100)
    build_body_mesh_index(body_mesh_path, out_dir=tmp_path / "second", quantization_scale=100)

    assert (tmp_path / "first" / "body_mesh_index.json").read_bytes() == (tmp_path / "second" / "body_mesh_index.json").read_bytes()
    assert (tmp_path / "first" / "body_mesh_faces.json").read_bytes() == (tmp_path / "second" / "body_mesh_faces.json").read_bytes()
    assert (tmp_path / "first" / "body_mesh_chunks" / "window_000.bin.gz").read_bytes() == (
        tmp_path / "second" / "body_mesh_chunks" / "window_000.bin.gz"
    ).read_bytes()


def test_build_body_mesh_index_orders_bytes_with_sorted_player_metadata(tmp_path: Path) -> None:
    payload = _body_mesh_payload()
    first_player = payload["players"][0]
    second_player = {
        "id": 3,
        "frames": [
            {
                **first_player["frames"][0],
                "mesh_vertices_world": [[9.0, 8.0, 7.0]],
                "joints_world": [[6.0, 5.0, 4.0]],
            }
        ],
    }
    payload["players"] = [first_player, second_player]
    # Exercise the arbitrary input-order contract: metadata sorts IDs, and the
    # binary stream must follow that same order.
    payload["players"].reverse()

    out_dir = tmp_path / "indexed"
    build_body_mesh_index_from_payload(payload, out_dir=out_dir, quantization_scale=100)
    index = json.loads((out_dir / "body_mesh_index.json").read_text(encoding="utf-8"))
    chunk_path = out_dir / index["windows"][0]["url"]
    raw_bytes = gzip.decompress(chunk_path.read_bytes())
    raw_values = struct.unpack(f"<{len(raw_bytes) // 2}h", raw_bytes)

    assert [player["id"] for player in index["windows"][0]["players"]] == [3, 7]
    # Player 3's one vertex and joint precede player 7's two frames.
    assert raw_values[:6] == (900, 800, 700, 600, 500, 400)
    assert raw_values[6:18] == (100, 200, 30, 110, 210, 40, 120, 220, 50, 10, 20, 100)


def test_build_body_mesh_index_writes_compact_json_metadata(tmp_path: Path) -> None:
    body_mesh_path = tmp_path / "clip" / "body_mesh.json"
    out_dir = tmp_path / "indexed"
    _write_json(body_mesh_path, _body_mesh_payload())

    build_body_mesh_index(body_mesh_path, out_dir=out_dir, quantization_scale=100)

    assert b"\n  " not in (out_dir / "body_mesh_index.json").read_bytes()
    assert b"\n  " not in (out_dir / "body_mesh_faces.json").read_bytes()


def test_vectorized_quantization_matches_scalar_wire_bytes() -> None:
    import numpy as np

    values = [
        [-32.767, -0.0015, -0.0005],
        [0.0005, 0.0015, 1.2344],
        [12.3456, 20.0004, 32.767],
    ]
    expected_list = body_mesh_index_module._quantized_vec3_bytes_scalar(values, name="points", scale=1000)
    actual_list = body_mesh_index_module._quantized_vec3_bytes_from_raw(values, name="points", scale=1000)
    assert actual_list == expected_list

    float32_values = np.asarray(values, dtype=np.float32)
    expected_float32 = body_mesh_index_module._quantized_vec3_bytes_scalar(
        float32_values.tolist(),
        name="points",
        scale=1000,
    )
    actual_float32 = body_mesh_index_module._quantized_vec3_bytes_from_raw(
        float32_values,
        name="points",
        scale=1000,
    )
    assert actual_float32 == expected_float32


def test_build_body_mesh_index_decompressed_chunks_are_equivalent_across_compresslevels(tmp_path: Path) -> None:
    body_mesh_path = tmp_path / "clip" / "body_mesh.json"
    _write_json(body_mesh_path, _body_mesh_payload())

    build_body_mesh_index(body_mesh_path, out_dir=tmp_path / "level9", quantization_scale=100, compresslevel=9)
    build_body_mesh_index(body_mesh_path, out_dir=tmp_path / "level1", quantization_scale=100, compresslevel=1)

    level9_index = json.loads((tmp_path / "level9" / "body_mesh_index.json").read_text(encoding="utf-8"))
    level1_index = json.loads((tmp_path / "level1" / "body_mesh_index.json").read_text(encoding="utf-8"))
    level9_index["windows"][0]["byte_size"] = 0
    level1_index["windows"][0]["byte_size"] = 0

    assert level1_index == level9_index
    assert gzip.decompress((tmp_path / "level1" / "body_mesh_chunks" / "window_000.bin.gz").read_bytes()) == gzip.decompress(
        (tmp_path / "level9" / "body_mesh_chunks" / "window_000.bin.gz").read_bytes()
    )


def test_build_body_mesh_index_uses_measured_default_compresslevel(tmp_path: Path) -> None:
    body_mesh_path = tmp_path / "clip" / "body_mesh.json"
    _write_json(body_mesh_path, _body_mesh_payload())

    result = build_body_mesh_index(body_mesh_path, out_dir=tmp_path / "default")

    assert result["gzip_compresslevel"] == 6


def test_build_body_mesh_index_from_payload_matches_file_builder(tmp_path: Path) -> None:
    payload = _body_mesh_payload()
    body_mesh_path = tmp_path / "clip" / "body_mesh.json"
    _write_json(body_mesh_path, payload)

    build_body_mesh_index(body_mesh_path, out_dir=tmp_path / "from_file", quantization_scale=100, compresslevel=6)
    build_body_mesh_index_from_payload(payload, out_dir=tmp_path / "from_memory", quantization_scale=100, compresslevel=6)

    assert (tmp_path / "from_file" / "body_mesh_index.json").read_bytes() == (
        tmp_path / "from_memory" / "body_mesh_index.json"
    ).read_bytes()
    assert (tmp_path / "from_file" / "body_mesh_faces.json").read_bytes() == (
        tmp_path / "from_memory" / "body_mesh_faces.json"
    ).read_bytes()
    assert gzip.decompress((tmp_path / "from_file" / "body_mesh_chunks" / "window_000.bin.gz").read_bytes()) == gzip.decompress(
        (tmp_path / "from_memory" / "body_mesh_chunks" / "window_000.bin.gz").read_bytes()
    )


def test_build_body_mesh_index_from_arrays_matches_file_builder(tmp_path: Path) -> None:
    import numpy as np

    payload = _body_mesh_payload()
    body_mesh_path = tmp_path / "clip" / "body_mesh.json"
    _write_json(body_mesh_path, payload)
    array_players = []
    for player in payload["players"]:  # type: ignore[index]
        frames = []
        for frame in player["frames"]:  # type: ignore[index]
            frames.append(
                {
                    **frame,
                    "mesh_vertices_world": np.asarray(frame["mesh_vertices_world"], dtype=np.float32),
                    "joints_world": np.asarray(frame["joints_world"], dtype=np.float32),
                    "joint_conf": np.asarray(frame["joint_conf"], dtype=np.float32),
                }
            )
        array_players.append({"id": player["id"], "frames": frames})  # type: ignore[index]

    build_body_mesh_index(body_mesh_path, out_dir=tmp_path / "from_file", quantization_scale=100, compresslevel=6)
    build_body_mesh_index_from_arrays(
        metadata={
            "clip": payload["clip"],
            "model": payload["model"],
            "fps": payload["fps"],
            "world_frame": payload["world_frame"],
            "faces_ref": payload["faces_ref"],
            "mesh_faces": payload["mesh_faces"],
            "windows": payload["windows"],
        },
        players=array_players,
        out_dir=tmp_path / "from_arrays",
        quantization_scale=100,
        compresslevel=6,
    )

    assert (tmp_path / "from_file" / "body_mesh_index.json").read_bytes() == (
        tmp_path / "from_arrays" / "body_mesh_index.json"
    ).read_bytes()
    assert (tmp_path / "from_file" / "body_mesh_faces.json").read_bytes() == (
        tmp_path / "from_arrays" / "body_mesh_faces.json"
    ).read_bytes()
    assert gzip.decompress((tmp_path / "from_file" / "body_mesh_chunks" / "window_000.bin.gz").read_bytes()) == gzip.decompress(
        (tmp_path / "from_arrays" / "body_mesh_chunks" / "window_000.bin.gz").read_bytes()
    )


@pytest.mark.slow
def test_benchmark_body_mesh_index_compresslevels_prints_sandbox_throughput(tmp_path: Path) -> None:
    target_mb = float(os.environ.get("BODY_MESH_INDEX_BENCH_MB", "0"))
    if target_mb <= 0:
        pytest.skip("set BODY_MESH_INDEX_BENCH_MB=50 to run the slow compresslevel benchmark")
    body_mesh_path = tmp_path / "clip" / "body_mesh.json"
    _write_synthetic_body_mesh(body_mesh_path, target_mb=target_mb)
    input_mb = body_mesh_path.stat().st_size / (1024 * 1024)
    measurements: dict[int, dict[str, float]] = {}

    for level in (9, 6, 4, 1):
        started = time.perf_counter()
        build_body_mesh_index(
            body_mesh_path,
            out_dir=tmp_path / f"level{level}",
            quantization_scale=100,
            compresslevel=level,
        )
        wall_s = time.perf_counter() - started
        chunk_bytes = sum(path.stat().st_size for path in (tmp_path / f"level{level}" / "body_mesh_chunks").glob("*.gz"))
        measurements[level] = {
            "wall_s": wall_s,
            "mb_s": input_mb / wall_s,
            "chunk_mb": chunk_bytes / (1024 * 1024),
        }

    print("body_mesh_index_compresslevel_benchmark " + json.dumps(measurements, sort_keys=True))
    assert measurements[4]["mb_s"] > 0.0


def test_build_body_mesh_index_cli_accepts_clip_dir_and_writes_index(tmp_path: Path) -> None:
    clip_dir = tmp_path / "clip"
    _write_json(clip_dir / "body_mesh.json", _body_mesh_payload())
    out_dir = tmp_path / "indexed"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/build_body_mesh_index.py",
            str(clip_dir),
            "--out-dir",
            str(out_dir),
            "--quantization-scale",
            "100",
        ],
        cwd=Path(__file__).resolve().parents[2],
        check=True,
        text=True,
        capture_output=True,
    )

    summary = json.loads(completed.stdout)
    assert summary["artifact_type"] == "racketsport_body_mesh_index_build"
    assert summary["summary"]["window_count"] == 1
    assert (out_dir / "body_mesh_index.json").is_file()


def _write_synthetic_body_mesh(path: Path, *, target_mb: float) -> None:
    vertex_count = 1200
    joint_count = 70
    vertex = [[round((idx % 37) * 0.001, 4), round((idx % 53) * 0.001, 4), round((idx % 29) * 0.001, 4)] for idx in range(vertex_count)]
    joints = [[round((idx % 17) * 0.002, 4), round((idx % 19) * 0.002, 4), round(0.5 + (idx % 23) * 0.002, 4)] for idx in range(joint_count)]
    players: list[dict[str, object]] = []
    frame_count = max(16, int(target_mb * 10))
    for player_id in range(1, 5):
        frames = [
            {
                "frame_idx": frame_idx,
                "t": frame_idx / 30.0,
                "source_window_index": frame_idx // 16,
                "blend_weight": 1.0,
                "mesh_vertices_world": vertex,
                "joints_world": joints,
                "joint_conf": [0.9] * joint_count,
                "reasons": ["benchmark"],
            }
            for frame_idx in range(frame_count)
        ]
        players.append({"id": player_id, "frames": frames})
    payload = {
        "schema_version": 1,
        "artifact_type": "racketsport_body_mesh",
        "clip": "bench",
        "model": "sam3dbody_world_joints",
        "fps": 30.0,
        "world_frame": "court_Z0",
        "faces_ref": "mhr_faces_static",
        "mesh_faces": [[0, 1, 2]],
        "windows": [
            {
                "source_window_index": index,
                "frame_start": index * 16,
                "frame_end": min(frame_count - 1, index * 16 + 15),
                "frame_count": min(16, frame_count - index * 16),
                "target_player_ids": [1, 2, 3, 4],
                "reason_counts": {"benchmark": 1},
            }
            for index in range((frame_count + 15) // 16)
        ],
        "players": players,
        "summary": {"mesh_frame_count": frame_count * 4, "player_count": 4},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8")

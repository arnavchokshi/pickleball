#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
import struct
from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping

try:
    import ijson
except ModuleNotFoundError:  # pragma: no cover - exercised in repo venvs without ijson installed.
    ijson = None


SCHEMA_VERSION = 1
INDEX_ARTIFACT_TYPE = "racketsport_body_mesh_index"
FACES_ARTIFACT_TYPE = "racketsport_body_mesh_faces"
CHUNK_ENCODING = "gzip_int16_world_vertices_v1"
DEFAULT_QUANTIZATION_SCALE = 100


def chunk_body_mesh_assets(
    *,
    body_mesh_path: str | Path,
    frame_compute_plan_path: str | Path,
    out_dir: str | Path,
    quantization_scale: int = DEFAULT_QUANTIZATION_SCALE,
) -> dict[str, Any]:
    body_mesh = Path(body_mesh_path)
    frame_compute_plan = Path(frame_compute_plan_path)
    output = Path(out_dir)
    if quantization_scale <= 0:
        raise ValueError("quantization_scale must be positive")
    if not body_mesh.is_file():
        raise FileNotFoundError(f"body_mesh file not found: {body_mesh}")
    if not frame_compute_plan.is_file():
        raise FileNotFoundError(f"frame_compute_plan file not found: {frame_compute_plan}")

    output.mkdir(parents=True, exist_ok=True)
    chunks_dir = output / "body_mesh_chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    header = _read_body_mesh_header(body_mesh)
    faces = _read_mesh_faces(body_mesh)
    plan = _read_json_mapping(frame_compute_plan)
    windows = _deep_mesh_windows(plan)

    faces_asset = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": FACES_ARTIFACT_TYPE,
        "faces_ref": header["faces_ref"],
        "mesh_faces": faces,
    }
    faces_path = output / "body_mesh_faces.json"
    _write_json(faces_path, faces_asset)

    chunk_builders = [_window_builder(window, index, quantization_scale) for index, window in enumerate(windows)]
    builders_by_source_index = {builder["source_window_index"]: builder for builder in chunk_builders}
    _encode_player_frames(body_mesh, builders_by_source_index, windows=chunk_builders, quantization_scale=quantization_scale)

    index_windows = []
    chunk_reports = []
    for builder in chunk_builders:
        source_window_index = builder["source_window_index"]
        chunk_rel_path = Path("body_mesh_chunks") / f"window_{source_window_index:03d}.bin.gz"
        chunk_path = output / chunk_rel_path
        chunk_bytes = gzip.compress(bytes(builder.pop("_buffer")), compresslevel=6)
        chunk_path.write_bytes(chunk_bytes)

        players = list(builder.pop("_players").values())
        builder.pop("_quantization_scale", None)
        player_ids = [player["id"] for player in players if player["frames"]]
        player_frame_count = sum(len(player["frames"]) for player in players)
        window_entry = {
            **builder,
            "player_ids": player_ids,
            "players": players,
            "url": chunk_rel_path.as_posix(),
            "byte_size": chunk_path.stat().st_size,
            "encoding": CHUNK_ENCODING,
            "quantization": {
                "scale": quantization_scale,
                "unit": "m",
            },
            "player_frame_count": player_frame_count,
        }
        index_windows.append(window_entry)
        chunk_reports.append(
            {
                "source_window_index": source_window_index,
                "url": chunk_rel_path.as_posix(),
                "byte_size": chunk_path.stat().st_size,
                "byte_size_mb": round(chunk_path.stat().st_size / 1024 / 1024, 4),
                "player_ids": player_ids,
                "player_frame_count": player_frame_count,
            }
        )

    mesh_frame_count = sum(window["player_frame_count"] for window in index_windows)
    player_ids_all = sorted({player_id for window in index_windows for player_id in window["player_ids"]})
    index = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": INDEX_ARTIFACT_TYPE,
        "clip": header["clip"],
        "model": header["model"],
        "fps": header["fps"],
        "world_frame": header["world_frame"],
        "faces_ref": header["faces_ref"],
        "faces_url": faces_path.name,
        "windows": index_windows,
        "summary": {
            "window_count": len(index_windows),
            "mesh_frame_count": mesh_frame_count,
            "player_count": len(player_ids_all),
            "faces_count": len(faces),
        },
    }
    index_path = output / "body_mesh_index.json"
    _write_json(index_path, index)

    report = {
        "artifact_type": "racketsport_body_mesh_chunk_report",
        "body_mesh_path": body_mesh.as_posix(),
        "frame_compute_plan_path": frame_compute_plan.as_posix(),
        "index_path": index_path.as_posix(),
        "faces_path": faces_path.as_posix(),
        "chunk_count": len(index_windows),
        "quantization_scale": quantization_scale,
        "chunks": chunk_reports,
        "summary": index["summary"],
    }
    _write_json(output / "body_mesh_chunk_report.json", report)
    return report


def _read_body_mesh_header(path: Path) -> dict[str, Any]:
    if ijson is None:
        payload = _read_json_mapping(path)
        return {key: payload[key] for key in ("clip", "model", "fps", "world_frame", "faces_ref")}
    header: dict[str, Any] = {
        "clip": None,
        "model": None,
        "fps": None,
        "world_frame": None,
        "faces_ref": None,
    }
    scalar_prefixes = set(header)
    with path.open("rb") as handle:
        for prefix, event, value in ijson.parse(handle, use_float=True):
            if prefix in scalar_prefixes and event in {"string", "number"}:
                header[prefix] = value
    missing = [key for key, value in header.items() if value is None]
    if missing:
        raise ValueError(f"body_mesh missing required fields: {', '.join(missing)}")
    return header


def _read_mesh_faces(path: Path) -> list[list[int]]:
    if ijson is None:
        payload = _read_json_mapping(path)
        faces = payload.get("mesh_faces", [])
        if not isinstance(faces, list) or not faces:
            raise ValueError(f"body_mesh has no top-level mesh_faces to share: {path}")
        return [[int(face[0]), int(face[1]), int(face[2])] for face in faces]
    faces: list[list[int]] = []
    with path.open("rb") as handle:
        for face in ijson.items(handle, "mesh_faces.item", use_float=True):
            if not isinstance(face, list) or len(face) != 3:
                raise ValueError(f"invalid mesh face in {path}: {face!r}")
            faces.append([int(face[0]), int(face[1]), int(face[2])])
    if not faces:
        raise ValueError(f"body_mesh has no top-level mesh_faces to share: {path}")
    return faces


def _read_json_mapping(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _deep_mesh_windows(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_windows = plan.get("deep_mesh_windows")
    if not isinstance(raw_windows, list):
        raise ValueError("frame_compute_plan.deep_mesh_windows must be a list")
    windows: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_windows):
        if not isinstance(raw, dict):
            raise ValueError(f"deep_mesh_windows[{index}] must be an object")
        source_window_index = int(raw.get("source_window_index", index))
        windows.append(
            {
                "source_window_index": source_window_index,
                "frame_start": int(raw["frame_start"]),
                "frame_end": int(raw["frame_end"]),
                "t0": float(raw["t0"]),
                "t1": float(raw["t1"]),
                "frame_count": int(raw.get("frame_count", max(0, int(raw["frame_end"]) - int(raw["frame_start"]) + 1))),
                "target_player_ids": [int(player_id) for player_id in raw.get("target_player_ids", [])],
                "target_representation": str(raw.get("target_representation", "world_mesh")),
                "fallback_representation": str(raw.get("fallback_representation", "lane_a_skeleton")),
                "reason_counts": {str(key): int(value) for key, value in dict(raw.get("reason_counts", {})).items()},
                "max_score": float(raw.get("max_score", 0.0)),
            }
        )
    return windows


def _window_builder(window: Mapping[str, Any], index: int, quantization_scale: int) -> dict[str, Any]:
    return {
        "source_window_index": int(window.get("source_window_index", index)),
        "frame_start": int(window["frame_start"]),
        "frame_end": int(window["frame_end"]),
        "t0": float(window["t0"]),
        "t1": float(window["t1"]),
        "frame_count": int(window["frame_count"]),
        "target_player_ids": list(window["target_player_ids"]),
        "target_representation": str(window["target_representation"]),
        "fallback_representation": str(window["fallback_representation"]),
        "reason_counts": dict(window["reason_counts"]),
        "max_score": float(window["max_score"]),
        "_buffer": bytearray(),
        "_players": OrderedDict(),
        "_quantization_scale": quantization_scale,
    }


def _encode_player_frames(
    body_mesh_path: Path,
    builders_by_source_index: dict[int, dict[str, Any]],
    *,
    windows: list[dict[str, Any]],
    quantization_scale: int,
) -> None:
    if ijson is None:
        for player in _read_json_mapping(body_mesh_path).get("players", []):
            _encode_player(player, builders_by_source_index, windows=windows, quantization_scale=quantization_scale)
        return
    with body_mesh_path.open("rb") as handle:
        for player in ijson.items(handle, "players.item", use_float=True):
            _encode_player(player, builders_by_source_index, windows=windows, quantization_scale=quantization_scale)


def _encode_player(
    player: Mapping[str, Any],
    builders_by_source_index: dict[int, dict[str, Any]],
    *,
    windows: list[dict[str, Any]],
    quantization_scale: int,
) -> None:
    player_id = _normalized_player_id(player)
    for frame in player.get("frames", []):
        source_window_index = _frame_source_window_index(frame, windows)
        if source_window_index is None:
            continue
        builder = builders_by_source_index.get(source_window_index)
        if builder is None:
            continue
        target_player_ids = set(builder["target_player_ids"])
        if target_player_ids and player_id not in target_player_ids:
            continue
        player_entry = builder["_players"].setdefault(player_id, {"id": player_id, "frames": []})
        vertices = frame.get("mesh_vertices_world", [])
        joints = frame.get("joints_world", [])
        _encode_points(builder["_buffer"], vertices, quantization_scale, field="mesh_vertices_world")
        _encode_points(builder["_buffer"], joints, quantization_scale, field="joints_world")
        player_entry["frames"].append(
            {
                "frame_idx": int(frame["frame_idx"]),
                "t": float(frame["t"]),
                "source_window_index": source_window_index,
                "blend_weight": float(frame.get("blend_weight", 1.0)),
                "vertex_count": len(vertices),
                "joint_count": len(joints),
                "joint_conf": [float(value) for value in frame.get("joint_conf", [])],
                "reasons": [str(value) for value in frame.get("reasons", [])],
            }
        )


def _normalized_player_id(player: Mapping[str, Any]) -> int:
    raw = player.get("id", player.get("player_id"))
    if raw is None:
        raise ValueError("body_mesh player is missing id/player_id")
    return int(raw)


def _frame_source_window_index(frame: Mapping[str, Any], windows: list[dict[str, Any]]) -> int | None:
    frame_idx = int(frame["frame_idx"])
    t = float(frame["t"])
    for window in windows:
        if int(window["frame_start"]) <= frame_idx <= int(window["frame_end"]):
            return int(window["source_window_index"])
        if float(window["t0"]) <= t <= float(window["t1"]):
            return int(window["source_window_index"])
    raw = frame.get("source_window_index")
    if raw is not None:
        return int(raw)
    return None


def _encode_points(buffer: bytearray, points: Any, scale: int, *, field: str) -> None:
    if not isinstance(points, list):
        raise ValueError(f"{field} must be a list")
    for point in points:
        if not isinstance(point, list) or len(point) != 3:
            raise ValueError(f"{field} entries must be [x, y, z]")
        for coordinate in point:
            quantized = int(round(float(coordinate) * scale))
            if quantized < -32768 or quantized > 32767:
                raise ValueError(f"{field} coordinate {coordinate!r} is outside int16 range at scale {scale}")
            buffer.extend(struct.pack("<h", quantized))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Chunk large body_mesh.json assets for browser replay review.")
    parser.add_argument("--body-mesh", type=Path, required=True, help="Source body_mesh.json.")
    parser.add_argument("--frame-compute-plan", type=Path, required=True, help="frame_compute_plan.json with deep_mesh_windows.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for body_mesh_index.json and chunks.")
    parser.add_argument(
        "--quantization-scale",
        type=int,
        default=DEFAULT_QUANTIZATION_SCALE,
        help="World-coordinate int16 quantization scale in units per meter; 100 means centimeter precision.",
    )
    args = parser.parse_args(argv)

    report = chunk_body_mesh_assets(
        body_mesh_path=args.body_mesh,
        frame_compute_plan_path=args.frame_compute_plan,
        out_dir=args.out_dir,
        quantization_scale=args.quantization_scale,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

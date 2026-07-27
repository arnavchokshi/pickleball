#!/usr/bin/env python3
"""Extract one short, dense player-mesh interval into a compact viewer index.

The full BODY viewer chunks intentionally contain every requested player/frame.
That is useful for replay, but wasteful for a two-person coaching comparison.
This tool decodes the immutable source chunk, retains one player's requested
interval, and writes the same versioned body-mesh index format. It never
reconstructs a mesh from joints and never emits a skeleton fallback.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.body_mesh_index import build_body_mesh_index_from_arrays  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _gzip_payload_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with gzip.open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _output_asset_provenance(path: Path) -> dict[str, Any]:
    item: dict[str, Any] = {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
    }
    if path.name.endswith(".gz"):
        item["canonical_payload_sha256"] = _gzip_payload_sha256(path)
    return item


def _read_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _decode_array(
    raw: memoryview,
    *,
    offset: int,
    point_count: int,
    scale: int,
    previous: np.ndarray | None,
    use_delta: bool,
) -> tuple[np.ndarray, int]:
    scalar_count = int(point_count) * 3
    byte_count = scalar_count * 2
    end = offset + byte_count
    if end > len(raw):
        raise ValueError("body mesh chunk ended before the declared frame data")
    encoded = np.frombuffer(raw[offset:end], dtype="<i2", count=scalar_count).astype(np.int32)
    if use_delta:
        if previous is None or previous.shape != encoded.shape:
            raise ValueError("delta frame is missing a shape-compatible previous frame")
        encoded = encoded + previous
    points = encoded.reshape((-1, 3)).astype(np.float32) / float(scale)
    return points, end


def extract_player_interval(
    index_path: Path,
    *,
    player_id: int,
    t0: float,
    t1: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    if t1 <= t0:
        raise ValueError("t1 must be greater than t0")
    index = _read_json(index_path)
    if index.get("artifact_type") != "racketsport_body_mesh_index":
        raise ValueError("source must be a racketsport_body_mesh_index artifact")
    faces_path = (index_path.parent / str(index["faces_url"])).resolve()
    faces = _read_json(faces_path)
    scale_fallback = 1000
    selected: list[dict[str, Any]] = []
    source_chunks: list[dict[str, Any]] = []

    for window in index.get("windows", []):
        if not isinstance(window, Mapping):
            continue
        if float(window.get("t1", 0.0)) < t0 or float(window.get("t0", 0.0)) > t1:
            continue
        chunk_path = (index_path.parent / str(window["url"])).resolve()
        encoded = chunk_path.read_bytes()
        encoding = str(window.get("encoding", ""))
        decoded = gzip.decompress(encoded) if encoding.startswith("gzip_") else encoded
        raw = memoryview(decoded)
        offset = 0
        scale = int(window.get("quantization", {}).get("scale", scale_fallback))
        for player in window.get("players", []):
            if not isinstance(player, Mapping):
                continue
            current_player_id = int(player["id"])
            if current_player_id != int(player_id):
                for frame in player.get("frames", []):
                    if not isinstance(frame, Mapping):
                        continue
                    offset += (int(frame.get("vertex_count", 0)) + int(frame.get("joint_count", 0))) * 3 * 2
                if offset > len(raw):
                    raise ValueError("body mesh chunk ended before the declared non-selected player data")
                continue
            previous_vertices: np.ndarray | None = None
            previous_joints: np.ndarray | None = None
            for frame in player.get("frames", []):
                if not isinstance(frame, Mapping):
                    continue
                use_delta = bool(frame.get("delta_from_previous", False))
                vertices, offset = _decode_array(
                    raw,
                    offset=offset,
                    point_count=int(frame.get("vertex_count", 0)),
                    scale=scale,
                    previous=previous_vertices,
                    use_delta=use_delta,
                )
                joints, offset = _decode_array(
                    raw,
                    offset=offset,
                    point_count=int(frame.get("joint_count", 0)),
                    scale=scale,
                    previous=previous_joints,
                    use_delta=use_delta,
                )
                previous_vertices = np.rint(vertices.reshape(-1) * scale).astype(np.int32)
                previous_joints = np.rint(joints.reshape(-1) * scale).astype(np.int32)
                time_seconds = float(frame.get("t", 0.0))
                if not (t0 <= time_seconds <= t1):
                    continue
                selected.append(
                    {
                        "frame_idx": int(frame.get("frame_idx", round(time_seconds * float(index.get("fps", 30.0))))),
                        "t": time_seconds,
                        "source_window_index": 0,
                        "blend_weight": float(frame.get("blend_weight", 1.0)),
                        "trust_badge": frame.get("trust_badge"),
                        "joint_conf": list(frame.get("joint_conf", [])),
                        "reasons": list(frame.get("reasons", [])),
                        "mesh_vertices_world": vertices,
                        "joints_world": joints,
                    }
                )
        if offset != len(raw):
            raise ValueError(f"decoded {offset} bytes but chunk contains {len(raw)} bytes: {chunk_path}")
        source_chunks.append(
            {
                "path": str(chunk_path),
                "sha256": _sha256(chunk_path),
                "encoded_bytes": len(encoded),
                "decoded_bytes": len(decoded),
            }
        )

    selected.sort(key=lambda frame: (float(frame["t"]), int(frame["frame_idx"])))
    if not selected:
        raise ValueError(f"no dense mesh frames found for player {player_id} in [{t0}, {t1}]")
    vertex_counts = {int(frame["mesh_vertices_world"].shape[0]) for frame in selected}
    if 0 in vertex_counts or len(vertex_counts) != 1:
        raise ValueError(f"selected frames do not share one nonzero dense vertex count: {sorted(vertex_counts)}")
    metadata = {
        "clip": str(index["clip"]),
        "model": str(index["model"]),
        "fps": float(index["fps"]),
        "world_frame": str(index["world_frame"]),
        "faces_ref": str(index["faces_ref"]),
        "mesh_faces": faces["mesh_faces"],
        "windows": [
            {
                "source_window_index": 0,
                "frame_start": int(selected[0]["frame_idx"]),
                "frame_end": int(selected[-1]["frame_idx"]),
                "frame_count": len({int(frame["frame_idx"]) for frame in selected}),
                "t0": float(selected[0]["t"]),
                "t1": float(selected[-1]["t"]),
                "target_player_ids": [int(player_id)],
                "player_ids": [int(player_id)],
                "target_representation": "world_mesh",
                "fallback_representation": "none",
                "reason_counts": {},
                "max_score": 0.0,
            }
        ],
    }
    provenance = {
        "artifact_type": "racketsport_body_mesh_subset_provenance",
        "source_index_path": str(index_path.resolve()),
        "source_index_sha256": _sha256(index_path),
        "source_faces_path": str(faces_path),
        "source_faces_sha256": _sha256(faces_path),
        "source_chunks": source_chunks,
        "player_id": int(player_id),
        "requested_interval": {"t0": float(t0), "t1": float(t1)},
        "selected_interval": {"t0": float(selected[0]["t"]), "t1": float(selected[-1]["t"])},
        "mesh_frame_count": len(selected),
        "vertex_count": next(iter(vertex_counts)),
        "faces_count": len(faces["mesh_faces"]),
        "skeleton_fallback": False,
    }
    return metadata, [{"id": int(player_id), "frames": selected}], provenance


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_index")
    parser.add_argument("--player-id", type=int, required=True)
    parser.add_argument("--t0", type=float, required=True)
    parser.add_argument("--t1", type=float, required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--source-world", help="Optional virtual_world.json to subset to the same player and interval.")
    parser.add_argument("--source-manifest", help="Optional replay manifest to rewrite around the compact world/index.")
    return parser


def _write_compact_world(
    source_world_path: Path,
    *,
    out_dir: Path,
    player_id: int,
    t0: float,
    t1: float,
) -> Path:
    world = dict(_read_json(source_world_path))
    source_players = world.get("players", [])
    selected_players: list[dict[str, Any]] = []
    for player in source_players if isinstance(source_players, list) else []:
        if not isinstance(player, Mapping) or int(player.get("id", -1)) != int(player_id):
            continue
        selected_player = dict(player)
        selected_player["frames"] = [
            frame
            for frame in player.get("frames", [])
            if isinstance(frame, Mapping) and t0 <= float(frame.get("t", -1.0)) <= t1
        ]
        selected_players.append(selected_player)
    if len(selected_players) != 1 or not selected_players[0]["frames"]:
        raise ValueError(f"source world has no frames for player {player_id} in [{t0}, {t1}]")
    world["players"] = selected_players
    world["ball"] = {"source": None, "frames": []}
    world["paddles"] = []
    summary = dict(world.get("summary", {}))
    frame_count = len(selected_players[0]["frames"])
    summary.update(
        {
            "player_count": 1,
            "mesh_player_count": 1,
            "mesh_player_frame_count": frame_count,
            "joint_player_frame_count": frame_count,
            "track_only_player_frame_count": 0,
            "floor_placed_player_frame_count": sum(
                1 for frame in selected_players[0]["frames"] if frame.get("floor_world_xyz") is not None
            ),
            "floor_contact_player_frame_count": sum(
                1 for frame in selected_players[0]["frames"] if frame.get("foot_contact") is not None
            ),
            "ball_frame_count": 0,
            "approx_ball_frame_count": 0,
            "paddle_player_count": 0,
            "paddle_frame_count": 0,
            "ambiguous_paddle_frame_count": 0,
            "warnings": [
                "mesh_compare_compact_world",
                "ball_paddle_and_gameplay_layers_intentionally_omitted",
            ],
        }
    )
    summary.pop("temporal_coverage", None)
    world["summary"] = summary
    output_path = out_dir / "virtual_world.json"
    output_path.write_text(json.dumps(world, separators=(",", ":"), sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def _write_compact_manifest(source_manifest_path: Path, *, out_dir: Path) -> Path:
    manifest = dict(_read_json(source_manifest_path))
    manifest["virtual_world_url"] = "virtual_world.json"
    manifest["body_mesh_index_url"] = "body_mesh_index.json"
    manifest["body_mesh_url"] = None
    manifest["mesh_status"] = "windowed_index"
    for key in (
        "auto_bounce_candidates_url",
        "ball_arc_render_url",
        "ball_arc_solved_url",
        "ball_bounce_candidates_url",
        "ball_flight_sanity_url",
        "ball_inflections_url",
        "coaching_card_facts_url",
        "contact_windows_url",
        "court_calibration_url",
        "court_evidence_url",
        "match_stats_url",
        "physics_refinement_url",
        "rally_spans_url",
        "replay_scene_url",
        "reviewed_bounces_url",
        "skeleton_evidence_url",
    ):
        if key in manifest:
            manifest[key] = None
    manifest["annotation_sources"] = []
    manifest["label_overlays"] = []
    manifest["notes"] = [
        "Local mesh-comparison bundle with one immutable dense BODY interval.",
        "No skeleton fallback, ball, paddle, court, analytics, or legacy replay layers are included.",
        "VERIFIED=0; visualization and local review only.",
    ]
    output_path = out_dir / "replay_viewer_manifest.json"
    output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source_index = Path(args.source_index).resolve()
    out_dir = Path(args.out_dir).resolve()
    metadata, players, provenance = extract_player_interval(
        source_index,
        player_id=args.player_id,
        t0=args.t0,
        t1=args.t1,
    )
    result = build_body_mesh_index_from_arrays(metadata=metadata, players=players, out_dir=out_dir)
    compact_world_path: Path | None = None
    compact_manifest_path: Path | None = None
    if bool(args.source_world) != bool(args.source_manifest):
        raise ValueError("--source-world and --source-manifest must be supplied together")
    if args.source_world and args.source_manifest:
        compact_world_path = _write_compact_world(
            Path(args.source_world).resolve(),
            out_dir=out_dir,
            player_id=args.player_id,
            t0=args.t0,
            t1=args.t1,
        )
        compact_manifest_path = _write_compact_manifest(Path(args.source_manifest).resolve(), out_dir=out_dir)
    output_assets = [Path(result["index_path"]), Path(result["faces_path"])]
    output_assets.extend(sorted((out_dir / "body_mesh_chunks").glob("*.gz")))
    if compact_world_path is not None:
        output_assets.append(compact_world_path)
    if compact_manifest_path is not None:
        output_assets.append(compact_manifest_path)
    provenance["output_assets"] = [_output_asset_provenance(path) for path in output_assets]
    provenance_path = out_dir / "subset_provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "index_path": str(result["index_path"]),
                "faces_path": str(result["faces_path"]),
                "provenance_path": str(provenance_path),
                "virtual_world_path": str(compact_world_path) if compact_world_path else None,
                "manifest_path": str(compact_manifest_path) if compact_manifest_path else None,
                "summary": result["summary"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

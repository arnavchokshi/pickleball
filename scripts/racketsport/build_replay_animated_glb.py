#!/usr/bin/env python3
"""W3-REPLAY-NATIVE phase 1: bake a `body_mesh.json` (`racketsport_body_mesh`,
see `scripts/racketsport/build_body_mesh_export.py`) into an animated,
morph-target-driven `.glb`.

This writes an *uncompressed* GLB (plain float32 accessors). Run
`npx @gltf-transform/cli optimize <out> <compressed-out> --compress meshopt`
afterwards for the meshopt+quantization compression pass the W3-REPLAY-NATIVE
gate calls for -- kept as a separate step so this script's numerically
sensitive part (delta encoding, animation timing) stays easy to test without
depending on Node/npm tooling being importable from pytest.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.replay_glb_bake import BodyMeshBakeError, build_animated_body_glb  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Bake body_mesh.json into an animated GLB.")
    parser.add_argument("--clip", required=True, help="Clip id.")
    parser.add_argument("--body-mesh", type=Path, required=True, help="body_mesh.json path (racketsport_body_mesh).")
    parser.add_argument("--out", type=Path, required=True, help="Output .glb path.")
    args = parser.parse_args(argv)

    try:
        body_mesh = json.loads(args.body_mesh.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: failed to read --body-mesh: {exc}", file=sys.stderr)
        return 1

    try:
        glb_bytes = build_animated_body_glb(body_mesh, clip=args.clip)
    except BodyMeshBakeError as exc:
        print(f"ERROR: GLB bake failed: {exc}", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(glb_bytes)

    player_summaries = [
        {
            "id": player.get("id"),
            "frame_count": len(player.get("frames", [])),
        }
        for player in body_mesh.get("players", [])
        if player.get("frames")
    ]
    print(
        json.dumps(
            {
                "schema_version": 1,
                "clip": args.clip,
                "out": str(args.out),
                "out_bytes": args.out.stat().st_size,
                "players": player_summaries,
                "mesh_faces_count": len(body_mesh.get("mesh_faces", [])),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

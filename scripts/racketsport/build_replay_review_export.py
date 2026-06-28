#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.replay_export import (  # noqa: E402
    build_replay_review_export_from_virtual_world,
    write_replay_scene,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build CPU-only review GLBs and replay_scene.json from virtual_world.json.")
    parser.add_argument("--virtual-world", type=Path, required=True, help="virtual_world.json or virtual_world_paddle_preview.json.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Directory for generated review GLB files.")
    parser.add_argument("--scene-out", type=Path, help="Output replay_scene.json path. Defaults to <out-dir>/replay_scene.json.")
    parser.add_argument("--point-id", type=int, default=1, help="Replay point id for the generated point GLB.")
    args = parser.parse_args(argv)

    try:
        virtual_world = json.loads(args.virtual_world.read_text(encoding="utf-8"))
        if not isinstance(virtual_world, dict):
            raise ValueError("virtual world payload must be a JSON object")
        scene_out = args.scene_out or args.out_dir / "replay_scene.json"
        scene = build_replay_review_export_from_virtual_world(
            virtual_world,
            export_root=args.out_dir,
            scene_root=scene_out.parent,
            point_id=args.point_id,
        )
        write_replay_scene(scene_out, scene)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: replay review export failed: {exc}", file=sys.stderr)
        return 1

    payload = scene.model_dump(mode="json")
    print(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_type": "racketsport_replay_review_export",
                "status": "review_only_not_gate_verified",
                "scene_out": str(scene_out),
                "export_root": str(args.out_dir),
                "court_glb": payload["court_glb"],
                "point_glbs": [point["glb_url"] for point in payload["points"]],
                "players": payload["players"],
                "fps": payload["fps"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

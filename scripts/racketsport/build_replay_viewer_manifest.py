#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.replay_viewer_manifest import (  # noqa: E402
    build_replay_viewer_manifest,
    write_replay_viewer_manifest,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a local browser replay-viewer manifest.")
    parser.add_argument("--clip", required=True, help="Clip id.")
    parser.add_argument("--video", type=Path, required=True, help="Source video path.")
    parser.add_argument("--virtual-world", type=Path, required=True, help="virtual_world.json path.")
    parser.add_argument("--player-labels", type=Path, help="Optional labels/players.json box overlay.")
    parser.add_argument("--replay-scene", type=Path, help="Optional replay_scene.json path.")
    parser.add_argument("--physics-refinement", type=Path, help="Optional physics_refinement.json path.")
    parser.add_argument("--contact-windows", type=Path, help="Optional contact_windows.json path.")
    parser.add_argument(
        "--annotation-source",
        type=Path,
        action="append",
        default=[],
        help="Optional extra annotation JSON such as person_ground_truth.json. Repeatable.",
    )
    parser.add_argument(
        "--vite-allow-root",
        type=Path,
        default=None,
        help="Root directory the local Vite replay server is configured to serve. Defaults to the repo root.",
    )
    parser.add_argument("--out", type=Path, required=True, help="Output replay_viewer_manifest.json path.")
    args = parser.parse_args(argv)

    try:
        manifest = build_replay_viewer_manifest(
            clip=args.clip,
            video_path=args.video,
            virtual_world_path=args.virtual_world,
            player_labels_path=args.player_labels,
            replay_scene_path=args.replay_scene,
            physics_refinement_path=args.physics_refinement,
            contact_windows_path=args.contact_windows,
            annotation_sources=args.annotation_source,
            vite_allow_root=args.vite_allow_root,
        )
        write_replay_viewer_manifest(args.out, manifest)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: replay viewer manifest failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps({"schema_version": 1, "out": str(args.out), "clip": args.clip}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

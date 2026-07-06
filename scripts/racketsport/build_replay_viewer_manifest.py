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
    parser.add_argument("--body-mesh", type=Path, help="Optional body_mesh.json path.")
    parser.add_argument("--body-mesh-index", type=Path, help="Optional chunked body_mesh_index.json path.")
    parser.add_argument("--physics-refinement", type=Path, help="Optional physics_refinement.json path.")
    parser.add_argument("--contact-windows", type=Path, help="Optional contact_windows.json path.")
    parser.add_argument("--ball-inflections", type=Path, help="Optional ball_inflections.json path for timeline bounce markers.")
    parser.add_argument("--ball-arc-render", type=Path, help="Optional ball_arc_render.json path for dense parametric trail rendering.")
    parser.add_argument("--reviewed-bounces", type=Path, help="Optional reviewed_ball_bounces.json path.")
    parser.add_argument("--coaching-card-facts", type=Path, help="Optional coaching_card_facts.json path.")
    parser.add_argument("--rally-spans", type=Path, help="Optional rally_spans.json path.")
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
            body_mesh_path=args.body_mesh,
            body_mesh_index_path=args.body_mesh_index,
            physics_refinement_path=args.physics_refinement,
            contact_windows_path=args.contact_windows,
            ball_inflections_path=args.ball_inflections,
            ball_arc_render_path=args.ball_arc_render,
            reviewed_bounces_path=args.reviewed_bounces,
            coaching_card_facts_path=args.coaching_card_facts,
            rally_spans_path=args.rally_spans,
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

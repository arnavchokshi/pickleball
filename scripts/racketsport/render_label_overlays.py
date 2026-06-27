#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.label_overlay import render_label_overlays, render_prototype_gate


def main() -> int:
    parser = argparse.ArgumentParser(description="Render qualitative overlays for prototype draft labels.")
    parser.add_argument("--video", type=Path)
    parser.add_argument("--draft-label-dir", type=Path)
    parser.add_argument("--out-root", type=Path)
    parser.add_argument("--clip-name")
    parser.add_argument("--root", type=Path, help="Repo/workspace root for prototype-gate defaults.")
    parser.add_argument("--clip", action="append", dest="clips")
    parser.add_argument("--markdown", action="store_true")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--frame-pack-only", action="store_true", help="Render sampled label frames directly instead of decoding source video.")
    args = parser.parse_args()

    if args.video or args.draft_label_dir:
        if not (args.video and args.draft_label_dir and args.out_root):
            parser.error("--video, --draft-label-dir, and --out-root must be provided together")
        summary = render_label_overlays(
            video_path=args.video,
            draft_label_dir=args.draft_label_dir,
            output_root=args.out_root,
            clip_name=args.clip_name,
            write_index=True,
            write_markdown=args.markdown,
            max_frames=args.max_frames,
            frame_pack_only=args.frame_pack_only,
        )
    else:
        root = args.root or Path(".")
        summary = render_prototype_gate(
            root=root,
            clips=args.clips,
            write_markdown=args.markdown,
            max_frames=args.max_frames,
            frame_pack_only=args.frame_pack_only,
        )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

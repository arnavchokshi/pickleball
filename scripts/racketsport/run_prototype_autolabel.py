#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.autolabel import PROTOTYPE_GATE_CLIPS, bootstrap_prototype_gate, h100_defaults


def main() -> int:
    parser = argparse.ArgumentParser(description="Bootstrap draft labels for the 5-clip prototype gate.")
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--frames-root", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--teacher-root", type=Path, default=None)
    parser.add_argument("--clip", action="append", dest="clips", help="Clip name to include. Repeatable.")
    parser.add_argument("--output-space", choices=["eval0", "label_drafts"], default="eval0")
    parser.add_argument("--h100-defaults", action="store_true", help="Use /workspace/pickleball defaults explicitly.")
    args = parser.parse_args()

    defaults = h100_defaults(output_space=args.output_space)
    root = args.root or defaults["root"]
    frames_root = args.frames_root or defaults["frames_root"]
    out = args.out or defaults["out"]
    clip_names = args.clips or (defaults["clip_names"] if args.h100_defaults else None)
    if clip_names is None and args.root is None:
        clip_names = list(PROTOTYPE_GATE_CLIPS)

    summary = bootstrap_prototype_gate(
        root=root,
        out=out,
        frames_root=frames_root,
        teacher_root=args.teacher_root,
        clip_names=clip_names,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

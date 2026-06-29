#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.label_review import export_review_bundle


def main() -> int:
    parser = argparse.ArgumentParser(description="Export uncertain prototype-label frames for human review.")
    parser.add_argument("--drafts-root", type=Path, default=Path("runs/eval0/prototype_gate"))
    parser.add_argument("--frames-root", type=Path, default=Path("runs/label_frames"))
    parser.add_argument("--out", type=Path, default=Path("runs/eval0/prototype_gate/review_bundle"))
    parser.add_argument("--confidence-threshold", type=float, default=0.7)
    args = parser.parse_args()
    summary = export_review_bundle(
        drafts_root=args.drafts_root,
        frames_root=args.frames_root,
        out=args.out,
        confidence_threshold=args.confidence_threshold,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] == "ready_for_human_review" else 2


if __name__ == "__main__":
    raise SystemExit(main())

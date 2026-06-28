#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.detection_bridge import fps_from_manifest  # noqa: E402
from threed.racketsport.racket_candidates import (  # noqa: E402
    load_json_object,
    racket_labels_to_candidates,
    write_racket_candidates,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert draft racket label boxes into strict racket_candidates.json for prototype PnP review."
    )
    parser.add_argument("--racket-labels", type=Path, required=True, help="Input labels/racket_pose.json draft artifact.")
    parser.add_argument("--out", type=Path, required=True, help="Output racket_candidates.json path.")
    parser.add_argument("--manifest", type=Path, default=None, help="Optional prototype_autolabel_manifest.json for fps.")
    parser.add_argument("--fps", type=float, default=None, help="Explicit FPS override.")
    parser.add_argument("--min-confidence", type=float, default=0.0, help="Minimum draft racket-box confidence to keep.")
    parser.add_argument("--exclude-uncertain", action="store_true", help="Skip uncertain draft boxes.")
    args = parser.parse_args()

    fps = args.fps or fps_from_manifest(args.manifest)
    if fps is None:
        print("fps is required via --fps or --manifest", file=sys.stderr)
        return 2

    try:
        labels = load_json_object(args.racket_labels)
        candidates, counts = racket_labels_to_candidates(
            labels,
            fps=float(fps),
            min_confidence=args.min_confidence,
            include_uncertain=not args.exclude_uncertain,
        )
        write_racket_candidates(args.out, candidates)
    except Exception as exc:
        print(f"racket-candidate conversion failed: {exc}", file=sys.stderr)
        return 1

    print(
        "racket-candidate conversion: "
        f"accepted={counts['accepted']} "
        f"skipped_status={counts['skipped_status']} "
        f"skipped_confidence={counts['skipped_confidence']} "
        f"skipped_invalid={counts['skipped_invalid']}",
        file=sys.stderr,
    )
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

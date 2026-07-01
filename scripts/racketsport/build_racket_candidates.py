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
    cvat_paddle_boxes_to_candidates,
    load_json_object,
    racket_labels_to_candidates,
    write_racket_candidates,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert draft racket label boxes into strict racket_candidates.json for prototype PnP review."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--racket-labels", type=Path, help="Input labels/racket_pose.json draft artifact.")
    source.add_argument("--cvat-reviewed-boxes", type=Path, help="Input CVAT reviewed_boxes.json artifact.")
    parser.add_argument("--out", type=Path, required=True, help="Output racket_candidates.json path.")
    parser.add_argument("--manifest", type=Path, default=None, help="Optional prototype_autolabel_manifest.json for fps.")
    parser.add_argument("--fps", type=float, default=None, help="Explicit FPS override.")
    parser.add_argument("--min-confidence", type=float, default=0.0, help="Minimum draft racket-box confidence to keep.")
    parser.add_argument("--exclude-uncertain", action="store_true", help="Skip uncertain draft boxes.")
    parser.add_argument("--cvat-label-name", default="paddle", help="CVAT label name to convert when --cvat-reviewed-boxes is used.")
    args = parser.parse_args()

    fps = args.fps or fps_from_manifest(args.manifest)
    if fps is None:
        print("fps is required via --fps or --manifest", file=sys.stderr)
        return 2

    try:
        if args.cvat_reviewed_boxes is not None:
            labels = load_json_object(args.cvat_reviewed_boxes)
            candidates, counts = cvat_paddle_boxes_to_candidates(
                labels,
                fps=float(fps),
                label_name=args.cvat_label_name,
            )
        else:
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

    count_summary = " ".join(f"{key}={value}" for key, value in sorted(counts.items()))
    print(f"racket-candidate conversion: {count_summary}", file=sys.stderr)
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

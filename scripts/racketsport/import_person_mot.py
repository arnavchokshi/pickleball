#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.person_mot import import_mot_zip, write_person_ground_truth  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Import a CVAT MOT 1.1 ZIP into person_ground_truth.json.")
    parser.add_argument("--mot-zip", type=Path, required=True, help="Input CVAT MOT 1.1 ZIP.")
    parser.add_argument("--out", type=Path, required=True, help="Output person_ground_truth.json path.")
    parser.add_argument("--clip-id", required=True, help="Stable clip id to write into the ground-truth artifact.")
    parser.add_argument("--fps", type=float, default=None, help="Optional source video FPS.")
    args = parser.parse_args()

    try:
        ground_truth = import_mot_zip(args.mot_zip, clip_id=args.clip_id, fps=args.fps)
        write_person_ground_truth(args.out, ground_truth)
    except Exception as exc:
        print(f"person MOT import failed: {exc}", file=sys.stderr)
        return 1

    print(
        "person MOT import: "
        f"clip_id={ground_truth.clip_id} "
        f"frames={ground_truth.summary.frame_count} "
        f"valid_labels={ground_truth.summary.valid_label_count} "
        f"ignored_labels={ground_truth.summary.ignored_label_count}",
        file=sys.stderr,
    )
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

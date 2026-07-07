#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.roboflow_corpus import (
    DEFAULT_DEDUP_THRESHOLD,
    DEFAULT_EVAL_SAMPLE_EVERY_S,
    aggregate_roboflow_corpus,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build the index-based Roboflow public pretrain corpus without copying source images."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/roboflow_universe_20260706/manifest.json"),
        help="Roboflow universe manifest with downloaded COCO source entries.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/roboflow_universe_20260706/aggregated"),
        help="Directory for corpus index, per-dataset indexes, subset indexes, and corpus card.",
    )
    parser.add_argument(
        "--lane-dir",
        type=Path,
        default=Path("runs/lanes/p10_roboflow_aggregate_20260706"),
        help="Lane artifact directory for summary/report sidecars.",
    )
    parser.add_argument(
        "--eval-root",
        type=Path,
        default=Path("eval_clips/ball"),
        help="Protected eval clip root; only source.mp4 dHashes are sampled, labels are never read.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("."),
        help="Repository root for resolving relative source paths.",
    )
    parser.add_argument(
        "--dedup-threshold",
        type=int,
        default=DEFAULT_DEDUP_THRESHOLD,
        help="Maximum dHash Hamming distance considered a collision or duplicate.",
    )
    parser.add_argument(
        "--eval-sample-every-s",
        type=float,
        default=DEFAULT_EVAL_SAMPLE_EVERY_S,
        help="Protected eval video dHash sampling cadence in seconds.",
    )
    args = parser.parse_args()

    result = aggregate_roboflow_corpus(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        lane_dir=args.lane_dir,
        eval_root=args.eval_root,
        repo_root=args.repo_root,
        dedup_threshold=args.dedup_threshold,
        eval_sample_every_s=args.eval_sample_every_s,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.strict_placement_rollup import (  # noqa: E402
    StrictPlacementRollupConfig,
    build_strict_placement_rollup,
    write_strict_placement_rollup,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a fail-closed visual-placement preview rollup from small process_video artifacts."
    )
    parser.add_argument("clip_dir", type=Path, help="process_video clip directory to inspect.")
    parser.add_argument("--out-dir", type=Path, help="Output directory. Defaults to clip_dir.")
    parser.add_argument("--membership", type=Path, help="Optional target-court membership JSON.")
    parser.add_argument("--gate-table", type=Path, help="Optional gate_table.json with rows containing gate/status.")
    parser.add_argument("--side-match-threshold", type=float, default=0.90)
    parser.add_argument("--net-crossing-limit", type=int, default=0)
    parser.add_argument("--same-quadrant-max-fraction", type=float, default=0.60)
    parser.add_argument("--min-pairwise-p10-m", type=float, default=0.50)
    parser.add_argument("--membership-coverage-threshold", type=float, default=0.80)
    args = parser.parse_args(argv)

    config = StrictPlacementRollupConfig(
        side_match_threshold=args.side_match_threshold,
        net_crossing_limit=args.net_crossing_limit,
        same_quadrant_max_fraction=args.same_quadrant_max_fraction,
        min_pairwise_distance_p10_m=args.min_pairwise_p10_m,
        membership_coverage_threshold=args.membership_coverage_threshold,
    )
    out_dir = args.out_dir or args.clip_dir
    try:
        report = build_strict_placement_rollup(
            args.clip_dir,
            membership_path=args.membership,
            gate_table_path=args.gate_table,
            config=config,
        )
        write_strict_placement_rollup(report, out_dir)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"{parser.prog}: error: {exc}\n")

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

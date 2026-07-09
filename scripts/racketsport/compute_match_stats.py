#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.match_stats import compute_match_stats_for_run_dir, write_match_stats_json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compute BODY+COURT-only post-hoc match_stats.json from a banked run directory."
    )
    parser.add_argument("--run-dir", type=Path, required=True, help="Existing pipeline run directory to consume.")
    parser.add_argument("--out-json", type=Path, required=True, help="Output match_stats.json path.")
    args = parser.parse_args(argv)

    try:
        payload = compute_match_stats_for_run_dir(args.run_dir)
        write_match_stats_json(payload, args.out_json)
    except (OSError, ValueError) as exc:
        print(f"ERROR: match stats failed: {exc}", file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "out_json": str(args.out_json),
                "player_count": payload["player_count"],
                "world_jump_count": payload["summary"]["world_jump_count"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

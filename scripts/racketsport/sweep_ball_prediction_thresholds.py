#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.ball_threshold_sweep import sweep_prediction_thresholds  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Sweep ball prediction visibility thresholds and benchmark outputs.")
    parser.add_argument("--family", choices=("totnet", "pbmat"), required=True)
    parser.add_argument("--candidate-prefix", required=True)
    parser.add_argument("--review-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument(
        "--prediction",
        action="append",
        default=[],
        help="Clip prediction spec in the form clip_id=/path/to/predictions.json. May be repeated.",
    )
    parser.add_argument("--threshold", action="append", type=float, default=[])
    parser.add_argument("--category", default="threshold_sweep")
    parser.add_argument("--hit-radius-px", type=float, default=36.0)
    parser.add_argument("--teleport-px-per-frame", type=float, default=160.0)
    parser.add_argument("--max-jump-gap-frames", type=int, default=3)
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--out-md", type=Path, default=None)
    args = parser.parse_args()

    try:
        predictions_by_clip = _parse_predictions(args.prediction)
        summary = sweep_prediction_thresholds(
            predictions_by_clip=predictions_by_clip,
            review_root=args.review_root,
            out_root=args.out_root,
            family=args.family,
            candidate_name_prefix=args.candidate_prefix,
            thresholds=args.threshold,
            category=args.category,
            hit_radius_px=args.hit_radius_px,
            teleport_px_per_frame=args.teleport_px_per_frame,
            max_jump_gap_frames=args.max_jump_gap_frames,
            out_json=args.out_json,
            out_markdown=args.out_md,
        )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(
        json.dumps(
            {
                "out_root": str(args.out_root),
                "best_candidate": summary["best_candidate"],
                "best_threshold": summary["best_threshold"],
                "aggregate": summary["benchmark"]["aggregate"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _parse_predictions(specs: list[str]) -> dict[str, Path]:
    if not specs:
        raise ValueError("at least one --prediction is required")
    result: dict[str, Path] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"prediction spec must contain '=': {spec}")
        clip, path = spec.split("=", 1)
        if not clip or not path:
            raise ValueError(f"prediction spec missing clip or path: {spec}")
        result[clip] = Path(path)
    return result


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.ball_benchmark import BallCandidate, write_ball_tracker_benchmark  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark ball-track candidates against sparse click labels.")
    parser.add_argument("--run-root", type=Path, required=True, help="Prototype run root containing clip directories.")
    parser.add_argument("--review-root", type=Path, required=True, help="Root containing <clip>/ball_points.json files.")
    parser.add_argument("--clip", action="append", default=[], help="Clip id to include. May be repeated.")
    parser.add_argument(
        "--candidate",
        action="append",
        default=[],
        help="Candidate spec name=relative/path.json or name:category=relative/path.json under each clip.",
    )
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, default=None)
    parser.add_argument("--hit-radius-px", type=float, default=36.0)
    parser.add_argument("--teleport-px-per-frame", type=float, default=160.0)
    parser.add_argument("--max-jump-gap-frames", type=int, default=3)
    args = parser.parse_args()

    try:
        if not args.clip:
            raise ValueError("at least one --clip is required")
        if not args.candidate:
            raise ValueError("at least one --candidate is required")
        candidates = _expand_candidates(args.run_root, args.clip, args.candidate)
        summary = write_ball_tracker_benchmark(
            candidates=candidates,
            review_root=args.review_root,
            out_json=args.out_json,
            out_markdown=args.out_md,
            hit_radius_px=args.hit_radius_px,
            teleport_px_per_frame=args.teleport_px_per_frame,
            max_jump_gap_frames=args.max_jump_gap_frames,
        )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(json.dumps(summary["aggregate"], indent=2, sort_keys=True))
    return 0


def _expand_candidates(run_root: Path, clips: list[str], specs: list[str]) -> list[BallCandidate]:
    candidates: list[BallCandidate] = []
    for clip in clips:
        for spec in specs:
            name, category, rel_path = _parse_candidate_spec(spec)
            path = run_root / clip / rel_path
            if not path.is_file():
                raise ValueError(f"missing candidate for {clip} {name}: {path}")
            candidates.append(BallCandidate(clip=clip, name=name, category=category, path=path))
    return candidates


def _parse_candidate_spec(spec: str) -> tuple[str, str, Path]:
    if "=" not in spec:
        raise ValueError(f"candidate spec must contain '=': {spec}")
    left, right = spec.split("=", 1)
    if not left or not right:
        raise ValueError(f"candidate spec missing name or path: {spec}")
    if ":" in left:
        name, category = left.split(":", 1)
    else:
        name = left
        category = "generalizable"
    if not name or not category:
        raise ValueError(f"candidate spec missing name or category: {spec}")
    return name, category, Path(right)


if __name__ == "__main__":
    raise SystemExit(main())

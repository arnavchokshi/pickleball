#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.ball_cvat_benchmark import (  # noqa: E402
    DEFAULT_MAX_CUE_DELTA_FRAMES,
    CvatBallCandidate,
    write_cvat_ball_tracker_benchmark,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark existing ball tracks against reviewed CVAT ball labels.")
    parser.add_argument("--run-root", type=Path, required=True, help="Prototype run root containing clip directories.")
    parser.add_argument("--cvat-root", type=Path, required=True, help="Root containing <clip>/reviewed_boxes.json files.")
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
    parser.add_argument("--f1-radius-px", type=float, default=20.0)
    parser.add_argument("--teleport-px-per-frame", type=float, default=160.0)
    parser.add_argument("--max-jump-gap-frames", type=int, default=3)
    parser.add_argument("--review-input", type=Path, default=None, help="Optional saved review input JSON for contact cues.")
    parser.add_argument(
        "--cue-root",
        type=Path,
        default=None,
        help="Optional root containing <clip>/ball_inflections.json. Defaults to --run-root when --review-input is set.",
    )
    parser.add_argument("--contact-fps", type=float, default=60.0, help="Frame rate used for contact cue timing deltas.")
    parser.add_argument("--max-cue-delta-frames", type=float, default=DEFAULT_MAX_CUE_DELTA_FRAMES)
    args = parser.parse_args()

    try:
        if not args.clip:
            raise ValueError("at least one --clip is required")
        if not args.candidate:
            raise ValueError("at least one --candidate is required")
        candidates = _expand_candidates(args.run_root, args.clip, args.candidate)
        cue_root = args.cue_root
        if args.review_input is not None and cue_root is None:
            cue_root = args.run_root
        summary = write_cvat_ball_tracker_benchmark(
            candidates=candidates,
            cvat_root=args.cvat_root,
            out_json=args.out_json,
            out_markdown=args.out_md,
            hit_radius_px=args.hit_radius_px,
            f1_radius_px=args.f1_radius_px,
            teleport_px_per_frame=args.teleport_px_per_frame,
            max_jump_gap_frames=args.max_jump_gap_frames,
            review_input_path=args.review_input,
            cue_root=cue_root,
            contact_fps=args.contact_fps,
            max_cue_delta_frames=args.max_cue_delta_frames,
        )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(json.dumps(summary["aggregate"], indent=2, sort_keys=True))
    if summary.get("contact_cue_coverage") is not None:
        print(json.dumps({"contact_cue_coverage": summary["contact_cue_coverage"]["summary"]}, indent=2, sort_keys=True))
    return 0


def _expand_candidates(run_root: Path, clips: list[str], specs: list[str]) -> list[CvatBallCandidate]:
    candidates: list[CvatBallCandidate] = []
    for clip in clips:
        for spec in specs:
            name, category, rel_path = _parse_candidate_spec(spec)
            path = run_root / clip / rel_path
            if not path.is_file():
                raise ValueError(f"missing candidate for {clip} {name}: {path}")
            candidates.append(CvatBallCandidate(clip=clip, name=name, category=category, path=path))
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

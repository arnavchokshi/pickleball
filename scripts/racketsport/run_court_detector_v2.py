#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.court_detector_v2 import detect_court_v2_from_frame  # noqa: E402
from threed.racketsport.net_anchor_court import load_player_suppressed_frame  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run court detector v2 and write proposal artifacts.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--clip-id", default="")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=72)
    parser.add_argument("--stride", type=int, default=6)
    parser.add_argument("--start-frame", type=int, default=0)
    args = parser.parse_args(argv)

    try:
        if not args.input.exists():
            raise ValueError(f"input does not exist: {args.input}")
        frame, frame_meta = load_player_suppressed_frame(
            args.input,
            max_frames=args.max_frames,
            stride=args.stride,
            start_frame=args.start_frame,
        )
        artifact = detect_court_v2_from_frame(
            frame,
            clip_id=args.clip_id,
            source_frame=str(frame_meta.get("source_frame") or args.input.name),
        )
        args.out_dir.mkdir(parents=True, exist_ok=True)
        proposal_path = args.out_dir / "court_detector_v2_proposals.json"
        proposal_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(
            json.dumps(
                {
                    "proposal_path": str(proposal_path),
                    "promoted": artifact["promoted"],
                    "promotion_status": artifact["promotion_status"],
                    "promotion_blockers": artifact["promotion_blockers"],
                    "needs_user_input": artifact["needs_user_input"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: court detector v2 failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

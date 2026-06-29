#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.ball_line_calls import classify_ball_line_calls


def main() -> int:
    parser = argparse.ArgumentParser(description="Classify ball bounces against court and NVZ lines.")
    parser.add_argument("--ball-track", type=Path, required=True, help="Input ball_track.json with bounces.")
    parser.add_argument("--sport", choices=("pickleball", "tennis"), default="pickleball")
    parser.add_argument("--uncertainty-radius-m", type=float, default=0.05)
    parser.add_argument("--out", type=Path, required=True, help="Output ball_line_calls.json.")
    args = parser.parse_args()

    try:
        ball_track = json.loads(args.ball_track.read_text(encoding="utf-8"))
        payload = classify_ball_line_calls(
            ball_track,
            sport=args.sport,
            uncertainty_radius_m=args.uncertainty_radius_m,
            input_ball_track=str(args.ball_track),
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

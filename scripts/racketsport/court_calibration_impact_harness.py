#!/usr/bin/env python3
"""Score a court-calibration candidate by downstream product deltas.

README-style usage for a CAL gate shot:

```bash
.venv/bin/python scripts/racketsport/court_calibration_impact_harness.py \
  --baseline-calibration runs/current_clip/court_calibration.json \
  --candidate-calibration runs/candidate_clip/court_calibration.json \
  --tracks runs/current_clip/tracks.json \
  --placement runs/current_clip/placement.json \
  --body-grounding-quality runs/current_clip/body_grounding_quality.json \
  --ball-track runs/current_clip/ball_track.json \
  --ball-track-arc-solved runs/current_clip/ball_track_arc_solved.json \
  --video eval_clips/ball/<clip>/source.mp4 \
  --clip <clip> \
  --out runs/lanes/calv1_impact_20260708/<clip>_impact_report.json
```

The JSON report is advisory only. A CAL promotion still needs the ledger row and
manager go; this CLI never auto-promotes from downstream deltas.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.court_cal_impact import (  # noqa: E402
    DEFAULT_OUT_DIR,
    build_impact_report,
    write_impact_report,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare two court_calibration.json artifacts by cheap downstream placement/BALL deltas."
    )
    parser.add_argument("--baseline-calibration", type=Path, required=True, help="Baseline court_calibration.json.")
    parser.add_argument("--candidate-calibration", type=Path, required=True, help="Candidate court_calibration.json.")
    parser.add_argument("--tracks", type=Path, required=True, help="tracks.json from the same clip.")
    parser.add_argument("--placement", type=Path, help="Optional placement.json used for placement residual metrics.")
    parser.add_argument("--body-grounding-quality", type=Path, help="Optional body_grounding_quality.json for deferred grounding keys.")
    parser.add_argument("--ball-track", type=Path, help="Optional ball_track.json for court-plane approximation deltas.")
    parser.add_argument("--ball-track-arc-solved", type=Path, help="Optional ball_track_arc_solved.json for deferred BALL-3D context.")
    parser.add_argument("--video", type=Path, help="Optional source video path used only to print exact deferred rerun commands.")
    parser.add_argument("--clip", help="Stable clip id.")
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT_DIR / "court_calibration_impact_report.json",
        help="Output JSON path.",
    )
    args = parser.parse_args(argv)

    try:
        report = build_impact_report(
            baseline_calibration_path=args.baseline_calibration,
            candidate_calibration_path=args.candidate_calibration,
            tracks_path=args.tracks,
            placement_path=args.placement,
            body_grounding_quality_path=args.body_grounding_quality,
            ball_track_path=args.ball_track,
            ball_track_arc_solved_path=args.ball_track_arc_solved,
            video_path=args.video,
            clip=args.clip,
            out_dir=args.out.parent,
        )
        write_impact_report(report, args.out)
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        parser.exit(2, f"{parser.prog}: error: {exc}\n")

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

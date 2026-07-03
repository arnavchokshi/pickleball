#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.overlapping_court_calibration import (  # noqa: E402
    render_metric_plane_outlier_review_packet,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Render diagnostic visual crops for metric-plane court-calibration outliers."
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--eval-root", type=Path, default=Path("eval_clips/ball"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--crop-radius-px", type=int, default=96)
    parser.add_argument("--max-candidates", type=int, default=None)
    args = parser.parse_args(argv)

    try:
        packet = render_metric_plane_outlier_review_packet(
            report_path=args.report,
            eval_root=args.eval_root,
            out_dir=args.out_dir,
            crop_radius_px=args.crop_radius_px,
            max_candidates=args.max_candidates,
        )
        print(json.dumps(packet, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print(f"ERROR: metric-plane outlier review packet failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

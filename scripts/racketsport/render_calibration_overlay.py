#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.calibration_overlay import (  # noqa: E402
    load_calibration_artifact,
    load_net_plane_artifact,
    write_overlay_artifacts,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Render a CPU-safe calibration overlay SVG from court calibration artifacts.")
    parser.add_argument("--calibration", type=Path, required=True, help="Path to court_calibration.json.")
    parser.add_argument("--net-plane", type=Path, default=None, help="Optional path to net_plane.json; defaults to regulation net for calibration sport.")
    parser.add_argument("--out", type=Path, required=True, help="Output SVG path.")
    parser.add_argument("--summary-out", type=Path, default=None, help="Optional JSON path for projected overlay points and summary.")
    args = parser.parse_args()

    try:
        calibration = load_calibration_artifact(args.calibration)
        net_plane = load_net_plane_artifact(args.net_plane) if args.net_plane is not None else None
        write_overlay_artifacts(args.out, calibration, net_plane=net_plane, summary_out=args.summary_out)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(f"wrote {args.out}")
    if args.summary_out is not None:
        print(f"wrote {args.summary_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

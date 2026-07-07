#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.ball_sst_dataset import build_sst_disagreement_queue  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export ranked teacher/student BALL SST disagreement frames for CVAT review.",
    )
    parser.add_argument("--teacher-predictions", type=Path, required=True, help="Teacher sidecar root or prediction JSON.")
    parser.add_argument("--student-predictions", type=Path, required=True, help="Student sidecar root or prediction JSON.")
    parser.add_argument("--out", type=Path, required=True, help="Output disagreement queue JSON.")
    parser.add_argument("--large-offset-px", type=float, default=25.0)
    args = parser.parse_args(argv)

    try:
        queue = build_sst_disagreement_queue(
            teacher_predictions=args.teacher_predictions,
            student_predictions=args.student_predictions,
            out_path=args.out,
            large_offset_px=args.large_offset_px,
        )
    except Exception as exc:
        print(f"SST disagreement export failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"out": str(args.out), "summary": queue["summary"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

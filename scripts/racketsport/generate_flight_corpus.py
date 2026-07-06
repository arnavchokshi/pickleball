#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.flight_simulator import (
    DEFAULT_CALIBRATION_PATH,
    generate_corpus,
    load_court_calibration,
)


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    calibration = load_court_calibration(args.calibration)
    records, report = generate_corpus(
        count=args.count,
        seed=args.seed,
        calibration=calibration,
        roundtrip_samples=args.roundtrip_samples,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_type": "racketsport_flight_corpus_generation",
                "count": args.count,
                "seed": args.seed,
                "jsonl": str(args.out),
                "report": str(args.report) if args.report is not None else None,
                "acceptance": report["acceptance"],
                "round_trip": report["round_trip"]["position_error_m"],
                "performance": report["performance"],
            },
            sort_keys=True,
        )
    )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate synthetic pickleball 2D<->3D flight pairs as JSONL.")
    parser.add_argument("--count", type=int, required=True, help="Number of trajectories to emit.")
    parser.add_argument("--seed", type=int, required=True, help="Deterministic numpy RNG seed.")
    parser.add_argument(
        "--calibration",
        type=Path,
        default=DEFAULT_CALIBRATION_PATH,
        help="court_calibration_metric15pt.json used for camera projection.",
    )
    parser.add_argument("--out", type=Path, required=True, help="Output JSONL path.")
    parser.add_argument("--report", type=Path, help="Optional aggregate report JSON path.")
    parser.add_argument(
        "--roundtrip-samples",
        type=int,
        default=10,
        help="Number of clean trajectories to fit with ball_arc_solver for the round-trip report.",
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

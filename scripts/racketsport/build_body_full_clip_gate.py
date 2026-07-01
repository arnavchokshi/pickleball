#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.body_full_clip_gate import (  # noqa: E402
    DEFAULT_MIN_COVERAGE,
    build_body_full_clip_gate_from_paths,
    write_body_full_clip_gate,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a fail-closed BODY full-clip coverage gate.")
    parser.add_argument("--clip", required=True, help="Clip id.")
    parser.add_argument("--tracks", type=Path, help="tracks.json path.")
    parser.add_argument("--body-compute-execution", type=Path, help="body_compute_execution.json path.")
    parser.add_argument("--body-joint-quality", type=Path, help="body_joint_quality.json path.")
    parser.add_argument("--contact-splice", type=Path, help="contact_splice.json path.")
    parser.add_argument("--runtime-timing", type=Path, help="runtime timing sidecar path.")
    parser.add_argument("--out", type=Path, required=True, help="Output body_full_clip_gate.json path.")
    parser.add_argument("--min-coverage", type=float, default=DEFAULT_MIN_COVERAGE)
    args = parser.parse_args(argv)

    try:
        payload = build_body_full_clip_gate_from_paths(
            clip=args.clip,
            tracks_path=args.tracks,
            body_compute_execution_path=args.body_compute_execution,
            body_joint_quality_path=args.body_joint_quality,
            contact_splice_path=args.contact_splice,
            runtime_timing_path=args.runtime_timing,
            min_coverage=args.min_coverage,
        )
        write_body_full_clip_gate(args.out, payload)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: BODY full-clip gate failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Apply default-OFF rally-sequence DP to a saved event-head score artifact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.event_head.sequence_dp import (  # noqa: E402
    SequenceDPError,
    apply_event_sequence_dp,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Saved logits/probabilities JSON")
    parser.add_argument("--out", type=Path, required=True, help="Output JSON")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--enable", action="store_true",
        help="Explicitly enable the unverified DP (disabled when omitted)",
    )
    mode.add_argument(
        "--off", action="store_true",
        help="Explicit identity mode; equivalent to omitting --enable",
    )
    args = parser.parse_args()
    if args.input.resolve() == args.out.resolve():
        parser.error("--out must differ from --input so raw saved scores remain immutable")
    if args.out.suffix != ".json":
        parser.error("--out must be a .json artifact")
    try:
        raw_bytes = args.input.read_bytes()
        if not args.enable:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_bytes(raw_bytes)
            result = {"enabled": False, "identity": True, "out": str(args.out), "verified": False}
        else:
            payload = json.loads(raw_bytes)
            output = apply_event_sequence_dp(payload, enabled=True)
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
            statuses = [
                rally["status"]
                for clip in output["clips"]
                for rally in clip["sequence_dp_rallies"]
            ]
            result = {
                "enabled": True,
                "out": str(args.out),
                "rallies": len(statuses),
                "applied": statuses.count("applied"),
                "scoreable": output["sequence_dp_evaluation"]["scoreable"],
                "verified": False,
            }
    except (FileNotFoundError, json.JSONDecodeError, SequenceDPError) as exc:
        parser.exit(3, f"event-sequence DP failed: {exc}\n")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

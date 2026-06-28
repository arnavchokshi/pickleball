#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.contact_window_candidates import (
    build_contact_window_candidates_from_label_events,
    write_contact_window_candidates,
)
from threed.racketsport.schemas import ContactWindowCandidates


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build review-only contact-window candidates from labels/events.json.")
    parser.add_argument("--events", type=Path, required=True, help="Prototype labels/events.json file.")
    parser.add_argument("--fps", type=float, help="Override FPS when missing from event metadata.")
    parser.add_argument("--pre-s", type=float, default=0.08, help="Candidate window seconds before event timestamp.")
    parser.add_argument("--post-s", type=float, default=0.08, help="Candidate window seconds after event timestamp.")
    parser.add_argument("--out", type=Path, required=True, help="Output contact_window_candidates.json path.")
    args = parser.parse_args(argv)

    try:
        payload = build_contact_window_candidates_from_label_events(
            args.events,
            fps=args.fps,
            pre_s=args.pre_s,
            post_s=args.post_s,
        )
        ContactWindowCandidates.model_validate(payload)
        write_contact_window_candidates(args.out, payload)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: contact-window candidates failed: {exc}", file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "schema_version": 1,
                "out": str(args.out),
                "summary": payload["summary"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

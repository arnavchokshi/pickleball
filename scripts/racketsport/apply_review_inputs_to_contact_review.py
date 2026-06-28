#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.contact_window_review import (  # noqa: E402
    read_contact_window_candidates,
    read_contact_window_review,
    write_contact_window_review,
)
from threed.racketsport.review_input_contact_decisions import apply_review_input_contacts_to_review  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Apply explicit browser review UI contact marks to contact_window_review.json decisions."
    )
    parser.add_argument("--candidates", type=Path, required=True, help="Input contact_window_candidates.json.")
    parser.add_argument("--review", type=Path, required=True, help="Input contact_window_review.json.")
    parser.add_argument("--review-input", type=Path, required=True, help="Saved review_input_server.py JSON.")
    parser.add_argument("--clip", required=True, help="Clip id to apply from the review input.")
    parser.add_argument("--out-review", type=Path, required=True, help="Output updated contact_window_review.json.")
    parser.add_argument("--reviewer", default="review-ui", help="Reviewer name written to accepted decisions.")
    parser.add_argument("--max-delta-s", type=float, default=0.25, help="Maximum UI-contact to candidate time delta.")
    parser.add_argument(
        "--player-map",
        action="append",
        default=[],
        metavar="LABEL=ID",
        help="Override player labels from the review UI, e.g. P1=7. May be passed multiple times.",
    )
    args = parser.parse_args(argv)

    try:
        candidates = read_contact_window_candidates(args.candidates)
        review = read_contact_window_review(args.review)
        review_input = _read_json_object(args.review_input)
        updated = apply_review_input_contacts_to_review(
            candidates,
            review,
            review_input,
            clip=args.clip,
            reviewer=args.reviewer,
            max_delta_s=args.max_delta_s,
            player_map=_parse_player_map(args.player_map),
        )
        write_contact_window_review(args.out_review, updated)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: review-input contact application failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(updated["summary"], indent=2, sort_keys=True))
    return 0


def _read_json_object(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("review input must be a JSON object")
    return payload


def _parse_player_map(items: list[str]) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"player map entry {item!r} must be LABEL=ID")
        label, value = item.split("=", 1)
        label = label.strip()
        if not label:
            raise ValueError(f"player map entry {item!r} has an empty label")
        try:
            mapping[label] = int(value)
        except ValueError as exc:
            raise ValueError(f"player map entry {item!r} has a non-integer id") from exc
    return mapping


if __name__ == "__main__":
    raise SystemExit(main())

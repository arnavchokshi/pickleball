#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.contact_window_review import write_contact_windows  # noqa: E402
from threed.racketsport.review_input_contact_decisions import (  # noqa: E402
    build_contact_windows_from_review_input_contacts,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build contact_windows.json from browser review UI contact marks.")
    parser.add_argument("--review-input", type=Path, required=True, help="Saved review_input_server.py JSON.")
    parser.add_argument("--clip", required=True, help="Clip id to export from the review input.")
    parser.add_argument("--out", type=Path, required=True, help="Output contact_windows.json path.")
    parser.add_argument("--fps", type=float, default=60.0, help="Frame rate used to map contact times to frame indexes.")
    parser.add_argument("--window-radius-s", type=float, default=0.08, help="Half-width of each human contact window.")
    parser.add_argument(
        "--trust-player-labels",
        action="store_true",
        help="Keep review UI player labels as numeric IDs. Omit when labels are placeholders.",
    )
    args = parser.parse_args(argv)

    try:
        review_input = _read_json_object(args.review_input)
        payload = build_contact_windows_from_review_input_contacts(
            review_input,
            clip=args.clip,
            fps=args.fps,
            window_radius_s=args.window_radius_s,
            trust_player_labels=args.trust_player_labels,
        )
        write_contact_windows(args.out, payload)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: contact-window export failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps({"clip": args.clip, "event_count": len(payload["events"]), "out": str(args.out)}, indent=2))
    return 0


def _read_json_object(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("review input must be a JSON object")
    return payload


if __name__ == "__main__":
    raise SystemExit(main())

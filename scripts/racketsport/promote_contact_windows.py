#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.contact_window_review import (
    build_contact_window_review_template,
    promote_reviewed_contact_windows,
    read_contact_window_candidates,
    read_contact_window_review,
    write_contact_window_review,
    write_contact_windows,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Create contact-window review templates and promote accepted decisions fail-closed."
    )
    parser.add_argument("--candidates", type=Path, required=True, help="Input contact_window_candidates.json.")
    parser.add_argument("--template-out", type=Path, help="Output editable contact_window_review.json template.")
    parser.add_argument("--review", type=Path, help="Reviewed contact_window_review.json with accepted decisions.")
    parser.add_argument("--out-contact-windows", type=Path, help="Output promoted contact_windows.json.")
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Allow writing contact_windows.json with zero accepted contact decisions.",
    )
    args = parser.parse_args(argv)

    if args.template_out is None and (args.review is None or args.out_contact_windows is None):
        parser.error("provide --template-out or both --review and --out-contact-windows")

    result: dict[str, object] = {"schema_version": 1}
    try:
        candidates = read_contact_window_candidates(args.candidates)
        if args.template_out is not None:
            template = build_contact_window_review_template(candidates, candidate_path=args.candidates)
            write_contact_window_review(args.template_out, template)
            result["template_out"] = str(args.template_out)
            result["template_summary"] = template["summary"]

        if args.review is not None or args.out_contact_windows is not None:
            if args.review is None or args.out_contact_windows is None:
                parser.error("--review and --out-contact-windows must be provided together")
            review = read_contact_window_review(args.review)
            contact_windows = promote_reviewed_contact_windows(candidates, review)
            if not contact_windows["events"] and not args.allow_empty:
                raise ValueError("no accepted contact decisions; refusing to write empty contact_windows.json")
            write_contact_windows(args.out_contact_windows, contact_windows)
            result["out_contact_windows"] = str(args.out_contact_windows)
            result["accepted_count"] = len(contact_windows["events"])
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: contact-window review promotion failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

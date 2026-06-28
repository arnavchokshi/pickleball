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
    contact_review_media_paths,
    read_contact_window_candidates,
    read_contact_window_review,
    render_contact_window_review_html,
    write_contact_window_review_html,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render a contact-window review HTML page.")
    parser.add_argument("--candidates", type=Path, required=True, help="Input contact_window_candidates.json.")
    parser.add_argument("--review", type=Path, required=True, help="Input contact_window_review.json.")
    parser.add_argument("--out-html", type=Path, required=True, help="Output contact_window_review.html path.")
    args = parser.parse_args(argv)

    try:
        candidates = read_contact_window_candidates(args.candidates)
        review = read_contact_window_review(args.review)
        html = render_contact_window_review_html(
            candidates,
            review,
            review_filename=str(args.review),
            media_paths=contact_review_media_paths(args.out_html.parent),
        )
        write_contact_window_review_html(args.out_html, html)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: contact-window review render failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps({"schema_version": 1, "html_path": str(args.out_html)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

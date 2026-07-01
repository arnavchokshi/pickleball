#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.body_world_label_review_corrections import (  # noqa: E402
    apply_body_world_label_review_corrections,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Apply accepted BODY world-label review corrections to a review bundle."
    )
    parser.add_argument("--template", type=Path, required=True, help="Input body_world_joints.template.json.")
    parser.add_argument("--overlay-index", type=Path, help="Input body_world_label_review_overlay_index.json.")
    parser.add_argument("--corrections", type=Path, required=True, help="Corrections manifest JSON.")
    parser.add_argument("--out-template", type=Path, required=True, help="Output reviewed template path.")
    parser.add_argument("--out-overlay-index", type=Path, help="Output reviewed overlay index path.")
    args = parser.parse_args(argv)

    try:
        summary = apply_body_world_label_review_corrections(
            template_path=args.template,
            overlay_index_path=args.overlay_index,
            corrections_path=args.corrections,
            out_template_path=args.out_template,
            out_overlay_index_path=args.out_overlay_index,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: BODY world-label review correction apply failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] in {"applied", "no_accepted_corrections"} else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.body_world_label_review_corrections import (  # noqa: E402
    build_body_world_label_review_corrections_template,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build pending corrections for a BODY world-label review bundle."
    )
    parser.add_argument("--template", type=Path, required=True, help="Input body_world_joints.template.json.")
    parser.add_argument("--overlay-index", type=Path, help="Input body_world_label_review_overlay_index.json.")
    parser.add_argument("--out", type=Path, required=True, help="Output corrections manifest path.")
    parser.add_argument("--manifest-id", required=True, help="Corrections manifest id.")
    parser.add_argument(
        "--created-at",
        default=None,
        help="RFC3339 timestamp. Defaults to the current UTC time.",
    )
    parser.add_argument("--annotator", default="human_reviewer", help="Annotator id for pending corrections.")
    args = parser.parse_args(argv)

    created_at = args.created_at or datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    try:
        summary = build_body_world_label_review_corrections_template(
            template_path=args.template,
            overlay_index_path=args.overlay_index,
            out_path=args.out,
            manifest_id=args.manifest_id,
            created_at=created_at,
            annotator=args.annotator,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: BODY world-label correction template failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

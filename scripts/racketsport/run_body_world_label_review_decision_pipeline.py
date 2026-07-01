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
    run_body_world_label_review_decision_pipeline,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Merge saved BODY overlay-review decisions, apply accepted corrections, "
            "and finalize BODY world labels only when all gates are satisfied."
        )
    )
    parser.add_argument("--template", type=Path, required=True, help="Input body_world_joints.template.json.")
    parser.add_argument("--overlay-index", type=Path, required=True, help="Input body_world_label_review_overlay_index.json.")
    parser.add_argument("--corrections", type=Path, required=True, help="Pending BODY corrections manifest JSON.")
    parser.add_argument("--review-input", type=Path, required=True, help="Saved review-input JSON.")
    parser.add_argument("--run-id", required=True, help="BODY run id key inside body_world_label_review.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for merged/reviewed/final artifacts.")
    parser.add_argument("--final-labels-out", type=Path, help="Optional canonical output path for body_world_joints.json.")
    parser.add_argument(
        "--finalization-report-out",
        type=Path,
        help="Optional canonical output path for body_world_label_finalization.json.",
    )
    parser.add_argument("--summary-out", type=Path, help="Optional path to write the pipeline summary JSON.")
    parser.add_argument("--allow-blocked", action="store_true", help="Exit 0 even when the pipeline remains blocked.")
    args = parser.parse_args(argv)

    try:
        summary = run_body_world_label_review_decision_pipeline(
            template_path=args.template,
            overlay_index_path=args.overlay_index,
            corrections_path=args.corrections,
            review_input_path=args.review_input,
            run_id=args.run_id,
            out_dir=args.out_dir,
            final_labels_path=args.final_labels_out,
            finalization_report_path=args.finalization_report_out,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: BODY review-decision pipeline failed: {exc}", file=sys.stderr)
        return 1

    if args.summary_out is not None:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] == "finalized" or args.allow_blocked else 1


if __name__ == "__main__":
    raise SystemExit(main())

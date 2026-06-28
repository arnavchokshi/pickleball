#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.racket_true_corners import (  # noqa: E402
    build_paddle_true_corner_review,
    load_json_object,
    render_paddle_true_corner_crop_sheet,
    true_corner_labels_to_candidates,
    write_json_artifact,
)
from threed.racketsport.schemas import RacketCandidates, validate_artifact_file  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build fail-closed paddle true-corner review artifacts.")
    parser.add_argument("--clip", required=True, help="Clip identifier.")
    parser.add_argument("--racket-candidates", type=Path, required=True, help="Existing candidate artifact, including box-preview candidates.")
    parser.add_argument("--true-corner-labels", type=Path, help="Optional reviewed true-corner label artifact.")
    parser.add_argument("--true-corner-candidates-out", type=Path, help="Optional output path for converted true-corner racket_candidates.json.")
    parser.add_argument("--video", type=Path, help="Optional source video used to render true-corner crop sheet.")
    parser.add_argument("--crop-sheet", type=Path, help="Optional output PNG crop sheet for labeling.")
    parser.add_argument("--overlay", type=Path, help="Optional existing candidate overlay video path to index in the review artifact.")
    parser.add_argument("--max-crops", type=int, default=48, help="Maximum candidate crops to render in the crop sheet.")
    parser.add_argument("--max-required-labels", type=int, default=None, help="Optional cap for required_labels JSON entries.")
    parser.add_argument("--out", type=Path, required=True, help="Output paddle_true_corner_review.json path.")
    args = parser.parse_args(argv)

    try:
        candidates = validate_artifact_file("racket_candidates", args.racket_candidates)
        if not isinstance(candidates, RacketCandidates):
            raise ValueError("racket candidates artifact did not parse as RacketCandidates")

        true_candidates = None
        true_summary = None
        if args.true_corner_labels is not None:
            true_candidates_payload, true_summary = true_corner_labels_to_candidates(load_json_object(args.true_corner_labels))
            true_candidates = RacketCandidates.model_validate(true_candidates_payload)
            if args.true_corner_candidates_out is not None:
                write_json_artifact(args.true_corner_candidates_out, true_candidates_payload)

        crop_summary = None
        if args.crop_sheet is not None:
            if args.video is None:
                raise ValueError("--video is required when --crop-sheet is provided")
            crop_summary = render_paddle_true_corner_crop_sheet(
                video_path=args.video,
                racket_candidates=candidates,
                output_path=args.crop_sheet,
                max_items=args.max_crops,
            )

        payload = build_paddle_true_corner_review(
            clip=args.clip,
            racket_candidates=candidates,
            true_corner_candidates=true_candidates,
            crop_sheet_path=args.crop_sheet,
            overlay_path=args.overlay,
            max_required_labels=args.max_required_labels,
        )
        if true_summary is not None:
            payload["true_corner_label_ingest_summary"] = true_summary
        if crop_summary is not None:
            payload["crop_sheet_summary"] = crop_summary
        write_json_artifact(args.out, payload)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: paddle true-corner review failed: {exc}", file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "schema_version": 1,
                "out": str(args.out),
                "status": payload["status"],
                "trusted_for_rkt_promotion": payload["trusted_for_rkt_promotion"],
                "required_label_count": payload["required_label_count"],
                "true_corner_label_count": payload["true_corner_label_count"],
                "visuals": payload["visuals"],
                "promotion_blockers": payload["promotion_blockers"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

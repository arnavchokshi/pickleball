#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.court_keypoint_eval import (  # noqa: E402
    DEFAULT_ACCEPTED_CLIPS,
    build_court_keypoint_no_tap_eval_report,
    render_court_keypoint_no_tap_eval_markdown,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate CAL-3 no-tap court-keypoint checkpoint plumbing without claiming CAL-3 verification.",
    )
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument(
        "--eval-root",
        type=Path,
        default=None,
        help="Eval clip root for partial visible-label artifacts, e.g. eval_clips/ball.",
    )
    parser.add_argument(
        "--include-partial",
        action="append",
        default=[],
        help="Clip id with labels/court_keypoints_partial.json to include in the visible-point gate.",
    )
    parser.add_argument(
        "--partial-proposal-root",
        type=Path,
        default=None,
        help="Optional root containing court_corner_proposals.json for included partial clips.",
    )
    parser.add_argument(
        "--detector-v2-proposal-root",
        type=Path,
        default=None,
        help="Optional root containing court_detector_v2_proposals.json for included partial clips.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("runs/pickleball_pretraining/court_keypoint_20260628/court_keypoint_heatmap.pt"),
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("runs/pickleball_pretraining/court_keypoint_20260628/court_keypoint_metrics.json"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("runs/pickleball_pretraining/court_keypoint_20260628/court_keypoint_no_tap_eval.json"),
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and report the H100 gate command only.")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--frames-per-clip", type=int, default=5)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument(
        "--thresholds",
        default=None,
        help="Comma-separated threshold sweep. Defaults to the evaluator sweep plus --min-confidence.",
    )
    parser.add_argument("--markdown-out", type=Path, default=None, help="Optional Markdown diagnostic output path.")
    args = parser.parse_args(argv)
    run_root = args.run_root or args.eval_root or Path("runs/eval0/prototype_gate_h100_v2")
    accepted_clips = () if args.run_root is None and args.eval_root is not None else DEFAULT_ACCEPTED_CLIPS

    try:
        report = build_court_keypoint_no_tap_eval_report(
            run_root=run_root,
            checkpoint=args.checkpoint,
            metrics=args.metrics,
            out=args.out,
            dry_run=args.dry_run,
            device=args.device,
            frames_per_clip=args.frames_per_clip,
            min_confidence=args.min_confidence,
            thresholds=_parse_thresholds(args.thresholds),
            accepted_clips=accepted_clips,
            eval_root=args.eval_root,
            include_partial=args.include_partial,
            partial_proposal_root=args.partial_proposal_root,
            detector_v2_proposal_root=args.detector_v2_proposal_root,
        )
        if args.markdown_out is not None:
            args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
            args.markdown_out.write_text(
                render_court_keypoint_no_tap_eval_markdown(report),
                encoding="utf-8",
            )
    except Exception as exc:
        print(f"ERROR: court-keypoint no-tap evaluation failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True))
    return 0 if report.status in {"ready_for_h100", "ran_not_verified"} else 1


def _parse_thresholds(raw: str | None) -> list[float] | None:
    if raw is None or not raw.strip():
        return None
    values: list[float] = []
    for chunk in raw.split(","):
        stripped = chunk.strip()
        if not stripped:
            continue
        values.append(float(stripped))
    return values


if __name__ == "__main__":
    raise SystemExit(main())

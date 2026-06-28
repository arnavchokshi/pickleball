#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.tracknetv4_adapter import run_tracknetv4_or_convert


def main() -> int:
    parser = argparse.ArgumentParser(description="Run or convert TrackNetV4 predictions into ball_track.json.")
    parser.add_argument("--predictions-csv", type=Path, default=None, help="Existing TrackNetV4 CSV to convert.")
    parser.add_argument("--video", type=Path, default=None, help="Video input for TrackNetV4 src/predict.py.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="TrackNetV4 .keras checkpoint.")
    parser.add_argument("--tracknetv4-repo", type=Path, default=None, help="TrackNetV4 repo containing src/predict.py.")
    parser.add_argument("--prediction-dir", type=Path, default=None, help="Keep TrackNetV4 CSV outputs in this directory.")
    parser.add_argument("--expected-csv", type=Path, default=None, help="Expected CSV path or name after prediction.")
    parser.add_argument(
        "--command",
        type=str,
        default=None,
        help=(
            "Optional shlex command template. Placeholders: {python}, {repo}, {predict_py}, "
            "{video}, {checkpoint}, {output_dir}, {queue_length}."
        ),
    )
    parser.add_argument("--queue-length", type=int, default=5, help="TrackNetV4 trajectory queue length.")
    parser.add_argument(
        "--mark-real-run-succeeded",
        action="store_true",
        help="Set metadata verified=true only after a real external TrackNetV4 command succeeds.",
    )
    parser.add_argument("--fps", type=float, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--metadata-out", type=Path, default=None)
    args = parser.parse_args()

    try:
        summary = run_tracknetv4_or_convert(
            out=args.out,
            fps=args.fps,
            metadata_out=args.metadata_out,
            predictions_csv=args.predictions_csv,
            video=args.video,
            checkpoint=args.checkpoint,
            tracknetv4_repo=args.tracknetv4_repo,
            prediction_dir=args.prediction_dir,
            command=args.command,
            queue_length=args.queue_length,
            expected_csv=args.expected_csv,
            mark_real_run_succeeded=args.mark_real_run_succeeded,
        )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

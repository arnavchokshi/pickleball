#!/usr/bin/env python3
"""Build review-only audio onset cue artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.audio_onsets import (  # noqa: E402
    build_audio_onsets_from_video,
    build_audio_onsets_from_wav,
    write_audio_onsets,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build review-only audio_onsets.json from a WAV or video file.")
    parser.add_argument("--input", type=Path, required=True, help="Input WAV or video file.")
    parser.add_argument("--out", type=Path, required=True, help="Output audio_onsets.json path.")
    parser.add_argument("--clip", help="Optional clip id to store in the review artifact.")
    parser.add_argument("--frame-rate", type=float, help="Optional frame rate for nearest-frame onset metadata.")
    parser.add_argument("--threshold-score", type=float, default=0.55, help="Normalized onset delta threshold.")
    parser.add_argument("--frame-size-s", type=float, default=0.020, help="RMS frame size in seconds.")
    parser.add_argument("--hop-s", type=float, default=0.005, help="RMS hop size in seconds.")
    parser.add_argument("--min-separation-s", type=float, default=0.080, help="Minimum spacing between kept onsets.")
    parser.add_argument("--sample-rate-hz", type=int, help="Video audio decode sample rate.")
    parser.add_argument("--analysis-sample-rate-hz", type=int, default=16_000, help="Video audio decode/analysis sample rate.")
    parser.add_argument("--start-s", type=float, default=0.0, help="Start time inside the input media for video audio.")
    parser.add_argument("--duration-s", type=float, help="Optional video audio duration to analyze.")
    args = parser.parse_args()
    analysis_sample_rate_hz = args.sample_rate_hz or args.analysis_sample_rate_hz

    if args.input.suffix.lower() == ".wav":
        payload = build_audio_onsets_from_wav(
            args.input,
            threshold_score=args.threshold_score,
            frame_size_s=args.frame_size_s,
            hop_s=args.hop_s,
            min_separation_s=args.min_separation_s,
            clip=args.clip,
            frame_rate=args.frame_rate,
            analysis_sample_rate_hz=analysis_sample_rate_hz,
        )
    else:
        payload = build_audio_onsets_from_video(
            args.input,
            sample_rate_hz=analysis_sample_rate_hz,
            analysis_sample_rate_hz=analysis_sample_rate_hz,
            threshold_score=args.threshold_score,
            frame_size_s=args.frame_size_s,
            hop_s=args.hop_s,
            min_separation_s=args.min_separation_s,
            start_s=args.start_s,
            duration_s=args.duration_s,
            clip=args.clip,
            frame_rate=args.frame_rate,
        )
    write_audio_onsets(args.out, payload)
    print(f"wrote {args.out} ({payload['status']}, onsets={payload['summary']['onset_count']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

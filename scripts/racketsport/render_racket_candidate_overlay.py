#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.racket_candidate_overlay import (  # noqa: E402
    load_racket_candidates,
    render_racket_candidate_overlay,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render a CPU OpenCV racket/paddle-candidate overlay video.")
    parser.add_argument("--video", type=Path, required=True, help="Source video path.")
    parser.add_argument("--racket-candidates", type=Path, required=True, help="Schema-valid racket_candidates.json path.")
    parser.add_argument("--out", type=Path, required=True, help="Output overlay MP4 path.")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum number of output frames to render.")
    parser.add_argument("--h264-out", type=Path, default=None, help="Optional browser-safe H.264 MP4 output path.")
    parser.add_argument(
        "--candidate-coordinate-width",
        type=int,
        default=None,
        help="Width of the image coordinate space used by racket_candidates.json.",
    )
    parser.add_argument(
        "--candidate-coordinate-height",
        type=int,
        default=None,
        help="Height of the image coordinate space used by racket_candidates.json.",
    )
    args = parser.parse_args(argv)

    try:
        candidates = load_racket_candidates(args.racket_candidates)
        summary = render_racket_candidate_overlay(
            video_path=args.video,
            candidates=candidates,
            output_path=args.out,
            max_frames=args.max_frames,
            candidate_coord_width=args.candidate_coordinate_width,
            candidate_coord_height=args.candidate_coordinate_height,
        )
        if args.h264_out is not None:
            _transcode_h264(args.out, args.h264_out)
            summary["source_overlay_path"] = summary["overlay_path"]
            summary["overlay_path"] = str(args.h264_out)
            summary["video_codec"] = "h264"
            (args.out.parent / "racket_candidate_overlay_index.json").write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _transcode_h264(source: Path, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(source),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(out),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"ffmpeg h264 transcode failed: {completed.stderr.strip()}")


if __name__ == "__main__":
    raise SystemExit(main())

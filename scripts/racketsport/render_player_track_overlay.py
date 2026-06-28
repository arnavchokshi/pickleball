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

from threed.racketsport.player_track_overlay import load_tracks, render_player_track_overlay  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render a CPU OpenCV player-track overlay video.")
    parser.add_argument("--video", type=Path, required=True, help="Source video path.")
    parser.add_argument("--tracks", type=Path, required=True, help="Schema-valid tracks.json path.")
    parser.add_argument("--out", type=Path, required=True, help="Output overlay MP4 path.")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum number of output frames to render.")
    parser.add_argument("--bbox-scale-x", type=float, default=1.0, help="Scale track bbox x-coordinates for the source video.")
    parser.add_argument("--bbox-scale-y", type=float, default=1.0, help="Scale track bbox y-coordinates for the source video.")
    parser.add_argument("--h264-out", type=Path, default=None, help="Optional browser-safe H.264 MP4 output path.")
    args = parser.parse_args(argv)

    try:
        tracks = load_tracks(args.tracks)
        summary = render_player_track_overlay(
            video_path=args.video,
            tracks=tracks,
            output_path=args.out,
            max_frames=args.max_frames,
            bbox_scale_x=args.bbox_scale_x,
            bbox_scale_y=args.bbox_scale_y,
        )
        if args.h264_out is not None:
            _transcode_h264(args.out, args.h264_out)
            summary["source_overlay_path"] = summary["overlay_path"]
            summary["overlay_path"] = str(args.h264_out)
            summary["video_codec"] = "h264"
            (args.out.parent / "player_track_overlay_index.json").write_text(
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

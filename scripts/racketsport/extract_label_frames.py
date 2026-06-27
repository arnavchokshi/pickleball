#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.io_decode import probe_clip


VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".avi", ".mkv"}


def _clip_dirs(root: Path) -> list[Path]:
    return sorted(path for path in root.iterdir() if path.is_dir() and not path.name.startswith("."))


def _source_video(clip_dir: Path) -> Path:
    source_candidates = sorted(
        path for path in clip_dir.iterdir() if path.name.startswith("source") and path.suffix.lower() in VIDEO_SUFFIXES
    )
    if source_candidates:
        return source_candidates[0]
    candidates = sorted(path for path in clip_dir.iterdir() if path.suffix.lower() in VIDEO_SUFFIXES)
    if not candidates:
        raise FileNotFoundError(f"no video source found in {clip_dir}")
    return candidates[0]


def _clean_frames(frames_dir: Path) -> None:
    for frame in frames_dir.glob("frame_*.jpg"):
        frame.unlink()


def extract_clip_label_frames(
    *,
    clip_dir: Path,
    out_dir: Path,
    every_frames: int = 30,
    max_width: int = 1280,
    max_frames: int | None = None,
) -> dict[str, Any]:
    if every_frames <= 0:
        raise ValueError("every_frames must be positive")
    if max_width <= 0:
        raise ValueError("max_width must be positive")
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be positive")

    source = _source_video(clip_dir)
    source_meta = probe_clip(source)
    frames_dir = out_dir / clip_dir.name
    frames_dir.mkdir(parents=True, exist_ok=True)
    _clean_frames(frames_dir)

    vf = f"select='not(mod(n\\,{every_frames}))',scale=min({max_width}\\,iw):-2"
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-vf",
        vf,
        "-vsync",
        "vfr",
    ]
    if max_frames is not None:
        command.extend(["-frames:v", str(max_frames)])
    command.append(str(frames_dir / "frame_%06d.jpg"))

    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg is required for label-frame extraction") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"ffmpeg label-frame extraction failed for {source}: {exc.stderr.strip()}") from exc

    frames = sorted(frames_dir.glob("frame_*.jpg"))
    manifest = {
        "schema_version": 1,
        "clip": clip_dir.name,
        "clip_dir": str(clip_dir),
        "source": str(source),
        "source_resolution": [source_meta.width, source_meta.height],
        "source_fps": source_meta.fps,
        "source_duration_s": source_meta.duration_s,
        "sample_every_frames": every_frames,
        "max_width": max_width,
        "frame_count": len(frames),
        "frames": [frame.name for frame in frames],
    }
    (frames_dir / "label_frame_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest | {"frames_dir": str(frames_dir)}


def extract_label_frames(
    *,
    root: Path,
    out: Path,
    every_frames: int = 30,
    max_width: int = 1280,
    max_frames: int | None = None,
) -> dict[str, Any]:
    if not root.exists():
        raise FileNotFoundError(root)
    clips = [
        extract_clip_label_frames(
            clip_dir=clip_dir,
            out_dir=out,
            every_frames=every_frames,
            max_width=max_width,
            max_frames=max_frames,
        )
        for clip_dir in _clip_dirs(root)
    ]
    return {
        "schema_version": 1,
        "root": str(root),
        "out": str(out),
        "clip_count": len(clips),
        "total_frames": sum(clip["frame_count"] for clip in clips),
        "clips": clips,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract DATA-1 frame packs for manual label review.")
    parser.add_argument("--root", type=Path, default=Path("data/testclips"))
    parser.add_argument("--out", type=Path, default=Path("runs/label_frames"))
    parser.add_argument("--every-frames", type=int, default=30)
    parser.add_argument("--max-width", type=int, default=1280)
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args()

    summary = extract_label_frames(
        root=args.root,
        out=args.out,
        every_frames=args.every_frames,
        max_width=args.max_width,
        max_frames=args.max_frames,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

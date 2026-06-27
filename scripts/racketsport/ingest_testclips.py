#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.io_decode import analyze_clip_qc, probe_clip


VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".avi", ".mkv"}


def _clip_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.suffix.lower() in VIDEO_SUFFIXES)


def ingest_testclips(
    root: Path,
    out: Path,
    *,
    include_qc: bool = True,
    qc_sample_fps: float = 1.0,
    qc_max_frames: int = 30,
    qc_max_width: int = 160,
) -> list[Path]:
    written: list[Path] = []
    for clip in _clip_files(root):
        rel = clip.relative_to(root)
        clip_out = out / rel.with_suffix("") / "frames_meta.json"
        clip_out.parent.mkdir(parents=True, exist_ok=True)
        source = probe_clip(clip)
        metadata = source.to_frames_meta()
        metadata["source_relpath"] = str(rel)
        if include_qc:
            qc = analyze_clip_qc(
                clip,
                source=source,
                sample_fps=qc_sample_fps,
                max_frames=qc_max_frames,
                max_width=qc_max_width,
            )
            metadata["clip_qc"] = qc.to_dict()
            metadata["capture_quality"] = qc.capture_quality.model_dump()
        clip_out.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        written.append(clip_out)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest Phase 0 racket-sport test clips.")
    parser.add_argument("--root", type=Path, default=Path("data/testclips"))
    parser.add_argument("--out", type=Path, default=Path("runs/phase0"))
    parser.add_argument("--skip-qc", action="store_true", help="Only write ffprobe metadata; skip sampled QC.")
    parser.add_argument("--qc-sample-fps", type=float, default=1.0)
    parser.add_argument("--qc-max-frames", type=int, default=30)
    parser.add_argument("--qc-max-width", type=int, default=160)
    args = parser.parse_args()

    if not args.root.exists():
        raise FileNotFoundError(args.root)
    written = ingest_testclips(
        args.root,
        args.out,
        include_qc=not args.skip_qc,
        qc_sample_fps=args.qc_sample_fps,
        qc_max_frames=args.qc_max_frames,
        qc_max_width=args.qc_max_width,
    )
    print(f"wrote {len(written)} frames_meta.json files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

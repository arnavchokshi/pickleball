#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.testclips import TestClipMetadata, build_clip_manifest


VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".avi", ".mkv"}


def _safe_clip_name(name: str) -> str:
    if not name or any(part in {".", ".."} for part in Path(name).parts) or "/" in name:
        raise ValueError("clip name must be a single directory name")
    return name


def register_testclip(
    *,
    source: Path,
    root: Path,
    name: str,
    metadata: TestClipMetadata,
    symlink: bool = False,
) -> dict:
    if not source.is_file():
        raise FileNotFoundError(source)
    if source.suffix.lower() not in VIDEO_SUFFIXES:
        raise ValueError(f"unsupported video suffix: {source.suffix}")

    clip_name = _safe_clip_name(name)
    clip_dir = root / clip_name
    clip_dir.mkdir(parents=True, exist_ok=True)
    labels_dir = clip_dir / "labels"
    labels_dir.mkdir(exist_ok=True)

    target = clip_dir / f"source{source.suffix.lower()}"
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    if symlink:
        target.symlink_to(source.resolve())
    else:
        shutil.copy2(source, target)

    metadata_path = clip_dir / "clip_metadata.json"
    if metadata_path.exists():
        raise FileExistsError(metadata_path)
    metadata_path.write_text(metadata.model_dump_json(indent=2) + "\n", encoding="utf-8")

    manifest = build_clip_manifest(clip_dir)
    return {
        "clip": clip_name,
        "clip_dir": str(clip_dir),
        "source": str(target),
        "metadata_path": str(metadata_path),
        "labels_dir": str(labels_dir),
        "metadata_present": manifest.metadata_present,
        "ready": manifest.is_ready,
        "missing_label_files": manifest.missing_label_files,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Register a candidate DATA-1 test clip with metadata.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path("data/testclips"))
    parser.add_argument("--name", required=True)
    parser.add_argument("--camera-height", choices=["low", "mid", "high"], required=True)
    parser.add_argument(
        "--camera-angle",
        choices=["shallow_baseline", "steep_corner", "side_fence", "near_overhead"],
        required=True,
    )
    parser.add_argument("--play-type", choices=["doubles", "singles_drill", "messy_real_world"], required=True)
    parser.add_argument("--environment", choices=["indoor", "outdoor"], required=True)
    parser.add_argument("--frame-rate-fps", type=int, required=True)
    parser.add_argument("--duration-s", type=float, required=True)
    parser.add_argument("--racket-gt", action="store_true")
    parser.add_argument("--symlink", action="store_true", help="Symlink the source clip instead of copying it.")
    args = parser.parse_args()

    metadata = TestClipMetadata(
        schema_version=1,
        camera_height=args.camera_height,
        camera_angle=args.camera_angle,
        play_type=args.play_type,
        environment=args.environment,
        frame_rate_fps=args.frame_rate_fps,
        duration_s=args.duration_s,
        racket_gt=args.racket_gt,
    )
    result = register_testclip(
        source=args.source,
        root=args.root,
        name=args.name,
        metadata=metadata,
        symlink=args.symlink,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

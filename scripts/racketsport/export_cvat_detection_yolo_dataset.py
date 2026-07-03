#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.cvat_detection_dataset import (  # noqa: E402
    ClipSpec,
    class_map_for_preset,
    export_cvat_detection_yolo_dataset,
)
from threed.racketsport.eval_guard import assert_not_training_on_eval_clip  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Export reviewed CVAT video annotations as a YOLO detection dataset.")
    parser.add_argument("--clip", action="append", default=[], required=True, help="clip_id=video.mp4=reviewed_boxes.json")
    parser.add_argument("--preset", choices=("player", "paddle", "ball", "combined"), required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--split-mode", choices=("alternating", "by_clip"), default="by_clip")
    parser.add_argument("--val-clip", action="append", default=[])
    parser.add_argument("--val-every", type=int, default=5)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    args = parser.parse_args()

    try:
        clips = _parse_clips(args.clip)
        # Eval-clip integrity gate (fail closed): this builder writes YOLO
        # detector training data directly from CVAT clips, so it counts as
        # training-input creation. See threed/racketsport/eval_guard.py.
        assert_not_training_on_eval_clip(
            (value for clip in clips for value in (clip.clip_id, str(clip.video_path), str(clip.reviewed_boxes_path))),
            allow_internal_val=False,
        )
        summary = export_cvat_detection_yolo_dataset(
            clips=clips,
            out_dir=args.out_dir,
            class_map=class_map_for_preset(args.preset),
            split_mode=args.split_mode,
            val_clips=tuple(args.val_clip),
            val_every=int(args.val_every),
            frame_stride=int(args.frame_stride),
            jpeg_quality=int(args.jpeg_quality),
        )
    except Exception as exc:
        print(f"CVAT detection YOLO dataset export failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(_compact_summary(summary), sort_keys=True))
    return 0


def _parse_clips(specs: Sequence[str]) -> list[ClipSpec]:
    clips: list[ClipSpec] = []
    for spec in specs:
        parts = spec.split("=", 2)
        if len(parts) != 3:
            raise ValueError(f"clip spec must be clip_id=video=reviewed_boxes: {spec}")
        clip_id, video, reviewed = parts
        clip = ClipSpec(clip_id=clip_id, video_path=Path(video), reviewed_boxes_path=Path(reviewed))
        if not clip.video_path.is_file():
            raise FileNotFoundError(f"missing video for {clip_id}: {clip.video_path}")
        if not clip.reviewed_boxes_path.is_file():
            raise FileNotFoundError(f"missing reviewed boxes for {clip_id}: {clip.reviewed_boxes_path}")
        clips.append(clip)
    return clips


def _compact_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in summary.items() if key != "rows"}


if __name__ == "__main__":
    raise SystemExit(main())

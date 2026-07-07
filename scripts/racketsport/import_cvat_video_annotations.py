#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.cvat_video import (  # noqa: E402
    import_cvat_video_zip,
    write_cvat_video_annotations,
    write_person_ground_truth_from_cvat_video,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Import CVAT for video 1.1 ZIP annotations into reviewed pickleball artifacts.")
    parser.add_argument("--cvat-zip", type=Path, required=True, help="Input CVAT for video 1.1 ZIP.")
    parser.add_argument("--clip-id", required=True, help="Stable clip id to write into the imported artifacts.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for imported JSON artifacts.")
    parser.add_argument("--fps", type=float, default=None, help="Optional source video FPS.")
    parser.add_argument("--max-frame-index", type=int, default=None, help="Discard annotations after this 0-based source frame index.")
    args = parser.parse_args()

    try:
        annotations, person_ground_truth = import_cvat_video_zip(
            args.cvat_zip,
            clip_id=args.clip_id,
            fps=args.fps,
            max_frame_index=args.max_frame_index,
        )
        args.out_dir.mkdir(parents=True, exist_ok=True)
        reviewed_out = args.out_dir / "reviewed_boxes.json"
        person_out = args.out_dir / "person_ground_truth.json"
        summary_out = args.out_dir / "import_summary.json"
        write_cvat_video_annotations(reviewed_out, annotations)
        write_person_ground_truth_from_cvat_video(person_out, person_ground_truth)
        summary_out.write_text(
            json.dumps(
                {
                    "clip_id": annotations.clip_id,
                    "source_zip": str(args.cvat_zip),
                    "reviewed_boxes": str(reviewed_out),
                    "person_ground_truth": str(person_out),
                    "source_format": annotations.source_format,
                    "frame_count": annotations.summary.frame_count,
                    "visible_box_count_by_label": annotations.summary.visible_box_count_by_label,
                    "track_count_by_label": annotations.summary.track_count_by_label,
                    "outside_box_count": annotations.summary.outside_box_count,
                    "reviewed_frame_count": len(annotations.reviewed_frame_indices or annotations.frames),
                    "reviewed_frame_indices_source": annotations.reviewed_frame_indices_source or "dense_all_frames",
                    "max_frame_index": args.max_frame_index,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    except Exception as exc:
        print(f"CVAT video import failed: {exc}", file=sys.stderr)
        return 1

    print(
        "CVAT video import: "
        f"clip_id={annotations.clip_id} "
        f"frames={annotations.summary.frame_count} "
        f"labels={annotations.summary.visible_box_count_by_label}",
        file=sys.stderr,
    )
    print(reviewed_out)
    print(person_out)
    print(summary_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

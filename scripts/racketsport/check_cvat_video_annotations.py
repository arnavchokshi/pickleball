#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.cvat_annotation_sanity import build_annotation_sanity_report  # noqa: E402
from threed.racketsport.schemas import CvatVideoAnnotations, validate_artifact_file  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Write deterministic sanity reports for reviewed CVAT video annotations.")
    parser.add_argument("--clip", action="append", default=[], required=True, help="clip_id=reviewed_boxes.json")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-players", type=int, default=4)
    parser.add_argument("--long-gap-frames", type=int, default=30)
    parser.add_argument("--jump-factor", type=float, default=8.0)
    args = parser.parse_args()

    try:
        reports = []
        for clip_id, path in _parse_clips(args.clip):
            annotations = validate_artifact_file("cvat_video_annotations", path)
            if not isinstance(annotations, CvatVideoAnnotations):
                raise ValueError(f"reviewed boxes artifact did not parse as CvatVideoAnnotations: {path}")
            report = build_annotation_sanity_report(
                annotations,
                expected_players=args.expected_players,
                long_gap_frames=args.long_gap_frames,
                jump_factor=args.jump_factor,
            )
            report["requested_clip_id"] = clip_id
            report["reviewed_boxes_path"] = str(path)
            reports.append(report)
        payload = {
            "schema_version": 1,
            "artifact_type": "racketsport_cvat_annotation_sanity_report",
            "clip_count": len(reports),
            "clips": reports,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except Exception as exc:
        print(f"CVAT annotation sanity check failed: {exc}", file=sys.stderr)
        return 1
    print(args.out)
    return 0


def _parse_clips(specs: Sequence[str]) -> list[tuple[str, Path]]:
    clips: list[tuple[str, Path]] = []
    for spec in specs:
        parts = spec.split("=", 1)
        if len(parts) != 2:
            raise ValueError(f"clip spec must be clip_id=reviewed_boxes: {spec}")
        clip_id, reviewed = parts
        path = Path(reviewed)
        if not path.is_file():
            raise FileNotFoundError(f"missing reviewed boxes for {clip_id}: {path}")
        clips.append((clip_id, path))
    return clips


if __name__ == "__main__":
    raise SystemExit(main())

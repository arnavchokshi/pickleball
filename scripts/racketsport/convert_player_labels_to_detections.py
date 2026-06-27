#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.detection_bridge import (  # noqa: E402
    fps_from_manifest,
    load_json_object,
    player_labels_to_detections,
    write_detections,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert prototype player labels into track.py detections.json.")
    parser.add_argument("--players", type=Path, required=True, help="Input labels/players.json file.")
    parser.add_argument("--out", type=Path, required=True, help="Output detections.json path.")
    parser.add_argument("--manifest", type=Path, default=None, help="Optional prototype_autolabel_manifest.json for fps.")
    parser.add_argument("--fps", type=float, default=None, help="Explicit FPS override.")
    parser.add_argument("--include-uncertain", action="store_true", help="Include uncertain player boxes.")
    parser.add_argument("--preserve-label-ids", action="store_true", help="Pass label ids as temporary track ids.")
    args = parser.parse_args()

    fps = args.fps or fps_from_manifest(args.manifest)
    if fps is None:
        print("fps is required via --fps or --manifest", file=sys.stderr)
        return 2

    try:
        labels = load_json_object(args.players)
        detections = player_labels_to_detections(
            labels,
            fps=float(fps),
            include_uncertain=args.include_uncertain,
            preserve_label_ids=args.preserve_label_ids,
        )
        write_detections(args.out, detections)
    except Exception as exc:
        print(f"player-label conversion failed: {exc}", file=sys.stderr)
        return 1

    counts = detections["counts"]
    print(
        "player-label conversion: "
        f"accepted={counts['accepted']} "
        f"skipped_status={counts['skipped_status']} "
        f"skipped_invalid={counts['skipped_invalid']}",
        file=sys.stderr,
    )
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

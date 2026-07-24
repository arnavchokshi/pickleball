#!/usr/bin/env python3
"""Build or score the compact court/foot human-reference packet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from threed.racketsport.court_foot_gold import (  # noqa: E402
    GoldClipSpec,
    build_gold_packet,
    score_gold_review,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build", help="extract frames and prelabels")
    build.add_argument("--config", type=Path, required=True)
    build.add_argument("--out", type=Path, required=True)
    build.add_argument("--frames-per-clip", type=int, default=12)
    score = subparsers.add_parser("score", help="decompose reviewed court/foot errors")
    score.add_argument("--packet", type=Path, required=True)
    score.add_argument("--review", type=Path, required=True)
    score.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "build":
        config = _object(args.config)
        specs = [
            GoldClipSpec(
                clip_id=str(row["clip_id"]),
                video_path=_resolve(args.config.parent, row["video"]),
                court_lock_path=_resolve(args.config.parent, row["court_lock"]),
                artifacts_dir=(
                    None
                    if row.get("artifacts") is None
                    else _resolve(args.config.parent, row["artifacts"])
                ),
            )
            for row in config.get("clips") or []
        ]
        if not specs:
            raise ValueError("config.clips must contain at least one clip")
        packet = build_gold_packet(specs, args.out, frames_per_clip=args.frames_per_clip)
        print(
            json.dumps(
                {
                    "status": "ok",
                    "frame_count": packet["frame_count"],
                    "clip_count": len(packet["clips"]),
                    "start_here": str((args.out / "START_HERE.html").resolve()),
                },
                sort_keys=True,
            )
        )
        return 0
    packet = _object(args.packet)
    review = _object(args.review)
    report = score_gold_review(packet, review)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "ok", "sample_count": report["sample_count"]}, sort_keys=True))
    return 0


def _object(path: Path) -> dict:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _resolve(root: Path, value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else (root / path).resolve()


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.player_global_association import (  # noqa: E402
    GlobalAssociationConfig,
    associate_global_identities,
    tracks_to_global_detections,
)
from threed.racketsport.schemas import Tracks, validate_artifact_file  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Global exactly-N player identity association for source-only person tracks.")
    parser.add_argument("--tracks", type=Path, required=True, help="Input tracks.json from detector/tracker stage.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Directory for repaired tracks.json and summary.")
    parser.add_argument("--embedding-export", type=Path, default=None, help="Optional source-only OSNet/ReID embedding export JSON.")
    parser.add_argument(
        "--embedding-bbox-scale",
        type=float,
        default=1.0,
        help="Scale applied to tracks.json bboxes before matching embedding-export bboxes.",
    )
    parser.add_argument("--max-embedding-bbox-delta-px", type=float, default=2.5)
    parser.add_argument("--expected-players", type=int, default=4)
    parser.add_argument("--max-gap-fill-frames", type=int, default=24)
    parser.add_argument("--max-merge-gap-frames", type=int, default=240)
    parser.add_argument("--max-merge-speed-m-s", type=float, default=9.0)
    parser.add_argument("--appearance-weight", type=float, default=1.0)
    parser.add_argument("--motion-weight", type=float, default=1.0)
    parser.add_argument(
        "--court-margin-m",
        type=float,
        default=0.0,
        help="Apron margin around the court template before outside-court detections are rejected.",
    )
    parser.add_argument(
        "--drop-outside-court",
        action="store_true",
        help="Reject candidate detections outside the court template (plus --court-margin-m) before association.",
    )
    parser.add_argument(
        "--post-association-court-margin-m",
        type=float,
        default=None,
        help=(
            "Optional second, typically tighter court-polygon margin applied to the "
            "final selected tracks after association. Frames outside this margin are "
            "dropped from the output track without changing which fragment/identity "
            "was selected. Use 0.0 to mirror the strict court-only definition used by "
            "off_court_false_positive_frames scoring."
        ),
    )
    args = parser.parse_args()

    try:
        parsed = validate_artifact_file("tracks", args.tracks)
        if not isinstance(parsed, Tracks):
            raise ValueError(f"{args.tracks} did not parse as Tracks")
        config = GlobalAssociationConfig(
            expected_players=args.expected_players,
            max_gap_fill_frames=args.max_gap_fill_frames,
            max_merge_gap_frames=args.max_merge_gap_frames,
            max_merge_speed_m_s=args.max_merge_speed_m_s,
            appearance_weight=args.appearance_weight,
            motion_weight=args.motion_weight,
            drop_outside_court=args.drop_outside_court,
            court_margin_m=args.court_margin_m,
            post_association_court_margin_m=args.post_association_court_margin_m,
        )
        embedding_payload = _read_json_object(args.embedding_export) if args.embedding_export is not None else None
        detections = tracks_to_global_detections(
            parsed,
            embedding_payload=embedding_payload,
            embedding_bbox_scale=args.embedding_bbox_scale,
            max_embedding_bbox_delta_px=args.max_embedding_bbox_delta_px,
        )
        tracks, summary = associate_global_identities(detections, fps=float(parsed.fps), config=config)
        args.out_dir.mkdir(parents=True, exist_ok=True)
        tracks_path = args.out_dir / "tracks.json"
        summary_path = args.out_dir / "global_association_summary.json"
        tracks_path.write_text(json.dumps(tracks.model_dump(mode="json"), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        summary_path.write_text(json.dumps(summary.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except Exception as exc:
        print(f"global association failed: {exc}", file=sys.stderr)
        return 1

    print(tracks_path)
    print(summary_path)
    return 0


def _read_json_object(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""CLI: emit net-anchor court-corner proposals from an image/video/frame directory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.net_anchor_court import (  # noqa: E402
    DEFAULT_USER_INPUT_CONFIDENCE_THRESHOLD,
    draw_net_anchor_overlay,
    load_player_foot_points_from_tracks,
    load_player_suppressed_frame,
    score_corner_proposals,
    solve_net_anchor_court_from_frame,
)

INTERNAL_VAL_SLUGS = {
    "burlington_gold_0300_low_steep_corner",
    "wolverine_mixed_0200_mid_steep_corner",
}
PROTECTED_HELDOUT_SLUGS = {
    "outdoor_webcam_iynbd_1500_long_high_baseline",
    "indoor_doubles_fwuks_0500_long_mid_baseline",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Training-free net-anchor geometric court solver. Writes proposal artifacts only; "
            "does not write court_calibration.json."
        )
    )
    parser.add_argument("--input", type=Path, required=True, help="Source image, video, or directory of JPG frames.")
    parser.add_argument("--clip-id", default="", help="Optional clip identifier stored in the proposal artifact.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for proposals and overlay.")
    parser.add_argument("--max-frames", type=int, default=72, help="Max video/directory frames for temporal median.")
    parser.add_argument("--stride", type=int, default=6, help="Video frame stride for temporal median.")
    parser.add_argument("--start-frame", type=int, default=0, help="Video frame offset for temporal median.")
    parser.add_argument("--tracks-json", type=Path, help="Optional tracks.json used as a weak player-feet ground prior.")
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=DEFAULT_USER_INPUT_CONFIDENCE_THRESHOLD,
        help="Corner confidence below this threshold is listed under needs_user_input.",
    )
    parser.add_argument(
        "--gt-corners",
        type=Path,
        help="Optional INTERNAL-VAL court_corners.json for Burlington/Wolverine scoring only.",
    )
    parser.add_argument(
        "--allow-internal-val-labels",
        action="store_true",
        help="Required with --gt-corners; only Burlington/Wolverine are allowed.",
    )
    args = parser.parse_args(argv)

    try:
        if args.max_frames <= 0:
            raise ValueError("--max-frames must be positive")
        if args.stride <= 0:
            raise ValueError("--stride must be positive")
        if args.confidence_threshold <= 0.0 or args.confidence_threshold > 1.0:
            raise ValueError("--confidence-threshold must be in (0, 1]")
        if not args.input.exists():
            raise ValueError(f"input does not exist: {args.input}")
        if args.tracks_json is not None and not args.tracks_json.exists():
            raise ValueError(f"tracks JSON does not exist: {args.tracks_json}")
        _validate_gt_policy(args.gt_corners, allow_internal_val=args.allow_internal_val_labels)

        frame, frame_meta = load_player_suppressed_frame(
            args.input,
            max_frames=args.max_frames,
            stride=args.stride,
            start_frame=args.start_frame,
        )
        player_foot_points = (
            load_player_foot_points_from_tracks(args.tracks_json)
            if args.tracks_json is not None
            else []
        )
        artifact = solve_net_anchor_court_from_frame(
            frame,
            clip_id=args.clip_id,
            player_foot_points=player_foot_points,
            confidence_threshold=args.confidence_threshold,
        )
        artifact["source"] = {
            **artifact["source"],
            "input": str(args.input),
            **frame_meta,
        }

        args.out_dir.mkdir(parents=True, exist_ok=True)
        proposal_path = args.out_dir / "court_corner_proposals.json"
        proposal_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        overlay = draw_net_anchor_overlay(frame, artifact)
        _write_image(args.out_dir / "court_corner_proposals_overlay.jpg", overlay)

        score_path = None
        score = None
        if args.gt_corners is not None:
            gt = _load_gt_corners_scaled(args.gt_corners, image_size=tuple(artifact["source"]["image_size"]))
            score = score_corner_proposals(artifact, gt)
            score_path = args.out_dir / "corner_score_internal_val.json"
            score_path.write_text(json.dumps(score, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        summary = {
            "proposal_path": str(proposal_path),
            "overlay_path": str(args.out_dir / "court_corner_proposals_overlay.jpg"),
            "score_path": None if score_path is None else str(score_path),
            "solver_confidence": artifact["solver_confidence"],
            "needs_user_input": artifact["needs_user_input"],
            "score_verdict": None if score is None else score["verdict"],
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"ERROR: net-anchor court solve failed: {exc}", file=sys.stderr)
        return 1


def _write_image(path: Path, frame: Any) -> None:
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("opencv-python is required to write overlays") from exc
    if not cv2.imwrite(str(path), frame):
        raise ValueError(f"failed to write overlay image: {path}")


def _validate_gt_policy(gt_path: Path | None, *, allow_internal_val: bool) -> None:
    if gt_path is None:
        return
    text = str(gt_path)
    blocked = [slug for slug in PROTECTED_HELDOUT_SLUGS if slug in text]
    if blocked:
        raise ValueError(f"Outdoor/Indoor labels are never allowed for this solver lane: {blocked[0]}")
    allowed = [slug for slug in INTERNAL_VAL_SLUGS if slug in text]
    if not allowed:
        raise ValueError("--gt-corners is allowed only for Burlington/Wolverine internal-val labels")
    if not allow_internal_val:
        raise ValueError("--allow-internal-val-labels is required before reading Burlington/Wolverine GT")
    if not gt_path.exists():
        raise ValueError(f"GT file does not exist: {gt_path}")


def _load_gt_corners_scaled(gt_path: Path, *, image_size: tuple[int, int]) -> dict[str, list[float]]:
    payload = json.loads(gt_path.read_text(encoding="utf-8"))
    items = payload.get("annotation", {}).get("items", [])
    if not items:
        raise ValueError(f"{gt_path}: no annotation items")
    item = items[0]
    corners = item.get("court_corners")
    if not isinstance(corners, Mapping):
        raise ValueError(f"{gt_path}: missing court_corners")
    label_size = item.get("image_size")
    if not isinstance(label_size, Sequence) or len(label_size) != 2:
        raise ValueError(f"{gt_path}: missing image_size")
    sx = float(image_size[0]) / float(label_size[0])
    sy = float(image_size[1]) / float(label_size[1])
    scaled: dict[str, list[float]] = {}
    for name in ("near_left", "near_right", "far_right", "far_left"):
        value = corners.get(name)
        if not isinstance(value, Sequence) or len(value) != 2:
            raise ValueError(f"{gt_path}: missing corner {name}")
        scaled[name] = [float(value[0]) * sx, float(value[1]) * sy]
    return scaled


if __name__ == "__main__":
    raise SystemExit(main())

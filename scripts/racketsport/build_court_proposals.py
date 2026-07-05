#!/usr/bin/env python3
"""Build fail-closed court proposal artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.court_proposals import (
    CourtProposal,
    CourtProposalReport,
    write_court_proposal_report,
)
from threed.racketsport.court_line_bank import build_line_bank_from_image
from threed.racketsport.court_regulation_proposals import propose_regulation_courts_from_line_bank


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build fail-closed court proposal artifact.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--clip", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-frames", type=int, default=5)
    return parser.parse_args()


def build_empty_proposal_report(
    *,
    video: str,
    clip: str,
    max_frames: int,
) -> CourtProposalReport:
    return CourtProposalReport(
        clip=clip,
        video=video,
        image_size=(0, 0),
        frame_indices=list(range(max(0, max_frames))),
        proposals=[
            CourtProposal(
                proposal_id="proposal_empty_0001",
                source="empty_scaffold",
                court_keypoints={},
                scores={"overall": 0.0},
                gate={
                    "auto_usable": False,
                    "review_usable": False,
                    "failed": ["no_candidate_generation_implemented", "not_verified"],
                    "warnings": [],
                },
            )
        ],
    )


def build_court_proposal_report(
    *,
    video: str,
    clip: str,
    max_frames: int,
) -> CourtProposalReport:
    image, frame_meta = _load_preview_frame(video, max_frames=max_frames)
    if image is None:
        return build_empty_proposal_report(video=video, clip=clip, max_frames=max_frames)
    image_size = (int(image.shape[1]), int(image.shape[0]))
    line_bank = build_line_bank_from_image(image)
    proposals = propose_regulation_courts_from_line_bank(line_bank, image_size=image_size)
    if not proposals:
        proposals = build_empty_proposal_report(video=video, clip=clip, max_frames=max_frames).proposals
    return CourtProposalReport(
        clip=clip,
        video=video,
        image_size=image_size,
        frame_indices=[int(index) for index in frame_meta.get("frame_indices", list(range(max(0, max_frames))))],
        proposals=proposals,
        motion_mode="static",
    )


def _load_preview_frame(video: str, *, max_frames: int) -> tuple[object | None, dict[str, object]]:
    import cv2  # type: ignore[import-not-found]
    import numpy as np

    path = Path(video)
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is not None:
        return image, {"input_kind": "image", "frame_indices": [0]}
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None, {"input_kind": "unreadable", "frame_indices": []}
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    count = max(1, min(max_frames, total if total > 0 else max_frames))
    if total > 0 and count > 1:
        positions = [int(round(value)) for value in np.linspace(0, max(0, total - 1), count)]
    else:
        positions = [0]
    frames = []
    used: list[int] = []
    for frame_index in positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if ok and frame is not None:
            frames.append(frame)
            used.append(int(frame_index))
    cap.release()
    if not frames:
        return None, {"input_kind": "video", "frame_indices": []}
    return np.median(np.stack(frames, axis=0), axis=0).astype(np.uint8), {
        "input_kind": "video",
        "frame_indices": used,
    }


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = build_court_proposal_report(
        video=args.video,
        clip=args.clip,
        max_frames=args.max_frames,
    )
    write_court_proposal_report(out_dir / "court_proposals.json", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

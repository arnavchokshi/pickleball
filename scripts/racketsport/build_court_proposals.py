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
from threed.racketsport.court_model_infer import resolve_court_model_checkpoint_path
from threed.racketsport.court_static_inference import infer_static_court_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build fail-closed court proposal artifact.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--clip", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-frames", type=int, default=8)
    parser.add_argument(
        "--checkpoint",
        help="Optional v2/v3 court evidence checkpoint; defaults to the selected court checkpoint.",
    )
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
    checkpoint_path: str | Path | None = None,
    court_lock_path: str | Path | None = None,
) -> CourtProposalReport:
    image, frame_meta = _load_preview_frame(video, max_frames=max_frames)
    if image is None:
        return build_empty_proposal_report(video=video, clip=clip, max_frames=max_frames)
    image_size = (int(image.shape[1]), int(image.shape[0]))
    line_bank = build_line_bank_from_image(image)
    line_proposals = propose_regulation_courts_from_line_bank(line_bank, image_size=image_size)
    proposals: list[CourtProposal] = []
    assist: dict[str, object] = {"structured_v31": {"status": "checkpoint_unavailable"}}
    motion_mode = "static"
    resolution = resolve_court_model_checkpoint_path(checkpoint_path)
    if resolution is not None:
        try:
            structured = infer_static_court_model(
                list(frame_meta.get("frames") or [image]),
                resolution.path,
                frame_indices=list(frame_meta.get("frame_indices") or [0]),
                court_lock_path=court_lock_path,
            )
            structured_proposal = _structured_proposal(structured)
            if structured_proposal is not None:
                proposals.append(structured_proposal)
            motion_mode = str((structured.get("static_motion") or {}).get("status") or "unknown")
            assist = {
                "structured_v31": {
                    "status": "ok" if structured_proposal is not None else "no_floor_solution",
                    "checkpoint_path": str(resolution.path),
                    "checkpoint_sha256": resolution.sha256,
                    "selected_frame_indices": list(structured.get("selected_frame_indices") or []),
                    "court_lock_written": bool(court_lock_path and structured.get("court_lock")),
                }
            }
        except Exception as exc:  # noqa: BLE001 - advisory path must preserve geometric fallback
            assist = {
                "structured_v31": {
                    "status": "failed_geometric_fallback_used",
                    "checkpoint_path": str(resolution.path),
                    "checkpoint_sha256": resolution.sha256,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            }
    proposals.extend(line_proposals)
    if not proposals:
        proposals = build_empty_proposal_report(video=video, clip=clip, max_frames=max_frames).proposals
    return CourtProposalReport(
        clip=clip,
        video=video,
        image_size=image_size,
        frame_indices=[int(index) for index in frame_meta.get("frame_indices", list(range(max(0, max_frames))))],
        proposals=proposals,
        motion_mode=motion_mode,
        assist=assist,
    )


def _structured_proposal(result: dict[str, object]) -> CourtProposal | None:
    best = result.get("best_court")
    if not isinstance(best, dict):
        return None
    raw_keypoints = best.get("keypoints_xy")
    if not isinstance(raw_keypoints, dict) or len(raw_keypoints) < 4:
        return None
    keypoints = {
        str(name): (float(xy[0]), float(xy[1]))
        for name, xy in raw_keypoints.items()
        if isinstance(xy, (list, tuple)) and len(xy) == 2
    }
    if len(keypoints) < 4:
        return None
    residual = best.get("residual_stats_px")
    residual = residual if isinstance(residual, dict) else {}
    score_components = best.get("score_components")
    score_components = score_components if isinstance(score_components, dict) else {}
    ignored = best.get("ignored_observations")
    inliers = best.get("inlier_observations")
    return CourtProposal(
        proposal_id="proposal_structured_v31_0001",
        source="confidence_aware_structured_court_v31",
        court_keypoints=keypoints,
        homography_image_from_court=best.get("homography_image_from_court"),  # type: ignore[arg-type]
        scores={
            "overall": float(best.get("court_confidence") or 0.0),
            "court_confidence": float(best.get("court_confidence") or 0.0),
            "hypothesis_margin": _optional_float(best.get("hypothesis_margin")),
            "line_support": _optional_float(score_components.get("line_alignment")),
            "mask_support": _optional_float(score_components.get("surface_overlap")),
            "reprojection_px_median": _optional_float(residual.get("median")),
            "reprojection_px_p95": _optional_float(residual.get("p95") or residual.get("p90")),
            "inlier_ratio": _optional_float(best.get("inlier_ratio")),
        },
        gate={
            "auto_usable": False,
            "review_usable": True,
            "failed": ["not_verified", "structured_v31_measurement_authority_disabled"],
            "warnings": (["unsupported_view"] if best.get("supported_view") is False else []),
        },
        evidence={
            "best_court": best,
            "static_motion": result.get("static_motion"),
            "appearance_motion": result.get("appearance_motion"),
            "court_lock": result.get("court_lock"),
            "selected_frame_indices": result.get("selected_frame_indices"),
            "inlier_observation_count": len(inliers) if isinstance(inliers, list) else 0,
            "ignored_observation_count": len(ignored) if isinstance(ignored, list) else 0,
            "fallback_used": str(best.get("source") or "").endswith("prior"),
        },
    )


def _optional_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _load_preview_frame(video: str, *, max_frames: int) -> tuple[object | None, dict[str, object]]:
    import cv2  # type: ignore[import-not-found]
    import numpy as np

    path = Path(video)
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is not None:
        return image, {"input_kind": "image", "frame_indices": [0], "frames": [image]}
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
        "frames": frames,
    }


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = build_court_proposal_report(
        video=args.video,
        clip=args.clip,
        max_frames=args.max_frames,
        checkpoint_path=args.checkpoint,
        court_lock_path=out_dir / "court_lock.json",
    )
    write_court_proposal_report(out_dir / "court_proposals.json", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

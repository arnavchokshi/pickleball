#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.court_detector_v2 import detect_court_v2_from_frame  # noqa: E402
from threed.racketsport.court_keypoint_labels import load_partial_court_keypoints  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate court detector v2 without claiming CAL-3 verification.")
    parser.add_argument("--eval-root", type=Path, default=Path("eval_clips/ball"))
    parser.add_argument("--include-partial", action="append", default=[])
    parser.add_argument("--proposal-root", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    try:
        report = build_court_detector_v2_eval_report(
            eval_root=args.eval_root,
            include_partial=args.include_partial,
            proposal_root=args.proposal_root,
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print(f"ERROR: court detector v2 eval failed: {exc}", file=sys.stderr)
        return 1


def build_court_detector_v2_eval_report(
    *,
    eval_root: Path,
    include_partial: list[str],
    proposal_root: Path | None = None,
) -> dict[str, Any]:
    clips: list[dict[str, Any]] = []
    for clip in include_partial:
        label_path = eval_root / clip / "labels" / "court_keypoints_partial.json"
        labels = load_partial_court_keypoints(label_path)
        frame = labels.frames[0]
        frame_path = eval_root / clip / "labels" / "court_keypoint_partial_frames" / frame.frame
        if proposal_root is not None:
            proposal_path = proposal_root / clip / "court_detector_v2_proposals.json"
            proposal = json.loads(proposal_path.read_text(encoding="utf-8"))
        else:
            image = _load_image(frame_path)
            proposal = detect_court_v2_from_frame(
                image,
                clip_id=clip,
                source_frame=frame.frame,
                visible_error_px=_proposal_visible_errors(frame.keypoints),
            )
        clips.append(
            {
                "clip": clip,
                "status": "promoted" if proposal["promoted"] else "blocked",
                "partial_label_path": str(label_path),
                "proposal": proposal,
                "visible_keypoint_count": len(frame.keypoints),
                "missing_keypoints": [
                    name
                    for name, visibility in frame.visibility_by_keypoint.items()
                    if visibility == "missing_occluded_or_off_frame"
                ],
            }
        )

    promoted = sum(1 for clip in clips if clip["status"] == "promoted")
    blocked = sum(1 for clip in clips if clip["status"] == "blocked")
    report = {
        "schema_version": 1,
        "artifact_type": "racketsport_court_detector_v2_eval",
        "status": "ran_not_verified",
        "verified": False,
        "not_cal3_verified": True,
        "summary": {
            "full_clip_count": 0,
            "partial_clip_count": len(include_partial),
            "promoted_clip_count": promoted,
            "blocked_clip_count": blocked,
            "needs_user_input_clip_count": sum(1 for clip in clips if clip["proposal"].get("needs_user_input")),
        },
        "clips": clips,
    }
    return report


def _proposal_visible_errors(_visible_keypoints: dict[str, tuple[float, float]]) -> dict[str, Any]:
    # Until detector v2 can recover the court, keep benchmark output blocked
    # against the known IMG_1605-sized residual shape instead of fabricating a pass.
    return {
        "floor_visible": {"median": 581.0, "p95": 786.0},
        "visible_corners": {"median": 604.0},
        "high_confidence_over_30px_count": 0,
    }


def _load_image(path: Path) -> Any:
    import cv2  # type: ignore[import-not-found]

    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"failed to read image: {path}")
    return image


if __name__ == "__main__":
    raise SystemExit(main())

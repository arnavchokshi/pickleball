#!/usr/bin/env python3
"""Render per-clip detector_v2_multiframe overlay images + a contact sheet.

CAL-GEO 2026-07-05: mandatory manager-review artifact for the real Stage
0-6 multi-frame court solver (`court_proposals.propose_court_from_video`).
Draws the chosen representative frame with projected court lines, consensus
keypoints, confidence, and `needs_user_input`/gate status text, then tiles
all rendered clips into one contact sheet image. Never writes
`court_calibration.json` and never claims verification.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.court_finding_technology_benchmark import discover_court_finding_samples  # noqa: E402
from threed.racketsport.court_proposals import propose_court_from_video, select_frames_for_proposals  # noqa: E402

_FLOOR_LINE_ENDPOINTS: dict[str, tuple[str, str]] = {
    "near_baseline": ("near_left_corner", "near_right_corner"),
    "far_baseline": ("far_left_corner", "far_right_corner"),
    "near_nvz": ("near_nvz_left", "near_nvz_right"),
    "far_nvz": ("far_nvz_left", "far_nvz_right"),
    "left_sideline": ("near_left_corner", "far_left_corner"),
    "right_sideline": ("near_right_corner", "far_right_corner"),
    "near_centerline": ("near_baseline_center", "near_nvz_center"),
    "far_centerline": ("far_baseline_center", "far_nvz_center"),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render detector_v2_multiframe overlays + contact sheet.")
    parser.add_argument("--eval-root", type=Path, default=Path("eval_clips/ball"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=24)
    parser.add_argument("--top-k", type=int, default=8)
    return parser.parse_args(argv)


def _representative_frame(video_path: Path | str, *, frame_index: int, max_frames: int, top_k: int) -> Any:
    selection = select_frames_for_proposals(video_path, max_frames=max_frames, top_k=top_k)
    for item in selection["selected"]:
        if item["frame_index"] == frame_index:
            return item["frame"]
    return selection["selected"][0]["frame"] if selection["selected"] else None


def render_clip_overlay(*, clip: str, video_path: Path | str, max_frames: int, top_k: int) -> tuple[Any, dict[str, Any]]:
    import cv2  # type: ignore[import-not-found]

    report = propose_court_from_video(video_path, max_frames=max_frames, top_k=top_k)
    proposals = report.get("proposals") or [{}]
    proposal = proposals[0]
    evidence = proposal.get("evidence") or {}
    # When the fallback safety net fired, the detector_v2 proposal rides
    # second in the list; its evidence still carries the representative frame
    # and verification for review context.
    detector_evidence = evidence
    if evidence.get("fallback_used") and len(proposals) > 1:
        detector_evidence = proposals[1].get("evidence") or {}
    frame_index = int(detector_evidence.get("representative_frame_index") or 0)
    frame = _representative_frame(video_path, frame_index=frame_index, max_frames=max_frames, top_k=top_k)
    if frame is None:
        canvas = 32 * cv2_zeros_placeholder()
        return canvas, {"clip": clip, "status": "no_frame"}
    canvas = frame.copy()

    keypoints = {name: (float(xy[0]), float(xy[1])) for name, xy in (proposal.get("court_keypoints") or {}).items()}
    for line_name, (p1_name, p2_name) in _FLOOR_LINE_ENDPOINTS.items():
        if p1_name in keypoints and p2_name in keypoints:
            p1 = tuple(int(round(v)) for v in keypoints[p1_name])
            p2 = tuple(int(round(v)) for v in keypoints[p2_name])
            cv2.line(canvas, p1, p2, (0, 255, 255), 2, cv2.LINE_AA)
    for name, (x, y) in keypoints.items():
        point = (int(round(x)), int(round(y)))
        cv2.circle(canvas, point, 5, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(canvas, name, (point[0] + 6, point[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    # Runner-up tennis-overlay hypothesis, drawn in magenta for review context.
    runner_up = (detector_evidence.get("runner_up_hypotheses") or [None])[0]
    if runner_up:
        tennis_keypoints = {name: (float(xy[0]), float(xy[1])) for name, xy in (runner_up.get("keypoints") or {}).items()}
        for line_name, (p1_name, p2_name) in _FLOOR_LINE_ENDPOINTS.items():
            if p1_name in tennis_keypoints and p2_name in tennis_keypoints:
                p1 = tuple(int(round(v)) for v in tennis_keypoints[p1_name])
                p2 = tuple(int(round(v)) for v in tennis_keypoints[p2_name])
                cv2.line(canvas, p1, p2, (255, 0, 255), 1, cv2.LINE_AA)

    gate = proposal.get("gate") or {}
    scores = proposal.get("scores") or {}
    verification = detector_evidence.get("verification") or {}
    lines = [
        f"clip: {clip}",
        f"source: {proposal.get('source')} fallback_used: {bool(evidence.get('fallback_used'))}",
        f"auto_usable: {gate.get('auto_usable')} review_usable: {gate.get('review_usable')}",
        f"overall_score: {scores.get('overall')}",
        f"reproj_median_px(evidence): {scores.get('reprojection_px_median')}",
        f"internal_support_score: {detector_evidence.get('internal_support_score')}",
        f"blockers: {','.join(verification.get('blockers') or []) or 'none'}",
        f"frames_with_pickleball_hyp: {detector_evidence.get('frames_with_pickleball_hypothesis')}/{detector_evidence.get('frames_evaluated')}",
    ]
    for index, text in enumerate(lines):
        cv2.putText(canvas, text, (10, 24 + 22 * index), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, text, (10, 24 + 22 * index), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)

    summary = {
        "clip": clip,
        "source": proposal.get("source"),
        "fallback_used": bool(evidence.get("fallback_used")),
        "internal_support_score": detector_evidence.get("internal_support_score"),
        "gate": gate,
        "scores": scores,
        "blockers": verification.get("blockers"),
        "representative_frame_index": frame_index,
    }
    return canvas, summary


def cv2_zeros_placeholder() -> Any:
    import numpy as np

    return np.ones((360, 640, 3), dtype="uint8")


def build_contact_sheet(images: list[Any], *, columns: int = 2) -> Any:
    import cv2  # type: ignore[import-not-found]
    import numpy as np

    if not images:
        raise ValueError("no overlay images to build a contact sheet from")
    tile_h, tile_w = 480, 854
    resized = [cv2.resize(image, (tile_w, tile_h)) for image in images]
    rows = -(-len(resized) // columns)
    sheet = np.zeros((rows * tile_h, columns * tile_w, 3), dtype=np.uint8)
    for index, tile in enumerate(resized):
        row, col = divmod(index, columns)
        sheet[row * tile_h : (row + 1) * tile_h, col * tile_w : (col + 1) * tile_w] = tile
    return sheet


def main(argv: list[str] | None = None) -> int:
    import cv2  # type: ignore[import-not-found]

    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    samples = discover_court_finding_samples(args.eval_root)

    images = []
    summaries: list[dict[str, Any]] = []
    for sample in samples:
        try:
            canvas, summary = render_clip_overlay(
                clip=sample.clip, video_path=sample.frame_input, max_frames=args.max_frames, top_k=args.top_k
            )
        except Exception as exc:  # pragma: no cover - defensive; report per-clip failure honestly
            summary = {"clip": sample.clip, "status": "error", "error": str(exc)}
            canvas = cv2_zeros_placeholder()
        overlay_path = args.out_dir / f"{sample.clip}_overlay.jpg"
        cv2.imwrite(str(overlay_path), canvas)
        summary["overlay_path"] = str(overlay_path)
        summaries.append(summary)
        images.append(canvas)

    if images:
        sheet = build_contact_sheet(images)
        cv2.imwrite(str(args.out_dir / "contact_sheet.jpg"), sheet)

    manifest = {
        "schema_version": 1,
        "artifact_type": "racketsport_court_detector_v2_multiframe_overlay_manifest",
        "verified": False,
        "not_cal3_verified": True,
        "clips": summaries,
        "contact_sheet": str(args.out_dir / "contact_sheet.jpg") if images else None,
    }
    (args.out_dir / "overlay_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

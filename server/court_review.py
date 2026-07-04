from __future__ import annotations

from pathlib import Path
from typing import Any

from threed.racketsport.court_detector_v2 import detect_court_v2_from_frame
from threed.racketsport.court_detector_v2_hypotheses import _template_projection
from threed.racketsport.net_anchor_court import load_player_suppressed_frame
from threed.racketsport.schemas import PICKLEBALL_COURT_KEYPOINT_NAMES


def predict_court_layout_from_video(
    *,
    video_path: Path,
    clip: str,
    frame_index: int | None = None,
) -> dict[str, Any]:
    frame, frame_meta = load_player_suppressed_frame(
        video_path,
        max_frames=1 if frame_index is not None else 72,
        stride=1 if frame_index is not None else 6,
        start_frame=max(0, int(frame_index or 0)),
    )
    detected = detect_court_v2_from_frame(
        frame,
        clip_id=clip,
        source_frame=str(frame_meta.get("source_frame") or video_path.name),
    )
    height, width = int(frame.shape[0]), int(frame.shape[1])
    selected = _selected_hypothesis(detected)
    selected_points = dict(selected.get("projected_keypoints") or {}) if selected else _template_projection(width=width, height=height)
    score_components = dict(selected.get("score_components") or {}) if selected else {}
    confidence = float(score_components.get("evidence_score", 0.0))
    frame_number = int(frame_meta.get("frame_index") or frame_index or 0)
    fps = float(frame_meta.get("fps") or 30.0)
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_court_layout_prediction",
        "clip": clip,
        "image_size": [width, height],
        "frame_index": frame_number,
        "frame_time_s": frame_number / fps if fps > 0 else 0.0,
        "prediction_source": f"court_detector_v2:selected_hypothesis={detected.get('selected_hypothesis_id') or 'none'}",
        "verified": False,
        "not_cal3_verified": True,
        "promotion_status": detected.get("promotion_status", "needs_user_input"),
        "promotion_blockers": list(detected.get("promotion_blockers") or []),
        "points": {
            name: {"xy": [float(selected_points[name][0]), float(selected_points[name][1])], "confidence": max(0.0, min(1.0, confidence))}
            for name in PICKLEBALL_COURT_KEYPOINT_NAMES
            if name in selected_points
        },
        "lines": _lines_from_points(selected_points),
        "warnings": ["auto_court_detection_preview_not_verified"],
        "detector_proposal": detected,
    }


def _selected_hypothesis(proposal: dict[str, Any]) -> dict[str, Any] | None:
    selected_id = proposal.get("selected_hypothesis_id")
    hypotheses = proposal.get("hypotheses")
    if not isinstance(hypotheses, list):
        return None
    for hypothesis in hypotheses:
        if isinstance(hypothesis, dict) and hypothesis.get("hypothesis_id") == selected_id:
            return hypothesis
    return next((hypothesis for hypothesis in hypotheses if isinstance(hypothesis, dict)), None)


def _lines_from_points(points: dict[str, Any]) -> list[dict[str, Any]]:
    pairs = {
        "near_baseline": ("near_left_corner", "near_right_corner"),
        "far_baseline": ("far_left_corner", "far_right_corner"),
        "left_sideline": ("near_left_corner", "far_left_corner"),
        "right_sideline": ("near_right_corner", "far_right_corner"),
        "near_nvz": ("near_nvz_left", "near_nvz_right"),
        "far_nvz": ("far_nvz_left", "far_nvz_right"),
        "net": ("net_left_sideline", "net_right_sideline"),
    }
    lines: list[dict[str, Any]] = []
    for line_id, (start, end) in pairs.items():
        if start in points and end in points:
            lines.append({"id": line_id, "points": [points[start], points[end]]})
    return lines

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
from typing import Any

from threed.racketsport.court_detector_v2_hypotheses import _template_projection
from threed.racketsport.schemas import PICKLEBALL_COURT_KEYPOINT_NAMES


def predict_court_layout_from_video(
    *,
    video_path: Path,
    clip: str,
    frame_index: int | None = None,
) -> dict[str, Any]:
    mode = os.environ.get("PICKLEBALL_COURT_PREDICTOR_MODE", "template").strip().lower()
    if mode in {"detector", "court_detector_v2"}:
        try:
            return _predict_with_detector(video_path=video_path, clip=clip, frame_index=frame_index)
        except Exception as exc:
            if os.environ.get("PICKLEBALL_COURT_PREDICTOR_STRICT", "").strip() == "1":
                raise
            return _template_prediction(
                video_path=video_path,
                clip=clip,
                frame_index=frame_index,
                warnings=[
                    "auto_court_detection_preview_not_verified",
                    f"court_detector_v2_failed_fell_back_to_template:{type(exc).__name__}",
                ],
            )
    return _template_prediction(video_path=video_path, clip=clip, frame_index=frame_index)


def _predict_with_detector(*, video_path: Path, clip: str, frame_index: int | None = None) -> dict[str, Any]:
    from threed.racketsport.court_detector_v2 import detect_court_v2_from_frame
    from threed.racketsport.net_anchor_court import load_player_suppressed_frame

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


def _template_prediction(
    *,
    video_path: Path,
    clip: str,
    frame_index: int | None = None,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    metadata = _probe_video_metadata(video_path)
    width = int(metadata.get("width") or 1280)
    height = int(metadata.get("height") or 720)
    fps = float(metadata.get("fps") or 30.0)
    resolved_frame_index = max(0, int(frame_index or 0))
    selected_points = _template_projection(width=width, height=height)
    prediction_warnings = warnings or [
        "auto_court_detection_preview_not_verified",
        "template_seed_requires_user_review",
    ]
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_court_layout_prediction",
        "clip": clip,
        "image_size": [width, height],
        "frame_index": resolved_frame_index,
        "frame_time_s": resolved_frame_index / fps if fps > 0 else 0.0,
        "prediction_source": "template_projection_seed:ffprobe_metadata",
        "verified": False,
        "not_cal3_verified": True,
        "promotion_status": "needs_user_input",
        "promotion_blockers": ["template_seed_not_automatic_detection", "manual_review_required"],
        "points": {
            name: {"xy": [float(selected_points[name][0]), float(selected_points[name][1])], "confidence": 0.2}
            for name in PICKLEBALL_COURT_KEYPOINT_NAMES
            if name in selected_points
        },
        "lines": _lines_from_points(selected_points),
        "warnings": prediction_warnings,
        "detector_proposal": {
            "source": "template_projection_seed",
            "video_metadata": metadata,
            "promotion_allowed": False,
        },
    }


def _probe_video_metadata(video_path: Path) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,avg_frame_rate,r_frame_rate,duration,nb_frames",
                "-of",
                "json",
                str(video_path),
            ],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return {}
    if completed.returncode != 0:
        return {}
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return {}
    streams = payload.get("streams")
    if not isinstance(streams, list) or not streams or not isinstance(streams[0], dict):
        return {}
    stream = streams[0]
    return {
        "width": _positive_int(stream.get("width")),
        "height": _positive_int(stream.get("height")),
        "fps": _fps_from_stream(stream),
        "duration_s": _positive_float(stream.get("duration")),
        "frame_count": _positive_int(stream.get("nb_frames")),
    }


def _fps_from_stream(stream: dict[str, Any]) -> float | None:
    for key in ("avg_frame_rate", "r_frame_rate"):
        value = stream.get(key)
        if not isinstance(value, str) or "/" not in value:
            continue
        numerator, denominator = value.split("/", 1)
        try:
            fps = float(numerator) / float(denominator)
        except (ValueError, ZeroDivisionError):
            continue
        if fps > 0:
            return fps
    return None


def _positive_int(value: object) -> int | None:
    try:
        parsed = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _positive_float(value: object) -> float | None:
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


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

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import subprocess
from typing import Any

from threed.racketsport.court_detector_v2_hypotheses import _template_projection
from threed.racketsport.schemas import PICKLEBALL_COURT_KEYPOINT_NAMES

# Below this per-point confidence, a keypoint is flagged for mandatory human review
# rather than trusted as an unattended auto-prediction. This threshold is deliberately
# permissive (fails toward "needs review") since nothing produced here is ever promoted
# to a trusted channel without an explicit human Confirm in the web/iOS review UI.
NEEDS_REVIEW_CONFIDENCE_THRESHOLD = 0.5

DEFAULT_ASSIST_SEED: dict[str, Any] = {"mode": "none", "tap_points": [], "line_label": None}


def predict_court_layout_from_video(
    *,
    video_path: Path,
    clip: str,
    frame_index: int | None = None,
) -> dict[str, Any]:
    mode = os.environ.get("PICKLEBALL_COURT_PREDICTOR_MODE", "proposals").strip().lower()
    if mode in {"proposals", "court_proposals"}:
        prediction = _predict_with_proposals_or_fallback(video_path=video_path, clip=clip, frame_index=frame_index)
    elif mode in {"detector", "court_detector_v2"}:
        prediction = _predict_with_detector_or_fallback(video_path=video_path, clip=clip, frame_index=frame_index)
    else:
        prediction = _template_prediction(video_path=video_path, clip=clip, frame_index=frame_index)
    return _finalize_prediction(prediction, video_path=video_path)


def _finalize_prediction(prediction: dict[str, Any], *, video_path: Path) -> dict[str, Any]:
    """Apply cross-mode invariants: needs_user_input, assist default, and preview frame."""

    finalized = dict(prediction)
    points = finalized.get("points") if isinstance(finalized.get("points"), dict) else {}
    finalized.setdefault("needs_user_input", _needs_user_input_from_confidence(points))
    finalized.setdefault("assist", dict(DEFAULT_ASSIST_SEED))
    frame_index = int(finalized.get("frame_index") or 0)
    finalized.setdefault("preview_frame_index", frame_index)
    finalized["preview_frame_jpeg_base64"] = _load_preview_frame_jpeg_base64(video_path, frame_index=frame_index)
    return finalized


def _needs_user_input_from_confidence(points: dict[str, Any]) -> list[str]:
    return [
        name
        for name in PICKLEBALL_COURT_KEYPOINT_NAMES
        if float((points.get(name) or {}).get("confidence", 0.0)) < NEEDS_REVIEW_CONFIDENCE_THRESHOLD
    ]


def _load_preview_frame_jpeg_base64(video_path: Path, *, frame_index: int) -> str | None:
    """Best-effort preview frame extraction for the review UI. Never raises."""

    try:
        import cv2

        from threed.racketsport.net_anchor_court import load_player_suppressed_frame

        frame, _frame_meta = load_player_suppressed_frame(
            video_path,
            max_frames=1,
            stride=1,
            start_frame=max(0, int(frame_index)),
        )
        ok, buffer = cv2.imencode(".jpg", frame)
        if not ok:
            return None
        return base64.b64encode(buffer.tobytes()).decode("ascii")
    except Exception:
        return None


def _predict_with_proposals_or_fallback(
    *, video_path: Path, clip: str, frame_index: int | None = None
) -> dict[str, Any]:
    try:
        return _predict_with_proposals(video_path=video_path, clip=clip, frame_index=frame_index)
    except Exception as exc:
        if os.environ.get("PICKLEBALL_COURT_PREDICTOR_STRICT", "").strip() == "1":
            raise
        return _template_prediction(
            video_path=video_path,
            clip=clip,
            frame_index=frame_index,
            warnings=[
                "auto_court_detection_preview_not_verified",
                "proposal_pipeline_unavailable",
                f"court_proposals_failed:{type(exc).__name__}",
            ],
        )


def _predict_with_proposals(*, video_path: Path, clip: str, frame_index: int | None = None) -> dict[str, Any]:
    from threed.racketsport.court_proposals import propose_court_from_video

    report = propose_court_from_video(video_path, max_frames=24, top_k=8)
    if not isinstance(report, dict):
        raise TypeError("propose_court_from_video must return a dict")
    if report.get("verified") is not False or report.get("not_cal3_verified") is not True:
        raise ValueError("court proposal report must remain fail-closed (verified=False, not_cal3_verified=True)")

    metadata = _probe_video_metadata(video_path)
    input_meta = report.get("input") if isinstance(report.get("input"), dict) else {}
    width, height = _report_image_size(input_meta, metadata)
    frame_indices = [int(value) for value in (input_meta.get("frame_indices") or []) if isinstance(value, (int, float))]
    chosen_frame_index = frame_indices[0] if frame_indices else max(0, int(frame_index or 0))
    fps = float(metadata.get("fps") or 30.0)

    proposals = [proposal for proposal in (report.get("proposals") or []) if isinstance(proposal, dict)]
    ranking = report.get("ranking") if isinstance(report.get("ranking"), dict) else {}
    selected_id = ranking.get("selected_proposal_id")
    selected = next((proposal for proposal in proposals if proposal.get("proposal_id") == selected_id), None)
    if selected is None:
        selected = proposals[0] if proposals else None
    if not isinstance(selected, dict):
        raise ValueError("court proposal report has no usable proposals")

    keypoints = selected.get("court_keypoints")
    if not isinstance(keypoints, dict) or not all(name in keypoints for name in PICKLEBALL_COURT_KEYPOINT_NAMES):
        raise ValueError("selected court proposal is missing one or more required keypoints")

    confidence_by_point = _proposal_point_confidence(selected)
    points = {
        name: {
            "xy": [float(keypoints[name][0]), float(keypoints[name][1])],
            "confidence": confidence_by_point.get(name, 0.0),
        }
        for name in PICKLEBALL_COURT_KEYPOINT_NAMES
    }

    gate = selected.get("gate") if isinstance(selected.get("gate"), dict) else {}
    declared_needs_review = report.get("needs_user_input")
    needs_user_input = (
        [str(name) for name in declared_needs_review if str(name) in PICKLEBALL_COURT_KEYPOINT_NAMES]
        if isinstance(declared_needs_review, list) and declared_needs_review
        else _needs_user_input_from_confidence(points)
    )
    runner_ups = [_proposal_summary(proposal) for proposal in proposals if proposal is not selected]

    return {
        "schema_version": 1,
        "artifact_type": "racketsport_court_layout_prediction",
        "clip": clip,
        "image_size": [width, height],
        "frame_index": chosen_frame_index,
        "frame_time_s": chosen_frame_index / fps if fps > 0 else 0.0,
        "prediction_source": f"court_proposals:selected_proposal={selected.get('proposal_id') or 'unknown'}",
        "verified": False,
        "not_cal3_verified": True,
        "promotion_status": "needs_user_input",
        "promotion_blockers": [str(item) for item in (gate.get("failed") or ["not_verified"])],
        "points": points,
        "lines": _lines_from_points({name: point["xy"] for name, point in points.items()}),
        "warnings": ["auto_court_detection_preview_not_verified"],
        "needs_user_input": needs_user_input,
        "assist": _assist_from_report(report),
        "selected_proposal_id": selected.get("proposal_id"),
        "proposals": [_proposal_summary(selected)] + runner_ups,
        "proposal_report": report,
    }


def _proposal_summary(proposal: dict[str, Any]) -> dict[str, Any]:
    gate = proposal.get("gate") if isinstance(proposal.get("gate"), dict) else {}
    return {
        "proposal_id": proposal.get("proposal_id"),
        "source": proposal.get("source"),
        "scores": proposal.get("scores") if isinstance(proposal.get("scores"), dict) else {},
        "review_usable": bool(gate.get("review_usable", False)),
    }


def _proposal_point_confidence(proposal: dict[str, Any]) -> dict[str, float]:
    for key in ("point_confidence", "keypoint_confidence", "per_keypoint_confidence"):
        candidate = proposal.get(key)
        confidence = _unit_confidence_map(candidate)
        if confidence:
            return confidence
    evidence = proposal.get("evidence") if isinstance(proposal.get("evidence"), dict) else {}
    confidence = _unit_confidence_map(evidence.get("point_confidence"))
    if confidence:
        return confidence

    scores = proposal.get("scores") if isinstance(proposal.get("scores"), dict) else {}
    overall = scores.get("overall") if isinstance(scores.get("overall"), (int, float)) else scores.get("evidence_score")
    uniform = _clamp_unit(overall) if isinstance(overall, (int, float)) and not isinstance(overall, bool) else 0.3
    return {name: uniform for name in PICKLEBALL_COURT_KEYPOINT_NAMES}


def _unit_confidence_map(candidate: Any) -> dict[str, float]:
    if not isinstance(candidate, dict):
        return {}
    return {
        name: _clamp_unit(value)
        for name, value in candidate.items()
        if name in PICKLEBALL_COURT_KEYPOINT_NAMES and isinstance(value, (int, float)) and not isinstance(value, bool)
    }


def _clamp_unit(value: Any) -> float:
    return max(0.0, min(1.0, float(value)))


def _assist_from_report(report: dict[str, Any]) -> dict[str, Any]:
    assist = report.get("assist")
    if not isinstance(assist, dict):
        return dict(DEFAULT_ASSIST_SEED)
    tap_points = assist.get("tap_points")
    return {
        "mode": str(assist.get("mode")) if isinstance(assist.get("mode"), str) else "none",
        "tap_points": [
            [float(point[0]), float(point[1])]
            for point in tap_points
            if isinstance(point, (list, tuple)) and len(point) == 2
        ]
        if isinstance(tap_points, list)
        else [],
        "line_label": assist.get("line_label") if isinstance(assist.get("line_label"), str) else None,
    }


def _report_image_size(input_meta: dict[str, Any], metadata: dict[str, Any]) -> tuple[int, int]:
    image_size = input_meta.get("image_size")
    if isinstance(image_size, (list, tuple)) and len(image_size) == 2:
        try:
            width, height = int(image_size[0]), int(image_size[1])
            if width > 0 and height > 0:
                return width, height
        except (TypeError, ValueError):
            pass
    return int(metadata.get("width") or 1280), int(metadata.get("height") or 720)


def _predict_with_detector_or_fallback(
    *, video_path: Path, clip: str, frame_index: int | None = None
) -> dict[str, Any]:
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

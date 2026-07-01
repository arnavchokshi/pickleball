"""M5 uncertainty-gated in/out report for BALL-only tracking."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .io_decode import FrameSource, probe_clip
from .schemas import BallTrack, validate_artifact_file


M5_STATUS_TESTED = "TESTED-ON-REAL-DATA"
M5_STATUS_SCAFFOLD = "SCAFFOLD"
MIN_CONFIDENT_AGREEMENT = 0.95
MAX_REVIEW_DELTA_FRAMES = 2.0


def build_ball_inout_gate_report(
    *,
    ball_track_path: str | Path,
    video_path: str | Path | None = None,
    m4_bounce_report_path: str | Path | None = None,
    reviewed_inout_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build a fail-closed M5 report from per-bounce uncertainty calls."""

    video = probe_clip(video_path) if video_path is not None else None
    track_path = Path(ball_track_path)
    track = validate_artifact_file("ball_track", track_path)
    if not isinstance(track, BallTrack):
        raise ValueError(f"{track_path} did not validate as BallTrack")

    m4_report = _load_optional_json(m4_bounce_report_path)
    reviewed = _load_optional_json(reviewed_inout_path)
    call_checks = _call_checks(track)
    agreement = _review_agreement(track, reviewed=reviewed)
    near_far_split = _near_far_split(track)

    violations: list[str] = []
    _extend_unique(violations, _m4_violations(m4_report, ball_track_path=track_path, bounce_count=len(track.bounces)))
    _extend_unique(violations, call_checks["violations"])
    _extend_unique(violations, agreement["violations"])

    total_bounces = len(track.bounces)
    gray_count = call_checks["gray_call_count"]
    confident_count = call_checks["confident_call_count"]

    return {
        "schema_version": 1,
        "artifact_type": "racketsport_ball_inout_gate_report",
        "milestone": "M5 In/out",
        "status": M5_STATUS_TESTED if video is not None else M5_STATUS_SCAFFOLD,
        "gate_result": "fail" if violations else "pass",
        "blocked_reason": "ball_inout_gate_failed" if violations else None,
        "ball_track_path": str(track_path),
        "video": _video_summary(video) if video is not None else None,
        "m4_bounce_gate": _m4_summary(m4_report),
        "source": track.source,
        "fps": float(track.fps),
        "frame_count": len(track.frames),
        "bounce_count": total_bounces,
        "confident_call_count": confident_count,
        "gray_call_count": gray_count,
        "gray_zone_rate": _ratio(gray_count, total_bounces),
        "near_far_split": near_far_split,
        "required_thresholds": {
            "confident_agreement_min": MIN_CONFIDENT_AGREEMENT,
            "max_review_delta_frames": MAX_REVIEW_DELTA_FRAMES,
        },
        "calls": call_checks["calls"],
        "review_agreement": agreement["summary"],
        "violations": violations,
        "not_ground_truth": True,
    }


def write_ball_inout_gate_report(
    *,
    ball_track_path: str | Path,
    out: str | Path,
    video_path: str | Path | None = None,
    m4_bounce_report_path: str | Path | None = None,
    reviewed_inout_path: str | Path | None = None,
) -> dict[str, Any]:
    report = build_ball_inout_gate_report(
        ball_track_path=ball_track_path,
        video_path=video_path,
        m4_bounce_report_path=m4_bounce_report_path,
        reviewed_inout_path=reviewed_inout_path,
    )
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _m4_violations(
    m4_report: Mapping[str, Any] | None,
    *,
    ball_track_path: Path,
    bounce_count: int,
) -> list[str]:
    if m4_report is None:
        return ["missing_m4_bounce_gate_report"]
    violations: list[str] = []
    if m4_report.get("artifact_type") != "racketsport_ball_bounce_gate_report":
        violations.append("m4_bounce_gate_artifact_type_invalid")
    if m4_report.get("gate_result") != "pass":
        violations.append("m4_bounce_gate_not_passed")
    m4_ball_track_path = m4_report.get("ball_track_path")
    if not isinstance(m4_ball_track_path, str) or not m4_ball_track_path:
        violations.append("m4_bounce_gate_ball_track_missing")
    elif not _paths_match(m4_ball_track_path, ball_track_path):
        violations.append("m4_bounce_gate_ball_track_mismatch")
    m4_bounce_count = m4_report.get("bounce_count")
    if not isinstance(m4_bounce_count, int):
        violations.append("m4_bounce_gate_bounce_count_missing")
    elif m4_bounce_count != bounce_count:
        violations.append("m4_bounce_gate_bounce_count_mismatch")
    if m4_report.get("status") != M5_STATUS_TESTED:
        violations.append("m4_bounce_gate_not_tested_on_real_data")
    return sorted(set(violations))


def _m4_summary(m4_report: Mapping[str, Any] | None) -> dict[str, Any]:
    if m4_report is None:
        return {"path_present": False, "gate_result": None, "blocked_reason": None}
    return {
        "path_present": True,
        "gate_result": m4_report.get("gate_result"),
        "blocked_reason": m4_report.get("blocked_reason"),
        "status": m4_report.get("status"),
    }


def _call_checks(track: BallTrack) -> dict[str, Any]:
    violations: list[str] = []
    calls: list[dict[str, Any]] = []
    confident_count = 0
    gray_count = 0

    if not track.bounces:
        return {
            "calls": [],
            "confident_call_count": 0,
            "gray_call_count": 0,
            "violations": ["ball_track_has_no_bounces", "no_confident_inout_calls"],
        }

    for bounce in track.bounces:
        call = bounce.call
        margin = bounce.margin_m
        uncertainty = bounce.uncertainty_m
        confidence = bounce.confidence
        expected = _expected_call(margin, uncertainty)
        frame = bounce.frame if bounce.frame is not None else round(float(bounce.t) * float(track.fps))

        if call is None:
            violations.append("inout_call_missing")
        elif call == "too_close_to_call":
            gray_count += 1
        else:
            confident_count += 1

        if margin is None:
            violations.append("margin_m_missing")
        if uncertainty is None:
            violations.append("uncertainty_m_missing")
        if confidence is None:
            violations.append("inout_confidence_missing")
        if bounce.nearest_line is None:
            violations.append("nearest_line_missing")
        if bounce.region is None:
            violations.append("region_missing")
        if bounce.dominant_uncertainty_term is None:
            violations.append("dominant_uncertainty_term_missing")
        if expected is not None and call != expected:
            violations.append("inout_call_violates_uncertainty_rule")
        if margin is not None and uncertainty is not None and confidence is not None:
            expected_confidence = _expected_confidence(float(margin), float(uncertainty))
            if not math.isclose(float(confidence), expected_confidence, rel_tol=1e-3, abs_tol=1e-3):
                violations.append("inout_confidence_mismatch")

        calls.append(
            {
                "t": float(bounce.t),
                "frame": int(frame),
                "call": call,
                "margin_m": float(margin) if margin is not None else None,
                "uncertainty_m": float(uncertainty) if uncertainty is not None else None,
                "confidence": float(confidence) if confidence is not None else None,
                "nearest_line": bounce.nearest_line,
                "region": bounce.region,
                "dominant_uncertainty_term": bounce.dominant_uncertainty_term,
                "expected_call": expected,
            }
        )

    if confident_count == 0:
        violations.append("no_confident_inout_calls")
    return {
        "calls": calls,
        "confident_call_count": confident_count,
        "gray_call_count": gray_count,
        "violations": sorted(set(violations)),
    }


def _review_agreement(track: BallTrack, *, reviewed: Mapping[str, Any] | None) -> dict[str, Any]:
    confident = [
        {
            "frame": int(bounce.frame) if bounce.frame is not None else round(float(bounce.t) * float(track.fps)),
            "t": float(bounce.t),
            "call": bounce.call,
        }
        for bounce in track.bounces
        if bounce.call in {"in", "out"}
    ]
    if reviewed is None:
        return {
            "summary": {
                "reviewed_call_count": 0,
                "confident_call_count": len(confident),
                "matched_confident_call_count": 0,
                "confident_agreement_count": 0,
                "confident_agreement_rate": None,
                "matches": [],
            },
            "violations": ["missing_reviewed_inout_labels"],
        }

    reviewed_calls = _reviewed_calls(reviewed, fps=float(track.fps))
    violations: list[str] = []
    if reviewed.get("artifact_type") != "racketsport_reviewed_ball_inout":
        violations.append("reviewed_inout_artifact_type_invalid")
    if not reviewed_calls:
        violations.append("reviewed_inout_labels_empty")
    matches, unmatched_confident = _match_confident_calls(confident, reviewed_calls)
    agreement_count = sum(1 for match in matches if match["predicted_call"] == match["reviewed_call"])
    agreement_rate = _ratio(agreement_count, len(confident))
    if unmatched_confident:
        violations.append("confident_inout_call_unmatched")
    if confident and (agreement_rate is None or agreement_rate < MIN_CONFIDENT_AGREEMENT):
        violations.append("confident_inout_agreement_below_0_95")
    return {
        "summary": {
            "reviewed_call_count": len(reviewed_calls),
            "confident_call_count": len(confident),
            "matched_confident_call_count": len(matches),
            "confident_agreement_count": agreement_count,
            "confident_agreement_rate": agreement_rate,
            "matches": matches,
            "unmatched_confident_calls": unmatched_confident,
        },
        "violations": sorted(set(violations)),
    }


def _match_confident_calls(
    predicted: Sequence[dict[str, Any]],
    reviewed: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidate_pairs: list[tuple[float, int, int, float]] = []
    for pred_idx, pred in enumerate(predicted):
        for review_idx, truth in enumerate(reviewed):
            signed_delta = float(pred["frame"]) - float(truth["frame"])
            abs_delta = abs(signed_delta)
            if abs_delta <= MAX_REVIEW_DELTA_FRAMES + 1e-9:
                candidate_pairs.append((abs_delta, pred_idx, review_idx, signed_delta))

    used_predicted: set[int] = set()
    used_reviewed: set[int] = set()
    matches: list[dict[str, Any]] = []
    for _, pred_idx, review_idx, signed_delta in sorted(candidate_pairs):
        if pred_idx in used_predicted or review_idx in used_reviewed:
            continue
        used_predicted.add(pred_idx)
        used_reviewed.add(review_idx)
        pred = predicted[pred_idx]
        truth = reviewed[review_idx]
        matches.append(
            {
                "predicted_frame": pred["frame"],
                "reviewed_frame": truth["frame"],
                "signed_delta_frames": signed_delta,
                "predicted_call": pred["call"],
                "reviewed_call": truth["call"],
            }
        )
    unmatched = [pred for idx, pred in enumerate(predicted) if idx not in used_predicted]
    return matches, unmatched


def _reviewed_calls(payload: Mapping[str, Any], *, fps: float) -> list[dict[str, Any]]:
    calls = payload.get("calls")
    if not isinstance(calls, list):
        return []
    parsed: list[dict[str, Any]] = []
    for item in calls:
        if not isinstance(item, Mapping):
            continue
        call = item.get("call")
        if call not in {"in", "out"}:
            continue
        frame = item.get("frame")
        t = _finite_or_none(item.get("t"))
        frame_index = int(frame) if isinstance(frame, int) else round(float(t) * fps) if t is not None else None
        if frame_index is None or frame_index < 0:
            continue
        parsed.append({"frame": frame_index, "t": float(t) if t is not None else frame_index / fps, "call": call})
    return parsed


def _near_far_split(track: BallTrack) -> dict[str, dict[str, int]]:
    split: dict[str, dict[str, int]] = {
        "near": {"total": 0, "confident": 0, "gray": 0},
        "far": {"total": 0, "confident": 0, "gray": 0},
    }
    for bounce in track.bounces:
        bucket = "far" if (bounce.region or "").lower().startswith("far") else "near"
        split[bucket]["total"] += 1
        if bounce.call == "too_close_to_call":
            split[bucket]["gray"] += 1
        elif bounce.call in {"in", "out"}:
            split[bucket]["confident"] += 1
    return split


def _expected_call(margin: Any, uncertainty: Any) -> str | None:
    if not _finite_like(margin) or not _finite_like(uncertainty):
        return None
    margin_f = float(margin)
    uncertainty_f = float(uncertainty)
    if margin_f > uncertainty_f:
        return "in"
    if margin_f < -uncertainty_f:
        return "out"
    return "too_close_to_call"


def _expected_confidence(margin: float, uncertainty: float) -> float:
    denominator = abs(margin) + uncertainty
    if denominator <= 0.0:
        return 0.0
    return max(0.0, min(1.0, abs(margin) / denominator))


def _load_optional_json(path: str | Path | None) -> Mapping[str, Any] | None:
    if path is None:
        return None
    json_path = Path(path)
    if not json_path.is_file():
        return None
    with json_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{json_path} must contain a JSON object")
    return payload


def _video_summary(video: FrameSource) -> dict[str, Any]:
    return {
        "path": str(video.path),
        "resolution": [int(video.width), int(video.height)],
        "fps": float(video.fps),
        "duration_s": float(video.duration_s),
        "frame_count": video.frame_count,
        "audio_present": video.audio_sample_rate is not None,
        "audio_sample_rate": video.audio_sample_rate,
    }


def _finite_like(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, int | float) and math.isfinite(float(value))


def _finite_or_none(value: Any) -> float | None:
    return float(value) if _finite_like(value) else None


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return float(numerator) / float(denominator)


def _extend_unique(target: list[str], values: Sequence[str]) -> None:
    for value in values:
        if value not in target:
            target.append(value)


def _paths_match(left: str, right: Path) -> bool:
    try:
        return Path(left).resolve() == right.resolve()
    except OSError:
        return str(left) == str(right)


__all__ = [
    "M5_STATUS_SCAFFOLD",
    "M5_STATUS_TESTED",
    "build_ball_inout_gate_report",
    "write_ball_inout_gate_report",
]

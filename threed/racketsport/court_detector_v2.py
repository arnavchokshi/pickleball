"""Court detector v2 orchestration."""

from __future__ import annotations

from typing import Any

from .court_detector_v2_hypotheses import generate_court_hypotheses
from .court_detector_v2_net import detect_court_net_evidence
from .court_detector_v2_schema import build_blocked_detector_v2_proposal, build_promoted_detector_v2_proposal
from .court_detector_v2_surface import build_surface_paint_evidence
from .court_detector_v2_verify import verify_court_hypothesis

DEFAULT_NEEDS_USER_INPUT = ("near_left_corner", "near_right_corner", "far_right_corner", "far_left_corner")


def detect_court_v2_from_frame(
    frame_bgr: Any,
    *,
    clip_id: str = "",
    source_frame: str | None = None,
    visible_error_px: dict[str, Any] | None = None,
    temporal_stability_px: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if frame_bgr is None or not hasattr(frame_bgr, "shape") or len(frame_bgr.shape) < 2:
        raise ValueError("frame_bgr must be an image array")
    height, width = int(frame_bgr.shape[0]), int(frame_bgr.shape[1])
    if width <= 0 or height <= 0:
        raise ValueError("frame_bgr must have positive dimensions")

    net_evidence = detect_court_net_evidence(frame_bgr)
    surface_evidence = build_surface_paint_evidence(frame_bgr, roi=net_evidence.get("roi"))
    hypotheses = generate_court_hypotheses(
        image_size=(width, height),
        net_evidence=net_evidence,
        surface_evidence=surface_evidence,
    )

    verified_hypotheses: list[dict[str, Any]] = []
    for hypothesis in hypotheses:
        line_support = dict(hypothesis.get("line_support") or {})
        verification = verify_court_hypothesis(
            hypothesis=hypothesis,
            visible_error_px=visible_error_px,
            line_support=line_support,
            temporal_stability_px=temporal_stability_px or {"median": 0.0},
            top_net_validation={"passed": bool(net_evidence.get("top_tape_line"))},
            tennis_overlay_rejection={"passed": False},
        )
        item = {**hypothesis, "verification": verification, "promotion_allowed": bool(verification["promotion_allowed"])}
        verified_hypotheses.append(item)

    selected = _select_hypothesis(verified_hypotheses)
    selected_id = str(selected.get("hypothesis_id")) if selected else None
    selected_verification = dict(selected.get("verification") or {}) if selected else {}
    blockers = list(selected_verification.get("blockers") or ["no_detector_v2_hypothesis"])
    clip = clip_id or "unknown_clip"

    if selected and bool(selected_verification.get("promotion_allowed", False)):
        return build_promoted_detector_v2_proposal(
            clip=clip,
            source_frame=source_frame,
            image_size=(width, height),
            selected_hypothesis_id=selected_id or "hypothesis_0000",
            hypotheses=verified_hypotheses,
            net_evidence=net_evidence,
            surface_evidence=surface_evidence,
            verification=selected_verification,
        )

    return build_blocked_detector_v2_proposal(
        clip=clip,
        source_frame=source_frame,
        image_size=(width, height),
        blockers=blockers,
        needs_user_input=DEFAULT_NEEDS_USER_INPUT,
        selected_hypothesis_id=selected_id,
        hypotheses=verified_hypotheses,
        net_evidence=net_evidence,
        surface_evidence=surface_evidence,
        verification=selected_verification,
    )


def _select_hypothesis(hypotheses: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not hypotheses:
        return None

    def score(item: dict[str, Any]) -> tuple[bool, float]:
        components = item.get("score_components") or {}
        return (bool(item.get("promotion_allowed")), float(components.get("evidence_score", 0.0)))

    return max(hypotheses, key=score)

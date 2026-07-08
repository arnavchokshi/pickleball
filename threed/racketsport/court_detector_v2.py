"""Court detector v2 orchestration."""

from __future__ import annotations

from typing import Any

from .court_detector_v2_hypotheses import generate_court_hypotheses, generate_homography_hypotheses
from .court_detector_v2_net import detect_court_net_evidence
from .court_detector_v2_schema import build_blocked_detector_v2_proposal, build_promoted_detector_v2_proposal
from .court_detector_v2_surface import build_surface_paint_evidence
from .court_detector_v2_verify import (
    compute_tennis_overlay_rejection,
    compute_top_net_validation,
    compute_visible_error_px_against_evidence,
    verify_court_hypothesis,
)

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
    try:
        real_hypotheses = generate_homography_hypotheses(
            frame_bgr, net_evidence=net_evidence, surface_evidence=surface_evidence, max_hypotheses=8
        )
    except Exception:
        real_hypotheses = []
    # Only pickleball-tagged real hypotheses are eligible to become THIS
    # court's proposal; tennis-tagged ones exist purely as joint-competition
    # evidence and are folded into the caller-visible hypothesis list so the
    # review UI can see the runner-up/overlay explanation, but they can never
    # be selected below (see `_select_hypothesis`).
    hypotheses = list(hypotheses) + real_hypotheses

    verified_hypotheses: list[dict[str, Any]] = []
    for hypothesis in hypotheses:
        line_support = dict(hypothesis.get("line_support") or {})
        is_real = "template" in hypothesis
        if is_real:
            frame_visible_error_px = (
                visible_error_px
                if visible_error_px is not None
                else compute_visible_error_px_against_evidence(frame_bgr, hypothesis.get("projected_keypoints") or {})
            )
            top_net_validation = compute_top_net_validation(net_evidence, hypothesis.get("projected_keypoints") or {})
            tennis_overlay_rejection = compute_tennis_overlay_rejection(hypothesis)
        else:
            frame_visible_error_px = visible_error_px
            top_net_validation = {"passed": bool(net_evidence.get("top_tape_line"))}
            tennis_overlay_rejection = {"passed": False}
        verification = verify_court_hypothesis(
            hypothesis=hypothesis,
            visible_error_px=frame_visible_error_px,
            line_support=line_support,
            temporal_stability_px=temporal_stability_px or {"median": 0.0},
            top_net_validation=top_net_validation,
            tennis_overlay_rejection=tennis_overlay_rejection,
        )
        promotion_allowed = bool(verification["promotion_allowed"]) and (not is_real or bool(hypothesis.get("promotable_as_pickleball", False)))
        item = {**hypothesis, "verification": verification, "promotion_allowed": promotion_allowed}
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

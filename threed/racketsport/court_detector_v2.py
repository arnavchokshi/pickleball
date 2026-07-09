"""Court detector v2 orchestration."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import logging
from pathlib import Path
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
_LOGGER = logging.getLogger(__name__)


def detect_court_v2_from_frame(
    frame_bgr: Any,
    *,
    clip_id: str = "",
    source_frame: str | None = None,
    visible_error_px: dict[str, Any] | None = None,
    temporal_stability_px: dict[str, Any] | None = None,
    neural_checkpoint_path: str | Path | None = None,
    neural_infer: Callable[[Any], Mapping[str, Any]] | None = None,
    neural_device: str = "cpu",
) -> dict[str, Any]:
    if frame_bgr is None or not hasattr(frame_bgr, "shape") or len(frame_bgr.shape) < 2:
        raise ValueError("frame_bgr must be an image array")
    height, width = int(frame_bgr.shape[0]), int(frame_bgr.shape[1])
    if width <= 0 or height <= 0:
        raise ValueError("frame_bgr must have positive dimensions")

    neural_provider = _make_optional_neural_provider(
        neural_checkpoint_path=neural_checkpoint_path,
        neural_infer=neural_infer,
        neural_device=neural_device,
    )
    net_evidence = detect_court_net_evidence(frame_bgr)
    surface_evidence = build_surface_paint_evidence(frame_bgr, roi=net_evidence.get("roi"))
    hypotheses = generate_court_hypotheses(
        image_size=(width, height),
        net_evidence=net_evidence,
        surface_evidence=surface_evidence,
    )
    neural_inference = _run_optional_neural_inference(frame_bgr, neural_provider=neural_provider)
    try:
        if neural_inference is None:
            real_hypotheses = generate_homography_hypotheses(
                frame_bgr, net_evidence=net_evidence, surface_evidence=surface_evidence, max_hypotheses=8
            )
        else:
            real_hypotheses = generate_homography_hypotheses(
                frame_bgr,
                net_evidence=net_evidence,
                surface_evidence=surface_evidence,
                max_hypotheses=8,
                neural_inference=neural_inference,
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
            hypothesis_keypoints = hypothesis.get("projected_keypoints") or hypothesis.get("keypoints") or {}
            frame_visible_error_px = (
                visible_error_px
                if visible_error_px is not None
                else compute_visible_error_px_against_evidence(frame_bgr, hypothesis_keypoints)
            )
            top_net_validation = compute_top_net_validation(net_evidence, hypothesis_keypoints)
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
    if (
        selected
        and not bool(selected.get("promotion_allowed", False))
        and bool(selected_verification.get("promotion_allowed", False))
        and selected.get("source_tag") == "neural_seeded"
    ):
        selected_verification = {**selected_verification, "promotion_allowed": False}
        selected_verification["blockers"] = ["neural_seed_review_only"]
    blockers = list(selected_verification.get("blockers") or ["no_detector_v2_hypothesis"])
    clip = clip_id or "unknown_clip"

    if selected and bool(selected.get("promotion_allowed", False)):
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


def detect_court_v2_from_frames(
    frames_bgr: Sequence[Any],
    *,
    clip_id: str = "",
    neural_checkpoint_path: str | Path | None = None,
    neural_infer: Callable[[Any], Mapping[str, Any]] | None = None,
    neural_device: str = "cpu",
    geo_r3_enabled: bool = True,
    geo_r3_vote_top_k: int | None = None,
    geo_r3_identity_link_median_px: float | None = None,
    geo_r3_identity_min_shared_keypoints: int | None = None,
) -> dict[str, Any]:
    """Lightweight E4 multi-frame fusion API for the GEO r3 top-3 vote.

    This accepts already-decoded frames so tests and checkpoint smoke runs can
    exercise the fusion seam without editing the existing video proposal CLI.
    """

    from .court_proposals import (
        GEO_R3_IDENTITY_LINK_MEDIAN_PX,
        GEO_R3_IDENTITY_MIN_SHARED_KEYPOINTS,
        GEO_R3_VOTE_TOP_K,
        _geor3_json_safe_vote,
        _geor3_ranked_top_hypotheses,
        _geor3_select_identity_vote,
        _geor3_temporal_trigger_fires,
        _temporal_consensus,
    )

    frames = list(frames_bgr)
    if not frames:
        raise ValueError("frames_bgr must contain at least one frame")
    for frame in frames:
        if frame is None or not hasattr(frame, "shape") or len(frame.shape) < 2:
            raise ValueError("frames_bgr must contain image arrays")
    height, width = int(frames[0].shape[0]), int(frames[0].shape[1])
    if width <= 0 or height <= 0:
        raise ValueError("frames_bgr must contain positive image dimensions")

    top_k = GEO_R3_VOTE_TOP_K if geo_r3_vote_top_k is None else int(geo_r3_vote_top_k)
    link_px = GEO_R3_IDENTITY_LINK_MEDIAN_PX if geo_r3_identity_link_median_px is None else float(geo_r3_identity_link_median_px)
    min_shared = (
        GEO_R3_IDENTITY_MIN_SHARED_KEYPOINTS
        if geo_r3_identity_min_shared_keypoints is None
        else int(geo_r3_identity_min_shared_keypoints)
    )
    neural_provider = _make_optional_neural_provider(
        neural_checkpoint_path=neural_checkpoint_path,
        neural_infer=neural_infer,
        neural_device=neural_device,
    )
    provider_enabled = neural_provider is not None
    per_frame_results: list[dict[str, Any]] = []
    for frame_index, frame in enumerate(frames):
        try:
            net_evidence = detect_court_net_evidence(frame)
        except Exception:
            net_evidence = {}
        try:
            surface_evidence = build_surface_paint_evidence(frame, roi=net_evidence.get("roi"))
        except Exception:
            surface_evidence = {}
        neural_inference = _run_optional_neural_inference(frame, neural_provider=neural_provider)
        if neural_inference is None:
            hypotheses = generate_homography_hypotheses(
                frame,
                net_evidence=net_evidence,
                surface_evidence=surface_evidence,
                max_hypotheses=40,
            )
        else:
            hypotheses = generate_homography_hypotheses(
                frame,
                net_evidence=net_evidence,
                surface_evidence=surface_evidence,
                max_hypotheses=40,
                neural_inference=neural_inference,
            )
        pickleball_hypotheses = [item for item in hypotheses if item.get("template") == "pickleball"]
        best = pickleball_hypotheses[0] if pickleball_hypotheses else None
        top_pickleball_hypotheses = _geor3_ranked_top_hypotheses(
            pickleball_hypotheses,
            best,
            top_k=top_k,
        )
        per_frame_results.append(
            {
                "frame_index": frame_index,
                "best": best,
                "top_pickleball_hypotheses": top_pickleball_hypotheses,
                "hypothesis_count": len(hypotheses),
            }
        )

    frames_with_hypothesis = [item for item in per_frame_results if item["best"] is not None]
    selected_frames = frames_with_hypothesis
    if frames_with_hypothesis:
        _consensus, temporal_stability, _retained_indices = _temporal_consensus(
            [item["best"]["keypoints"] for item in frames_with_hypothesis]
        )
    else:
        temporal_stability = {"median": None, "p95": None, "frame_count": 0}
    geor3_vote: dict[str, Any] = {
        "attempted": False,
        "enabled": bool(geo_r3_enabled),
        "triggered": False,
        "selected": False,
        "config": {
            "top_k": top_k,
            "identity_link_median_px": link_px,
            "identity_min_shared_keypoints": min_shared,
        },
        "trigger": {"r2_temporal_stability_px": temporal_stability},
    }
    if _geor3_temporal_trigger_fires(temporal_stability, enabled=geo_r3_enabled):
        vote = _geor3_select_identity_vote(
            frames_with_hypothesis,
            top_k=top_k,
            identity_link_median_px=link_px,
            min_shared_keypoints=min_shared,
        )
        geor3_vote = {
            **_geor3_json_safe_vote(vote),
            "enabled": bool(geo_r3_enabled),
            "triggered": True,
            "config": {
                "top_k": top_k,
                "identity_link_median_px": link_px,
                "identity_min_shared_keypoints": min_shared,
            },
            "trigger": {"r2_temporal_stability_px": temporal_stability},
        }
        if vote.get("selected"):
            selected_frames = list(vote.get("selected_frames") or [])

    selected_hypothesis = None
    if selected_frames:
        selected_frame = min(selected_frames, key=lambda item: float((item.get("best") or {}).get("score") or 0.0))
        selected_hypothesis = _json_safe_hypothesis(selected_frame["best"])

    return {
        "schema_version": 1,
        "artifact_type": "racketsport_court_detector_v2_multiframe_fusion",
        "clip": clip_id or "unknown_clip",
        "image_size": [width, height],
        "provider_enabled": bool(provider_enabled),
        "selected_hypothesis": selected_hypothesis,
        "geor3": geor3_vote,
        "frames": [
            {
                "frame_index": item["frame_index"],
                "hypothesis_count": item["hypothesis_count"],
                "best": None if item["best"] is None else _json_safe_hypothesis(item["best"]),
                "top_pickleball_hypotheses": [_json_safe_hypothesis(hypothesis) for hypothesis in item["top_pickleball_hypotheses"]],
            }
            for item in per_frame_results
        ],
    }


def _select_hypothesis(hypotheses: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not hypotheses:
        return None

    def score(item: dict[str, Any]) -> tuple[bool, float]:
        components = item.get("score_components") or {}
        return (bool(item.get("promotion_allowed")), float(components.get("evidence_score", 0.0)))

    return max(hypotheses, key=score)


def _make_optional_neural_provider(
    *,
    neural_checkpoint_path: str | Path | None,
    neural_infer: Callable[[Any], Mapping[str, Any]] | None,
    neural_device: str,
) -> Callable[[Any], Mapping[str, Any]] | None:
    if neural_checkpoint_path is not None and neural_infer is not None:
        raise ValueError("provide either neural_checkpoint_path or neural_infer, not both")
    if neural_infer is not None:
        return neural_infer
    from .court_model_infer import make_court_model_infer_provider

    try:
        return make_court_model_infer_provider(checkpoint_path=neural_checkpoint_path, device=neural_device)
    except Exception as exc:
        _LOGGER.warning("court_unet_v2 provider unavailable reason=%s; using geometric-only", exc)
        return None


def _run_optional_neural_inference(
    frame_bgr: Any,
    *,
    neural_provider: Callable[[Any], Mapping[str, Any]] | None,
) -> Mapping[str, Any] | None:
    if neural_provider is None:
        return None
    try:
        return neural_provider(frame_bgr)
    except Exception as exc:
        _LOGGER.warning("court_unet_v2 inference failed reason=%s; using geometric-only for frame", exc)
        return None


def _json_safe_hypothesis(hypothesis: Mapping[str, Any]) -> dict[str, Any]:
    def _safe(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {str(key): _safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [_safe(item) for item in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    source_tag = hypothesis.get("source_tag")
    if source_tag is None:
        source_tag = "neural_seeded" if hypothesis.get("model_confidence") is not None else "geometric"
    return {
        "hypothesis_id": str(hypothesis.get("hypothesis_id") or ""),
        "source": str(hypothesis.get("source") or ""),
        "source_tag": str(source_tag),
        "model_confidence": _safe(hypothesis.get("model_confidence")),
        "score": _safe(hypothesis.get("score")),
        "evidence_score": _safe((hypothesis.get("score_components") or {}).get("evidence_score", hypothesis.get("evidence_score"))),
        "keypoints": _safe(hypothesis.get("keypoints") or hypothesis.get("projected_keypoints") or {}),
        "score_components": _safe(hypothesis.get("score_components") or {}),
    }

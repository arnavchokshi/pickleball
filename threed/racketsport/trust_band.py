"""Trust-band confidence provenance for `virtual_world.json` entities.

Per NORTH_STAR_ROADMAP.md's "3D world / scrubber surfaces" section, the scrubber must
carry a mandatory per-entity `{stage, gate_id, gate_status, badge, reason,
evidence_path}` field so it can grey out (never charge for, never render as
final) any mesh/track/ball/paddle/court segment whose upstream stage has not
passed its gate.

This module classifies already-computed upstream gate state into a display
badge. It does not compute or run any gate itself, and it never imports a
gate module: it only reads already-produced gate-report JSON (or explicit
facts the caller supplies) and derives an honest
`verified | preview | low_confidence` badge plus a human-readable reason.
"""

from __future__ import annotations

from typing import Any, Mapping

TRUST_BADGES = ("verified", "preview", "low_confidence")

# Per the completed calibration diagnostic (runs/cal_body_projection_bias_20260702T014121Z/):
# the originally-suspected "~14px vertical overlay offset" turned out to be a metric-space
# artifact, not an overlay-rendering bug -- pixel-space overlays are sound (feet pin to the
# bbox bottom within ~0.1px). The real, universal defect is in world-frame *meters*: guessed
# camera intrinsics + a 4-corner PnP solve (not the full 15-keypoint metric calibration) drive
# every court_Z0 world position on every clip. Skeletal motion/shape is fine to show; it is the
# absolute world-frame scale/position (on-court meters, coverage distances, world paths) that is
# provisional until the 15-keypoint metric calibration lands.
_WORLD_SCALE_CALIBRATION_NOTE = (
    "World-scale preview -- calibration upgrade pending: world-frame meter positions "
    "(joints_world/vertices_world, on-court coordinates, coverage distances) inherit a "
    "universal calibration defect from guessed intrinsics + a 4-corner PnP solve on every clip, "
    "not the full 15-keypoint metric calibration; treat absolute world-scale/position as preview, "
    "not verified, until that upgrade lands. Skeletal motion/shape itself is fine to show -- this "
    "caveat is about on-court meters, not pose quality. (Pixel-space overlays are sound: feet pin "
    "to the bbox bottom within ~0.1px -- the earlier-suspected ~14px vertical offset was a "
    "metric-space artifact, not an overlay bug.)"
)


def build_trust_band(
    *,
    stage: str,
    gate_id: str,
    gate_status: str,
    badge: str,
    reason: str,
    evidence_path: str | None = None,
) -> dict[str, Any]:
    """Build one schema-valid trust-band payload.

    Raises ``ValueError`` for an unknown badge or missing required text
    fields so a caller cannot silently attach an empty/placeholder trust
    band to a scrubber entity.
    """

    if badge not in TRUST_BADGES:
        raise ValueError(f"badge must be one of {TRUST_BADGES}, got {badge!r}")
    for field_name, value in (("stage", stage), ("gate_id", gate_id), ("gate_status", gate_status), ("reason", reason)):
        if not value or not str(value).strip():
            raise ValueError(f"{field_name} is required")
    return {
        "stage": stage,
        "gate_id": gate_id,
        "gate_status": gate_status,
        "badge": badge,
        "reason": reason,
        "evidence_path": evidence_path,
    }


def derive_body_trust_band(
    body_gate_report_clip: Mapping[str, Any],
    *,
    evidence_path: str,
    calibration_offset_note: str = _WORLD_SCALE_CALIBRATION_NOTE,
) -> dict[str, Any]:
    """Classify a BODY `body_gate_report.json` clip entry into a trust band.

    Structural pass + human-ratified overlay review is enough to start the
    scrubber surface (W3-SCRUBBER-V0's stated dependency on W2-BODY is
    "structural pass suffices to start; accuracy gate improves badge trust
    over time"), but BODY entities must never claim `verified` until the
    world-MPJPE accuracy gate also clears.
    """

    full_clip_gate = body_gate_report_clip.get("full_clip_body_gate") or {}
    grounding_quality = body_gate_report_clip.get("body_grounding_quality") or {}
    overlay_alignment = body_gate_report_clip.get("body_review_overlay_alignment") or {}
    world_mpjpe = body_gate_report_clip.get("world_mpjpe") or {}

    structural_pass = (
        bool(full_clip_gate.get("passed"))
        and grounding_quality.get("status") == "pass"
        and overlay_alignment.get("status") == "pass"
        and int(overlay_alignment.get("unresolved_warning_sample_count") or 0) == 0
    )
    accuracy_blockers = [str(blocker) for blocker in (world_mpjpe.get("blockers") or [])]
    accuracy_verified = structural_pass and not accuracy_blockers

    if accuracy_verified:
        return build_trust_band(
            stage="BODY",
            gate_id="body_world_mpjpe_gate",
            gate_status="pass",
            badge="verified",
            reason="Structural BODY gates and the world-MPJPE accuracy gate both pass.",
            evidence_path=evidence_path,
        )
    if structural_pass:
        rendered = overlay_alignment.get("rendered_count", 0)
        sample_count = overlay_alignment.get("sample_count", 0)
        resolved = overlay_alignment.get("resolved_warning_sample_count", 0)
        return build_trust_band(
            stage="BODY",
            gate_id="body_full_clip_gate+body_review_overlay_alignment",
            gate_status="structural_pass_accuracy_unmeasured",
            badge="preview",
            reason=(
                "Structural BODY gates pass (coverage, foot-slide, 0 unresolved overlay warnings) and "
                f"overlay review is human-ratified ({rendered}/{sample_count} samples rendered, "
                f"{resolved} warnings reviewed and resolved), but the world-MPJPE accuracy gate has not "
                f"passed ({', '.join(accuracy_blockers) or 'no finalized labels yet'}). " + calibration_offset_note
            ),
            evidence_path=evidence_path,
        )
    return build_trust_band(
        stage="BODY",
        gate_id="body_full_clip_gate",
        gate_status=str(body_gate_report_clip.get("status") or "unknown"),
        badge="low_confidence",
        reason="BODY structural gates have not passed for this clip; joint/mesh positions are not reliable.",
        evidence_path=evidence_path,
    )


def derive_court_trust_band(
    court_calibration: Mapping[str, Any],
    *,
    evidence_path: str,
    world_scale_note: str = _WORLD_SCALE_CALIBRATION_NOTE,
) -> dict[str, Any]:
    """Classify a `court_calibration.json` payload into a trust band.

    CAL has 0 held-out gate passes today (held-out PCK@5px>=0.95 per
    viewpoint, NORTH_STAR_ROADMAP.md). A human-reviewed manual corner sidecar is
    more trustworthy than an unreviewed automated guess, but it is still not
    a verified calibration -- and per the completed calibration diagnostic,
    this calibration's guessed intrinsics + 4-corner PnP solve is the actual
    source of the universal world-frame-meter imprecision every court_Z0
    position inherits (see `_WORLD_SCALE_CALIBRATION_NOTE`).
    """

    capture_quality = court_calibration.get("capture_quality") or {}
    reasons = [str(reason) for reason in (capture_quality.get("reasons") or [])]
    grade = capture_quality.get("grade")
    intrinsics_source = str((court_calibration.get("intrinsics") or {}).get("source") or "")
    calibration_source = str(court_calibration.get("source") or "")
    metric_confidence = court_calibration.get("metric_confidence")
    is_metric15 = (
        calibration_source == "metric_15pt_reviewed"
        or intrinsics_source == "metric_15pt_reviewed"
        or "reviewed_15pt_correspondences" in reasons
    )
    is_manual_review = (
        any("human_review" in reason or reason == "manual" for reason in reasons)
        or intrinsics_source == "manual"
    )

    if is_metric15:
        return build_trust_band(
            stage="CAL",
            gate_id="court_calibration_pck5px_gate",
            gate_status="metric15_unverified",
            badge="preview",
            reason=(
                "Placement uses the metric-15pt reviewed calibration "
                f"(source={calibration_source or 'unknown'}, intrinsics.source={intrinsics_source or 'unknown'}, "
                f"grade={grade or 'unknown'}, metric_confidence={metric_confidence or 'unknown'}, "
                f"reasons={', '.join(reasons) or 'none'}). The held-out PCK@5px>=0.95 CAL gate has not "
                "passed, and the single-view planar calibration still carries the documented metric-scale "
                "identifiability caveats; treat court_Z0 meter placement as preview, not verified."
            ),
            evidence_path=evidence_path,
        )

    if grade == "good" and not reasons:
        return build_trust_band(
            stage="CAL",
            gate_id="court_calibration_pck5px_gate",
            gate_status="held_out_gate_not_passed",
            badge="preview",
            reason=(
                "Capture quality grade is good, but the held-out PCK@5px>=0.95 gate has not passed. "
                + world_scale_note
            ),
            evidence_path=evidence_path,
        )
    if is_manual_review or grade == "warn":
        return build_trust_band(
            stage="CAL",
            gate_id="court_calibration_pck5px_gate",
            gate_status="manual_sidecar_unverified",
            badge="preview",
            reason=(
                "Calibration comes from a human-reviewed manual corner sidecar "
                f"({', '.join(reasons) or 'manual corners'}) with guessed/estimated intrinsics "
                f"(source={intrinsics_source or 'unknown'}) and a 4-corner PnP solve, not a passing "
                "automated detector or the full 15-keypoint metric calibration; the held-out "
                "PCK@5px>=0.95 gate has not passed. " + world_scale_note
            ),
            evidence_path=evidence_path,
        )
    return build_trust_band(
        stage="CAL",
        gate_id="court_calibration_pck5px_gate",
        gate_status="low_quality",
        badge="low_confidence",
        reason=f"Capture quality grade is {grade or 'unknown'}: {', '.join(reasons) or 'no supporting evidence'}.",
        evidence_path=evidence_path,
    )


def derive_ball_trust_band(
    *,
    source: str | None,
    evidence_path: str,
    gate_status: str = "0/8 milestones pass",
) -> dict[str, Any]:
    """BALL has no passing milestone gate today; every ball track is low_confidence."""

    return build_trust_band(
        stage="BALL",
        gate_id="ball_m1_f1_at_20_gate",
        gate_status=gate_status,
        badge="low_confidence",
        reason=(
            f"BALL track source is {source or 'unknown'}; the M1 gate "
            "(F1@20>=0.90, recall@20>=0.75, hidden-FP<=0.05) has not passed for any milestone "
            "(0/8 per NORTH_STAR_ROADMAP.md); treat ball position as a rough cue, not a verified track."
        ),
        evidence_path=evidence_path,
    )


def derive_track_trust_band(*, idf1: float | None, evidence_path: str) -> dict[str, Any]:
    """TRK (person tracking) trust band for players without BODY output."""

    idf1_text = f"{idf1:.4f}" if idf1 is not None else "unknown"
    return build_trust_band(
        stage="TRK",
        gate_id="trk_idf1_gate",
        gate_status="do_not_promote",
        badge="low_confidence",
        reason=(
            f"Person-track IDF1={idf1_text} has not cleared the per-clip gate "
            "(IDF1>=0.85, 0 ID switches, 0 spectator/off-court FP, cov4>=0.95); do_not_promote."
        ),
        evidence_path=evidence_path,
    )


def derive_paddle_trust_band(*, evidence_path: str) -> dict[str, Any]:
    """RKT (paddle 6DoF) trust band; box-derived-only, 0 true-corner labels exist."""

    return build_trust_band(
        stage="RKT",
        gate_id="racket_face_angle_p90_gate",
        gate_status="unscoreable_no_gt",
        badge="low_confidence",
        reason=(
            "0 true paddle-face corner labels exist anywhere in the project; paddle pose is "
            "box-derived preview only (review_only_no_rkt_promotion)."
        ),
        evidence_path=evidence_path,
    )


__all__ = [
    "TRUST_BADGES",
    "build_trust_band",
    "derive_body_trust_band",
    "derive_court_trust_band",
    "derive_ball_trust_band",
    "derive_track_trust_band",
    "derive_paddle_trust_band",
]

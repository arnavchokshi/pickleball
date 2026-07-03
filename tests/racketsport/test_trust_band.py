from __future__ import annotations

import pytest

from threed.racketsport.schemas import TrustBand
from threed.racketsport.trust_band import (
    build_trust_band,
    derive_ball_trust_band,
    derive_body_trust_band,
    derive_court_trust_band,
    derive_paddle_trust_band,
    derive_track_trust_band,
)


def _overlay_alignment(*, status: str = "pass", unresolved: int = 0) -> dict:
    return {
        "status": status,
        "rendered_count": 20,
        "sample_count": 20,
        "resolved_warning_sample_count": 3,
        "unresolved_warning_sample_count": unresolved,
    }


def _body_gate_report_clip(
    *,
    full_clip_passed: bool = True,
    grounding_status: str = "pass",
    overlay_status: str = "pass",
    unresolved_overlay: int = 0,
    world_mpjpe_blockers: list[str] | None = None,
    status: str = "blocked",
) -> dict:
    return {
        "status": status,
        "full_clip_body_gate": {"passed": full_clip_passed},
        "body_grounding_quality": {"status": grounding_status},
        "body_review_overlay_alignment": _overlay_alignment(status=overlay_status, unresolved=unresolved_overlay),
        "world_mpjpe": {"blockers": world_mpjpe_blockers if world_mpjpe_blockers is not None else ["missing_world_mpjpe_gate"]},
    }


def test_build_trust_band_round_trips_through_schema() -> None:
    payload = build_trust_band(
        stage="BODY",
        gate_id="body_world_mpjpe_gate",
        gate_status="pass",
        badge="verified",
        reason="ok",
        evidence_path="runs/x/body_gate_report.json",
    )
    band = TrustBand.model_validate(payload)
    assert band.badge == "verified"
    assert band.evidence_path == "runs/x/body_gate_report.json"


def test_build_trust_band_rejects_unknown_badge() -> None:
    with pytest.raises(ValueError, match="badge must be one of"):
        build_trust_band(stage="BODY", gate_id="g", gate_status="s", badge="trusted", reason="r")


@pytest.mark.parametrize("field", ["stage", "gate_id", "gate_status", "reason"])
def test_build_trust_band_rejects_missing_required_text(field: str) -> None:
    kwargs = {"stage": "BODY", "gate_id": "g", "gate_status": "s", "badge": "preview", "reason": "r"}
    kwargs[field] = ""
    with pytest.raises(ValueError, match=f"{field} is required"):
        build_trust_band(**kwargs)


def test_derive_body_trust_band_is_preview_for_structural_pass_with_accuracy_gate_missing() -> None:
    band = derive_body_trust_band(_body_gate_report_clip(), evidence_path="runs/body_gate_report.json")
    assert band["badge"] == "preview"
    assert band["stage"] == "BODY"
    assert "World-scale preview" in band["reason"]
    assert "calibration upgrade pending" in band["reason"]
    assert "4-corner PnP" in band["reason"]
    assert "missing_world_mpjpe_gate" in band["reason"]
    TrustBand.model_validate(band)


def test_derive_body_trust_band_is_verified_when_structural_and_accuracy_gates_both_pass() -> None:
    clip = _body_gate_report_clip(world_mpjpe_blockers=[])
    band = derive_body_trust_band(clip, evidence_path="runs/body_gate_report.json")
    assert band["badge"] == "verified"


def test_derive_body_trust_band_is_low_confidence_when_structural_gate_fails() -> None:
    clip = _body_gate_report_clip(full_clip_passed=False)
    band = derive_body_trust_band(clip, evidence_path="runs/body_gate_report.json")
    assert band["badge"] == "low_confidence"


def test_derive_body_trust_band_is_low_confidence_when_overlay_has_unresolved_warnings() -> None:
    clip = _body_gate_report_clip(unresolved_overlay=2)
    band = derive_body_trust_band(clip, evidence_path="runs/body_gate_report.json")
    assert band["badge"] == "low_confidence"


def test_derive_court_trust_band_is_preview_for_manual_reviewed_corners() -> None:
    court_calibration = {
        "capture_quality": {
            "grade": "warn",
            "reasons": ["prototype_human_review_corners", "estimated_intrinsics", "corrected_unverified"],
        },
        "intrinsics": {"source": "estimated_from_review_frame"},
    }
    band = derive_court_trust_band(court_calibration, evidence_path="runs/court_calibration.json")
    assert band["badge"] == "preview"
    assert band["stage"] == "CAL"
    assert "World-scale preview" in band["reason"]
    assert "calibration upgrade pending" in band["reason"]
    TrustBand.model_validate(band)


def test_derive_court_trust_band_surfaces_metric15_grade_without_manual_sidecar_claim() -> None:
    court_calibration = {
        "source": "metric_15pt_reviewed",
        "metric_confidence": "low",
        "capture_quality": {
            "grade": "warn",
            "reasons": [
                "reprojection_high",
                "single_view_planar_full_calibration",
                "reviewed_15pt_correspondences",
            ],
        },
        "intrinsics": {"source": "metric_15pt_reviewed"},
    }
    band = derive_court_trust_band(court_calibration, evidence_path="runs/court_calibration_metric15pt.json")
    assert band["badge"] == "preview"
    assert band["gate_status"] == "metric15_unverified"
    assert "metric-15pt reviewed calibration" in band["reason"]
    assert "grade=warn" in band["reason"]
    assert "metric_confidence=low" in band["reason"]
    assert "manual corner sidecar" not in band["reason"]
    assert "4-corner PnP" not in band["reason"]
    TrustBand.model_validate(band)


def test_derive_court_trust_band_is_low_confidence_for_poor_unreviewed_grade() -> None:
    court_calibration = {"capture_quality": {"grade": "poor", "reasons": ["low_reprojection_confidence"]}}
    band = derive_court_trust_band(court_calibration, evidence_path="runs/court_calibration.json")
    assert band["badge"] == "low_confidence"


def test_derive_ball_trust_band_is_always_low_confidence() -> None:
    band = derive_ball_trust_band(source="tracknet", evidence_path="runs/ball_track.json")
    assert band["badge"] == "low_confidence"
    assert band["stage"] == "BALL"
    assert "tracknet" in band["reason"]


def test_derive_track_trust_band_reports_idf1_in_reason() -> None:
    band = derive_track_trust_band(idf1=0.8904, evidence_path="runs/person_track_gt_score.json")
    assert band["badge"] == "low_confidence"
    assert "0.8904" in band["reason"]


def test_derive_paddle_trust_band_is_low_confidence() -> None:
    band = derive_paddle_trust_band(evidence_path="runs/racket_promotion_audit.json")
    assert band["badge"] == "low_confidence"
    assert band["stage"] == "RKT"

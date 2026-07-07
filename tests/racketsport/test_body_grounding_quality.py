from __future__ import annotations

import pytest

from threed.racketsport.body_grounding_quality import build_body_grounding_quality


def _gate_stream(*, slide_m: float, rejection_reason: str | None = None, lock_metric_included: bool = True) -> dict:
    phase_row = {
        "clip": "clip_001",
        "player_id": "p1",
        "foot": "left",
        "phase_id": "p1:left:0",
        "start_frame_index": 0,
        "end_frame_index": 2,
        "slide_m": slide_m,
        "max_contributing_frame_index": 2,
        "anchor_position_xyz": [0.0, 0.0, 0.0],
        "contact_source": "body_foot_contact_detector",
        "foot_assignment": "per_foot_body_contact",
        "weak": bool(rejection_reason),
        "demoted": bool(rejection_reason),
        "split": False,
        "rejection_reason": rejection_reason,
        "lock_metric_included": lock_metric_included,
    }
    return {
        "schema_version": 1,
        "artifact_type": "foot_lock_gate_stream",
        "clip": "clip_001",
        "phase_rows": [phase_row],
        "frame_rows": [
            {
                "clip": "clip_001",
                "player_id": "p1",
                "foot": "left",
                "phase_id": "p1:left:0",
                "frame_idx": 2,
                "contact_state": True,
                "selected_foot": "left",
                "lock_anchor_xyz": [0.0, 0.0, 0.0],
                "raw_xy": [slide_m, 0.0],
                "fused_xy": [slide_m, 0.0],
                "smoothed_xy": [slide_m, 0.0],
                "original_xy": [slide_m, 0.0],
                "body_root_world": [0.0, 0.0, 0.0],
                "output_source": "body",
                "divergence_flag": False,
                "speed_cap_flag": False,
                "residuals": {},
                "bbox_margin_px": None,
                "source_counts": {},
                "foot_pin_correction_m": 0.0,
            }
        ],
        "summary": {
            "top_20_phases_by_slide_m": [phase_row],
            "phases_over_threshold": [phase_row] if slide_m > 0.03 else [],
            "weak_rejection_reasons": {rejection_reason: 1} if rejection_reason else {},
            "candidate_phase_rejection_reason_counts": {rejection_reason: 1} if rejection_reason else {},
            "max_candidate_phase_slide_m": slide_m,
            "frame_row_stride": 1,
            "frame_rows_unstrided_count": 1,
        },
        "artifact_size_policy": {"max_bytes": 20_000_000, "action": "stride_frame_rows_when_needed"},
    }


def test_body_grounding_quality_passes_foot_slide_at_objective_threshold() -> None:
    payload = build_body_grounding_quality(
        clip="clip_001",
        grounding_metrics={"max_foot_lock_slide_m": 0.03, "foot_lock_contact_samples": 12},
    )

    assert payload["artifact_type"] == "racketsport_body_grounding_quality"
    assert payload["status"] == "pass"
    assert payload["foot_slide_gate"] == {
        "name": "foot_slide_max_m",
        "threshold_m": 0.03,
        "value_m": 0.03,
        "passed": True,
    }
    assert payload["blockers"] == []


def test_body_grounding_quality_fails_foot_slide_above_objective_threshold() -> None:
    payload = build_body_grounding_quality(
        clip="clip_001",
        grounding_metrics={"max_foot_lock_slide_m": 0.031, "foot_lock_contact_samples": 12},
    )

    assert payload["status"] == "fail"
    assert payload["foot_slide_gate"]["threshold_m"] == pytest.approx(0.03)
    assert payload["foot_slide_gate"]["value_m"] == pytest.approx(0.031)
    assert payload["foot_slide_gate"]["passed"] is False
    assert payload["blockers"] == ["foot_slide_gate_failed"]


def test_body_grounding_quality_fails_closed_on_overthreshold_gate_stream_phase() -> None:
    payload = build_body_grounding_quality(
        clip="clip_001",
        grounding_metrics={
            "max_foot_lock_slide_m": 0.0,
            "foot_lock_gate_stream": _gate_stream(slide_m=0.04),
        },
    )

    assert payload["status"] == "fail"
    assert payload["foot_slide_gate"]["passed"] is False
    assert "foot_lock_gate_stream_over_threshold_phase" in payload["blockers"]


def test_body_grounding_quality_treats_independent_rejected_candidate_slide_as_companion_only() -> None:
    payload = build_body_grounding_quality(
        clip="clip_001",
        grounding_metrics={
            "max_foot_lock_slide_m": 0.0,
            "max_candidate_phase_slide_m": 0.04,
            "foot_lock_gate_stream": _gate_stream(
                slide_m=0.04,
                rejection_reason="phase_penetrates_ground",
                lock_metric_included=False,
            ),
        },
    )

    assert payload["status"] == "pass"
    assert payload["foot_slide_gate"]["passed"] is True
    assert payload["blockers"] == []


def test_body_grounding_quality_blocks_missing_foot_slide_metric() -> None:
    payload = build_body_grounding_quality(clip="clip_001", grounding_metrics={})

    assert payload["status"] == "blocked"
    assert payload["foot_slide_gate"]["value_m"] is None
    assert payload["foot_slide_gate"]["passed"] is False
    assert payload["blockers"] == ["missing_foot_slide_metric"]


def test_body_grounding_quality_embeds_schema_validated_gate_stream_without_changing_threshold() -> None:
    gate_stream = {
        "schema_version": 1,
        "artifact_type": "foot_lock_gate_stream",
        "clip": "clip_001",
        "phase_rows": [
            {
                "clip": "clip_001",
                "player_id": "p1",
                "foot": "left",
                "phase_id": "p1:left:0",
                "start_frame_index": 0,
                "end_frame_index": 2,
                "slide_m": 0.012,
                "max_contributing_frame_index": 2,
                "anchor_position_xyz": [0.0, 0.0, 0.0],
                "contact_source": "body_foot_contact_detector",
                "foot_assignment": "per_foot_body_contact",
                "weak": False,
                "demoted": False,
                "split": False,
            }
        ],
        "frame_rows": [
            {
                "clip": "clip_001",
                "player_id": "p1",
                "foot": "left",
                "phase_id": "p1:left:0",
                "frame_idx": 2,
                "contact_state": True,
                "selected_foot": "left",
                "lock_anchor_xyz": [0.0, 0.0, 0.0],
                "raw_xy": [0.01, 0.0],
                "fused_xy": [0.01, 0.0],
                "smoothed_xy": [0.01, 0.0],
                "original_xy": [0.01, 0.0],
                "body_root_world": [0.0, 0.0, 0.0],
                "output_source": "body",
                "divergence_flag": False,
                "speed_cap_flag": False,
                "residuals": {},
                "bbox_margin_px": None,
                "source_counts": {},
                "foot_pin_correction_m": 0.0,
            }
        ],
        "summary": {
            "top_20_phases_by_slide_m": [],
            "phases_over_threshold": [],
            "weak_rejection_reasons": {},
            "frame_row_stride": 1,
            "frame_rows_unstrided_count": 1,
        },
        "artifact_size_policy": {"max_bytes": 20_000_000, "action": "stride_frame_rows_when_needed"},
    }

    payload = build_body_grounding_quality(
        clip="clip_001",
        grounding_metrics={
            "max_foot_lock_slide_m": 0.012,
            "foot_lock_contact_samples": 3,
            "foot_lock_gate_stream": gate_stream,
        },
    )

    assert payload["foot_slide_gate"]["threshold_m"] == pytest.approx(0.03)
    assert payload["foot_lock_gate_stream"]["artifact_type"] == "foot_lock_gate_stream"
    assert payload["foot_lock_gate_stream"]["phase_rows"][0]["slide_m"] == pytest.approx(0.012)

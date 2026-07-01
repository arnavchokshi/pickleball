from __future__ import annotations

import pytest

from threed.racketsport.body_grounding_quality import build_body_grounding_quality


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


def test_body_grounding_quality_blocks_missing_foot_slide_metric() -> None:
    payload = build_body_grounding_quality(clip="clip_001", grounding_metrics={})

    assert payload["status"] == "blocked"
    assert payload["foot_slide_gate"]["value_m"] is None
    assert payload["foot_slide_gate"]["passed"] is False
    assert payload["blockers"] == ["missing_foot_slide_metric"]

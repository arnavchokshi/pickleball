from __future__ import annotations

import pytest

from threed.racketsport import confidence, insight_rules


def test_confidence_grade_combines_metric_confidence_capture_quality_and_missing_upstream() -> None:
    confidence_grade = getattr(confidence, "confidence_grade", None)
    assert callable(confidence_grade)

    assert confidence_grade(0.91, {"grade": "good", "reasons": []}) == {
        "grade": "high",
        "score": pytest.approx(0.91),
        "reasons": [],
    }
    assert confidence_grade(0.78, {"grade": "warn", "reasons": ["motion_blur"]}) == {
        "grade": "medium",
        "score": pytest.approx(0.585),
        "reasons": ["capture_quality:warn", "capture_reason:motion_blur"],
    }
    assert confidence_grade(
        0.94,
        {"grade": "good", "reasons": []},
        missing_upstream_reasons=["no_ball_track"],
    ) == {
        "grade": "omit",
        "score": 0.0,
        "reasons": ["missing_upstream:no_ball_track"],
    }


def test_gate_metric_claim_returns_allow_relative_or_omit() -> None:
    gate_metric_claim = getattr(confidence, "gate_metric_claim", None)
    assert callable(gate_metric_claim)

    assert gate_metric_claim(-0.4, 0.9, threshold=0.8, claim_name="nvz_margin") == {
        "claim_name": "nvz_margin",
        "decision": "allow",
        "value": -0.4,
        "metric_confidence": 0.9,
        "confidence": pytest.approx(0.9),
        "threshold": 0.8,
        "grade": "high",
        "reasons": [],
    }
    assert gate_metric_claim(
        -0.4,
        0.72,
        threshold=0.8,
        claim_name="nvz_margin",
        capture_quality={"grade": "warn", "reasons": ["low_light"]},
    )["decision"] == "relative"
    assert gate_metric_claim(
        -0.4,
        0.9,
        threshold=0.8,
        claim_name="nvz_margin",
        missing_upstream_reasons=["no_court_calibration"],
    )["decision"] == "omit"
    assert gate_metric_claim(None, 0.95, threshold=0.8, claim_name="nvz_margin")["decision"] == "omit"


def test_select_insight_rules_emits_fact_only_habit_like_dicts_from_metric_dicts() -> None:
    select_insight_rules = getattr(insight_rules, "select_insight_rules", None)
    assert callable(select_insight_rules)

    metrics_payload = {
        "schema_version": 1,
        "capture_quality": {"grade": "good", "reasons": []},
        "players": [
            {
                "id": 7,
                "shots": [
                    {
                        "t": 1.25,
                        "type": "dink",
                        "type_conf": 0.93,
                        "metrics": {
                            "nvz_margin_ft": {"value": -0.6, "conf": 0.86, "gated": False, "frames": 8},
                            "balance_score": {"value": 0.92, "conf": 0.88, "gated": False},
                            "paddle_face_deg": {"value": 18.0, "conf": 0.66, "gated": False},
                            "unknown_metric": {"value": 123, "conf": 0.99, "gated": False},
                        },
                    }
                ],
            }
        ],
    }

    habits = select_insight_rules(metrics_payload, threshold=0.8)

    assert habits == [
        {
            "id": "p7_s000_nvz_margin_ft",
            "rule_id": "nvz_margin_below_zero",
            "title": "P7 dink nvz_margin_ft",
            "summary": "Measured nvz_margin_ft=-0.6 at t=1.250s.",
            "confidence": pytest.approx(0.86),
            "clip_ref": {"t0_sec": 1.0, "t1_sec": 2.25},
            "cue": "Fact-only metric review: nvz_margin_ft.",
            "drill": {"name": "Metric review: nvz_margin_ft", "duration_min": 5.0},
            "source": {
                "player_id": 7,
                "shot_index": 0,
                "shot_type": "dink",
                "shot_time_s": 1.25,
                "metric": "nvz_margin_ft",
                "value": -0.6,
                "frames": 8,
            },
            "claim_gate": {
                "claim_name": "nvz_margin_ft",
                "decision": "allow",
                "value": -0.6,
                "metric_confidence": 0.86,
                "confidence": pytest.approx(0.86),
                "threshold": 0.8,
                "grade": "high",
                "reasons": [],
            },
            "llm_used": False,
        },
        {
            "id": "p7_s000_paddle_face_deg",
            "rule_id": "paddle_face_large_abs",
            "title": "P7 dink paddle_face_deg",
            "summary": "Relative signal: measured paddle_face_deg=18.0 at t=1.250s.",
            "confidence": pytest.approx(0.66),
            "clip_ref": {"t0_sec": 1.0, "t1_sec": 2.25},
            "cue": "Fact-only metric review: paddle_face_deg.",
            "drill": {"name": "Metric review: paddle_face_deg", "duration_min": 5.0},
            "source": {
                "player_id": 7,
                "shot_index": 0,
                "shot_type": "dink",
                "shot_time_s": 1.25,
                "metric": "paddle_face_deg",
                "value": 18.0,
            },
            "claim_gate": {
                "claim_name": "paddle_face_deg",
                "decision": "relative",
                "value": 18.0,
                "metric_confidence": 0.66,
                "confidence": pytest.approx(0.66),
                "threshold": 0.8,
                "grade": "medium",
                "reasons": [],
            },
            "llm_used": False,
        },
    ]


def test_select_insight_rules_skips_gated_and_missing_upstream_metrics() -> None:
    select_insight_rules = getattr(insight_rules, "select_insight_rules", None)
    assert callable(select_insight_rules)

    metrics_payload = {
        "schema_version": 1,
        "capture_quality": {"grade": "good", "reasons": []},
        "players": [
            {
                "id": 3,
                "shots": [
                    {
                        "t": 2.0,
                        "type": "drive",
                        "type_conf": 0.91,
                        "metrics": {
                            "nvz_margin_ft": {"value": -1.0, "conf": 0.97, "gated": True},
                            "balance_score": {
                                "value": 0.42,
                                "conf": 0.93,
                                "gated": False,
                                "missing_upstream_reasons": ["no_skeleton3d"],
                            },
                        },
                    }
                ],
            }
        ],
    }

    assert select_insight_rules(metrics_payload, threshold=0.8) == []

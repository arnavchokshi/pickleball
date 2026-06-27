from __future__ import annotations

from threed.racketsport import llm_copy


def _habit_report_payload(*, priority_habit_id: str = "p7_s000_nvz_margin_ft") -> dict:
    return {
        "schema_version": 1,
        "sport": "pickleball",
        "coverage": {"overall": 1.0, "skipped_reason_counts": {}},
        "priority_habit_id": priority_habit_id,
        "replay_ref": None,
        "habits": [
            {
                "id": "p7_s000_nvz_margin_ft",
                "title": "P7 dink nvz_margin_ft",
                "summary": "Measured nvz_margin_ft=-0.6 at t=1.250s.",
                "confidence": 0.86,
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
                },
            }
        ],
    }


def test_copy_faithfulness_passes_for_known_fact_keys_values_and_priority() -> None:
    summarize_copy_faithfulness = getattr(llm_copy, "summarize_copy_faithfulness", None)
    assert callable(summarize_copy_faithfulness)

    summary = summarize_copy_faithfulness(
        facts={"shot_type": "dink", "nvz_margin_ft": -0.6, "shot_time_s": 1.25},
        habit_report=_habit_report_payload(),
        generated_copy=(
            "Priority p7_s000_nvz_margin_ft: your dink shows "
            "nvz_margin_ft=-0.6 at shot_time_s=1.25, so keep the cue tied to that metric."
        ),
        generated_priority_habit_id="p7_s000_nvz_margin_ft",
    )

    assert summary.passed is True
    assert summary.llm_used is False
    assert summary.priority_habit_preserved is True
    assert summary.unknown_fact_keys == []
    assert summary.unknown_fact_values == []
    assert summary.reason_counts == {}


def test_copy_faithfulness_fails_for_unknown_metric_key_and_invented_number() -> None:
    summary = llm_copy.summarize_copy_faithfulness(
        facts={"shot_type": "dink", "nvz_margin_ft": -0.6, "shot_time_s": 1.25},
        habit_report=_habit_report_payload(),
        generated_copy=(
            "Priority p7_s000_nvz_margin_ft: your dink shows hip_rotation_deg=42 "
            "and nvz_margin_ft=-0.6."
        ),
        generated_priority_habit_id="p7_s000_nvz_margin_ft",
    )

    assert summary.passed is False
    assert summary.unknown_fact_keys == ["hip_rotation_deg"]
    assert summary.unknown_fact_values == ["42"]
    assert summary.reason_counts == {"unknown_fact_key": 1, "unknown_fact_value": 1}


def test_copy_faithfulness_fails_when_priority_habit_is_not_preserved() -> None:
    summary = llm_copy.summarize_copy_faithfulness(
        facts={"nvz_margin_ft": -0.6},
        habit_report=_habit_report_payload(priority_habit_id="p7_s000_nvz_margin_ft"),
        generated_copy="Coach a secondary habit with nvz_margin_ft=-0.6.",
        generated_priority_habit_id="p7_s000_balance_score",
    )

    assert summary.passed is False
    assert summary.priority_habit_preserved is False
    assert summary.reason_counts == {
        "missing_priority_habit_reference": 1,
        "priority_habit_mismatch": 1,
    }

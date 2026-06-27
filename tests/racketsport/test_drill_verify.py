from __future__ import annotations

import pytest

from threed.racketsport.drill_verify import (
    DrillEventFlag,
    WristVelocitySample,
    build_drill_report,
    classify_rep_quality,
    detect_reps,
)
from threed.racketsport.schemas import DrillReport


def test_detect_reps_counts_one_contact_per_wrist_swing_state():
    samples = [
        WristVelocitySample(t=0.00, speed_mps=0.20),
        WristVelocitySample(t=0.10, speed_mps=2.40),
        WristVelocitySample(t=0.20, speed_mps=2.80),
        WristVelocitySample(t=0.35, speed_mps=0.30),
        WristVelocitySample(t=0.50, speed_mps=2.50),
        WristVelocitySample(t=0.62, speed_mps=0.40),
    ]

    reps = detect_reps(
        contact_timestamps=[0.05, 0.25, 0.28, 0.56],
        wrist_velocity_samples=samples,
        enter_speed_mps=2.0,
        exit_speed_mps=0.75,
        max_contact_delay_s=0.30,
    )

    assert [rep.t for rep in reps] == pytest.approx([0.25, 0.56])
    assert reps[0].start_t == pytest.approx(0.10)
    assert reps[0].peak_speed_mps == pytest.approx(2.80)
    assert reps[1].start_t == pytest.approx(0.50)
    assert reps[1].peak_speed_mps == pytest.approx(2.50)


def test_classify_rep_quality_marks_supplied_fault_flags_near_contact():
    rep = detect_reps(
        contact_timestamps=[0.25],
        wrist_velocity_samples=[
            WristVelocitySample(t=0.10, speed_mps=2.40),
            WristVelocitySample(t=0.20, speed_mps=2.80),
            WristVelocitySample(t=0.35, speed_mps=0.30),
        ],
    )[0]

    verdict = classify_rep_quality(
        rep,
        event_flags=[
            DrillEventFlag(t=0.23, flag="late_contact"),
            DrillEventFlag(t=0.27, flag="off_balance"),
            DrillEventFlag(t=0.25, flag="coach_note", fault=False),
            DrillEventFlag(t=0.45, flag="outside_window"),
        ],
        tolerance_s=0.04,
    )

    assert verdict.quality == "fault"
    assert verdict.reasons == ["late_contact", "off_balance"]


def test_build_drill_report_returns_drillreport_compatible_dict():
    report = build_drill_report(
        drill="forehand_dink",
        contact_timestamps=[0.25, 0.56],
        wrist_velocity_samples=[
            WristVelocitySample(t=0.10, speed_mps=2.40),
            WristVelocitySample(t=0.20, speed_mps=2.80),
            WristVelocitySample(t=0.35, speed_mps=0.30),
            WristVelocitySample(t=0.50, speed_mps=2.50),
            WristVelocitySample(t=0.62, speed_mps=0.40),
        ],
        event_flags=[
            DrillEventFlag(t=0.56, flag="net_fault"),
        ],
    )

    assert report == {
        "schema_version": 1,
        "drill": "forehand_dink",
        "reps": 2,
        "clean_reps": 1,
        "per_rep": [
            {"t": 0.25, "quality": "clean", "reasons": []},
            {"t": 0.56, "quality": "fault", "reasons": ["net_fault"]},
        ],
    }
    assert DrillReport.model_validate(report).model_dump(mode="json") == report

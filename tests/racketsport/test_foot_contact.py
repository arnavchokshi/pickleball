from __future__ import annotations

import os
import sys

import pytest


def _honor_explicit_mutant_sys_path() -> None:
    for entry in reversed(os.environ.get("PYTHONPATH", "").split(os.pathsep)):
        if not entry or os.path.basename(entry) != "mutant_sys_path":
            continue
        normalized = os.path.abspath(entry)
        if normalized not in sys.path:
            continue
        sys.path.remove(normalized)
        sys.path.insert(0, normalized)


_honor_explicit_mutant_sys_path()

from threed.racketsport.foot_contact import (
    build_body_skeleton_foot_contact_phases,
    build_body_skeleton_foot_contact_phases_from_gate_stream,
    ContactThresholds,
    SkeletonFrame,
    detect_contact_phases,
    measure_contact_metrics,
    resolve_foot_joint_indices,
)


JOINT_NAMES_65 = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_big_toe",
    "left_small_toe",
    "left_heel",
    "right_big_toe",
    "right_small_toe",
    "right_heel",
)


def _frame(frame_index: int, *, left_x: float, left_z: float, right_x: float = 1.0, right_z: float = 0.30) -> SkeletonFrame:
    joints = [[0.0, 0.0, 1.0] for _ in JOINT_NAMES_65]
    for idx in (15, 17, 18, 19):
        joints[idx] = [left_x, 0.0, left_z]
    for idx in (16, 20, 21, 22):
        joints[idx] = [right_x, 0.0, right_z]
    joints[0] = [0.2 + left_x, 0.0, 1.7]
    return SkeletonFrame(
        player_id="p1",
        frame_index=frame_index,
        t=frame_index / 30.0,
        joints_world=joints,
        joint_conf=[0.9] * len(joints),
    )


def test_resolve_foot_joint_indices_prefers_payload_names_for_65_joint_schema():
    indices = resolve_foot_joint_indices(JOINT_NAMES_65, joint_count=len(JOINT_NAMES_65))

    assert indices.left == (15, 17, 18, 19)
    assert indices.right == (16, 20, 21, 22)


def test_detect_contact_phases_uses_floor_height_speed_and_hysteresis():
    frames = [
        _frame(0, left_x=0.000, left_z=0.020),
        _frame(1, left_x=0.002, left_z=0.025),
        _frame(2, left_x=0.004, left_z=0.055),
        _frame(3, left_x=0.006, left_z=0.070),
        _frame(4, left_x=0.009, left_z=0.130),
    ]

    phases = detect_contact_phases(
        frames,
        joint_names=JOINT_NAMES_65,
        thresholds=ContactThresholds(
            enter_height_m=0.030,
            exit_height_m=0.080,
            enter_speed_mps=0.25,
            exit_speed_mps=0.40,
            min_phase_frames=2,
        ),
    )

    assert len(phases) == 1
    assert phases[0].foot == "left"
    assert phases[0].start_frame_index == 0
    assert phases[0].end_frame_index == 3
    assert phases[0].frame_count == 4


def test_detect_contact_phases_splits_internal_speed_inflection_and_exports_quality_fields():
    frames = [
        _frame(0, left_x=0.000, left_z=0.020),
        _frame(1, left_x=0.005, left_z=0.020),
        _frame(2, left_x=0.010, left_z=0.020),
        _frame(3, left_x=0.060, left_z=0.020),
        _frame(4, left_x=0.065, left_z=0.020),
        _frame(5, left_x=0.070, left_z=0.020),
    ]

    phases = detect_contact_phases(
        frames,
        joint_names=JOINT_NAMES_65,
        thresholds=ContactThresholds(
            enter_height_m=0.030,
            exit_height_m=0.080,
            enter_speed_mps=0.75,
            exit_speed_mps=2.00,
            split_speed_mps=0.75,
            min_phase_frames=2,
        ),
    )

    left_phases = [phase for phase in phases if phase.foot == "left"]
    assert [(phase.start_frame_index, phase.end_frame_index) for phase in left_phases] == [(0, 1), (4, 5)]
    payload = left_phases[0].to_dict()
    assert payload["foot_assignment"] == "per_foot_body_contact"
    assert payload["weak"] is False
    assert payload["demoted"] is False
    assert payload["split_reason"] == "internal_speed_inflection"
    assert payload["source_thresholds"]["split_speed_mps"] == pytest.approx(0.75)
    assert payload["min_confidence"] == pytest.approx(0.9)
    assert payload["max_height_m"] <= 0.03
    assert payload["max_speed_mps"] <= 0.75


def test_measure_contact_metrics_reports_phase_slide_and_foot_penetration():
    frames = [
        _frame(0, left_x=0.000, left_z=0.000),
        _frame(1, left_x=0.020, left_z=-0.010),
        _frame(2, left_x=0.035, left_z=0.010),
        _frame(3, left_x=0.100, left_z=0.200),
    ]
    phases = detect_contact_phases(
        frames[:3],
        joint_names=JOINT_NAMES_65,
        thresholds=ContactThresholds(
            enter_height_m=0.040,
            exit_height_m=0.070,
            enter_speed_mps=1.20,
            exit_speed_mps=1.20,
            min_phase_frames=2,
        ),
    )

    metrics = measure_contact_metrics(frames, phases, joint_names=JOINT_NAMES_65)

    assert metrics.phase_metrics[0].slide_mm == pytest.approx(35.0)
    assert metrics.penetration.max_penetration_mm == pytest.approx(10.0)
    assert metrics.summary_by_player["p1"].phase_count == 1
    assert metrics.summary_by_player["p1"].max_slide_mm == pytest.approx(35.0)


def _skeleton_payload(frames: list[SkeletonFrame]) -> dict:
    return {
        "schema_version": 1,
        "artifact_type": "synthetic_skeleton3d",
        "fps": 30.0,
        "joint_names": list(JOINT_NAMES_65),
        "players": [
            {
                "id": "p1",
                "frames": [
                    {
                        "frame_idx": frame.frame_index,
                        "t": frame.t,
                        "joints_world": frame.joints_world,
                        "joint_conf": frame.joint_conf,
                    }
                    for frame in frames
                ],
            }
        ],
    }


def _gate_stream_row(
    *,
    foot: str = "left",
    source_phase_foot: str = "left",
    body_detector_agreement: float = 0.91,
    rejection_reason: str | None = None,
    lock_metric_included: bool = True,
    slide_m: float = 0.001,
) -> dict:
    return {
        "phase_id": f"p1:{foot}:0-2:test",
        "player_id": "p1",
        "foot": foot,
        "source_phase_foot": source_phase_foot,
        "start_frame_index": 0,
        "end_frame_index": 2,
        "frame_count": 3,
        "min_confidence": 0.95,
        "max_height_m": 0.0,
        "max_speed_mps": 0.01,
        "slide_m": slide_m,
        "lock_metric_included": lock_metric_included,
        "rejection_reason": rejection_reason,
        "anchor_position_xyz": [0.0, 0.0, 0.0],
        "assignment_evidence": {
            "body_detector_agreement": body_detector_agreement,
            "body_detector_exact_agreement": body_detector_agreement,
            "source_phase_foot": source_phase_foot,
        },
    }


def _gate_stream_payload(row: dict) -> dict:
    return {"artifact_type": "foot_lock_gate_stream", "phase_rows": [row]}


def _crossover_skeleton_payload() -> dict:
    frames = []
    for idx, offset in enumerate((0.000, 0.001, 0.002)):
        joints = [[0.0, 0.0, 1.0] for _ in JOINT_NAMES_65]
        for joint_idx in (15, 17, 18, 19):
            joints[joint_idx] = [offset, 0.0, 0.010]
        for joint_idx in (16, 20, 21, 22):
            joints[joint_idx] = [0.005 - offset, 0.0, 0.010]
        joints[0] = [0.002, 0.0, 1.7]
        frames.append(
            {
                "frame_idx": idx,
                "t": idx / 30.0,
                "joints_world": joints,
                "joint_conf": [0.99 for _ in joints],
            }
        )
    return {
        "schema_version": 1,
        "artifact_type": "synthetic_skeleton3d",
        "fps": 30.0,
        "joint_names": list(JOINT_NAMES_65),
        "players": [{"id": "p1", "frames": frames}],
    }


def test_body_skeleton_phase_producer_emits_confident_left_contact() -> None:
    frames = [
        _frame(0, left_x=0.000, left_z=0.010, right_z=0.30),
        _frame(1, left_x=0.002, left_z=0.012, right_z=0.30),
        _frame(2, left_x=0.004, left_z=0.014, right_z=0.30),
    ]

    payload = build_body_skeleton_foot_contact_phases(_skeleton_payload(frames), clip="clear_left")

    assert payload["source_kind"] == "body_skeleton_direct"
    assert payload["phase_count"] == 1
    assert payload["rejected_phase_count"] == 0
    phase = payload["phases"][0]
    assert phase["foot"] == "left"
    assert phase["foot_assignment"] == "per_foot_body_contact"
    assert phase["source_phase_foot"] == "left"
    assert phase["assignment_evidence"]["body_detector_agreement"] == pytest.approx(1.0)
    assert phase["min_confidence"] >= 0.90
    assert payload["summary"]["confidence_formula"] == (
        "confident iff foot in {left,right}, min_confidence >= 0.90, "
        "source_phase_foot agrees with foot when present, "
        "assignment_evidence.body_detector_agreement >= 0.90, required quality fields present, "
        "no simultaneous confident opposite-foot single overlaps the same frame, "
        "and no independent rejection reason"
    )


def test_body_skeleton_phase_producer_emits_confident_right_contact() -> None:
    frames = [
        _frame(0, left_x=0.0, left_z=0.30, right_x=1.000, right_z=0.010),
        _frame(1, left_x=0.0, left_z=0.30, right_x=1.002, right_z=0.012),
        _frame(2, left_x=0.0, left_z=0.30, right_x=1.004, right_z=0.014),
    ]

    payload = build_body_skeleton_foot_contact_phases(_skeleton_payload(frames), clip="clear_right")

    assert payload["phase_count"] == 1
    assert payload["phases"][0]["foot"] == "right"
    assert payload["phases"][0]["assignment_evidence"]["body_detector_exact_agreement"] == pytest.approx(1.0)


def test_body_skeleton_phase_producer_rejects_ambiguous_low_confidence_contact() -> None:
    frames = [
        _frame(0, left_x=0.000, left_z=0.010, right_z=0.010),
        _frame(1, left_x=0.002, left_z=0.010, right_z=0.010),
        _frame(2, left_x=0.004, left_z=0.010, right_z=0.010),
    ]
    for frame in frames:
        frame.joint_conf[15] = 0.75
        frame.joint_conf[16] = 0.75

    payload = build_body_skeleton_foot_contact_phases(_skeleton_payload(frames), clip="ambiguous")

    assert payload["phase_count"] == 0
    assert payload["rejected_phase_count"] == 2
    assert payload["summary"]["rejected_reasons"] == {"low_body_contact_confidence": 2}
    assert all(phase["weak"] for phase in payload["rejected_phases"])


def test_body_skeleton_phase_producer_pins_detector_agreement_threshold() -> None:
    low = build_body_skeleton_foot_contact_phases_from_gate_stream(
        _gate_stream_payload(_gate_stream_row(body_detector_agreement=0.89)),
        clip="agreement_low",
    )
    high = build_body_skeleton_foot_contact_phases_from_gate_stream(
        _gate_stream_payload(_gate_stream_row(body_detector_agreement=0.91)),
        clip="agreement_high",
    )

    assert low["phase_count"] == 0
    assert low["rejected_phase_count"] == 1
    assert low["summary"]["rejected_reasons"] == {"low_body_detector_agreement": 1}
    assert high["phase_count"] == 1
    assert high["rejected_phase_count"] == 0
    assert high["phases"][0]["assignment_evidence"]["body_detector_agreement"] == pytest.approx(0.91)


def test_body_skeleton_phase_producer_rejects_source_phase_foot_mismatch() -> None:
    payload = build_body_skeleton_foot_contact_phases_from_gate_stream(
        _gate_stream_payload(_gate_stream_row(foot="left", source_phase_foot="right", body_detector_agreement=0.99)),
        clip="mismatch",
    )

    assert payload["phase_count"] == 0
    assert payload["rejected_phase_count"] == 1
    assert payload["summary"]["rejected_reasons"] == {"source_phase_foot_mismatch": 1}


def test_body_skeleton_phase_producer_demotes_crossover_not_dual_confident() -> None:
    payload = build_body_skeleton_foot_contact_phases(_crossover_skeleton_payload(), clip="crossover")

    feet_by_frame: dict[int, set[str]] = {}
    for phase in payload["phases"]:
        for frame_index in phase["frame_indices"]:
            feet_by_frame.setdefault(int(frame_index), set()).add(str(phase["foot"]))

    assert not any(feet == {"left", "right"} for feet in feet_by_frame.values())
    assert payload["phase_count"] == 0
    assert payload["summary"]["rejected_reasons"] == {"simultaneous_bilateral_contact": 2}


def test_body_skeleton_phase_producer_does_not_reject_using_gate_threshold_reason() -> None:
    payload = build_body_skeleton_foot_contact_phases_from_gate_stream(
        _gate_stream_payload(
            _gate_stream_row(
                body_detector_agreement=0.99,
                rejection_reason="_".join(("phase", "slide", "exceeds", "lock", "gate")),
                lock_metric_included=False,
                slide_m=0.040,
            )
        ),
        clip="threshold_exclusion",
    )

    assert payload["phase_count"] == 1
    assert payload["rejected_phase_count"] == 0
    assert payload["summary"]["rejected_reasons"] == {}
    assert payload["summary"]["max_candidate_phase_slide_m"] == pytest.approx(0.040)

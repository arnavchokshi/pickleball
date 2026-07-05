from __future__ import annotations

import copy
import math

import pytest

from threed.racketsport.external_gt_body_prediction_schema import MHR70_JOINT_NAMES
from threed.racketsport.foot_pin import FootPinSettings, apply_foot_pin_to_payload


LEFT_FOOT = (13, 15, 16, 17)
RIGHT_FOOT = (14, 18, 19, 20)
BONE_PAIRS = ((5, 7), (7, 62), (9, 11), (11, 13), (10, 12), (12, 14))


def _player_frame(
    frame_idx: int,
    *,
    left_x: float,
    right_x: float,
    conf: float = 0.95,
    right_contact: bool = False,
) -> dict:
    joints = [[0.0, 0.0, 1.0] for _idx in range(70)]
    joints[5] = [0.0 + left_x, 0.30, 1.45]
    joints[7] = [0.15 + left_x, 0.28, 1.15]
    joints[62] = [0.25 + left_x, 0.25, 0.95]
    joints[9] = [0.0 + left_x, 0.10, 0.95]
    joints[11] = [0.0 + left_x, 0.08, 0.48]
    for idx in LEFT_FOOT:
        joints[idx] = [left_x, 0.0, 0.0]

    joints[6] = [right_x, -0.30, 1.45]
    joints[8] = [right_x - 0.15, -0.28, 1.15]
    joints[41] = [right_x - 0.25, -0.25, 0.95]
    joints[10] = [right_x, -0.10, 0.95]
    joints[12] = [right_x, -0.08, 0.48]
    right_z = 0.0 if right_contact else 0.20
    for idx in RIGHT_FOOT:
        joints[idx] = [right_x, 0.0, right_z]

    return {
        "frame_idx": frame_idx,
        "t": frame_idx / 30.0,
        "track_world_xy": [left_x, 0.0],
        "floor_world_xyz": [left_x, 0.0, 0.0],
        "transl_world": [left_x, 0.0, 0.0],
        "joints_world": joints,
        "joint_conf": [conf] * 70,
        "sentinel": {"unchanged": True},
    }


def _payload(frames: list[dict]) -> dict:
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_virtual_world",
        "fps": 30.0,
        "world_frame": "court_Z0",
        "joint_names": list(MHR70_JOINT_NAMES),
        "players": [{"id": "p1", "frames": frames, "role": "player"}],
        "summary": {"kept": "byte-identical"},
    }


def _xy(point: list[float]) -> tuple[float, float]:
    return float(point[0]), float(point[1])


def _distance(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b, strict=True)))


def _bone_lengths(frame: dict) -> list[float]:
    joints = frame["joints_world"]
    return [_distance(joints[a], joints[b]) for a, b in BONE_PAIRS]


def test_foot_pin_pins_confident_stance_to_median_anchor_and_preserves_limb_geometry() -> None:
    source = _payload([_player_frame(idx, left_x=0.01 * idx, right_x=1.0) for idx in range(5)])
    before_lengths = _bone_lengths(source["players"][0]["frames"][2])

    result = apply_foot_pin_to_payload(source, settings=FootPinSettings(taper_frames=0))

    corrected_frames = result.payload["players"][0]["frames"]
    left_xs = [corrected_frames[idx]["joints_world"][13][0] for idx in range(5)]
    assert left_xs == pytest.approx([0.02] * 5)
    assert [frame["track_world_xy"][0] for frame in corrected_frames] == pytest.approx([0.00, 0.01, 0.02, 0.03, 0.04])
    assert result.audit["summary"]["stance_slide_after_mm"]["median"] == pytest.approx(0.0)
    assert result.audit["players"]["p1"]["phase_count"] == 1
    assert result.audit["players"]["p1"]["max_limb_length_delta_m"] == pytest.approx(0.0)
    assert _bone_lengths(corrected_frames[2]) == pytest.approx(before_lengths)
    assert source["players"][0]["frames"][0]["joints_world"][13][0] == pytest.approx(0.0)


def test_foot_pin_applies_full_strength_inside_stance_without_moving_track_or_arms() -> None:
    source = _payload([_player_frame(idx, left_x=0.08 * idx, right_x=1.0) for idx in range(5)])
    before_left_arm = [
        [*source["players"][0]["frames"][frame_idx]["joints_world"][joint_idx]]
        for frame_idx in range(5)
        for joint_idx in (5, 7, 62)
    ]

    result = apply_foot_pin_to_payload(
        source,
        settings=FootPinSettings(taper_frames=0, enter_speed_mps=3.0, exit_speed_mps=3.0),
    )

    corrected_frames = result.payload["players"][0]["frames"]
    assert [frame["track_world_xy"][0] for frame in corrected_frames] == pytest.approx([0.0, 0.08, 0.16, 0.24, 0.32])
    assert [frame["joints_world"][13][0] for frame in corrected_frames] == pytest.approx([0.16] * 5)
    assert result.audit["summary"]["max_correction_m"] > 0.02
    assert result.audit["summary"]["stance_slide_after_mm"]["p95"] == pytest.approx(0.0)
    after_left_arm = [
        [*corrected_frames[frame_idx]["joints_world"][joint_idx]]
        for frame_idx in range(5)
        for joint_idx in (5, 7, 62)
    ]
    for actual, expected in zip(after_left_arm, before_left_arm, strict=True):
        assert actual == pytest.approx(expected)


def test_foot_pin_solves_both_feet_with_shared_root_offset() -> None:
    frames = [
        _player_frame(idx, left_x=0.01 * idx, right_x=1.0 + 0.01 * idx, right_contact=True)
        for idx in range(5)
    ]
    source = _payload(frames)

    result = apply_foot_pin_to_payload(source, settings=FootPinSettings(taper_frames=0))

    corrected_frames = result.payload["players"][0]["frames"]
    assert [corrected_frames[idx]["joints_world"][13][0] for idx in range(5)] == pytest.approx([0.02] * 5)
    assert [corrected_frames[idx]["joints_world"][14][0] for idx in range(5)] == pytest.approx([1.02] * 5)
    active = result.audit["players"]["p1"]["frame_corrections"][2]["active_contacts"]
    assert {contact["foot"] for contact in active} == {"left", "right"}
    assert result.audit["summary"]["stance_slide_after_mm"]["p95"] == pytest.approx(0.0)


def test_foot_pin_skips_low_confidence_phases_without_mutating_payload() -> None:
    source = _payload([_player_frame(idx, left_x=0.01 * idx, right_x=1.0, conf=0.05) for idx in range(4)])
    original = copy.deepcopy(source)

    result = apply_foot_pin_to_payload(source, settings=FootPinSettings(taper_frames=0, min_phase_confidence=0.25))

    assert result.payload["players"][0]["frames"] == original["players"][0]["frames"]
    assert result.audit["players"]["p1"]["phase_count"] == 0
    assert result.audit["players"]["p1"]["skipped_low_confidence_phase_count"] == 1
    assert result.audit["summary"]["total_skipped_low_confidence_phases"] == 1


def test_foot_pin_records_provenance_without_changing_unrelated_fields() -> None:
    source = _payload([_player_frame(idx, left_x=0.01 * idx, right_x=1.0) for idx in range(4)])

    result = apply_foot_pin_to_payload(source, settings=FootPinSettings(taper_frames=0))

    payload = result.payload
    assert payload["summary"] == source["summary"]
    assert payload["players"][0]["role"] == source["players"][0]["role"]
    assert payload["players"][0]["frames"][0]["sentinel"] == source["players"][0]["frames"][0]["sentinel"]
    assert payload["foot_pin"]["version"] == 1
    assert payload["foot_pin"]["audit"]["artifact_type"] == "foot_pin_audit"


def test_foot_pin_preserves_no_data_world_frames_without_using_them_for_detection() -> None:
    frames = [_player_frame(0, left_x=0.00, right_x=1.0), {"t": 1 / 30.0, "joints_world": [], "joint_conf": []}]
    frames.extend(_player_frame(idx, left_x=0.01 * idx, right_x=1.0) for idx in range(2, 4))
    source = _payload(frames)

    result = apply_foot_pin_to_payload(source, settings=FootPinSettings(taper_frames=0))

    assert result.payload["players"][0]["frames"][1] == source["players"][0]["frames"][1]
    assert result.audit["players"]["p1"]["phase_count"] == 1
    assert result.audit["summary"]["stance_slide_after_mm"]["median"] == pytest.approx(0.0)


def test_foot_pin_does_not_interpolate_or_mutate_track_anchor_between_stance_knots() -> None:
    frames = [
        _player_frame(0, left_x=0.00, right_x=1.0),
        _player_frame(1, left_x=0.50, right_x=1.0),
        _player_frame(2, left_x=0.02, right_x=1.0),
    ]
    frames[1]["track_world_xy"] = [0.50, 0.0]
    for idx in LEFT_FOOT:
        frames[1]["joints_world"][idx][2] = 0.30
    source = _payload(frames)

    result = apply_foot_pin_to_payload(
        source,
        settings=FootPinSettings(
            min_phase_frames=1,
            enter_speed_mps=100.0,
            exit_speed_mps=100.0,
            interpolate_between_stances=True,
        ),
    )

    middle = result.payload["players"][0]["frames"][1]
    assert middle["track_world_xy"] == pytest.approx([0.50, 0.0])
    assert middle["transl_world"] == pytest.approx([0.50, 0.0, 0.0])
    assert result.audit["summary"]["total_corrected_frame_count"] == 0


def test_foot_pin_skips_over_cap_corrections_per_foot_frame_instead_of_failing() -> None:
    from threed.racketsport.foot_contact import ContactPhase

    # Left foot sits at x=0 for frames 0-1 and x=0.7 for frames 2-4; the phase
    # median anchor lands at 0.7, so frames 0-1 would need a 0.7 m correction --
    # far beyond the 0.30 m in-stance cap. The refine must NOT raise: it skips
    # those foot/frames fail-closed and records them in the audit.
    frames = [
        _player_frame(idx, left_x=(0.0 if idx < 2 else 0.7), right_x=1.0) for idx in range(5)
    ]
    source = _payload(frames)
    phase = ContactPhase(
        player_id="p1",
        foot="left",
        frame_indices=(0, 1, 2, 3, 4),
        start_time_s=0.0,
        end_time_s=4 / 30.0,
        anchor_position_xyz=(0.7, 0.0, 0.0),
        max_height_m=0.0,
        max_speed_mps=0.0,
        min_confidence=0.95,
    )

    result = apply_foot_pin_to_payload(
        source, settings=FootPinSettings(taper_frames=0), contact_phases=[phase]
    )

    summary = result.audit["summary"]
    assert summary["cap_exceeded_skip_count"] >= 1
    skips = result.audit["phase_detection"]["cap_exceeded_skips"]
    assert all(event["magnitude_m"] > event["cap_m"] for event in skips)
    # The whole inconsistent phase is dropped (not just its over-cap frames), so no
    # pin lands anywhere in frames 0-4 and there is no on/off jump inside the stance.
    assert any(
        event.get("kind") == "phase_skipped"
        and event["start_frame_index"] == 0
        and event["end_frame_index"] == 4
        for event in skips
    )
    corrected_frames = result.payload["players"][0]["frames"]
    for idx in range(5):
        assert corrected_frames[idx]["joints_world"][13][0] == pytest.approx(
            source["players"][0]["frames"][idx]["joints_world"][13][0]
        )

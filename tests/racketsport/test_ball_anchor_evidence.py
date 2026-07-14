from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.racketsport.test_ball_arc_solver import _project, _projection_calibration
from threed.racketsport.ball_anchor_evidence import (
    AnchorEvidenceConfig,
    SOURCE_AUDIO,
    SOURCE_BLUR,
    SOURCE_COURT,
    SOURCE_KINEMATICS,
    build_anchor_evidence_payload,
)


def test_anchor_evidence_refuses_when_no_source_has_usable_evidence() -> None:
    payload = build_anchor_evidence_payload(
        ball_track={"fps": 30.0, "frames": []},
        calibration=None,
        audio_onsets=None,
        raw_ball_candidates=None,
        blur_sidecar=None,
        clip_id="empty",
    )

    assert payload["status"] == "refused_no_evidence"
    assert payload["candidates"] == []
    assert payload["solver_wired"] is False
    assert payload["policy"]["reviewed_timestamps_consumed"] is False
    assert set(payload["refusal_reasons"]) == {
        "no_ball_track_frames",
        "no_raw_ball_candidates",
        "no_audio_onsets",
        "no_blur_records",
        "no_calibration",
    }


def test_audio_source_uses_corrected_time_and_preserves_spine_provenance(tmp_path: Path) -> None:
    audio_path = tmp_path / "audio_onsets_v2.json"
    audio_path.write_text("{}\n", encoding="utf-8")
    track = {"fps": 30.0, "frames": [_hidden_frame(index / 30.0) for index in range(31)]}

    payload = build_anchor_evidence_payload(
        ball_track=track,
        calibration=None,
        audio_onsets={
            "onsets": [
                {
                    "raw_time_s": 0.240,
                    "corrected_time_s": 0.200,
                    "confidence": 0.8,
                }
            ]
        },
        clip_id="audio",
        audio_source_path=audio_path,
    )

    candidate = payload["candidates"][0]
    assert candidate["t"] == pytest.approx(0.2)
    assert candidate["anchor_type"] == "contact"
    assert candidate["source_types"] == [SOURCE_AUDIO]
    source = candidate["sources"][0]
    assert source["raw_time_s"] == pytest.approx(0.24)
    assert source["corrected_time_s"] == pytest.approx(0.2)
    assert source["timing_used"] == "corrected_time_s"
    assert source["spine_audio_provenance"]["status"] == "present"


def test_audio_source_repairs_corrected_order_and_preserves_classified_event_type(tmp_path: Path) -> None:
    audio_path = tmp_path / "audio_onsets_v2.json"
    audio_path.write_text("{}\n", encoding="utf-8")
    track = {"fps": 30.0, "frames": [_hidden_frame(index / 30.0) for index in range(61)]}

    payload = build_anchor_evidence_payload(
        ball_track=track,
        calibration=None,
        audio_onsets={
            "onsets": [
                {
                    "raw_time_s": 1.10,
                    "corrected_time_s": 1.00,
                    "corrected_order": 1,
                    "class_label": "contact",
                    "confidence": 0.8,
                },
                {
                    "raw_time_s": 0.95,
                    "corrected_time_s": 0.90,
                    "corrected_order": 0,
                    "class_label": "bounce",
                    "confidence": 0.9,
                },
            ]
        },
        clip_id="classified-audio",
        audio_source_path=audio_path,
    )

    by_time = sorted(payload["candidates"], key=lambda item: item["t"])
    assert [item["anchor_type"] for item in by_time] == ["bounce", "contact"]
    first_source = by_time[0]["sources"][0]
    second_source = by_time[1]["sources"][0]
    assert first_source["classification"] == "bounce"
    assert first_source["corrected_order"] == 0
    assert second_source["classification"] == "contact"
    assert second_source["corrected_order"] == 1


def test_kinematics_uses_low_confidence_raw_candidate_below_acceptance_threshold() -> None:
    track = {
        "fps": 30.0,
        "frames": [
            {"t": 0.0, "xy": [100.0, 100.0], "conf": 0.9, "visible": True},
            _hidden_frame(1.0 / 30.0),
            {"t": 2.0 / 30.0, "xy": [100.0, 100.0], "conf": 0.9, "visible": True},
        ],
    }
    raw = {
        "fps": 30.0,
        "artifact_type": "racketsport_wasb_below_threshold_candidates",
        "frames": [
            {"frame": 1, "pts_seconds": 0.041, "candidates": [{"xy": [100.0, 115.0], "score": 0.4, "source_detector": "wasb_concomp"}]}
        ],
    }

    payload = build_anchor_evidence_payload(
        ball_track=track,
        calibration=None,
        raw_ball_candidates=raw,
        clip_id="raw-low-confidence",
        config=AnchorEvidenceConfig(source_min_separation_s=0.01),
    )

    assert payload["status"] == "ranked_candidates"
    assert payload["source_summary"][SOURCE_KINEMATICS]["cues_below_accepted_confidence_threshold"] >= 1
    candidate = payload["candidates"][0]
    assert candidate["anchor_type"] == "bounce"
    assert candidate["t"] == pytest.approx(0.041)
    kinematic = next(source for source in candidate["sources"] if source["source_type"] == SOURCE_KINEMATICS)
    assert kinematic["raw_candidate_support"] is True
    assert kinematic["below_accepted_confidence_support"] is True
    assert any(
        item.get("artifact") == "ball_candidates" and item.get("below_primary_selection") is True
        for item in kinematic["trajectory_inputs"]
    )
    raw_input = next(item for item in kinematic["trajectory_inputs"] if item.get("artifact") == "ball_candidates")
    assert raw_input["pts_seconds"] == pytest.approx(0.041)
    assert raw_input["source_artifact_type"] == "racketsport_wasb_below_threshold_candidates"


def test_kinematics_emits_vertical_flip_and_direction_break_with_court_hypothesis() -> None:
    calibration = _projection_calibration()
    world = [
        (-1.0, -2.0, 1.0),
        (-0.5, -1.0, 0.5),
        (0.0, 0.0, 0.1),
        (0.5, -1.0, 0.5),
        (1.0, -2.0, 1.0),
    ]
    track = {
        "fps": 10.0,
        "frames": [
            {"t": index / 10.0, "xy": list(_project(calibration, xyz)), "conf": 0.95, "visible": True}
            for index, xyz in enumerate(world)
        ],
    }

    payload = build_anchor_evidence_payload(
        ball_track=track,
        calibration=calibration,
        clip_id="kinematic-court",
        config=AnchorEvidenceConfig(
            source_min_separation_s=0.01,
            min_vertical_speed_px_s=1.0,
            min_direction_break_deg=5.0,
            max_track_step_px=1000.0,
        ),
    )

    assert payload["source_summary"][SOURCE_KINEMATICS]["cue_count"] >= 2
    assert payload["source_summary"][SOURCE_COURT]["cue_count"] >= 1
    assert any(SOURCE_COURT in candidate["source_types"] for candidate in payload["candidates"])
    court_positions = [
        position
        for candidate in payload["candidates"]
        for position in candidate["position_hypotheses"]
        if position["source_type"] == SOURCE_COURT
    ]
    assert court_positions
    assert all(position["semantics"] == "ray_intersection_with_ball_radius_plane" for position in court_positions)


def test_blur_transition_contributes_typed_contact_candidate() -> None:
    payload = build_anchor_evidence_payload(
        ball_track={"fps": 30.0, "frames": [_hidden_frame(index / 30.0) for index in range(5)]},
        calibration=None,
        blur_sidecar={
            "artifact_type": "racketsport_ball_blur_sidecar",
            "frames": [
                {
                    "frame_index": 1,
                    "center_xy": [50.0, 60.0],
                    "blur_length_px": 2.0,
                    "blur_angle_deg": 10.0,
                    "quality": "clear",
                },
                {
                    "frame_index": 2,
                    "center_xy": [52.0, 61.0],
                    "blur_length_px": 12.0,
                    "blur_angle_deg": 70.0,
                    "quality": "clear",
                },
            ],
        },
        clip_id="blur",
    )

    assert payload["source_summary"][SOURCE_BLUR]["cue_count"] == 1
    candidate = payload["candidates"][0]
    assert candidate["anchor_type"] == "contact"
    assert candidate["source_types"] == [SOURCE_BLUR]
    assert candidate["position_hypotheses"][0]["space"] == "image_px"


def test_blur_vertical_motion_reversal_contributes_typed_bounce_with_provenance() -> None:
    payload = build_anchor_evidence_payload(
        ball_track={"fps": 30.0, "frames": [_hidden_frame(index / 30.0) for index in range(6)]},
        calibration=None,
        blur_sidecar={
            "artifact_type": "racketsport_ball_blur_sidecar",
            "frames": [
                {
                    "frame_index": 1,
                    "center_xy": [50.0, 50.0],
                    "blur_length_px": 8.0,
                    "blur_angle_deg": 70.0,
                    "quality": "clear",
                },
                {
                    "frame_index": 2,
                    "center_xy": [52.0, 60.0],
                    "blur_length_px": 2.0,
                    "blur_angle_deg": 5.0,
                    "quality": "clear",
                },
                {
                    "frame_index": 3,
                    "center_xy": [54.0, 50.0],
                    "blur_length_px": 8.0,
                    "blur_angle_deg": 110.0,
                    "quality": "clear",
                },
            ],
        },
        clip_id="blur-bounce",
        config=AnchorEvidenceConfig(source_min_separation_s=0.01),
    )

    candidate = next(item for item in payload["candidates"] if item["anchor_type"] == "bounce")
    source = next(item for item in candidate["sources"] if item["source_type"] == SOURCE_BLUR)
    assert source["cue_type"] == "motion_blur_bounce_transition"
    assert source["proposal_type"] == "bounce"
    assert source["vertical_velocity_sign_flip"] is True
    assert source["signature_frames"] == [1, 2, 3]
    assert source["extraction"] == "frame_difference_ball_crop_principal_axis"


def test_fusion_ranks_multisource_agreement_above_isolated_audio(tmp_path: Path) -> None:
    audio_path = tmp_path / "audio_onsets_v2.json"
    audio_path.write_text(json.dumps({"artifact_type": "audio"}), encoding="utf-8")
    track = {
        "fps": 10.0,
        "frames": [
            {"t": 0.0, "xy": [100.0, 100.0], "conf": 0.9, "visible": True},
            {"t": 0.1, "xy": [110.0, 110.0], "conf": 0.9, "visible": True},
            {"t": 0.2, "xy": [120.0, 120.0], "conf": 0.9, "visible": True},
            {"t": 0.3, "xy": [130.0, 110.0], "conf": 0.9, "visible": True},
            {"t": 0.4, "xy": [140.0, 100.0], "conf": 0.9, "visible": True},
            *[_hidden_frame(index / 10.0) for index in range(5, 11)],
        ],
    }
    blur = {
        "artifact_type": "racketsport_ball_blur_sidecar",
        "frames": [
            {"frame_index": 1, "center_xy": [110.0, 110.0], "blur_length_px": 2.0, "blur_angle_deg": 5.0, "quality": "clear"},
            {"frame_index": 2, "center_xy": [120.0, 120.0], "blur_length_px": 10.0, "blur_angle_deg": 50.0, "quality": "clear"},
        ],
    }

    payload = build_anchor_evidence_payload(
        ball_track=track,
        calibration=None,
        audio_onsets={
            "onsets": [
                {"raw_time_s": 0.22, "corrected_time_s": 0.205, "confidence": 0.8},
                {"raw_time_s": 0.80, "corrected_time_s": 0.80, "confidence": 0.8},
            ]
        },
        blur_sidecar=blur,
        clip_id="fusion",
        audio_source_path=audio_path,
        config=AnchorEvidenceConfig(
            source_min_separation_s=0.01,
            min_vertical_speed_px_s=1.0,
            min_direction_break_deg=5.0,
        ),
    )

    first, second = payload["candidates"][:2]
    assert first["rank_in_rally"] == 1
    assert set(first["source_types"]) == {SOURCE_AUDIO, SOURCE_BLUR, SOURCE_KINEMATICS}
    assert first["confidence"] > second["confidence"]
    assert second["source_types"] == [SOURCE_AUDIO]


def _hidden_frame(t: float) -> dict[str, object]:
    return {"t": t, "xy": [0.0, 0.0], "conf": 0.0, "visible": False}

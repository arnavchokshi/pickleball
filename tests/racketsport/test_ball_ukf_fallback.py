from __future__ import annotations

import pytest

from threed.racketsport.ball_ukf_fallback import (
    RecoveryPolicyV2Config,
    UkfFallbackConfig,
    build_recovery_policy_v2,
    build_ukf_fallback,
)


def _segment(
    segment_id: int,
    frame_start: int,
    frame_end: int,
    *,
    status: str,
    p0: tuple[float, float, float],
    v0: tuple[float, float, float],
) -> dict[str, object]:
    return {
        "segment_id": segment_id,
        "frame_start": frame_start,
        "frame_end": frame_end,
        "t0": frame_start / 30.0,
        "t1": frame_end / 30.0,
        "initial_position_m": list(p0),
        "initial_velocity_mps": list(v0),
        "initial_speed_mps": sum(value * value for value in v0) ** 0.5,
        "status": status,
        "physical_sanity": {"violation": False, "violations": []},
        "inlier_count": 6 if status == "fit" else 0,
        "outlier_count": 0 if status == "fit" else frame_end - frame_start + 1,
        "max_reprojection_error_px": 2.0 if status == "fit" else 200.0,
    }


def _artifact(*, fallback_end: int = 5, seed_speed_mps: float = 3.0) -> dict[str, object]:
    frames = [
        {"t": index / 30.0, "visible": False, "xy": None, "conf": 0.0}
        for index in range(12)
    ]
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_ball_track_arc_solved",
        "clip_id": "unit_clip",
        "status": "ran",
        "fps": 30.0,
        "frames": frames,
        "physics_parameters": {"gravity_mps2": 9.81, "drag_coefficient": 0.0},
        "segments": [
            _segment(0, 0, 2, status="fit", p0=(0.0, -3.0, 2.0), v0=(0.0, seed_speed_mps, 2.0)),
            _segment(1, 3, fallback_end, status="fit_bvp_fallback", p0=(0.0, -2.7, 2.0), v0=(0.0, 2.0, 1.0)),
            _segment(2, fallback_end + 1, fallback_end + 3, status="fit", p0=(0.0, -2.4, 2.0), v0=(0.0, seed_speed_mps, 1.0)),
        ],
    }


def test_bracketed_short_gap_uses_ukf_and_rts_with_exact_pts_and_predicted_band() -> None:
    artifact = _artifact()

    result = build_ukf_fallback(artifact, generated_at="2026-07-10T00:00:00Z")

    assert result["summary"]["recovered_sample_count"] == 3
    assert result["summary"]["rts_smoothed_gap_count"] == 1
    assert [sample["frame_index"] for sample in result["samples"]] == [3, 4, 5]
    assert [sample["t"] for sample in result["samples"]] == pytest.approx(
        [artifact["frames"][index]["t"] for index in (3, 4, 5)]
    )
    for sample in result["samples"]:
        assert sample["source"] == "physics_interpolated"
        assert sample["band"] == "physics_predicted"
        assert sample["trust_band"]["band"] == "low_confidence"
        assert sample["method"] == "ukf_forward_rts_backward"
        assert len(sample["covariance_position_m2"]) == 3
        assert sample["horizon_age_frames"] >= 1
        assert sample["measured"] is False
    assert result["measurement_covariance_policy"]["wasb_heatmap_footprint"] == "not_persisted"


def test_gap_crossing_contact_proposal_is_refused_whole() -> None:
    result = build_ukf_fallback(
        _artifact(),
        contact_proposals={"events": [{"type": "contact", "frame": 4, "confidence": 0.6}]},
    )

    assert result["samples"] == []
    assert result["summary"]["contact_refused_gap_count"] == 1
    assert result["refused_gaps"][0]["reason"] == "contact_proposal_inside_gap"


def test_contact_on_seed_boundary_is_still_across_the_prediction_and_refused() -> None:
    result = build_ukf_fallback(
        _artifact(),
        contact_proposals={"events": [{"type": "contact", "frame": 2, "confidence": 0.6}]},
    )

    assert result["samples"] == []
    assert result["summary"]["contact_refused_gap_count"] == 1
    assert result["refused_gaps"][0]["reason"] == "contact_proposal_inside_gap"


def test_contact_window_pts_intersection_is_refused_even_when_peak_frame_is_elsewhere() -> None:
    result = build_ukf_fallback(
        _artifact(),
        contact_proposals={
            "events": [
                {
                    "type": "contact",
                    "frame": 10,
                    "confidence": 0.6,
                    "window": {"t0": 0.09, "t1": 0.12},
                }
            ]
        },
    )

    assert result["samples"] == []
    assert result["summary"]["contact_refused_gap_count"] == 1
    assert result["refused_gaps"][0]["reason"] == "contact_proposal_inside_gap"


def test_net_gate_includes_seed_to_first_prediction_boundary() -> None:
    artifact = _artifact()
    left, _, right = artifact["segments"]
    left["initial_position_m"] = [0.0, -0.60, 0.20]
    left["initial_velocity_mps"] = [0.0, 8.0, 0.0]
    right["initial_position_m"] = [0.0, 1.0, 1.50]
    right["initial_velocity_mps"] = [0.0, 8.0, 0.0]

    result = build_ukf_fallback(artifact)

    assert result["samples"] == []
    assert result["summary"]["hard_gate_violation_count"] == 1
    assert result["refused_gaps"][0]["reason"] == "net_clearance_below_slack"


def test_speed_gate_refuses_physically_implausible_prediction() -> None:
    result = build_ukf_fallback(_artifact(seed_speed_mps=70.0))

    assert result["samples"] == []
    assert result["summary"]["hard_gate_violation_count"] == 1
    assert result["refused_gaps"][0]["reason"] == "speed_above_ceiling"


def test_one_sided_gap_obeys_documented_horizon_cap() -> None:
    artifact = _artifact(fallback_end=9)
    artifact["segments"] = artifact["segments"][:2]

    result = build_ukf_fallback(
        artifact,
        config=UkfFallbackConfig(max_gap_frames=12, max_one_sided_horizon_frames=6),
    )

    assert result["samples"] == []
    assert result["refused_gaps"][0]["reason"] == "one_sided_horizon_exceeded"


@pytest.mark.parametrize(
    ("mutation", "config", "expected_reason"),
    [
        ("court", None, "outside_court_volume"),
        ("height", None, "height_above_ceiling"),
        (
            "covariance",
            UkfFallbackConfig(max_position_covariance_m2=1e-8),
            "covariance_above_ceiling",
        ),
        ("net", None, "net_clearance_below_slack"),
    ],
)
def test_each_remaining_hard_gate_refuses_the_whole_gap(
    mutation: str,
    config: UkfFallbackConfig | None,
    expected_reason: str,
) -> None:
    artifact = _artifact()
    left, _, right = artifact["segments"]
    if mutation == "court":
        left["initial_position_m"] = [8.0, -3.0, 2.0]
        right["initial_position_m"] = [8.0, -2.4, 2.0]
    elif mutation == "height":
        left["initial_position_m"] = [0.0, -3.0, 13.0]
        right["initial_position_m"] = [0.0, -2.4, 13.0]
    elif mutation == "net":
        left["initial_position_m"] = [0.0, -0.30, 0.25]
        left["initial_velocity_mps"] = [0.0, 2.0, 0.0]
        right["initial_position_m"] = [0.0, 0.10, 0.20]
        right["initial_velocity_mps"] = [0.0, 2.0, 0.0]

    result = build_ukf_fallback(artifact, config=config)

    assert result["samples"] == []
    assert result["summary"]["hard_gate_violation_count"] == 1
    assert result["refused_gaps"][0]["reason"] == expected_reason


def test_v2_bridges_a_full_multi_segment_suppressed_span_with_honest_provenance() -> None:
    artifact = _artifact(fallback_end=6)
    artifact["segments"] = [
        artifact["segments"][0],
        _segment(1, 3, 4, status="fit_bvp_fallback", p0=(0.0, -2.7, 2.0), v0=(0.0, 3.0, 1.0)),
        _segment(2, 4, 6, status="fit_bvp_fallback", p0=(0.0, -2.5, 2.0), v0=(0.0, 3.0, 1.0)),
        _segment(3, 7, 9, status="fit", p0=(0.0, -2.3, 1.9), v0=(0.0, 3.0, -0.3)),
    ]

    result = build_recovery_policy_v2(artifact, generated_at="2026-07-12T00:00:00Z")

    assert [sample["frame_index"] for sample in result["samples"]] == [3, 4, 5, 6]
    assert result["summary"]["two_sided_bridge_gap_count"] == 1
    assert result["policy_status"]["two_sided_requires_full_gap"] is True
    for sample in result["samples"]:
        assert sample["band"] == "physics_predicted"
        assert sample["source"] == "physics_interpolated"
        assert sample["measured"] is False
        assert sample["horizon_age_frames"] >= 1
        assert sample["position_covariance_max_m2"] <= 0.25


def test_v2_two_sided_policy_is_separately_killable() -> None:
    result = build_recovery_policy_v2(
        _artifact(),
        config=RecoveryPolicyV2Config(enable_two_sided_bridge=False),
    )

    assert result["samples"] == []
    assert result["refused_gaps"][0]["reason"] == "two_sided_bridge_policy_disabled"


def test_v2_one_sided_covariance_horizon_can_exceed_v1_twelve_frame_cap() -> None:
    artifact = _artifact(fallback_end=20)
    artifact["frames"] = [
        {"t": index / 30.0, "visible": False, "xy": None, "conf": 0.0}
        for index in range(30)
    ]
    artifact["segments"] = artifact["segments"][:2]

    result = build_recovery_policy_v2(artifact)

    assert result["summary"]["one_sided_gap_count"] == 1
    assert len(result["samples"]) > 12
    assert max(sample["position_covariance_max_m2"] for sample in result["samples"]) <= 0.25
    assert all(sample["horizon_age_frames"] >= 1 for sample in result["samples"])


def test_v2_covariance_one_sided_policy_is_separately_killable() -> None:
    artifact = _artifact(fallback_end=9)
    artifact["segments"] = artifact["segments"][:2]

    result = build_recovery_policy_v2(
        artifact,
        config=RecoveryPolicyV2Config(enable_covariance_one_sided=False),
    )

    assert result["samples"] == []
    assert result["refused_gaps"][0]["reason"] == "covariance_one_sided_policy_disabled"


def test_v2_optional_low_confidence_2d_updates_keep_samples_predicted() -> None:
    artifact = _artifact()
    for index in (3, 4, 5):
        artifact["frames"][index].update({"visible": True, "conf": 0.2, "xy": [960.0, 320.0]})
    calibration = {
        "intrinsics": {"fx": 1000.0, "fy": 1000.0, "cx": 960.0, "cy": 540.0},
        "extrinsics": {
            "R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "t": [0.0, 0.0, 12.0],
        },
    }

    result = build_recovery_policy_v2(
        artifact,
        calibration=calibration,
        config=RecoveryPolicyV2Config(enable_low_confidence_2d_updates=True),
    )

    assert result["summary"]["low_confidence_2d_update_count"] > 0
    supported = [sample for sample in result["samples"] if sample["low_confidence_2d_measurement_support"]]
    assert supported
    assert all(sample["measured"] is False for sample in supported)
    assert all(sample["band"] == "physics_predicted" for sample in supported)


def test_v2_low_confidence_policy_is_separately_killable() -> None:
    artifact = _artifact()
    for index in (3, 4, 5):
        artifact["frames"][index].update({"visible": True, "conf": 0.2, "xy": [960.0, 320.0]})

    result = build_recovery_policy_v2(
        artifact,
        config=RecoveryPolicyV2Config(enable_low_confidence_2d_updates=False),
    )

    assert result["summary"]["low_confidence_2d_update_count"] == 0
    assert all(not sample["low_confidence_2d_measurement_support"] for sample in result["samples"])


def test_v2_contact_and_step_speed_gates_refuse_complete_bridge() -> None:
    contact = build_recovery_policy_v2(
        _artifact(),
        contact_proposals={"events": [{"type": "contact", "frame": 4}]},
    )
    speed = build_recovery_policy_v2(_artifact(seed_speed_mps=36.0))

    assert contact["samples"] == []
    assert contact["refused_gaps"][0]["reason"] == "contact_proposal_inside_gap"
    assert speed["samples"] == []
    assert speed["refused_gaps"][0]["reason"] in {"step_speed_above_35_mps", "speed_above_ceiling"}

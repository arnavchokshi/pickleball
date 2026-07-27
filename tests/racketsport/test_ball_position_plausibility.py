"""Regression tests for the depth-aware physical plausibility gate.

The named cases below are taken from real solver output, not invented. Their
source is ``runs/lanes/ballarc_anchorfusion_20260726/...`` (see
``runs/lanes/retire_reprojection_gate_20260726/report.json`` for the full
scan), where the arc solver emitted 79 frames between 15.1 m and 23.5 m of
altitude and 3 frames underground, all banded ``arc_weak``, all carrying a
``sigma_m`` under 0.28 m, and none of them caught by any check that existed
before this gate.
"""

from __future__ import annotations

import pytest

from threed.racketsport.ball_flight_sanity import apply_flight_sanity_demotions, evaluate_ball_flight_sanity
from threed.racketsport.ball_position_plausibility import (
    BallPlausibilityBounds,
    evaluate_ball_track_plausibility,
    evaluate_position,
    is_absurd,
    is_plausible,
    position_violations,
    segment_physical_sanity_violations,
    speed_violations,
)


# (label, world_xyz) sampled from wolverine_no_soft_current's arc_weak frames.
_WOLVERINE_ABSURD_FRAMES = [
    ("frame_046", [-2.66, 7.29, 15.13]),
    ("frame_048", [-2.61, 7.25, 16.09]),
    ("frame_050", [-2.56, 7.21, 16.97]),
    ("frame_123", [-1.12, 6.01, 15.71]),
    ("apex_of_segment_2", [-2.66, 7.29, 23.52]),
]

# Same clip, the frames that are wrong but only by centimetres.
_WOLVERINE_IMPLAUSIBLE_FRAMES = [
    ("frame_011", [-2.57, 5.26, -0.13]),
    ("frame_157", [-0.71, 5.67, -0.32]),
    ("frame_216", [-0.53, 4.92, -0.10001]),
]


@pytest.mark.parametrize(("label", "world_xyz"), _WOLVERINE_ABSURD_FRAMES)
def test_twenty_metres_in_the_air_is_absurd(label: str, world_xyz: list[float]) -> None:
    verdict = evaluate_position(world_xyz)

    assert verdict["plausible"] is False, label
    assert verdict["absurd"] is True, label
    assert "above_plausible_apex" in verdict["violations"]
    assert "far_above_plausible_apex" in verdict["absurd_violations"]
    assert is_absurd(world_xyz) is True


@pytest.mark.parametrize(("label", "world_xyz"), _WOLVERINE_IMPLAUSIBLE_FRAMES)
def test_slightly_underground_is_implausible_but_not_absurd(label: str, world_xyz: list[float]) -> None:
    verdict = evaluate_position(world_xyz)

    assert verdict["plausible"] is False, label
    assert verdict["absurd"] is False, label
    assert verdict["violations"] == ["below_ground_plane"]


def test_normal_rally_positions_pass() -> None:
    for world_xyz in ([0.0, 0.0, 0.9], [1.1, 3.4, 0.03], [-2.9, -6.5, 2.4], [0.0, 0.0, 7.9]):
        assert is_plausible(world_xyz) is True, world_xyz
        assert position_violations(world_xyz) == ()


def test_absent_position_is_not_a_violation() -> None:
    assert is_plausible(None) is True
    assert evaluate_position(None)["evaluated"] is False
    assert is_plausible([0.0, 0.0, float("nan")]) is True


def test_far_off_court_is_absurd_and_nearby_off_court_is_not() -> None:
    # 28.98 m of |x| appears in demo_short_slice_current_no_soft30.
    assert evaluate_position([28.98, 0.0, 1.0])["absurd"] is True
    # 13.4 m of |y| appears in burlington_gold's arc_weak frames: outside the
    # court plus margin, but not off the planet.
    nearby = evaluate_position([0.0, 13.39, 1.0])
    assert nearby["violations"] == ["outside_court_footprint"]
    assert nearby["absurd"] is False


def test_speed_ceiling_flags_absurd_but_never_slow_arcs() -> None:
    assert speed_violations(8.0) == ()
    assert speed_violations(None) == ()
    # Deliberately not flagged: a slow arc is depth-ambiguous, not junk.
    assert speed_violations(0.2) == ()
    assert speed_violations(41.0) == ("speed_above_plausible_range",)
    assert speed_violations(120.0) == ("speed_absurd", "speed_above_plausible_range")


def test_segment_violations_read_the_solvers_own_physical_sanity() -> None:
    # Segment 2 of wolverine_no_soft_current: max_reprojection_error_px was
    # 3585 px, but the criterion that matters is the 23.52 m apex.
    sanity = {
        "apex_height_m": 23.521538,
        "initial_speed_mps": 12.0,
        "court_volume": {"violation": False, "max_overage_m": 0.0},
        "violations": ["apex_height_implausible"],
    }

    assert segment_physical_sanity_violations(sanity) == (
        "above_plausible_apex",
        "far_above_plausible_apex",
    )
    assert segment_physical_sanity_violations({}) == ()
    assert segment_physical_sanity_violations(None) == ()


def test_bounds_reject_incoherent_configuration() -> None:
    with pytest.raises(ValueError):
        BallPlausibilityBounds(z_min_m=-0.1, hard_z_min_m=0.0)
    with pytest.raises(ValueError):
        BallPlausibilityBounds(z_max_m=8.0, hard_z_max_m=4.0)
    with pytest.raises(ValueError):
        BallPlausibilityBounds(court_margin_m=10.0, hard_court_margin_m=1.0)


def test_track_sweep_reaches_frames_in_no_segment() -> None:
    frames = [
        {"world_xyz": [0.0, 0.0, 0.9]},
        {"world_xyz": None},
        {"world_xyz": [-2.66, 7.29, 23.52]},
        {"world_xyz": [-0.71, 5.67, -0.32]},
        {},
    ]

    report = evaluate_ball_track_plausibility(frames)

    assert report["summary"]["evaluated_frame_count"] == 3
    assert report["summary"]["implausible_frame_count"] == 2
    assert report["summary"]["absurd_frame_count"] == 1
    assert report["depth_unvalidated"] is True
    assert [item["frame"] for item in report["frames"]] == [2, 3]


def test_anisotropic_sigma_seam_is_off_by_default_and_never_changes_a_verdict() -> None:
    frames = [{"world_xyz": [-2.66, 7.29, 23.52]}]

    off = evaluate_ball_track_plausibility(frames)
    assert "depth_sigma_m" not in off["frames"][0]

    # Seam for the anisotropic-uncertainty work. Supplying a sigma so wide it
    # covers the overage annotates the frame but must not rescue it.
    on = evaluate_ball_track_plausibility(frames, depth_sigma_m_by_frame={0: 100.0})
    assert on["frames"][0]["depth_sigma_m"] == 100.0
    assert on["frames"][0]["overage_within_claimed_sigma"] is True
    assert on["frames"][0]["absurd"] is True
    assert on["summary"]["absurd_frame_count"] == 1


def test_flight_sanity_suppresses_a_ball_twenty_metres_in_the_air() -> None:
    """The whole point: a frame no pre-existing check could see.

    The frame below is kinematically unremarkable inside its segment, sits
    inside the court footprint, and carries a confident-looking sigma. Every
    check that existed before this gate passed it, because none of them had an
    upper bound on height.
    """

    frames = [
        {"t": index / 30.0, "visible": True, "world_xyz": [0.0, 0.0, 0.9], "band": "anchored_measured", "sigma_m": 0.19}
        for index in range(31)
    ]
    frames[15]["world_xyz"] = [-2.66, 7.29, 23.52]
    artifact = {
        "artifact_type": "racketsport_ball_track_arc_solved",
        "clip_id": "synthetic_twenty_metres_up",
        "fps": 30.0,
        "anchors": [],
        "frames": frames,
        "summary": {},
    }

    report = evaluate_ball_flight_sanity(artifact)

    # No segments at all: there are no bounce/contact anchors, so every
    # segment-scoped check is inert here. The sweep still reaches the frame.
    assert report["summary"]["segment_count"] == 0
    assert report["summary"]["absurd_frame_count"] == 1
    assert report["suppress_frames"]["15"] == ["above_plausible_apex", "far_above_plausible_apex"]

    gated = apply_flight_sanity_demotions(artifact, report)

    assert gated["frames"][15]["world_xyz"] is None
    assert gated["frames"][15]["sigma_m"] is None
    assert gated["frames"][15]["band"] == "hidden"
    assert gated["frames"][15]["depth_unvalidated"] is True
    assert gated["frames"][15]["flight_sanity_original"]["world_xyz"] == [-2.66, 7.29, 23.52]
    assert gated["frames"][14]["world_xyz"] == [0.0, 0.0, 0.9]
    assert gated["summary"]["absurd_frame_suppressed_count"] == 1


def test_absurd_frame_is_suppressed_even_on_a_bvp_fallback_segment() -> None:
    """The BVP-fallback exemption must not rescue an impossible position.

    ``outside_court_volume`` deliberately spares BVP-fallback segments so a
    depth-ambiguous arc is not thrown away. There is nothing ambiguous about a
    ball 23 m in the air, and every one of the real 79 frames sat on a
    ``fit_bvp_fallback`` segment, so that exemption is exactly what let them
    through.
    """

    frames = [
        {"t": index / 30.0, "visible": True, "world_xyz": [0.0, 0.0, 0.9], "band": "arc_weak", "sigma_m": 0.19}
        for index in range(31)
    ]
    frames[15]["world_xyz"] = [-2.66, 7.29, 23.52]
    for frame in frames:
        frame["arc_solver"] = {"segment_id": 0, "segment_status": "fit_bvp_fallback"}
    artifact = {
        "artifact_type": "racketsport_ball_track_arc_solved",
        "clip_id": "synthetic_bvp_fallback",
        "fps": 30.0,
        "anchors": [],
        "segments": [{"segment_id": 0, "status": "fit_bvp_fallback", "frame_start": 0, "frame_end": 30}],
        "frames": frames,
        "summary": {},
    }

    gated = apply_flight_sanity_demotions(artifact, evaluate_ball_flight_sanity(artifact))

    assert gated["frames"][15]["world_xyz"] is None
    assert gated["frames"][15]["band"] == "hidden"
    # The merely-weak neighbours keep their positions.
    assert gated["frames"][14]["world_xyz"] == [0.0, 0.0, 0.9]

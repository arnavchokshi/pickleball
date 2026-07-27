"""Tests for camera-ray / court-volume containment.

The regression cases are real rays from ``burlington_gold_0300_low_steep_corner``
(reviewed ``metric_15pt`` calibration, 6.39 px median reprojection). Frames 30
and 369 are the two frames where the owner's own click and the detector's
emitted pixel disagree by more than 100 px, and where visual review confirmed
the detector is locked on a game being played on the *adjacent* court.

They are here to pin the refutation, not a success: the whole point is that the
adjacent-court ray is still *contained*, so this gate cannot catch it.
"""

from __future__ import annotations

import math

import pytest

from threed.racketsport.ball_ray_court_volume import (
    COURT_HALF_LENGTH_M,
    COURT_HALF_WIDTH_M,
    DEFAULT_BOUNDS,
    DISJOINT_MARKS,
    GRAZING_MARKS,
    POLICY,
    REFUTED_NOTE,
    SCHEMA_VERSION,
    CourtVolumeBounds,
    evaluate_ball_track_court_volume,
    evaluate_ray,
    ray_box_chord,
)


# Camera centre of the burlington clip, in court_netcenter_z_up_m.
BURLINGTON_ORIGIN = (4.548781, -8.497264, 1.325928)
# Unit ray directions measured from the clip's reviewed calibration.
BURLINGTON_F30_DETECTOR = (-0.984288, 0.169538, -0.04934)
BURLINGTON_F30_OWNER = (-0.211767, 0.975849, -0.053603)
BURLINGTON_F369_DETECTOR = (-0.931779, 0.358474, 0.057312)
BURLINGTON_F369_OWNER = (-0.148171, 0.981772, -0.119032)


def test_straight_down_the_court_is_contained() -> None:
    report = evaluate_ray((0.0, -9.0, 1.5), (0.0, 1.0, 0.0))
    assert report["evaluated"] is True
    assert report["contained"] is True
    assert report["verdict"] == "contained"
    assert report["marks"] == []
    # Enters at the -margin baseline and exits at the +margin baseline.
    assert report["chord_length_m"] == pytest.approx(
        2.0 * (COURT_HALF_LENGTH_M + DEFAULT_BOUNDS.margin_m), abs=1e-6
    )


def test_ray_pointing_away_from_the_court_is_disjoint() -> None:
    report = evaluate_ray((0.0, -9.0, 1.5), (0.0, -1.0, 0.0))
    assert report["verdict"] == "disjoint"
    assert report["contained"] is False
    assert report["marks"] == list(DISJOINT_MARKS)
    assert report["chord_length_m"] == 0.0


def test_ray_above_the_apex_never_enters() -> None:
    """A ray held above the ceiling for the whole court length misses."""

    bounds = CourtVolumeBounds(apex_m=4.0)
    report = evaluate_ray((0.0, -9.0, 9.0), (0.0, 1.0, 0.0), bounds)
    assert report["verdict"] == "disjoint"


def test_airborne_ball_is_not_penalised_for_being_airborne() -> None:
    """The confound this gate is designed to avoid.

    A ball 3 m above the far service line projects, on the ground plane, to a
    point far behind the baseline. Height-agnostic containment must still
    accept it.
    """

    origin = (0.0, -9.0, 1.5)
    target = (0.0, 3.0, 3.0)
    direction = tuple(t - o for t, o in zip(target, origin))
    report = evaluate_ray(origin, direction)
    assert report["verdict"] == "contained"
    assert report["marks"] == []


def test_corner_clip_is_grazing_not_disjoint() -> None:
    bounds = CourtVolumeBounds(margin_m=0.0, min_chord_m=1.0)
    # Cut the far +x corner diagonally: in through the sideline, straight back
    # out through the baseline a few centimetres later.
    offset = 5.0 / math.sqrt(2.0)
    origin = (COURT_HALF_WIDTH_M + offset, COURT_HALF_LENGTH_M - 0.2056 - offset, 1.0)
    direction = (-1.0, 1.0, 0.0)
    report = evaluate_ray(origin, direction, bounds)
    assert report["verdict"] == "grazing"
    assert report["marks"] == list(GRAZING_MARKS)
    assert 0.0 < report["chord_length_m"] < bounds.min_chord_m


@pytest.mark.parametrize(
    "direction",
    [
        BURLINGTON_F30_DETECTOR,
        BURLINGTON_F30_OWNER,
        BURLINGTON_F369_DETECTOR,
        BURLINGTON_F369_OWNER,
    ],
)
def test_burlington_rays_are_all_contained(direction: tuple[float, float, float]) -> None:
    """Regression: the gate cannot separate the adjacent-court ball.

    On frame 30 the detector is on a ball roughly 19 m outside our left
    sideline, yet its ray is contained exactly like the owner's click on the
    real ball. This is the measured refutation and it must not silently
    regress into a false confidence that the gate works.
    """

    report = evaluate_ray(BURLINGTON_ORIGIN, direction)
    assert report["verdict"] == "contained"
    assert report["marks"] == []


def test_burlington_f30_wrong_ball_ray_crosses_our_airspace() -> None:
    """Pin the mechanism, not just the verdict.

    The ball is roughly 19 m outside our left sideline, but on its way there
    the ray crosses our near-baseline corner about a metre off the ground --
    an entirely ordinary place for a real ball to be.
    """

    bounds = CourtVolumeBounds(margin_m=0.5)
    span = ray_box_chord(BURLINGTON_ORIGIN, BURLINGTON_F30_DETECTOR, bounds)
    assert span is not None
    enter, exit_ = span
    point = [
        BURLINGTON_ORIGIN[axis] + enter * BURLINGTON_F30_DETECTOR[axis]
        for axis in range(3)
    ]
    assert point[2] == pytest.approx(0.95, abs=0.10)
    assert abs(point[0]) <= COURT_HALF_WIDTH_M + bounds.margin_m + 1e-6
    assert exit_ > enter


def test_burlington_camera_sits_inside_the_volume_at_the_default_margin() -> None:
    """Why the default margin flags nothing on this clip: the test is vacuous.

    A 2 m legal-play margin puts the burlington camera centre itself inside
    the box, so every forward ray is contained by construction and the gate
    cannot reject anything at all.
    """

    span = ray_box_chord(BURLINGTON_ORIGIN, BURLINGTON_F30_DETECTOR)
    assert span is not None
    assert span[0] == pytest.approx(0.0, abs=1e-9)


def test_chord_length_separates_burlington_f30_but_not_f369() -> None:
    """Chord length is a better signal than containment -- and still not enough.

    On frame 30 the wrong-ball ray only clips a 0.61 m corner while the
    owner's click cuts 11.4 m through the court. On frame 369 the same
    comparison collapses: the wrong ball's chord is *longer* than the real
    ball's. One clean separation and one inversion, from two frames of one
    clip, is not a discriminator.
    """

    bounds = CourtVolumeBounds(margin_m=0.5)
    f30_wrong = evaluate_ray(BURLINGTON_ORIGIN, BURLINGTON_F30_DETECTOR, bounds)
    f30_real = evaluate_ray(BURLINGTON_ORIGIN, BURLINGTON_F30_OWNER, bounds)
    assert f30_wrong["chord_length_m"] == pytest.approx(0.607, abs=0.02)
    assert f30_real["chord_length_m"] == pytest.approx(11.366, abs=0.02)
    assert f30_wrong["chord_length_m"] < f30_real["chord_length_m"]

    f369_wrong = evaluate_ray(BURLINGTON_ORIGIN, BURLINGTON_F369_DETECTOR, bounds)
    f369_real = evaluate_ray(BURLINGTON_ORIGIN, BURLINGTON_F369_OWNER, bounds)
    assert f369_wrong["chord_length_m"] > f369_real["chord_length_m"]


def test_non_finite_inputs_are_not_evaluated_and_carry_no_marks() -> None:
    for bad in [(float("nan"), 0.0, 0.0), (0.0, float("inf"), 0.0)]:
        report = evaluate_ray(bad, (0.0, 1.0, 0.0))
        assert report["evaluated"] is False
        assert report["verdict"] == "not_evaluated"
        assert report["marks"] == []
    zero = evaluate_ray((0.0, -9.0, 1.5), (0.0, 0.0, 0.0))
    assert zero["evaluated"] is True
    assert zero["verdict"] == "disjoint"


def test_direction_need_not_be_normalised() -> None:
    unit = evaluate_ray((0.0, -9.0, 1.5), (0.0, 1.0, 0.0))
    scaled = evaluate_ray((0.0, -9.0, 1.5), (0.0, 17.0, 0.0))
    assert unit["chord_length_m"] == pytest.approx(scaled["chord_length_m"], abs=1e-9)


def test_track_report_skips_frames_the_detector_never_emitted() -> None:
    frames = [
        {"visible": False, "xy": [0.0, 0.0]},
        {"visible": True, "xy": [10.0, 10.0]},
        {"visible": True, "xy": [20.0, 20.0]},
    ]
    rays = {
        1: ((0.0, -9.0, 1.5), (0.0, 1.0, 0.0)),
        2: ((0.0, -9.0, 1.5), (0.0, -1.0, 0.0)),
    }
    report = evaluate_ball_track_court_volume(frames, rays)
    assert report["schema_version"] == SCHEMA_VERSION
    assert report["policy"] == POLICY
    assert report["not_ground_truth"] is True
    assert report["emitted_frame_count"] == 2
    assert report["evaluated_frame_count"] == 2
    assert report["counts"]["contained"] == 1
    assert report["counts"]["disjoint"] == 1
    assert report["disjoint_rate"] == pytest.approx(0.5)
    assert [entry["frame"] for entry in report["frames"]] == [1, 2]
    assert report["refuted_note"] == REFUTED_NOTE


def test_track_report_never_mutates_or_drops_frames() -> None:
    frames = [{"visible": True, "xy": [1.0, 2.0]}]
    before = [dict(frame) for frame in frames]
    rays = {0: ((0.0, -9.0, 1.5), (0.0, -1.0, 0.0))}
    report = evaluate_ball_track_court_volume(frames, rays)
    assert frames == before
    assert report["counts"]["disjoint"] == 1
    # The detection survives; only a mark is produced.
    assert report["frames"][0]["marks"] == list(DISJOINT_MARKS)


def test_missing_ray_is_reported_not_guessed() -> None:
    frames = [{"visible": True, "xy": [1.0, 2.0]}]
    report = evaluate_ball_track_court_volume(frames, {})
    assert report["counts"]["not_evaluated"] == 1
    assert report["evaluated_frame_count"] == 0
    assert report["disjoint_rate"] == 0.0
    assert report["frames"][0]["marks"] == []


def test_bounds_serialise_with_the_refutation_attached() -> None:
    payload = CourtVolumeBounds().to_dict()
    assert payload["margin_m"] == pytest.approx(2.0)
    assert payload["apex_m"] == pytest.approx(8.0)
    assert payload["refuted_note"] == REFUTED_NOTE
    assert math.isfinite(payload["min_chord_m"])

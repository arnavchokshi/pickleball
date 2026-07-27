"""Tests for the calibrated-image-envelope guard.

The named cases are the real ones. ``_OUTDOOR_CALIBRATION`` is the reviewed
15-point camera shipped in
``runs/full_mesh_examples_20260725/outdoor_mesh_final/outdoor_webcam_20s_fullmesh_final/court_calibration.json``
(sha256 bb7bb05b...), reproduced here so the test does not depend on an
untracked run directory. ``_OUTDOOR_BOUNCES`` are the six owner-labelled
bounces from
``runs/lanes/ball_label_tool_20260726/labels/outdoor_webcam_20s/ball_human_labels.json``
with the court x each pixel back-projects to, measured in
``runs/lanes/farfield_extrapolation_20260727/``.

Those six are the regression cases: two central, one just inside the calibrated
radius, and three past it, one of them at 79.6% of the half-diagonal on a model
whose evidence stops at 50.0%.
"""

from __future__ import annotations

import math

import pytest

from threed.racketsport.calibration_extrapolation import (
    DEFAULT_RESIDUAL_PX,
    FAR_EXTRAPOLATION_RATIO,
    MAX_EXTRAPOLATION_RATIO,
    VERDICT_EXTRAPOLATED,
    VERDICT_FAR_EXTRAPOLATED,
    VERDICT_WITHIN,
    CalibratedImageEnvelope,
    calibrated_image_envelope,
    calibration_residual_px,
    envelope_block_for_calibration,
    evaluate_ball_track_extrapolation,
    evaluate_pixel,
    is_extrapolated,
    pixel_verdict,
    radial_extrapolation_pixel_allowance,
    with_calibration_envelope,
)


_OUTDOOR_IMAGE_PTS = [
    [438.6013986013986, 708.2867132867133],
    [953.2867132867133, 710.5244755244755],
    [1485.8741258741259, 703.8111888111888],
    [1235.2447552447552, 318.9160839160839],
    [953.2867132867133, 318.9160839160839],
    [673.5664335664336, 323.3916083916084],
    [552.6573426573427, 520.3496503496504],
    [962.2377622377623, 518.1118881118881],
    [1360.5594405594406, 518.1118881118881],
    [602.0, 442.0],
    [955.5, 442.0],
    [1313.6083916083916, 448.7132867132867],
    [615.4125874125874, 404.041958041958],
    [957.7622377622377, 404.041958041958],
    [1291.2307692307693, 404.041958041958],
]

_OUTDOOR_WORLD_PTS = [
    [-3.048, -6.7056000000000004, 0.0],
    [0.0, -6.7056000000000004, 0.0],
    [3.048, -6.7056000000000004, 0.0],
    [3.048, 6.7056000000000004, 0.0],
    [0.0, 6.7056000000000004, 0.0],
    [-3.048, 6.7056000000000004, 0.0],
    [-3.048, -2.1336, 0.0],
    [0.0, -2.1336, 0.0],
    [3.048, -2.1336, 0.0],
    [-3.048, 0.0, 0.0],
    [0.0, 0.0, 0.0],
    [3.048, 0.0, 0.0],
    [-3.048, 2.1336, 0.0],
    [0.0, 2.1336, 0.0],
    [3.048, 2.1336, 0.0],
]

_OUTDOOR_CALIBRATION = {
    "sport": "pickleball",
    "image_size": [1920, 1080],
    "intrinsics": {
        "fx": 2537.9139649536,
        "fy": 2537.9139649536,
        "cx": 960.0,
        "cy": 540.0,
        "dist": [0.0, 0.0, 0.0, 0.0],
        "source": "metric_15pt_reviewed",
    },
    "reprojection_error_px": {"median": 4.7834997281882865, "p95": 12.276624313829716},
    "extrinsics": {
        "R": [
            [0.9999897798981388, -0.004512286365101776, -0.0002817286488120169],
            [-0.0014155890752480592, -0.25331620184779297, -0.9673824982854395],
            [0.0042937404255827104, 0.9673730103497655, -0.2533200004736898],
        ],
        "t": [-0.029131474628101135, -0.7281538153162044, 21.25745651752845],
        "camera_height_m": 4.680527430922098,
    },
    "image_pts": _OUTDOOR_IMAGE_PTS,
    "world_pts": _OUTDOOR_WORLD_PTS,
}

# (frame, pixel_xy, radius % of half-diagonal, solved court x, verdict)
_OUTDOOR_BOUNCES = [
    (124, [1081.6, 653.2], 15.1, 0.753, VERDICT_WITHIN),
    (168, [1077.1, 381.5], 17.9, 1.153, VERDICT_WITHIN),
    (368, [490.6, 585.7], 42.8, -3.163, VERDICT_WITHIN),
    (388, [353.4, 662.2], 56.2, -3.699, VERDICT_EXTRAPOLATED),
    (402, [233.9, 732.0], 68.2, -4.073, VERDICT_FAR_EXTRAPOLATED),
    (414, [118.9, 786.3], 79.6, -4.441, VERDICT_FAR_EXTRAPOLATED),
]


# ---------------------------------------------------------------------------
# envelope computation
# ---------------------------------------------------------------------------


def test_outdoor_envelope_reproduces_the_measured_radial_span() -> None:
    envelope = calibrated_image_envelope(_OUTDOOR_CALIBRATION)

    assert envelope is not None
    assert envelope.correspondence_count == 15
    assert envelope.radius_pct(envelope.radius_px_min) == pytest.approx(2.0, abs=0.05)
    assert envelope.radius_pct(envelope.radius_px_median) == pytest.approx(32.6, abs=0.05)
    assert envelope.radius_pct(envelope.calibrated_radius_px) == pytest.approx(50.0, abs=0.05)
    # The whole point: the court supplies no correspondence past 50%.
    assert envelope.radius_pct(envelope.calibrated_radius_px) < 60.0


def test_envelope_uses_the_max_correspondence_radius_not_a_percentile() -> None:
    envelope = calibrated_image_envelope(_OUTDOOR_CALIBRATION)

    assert envelope is not None
    # The generous reading: the model is credited with everything it saw, so
    # the gate stays silent right up to the outermost correspondence.
    assert envelope.calibrated_radius_px == pytest.approx(envelope.radius_px_max)
    assert envelope.radius_px_p95 < envelope.radius_px_max


def test_far_radius_is_where_the_unvalidated_term_equals_the_measured_residual() -> None:
    envelope = calibrated_image_envelope(_OUTDOOR_CALIBRATION)

    assert envelope is not None
    ratio = envelope.far_radius_px / envelope.calibrated_radius_px
    assert ratio == pytest.approx(FAR_EXTRAPOLATION_RATIO)
    assert ratio**3 - 1.0 == pytest.approx(1.0)


def test_envelope_falls_back_to_the_principal_point_when_image_size_is_absent() -> None:
    calibration = {
        key: value for key, value in _OUTDOOR_CALIBRATION.items() if key != "image_size"
    }
    envelope = calibrated_image_envelope(calibration)

    assert envelope is not None
    assert envelope.image_size_px == (1920.0, 1080.0)


def test_absent_correspondences_yield_no_envelope_rather_than_a_fake_one() -> None:
    calibration = {
        key: value for key, value in _OUTDOOR_CALIBRATION.items() if key != "image_pts"
    }

    assert calibrated_image_envelope(calibration) is None
    block = envelope_block_for_calibration(calibration)
    assert block["available"] is False
    assert "reason" in block


def test_envelope_rejects_degenerate_construction() -> None:
    with pytest.raises(ValueError):
        CalibratedImageEnvelope(
            image_size_px=(0.0, 1080.0),
            correspondence_count=4,
            radius_px_min=1.0,
            radius_px_median=2.0,
            radius_px_p95=3.0,
            radius_px_max=4.0,
            bbox_px=(0.0, 1.0, 0.0, 1.0),
        )
    with pytest.raises(ValueError):
        CalibratedImageEnvelope(
            image_size_px=(1920.0, 1080.0),
            correspondence_count=0,
            radius_px_min=1.0,
            radius_px_median=2.0,
            radius_px_p95=3.0,
            radius_px_max=4.0,
            bbox_px=(0.0, 1.0, 0.0, 1.0),
        )


# ---------------------------------------------------------------------------
# the six owner-labelled bounces, as regression cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("frame", "pixel_xy", "radius_pct", "court_x", "verdict"), _OUTDOOR_BOUNCES
)
def test_owner_bounce_labels_are_classified_by_measured_radius(
    frame: int,
    pixel_xy: list[float],
    radius_pct: float,
    court_x: float,
    verdict: str,
) -> None:
    envelope = calibrated_image_envelope(_OUTDOOR_CALIBRATION)
    result = evaluate_pixel(pixel_xy, envelope)

    assert result["evaluated"] is True, frame
    assert result["radius_pct_of_half_diagonal"] == pytest.approx(radius_pct, abs=0.1), frame
    assert result["verdict"] == verdict, frame
    assert pixel_verdict(pixel_xy, envelope) == verdict, frame
    assert is_extrapolated(pixel_xy, envelope) is (verdict != VERDICT_WITHIN), frame


def test_frame_368_is_inside_the_envelope_even_though_it_is_out_of_bounds() -> None:
    """The owner's first out-of-bounds bounce is NOT an extrapolation case.

    It sits at 42.8% of the half-diagonal, inside the 50.0% the correspondences
    reach. Out-of-court is a court fact; out-of-envelope is a camera fact, and
    conflating them would let this gate take credit it has not earned.
    """

    envelope = calibrated_image_envelope(_OUTDOOR_CALIBRATION)
    result = evaluate_pixel([490.6, 585.7], envelope)

    assert result["verdict"] == VERDICT_WITHIN
    assert result["violations"] == []
    assert result["overage_px"] == 0.0


def test_the_three_far_bounces_are_flagged_and_ordered_by_radius() -> None:
    envelope = calibrated_image_envelope(_OUTDOOR_CALIBRATION)
    ratios = [
        evaluate_pixel(pixel, envelope)["extrapolation_ratio"]
        for _, pixel, _, _, _ in _OUTDOOR_BOUNCES
    ]

    assert ratios == sorted(ratios)
    assert ratios[3] > 1.0 and ratios[4] > 1.0 and ratios[5] > 1.0
    assert ratios[2] < 1.0


def test_nothing_is_suppressed_by_this_gate() -> None:
    """The hard tier must stay a label, never a deletion."""

    envelope = calibrated_image_envelope(_OUTDOOR_CALIBRATION)
    for frame, pixel, _, _, _ in _OUTDOOR_BOUNCES:
        result = evaluate_pixel(pixel, envelope)
        assert "suppress" not in result, frame
        assert result["pixel_xy"] == [
            pytest.approx(pixel[0]),
            pytest.approx(pixel[1]),
        ], frame


# ---------------------------------------------------------------------------
# the pixel allowance
# ---------------------------------------------------------------------------


def test_allowance_is_exactly_zero_inside_the_calibrated_radius() -> None:
    for frame, pixel, _, _, verdict in _OUTDOOR_BOUNCES:
        allowance = radial_extrapolation_pixel_allowance(_OUTDOOR_CALIBRATION, pixel)
        if verdict == VERDICT_WITHIN:
            assert allowance["allowance_px"] == 0.0, frame
        else:
            assert allowance["allowance_px"] > 0.0, frame


def test_allowance_grows_as_the_cube_of_the_radius_ratio() -> None:
    residual = 4.7834997281882865
    for _, pixel, _, _, _ in _OUTDOOR_BOUNCES:
        allowance = radial_extrapolation_pixel_allowance(_OUTDOOR_CALIBRATION, pixel)
        ratio = allowance["extrapolation_ratio"]
        expected = residual * max(0.0, ratio**3 - 1.0)
        assert allowance["allowance_px"] == pytest.approx(expected, abs=1e-5)
        assert allowance["residual_px"] == pytest.approx(residual)
        assert allowance["residual_provenance"] == "calibration_reprojection_median_in_sample"


def test_measured_allowances_on_the_three_far_bounces() -> None:
    expected = {388: 2.0, 402: 7.3, 414: 14.5}
    for frame, pixel, _, _, _ in _OUTDOOR_BOUNCES:
        if frame not in expected:
            continue
        allowance = radial_extrapolation_pixel_allowance(_OUTDOOR_CALIBRATION, pixel)
        assert allowance["allowance_px"] == pytest.approx(expected[frame], abs=0.1), frame


def test_allowance_ratio_is_capped_so_sigma_cannot_run_away() -> None:
    envelope = calibrated_image_envelope(_OUTDOOR_CALIBRATION)
    assert envelope is not None
    # A pixel far outside the frame entirely.
    allowance = radial_extrapolation_pixel_allowance(_OUTDOOR_CALIBRATION, [-40000.0, 540.0])

    assert allowance["ratio_capped"] is True
    assert allowance["ratio_used"] == pytest.approx(MAX_EXTRAPOLATION_RATIO)
    residual, _ = calibration_residual_px(_OUTDOOR_CALIBRATION)
    assert allowance["allowance_px"] == pytest.approx(
        residual * (MAX_EXTRAPOLATION_RATIO**3 - 1.0)
    )


def test_allowance_uses_a_supplied_residual_when_given() -> None:
    allowance = radial_extrapolation_pixel_allowance(
        _OUTDOOR_CALIBRATION, [118.9, 786.3], residual_px=1.0
    )

    assert allowance["residual_provenance"] == "supplied"
    assert allowance["residual_px"] == pytest.approx(1.0)
    assert allowance["allowance_px"] == pytest.approx(
        allowance["extrapolation_ratio"] ** 3 - 1.0, abs=1e-6
    )


def test_residual_falls_back_to_a_stated_default_without_a_reprojection_error() -> None:
    calibration = {
        key: value
        for key, value in _OUTDOOR_CALIBRATION.items()
        if key != "reprojection_error_px"
    }
    residual, provenance = calibration_residual_px(calibration)

    assert residual == DEFAULT_RESIDUAL_PX
    assert provenance == "default_no_reprojection_error_recorded"


def test_allowance_is_unavailable_rather_than_zero_without_an_envelope() -> None:
    calibration = {
        key: value for key, value in _OUTDOOR_CALIBRATION.items() if key != "image_pts"
    }
    allowance = radial_extrapolation_pixel_allowance(calibration, [118.9, 786.3])

    assert allowance["available"] is False
    assert allowance["allowance_px"] == 0.0
    assert "reason" in allowance


def test_absent_pixel_is_not_a_violation() -> None:
    envelope = calibrated_image_envelope(_OUTDOOR_CALIBRATION)

    for absent in (None, [], [float("nan"), 1.0], "1,2"):
        result = evaluate_pixel(absent, envelope)
        assert result["evaluated"] is False
        assert result["extrapolated"] is False
        assert result["violations"] == []


# ---------------------------------------------------------------------------
# artifact block and track sweep
# ---------------------------------------------------------------------------


def test_envelope_block_is_additive_and_idempotent() -> None:
    once = with_calibration_envelope(_OUTDOOR_CALIBRATION)
    twice = with_calibration_envelope(once)

    assert once["calibrated_image_envelope"]["available"] is True
    assert twice["calibrated_image_envelope"] == once["calibrated_image_envelope"]
    for key, value in _OUTDOOR_CALIBRATION.items():
        assert once[key] == value
    assert "calibrated_image_envelope" not in _OUTDOOR_CALIBRATION


def test_envelope_block_carries_provenance_and_the_residual_it_will_use() -> None:
    block = envelope_block_for_calibration(_OUTDOOR_CALIBRATION)

    assert block["provenance"] == "calibration_image_pts"
    assert block["correspondence_count"] == 15
    assert block["residual_provenance"] == "calibration_reprojection_median_in_sample"
    assert block["residual_px"] == pytest.approx(4.7835, abs=1e-3)
    assert block["policy"] == "calibration_extrapolation_v1"


def test_track_sweep_counts_only_frames_that_emitted_a_position() -> None:
    frames = [
        {"xy": [118.9, 786.3], "world_xyz": None},  # detected, nothing emitted
        {"xy": [118.9, 786.3], "world_xyz": [-4.441, -8.116, 0.037]},
        {"xy": [233.9, 732.0], "world_xyz": [-4.073, -7.242, 0.037]},
        {"xy": [353.4, 662.2], "world_xyz": [-3.699, -5.939, 0.037]},
        {"xy": [962.2, 518.1], "world_xyz": [0.0, -2.134, 0.037]},
        {"world_xyz": [0.0, 0.0, 0.037]},  # position with no pixel
    ]
    report = evaluate_ball_track_extrapolation(frames, _OUTDOOR_CALIBRATION)

    assert report["summary"]["emitted_position_count"] == 5
    assert report["summary"]["evaluated_frame_count"] == 4
    assert report["summary"]["extrapolated_frame_count"] == 3
    assert report["summary"]["far_extrapolated_frame_count"] == 2
    assert report["summary"]["violation_counts"] == {
        "far_outside_calibrated_radius": 2,
        "outside_calibrated_radius": 3,
    }
    assert [entry["frame"] for entry in report["frames"]] == [1, 2, 3]


def test_track_sweep_without_an_envelope_reports_nothing_rather_than_clean() -> None:
    calibration = {
        key: value for key, value in _OUTDOOR_CALIBRATION.items() if key != "image_pts"
    }
    report = evaluate_ball_track_extrapolation(
        [{"xy": [118.9, 786.3], "world_xyz": [-4.4, -8.1, 0.037]}], calibration
    )

    assert report["envelope"] is None
    assert report["summary"]["emitted_position_count"] == 1
    assert report["summary"]["evaluated_frame_count"] == 0
    assert report["summary"]["extrapolated_frame_count"] == 0


def test_a_symmetric_synthetic_calibration_flags_by_radius_alone() -> None:
    """No court, no pose: the verdict is a function of radius and nothing else."""

    calibration = {
        "image_size": [1000, 1000],
        "image_pts": [[500.0, 400.0], [600.0, 500.0], [500.0, 600.0], [400.0, 500.0]],
        "world_pts": [[0, 1, 0], [1, 0, 0], [0, -1, 0], [-1, 0, 0]],
        "reprojection_error_px": {"median": 1.0, "p95": 2.0},
    }
    envelope = calibrated_image_envelope(calibration)
    assert envelope is not None
    assert envelope.calibrated_radius_px == pytest.approx(100.0)

    for angle in (0.0, math.pi / 3.0, math.pi, 4.0 * math.pi / 3.0):
        inside = [500.0 + 99.0 * math.cos(angle), 500.0 + 99.0 * math.sin(angle)]
        soft = [500.0 + 110.0 * math.cos(angle), 500.0 + 110.0 * math.sin(angle)]
        hard = [500.0 + 200.0 * math.cos(angle), 500.0 + 200.0 * math.sin(angle)]
        assert pixel_verdict(inside, envelope) == VERDICT_WITHIN
        assert pixel_verdict(soft, envelope) == VERDICT_EXTRAPOLATED
        assert pixel_verdict(hard, envelope) == VERDICT_FAR_EXTRAPOLATED


# ---------------------------------------------------------------------------
# the bounce-anchor uncertainty seam
# ---------------------------------------------------------------------------


def _uncertainty(pixel_xy: list[float]):
    from threed.racketsport.ball_arc_solver import anchor_uncertainty_for_bounce

    return anchor_uncertainty_for_bounce(
        _OUTDOOR_CALIBRATION,
        pixel_xy,
        base_sigma_m=0.05,
        pixel_sigma_px=2.0,
        fps=30.0,
        calibration_residual_m=0.100551,
    )


def test_bounce_sigma_is_untouched_inside_the_calibrated_radius() -> None:
    for frame, pixel, _, _, verdict in _OUTDOOR_BOUNCES:
        if verdict != VERDICT_WITHIN:
            continue
        uncertainty = _uncertainty(pixel)
        assert uncertainty is not None, frame
        assert uncertainty.terms["extrapolation_verdict"] == VERDICT_WITHIN, frame
        assert uncertainty.terms["extrapolation_allowance_px"] == 0.0, frame
        assert uncertainty.terms["extrapolation_along_ray_m"] == 0.0, frame
        assert uncertainty.terms["extrapolation_perp_m"] == 0.0, frame
        assert "EXTRAPOLATED" not in uncertainty.basis, frame


def test_bounce_sigma_grows_materially_past_the_calibrated_radius() -> None:
    """A position at 79.6% radius on a model fit to 50.0% earns a bigger sigma."""

    inside = _uncertainty([490.6, 585.7])  # frame 368, 42.8%
    far = _uncertainty([118.9, 786.3])  # frame 414, 79.6%

    assert inside is not None and far is not None
    assert far.terms["extrapolation_verdict"] == VERDICT_FAR_EXTRAPOLATED
    assert far.terms["extrapolation_along_ray_m"] == pytest.approx(0.219, abs=0.01)
    assert far.sigma_along_ray_m > inside.sigma_along_ray_m
    # Materially larger, not a rounding nudge: the 79.6% anchor's along-ray
    # sigma is at least half again the 42.8% one's.
    assert far.sigma_along_ray_m / inside.sigma_along_ray_m > 1.5
    assert "EXTRAPOLATED" in far.basis
    assert "79.6% of the half-diagonal" in far.basis
    assert "50.0%" in far.basis


def test_bounce_sigma_is_monotone_in_radius_across_the_four_far_bounces() -> None:
    sigmas = []
    for _, pixel, _, _, _ in _OUTDOOR_BOUNCES[2:]:
        uncertainty = _uncertainty(pixel)
        assert uncertainty is not None
        sigmas.append(uncertainty.terms["extrapolation_along_ray_m"])

    assert sigmas == sorted(sigmas)
    assert sigmas[0] == 0.0


def test_bounce_sigma_reports_an_unknown_verdict_without_correspondences() -> None:
    from threed.racketsport.ball_arc_solver import anchor_uncertainty_for_bounce

    calibration = {
        key: value for key, value in _OUTDOOR_CALIBRATION.items() if key != "image_pts"
    }
    uncertainty = anchor_uncertainty_for_bounce(
        calibration, [118.9, 786.3], base_sigma_m=0.05, calibration_residual_m=0.1
    )

    assert uncertainty is not None
    assert uncertainty.terms["extrapolation_verdict"] == "unknown"
    assert "extrapolation_along_ray_m" not in uncertainty.terms


# ---------------------------------------------------------------------------
# the human-label artifact seam
# ---------------------------------------------------------------------------


def _label(**overrides):
    from threed.racketsport.ball_label_schema import BALL_RADIUS_M, BallLabel

    base = dict(
        frame=414,
        timestamp_s=13.8,
        pixel_xy=(118.9, 786.3),
        world_xyz_m=(-4.441, -8.116, BALL_RADIUS_M),
        kind="bounce",
        depth_along_ray_m=14.15,
        ray_origin_m=(0.0, 0.0, 0.0),
        ray_direction_unit=(0.0, 0.0, 1.0),
        depth_source="ray_plane_intersection",
        sigma_xyz_m=(0.1, 0.2, 0.1),
        sigma_along_ray_m=0.26,
        sigma_perp_m=0.17,
        uncertainty_basis="test",
        human_confidence="medium",
        origin="fresh",
    )
    base.update(overrides)
    # keep world_xyz_m on the ray so the contract's own consistency guard holds
    base["world_xyz_m"] = tuple(
        base["ray_origin_m"][i] + base["depth_along_ray_m"] * base["ray_direction_unit"][i]
        for i in range(3)
    )
    base["world_xyz_m"] = (base["world_xyz_m"][0], base["world_xyz_m"][1], BALL_RADIUS_M)
    base["ray_origin_m"] = (0.0, 0.0, BALL_RADIUS_M)
    base["ray_direction_unit"] = (1.0, 0.0, 0.0)
    base["world_xyz_m"] = (base["depth_along_ray_m"], 0.0, BALL_RADIUS_M)
    return BallLabel(**base)


def test_label_schema_verdicts_match_the_gate_module() -> None:
    from threed.racketsport.ball_label_schema import _EXTRAPOLATION_VERDICTS

    assert _EXTRAPOLATION_VERDICTS == {
        VERDICT_WITHIN,
        VERDICT_EXTRAPOLATED,
        VERDICT_FAR_EXTRAPOLATED,
    }


def test_label_without_an_extrapolation_block_serializes_unchanged() -> None:
    label = _label()
    label.validate()
    payload = label.to_json_dict()

    assert "extrapolation" not in payload
    assert label.is_extrapolated is False


def test_label_records_and_round_trips_a_far_extrapolated_verdict() -> None:
    from threed.racketsport.ball_label_schema import BallLabel

    label = _label(
        extrapolation={
            "available": True,
            "verdict": VERDICT_FAR_EXTRAPOLATED,
            "extrapolated": True,
            "far_extrapolated": True,
            "radius_pct_of_half_diagonal": 79.6,
            "calibrated_radius_pct_of_half_diagonal": 50.0,
        }
    )
    label.validate()
    payload = label.to_json_dict()

    assert payload["extrapolation"]["verdict"] == VERDICT_FAR_EXTRAPOLATED
    assert label.is_extrapolated is True
    # Still a ground-truth candidate: the click is correct, the camera model
    # under it is unvalidated. Those are different claims.
    assert label.is_ground_truth_candidate is True
    restored = BallLabel.from_json_dict(payload)
    assert restored.extrapolation == payload["extrapolation"]
    assert restored.is_extrapolated is True


def test_label_rejects_an_unknown_extrapolation_verdict() -> None:
    from threed.racketsport.ball_label_schema import LabelContractError

    label = _label(extrapolation={"verdict": "probably_fine"})

    with pytest.raises(LabelContractError, match="extrapolation.verdict"):
        label.validate()


def test_label_set_summary_separates_extrapolated_ground_truth_candidates() -> None:
    from threed.racketsport.ball_label_schema import BallLabelSet

    labels = [
        _label(frame=10, extrapolation={"verdict": VERDICT_WITHIN, "extrapolated": False}),
        _label(
            frame=20,
            extrapolation={"verdict": VERDICT_EXTRAPOLATED, "extrapolated": True},
        ),
        _label(
            frame=30,
            extrapolation={"verdict": VERDICT_FAR_EXTRAPOLATED, "extrapolated": True},
        ),
        _label(frame=40),  # never checked
    ]
    label_set = BallLabelSet(
        clip_id="test",
        fps=30.0,
        frame_count=600,
        image_size=(1920, 1080),
        labels=labels,
        calibration_evidence={},
        source_artifacts={},
    )
    summary = label_set.summary()

    assert summary["ground_truth_candidate_count"] == 4
    assert summary["extrapolated_ground_truth_candidate_count"] == 2
    assert summary["by_extrapolation_verdict"] == {
        VERDICT_EXTRAPOLATED: 1,
        VERDICT_FAR_EXTRAPOLATED: 1,
        VERDICT_WITHIN: 1,
        "not_checked": 1,
    }


def test_geometry_helper_flags_the_owner_far_bounce() -> None:
    from threed.racketsport.ball_label_geometry import pixel_extrapolation

    record = pixel_extrapolation(_OUTDOOR_CALIBRATION, [118.9, 786.3])

    assert record["available"] is True
    assert record["verdict"] == VERDICT_FAR_EXTRAPOLATED
    assert record["extrapolated"] is True
    assert record["radius_pct_of_half_diagonal"] == pytest.approx(79.6, abs=0.1)
    assert record["note"]

    inside = pixel_extrapolation(_OUTDOOR_CALIBRATION, [490.6, 585.7])
    assert inside["verdict"] == VERDICT_WITHIN
    assert inside["extrapolated"] is False
    assert inside["note"] == ""

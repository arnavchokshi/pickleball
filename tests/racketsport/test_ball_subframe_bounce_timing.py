"""Sub-frame bounce timing: recovery, guards, honesty, and off-path identity.

A bounce lands between frames, so the marked frame shows the ball still above
the plane and forcing its ray down to z=ball_radius overshoots along the ray by
height/|ray_z| -- always positive, hence a bias rather than scatter.  These
tests cover the estimator that finds the contact instant instead, the guards
that make it abstain rather than guess, and the requirement that the whole
thing is invisible while it is switched off.
"""

from __future__ import annotations

import hashlib
import json
import math

import pytest

from threed.racketsport import ball_arc_solver as solver
from threed.racketsport.ball_arc_solver import (
    BallArcSolverConfig,
    BallObservation,
    SubFrameBounceTiming,
    anchor_uncertainty_for_bounce,
    build_bounce_anchor,
    intersect_ray_z,
    pixel_ray_world,
    refine_bounce_contact_time,
    solve_ball_arc_track,
)

BALL_RADIUS_M = 0.0371
GRAVITY = 9.80665

# Generated at base commit e209112 with the sub-frame arguments absent.  Any
# drift in the default bounce-anchor payload -- including a new key inside
# `uncertainty.terms` -- breaks this hash.
GOLDEN_ANCHOR_SHA256 = "58dd3851a35b35284937141304b3bf2c5c179a46a8af3a19491e744be39266ac"


def _golden_calibration() -> dict:
    """Verbatim `_grazing_calibration` from test_ball_arc_solver.

    Used only to reproduce the committed byte-identity digest, so it must not
    be altered. Its camera centre is actually BELOW the bounce plane, which is
    fine for hashing arithmetic and wrong for reasoning about which way a
    bounce anchor overshoots -- hence the separate court camera below.
    """

    return {
        "intrinsics": {"fx": 1400.0, "fy": 1400.0, "cx": 960.0, "cy": 540.0},
        "extrinsics": {
            "R": [
                [1.0, 0.0, 0.0],
                [0.0, math.sin(math.radians(10.0)), -math.cos(math.radians(10.0))],
                [0.0, math.cos(math.radians(10.0)), math.sin(math.radians(10.0))],
            ],
            "t": [0.0, 4.5 * math.sin(math.radians(10.0)), 20.0],
        },
        "reprojection_error_px": {"median": 0.0, "p95": 0.0},
        "image_size": [1920, 1080],
    }


def _look_at(centre: tuple[float, float, float], target: tuple[float, float, float]) -> dict:
    """OpenCV extrinsics for a camera at ``centre`` aimed at ``target``.

    Rows of R are the camera axes in world coordinates (right, down, forward),
    and ``t = -R @ centre``.
    """

    forward = solver._normalize(solver._sub(target, centre))
    right = solver._normalize(
        (forward[1] * 1.0 - forward[2] * 0.0, forward[2] * 0.0 - forward[0] * 1.0, 0.0)
    )
    down = (
        forward[1] * right[2] - forward[2] * right[1],
        forward[2] * right[0] - forward[0] * right[2],
        forward[0] * right[1] - forward[1] * right[0],
    )
    rotation = [list(right), list(down), list(forward)]
    translation = [
        -sum(rotation[i][j] * centre[j] for j in range(3)) for i in range(3)
    ]
    return {
        "intrinsics": {"fx": 1400.0, "fy": 1400.0, "cx": 960.0, "cy": 540.0},
        "extrinsics": {"R": rotation, "t": translation},
        "reprojection_error_px": {"median": 0.0, "p95": 0.0},
        "image_size": [1920, 1080],
    }


def _grazing_calibration() -> dict:
    """A real court camera: behind the baseline, 3 m up, looking down the court.

    Above the bounce plane and shallow to it, which is the geometry where a
    marked-frame anchor overshoots AWAY from the camera.
    """

    return _look_at((0.0, -12.0, 3.0), (0.0, 5.0, 0.0))


def _project(calibration: dict, world: tuple[float, float, float]) -> tuple[float, float]:
    rotation = calibration["extrinsics"]["R"]
    translation = calibration["extrinsics"]["t"]
    cam = [
        sum(rotation[i][j] * world[j] for j in range(3)) + translation[i]
        for i in range(3)
    ]
    intrinsics = calibration["intrinsics"]
    return (
        intrinsics["fx"] * cam[0] / cam[2] + intrinsics["cx"],
        intrinsics["fy"] * cam[1] / cam[2] + intrinsics["cy"],
    )


def _bounce_track(
    calibration: dict,
    *,
    fps: float = 30.0,
    t_contact: float = 1.0 + 0.4 / 30.0,
    contact_xy: tuple[float, float] = (0.4, 4.0),
    incoming: tuple[float, float, float] = (1.2, 3.0, -4.2),
    outgoing: tuple[float, float, float] = (0.9, 2.2, 3.1),
    frames: range | None = None,
) -> list[BallObservation]:
    """A ballistic bounce whose contact deliberately falls BETWEEN frames."""

    contact = (contact_xy[0], contact_xy[1], BALL_RADIUS_M)
    observations: list[BallObservation] = []
    for frame in frames or range(22, 39):
        t = frame / fps
        d = t - t_contact
        velocity = incoming if d <= 0.0 else outgoing
        world = (
            contact[0] + velocity[0] * d,
            contact[1] + velocity[1] * d,
            contact[2] + velocity[2] * d - 0.5 * GRAVITY * d * d,
        )
        observations.append(
            BallObservation(
                frame=frame,
                t=t,
                xy=_project(calibration, world),
                confidence=0.95,
                visible=True,
            )
        )
    return observations


def _marked_frame(observations: list[BallObservation], t_contact: float) -> int:
    return min(observations, key=lambda obs: abs(obs.t - t_contact)).frame


# --------------------------------------------------------------------------
# The estimator itself
# --------------------------------------------------------------------------
def test_recovers_a_contact_instant_that_falls_between_two_frames() -> None:
    calibration = _grazing_calibration()
    fps = 30.0
    t_contact = 1.0 + 0.4 / fps
    observations = _bounce_track(calibration, fps=fps, t_contact=t_contact)
    frame = _marked_frame(observations, t_contact)

    timing = refine_bounce_contact_time(observations, frame, fps=fps)

    assert timing is not None
    assert timing.status == "refined"
    # The marked frame is 0.4/30 s = 13.3 ms away from contact; the estimate
    # has to land far inside that or it has not bought anything.
    assert abs(timing.t_contact_s - t_contact) < 1e-3
    assert abs(timing.t_frame_s - t_contact) > 0.010
    assert abs(timing.dt_from_frame_s) > 0.0
    # The contact pixel must be the projection of the true contact point.
    expected_pixel = _project(calibration, (0.4, 4.0, BALL_RADIUS_M))
    assert math.dist(timing.pixel_xy, expected_pixel) < 0.5
    # The marked frame's own pixel is several px off the contact pixel, which
    # is the whole error the refinement removes.
    assert math.dist(timing.frame_pixel_xy, expected_pixel) > 8 * math.dist(
        timing.pixel_xy, expected_pixel
    )


def test_refined_ray_lands_on_the_plane_while_the_marked_frame_overshoots() -> None:
    """The defect and its fix, measured on one synthetic bounce.

    The marked frame's ray must overshoot AWAY from the camera by the modelled
    height/|ray_z|; the refined ray must not.
    """

    calibration = _grazing_calibration()
    fps = 30.0
    t_contact = 1.0 + 0.4 / fps
    observations = _bounce_track(calibration, fps=fps, t_contact=t_contact)
    frame = _marked_frame(observations, t_contact)
    marked = next(obs for obs in observations if obs.frame == frame)
    truth = (0.4, 4.0, BALL_RADIUS_M)

    origin, direction = pixel_ray_world(calibration, marked.xy)
    naive = intersect_ray_z(origin, direction, BALL_RADIUS_M)

    timing = refine_bounce_contact_time(observations, frame, fps=fps)
    origin, direction = pixel_ray_world(calibration, timing.pixel_xy)
    refined = intersect_ray_z(origin, direction, BALL_RADIUS_M)

    naive_error = math.dist(naive, truth)
    refined_error = math.dist(refined, truth)
    assert refined_error < 0.1 * naive_error

    # The naive error is directed away from the camera, which is what makes it
    # a bias rather than noise.
    camera = pixel_ray_world(calibration, marked.xy)[0]
    away = solver._normalize(solver._sub(truth, camera))
    assert solver._dot(solver._sub(naive, truth), away) > 0.0


def test_estimator_never_reads_the_calibration_or_the_court() -> None:
    """A 2D-only estimator degrades independently of the court solve."""

    fps = 30.0
    t_contact = 1.0 + 0.4 / fps
    observations = _bounce_track(_grazing_calibration(), fps=fps, t_contact=t_contact)
    frame = _marked_frame(observations, t_contact)

    # Same call, no calibration argument exists to pass.
    timing = refine_bounce_contact_time(observations, frame, fps=fps)
    assert timing.status == "refined"
    assert "calibration" not in refine_bounce_contact_time.__code__.co_varnames


# --------------------------------------------------------------------------
# Guards: abstain rather than guess
# --------------------------------------------------------------------------
def test_a_track_with_no_velocity_discontinuity_is_refused() -> None:
    """No kink means no contact to localise, whatever the caller marked."""

    calibration = _grazing_calibration()
    fps = 30.0
    observations = [
        BallObservation(
            frame=frame,
            t=frame / fps,
            xy=_project(
                calibration,
                (0.4 + 1.0 * frame / fps, 4.0 + 2.0 * frame / fps, 1.5),
            ),
            confidence=0.95,
            visible=True,
        )
        for frame in range(22, 39)
    ]

    timing = refine_bounce_contact_time(observations, 30, fps=fps)

    assert timing is not None
    assert timing.status != "refined"
    assert timing.status in {"rejected_weak_kink", "rejected_search_bound", "rejected_displacement"}


def test_a_runaway_fit_is_capped_by_the_displacement_guard() -> None:
    calibration = _grazing_calibration()
    fps = 30.0
    t_contact = 1.0 + 0.4 / fps
    observations = _bounce_track(calibration, fps=fps, t_contact=t_contact)
    frame = _marked_frame(observations, t_contact)

    timing = refine_bounce_contact_time(
        observations, frame, fps=fps, max_displacement_factor=0.0
    )

    assert timing.status == "rejected_displacement"
    assert timing.refined is False


def test_a_frame_with_no_observation_yields_no_timing() -> None:
    calibration = _grazing_calibration()
    observations = _bounce_track(calibration)
    assert refine_bounce_contact_time(observations, 9999) is None


def test_too_few_samples_on_one_side_abstains_instead_of_extrapolating() -> None:
    calibration = _grazing_calibration()
    fps = 30.0
    t_contact = 1.0 + 0.4 / fps
    # Only one observation after contact.
    observations = _bounce_track(
        calibration, fps=fps, t_contact=t_contact, frames=range(22, 32)
    )
    frame = _marked_frame(observations, t_contact)

    timing = refine_bounce_contact_time(observations, frame, fps=fps)

    assert timing is not None
    assert timing.refined is False


# --------------------------------------------------------------------------
# Honesty of the reported uncertainty
# --------------------------------------------------------------------------
def test_reported_instant_is_never_tighter_than_the_declared_floor() -> None:
    calibration = _grazing_calibration()
    fps = 30.0
    t_contact = 1.0 + 0.4 / fps
    observations = _bounce_track(calibration, fps=fps, t_contact=t_contact)
    frame = _marked_frame(observations, t_contact)

    timing = refine_bounce_contact_time(observations, frame, fps=fps)

    interval = 1.0 / fps
    floor = solver.DEFAULT_SUBFRAME_TIMING_SD_FLOOR_FRAMES * interval
    # Propagating pixel noise alone predicts a far tighter instant than the
    # measured spread on TT3D, so the floor -- not the pixel term -- must bind.
    assert timing.timing_sd_s == pytest.approx(floor)
    # And it may never claim to be worse than not refining at all.
    assert timing.timing_sd_s <= timing.unrefined_timing_sd_s


def test_timing_term_shrinks_but_the_bias_keeps_its_sign() -> None:
    """Reduce the bias by the timing error removed -- not to zero on faith."""

    calibration = _grazing_calibration()
    xy = _project(calibration, (0.4, 4.0, BALL_RADIUS_M))

    unrefined = anchor_uncertainty_for_bounce(
        calibration, xy, base_sigma_m=0.05, fps=30.0
    )
    refined = anchor_uncertainty_for_bounce(
        calibration,
        xy,
        base_sigma_m=0.05,
        fps=30.0,
        subframe_timing_sd_s=solver.DEFAULT_SUBFRAME_TIMING_SD_FLOOR_FRAMES / 30.0,
    )

    assert refined.bias_along_ray_m < unrefined.bias_along_ray_m
    assert refined.bias_along_ray_m > 0.0, (
        "|dt| is still one-sided after refinement, so the overshoot shrinks "
        "but does not vanish; claiming zero would be a faith-based correction"
    )
    assert refined.sigma_along_ray_m < unrefined.sigma_along_ray_m
    assert refined.terms["timing_model"] == "subframe_refined_zero_mean_gaussian"
    assert "timing_model" not in unrefined.terms


def test_a_larger_reported_timing_sigma_gives_a_larger_bias() -> None:
    """The bias term must track the timing error, not a constant."""

    calibration = _grazing_calibration()
    xy = _project(calibration, (0.4, 4.0, BALL_RADIUS_M))
    biases = [
        anchor_uncertainty_for_bounce(
            calibration, xy, base_sigma_m=0.05, fps=30.0, subframe_timing_sd_s=sd
        ).bias_along_ray_m
        for sd in (0.001, 0.004, 0.008)
    ]
    assert biases == sorted(biases)


# --------------------------------------------------------------------------
# Anchor wiring
# --------------------------------------------------------------------------
def test_anchor_moves_to_the_refined_instant_and_records_it() -> None:
    calibration = _grazing_calibration()
    fps = 30.0
    t_contact = 1.0 + 0.4 / fps
    observations = _bounce_track(calibration, fps=fps, t_contact=t_contact)
    frame = _marked_frame(observations, t_contact)
    marked = next(obs for obs in observations if obs.frame == frame)
    timing = refine_bounce_contact_time(observations, frame, fps=fps)

    anchor = build_bounce_anchor(
        {"frame": frame, "t": marked.t, "fps": fps, "xy": list(marked.xy)},
        calibration,
        ball_radius_m=BALL_RADIUS_M,
        subframe_timing=timing,
    )

    assert anchor.frame == frame, "the anchor still belongs to the marked frame"
    assert anchor.t == pytest.approx(timing.t_contact_s)
    assert anchor.t != pytest.approx(marked.t)
    assert anchor.details["pixel_xy"] == pytest.approx(list(timing.pixel_xy))
    assert anchor.details["subframe_timing"]["applied"] is True
    assert math.dist(anchor.world_xyz, (0.4, 4.0, BALL_RADIUS_M)) < 0.05


def test_a_refused_timing_is_recorded_but_never_applied() -> None:
    calibration = _grazing_calibration()
    fps = 30.0
    t_contact = 1.0 + 0.4 / fps
    observations = _bounce_track(calibration, fps=fps, t_contact=t_contact)
    frame = _marked_frame(observations, t_contact)
    marked = next(obs for obs in observations if obs.frame == frame)
    refused = refine_bounce_contact_time(
        observations, frame, fps=fps, max_displacement_factor=0.0
    )

    anchor = build_bounce_anchor(
        {"frame": frame, "t": marked.t, "fps": fps, "xy": list(marked.xy)},
        calibration,
        ball_radius_m=BALL_RADIUS_M,
        subframe_timing=refused,
    )
    baseline = build_bounce_anchor(
        {"frame": frame, "t": marked.t, "fps": fps, "xy": list(marked.xy)},
        calibration,
        ball_radius_m=BALL_RADIUS_M,
    )

    assert anchor.details["subframe_timing"]["applied"] is False
    assert anchor.t == pytest.approx(baseline.t)
    assert anchor.world_xyz == pytest.approx(baseline.world_xyz)
    assert anchor.details["uncertainty"] == baseline.details["uncertainty"]


# --------------------------------------------------------------------------
# Default OFF, byte-identical
# --------------------------------------------------------------------------
def test_default_bounce_anchor_payload_is_byte_identical_to_the_base_commit() -> None:
    """No new key, no changed digit, while the feature is off."""

    anchor = build_bounce_anchor(
        {"frame": 30, "t": 1.0, "fps": 30.0, "xy": [1012.5, 636.25]},
        _golden_calibration(),
        ball_radius_m=BALL_RADIUS_M,
    )
    payload = json.dumps(anchor.to_json(), sort_keys=True, indent=2) + "\n"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()

    assert digest == GOLDEN_ANCHOR_SHA256
    assert "subframe_timing" not in anchor.details
    assert "timing_model" not in anchor.details["uncertainty"]["terms"]
    assert "subframe_timing_sd_s" not in anchor.details["uncertainty"]["terms"]


def test_solver_default_never_calls_the_estimator(monkeypatch: pytest.MonkeyPatch) -> None:
    """Off means the code is not merely inert, it is not reached."""

    assert BallArcSolverConfig().enable_subframe_bounce_timing is False

    def explode(*args: object, **kwargs: object) -> None:
        raise AssertionError("sub-frame timing ran while the knob was off")

    monkeypatch.setattr(solver, "refine_bounce_contact_time", explode)

    calibration = _grazing_calibration()
    fps = 30.0
    observations = _bounce_track(calibration, fps=fps, frames=range(0, 40))
    frames = [
        {
            "t": obs.t,
            "xy": list(obs.xy),
            "conf": obs.confidence,
            "visible": True,
            "world_xyz": None,
            "approx": False,
        }
        for obs in observations
    ]
    artifact = solve_ball_arc_track(
        ball_track={
            "schema_version": 1,
            "fps": fps,
            "source": "synthetic",
            "frames": frames,
            "bounces": [],
        },
        calibration=calibration,
        reviewed_bounces={
            "schema_version": 1,
            "status": "human_reviewed",
            "bounces": [
                {
                    "frame": 30,
                    "t": 30 / fps,
                    "source": "human_review",
                    "human_reviewed": True,
                    "review_id": "bounce_mid",
                }
            ],
        },
        config=BallArcSolverConfig(),
    )
    # The assertion that matters is the monkeypatch not firing; the artifact is
    # only here to prove the bounce-anchor paths were exercised at all.
    assert isinstance(artifact["status"], str) and artifact["status"]
    assert artifact["anchors"]


def test_enabling_the_knob_reaches_the_estimator() -> None:
    """The off-path proof above is only meaningful if the on-path works."""

    calls: list[int] = []
    original = solver.refine_bounce_contact_time

    def counted(observations, frame, **kwargs):  # type: ignore[no-untyped-def]
        calls.append(int(frame))
        return original(observations, frame, **kwargs)

    calibration = _grazing_calibration()
    fps = 30.0
    observations = _bounce_track(calibration, fps=fps, frames=range(0, 40))
    frames = [
        {
            "t": obs.t,
            "xy": list(obs.xy),
            "conf": obs.confidence,
            "visible": True,
            "world_xyz": None,
            "approx": False,
        }
        for obs in observations
    ]
    solver.refine_bounce_contact_time = counted  # type: ignore[assignment]
    try:
        solve_ball_arc_track(
            ball_track={
                "schema_version": 1,
                "fps": fps,
                "source": "synthetic",
                "frames": frames,
                "bounces": [],
            },
            calibration=calibration,
            reviewed_bounces={
                "schema_version": 1,
                "status": "human_reviewed",
                "bounces": [
                    {
                        "frame": 30,
                        "t": 30 / fps,
                        "source": "human_review",
                        "human_reviewed": True,
                        "review_id": "bounce_mid",
                    }
                ],
            },
            config=BallArcSolverConfig(enable_subframe_bounce_timing=True),
        )
    finally:
        solver.refine_bounce_contact_time = original  # type: ignore[assignment]

    assert calls, "enabling the knob must actually reach the estimator"


def test_timing_payload_round_trips_as_json() -> None:
    calibration = _grazing_calibration()
    fps = 30.0
    t_contact = 1.0 + 0.4 / fps
    observations = _bounce_track(calibration, fps=fps, t_contact=t_contact)
    frame = _marked_frame(observations, t_contact)
    timing = refine_bounce_contact_time(observations, frame, fps=fps)

    payload = timing.to_json()
    assert json.loads(json.dumps(payload)) == payload
    assert payload["schema_version"] == solver.SUBFRAME_TIMING_SCHEMA_VERSION
    assert payload["method"] == solver.SUBFRAME_TIMING_METHOD
    assert isinstance(timing, SubFrameBounceTiming)

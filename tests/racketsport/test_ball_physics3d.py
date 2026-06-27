from __future__ import annotations

import pytest

from threed.racketsport.ball_physics3d import (
    BallSample3D,
    BounceEvent,
    detect_bounce_events,
    fit_parabola_segment,
    project_bounces_to_ball_track,
)


def _arc_sample(t: float, *, z_offset: float = 0.0) -> BallSample3D:
    return BallSample3D(
        t=t,
        x=0.4 + 1.5 * t,
        y=-0.2 + 0.6 * t,
        z=1.2 + 2.1 * t - 0.5 * 9.81 * t * t + z_offset,
    )


def test_fit_parabola_segment_rejects_height_outlier_and_keeps_gravity_like_arc() -> None:
    samples = [
        _arc_sample(0.0),
        _arc_sample(0.1),
        _arc_sample(0.2),
        _arc_sample(0.3, z_offset=0.85),
        _arc_sample(0.4),
        _arc_sample(0.5),
    ]

    fit = fit_parabola_segment(samples, residual_threshold_m=0.18)

    assert fit.inlier_indices == (0, 1, 2, 4, 5)
    assert fit.outlier_indices == (3,)
    assert fit.accel_z < 0.0
    predicted = fit.predict(0.35)
    assert predicted == pytest.approx(
        (
            0.4 + 1.5 * 0.35,
            -0.2 + 0.6 * 0.35,
            1.2 + 2.1 * 0.35 - 0.5 * 9.81 * 0.35 * 0.35,
        ),
        abs=1e-9,
    )
    assert fit.rms_residual_m < 1e-9


def test_detect_bounce_events_uses_vertical_velocity_sign_change_at_z_minimum() -> None:
    samples = [
        BallSample3D(t=0.00, x=1.00, y=2.00, z=0.42),
        BallSample3D(t=0.05, x=1.10, y=2.02, z=0.18),
        BallSample3D(t=0.10, x=1.20, y=2.04, z=0.04),
        BallSample3D(t=0.15, x=1.30, y=2.06, z=0.16),
        BallSample3D(t=0.20, x=1.40, y=2.08, z=0.39),
    ]

    events = detect_bounce_events(samples)

    assert events == (
        BounceEvent(t=0.10, world_xy=(1.20, 2.04), sample_index=2, z_min=0.04),
    )


def test_project_bounces_to_ball_track_returns_schema_friendly_bounces() -> None:
    events = (
        BounceEvent(t=0.10, world_xy=(1.20, 2.04), sample_index=2, z_min=0.04),
        BounceEvent(t=0.85, world_xy=(-0.30, 1.10), sample_index=17, z_min=0.03),
    )

    assert project_bounces_to_ball_track(events) == [
        {"t": 0.10, "world_xy": [1.20, 2.04]},
        {"t": 0.85, "world_xy": [-0.30, 1.10]},
    ]

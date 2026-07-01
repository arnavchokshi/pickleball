from __future__ import annotations

import pytest

from threed.racketsport.ball_physics3d import (
    BallSample3D,
    BounceEvent,
    detect_bounce_events,
    fit_parabola_segment,
    project_bounces_to_ball_track,
    reconstruct_bounce_arcs_from_image_track,
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


def test_reconstruct_bounce_arcs_from_image_track_fits_image_only_bounce() -> None:
    calibration = _ballistic_projection_calibration()
    times = [0.00, 0.05, 0.10, 0.15, 0.20]
    frames = [
        {
            "t": t,
            "xy": _project_with_test_calibration(calibration, _synthetic_bounce_world_xyz(t)),
            "conf": 0.9,
            "visible": True,
            "approx": False,
        }
        for t in times
    ]
    ball_payload = {"schema_version": 1, "fps": 20.0, "source": "tracknet", "frames": frames, "bounces": []}

    reconstruction = reconstruct_bounce_arcs_from_image_track(
        ball_payload,
        calibration,
        image_size=(1920, 1080),
        max_reprojection_rmse_px=1.0,
    )

    assert reconstruction.status == "ran"
    assert reconstruction.sample_count == len(times)
    assert reconstruction.reprojection_rmse_px < 1.0
    assert reconstruction.bounces == [{"t": pytest.approx(0.10), "world_xy": pytest.approx([1.20, 2.04])}]
    assert reconstruction.samples[2].z == pytest.approx(0.04, abs=0.05)


def _ballistic_projection_calibration() -> dict:
    return {
        "schema_version": 1,
        "sport": "pickleball",
        "homography": [[1000.0 / 12.0, 0.0, 960.0], [0.0, 1000.0 / 12.0, 540.0], [0.0, 0.0, 1.0]],
        "intrinsics": {"fx": 1000.0, "fy": 1000.0, "cx": 960.0, "cy": 540.0, "dist": [], "source": "test"},
        "extrinsics": {
            "R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "t": [0.0, 0.0, 12.0],
            "camera_height_m": 12.0,
        },
        "reprojection_error_px": {"median": 0.0, "p95": 0.0},
        "capture_quality": {"grade": "good", "reasons": []},
        "image_pts": [[876.6666667, -18.8], [1043.3333333, -18.8], [1043.3333333, 1098.8], [876.6666667, 1098.8]],
        "world_pts": [[-1.0, -6.7056, 0.0], [1.0, -6.7056, 0.0], [1.0, 6.7056, 0.0], [-1.0, 6.7056, 0.0]],
        "image_size": [1920, 1080],
    }


def _synthetic_bounce_world_xyz(t: float) -> tuple[float, float, float]:
    bounce_t = 0.10
    dt = t - bounce_t
    x = 1.20 + 4.0 * dt
    y = 2.04 + 0.4 * dt
    if dt <= 0.0:
        z = 0.04 + 3.0 * (-dt) - 0.5 * 9.81 * dt * dt
    else:
        z = 0.04 + 2.6 * dt - 0.5 * 9.81 * dt * dt
    return x, y, z


def _project_with_test_calibration(calibration: dict, world_xyz: tuple[float, float, float]) -> list[float]:
    intrinsics = calibration["intrinsics"]
    translation = calibration["extrinsics"]["t"]
    camera_x = world_xyz[0] + translation[0]
    camera_y = world_xyz[1] + translation[1]
    camera_z = world_xyz[2] + translation[2]
    return [
        intrinsics["fx"] * camera_x / camera_z + intrinsics["cx"],
        intrinsics["fy"] * camera_y / camera_z + intrinsics["cy"],
    ]

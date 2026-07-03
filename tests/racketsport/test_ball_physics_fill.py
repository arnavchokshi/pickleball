from __future__ import annotations

import pytest

from threed.racketsport.ball_physics3d import BallSample3D, BounceArcReconstruction
from threed.racketsport.ball_physics_fill import (
    PhysicsFillConfig,
    fill_ball_track_physics,
    fit_ballistic_segments,
    validate_physics_fill,
)


def _arc_xyz(t: float) -> list[float]:
    return [
        1.0 + 2.0 * t,
        -0.5 + 0.4 * t,
        1.2 + 2.6 * t - 0.5 * 9.81 * t * t,
    ]


def _frame(index: int, *, world_xyz: list[float] | None, visible: bool = True, conf: float = 0.9) -> dict:
    t = index / 10.0
    return {
        "t": t,
        "xy": [100.0 + index * 4.0, 200.0 - index],
        "conf": conf,
        "visible": visible,
        "world_xyz": world_xyz,
        "approx": False,
    }


def _track(frames: list[dict], *, fps: float = 10.0) -> dict:
    return {"schema_version": 1, "fps": fps, "source": "wasb", "frames": frames, "bounces": []}


def test_fill_preserves_confident_world_samples_and_marks_interpolated_frames() -> None:
    frames = [_frame(index, world_xyz=None, visible=False, conf=0.1) for index in range(7)]
    for index in (0, 4, 6):
        frames[index] = _frame(index, world_xyz=_arc_xyz(index / 10.0), conf=0.95)
    config = PhysicsFillConfig(
        min_segment_samples=3,
        max_anchor_gap_frames=4,
        max_fit_rms_m=1e-9,
        max_fit_max_residual_m=1e-8,
        max_extrapolate_frames=0,
        uncertainty_per_frame_m=0.03,
    )

    filled = fill_ball_track_physics(_track(frames), config=config, evidence_path="synthetic/ball_track.json")

    for index in (0, 4, 6):
        assert filled["frames"][index]["world_xyz"] == pytest.approx(_arc_xyz(index / 10.0), abs=1e-9)
        assert filled["frames"][index].get("source") != "physics_interpolated"

    for index in (1, 2, 3, 5):
        frame = filled["frames"][index]
        assert frame["world_xyz"] == pytest.approx(_arc_xyz(index / 10.0), abs=1e-9)
        assert frame["source"] == "physics_interpolated"
        assert frame["trust_band"]["stage"] == "PHYS-BALLFILL"
        assert frame["trust_band"]["badge"] == "low_confidence"
        assert frame["physics_fill"]["render_only"] is True
        assert frame["physics_fill"]["not_for_detection_metrics"] is True

    assert filled["frames"][2]["physics_fill"]["uncertainty_m"] > filled["frames"][1]["physics_fill"]["uncertainty_m"]
    assert filled["physics_fill"]["coverage"]["filled_frame_count"] == 4
    assert filled["physics_fill"]["coverage"]["input_world_xyz_count"] == 3
    assert filled["physics_fill"]["coverage"]["output_world_xyz_count"] == 7


def test_fit_ballistic_segments_splits_at_bounce_boundary() -> None:
    frames = [
        _frame(0, world_xyz=[0.0, 0.0, 0.30]),
        _frame(1, world_xyz=[0.1, 0.0, 0.12]),
        _frame(2, world_xyz=[0.2, 0.0, 0.00]),
        _frame(3, world_xyz=[0.3, 0.0, 0.05]),
        _frame(4, world_xyz=[0.4, 0.0, 0.16]),
    ]
    config = PhysicsFillConfig(
        min_segment_samples=3,
        max_anchor_gap_frames=2,
        max_fit_rms_m=0.50,
        max_fit_max_residual_m=0.75,
        bounce_z_epsilon_m=0.03,
        restitution_bounds=(0.3, 0.5),
    )

    result = fit_ballistic_segments(_track(frames), config=config)

    assert len(result.segments) == 2
    assert result.segments[0].frame_end == 2
    assert result.segments[1].frame_start == 2
    assert len(result.bounce_boundaries) == 1
    assert result.bounce_boundaries[0]["frame_index"] == 2
    assert result.bounce_boundaries[0]["within_restitution_bounds"] is True


def test_reviewed_bounces_force_segment_boundaries_even_without_z_minimum() -> None:
    frames = [_frame(index, world_xyz=_arc_xyz(index / 10.0), conf=0.95) for index in range(7)]
    reviewed_bounces = {
        "artifact_type": "racketsport_reviewed_ball_bounces",
        "status": "human_reviewed",
        "source": "human_review",
        "bounces": [
            {"frame": 3, "t": 0.3, "review_id": "bounce_0001"},
            {"frame": 99, "t": 9.9, "review_id": "bounce_outside_samples"},
        ],
    }
    config = PhysicsFillConfig(
        min_segment_samples=4,
        max_anchor_gap_frames=2,
        max_fit_rms_m=1e-9,
        max_fit_max_residual_m=1e-8,
    )

    result = fit_ballistic_segments(_track(frames), config=config, reviewed_bounces=reviewed_bounces)

    assert len(result.segments) == 2
    assert result.segments[0].frame_end == 3
    assert result.segments[1].frame_start == 3
    assert [boundary["frame_index"] for boundary in result.bounce_boundaries] == [3, 99]
    assert result.bounce_boundaries[0] == {
        "frame_index": 3,
        "t": 0.3,
        "source": "human_reviewed",
        "review_id": "bounce_0001",
        "forced_split": True,
    }
    assert result.bounce_boundaries[1]["source"] == "human_reviewed"
    assert result.bounce_boundaries[1]["review_id"] == "bounce_outside_samples"


def test_unreviewed_inflection_split_requires_wrist_cue_and_speed_sanity() -> None:
    frames = [_frame(index, world_xyz=_arc_xyz(index / 10.0), conf=0.95) for index in range(9)]
    ball_inflections = {
        "candidates": [
            {"frame": 29, "t": 2.9, "speed_before_px_s": 30784.0, "requires_additional_cues": ["wrist_velocity_peaks"]},
            {"frame": 4, "t": 0.4, "speed_before_px_s": 1100.0, "requires_additional_cues": ["wrist_velocity_peaks"]},
        ]
    }
    wrist_velocity_peaks = {"peaks": [{"frame": 4, "t": 0.4, "player_id": 1}]}
    config = PhysicsFillConfig(
        min_segment_samples=5,
        max_anchor_gap_frames=2,
        max_fit_rms_m=1e-9,
        max_fit_max_residual_m=1e-8,
        max_unreviewed_inflection_speed_px_s=5000.0,
        inflection_wrist_tolerance_frames=2,
    )

    result = fit_ballistic_segments(
        _track(frames),
        config=config,
        ball_inflections=ball_inflections,
        wrist_velocity_peaks=wrist_velocity_peaks,
    )

    boundary_frames = [boundary["frame_index"] for boundary in result.bounce_boundaries]
    assert boundary_frames == [4]
    assert result.bounce_boundaries[0]["source"] == "unreviewed_inflection_wrist_cross_checked"
    assert any("frame 29" in note and "speed_before_px_s" in note for note in result.notes)


def test_segment_fitter_recovers_local_segments_when_full_run_is_not_one_arc() -> None:
    frames = []
    for index in range(10):
        frames.append(_frame(index, world_xyz=[float(index), 0.0, 0.0], conf=0.95))
    config = PhysicsFillConfig(
        min_segment_samples=4,
        max_local_segment_samples=4,
        max_anchor_gap_frames=2,
        max_anchor_speed_mps=999.0,
        max_fit_rms_m=0.09,
        max_fit_max_residual_m=0.13,
    )

    result = fit_ballistic_segments(_track(frames), config=config)

    assert len(result.segments) >= 2
    assert result.segments[0].sample_frame_indices == (0, 1, 2, 3)


def test_fill_applies_physics3d_reconstruction_as_render_only_z_without_using_detection_metrics() -> None:
    frames = [_frame(index, world_xyz=[float(index), 0.0, 0.0], conf=0.95) for index in range(4)]
    reconstruction = BounceArcReconstruction(
        status="ran",
        samples=(
            BallSample3D(t=0.1, x=1.0, y=0.0, z=0.32),
            BallSample3D(t=0.2, x=2.0, y=0.0, z=0.04),
        ),
        frame_indices=(1, 2),
        reprojection_rmse_px=2.25,
        max_reprojection_error_px=3.5,
        candidate_count=12,
    )
    config = PhysicsFillConfig(min_segment_samples=3)

    filled = fill_ball_track_physics(
        _track(frames),
        config=config,
        physics3d_reconstruction=reconstruction,
        evidence_path="synthetic/ball_track.json",
    )

    airborne = filled["frames"][1]
    assert airborne["world_xyz"] == pytest.approx([1.0, 0.0, 0.32])
    assert airborne["original_world_xyz"] == [1.0, 0.0, 0.0]
    assert airborne["source"] == "physics3d_reconstructed"
    assert airborne["physics_fill"]["source_stage"] == "ball_physics3d"
    assert airborne["physics_fill"]["render_only"] is True
    assert airborne["physics_fill"]["not_for_detection_metrics"] is True
    assert airborne["physics_fill"]["reprojection_rmse_px"] == pytest.approx(2.25)
    assert airborne["render_uncertainty_m"] > 0.0
    assert filled["physics_fill"]["coverage"]["physics3d_reconstructed_frame_count"] == 2


def test_fill_adds_short_gap_xy_interpolation_without_overwriting_measured_xy_or_visible() -> None:
    frames = [_frame(index, world_xyz=None, visible=False, conf=0.1) for index in range(5)]
    frames[0] = _frame(0, world_xyz=None, visible=True, conf=0.95)
    frames[0]["xy"] = [100.0, 200.0]
    frames[2]["xy"] = None
    frames[3] = _frame(3, world_xyz=None, visible=True, conf=0.95)
    frames[3]["xy"] = [130.0, 200.0]
    frames[4] = _frame(4, world_xyz=None, visible=True, conf=0.95)
    frames[4]["xy"] = [140.0, 200.0]
    config = PhysicsFillConfig(min_segment_samples=3, max_xy_interpolate_gap_frames=5)

    filled = fill_ball_track_physics(_track(frames), config=config)

    assert filled["frames"][2].get("xy") is None
    assert filled["frames"][2]["visible"] is False
    assert filled["frames"][2]["xy_interpolated"]["xy"] == pytest.approx([120.0, 200.0])
    assert filled["frames"][2]["xy_interpolated"]["source"] == "short_gap_linear_interpolation"
    assert filled["frames"][2]["xy_interpolated"]["render_only"] is True
    assert filled["frames"][2]["xy_interpolated"]["not_for_detection_metrics"] is True
    assert filled["frames"][3]["xy"] == [130.0, 200.0]
    assert filled["frames"][3]["visible"] is True
    assert filled["frames"][3]["xy_interpolated"]["source"] == "measured_xy_no_world"
    assert filled["frames"][0]["xy_interpolated"]["source"] == "measured_xy_no_world"
    assert filled["frames"][4]["xy_interpolated"]["source"] == "measured_xy_no_world"
    assert filled["physics_fill"]["coverage"]["xy_interpolated_frame_count"] == 5


def test_2d_only_lift_requires_small_reprojection_error() -> None:
    calibration = _projection_calibration()
    frames = [_frame(index, world_xyz=None, conf=0.95) for index in range(5)]
    for index in (0, 2, 4):
        world_xyz = _arc_xyz(index / 10.0)
        frames[index] = _frame(index, world_xyz=world_xyz, conf=0.95)
        frames[index]["xy"] = _project(calibration, world_xyz)

    frames[1]["xy"] = _project(calibration, _arc_xyz(0.1))
    frames[3]["xy"] = [1800.0, 900.0]
    config = PhysicsFillConfig(
        min_segment_samples=3,
        max_anchor_gap_frames=3,
        max_fit_rms_m=1e-9,
        max_fit_max_residual_m=1e-8,
        max_reprojection_error_px=2.0,
        max_extrapolate_frames=0,
    )

    filled = fill_ball_track_physics(_track(frames), calibration=calibration, config=config)

    assert filled["frames"][1]["source"] == "physics_interpolated"
    assert filled["frames"][1]["physics_fill"]["reprojection_error_px"] == pytest.approx(0.0, abs=1e-6)
    assert filled["frames"][3].get("source") != "physics_interpolated"
    assert filled["frames"][3]["world_xyz"] is None
    assert filled["physics_fill"]["coverage"]["reprojection_rejected_frame_count"] == 1


def test_fill_does_not_extrapolate_beyond_configured_frame_limit() -> None:
    frames = [_frame(index, world_xyz=None, visible=False, conf=0.1) for index in range(8)]
    for index in (2, 4, 6):
        frames[index] = _frame(index, world_xyz=_arc_xyz(index / 10.0), conf=0.95)
    config = PhysicsFillConfig(
        min_segment_samples=3,
        max_anchor_gap_frames=3,
        max_fit_rms_m=1e-9,
        max_fit_max_residual_m=1e-8,
        max_extrapolate_frames=1,
    )

    filled = fill_ball_track_physics(_track(frames), config=config)

    assert filled["frames"][0]["world_xyz"] is None
    assert filled["frames"][1]["source"] == "physics_interpolated"
    assert filled["frames"][7]["source"] == "physics_interpolated"


def test_generated_samples_clamp_small_negative_z_to_court_plane() -> None:
    frames = [_frame(index, world_xyz=None, visible=False, conf=0.1) for index in range(5)]
    for index in (1, 2, 3):
        frames[index] = _frame(index, world_xyz=[float(index), 0.0, 0.0], conf=0.95)
    config = PhysicsFillConfig(
        min_segment_samples=3,
        max_local_segment_samples=3,
        max_anchor_gap_frames=2,
        max_anchor_speed_mps=999.0,
        max_fit_rms_m=0.10,
        max_fit_max_residual_m=0.20,
        max_extrapolate_frames=1,
        floor_tolerance_m=0.20,
    )

    filled = fill_ball_track_physics(_track(frames), config=config)

    assert filled["frames"][0]["source"] == "physics_interpolated"
    assert filled["frames"][0]["world_xyz"][2] == 0.0
    assert filled["frames"][0]["physics_fill"]["raw_world_xyz"][2] < 0.0
    assert filled["frames"][0]["physics_fill"]["clamped_to_court_plane"] is True
    assert filled["physics_fill"]["coverage"]["clamped_to_court_plane_frame_count"] >= 1


def test_leave_one_out_validation_reports_3d_and_2d_error_distributions() -> None:
    calibration = _projection_calibration()
    frames = []
    for index in range(6):
        world_xyz = _arc_xyz(index / 10.0)
        frame = _frame(index, world_xyz=world_xyz, conf=0.95)
        frame["xy"] = _project(calibration, world_xyz)
        frames.append(frame)
    config = PhysicsFillConfig(
        min_segment_samples=4,
        max_anchor_gap_frames=2,
        max_fit_rms_m=1e-9,
        max_fit_max_residual_m=1e-8,
    )

    report = validate_physics_fill(_track(frames), calibration=calibration, config=config, seed=7)

    assert report["leave_one_out"]["sample_count"] == 6
    assert report["leave_one_out"]["error_3d_m"]["max"] == pytest.approx(0.0, abs=1e-8)
    assert report["leave_one_out"]["reprojection_error_px"]["max"] == pytest.approx(0.0, abs=1e-5)


def _projection_calibration() -> dict:
    return {
        "intrinsics": {"fx": 800.0, "fy": 800.0, "cx": 640.0, "cy": 360.0},
        "extrinsics": {
            "R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "t": [0.0, 0.0, 10.0],
        },
    }


def _project(calibration: dict, world_xyz: list[float]) -> list[float]:
    intrinsics = calibration["intrinsics"]
    tx, ty, tz = calibration["extrinsics"]["t"]
    camera_x = world_xyz[0] + tx
    camera_y = world_xyz[1] + ty
    camera_z = world_xyz[2] + tz
    return [
        intrinsics["fx"] * camera_x / camera_z + intrinsics["cx"],
        intrinsics["fy"] * camera_y / camera_z + intrinsics["cy"],
    ]

from __future__ import annotations

import pytest

from threed.racketsport.physics_predictors import (
    BallBallisticAdapter,
    FootContactLockAdapter,
    JointKinematicPredictor,
    PaddleNullPredictor,
)


def test_paddle_null_predictor_returns_no_prediction() -> None:
    result = PaddleNullPredictor().predict(history=[{"frame_index": 10}], horizon_frames=3)

    assert result.available is False
    assert result.state is None
    assert result.sigma_m is None
    assert result.reason == "no_paddle_prediction_possible"


def test_joint_kinematic_predictor_clamps_velocity_for_short_gaps() -> None:
    predictor = JointKinematicPredictor(fps=30.0, max_speed_mps=3.0, base_sigma_m=0.02, sigma_per_frame_m=0.01)
    history = [
        {"frame_index": 0, "state": [0.0, 0.0, 0.0]},
        {"frame_index": 1, "state": [1.0, 0.0, 0.0]},
    ]

    result = predictor.predict(history=history, horizon_frames=10)

    assert result.available is True
    assert result.state == pytest.approx([2.0, 0.0, 0.0])
    assert result.sigma_m == pytest.approx(0.12)


def test_ball_ballistic_adapter_returns_existing_filled_sample_without_recomputing() -> None:
    adapter = BallBallisticAdapter(
        {
            "frames": [
                {"t": 0.0, "world_xyz": [0.0, 0.0, 1.0], "conf": 0.9},
                {
                    "t": 1.0 / 30.0,
                    "world_xyz": [0.1, 0.0, 0.9],
                    "conf": 0.2,
                    "source": "physics_interpolated",
                    "physics_fill": {"uncertainty_m": 0.31},
                },
            ]
        }
    )

    result = adapter.predict(history=[{"frame_index": 0}], horizon_frames=1)

    assert result.available is True
    assert result.state == [0.1, 0.0, 0.9]
    assert result.sigma_m == pytest.approx(0.31)
    assert result.provenance["source"] == "physics_interpolated"


def test_foot_contact_lock_adapter_returns_existing_corrected_joint_frame() -> None:
    adapter = FootContactLockAdapter(
        {
            "players": [
                {
                    "id": 7,
                    "frames": [
                        {"frame_index": 4, "joints_world": [[0.0, 0.0, 0.0]], "joint_conf": [0.8]},
                        {"frame_index": 5, "joints_world": [[0.1, 0.0, 0.0]], "joint_conf": [0.7]},
                    ],
                }
            ]
        }
    )

    result = adapter.predict(history=[{"player_id": 7, "frame_index": 4}], horizon_frames=1)

    assert result.available is True
    assert result.state == [[0.1, 0.0, 0.0]]
    assert result.sigma_m is not None
    assert result.provenance["player_id"] == 7

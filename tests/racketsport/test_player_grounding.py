from __future__ import annotations

import pytest

from threed.racketsport.court_positioning import CameraFloorGeometry
from threed.racketsport.player_grounding import FootImageObservation, ground_foot_observation, ground_player_feet


def _camera_geometry() -> CameraFloorGeometry:
    return CameraFloorGeometry(
        intrinsics={"fx": 100.0, "fy": 100.0, "cx": 50.0, "cy": 50.0, "dist": []},
        camera_origin_world=[0.0, 0.0, 2.0],
        R_world_camera=[
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, -1.0],
        ],
        floor_plane_point=[0.0, 0.0, 0.0],
        floor_plane_normal=[0.0, 0.0, 1.0],
    )


def test_ground_foot_observation_projects_pose_points_to_court_xy_and_contact() -> None:
    observation = FootImageObservation(
        side="L",
        pixels={"ankle": [50.0, 50.0], "heel": [51.0, 50.0], "toe": [49.0, 50.0]},
        confidence=0.9,
        height_m=0.015,
        previous_court_xy=[0.0, -0.01],
        dt_s=0.1,
    )

    foot = ground_foot_observation(
        observation,
        geometry=_camera_geometry(),
        T_world_court=[
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        metric_confidence="high",
    )

    assert foot.side == "L"
    assert foot.court_xy == pytest.approx([0.0, 0.0])
    assert foot.height_m == pytest.approx(0.015)
    assert foot.contact is True
    assert foot.sigma_p_m > 0.0
    assert foot.source_points == ("ankle", "heel", "toe")


def test_ground_player_feet_fails_contact_closed_for_low_pose_or_metric_confidence() -> None:
    observations = [
        FootImageObservation(side="L", pixels={"ankle": [50.0, 50.0]}, confidence=0.4, height_m=0.010),
        FootImageObservation(side="R", pixels={"ankle": [60.0, 50.0]}, confidence=0.9, height_m=0.010),
    ]

    low_pose, low_metric = ground_player_feet(
        observations,
        geometry=_camera_geometry(),
        T_world_court=[
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        metric_confidence="low",
    )

    assert low_pose.contact is False
    assert low_metric.contact is False
    assert low_pose.court_xy == pytest.approx([0.0, 0.0])
    assert low_metric.court_xy == pytest.approx([0.2, 0.0])

from __future__ import annotations

import copy
import json

import pytest

from threed.racketsport.placement_trajectory_refine import (
    MalformedPlacementInputError,
    MissingPlacementInputError,
    PlacementTrajectoryConfig,
    refine_placement_trajectory,
    score_placement_slide,
)


JOINT_NAMES = ["left_ankle", "right_ankle", "left_heel", "right_heel"]


def _skeleton(*, jitter: list[float], sole_z: float = 0.01) -> dict:
    frames = []
    for frame_index, offset in enumerate(jitter):
        root = [frame_index * 0.02 + offset, 1.0, 1.0]
        frames.append(
            {
                "frame_idx": frame_index,
                "t": frame_index / 30.0,
                "transl_world": root,
                "joints_world": [
                    [root[0] - 0.1, 1.0, sole_z],
                    [root[0] + 0.1, 1.0, sole_z],
                    [root[0] - 0.12, 1.02, sole_z],
                    [root[0] + 0.12, 1.02, sole_z],
                ],
                "joint_conf": [0.98] * 4,
            }
        )
    return {
        "schema_version": 1,
        "artifact_type": "skeleton3d",
        "world_frame": "court_Z0",
        "fps": 30.0,
        "joint_names": JOINT_NAMES,
        "players": [{"id": "p1", "frames": frames}],
    }


def _tracks(frame_count: int, *, outlier_frame: int | None = None) -> dict:
    frames = []
    for frame_index in range(frame_count):
        world_x = frame_index * 0.02
        if frame_index == outlier_frame:
            world_x += 2.0
        frames.append(
            {
                "frame_idx": frame_index,
                "t": frame_index / 30.0,
                "world_xy": [world_x, 1.0],
                "conf": 0.99,
                "bbox": [100.0, 100.0, 200.0, 300.0],
            }
        )
    return {"schema_version": 1, "fps": 30.0, "players": [{"id": "p1", "frames": frames}]}


def _phases(frame_indices: list[int]) -> dict:
    return {
        "schema_version": 1,
        "artifact_type": "foot_contact_phases",
        "phases": [
            {
                "player_id": "p1",
                "foot": "left",
                "frame_indices": frame_indices,
                "start_frame_index": frame_indices[0],
                "end_frame_index": frame_indices[-1],
                "min_confidence": 0.96,
                "assignment_evidence": {"body_detector_agreement": 0.97},
            }
        ],
    }


def _left_x(payload: dict, frame_index: int) -> float:
    return float(payload["players"][0]["frames"][frame_index]["joints_world"][0][0])


def test_known_plant_jitter_reduces_below_bound_and_clean_segment_stays_put() -> None:
    source = _skeleton(jitter=[0.0, 0.0, 0.04, -0.03, 0.05, -0.02, 0.0, 0.0])
    refined = refine_placement_trajectory(
        source,
        tracks_payload=_tracks(8),
        foot_contact_phases=_phases([2, 3, 4, 5]),
    )

    plant = [_left_x(refined, index) for index in [2, 3, 4, 5]]
    assert max(plant) - min(plant) < 0.035
    for index in [0, 1, 6, 7]:
        correction = refined["players"][0]["frames"][index]["placement_trajectory_refinement"][
            "correction_magnitude_m"
        ]
        assert correction < 0.001


def test_two_metre_track_outlier_is_huber_bounded_and_does_not_drag_plant() -> None:
    source = _skeleton(jitter=[0.0, 0.03, -0.02, 0.04, 0.0])
    refined = refine_placement_trajectory(
        source,
        tracks_payload=_tracks(5, outlier_frame=2),
        foot_contact_phases=_phases([1, 2, 3]),
    )
    correction = refined["players"][0]["frames"][2]["placement_trajectory_refinement"][
        "rigid_correction_xyz_m"
    ]
    assert abs(correction[0]) < 0.08
    plant = [_left_x(refined, index) for index in [1, 2, 3]]
    assert max(plant) - min(plant) < 0.04


def test_soft_court_prior_never_snaps_nonzero_sole_to_zero() -> None:
    source = _skeleton(jitter=[0.0, 0.0], sole_z=0.05)
    refined = refine_placement_trajectory(
        source,
        tracks_payload=_tracks(2),
        foot_contact_phases=_phases([0, 1]),
        config=PlacementTrajectoryConfig(z_plane_weight=10.0, z_body_weight=20.0),
    )
    after = refined["players"][0]["frames"][0]["joints_world"][0][2]
    assert 0.0 < after < 0.05
    provenance = refined["players"][0]["frames"][0]["placement_trajectory_refinement"]["provenance"]
    assert provenance["z_soft_prior"]["clamped_to_plane"] is False


def test_fail_closed_on_missing_calibration_empty_phases_and_nan_confidence() -> None:
    source = _skeleton(jitter=[0.0, 0.0])
    tracks = _tracks(2)
    with pytest.raises(MissingPlacementInputError, match="at least one phase"):
        refine_placement_trajectory(source, tracks_payload=tracks, foot_contact_phases={"phases": []})

    bad_tracks = copy.deepcopy(tracks)
    bad_tracks["players"][0]["frames"][0]["conf"] = float("nan")
    with pytest.raises(MalformedPlacementInputError, match="finite"):
        refine_placement_trajectory(source, tracks_payload=bad_tracks, foot_contact_phases=_phases([0, 1]))

    with pytest.raises(MissingPlacementInputError, match="calibration"):
        score_placement_slide(
            source,
            frozen_phases_payload=_phases([0, 1]),
            calibration_payload={},
            tracks_payload=tracks,
        )


def test_refiner_is_byte_deterministic() -> None:
    source = _skeleton(jitter=[0.0, 0.02, -0.01, 0.0])
    kwargs = {"tracks_payload": _tracks(4), "foot_contact_phases": _phases([0, 1, 2, 3])}
    first = refine_placement_trajectory(source, **kwargs)
    second = refine_placement_trajectory(source, **kwargs)
    assert json.dumps(first, indent=2, sort_keys=True) == json.dumps(second, indent=2, sort_keys=True)


from __future__ import annotations

import pytest

from threed.racketsport.person_identity_fragments import build_identity_fragment_artifacts


def _raw_pool() -> dict:
    frames = []
    for frame_idx in range(2):
        frames.append(
            {
                "frame": frame_idx,
                "detections": [
                    {
                        "bbox": [10.0, 10.0, 30.0, 90.0],
                        "class": "person",
                        "conf": 0.91,
                        "track_id": 1,
                        "world_xy": [-1.0 + frame_idx * 0.1, -5.0],
                    },
                    {
                        "bbox": [40.0, 10.0, 60.0, 90.0],
                        "class": "person",
                        "conf": 0.92,
                        "track_id": 2,
                        "world_xy": [1.0 + frame_idx * 0.1, -5.0],
                    },
                    {
                        "bbox": [70.0, 10.0, 90.0, 70.0],
                        "class": "person",
                        "conf": 0.86,
                        "track_id": 3,
                        "world_xy": [-1.0, 4.5 + frame_idx * 0.1],
                    },
                    {
                        "bbox": [100.0, 10.0, 120.0, 70.0],
                        "class": "person",
                        "conf": 0.87,
                        "track_id": 4,
                        "world_xy": [1.0, 4.5 + frame_idx * 0.1],
                    },
                    {
                        "bbox": [200.0, 10.0, 220.0, 60.0],
                        "class": "person",
                        "conf": 0.74,
                        "track_id": 99,
                        "world_xy": [9.0, 4.0 + frame_idx * 0.1],
                    },
                ],
            }
        )
    return {"schema_version": 1, "fps": 10.0, "frames": frames}


def test_all_human_fragments_preserve_adjacent_court_human_before_target_selection() -> None:
    artifacts = build_identity_fragment_artifacts(
        raw_pool_payload=_raw_pool(),
        source_name="synthetic_raw_pool",
        expected_target_players=4,
    )

    observations = artifacts["human_observations"]
    fragments = artifacts["identity_fragments"]
    report = artifacts["identity_association_report"]

    assert observations["artifact_type"] == "racketsport_human_observations"
    assert observations["source_only"] is True
    assert observations["uses_cvat_labels"] is False
    assert len(observations["observations"]) == 10

    assert fragments["artifact_type"] == "racketsport_identity_fragments"
    assert len(fragments["fragments"]) == 5
    assert {fragment["source_tracker_id"] for fragment in fragments["fragments"]} == {1, 2, 3, 4, 99}
    spectator = next(fragment for fragment in fragments["fragments"] if fragment["source_tracker_id"] == 99)
    assert spectator["coverage_frames"] == 2
    assert spectator["eligible_for_target_selection"] is False
    assert "outside_target_court_geometry" in spectator["target_selection_blockers"]

    assert report["input_human_observation_count"] == 10
    assert report["fragment_count"] == 5
    assert report["target_player_candidate_count"] == 4
    assert report["non_target_diagnostic_fragment_count"] == 1
    assert report["final_tracks_written"] is False


def test_all_human_fragments_project_raw_pool_footpoints_from_calibration_when_world_missing() -> None:
    raw_pool = {
        "schema_version": 1,
        "fps": 10.0,
        "frames": [
            {
                "frame": 0,
                "detections": [
                    {
                        "bbox": [390.0, -50.0, 410.0, 0.0],
                        "class": "person",
                        "conf": 0.91,
                        "track_id": 1,
                    },
                    {
                        "bbox": [1400.0, 950.0, 1420.0, 1000.0],
                        "class": "person",
                        "conf": 0.74,
                        "track_id": 99,
                    },
                ],
            }
        ],
    }
    calibration = {
        "homography": [
            [100.0, 0.0, 500.0],
            [0.0, 100.0, 500.0],
            [0.0, 0.0, 1.0],
        ]
    }

    artifacts = build_identity_fragment_artifacts(
        raw_pool_payload=raw_pool,
        calibration_payload=calibration,
        source_name="synthetic_raw_pool",
        expected_target_players=4,
    )

    observations = artifacts["human_observations"]["observations"]
    assert observations[0]["projected_world_xy"] == pytest.approx([-1.0, -5.0])
    assert observations[1]["projected_world_xy"] == pytest.approx([9.1, 5.0])
    spectator = next(
        fragment for fragment in artifacts["identity_fragments"]["fragments"] if fragment["source_tracker_id"] == 99
    )
    assert spectator["eligible_for_target_selection"] is False
    assert "outside_target_court_geometry" in spectator["target_selection_blockers"]

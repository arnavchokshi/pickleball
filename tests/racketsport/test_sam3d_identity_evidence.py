from __future__ import annotations

import pytest

from threed.racketsport.sam3d_identity_evidence import build_sam3d_identity_evidence


def test_sam3d_identity_evidence_binds_crop_to_track_and_flags_inherited_anchor() -> None:
    tracks = {
        "schema_version": 1,
        "fps": 10.0,
        "players": [
            {
                "id": 1,
                "frames": [
                    {
                        "t": 1.0,
                        "bbox": [0.0, 0.0, 100.0, 200.0],
                        "world_xy": [1.0, 2.0],
                        "conf": 0.9,
                    }
                ],
            }
        ],
    }
    prep = {
        "schema_version": 1,
        "artifact_type": "racketsport_sam3d_body_input_prep",
        "records": [
            {
                "request_id": "10:1",
                "frame_idx": 10,
                "player_id": 1,
                "original_bbox_xyxy": [0.0, 0.0, 100.0, 200.0],
                "prepared_bbox_xyxy": [-5.0, -5.0, 105.0, 205.0],
            }
        ],
    }
    keypoints = {
        "schema_version": 1,
        "artifact_type": "racketsport_sam3d_keypoints_2d",
        "players": [
            {
                "id": 1,
                "frames": [
                    {
                        "frame_idx": 10,
                        "t": 1.0,
                        "keypoints": [
                            {"name": "left_ankle", "xy_px": [20.0, 190.0], "conf": 0.7},
                            {"name": "right_ankle", "xy_px": [80.0, 190.0], "conf": 0.9},
                        ],
                    }
                ],
            }
        ],
    }
    skeleton = {
        "schema_version": 1,
        "artifact_type": "racketsport_skeleton3d",
        "fps": 10.0,
        "provenance": {"grounding_anchor_source": "placement_track_world_xy"},
        "players": [
            {
                "id": 1,
                "frames": [
                    {
                        "frame_idx": 10,
                        "t": 1.0,
                        "transl_world": [1.1, 2.0, 0.0],
                        "joint_conf": [0.8, 0.9, 1.0],
                        "confidence_provenance": {"grounding_anchor_source": "placement_track_world_xy"},
                    }
                ],
            }
        ],
    }

    evidence = build_sam3d_identity_evidence(
        tracks_payload=tracks,
        sam3d_body_input_prep=prep,
        sam3d_keypoints_2d=keypoints,
        skeleton3d_payload=skeleton,
    )

    assert evidence["artifact_type"] == "racketsport_sam3d_identity_evidence"
    assert evidence["source_only"] is True
    assert evidence["uses_cvat_labels"] is False
    assert evidence["summary"]["body_observation_count"] == 1
    row = evidence["body_observations"][0]
    assert row["body_observation_id"] == "10:1"
    assert row["frame_idx"] == 10
    assert row["player_id"] == 1
    assert row["crop_detection_residual_px"] == pytest.approx(0.0)
    assert row["joint_confidence_mean"] == pytest.approx(0.9)
    assert row["footpoint_inside_track_bbox"] is True
    assert row["root_track_residual_m"] == pytest.approx(0.1)
    assert row["transl_world_independent"] is False
    assert "placement_track_world_xy_anchor" in row["risk_flags"]


def test_sam3d_identity_evidence_flags_large_root_track_residual() -> None:
    tracks = {
        "schema_version": 1,
        "fps": 10.0,
        "players": [
            {
                "id": 3,
                "frames": [
                    {
                        "t": 2.0,
                        "bbox": [100.0, 100.0, 180.0, 260.0],
                        "world_xy": [0.0, 0.0],
                        "conf": 0.9,
                    }
                ],
            }
        ],
    }
    prep = {
        "schema_version": 1,
        "artifact_type": "racketsport_sam3d_body_input_prep",
        "records": [
            {
                "request_id": "20:3",
                "frame_idx": 20,
                "player_id": 3,
                "original_bbox_xyxy": [100.0, 100.0, 180.0, 260.0],
                "prepared_bbox_xyxy": [100.0, 100.0, 180.0, 260.0],
            }
        ],
    }
    skeleton = {
        "schema_version": 1,
        "artifact_type": "racketsport_skeleton3d",
        "fps": 10.0,
        "players": [
            {
                "id": 3,
                "frames": [
                    {
                        "frame_idx": 20,
                        "t": 2.0,
                        "transl_world": [1.25, 0.0, 0.0],
                        "joint_conf": [0.8, 0.8, 0.8],
                    }
                ],
            }
        ],
    }

    evidence = build_sam3d_identity_evidence(
        tracks_payload=tracks,
        sam3d_body_input_prep=prep,
        skeleton3d_payload=skeleton,
    )

    row = evidence["body_observations"][0]
    assert row["root_track_residual_m"] == pytest.approx(1.25)
    assert row["transl_world_independent"] is True
    assert "sam3d_root_track_residual_over_1m" in row["risk_flags"]
    assert evidence["summary"]["root_track_residual_over_1m_count"] == 1

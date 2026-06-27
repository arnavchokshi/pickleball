from __future__ import annotations

import pytest

from threed.racketsport.hmr_deep import (
    PlayerCropRequest,
    build_player_hmr_artifact,
    gate_deep_hmr_artifact,
    normalize_deep_hmr_payload,
)


def test_player_crop_request_validates_and_serializes_crop_geometry() -> None:
    request = PlayerCropRequest(
        frame_idx=42,
        player_id=3,
        bbox_xyxy=[10, 20, 110, 220],
        image_size_px=[1920, 1080],
        track_confidence=0.92,
        source_track_id="trk-3",
        rally_span_id="rally-1",
    )

    assert request.bbox_xyxy == pytest.approx((10.0, 20.0, 110.0, 220.0))
    assert request.crop_xywh == pytest.approx((10.0, 20.0, 100.0, 200.0))
    assert request.area_px == pytest.approx(20000.0)
    assert request.to_dict() == {
        "frame_idx": 42,
        "player_id": 3,
        "bbox_xyxy": [10.0, 20.0, 110.0, 220.0],
        "crop_xywh": [10.0, 20.0, 100.0, 200.0],
        "image_size_px": [1920, 1080],
        "track_confidence": 0.92,
        "source_track_id": "trk-3",
        "rally_span_id": "rally-1",
        "scaffold": "cpu_hmr_deep_primitives_no_model_inference",
    }


def test_player_crop_request_rejects_invalid_boxes_or_confidence() -> None:
    with pytest.raises(ValueError, match="frame_idx must be a non-negative integer"):
        PlayerCropRequest(
            frame_idx=1.2,  # type: ignore[arg-type]
            player_id=1,
            bbox_xyxy=[10, 10, 50, 50],
            image_size_px=[1280, 720],
            track_confidence=0.9,
        )

    with pytest.raises(ValueError, match="image_size_px/0 must be a positive integer"):
        PlayerCropRequest(
            frame_idx=0,
            player_id=1,
            bbox_xyxy=[10, 10, 50, 50],
            image_size_px=[1280.5, 720],  # type: ignore[list-item]
            track_confidence=0.9,
        )

    with pytest.raises(ValueError, match="bbox_xyxy must be ordered"):
        PlayerCropRequest(
            frame_idx=0,
            player_id=1,
            bbox_xyxy=[10, 10, 5, 50],
            image_size_px=[1280, 720],
            track_confidence=0.9,
        )

    with pytest.raises(ValueError, match="inside image_size_px"):
        PlayerCropRequest(
            frame_idx=0,
            player_id=1,
            bbox_xyxy=[10, 10, 1290, 50],
            image_size_px=[1280, 720],
            track_confidence=0.9,
        )

    with pytest.raises(ValueError, match="track_confidence"):
        PlayerCropRequest(
            frame_idx=0,
            player_id=1,
            bbox_xyxy=[10, 10, 50, 50],
            image_size_px=[1280, 720],
            track_confidence=1.01,
        )


def test_normalize_deep_hmr_payload_preserves_request_identity_and_smpl_like_output() -> None:
    request = PlayerCropRequest(
        frame_idx=7,
        player_id=2,
        bbox_xyxy=[100, 120, 220, 420],
        image_size_px=[1920, 1080],
        track_confidence=0.91,
    )
    normalized = normalize_deep_hmr_payload(
        {
            "smpl": {
                "global_orient": [0.1, 0.2, 0.3],
                "body_pose": [0.0, 0.01, 0.02, 0.03, 0.04, 0.05],
                "betas": [0.5, -0.1],
                "transl": [1, 2, 3],
            },
            "mhr": {"pose_confidence": 0.82},
            "vertices": [[0, 0, 0], [1.0, 2.0, 3.0]],
            "joints3d": [[0.5, 0.5, 0.5]],
            "confidence": 0.74,
        },
        request=request,
    )

    assert normalized == {
        "schema_version": "body_hmr_deep.v0",
        "frame_idx": 7,
        "player_id": 2,
        "model_family": "fast_sam_3d_body_mhr_to_smpl",
        "representation": "smpl_ish_cpu_normalized",
        "smpl": {
            "global_orient": [0.1, 0.2, 0.3],
            "body_pose": [0.0, 0.01, 0.02, 0.03, 0.04, 0.05],
            "betas": [0.5, -0.1],
            "transl": [1.0, 2.0, 3.0],
        },
        "mhr": {"pose_confidence": 0.82},
        "mesh_vertices_xyz": [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]],
        "joints3d_xyz": [[0.5, 0.5, 0.5]],
        "confidence": 0.74,
        "confidence_components": {
            "model_confidence": 0.74,
            "track_confidence": 0.91,
            "mhr_pose_confidence": 0.82,
        },
        "scaffold": "cpu_hmr_deep_primitives_no_model_inference",
    }


def test_normalize_deep_hmr_payload_rejects_missing_smpl_or_bad_vectors() -> None:
    request = PlayerCropRequest(
        frame_idx=0,
        player_id=1,
        bbox_xyxy=[10, 10, 50, 50],
        image_size_px=[1280, 720],
        track_confidence=0.9,
    )

    with pytest.raises(ValueError, match="smpl"):
        normalize_deep_hmr_payload({"confidence": 0.8}, request=request)

    with pytest.raises(ValueError, match="smpl.transl must be a 3-vector"):
        normalize_deep_hmr_payload(
            {
                "smpl": {"global_orient": [0, 0, 0], "body_pose": [], "betas": [], "transl": [0, 0]},
                "confidence": 0.8,
            },
            request=request,
        )


def test_gate_and_package_deep_hmr_artifact_marks_scaffold_as_not_usable() -> None:
    request = PlayerCropRequest(
        frame_idx=8,
        player_id=9,
        bbox_xyxy=[10, 20, 80, 200],
        image_size_px=[1280, 720],
        track_confidence=0.95,
    )
    normalized = normalize_deep_hmr_payload(
        {
            "smpl": {"global_orient": [0, 0, 0], "body_pose": [0.1, 0.2, 0.3], "betas": [], "transl": [0, 0, 0]},
            "vertices": [[0, 0, 0]],
            "joints3d": [[0, 0, 0]],
            "confidence": 0.88,
        },
        request=request,
    )

    gate = gate_deep_hmr_artifact(normalized, model_inference_ran=False, min_confidence=0.65)
    artifact = build_player_hmr_artifact(
        request,
        normalized,
        model_inference_ran=False,
        min_confidence=0.65,
    )

    assert gate == {
        "decision": "reject",
        "confidence": 0.88,
        "threshold": 0.65,
        "reasons": ["scaffold_only_no_model_inference"],
    }
    assert artifact["artifact_type"] == "deep_hmr_player_frame"
    assert artifact["crop_request"] == request.to_dict()
    assert artifact["hmr_output"] == normalized
    assert artifact["gate"] == gate
    assert artifact["metadata"]["model_inference_ran"] is False
    assert artifact["metadata"]["scaffold"] == "cpu_hmr_deep_primitives_no_model_inference"


def test_gate_deep_hmr_artifact_reports_low_confidence_and_missing_payload_parts() -> None:
    gate = gate_deep_hmr_artifact(
        {"confidence": 0.3, "mesh_vertices_xyz": [], "joints3d_xyz": []},
        model_inference_ran=True,
        min_confidence=0.65,
    )

    assert gate == {
        "decision": "reject",
        "confidence": 0.3,
        "threshold": 0.65,
        "reasons": ["low_confidence", "missing_mesh_vertices", "missing_joints3d"],
    }

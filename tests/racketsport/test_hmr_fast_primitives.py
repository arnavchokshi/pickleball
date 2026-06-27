from __future__ import annotations

import pytest

from threed.racketsport.hmr_fast import (
    FastTierMetadata,
    PreviewMesh,
    build_fail_closed_payload,
    package_frame_preview,
)


def test_preview_mesh_validates_camera_space_vertices_faces_and_bbox() -> None:
    mesh = PreviewMesh(
        vertices_camera=[
            [0.0, 0.0, 2.0],
            [1.0, 0.0, 2.0],
            [0.0, 1.0, 2.0],
        ],
        faces=[[0, 1, 2]],
        bbox_xyxy=[10, 20, 110, 220],
    )

    assert mesh.vertices_camera == [
        [0.0, 0.0, 2.0],
        [1.0, 0.0, 2.0],
        [0.0, 1.0, 2.0],
    ]
    assert mesh.faces == [(0, 1, 2)]
    assert mesh.bbox_xyxy == pytest.approx((10.0, 20.0, 110.0, 220.0))
    assert mesh.vertex_count == 3
    assert mesh.face_count == 1


def test_preview_mesh_rejects_invalid_preview_geometry() -> None:
    with pytest.raises(ValueError, match="vertices_camera must contain at least one vertex"):
        PreviewMesh(vertices_camera=[], faces=[], bbox_xyxy=[0.0, 0.0, 1.0, 1.0])

    with pytest.raises(ValueError, match="faces/0 index 3 is outside vertices_camera"):
        PreviewMesh(
            vertices_camera=[[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]],
            faces=[[0, 1, 3]],
            bbox_xyxy=[0.0, 0.0, 1.0, 1.0],
        )

    with pytest.raises(ValueError, match="bbox_xyxy must be ordered"):
        PreviewMesh(
            vertices_camera=[[0.0, 0.0, 1.0]],
            faces=[],
            bbox_xyxy=[10.0, 20.0, 5.0, 25.0],
        )

    with pytest.raises(ValueError, match="bbox_xyxy must be ordered"):
        PreviewMesh(
            vertices_camera=[[0.0, 0.0, 1.0]],
            faces=[],
            bbox_xyxy=[10.0, 20.0, 10.0, 25.0],
        )

    with pytest.raises(ValueError, match="finite"):
        PreviewMesh(
            vertices_camera=[[0.0, float("nan"), 1.0]],
            faces=[],
            bbox_xyxy=[0.0, 0.0, 1.0, 1.0],
        )


def test_fast_tier_metadata_is_explicitly_scaffold_and_preview_only() -> None:
    metadata = FastTierMetadata(
        model_family="multi_hmr2_candidate",
        elapsed_ms=8420.25,
        target_latency_ms=10_000.0,
    )

    assert metadata.latency_tier == "fast"
    assert metadata.preview_only is True
    assert metadata.scaffold_only is True
    assert metadata.real_inference is False
    assert metadata.checkpoint is None
    assert metadata.coordinate_frame == "camera_space"
    assert metadata.target_latency_ms == pytest.approx(10_000.0)
    assert metadata.elapsed_ms == pytest.approx(8420.25)


def test_package_frame_preview_builds_deterministic_downstream_payload() -> None:
    mesh = PreviewMesh(
        vertices_camera=[[0.0, 0.0, 2.0], [1.0, 0.0, 2.0], [0.0, 1.0, 2.0]],
        faces=[[0, 1, 2]],
        bbox_xyxy=[10.0, 20.0, 110.0, 220.0],
    )
    metadata = FastTierMetadata(model_family="sat_hmr_candidate", elapsed_ms=3210.0)

    payload = package_frame_preview(
        frame_idx=42,
        t=1.4,
        metadata=metadata,
        players=[
            {
                "player_id": 7,
                "track_id": 7001,
                "confidence": 0.82,
                "mesh": mesh,
            }
        ],
    )

    assert payload == {
        "schema_version": 1,
        "task": "BODY-3",
        "status": "preview",
        "preview_only": True,
        "frame_idx": 42,
        "t": 1.4,
        "metadata": {
            "latency_tier": "fast",
            "model_family": "sat_hmr_candidate",
            "coordinate_frame": "camera_space",
            "target_latency_ms": 10000.0,
            "elapsed_ms": 3210.0,
            "scaffold_only": True,
            "real_inference": False,
            "checkpoint": None,
        },
        "fallback": {"active": False, "reason": None},
        "players": [
            {
                "player_id": 7,
                "track_id": 7001,
                "confidence": 0.82,
                "mesh": {
                    "coordinate_frame": "camera_space",
                    "vertices_camera": [[0.0, 0.0, 2.0], [1.0, 0.0, 2.0], [0.0, 1.0, 2.0]],
                    "faces": [[0, 1, 2]],
                    "bbox_xyxy": [10.0, 20.0, 110.0, 220.0],
                    "vertex_count": 3,
                    "face_count": 1,
                },
            }
        ],
    }


def test_package_frame_preview_rejects_invalid_player_confidence_and_empty_metadata() -> None:
    mesh = PreviewMesh(
        vertices_camera=[[0.0, 0.0, 2.0]],
        faces=[],
        bbox_xyxy=[0.0, 0.0, 10.0, 10.0],
    )

    with pytest.raises(ValueError, match="confidence must be between 0.0 and 1.0"):
        package_frame_preview(
            frame_idx=0,
            t=0.0,
            metadata=FastTierMetadata(model_family="multi_hmr2_candidate"),
            players=[{"player_id": 1, "track_id": 1, "confidence": 1.1, "mesh": mesh}],
        )

    with pytest.raises(ValueError, match="model_family must be non-empty"):
        FastTierMetadata(model_family="")


def test_build_fail_closed_payload_omits_meshes_and_records_reason() -> None:
    payload = build_fail_closed_payload(
        frame_idx=3,
        t=0.1,
        metadata=FastTierMetadata(model_family="scaffold_no_model"),
        reason="no valid preview mesh from fast-tier candidate",
    )

    assert payload["status"] == "fail_closed"
    assert payload["preview_only"] is True
    assert payload["fallback"] == {
        "active": True,
        "reason": "no valid preview mesh from fast-tier candidate",
    }
    assert payload["players"] == []
    assert payload["metadata"]["scaffold_only"] is True
    assert payload["metadata"]["real_inference"] is False

    with pytest.raises(ValueError, match="reason must be non-empty"):
        build_fail_closed_payload(
            frame_idx=3,
            t=0.1,
            metadata=FastTierMetadata(model_family="scaffold_no_model"),
            reason="",
        )

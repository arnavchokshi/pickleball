from __future__ import annotations

import pytest

from threed.racketsport import worldhmr
from threed.racketsport.schemas import (
    CameraIntrinsics,
    CaptureQuality,
    CourtCalibration,
    CourtExtrinsics,
    ReprojectionError,
)


def _identity_calibration() -> CourtCalibration:
    return CourtCalibration(
        schema_version=1,
        sport="pickleball",
        homography=[[100.0, 0.0, 1000.0], [0.0, 100.0, 1000.0], [0.0, 0.0, 1.0]],
        intrinsics=CameraIntrinsics(fx=1000.0, fy=1000.0, cx=960.0, cy=540.0, dist=[], source="manual"),
        extrinsics=CourtExtrinsics(
            R=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            t=[0.0, 0.0, 0.0],
            camera_height_m=1.5,
        ),
        reprojection_error_px=ReprojectionError(median=0.0, p95=0.0),
        capture_quality=CaptureQuality(grade="good", reasons=[]),
        image_pts=[],
        world_pts=[],
    )


def test_snap_player_translation_to_court_projects_root_and_mesh_without_mutating_input() -> None:
    sample = worldhmr.WorldTranslationSample(
        frame_idx=12,
        player_id=7,
        root_xyz=[1.25, -0.50, 0.32],
        mesh_vertices_xyz=[
            [1.00, -0.60, 0.10],
            [1.30, -0.45, 1.95],
        ],
    )

    snapped = worldhmr.snap_player_translation_to_court(sample, court_z_m=0.0)

    assert snapped is not sample
    assert snapped.frame_idx == 12
    assert snapped.player_id == 7
    assert snapped.root_xyz == pytest.approx([1.25, -0.50, 0.0])
    assert len(snapped.mesh_vertices_xyz) == 2
    assert snapped.mesh_vertices_xyz[0] == pytest.approx([1.00, -0.60, -0.22])
    assert snapped.mesh_vertices_xyz[1] == pytest.approx([1.30, -0.45, 1.63])
    assert sample.root_xyz == pytest.approx([1.25, -0.50, 0.32])
    assert sample.mesh_vertices_xyz[0] == pytest.approx([1.00, -0.60, 0.10])
    assert sample.mesh_vertices_xyz[1] == pytest.approx([1.30, -0.45, 1.95])


def test_smooth_world_translations_applies_per_player_temporal_ema() -> None:
    samples = [
        worldhmr.WorldTranslationSample(frame_idx=0, player_id=1, root_xyz=[0.0, 0.0, 0.0]),
        worldhmr.WorldTranslationSample(frame_idx=0, player_id=2, root_xyz=[10.0, 0.0, 0.0]),
        worldhmr.WorldTranslationSample(frame_idx=1, player_id=1, root_xyz=[2.0, 2.0, 0.0]),
        worldhmr.WorldTranslationSample(frame_idx=1, player_id=2, root_xyz=[14.0, 0.0, 0.0]),
        worldhmr.WorldTranslationSample(frame_idx=2, player_id=1, root_xyz=[4.0, 2.0, 0.0]),
    ]

    smoothed = worldhmr.smooth_world_translations(samples, alpha=0.5)

    expected_roots = [
        [0.0, 0.0, 0.0],
        [10.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [12.0, 0.0, 0.0],
        [2.5, 1.5, 0.0],
    ]
    for sample, expected_root in zip(smoothed, expected_roots):
        assert sample.root_xyz == pytest.approx(expected_root)
    assert [sample.player_id for sample in smoothed] == [1, 2, 1, 2, 1]
    assert samples[2].root_xyz == pytest.approx([2.0, 2.0, 0.0])


def test_residual_metrics_report_root_translation_error_and_grounding_z_error() -> None:
    observed = [
        worldhmr.WorldTranslationSample(frame_idx=0, player_id=1, root_xyz=[0.0, 0.0, 0.10]),
        worldhmr.WorldTranslationSample(frame_idx=1, player_id=1, root_xyz=[1.0, 0.0, -0.20]),
    ]
    adjusted = [worldhmr.snap_player_translation_to_court(sample) for sample in observed]

    metrics = worldhmr.residual_metrics(observed, adjusted, court_z_m=0.0)

    assert metrics == worldhmr.WorldGroundingMetrics(
        sample_count=2,
        rms_root_residual_m=pytest.approx((0.025) ** 0.5),
        max_root_residual_m=0.20,
        rms_ground_z_error_m=0.0,
        max_ground_z_error_m=0.0,
        scaffold="cpu_worldhmr_primitives_no_sam3dbody_integration",
    )


def test_build_body_artifacts_marks_floor_contact_from_grounded_joints() -> None:
    samples = [
        {
            "frame_idx": 12,
            "player_id": 1,
            "t": 0.4,
            "confidence": 0.9,
            "track_world_xy": [2.0, 3.0],
            "camera_translation": [0.0, 0.0, 0.0],
            "joints_camera": [
                [0.0, 0.0, 1.1],
                [-0.16, 0.0, 0.0],
                [0.18, 0.0, 0.015],
                [0.0, 0.0, 0.7],
            ],
            "vertices_camera": [],
            "global_orient": [0.0, 0.0, 0.0],
            "body_pose": [0.0, 0.0, 0.0],
            "betas": [0.0],
        }
    ]

    smpl_motion, _skeleton3d, metrics = worldhmr.build_body_artifacts_from_fast_sam(
        samples,
        calibration=_identity_calibration(),
        fps=30.0,
    )

    player = smpl_motion["players"][0]
    frame = player["frames"][0]
    assert frame["foot_contact"] == {"left": True, "right": True}
    assert frame["grf"] == [[0.0, 0.0, 1.0]]
    assert player["skate_free"] is True
    assert player["physics"] == "worldhmr_grounded_floor_contact_heuristic"
    assert metrics["foot_contact_frames"] == 1


def test_world_grounding_helpers_validate_vectors_and_smoothing_alpha() -> None:
    with pytest.raises(ValueError, match="root_xyz must be a 3-vector"):
        worldhmr.WorldTranslationSample(frame_idx=0, player_id=1, root_xyz=[1.0, 2.0])

    with pytest.raises(ValueError, match="mesh_vertices_xyz/0 must be a 3-vector"):
        worldhmr.WorldTranslationSample(
            frame_idx=0,
            player_id=1,
            root_xyz=[1.0, 2.0, 3.0],
            mesh_vertices_xyz=[[1.0, 2.0]],
        )

    with pytest.raises(ValueError, match="alpha must be greater than 0 and less than or equal to 1"):
        worldhmr.smooth_world_translations(
            [worldhmr.WorldTranslationSample(frame_idx=0, player_id=1, root_xyz=[0.0, 0.0, 0.0])],
            alpha=0.0,
        )

    with pytest.raises(ValueError, match="observed and adjusted must have the same length"):
        worldhmr.residual_metrics(
            [worldhmr.WorldTranslationSample(frame_idx=0, player_id=1, root_xyz=[0.0, 0.0, 0.0])],
            [],
        )

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from tests.racketsport.calibration_fixtures import minimal_calibration_image_pts, minimal_calibration_world_pts
from threed.racketsport import hmr_deep, worldhmr
from threed.racketsport.body_postchain import BodyPostChainConfig
from threed.racketsport.body_joint_quality import build_body_joint_quality
from threed.racketsport.schemas import (
    CameraIntrinsics,
    CaptureQuality,
    CourtCalibration,
    CourtExtrinsics,
    ReprojectionError,
)
from threed.racketsport.skeleton_upright import ROTATION_CONVENTION_OFFSET_ROW_TIMES_R


SAM3D_FIXTURE = Path("tests/racketsport/fixtures/sam3d_smpl_motion_237_excerpt_skeleton3d.json")


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
        image_pts=minimal_calibration_image_pts(),
        world_pts=minimal_calibration_world_pts(),
    )


def _camera_y_to_court_z_calibration() -> CourtCalibration:
    calibration = _identity_calibration()
    return calibration.model_copy(
        update={
            "extrinsics": CourtExtrinsics(
                R=[[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
                t=[0.0, 0.0, 0.0],
                camera_height_m=1.5,
            )
        }
    )


def test_sam3d_foot_pixels_override_bbox_anchor_per_safe_axis() -> None:
    sample = {
        "frame_idx": 0,
        "player_id": 3,
        "t": 0.0,
        "confidence": 0.9,
        "track_world_xy": [0.0, 1.5],
        "joints_camera": [[0.0, 0.0, 1.0] for _idx in range(21)],
        "vertices_camera": [],
        "pred_foot_keypoints_2d": [
            {"name": "left_heel", "index": 17, "xy_px": [1020.0, 1250.0], "conf": 0.9},
            {"name": "right_heel", "index": 20, "xy_px": [1040.0, 1250.0], "conf": 0.9},
        ],
    }

    grounded = worldhmr._ground_fast_sam_sample(sample, calibration=_identity_calibration())

    assert grounded["placement_track_world_xy"] == pytest.approx([0.0, 1.5])
    assert grounded["track_world_xy"] == pytest.approx([0.3, 2.5])
    assert grounded["grounding_target_source"] == "sam3d_foot_pixels_xy"
    assert grounded["grounding_target_correction_xy_m"] == pytest.approx([0.3, 1.0])
    assert grounded["transl_world"] == pytest.approx([0.3, 2.5, 0.0])


def test_sam3d_foot_grounding_rejects_legacy_mislabeled_right_toe_and_unsafe_axis() -> None:
    sample = {
        "pred_foot_keypoints_2d": [
            {"name": "left_heel", "index": 17, "xy_px": [1020.0, 1200.0], "conf": 0.9},
            {"name": "right_heel", "index": 20, "xy_px": [1040.0, 1200.0], "conf": 0.9},
            {"name": "right_toe", "index": 16, "xy_px": [1900.0, 1200.0], "conf": 0.9},
        ]
    }

    target = worldhmr._sam3d_foot_grounding_target(
        sample,
        calibration=_identity_calibration(),
        placement_track_world_xy=[0.0, 1.5],
        camera_motion=None,
    )

    assert target["world_xy"] == pytest.approx([0.3, 2.0])
    assert target["candidate_count"] == 2


def test_persisted_skeleton_reanchor_translates_existing_joints_without_filling_gaps() -> None:
    skeleton = {
        "schema_version": 1,
        "artifact_type": "racketsport_skeleton3d",
        "fps": 30.0,
        "players": [
            {
                "id": 3,
                "frames": [
                    {
                        "frame_idx": 0,
                        "t": 0.0,
                        "transl_world": [0.0, 1.5, 0.0],
                        "joints_world": [[0.0, 1.5, 0.0], [0.0, 1.5, 1.0]],
                    }
                ],
            }
        ],
        "provenance": {},
    }
    sidecar = {
        "players": [
            {
                "id": 3,
                "frames": [
                    {
                        "frame_idx": 0,
                        "keypoints": [
                            {"name": "left_heel", "index": 17, "xy_px": [1020.0, 1250.0]},
                            {"name": "right_heel", "index": 20, "xy_px": [1040.0, 1250.0]},
                        ],
                    }
                ],
            }
        ]
    }
    tracks = {
        "fps": 30.0,
        "players": [{"id": 3, "frames": [{"frame_idx": 0, "t": 0.0, "world_xy": [0.0, 1.5]}]}],
    }

    refined, report = worldhmr.reanchor_skeleton3d_to_sam3d_foot_pixels(
        skeleton,
        sam3d_keypoints_2d=sidecar,
        anchor_tracks=tracks,
        calibration=_identity_calibration(),
    )

    frame = refined["players"][0]["frames"][0]
    assert frame["transl_world"] == pytest.approx([0.3, 2.5, 0.0])
    assert frame["joints_world"][1] == pytest.approx([0.3, 2.5, 1.0])
    assert frame["confidence_provenance"]["posthoc_translation_only"] is True
    assert report["aligned_frame_count"] == 1
    assert report["zone_correction_count"] == 1
    assert report["creates_missing_skeletons"] is False
    assert skeleton["players"][0]["frames"][0]["transl_world"] == [0.0, 1.5, 0.0]


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

    smpl_motion, skeleton3d, metrics = worldhmr.build_body_artifacts_from_fast_sam(
        samples,
        calibration=_identity_calibration(),
        fps=30.0,
    )

    player = smpl_motion["players"][0]
    frame = player["frames"][0]
    skeleton_frame = skeleton3d["players"][0]["frames"][0]
    assert smpl_motion["model"] == "sam3dbody_world_joints"
    assert frame["frame_idx"] == 12
    assert skeleton_frame["frame_idx"] == 12
    assert skeleton_frame["transl_world"] == pytest.approx(frame["transl_world"])
    assert frame["foot_contact"] == {"left": True, "right": True}
    assert frame["foot_lock"] == {"left": True, "right": True}
    assert frame["grf"] is None
    assert player["foot_lock"]["contact_frames"] == 1
    assert player["foot_lock"]["contact_samples"] == 2
    assert player["skate_free"] is False
    assert player["physics"] == "worldhmr_floor_contact_footlock_z_snap"
    assert metrics["foot_contact_frames"] == 1
    assert metrics["foot_lock_contact_frames"] == 1
    assert metrics["foot_lock_contact_samples"] == 2
    assert metrics["grf_frames"] == 0
    assert metrics["skate_free_players"] == 0


def test_sam3d_temporal_refine_gate_executes_on_real_fixture_before_serialization() -> None:
    skeleton = json.loads(SAM3D_FIXTURE.read_text(encoding="utf-8"))

    refined = worldhmr.apply_sam3d_temporal_refine_gate(skeleton, fps=30.0)

    gate = refined["provenance"]["sam3d_temporal_refine"]
    temporal = refined["provenance"]["temporal_refine"]
    assert gate["status"] == "applied"
    assert gate["wrist_peak_timing_gate_pass"] is True
    assert temporal["wrist_peak_timing"]["status"] == "pass"
    assert all(len(frame["joints_world"]) == 70 for player in refined["players"] for frame in player["frames"])


def test_build_body_artifacts_runs_sam3d_temporal_refine_gate_in_pipeline_path(monkeypatch) -> None:  # noqa: ANN001
    calls: list[dict[str, object]] = []

    def fake_refine(skeleton3d, *, fps=None, **_kwargs):  # noqa: ANN001
        calls.append({"source_model": skeleton3d["source_model"], "fps": fps})
        refined = copy.deepcopy(skeleton3d)
        refined["players"][0]["frames"][0]["pipeline_gate_marker"] = "refined"
        refined["provenance"]["temporal_refine"] = {
            "source": "sam3d_body_joints",
            "wrist_peak_timing": {"status": "pass"},
            "wrist_peak_timing_gate_pass": True,
        }
        return refined

    monkeypatch.setattr(worldhmr, "refine_sam3d_skeleton3d", fake_refine)
    joints = [[0.0, 0.0, 1.0] for _idx in range(70)]
    joints[0] = [0.0, 0.0, 0.0]
    samples = [
        {
            "frame_idx": 0,
            "player_id": 1,
            "t": 0.0,
            "confidence": 0.9,
            "track_world_xy": [2.0, 3.0],
            "camera_translation": [0.0, 0.0, 0.0],
            "joints_camera": joints,
            "vertices_camera": [],
            "global_orient": [0.0, 0.0, 0.0],
            "body_pose": [0.0, 0.0, 0.0],
            "betas": [0.0],
        }
    ]

    _smpl_motion, skeleton3d, metrics = worldhmr.build_body_artifacts_from_fast_sam(
        samples,
        calibration=_identity_calibration(),
        fps=30.0,
    )

    assert calls == [{"source_model": "sam3d_body_joints", "fps": 30.0}]
    assert skeleton3d["players"][0]["frames"][0]["pipeline_gate_marker"] == "refined"
    assert skeleton3d["provenance"]["sam3d_temporal_refine"]["status"] == "applied"
    assert metrics["sam3d_temporal_refine_status"] == "applied"


def _install_pass_through_sam3d_refine(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_refine(skeleton3d, *, fps=None, **_kwargs):  # noqa: ANN001, ARG001
        refined = copy.deepcopy(skeleton3d)
        refined["provenance"]["temporal_refine"] = {
            "source": "sam3d_body_joints",
            "wrist_peak_timing": {"status": "pass"},
            "wrist_peak_timing_gate_pass": True,
        }
        return refined

    monkeypatch.setattr(worldhmr, "refine_sam3d_skeleton3d", fake_refine)


def test_body_postchain_temporal_smoothing_knob_keeps_raw_grounded_translation() -> None:
    samples = [
        {
            "frame_idx": 0,
            "player_id": 1,
            "t": 0.0,
            "confidence": 0.9,
            "track_world_xy": [0.0, 0.0],
            "camera_translation": [0.0, 0.0, 0.0],
            "joints_camera": [[0.0, 0.0, 1.0]],
            "vertices_camera": [],
            "global_orient": [0.0, 0.0, 0.0],
            "body_pose": [0.0, 0.0, 0.0],
            "betas": [0.0],
        },
        {
            "frame_idx": 1,
            "player_id": 1,
            "t": 1.0 / 30.0,
            "confidence": 0.9,
            "track_world_xy": [10.0, 0.0],
            "camera_translation": [0.0, 0.0, 0.0],
            "joints_camera": [[0.0, 0.0, 1.0]],
            "vertices_camera": [],
            "global_orient": [0.0, 0.0, 0.0],
            "body_pose": [0.0, 0.0, 0.0],
            "betas": [0.0],
        },
    ]

    default_smpl, _default_skeleton, default_metrics = worldhmr.build_body_artifacts_from_fast_sam(
        copy.deepcopy(samples),
        calibration=_identity_calibration(),
        fps=30.0,
        smoothing_alpha=0.5,
    )
    raw_smpl, raw_skeleton, raw_metrics = worldhmr.build_body_artifacts_from_fast_sam(
        copy.deepcopy(samples),
        calibration=_identity_calibration(),
        fps=30.0,
        smoothing_alpha=0.5,
        body_postchain=BodyPostChainConfig(temporal_smoothing=False),
    )

    assert default_smpl["players"][0]["frames"][1]["transl_world"] == pytest.approx([5.0, 0.0, 0.0])
    assert raw_smpl["players"][0]["frames"][1]["transl_world"] == pytest.approx([10.0, 0.0, 0.0])
    assert raw_metrics["postchain_bypassed_stages"] == ["temporal_smoothing"]
    assert raw_skeleton["provenance"]["body_postchain_bypass"]["stages"] == ["temporal_smoothing"]
    assert "postchain_bypassed_stages" not in default_metrics


def test_body_postchain_raw_mode_persists_schema_valid_raw_grounded_joints() -> None:
    samples = [
        {
            "frame_idx": 0,
            "player_id": 1,
            "t": 0.0,
            "confidence": 0.9,
            "track_world_xy": [0.0, 0.0],
            "camera_translation": [0.0, 0.0, 0.0],
            "joints_camera": [[0.0, 0.0, 1.0]],
            "vertices_camera": [],
            "global_orient": [0.0, 0.0, 0.0],
            "body_pose": [0.0, 0.0, 0.0],
            "betas": [0.0],
        },
        {
            "frame_idx": 1,
            "player_id": 1,
            "t": 1.0 / 30.0,
            "confidence": 0.9,
            "track_world_xy": [10.0, 0.0],
            "camera_translation": [0.0, 0.0, 0.0],
            "joints_camera": [[0.0, 0.0, 1.0]],
            "vertices_camera": [],
            "global_orient": [0.0, 0.0, 0.0],
            "body_pose": [0.0, 0.0, 0.0],
            "betas": [0.0],
        },
    ]

    computed = worldhmr.compute_body_skeleton_and_metrics(
        samples,
        calibration=_identity_calibration(),
        fps=30.0,
        smoothing_alpha=0.5,
        body_postchain=BodyPostChainConfig.raw(),
    )

    assert computed.raw_grounded_joints is not None
    assert computed.raw_grounded_joints["schema_version"] == 1
    assert computed.raw_grounded_joints["artifact_type"] == "racketsport_body_raw_grounded_joints"
    assert computed.raw_grounded_joints["postchain_bypassed_stages"] == [
        "temporal_smoothing",
        "foot_lock",
        "foot_pin",
        "contact_splice",
        "wrist_lock",
        "world_joint_visual_smoothing",
    ]
    assert computed.raw_grounded_joints["players"][0]["frames"][1]["joints_world"][0] == pytest.approx([10.0, 0.0, 0.0])
    assert computed.smpl_motion_view["players"][0]["frames"][1]["joints_world"][0] == pytest.approx([10.0, 0.0, 0.0])
    assert computed.metrics["raw_grounded_joints_sidecar"] == "body_raw_grounded_joints.json"


def test_body_postchain_foot_lock_knob_bypasses_foot_lock_stage() -> None:
    samples = [
        {
            "frame_idx": 0,
            "player_id": 1,
            "t": 0.0,
            "confidence": 0.9,
            "track_world_xy": [1.0, 2.0],
            "camera_translation": [0.0, 0.0, 0.0],
            "joints_camera": [[0.0, 0.0, 1.2], [0.0, 0.0, 0.0]],
            "vertices_camera": [[1.0, 0.0, 0.0]],
            "global_orient": [0.0, 0.0, 0.0],
            "body_pose": [0.0, 0.0, 0.0],
            "betas": [0.0],
        },
        {
            "frame_idx": 1,
            "player_id": 1,
            "t": 1.0 / 30.0,
            "confidence": 0.9,
            "track_world_xy": [1.5, 2.0],
            "camera_translation": [0.0, 0.0, 0.0],
            "joints_camera": [[0.0, 0.0, 1.2], [0.0, 0.0, 0.0]],
            "vertices_camera": [[1.0, 0.0, 0.0]],
            "global_orient": [0.0, 0.0, 0.0],
            "body_pose": [0.0, 0.0, 0.0],
            "betas": [0.0],
        },
    ]

    smpl_motion, _skeleton3d, metrics = worldhmr.build_body_artifacts_from_fast_sam(
        samples,
        calibration=_identity_calibration(),
        fps=30.0,
        smoothing_alpha=1.0,
        body_postchain=BodyPostChainConfig(foot_lock=False),
    )

    first, second = smpl_motion["players"][0]["frames"]
    assert first["foot_lock"] == {"left": False, "right": False}
    assert second["foot_lock"] == {"left": False, "right": False}
    assert second["joints_world"][1] == pytest.approx([1.5, 2.0, 0.0])
    assert metrics["foot_lock_contact_frames"] == 0
    assert metrics["postchain_bypassed_stages"] == ["foot_lock"]


def test_body_postchain_foot_pin_knob_bypasses_refined_stance_pin(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_pass_through_sam3d_refine(monkeypatch)
    calls: list[str] = []

    def fail_if_called(*_args, **_kwargs):  # noqa: ANN001
        calls.append("called")
        raise AssertionError("foot pin stage should be bypassed")

    monkeypatch.setattr(worldhmr, "_apply_refined_stance_phase_lock_and_pin", fail_if_called)
    samples = [
        {
            "frame_idx": 0,
            "player_id": 1,
            "t": 0.0,
            "confidence": 0.9,
            "track_world_xy": [0.0, 0.0],
            "camera_translation": [0.0, 0.0, 0.0],
            "joints_camera": [[0.0, 0.0, 1.2], [0.0, 0.0, 0.0]],
            "vertices_camera": [],
            "global_orient": [0.0, 0.0, 0.0],
            "body_pose": [0.0, 0.0, 0.0],
            "betas": [0.0],
        }
    ]

    _smpl_motion, skeleton3d, metrics = worldhmr.build_body_artifacts_from_fast_sam(
        samples,
        calibration=_identity_calibration(),
        fps=30.0,
        stance_index={(1, 0): {"stance": True}},
        grounding_anchor_source="placement_track_world_xy",
        body_postchain=BodyPostChainConfig(foot_pin=False),
    )

    assert calls == []
    assert metrics["postchain_bypassed_stages"] == ["foot_pin"]
    assert skeleton3d["provenance"]["body_postchain_bypass"]["stages"] == ["foot_pin"]


def test_body_postchain_wrist_lock_knob_bypasses_lock_stage(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_pass_through_sam3d_refine(monkeypatch)
    calls: list[str] = []

    def fail_if_called(payload, **_kwargs):  # noqa: ANN001
        calls.append("called")
        return copy.deepcopy(dict(payload))

    monkeypatch.setattr(worldhmr, "apply_sam3d_wrist_bone_lock", fail_if_called)
    samples = [
        {
            "frame_idx": 0,
            "player_id": 1,
            "t": 0.0,
            "confidence": 0.9,
            "track_world_xy": [0.0, 0.0],
            "camera_translation": [0.0, 0.0, 0.0],
            "joints_camera": [[0.0, 0.0, 1.0] for _idx in range(70)],
            "vertices_camera": [],
            "global_orient": [0.0, 0.0, 0.0],
            "body_pose": [0.0, 0.0, 0.0],
            "betas": [0.0],
        }
    ]

    _smpl_motion, _skeleton3d, metrics = worldhmr.build_body_artifacts_from_fast_sam(
        samples,
        calibration=_identity_calibration(),
        fps=30.0,
        body_postchain=BodyPostChainConfig(wrist_lock=False),
    )

    assert calls == []
    assert metrics["sam3d_wrist_bone_lock_status"] == "disabled"
    assert metrics["postchain_bypassed_stages"] == ["wrist_lock"]


def _camera_motion_payload(matrix: list[list[float]], *, compensated: bool = True) -> dict[str, object]:
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_camera_motion",
        "fps": 30.0,
        "frames": [
            {
                "frame_idx": 0,
                "compensated": bool(compensated),
                "model": "homography" if compensated else "identity",
                "reason": None if compensated else "unit_test_uncompensated",
                "M": matrix if compensated else [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            }
        ],
    }


def _sam3d_foot_pixel_sample() -> dict[str, object]:
    joints = [[0.0, 0.0, 1.0] for _idx in range(70)]
    joints[13] = [-0.5, 0.0, 0.0]
    joints[14] = [0.5, 0.0, 0.0]
    return {
        "frame_idx": 0,
        "player_id": 1,
        "t": 0.0,
        "confidence": 0.95,
        "track_world_xy": [0.0, 0.0],
        "bbox_xyxy": [940.0, 800.0, 1060.0, 1000.0],
        "joints_camera": joints,
        "vertices_camera": [],
        "pred_foot_keypoints_2d": [
            {"name": "left_ankle", "index": 13, "xy_px": [950.0, 1000.0], "conf": 0.95},
            {"name": "right_ankle", "index": 14, "xy_px": [1050.0, 1000.0], "conf": 0.95},
        ],
        "global_orient": [0.0, 0.0, 0.0],
        "body_pose": [0.0, 0.0, 0.0],
        "betas": [0.0],
    }


def test_camera_motion_warps_sam3d_foot_pixels_before_world_grounding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_pass_through_sam3d_refine(monkeypatch)
    samples = [_sam3d_foot_pixel_sample()]
    camera_motion_path = tmp_path / "camera_motion.json"
    camera_motion_path.write_text(
        json.dumps(_camera_motion_payload([[1.2, 0.0, -180.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])),
        encoding="utf-8",
    )

    static_smpl, _static_skeleton, static_metrics = worldhmr.build_body_artifacts_from_fast_sam(
        copy.deepcopy(samples),
        calibration=_identity_calibration(),
        fps=30.0,
        smoothing_alpha=1.0,
        camera_motion_path=None,
    )
    motion_smpl, motion_skeleton, motion_metrics = worldhmr.build_body_artifacts_from_fast_sam(
        copy.deepcopy(samples),
        calibration=_identity_calibration(),
        fps=30.0,
        smoothing_alpha=1.0,
        camera_motion_path=camera_motion_path,
    )

    static_joints = static_smpl["players"][0]["frames"][0]["joints_world"]
    motion_joints = motion_smpl["players"][0]["frames"][0]["joints_world"]
    static_foot_span_m = static_joints[14][0] - static_joints[13][0]
    motion_foot_span_m = motion_joints[14][0] - motion_joints[13][0]

    assert static_foot_span_m == pytest.approx(1.0)
    assert motion_foot_span_m == pytest.approx(1.2)
    assert motion_joints[13][0] == pytest.approx(-0.4)
    assert motion_joints[14][0] == pytest.approx(0.8)
    assert "camera_motion_frames_used" not in static_metrics
    assert motion_metrics["camera_motion_frames_used"] == 1
    assert motion_metrics["camera_motion_frames_uncompensated"] == 0
    assert motion_metrics["grounding_target_source_counts"] == {"sam3d_foot_pixels_xy": 1}
    assert motion_skeleton["provenance"]["camera_motion"]["frames_used"] == 1
    assert motion_skeleton["provenance"]["camera_motion"]["frames_uncompensated"] == 0


def test_uncompensated_camera_motion_frame_uses_static_world_grounding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_pass_through_sam3d_refine(monkeypatch)
    samples = [_sam3d_foot_pixel_sample()]
    camera_motion_path = tmp_path / "camera_motion.json"
    camera_motion_path.write_text(
        json.dumps(_camera_motion_payload([[1.2, 0.0, -180.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], compensated=False)),
        encoding="utf-8",
    )

    static_smpl, _static_skeleton, _static_metrics = worldhmr.build_body_artifacts_from_fast_sam(
        copy.deepcopy(samples),
        calibration=_identity_calibration(),
        fps=30.0,
        smoothing_alpha=1.0,
        camera_motion_path=None,
    )
    motion_smpl, motion_skeleton, motion_metrics = worldhmr.build_body_artifacts_from_fast_sam(
        copy.deepcopy(samples),
        calibration=_identity_calibration(),
        fps=30.0,
        smoothing_alpha=1.0,
        camera_motion_path=camera_motion_path,
    )

    assert motion_smpl == static_smpl
    assert motion_metrics["camera_motion_frames_used"] == 0
    assert motion_metrics["camera_motion_frames_uncompensated"] == 1
    assert motion_skeleton["provenance"]["camera_motion"]["frames_used"] == 0
    assert motion_skeleton["provenance"]["camera_motion"]["frames_uncompensated"] == 1


def test_omitted_camera_motion_keeps_worldhmr_outputs_byte_identical(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_pass_through_sam3d_refine(monkeypatch)
    samples = [_sam3d_foot_pixel_sample()]

    first = worldhmr.build_body_artifacts_from_fast_sam(
        copy.deepcopy(samples),
        calibration=_identity_calibration(),
        fps=30.0,
        smoothing_alpha=1.0,
    )
    second = worldhmr.build_body_artifacts_from_fast_sam(
        copy.deepcopy(samples),
        calibration=_identity_calibration(),
        fps=30.0,
        smoothing_alpha=1.0,
        camera_motion_path=None,
    )

    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_malformed_camera_motion_is_ignored_with_warning_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_pass_through_sam3d_refine(monkeypatch)
    samples = [_sam3d_foot_pixel_sample()]
    camera_motion_path = tmp_path / "camera_motion.json"
    camera_motion_path.write_text("{not-json", encoding="utf-8")

    static_smpl, _static_skeleton, _static_metrics = worldhmr.build_body_artifacts_from_fast_sam(
        copy.deepcopy(samples),
        calibration=_identity_calibration(),
        fps=30.0,
        smoothing_alpha=1.0,
        camera_motion_path=None,
    )
    motion_smpl, motion_skeleton, motion_metrics = worldhmr.build_body_artifacts_from_fast_sam(
        copy.deepcopy(samples),
        calibration=_identity_calibration(),
        fps=30.0,
        smoothing_alpha=1.0,
        camera_motion_path=camera_motion_path,
    )

    assert motion_smpl == static_smpl
    assert motion_metrics["camera_motion_status"] == "ignored_malformed"
    assert motion_metrics["camera_motion_warnings"]
    assert motion_skeleton["provenance"]["camera_motion"]["status"] == "ignored_malformed"


def test_build_body_artifacts_rotates_camera_frame_offsets_upright_and_regrounds_feet() -> None:
    samples = [
        {
            "frame_idx": 0,
            "player_id": 1,
            "t": 0.0,
            "confidence": 0.93,
            "track_world_xy": [2.0, 3.0],
            "camera_translation": [10.0, 20.0, -5.0],
            "joints_camera": [
                [0.0, 0.0, 0.0],  # support foot in camera frame
                [0.0, 1.6, 0.0],  # body long axis is camera-Y; must become court-Z
                [0.4, 0.8, 0.0],
            ],
            "vertices_camera": [],
            "global_orient": [0.0, 0.0, 0.0],
            "body_pose": [0.0, 0.0, 0.0],
            "betas": [0.0],
        }
    ]

    smpl_motion, skeleton3d, metrics = worldhmr.build_body_artifacts_from_fast_sam(
        samples,
        calibration=_camera_y_to_court_z_calibration(),
        fps=30.0,
        smoothing_alpha=1.0,
    )

    frame = smpl_motion["players"][0]["frames"][0]
    assert frame["joints_world"][0] == pytest.approx([2.0, 3.0, 0.0])
    assert frame["joints_world"][1] == pytest.approx([2.0, 3.0, 1.6])
    assert frame["transl_world"] == pytest.approx([2.0, 3.0, 0.0])
    assert metrics["grounding"] == "camera_offset_row_times_R_plus_track_footpoint_court_z0"
    assert metrics["camera_offset_rotation_convention"] == ROTATION_CONVENTION_OFFSET_ROW_TIMES_R
    skeleton_frame = skeleton3d["players"][0]["frames"][0]
    assert skeleton3d["source_model"] == "sam3d_body_joints"
    assert skeleton3d["provenance"]["source"] == "sam3d_body_joints"
    assert skeleton_frame["confidence_provenance"]["source"] == "sam3d_body_joints"


def test_build_body_artifacts_preserves_static_mesh_faces_for_body_mesh_export() -> None:
    samples = [
        {
            "frame_idx": 12,
            "player_id": 1,
            "t": 0.4,
            "confidence": 0.9,
            "track_world_xy": [2.0, 3.0],
            "camera_translation": [0.0, 0.0, 0.0],
            "joints_camera": [[0.0, 0.0, 1.1]],
            "vertices_camera": [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.2, 0.1, 1.7]],
            "mesh_faces": [[0, 1, 2]],
            "global_orient": [0.0, 0.0, 0.0],
            "body_pose": [0.0, 0.0, 0.0],
            "betas": [0.0],
        }
    ]

    smpl_motion, _skeleton3d, _metrics = worldhmr.build_body_artifacts_from_fast_sam(
        samples,
        calibration=_identity_calibration(),
        fps=30.0,
    )

    assert smpl_motion["mesh_faces"] == [[0, 1, 2]]


def test_build_body_artifacts_interns_normalized_topology_through_grounding_and_common_faces() -> None:
    first_faces = hmr_deep._face_list(
        [[0, 1, 2], [0, 2, 3]],
        vertex_count=4,
        name="mesh_faces",
        vertices_name="pred_vertices",
    )
    second_faces = hmr_deep._face_list(
        [[0, 1, 2], [0, 2, 3]],
        vertex_count=4,
        name="mesh_faces",
        vertices_name="pred_vertices",
    )
    assert first_faces is not second_faces

    base = {
        "player_id": 1,
        "confidence": 0.9,
        "track_world_xy": [2.0, 3.0],
        "camera_translation": [0.0, 0.0, 0.0],
        "joints_camera": [[0.0, 0.0, 1.1]],
        "vertices_camera": [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.2, 0.1, 1.7], [0.1, 0.3, 0.8]],
        "global_orient": [0.0, 0.0, 0.0],
        "body_pose": [0.0, 0.0, 0.0],
        "betas": [0.0],
    }
    samples = [
        {**base, "frame_idx": 12, "t": 0.4, "mesh_faces": first_faces},
        {**base, "frame_idx": 13, "t": 13.0 / 30.0, "mesh_faces": second_faces},
    ]

    smpl_motion, _skeleton3d, _metrics = worldhmr.build_body_artifacts_from_fast_sam(
        samples,
        calibration=_identity_calibration(),
        fps=30.0,
    )

    assert smpl_motion["mesh_faces"] is first_faces


def test_build_body_artifacts_keeps_inconsistent_topology_error_with_interning() -> None:
    base = {
        "player_id": 1,
        "confidence": 0.9,
        "track_world_xy": [2.0, 3.0],
        "camera_translation": [0.0, 0.0, 0.0],
        "joints_camera": [[0.0, 0.0, 1.1]],
        "vertices_camera": [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.2, 0.1, 1.7], [0.1, 0.3, 0.8]],
        "global_orient": [0.0, 0.0, 0.0],
        "body_pose": [0.0, 0.0, 0.0],
        "betas": [0.0],
    }
    samples = [
        {**base, "frame_idx": 12, "t": 0.4, "mesh_faces": [[0, 1, 2], [0, 2, 3]]},
        {**base, "frame_idx": 13, "t": 13.0 / 30.0, "mesh_faces": [[0, 1, 2], [1, 2, 3]]},
    ]

    with pytest.raises(ValueError, match="Fast SAM-3D-Body samples produced inconsistent mesh_faces"):
        worldhmr.build_body_artifacts_from_fast_sam(
            samples,
            calibration=_identity_calibration(),
            fps=30.0,
        )


def test_numpy_bulk_body_inputs_preserve_list_contract_and_quantized_world_mesh() -> None:
    vertices = np.asarray(
        [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.2, 0.1, 1.7], [0.1, 0.3, 0.8]],
        dtype=np.float32,
    )
    joints = np.asarray([[0.0, 0.0, 1.1], [0.2, 0.0, 0.0]], dtype=np.float32)
    faces = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    sample = {
        "frame_idx": 12,
        "player_id": 1,
        "t": 0.4,
        "confidence": 0.9,
        "track_world_xy": [2.0, 3.0],
        "camera_translation": [0.0, 0.0, 0.0],
        "joints_camera": joints,
        "vertices_camera": vertices,
        "mesh_faces": faces,
        "global_orient": [0.0, 0.0, 0.0],
        "body_pose": [0.0, 0.0, 0.0],
        "betas": [0.0],
    }

    array_result = worldhmr.build_body_artifacts_from_fast_sam(
        [sample],
        calibration=_camera_y_to_court_z_calibration(),
        fps=30.0,
    )
    list_result = worldhmr.build_body_artifacts_from_fast_sam(
        [{**sample, "joints_camera": joints.tolist(), "vertices_camera": vertices.tolist(), "mesh_faces": faces.tolist()}],
        calibration=_camera_y_to_court_z_calibration(),
        fps=30.0,
    )

    assert array_result == list_result
    mesh = array_result[0]["players"][0]["frames"][0]["mesh_vertices_world"]
    quantized = np.rint(np.asarray(mesh, dtype=np.float64) * 1000.0).astype(np.int16)
    expected = np.rint(
        np.asarray(list_result[0]["players"][0]["frames"][0]["mesh_vertices_world"], dtype=np.float64) * 1000.0
    ).astype(np.int16)
    assert np.array_equal(quantized, expected)


def test_mesh_face_fast_path_preserves_bool_and_malformed_row_errors() -> None:
    with pytest.raises(ValueError, match="mesh_faces/0 must be a triangle index triple"):
        worldhmr._mesh_faces([[True, 1, 2]], vertex_count=3)
    with pytest.raises(ValueError, match="mesh_faces/0 must be a triangle index triple"):
        worldhmr._mesh_faces([[], []], vertex_count=3)


def test_camera_to_world_preserves_legacy_half_millimetre_quantization_boundary() -> None:
    point = [-1.8553603646411765, 0.34439941221241344, 0.671964913302858]
    rotation = [
        [-0.9335697082225718, 0.29882721259079315, 0.19786332885211252],
        [0.007597074458510103, -0.5354562071719233, 0.8445288240555974],
        [0.35831534205067767, 0.7899297103010338, 0.49761548251117615],
    ]
    calibration = _identity_calibration().model_copy(
        update={
            "extrinsics": CourtExtrinsics(
                R=rotation,
                t=[0.0, 0.0, 0.0],
                camera_height_m=1.5,
            )
        }
    )
    legacy = [sum(float(point[row]) * float(rotation[row][column]) for row in range(3)) for column in range(3)]

    actual = worldhmr._camera_offsets_to_world([point], calibration=calibration, root_camera=[0.0, 0.0, 0.0])

    assert actual == [legacy]
    assert np.array_equal(
        np.rint(np.asarray(actual, dtype=np.float64) * 1000.0).astype(np.int16),
        np.rint(np.asarray([legacy], dtype=np.float64) * 1000.0).astype(np.int16),
    )


def test_build_body_artifacts_wires_footlock_z_snap_and_metrics() -> None:
    samples = [
        {
            "frame_idx": 0,
            "player_id": 1,
            "t": 0.0,
            "confidence": 0.9,
            "track_world_xy": [1.0, 2.0],
            "camera_translation": [0.0, 0.0, 0.0],
            "joints_camera": [
                [0.0, 0.0, 1.2],
                [-0.20, 0.0, 0.02],
                [0.20, 0.0, 0.03],
            ],
            "vertices_camera": [[0.0, 0.0, 0.0]],
            "global_orient": [0.0, 0.0, 0.0],
            "body_pose": [0.0, 0.0, 0.0],
            "betas": [0.0],
        },
        {
            "frame_idx": 1,
            "player_id": 1,
            "t": 1.0 / 30.0,
            "confidence": 0.9,
            "track_world_xy": [1.001, 2.0],
            "camera_translation": [0.0, 0.0, 0.0],
            "joints_camera": [
                [0.0, 0.0, 1.2],
                [-0.20, 0.0, 0.02],
                [0.20, 0.0, 0.03],
            ],
            "vertices_camera": [[0.0, 0.0, 0.0]],
            "global_orient": [0.0, 0.0, 0.0],
            "body_pose": [0.0, 0.0, 0.0],
            "betas": [0.0],
        },
    ]

    smpl_motion, _skeleton3d, metrics = worldhmr.build_body_artifacts_from_fast_sam(
        samples,
        calibration=_identity_calibration(),
        fps=30.0,
        smoothing_alpha=1.0,
    )

    player = smpl_motion["players"][0]
    first, second = player["frames"]
    assert first["joints_world"][1][2] == pytest.approx(0.0)
    assert first["joints_world"][2][2] == pytest.approx(0.0)
    assert second["joints_world"][1][2] == pytest.approx(0.0)
    assert second["joints_world"][2][2] == pytest.approx(0.0)
    assert first["foot_lock"] == {"left": True, "right": True}
    assert second["foot_lock"] == {"left": True, "right": True}
    assert player["foot_lock"]["scaffold"] == "cpu_foot_lock_primitives_no_smpl_ik"
    assert player["foot_lock"]["contact_frames"] == 2
    assert player["foot_lock"]["contact_samples"] == 4
    assert player["foot_lock"]["max_slide_m"] == pytest.approx(0.0)
    assert player["foot_lock"]["max_penetration_m"] == pytest.approx(0.0)
    assert player["skate_free"] is True
    assert player["physics"] == "worldhmr_floor_contact_footlock_z_snap"
    assert metrics["foot_lock_contact_frames"] == 2
    assert metrics["foot_lock_contact_samples"] == 4
    assert metrics["max_foot_lock_slide_m"] == pytest.approx(0.0)
    assert metrics["max_foot_lock_penetration_m"] == pytest.approx(0.0)
    assert metrics["foot_lock_skate_free_players"] == 1


def test_footlock_pins_continuous_contact_by_shifting_body_together() -> None:
    samples = [
        {
            "frame_idx": 0,
            "player_id": 1,
            "t": 0.0,
            "confidence": 0.9,
            "track_world_xy": [1.0, 2.0],
            "camera_translation": [0.0, 0.0, 0.0],
            "joints_camera": [
                [0.0, 0.0, 1.2],
                [0.0, 0.0, 0.0],
            ],
            "vertices_camera": [[1.0, 0.0, 0.0]],
            "global_orient": [0.0, 0.0, 0.0],
            "body_pose": [0.0, 0.0, 0.0],
            "betas": [0.0],
        },
        {
            "frame_idx": 1,
            "player_id": 1,
            "t": 1.0 / 30.0,
            "confidence": 0.9,
            "track_world_xy": [1.5, 2.0],
            "camera_translation": [0.0, 0.0, 0.0],
            "joints_camera": [
                [0.0, 0.0, 1.2],
                [0.0, 0.0, 0.0],
            ],
            "vertices_camera": [[1.0, 0.0, 0.0]],
            "global_orient": [0.0, 0.0, 0.0],
            "body_pose": [0.0, 0.0, 0.0],
            "betas": [0.0],
        },
    ]

    smpl_motion, _skeleton3d, metrics = worldhmr.build_body_artifacts_from_fast_sam(
        samples,
        calibration=_identity_calibration(),
        fps=30.0,
        smoothing_alpha=1.0,
    )

    first, second = smpl_motion["players"][0]["frames"]
    assert first["joints_world"][1] == pytest.approx([1.0, 2.0, 0.0])
    assert second["joints_world"][1] == pytest.approx(first["joints_world"][1])
    assert second["transl_world"] == pytest.approx(first["transl_world"])
    assert second["mesh_vertices_world"][0] == pytest.approx(first["mesh_vertices_world"][0])
    assert metrics["max_foot_lock_slide_m"] <= 0.003
    assert smpl_motion["players"][0]["skate_free"] is True


def test_footlock_pins_multiple_contact_joints_when_pose_jitters() -> None:
    samples = [
        {
            "frame_idx": 0,
            "player_id": 1,
            "t": 0.0,
            "confidence": 0.9,
            "track_world_xy": [1.0, 2.0],
            "camera_translation": [0.0, 0.0, 0.0],
            "joints_camera": [
                [0.0, 0.0, 1.2],
                [0.0, 0.0, 0.0],
                [0.4, 0.0, 0.0],
            ],
            "vertices_camera": [[0.2, 0.0, 0.0]],
            "global_orient": [0.0, 0.0, 0.0],
            "body_pose": [0.0, 0.0, 0.0],
            "betas": [0.0],
        },
        {
            "frame_idx": 1,
            "player_id": 1,
            "t": 1.0 / 30.0,
            "confidence": 0.9,
            "track_world_xy": [1.0, 2.0],
            "camera_translation": [0.0, 0.0, 0.0],
            "joints_camera": [
                [0.0, 0.0, 1.2],
                [0.1, 0.0, 0.0],
                [0.3, 0.0, 0.0],
            ],
            "vertices_camera": [[0.2, 0.0, 0.0]],
            "global_orient": [0.0, 0.0, 0.0],
            "body_pose": [0.0, 0.0, 0.0],
            "betas": [0.0],
        },
    ]

    smpl_motion, _skeleton3d, metrics = worldhmr.build_body_artifacts_from_fast_sam(
        samples,
        calibration=_identity_calibration(),
        fps=30.0,
        smoothing_alpha=1.0,
    )

    first, second = smpl_motion["players"][0]["frames"]
    assert second["joints_world"][1] == pytest.approx(first["joints_world"][1])
    assert second["joints_world"][2] == pytest.approx(first["joints_world"][2])
    assert metrics["max_foot_lock_slide_m"] <= 0.003
    assert smpl_motion["players"][0]["foot_lock"]["max_slide_m"] <= 0.003


def test_footlock_resets_slide_when_contact_breaks() -> None:
    samples = [
        {
            "frame_idx": 0,
            "player_id": 1,
            "t": 0.0,
            "confidence": 0.9,
            "track_world_xy": [0.0, 0.0],
            "camera_translation": [0.0, 0.0, 0.0],
            "joints_camera": [
                [0.0, 0.0, 1.2],
                [0.0, 0.0, 0.0],
            ],
            "vertices_camera": [[0.0, 0.0, 0.0]],
            "global_orient": [0.0, 0.0, 0.0],
            "body_pose": [0.0, 0.0, 0.0],
            "betas": [0.0],
        },
        {
            "frame_idx": 1,
            "player_id": 1,
            "t": 1.0 / 30.0,
            "confidence": 0.9,
            "track_world_xy": [0.0, 0.0],
            "camera_translation": [0.0, 0.0, 0.0],
            "joints_camera": [
                [0.0, 0.0, 1.2],
                [0.0, 0.0, 0.2],
            ],
            "vertices_camera": [[0.0, 0.0, 0.0]],
            "global_orient": [0.0, 0.0, 0.0],
            "body_pose": [0.0, 0.0, 0.0],
            "betas": [0.0],
        },
        {
            "frame_idx": 2,
            "player_id": 1,
            "t": 2.0 / 30.0,
            "confidence": 0.9,
            "track_world_xy": [1.0, 0.0],
            "camera_translation": [0.0, 0.0, 0.0],
            "joints_camera": [
                [0.0, 0.0, 1.2],
                [0.0, 0.0, 0.0],
            ],
            "vertices_camera": [[0.0, 0.0, 0.0]],
            "global_orient": [0.0, 0.0, 0.0],
            "body_pose": [0.0, 0.0, 0.0],
            "betas": [0.0],
        },
    ]

    smpl_motion, _skeleton3d, metrics = worldhmr.build_body_artifacts_from_fast_sam(
        samples,
        calibration=_identity_calibration(),
        fps=30.0,
        smoothing_alpha=1.0,
    )

    frames = smpl_motion["players"][0]["frames"]
    assert frames[0]["foot_lock"] == {"left": True, "right": True}
    assert frames[1]["foot_lock"] == {"left": False, "right": False}
    assert frames[2]["foot_lock"] == {"left": True, "right": True}
    assert metrics["max_foot_lock_slide_m"] == pytest.approx(0.0)
    assert smpl_motion["players"][0]["foot_lock"]["max_slide_m"] == pytest.approx(0.0)


def test_footlock_marks_temporal_reset_after_long_sparse_output_gap() -> None:
    samples = [
        {
            "frame_idx": 0,
            "player_id": 1,
            "t": 0.0,
            "confidence": 0.9,
            "track_world_xy": [0.0, 0.0],
            "camera_translation": [0.0, 0.0, 0.0],
            "joints_camera": [
                [0.0, 0.0, 1.2],
                [-0.2, 0.0, 0.0],
                [0.2, 0.0, 0.0],
            ],
            "vertices_camera": [[0.0, 0.0, 0.0]],
            "global_orient": [0.0, 0.0, 0.0],
            "body_pose": [0.0, 0.0, 0.0],
            "betas": [0.0],
        },
        {
            "frame_idx": 12,
            "player_id": 1,
            "t": 12.0 / 30.0,
            "confidence": 0.9,
            "track_world_xy": [4.0, 0.0],
            "camera_translation": [0.0, 0.0, 0.0],
            "joints_camera": [
                [0.0, 0.0, 1.2],
                [-0.2, 0.0, 0.0],
                [0.2, 0.0, 0.0],
            ],
            "vertices_camera": [[0.0, 0.0, 0.0]],
            "global_orient": [0.0, 0.0, 0.0],
            "body_pose": [0.0, 0.0, 0.0],
            "betas": [0.0],
        },
    ]

    smpl_motion, skeleton3d, _metrics = worldhmr.build_body_artifacts_from_fast_sam(
        samples,
        calibration=_identity_calibration(),
        fps=30.0,
        smoothing_alpha=1.0,
        max_root_speed_mps=None,
    )
    quality = build_body_joint_quality(
        clip="clip_001",
        smpl_motion=smpl_motion,
        skeleton3d=skeleton3d,
        body_compute_execution={"summary": {"scheduled_frame_count": 2, "scheduled_player_frame_count": 2}},
        max_root_speed_for_review_mps=10.0,
        max_track_anchor_residual_for_review_m=12.0,
    )

    first, second = smpl_motion["players"][0]["frames"]
    assert first["temporal_smoothing_reset"] is False
    assert second["temporal_smoothing_reset"] is True
    assert quality["summary"]["temporal_smoothing_reset_count"] == 1
    assert "root_motion_temporal_jump" not in quality["quality_blockers"]


def test_short_sparse_output_gap_carries_filter_state_without_reset_metadata_loss() -> None:
    samples = [
        {
            "frame_idx": 0,
            "player_id": 1,
            "t": 0.0,
            "confidence": 0.9,
            "track_world_xy": [0.0, 0.0],
            "camera_translation": [0.0, 0.0, 0.0],
            "joints_camera": [
                [0.0, 0.0, 1.2],
                [-0.2, 0.0, 0.0],
                [0.2, 0.0, 0.0],
            ],
            "vertices_camera": [[0.0, 0.0, 0.0]],
            "global_orient": [0.0, 0.0, 0.0],
            "body_pose": [0.0, 0.0, 0.0],
            "betas": [0.0],
        },
        {
            "frame_idx": 4,
            "player_id": 1,
            "t": 4.0 / 30.0,
            "confidence": 0.9,
            "track_world_xy": [0.25, 0.0],
            "camera_translation": [0.0, 0.0, 0.0],
            "joints_camera": [
                [0.0, 0.0, 1.2],
                [-0.2, 0.0, 0.0],
                [0.2, 0.0, 0.0],
            ],
            "vertices_camera": [[0.0, 0.0, 0.0]],
            "global_orient": [0.0, 0.0, 0.0],
            "body_pose": [0.0, 0.0, 0.0],
            "betas": [0.0],
        },
    ]

    smpl_motion, skeleton3d, metrics = worldhmr.build_body_artifacts_from_fast_sam(
        samples,
        calibration=_identity_calibration(),
        fps=30.0,
        smoothing_alpha=1.0,
        max_root_speed_mps=8.0,
        smoothing_gap_carry_frames=8,
    )
    quality = build_body_joint_quality(
        clip="clip_001",
        smpl_motion=smpl_motion,
        skeleton3d=skeleton3d,
        body_compute_execution={"summary": {"scheduled_frame_count": 2, "scheduled_player_frame_count": 2}},
        max_root_speed_for_review_mps=10.0,
        max_track_anchor_residual_for_review_m=12.0,
    )

    first, second = smpl_motion["players"][0]["frames"]
    assert first["temporal_smoothing_reset"] is False
    assert second["temporal_smoothing_reset"] is False
    assert second["temporal_smoothing_metadata"]["gap"]["status"] == "carried"
    assert second["temporal_smoothing_metadata"]["gap"]["missing_frame_count"] == 3
    assert metrics["foot_lock_gap_carried_frames"] == 1
    assert quality["summary"]["temporal_smoothing_reset_count"] == 0


def test_long_sparse_output_gap_remains_honest_reset() -> None:
    samples = [
        {
            "frame_idx": 0,
            "player_id": 1,
            "t": 0.0,
            "confidence": 0.9,
            "track_world_xy": [0.0, 0.0],
            "camera_translation": [0.0, 0.0, 0.0],
            "joints_camera": [
                [0.0, 0.0, 1.2],
                [-0.2, 0.0, 0.0],
                [0.2, 0.0, 0.0],
            ],
            "vertices_camera": [[0.0, 0.0, 0.0]],
            "global_orient": [0.0, 0.0, 0.0],
            "body_pose": [0.0, 0.0, 0.0],
            "betas": [0.0],
        },
        {
            "frame_idx": 12,
            "player_id": 1,
            "t": 12.0 / 30.0,
            "confidence": 0.9,
            "track_world_xy": [0.35, 0.0],
            "camera_translation": [0.0, 0.0, 0.0],
            "joints_camera": [
                [0.0, 0.0, 1.2],
                [-0.2, 0.0, 0.0],
                [0.2, 0.0, 0.0],
            ],
            "vertices_camera": [[0.0, 0.0, 0.0]],
            "global_orient": [0.0, 0.0, 0.0],
            "body_pose": [0.0, 0.0, 0.0],
            "betas": [0.0],
        },
    ]

    smpl_motion, skeleton3d, metrics = worldhmr.build_body_artifacts_from_fast_sam(
        samples,
        calibration=_identity_calibration(),
        fps=30.0,
        smoothing_alpha=1.0,
        max_root_speed_mps=8.0,
        smoothing_gap_carry_frames=8,
    )
    quality = build_body_joint_quality(
        clip="clip_001",
        smpl_motion=smpl_motion,
        skeleton3d=skeleton3d,
        body_compute_execution={"summary": {"scheduled_frame_count": 2, "scheduled_player_frame_count": 2}},
        max_root_speed_for_review_mps=10.0,
        max_track_anchor_residual_for_review_m=12.0,
    )

    second = smpl_motion["players"][0]["frames"][1]
    assert second["temporal_smoothing_reset"] is True
    assert second["temporal_smoothing_metadata"]["reset_reason"] == "sparse_output_gap"
    assert second["temporal_smoothing_metadata"]["gap"]["missing_frame_count"] == 11
    assert metrics["foot_lock_gap_reset_frames"] == 1
    assert quality["summary"]["temporal_smoothing_reset_count"] == 1
    assert "root_motion_temporal_jump" not in quality["quality_blockers"]


def test_build_body_artifacts_anchors_track_xy_to_low_joint_cluster_not_first_joint() -> None:
    samples = [
        {
            "frame_idx": 12,
            "player_id": 1,
            "t": 0.4,
            "confidence": 0.9,
            "track_world_xy": [2.0, 3.0],
            "camera_translation": [0.0, 0.0, 0.0],
            "joints_camera": [
                [0.0, 0.0, 1.4],
                [0.45, -0.15, 0.0],
                [0.55, 0.15, 0.02],
                [0.1, 0.0, 0.75],
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
        smoothing_alpha=1.0,
        max_root_speed_mps=None,
    )

    frame = smpl_motion["players"][0]["frames"][0]
    low_joints = [joint for joint in frame["joints_world"] if joint[2] <= 0.08]
    low_center = [
        sum(joint[0] for joint in low_joints) / len(low_joints),
        sum(joint[1] for joint in low_joints) / len(low_joints),
    ]
    assert low_center == pytest.approx(frame["track_world_xy"])
    assert frame["joints_world"][0][:2] != pytest.approx(frame["track_world_xy"])
    assert metrics["grounding_anchor"] == "low_joint_cluster"


def test_body_artifact_smoothing_translates_mesh_vertices_with_root_delta() -> None:
    samples = [
        {
            "frame_idx": 0,
            "player_id": 1,
            "t": 0.0,
            "confidence": 0.9,
            "track_world_xy": [0.0, 0.0],
            "camera_translation": [0.0, 0.0, 0.0],
            "joints_camera": [[0.0, 0.0, 1.0]],
            "vertices_camera": [[1.0, 0.0, 1.0]],
            "global_orient": [0.0, 0.0, 0.0],
            "body_pose": [0.0, 0.0, 0.0],
            "betas": [0.0],
        },
        {
            "frame_idx": 1,
            "player_id": 1,
            "t": 1.0 / 30.0,
            "confidence": 0.9,
            "track_world_xy": [10.0, 0.0],
            "camera_translation": [0.0, 0.0, 0.0],
            "joints_camera": [[0.0, 0.0, 1.0]],
            "vertices_camera": [[1.0, 0.0, 1.0]],
            "global_orient": [0.0, 0.0, 0.0],
            "body_pose": [0.0, 0.0, 0.0],
            "betas": [0.0],
        },
    ]

    smpl_motion, _skeleton3d, _metrics = worldhmr.build_body_artifacts_from_fast_sam(
        samples,
        calibration=_identity_calibration(),
        fps=30.0,
        smoothing_alpha=0.5,
    )

    second = smpl_motion["players"][0]["frames"][1]
    assert second["transl_world"] == pytest.approx([5.0, 0.0, 0.0])
    assert second["joints_world"][0] == pytest.approx([5.0, 0.0, 0.0])
    assert second["mesh_vertices_world"][0] == pytest.approx([6.0, 0.0, 0.0])


def test_body_artifact_speed_limit_caps_root_motion_and_records_track_residual() -> None:
    samples = [
        {
            "frame_idx": 0,
            "player_id": 1,
            "t": 0.0,
            "confidence": 0.9,
            "track_world_xy": [0.0, 0.0],
            "camera_translation": [0.0, 0.0, 0.0],
            "joints_camera": [[0.0, 0.0, 1.0]],
            "vertices_camera": [[1.0, 0.0, 1.0]],
            "global_orient": [0.0, 0.0, 0.0],
            "body_pose": [0.0, 0.0, 0.0],
            "betas": [0.0],
        },
        {
            "frame_idx": 1,
            "player_id": 1,
            "t": 1.0 / 30.0,
            "confidence": 0.9,
            "track_world_xy": [10.0, 0.0],
            "camera_translation": [0.0, 0.0, 0.0],
            "joints_camera": [[0.0, 0.0, 1.0]],
            "vertices_camera": [[1.0, 0.0, 1.0]],
            "global_orient": [0.0, 0.0, 0.0],
            "body_pose": [0.0, 0.0, 0.0],
            "betas": [0.0],
        },
    ]

    smpl_motion, _skeleton3d, metrics = worldhmr.build_body_artifacts_from_fast_sam(
        samples,
        calibration=_identity_calibration(),
        fps=30.0,
        smoothing_alpha=1.0,
        max_root_speed_mps=3.0,
    )

    second = smpl_motion["players"][0]["frames"][1]
    assert second["track_world_xy"] == pytest.approx([10.0, 0.0])
    assert second["transl_world"] == pytest.approx([0.1, 0.0, 0.0])
    assert second["joints_world"][0] == pytest.approx([0.1, 0.0, 0.0])
    assert second["mesh_vertices_world"][0] == pytest.approx([1.1, 0.0, 0.0])
    assert metrics["root_speed_limited_frames"] == 1
    assert metrics["max_track_anchor_residual_m"] == pytest.approx(9.9)


def test_body_artifact_track_anchor_reset_prevents_wrong_player_smoothing() -> None:
    samples = [
        {
            "frame_idx": 0,
            "player_id": 1,
            "t": 0.0,
            "confidence": 0.9,
            "track_world_xy": [0.0, 0.0],
            "camera_translation": [0.0, 0.0, 0.0],
            "joints_camera": [[0.0, 0.0, 1.0]],
            "vertices_camera": [[1.0, 0.0, 1.0]],
            "global_orient": [0.0, 0.0, 0.0],
            "body_pose": [0.0, 0.0, 0.0],
            "betas": [0.0],
        },
        {
            "frame_idx": 1,
            "player_id": 1,
            "t": 1.0 / 30.0,
            "confidence": 0.9,
            "track_world_xy": [2.0, 0.0],
            "camera_translation": [0.0, 0.0, 0.0],
            "joints_camera": [[0.0, 0.0, 1.0]],
            "vertices_camera": [[1.0, 0.0, 1.0]],
            "global_orient": [0.0, 0.0, 0.0],
            "body_pose": [0.0, 0.0, 0.0],
            "betas": [0.0],
        },
    ]

    smpl_motion, _skeleton3d, metrics = worldhmr.build_body_artifacts_from_fast_sam(
        samples,
        calibration=_identity_calibration(),
        fps=30.0,
        smoothing_alpha=1.0,
        max_root_speed_mps=3.0,
        max_track_anchor_smoothing_residual_m=0.5,
    )

    second = smpl_motion["players"][0]["frames"][1]
    assert second["temporal_smoothing_reset"] is True
    assert second["track_world_xy"] == pytest.approx([2.0, 0.0])
    assert second["transl_world"] == pytest.approx([2.0, 0.0, 0.0])
    assert second["joints_world"][0] == pytest.approx([2.0, 0.0, 0.0])
    assert second["mesh_vertices_world"][0] == pytest.approx([3.0, 0.0, 0.0])
    assert metrics["root_speed_limited_frames"] == 1
    assert metrics["track_anchor_residual_reset_frames"] == 1
    assert metrics["max_pre_reset_track_anchor_residual_m"] == pytest.approx(1.9)
    assert metrics["max_track_anchor_residual_m"] == pytest.approx(0.0)


def test_track_anchor_residual_carry_respects_root_speed_cap_without_reset() -> None:
    samples = [
        {
            "frame_idx": 0,
            "player_id": 1,
            "t": 0.0,
            "confidence": 0.9,
            "track_world_xy": [0.0, 0.0],
            "camera_translation": [0.0, 0.0, 0.0],
            "joints_camera": [[0.0, 0.0, 1.0]],
            "vertices_camera": [[1.0, 0.0, 1.0]],
            "global_orient": [0.0, 0.0, 0.0],
            "body_pose": [0.0, 0.0, 0.0],
            "betas": [0.0],
        },
        {
            "frame_idx": 1,
            "player_id": 1,
            "t": 1.0 / 30.0,
            "confidence": 0.9,
            "track_world_xy": [0.6, 0.0],
            "camera_translation": [0.0, 0.0, 0.0],
            "joints_camera": [[0.0, 0.0, 1.0]],
            "vertices_camera": [[1.0, 0.0, 1.0]],
            "global_orient": [0.0, 0.0, 0.0],
            "body_pose": [0.0, 0.0, 0.0],
            "betas": [0.0],
        },
    ]

    smpl_motion, skeleton3d, metrics = worldhmr.build_body_artifacts_from_fast_sam(
        samples,
        calibration=_identity_calibration(),
        fps=30.0,
        smoothing_alpha=1.0,
        max_root_speed_mps=8.0,
        max_track_anchor_smoothing_residual_m=0.2,
        smoothing_residual_identity_reset_m=1.0,
    )
    quality = build_body_joint_quality(
        clip="clip_001",
        smpl_motion=smpl_motion,
        skeleton3d=skeleton3d,
        body_compute_execution={"summary": {"scheduled_frame_count": 2, "scheduled_player_frame_count": 2}},
        min_joint_count=1,
        max_root_speed_for_review_mps=10.0,
        max_track_anchor_residual_for_review_m=12.0,
    )

    second = smpl_motion["players"][0]["frames"][1]
    assert second["temporal_smoothing_reset"] is False
    assert second["transl_world"] == pytest.approx([8.0 / 30.0, 0.0, 0.0])
    assert second["temporal_smoothing_metadata"]["residual"]["status"] == "carried"
    assert metrics["track_anchor_residual_reset_frames"] == 0
    assert metrics["track_anchor_residual_carried_frames"] == 1
    assert quality["summary"]["root_motion_temporal_jump_count"] == 0
    assert quality["summary"]["max_root_speed_mps"] <= 8.0 + 1e-9


def test_world_joint_visual_smoothing_keeps_wrist_peak_frame_and_limb_lengths() -> None:
    joint_names = [
        "nose",
        "left_eye",
        "right_eye",
        "left_ear",
        "right_ear",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
        "left_big_toe_tip",
        "left_small_toe_tip",
        "left_heel",
        "right_big_toe_tip",
        "right_small_toe_tip",
        "right_heel",
        *[f"unused_{idx}" for idx in range(21, 41)],
        "right_wrist",
        *[f"unused_{idx}" for idx in range(42, 62)],
        "left_wrist",
        *[f"unused_{idx}" for idx in range(63, 70)],
    ]
    idx = {name: pos for pos, name in enumerate(joint_names)}
    wrist_offsets = [0.0, 0.45, 1.20, 0.45, 0.0]
    ankle_noise = [0.0, 0.08, -0.08, 0.08, 0.0]
    frames = []
    for frame_idx, wrist_offset in enumerate(wrist_offsets):
        joints = [[0.0, 0.0, 1.0] for _ in joint_names]
        joints[idx["left_shoulder"]] = [0.0, 0.0, 1.5]
        joints[idx["left_elbow"]] = [0.3, 0.0, 1.2]
        joints[idx["left_wrist"]] = [0.3 + wrist_offset, 0.0, 1.2]
        joints[idx["right_shoulder"]] = [0.0, 0.0, 1.5]
        joints[idx["right_elbow"]] = [-0.3, 0.0, 1.2]
        joints[idx["right_wrist"]] = [-0.6, 0.0, 1.2]
        joints[idx["left_hip"]] = [0.0, 0.0, 1.0]
        joints[idx["left_knee"]] = [0.0, 0.0, 0.5]
        joints[idx["left_ankle"]] = [ankle_noise[frame_idx], 0.0, 0.0]
        joints[idx["left_heel"]] = [ankle_noise[frame_idx] - 0.1, 0.0, 0.0]
        frames.append(
            {
                "frame_idx": frame_idx,
                "t": frame_idx / 30.0,
                "transl_world": [0.0, 0.0, 0.0],
                "track_world_xy": [0.0, 0.0],
                "joints_world": joints,
                "joint_conf": [0.9] * len(joints),
            }
        )
    smpl_motion = {
        "schema_version": 1,
        "model": "sam3dbody_world_joints",
        "fps": 30.0,
        "world_frame": "court_Z0",
        "players": [
            {
                "id": 1,
                "betas": [0.0],
                "skate_free": False,
                "physics": "worldhmr_grounded_not_footlocked",
                "frames": [
                    {
                        **frame,
                        "global_orient": [0.0, 0.0, 0.0],
                        "body_pose": [0.0, 0.0, 0.0],
                        "left_hand_pose": [],
                        "right_hand_pose": [],
                        "mesh_vertices_world": [],
                        "foot_contact": {"left": False, "right": False},
                        "foot_lock": {"left": False, "right": False},
                        "grf": None,
                    }
                    for frame in frames
                ],
            }
        ],
    }
    skeleton3d = {
        "schema_version": 1,
        "artifact_type": "racketsport_skeleton3d",
        "fps": 30.0,
        "world_frame": "court_Z0",
        "source_model": "sam3d_body_joints",
        "joint_names": joint_names,
        "preview_only": False,
        "players": [{"id": 1, "frames": copy.deepcopy(frames)}],
        "provenance": {},
    }

    smoothed_smpl, smoothed_skeleton, metrics = worldhmr._apply_world_joint_visual_smoothing(
        smpl_motion,
        skeleton3d,
        fps=30.0,
    )

    before_frames = skeleton3d["players"][0]["frames"]
    after_frames = smoothed_skeleton["players"][0]["frames"]
    before_peak = max(before_frames, key=lambda frame: frame["joints_world"][idx["left_wrist"]][0])["frame_idx"]
    after_peak = max(after_frames, key=lambda frame: frame["joints_world"][idx["left_wrist"]][0])["frame_idx"]
    assert after_peak == before_peak
    for before, after in zip(before_frames, after_frames, strict=True):
        assert worldhmr._distance3(
            before["joints_world"][idx["left_elbow"]],
            before["joints_world"][idx["left_wrist"]],
        ) == pytest.approx(
            worldhmr._distance3(
                after["joints_world"][idx["left_elbow"]],
                after["joints_world"][idx["left_wrist"]],
            )
        )
        assert worldhmr._distance3(
            before["joints_world"][idx["left_knee"]],
            before["joints_world"][idx["left_ankle"]],
        ) == pytest.approx(
            worldhmr._distance3(
                after["joints_world"][idx["left_knee"]],
                after["joints_world"][idx["left_ankle"]],
            )
        )
    assert metrics["enabled"] is True
    assert metrics["max_wrist_peak_delta_frames"] == 0
    assert metrics["max_lag_frames"] <= 1
    assert metrics["limb_length_max_delta_m"] <= 1e-9
    assert smoothed_smpl["players"][0]["frames"][2]["joints_world"][idx["left_wrist"]][0] > 0.0


def test_body_artifact_default_speed_cap_feeds_review_quality_gate() -> None:
    samples = [
        {
            "frame_idx": 0,
            "player_id": 1,
            "t": 0.0,
            "confidence": 0.9,
            "track_world_xy": [0.0, 0.0],
            "camera_translation": [0.0, 0.0, 0.0],
            "joints_camera": [[0.0, 0.0, 1.0]],
            "vertices_camera": [[1.0, 0.0, 1.0]],
            "global_orient": [0.0, 0.0, 0.0],
            "body_pose": [0.0, 0.0, 0.0],
            "betas": [0.0],
        },
        {
            "frame_idx": 1,
            "player_id": 1,
            "t": 1.0 / 30.0,
            "confidence": 0.9,
            "track_world_xy": [10.0, 0.0],
            "camera_translation": [0.0, 0.0, 0.0],
            "joints_camera": [[0.0, 0.0, 1.0]],
            "vertices_camera": [[1.0, 0.0, 1.0]],
            "global_orient": [0.0, 0.0, 0.0],
            "body_pose": [0.0, 0.0, 0.0],
            "betas": [0.0],
        },
    ]

    smpl_motion, skeleton3d, metrics = worldhmr.build_body_artifacts_from_fast_sam(
        samples,
        calibration=_identity_calibration(),
        fps=30.0,
        smoothing_alpha=1.0,
        max_root_speed_mps=8.0,
    )
    quality = build_body_joint_quality(
        clip="clip_001",
        smpl_motion=smpl_motion,
        skeleton3d=skeleton3d,
        body_compute_execution={"summary": {"scheduled_frame_count": 2, "scheduled_player_frame_count": 2}},
        min_joint_count=1,
        max_root_speed_for_review_mps=10.0,
        max_track_anchor_residual_for_review_m=12.0,
    )

    assert metrics["root_speed_limited_frames"] == 1
    assert metrics["max_track_anchor_residual_m"] > 0.0
    assert quality["status"] == "quality_checked_needs_accuracy_gate"
    assert "root_motion_temporal_jump" not in quality["quality_blockers"]
    assert quality["summary"]["max_root_speed_mps"] < quality["summary"]["max_root_speed_for_review_mps"]


def test_footlock_preserves_configured_root_speed_cap_after_pinning() -> None:
    samples = [
        {
            "frame_idx": 0,
            "player_id": 1,
            "t": 0.0,
            "confidence": 0.9,
            "track_world_xy": [0.0, 0.0],
            "camera_translation": [0.0, 0.0, 0.0],
            "joints_camera": [
                [0.0, 0.0, 1.2],
                [-1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
            "vertices_camera": [[0.0, 0.0, 0.0]],
            "global_orient": [0.0, 0.0, 0.0],
            "body_pose": [0.0, 0.0, 0.0],
            "betas": [0.0],
        },
        {
            "frame_idx": 1,
            "player_id": 1,
            "t": 1.0 / 30.0,
            "confidence": 0.9,
            "track_world_xy": [0.2, 0.0],
            "camera_translation": [0.0, 0.0, 0.0],
            "joints_camera": [
                [0.0, 0.0, 1.2],
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.2],
            ],
            "vertices_camera": [[0.0, 0.0, 0.0]],
            "global_orient": [0.0, 0.0, 0.0],
            "body_pose": [0.0, 0.0, 0.0],
            "betas": [0.0],
        },
    ]

    smpl_motion, skeleton3d, metrics = worldhmr.build_body_artifacts_from_fast_sam(
        samples,
        calibration=_identity_calibration(),
        fps=30.0,
        smoothing_alpha=1.0,
        max_root_speed_mps=3.0,
    )
    quality = build_body_joint_quality(
        clip="clip_001",
        smpl_motion=smpl_motion,
        skeleton3d=skeleton3d,
        body_compute_execution={"summary": {"scheduled_frame_count": 2, "scheduled_player_frame_count": 2}},
        min_joint_count=3,
        max_root_speed_for_review_mps=10.0,
        max_track_anchor_residual_for_review_m=3.0,
    )

    assert metrics["root_speed_limited_frames"] == 1
    assert "foot_lock_root_speed_limited_frames" in metrics
    assert quality["status"] == "quality_checked_needs_accuracy_gate"
    assert "root_motion_temporal_jump" not in quality["quality_blockers"]
    assert quality["summary"]["max_root_speed_mps"] <= 3.0 + 1e-9


def test_footlock_breaks_contact_when_low_joint_moves_too_fast_relative_to_root() -> None:
    samples = [
        {
            "frame_idx": 0,
            "player_id": 1,
            "t": 0.0,
            "confidence": 0.9,
            "track_world_xy": [0.0, 0.0],
            "camera_translation": [0.0, 0.0, 0.0],
            "joints_camera": [
                [0.0, 0.0, 1.2],
                [-1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
            "vertices_camera": [[0.0, 0.0, 0.0]],
            "global_orient": [0.0, 0.0, 0.0],
            "body_pose": [0.0, 0.0, 0.0],
            "betas": [0.0],
        },
        {
            "frame_idx": 1,
            "player_id": 1,
            "t": 1.0 / 30.0,
            "confidence": 0.9,
            "track_world_xy": [0.2, 0.0],
            "camera_translation": [0.0, 0.0, 0.0],
            "joints_camera": [
                [0.0, 0.0, 1.2],
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.2],
            ],
            "vertices_camera": [[0.0, 0.0, 0.0]],
            "global_orient": [0.0, 0.0, 0.0],
            "body_pose": [0.0, 0.0, 0.0],
            "betas": [0.0],
        },
    ]

    _smpl_motion, _skeleton3d, metrics = worldhmr.build_body_artifacts_from_fast_sam(
        samples,
        calibration=_identity_calibration(),
        fps=30.0,
        smoothing_alpha=1.0,
        max_root_speed_mps=3.0,
    )

    assert metrics["max_foot_lock_slide_m"] <= 0.03


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

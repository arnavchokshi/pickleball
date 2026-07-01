from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import types

import numpy as np
import pytest

from threed.racketsport.pose_fast import (
    LANE_A_RTMW3D_JOINT_NAMES,
    RTMW3D_WHOLEBODY_133_JOINT_NAMES,
    PoseCropResult,
    PoseCropRequest,
    RTMW3DPoseRuntime,
    build_lane_a_skeleton3d_from_rtmw3d,
)
from threed.racketsport.schemas import CameraIntrinsics, CaptureQuality, CourtCalibration, CourtExtrinsics, ReprojectionError, Tracks


def _tracks_payload() -> dict:
    return {
        "schema_version": 1,
        "fps": 30.0,
        "players": [
            {
                "id": 7,
                "side": "near",
                "role": "left",
                "frames": [
                    {"t": 0.0, "bbox": [100.0, 100.0, 200.0, 300.0], "world_xy": [2.0, 3.0], "conf": 0.92},
                    {"t": 1.0 / 30.0, "bbox": [102.0, 100.0, 202.0, 300.0], "world_xy": [2.1, 3.0], "conf": 0.91},
                ],
            },
            {
                "id": 8,
                "side": "far",
                "role": "right",
                "frames": [
                    {"t": 0.0, "bbox": [300.0, 120.0, 380.0, 310.0], "world_xy": [-2.0, -3.0], "conf": 0.88},
                ],
            },
        ],
        "rally_spans": [],
    }


def _wholebody_joints(*, wrist_x: float = 0.45) -> list[list[float]]:
    joints = [[0.01 * idx, 0.0, 1.0 + 0.001 * idx] for idx in range(133)]
    joints[17] = [-0.15, 0.0, 0.02]
    joints[18] = [-0.10, 0.0, 0.01]
    joints[19] = [-0.20, 0.0, 0.0]
    joints[20] = [0.15, 0.0, 0.03]
    joints[21] = [0.10, 0.0, 0.02]
    joints[22] = [0.20, 0.0, 0.02]
    joints[9] = [-wrist_x, 0.05, 1.25]
    joints[10] = [wrist_x, 0.05, 1.25]
    return joints


def _pixel_grounding_calibration() -> CourtCalibration:
    return CourtCalibration(
        schema_version=1,
        sport="pickleball",
        homography=[[0.02, 0.0, -1.0], [0.0, 0.02, -1.0], [0.0, 0.0, 1.0]],
        intrinsics=CameraIntrinsics(fx=100.0, fy=100.0, cx=50.0, cy=50.0, dist=[], source="solvepnp_test"),
        extrinsics=CourtExtrinsics(
            R=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            t=[0.0, 0.0, 2.0],
            camera_height_m=2.0,
        ),
        reprojection_error_px=ReprojectionError(median=0.0, p95=0.0),
        capture_quality=CaptureQuality(grade="good", reasons=[]),
        image_pts=[[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]],
        world_pts=[[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0]],
    )


def test_lane_a_joint_names_keep_body_feet_hands_and_drop_face() -> None:
    assert len(RTMW3D_WHOLEBODY_133_JOINT_NAMES) == 133
    assert len(LANE_A_RTMW3D_JOINT_NAMES) == 65
    assert "left_wrist" in LANE_A_RTMW3D_JOINT_NAMES
    assert "right_wrist" in LANE_A_RTMW3D_JOINT_NAMES
    assert "face_00" not in LANE_A_RTMW3D_JOINT_NAMES
    assert "left_hand_20" in LANE_A_RTMW3D_JOINT_NAMES
    assert "right_hand_20" in LANE_A_RTMW3D_JOINT_NAMES


def test_build_lane_a_skeleton3d_keeps_all_tracked_player_frames_and_grounds_support_foot() -> None:
    results = [
        PoseCropResult(
            frame_idx=0,
            player_id=7,
            joints_m=_wholebody_joints(),
            joint_conf=[0.9] * 133,
        ),
        PoseCropResult(
            frame_idx=1,
            player_id=7,
            joints_m=_wholebody_joints(wrist_x=0.50),
            joint_conf=[0.9] * 133,
        ),
        PoseCropResult(
            frame_idx=0,
            player_id=8,
            joints_m=_wholebody_joints(),
            joint_conf=[0.8] * 133,
        ),
    ]

    skeleton = build_lane_a_skeleton3d_from_rtmw3d(
        Tracks.model_validate(_tracks_payload()),
        results,
        world_frame="court_Z0",
        source_model="rtmw3d_x",
    )

    assert skeleton["artifact_type"] == "racketsport_skeleton3d"
    assert skeleton["preview_only"] is False
    assert skeleton["world_frame"] == "court_Z0"
    assert skeleton["source_model"] == "rtmw3d_x"
    assert skeleton["joint_names"] == list(LANE_A_RTMW3D_JOINT_NAMES)
    assert [(player["id"], len(player["frames"])) for player in skeleton["players"]] == [(7, 2), (8, 1)]
    first_frame = skeleton["players"][0]["frames"][0]
    foot_z_values = [joint[2] for name, joint in zip(skeleton["joint_names"], first_frame["joints_world"]) if "toe" in name or "heel" in name]
    assert min(foot_z_values) == pytest.approx(0.0)
    support_xy = first_frame["joints_world"][19]
    assert support_xy[:2] == pytest.approx([2.0, 3.0])


def test_build_lane_a_skeleton3d_backprojects_support_foot_pixels_when_calibrated() -> None:
    joint_pixels = [[50.0, 50.0] for _name in RTMW3D_WHOLEBODY_133_JOINT_NAMES]
    joint_pixels[19] = [150.0, 50.0]
    results = [
        PoseCropResult(
            frame_idx=0,
            player_id=7,
            joints_m=_wholebody_joints(),
            joint_conf=[0.9] * 133,
            joint_pixels=joint_pixels,
        ),
        PoseCropResult(
            frame_idx=1,
            player_id=7,
            joints_m=_wholebody_joints(),
            joint_conf=[0.9] * 133,
            joint_pixels=joint_pixels,
        ),
        PoseCropResult(
            frame_idx=0,
            player_id=8,
            joints_m=_wholebody_joints(),
            joint_conf=[0.9] * 133,
            joint_pixels=joint_pixels,
        ),
    ]

    skeleton = build_lane_a_skeleton3d_from_rtmw3d(
        Tracks.model_validate(_tracks_payload()),
        results,
        world_frame="court_Z0",
        source_model="rtmw3d_x",
        calibration=_pixel_grounding_calibration(),
    )

    frame = skeleton["players"][0]["frames"][0]
    assert frame["joints_world"][19][:2] == pytest.approx([2.0, 0.0])
    assert frame["joints_world"][19][:2] != pytest.approx([2.0, 3.0])
    assert skeleton["provenance"]["grounding"] == "support_foot_pixel_backprojected_to_court_z0"
    assert skeleton["provenance"]["grounding_fallback_count"] == 0


def test_rtmw3d_runtime_fails_closed_when_manifest_checkpoint_is_not_available(tmp_path: Path) -> None:
    manifest_path = tmp_path / "MANIFEST.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "models": [
                    {
                        "id": "rtmw3d_x",
                        "stage": "lane_a_3d_pose",
                        "use": "Lane A always-on 3D whole-body pose source",
                        "source": "https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose3d",
                        "license": "Apache-2.0",
                        "commercial_posture": "ok",
                        "status": "pending_download",
                        "local_path": "/workspace/checkpoints/body4d/mmpose/rtmw3d-x.pth",
                        "sha256": None,
                        "fallbacks": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    runtime = RTMW3DPoseRuntime(manifest_path=manifest_path)

    with pytest.raises(RuntimeError, match="model rtmw3d_x is not available_on_h100: status=pending_download"):
        runtime.infer_frame(
            tmp_path / "frame_000000.jpg",
            [
                PoseCropRequest(
                    frame_idx=0,
                    player_id=7,
                    bbox_xyxy=[100.0, 100.0, 200.0, 300.0],
                    track_world_xy=[0.0, 0.0],
                    track_confidence=0.9,
                )
            ],
        )


def test_rtmw3d_runtime_uses_mmpose_topdown_and_returns_133_joint_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "rtmw3d-x_8xb32_cocktail14-384x288.py"
    config_path.write_text("# fake config used by fake mmpose init\n", encoding="utf-8")
    checkpoint_path = tmp_path / "rtmw3d-x_8xb64_cocktail14-384x288-b0a0eab7_20240626.pth"
    checkpoint_bytes = b"fake rtmw3d checkpoint"
    checkpoint_path.write_bytes(checkpoint_bytes)
    manifest_path = tmp_path / "MANIFEST.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "models": [
                    {
                        "id": "rtmw3d_x",
                        "stage": "lane_a_3d_pose",
                        "use": "Lane A always-on 3D whole-body pose source",
                        "source": "https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose3d",
                        "license": "Apache-2.0",
                        "commercial_posture": "ok",
                        "status": "available_on_h100",
                        "local_path": str(checkpoint_path),
                        "sha256": hashlib.sha256(checkpoint_bytes).hexdigest(),
                        "fallbacks": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    calls: dict[str, object] = {}

    def fake_register_all_modules() -> None:
        calls["registered"] = True

    def fake_init_model(config: str, checkpoint: str, device: str) -> object:
        calls["init"] = {"config": config, "checkpoint": checkpoint, "device": device}
        return {"model": "rtmw3d_x"}

    def fake_inference_topdown(
        model: object,
        img: str,
        *,
        bboxes: np.ndarray,
        bbox_format: str,
    ) -> list[object]:
        calls["inference"] = {"model": model, "img": img, "bboxes": bboxes.copy(), "bbox_format": bbox_format}
        samples: list[object] = []
        for index in range(len(bboxes)):
            keypoints = np.zeros((1, 133, 3), dtype=np.float64)
            keypoints[0, :, 0] = float(index + 1)
            keypoints[0, :, 1] = 0.5
            keypoints[0, :, 2] = 1.25
            transformed_keypoints = np.zeros((1, 133, 2), dtype=np.float64)
            transformed_keypoints[0, :, 0] = 100.0 + index
            transformed_keypoints[0, :, 1] = 200.0 + index
            scores = np.full((1, 133), 0.9 - index * 0.1, dtype=np.float32)
            pred_instances = types.SimpleNamespace(
                keypoints=keypoints,
                keypoint_scores=scores,
                transformed_keypoints=transformed_keypoints,
            )
            samples.append(types.SimpleNamespace(pred_instances=pred_instances))
        return samples

    fake_mmpose = types.ModuleType("mmpose")
    fake_apis = types.ModuleType("mmpose.apis")
    fake_apis.init_model = fake_init_model
    fake_apis.inference_topdown = fake_inference_topdown
    fake_utils = types.ModuleType("mmpose.utils")
    fake_utils.register_all_modules = fake_register_all_modules
    monkeypatch.setitem(sys.modules, "mmpose", fake_mmpose)
    monkeypatch.setitem(sys.modules, "mmpose.apis", fake_apis)
    monkeypatch.setitem(sys.modules, "mmpose.utils", fake_utils)

    image_path = tmp_path / "frame_000000.jpg"
    image_path.write_bytes(b"fake image path only")
    runtime = RTMW3DPoseRuntime(
        manifest_path=manifest_path,
        config_path=config_path,
        project_pythonpath=tmp_path / "project" / "rtmpose3d",
        device="cuda:0",
    )

    results = runtime.infer_frame(
        image_path,
        [
            PoseCropRequest(
                frame_idx=0,
                player_id=7,
                bbox_xyxy=[1.0, 2.0, 11.0, 22.0],
                track_world_xy=[0.0, 0.0],
                track_confidence=0.9,
            ),
            PoseCropRequest(
                frame_idx=0,
                player_id=8,
                bbox_xyxy=[3.0, 4.0, 13.0, 24.0],
                track_world_xy=[0.0, 0.0],
                track_confidence=0.8,
            ),
        ],
    )

    assert calls["registered"] is True
    assert calls["init"] == {"config": str(config_path), "checkpoint": str(checkpoint_path), "device": "cuda:0"}
    inference = calls["inference"]
    assert isinstance(inference, dict)
    assert inference["model"] == {"model": "rtmw3d_x"}
    assert inference["img"] == str(image_path)
    np.testing.assert_array_equal(
        inference["bboxes"],
        np.array([[1.0, 2.0, 11.0, 22.0], [3.0, 4.0, 13.0, 24.0]], dtype=np.float32),
    )
    assert inference["bbox_format"] == "xyxy"
    assert [(result.frame_idx, result.player_id) for result in results] == [(0, 7), (0, 8)]
    assert [len(result.joints_m) for result in results] == [133, 133]
    assert [len(result.joint_conf) for result in results] == [133, 133]
    assert results[0].joint_names == RTMW3D_WHOLEBODY_133_JOINT_NAMES
    assert results[0].joints_m[9] == [1.0, 0.5, 1.25]
    assert results[1].joints_m[9] == [2.0, 0.5, 1.25]
    assert results[0].joint_pixels[9] == [100.0, 200.0]
    assert results[1].joint_pixels[9] == [101.0, 201.0]
    assert results[0].joint_conf[9] == pytest.approx(0.9)
    assert results[1].joint_conf[9] == pytest.approx(0.8)

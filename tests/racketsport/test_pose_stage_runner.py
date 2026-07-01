from __future__ import annotations

import json
from pathlib import Path

from tests.racketsport.calibration_fixtures import minimal_calibration_image_pts, minimal_calibration_world_pts
import threed.racketsport.orchestrator as orchestrator_module
from threed.racketsport.orchestrator import PoseStageRunner, StageContext
from threed.racketsport.pose_fast import PoseCropResult, RTMW3D_WHOLEBODY_133_JOINT_NAMES
from threed.racketsport.pose_temporal import MOTIONBERT_CONFIG_ENV
from threed.racketsport.schemas import (
    CameraIntrinsics,
    CaptureQuality,
    CourtCalibration,
    CourtExtrinsics,
    ReprojectionError,
    Skeleton3D,
    validate_artifact_file,
)


def _calibration() -> CourtCalibration:
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


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


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
                    {"t": 0.0, "bbox": [100.0, 100.0, 200.0, 300.0], "world_xy": [1.0, 2.0], "conf": 0.92},
                ],
            },
            {
                "id": 8,
                "side": "far",
                "role": "right",
                "frames": [
                    {"t": 0.0, "bbox": [300.0, 120.0, 380.0, 320.0], "world_xy": [-1.0, -2.0], "conf": 0.88},
                ],
            },
        ],
        "rally_spans": [],
    }


def _joints() -> list[list[float]]:
    joints = [[0.0, 0.0, 1.0] for _name in RTMW3D_WHOLEBODY_133_JOINT_NAMES]
    joints[19] = [0.0, 0.0, 0.0]
    joints[9] = [-0.35, 0.0, 1.25]
    joints[10] = [0.35, 0.0, 1.25]
    return joints


class FakeRTMW3DRuntime:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def infer_frame(self, image_path: Path, requests: list[object]) -> list[PoseCropResult]:
        self.calls.append({"image_path": image_path, "requests": requests})
        return [
            PoseCropResult(
                frame_idx=request.frame_idx,
                player_id=request.player_id,
                joints_m=_joints(),
                joint_conf=[0.95] * len(RTMW3D_WHOLEBODY_133_JOINT_NAMES),
            )
            for request in requests
        ]


def test_pose_stage_runner_batches_all_player_crops_and_writes_real_skeleton3d(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs"
    run_dir = tmp_path / "run"
    _write_json(run_dir / "tracks.json", _tracks_payload())
    _write_json(run_dir / "court_calibration.json", _calibration().model_dump(mode="json"))
    frames = inputs / "body_frames"
    frames.mkdir(parents=True)
    (frames / "frame_000000.jpg").write_bytes(b"not decoded by fake runtime")
    runtime = FakeRTMW3DRuntime()

    result = PoseStageRunner(runtime=runtime).run(
        StageContext(
            clip="clip_001",
            inputs_dir=inputs,
            run_dir=run_dir,
            sport="pickleball",
        )
    )

    assert result.stage == "pose"
    assert result.status == "ran"
    assert result.real_model is True
    assert result.source_mode == "rtmw3d_x_lane_a"
    assert result.produced_artifacts == ("skeleton3d.json",)
    assert len(runtime.calls) == 1
    assert Path(runtime.calls[0]["image_path"]).name == "frame_000000.jpg"
    assert [request.player_id for request in runtime.calls[0]["requests"]] == [7, 8]
    skeleton = validate_artifact_file("skeleton3d", run_dir / "skeleton3d.json")
    assert isinstance(skeleton, Skeleton3D)
    assert skeleton.preview_only is False
    assert skeleton.source_model == "rtmw3d_x"
    assert skeleton.provenance["temporal_refine"]["motionbert"] == "not_configured"
    assert skeleton.provenance["temporal_refine"]["motionbert_window_max_frames"] == 243
    assert [player.id for player in skeleton.players] == [7, 8]
    assert [len(player.frames) for player in skeleton.players] == [1, 1]


def test_pose_stage_runner_constructs_default_motionbert_runtime_when_configured(
    tmp_path: Path,
    monkeypatch,
) -> None:
    inputs = tmp_path / "inputs"
    run_dir = tmp_path / "run"
    manifest = tmp_path / "MANIFEST.json"
    _write_json(run_dir / "tracks.json", _tracks_payload())
    _write_json(run_dir / "court_calibration.json", _calibration().model_dump(mode="json"))
    _write_json(manifest, {"models": []})
    frames = inputs / "body_frames"
    frames.mkdir(parents=True)
    (frames / "frame_000000.jpg").write_bytes(b"not decoded by fake runtime")
    config_path = tmp_path / "motionbert.yaml"
    config_path.write_text("model: fake\n", encoding="utf-8")
    monkeypatch.setenv(MOTIONBERT_CONFIG_ENV, str(config_path))
    constructed: list[object] = []

    class FakeMotionBERTTemporalRuntime:
        model_id = "motionbert_lift_smooth"

        def __init__(self, *, manifest_path: str | Path, device: str | None) -> None:
            self.manifest_path = Path(manifest_path)
            self.device = device
            self.calls: list[dict[str, object]] = []
            constructed.append(self)

        def refine_body17_window(
            self,
            *,
            player_id: int,
            frames: list[dict],
            joint_names: list[str],
        ) -> list[list[list[float]]]:
            self.calls.append({"player_id": player_id, "frames": frames, "joint_names": joint_names})
            return [
                [
                    [float(joint[0]), float(joint[1]), float(joint[2])]
                    for joint in frame["joints_world"][:17]
                ]
                for frame in frames
            ]

    monkeypatch.setattr(orchestrator_module, "MotionBERTTemporalRuntime", FakeMotionBERTTemporalRuntime, raising=False)
    runtime = FakeRTMW3DRuntime()

    result = PoseStageRunner(manifest_path=manifest, runtime=runtime).run(
        StageContext(
            clip="clip_001",
            inputs_dir=inputs,
            run_dir=run_dir,
            sport="pickleball",
            device="cuda:0",
        )
    )

    assert len(constructed) == 1
    motionbert_runtime = constructed[0]
    assert motionbert_runtime.manifest_path == manifest
    assert motionbert_runtime.device == "cuda:0"
    assert [call["player_id"] for call in motionbert_runtime.calls] == [7, 8]
    assert motionbert_runtime.calls[0]["joint_names"] == list(RTMW3D_WHOLEBODY_133_JOINT_NAMES[:17])
    assert result.metrics["motionbert"] == "applied"
    skeleton = validate_artifact_file("skeleton3d", run_dir / "skeleton3d.json")
    assert isinstance(skeleton, Skeleton3D)
    assert skeleton.provenance["temporal_refine"]["motionbert"] == "applied"
    assert skeleton.provenance["temporal_refine"]["motionbert_model_id"] == "motionbert_lift_smooth"

from __future__ import annotations

import json
from pathlib import Path

import pytest

from threed.racketsport.orchestrator import BodyStageRunner, run_pipeline
from threed.racketsport.schemas import Skeleton3D, SmplMotion, validate_artifact_file


def _manifest(root: Path, *, yolo_sha: str | None = None) -> Path:
    fast = root / "sam-3d-body-dinov3" / "model.ckpt"
    mhr = root / "sam-3d-body-dinov3" / "assets" / "mhr_model.pt"
    moge = root / "moge-2-vitl-normal" / "model.pt"
    yolo = root / "yolo26" / "yolo26m.pt"
    for path, body in (
        (fast, b"fast-sam-body"),
        (mhr, b"mhr-model"),
        (moge, b"moge-model"),
        (yolo, b"yolo26m"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)

    payload = {
        "schema_version": 1,
        "models": [
            _entry("fast_sam_3d_body_dinov3", "3d_body_backbone", fast),
            _entry("sam_3d_body_mhr_model", "3d_body_backbone", mhr),
            _entry("moge_2_vitl_normal", "camera_fov_depth_prior", moge),
            _entry("yolo26m", "person_detect", yolo, sha256=yolo_sha),
        ],
    }
    path = root / "MANIFEST.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _entry(model_id: str, stage: str, path: Path, *, sha256: str | None = None) -> dict[str, object]:
    import hashlib

    return {
        "id": model_id,
        "stage": stage,
        "use": "test",
        "source": "test",
        "license": "test",
        "commercial_posture": "ok",
        "status": "available_on_h100",
        "local_path": str(path),
        "sha256": sha256 or hashlib.sha256(path.read_bytes()).hexdigest(),
        "fallbacks": [],
    }


def _sidecar_payload() -> dict:
    return {
        "schema_version": 1,
        "device_tier": "B_standard",
        "device_model": "iPhone16,2",
        "fps": 30,
        "format": "hevc",
        "resolution": [1920, 1080],
        "orientation": "landscape",
        "locked": {"exposure_s": 0.001, "iso": 320, "focus": 0.7, "wb_locked": True},
        "intrinsics": {"fx": 1000.0, "fy": 1000.0, "cx": 960.0, "cy": 540.0, "dist": [], "source": "manual"},
        "arkit_camera_pose": {"R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], "t": [0.0, 0.0, 12.0]},
        "court_plane": {"point": [0.0, 0.0, 0.0], "normal": [0.0, 0.0, 1.0]},
        "manual_court_taps": [[756.8, 88.4896], [1163.2, 88.4896], [1163.2, 991.5104], [756.8, 991.5104]],
        "gravity": [0.0, -1.0, 0.0],
        "lidar_depth_refs": [],
        "ondevice_pose_track": None,
        "capture_quality": {"grade": "good", "reasons": []},
    }


def _write_inputs(inputs_dir: Path, *, frame_indexes: tuple[int, ...] = (0,)) -> None:
    inputs_dir.mkdir(parents=True)
    (inputs_dir / "capture_sidecar.json").write_text(json.dumps(_sidecar_payload()), encoding="utf-8")
    (inputs_dir / "detections.json").write_text(
        json.dumps(
            {
                "fps": 30.0,
                "frames": [
                    {
                        "frame": frame_idx,
                        "detections": [
                            {"bbox": [940.0, 440.0, 980.0, 540.0], "conf": 0.91, "class": "person", "player_id": 7}
                        ],
                    }
                    for frame_idx in frame_indexes
                ],
            }
        ),
        encoding="utf-8",
    )
    frames = inputs_dir / "body_frames"
    frames.mkdir()
    for frame_idx in frame_indexes:
        (frames / f"frame_{frame_idx:06d}.jpg").write_bytes(b"not decoded by fake runtime")


def _write_frame_compute_plan(run_dir: Path) -> None:
    (run_dir / "frame_compute_plan.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_type": "racketsport_frame_compute_plan",
                "fps": 30.0,
                "expected_players": 1,
                "frame_count": 3,
                "frames": [
                    {
                        "frame_idx": 0,
                        "t": 0.0,
                        "score": 0.8,
                        "recommended_tier": "human_review",
                        "target_representation": "manual_review_required",
                        "reasons": ["missing_expected_players"],
                        "active_players": 1,
                        "active_player_ids": [7],
                        "missing_players": 0,
                        "min_track_conf": 0.91,
                        "ball_conf": None,
                    },
                    {
                        "frame_idx": 1,
                        "t": 1.0 / 30.0,
                        "score": 0.75,
                        "recommended_tier": "deep_mesh",
                        "target_representation": "world_mesh",
                        "reasons": ["contact_window", "ball_uncertain"],
                        "active_players": 1,
                        "active_player_ids": [7],
                        "missing_players": 0,
                        "min_track_conf": 0.91,
                        "ball_conf": 0.22,
                    },
                    {
                        "frame_idx": 2,
                        "t": 2.0 / 30.0,
                        "score": 0.2,
                        "recommended_tier": "skeleton_preview",
                        "target_representation": "joints_or_preview_mesh",
                        "reasons": ["ball_missing"],
                        "active_players": 1,
                        "active_player_ids": [7],
                        "missing_players": 0,
                        "min_track_conf": 0.91,
                        "ball_conf": None,
                    },
                ],
                "deep_mesh_windows": [
                    {
                        "frame_start": 1,
                        "frame_end": 1,
                        "t0": 1.0 / 30.0,
                        "t1": 2.0 / 30.0,
                        "frame_count": 1,
                        "target_representation": "world_mesh",
                        "fallback_representation": "skeleton_preview",
                        "target_player_ids": [7],
                        "reason_counts": {"ball_uncertain": 1, "contact_window": 1},
                        "max_score": 0.75,
                    }
                ],
                "summary": {
                    "by_tier": {"deep_mesh": 1, "human_review": 1, "skeleton_preview": 1},
                    "by_reason": {"ball_missing": 1, "ball_uncertain": 1, "contact_window": 1, "missing_expected_players": 1},
                    "max_score": 0.8,
                    "deep_mesh_window_count": 1,
                    "deep_mesh_frame_count": 1,
                    "human_review_frame_count": 1,
                },
            }
        ),
        encoding="utf-8",
    )


class FakeFastSamRuntime:
    def __init__(self, *, bbox_as_numpy: bool = False) -> None:
        self.calls: list[dict[str, object]] = []
        self.bbox_as_numpy = bbox_as_numpy

    def process_frame(self, image_path: Path, *, bboxes_xyxy: list[list[float]]) -> list[dict[str, object]]:
        bbox: object = bboxes_xyxy[0]
        if self.bbox_as_numpy:
            np = pytest.importorskip("numpy")
            bbox = np.asarray(bbox, dtype=np.float32)
        self.calls.append({"image_path": str(image_path), "bboxes_xyxy": bboxes_xyxy})
        return [
            {
                "bbox": bbox,
                "global_rot": [0.01, 0.02, 0.03],
                "body_pose_params": [0.1] * 63,
                "hand_pose_params": [0.2] * 108,
                "shape_params": [0.0] * 10,
                "pred_cam_t": [0.0, 0.0, 10.0],
                "pred_vertices": [[0.0, 0.0, 0.1], [0.1, 0.0, 1.7]],
                "pred_keypoints_3d": [[0.0, 0.0, 0.2], [0.2, 0.0, 1.4]],
                "confidence": 0.86,
            }
        ]


def test_body_runner_verifies_manifest_uses_yolo26m_and_writes_contract_artifacts(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs"
    run_dir = tmp_path / "run"
    manifest = _manifest(tmp_path / "models")
    _write_inputs(inputs)
    runtime = FakeFastSamRuntime()

    summary = run_pipeline(
        clip="clip_001",
        inputs_dir=inputs,
        run_dir=run_dir,
        stage="body",
        max_frames=1,
        tracking_mode="precomputed",
        runners={"body": BodyStageRunner(manifest_path=manifest, runtime=runtime)},
    )

    assert summary["status"] == "pass"
    body_stage = summary["stages"][2]
    assert body_stage["stage"] == "body"
    assert body_stage["status"] == "ran"
    assert body_stage["real_model"] is True
    assert body_stage["source_mode"] == "fast_sam_3d_body"
    assert body_stage["metrics"]["verified_model_ids"] == [
        "fast_sam_3d_body_dinov3",
        "sam_3d_body_mhr_model",
        "moge_2_vitl_normal",
        "yolo26m",
    ]
    assert body_stage["metrics"]["detector_model_id"] == "yolo26m"
    assert runtime.calls[0]["bboxes_xyxy"] == [[940.0, 440.0, 980.0, 540.0]]

    smpl = validate_artifact_file("smpl_motion", run_dir / "smpl_motion.json")
    skeleton = validate_artifact_file("skeleton3d", run_dir / "skeleton3d.json")
    assert isinstance(smpl, SmplMotion)
    assert isinstance(skeleton, Skeleton3D)
    assert smpl.players[0].id == 7
    assert smpl.players[0].frames[0].transl_world == pytest.approx([0.0, 0.0, 0.0], abs=1e-9)
    assert min(joint[2] for joint in smpl.players[0].frames[0].joints_world) >= 0.0
    assert len(smpl.players[0].frames[0].mesh_vertices_world) == 2
    assert min(vertex[2] for vertex in smpl.players[0].frames[0].mesh_vertices_world) >= 0.0
    assert skeleton.preview_only is True


def test_body_runner_uses_adaptive_frame_plan_for_deep_mesh_work(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs"
    run_dir = tmp_path / "run"
    manifest = _manifest(tmp_path / "models")
    _write_inputs(inputs, frame_indexes=(0, 1, 2))
    run_dir.mkdir()
    _write_frame_compute_plan(run_dir)
    runtime = FakeFastSamRuntime()

    summary = run_pipeline(
        clip="clip_001",
        inputs_dir=inputs,
        run_dir=run_dir,
        stage="body",
        max_frames=None,
        max_players=1,
        tracking_mode="precomputed",
        runners={"body": BodyStageRunner(manifest_path=manifest, runtime=runtime)},
    )

    assert summary["status"] == "pass"
    assert [Path(call["image_path"]).name for call in runtime.calls] == ["frame_000001.jpg"]
    execution = json.loads((run_dir / "body_compute_execution.json").read_text(encoding="utf-8"))
    assert execution["artifact_type"] == "racketsport_body_compute_execution"
    assert execution["mode"] == "adaptive_frame_compute_plan"
    assert execution["summary"]["scheduled_frame_count"] == 1
    assert execution["summary"]["skipped_by_tier"] == {"human_review": 1, "skeleton_preview": 1}
    assert execution["scheduled_frames"][0]["frame_idx"] == 1
    assert execution["scheduled_frames"][0]["target_player_ids"] == [7]
    assert execution["skipped_frames"][0]["skip_reason"] == "manual_review_required"

    smpl = validate_artifact_file("smpl_motion", run_dir / "smpl_motion.json")
    assert isinstance(smpl, SmplMotion)
    assert [frame.t for frame in smpl.players[0].frames] == [pytest.approx(1.0 / 30.0)]


def test_body_runner_accepts_numpy_bbox_from_real_fast_sam_runtime(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs"
    run_dir = tmp_path / "run"
    manifest = _manifest(tmp_path / "models")
    _write_inputs(inputs)

    summary = run_pipeline(
        clip="clip_001",
        inputs_dir=inputs,
        run_dir=run_dir,
        stage="body",
        max_frames=1,
        tracking_mode="precomputed",
        runners={"body": BodyStageRunner(manifest_path=manifest, runtime=FakeFastSamRuntime(bbox_as_numpy=True))},
    )

    assert summary["status"] == "pass"
    assert validate_artifact_file("smpl_motion", run_dir / "smpl_motion.json")


def test_body_runner_fails_loudly_on_detector_sha_mismatch(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs"
    run_dir = tmp_path / "run"
    manifest = _manifest(tmp_path / "models", yolo_sha="0" * 64)
    _write_inputs(inputs)

    summary = run_pipeline(
        clip="clip_001",
        inputs_dir=inputs,
        run_dir=run_dir,
        stage="body",
        max_frames=1,
        tracking_mode="precomputed",
        runners={"body": BodyStageRunner(manifest_path=manifest, runtime=FakeFastSamRuntime())},
    )

    assert summary["status"] == "fail"
    assert summary["stages"][2]["stage"] == "body"
    assert summary["stages"][2]["real_model"] is True
    assert any("sha256 mismatch for yolo26m" in note for note in summary["stages"][2]["notes"])
    execution = json.loads((run_dir / "body_compute_execution.json").read_text(encoding="utf-8"))
    assert execution["artifact_type"] == "racketsport_body_compute_execution"
    assert execution["mode"] == "adaptive_frame_compute_plan"
    assert execution["summary"]["scheduled_frame_count"] == 0
    assert not (run_dir / "smpl_motion.json").exists()

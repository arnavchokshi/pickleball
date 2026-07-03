from __future__ import annotations

import json
from pathlib import Path

import pytest

from threed.racketsport.orchestrator import BodyStageRunner, StageRun, run_pipeline
from threed.racketsport.schemas import ContactWindows, Skeleton3D, SmplMotion, validate_artifact_file


def _assert_pipeline_semantically_blocked(summary: dict, *expected_blockers: str) -> None:
    assert summary["status"] == "blocked"
    assert summary["readiness"]["status"] == "not_ready"
    for blocker in expected_blockers:
        assert blocker in summary["readiness"]["semantic_blockers"]


def _stage(summary: dict, stage: str) -> dict:
    return next(item for item in summary["stages"] if item["stage"] == stage)


def _manifest(root: Path, *, moge_sha: str | None = None, yolo_sha: str | None = None) -> Path:
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
            _entry("moge_2_vitl_normal", "camera_fov_depth_prior", moge, sha256=moge_sha),
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


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
    np = pytest.importorskip("numpy")
    cv2 = pytest.importorskip("cv2")
    frames = inputs_dir / "body_frames"
    frames.mkdir()
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    for frame_idx in frame_indexes:
        assert cv2.imwrite(str(frames / f"frame_{frame_idx:06d}.jpg"), frame)


def _write_multi_player_inputs(inputs_dir: Path) -> None:
    inputs_dir.mkdir(parents=True)
    (inputs_dir / "capture_sidecar.json").write_text(json.dumps(_sidecar_payload()), encoding="utf-8")
    (inputs_dir / "detections.json").write_text(
        json.dumps(
            {
                "fps": 30.0,
                "frames": [
                    {
                        "frame": 0,
                        "detections": [
                            {"bbox": [940.0, 440.0, 980.0, 540.0], "conf": 0.91, "class": "person", "player_id": 7},
                            {"bbox": [1040.0, 430.0, 1090.0, 560.0], "conf": 0.88, "class": "person", "player_id": 8},
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    np = pytest.importorskip("numpy")
    cv2 = pytest.importorskip("cv2")
    frames = inputs_dir / "body_frames"
    frames.mkdir()
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    assert cv2.imwrite(str(frames / "frame_000000.jpg"), frame)


def _write_scaled_body_frame_inputs(inputs_dir: Path) -> None:
    np = pytest.importorskip("numpy")
    cv2 = pytest.importorskip("cv2")
    sidecar = _sidecar_payload()
    sidecar["resolution"] = [960, 540]
    sidecar["intrinsics"] = {"fx": 500.0, "fy": 500.0, "cx": 480.0, "cy": 270.0, "dist": [], "source": "manual"}
    sidecar["manual_court_taps"] = [[100.0, 50.0], [860.0, 50.0], [860.0, 490.0], [100.0, 490.0]]
    inputs_dir.mkdir(parents=True)
    (inputs_dir / "capture_sidecar.json").write_text(json.dumps(sidecar), encoding="utf-8")
    (inputs_dir / "detections.json").write_text(
        json.dumps(
            {
                "fps": 30.0,
                "frames": [
                    {
                        "frame": 0,
                        "detections": [
                            {"bbox": [100.0, 100.0, 200.0, 300.0], "conf": 0.91, "class": "person", "player_id": 7}
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    frames = inputs_dir / "body_frames"
    frames.mkdir()
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    assert cv2.imwrite(str(frames / "frame_000000.jpg"), frame)


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


def _write_deep_mesh_frame_compute_plan(
    run_dir: Path,
    *,
    frame_idx: int = 0,
    target_player_ids: tuple[int, ...] = (7,),
    expected_players: int = 1,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    player_targets = [
        {
            "player_id": player_id,
            "track_conf": 0.91,
            "score": 0.75,
            "recommended_tier": "deep_mesh",
            "target_representation": "world_mesh",
            "reasons": ["contact_window"],
        }
        for player_id in target_player_ids
    ]
    (run_dir / "frame_compute_plan.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_type": "racketsport_frame_compute_plan",
                "fps": 30.0,
                "expected_players": expected_players,
                "frame_count": 1,
                "frames": [
                    {
                        "frame_idx": frame_idx,
                        "t": frame_idx / 30.0,
                        "score": 0.75,
                        "recommended_tier": "deep_mesh",
                        "target_representation": "world_mesh",
                        "reasons": ["contact_window"],
                        "active_players": len(target_player_ids),
                        "active_player_ids": list(target_player_ids),
                        "missing_players": 0,
                        "min_track_conf": 0.91,
                        "ball_conf": None,
                        "player_targets": player_targets,
                    }
                ],
                "deep_mesh_windows": [
                    {
                        "frame_start": frame_idx,
                        "frame_end": frame_idx,
                        "t0": frame_idx / 30.0,
                        "t1": (frame_idx + 1) / 30.0,
                        "frame_count": 1,
                        "target_representation": "world_mesh",
                        "fallback_representation": "skeleton_preview",
                        "target_player_ids": list(target_player_ids),
                        "reason_counts": {"contact_window": 1},
                        "max_score": 0.75,
                    }
                ],
                "summary": {
                    "by_tier": {"deep_mesh": 1},
                    "by_reason": {"contact_window": 1},
                    "by_player_target_representation": {"world_mesh": len(target_player_ids)},
                    "max_score": 0.75,
                    "deep_mesh_window_count": 1,
                    "deep_mesh_frame_count": 1,
                    "human_review_frame_count": 0,
                },
            }
        ),
        encoding="utf-8",
    )


def _write_sam3d_contact_skeleton(run_dir: Path) -> None:
    joint_names = [f"sam3dbody_joint_{idx:03d}" for idx in range(70)]
    frames = [
        {
            "frame_idx": 0,
            "t": 0.0,
            "joints_world": _sam3d_fixture_joints(left_wrist_x=0.0, right_wrist_x=0.0),
            "joint_conf": [0.9] * 70,
        },
        {
            "frame_idx": 1,
            "t": 1.0 / 30.0,
            "joints_world": _sam3d_fixture_joints(left_wrist_x=0.5, right_wrist_x=0.0),
            "joint_conf": [0.9] * 70,
        },
        {
            "frame_idx": 2,
            "t": 2.0 / 30.0,
            "joints_world": _sam3d_fixture_joints(left_wrist_x=0.55, right_wrist_x=0.0),
            "joint_conf": [0.9] * 70,
        },
    ]
    _write_json(
        run_dir / "skeleton3d.json",
        {
            "schema_version": 1,
            "artifact_type": "racketsport_skeleton3d",
            "fps": 30.0,
            "world_frame": "court_Z0",
            "source_model": "sam3d_body_joints",
            "joint_names": joint_names,
            "preview_only": False,
            "players": [{"id": 7, "frames": frames}],
            "provenance": {
                "lane": "BODY_TIER2",
                "source": "sam3d_body_joints",
                "model_family": "sam3dbody_world_joints",
                "protected_eval_labels_used": False,
            },
        },
    )


def _sam3d_fixture_joints(*, left_wrist_x: float, right_wrist_x: float) -> list[list[float]]:
    joints = [[0.0, 0.0, 1.0] for _idx in range(70)]
    joints[5] = [-0.2, 0.0, 1.4]
    joints[6] = [0.2, 0.0, 1.4]
    joints[7] = [-0.35, 0.0, 1.2]
    joints[8] = [0.35, 0.0, 1.2]
    joints[9] = [-0.15, -0.1, 1.0]
    joints[10] = [0.15, -0.1, 1.0]
    joints[11] = [-0.12, -0.1, 0.5]
    joints[12] = [0.12, -0.1, 0.5]
    joints[13] = [-0.1, -0.1, 0.05]
    joints[14] = [0.1, -0.1, 0.05]
    joints[41] = [right_wrist_x, 0.0, 1.2]
    joints[62] = [left_wrist_x, 0.0, 1.2]
    return joints


def _write_contact_ball_inflections(inputs: Path) -> None:
    _write_json(
        inputs / "ball_inflections.json",
        {
            "schema_version": 1,
            "artifact_type": "racketsport_ball_inflections",
            "source": "test_fixture_ball_turn",
            "not_gate_verified": True,
            "warnings": ["test_fixture"],
            "summary": {"candidate_count": 1},
            "candidates": [
                {
                    "time_s": 1.0 / 30.0,
                    "frame": 1,
                    "ball_world_xyz": [0.48, 0.0, 1.2],
                    "confidence": 0.74,
                }
            ],
        },
    )


def _write_stale_mesh_wrist_velocity_peaks(inputs: Path) -> None:
    _write_json(
        inputs / "wrist_velocity_peaks.json",
        {
            "schema_version": 1,
            "artifact_type": "racketsport_wrist_velocity_peaks",
            "status": "review_only",
            "source": "mesh_only_sam3dbody_world_joints",
            "source_path": str(inputs / "body_mesh.json"),
            "not_gate_verified": True,
            "trusted_for_contact": False,
            "joint_mapping": {"left_wrist": 62},
            "blockers": [],
            "warnings": ["stale_mesh_only_fixture"],
            "summary": {
                "player_count": 1,
                "usable_sample_count": 3,
                "raw_peak_count": 1,
                "peak_count": 1,
                "min_speed_mps": 4.0,
                "min_confidence": 0.25,
                "min_separation_s": 0.1,
            },
            "peaks": [
                {
                    "time_s": 1.0 / 30.0,
                    "frame": 1,
                    "player_id": 99,
                    "wrist_side": "left",
                    "wrist_world_xyz": [9.0, 9.0, 1.2],
                    "speed_mps": 12.0,
                    "confidence": 0.95,
                    "source": "mesh_only_world_joints",
                }
            ],
        },
    )


class FakeFastSamRuntime:
    def __init__(self, *, bbox_as_numpy: bool = False, joint_count: int = 70) -> None:
        self.calls: list[dict[str, object]] = []
        self.bbox_as_numpy = bbox_as_numpy
        self.joint_count = joint_count

    def process_frame(
        self,
        image_path: Path,
        *,
        bboxes_xyxy: list[list[float]],
        **_kwargs: object,
    ) -> list[dict[str, object]]:
        bbox: object = bboxes_xyxy[0]
        if self.bbox_as_numpy:
            np = pytest.importorskip("numpy")
            bbox = np.asarray(bbox, dtype=np.float32)
        self.calls.append({"image_path": str(image_path), "bboxes_xyxy": bboxes_xyxy})
        joints = [[0.02 * idx, 0.0, 0.2 + 0.05 * (idx % 12)] for idx in range(self.joint_count)]
        if self.joint_count >= 70:
            joints[41] = [0.41, 0.0, 1.3]
            joints[62] = [0.62, 0.0, 1.4]
        return [
            {
                "bbox": bbox,
                "global_rot": [0.01, 0.02, 0.03],
                "body_pose_params": [0.1] * 63,
                "hand_pose_params": [0.2] * 108,
                "shape_params": [0.0] * 10,
                "pred_cam_t": [0.0, 0.0, 10.0],
                "pred_vertices": [[0.0, 0.0, 0.1], [0.1, 0.0, 1.7], [0.1, 0.2, 0.9]],
                "mesh_faces": [[0, 1, 2]],
                "pred_keypoints_3d": joints,
                "confidence": 0.86,
            }
        ]


class FakeBatchFastSamRuntime(FakeFastSamRuntime):
    def __init__(self) -> None:
        super().__init__()
        self.batch_calls: list[dict[str, object]] = []

    def process_frame(
        self,
        image_path: Path,
        *,
        bboxes_xyxy: list[list[float]],
        **_kwargs: object,
    ) -> list[dict[str, object]]:
        raise AssertionError("batch-capable runtime should not use process_frame")

    def process_frame_batches(self, requests: list[dict[str, object]], **_kwargs: object) -> list[list[dict[str, object]]]:
        self.batch_calls.append(
            {
                "image_paths": [str(request["image_path"]) for request in requests],
                "bboxes_xyxy": [request["bboxes"] for request in requests],
            }
        )
        return [
            FakeFastSamRuntime.process_frame(
                self,
                Path(str(request["image_path"])),
                bboxes_xyxy=request["bboxes"],  # type: ignore[arg-type]
            )
            for request in requests
        ]


class FakeMissingFastSamRuntime:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def process_frame(
        self,
        image_path: Path,
        *,
        bboxes_xyxy: list[list[float]],
        **_kwargs: object,
    ) -> list[dict[str, object]]:
        self.calls.append({"image_path": str(image_path), "bboxes_xyxy": bboxes_xyxy})
        return []


class FakePoseStageRunner:
    stage = "pose"
    real_model = True
    source_mode = "test_lane_a_pose"

    def run(self, context) -> StageRun:
        skeleton_path = context.run_dir / "skeleton3d.json"
        if not skeleton_path.is_file():
            _write_json(
                skeleton_path,
                {
                    "schema_version": 1,
                    "artifact_type": "racketsport_skeleton3d",
                    "fps": 30.0,
                    "world_frame": "court_Z0",
                    "source_model": "test_pose_runner",
                    "joint_names": ["pelvis"],
                    "preview_only": True,
                    "players": [
                        {
                            "id": 7,
                            "frames": [
                                {
                                    "frame_idx": 0,
                                    "t": 0.0,
                                    "joints_world": [[0.0, 0.0, 1.0]],
                                    "joint_conf": [0.9],
                                }
                            ],
                        }
                    ],
                    "provenance": {"lane": "A", "test_fixture": True},
                },
            )
        return StageRun(
            stage=self.stage,
            status="ran",
            real_model=True,
            source_mode=self.source_mode,
            produced_artifacts=("skeleton3d.json",),
            notes=("test Lane A skeleton fixture",),
        )


def test_body_runner_verifies_manifest_uses_yolo26m_and_writes_contract_artifacts(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs"
    run_dir = tmp_path / "run"
    manifest = _manifest(tmp_path / "models")
    _write_inputs(inputs)
    _write_deep_mesh_frame_compute_plan(run_dir)
    runtime = FakeFastSamRuntime()

    summary = run_pipeline(
        clip="clip_001",
        inputs_dir=inputs,
        run_dir=run_dir,
        stage="body",
        max_frames=1,
        tracking_mode="precomputed",
        runners={"pose": FakePoseStageRunner(), "body": BodyStageRunner(manifest_path=manifest, runtime=runtime)},
    )

    _assert_pipeline_semantically_blocked(
        summary,
        "calibration:court_line_evidence_not_ready",
        "body:body_mesh_world_mesh_unverified",
        "body:body_mesh_not_trusted_for_promotion",
    )
    body_stage = _stage(summary, "body")
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
    assert body_stage["metrics"]["max_root_speed_mps"] == pytest.approx(8.0)
    assert runtime.calls[0]["bboxes_xyxy"] == [[940.0, 440.0, 980.0, 540.0]]

    smpl = validate_artifact_file("smpl_motion", run_dir / "smpl_motion.json")
    skeleton = validate_artifact_file("skeleton3d", run_dir / "skeleton3d.json")
    body_mesh = json.loads((run_dir / "body_mesh.json").read_text(encoding="utf-8"))
    assert isinstance(smpl, SmplMotion)
    assert isinstance(skeleton, Skeleton3D)
    assert body_mesh["artifact_type"] == "racketsport_body_mesh"
    assert body_mesh["mesh_faces"] == [[0, 1, 2]]
    assert body_mesh["summary"]["mesh_frame_count"] == 1
    assert body_mesh["players"][0]["frames"][0]["mesh_vertices_world"]
    assert smpl.players[0].id == 7
    assert smpl.mesh_faces == [(0, 1, 2)]
    assert smpl.players[0].frames[0].transl_world == pytest.approx([0.0, 0.0, 0.0], abs=1e-9)
    assert min(joint[2] for joint in smpl.players[0].frames[0].joints_world) >= 0.0
    assert len(smpl.players[0].frames[0].mesh_vertices_world) == 3
    assert min(vertex[2] for vertex in smpl.players[0].frames[0].mesh_vertices_world) >= 0.0
    assert skeleton.preview_only is False
    assert skeleton.source_model == "sam3d_body_joints"
    assert len(skeleton.joint_names) == 70
    body_joint_quality = json.loads((run_dir / "body_joint_quality.json").read_text(encoding="utf-8"))
    assert body_joint_quality["artifact_type"] == "racketsport_body_joint_quality"
    assert body_joint_quality["status"] == "quality_checked_needs_accuracy_gate"
    assert body_joint_quality["summary"]["joint_frame_count"] == 1
    assert body_joint_quality["summary"]["joint_count_min"] == 70


def test_body_runner_scales_track_bboxes_to_materialized_body_frame_size(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs"
    run_dir = tmp_path / "run"
    manifest = _manifest(tmp_path / "models")
    _write_scaled_body_frame_inputs(inputs)
    _write_deep_mesh_frame_compute_plan(run_dir)
    runtime = FakeFastSamRuntime()

    summary = run_pipeline(
        clip="clip_001",
        inputs_dir=inputs,
        run_dir=run_dir,
        stage="body",
        max_frames=1,
        tracking_mode="precomputed",
        runners={
            "pose": FakePoseStageRunner(),
            "body": BodyStageRunner(manifest_path=manifest, runtime=runtime, detector_name="", fov_name=""),
        },
    )

    assert _stage(summary, "body")["status"] == "ran"
    assert runtime.calls[0]["bboxes_xyxy"] == [[200.0, 200.0, 400.0, 600.0]]


def test_body_runner_default_caps_root_speed_for_smoother_world_alignment(tmp_path: Path) -> None:
    runner = BodyStageRunner(manifest_path=tmp_path / "MANIFEST.json", runtime=FakeFastSamRuntime())

    assert runner.smoothing_alpha == pytest.approx(1.0)
    assert runner.max_root_speed_mps == pytest.approx(8.0)


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
        runners={"pose": FakePoseStageRunner(), "body": BodyStageRunner(manifest_path=manifest, runtime=runtime)},
    )

    _assert_pipeline_semantically_blocked(
        summary,
        "calibration:court_line_evidence_not_ready",
        "body:body_mesh_world_mesh_unverified",
        "body:body_mesh_not_trusted_for_promotion",
    )
    assert [Path(call["image_path"]).name for call in runtime.calls] == ["frame_000001.jpg"]


def test_body_runner_derives_body_plan_from_sam3d_wrist_peaks(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs"
    run_dir = tmp_path / "run"
    manifest = _manifest(tmp_path / "models")
    _write_inputs(inputs, frame_indexes=(0, 1, 2))
    run_dir.mkdir()
    _write_sam3d_contact_skeleton(run_dir)
    _write_contact_ball_inflections(inputs)
    runtime = FakeFastSamRuntime()

    summary = run_pipeline(
        clip="clip_001",
        inputs_dir=inputs,
        run_dir=run_dir,
        stage="body",
        max_frames=None,
        max_players=1,
        tracking_mode="precomputed",
        runners={
            "pose": FakePoseStageRunner(),
            "body": BodyStageRunner(manifest_path=manifest, runtime=runtime, detector_name="", fov_name=""),
        },
    )

    body_stage = _stage(summary, "body")
    assert body_stage["stage"] == "body"
    assert body_stage["status"] == "ran"
    assert body_stage["metrics"]["lane_b_frame_plan_source"] == "sam3d_wrist_velocity_peaks"
    assert body_stage["metrics"]["lane_b_contact_event_count"] == 1
    assert [Path(call["image_path"]).name for call in runtime.calls] == [
        "frame_000000.jpg",
        "frame_000001.jpg",
        "frame_000002.jpg",
    ]

    wrist_peaks = json.loads((run_dir / "wrist_velocity_peaks.json").read_text(encoding="utf-8"))
    contact_windows = validate_artifact_file("contact_windows", run_dir / "contact_windows.json")
    frame_plan = json.loads((run_dir / "frame_compute_plan.json").read_text(encoding="utf-8"))
    execution = json.loads((run_dir / "body_compute_execution.json").read_text(encoding="utf-8"))
    full_clip_gate = json.loads((run_dir / "body_full_clip_gate.json").read_text(encoding="utf-8"))
    mesh_readiness = json.loads((run_dir / "body_mesh_readiness.json").read_text(encoding="utf-8"))
    joint_quality = json.loads((run_dir / "body_joint_quality.json").read_text(encoding="utf-8"))
    skeleton = validate_artifact_file("skeleton3d", run_dir / "skeleton3d.json")

    assert wrist_peaks["source"] == "sam3d_body_skeleton3d_world_joints"
    assert wrist_peaks["source_provenance"]["source_model"] == "sam3d_body_joints"
    assert wrist_peaks["source_provenance"]["joint_count"] == 70
    assert wrist_peaks["summary"]["peak_count"] == 1
    assert isinstance(contact_windows, ContactWindows)
    assert contact_windows.events[0].player_id == 7
    assert contact_windows.events[0].sources.wrist_vel == pytest.approx(0.9)
    assert frame_plan["deep_mesh_windows"][0]["target_player_ids"] == [7]
    assert frame_plan["summary"]["deep_mesh_window_count"] == 1
    assert execution["summary"]["scheduled_frame_count"] == 3
    assert full_clip_gate["artifact_type"] == "racketsport_body_full_clip_gate"
    assert full_clip_gate["passed"] is True
    assert full_clip_gate["min_coverage"] == pytest.approx(0.98)
    assert full_clip_gate["coverage"] == pytest.approx(1.0)
    assert full_clip_gate["contact_mesh_coverage"] == pytest.approx(1.0)
    assert full_clip_gate["latency_seconds_per_video_minute"] is not None
    assert full_clip_gate["summary"]["scheduled_contact_count"] == 3
    assert full_clip_gate["summary"]["contact_mesh_frame_count"] == 3
    assert full_clip_gate["summary"]["mesh_unavailable_contact_count"] == 0
    assert full_clip_gate["summary"]["contact_mesh_accounted_count"] == 3
    assert full_clip_gate["paths"]["contact_splice"] == str(run_dir / "contact_splice.json")
    assert mesh_readiness["body_full_clip_gate_path"] == str(run_dir / "body_full_clip_gate.json")
    assert "missing_full_clip_body_gate" not in mesh_readiness["blockers"]
    assert "missing_full_clip_body_gate" not in joint_quality["promotion_blockers"]
    assert isinstance(skeleton, Skeleton3D)
    assert skeleton.preview_only is False


def test_body_runner_regenerates_non_lane_a_input_wrist_peaks(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs"
    run_dir = tmp_path / "run"
    manifest = _manifest(tmp_path / "models")
    _write_inputs(inputs, frame_indexes=(0, 1, 2))
    _write_stale_mesh_wrist_velocity_peaks(inputs)
    run_dir.mkdir()
    _write_sam3d_contact_skeleton(run_dir)
    _write_contact_ball_inflections(inputs)
    runtime = FakeFastSamRuntime()

    summary = run_pipeline(
        clip="clip_001",
        inputs_dir=inputs,
        run_dir=run_dir,
        stage="body",
        max_frames=None,
        max_players=1,
        tracking_mode="precomputed",
        runners={
            "pose": FakePoseStageRunner(),
            "body": BodyStageRunner(manifest_path=manifest, runtime=runtime, detector_name="", fov_name=""),
        },
    )

    body_stage = _stage(summary, "body")
    wrist_peaks = json.loads((run_dir / "wrist_velocity_peaks.json").read_text(encoding="utf-8"))
    contact_windows = validate_artifact_file("contact_windows", run_dir / "contact_windows.json")

    assert body_stage["status"] == "ran"
    assert body_stage["metrics"]["lane_b_frame_plan_source"] == "sam3d_wrist_velocity_peaks"
    assert "wrist_velocity_peaks.json" in body_stage["metrics"]["lane_b_frame_plan_generated_artifacts"]
    assert wrist_peaks["source"] == "sam3d_body_skeleton3d_world_joints"
    assert wrist_peaks["source_path"] == str(run_dir / "skeleton3d.json")
    assert wrist_peaks["source_provenance"]["source_model"] == "sam3d_body_joints"
    assert wrist_peaks["source_provenance"]["joint_count"] == 70
    assert wrist_peaks["summary"]["peak_count"] == 1
    assert isinstance(contact_windows, ContactWindows)
    assert contact_windows.events[0].player_id == 7
    assert [Path(call["image_path"]).name for call in runtime.calls] == [
        "frame_000000.jpg",
        "frame_000001.jpg",
        "frame_000002.jpg",
    ]


def test_body_runner_replaces_existing_legacy_skeleton3d_with_sam3d_output(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs"
    run_dir = tmp_path / "run"
    manifest = _manifest(tmp_path / "models")
    _write_inputs(inputs, frame_indexes=(0, 1, 2))
    run_dir.mkdir()
    _write_frame_compute_plan(run_dir)
    _write_json(
        run_dir / "skeleton3d.json",
        {
            "schema_version": 1,
            "artifact_type": "racketsport_skeleton3d",
            "fps": 30.0,
            "world_frame": "court_Z0",
            "source_model": "rtmw3d_x",
            "joint_names": ["pelvis", "left_wrist", "right_wrist"],
            "preview_only": False,
            "players": [
                {
                    "id": 7,
                    "frames": [
                        {
                            "frame_idx": 0,
                            "t": 0.0,
                            "joints_world": [[0.0, 0.0, 1.0], [-0.2, 0.0, 1.2], [0.2, 0.0, 1.2]],
                            "joint_conf": [0.9, 0.9, 0.9],
                        },
                        {
                            "frame_idx": 1,
                            "t": 1.0 / 30.0,
                            "joints_world": [[0.0, 0.0, 1.0], [-0.25, 0.0, 1.2], [0.25, 0.0, 1.2]],
                            "joint_conf": [0.9, 0.9, 0.9],
                        },
                    ],
                }
            ],
            "provenance": {"lane": "A"},
        },
    )
    runtime = FakeFastSamRuntime(joint_count=70)

    run_pipeline(
        clip="clip_001",
        inputs_dir=inputs,
        run_dir=run_dir,
        stage="body",
        max_frames=None,
        max_players=1,
        tracking_mode="precomputed",
        runners={"pose": FakePoseStageRunner(), "body": BodyStageRunner(manifest_path=manifest, runtime=runtime)},
    )

    skeleton = validate_artifact_file("skeleton3d", run_dir / "skeleton3d.json")
    assert isinstance(skeleton, Skeleton3D)
    assert skeleton.preview_only is False
    assert skeleton.source_model == "sam3d_body_joints"
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
    skeleton_payload = json.loads((run_dir / "skeleton3d.json").read_text(encoding="utf-8"))
    body_mesh = json.loads((run_dir / "body_mesh.json").read_text(encoding="utf-8"))
    contact_splice = json.loads((run_dir / "contact_splice.json").read_text(encoding="utf-8"))
    contact_frame = skeleton_payload["players"][0]["frames"][0]
    body_mesh_frame = body_mesh["players"][0]["frames"][0]
    assert body_mesh["joint_names"][41] == "sam3dbody_joint_041"
    assert contact_frame["joints_world"][62] == pytest.approx(body_mesh_frame["joints_world"][62])
    assert contact_frame["joints_world"][41] == pytest.approx(body_mesh_frame["joints_world"][41])
    assert contact_splice["summary"]["spliced_contact_count"] == 1
    assert contact_splice["summary"]["overridden_joint_count"] == 2
    assert skeleton_payload["provenance"]["contact_splice"]["mesh_source"] == "body_mesh.json"


def test_body_runner_does_not_backfill_missing_contact_mesh_with_pose_fallback(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs"
    run_dir = tmp_path / "run"
    manifest = _manifest(tmp_path / "models")
    _write_inputs(inputs, frame_indexes=(0,))
    run_dir.mkdir()
    _write_deep_mesh_frame_compute_plan(run_dir, frame_idx=0)
    _write_sam3d_contact_skeleton(run_dir)
    runtime = FakeMissingFastSamRuntime()

    summary = run_pipeline(
        clip="clip_001",
        inputs_dir=inputs,
        run_dir=run_dir,
        stage="body",
        max_frames=None,
        max_players=1,
        tracking_mode="precomputed",
        runners={
            "pose": FakePoseStageRunner(),
            "body": BodyStageRunner(
                manifest_path=manifest,
                runtime=runtime,
                detector_name="",
                fov_name="",
            ),
        },
    )

    body_stage = _stage(summary, "body")
    assert body_stage["status"] == "ran", body_stage
    assert body_stage["metrics"]["body_mesh_frame_count"] == 0
    assert body_stage["metrics"]["sam3d_missing_output_count"] == 1
    assert body_stage["metrics"]["contact_splice_mesh_unavailable_count"] == 1
    assert body_stage["metrics"]["contact_splice_fallback_spliced_count"] == 0
    assert len(runtime.calls) == 1

    body_mesh = json.loads((run_dir / "body_mesh.json").read_text(encoding="utf-8"))
    skeleton_payload = json.loads((run_dir / "skeleton3d.json").read_text(encoding="utf-8"))
    contact_splice = json.loads((run_dir / "contact_splice.json").read_text(encoding="utf-8"))

    assert body_mesh["summary"]["mesh_frame_count"] == 0
    assert not (run_dir / "body_pose_fallback.json").exists()
    assert contact_splice["events"][0]["status"] == "mesh_unavailable"
    assert contact_splice["events"][0]["mesh_unavailable"] is True
    contact_frame = skeleton_payload["players"][0]["frames"][0]
    assert contact_frame["joint_conf"][62] == pytest.approx(0.9)


def test_body_runner_leaves_missing_outputs_when_fast_sam_runtime_is_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = tmp_path / "inputs"
    run_dir = tmp_path / "run"
    manifest = _manifest(tmp_path / "models")
    fast_sam_repo = tmp_path / "fast-sam"
    _write_inputs(inputs, frame_indexes=(0,))
    run_dir.mkdir()
    fast_sam_repo.mkdir()
    _write_deep_mesh_frame_compute_plan(run_dir, frame_idx=0)
    _write_sam3d_contact_skeleton(run_dir)
    monkeypatch.delenv("FAST_SAM_PYTHON", raising=False)

    class BrokenFastSamRuntime:
        def __init__(self, **_kwargs: object) -> None:
            raise RuntimeError("could not import FastSAM-3D-Body notebook.utils.setup_sam_3d_body")

    monkeypatch.setattr("threed.racketsport.orchestrator.FastSam3DBodyRuntime", BrokenFastSamRuntime)

    summary = run_pipeline(
        clip="clip_001",
        inputs_dir=inputs,
        run_dir=run_dir,
        stage="body",
        max_frames=None,
        max_players=1,
        tracking_mode="precomputed",
        runners={
            "pose": FakePoseStageRunner(),
            "body": BodyStageRunner(
                manifest_path=manifest,
                fast_sam_repo=fast_sam_repo,
                detector_name="",
                fov_name="",
            ),
        },
    )

    body_stage = _stage(summary, "body")
    assert body_stage["status"] == "ran", body_stage
    assert body_stage["metrics"]["fast_sam_runtime_unavailable"] is True
    assert body_stage["metrics"]["body_mesh_frame_count"] == 0
    assert body_stage["metrics"]["sam3d_missing_output_count"] == 1
    assert body_stage["metrics"]["contact_splice_mesh_unavailable_count"] == 1
    assert any("Fast SAM-3D-Body runtime unavailable" in note for note in body_stage["notes"])

    body_mesh = json.loads((run_dir / "body_mesh.json").read_text(encoding="utf-8"))
    contact_splice = json.loads((run_dir / "contact_splice.json").read_text(encoding="utf-8"))

    assert body_mesh["summary"]["mesh_frame_count"] == 0
    assert not (run_dir / "body_pose_fallback.json").exists()
    assert contact_splice["events"][0]["status"] == "mesh_unavailable"
    assert contact_splice["events"][0]["mesh_unavailable"] is True


def test_body_runner_can_disable_detector_and_fov_asset_checks_for_tracked_bboxes(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs"
    run_dir = tmp_path / "run"
    manifest = _manifest(tmp_path / "models", moge_sha="1" * 64, yolo_sha="0" * 64)
    _write_inputs(inputs)
    _write_deep_mesh_frame_compute_plan(run_dir)
    runtime = FakeFastSamRuntime()

    summary = run_pipeline(
        clip="clip_001",
        inputs_dir=inputs,
        run_dir=run_dir,
        stage="body",
        max_frames=1,
        tracking_mode="precomputed",
        runners={
            "pose": FakePoseStageRunner(),
            "body": BodyStageRunner(manifest_path=manifest, runtime=runtime, detector_name="", fov_name=""),
        },
    )

    body_stage = _stage(summary, "body")
    assert body_stage["stage"] == "body"
    assert body_stage["status"] == "ran"
    assert body_stage["metrics"]["verified_model_ids"] == [
        "fast_sam_3d_body_dinov3",
        "sam_3d_body_mhr_model",
    ]
    assert body_stage["metrics"]["detector_model_id"] == ""
    assert body_stage["metrics"]["detector_model_path"] == ""
    assert body_stage["metrics"]["fov_model_id"] == ""
    assert runtime.calls[0]["bboxes_xyxy"] == [[940.0, 440.0, 980.0, 540.0]]


def test_body_runner_uses_fast_sam_subprocess_runtime_when_python_env_is_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = tmp_path / "inputs"
    run_dir = tmp_path / "run"
    manifest = _manifest(tmp_path / "models")
    fast_sam_repo = tmp_path / "fast-sam"
    fast_sam_repo.mkdir()
    _write_inputs(inputs)
    _write_deep_mesh_frame_compute_plan(run_dir)
    runtime_inits: list[dict[str, object]] = []

    class FakeSubprocessRuntime:
        def __init__(self, **kwargs: object) -> None:
            runtime_inits.append(kwargs)
            self.calls: list[dict[str, object]] = []

        def process_frame(
            self,
            image_path: Path,
            *,
            bboxes_xyxy: list[list[float]],
            **_kwargs: object,
        ) -> list[dict[str, object]]:
            self.calls.append({"image_path": str(image_path), "bboxes_xyxy": bboxes_xyxy})
            return [
                {
                    "bbox": bboxes_xyxy[0],
                    "body_pose_params": [0.1] * 133,
                    "hand_pose_params": [0.2] * 108,
                    "shape_params": [0.0] * 10,
                    "pred_cam_t": [0.0, 0.0, 10.0],
                    "pred_vertices": [[0.0, 0.0, 0.1], [0.1, 0.0, 1.7], [0.1, 0.2, 0.9]],
                    "mesh_faces": [[0, 1, 2]],
                    "pred_keypoints_3d": [[0.02 * idx, 0.0, 0.2 + 0.05 * (idx % 12)] for idx in range(70)],
                    "confidence": 0.86,
                }
            ]

    monkeypatch.setenv("FAST_SAM_PYTHON", "/opt/fast-sam/bin/python")
    monkeypatch.setattr("threed.racketsport.orchestrator.FastSam3DBodySubprocessRuntime", FakeSubprocessRuntime, raising=False)

    summary = run_pipeline(
        clip="clip_001",
        inputs_dir=inputs,
        run_dir=run_dir,
        stage="body",
        max_frames=1,
        tracking_mode="precomputed",
        runners={
            "pose": FakePoseStageRunner(),
            "body": BodyStageRunner(manifest_path=manifest, fast_sam_repo=fast_sam_repo, detector_name="", fov_name=""),
        },
    )

    body_stage = _stage(summary, "body")
    assert body_stage["status"] == "ran"
    assert runtime_inits
    assert runtime_inits[0]["python_executable"] == "/opt/fast-sam/bin/python"
    assert runtime_inits[0]["fast_sam_repo"] == fast_sam_repo
    assert runtime_inits[0]["detector_name"] == ""
    assert runtime_inits[0]["fov_name"] == ""
    assert body_stage["metrics"]["body_mesh_frame_count"] == 1


def test_body_runner_processes_multi_player_bboxes_individually_to_preserve_identity(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs"
    run_dir = tmp_path / "run"
    manifest = _manifest(tmp_path / "models")
    _write_multi_player_inputs(inputs)
    _write_deep_mesh_frame_compute_plan(run_dir, target_player_ids=(7, 8), expected_players=2)
    runtime = FakeFastSamRuntime()

    summary = run_pipeline(
        clip="clip_001",
        inputs_dir=inputs,
        run_dir=run_dir,
        stage="body",
        max_frames=1,
        tracking_mode="precomputed",
        runners={
            "pose": FakePoseStageRunner(),
            "body": BodyStageRunner(manifest_path=manifest, runtime=runtime, detector_name="", fov_name=""),
        },
    )

    _assert_pipeline_semantically_blocked(
        summary,
        "calibration:court_line_evidence_not_ready",
        "body:body_mesh_world_mesh_unverified",
        "body:body_mesh_not_trusted_for_promotion",
    )
    assert [call["bboxes_xyxy"] for call in runtime.calls] == [
        [[940.0, 440.0, 980.0, 540.0]],
        [[1040.0, 430.0, 1090.0, 560.0]],
    ]
    smpl = validate_artifact_file("smpl_motion", run_dir / "smpl_motion.json")
    assert isinstance(smpl, SmplMotion)
    assert [player.id for player in smpl.players] == [7, 8]


def test_body_runner_batches_fast_sam_subprocess_requests_without_merging_player_identity(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs"
    run_dir = tmp_path / "run"
    manifest = _manifest(tmp_path / "models")
    _write_multi_player_inputs(inputs)
    _write_deep_mesh_frame_compute_plan(run_dir, target_player_ids=(7, 8), expected_players=2)
    runtime = FakeBatchFastSamRuntime()

    summary = run_pipeline(
        clip="clip_001",
        inputs_dir=inputs,
        run_dir=run_dir,
        stage="body",
        max_frames=1,
        tracking_mode="precomputed",
        runners={
            "pose": FakePoseStageRunner(),
            "body": BodyStageRunner(manifest_path=manifest, runtime=runtime, detector_name="", fov_name=""),
        },
    )

    _assert_pipeline_semantically_blocked(
        summary,
        "calibration:court_line_evidence_not_ready",
        "body:body_mesh_world_mesh_unverified",
        "body:body_mesh_not_trusted_for_promotion",
    )
    assert len(runtime.batch_calls) == 1
    assert runtime.batch_calls[0]["bboxes_xyxy"] == [
        [[940.0, 440.0, 980.0, 540.0]],
        [[1040.0, 430.0, 1090.0, 560.0]],
    ]
    smpl = validate_artifact_file("smpl_motion", run_dir / "smpl_motion.json")
    assert isinstance(smpl, SmplMotion)
    assert [player.id for player in smpl.players] == [7, 8]


def test_body_runner_accepts_numpy_bbox_from_real_fast_sam_runtime(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs"
    run_dir = tmp_path / "run"
    manifest = _manifest(tmp_path / "models")
    _write_inputs(inputs)
    _write_deep_mesh_frame_compute_plan(run_dir)

    summary = run_pipeline(
        clip="clip_001",
        inputs_dir=inputs,
        run_dir=run_dir,
        stage="body",
        max_frames=1,
        tracking_mode="precomputed",
        runners={
            "pose": FakePoseStageRunner(),
            "body": BodyStageRunner(manifest_path=manifest, runtime=FakeFastSamRuntime(bbox_as_numpy=True)),
        },
    )

    _assert_pipeline_semantically_blocked(
        summary,
        "calibration:court_line_evidence_not_ready",
        "body:body_mesh_world_mesh_unverified",
        "body:body_mesh_not_trusted_for_promotion",
    )
    assert validate_artifact_file("smpl_motion", run_dir / "smpl_motion.json")


def test_body_runner_fails_loudly_on_detector_sha_mismatch(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs"
    run_dir = tmp_path / "run"
    manifest = _manifest(tmp_path / "models", yolo_sha="0" * 64)
    _write_inputs(inputs)
    _write_deep_mesh_frame_compute_plan(run_dir)

    summary = run_pipeline(
        clip="clip_001",
        inputs_dir=inputs,
        run_dir=run_dir,
        stage="body",
        max_frames=1,
        tracking_mode="precomputed",
        runners={"pose": FakePoseStageRunner(), "body": BodyStageRunner(manifest_path=manifest, runtime=FakeFastSamRuntime())},
    )

    assert summary["status"] == "fail"
    body_stage = _stage(summary, "body")
    assert body_stage["real_model"] is True
    assert any("sha256 mismatch for yolo26m" in note for note in body_stage["notes"])
    execution = json.loads((run_dir / "body_compute_execution.json").read_text(encoding="utf-8"))
    assert execution["artifact_type"] == "racketsport_body_compute_execution"
    assert execution["mode"] == "adaptive_frame_compute_plan"
    assert execution["summary"]["scheduled_frame_count"] == 0
    assert not (run_dir / "smpl_motion.json").exists()

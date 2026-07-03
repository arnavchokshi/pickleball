from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from tests.racketsport.calibration_fixtures import minimal_calibration_image_pts, minimal_calibration_world_pts
from threed.racketsport import hmr_deep
from threed.racketsport import orchestrator
from threed.racketsport.schemas import CameraIntrinsics, CaptureQuality, CourtCalibration, CourtExtrinsics, ReprojectionError


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _calibration_payload() -> dict[str, Any]:
    calibration = CourtCalibration(
        schema_version=1,
        sport="pickleball",
        homography=[[100.0, 0.0, 1000.0], [0.0, 100.0, 1000.0], [0.0, 0.0, 1.0]],
        intrinsics=CameraIntrinsics(fx=1000.0, fy=1000.0, cx=960.0, cy=540.0, dist=[], source="manual"),
        extrinsics=CourtExtrinsics(
            R=[[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
            t=[0.0, 0.0, 0.0],
            camera_height_m=1.5,
        ),
        reprojection_error_px=ReprojectionError(median=0.0, p95=0.0),
        capture_quality=CaptureQuality(grade="good", reasons=[]),
        image_pts=minimal_calibration_image_pts(),
        world_pts=minimal_calibration_world_pts(),
    )
    return calibration.model_dump(mode="json")


def _tracks_payload() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "fps": 30.0,
        "players": [
            {
                "id": 7,
                "side": "near",
                "role": "left",
                "frames": [
                    {"t": 0.0, "bbox": [100.0, 100.0, 200.0, 300.0], "world_xy": [1.0, 2.0], "conf": 0.9},
                    {"t": 1.0 / 30.0, "bbox": [102.0, 100.0, 202.0, 300.0], "world_xy": [1.1, 2.0], "conf": 0.9},
                ],
            }
        ],
        "rally_spans": [],
    }


def _frame_plan_payload() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_frame_compute_plan",
        "fps": 30.0,
        "expected_players": 1,
        "frame_count": 2,
        "frames": [
            {
                "frame_idx": 0,
                "t": 0.0,
                "score": 0.1,
                "recommended_tier": "baseline",
                "target_representation": "lane_a_skeleton",
                "reasons": [],
                "active_player_ids": [7],
                "player_targets": [],
            },
            {
                "frame_idx": 1,
                "t": 1.0 / 30.0,
                "score": 0.9,
                "recommended_tier": "deep_mesh",
                "target_representation": "world_mesh",
                "reasons": ["ball_aware_contact"],
                "active_player_ids": [7],
                "player_targets": [
                    {
                        "player_id": 7,
                        "track_conf": 0.9,
                        "score": 0.9,
                        "recommended_tier": "deep_mesh",
                        "target_representation": "world_mesh",
                        "reasons": ["ball_aware_contact"],
                    }
                ],
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
                "fallback_representation": "body_joints",
                "target_player_ids": [7],
                "reason_counts": {"ball_aware_contact": 1},
                "max_score": 0.9,
            }
        ],
        "summary": {"by_tier": {"deep_mesh": 1}, "deep_mesh_frame_count": 1},
    }


class _FakeSam3DRuntime:
    def __init__(self) -> None:
        self.requests: list[Any] = []
        self.batch_kwargs: dict[str, Any] = {}

    def process_frame_batches(self, requests: list[Any], **kwargs: Any) -> list[list[dict[str, Any]]]:
        self.requests = requests
        self.batch_kwargs = dict(kwargs)
        return [
            [
                {
                    "pred_keypoints_3d": [[0.0, 0.0, 0.0], [0.0, 1.6, 0.0], [0.3, 0.8, 0.0]],
                    "pred_vertices": [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 1.6, 0.0]],
                    "mesh_faces": [[0, 1, 2]],
                    "confidence": 0.9,
                }
            ]
            for _request in requests
        ]


def test_body_runner_serializes_mesh_vertices_only_for_tier1_frames(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    run_dir = tmp_path / "clip"
    (run_dir / "body_frames").mkdir(parents=True)
    (run_dir / "body_frames" / "frame_000000.jpg").write_bytes(b"not-a-real-jpeg")
    (run_dir / "body_frames" / "frame_000001.jpg").write_bytes(b"not-a-real-jpeg")
    (run_dir / "body_frames" / "frame_000001.jpg").write_bytes(b"not-a-real-jpeg")
    _write_json(run_dir / "tracks.json", _tracks_payload())
    _write_json(run_dir / "court_calibration.json", _calibration_payload())
    _write_json(run_dir / "frame_compute_plan.json", _frame_plan_payload())
    monkeypatch.setattr(
        orchestrator,
        "verify_fast_sam_manifest_assets",
        lambda *args, **kwargs: {"fast_sam_3d_body_dinov3": SimpleNamespace(path=tmp_path / "model.ckpt")},
    )
    monkeypatch.setattr(orchestrator, "_read_image_size", lambda _path: (1920, 1080))

    runtime = _FakeSam3DRuntime()
    runner = orchestrator.BodyStageRunner(
        runtime=runtime,
        tier2_body_joints_all_tracked=True,
        mesh_vertex_serialization_policy="tier1_only",
        sam3d_body_input_size_px=384,
        sam3d_crop_bucket_sizes=(8, 16),
        sam3d_torch_compile=True,
        sam3d_compile_warmup_buckets=(8, 16),
    )
    result = runner.run(
        orchestrator.StageContext(
            clip="clip",
            inputs_dir=run_dir,
            run_dir=run_dir,
            sport="pickleball",
            expected_players=1,
        )
    )

    assert result.status == "ran"
    smpl_motion = json.loads((run_dir / "smpl_motion.json").read_text(encoding="utf-8"))
    frames = {frame["frame_idx"]: frame for frame in smpl_motion["players"][0]["frames"]}
    assert frames[0]["joints_world"]
    assert frames[0]["mesh_vertices_world"] == []
    assert frames[1]["joints_world"]
    assert len(frames[1]["mesh_vertices_world"]) == 3
    body_mesh = json.loads((run_dir / "body_mesh.json").read_text(encoding="utf-8"))
    mesh_frames = body_mesh["players"][0]["frames"]
    assert [frame["frame_idx"] for frame in mesh_frames] == [1]
    skeleton3d = json.loads((run_dir / "skeleton3d.json").read_text(encoding="utf-8"))
    assert skeleton3d["source_model"] == "sam3d_body_joints"
    assert skeleton3d["players"][0]["frames"][0]["confidence_provenance"]["source"] == "sam3d_body_joints"
    config = json.loads((run_dir / "sam3d_tier2_config.json").read_text(encoding="utf-8"))
    assert config["serialization"]["mesh_vertex_serialization_policy"] == "tier1_only"
    assert config["optimization"]["crop_bucket_sizes"] == [8, 16]
    assert config["optimization"]["static_clip_intrinsics"] is True
    assert config["optimization"]["compile_warmup_static_intrinsics"] is True
    assert config["optimization"]["compile_stall_regression_target_s"] == 1.0
    assert config["optimization"]["batching"] == "static_intrinsics_cross_frame_bucketed_body_batch"
    assert config["optimization"]["steady_state_empty_cache"] is True
    assert config["optimization"]["inner_bucket_sync"] is True
    assert config["optimization"]["upstream_env"] == {}
    assert config["optimization"]["tier2_output_lite"] is False
    assert runtime.batch_kwargs["torch_compile"] is True
    assert runtime.batch_kwargs["compile_warmup_buckets"] == (8, 16)
    assert runtime.batch_kwargs["crop_bucket_sizes"] == (8, 16)
    assert runtime.batch_kwargs["sam3d_body_input_size_px"] == 384
    assert runtime.batch_kwargs["steady_state_empty_cache"] is True
    assert runtime.batch_kwargs["inner_bucket_sync"] is True
    assert runtime.batch_kwargs["upstream_env"] == {}
    assert runtime.batch_kwargs["tier2_output_lite"] is False
    assert runtime.batch_kwargs["clip_intrinsics"] == {
        "fx": 1000.0,
        "fy": 1000.0,
        "cx": 960.0,
        "cy": 540.0,
        "dist": [],
        "source": "manual",
        "matrix": [[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]],
        "static_per_clip": True,
    }


class _LiteSam3DRuntime:
    def __init__(self) -> None:
        self.requests: list[Any] = []
        self.batch_kwargs: dict[str, Any] = {}

    def process_frame_batches(self, requests: list[Any], **kwargs: Any) -> list[list[dict[str, Any]]]:
        self.requests = requests
        self.batch_kwargs = dict(kwargs)
        outputs: list[list[dict[str, Any]]] = []
        for request in requests:
            record = {
                "pred_keypoints_3d": [[0.0, 0.0, 0.0], [0.0, 1.6, 0.0], [0.3, 0.8, 0.0]],
                "pred_keypoints_2d": [[0.0, 0.0], [0.0, 1.0], [0.3, 0.8]],
                "pred_cam_t": [0.0, 0.0, 1.0],
                "confidence": 0.9,
            }
            if request["target_representation"] == "world_mesh":
                record["pred_vertices"] = [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 1.6, 0.0]]
                record["mesh_faces"] = [[0, 1, 2]]
            outputs.append([record])
        return outputs


def test_body_runner_output_lite_tolerates_tier2_without_dense_mesh_fields(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    run_dir = tmp_path / "clip"
    (run_dir / "body_frames").mkdir(parents=True)
    (run_dir / "body_frames" / "frame_000000.jpg").write_bytes(b"not-a-real-jpeg")
    (run_dir / "body_frames" / "frame_000001.jpg").write_bytes(b"not-a-real-jpeg")
    _write_json(run_dir / "tracks.json", _tracks_payload())
    _write_json(run_dir / "court_calibration.json", _calibration_payload())
    _write_json(run_dir / "frame_compute_plan.json", _frame_plan_payload())
    monkeypatch.setattr(
        orchestrator,
        "verify_fast_sam_manifest_assets",
        lambda *args, **kwargs: {"fast_sam_3d_body_dinov3": SimpleNamespace(path=tmp_path / "model.ckpt")},
    )
    monkeypatch.setattr(orchestrator, "_read_image_size", lambda _path: (1920, 1080))

    runtime = _LiteSam3DRuntime()
    runner = orchestrator.BodyStageRunner(
        runtime=runtime,
        tier2_body_joints_all_tracked=True,
        mesh_vertex_serialization_policy="tier1_only",
        sam3d_crop_bucket_sizes=(16, 24, 32, 48, 64),
        sam3d_torch_compile=True,
        sam3d_compile_warmup_buckets=(16, 24, 32, 48, 64),
        sam3d_steady_state_empty_cache=False,
        sam3d_inner_bucket_sync=False,
        sam3d_upstream_env={"USE_COMPILE_BACKBONE": "1", "COMPILE_MODE": "reduce-overhead"},
        sam3d_tier2_output_lite=True,
    )
    result = runner.run(
        orchestrator.StageContext(
            clip="clip",
            inputs_dir=run_dir,
            run_dir=run_dir,
            sport="pickleball",
            expected_players=1,
        )
    )

    assert result.status == "ran"
    assert runtime.batch_kwargs["crop_bucket_sizes"] == (16, 24, 32, 48, 64)
    assert runtime.batch_kwargs["compile_warmup_buckets"] == (16, 24, 32, 48, 64)
    assert runtime.batch_kwargs["steady_state_empty_cache"] is False
    assert runtime.batch_kwargs["inner_bucket_sync"] is False
    assert runtime.batch_kwargs["upstream_env"] == {"USE_COMPILE_BACKBONE": "1", "COMPILE_MODE": "reduce-overhead"}
    assert runtime.batch_kwargs["tier2_output_lite"] is True
    assert {request["target_representation"] for request in runtime.requests} == {"body_joints", "world_mesh"}

    smpl_motion = json.loads((run_dir / "smpl_motion.json").read_text(encoding="utf-8"))
    frames = {frame["frame_idx"]: frame for frame in smpl_motion["players"][0]["frames"]}
    assert frames[0]["joints_world"]
    assert frames[0]["mesh_vertices_world"] == []
    assert frames[1]["joints_world"]
    assert len(frames[1]["mesh_vertices_world"]) == 3
    config = json.loads((run_dir / "sam3d_tier2_config.json").read_text(encoding="utf-8"))
    assert config["optimization"]["crop_bucket_sizes"] == [16, 24, 32, 48, 64]
    assert config["optimization"]["compile_warmup_buckets"] == [16, 24, 32, 48, 64]
    assert config["optimization"]["steady_state_empty_cache"] is False
    assert config["optimization"]["inner_bucket_sync"] is False
    assert config["optimization"]["upstream_env"] == {"USE_COMPILE_BACKBONE": "1", "COMPILE_MODE": "reduce-overhead"}
    assert config["optimization"]["tier2_output_lite"] is True


def test_static_clip_intrinsics_payload_uses_scaled_matrix_values() -> None:
    calibration = CourtCalibration.model_validate(_calibration_payload())

    payload = orchestrator._sam3d_static_clip_intrinsics_payload(
        calibration=calibration,
        camera_intrinsics_k=[[500.0, 0.0, 480.0], [0.0, 500.0, 270.0], [0.0, 0.0, 1.0]],
    )

    assert payload == {
        "fx": 500.0,
        "fy": 500.0,
        "cx": 480.0,
        "cy": 270.0,
        "dist": [],
        "source": "manual",
        "matrix": [[500.0, 0.0, 480.0], [0.0, 500.0, 270.0], [0.0, 0.0, 1.0]],
        "static_per_clip": True,
    }


def test_subprocess_batch_runtime_payload_carries_speed_flags_and_target_representations(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    image = tmp_path / "frame_000000.jpg"
    image.write_bytes(b"not-a-real-jpeg")
    runtime = hmr_deep.FastSam3DBodySubprocessRuntime(
        python_executable="/usr/bin/python3",
        fast_sam_repo=tmp_path / "Fast-SAM-3D-Body",
        checkpoint_dir=tmp_path / "checkpoints",
        detector_name="",
        fov_name="",
        work_dir=tmp_path / "work",
    )
    captured: dict[str, Any] = {}

    def fake_run(command, check, capture_output, text):  # noqa: ANN001
        request_path = Path(command[command.index("--requests") + 1])
        out_path = Path(command[command.index("--out") + 1])
        captured["payload"] = json.loads(request_path.read_text(encoding="utf-8"))
        out_path.write_text(
            json.dumps(
                {
                    "frames": [
                        {"request_id": "0:7", "records": []},
                        {"request_id": "1:7", "records": []},
                    ]
                }
            ),
            encoding="utf-8",
        )
        return hmr_deep.subprocess.CompletedProcess(args=command, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(hmr_deep.subprocess, "run", fake_run)

    outputs = runtime.process_frame_batches(
        [
            {
                "request_id": "0:7",
                "image_path": image,
                "bboxes": [[10.0, 20.0, 110.0, 220.0]],
                "target_representation": "body_joints",
            },
            {
                "request_id": "1:7",
                "image_path": image,
                "bboxes": [[12.0, 22.0, 112.0, 222.0]],
                "target_representation": "world_mesh",
            },
        ],
        crop_bucket_sizes=(16, 24, 32, 48, 64),
        torch_compile=True,
        compile_warmup_buckets=(16, 24, 32, 48, 64),
        steady_state_empty_cache=False,
        inner_bucket_sync=False,
        upstream_env={"USE_COMPILE_BACKBONE": "1", "COMPILE_MODE": "reduce-overhead"},
        tier2_output_lite=True,
    )

    assert outputs == [[], []]
    assert captured["payload"]["optimization"]["crop_bucket_sizes"] == [16, 24, 32, 48, 64]
    assert captured["payload"]["optimization"]["compile_warmup_buckets"] == [16, 24, 32, 48, 64]
    assert captured["payload"]["optimization"]["steady_state_empty_cache"] is False
    assert captured["payload"]["optimization"]["inner_bucket_sync"] is False
    assert captured["payload"]["optimization"]["upstream_env"] == {
        "USE_COMPILE_BACKBONE": "1",
        "COMPILE_MODE": "reduce-overhead",
    }
    assert captured["payload"]["optimization"]["tier2_output_lite"] is True
    assert [request["target_representation"] for request in captured["payload"]["requests"]] == [
        "body_joints",
        "world_mesh",
    ]


def test_body_runner_fails_closed_when_body_frame_sizes_change(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    run_dir = tmp_path / "clip"
    (run_dir / "body_frames").mkdir(parents=True)
    (run_dir / "body_frames" / "frame_000000.jpg").write_bytes(b"not-a-real-jpeg")
    (run_dir / "body_frames" / "frame_000001.jpg").write_bytes(b"not-a-real-jpeg")
    _write_json(run_dir / "tracks.json", _tracks_payload())
    _write_json(run_dir / "court_calibration.json", _calibration_payload())
    _write_json(run_dir / "frame_compute_plan.json", _frame_plan_payload())
    monkeypatch.setattr(
        orchestrator,
        "verify_fast_sam_manifest_assets",
        lambda *args, **kwargs: {"fast_sam_3d_body_dinov3": SimpleNamespace(path=tmp_path / "model.ckpt")},
    )
    sizes = {
        "frame_000000.jpg": (1920, 1080),
        "frame_000001.jpg": (1280, 720),
    }
    monkeypatch.setattr(orchestrator, "_read_image_size", lambda path: sizes[Path(path).name])

    runner = orchestrator.BodyStageRunner(
        runtime=_FakeSam3DRuntime(),
        tier2_body_joints_all_tracked=True,
        mesh_vertex_serialization_policy="tier1_only",
    )

    try:
        runner.run(
            orchestrator.StageContext(
                clip="clip",
                inputs_dir=run_dir,
                run_dir=run_dir,
                sport="pickleball",
                expected_players=1,
            )
        )
    except ValueError as exc:
        assert "BODY frame image size changed" in str(exc)
        assert "frame_000001.jpg" in str(exc)
    else:  # pragma: no cover - expectation path
        raise AssertionError("mismatched BODY frame dimensions must fail closed")


def test_body_runner_refuses_unreadable_body_frame_size_fallback(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    run_dir = tmp_path / "clip"
    (run_dir / "body_frames").mkdir(parents=True)
    (run_dir / "body_frames" / "frame_000000.jpg").write_bytes(b"not-a-real-jpeg")
    _write_json(run_dir / "tracks.json", _tracks_payload())
    _write_json(run_dir / "court_calibration.json", _calibration_payload())
    _write_json(run_dir / "frame_compute_plan.json", _frame_plan_payload())
    monkeypatch.setattr(
        orchestrator,
        "verify_fast_sam_manifest_assets",
        lambda *args, **kwargs: {"fast_sam_3d_body_dinov3": SimpleNamespace(path=tmp_path / "model.ckpt")},
    )
    monkeypatch.setattr(orchestrator, "_read_image_size", lambda _path: None)

    runner = orchestrator.BodyStageRunner(
        runtime=_FakeSam3DRuntime(),
        tier2_body_joints_all_tracked=True,
        mesh_vertex_serialization_policy="tier1_only",
    )

    try:
        runner.run(
            orchestrator.StageContext(
                clip="clip",
                inputs_dir=run_dir,
                run_dir=run_dir,
                sport="pickleball",
                expected_players=1,
            )
        )
    except ValueError as exc:
        assert "unable to read BODY frame image size" in str(exc)
        assert "refusing to derive SAM3D static intrinsics" in str(exc)
    else:  # pragma: no cover - expectation path
        raise AssertionError("unreadable BODY frame size must fail closed")


def test_body_runner_prepares_static_intrinsics_padded_crop_and_mask_prompt(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    run_dir = tmp_path / "clip"
    (run_dir / "body_frames").mkdir(parents=True)
    (run_dir / "body_frames" / "frame_000000.jpg").write_bytes(b"not-a-real-jpeg")
    (run_dir / "body_frames" / "frame_000001.jpg").write_bytes(b"not-a-real-jpeg")
    mask_path = run_dir / "sam3d_body_masks" / "frame_000001_player_7.png"
    mask_path.parent.mkdir(parents=True)
    mask_path.write_bytes(b"mask-bytes")
    _write_json(run_dir / "sam3d_body_mask_prompts.json", {
        "schema_version": 1,
        "artifact_type": "racketsport_sam3d_body_mask_prompts",
        "frames": [
            {
                "frame_idx": 1,
                "player_id": 7,
                "mask_path": "sam3d_body_masks/frame_000001_player_7.png",
                "bucket": "contact",
                "source": "sam2_box_prompt",
            }
        ],
    })
    _write_json(run_dir / "tracks.json", _tracks_payload())
    _write_json(run_dir / "court_calibration.json", _calibration_payload())
    _write_json(run_dir / "frame_compute_plan.json", _frame_plan_payload())
    monkeypatch.setattr(
        orchestrator,
        "verify_fast_sam_manifest_assets",
        lambda *args, **kwargs: {"fast_sam_3d_body_dinov3": SimpleNamespace(path=tmp_path / "model.ckpt")},
    )
    monkeypatch.setattr(orchestrator, "_read_image_size", lambda _path: (1920, 1080))

    runtime = _FakeSam3DRuntime()
    runner = orchestrator.BodyStageRunner(
        runtime=runtime,
        tier2_body_joints_all_tracked=True,
        mesh_vertex_serialization_policy="tier1_only",
        sam3d_body_input_size_px=448,
        sam3d_crop_padding_scale=1.20,
        sam3d_mask_prompt_mode="manifest",
        sam3d_soft_background_alpha=0.65,
    )
    result = runner.run(
        orchestrator.StageContext(
            clip="clip",
            inputs_dir=run_dir,
            run_dir=run_dir,
            sport="pickleball",
            expected_players=1,
        )
    )

    assert result.status == "ran"
    assert runtime.requests
    request_by_frame = {request["frame_idx"]: request for request in runtime.requests}
    tier2_request = request_by_frame[0]
    mesh_request = request_by_frame[1]
    assert tier2_request["mask_paths"] == []
    assert mesh_request["mask_paths"] == [mask_path]
    assert mesh_request["bboxes"] == [[92.0, 80.0, 212.0, 320.0]]
    assert mesh_request["camera_intrinsics"] == [[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]]
    assert mesh_request["camera_intrinsics_source"] == "court_calibration.json"
    assert mesh_request["sam3d_body_input_size_px"] == 448
    assert mesh_request["crop_padding_scale"] == 1.20
    assert mesh_request["soft_background_alpha"] == 0.65

    prep = json.loads((run_dir / "sam3d_body_input_prep.json").read_text(encoding="utf-8"))
    assert prep["artifact_type"] == "racketsport_sam3d_body_input_prep"
    assert prep["camera_intrinsics"]["static_per_clip"] is True
    assert prep["camera_intrinsics"]["K"] == [[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]]
    assert prep["mask_prompts"]["mode"] == "manifest"
    assert prep["mask_prompts"]["available_count"] == 1
    assert prep["mask_prompts"]["missing_count"] == 1

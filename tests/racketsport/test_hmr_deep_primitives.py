from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from threed.racketsport.hmr_deep import (
    FastSam3DBodyRuntime,
    FastSam3DBodySubprocessRuntime,
    PlayerCropRequest,
    VerifiedModelAsset,
    _ensure_torch_amp_custom_decorators,
    _ensure_torch_dynamo_accumulated_cache_limit,
    _load_setup_sam_3d_body,
    build_player_hmr_artifact,
    gate_deep_hmr_artifact,
    normalize_fast_sam_body_output,
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


def test_normalize_fast_sam_body_output_preserves_static_mesh_faces_for_mhr_surface() -> None:
    request = PlayerCropRequest(
        frame_idx=5,
        player_id=7,
        bbox_xyxy=[100, 120, 260, 520],
        image_size_px=[1920, 1080],
        track_confidence=0.93,
    )

    normalized = normalize_fast_sam_body_output(
        {
            "pred_vertices": [[0, 0, 0.1], [0.3, 0, 0.1], [0.3, 0.2, 1.6]],
            "pred_keypoints_3d": [[0.1, 0.0, 1.0]],
            "faces": [[0, 1, 2]],
            "confidence": 0.91,
        },
        request=request,
    )

    assert normalized["mesh_faces"] == [[0, 1, 2]]


def test_normalize_fast_sam_body_output_applies_pred_cam_t_exactly_once() -> None:
    request = PlayerCropRequest(
        frame_idx=5,
        player_id=7,
        bbox_xyxy=[100, 120, 260, 520],
        image_size_px=[1920, 1080],
        track_confidence=0.93,
    )
    raw = normalize_fast_sam_body_output(
        {
            "pred_vertices": [[4.0, 5.0, 6.0]],
            "pred_keypoints_3d": [[1.0, 2.0, 3.0]],
            "pred_cam_t": [0.25, -0.5, 1.0],
            "confidence": 0.91,
        },
        request=request,
    )

    assert raw["camera_translation"] == [0.25, -0.5, 1.0]
    assert raw["joints_camera"] == [[1.25, 1.5, 4.0]]
    assert raw["vertices_camera"] == [[4.25, 4.5, 7.0]]

    already_translated = normalize_fast_sam_body_output(
        {
            "pred_vertices": [[4.25, 4.5, 7.0]],
            "pred_keypoints_3d": [[1.25, 1.5, 4.0]],
            "pred_cam_t": [0.25, -0.5, 1.0],
            "pred_cam_t_already_applied": True,
            "confidence": 0.91,
        },
        request=request,
    )

    assert already_translated["camera_translation"] == [0.25, -0.5, 1.0]
    assert already_translated["joints_camera"] == [[1.25, 1.5, 4.0]]
    assert already_translated["vertices_camera"] == [[4.25, 4.5, 7.0]]


def test_normalize_fast_sam_body_output_keeps_compact_foot_keypoints_only_by_default() -> None:
    request = PlayerCropRequest(
        frame_idx=5,
        player_id=7,
        bbox_xyxy=[100, 120, 260, 520],
        image_size_px=[1920, 1080],
        track_confidence=0.93,
    )
    keypoints = [[float(idx), float(idx + 100)] for idx in range(70)]

    normalized = normalize_fast_sam_body_output(
        {
            "pred_vertices": [[0, 0, 0.1], [0.3, 0, 0.1], [0.3, 0.2, 1.6]],
            "pred_keypoints_3d": [[0.1, 0.0, 1.0]],
            "pred_keypoints_2d": keypoints,
            "confidence": 0.91,
        },
        request=request,
    )

    assert sorted(item["index"] for item in normalized["pred_foot_keypoints_2d"]) == [13, 14, 15, 16, 17, 20]
    assert "pred_keypoints_2d" not in normalized


def test_normalize_fast_sam_body_output_rejects_faces_outside_pred_vertices() -> None:
    request = PlayerCropRequest(
        frame_idx=5,
        player_id=7,
        bbox_xyxy=[100, 120, 260, 520],
        image_size_px=[1920, 1080],
        track_confidence=0.93,
    )

    with pytest.raises(ValueError, match="mesh_faces/0 index 3 is outside pred_vertices"):
        normalize_fast_sam_body_output(
            {
                "pred_vertices": [[0, 0, 0.1], [0.3, 0, 0.1], [0.3, 0.2, 1.6]],
                "pred_keypoints_3d": [[0.1, 0.0, 1.0]],
                "mesh_faces": [[0, 1, 3]],
                "confidence": 0.91,
            },
            request=request,
        )


def test_fast_sam_runtime_allows_bbox_only_setup_without_detector_asset(tmp_path, monkeypatch) -> None:
    repo = tmp_path / "fast-sam"
    checkpoint_dir = tmp_path / "sam-3d-body-dinov3"
    repo.mkdir()
    checkpoint_dir.mkdir()
    checkpoint = checkpoint_dir / "model.ckpt"
    checkpoint.write_bytes(b"body")

    calls = []

    def fake_loader(path):
        assert path == repo

        def fake_setup(**kwargs):
            calls.append(kwargs)
            return object()

        return fake_setup

    monkeypatch.setattr("threed.racketsport.hmr_deep._load_setup_sam_3d_body", fake_loader)

    runtime = FastSam3DBodyRuntime(
        assets={
            "fast_sam_3d_body_dinov3": VerifiedModelAsset(
                model_id="fast_sam_3d_body_dinov3",
                path=checkpoint,
                sha256="x",
            ),
            "sam_3d_body_mhr_model": VerifiedModelAsset(
                model_id="sam_3d_body_mhr_model",
                path=checkpoint_dir / "assets" / "mhr_model.pt",
                sha256="y",
            ),
        },
        fast_sam_repo=repo,
        detector_name="",
        fov_name="",
    )

    assert runtime.detector_name == ""
    assert runtime.detector_model is None
    assert runtime.fov_name == ""
    assert calls == [
        {
            "hf_repo_id": "facebook/sam-3d-body-dinov3",
            "detector_name": "",
            "detector_model": "",
            "fov_name": "",
            "local_checkpoint_path": str(checkpoint_dir),
        }
    ]


def test_fast_sam_subprocess_runtime_returns_frame_records(tmp_path, monkeypatch) -> None:
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"jpg")
    repo = tmp_path / "fast-sam"
    repo.mkdir()
    checkpoint_dir = tmp_path / "sam-3d-body-dinov3"
    checkpoint_dir.mkdir()
    work_dir = tmp_path / "work"
    commands: list[list[str]] = []

    def fake_run(command, check, capture_output, text):
        commands.append([str(value) for value in command])
        out_path = command[command.index("--out") + 1]
        Path(out_path).write_text(
            """
{
  "schema_version": 1,
  "artifact_type": "racketsport_sam3dbody_frame",
  "records": [
    {
      "bbox": [1.0, 2.0, 3.0, 4.0],
      "pred_vertices": [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0]],
      "pred_keypoints_3d": [[0.0, 0.0, 1.0]],
      "mesh_faces": [[0, 1, 2]]
    }
  ]
}
""",
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stdout=str(out_path), stderr="")

    monkeypatch.setattr("threed.racketsport.hmr_deep.subprocess.run", fake_run)

    runtime = FastSam3DBodySubprocessRuntime(
        python_executable=tmp_path / "fast_sam_python",
        fast_sam_repo=repo,
        checkpoint_dir=checkpoint_dir,
        detector_name="",
        fov_name="",
        work_dir=work_dir,
    )
    records = runtime.process_frame(image, bboxes_xyxy=[[1.0, 2.0, 3.0, 4.0]])

    assert records[0]["mesh_faces"] == [[0, 1, 2]]
    assert commands
    command = commands[0]
    assert command[0] == str(tmp_path / "fast_sam_python")
    assert any(part.endswith("scripts/racketsport/run_sam3dbody_frame.py") for part in command)
    assert command[command.index("--bbox") + 1] == "1.0,2.0,3.0,4.0"
    assert command[command.index("--detector-name") + 1] == ""
    assert command[command.index("--fov-name") + 1] == ""


def test_fast_sam_subprocess_runtime_batches_frame_requests_in_one_process(tmp_path, monkeypatch) -> None:
    image_a = tmp_path / "frame_a.jpg"
    image_b = tmp_path / "frame_b.jpg"
    image_a.write_bytes(b"jpg")
    image_b.write_bytes(b"jpg")
    repo = tmp_path / "fast-sam"
    repo.mkdir()
    checkpoint_dir = tmp_path / "sam-3d-body-dinov3"
    checkpoint_dir.mkdir()
    work_dir = tmp_path / "work"
    commands: list[list[str]] = []

    def fake_run(command, check, capture_output, text):
        commands.append([str(value) for value in command])
        requests_path = Path(command[command.index("--requests") + 1])
        out_path = Path(command[command.index("--out") + 1])
        requests = __import__("json").loads(requests_path.read_text(encoding="utf-8"))["requests"]
        out_path.write_text(
            __import__("json").dumps(
                {
                    "schema_version": 1,
                    "artifact_type": "racketsport_sam3dbody_batch",
                    "frames": [
                        {
                            "request_id": request["request_id"],
                            "records": [
                                {
                                    "bbox": request["bboxes"][0],
                                    "pred_vertices": [[float(idx), 0.0, 0.0] for idx in range(3)],
                                    "pred_keypoints_3d": [[0.0, 0.0, 1.0]],
                                    "mesh_faces": [[0, 1, 2]],
                                }
                            ],
                        }
                        for request in requests
                    ],
                }
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stdout=str(out_path), stderr="")

    monkeypatch.setattr("threed.racketsport.hmr_deep.subprocess.run", fake_run)

    runtime = FastSam3DBodySubprocessRuntime(
        python_executable=tmp_path / "fast_sam_python",
        fast_sam_repo=repo,
        checkpoint_dir=checkpoint_dir,
        detector_name="",
        fov_name="",
        work_dir=work_dir,
    )
    records = runtime.process_frame_batches(
        [
            (image_a, [[1.0, 2.0, 3.0, 4.0]]),
            (image_b, [[5.0, 6.0, 7.0, 8.0]]),
        ]
    )

    assert [frame_records[0]["bbox"] for frame_records in records] == [
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
    ]
    assert len(commands) == 1
    command = commands[0]
    assert command[0] == str(tmp_path / "fast_sam_python")
    assert any(part.endswith("scripts/racketsport/run_sam3dbody_batch.py") for part in command)
    requests_path = Path(command[command.index("--requests") + 1])
    payload = __import__("json").loads(requests_path.read_text(encoding="utf-8"))
    assert [request["image"] for request in payload["requests"]] == [str(image_a), str(image_b)]
    assert command[command.index("--detector-name") + 1] == ""
    assert command[command.index("--fov-name") + 1] == ""


def test_fast_sam_loader_falls_back_to_direct_runtime_when_notebook_import_fails(tmp_path, monkeypatch) -> None:
    repo = tmp_path / "fast-sam"
    repo.mkdir()
    imported: list[str] = []

    def fake_import_module(name: str):
        imported.append(name)
        if name == "notebook.utils":
            raise ModuleNotFoundError("notebook dependency unavailable")
        if name == "sam_3d_body":
            return object()
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr("threed.racketsport.hmr_deep.importlib.import_module", fake_import_module)

    setup = _load_setup_sam_3d_body(repo)

    assert setup.__name__ == "_direct_setup_sam_3d_body"
    assert imported == ["notebook.utils", "sam_3d_body"]


def test_fast_sam_direct_setup_installs_torch_amp_compat_decorators() -> None:
    calls: list[tuple[str, object]] = []

    class FakeCudaAmp:
        @staticmethod
        def custom_fwd(fwd=None, *, cast_inputs=None):
            calls.append(("custom_fwd", cast_inputs))
            return fwd if fwd is not None else (lambda fn: fn)

        @staticmethod
        def custom_bwd(bwd):
            calls.append(("custom_bwd", None))
            return bwd

    fake_torch = SimpleNamespace(amp=SimpleNamespace(), cuda=SimpleNamespace(amp=FakeCudaAmp()))

    _ensure_torch_amp_custom_decorators(fake_torch)

    def forward():
        return "forward"

    def backward():
        return "backward"

    assert fake_torch.amp.custom_fwd(device_type="cuda", cast_inputs="float32")(forward)() == "forward"
    assert fake_torch.amp.custom_bwd(device_type="cuda")(backward)() == "backward"
    assert calls == [("custom_fwd", "float32"), ("custom_bwd", None)]


def test_fast_sam_direct_setup_installs_torch_dynamo_accumulated_cache_limit() -> None:
    class FakeDynamoConfig:
        def __init__(self) -> None:
            object.__setattr__(self, "_config", {"cache_size_limit": 64})
            object.__setattr__(self, "_default", {"cache_size_limit": 64})
            object.__setattr__(self, "_allowed_keys", {"cache_size_limit"})

        def __getattr__(self, name: str):
            config = object.__getattribute__(self, "_config")
            if name in config:
                return config[name]
            raise AttributeError(name)

        def __setattr__(self, name: str, value) -> None:
            allowed = object.__getattribute__(self, "_allowed_keys")
            if name not in allowed:
                raise AttributeError(name)
            object.__getattribute__(self, "_config")[name] = value

    fake_config = FakeDynamoConfig()
    fake_torch = SimpleNamespace(_dynamo=SimpleNamespace(config=fake_config))

    _ensure_torch_dynamo_accumulated_cache_limit(fake_torch)
    fake_torch._dynamo.config.accumulated_cache_size_limit = 1024

    assert fake_torch._dynamo.config.accumulated_cache_size_limit == 1024
    assert fake_config._default["accumulated_cache_size_limit"] == 64
    assert "accumulated_cache_size_limit" in fake_config._allowed_keys

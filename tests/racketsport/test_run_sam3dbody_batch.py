from __future__ import annotations

import subprocess
import sys
import types
from pathlib import Path
from typing import Any, Mapping

from scripts.racketsport import run_sam3dbody_batch as batch


def test_parse_batch_payload_accepts_static_clip_intrinsics_and_phase_d_optimization(tmp_path: Path) -> None:
    image = tmp_path / "frame_000001.jpg"
    image.write_bytes(b"not-a-real-jpeg")

    payload = {
        "schema_version": 1,
        "clip_intrinsics": {
            "fx": 1000.0,
            "fy": 1000.0,
            "cx": 960.0,
            "cy": 540.0,
            "dist": [],
            "source": "metric_15pt_reviewed",
            "static_per_clip": True,
        },
        "optimization": {
            "sam3d_body_input_size_px": 384,
            "crop_bucket_sizes": [8, 16],
            "torch_compile": True,
            "compile_warmup_buckets": [8, 16],
            "batching": "static_intrinsics_cross_frame_bucketed_body_batch",
            "steady_state_empty_cache": True,
            "inner_bucket_sync": True,
            "upstream_env": {},
            "tier2_output_lite": False,
        },
        "requests": [
            {
                "request_id": "frame1-player7",
                "image": str(image),
                "bboxes": [[10.0, 20.0, 110.0, 220.0]],
            }
        ],
    }

    parsed = batch._parse_batch_payload(payload)

    assert parsed["clip_intrinsics"]["matrix"] == [[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]]
    assert parsed["clip_intrinsics"]["static_per_clip"] is True
    assert parsed["optimization"] == {
        "sam3d_body_input_size_px": 384,
        "crop_bucket_sizes": [8, 16],
        "torch_compile": True,
        "compile_warmup_buckets": [8, 16],
        "compile_warmup_passes": 2,
        "batching": "static_intrinsics_cross_frame_bucketed_body_batch",
        "steady_state_empty_cache": True,
        "inner_bucket_sync": True,
        "upstream_env": {},
        "tier2_output_lite": False,
    }
    assert parsed["bucket_plan"] == [
        {
            "bucket_size": 8,
            "request_ids": ["frame1-player7"],
            "real_request_count": 1,
            "padding_count": 7,
            "padded_request_count": 8,
            "padded_crop_ratio": 0.875,
        }
    ]
    assert parsed["requests"][0]["target_representation"] == "world_mesh"


def test_compile_warmup_passes_defaults_to_two_and_accepts_override() -> None:
    assert batch._parse_optimization({})["compile_warmup_passes"] == 2
    assert batch._parse_optimization({"compile_warmup_passes": 1})["compile_warmup_passes"] == 1


def test_batch_payload_rejects_request_intrinsics_that_disagree_with_static_clip_intrinsics(tmp_path: Path) -> None:
    image = tmp_path / "frame_000001.jpg"
    image.write_bytes(b"not-a-real-jpeg")
    payload = {
        "schema_version": 1,
        "clip_intrinsics": {
            "matrix": [[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]],
            "source": "court_calibration.json",
            "static_per_clip": True,
        },
        "optimization": {"crop_bucket_sizes": [8, 16]},
        "requests": [
            {
                "request_id": "1:7",
                "image": str(image),
                "bboxes": [[10.0, 20.0, 110.0, 220.0]],
                "camera_intrinsics": [[999.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]],
            }
        ],
    }

    try:
        batch._parse_batch_payload(payload)
    except ValueError as exc:
        assert "batch-level clip_intrinsics is authoritative" in str(exc)
        assert "requests/1:7/camera_intrinsics" in str(exc)
    else:  # pragma: no cover - expectation path
        raise AssertionError("mismatched per-request camera intrinsics must fail closed")


def test_bucketed_inference_pads_tail_and_preserves_request_identity(tmp_path: Path) -> None:
    images = []
    for index in range(18):
        image = tmp_path / f"frame_{index:06d}.jpg"
        image.write_bytes(b"not-a-real-jpeg")
        images.append(image)
    requests = [
        {
            "request_id": f"frame{index}:player{index + 100}",
            "image": images[index],
            "bboxes": [[float(index), 10.0, float(index + 100), 210.0]],
            "mask_paths": [],
            "camera_intrinsics": None,
        }
        for index in range(18)
    ]

    class FakeEstimator:
        faces = [[0, 1, 2]]

        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def process_body_bucket(self, bucket_items: list[dict[str, Any]], **_kwargs: Any) -> list[dict[str, Any]]:
            self.calls.append(
                {
                    "bucket_size": len(bucket_items),
                    "real_request_ids": [
                        item["request"]["request_id"] for item in bucket_items if not item["is_padding"]
                    ],
                    "pad_count": sum(1 for item in bucket_items if item["is_padding"]),
                }
            )
            return [
                {
                    "request_id": item["request"]["request_id"],
                    "bbox_echo": list(item["bbox"]),
                    "pad": bool(item["is_padding"]),
                }
                for item in bucket_items
            ]

    plan = batch._bucket_plan([str(request["request_id"]) for request in requests], bucket_sizes=[8, 16])
    frames, execution = batch._run_bucketed_inference(
        FakeEstimator(),
        requests,
        bucket_plan=plan,
        clip_cam_int=None,
        faces=[[0, 1, 2]],
        body_input_size=384,
        optimization=batch._parse_optimization({}),
    )

    assert [call["bucket_size"] for call in execution["buckets"]] == [16, 8]
    assert [call["padded_request_count"] for call in execution["buckets"]] == [16, 8]
    assert [call["padding_count"] for call in execution["buckets"]] == [0, 6]
    assert len(frames) == len(requests)
    assert [frame["request_id"] for frame in frames] == [request["request_id"] for request in requests]
    for frame, request in zip(frames, requests):
        assert frame["records"][0]["request_id"] == request["request_id"]
        assert frame["records"][0]["bbox_echo"] == request["bboxes"][0]


def test_run_sam3dbody_batch_cli_help_direct_reference() -> None:
    command_path = "scripts/racketsport/run_sam3dbody_batch.py"

    completed = subprocess.run(
        [sys.executable, command_path, "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--requests" in completed.stdout
    assert "--out" in completed.stdout
    assert "--fast-sam-repo" in completed.stdout
    assert "--checkpoint-dir" in completed.stdout


def test_arbitrary_bucket_sizes_report_padding_ratio_and_warmup_must_match(tmp_path: Path) -> None:
    images = []
    for index in range(18):
        image = tmp_path / f"frame_{index:06d}.jpg"
        image.write_bytes(b"not-a-real-jpeg")
        images.append(image)

    payload = {
        "schema_version": 1,
        "optimization": {
            "crop_bucket_sizes": [16, 24, 32, 48, 64],
            "torch_compile": True,
            "compile_warmup_buckets": [16, 24, 32, 48, 64],
        },
        "requests": [
            {
                "request_id": f"{index}:7",
                "image": str(images[index]),
                "bboxes": [[10.0, 20.0, 110.0, 220.0]],
            }
            for index in range(18)
        ],
    }

    parsed = batch._parse_batch_payload(payload)

    assert parsed["optimization"]["crop_bucket_sizes"] == [16, 24, 32, 48, 64]
    assert parsed["optimization"]["compile_warmup_buckets"] == [16, 24, 32, 48, 64]
    assert parsed["bucket_plan"] == [
        {
            "bucket_size": 24,
            "request_ids": [f"{index}:7" for index in range(18)],
            "real_request_count": 18,
            "padding_count": 6,
            "padded_request_count": 24,
            "padded_crop_ratio": 0.25,
        }
    ]

    mismatch_payload = {
        **payload,
        "optimization": {
            "crop_bucket_sizes": [16, 24, 32],
            "torch_compile": True,
            "compile_warmup_buckets": [16, 24],
        },
    }
    try:
        batch._parse_batch_payload(mismatch_payload)
    except ValueError as exc:
        assert "compile_warmup_buckets must match crop_bucket_sizes" in str(exc)
    else:  # pragma: no cover - expectation path
        raise AssertionError("compile warmup/execution shape mismatch must fail closed")


def test_upstream_env_exports_whitelist_in_payload_order_and_defaults_empty() -> None:
    default_optimization = batch._parse_optimization({})
    default_recorder: dict[str, str] = {}

    assert default_optimization["upstream_env"] == {}
    assert batch._apply_upstream_env(default_optimization["upstream_env"], environ=default_recorder) == {}
    assert default_recorder == {}

    recorder: dict[str, str] = {}
    applied = batch._apply_upstream_env(
        {
            "USE_COMPILE_BACKBONE": "1",
            "DECODER_COMPILE": 1,
            "INTERM_COMPILE": True,
            "INTERM_SLIM": False,
            "COMPILE_MODE": "reduce-overhead",
            "MHR_NO_CORRECTIVES": "1",
        },
        environ=recorder,
    )

    assert list(recorder.items()) == [
        ("USE_COMPILE_BACKBONE", "1"),
        ("DECODER_COMPILE", "1"),
        ("INTERM_COMPILE", "1"),
        ("INTERM_SLIM", "0"),
        ("COMPILE_MODE", "reduce-overhead"),
        ("MHR_NO_CORRECTIVES", "1"),
    ]
    assert applied == recorder

    try:
        batch._parse_optimization({"upstream_env": {"USE_TRT_BACKBONE": "1"}})
    except ValueError as exc:
        assert "unsupported upstream_env key" in str(exc)
    else:  # pragma: no cover - expectation path
        raise AssertionError("unknown upstream env keys must fail closed")


def test_sync_cache_flags_skip_helpers_when_disabled() -> None:
    class FakeCuda:
        def __init__(self) -> None:
            self.empty_cache_calls = 0
            self.synchronize_calls = 0

        def empty_cache(self) -> None:
            self.empty_cache_calls += 1

        def is_available(self) -> bool:
            return True

        def synchronize(self) -> None:
            self.synchronize_calls += 1

    class FakeTorch:
        def __init__(self) -> None:
            self.cuda = FakeCuda()

    fake_torch = FakeTorch()

    batch._clear_cuda_cache_if_enabled(fake_torch, enabled=True)
    batch._synchronize_cuda_if_available(fake_torch, enabled=True)
    batch._clear_cuda_cache_if_enabled(fake_torch, enabled=False)
    batch._synchronize_cuda_if_available(fake_torch, enabled=False)

    assert fake_torch.cuda.empty_cache_calls == 1
    assert fake_torch.cuda.synchronize_calls == 1


def test_output_lite_prunes_tier2_dense_fields_but_preserves_tier1_and_identity(tmp_path: Path) -> None:
    images = []
    requests = []
    target_representations = ["body_joints", "world_mesh", "body_joints"]
    for index, target_representation in enumerate(target_representations):
        image = tmp_path / f"frame_{index:06d}.jpg"
        image.write_bytes(b"not-a-real-jpeg")
        images.append(image)
        requests.append(
            {
                "request_id": f"{index}:7",
                "image": image,
                "bboxes": [[float(index), 20.0, 110.0, 220.0]],
                "mask_paths": [],
                "camera_intrinsics": None,
                "target_representation": target_representation,
            }
        )

    class FakeEstimator:
        faces = [[0, 1, 2]]

        def process_body_bucket(self, bucket_items: list[dict[str, Any]], **_kwargs: Any) -> list[dict[str, Any]]:
            return [
                {
                    "request_id": item["request"]["request_id"],
                    "pred_keypoints_3d": [[0.0, 0.0, 0.0]],
                    "pred_keypoints_2d": [[0.0, 0.0]],
                    "pred_cam_t": [0.0, 0.0, 1.0],
                    "pred_vertices": [[0.0, 0.0, 0.0]],
                    "pred_joint_coords": [[0.0, 0.0, 0.0]],
                    "pred_global_rots": [[[1.0, 0.0, 0.0]]],
                    "pred_pose_raw": [0.0],
                    "body_pose_params": [0.0],
                    "hand_pose_params": [0.0],
                    "scale_params": [1.0],
                    "shape_params": [0.0],
                    "expr_params": [0.0],
                }
                for item in bucket_items
            ]

    frames, execution = batch._run_bucketed_inference(
        FakeEstimator(),
        requests,
        bucket_plan=batch._bucket_plan([request["request_id"] for request in requests], bucket_sizes=[4]),
        clip_cam_int=None,
        faces=[[0, 1, 2]],
        body_input_size=384,
        optimization=batch._parse_optimization({"tier2_output_lite": True}),
    )

    assert [frame["request_id"] for frame in frames] == [request["request_id"] for request in requests]
    assert execution["timing_mode"] == "sync_per_bucket"
    tier2_first = frames[0]["records"][0]
    tier1 = frames[1]["records"][0]
    tier2_second = frames[2]["records"][0]
    for tier2 in (tier2_first, tier2_second):
        assert tier2["request_id"] in {"0:7", "2:7"}
        assert "pred_keypoints_3d" in tier2
        assert "pred_vertices" not in tier2
        assert "mesh_faces" not in tier2
        assert "hand_pose_params" not in tier2
        assert "expr_params" not in tier2
        assert "pred_joint_coords" not in tier2
        assert "pred_global_rots" not in tier2
    assert tier1["request_id"] == "1:7"
    assert tier1["pred_vertices"] == [[0.0, 0.0, 0.0]]
    assert tier1["mesh_faces"] == [[0, 1, 2]]


def test_detect_mhr_correctives_active_from_loaded_model() -> None:
    class Head:
        apply_correctives = True

    class Model:
        def modules(self) -> list[Any]:
            return [object(), Head()]

    finding = batch._detect_mhr_correctives_active(type("Estimator", (), {"model": Model()})())

    assert finding == {
        "status": "detected",
        "active": True,
        "source": "model.modules.apply_correctives",
    }


def _install_fake_sam3d_modules(monkeypatch) -> None:  # noqa: ANN001
    import numpy as np
    import torch

    io_module = types.ModuleType("sam_3d_body.data.utils.io")

    def load_image(_path: str, *, backend: str, image_format: str):  # noqa: ANN001
        assert backend == "cv2"
        assert image_format == "bgr"
        return np.zeros((10, 12, 3), dtype=np.uint8)

    io_module.load_image = load_image

    utils_module = types.ModuleType("sam_3d_body.utils")

    def recursive_to(value: Any, target: str) -> Any:
        if isinstance(value, dict):
            return {key: recursive_to(item, target) for key, item in value.items()}
        if isinstance(value, list):
            return [recursive_to(item, target) for item in value]
        if isinstance(value, tuple):
            return tuple(recursive_to(item, target) for item in value)
        if torch.is_tensor(value):
            if target == "numpy":
                return value.numpy()
            return value
        return value

    utils_module.recursive_to = recursive_to

    monkeypatch.setitem(sys.modules, "sam_3d_body", types.ModuleType("sam_3d_body"))
    monkeypatch.setitem(sys.modules, "sam_3d_body.data", types.ModuleType("sam_3d_body.data"))
    monkeypatch.setitem(sys.modules, "sam_3d_body.data.utils", types.ModuleType("sam_3d_body.data.utils"))
    monkeypatch.setitem(sys.modules, "sam_3d_body.data.utils.io", io_module)
    monkeypatch.setitem(sys.modules, "sam_3d_body.utils", utils_module)


class _FakeDirectEstimator:
    transform_hand = object()
    thresh_wrist_angle = 0.0

    def __init__(self, *, assert_inference_mode: bool) -> None:
        self.model = _FakeDirectModel(assert_inference_mode=assert_inference_mode)
        self.batch = None
        self.image_embeddings = None
        self.output = None
        self.prev_prompt = []

    def transform(self, data_info: Mapping[str, Any]) -> dict[str, Any]:
        import torch

        image = data_info["img"]
        mask = data_info["mask"]
        height, width = image.shape[:2]
        return {
            "img": torch.zeros((3, 4, 4), dtype=torch.float32),
            "img_size": torch.tensor([4, 4], dtype=torch.float32),
            "ori_img_size": torch.tensor([width, height], dtype=torch.float32),
            "bbox_center": torch.tensor([width / 2.0, height / 2.0], dtype=torch.float32),
            "bbox_scale": torch.tensor([width, height], dtype=torch.float32),
            "bbox": torch.tensor(data_info["bbox"], dtype=torch.float32),
            "affine_trans": torch.eye(3, dtype=torch.float32),
            "mask": torch.as_tensor(mask[:, :, 0], dtype=torch.float32),
            "mask_score": torch.as_tensor(data_info["mask_score"], dtype=torch.float32),
        }


class _FakeDirectModel:
    def __init__(self, *, assert_inference_mode: bool) -> None:
        self.assert_inference_mode = assert_inference_mode
        self.mode_observations: list[tuple[str, bool, bool]] = []
        self.batch_signatures: list[dict[str, Any]] = []

    def _initialize_batch(self, model_batch: Mapping[str, Any]) -> None:
        import torch

        self.mode_observations.append(
            ("initialize", torch.is_grad_enabled(), torch.is_inference_mode_enabled())
        )
        if self.assert_inference_mode:
            assert torch.is_grad_enabled() is False
            assert torch.is_inference_mode_enabled() is True
        self.batch_signatures.append(batch._sam3d_batch_guard_signature(model_batch))

    def run_inference(
        self,
        _image: Any,
        model_batch: Mapping[str, Any],
        **_kwargs: Any,
    ) -> dict[str, dict[str, Any]]:
        import torch

        self.mode_observations.append(
            ("run_inference", torch.is_grad_enabled(), torch.is_inference_mode_enabled())
        )
        if self.assert_inference_mode:
            assert torch.is_grad_enabled() is False
            assert torch.is_inference_mode_enabled() is True
        count = int(model_batch["img"].shape[1])
        seed = torch.ones((count, 1), requires_grad=True)
        grad_sensitive = seed * 2.0

        def values(shape: tuple[int, ...]):
            return grad_sensitive.reshape(count, *([1] * (len(shape) - 1))).expand(shape)

        return {
            "mhr": {
                "focal_length": values((count, 1)),
                "pred_keypoints_3d": values((count, 1, 3)),
                "pred_keypoints_2d": values((count, 1, 2)),
                "pred_vertices": values((count, 1, 3)),
                "pred_cam_t": values((count, 3)),
                "pred_pose_raw": values((count, 1)),
                "global_rot": values((count, 3, 3)),
                "body_pose": values((count, 1)),
                "hand": values((count, 1)),
                "scale": values((count, 1)),
                "shape": values((count, 1)),
                "face": values((count, 1)),
                "pred_joint_coords": values((count, 1, 3)),
                "joint_global_rots": values((count, 1, 3, 3)),
            }
        }


def test_direct_bucket_model_calls_and_numpy_conversion_run_under_inference_mode(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    import pytest

    torch = pytest.importorskip("torch")

    _install_fake_sam3d_modules(monkeypatch)
    estimator = _FakeDirectEstimator(assert_inference_mode=True)
    image = tmp_path / "frame_000001.jpg"
    image.write_bytes(b"unused by fake loader")

    records = batch._process_sam3d_body_bucket_direct(
        estimator,
        [
            {
                "request": {"request_id": "1:7", "image": image},
                "bbox": [1.0, 2.0, 10.0, 12.0],
                "mask_path": None,
                "is_padding": False,
                "target_representation": "world_mesh",
            }
        ],
        clip_cam_int=torch.tensor([[[1000.0, 0.0, 6.0], [0.0, 1000.0, 5.0], [0.0, 0.0, 1.0]]]),
        optimization=batch._parse_optimization(
            {"steady_state_empty_cache": False, "inner_bucket_sync": False}
        ),
    )

    assert records[0]["request_id"] == "1:7"
    assert estimator.model.mode_observations == [
        ("initialize", False, True),
        ("run_inference", False, True),
    ]
    assert records[0]["focal_length"].tolist() == [2.0]


def test_warmup_and_real_synthetic_batches_have_matching_guard_signatures(monkeypatch) -> None:  # noqa: ANN001
    import pytest

    torch = pytest.importorskip("torch")

    _install_fake_sam3d_modules(monkeypatch)
    estimator = _FakeDirectEstimator(assert_inference_mode=False)
    clip_cam_int = torch.tensor([[[1000.0, 0.0, 6.0], [0.0, 1000.0, 5.0], [0.0, 0.0, 1.0]]])

    for bucket_size in (2, 3):
        real_items = [
            batch._synthetic_sam3d_bucket_item(index, bucket_size=bucket_size, image_size_hw=(10, 12))
            for index in range(bucket_size)
        ]
        warmup_items = batch._synthetic_sam3d_warmup_bucket_items(
            bucket_size,
            image_size_hw=(10, 12),
        )
        with torch.inference_mode():
            real_batch = batch._build_sam3d_body_batch(
                estimator,
                real_items,
                clip_cam_int=clip_cam_int,
            )
            warmup_batch = batch._build_sam3d_body_batch(
                estimator,
                warmup_items,
                clip_cam_int=clip_cam_int,
            )
            real_signature = batch._sam3d_batch_guard_signature(real_batch.batch)
            warmup_signature = batch._sam3d_batch_guard_signature(warmup_batch.batch)

        assert real_signature == warmup_signature
        assert "cam_int" in warmup_signature["tensors"]
        assert warmup_signature["grad_enabled"] is False
        assert warmup_signature["inference_mode"] is True


def test_static_clip_intrinsics_warmup_runs_each_bucket_shape_configured_passes(monkeypatch) -> None:  # noqa: ANN001
    import pytest

    torch = pytest.importorskip("torch")

    _install_fake_sam3d_modules(monkeypatch)
    estimator = _FakeDirectEstimator(assert_inference_mode=True)
    optimization = batch._parse_optimization(
        {
            "crop_bucket_sizes": [2, 3],
            "torch_compile": True,
            "compile_warmup_buckets": [2, 3],
            "compile_warmup_passes": 2,
        }
    )

    result = batch._warmup_static_clip_intrinsics(
        estimator,
        clip_intrinsics={
            "matrix": [[1000.0, 0.0, 6.0], [0.0, 1000.0, 5.0], [0.0, 0.0, 1.0]],
            "source": "court_calibration.json",
        },
        optimization=optimization,
    )

    assert result["status"] == "ran"
    assert result["warmup_passes_per_shape"] == 2
    assert result["warmup_call_count_by_bucket"] == {"2": 2, "3": 2}
    assert result["batch_guard_signatures"].keys() == {"2", "3"}
    assert [
        signature["tensors"]["img"]["shape"][1]
        for signature in estimator.model.batch_signatures
    ] == [2, 2, 3, 3]
    assert torch.is_grad_enabled() is True

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
import types
from pathlib import Path
from typing import Any, Mapping

import pytest

from scripts.racketsport import run_sam3dbody_batch as batch


def _simple_requests(tmp_path: Path, count: int) -> list[dict[str, Any]]:
    requests = []
    for index in range(count):
        image = tmp_path / f"frame_{index:06d}.jpg"
        image.write_bytes(b"not-a-real-jpeg")
        requests.append(
            {
                "request_id": f"{index}:7",
                "image": image,
                "bboxes": [[float(index), 10.0, float(index + 100), 210.0]],
                "mask_paths": [],
                "camera_intrinsics": None,
                "target_representation": "world_mesh",
            }
        )
    return requests


def _payload_header(request_count: int) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_sam3dbody_batch",
        "request_count": request_count,
        "clip_intrinsics": None,
        "optimization": batch._parse_optimization({}),
        "metadata": {"timing_mode": "sync_per_bucket", "timings": []},
        "bucket_plan": [],
        "compile_warmup": {"status": "skipped"},
    }


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
        "prefetch_buckets": 1,
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


def test_timing_summary_splits_model_warmup_steady_prep_and_handoff() -> None:
    timer = batch._TimingRecorder()
    base = timer.origin_monotonic
    timer.record("request_parse", base + 0.0, base + 0.1)
    timer.record("model_setup_load", base + 0.1, base + 2.1)
    timer.record("compile_warmup_bucket", base + 2.1, base + 2.7, bucket_size=4, pass_count=2)
    timer.record("compile_warmup_bucket", base + 2.7, base + 3.7, bucket_size=8, pass_count=2)
    timer.record("compile_warmup", base + 2.1, base + 3.7)
    timer.record("request_prep", base + 3.7, base + 4.0, bucket_index=0, bucket_size=4, request_count=4)
    timer.record("bucket_inference", base + 4.0, base + 5.2, bucket_index=0, bucket_size=4, request_count=4)
    timer.record("request_prep", base + 5.2, base + 5.4, bucket_index=1, bucket_size=8, request_count=1)
    timer.record("bucket_inference", base + 5.4, base + 6.0, bucket_index=1, bucket_size=8, request_count=1)
    timer.record("bucket_postprocess", base + 6.0, base + 6.2, bucket_index=1, bucket_size=8, request_count=1)
    timer.record("output_write_bucket", base + 6.2, base + 6.5, bucket_index=0, bucket_size=4, frame_count=4)
    timer.record("output_write_monolithic", base + 6.5, base + 7.0)

    summary = batch._sam3d_batch_timing_summary(timer.events, person_frame_count=5)

    assert summary["artifact_type"] == "racketsport_sam3dbody_batch_timing"
    assert summary["model_setup_load_s"] == 2.0
    assert summary["compile_warmup_s"] == 1.6
    assert summary["steady_inference_s"] == 1.8
    assert summary["person_frame_count"] == 5
    assert summary["ms_per_person_steady"] == pytest.approx(360.0)
    assert summary["crop_bucket_tensor_prep_s"] == 0.5
    assert summary["postprocessing_s"] == 0.2
    assert summary["result_serialization_handoff_s"] == 0.8
    assert summary["per_bucket"] == [
        {"bucket_size": 4, "warmup_s": 0.6, "steady_s": 1.2, "frames": 4},
        {"bucket_size": 8, "warmup_s": 1.0, "steady_s": 0.6, "frames": 1},
    ]


def test_parse_timing_summary_from_fake_subprocess_stdout() -> None:
    expected = {
        "schema_version": 1,
        "artifact_type": "racketsport_sam3dbody_batch_timing",
        "steady_inference_s": 3.5,
        "person_frame_count": 7,
    }
    stdout = "loading model\n" + batch.SAM3D_BATCH_TIMING_STDOUT_MARKER + json.dumps(expected) + "\nfinished\n"

    parsed = batch.parse_sam3d_batch_timing_stdout(stdout)

    assert parsed == expected


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
    assert "--bucket-size" in completed.stdout
    assert "--chunk-format" in completed.stdout
    assert "--convert-chunks" in completed.stdout
    assert "--no-monolithic-output" in completed.stdout


def test_compile_environment_disables_upstream_combined_warmup_but_preserves_script_warmup_contract() -> None:
    recorder: dict[str, str] = {}
    optimization = batch._parse_optimization(
        {
            "crop_bucket_sizes": [8, 16],
            "torch_compile": True,
            "compile_warmup_buckets": [8, 16],
        }
    )

    result = batch._configure_compile_environment(optimization, environ=recorder)

    assert recorder["USE_COMPILE"] == "1"
    assert recorder["COMPILE_WARMUP_BATCH_SIZES"] == ""
    assert result["upstream_estimator_compile_warmup"] == "disabled"
    assert result["script_compile_warmup_buckets"] == [8, 16]


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
    assert batch._apply_bucket_size_override(
        parsed["optimization"],
        bucket_size=32,
    )["crop_bucket_sizes"] == [32]
    assert batch._apply_bucket_size_override(
        parsed["optimization"],
        bucket_size=32,
    )["compile_warmup_buckets"] == [32]
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


def test_prefetch_pipeline_overlaps_next_bucket_prep_and_preserves_order(tmp_path: Path) -> None:
    requests = _simple_requests(tmp_path, 5)
    plan = batch._bucket_plan([request["request_id"] for request in requests], bucket_sizes=[2])

    class FakeEstimator:
        faces = [[0, 1, 2]]

        def __init__(self) -> None:
            self.events: list[tuple[str, int, float]] = []

        def prepare_body_bucket(self, bucket_items: list[dict[str, Any]], **_kwargs: Any) -> list[str]:
            bucket_index = int(bucket_items[0]["request"]["request_id"].split(":")[0]) // 2
            self.events.append(("prepare_start", bucket_index, time.monotonic()))
            time.sleep(0.02)
            self.events.append(("prepare_end", bucket_index, time.monotonic()))
            return [str(item["request"]["request_id"]) for item in bucket_items]

        def process_prepared_body_bucket(
            self,
            bucket_items: list[dict[str, Any]],
            prepared_bucket: list[str],
            **_kwargs: Any,
        ) -> list[dict[str, Any]]:
            bucket_index = int(bucket_items[0]["request"]["request_id"].split(":")[0]) // 2
            self.events.append(("process_start", bucket_index, time.monotonic()))
            time.sleep(0.08)
            self.events.append(("process_end", bucket_index, time.monotonic()))
            return [
                {
                    "request_id": item["request"]["request_id"],
                    "prepared_request_id": prepared_bucket[index],
                    "bbox_echo": list(item["bbox"]),
                }
                for index, item in enumerate(bucket_items)
            ]

    estimator = FakeEstimator()

    frames, execution = batch._run_bucketed_inference(
        estimator,
        requests,
        bucket_plan=plan,
        clip_cam_int=None,
        faces=[[0, 1, 2]],
        body_input_size=384,
        optimization=batch._parse_optimization({"prefetch_buckets": 1}),
    )

    assert [frame["request_id"] for frame in frames] == [request["request_id"] for request in requests]
    assert [frame["records"][0]["request_id"] for frame in frames] == [request["request_id"] for request in requests]
    assert execution["request_count"] == 5
    assert execution["output_count"] == 5
    event_times = {(name, bucket_index): timestamp for name, bucket_index, timestamp in estimator.events}
    assert event_times[("prepare_start", 1)] < event_times[("process_end", 0)]


def test_stream_writer_writes_chunks_once_and_converter_matches_monolithic(tmp_path: Path) -> None:
    requests = _simple_requests(tmp_path, 3)
    plan = batch._bucket_plan([request["request_id"] for request in requests], bucket_sizes=[2])

    class FakeEstimator:
        faces = [[0, 1, 2]]

        def process_body_bucket(self, bucket_items: list[dict[str, Any]], **_kwargs: Any) -> list[dict[str, Any]]:
            return [
                {
                    "request_id": item["request"]["request_id"],
                    "bbox_echo": list(item["bbox"]),
                    "pad": bool(item["is_padding"]),
                }
                for item in bucket_items
            ]

    header = _payload_header(len(requests))
    header["bucket_plan"] = plan
    out_path = tmp_path / "out.json"
    stream = batch._BatchOutputStream(
        out_path=out_path,
        chunk_dir=tmp_path / "out.json.chunks",
        request_ids=[request["request_id"] for request in requests],
        payload_header=header,
        write_monolithic=True,
    )
    frames, execution = batch._run_bucketed_inference(
        FakeEstimator(),
        requests,
        bucket_plan=plan,
        clip_cam_int=None,
        faces=[[0, 1, 2]],
        body_input_size=384,
        optimization=batch._parse_optimization({}),
        output_stream=stream,
    )

    index = batch._read_chunk_index(tmp_path / "out.json.chunks" / "index.json")
    assert index["status"] == "complete"
    assert index["result_count"] == 3
    assert index["batch_execution"]["streaming_output"]["format"] == "pickle_chunks_with_monolithic_converter"
    assert {chunk["format"] for chunk in index["chunks"]} == {"pickle"}
    assert sorted(index["request_ids"]) == sorted(request["request_id"] for request in requests)
    chunked_frames = batch._frames_from_chunk_index(tmp_path / "out.json.chunks" / "index.json")
    assert [frame["request_id"] for frame in chunked_frames] == [request["request_id"] for request in requests]
    assert len({frame["request_id"] for frame in chunked_frames}) == 3

    monolithic = batch._monolithic_payload_from_header(header, frames=frames, batch_execution=execution)
    converted = batch._monolithic_payload_from_chunk_index(tmp_path / "out.json.chunks" / "index.json")
    assert batch._encode_json(monolithic) == batch._encode_json(converted)
    assert out_path.read_text(encoding="utf-8") == batch._encode_json(converted)


def test_binary_stream_writer_round_trips_bulk_arrays_without_pickle_chunks(tmp_path: Path) -> None:
    import numpy as np

    requests = _simple_requests(tmp_path, 2)
    plan = batch._bucket_plan([request["request_id"] for request in requests], bucket_sizes=[2])

    class FakeEstimator:
        faces = np.asarray([[0, 1, 2]], dtype=np.int32)

        def process_body_bucket(self, bucket_items: list[dict[str, Any]], **_kwargs: Any) -> list[dict[str, Any]]:
            rows = []
            for index, item in enumerate(bucket_items):
                rows.append(
                    {
                        "request_id": item["request"]["request_id"],
                        "pred_vertices": np.asarray(
                            [
                                [1.0 + index, 2.0, 0.3],
                                [1.1 + index, 2.1, 0.4],
                                [1.2 + index, 2.2, 0.5],
                            ],
                            dtype=np.float32,
                        ),
                        "pred_keypoints_3d": np.asarray([[0.1 + index, 0.2, 1.0]], dtype=np.float32),
                        "global_rot": np.asarray([0.01, 0.02, 0.03], dtype=np.float32),
                        "body_pose_params": np.asarray([0.1, 0.2, 0.3], dtype=np.float32),
                        "confidence": 0.75,
                    }
                )
            return rows

    stream = batch._BatchOutputStream(
        out_path=tmp_path / "out.json",
        chunk_dir=tmp_path / "out.json.chunks",
        request_ids=[request["request_id"] for request in requests],
        payload_header={**_payload_header(len(requests)), "bucket_plan": plan},
        write_monolithic=False,
        chunk_format="binary",
    )

    frames, execution = batch._run_bucketed_inference(
        FakeEstimator(),
        requests,
        bucket_plan=plan,
        clip_cam_int=None,
        faces=FakeEstimator.faces,
        body_input_size=384,
        optimization=batch._parse_optimization({}),
        output_stream=stream,
        materialize_streamed_frames=False,
    )

    assert frames == []
    assert execution["streaming_output"]["format"] == "binary_chunks_with_monolithic_converter"
    assert not list((tmp_path / "out.json.chunks").glob("*.pkl"))
    index_path = tmp_path / "out.json.chunks" / "index.json"
    index = batch._read_chunk_index(index_path)
    assert {chunk["format"] for chunk in index["chunks"]} == {"binary"}

    outputs = batch.load_sam3dbody_binary_outputs_from_chunk_index(
        index_path,
        request_ids=[request["request_id"] for request in requests],
        mmap_mode="r",
    )

    assert len(outputs) == 2
    np.testing.assert_allclose(outputs[0][0]["pred_vertices"], np.asarray([[1.0, 2.0, 0.3], [1.1, 2.1, 0.4], [1.2, 2.2, 0.5]], dtype=np.float32))
    np.testing.assert_allclose(outputs[1][0]["pred_keypoints_3d"], np.asarray([[1.1, 0.2, 1.0]], dtype=np.float32))
    assert outputs[0][0]["confidence"] == 0.75


def test_stream_only_pickle_writer_does_not_space_inference_on_slow_writer(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    requests = _simple_requests(tmp_path, 5)
    plan = batch._bucket_plan([request["request_id"] for request in requests], bucket_sizes=[1])
    process_events: list[tuple[str, str, float]] = []
    write_events: list[tuple[str, float]] = []
    writer_entered = threading.Event()
    release_writer = threading.Event()
    run_result: dict[str, Any] = {}

    class FakeEstimator:
        faces = []

        def process_body_bucket(self, bucket_items: list[dict[str, Any]], **_kwargs: Any) -> list[dict[str, Any]]:
            request_id = str(bucket_items[0]["request"]["request_id"])
            process_events.append(("process_start", request_id, time.monotonic()))
            time.sleep(0.005)
            process_events.append(("process_end", request_id, time.monotonic()))
            return [{"request_id": item["request"]["request_id"]} for item in bucket_items]

    def blocked_pickle_write(path: Path, payload: Mapping[str, Any]) -> None:
        write_events.append((str(payload["bucket_index"]), time.monotonic()))
        if len(write_events) == 1:
            writer_entered.set()
            assert release_writer.wait(timeout=2.0)
        batch._write_pickle_payload_original(path, payload)

    monkeypatch.setattr(batch, "_write_pickle_payload_original", batch._write_pickle_payload, raising=False)
    monkeypatch.setattr(batch, "_write_pickle_payload", blocked_pickle_write)
    stream = batch._BatchOutputStream(
        out_path=tmp_path / "out.json",
        chunk_dir=tmp_path / "out.json.chunks",
        request_ids=[request["request_id"] for request in requests],
        payload_header={**_payload_header(len(requests)), "bucket_plan": plan},
        write_monolithic=False,
        chunk_format="pickle",
    )

    def run_inference() -> None:
        run_result["value"] = batch._run_bucketed_inference(
            FakeEstimator(),
            requests,
            bucket_plan=plan,
            clip_cam_int=None,
            faces=[],
            body_input_size=384,
            optimization=batch._parse_optimization({"prefetch_buckets": 0}),
            output_stream=stream,
            materialize_streamed_frames=False,
        )

    runner = threading.Thread(target=run_inference)
    runner.start()
    assert writer_entered.wait(timeout=1.0)
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        if sum(1 for name, _request_id, _timestamp in process_events if name == "process_end") == len(requests):
            break
        time.sleep(0.005)
    release_writer.set()
    runner.join(timeout=2.0)
    assert not runner.is_alive()
    frames, execution = run_result["value"]

    assert frames == []
    assert execution["output_count"] == len(requests)
    starts = [timestamp for name, _request_id, timestamp in process_events if name == "process_start"]
    assert len(starts) == len(requests)
    assert max(later - earlier for earlier, later in zip(starts, starts[1:])) < 0.05
    assert len(write_events) == len(requests)


def test_warmed_production_callable_identity_assertion_detects_path_change() -> None:
    class FakeModel:
        def forward_decoder(self) -> None:
            return None

    estimator = type("Estimator", (), {"model": FakeModel()})()
    warmed_identity = batch._remember_warmed_production_callable(estimator)

    assert batch._assert_warmed_production_callable(estimator) == warmed_identity

    estimator.model.forward_decoder = lambda: None
    try:
        batch._assert_warmed_production_callable(estimator)
    except RuntimeError as exc:
        assert "production SAM3D decoder callable changed after warmup" in str(exc)
    else:  # pragma: no cover - expectation path
        raise AssertionError("production callable mismatch must fail closed")


def test_stream_index_finalizes_on_loader_failure_and_propagates(tmp_path: Path) -> None:
    requests = _simple_requests(tmp_path, 3)
    plan = batch._bucket_plan([request["request_id"] for request in requests], bucket_sizes=[2])

    class FailingLoaderEstimator:
        faces = []

        def prepare_body_bucket(self, bucket_items: list[dict[str, Any]], **_kwargs: Any) -> None:
            if str(bucket_items[0]["request"]["request_id"]) == "2:7":
                raise RuntimeError("loader boom")
            return None

        def process_prepared_body_bucket(
            self,
            bucket_items: list[dict[str, Any]],
            prepared_bucket: Any,
            **_kwargs: Any,
        ) -> list[dict[str, Any]]:
            return [{"request_id": item["request"]["request_id"]} for item in bucket_items]

    stream = batch._BatchOutputStream(
        out_path=tmp_path / "out.json",
        chunk_dir=tmp_path / "out.json.chunks",
        request_ids=[request["request_id"] for request in requests],
        payload_header={**_payload_header(len(requests)), "bucket_plan": plan},
        write_monolithic=True,
        chunk_format="jsonl",
    )

    try:
        batch._run_bucketed_inference(
            FailingLoaderEstimator(),
            requests,
            bucket_plan=plan,
            clip_cam_int=None,
            faces=[],
            body_input_size=384,
            optimization=batch._parse_optimization({"prefetch_buckets": 1}),
            output_stream=stream,
        )
    except RuntimeError as exc:
        assert "loader boom" in str(exc)
    else:  # pragma: no cover - expectation path
        raise AssertionError("loader failure must propagate")

    index = batch._read_chunk_index(tmp_path / "out.json.chunks" / "index.json")
    assert index["status"] == "failed"
    assert index["result_count"] == 2
    assert index["error"]["message"] == "loader boom"


def test_writer_exception_finalizes_failed_index_and_propagates(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    requests = _simple_requests(tmp_path, 1)
    plan = batch._bucket_plan([request["request_id"] for request in requests], bucket_sizes=[1])

    class FakeEstimator:
        faces = []

        def process_body_bucket(self, bucket_items: list[dict[str, Any]], **_kwargs: Any) -> list[dict[str, Any]]:
            return [{"request_id": item["request"]["request_id"]} for item in bucket_items]

    def fail_jsonl(*_args: Any, **_kwargs: Any) -> None:
        raise OSError("writer boom")

    monkeypatch.setattr(batch, "_write_jsonl_lines", fail_jsonl)
    stream = batch._BatchOutputStream(
        out_path=tmp_path / "out.json",
        chunk_dir=tmp_path / "out.json.chunks",
        request_ids=[request["request_id"] for request in requests],
        payload_header={**_payload_header(len(requests)), "bucket_plan": plan},
        write_monolithic=True,
        chunk_format="jsonl",
    )

    try:
        batch._run_bucketed_inference(
            FakeEstimator(),
            requests,
            bucket_plan=plan,
            clip_cam_int=None,
            faces=[],
            body_input_size=384,
            optimization=batch._parse_optimization({}),
            output_stream=stream,
        )
    except OSError as exc:
        assert "writer boom" in str(exc)
    else:  # pragma: no cover - expectation path
        raise AssertionError("writer failure must propagate")

    index = batch._read_chunk_index(tmp_path / "out.json.chunks" / "index.json")
    assert index["status"] == "failed"
    assert index["error"]["message"] == "writer boom"


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

    def forward_decoder(self) -> None:
        return None

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
    assert result["production_callable_identity_before"] == result["production_callable_identity_after"]
    assert batch._assert_warmed_production_callable(estimator) == result["production_callable_identity_after"]
    assert result["batch_guard_signatures"].keys() == {"2", "3"}
    assert [
        signature["tensors"]["img"]["shape"][1]
        for signature in estimator.model.batch_signatures
    ] == [2, 2, 3, 3]
    assert torch.is_grad_enabled() is True

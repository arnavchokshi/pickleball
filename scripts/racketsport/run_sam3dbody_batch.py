#!/usr/bin/env python3
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
import pickle
import queue
import sys
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.racketsport.run_sam3dbody_frame import (  # noqa: E402
    EX_CONFIG,
    _bbox_array_or_list,
    _extract_person_records,
    _json_safe,
)
from scripts.racketsport.run_sam3dbody_probe import (  # noqa: E402
    _detector_name,
    _load_setup_sam_3d_body,
    _runtime_path_errors,
    _setup_estimator,
    parse_bbox_arg,
)
from threed.racketsport.sam3d_body_input_prep import load_mask_prompt_arrays, normalize_body_input_size  # noqa: E402


TIER2_BODY_JOINTS_REPRESENTATION = "body_joints"
UPSTREAM_ENV_WHITELIST = frozenset(
    {
        "USE_COMPILE_BACKBONE",
        "DECODER_COMPILE",
        "INTERM_COMPILE",
        "INTERM_SLIM",
        "COMPILE_MODE",
        "MHR_NO_CORRECTIVES",
    }
)
COMPILE_MODE_VALUES = frozenset({"default", "reduce-overhead", "max-autotune"})
TIER2_OUTPUT_LITE_OMIT_FIELDS = frozenset(
    {
        "pred_vertices",
        "pred_joint_coords",
        "pred_global_rots",
        "pred_pose_raw",
        "global_rot",
        "body_pose_params",
        "hand_pose_params",
        "scale_params",
        "shape_params",
        "expr_params",
        "mesh_faces",
        "faces",
    }
)
CHUNK_FORMATS = frozenset({"binary", "jsonl", "pickle"})
PRODUCTION_CALLABLE_IDENTITY_ATTR = "_sam3d_body_batch_warmed_production_callable_identity"
SAM3D_BATCH_TIMING_ARTIFACT_TYPE = "racketsport_sam3dbody_batch_timing"
SAM3D_BATCH_TIMING_STDOUT_MARKER = "SAM3DBODY_BATCH_TIMING_JSON "
SAM3D_BATCH_BINARY_CHUNK_ARTIFACT_TYPE = "racketsport_sam3dbody_batch_binary_chunk"
SAM3D_BATCH_BINARY_CONTRACT_VERSION = 1
SAM3D_ARRAY_REF_KEY = "__sam3d_array_ref__"
# Pickle chunks may cross a venv boundary (subprocess python_executable vs the
# orchestrator's interpreter). numpy's OWN pickle protocol embeds the writer's
# internal module path (e.g. numpy 2.x's numpy._core.numeric vs numpy 1.x's
# numpy.core.numeric) and fails to load across a numpy major-version gap --
# measured live on the A100 2026-07-05 (FAST_SAM_PYTHON venv numpy 2.2.6 vs
# orchestrator venv numpy 1.26.4): ModuleNotFoundError: numpy._core.numeric.
# _pickle_safe_arrays/_restore_pickle_safe_arrays replace bulk ndarrays with a
# dtype+shape+raw-bytes descriptor built from built-in types only (bytes,
# str, list), which pickle/unpickle identically regardless of numpy version
# on either end, then reconstruct via np.frombuffer on read.
SAM3D_PICKLE_NDARRAY_MARKER = "__sam3d_pickle_ndarray__"
SAM3D_BULK_ARRAY_FIELDS = frozenset(
    {
        "pred_vertices",
        "vertices",
        "mesh_vertices_xyz",
        "pred_keypoints_3d",
        "pred_joint_coords",
        "joints3d",
        "joints3d_xyz",
        "pred_keypoints_2d",
        "pred_cam_t",
        "global_rot",
        "global_orient",
        "pred_global_orient",
        "body_pose_params",
        "body_pose",
        "pred_pose_raw",
        "hand_pose_params",
        "hand_pose",
        "shape_params",
        "betas",
        "scale_params",
        "expr_params",
        "pred_global_rots",
    }
)


@dataclass(frozen=True)
class _Sam3DBodyBatch:
    batch: Mapping[str, Any]
    rgb_images: list[Any]
    mask_scores: list[float]


@dataclass(frozen=True)
class _PreparedBucket:
    bucket_index: int
    bucket_size: int
    request_ids: list[str]
    real_items: list[dict[str, Any]]
    bucket_items: list[dict[str, Any]]
    execution_entry: dict[str, Any]
    prepared_bucket: Any


@dataclass(frozen=True)
class _BucketInferenceOutput:
    prepared: _PreparedBucket
    raw_records: list[Any]
    faces: Any
    body_input_size: int | None
    clip_intrinsics: Mapping[str, Any] | None
    optimization: Mapping[str, Any]


@dataclass(frozen=True)
class _WriterFinish:
    status: str
    error: BaseException | None = None


class _TimingRecorder:
    def __init__(self) -> None:
        self.origin_monotonic = time.monotonic()
        self.events: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    @contextmanager
    def span(self, name: str, **fields: Any) -> Any:
        start = time.monotonic()
        try:
            yield
        finally:
            self.record(name, start, time.monotonic(), **fields)

    def record(self, name: str, start: float, end: float, **fields: Any) -> dict[str, Any]:
        event = {
            "name": str(name),
            "start_s": round(float(start - self.origin_monotonic), 6),
            "end_s": round(float(end - self.origin_monotonic), 6),
            "duration_s": round(float(end - start), 6),
        }
        for key, value in fields.items():
            event[str(key)] = _json_safe(value)
        with self._lock:
            self.events.append(event)
        return event


def _timing_sidecar_path(out_path: Path) -> Path:
    return out_path.with_name(f"{out_path.name}.timing.json")


def _event_duration_s(event: Mapping[str, Any]) -> float:
    try:
        return float(event.get("duration_s", 0.0))
    except (TypeError, ValueError):
        return 0.0


def _event_int(event: Mapping[str, Any], key: str, default: int = 0) -> int:
    try:
        return int(event.get(key, default))
    except (TypeError, ValueError):
        return default


def _round_s(value: float) -> float:
    return round(float(value), 6)


def _sum_events(events: Sequence[Mapping[str, Any]], names: set[str]) -> float:
    return _round_s(sum(_event_duration_s(event) for event in events if str(event.get("name", "")) in names))


def _sam3d_batch_timing_summary(events: Sequence[Mapping[str, Any]], *, person_frame_count: int) -> dict[str, Any]:
    """Summarize runner-internal timing events without changing inference behavior."""

    timing_events = [dict(event) for event in events]
    request_parse_s = _sum_events(timing_events, {"request_parse"})
    model_setup_load_s = _sum_events(timing_events, {"model_setup_load"})
    compile_warmup_bucket_s = _sum_events(timing_events, {"compile_warmup_bucket"})
    compile_warmup_s = compile_warmup_bucket_s or _sum_events(timing_events, {"compile_warmup"})
    steady_inference_s = _sum_events(timing_events, {"bucket_inference"})
    crop_bucket_tensor_prep_s = _sum_events(timing_events, {"request_prep"})
    postprocessing_s = _sum_events(timing_events, {"bucket_postprocess"})
    result_serialization_handoff_s = _sum_events(timing_events, {"output_write_bucket", "output_write_monolithic"})
    total_s = _round_s(max((float(event.get("end_s", 0.0)) for event in timing_events), default=0.0))
    attributed_s = _round_s(
        request_parse_s
        + model_setup_load_s
        + compile_warmup_s
        + steady_inference_s
        + crop_bucket_tensor_prep_s
        + postprocessing_s
        + result_serialization_handoff_s
    )
    person_count = int(person_frame_count)
    per_bucket = _per_bucket_timing_summary(timing_events)
    return {
        "schema_version": 1,
        "artifact_type": SAM3D_BATCH_TIMING_ARTIFACT_TYPE,
        "total_s": total_s,
        "request_parse_s": request_parse_s,
        "model_setup_load_s": model_setup_load_s,
        "compile_warmup_s": compile_warmup_s,
        "steady_inference_s": steady_inference_s,
        "person_frame_count": person_count,
        "ms_per_person_steady": _round_s((steady_inference_s * 1000.0) / person_count) if person_count > 0 else None,
        "crop_bucket_tensor_prep_s": crop_bucket_tensor_prep_s,
        "preprocessing_s": crop_bucket_tensor_prep_s,
        "postprocessing_s": postprocessing_s,
        "result_serialization_handoff_s": result_serialization_handoff_s,
        "attributed_s": attributed_s,
        "other_s": _round_s(max(0.0, total_s - attributed_s)),
        "per_bucket": per_bucket,
    }


def _per_bucket_timing_summary(events: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[int, dict[str, float | int]] = {}
    for event in events:
        name = str(event.get("name", ""))
        if name not in {"compile_warmup_bucket", "bucket_inference"}:
            continue
        bucket_size = _event_int(event, "bucket_size", 0)
        if bucket_size <= 0:
            continue
        bucket = buckets.setdefault(bucket_size, {"bucket_size": bucket_size, "warmup_s": 0.0, "steady_s": 0.0, "frames": 0})
        if name == "compile_warmup_bucket":
            bucket["warmup_s"] = float(bucket["warmup_s"]) + _event_duration_s(event)
        elif name == "bucket_inference":
            bucket["steady_s"] = float(bucket["steady_s"]) + _event_duration_s(event)
            bucket["frames"] = int(bucket["frames"]) + _event_int(event, "request_count", _event_int(event, "frame_count", bucket_size))
    return [
        {
            "bucket_size": int(bucket["bucket_size"]),
            "warmup_s": _round_s(float(bucket["warmup_s"])),
            "steady_s": _round_s(float(bucket["steady_s"])),
            "frames": int(bucket["frames"]),
        }
        for _size, bucket in sorted(buckets.items())
    ]


def parse_sam3d_batch_timing_stdout(stdout: str) -> dict[str, Any] | None:
    for line in stdout.splitlines():
        if not line.startswith(SAM3D_BATCH_TIMING_STDOUT_MARKER):
            continue
        payload = json.loads(line[len(SAM3D_BATCH_TIMING_STDOUT_MARKER) :])
        if isinstance(payload, dict):
            return payload
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run FastSAM-3D-Body on a batch of frame/bbox requests. "
            "Request JSON may include mask_paths and static camera_intrinsics for Phase C."
        )
    )
    parser.add_argument("--requests", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--fast-sam-repo", type=Path)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--detector-model", default="")
    parser.add_argument("--detector-name", default=None)
    parser.add_argument("--fov-name", default="")
    parser.add_argument("--body-input-size", type=int, default=None, help="Optional SAM-3D body crop size: 384, 448, or 512.")
    parser.add_argument(
        "--bucket-size",
        type=int,
        default=None,
        help="Override request-payload crop_bucket_sizes/compile_warmup_buckets for bucket-size experiments.",
    )
    parser.add_argument(
        "--chunk-dir",
        type=Path,
        default=None,
        help="Directory for incremental per-bucket chunks. Defaults to <out>.chunks.",
    )
    parser.add_argument(
        "--convert-chunks",
        type=Path,
        default=None,
        help="Convert a chunk index written by this runner back to the monolithic --out JSON and exit.",
    )
    parser.add_argument(
        "--chunk-format",
        choices=sorted(CHUNK_FORMATS),
        default="pickle",
        help=(
            "Per-bucket stream chunk encoding. pickle (default) writes each bucket synchronously "
            "in the caller's thread for the fastest subprocess handoff (avoids writer-thread GIL "
            "contention with SAM3D inference/prep, measured live on the A100 2026-07-05); binary "
            "streams bulk numeric fields to numpy .npy sidecars via an async writer thread (opt-in, "
            "useful for very large mmap-friendly payloads); jsonl is human-readable but slower."
        ),
    )
    parser.add_argument(
        "--no-monolithic-output",
        action="store_true",
        help="Write only stream chunks and index; use --convert-chunks later to materialize the legacy monolithic JSON.",
    )
    args = parser.parse_args(argv)
    if args.out is None:
        parser.error("--out is required")
    if args.convert_chunks is not None:
        _convert_chunked_output_to_monolithic(args.convert_chunks, args.out)
        print(args.out)
        return 0
    for required_name in ("requests", "fast_sam_repo", "checkpoint_dir"):
        if getattr(args, required_name) is None:
            parser.error(f"--{required_name.replace('_', '-')} is required unless --convert-chunks is used")

    timer = _TimingRecorder()
    try:
        with timer.span("request_parse"):
            payload = json.loads(args.requests.read_text(encoding="utf-8"))
            batch_payload = _parse_batch_payload(payload)
            requests = batch_payload["requests"]
            optimization = dict(batch_payload["optimization"])
            optimization = _apply_bucket_size_override(optimization, bucket_size=args.bucket_size)
            batch_payload["optimization"] = optimization
            if args.bucket_size is not None:
                batch_payload["bucket_plan"] = _bucket_plan(
                    [str(request["request_id"]) for request in requests],
                    bucket_sizes=optimization["crop_bucket_sizes"],
                )
            body_input_size = normalize_body_input_size(args.body_input_size or optimization.get("sam3d_body_input_size_px"))
            optimization["sam3d_body_input_size_px"] = body_input_size
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"invalid request batch: {exc}", file=sys.stderr)
        return EX_CONFIG
    # Ordering constraint: upstream Fast-SAM modules read these environment
    # variables during import/model construction, so this must run before
    # _load_setup_sam_3d_body() and _setup_estimator().
    runtime_environment = _configure_runtime_environment(optimization)
    if body_input_size is not None:
        os.environ["IMG_SIZE"] = str(body_input_size)

    all_path_errors = []
    for request in requests:
        all_path_errors.extend(_runtime_path_errors(request["image"], args.fast_sam_repo, args.checkpoint_dir))
    if all_path_errors:
        for error in all_path_errors:
            print(error, file=sys.stderr)
        return EX_CONFIG

    output_stream: _BatchOutputStream | None = None
    try:
        with timer.span("model_setup_load"):
            setup_sam_3d_body = _load_setup_sam_3d_body(args.fast_sam_repo)
            estimator = _setup_estimator(
                setup_sam_3d_body,
                checkpoint_dir=args.checkpoint_dir.resolve(),
                detector_name=_detector_name(args.detector_name, [bbox for request in requests for bbox in request["bboxes"]]),
                detector_model=args.detector_model,
                fov_name=args.fov_name,
            )
        with timer.span("compile_warmup"):
            compile_warmup = _warmup_static_clip_intrinsics(
                estimator,
                clip_intrinsics=batch_payload["clip_intrinsics"],
                optimization=optimization,
                timing=timer,
            )
        mhr_correctives = _detect_mhr_correctives_active(estimator)
        faces = _json_safe(getattr(estimator, "faces", None))
        clip_cam_int = _camera_intrinsics_tensor(
            batch_payload["clip_intrinsics"]["matrix"] if batch_payload["clip_intrinsics"] is not None else None
        )
        metadata = _output_metadata(
            optimization,
            runtime_environment=runtime_environment,
            mhr_correctives=mhr_correctives,
            timing_events=timer.events,
        )
        output_stream = _BatchOutputStream(
            out_path=args.out,
            chunk_dir=args.chunk_dir or _default_chunk_dir(args.out),
            request_ids=[str(request["request_id"]) for request in requests],
            payload_header={
                "schema_version": 1,
                "artifact_type": "racketsport_sam3dbody_batch",
                "request_count": len(requests),
                "clip_intrinsics": batch_payload["clip_intrinsics"],
                "optimization": optimization,
                "metadata": metadata,
                "bucket_plan": batch_payload["bucket_plan"],
                "compile_warmup": compile_warmup,
            },
            write_monolithic=not args.no_monolithic_output,
            chunk_format=args.chunk_format,
            timing=timer,
            # Restore the pre-S4 direct handoff profile as default: pickle writes
            # synchronously (no writer thread => no GIL contention with inference);
            # binary/jsonl keep the async writer thread (opt-in, mmap-friendly).
            async_write=(args.chunk_format != "pickle"),
        )
        frames, batch_execution = _run_bucketed_inference(
            estimator,
            requests,
            bucket_plan=batch_payload["bucket_plan"],
            clip_cam_int=clip_cam_int,
            clip_intrinsics=batch_payload["clip_intrinsics"],
            faces=faces,
            body_input_size=body_input_size,
            optimization=optimization,
            output_stream=output_stream,
            timing=timer,
            materialize_streamed_frames=False,
        )
    except Exception as exc:
        _write_failure_output(
            args.out,
            exc,
            request_count=len(requests) if "requests" in locals() else 0,
            optimization=optimization if "optimization" in locals() else {},
            compile_warmup=compile_warmup if "compile_warmup" in locals() else None,
        )
        print(f"FastSAM-3D-Body batch failed: {exc}", file=sys.stderr)
        return 1

    timing_summary = _sam3d_batch_timing_summary(timer.events, person_frame_count=len(requests))
    _write_json_payload(_timing_sidecar_path(args.out), timing_summary)
    print(SAM3D_BATCH_TIMING_STDOUT_MARKER + json.dumps(timing_summary, separators=(",", ":"), sort_keys=True))
    print(args.out)
    return 0


def _write_failure_output(
    out_path: Path,
    exc: BaseException,
    *,
    request_count: int,
    optimization: Mapping[str, Any],
    compile_warmup: Mapping[str, Any] | None,
) -> None:
    payload = {
        "schema_version": 1,
        "artifact_type": "racketsport_sam3dbody_batch_error",
        "status": "failed",
        "request_count": int(request_count),
        "optimization": dict(optimization),
        "compile_warmup": dict(compile_warmup) if compile_warmup is not None else None,
        "error": {
            "type": type(exc).__name__,
            "message": str(exc),
        },
    }
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(_json_safe(payload), separators=(",", ":")) + "\n", encoding="utf-8")
    except OSError:
        pass


def _default_chunk_dir(out_path: Path) -> Path:
    return out_path.with_name(f"{out_path.name}.chunks")


def _encode_json(payload: Any) -> str:
    return json.dumps(_json_safe(payload), separators=(",", ":")) + "\n"


def _write_jsonl_lines(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(_encode_json(row) for row in rows), encoding="utf-8")


def _write_json_payload(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_encode_json(payload), encoding="utf-8")


def _write_pickle_payload(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(dict(payload), handle, protocol=pickle.HIGHEST_PROTOCOL)


def _read_pickle_payload(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"pickle chunk is not an object: {path}")
    return payload


class _BinaryArrayWriter:
    def __init__(self, *, chunk_path: Path) -> None:
        self.chunk_path = chunk_path
        self.array_dir = chunk_path.with_suffix("")
        self.array_dir.mkdir(parents=True, exist_ok=True)
        self.count = 0

    def maybe_ref(self, field_name: str, value: Any) -> Any:
        if field_name not in SAM3D_BULK_ARRAY_FIELDS:
            return _json_safe(value)
        array = _numpy_array_or_none(value)
        if array is None:
            return _json_safe(value)
        rel_path = f"arrays/{_safe_array_name(field_name)}_{self.count:06d}.npy"
        path = self.array_dir / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        _numpy_save(path, array)
        self.count += 1
        return {
            SAM3D_ARRAY_REF_KEY: {
                "path": rel_path,
                "dtype": str(array.dtype),
                "shape": [int(value) for value in array.shape],
            }
        }


def _safe_array_name(field_name: str) -> str:
    return "".join(char if char.isalnum() or char == "_" else "_" for char in field_name)


def _numpy_array_or_none(value: Any) -> Any | None:
    try:
        import numpy as np  # type: ignore[import-not-found]
    except ImportError:
        return None
    item = value
    for method_name in ("detach", "cpu"):
        method = getattr(item, method_name, None)
        if callable(method):
            try:
                item = method()
            except Exception:
                return None
    numpy_method = getattr(item, "numpy", None)
    if callable(numpy_method):
        try:
            item = numpy_method()
        except Exception:
            return None
    try:
        array = np.asarray(item)
    except Exception:
        return None
    if array.dtype == object or array.ndim == 0:
        return None
    if not (np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.bool_)):
        return None
    return array


def _pickle_safe_arrays(value: Any) -> Any:
    # Container types are recursed into FIRST (not array-probed): np.asarray([])
    # happily coerces an empty *plain* Python list into array([], dtype=float64),
    # so probing containers before recursing would wrongly treat ordinary empty
    # list metadata (e.g. optimization.crop_bucket_sizes=[]) as a bulk ndarray.
    if isinstance(value, Mapping):
        return {str(key): _pickle_safe_arrays(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_pickle_safe_arrays(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_pickle_safe_arrays(item) for item in value)
    array = _numpy_array_or_none(value)
    if array is not None:
        return {
            SAM3D_PICKLE_NDARRAY_MARKER: True,
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "data": array.tobytes(),
        }
    return value


def _restore_pickle_safe_arrays(value: Any) -> Any:
    import numpy as np  # type: ignore[import-not-found]

    if isinstance(value, Mapping):
        if value.get(SAM3D_PICKLE_NDARRAY_MARKER):
            array = np.frombuffer(value["data"], dtype=value["dtype"])
            shape = value.get("shape") or []
            return array.reshape(shape).copy() if shape else array.copy()
        return {str(key): _restore_pickle_safe_arrays(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_restore_pickle_safe_arrays(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_restore_pickle_safe_arrays(item) for item in value)
    return value


def _numpy_save(path: Path, array: Any) -> None:
    import numpy as np  # type: ignore[import-not-found]

    np.save(path, array, allow_pickle=False)


def _numpy_load(path: Path, *, mmap_mode: str | None) -> Any:
    import numpy as np  # type: ignore[import-not-found]

    return np.load(path, mmap_mode=mmap_mode, allow_pickle=False)


def _record_public_mapping_preserve_arrays(record: Any) -> dict[str, Any]:
    if isinstance(record, Mapping):
        return {str(key): value for key, value in record.items()}
    return _json_safe(record)


def _write_binary_chunk_payload(chunk_path: Path, frames: Sequence[Mapping[str, Any]]) -> None:
    payload = {
        "schema_version": 1,
        "artifact_type": SAM3D_BATCH_BINARY_CHUNK_ARTIFACT_TYPE,
        "contract_version": SAM3D_BATCH_BINARY_CONTRACT_VERSION,
        "array_encoding": "npy_per_bulk_field",
        "frames": list(frames),
    }
    _write_json_payload(chunk_path, payload)


def _resolve_binary_refs(value: Any, *, chunk_path: Path, mmap_mode: str | None, arrays_as: str) -> Any:
    if isinstance(value, Mapping):
        ref = value.get(SAM3D_ARRAY_REF_KEY)
        if isinstance(ref, Mapping):
            rel_path = str(ref.get("path", ""))
            if not rel_path:
                raise ValueError(f"binary SAM3D array ref missing path in {chunk_path}")
            array = _numpy_load(chunk_path.with_suffix("") / rel_path, mmap_mode=mmap_mode)
            return array.tolist() if arrays_as == "list" else array
        return {str(key): _resolve_binary_refs(item, chunk_path=chunk_path, mmap_mode=mmap_mode, arrays_as=arrays_as) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_binary_refs(item, chunk_path=chunk_path, mmap_mode=mmap_mode, arrays_as=arrays_as) for item in value]
    return value


def _frames_from_binary_chunk(
    chunk_path: Path,
    *,
    mmap_mode: str | None = None,
    arrays_as: str = "list",
) -> list[dict[str, Any]]:
    payload = json.loads(chunk_path.read_text(encoding="utf-8"))
    if payload.get("artifact_type") != SAM3D_BATCH_BINARY_CHUNK_ARTIFACT_TYPE:
        raise ValueError(f"binary SAM3D chunk has unexpected artifact_type: {chunk_path}")
    if int(payload.get("contract_version", 0)) != SAM3D_BATCH_BINARY_CONTRACT_VERSION:
        raise ValueError(
            f"unsupported SAM3D binary chunk contract {payload.get('contract_version')!r}; "
            f"expected {SAM3D_BATCH_BINARY_CONTRACT_VERSION}"
        )
    frames = payload.get("frames", [])
    if not isinstance(frames, list):
        raise ValueError(f"binary SAM3D chunk frames are not a list: {chunk_path}")
    resolved = _resolve_binary_refs(frames, chunk_path=chunk_path, mmap_mode=mmap_mode, arrays_as=arrays_as)
    return [dict(frame) for frame in resolved if isinstance(frame, Mapping)]


def _monolithic_payload_from_header(
    payload_header: Mapping[str, Any],
    *,
    frames: Sequence[Mapping[str, Any]],
    batch_execution: Mapping[str, Any],
) -> dict[str, Any]:
    payload = dict(payload_header)
    payload["batch_execution"] = dict(batch_execution)
    payload["frames"] = list(frames)
    return payload


def _read_chunk_index(index_path: Path) -> dict[str, Any]:
    return json.loads(index_path.read_text(encoding="utf-8"))


def _frames_from_chunk_index(index_path: Path) -> list[dict[str, Any]]:
    index = _read_chunk_index(index_path)
    chunk_dir = index_path.parent
    frames: list[dict[str, Any]] = []
    for chunk in index.get("chunks", []):
        chunk_path = chunk_dir / str(chunk["path"])
        frames.extend(_frames_from_chunk(chunk_path, chunk_format=str(chunk.get("format", ""))))
    return frames


def _frames_from_chunk(chunk_path: Path, *, chunk_format: str) -> list[dict[str, Any]]:
    if chunk_format == "binary" or chunk_path.name.endswith(".binary.json"):
        return _frames_from_binary_chunk(chunk_path, mmap_mode=None, arrays_as="list")
    if chunk_format == "pickle" or chunk_path.suffix == ".pkl":
        return _frames_from_pickle_chunk(chunk_path, preserve_arrays=False)
    if chunk_format in ("", "jsonl") or chunk_path.suffix == ".jsonl":
        frames: list[dict[str, Any]] = []
        for line in chunk_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                frame = json.loads(line)
                if isinstance(frame, dict):
                    frames.append(frame)
        return frames
    raise ValueError(f"unsupported SAM3D chunk format {chunk_format!r} for {chunk_path}")


def _frames_from_pickle_chunk(chunk_path: Path, *, preserve_arrays: bool) -> list[dict[str, Any]]:
    output = _bucket_output_from_raw_chunk_payload(_read_pickle_payload(chunk_path))
    frames, _execution = _bucket_output_to_frames_and_execution(output, preserve_arrays=preserve_arrays)
    return frames


def _ordered_frames_from_chunk_index(index_path: Path, request_ids: Sequence[str]) -> list[dict[str, Any]]:
    frames_by_request: dict[str, dict[str, Any]] = {}
    for frame in _frames_from_chunk_index(index_path):
        request_id = str(frame.get("request_id"))
        if request_id in frames_by_request:
            raise RuntimeError(f"duplicate SAM3D output for request {request_id!r}")
        frames_by_request[request_id] = frame
    missing = [str(request_id) for request_id in request_ids if str(request_id) not in frames_by_request]
    if missing:
        raise RuntimeError(f"SAM3D bucket execution produced no output for request {missing[0]!r}")
    return [frames_by_request[str(request_id)] for request_id in request_ids]


def load_sam3dbody_binary_outputs_from_chunk_index(
    index_path: Path,
    *,
    request_ids: Sequence[str],
    mmap_mode: str | None = "r",
) -> list[list[dict[str, Any]]]:
    index = _read_chunk_index(index_path)
    chunk_dir = index_path.parent
    frames_by_request: dict[str, dict[str, Any]] = {}
    for chunk in index.get("chunks", []):
        chunk_path = chunk_dir / str(chunk["path"])
        if str(chunk.get("format", "")) == "binary" or chunk_path.name.endswith(".binary.json"):
            frames = _frames_from_binary_chunk(chunk_path, mmap_mode=mmap_mode, arrays_as="numpy")
        elif str(chunk.get("format", "")) == "pickle" or chunk_path.suffix == ".pkl":
            frames = _frames_from_pickle_chunk(chunk_path, preserve_arrays=True)
        else:
            frames = _frames_from_chunk(chunk_path, chunk_format=str(chunk.get("format", "")))
        for frame in frames:
            request_id = str(frame.get("request_id"))
            if request_id in frames_by_request:
                raise RuntimeError(f"duplicate SAM3D output for request {request_id!r}")
            frames_by_request[request_id] = frame
    missing = [str(request_id) for request_id in request_ids if str(request_id) not in frames_by_request]
    if missing:
        raise RuntimeError(f"SAM3D bucket execution produced no output for request {missing[0]!r}")
    outputs: list[list[dict[str, Any]]] = []
    for request_id in request_ids:
        records = frames_by_request[str(request_id)].get("records", [])
        if not isinstance(records, list):
            raise RuntimeError(f"SAM3D batch records are not a list for request {request_id!r}: {index_path}")
        outputs.append([dict(record) for record in records if isinstance(record, Mapping)])
    return outputs


def _monolithic_payload_from_chunk_index(index_path: Path) -> dict[str, Any]:
    index = _read_chunk_index(index_path)
    template = index.get("monolithic_template")
    if not isinstance(template, Mapping):
        raise ValueError(f"chunk index missing monolithic_template: {index_path}")
    batch_execution = template.get("batch_execution")
    if not isinstance(batch_execution, Mapping):
        raise ValueError(f"chunk index template missing batch_execution: {index_path}")
    return _monolithic_payload_from_header(
        template,
        frames=_frames_from_chunk_index(index_path),
        batch_execution=batch_execution,
    )


def _convert_chunked_output_to_monolithic(index_path: Path, out_path: Path) -> None:
    payload = _monolithic_payload_from_chunk_index(index_path)
    _write_json_payload(out_path, payload)


def _raw_bucket_chunk_payload(output: _BucketInferenceOutput) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_sam3dbody_batch_raw_bucket",
        "bucket_index": int(output.prepared.bucket_index),
        "bucket_size": int(output.prepared.bucket_size),
        "request_ids": [str(request_id) for request_id in output.prepared.request_ids],
        "bucket_items": [dict(item) for item in output.prepared.bucket_items],
        "execution_entry": dict(output.prepared.execution_entry),
        "raw_records": list(output.raw_records),
        "faces": output.faces,
        "body_input_size": output.body_input_size,
        "clip_intrinsics": dict(output.clip_intrinsics) if output.clip_intrinsics is not None else None,
        "optimization": dict(output.optimization),
    }


def _bucket_output_from_raw_chunk_payload(payload: Mapping[str, Any]) -> _BucketInferenceOutput:
    payload = _restore_pickle_safe_arrays(payload)
    if payload.get("artifact_type") != "racketsport_sam3dbody_batch_raw_bucket":
        raise ValueError("SAM3D pickle chunk has unexpected artifact_type")
    bucket_items = [dict(item) for item in payload.get("bucket_items", [])]
    request_ids = [str(request_id) for request_id in payload.get("request_ids", [])]
    real_items = [item for item in bucket_items if not bool(item.get("is_padding"))]
    prepared = _PreparedBucket(
        bucket_index=int(payload["bucket_index"]),
        bucket_size=int(payload["bucket_size"]),
        request_ids=request_ids,
        real_items=real_items,
        bucket_items=bucket_items,
        execution_entry=dict(payload.get("execution_entry", {})),
        prepared_bucket=None,
    )
    return _BucketInferenceOutput(
        prepared=prepared,
        raw_records=list(payload.get("raw_records", [])),
        faces=payload.get("faces"),
        body_input_size=payload.get("body_input_size"),
        clip_intrinsics=payload.get("clip_intrinsics"),
        optimization=payload.get("optimization", {}),
    )


class _BatchOutputStream:
    def __init__(
        self,
        *,
        out_path: Path,
        chunk_dir: Path,
        request_ids: Sequence[str],
        payload_header: Mapping[str, Any],
        write_monolithic: bool,
        chunk_format: str = "pickle",
        timing: _TimingRecorder | None = None,
        async_write: bool = True,
    ) -> None:
        if chunk_format not in CHUNK_FORMATS:
            raise ValueError(f"unsupported chunk format {chunk_format!r}")
        self.out_path = out_path
        self.chunk_dir = chunk_dir
        self.index_path = chunk_dir / "index.json"
        self.request_ids = [str(request_id) for request_id in request_ids]
        self.payload_header = dict(payload_header)
        self.write_monolithic = bool(write_monolithic)
        self.chunk_format = str(chunk_format)
        self.timing = timing or _TimingRecorder()
        # Sync (non-threaded) writes are the default for chunk_format="pickle" (main()
        # passes async_write=False for it): live A100 measurement 2026-07-05 showed the
        # background writer thread contending for the GIL with the main SAM3D
        # inference/prep thread, inflating steady inference 15.3->150ms/person,
        # preprocessing 13->131s, and handoff 376->489s far more than the write I/O
        # itself costs. binary/jsonl keep the async writer thread (opt-in, useful for
        # overlapping slow/large writes with compute).
        self.async_write = bool(async_write)
        self._queue: queue.Queue[_BucketInferenceOutput | _WriterFinish] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None
        self._frames_by_request: dict[str, dict[str, Any]] = {}
        self._seen_request_ids: dict[str, None] = {}
        self._execution_buckets: list[dict[str, Any]] = []
        self._chunks: list[dict[str, Any]] = []

    def start(self) -> None:
        if not self.async_write:
            self.chunk_dir.mkdir(parents=True, exist_ok=True)
            return
        if self._thread is not None:
            return
        self.chunk_dir.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self._run, name="sam3dbody-batch-writer", daemon=True)
        self._thread.start()

    def submit(self, output: _BucketInferenceOutput) -> None:
        self._raise_if_failed()
        if not self.async_write:
            try:
                self._write_bucket_output(output)
            except BaseException as exc:  # noqa: BLE001 - mirrors the async writer's failure handling.
                self._error = exc
                raise
            return
        self._queue.put(output)

    def finish_success(self, *, materialize_frames: bool = True) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        return self._finish(_WriterFinish(status="complete"), materialize_frames=materialize_frames)

    def finish_failure(self, exc: BaseException) -> None:
        try:
            self._finish(_WriterFinish(status="failed", error=exc))
        except BaseException:
            raise

    def _finish(
        self,
        finish: _WriterFinish,
        *,
        materialize_frames: bool = True,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        self.start()
        if self.async_write:
            self._queue.put(finish)
            assert self._thread is not None
            self._thread.join()
        else:
            try:
                self._finalize(finish)
            except BaseException as exc:  # noqa: BLE001 - mirrors the async writer's failure handling.
                self._error = exc
                try:
                    self._write_index(status="failed", batch_execution=self._batch_execution(), template=None, error=exc)
                except OSError:
                    pass
        self._raise_if_failed()
        batch_execution = self._batch_execution()
        if finish.status != "complete":
            return list(self._frames_by_request.values()), batch_execution
        if not materialize_frames:
            return [], batch_execution
        frames = _ordered_frames_from_chunk_index(self.index_path, self.request_ids)
        return frames, batch_execution

    def _finalize(self, finish: _WriterFinish) -> None:
        if finish.status == "complete":
            self._validate_complete()
            batch_execution = self._batch_execution()
            payload = _monolithic_payload_from_header(
                self.payload_header,
                frames=[],
                batch_execution=batch_execution,
            )
            self._write_index(status="complete", batch_execution=batch_execution, template=payload, error=None)
            if self.write_monolithic:
                with self.timing.span("output_write_monolithic", path=str(self.out_path)):
                    _convert_chunked_output_to_monolithic(self.index_path, self.out_path)
        else:
            self._write_index(status="failed", batch_execution=self._batch_execution(), template=None, error=finish.error)

    def _run(self) -> None:
        finish = _WriterFinish(status="failed", error=RuntimeError("writer stopped before final status"))
        try:
            while True:
                item = self._queue.get()
                if isinstance(item, _WriterFinish):
                    finish = item
                    break
                self._write_bucket_output(item)
            self._finalize(finish)
        except BaseException as exc:  # noqa: BLE001 - writer must report failures through the main thread.
            self._error = exc
            try:
                self._write_index(status="failed", batch_execution=self._batch_execution(), template=None, error=exc)
            except OSError:
                pass

    def _write_bucket_output(self, output: _BucketInferenceOutput) -> None:
        execution_entry = dict(output.prepared.execution_entry)
        if self.chunk_format == "jsonl":
            with self.timing.span(
                "bucket_postprocess",
                bucket_index=output.prepared.bucket_index,
                bucket_size=output.prepared.bucket_size,
                request_count=len(output.prepared.real_items),
            ):
                frames, execution_entry = _bucket_output_to_frames_and_execution(output)
            chunk_path = self.chunk_dir / f"bucket_{output.prepared.bucket_index:06d}.jsonl"
            request_ids = [str(frame["request_id"]) for frame in frames]
            with self.timing.span(
                "output_write_bucket",
                bucket_index=output.prepared.bucket_index,
                bucket_size=output.prepared.bucket_size,
                path=str(chunk_path),
                frame_count=len(frames),
                chunk_format=self.chunk_format,
            ):
                _write_jsonl_lines(chunk_path, frames)
            for frame in frames:
                request_id = str(frame["request_id"])
                self._mark_request_id(request_id)
                self._frames_by_request[request_id] = frame
        elif self.chunk_format == "binary":
            with self.timing.span(
                "bucket_postprocess",
                bucket_index=output.prepared.bucket_index,
                bucket_size=output.prepared.bucket_size,
                request_count=len(output.prepared.real_items),
            ):
                frames, execution_entry = _bucket_output_to_frames_and_execution(output, preserve_arrays=True)
            chunk_path = self.chunk_dir / f"bucket_{output.prepared.bucket_index:06d}.binary.json"
            request_ids = [str(frame["request_id"]) for frame in frames]
            with self.timing.span(
                "output_write_bucket",
                bucket_index=output.prepared.bucket_index,
                bucket_size=output.prepared.bucket_size,
                path=str(chunk_path),
                frame_count=len(frames),
                chunk_format=self.chunk_format,
            ):
                writer = _BinaryArrayWriter(chunk_path=chunk_path)
                binary_frames = [_frame_with_binary_arrays(frame, writer=writer) for frame in frames]
                _write_binary_chunk_payload(chunk_path, binary_frames)
            for frame in frames:
                request_id = str(frame["request_id"])
                self._mark_request_id(request_id)
                self._frames_by_request[request_id] = frame
        else:
            frames = []
            chunk_path = self.chunk_dir / f"bucket_{output.prepared.bucket_index:06d}.pkl"
            request_ids = [str(request_id) for request_id in output.prepared.request_ids]
            payload = _pickle_safe_arrays(_raw_bucket_chunk_payload(output))
            with self.timing.span(
                "output_write_bucket",
                bucket_index=output.prepared.bucket_index,
                bucket_size=output.prepared.bucket_size,
                path=str(chunk_path),
                frame_count=len(request_ids),
                chunk_format=self.chunk_format,
            ):
                _write_pickle_payload(chunk_path, payload)
            for request_id in request_ids:
                self._mark_request_id(request_id)
        execution_entry = dict(execution_entry)
        execution_entry["output_chunk"] = str(chunk_path.relative_to(self.chunk_dir))
        execution_entry["output_chunk_format"] = self.chunk_format
        self._execution_buckets.append(execution_entry)
        self._chunks.append(
            {
                "bucket_index": output.prepared.bucket_index,
                "path": str(chunk_path.relative_to(self.chunk_dir)),
                "format": self.chunk_format,
                "request_ids": request_ids,
                "result_count": len(request_ids),
            }
        )

    def _mark_request_id(self, request_id: str) -> None:
        if request_id in self._seen_request_ids:
            raise RuntimeError(f"duplicate SAM3D output for request {request_id!r}")
        self._seen_request_ids[request_id] = None

    def _batch_execution(self) -> dict[str, Any]:
        timing_mode = _timing_mode(self.payload_header.get("optimization", {}))
        return {
            "mode": "real_bucketed_body_batch",
            "timing_mode": timing_mode,
            "timing_notes": _timing_notes(timing_mode),
            "buckets": list(self._execution_buckets),
            "request_count": len(self.request_ids),
            "output_count": len(self._seen_request_ids),
            "streaming_output": {
                "format": f"{self.chunk_format}_chunks_with_monolithic_converter",
                "chunk_dir": str(self.chunk_dir),
                "index_path": str(self.index_path),
                "chunk_count": len(self._chunks),
                "monolithic_written": bool(self.write_monolithic),
            },
        }

    def _validate_complete(self) -> None:
        missing = [request_id for request_id in self.request_ids if request_id not in self._seen_request_ids]
        if missing:
            raise RuntimeError(f"SAM3D bucket execution produced no output for request {missing[0]!r}")
        if len(self._seen_request_ids) != len(self.request_ids):
            raise RuntimeError(
                f"SAM3D bucket execution produced {len(self._seen_request_ids)} unique outputs "
                f"for {len(self.request_ids)} requests"
            )

    def _write_index(
        self,
        *,
        status: str,
        batch_execution: Mapping[str, Any],
        template: Mapping[str, Any] | None,
        error: BaseException | None,
    ) -> None:
        payload = {
            "schema_version": 1,
            "artifact_type": "racketsport_sam3dbody_batch_chunk_index",
            "status": status,
            "out_path": str(self.out_path),
            "chunk_dir": str(self.chunk_dir),
            "request_count": len(self.request_ids),
            "result_count": len(self._seen_request_ids),
            "request_ids": list(self._seen_request_ids.keys()),
            "chunks": list(self._chunks),
            "batch_execution": dict(batch_execution),
            "monolithic_template": dict(template) if template is not None else None,
            "error": (
                {
                    "type": type(error).__name__,
                    "message": str(error),
                }
                if error is not None
                else None
            ),
        }
        _write_json_payload(self.index_path, payload)

    def _raise_if_failed(self) -> None:
        if self._error is not None:
            raise self._error


def _parse_batch_payload(payload: Any) -> dict[str, Any]:
    requests = _parse_requests(payload)
    optimization = _parse_optimization(payload.get("optimization") if isinstance(payload, Mapping) else None)
    _validate_warmup_shape_contract(optimization)
    clip_intrinsics = _parse_clip_intrinsics(payload.get("clip_intrinsics") if isinstance(payload, Mapping) else None)
    _validate_request_intrinsics_against_clip(requests, clip_intrinsics)
    return {
        "requests": requests,
        "clip_intrinsics": clip_intrinsics,
        "optimization": optimization,
        "bucket_plan": _bucket_plan(
            [str(request["request_id"]) for request in requests],
            bucket_sizes=optimization["crop_bucket_sizes"],
        ),
    }


def _parse_optimization(raw_value: Any) -> dict[str, Any]:
    raw = raw_value if isinstance(raw_value, Mapping) else {}
    return {
        "sam3d_body_input_size_px": (
            normalize_body_input_size(raw.get("sam3d_body_input_size_px"))
            if raw.get("sam3d_body_input_size_px") is not None
            else None
        ),
        "crop_bucket_sizes": _positive_int_list(raw.get("crop_bucket_sizes", []), name="optimization.crop_bucket_sizes"),
        "torch_compile": bool(raw.get("torch_compile", False)),
        "compile_warmup_buckets": _positive_int_list(
            raw.get("compile_warmup_buckets", []),
            name="optimization.compile_warmup_buckets",
        ),
        "compile_warmup_passes": _positive_int(
            raw.get("compile_warmup_passes", 2),
            name="optimization.compile_warmup_passes",
        ),
        "batching": str(raw.get("batching", "static_intrinsics_cross_frame_bucketed_body_batch")),
        "steady_state_empty_cache": _bool_flag(
            raw.get("steady_state_empty_cache", True),
            name="optimization.steady_state_empty_cache",
        ),
        "inner_bucket_sync": _bool_flag(
            raw.get("inner_bucket_sync", True),
            name="optimization.inner_bucket_sync",
        ),
        "upstream_env": _parse_upstream_env(raw.get("upstream_env", {})),
        "tier2_output_lite": _bool_flag(
            raw.get("tier2_output_lite", False),
            name="optimization.tier2_output_lite",
        ),
        "prefetch_buckets": _nonnegative_int(
            raw.get("prefetch_buckets", 1),
            name="optimization.prefetch_buckets",
        ),
    }


def _bool_flag(raw_value: Any, *, name: str) -> bool:
    if isinstance(raw_value, bool):
        return raw_value
    if isinstance(raw_value, int) and raw_value in (0, 1):
        return bool(raw_value)
    if isinstance(raw_value, str):
        normalized = raw_value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"{name} must be a boolean")


def _parse_upstream_env(raw_value: Any) -> dict[str, str]:
    if raw_value in (None, ""):
        return {}
    if not isinstance(raw_value, Mapping):
        raise ValueError("optimization.upstream_env must be an object")
    parsed: dict[str, str] = {}
    for raw_key, raw_env_value in raw_value.items():
        key = str(raw_key)
        if key not in UPSTREAM_ENV_WHITELIST:
            raise ValueError(f"unsupported upstream_env key {key!r}; allowed keys are {sorted(UPSTREAM_ENV_WHITELIST)}")
        value = _env_value(raw_env_value, name=f"optimization.upstream_env.{key}")
        if key == "COMPILE_MODE" and value not in COMPILE_MODE_VALUES:
            raise ValueError(f"optimization.upstream_env.COMPILE_MODE must be one of {sorted(COMPILE_MODE_VALUES)}")
        parsed[key] = value
    return parsed


def _env_value(raw_value: Any, *, name: str) -> str:
    if isinstance(raw_value, bool):
        return "1" if raw_value else "0"
    if isinstance(raw_value, int):
        return str(raw_value)
    if isinstance(raw_value, str) and raw_value:
        return raw_value
    raise ValueError(f"{name} must be a non-empty string, integer, or boolean")


def _validate_warmup_shape_contract(optimization: Mapping[str, Any]) -> None:
    crop_bucket_sizes = [int(value) for value in optimization.get("crop_bucket_sizes", [])]
    warmup_buckets = [int(value) for value in optimization.get("compile_warmup_buckets", [])]
    if not optimization.get("torch_compile") or not crop_bucket_sizes and not warmup_buckets:
        return
    if sorted(set(crop_bucket_sizes)) != sorted(set(warmup_buckets)):
        raise ValueError(
            "compile_warmup_buckets must match crop_bucket_sizes for static-shape SAM3D compile warmup; "
            f"crop_bucket_sizes={crop_bucket_sizes}, compile_warmup_buckets={warmup_buckets}"
        )


def _parse_clip_intrinsics(raw_value: Any) -> dict[str, Any] | None:
    if raw_value in (None, ""):
        return None
    if not isinstance(raw_value, Mapping):
        raise ValueError("clip_intrinsics must be an object")
    matrix = raw_value.get("matrix")
    if matrix is None:
        try:
            fx = float(raw_value["fx"])
            fy = float(raw_value["fy"])
            cx = float(raw_value["cx"])
            cy = float(raw_value["cy"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("clip_intrinsics requires matrix or fx/fy/cx/cy") from exc
        matrix = [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]
    parsed_matrix = _parse_camera_intrinsics(matrix)
    if parsed_matrix is None:
        raise ValueError("clip_intrinsics matrix must be a 3x3 array")
    for name, row_index, col_index in (("fx", 0, 0), ("fy", 1, 1), ("cx", 0, 2), ("cy", 1, 2)):
        if name in raw_value and raw_value.get(name) is not None:
            raw_number = float(raw_value[name])
            matrix_number = float(parsed_matrix[row_index][col_index])
            if abs(raw_number - matrix_number) > 1e-6:
                raise ValueError(
                    f"clip_intrinsics.{name}={raw_number} disagrees with clip_intrinsics.matrix"
                    f"[{row_index}][{col_index}]={matrix_number}"
                )
    return {
        "fx": float(parsed_matrix[0][0]),
        "fy": float(parsed_matrix[1][1]),
        "cx": float(parsed_matrix[0][2]),
        "cy": float(parsed_matrix[1][2]),
        "dist": [float(value) for value in (raw_value.get("dist") or [])],
        "source": str(raw_value.get("source", "")),
        "matrix": parsed_matrix,
        "static_per_clip": bool(raw_value.get("static_per_clip", True)),
    }


def _validate_request_intrinsics_against_clip(
    requests: Sequence[Mapping[str, Any]],
    clip_intrinsics: Mapping[str, Any] | None,
) -> None:
    if clip_intrinsics is None:
        return
    clip_matrix = clip_intrinsics.get("matrix")
    for request in requests:
        request_matrix = request.get("camera_intrinsics")
        if request_matrix is None:
            continue
        if not _matrix_close(request_matrix, clip_matrix):
            request_id = str(request.get("request_id", ""))
            raise ValueError(
                f"requests/{request_id}/camera_intrinsics does not match clip_intrinsics.matrix; "
                "batch-level clip_intrinsics is authoritative"
            )


def _matrix_close(left: Any, right: Any, *, tolerance: float = 1e-6) -> bool:
    if not isinstance(left, list) or not isinstance(right, list) or len(left) != len(right):
        return False
    for left_row, right_row in zip(left, right, strict=True):
        if not isinstance(left_row, list) or not isinstance(right_row, list) or len(left_row) != len(right_row):
            return False
        for left_value, right_value in zip(left_row, right_row, strict=True):
            if abs(float(left_value) - float(right_value)) > tolerance:
                return False
    return True


def _positive_int_list(raw_value: Any, *, name: str) -> list[int]:
    if raw_value in (None, ""):
        return []
    if not isinstance(raw_value, list):
        raise ValueError(f"{name} must be a list")
    values: list[int] = []
    for index, raw in enumerate(raw_value):
        value = int(raw)
        if value <= 0:
            raise ValueError(f"{name}/{index} must be positive")
        values.append(value)
    return values


def _positive_int(raw_value: Any, *, name: str) -> int:
    if isinstance(raw_value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        value = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _nonnegative_int(raw_value: Any, *, name: str) -> int:
    if isinstance(raw_value, bool):
        raise ValueError(f"{name} must be a non-negative integer")
    try:
        value = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a non-negative integer") from exc
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _apply_bucket_size_override(optimization: Mapping[str, Any], *, bucket_size: int | None) -> dict[str, Any]:
    parsed = _parse_optimization(optimization)
    if bucket_size is None:
        return parsed
    size = _positive_int(bucket_size, name="--bucket-size")
    parsed["crop_bucket_sizes"] = [size]
    if parsed.get("torch_compile"):
        parsed["compile_warmup_buckets"] = [size]
    return parsed


def _bucket_plan(request_ids: list[str], *, bucket_sizes: list[int]) -> list[dict[str, Any]]:
    if not request_ids:
        return []
    buckets = sorted(set(bucket_sizes))
    if not buckets:
        return [
            {
                "bucket_size": len(request_ids),
                "request_ids": list(request_ids),
                "real_request_count": len(request_ids),
                "padding_count": 0,
                "padded_request_count": len(request_ids),
                "padded_crop_ratio": 0.0,
            }
        ]
    plan: list[dict[str, Any]] = []
    pending = list(request_ids)
    while pending:
        remaining = len(pending)
        bucket_size = next((bucket for bucket in buckets if bucket >= remaining), buckets[-1])
        take = min(remaining, bucket_size)
        group = pending[:take]
        del pending[:take]
        padding_count = bucket_size - len(group)
        plan.append(
            {
                "bucket_size": bucket_size,
                "request_ids": group,
                "real_request_count": len(group),
                "padding_count": padding_count,
                "padded_request_count": bucket_size,
                "padded_crop_ratio": padding_count / bucket_size if bucket_size else 0.0,
            }
        )
    return plan


def _production_decoder_callable(estimator: Any) -> Any | None:
    model = getattr(estimator, "model", None)
    if model is None:
        return None
    candidate = getattr(model, "forward_decoder", None)
    return candidate if callable(candidate) else None


def _callable_identity(callable_obj: Any) -> dict[str, Any]:
    bound_self = getattr(callable_obj, "__self__", None)
    bound_func = getattr(callable_obj, "__func__", None)
    if bound_self is not None and bound_func is not None:
        return {
            "kind": "bound_method",
            "self_type": type(bound_self).__name__,
            "self_id": id(bound_self),
            "function_module": str(getattr(bound_func, "__module__", "")),
            "function_qualname": str(getattr(bound_func, "__qualname__", "")),
            "function_id": id(bound_func),
        }
    return {
        "kind": "callable",
        "callable_type": type(callable_obj).__name__,
        "callable_id": id(callable_obj),
        "module": str(getattr(callable_obj, "__module__", "")),
        "qualname": str(getattr(callable_obj, "__qualname__", "")),
    }


def _production_callable_identity(estimator: Any) -> dict[str, Any] | None:
    candidate = _production_decoder_callable(estimator)
    if candidate is None:
        return None
    identity = _callable_identity(candidate)
    identity["path"] = "estimator.model.forward_decoder"
    return identity


def _remember_warmed_production_callable(estimator: Any) -> dict[str, Any] | None:
    identity = _production_callable_identity(estimator)
    if identity is not None:
        setattr(estimator, PRODUCTION_CALLABLE_IDENTITY_ATTR, identity)
    return identity


def _assert_warmed_production_callable(estimator: Any) -> dict[str, Any] | None:
    expected = getattr(estimator, PRODUCTION_CALLABLE_IDENTITY_ATTR, None)
    if expected is None:
        return None
    current = _production_callable_identity(estimator)
    if current != expected:
        raise RuntimeError(
            "production SAM3D decoder callable changed after warmup; "
            f"warmed={expected}, current={current}"
        )
    return current


def _run_bucketed_inference(
    estimator: Any,
    requests: Sequence[Mapping[str, Any]],
    *,
    bucket_plan: Sequence[Mapping[str, Any]],
    clip_cam_int: Any | None,
    faces: Any,
    body_input_size: int | None,
    clip_intrinsics: Mapping[str, Any] | None = None,
    optimization: Mapping[str, Any] | None = None,
    output_stream: _BatchOutputStream | None = None,
    timing: _TimingRecorder | None = None,
    materialize_streamed_frames: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    optimization = _parse_optimization(optimization or {})
    request_by_id = {str(request["request_id"]): request for request in requests}
    frames_by_request: dict[str, dict[str, Any]] = {}
    execution_buckets: list[dict[str, Any]] = []
    timing = timing or _TimingRecorder()
    if output_stream is not None:
        output_stream.start()
    if not optimization["inner_bucket_sync"]:
        _synchronize_cuda_boundary()
    try:
        for prepared in _iter_prepared_buckets(
            estimator,
            bucket_plan,
            request_by_id=request_by_id,
            clip_cam_int=clip_cam_int,
            optimization=optimization,
            timing=timing,
        ):
            with timing.span(
                "bucket_inference",
                bucket_index=prepared.bucket_index,
                bucket_size=prepared.bucket_size,
                request_count=len(prepared.real_items),
                padded_request_count=len(prepared.bucket_items),
            ):
                _assert_warmed_production_callable(estimator)
                raw_records = _process_sam3d_body_bucket(
                    estimator,
                    prepared.bucket_items,
                    clip_cam_int=clip_cam_int,
                    optimization=optimization,
                    prepared_bucket=prepared.prepared_bucket,
                )
            output = _BucketInferenceOutput(
                prepared=prepared,
                raw_records=raw_records,
                faces=faces,
                body_input_size=body_input_size,
                clip_intrinsics=clip_intrinsics,
                optimization=optimization,
            )
            if output_stream is not None:
                output_stream.submit(output)
            else:
                with timing.span(
                    "bucket_postprocess",
                    bucket_index=prepared.bucket_index,
                    bucket_size=prepared.bucket_size,
                    request_count=len(prepared.real_items),
                ):
                    frames, execution_entry = _bucket_output_to_frames_and_execution(output)
                for frame in frames:
                    request_id = str(frame["request_id"])
                    if request_id in frames_by_request:
                        raise RuntimeError(f"duplicate SAM3D output for request {request_id!r}")
                    frames_by_request[request_id] = frame
                execution_buckets.append(execution_entry)
        if output_stream is not None:
            frames, batch_execution = output_stream.finish_success(materialize_frames=materialize_streamed_frames)
        else:
            frames = []
            for request in requests:
                request_id = str(request["request_id"])
                if request_id not in frames_by_request:
                    raise RuntimeError(f"SAM3D bucket execution produced no output for request {request_id!r}")
                frames.append(frames_by_request[request_id])
            timing_mode = _timing_mode(optimization)
            batch_execution = {
                "mode": "real_bucketed_body_batch",
                "timing_mode": timing_mode,
                "timing_notes": _timing_notes(timing_mode),
                "buckets": execution_buckets,
                "request_count": len(requests),
                "output_count": len(frames),
            }
    except BaseException as exc:
        if output_stream is not None:
            output_stream.finish_failure(exc)
        raise
    if output_stream is None:
        frames = []
        for request in requests:
            request_id = str(request["request_id"])
            frames.append(frames_by_request[request_id])
    if not optimization["inner_bucket_sync"]:
        _synchronize_cuda_boundary()
    return frames, batch_execution


def _iter_prepared_buckets(
    estimator: Any,
    bucket_plan: Sequence[Mapping[str, Any]],
    *,
    request_by_id: Mapping[str, Mapping[str, Any]],
    clip_cam_int: Any | None,
    optimization: Mapping[str, Any],
    timing: _TimingRecorder,
) -> Any:
    prefetch_buckets = int(optimization.get("prefetch_buckets", 1))
    if prefetch_buckets <= 0:
        for bucket_index, bucket in enumerate(bucket_plan):
            prepared = _prepare_bucket_job(
                estimator,
                bucket_index,
                bucket,
                request_by_id=request_by_id,
                clip_cam_int=clip_cam_int,
                optimization=optimization,
                timing=timing,
            )
            if prepared is not None:
                yield prepared
        return

    work_queue: queue.Queue[_PreparedBucket | BaseException | None] = queue.Queue(maxsize=prefetch_buckets)
    stop_event = threading.Event()

    def put_item(item: _PreparedBucket | BaseException | None) -> None:
        while not stop_event.is_set():
            try:
                work_queue.put(item, timeout=0.1)
                return
            except queue.Full:
                continue

    def load() -> None:
        try:
            for bucket_index, bucket in enumerate(bucket_plan):
                if stop_event.is_set():
                    return
                prepared = _prepare_bucket_job(
                    estimator,
                    bucket_index,
                    bucket,
                    request_by_id=request_by_id,
                    clip_cam_int=clip_cam_int,
                    optimization=optimization,
                    timing=timing,
                )
                if prepared is not None:
                    put_item(prepared)
        except BaseException as exc:  # noqa: BLE001 - propagated on the consumer thread.
            put_item(exc)
        finally:
            put_item(None)

    thread = threading.Thread(target=load, name="sam3dbody-bucket-prefetch", daemon=True)
    thread.start()
    try:
        while True:
            item = work_queue.get()
            if item is None:
                break
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        stop_event.set()
        thread.join()


def _prepare_bucket_job(
    estimator: Any,
    bucket_index: int,
    bucket: Mapping[str, Any],
    *,
    request_by_id: Mapping[str, Mapping[str, Any]],
    clip_cam_int: Any | None,
    optimization: Mapping[str, Any],
    timing: _TimingRecorder,
) -> _PreparedBucket | None:
    request_ids = [str(request_id) for request_id in bucket.get("request_ids", [])]
    real_items: list[dict[str, Any]] = []
    for request_id in request_ids:
        if request_id not in request_by_id:
            raise ValueError(f"bucket_plan/{bucket_index} references unknown request_id {request_id!r}")
        request = request_by_id[request_id]
        bboxes = request.get("bboxes", [])
        if len(bboxes) != 1:
            raise ValueError(
                "real SAM3D bucketed body execution requires exactly one crop per request; "
                f"request {request_id!r} has {len(bboxes)} bboxes"
            )
        mask_paths = list(request.get("mask_paths", []))
        real_items.append(
            {
                "request": request,
                "request_id": request_id,
                "bbox": list(bboxes[0]),
                "bbox_index": 0,
                "mask_path": mask_paths[0] if mask_paths else None,
                "is_padding": False,
                "target_representation": str(request.get("target_representation", "world_mesh")),
            }
        )
    if not real_items:
        return None
    bucket_size = int(bucket.get("bucket_size", len(real_items)))
    if bucket_size < len(real_items):
        raise ValueError(
            f"bucket_plan/{bucket_index} bucket_size {bucket_size} is smaller than real request count "
            f"{len(real_items)}"
        )
    bucket_items = list(real_items)
    while len(bucket_items) < bucket_size:
        padded = dict(real_items[-1])
        padded["is_padding"] = True
        bucket_items.append(padded)
    with timing.span("request_prep", bucket_index=bucket_index, bucket_size=bucket_size, request_count=len(real_items)):
        prepared_bucket = _prepare_sam3d_body_bucket(
            estimator,
            bucket_items,
            clip_cam_int=clip_cam_int,
        )
    return _PreparedBucket(
        bucket_index=bucket_index,
        bucket_size=bucket_size,
        request_ids=request_ids,
        real_items=real_items,
        bucket_items=bucket_items,
        execution_entry={
            "bucket_index": bucket_index,
            "bucket_size": bucket_size,
            "real_request_count": len(real_items),
            "padding_count": bucket_size - len(real_items),
            "padded_request_count": len(bucket_items),
            "padded_crop_ratio": (bucket_size - len(real_items)) / bucket_size if bucket_size else 0.0,
            "request_ids": request_ids,
            "execution_shape": [1, bucket_size],
            "warmup_shape_contract": "same_batch_size_as_real_execution",
        },
        prepared_bucket=prepared_bucket,
    )


def _prepare_sam3d_body_bucket(
    estimator: Any,
    bucket_items: Sequence[Mapping[str, Any]],
    *,
    clip_cam_int: Any | None,
) -> Any:
    prepare_hook = getattr(estimator, "prepare_body_bucket", None)
    if callable(prepare_hook):
        return prepare_hook(list(bucket_items), cam_int=clip_cam_int)
    if callable(getattr(estimator, "process_body_bucket", None)):
        return None
    return _build_sam3d_body_batch(
        estimator,
        bucket_items,
        clip_cam_int=clip_cam_int,
        target_device=None,
    )


def _bucket_output_to_frames_and_execution(
    output: _BucketInferenceOutput,
    *,
    preserve_arrays: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if preserve_arrays:
        records = [_record_public_mapping_preserve_arrays(record) for record in _extract_person_records(output.raw_records)]
    else:
        records = [_json_safe(record) for record in _extract_person_records(output.raw_records)]
    bucket_items = output.prepared.bucket_items
    if len(records) != len(bucket_items):
        raise RuntimeError(
            f"SAM3D bucket {output.prepared.bucket_index} returned {len(records)} records "
            f"for bucket size {len(bucket_items)}"
        )
    frames: list[dict[str, Any]] = []
    optimization = _parse_optimization(output.optimization)
    for item, record in zip(bucket_items, records, strict=True):
        if item["is_padding"]:
            continue
        target_representation = str(item.get("target_representation", "world_mesh"))
        record = _output_record_for_representation(
            dict(record),
            target_representation=target_representation,
            tier2_output_lite=optimization["tier2_output_lite"],
        )
        if (
            _has_faces(output.faces)
            and target_representation != TIER2_BODY_JOINTS_REPRESENTATION
            and isinstance(record, dict)
            and "mesh_faces" not in record
            and "faces" not in record
        ):
            record["mesh_faces"] = output.faces
        request = item["request"]
        request_id = str(request["request_id"])
        frames.append(
            {
                "request_id": request_id,
                "image_path": str(Path(request["image"]).resolve()),
                "requested_bboxes": request["bboxes"],
                "requested_masks": [str(path) for path in request.get("mask_paths", [])],
                "camera_intrinsics": (
                    output.clip_intrinsics["matrix"]
                    if output.clip_intrinsics is not None
                    else request.get("camera_intrinsics")
                ),
                "camera_intrinsics_source": (
                    output.clip_intrinsics.get("source", "")
                    if output.clip_intrinsics is not None
                    else ("request" if request.get("camera_intrinsics") is not None else "")
                ),
                "sam3d_body_input_size_px": output.body_input_size or request.get("sam3d_body_input_size_px"),
                "records": [record],
                "summary": {"record_count": 1},
                "target_representation": target_representation,
            }
        )
    return frames, dict(output.prepared.execution_entry)


def _has_faces(value: Any) -> bool:
    if value is None:
        return False
    item = value
    tolist = getattr(item, "tolist", None)
    if callable(tolist):
        try:
            item = tolist()
        except Exception:
            return True
    if isinstance(item, Sequence) and not isinstance(item, str | bytes | bytearray):
        return len(item) > 0
    return bool(item)


def _frame_with_binary_arrays(frame: Mapping[str, Any], *, writer: _BinaryArrayWriter) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in frame.items():
        if key == "records" and isinstance(value, list):
            out[key] = [
                {
                    str(record_key): writer.maybe_ref(str(record_key), record_value)
                    for record_key, record_value in record.items()
                }
                if isinstance(record, Mapping)
                else _json_safe(record)
                for record in value
            ]
        else:
            out[str(key)] = _json_safe(value)
    return out


def _process_sam3d_body_bucket(
    estimator: Any,
    bucket_items: Sequence[Mapping[str, Any]],
    *,
    clip_cam_int: Any | None,
    optimization: Mapping[str, Any],
    prepared_bucket: Any = None,
) -> list[Any]:
    prepared_hook = getattr(estimator, "process_prepared_body_bucket", None)
    if callable(prepared_hook):
        return list(prepared_hook(list(bucket_items), prepared_bucket, cam_int=clip_cam_int))
    test_hook = getattr(estimator, "process_body_bucket", None)
    if callable(test_hook):
        return list(test_hook(list(bucket_items), cam_int=clip_cam_int))
    return _process_sam3d_body_bucket_direct(
        estimator,
        bucket_items,
        clip_cam_int=clip_cam_int,
        optimization=optimization,
        prepared_batch=prepared_bucket,
    )


def _prepare_estimator_for_bucket(estimator: Any, torch_module: Any, *, optimization: Mapping[str, Any]) -> None:
    estimator.batch = None
    estimator.image_embeddings = None
    estimator.output = None
    estimator.prev_prompt = []
    _clear_cuda_cache_if_enabled(torch_module, enabled=optimization["steady_state_empty_cache"])


def _build_sam3d_body_batch(
    estimator: Any,
    bucket_items: Sequence[Mapping[str, Any]],
    *,
    clip_cam_int: Any | None,
    target_device: str | None = "cuda",
) -> _Sam3DBodyBatch:
    import cv2  # type: ignore[import-not-found]
    import numpy as np  # type: ignore[import-not-found]
    import torch  # type: ignore[import-not-found]
    from sam_3d_body.data.utils.io import load_image  # type: ignore[import-not-found]
    from sam_3d_body.utils import recursive_to  # type: ignore[import-not-found]
    from torch.utils.data import default_collate  # type: ignore[import-not-found]

    data_list: list[Any] = []
    rgb_images: list[Any] = []
    mask_scores: list[float] = []
    for item in bucket_items:
        img_bgr = _load_bucket_item_bgr(item, load_image=load_image, np=np)
        height, width = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        mask, mask_score = _load_bucket_item_mask(
            item,
            image_shape_hw=(height, width),
            cv2=cv2,
            np=np,
        )
        mask_scores.append(mask_score)
        data_info = {
            "img": img_rgb,
            "bbox": np.asarray(item["bbox"], dtype=np.float32),
            "bbox_format": "xyxy",
            "mask": mask,
            "mask_score": np.asarray(mask_score, dtype=np.float32),
        }
        data_list.append(estimator.transform(data_info))
        rgb_images.append(img_rgb)

    batch = default_collate(data_list)
    for key in [
        "img",
        "img_size",
        "ori_img_size",
        "bbox_center",
        "bbox_scale",
        "bbox",
        "affine_trans",
        "mask",
        "mask_score",
    ]:
        if key in batch:
            batch[key] = batch[key].unsqueeze(0).float()
    if "mask" in batch:
        batch["mask"] = batch["mask"].unsqueeze(2)
    batch["person_valid"] = torch.ones((1, len(bucket_items)))
    if clip_cam_int is not None:
        batch["cam_int"] = clip_cam_int.to(batch["img"]).clone()
    elif "cam_int" not in batch:
        first_image = rgb_images[0]
        height, width = first_image.shape[:2]
        focal = float((height**2 + width**2) ** 0.5)
        batch["cam_int"] = torch.tensor(
            [[[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]]],
            dtype=torch.float32,
        )

    return _Sam3DBodyBatch(
        batch=recursive_to(batch, target_device) if target_device is not None else batch,
        rgb_images=rgb_images,
        mask_scores=mask_scores,
    )


def _load_bucket_item_bgr(item: Mapping[str, Any], *, load_image: Any, np: Any) -> Any:
    synthetic = item.get("synthetic_img_bgr")
    if synthetic is not None:
        return np.asarray(synthetic, dtype=np.uint8).copy()
    image_path = Path(item["request"]["image"])
    return load_image(str(image_path.resolve()), backend="cv2", image_format="bgr")


def _load_bucket_item_mask(
    item: Mapping[str, Any],
    *,
    image_shape_hw: tuple[int, int],
    cv2: Any,
    np: Any,
) -> tuple[Any, float]:
    height, width = image_shape_hw
    synthetic = item.get("synthetic_mask")
    if synthetic is not None:
        mask = np.asarray(synthetic, dtype=np.uint8)
        if mask.ndim == 2:
            mask = mask.reshape(height, width, 1)
        return mask.copy(), float(item.get("synthetic_mask_score", 0.0))
    mask_path = item.get("mask_path")
    if mask_path is not None:
        mask = cv2.imread(str(Path(mask_path)), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"unable to read SAM3D mask prompt {mask_path}")
        if mask.shape[:2] != (height, width):
            raise ValueError(
                f"SAM3D mask prompt {mask_path} has shape {mask.shape[:2]}, expected {(height, width)}"
            )
        return mask.reshape(height, width, 1).astype(np.uint8), 1.0
    return np.zeros((height, width, 1), dtype=np.uint8), 0.0


def _run_sam3d_body_model(
    estimator: Any,
    batch_input: _Sam3DBodyBatch,
    *,
    optimization: Mapping[str, Any],
    torch_module: Any,
) -> Any:
    estimator.model._initialize_batch(batch_input.batch)
    _synchronize_cuda_if_available(torch_module, enabled=optimization["inner_bucket_sync"])
    outputs = estimator.model.run_inference(
        batch_input.rgb_images[0],
        batch_input.batch,
        inference_type="body",
        transform_hand=estimator.transform_hand,
        thresh_wrist_angle=estimator.thresh_wrist_angle,
        hand_box_source="body_decoder",
    )
    _synchronize_cuda_if_available(torch_module, enabled=optimization["inner_bucket_sync"])
    return outputs


def _synthetic_sam3d_bucket_item(
    index: int,
    *,
    bucket_size: int,
    image_size_hw: tuple[int, int],
) -> dict[str, Any]:
    import numpy as np  # type: ignore[import-not-found]

    height, width = int(image_size_hw[0]), int(image_size_hw[1])
    bbox = [
        float(width) * 0.25,
        float(height) * 0.15,
        float(width) * 0.75,
        float(height) * 0.95,
    ]
    request_id = f"warmup-{bucket_size}-{index}"
    return {
        "request": {
            "request_id": request_id,
            "image": Path(f"__synthetic_sam3d_warmup_{bucket_size}_{index}.jpg"),
            "bboxes": [bbox],
            "mask_paths": [],
            "camera_intrinsics": None,
            "target_representation": "world_mesh",
        },
        "request_id": request_id,
        "bbox": bbox,
        "bbox_index": 0,
        "mask_path": None,
        "is_padding": False,
        "target_representation": "world_mesh",
        "synthetic_img_bgr": np.zeros((height, width, 3), dtype=np.uint8),
        "synthetic_mask": np.zeros((height, width, 1), dtype=np.uint8),
        "synthetic_mask_score": 0.0,
    }


def _synthetic_sam3d_warmup_bucket_items(
    bucket_size: int,
    *,
    image_size_hw: tuple[int, int],
) -> list[dict[str, Any]]:
    return [
        _synthetic_sam3d_bucket_item(index, bucket_size=bucket_size, image_size_hw=image_size_hw)
        for index in range(int(bucket_size))
    ]


def _sam3d_batch_guard_signature(batch: Mapping[str, Any], torch_module: Any | None = None) -> dict[str, Any]:
    if torch_module is None:
        import torch as torch_module  # type: ignore[import-not-found,no-redef]

    tensors = {}
    for key, value in batch.items():
        if torch_module.is_tensor(value):
            tensors[key] = {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "device": str(value.device),
                "stride": list(value.stride()),
                "is_contiguous": bool(value.is_contiguous()),
                "requires_grad": bool(value.requires_grad),
            }
    return {
        "keys": sorted(str(key) for key in batch.keys()),
        "tensors": tensors,
        "grad_enabled": bool(torch_module.is_grad_enabled()),
        "inference_mode": bool(torch_module.is_inference_mode_enabled()),
    }


def _sam3d_body_batch_to_cuda(batch_input: _Sam3DBodyBatch) -> _Sam3DBodyBatch:
    from sam_3d_body.utils import recursive_to  # type: ignore[import-not-found]

    return _Sam3DBodyBatch(
        batch=recursive_to(batch_input.batch, "cuda"),
        rgb_images=batch_input.rgb_images,
        mask_scores=batch_input.mask_scores,
    )


def _process_sam3d_body_bucket_direct(
    estimator: Any,
    bucket_items: Sequence[Mapping[str, Any]],
    *,
    clip_cam_int: Any | None,
    optimization: Mapping[str, Any],
    prepared_batch: Any = None,
) -> list[dict[str, Any]]:
    import numpy as np  # type: ignore[import-not-found]
    import torch  # type: ignore[import-not-found]
    from sam_3d_body.utils import recursive_to  # type: ignore[import-not-found]

    with torch.inference_mode():
        _prepare_estimator_for_bucket(estimator, torch, optimization=optimization)
        if isinstance(prepared_batch, _Sam3DBodyBatch):
            batch_input = _sam3d_body_batch_to_cuda(prepared_batch)
        else:
            batch_input = _build_sam3d_body_batch(
                estimator,
                bucket_items,
                clip_cam_int=clip_cam_int,
            )
        pose_output = _run_sam3d_body_model(
            estimator,
            batch_input,
            optimization=optimization,
            torch_module=torch,
        )
        if optimization["tier2_output_lite"]:
            return _direct_output_lite_records(
                pose_output["mhr"],
                bucket_items,
                batch=batch_input.batch,
                mask_scores=batch_input.mask_scores,
                recursive_to=recursive_to,
                np=np,
            )
        out = recursive_to(pose_output["mhr"], "cpu")
        out = recursive_to(out, "numpy")
        all_out: list[dict[str, Any]] = []
        for idx, item in enumerate(bucket_items):
            all_out.append(
                {
                    "request_id": item["request"]["request_id"],
                    "bbox": np.asarray(item["bbox"], dtype=np.float32),
                    "focal_length": out["focal_length"][idx],
                    "pred_keypoints_3d": out["pred_keypoints_3d"][idx],
                    "pred_keypoints_2d": out["pred_keypoints_2d"][idx],
                    "pred_vertices": out["pred_vertices"][idx],
                    "pred_cam_t": out["pred_cam_t"][idx],
                    "pred_pose_raw": out["pred_pose_raw"][idx],
                    "global_rot": out["global_rot"][idx],
                    "body_pose_params": out["body_pose"][idx],
                    "hand_pose_params": out["hand"][idx],
                    "scale_params": out["scale"][idx],
                    "shape_params": out["shape"][idx],
                    "expr_params": out["face"][idx],
                    "mask": (
                        None
                        if float(batch_input.mask_scores[idx]) <= 0.0
                        else batch_input.batch["mask"][0, idx].detach().cpu().numpy()
                    ),
                    "pred_joint_coords": out["pred_joint_coords"][idx],
                    "pred_global_rots": out["joint_global_rots"][idx],
                }
            )
        return all_out


def _direct_output_lite_records(
    mhr_output: Mapping[str, Any],
    bucket_items: Sequence[Mapping[str, Any]],
    *,
    batch: Mapping[str, Any],
    mask_scores: Sequence[float],
    recursive_to: Any,
    np: Any,
) -> list[dict[str, Any]]:
    all_out: list[dict[str, Any]] = []
    for idx, item in enumerate(bucket_items):
        target_representation = str(item.get("target_representation", "world_mesh"))
        include_dense = target_representation != TIER2_BODY_JOINTS_REPRESENTATION
        record: dict[str, Any] = {
            "request_id": item["request"]["request_id"],
            "bbox": np.asarray(item["bbox"], dtype=np.float32),
        }
        for output_name, source_name in (
            ("focal_length", "focal_length"),
            ("pred_keypoints_3d", "pred_keypoints_3d"),
            ("pred_keypoints_2d", "pred_keypoints_2d"),
            ("pred_vertices", "pred_vertices"),
            ("pred_cam_t", "pred_cam_t"),
            ("pred_pose_raw", "pred_pose_raw"),
            ("global_rot", "global_rot"),
            ("body_pose_params", "body_pose"),
            ("hand_pose_params", "hand"),
            ("scale_params", "scale"),
            ("shape_params", "shape"),
            ("expr_params", "face"),
            ("pred_joint_coords", "pred_joint_coords"),
            ("pred_global_rots", "joint_global_rots"),
        ):
            if not include_dense and output_name in TIER2_OUTPUT_LITE_OMIT_FIELDS:
                continue
            if source_name not in mhr_output:
                continue
            record[output_name] = _indexed_cpu_numpy(mhr_output[source_name], idx, recursive_to=recursive_to)
        record["mask"] = None if float(mask_scores[idx]) <= 0.0 else batch["mask"][0, idx].detach().cpu().numpy()
        all_out.append(record)
    return all_out


def _indexed_cpu_numpy(value: Any, index: int, *, recursive_to: Any) -> Any:
    indexed = value[index]
    indexed = recursive_to(indexed, "cpu")
    return recursive_to(indexed, "numpy")


def _output_record_for_representation(
    record: dict[str, Any],
    *,
    target_representation: str,
    tier2_output_lite: bool,
) -> dict[str, Any]:
    if not tier2_output_lite or target_representation != TIER2_BODY_JOINTS_REPRESENTATION:
        return record
    return {
        key: value
        for key, value in record.items()
        if key not in TIER2_OUTPUT_LITE_OMIT_FIELDS
    }


def _clear_cuda_cache_if_enabled(torch_module: Any, *, enabled: bool) -> None:
    if not enabled:
        return
    try:
        torch_module.cuda.empty_cache()
    except RuntimeError:
        pass


def _synchronize_cuda_if_available(torch_module: Any, *, enabled: bool) -> None:
    if enabled and hasattr(torch_module, "cuda") and torch_module.cuda.is_available():
        torch_module.cuda.synchronize()


def _synchronize_cuda_boundary() -> None:
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError:
        return
    _synchronize_cuda_if_available(torch, enabled=True)


def _timing_mode(optimization: Mapping[str, Any]) -> str:
    return "sync_per_bucket" if optimization.get("inner_bucket_sync", True) else "async_steady_state"


def _timing_notes(timing_mode: str) -> list[str]:
    if timing_mode == "sync_per_bucket":
        return ["per-bucket run_inference timing is synchronized before and after each bucket"]
    return [
        "inner per-bucket run_inference timings are async-inaccurate when inner_bucket_sync=false; "
        "use overall wall timing after the runner's outer synchronize/output-copy boundary"
    ]


def _output_metadata(
    optimization: Mapping[str, Any],
    *,
    runtime_environment: Mapping[str, Any],
    mhr_correctives: Mapping[str, Any],
    timing_events: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    timing_mode = _timing_mode(optimization)
    return {
        "timing_mode": timing_mode,
        "timing_notes": _timing_notes(timing_mode),
        "upstream_env": dict(runtime_environment.get("upstream_env", {})),
        "compile_environment": dict(runtime_environment.get("compile_environment", {})),
        "mhr_correctives": dict(mhr_correctives),
        "timings": timing_events if timing_events is not None else [],
    }


def _configure_runtime_environment(optimization: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "compile_environment": _configure_compile_environment(optimization),
        "upstream_env": _apply_upstream_env(optimization.get("upstream_env", {})),
    }


def _apply_upstream_env(raw_env: Mapping[str, Any], *, environ: Any = os.environ) -> dict[str, str]:
    parsed = _parse_upstream_env(raw_env)
    applied: dict[str, str] = {}
    for key, value in parsed.items():
        environ[key] = value
        applied[key] = value
    return applied


def _detect_mhr_correctives_active(estimator: Any) -> dict[str, Any]:
    model = getattr(estimator, "model", None)
    for obj, source in (
        (estimator, "estimator.apply_correctives"),
        (model, "model.apply_correctives"),
    ):
        if obj is not None and hasattr(obj, "apply_correctives"):
            return {
                "status": "detected",
                "active": bool(getattr(obj, "apply_correctives")),
                "source": source,
            }
    modules = getattr(model, "modules", None)
    if callable(modules):
        for module in modules():
            if hasattr(module, "apply_correctives"):
                return {
                    "status": "detected",
                    "active": bool(getattr(module, "apply_correctives")),
                    "source": "model.modules.apply_correctives",
                }
    return {"status": "unknown", "active": None, "source": "apply_correctives_not_found"}


def _configure_compile_environment(optimization: Mapping[str, Any], *, environ: Any = os.environ) -> dict[str, Any]:
    warmup_buckets = [int(value) for value in optimization.get("compile_warmup_buckets", [])]
    if optimization.get("torch_compile"):
        environ["USE_COMPILE"] = "1"
        # Upstream Fast-SAM warmup calls forward_step_merged, which compiles
        # _forward_decoders_combined. Production buckets use run_inference body
        # mode and forward_decoder_body, so the runner owns the shape warmup.
        environ["COMPILE_WARMUP_BATCH_SIZES"] = ""
        return {
            "use_compile": True,
            "upstream_estimator_compile_warmup": "disabled",
            "script_compile_warmup_buckets": warmup_buckets,
            "warmup_path": "run_inference_body_forward_decoder_body",
        }
    environ["USE_COMPILE"] = "0"
    environ["COMPILE_WARMUP_BATCH_SIZES"] = ""
    return {
        "use_compile": False,
        "upstream_estimator_compile_warmup": "disabled",
        "script_compile_warmup_buckets": warmup_buckets,
        "warmup_path": "skipped_torch_compile_disabled",
    }


def _warmup_static_clip_intrinsics(
    estimator: Any,
    *,
    clip_intrinsics: Mapping[str, Any] | None,
    optimization: Mapping[str, Any],
    timing: _TimingRecorder | None = None,
) -> dict[str, Any]:
    warmup_buckets = [int(value) for value in optimization.get("compile_warmup_buckets", [])]
    warmup_passes = _positive_int(
        optimization.get("compile_warmup_passes", 2),
        name="optimization.compile_warmup_passes",
    )
    if not optimization.get("torch_compile") or not warmup_buckets or clip_intrinsics is None:
        return {
            "status": "skipped",
            "reason": "torch_compile_or_static_intrinsics_disabled",
            "warmup_buckets": warmup_buckets,
            "warmup_passes_per_shape": warmup_passes,
            "production_callable_identity": _production_callable_identity(estimator),
        }
    try:
        import torch  # type: ignore[import-not-found]

        production_callable_identity_before = _production_callable_identity(estimator)
        image_size_hw = _warmup_image_size_hw(estimator, clip_intrinsics=clip_intrinsics)
        cam_int = _camera_intrinsics_tensor(clip_intrinsics["matrix"])
        warmed = []
        warmup_signatures = {}
        warmup_call_count_by_bucket = {}
        warmup_call_sequence = []
        for batch_size in warmup_buckets:
            real_batch_size = int(batch_size)
            bucket_start = time.monotonic()
            warmup_items = _synthetic_sam3d_warmup_bucket_items(
                real_batch_size,
                image_size_hw=image_size_hw,
            )
            warmup_call_count_by_bucket[str(real_batch_size)] = 0
            for pass_index in range(warmup_passes):
                with torch.inference_mode():
                    _prepare_estimator_for_bucket(estimator, torch, optimization=optimization)
                    batch_input = _build_sam3d_body_batch(
                        estimator,
                        warmup_items,
                        clip_cam_int=cam_int,
                    )
                    if pass_index == 0:
                        warmup_signatures[str(real_batch_size)] = _sam3d_batch_guard_signature(batch_input.batch, torch)
                    _run_sam3d_body_model(
                        estimator,
                        batch_input,
                        optimization=optimization,
                        torch_module=torch,
                    )
                if hasattr(torch, "cuda") and torch.cuda.is_available():
                    torch.cuda.synchronize()
                warmup_call_count_by_bucket[str(real_batch_size)] += 1
                warmup_call_sequence.append({"bucket_size": real_batch_size, "pass_index": pass_index + 1})
            if timing is not None:
                timing.record(
                    "compile_warmup_bucket",
                    bucket_start,
                    time.monotonic(),
                    bucket_size=real_batch_size,
                    pass_count=warmup_passes,
                )
            warmed.append(real_batch_size)
        production_callable_identity_after = _remember_warmed_production_callable(estimator)
        return {
            "status": "ran",
            "warmup_buckets": warmed,
            "warmup_passes_per_shape": warmup_passes,
            "warmup_call_count_by_bucket": warmup_call_count_by_bucket,
            "warmup_call_sequence": warmup_call_sequence,
            "execution_shapes": [[1, value] for value in warmed],
            "warmup_shape_contract": "same_batch_sizes_as_real_bucketed_body_execution",
            "warmup_batch_builder": "shared_with_real_bucketed_body_execution",
            "model_entrypoint": "run_inference",
            "production_callable_identity_before": production_callable_identity_before,
            "production_callable_identity_after": production_callable_identity_after,
            "production_callable_identity_contract": "warmup_and_bucket_execution_must_match",
            "grad_enabled": False,
            "inference_mode": True,
            "batch_guard_signatures": warmup_signatures,
            "static_clip_intrinsics": True,
            "cam_int_source": clip_intrinsics.get("source", ""),
        }
    except Exception as exc:  # pragma: no cover - exercised on A100 validator, not local CPU tests.
        return {
            "status": "failed",
            "warmup_buckets": warmup_buckets,
            "warmup_passes_per_shape": warmup_passes,
            "static_clip_intrinsics": True,
            "production_callable_identity": _production_callable_identity(estimator),
            "error": str(exc),
        }


def _warmup_image_size_hw(estimator: Any, *, clip_intrinsics: Mapping[str, Any]) -> tuple[int, int]:
    matrix = clip_intrinsics.get("matrix") or []
    try:
        width = max(2, int(round(float(matrix[0][2]) * 2.0)))
        height = max(2, int(round(float(matrix[1][2]) * 2.0)))
        if width > 2 and height > 2:
            return height, width
    except (IndexError, TypeError, ValueError):
        pass
    image_size = getattr(getattr(getattr(estimator, "cfg", None), "MODEL", None), "IMAGE_SIZE", (512, 512))
    return int(image_size[0]), int(image_size[1])


def _parse_requests(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, Mapping):
        raise ValueError("payload must be an object")
    raw_requests = payload.get("requests")
    if not isinstance(raw_requests, list):
        raise ValueError("payload.requests must be a list")
    requests = []
    for index, raw in enumerate(raw_requests):
        if not isinstance(raw, Mapping):
            raise ValueError(f"requests/{index} must be an object")
        image = raw.get("image")
        if not isinstance(image, str) or not image:
            raise ValueError(f"requests/{index}/image must be a non-empty string")
        bboxes = raw.get("bboxes")
        if not isinstance(bboxes, list):
            raise ValueError(f"requests/{index}/bboxes must be a list")
        requests.append(
            {
                "request_id": str(raw.get("request_id", index)),
                "image": Path(image),
                "bboxes": [parse_bbox_arg(",".join(str(value) for value in bbox)) for bbox in bboxes],
                "mask_paths": _parse_mask_paths(raw.get("mask_paths", []), expected_count=len(bboxes), index=index),
                "camera_intrinsics": _parse_camera_intrinsics(raw.get("camera_intrinsics")),
                "sam3d_body_input_size_px": (
                    normalize_body_input_size(raw.get("sam3d_body_input_size_px"))
                    if raw.get("sam3d_body_input_size_px") is not None
                    else None
                ),
                "target_representation": str(raw.get("target_representation", "world_mesh")),
            }
        )
    return requests


def _parse_mask_paths(raw_paths: Any, *, expected_count: int, index: int) -> list[Path]:
    if raw_paths in (None, ""):
        return []
    if not isinstance(raw_paths, list):
        raise ValueError(f"requests/{index}/mask_paths must be a list")
    paths = [Path(path) for path in raw_paths if path]
    if paths and len(paths) != expected_count:
        raise ValueError(f"requests/{index}/mask_paths must match bboxes length")
    return paths


def _parse_camera_intrinsics(raw_value: Any) -> list[list[float]] | None:
    if raw_value in (None, ""):
        return None
    if not isinstance(raw_value, list) or len(raw_value) != 3:
        raise ValueError("camera_intrinsics must be a 3x3 array")
    rows = []
    for row in raw_value:
        if not isinstance(row, list) or len(row) != 3:
            raise ValueError("camera_intrinsics must be a 3x3 array")
        rows.append([float(item) for item in row])
    return rows


def _camera_intrinsics_tensor(camera_intrinsics: list[list[float]] | None) -> Any | None:
    if camera_intrinsics is None:
        return None
    import torch  # type: ignore[import-not-found]

    return torch.tensor([camera_intrinsics], dtype=torch.float32)


if __name__ == "__main__":
    raise SystemExit(main())

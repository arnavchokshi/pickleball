#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
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


@dataclass(frozen=True)
class _Sam3DBodyBatch:
    batch: Mapping[str, Any]
    rgb_images: list[Any]
    mask_scores: list[float]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run FastSAM-3D-Body on a batch of frame/bbox requests. "
            "Request JSON may include mask_paths and static camera_intrinsics for Phase C."
        )
    )
    parser.add_argument("--requests", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--fast-sam-repo", required=True, type=Path)
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--detector-model", default="")
    parser.add_argument("--detector-name", default=None)
    parser.add_argument("--fov-name", default="")
    parser.add_argument("--body-input-size", type=int, default=None, help="Optional SAM-3D body crop size: 384, 448, or 512.")
    args = parser.parse_args(argv)

    try:
        payload = json.loads(args.requests.read_text(encoding="utf-8"))
        batch_payload = _parse_batch_payload(payload)
        requests = batch_payload["requests"]
        optimization = dict(batch_payload["optimization"])
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

    try:
        setup_sam_3d_body = _load_setup_sam_3d_body(args.fast_sam_repo)
        estimator = _setup_estimator(
            setup_sam_3d_body,
            checkpoint_dir=args.checkpoint_dir.resolve(),
            detector_name=_detector_name(args.detector_name, [bbox for request in requests for bbox in request["bboxes"]]),
            detector_model=args.detector_model,
            fov_name=args.fov_name,
        )
        compile_warmup = _warmup_static_clip_intrinsics(
            estimator,
            clip_intrinsics=batch_payload["clip_intrinsics"],
            optimization=optimization,
        )
        mhr_correctives = _detect_mhr_correctives_active(estimator)
        faces = _json_safe(getattr(estimator, "faces", None))
        frames = []
        clip_cam_int = _camera_intrinsics_tensor(
            batch_payload["clip_intrinsics"]["matrix"] if batch_payload["clip_intrinsics"] is not None else None
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

    out_payload = {
        "schema_version": 1,
        "artifact_type": "racketsport_sam3dbody_batch",
        "request_count": len(requests),
        "clip_intrinsics": batch_payload["clip_intrinsics"],
        "optimization": optimization,
        "metadata": _output_metadata(
            optimization,
            runtime_environment=runtime_environment,
            mhr_correctives=mhr_correctives,
        ),
        "bucket_plan": batch_payload["bucket_plan"],
        "compile_warmup": compile_warmup,
        "batch_execution": batch_execution,
        "frames": frames,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out_payload, separators=(",", ":")) + "\n", encoding="utf-8")
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
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    optimization = _parse_optimization(optimization or {})
    request_by_id = {str(request["request_id"]): request for request in requests}
    frames_by_request: dict[str, dict[str, Any]] = {}
    execution_buckets: list[dict[str, Any]] = []
    if not optimization["inner_bucket_sync"]:
        _synchronize_cuda_boundary()
    for bucket_index, bucket in enumerate(bucket_plan):
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
            continue
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
        raw_records = _process_sam3d_body_bucket(
            estimator,
            bucket_items,
            clip_cam_int=clip_cam_int,
            optimization=optimization,
        )
        records = [_json_safe(record) for record in _extract_person_records(raw_records)]
        if len(records) != len(bucket_items):
            raise RuntimeError(
                f"SAM3D bucket {bucket_index} returned {len(records)} records for bucket size {len(bucket_items)}"
            )
        for item, record in zip(bucket_items, records, strict=True):
            if item["is_padding"]:
                continue
            target_representation = str(item.get("target_representation", "world_mesh"))
            record = _output_record_for_representation(
                record,
                target_representation=target_representation,
                tier2_output_lite=optimization["tier2_output_lite"],
            )
            if (
                faces
                and target_representation != TIER2_BODY_JOINTS_REPRESENTATION
                and isinstance(record, dict)
                and "mesh_faces" not in record
                and "faces" not in record
            ):
                record["mesh_faces"] = faces
            request = item["request"]
            request_id = str(request["request_id"])
            frames_by_request[request_id] = {
                "request_id": request_id,
                "image_path": str(Path(request["image"]).resolve()),
                "requested_bboxes": request["bboxes"],
                "requested_masks": [str(path) for path in request.get("mask_paths", [])],
                "camera_intrinsics": (
                    clip_intrinsics["matrix"]
                    if clip_intrinsics is not None
                    else request.get("camera_intrinsics")
                ),
                "camera_intrinsics_source": (
                    clip_intrinsics.get("source", "")
                    if clip_intrinsics is not None
                    else ("request" if request.get("camera_intrinsics") is not None else "")
                ),
                "sam3d_body_input_size_px": body_input_size or request.get("sam3d_body_input_size_px"),
                "records": [record],
                "summary": {"record_count": 1},
                "target_representation": target_representation,
            }
        execution_buckets.append(
            {
                "bucket_index": bucket_index,
                "bucket_size": bucket_size,
                "real_request_count": len(real_items),
                "padding_count": bucket_size - len(real_items),
                "padded_request_count": len(bucket_items),
                "padded_crop_ratio": (bucket_size - len(real_items)) / bucket_size if bucket_size else 0.0,
                "request_ids": request_ids,
                "execution_shape": [1, bucket_size],
                "warmup_shape_contract": "same_batch_size_as_real_execution",
            }
        )
    frames = []
    for request in requests:
        request_id = str(request["request_id"])
        if request_id not in frames_by_request:
            raise RuntimeError(f"SAM3D bucket execution produced no output for request {request_id!r}")
        frames.append(frames_by_request[request_id])
    if not optimization["inner_bucket_sync"]:
        _synchronize_cuda_boundary()
    timing_mode = _timing_mode(optimization)
    return frames, {
        "mode": "real_bucketed_body_batch",
        "timing_mode": timing_mode,
        "timing_notes": _timing_notes(timing_mode),
        "buckets": execution_buckets,
        "request_count": len(requests),
        "output_count": len(frames),
    }


def _process_sam3d_body_bucket(
    estimator: Any,
    bucket_items: Sequence[Mapping[str, Any]],
    *,
    clip_cam_int: Any | None,
    optimization: Mapping[str, Any],
) -> list[Any]:
    test_hook = getattr(estimator, "process_body_bucket", None)
    if callable(test_hook):
        return list(test_hook(list(bucket_items), cam_int=clip_cam_int))
    return _process_sam3d_body_bucket_direct(
        estimator,
        bucket_items,
        clip_cam_int=clip_cam_int,
        optimization=optimization,
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
        batch=recursive_to(batch, "cuda"),
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


def _process_sam3d_body_bucket_direct(
    estimator: Any,
    bucket_items: Sequence[Mapping[str, Any]],
    *,
    clip_cam_int: Any | None,
    optimization: Mapping[str, Any],
) -> list[dict[str, Any]]:
    import numpy as np  # type: ignore[import-not-found]
    import torch  # type: ignore[import-not-found]
    from sam_3d_body.utils import recursive_to  # type: ignore[import-not-found]

    with torch.inference_mode():
        _prepare_estimator_for_bucket(estimator, torch, optimization=optimization)
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
) -> dict[str, Any]:
    timing_mode = _timing_mode(optimization)
    return {
        "timing_mode": timing_mode,
        "timing_notes": _timing_notes(timing_mode),
        "upstream_env": dict(runtime_environment.get("upstream_env", {})),
        "mhr_correctives": dict(mhr_correctives),
    }


def _configure_runtime_environment(optimization: Mapping[str, Any]) -> dict[str, Any]:
    _configure_compile_environment(optimization)
    return {"upstream_env": _apply_upstream_env(optimization.get("upstream_env", {}))}


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


def _configure_compile_environment(optimization: Mapping[str, Any]) -> None:
    if optimization.get("torch_compile"):
        os.environ["USE_COMPILE"] = "1"
        warmup_buckets = [int(value) for value in optimization.get("compile_warmup_buckets", [])]
        if warmup_buckets:
            os.environ["COMPILE_WARMUP_BATCH_SIZES"] = ",".join(str(value) for value in warmup_buckets)
    else:
        os.environ["USE_COMPILE"] = "0"


def _warmup_static_clip_intrinsics(
    estimator: Any,
    *,
    clip_intrinsics: Mapping[str, Any] | None,
    optimization: Mapping[str, Any],
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
        }
    try:
        import torch  # type: ignore[import-not-found]

        image_size_hw = _warmup_image_size_hw(estimator, clip_intrinsics=clip_intrinsics)
        cam_int = _camera_intrinsics_tensor(clip_intrinsics["matrix"])
        warmed = []
        warmup_signatures = {}
        warmup_call_count_by_bucket = {}
        warmup_call_sequence = []
        for batch_size in warmup_buckets:
            real_batch_size = int(batch_size)
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
            warmed.append(real_batch_size)
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

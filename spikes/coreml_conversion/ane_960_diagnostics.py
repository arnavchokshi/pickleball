#!/usr/bin/env python3
"""Diagnose YOLO26n 960 Core ML ANE compile behavior.

The previous iPhone gate showed that the converted 960 detector package loads
but fails ANE runtime compilation on-device, then falls back to GPU/CPU. This
script keeps the reproducible investigation in one lane-local place:

* summarize the 640 vs 960 MIL graph shapes;
* export conversion variants that change one suspect at a time;
* compile and run each candidate with macOS `CPU_AND_NE` / `ALL` as a proxy;
* write a concise next-device-benchmark recommendation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
LIVE_BUDGET_MS = 1000.0 / 30.0
DEFAULT_EXISTING_COREML_PREP = REPO_ROOT / "runs/coreml_prep_20260702T023932Z"

DEVICE_REFERENCE_MS = {
    "yolo26n_640_all_mean": 3.185,
    "yolo26n_640_all_p90": 3.726,
    "yolo26n_960_all_fallback_mean": 14.533,
    "yolo26n_960_all_fallback_p90": 14.800,
    "ball_student_all_mean": 1.407,
    "ball_student_all_p90": 1.489,
}


def yolo_grid_points(imgsz: int, strides: tuple[int, ...] = (8, 16, 32)) -> int:
    """Return the number of YOLO detection points for square `imgsz`."""

    if imgsz <= 0:
        raise ValueError("imgsz must be positive")
    if imgsz % max(strides) != 0:
        raise ValueError(f"imgsz must be divisible by {max(strides)} for this YOLO export")
    return sum((imgsz // stride) ** 2 for stride in strides)


def postprocess_tensor_shapes(imgsz: int, classes: int = 80, topk: int = 300) -> dict[str, list[int]]:
    """Expected YOLO26n export tail shapes for the end-to-end detector."""

    grid = yolo_grid_points(imgsz)
    return {
        "raw_predictions": [1, classes + 4, grid],
        "transposed_predictions": [1, grid, classes + 4],
        "first_topk_input": [1, grid],
        "first_topk_scores": [1, topk],
        "class_scores_after_first_topk": [1, topk, classes],
        "second_topk_input": [1, topk * classes],
        "second_topk_scores": [1, topk],
        "boxes_after_gather": [1, topk, 4],
        "final_output": [1, topk, 6],
    }


def project_partial_loop_ms(detector_ms: float, ball_ms: float, detector_runs: int = 1) -> dict[str, float]:
    """Project player-detector + one ball-student partial loop latency."""

    if detector_runs <= 0:
        raise ValueError("detector_runs must be positive")
    mean_ms = detector_ms * detector_runs + ball_ms
    return {
        "mean_ms": round(mean_ms, 3),
        "fps": round(1000.0 / mean_ms, 2),
    }


def torch_trace_kwargs() -> dict[str, bool]:
    """Trace options needed for Ultralytics detect-head cache mutation."""

    return {"strict": False, "check_trace": False}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_size_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def run_command(args: list[str]) -> dict[str, Any]:
    started = time.perf_counter()
    proc = subprocess.run(args, capture_output=True, text=True, check=False)
    return {
        "args": args,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "duration_seconds": time.perf_counter() - started,
    }


def import_version(name: str) -> dict[str, str]:
    try:
        module = __import__(name)
        return {
            "version": str(getattr(module, "__version__", "unknown")),
            "file": str(getattr(module, "__file__", "")),
        }
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def collect_environment() -> dict[str, Any]:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "packages": {
            "coremltools": import_version("coremltools"),
            "torch": import_version("torch"),
            "ultralytics": import_version("ultralytics"),
            "numpy": import_version("numpy"),
        },
        "note": "macOS CPU_AND_NE/ALL prediction is a compile-and-run proxy only; final truth requires the iPhone gate.",
    }


def select_yolo_checkpoint() -> Path:
    preferred = REPO_ROOT / "models/checkpoints/yolo26n.pt"
    fallback = REPO_ROOT / "models/checkpoints/yolo26m.pt"
    if preferred.exists():
        return preferred
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"missing both {preferred} and {fallback}")


def _coreml_imports() -> tuple[Any, Any, Any]:
    os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mpl")
    os.environ.setdefault("COREMLTOOLS_HOME", "/private/tmp/coremltools")
    os.environ.setdefault("TMPDIR", "/private/tmp")
    os.environ.setdefault("TEMP", "/private/tmp")
    os.environ.setdefault("TMP", "/private/tmp")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

    import numpy as np
    import torch
    import coremltools as ct

    return ct, np, torch


def _torch_yolo_import() -> Any:
    from ultralytics import YOLO

    return YOLO


def _copy_package(src: Path, dst: Path) -> None:
    if dst.exists():
        if dst.is_dir():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    if src.is_dir():
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)


def copy_existing_variant(name: str, source: Path, work_dir: Path) -> dict[str, Any]:
    started = time.perf_counter()
    dst = work_dir / source.name
    try:
        _copy_package(source, dst)
        return {
            "name": name,
            "status": "ok",
            "conversion_kind": "copied_existing_package",
            "source_path": source,
            "model_path": dst,
            "duration_seconds": time.perf_counter() - started,
            "package_size_bytes": package_size_bytes(dst),
            "sha256_manifest_or_file": sha256_file(dst / "Manifest.json") if dst.is_dir() else sha256_file(dst),
        }
    except Exception as exc:
        return {
            "name": name,
            "status": "error",
            "conversion_kind": "copied_existing_package",
            "source_path": source,
            "duration_seconds": time.perf_counter() - started,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }


def export_ultralytics_variant(name: str, checkpoint: Path, imgsz: int, work_dir: Path, *, half: bool) -> dict[str, Any]:
    YOLO = _torch_yolo_import()
    local_checkpoint = work_dir / checkpoint.name
    shutil.copy2(checkpoint, local_checkpoint)
    old_cwd = Path.cwd()
    started = time.perf_counter()
    try:
        os.chdir(work_dir)
        model = YOLO(local_checkpoint.name)
        exported = model.export(format="coreml", imgsz=imgsz, half=half, nms=False, batch=1)
        exported_path = Path(exported)
        if not exported_path.is_absolute():
            exported_path = work_dir / exported_path
        return {
            "name": name,
            "status": "ok",
            "conversion_kind": "ultralytics_export",
            "imgsz": imgsz,
            "half": half,
            "model_path": exported_path,
            "duration_seconds": time.perf_counter() - started,
            "package_size_bytes": package_size_bytes(exported_path),
        }
    except Exception as exc:
        return {
            "name": name,
            "status": "error",
            "conversion_kind": "ultralytics_export",
            "imgsz": imgsz,
            "half": half,
            "duration_seconds": time.perf_counter() - started,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
    finally:
        os.chdir(old_cwd)


def _target(ct: Any, name: str | None) -> Any | None:
    if not name:
        return None
    return getattr(ct.target, name)


def _precision(ct: Any, name: str | None) -> Any | None:
    if not name:
        return None
    return getattr(ct.precision, name)


def _input_spec(ct: Any, np: Any, shape: tuple[int, int, int, int], kind: str, enumerated_imgszs: list[int] | None) -> list[Any]:
    if kind == "image":
        return [ct.ImageType(name="image", shape=shape, scale=1 / 255.0, bias=[0.0, 0.0, 0.0])]
    if kind == "tensor":
        return [ct.TensorType(name="image", shape=shape, dtype=np.float32)]
    if kind == "tensor_enumerated":
        if not enumerated_imgszs:
            raise ValueError("tensor_enumerated requires enumerated_imgszs")
        enum_shapes = [(1, 3, size, size) for size in enumerated_imgszs]
        return [ct.TensorType(name="image", shape=ct.EnumeratedShapes(shapes=enum_shapes, default=shape), dtype=np.float32)]
    raise ValueError(f"unknown input kind: {kind}")


def export_direct_variant(
    name: str,
    checkpoint: Path,
    imgsz: int,
    work_dir: Path,
    *,
    output_mode: str,
    input_kind: str,
    precision_name: str | None,
    target_name: str | None,
    enumerated_imgszs: list[int] | None = None,
) -> dict[str, Any]:
    ct, np, torch = _coreml_imports()
    YOLO = _torch_yolo_import()
    started = time.perf_counter()
    output_path = work_dir / f"{name}.mlpackage"

    class FinalOutputWrapper(torch.nn.Module):
        def __init__(self, model: torch.nn.Module):
            super().__init__()
            self.model = model

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            y = self.model(x)
            return y[0] if isinstance(y, (tuple, list)) else y

    class RawSplitWrapper(torch.nn.Module):
        def __init__(self, model: torch.nn.Module):
            super().__init__()
            self.model = model

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            y = self.model(x)
            aux = y[1]["one2one"]
            return aux["boxes"], aux["scores"]

    try:
        base = YOLO(str(checkpoint)).model.eval()
        wrapper: torch.nn.Module
        outputs: list[Any]
        if output_mode == "final":
            wrapper = FinalOutputWrapper(base).eval()
            outputs = [ct.TensorType(name="detections")]
        elif output_mode == "raw_split":
            wrapper = RawSplitWrapper(base).eval()
            outputs = [ct.TensorType(name="boxes"), ct.TensorType(name="scores")]
        else:
            raise ValueError(f"unknown output mode: {output_mode}")

        example = torch.zeros(1, 3, imgsz, imgsz)
        with torch.no_grad():
            traced = torch.jit.trace(wrapper, example, **torch_trace_kwargs())

        convert_kwargs: dict[str, Any] = {
            "convert_to": "mlprogram",
            "inputs": _input_spec(ct, np, tuple(example.shape), input_kind, enumerated_imgszs),
            "outputs": outputs,
        }
        precision = _precision(ct, precision_name)
        if precision is not None:
            convert_kwargs["compute_precision"] = precision
        minimum_target = _target(ct, target_name)
        if minimum_target is not None:
            convert_kwargs["minimum_deployment_target"] = minimum_target

        mlmodel = ct.convert(traced, **convert_kwargs)
        if output_path.exists():
            shutil.rmtree(output_path)
        mlmodel.save(str(output_path))
        return {
            "name": name,
            "status": "ok",
            "conversion_kind": "direct_coremltools",
            "imgsz": imgsz,
            "output_mode": output_mode,
            "input_kind": input_kind,
            "precision": precision_name,
            "minimum_deployment_target": target_name,
            "enumerated_imgszs": enumerated_imgszs,
            "model_path": output_path,
            "duration_seconds": time.perf_counter() - started,
            "package_size_bytes": package_size_bytes(output_path),
        }
    except Exception as exc:
        return {
            "name": name,
            "status": "error",
            "conversion_kind": "direct_coremltools",
            "imgsz": imgsz,
            "output_mode": output_mode,
            "input_kind": input_kind,
            "precision": precision_name,
            "minimum_deployment_target": target_name,
            "enumerated_imgszs": enumerated_imgszs,
            "duration_seconds": time.perf_counter() - started,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }


def summarize_mlprogram_ops(model_path: Path) -> dict[str, Any]:
    ct, _np, _torch = _coreml_imports()
    model = ct.models.MLModel(str(model_path), compute_units=ct.ComputeUnit.CPU_ONLY, skip_model_load=True)
    spec = model.get_spec()
    summary: dict[str, Any] = {
        "spec_type": spec.WhichOneof("Type"),
        "specification_version": spec.specificationVersion,
        "inputs": [],
        "outputs": [],
        "operation_counts": {},
        "topk_ops": [],
        "selected_tensor_shapes": {},
    }
    for feature in spec.description.input:
        ftype = feature.type.WhichOneof("Type")
        item: dict[str, Any] = {"name": feature.name, "type": ftype}
        if ftype == "imageType":
            item["shape"] = [int(feature.type.imageType.height), int(feature.type.imageType.width)]
        elif ftype == "multiArrayType":
            item["shape"] = [int(dim) for dim in feature.type.multiArrayType.shape]
        summary["inputs"].append(item)
    for feature in spec.description.output:
        ftype = feature.type.WhichOneof("Type")
        item = {"name": feature.name, "type": ftype}
        if ftype == "multiArrayType":
            item["shape"] = [int(dim) for dim in feature.type.multiArrayType.shape]
        summary["outputs"].append(item)

    if spec.WhichOneof("Type") != "mlProgram":
        return summary

    interesting = {
        "y_cast_fp16",
        "var_1409_cast_fp16",
        "reduce_max_0_cast_fp16",
        "scores_3_cast_fp16",
        "var_1425_cast_fp16",
        "scores_5_cast_fp16_0",
        "boxes_cast_fp16",
        "var_1441_cast_fp16",
        "boxes",
        "scores",
        "detections",
    }

    for function in spec.mlProgram.functions.values():
        for block in function.block_specializations.values():
            for index, op in enumerate(block.operations):
                summary["operation_counts"][op.type] = summary["operation_counts"].get(op.type, 0) + 1
                if op.type == "topk":
                    summary["topk_ops"].append(
                        {
                            "index": index,
                            "inputs": {
                                key: [arg.name for arg in value.arguments]
                                for key, value in op.inputs.items()
                            },
                            "outputs": [_tensor_summary(output) for output in op.outputs],
                        }
                    )
                for output in op.outputs:
                    if output.name in interesting:
                        summary["selected_tensor_shapes"][output.name] = _tensor_summary(output)
    return summary


def _tensor_summary(output: Any) -> dict[str, Any]:
    tensor_type = output.type.tensorType
    dims: list[Any] = []
    for dim in tensor_type.dimensions:
        kind = dim.WhichOneof("dimension")
        if kind == "constant":
            dims.append(int(dim.constant.size))
        else:
            dims.append(kind)
    return {"name": output.name, "data_type": str(tensor_type.dataType), "shape": dims}


def _zero_inputs_for_model(model: Any) -> dict[str, Any]:
    ct, np, _torch = _coreml_imports()
    inputs: dict[str, Any] = {}
    for feature in model.get_spec().description.input:
        ftype = feature.type.WhichOneof("Type")
        if ftype == "imageType":
            from PIL import Image

            width = int(feature.type.imageType.width)
            height = int(feature.type.imageType.height)
            inputs[feature.name] = Image.fromarray(np.zeros((height, width, 3), dtype=np.uint8), mode="RGB")
        elif ftype == "multiArrayType":
            shape = [int(dim) for dim in feature.type.multiArrayType.shape]
            if not shape and feature.type.multiArrayType.WhichOneof("ShapeFlexibility") == "enumeratedShapes":
                first = feature.type.multiArrayType.enumeratedShapes.shapes[0]
                shape = [int(dim) for dim in first.shape]
            inputs[feature.name] = np.zeros(shape, dtype=np.float32)
        else:
            raise ValueError(f"unsupported input type {feature.name}: {ftype}")
    return inputs


def compile_and_probe(model_path: Path, work_dir: Path, iterations: int) -> dict[str, Any]:
    ct, _np, _torch = _coreml_imports()
    compiled_dir = work_dir / f"compiled_{model_path.stem}"
    if compiled_dir.exists():
        shutil.rmtree(compiled_dir)
    compiled_dir.mkdir(parents=True, exist_ok=True)
    compile_result = run_command(["xcrun", "coremlc", "compile", str(model_path), str(compiled_dir)])
    compiled = sorted(compiled_dir.glob("*.mlmodelc"))
    result: dict[str, Any] = {
        "model_path": model_path,
        "coremlc": compile_result,
        "compiled_model_path": compiled[0] if compiled else None,
        "predict": [],
    }
    if compile_result["returncode"] != 0 or not compiled:
        result["status"] = "coremlc_error"
        return result

    spec_model = ct.models.MLModel(str(model_path), compute_units=ct.ComputeUnit.CPU_ONLY, skip_model_load=True)
    inputs = _zero_inputs_for_model(spec_model)
    for compute_unit_name in ("CPU_AND_NE", "ALL", "CPU_ONLY"):
        row: dict[str, Any] = {"compute_unit": compute_unit_name}
        try:
            compute_unit = getattr(ct.ComputeUnit, compute_unit_name)
            model = ct.models.CompiledMLModel(str(compiled[0]), compute_units=compute_unit)
            _ = model.predict(inputs)
            samples = []
            for _index in range(iterations):
                started = time.perf_counter()
                _ = model.predict(inputs)
                samples.append((time.perf_counter() - started) * 1000.0)
            samples_sorted = sorted(samples)
            row.update(
                {
                    "status": "ok",
                    "samples_ms": samples,
                    "median_ms": samples_sorted[len(samples_sorted) // 2],
                    "min_ms": min(samples),
                    "max_ms": max(samples),
                }
            )
        except Exception as exc:
            row.update(
                {
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
            )
        result["predict"].append(row)
    result["status"] = "ok" if any(row.get("status") == "ok" for row in result["predict"]) else "predict_error"
    return result


def default_variants(include_sweep: bool) -> list[dict[str, Any]]:
    variants: list[dict[str, Any]] = [
        {
            "name": "baseline_960_ultralytics_fp16_existing",
            "kind": "copy_existing",
            "source": DEFAULT_EXISTING_COREML_PREP / "yolo_player_detector/imgsz_960/yolo26n.mlpackage",
        },
        {
            "name": "baseline_640_ultralytics_fp16_existing",
            "kind": "copy_existing",
            "source": DEFAULT_EXISTING_COREML_PREP / "yolo_player_detector/imgsz_640/yolo26n.mlpackage",
        },
        {
            "name": "direct_final_960_fp32_image_iOS17",
            "kind": "direct",
            "imgsz": 960,
            "output_mode": "final",
            "input_kind": "image",
            "precision": "FLOAT32",
            "target": "iOS17",
        },
        {
            "name": "direct_final_960_fp16_tensor_enumerated_640_960",
            "kind": "direct",
            "imgsz": 960,
            "output_mode": "final",
            "input_kind": "tensor_enumerated",
            "precision": "FLOAT16",
            "target": None,
            "enumerated_imgszs": [640, 960],
        },
        {
            "name": "raw_split_960_fp16_image",
            "kind": "direct",
            "imgsz": 960,
            "output_mode": "raw_split",
            "input_kind": "image",
            "precision": "FLOAT16",
            "target": None,
        },
        {
            "name": "raw_split_960_fp16_tensor_enumerated_640_960",
            "kind": "direct",
            "imgsz": 960,
            "output_mode": "raw_split",
            "input_kind": "tensor_enumerated",
            "precision": "FLOAT16",
            "target": None,
            "enumerated_imgszs": [640, 960],
        },
    ]
    if include_sweep:
        variants.extend(
            [
                {"name": "ultralytics_896_fp16_image", "kind": "ultralytics", "imgsz": 896, "half": True},
                {"name": "ultralytics_832_fp16_image", "kind": "ultralytics", "imgsz": 832, "half": True},
                {"name": "ultralytics_800_fp16_image", "kind": "ultralytics", "imgsz": 800, "half": True},
            ]
        )
    return variants


def fallback_projection() -> dict[str, Any]:
    mean = DEVICE_REFERENCE_MS
    gpu_960 = project_partial_loop_ms(mean["yolo26n_960_all_fallback_mean"], mean["ball_student_all_mean"])
    tiled_640_4 = project_partial_loop_ms(mean["yolo26n_640_all_mean"], mean["ball_student_all_mean"], detector_runs=4)
    tiled_640_2 = project_partial_loop_ms(mean["yolo26n_640_all_mean"], mean["ball_student_all_mean"], detector_runs=2)
    return {
        "source": "runs/ios_device_gate_20260702T025809Z/LATENCY_TABLE_DEVICE.md",
        "scope": "player detector + one ball student only; no camera/tracking/render/thermal.",
        "gpu_cpu_fallback_960_mean": gpu_960,
        "two_tile_640_ane_mean": tiled_640_2,
        "four_tile_640_ane_mean": tiled_640_4,
        "budget_ms": LIVE_BUDGET_MS,
    }


def write_markdown_report(run_dir: Path, results: list[dict[str, Any]], graph_summaries: dict[str, Any]) -> None:
    passing_raw = [
        item
        for item in results
        if item.get("status") == "ok"
        and item.get("conversion", {}).get("output_mode") == "raw_split"
        and item.get("probe", {}).get("status") == "ok"
        and not graph_summaries.get(item["name"], {}).get("topk_ops")
    ]
    projection = fallback_projection()
    lines = [
        "# COREML-960 ANE Diagnostic Report",
        "",
        "## Root-Cause Evidence",
        "",
        "- The iPhone failure is runtime ANE compilation, not package creation: `coremlc compile` succeeded for the existing 960 package, while the device logged `ANECCompile() FAILED (11)` and fell back to GPU/CPU.",
        "- The existing 640 and 960 Core ML packages have the same op families, including two detector-tail `topk` ops.",
        f"- The first `topk` input grows from `{postprocess_tensor_shapes(640)['first_topk_input']}` at 640 to `{postprocess_tensor_shapes(960)['first_topk_input']}` at 960. The second `topk` remains `{postprocess_tensor_shapes(960)['second_topk_input']}` because it operates after top-300 candidate pruning.",
        "- Working hypothesis: the iPhone ANE compiler accepts this postprocess at 8,400 grid points but rejects the 18,900-point first `topk`/gather path at 960.",
        "",
        "## Candidate Results",
        "",
        "| Variant | Convert | topk ops | Mac CPU_AND_NE probe | Mac ALL probe | Notes |",
        "|---|---|---:|---|---|---|",
    ]
    for item in results:
        name = item["name"]
        conversion = item.get("conversion", {})
        probe = item.get("probe", {})
        summary = graph_summaries.get(name, {})
        topk_count = len(summary.get("topk_ops", []))
        cpu_ne = next((row for row in probe.get("predict", []) if row.get("compute_unit") == "CPU_AND_NE"), {})
        all_units = next((row for row in probe.get("predict", []) if row.get("compute_unit") == "ALL"), {})
        notes = []
        if conversion.get("output_mode") == "raw_split":
            notes.append("split: ANE detector body, Swift decode/top-k required")
        if conversion.get("input_kind") == "tensor_enumerated":
            notes.append("enumerated TensorType input, not drop-in image input")
        if conversion.get("precision") == "FLOAT32":
            notes.append("FP32, likely poor ANE target")
        if conversion.get("imgsz"):
            notes.append(f"grid={yolo_grid_points(int(conversion['imgsz']))}")
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    conversion.get("status", "n/a"),
                    str(topk_count),
                    _probe_cell(cpu_ne),
                    _probe_cell(all_units),
                    "; ".join(notes),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Fallback Projection",
            "",
            f"- Existing 960 GPU/CPU fallback + ball student: {projection['gpu_cpu_fallback_960_mean']['mean_ms']} ms mean, {projection['gpu_cpu_fallback_960_mean']['fps']} fps.",
            f"- 2x 640 ANE tiles + ball student: {projection['two_tile_640_ane_mean']['mean_ms']} ms mean, {projection['two_tile_640_ane_mean']['fps']} fps.",
            f"- 4x 640 ANE tiles + ball student: {projection['four_tile_640_ane_mean']['mean_ms']} ms mean, {projection['four_tile_640_ane_mean']['fps']} fps.",
            "- These are compute-only projections from the iPhone burst test; they exclude camera conversion, tracking, rendering, and thermal soak.",
            "",
            "## Recommendation",
            "",
        ]
    )
    if passing_raw:
        lines.extend(
            [
                f"Benchmark `{passing_raw[0]['name']}` on-device next. It removes the suspected 960 top-k tail from the ANE graph; Swift must consume boxes/scores and run class filtering plus top-k outside the model.",
                "",
                "Keep the existing 960 package as the quality fallback for the next phone run, but label it GPU/CPU fallback unless a new device log proves ANE compile succeeds.",
            ]
        )
    else:
        lines.extend(
            [
                "No 960 raw-split candidate both removed top-k and passed the Mac proxy in this run. Treat true ANE-960 as unresolved/impossible for now.",
                "",
                "Prefer the measured 960 GPU/CPU fallback for single-pass recall, or 640 ANE tiling when the live tier needs ANE residency and thermal margin.",
            ]
        )
    (run_dir / "ANE_960_DIAGNOSTIC_REPORT.md").write_text("\n".join(lines) + "\n")


def _probe_cell(row: dict[str, Any]) -> str:
    if not row:
        return "n/a"
    if row.get("status") != "ok":
        return f"error: {row.get('error', '')[:60]}"
    return f"ok {float(row.get('median_ms', 0.0)):.2f}ms"


def parse_args() -> argparse.Namespace:
    default_run_dir = REPO_ROOT / "runs" / f"coreml_960_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=default_run_dir)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--include-resolution-sweep", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir if args.run_dir.is_absolute() else REPO_ROOT / args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "environment.json", collect_environment())
    write_json(
        run_dir / "shape_hypothesis.json",
        {
            "640": postprocess_tensor_shapes(640),
            "960": postprocess_tensor_shapes(960),
            "hypothesis": "960 increases first topk input from 8400 to 18900 grid points; device ANE compiler rejects that tail.",
        },
    )

    checkpoint = select_yolo_checkpoint()
    variants = default_variants(args.include_resolution_sweep)
    results: list[dict[str, Any]] = []
    graph_summaries: dict[str, Any] = {}

    for variant in variants:
        name = variant["name"]
        work_dir = run_dir / name
        work_dir.mkdir(parents=True, exist_ok=True)
        if variant["kind"] == "copy_existing":
            conversion = copy_existing_variant(name, Path(variant["source"]), work_dir)
        elif variant["kind"] == "ultralytics":
            conversion = export_ultralytics_variant(name, checkpoint, int(variant["imgsz"]), work_dir, half=bool(variant["half"]))
        elif variant["kind"] == "direct":
            conversion = export_direct_variant(
                name,
                checkpoint,
                int(variant["imgsz"]),
                work_dir,
                output_mode=str(variant["output_mode"]),
                input_kind=str(variant["input_kind"]),
                precision_name=variant.get("precision"),
                target_name=variant.get("target"),
                enumerated_imgszs=variant.get("enumerated_imgszs"),
            )
        else:
            conversion = {"name": name, "status": "error", "error": f"unknown variant kind {variant['kind']}"}

        result: dict[str, Any] = {"name": name, "variant": variant, "conversion": conversion}
        if conversion.get("status") == "ok":
            model_path = Path(conversion["model_path"])
            try:
                graph_summaries[name] = summarize_mlprogram_ops(model_path)
                result["probe"] = compile_and_probe(model_path, work_dir, args.iterations)
            except Exception as exc:
                result["probe"] = {
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
        result["status"] = (
            "ok"
            if conversion.get("status") == "ok" and result.get("probe", {}).get("status") == "ok"
            else "error"
        )
        write_json(work_dir / "variant_result.json", result)
        results.append(result)
        write_json(run_dir / "variant_results_partial.json", results)

    write_json(run_dir / "variant_results.json", results)
    write_json(run_dir / "graph_summaries.json", graph_summaries)
    write_json(run_dir / "fallback_projection.json", fallback_projection())
    write_markdown_report(run_dir, results, graph_summaries)
    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

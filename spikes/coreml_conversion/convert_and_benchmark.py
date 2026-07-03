#!/usr/bin/env python3
"""CoreML conversion and Mac proxy benchmarking for the live-tier spike."""

from __future__ import annotations

import argparse
import asyncio
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

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mpl")
os.environ.setdefault("COREMLTOOLS_HOME", "/private/tmp/coremltools")
os.environ.setdefault("TMPDIR", "/private/tmp")
os.environ.setdefault("TEMP", "/private/tmp")
os.environ.setdefault("TMP", "/private/tmp")

import numpy as np
import torch

import coremltools as ct
from coremltools import proto
from coremltools.models.ml_program.experimental.perf_utils import MLModelBenchmarker
from ultralytics import YOLO

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from ball_student import WASBLiteBallStudent, count_parameters

COMPUTE_UNITS = ("CPU_ONLY", "CPU_AND_NE")
YOLO_IMGSZS = (640, 960)
BALL_IMGSZ = "288x512"
LIVE_BUDGET_MS = 1000.0 / 30.0


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


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n")


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
        return {"version": str(getattr(module, "__version__", "unknown")), "file": str(getattr(module, "__file__", ""))}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def collect_environment() -> dict[str, Any]:
    pyvenv_cfg = SCRIPT_DIR / ".venv" / "pyvenv.cfg"
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
            "pytest": import_version("pytest"),
        },
        "pyvenv_cfg": pyvenv_cfg.read_text() if pyvenv_cfg.exists() else None,
        "package_install_note": (
            "A normal isolated pip install was attempted first and failed because DNS/network "
            "is unavailable in this sandbox. .venv is configured with include-system-site-packages=true "
            "for this run, so package imports resolve to the existing local Anaconda installs."
        ),
    }


def select_yolo_checkpoint() -> Path:
    preferred = REPO_ROOT / "models/checkpoints/yolo26n.pt"
    fallback = REPO_ROOT / "models/checkpoints/yolo26m.pt"
    if preferred.exists():
        return preferred
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"Missing both {preferred} and {fallback}")


def spec_type(path: Path) -> str:
    model = ct.models.MLModel(str(path), compute_units=ct.ComputeUnit.CPU_ONLY)
    return model.get_spec().WhichOneof("Type")


def feature_shape(feature: Any) -> list[int]:
    feature_type = feature.type.WhichOneof("Type")
    if feature_type == "multiArrayType":
        multi = feature.type.multiArrayType
        if multi.shape:
            return [int(dim) for dim in multi.shape]
        if multi.WhichOneof("ShapeFlexibility") == "shapeRange":
            return [int(dim.lowerBound) for dim in multi.shapeRange.sizeRanges]
    if feature_type == "imageType":
        image = feature.type.imageType
        return [int(image.height), int(image.width)]
    return []


def describe_model_io(path: Path) -> dict[str, Any]:
    model = ct.models.MLModel(str(path), compute_units=ct.ComputeUnit.CPU_ONLY)
    spec = model.get_spec()
    return {
        "spec_type": spec.WhichOneof("Type"),
        "inputs": [
            {
                "name": feature.name,
                "type": feature.type.WhichOneof("Type"),
                "shape": feature_shape(feature),
            }
            for feature in spec.description.input
        ],
        "outputs": [
            {
                "name": feature.name,
                "type": feature.type.WhichOneof("Type"),
                "shape": feature_shape(feature),
            }
            for feature in spec.description.output
        ],
    }


def random_multiarray(feature_type: Any) -> np.ndarray:
    dtype_map = {
        proto.FeatureTypes_pb2.ArrayFeatureType.FLOAT32: np.float32,
        proto.FeatureTypes_pb2.ArrayFeatureType.FLOAT16: np.float16,
        proto.FeatureTypes_pb2.ArrayFeatureType.DOUBLE: np.float64,
        proto.FeatureTypes_pb2.ArrayFeatureType.INT32: np.int32,
    }
    dtype = dtype_map.get(feature_type.dataType, np.float32)
    shape = [int(dim) for dim in feature_type.shape]
    if not shape and feature_type.WhichOneof("ShapeFlexibility") == "shapeRange":
        shape = [int(dim.lowerBound) for dim in feature_type.shapeRange.sizeRanges]
    if dtype == np.int32:
        return np.random.randint(0, 10, shape, dtype=dtype)
    return np.random.random(shape).astype(dtype)


def random_image(feature_type: Any) -> Any:
    from PIL import Image

    color = feature_type.colorSpace
    width = int(feature_type.width)
    height = int(feature_type.height)
    if color == proto.FeatureTypes_pb2.ImageFeatureType.GRAYSCALE:
        return Image.fromarray(np.random.randint(0, 256, (height, width), dtype=np.uint8), mode="L")
    if color == proto.FeatureTypes_pb2.ImageFeatureType.GRAYSCALE_FLOAT16:
        return Image.fromarray(np.random.random((height, width)).astype(np.float32), mode="F")
    return Image.fromarray(np.random.randint(0, 256, (height, width, 3), dtype=np.uint8), mode="RGB")


def random_inputs(model: ct.models.MLModel) -> dict[str, Any]:
    inputs = {}
    for feature in model.get_spec().description.input:
        feature_type = feature.type.WhichOneof("Type")
        if feature_type == "multiArrayType":
            inputs[feature.name] = random_multiarray(feature.type.multiArrayType)
        elif feature_type == "imageType":
            inputs[feature.name] = random_image(feature.type.imageType)
        else:
            raise ValueError(f"Unsupported input feature {feature.name}: {feature_type}")
    return inputs


async def benchmark_with_coremltools(
    model: ct.models.MLModel,
    inputs: dict[str, Any],
    iterations: int,
) -> MLModelBenchmarker.Measurement:
    benchmarker = MLModelBenchmarker(model)
    return await benchmarker.benchmark_predict(inputs=inputs, iterations=iterations, warmup=True)


def compile_with_coremlc(model_path: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    result = run_command(["xcrun", "coremlc", "compile", str(model_path), str(output_dir)])
    compiled = sorted(output_dir.glob("*.mlmodelc"))
    result["compiled_model_path"] = compiled[0] if compiled else None
    if result["returncode"] != 0:
        raise RuntimeError(result["stderr"] or result["stdout"] or "coremlc compile failed")
    if not compiled:
        raise FileNotFoundError(f"coremlc did not create an .mlmodelc under {output_dir}")
    return result


def measurement_to_dict(measurement: MLModelBenchmarker.Measurement) -> dict[str, Any]:
    stats = measurement.statistics
    return {
        "samples_ms": measurement.samples,
        "minimum_ms": None if stats is None else stats.minimum,
        "maximum_ms": None if stats is None else stats.maximum,
        "average_ms": None if stats is None else stats.average,
        "std_dev_ms": None if stats is None else stats.std_dev,
        "median_ms": None if stats is None else stats.median,
    }


def benchmark_compiled_model(
    compiled_model_path: Path,
    compute_unit_name: str,
    inputs: dict[str, Any],
    iterations: int,
) -> dict[str, Any]:
    compute_unit = getattr(ct.ComputeUnit, compute_unit_name)
    model = ct.models.CompiledMLModel(str(compiled_model_path), compute_units=compute_unit)
    _ = model.predict(inputs)
    samples = []
    for _index in range(iterations):
        started = time.perf_counter()
        _ = model.predict(inputs)
        samples.append((time.perf_counter() - started) * 1000.0)
    return measurement_to_dict(MLModelBenchmarker.Measurement.from_samples(samples))


def benchmark_model(
    model_path: Path,
    model_name: str,
    imgsz: str | int,
    run_dir: Path,
    iterations: int,
) -> list[dict[str, Any]]:
    rows = []
    compile_result: dict[str, Any] | None = None
    compiled_model_path: Path | None = None
    for compute_unit_name in COMPUTE_UNITS:
        row: dict[str, Any] = {
            "model": model_name,
            "imgsz": imgsz,
            "compute_unit": compute_unit_name,
            "latency_label": "Mac ANE proxy -- not iPhone numbers",
            "model_path": model_path,
            "benchmark_api_requested": "coremltools.models.ml_program.experimental.perf_utils.MLModelBenchmarker.benchmark_predict",
        }
        try:
            compute_unit = getattr(ct.ComputeUnit, compute_unit_name)
            model = ct.models.MLModel(str(model_path), compute_units=compute_unit, skip_model_load=True)
            inputs = random_inputs(model)
            try:
                measurement = asyncio.run(benchmark_with_coremltools(model, inputs, iterations))
                row.update(
                    {
                        "status": "ok",
                        "benchmark_api_used": row["benchmark_api_requested"],
                        "coremltools_benchmarker_status": "ok",
                        **measurement_to_dict(measurement),
                    }
                )
            except Exception as benchmarker_exc:
                row["coremltools_benchmarker_status"] = "error"
                row["coremltools_benchmarker_error"] = f"{type(benchmarker_exc).__name__}: {benchmarker_exc}"
                row["coremltools_benchmarker_traceback"] = traceback.format_exc()
                if compiled_model_path is None:
                    compile_result = compile_with_coremlc(
                        model_path,
                        model_path.parent / f"compiled_{model_path.stem}",
                    )
                    compiled_model_path = Path(compile_result["compiled_model_path"])
                    write_json(model_path.parent / "coremlc_compile_result.json", compile_result)
                row.update(
                    {
                        "status": "ok",
                        "benchmark_api_used": "coremltools.models.CompiledMLModel.predict manual timing after xcrun coremlc compile",
                        "compiled_model_path": compiled_model_path,
                        "coremlc_compile_result": compile_result,
                        **benchmark_compiled_model(compiled_model_path, compute_unit_name, inputs, iterations),
                    }
                )
            median = row["median_ms"]
            row["projected_fps"] = None if median in (None, 0) else 1000.0 / float(median)
            row["single_model_budget"] = "pass" if median is not None and float(median) <= LIVE_BUDGET_MS else "fail"
        except Exception as exc:
            row.update(
                {
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
            )
        rows.append(row)
    write_json(run_dir / "benchmark_rows_partial.json", rows)
    return rows


def convert_yolo_checkpoint(checkpoint: Path, imgsz: int, run_dir: Path) -> dict[str, Any]:
    work_dir = run_dir / "yolo_player_detector" / f"imgsz_{imgsz}"
    work_dir.mkdir(parents=True, exist_ok=True)
    local_checkpoint = work_dir / checkpoint.name
    shutil.copy2(checkpoint, local_checkpoint)
    old_cwd = Path.cwd()
    started = time.perf_counter()
    result: dict[str, Any] = {
        "source_checkpoint": checkpoint,
        "local_checkpoint": local_checkpoint,
        "source_sha256": sha256_file(checkpoint),
        "imgsz": imgsz,
        "format": "coreml",
        "precision_request": "FP16",
        "program_request": "mlprogram",
    }
    try:
        os.chdir(work_dir)
        model = YOLO(local_checkpoint.name)
        exported = model.export(format="coreml", imgsz=imgsz, half=True, nms=False, batch=1)
        exported_path = Path(exported)
        if not exported_path.is_absolute():
            exported_path = work_dir / exported_path
        if not exported_path.exists():
            candidates = sorted(work_dir.glob("*.mlpackage")) + sorted(work_dir.glob("*.mlmodel"))
            if not candidates:
                raise FileNotFoundError(f"Ultralytics export returned {exported!r}, but no CoreML artifact exists in {work_dir}")
            exported_path = candidates[0]
        result.update(
            {
                "status": "ok",
                "exported_path": exported_path,
                "duration_seconds": time.perf_counter() - started,
                "package_size_bytes": package_size_bytes(exported_path),
                "spec_type": spec_type(exported_path),
                "io": describe_model_io(exported_path),
            }
        )
    except Exception as exc:
        result.update(
            {
                "status": "error",
                "duration_seconds": time.perf_counter() - started,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        os.chdir(old_cwd)
    write_json(work_dir / "conversion_result.json", result)
    return result


def convert_ball_student(run_dir: Path) -> dict[str, Any]:
    work_dir = run_dir / "ball_student_untrained"
    work_dir.mkdir(parents=True, exist_ok=True)
    output_path = work_dir / "wasb_lite_ball_student_untrained_fp16.mlpackage"
    started = time.perf_counter()
    result: dict[str, Any] = {
        "model": "WASBLiteBallStudent",
        "input_shape": [1, 9, 288, 512],
        "output_shape": [1, 1, 72, 128],
        "format": "coreml",
        "precision_request": "FP16",
        "program_request": "mlprogram",
        "trained": False,
    }
    try:
        model = WASBLiteBallStudent().eval()
        params = count_parameters(model)
        example = torch.zeros(1, 9, 288, 512)
        with torch.no_grad():
            traced = torch.jit.trace(model, example, strict=False)
        mlmodel = ct.convert(
            traced,
            convert_to="mlprogram",
            inputs=[ct.TensorType(name="frames", shape=example.shape, dtype=np.float32)],
            outputs=[ct.TensorType(name="heatmap")],
            compute_precision=ct.precision.FLOAT16,
            minimum_deployment_target=ct.target.macOS13,
        )
        mlmodel.save(str(output_path))
        result.update(
            {
                "status": "ok",
                "exported_path": output_path,
                "parameter_count": params,
                "duration_seconds": time.perf_counter() - started,
                "package_size_bytes": package_size_bytes(output_path),
                "spec_type": spec_type(output_path),
                "io": describe_model_io(output_path),
            }
        )
    except Exception as exc:
        result.update(
            {
                "status": "error",
                "duration_seconds": time.perf_counter() - started,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )
    write_json(work_dir / "conversion_result.json", result)
    return result


def load_manifest_models() -> list[dict[str, Any]]:
    manifest = REPO_ROOT / "models/MANIFEST.json"
    if not manifest.exists():
        return []
    payload = json.loads(manifest.read_text())
    return payload.get("models", [])


def rtmpose_status(run_dir: Path) -> dict[str, Any]:
    patterns = ("*rtmpose*m*.pth", "*rtmpose-m*.pth", "*rtmpose_m*.pth", "*mmpose*rtmpose*.pth")
    local_matches: list[Path] = []
    for base in (REPO_ROOT / "models", REPO_ROOT / "third_party"):
        if base.exists():
            for pattern in patterns:
                local_matches.extend(base.rglob(pattern))
    local_matches = sorted(set(local_matches))
    manifest_records = [
        item
        for item in load_manifest_models()
        if item.get("id") in {"rtmpose_m_body26_384", "rtmpose_m_wholebody_256"}
    ]
    body26 = next((item for item in manifest_records if item.get("id") == "rtmpose_m_body26_384"), None)
    status = {
        "local_rtmpose_weight_matches": local_matches,
        "manifest_records": manifest_records,
        "will_convert": bool(local_matches),
        "decision": "",
        "acquisition_path": {
            "repo": "https://github.com/open-mmlab/mmpose",
            "checkpoint_url": (
                (body26 or {}).get("source", "").split(" and ")[-1]
                or "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-body7_pt-body7-halpe26_700e-384x288-89e6428b_20230605.pth"
            ),
            "expected_manifest_id": "rtmpose_m_body26_384",
            "expected_manifest_sha256": (body26 or {}).get("sha256"),
            "expected_manifest_local_path": (body26 or {}).get("local_path"),
            "note": "Acquire the MMPose RTMPose project/config plus this checkpoint before conversion; do not use YOLO pose weights as an RTMPose substitute.",
        },
    }
    if local_matches:
        status["decision"] = "Local RTMPose-looking weights exist, but this script does not auto-convert MMPose graphs without the matching config/runtime path."
    else:
        status["decision"] = "No RTMPose-m weights found under models/ or third_party/ on this Mac; conversion skipped and acquisition path documented."
    write_json(run_dir / "rtmpose_status.json", status)
    md = [
        "# RTMPose-m Status",
        "",
        "No RTMPose-m conversion was run unless `local_rtmpose_weight_matches` is non-empty and a matching MMPose config/runtime is present.",
        "",
        f"- Decision: {status['decision']}",
        f"- MMPose repo: {status['acquisition_path']['repo']}",
        f"- Checkpoint URL: {status['acquisition_path']['checkpoint_url']}",
        "- Official source context checked: MMPose repository/model zoo and RTMPose paper/project references list RTMPose as an MMPose/OpenMMLab model family and include RTMPose-m 384x288 rows for Halpe/body variants.",
        f"- Manifest id: {status['acquisition_path']['expected_manifest_id']}",
        f"- Manifest sha256: {status['acquisition_path']['expected_manifest_sha256']}",
        f"- Manifest local path: {status['acquisition_path']['expected_manifest_local_path']}",
        "",
    ]
    (run_dir / "RTMPOSE_STATUS.md").write_text("\n".join(md))
    return status


def format_ms(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.2f}"


def format_fps(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.1f}"


def summarize_total_loop(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    totals = []
    for compute_unit in COMPUTE_UNITS:
        ball = next((row for row in rows if row["model"] == "ball_student_untrained" and row["compute_unit"] == compute_unit and row.get("status") == "ok"), None)
        for imgsz in YOLO_IMGSZS:
            yolo = next((row for row in rows if row["model"] == "yolo26n_player_detector" and row["imgsz"] == imgsz and row["compute_unit"] == compute_unit and row.get("status") == "ok"), None)
            if not ball or not yolo:
                totals.append(
                    {
                        "loop": f"player_yolo_{imgsz}+ball_student_288x512",
                        "compute_unit": compute_unit,
                        "status": "not_projectable",
                        "reason": "missing successful model benchmark",
                    }
                )
                continue
            total_ms = float(ball["median_ms"]) + float(yolo["median_ms"])
            totals.append(
                {
                    "loop": f"player_yolo_{imgsz}+ball_student_288x512",
                    "compute_unit": compute_unit,
                    "status": "pass" if total_ms <= LIVE_BUDGET_MS else "fail",
                    "median_ms_sum": total_ms,
                    "projected_fps": 1000.0 / total_ms,
                    "note": "Player detector + ball student only; RTMPose/iOS camera/render overhead absent.",
                }
            )
    return totals


def write_latency_table(run_dir: Path, rows: list[dict[str, Any]], conversions: dict[str, Any], totals: list[dict[str, Any]]) -> None:
    lines = [
        "# CoreML Live-Tier Latency Table",
        "",
        "**All latency numbers are Mac ANE proxy -- not iPhone numbers.**",
        "",
        f"Live-tier budget from MASTER_PLAN/W3-LIVE-MLP: total loop must sustain >=30fps, i.e. <= {LIVE_BUDGET_MS:.2f} ms/frame before camera/render overhead.",
        "",
        "## Package Setup",
        "",
        "A normal isolated pip install into `spikes/coreml_conversion/.venv` failed because the sandbox has no DNS/network access. The venv was then configured with `include-system-site-packages=true`, so this run uses the existing local Anaconda installs through `.venv/bin/python`.",
        "",
        "CoreMLTools' `MLModelBenchmarker.benchmark_predict` was attempted first. In this sandbox it failed to load generated packages with `ValueError: Failed to load model` after CoreML reported a temp working-directory error, so the reported timings use `xcrun coremlc compile` plus `coremltools.models.CompiledMLModel.predict` manual timing. The exact benchmarker errors are preserved in `benchmark_rows.json`.",
        "",
        "## Model Benchmarks",
        "",
        "| Model | imgsz | Compute unit | Status | Median ms/frame | Avg ms/frame | Projected fps | Single-model budget | Benchmark API used | Artifact/error |",
        "|---|---:|---|---|---:|---:|---:|---|---|---|",
    ]
    for row in rows:
        artifact = row.get("model_path") or row.get("error", "")
        if row.get("status") != "ok":
            artifact = row.get("error", "error")
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["model"]),
                    str(row["imgsz"]),
                    str(row["compute_unit"]),
                    str(row.get("status")),
                    format_ms(row.get("median_ms")),
                    format_ms(row.get("average_ms")),
                    format_fps(row.get("projected_fps")),
                    str(row.get("single_model_budget", "n/a")),
                    str(row.get("benchmark_api_used", row.get("benchmark_api_requested", "n/a"))),
                    f"`{artifact}`",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Total Loop Projection",
            "",
            "This is a partial live-loop projection only. It sums the player detector and untrained ball student medians where both benchmarks succeeded. It excludes RTMPose, camera capture, tracking, rendering, thermal throttling, and physical iPhone overhead.",
            "",
            "| Loop | Compute unit | Status vs 30fps total loop | Median sum ms/frame | Projected fps | Note |",
            "|---|---|---|---:|---:|---|",
        ]
    )
    for item in totals:
        lines.append(
            "| "
            + " | ".join(
                [
                    item["loop"],
                    item["compute_unit"],
                    item["status"],
                    format_ms(item.get("median_ms_sum")),
                    format_fps(item.get("projected_fps")),
                    item.get("note") or item.get("reason", ""),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Conversion Summary", ""])
    for name, payload in conversions.items():
        if isinstance(payload, list):
            for item in payload:
                lines.append(f"- {name} imgsz {item.get('imgsz')}: {item.get('status')} ({item.get('exported_path') or item.get('error')})")
        else:
            lines.append(f"- {name}: {payload.get('status')} ({payload.get('exported_path') or payload.get('decision') or payload.get('error')})")
    lines.append("")
    (run_dir / "LATENCY_TABLE.md").write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    default_run_dir = REPO_ROOT / "runs" / f"coreml_prep_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=default_run_dir)
    parser.add_argument("--iterations", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir
    if not run_dir.is_absolute():
        run_dir = REPO_ROOT / run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    write_json(run_dir / "environment.json", collect_environment())

    conversions: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []

    try:
        yolo_checkpoint = select_yolo_checkpoint()
        yolo_results = []
        yolo_model_name = f"{yolo_checkpoint.stem}_player_detector"
        for imgsz in YOLO_IMGSZS:
            result = convert_yolo_checkpoint(yolo_checkpoint, imgsz, run_dir)
            yolo_results.append(result)
            if result.get("status") == "ok":
                rows.extend(benchmark_model(Path(result["exported_path"]), yolo_model_name, imgsz, run_dir, args.iterations))
        conversions["yolo_player_detector"] = yolo_results
    except Exception as exc:
        conversions["yolo_player_detector"] = {
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }

    ball_result = convert_ball_student(run_dir)
    conversions["ball_student_untrained"] = ball_result
    if ball_result.get("status") == "ok":
        rows.extend(benchmark_model(Path(ball_result["exported_path"]), "ball_student_untrained", BALL_IMGSZ, run_dir, args.iterations))

    conversions["rtmpose_m"] = rtmpose_status(run_dir)

    write_json(run_dir / "benchmark_rows.json", rows)
    write_json(run_dir / "conversion_summary.json", conversions)
    totals = summarize_total_loop(rows)
    write_json(run_dir / "total_loop_projection.json", totals)
    write_latency_table(run_dir, rows, conversions, totals)

    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

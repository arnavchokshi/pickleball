#!/usr/bin/env python3
"""Synthetic BODY decode-fidelity gate instrument.

This CPU-runnable arm authors simple metric body geometry, renders an inspectable
projection, applies a decoder adapter, and reports the standing metric keys used
by the R1 decode-fidelity checklist. The `mock` decoder is a deterministic
self-check. The `sam3d` decoder mode is a VM-facing placeholder that fails loud
until the GPU runtime adapter is wired.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport import mhr_decode  # noqa: E402


ARTIFACT_TYPE = "racketsport_synthetic_body_decode_gate"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Author a metric synthetic body render and measure BODY decode round-trip "
            "keys: gate_1b_world_round_trip.joints_world_p95_abs_error_mm and "
            "mesh_skeleton_divergence.p95_mm."
        )
    )
    parser.add_argument("--out", type=Path, required=True, help="Output JSON report path.")
    parser.add_argument("--render-dir", type=Path, required=True, help="Directory for synthetic render PPM files.")
    parser.add_argument("--samples", type=int, default=3, help="Number of synthetic poses to author.")
    parser.add_argument("--seed", type=int, default=0, help="Deterministic synthetic sample seed.")
    parser.add_argument(
        "--decoder",
        choices=("mock", "sam3d"),
        default="mock",
        help="'mock' runs CPU self-check; 'sam3d' is the same command shape for GPU VM wiring.",
    )
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.samples < 1:
        raise ValueError("--samples must be >= 1")
    t0 = time.time()
    args.render_dir.mkdir(parents=True, exist_ok=True)
    samples = [_author_sample(index, seed=args.seed) for index in range(args.samples)]
    renders = []
    for sample in samples:
        render_path = args.render_dir / f"synthetic_body_{sample['sample_id']:04d}.ppm"
        _write_ppm_render(render_path, sample["authored_joints_world"], sample["authored_vertices_world"])
        renders.append({"sample_id": sample["sample_id"], "path": render_path.name, "width": 320, "height": 240})

    if args.decoder == "sam3d":
        report = _base_report(args=args, samples=samples, renders=renders, wall_seconds=time.time() - t0)
        report["measurement_status"] = "blocked_sam3d_runtime_unwired"
        report["blocker"] = (
            "GPU SAM-3D-Body render-ingest adapter is not wired in this CPU lane; "
            "the authored renders and metric report shape are ready for the VM arm."
        )
        _write_report(args.out, report)
        return report

    decoded_samples = [_mock_decode(sample) for sample in samples]
    joint_errors = []
    vertex_errors = []
    divergences = []
    for sample, decoded in zip(samples, decoded_samples, strict=True):
        joint_errors.extend(_point_errors_mm(decoded["joints_world"], sample["authored_joints_world"]))
        vertex_errors.extend(_point_errors_mm(decoded["vertices_world"], sample["authored_vertices_world"]))
        divergences.extend(_nearest_errors_mm(decoded["joints_world"], decoded["vertices_world"]))

    joints_p95 = _percentile(joint_errors, 95)
    vertices_p95 = _percentile(vertex_errors, 95)
    divergence_p95 = _percentile(divergences, 95)
    report = _base_report(args=args, samples=samples, renders=renders, wall_seconds=time.time() - t0)
    report.update(
        {
            "measurement_status": "measured_mock_decoder",
            "blocker": None,
            "gate_1b_world_round_trip": {
                "metric": "gate_1b_world_round_trip.joints_world_p95_abs_error_mm",
                "joints_world_p95_abs_error_mm": joints_p95,
                "vertices_world_p95_abs_error_mm": vertices_p95,
                "target_joints_world_p95_abs_error_mm": mhr_decode.GATE_1B_MAX_ABS_ERROR_MM,
                "passed": bool(
                    joints_p95 <= mhr_decode.GATE_1B_MAX_ABS_ERROR_MM
                    and vertices_p95 <= mhr_decode.GATE_1B_MAX_ABS_ERROR_MM
                ),
            },
            "mesh_skeleton_divergence": {
                "metric": "mesh_skeleton_divergence.p95_mm",
                "p95_mm": divergence_p95,
                "target_p95_mm": mhr_decode.MESH_SKELETON_DIVERGENCE_P95_MM,
                "passed": bool(divergence_p95 <= mhr_decode.MESH_SKELETON_DIVERGENCE_P95_MM),
            },
        }
    )
    _write_report(args.out, report)
    return report


def _base_report(
    *,
    args: argparse.Namespace,
    samples: Sequence[dict[str, Any]],
    renders: Sequence[dict[str, Any]],
    wall_seconds: float,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact_type": ARTIFACT_TYPE,
        "measurement_status": "not_measured",
        "blocker": None,
        "decoder": args.decoder,
        "sample_count": len(samples),
        "recipe": {
            "author_geometry": "metric stick-body joints plus mesh vertices in meters",
            "camera_model": "identity extrinsics plus pinhole projection for render inspection",
            "translation_policy": "pred_cam_t is applied exactly once to both joints and vertices",
            "cpu_arm": "mock decoder validates metric plumbing without GPU runtime",
            "gpu_arm": "same CLI shape with --decoder sam3d for future VM adapter",
        },
        "gate_1b_world_round_trip": {
            "metric": "gate_1b_world_round_trip.joints_world_p95_abs_error_mm",
            "joints_world_p95_abs_error_mm": None,
            "vertices_world_p95_abs_error_mm": None,
            "target_joints_world_p95_abs_error_mm": mhr_decode.GATE_1B_MAX_ABS_ERROR_MM,
            "passed": None,
        },
        "mesh_skeleton_divergence": {
            "metric": "mesh_skeleton_divergence.p95_mm",
            "p95_mm": None,
            "target_p95_mm": mhr_decode.MESH_SKELETON_DIVERGENCE_P95_MM,
            "passed": None,
        },
        "renders": list(renders),
        "samples": [
            {
                "sample_id": sample["sample_id"],
                "scale_m": sample["scale_m"],
                "pred_cam_t": sample["pred_cam_t"],
                "joint_count": len(sample["authored_joints_world"]),
                "vertex_count": len(sample["authored_vertices_world"]),
            }
            for sample in samples
        ],
        "wall_seconds": wall_seconds,
    }


def _author_sample(index: int, *, seed: int) -> dict[str, Any]:
    rng = random.Random(seed + index * 1009)
    scale = 0.95 + 0.1 * rng.random()
    lean = (rng.random() - 0.5) * 0.08
    local_joints = [
        [0.0, 0.0, 0.0],
        [0.0 + lean, 0.0, 0.85 * scale],
        [0.0 + lean * 1.5, 0.0, 1.55 * scale],
        [-0.25 * scale, 0.0, 1.20 * scale],
        [0.25 * scale, 0.0, 1.20 * scale],
        [-0.16 * scale, 0.0, 0.45 * scale],
        [0.16 * scale, 0.0, 0.45 * scale],
    ]
    # Include every joint as a mesh vertex so the synthetic mesh/joint
    # divergence is zero for the mock adapter, plus two extra body extents.
    local_vertices = [list(point) for point in local_joints]
    local_vertices.extend([[-0.18 * scale, -0.08 * scale, 0.9 * scale], [0.18 * scale, 0.08 * scale, 0.9 * scale]])
    pred_cam_t = [0.2 + 0.03 * index, -0.1 + 0.02 * index, 3.0 + 0.1 * index]
    authored_joints_world = mhr_decode.apply_pred_cam_t_once(local_joints, pred_cam_t=pred_cam_t)
    authored_vertices_world = mhr_decode.apply_pred_cam_t_once(local_vertices, pred_cam_t=pred_cam_t)
    return {
        "sample_id": index,
        "scale_m": scale,
        "pred_cam_t": pred_cam_t,
        "model_joints_without_cam_t": local_joints,
        "model_vertices_without_cam_t": local_vertices,
        "authored_joints_world": authored_joints_world,
        "authored_vertices_world": authored_vertices_world,
    }


def _mock_decode(sample: dict[str, Any]) -> dict[str, list[list[float]]]:
    return {
        "joints_world": mhr_decode.apply_pred_cam_t_once(
            sample["model_joints_without_cam_t"],
            pred_cam_t=sample["pred_cam_t"],
        ),
        "vertices_world": mhr_decode.apply_pred_cam_t_once(
            sample["model_vertices_without_cam_t"],
            pred_cam_t=sample["pred_cam_t"],
        ),
    }


def _write_ppm_render(path: Path, joints: Sequence[Sequence[float]], vertices: Sequence[Sequence[float]]) -> None:
    width, height = 320, 240
    pixels = [[[255, 255, 255] for _ in range(width)] for _ in range(height)]
    for point in vertices:
        _draw_point(pixels, _project(point, width=width, height=height), [80, 120, 220])
    for point in joints:
        _draw_point(pixels, _project(point, width=width, height=height), [220, 40, 40])
    lines = [f"P3\n{width} {height}\n255\n"]
    for row in pixels:
        lines.append(" ".join(f"{r} {g} {b}" for r, g, b in row))
        lines.append("\n")
    path.write_text("".join(lines), encoding="ascii")


def _project(point: Sequence[float], *, width: int, height: int) -> tuple[int, int]:
    x, y, z = [float(value) for value in point]
    focal = 95.0
    safe_z = max(z, 0.1)
    u = int(round(width / 2 + focal * x / safe_z))
    v = int(round(height * 0.88 - focal * y / safe_z - focal * (z - 3.0) / safe_z))
    return max(0, min(width - 1, u)), max(0, min(height - 1, v))


def _draw_point(pixels: list[list[list[int]]], point: tuple[int, int], color: list[int]) -> None:
    cx, cy = point
    height = len(pixels)
    width = len(pixels[0])
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            x = cx + dx
            y = cy + dy
            if 0 <= x < width and 0 <= y < height:
                pixels[y][x] = color


def _point_errors_mm(decoded: Sequence[Sequence[float]], expected: Sequence[Sequence[float]]) -> list[float]:
    errors = []
    for got, want in zip(decoded, expected, strict=True):
        errors.append(math.sqrt(sum((float(got[idx]) - float(want[idx])) ** 2 for idx in range(3))) * 1000.0)
    return errors


def _nearest_errors_mm(points: Sequence[Sequence[float]], cloud: Sequence[Sequence[float]]) -> list[float]:
    errors = []
    for point in points:
        errors.append(min(_distance_m(point, candidate) for candidate in cloud) * 1000.0)
    return errors


def _distance_m(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((float(a[idx]) - float(b[idx])) ** 2 for idx in range(3)))


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    rank = (len(ordered) - 1) * percentile / 100.0
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return ordered[lower]
    frac = rank - lower
    return ordered[lower] * (1.0 - frac) + ordered[upper] * frac


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        report = run(args)
    except ValueError as exc:
        parser.exit(2, f"{parser.prog}: error: {exc}\n")
    print(
        json.dumps(
            {
                "measurement_status": report["measurement_status"],
                "out": str(args.out),
                "sample_count": report["sample_count"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 2 if str(report["measurement_status"]).startswith("blocked_") else 0


if __name__ == "__main__":
    raise SystemExit(main())

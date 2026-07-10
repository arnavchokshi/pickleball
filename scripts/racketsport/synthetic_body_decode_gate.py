#!/usr/bin/env python3
"""Synthetic BODY decode-fidelity gate instrument.

This CPU-runnable arm authors simple metric body geometry, renders an inspectable
projection, applies a decoder adapter, and reports the standing metric keys used
by the R1 decode-fidelity checklist. The `mock` decoder is a deterministic
self-check. The `sam3d` decoder mode authors an actual MHR mesh, renders a
two-attempt shaded realism ladder, and invokes the production Fast-SAM runtime;
optional-runtime failures are explicit blocked statuses.
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

import numpy as np

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
    parser.add_argument("--checkpoint", type=Path, default=Path(mhr_decode.DEFAULT_CHECKPOINT_PATH))
    parser.add_argument("--mhr-asset", type=Path, default=Path(mhr_decode.DEFAULT_MHR_ASSET_PATH))
    parser.add_argument("--device", default=None, help="MHR/SAM-3D-Body device override, e.g. cuda:0.")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.samples < 1:
        raise ValueError("--samples must be >= 1")
    t0 = time.time()
    args.render_dir.mkdir(parents=True, exist_ok=True)
    if args.decoder == "sam3d":
        return _run_sam3d_adapter(args, started_at=t0)

    samples = [_author_sample(index, seed=args.seed) for index in range(args.samples)]
    renders = []
    for sample in samples:
        render_path = args.render_dir / f"synthetic_body_{sample['sample_id']:04d}.ppm"
        _write_ppm_render(render_path, sample["authored_joints_world"], sample["authored_vertices_world"])
        renders.append({"sample_id": sample["sample_id"], "path": render_path.name, "width": 320, "height": 240})

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
    authored_joints_world = _author_expected_translated_points(local_joints, pred_cam_t)
    authored_vertices_world = _author_expected_translated_points(local_vertices, pred_cam_t)
    return {
        "sample_id": index,
        "scale_m": scale,
        "pred_cam_t": pred_cam_t,
        "model_joints_without_cam_t": local_joints,
        "model_vertices_without_cam_t": local_vertices,
        "authored_joints_world": authored_joints_world,
        "authored_vertices_world": authored_vertices_world,
    }


def _author_expected_translated_points(
    points_camera: Sequence[Sequence[float]],
    pred_cam_t: Sequence[float],
) -> list[list[float]]:
    """Author synthetic ground truth without using decode-path helpers."""
    if len(pred_cam_t) != 3:
        raise ValueError("pred_cam_t must be a 3-vector")
    tx, ty, tz = [float(value) for value in pred_cam_t]
    return [[float(point[0]) + tx, float(point[1]) + ty, float(point[2]) + tz] for point in points_camera]


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


def _run_sam3d_adapter(args: argparse.Namespace, *, started_at: float) -> dict[str, Any]:
    """Author real MHR meshes, render them, and run the production SAM runtime."""

    if not mhr_decode.MHR_RUNTIME_AVAILABLE:
        return _write_blocked_sam3d_report(
            args,
            status="blocked_sam3d_runtime_unavailable",
            blocker=repr(mhr_decode.MHR_RUNTIME_IMPORT_ERROR),
            samples=[],
            renders=[],
            attempts=[],
            started_at=started_at,
        )
    try:
        decoder = mhr_decode.MHRDecoder(
            checkpoint_path=str(args.checkpoint),
            mhr_path=str(args.mhr_asset),
            device=args.device,
        )
        samples = [_author_mhr_sample(index, decoder=decoder) for index in range(args.samples)]
        estimator = _load_sam3d_estimator(args)
    except Exception as exc:  # noqa: BLE001 - honest optional-runtime boundary.
        return _write_blocked_sam3d_report(
            args,
            status="blocked_sam3d_runtime_unavailable",
            blocker=f"{type(exc).__name__}: {exc}",
            samples=[],
            renders=[],
            attempts=[],
            started_at=started_at,
        )

    renders: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []
    decoded_samples: list[dict[str, list[list[float]]]] = []
    for sample in samples:
        decoded: dict[str, list[list[float]]] | None = None
        for realism_attempt in (1, 2):
            render_path = args.render_dir / f"synthetic_body_{sample['sample_id']:04d}_attempt{realism_attempt}.png"
            _write_shaded_mesh_render(
                render_path,
                sample["authored_vertices_world"],
                sample["mesh_faces"],
                realism_attempt=realism_attempt,
            )
            renders.append(
                {
                    "sample_id": sample["sample_id"],
                    "attempt": realism_attempt,
                    "path": render_path.name,
                    "width": 512,
                    "height": 512,
                }
            )
            try:
                decoded, evidence = _decode_sam3d_render(
                    estimator,
                    render_path=render_path,
                    width=512,
                    height=512,
                )
            except Exception as exc:  # noqa: BLE001 - saved per-attempt evidence.
                decoded = None
                evidence = {"valid_detection": False, "error": f"{type(exc).__name__}: {exc}"}
            attempts.append(
                {
                    "sample_id": sample["sample_id"],
                    "attempt": realism_attempt,
                    "render": render_path.name,
                    **evidence,
                }
            )
            if decoded is not None:
                break
        if decoded is None:
            return _write_blocked_sam3d_report(
                args,
                status="blocked_synthetic_render_not_detectable",
                blocker="SAM-3D-Body produced no valid person record after the two-attempt realism ladder.",
                samples=samples,
                renders=renders,
                attempts=attempts,
                started_at=started_at,
            )
        decoded_samples.append(decoded)

    joint_errors: list[float] = []
    vertex_errors: list[float] = []
    divergences: list[float] = []
    for sample, decoded in zip(samples, decoded_samples, strict=True):
        joint_errors.extend(_point_errors_mm(decoded["joints_world"], sample["authored_joints_world"]))
        vertex_errors.extend(_point_errors_mm(decoded["vertices_world"], sample["authored_vertices_world"]))
        divergences.extend(_nearest_errors_mm(decoded["joints_world"], decoded["vertices_world"]))
    joints_p95 = _percentile(joint_errors, 95)
    vertices_p95 = _percentile(vertex_errors, 95)
    divergence_p95 = _percentile(divergences, 95)
    report = _base_report(args=args, samples=samples, renders=renders, wall_seconds=time.time() - started_at)
    report.update(
        {
            "measurement_status": "measured",
            "blocker": None,
            "attempts": attempts,
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
    report["recipe"].update(
        {
            "author_geometry": "MHRDecoder neutral body mesh and model head faces",
            "render": "512px filled triangles with Lambert shading; attempt 2 adds floor and gradient",
            "gpu_arm": "hmr_deep._direct_setup_sam_3d_body production subprocess fallback path",
        }
    )
    _write_report(args.out, report)
    return report


def _write_blocked_sam3d_report(
    args: argparse.Namespace,
    *,
    status: str,
    blocker: str,
    samples: Sequence[dict[str, Any]],
    renders: Sequence[dict[str, Any]],
    attempts: Sequence[dict[str, Any]],
    started_at: float,
) -> dict[str, Any]:
    report = _base_report(args=args, samples=samples, renders=renders, wall_seconds=time.time() - started_at)
    report.update({"measurement_status": status, "blocker": blocker, "attempts": list(attempts)})
    report["recipe"].update(
        {
            "author_geometry": "MHRDecoder neutral body mesh when the production runtime is available",
            "render": "two-attempt 512px shaded mesh realism ladder",
            "gpu_arm": "hmr_deep._direct_setup_sam_3d_body production subprocess fallback path",
        }
    )
    _write_report(args.out, report)
    return report


def _author_mhr_sample(index: int, *, decoder: Any) -> dict[str, Any]:
    decoded = decoder.decode_euler_frame(
        global_orient_euler=[0.0] * mhr_decode.GLOBAL_ROT_EULER_DIM,
        body_pose_euler=[0.0] * mhr_decode.BODY_POSE_EULER_DIM,
        shape=[0.0] * mhr_decode.SHAPE_DIM,
        scale=[0.0] * mhr_decode.SCALE_DIM,
        hand_pose=None,
    )
    joints = np.asarray(decoded["joints_camera"][0], dtype=np.float64).tolist()
    vertices = np.asarray(decoded["vertices_camera"][0], dtype=np.float64).tolist()
    faces_value = getattr(decoder.head, "faces", None)
    if faces_value is None:
        raise RuntimeError("MHR decoder head does not expose mesh faces")
    if hasattr(faces_value, "detach"):
        faces_value = faces_value.detach().cpu().numpy()
    faces = np.asarray(faces_value, dtype=np.int64).tolist()
    pred_cam_t = [0.15 + index * 0.03, -0.05, 3.5 + index * 0.1]
    z_values = [float(point[2]) for point in vertices]
    return {
        "sample_id": index,
        "scale_m": max(z_values) - min(z_values),
        "pred_cam_t": pred_cam_t,
        "model_joints_without_cam_t": joints,
        "model_vertices_without_cam_t": vertices,
        "authored_joints_world": _author_expected_translated_points(joints, pred_cam_t),
        "authored_vertices_world": _author_expected_translated_points(vertices, pred_cam_t),
        "mesh_faces": faces,
    }


def _load_sam3d_estimator(args: argparse.Namespace) -> Any:
    from threed.racketsport import hmr_deep

    return hmr_deep._direct_setup_sam_3d_body(
        detector_name="",
        fov_name="",
        device=args.device or "cuda",
        local_checkpoint_path=str(args.checkpoint.parent),
        local_mhr_path=str(args.mhr_asset),
    )


def _decode_sam3d_render(
    estimator: Any,
    *,
    render_path: Path,
    width: int,
    height: int,
) -> tuple[dict[str, list[list[float]]] | None, dict[str, Any]]:
    from threed.racketsport import hmr_deep

    focal = 0.9 * max(width, height)
    intrinsics = [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]]
    raw_output = estimator.process_one_image(
        str(render_path.resolve()),
        bboxes=np.asarray([[2.0, 2.0, width - 2.0, height - 2.0]], dtype=np.float32),
        masks=None,
        cam_int=hmr_deep._camera_intrinsics_tensor(intrinsics),
        use_mask=False,
        hand_box_source="body_decoder",
    )
    records = hmr_deep.extract_fast_sam_person_records(raw_output)
    evidence: dict[str, Any] = {"record_count": len(records), "valid_detection": False}
    for record in records:
        joints = record.get("pred_keypoints_3d")
        vertices = record.get("pred_vertices")
        pred_cam_t = record.get("pred_cam_t")
        if joints is None or vertices is None or pred_cam_t is None:
            continue
        joints_list = np.asarray(joints, dtype=np.float64).tolist()
        vertices_list = np.asarray(vertices, dtype=np.float64).tolist()
        if not joints_list or not vertices_list:
            continue
        evidence.update(
            {
                "valid_detection": True,
                "joint_count": len(joints_list),
                "vertex_count": len(vertices_list),
                "pred_cam_t": np.asarray(pred_cam_t, dtype=np.float64).tolist(),
            }
        )
        return (
            {
                "joints_world": mhr_decode.apply_pred_cam_t_once(joints_list, pred_cam_t=pred_cam_t),
                "vertices_world": mhr_decode.apply_pred_cam_t_once(vertices_list, pred_cam_t=pred_cam_t),
            },
            evidence,
        )
    return None, evidence


def _write_shaded_mesh_render(
    path: Path,
    vertices: Sequence[Sequence[float]],
    faces: Sequence[Sequence[int]],
    *,
    realism_attempt: int,
) -> None:
    if realism_attempt not in (1, 2):
        raise ValueError("realism_attempt must be 1 or 2")
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt
    from matplotlib.collections import PolyCollection

    verts = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(faces, dtype=np.int64)
    if verts.ndim != 2 or verts.shape[1] != 3 or triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("MHR render requires vertices Nx3 and faces Mx3")
    focal = 0.9 * 512.0
    safe_z = np.maximum(verts[:, 2], 0.1)
    projected = np.column_stack(
        [256.0 + focal * verts[:, 0] / safe_z, 256.0 + focal * verts[:, 1] / safe_z]
    )
    tri3 = verts[triangles]
    normals = np.cross(tri3[:, 1] - tri3[:, 0], tri3[:, 2] - tri3[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1)[:, None], 1e-12)
    light = np.asarray([-0.25, -0.35, -1.0], dtype=np.float64)
    light /= np.linalg.norm(light)
    lambert = 0.25 + 0.75 * np.abs(normals @ light)
    colors = np.clip(lambert[:, None] * np.asarray([0.20, 0.48, 0.72])[None, :], 0.0, 1.0)
    depth_order = np.argsort(np.mean(tri3[:, :, 2], axis=1))[::-1]
    fig = plt.figure(figsize=(5.12, 5.12), dpi=100)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    if realism_attempt == 2:
        gradient = np.linspace(0.96, 0.72, 512)[:, None]
        base = gradient * np.ones((1, 512))
        background = np.dstack([base, base, np.minimum(1.0, base + 0.04)])
        ax.imshow(background, extent=(0, 512, 512, 0), interpolation="nearest")
        ax.fill([0, 512, 512, 0], [420, 420, 512, 512], color=(0.63, 0.67, 0.58), zorder=1)
    else:
        ax.set_facecolor((0.94, 0.95, 0.97))
    ax.add_collection(
        PolyCollection(
            projected[triangles[depth_order]],
            facecolors=colors[depth_order],
            edgecolors="none",
            antialiased=False,
            zorder=2,
        )
    )
    ax.set_xlim(0, 512)
    ax.set_ylim(512, 0)
    ax.axis("off")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100, facecolor=ax.get_facecolor())
    plt.close(fig)


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

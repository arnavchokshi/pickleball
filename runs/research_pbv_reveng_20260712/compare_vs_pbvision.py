#!/usr/bin/env python3
"""Deterministic no-GT scorecard for DinkVision vs a pb.vision cv export.

This is a competitor-reference diagnostic.  It does not treat pb.vision as
ground truth, does not read protected labels, and does not produce promotion
evidence.  The script accepts either a run directory or a ball/world JSON.

Examples:
  .venv/bin/python runs/lanes/pbv_reveng_20260712/compare_vs_pbvision.py \
    --pb-export runs/research_ball3d_20260709/pbvision_cv_export/cv_export.json \
    --ours runs/lanes/demo_beststack_render_20260710/after_wolv/ball_track_arc_solved.json \
    --output runs/lanes/pbv_reveng_20260712/scorecard_raw_arc.json

  .venv/bin/python runs/lanes/pbv_reveng_20260712/compare_vs_pbvision.py \
    --pb-export path/to/future/cv_export.json --ours path/to/our/run_dir \
    --frame-offset auto --output /tmp/pbvision_scorecard.json
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from scipy.stats import spearmanr


FT_TO_M = 0.3048
PB_COURT_WIDTH_FT = 20.0
PB_COURT_LENGTH_FT = 44.0
DEFAULT_BALL_RADIUS_M = 0.0371
GRAVITY_MPS2 = 9.80665
PHYSICS_RMSE_LIMIT_M = 0.15
PHYSICS_P95_LIMIT_M = 0.30
MAX_PLAUSIBLE_SPEED_MPS = 35.0
ACTION_KINDS = ("ball", "bounce", "net", "shot")


@dataclass(frozen=True)
class Sample:
    frame: int
    t: float
    xyz_m: np.ndarray | None
    xy_px: np.ndarray | None
    confidence: float | None
    visible: bool
    interpolated: bool
    kind: str
    segment: int | None = None
    status: str | None = None
    band: str | None = None


@dataclass
class Trajectory:
    name: str
    fps: float
    samples: dict[int, Sample]
    two_d_samples: dict[int, Sample]
    source_path: Path
    calibration: dict[str, Any] | None = None
    arc: dict[str, Any] | None = None


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def finite_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def finite_vec(value: Any, size: int) -> np.ndarray | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) < size:
        return None
    parsed = [finite_float(value[index]) for index in range(size)]
    if any(item is None for item in parsed):
        return None
    return np.asarray(parsed, dtype=float)


def position_mapping(value: Any) -> np.ndarray | None:
    if not isinstance(value, Mapping):
        return None
    parsed = [finite_float(value.get(axis)) for axis in ("x", "y", "z")]
    if any(item is None for item in parsed):
        return None
    return np.asarray(parsed, dtype=float)


def pb_feet_to_ours_m(xyz_ft: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            (xyz_ft[0] - PB_COURT_WIDTH_FT / 2.0) * FT_TO_M,
            (PB_COURT_LENGTH_FT / 2.0 - xyz_ft[1]) * FT_TO_M,
            xyz_ft[2] * FT_TO_M,
        ],
        dtype=float,
    )


def ours_m_to_pb_feet(xyz_m: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            xyz_m[0] / FT_TO_M + PB_COURT_WIDTH_FT / 2.0,
            PB_COURT_LENGTH_FT / 2.0 - xyz_m[1] / FT_TO_M,
            xyz_m[2] / FT_TO_M,
        ],
        dtype=float,
    )


def percentile(values: Iterable[float], q: float) -> float | None:
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        return None
    return float(np.quantile(array, q))


def summary_stats(values: Iterable[float]) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        return {"count": 0, "mean": None, "rmse": None, "p50": None, "p95": None, "max": None}
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "rmse": float(np.sqrt(np.mean(array * array))),
        "p50": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
    }


def contiguous_runs(frames: Iterable[int]) -> list[dict[str, int]]:
    ordered = sorted(set(int(frame) for frame in frames))
    if not ordered:
        return []
    runs: list[dict[str, int]] = []
    start = previous = ordered[0]
    for frame in ordered[1:]:
        if frame != previous + 1:
            runs.append({"start": start, "end": previous, "length": previous - start + 1})
            start = frame
        previous = frame
    runs.append({"start": start, "end": previous, "length": previous - start + 1})
    return runs


def parse_pb_export(path: Path, width: int, height: int) -> tuple[Trajectory, dict[str, Any]]:
    payload = read_json(path)
    camera = payload.get("camera")
    sessions = payload.get("sessions")
    if not isinstance(camera, Mapping) or not isinstance(sessions, list) or not sessions:
        raise ValueError(f"{path}: missing camera/sessions")
    fps = finite_float(camera.get("fps")) or 30.0
    rallies = sessions[0].get("rallies") if isinstance(sessions[0], Mapping) else None
    if not isinstance(rallies, list) or not rallies:
        raise ValueError(f"{path}: missing sessions[0].rallies")

    samples: dict[int, Sample] = {}
    alignment: dict[int, Sample] = {}
    frame_records: dict[int, dict[str, Any]] = {}
    rally_ranges: list[dict[str, int]] = []
    boundary_frames: set[int] = set()
    segment_counter = 0
    current_segment: int | None = None
    for rally in rallies:
        if not isinstance(rally, Mapping) or not isinstance(rally.get("frames"), list):
            continue
        start = int(rally.get("frame_index") or 0)
        frames = rally["frames"]
        rally_ranges.append({"start": start, "end": start + len(frames) - 1, "count": len(frames)})
        for local_frame, raw in enumerate(frames):
            if not isinstance(raw, Mapping):
                continue
            frame = start + local_frame
            frame_records[frame] = dict(raw)
            actions = raw.get("actions") if isinstance(raw.get("actions"), Mapping) else {}
            ball_action = actions.get("ball") if isinstance(actions.get("ball"), Mapping) else {}
            ball_xy = np.asarray(
                [
                    (finite_float(ball_action.get("u")) or 0.0) * width,
                    (finite_float(ball_action.get("v")) or 0.0) * height,
                ],
                dtype=float,
            )
            ball_conf = finite_float(ball_action.get("confidence"))
            alignment[frame] = Sample(
                frame=frame,
                t=frame / fps,
                xyz_m=None,
                xy_px=ball_xy,
                confidence=ball_conf,
                visible=ball_conf is not None and ball_conf >= 0.5,
                interpolated=False,
                kind="ball_action",
            )

            balls = raw.get("balls") if isinstance(raw.get("balls"), Mapping) else {}
            selected = balls.get("selected") if isinstance(balls.get("selected"), str) else None
            selected_record = balls.get(selected) if selected and isinstance(balls.get(selected), Mapping) else None
            if selected_record is None:
                continue
            xyz_ft = position_mapping(selected_record.get("court_position"))
            selected_action = actions.get(selected) if isinstance(actions.get(selected), Mapping) else {}
            u = finite_float(selected_action.get("u"))
            v = finite_float(selected_action.get("v"))
            xy_px = None if u is None or v is None else np.asarray([u * width, v * height], dtype=float)
            if selected in {"shot", "bounce", "net"}:
                boundary_frames.add(frame)
                if current_segment is None:
                    current_segment = segment_counter
                elif frame != min(boundary_frames):
                    segment_counter += 1
                    current_segment = segment_counter
            elif current_segment is None:
                current_segment = segment_counter
            samples[frame] = Sample(
                frame=frame,
                t=frame / fps,
                xyz_m=None if xyz_ft is None else pb_feet_to_ours_m(xyz_ft),
                xy_px=xy_px,
                confidence=finite_float(selected_action.get("confidence")),
                visible=True,
                interpolated=bool(selected_record.get("interpolated", False)),
                kind=selected,
                segment=current_segment,
                status="interpolated" if selected_record.get("interpolated", False) else "emitted",
            )

    # Re-assign free-flight segment ids using consecutive special-event boundaries.
    ordered_boundaries = sorted(boundary_frames)
    for frame, sample in list(samples.items()):
        segment = 0
        for boundary in ordered_boundaries[1:]:
            if frame >= boundary:
                segment += 1
        samples[frame] = Sample(**{**sample.__dict__, "segment": segment})

    trajectory = Trajectory(
        name="pbvision_selected_3d",
        fps=fps,
        samples=samples,
        two_d_samples=alignment,
        source_path=path,
    )
    context = {
        "payload": payload,
        "frame_records": frame_records,
        "rally_ranges": rally_ranges,
        "boundary_frames": ordered_boundaries,
        "width": width,
        "height": height,
    }
    return trajectory, context


def resolve_ours_path(path: Path) -> Path:
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(path)
    for name in (
        "confidence_gated_world.json",
        "virtual_world.json",
        "ball_track_arc_solved.json",
        "ball_track_physics_filled.json",
        "ball_track.json",
    ):
        candidate = path / name
        if candidate.is_file():
            return candidate
    raise ValueError(f"{path}: no supported ball/world JSON found")


def companion(path: Path, explicit: Path | None, name: str) -> Path | None:
    if explicit is not None:
        return explicit
    candidate = path.parent / name
    return candidate if candidate.is_file() else None


def extract_frames(payload: Mapping[str, Any]) -> list[Any]:
    frames = payload.get("frames")
    if isinstance(frames, list):
        return frames
    ball = payload.get("ball")
    if isinstance(ball, Mapping) and isinstance(ball.get("frames"), list):
        return ball["frames"]
    raise ValueError("supported JSON must contain frames[] or ball.frames[]")


def parse_ours(
    input_path: Path,
    *,
    ball_track_path: Path | None,
    calibration_path: Path | None,
    fps_override: float | None,
) -> Trajectory:
    path = resolve_ours_path(input_path)
    payload = read_json(path)
    frames = extract_frames(payload)
    track_path = companion(path, ball_track_path, "ball_track.json")
    cal_path = companion(path, calibration_path, "court_calibration.json")
    arc_path = path if path.name == "ball_track_arc_solved.json" else companion(path, None, "ball_track_arc_solved.json")
    arc = read_json(arc_path) if arc_path is not None else None
    track_payload = read_json(track_path) if track_path is not None else None
    calibration = read_json(cal_path) if cal_path is not None else None
    fps = (
        fps_override
        or finite_float(payload.get("fps"))
        or (finite_float(track_payload.get("fps")) if track_payload else None)
        or 30.0
    )

    arc_frames = extract_frames(arc) if arc is not None else []
    samples: dict[int, Sample] = {}
    for index, raw in enumerate(frames):
        if not isinstance(raw, Mapping):
            continue
        frame = int(raw.get("frame", raw.get("frame_index", index)))
        t = finite_float(raw.get("t"))
        xy = finite_vec(raw.get("xy"), 2)
        xyz = finite_vec(raw.get("world_xyz"), 3)
        arc_raw = arc_frames[frame] if 0 <= frame < len(arc_frames) and isinstance(arc_frames[frame], Mapping) else {}
        solver = raw.get("arc_solver") if isinstance(raw.get("arc_solver"), Mapping) else {}
        if not solver and isinstance(arc_raw.get("arc_solver"), Mapping):
            solver = arc_raw["arc_solver"]
        # Arc outputs can rescue a non-primary top-K candidate.  Score the 3D
        # against that chosen observation, not blindly against the stale
        # primary-track xy retained for provenance.
        chosen = raw.get("candidate_selection") if isinstance(raw.get("candidate_selection"), Mapping) else None
        if chosen is None and isinstance(arc_raw.get("candidate_selection"), Mapping):
            chosen = arc_raw["candidate_selection"]
        chosen_xy = finite_vec(chosen.get("xy"), 2) if chosen is not None else None
        if chosen_xy is not None:
            xy = chosen_xy
        segment = solver.get("segment_id") if isinstance(solver.get("segment_id"), int) else None
        status = solver.get("segment_status") if isinstance(solver.get("segment_status"), str) else None
        provenance = raw.get("confidence_provenance") if isinstance(raw.get("confidence_provenance"), Mapping) else {}
        band = provenance.get("display_band") if isinstance(provenance.get("display_band"), str) else raw.get("band")
        samples[frame] = Sample(
            frame=frame,
            t=t if t is not None else frame / fps,
            xyz_m=xyz,
            xy_px=xy,
            confidence=finite_float(raw.get("conf", raw.get("confidence"))),
            visible=bool(raw.get("visible", xyz is not None or xy is not None)),
            interpolated=bool(raw.get("approx", False)),
            kind="ours",
            segment=segment,
            status=status,
            band=band if isinstance(band, str) else None,
        )

    two_d: dict[int, Sample] = {}
    track_frames = extract_frames(track_payload) if track_payload is not None else frames
    for index, raw in enumerate(track_frames):
        if not isinstance(raw, Mapping):
            continue
        frame = int(raw.get("frame", raw.get("frame_index", index)))
        xy = finite_vec(raw.get("xy"), 2)
        confidence = finite_float(raw.get("conf", raw.get("confidence")))
        two_d[frame] = Sample(
            frame=frame,
            t=finite_float(raw.get("t")) or frame / fps,
            xyz_m=None,
            xy_px=xy,
            confidence=confidence,
            visible=bool(raw.get("visible", xy is not None)) and xy is not None,
            interpolated=bool(raw.get("approx", False)),
            kind="ours_2d",
        )
    return Trajectory(
        name=path.name,
        fps=fps,
        samples=samples,
        two_d_samples=two_d,
        source_path=path,
        calibration=calibration,
        arc=arc,
    )


def standardized_correlation(pairs: list[tuple[np.ndarray, np.ndarray]]) -> float | None:
    if len(pairs) < 6:
        return None
    first = np.stack([pair[0] for pair in pairs])
    second = np.stack([pair[1] for pair in pairs])
    if np.any(first.std(axis=0) <= 1e-12) or np.any(second.std(axis=0) <= 1e-12):
        return None
    first = (first - first.mean(axis=0)) / first.std(axis=0)
    second = (second - second.mean(axis=0)) / second.std(axis=0)
    return float(np.corrcoef(first.ravel(), second.ravel())[0, 1])


def align_trajectories(pb: Trajectory, ours: Trajectory, max_lag: int) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for lag in range(-max_lag, max_lag + 1):
        pairs: list[tuple[np.ndarray, np.ndarray]] = []
        errors: list[float] = []
        for pb_frame, pb_sample in pb.two_d_samples.items():
            ours_sample = ours.two_d_samples.get(pb_frame + lag)
            if (
                not pb_sample.visible
                or pb_sample.xy_px is None
                or ours_sample is None
                or not ours_sample.visible
                or ours_sample.xy_px is None
            ):
                continue
            pairs.append((pb_sample.xy_px, ours_sample.xy_px))
            errors.append(float(np.linalg.norm(pb_sample.xy_px - ours_sample.xy_px)))
        correlation = standardized_correlation(pairs)
        if correlation is not None:
            rows.append(
                {
                    "lag": lag,
                    "correlation": correlation,
                    "paired_count": len(pairs),
                    "pixel_error_p50": percentile(errors, 0.5),
                }
            )
    if not rows:
        raise ValueError("auto alignment failed: fewer than six paired visible 2D samples")
    rows.sort(key=lambda row: (-row["correlation"], -row["paired_count"], abs(row["lag"]), row["lag"]))
    return {
        "selected_offset": int(rows[0]["lag"]),
        "mapping": "ours_frame = pb_global_frame + selected_offset",
        "best": rows[0],
        "runner_up": rows[1] if len(rows) > 1 else None,
        "top_candidates": rows[:7],
    }


def project_pb_points(
    xyz_ft: np.ndarray, camera_segment: Mapping[str, Any], width: int, height: int
) -> tuple[np.ndarray, np.ndarray]:
    position_map = camera_segment.get("position")
    orientation = camera_segment.get("orientation")
    if not isinstance(position_map, Mapping) or not isinstance(orientation, Mapping):
        raise ValueError("pb camera segment lacks position/orientation")
    position = np.asarray([position_map[axis] for axis in ("x", "y", "z")], dtype=float)
    rotation = Rotation.from_euler(
        "zyx",
        [-float(orientation["yaw"]), float(orientation["pitch"]), -float(orientation["roll"])],
    ).as_matrix()
    vectors = (rotation @ (xyz_ft - position).T).T
    camera_xyz = np.column_stack([vectors[:, 1], -vectors[:, 2], vectors[:, 0]])
    focal = width / (2.0 * math.tan(float(camera_segment["fov"]) / 2.0))
    projected = np.column_stack(
        [
            focal * camera_xyz[:, 0] / camera_xyz[:, 2] + width / 2.0,
            focal * camera_xyz[:, 1] / camera_xyz[:, 2] + height / 2.0,
        ]
    )
    return projected, camera_xyz


def project_ours_points(xyz_m: np.ndarray, calibration: Mapping[str, Any]) -> np.ndarray:
    extrinsics = calibration.get("extrinsics")
    intrinsics = calibration.get("intrinsics")
    if not isinstance(extrinsics, Mapping) or not isinstance(intrinsics, Mapping):
        raise ValueError("our calibration lacks intrinsics/extrinsics")
    rotation = np.asarray(extrinsics["R"], dtype=float)
    translation = np.asarray(extrinsics["t"], dtype=float)
    camera_xyz = (rotation @ xyz_m.T + translation[:, None]).T
    fx = float(intrinsics["fx"])
    fy = float(intrinsics.get("fy", fx))
    cx = float(intrinsics.get("cx", calibration.get("image_size", [1920, 1080])[0] / 2.0))
    cy = float(intrinsics.get("cy", calibration.get("image_size", [1920, 1080])[1] / 2.0))
    return np.column_stack(
        [fx * camera_xyz[:, 0] / camera_xyz[:, 2] + cx, fy * camera_xyz[:, 1] / camera_xyz[:, 2] + cy]
    )


def rk4_positions(
    p0: np.ndarray,
    v0: np.ndarray,
    times: np.ndarray,
    drag_k: float,
    *,
    max_step: float = 1.0 / 240.0,
) -> np.ndarray:
    state = np.concatenate([p0.astype(float), v0.astype(float)])
    output = [state[:3].copy()]
    current = float(times[0])

    def derivative(value: np.ndarray) -> np.ndarray:
        velocity = value[3:]
        acceleration = np.asarray([0.0, 0.0, -GRAVITY_MPS2]) - drag_k * np.linalg.norm(velocity) * velocity
        return np.concatenate([velocity, acceleration])

    for target in times[1:]:
        remaining = float(target) - current
        while remaining > 1e-12:
            step = min(max_step, remaining)
            k1 = derivative(state)
            k2 = derivative(state + 0.5 * step * k1)
            k3 = derivative(state + 0.5 * step * k2)
            k4 = derivative(state + step * k3)
            state = state + step * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
            current += step
            remaining = float(target) - current
        output.append(state[:3].copy())
    return np.stack(output)


def group_segments(trajectory: Trajectory) -> list[list[Sample]]:
    emitted = [sample for sample in trajectory.samples.values() if sample.xyz_m is not None]
    emitted.sort(key=lambda sample: sample.frame)
    groups: list[list[Sample]] = []
    current: list[Sample] = []
    current_segment: int | None = None
    for sample in emitted:
        if current and (
            (sample.segment is not None and current_segment is not None and sample.segment != current_segment)
            or (sample.segment is None and sample.frame > current[-1].frame + 1)
        ):
            groups.append(current)
            current = []
        current.append(sample)
        current_segment = sample.segment
    if current:
        groups.append(current)
    return groups


def physics_fit(samples: Sequence[Sample], segment_index: int) -> dict[str, Any]:
    if len(samples) < 4:
        return {"segment": segment_index, "status": "insufficient_samples", "sample_count": len(samples)}
    times = np.asarray([sample.t for sample in samples], dtype=float)
    points = np.stack([sample.xyz_m for sample in samples if sample.xyz_m is not None])
    order = np.argsort(times)
    times = times[order]
    points = points[order]
    unique = np.concatenate([[True], np.diff(times) > 1e-9])
    times = times[unique]
    points = points[unique]
    if len(times) < 4 or times[-1] - times[0] < 0.1:
        return {"segment": segment_index, "status": "insufficient_duration", "sample_count": len(times)}
    relative_times = times - times[0]
    velocity_seed = (points[min(2, len(points) - 1)] - points[0]) / relative_times[min(2, len(points) - 1)]

    def residual(parameters: np.ndarray) -> np.ndarray:
        predicted = rk4_positions(points[0], parameters[:3], relative_times, float(parameters[3]))
        return (predicted - points).ravel()

    fitted = least_squares(
        residual,
        np.concatenate([velocity_seed, [0.01]]),
        bounds=(np.asarray([-60.0, -60.0, -60.0, 0.0]), np.asarray([60.0, 60.0, 60.0, 1.0])),
        loss="huber",
        f_scale=0.05,
        max_nfev=800,
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
    )
    predicted = rk4_positions(points[0], fitted.x[:3], relative_times, float(fitted.x[3]))
    errors = np.linalg.norm(predicted - points, axis=1)
    time_deltas = np.diff(times)
    speeds = np.linalg.norm(np.diff(points, axis=0) / time_deltas[:, None], axis=1)
    rmse = float(np.sqrt(np.mean(errors * errors)))
    p95 = float(np.quantile(errors, 0.95))
    plausible = (
        rmse <= PHYSICS_RMSE_LIMIT_M
        and p95 <= PHYSICS_P95_LIMIT_M
        and float(speeds.max(initial=0.0)) <= MAX_PLAUSIBLE_SPEED_MPS
        and float(points[:, 2].min()) >= -0.05
    )
    residual_signal = predicted - points
    high_energy = total_energy = 0.0
    if len(residual_signal) >= 8:
        for axis in range(3):
            signal = residual_signal[:, axis] - residual_signal[:, axis].mean()
            spectrum = np.fft.rfft(signal)
            frequencies = np.fft.rfftfreq(len(signal), d=float(np.median(time_deltas)))
            energy = np.abs(spectrum) ** 2
            total_energy += float(energy[1:].sum())
            high_energy += float(energy[frequencies >= 5.0].sum())
    return {
        "segment": segment_index,
        "source_segment": samples[0].segment,
        "frame_start": int(samples[0].frame),
        "frame_end": int(samples[-1].frame),
        "sample_count": int(len(times)),
        "duration_s": float(relative_times[-1]),
        "status": "fit",
        "drag_k_per_m": float(fitted.x[3]),
        "initial_speed_mps": float(np.linalg.norm(fitted.x[:3])),
        "observed_speed_p95_mps": percentile(speeds, 0.95),
        "observed_speed_max_mps": float(speeds.max(initial=0.0)),
        "reintegration_rmse_m": rmse,
        "reintegration_p95_m": p95,
        "reintegration_max_m": float(errors.max()),
        "high_frequency_residual_energy_fraction_ge_5hz": None if total_energy <= 1e-18 else high_energy / total_energy,
        "physics_plausible": bool(plausible),
        "plausibility_rule": (
            f"rmse<={PHYSICS_RMSE_LIMIT_M}m,p95<={PHYSICS_P95_LIMIT_M}m,"
            f"speed<={MAX_PLAUSIBLE_SPEED_MPS}m/s,z>=-0.05m"
        ),
    }


def physics_pillar(trajectory: Trajectory) -> dict[str, Any]:
    rows = [physics_fit(group, index) for index, group in enumerate(group_segments(trajectory))]
    fitted = [row for row in rows if row.get("status") == "fit"]
    plausible = [row for row in fitted if row.get("physics_plausible")]
    return {
        "segment_count": len(rows),
        "fit_segment_count": len(fitted),
        "physics_plausible_segment_count": len(plausible),
        "physics_plausible_fraction": None if not fitted else len(plausible) / len(fitted),
        "reintegration_rmse_m": summary_stats(row["reintegration_rmse_m"] for row in fitted),
        "reintegration_p95_m": summary_stats(row["reintegration_p95_m"] for row in fitted),
        "segments": rows,
    }


def camera_segment(context: Mapping[str, Any]) -> Mapping[str, Any]:
    payload = context["payload"]
    segments = payload["camera"].get("cameraSegments")
    if not isinstance(segments, list) or not segments or not isinstance(segments[0], Mapping):
        raise ValueError("pb export has no camera segment")
    return segments[0]


def pb_reprojection_pillar(pb: Trajectory, context: Mapping[str, Any]) -> dict[str, Any]:
    width, height = int(context["width"]), int(context["height"])
    segment = camera_segment(context)
    samples = [sample for sample in pb.samples.values() if sample.xyz_m is not None and sample.xy_px is not None]
    xyz_ft = np.stack([ours_m_to_pb_feet(sample.xyz_m) for sample in samples])
    projected, _ = project_pb_points(xyz_ft, segment, width, height)
    observed = np.stack([sample.xy_px for sample in samples])
    errors = np.linalg.norm(projected - observed, axis=1)
    by_kind: dict[str, Any] = {}
    for kind in ACTION_KINDS:
        indices = [index for index, sample in enumerate(samples) if sample.kind == kind]
        by_kind[kind] = summary_stats(errors[indices]) if indices else summary_stats([])
    high_confidence_indices = [
        index for index, sample in enumerate(samples) if sample.confidence is not None and sample.confidence >= 0.5
    ]
    low_confidence_indices = [
        index for index, sample in enumerate(samples) if sample.confidence is None or sample.confidence < 0.5
    ]

    return {
        "image_size_assumption": [width, height],
        "camera_convention": "inferred zyx(-yaw,pitch,-roll), then [y,-z,x] camera axes",
        "selected_3d_to_selected_2d_px": summary_stats(errors),
        "selected_action_confidence_ge_0_5_px": summary_stats(errors[high_confidence_indices]),
        "selected_action_confidence_lt_0_5_px": summary_stats(errors[low_confidence_indices]),
        "by_selected_kind_px": by_kind,
        "court_point_self_reprojection_px": None,
        "court_point_limit": "court_points lack exported semantic identities; matched-clip calibration is required to assign their world coordinates without circular guessing",
    }


def ours_reprojection_pillar(ours: Trajectory) -> dict[str, Any] | None:
    if ours.calibration is None:
        return None
    samples = [sample for sample in ours.samples.values() if sample.xyz_m is not None and sample.xy_px is not None]
    if not samples:
        return None
    xyz = np.stack([sample.xyz_m for sample in samples])
    projected = project_ours_points(xyz, ours.calibration)
    observed = np.stack([sample.xy_px for sample in samples])
    errors = np.linalg.norm(projected - observed, axis=1)
    by_status: dict[str, Any] = {}
    for status in sorted(set(sample.status or "unknown" for sample in samples)):
        indices = [index for index, sample in enumerate(samples) if (sample.status or "unknown") == status]
        by_status[status] = summary_stats(errors[indices])
    return {
        "our_3d_to_our_2d_px": summary_stats(errors),
        "by_segment_status_px": by_status,
    }


def local_maxima(values: Mapping[int, float], floor: float) -> list[int]:
    frames = sorted(values)
    result: list[int] = []
    for frame in frames:
        value = values[frame]
        if value < floor:
            continue
        previous = values.get(frame - 1, -math.inf)
        following = values.get(frame + 1, -math.inf)
        if value >= previous and value > following:
            result.append(frame)
    return result


def pb_bounce_pillar(pb: Trajectory, context: Mapping[str, Any]) -> dict[str, Any]:
    selected = [sample for sample in pb.samples.values() if sample.kind == "bounce" and sample.xyz_m is not None]
    inferred_radius = float(np.median([sample.xyz_m[2] for sample in selected])) if selected else DEFAULT_BALL_RADIUS_M
    selected_rows = [
        {
            "frame": sample.frame,
            "confidence": sample.confidence,
            "z_m": float(sample.xyz_m[2]),
            "abs_z_minus_inferred_radius_m": abs(float(sample.xyz_m[2]) - inferred_radius),
        }
        for sample in selected
    ]
    bounce_confidence: dict[int, float] = {}
    for frame, raw in context["frame_records"].items():
        actions = raw.get("actions") if isinstance(raw.get("actions"), Mapping) else {}
        bounce = actions.get("bounce") if isinstance(actions.get("bounce"), Mapping) else {}
        confidence = finite_float(bounce.get("confidence"))
        if confidence is not None:
            bounce_confidence[frame] = confidence
    peaks = local_maxima(bounce_confidence, 0.5)
    peak_rows: list[dict[str, Any]] = []
    for frame in peaks:
        nearest = min(
            (sample for sample in pb.samples.values() if sample.xyz_m is not None),
            key=lambda sample: abs(sample.frame - frame),
        )
        peak_rows.append(
            {
                "peak_frame": frame,
                "confidence": bounce_confidence[frame],
                "nearest_3d_frame": nearest.frame,
                "frame_delta": nearest.frame - frame,
                "nearest_z_m": float(nearest.xyz_m[2]),
                "abs_z_minus_inferred_radius_m": abs(float(nearest.xyz_m[2]) - inferred_radius),
            }
        )
    return {
        "inferred_ball_radius_m_from_selected_bounce_z": inferred_radius,
        "selected_bounces": selected_rows,
        "selected_bounce_z_error_m": summary_stats(row["abs_z_minus_inferred_radius_m"] for row in selected_rows),
        "bounce_confidence_local_peaks_ge_0_5": peak_rows,
        "peak_nearest_z_error_m": summary_stats(row["abs_z_minus_inferred_radius_m"] for row in peak_rows),
    }


def radius_depth_forensics(pb: Trajectory, context: Mapping[str, Any]) -> dict[str, Any]:
    segment = camera_segment(context)
    position_map = segment["position"]
    camera_position = np.asarray([position_map[axis] for axis in ("x", "y", "z")], dtype=float)
    radius_values: list[float] = []
    inverse_distances: list[float] = []
    inverse_depths: list[float] = []
    center_errors: list[float] = []
    frames: list[int] = []
    for frame, sample in sorted(pb.samples.items()):
        if sample.xyz_m is None:
            continue
        raw = context["frame_records"][frame]
        actions = raw.get("actions") if isinstance(raw.get("actions"), Mapping) else {}
        radius = actions.get("ball_radius") if isinstance(actions.get("ball_radius"), Mapping) else None
        if radius is None:
            continue
        radius_v = finite_float(radius.get("radius_v"))
        u = finite_float(radius.get("u"))
        v = finite_float(radius.get("v"))
        if radius_v is None or radius_v <= 0.0:
            continue
        xyz_ft = ours_m_to_pb_feet(sample.xyz_m)
        _, camera_xyz = project_pb_points(xyz_ft[None, :], segment, int(context["width"]), int(context["height"]))
        distance = float(np.linalg.norm(xyz_ft - camera_position))
        depth = float(camera_xyz[0, 2])
        if distance <= 0.0 or depth <= 0.0:
            continue
        frames.append(frame)
        radius_values.append(radius_v)
        inverse_distances.append(1.0 / distance)
        inverse_depths.append(1.0 / depth)
        if sample.xy_px is not None and u is not None and v is not None:
            center = np.asarray([u * context["width"], v * context["height"]])
            center_errors.append(float(np.linalg.norm(center - sample.xy_px)))

    radius_array = np.asarray(radius_values)
    inverse_distance = np.asarray(inverse_distances)
    inverse_depth = np.asarray(inverse_depths)

    def relationship(cue: np.ndarray) -> dict[str, Any]:
        if len(cue) < 3:
            return {"count": len(cue), "pearson_r": None, "spearman_r": None, "linear_r2": None}
        pearson = float(np.corrcoef(radius_array, cue)[0, 1])
        spear = float(spearmanr(radius_array, cue).statistic)
        design = np.column_stack([cue, np.ones_like(cue)])
        coefficients, _, _, _ = np.linalg.lstsq(design, radius_array, rcond=None)
        predicted = design @ coefficients
        residual = radius_array - predicted
        denominator = float(np.sum((radius_array - radius_array.mean()) ** 2))
        r2 = None if denominator <= 1e-18 else 1.0 - float(np.sum(residual * residual)) / denominator
        return {
            "count": len(cue),
            "pearson_r": pearson,
            "spearman_r": spear,
            "linear_r2": r2,
            "slope": float(coefficients[0]),
            "intercept": float(coefficients[1]),
            "relative_residual_p50": percentile(np.abs(residual) / radius_array, 0.50),
            "relative_residual_p95": percentile(np.abs(residual) / radius_array, 0.95),
        }

    return {
        "paired_3d_radius_count": len(frames),
        "radius_v_vs_inverse_euclidean_camera_distance_ft": relationship(inverse_distance),
        "radius_v_vs_inverse_camera_depth_ft": relationship(inverse_depth),
        "ball_radius_center_vs_selected_2d_px": summary_stats(center_errors),
        "causal_limit": "correlation is evidence of a usable depth cue, not proof it entered pb.vision's lift",
    }


def selection_forensics(pb: Trajectory, context: Mapping[str, Any]) -> dict[str, Any]:
    selected_match = 0
    selected_confidences: list[float] = []
    missing_max_confidences: list[float] = []
    selected_counts: Counter[str] = Counter()
    radius_count = 0
    for frame, raw in sorted(context["frame_records"].items()):
        actions = raw.get("actions") if isinstance(raw.get("actions"), Mapping) else {}
        balls = raw.get("balls") if isinstance(raw.get("balls"), Mapping) else {}
        selected = balls.get("selected") if isinstance(balls.get("selected"), str) else None
        confidences = {
            kind: finite_float(actions.get(kind, {}).get("confidence"))
            for kind in ACTION_KINDS
            if isinstance(actions.get(kind), Mapping)
        }
        confidences = {kind: value for kind, value in confidences.items() if value is not None}
        if isinstance(actions.get("ball_radius"), Mapping):
            radius_count += 1
        if selected is None:
            if confidences:
                missing_max_confidences.append(max(confidences.values()))
            continue
        selected_counts[selected] += 1
        if selected in confidences:
            selected_confidences.append(confidences[selected])
            argmax = max(confidences, key=lambda kind: confidences[kind])
            if argmax == selected:
                selected_match += 1
    emitted = sorted(pb.samples)
    rally_frames = sorted(context["frame_records"])
    missing = sorted(set(rally_frames) - set(emitted))
    interpolated = sorted(sample.frame for sample in pb.samples.values() if sample.interpolated)
    return {
        "selected_counts": dict(sorted(selected_counts.items())),
        "selected_total": sum(selected_counts.values()),
        "selected_is_action_argmax_count": selected_match,
        "selected_is_action_argmax_fraction": None if not selected_confidences else selected_match / len(selected_confidences),
        "selected_action_confidence": summary_stats(selected_confidences),
        "missing_frame_max_action_confidence": summary_stats(missing_max_confidences),
        "emitted_runs": contiguous_runs(emitted),
        "missing_runs": contiguous_runs(missing),
        "interpolated_frames": interpolated,
        "interpolated_runs": contiguous_runs(interpolated),
        "interpolated_fraction_of_emitted": None if not emitted else len(interpolated) / len(emitted),
        "ball_radius_frame_count": radius_count,
    }


def kinematic_metrics(trajectory: Trajectory) -> dict[str, Any]:
    speeds: list[float] = []
    accelerations: list[float] = []
    teleports: list[dict[str, Any]] = []
    for group in group_segments(trajectory):
        for first, second in zip(group, group[1:]):
            if first.xyz_m is None or second.xyz_m is None:
                continue
            dt = second.t - first.t
            if dt <= 0.0:
                continue
            speed = float(np.linalg.norm(second.xyz_m - first.xyz_m) / dt)
            speeds.append(speed)
            if speed > MAX_PLAUSIBLE_SPEED_MPS:
                teleports.append({"frame_from": first.frame, "frame_to": second.frame, "speed_mps": speed})
        for first, middle, last in zip(group, group[1:], group[2:]):
            if first.xyz_m is None or middle.xyz_m is None or last.xyz_m is None:
                continue
            dt1, dt2 = middle.t - first.t, last.t - middle.t
            if dt1 <= 0.0 or dt2 <= 0.0:
                continue
            v1 = (middle.xyz_m - first.xyz_m) / dt1
            v2 = (last.xyz_m - middle.xyz_m) / dt2
            accelerations.append(float(np.linalg.norm(v2 - v1) / ((dt1 + dt2) / 2.0)))
    return {
        "speed_mps": summary_stats(speeds),
        "acceleration_mps2": summary_stats(accelerations),
        "teleport_threshold_mps": MAX_PLAUSIBLE_SPEED_MPS,
        "teleport_count": len(teleports),
        "teleports": teleports,
    }


def pb_boundary_smoothness(pb: Trajectory, context: Mapping[str, Any]) -> dict[str, Any]:
    event_delta_v: list[float] = []
    interior_delta_v: list[float] = []
    boundary_set = set(context["boundary_frames"])
    for frame, middle in pb.samples.items():
        before, after = pb.samples.get(frame - 1), pb.samples.get(frame + 1)
        if before is None or after is None or before.xyz_m is None or middle.xyz_m is None or after.xyz_m is None:
            continue
        v_before = (middle.xyz_m - before.xyz_m) * pb.fps
        v_after = (after.xyz_m - middle.xyz_m) * pb.fps
        delta = float(np.linalg.norm(v_after - v_before))
        (event_delta_v if frame in boundary_set else interior_delta_v).append(delta)
    event_median = percentile(event_delta_v, 0.5)
    interior_median = percentile(interior_delta_v, 0.5)
    return {
        "event_boundary_delta_velocity_mps": summary_stats(event_delta_v),
        "interior_delta_velocity_mps": summary_stats(interior_delta_v),
        "event_to_interior_median_ratio": (
            None if event_median is None or interior_median is None or interior_median <= 1e-12 else event_median / interior_median
        ),
    }


def net_clearance(trajectory: Trajectory, net_height_m: float = 0.8636) -> dict[str, Any]:
    crossings: list[dict[str, Any]] = []
    for group in group_segments(trajectory):
        for first, second in zip(group, group[1:]):
            if first.xyz_m is None or second.xyz_m is None:
                continue
            y0, y1 = float(first.xyz_m[1]), float(second.xyz_m[1])
            if y0 == y1 or y0 * y1 > 0.0:
                continue
            alpha = -y0 / (y1 - y0)
            if not 0.0 <= alpha <= 1.0:
                continue
            z = float(first.xyz_m[2] + alpha * (second.xyz_m[2] - first.xyz_m[2]))
            crossings.append(
                {
                    "frame_interval": [first.frame, second.frame],
                    "height_m": z,
                    "clearance_m": z - net_height_m,
                }
            )
    clearances = [row["clearance_m"] for row in crossings]
    return {
        "net_height_m": net_height_m,
        "crossing_count": len(crossings),
        "positive_clearance_count": sum(value >= 0.0 for value in clearances),
        "clearance_m": summary_stats(clearances),
        "crossings": crossings,
    }


def our_bounce_metrics(ours: Trajectory, pb_bounces: Sequence[int], offset: int) -> dict[str, Any]:
    radius = DEFAULT_BALL_RADIUS_M
    declared: list[dict[str, Any]] = []
    if ours.arc is not None and isinstance(ours.arc.get("anchors"), list):
        for anchor in ours.arc["anchors"]:
            if not isinstance(anchor, Mapping) or anchor.get("kind") != "bounce":
                continue
            xyz = finite_vec(anchor.get("world_xyz"), 3)
            frame = anchor.get("frame")
            if xyz is None or not isinstance(frame, int):
                continue
            declared.append(
                {
                    "frame": frame,
                    "z_m": float(xyz[2]),
                    "abs_z_minus_radius_m": abs(float(xyz[2]) - radius),
                    "status": anchor.get("status"),
                }
            )
    aligned_rows: list[dict[str, Any]] = []
    for pb_frame in pb_bounces:
        our_frame = pb_frame + offset
        sample = ours.samples.get(our_frame)
        aligned_rows.append(
            {
                "pb_frame": pb_frame,
                "our_frame": our_frame,
                "our_3d_emitted": sample is not None and sample.xyz_m is not None,
                "our_z_m": None if sample is None or sample.xyz_m is None else float(sample.xyz_m[2]),
                "abs_z_minus_radius_m": (
                    None if sample is None or sample.xyz_m is None else abs(float(sample.xyz_m[2]) - radius)
                ),
            }
        )
    declared_frames = [row["frame"] for row in declared]
    timing_rows = [
        {
            "pb_frame": frame,
            "nearest_our_bounce_frame": None if not declared_frames else min(declared_frames, key=lambda ours_frame: abs(ours_frame - (frame + offset))),
        }
        for frame in pb_bounces
    ]
    for row in timing_rows:
        nearest = row["nearest_our_bounce_frame"]
        row["frame_delta"] = None if nearest is None else nearest - (row["pb_frame"] + offset)
    return {
        "assumed_ball_radius_m": radius,
        "our_declared_bounce_count": len(declared),
        "our_declared_bounces": declared,
        "our_declared_bounce_z_error_m": summary_stats(row["abs_z_minus_radius_m"] for row in declared),
        "at_pb_selected_bounce_frames": aligned_rows,
        "paired_z_error_m": summary_stats(
            row["abs_z_minus_radius_m"] for row in aligned_rows if row["abs_z_minus_radius_m"] is not None
        ),
        "nearest_declared_bounce_timing": timing_rows,
        "matched_within_2_frames": sum(
            row["frame_delta"] is not None and abs(row["frame_delta"]) <= 2 for row in timing_rows
        ),
    }


def coverage(trajectory: Trajectory, frames: Sequence[int], offset: int = 0) -> dict[str, Any]:
    emitted = 0
    visible_2d = 0
    status_counts: Counter[str] = Counter()
    band_counts: Counter[str] = Counter()
    for frame in frames:
        sample = trajectory.samples.get(frame + offset)
        if sample is not None:
            if sample.xyz_m is not None:
                emitted += 1
            if sample.status:
                status_counts[sample.status] += 1
            if sample.band:
                band_counts[sample.band] += 1
        two_d = trajectory.two_d_samples.get(frame + offset)
        if two_d is not None and two_d.visible:
            visible_2d += 1
    return {
        "frame_count": len(frames),
        "emitted_3d_count": emitted,
        "emitted_3d_fraction": None if not frames else emitted / len(frames),
        "visible_2d_count": visible_2d,
        "visible_2d_fraction": None if not frames else visible_2d / len(frames),
        "segment_status_counts": dict(sorted(status_counts.items())),
        "display_band_counts": dict(sorted(band_counts.items())),
    }


def head_to_head(pb: Trajectory, ours: Trajectory, frames: Sequence[int], offset: int) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    by_status: dict[str, list[float]] = {}
    axis_deltas: list[np.ndarray] = []
    for pb_frame in frames:
        pb_sample = pb.samples.get(pb_frame)
        our_sample = ours.samples.get(pb_frame + offset)
        if (
            pb_sample is None
            or pb_sample.xyz_m is None
            or our_sample is None
            or our_sample.xyz_m is None
        ):
            continue
        delta = our_sample.xyz_m - pb_sample.xyz_m
        error = float(np.linalg.norm(delta))
        axis_deltas.append(delta)
        status = our_sample.status or "unknown"
        by_status.setdefault(status, []).append(error)
        rows.append(
            {
                "pb_frame": pb_frame,
                "our_frame": pb_frame + offset,
                "pb_kind": pb_sample.kind,
                "our_status": status,
                "euclidean_delta_m": error,
            }
        )
    axis = np.stack(axis_deltas) if axis_deltas else np.empty((0, 3))
    z = axis[:, 2] if len(axis) else np.asarray([])
    z_bias = None if not len(z) else float(z.mean())
    z_loa = None
    if len(z) >= 2:
        sd = float(z.std(ddof=1))
        z_loa = [z_bias - 1.96 * sd, z_bias + 1.96 * sd]
    return {
        "paired_3d_count": len(rows),
        "euclidean_delta_m": summary_stats(row["euclidean_delta_m"] for row in rows),
        "axis_bias_ours_minus_pb_m": None if not len(axis) else axis.mean(axis=0).tolist(),
        "height_bland_altman_bias_m": z_bias,
        "height_bland_altman_95pct_limits_m": z_loa,
        "euclidean_delta_by_our_status_m": {key: summary_stats(value) for key, value in sorted(by_status.items())},
    }


def two_d_head_to_head(pb: Trajectory, ours: Trajectory, frames: Sequence[int], offset: int) -> dict[str, Any]:
    pb_count = ours_count = both_count = 0
    pb_only = ours_only = 0
    errors: list[float] = []
    for pb_frame in frames:
        pb_sample = pb.two_d_samples.get(pb_frame)
        our_sample = ours.two_d_samples.get(pb_frame + offset)
        pb_visible = pb_sample is not None and pb_sample.visible and pb_sample.xy_px is not None
        our_visible = our_sample is not None and our_sample.visible and our_sample.xy_px is not None
        pb_count += int(pb_visible)
        ours_count += int(our_visible)
        both_count += int(pb_visible and our_visible)
        pb_only += int(pb_visible and not our_visible)
        ours_only += int(our_visible and not pb_visible)
        if pb_visible and our_visible:
            errors.append(float(np.linalg.norm(pb_sample.xy_px - our_sample.xy_px)))
    return {
        "frame_count": len(frames),
        "pb_confidence_ge_0_5_count": pb_count,
        "our_visible_count": ours_count,
        "both_count": both_count,
        "pb_only_count": pb_only,
        "ours_only_count": ours_only,
        "paired_pixel_error": summary_stats(errors),
        "paired_error_le_20px_fraction": None if not errors else sum(error <= 20.0 for error in errors) / len(errors),
        "paired_error_gt_50px_count": sum(error > 50.0 for error in errors),
    }


def camera_crosscheck(pb_context: Mapping[str, Any], ours: Trajectory) -> dict[str, Any] | None:
    if ours.calibration is None:
        return None
    segment = camera_segment(pb_context)
    calibration = ours.calibration
    extrinsics = calibration.get("extrinsics")
    intrinsics = calibration.get("intrinsics")
    if not isinstance(extrinsics, Mapping) or not isinstance(intrinsics, Mapping):
        return None
    rotation = np.asarray(extrinsics["R"], dtype=float)
    translation = np.asarray(extrinsics["t"], dtype=float)
    our_center = -rotation.T @ translation
    pb_position = segment["position"]
    pb_center = pb_feet_to_ours_m(np.asarray([pb_position[axis] for axis in ("x", "y", "z")], dtype=float))
    image_size = calibration.get("image_size", [pb_context["width"], pb_context["height"]])
    width = float(image_size[0])
    our_fov = 2.0 * math.atan(width / (2.0 * float(intrinsics["fx"])))
    result: dict[str, Any] = {
        "camera_center_delta_m": float(np.linalg.norm(our_center - pb_center)),
        "horizontal_fov_delta_deg": math.degrees(float(segment["fov"]) - our_fov),
        "interpretation": "small camera agreement can rule against camera/scale as the dominant shared-clip gap, not prove either solve is accurate",
    }
    pb_points = segment.get("court_points") if isinstance(segment.get("court_points"), list) else []
    image_pts = calibration.get("image_pts")
    world_pts = calibration.get("world_pts")
    if len(pb_points) == 12 and isinstance(image_pts, list) and isinstance(world_pts, list) and len(image_pts) == len(world_pts):
        pb_observed = np.asarray(
            [[point["u"] * pb_context["width"], point["v"] * pb_context["height"]] for point in pb_points],
            dtype=float,
        )
        our_observed = np.asarray(image_pts, dtype=float)
        nearest = np.linalg.norm(pb_observed[:, None] - our_observed[None, :], axis=2).argmin(axis=1)
        if len(set(nearest.tolist())) == 12:
            shared_world_m = np.asarray(world_pts, dtype=float)[nearest]
            shared_world_ft = np.stack([ours_m_to_pb_feet(point) for point in shared_world_m])
            pb_projected, _ = project_pb_points(
                shared_world_ft, segment, int(pb_context["width"]), int(pb_context["height"])
            )
            our_projected = project_ours_points(shared_world_m, calibration)
            result["matched_court_point_assignment"] = "nearest unique point in matched reviewed calibration"
            result["pb_camera_to_pb_court_observations_px"] = summary_stats(
                np.linalg.norm(pb_projected - pb_observed, axis=1)
            )
            result["pb_camera_to_our_camera_projection_px"] = summary_stats(
                np.linalg.norm(pb_projected - our_projected, axis=1)
            )
            result["pb_court_observation_to_our_reviewed_point_px"] = summary_stats(
                np.linalg.norm(pb_observed - our_observed[nearest], axis=1)
            )
    return result


def build_scorecard(args: argparse.Namespace) -> dict[str, Any]:
    width, height = (int(value) for value in args.image_size.lower().split("x", 1))
    pb, pb_context = parse_pb_export(args.pb_export, width, height)
    rally_frames = sorted(pb_context["frame_records"])
    pb_physics = physics_pillar(pb)
    pb_bounce = pb_bounce_pillar(pb, pb_context)
    result: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "pbvision_no_gt_comparison_scorecard",
        "policy": {
            "competitor_reference_only": True,
            "pbvision_is_not_ground_truth": True,
            "promotion_evidence": False,
            "protected_clip_labels_read": False,
            "deterministic": True,
        },
        "inputs": {
            "pb_export": str(args.pb_export),
            "ours": None if args.ours is None else str(args.ours),
            "image_size": [width, height],
        },
        "pbvision": {
            "fps": pb.fps,
            "rally_ranges": pb_context["rally_ranges"],
            "coverage": coverage(pb, rally_frames),
            "selection_and_temporal_forensics": selection_forensics(pb, pb_context),
            "radius_depth_forensics": radius_depth_forensics(pb, pb_context),
            "smoothness_and_teleports": kinematic_metrics(pb),
            "boundary_smoothness": pb_boundary_smoothness(pb, pb_context),
            "net_clearance": net_clearance(pb),
        },
        "shared_no_gt_pillars": {
            "pbvision_physics_reintegration": pb_physics,
            "pbvision_court_plane_bounce": pb_bounce,
            "pbvision_internal_reprojection": pb_reprojection_pillar(pb, pb_context),
        },
        "thresholds": {
            "pb_ball_action_detection_confidence": 0.5,
            "bounce_peak_confidence": 0.5,
            "teleport_speed_mps": MAX_PLAUSIBLE_SPEED_MPS,
            "physics_plausible_rmse_m": PHYSICS_RMSE_LIMIT_M,
            "physics_plausible_p95_m": PHYSICS_P95_LIMIT_M,
            "net_height_m": 0.8636,
        },
    }
    if args.ours is None:
        return result

    ours = parse_ours(
        args.ours,
        ball_track_path=args.ours_ball_track,
        calibration_path=args.ours_calibration,
        fps_override=args.fps,
    )
    if args.frame_offset == "auto":
        alignment = align_trajectories(pb, ours, args.max_lag)
        offset = int(alignment["selected_offset"])
    else:
        offset = int(args.frame_offset)
        alignment = {
            "selected_offset": offset,
            "mapping": "ours_frame = pb_global_frame + selected_offset",
            "mode": "explicit",
        }
    pb_bounce_frames = [sample.frame for sample in pb.samples.values() if sample.kind == "bounce"]
    result["inputs"].update(
        {
            "ours_resolved": str(ours.source_path),
            "ours_ball_track": None if args.ours_ball_track is None else str(args.ours_ball_track),
            "ours_calibration": None if args.ours_calibration is None else str(args.ours_calibration),
        }
    )
    result["alignment"] = alignment
    result["ours"] = {
        "fps": ours.fps,
        "coverage_on_pb_rally": coverage(ours, rally_frames, offset),
        "physics_reintegration": physics_pillar(ours),
        "internal_reprojection": ours_reprojection_pillar(ours),
        "bounce_plausibility": our_bounce_metrics(ours, pb_bounce_frames, offset),
        "smoothness_and_teleports": kinematic_metrics(ours),
        "net_clearance": net_clearance(ours),
    }
    result["head_to_head"] = head_to_head(pb, ours, rally_frames, offset)
    result["two_d_head_to_head"] = two_d_head_to_head(pb, ours, rally_frames, offset)
    result["camera_crosscheck"] = camera_crosscheck(pb_context, ours)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pb-export", type=Path, required=True, help="pb.vision cv_export.json")
    parser.add_argument("--ours", type=Path, help="our run directory or ball/world JSON; omit for PB-only pillars")
    parser.add_argument("--ours-ball-track", type=Path, help="explicit our 2D ball_track.json for auto alignment")
    parser.add_argument("--ours-calibration", type=Path, help="explicit our court_calibration.json")
    parser.add_argument("--frame-offset", default="auto", help="integer or auto; mapping is ours=PB_global+offset")
    parser.add_argument("--max-lag", type=int, default=90, help="maximum absolute frame lag for auto alignment")
    parser.add_argument("--fps", type=float, help="override our FPS")
    parser.add_argument("--image-size", default="1920x1080", help="pixel size for normalized PB coordinates")
    parser.add_argument("--output", type=Path, help="write JSON here; stdout when omitted")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scorecard = build_scorecard(args)
    rendered = json.dumps(scorecard, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
        print(args.output)


if __name__ == "__main__":
    main()

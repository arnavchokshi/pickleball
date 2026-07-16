"""CPU-only 3D ball physics reconstruction primitives."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Any, Iterable, Mapping, Sequence

from .court_auto_evidence import calibration_for_image_size
from .coordinates import (
    CoordinateSpace,
    homography_pixel_space,
    project_world_array_pinhole,
    resolve_homography_pixel_convention,
    unproject_image_points_to_world,
)
from .schemas import CourtCalibration


@dataclass(frozen=True)
class BallSample3D:
    """Single reconstructed ball sample in world coordinates."""

    t: float
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class BounceEvent:
    """Simple bounce event that can be projected into BallTrack.bounces."""

    t: float
    world_xy: tuple[float, float]
    sample_index: int
    z_min: float


@dataclass(frozen=True)
class BounceArcReconstruction:
    """Camera-calibrated reconstruction from image observations around one bounce."""

    status: str
    samples: tuple[BallSample3D, ...] = ()
    bounces: list[dict[str, object]] | None = None
    frame_indices: tuple[int, ...] = ()
    reprojection_rmse_px: float | None = None
    max_reprojection_error_px: float | None = None
    candidate_count: int = 0
    selected_bounce_time_s: float | None = None
    effective_accel_z_mps2: float | None = None
    notes: tuple[str, ...] = ()

    @property
    def sample_count(self) -> int:
        return len(self.samples)

    @property
    def bounce_count(self) -> int:
        return len(self.bounces or [])

    def summary(self) -> dict[str, object]:
        return {
            "status": self.status,
            "sample_count": self.sample_count,
            "bounce_count": self.bounce_count,
            "reprojection_rmse_px": self.reprojection_rmse_px,
            "max_reprojection_error_px": self.max_reprojection_error_px,
            "candidate_count": self.candidate_count,
            "selected_bounce_time_s": self.selected_bounce_time_s,
            "effective_accel_z_mps2": self.effective_accel_z_mps2,
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class ParabolaFit3D:
    """Linear x/y plus parabolic z fit over a short ball segment."""

    t0: float
    x0: float
    y0: float
    z0: float
    vx: float
    vy: float
    vz: float
    accel_z: float
    inlier_indices: tuple[int, ...]
    outlier_indices: tuple[int, ...]
    rms_residual_m: float
    max_residual_m: float

    def predict(self, t: float) -> tuple[float, float, float]:
        dt = t - self.t0
        return (
            self.x0 + self.vx * dt,
            self.y0 + self.vy * dt,
            self.z0 + self.vz * dt + 0.5 * self.accel_z * dt * dt,
        )

    def residual_m(self, sample: BallSample3D) -> float:
        x, y, z = self.predict(sample.t)
        return sqrt((sample.x - x) ** 2 + (sample.y - y) ** 2 + (sample.z - z) ** 2)


def reconstruct_bounce_arcs_from_image_track(
    ball_payload: Mapping[str, Any],
    calibration: CourtCalibration | Mapping[str, Any],
    *,
    image_size: tuple[int, int] | None = None,
    max_reprojection_rmse_px: float = 12.0,
    max_neighbor_gap_s: float | None = None,
    min_samples: int = 5,
    max_fit_samples: int = 21,
) -> BounceArcReconstruction:
    """Fit a calibrated two-arc bounce model to image-only ball observations.

    A bounce is a collision discontinuity, so this uses two vertical arcs joined
    at an observed bounce candidate instead of a single smooth parabola. The fit
    is accepted only when reprojection error stays under
    ``max_reprojection_rmse_px``.
    """

    if max_reprojection_rmse_px <= 0.0:
        raise ValueError("max_reprojection_rmse_px must be positive")
    if min_samples < 5:
        raise ValueError("min_samples must be at least 5")
    if max_fit_samples < min_samples:
        raise ValueError("max_fit_samples must be >= min_samples")

    try:
        import numpy as np  # type: ignore[import-not-found]
        from scipy.optimize import least_squares  # type: ignore[import-not-found]
    except ImportError as exc:
        return BounceArcReconstruction(status="missing_numeric_dependency", notes=(str(exc),))

    parsed_calibration = _scaled_calibration(calibration, image_size=image_size)
    observations = _visible_image_observations(ball_payload)
    if len(observations) < min_samples:
        return BounceArcReconstruction(
            status="insufficient_image_samples",
            candidate_count=0,
            notes=(f"need at least {min_samples} visible image observations",),
        )

    runs = _contiguous_observation_runs(
        observations,
        max_gap_s=max_neighbor_gap_s or _default_max_neighbor_gap_s(ball_payload),
    )
    camera = _camera_projection_arrays(parsed_calibration, np_module=np)
    best: dict[str, Any] | None = None
    candidate_count = 0

    for run in runs:
        if len(run) < min_samples:
            continue
        for bounce_offset in range(2, len(run) - 2):
            window = _fit_window(run, bounce_offset=bounce_offset, max_fit_samples=max_fit_samples)
            if len(window) < min_samples:
                continue
            local_bounce_offset = next(
                (index for index, obs in enumerate(window) if obs["frame_index"] == run[bounce_offset]["frame_index"]),
                len(window) // 2,
            )
            if local_bounce_offset < 2 or len(window) - local_bounce_offset - 1 < 2:
                continue
            candidate_count += 1
            fitted = _fit_bounce_window(
                window,
                bounce_offset=local_bounce_offset,
                calibration=parsed_calibration,
                camera=camera,
                np_module=np,
                least_squares=least_squares,
            )
            if fitted is None:
                continue
            if best is None or _fit_rank(fitted) < _fit_rank(best):
                best = fitted

    if best is None:
        return BounceArcReconstruction(
            status="no_fit",
            candidate_count=candidate_count,
            notes=("no calibrated two-arc bounce candidate could be fitted",),
        )
    if float(best["reprojection_rmse_px"]) > max_reprojection_rmse_px:
        return BounceArcReconstruction(
            status="no_fit_under_reprojection_gate",
            samples=tuple(best["samples"]),
            bounces=best["bounces"],
            frame_indices=tuple(best["frame_indices"]),
            reprojection_rmse_px=float(best["reprojection_rmse_px"]),
            max_reprojection_error_px=float(best["max_reprojection_error_px"]),
            candidate_count=candidate_count,
            selected_bounce_time_s=float(best["bounce_time_s"]),
            effective_accel_z_mps2=float(best["accel_z_mps2"]),
            notes=(f"best fit exceeded {max_reprojection_rmse_px:.3f}px reprojection RMSE gate",),
        )

    return BounceArcReconstruction(
        status="ran",
        samples=tuple(best["samples"]),
        bounces=best["bounces"],
        frame_indices=tuple(best["frame_indices"]),
        reprojection_rmse_px=float(best["reprojection_rmse_px"]),
        max_reprojection_error_px=float(best["max_reprojection_error_px"]),
        candidate_count=candidate_count,
        selected_bounce_time_s=float(best["bounce_time_s"]),
        effective_accel_z_mps2=float(best["accel_z_mps2"]),
        notes=("fit calibrated two-arc bounce model from image observations",),
    )


def _scaled_calibration(
    calibration: CourtCalibration | Mapping[str, Any],
    *,
    image_size: tuple[int, int] | None,
) -> CourtCalibration:
    parsed = calibration if isinstance(calibration, CourtCalibration) else CourtCalibration.model_validate(calibration)
    if image_size is None:
        return parsed
    width, height = image_size
    if width <= 0 or height <= 0:
        raise ValueError("image_size must contain positive width and height")
    return calibration_for_image_size(parsed, width=int(width), height=int(height))


def _visible_image_observations(ball_payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    frames = ball_payload.get("frames")
    if not isinstance(frames, list):
        return observations
    for index, frame in enumerate(frames):
        if not isinstance(frame, Mapping):
            continue
        if frame.get("visible") is not True:
            continue
        xy = frame.get("xy")
        if not isinstance(xy, Sequence) or isinstance(xy, (str, bytes)) or len(xy) != 2:
            continue
        try:
            t = float(frame.get("t"))
            x = float(xy[0])
            y = float(xy[1])
        except (TypeError, ValueError):
            continue
        observations.append({"frame_index": index, "t": t, "xy": (x, y)})
    return sorted(observations, key=lambda item: (item["t"], item["frame_index"]))


def _default_max_neighbor_gap_s(ball_payload: Mapping[str, Any]) -> float:
    fps = ball_payload.get("fps")
    try:
        fps_value = float(fps)
    except (TypeError, ValueError):
        fps_value = 0.0
    if fps_value > 0.0:
        return max(0.12, 3.5 / fps_value)
    return 0.18


def _contiguous_observation_runs(observations: Sequence[dict[str, Any]], *, max_gap_s: float) -> list[list[dict[str, Any]]]:
    if max_gap_s <= 0.0:
        raise ValueError("max_gap_s must be positive")
    runs: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    previous: dict[str, Any] | None = None
    for observation in observations:
        if (
            previous is not None
            and (
                int(observation["frame_index"]) - int(previous["frame_index"]) > 1
                or float(observation["t"]) - float(previous["t"]) > max_gap_s
            )
        ):
            runs.append(current)
            current = []
        current.append(observation)
        previous = observation
    if current:
        runs.append(current)
    return runs


def _fit_window(
    run: Sequence[dict[str, Any]],
    *,
    bounce_offset: int,
    max_fit_samples: int,
) -> list[dict[str, Any]]:
    if len(run) <= max_fit_samples:
        return list(run)
    radius = max_fit_samples // 2
    start = max(0, bounce_offset - radius)
    end = min(len(run), start + max_fit_samples)
    start = max(0, end - max_fit_samples)
    return list(run[start:end])


def _camera_projection_arrays(calibration: CourtCalibration, *, np_module: Any) -> dict[str, Any]:
    return {
        "intrinsics": calibration.intrinsics,
        "rotation": np_module.asarray(calibration.extrinsics.R, dtype=float),
        "translation": np_module.asarray(calibration.extrinsics.t, dtype=float),
        "reference_space": _calibration_homography_space(calibration),
    }


def _fit_bounce_window(
    observations: Sequence[dict[str, Any]],
    *,
    bounce_offset: int,
    calibration: CourtCalibration,
    camera: Mapping[str, Any],
    np_module: Any,
    least_squares: Any,
) -> dict[str, Any] | None:
    times = np_module.asarray([float(observation["t"]) for observation in observations], dtype=float)
    image = np_module.asarray([observation["xy"] for observation in observations], dtype=float)
    bounce_time = float(times[bounce_offset])
    homography_space = _calibration_homography_space(calibration)
    try:
        ground_xy = unproject_image_points_to_world(
            calibration.homography,
            [observation["xy"] for observation in observations],
            input_space=homography_space,
            homography_space=homography_space,
            output_space=CoordinateSpace.WORLD_XY_HOMOGRAPHY_M,
        )
    except ValueError:
        return None
    ground = np_module.asarray(ground_xy, dtype=float)
    bounce_ground = ground[bounce_offset]
    dt_span = max(float(times[-1] - times[0]), 1e-3)
    vx0 = float((ground[-1, 0] - ground[0, 0]) / dt_span)
    vy0 = float((ground[-1, 1] - ground[0, 1]) / dt_span)
    initial = np_module.asarray([bounce_ground[0], bounce_ground[1], vx0, vy0, 0.04, 3.0, 2.8, -9.81], dtype=float)
    lower = np_module.asarray([-30.0, -40.0, -80.0, -80.0, 0.0, 0.05, 0.05, -15.0], dtype=float)
    upper = np_module.asarray([30.0, 40.0, 80.0, 80.0, 0.35, 35.0, 35.0, 4.0], dtype=float)

    def residuals(params: Any) -> Any:
        world = _piecewise_bounce_points(params, times, bounce_time=bounce_time, np_module=np_module)
        projected = _project_world_array(world, camera=camera, np_module=np_module)
        residual = (projected - image).reshape(-1)
        z = world[:, 2]
        low_penalty = np_module.minimum(z, 0.0) * 120.0
        high_penalty = np_module.maximum(z - 5.0, 0.0) * 20.0
        accel_prior = np_module.asarray([(float(params[7]) + 9.81) / 8.0], dtype=float)
        return np_module.concatenate([residual, low_penalty, high_penalty, accel_prior])

    try:
        result = least_squares(residuals, initial, bounds=(lower, upper), max_nfev=3000)
    except ValueError:
        return None
    if not result.success:
        return None

    world = _piecewise_bounce_points(result.x, times, bounce_time=bounce_time, np_module=np_module)
    projected = _project_world_array(world, camera=camera, np_module=np_module)
    errors = np_module.linalg.norm(projected - image, axis=1)
    rmse = float(np_module.sqrt(np_module.mean(errors * errors)))
    samples = tuple(
        BallSample3D(
            t=float(time),
            x=float(point[0]),
            y=float(point[1]),
            z=float(point[2]),
        )
        for time, point in zip(times, world, strict=True)
    )
    bounce = BounceEvent(
        t=bounce_time,
        world_xy=(float(world[bounce_offset, 0]), float(world[bounce_offset, 1])),
        sample_index=bounce_offset,
        z_min=float(world[bounce_offset, 2]),
    )
    return {
        "samples": samples,
        "bounces": project_bounces_to_ball_track((bounce,)),
        "frame_indices": tuple(int(observation["frame_index"]) for observation in observations),
        "reprojection_rmse_px": rmse,
        "max_reprojection_error_px": float(np_module.max(errors)),
        "bounce_time_s": bounce_time,
        "accel_z_mps2": float(result.x[7]),
    }


def _piecewise_bounce_points(params: Any, times: Any, *, bounce_time: float, np_module: Any) -> Any:
    xb, yb, vx, vy, zmin, vdown, vup, accel_z = [float(value) for value in params]
    dt = times - bounce_time
    x = xb + vx * dt
    y = yb + vy * dt
    z_before = zmin - vdown * dt + 0.5 * accel_z * dt * dt
    z_after = zmin + vup * dt + 0.5 * accel_z * dt * dt
    z = np_module.where(dt <= 0.0, z_before, z_after)
    return np_module.column_stack([x, y, z])


def _project_world_array(world: Any, *, camera: Mapping[str, Any], np_module: Any) -> Any:
    return project_world_array_pinhole(
        world,
        rotation=camera["rotation"],
        translation=camera["translation"],
        intrinsics=camera["intrinsics"],
        input_space=CoordinateSpace.WORLD_COURT_NETCENTER_Z_UP_M,
        output_space=CoordinateSpace.PIXELS_UNDISTORTED_NATIVE,
        reference_space=CoordinateSpace(camera["reference_space"]),
        np_module=np_module,
    )


def _calibration_homography_space(calibration: CourtCalibration) -> CoordinateSpace:
    payload = calibration.model_dump(mode="python", exclude_none=True)
    convention = resolve_homography_pixel_convention(payload, default="raw_pixels")
    return homography_pixel_space(convention)


def _fit_rank(fit: Mapping[str, Any]) -> tuple[float, float]:
    return (float(fit["reprojection_rmse_px"]), float(fit["max_reprojection_error_px"]))


def fit_parabola_segment(
    samples: Sequence[BallSample3D],
    *,
    residual_threshold_m: float | None = None,
    min_inliers: int = 3,
) -> ParabolaFit3D:
    """Fit linear x/y and constant-acceleration height to a short segment.

    If ``residual_threshold_m`` is provided, the largest residual outlier is
    removed deterministically until all retained samples are under threshold or
    ``min_inliers`` would be violated.
    """

    if len(samples) < min_inliers:
        raise ValueError("fit_parabola_segment requires at least min_inliers samples")
    if min_inliers < 3:
        raise ValueError("min_inliers must be at least 3 for a height parabola")
    if residual_threshold_m is not None and residual_threshold_m < 0.0:
        raise ValueError("residual_threshold_m must be non-negative")

    active = list(range(len(samples)))
    outliers: list[int] = []

    while True:
        fit = _fit_indices(samples, active, tuple(sorted(outliers)))
        residuals = [(idx, fit.residual_m(samples[idx])) for idx in active]
        max_idx, max_residual = max(residuals, key=lambda item: (item[1], -item[0]))
        if residual_threshold_m is None or max_residual <= residual_threshold_m:
            return fit
        if len(active) <= min_inliers:
            return fit
        active.remove(max_idx)
        outliers.append(max_idx)


def detect_bounce_events(
    samples: Sequence[BallSample3D],
    *,
    min_vertical_speed_mps: float = 0.0,
    min_separation_s: float = 0.05,
) -> tuple[BounceEvent, ...]:
    """Detect bounces from a z local minimum with downward-to-upward velocity."""

    if len(samples) < 3:
        return ()
    ordered = sorted(enumerate(samples), key=lambda item: (item[1].t, item[0]))
    events: list[BounceEvent] = []
    last_t: float | None = None

    for ordered_idx in range(1, len(ordered) - 1):
        _, prev_sample = ordered[ordered_idx - 1]
        original_idx, sample = ordered[ordered_idx]
        _, next_sample = ordered[ordered_idx + 1]

        before_dt = sample.t - prev_sample.t
        after_dt = next_sample.t - sample.t
        if before_dt <= 0.0 or after_dt <= 0.0:
            continue

        vz_before = (sample.z - prev_sample.z) / before_dt
        vz_after = (next_sample.z - sample.z) / after_dt
        is_z_minimum = sample.z <= prev_sample.z and sample.z <= next_sample.z
        has_sign_change = vz_before < -min_vertical_speed_mps and vz_after > min_vertical_speed_mps
        separated = last_t is None or sample.t - last_t >= min_separation_s

        if is_z_minimum and has_sign_change and separated:
            events.append(
                BounceEvent(
                    t=sample.t,
                    world_xy=(sample.x, sample.y),
                    sample_index=original_idx,
                    z_min=sample.z,
                )
            )
            last_t = sample.t

    return tuple(events)


def project_bounces_to_ball_track(events: Iterable[BounceEvent]) -> list[dict[str, object]]:
    """Return dictionaries compatible with the existing BallTrack.bounces field."""

    return [
        {"t": event.t, "world_xy": [event.world_xy[0], event.world_xy[1]]}
        for event in sorted(events, key=lambda event: event.t)
    ]


def _fit_indices(
    samples: Sequence[BallSample3D],
    indices: Sequence[int],
    outlier_indices: tuple[int, ...],
) -> ParabolaFit3D:
    if len(indices) < 3:
        raise ValueError("at least 3 inliers are required for a height parabola")

    t0 = samples[indices[0]].t
    linear_rows = [[1.0, samples[idx].t - t0] for idx in indices]
    parabola_rows = [
        [1.0, samples[idx].t - t0, (samples[idx].t - t0) ** 2]
        for idx in indices
    ]

    x0, vx = _least_squares(linear_rows, [samples[idx].x for idx in indices])
    y0, vy = _least_squares(linear_rows, [samples[idx].y for idx in indices])
    z0, vz, z_quadratic = _least_squares(parabola_rows, [samples[idx].z for idx in indices])
    accel_z = 2.0 * z_quadratic

    provisional = ParabolaFit3D(
        t0=t0,
        x0=x0,
        y0=y0,
        z0=z0,
        vx=vx,
        vy=vy,
        vz=vz,
        accel_z=accel_z,
        inlier_indices=tuple(indices),
        outlier_indices=outlier_indices,
        rms_residual_m=0.0,
        max_residual_m=0.0,
    )
    residuals = [provisional.residual_m(samples[idx]) for idx in indices]
    rms = sqrt(sum(residual * residual for residual in residuals) / len(residuals))
    max_residual = max(residuals)
    return ParabolaFit3D(
        t0=t0,
        x0=x0,
        y0=y0,
        z0=z0,
        vx=vx,
        vy=vy,
        vz=vz,
        accel_z=accel_z,
        inlier_indices=tuple(indices),
        outlier_indices=outlier_indices,
        rms_residual_m=rms,
        max_residual_m=max_residual,
    )


def _least_squares(rows: Sequence[Sequence[float]], values: Sequence[float]) -> tuple[float, ...]:
    if not rows:
        raise ValueError("least squares requires at least one row")
    width = len(rows[0])
    normal = [[0.0 for _ in range(width)] for _ in range(width)]
    rhs = [0.0 for _ in range(width)]

    for row, value in zip(rows, values):
        if len(row) != width:
            raise ValueError("least squares rows must have consistent width")
        for i in range(width):
            rhs[i] += row[i] * value
            for j in range(width):
                normal[i][j] += row[i] * row[j]

    return tuple(_solve_linear_system(normal, rhs))


def _solve_linear_system(matrix: list[list[float]], rhs: list[float]) -> list[float]:
    n = len(rhs)
    for col in range(n):
        pivot_row = max(range(col, n), key=lambda row: abs(matrix[row][col]))
        if abs(matrix[pivot_row][col]) < 1e-12:
            raise ValueError("cannot fit parabola to singular sample times")
        if pivot_row != col:
            matrix[col], matrix[pivot_row] = matrix[pivot_row], matrix[col]
            rhs[col], rhs[pivot_row] = rhs[pivot_row], rhs[col]

        pivot = matrix[col][col]
        for j in range(col, n):
            matrix[col][j] /= pivot
        rhs[col] /= pivot

        for row in range(n):
            if row == col:
                continue
            factor = matrix[row][col]
            if factor == 0.0:
                continue
            for j in range(col, n):
                matrix[row][j] -= factor * matrix[col][j]
            rhs[row] -= factor * rhs[col]

    return rhs


__all__ = [
    "BallSample3D",
    "BounceArcReconstruction",
    "BounceEvent",
    "ParabolaFit3D",
    "detect_bounce_events",
    "fit_parabola_segment",
    "project_bounces_to_ball_track",
    "reconstruct_bounce_arcs_from_image_track",
]

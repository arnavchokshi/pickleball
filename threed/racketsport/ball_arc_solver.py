"""Event-anchored 3D ball arc solver for render-only world continuity.

The solver treats monocular 2D ball sightings as camera rays. Human-reviewed
bounces and loose contact priors provide 3D anchors; each consecutive anchor
pair bounds one free-flight segment. Outputs are explicitly render-only and
must not feed BALL detector metrics, gates, training, or promotion.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from statistics import median
from typing import Any, Mapping, Sequence

from .ball_physics3d import _project_world_array, reconstruct_bounce_arcs_from_image_track
from .skeleton3d import semanticize_skeleton_payload


SCHEMA_VERSION = 1
ARTIFACT_TYPE = "racketsport_ball_track_arc_solved"
LANE = "BALL-ARC-SOLVER"
SOURCE = "event_anchored_drag_arc_solver"
BALL_RADIUS_M = 0.0371


@dataclass(frozen=True)
class PhysicsParameters:
    """Fixed round-1 pickleball flight parameters."""

    ball_type: str = "outdoor"
    gravity_mps2: float = 9.80665
    mass_kg: float = 0.0255
    diameter_m: float = 0.0742
    rho_air_kg_m3: float = 1.20
    drag_cd: float = 0.33

    @classmethod
    def for_ball_type(cls, ball_type: str) -> PhysicsParameters:
        if ball_type == "indoor":
            return cls(ball_type="indoor", drag_cd=0.45)
        if ball_type == "no_drag_test":
            return cls.no_drag()
        return cls(ball_type="outdoor", drag_cd=0.33)

    @classmethod
    def no_drag(cls) -> PhysicsParameters:
        return cls(ball_type="no_drag_test", drag_cd=0.0)

    @property
    def radius_m(self) -> float:
        return self.diameter_m / 2.0

    @property
    def drag_k_per_m(self) -> float:
        if self.drag_cd <= 0.0:
            return 0.0
        area = math.pi * self.radius_m * self.radius_m
        return 0.5 * self.rho_air_kg_m3 * self.drag_cd * area / self.mass_kg

    def summary(self) -> dict[str, Any]:
        return {
            "ball_type": self.ball_type,
            "gravity_mps2": self.gravity_mps2,
            "mass_kg": self.mass_kg,
            "diameter_m": self.diameter_m,
            "radius_m": self.radius_m,
            "rho_air_kg_m3": self.rho_air_kg_m3,
            "drag_cd": self.drag_cd,
            "drag_k_per_m": self.drag_k_per_m,
        }


@dataclass(frozen=True)
class BallArcSolverConfig:
    """Numerical and reporting controls for the event-anchored solver."""

    robust_pixel_sigma: float = 6.0
    robust_loss: str = "huber"
    robust_f_scale: float = 1.0
    endpoint_anchor_weight: float = 1.0
    anchor_relax_sigma_multiplier: float = 3.0
    max_reprojection_inlier_px: float = 18.0
    min_segment_dt_s: float = 0.045
    min_segment_observations: int = 3
    integrator_max_step_s: float = 1.0 / 240.0
    min_anchor_sigma_m: float = 0.01
    enable_event_subset_selection: bool = True
    selection_max_speed_mps: float = 35.0
    selection_min_residual_reduction: float = 0.05
    selection_split_penalty: float = 0.25
    selection_max_nfev: int = 350
    loo_max_nfev: int = 600
    rally_endpoint_sigma_m: float = 2.0
    contact_anchor_sigma_m: float = 0.35
    contact_reach_offset_m: float = 0.15
    min_joint_confidence: float = 0.20
    reviewed_bounce_base_sigma_m: float = 0.05
    proposed_bounce_sigma_m: float = 0.18
    enable_event_discovery: bool = True
    discovery_reprojection_px: float = 60.0
    discovery_min_neighbor_px: float = 18.0
    discovery_anchor_separation_frames: int = 4
    visible_confidence_min: float = 0.50
    net_clearance_slack_m: float = 0.08
    min_plausible_speed_mps: float = 3.0
    max_plausible_speed_mps: float = 35.0
    max_plausible_apex_m: float = 8.0
    baseline_loo_median_m: float | None = None
    max_physical_violation_fraction: float = 0.20
    enable_size_depth_residual: bool = True
    size_depth_sigma_m: float = 200.0
    size_depth_relative_sigma: float = 0.18
    size_confidence_floor: float = 0.10
    enable_weak_segments: bool = True
    weak_segment_min_observations: int = 4
    weak_segment_max_gap_s: float = 0.25
    weak_segment_anchor_sigma_multiplier: float = 8.0
    weak_segment_pixel_sigma_multiplier: float = 2.5
    weak_size_depth_sigma_m: float = 6.0
    weak_segment_max_nfev: int = 800

    def __post_init__(self) -> None:
        if self.robust_pixel_sigma <= 0.0:
            raise ValueError("robust_pixel_sigma must be positive")
        if self.max_reprojection_inlier_px <= 0.0:
            raise ValueError("max_reprojection_inlier_px must be positive")
        if self.min_segment_dt_s <= 0.0:
            raise ValueError("min_segment_dt_s must be positive")
        if self.min_segment_observations < 0:
            raise ValueError("min_segment_observations must be non-negative")
        if self.integrator_max_step_s <= 0.0:
            raise ValueError("integrator_max_step_s must be positive")
        if self.min_anchor_sigma_m <= 0.0:
            raise ValueError("min_anchor_sigma_m must be positive")
        if self.selection_max_speed_mps <= 0.0:
            raise ValueError("selection_max_speed_mps must be positive")
        if self.selection_min_residual_reduction < 0.0:
            raise ValueError("selection_min_residual_reduction must be non-negative")
        if self.selection_split_penalty < 0.0:
            raise ValueError("selection_split_penalty must be non-negative")
        if self.selection_max_nfev <= 0:
            raise ValueError("selection_max_nfev must be positive")
        if self.loo_max_nfev <= 0:
            raise ValueError("loo_max_nfev must be positive")
        if self.rally_endpoint_sigma_m <= 0.0:
            raise ValueError("rally_endpoint_sigma_m must be positive")
        if self.contact_anchor_sigma_m <= 0.0:
            raise ValueError("contact_anchor_sigma_m must be positive")
        if self.reviewed_bounce_base_sigma_m <= 0.0:
            raise ValueError("reviewed_bounce_base_sigma_m must be positive")
        if self.proposed_bounce_sigma_m <= 0.0:
            raise ValueError("proposed_bounce_sigma_m must be positive")
        if self.size_depth_sigma_m <= 0.0:
            raise ValueError("size_depth_sigma_m must be positive")
        if self.size_depth_relative_sigma < 0.0:
            raise ValueError("size_depth_relative_sigma must be non-negative")
        if not 0.0 < self.size_confidence_floor <= 1.0:
            raise ValueError("size_confidence_floor must be in (0, 1]")
        if self.weak_segment_min_observations < 2:
            raise ValueError("weak_segment_min_observations must be at least 2")
        if self.weak_segment_max_gap_s <= 0.0:
            raise ValueError("weak_segment_max_gap_s must be positive")
        if self.weak_segment_anchor_sigma_multiplier <= 0.0:
            raise ValueError("weak_segment_anchor_sigma_multiplier must be positive")
        if self.weak_segment_pixel_sigma_multiplier <= 0.0:
            raise ValueError("weak_segment_pixel_sigma_multiplier must be positive")
        if self.weak_size_depth_sigma_m <= 0.0:
            raise ValueError("weak_size_depth_sigma_m must be positive")
        if self.weak_segment_max_nfev <= 0:
            raise ValueError("weak_segment_max_nfev must be positive")


@dataclass(frozen=True)
class BallObservation:
    frame: int
    t: float
    xy: tuple[float, float]
    confidence: float = 1.0
    visible: bool = True
    diameter_px: float | None = None
    size_confidence: float | None = None
    size_source: str | None = None


@dataclass(frozen=True)
class AnchorEvent:
    anchor_id: str
    kind: str
    t: float
    frame: int
    world_xyz: tuple[float, float, float]
    sigma_m: float
    status: str
    player_id: int | str | None = None
    immovable: bool = False
    source: str | None = None
    details: Mapping[str, Any] | None = None

    def to_json(self) -> dict[str, Any]:
        payload = {
            "anchor_id": self.anchor_id,
            "kind": self.kind,
            "t": _round(self.t, 9),
            "frame": int(self.frame),
            "world_xyz": _vec_json(self.world_xyz),
            "sigma_m": _round(self.sigma_m, 6),
            "status": self.status,
            "immovable": bool(self.immovable),
        }
        if self.player_id is not None:
            payload["player_id"] = self.player_id
        if self.source is not None:
            payload["source"] = self.source
        if self.details:
            payload["details"] = dict(self.details)
        return payload


@dataclass(frozen=True)
class FlightSegmentFit:
    segment_id: int
    status: str
    start_anchor: AnchorEvent
    end_anchor: AnchorEvent
    initial_position_m: tuple[float, float, float]
    initial_velocity_mps: tuple[float, float, float]
    observations: tuple[BallObservation, ...]
    inlier_frames: tuple[int, ...]
    outlier_frames: tuple[int, ...]
    reprojection_errors_px: Mapping[int, float]
    reprojection_rmse_px: float | None
    max_reprojection_error_px: float | None
    endpoint_error_m: float
    net_clearance_m: float | None
    net_clearance_ok: bool | None
    physical_sanity: Mapping[str, Any]
    size_residuals_m: Mapping[str, Any]

    @property
    def inlier_count(self) -> int:
        return len(self.inlier_frames)

    @property
    def outlier_count(self) -> int:
        return len(self.outlier_frames)

    def predict(self, t: float, physics: PhysicsParameters, config: BallArcSolverConfig) -> tuple[float, float, float]:
        return _integrate_positions(
            self.initial_position_m,
            self.initial_velocity_mps,
            [float(t)],
            t0=self.start_anchor.t,
            physics=physics,
            config=config,
        )[0]

    def to_json(self) -> dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "status": self.status,
            "t0": _round(self.start_anchor.t, 9),
            "t1": _round(self.end_anchor.t, 9),
            "frame_start": int(self.start_anchor.frame),
            "frame_end": int(self.end_anchor.frame),
            "start_anchor": self.start_anchor.anchor_id,
            "end_anchor": self.end_anchor.anchor_id,
            "anchors_used": [self.start_anchor.to_json(), self.end_anchor.to_json()],
            "initial_position_m": _vec_json(self.initial_position_m),
            "initial_velocity_mps": _vec_json(self.initial_velocity_mps),
            "initial_speed_mps": _round(_norm(self.initial_velocity_mps), 6),
            "inlier_count": self.inlier_count,
            "outlier_count": self.outlier_count,
            "inlier_frames": list(self.inlier_frames),
            "outlier_frames": list(self.outlier_frames),
            "reprojection_rmse_px": _optional_round(self.reprojection_rmse_px, 6),
            "max_reprojection_error_px": _optional_round(self.max_reprojection_error_px, 6),
            "endpoint_error_m": _round(self.endpoint_error_m, 6),
            "net_clearance_m": _optional_round(self.net_clearance_m, 6),
            "net_clearance_ok": self.net_clearance_ok,
            "physical_sanity": dict(self.physical_sanity),
            "size_residuals_m": dict(self.size_residuals_m),
        }


def build_bounce_anchor(
    bounce: Mapping[str, Any],
    calibration: Mapping[str, Any],
    *,
    ball_radius_m: float = BALL_RADIUS_M,
    ball_xy: Sequence[float] | None = None,
    status: str = "human_reviewed",
    sigma_m: float | None = None,
) -> AnchorEvent:
    """Build an exact court-plane anchor from a reviewed or proposed bounce."""

    frame = _frame_from_mapping(bounce)
    if frame is None:
        raise ValueError("bounce anchor requires frame/frame_index")
    t = _float_or_none(bounce.get("t"))
    if t is None:
        fps = _float_or_none(bounce.get("fps")) or 30.0
        t = frame / fps
    xy = _xy_tuple(ball_xy if ball_xy is not None else bounce.get("xy"))
    if xy is None:
        raise ValueError("bounce anchor requires a visible ball xy at the bounce frame")
    origin, direction = pixel_ray_world(calibration, xy)
    world_xyz = intersect_ray_z(origin, direction, ball_radius_m)
    if sigma_m is None:
        sigma_m = anchor_sigma_for_bounce(calibration, xy, base_sigma_m=0.05 if status == "human_reviewed" else 0.12)
    review_id = bounce.get("review_id")
    anchor_id = str(review_id) if isinstance(review_id, str) and review_id else f"{status}_bounce_{frame:06d}"
    return AnchorEvent(
        anchor_id=anchor_id,
        kind="bounce",
        t=float(t),
        frame=frame,
        world_xyz=world_xyz,
        sigma_m=float(sigma_m),
        status=status,
        immovable=status == "human_reviewed",
        source="ray_intersection_z_ball_radius",
        details={"pixel_xy": [float(xy[0]), float(xy[1])], "ball_radius_m": ball_radius_m},
    )


def order_event_anchors(anchors: Sequence[AnchorEvent]) -> list[AnchorEvent]:
    """Sort anchors by time and prefer human-reviewed duplicates."""

    priority = {"human_reviewed": 0, "contact_prior": 1, "solver_proposed": 2}
    ordered = sorted(
        anchors,
        key=lambda item: (
            item.t,
            priority.get(item.status, 9),
            item.frame,
            item.anchor_id,
        ),
    )
    deduped: list[AnchorEvent] = []
    for anchor in ordered:
        duplicate_index = None
        for index, existing in enumerate(deduped):
            if anchor.kind != existing.kind:
                continue
            if abs(anchor.t - existing.t) <= 1e-6 or abs(anchor.frame - existing.frame) <= 1:
                duplicate_index = index
                break
        if duplicate_index is None:
            deduped.append(anchor)
            continue
        existing = deduped[duplicate_index]
        if priority.get(anchor.status, 9) < priority.get(existing.status, 9):
            deduped[duplicate_index] = anchor
    return sorted(deduped, key=lambda item: (item.t, item.frame, priority.get(item.status, 9)))


def fit_flight_segment(
    *,
    segment_id: int,
    start_anchor: AnchorEvent,
    end_anchor: AnchorEvent,
    observations: Sequence[BallObservation],
    calibration: Mapping[str, Any],
    physics: PhysicsParameters | None = None,
    config: BallArcSolverConfig | None = None,
    net_plane: Mapping[str, Any] | None = None,
    max_nfev: int | None = None,
) -> FlightSegmentFit:
    """Fit one event-bounded free-flight segment."""

    cfg = config or BallArcSolverConfig()
    phys = physics or PhysicsParameters()
    if end_anchor.t - start_anchor.t < cfg.min_segment_dt_s:
        return _blocked_segment(segment_id, start_anchor, end_anchor, "duration_below_minimum")
    try:
        import numpy as np
        from scipy.optimize import least_squares
    except ImportError as exc:
        return _blocked_segment(segment_id, start_anchor, end_anchor, f"missing_numeric_dependency:{exc}")

    observations = tuple(
        obs for obs in observations if start_anchor.t - 1e-9 <= obs.t <= end_anchor.t + 1e-9 and obs.visible
    )
    dt = end_anchor.t - start_anchor.t
    initial_velocity = _initial_velocity_guess(start_anchor.world_xyz, end_anchor.world_xyz, dt, phys)
    start_sigma = _anchor_sigma_m(start_anchor, cfg)
    end_sigma = _anchor_sigma_m(end_anchor, cfg)
    p0 = np.asarray(start_anchor.world_xyz, dtype=float)
    if not np.all(np.isfinite(p0)) or not np.all(np.isfinite(np.asarray(end_anchor.world_xyz, dtype=float))):
        return _blocked_segment(segment_id, start_anchor, end_anchor, "nonfinite_anchor")
    initial = np.asarray([p0[0], p0[1], p0[2], *initial_velocity], dtype=float)
    relax = max(0.02, cfg.anchor_relax_sigma_multiplier * start_sigma)
    lower = np.asarray([p0[0] - relax, p0[1] - relax, max(0.0, p0[2] - relax), -60.0, -60.0, -60.0])
    upper = np.asarray([p0[0] + relax, p0[1] + relax, p0[2] + relax, 60.0, 60.0, 60.0])
    if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)) or not np.all(lower < upper):
        return _blocked_segment(segment_id, start_anchor, end_anchor, "invalid_segment_bounds")
    initial = np.minimum(np.maximum(initial, lower + 1e-9), upper - 1e-9)
    times = [start_anchor.t, end_anchor.t, *[obs.t for obs in observations]]

    def residuals(params: Any) -> Any:
        initial_position = (float(params[0]), float(params[1]), float(params[2]))
        velocity = (float(params[3]), float(params[4]), float(params[5]))
        predicted = _integrate_positions(initial_position, velocity, times, t0=start_anchor.t, physics=phys, config=cfg)
        by_time = {round(t, 9): point for t, point in zip(times, predicted, strict=True)}
        residual: list[float] = []
        residual.extend(_scaled_vec(_sub(initial_position, start_anchor.world_xyz), start_sigma / cfg.endpoint_anchor_weight))
        endpoint = by_time[round(end_anchor.t, 9)]
        residual.extend(_scaled_vec(_sub(endpoint, end_anchor.world_xyz), end_sigma / cfg.endpoint_anchor_weight))
        for obs in observations:
            projected = _project_world_point(calibration, by_time[round(obs.t, 9)])
            sigma_px = cfg.robust_pixel_sigma / max(0.35, math.sqrt(max(obs.confidence, 1e-6)))
            residual.append((projected[0] - obs.xy[0]) / sigma_px)
            residual.append((projected[1] - obs.xy[1]) / sigma_px)
            size_residual = _size_depth_residual(
                calibration,
                obs,
                by_time[round(obs.t, 9)],
                physics=phys,
                config=cfg,
                sigma_floor_m=cfg.size_depth_sigma_m,
            )
            if size_residual is not None:
                residual.append(size_residual[0] / size_residual[1])
        if net_plane is not None:
            residual.append(
                _net_soft_residual(
                    initial_position,
                    velocity,
                    start_anchor.t,
                    end_anchor.t,
                    phys,
                    cfg,
                    net_plane,
                )
            )
        return np.asarray(residual, dtype=float)

    result = least_squares(
        residuals,
        initial,
        bounds=(lower, upper),
        loss=cfg.robust_loss,
        f_scale=cfg.robust_f_scale,
        max_nfev=max_nfev or 4000,
    )
    params = result.x if result.success else initial
    initial_position = (float(params[0]), float(params[1]), float(params[2]))
    velocity = (float(params[3]), float(params[4]), float(params[5]))
    obs_errors = _observation_reprojection_errors(
        observations,
        calibration=calibration,
        initial_position=initial_position,
        velocity=velocity,
        t0=start_anchor.t,
        physics=phys,
        config=cfg,
    )
    inlier_frames = tuple(obs.frame for obs in observations if obs_errors.get(obs.frame, math.inf) <= cfg.max_reprojection_inlier_px)
    outlier_frames = tuple(obs.frame for obs in observations if obs_errors.get(obs.frame, 0.0) > cfg.max_reprojection_inlier_px)
    errors = [obs_errors[obs.frame] for obs in observations if obs.frame in obs_errors]
    inlier_errors = [
        obs_errors[obs.frame]
        for obs in observations
        if obs.frame in obs_errors and obs_errors[obs.frame] <= cfg.max_reprojection_inlier_px
    ]
    endpoint_pred = _integrate_positions(initial_position, velocity, [end_anchor.t], t0=start_anchor.t, physics=phys, config=cfg)[0]
    net_clearance = _net_clearance_m(initial_position, velocity, start_anchor.t, end_anchor.t, phys, cfg, net_plane)
    net_ok = None if net_clearance is None else net_clearance >= -cfg.net_clearance_slack_m
    physical = _physical_sanity(
        initial_position,
        velocity,
        start_anchor.t,
        end_anchor.t,
        phys,
        cfg,
        net_clearance,
    )
    size_residuals = _size_residual_distribution(
        observations,
        calibration=calibration,
        initial_position=initial_position,
        velocity=velocity,
        t0=start_anchor.t,
        physics=phys,
        config=cfg,
        sigma_floor_m=cfg.size_depth_sigma_m,
    )
    return FlightSegmentFit(
        segment_id=segment_id,
        status="fit" if result.success else "fit_optimizer_not_converged",
        start_anchor=start_anchor,
        end_anchor=end_anchor,
        initial_position_m=initial_position,
        initial_velocity_mps=velocity,
        observations=observations,
        inlier_frames=inlier_frames,
        outlier_frames=outlier_frames,
        reprojection_errors_px=obs_errors,
        reprojection_rmse_px=_rmse(inlier_errors),
        max_reprojection_error_px=max(errors) if errors else None,
        endpoint_error_m=_distance(endpoint_pred, end_anchor.world_xyz),
        net_clearance_m=net_clearance,
        net_clearance_ok=net_ok,
        physical_sanity=physical,
        size_residuals_m=size_residuals,
    )


def fit_weak_flight_segment(
    *,
    segment_id: int,
    anchor: AnchorEvent,
    observations: Sequence[BallObservation],
    calibration: Mapping[str, Any],
    physics: PhysicsParameters | None = None,
    config: BallArcSolverConfig | None = None,
    net_plane: Mapping[str, Any] | None = None,
    max_nfev: int | None = None,
) -> FlightSegmentFit:
    """Fit a render-only single-anchor segment for otherwise hidden 2D sightings."""

    cfg = config or BallArcSolverConfig()
    phys = physics or PhysicsParameters()
    observations = tuple(obs for obs in observations if obs.visible and obs.t >= anchor.t - 1e-9)
    if len(observations) < cfg.weak_segment_min_observations:
        return _blocked_segment(segment_id, anchor, anchor, "weak_insufficient_observations")
    if observations[-1].t - anchor.t < cfg.min_segment_dt_s:
        return _blocked_segment(segment_id, anchor, anchor, "weak_duration_below_minimum")
    try:
        import numpy as np
        from scipy.optimize import least_squares
    except ImportError as exc:
        return _blocked_segment(segment_id, anchor, anchor, f"missing_numeric_dependency:{exc}")

    p0 = np.asarray(anchor.world_xyz, dtype=float)
    if not np.all(np.isfinite(p0)):
        return _blocked_segment(segment_id, anchor, anchor, "nonfinite_anchor")
    target = _weak_initial_target_from_observations(observations, calibration=calibration, physics=phys) or _ray_plane_target(
        observations[-1],
        calibration=calibration,
        z=max(phys.radius_m, anchor.world_xyz[2]),
    )
    dt = max(observations[-1].t - anchor.t, cfg.min_segment_dt_s)
    if target is None:
        initial_velocity = (0.0, 0.0, 0.0)
    else:
        initial_velocity = _initial_velocity_guess(anchor.world_xyz, target, dt, phys)
    initial = np.asarray([p0[0], p0[1], p0[2], *initial_velocity], dtype=float)
    anchor_sigma = _anchor_sigma_m(anchor, cfg) * cfg.weak_segment_anchor_sigma_multiplier
    relax = max(0.20, anchor_sigma)
    speed_bound = max(cfg.selection_max_speed_mps, cfg.max_plausible_speed_mps)
    lower = np.asarray([p0[0] - relax, p0[1] - relax, max(0.0, p0[2] - relax), -speed_bound, -speed_bound, -speed_bound])
    upper = np.asarray([p0[0] + relax, p0[1] + relax, p0[2] + relax, speed_bound, speed_bound, speed_bound])
    if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)) or not np.all(lower < upper):
        return _blocked_segment(segment_id, anchor, anchor, "invalid_weak_segment_bounds")
    initial = np.minimum(np.maximum(initial, lower + 1e-9), upper - 1e-9)
    times = [anchor.t, *[obs.t for obs in observations]]

    def residuals(params: Any) -> Any:
        initial_position = (float(params[0]), float(params[1]), float(params[2]))
        velocity = (float(params[3]), float(params[4]), float(params[5]))
        predicted = _integrate_positions(initial_position, velocity, times, t0=anchor.t, physics=phys, config=cfg)
        by_time = {round(t, 9): point for t, point in zip(times, predicted, strict=True)}
        residual: list[float] = []
        residual.extend(_scaled_vec(_sub(initial_position, anchor.world_xyz), anchor_sigma))
        for obs in observations:
            point = by_time[round(obs.t, 9)]
            projected = _project_world_point(calibration, point)
            sigma_px = (cfg.robust_pixel_sigma * cfg.weak_segment_pixel_sigma_multiplier) / max(
                0.35,
                math.sqrt(max(obs.confidence, 1e-6)),
            )
            residual.append((projected[0] - obs.xy[0]) / sigma_px)
            residual.append((projected[1] - obs.xy[1]) / sigma_px)
            size_residual = _size_depth_residual(
                calibration,
                obs,
                point,
                physics=phys,
                config=cfg,
                sigma_floor_m=cfg.weak_size_depth_sigma_m,
            )
            if size_residual is not None:
                residual.append(size_residual[0] / size_residual[1])
        if net_plane is not None:
            residual.append(
                _net_soft_residual(
                    initial_position,
                    velocity,
                    anchor.t,
                    observations[-1].t,
                    phys,
                    cfg,
                    net_plane,
                )
            )
        return np.asarray(residual, dtype=float)

    result = least_squares(
        residuals,
        initial,
        bounds=(lower, upper),
        loss=cfg.robust_loss,
        f_scale=cfg.robust_f_scale,
        max_nfev=max_nfev or cfg.weak_segment_max_nfev,
    )
    params = result.x if result.success else initial
    initial_position = (float(params[0]), float(params[1]), float(params[2]))
    velocity = (float(params[3]), float(params[4]), float(params[5]))
    obs_errors = _observation_reprojection_errors(
        observations,
        calibration=calibration,
        initial_position=initial_position,
        velocity=velocity,
        t0=anchor.t,
        physics=phys,
        config=cfg,
    )
    weak_threshold = cfg.max_reprojection_inlier_px * cfg.weak_segment_pixel_sigma_multiplier
    inlier_frames = tuple(obs.frame for obs in observations if obs_errors.get(obs.frame, math.inf) <= weak_threshold)
    outlier_frames = tuple(obs.frame for obs in observations if obs_errors.get(obs.frame, 0.0) > weak_threshold)
    errors = [obs_errors[obs.frame] for obs in observations if obs.frame in obs_errors]
    inlier_errors = [
        obs_errors[obs.frame]
        for obs in observations
        if obs.frame in obs_errors and obs_errors[obs.frame] <= weak_threshold
    ]
    endpoint_t = observations[-1].t
    endpoint_pred = _integrate_positions(initial_position, velocity, [endpoint_t], t0=anchor.t, physics=phys, config=cfg)[0]
    end_anchor = AnchorEvent(
        anchor_id=f"weak_ray_endpoint_{segment_id:03d}_{observations[-1].frame:06d}",
        kind="weak_ray_endpoint",
        t=endpoint_t,
        frame=observations[-1].frame,
        world_xyz=endpoint_pred,
        sigma_m=max(anchor_sigma, cfg.weak_size_depth_sigma_m),
        status="weak_ray_endpoint",
        immovable=False,
        source="single_anchor_ray_size_depth_fit",
        details={
            "anchor_id": anchor.anchor_id,
            "observation_count": len(observations),
            "size_observation_count": sum(1 for obs in observations if obs.diameter_px is not None),
            "render_only": True,
            "low_confidence": True,
        },
    )
    net_clearance = _net_clearance_m(initial_position, velocity, anchor.t, endpoint_t, phys, cfg, net_plane)
    net_ok = None if net_clearance is None else net_clearance >= -cfg.net_clearance_slack_m
    physical = _physical_sanity(initial_position, velocity, anchor.t, endpoint_t, phys, cfg, net_clearance)
    size_residuals = _size_residual_distribution(
        observations,
        calibration=calibration,
        initial_position=initial_position,
        velocity=velocity,
        t0=anchor.t,
        physics=phys,
        config=cfg,
        sigma_floor_m=cfg.weak_size_depth_sigma_m,
    )
    return FlightSegmentFit(
        segment_id=segment_id,
        status="fit_weak" if result.success else "fit_weak_optimizer_not_converged",
        start_anchor=anchor,
        end_anchor=end_anchor,
        initial_position_m=initial_position,
        initial_velocity_mps=velocity,
        observations=observations,
        inlier_frames=inlier_frames,
        outlier_frames=outlier_frames,
        reprojection_errors_px=obs_errors,
        reprojection_rmse_px=_rmse(inlier_errors),
        max_reprojection_error_px=max(errors) if errors else None,
        endpoint_error_m=0.0,
        net_clearance_m=net_clearance,
        net_clearance_ok=net_ok,
        physical_sanity=physical,
        size_residuals_m=size_residuals,
    )


def solve_ball_arc_track(
    *,
    ball_track: Mapping[str, Any],
    calibration: Mapping[str, Any],
    ball_sizes: Mapping[str, Any] | None = None,
    contact_windows: Mapping[str, Any] | None = None,
    skeleton3d: Mapping[str, Any] | None = None,
    reviewed_bounces: Mapping[str, Any] | None = None,
    rally_spans: Mapping[str, Any] | None = None,
    net_plane: Mapping[str, Any] | None = None,
    extra_anchors: Sequence[AnchorEvent] = (),
    physics: PhysicsParameters | None = None,
    config: BallArcSolverConfig | None = None,
    clip_id: str | None = None,
) -> dict[str, Any]:
    """Build a render-only ball_track_arc_solved artifact."""

    cfg = config or BallArcSolverConfig()
    phys = physics or PhysicsParameters()
    frames = _frames(ball_track)
    fps = _payload_fps(ball_track, frames)
    observations = _ball_observations(frames, fps=fps, ball_sizes=ball_sizes)
    observation_by_frame = {obs.frame: obs for obs in observations}
    anchors: list[AnchorEvent] = list(extra_anchors)
    anchors.extend(
        _reviewed_bounce_anchors(
            reviewed_bounces,
            calibration=calibration,
            observations_by_frame=observation_by_frame,
            physics=phys,
            config=cfg,
        )
    )
    anchors.extend(
        _contact_anchors(
            contact_windows,
            skeleton3d,
            calibration=calibration,
            observations=observations,
            config=cfg,
        )
    )
    candidate_anchors = [_anchor_with_sigma_floor(anchor, cfg) for anchor in _filter_anchors_to_rally_spans(order_event_anchors(anchors), rally_spans)]
    if cfg.enable_event_subset_selection:
        anchors, segments, event_selection = _select_event_subset(
            candidate_anchors,
            observations=observations,
            calibration=calibration,
            physics=phys,
            config=cfg,
            net_plane=net_plane,
            rally_spans=rally_spans,
        )
    else:
        anchors = candidate_anchors
        segments = _fit_segments_from_anchors(
            anchors,
            observations=observations,
            calibration=calibration,
            physics=phys,
            config=cfg,
            net_plane=net_plane,
        )
        event_selection = _event_selection_passthrough(anchors)
    discovered: list[AnchorEvent] = []
    if cfg.enable_event_discovery:
        discovered = _discover_bounce_anchors(
            segments,
            observations_by_frame=observation_by_frame,
            calibration=calibration,
            physics=phys,
            config=cfg,
            existing_anchors=anchors,
        )
        if discovered:
            candidate_anchors = [_anchor_with_sigma_floor(anchor, cfg) for anchor in order_event_anchors([*candidate_anchors, *discovered])]
            if cfg.enable_event_subset_selection:
                anchors, segments, event_selection = _select_event_subset(
                    candidate_anchors,
                    observations=observations,
                    calibration=calibration,
                    physics=phys,
                    config=cfg,
                    net_plane=net_plane,
                    rally_spans=rally_spans,
                )
            else:
                anchors = candidate_anchors
                segments = _fit_segments_from_anchors(
                    anchors,
                    observations=observations,
                    calibration=calibration,
                    physics=phys,
                    config=cfg,
                    net_plane=net_plane,
                )
                event_selection = _event_selection_passthrough(anchors)
    confident_segments = list(segments)
    weak_segments, weak_validation = _build_weak_segments(
        anchors,
        confident_segments,
        observations=observations,
        calibration=calibration,
        physics=phys,
        config=cfg,
        net_plane=net_plane,
    )
    all_segments = [*confident_segments, *weak_segments]
    frame_payloads, coverage = _solved_frames(
        frames,
        fps=fps,
        segments=all_segments,
        physics=phys,
        config=cfg,
    )
    loo = _leave_one_out_validation(
        confident_segments,
        calibration=calibration,
        physics=phys,
        config=cfg,
    )
    loo_size_ablation = _leave_one_out_validation(
        confident_segments,
        calibration=calibration,
        physics=phys,
        config=replace(cfg, enable_size_depth_residual=False),
    )
    physical_summary = _physical_summary(confident_segments, config=cfg)
    size_summary = _size_depth_validation_summary(confident_segments, weak_segments)
    status = "ran"
    kill_reasons: list[str] = []
    loo_median = loo["ray_distance_m"]["median"]
    if cfg.baseline_loo_median_m is not None and loo_median is not None and loo_median > cfg.baseline_loo_median_m:
        status = "experimental_off"
        kill_reasons.append(
            f"loo_median_m {loo_median:.6f} regressed beyond baseline {cfg.baseline_loo_median_m:.6f}"
        )
    violation_fraction = physical_summary["violation_fraction"]
    if violation_fraction is not None and violation_fraction > cfg.max_physical_violation_fraction:
        status = "experimental_off"
        kill_reasons.append(
            f"physical_sanity_violation_fraction {violation_fraction:.6f} exceeds {cfg.max_physical_violation_fraction:.6f}"
        )
    try:
        physics3d_summary = reconstruct_bounce_arcs_from_image_track(
            ball_track,
            calibration,
            max_reprojection_rmse_px=12.0,
            max_fit_samples=13,
        ).summary()
    except Exception as exc:
        physics3d_summary = {"status": "not_run", "notes": [str(exc)]}

    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "lane": LANE,
        "clip_id": str(clip_id or ""),
        "status": status,
        "kill_reasons": kill_reasons,
        "source": SOURCE,
        "render_only": True,
        "not_for_detection_metrics": True,
        "trusted_for_ball_detection_metrics": False,
        "protected_eval_labels_used": False,
        "policy": {
            "render_only": True,
            "not_for_detection_metrics": True,
            "feeds_world_only": True,
            "outdoor_indoor_labels_read": False,
            "notes": [
                "Monocular sightings are fit as camera rays through event-anchored physics arcs.",
                "Output is for world rendering and diagnostics only.",
                "Do not use this artifact for BALL detector metrics, gates, training, or promotion.",
            ],
        },
        "physics_parameters": phys.summary(),
        "config": _config_summary(cfg),
        "anchors": [anchor.to_json() for anchor in anchors],
        "event_selection": event_selection,
        "segments": [segment.to_json() for segment in all_segments],
        "frames": frame_payloads,
        "summary": {
            "input_frame_count": len(frames),
            "input_visible_count": len(observations),
            "size_observation_count": sum(1 for obs in observations if obs.diameter_px is not None),
            "coverage_world_xyz_count": coverage["coverage_world_xyz_count"],
            "confident_coverage_world_xyz_count": coverage["coverage_world_xyz_count"] - coverage["arc_weak_count"],
            "anchored_measured_count": coverage["anchored_measured_count"],
            "arc_interpolated_count": coverage["arc_interpolated_count"],
            "arc_extrapolated_count": coverage["arc_extrapolated_count"],
            "arc_weak_count": coverage["arc_weak_count"],
            "hidden_count": coverage["hidden_count"],
            "segment_count": len(all_segments),
            "confident_segment_count": len(confident_segments),
            "weak_segment_count": len(weak_segments),
            "fit_segment_count": sum(1 for segment in all_segments if segment.status.startswith("fit")),
            "fp_sightings_pruned_count": sum(segment.outlier_count for segment in confident_segments),
            "human_reviewed_bounce_count": sum(1 for anchor in anchors if anchor.kind == "bounce" and anchor.status == "human_reviewed"),
            "discovered_bounce_count": sum(1 for anchor in anchors if anchor.kind == "bounce" and anchor.status == "solver_proposed"),
            "contact_anchor_count": sum(1 for anchor in anchors if anchor.kind == "contact"),
            "selected_event_count": event_selection["selected_count"],
            "selected_optional_event_count": event_selection["selected_optional_count"],
            "rejected_optional_event_count": event_selection["rejected_optional_count"],
        },
        "validation": {
            "leave_one_out": loo,
            "leave_one_out_size_ablation": loo_size_ablation,
            "size_depth_residuals_m": size_summary,
            "weak_segments": weak_validation,
            "physical_sanity": physical_summary,
            "ball_physics3d_reference": physics3d_summary,
        },
    }


def pixel_ray_world(
    calibration: Mapping[str, Any],
    xy: Sequence[float],
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Return camera origin and unit ray direction in world coordinates."""

    u, v = _xy_required(xy)
    camera = _camera_arrays(calibration)
    fx = float(_intrinsics(calibration)["fx"])
    fy = float(_intrinsics(calibration)["fy"])
    cx = float(_intrinsics(calibration)["cx"])
    cy = float(_intrinsics(calibration)["cy"])
    camera_ray = (float((u - cx) / fx), float((v - cy) / fy), 1.0)
    rotation_t = _transpose3(camera["rotation"])
    origin = _mat_vec(rotation_t, _scale(tuple(camera["translation"]), -1.0))
    direction = _normalize(_mat_vec(rotation_t, camera_ray))
    return origin, direction


def intersect_ray_z(
    origin: tuple[float, float, float],
    direction: tuple[float, float, float],
    z: float,
) -> tuple[float, float, float]:
    if abs(direction[2]) < 1e-12:
        raise ValueError("ray is nearly parallel to z plane")
    scale = (float(z) - origin[2]) / direction[2]
    return (origin[0] + direction[0] * scale, origin[1] + direction[1] * scale, float(z))


def anchor_sigma_for_bounce(
    calibration: Mapping[str, Any],
    xy: Sequence[float],
    *,
    base_sigma_m: float,
) -> float:
    gsd_sigma = _gsd_sigma(calibration, xy)
    finite_diff = _ray_plane_pixel_sigma(calibration, xy, BALL_RADIUS_M)
    components = [base_sigma_m]
    if gsd_sigma is not None:
        components.append(gsd_sigma)
    if finite_diff is not None:
        components.append(min(0.15, finite_diff))
    reproj_p95 = _float_or_none(calibration.get("reprojection_error_px", {}).get("p95") if isinstance(calibration.get("reprojection_error_px"), Mapping) else None)
    if reproj_p95 is not None:
        components.append(min(0.18, 0.004 * reproj_p95))
    sigma = math.sqrt(sum(value * value for value in components))
    return max(base_sigma_m, min(0.35, sigma))


def _fit_segments_from_anchors(
    anchors: Sequence[AnchorEvent],
    *,
    observations: Sequence[BallObservation],
    calibration: Mapping[str, Any],
    physics: PhysicsParameters,
    config: BallArcSolverConfig,
    net_plane: Mapping[str, Any] | None,
) -> list[FlightSegmentFit]:
    segments: list[FlightSegmentFit] = []
    for start, end in zip(anchors, anchors[1:]):
        segment = _fit_anchor_pair(
            len(segments),
            start,
            end,
            observations=observations,
            calibration=calibration,
            physics=physics,
            config=config,
            net_plane=net_plane,
            block_insufficient_observations=False,
        )
        if segment is None:
            continue
        segments.append(segment)
    return segments


def _fit_anchor_pair(
    segment_id: int,
    start: AnchorEvent,
    end: AnchorEvent,
    *,
    observations: Sequence[BallObservation],
    calibration: Mapping[str, Any],
    physics: PhysicsParameters,
    config: BallArcSolverConfig,
    net_plane: Mapping[str, Any] | None,
    block_insufficient_observations: bool,
    max_nfev: int | None = None,
) -> FlightSegmentFit | None:
    if end.t - start.t < config.min_segment_dt_s:
        return _blocked_segment(segment_id, start, end, "duration_below_minimum")
    segment_observations = [obs for obs in observations if start.t - 1e-9 <= obs.t <= end.t + 1e-9]
    if len(segment_observations) < config.min_segment_observations:
        if block_insufficient_observations:
            return _blocked_segment(segment_id, start, end, "insufficient_observations")
        return None
    return fit_flight_segment(
        segment_id=segment_id,
        start_anchor=start,
        end_anchor=end,
        observations=segment_observations,
        calibration=calibration,
        physics=physics,
        config=config,
        net_plane=net_plane,
        max_nfev=max_nfev,
    )


def _select_event_subset(
    anchors: Sequence[AnchorEvent],
    *,
    observations: Sequence[BallObservation],
    calibration: Mapping[str, Any],
    physics: PhysicsParameters,
    config: BallArcSolverConfig,
    net_plane: Mapping[str, Any] | None,
    rally_spans: Mapping[str, Any] | None,
) -> tuple[list[AnchorEvent], list[FlightSegmentFit], dict[str, Any]]:
    endpoints = _rally_endpoint_anchors(
        observations,
        calibration=calibration,
        physics=physics,
        config=config,
        rally_spans=rally_spans,
        existing_anchors=anchors,
    )
    candidates = [_anchor_with_sigma_floor(anchor, config) for anchor in order_event_anchors([*anchors, *endpoints])]
    mandatory = [anchor for anchor in candidates if _is_mandatory_event(anchor)]
    optional = [anchor for anchor in candidates if not _is_mandatory_event(anchor)]
    selected = order_event_anchors(mandatory)
    selected_keys = {_anchor_key(anchor) for anchor in selected}
    fit_cache: dict[tuple[str, str], FlightSegmentFit | None] = {}
    selected_rationales: dict[str, dict[str, Any]] = {
        _anchor_key(anchor): {"selection": "mandatory", "reason": _mandatory_reason(anchor)}
        for anchor in selected
    }
    rejected_rationales: dict[str, dict[str, Any]] = {}

    while True:
        best_anchor: AnchorEvent | None = None
        best_eval: dict[str, Any] | None = None
        for candidate in optional:
            key = _anchor_key(candidate)
            if key in selected_keys:
                continue
            evaluation = _evaluate_candidate_event(
                candidate,
                selected,
                observations=observations,
                calibration=calibration,
                physics=physics,
                config=config,
                net_plane=net_plane,
                fit_cache=fit_cache,
            )
            rejected_rationales[key] = evaluation
            if not evaluation.get("accepted"):
                continue
            if best_eval is None or float(evaluation["score_gain"]) > float(best_eval["score_gain"]):
                best_anchor = candidate
                best_eval = evaluation
        if best_anchor is None or best_eval is None:
            break
        selected.append(best_anchor)
        selected = order_event_anchors(selected)
        key = _anchor_key(best_anchor)
        selected_keys.add(key)
        selected_rationales[key] = {**best_eval, "selection": "selected_optional"}
        rejected_rationales.pop(key, None)

    segments = _fit_segments_from_anchors(
        selected,
        observations=observations,
        calibration=calibration,
        physics=physics,
        config=config,
        net_plane=net_plane,
    )
    selected, segments, endpoint_rejections = _prune_implausible_weak_endpoints(
        selected,
        segments,
        observations=observations,
        calibration=calibration,
        physics=physics,
        config=config,
        net_plane=net_plane,
    )
    selected_payload = [
        _event_selection_payload(anchor, selected_rationales.get(_anchor_key(anchor), {}), selected=True)
        for anchor in selected
    ]
    rejected_payload = [
        _event_selection_payload(anchor, rejected_rationales.get(_anchor_key(anchor), {"reason": "not_selected"}), selected=False)
        for anchor in optional
        if _anchor_key(anchor) not in selected_keys
    ]
    rejected_payload.extend(endpoint_rejections)
    rejected_optional_count = sum(1 for item in rejected_payload if item["selection"] != "rejected_endpoint")
    report = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "racketsport_ball_arc_events_selected",
        "candidate_prediction": True,
        "not_ground_truth": True,
        "selection_policy": {
            "mandatory": "human-reviewed bounces and weak rally endpoints",
            "optional": "fused contacts and solver-proposed bounces",
            "selection_max_speed_mps": config.selection_max_speed_mps,
            "min_residual_reduction": config.selection_min_residual_reduction,
            "split_penalty": config.selection_split_penalty,
        },
        "selected_count": len(selected_payload),
        "selected_optional_count": sum(1 for item in selected_payload if item["selection"] == "selected_optional"),
        "rejected_count": len(rejected_payload),
        "rejected_optional_count": rejected_optional_count,
        "selected": selected_payload,
        "rejected": rejected_payload,
    }
    return selected, segments, report


def _evaluate_candidate_event(
    candidate: AnchorEvent,
    selected: Sequence[AnchorEvent],
    *,
    observations: Sequence[BallObservation],
    calibration: Mapping[str, Any],
    physics: PhysicsParameters,
    config: BallArcSolverConfig,
    net_plane: Mapping[str, Any] | None,
    fit_cache: dict[tuple[str, str], FlightSegmentFit | None],
) -> dict[str, Any]:
    neighbors = _selected_neighbors(candidate, selected)
    if neighbors is None:
        return {"accepted": False, "reason": "outside_selected_interval", "confidence": _candidate_confidence(candidate)}
    left, right = neighbors
    endpoint_speed_left = _endpoint_speed_floor(left, candidate)
    endpoint_speed_right = _endpoint_speed_floor(candidate, right)
    endpoint_speed_cap = config.selection_max_speed_mps * 1.15
    if endpoint_speed_left is None or endpoint_speed_right is None:
        return {"accepted": False, "reason": "duration_below_minimum", "confidence": _candidate_confidence(candidate)}
    if endpoint_speed_left > endpoint_speed_cap or endpoint_speed_right > endpoint_speed_cap:
        return {
            "accepted": False,
            "reason": "endpoint_speed_exceeds_selection_cap",
            "confidence": _candidate_confidence(candidate),
            "endpoint_speed_floor_mps": [_round(endpoint_speed_left, 6), _round(endpoint_speed_right, 6)],
            "selection_max_speed_mps": config.selection_max_speed_mps,
        }
    parent = _cached_selection_fit(
        fit_cache,
        0,
        left,
        right,
        observations=observations,
        calibration=calibration,
        physics=physics,
        config=config,
    )
    child_left = _cached_selection_fit(
        fit_cache,
        0,
        left,
        candidate,
        observations=observations,
        calibration=calibration,
        physics=physics,
        config=config,
    )
    child_right = _cached_selection_fit(
        fit_cache,
        1,
        candidate,
        right,
        observations=observations,
        calibration=calibration,
        physics=physics,
        config=config,
    )
    parent_score = _segment_selection_score(parent, config)
    child_score = _segment_selection_score(child_left, config) + _segment_selection_score(child_right, config) + config.selection_split_penalty
    confidence = _candidate_confidence(candidate)
    payload = {
        "accepted": False,
        "reason": "not_selected",
        "confidence": _round(confidence, 6),
        "parent_start": left.anchor_id,
        "parent_end": right.anchor_id,
        "parent_score": _optional_round(parent_score if math.isfinite(parent_score) else None, 6),
        "split_score": _optional_round(child_score if math.isfinite(child_score) else None, 6),
        "score_gain": None,
        "residual_reduction": None,
        "child_initial_speed_mps": [
            _optional_round(_segment_speed(child_left), 6),
            _optional_round(_segment_speed(child_right), 6),
        ],
    }
    if not _segment_plausible_for_selection(child_left, config, physics, net_plane) or not _segment_plausible_for_selection(child_right, config, physics, net_plane):
        payload["reason"] = "split_not_physically_plausible"
        return payload
    if not math.isfinite(parent_score) or parent_score <= 0.0:
        payload.update({"accepted": True, "reason": "parent_unfit_children_plausible", "score_gain": None, "residual_reduction": None})
        return payload
    score_gain = parent_score - child_score
    residual_reduction = score_gain / max(parent_score, 1e-9)
    payload["score_gain"] = _round(score_gain * confidence, 6)
    payload["residual_reduction"] = _round(residual_reduction, 6)
    if score_gain > 0.0 and residual_reduction >= config.selection_min_residual_reduction:
        payload.update({"accepted": True, "reason": "reduced_residual_and_plausible"})
    else:
        payload["reason"] = "insufficient_residual_reduction"
    return payload


def _prune_implausible_weak_endpoints(
    selected: Sequence[AnchorEvent],
    segments: Sequence[FlightSegmentFit],
    *,
    observations: Sequence[BallObservation],
    calibration: Mapping[str, Any],
    physics: PhysicsParameters,
    config: BallArcSolverConfig,
    net_plane: Mapping[str, Any] | None,
) -> tuple[list[AnchorEvent], list[FlightSegmentFit], list[dict[str, Any]]]:
    anchors = list(selected)
    rejected: list[dict[str, Any]] = []
    current_segments = list(segments)
    while True:
        bad_segment = next(
            (
                segment
                for segment in current_segments
                if not _segment_plausible_for_selection(segment, config, physics, net_plane)
                and (segment.start_anchor.kind == "rally_endpoint" or segment.end_anchor.kind == "rally_endpoint")
            ),
            None,
        )
        if bad_segment is None:
            return anchors, current_segments, rejected
        endpoint = bad_segment.start_anchor if bad_segment.start_anchor.kind == "rally_endpoint" else bad_segment.end_anchor
        rejected.append(
            _event_selection_payload(
                endpoint,
                {
                    "selection": "rejected_endpoint",
                    "reason": "weak_endpoint_adjacent_segment_implausible",
                    "child_initial_speed_mps": [_segment_speed(bad_segment)],
                },
                selected=False,
            )
        )
        anchors = [anchor for anchor in anchors if _anchor_key(anchor) != _anchor_key(endpoint)]
        current_segments = _fit_segments_from_anchors(
            anchors,
            observations=observations,
            calibration=calibration,
            physics=physics,
            config=config,
            net_plane=net_plane,
        )


def _build_weak_segments(
    anchors: Sequence[AnchorEvent],
    confident_segments: Sequence[FlightSegmentFit],
    *,
    observations: Sequence[BallObservation],
    calibration: Mapping[str, Any],
    physics: PhysicsParameters,
    config: BallArcSolverConfig,
    net_plane: Mapping[str, Any] | None,
) -> tuple[list[FlightSegmentFit], dict[str, Any]]:
    if not config.enable_weak_segments:
        return [], _weak_segment_report([], [], mode="disabled")
    hidden_groups = _hidden_observation_groups(
        observations,
        confident_segments,
        max_gap_s=config.weak_segment_max_gap_s,
    )
    weak_segments: list[FlightSegmentFit] = []
    rejected: list[dict[str, Any]] = []
    ordered_anchors = sorted(anchors, key=lambda item: (item.t, item.frame, item.anchor_id))
    next_segment_id = len(confident_segments)
    for group in hidden_groups:
        if len(group) < config.weak_segment_min_observations:
            rejected.append(_weak_rejection_payload(group, "insufficient_observations"))
            continue
        anchor = _weak_anchor_before_group(ordered_anchors, group)
        if anchor is None:
            rejected.append(_weak_rejection_payload(group, "no_prior_anchor"))
            continue
        weak = fit_weak_flight_segment(
            segment_id=next_segment_id,
            anchor=anchor,
            observations=group,
            calibration=calibration,
            physics=physics,
            config=config,
            net_plane=net_plane,
            max_nfev=config.weak_segment_max_nfev,
        )
        if not _segment_plausible_for_weak_render(weak, config, physics, net_plane):
            rejected.append(
                {
                    **_weak_rejection_payload(group, "weak_segment_implausible"),
                    "status": weak.status,
                    "initial_speed_mps": _optional_round(_segment_speed(weak), 6),
                    "physical_sanity": dict(weak.physical_sanity),
                }
            )
            continue
        weak_segments.append(weak)
        next_segment_id += 1
    return weak_segments, _weak_segment_report(weak_segments, rejected, mode="enabled")


def _hidden_observation_groups(
    observations: Sequence[BallObservation],
    confident_segments: Sequence[FlightSegmentFit],
    *,
    max_gap_s: float,
) -> list[list[BallObservation]]:
    hidden: list[BallObservation] = []
    for obs in observations:
        segment = _segment_for_time(confident_segments, obs.t)
        if segment is None or not segment.status.startswith("fit"):
            hidden.append(obs)
    groups: list[list[BallObservation]] = []
    for obs in sorted(hidden, key=lambda item: (item.t, item.frame)):
        if not groups or obs.t - groups[-1][-1].t > max_gap_s:
            groups.append([obs])
        else:
            groups[-1].append(obs)
    return groups


def _weak_anchor_before_group(anchors: Sequence[AnchorEvent], group: Sequence[BallObservation]) -> AnchorEvent | None:
    if not group:
        return None
    first_t = group[0].t
    candidates = [anchor for anchor in anchors if anchor.t <= first_t + 1e-9]
    if not candidates:
        return None
    return max(candidates, key=lambda anchor: (anchor.t, anchor.frame))


def _segment_plausible_for_weak_render(
    segment: FlightSegmentFit | None,
    config: BallArcSolverConfig,
    physics: PhysicsParameters,
    net_plane: Mapping[str, Any] | None,
) -> bool:
    if segment is None or segment.status != "fit_weak":
        return False
    physical = segment.physical_sanity
    speed = _float_or_none(physical.get("initial_speed_mps"))
    apex = _float_or_none(physical.get("apex_height_m"))
    if speed is None or speed > config.selection_max_speed_mps:
        return False
    if apex is None or apex > config.max_plausible_apex_m:
        return False
    clearance = segment.net_clearance_m
    if clearance is None and net_plane is not None:
        clearance = _net_clearance_m(
            segment.initial_position_m,
            segment.initial_velocity_mps,
            segment.start_anchor.t,
            segment.end_anchor.t,
            physics,
            config,
            net_plane,
        )
    if clearance is not None and clearance < -config.net_clearance_slack_m:
        return False
    return not bool(physical.get("violation"))


def _weak_rejection_payload(group: Sequence[BallObservation], reason: str) -> dict[str, Any]:
    return {
        "reason": reason,
        "frame_start": int(group[0].frame) if group else None,
        "frame_end": int(group[-1].frame) if group else None,
        "t0": _round(group[0].t, 9) if group else None,
        "t1": _round(group[-1].t, 9) if group else None,
        "observation_count": len(group),
    }


def _weak_segment_report(
    weak_segments: Sequence[FlightSegmentFit],
    rejected: Sequence[Mapping[str, Any]],
    *,
    mode: str,
) -> dict[str, Any]:
    return {
        "mode": mode,
        "fit_count": len(weak_segments),
        "rejected_count": len(rejected),
        "covered_observation_count": sum(len(segment.observations) for segment in weak_segments),
        "segments": [
            {
                "segment_id": segment.segment_id,
                "frame_start": segment.start_anchor.frame,
                "frame_end": segment.end_anchor.frame,
                "t0": _round(segment.start_anchor.t, 9),
                "t1": _round(segment.end_anchor.t, 9),
                "initial_speed_mps": _optional_round(_segment_speed(segment), 6),
                "observation_count": len(segment.observations),
                "size_observation_count": sum(1 for obs in segment.observations if obs.diameter_px is not None),
                "physical_sanity": dict(segment.physical_sanity),
            }
            for segment in weak_segments
        ],
        "rejected": [dict(item) for item in rejected],
    }


def _cached_selection_fit(
    fit_cache: dict[tuple[str, str], FlightSegmentFit | None],
    segment_id: int,
    start: AnchorEvent,
    end: AnchorEvent,
    *,
    observations: Sequence[BallObservation],
    calibration: Mapping[str, Any],
    physics: PhysicsParameters,
    config: BallArcSolverConfig,
) -> FlightSegmentFit | None:
    key = (_anchor_key(start), _anchor_key(end))
    if key not in fit_cache:
        selection_config = replace(config, enable_size_depth_residual=False)
        fit_cache[key] = _fit_anchor_pair(
            segment_id,
            start,
            end,
            observations=observations,
            calibration=calibration,
            physics=physics,
            config=selection_config,
            net_plane=None,
            block_insufficient_observations=True,
            max_nfev=selection_config.selection_max_nfev,
        )
    segment = fit_cache[key]
    if segment is None or segment.segment_id == segment_id:
        return segment
    return replace(segment, segment_id=segment_id)


def _rally_endpoint_anchors(
    observations: Sequence[BallObservation],
    *,
    calibration: Mapping[str, Any],
    physics: PhysicsParameters,
    config: BallArcSolverConfig,
    rally_spans: Mapping[str, Any] | None,
    existing_anchors: Sequence[AnchorEvent],
) -> list[AnchorEvent]:
    if not observations:
        return []
    spans = _rally_span_ranges(rally_spans, observations)
    anchors: list[AnchorEvent] = []
    for index, (t0, t1) in enumerate(spans):
        in_span = [obs for obs in observations if t0 - 1e-9 <= obs.t <= t1 + 1e-9]
        if not in_span:
            continue
        for role, obs in (("start", in_span[0]), ("end", in_span[-1])):
            if any(
                _is_mandatory_event(anchor)
                and (abs(anchor.t - obs.t) < config.min_segment_dt_s or abs(anchor.frame - obs.frame) <= 1)
                for anchor in existing_anchors
            ):
                continue
            try:
                origin, direction = pixel_ray_world(calibration, obs.xy)
                world_xyz = intersect_ray_z(origin, direction, physics.radius_m)
            except ValueError:
                continue
            anchors.append(
                AnchorEvent(
                    anchor_id=f"rally_endpoint_{index}_{role}_{obs.frame:06d}",
                    kind="rally_endpoint",
                    t=obs.t,
                    frame=obs.frame,
                    world_xyz=world_xyz,
                    sigma_m=config.rally_endpoint_sigma_m,
                    status="rally_endpoint_weak",
                    immovable=True,
                    source="ball_ray_plane_weak_endpoint_prior",
                    details={"role": role, "candidate_prediction": False},
                )
            )
    return anchors


def _rally_span_ranges(
    rally_spans: Mapping[str, Any] | None,
    observations: Sequence[BallObservation],
) -> list[tuple[float, float]]:
    if isinstance(rally_spans, Mapping) and isinstance(rally_spans.get("spans"), list):
        parsed: list[tuple[float, float]] = []
        for span in rally_spans["spans"]:
            if not isinstance(span, Mapping):
                continue
            t0 = _float_or_none(span.get("t0"))
            t1 = _float_or_none(span.get("t1"))
            if t0 is not None and t1 is not None and t1 >= t0:
                parsed.append((t0, t1))
        if parsed:
            return parsed
    return [(observations[0].t, observations[-1].t)]


def _is_mandatory_event(anchor: AnchorEvent) -> bool:
    if anchor.kind == "rally_endpoint":
        return True
    if anchor.kind == "bounce" and anchor.status == "human_reviewed":
        return True
    return bool(anchor.immovable and anchor.status != "solver_proposed")


def _mandatory_reason(anchor: AnchorEvent) -> str:
    if anchor.kind == "rally_endpoint":
        return "rally_endpoint"
    if anchor.kind == "bounce" and anchor.status == "human_reviewed":
        return "human_reviewed_bounce"
    return "immovable_anchor"


def _selected_neighbors(candidate: AnchorEvent, selected: Sequence[AnchorEvent]) -> tuple[AnchorEvent, AnchorEvent] | None:
    ordered = sorted(selected, key=lambda anchor: (anchor.t, anchor.frame, anchor.anchor_id))
    left = [anchor for anchor in ordered if anchor.t < candidate.t - 1e-9]
    right = [anchor for anchor in ordered if anchor.t > candidate.t + 1e-9]
    if not left or not right:
        return None
    return left[-1], right[0]


def _endpoint_speed_floor(start: AnchorEvent, end: AnchorEvent) -> float | None:
    dt = end.t - start.t
    if dt <= 1e-9:
        return None
    return _distance(start.world_xyz, end.world_xyz) / dt


def _segment_selection_score(segment: FlightSegmentFit | None, config: BallArcSolverConfig) -> float:
    if segment is None or segment.status != "fit" or not segment.reprojection_errors_px:
        return math.inf
    sigma = max(config.robust_pixel_sigma, 1e-9)
    total = 0.0
    for error in segment.reprojection_errors_px.values():
        scaled = abs(error) / sigma
        if scaled <= config.robust_f_scale:
            total += 0.5 * scaled * scaled
        else:
            total += config.robust_f_scale * (scaled - 0.5 * config.robust_f_scale)
    return total


def _segment_plausible_for_selection(
    segment: FlightSegmentFit | None,
    config: BallArcSolverConfig,
    physics: PhysicsParameters,
    net_plane: Mapping[str, Any] | None,
) -> bool:
    if segment is None or segment.status != "fit":
        return False
    physical = segment.physical_sanity
    speed = _float_or_none(physical.get("initial_speed_mps"))
    apex = _float_or_none(physical.get("apex_height_m"))
    if speed is None or speed > config.selection_max_speed_mps:
        return False
    if apex is None or apex > config.max_plausible_apex_m:
        return False
    clearance = segment.net_clearance_m
    if clearance is None and net_plane is not None:
        clearance = _net_clearance_m(
            segment.initial_position_m,
            segment.initial_velocity_mps,
            segment.start_anchor.t,
            segment.end_anchor.t,
            physics,
            config,
            net_plane,
        )
    if clearance is not None and clearance < -config.net_clearance_slack_m:
        return False
    return True


def _segment_speed(segment: FlightSegmentFit | None) -> float | None:
    if segment is None:
        return None
    return _float_or_none(segment.physical_sanity.get("initial_speed_mps"))


def _candidate_confidence(anchor: AnchorEvent) -> float:
    details = anchor.details or {}
    value = _float_or_none(details.get("event_confidence"))
    if value is None:
        value = _float_or_none(details.get("confidence"))
    if value is None:
        return 0.5 if anchor.kind == "contact" else 0.65
    return max(0.0, min(1.0, value))


def _anchor_key(anchor: AnchorEvent) -> str:
    return f"{anchor.kind}:{anchor.status}:{anchor.anchor_id}:{anchor.frame}:{anchor.t:.9f}"


def _event_selection_payload(anchor: AnchorEvent, rationale: Mapping[str, Any], *, selected: bool) -> dict[str, Any]:
    payload = anchor.to_json()
    payload["selected"] = bool(selected)
    payload["selection"] = str(rationale.get("selection") or ("selected_optional" if selected else "rejected_optional"))
    payload["status"] = anchor.status if selected and _is_mandatory_event(anchor) else "candidate_prediction"
    payload["selection_reason"] = str(rationale.get("reason") or "not_selected")
    payload["candidate_confidence"] = _round(_candidate_confidence(anchor), 6)
    for key in (
        "parent_start",
        "parent_end",
        "parent_score",
        "split_score",
        "score_gain",
        "residual_reduction",
        "child_initial_speed_mps",
        "endpoint_speed_floor_mps",
        "selection_max_speed_mps",
    ):
        if key in rationale:
            payload[key] = rationale[key]
    return payload


def _event_selection_passthrough(anchors: Sequence[AnchorEvent]) -> dict[str, Any]:
    selected = [
        _event_selection_payload(anchor, {"selection": "selected_passthrough", "reason": "event_subset_selection_disabled"}, selected=True)
        for anchor in anchors
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "racketsport_ball_arc_events_selected",
        "candidate_prediction": True,
        "not_ground_truth": True,
        "selection_policy": {"mode": "disabled_passthrough"},
        "selected_count": len(selected),
        "selected_optional_count": 0,
        "rejected_count": 0,
        "rejected_optional_count": 0,
        "selected": selected,
        "rejected": [],
    }


def _discover_bounce_anchors(
    segments: Sequence[FlightSegmentFit],
    *,
    observations_by_frame: Mapping[int, BallObservation],
    calibration: Mapping[str, Any],
    physics: PhysicsParameters,
    config: BallArcSolverConfig,
    existing_anchors: Sequence[AnchorEvent],
) -> list[AnchorEvent]:
    discovered: list[AnchorEvent] = []
    existing_frames = [anchor.frame for anchor in existing_anchors]
    for segment in segments:
        trigger_px = min(config.discovery_reprojection_px, config.max_reprojection_inlier_px * 0.8)
        neighbor_px = min(config.discovery_min_neighbor_px, config.max_reprojection_inlier_px * 0.75)
        if segment.max_reprojection_error_px is None or segment.max_reprojection_error_px < trigger_px:
            continue
        if segment.start_anchor.kind == "bounce" or segment.end_anchor.kind == "bounce":
            continue
        errors = dict(segment.reprojection_errors_px)
        if not errors:
            continue
        ordered = sorted(errors.items(), key=lambda item: item[1], reverse=True)
        for frame, error in ordered:
            if any(abs(frame - existing_frame) <= config.discovery_anchor_separation_frames for existing_frame in existing_frames):
                continue
            obs = observations_by_frame.get(frame)
            if obs is None:
                continue
            neighbor_errors = [
                value
                for other_frame, value in errors.items()
                if other_frame != frame and abs(other_frame - frame) <= 2
            ]
            if neighbor_errors and max(neighbor_errors) < neighbor_px:
                continue
            try:
                anchor = build_bounce_anchor(
                    {"frame": frame, "t": obs.t, "review_id": f"solver_bounce_{frame:06d}", "xy": obs.xy},
                    calibration,
                    ball_radius_m=physics.radius_m,
                    ball_xy=obs.xy,
                    status="solver_proposed",
                    sigma_m=config.proposed_bounce_sigma_m,
                )
            except ValueError:
                continue
            anchor = replace(
                anchor,
                immovable=False,
                details={
                    **dict(anchor.details or {}),
                    "discovery_reprojection_error_px": _round(error, 6),
                    "parent_segment_id": segment.segment_id,
                    "confidence": "lower_confidence_solver_proposed",
                },
            )
            discovered.append(anchor)
            existing_frames.append(frame)
            break
    return discovered


def _reviewed_bounce_anchors(
    reviewed_bounces: Mapping[str, Any] | None,
    *,
    calibration: Mapping[str, Any],
    observations_by_frame: Mapping[int, BallObservation],
    physics: PhysicsParameters,
    config: BallArcSolverConfig,
) -> list[AnchorEvent]:
    if not isinstance(reviewed_bounces, Mapping):
        return []
    if reviewed_bounces.get("status") != "human_reviewed":
        return []
    bounces = reviewed_bounces.get("bounces")
    if not isinstance(bounces, list):
        return []
    anchors: list[AnchorEvent] = []
    for item in bounces:
        if not isinstance(item, Mapping):
            continue
        frame = _frame_from_mapping(item)
        if frame is None:
            continue
        obs = observations_by_frame.get(frame) or _nearest_observation_by_frame(observations_by_frame, frame, max_gap=2)
        if obs is None:
            continue
        try:
            anchors.append(
                build_bounce_anchor(
                    item,
                    calibration,
                    ball_radius_m=physics.radius_m,
                    ball_xy=obs.xy,
                    status="human_reviewed",
                    sigma_m=anchor_sigma_for_bounce(
                        calibration,
                        obs.xy,
                        base_sigma_m=config.reviewed_bounce_base_sigma_m,
                    ),
                )
            )
        except ValueError:
            continue
    return anchors


def _contact_anchors(
    contact_windows: Mapping[str, Any] | None,
    skeleton3d: Mapping[str, Any] | None,
    *,
    calibration: Mapping[str, Any],
    observations: Sequence[BallObservation],
    config: BallArcSolverConfig,
) -> list[AnchorEvent]:
    if not isinstance(contact_windows, Mapping) or not isinstance(skeleton3d, Mapping):
        return []
    events = contact_windows.get("events")
    if not isinstance(events, list):
        return []
    wrist_index = _build_wrist_index(skeleton3d, min_joint_confidence=config.min_joint_confidence)
    if not wrist_index:
        return []
    anchors: list[AnchorEvent] = []
    for index, event in enumerate(events):
        if not isinstance(event, Mapping):
            continue
        if event.get("type", "contact") != "contact":
            continue
        t = _float_or_none(event.get("t"))
        if t is None:
            continue
        player_id = event.get("player_id")
        if player_id is None:
            continue
        selected = _select_contact_wrist(
            wrist_index,
            t=t,
            player_id=player_id,
            observation=_nearest_observation_by_time(observations, t, max_gap_s=0.08),
            calibration=calibration,
            reach_offset_m=config.contact_reach_offset_m,
        )
        if selected is None:
            continue
        frame = _frame_from_mapping(event)
        if frame is None:
            frame = int(round(t * 30.0))
        event_confidence = _float_or_none(event.get("confidence"))
        anchor_id = f"contact_{index:03d}_p{player_id}_{selected['side']}"
        anchors.append(
            AnchorEvent(
                anchor_id=anchor_id,
                kind="contact",
                t=t,
                frame=frame,
                world_xyz=selected["paddle_center"],
                sigma_m=config.contact_anchor_sigma_m,
                status="contact_prior",
                player_id=player_id,
                immovable=False,
                source="skeleton3d_wrist_reach_prior",
                details={
                    "side": selected["side"],
                    "wrist_world_xyz": _vec_json(selected["wrist"]),
                    "elbow_world_xyz": _vec_json(selected["elbow"]),
                    "joint_confidence": _round(selected["confidence"], 6),
                    "event_confidence": _optional_round(event_confidence, 6),
                    "reach_offset_m": config.contact_reach_offset_m,
                },
            )
        )
    return anchors


def _build_wrist_index(
    skeleton3d: Mapping[str, Any],
    *,
    min_joint_confidence: float,
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    skeleton = semanticize_skeleton_payload(skeleton3d) or skeleton3d
    joint_names = skeleton.get("joint_names")
    mapping = _semantic_joint_indexes(joint_names)
    required = {"left_wrist", "right_wrist", "left_elbow", "right_elbow"}
    if not required <= set(mapping):
        return {}
    players = skeleton.get("players")
    if not isinstance(players, Sequence) or isinstance(players, (str, bytes)):
        return {}
    index: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for player in players:
        if not isinstance(player, Mapping):
            continue
        player_id = player.get("id")
        frames = player.get("frames")
        if player_id is None or not isinstance(frames, Sequence) or isinstance(frames, (str, bytes)):
            continue
        for frame in frames:
            if not isinstance(frame, Mapping):
                continue
            t = _float_or_none(frame.get("t"))
            joints = frame.get("joints_world")
            if t is None or not isinstance(joints, Sequence) or isinstance(joints, (str, bytes)):
                continue
            confs = frame.get("joint_conf")
            for side in ("left", "right"):
                wrist_idx = mapping[f"{side}_wrist"]
                elbow_idx = mapping[f"{side}_elbow"]
                wrist = _vec3(joints[wrist_idx] if wrist_idx < len(joints) else None)
                elbow = _vec3(joints[elbow_idx] if elbow_idx < len(joints) else None)
                if wrist is None or elbow is None:
                    continue
                confidence = min(_joint_confidence(confs, wrist_idx), _joint_confidence(confs, elbow_idx))
                if confidence < min_joint_confidence:
                    continue
                index.setdefault((str(player_id), side), []).append(
                    {"t": t, "wrist": wrist, "elbow": elbow, "confidence": confidence}
                )
    return index


def _select_contact_wrist(
    wrist_index: Mapping[tuple[str, str], Sequence[Mapping[str, Any]]],
    *,
    t: float,
    player_id: Any,
    observation: BallObservation | None,
    calibration: Mapping[str, Any],
    reach_offset_m: float,
) -> dict[str, Any] | None:
    candidates: list[tuple[float, dict[str, Any]]] = []
    ray = pixel_ray_world(calibration, observation.xy) if observation is not None else None
    for side in ("right", "left"):
        samples = wrist_index.get((str(player_id), side))
        if not samples:
            continue
        interpolated = _interpolate_wrist_sample(samples, t)
        if interpolated is None:
            continue
        forearm = _sub(interpolated["wrist"], interpolated["elbow"])
        if _norm(forearm) > 1e-9:
            paddle_center = _add(interpolated["wrist"], _scale(_normalize(forearm), reach_offset_m))
        else:
            paddle_center = interpolated["wrist"]
        if ray is not None:
            score = _distance_point_to_ray(paddle_center, ray[0], ray[1])
        else:
            score = abs(float(interpolated["t"]) - t)
        candidates.append((score, {**interpolated, "side": side, "paddle_center": paddle_center}))
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: item[0])[0][1]


def _interpolate_wrist_sample(samples: Sequence[Mapping[str, Any]], t: float) -> dict[str, Any] | None:
    ordered = sorted(samples, key=lambda item: float(item["t"]))
    before = [item for item in ordered if float(item["t"]) <= t]
    after = [item for item in ordered if float(item["t"]) >= t]
    if before and after:
        left = before[-1]
        right = after[0]
        dt = float(right["t"]) - float(left["t"])
        if dt <= 1e-9:
            return dict(left)
        alpha = (t - float(left["t"])) / dt
        return {
            "t": t,
            "wrist": _lerp(left["wrist"], right["wrist"], alpha),
            "elbow": _lerp(left["elbow"], right["elbow"], alpha),
            "confidence": min(float(left["confidence"]), float(right["confidence"])),
        }
    if not ordered:
        return None
    nearest = min(ordered, key=lambda item: abs(float(item["t"]) - t))
    return dict(nearest)


def _solved_frames(
    frames: Sequence[Mapping[str, Any]],
    *,
    fps: float,
    segments: Sequence[FlightSegmentFit],
    physics: PhysicsParameters,
    config: BallArcSolverConfig,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    payloads: list[dict[str, Any]] = []
    counts = {
        "coverage_world_xyz_count": 0,
        "anchored_measured_count": 0,
        "arc_interpolated_count": 0,
        "arc_extrapolated_count": 0,
        "arc_weak_count": 0,
        "hidden_count": 0,
    }
    for frame_index, source_frame in enumerate(frames):
        frame = dict(source_frame)
        t = _float_or_none(frame.get("t"))
        if t is None:
            t = frame_index / max(fps, 1e-9)
        segment = _segment_for_time(segments, t)
        if segment is None or not segment.status.startswith("fit"):
            frame["world_xyz"] = None
            frame["sigma_m"] = None
            frame["band"] = "hidden"
            counts["hidden_count"] += 1
            payloads.append(frame)
            continue
        if segment.status == "fit_weak" and frame.get("visible") is not True:
            frame["world_xyz"] = None
            frame["sigma_m"] = None
            frame["band"] = "hidden"
            counts["hidden_count"] += 1
            payloads.append(frame)
            continue
        predicted = segment.predict(t, physics, config)
        sigma = _frame_sigma(segment, t)
        if segment.status == "fit_weak":
            band = "arc_weak"
        elif frame_index in segment.inlier_frames:
            band = "anchored_measured"
        elif segment.start_anchor.status == "solver_proposed" or segment.end_anchor.status == "solver_proposed":
            band = "arc_extrapolated"
        else:
            band = "arc_interpolated"
        frame["world_xyz"] = _vec_json(predicted)
        frame["sigma_m"] = _round(sigma, 6)
        frame["band"] = band
        frame["source"] = SOURCE
        frame["render_only"] = True
        frame["not_for_detection_metrics"] = True
        frame["arc_solver"] = {
            "lane": LANE,
            "segment_id": segment.segment_id,
            "weak_segment": segment.status == "fit_weak",
            "inlier_sighting": frame_index in segment.inlier_frames,
            "outlier_sighting_pruned": frame_index in segment.outlier_frames,
            "render_only": True,
            "not_for_detection_metrics": True,
        }
        counts["coverage_world_xyz_count"] += 1
        counts[f"{band}_count"] += 1
        payloads.append(frame)
    return payloads, counts


def _leave_one_out_validation(
    segments: Sequence[FlightSegmentFit],
    *,
    calibration: Mapping[str, Any],
    physics: PhysicsParameters,
    config: BallArcSolverConfig,
) -> dict[str, Any]:
    errors_m: list[float] = []
    errors_px: list[float] = []
    skipped: list[dict[str, Any]] = []
    for segment in segments:
        candidate_obs = [
            obs
            for obs in segment.observations
            if obs.frame in segment.inlier_frames and obs.confidence >= config.visible_confidence_min
        ]
        for held_out in candidate_obs:
            retained = [obs for obs in segment.observations if obs.frame != held_out.frame]
            if len(retained) < config.min_segment_observations:
                skipped.append({"frame": held_out.frame, "reason": "insufficient_observations_after_holdout"})
                continue
            refit = fit_flight_segment(
                segment_id=segment.segment_id,
                start_anchor=segment.start_anchor,
                end_anchor=segment.end_anchor,
                observations=retained,
                calibration=calibration,
                physics=physics,
                config=config,
                max_nfev=config.loo_max_nfev,
            )
            if not refit.status.startswith("fit"):
                skipped.append({"frame": held_out.frame, "reason": refit.status})
                continue
            predicted = refit.predict(held_out.t, physics, config)
            origin, direction = pixel_ray_world(calibration, held_out.xy)
            errors_m.append(_distance_point_to_ray(predicted, origin, direction))
            projected = _project_world_point(calibration, predicted)
            errors_px.append(_distance2(projected, held_out.xy))
    return {
        "sample_count": len(errors_m),
        "skipped": skipped,
        "ray_distance_m": _distribution(errors_m),
        "reprojection_error_px": _distribution(errors_px),
    }


def _physical_summary(segments: Sequence[FlightSegmentFit], *, config: BallArcSolverConfig) -> dict[str, Any]:
    items = [dict(segment.physical_sanity) for segment in segments]
    violations = [item for item in items if item.get("violation") is True]
    violation_fraction = (len(violations) / len(items)) if items else None
    return {
        "segment_count": len(items),
        "violation_count": len(violations),
        "violation_fraction": _optional_round(violation_fraction, 6),
        "kill_threshold_fraction": config.max_physical_violation_fraction,
        "segments": items,
    }


def _physical_sanity(
    p0: tuple[float, float, float],
    v0: tuple[float, float, float],
    t0: float,
    t1: float,
    physics: PhysicsParameters,
    config: BallArcSolverConfig,
    net_clearance_m: float | None,
) -> dict[str, Any]:
    speed = _norm(v0)
    times = [t0 + (t1 - t0) * i / 40.0 for i in range(41)]
    points = _integrate_positions(p0, v0, times, t0=t0, physics=physics, config=config)
    apex = max(point[2] for point in points)
    violations: list[str] = []
    if speed < config.min_plausible_speed_mps or speed > config.max_plausible_speed_mps:
        violations.append("initial_speed_outside_plausible_range_mps")
    if apex < physics.radius_m or apex > config.max_plausible_apex_m:
        violations.append("apex_height_implausible")
    if net_clearance_m is not None and net_clearance_m < -config.net_clearance_slack_m:
        violations.append("net_clearance_below_slack")
    return {
        "initial_speed_mps": _round(speed, 6),
        "apex_height_m": _round(apex, 6),
        "net_clearance_m": _optional_round(net_clearance_m, 6),
        "violations": violations,
        "violation": bool(violations),
    }


def _integrate_positions(
    p0: tuple[float, float, float],
    v0: tuple[float, float, float],
    times: Sequence[float],
    *,
    t0: float,
    physics: PhysicsParameters,
    config: BallArcSolverConfig,
) -> list[tuple[float, float, float]]:
    if physics.drag_k_per_m <= 0.0:
        return [_analytic_no_drag_position(p0, v0, float(t) - t0, physics.gravity_mps2) for t in times]
    indexed = sorted(enumerate(times), key=lambda item: float(item[1]))
    positions: list[tuple[float, float, float] | None] = [None] * len(times)
    state = (*p0, *v0)
    current_t = t0
    for index, target_t_raw in indexed:
        target_t = float(target_t_raw)
        if target_t < current_t - 1e-9:
            position = _integrate_positions(p0, v0, [target_t], t0=t0, physics=physics, config=config)[0]
            positions[index] = position
            continue
        while current_t < target_t - 1e-12:
            step = min(config.integrator_max_step_s, target_t - current_t)
            state = _rk4_step(state, step, physics)
            current_t += step
        positions[index] = (state[0], state[1], state[2])
    return [position if position is not None else p0 for position in positions]


def _rk4_step(
    state: tuple[float, float, float, float, float, float],
    dt: float,
    physics: PhysicsParameters,
) -> tuple[float, float, float, float, float, float]:
    def deriv(s: tuple[float, float, float, float, float, float]) -> tuple[float, float, float, float, float, float]:
        vx, vy, vz = s[3], s[4], s[5]
        speed = math.sqrt(vx * vx + vy * vy + vz * vz)
        drag = physics.drag_k_per_m * speed
        return (vx, vy, vz, -drag * vx, -drag * vy, -physics.gravity_mps2 - drag * vz)

    k1 = deriv(state)
    k2 = deriv(_add_state(state, _scale(k1, dt / 2.0)))
    k3 = deriv(_add_state(state, _scale(k2, dt / 2.0)))
    k4 = deriv(_add_state(state, _scale(k3, dt)))
    return tuple(state[i] + dt * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) / 6.0 for i in range(6))  # type: ignore[return-value]


def _project_world_point(
    calibration: Mapping[str, Any],
    world_xyz: tuple[float, float, float],
) -> tuple[float, float]:
    intrinsics = _intrinsics(calibration)
    camera = _camera_arrays(calibration)
    camera_point = _add(_mat_vec(camera["rotation"], world_xyz), camera["translation"])
    depth = camera_point[2] if abs(camera_point[2]) > 1e-9 else 1e-9
    return (
        float(intrinsics["fx"]) * camera_point[0] / depth + float(intrinsics["cx"]),
        float(intrinsics["fy"]) * camera_point[1] / depth + float(intrinsics["cy"]),
    )


def _observation_reprojection_errors(
    observations: Sequence[BallObservation],
    *,
    calibration: Mapping[str, Any],
    initial_position: tuple[float, float, float],
    velocity: tuple[float, float, float],
    t0: float,
    physics: PhysicsParameters,
    config: BallArcSolverConfig,
) -> dict[int, float]:
    positions = _integrate_positions(
        initial_position,
        velocity,
        [obs.t for obs in observations],
        t0=t0,
        physics=physics,
        config=config,
    )
    errors: dict[int, float] = {}
    for obs, position in zip(observations, positions, strict=True):
        projected = _project_world_point(calibration, position)
        errors[obs.frame] = _distance2(projected, obs.xy)
    return errors


def _size_depth_residual(
    calibration: Mapping[str, Any],
    obs: BallObservation,
    world_xyz: tuple[float, float, float],
    *,
    physics: PhysicsParameters,
    config: BallArcSolverConfig,
    sigma_floor_m: float,
) -> tuple[float, float] | None:
    if not config.enable_size_depth_residual:
        return None
    measured_range = _range_from_apparent_diameter_px(calibration, obs, physics=physics)
    if measured_range is None:
        return None
    origin, direction = pixel_ray_world(calibration, obs.xy)
    predicted_range = _dot(_sub(world_xyz, origin), direction)
    confidence = max(config.size_confidence_floor, min(1.0, obs.size_confidence if obs.size_confidence is not None else obs.confidence))
    sigma = max(float(sigma_floor_m), measured_range * config.size_depth_relative_sigma) / math.sqrt(confidence)
    return predicted_range - measured_range, sigma


def _range_from_apparent_diameter_px(
    calibration: Mapping[str, Any],
    obs: BallObservation,
    *,
    physics: PhysicsParameters,
) -> float | None:
    if obs.diameter_px is None or obs.diameter_px <= 0.0:
        return None
    intrinsics = _intrinsics(calibration)
    fx = _float_or_none(intrinsics.get("fx"))
    fy = _float_or_none(intrinsics.get("fy"))
    cx = _float_or_none(intrinsics.get("cx"))
    cy = _float_or_none(intrinsics.get("cy"))
    if fx is None or fy is None or cx is None or cy is None:
        return None
    f_eff = (fx + fy) * 0.5
    depth_z = f_eff * physics.diameter_m / obs.diameter_px
    ray_norm = math.sqrt(((obs.xy[0] - cx) / fx) ** 2 + ((obs.xy[1] - cy) / fy) ** 2 + 1.0)
    value = depth_z * ray_norm
    return value if math.isfinite(value) and value > 0.0 else None


def _size_residual_distribution(
    observations: Sequence[BallObservation],
    *,
    calibration: Mapping[str, Any],
    initial_position: tuple[float, float, float],
    velocity: tuple[float, float, float],
    t0: float,
    physics: PhysicsParameters,
    config: BallArcSolverConfig,
    sigma_floor_m: float,
) -> dict[str, Any]:
    positions = _integrate_positions(
        initial_position,
        velocity,
        [obs.t for obs in observations],
        t0=t0,
        physics=physics,
        config=config,
    )
    residuals: list[float] = []
    for obs, position in zip(observations, positions, strict=True):
        residual = _size_depth_residual(
            calibration,
            obs,
            position,
            physics=physics,
            config=config,
            sigma_floor_m=sigma_floor_m,
        )
        if residual is not None:
            residuals.append(abs(residual[0]))
    payload = _distribution(residuals)
    payload["enabled"] = bool(config.enable_size_depth_residual)
    return payload


def _size_depth_validation_summary(
    confident_segments: Sequence[FlightSegmentFit],
    weak_segments: Sequence[FlightSegmentFit],
) -> dict[str, Any]:
    confident_values = _collect_size_residual_values(confident_segments)
    weak_values = _collect_size_residual_values(weak_segments)
    return {
        "enabled": True,
        "with_size": _distribution([*confident_values, *weak_values]),
        "confident": _distribution(confident_values),
        "weak": _distribution(weak_values),
    }


def _collect_size_residual_values(segments: Sequence[FlightSegmentFit]) -> list[float]:
    values: list[float] = []
    for segment in segments:
        residuals = segment.size_residuals_m
        count = int(residuals.get("count") or 0)
        median_value = _float_or_none(residuals.get("median"))
        if count > 0 and median_value is not None:
            values.extend([median_value] * count)
    return values


def _weak_initial_target_from_observations(
    observations: Sequence[BallObservation],
    *,
    calibration: Mapping[str, Any],
    physics: PhysicsParameters,
) -> tuple[float, float, float] | None:
    sized = [obs for obs in observations if obs.diameter_px is not None and obs.diameter_px > 0.0]
    if not sized:
        return None
    obs = sized[-1]
    measured_range = _range_from_apparent_diameter_px(calibration, obs, physics=physics)
    if measured_range is None:
        return None
    origin, direction = pixel_ray_world(calibration, obs.xy)
    return _add(origin, _scale(direction, measured_range))  # type: ignore[return-value]


def _ray_plane_target(
    obs: BallObservation,
    *,
    calibration: Mapping[str, Any],
    z: float,
) -> tuple[float, float, float] | None:
    try:
        origin, direction = pixel_ray_world(calibration, obs.xy)
        return intersect_ray_z(origin, direction, z)
    except ValueError:
        return None


def _net_clearance_m(
    p0: tuple[float, float, float],
    v0: tuple[float, float, float],
    t0: float,
    t1: float,
    physics: PhysicsParameters,
    config: BallArcSolverConfig,
    net_plane: Mapping[str, Any] | None,
) -> float | None:
    plane = net_plane.get("plane") if isinstance(net_plane, Mapping) else None
    if not isinstance(plane, Mapping):
        return None
    point = _vec3(plane.get("point"))
    normal = _vec3(plane.get("normal"))
    if point is None or normal is None:
        return None
    net_height = _net_height_m(net_plane)
    times = [t0 + (t1 - t0) * i / 80.0 for i in range(81)]
    points = _integrate_positions(p0, v0, times, t0=t0, physics=physics, config=config)
    values = [_dot(_sub(point3, point), normal) for point3 in points]
    crossings: list[tuple[float, float]] = []
    for idx, (a, b) in enumerate(zip(values, values[1:])):
        if a == 0.0:
            crossings.append((times[idx], points[idx][2]))
        if a * b > 0.0:
            continue
        denom = abs(a) + abs(b)
        alpha = 0.0 if denom <= 1e-12 else abs(a) / denom
        z = points[idx][2] + (points[idx + 1][2] - points[idx][2]) * alpha
        crossings.append((times[idx] + (times[idx + 1] - times[idx]) * alpha, z))
    if not crossings:
        return None
    return min(z - (net_height + physics.radius_m) for _, z in crossings)


def _net_soft_residual(
    p0: tuple[float, float, float],
    v0: tuple[float, float, float],
    t0: float,
    t1: float,
    physics: PhysicsParameters,
    config: BallArcSolverConfig,
    net_plane: Mapping[str, Any] | None,
) -> float:
    clearance = _net_clearance_m(p0, v0, t0, t1, physics, config, net_plane)
    if clearance is None or clearance >= -config.net_clearance_slack_m:
        return 0.0
    return (-config.net_clearance_slack_m - clearance) / max(config.net_clearance_slack_m, 1e-9)


def _net_height_m(net_plane: Mapping[str, Any]) -> float:
    center_height_in = _float_or_none(net_plane.get("center_height_in"))
    if center_height_in is not None:
        return center_height_in * 0.0254
    endpoints = net_plane.get("endpoints")
    if isinstance(endpoints, Sequence) and len(endpoints) == 2:
        zs = [_vec3(endpoint)[2] for endpoint in endpoints if _vec3(endpoint) is not None]
        if zs:
            return float(sum(zs) / len(zs))
    return 0.8636


def _blocked_segment(segment_id: int, start: AnchorEvent, end: AnchorEvent, reason: str) -> FlightSegmentFit:
    return FlightSegmentFit(
        segment_id=segment_id,
        status=f"blocked:{reason}",
        start_anchor=start,
        end_anchor=end,
        initial_position_m=start.world_xyz,
        initial_velocity_mps=(0.0, 0.0, 0.0),
        observations=(),
        inlier_frames=(),
        outlier_frames=(),
        reprojection_errors_px={},
        reprojection_rmse_px=None,
        max_reprojection_error_px=None,
        endpoint_error_m=0.0,
        net_clearance_m=None,
        net_clearance_ok=None,
        physical_sanity={"violation": True, "violations": [reason]},
        size_residuals_m={"enabled": False, "count": 0, "median": None, "mean": None, "p90": None, "p95": None, "max": None},
    )


def _anchor_sigma_m(anchor: AnchorEvent, config: BallArcSolverConfig) -> float:
    sigma = _float_or_none(anchor.sigma_m)
    if sigma is None or not math.isfinite(sigma):
        return config.min_anchor_sigma_m
    return max(config.min_anchor_sigma_m, float(sigma))


def _anchor_with_sigma_floor(anchor: AnchorEvent, config: BallArcSolverConfig) -> AnchorEvent:
    sigma = _anchor_sigma_m(anchor, config)
    if sigma == anchor.sigma_m:
        return anchor
    return replace(anchor, sigma_m=sigma)


def _frames(ball_track: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    frames = ball_track.get("frames")
    if not isinstance(frames, list):
        raise ValueError("ball_track.frames must be a list")
    return [frame for frame in frames if isinstance(frame, Mapping)]


def _ball_observations(
    frames: Sequence[Mapping[str, Any]],
    *,
    fps: float,
    ball_sizes: Mapping[str, Any] | None = None,
) -> list[BallObservation]:
    observations: list[BallObservation] = []
    size_by_frame = _ball_size_observations_by_frame(ball_sizes)
    for index, frame in enumerate(frames):
        if frame.get("visible") is not True:
            continue
        xy = _xy_tuple(frame.get("xy"))
        if xy is None:
            continue
        t = _float_or_none(frame.get("t"))
        if t is None:
            t = index / max(fps, 1e-9)
        confidence = _float_or_none(frame.get("conf", frame.get("confidence"))) or 1.0
        size = _frame_size_observation(frame) or size_by_frame.get(index)
        observations.append(
            BallObservation(
                frame=index,
                t=t,
                xy=xy,
                confidence=confidence,
                visible=True,
                diameter_px=size["diameter_px"] if size else None,
                size_confidence=size["confidence"] if size else None,
                size_source=size["source"] if size else None,
            )
        )
    return sorted(observations, key=lambda obs: (obs.t, obs.frame))


def _ball_size_observations_by_frame(ball_sizes: Mapping[str, Any] | None) -> dict[int, dict[str, Any]]:
    if not isinstance(ball_sizes, Mapping):
        return {}
    frames = ball_sizes.get("frames")
    if not isinstance(frames, Sequence) or isinstance(frames, (str, bytes)):
        return {}
    parsed: dict[int, dict[str, Any]] = {}
    for item in frames:
        if not isinstance(item, Mapping):
            continue
        frame = _frame_from_mapping(item)
        if frame is None:
            continue
        size = _frame_size_observation(item)
        if size is not None:
            parsed[frame] = size
    return parsed


def _frame_size_observation(frame: Mapping[str, Any]) -> dict[str, Any] | None:
    diameter = _float_or_none(
        _first_present(
            frame,
            "diameter_px",
            "apparent_diameter_px",
            "ball_diameter_px",
            "heatmap_extent_px",
        )
    )
    if diameter is None:
        radius = _float_or_none(_first_present(frame, "radius_px", "ball_radius_px", "heatmap_radius_px"))
        if radius is not None:
            diameter = radius * 2.0
    if diameter is None:
        bbox = frame.get("bbox_xywh", frame.get("bbox"))
        if isinstance(bbox, Sequence) and not isinstance(bbox, (str, bytes)) and len(bbox) == 4:
            width = _float_or_none(bbox[2])
            height = _float_or_none(bbox[3])
            if width is not None and height is not None:
                diameter = (width + height) * 0.5
    if diameter is None or diameter <= 0.0:
        return None
    confidence = _float_or_none(_first_present(frame, "size_confidence", "size_conf", "confidence", "conf"))
    if confidence is None:
        confidence = 0.5
    return {
        "diameter_px": float(diameter),
        "confidence": max(0.0, min(1.0, float(confidence))),
        "source": str(frame.get("source") or "ball_track_size_field"),
    }


def _payload_fps(ball_track: Mapping[str, Any], frames: Sequence[Mapping[str, Any]]) -> float:
    fps = _float_or_none(ball_track.get("fps"))
    if fps is not None and fps > 0.0:
        return fps
    times = [_float_or_none(frame.get("t")) for frame in frames]
    valid = [time for time in times if time is not None]
    deltas = [b - a for a, b in zip(valid, valid[1:]) if b > a]
    return 1.0 / (sum(deltas) / len(deltas)) if deltas else 30.0


def _filter_anchors_to_rally_spans(
    anchors: Sequence[AnchorEvent],
    rally_spans: Mapping[str, Any] | None,
) -> list[AnchorEvent]:
    if not isinstance(rally_spans, Mapping):
        return list(anchors)
    spans = rally_spans.get("spans")
    if not isinstance(spans, list) or not spans:
        return list(anchors)
    parsed: list[tuple[float, float]] = []
    for span in spans:
        if not isinstance(span, Mapping):
            continue
        t0 = _float_or_none(span.get("t0"))
        t1 = _float_or_none(span.get("t1"))
        if t0 is not None and t1 is not None and t1 >= t0:
            parsed.append((t0, t1))
    if not parsed:
        return list(anchors)
    return [anchor for anchor in anchors if any(t0 - 1e-9 <= anchor.t <= t1 + 1e-9 for t0, t1 in parsed)]


def _segment_for_time(segments: Sequence[FlightSegmentFit], t: float) -> FlightSegmentFit | None:
    candidates = [segment for segment in segments if segment.start_anchor.t - 1e-9 <= t <= segment.end_anchor.t + 1e-9]
    if not candidates:
        return None
    return min(candidates, key=lambda segment: abs((segment.start_anchor.t + segment.end_anchor.t) / 2.0 - t))


def _frame_sigma(segment: FlightSegmentFit, t: float) -> float:
    span = max(segment.end_anchor.t - segment.start_anchor.t, 1e-9)
    alpha = max(0.0, min(1.0, (t - segment.start_anchor.t) / span))
    anchor_sigma = (1.0 - alpha) * segment.start_anchor.sigma_m + alpha * segment.end_anchor.sigma_m
    reproj_component = 0.0 if segment.reprojection_rmse_px is None else min(0.25, segment.reprojection_rmse_px * 0.01)
    return math.sqrt(anchor_sigma * anchor_sigma + reproj_component * reproj_component)


def _initial_velocity_guess(
    p0: tuple[float, float, float],
    p1: tuple[float, float, float],
    dt: float,
    physics: PhysicsParameters,
) -> tuple[float, float, float]:
    return (
        (p1[0] - p0[0]) / dt,
        (p1[1] - p0[1]) / dt,
        (p1[2] - p0[2] + 0.5 * physics.gravity_mps2 * dt * dt) / dt,
    )


def _analytic_no_drag_position(
    p0: tuple[float, float, float],
    v0: tuple[float, float, float],
    dt: float,
    gravity: float,
) -> tuple[float, float, float]:
    return (
        p0[0] + v0[0] * dt,
        p0[1] + v0[1] * dt,
        p0[2] + v0[2] * dt - 0.5 * gravity * dt * dt,
    )


def _camera_arrays(calibration: Mapping[str, Any]) -> dict[str, Any]:
    extrinsics = calibration.get("extrinsics")
    if not isinstance(extrinsics, Mapping):
        raise ValueError("calibration.extrinsics is required")
    return {
        "rotation": tuple(tuple(float(value) for value in row) for row in extrinsics["R"]),
        "translation": tuple(float(value) for value in extrinsics["t"]),
    }


def _intrinsics(calibration: Mapping[str, Any]) -> Mapping[str, Any]:
    intrinsics = calibration.get("intrinsics")
    if not isinstance(intrinsics, Mapping):
        raise ValueError("calibration.intrinsics is required")
    return intrinsics


def _gsd_sigma(calibration: Mapping[str, Any], xy: Sequence[float]) -> float | None:
    del xy
    gsd_model = calibration.get("gsd_model")
    if not isinstance(gsd_model, Mapping):
        return None
    samples = gsd_model.get("samples")
    sigma_values: list[float] = []
    if isinstance(samples, Sequence) and not isinstance(samples, (str, bytes)):
        for sample in samples:
            if not isinstance(sample, Mapping):
                continue
            sigma = _float_or_none(sample.get("sigma_p_m"))
            if sigma is not None:
                sigma_values.append(sigma)
    calibration_sigma = _float_or_none(gsd_model.get("calibration_sigma_m"))
    plane_sigma = _float_or_none(gsd_model.get("plane_sigma_m"))
    if sigma_values:
        base = min(sigma_values)
        extras = [value for value in (calibration_sigma, plane_sigma) if value is not None]
        return math.sqrt(base * base + sum(value * value for value in extras))
    if calibration_sigma is not None or plane_sigma is not None:
        return math.sqrt(sum(value * value for value in (calibration_sigma or 0.0, plane_sigma or 0.0)))
    return None


def _ray_plane_pixel_sigma(calibration: Mapping[str, Any], xy: Sequence[float], z: float) -> float | None:
    try:
        base_origin, base_dir = pixel_ray_world(calibration, xy)
        base = intersect_ray_z(base_origin, base_dir, z)
        x_origin, x_dir = pixel_ray_world(calibration, (float(xy[0]) + 1.0, float(xy[1])))
        y_origin, y_dir = pixel_ray_world(calibration, (float(xy[0]), float(xy[1]) + 1.0))
        dx = _distance(base, intersect_ray_z(x_origin, x_dir, z))
        dy = _distance(base, intersect_ray_z(y_origin, y_dir, z))
        return math.sqrt(dx * dx + dy * dy)
    except Exception:
        return None


def _nearest_observation_by_frame(
    observations_by_frame: Mapping[int, BallObservation],
    frame: int,
    *,
    max_gap: int,
) -> BallObservation | None:
    candidates = [
        (abs(item_frame - frame), obs)
        for item_frame, obs in observations_by_frame.items()
        if abs(item_frame - frame) <= max_gap
    ]
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: item[0])[0][1]


def _nearest_observation_by_time(
    observations: Sequence[BallObservation],
    t: float,
    *,
    max_gap_s: float,
) -> BallObservation | None:
    candidates = [(abs(obs.t - t), obs) for obs in observations if abs(obs.t - t) <= max_gap_s]
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: item[0])[0][1]


def _semantic_joint_indexes(joint_names: Any) -> dict[str, int]:
    if not isinstance(joint_names, Sequence) or isinstance(joint_names, (str, bytes)):
        return {}
    output: dict[str, int] = {}
    aliases = {
        "leftwrist": "left_wrist",
        "lwrist": "left_wrist",
        "lefthand": "left_wrist",
        "lhand": "left_wrist",
        "rightwrist": "right_wrist",
        "rwrist": "right_wrist",
        "righthand": "right_wrist",
        "rhand": "right_wrist",
        "leftelbow": "left_elbow",
        "lelbow": "left_elbow",
        "rightelbow": "right_elbow",
        "relbow": "right_elbow",
    }
    for index, name in enumerate(joint_names):
        normalized = "".join(part for part in str(name).lower().replace("-", "_").split("_") if part)
        semantic = aliases.get(normalized)
        if semantic is not None:
            output[semantic] = index
    return output


def _joint_confidence(confs: Any, index: int) -> float:
    if not isinstance(confs, Sequence) or isinstance(confs, (str, bytes)) or index >= len(confs):
        return 1.0
    value = _float_or_none(confs[index])
    if value is None:
        return 0.0
    return max(0.0, min(1.0, value))


def _config_summary(config: BallArcSolverConfig) -> dict[str, Any]:
    return {
        key: value
        for key, value in config.__dict__.items()
        if isinstance(value, (str, int, float, bool)) or value is None
    }


def _distribution(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "median": None, "mean": None, "p90": None, "p95": None, "max": None}
    ordered = sorted(float(value) for value in values)
    return {
        "count": len(ordered),
        "median": _round(float(median(ordered)), 6),
        "mean": _round(sum(ordered) / len(ordered), 6),
        "p90": _round(_percentile(ordered, 90.0), 6),
        "p95": _round(_percentile(ordered, 95.0), 6),
        "max": _round(max(ordered), 6),
    }


def _percentile(ordered: Sequence[float], percentile: float) -> float:
    if not ordered:
        return float("nan")
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percentile / 100.0
    low = int(math.floor(rank))
    high = int(math.ceil(rank))
    if low == high:
        return ordered[low]
    alpha = rank - low
    return ordered[low] * (1.0 - alpha) + ordered[high] * alpha


def _xy_required(xy: Sequence[float]) -> tuple[float, float]:
    parsed = _xy_tuple(xy)
    if parsed is None:
        raise ValueError("xy must contain two finite values")
    return parsed


def _xy_tuple(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        return None
    try:
        xy = (float(value[0]), float(value[1]))
    except (TypeError, ValueError):
        return None
    return xy if all(math.isfinite(component) for component in xy) else None


def _vec3(value: Any) -> tuple[float, float, float] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 3:
        return None
    try:
        vector = (float(value[0]), float(value[1]), float(value[2]))
    except (TypeError, ValueError):
        return None
    return vector if all(math.isfinite(component) for component in vector) else None


def _frame_from_mapping(item: Mapping[str, Any]) -> int | None:
    for key in ("frame", "frame_index", "frame_idx"):
        value = item.get(key)
        try:
            frame = int(value)
        except (TypeError, ValueError):
            continue
        if frame >= 0:
            return frame
    return None


def _float_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _first_present(mapping: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def _mat_vec(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> tuple[float, float, float]:
    return (
        sum(float(matrix[0][i]) * float(vector[i]) for i in range(3)),
        sum(float(matrix[1][i]) * float(vector[i]) for i in range(3)),
        sum(float(matrix[2][i]) * float(vector[i]) for i in range(3)),
    )


def _transpose3(matrix: Sequence[Sequence[float]]) -> tuple[tuple[float, float, float], ...]:
    return tuple(tuple(float(matrix[row][col]) for row in range(3)) for col in range(3))


def _add(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float]:
    return (float(a[0]) + float(b[0]), float(a[1]) + float(b[1]), float(a[2]) + float(b[2]))


def _sub(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float]:
    return (float(a[0]) - float(b[0]), float(a[1]) - float(b[1]), float(a[2]) - float(b[2]))


def _scale(a: Sequence[float], scalar: float) -> tuple[float, ...]:
    return tuple(float(value) * scalar for value in a)


def _scaled_vec(a: Sequence[float], sigma: float) -> list[float]:
    safe = max(float(sigma), 1e-9)
    return [float(value) / safe for value in a]


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return float(a[0]) * float(b[0]) + float(a[1]) * float(b[1]) + float(a[2]) * float(b[2])


def _norm(a: Sequence[float]) -> float:
    return math.sqrt(sum(float(value) * float(value) for value in a))


def _normalize(a: Sequence[float]) -> tuple[float, float, float]:
    norm = _norm(a)
    if norm <= 1e-12:
        return (0.0, 0.0, 0.0)
    return (float(a[0]) / norm, float(a[1]) / norm, float(a[2]) / norm)


def _distance(a: Sequence[float], b: Sequence[float]) -> float:
    return _norm(_sub(a, b))


def _distance2(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt((float(a[0]) - float(b[0])) ** 2 + (float(a[1]) - float(b[1])) ** 2)


def _distance_point_to_ray(
    point: tuple[float, float, float],
    origin: tuple[float, float, float],
    direction: tuple[float, float, float],
) -> float:
    op = _sub(point, origin)
    along = _dot(op, direction)
    closest = _add(origin, _scale(direction, along))
    return _distance(point, closest)


def _lerp(a: Sequence[float], b: Sequence[float], alpha: float) -> tuple[float, float, float]:
    return (
        float(a[0]) + (float(b[0]) - float(a[0])) * alpha,
        float(a[1]) + (float(b[1]) - float(a[1])) * alpha,
        float(a[2]) + (float(b[2]) - float(a[2])) * alpha,
    )


def _add_state(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float, float, float, float]:
    return tuple(float(a[i]) + float(b[i]) for i in range(6))  # type: ignore[return-value]


def _rmse(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return math.sqrt(sum(value * value for value in values) / len(values))


def _vec_json(vector: Sequence[float]) -> list[float]:
    return [_round(float(value), 9) for value in vector]


def _round(value: float, digits: int) -> float:
    return round(float(value), digits)


def _optional_round(value: float | None, digits: int) -> float | None:
    return None if value is None else _round(value, digits)


__all__ = [
    "ARTIFACT_TYPE",
    "LANE",
    "AnchorEvent",
    "BallArcSolverConfig",
    "BallObservation",
    "FlightSegmentFit",
    "PhysicsParameters",
    "anchor_sigma_for_bounce",
    "build_bounce_anchor",
    "fit_flight_segment",
    "fit_weak_flight_segment",
    "intersect_ray_z",
    "order_event_anchors",
    "pixel_ray_world",
    "solve_ball_arc_track",
]

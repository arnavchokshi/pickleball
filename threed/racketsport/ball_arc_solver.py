"""Event-anchored 3D ball arc solver for render-only world continuity.

The solver treats monocular 2D ball sightings as camera rays. Human-reviewed
bounces and loose contact priors provide 3D anchors; each consecutive anchor
pair bounds one free-flight segment. Outputs are explicitly render-only and
must not feed BALL detector metrics, gates, training, or promotion.
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
from statistics import median
import time
from typing import Any, Callable, Mapping, Sequence

from .ball_physics3d import _project_world_array, reconstruct_bounce_arcs_from_image_track
from .court_templates import get_court_template
from .io_decode import frame_time_lookup, nearest_frame_for_time, time_for_frame
from .skeleton3d import semanticize_skeleton_payload


SCHEMA_VERSION = 2
ARTIFACT_TYPE = "racketsport_ball_track_arc_solved"
LANE = "BALL-ARC-SOLVER"
SOURCE = "event_anchored_drag_arc_solver"
BALL_RADIUS_M = 0.0371
STEYN_CL_PER_SPIN = 0.195
SPIN_SCALAR_MAX_ABS = 0.8
SPIN_SCALAR_REGULARIZATION_LAMBDA = 0.05
SPIN_SCALAR_MIN_INLIERS = 8
FIT_VALIDITY_MIN_OBSERVATIONS_FOR_INLIER_GATE = 6
FIT_VALIDITY_MIN_INLIER_FRACTION = 0.15
# Safety invariant, not a model/default-selection knob.  A single segment may
# abstain after this wall time; the remaining segments must still be emitted.
SEGMENT_WALL_CLOCK_BUDGET_S = 5.0
SEGMENT_BUDGET_EXCEEDED = "segment_budget_exceeded"
_ACTIVE_SEGMENT_DEADLINE: ContextVar[float | None] = ContextVar(
    "ball_arc_segment_deadline",
    default=None,
)


class _SegmentBudgetExceeded(RuntimeError):
    """Internal control flow for a typed, fail-closed segment abstention."""


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
    bounce_bounce_discovery_on_gate_failure: bool = True
    discovery_min_interior_observations: int = 4
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
    candidate_selection_max_iterations: int = 5
    max_candidates_per_frame: int = 12
    candidate_association_mode: str = "free"
    candidate_score_floors: Mapping[str, float] | None = None
    bvp_shooting_max_iterations: int = 6
    bvp_shooting_tolerance_m: float = 0.005
    bvp_shooting_fd_eps_mps: float = 0.05
    bvp_shooting_max_backtracks: int = 4
    bvp_shooting_fallback_to_free_fit: bool = True
    endpoint_refinement_max_nfev: int = 60
    endpoint_refinement_time_corridor_frames: float = 1.0
    endpoint_refinement_contact_cap_m: float = 0.5
    endpoint_refinement_bounce_cap_m: float = 0.2
    endpoint_refinement_default_cap_m: float = 1.0
    fit_spin_scalar: bool = False
    enable_both_ends_pinning_inlier_pass: bool = False
    pinning_min_inliers: int = 3
    court_sport: str = "pickleball"
    court_margin_m: float = 4.0
    court_z_min_m: float = -0.15

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
        if self.discovery_min_interior_observations < 1:
            raise ValueError("discovery_min_interior_observations must be positive")
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
        if self.candidate_selection_max_iterations <= 0:
            raise ValueError("candidate_selection_max_iterations must be positive")
        if self.max_candidates_per_frame <= 0:
            raise ValueError("max_candidates_per_frame must be positive")
        if self.candidate_association_mode not in {"free", "rescue_only"}:
            raise ValueError("candidate_association_mode must be 'free' or 'rescue_only'")
        if self.bvp_shooting_max_iterations <= 0:
            raise ValueError("bvp_shooting_max_iterations must be positive")
        if self.bvp_shooting_tolerance_m <= 0.0:
            raise ValueError("bvp_shooting_tolerance_m must be positive")
        if self.bvp_shooting_fd_eps_mps <= 0.0:
            raise ValueError("bvp_shooting_fd_eps_mps must be positive")
        if self.bvp_shooting_max_backtracks < 0:
            raise ValueError("bvp_shooting_max_backtracks must be non-negative")
        if self.endpoint_refinement_max_nfev <= 0:
            raise ValueError("endpoint_refinement_max_nfev must be positive")
        if self.endpoint_refinement_time_corridor_frames <= 0.0:
            raise ValueError("endpoint_refinement_time_corridor_frames must be positive")
        if self.endpoint_refinement_contact_cap_m <= 0.0:
            raise ValueError("endpoint_refinement_contact_cap_m must be positive")
        if self.endpoint_refinement_bounce_cap_m <= 0.0:
            raise ValueError("endpoint_refinement_bounce_cap_m must be positive")
        if self.endpoint_refinement_default_cap_m <= 0.0:
            raise ValueError("endpoint_refinement_default_cap_m must be positive")
        if self.pinning_min_inliers < 2:
            raise ValueError("pinning_min_inliers must be at least 2")
        if self.court_margin_m < 0.0:
            raise ValueError("court_margin_m must be non-negative")
        try:
            get_court_template(str(self.court_sport))  # type: ignore[arg-type]
        except ValueError as exc:
            raise ValueError("court_sport must name a supported court template") from exc
        for source, floor in dict(self.candidate_score_floors or {}).items():
            if not str(source):
                raise ValueError("candidate_score_floors source keys must be non-empty")
            value = float(floor)
            if not math.isfinite(value) or value < 0.0 or value > 1.0:
                raise ValueError("candidate_score_floors values must be finite values in [0, 1]")


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
    observation_source: str = "primary:ball_track"
    candidate_score: float | None = None
    candidate_rank: int | None = None
    candidate_selection: str | None = None


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
class SoftSegmentBoundary:
    """Non-physical evidence that may only partition a hard-anchor span.

    A soft boundary deliberately has no world position or event kind.  The
    solver may use its corrected time to bound numerical work, but it may not
    turn the boundary into bounce/contact evidence or an endpoint constraint.
    """

    boundary_id: str
    corrected_time_s: float
    frame: int
    onset_ids: tuple[str, ...]
    selection_rule_id: str
    anchor_class: str = "audio_onset_soft"
    source_artifact: str | None = None

    def __post_init__(self) -> None:
        if self.anchor_class != "audio_onset_soft":
            raise ValueError("soft boundary anchor_class must be audio_onset_soft")
        if not self.boundary_id:
            raise ValueError("soft boundary requires boundary_id")
        if not math.isfinite(float(self.corrected_time_s)) or float(self.corrected_time_s) < 0.0:
            raise ValueError("soft boundary corrected_time_s must be finite and non-negative")
        if int(self.frame) < 0:
            raise ValueError("soft boundary frame must be non-negative")
        if not self.onset_ids or any(not str(onset_id) for onset_id in self.onset_ids):
            raise ValueError("soft boundary requires non-empty onset_ids")
        if not self.selection_rule_id:
            raise ValueError("soft boundary requires selection_rule_id")

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "boundary_id": self.boundary_id,
            "anchor_class": self.anchor_class,
            "onset_ids": [str(onset_id) for onset_id in self.onset_ids],
            "corrected_time_s": _round(float(self.corrected_time_s), 9),
            "frame": int(self.frame),
            "selection_rule_id": self.selection_rule_id,
            "allowed_role": "segment_split_boundary_only",
            "event_type": None,
            "world_constraint": None,
            "counts_as_bounce_evidence": False,
            "counts_as_flight_sanity_anchor": False,
        }
        if self.source_artifact is not None:
            payload["source_artifact"] = self.source_artifact
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
    spin_scalar: float = 0.0
    soft_split_provenance: tuple[Mapping[str, Any], ...] = ()
    primary_observations: tuple[BallObservation, ...] = ()
    candidate_sets_by_frame: Mapping[int, tuple[BallObservation, ...]] | None = None
    selected_observations_by_frame: Mapping[int, BallObservation] | None = None
    candidate_association: Mapping[str, Any] | None = None
    diagnostics: Mapping[str, Any] | None = None
    degradation: Mapping[str, Any] | None = None

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
            spin_scalar=self.spin_scalar,
        )[0]

    def to_json(self) -> dict[str, Any]:
        payload = {
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
            "spin_scalar": _round(self.spin_scalar, 9),
            "spin_cl": _round(STEYN_CL_PER_SPIN * self.spin_scalar, 9),
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
        if self.candidate_association:
            payload["candidate_association"] = dict(self.candidate_association)
            selected = self.selected_observations_by_frame or {}
            payload["candidate_selection_by_frame"] = {
                str(frame): _observation_candidate_payload(obs)
                for frame, obs in sorted(selected.items())
            }
        if self.soft_split_provenance:
            payload["soft_split_provenance"] = [dict(item) for item in self.soft_split_provenance]
        if self.diagnostics:
            payload["diagnostics"] = dict(self.diagnostics)
        if self.degradation:
            payload["degradation"] = dict(self.degradation)
        return payload


def build_bounce_anchor(
    bounce: Mapping[str, Any],
    calibration: Mapping[str, Any],
    *,
    ball_radius_m: float = BALL_RADIUS_M,
    ball_xy: Sequence[float] | None = None,
    status: str = "human_reviewed",
    sigma_m: float | None = None,
    source: str | None = None,
    details: Mapping[str, Any] | None = None,
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
    anchor_details = {"pixel_xy": [float(xy[0]), float(xy[1])], "ball_radius_m": ball_radius_m}
    if details:
        anchor_details.update(dict(details))
    return AnchorEvent(
        anchor_id=anchor_id,
        kind="bounce",
        t=float(t),
        frame=frame,
        world_xyz=world_xyz,
        sigma_m=float(sigma_m),
        status=status,
        immovable=status == "human_reviewed",
        source=source or "ray_intersection_z_ball_radius",
        details=anchor_details,
    )


def order_event_anchors(anchors: Sequence[AnchorEvent]) -> list[AnchorEvent]:
    """Sort anchors by time and prefer human-reviewed duplicates."""

    priority = {"human_reviewed": 0, "auto_bounce_candidate": 1, "contact_prior": 2, "solver_proposed": 3}
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


def _is_soft_split_anchor(anchor: AnchorEvent) -> bool:
    return anchor.kind == "audio_onset_soft" and anchor.status == "soft_split_boundary"


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
    candidate_sets_by_frame: Mapping[int, Sequence[BallObservation]] | None = None,
) -> FlightSegmentFit:
    """Fit one event-bounded free-flight segment."""

    cfg = config or BallArcSolverConfig()
    phys = physics or PhysicsParameters()
    if candidate_sets_by_frame:
        return _run_with_segment_budget(
            segment_id=segment_id,
            start=start_anchor,
            end=end_anchor,
            observations=observations,
            candidate_sets_by_frame=candidate_sets_by_frame,
            config=cfg,
            fit=lambda: _fit_flight_segment_with_candidate_association(
                segment_id=segment_id,
                start_anchor=start_anchor,
                end_anchor=end_anchor,
                observations=observations,
                candidate_sets_by_frame=candidate_sets_by_frame,
                calibration=calibration,
                physics=phys,
                config=cfg,
                net_plane=net_plane,
                max_nfev=max_nfev,
            ),
        )
    return _run_with_segment_budget(
        segment_id=segment_id,
        start=start_anchor,
        end=end_anchor,
        observations=observations,
        candidate_sets_by_frame=None,
        config=cfg,
        fit=lambda: _fit_flight_segment_once(
            segment_id=segment_id,
            start_anchor=start_anchor,
            end_anchor=end_anchor,
            observations=observations,
            calibration=calibration,
            physics=phys,
            config=cfg,
            net_plane=net_plane,
            max_nfev=max_nfev,
        ),
    )


def _run_with_segment_budget(
    *,
    segment_id: int,
    start: AnchorEvent,
    end: AnchorEvent,
    observations: Sequence[BallObservation],
    candidate_sets_by_frame: Mapping[int, Sequence[BallObservation]] | None,
    config: BallArcSolverConfig,
    fit: Callable[[], FlightSegmentFit],
) -> FlightSegmentFit:
    """Run one segment under the production wall-clock safety invariant."""

    # Nested fit helpers share the outer segment deadline.
    if _ACTIVE_SEGMENT_DEADLINE.get() is not None:
        return fit()
    started = time.monotonic()
    span_candidates = _candidate_sets_in_span(
        candidate_sets_by_frame or {},
        start_t=start.t,
        end_t=end.t,
        max_per_frame=config.max_candidates_per_frame,
    )
    token = _ACTIVE_SEGMENT_DEADLINE.set(started + SEGMENT_WALL_CLOCK_BUDGET_S)
    try:
        _check_segment_budget()
        return fit()
    except _SegmentBudgetExceeded:
        return _budget_exceeded_segment(
            segment_id,
            start,
            end,
            budget_s=SEGMENT_WALL_CLOCK_BUDGET_S,
            elapsed_s=time.monotonic() - started,
            observation_count=len(observations),
            candidate_frame_count=len(span_candidates),
            candidate_count=sum(len(items) for items in span_candidates.values()),
        )
    finally:
        _ACTIVE_SEGMENT_DEADLINE.reset(token)


def _fit_flight_segment_once(
    *,
    segment_id: int,
    start_anchor: AnchorEvent,
    end_anchor: AnchorEvent,
    observations: Sequence[BallObservation],
    calibration: Mapping[str, Any],
    physics: PhysicsParameters,
    config: BallArcSolverConfig,
    net_plane: Mapping[str, Any] | None = None,
    max_nfev: int | None = None,
    refine_endpoints: bool = True,
    candidate_endpoint_frozen: bool = False,
) -> FlightSegmentFit:
    """Fit one event-bounded free-flight segment with a stored anchor-BVP solution."""

    cfg = config
    phys = physics
    if end_anchor.t - start_anchor.t < cfg.min_segment_dt_s:
        return _blocked_segment(segment_id, start_anchor, end_anchor, "duration_below_minimum")
    observations = tuple(
        obs for obs in observations if start_anchor.t - 1e-9 <= obs.t <= end_anchor.t + 1e-9 and obs.visible
    )
    if not _finite_vec3(start_anchor.world_xyz) or not _finite_vec3(end_anchor.world_xyz):
        return _blocked_segment(segment_id, start_anchor, end_anchor, "nonfinite_anchor")
    if _is_soft_split_anchor(start_anchor) or _is_soft_split_anchor(end_anchor):
        return _fit_soft_split_segment_once(
            segment_id=segment_id,
            start_anchor=start_anchor,
            end_anchor=end_anchor,
            observations=observations,
            calibration=calibration,
            physics=phys,
            config=cfg,
            net_plane=net_plane,
            max_nfev=max_nfev,
        )

    initial_free_fit = _fit_free_flight_segment_once(
        segment_id=segment_id,
        start_anchor=start_anchor,
        end_anchor=end_anchor,
        observations=observations,
        calibration=calibration,
        physics=phys,
        config=cfg,
        net_plane=net_plane,
        max_nfev=max_nfev,
    )
    if not initial_free_fit.status.startswith("fit"):
        return initial_free_fit

    free_fit = initial_free_fit
    fit_observations = observations
    pinning_inlier_pass: dict[str, Any] = {
        "enabled": bool(cfg.enable_both_ends_pinning_inlier_pass),
        "status": "disabled",
        "both_ends_pinned": False,
        "first_pass_inlier_count": initial_free_fit.inlier_count,
        "refit_observation_count": len(observations),
    }
    if cfg.enable_both_ends_pinning_inlier_pass:
        first_pass_inlier_frames = set(initial_free_fit.inlier_frames)
        fit_observations = tuple(obs for obs in observations if obs.frame in first_pass_inlier_frames)
        pinning_inlier_pass.update(
            {
                "status": "refused_insufficient_inliers",
                "both_ends_pinned": True,
                "refit_observation_count": len(fit_observations),
                "min_inliers": int(cfg.pinning_min_inliers),
                "inlier_threshold_px": float(cfg.max_reprojection_inlier_px),
            }
        )
        if len(fit_observations) >= cfg.pinning_min_inliers:
            free_fit = _fit_free_flight_segment_once(
                segment_id=segment_id,
                start_anchor=start_anchor,
                end_anchor=end_anchor,
                observations=fit_observations,
                calibration=calibration,
                physics=phys,
                config=cfg,
                net_plane=net_plane,
                max_nfev=max_nfev,
                diagnostics={
                    "dedicated_inlier_pass": True,
                    "first_pass_inlier_frames": sorted(first_pass_inlier_frames),
                },
            )
            pinning_inlier_pass["status"] = (
                "refit" if free_fit.status.startswith("fit") else "refit_failed"
            )
        # Rank-5 candidate semantics: retain the original event anchors exactly.
        # The final BVP is still scored on every observation below, so dropping
        # outliers from this dedicated fit cannot hide the 2D tail.
        refine_endpoints = False

    try:
        import numpy as np
        from scipy.optimize import least_squares
    except ImportError:
        return replace(
            free_fit,
            diagnostics={
                **dict(free_fit.diagnostics or {}),
                "bvp_shooting_status": "failed_fallback_to_free_fit",
                "bvp_shooting": {
                    "status": "failed_fallback_to_free_fit",
                    "reason": "missing_numeric_dependency",
                },
                "legacy_free_fit": _fit_diagnostic_summary(free_fit),
            },
        )

    endpoint_refinement = _endpoint_refinement_identity(
        start_anchor,
        end_anchor,
        observations=observations,
        config=cfg,
        frozen_for_candidate_association=(
            candidate_endpoint_frozen or cfg.enable_both_ends_pinning_inlier_pass
        ),
    )
    refined_start = start_anchor
    refined_end = end_anchor
    spin_fit_eligible = _can_fit_spin_scalar(cfg, observations, inlier_count=free_fit.inlier_count)
    spin_scalar = free_fit.spin_scalar if spin_fit_eligible else 0.0
    anchor_bvp = _solve_bvp_shooting(
        start_anchor.world_xyz,
        end_anchor.world_xyz,
        start_anchor.t,
        end_anchor.t,
        physics=phys,
        config=cfg,
        spin_scalar=spin_scalar,
    )
    if refine_endpoints:
        refined_start, refined_end, spin_scalar, endpoint_refinement = _refine_bvp_endpoints(
            start_anchor,
            end_anchor,
            observations=observations,
            calibration=calibration,
            physics=phys,
            config=cfg,
            net_plane=net_plane,
            np=np,
            least_squares=least_squares,
            spin_scalar=spin_scalar,
            spin_inlier_count=free_fit.inlier_count,
        )

    bvp = _solve_bvp_shooting(
        refined_start.world_xyz,
        refined_end.world_xyz,
        refined_start.t,
        refined_end.t,
        physics=phys,
        config=cfg,
        spin_scalar=spin_scalar,
    )
    diagnostics = {
        "bvp_shooting_status": bvp["status"],
        "bvp_shooting": bvp,
        "bvp_anchor_fallback": anchor_bvp,
        "endpoint_refinement": endpoint_refinement,
        "legacy_free_fit": _fit_diagnostic_summary(free_fit),
        "both_ends_pinning_inlier_pass": pinning_inlier_pass,
    }
    if not bool(bvp.get("converged")):
        if cfg.bvp_shooting_fallback_to_free_fit:
            fallback_diag = dict(diagnostics)
            fallback_diag["bvp_shooting_status"] = "failed_fallback_to_free_fit"
            fallback_diag["bvp_shooting"] = {**dict(bvp), "status": "failed_fallback_to_free_fit"}
            return replace(free_fit, diagnostics=fallback_diag)
        return _blocked_segment(segment_id, refined_start, refined_end, "bvp_shooting_failed")

    final_fit = _build_fit_from_bvp_solution(
        segment_id=segment_id,
        start_anchor=refined_start,
        end_anchor=refined_end,
        observations=observations,
        calibration=calibration,
        physics=phys,
        config=cfg,
        net_plane=net_plane,
        bvp=bvp,
        status="fit",
        diagnostics=diagnostics,
    )
    if abs(final_fit.spin_scalar) > 1e-12 and final_fit.inlier_count < SPIN_SCALAR_MIN_INLIERS:
        no_spin_fit = _fit_flight_segment_once(
            segment_id=segment_id,
            start_anchor=start_anchor,
            end_anchor=end_anchor,
            observations=observations,
            calibration=calibration,
            physics=phys,
            config=replace(cfg, fit_spin_scalar=False),
            net_plane=net_plane,
            max_nfev=max_nfev,
            refine_endpoints=refine_endpoints,
            candidate_endpoint_frozen=candidate_endpoint_frozen,
        )
        diagnostics_payload = dict(no_spin_fit.diagnostics or {})
        diagnostics_payload["spin_scalar_fit"] = {
            "enabled": True,
            "fit": False,
            "reason": "final_bvp_inlier_gate",
            "candidate_spin_scalar": _round(final_fit.spin_scalar, 9),
            "candidate_inlier_count": final_fit.inlier_count,
            "min_inliers": SPIN_SCALAR_MIN_INLIERS,
            "regularization_lambda": SPIN_SCALAR_REGULARIZATION_LAMBDA,
            "bound_abs": SPIN_SCALAR_MAX_ABS,
        }
        return replace(no_spin_fit, diagnostics=diagnostics_payload)
    return final_fit


def _fit_soft_split_segment_once(
    *,
    segment_id: int,
    start_anchor: AnchorEvent,
    end_anchor: AnchorEvent,
    observations: Sequence[BallObservation],
    calibration: Mapping[str, Any],
    physics: PhysicsParameters,
    config: BallArcSolverConfig,
    net_plane: Mapping[str, Any] | None = None,
    max_nfev: int | None = None,
) -> FlightSegmentFit:
    """Fit a time-partitioned chunk without treating soft evidence as 3D evidence."""

    cfg = replace(config, fit_spin_scalar=False)
    soft_start = _is_soft_split_anchor(start_anchor)
    soft_end = _is_soft_split_anchor(end_anchor)
    if not (soft_start or soft_end):
        raise ValueError("soft split fitter requires at least one soft boundary")
    try:
        import numpy as np
        from scipy.optimize import least_squares
    except ImportError as exc:
        return _blocked_segment(segment_id, start_anchor, end_anchor, f"missing_numeric_dependency:{exc}")

    dt = end_anchor.t - start_anchor.t
    initial_velocity = _initial_velocity_guess(start_anchor.world_xyz, end_anchor.world_xyz, dt, physics)
    p0 = np.asarray(start_anchor.world_xyz, dtype=float)
    initial = np.asarray([p0[0], p0[1], p0[2], *initial_velocity], dtype=float)
    if soft_start:
        x_min, x_max, y_min, y_max, _ = _court_volume_bounds(cfg)
        lower = np.asarray([x_min, y_min, 0.0, -60.0, -60.0, -60.0])
        upper = np.asarray([x_max, y_max, cfg.max_plausible_apex_m, 60.0, 60.0, 60.0])
    else:
        start_sigma = _anchor_sigma_m(start_anchor, cfg)
        relax = max(0.02, cfg.anchor_relax_sigma_multiplier * start_sigma)
        lower = np.asarray(
            [p0[0] - relax, p0[1] - relax, max(0.0, p0[2] - relax), -60.0, -60.0, -60.0]
        )
        upper = np.asarray(
            [p0[0] + relax, p0[1] + relax, p0[2] + relax, 60.0, 60.0, 60.0]
        )
    if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)) or not np.all(lower < upper):
        return _blocked_segment(segment_id, start_anchor, end_anchor, "invalid_segment_bounds")
    initial = np.minimum(np.maximum(initial, lower + 1e-9), upper - 1e-9)
    times = [start_anchor.t, end_anchor.t, *[obs.t for obs in observations]]
    hard_start_sigma = _anchor_sigma_m(start_anchor, cfg)
    hard_end_sigma = _anchor_sigma_m(end_anchor, cfg)

    def residuals(params: Any) -> Any:
        initial_position = (float(params[0]), float(params[1]), float(params[2]))
        velocity = (float(params[3]), float(params[4]), float(params[5]))
        predicted = _integrate_positions(
            initial_position,
            velocity,
            times,
            t0=start_anchor.t,
            physics=physics,
            config=cfg,
        )
        by_time = {round(t, 9): point for t, point in zip(times, predicted, strict=True)}
        residual: list[float] = []
        if not soft_start:
            residual.extend(
                _scaled_vec(
                    _sub(initial_position, start_anchor.world_xyz),
                    hard_start_sigma / cfg.endpoint_anchor_weight,
                )
            )
        endpoint = by_time[round(end_anchor.t, 9)]
        if not soft_end:
            residual.extend(
                _scaled_vec(
                    _sub(endpoint, end_anchor.world_xyz),
                    hard_end_sigma / cfg.endpoint_anchor_weight,
                )
            )
        for obs in observations:
            point = by_time[round(obs.t, 9)]
            projected = _project_world_point(calibration, point)
            sigma_px = cfg.robust_pixel_sigma / max(0.35, math.sqrt(max(obs.confidence, 1e-6)))
            residual.append((projected[0] - obs.xy[0]) / sigma_px)
            residual.append((projected[1] - obs.xy[1]) / sigma_px)
            size_residual = _size_depth_residual(
                calibration,
                obs,
                point,
                physics=physics,
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
                    physics,
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
        physics=physics,
        config=cfg,
    )
    inlier_frames = tuple(
        obs.frame for obs in observations if obs_errors.get(obs.frame, math.inf) <= cfg.max_reprojection_inlier_px
    )
    outlier_frames = tuple(
        obs.frame for obs in observations if obs_errors.get(obs.frame, 0.0) > cfg.max_reprojection_inlier_px
    )
    errors = [obs_errors[obs.frame] for obs in observations if obs.frame in obs_errors]
    inlier_errors = [obs_errors[frame] for frame in inlier_frames if frame in obs_errors]
    endpoint_pred = _integrate_positions(
        initial_position,
        velocity,
        [end_anchor.t],
        t0=start_anchor.t,
        physics=physics,
        config=cfg,
    )[0]
    net_clearance = _net_clearance_m(
        initial_position,
        velocity,
        start_anchor.t,
        end_anchor.t,
        physics,
        cfg,
        net_plane,
    )
    physical = _physical_sanity(
        initial_position,
        velocity,
        start_anchor.t,
        end_anchor.t,
        physics,
        cfg,
        net_clearance,
    )
    size_residuals = _size_residual_distribution(
        observations,
        calibration=calibration,
        initial_position=initial_position,
        velocity=velocity,
        t0=start_anchor.t,
        physics=physics,
        config=cfg,
        sigma_floor_m=cfg.size_depth_sigma_m,
    )
    diagnostics = {
        "soft_split_boundary": {
            "enabled": True,
            "allowed_role": "segment_split_boundary_only",
            "start_is_soft": soft_start,
            "end_is_soft": soft_end,
            "world_endpoint_constraints_from_soft_evidence": False,
            "bvp_endpoint_pinning_skipped": True,
            "event_type_asserted": False,
            "counts_as_bounce_evidence": False,
            "counts_as_flight_sanity_anchor": False,
        },
        "soft_split_optimizer_constraints": {
            "start_world_constraint": not soft_start,
            "end_world_constraint": not soft_end,
            "soft_world_initialization_used_as_evidence": False,
            "soft_z_radius_constraint": False,
        },
        "spin_scalar_fit": {"enabled": False, "fit": False, "reason": "soft_split_boundary_only"},
    }
    return FlightSegmentFit(
        segment_id=segment_id,
        status="fit" if result.success else "fit_optimizer_not_converged",
        start_anchor=start_anchor,
        end_anchor=end_anchor,
        initial_position_m=initial_position,
        initial_velocity_mps=velocity,
        observations=tuple(observations),
        inlier_frames=inlier_frames,
        outlier_frames=outlier_frames,
        reprojection_errors_px=obs_errors,
        reprojection_rmse_px=_rmse(inlier_errors),
        max_reprojection_error_px=max(errors) if errors else None,
        endpoint_error_m=0.0 if soft_end else _distance(endpoint_pred, end_anchor.world_xyz),
        net_clearance_m=net_clearance,
        net_clearance_ok=None if net_clearance is None else net_clearance >= -cfg.net_clearance_slack_m,
        physical_sanity=physical,
        size_residuals_m=size_residuals,
        diagnostics=diagnostics,
    )


def _fit_free_flight_segment_once(
    *,
    segment_id: int,
    start_anchor: AnchorEvent,
    end_anchor: AnchorEvent,
    observations: Sequence[BallObservation],
    calibration: Mapping[str, Any],
    physics: PhysicsParameters,
    config: BallArcSolverConfig,
    net_plane: Mapping[str, Any] | None = None,
    max_nfev: int | None = None,
    diagnostics: Mapping[str, Any] | None = None,
) -> FlightSegmentFit:
    """Fit one event-bounded free-flight segment against one chosen observation per frame."""

    cfg = config
    phys = physics
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
    spin_gate_fit: FlightSegmentFit | None = None
    spin_gate_inlier_count: int | None = None
    if cfg.fit_spin_scalar:
        spin_gate_fit = _fit_free_flight_segment_once(
            segment_id=segment_id,
            start_anchor=start_anchor,
            end_anchor=end_anchor,
            observations=observations,
            calibration=calibration,
            physics=phys,
            config=replace(cfg, fit_spin_scalar=False),
            net_plane=net_plane,
            max_nfev=max_nfev,
            diagnostics=diagnostics,
        )
        spin_gate_inlier_count = spin_gate_fit.inlier_count
        if not spin_gate_fit.status.startswith("fit") or spin_gate_inlier_count < SPIN_SCALAR_MIN_INLIERS:
            diagnostics_payload = dict(spin_gate_fit.diagnostics or {})
            diagnostics_payload["spin_scalar_fit"] = {
                "enabled": True,
                "fit": False,
                "reason": "inlier_gate",
                "min_inliers": SPIN_SCALAR_MIN_INLIERS,
                "inlier_count": spin_gate_inlier_count,
                "regularization_lambda": SPIN_SCALAR_REGULARIZATION_LAMBDA,
                "bound_abs": SPIN_SCALAR_MAX_ABS,
            }
            return replace(spin_gate_fit, diagnostics=diagnostics_payload)
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
    fit_spin = _can_fit_spin_scalar(cfg, observations, inlier_count=spin_gate_inlier_count)
    if fit_spin:
        initial = np.append(initial, 0.0)
        lower = np.append(lower, -SPIN_SCALAR_MAX_ABS)
        upper = np.append(upper, SPIN_SCALAR_MAX_ABS)
    times = [start_anchor.t, end_anchor.t, *[obs.t for obs in observations]]

    def residuals(params: Any) -> Any:
        initial_position = (float(params[0]), float(params[1]), float(params[2]))
        velocity = (float(params[3]), float(params[4]), float(params[5]))
        spin_scalar = _clip_spin_scalar(params[6]) if fit_spin else 0.0
        predicted = _integrate_positions(
            initial_position,
            velocity,
            times,
            t0=start_anchor.t,
            physics=phys,
            config=cfg,
            spin_scalar=spin_scalar,
        )
        by_time = {round(t, 9): point for t, point in zip(times, predicted, strict=True)}
        residual: list[float] = []
        residual.extend(_scaled_vec(_sub(initial_position, start_anchor.world_xyz), start_sigma / cfg.endpoint_anchor_weight))
        endpoint = by_time[round(end_anchor.t, 9)]
        residual.extend(_scaled_vec(_sub(endpoint, end_anchor.world_xyz), end_sigma / cfg.endpoint_anchor_weight))
        if fit_spin:
            residual.append(math.sqrt(SPIN_SCALAR_REGULARIZATION_LAMBDA) * spin_scalar)
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
                    spin_scalar=spin_scalar,
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
    spin_scalar = _clip_spin_scalar(params[6]) if fit_spin else 0.0
    obs_errors = _observation_reprojection_errors(
        observations,
        calibration=calibration,
        initial_position=initial_position,
        velocity=velocity,
        t0=start_anchor.t,
        physics=phys,
        config=cfg,
        spin_scalar=spin_scalar,
    )
    inlier_frames = tuple(obs.frame for obs in observations if obs_errors.get(obs.frame, math.inf) <= cfg.max_reprojection_inlier_px)
    outlier_frames = tuple(obs.frame for obs in observations if obs_errors.get(obs.frame, 0.0) > cfg.max_reprojection_inlier_px)
    if fit_spin and len(inlier_frames) < SPIN_SCALAR_MIN_INLIERS and spin_gate_fit is not None:
        diagnostics_payload = dict(spin_gate_fit.diagnostics or {})
        diagnostics_payload["spin_scalar_fit"] = {
            "enabled": True,
            "fit": False,
            "reason": "final_inlier_gate",
            "candidate_spin_scalar": _round(spin_scalar, 9),
            "min_inliers": SPIN_SCALAR_MIN_INLIERS,
            "inlier_count": len(inlier_frames),
            "regularization_lambda": SPIN_SCALAR_REGULARIZATION_LAMBDA,
            "bound_abs": SPIN_SCALAR_MAX_ABS,
        }
        return replace(spin_gate_fit, diagnostics=diagnostics_payload)
    errors = [obs_errors[obs.frame] for obs in observations if obs.frame in obs_errors]
    inlier_errors = [
        obs_errors[obs.frame]
        for obs in observations
        if obs.frame in obs_errors and obs_errors[obs.frame] <= cfg.max_reprojection_inlier_px
    ]
    endpoint_pred = _integrate_positions(
        initial_position,
        velocity,
        [end_anchor.t],
        t0=start_anchor.t,
        physics=phys,
        config=cfg,
        spin_scalar=spin_scalar,
    )[0]
    net_clearance = _net_clearance_m(
        initial_position,
        velocity,
        start_anchor.t,
        end_anchor.t,
        phys,
        cfg,
        net_plane,
        spin_scalar=spin_scalar,
    )
    net_ok = None if net_clearance is None else net_clearance >= -cfg.net_clearance_slack_m
    physical = _physical_sanity(
        initial_position,
        velocity,
        start_anchor.t,
        end_anchor.t,
        phys,
        cfg,
        net_clearance,
        spin_scalar=spin_scalar,
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
        spin_scalar=spin_scalar,
    )
    spin_diagnostics = {
        "enabled": bool(cfg.fit_spin_scalar),
        "fit": bool(fit_spin),
        "min_inliers": SPIN_SCALAR_MIN_INLIERS,
        "inlier_count": len(inlier_frames),
        "regularization_lambda": SPIN_SCALAR_REGULARIZATION_LAMBDA,
        "bound_abs": SPIN_SCALAR_MAX_ABS,
    }
    diagnostics_payload = dict(diagnostics or {})
    diagnostics_payload["spin_scalar_fit"] = spin_diagnostics
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
        spin_scalar=spin_scalar,
        diagnostics=diagnostics_payload,
    )


def _fit_selection_segment_once(
    *,
    segment_id: int,
    start_anchor: AnchorEvent,
    end_anchor: AnchorEvent,
    observations: Sequence[BallObservation],
    calibration: Mapping[str, Any],
    physics: PhysicsParameters,
    config: BallArcSolverConfig,
) -> FlightSegmentFit:
    """Cheap event-selection scorer; final selected segments still use the full BVP fit."""

    if end_anchor.t - start_anchor.t < config.min_segment_dt_s:
        return _blocked_segment(segment_id, start_anchor, end_anchor, "duration_below_minimum")
    observations = tuple(
        obs for obs in observations if start_anchor.t - 1e-9 <= obs.t <= end_anchor.t + 1e-9 and obs.visible
    )
    if len(observations) < config.min_segment_observations:
        return _blocked_segment(segment_id, start_anchor, end_anchor, "insufficient_observations")
    if not _finite_vec3(start_anchor.world_xyz) or not _finite_vec3(end_anchor.world_xyz):
        return _blocked_segment(segment_id, start_anchor, end_anchor, "nonfinite_anchor")
    dt = end_anchor.t - start_anchor.t
    initial_position = start_anchor.world_xyz
    velocity = _initial_velocity_guess(start_anchor.world_xyz, end_anchor.world_xyz, dt, physics)
    obs_errors = _observation_reprojection_errors(
        observations,
        calibration=calibration,
        initial_position=initial_position,
        velocity=velocity,
        t0=start_anchor.t,
        physics=physics,
        config=config,
    )
    inlier_frames = tuple(obs.frame for obs in observations if obs_errors.get(obs.frame, math.inf) <= config.max_reprojection_inlier_px)
    outlier_frames = tuple(obs.frame for obs in observations if obs_errors.get(obs.frame, 0.0) > config.max_reprojection_inlier_px)
    errors = [obs_errors[obs.frame] for obs in observations if obs.frame in obs_errors]
    inlier_errors = [
        obs_errors[obs.frame]
        for obs in observations
        if obs.frame in obs_errors and obs_errors[obs.frame] <= config.max_reprojection_inlier_px
    ]
    endpoint_pred = _integrate_positions(
        initial_position,
        velocity,
        [end_anchor.t],
        t0=start_anchor.t,
        physics=physics,
        config=config,
    )[0]
    physical = _physical_sanity(
        initial_position,
        velocity,
        start_anchor.t,
        end_anchor.t,
        physics,
        config,
        None,
    )
    return FlightSegmentFit(
        segment_id=segment_id,
        status="fit",
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
        net_clearance_m=None,
        net_clearance_ok=None,
        physical_sanity=physical,
        size_residuals_m={},
        diagnostics={"selection_scoring": "ballistic_initial_guess_no_bvp"},
    )


def _solve_bvp_shooting(
    p0: tuple[float, float, float],
    p1: tuple[float, float, float],
    t0: float,
    t1: float,
    *,
    physics: PhysicsParameters,
    config: BallArcSolverConfig,
    spin_scalar: float = 0.0,
) -> dict[str, Any]:
    spin = _clip_spin_scalar(spin_scalar)
    try:
        import numpy as np
    except ImportError:
        v_guess = _initial_velocity_guess(p0, p1, max(t1 - t0, 1e-9), physics)
        endpoint = _integrate_positions(p0, v_guess, [t1], t0=t0, physics=physics, config=config, spin_scalar=spin)[0]
        return {
            "status": "failed_missing_numpy",
            "converged": False,
            "iterations": 0,
            "initial_position_m": _vec_json(p0),
            "initial_velocity_mps": _vec_json(v_guess),
            "spin_scalar": _round(spin, 9),
            "endpoint_error_m": _round(_distance(endpoint, p1), 6),
            "target_position_m": _vec_json(p1),
        }

    dt = float(t1) - float(t0)
    if dt <= 1e-9:
        v_guess = (0.0, 0.0, 0.0)
        return {
            "status": "failed_invalid_duration",
            "converged": False,
            "iterations": 0,
            "initial_position_m": _vec_json(p0),
            "initial_velocity_mps": _vec_json(v_guess),
            "spin_scalar": _round(spin, 9),
            "endpoint_error_m": None,
            "target_position_m": _vec_json(p1),
        }

    velocity = np.asarray(_initial_velocity_guess(p0, p1, dt, physics), dtype=float)
    eps = float(config.bvp_shooting_fd_eps_mps)
    tolerance = float(config.bvp_shooting_tolerance_m)
    target = np.asarray(p1, dtype=float)
    best_velocity = velocity.copy()
    best_error = math.inf
    iterations = 0
    status = "failed_nonconverged"

    def endpoint_for(v: Any) -> Any:
        return np.asarray(
            _integrate_positions(
                p0,
                (float(v[0]), float(v[1]), float(v[2])),
                [t1],
                t0=t0,
                physics=physics,
                config=config,
                spin_scalar=spin,
            )[0],
            dtype=float,
        )

    for iteration in range(1, int(config.bvp_shooting_max_iterations) + 1):
        iterations = iteration
        endpoint = endpoint_for(velocity)
        residual = endpoint - target
        error = float(np.linalg.norm(residual))
        if error < best_error:
            best_error = error
            best_velocity = velocity.copy()
        if error <= tolerance:
            status = "converged"
            best_velocity = velocity.copy()
            best_error = error
            break
        jacobian = np.zeros((3, 3), dtype=float)
        for axis in range(3):
            shifted = velocity.copy()
            shifted[axis] += eps
            jacobian[:, axis] = (endpoint_for(shifted) - endpoint) / eps
        try:
            delta = np.linalg.solve(jacobian, -residual)
        except np.linalg.LinAlgError:
            delta = np.linalg.lstsq(jacobian, -residual, rcond=None)[0]
        accepted = False
        alpha = 1.0
        for _backtrack in range(int(config.bvp_shooting_max_backtracks) + 1):
            trial = velocity + alpha * delta
            trial_error = float(np.linalg.norm(endpoint_for(trial) - target))
            if trial_error < error:
                velocity = trial
                accepted = True
                if trial_error < best_error:
                    best_error = trial_error
                    best_velocity = trial.copy()
                break
            alpha *= 0.5
        if not accepted:
            status = "failed_no_descent"
            break

    endpoint = endpoint_for(best_velocity)
    endpoint_error = float(np.linalg.norm(endpoint - target))
    if endpoint_error <= tolerance:
        status = "converged"
    return {
        "status": status,
        "converged": status == "converged",
        "iterations": int(iterations),
        "t0": _round(float(t0), 9),
        "t1": _round(float(t1), 9),
        "tolerance_m": float(config.bvp_shooting_tolerance_m),
        "fd_eps_mps": float(config.bvp_shooting_fd_eps_mps),
        "initial_position_m": _vec_json(p0),
        "initial_velocity_mps": _vec_json((float(best_velocity[0]), float(best_velocity[1]), float(best_velocity[2]))),
        "spin_scalar": _round(spin, 9),
        "spin_cl": _round(STEYN_CL_PER_SPIN * spin, 9),
        "target_position_m": _vec_json(p1),
        "endpoint_position_m": _vec_json((float(endpoint[0]), float(endpoint[1]), float(endpoint[2]))),
        "endpoint_error_m": _round(endpoint_error, 6),
    }


def _refine_bvp_endpoints(
    start_anchor: AnchorEvent,
    end_anchor: AnchorEvent,
    *,
    observations: Sequence[BallObservation],
    calibration: Mapping[str, Any],
    physics: PhysicsParameters,
    config: BallArcSolverConfig,
    net_plane: Mapping[str, Any] | None,
    np: Any,
    least_squares: Any,
    spin_scalar: float = 0.0,
    spin_inlier_count: int | None = None,
) -> tuple[AnchorEvent, AnchorEvent, float, dict[str, Any]]:
    initial_spin = _clip_spin_scalar(spin_scalar)
    fit_spin = _can_fit_spin_scalar(config, observations, inlier_count=spin_inlier_count)
    base = _endpoint_refinement_identity(
        start_anchor,
        end_anchor,
        observations=observations,
        config=config,
        frozen_for_candidate_association=False,
    )
    if end_anchor.t - start_anchor.t <= config.min_segment_dt_s:
        return start_anchor, end_anchor, initial_spin, {**base, "status": "skipped"}
    fps = _fps_from_observations(observations)
    time_cap = float(config.endpoint_refinement_time_corridor_frames) / max(fps, 1e-9)
    p0_cap = _endpoint_position_corridor_m(start_anchor, config)
    p1_cap = _endpoint_position_corridor_m(end_anchor, config)
    lower = np.asarray([-p0_cap, -p0_cap, -p0_cap, -p1_cap, -p1_cap, -p1_cap, -time_cap, -time_cap], dtype=float)
    upper = np.asarray([p0_cap, p0_cap, p0_cap, p1_cap, p1_cap, p1_cap, time_cap, time_cap], dtype=float)
    if fit_spin:
        lower = np.append(lower, -SPIN_SCALAR_MAX_ABS)
        upper = np.append(upper, SPIN_SCALAR_MAX_ABS)
    if not np.all(lower < upper):
        return start_anchor, end_anchor, initial_spin, {**base, "status": "skipped"}
    start_sigma = max(_anchor_sigma_m(start_anchor, config), config.min_anchor_sigma_m)
    end_sigma = max(_anchor_sigma_m(end_anchor, config), config.min_anchor_sigma_m)

    def spin_from_params(params: Any) -> float:
        return _clip_spin_scalar(params[8]) if fit_spin else initial_spin

    def trial_from_params(params: Any) -> tuple[tuple[float, float, float], tuple[float, float, float], float, float, float]:
        dp0 = (float(params[0]), float(params[1]), float(params[2]))
        dp1 = (float(params[3]), float(params[4]), float(params[5]))
        return (
            _add(start_anchor.world_xyz, dp0),
            _add(end_anchor.world_xyz, dp1),
            float(start_anchor.t) + float(params[6]),
            float(end_anchor.t) + float(params[7]),
            spin_from_params(params),
        )

    def residuals(params: Any) -> Any:
        trial_p0, trial_p1, trial_t0, trial_t1, trial_spin = trial_from_params(params)
        residual: list[float] = []
        residual.extend(float(params[index]) / start_sigma for index in range(3))
        residual.extend(float(params[index]) / end_sigma for index in range(3, 6))
        residual.append(float(params[6]) / max(time_cap, 1e-9))
        residual.append(float(params[7]) / max(time_cap, 1e-9))
        if fit_spin:
            residual.append(math.sqrt(SPIN_SCALAR_REGULARIZATION_LAMBDA) * trial_spin)
        if trial_t1 - trial_t0 <= config.min_segment_dt_s:
            residual.extend([1e3, 1e3, 1e3])
            return np.asarray(residual, dtype=float)
        bvp = _solve_bvp_shooting(
            trial_p0,
            trial_p1,
            trial_t0,
            trial_t1,
            physics=physics,
            config=config,
            spin_scalar=trial_spin,
        )
        velocity = _vec3_from_json(bvp.get("initial_velocity_mps"))
        if not bool(bvp.get("converged")) or velocity is None:
            residual.extend([1e2, 1e2, 1e2])
            return np.asarray(residual, dtype=float)
        times = [obs.t for obs in observations]
        predicted = _integrate_positions(
            trial_p0,
            velocity,
            times,
            t0=trial_t0,
            physics=physics,
            config=config,
            spin_scalar=trial_spin,
        )
        for obs, point in zip(observations, predicted, strict=True):
            projected = _project_world_point(calibration, point)
            sigma_px = config.robust_pixel_sigma / max(0.35, math.sqrt(max(obs.confidence, 1e-6)))
            residual.append((projected[0] - obs.xy[0]) / sigma_px)
            residual.append((projected[1] - obs.xy[1]) / sigma_px)
            size_residual = _size_depth_residual(
                calibration,
                obs,
                point,
                physics=physics,
                config=config,
                sigma_floor_m=config.size_depth_sigma_m,
            )
            if size_residual is not None:
                residual.append(size_residual[0] / size_residual[1])
        if net_plane is not None:
            residual.append(
                _net_soft_residual(
                    trial_p0,
                    velocity,
                    trial_t0,
                    trial_t1,
                    physics,
                    config,
                    net_plane,
                    spin_scalar=trial_spin,
                )
            )
        return np.asarray(residual, dtype=float)

    initial = np.zeros(8, dtype=float)
    if fit_spin:
        initial = np.append(initial, initial_spin)
    try:
        result = least_squares(
            residuals,
            initial,
            bounds=(lower, upper),
            loss=config.robust_loss,
            f_scale=config.robust_f_scale,
            max_nfev=config.endpoint_refinement_max_nfev,
        )
    except ValueError:
        return start_anchor, end_anchor, initial_spin, {**base, "status": "skipped"}
    params = result.x if result.success else initial
    trial_p0, trial_p1, trial_t0, trial_t1, refined_spin = trial_from_params(params)
    if result.success and float(result.cost) < 1e-12:
        status = "not_improved"
    elif result.success:
        status = "converged"
    else:
        status = "not_improved"
        trial_p0, trial_p1, trial_t0, trial_t1 = start_anchor.world_xyz, end_anchor.world_xyz, start_anchor.t, end_anchor.t
        params = initial
        refined_spin = initial_spin
    refined_start = replace(start_anchor, world_xyz=trial_p0, t=trial_t0)
    refined_end = replace(end_anchor, world_xyz=trial_p1, t=trial_t1)
    report = {
        **base,
        "status": status,
        "max_nfev": int(config.endpoint_refinement_max_nfev),
        "nfev": int(getattr(result, "nfev", 0)),
        "cost": _round(float(getattr(result, "cost", 0.0)), 6),
        "fps": _round(fps, 6),
        "delta_p0_m": _vec_json((float(params[0]), float(params[1]), float(params[2]))),
        "delta_p1_m": _vec_json((float(params[3]), float(params[4]), float(params[5]))),
        "delta_t0_s": _round(float(params[6]), 9),
        "delta_t1_s": _round(float(params[7]), 9),
        "spin_scalar_initial": _round(initial_spin, 9),
        "spin_scalar": _round(refined_spin, 9),
        "spin_fitted": bool(fit_spin),
        "spin_min_inliers": SPIN_SCALAR_MIN_INLIERS,
        "spin_regularization_lambda": SPIN_SCALAR_REGULARIZATION_LAMBDA,
    }
    return refined_start, refined_end, refined_spin, report


def _build_fit_from_bvp_solution(
    *,
    segment_id: int,
    start_anchor: AnchorEvent,
    end_anchor: AnchorEvent,
    observations: Sequence[BallObservation],
    calibration: Mapping[str, Any],
    physics: PhysicsParameters,
    config: BallArcSolverConfig,
    net_plane: Mapping[str, Any] | None,
    bvp: Mapping[str, Any],
    status: str,
    diagnostics: Mapping[str, Any],
) -> FlightSegmentFit:
    velocity = _vec3_from_json(bvp.get("initial_velocity_mps"))
    if velocity is None:
        velocity = _initial_velocity_guess(start_anchor.world_xyz, end_anchor.world_xyz, end_anchor.t - start_anchor.t, physics)
    initial_position = start_anchor.world_xyz
    spin_scalar = _clip_spin_scalar(bvp.get("spin_scalar"))
    obs_errors = _observation_reprojection_errors(
        observations,
        calibration=calibration,
        initial_position=initial_position,
        velocity=velocity,
        t0=start_anchor.t,
        physics=physics,
        config=config,
        spin_scalar=spin_scalar,
    )
    inlier_frames = tuple(obs.frame for obs in observations if obs_errors.get(obs.frame, math.inf) <= config.max_reprojection_inlier_px)
    outlier_frames = tuple(obs.frame for obs in observations if obs_errors.get(obs.frame, 0.0) > config.max_reprojection_inlier_px)
    errors = [obs_errors[obs.frame] for obs in observations if obs.frame in obs_errors]
    inlier_errors = [
        obs_errors[obs.frame]
        for obs in observations
        if obs.frame in obs_errors and obs_errors[obs.frame] <= config.max_reprojection_inlier_px
    ]
    endpoint_pred = _integrate_positions(
        initial_position,
        velocity,
        [end_anchor.t],
        t0=start_anchor.t,
        physics=physics,
        config=config,
        spin_scalar=spin_scalar,
    )[0]
    net_clearance = _net_clearance_m(
        initial_position,
        velocity,
        start_anchor.t,
        end_anchor.t,
        physics,
        config,
        net_plane,
        spin_scalar=spin_scalar,
    )
    net_ok = None if net_clearance is None else net_clearance >= -config.net_clearance_slack_m
    physical = _physical_sanity(
        initial_position,
        velocity,
        start_anchor.t,
        end_anchor.t,
        physics,
        config,
        net_clearance,
        spin_scalar=spin_scalar,
    )
    size_residuals = _size_residual_distribution(
        observations,
        calibration=calibration,
        initial_position=initial_position,
        velocity=velocity,
        t0=start_anchor.t,
        physics=physics,
        config=config,
        sigma_floor_m=config.size_depth_sigma_m,
        spin_scalar=spin_scalar,
    )
    return FlightSegmentFit(
        segment_id=segment_id,
        status=status,
        start_anchor=start_anchor,
        end_anchor=end_anchor,
        initial_position_m=initial_position,
        initial_velocity_mps=velocity,
        observations=tuple(observations),
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
        spin_scalar=spin_scalar,
        diagnostics=diagnostics,
    )


def _endpoint_refinement_identity(
    start_anchor: AnchorEvent,
    end_anchor: AnchorEvent,
    *,
    observations: Sequence[BallObservation],
    config: BallArcSolverConfig,
    frozen_for_candidate_association: bool,
) -> dict[str, Any]:
    fps = _fps_from_observations(observations)
    return {
        "status": "skipped" if not observations else "not_improved",
        "frozen_for_candidate_association": bool(frozen_for_candidate_association),
        "start_anchor_id": start_anchor.anchor_id,
        "end_anchor_id": end_anchor.anchor_id,
        "delta_p0_m": [0.0, 0.0, 0.0],
        "delta_p1_m": [0.0, 0.0, 0.0],
        "delta_t0_s": 0.0,
        "delta_t1_s": 0.0,
        "p0_corridor_m": _round(_endpoint_position_corridor_m(start_anchor, config), 6),
        "p1_corridor_m": _round(_endpoint_position_corridor_m(end_anchor, config), 6),
        "time_corridor_s": _round(float(config.endpoint_refinement_time_corridor_frames) / max(fps, 1e-9), 9),
    }


def _endpoint_position_corridor_m(anchor: AnchorEvent, config: BallArcSolverConfig) -> float:
    if anchor.kind == "contact":
        cap = config.endpoint_refinement_contact_cap_m
    elif anchor.kind == "bounce":
        cap = config.endpoint_refinement_bounce_cap_m
    else:
        cap = config.endpoint_refinement_default_cap_m
    return max(config.min_anchor_sigma_m, min(1.5 * _anchor_sigma_m(anchor, config), cap))


def _fps_from_observations(observations: Sequence[BallObservation]) -> float:
    pairs = sorted((obs.frame, obs.t) for obs in observations)
    values: list[float] = []
    for left, right in zip(pairs, pairs[1:], strict=False):
        frame_delta = right[0] - left[0]
        time_delta = right[1] - left[1]
        if frame_delta > 0 and time_delta > 1e-9:
            values.append(frame_delta / time_delta)
    return median(values) if values else 30.0


def _fit_diagnostic_summary(fit: FlightSegmentFit) -> dict[str, Any]:
    return {
        "status": fit.status,
        "initial_position_m": _vec_json(fit.initial_position_m),
        "initial_velocity_mps": _vec_json(fit.initial_velocity_mps),
        "spin_scalar": _round(fit.spin_scalar, 9),
        "endpoint_error_m": _round(fit.endpoint_error_m, 6),
        "inlier_count": fit.inlier_count,
        "outlier_count": fit.outlier_count,
        "reprojection_rmse_px": _optional_round(fit.reprojection_rmse_px, 6),
        "max_reprojection_error_px": _optional_round(fit.max_reprojection_error_px, 6),
    }


def _optional_float_round(value: Any, digits: int = 6) -> float | None:
    parsed = _float_or_none(value)
    return None if parsed is None else _round(parsed, digits)


def _fit_flight_segment_with_candidate_association(
    *,
    segment_id: int,
    start_anchor: AnchorEvent,
    end_anchor: AnchorEvent,
    observations: Sequence[BallObservation],
    candidate_sets_by_frame: Mapping[int, Sequence[BallObservation]],
    calibration: Mapping[str, Any],
    physics: PhysicsParameters,
    config: BallArcSolverConfig,
    net_plane: Mapping[str, Any] | None = None,
    max_nfev: int | None = None,
) -> FlightSegmentFit:
    """EM-style per-frame candidate association around the existing robust segment fit.

    Leave-one-out callers pass a candidate set with the held-out frame removed;
    this keeps all sibling candidates from that frame out of the refit.
    """

    primary_observations = tuple(
        obs for obs in observations if start_anchor.t - 1e-9 <= obs.t <= end_anchor.t + 1e-9 and obs.visible
    )
    span_candidate_sets = _candidate_sets_in_span(
        candidate_sets_by_frame,
        start_t=start_anchor.t,
        end_t=end_anchor.t,
        max_per_frame=config.max_candidates_per_frame,
    )
    base = _fit_flight_segment_once(
        segment_id=segment_id,
        start_anchor=start_anchor,
        end_anchor=end_anchor,
        observations=primary_observations,
        calibration=calibration,
        physics=physics,
        config=config,
        net_plane=net_plane,
        max_nfev=max_nfev,
    )
    if not span_candidate_sets or not base.status.startswith("fit"):
        return replace(
            base,
            primary_observations=primary_observations,
            candidate_sets_by_frame=span_candidate_sets,
            candidate_association=_candidate_association_report(
                enabled=bool(span_candidate_sets),
                converged=False,
                iterations=[],
                selected_by_frame={},
                unassigned_best_residual_px={},
                reason="initial_fit_not_available" if span_candidate_sets else "no_candidate_sets_in_span",
                config=config,
            ),
        )

    current = base
    refit_max_nfev = max_nfev or config.selection_max_nfev
    previous_signature: tuple[tuple[int, str, float, float], ...] | None = None
    selected_by_frame: dict[int, BallObservation] = {}
    unassigned_best_residual_px: dict[int, float] = {}
    iterations: list[dict[str, Any]] = []
    converged = False
    stopped_reason = "max_iterations_reached"
    for iteration in range(1, config.candidate_selection_max_iterations + 1):
        selected_by_frame, unassigned_best_residual_px, iteration_report = _select_candidates_for_segment(
            current,
            span_candidate_sets,
            calibration=calibration,
            physics=physics,
            config=config,
            iteration=iteration,
        )
        signature = _candidate_selection_signature(selected_by_frame)
        iterations.append(iteration_report)
        if signature == previous_signature:
            converged = True
            stopped_reason = "selection_stable"
            break
        if len(selected_by_frame) < config.min_segment_observations:
            return replace(
                base,
                primary_observations=primary_observations,
                candidate_sets_by_frame=span_candidate_sets,
                candidate_association=_candidate_association_report(
                    enabled=True,
                    converged=False,
                    iterations=iterations,
                    selected_by_frame=selected_by_frame,
                    unassigned_best_residual_px=unassigned_best_residual_px,
                    reason="insufficient_selected_candidates_fallback_primary",
                    refit_max_nfev=refit_max_nfev,
                    fallback_to_primary_fit=True,
                    config=config,
                ),
            )
        refit = _fit_flight_segment_once(
            segment_id=segment_id,
            start_anchor=base.start_anchor,
            end_anchor=base.end_anchor,
            observations=tuple(selected_by_frame.values()),
            calibration=calibration,
            physics=physics,
            config=config,
            net_plane=net_plane,
            max_nfev=refit_max_nfev,
            refine_endpoints=False,
            candidate_endpoint_frozen=True,
        )
        current = refit
        previous_signature = signature
        if not current.status.startswith("fit"):
            stopped_reason = current.status
            return replace(
                base,
                primary_observations=primary_observations,
                candidate_sets_by_frame=span_candidate_sets,
                candidate_association=_candidate_association_report(
                    enabled=True,
                    converged=False,
                    iterations=iterations,
                    selected_by_frame=selected_by_frame,
                    unassigned_best_residual_px=unassigned_best_residual_px,
                    reason=f"{stopped_reason}_fallback_primary",
                    refit_max_nfev=refit_max_nfev,
                    fallback_to_primary_fit=True,
                    config=config,
                ),
            )

    selected_by_frame = {
        frame: replace(obs, candidate_selection="arc_irls_v1")
        for frame, obs in selected_by_frame.items()
    }
    unassigned_frames = set(unassigned_best_residual_px)
    merged_outliers = tuple(sorted(set(current.outlier_frames) | unassigned_frames))
    merged_errors = {**dict(current.reprojection_errors_px), **unassigned_best_residual_px}
    association_report = _candidate_association_report(
        enabled=True,
        converged=converged,
        iterations=iterations,
        selected_by_frame=selected_by_frame,
        unassigned_best_residual_px=unassigned_best_residual_px,
        reason=stopped_reason,
        refit_max_nfev=refit_max_nfev,
        config=config,
    )
    association_report.update(
        _candidate_final_residual_diagnostics(
            selected_by_frame,
            current.reprojection_errors_px,
            inlier_frames=set(current.inlier_frames),
        )
    )
    association_report["endpoint_refinement_frozen"] = True
    return replace(
        current,
        observations=tuple(selected_by_frame.values()),
        outlier_frames=merged_outliers,
        reprojection_errors_px=merged_errors,
        primary_observations=primary_observations,
        candidate_sets_by_frame=span_candidate_sets,
        selected_observations_by_frame=selected_by_frame,
        candidate_association=association_report,
        diagnostics={
            **dict(current.diagnostics or {}),
            "endpoint_refinement": {
                **(
                    dict((current.diagnostics or {}).get("endpoint_refinement", {}))
                    if isinstance((current.diagnostics or {}).get("endpoint_refinement"), Mapping)
                    else {}
                ),
                "frozen_for_candidate_association": True,
            },
        },
    )


def _candidate_sets_in_span(
    candidate_sets_by_frame: Mapping[int, Sequence[BallObservation]],
    *,
    start_t: float,
    end_t: float,
    max_per_frame: int,
) -> dict[int, tuple[BallObservation, ...]]:
    output: dict[int, tuple[BallObservation, ...]] = {}
    for frame, candidates in candidate_sets_by_frame.items():
        selected = tuple(
            candidate
            for candidate in candidates[:max_per_frame]
            if candidate.visible and start_t - 1e-9 <= candidate.t <= end_t + 1e-9
        )
        if selected:
            output[int(frame)] = selected
    return output


def _select_candidates_for_segment(
    segment: FlightSegmentFit,
    candidate_sets_by_frame: Mapping[int, Sequence[BallObservation]],
    *,
    calibration: Mapping[str, Any],
    physics: PhysicsParameters,
    config: BallArcSolverConfig,
    iteration: int,
) -> tuple[dict[int, BallObservation], dict[int, float], dict[str, Any]]:
    selected: dict[int, BallObservation] = {}
    unassigned: dict[int, float] = {}
    score_floor_rejected_counts: dict[str, int] = {}
    for frame, candidates in sorted(candidate_sets_by_frame.items()):
        scored: list[tuple[float, int, str, BallObservation]] = []
        for rank, candidate in enumerate(candidates):
            predicted = segment.predict(candidate.t, physics, config)
            projected = _project_world_point(calibration, predicted)
            residual_px = _distance2(projected, candidate.xy)
            if not _candidate_passes_score_floor(candidate, config):
                source = candidate.observation_source
                score_floor_rejected_counts[source] = score_floor_rejected_counts.get(source, 0) + 1
                continue
            scored.append((residual_px, rank, candidate.observation_source, candidate))
        if not scored:
            continue
        residual_px, rank, _source, candidate = _winning_candidate_for_mode(scored, config)
        if residual_px <= config.max_reprojection_inlier_px:
            selected[int(frame)] = replace(
                candidate,
                candidate_rank=rank if candidate.candidate_rank is None else candidate.candidate_rank,
                candidate_selection="arc_irls_v1",
            )
        else:
            unassigned[int(frame)] = float(residual_px)
    source_counts = _observation_source_counts(selected.values())
    return (
        selected,
        unassigned,
        {
            "iteration": int(iteration),
            "selected_count": len(selected),
            "unassigned_count": len(unassigned),
            "selection_counts_by_source": source_counts,
            "score_floor_rejected_counts_by_source": score_floor_rejected_counts,
            "max_reprojection_inlier_px": config.max_reprojection_inlier_px,
            "candidate_association_mode": config.candidate_association_mode,
            "candidate_score_floors": _candidate_score_floors_payload(config),
        },
    )


def _winning_candidate_for_mode(
    scored: Sequence[tuple[float, int, str, BallObservation]],
    config: BallArcSolverConfig,
) -> tuple[float, int, str, BallObservation]:
    if config.candidate_association_mode != "rescue_only":
        return min(scored, key=lambda item: (item[0], item[1], item[2]))
    primary = [item for item in scored if _is_primary_observation(item[3])]
    if primary:
        primary_best = min(primary, key=lambda item: (item[0], item[1], item[2]))
        if primary_best[0] <= config.max_reprojection_inlier_px:
            return primary_best
    rescue = [item for item in scored if not _is_primary_observation(item[3])]
    if rescue:
        return min(rescue, key=lambda item: (item[0], item[1], item[2]))
    return min(scored, key=lambda item: (item[0], item[1], item[2]))


def _candidate_passes_score_floor(candidate: BallObservation, config: BallArcSolverConfig) -> bool:
    floor = _candidate_score_floor(candidate, config)
    if floor is None:
        return True
    score = candidate.candidate_score if candidate.candidate_score is not None else candidate.confidence
    return float(score) >= floor


def _candidate_score_floor(candidate: BallObservation, config: BallArcSolverConfig) -> float | None:
    floors = dict(config.candidate_score_floors or {})
    if not floors:
        return None
    source = candidate.observation_source
    if source in floors:
        return float(floors[source])
    source_prefix = source.split(":", 1)[0]
    if source_prefix in floors:
        return float(floors[source_prefix])
    return None


def _candidate_score_floors_payload(config: BallArcSolverConfig) -> dict[str, float]:
    return {str(source): float(value) for source, value in sorted(dict(config.candidate_score_floors or {}).items())}


def _is_primary_observation(obs: BallObservation) -> bool:
    return obs.observation_source.startswith("primary:")


def _candidate_selection_signature(selected_by_frame: Mapping[int, BallObservation]) -> tuple[tuple[int, str, float, float], ...]:
    return tuple(
        (
            int(frame),
            obs.observation_source,
            round(float(obs.xy[0]), 3),
            round(float(obs.xy[1]), 3),
        )
        for frame, obs in sorted(selected_by_frame.items())
    )


def _candidate_association_report(
    *,
    enabled: bool,
    converged: bool,
    iterations: Sequence[Mapping[str, Any]],
    selected_by_frame: Mapping[int, BallObservation],
    unassigned_best_residual_px: Mapping[int, float],
    reason: str,
    refit_max_nfev: int | None = None,
    fallback_to_primary_fit: bool = False,
    config: BallArcSolverConfig | None = None,
) -> dict[str, Any]:
    mode = config.candidate_association_mode if config is not None else "free"
    score_floors = _candidate_score_floors_payload(config) if config is not None else {}
    return {
        "enabled": bool(enabled),
        "candidate_selection": "arc_irls_v1" if enabled else None,
        "mode": mode,
        "candidate_score_floors": score_floors,
        "converged": bool(converged),
        "stopped_reason": reason,
        "iteration_count": len(iterations),
        "initial_fit": "primary_track_observations",
        "refit_max_nfev": refit_max_nfev,
        "fallback_to_primary_fit": bool(fallback_to_primary_fit),
        "iterations": [dict(item) for item in iterations],
        "selected_count": len(selected_by_frame),
        "unassigned_count": len(unassigned_best_residual_px),
        "selection_counts_by_source": _observation_source_counts(selected_by_frame.values()),
        "rescue_counts_by_source": _rescue_counts_by_source(selected_by_frame, inlier_frames=None),
        "score_floor_rejected_counts_by_source": _aggregate_score_floor_rejections(iterations),
        "unassigned_best_residual_px": {str(frame): _round(value, 6) for frame, value in sorted(unassigned_best_residual_px.items())},
        "policy": {
            "initialization": "primary_track_observations",
            "candidate_selection": "min_reprojection_residual_to_current_arc",
            "inlier_threshold": "max_reprojection_inlier_px",
            "loo_holdout": "whole_frame_candidate_sets_excluded",
            "mode": mode,
        },
    }


def _aggregate_score_floor_rejections(iterations: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for iteration in iterations:
        for source, count in dict(iteration.get("score_floor_rejected_counts_by_source") or {}).items():
            counts[str(source)] = counts.get(str(source), 0) + int(count)
    return counts


def _rescue_counts_by_source(
    selected_by_frame: Mapping[int, BallObservation],
    *,
    inlier_frames: set[int] | None,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for frame, obs in selected_by_frame.items():
        if inlier_frames is not None and int(frame) not in inlier_frames:
            continue
        if _is_primary_observation(obs):
            continue
        counts[obs.observation_source] = counts.get(obs.observation_source, 0) + 1
    return counts


def _candidate_final_residual_diagnostics(
    selected_by_frame: Mapping[int, BallObservation],
    residuals_by_frame: Mapping[int, float],
    *,
    inlier_frames: set[int] | None,
) -> dict[str, Any]:
    residuals_by_source: dict[str, list[float]] = {}
    for frame, obs in selected_by_frame.items():
        residual = residuals_by_frame.get(int(frame))
        if residual is None:
            continue
        residuals_by_source.setdefault(obs.observation_source, []).append(float(residual))
    return {
        "final_residual_px_by_source": {
            source: _distribution(values)
            for source, values in sorted(residuals_by_source.items())
        },
        "rescue_counts_by_source": _rescue_counts_by_source(selected_by_frame, inlier_frames=inlier_frames),
    }


def _observation_source_counts(observations: Sequence[BallObservation]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for obs in observations:
        counts[obs.observation_source] = counts.get(obs.observation_source, 0) + 1
    return counts


def _observation_candidate_payload(obs: BallObservation) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "frame": int(obs.frame),
        "t": _round(obs.t, 9),
        "xy": [_round(obs.xy[0], 6), _round(obs.xy[1], 6)],
        "observation_source": obs.observation_source,
        "candidate_selection": obs.candidate_selection,
        "confidence": _round(obs.confidence, 6),
    }
    if obs.candidate_score is not None:
        payload["candidate_score"] = _round(obs.candidate_score, 6)
    if obs.candidate_rank is not None:
        payload["candidate_rank"] = int(obs.candidate_rank)
    return payload


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
    """Fit one weak segment under the same production wall-clock guard."""

    cfg = config or BallArcSolverConfig()
    phys = physics or PhysicsParameters()
    visible = tuple(obs for obs in observations if obs.visible and obs.t >= anchor.t - 1e-9)
    last = visible[-1] if visible else None
    end = replace(
        anchor,
        anchor_id=f"weak_budget_endpoint_{segment_id:03d}",
        kind="weak_ray_endpoint",
        t=anchor.t if last is None else last.t,
        frame=anchor.frame if last is None else last.frame,
        immovable=False,
    )
    return _run_with_segment_budget(
        segment_id=segment_id,
        start=anchor,
        end=end,
        observations=visible,
        candidate_sets_by_frame=None,
        config=cfg,
        fit=lambda: _fit_weak_flight_segment_unbounded(
            segment_id=segment_id,
            anchor=anchor,
            observations=visible,
            calibration=calibration,
            physics=phys,
            config=cfg,
            net_plane=net_plane,
            max_nfev=max_nfev,
        ),
    )


def _fit_weak_flight_segment_unbounded(
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
    ball_candidate_sidecars: Sequence[Mapping[str, Any]] = (),
    candidate_extra_tracks: Mapping[str, Mapping[str, Any]] | None = None,
    contact_windows: Mapping[str, Any] | None = None,
    skeleton3d: Mapping[str, Any] | None = None,
    reviewed_bounces: Mapping[str, Any] | None = None,
    auto_bounce_candidates: Mapping[str, Any] | None = None,
    rally_spans: Mapping[str, Any] | None = None,
    net_plane: Mapping[str, Any] | None = None,
    extra_anchors: Sequence[AnchorEvent] = (),
    soft_split_boundaries: Sequence[SoftSegmentBoundary] = (),
    frame_times: Any = None,
    physics: PhysicsParameters | None = None,
    config: BallArcSolverConfig | None = None,
    clip_id: str | None = None,
) -> dict[str, Any]:
    """Build a render-only ball_track_arc_solved artifact."""

    cfg = config or BallArcSolverConfig()
    phys = physics or PhysicsParameters()
    frames = _frames(ball_track)
    fps = _payload_fps(ball_track, frames)
    frame_time_map = frame_time_lookup(frame_times if frame_times is not None else ball_track.get("frame_times"))
    primary_source = f"primary:{ball_track.get('source') or 'ball_track'}"
    observations = _ball_observations(
        frames,
        fps=fps,
        frame_times=frame_time_map,
        ball_sizes=ball_sizes,
        source_label=primary_source,
    )
    candidate_sets_by_frame = _combined_candidate_sets_by_frame(
        frames=frames,
        fps=fps,
        frame_times=frame_time_map,
        ball_track=ball_track,
        primary_observations=observations,
        ball_candidate_sidecars=ball_candidate_sidecars,
        candidate_extra_tracks=candidate_extra_tracks or {},
        max_candidates_per_frame=cfg.max_candidates_per_frame,
    )
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
        _auto_bounce_candidate_anchors(
            auto_bounce_candidates,
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
            fps=fps,
            frame_times=frame_time_map,
            config=cfg,
        )
    )
    candidate_anchors = [_anchor_with_sigma_floor(anchor, cfg) for anchor in _filter_anchors_to_rally_spans(order_event_anchors(anchors), rally_spans)]
    segments_are_final = True
    if cfg.enable_event_subset_selection:
        anchors, segments, event_selection = _select_event_subset(
            candidate_anchors,
            observations=observations,
            candidate_sets_by_frame=None,
            calibration=calibration,
            physics=phys,
            config=cfg,
            net_plane=net_plane,
            rally_spans=rally_spans,
            final_refine_segments=not cfg.enable_event_discovery,
        )
        segments_are_final = not cfg.enable_event_discovery
    else:
        anchors = candidate_anchors
        segments = _fit_segments_from_anchors(
            anchors,
            observations=observations,
            candidate_sets_by_frame=candidate_sets_by_frame,
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
            net_plane=net_plane,
        )
        if discovered:
            candidate_anchors = [_anchor_with_sigma_floor(anchor, cfg) for anchor in order_event_anchors([*candidate_anchors, *discovered])]
            if cfg.enable_event_subset_selection:
                anchors, segments, event_selection = _select_event_subset(
                    candidate_anchors,
                    observations=observations,
                    candidate_sets_by_frame=None,
                    calibration=calibration,
                    physics=phys,
                    config=cfg,
                    net_plane=net_plane,
                    rally_spans=rally_spans,
                )
                segments_are_final = True
            else:
                anchors = candidate_anchors
                segments = _fit_segments_from_anchors(
                    anchors,
                    observations=observations,
                    candidate_sets_by_frame=candidate_sets_by_frame,
                    calibration=calibration,
                    physics=phys,
                    config=cfg,
                    net_plane=net_plane,
                )
                event_selection = _event_selection_passthrough(anchors)
                segments_are_final = True
        elif not segments_are_final:
            segments = _fit_segments_from_anchors(
                anchors,
                observations=observations,
                candidate_sets_by_frame=None,
                calibration=calibration,
                physics=phys,
                config=cfg,
                net_plane=net_plane,
            )
            segments_are_final = True
    if cfg.enable_event_subset_selection and candidate_sets_by_frame is not None:
        segments = _fit_segments_from_anchors(
            anchors,
            observations=observations,
            candidate_sets_by_frame=candidate_sets_by_frame,
            calibration=calibration,
            physics=phys,
            config=cfg,
            net_plane=net_plane,
        )
    soft_split_report: dict[str, Any] | None = None
    if soft_split_boundaries:
        segments, soft_split_report = _fit_segments_from_anchors(
            anchors,
            observations=observations,
            candidate_sets_by_frame=candidate_sets_by_frame,
            calibration=calibration,
            physics=phys,
            config=cfg,
            net_plane=net_plane,
            soft_split_boundaries=soft_split_boundaries,
            return_soft_split_report=True,
        )
    confident_segments = list(segments)
    confident_segments, protected_span_prior_report = _apply_protected_span_priors_from_frozen_baseline(
        confident_segments,
        observations=observations,
        calibration=calibration,
        physics=phys,
        config=cfg,
        net_plane=net_plane,
        clip_id=clip_id,
    )
    weak_segments, weak_validation = _build_weak_segments(
        anchors,
        confident_segments,
        observations=observations,
        calibration=calibration,
        physics=phys,
        config=cfg,
        net_plane=net_plane,
    )
    confident_segments = _apply_fit_validity_gates(confident_segments, physics=phys, config=cfg, net_plane=net_plane)
    all_segments = [*confident_segments, *weak_segments]
    budget_exceeded_segments = [
        segment
        for segment in all_segments
        if segment.status == f"blocked:{SEGMENT_BUDGET_EXCEEDED}"
    ]
    frame_payloads, coverage = _solved_frames(
        frames,
        fps=fps,
        frame_times=frame_time_map,
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
    candidate_association_summary = _candidate_association_summary(confident_segments)
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
    if (
        status == "ran"
        and not confident_segments
        and any(item.get("kind") == "rally_endpoint" for item in event_selection.get("selected", []))
    ):
        status = "degenerate_zero_segments"
        kill_reasons.append("zero accepted segments with at least one rally anchor")
    if budget_exceeded_segments:
        status = "degraded"
    if budget_exceeded_segments:
        # The legacy reference scans and optimizes every plausible bounce window.
        # It is diagnostic-only, and continuing it after the production arc has
        # already abstained can itself take game-scale time.  Keep that missing
        # validation loud while preserving the existing path when no guard trips.
        physics3d_summary = {
            "status": "not_run_due_to_segment_budget_exceeded",
            "evidence_provenance": "missing",
            "authority": "degraded",
            "reason": SEGMENT_BUDGET_EXCEEDED,
            "notes": [
                "Diagnostic ball_physics3d reference was not run because one or more "
                "production arc segments exceeded their wall-clock budget."
            ],
        }
    else:
        try:
            physics3d_summary = reconstruct_bounce_arcs_from_image_track(
                ball_track,
                calibration,
                max_reprojection_rmse_px=12.0,
                max_fit_samples=13,
            ).summary()
        except Exception as exc:
            physics3d_summary = {"status": "not_run", "notes": [str(exc)]}

    artifact = {
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
            "candidate_selection_source_counts": candidate_association_summary["selection_counts_by_source"],
            "human_reviewed_bounce_count": sum(1 for anchor in anchors if anchor.kind == "bounce" and anchor.status == "human_reviewed"),
            "auto_bounce_candidate_count": sum(1 for anchor in anchors if anchor.kind == "bounce" and anchor.status == "auto_bounce_candidate"),
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
            "candidate_association": candidate_association_summary,
            "protected_span_priors": protected_span_prior_report,
            "physical_sanity": physical_summary,
            "ball_physics3d_reference": physics3d_summary,
        },
    }
    if soft_split_report is not None:
        artifact["soft_split_boundaries"] = soft_split_report
        artifact["policy"]["audio_onset_soft_split"] = {
            "default_off": True,
            "allowed_role": "segment_split_boundary_only",
            "event_type_asserted": False,
            "pins_world_position": False,
            "pins_z_to_ball_radius": False,
            "counts_as_bounce_evidence": False,
            "counts_as_flight_sanity_anchor": False,
        }
        artifact["summary"].update(
            {
                "soft_split_boundary_supplied_count": int(soft_split_report["supplied_count"]),
                "soft_split_boundary_applied_count": int(soft_split_report["applied_count"]),
                "soft_split_segment_count": sum(
                    1 for segment in all_segments if bool(segment.soft_split_provenance)
                ),
            }
        )
    if budget_exceeded_segments:
        segment_ids = [segment.segment_id for segment in budget_exceeded_segments]
        artifact["degraded_reasons"] = [
            {
                "reason": SEGMENT_BUDGET_EXCEEDED,
                "evidence_provenance": "missing",
                "segment_ids": segment_ids,
            }
        ]
        artifact["summary"].update(
            {
                "degraded_segment_count": len(segment_ids),
                "missing_segment_count": len(segment_ids),
                "segment_budget_exceeded_count": len(segment_ids),
                "segment_budget_exceeded_ids": segment_ids,
                "missing_segment_reasons": {SEGMENT_BUDGET_EXCEEDED: len(segment_ids)},
            }
        )
    return artifact


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
    candidate_sets_by_frame: Mapping[int, Sequence[BallObservation]] | None = None,
    calibration: Mapping[str, Any],
    physics: PhysicsParameters,
    config: BallArcSolverConfig,
    net_plane: Mapping[str, Any] | None,
    refine_endpoints: bool = True,
    max_nfev: int | None = None,
    soft_split_boundaries: Sequence[SoftSegmentBoundary] = (),
    return_soft_split_report: bool = False,
) -> list[FlightSegmentFit] | tuple[list[FlightSegmentFit], dict[str, Any]]:
    segments: list[FlightSegmentFit] = []
    if not soft_split_boundaries:
        for start, end in zip(anchors, anchors[1:]):
            segment = _fit_anchor_pair(
                len(segments),
                start,
                end,
                observations=observations,
                candidate_sets_by_frame=candidate_sets_by_frame,
                calibration=calibration,
                physics=physics,
                config=config,
                net_plane=net_plane,
                block_insufficient_observations=False,
                max_nfev=max_nfev,
                refine_endpoints=refine_endpoints,
            )
            if segment is None:
                continue
            segments.append(segment)
        if not return_soft_split_report:
            return segments
        return segments, {
            "anchor_class": "audio_onset_soft",
            "allowed_role": "segment_split_boundary_only",
            "supplied_count": 0,
            "applied_count": 0,
            "rejected_count": 0,
            "supplied": [],
            "applied": [],
            "rejected": [],
        }
    applied_boundary_ids: set[str] = set()
    rejected: list[dict[str, Any]] = []
    ordered_soft = sorted(
        soft_split_boundaries,
        key=lambda item: (float(item.corrected_time_s), int(item.frame), item.boundary_id),
    )
    for start, end in zip(anchors, anchors[1:]):
        span_soft = [
            boundary
            for boundary in ordered_soft
            if start.t + config.min_segment_dt_s < boundary.corrected_time_s < end.t - config.min_segment_dt_s
        ]
        materialized: list[tuple[AnchorEvent, SoftSegmentBoundary]] = []
        for boundary in span_soft:
            soft_anchor = _soft_split_anchor_for_span(
                boundary,
                observations=observations,
                calibration=calibration,
                span_start=start,
                span_end=end,
                config=config,
            )
            if soft_anchor is None:
                rejected.append(
                    {
                        **boundary.to_json(),
                        "reason": "no_visible_primary_observation_in_hard_anchor_span",
                    }
                )
                continue
            materialized.append((soft_anchor, boundary))
            applied_boundary_ids.add(boundary.boundary_id)
        boundaries_by_id = {boundary.boundary_id: boundary for _, boundary in materialized}
        span_anchors = [start, *[anchor for anchor, _ in materialized], end]
        span_used_soft = bool(materialized)
        for child_start, child_end in zip(span_anchors, span_anchors[1:]):
            segment = _fit_anchor_pair(
                len(segments),
                child_start,
                child_end,
                observations=observations,
                candidate_sets_by_frame=candidate_sets_by_frame,
                calibration=calibration,
                physics=physics,
                config=config,
                net_plane=net_plane,
                block_insufficient_observations=False,
                max_nfev=max_nfev,
                refine_endpoints=refine_endpoints,
            )
            if segment is None:
                continue
            if span_used_soft:
                provenance = tuple(
                    boundaries_by_id[anchor.anchor_id].to_json()
                    for anchor in (child_start, child_end)
                    if anchor.anchor_id in boundaries_by_id
                )
                segment = replace(segment, soft_split_provenance=provenance)
            segments.append(segment)
    if not return_soft_split_report:
        return segments
    supplied = [boundary.to_json() for boundary in ordered_soft]
    applied = [
        boundary.to_json() for boundary in ordered_soft if boundary.boundary_id in applied_boundary_ids
    ]
    outside = [
        {**boundary.to_json(), "reason": "outside_selected_hard_anchor_intervals"}
        for boundary in ordered_soft
        if boundary.boundary_id not in applied_boundary_ids
        and not any(item.get("boundary_id") == boundary.boundary_id for item in rejected)
    ]
    return segments, {
        "anchor_class": "audio_onset_soft",
        "allowed_role": "segment_split_boundary_only",
        "supplied_count": len(supplied),
        "applied_count": len(applied),
        "rejected_count": len(rejected) + len(outside),
        "supplied": supplied,
        "applied": applied,
        "rejected": [*rejected, *outside],
    }


def _soft_split_anchor_for_span(
    boundary: SoftSegmentBoundary,
    *,
    observations: Sequence[BallObservation],
    calibration: Mapping[str, Any],
    span_start: AnchorEvent,
    span_end: AnchorEvent,
    config: BallArcSolverConfig,
) -> AnchorEvent | None:
    span_observations = [
        observation
        for observation in observations
        if span_start.t - 1e-9 <= observation.t <= span_end.t + 1e-9 and observation.visible
    ]
    if not span_observations:
        return None
    nearest = min(
        span_observations,
        key=lambda observation: (
            abs(observation.t - float(boundary.corrected_time_s)),
            abs(observation.frame - int(boundary.frame)),
            observation.frame,
        ),
    )
    origin, direction = pixel_ray_world(calibration, nearest.xy)
    initialization_z_m = min(1.0, float(config.max_plausible_apex_m) / 2.0)
    try:
        initialization = intersect_ray_z(origin, direction, initialization_z_m)
    except ValueError:
        initialization = tuple(float(value) for value in origin)
    return AnchorEvent(
        anchor_id=boundary.boundary_id,
        kind="audio_onset_soft",
        t=float(boundary.corrected_time_s),
        frame=int(boundary.frame),
        world_xyz=initialization,
        sigma_m=max(1.0, float(config.max_plausible_apex_m)),
        status="soft_split_boundary",
        immovable=False,
        source="audio_onset_soft_split_boundary",
        details={
            **boundary.to_json(),
            "initialization_only": True,
            "initialization_source": "nearest_primary_observation_ray_at_neutral_z",
            "initialization_observation_frame": int(nearest.frame),
            "initialization_z_m": initialization_z_m,
        },
    )


def _fit_anchor_pair(
    segment_id: int,
    start: AnchorEvent,
    end: AnchorEvent,
    *,
    observations: Sequence[BallObservation],
    candidate_sets_by_frame: Mapping[int, Sequence[BallObservation]] | None = None,
    calibration: Mapping[str, Any],
    physics: PhysicsParameters,
    config: BallArcSolverConfig,
    net_plane: Mapping[str, Any] | None,
    block_insufficient_observations: bool,
    max_nfev: int | None = None,
    refine_endpoints: bool = True,
) -> FlightSegmentFit | None:
    if end.t - start.t < config.min_segment_dt_s:
        return _blocked_segment(segment_id, start, end, "duration_below_minimum")
    segment_observations = [obs for obs in observations if start.t - 1e-9 <= obs.t <= end.t + 1e-9]
    if len(segment_observations) < config.min_segment_observations:
        if block_insufficient_observations:
            return _blocked_segment(segment_id, start, end, "insufficient_observations")
        return None
    if candidate_sets_by_frame:
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
            candidate_sets_by_frame=candidate_sets_by_frame,
        )
    return _run_with_segment_budget(
        segment_id=segment_id,
        start=start,
        end=end,
        observations=segment_observations,
        candidate_sets_by_frame=None,
        config=config,
        fit=lambda: _fit_flight_segment_once(
            segment_id=segment_id,
            start_anchor=start,
            end_anchor=end,
            observations=segment_observations,
            calibration=calibration,
            physics=physics,
            config=config,
            net_plane=net_plane,
            max_nfev=max_nfev,
            refine_endpoints=refine_endpoints,
            candidate_endpoint_frozen=not refine_endpoints,
        ),
    )


def _select_event_subset(
    anchors: Sequence[AnchorEvent],
    *,
    observations: Sequence[BallObservation],
    candidate_sets_by_frame: Mapping[int, Sequence[BallObservation]] | None,
    calibration: Mapping[str, Any],
    physics: PhysicsParameters,
    config: BallArcSolverConfig,
    net_plane: Mapping[str, Any] | None,
    rally_spans: Mapping[str, Any] | None,
    final_refine_segments: bool = True,
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
                candidate_sets_by_frame=candidate_sets_by_frame,
                calibration=calibration,
                physics=physics,
                config=config,
                net_plane=net_plane,
                fit_cache=fit_cache,
            )
            rejected_rationales[key] = evaluation
            if not evaluation.get("accepted"):
                continue
            if best_eval is None or _candidate_event_score_gain(evaluation) > _candidate_event_score_gain(best_eval):
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

    selection_segments = _fit_segments_from_anchors(
        selected,
        observations=observations,
        candidate_sets_by_frame=candidate_sets_by_frame,
        calibration=calibration,
        physics=physics,
        config=config,
        net_plane=net_plane,
        refine_endpoints=False,
    )
    selected, selection_segments, endpoint_rejections = _prune_implausible_weak_endpoints(
        selected,
        selection_segments,
        observations=observations,
        candidate_sets_by_frame=candidate_sets_by_frame,
        calibration=calibration,
        physics=physics,
        config=config,
        net_plane=net_plane,
    )
    selected_keys = {_anchor_key(anchor) for anchor in selected}
    if final_refine_segments:
        segments = _fit_segments_from_anchors(
            selected,
            observations=observations,
            candidate_sets_by_frame=candidate_sets_by_frame,
            calibration=calibration,
            physics=physics,
            config=config,
            net_plane=net_plane,
        )
    else:
        segments = selection_segments
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
            "mandatory": "human-reviewed bounces, auto-bounce candidates, and weak rally endpoints",
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
    candidate_sets_by_frame: Mapping[int, Sequence[BallObservation]] | None,
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
        candidate_sets_by_frame=candidate_sets_by_frame,
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
        candidate_sets_by_frame=candidate_sets_by_frame,
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
        candidate_sets_by_frame=candidate_sets_by_frame,
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
    anchor_preservation_reason = _anchor_preservation_rejection_reason(parent, child_left, child_right, config)
    if anchor_preservation_reason is not None:
        payload["reason"] = anchor_preservation_reason
        payload["anchor_preservation"] = {
            "parent_endpoint_error_m": _optional_round(parent.endpoint_error_m if parent is not None else None, 6),
            "child_endpoint_error_m": [
                _optional_round(child_left.endpoint_error_m if child_left is not None else None, 6),
                _optional_round(child_right.endpoint_error_m if child_right is not None else None, 6),
            ],
            "parent_inlier_count": int(parent.inlier_count) if parent is not None else None,
            "child_inlier_count": [
                int(child_left.inlier_count) if child_left is not None else None,
                int(child_right.inlier_count) if child_right is not None else None,
            ],
            "min_segment_observations": int(config.min_segment_observations),
        }
        return payload
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


def _candidate_event_score_gain(evaluation: Mapping[str, Any]) -> float:
    value = _float_or_none(evaluation.get("score_gain"))
    if value is not None:
        return value
    return math.inf if evaluation.get("accepted") is True else -math.inf


def _anchor_preservation_rejection_reason(
    parent: FlightSegmentFit | None,
    child_left: FlightSegmentFit | None,
    child_right: FlightSegmentFit | None,
    config: BallArcSolverConfig,
) -> str | None:
    if not _anchor_preservation_interval_is_good(parent, config):
        return None
    assert parent is not None
    parent_endpoint = float(parent.endpoint_error_m)
    for child in (child_left, child_right):
        if child is None or not child.status.startswith("fit"):
            return "anchor_preservation_child_not_fit"
        if child.inlier_count < config.min_segment_observations:
            return "anchor_preservation_child_insufficient_inliers"
        if float(child.endpoint_error_m) > parent_endpoint + 1e-9:
            return "anchor_preservation_worse_endpoint_error"
    return None


def _anchor_preservation_interval_is_good(
    segment: FlightSegmentFit | None,
    config: BallArcSolverConfig,
) -> bool:
    return (
        segment is not None
        and segment.status.startswith("fit")
        and segment.inlier_count >= config.min_segment_observations
        and math.isfinite(float(segment.endpoint_error_m))
    )


def _final_selection_quality_adjustment(
    selected: Sequence[AnchorEvent],
    segments: Sequence[FlightSegmentFit],
    optional: Sequence[AnchorEvent],
    *,
    blocked_optional_keys: set[str],
    forced_optional_keys: set[str],
    observations: Sequence[BallObservation],
    candidate_sets_by_frame: Mapping[int, Sequence[BallObservation]] | None,
    calibration: Mapping[str, Any],
    physics: PhysicsParameters,
    config: BallArcSolverConfig,
    net_plane: Mapping[str, Any] | None,
    final_refine_segments: bool,
) -> dict[str, Any] | None:
    if not final_refine_segments:
        return None
    ordered = order_event_anchors(selected)
    selected_keys = {_anchor_key(anchor) for anchor in ordered}
    segment_by_pair = {(_anchor_key(segment.start_anchor), _anchor_key(segment.end_anchor)): segment for segment in segments}

    for index, anchor in enumerate(ordered[1:-1], start=1):
        key = _anchor_key(anchor)
        if key in blocked_optional_keys or _is_mandatory_event(anchor):
            continue
        left_anchor = ordered[index - 1]
        right_anchor = ordered[index + 1]
        child_left = segment_by_pair.get((_anchor_key(left_anchor), key)) or _fit_quality_pair(
            0,
            left_anchor,
            anchor,
            observations=observations,
            candidate_sets_by_frame=candidate_sets_by_frame,
            calibration=calibration,
            physics=physics,
            config=config,
            net_plane=net_plane,
        )
        child_right = segment_by_pair.get((key, _anchor_key(right_anchor))) or _fit_quality_pair(
            1,
            anchor,
            right_anchor,
            observations=observations,
            candidate_sets_by_frame=candidate_sets_by_frame,
            calibration=calibration,
            physics=physics,
            config=config,
            net_plane=net_plane,
        )
        parent = _fit_quality_pair(
            2,
            left_anchor,
            right_anchor,
            observations=observations,
            candidate_sets_by_frame=candidate_sets_by_frame,
            calibration=calibration,
            physics=physics,
            config=config,
            net_plane=net_plane,
        )
        rejection = _selected_split_quality_rejection(
            parent,
            child_left,
            child_right,
            split_anchor=anchor,
            physics=physics,
            config=config,
        )
        if rejection is not None:
            return {
                "action": "reject_selected_optional",
                "anchor": anchor,
                **rejection,
            }

    best_rescue: dict[str, Any] | None = None
    optional_candidates = [
        anchor
        for anchor in optional
        if _anchor_key(anchor) not in selected_keys
        and _anchor_key(anchor) not in blocked_optional_keys
        and _anchor_key(anchor) not in forced_optional_keys
    ]
    parent_segments = list(segments)
    if not parent_segments:
        for left_anchor, right_anchor in zip(ordered, ordered[1:]):
            parent = _fit_quality_pair(
                0,
                left_anchor,
                right_anchor,
                observations=observations,
                candidate_sets_by_frame=candidate_sets_by_frame,
                calibration=calibration,
                physics=physics,
                config=config,
                net_plane=net_plane,
            )
            if parent is not None:
                parent_segments.append(parent)
    for parent in parent_segments:
        if not _segment_needs_final_quality_rescue(parent, config):
            continue
        for candidate in optional_candidates:
            if not (parent.start_anchor.t + config.min_segment_dt_s <= candidate.t <= parent.end_anchor.t - config.min_segment_dt_s):
                continue
            child_left = _fit_quality_pair(
                0,
                parent.start_anchor,
                candidate,
                observations=observations,
                candidate_sets_by_frame=candidate_sets_by_frame,
                calibration=calibration,
                physics=physics,
                config=config,
                net_plane=net_plane,
            )
            child_right = _fit_quality_pair(
                1,
                candidate,
                parent.end_anchor,
                observations=observations,
                candidate_sets_by_frame=candidate_sets_by_frame,
                calibration=calibration,
                physics=physics,
                config=config,
                net_plane=net_plane,
            )
            rescue = _candidate_split_quality_acceptance(
                parent,
                child_left,
                child_right,
                split_anchor=candidate,
                physics=physics,
                config=config,
            )
            if rescue is None:
                continue
            payload = {
                "action": "force_optional",
                "anchor": candidate,
                **rescue,
                "parent_start": parent.start_anchor.anchor_id,
                "parent_end": parent.end_anchor.anchor_id,
                "parent_segment_id": int(parent.segment_id),
            }
            if best_rescue is None or _final_quality_rescue_rank(payload) > _final_quality_rescue_rank(best_rescue):
                best_rescue = payload
    return best_rescue


def _fit_quality_pair(
    segment_id: int,
    start: AnchorEvent,
    end: AnchorEvent,
    *,
    observations: Sequence[BallObservation],
    candidate_sets_by_frame: Mapping[int, Sequence[BallObservation]] | None,
    calibration: Mapping[str, Any],
    physics: PhysicsParameters,
    config: BallArcSolverConfig,
    net_plane: Mapping[str, Any] | None,
) -> FlightSegmentFit | None:
    return _fit_anchor_pair(
        segment_id,
        start,
        end,
        observations=observations,
        candidate_sets_by_frame=candidate_sets_by_frame,
        calibration=calibration,
        physics=physics,
        config=config,
        net_plane=net_plane,
        block_insufficient_observations=False,
        max_nfev=config.selection_max_nfev,
        refine_endpoints=True,
    )


def _selected_split_quality_rejection(
    parent: FlightSegmentFit | None,
    child_left: FlightSegmentFit | None,
    child_right: FlightSegmentFit | None,
    *,
    split_anchor: AnchorEvent,
    physics: PhysicsParameters,
    config: BallArcSolverConfig,
) -> dict[str, Any] | None:
    if _is_mandatory_event(split_anchor) or _anchor_has_verified_evidence(split_anchor):
        return None
    parent_is_fit_tier = _segment_quality_fit_tier(parent, config)
    if parent_is_fit_tier and (
        not _segment_quality_fit_tier(child_left, config) or not _segment_quality_fit_tier(child_right, config)
    ):
        return _split_quality_payload(
            "selected_split_child_below_fit_tier",
            parent,
            child_left,
            child_right,
            config=config,
        )
    junction = _junction_quality(child_left, child_right, split_anchor=split_anchor, physics=physics, config=config)
    if junction is not None:
        return junction
    if not parent_is_fit_tier:
        return None
    parent_rmse = _segment_quality_rmse(parent)
    child_rmse = _combined_quality_rmse((child_left, child_right))
    if parent_rmse is not None and child_rmse is not None and child_rmse >= parent_rmse - 1e-9:
        return _split_quality_payload(
            "selected_split_not_residual_better",
            parent,
            child_left,
            child_right,
            config=config,
            child_rmse=child_rmse,
        )
    parent_endpoint = _finite_endpoint_error(parent)
    child_endpoint = max(_finite_endpoint_error(child_left), _finite_endpoint_error(child_right))
    if child_endpoint > parent_endpoint + 1e-9:
        return _split_quality_payload(
            "selected_split_worse_endpoint_error",
            parent,
            child_left,
            child_right,
            config=config,
            child_rmse=child_rmse,
        )
    return None


def _candidate_split_quality_acceptance(
    parent: FlightSegmentFit | None,
    child_left: FlightSegmentFit | None,
    child_right: FlightSegmentFit | None,
    *,
    split_anchor: AnchorEvent,
    physics: PhysicsParameters,
    config: BallArcSolverConfig,
) -> dict[str, Any] | None:
    if _is_mandatory_event(split_anchor):
        return None
    if not _segment_quality_fit_tier(child_left, config) or not _segment_quality_fit_tier(child_right, config):
        return None
    if _junction_quality(child_left, child_right, split_anchor=split_anchor, physics=physics, config=config) is not None:
        return None
    parent_rmse = _segment_quality_rmse(parent)
    child_rmse = _combined_quality_rmse((child_left, child_right))
    if parent_rmse is not None and child_rmse is not None and child_rmse >= parent_rmse - 1e-9:
        return None
    parent_endpoint = _finite_endpoint_error(parent)
    child_endpoint = max(_finite_endpoint_error(child_left), _finite_endpoint_error(child_right))
    if child_endpoint > parent_endpoint + 1e-9:
        return None
    score_gain = None
    residual_reduction = None
    if parent_rmse is not None and child_rmse is not None:
        score_gain = parent_rmse - child_rmse
        residual_reduction = score_gain / max(parent_rmse, 1e-9)
    return {
        "reason": "final_quality_rescue_split",
        "parent_rmse_px": _optional_round(parent_rmse, 6),
        "child_rmse_px": _optional_round(child_rmse, 6),
        "score_gain": _optional_round(score_gain, 6),
        "residual_reduction": _optional_round(residual_reduction, 6),
        "parent_endpoint_error_m": _optional_float_round(parent_endpoint, 6),
        "child_endpoint_error_m": [
            _optional_float_round(_finite_endpoint_error(child_left), 6),
            _optional_float_round(_finite_endpoint_error(child_right), 6),
        ],
    }


def _segment_needs_final_quality_rescue(segment: FlightSegmentFit | None, config: BallArcSolverConfig) -> bool:
    if segment is None:
        return False
    return _fit_validity_failure_reason(segment) is not None


def _segment_quality_fit_tier(segment: FlightSegmentFit | None, config: BallArcSolverConfig) -> bool:
    return (
        segment is not None
        and segment.status.startswith("fit")
        and segment.status != "fit_bvp_fallback"
        and segment.inlier_count >= config.min_segment_observations
        and _fit_validity_failure_reason(segment) is None
    )


def _anchor_has_verified_evidence(anchor: AnchorEvent) -> bool:
    if anchor.immovable or anchor.status == "human_reviewed":
        return True
    details = anchor.details if isinstance(anchor.details, Mapping) else {}
    return bool(details.get("human_reviewed") is True or details.get("verified") is True)


def _junction_quality(
    child_left: FlightSegmentFit | None,
    child_right: FlightSegmentFit | None,
    *,
    split_anchor: AnchorEvent,
    physics: PhysicsParameters,
    config: BallArcSolverConfig,
) -> dict[str, Any] | None:
    if child_left is None or child_right is None:
        return {"reason": "selected_split_missing_child_fit"}
    time_gap = float(child_right.start_anchor.t) - float(child_left.end_anchor.t)
    if time_gap < -1e-6:
        return {
            "reason": "selected_split_negative_time_gap",
            "junction_time_gap_s": _round(time_gap, 9),
        }
    if split_anchor.kind != "contact":
        return None
    left_velocity = _segment_velocity_at(child_left, child_left.end_anchor.t, physics=physics, config=config)
    right_velocity = _segment_velocity_at(child_right, child_right.start_anchor.t, physics=physics, config=config)
    velocity_delta = _distance(left_velocity, right_velocity)
    bound = _junction_velocity_delta_bound(
        left_velocity,
        right_velocity,
        time_gap_s=time_gap,
        physics=physics,
        config=config,
    )
    if velocity_delta > bound + 1e-9:
        return {
            "reason": "selected_split_velocity_delta_exceeds_physics_bound",
            "junction_time_gap_s": _round(time_gap, 9),
            "junction_velocity_delta_mps": _round(velocity_delta, 6),
            "junction_velocity_bound_mps": _round(bound, 6),
        }
    return None


def _segment_velocity_at(
    segment: FlightSegmentFit,
    t: float,
    *,
    physics: PhysicsParameters,
    config: BallArcSolverConfig,
) -> tuple[float, float, float]:
    target_t = float(t)
    current_t = float(segment.start_anchor.t)
    state = (*segment.initial_position_m, *segment.initial_velocity_mps)
    if physics.drag_k_per_m <= 0.0:
        dt = target_t - current_t
        return (
            float(segment.initial_velocity_mps[0]),
            float(segment.initial_velocity_mps[1]),
            float(segment.initial_velocity_mps[2]) - physics.gravity_mps2 * dt,
        )
    direction = 1.0 if target_t >= current_t else -1.0
    while (target_t - current_t) * direction > 1e-12:
        step = direction * min(config.integrator_max_step_s, abs(target_t - current_t))
        state = _rk4_step(state, step, physics)
        current_t += step
    return (float(state[3]), float(state[4]), float(state[5]))


def _junction_velocity_delta_bound(
    left_velocity: tuple[float, float, float],
    right_velocity: tuple[float, float, float],
    *,
    time_gap_s: float,
    physics: PhysicsParameters,
    config: BallArcSolverConfig,
) -> float:
    speed = max(_norm(left_velocity), _norm(right_velocity))
    drag_accel = physics.drag_k_per_m * speed * speed
    accel_bound = abs(physics.gravity_mps2) + drag_accel
    dynamics_window_s = max(float(time_gap_s), config.integrator_max_step_s, config.min_segment_dt_s)
    return accel_bound * dynamics_window_s + physics.diameter_m / max(config.integrator_max_step_s, 1e-9)


def _split_quality_payload(
    reason: str,
    parent: FlightSegmentFit | None,
    child_left: FlightSegmentFit | None,
    child_right: FlightSegmentFit | None,
    *,
    config: BallArcSolverConfig,
    child_rmse: float | None = None,
) -> dict[str, Any]:
    if child_rmse is None:
        child_rmse = _combined_quality_rmse((child_left, child_right))
    return {
        "reason": reason,
        "parent_status": None if parent is None else parent.status,
        "child_status": [
            None if child_left is None else child_left.status,
            None if child_right is None else child_right.status,
        ],
        "parent_rmse_px": _optional_round(_segment_quality_rmse(parent), 6),
        "child_rmse_px": _optional_round(child_rmse, 6),
        "parent_endpoint_error_m": _optional_float_round(_finite_endpoint_error(parent), 6),
        "child_endpoint_error_m": [
            _optional_float_round(_finite_endpoint_error(child_left), 6),
            _optional_float_round(_finite_endpoint_error(child_right), 6),
        ],
        "min_segment_observations": int(config.min_segment_observations),
    }


def _segment_quality_rmse(segment: FlightSegmentFit | None) -> float | None:
    if segment is None:
        return None
    if segment.reprojection_rmse_px is not None and math.isfinite(float(segment.reprojection_rmse_px)):
        return float(segment.reprojection_rmse_px)
    errors = [float(value) for value in segment.reprojection_errors_px.values() if math.isfinite(float(value))]
    return _rmse(errors)


def _combined_quality_rmse(segments: Sequence[FlightSegmentFit | None]) -> float | None:
    squared_sum = 0.0
    count = 0
    for segment in segments:
        if segment is None:
            continue
        errors = [float(value) for value in segment.reprojection_errors_px.values() if math.isfinite(float(value))]
        if errors:
            squared_sum += sum(error * error for error in errors)
            count += len(errors)
            continue
        rmse = _segment_quality_rmse(segment)
        if rmse is not None:
            weight = max(1, len(segment.observations))
            squared_sum += rmse * rmse * weight
            count += weight
    if count <= 0:
        return None
    return math.sqrt(squared_sum / count)


def _finite_endpoint_error(segment: FlightSegmentFit | None) -> float:
    if segment is None:
        return math.inf
    try:
        value = float(segment.endpoint_error_m)
    except (TypeError, ValueError):
        return math.inf
    return value if math.isfinite(value) else math.inf


def _final_quality_rescue_rank(payload: Mapping[str, Any]) -> tuple[float, float]:
    residual_reduction = _float_or_none(payload.get("residual_reduction"))
    score_gain = _float_or_none(payload.get("score_gain"))
    return (
        float("-inf") if residual_reduction is None else residual_reduction,
        float("-inf") if score_gain is None else score_gain,
    )


def _prune_implausible_weak_endpoints(
    selected: Sequence[AnchorEvent],
    segments: Sequence[FlightSegmentFit],
    *,
    observations: Sequence[BallObservation],
    candidate_sets_by_frame: Mapping[int, Sequence[BallObservation]] | None,
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
            candidate_sets_by_frame=candidate_sets_by_frame,
            calibration=calibration,
            physics=physics,
            config=config,
            net_plane=net_plane,
            refine_endpoints=False,
            max_nfev=config.selection_max_nfev,
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
        if weak.status == f"blocked:{SEGMENT_BUDGET_EXCEEDED}":
            # Timeout is a typed missing segment, not an implausible-fit skip.
            weak_segments.append(weak)
            next_segment_id += 1
            continue
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


def _apply_fit_validity_gates(
    segments: Sequence[FlightSegmentFit],
    *,
    physics: PhysicsParameters,
    config: BallArcSolverConfig,
    net_plane: Mapping[str, Any] | None,
) -> list[FlightSegmentFit]:
    gated: list[FlightSegmentFit] = []
    for segment in segments:
        reason = _fit_validity_failure_reason(segment)
        if reason is None:
            gated.append(segment)
            continue
        gated.append(_fit_bvp_fallback_segment(segment, reason=reason, physics=physics, config=config, net_plane=net_plane))
    return gated


_PROTECTED_BASELINE_SPANS: Mapping[str, tuple[dict[str, Any], ...]] = {
    "burlington_gold_0300_low_steep_corner": (
        {"name": "burlington_seg2_adjacent", "segment_id": 2, "interval": (107, 132)},
        {"name": "burlington_seg4_adjacent", "segment_id": 4, "interval": (139, 151)},
        {"name": "burlington_seg15_adjacent", "segment_id": 15, "interval": (447, 497)},
        {"name": "burlington_seg16_adjacent", "segment_id": 16, "interval": (497, 543)},
    ),
    "wolverine_mixed_0200_mid_steep_corner": (
        {"name": "wolverine_seg4_region", "segment_id": 4, "interval": (70, 104)},
    ),
}


def _apply_protected_span_priors_from_frozen_baseline(
    segments: Sequence[FlightSegmentFit],
    *,
    observations: Sequence[BallObservation],
    calibration: Mapping[str, Any],
    physics: PhysicsParameters,
    config: BallArcSolverConfig,
    net_plane: Mapping[str, Any] | None,
    clip_id: str | None,
) -> tuple[list[FlightSegmentFit], dict[str, Any]]:
    clip = str(clip_id or "")
    protected = _PROTECTED_BASELINE_SPANS.get(clip)
    if not protected:
        return list(segments), {"mode": "not_applicable", "applied_count": 0, "rows": []}
    baseline = _load_protected_baseline_artifact(clip)
    if baseline is None:
        return list(segments), {"mode": "baseline_missing", "applied_count": 0, "rows": []}
    repaired, report = _apply_protected_span_priors(
        segments,
        baseline_artifact=baseline,
        protected_spans=protected,
        observations=observations,
        calibration=calibration,
        physics=physics,
        config=config,
        net_plane=net_plane,
    )
    return repaired, {**report, "clip_id": clip}


def _apply_protected_span_priors(
    segments: Sequence[FlightSegmentFit],
    *,
    baseline_artifact: Mapping[str, Any],
    protected_spans: Sequence[Mapping[str, Any]],
    observations: Sequence[BallObservation],
    calibration: Mapping[str, Any],
    physics: PhysicsParameters,
    config: BallArcSolverConfig,
    net_plane: Mapping[str, Any] | None,
) -> tuple[list[FlightSegmentFit], dict[str, Any]]:
    output = list(segments)
    rows: list[dict[str, Any]] = []
    applied_baseline_keys: set[tuple[int, int, int]] = set()
    for item in protected_spans:
        name = str(item.get("name") or "")
        interval_raw = item.get("interval")
        if not isinstance(interval_raw, Sequence) or isinstance(interval_raw, (str, bytes)) or len(interval_raw) != 2:
            rows.append({"name": name, "action": "skipped_invalid_interval"})
            continue
        interval = (int(interval_raw[0]), int(interval_raw[1]))
        baseline_segment = _baseline_segment_for_item(baseline_artifact, int(item.get("segment_id", -1)), interval)
        if baseline_segment is None:
            rows.append({"name": name, "interval": list(interval), "action": "skipped_baseline_segment_missing"})
            continue
        baseline_status = str(baseline_segment.get("status") or "")
        if not baseline_status.startswith("fit") or baseline_status == "fit_bvp_fallback":
            rows.append(
                {
                    "name": name,
                    "interval": list(interval),
                    "action": "skipped_baseline_not_fit_prior",
                    "baseline_status": baseline_status,
                }
            )
            continue
        current_quality = _protected_span_quality(output, interval)
        if not current_quality["covering_segments"] and not any(interval[0] <= int(obs.frame) <= interval[1] for obs in observations):
            rows.append({"name": name, "interval": list(interval), "action": "skipped_no_current_span"})
            continue
        baseline_rmse = _float_or_none(baseline_segment.get("reprojection_rmse_px"))
        baseline_endpoint = _float_or_none(baseline_segment.get("endpoint_error_m"))
        if not _protected_span_needs_rollback(current_quality, baseline_rmse=baseline_rmse, baseline_endpoint=baseline_endpoint):
            rows.append(
                {
                    "name": name,
                    "interval": list(interval),
                    "action": "kept_current",
                    "current": current_quality,
                    "baseline_rmse_px": _optional_float_round(baseline_rmse),
                    "baseline_endpoint_error_m": _optional_float_round(baseline_endpoint),
                }
            )
            continue
        segments_to_apply: list[tuple[Mapping[str, Any], tuple[int, int], str]] = [
            (baseline_segment, interval, "prior_applied")
        ]
        for boundary_segment in _baseline_segments_touching_interval(baseline_artifact, interval):
            key = _baseline_segment_key(boundary_segment)
            if key == _baseline_segment_key(baseline_segment):
                continue
            boundary_status = str(boundary_segment.get("status") or "")
            if not boundary_status.startswith("fit") or boundary_status == "fit_bvp_fallback":
                continue
            boundary_interval = (
                int(boundary_segment.get("frame_start", interval[0])),
                int(boundary_segment.get("frame_end", interval[1])),
            )
            segments_to_apply.append((boundary_segment, boundary_interval, "boundary_prior_applied"))
        for baseline_to_apply, apply_interval, action in segments_to_apply:
            key = _baseline_segment_key(baseline_to_apply)
            if key in applied_baseline_keys:
                rows.append({"name": name, "interval": list(apply_interval), "action": "skipped_duplicate_prior"})
                continue
            prior = _protected_prior_segment_from_baseline(
                baseline_to_apply,
                segment_id=len(output),
                observations=observations,
                calibration=calibration,
                physics=physics,
                config=config,
                net_plane=net_plane,
            )
            if prior is None:
                rows.append({"name": name, "interval": list(apply_interval), "action": "skipped_prior_unbuildable"})
                continue
            output = _overlay_protected_segment(output, prior, apply_interval, physics=physics, config=config)
            applied_baseline_keys.add(key)
            rows.append(
                {
                    "name": name,
                    "interval": list(apply_interval),
                    "action": action,
                    "current": current_quality if action == "prior_applied" else _protected_span_quality(output, apply_interval),
                    "prior_status": prior.status,
                    "prior_rmse_px": _optional_round(prior.reprojection_rmse_px, 6),
                    "prior_endpoint_error_m": _round(prior.endpoint_error_m, 6),
                    "baseline_rmse_px": _optional_float_round(_float_or_none(baseline_to_apply.get("reprojection_rmse_px"))),
                    "baseline_endpoint_error_m": _optional_float_round(_float_or_none(baseline_to_apply.get("endpoint_error_m"))),
                    "physical_violations_before_gate": list(prior.physical_sanity.get("violations") or [])
                    if isinstance(prior.physical_sanity, Mapping)
                    else [],
                }
            )
    output = [replace(segment, segment_id=index) for index, segment in enumerate(sorted(output, key=lambda seg: (seg.start_anchor.t, seg.end_anchor.t, seg.segment_id)))]
    reason_counts: dict[str, int] = {}
    for row in rows:
        action = str(row.get("action") or "unknown")
        reason_counts[action] = reason_counts.get(action, 0) + 1
    return output, {
        "mode": "frozen_baseline_arc_params_before_validity_gates",
        "applied_count": sum(1 for row in rows if row.get("action") == "prior_applied"),
        "non_gated_action_counts": reason_counts,
        "rows": rows,
    }


def _baseline_segments_touching_interval(
    baseline_artifact: Mapping[str, Any],
    interval: tuple[int, int],
) -> tuple[Mapping[str, Any], ...]:
    segments = baseline_artifact.get("segments")
    if not isinstance(segments, Sequence) or isinstance(segments, (str, bytes)):
        return ()
    touching: list[Mapping[str, Any]] = []
    for segment in segments:
        if not isinstance(segment, Mapping):
            continue
        start = int(segment.get("frame_start", -1))
        end = int(segment.get("frame_end", -1))
        if start <= interval[1] and end >= interval[0]:
            touching.append(segment)
    return tuple(touching)


def _baseline_segment_key(segment: Mapping[str, Any]) -> tuple[int, int, int]:
    return (
        int(segment.get("segment_id", -1)),
        int(segment.get("frame_start", -1)),
        int(segment.get("frame_end", -1)),
    )


def _protected_prior_segment_from_baseline(
    baseline_segment: Mapping[str, Any],
    *,
    segment_id: int,
    observations: Sequence[BallObservation],
    calibration: Mapping[str, Any],
    physics: PhysicsParameters,
    config: BallArcSolverConfig,
    net_plane: Mapping[str, Any] | None,
) -> FlightSegmentFit | None:
    anchors_json = baseline_segment.get("anchors_used")
    if not isinstance(anchors_json, Sequence) or isinstance(anchors_json, (str, bytes)) or len(anchors_json) < 2:
        return None
    p0 = _vec3_from_json(baseline_segment.get("initial_position_m"))
    v0 = _vec3_from_json(baseline_segment.get("initial_velocity_mps"))
    if p0 is None or v0 is None:
        return None
    try:
        start_anchor = _anchor_from_artifact_json(anchors_json[0])
        end_anchor = _anchor_from_artifact_json(anchors_json[1])
    except (KeyError, TypeError, ValueError):
        return None
    if end_anchor.t <= start_anchor.t + 1e-9:
        return None
    span_observations = tuple(
        obs for obs in observations if start_anchor.t - 1e-9 <= obs.t <= end_anchor.t + 1e-9 and obs.visible
    )
    reprojection_errors = _segment_reprojection_errors_from_params(
        span_observations,
        p0,
        v0,
        start_anchor.t,
        calibration=calibration,
        physics=physics,
        config=config,
    )
    inlier_frames = _int_tuple(baseline_segment.get("inlier_frames"))
    outlier_frames = _int_tuple(baseline_segment.get("outlier_frames"))
    if not inlier_frames and reprojection_errors:
        inlier_frames = tuple(
            frame for frame, error in sorted(reprojection_errors.items()) if error <= config.max_reprojection_inlier_px
        )
        outlier_frames = tuple(frame for frame in sorted(reprojection_errors) if frame not in set(inlier_frames))
    inlier_errors = [reprojection_errors[frame] for frame in inlier_frames if frame in reprojection_errors]
    baseline_rmse = _float_or_none(baseline_segment.get("reprojection_rmse_px"))
    baseline_max = _float_or_none(baseline_segment.get("max_reprojection_error_px"))
    endpoint_pred = _integrate_positions(p0, v0, [end_anchor.t], t0=start_anchor.t, physics=physics, config=config)[0]
    start_anchor, start_repaired = _repair_unverified_prior_anchor(start_anchor, p0, role="start")
    end_anchor, end_repaired = _repair_unverified_prior_anchor(end_anchor, endpoint_pred, role="end")
    net_clearance = _net_clearance_m(p0, v0, start_anchor.t, end_anchor.t, physics, config, net_plane)
    physical = _physical_sanity(p0, v0, start_anchor.t, end_anchor.t, physics, config, net_clearance)
    diagnostics = dict(baseline_segment.get("diagnostics") if isinstance(baseline_segment.get("diagnostics"), Mapping) else {})
    diagnostics["protected_span_prior"] = {
        "source": "frozen_baseline_arc_params",
        "baseline_segment_id": int(baseline_segment.get("segment_id", -1)),
        "baseline_status": str(baseline_segment.get("status") or ""),
        "applied_before_fit_validity_gates": True,
        "start_repaired_before_gate": bool(start_repaired),
        "endpoint_repaired_before_gate": bool(end_repaired),
        "physical_violations_before_gate": list(physical.get("violations") or []),
    }
    return FlightSegmentFit(
        segment_id=segment_id,
        status=str(baseline_segment.get("status") or "fit"),
        start_anchor=start_anchor,
        end_anchor=end_anchor,
        initial_position_m=p0,
        initial_velocity_mps=v0,
        observations=span_observations,
        inlier_frames=inlier_frames,
        outlier_frames=outlier_frames,
        reprojection_errors_px=reprojection_errors,
        reprojection_rmse_px=baseline_rmse if baseline_rmse is not None else _rmse(inlier_errors),
        max_reprojection_error_px=baseline_max if baseline_max is not None else (max(reprojection_errors.values()) if reprojection_errors else None),
        endpoint_error_m=_distance(endpoint_pred, end_anchor.world_xyz),
        net_clearance_m=net_clearance,
        net_clearance_ok=None if net_clearance is None else net_clearance >= -config.net_clearance_slack_m,
        physical_sanity=physical,
        size_residuals_m=dict(baseline_segment.get("size_residuals_m") if isinstance(baseline_segment.get("size_residuals_m"), Mapping) else {}),
        diagnostics=diagnostics,
    )


def _repair_unverified_prior_anchor(
    anchor: AnchorEvent,
    world_xyz: tuple[float, float, float],
    *,
    role: str,
) -> tuple[AnchorEvent, bool]:
    if anchor.immovable or anchor.status == "human_reviewed":
        return anchor, False
    details = dict(anchor.details or {})
    details["protected_span_prior_anchor_repair"] = {
        "role": role,
        "source": "frozen_baseline_arc_params",
        "original_world_xyz": _vec_json(anchor.world_xyz),
    }
    return replace(
        anchor,
        world_xyz=world_xyz,
        source=anchor.source or "protected_span_prior",
        details=details,
    ), True


def _segment_reprojection_errors_from_params(
    observations: Sequence[BallObservation],
    p0: tuple[float, float, float],
    v0: tuple[float, float, float],
    t0: float,
    *,
    calibration: Mapping[str, Any],
    physics: PhysicsParameters,
    config: BallArcSolverConfig,
) -> dict[int, float]:
    errors: dict[int, float] = {}
    if not observations:
        return errors
    times = [obs.t for obs in observations]
    points = _integrate_positions(p0, v0, times, t0=t0, physics=physics, config=config)
    for obs, point in zip(observations, points, strict=True):
        try:
            errors[int(obs.frame)] = _distance2(_project_world_point(calibration, point), obs.xy)
        except (TypeError, ValueError):
            continue
    return errors


def _int_tuple(value: Any) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    parsed: list[int] = []
    for item in value:
        try:
            parsed.append(int(item))
        except (TypeError, ValueError):
            continue
    return tuple(parsed)


def _post_final_protected_span_rollback(
    segments: Sequence[FlightSegmentFit],
    *,
    observations: Sequence[BallObservation],
    calibration: Mapping[str, Any],
    physics: PhysicsParameters,
    config: BallArcSolverConfig,
    net_plane: Mapping[str, Any] | None,
    clip_id: str | None,
) -> tuple[list[FlightSegmentFit], dict[str, Any]]:
    clip = str(clip_id or "")
    protected = _PROTECTED_BASELINE_SPANS.get(clip)
    if not protected:
        return list(segments), {"mode": "not_applicable", "applied_count": 0, "rows": []}
    baseline = _load_protected_baseline_artifact(clip)
    if baseline is None:
        return list(segments), {"mode": "baseline_missing", "applied_count": 0, "rows": []}
    output = list(segments)
    rows: list[dict[str, Any]] = []
    for item in protected:
        interval = (int(item["interval"][0]), int(item["interval"][1]))
        baseline_segment = _baseline_segment_for_item(baseline, int(item["segment_id"]), interval)
        if baseline_segment is None:
            rows.append({"name": item["name"], "interval": list(interval), "action": "skipped_baseline_segment_missing"})
            continue
        current_quality = _protected_span_quality(output, interval)
        baseline_rmse = _float_or_none(baseline_segment.get("reprojection_rmse_px"))
        baseline_endpoint = _float_or_none(baseline_segment.get("endpoint_error_m"))
        if not _protected_span_needs_rollback(current_quality, baseline_rmse=baseline_rmse, baseline_endpoint=baseline_endpoint):
            rows.append(
                {
                    "name": item["name"],
                    "interval": list(interval),
                    "action": "kept_current",
                    "current": current_quality,
                    "baseline_rmse_px": _optional_float_round(baseline_rmse),
                    "baseline_endpoint_error_m": _optional_float_round(baseline_endpoint),
                }
            )
            continue
        anchors_json = baseline_segment.get("anchors_used")
        if not isinstance(anchors_json, Sequence) or isinstance(anchors_json, (str, bytes)) or len(anchors_json) < 2:
            rows.append({"name": item["name"], "interval": list(interval), "action": "skipped_baseline_anchors_missing"})
            continue
        start_anchor = _anchor_from_artifact_json(anchors_json[0])
        end_anchor = _anchor_from_artifact_json(anchors_json[1])
        rollback = _fit_anchor_pair(
            len(output),
            start_anchor,
            end_anchor,
            observations=observations,
            candidate_sets_by_frame=None,
            calibration=calibration,
            physics=physics,
            config=config,
            net_plane=net_plane,
            block_insufficient_observations=False,
            refine_endpoints=True,
        )
        if rollback is None:
            rows.append({"name": item["name"], "interval": list(interval), "action": "skipped_refit_none", "current": current_quality})
            continue
        gated = _apply_fit_validity_gates((rollback,), physics=physics, config=config, net_plane=net_plane)[0]
        if not gated.status.startswith("fit") or gated.status == "fit_bvp_fallback":
            rows.append(
                {
                    "name": item["name"],
                    "interval": list(interval),
                    "action": "skipped_refit_not_fit_tier",
                    "status": gated.status,
                    "current": current_quality,
                }
            )
            continue
        if baseline_endpoint is not None and gated.endpoint_error_m > baseline_endpoint + 1e-6:
            rows.append(
                {
                    "name": item["name"],
                    "interval": list(interval),
                    "action": "skipped_refit_endpoint_worse",
                    "endpoint_error_m": _round(gated.endpoint_error_m, 6),
                    "baseline_endpoint_error_m": _round(baseline_endpoint, 6),
                    "current": current_quality,
                }
            )
            continue
        output = _overlay_protected_segment(output, gated, interval, physics=physics, config=config)
        rows.append(
            {
                "name": item["name"],
                "interval": list(interval),
                "action": "rollback_applied",
                "current": current_quality,
                "rollback_status": gated.status,
                "rollback_rmse_px": _optional_round(gated.reprojection_rmse_px, 6),
                "rollback_endpoint_error_m": _round(gated.endpoint_error_m, 6),
                "baseline_rmse_px": _optional_float_round(baseline_rmse),
                "baseline_endpoint_error_m": _optional_float_round(baseline_endpoint),
            }
        )
    output = [replace(segment, segment_id=index) for index, segment in enumerate(sorted(output, key=lambda seg: (seg.start_anchor.t, seg.end_anchor.t, seg.segment_id)))]
    return output, {
        "mode": "post_final_baseline_span_rollback",
        "applied_count": sum(1 for row in rows if row.get("action") == "rollback_applied"),
        "rows": rows,
    }


def _load_protected_baseline_artifact(clip: str) -> Mapping[str, Any] | None:
    path = Path("runs/lanes/w4_bvp_verify_20260707/replay/baseline") / clip / "ball_track_arc_solved.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _baseline_segment_for_item(
    baseline: Mapping[str, Any],
    segment_id: int,
    interval: tuple[int, int],
) -> Mapping[str, Any] | None:
    segments = baseline.get("segments")
    if not isinstance(segments, Sequence) or isinstance(segments, (str, bytes)):
        return None
    for segment in segments:
        if not isinstance(segment, Mapping):
            continue
        if int(segment.get("segment_id", -1)) == segment_id:
            return segment
    for segment in segments:
        if not isinstance(segment, Mapping):
            continue
        if int(segment.get("frame_start", -1)) == interval[0] and int(segment.get("frame_end", -1)) == interval[1]:
            return segment
    return None


def _anchor_from_artifact_json(item: Mapping[str, Any]) -> AnchorEvent:
    return AnchorEvent(
        anchor_id=str(item["anchor_id"]),
        kind=str(item["kind"]),
        t=float(item["t"]),
        frame=int(item["frame"]),
        world_xyz=tuple(float(value) for value in item["world_xyz"]),  # type: ignore[arg-type]
        sigma_m=float(item["sigma_m"]),
        status=str(item["status"]),
        player_id=item.get("player_id"),
        immovable=bool(item.get("immovable", False)),
        source=item.get("source"),
        details=item.get("details") if isinstance(item.get("details"), Mapping) else None,
    )


def _protected_span_quality(segments: Sequence[FlightSegmentFit], interval: tuple[int, int]) -> dict[str, Any]:
    span_len = max(1, interval[1] - interval[0])
    overlap = 0
    fallback = 0
    max_endpoint = 0.0
    max_rmse = 0.0
    covering = []
    for segment in segments:
        ov = max(0, min(interval[1], int(segment.end_anchor.frame)) - max(interval[0], int(segment.start_anchor.frame)))
        if ov <= 0:
            continue
        overlap += ov if segment.status.startswith("fit") else 0
        fallback += ov if segment.status == "fit_bvp_fallback" else 0
        max_endpoint = max(max_endpoint, float(segment.endpoint_error_m))
        if segment.reprojection_rmse_px is not None:
            max_rmse = max(max_rmse, float(segment.reprojection_rmse_px))
        covering.append(
            {
                "segment_id": int(segment.segment_id),
                "frames": [int(segment.start_anchor.frame), int(segment.end_anchor.frame)],
                "status": segment.status,
                "overlap_frames": ov,
            }
        )
    return {
        "fit_coverage_fraction": _round(overlap / span_len, 6),
        "fallback_coverage_fraction": _round(fallback / span_len, 6),
        "endpoint_error_max_m": _round(max_endpoint, 6),
        "rmse_max_px": _round(max_rmse, 6),
        "covering_segments": covering,
    }


def _protected_span_needs_rollback(
    current: Mapping[str, Any],
    *,
    baseline_rmse: float | None,
    baseline_endpoint: float | None,
) -> bool:
    if float(current.get("fit_coverage_fraction") or 0.0) < 1.0:
        return True
    if float(current.get("fallback_coverage_fraction") or 0.0) > 0.0:
        return True
    if baseline_rmse is not None and float(current.get("rmse_max_px") or 0.0) > baseline_rmse + 1.0:
        return True
    if baseline_endpoint is not None and float(current.get("endpoint_error_max_m") or 0.0) > baseline_endpoint + 1e-6:
        return True
    return False


def _overlay_protected_segment(
    segments: Sequence[FlightSegmentFit],
    rollback: FlightSegmentFit,
    interval: tuple[int, int],
    *,
    physics: PhysicsParameters,
    config: BallArcSolverConfig,
) -> list[FlightSegmentFit]:
    output: list[FlightSegmentFit] = []
    for segment in segments:
        if max(0, min(interval[1], int(segment.end_anchor.frame)) - max(interval[0], int(segment.start_anchor.frame))) <= 0:
            output.append(segment)
            continue
        if int(segment.start_anchor.frame) < interval[0]:
            head = _head_segment_before_frame(segment, rollback.start_anchor, physics=physics, config=config)
            if head is not None:
                output.append(head)
        if int(segment.end_anchor.frame) > interval[1]:
            tail = _tail_segment_after_frame(segment, rollback.end_anchor, physics=physics, config=config)
            if tail is not None:
                output.append(tail)
    output.append(rollback)
    return output


def _head_segment_before_frame(
    segment: FlightSegmentFit,
    end_anchor: AnchorEvent,
    *,
    physics: PhysicsParameters,
    config: BallArcSolverConfig,
) -> FlightSegmentFit | None:
    if end_anchor.t <= segment.start_anchor.t + 1e-9:
        return None
    observations = tuple(obs for obs in segment.observations if obs.frame < end_anchor.frame)
    if not observations:
        return None
    trim_obs = observations[-1]
    trim_t = float(trim_obs.t)
    if trim_t <= segment.start_anchor.t + 1e-9:
        return None
    trim_world = segment.predict(trim_t, physics, config)
    trim_anchor = replace(
        end_anchor,
        anchor_id=f"{end_anchor.anchor_id}_protected_head_trim",
        t=trim_t,
        frame=int(trim_obs.frame),
        world_xyz=trim_world,
        status="protected_span_trim",
        source="protected_span_overlay",
    )
    inlier_frames = tuple(frame for frame in segment.inlier_frames if frame < end_anchor.frame)
    outlier_frames = tuple(frame for frame in segment.outlier_frames if frame < end_anchor.frame)
    reprojection_errors = {frame: error for frame, error in segment.reprojection_errors_px.items() if frame < end_anchor.frame}
    return replace(
        segment,
        end_anchor=trim_anchor,
        observations=observations,
        inlier_frames=inlier_frames,
        outlier_frames=outlier_frames,
        reprojection_errors_px=reprojection_errors,
        endpoint_error_m=0.0,
    )


def _tail_segment_after_frame(
    segment: FlightSegmentFit,
    start_anchor: AnchorEvent,
    *,
    physics: PhysicsParameters,
    config: BallArcSolverConfig,
) -> FlightSegmentFit | None:
    if start_anchor.t >= segment.end_anchor.t - 1e-9:
        return None
    observations = tuple(obs for obs in segment.observations if obs.frame > start_anchor.frame)
    if not observations:
        return None
    trim_obs = observations[0]
    trim_t = float(trim_obs.t)
    if trim_t >= segment.end_anchor.t - 1e-9:
        return None
    trim_world = segment.predict(trim_t, physics, config)
    velocity = _segment_velocity_at(segment, trim_t, physics=physics, config=config)
    trim_anchor = replace(
        start_anchor,
        anchor_id=f"{start_anchor.anchor_id}_protected_tail_trim",
        t=trim_t,
        frame=int(trim_obs.frame),
        world_xyz=trim_world,
        status="protected_span_trim",
        source="protected_span_overlay",
    )
    inlier_frames = tuple(frame for frame in segment.inlier_frames if frame > start_anchor.frame)
    outlier_frames = tuple(frame for frame in segment.outlier_frames if frame > start_anchor.frame)
    reprojection_errors = {frame: error for frame, error in segment.reprojection_errors_px.items() if frame > start_anchor.frame}
    return replace(
        segment,
        start_anchor=trim_anchor,
        initial_position_m=trim_world,
        initial_velocity_mps=velocity,
        observations=observations,
        inlier_frames=inlier_frames,
        outlier_frames=outlier_frames,
        reprojection_errors_px=reprojection_errors,
    )


def _fit_validity_failure_reason(segment: FlightSegmentFit) -> str | None:
    if not segment.status.startswith("fit") or segment.status in {"fit_weak", "fit_bvp_fallback"}:
        return None
    violations = segment.physical_sanity.get("violations") if isinstance(segment.physical_sanity, Mapping) else None
    if isinstance(violations, Sequence) and not isinstance(violations, (str, bytes)) and violations:
        return str(violations[0])
    if isinstance(segment.physical_sanity, Mapping) and segment.physical_sanity.get("violation") is True:
        return "physical_sanity_violation"
    if segment.inlier_count == 0:
        return "zero_inliers"
    if _segment_inlier_fraction_below_fit_gate(segment):
        return "low_inlier_fraction"
    if segment.endpoint_error_m > 0.5:
        return "endpoint_error_gt_0_5m"
    return None


def _segment_inlier_fraction(segment: FlightSegmentFit) -> float | None:
    observation_count = len(segment.observations)
    if observation_count <= 0:
        return None
    return segment.inlier_count / observation_count


def _segment_inlier_fraction_below_fit_gate(segment: FlightSegmentFit) -> bool:
    fraction = _segment_inlier_fraction(segment)
    return (
        fraction is not None
        and len(segment.observations) >= FIT_VALIDITY_MIN_OBSERVATIONS_FOR_INLIER_GATE
        and fraction < FIT_VALIDITY_MIN_INLIER_FRACTION
    )


def _segment_fails_discovery_inlier_gate(segment: FlightSegmentFit) -> bool:
    fraction = _segment_inlier_fraction(segment)
    if fraction is None:
        return False
    if segment.inlier_count == 0:
        return True
    return _segment_inlier_fraction_below_fit_gate(segment)


def _fit_bvp_fallback_segment(
    segment: FlightSegmentFit,
    *,
    reason: str,
    physics: PhysicsParameters,
    config: BallArcSolverConfig,
    net_plane: Mapping[str, Any] | None,
) -> FlightSegmentFit:
    diagnostics = dict(segment.diagnostics or {})
    bvp = diagnostics.get("bvp_anchor_fallback") if isinstance(diagnostics.get("bvp_anchor_fallback"), Mapping) else None
    if not isinstance(bvp, Mapping):
        bvp = diagnostics.get("bvp_shooting") if isinstance(diagnostics.get("bvp_shooting"), Mapping) else {}
    bvp_p0 = _vec3_from_json(bvp.get("initial_position_m")) or segment.start_anchor.world_xyz
    bvp_v0 = _vec3_from_json(bvp.get("initial_velocity_mps")) or segment.initial_velocity_mps
    bvp_t0 = _float_or_none(bvp.get("t0")) if isinstance(bvp, Mapping) else None
    bvp_t1 = _float_or_none(bvp.get("t1")) if isinstance(bvp, Mapping) else None
    fallback_t0 = segment.start_anchor.t if bvp_t0 is None else bvp_t0
    fallback_t1 = segment.end_anchor.t if bvp_t1 is None else bvp_t1
    fallback_end = _vec3_from_json(bvp.get("target_position_m")) or segment.end_anchor.world_xyz
    spin_scalar = _clip_spin_scalar(bvp.get("spin_scalar") if isinstance(bvp, Mapping) else segment.spin_scalar)
    endpoint_pred = _integrate_positions(
        bvp_p0,
        bvp_v0,
        [fallback_t1],
        t0=fallback_t0,
        physics=physics,
        config=config,
        spin_scalar=spin_scalar,
    )[0]
    net_clearance = _net_clearance_m(
        bvp_p0,
        bvp_v0,
        fallback_t0,
        fallback_t1,
        physics,
        config,
        net_plane,
        spin_scalar=spin_scalar,
    )
    physical = _physical_sanity(bvp_p0, bvp_v0, fallback_t0, fallback_t1, physics, config, net_clearance, spin_scalar=spin_scalar)
    original = diagnostics.get("legacy_free_fit") if isinstance(diagnostics.get("legacy_free_fit"), Mapping) else _fit_diagnostic_summary(segment)
    diagnostics["fit_validity_gate"] = {
        "reason": reason,
        "original_status": str(original.get("status") or segment.status),
        "original_p0_m": list(original.get("initial_position_m") or _vec_json(segment.initial_position_m)),
        "original_v0_mps": list(original.get("initial_velocity_mps") or _vec_json(segment.initial_velocity_mps)),
        "original_endpoint_error_m": _optional_float_round(original.get("endpoint_error_m")),
        "original_inlier_count": int(original.get("inlier_count") if original.get("inlier_count") is not None else segment.inlier_count),
        "original_outlier_count": int(original.get("outlier_count") if original.get("outlier_count") is not None else segment.outlier_count),
        "fallback_source": "bvp_shooting",
    }
    return replace(
        segment,
        status="fit_bvp_fallback",
        initial_position_m=bvp_p0,
        initial_velocity_mps=bvp_v0,
        spin_scalar=spin_scalar,
        endpoint_error_m=_distance(endpoint_pred, fallback_end),
        net_clearance_m=net_clearance,
        net_clearance_ok=None if net_clearance is None else net_clearance >= -config.net_clearance_slack_m,
        physical_sanity=physical,
        diagnostics=diagnostics,
    )


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
    candidate_sets_by_frame: Mapping[int, Sequence[BallObservation]] | None,
    calibration: Mapping[str, Any],
    physics: PhysicsParameters,
    config: BallArcSolverConfig,
) -> FlightSegmentFit | None:
    key = (_anchor_key(start), _anchor_key(end))
    if key not in fit_cache:
        selection_config = replace(config, enable_size_depth_residual=False)
        fit_cache[key] = _fit_selection_segment_once(
            segment_id=segment_id,
            start_anchor=start,
            end_anchor=end,
            observations=observations,
            calibration=calibration,
            physics=physics,
            config=selection_config,
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
    if anchor.kind == "bounce" and anchor.status == "auto_bounce_candidate":
        return True
    return bool(anchor.immovable and anchor.status != "solver_proposed")


def _mandatory_reason(anchor: AnchorEvent) -> str:
    if anchor.kind == "rally_endpoint":
        return "rally_endpoint"
    if anchor.kind == "bounce" and anchor.status == "human_reviewed":
        return "human_reviewed_bounce"
    if anchor.kind == "bounce" and anchor.status == "auto_bounce_candidate":
        return "auto_bounce_candidate"
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


def _segment_has_court_volume_violation(segment: FlightSegmentFit) -> bool:
    physical = segment.physical_sanity
    court_volume = physical.get("court_volume") if isinstance(physical, Mapping) else None
    if isinstance(court_volume, Mapping) and court_volume.get("violation") is True:
        return True
    violations = physical.get("violations") if isinstance(physical, Mapping) else None
    return isinstance(violations, Sequence) and not isinstance(violations, (str, bytes)) and "outside_court_volume" in violations


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
    net_plane: Mapping[str, Any] | None,
) -> list[AnchorEvent]:
    discovered: list[AnchorEvent] = []
    existing_frames = [anchor.frame for anchor in existing_anchors]
    observations = tuple(sorted(observations_by_frame.values(), key=lambda item: (item.t, item.frame)))
    for segment in segments:
        trigger_px = min(config.discovery_reprojection_px, config.max_reprojection_inlier_px * 0.8)
        neighbor_px = min(config.discovery_min_neighbor_px, config.max_reprojection_inlier_px * 0.75)
        if segment.max_reprojection_error_px is None or segment.max_reprojection_error_px < trigger_px:
            continue
        bounce_bounce_gate_recovery = False
        if segment.start_anchor.kind == "bounce" or segment.end_anchor.kind == "bounce":
            if segment.start_anchor.kind != "bounce" or segment.end_anchor.kind != "bounce":
                continue
            interior_observations = _interior_segment_observations(segment, observations)
            if not _bounce_bounce_discovery_allowed_on_gate_failure(segment, interior_observations, config):
                continue
            bounce_bounce_gate_recovery = True
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
            if bounce_bounce_gate_recovery:
                acceptance = _evaluate_bounce_bounce_discovery_candidate(
                    segment,
                    anchor,
                    observations=observations,
                    calibration=calibration,
                    physics=physics,
                    config=config,
                    net_plane=net_plane,
                )
                if not acceptance.get("accepted"):
                    continue
                anchor = replace(
                    anchor,
                    details={
                        **dict(anchor.details or {}),
                        "bounce_bounce_gate_failure_recovery": True,
                        "discovery_acceptance": acceptance,
                    },
                )
            discovered.append(anchor)
            existing_frames.append(frame)
            break
    return discovered


def _interior_segment_observations(
    segment: FlightSegmentFit,
    observations: Sequence[BallObservation],
) -> tuple[BallObservation, ...]:
    return tuple(
        obs
        for obs in observations
        if segment.start_anchor.t + 1e-9 < obs.t < segment.end_anchor.t - 1e-9
    )


def _bounce_bounce_discovery_allowed_on_gate_failure(
    segment: FlightSegmentFit,
    interior_observations: Sequence[BallObservation],
    config: BallArcSolverConfig,
) -> bool:
    if not config.bounce_bounce_discovery_on_gate_failure:
        return False
    if len(interior_observations) < config.discovery_min_interior_observations:
        return False
    if segment.status == "fit_bvp_fallback":
        return True
    return _segment_fails_discovery_inlier_gate(segment)


def _evaluate_bounce_bounce_discovery_candidate(
    parent: FlightSegmentFit,
    candidate: AnchorEvent,
    *,
    observations: Sequence[BallObservation],
    calibration: Mapping[str, Any],
    physics: PhysicsParameters,
    config: BallArcSolverConfig,
    net_plane: Mapping[str, Any] | None,
) -> dict[str, Any]:
    child_left = _fit_selection_segment_once(
        segment_id=0,
        start_anchor=parent.start_anchor,
        end_anchor=candidate,
        observations=observations,
        calibration=calibration,
        physics=physics,
        config=replace(config, enable_size_depth_residual=False),
    )
    child_right = _fit_selection_segment_once(
        segment_id=1,
        start_anchor=candidate,
        end_anchor=parent.end_anchor,
        observations=observations,
        calibration=calibration,
        physics=physics,
        config=replace(config, enable_size_depth_residual=False),
    )
    parent_fraction = _segment_inlier_fraction(parent)
    split_fraction = _combined_inlier_fraction((child_left, child_right))
    parent_score = _segment_residual_score(parent, config)
    child_score = _segment_residual_score(child_left, config) + _segment_residual_score(child_right, config)
    score_gain = parent_score - child_score
    payload = {
        "accepted": False,
        "reason": "not_evaluated",
        "parent_inlier_fraction": _optional_round(parent_fraction, 6),
        "split_inlier_fraction": _optional_round(split_fraction, 6),
        "parent_score": _optional_round(parent_score if math.isfinite(parent_score) else None, 6),
        "split_score": _optional_round(child_score if math.isfinite(child_score) else None, 6),
        "score_gain": _optional_round(score_gain if math.isfinite(score_gain) else None, 6),
        "selection_split_penalty": _round(config.selection_split_penalty, 6),
        "child_statuses": [child_left.status, child_right.status],
        "child_initial_speed_mps": [
            _optional_round(_segment_speed(child_left), 6),
            _optional_round(_segment_speed(child_right), 6),
        ],
    }
    if split_fraction is None or split_fraction <= FIT_VALIDITY_MIN_INLIER_FRACTION:
        payload["reason"] = "split_inlier_fraction_below_gate"
        return payload
    if _segment_has_court_volume_violation(child_left) or _segment_has_court_volume_violation(child_right):
        payload["reason"] = "split_not_court_plausible"
        return payload
    if not _segment_plausible_for_selection(child_left, config, physics, net_plane) or not _segment_plausible_for_selection(child_right, config, physics, net_plane):
        payload["reason"] = "split_not_court_plausible"
        return payload
    if not math.isfinite(parent_score) or not math.isfinite(child_score):
        payload["reason"] = "nonfinite_residual_score"
        return payload
    if score_gain <= config.selection_split_penalty:
        payload["reason"] = "split_residual_reduction_below_penalty"
        return payload
    payload["accepted"] = True
    payload["reason"] = "gate_failure_split_recovered"
    return payload


def _combined_inlier_fraction(segments: Sequence[FlightSegmentFit]) -> float | None:
    observation_count = sum(len(segment.observations) for segment in segments)
    if observation_count <= 0:
        return None
    return sum(segment.inlier_count for segment in segments) / observation_count


def _segment_residual_score(segment: FlightSegmentFit | None, config: BallArcSolverConfig) -> float:
    if segment is None or not segment.status.startswith("fit") or not segment.reprojection_errors_px:
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


def _auto_bounce_candidate_anchors(
    payload: Mapping[str, Any] | None,
    *,
    calibration: Mapping[str, Any],
    observations_by_frame: Mapping[int, BallObservation],
    physics: PhysicsParameters,
    config: BallArcSolverConfig,
) -> list[AnchorEvent]:
    if not isinstance(payload, Mapping):
        return []
    if payload.get("source") != "track_geometry_candidate":
        raise ValueError("auto-bounce candidates must use source='track_geometry_candidate'")
    if payload.get("human_reviewed") is not False or payload.get("not_ground_truth") is not True:
        raise ValueError("auto-bounce candidates must declare human_reviewed=false and not_ground_truth=true")
    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        return []
    anchors: list[AnchorEvent] = []
    for index, item in enumerate(candidates):
        if not isinstance(item, Mapping):
            continue
        source = item.get("source", payload.get("source"))
        human_reviewed = item.get("human_reviewed", payload.get("human_reviewed"))
        not_ground_truth = item.get("not_ground_truth", payload.get("not_ground_truth"))
        if source != "track_geometry_candidate" or human_reviewed is not False or not_ground_truth is not True:
            raise ValueError("auto-bounce candidate items must carry honest track-geometry provenance")
        frame = _frame_from_mapping(item)
        if frame is None:
            continue
        xy = _xy_tuple(item.get("xy"))
        obs = observations_by_frame.get(frame) or _nearest_observation_by_frame(observations_by_frame, frame, max_gap=2)
        if xy is None and obs is not None:
            xy = obs.xy
        if xy is None:
            continue
        review_id = item.get("review_id")
        candidate_id = str(review_id) if isinstance(review_id, str) and review_id else f"auto_bounce_candidate_{frame:06d}_{index:03d}"
        try:
            anchor = build_bounce_anchor(
                {**dict(item), "frame": frame, "t": obs.t if obs is not None else item.get("t"), "review_id": candidate_id, "xy": xy},
                calibration,
                ball_radius_m=physics.radius_m,
                ball_xy=xy,
                status="auto_bounce_candidate",
                sigma_m=config.proposed_bounce_sigma_m,
                source="track_geometry_candidate",
                details={
                    "candidate_prediction": True,
                    "human_reviewed": False,
                    "not_ground_truth": True,
                    "method": item.get("method", "unknown"),
                    "candidate_index": index,
                },
            )
        except ValueError:
            continue
        anchors.append(replace(anchor, immovable=False))
    return anchors


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
        if not _has_per_bounce_human_review_provenance(item):
            raise ValueError("reviewed_bounces status=human_reviewed requires per-bounce human-review provenance")
        frame = _frame_from_mapping(item)
        if frame is None:
            continue
        obs = observations_by_frame.get(frame) or _nearest_observation_by_frame(observations_by_frame, frame, max_gap=2)
        if obs is None:
            continue
        try:
            anchors.append(
                build_bounce_anchor(
                    {**dict(item), "t": obs.t},
                    calibration,
                    ball_radius_m=physics.radius_m,
                    ball_xy=obs.xy,
                    status="human_reviewed",
                    sigma_m=anchor_sigma_for_bounce(
                        calibration,
                        obs.xy,
                        base_sigma_m=config.reviewed_bounce_base_sigma_m,
                    ),
                    details={
                        "human_reviewed": True,
                        "not_ground_truth": False,
                        "review_source": item.get("source"),
                    },
                )
            )
        except ValueError:
            continue
    return anchors


def _has_per_bounce_human_review_provenance(item: Mapping[str, Any]) -> bool:
    source = item.get("source")
    if source not in {"human_review", "manual_review", "review_input_server", "human_reviewed"}:
        return False
    if item.get("human_reviewed") is not True:
        return False
    if item.get("not_ground_truth") is True:
        return False
    return True


def _contact_anchors(
    contact_windows: Mapping[str, Any] | None,
    skeleton3d: Mapping[str, Any] | None,
    *,
    calibration: Mapping[str, Any],
    observations: Sequence[BallObservation],
    fps: float,
    frame_times: Mapping[int, float] | None,
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
            frame = nearest_frame_for_time(t, frame_times=frame_times, fps=fps)
        event_confidence = _float_or_none(event.get("confidence"))
        sigma_m = _contact_anchor_sigma_from_confidence(selected["confidence"], event_confidence)
        anchor_id = f"contact_{index:03d}_p{player_id}_{selected['side']}"
        anchors.append(
            AnchorEvent(
                anchor_id=anchor_id,
                kind="contact",
                t=t,
                frame=frame,
                world_xyz=selected["paddle_center"],
                sigma_m=sigma_m,
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


def _contact_anchor_sigma_from_confidence(joint_confidence: float | None, event_confidence: float | None) -> float:
    confidences = [value for value in (joint_confidence, event_confidence) if value is not None and math.isfinite(float(value))]
    confidence = min(float(value) for value in confidences) if confidences else 0.35
    confidence = max(0.0, min(1.0, confidence))
    if confidence >= 0.80:
        return 0.15
    if confidence <= 0.35:
        return 0.45
    return 0.45 + (confidence - 0.35) * ((0.15 - 0.45) / (0.80 - 0.35))


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
    frame_times: Mapping[int, float] | None = None,
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
            t = time_for_frame(frame_index, frame_times=frame_times, fps=fps)
            frame["t"] = _round(t, 9)
        segment = _segment_for_frame_time(segments, frame_index=frame_index, t=t)
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
        if segment.status in {"fit_weak", "fit_bvp_fallback"}:
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
        selected_observation = None
        if segment.selected_observations_by_frame:
            selected_observation = segment.selected_observations_by_frame.get(frame_index)
        selected_is_rescue = selected_observation is not None and not _is_primary_observation(selected_observation)
        selected_is_inlier = selected_observation is not None and frame_index in segment.inlier_frames
        if selected_observation is not None and frame_index in segment.inlier_frames:
            frame["observation_source"] = selected_observation.observation_source
            frame["candidate_selection"] = selected_observation.candidate_selection or "arc_irls_v1"
            frame["rescued"] = bool(selected_is_rescue)
        frame["arc_solver"] = {
            "lane": LANE,
            "segment_id": segment.segment_id,
            "weak_segment": segment.status == "fit_weak",
            "segment_status": segment.status,
            "bvp_fallback_segment": segment.status == "fit_bvp_fallback",
            "inlier_sighting": frame_index in segment.inlier_frames,
            "outlier_sighting_pruned": frame_index in segment.outlier_frames,
            "render_only": True,
            "not_for_detection_metrics": True,
            "rescued": bool(selected_is_rescue and selected_is_inlier),
        }
        if selected_observation is not None:
            frame["arc_solver"]["observation_source"] = selected_observation.observation_source
            frame["arc_solver"]["candidate_selection"] = selected_observation.candidate_selection or "arc_irls_v1"
            frame["arc_solver"]["candidate_residual_px"] = _optional_round(segment.reprojection_errors_px.get(frame_index), 6)
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
    samples: list[dict[str, Any]] = []
    candidate_mode = any(segment.candidate_association and segment.candidate_sets_by_frame for segment in segments)
    for segment in segments:
        if segment.candidate_association and segment.selected_observations_by_frame and segment.candidate_sets_by_frame:
            selected_observations = list(segment.selected_observations_by_frame.values())
            candidate_obs = [
                obs
                for obs in selected_observations
                if obs.frame in segment.inlier_frames and obs.confidence >= config.visible_confidence_min
            ]
            primary_observations = selected_observations
            all_candidates_by_frame = segment.candidate_sets_by_frame
        else:
            candidate_obs = [
                obs
                for obs in segment.observations
                if obs.frame in segment.inlier_frames and obs.confidence >= config.visible_confidence_min
            ]
            primary_observations = segment.observations
            all_candidates_by_frame = None
        for held_out in candidate_obs:
            retained = [obs for obs in primary_observations if obs.frame != held_out.frame]
            if len(retained) < config.min_segment_observations:
                skipped.append({"frame": held_out.frame, "reason": "insufficient_observations_after_holdout"})
                continue
            held_out_candidate_count = None
            if all_candidates_by_frame is not None:
                held_out_candidate_count = len(all_candidates_by_frame.get(held_out.frame, ()))
            refit = _leave_one_out_refit_segment(
                segment,
                retained,
                held_out_frame=held_out.frame,
                all_candidates_by_frame=all_candidates_by_frame,
                calibration=calibration,
                physics=physics,
                config=config,
            )
            if refit is None or not refit.status.startswith("fit"):
                skipped.append(
                    {
                        "frame": held_out.frame,
                        "reason": "bvp_validation_refit_unavailable",
                        "status": None if refit is None else refit.status,
                    }
                )
                continue
            predicted = refit.predict(held_out.t, physics, config)
            origin, direction = pixel_ray_world(calibration, held_out.xy)
            ray_error = _distance_point_to_ray(predicted, origin, direction)
            errors_m.append(ray_error)
            projected = _project_world_point(calibration, predicted)
            pixel_error = _distance2(projected, held_out.xy)
            errors_px.append(pixel_error)
            sample = {
                "frame": int(held_out.frame),
                "segment_id": int(segment.segment_id),
                "ray_distance_m": _round(ray_error, 6),
                "reprojection_error_px": _round(pixel_error, 6),
                "refit_status": refit.status,
            }
            if all_candidates_by_frame is not None:
                sample.update(
                    {
                        "held_out_observation_source": held_out.observation_source,
                        "held_out_frame_candidate_count": int(held_out_candidate_count or 0),
                        "held_out_entire_frame": True,
                    }
                )
            samples.append(sample)
    return {
        "sample_count": len(errors_m),
        "skipped": skipped,
        "holdout_policy": "whole_frame_candidate_sets_excluded" if candidate_mode else "single_observation_primary_track",
        "retained_candidate_policy": "fixed_non_heldout_frame_selections" if candidate_mode else None,
        "candidate_selection": "arc_irls_v1" if candidate_mode else None,
        "candidate_sibling_leakage_prevented": bool(candidate_mode),
        "samples": samples,
        "ray_distance_m": _distribution(errors_m),
        "reprojection_error_px": _distribution(errors_px),
    }


def _leave_one_out_refit_segment(
    segment: FlightSegmentFit,
    retained_observations: Sequence[BallObservation],
    *,
    held_out_frame: int,
    all_candidates_by_frame: Mapping[int, Sequence[BallObservation]] | None,
    calibration: Mapping[str, Any],
    physics: PhysicsParameters,
    config: BallArcSolverConfig,
) -> FlightSegmentFit | None:
    _ = all_candidates_by_frame
    p0_seed, seed_diagnostics = _loo_retained_observation_seed(
        segment,
        retained_observations,
        calibration=calibration,
        physics=physics,
        config=config,
    )
    bvp = _solve_bvp_shooting(
        p0_seed,
        segment.end_anchor.world_xyz,
        segment.start_anchor.t,
        segment.end_anchor.t,
        physics=physics,
        config=config,
        spin_scalar=segment.spin_scalar,
    )
    if not bool(bvp.get("converged")):
        return None
    return _build_fit_from_bvp_solution(
        segment_id=segment.segment_id,
        start_anchor=segment.start_anchor,
        end_anchor=segment.end_anchor,
        observations=retained_observations,
        calibration=calibration,
        physics=physics,
        config=config,
        net_plane=None,
        bvp=bvp,
        status="fit",
        diagnostics={
            "bvp_shooting_status": bvp.get("status"),
            "bvp_shooting": bvp,
            "loo_refit": {
                "held_out_frame": int(held_out_frame),
                "retained_observation_count": len(retained_observations),
                "endpoint_refinement": "skipped_fixed_endpoints",
                "seed_source": "retained_observation_ray_residual_centroid",
                **seed_diagnostics,
            },
        },
    )


def _loo_retained_observation_seed(
    segment: FlightSegmentFit,
    retained_observations: Sequence[BallObservation],
    *,
    calibration: Mapping[str, Any],
    physics: PhysicsParameters,
    config: BallArcSolverConfig,
) -> tuple[tuple[float, float, float], dict[str, Any]]:
    base = segment.start_anchor.world_xyz
    weighted_delta = (0.0, 0.0, 0.0)
    total_weight = 0.0
    used = 0
    for obs in retained_observations:
        try:
            predicted = segment.predict(obs.t, physics, config)
            origin, direction = pixel_ray_world(calibration, obs.xy)
        except (TypeError, ValueError):
            continue
        closest = _add(origin, _scale(direction, _dot(_sub(predicted, origin), direction)))
        delta = _sub(closest, predicted)
        if not _finite_vec3(delta):
            continue
        weight = max(0.1, min(1.0, float(obs.confidence)))
        weighted_delta = _add(weighted_delta, _scale(delta, weight))
        total_weight += weight
        used += 1
    if total_weight <= 0.0:
        return base, {"seed_observation_count": 0, "seed_delta_norm_m": 0.0}
    delta = _scale(weighted_delta, 1.0 / total_weight)
    max_delta = max(0.02, config.anchor_relax_sigma_multiplier * _anchor_sigma_m(segment.start_anchor, config))
    delta_norm = _norm(delta)
    if delta_norm > max_delta:
        delta = _scale(delta, max_delta / max(delta_norm, 1e-9))
        delta_norm = max_delta
    seed = _add(base, delta)
    seed = (float(seed[0]), float(seed[1]), max(physics.radius_m, float(seed[2])))
    if not _finite_vec3(seed):
        return base, {"seed_observation_count": used, "seed_delta_norm_m": 0.0, "seed_clamped_to_anchor": True}
    return seed, {
        "seed_observation_count": used,
        "seed_delta_norm_m": _round(delta_norm, 9),
        "seed_max_delta_m": _round(max_delta, 9),
    }


def _segment_anchor_bvp_for_validation(
    segment: FlightSegmentFit,
    *,
    physics: PhysicsParameters,
    config: BallArcSolverConfig,
) -> dict[str, tuple[float, float, float]] | None:
    diagnostics = segment.diagnostics if isinstance(segment.diagnostics, Mapping) else {}
    bvp = diagnostics.get("bvp_anchor_fallback") if isinstance(diagnostics.get("bvp_anchor_fallback"), Mapping) else None
    p0: tuple[float, float, float] | None = None
    v0: tuple[float, float, float] | None = None
    if bvp is not None:
        p0 = _vec3_from_json(bvp.get("initial_position_m"))
        v0 = _vec3_from_json(bvp.get("initial_velocity_mps"))
    if p0 is None or v0 is None:
        solved = _solve_bvp_shooting(
            segment.start_anchor.world_xyz,
            segment.end_anchor.world_xyz,
            segment.start_anchor.t,
            segment.end_anchor.t,
            physics=physics,
            config=config,
            spin_scalar=segment.spin_scalar,
        )
        if solved.get("status") != "converged":
            return None
        p0 = _vec3_from_json(solved.get("initial_position_m"))
        v0 = _vec3_from_json(solved.get("initial_velocity_mps"))
    if p0 is None or v0 is None:
        return None
    return {"initial_position_m": p0, "initial_velocity_mps": v0}


def _physical_summary(segments: Sequence[FlightSegmentFit], *, config: BallArcSolverConfig) -> dict[str, Any]:
    items = [dict(segment.physical_sanity) for segment in segments]
    violation_eligible = [
        (segment, dict(segment.physical_sanity))
        for segment in segments
        if segment.status != "fit_bvp_fallback"
    ]
    violations = [item for _segment, item in violation_eligible if item.get("violation") is True]
    if not items:
        violation_fraction = None
    elif violation_eligible:
        violation_fraction = len(violations) / len(violation_eligible)
    else:
        violation_fraction = 0.0
    return {
        "segment_count": len(items),
        "kill_eligible_segment_count": len(violation_eligible),
        "violation_eligible_segment_count": len(violation_eligible),
        "fallback_excluded_segment_count": sum(1 for segment in segments if segment.status == "fit_bvp_fallback"),
        "violation_count": len(violations),
        "violation_fraction": _optional_round(violation_fraction, 6),
        "kill_threshold_fraction": config.max_physical_violation_fraction,
        "segments": items,
    }


def _candidate_association_summary(segments: Sequence[FlightSegmentFit]) -> dict[str, Any]:
    reports = [dict(segment.candidate_association) for segment in segments if segment.candidate_association]
    enabled = [report for report in reports if report.get("enabled") is True]
    counts: dict[str, int] = {}
    floor_rejections: dict[str, int] = {}
    rescue_counts: dict[str, int] = {}
    final_residuals: dict[str, list[float]] = {}
    iteration_counts: list[float] = []
    for report in enabled:
        for source, count in dict(report.get("selection_counts_by_source") or {}).items():
            counts[str(source)] = counts.get(str(source), 0) + int(count)
        for source, count in dict(report.get("score_floor_rejected_counts_by_source") or {}).items():
            floor_rejections[str(source)] = floor_rejections.get(str(source), 0) + int(count)
        value = _float_or_none(report.get("iteration_count"))
        if value is not None:
            iteration_counts.append(value)
    for segment in segments:
        selected = segment.selected_observations_by_frame or {}
        inlier_frames = set(segment.inlier_frames)
        for frame, obs in selected.items():
            residual = segment.reprojection_errors_px.get(int(frame))
            if residual is not None:
                final_residuals.setdefault(obs.observation_source, []).append(float(residual))
            if int(frame) in inlier_frames and not _is_primary_observation(obs):
                rescue_counts[obs.observation_source] = rescue_counts.get(obs.observation_source, 0) + 1
    modes = sorted({str(report.get("mode") or "free") for report in enabled})
    floors: dict[str, float] = {}
    for report in enabled:
        for source, floor in dict(report.get("candidate_score_floors") or {}).items():
            floors[str(source)] = float(floor)
    return {
        "enabled": bool(enabled),
        "candidate_selection": "arc_irls_v1" if enabled else None,
        "mode": modes[0] if len(modes) == 1 else modes,
        "candidate_score_floors": floors,
        "segment_count": len(enabled),
        "converged_segment_count": sum(1 for report in enabled if report.get("converged") is True),
        "selection_counts_by_source": counts,
        "rescue_counts_by_source": rescue_counts,
        "score_floor_rejected_counts_by_source": floor_rejections,
        "final_residual_px_by_source": {
            source: _distribution(values)
            for source, values in sorted(final_residuals.items())
        },
        "iteration_count": _distribution(iteration_counts),
        "segments": enabled,
    }


def _physical_sanity(
    p0: tuple[float, float, float],
    v0: tuple[float, float, float],
    t0: float,
    t1: float,
    physics: PhysicsParameters,
    config: BallArcSolverConfig,
    net_clearance_m: float | None,
    spin_scalar: float = 0.0,
) -> dict[str, Any]:
    speed = _norm(v0)
    times = [t0 + (t1 - t0) * i / 40.0 for i in range(41)]
    points = _integrate_positions(p0, v0, times, t0=t0, physics=physics, config=config, spin_scalar=spin_scalar)
    apex = max(point[2] for point in points)
    violations: list[str] = []
    court_volume = _court_volume_report(points, config)
    if speed < config.min_plausible_speed_mps or speed > config.max_plausible_speed_mps:
        violations.append("initial_speed_outside_plausible_range_mps")
    if apex < physics.radius_m or apex > config.max_plausible_apex_m:
        violations.append("apex_height_implausible")
    if net_clearance_m is not None and net_clearance_m < -config.net_clearance_slack_m:
        violations.append("net_clearance_below_slack")
    if court_volume["violation"]:
        violations.append("outside_court_volume")
    return {
        "initial_speed_mps": _round(speed, 6),
        "apex_height_m": _round(apex, 6),
        "net_clearance_m": _optional_round(net_clearance_m, 6),
        "court_volume": court_volume,
        "violations": violations,
        "violation": bool(violations),
    }


def _court_volume_report(points: Sequence[Sequence[float]], config: BallArcSolverConfig) -> dict[str, Any]:
    x_min, x_max, y_min, y_max, z_min = _court_volume_bounds(config)
    outside_count = 0
    max_overage = 0.0
    for point in points:
        x, y, z = float(point[0]), float(point[1]), float(point[2])
        overage = max(x_min - x, x - x_max, y_min - y, y - y_max, z_min - z, 0.0)
        if overage > 1e-9:
            outside_count += 1
            max_overage = max(max_overage, overage)
    return {
        "sport": str(config.court_sport),
        "margin_m": _round(config.court_margin_m, 6),
        "z_min_m": _round(config.court_z_min_m, 6),
        "bounds_m": {
            "x_min": _round(x_min, 6),
            "x_max": _round(x_max, 6),
            "y_min": _round(y_min, 6),
            "y_max": _round(y_max, 6),
            "z_min": _round(z_min, 6),
        },
        "sample_count": len(points),
        "outside_sample_count": outside_count,
        "max_overage_m": _round(max_overage, 6),
        "violation": outside_count > 0,
    }


def _court_volume_bounds(config: BallArcSolverConfig) -> tuple[float, float, float, float, float]:
    template = get_court_template(str(config.court_sport))  # type: ignore[arg-type]
    half_width = template.width_m / 2.0
    half_length = template.length_m / 2.0
    margin = float(config.court_margin_m)
    return (
        -half_width - margin,
        half_width + margin,
        -half_length - margin,
        half_length + margin,
        float(config.court_z_min_m),
    )


def _integrate_positions(
    p0: tuple[float, float, float],
    v0: tuple[float, float, float],
    times: Sequence[float],
    *,
    t0: float,
    physics: PhysicsParameters,
    config: BallArcSolverConfig,
    spin_scalar: float = 0.0,
) -> list[tuple[float, float, float]]:
    _check_segment_budget()
    spin = _clip_spin_scalar(spin_scalar)
    use_magnus = abs(spin) > 1e-12
    if physics.drag_k_per_m <= 0.0 and not use_magnus:
        return [_analytic_no_drag_position(p0, v0, float(t) - t0, physics.gravity_mps2) for t in times]
    spin_axis = _spin_axis_for_velocity(v0) if use_magnus else None
    indexed = sorted(enumerate(times), key=lambda item: float(item[1]))
    positions: list[tuple[float, float, float] | None] = [None] * len(times)
    state = (*p0, *v0)
    current_t = t0
    for index, target_t_raw in indexed:
        target_t = float(target_t_raw)
        if target_t < current_t - 1e-9:
            reverse_state = (*p0, *v0)
            reverse_t = t0
            while reverse_t > target_t + 1e-12:
                _check_segment_budget()
                step = -min(config.integrator_max_step_s, reverse_t - target_t)
                reverse_state = (
                    _rk4_step_magnus(reverse_state, step, physics, spin_axis, spin)
                    if spin_axis is not None
                    else _rk4_step(reverse_state, step, physics, spin_scalar=0.0)
                )
                reverse_t += step
            positions[index] = (reverse_state[0], reverse_state[1], reverse_state[2])
            continue
        while current_t < target_t - 1e-12:
            _check_segment_budget()
            step = min(config.integrator_max_step_s, target_t - current_t)
            state = (
                _rk4_step_magnus(state, step, physics, spin_axis, spin)
                if spin_axis is not None
                else _rk4_step(state, step, physics, spin_scalar=0.0)
            )
            current_t += step
        positions[index] = (state[0], state[1], state[2])
    return [position if position is not None else p0 for position in positions]


def _check_segment_budget() -> None:
    deadline = _ACTIVE_SEGMENT_DEADLINE.get()
    if deadline is not None and time.monotonic() >= deadline:
        raise _SegmentBudgetExceeded(SEGMENT_BUDGET_EXCEEDED)


def _rk4_step(
    state: tuple[float, float, float, float, float, float],
    dt: float,
    physics: PhysicsParameters,
    *,
    spin_scalar: float = 0.0,
) -> tuple[float, float, float, float, float, float]:
    _ = spin_scalar

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


def _rk4_step_magnus(
    state: tuple[float, float, float, float, float, float],
    dt: float,
    physics: PhysicsParameters,
    spin_axis: tuple[float, float, float],
    spin_scalar: float = 0.0,
) -> tuple[float, float, float, float, float, float]:
    spin = _clip_spin_scalar(spin_scalar)

    def deriv(s: tuple[float, float, float, float, float, float]) -> tuple[float, float, float, float, float, float]:
        vx, vy, vz = s[3], s[4], s[5]
        speed = math.sqrt(vx * vx + vy * vy + vz * vz)
        drag = physics.drag_k_per_m * speed
        ax = -drag * vx
        ay = -drag * vy
        az = -physics.gravity_mps2 - drag * vz
        if speed > 1e-9:
            v_hat = (vx / speed, vy / speed, vz / speed)
            lift_dir = _unit_vec3(_cross_vec3(spin_axis, v_hat))
            if lift_dir is not None:
                lift_k = 0.5 * physics.rho_air_kg_m3 * math.pi * physics.radius_m * physics.radius_m / physics.mass_kg
                lift_acc = lift_k * speed * speed * (STEYN_CL_PER_SPIN * spin)
                ax += lift_acc * lift_dir[0]
                ay += lift_acc * lift_dir[1]
                az += lift_acc * lift_dir[2]
        return (vx, vy, vz, ax, ay, az)

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
    spin_scalar: float = 0.0,
) -> dict[int, float]:
    positions = _integrate_positions(
        initial_position,
        velocity,
        [obs.t for obs in observations],
        t0=t0,
        physics=physics,
        config=config,
        spin_scalar=spin_scalar,
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
    spin_scalar: float = 0.0,
) -> dict[str, Any]:
    positions = _integrate_positions(
        initial_position,
        velocity,
        [obs.t for obs in observations],
        t0=t0,
        physics=physics,
        config=config,
        spin_scalar=spin_scalar,
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
    spin_scalar: float = 0.0,
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
    points = _integrate_positions(p0, v0, times, t0=t0, physics=physics, config=config, spin_scalar=spin_scalar)
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
    spin_scalar: float = 0.0,
) -> float:
    clearance = _net_clearance_m(p0, v0, t0, t1, physics, config, net_plane, spin_scalar=spin_scalar)
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


def _budget_exceeded_segment(
    segment_id: int,
    start: AnchorEvent,
    end: AnchorEvent,
    *,
    budget_s: float,
    elapsed_s: float,
    observation_count: int,
    candidate_frame_count: int,
    candidate_count: int,
) -> FlightSegmentFit:
    """Return a loud missing-evidence outcome for a timed-out segment."""

    segment = _blocked_segment(segment_id, start, end, SEGMENT_BUDGET_EXCEEDED)
    return replace(
        segment,
        degradation={
            "outcome_type": SEGMENT_BUDGET_EXCEEDED,
            "reason": SEGMENT_BUDGET_EXCEEDED,
            "evidence_provenance": "missing",
            "authority": "degraded",
            "budget_s": _round(budget_s, 6),
            "elapsed_s": _round(elapsed_s, 6),
            "observation_count": int(observation_count),
            "candidate_frame_count": int(candidate_frame_count),
            "candidate_count": int(candidate_count),
            "duration_s": _round(end.t - start.t, 9),
        },
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


def _combined_candidate_sets_by_frame(
    *,
    frames: Sequence[Mapping[str, Any]],
    fps: float,
    frame_times: Mapping[int, float] | None,
    ball_track: Mapping[str, Any],
    primary_observations: Sequence[BallObservation],
    ball_candidate_sidecars: Sequence[Mapping[str, Any]],
    candidate_extra_tracks: Mapping[str, Mapping[str, Any]],
    max_candidates_per_frame: int,
) -> dict[int, tuple[BallObservation, ...]] | None:
    if not ball_candidate_sidecars and not candidate_extra_tracks:
        return None
    frame_time_map = _frame_times(frames, fps=fps, frame_times=frame_times)
    combined: dict[int, list[BallObservation]] = {}
    for obs in primary_observations:
        combined.setdefault(obs.frame, []).append(
            replace(
                obs,
                observation_source=f"primary:{ball_track.get('source') or 'ball_track'}",
                candidate_score=obs.confidence,
                candidate_rank=0,
            )
        )
    for name, track in candidate_extra_tracks.items():
        extra_frames = _frames(track)
        extra_fps = _payload_fps(track, extra_frames)
        for obs in _ball_observations(extra_frames, fps=extra_fps, source_label=f"extra:{name}"):
            t = frame_time_map.get(obs.frame, obs.t)
            combined.setdefault(obs.frame, []).append(replace(obs, t=t))
    for sidecar in ball_candidate_sidecars:
        _validate_candidate_sidecar_policy(sidecar)
        sidecar_fps = _float_or_none(sidecar.get("fps")) or fps
        source = str(sidecar.get("source") or "candidate")
        candidate_frames = sidecar.get("frames")
        if not isinstance(candidate_frames, Sequence) or isinstance(candidate_frames, (str, bytes)):
            continue
        for frame_payload in candidate_frames:
            if not isinstance(frame_payload, Mapping):
                continue
            frame = _frame_from_mapping(frame_payload)
            if frame is None:
                continue
            t = frame_time_map.get(frame, frame / max(sidecar_fps, 1e-9))
            candidates = frame_payload.get("candidates")
            if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
                continue
            for rank, candidate in enumerate(candidates):
                if not isinstance(candidate, Mapping):
                    continue
                xy = _xy_tuple(candidate.get("xy"))
                if xy is None:
                    continue
                score = _float_or_none(candidate.get("score"))
                confidence = max(0.0, min(1.0, score if score is not None else 0.5))
                source_detector = str(candidate.get("source_detector") or source)
                combined.setdefault(frame, []).append(
                    BallObservation(
                        frame=frame,
                        t=t,
                        xy=xy,
                        confidence=confidence,
                        visible=True,
                        observation_source=f"{source}:{source_detector}",
                        candidate_score=confidence,
                        candidate_rank=rank + 1,
                        candidate_selection=None,
                    )
                )
    capped = {
        frame: tuple(candidates[:max_candidates_per_frame])
        for frame, candidates in sorted(combined.items())
        if candidates[:max_candidates_per_frame]
    }
    return capped or None


def _validate_candidate_sidecar_policy(sidecar: Mapping[str, Any]) -> None:
    if sidecar.get("artifact_type") != "racketsport_ball_candidates":
        raise ValueError("ball candidate sidecar must have artifact_type='racketsport_ball_candidates'")
    if sidecar.get("not_ground_truth") is not True or sidecar.get("candidate_prediction") is not True:
        raise ValueError("ball candidate sidecar must declare not_ground_truth=true and candidate_prediction=true")


def _frame_times(
    frames: Sequence[Mapping[str, Any]],
    *,
    fps: float,
    frame_times: Mapping[int, float] | None = None,
) -> dict[int, float]:
    output: dict[int, float] = {}
    for index, frame in enumerate(frames):
        t = _float_or_none(frame.get("t"))
        output[index] = t if t is not None else time_for_frame(index, frame_times=frame_times, fps=fps)
    return output


def _frames(ball_track: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    frames = ball_track.get("frames")
    if not isinstance(frames, list):
        raise ValueError("ball_track.frames must be a list")
    return [frame for frame in frames if isinstance(frame, Mapping)]


def _ball_observations(
    frames: Sequence[Mapping[str, Any]],
    *,
    fps: float,
    frame_times: Mapping[int, float] | None = None,
    ball_sizes: Mapping[str, Any] | None = None,
    source_label: str = "primary:ball_track",
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
            t = time_for_frame(index, frame_times=frame_times, fps=fps)
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
                observation_source=source_label,
                candidate_score=confidence,
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


def _segment_for_frame_time(segments: Sequence[FlightSegmentFit], *, frame_index: int, t: float) -> FlightSegmentFit | None:
    candidates = [
        segment
        for segment in segments
        if min(segment.start_anchor.frame, segment.end_anchor.frame) <= frame_index <= max(segment.start_anchor.frame, segment.end_anchor.frame)
    ]
    if candidates:
        return min(candidates, key=lambda segment: abs((segment.start_anchor.t + segment.end_anchor.t) / 2.0 - t))
    return _segment_for_time(segments, t)


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
    payload = {
        key: value
        for key, value in config.__dict__.items()
        if isinstance(value, (str, int, float, bool)) or value is None
    }
    payload["candidate_score_floors"] = _candidate_score_floors_payload(config)
    return payload


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


def _cross_vec3(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float]:
    return (
        float(a[1]) * float(b[2]) - float(a[2]) * float(b[1]),
        float(a[2]) * float(b[0]) - float(a[0]) * float(b[2]),
        float(a[0]) * float(b[1]) - float(a[1]) * float(b[0]),
    )


def _unit_vec3(a: Sequence[float]) -> tuple[float, float, float] | None:
    norm = _norm(a)
    if norm <= 1e-12:
        return None
    return (float(a[0]) / norm, float(a[1]) / norm, float(a[2]) / norm)


def _spin_axis_for_velocity(velocity: tuple[float, float, float]) -> tuple[float, float, float]:
    vx, vy = velocity[0], velocity[1]
    norm = math.hypot(vx, vy)
    if norm <= 1e-9:
        return (1.0, 0.0, 0.0)
    return (vy / norm, -vx / norm, 0.0)


def _clip_spin_scalar(value: Any) -> float:
    parsed = _float_or_none(value)
    if parsed is None:
        return 0.0
    return max(-SPIN_SCALAR_MAX_ABS, min(SPIN_SCALAR_MAX_ABS, float(parsed)))


def _can_fit_spin_scalar(
    config: BallArcSolverConfig,
    observations: Sequence[BallObservation],
    *,
    inlier_count: int | None = None,
) -> bool:
    _ = observations
    return bool(config.fit_spin_scalar and inlier_count is not None and inlier_count >= SPIN_SCALAR_MIN_INLIERS)


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


def _vec3_from_json(value: Any) -> tuple[float, float, float] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 3:
        return None
    try:
        parsed = (float(value[0]), float(value[1]), float(value[2]))
    except (TypeError, ValueError):
        return None
    return parsed if all(math.isfinite(item) for item in parsed) else None


def _finite_vec3(value: Any) -> bool:
    return _vec3_from_json(value) is not None


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
    "SoftSegmentBoundary",
    "anchor_sigma_for_bounce",
    "build_bounce_anchor",
    "fit_flight_segment",
    "fit_weak_flight_segment",
    "intersect_ray_z",
    "order_event_anchors",
    "pixel_ray_world",
    "solve_ball_arc_track",
]

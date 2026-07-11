"""Pure TT3D-pattern joint anchor-hypothesis search.

This module searches *candidate* bounce states from immutable image
observations.  It deliberately does not call the production arc solver, assign
trust bands, mark samples measured, or alter selected defaults.  The emitted
``anchor_events`` dictionaries follow the read contract of
``ball_arc_solver.AnchorEvent.to_json`` so a later, separately owned
integration can pass a selected hypothesis to that solver.

The search model is intentionally small: two gravity-only flight pieces meet
at a free court-plane bounce state.  Incoming velocity, restitution,
tangential retention, bounce time, and bounce plane coordinates are optimized
jointly with a Huber reprojection objective.  This is a hypothesis generator,
not an accuracy or physics-promotion gate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import json
import math
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = 1
ARTIFACT_TYPE = "racketsport_ball_joint_anchor_candidates"
SEARCH_METHOD = "tt3d_pattern_piecewise_ballistic_huber_v1"


class RefusalCode(str, Enum):
    """Machine-readable reasons for refusing to manufacture hypotheses."""

    INSUFFICIENT_OBSERVATIONS = "insufficient_observations"
    INSUFFICIENT_TEMPORAL_SUPPORT = "insufficient_temporal_support"
    INVALID_OBSERVATION = "invalid_observation"
    INVALID_CAMERA = "invalid_camera"
    DEGENERATE_COURT_PLANE = "degenerate_court_plane"
    DEGENERATE_RAY_PLANE = "degenerate_ray_plane"
    NUMERIC_DEPENDENCY_UNAVAILABLE = "numeric_dependency_unavailable"
    OPTIMIZATION_FAILED = "optimization_failed"


class AnchorSearchRefusal(ValueError):
    """Typed, fail-closed refusal returned instead of an empty candidate list."""

    def __init__(self, code: RefusalCode, message: str, *, details: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.details = dict(details or {})

    def to_json(self) -> dict[str, Any]:
        return {
            "status": "refused",
            "code": self.code.value,
            "message": str(self),
            "details": dict(self.details),
        }


@dataclass(frozen=True, order=True)
class ImageObservation:
    """One immutable 2D sighting in source PTS time."""

    frame: int
    pts_s: float
    u: float
    v: float
    confidence: float = 1.0

    @property
    def xy(self) -> tuple[float, float]:
        return (self.u, self.v)


@dataclass(frozen=True)
class CameraModel:
    """Pinhole world-to-camera model matching the arc-solver calibration contract."""

    fx: float
    fy: float
    cx: float
    cy: float
    rotation_world_to_camera: tuple[tuple[float, float, float], ...]
    translation_world_to_camera: tuple[float, float, float]

    @classmethod
    def from_calibration(cls, calibration: Mapping[str, Any]) -> CameraModel:
        intrinsics = calibration.get("intrinsics")
        extrinsics = calibration.get("extrinsics")
        if not isinstance(intrinsics, Mapping) or not isinstance(extrinsics, Mapping):
            raise AnchorSearchRefusal(
                RefusalCode.INVALID_CAMERA,
                "camera calibration requires intrinsics and extrinsics mappings",
            )
        try:
            rotation = tuple(tuple(float(value) for value in row) for row in extrinsics["R"])
            translation = tuple(float(value) for value in extrinsics["t"])
            model = cls(
                fx=float(intrinsics["fx"]),
                fy=float(intrinsics["fy"]),
                cx=float(intrinsics["cx"]),
                cy=float(intrinsics["cy"]),
                rotation_world_to_camera=rotation,
                translation_world_to_camera=translation,  # type: ignore[arg-type]
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise AnchorSearchRefusal(RefusalCode.INVALID_CAMERA, f"invalid camera calibration: {exc}") from exc
        _validate_camera(model)
        return model

    def to_json(self) -> dict[str, Any]:
        return {
            "intrinsics": {"fx": self.fx, "fy": self.fy, "cx": self.cx, "cy": self.cy},
            "extrinsics": {
                "R": [list(row) for row in self.rotation_world_to_camera],
                "t": list(self.translation_world_to_camera),
            },
        }


@dataclass(frozen=True)
class BoundedPlane:
    """A metric plane with bounded coordinates in a deterministic local basis."""

    point: tuple[float, float, float]
    normal: tuple[float, float, float]
    u_bounds_m: tuple[float, float]
    v_bounds_m: tuple[float, float]


@dataclass(frozen=True)
class NetConstraint:
    """Vertical net plane and required ball-center height at a crossing."""

    point: tuple[float, float, float]
    normal: tuple[float, float, float]
    height_m: float

    @classmethod
    def from_solver_mapping(cls, payload: Mapping[str, Any]) -> NetConstraint:
        plane = payload.get("plane")
        if not isinstance(plane, Mapping):
            raise ValueError("net payload requires plane mapping")
        point = _vec3(plane.get("point"), "net.plane.point")
        normal = _vec3(plane.get("normal"), "net.plane.normal")
        height = payload.get("center_height_in")
        if height is not None:
            height_m = float(height) * 0.0254
        else:
            height_m = float(payload.get("height_m", 0.8636))
        return cls(point=point, normal=normal, height_m=height_m)


@dataclass(frozen=True)
class JointAnchorSearchConfig:
    """Bounded numerical controls; none are pipeline defaults."""

    ball_radius_m: float = 0.0371
    gravity_mps2: float = 9.80665
    robust_pixel_sigma: float = 3.0
    robust_f_scale: float = 1.0
    ray_constraint_sigma_m: float = 0.03
    net_clearance_slack_m: float = 0.08
    net_constraint_sigma_m: float = 0.05
    min_observations: int = 8
    min_observations_each_side: int = 3
    min_time_span_s: float = 0.20
    max_speed_mps: float = 35.0
    min_incoming_normal_speed_mps: float = 0.15
    restitution_bounds: tuple[float, float] = (0.25, 0.95)
    tangential_retention_bounds: tuple[float, float] = (0.55, 1.0)
    max_time_seeds: int = 25
    starts_per_seed: int = 2
    max_ranked_candidates: int = 8
    dedupe_time_s: float = 1.0 / 120.0
    dedupe_position_m: float = 0.05
    max_nfev: int = 1800
    candidate_sigma_m: float = 0.18

    def __post_init__(self) -> None:
        positive = {
            "ball_radius_m": self.ball_radius_m,
            "gravity_mps2": self.gravity_mps2,
            "robust_pixel_sigma": self.robust_pixel_sigma,
            "robust_f_scale": self.robust_f_scale,
            "ray_constraint_sigma_m": self.ray_constraint_sigma_m,
            "net_constraint_sigma_m": self.net_constraint_sigma_m,
            "min_time_span_s": self.min_time_span_s,
            "max_speed_mps": self.max_speed_mps,
            "max_nfev": float(self.max_nfev),
            "candidate_sigma_m": self.candidate_sigma_m,
        }
        invalid = [name for name, value in positive.items() if not math.isfinite(float(value)) or value <= 0]
        if invalid:
            raise ValueError(f"search config fields must be positive: {', '.join(invalid)}")
        if self.min_observations < 2 * self.min_observations_each_side:
            raise ValueError("min_observations must support both temporal sides")
        if self.max_time_seeds < 1 or self.starts_per_seed < 1 or self.max_ranked_candidates < 1:
            raise ValueError("seed/start/candidate counts must be positive")


@dataclass(frozen=True)
class AnchorCandidate:
    """One ranked, untrusted anchor-set hypothesis."""

    candidate_id: str
    rank: int
    cost: float
    bounce_time_s: float
    bounce_frame: int
    bounce_world_xyz: tuple[float, float, float]
    incoming_velocity_mps: tuple[float, float, float]
    outgoing_velocity_mps: tuple[float, float, float]
    restitution: float
    tangential_retention: float
    reprojection_rmse_px: float
    reprojection_huber_cost: float
    ray_plane_distance_m: float
    net_clearance_m: float | None
    net_constraint_satisfied: bool | None
    anchor_events: tuple[Mapping[str, Any], ...]
    provenance: Mapping[str, Any]

    def to_json(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "rank": self.rank,
            "hypothesis_only": True,
            "cost": _round(self.cost, 9),
            "event": {
                "kind": "bounce",
                "t": _round(self.bounce_time_s, 9),
                "frame": self.bounce_frame,
                "world_xyz": _vec_json(self.bounce_world_xyz),
            },
            "state": {
                "incoming_velocity_mps": _vec_json(self.incoming_velocity_mps),
                "outgoing_velocity_mps": _vec_json(self.outgoing_velocity_mps),
                "restitution": _round(self.restitution, 9),
                "tangential_retention": _round(self.tangential_retention, 9),
            },
            "cost_components": {
                "reprojection_rmse_px": _round(self.reprojection_rmse_px, 9),
                "reprojection_huber_cost": _round(self.reprojection_huber_cost, 9),
                "ray_plane_distance_m": _round(self.ray_plane_distance_m, 9),
                "net_clearance_m": None if self.net_clearance_m is None else _round(self.net_clearance_m, 9),
                "net_constraint_satisfied": self.net_constraint_satisfied,
            },
            "anchor_events": [dict(anchor) for anchor in self.anchor_events],
            "provenance": dict(self.provenance),
        }


@dataclass(frozen=True)
class AnchorSearchResult:
    """Deterministic candidate artifact returned by the pure search API."""

    candidates: tuple[AnchorCandidate, ...]
    observation_digest_sha256: str
    camera_digest_sha256: str
    seed: int
    enumeration_count: int
    optimization_attempt_count: int
    config: JointAnchorSearchConfig

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": ARTIFACT_TYPE,
            "status": "candidates_only",
            "hypothesis_only": True,
            "search_method": SEARCH_METHOD,
            "seed": self.seed,
            "observation_digest_sha256": self.observation_digest_sha256,
            "camera_digest_sha256": self.camera_digest_sha256,
            "config": _jsonable_config(self.config),
            "summary": {
                "enumeration_count": self.enumeration_count,
                "optimization_attempt_count": self.optimization_attempt_count,
                "candidate_count": len(self.candidates),
            },
            "candidates": [candidate.to_json() for candidate in self.candidates],
            "policy": {
                "candidate_only": True,
                "marks_measured": False,
                "alters_trust_bands": False,
                "alters_defaults": False,
                "protected_labels_read": False,
            },
        }


@dataclass(frozen=True)
class _RawCandidate:
    cost: float
    bounce_time_s: float
    bounce_world_xyz: tuple[float, float, float]
    incoming_velocity_mps: tuple[float, float, float]
    outgoing_velocity_mps: tuple[float, float, float]
    restitution: float
    retention: float
    reprojection_rmse_px: float
    reprojection_huber_cost: float
    ray_plane_distance_m: float
    net_clearance_m: float | None
    net_ok: bool | None
    court_normal: tuple[float, float, float]
    seed_time_s: float
    start_index: int
    optimizer_success: bool
    optimizer_status: int
    optimizer_nfev: int
    optimizer_message: str


def search_joint_anchor_candidates(
    observations: Sequence[ImageObservation],
    camera: CameraModel,
    court_plane: BoundedPlane,
    *,
    net: NetConstraint | None = None,
    config: JointAnchorSearchConfig | None = None,
    seed: int = 0,
) -> AnchorSearchResult:
    """Enumerate and optimize ranked, hypothesis-only anchor sets.

    Raises :class:`AnchorSearchRefusal` when observations or geometry cannot
    honestly support a search.  It never returns an empty successful result.
    """

    cfg = config or JointAnchorSearchConfig()
    obs = tuple(sorted(tuple(observations), key=lambda item: (item.pts_s, item.frame)))
    _validate_observations(obs, cfg)
    _validate_camera(camera)
    plane = _normalized_plane(court_plane, ball_radius_m=cfg.ball_radius_m)
    plane_u, plane_v = _plane_basis(plane.normal)
    if net is not None:
        _normalized_vec(net.normal, RefusalCode.DEGENERATE_COURT_PLANE, "net.normal")

    try:
        import numpy as np
        from scipy.optimize import least_squares
    except ImportError as exc:
        raise AnchorSearchRefusal(
            RefusalCode.NUMERIC_DEPENDENCY_UNAVAILABLE,
            f"numpy/scipy required for anchor search: {exc}",
        ) from exc

    obs_digest = _digest([asdict(item) for item in obs])
    camera_digest = _digest(camera.to_json())
    seed_times = _enumerate_seed_times(obs, cfg)
    rng = np.random.default_rng(int(seed))
    raw: list[_RawCandidate] = []
    attempts = 0

    for seed_time in seed_times:
        seed_pixel = _interpolated_pixel(obs, seed_time)
        origin, direction = pixel_ray_world(camera, seed_pixel)
        try:
            seed_anchor = intersect_ray_plane(origin, direction, plane.point, plane.normal)
        except AnchorSearchRefusal:
            continue
        seed_a = _dot(_sub(seed_anchor, plane.point), plane_u)
        seed_b = _dot(_sub(seed_anchor, plane.point), plane_v)
        seed_a = min(max(seed_a, plane.u_bounds_m[0]), plane.u_bounds_m[1])
        seed_b = min(max(seed_b, plane.v_bounds_m[0]), plane.v_bounds_m[1])
        tangent_guess = _tangent_velocity_guess(obs, camera, plane, plane_u, plane_v, seed_time)
        initial_base = np.asarray(
            [
                seed_time,
                seed_a,
                seed_b,
                tangent_guess[0],
                tangent_guess[1],
                -max(1.5, abs(tangent_guess[2])),
                0.62,
                0.86,
            ],
            dtype=float,
        )
        lower = np.asarray(
            [
                obs[cfg.min_observations_each_side - 1].pts_s + 1e-6,
                plane.u_bounds_m[0],
                plane.v_bounds_m[0],
                -cfg.max_speed_mps,
                -cfg.max_speed_mps,
                -cfg.max_speed_mps,
                cfg.restitution_bounds[0],
                cfg.tangential_retention_bounds[0],
            ],
            dtype=float,
        )
        upper = np.asarray(
            [
                obs[-cfg.min_observations_each_side].pts_s - 1e-6,
                plane.u_bounds_m[1],
                plane.v_bounds_m[1],
                cfg.max_speed_mps,
                cfg.max_speed_mps,
                -cfg.min_incoming_normal_speed_mps,
                cfg.restitution_bounds[1],
                cfg.tangential_retention_bounds[1],
            ],
            dtype=float,
        )
        if not bool(np.all(lower < upper)):
            raise AnchorSearchRefusal(
                RefusalCode.INSUFFICIENT_TEMPORAL_SUPPORT,
                "temporal or plane bounds leave no feasible optimization interval",
            )

        for start_index in range(cfg.starts_per_seed):
            attempts += 1
            initial = initial_base.copy()
            if start_index:
                initial[0] += float(rng.normal(0.0, max(1e-4, (upper[0] - lower[0]) / 80.0)))
                initial[1:3] += rng.normal(0.0, 0.12, size=2)
                initial[3:5] += rng.normal(0.0, 1.5, size=2)
                initial[5] += float(rng.normal(0.0, 1.0))
                initial[6:8] += rng.normal(0.0, 0.035, size=2)
            initial = np.minimum(np.maximum(initial, lower + 1e-8), upper - 1e-8)

            def residuals(params: Any) -> Any:
                state = _state_from_params(params, plane, plane_u, plane_v, cfg)
                values: list[float] = []
                for item in obs:
                    point = _piecewise_position(item.pts_s, state, plane.normal, cfg.gravity_mps2)
                    projected, depth = project_world(camera, point)
                    sigma = cfg.robust_pixel_sigma / math.sqrt(max(0.05, min(1.0, item.confidence)))
                    if depth <= 1e-6 or not all(math.isfinite(value) for value in projected):
                        values.extend((1e4, 1e4))
                    else:
                        values.extend(((projected[0] - item.u) / sigma, (projected[1] - item.v) / sigma))
                bounce_pixel = _interpolated_pixel(obs, state["bounce_time_s"])
                ray_origin, ray_direction = pixel_ray_world(camera, bounce_pixel)
                ray_distance = _distance_point_to_ray(state["bounce_world_xyz"], ray_origin, ray_direction)
                values.append(ray_distance / cfg.ray_constraint_sigma_m)
                net_clearance = _net_clearance(state, obs[0].pts_s, obs[-1].pts_s, plane.normal, cfg, net)
                if net_clearance is not None:
                    violation = max(0.0, -cfg.net_clearance_slack_m - net_clearance)
                    values.append(violation / cfg.net_constraint_sigma_m)
                else:
                    values.append(0.0)
                return np.asarray(values, dtype=float)

            result = least_squares(
                residuals,
                initial,
                bounds=(lower, upper),
                loss="huber",
                f_scale=cfg.robust_f_scale,
                max_nfev=cfg.max_nfev,
            )
            params = result.x if bool(np.all(np.isfinite(result.x))) else initial
            candidate = _evaluate_candidate(
                params,
                obs,
                camera,
                plane,
                plane_u,
                plane_v,
                net,
                cfg,
                seed_time=seed_time,
                start_index=start_index,
                optimizer_success=bool(result.success),
                optimizer_status=int(result.status),
                optimizer_nfev=int(result.nfev),
                optimizer_message=str(result.message),
            )
            if math.isfinite(candidate.cost):
                raw.append(candidate)

    if not raw:
        raise AnchorSearchRefusal(
            RefusalCode.OPTIMIZATION_FAILED,
            "no finite anchor hypothesis survived optimization",
            details={"enumeration_count": len(seed_times), "attempt_count": attempts},
        )

    selected = _dedupe_candidates(sorted(raw, key=_raw_candidate_sort_key), cfg)
    selected = selected[: cfg.max_ranked_candidates]
    candidates = tuple(
        _materialize_candidate(
            item,
            rank=index + 1,
            observations=obs,
            observation_digest=obs_digest,
            camera_digest=camera_digest,
            court_plane=plane,
            net=net,
            config=cfg,
            seed=seed,
        )
        for index, item in enumerate(selected)
    )
    return AnchorSearchResult(
        candidates=candidates,
        observation_digest_sha256=obs_digest,
        camera_digest_sha256=camera_digest,
        seed=int(seed),
        enumeration_count=len(seed_times),
        optimization_attempt_count=attempts,
        config=cfg,
    )


def pixel_ray_world(
    camera: CameraModel,
    xy: Sequence[float],
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Return a world-space camera ray without importing solver internals."""

    camera_ray = ((float(xy[0]) - camera.cx) / camera.fx, (float(xy[1]) - camera.cy) / camera.fy, 1.0)
    rotation_t = _transpose3(camera.rotation_world_to_camera)
    origin = _mat_vec(rotation_t, _scale(camera.translation_world_to_camera, -1.0))
    return origin, _normalized_vec(_mat_vec(rotation_t, camera_ray), RefusalCode.INVALID_CAMERA, "camera ray")


def intersect_ray_plane(
    origin: tuple[float, float, float],
    direction: tuple[float, float, float],
    plane_point: tuple[float, float, float],
    plane_normal: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Intersect a forward camera ray with an arbitrary plane."""

    normal = _normalized_vec(plane_normal, RefusalCode.DEGENERATE_COURT_PLANE, "court plane normal")
    denom = _dot(direction, normal)
    if abs(denom) < 1e-10:
        raise AnchorSearchRefusal(RefusalCode.DEGENERATE_RAY_PLANE, "camera ray is parallel to court plane")
    distance = _dot(_sub(plane_point, origin), normal) / denom
    if not math.isfinite(distance) or distance <= 0.0:
        raise AnchorSearchRefusal(
            RefusalCode.DEGENERATE_RAY_PLANE,
            "court-plane intersection is not in front of the camera",
        )
    return _add(origin, _scale(direction, distance))


def project_world(camera: CameraModel, point: Sequence[float]) -> tuple[tuple[float, float], float]:
    camera_point = _add(_mat_vec(camera.rotation_world_to_camera, point), camera.translation_world_to_camera)
    depth = float(camera_point[2])
    safe_depth = depth if abs(depth) > 1e-12 else math.copysign(1e-12, depth or 1.0)
    return (
        (camera.fx * camera_point[0] / safe_depth + camera.cx, camera.fy * camera_point[1] / safe_depth + camera.cy),
        depth,
    )


def _validate_observations(obs: Sequence[ImageObservation], cfg: JointAnchorSearchConfig) -> None:
    if len(obs) < cfg.min_observations:
        raise AnchorSearchRefusal(
            RefusalCode.INSUFFICIENT_OBSERVATIONS,
            f"need at least {cfg.min_observations} observations, got {len(obs)}",
            details={"required": cfg.min_observations, "actual": len(obs)},
        )
    for index, item in enumerate(obs):
        values = (item.pts_s, item.u, item.v, item.confidence)
        if item.frame < 0 or not all(math.isfinite(float(value)) for value in values):
            raise AnchorSearchRefusal(
                RefusalCode.INVALID_OBSERVATION,
                f"observation {index} contains non-finite or negative-frame data",
            )
        if item.confidence <= 0.0 or item.confidence > 1.0:
            raise AnchorSearchRefusal(
                RefusalCode.INVALID_OBSERVATION,
                f"observation {index} confidence must be in (0, 1]",
            )
    times = [item.pts_s for item in obs]
    if any(right <= left for left, right in zip(times, times[1:])):
        raise AnchorSearchRefusal(
            RefusalCode.INSUFFICIENT_TEMPORAL_SUPPORT,
            "observation PTS values must be strictly increasing",
        )
    if times[-1] - times[0] < cfg.min_time_span_s:
        raise AnchorSearchRefusal(
            RefusalCode.INSUFFICIENT_TEMPORAL_SUPPORT,
            "observation time span is below the configured minimum",
            details={"span_s": times[-1] - times[0], "required_s": cfg.min_time_span_s},
        )


def _validate_camera(camera: CameraModel) -> None:
    if camera.fx <= 0.0 or camera.fy <= 0.0:
        raise AnchorSearchRefusal(RefusalCode.INVALID_CAMERA, "camera focal lengths must be positive")
    if len(camera.rotation_world_to_camera) != 3 or any(len(row) != 3 for row in camera.rotation_world_to_camera):
        raise AnchorSearchRefusal(RefusalCode.INVALID_CAMERA, "camera rotation must be 3x3")
    if len(camera.translation_world_to_camera) != 3:
        raise AnchorSearchRefusal(RefusalCode.INVALID_CAMERA, "camera translation must have length 3")
    values = [camera.fx, camera.fy, camera.cx, camera.cy, *camera.translation_world_to_camera]
    values.extend(value for row in camera.rotation_world_to_camera for value in row)
    if not all(math.isfinite(float(value)) for value in values):
        raise AnchorSearchRefusal(RefusalCode.INVALID_CAMERA, "camera values must be finite")


def _normalized_plane(plane: BoundedPlane, *, ball_radius_m: float) -> BoundedPlane:
    normal = _normalized_vec(plane.normal, RefusalCode.DEGENERATE_COURT_PLANE, "court plane normal")
    if not all(math.isfinite(value) for value in (*plane.point, *plane.u_bounds_m, *plane.v_bounds_m)):
        raise AnchorSearchRefusal(RefusalCode.DEGENERATE_COURT_PLANE, "court plane values must be finite")
    if plane.u_bounds_m[0] >= plane.u_bounds_m[1] or plane.v_bounds_m[0] >= plane.v_bounds_m[1]:
        raise AnchorSearchRefusal(RefusalCode.DEGENERATE_COURT_PLANE, "court plane bounds must be increasing")
    # The supplied plane is the court surface.  The ball-center bounce anchor
    # is one radius above it, matching the existing solver's bounce contract.
    center_plane_point = _add(plane.point, _scale(normal, ball_radius_m))
    return BoundedPlane(center_plane_point, normal, plane.u_bounds_m, plane.v_bounds_m)


def _plane_basis(normal: Sequence[float]) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    reference = (0.0, 0.0, 1.0) if abs(normal[2]) < 0.9 else (1.0, 0.0, 0.0)
    u = _normalized_vec(_cross(reference, normal), RefusalCode.DEGENERATE_COURT_PLANE, "court basis u")
    v = _normalized_vec(_cross(normal, u), RefusalCode.DEGENERATE_COURT_PLANE, "court basis v")
    return u, v


def _enumerate_seed_times(obs: Sequence[ImageObservation], cfg: JointAnchorSearchConfig) -> list[float]:
    lo_index = cfg.min_observations_each_side - 1
    hi_index = len(obs) - cfg.min_observations_each_side
    values = [obs[index].pts_s for index in range(lo_index, hi_index + 1)]
    values.extend((obs[index].pts_s + obs[index + 1].pts_s) * 0.5 for index in range(lo_index, hi_index))
    values = sorted(set(round(value, 12) for value in values))
    if len(values) <= cfg.max_time_seeds:
        return values
    if cfg.max_time_seeds == 1:
        return [values[len(values) // 2]]
    indexes = [round(index * (len(values) - 1) / (cfg.max_time_seeds - 1)) for index in range(cfg.max_time_seeds)]
    return [values[index] for index in sorted(set(indexes))]


def _interpolated_pixel(obs: Sequence[ImageObservation], t: float) -> tuple[float, float]:
    if t <= obs[0].pts_s:
        return obs[0].xy
    if t >= obs[-1].pts_s:
        return obs[-1].xy
    for left, right in zip(obs, obs[1:]):
        if left.pts_s <= t <= right.pts_s:
            alpha = (t - left.pts_s) / max(right.pts_s - left.pts_s, 1e-12)
            return (left.u + alpha * (right.u - left.u), left.v + alpha * (right.v - left.v))
    return obs[-1].xy


def _tangent_velocity_guess(
    obs: Sequence[ImageObservation],
    camera: CameraModel,
    plane: BoundedPlane,
    basis_u: Sequence[float],
    basis_v: Sequence[float],
    seed_time: float,
) -> tuple[float, float, float]:
    projected: list[tuple[float, float, float]] = []
    for item in obs:
        try:
            origin, direction = pixel_ray_world(camera, item.xy)
            point = intersect_ray_plane(origin, direction, plane.point, plane.normal)
        except AnchorSearchRefusal:
            continue
        projected.append((item.pts_s, _dot(_sub(point, plane.point), basis_u), _dot(_sub(point, plane.point), basis_v)))
    local = sorted(projected, key=lambda item: abs(item[0] - seed_time))[: max(4, min(10, len(projected)))]
    if len(local) < 2:
        return (0.0, 0.0, -3.0)
    t_mean = sum(item[0] for item in local) / len(local)
    denom = sum((item[0] - t_mean) ** 2 for item in local)
    if denom <= 1e-12:
        return (0.0, 0.0, -3.0)
    vu = sum((item[0] - t_mean) * item[1] for item in local) / denom
    vv = sum((item[0] - t_mean) * item[2] for item in local) / denom
    return (vu, vv, -3.0)


def _state_from_params(
    params: Sequence[float],
    plane: BoundedPlane,
    basis_u: Sequence[float],
    basis_v: Sequence[float],
    cfg: JointAnchorSearchConfig,
) -> dict[str, Any]:
    bounce = _add(plane.point, _add(_scale(basis_u, float(params[1])), _scale(basis_v, float(params[2]))))
    incoming = _add(
        _add(_scale(basis_u, float(params[3])), _scale(basis_v, float(params[4]))),
        _scale(plane.normal, float(params[5])),
    )
    restitution = float(params[6])
    retention = float(params[7])
    tangent = _sub(incoming, _scale(plane.normal, _dot(incoming, plane.normal)))
    normal_velocity = _scale(plane.normal, _dot(incoming, plane.normal))
    outgoing = _sub(_scale(tangent, retention), _scale(normal_velocity, restitution))
    return {
        "bounce_time_s": float(params[0]),
        "bounce_world_xyz": bounce,
        "incoming_velocity_mps": incoming,
        "outgoing_velocity_mps": outgoing,
        "restitution": restitution,
        "retention": retention,
        "gravity": _scale(plane.normal, -cfg.gravity_mps2),
    }


def _piecewise_position(t: float, state: Mapping[str, Any], normal: Sequence[float], gravity: float) -> tuple[float, float, float]:
    dt = float(t) - float(state["bounce_time_s"])
    velocity = state["incoming_velocity_mps"] if dt <= 0.0 else state["outgoing_velocity_mps"]
    acceleration = _scale(normal, -gravity)
    return _add(state["bounce_world_xyz"], _add(_scale(velocity, dt), _scale(acceleration, 0.5 * dt * dt)))


def _net_clearance(
    state: Mapping[str, Any],
    t0: float,
    t1: float,
    court_normal: Sequence[float],
    cfg: JointAnchorSearchConfig,
    net: NetConstraint | None,
) -> float | None:
    if net is None:
        return None
    normal = _normalized_vec(net.normal, RefusalCode.DEGENERATE_COURT_PLANE, "net normal")
    samples = [t0 + (t1 - t0) * index / 80.0 for index in range(81)]
    points = [_piecewise_position(t, state, court_normal, cfg.gravity_mps2) for t in samples]
    signed = [_dot(_sub(point, net.point), normal) for point in points]
    crossings: list[float] = []
    for index, (left, right) in enumerate(zip(signed, signed[1:])):
        if left * right > 0.0:
            continue
        denom = abs(left) + abs(right)
        alpha = 0.0 if denom <= 1e-12 else abs(left) / denom
        point = _add(points[index], _scale(_sub(points[index + 1], points[index]), alpha))
        height = _dot(_sub(point, net.point), court_normal)
        crossings.append(height - (net.height_m + cfg.ball_radius_m))
    return min(crossings) if crossings else None


def _evaluate_candidate(
    params: Sequence[float],
    obs: Sequence[ImageObservation],
    camera: CameraModel,
    plane: BoundedPlane,
    basis_u: Sequence[float],
    basis_v: Sequence[float],
    net: NetConstraint | None,
    cfg: JointAnchorSearchConfig,
    *,
    seed_time: float,
    start_index: int,
    optimizer_success: bool,
    optimizer_status: int,
    optimizer_nfev: int,
    optimizer_message: str,
) -> _RawCandidate:
    state = _state_from_params(params, plane, basis_u, basis_v, cfg)
    errors: list[float] = []
    robust_cost = 0.0
    for item in obs:
        point = _piecewise_position(item.pts_s, state, plane.normal, cfg.gravity_mps2)
        pixel, depth = project_world(camera, point)
        error = math.hypot(pixel[0] - item.u, pixel[1] - item.v) if depth > 1e-6 else math.inf
        errors.append(error)
        sigma = cfg.robust_pixel_sigma / math.sqrt(max(0.05, min(1.0, item.confidence)))
        robust_cost += _huber(error / sigma, cfg.robust_f_scale)
    bounce_pixel = _interpolated_pixel(obs, state["bounce_time_s"])
    origin, direction = pixel_ray_world(camera, bounce_pixel)
    ray_distance = _distance_point_to_ray(state["bounce_world_xyz"], origin, direction)
    net_clearance = _net_clearance(state, obs[0].pts_s, obs[-1].pts_s, plane.normal, cfg, net)
    net_penalty = 0.0 if net_clearance is None else _huber(
        max(0.0, -cfg.net_clearance_slack_m - net_clearance) / cfg.net_constraint_sigma_m,
        cfg.robust_f_scale,
    )
    constraint_cost = _huber(ray_distance / cfg.ray_constraint_sigma_m, cfg.robust_f_scale) + net_penalty
    finite_errors = [value for value in errors if math.isfinite(value)]
    rmse = math.sqrt(sum(value * value for value in finite_errors) / len(finite_errors)) if len(finite_errors) == len(errors) else math.inf
    return _RawCandidate(
        cost=robust_cost + constraint_cost,
        bounce_time_s=state["bounce_time_s"],
        bounce_world_xyz=state["bounce_world_xyz"],
        incoming_velocity_mps=state["incoming_velocity_mps"],
        outgoing_velocity_mps=state["outgoing_velocity_mps"],
        restitution=state["restitution"],
        retention=state["retention"],
        reprojection_rmse_px=rmse,
        reprojection_huber_cost=robust_cost,
        ray_plane_distance_m=ray_distance,
        net_clearance_m=net_clearance,
        net_ok=None if net_clearance is None else net_clearance >= -cfg.net_clearance_slack_m,
        court_normal=plane.normal,
        seed_time_s=seed_time,
        start_index=start_index,
        optimizer_success=optimizer_success,
        optimizer_status=optimizer_status,
        optimizer_nfev=optimizer_nfev,
        optimizer_message=optimizer_message,
    )


def _dedupe_candidates(raw: Sequence[_RawCandidate], cfg: JointAnchorSearchConfig) -> list[_RawCandidate]:
    selected: list[_RawCandidate] = []
    for candidate in raw:
        if any(
            abs(candidate.bounce_time_s - existing.bounce_time_s) <= cfg.dedupe_time_s
            and _distance(candidate.bounce_world_xyz, existing.bounce_world_xyz) <= cfg.dedupe_position_m
            for existing in selected
        ):
            continue
        selected.append(candidate)
    return selected


def _materialize_candidate(
    raw: _RawCandidate,
    *,
    rank: int,
    observations: Sequence[ImageObservation],
    observation_digest: str,
    camera_digest: str,
    court_plane: BoundedPlane,
    net: NetConstraint | None,
    config: JointAnchorSearchConfig,
    seed: int,
) -> AnchorCandidate:
    candidate_id = f"tt3d_anchor_candidate_{rank:03d}"
    bounce_frame = min(observations, key=lambda item: (abs(item.pts_s - raw.bounce_time_s), item.frame)).frame
    start_position = _ballistic_point_from_raw(observations[0].pts_s, raw, config)
    end_position = _ballistic_point_from_raw(observations[-1].pts_s, raw, config)
    anchors = (
        _solver_anchor_json(
            anchor_id=f"{candidate_id}_boundary_start",
            kind="contact",
            t=observations[0].pts_s,
            frame=observations[0].frame,
            world_xyz=start_position,
            sigma_m=config.candidate_sigma_m,
            boundary_role="search_window_start_hypothesis",
        ),
        _solver_anchor_json(
            anchor_id=f"{candidate_id}_bounce",
            kind="bounce",
            t=raw.bounce_time_s,
            frame=bounce_frame,
            world_xyz=raw.bounce_world_xyz,
            sigma_m=config.candidate_sigma_m,
            boundary_role="jointly_optimized_court_plane_bounce",
        ),
        _solver_anchor_json(
            anchor_id=f"{candidate_id}_boundary_end",
            kind="contact",
            t=observations[-1].pts_s,
            frame=observations[-1].frame,
            world_xyz=end_position,
            sigma_m=config.candidate_sigma_m,
            boundary_role="search_window_end_hypothesis",
        ),
    )
    provenance = {
        "search_method": SEARCH_METHOD,
        "hypothesis_only": True,
        "seed": int(seed),
        "time_seed_s": _round(raw.seed_time_s, 12),
        "start_index": raw.start_index,
        "observation_digest_sha256": observation_digest,
        "camera_digest_sha256": camera_digest,
        "geometry": {
            "ball_center_court_plane": {
                "point": _vec_json(court_plane.point),
                "normal": _vec_json(court_plane.normal),
                "u_bounds_m": list(court_plane.u_bounds_m),
                "v_bounds_m": list(court_plane.v_bounds_m),
            },
            "net_constraint": None
            if net is None
            else {
                "point": _vec_json(net.point),
                "normal": _vec_json(net.normal),
                "height_m": _round(net.height_m, 9),
            },
        },
        "observation_refs": [
            {
                "frame": item.frame,
                "pts_s": _round(item.pts_s, 12),
                "u": _round(item.u, 9),
                "v": _round(item.v, 9),
                "confidence": _round(item.confidence, 9),
            }
            for item in observations
        ],
        "optimizer": {
            "name": "scipy.optimize.least_squares",
            "method": "trf",
            "loss": "huber",
            "success": raw.optimizer_success,
            "status": raw.optimizer_status,
            "nfev": raw.optimizer_nfev,
            "message": raw.optimizer_message,
        },
        "constraints": {
            "bounce_on_court_plane": True,
            "bounce_ray_constraint": True,
            "net_clearance_constraint_evaluated": raw.net_clearance_m is not None,
            "incoming_normal_velocity_negative": True,
            "bounded_speed": True,
        },
        "config": _jsonable_config(config),
        "policy": {
            "candidate_only": True,
            "marks_measured": False,
            "touches_trust_bands": False,
            "alters_defaults": False,
        },
    }
    return AnchorCandidate(
        candidate_id=candidate_id,
        rank=rank,
        cost=raw.cost,
        bounce_time_s=raw.bounce_time_s,
        bounce_frame=bounce_frame,
        bounce_world_xyz=raw.bounce_world_xyz,
        incoming_velocity_mps=raw.incoming_velocity_mps,
        outgoing_velocity_mps=raw.outgoing_velocity_mps,
        restitution=raw.restitution,
        tangential_retention=raw.retention,
        reprojection_rmse_px=raw.reprojection_rmse_px,
        reprojection_huber_cost=raw.reprojection_huber_cost,
        ray_plane_distance_m=raw.ray_plane_distance_m,
        net_clearance_m=raw.net_clearance_m,
        net_constraint_satisfied=raw.net_ok,
        anchor_events=anchors,
        provenance=provenance,
    )


def _ballistic_point_from_raw(t: float, raw: _RawCandidate, cfg: JointAnchorSearchConfig) -> tuple[float, float, float]:
    dt = t - raw.bounce_time_s
    velocity = raw.incoming_velocity_mps if dt <= 0.0 else raw.outgoing_velocity_mps
    acceleration = _scale(raw.court_normal, -cfg.gravity_mps2)
    return _add(raw.bounce_world_xyz, _add(_scale(velocity, dt), _scale(acceleration, 0.5 * dt * dt)))


def _solver_anchor_json(
    *,
    anchor_id: str,
    kind: str,
    t: float,
    frame: int,
    world_xyz: Sequence[float],
    sigma_m: float,
    boundary_role: str,
) -> dict[str, Any]:
    return {
        "anchor_id": anchor_id,
        "kind": kind,
        "t": _round(t, 9),
        "frame": int(frame),
        "world_xyz": _vec_json(world_xyz),
        "sigma_m": _round(sigma_m, 6),
        "status": "candidate_hypothesis",
        "immovable": False,
        "source": SEARCH_METHOD,
        "details": {
            "hypothesis_only": True,
            "boundary_role": boundary_role,
            "measured": False,
        },
    }


def _raw_candidate_sort_key(item: _RawCandidate) -> tuple[float, float, float, float, int]:
    return (item.cost, item.reprojection_rmse_px, item.bounce_time_s, item.seed_time_s, item.start_index)


def _huber(value: float, delta: float) -> float:
    absolute = abs(float(value))
    if absolute <= delta:
        return 0.5 * absolute * absolute
    return delta * (absolute - 0.5 * delta)


def _distance_point_to_ray(point: Sequence[float], origin: Sequence[float], direction: Sequence[float]) -> float:
    offset = _sub(point, origin)
    projection = _dot(offset, direction)
    return _norm(_sub(offset, _scale(direction, projection)))


def _digest(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _jsonable_config(config: JointAnchorSearchConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["restitution_bounds"] = list(config.restitution_bounds)
    payload["tangential_retention_bounds"] = list(config.tangential_retention_bounds)
    return payload


def _vec3(value: Any, name: str) -> tuple[float, float, float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 3:
        raise ValueError(f"{name} must be a length-3 sequence")
    parsed = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in parsed):
        raise ValueError(f"{name} must be finite")
    return parsed  # type: ignore[return-value]


def _normalized_vec(value: Sequence[float], code: RefusalCode, name: str) -> tuple[float, float, float]:
    norm = _norm(value)
    if not math.isfinite(norm) or norm <= 1e-12:
        raise AnchorSearchRefusal(code, f"{name} is degenerate")
    return _scale(value, 1.0 / norm)


def _transpose3(matrix: Sequence[Sequence[float]]) -> tuple[tuple[float, float, float], ...]:
    return tuple(tuple(float(matrix[row][column]) for row in range(3)) for column in range(3))


def _mat_vec(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> tuple[float, float, float]:
    return tuple(sum(float(matrix[row][column]) * float(vector[column]) for column in range(3)) for row in range(3))  # type: ignore[return-value]


def _add(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float]:
    return (float(a[0]) + float(b[0]), float(a[1]) + float(b[1]), float(a[2]) + float(b[2]))


def _sub(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float]:
    return (float(a[0]) - float(b[0]), float(a[1]) - float(b[1]), float(a[2]) - float(b[2]))


def _scale(value: Sequence[float], factor: float) -> tuple[float, float, float]:
    return (float(value[0]) * factor, float(value[1]) * factor, float(value[2]) * factor)


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(float(left) * float(right) for left, right in zip(a, b))


def _cross(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float]:
    return (
        float(a[1]) * float(b[2]) - float(a[2]) * float(b[1]),
        float(a[2]) * float(b[0]) - float(a[0]) * float(b[2]),
        float(a[0]) * float(b[1]) - float(a[1]) * float(b[0]),
    )


def _norm(value: Sequence[float]) -> float:
    return math.sqrt(_dot(value, value))


def _distance(a: Sequence[float], b: Sequence[float]) -> float:
    return _norm(_sub(a, b))


def _round(value: float, digits: int) -> float:
    return round(float(value), digits)


def _vec_json(value: Sequence[float]) -> list[float]:
    return [_round(float(item), 9) for item in value]

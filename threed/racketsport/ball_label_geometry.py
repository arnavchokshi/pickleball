"""Camera geometry for the human ball-labelling studio.

A single camera cannot see depth. A clicked ball pixel fixes a **ray**, not a
point: two of the three degrees of freedom. This module owns the one
remaining degree of freedom -- depth along that ray -- and every conversion
around it. All camera math lives here in Python so it is testable; the
browser never re-derives a projection, it only adds and scales the vectors
this module hands it.

The primitives are reused verbatim from the production arc solver
(``ball_arc_solver.pixel_ray_world`` / ``intersect_ray_z`` /
``_project_world_point`` / ``anchor_sigma_for_bounce``) so a label is
expressed in exactly the coordinates the solver would have produced for the
same pixel. That is the point: the labels have to be directly comparable to
the solver's output, otherwise a correction is not a measurement of our error.

Parameterisation, fixed once here and asserted in tests:

    world(t) = ray_origin + t * ray_direction_unit,  t in metres, t > 0

``ray_direction_unit`` points away from the camera into the scene, so ``t``
is the straight-line distance from the camera centre to the ball and larger
``t`` always means "further away". ``t <= 0`` is behind the camera and is
rejected rather than clamped.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .ball_arc_solver import (
    BALL_RADIUS_M,
    DEFAULT_BOUNCE_SPEED_CV,
    _project_world_point,
    anchor_uncertainty_for_bounce,
    calibration_plane_residuals,
    intersect_ray_z,
    pixel_ray_world,
    sigma_xyz_from_ray,
)

Vec3 = tuple[float, float, float]

# Default human-judgement sigmas along the ray. These are honest priors on how
# well a person can resolve depth, NOT measurements.
#
# near_player: the owner is judging the ball against a skeleton whose 3D
#   position we already know. Half a metre is roughly "within an arm's length
#   of the reference joint, in depth" -- deliberately conservative.
# free_flight: the owner has no depth reference at all. Over a ~10 m deep
#   pickleball court a person cannot do better than "somewhere in that third
#   of the court"; 2 m is an honest number for that, and it is large enough
#   that no downstream consumer will mistake it for a measurement.
NEAR_PLAYER_DEPTH_SIGMA_M = 0.5
FREE_FLIGHT_DEPTH_SIGMA_M = 2.0

# Click precision on a small, fast, motion-blurred ball, in pixels. The
# magnifier in the UI is what keeps this small; without it, assume worse.
DEFAULT_CLICK_SIGMA_PX = 2.0

# A near_player label is only meaningful if a tracked joint is actually close
# to the ray. Beyond this the "reference" is not a reference, and the kind is
# refused so the owner is forced to record it as the free-flight estimate it is.
NEAR_PLAYER_MAX_OFFSET_M = 2.5

# Depth-slider travel. The camera sits behind a baseline ~10 m from the net,
# so a plausible ball is a few metres to ~30 m away.
MIN_DEPTH_M = 0.5
MAX_DEPTH_M = 40.0

GRAVITY_MPS2 = 9.80665

# Fallback pickleball physics, overridden per-run from
# ball_track_arc_solved.json:physics_parameters when that artifact is present.
DEFAULT_PHYSICS = {
    "mass_kg": 0.0255,
    "radius_m": BALL_RADIUS_M,
    "drag_cd": 0.33,
    "rho_air_kg_m3": 1.2,
    "gravity_mps2": GRAVITY_MPS2,
}


class GeometryError(ValueError):
    """Raised when a pixel or depth cannot produce a usable 3D position."""


@dataclass(frozen=True)
class PixelRay:
    """The 2-of-3 degrees of freedom a click fixes."""

    origin: Vec3
    direction: Vec3
    pixel_xy: tuple[float, float]

    def at_depth(self, depth_m: float) -> Vec3:
        """World point at ``depth_m`` metres along the ray. The one DOF."""

        depth = float(depth_m)
        if not math.isfinite(depth):
            raise GeometryError(f"depth must be finite, got {depth_m!r}")
        if depth <= 0.0:
            raise GeometryError(
                f"depth must be > 0 (in front of the camera), got {depth_m!r}"
            )
        return (
            self.origin[0] + depth * self.direction[0],
            self.origin[1] + depth * self.direction[1],
            self.origin[2] + depth * self.direction[2],
        )

    def depth_of(self, world_xyz: Sequence[float]) -> float:
        """Depth of the point on the ray closest to ``world_xyz``.

        This is the projection that turns an existing 3D position (a pipeline
        prefill, or a previous label) back into the single depth degree of
        freedom, so the owner corrects one number instead of three.
        """

        offset = (
            float(world_xyz[0]) - self.origin[0],
            float(world_xyz[1]) - self.origin[1],
            float(world_xyz[2]) - self.origin[2],
        )
        return _dot(offset, self.direction)

    def offset_from(self, world_xyz: Sequence[float]) -> float:
        """Perpendicular distance from ``world_xyz`` to the ray, in metres."""

        depth = self.depth_of(world_xyz)
        closest = (
            self.origin[0] + depth * self.direction[0],
            self.origin[1] + depth * self.direction[1],
            self.origin[2] + depth * self.direction[2],
        )
        return math.dist(closest, (float(world_xyz[0]), float(world_xyz[1]), float(world_xyz[2])))

    def depth_at_height(self, z_m: float) -> float:
        """Depth where the ray crosses the horizontal plane ``z = z_m``.

        This is the bounce case: the ball is resting on the court, so its
        height is known and depth stops being a free parameter at all.
        """

        if abs(self.direction[2]) < 1e-9:
            raise GeometryError("ray is parallel to the court plane; height cannot fix depth")
        depth = (float(z_m) - self.origin[2]) / self.direction[2]
        if depth <= 0.0:
            raise GeometryError(
                f"the ray crosses z={z_m} m behind the camera (depth {depth:.3f} m); "
                f"this pixel cannot be a bounce"
            )
        return depth

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "origin_m": [round(v, 6) for v in self.origin],
            "direction_unit": [round(v, 9) for v in self.direction],
            "pixel_xy": [round(v, 3) for v in self.pixel_xy],
        }


def ray_for_pixel(calibration: Mapping[str, Any], pixel_xy: Sequence[float]) -> PixelRay:
    """Clicked pixel -> world ray, via the production solver's own primitive."""

    origin, direction = pixel_ray_world(calibration, pixel_xy)
    return PixelRay(
        origin=(float(origin[0]), float(origin[1]), float(origin[2])),
        direction=(float(direction[0]), float(direction[1]), float(direction[2])),
        pixel_xy=(float(pixel_xy[0]), float(pixel_xy[1])),
    )


def bounce_world_point(
    calibration: Mapping[str, Any], pixel_xy: Sequence[float]
) -> tuple[Vec3, float]:
    """Solve a bounce: ray-plane intersection at one ball radius above the court.

    Returns ``(world_xyz, depth_m)``. No human depth input is involved, which
    is exactly why these are the highest-value labels.
    """

    ray = ray_for_pixel(calibration, pixel_xy)
    depth = ray.depth_at_height(BALL_RADIUS_M)
    origin, direction = pixel_ray_world(calibration, pixel_xy)
    world = intersect_ray_z(origin, direction, BALL_RADIUS_M)
    return (float(world[0]), float(world[1]), float(world[2])), depth


def project_world_to_pixel(
    calibration: Mapping[str, Any], world_xyz: Sequence[float]
) -> tuple[float, float]:
    """World point -> image pixel, using the production solver's projection."""

    point = (float(world_xyz[0]), float(world_xyz[1]), float(world_xyz[2]))
    u, v = _project_world_point(calibration, point)
    return (float(u), float(v))


def is_in_front_of_camera(calibration: Mapping[str, Any], world_xyz: Sequence[float]) -> bool:
    """True when the point is on the visible side of the image plane."""

    extrinsics = calibration.get("extrinsics")
    if not isinstance(extrinsics, Mapping):
        raise GeometryError("calibration.extrinsics is required")
    rotation = extrinsics["R"]
    translation = extrinsics["t"]
    depth = (
        float(rotation[2][0]) * float(world_xyz[0])
        + float(rotation[2][1]) * float(world_xyz[1])
        + float(rotation[2][2]) * float(world_xyz[2])
        + float(translation[2])
    )
    return depth > 0.0


# ---------------------------------------------------------------------------
# Uncertainty
# ---------------------------------------------------------------------------


def perpendicular_sigma_m(
    calibration: Mapping[str, Any], depth_m: float, *, click_sigma_px: float = DEFAULT_CLICK_SIGMA_PX
) -> float:
    """Across-ray sigma from click precision: a pixel error subtends metres.

    At depth ``d`` with focal length ``f`` px, an error of ``s`` px is
    ``d * s / f`` metres across the line of sight. This is the *good*
    direction: at a 10 m depth with a 2 px click error it is under 2 cm.
    """

    intrinsics = calibration.get("intrinsics")
    if not isinstance(intrinsics, Mapping):
        raise GeometryError("calibration.intrinsics is required")
    focal = 0.5 * (float(intrinsics["fx"]) + float(intrinsics["fy"]))
    if focal <= 0.0:
        raise GeometryError("calibration.intrinsics focal length must be positive")
    return max(1e-4, abs(float(depth_m)) * float(click_sigma_px) / focal)


def bounce_depth_sigma_m(
    calibration: Mapping[str, Any],
    pixel_xy: Sequence[float],
    *,
    click_sigma_px: float = DEFAULT_CLICK_SIGMA_PX,
    plane_residual_m: float | None = None,
    fps: float | None = None,
    vertical_speed_mps: float | None = None,
    horizontal_speed_mps: float | None = None,
) -> tuple[float, str]:
    """Along-ray sigma for a bounce, and a plain-English basis string.

    This now delegates to the production solver's
    :func:`~.ball_arc_solver.anchor_uncertainty_for_bounce`, which owns the one
    ray-aligned uncertainty model shared by the labeller and the solver.  The
    terms it accounts for are click sensitivity (a grazing ray near the horizon
    really is badly conditioned), the sub-frame timing overshoot, and this
    clip's measured calibration plane residual -- previously this function
    combined a *separately* computed click term with a solver sigma that
    double-counted part of it and omitted the calibration floor.
    """

    ray = ray_for_pixel(calibration, pixel_xy)
    ray.depth_at_height(BALL_RADIUS_M)  # raises GeometryError past the horizon

    kwargs: dict[str, Any] = {}
    if vertical_speed_mps is not None:
        kwargs["vertical_speed_mps"] = float(vertical_speed_mps)
    if horizontal_speed_mps is not None:
        kwargs["horizontal_speed_mps"] = float(horizontal_speed_mps)
    uncertainty = anchor_uncertainty_for_bounce(
        calibration,
        pixel_xy,
        base_sigma_m=0.05,
        ball_radius_m=BALL_RADIUS_M,
        pixel_sigma_px=float(click_sigma_px),
        fps=fps,
        calibration_residual_m=plane_residual_m,
        **kwargs,
    )
    if uncertainty is None:
        raise GeometryError(
            "this pixel has no usable ray-plane geometry; it cannot be a bounce"
        )
    floor_term = float(plane_residual_m or 0.0)
    sigma = max(uncertainty.sigma_along_ray_m, floor_term, 0.01)
    basis = (
        f"{uncertainty.basis} Systematic offset {uncertainty.bias_along_ray_m:+.3f} m along "
        f"the ray is reported separately as a bias, not folded into this sigma."
    )
    return sigma, basis


def near_player_depth_sigma_m(
    offset_from_ray_m: float,
    *,
    base_sigma_m: float = NEAR_PLAYER_DEPTH_SIGMA_M,
) -> tuple[float, str]:
    """Along-ray sigma when depth was judged against a tracked skeleton.

    The reference is only as good as its proximity to the ray: a joint sitting
    right under the ball is a strong depth anchor, one two metres off to the
    side is barely better than nothing. Sigma therefore grows linearly with
    the off-ray offset and is capped by the free-flight sigma, because a
    "reference" cannot make you worse off than having none.
    """

    offset = max(0.0, float(offset_from_ray_m))
    sigma = base_sigma_m * (1.0 + offset / max(1e-6, NEAR_PLAYER_MAX_OFFSET_M))
    sigma = min(sigma, FREE_FLIGHT_DEPTH_SIGMA_M)
    basis = (
        f"human depth judgement against a tracked player joint {offset:.2f} m off the ray; "
        f"prior {base_sigma_m:g} m widened by the off-ray offset. "
        f"This is a human estimate, not a measurement."
    )
    return sigma, basis


def free_flight_depth_sigma_m(
    *, base_sigma_m: float = FREE_FLIGHT_DEPTH_SIGMA_M
) -> tuple[float, str]:
    """Along-ray sigma with no depth reference at all: an honest guess."""

    basis = (
        f"unreferenced human depth estimate; prior {base_sigma_m:g} m along the camera ray. "
        f"No geometric or tracked-object constraint fixes depth here. "
        f"Do not treat as ground truth."
    )
    return float(base_sigma_m), basis


# `sigma_xyz_from_ray` and `calibration_plane_residuals` are imported from
# `ball_arc_solver` and re-exported here. They used to be duplicated in this
# module; the solver and the labeller must agree on the ray-aligned convention
# exactly, so there is now one definition and this module is a consumer of it.


def interpolation_extra_sigma_m(span_s: float, speed_mps: float, physics: Mapping[str, Any]) -> float:
    """Extra along-ray sigma from interpolating with a drag-free parabola.

    The interpolator fits ``p0 + v t + g t^2 / 2`` through two labels. Real
    pickleballs decelerate: at speed ``v`` the drag deceleration is
    ``rho Cd A v^2 / (2 m)``, and neglecting it over a span ``T`` displaces the
    arc by roughly ``a T^2 / 2``. That displacement is the honest cost of the
    convenience, so it is added to the sigma of anything accepted from it.
    """

    radius = float(physics.get("radius_m", BALL_RADIUS_M))
    area = math.pi * radius * radius
    drag_accel = (
        float(physics.get("rho_air_kg_m3", 1.2))
        * float(physics.get("drag_cd", 0.33))
        * area
        * float(speed_mps) ** 2
    ) / (2.0 * max(1e-6, float(physics.get("mass_kg", 0.0255))))
    return 0.5 * drag_accel * float(span_s) ** 2


# ---------------------------------------------------------------------------
# Ballistic interpolation between two labels
# ---------------------------------------------------------------------------


def ballistic_positions(
    start_xyz: Sequence[float],
    start_t: float,
    end_xyz: Sequence[float],
    end_t: float,
    sample_times: Sequence[float],
    *,
    gravity_mps2: float = GRAVITY_MPS2,
) -> list[Vec3]:
    """Positions on the unique drag-free parabola through two timed points.

    Solving ``p1 = p0 + v dt + g dt^2 / 2`` for ``v`` gives one arc through
    both endpoints under gravity alone -- the physically plausible shape the
    owner asked for between keyframes, rather than a straight line that would
    sag through the floor or float over the net.
    """

    span = float(end_t) - float(start_t)
    if span <= 0.0:
        raise GeometryError(f"interpolation needs end_t > start_t, got {start_t} -> {end_t}")
    gravity = (0.0, 0.0, -abs(float(gravity_mps2)))
    velocity = tuple(
        (float(end_xyz[axis]) - float(start_xyz[axis]) - 0.5 * gravity[axis] * span * span) / span
        for axis in range(3)
    )
    positions: list[Vec3] = []
    for sample in sample_times:
        dt = float(sample) - float(start_t)
        positions.append(
            (
                float(start_xyz[0]) + velocity[0] * dt + 0.5 * gravity[0] * dt * dt,
                float(start_xyz[1]) + velocity[1] * dt + 0.5 * gravity[1] * dt * dt,
                float(start_xyz[2]) + velocity[2] * dt + 0.5 * gravity[2] * dt * dt,
            )
        )
    return positions


def ballistic_speed_mps(
    start_xyz: Sequence[float],
    start_t: float,
    end_xyz: Sequence[float],
    end_t: float,
    *,
    gravity_mps2: float = GRAVITY_MPS2,
) -> float:
    """Mid-span speed of the fitted parabola, for the drag-neglect sigma."""

    span = float(end_t) - float(start_t)
    if span <= 0.0:
        raise GeometryError("interpolation needs end_t > start_t")
    mid = ballistic_positions(
        start_xyz, start_t, end_xyz, end_t, [start_t + span * 0.5], gravity_mps2=gravity_mps2
    )[0]
    before = ballistic_positions(
        start_xyz,
        start_t,
        end_xyz,
        end_t,
        [start_t + span * 0.5 - min(0.01, span * 0.1)],
        gravity_mps2=gravity_mps2,
    )[0]
    dt = min(0.01, span * 0.1)
    return math.dist(mid, before) / max(1e-6, dt)


# ---------------------------------------------------------------------------
# Small vector helpers (kept local so this module stays dependency-free)
# ---------------------------------------------------------------------------


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return float(a[0]) * float(b[0]) + float(a[1]) * float(b[1]) + float(a[2]) * float(b[2])


__all__ = [
    "BALL_RADIUS_M",
    "DEFAULT_CLICK_SIGMA_PX",
    "DEFAULT_PHYSICS",
    "FREE_FLIGHT_DEPTH_SIGMA_M",
    "GRAVITY_MPS2",
    "GeometryError",
    "MAX_DEPTH_M",
    "MIN_DEPTH_M",
    "NEAR_PLAYER_DEPTH_SIGMA_M",
    "NEAR_PLAYER_MAX_OFFSET_M",
    "PixelRay",
    "ballistic_positions",
    "ballistic_speed_mps",
    "bounce_depth_sigma_m",
    "bounce_world_point",
    "calibration_plane_residuals",
    "free_flight_depth_sigma_m",
    "interpolation_extra_sigma_m",
    "is_in_front_of_camera",
    "near_player_depth_sigma_m",
    "perpendicular_sigma_m",
    "project_world_to_pixel",
    "ray_for_pixel",
    "sigma_xyz_from_ray",
]

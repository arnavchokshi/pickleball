"""Depth-sensitive physical plausibility bounds for solved 3D ball positions.

Why this module exists
----------------------
Reprojection error is structurally blind to depth, so it cannot be used as a
3D quality gate. Measured on external ground truth
(``runs/lanes/tt3d_external_validation_20260726/report.json``):

- Sliding a solved ball 1.00 m along its own camera ray moves its reprojection
  by ~1e-13 px -- zero to machine precision (``E0_mapping_gate``, the
  ``no_noise`` rows are pure round-trip mapping error).
- One view showed 0.323 px median reprojection -- sub-pixel, passing any gate
  in this repo -- while the same solutions carried 0.305 m median and 1.381 m
  p95 3D error.
- Of a 0.3118 m mean total 3D error, 0.3115 m lay along the depth axis and
  0.0047 m in the image plane: the error *is* depth, ~40x larger than the
  component reprojection can see.

That is geometry, not a tuning defect. A reprojection threshold therefore
cannot be evidence that a 3D position is good. This module supplies a
criterion that *can* see depth: absolute physical bounds on where a
pickleball can possibly be.

What this gate does and does not claim
--------------------------------------
This is a NECESSARY condition, never a sufficient one. Passing it does not
mean depth was validated -- it only means the position is not absurd. A ball
solved 3 m too deep along the ray but still over the court passes every bound
here. Callers that survive this gate must still describe their depth as
``depth_unvalidated``; see :data:`DEPTH_UNVALIDATED_NOTE`.

Two severities
--------------
``implausible`` (soft)
    Outside what a real rally produces. Demote the band, never promote, mark
    the depth unvalidated -- but keep the position, because calibration error
    alone can push a genuinely grounded ball a few centimetres underground and
    suppressing that would trade honest coverage for nothing.

``absurd`` (hard)
    Physically impossible for a pickleball on this court: metres underground,
    twenty metres in the air, tens of metres off the sideline. Suppress the
    position outright. There is no depth estimate to salvage here.

Measured motivation for the hard height ceiling: on
``runs/lanes/ballarc_anchorfusion_20260716/wolverine_no_soft_current/ball_track_arc_solved.json``
the solver emitted 79 frames between 15.1 m and 23.5 m of altitude, each
carrying ``sigma_m = 0.197`` -- a claimed 20 cm uncertainty on a ball twenty
metres in the air. Nothing in the pre-existing court-volume check had an upper
z bound, so every one of those frames passed it.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

from .court_templates import get_court_template


SCHEMA_VERSION = 1
POLICY = "ball_position_plausibility_v1"

DEPTH_UNVALIDATED_NOTE = (
    "Physical plausibility bounds are a necessary condition only. They cannot "
    "confirm depth: a position wrong by metres along the camera ray can sit "
    "entirely inside them. Treat depth as unvalidated."
)

#: Soft violations. Position is kept but must never be promoted.
IMPLAUSIBLE_VIOLATIONS = (
    "below_ground_plane",
    "above_plausible_apex",
    "outside_court_footprint",
)

#: Hard violations. Position is physically impossible and must be suppressed.
ABSURD_VIOLATIONS = (
    "far_below_ground_plane",
    "far_above_plausible_apex",
    "far_outside_court_footprint",
)


@dataclass(frozen=True)
class BallPlausibilityBounds:
    """Absolute bounds on where a solved pickleball position can be.

    Defaults are deliberately generous. The soft bounds match the arc solver's
    own physical-sanity scale (``BallArcSolverConfig.court_margin_m`` 4.0 m,
    ``max_plausible_apex_m`` 8.0 m) so this gate does not silently re-tighten
    anything the solver already reasons about; the hard bounds sit far outside
    any rally and exist only to catch output that is not a ball trajectory at
    all.
    """

    sport: str = "pickleball"
    court_margin_m: float = 4.0
    hard_court_margin_m: float = 10.0
    z_min_m: float = -0.10
    hard_z_min_m: float = -0.50
    z_max_m: float = 8.0
    hard_z_max_m: float = 15.0
    max_speed_mps: float = 35.0
    hard_max_speed_mps: float = 60.0

    def __post_init__(self) -> None:
        if self.court_margin_m < 0.0 or self.hard_court_margin_m < self.court_margin_m:
            raise ValueError("require 0 <= court_margin_m <= hard_court_margin_m")
        if self.hard_z_min_m > self.z_min_m:
            raise ValueError("require hard_z_min_m <= z_min_m")
        if self.hard_z_max_m < self.z_max_m:
            raise ValueError("require hard_z_max_m >= z_max_m")
        if self.hard_max_speed_mps < self.max_speed_mps:
            raise ValueError("require hard_max_speed_mps >= max_speed_mps")
        get_court_template(str(self.sport))  # type: ignore[arg-type]

    @classmethod
    def from_solver_config(cls, config: Any) -> "BallPlausibilityBounds":
        """Build bounds from a ``BallArcSolverConfig``-shaped object.

        Only the court footprint and the plausible-apex scale are inherited.
        The solver's ``court_z_min_m`` is *not* used as the soft floor: it is a
        fit-time slack, and this gate wants a stated ground plane.
        """

        kwargs: dict[str, Any] = {}
        sport = getattr(config, "court_sport", None)
        if sport:
            kwargs["sport"] = str(sport)
        margin = _finite(getattr(config, "court_margin_m", None))
        if margin is not None and margin >= 0.0:
            kwargs["court_margin_m"] = margin
            kwargs["hard_court_margin_m"] = max(margin, cls.hard_court_margin_m)
        apex = _finite(getattr(config, "max_plausible_apex_m", None))
        if apex is not None and apex > 0.0:
            kwargs["z_max_m"] = apex
            kwargs["hard_z_max_m"] = max(apex, cls.hard_z_max_m)
        speed = _finite(getattr(config, "max_plausible_speed_mps", None))
        if speed is not None and speed > 0.0:
            kwargs["max_speed_mps"] = speed
            kwargs["hard_max_speed_mps"] = max(speed, cls.hard_max_speed_mps)
        return cls(**kwargs)

    def footprint_bounds_m(self, *, hard: bool = False) -> tuple[float, float, float, float]:
        template = get_court_template(str(self.sport))  # type: ignore[arg-type]
        margin = float(self.hard_court_margin_m if hard else self.court_margin_m)
        half_width = template.width_m / 2.0
        half_length = template.length_m / 2.0
        return (
            -half_width - margin,
            half_width + margin,
            -half_length - margin,
            half_length + margin,
        )

    def to_json(self) -> dict[str, Any]:
        x_min, x_max, y_min, y_max = self.footprint_bounds_m()
        hx_min, hx_max, hy_min, hy_max = self.footprint_bounds_m(hard=True)
        return {
            "policy": POLICY,
            "sport": str(self.sport),
            "implausible_bounds_m": {
                "x_min": round(x_min, 6),
                "x_max": round(x_max, 6),
                "y_min": round(y_min, 6),
                "y_max": round(y_max, 6),
                "z_min": round(float(self.z_min_m), 6),
                "z_max": round(float(self.z_max_m), 6),
            },
            "absurd_bounds_m": {
                "x_min": round(hx_min, 6),
                "x_max": round(hx_max, 6),
                "y_min": round(hy_min, 6),
                "y_max": round(hy_max, 6),
                "z_min": round(float(self.hard_z_min_m), 6),
                "z_max": round(float(self.hard_z_max_m), 6),
            },
            "court_margin_m": round(float(self.court_margin_m), 6),
            "hard_court_margin_m": round(float(self.hard_court_margin_m), 6),
            "max_speed_mps": round(float(self.max_speed_mps), 6),
            "hard_max_speed_mps": round(float(self.hard_max_speed_mps), 6),
            "depth_unvalidated_note": DEPTH_UNVALIDATED_NOTE,
        }


DEFAULT_BOUNDS = BallPlausibilityBounds()


def position_violations(
    world_xyz: Sequence[float] | None,
    bounds: BallPlausibilityBounds | None = None,
) -> tuple[str, ...]:
    """Return sorted violation names for one 3D position (empty when plausible)."""

    return tuple(sorted(evaluate_position(world_xyz, bounds)["violations"]))


def is_plausible(
    world_xyz: Sequence[float] | None,
    bounds: BallPlausibilityBounds | None = None,
) -> bool:
    """True when the position breaks no bound at all."""

    return not position_violations(world_xyz, bounds)


def is_absurd(
    world_xyz: Sequence[float] | None,
    bounds: BallPlausibilityBounds | None = None,
) -> bool:
    """True when the position is physically impossible and must be suppressed."""

    return bool(evaluate_position(world_xyz, bounds)["absurd"])


def evaluate_position(
    world_xyz: Sequence[float] | None,
    bounds: BallPlausibilityBounds | None = None,
) -> dict[str, Any]:
    """Classify one 3D ball position against the physical bounds.

    A ``None`` / non-finite position is *not* a violation: absent output is
    already honest, and this gate never invents a position.
    """

    limits = bounds or DEFAULT_BOUNDS
    xyz = _vec3(world_xyz)
    if xyz is None:
        return {
            "evaluated": False,
            "violations": [],
            "absurd_violations": [],
            "absurd": False,
            "plausible": True,
            "max_overage_m": 0.0,
        }
    x, y, z = xyz
    x_min, x_max, y_min, y_max = limits.footprint_bounds_m()
    hx_min, hx_max, hy_min, hy_max = limits.footprint_bounds_m(hard=True)

    violations: list[str] = []
    absurd: list[str] = []
    overages: list[float] = []

    if z < float(limits.z_min_m):
        violations.append("below_ground_plane")
        overages.append(float(limits.z_min_m) - z)
        if z < float(limits.hard_z_min_m):
            absurd.append("far_below_ground_plane")
    if z > float(limits.z_max_m):
        violations.append("above_plausible_apex")
        overages.append(z - float(limits.z_max_m))
        if z > float(limits.hard_z_max_m):
            absurd.append("far_above_plausible_apex")

    footprint_overage = max(x_min - x, x - x_max, y_min - y, y - y_max, 0.0)
    if footprint_overage > 0.0:
        violations.append("outside_court_footprint")
        overages.append(footprint_overage)
        if max(hx_min - x, x - hx_max, hy_min - y, y - hy_max, 0.0) > 0.0:
            absurd.append("far_outside_court_footprint")

    return {
        "evaluated": True,
        "world_xyz": [round(x, 6), round(y, 6), round(z, 6)],
        "violations": sorted(violations),
        "absurd_violations": sorted(absurd),
        "absurd": bool(absurd),
        "plausible": not violations,
        "max_overage_m": round(max(overages) if overages else 0.0, 6),
    }


def speed_violations(
    speed_mps: float | None,
    bounds: BallPlausibilityBounds | None = None,
) -> tuple[str, ...]:
    """Classify a scalar speed against the plausible/absurd speed ceilings.

    Only the upper ceiling is enforced. A *slow* arc that is consistent with
    the pixels is depth-ambiguous, not junk, and suppressing it would trade
    honest coverage for nothing (the rationale already recorded for
    ``arc_segment_fail_closed_v1`` in ``virtual_world.py``).
    """

    limits = bounds or DEFAULT_BOUNDS
    speed = _finite(speed_mps)
    if speed is None:
        return ()
    if speed > float(limits.hard_max_speed_mps):
        return ("speed_absurd", "speed_above_plausible_range")
    if speed > float(limits.max_speed_mps):
        return ("speed_above_plausible_range",)
    return ()


def evaluate_ball_track_plausibility(
    frames: Sequence[Mapping[str, Any]],
    *,
    bounds: BallPlausibilityBounds | None = None,
    depth_sigma_m_by_frame: Mapping[int, float] | None = None,
    sigma_multiple: float = 3.0,
) -> dict[str, Any]:
    """Evaluate every frame of a solved ball track against the bounds.

    Unlike the segment-scoped checks in ``ball_flight_sanity``, this sweeps
    *every* frame that carries a ``world_xyz``, including bridge frames, weak
    tails and frames that belong to no evaluated segment. Those are exactly
    the frames the segment-scoped checks never reached.

    ``depth_sigma_m_by_frame`` is the seam for the anisotropic-uncertainty
    work landing separately. It is OFF by default and, when supplied, is
    purely descriptive: each frame report gains ``depth_sigma_m`` and
    ``overage_within_claimed_sigma`` (whether the bound overage is smaller
    than ``sigma_multiple`` times the claimed depth sigma). It never changes a
    verdict, so nothing here depends on that work landing.
    """

    limits = bounds or DEFAULT_BOUNDS
    reports: list[dict[str, Any]] = []
    violation_counts: dict[str, int] = {}
    evaluated = 0
    implausible = 0
    absurd = 0
    for index, raw in enumerate(frames):
        frame = raw if isinstance(raw, Mapping) else {}
        verdict = evaluate_position(frame.get("world_xyz"), limits)
        if not verdict["evaluated"]:
            continue
        evaluated += 1
        if verdict["violations"]:
            implausible += 1
        if verdict["absurd"]:
            absurd += 1
        for name in verdict["violations"]:
            violation_counts[name] = violation_counts.get(name, 0) + 1
        for name in verdict["absurd_violations"]:
            violation_counts[name] = violation_counts.get(name, 0) + 1
        report = {
            "frame": index,
            "violations": verdict["violations"],
            "absurd_violations": verdict["absurd_violations"],
            "absurd": verdict["absurd"],
            "max_overage_m": verdict["max_overage_m"],
        }
        if depth_sigma_m_by_frame is not None:
            sigma = _finite(depth_sigma_m_by_frame.get(index))
            report["depth_sigma_m"] = sigma
            report["overage_within_claimed_sigma"] = (
                None
                if sigma is None
                else bool(verdict["max_overage_m"] <= float(sigma_multiple) * sigma)
            )
        reports.append(report)

    return {
        "schema_version": SCHEMA_VERSION,
        "policy": POLICY,
        "bounds": limits.to_json(),
        "depth_unvalidated": True,
        "depth_unvalidated_reason": DEPTH_UNVALIDATED_NOTE,
        "summary": {
            "evaluated_frame_count": evaluated,
            "implausible_frame_count": implausible,
            "absurd_frame_count": absurd,
            "violation_counts": dict(sorted(violation_counts.items())),
        },
        "frames": [report for report in reports if report["violations"]],
    }


def segment_physical_sanity_violations(
    physical_sanity: Any,
    bounds: BallPlausibilityBounds | None = None,
) -> tuple[str, ...]:
    """Classify an arc segment's own ``physical_sanity`` block.

    Reads the apex height, court-volume verdict and initial speed the solver
    already recorded, and returns the depth-aware violation names this gate
    would raise. Used where a *segment*, not a frame, is the unit of
    acceptance.
    """

    limits = bounds or DEFAULT_BOUNDS
    if not isinstance(physical_sanity, Mapping):
        return ()
    found: list[str] = []
    apex = _finite(physical_sanity.get("apex_height_m"))
    if apex is not None:
        if apex > float(limits.hard_z_max_m):
            found.extend(("far_above_plausible_apex", "above_plausible_apex"))
        elif apex > float(limits.z_max_m):
            found.append("above_plausible_apex")
    court_volume = physical_sanity.get("court_volume")
    if isinstance(court_volume, Mapping) and court_volume.get("violation") is True:
        found.append("outside_court_footprint")
        overage = _finite(court_volume.get("max_overage_m"))
        extra = float(limits.hard_court_margin_m) - float(limits.court_margin_m)
        if overage is not None and overage > extra:
            found.append("far_outside_court_footprint")
    found.extend(speed_violations(physical_sanity.get("initial_speed_mps"), limits))
    return tuple(sorted(set(found)))


def _vec3(value: Any) -> tuple[float, float, float] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 3:
        return None
    try:
        xyz = (float(value[0]), float(value[1]), float(value[2]))
    except (TypeError, ValueError):
        return None
    return xyz if all(math.isfinite(component) for component in xyz) else None


def _finite(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


__all__ = [
    "ABSURD_VIOLATIONS",
    "DEFAULT_BOUNDS",
    "DEPTH_UNVALIDATED_NOTE",
    "IMPLAUSIBLE_VIOLATIONS",
    "POLICY",
    "SCHEMA_VERSION",
    "BallPlausibilityBounds",
    "evaluate_ball_track_plausibility",
    "evaluate_position",
    "is_absurd",
    "is_plausible",
    "position_violations",
    "segment_physical_sanity_violations",
    "speed_violations",
]

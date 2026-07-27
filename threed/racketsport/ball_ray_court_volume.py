"""Camera-ray / court-volume containment for emitted 2D ball detections.

Why this module exists
----------------------
The product owner, hand-labelling four clips, reported that "a lot of the auto
detects are from a background ball". That observation is **confirmed**
(``runs/lanes/background_ball_20260727/report.json``): on the frozen 167-row
judge every one of the 11 indoor hidden false positives is a real pickleball
that is simply not the ball in play, and on ``burlington`` the detector tracks
a game on the *adjacent court* for tens of consecutive frames.

The obvious geometric defence is the one this module implements: a detection
belongs to our ball only if its camera ray passes through the court **volume**
-- the footprint plus a legal-play margin, extruded from the ground to a
plausible apex. It is height-agnostic, so unlike projecting a pixel onto
``z = BALL_RADIUS_M`` it does not mistake an ordinary airborne ball for an
off-court one.

What the measurement says: this test does NOT work
--------------------------------------------------
It is a sound *necessary* condition and a useless *discriminator*. Measured on
the four owner clips (reviewed ``metric_15pt`` calibrations, 1269 emitted
detections), the fraction of detections whose ray misses the volume is:

===================  ==========  =============  =============
clip                 emitted     margin 2 m,    margin 0.5 m,
                                 apex 8 m       apex 5 m
===================  ==========  =============  =============
wolverine                   243          0.0 %          1.6 %
outdoor_webcam_20s          306          1.3 %         11.8 %
burlington                  479          0.0 %          0.8 %
indoor                      241          0.0 %          0.0 %
===================  ==========  =============  =============

At the only settings that flag anything, the test also starts flagging the
owner's own clicks on the real ball (2 of 12 on ``outdoor_webcam_20s`` at
margin 0.5 m / apex 5 m). It buys no precision and costs recall.

The reason is geometry, not tuning, and it is worth stating precisely because
it kills the whole family of single-frame ray tests. On ``burlington`` frame 30
the detector is locked on a ball on the court to the *left*. Its ray hits the
ground at ``x = -21.9 m`` -- 19 m outside the sideline -- yet on the way there
it crosses the near-baseline plane at ``x = -2.95 m, z = 0.95 m``, which is
inside our court volume. A camera behind the baseline sees an adjacent court
*through* our own airspace. A ball 20 m off court and a ball 1 m above our own
near-left corner lie on the same ray, and no test that looks at one frame's
ray can tell them apart.

So this module ships **default-OFF and non-promotable**. It exists to record
the refutation in executable form, and because ray containment remains a
correct necessary condition that a future depth-resolving discriminator (one
that turns a ray into a *point*) will want to compose with.

Two tiers, and never a deletion
-------------------------------
``grazing`` (soft)
    The ray enters the volume but its chord is short -- it clips a corner
    rather than crossing the playing space. Consistent with an off-court ball
    seen past our court, and equally consistent with a genuine ball at that
    corner. Mark it; degrade any downstream confidence; keep the detection.

``disjoint`` (hard)
    The ray misses the volume at every height. There is no ball position on
    this ray that is on our court. The detection is still **kept and marked**,
    never removed: a wrong suppression must stay visible and recoverable, and
    on a clip whose calibration is merely adequate the margin, not the ball,
    is what failed.

Nothing here mutates a track. Callers receive marks and decide.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = 1
POLICY = "ball_ray_court_volume_v1"

#: Pickleball court, in the repo's ``court_netcenter_z_up_m`` frame.
COURT_HALF_WIDTH_M = 3.048
COURT_HALF_LENGTH_M = 6.7056

REFUTED_NOTE = (
    "Ray/court-volume containment is a necessary condition, not a "
    "discriminator. Measured on four owner clips it flags 0.0-1.3 percent of "
    "detections at a safe margin while background-ball acquisition runs far "
    "higher, because a camera behind the baseline sees adjacent courts "
    "through its own court's airspace. Do not use it as evidence that a "
    "detection is the ball in play."
)

#: Soft verdicts. Detection is kept, marked, and must never be promoted.
GRAZING_MARKS = ("ray_grazes_court_volume",)

#: Hard verdicts. Detection is kept and marked; suppression is the caller's
#: decision and must remain reversible.
DISJOINT_MARKS = ("ray_misses_court_volume",)


@dataclass(frozen=True)
class CourtVolumeBounds:
    """Extent of the volume a ball in play can occupy.

    ``margin_m`` covers legal out-of-bounds play: a ball may land outside the
    lines and a player may reach well past the sideline to hit it. ``apex_m``
    is the height ceiling; it is deliberately generous because a lob that
    clears the ceiling of this box is a solver defect, not a wrong ball, and
    ``ball_position_plausibility`` already owns that failure.

    ``min_chord_m`` is the chord below which containment is called
    ``grazing``. A ray that crosses less than this much of the volume has
    clipped a corner.
    """

    margin_m: float = 2.0
    apex_m: float = 8.0
    floor_m: float = 0.0
    min_chord_m: float = 1.0
    half_width_m: float = COURT_HALF_WIDTH_M
    half_length_m: float = COURT_HALF_LENGTH_M

    def box_bounds_m(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """Return ``(lo_xyz, hi_xyz)`` of the axis-aligned court volume."""

        margin = float(self.margin_m)
        lo = (
            -float(self.half_width_m) - margin,
            -float(self.half_length_m) - margin,
            float(self.floor_m),
        )
        hi = (
            float(self.half_width_m) + margin,
            float(self.half_length_m) + margin,
            float(self.apex_m),
        )
        return lo, hi

    def to_dict(self) -> dict[str, Any]:
        return {
            "margin_m": round(float(self.margin_m), 6),
            "apex_m": round(float(self.apex_m), 6),
            "floor_m": round(float(self.floor_m), 6),
            "min_chord_m": round(float(self.min_chord_m), 6),
            "half_width_m": round(float(self.half_width_m), 6),
            "half_length_m": round(float(self.half_length_m), 6),
            "refuted_note": REFUTED_NOTE,
        }


DEFAULT_BOUNDS = CourtVolumeBounds()


def ray_box_chord(
    origin: Sequence[float],
    direction: Sequence[float],
    bounds: CourtVolumeBounds | None = None,
) -> tuple[float, float] | None:
    """Return the ``(t_enter, t_exit)`` parameters of the forward ray inside
    the court volume, or ``None`` when the ray never enters it.

    ``t`` is metres along a unit ``direction`` from ``origin``. Only the
    forward half-ray is considered: a ball behind the camera is not a ball.
    """

    limits = bounds or DEFAULT_BOUNDS
    o = _vec3(origin)
    d = _vec3(direction)
    if o is None or d is None:
        return None
    norm = math.sqrt(sum(component * component for component in d))
    if not math.isfinite(norm) or norm <= 0.0:
        return None
    d = tuple(component / norm for component in d)
    lo, hi = limits.box_bounds_m()

    t_enter = 0.0
    t_exit = math.inf
    for axis in range(3):
        if abs(d[axis]) < 1e-12:
            if o[axis] < lo[axis] or o[axis] > hi[axis]:
                return None
            continue
        near = (lo[axis] - o[axis]) / d[axis]
        far = (hi[axis] - o[axis]) / d[axis]
        if near > far:
            near, far = far, near
        t_enter = max(t_enter, near)
        t_exit = min(t_exit, far)
        if t_enter > t_exit:
            return None
    if not math.isfinite(t_exit):
        return None
    return (t_enter, t_exit)


def evaluate_ray(
    origin: Sequence[float],
    direction: Sequence[float],
    bounds: CourtVolumeBounds | None = None,
) -> dict[str, Any]:
    """Classify one camera ray against the court volume.

    A ray that cannot be evaluated -- non-finite origin or a zero direction --
    is reported as ``evaluated: False`` and carries no marks. This gate never
    invents a verdict it did not measure.
    """

    limits = bounds or DEFAULT_BOUNDS
    span = ray_box_chord(origin, direction, limits)
    if _vec3(origin) is None or _vec3(direction) is None:
        return {
            "evaluated": False,
            "contained": False,
            "marks": [],
            "verdict": "not_evaluated",
            "chord_length_m": 0.0,
        }
    if span is None:
        return {
            "evaluated": True,
            "contained": False,
            "marks": list(DISJOINT_MARKS),
            "verdict": "disjoint",
            "chord_length_m": 0.0,
        }
    chord = max(span[1] - span[0], 0.0)
    grazing = chord < float(limits.min_chord_m)
    return {
        "evaluated": True,
        "contained": True,
        "marks": list(GRAZING_MARKS) if grazing else [],
        "verdict": "grazing" if grazing else "contained",
        "chord_length_m": round(chord, 6),
        "entry_distance_m": round(span[0], 6),
        "exit_distance_m": round(span[1], 6),
    }


def evaluate_ball_track_court_volume(
    frames: Sequence[Mapping[str, Any]],
    rays_by_frame: Mapping[int, tuple[Sequence[float], Sequence[float]]],
    *,
    bounds: CourtVolumeBounds | None = None,
) -> dict[str, Any]:
    """Mark every emitted detection of a 2D ball track against the volume.

    ``rays_by_frame`` maps a frame index to ``(origin_m, direction)`` in the
    court world frame; the caller owns un-distortion and pose, so this module
    stays free of any camera model and can be tested on synthetic geometry.

    Frames the detector did not emit are skipped rather than marked: an absent
    detection is already honest, and this gate never turns silence into a
    verdict. The returned report is descriptive only -- ``frames`` is not
    mutated and no detection is removed.
    """

    limits = bounds or DEFAULT_BOUNDS
    per_frame: list[dict[str, Any]] = []
    counts = {"contained": 0, "grazing": 0, "disjoint": 0, "not_evaluated": 0}
    for index, frame in enumerate(frames):
        if not bool(frame.get("visible")):
            continue
        ray = rays_by_frame.get(index)
        if ray is None:
            counts["not_evaluated"] += 1
            per_frame.append({"frame": index, "verdict": "not_evaluated", "marks": []})
            continue
        report = evaluate_ray(ray[0], ray[1], limits)
        counts[report["verdict"]] = counts.get(report["verdict"], 0) + 1
        per_frame.append({"frame": index, **report})

    evaluated = counts["contained"] + counts["grazing"] + counts["disjoint"]
    return {
        "schema_version": SCHEMA_VERSION,
        "policy": POLICY,
        "not_ground_truth": True,
        "candidate_prediction": True,
        "bounds": limits.to_dict(),
        "emitted_frame_count": len(per_frame),
        "evaluated_frame_count": evaluated,
        "counts": dict(counts),
        "disjoint_rate": round(counts["disjoint"] / evaluated, 6) if evaluated else 0.0,
        "grazing_rate": round(counts["grazing"] / evaluated, 6) if evaluated else 0.0,
        "frames": per_frame,
        "refuted_note": REFUTED_NOTE,
    }


def _vec3(value: Sequence[float] | None) -> tuple[float, float, float] | None:
    if value is None:
        return None
    try:
        components = [float(component) for component in value]
    except (TypeError, ValueError):
        return None
    if len(components) != 3:
        return None
    if not all(math.isfinite(component) for component in components):
        return None
    return (components[0], components[1], components[2])

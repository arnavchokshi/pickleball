"""Where a camera model was actually calibrated, and what lies outside it.

Why this module exists
----------------------
A court calibration is fit to a handful of reviewed correspondences and is then
used to back-project *any* pixel in the frame. Nothing records the image region
those correspondences covered, so a pixel at the frame edge is consumed with
exactly the same confidence as one between the baselines -- even when no
correspondence within 30% of the half-diagonal ever constrained the model
there.

Measured on ``outdoor_webcam_20s_fullmesh_final``
(``runs/lanes/farfield_extrapolation_20260727/``): the 15 reviewed
correspondences span **2.0% to 50.0%** of the half-diagonal, median 32.6%, and
the two extremes ARE the near-baseline corners -- the court itself supplies no
correspondence beyond 50%. Four owner-labelled bounces sit at 42.8%, 56.2%,
68.2% and 79.6%. Three of them are back-projected by a model that never saw a
correspondence at their radius, and the leave-one-out validation that blessed
that model held out only central points, so it had no leverage where the
extrapolation happens.

What this gate does and does not claim
--------------------------------------
This is an **honesty** gate, not an accuracy fix. A position outside the
envelope is not known to be wrong; it is known to be **unvalidated**. Nothing
here moves a position, and nothing here suppresses one -- see the two tiers
below. Passing the gate is equally weak: a pixel inside the calibrated radius
inherits the calibration's own residual floor and is still depth-blind, exactly
as :mod:`ball_position_plausibility` describes.

Two severities
--------------
``extrapolated`` (soft)
    The pixel's radius exceeds every correspondence the model was fit to. Keep
    the position, widen its uncertainty, never promote it -- but do not drop
    it: the click may be perfectly correct and the owner needs to see the
    label, flagged, rather than have it vanish.

``far_extrapolated`` (harder, still never suppressed)
    Far enough out that the *unvalidated* radial term alone exceeds the entire
    residual the calibration actually measured (see
    :data:`FAR_EXTRAPOLATION_RATIO`). This position must not be used as
    evidence about the camera, the court, or the ball's accuracy. It is still
    emitted, because deleting a real observation is a worse lie than labelling
    it.

The deliberate divergence from :mod:`ball_position_plausibility`: there,
``absurd`` suppresses the position, because a ball twenty metres underground is
not a measurement of anything. Here the hard tier does NOT suppress, because an
extrapolated position may be entirely correct -- we simply cannot vouch for it.
The vocabulary and the two-tier shape are shared on purpose; the consequence is
different and is stated rather than assumed.

How large is "unvalidated"?
---------------------------
:func:`radial_extrapolation_pixel_allowance` puts a number on it from first
principles, using only quantities the calibration already records.

The first term any rectilinear fit leaves unmodelled is radial and cubic. A
cubic term of amplitude ``a`` displaces a pixel at radius ``r`` by ``a * r^3``.
The fit could not have seen such a term if its displacement at the *calibrated*
radius stayed inside the fit's own residual noise ``eps_px``:

    a * r_cal^3 <= eps_px      =>      a <= eps_px / r_cal^3

so at a query radius ``r`` the term the calibration cannot exclude is bounded by

    |displacement(r)| <= eps_px * (r / r_cal)^3

and the part of that which the calibration residual does not already account
for is

    allowance_px = eps_px * max(0, (r / r_cal)^3 - 1)

It is exactly zero inside the calibrated radius, so every existing central
consumer is unchanged, and it grows as the cube of how far past the evidence a
caller has gone. It is a *bound on what the fit could not have detected*, not a
prediction of an error, and it is deliberately built on the calibration's own
measured residual rather than a tuned constant.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = 1
POLICY = "calibration_extrapolation_v1"

#: Radius ratio at which the unvalidated cubic allowance first equals the whole
#: measured calibration residual: ``(r / r_cal)^3 - 1 = 1``, i.e. ``2 ** (1/3)``.
#: Chosen from the model above rather than picked, so the hard tier means a
#: stateable thing: "past here, what we cannot rule out is larger than
#: everything we measured".
FAR_EXTRAPOLATION_RATIO = 2.0 ** (1.0 / 3.0)

#: Exponent of the leading unmodelled radial term. Cubic is the first term of
#: every radial distortion polynomial and also the leading error of a focal
#: length that traded off against one.
RADIAL_TERM_EXPONENT = 3.0

#: The ratio is capped before it enters the cube, so a wildly out-of-frame
#: pixel cannot produce an unbounded sigma. At the cap the allowance is 26x the
#: calibration residual, which is already far past any usable position.
MAX_EXTRAPOLATION_RATIO = 3.0

#: Used only when a calibration records no reprojection error at all. A fit
#: that cannot state its own residual gets a deliberately non-flattering one.
DEFAULT_RESIDUAL_PX = 4.0

EXTRAPOLATION_UNVALIDATED_NOTE = (
    "This pixel lies outside the image region the calibration's own "
    "correspondences cover. The camera model is being extrapolated there and "
    "no held-out evidence in this repo tests it at that radius. The position "
    "is kept and is not claimed to be wrong -- it is claimed to be "
    "unvalidated."
)

#: Soft violations. Position is kept, sigma widened, never promoted.
EXTRAPOLATED_VIOLATIONS = ("outside_calibrated_radius",)

#: Harder violations. Position is still kept and still emitted; it must not be
#: used as evidence about accuracy.
FAR_EXTRAPOLATED_VIOLATIONS = ("far_outside_calibrated_radius",)

VERDICT_WITHIN = "within_calibrated_envelope"
VERDICT_EXTRAPOLATED = "extrapolated"
VERDICT_FAR_EXTRAPOLATED = "far_extrapolated"


@dataclass(frozen=True)
class CalibratedImageEnvelope:
    """The image region a calibration's own correspondences actually cover.

    Radial about the frame centre, because that is the axis along which a
    rectilinear camera model degrades: both an unmodelled distortion term and a
    focal length fit against one are radial. The correspondence bounding box is
    carried alongside as *descriptive* provenance -- it says which part of the
    frame was sampled at all -- but it does not decide the verdict, because a
    court seen from behind a baseline is never azimuthally uniform and a box
    test would fire on almost every pixel.
    """

    image_size_px: tuple[float, float]
    correspondence_count: int
    radius_px_min: float
    radius_px_median: float
    radius_px_p95: float
    radius_px_max: float
    bbox_px: tuple[float, float, float, float]
    provenance: str = "calibration_image_pts"
    far_ratio: float = FAR_EXTRAPOLATION_RATIO

    def __post_init__(self) -> None:
        width, height = self.image_size_px
        if not (width > 0.0 and height > 0.0):
            raise ValueError("image_size_px must be positive")
        if self.correspondence_count < 1:
            raise ValueError("an envelope needs at least one correspondence")
        if self.radius_px_max <= 0.0:
            raise ValueError("calibrated radius must be positive")
        if self.far_ratio < 1.0:
            raise ValueError("far_ratio must be >= 1")

    @property
    def center_px(self) -> tuple[float, float]:
        return (float(self.image_size_px[0]) / 2.0, float(self.image_size_px[1]) / 2.0)

    @property
    def half_diagonal_px(self) -> float:
        return math.hypot(float(self.image_size_px[0]) / 2.0, float(self.image_size_px[1]) / 2.0)

    @property
    def calibrated_radius_px(self) -> float:
        """The radius beyond which the model is extrapolating.

        The MAXIMUM correspondence radius, not a percentile. That is the
        generous reading -- it credits the model with everything it saw -- and
        this gate should err towards silence, not towards crying wolf.
        """

        return float(self.radius_px_max)

    @property
    def far_radius_px(self) -> float:
        return self.calibrated_radius_px * float(self.far_ratio)

    def radius_pct(self, radius_px: float) -> float:
        """Radius as a percentage of the frame half-diagonal."""

        return 100.0 * float(radius_px) / self.half_diagonal_px

    def pixel_radius_px(self, pixel_xy: Sequence[float]) -> float:
        cx, cy = self.center_px
        return math.hypot(float(pixel_xy[0]) - cx, float(pixel_xy[1]) - cy)

    def contains_bbox(self, pixel_xy: Sequence[float]) -> bool:
        x_min, x_max, y_min, y_max = self.bbox_px
        x, y = float(pixel_xy[0]), float(pixel_xy[1])
        return x_min <= x <= x_max and y_min <= y <= y_max

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": int(SCHEMA_VERSION),
            "policy": POLICY,
            "provenance": str(self.provenance),
            "image_size_px": [round(float(v), 3) for v in self.image_size_px],
            "center_px": [round(v, 3) for v in self.center_px],
            "half_diagonal_px": round(self.half_diagonal_px, 3),
            "correspondence_count": int(self.correspondence_count),
            "calibrated_radius_px": round(self.calibrated_radius_px, 3),
            "calibrated_radius_pct_of_half_diagonal": round(
                self.radius_pct(self.calibrated_radius_px), 3
            ),
            "far_radius_px": round(self.far_radius_px, 3),
            "far_radius_pct_of_half_diagonal": round(self.radius_pct(self.far_radius_px), 3),
            "far_ratio": round(float(self.far_ratio), 6),
            "correspondence_radius_px": {
                "min": round(float(self.radius_px_min), 3),
                "median": round(float(self.radius_px_median), 3),
                "p95": round(float(self.radius_px_p95), 3),
                "max": round(float(self.radius_px_max), 3),
            },
            "correspondence_radius_pct_of_half_diagonal": {
                "min": round(self.radius_pct(self.radius_px_min), 3),
                "median": round(self.radius_pct(self.radius_px_median), 3),
                "p95": round(self.radius_pct(self.radius_px_p95), 3),
                "max": round(self.radius_pct(self.radius_px_max), 3),
            },
            "correspondence_bbox_px": {
                "x_min": round(float(self.bbox_px[0]), 3),
                "x_max": round(float(self.bbox_px[1]), 3),
                "y_min": round(float(self.bbox_px[2]), 3),
                "y_max": round(float(self.bbox_px[3]), 3),
            },
            "extrapolation_unvalidated_note": EXTRAPOLATION_UNVALIDATED_NOTE,
        }


def calibrated_image_envelope(
    calibration: Mapping[str, Any],
) -> CalibratedImageEnvelope | None:
    """Build the envelope from a calibration's own reviewed correspondences.

    Returns ``None`` when the artifact carries no ``image_pts`` or no usable
    image size. An absent envelope is reported as absent; it is never faked
    from the frame size, because "we do not know what this model was fit to" is
    a different statement from "it was fit everywhere".
    """

    image_pts = calibration.get("image_pts")
    if not isinstance(image_pts, Sequence) or isinstance(image_pts, (str, bytes)):
        return None
    size = _image_size(calibration)
    if size is None:
        return None
    width, height = size
    cx, cy = width / 2.0, height / 2.0

    radii: list[float] = []
    xs: list[float] = []
    ys: list[float] = []
    for point in image_pts:
        xy = _xy(point)
        if xy is None:
            continue
        xs.append(xy[0])
        ys.append(xy[1])
        radii.append(math.hypot(xy[0] - cx, xy[1] - cy))
    if not radii:
        return None
    radii.sort()
    return CalibratedImageEnvelope(
        image_size_px=(width, height),
        correspondence_count=len(radii),
        radius_px_min=radii[0],
        radius_px_median=_percentile(radii, 50.0),
        radius_px_p95=_percentile(radii, 95.0),
        radius_px_max=radii[-1],
        bbox_px=(min(xs), max(xs), min(ys), max(ys)),
    )


def evaluate_pixel(
    pixel_xy: Sequence[float] | None,
    envelope: CalibratedImageEnvelope | None,
) -> dict[str, Any]:
    """Classify one image pixel against the calibrated envelope.

    A ``None`` / non-finite pixel is not a violation: an absent observation is
    already honest, and this gate never invents one. A ``None`` envelope
    yields ``evaluated: False`` with a reason, so a caller can tell "inside the
    envelope" apart from "there is no envelope".
    """

    xy = _xy(pixel_xy)
    if xy is None:
        return {
            "evaluated": False,
            "reason": "no finite pixel",
            "verdict": VERDICT_WITHIN,
            "violations": [],
            "far_violations": [],
            "extrapolated": False,
            "far_extrapolated": False,
        }
    if envelope is None:
        return {
            "evaluated": False,
            "reason": "calibration records no correspondences to bound",
            "verdict": VERDICT_WITHIN,
            "violations": [],
            "far_violations": [],
            "extrapolated": False,
            "far_extrapolated": False,
        }

    radius = envelope.pixel_radius_px(xy)
    calibrated = envelope.calibrated_radius_px
    ratio = radius / calibrated
    violations: list[str] = []
    far: list[str] = []
    if radius > calibrated:
        violations.append("outside_calibrated_radius")
        if radius > envelope.far_radius_px:
            far.append("far_outside_calibrated_radius")
    if far:
        verdict = VERDICT_FAR_EXTRAPOLATED
    elif violations:
        verdict = VERDICT_EXTRAPOLATED
    else:
        verdict = VERDICT_WITHIN
    return {
        "evaluated": True,
        "verdict": verdict,
        "violations": sorted(violations),
        "far_violations": sorted(far),
        "extrapolated": bool(violations),
        "far_extrapolated": bool(far),
        "pixel_xy": [round(xy[0], 3), round(xy[1], 3)],
        "radius_px": round(radius, 3),
        "radius_pct_of_half_diagonal": round(envelope.radius_pct(radius), 3),
        "calibrated_radius_px": round(calibrated, 3),
        "calibrated_radius_pct_of_half_diagonal": round(envelope.radius_pct(calibrated), 3),
        "extrapolation_ratio": round(ratio, 6),
        "overage_px": round(max(0.0, radius - calibrated), 3),
        "inside_correspondence_bbox": bool(envelope.contains_bbox(xy)),
    }


def pixel_verdict(
    pixel_xy: Sequence[float] | None,
    envelope: CalibratedImageEnvelope | None,
) -> str:
    """Just the verdict string, for callers that only branch on it."""

    return str(evaluate_pixel(pixel_xy, envelope)["verdict"])


def is_extrapolated(
    pixel_xy: Sequence[float] | None,
    envelope: CalibratedImageEnvelope | None,
) -> bool:
    """True when the pixel is past every correspondence the model was fit to."""

    return bool(evaluate_pixel(pixel_xy, envelope)["extrapolated"])


def calibration_residual_px(calibration: Mapping[str, Any]) -> tuple[float, str]:
    """The fit's own residual scale, and where it came from.

    The MEDIAN reprojection error, not the p95: the p95 of 15 points is one
    point, and this quantity is used as "the size of an error the fit would
    not have noticed", which is a typical-case question.

    Caveat recorded rather than hidden: ``reprojection_error_px`` is an
    IN-SAMPLE residual (``runs/lanes/calib_distortion_fit_20260726/REPORT.md``
    section 3.1 makes the same point about the plane-residual floor). An
    in-sample residual understates what a fit failed to detect, so the
    allowance built on it is a LOWER bound on the unvalidated term.
    """

    reprojection = calibration.get("reprojection_error_px")
    if isinstance(reprojection, Mapping):
        median = _finite(reprojection.get("median"))
        if median is not None and median > 0.0:
            return float(median), "calibration_reprojection_median_in_sample"
    return float(DEFAULT_RESIDUAL_PX), "default_no_reprojection_error_recorded"


def radial_extrapolation_pixel_allowance(
    calibration: Mapping[str, Any],
    pixel_xy: Sequence[float] | None,
    *,
    envelope: CalibratedImageEnvelope | None = None,
    residual_px: float | None = None,
) -> dict[str, Any]:
    """Pixels of unvalidated radial error at ``pixel_xy``, and the reasoning.

    ``allowance_px = eps_px * max(0, (r / r_cal)^3 - 1)``; see the module
    docstring for the derivation. Exactly zero inside the calibrated radius,
    so this is a no-op for every consumer that stays where the evidence is.
    """

    resolved = envelope if envelope is not None else calibrated_image_envelope(calibration)
    verdict = evaluate_pixel(pixel_xy, resolved)
    if not verdict["evaluated"]:
        return {
            "available": False,
            "reason": str(verdict.get("reason", "not evaluated")),
            "allowance_px": 0.0,
            "verdict": VERDICT_WITHIN,
            "violations": [],
            "far_violations": [],
        }

    if residual_px is None:
        eps_px, provenance = calibration_residual_px(calibration)
    else:
        supplied = _finite(residual_px)
        eps_px = float(max(0.0, supplied)) if supplied is not None else float(DEFAULT_RESIDUAL_PX)
        provenance = "supplied" if supplied is not None else "default_supplied_not_finite"

    ratio = float(verdict["extrapolation_ratio"])
    capped_ratio = min(ratio, MAX_EXTRAPOLATION_RATIO)
    growth = max(0.0, capped_ratio**RADIAL_TERM_EXPONENT - 1.0)
    allowance_px = eps_px * growth
    return {
        "available": True,
        "policy": POLICY,
        "schema_version": int(SCHEMA_VERSION),
        "verdict": verdict["verdict"],
        "violations": verdict["violations"],
        "far_violations": verdict["far_violations"],
        "extrapolated": verdict["extrapolated"],
        "far_extrapolated": verdict["far_extrapolated"],
        "radius_px": verdict["radius_px"],
        "radius_pct_of_half_diagonal": verdict["radius_pct_of_half_diagonal"],
        "calibrated_radius_px": verdict["calibrated_radius_px"],
        "calibrated_radius_pct_of_half_diagonal": verdict[
            "calibrated_radius_pct_of_half_diagonal"
        ],
        "extrapolation_ratio": round(ratio, 6),
        "ratio_used": round(capped_ratio, 6),
        "ratio_capped": bool(ratio > MAX_EXTRAPOLATION_RATIO),
        "residual_px": round(eps_px, 6),
        "residual_provenance": provenance,
        "allowance_px": round(allowance_px, 6),
        "inside_correspondence_bbox": verdict["inside_correspondence_bbox"],
        "basis": (
            f"pixel radius {verdict['radius_px']:.1f} px = "
            f"{verdict['radius_pct_of_half_diagonal']:.1f}% of the half-diagonal; the "
            f"calibration's correspondences reach "
            f"{verdict['calibrated_radius_pct_of_half_diagonal']:.1f}%. Unvalidated cubic "
            f"radial term bounded by {eps_px:.3f} px x ((r/r_cal)^3 - 1) = "
            f"{allowance_px:.3f} px."
        ),
    }


def envelope_block_for_calibration(calibration: Mapping[str, Any]) -> dict[str, Any]:
    """The additive block a calibration artifact should carry.

    Always returns a block: when the envelope cannot be computed the block says
    so, so a downstream reader can distinguish "no correspondences recorded"
    from "this artifact predates the envelope".
    """

    envelope = calibrated_image_envelope(calibration)
    if envelope is None:
        return {
            "schema_version": int(SCHEMA_VERSION),
            "policy": POLICY,
            "available": False,
            "reason": "calibration records no usable image_pts / image_size",
            "extrapolation_unvalidated_note": EXTRAPOLATION_UNVALIDATED_NOTE,
        }
    residual_px, residual_provenance = calibration_residual_px(calibration)
    block = {"available": True, **envelope.to_json()}
    block["residual_px"] = round(residual_px, 6)
    block["residual_provenance"] = residual_provenance
    return block


def with_calibration_envelope(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return ``payload`` plus a ``calibrated_image_envelope`` block.

    Additive and idempotent. The canonical ``CourtCalibration`` schema is
    deliberately untouched, matching the precedent set by
    ``court_calibration.build_manual_tap_calibration_artifact``.
    """

    out = dict(payload)
    out["calibrated_image_envelope"] = envelope_block_for_calibration(payload)
    return out


def evaluate_ball_track_extrapolation(
    frames: Sequence[Mapping[str, Any]],
    calibration: Mapping[str, Any],
    *,
    pixel_key: str = "xy",
    position_key: str = "world_xyz",
    envelope: CalibratedImageEnvelope | None = None,
) -> dict[str, Any]:
    """Sweep a solved ball track and count emitted positions outside the envelope.

    Only frames that actually carry a 3D position are counted: a frame with a
    detection but no ``world_xyz`` emitted nothing, so there is nothing to
    mistrust. This mirrors ``ball_position_plausibility`` sweeping every frame
    with a ``world_xyz`` rather than only the ones some segment covered.
    """

    resolved = envelope if envelope is not None else calibrated_image_envelope(calibration)
    reports: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    evaluated = 0
    positions = 0
    extrapolated = 0
    far = 0
    for index, raw in enumerate(frames):
        frame = raw if isinstance(raw, Mapping) else {}
        if _xyz(frame.get(position_key)) is None:
            continue
        positions += 1
        verdict = evaluate_pixel(frame.get(pixel_key), resolved)
        if not verdict["evaluated"]:
            continue
        evaluated += 1
        if verdict["extrapolated"]:
            extrapolated += 1
        if verdict["far_extrapolated"]:
            far += 1
        for name in [*verdict["violations"], *verdict["far_violations"]]:
            counts[name] = counts.get(name, 0) + 1
        if verdict["extrapolated"]:
            reports.append(
                {
                    "frame": index,
                    "verdict": verdict["verdict"],
                    "radius_pct_of_half_diagonal": verdict["radius_pct_of_half_diagonal"],
                    "extrapolation_ratio": verdict["extrapolation_ratio"],
                    "violations": verdict["violations"],
                    "far_violations": verdict["far_violations"],
                }
            )

    return {
        "schema_version": int(SCHEMA_VERSION),
        "policy": POLICY,
        "envelope": resolved.to_json() if resolved is not None else None,
        "extrapolation_unvalidated_reason": EXTRAPOLATION_UNVALIDATED_NOTE,
        "summary": {
            "emitted_position_count": positions,
            "evaluated_frame_count": evaluated,
            "extrapolated_frame_count": extrapolated,
            "far_extrapolated_frame_count": far,
            "violation_counts": dict(sorted(counts.items())),
        },
        "frames": reports,
    }


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _image_size(calibration: Mapping[str, Any]) -> tuple[float, float] | None:
    size = calibration.get("image_size")
    if isinstance(size, Sequence) and not isinstance(size, (str, bytes)) and len(size) == 2:
        width = _finite(size[0])
        height = _finite(size[1])
        if width is not None and height is not None and width > 0.0 and height > 0.0:
            return float(width), float(height)
    intrinsics = calibration.get("intrinsics")
    if isinstance(intrinsics, Mapping):
        cx = _finite(intrinsics.get("cx"))
        cy = _finite(intrinsics.get("cy"))
        if cx is not None and cy is not None and cx > 0.0 and cy > 0.0:
            return cx * 2.0, cy * 2.0
    return None


def _xy(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        return None
    try:
        xy = (float(value[0]), float(value[1]))
    except (TypeError, ValueError):
        return None
    return xy if all(math.isfinite(component) for component in xy) else None


def _xyz(value: Any) -> tuple[float, float, float] | None:
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


def _percentile(ordered: Sequence[float], percentile: float) -> float:
    values = sorted(float(value) for value in ordered)
    if not values:
        raise ValueError("percentile requires at least one value")
    if len(values) == 1:
        return values[0]
    rank = (len(values) - 1) * float(percentile) / 100.0
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return values[lower]
    weight = rank - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


__all__ = [
    "DEFAULT_RESIDUAL_PX",
    "EXTRAPOLATED_VIOLATIONS",
    "EXTRAPOLATION_UNVALIDATED_NOTE",
    "FAR_EXTRAPOLATED_VIOLATIONS",
    "FAR_EXTRAPOLATION_RATIO",
    "MAX_EXTRAPOLATION_RATIO",
    "POLICY",
    "RADIAL_TERM_EXPONENT",
    "SCHEMA_VERSION",
    "VERDICT_EXTRAPOLATED",
    "VERDICT_FAR_EXTRAPOLATED",
    "VERDICT_WITHIN",
    "CalibratedImageEnvelope",
    "calibrated_image_envelope",
    "calibration_residual_px",
    "envelope_block_for_calibration",
    "evaluate_ball_track_extrapolation",
    "evaluate_pixel",
    "is_extrapolated",
    "pixel_verdict",
    "radial_extrapolation_pixel_allowance",
    "with_calibration_envelope",
]

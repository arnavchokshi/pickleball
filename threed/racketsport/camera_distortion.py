"""Brown-Conrady lens distortion, applied at the pinhole seams that ignored it.

`CameraIntrinsics.dist` has always been part of the calibration schema, but the
production monocular ball path (`ball_arc_solver.pixel_ray_world` and
`ball_arc_solver._project_world_point`) modelled the camera as a pure pinhole:
it read `fx/fy/cx/cy` and never looked at `dist`. That made a fitted `k1`
*inert* -- a calibration could estimate radial distortion, store it, and change
nothing downstream, while its focal length had been fit under a distortion model
the consumer did not apply. This module is the missing piece, kept separate so
the two call sites stay one line each.

Model (OpenCV `cv2.projectPoints` convention, normalized camera coordinates):

    r^2 = x^2 + y^2
    radial = 1 + k1 r^2 + k2 r^4
    x' = x * radial + 2 p1 x y + p2 (r^2 + 2 x^2)
    y' = y * radial + p1 (r^2 + 2 y^2) + 2 p2 x y

`dist` is stored as `[k1, k2, p1, p2]`. The metric-15pt fit constrains
`p1 = p2 = 0` (a single planar-ish view cannot identify tangential terms), so
the pure-radial inverse below is exact for every calibration this repo emits;
the general iterative branch exists for imported calibrations that do carry
tangential terms.

Every function is an exact no-op when `dist` is absent, all-zero, or shorter
than two entries, so zero-distortion calibrations keep their previous numeric
behaviour bit-for-bit.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

__all__ = [
    "distort_normalized",
    "distort_pixel",
    "distortion_coefficients",
    "has_distortion",
    "is_radially_invertible",
    "max_normalized_radius",
    "undistort_normalized",
    "undistort_pixel",
]

# Fixed-point iteration budget for the general (tangential) inverse, and the
# normalized-coordinate tolerance both inverses converge to. 1e-12 in normalized
# units is far below a micro-pixel at any realistic focal length.
_MAX_ITERATIONS = 200
_TOLERANCE = 1e-12


def distortion_coefficients(intrinsics: Mapping[str, Any] | None) -> tuple[float, float, float, float]:
    """Return `(k1, k2, p1, p2)`; all zero when nothing usable is declared."""

    if not isinstance(intrinsics, Mapping):
        return (0.0, 0.0, 0.0, 0.0)
    raw = intrinsics.get("dist")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return (0.0, 0.0, 0.0, 0.0)
    values: list[float] = []
    for index in range(4):
        if index >= len(raw):
            values.append(0.0)
            continue
        try:
            value = float(raw[index])
        except (TypeError, ValueError):
            return (0.0, 0.0, 0.0, 0.0)
        values.append(value if math.isfinite(value) else 0.0)
    return (values[0], values[1], values[2], values[3])


def has_distortion(intrinsics: Mapping[str, Any] | None) -> bool:
    """True when the declared coefficients would move any pixel at all."""

    return any(value != 0.0 for value in distortion_coefficients(intrinsics))


def distort_normalized(
    x: float, y: float, coefficients: tuple[float, float, float, float]
) -> tuple[float, float]:
    """Ideal normalized point -> observed (distorted) normalized point."""

    k1, k2, p1, p2 = coefficients
    if k1 == 0.0 and k2 == 0.0 and p1 == 0.0 and p2 == 0.0:
        return (x, y)
    r2 = x * x + y * y
    radial = 1.0 + k1 * r2 + k2 * r2 * r2
    x_out = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    y_out = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
    return (x_out, y_out)


def undistort_normalized(
    x: float, y: float, coefficients: tuple[float, float, float, float]
) -> tuple[float, float]:
    """Observed (distorted) normalized point -> ideal normalized point.

    Pure-radial coefficients (`p1 = p2 = 0`, the case every calibration this
    repo fits) get an exact scalar solve: the distortion only rescales the
    radius, so inverting it is a 1-D root find on
    ``f(r) = r (1 + k1 r^2 + k2 r^4) - r_observed``, done by Newton with a
    bisection fallback on a bracketed interval. That is robust at the strong
    barrel coefficients (k1 ~ -0.3) these cameras actually have, where the naive
    fixed-point iteration OpenCV uses can crawl or diverge near the frame corner.
    """

    k1, k2, p1, p2 = coefficients
    if k1 == 0.0 and k2 == 0.0 and p1 == 0.0 and p2 == 0.0:
        return (x, y)

    if p1 == 0.0 and p2 == 0.0:
        r_observed = math.hypot(x, y)
        if r_observed <= 0.0:
            return (0.0, 0.0)
        r_ideal = _invert_radial(r_observed, k1, k2)
        scale = r_ideal / r_observed
        return (x * scale, y * scale)

    # General case: fixed-point iteration on the full model.
    x_guess, y_guess = x, y
    for _ in range(_MAX_ITERATIONS):
        x_test, y_test = distort_normalized(x_guess, y_guess, coefficients)
        dx, dy = x - x_test, y - y_test
        x_guess += dx
        y_guess += dy
        if abs(dx) < _TOLERANCE and abs(dy) < _TOLERANCE:
            break
    return (x_guess, y_guess)


def _invert_radial(r_observed: float, k1: float, k2: float) -> float:
    """Solve ``r (1 + k1 r^2 + k2 r^4) = r_observed`` for ``r >= 0``."""

    def f(r: float) -> float:
        r2 = r * r
        return r * (1.0 + k1 * r2 + k2 * r2 * r2) - r_observed

    # Newton from the undistorted guess, which is already close for |k1| < 1.
    r = r_observed
    for _ in range(_MAX_ITERATIONS):
        r2 = r * r
        derivative = 1.0 + 3.0 * k1 * r2 + 5.0 * k2 * r2 * r2
        if abs(derivative) < 1e-12:
            break
        step = f(r) / derivative
        r -= step
        if r < 0.0:
            r = 0.0
            break
        if abs(step) < _TOLERANCE:
            return r

    # Newton did not settle (non-monotone model at this radius): bracket and
    # bisect over the region where the model is still invertible.
    lo = 0.0
    hi = max(r_observed, 1e-6)
    for _ in range(_MAX_ITERATIONS):
        if f(hi) >= 0.0:
            break
        hi *= 2.0
    else:
        # Monotonically below target everywhere we looked: the model cannot
        # explain this radius. Returning the observed radius degrades to the
        # undistorted pinhole rather than producing a wild extrapolation.
        return r_observed
    for _ in range(_MAX_ITERATIONS * 2):
        mid = 0.5 * (lo + hi)
        if f(mid) < 0.0:
            lo = mid
        else:
            hi = mid
        if hi - lo < _TOLERANCE:
            break
    return 0.5 * (lo + hi)


def undistort_pixel(
    intrinsics: Mapping[str, Any], uv: Sequence[float]
) -> tuple[float, float]:
    """Observed pixel -> the pixel an ideal pinhole with the same K would see."""

    coefficients = distortion_coefficients(intrinsics)
    u, v = float(uv[0]), float(uv[1])
    if not any(coefficients):
        return (u, v)
    fx, fy, cx, cy = _pinhole(intrinsics)
    x, y = undistort_normalized((u - cx) / fx, (v - cy) / fy, coefficients)
    return (x * fx + cx, y * fy + cy)


def distort_pixel(
    intrinsics: Mapping[str, Any], uv: Sequence[float]
) -> tuple[float, float]:
    """Ideal-pinhole pixel -> the pixel this lens actually puts it at."""

    coefficients = distortion_coefficients(intrinsics)
    u, v = float(uv[0]), float(uv[1])
    if not any(coefficients):
        return (u, v)
    fx, fy, cx, cy = _pinhole(intrinsics)
    x, y = distort_normalized((u - cx) / fx, (v - cy) / fy, coefficients)
    return (x * fx + cx, y * fy + cy)


def max_normalized_radius(
    image_size: Sequence[float], fx: float, fy: float, cx: float, cy: float
) -> float:
    """Largest normalized radius any pixel of this frame can reach.

    The distortion model only has to behave over the image it describes, so this
    is the domain every invertibility check below is quantified over.
    """

    width, height = float(image_size[0]), float(image_size[1])
    corners = ((0.0, 0.0), (width, 0.0), (0.0, height), (width, height))
    return max(math.hypot((u - cx) / fx, (v - cy) / fy) for u, v in corners)


def is_radially_invertible(k1: float, k2: float, max_observed_radius: float) -> bool:
    """True when every pixel of this frame can be undistorted unambiguously.

    ``max_observed_radius`` is the largest normalized radius the *image* reaches
    (see :func:`max_normalized_radius`). The model is usable iff the forward map
    ``f(r) = r (1 + k1 r^2 + k2 r^4)`` is strictly increasing from 0 up to some
    ``r`` with ``f(r) >= max_observed_radius`` -- i.e. it reaches every observed
    radius before it folds back on itself.

    A folded radial map is not a camera: past the fold two different scene rays
    land on the same pixel and the inverse a consumer needs does not exist.
    Strong barrel coefficients do this well inside a frame (``k1 = -0.4276``
    peaks at ``f = 0.589`` while a 1920x1080 frame at ``fx = 1253`` needs
    ``0.879``), so the metric fit treats this as a hard feasibility constraint
    rather than trusting a fixed numeric bound on ``k1``.
    """

    if k1 == 0.0 and k2 == 0.0:
        return True
    limit = max(float(max_observed_radius), 1e-9)
    steps = 4096
    r_max = 4.0 * limit
    for index in range(1, steps + 1):
        r = r_max * index / steps
        r2 = r * r
        if 1.0 + 3.0 * k1 * r2 + 5.0 * k2 * r2 * r2 <= 0.0:
            return False  # folded before covering the frame
        if r * (1.0 + k1 * r2 + k2 * r2 * r2) >= limit:
            return True  # covered the frame while still strictly increasing
    return False


def _pinhole(intrinsics: Mapping[str, Any]) -> tuple[float, float, float, float]:
    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    if fx == 0.0 or fy == 0.0:
        raise ValueError("intrinsics fx/fy must be non-zero to apply lens distortion")
    return (fx, fy, float(intrinsics["cx"]), float(intrinsics["cy"]))
